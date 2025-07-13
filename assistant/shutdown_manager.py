#!/usr/bin/env python3
"""
Sovereign Voice Assistant - Shutdown Manager
Handles graceful application termination with resource cleanup and state persistence
"""

import asyncio
import atexit
import logging
import signal
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union
import json
import os


class ShutdownReason(Enum):
    """Reasons for shutdown"""
    SIGNAL_TERM = "SIGTERM"
    SIGNAL_INT = "SIGINT"
    SIGNAL_HUP = "SIGHUP"
    USER_REQUEST = "user_request"
    ERROR = "error"
    SERVICE_STOP = "service_stop"
    SYSTEM_SHUTDOWN = "system_shutdown"
    RESTART = "restart"


class ResourceType(Enum):
    """Types of resources that need cleanup"""
    AUDIO_DEVICE = "audio_device"
    FILE_HANDLE = "file_handle"
    NETWORK_CONNECTION = "network_connection"
    DATABASE_CONNECTION = "database_connection"
    THREAD_POOL = "thread_pool"
    ASYNC_TASK = "async_task"
    SUBPROCESS = "subprocess"
    TEMP_FILE = "temp_file"
    MEMORY_MAPPED_FILE = "memory_mapped_file"
    LOCK = "lock"
    EVENT = "event"
    TIMER = "timer"


@dataclass
class ShutdownHook:
    """Represents a shutdown hook with metadata"""
    name: str
    callback: Callable
    priority: int = 0  # Higher priority runs first
    timeout: float = 30.0  # Timeout in seconds
    resource_type: Optional[ResourceType] = None
    is_async: bool = False
    critical: bool = False  # Critical hooks must complete
    cleanup_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShutdownStats:
    """Statistics about shutdown process"""
    start_time: float
    end_time: Optional[float] = None
    reason: Optional[ShutdownReason] = None
    hooks_executed: int = 0
    hooks_failed: int = 0
    hooks_timeout: int = 0
    total_duration: Optional[float] = None
    critical_failures: List[str] = field(default_factory=list)


class ShutdownManager:
    """
    Comprehensive shutdown manager for graceful application termination
    
    Features:
    - Signal handling (SIGTERM, SIGINT, SIGHUP)
    - Resource cleanup with priority ordering
    - State persistence
    - Timeout management
    - Async and sync hook support
    - Statistics and logging
    - Recovery preparation
    """
    
    def __init__(self, app_name: str = "SovereignAssistant", 
                 state_dir: Optional[Path] = None,
                 default_timeout: float = 60.0):
        """Initialize shutdown manager"""
        self.app_name = app_name
        self.state_dir = state_dir or Path.cwd() / "data" / "state"
        self.default_timeout = default_timeout
        
        # Shutdown state
        self._shutdown_requested = False
        self._shutdown_in_progress = False
        self._shutdown_completed = False
        self._shutdown_reason: Optional[ShutdownReason] = None
        self._shutdown_stats = ShutdownStats(start_time=time.time())
        
        # Hooks management
        self._hooks: List[ShutdownHook] = []
        self._hook_results: Dict[str, bool] = {}
        self._resources: Dict[str, Any] = {}
        self._cleanup_lock = threading.RLock()
        
        # Signal handling
        self._original_handlers: Dict[int, Any] = {}
        self._signal_setup_done = False
        
        # Async support
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._shutdown_event = asyncio.Event() if asyncio.iscoroutinefunction(self.__init__) else None
        
        # Thread pool for cleanup
        self._cleanup_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="shutdown-cleanup")
        
        # Logging
        self.logger = logging.getLogger(f"{app_name}.shutdown")
        
        # State persistence
        self.state_file = self.state_dir / "shutdown_state.json"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Register built-in cleanup
        self._register_builtin_hooks()
        
        # Register atexit handler
        atexit.register(self._emergency_cleanup)
        
        self.logger.info(f"Shutdown manager initialized for {app_name}")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        if self._signal_setup_done:
            return
        
        # Only setup if we're in the main thread
        if threading.current_thread() is not threading.main_thread():
            self.logger.warning("Signal handlers can only be setup from main thread")
            return
        
        # Store original handlers
        self._original_handlers = {
            signal.SIGTERM: signal.signal(signal.SIGTERM, self._signal_handler),
            signal.SIGINT: signal.signal(signal.SIGINT, self._signal_handler),
        }
        
        # Platform-specific signals
        if hasattr(signal, 'SIGHUP'):
            self._original_handlers[signal.SIGHUP] = signal.signal(signal.SIGHUP, self._signal_handler)
        
        self._signal_setup_done = True
        self.logger.info("Signal handlers registered")
    
    def _signal_handler(self, signum: int, frame):
        """Handle shutdown signals"""
        signal_map = {
            signal.SIGTERM: ShutdownReason.SIGNAL_TERM,
            signal.SIGINT: ShutdownReason.SIGNAL_INT,
        }
        
        if hasattr(signal, 'SIGHUP') and signum == signal.SIGHUP:
            signal_map[signal.SIGHUP] = ShutdownReason.SIGNAL_HUP
        
        reason = signal_map.get(signum, ShutdownReason.SIGNAL_TERM)
        
        self.logger.info(f"Received signal {signum} ({reason.value})")
        
        # Start shutdown process
        if self._event_loop and self._event_loop.is_running():
            # Schedule shutdown in event loop
            asyncio.create_task(self.shutdown_async(reason))
        else:
            # Synchronous shutdown
            self.shutdown(reason)
    
    def register_hook(self, name: str, callback: Callable, 
                     priority: int = 0, timeout: float = None,
                     resource_type: Optional[ResourceType] = None,
                     critical: bool = False, **kwargs) -> str:
        """Register a shutdown hook"""
        if self._shutdown_in_progress:
            raise RuntimeError("Cannot register hooks during shutdown")
        
        timeout = timeout or self.default_timeout
        is_async = asyncio.iscoroutinefunction(callback)
        
        hook = ShutdownHook(
            name=name,
            callback=callback,
            priority=priority,
            timeout=timeout,
            resource_type=resource_type,
            is_async=is_async,
            critical=critical,
            cleanup_data=kwargs
        )
        
        with self._cleanup_lock:
            self._hooks.append(hook)
            self._hooks.sort(key=lambda h: h.priority, reverse=True)
        
        self.logger.debug(f"Registered shutdown hook: {name} (priority={priority}, timeout={timeout}s)")
        return name
    
    def unregister_hook(self, name: str) -> bool:
        """Unregister a shutdown hook"""
        with self._cleanup_lock:
            for i, hook in enumerate(self._hooks):
                if hook.name == name:
                    self._hooks.pop(i)
                    self.logger.debug(f"Unregistered shutdown hook: {name}")
                    return True
        return False
    
    def register_resource(self, name: str, resource: Any, 
                         cleanup_func: Optional[Callable] = None,
                         resource_type: Optional[ResourceType] = None,
                         priority: int = 0) -> str:
        """Register a resource for automatic cleanup"""
        self._resources[name] = resource
        
        if cleanup_func:
            hook_name = f"resource_cleanup_{name}"
            self.register_hook(
                name=hook_name,
                callback=cleanup_func,
                priority=priority,
                resource_type=resource_type,
                resource_name=name
            )
            return hook_name
        
        return name
    
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested"""
        return self._shutdown_requested
    
    def is_shutdown_in_progress(self) -> bool:
        """Check if shutdown is in progress"""
        return self._shutdown_in_progress
    
    def shutdown(self, reason: ShutdownReason = ShutdownReason.USER_REQUEST):
        """Perform synchronous shutdown"""
        if self._shutdown_in_progress:
            self.logger.warning("Shutdown already in progress")
            return
        
        self.logger.info(f"Starting shutdown: {reason.value}")
        
        self._shutdown_requested = True
        self._shutdown_in_progress = True
        self._shutdown_reason = reason
        self._shutdown_stats.reason = reason
        self._shutdown_stats.start_time = time.time()
        
        try:
            # Save state before cleanup
            self._save_shutdown_state()
            
            # Execute shutdown hooks
            self._execute_hooks_sync()
            
            # Final cleanup
            self._final_cleanup()
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}", exc_info=True)
        finally:
            self._shutdown_completed = True
            self._shutdown_stats.end_time = time.time()
            self._shutdown_stats.total_duration = (
                self._shutdown_stats.end_time - self._shutdown_stats.start_time
            )
            
            self.logger.info(f"Shutdown completed in {self._shutdown_stats.total_duration:.2f}s")
            self._log_shutdown_stats()
    
    async def shutdown_async(self, reason: ShutdownReason = ShutdownReason.USER_REQUEST):
        """Perform asynchronous shutdown"""
        if self._shutdown_in_progress:
            self.logger.warning("Shutdown already in progress")
            return
        
        self.logger.info(f"Starting async shutdown: {reason.value}")
        
        self._shutdown_requested = True
        self._shutdown_in_progress = True
        self._shutdown_reason = reason
        self._shutdown_stats.reason = reason
        self._shutdown_stats.start_time = time.time()
        
        try:
            # Save state before cleanup
            await self._save_shutdown_state_async()
            
            # Execute shutdown hooks
            await self._execute_hooks_async()
            
            # Final cleanup
            await self._final_cleanup_async()
            
        except Exception as e:
            self.logger.error(f"Error during async shutdown: {e}", exc_info=True)
        finally:
            self._shutdown_completed = True
            self._shutdown_stats.end_time = time.time()
            self._shutdown_stats.total_duration = (
                self._shutdown_stats.end_time - self._shutdown_stats.start_time
            )
            
            self.logger.info(f"Async shutdown completed in {self._shutdown_stats.total_duration:.2f}s")
            self._log_shutdown_stats()
            
            if self._shutdown_event:
                self._shutdown_event.set()
    
    def _execute_hooks_sync(self):
        """Execute shutdown hooks synchronously"""
        for hook in self._hooks:
            if self._execute_hook_sync(hook):
                self._shutdown_stats.hooks_executed += 1
            else:
                self._shutdown_stats.hooks_failed += 1
                if hook.critical:
                    self._shutdown_stats.critical_failures.append(hook.name)
    
    async def _execute_hooks_async(self):
        """Execute shutdown hooks asynchronously"""
        async_hooks = [h for h in self._hooks if h.is_async]
        sync_hooks = [h for h in self._hooks if not h.is_async]
        
        # Execute async hooks concurrently
        if async_hooks:
            tasks = [self._execute_hook_async(hook) for hook in async_hooks]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result, hook in zip(results, async_hooks):
                if isinstance(result, Exception):
                    self.logger.error(f"Async hook {hook.name} failed: {result}")
                    self._shutdown_stats.hooks_failed += 1
                    if hook.critical:
                        self._shutdown_stats.critical_failures.append(hook.name)
                else:
                    self._shutdown_stats.hooks_executed += 1
        
        # Execute sync hooks in thread pool
        if sync_hooks:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(self._cleanup_executor, self._execute_hook_sync, hook)
                for hook in sync_hooks
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result, hook in zip(results, sync_hooks):
                if isinstance(result, Exception) or not result:
                    self._shutdown_stats.hooks_failed += 1
                    if hook.critical:
                        self._shutdown_stats.critical_failures.append(hook.name)
                else:
                    self._shutdown_stats.hooks_executed += 1
    
    def _execute_hook_sync(self, hook: ShutdownHook) -> bool:
        """Execute a single hook synchronously"""
        try:
            self.logger.debug(f"Executing hook: {hook.name}")
            
            if hook.is_async:
                self.logger.warning(f"Cannot execute async hook {hook.name} in sync context")
                return False
            
            # Execute with timeout
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(hook.callback)
                result = future.result(timeout=hook.timeout)
            
            self._hook_results[hook.name] = True
            self.logger.debug(f"Hook {hook.name} completed successfully")
            return True
            
        except concurrent.futures.TimeoutError:
            self.logger.warning(f"Hook {hook.name} timed out after {hook.timeout}s")
            self._shutdown_stats.hooks_timeout += 1
            return False
        except Exception as e:
            self.logger.error(f"Hook {hook.name} failed: {e}", exc_info=True)
            return False
    
    async def _execute_hook_async(self, hook: ShutdownHook) -> bool:
        """Execute a single hook asynchronously"""
        try:
            self.logger.debug(f"Executing async hook: {hook.name}")
            
            if hook.is_async:
                # Execute async hook with timeout
                result = await asyncio.wait_for(hook.callback(), timeout=hook.timeout)
            else:
                # Execute sync hook in executor
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(self._cleanup_executor, hook.callback),
                    timeout=hook.timeout
                )
            
            self._hook_results[hook.name] = True
            self.logger.debug(f"Async hook {hook.name} completed successfully")
            return True
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Async hook {hook.name} timed out after {hook.timeout}s")
            self._shutdown_stats.hooks_timeout += 1
            return False
        except Exception as e:
            self.logger.error(f"Async hook {hook.name} failed: {e}", exc_info=True)
            return False
    
    def _save_shutdown_state(self):
        """Save shutdown state synchronously"""
        try:
            state_data = {
                "timestamp": time.time(),
                "reason": self._shutdown_reason.value if self._shutdown_reason else None,
                "app_name": self.app_name,
                "hooks_registered": len(self._hooks),
                "resources_count": len(self._resources),
                "shutdown_stats": {
                    "start_time": self._shutdown_stats.start_time,
                    "reason": self._shutdown_stats.reason.value if self._shutdown_stats.reason else None
                }
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                
            self.logger.debug(f"Shutdown state saved to {self.state_file}")
        except Exception as e:
            self.logger.error(f"Failed to save shutdown state: {e}")
    
    async def _save_shutdown_state_async(self):
        """Save shutdown state asynchronously"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._save_shutdown_state)
    
    def _final_cleanup(self):
        """Final cleanup operations"""
        try:
            # Cleanup thread pool
            self._cleanup_executor.shutdown(wait=True, timeout=10)
            
            # Restore signal handlers
            if self._original_handlers:
                for signum, handler in self._original_handlers.items():
                    if handler is not None:
                        signal.signal(signum, handler)
            
            self.logger.debug("Final cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error in final cleanup: {e}")
    
    async def _final_cleanup_async(self):
        """Final cleanup operations asynchronously"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._final_cleanup)
    
    def _emergency_cleanup(self):
        """Emergency cleanup called by atexit"""
        if not self._shutdown_completed and not self._shutdown_in_progress:
            self.logger.warning("Emergency cleanup triggered")
            try:
                self.shutdown(ShutdownReason.ERROR)
            except:
                pass  # Ignore errors during emergency cleanup
    
    def _register_builtin_hooks(self):
        """Register built-in shutdown hooks"""
        # Close file handles
        self.register_hook(
            "close_file_handles",
            self._cleanup_file_handles,
            priority=100,
            resource_type=ResourceType.FILE_HANDLE
        )
        
        # Cleanup temporary files
        self.register_hook(
            "cleanup_temp_files",
            self._cleanup_temp_files,
            priority=90,
            resource_type=ResourceType.TEMP_FILE
        )
        
        # Final logging
        self.register_hook(
            "final_log",
            self._final_log,
            priority=-100  # Run last
        )
    
    def _cleanup_file_handles(self):
        """Cleanup open file handles"""
        import gc
        import io
        
        closed_count = 0
        for obj in gc.get_objects():
            if isinstance(obj, io.IOBase) and not obj.closed:
                try:
                    obj.close()
                    closed_count += 1
                except:
                    pass
        
        if closed_count > 0:
            self.logger.debug(f"Closed {closed_count} file handles")
    
    def _cleanup_temp_files(self):
        """Cleanup temporary files"""
        import tempfile
        import shutil
        
        temp_dir = Path(tempfile.gettempdir())
        pattern = f"{self.app_name}_*"
        
        cleanup_count = 0
        for temp_file in temp_dir.glob(pattern):
            try:
                if temp_file.is_file():
                    temp_file.unlink()
                elif temp_file.is_dir():
                    shutil.rmtree(temp_file)
                cleanup_count += 1
            except:
                pass
        
        if cleanup_count > 0:
            self.logger.debug(f"Cleaned up {cleanup_count} temporary files")
    
    def _final_log(self):
        """Final logging before shutdown"""
        self.logger.info(f"Shutdown process completed for {self.app_name}")
    
    def _log_shutdown_stats(self):
        """Log shutdown statistics"""
        stats = self._shutdown_stats
        self.logger.info(
            f"Shutdown Statistics: "
            f"executed={stats.hooks_executed}, "
            f"failed={stats.hooks_failed}, "
            f"timeout={stats.hooks_timeout}, "
            f"duration={stats.total_duration:.2f}s"
        )
        
        if stats.critical_failures:
            self.logger.error(f"Critical hook failures: {', '.join(stats.critical_failures)}")
    
    @contextmanager
    def shutdown_context(self):
        """Context manager for automatic shutdown handling"""
        try:
            yield self
        finally:
            if not self._shutdown_completed:
                self.shutdown()
    
    @asynccontextmanager
    async def async_shutdown_context(self):
        """Async context manager for automatic shutdown handling"""
        try:
            yield self
        finally:
            if not self._shutdown_completed:
                await self.shutdown_async()
    
    def wait_for_shutdown(self, timeout: Optional[float] = None):
        """Wait for shutdown to complete"""
        start_time = time.time()
        while not self._shutdown_completed:
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError("Shutdown did not complete within timeout")
            time.sleep(0.1)
    
    async def wait_for_shutdown_async(self, timeout: Optional[float] = None):
        """Wait for shutdown to complete asynchronously"""
        if self._shutdown_event:
            if timeout:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=timeout)
            else:
                await self._shutdown_event.wait()
        else:
            # Fallback polling
            start_time = time.time()
            while not self._shutdown_completed:
                if timeout and (time.time() - start_time) > timeout:
                    raise TimeoutError("Shutdown did not complete within timeout")
                await asyncio.sleep(0.1)


# Global shutdown manager instance
_shutdown_manager: Optional[ShutdownManager] = None


def get_shutdown_manager() -> ShutdownManager:
    """Get or create the global shutdown manager"""
    global _shutdown_manager
    if _shutdown_manager is None:
        _shutdown_manager = ShutdownManager()
    return _shutdown_manager


def register_shutdown_hook(name: str, callback: Callable, **kwargs) -> str:
    """Convenience function to register a shutdown hook"""
    return get_shutdown_manager().register_hook(name, callback, **kwargs)


def register_resource(name: str, resource: Any, cleanup_func: Optional[Callable] = None, **kwargs) -> str:
    """Convenience function to register a resource"""
    return get_shutdown_manager().register_resource(name, resource, cleanup_func, **kwargs)


def shutdown(reason: ShutdownReason = ShutdownReason.USER_REQUEST):
    """Convenience function to initiate shutdown"""
    get_shutdown_manager().shutdown(reason)


async def shutdown_async(reason: ShutdownReason = ShutdownReason.USER_REQUEST):
    """Convenience function to initiate async shutdown"""
    await get_shutdown_manager().shutdown_async(reason) 