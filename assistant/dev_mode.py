#!/usr/bin/env python3
"""
Sovereign Voice Assistant - Developer Mode
Provides hot reload, debugging features, and development utilities
"""

import asyncio
import logging
import os
import sys
import time
import threading
import traceback
import weakref
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable, Union
import json
import functools
import inspect
import psutil
import gc

# Development dependencies
try:
    import watchdog
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("Warning: watchdog not available. Install with: pip install watchdog")

try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    print("Warning: memory_profiler not available. Install with: pip install memory-profiler")

try:
    import line_profiler
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False

try:
    import py_spy
    PY_SPY_AVAILABLE = True
except ImportError:
    PY_SPY_AVAILABLE = False


@dataclass
class DevStats:
    """Development statistics and metrics"""
    start_time: float = field(default_factory=time.time)
    reload_count: int = 0
    last_reload: Optional[float] = None
    errors_count: int = 0
    warnings_count: int = 0
    memory_usage: List[float] = field(default_factory=list)
    cpu_usage: List[float] = field(default_factory=list)
    modules_watched: Set[str] = field(default_factory=set)
    debug_calls: int = 0
    profiling_active: bool = False


@dataclass
class HotReloadConfig:
    """Configuration for hot reload functionality"""
    enabled: bool = True
    watch_patterns: List[str] = field(default_factory=lambda: ["*.py"])
    ignore_patterns: List[str] = field(default_factory=lambda: ["__pycache__", "*.pyc", ".git"])
    debounce_seconds: float = 1.0
    recursive: bool = True
    reload_on_import: bool = True


@dataclass
class DebugConfig:
    """Configuration for debug features"""
    enabled: bool = True
    log_level: str = "DEBUG"
    memory_tracking: bool = True
    performance_tracking: bool = True
    profiling: bool = False
    auto_profiler: bool = False
    debug_server: bool = True
    debug_port: int = 8888


class FileChangeHandler(FileSystemEventHandler):
    """Handle file system changes for hot reload"""
    
    def __init__(self, dev_manager: 'DevManager'):
        super().__init__()
        self.dev_manager = dev_manager
        self.last_reload = {}
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        
        # Check if file matches watch patterns
        if not self._should_watch_file(file_path):
            return
            
        # Debounce rapid changes
        now = time.time()
        if file_path in self.last_reload:
            if now - self.last_reload[file_path] < self.dev_manager.hot_reload_config.debounce_seconds:
                return
                
        self.last_reload[file_path] = now
        
        # Schedule reload
        self.dev_manager._schedule_reload(file_path)
    
    def _should_watch_file(self, file_path: Path) -> bool:
        """Check if file should be watched"""
        config = self.dev_manager.hot_reload_config
        
        # Check ignore patterns
        for pattern in config.ignore_patterns:
            if file_path.match(pattern):
                return False
                
        # Check watch patterns
        for pattern in config.watch_patterns:
            if file_path.match(pattern):
                return True
                
        return False


class DevManager:
    """
    Developer mode manager with hot reload and debugging features
    
    Features:
    - Hot reload for Python modules
    - Performance and memory monitoring
    - Debug logging and statistics
    - Development server with API
    - Profiling integration
    - Error tracking and reporting
    """
    
    def __init__(self, 
                 project_root: Optional[Path] = None,
                 hot_reload_config: Optional[HotReloadConfig] = None,
                 debug_config: Optional[DebugConfig] = None):
        """Initialize developer mode manager"""
        self.project_root = project_root or Path.cwd()
        self.hot_reload_config = hot_reload_config or HotReloadConfig()
        self.debug_config = debug_config or DebugConfig()
        
        # Development state
        self.enabled = False
        self.stats = DevStats()
        self._reload_queue = asyncio.Queue() if sys.version_info >= (3, 7) else None
        self._reload_lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # File watching
        self.observer: Optional[Observer] = None
        self.file_handler: Optional[FileChangeHandler] = None
        
        # Module tracking
        self._original_modules = set(sys.modules.keys())
        self._watched_modules: Dict[str, float] = {}
        self._reload_callbacks: Dict[str, List[Callable]] = {}
        
        # Performance monitoring
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_interval = 5.0
        
        # Debug server
        self._debug_server: Optional['DebugServer'] = None
        
        # Profiling
        self._profiler = None
        self._profile_data = {}
        
        # Logging
        self.logger = logging.getLogger("dev_mode")
        self._setup_dev_logging()
        
        self.logger.info(f"Developer mode manager initialized (root: {self.project_root})")
    
    def _setup_dev_logging(self):
        """Setup development logging"""
        if not self.debug_config.enabled:
            return
            
        # Set debug level
        log_level = getattr(logging, self.debug_config.log_level.upper(), logging.DEBUG)
        
        # Create formatter with more details
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        
        # File handler
        log_file = self.project_root / "logs" / "dev_mode.log"
        log_file.parent.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
    
    def enable(self):
        """Enable developer mode"""
        if self.enabled:
            self.logger.warning("Developer mode already enabled")
            return
            
        self.logger.info("Enabling developer mode...")
        
        # Start file watching
        if self.hot_reload_config.enabled and WATCHDOG_AVAILABLE:
            self._start_file_watching()
        
        # Start performance monitoring
        if self.debug_config.performance_tracking:
            self._start_performance_monitoring()
            
        # Start debug server
        if self.debug_config.debug_server:
            self._start_debug_server()
            
        # Setup profiling
        if self.debug_config.profiling:
            self._setup_profiling()
        
        self.enabled = True
        self.stats.start_time = time.time()
        
        self.logger.info("Developer mode enabled successfully")
    
    def disable(self):
        """Disable developer mode"""
        if not self.enabled:
            return
            
        self.logger.info("Disabling developer mode...")
        
        # Stop file watching
        if self.observer:
            self.observer.stop()
            self.observer.join()
            
        # Stop monitoring
        self._shutdown_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
            
        # Stop debug server
        if self._debug_server:
            self._debug_server.stop()
            
        # Stop profiling
        self._stop_profiling()
        
        self.enabled = False
        self.logger.info("Developer mode disabled")
    
    def _start_file_watching(self):
        """Start file system watching for hot reload"""
        if not WATCHDOG_AVAILABLE:
            self.logger.warning("Watchdog not available, hot reload disabled")
            return
            
        self.logger.info("Starting file system watching...")
        
        self.file_handler = FileChangeHandler(self)
        self.observer = Observer()
        
        # Watch project root
        self.observer.schedule(
            self.file_handler,
            str(self.project_root),
            recursive=self.hot_reload_config.recursive
        )
        
        # Watch additional Python paths
        for path in sys.path:
            if Path(path).exists() and Path(path).is_dir():
                try:
                    self.observer.schedule(
                        self.file_handler,
                        str(path),
                        recursive=False
                    )
                except:
                    pass  # Ignore errors for system paths
        
        self.observer.start()
        self.logger.info("File system watching started")
    
    def _start_performance_monitoring(self):
        """Start performance monitoring thread"""
        self.logger.info("Starting performance monitoring...")
        
        def monitor_loop():
            while not self._shutdown_event.is_set():
                try:
                    # Memory usage
                    if MEMORY_PROFILER_AVAILABLE:
                        memory_mb = memory_profiler.memory_usage()[0]
                        self.stats.memory_usage.append(memory_mb)
                        
                        # Keep only last 100 readings
                        if len(self.stats.memory_usage) > 100:
                            self.stats.memory_usage.pop(0)
                    
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=None)
                    self.stats.cpu_usage.append(cpu_percent)
                    
                    if len(self.stats.cpu_usage) > 100:
                        self.stats.cpu_usage.pop(0)
                    
                    # Garbage collection stats
                    if len(self.stats.memory_usage) % 20 == 0:  # Every 100 seconds
                        collected = gc.collect()
                        if collected > 0:
                            self.logger.debug(f"Garbage collected {collected} objects")
                    
                except Exception as e:
                    self.logger.error(f"Error in performance monitoring: {e}")
                
                self._shutdown_event.wait(self._monitor_interval)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def _start_debug_server(self):
        """Start debug server for development tools"""
        try:
            from .debug_server import DebugServer
            self._debug_server = DebugServer(
                self,
                port=self.debug_config.debug_port
            )
            self._debug_server.start()
            self.logger.info(f"Debug server started on port {self.debug_config.debug_port}")
        except ImportError:
            self.logger.warning("Debug server not available")
        except Exception as e:
            self.logger.error(f"Failed to start debug server: {e}")
    
    def _setup_profiling(self):
        """Setup profiling tools"""
        if LINE_PROFILER_AVAILABLE:
            try:
                self._profiler = line_profiler.LineProfiler()
                self.stats.profiling_active = True
                self.logger.info("Line profiler enabled")
            except Exception as e:
                self.logger.error(f"Failed to setup line profiler: {e}")
    
    def _stop_profiling(self):
        """Stop profiling and save results"""
        if self._profiler:
            try:
                # Save profiling results
                profile_file = self.project_root / "logs" / "profile_results.txt"
                with open(profile_file, 'w') as f:
                    self._profiler.print_stats(stream=f)
                self.logger.info(f"Profiling results saved to {profile_file}")
            except Exception as e:
                self.logger.error(f"Error saving profiling results: {e}")
            
            self._profiler = None
            self.stats.profiling_active = False
    
    def _schedule_reload(self, file_path: Path):
        """Schedule a module reload"""
        if self._reload_queue:
            try:
                self._reload_queue.put_nowait(file_path)
            except asyncio.QueueFull:
                self.logger.warning("Reload queue full, skipping reload")
        else:
            # Sync reload
            self._perform_reload(file_path)
    
    def _perform_reload(self, file_path: Path):
        """Perform actual module reload"""
        with self._reload_lock:
            try:
                self.logger.info(f"Reloading module for file: {file_path}")
                
                # Find module name from file path
                module_name = self._get_module_name(file_path)
                if not module_name:
                    return
                
                # Reload module
                if module_name in sys.modules:
                    import importlib
                    module = sys.modules[module_name]
                    importlib.reload(module)
                    
                    # Execute reload callbacks
                    if module_name in self._reload_callbacks:
                        for callback in self._reload_callbacks[module_name]:
                            try:
                                callback(module)
                            except Exception as e:
                                self.logger.error(f"Reload callback failed: {e}")
                    
                    self.stats.reload_count += 1
                    self.stats.last_reload = time.time()
                    
                    self.logger.info(f"Successfully reloaded module: {module_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to reload {file_path}: {e}")
                self.stats.errors_count += 1
    
    def _get_module_name(self, file_path: Path) -> Optional[str]:
        """Get module name from file path"""
        try:
            # Convert file path to module name
            relative_path = file_path.relative_to(self.project_root)
            module_parts = list(relative_path.parts)
            
            # Remove .py extension
            if module_parts[-1].endswith('.py'):
                module_parts[-1] = module_parts[-1][:-3]
            
            # Handle __init__.py
            if module_parts[-1] == '__init__':
                module_parts.pop()
            
            return '.'.join(module_parts)
            
        except (ValueError, IndexError):
            return None
    
    def register_reload_callback(self, module_name: str, callback: Callable):
        """Register callback for module reload"""
        if module_name not in self._reload_callbacks:
            self._reload_callbacks[module_name] = []
        self._reload_callbacks[module_name].append(callback)
        self.logger.debug(f"Registered reload callback for {module_name}")
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile a function"""
        if not self.stats.profiling_active or not self._profiler:
            return func
            
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self._profiler.add_function(func)
            self._profiler.enable_by_count()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                self._profiler.disable_by_count()
        
        return wrapper
    
    def debug_point(self, message: str, **kwargs):
        """Add a debug point with context"""
        self.stats.debug_calls += 1
        
        # Get caller info
        frame = inspect.currentframe().f_back
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        func_name = frame.f_code.co_name
        
        # Log with context
        context = {
            'file': Path(filename).name,
            'line': lineno,
            'function': func_name,
            'locals': {k: str(v)[:100] for k, v in frame.f_locals.items() if not k.startswith('_')},
            **kwargs
        }
        
        self.logger.debug(f"DEBUG POINT: {message} | Context: {context}")
    
    def memory_snapshot(self, label: str = ""):
        """Take a memory snapshot"""
        if not MEMORY_PROFILER_AVAILABLE:
            self.logger.warning("Memory profiler not available")
            return
            
        try:
            memory_mb = memory_profiler.memory_usage()[0]
            self.logger.info(f"Memory snapshot [{label}]: {memory_mb:.2f} MB")
            
            # Detailed memory info
            process = psutil.Process()
            memory_info = process.memory_info()
            
            details = {
                'rss': memory_info.rss / 1024 / 1024,  # MB
                'vms': memory_info.vms / 1024 / 1024,  # MB
                'percent': process.memory_percent(),
                'gc_counts': gc.get_count()
            }
            
            self.logger.debug(f"Memory details [{label}]: {details}")
            
        except Exception as e:
            self.logger.error(f"Error taking memory snapshot: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get development statistics"""
        uptime = time.time() - self.stats.start_time
        
        return {
            'enabled': self.enabled,
            'uptime_seconds': uptime,
            'reload_count': self.stats.reload_count,
            'last_reload': self.stats.last_reload,
            'errors_count': self.stats.errors_count,
            'warnings_count': self.stats.warnings_count,
            'debug_calls': self.stats.debug_calls,
            'profiling_active': self.stats.profiling_active,
            'modules_watched': len(self.stats.modules_watched),
            'memory_usage_mb': self.stats.memory_usage[-1] if self.stats.memory_usage else None,
            'cpu_usage_percent': self.stats.cpu_usage[-1] if self.stats.cpu_usage else None,
            'performance_history': {
                'memory': self.stats.memory_usage[-10:],  # Last 10 readings
                'cpu': self.stats.cpu_usage[-10:]
            }
        }
    
    def save_debug_report(self, filename: Optional[str] = None):
        """Save comprehensive debug report"""
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"debug_report_{timestamp}.json"
        
        report_file = self.project_root / "logs" / filename
        report_file.parent.mkdir(exist_ok=True)
        
        report = {
            'timestamp': time.time(),
            'stats': self.get_stats(),
            'config': {
                'hot_reload': {
                    'enabled': self.hot_reload_config.enabled,
                    'watch_patterns': self.hot_reload_config.watch_patterns,
                    'ignore_patterns': self.hot_reload_config.ignore_patterns
                },
                'debug': {
                    'enabled': self.debug_config.enabled,
                    'log_level': self.debug_config.log_level,
                    'memory_tracking': self.debug_config.memory_tracking,
                    'performance_tracking': self.debug_config.performance_tracking
                }
            },
            'environment': {
                'python_version': sys.version,
                'platform': sys.platform,
                'pid': os.getpid(),
                'cwd': str(Path.cwd()),
                'modules_count': len(sys.modules)
            }
        }
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Debug report saved to {report_file}")
            return report_file
            
        except Exception as e:
            self.logger.error(f"Failed to save debug report: {e}")
            return None
    
    async def reload_loop(self):
        """Async loop for handling reload queue"""
        if not self._reload_queue:
            return
            
        while self.enabled:
            try:
                # Wait for reload request
                file_path = await asyncio.wait_for(
                    self._reload_queue.get(), 
                    timeout=1.0
                )
                
                # Perform reload in thread
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, 
                    self._perform_reload, 
                    file_path
                )
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in reload loop: {e}")
    
    @contextmanager
    def performance_context(self, label: str):
        """Context manager for performance measurement"""
        start_time = time.time()
        start_memory = None
        
        if MEMORY_PROFILER_AVAILABLE:
            start_memory = memory_profiler.memory_usage()[0]
        
        self.logger.debug(f"Performance context started: {label}")
        
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            memory_delta = None
            if start_memory and MEMORY_PROFILER_AVAILABLE:
                end_memory = memory_profiler.memory_usage()[0]
                memory_delta = end_memory - start_memory
            
            self.logger.info(
                f"Performance [{label}]: {duration:.3f}s"
                + (f", memory: {memory_delta:+.2f}MB" if memory_delta else "")
            )
    
    @asynccontextmanager
    async def async_performance_context(self, label: str):
        """Async context manager for performance measurement"""
        start_time = time.time()
        start_memory = None
        
        if MEMORY_PROFILER_AVAILABLE:
            start_memory = memory_profiler.memory_usage()[0]
        
        self.logger.debug(f"Async performance context started: {label}")
        
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            memory_delta = None
            if start_memory and MEMORY_PROFILER_AVAILABLE:
                end_memory = memory_profiler.memory_usage()[0]
                memory_delta = end_memory - start_memory
            
            self.logger.info(
                f"Async Performance [{label}]: {duration:.3f}s"
                + (f", memory: {memory_delta:+.2f}MB" if memory_delta else "")
            )


# Global developer mode instance
_dev_manager: Optional[DevManager] = None


def get_dev_manager() -> DevManager:
    """Get or create the global developer mode manager"""
    global _dev_manager
    if _dev_manager is None:
        _dev_manager = DevManager()
    return _dev_manager


def enable_dev_mode():
    """Enable developer mode"""
    get_dev_manager().enable()


def disable_dev_mode():
    """Disable developer mode"""
    get_dev_manager().disable()


def debug_point(message: str, **kwargs):
    """Convenience function for debug points"""
    get_dev_manager().debug_point(message, **kwargs)


def memory_snapshot(label: str = ""):
    """Convenience function for memory snapshots"""
    get_dev_manager().memory_snapshot(label)


def profile_function(func: Callable) -> Callable:
    """Convenience decorator for function profiling"""
    return get_dev_manager().profile_function(func)


def performance_context(label: str):
    """Convenience function for performance context"""
    return get_dev_manager().performance_context(label) 