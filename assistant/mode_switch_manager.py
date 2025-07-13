"""
Sovereign 4.0 Voice Assistant - Mode Manager

Manages operation modes for the voice assistant:
- REALTIME_ONLY: Pure OpenAI Realtime API mode
- TRADITIONAL_ONLY: Pure traditional STT->LLM->TTS pipeline  
- HYBRID_AUTO: Automatic hybrid mode switching

The ModeManager enforces mode restrictions, handles mode-specific initialization,
and manages transitions between modes with proper resource allocation.

Usage:
    mode_manager = ModeManager.get_instance()
    mode_manager.initialize(config)
    
    # Check if a feature is available in current mode
    if mode_manager.is_realtime_available():
        # Use realtime API
        pass
    
    # Switch modes (if supported)
    await mode_manager.switch_mode(OperationMode.REALTIME_ONLY)
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, List, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading
import sys
import os

# Add the current directory to path for direct import
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from config_manager import SovereignConfig, OperationMode

logger = logging.getLogger(__name__)

@dataclass
class ModeCapabilities:
    """Defines capabilities available in each operation mode"""
    can_use_realtime_api: bool = False
    can_use_traditional_pipeline: bool = False
    can_use_screen_monitoring: bool = True
    can_use_memory_system: bool = True
    can_use_code_agent: bool = True
    requires_openai_key: bool = False
    requires_stt_provider: bool = False
    requires_tts_provider: bool = False

@dataclass
class ModeTransition:
    """Represents a mode transition event"""
    from_mode: OperationMode
    to_mode: OperationMode
    timestamp: datetime
    reason: str
    success: bool
    duration_ms: float

@dataclass
class ModeMetrics:
    """Performance metrics for operation modes"""
    mode: OperationMode
    total_sessions: int = 0
    successful_sessions: int = 0
    failed_sessions: int = 0
    average_response_time_ms: float = 0.0
    total_cost_usd: float = 0.0
    last_used: Optional[datetime] = None
    total_uptime_seconds: float = 0.0

class ModeValidationError(Exception):
    """Raised when mode validation fails"""
    pass

class ModeTransitionError(Exception):
    """Raised when mode transition fails"""
    pass

class ModeManager:
    """
    Singleton Mode Manager for Sovereign Voice Assistant
    
    Manages operation modes with enforcement, initialization, and state management.
    Provides centralized control over feature availability and mode transitions.
    """
    
    _instance: Optional['ModeManager'] = None
    _lock = threading.Lock()
    
    def __init__(self):
        """Initialize ModeManager (use get_instance() instead)"""
        if ModeManager._instance is not None:
            raise RuntimeError("ModeManager is a singleton. Use get_instance() instead.")
        
        self._config: Optional[SovereignConfig] = None
        self._current_mode: Optional[OperationMode] = None
        self._initialized: bool = False
        self._mode_capabilities: Dict[OperationMode, ModeCapabilities] = {}
        self._mode_metrics: Dict[OperationMode, ModeMetrics] = {}
        self._mode_history: List[ModeTransition] = []
        self._mode_change_callbacks: List[Callable[[OperationMode, OperationMode], None]] = []
        self._resource_cleanup_callbacks: List[Callable[[], None]] = []
        self._initialization_start_time: Optional[datetime] = None
        
        # Initialize mode capabilities
        self._setup_mode_capabilities()
        
        logger.info("ModeManager singleton created")
    
    @classmethod
    def get_instance(cls) -> 'ModeManager':
        """Get singleton instance of ModeManager"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)"""
        with cls._lock:
            if cls._instance:
                cls._instance.cleanup()
            cls._instance = None
    
    def _setup_mode_capabilities(self) -> None:
        """Setup capabilities for each operation mode"""
        self._mode_capabilities = {
            OperationMode.REALTIME_ONLY: ModeCapabilities(
                can_use_realtime_api=True,
                can_use_traditional_pipeline=False,
                can_use_screen_monitoring=True,
                can_use_memory_system=True,
                can_use_code_agent=True,
                requires_openai_key=True,
                requires_stt_provider=False,
                requires_tts_provider=False
            ),
            OperationMode.TRADITIONAL_ONLY: ModeCapabilities(
                can_use_realtime_api=False,
                can_use_traditional_pipeline=True,
                can_use_screen_monitoring=True,
                can_use_memory_system=True,
                can_use_code_agent=True,
                requires_openai_key=False,
                requires_stt_provider=True,
                requires_tts_provider=True
            ),
            OperationMode.HYBRID_AUTO: ModeCapabilities(
                can_use_realtime_api=True,
                can_use_traditional_pipeline=True,
                can_use_screen_monitoring=True,
                can_use_memory_system=True,
                can_use_code_agent=True,
                requires_openai_key=False,  # Not strictly required, but enables realtime
                requires_stt_provider=True,  # Required for fallback
                requires_tts_provider=True   # Required for fallback
            )
        }
        
        # Initialize metrics for each mode
        for mode in OperationMode:
            self._mode_metrics[mode] = ModeMetrics(mode=mode)
    
    def initialize(self, config: SovereignConfig) -> None:
        """
        Initialize ModeManager with configuration
        
        Args:
            config: Sovereign configuration containing operation mode settings
        """
        self._initialization_start_time = datetime.now()
        
        try:
            logger.info(f"Initializing ModeManager with operation mode: {config.operation_mode}")
            
            self._config = config
            self._current_mode = config.operation_mode
            
            # Validate mode requirements
            self._validate_mode_requirements(config.operation_mode, config)
            
            # Perform mode-specific initialization
            self._initialize_mode_resources(config.operation_mode, config)
            
            self._initialized = True
            
            initialization_time = (datetime.now() - self._initialization_start_time).total_seconds() * 1000
            logger.info(f"ModeManager initialized successfully in {initialization_time:.2f}ms")
            
        except Exception as e:
            logger.error(f"Failed to initialize ModeManager: {e}")
            raise ModeValidationError(f"ModeManager initialization failed: {e}")
    
    def _validate_mode_requirements(self, mode: OperationMode, config: SovereignConfig) -> None:
        """Validate that configuration meets requirements for the specified mode"""
        capabilities = self._mode_capabilities[mode]
        errors = []
        
        # Check API key requirements
        if capabilities.requires_openai_key and not config.development.mock_apis:
            if not config.api.openai_api_key:
                errors.append(f"Mode {mode.value} requires OpenAI API key")
        
        # Check STT provider requirements
        if capabilities.requires_stt_provider:
            if not config.stt.primary_provider:
                errors.append(f"Mode {mode.value} requires STT provider configuration")
        
        # Check TTS provider requirements  
        if capabilities.requires_tts_provider:
            if not config.tts.primary_provider:
                errors.append(f"Mode {mode.value} requires TTS provider configuration")
        
        # Check realtime API requirements
        if mode == OperationMode.REALTIME_ONLY:
            if not config.realtime_api.enabled:
                errors.append("REALTIME_ONLY mode requires realtime_api.enabled to be true")
        
        if errors:
            raise ModeValidationError(f"Mode validation failed: {'; '.join(errors)}")
        
        logger.info(f"Mode {mode.value} validation passed")
    
    def _initialize_mode_resources(self, mode: OperationMode, config: SovereignConfig) -> None:
        """Initialize resources specific to the operation mode"""
        logger.info(f"Initializing resources for mode: {mode.value}")
        
        capabilities = self._mode_capabilities[mode]
        
        # Initialize realtime API resources if needed
        if capabilities.can_use_realtime_api and config.realtime_api.enabled:
            self._initialize_realtime_resources(config)
        
        # Initialize traditional pipeline resources if needed
        if capabilities.can_use_traditional_pipeline:
            self._initialize_traditional_resources(config)
        
        # Initialize shared resources
        self._initialize_shared_resources(config)
        
        logger.info(f"Resource initialization completed for mode: {mode.value}")
    
    def _initialize_realtime_resources(self, config: SovereignConfig) -> None:
        """Initialize OpenAI Realtime API resources"""
        logger.info("Initializing Realtime API resources")
        # TODO: Initialize realtime API connections, session management, etc.
        pass
    
    def _initialize_traditional_resources(self, config: SovereignConfig) -> None:
        """Initialize traditional pipeline resources (STT, LLM, TTS)"""
        logger.info("Initializing traditional pipeline resources")
        # TODO: Initialize STT, LLM, TTS providers
        pass
    
    def _initialize_shared_resources(self, config: SovereignConfig) -> None:
        """Initialize shared resources (memory, screen monitoring, etc.)"""
        logger.info("Initializing shared resources")
        # TODO: Initialize memory system, screen monitoring, etc.
        pass
    
    def is_initialized(self) -> bool:
        """Check if ModeManager is initialized"""
        return self._initialized
    
    def get_current_mode(self) -> Optional[OperationMode]:
        """Get current operation mode"""
        return self._current_mode
    
    def get_capabilities(self, mode: Optional[OperationMode] = None) -> ModeCapabilities:
        """Get capabilities for specified mode (or current mode)"""
        target_mode = mode or self._current_mode
        if target_mode is None:
            raise RuntimeError("ModeManager not initialized")
        return self._mode_capabilities[target_mode]
    
    def is_realtime_available(self) -> bool:
        """Check if Realtime API is available in current mode"""
        if not self._initialized:
            return False
        return self.get_capabilities().can_use_realtime_api
    
    def is_traditional_pipeline_available(self) -> bool:
        """Check if traditional pipeline is available in current mode"""
        if not self._initialized:
            return False
        return self.get_capabilities().can_use_traditional_pipeline
    
    def is_screen_monitoring_available(self) -> bool:
        """Check if screen monitoring is available in current mode"""
        if not self._initialized:
            return False
        return self.get_capabilities().can_use_screen_monitoring
    
    def is_memory_system_available(self) -> bool:
        """Check if memory system is available in current mode"""
        if not self._initialized:
            return False
        return self.get_capabilities().can_use_memory_system
    
    def is_code_agent_available(self) -> bool:
        """Check if code agent is available in current mode"""
        if not self._initialized:
            return False
        return self.get_capabilities().can_use_code_agent
    
    async def switch_mode(self, new_mode: OperationMode, reason: str = "Manual switch") -> bool:
        """
        Switch to a new operation mode
        
        Args:
            new_mode: Target operation mode
            reason: Reason for the mode switch
            
        Returns:
            True if switch was successful, False otherwise
        """
        if not self._initialized:
            raise RuntimeError("ModeManager not initialized")
        
        if self._current_mode == new_mode:
            logger.info(f"Already in mode {new_mode.value}, no switch needed")
            return True
        
        start_time = time.time()
        old_mode = self._current_mode
        
        try:
            logger.info(f"Switching from {old_mode.value} to {new_mode.value}: {reason}")
            
            # Validate new mode requirements
            self._validate_mode_requirements(new_mode, self._config)
            
            # Cleanup resources from old mode
            await self._cleanup_mode_resources(old_mode)
            
            # Initialize resources for new mode
            self._initialize_mode_resources(new_mode, self._config)
            
            # Update current mode
            self._current_mode = new_mode
            
            # Record transition
            transition_time = (time.time() - start_time) * 1000
            transition = ModeTransition(
                from_mode=old_mode,
                to_mode=new_mode,
                timestamp=datetime.now(),
                reason=reason,
                success=True,
                duration_ms=transition_time
            )
            self._mode_history.append(transition)
            
            # Notify callbacks
            for callback in self._mode_change_callbacks:
                try:
                    callback(old_mode, new_mode)
                except Exception as e:
                    logger.error(f"Mode change callback failed: {e}")
            
            logger.info(f"Successfully switched to {new_mode.value} in {transition_time:.2f}ms")
            return True
            
        except Exception as e:
            # Record failed transition
            transition_time = (time.time() - start_time) * 1000
            transition = ModeTransition(
                from_mode=old_mode,
                to_mode=new_mode,
                timestamp=datetime.now(),
                reason=reason,
                success=False,
                duration_ms=transition_time
            )
            self._mode_history.append(transition)
            
            logger.error(f"Failed to switch to {new_mode.value}: {e}")
            raise ModeTransitionError(f"Mode switch failed: {e}")
    
    async def _cleanup_mode_resources(self, mode: OperationMode) -> None:
        """Cleanup resources for the specified mode"""
        logger.info(f"Cleaning up resources for mode: {mode.value}")
        
        # Call cleanup callbacks
        for cleanup_callback in self._resource_cleanup_callbacks:
            try:
                cleanup_callback()
            except Exception as e:
                logger.error(f"Resource cleanup callback failed: {e}")
        
        # Mode-specific cleanup would go here
        # TODO: Implement specific cleanup logic for each mode
    
    def add_mode_change_callback(self, callback: Callable[[OperationMode, OperationMode], None]) -> None:
        """Add callback to be called when mode changes"""
        self._mode_change_callbacks.append(callback)
    
    def remove_mode_change_callback(self, callback: Callable[[OperationMode, OperationMode], None]) -> None:
        """Remove mode change callback"""
        if callback in self._mode_change_callbacks:
            self._mode_change_callbacks.remove(callback)
    
    def add_resource_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Add callback for resource cleanup during mode switches"""
        self._resource_cleanup_callbacks.append(callback)
    
    def get_mode_metrics(self, mode: Optional[OperationMode] = None) -> ModeMetrics:
        """Get performance metrics for specified mode (or current mode)"""
        target_mode = mode or self._current_mode
        if target_mode is None:
            raise RuntimeError("ModeManager not initialized")
        return self._mode_metrics[target_mode]
    
    def get_mode_history(self, limit: Optional[int] = None) -> List[ModeTransition]:
        """Get mode transition history"""
        if limit:
            return self._mode_history[-limit:]
        return self._mode_history.copy()
    
    def update_session_metrics(self, success: bool, response_time_ms: float, cost_usd: float = 0.0) -> None:
        """Update session metrics for current mode"""
        if not self._current_mode:
            return
        
        metrics = self._mode_metrics[self._current_mode]
        metrics.total_sessions += 1
        metrics.last_used = datetime.now()
        
        if success:
            metrics.successful_sessions += 1
        else:
            metrics.failed_sessions += 1
        
        # Update average response time
        if metrics.total_sessions > 1:
            metrics.average_response_time_ms = (
                (metrics.average_response_time_ms * (metrics.total_sessions - 1) + response_time_ms) 
                / metrics.total_sessions
            )
        else:
            metrics.average_response_time_ms = response_time_ms
        
        metrics.total_cost_usd += cost_usd
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary"""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        current_metrics = self.get_mode_metrics()
        capabilities = self.get_capabilities()
        
        return {
            "status": "initialized",
            "current_mode": self._current_mode.value,
            "initialization_time": self._initialization_start_time.isoformat() if self._initialization_start_time else None,
            "capabilities": {
                "realtime_api": capabilities.can_use_realtime_api,
                "traditional_pipeline": capabilities.can_use_traditional_pipeline,
                "screen_monitoring": capabilities.can_use_screen_monitoring,
                "memory_system": capabilities.can_use_memory_system,
                "code_agent": capabilities.can_use_code_agent
            },
            "metrics": {
                "total_sessions": current_metrics.total_sessions,
                "success_rate": (
                    current_metrics.successful_sessions / current_metrics.total_sessions
                    if current_metrics.total_sessions > 0 else 0.0
                ),
                "average_response_time_ms": current_metrics.average_response_time_ms,
                "total_cost_usd": current_metrics.total_cost_usd,
                "last_used": current_metrics.last_used.isoformat() if current_metrics.last_used else None
            },
            "transition_history_count": len(self._mode_history)
        }
    
    def cleanup(self) -> None:
        """Cleanup ModeManager resources"""
        logger.info("Cleaning up ModeManager")
        
        if self._current_mode:
            asyncio.create_task(self._cleanup_mode_resources(self._current_mode))
        
        self._mode_change_callbacks.clear()
        self._resource_cleanup_callbacks.clear()
        self._initialized = False
        self._current_mode = None
        self._config = None
        
        logger.info("ModeManager cleanup completed")

# Convenience functions
def get_mode_manager() -> ModeManager:
    """Get the singleton ModeManager instance"""
    return ModeManager.get_instance()

def is_realtime_mode() -> bool:
    """Check if currently in realtime-only mode"""
    manager = get_mode_manager()
    return manager.is_initialized() and manager.get_current_mode() == OperationMode.REALTIME_ONLY

def is_traditional_mode() -> bool:
    """Check if currently in traditional-only mode"""
    manager = get_mode_manager()
    return manager.is_initialized() and manager.get_current_mode() == OperationMode.TRADITIONAL_ONLY

def is_hybrid_mode() -> bool:
    """Check if currently in hybrid auto mode"""
    manager = get_mode_manager()
    return manager.is_initialized() and manager.get_current_mode() == OperationMode.HYBRID_AUTO 