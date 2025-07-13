"""
Comprehensive Error Handler for OpenAI Realtime API

Provides robust error handling with:
- Error categorization and classification
- Recovery strategies with exponential backoff
- Rate limit handling with retry-after headers
- Audio stream restart mechanisms
- Graceful degradation (audio quality, text mode fallback)
- Error aggregation and reporting
- User-friendly error messages and notifications
"""

import asyncio
import logging
import time
import json
from typing import Optional, Dict, List, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
import math
import random
from collections import defaultdict, deque
import traceback

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Error categories for Realtime API"""
    CONNECTION_ERROR = "connection_error"
    API_ERROR = "api_error"
    AUDIO_ERROR = "audio_error"
    TIMEOUT_ERROR = "timeout_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    AUTHENTICATION_ERROR = "authentication_error"
    NETWORK_ERROR = "network_error"
    SESSION_ERROR = "session_error"
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types"""
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    RECONNECT = "reconnect"
    RESTART_AUDIO = "restart_audio"
    REDUCE_QUALITY = "reduce_quality"
    FALLBACK_TO_TEXT = "fallback_to_text"
    WAIT_AND_RETRY = "wait_and_retry"
    TERMINATE_SESSION = "terminate_session"
    NO_RECOVERY = "no_recovery"


@dataclass
class ErrorInfo:
    """Detailed error information"""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    session_id: Optional[str] = None
    recovery_strategy: Optional[RecoveryStrategy] = None
    retry_count: int = 0
    max_retries: int = 3
    backoff_delay: float = 1.0
    user_message: Optional[str] = None
    recovery_action: Optional[str] = None
    traceback_info: Optional[str] = None


@dataclass
class RecoveryConfig:
    """Configuration for error recovery"""
    max_connection_retries: int = 5
    max_audio_retries: int = 3
    max_api_retries: int = 3
    base_backoff_delay: float = 1.0
    max_backoff_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter_range: float = 0.1
    rate_limit_retry_delay: float = 60.0
    audio_quality_reduction_steps: List[str] = field(default_factory=lambda: ["16khz", "8khz", "4khz"])
    enable_graceful_degradation: bool = True
    enable_fallback_to_text: bool = True
    error_notification_threshold: int = 5
    error_aggregation_window: int = 300  # 5 minutes


@dataclass
class ErrorMetrics:
    """Error tracking metrics"""
    total_errors: int = 0
    errors_by_category: Dict[ErrorCategory, int] = field(default_factory=lambda: defaultdict(int))
    errors_by_severity: Dict[ErrorSeverity, int] = field(default_factory=lambda: defaultdict(int))
    recovery_attempts: int = 0
    recovery_successes: int = 0
    recovery_failures: int = 0
    degradation_events: int = 0
    fallback_events: int = 0
    notification_count: int = 0
    last_error_time: Optional[datetime] = None
    error_rate_per_minute: float = 0.0


class RealtimeErrorHandler:
    """
    Comprehensive error handler for OpenAI Realtime API
    
    Handles error classification, recovery strategies, graceful degradation,
    and comprehensive error reporting for production usage.
    """
    
    def __init__(self, config: RecoveryConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Error tracking
        self.recent_errors: deque = deque(maxlen=100)  # Recent error history
        self.error_metrics = ErrorMetrics()
        self.error_aggregation: Dict[str, List[ErrorInfo]] = defaultdict(list)
        
        # Recovery state
        self.current_recovery_attempts: Dict[str, int] = defaultdict(int)
        self.backoff_delays: Dict[str, float] = defaultdict(lambda: self.config.base_backoff_delay)
        self.degradation_state: Dict[str, Any] = {}
        
        # Rate limiting
        self.rate_limit_until: Optional[datetime] = None
        self.last_request_time: float = 0
        
        # Event callbacks
        self.on_error: Optional[Callable] = None
        self.on_recovery_success: Optional[Callable] = None
        self.on_recovery_failure: Optional[Callable] = None
        self.on_degradation: Optional[Callable] = None
        self.on_critical_error: Optional[Callable] = None
        
        # Error classification patterns
        self._setup_error_patterns()
    
    def _setup_error_patterns(self):
        """Setup error classification patterns"""
        self.error_patterns = {
            # Connection errors
            r"Connection.*refused": (ErrorCategory.CONNECTION_ERROR, ErrorSeverity.HIGH),
            r"WebSocket.*closed": (ErrorCategory.CONNECTION_ERROR, ErrorSeverity.MEDIUM),
            r"Connection.*timeout": (ErrorCategory.CONNECTION_ERROR, ErrorSeverity.HIGH),
            r"Network.*unreachable": (ErrorCategory.NETWORK_ERROR, ErrorSeverity.HIGH),
            
            # API errors
            r"API.*rate limit": (ErrorCategory.RATE_LIMIT_ERROR, ErrorSeverity.MEDIUM),
            r"Invalid.*API.*key": (ErrorCategory.AUTHENTICATION_ERROR, ErrorSeverity.CRITICAL),
            r"Unauthorized": (ErrorCategory.AUTHENTICATION_ERROR, ErrorSeverity.CRITICAL),
            r"Quota.*exceeded": (ErrorCategory.RATE_LIMIT_ERROR, ErrorSeverity.HIGH),
            r"Bad.*request": (ErrorCategory.API_ERROR, ErrorSeverity.MEDIUM),
            r"Internal.*server.*error": (ErrorCategory.API_ERROR, ErrorSeverity.HIGH),
            
            # Audio errors
            r"Audio.*stream.*failed": (ErrorCategory.AUDIO_ERROR, ErrorSeverity.MEDIUM),
            r"Microphone.*not.*available": (ErrorCategory.AUDIO_ERROR, ErrorSeverity.HIGH),
            r"Speaker.*not.*available": (ErrorCategory.AUDIO_ERROR, ErrorSeverity.HIGH),
            r"Audio.*format.*not.*supported": (ErrorCategory.AUDIO_ERROR, ErrorSeverity.MEDIUM),
            
            # Timeout errors
            r"Request.*timeout": (ErrorCategory.TIMEOUT_ERROR, ErrorSeverity.MEDIUM),
            r"Response.*timeout": (ErrorCategory.TIMEOUT_ERROR, ErrorSeverity.MEDIUM),
            r"Operation.*timeout": (ErrorCategory.TIMEOUT_ERROR, ErrorSeverity.MEDIUM),
            
            # Session errors
            r"Session.*expired": (ErrorCategory.SESSION_ERROR, ErrorSeverity.MEDIUM),
            r"Session.*not.*found": (ErrorCategory.SESSION_ERROR, ErrorSeverity.HIGH),
            r"Session.*invalid": (ErrorCategory.SESSION_ERROR, ErrorSeverity.HIGH),
        }
        
        # Recovery strategy mapping
        self.recovery_strategies = {
            ErrorCategory.CONNECTION_ERROR: RecoveryStrategy.RETRY_WITH_BACKOFF,
            ErrorCategory.API_ERROR: RecoveryStrategy.RETRY_WITH_BACKOFF,
            ErrorCategory.AUDIO_ERROR: RecoveryStrategy.RESTART_AUDIO,
            ErrorCategory.TIMEOUT_ERROR: RecoveryStrategy.RETRY_WITH_BACKOFF,
            ErrorCategory.RATE_LIMIT_ERROR: RecoveryStrategy.WAIT_AND_RETRY,
            ErrorCategory.AUTHENTICATION_ERROR: RecoveryStrategy.TERMINATE_SESSION,
            ErrorCategory.NETWORK_ERROR: RecoveryStrategy.RETRY_WITH_BACKOFF,
            ErrorCategory.SESSION_ERROR: RecoveryStrategy.RECONNECT,
            ErrorCategory.UNKNOWN_ERROR: RecoveryStrategy.RETRY_WITH_BACKOFF,
        }
    
    async def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorInfo:
        """
        Handle an error with comprehensive classification and recovery
        
        Args:
            error: The exception that occurred
            context: Additional context information
            
        Returns:
            ErrorInfo object with classification and recovery strategy
        """
        try:
            # Classify the error
            error_info = self._classify_error(error, context)
            
            # Record the error
            self._record_error(error_info)
            
            # Log the error
            await self._log_error(error_info)
            
            # Determine recovery strategy
            await self._determine_recovery_strategy(error_info)
            
            # Check for degradation needs
            await self._check_degradation_needs(error_info)
            
            # Execute recovery if appropriate
            if error_info.recovery_strategy != RecoveryStrategy.NO_RECOVERY:
                await self._execute_recovery(error_info)
            
            # Check for critical error notifications
            await self._check_critical_notifications(error_info)
            
            # Trigger callbacks
            if self.on_error:
                await self.on_error(error_info)
            
            return error_info
            
        except Exception as e:
            self.logger.error(f"❌ Error in error handler: {e}")
            # Return basic error info as fallback
            return ErrorInfo(
                category=ErrorCategory.UNKNOWN_ERROR,
                severity=ErrorSeverity.CRITICAL,
                message=str(error),
                details={"handler_error": str(e)},
                timestamp=datetime.now()
            )
    
    def _classify_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorInfo:
        """Classify error based on patterns and context"""
        error_message = str(error)
        error_type = type(error).__name__
        context = context or {}
        
        # Default classification
        category = ErrorCategory.UNKNOWN_ERROR
        severity = ErrorSeverity.MEDIUM
        
        # Pattern matching
        import re
        for pattern, (cat, sev) in self.error_patterns.items():
            if re.search(pattern, error_message, re.IGNORECASE):
                category = cat
                severity = sev
                break
        
        # Context-based classification
        if "session_id" in context and "session" in error_message.lower():
            category = ErrorCategory.SESSION_ERROR
        elif "audio" in error_message.lower() or "stream" in error_message.lower():
            category = ErrorCategory.AUDIO_ERROR
        elif "timeout" in error_message.lower():
            category = ErrorCategory.TIMEOUT_ERROR
        elif "connection" in error_message.lower() or "websocket" in error_message.lower():
            category = ErrorCategory.CONNECTION_ERROR
        
        # Determine user-friendly message
        user_message = self._get_user_friendly_message(category, error_message)
        
        return ErrorInfo(
            category=category,
            severity=severity,
            message=error_message,
            details={
                "error_type": error_type,
                "context": context,
                "traceback": traceback.format_exc()
            },
            timestamp=datetime.now(),
            session_id=context.get("session_id"),
            user_message=user_message,
            traceback_info=traceback.format_exc()
        )
    
    def _get_user_friendly_message(self, category: ErrorCategory, error_message: str) -> str:
        """Generate user-friendly error messages"""
        messages = {
            ErrorCategory.CONNECTION_ERROR: "Connection issue detected. Attempting to reconnect...",
            ErrorCategory.API_ERROR: "Service temporarily unavailable. Retrying request...",
            ErrorCategory.AUDIO_ERROR: "Audio system issue detected. Restarting audio...",
            ErrorCategory.TIMEOUT_ERROR: "Request timeout. Retrying with adjusted settings...",
            ErrorCategory.RATE_LIMIT_ERROR: "Service is busy. Please wait a moment...",
            ErrorCategory.AUTHENTICATION_ERROR: "Authentication issue. Please check your API key.",
            ErrorCategory.NETWORK_ERROR: "Network issue detected. Checking connection...",
            ErrorCategory.SESSION_ERROR: "Session issue detected. Reconnecting...",
            ErrorCategory.UNKNOWN_ERROR: "Unexpected issue occurred. Attempting recovery..."
        }
        
        return messages.get(category, "An issue occurred. Working to resolve it...")
    
    def _record_error(self, error_info: ErrorInfo):
        """Record error for metrics and aggregation"""
        # Add to recent errors
        self.recent_errors.append(error_info)
        
        # Update metrics
        self.error_metrics.total_errors += 1
        self.error_metrics.errors_by_category[error_info.category] += 1
        self.error_metrics.errors_by_severity[error_info.severity] += 1
        self.error_metrics.last_error_time = error_info.timestamp
        
        # Calculate error rate
        now = datetime.now()
        recent_errors_count = sum(
            1 for err in self.recent_errors 
            if (now - err.timestamp).total_seconds() <= 60
        )
        self.error_metrics.error_rate_per_minute = recent_errors_count
        
        # Add to aggregation
        aggregation_key = f"{error_info.category.value}_{error_info.severity.value}"
        self.error_aggregation[aggregation_key].append(error_info)
        
        # Clean old aggregation data
        cutoff_time = now - timedelta(seconds=self.config.error_aggregation_window)
        for key in list(self.error_aggregation.keys()):
            self.error_aggregation[key] = [
                err for err in self.error_aggregation[key] 
                if err.timestamp > cutoff_time
            ]
            if not self.error_aggregation[key]:
                del self.error_aggregation[key]
    
    async def _log_error(self, error_info: ErrorInfo):
        """Log error with structured logging"""
        log_message = f"{error_info.category.value}: {error_info.message}"
        
        # Prepare safe log data (avoid 'message' key conflict)
        log_data = {
            "error_type": error_info.category.value,
            "severity": error_info.severity.value,
            "session_id": error_info.session_id,
            "timestamp": error_info.timestamp.isoformat(),
            "error_details": error_info.details,
            "recovery_strategy": error_info.recovery_strategy.value if error_info.recovery_strategy else None
        }
        
        if error_info.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.error(f"❌ {log_message}", extra=log_data)
        else:
            self.logger.warning(f"⚠️ {log_message}", extra=log_data)
    
    async def _determine_recovery_strategy(self, error_info: ErrorInfo):
        """Determine the appropriate recovery strategy"""
        base_strategy = self.recovery_strategies.get(error_info.category, RecoveryStrategy.NO_RECOVERY)
        
        # Adjust strategy based on retry count and context
        retry_key = f"{error_info.category.value}_{error_info.session_id}"
        current_retries = self.current_recovery_attempts[retry_key]
        
        # Check if we've exceeded max retries
        max_retries = self._get_max_retries(error_info.category)
        if current_retries >= max_retries:
            if self.config.enable_graceful_degradation:
                error_info.recovery_strategy = RecoveryStrategy.REDUCE_QUALITY
            elif self.config.enable_fallback_to_text:
                error_info.recovery_strategy = RecoveryStrategy.FALLBACK_TO_TEXT
            else:
                error_info.recovery_strategy = RecoveryStrategy.TERMINATE_SESSION
        else:
            error_info.recovery_strategy = base_strategy
        
        # Set retry count and backoff delay
        error_info.retry_count = current_retries
        error_info.max_retries = max_retries
        error_info.backoff_delay = self._calculate_backoff_delay(retry_key, current_retries)
    
    def _get_max_retries(self, category: ErrorCategory) -> int:
        """Get max retries for error category"""
        retry_limits = {
            ErrorCategory.CONNECTION_ERROR: self.config.max_connection_retries,
            ErrorCategory.API_ERROR: self.config.max_api_retries,
            ErrorCategory.AUDIO_ERROR: self.config.max_audio_retries,
            ErrorCategory.TIMEOUT_ERROR: self.config.max_api_retries,
            ErrorCategory.RATE_LIMIT_ERROR: 1,  # Wait, don't retry immediately
            ErrorCategory.AUTHENTICATION_ERROR: 0,  # Don't retry auth errors
            ErrorCategory.NETWORK_ERROR: self.config.max_connection_retries,
            ErrorCategory.SESSION_ERROR: self.config.max_connection_retries,
            ErrorCategory.UNKNOWN_ERROR: self.config.max_api_retries,
        }
        return retry_limits.get(category, 3)
    
    def _calculate_backoff_delay(self, retry_key: str, retry_count: int) -> float:
        """Calculate exponential backoff delay with jitter"""
        base_delay = self.config.base_backoff_delay
        multiplier = self.config.backoff_multiplier
        max_delay = self.config.max_backoff_delay
        jitter_range = self.config.jitter_range
        
        # Exponential backoff
        delay = base_delay * (multiplier ** retry_count)
        delay = min(delay, max_delay)
        
        # Add jitter to prevent thundering herd
        jitter = random.uniform(-jitter_range, jitter_range) * delay
        delay += jitter
        
        # Ensure minimum delay
        delay = max(delay, 0.1)
        
        self.backoff_delays[retry_key] = delay
        return delay
    
    async def _check_degradation_needs(self, error_info: ErrorInfo):
        """Check if graceful degradation is needed"""
        if not self.config.enable_graceful_degradation:
            return
        
        # Check error frequency
        error_count = self.error_metrics.errors_by_category[error_info.category]
        
        # Degrade audio quality if too many audio errors
        if (error_info.category == ErrorCategory.AUDIO_ERROR and 
            error_count >= 3 and 
            "audio_quality" not in self.degradation_state):
            
            self.degradation_state["audio_quality"] = "degraded"
            self.error_metrics.degradation_events += 1
            
            if self.on_degradation:
                await self.on_degradation("audio_quality", "Reducing audio quality due to repeated errors")
        
        # Fallback to text mode if connection is unstable
        if (error_info.category in [ErrorCategory.CONNECTION_ERROR, ErrorCategory.NETWORK_ERROR] and
            error_count >= 5 and
            "text_mode" not in self.degradation_state):
            
            self.degradation_state["text_mode"] = "active"
            self.error_metrics.fallback_events += 1
            
            if self.on_degradation:
                await self.on_degradation("text_mode", "Switching to text mode due to connection issues")
    
    async def _execute_recovery(self, error_info: ErrorInfo):
        """Execute the determined recovery strategy"""
        try:
            retry_key = f"{error_info.category.value}_{error_info.session_id}"
            
            if error_info.recovery_strategy == RecoveryStrategy.RETRY_WITH_BACKOFF:
                await self._retry_with_backoff(error_info, retry_key)
                
            elif error_info.recovery_strategy == RecoveryStrategy.WAIT_AND_RETRY:
                await self._wait_and_retry(error_info)
                
            elif error_info.recovery_strategy == RecoveryStrategy.RESTART_AUDIO:
                await self._restart_audio(error_info)
                
            elif error_info.recovery_strategy == RecoveryStrategy.REDUCE_QUALITY:
                await self._reduce_quality(error_info)
                
            elif error_info.recovery_strategy == RecoveryStrategy.FALLBACK_TO_TEXT:
                await self._fallback_to_text(error_info)
                
            # Record recovery attempt
            self.current_recovery_attempts[retry_key] += 1
            self.error_metrics.recovery_attempts += 1
            
            error_info.recovery_action = f"Executed {error_info.recovery_strategy.value}"
            
        except Exception as e:
            self.logger.error(f"❌ Recovery execution failed: {e}")
            self.error_metrics.recovery_failures += 1
            
            if self.on_recovery_failure:
                await self.on_recovery_failure(error_info, str(e))
    
    async def _retry_with_backoff(self, error_info: ErrorInfo, retry_key: str):
        """Implement retry with exponential backoff"""
        delay = error_info.backoff_delay
        
        self.logger.info(f"🔄 Retrying in {delay:.1f}s (attempt {error_info.retry_count + 1}/{error_info.max_retries})")
        
        # Wait for backoff delay
        await asyncio.sleep(delay)
        
        # The actual retry logic should be implemented by the calling component
        # This just handles the timing and logging
    
    async def _wait_and_retry(self, error_info: ErrorInfo):
        """Wait for rate limit and retry"""
        delay = self.config.rate_limit_retry_delay
        
        # Check if we have a specific retry-after value
        if "retry_after" in error_info.details:
            delay = float(error_info.details["retry_after"])
        
        self.rate_limit_until = datetime.now() + timedelta(seconds=delay)
        
        self.logger.info(f"⏳ Rate limited, waiting {delay:.1f}s before retry")
        await asyncio.sleep(delay)
    
    async def _restart_audio(self, error_info: ErrorInfo):
        """Restart audio streams"""
        self.logger.info("🎵 Restarting audio streams due to audio error")
        # Audio restart logic should be implemented by the audio manager
        # This is a placeholder for the recovery action
        error_info.recovery_action = "Audio restart initiated"
    
    async def _reduce_quality(self, error_info: ErrorInfo):
        """Reduce audio quality for better stability"""
        current_quality = self.degradation_state.get("audio_quality", "24khz")
        quality_steps = self.config.audio_quality_reduction_steps
        
        try:
            current_index = quality_steps.index(current_quality)
            if current_index < len(quality_steps) - 1:
                new_quality = quality_steps[current_index + 1]
                self.degradation_state["audio_quality"] = new_quality
                
                self.logger.info(f"📉 Reducing audio quality to {new_quality}")
                error_info.recovery_action = f"Audio quality reduced to {new_quality}"
        except ValueError:
            # Current quality not in steps, use first reduction step
            self.degradation_state["audio_quality"] = quality_steps[0]
            error_info.recovery_action = f"Audio quality reduced to {quality_steps[0]}"
    
    async def _fallback_to_text(self, error_info: ErrorInfo):
        """Fallback to text-only mode"""
        self.degradation_state["text_mode"] = "active"
        self.logger.info("💬 Falling back to text-only mode")
        error_info.recovery_action = "Switched to text-only mode"
    
    async def _check_critical_notifications(self, error_info: ErrorInfo):
        """Check if critical error notifications should be sent"""
        if error_info.severity == ErrorSeverity.CRITICAL:
            self.error_metrics.notification_count += 1
            
            if self.on_critical_error:
                await self.on_critical_error(error_info)
        
        # Check if error rate exceeds threshold
        if self.error_metrics.error_rate_per_minute >= self.config.error_notification_threshold:
            if self.on_critical_error:
                await self.on_critical_error(ErrorInfo(
                    category=ErrorCategory.UNKNOWN_ERROR,
                    severity=ErrorSeverity.CRITICAL,
                    message=f"High error rate detected: {self.error_metrics.error_rate_per_minute} errors/minute",
                    details={"error_rate": self.error_metrics.error_rate_per_minute},
                    timestamp=datetime.now(),
                    user_message="System experiencing high error rate. Please try again later."
                ))
    
    def is_rate_limited(self) -> bool:
        """Check if currently rate limited"""
        if self.rate_limit_until is None:
            return False
        return datetime.now() < self.rate_limit_until
    
    def get_error_metrics(self) -> Dict[str, Any]:
        """Get comprehensive error metrics"""
        return {
            "total_errors": self.error_metrics.total_errors,
            "errors_by_category": {k.value: v for k, v in self.error_metrics.errors_by_category.items()},
            "errors_by_severity": {k.value: v for k, v in self.error_metrics.errors_by_severity.items()},
            "recovery_attempts": self.error_metrics.recovery_attempts,
            "recovery_successes": self.error_metrics.recovery_successes,
            "recovery_failures": self.error_metrics.recovery_failures,
            "recovery_success_rate": (
                self.error_metrics.recovery_successes / self.error_metrics.recovery_attempts
                if self.error_metrics.recovery_attempts > 0 else 0
            ),
            "degradation_events": self.error_metrics.degradation_events,
            "fallback_events": self.error_metrics.fallback_events,
            "error_rate_per_minute": self.error_metrics.error_rate_per_minute,
            "last_error_time": self.error_metrics.last_error_time.isoformat() if self.error_metrics.last_error_time else None,
            "current_degradation_state": self.degradation_state,
            "rate_limited_until": self.rate_limit_until.isoformat() if self.rate_limit_until else None
        }
    
    def get_error_report(self) -> Dict[str, Any]:
        """Generate comprehensive error report"""
        return {
            "metrics": self.get_error_metrics(),
            "recent_errors": [
                {
                    "category": err.category.value,
                    "severity": err.severity.value,
                    "message": err.message,
                    "timestamp": err.timestamp.isoformat(),
                    "session_id": err.session_id,
                    "recovery_action": err.recovery_action
                }
                for err in list(self.recent_errors)[-10:]  # Last 10 errors
            ],
            "error_aggregation": {
                key: len(errors) for key, errors in self.error_aggregation.items()
            },
            "current_state": {
                "degradation_active": bool(self.degradation_state),
                "rate_limited": self.is_rate_limited(),
                "active_recovery_attempts": dict(self.current_recovery_attempts)
            }
        }
    
    async def reset_error_state(self, category: Optional[ErrorCategory] = None):
        """Reset error state for recovery"""
        if category:
            # Reset specific category
            retry_keys_to_remove = [
                key for key in self.current_recovery_attempts.keys()
                if key.startswith(category.value)
            ]
            for key in retry_keys_to_remove:
                del self.current_recovery_attempts[key]
                if key in self.backoff_delays:
                    del self.backoff_delays[key]
        else:
            # Reset all error state
            self.current_recovery_attempts.clear()
            self.backoff_delays.clear()
            self.degradation_state.clear()
            self.rate_limit_until = None
        
        self.logger.info(f"🔄 Reset error state for {category.value if category else 'all categories'}")


def create_error_handler(config: Optional[RecoveryConfig] = None) -> RealtimeErrorHandler:
    """Factory function to create RealtimeErrorHandler"""
    if config is None:
        config = RecoveryConfig()
    
    return RealtimeErrorHandler(config) 