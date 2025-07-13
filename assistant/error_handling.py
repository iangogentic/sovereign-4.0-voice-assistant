"""
Advanced Error Handling and Retry Logic System for Sovereign Voice Assistant
Based on 2024-2025 best practices for Python async error handling
"""

import asyncio
import logging
import time
import random
import math
import uuid
from enum import Enum
from typing import Dict, Any, Optional, Callable, Awaitable, List, Union
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from contextvars import ContextVar
import json

# Context variables for request tracking
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
user_session_var: ContextVar[str] = ContextVar('user_session', default='')
operation_var: ContextVar[str] = ContextVar('operation', default='')

class ErrorCategory(Enum):
    """Structured error categories for precise handling"""
    TRANSIENT = "transient"      # Temporary failures, retry recommended
    PERMANENT = "permanent"      # Permanent failures, don't retry
    RATE_LIMIT = "rate_limit"    # Rate limiting, backoff required
    TIMEOUT = "timeout"          # Timeout errors, may retry with longer timeout
    AUTHENTICATION = "authentication"  # Auth errors, check credentials
    QUOTA = "quota"              # Quota exceeded, wait or upgrade
    NETWORK = "network"          # Network connectivity issues
    AUDIO = "audio"              # Audio processing errors
    MODEL = "model"              # AI model errors

@dataclass
class ErrorContext:
    """Rich error context for better debugging and handling"""
    service_name: str
    operation: str
    timestamp: float
    request_id: Optional[str] = None
    user_context: Optional[Dict[str, Any]] = None
    system_state: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None
    audio_duration_ms: Optional[float] = None
    model_used: Optional[str] = None
    retry_count: int = 0

class VoiceAIException(Exception):
    """Base exception for voice AI operations"""
    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        context: Optional[ErrorContext] = None,
        retryable: bool = True,
        original_exception: Optional[Exception] = None,
        user_message: Optional[str] = None
    ):
        super().__init__(message)
        self.category = category
        self.context = context
        self.retryable = retryable
        self.original_exception = original_exception
        self.user_message = user_message or self._generate_user_message()
        self.timestamp = time.time()
    
    def _generate_user_message(self) -> str:
        """Generate user-friendly error message"""
        if self.category == ErrorCategory.TRANSIENT:
            return "Network connection issue, retrying..."
        elif self.category == ErrorCategory.NETWORK:
            return "Network connection issue. Switching to offline mode..."
        elif self.category == ErrorCategory.RATE_LIMIT:
            return "Service is busy. Please wait a moment..."
        elif self.category == ErrorCategory.TIMEOUT:
            return "Request taking longer than expected. Trying again..."
        elif self.category == ErrorCategory.AUDIO:
            return "Audio processing issue. Please try speaking again..."
        elif self.category == ErrorCategory.MODEL:
            return "AI model temporarily unavailable. Using backup..."
        else:
            return "Technical issue encountered. Trying alternative approach..."

class STTException(VoiceAIException):
    """Speech-to-text specific exceptions"""
    pass

class LLMException(VoiceAIException):
    """LLM processing specific exceptions"""
    pass

class TTSException(VoiceAIException):
    """Text-to-speech specific exceptions"""
    pass

class AudioException(VoiceAIException):
    """Audio processing specific exceptions"""
    pass

class CircuitBreakerOpenError(VoiceAIException):
    """Raised when circuit breaker is open"""
    def __init__(self, service_name: str, health_score: float):
        message = f"Circuit breaker open for {service_name} (health: {health_score:.2f})"
        super().__init__(
            message,
            ErrorCategory.TRANSIENT,
            retryable=False,
            user_message=f"Service {service_name} is temporarily unavailable. Using backup..."
        )
        self.service_name = service_name
        self.health_score = health_score

class AllTiersFailedError(VoiceAIException):
    """Raised when all service tiers have failed"""
    def __init__(self, operation: str, attempted_tiers: List[str], last_exception: Exception):
        message = f"All tiers failed for {operation}. Attempted: {attempted_tiers}"
        super().__init__(
            message,
            ErrorCategory.PERMANENT,
            retryable=False,
            original_exception=last_exception,
            user_message="All services are currently unavailable. Please try again later."
        )
        self.operation = operation
        self.attempted_tiers = attempted_tiers

# Advanced Exponential Backoff Configuration
@dataclass
class BackoffConfig:
    """Advanced backoff configuration"""
    base_delay: float = 1.0
    max_delay: float = 300.0  # 5 minutes max
    exponential_base: float = 2.0
    jitter_type: str = "full"  # "none", "equal", "full", "decorrelated"
    max_attempts: int = 3
    timeout_multiplier: float = 1.5  # Increase timeout on each retry
    backoff_multiplier: float = 1.0  # Additional multiplier for specific services

class AdvancedBackoff:
    """2024 best practice exponential backoff with multiple jitter strategies"""
    
    def __init__(self, config: BackoffConfig):
        self.config = config
        self.attempt_count = 0
        self.last_delay = 0.0
    
    def calculate_delay(self, attempt: int, context: Optional[ErrorContext] = None) -> float:
        """Calculate delay with advanced jitter strategies"""
        # Base exponential delay
        base_delay = self.config.base_delay * (
            self.config.exponential_base ** attempt
        ) * self.config.backoff_multiplier
        
        # Apply maximum delay limit
        base_delay = min(base_delay, self.config.max_delay)
        
        # Apply jitter based on strategy
        if self.config.jitter_type == "none":
            return base_delay
        elif self.config.jitter_type == "equal":
            # Equal jitter: delay/2 + random(0, delay/2)
            return base_delay / 2 + random.uniform(0, base_delay / 2)
        elif self.config.jitter_type == "full":
            # Full jitter: random(0, delay)
            return random.uniform(0, base_delay)
        elif self.config.jitter_type == "decorrelated":
            # Decorrelated jitter: random(base_delay, last_delay * 3)
            if self.last_delay == 0:
                delay = random.uniform(0, base_delay)
            else:
                delay = random.uniform(self.config.base_delay, self.last_delay * 3)
            self.last_delay = delay
            return min(delay, self.config.max_delay)
        
        return base_delay
    
    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """Determine if retry should be attempted"""
        if attempt >= self.config.max_attempts:
            return False
        
        # Check if exception is retryable
        if isinstance(exception, VoiceAIException):
            return exception.retryable
        
        # Default retry logic for unknown exceptions
        return True
    
    def get_timeout(self, attempt: int, base_timeout: float) -> float:
        """Calculate timeout for retry attempt"""
        return base_timeout * (self.config.timeout_multiplier ** attempt)

async def retry_with_backoff(
    func: Callable[[], Awaitable[Any]],
    config: BackoffConfig,
    context: Optional[ErrorContext] = None,
    status_callback: Optional[Callable[[str, int], Awaitable[None]]] = None
) -> Any:
    """Execute function with advanced retry and backoff logic"""
    backoff = AdvancedBackoff(config)
    last_exception = None
    
    for attempt in range(config.max_attempts):
        try:
            if attempt > 0:
                delay = backoff.calculate_delay(attempt - 1, context)
                
                if status_callback:
                    await status_callback(f"Retrying in {delay:.1f}s...", attempt)
                
                await asyncio.sleep(delay)
            
            # Execute with adaptive timeout
            base_timeout = getattr(func, '_timeout', 30.0)
            timeout = backoff.get_timeout(attempt, base_timeout)
            
            result = await asyncio.wait_for(func(), timeout=timeout)
            
            if attempt > 0:
                logging.info(f"Retry succeeded on attempt {attempt + 1}")
            
            return result
            
        except asyncio.TimeoutError as e:
            last_exception = VoiceAIException(
                f"Operation timed out after {timeout:.1f}s",
                ErrorCategory.TIMEOUT,
                context,
                retryable=True,
                original_exception=e
            )
            
            if not backoff.should_retry(attempt + 1, last_exception):
                break
            
            logging.warning(f"Attempt {attempt + 1} timed out, retrying...")
            
        except Exception as e:
            last_exception = e
            
            if not backoff.should_retry(attempt + 1, e):
                break
            
            # Log retry attempt
            logging.warning(
                f"Attempt {attempt + 1} failed: {str(e)}, "
                f"retrying in {backoff.calculate_delay(attempt, context):.1f}s"
            )
    
    # All retries exhausted
    raise last_exception

# Circuit Breaker Implementation
class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class HealthMetrics:
    """Comprehensive health metrics for circuit breaker"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    last_success_time: float = 0.0
    last_failure_time: float = 0.0
    response_times: List[float] = field(default_factory=list)
    
    @property
    def failure_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    @property
    def success_rate(self) -> float:
        return 1.0 - self.failure_rate
    
    @property
    def health_score(self) -> float:
        """Calculate overall health score (0.0 to 1.0)"""
        if self.total_requests == 0:
            return 1.0
        
        # Weighted health score considering multiple factors
        success_weight = 0.4
        latency_weight = 0.3
        consistency_weight = 0.3
        
        success_score = self.success_rate
        
        # Latency score (inverse of normalized response time)
        if self.avg_response_time > 0:
            latency_score = max(0, 1.0 - (self.avg_response_time / 10.0))  # 10s baseline
        else:
            latency_score = 1.0
        
        # Consistency score (inverse of consecutive failures)
        consistency_score = max(0, 1.0 - (self.consecutive_failures / 10.0))
        
        return (
            success_score * success_weight +
            latency_score * latency_weight +
            consistency_score * consistency_weight
        )

@dataclass
class CircuitBreakerConfig:
    """Advanced circuit breaker configuration"""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_duration: float = 60.0
    failure_rate_threshold: float = 0.5
    min_request_threshold: int = 10
    slow_call_threshold: float = 5.0  # Seconds
    slow_call_rate_threshold: float = 0.3
    health_score_threshold: float = 0.3
    adaptive_timeout: bool = True
    predictive_opening: bool = True

class ModernCircuitBreaker:
    """2024 state-of-the-art circuit breaker with predictive capabilities"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.metrics = HealthMetrics()
        self.state_change_time = time.time()
        self.half_open_test_count = 0
        
        # Predictive failure detection
        self.failure_trend = []
        self.response_time_trend = []
        
        # Setup logging
        self.logger = logging.getLogger(f"circuit_breaker.{name}")
        
    async def call(self, func: Callable[[], Awaitable[Any]]) -> Any:
        """Execute function with circuit breaker protection"""
        if not self._can_execute():
            raise CircuitBreakerOpenError(self.name, self.metrics.health_score)
        
        start_time = time.time()
        
        try:
            result = await func()
            response_time = time.time() - start_time
            self._record_success(response_time)
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            self._record_failure(response_time, e)
            raise
    
    def _can_execute(self) -> bool:
        """Check if circuit breaker allows execution"""
        current_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if current_time - self.state_change_time >= self._get_adaptive_timeout():
                self._transition_to_half_open()
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return True
        
        return False
    
    def _record_success(self, response_time: float):
        """Record successful execution"""
        self.metrics.total_requests += 1
        self.metrics.successful_requests += 1
        self.metrics.consecutive_successes += 1
        self.metrics.consecutive_failures = 0
        self.metrics.last_success_time = time.time()
        
        # Update response time metrics
        self._update_response_times(response_time)
        
        # Update trends for predictive analysis
        self.failure_trend.append(0)
        self.response_time_trend.append(response_time)
        self._trim_trends()
        
        # State transitions
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_test_count += 1
            if self.half_open_test_count >= self.config.success_threshold:
                self._transition_to_closed()
    
    def _record_failure(self, response_time: float, exception: Exception):
        """Record failed execution"""
        self.metrics.total_requests += 1
        self.metrics.failed_requests += 1
        self.metrics.consecutive_failures += 1
        self.metrics.consecutive_successes = 0
        self.metrics.last_failure_time = time.time()
        
        if isinstance(exception, asyncio.TimeoutError):
            self.metrics.timeout_requests += 1
        
        # Update response time metrics
        self._update_response_times(response_time)
        
        # Update trends
        self.failure_trend.append(1)
        self.response_time_trend.append(response_time)
        self._trim_trends()
        
        # Check if circuit should open
        if self.state == CircuitState.CLOSED:
            if self._should_open_circuit():
                self._transition_to_open()
        elif self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()
    
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should open based on multiple criteria"""
        # Traditional failure threshold (primary condition)
        if self.metrics.consecutive_failures >= self.config.failure_threshold:
            return True
        
        # For low request counts, only use consecutive failures to avoid premature opening
        if self.metrics.total_requests < max(self.config.min_request_threshold, self.config.failure_threshold):
            return False
        
        # Failure rate threshold (only for higher request counts)
        if (self.metrics.total_requests >= self.config.min_request_threshold and
            self.metrics.failure_rate >= self.config.failure_rate_threshold):
            return True
        
        # Slow call rate threshold
        slow_calls = sum(1 for rt in self.metrics.response_times[-50:] 
                        if rt > self.config.slow_call_threshold)
        if (len(self.metrics.response_times) >= self.config.min_request_threshold and
            slow_calls / len(self.metrics.response_times[-50:]) >= self.config.slow_call_rate_threshold):
            return True
        
        # Health score threshold (only for significant request counts)
        if (self.metrics.total_requests >= self.config.min_request_threshold and
            self.metrics.health_score < self.config.health_score_threshold):
            return True
        
        # Predictive opening based on trends (only for significant request counts)
        if (self.config.predictive_opening and 
            self.metrics.total_requests >= self.config.min_request_threshold and
            self._predict_failure()):
            return True
        
        return False
    
    def _predict_failure(self) -> bool:
        """Predictive failure detection based on trends"""
        if len(self.failure_trend) < 10:
            return False
        
        # Analyze failure trend
        recent_failures = sum(self.failure_trend[-10:])
        if recent_failures >= 7:  # 70% failure rate in last 10 requests
            return True
        
        # Analyze response time trend
        if len(self.response_time_trend) >= 10:
            recent_avg = sum(self.response_time_trend[-5:]) / 5
            older_avg = sum(self.response_time_trend[-10:-5]) / 5
            
            # If response time is increasing significantly
            if recent_avg > older_avg * 2 and recent_avg > self.config.slow_call_threshold:
                return True
        
        return False
    
    def _get_adaptive_timeout(self) -> float:
        """Calculate adaptive timeout based on failure patterns"""
        if not self.config.adaptive_timeout:
            return self.config.timeout_duration
        
        # Increase timeout based on consecutive failures
        multiplier = min(2.0, 1.0 + (self.metrics.consecutive_failures * 0.2))
        return self.config.timeout_duration * multiplier
    
    def _update_response_times(self, response_time: float):
        """Update response time metrics"""
        self.metrics.response_times.append(response_time)
        
        # Keep only last 100 response times
        if len(self.metrics.response_times) > 100:
            self.metrics.response_times = self.metrics.response_times[-100:]
        
        # Update average
        self.metrics.avg_response_time = sum(self.metrics.response_times) / len(self.metrics.response_times)
        
        # Update P95
        if len(self.metrics.response_times) >= 20:
            sorted_times = sorted(self.metrics.response_times)
            p95_index = int(len(sorted_times) * 0.95)
            self.metrics.p95_response_time = sorted_times[p95_index]
    
    def _trim_trends(self):
        """Keep trend data manageable"""
        max_trend_size = 50
        if len(self.failure_trend) > max_trend_size:
            self.failure_trend = self.failure_trend[-max_trend_size:]
        if len(self.response_time_trend) > max_trend_size:
            self.response_time_trend = self.response_time_trend[-max_trend_size:]
    
    def _transition_to_open(self):
        """Transition to OPEN state"""
        self.state = CircuitState.OPEN
        self.state_change_time = time.time()
        self.half_open_test_count = 0
        self.logger.warning(f"Circuit breaker {self.name} opened (health: {self.metrics.health_score:.2f})")
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state"""
        self.state = CircuitState.HALF_OPEN
        self.state_change_time = time.time()
        self.half_open_test_count = 0
        self.logger.info(f"Circuit breaker {self.name} half-open (testing recovery)")
    
    def _transition_to_closed(self):
        """Transition to CLOSED state"""
        self.state = CircuitState.CLOSED
        self.state_change_time = time.time()
        self.logger.info(f"Circuit breaker {self.name} closed (recovered)")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        return {
            "name": self.name,
            "state": self.state.value,
            "health_score": self.metrics.health_score,
            "total_requests": self.metrics.total_requests,
            "failure_rate": self.metrics.failure_rate,
            "avg_response_time": self.metrics.avg_response_time,
            "consecutive_failures": self.metrics.consecutive_failures,
            "last_state_change": self.state_change_time
        }

# Service Tier Management for Graceful Degradation
class ServiceTier(Enum):
    PRIMARY = "primary"      # Best quality, highest latency
    FAST = "fast"           # Good quality, low latency  
    BACKUP = "backup"       # Acceptable quality, reliable
    OFFLINE = "offline"     # Basic functionality, no network
    CACHED = "cached"       # Cached responses, instant

@dataclass
class ServiceCapability:
    """Defines what a service can do and its characteristics"""
    name: str
    tier: ServiceTier
    latency_target: float  # Target response time in seconds
    quality_score: float   # Quality rating 0.0-1.0
    reliability_score: float  # Reliability rating 0.0-1.0
    cost_per_request: float  # Relative cost
    supports_streaming: bool = False
    max_concurrent: int = 10

class GracefulDegradationManager:
    """Manages graceful degradation across service tiers"""
    
    def __init__(self):
        self.service_registry: Dict[str, Dict[ServiceTier, ServiceCapability]] = {}
        self.circuit_breakers: Dict[str, ModernCircuitBreaker] = {}
        self.fallback_chains: Dict[str, List[ServiceTier]] = {}
        self.logger = logging.getLogger("graceful_degradation")
        
        # Default fallback chains for different operations
        self.fallback_chains = {
            'stt': [ServiceTier.PRIMARY, ServiceTier.FAST, ServiceTier.OFFLINE],
            'llm': [ServiceTier.PRIMARY, ServiceTier.FAST, ServiceTier.BACKUP, ServiceTier.OFFLINE, ServiceTier.CACHED],
            'tts': [ServiceTier.PRIMARY, ServiceTier.FAST, ServiceTier.OFFLINE],
            'audio_processing': [ServiceTier.PRIMARY, ServiceTier.BACKUP, ServiceTier.OFFLINE],
            'test_operation': [ServiceTier.PRIMARY, ServiceTier.BACKUP, ServiceTier.OFFLINE]  # Default for tests
        }
    
    def register_service(
        self,
        operation: str,
        tier: ServiceTier,
        capability: ServiceCapability,
        circuit_breaker: Optional[ModernCircuitBreaker] = None
    ):
        """Register a service capability"""
        if operation not in self.service_registry:
            self.service_registry[operation] = {}
        
        self.service_registry[operation][tier] = capability
        
        if circuit_breaker:
            self.circuit_breakers[f"{operation}_{tier.value}"] = circuit_breaker
    
    async def execute_with_degradation(
        self,
        operation: str,
        service_functions: Dict[ServiceTier, Callable[[], Awaitable[Any]]],
        context: Optional[Dict[str, Any]] = None,
        user_feedback_callback: Optional[Callable[[str], Awaitable[None]]] = None
    ) -> Any:
        """Execute operation with graceful degradation"""
        
        fallback_chain = self.fallback_chains.get(operation, [ServiceTier.PRIMARY])
        last_exception = None
        attempted_tiers = []
        
        for tier in fallback_chain:
            if tier not in service_functions:
                continue
            
            attempted_tiers.append(tier)
            service_key = f"{operation}_{tier.value}"
            circuit_breaker = self.circuit_breakers.get(service_key)
            
            # Check if service is available
            if circuit_breaker and circuit_breaker.state == CircuitState.OPEN:
                self.logger.info(f"Skipping {tier.value} tier for {operation} (circuit breaker open)")
                continue
            
            try:
                # Provide user feedback for degradation
                if len(attempted_tiers) > 1 and user_feedback_callback:
                    await self._provide_degradation_feedback(
                        operation, tier, user_feedback_callback
                    )
                
                # Execute with circuit breaker protection
                if circuit_breaker:
                    result = await circuit_breaker.call(service_functions[tier])
                else:
                    result = await service_functions[tier]()
                
                # Log successful degradation
                if len(attempted_tiers) > 1:
                    self.logger.info(f"Successfully degraded {operation} to {tier.value} tier")
                
                return result
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"{tier.value} tier failed for {operation}: {str(e)}")
                continue
        
        # All tiers failed
        error_msg = f"All tiers failed for {operation}. Attempted: {[t.value for t in attempted_tiers]}"
        self.logger.error(error_msg)
        raise AllTiersFailedError(operation, [t.value for t in attempted_tiers], last_exception)
    
    async def _provide_degradation_feedback(
        self,
        operation: str,
        tier: ServiceTier,
        callback: Callable[[str], Awaitable[None]]
    ):
        """Provide user-friendly feedback about service degradation"""
        messages = {
            ServiceTier.FAST: {
                'stt': "Switching to faster speech recognition...",
                'llm': "Using quick response mode...",
                'tts': "Switching to faster voice synthesis..."
            },
            ServiceTier.BACKUP: {
                'stt': "Using backup speech recognition...",
                'llm': "Switching to backup AI model...",
                'tts': "Using backup voice synthesis..."
            },
            ServiceTier.OFFLINE: {
                'stt': "Switching to offline speech recognition...",
                'llm': "Switching to offline AI mode...",
                'tts': "Using offline voice synthesis..."
            },
            ServiceTier.CACHED: {
                'llm': "Using cached response...",
                'code_analysis': "Using cached analysis..."
            }
        }
        
        message = messages.get(tier, {}).get(operation, f"Switching to {tier.value} mode...")
        await callback(message)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current degradation manager status"""
        circuit_breaker_status = {}
        for name, cb in self.circuit_breakers.items():
            circuit_breaker_status[name] = cb.get_metrics()
        
        return {
            "registered_services": list(self.service_registry.keys()),
            "fallback_chains": self.fallback_chains,
            "circuit_breakers": circuit_breaker_status
        } 