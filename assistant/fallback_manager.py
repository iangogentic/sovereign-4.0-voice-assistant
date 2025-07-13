"""
Comprehensive Fallback and Error Handling System

This module provides enterprise-grade fallback logic and error handling including:
- Multi-tier model fallback hierarchy
- Circuit breaker pattern for service health monitoring
- Retry strategies with exponential backoff
- User-friendly error message system
- Service availability tracking
"""

import asyncio
import logging
import time
import random
import math
from typing import Dict, Any, Optional, List, Callable, Awaitable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ServiceState(Enum):
    """Circuit breaker service states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Service disabled due to failures
    HALF_OPEN = "half_open"  # Testing if service has recovered


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class RetryStrategy(Enum):
    """Retry strategy types"""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    CUSTOM = "custom"
    NONE = "none"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5  # Failures before opening circuit
    success_threshold: int = 2  # Successes to close circuit from half-open
    timeout_seconds: float = 60.0  # Time before trying half-open
    failure_rate_threshold: float = 0.5  # Minimum failure rate to open (0.0-1.0)
    min_request_threshold: int = 10  # Minimum requests before calculating failure rate


@dataclass
class RetryConfig:
    """Configuration for retry strategies"""
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    max_attempts: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    exponential_base: float = 2.0  # Base for exponential backoff
    jitter: bool = True  # Add randomness to prevent thundering herd
    timeout_multiplier: float = 1.5  # Multiply timeout on each retry


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior"""
    enable_fallback: bool = True
    fallback_chain: List[str] = field(default_factory=lambda: ['fast', 'deep', 'backup'])
    max_fallback_attempts: int = 3
    fallback_timeout_factor: float = 0.8  # Reduce timeout for fallback attempts
    preserve_context: bool = True  # Preserve conversation context in fallbacks


@dataclass
class ServiceMetrics:
    """Metrics for service monitoring"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    last_request_time: float = 0.0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    avg_response_time: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate"""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        return 1.0 - self.failure_rate


class CircuitBreaker:
    """
    Circuit breaker implementation for service health monitoring
    """
    
    def __init__(self, service_name: str, config: Optional[CircuitBreakerConfig] = None):
        self.service_name = service_name
        self.config = config or CircuitBreakerConfig()
        self.state = ServiceState.CLOSED
        self.metrics = ServiceMetrics()
        self.last_failure_time = 0.0
        self.state_change_time = time.time()
        
        logger.info(f"CircuitBreaker initialized for {service_name}")
    
    def can_execute(self) -> bool:
        """Check if service can execute requests"""
        current_time = time.time()
        
        if self.state == ServiceState.CLOSED:
            return True
        elif self.state == ServiceState.OPEN:
            # Check if timeout has elapsed to try half-open
            if current_time - self.state_change_time >= self.config.timeout_seconds:
                self._transition_to_half_open()
                return True
            return False
        elif self.state == ServiceState.HALF_OPEN:
            return True
        
        return False
    
    async def execute(self, func: Callable[[], Awaitable[Any]]) -> Any:
        """Execute function with circuit breaker protection"""
        if not self.can_execute():
            raise ServiceUnavailableError(f"Service {self.service_name} is currently unavailable")
        
        start_time = time.time()
        
        try:
            result = await func()
            self._record_success(time.time() - start_time)
            return result
            
        except Exception as e:
            self._record_failure()
            raise
    
    def _record_success(self, response_time: float):
        """Record successful request"""
        self.metrics.total_requests += 1
        self.metrics.successful_requests += 1
        self.metrics.consecutive_successes += 1
        self.metrics.consecutive_failures = 0
        self.metrics.last_success_time = time.time()
        self.metrics.last_request_time = time.time()
        
        # Update average response time
        if self.metrics.successful_requests == 1:
            self.metrics.avg_response_time = response_time
        else:
            self.metrics.avg_response_time = (
                (self.metrics.avg_response_time * (self.metrics.successful_requests - 1) + response_time) /
                self.metrics.successful_requests
            )
        
        # State transitions
        if self.state == ServiceState.HALF_OPEN:
            if self.metrics.consecutive_successes >= self.config.success_threshold:
                self._transition_to_closed()
    
    def _record_failure(self):
        """Record failed request"""
        self.metrics.total_requests += 1
        self.metrics.failed_requests += 1
        self.metrics.consecutive_failures += 1
        self.metrics.consecutive_successes = 0
        self.metrics.last_failure_time = time.time()
        self.metrics.last_request_time = time.time()
        
        # Check if we should open the circuit
        if self.state == ServiceState.CLOSED:
            should_open = (
                self.metrics.consecutive_failures >= self.config.failure_threshold or
                (self.metrics.total_requests >= self.config.min_request_threshold and
                 self.metrics.failure_rate >= self.config.failure_rate_threshold)
            )
            if should_open:
                self._transition_to_open()
        elif self.state == ServiceState.HALF_OPEN:
            self._transition_to_open()
    
    def _transition_to_open(self):
        """Transition to open state"""
        self.state = ServiceState.OPEN
        self.state_change_time = time.time()
        logger.warning(f"Circuit breaker OPENED for {self.service_name} - service unavailable")
    
    def _transition_to_half_open(self):
        """Transition to half-open state"""
        self.state = ServiceState.HALF_OPEN
        self.state_change_time = time.time()
        logger.info(f"Circuit breaker HALF-OPEN for {self.service_name} - testing recovery")
    
    def _transition_to_closed(self):
        """Transition to closed state"""
        self.state = ServiceState.CLOSED
        self.state_change_time = time.time()
        self.metrics.consecutive_failures = 0
        logger.info(f"Circuit breaker CLOSED for {self.service_name} - service recovered")
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get circuit breaker state information"""
        return {
            'service_name': self.service_name,
            'state': self.state.value,
            'metrics': {
                'total_requests': self.metrics.total_requests,
                'success_rate': self.metrics.success_rate,
                'failure_rate': self.metrics.failure_rate,
                'consecutive_failures': self.metrics.consecutive_failures,
                'consecutive_successes': self.metrics.consecutive_successes,
                'avg_response_time': self.metrics.avg_response_time
            },
            'time_in_current_state': time.time() - self.state_change_time
        }


class RetryHandler:
    """
    Advanced retry handler with multiple strategies
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        
    async def execute_with_retry(
        self,
        func: Callable[[], Awaitable[Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                if attempt > 0:
                    delay = self._calculate_delay(attempt)
                    logger.info(f"Retrying in {delay:.2f}s (attempt {attempt + 1}/{self.config.max_attempts})")
                    await asyncio.sleep(delay)
                
                result = await func()
                
                if attempt > 0:
                    logger.info(f"Retry succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
                # Don't retry on certain types of errors
                if not self._should_retry(e):
                    break
        
        # All retries failed
        logger.error(f"All {self.config.max_attempts} attempts failed")
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        if self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * attempt
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (self.config.exponential_base ** (attempt - 1))
        else:
            delay = self.config.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, self.config.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0.1, delay)  # Minimum 100ms delay
    
    def _should_retry(self, exception: Exception) -> bool:
        """Determine if exception should be retried"""
        # Don't retry on certain types of errors
        non_retryable_errors = (
            ValueError,  # Bad input
            TypeError,   # Type errors
            KeyError,    # Missing keys
        )
        
        if isinstance(exception, non_retryable_errors):
            return False
        
        # Retry on network, timeout, and service errors
        return True


class ErrorMessageManager:
    """
    Manages user-friendly error messages
    """
    
    def __init__(self):
        self.error_messages = {
            'timeout': [
                "I'm taking a bit longer than usual to respond. Let me try a faster approach.",
                "My processing is running slow right now. Switching to a quicker method.",
                "That's taking longer than expected. Let me try a different way."
            ],
            'service_unavailable': [
                "I'm having trouble connecting to my AI services. Let me try an alternative.",
                "My primary AI service is temporarily unavailable. Switching to backup.",
                "There's a temporary issue with my main processing service. Using fallback."
            ],
            'rate_limited': [
                "I'm being rate limited right now. Let me wait a moment and try again.",
                "Too many requests at once. Pausing briefly before continuing.",
                "Hit a rate limit. Taking a short break before retrying."
            ],
            'generic_error': [
                "I encountered an unexpected issue. Let me try a different approach.",
                "Something went wrong on my end. Attempting an alternative method.",
                "I ran into a problem processing that. Let me try again differently."
            ],
            'all_services_failed': [
                "I'm experiencing technical difficulties with all my AI services right now.",
                "All my processing services are temporarily unavailable. Please try again in a moment.",
                "I'm having widespread service issues. Please bear with me while I recover."
            ]
        }
    
    def get_user_friendly_message(self, error_type: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Get user-friendly error message"""
        messages = self.error_messages.get(error_type, self.error_messages['generic_error'])
        message = random.choice(messages)
        
        # Add context if available
        if context and 'service_name' in context:
            service_name = context['service_name']
            if 'fast' in service_name.lower():
                message += " (Quick response mode unavailable)"
            elif 'deep' in service_name.lower():
                message += " (Detailed analysis mode unavailable)"
        
        return message


class FallbackManager:
    """
    Comprehensive fallback management system
    """
    
    def __init__(
        self,
        config: Optional[FallbackConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[RetryConfig] = None
    ):
        self.config = config or FallbackConfig()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_handler = RetryHandler(retry_config)
        self.error_manager = ErrorMessageManager()
        
        # Initialize circuit breakers for each service in fallback chain
        for service_name in self.config.fallback_chain:
            self.circuit_breakers[service_name] = CircuitBreaker(
                service_name, circuit_breaker_config
            )
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'fallback_used': 0,
            'all_services_failed': 0,
            'circuit_breaker_trips': 0
        }
        
        logger.info(f"FallbackManager initialized with chain: {self.config.fallback_chain}")
    
    async def execute_with_fallback(
        self,
        primary_service: str,
        service_functions: Dict[str, Callable[[], Awaitable[Any]]],
        context: Optional[Dict[str, Any]] = None,
        status_callback: Optional[Callable[[str], Awaitable[None]]] = None
    ) -> Any:
        """
        Execute request with comprehensive fallback logic
        
        Args:
            primary_service: Name of primary service to try first
            service_functions: Dict mapping service names to async functions
            context: Optional context information
            status_callback: Optional callback for status updates
            
        Returns:
            Result from successful service execution
        """
        self.stats['total_requests'] += 1
        last_exception = None
        attempted_services = []
        
        # Build execution order: primary first, then fallback chain
        execution_order = [primary_service]
        for service in self.config.fallback_chain:
            if service != primary_service and service in service_functions:
                execution_order.append(service)
        
        execution_order = execution_order[:self.config.max_fallback_attempts]
        
        for i, service_name in enumerate(execution_order):
            if service_name not in service_functions:
                logger.warning(f"Service {service_name} not available in service_functions")
                continue
            
            attempted_services.append(service_name)
            circuit_breaker = self.circuit_breakers.get(service_name)
            
            # Check circuit breaker
            if circuit_breaker and not circuit_breaker.can_execute():
                logger.info(f"Circuit breaker open for {service_name}, skipping")
                continue
            
            try:
                # Send status update for fallback attempts
                if i > 0 and status_callback:
                    message = self.error_manager.get_user_friendly_message(
                        'service_unavailable',
                        {'service_name': service_name}
                    )
                    await status_callback(message)
                    self.stats['fallback_used'] += 1
                
                # Execute with circuit breaker protection
                if circuit_breaker:
                    result = await circuit_breaker.execute(service_functions[service_name])
                else:
                    result = await service_functions[service_name]()
                
                # Success
                self.stats['successful_requests'] += 1
                
                if i > 0:
                    logger.info(f"Fallback to {service_name} succeeded after {i} attempts")
                
                return result
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Service {service_name} failed: {str(e)}")
                
                # Record circuit breaker failure is handled by circuit breaker itself
                continue
        
        # All services failed
        self.stats['all_services_failed'] += 1
        
        if status_callback:
            message = self.error_manager.get_user_friendly_message('all_services_failed')
            await status_callback(message)
        
        error_msg = f"All services failed. Attempted: {attempted_services}"
        logger.error(error_msg)
        raise AllServicesFailed(error_msg, attempted_services, last_exception)
    
    def get_service_health(self) -> Dict[str, Any]:
        """Get health status of all services"""
        health_info = {}
        
        for service_name, circuit_breaker in self.circuit_breakers.items():
            health_info[service_name] = circuit_breaker.get_state_info()
        
        return {
            'services': health_info,
            'overall_stats': self.stats.copy()
        }
    
    def reset_circuit_breaker(self, service_name: str) -> bool:
        """Manually reset a circuit breaker"""
        if service_name in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[service_name]
            circuit_breaker._transition_to_closed()
            logger.info(f"Manually reset circuit breaker for {service_name}")
            return True
        return False


# Custom exceptions
class ServiceUnavailableError(Exception):
    """Raised when service is unavailable due to circuit breaker"""
    pass


class AllServicesFailed(Exception):
    """Raised when all fallback services have failed"""
    
    def __init__(self, message: str, attempted_services: List[str], last_exception: Exception):
        super().__init__(message)
        self.attempted_services = attempted_services
        self.last_exception = last_exception


# Factory functions
def create_fallback_manager(
    fallback_chain: Optional[List[str]] = None,
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    retry_config: Optional[RetryConfig] = None
) -> FallbackManager:
    """Factory function to create FallbackManager with custom configuration"""
    fallback_config = FallbackConfig()
    if fallback_chain:
        fallback_config.fallback_chain = fallback_chain
    
    return FallbackManager(
        config=fallback_config,
        circuit_breaker_config=circuit_breaker_config,
        retry_config=retry_config
    )


def create_circuit_breaker(
    service_name: str,
    failure_threshold: int = 5,
    timeout_seconds: float = 60.0
) -> CircuitBreaker:
    """Factory function to create CircuitBreaker with custom configuration"""
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        timeout_seconds=timeout_seconds
    )
    return CircuitBreaker(service_name, config) 