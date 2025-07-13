"""
Comprehensive tests for the fallback manager system

Tests cover:
- CircuitBreaker functionality and state transitions
- RetryHandler with various strategies
- ErrorMessageManager message generation
- FallbackManager comprehensive fallback logic
- Service health monitoring
- Error handling and user-friendly messages
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from assistant.fallback_manager import (
    CircuitBreaker, CircuitBreakerConfig, ServiceState, ServiceMetrics,
    RetryHandler, RetryConfig, RetryStrategy,
    ErrorMessageManager,
    FallbackManager, FallbackConfig,
    ServiceUnavailableError, AllServicesFailed,
    create_fallback_manager, create_circuit_breaker
)


class TestCircuitBreaker:
    """Test CircuitBreaker functionality"""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization"""
        cb = CircuitBreaker("test_service")
        
        assert cb.service_name == "test_service"
        assert cb.state == ServiceState.CLOSED
        assert cb.metrics.total_requests == 0
        assert cb.metrics.failed_requests == 0
        assert cb.metrics.successful_requests == 0
    
    def test_circuit_breaker_custom_config(self):
        """Test circuit breaker with custom configuration"""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout_seconds=30.0,
            failure_rate_threshold=0.8
        )
        cb = CircuitBreaker("test_service", config)
        
        assert cb.config.failure_threshold == 3
        assert cb.config.timeout_seconds == 30.0
        assert cb.config.failure_rate_threshold == 0.8
    
    def test_can_execute_closed_state(self):
        """Test can_execute in closed state"""
        cb = CircuitBreaker("test_service")
        assert cb.can_execute() == True
    
    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Test successful execution through circuit breaker"""
        cb = CircuitBreaker("test_service")
        
        async def mock_func():
            return "success"
        
        result = await cb.execute(mock_func)
        
        assert result == "success"
        assert cb.metrics.successful_requests == 1
        assert cb.metrics.total_requests == 1
        assert cb.metrics.consecutive_successes == 1
        assert cb.metrics.consecutive_failures == 0
        assert cb.state == ServiceState.CLOSED
    
    @pytest.mark.asyncio
    async def test_failed_execution(self):
        """Test failed execution through circuit breaker"""
        cb = CircuitBreaker("test_service")
        
        async def mock_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            await cb.execute(mock_func)
        
        assert cb.metrics.failed_requests == 1
        assert cb.metrics.total_requests == 1
        assert cb.metrics.consecutive_failures == 1
        assert cb.metrics.consecutive_successes == 0
        assert cb.state == ServiceState.CLOSED  # Not enough failures to open
    
    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self):
        """Test circuit opens after exceeding failure threshold"""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker("test_service", config)
        
        async def mock_func():
            raise ValueError("Test error")
        
        # First failure
        with pytest.raises(ValueError):
            await cb.execute(mock_func)
        assert cb.state == ServiceState.CLOSED
        
        # Second failure - should open circuit
        with pytest.raises(ValueError):
            await cb.execute(mock_func)
        assert cb.state == ServiceState.OPEN
        
        # Third attempt should be blocked
        with pytest.raises(ServiceUnavailableError):
            await cb.execute(mock_func)
    
    @pytest.mark.asyncio
    async def test_circuit_half_open_recovery(self):
        """Test circuit recovery through half-open state"""
        config = CircuitBreakerConfig(failure_threshold=1, success_threshold=1, timeout_seconds=0.1)
        cb = CircuitBreaker("test_service", config)
        
        # Force circuit open
        async def failing_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            await cb.execute(failing_func)
        assert cb.state == ServiceState.OPEN
        
        # Wait for timeout
        await asyncio.sleep(0.2)
        
        # Should be able to execute (half-open)
        assert cb.can_execute() == True
        assert cb.state == ServiceState.HALF_OPEN
        
        # Successful execution should close circuit
        async def success_func():
            return "success"
        
        result = await cb.execute(success_func)
        assert result == "success"
        assert cb.state == ServiceState.CLOSED
    
    def test_service_metrics_properties(self):
        """Test service metrics property calculations"""
        metrics = ServiceMetrics()
        
        # Test empty metrics
        assert metrics.failure_rate == 0.0
        assert metrics.success_rate == 1.0
        
        # Test with data
        metrics.total_requests = 10
        metrics.failed_requests = 3
        metrics.successful_requests = 7
        
        assert metrics.failure_rate == 0.3
        assert metrics.success_rate == 0.7
    
    def test_get_state_info(self):
        """Test circuit breaker state info"""
        cb = CircuitBreaker("test_service")
        state_info = cb.get_state_info()
        
        assert state_info['service_name'] == "test_service"
        assert state_info['state'] == "closed"
        assert 'metrics' in state_info
        assert 'time_in_current_state' in state_info


class TestRetryHandler:
    """Test RetryHandler functionality"""
    
    def test_retry_handler_initialization(self):
        """Test retry handler initialization"""
        handler = RetryHandler()
        
        assert handler.config.strategy == RetryStrategy.EXPONENTIAL
        assert handler.config.max_attempts == 3
        assert handler.config.base_delay == 1.0
        assert handler.config.jitter == True
    
    def test_retry_handler_custom_config(self):
        """Test retry handler with custom configuration"""
        config = RetryConfig(
            strategy=RetryStrategy.LINEAR,
            max_attempts=5,
            base_delay=0.5,
            jitter=False
        )
        handler = RetryHandler(config)
        
        assert handler.config.strategy == RetryStrategy.LINEAR
        assert handler.config.max_attempts == 5
        assert handler.config.base_delay == 0.5
        assert handler.config.jitter == False
    
    @pytest.mark.asyncio
    async def test_successful_execution_first_attempt(self):
        """Test successful execution on first attempt"""
        handler = RetryHandler()
        
        async def mock_func():
            return "success"
        
        result = await handler.execute_with_retry(mock_func)
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry behavior on failure"""
        handler = RetryHandler(RetryConfig(max_attempts=3, base_delay=0.1))
        
        attempt_count = 0
        async def mock_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise RuntimeError("Temporary failure")
            return "success"
        
        result = await handler.execute_with_retry(mock_func)
        assert result == "success"
        assert attempt_count == 3
    
    @pytest.mark.asyncio
    async def test_all_retries_fail(self):
        """Test behavior when all retries fail"""
        handler = RetryHandler(RetryConfig(max_attempts=2, base_delay=0.1))
        
        async def mock_func():
            raise RuntimeError("Persistent failure")
        
        with pytest.raises(RuntimeError, match="Persistent failure"):
            await handler.execute_with_retry(mock_func)
    
    @pytest.mark.asyncio
    async def test_no_retry_on_non_retryable_error(self):
        """Test no retry on non-retryable errors"""
        handler = RetryHandler(RetryConfig(max_attempts=3, base_delay=0.1))
        
        attempt_count = 0
        async def mock_func():
            nonlocal attempt_count
            attempt_count += 1
            raise ValueError("Bad input")  # Non-retryable
        
        with pytest.raises(ValueError, match="Bad input"):
            await handler.execute_with_retry(mock_func)
        
        assert attempt_count == 1  # No retries
    
    def test_calculate_delay_linear(self):
        """Test linear delay calculation"""
        config = RetryConfig(strategy=RetryStrategy.LINEAR, base_delay=1.0, jitter=False)
        handler = RetryHandler(config)
        
        assert handler._calculate_delay(1) == 1.0
        assert handler._calculate_delay(2) == 2.0
        assert handler._calculate_delay(3) == 3.0
    
    def test_calculate_delay_exponential(self):
        """Test exponential delay calculation"""
        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL, 
            base_delay=1.0, 
            exponential_base=2.0, 
            jitter=False
        )
        handler = RetryHandler(config)
        
        assert handler._calculate_delay(1) == 1.0
        assert handler._calculate_delay(2) == 2.0
        assert handler._calculate_delay(3) == 4.0
    
    def test_calculate_delay_with_max_limit(self):
        """Test delay calculation with maximum limit"""
        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay=1.0,
            max_delay=5.0,
            jitter=False
        )
        handler = RetryHandler(config)
        
        assert handler._calculate_delay(10) == 5.0  # Should be capped at max_delay
    
    def test_should_retry_logic(self):
        """Test should retry logic"""
        handler = RetryHandler()
        
        # Should retry on general exceptions
        assert handler._should_retry(RuntimeError("Network error")) == True
        assert handler._should_retry(ConnectionError("Connection failed")) == True
        
        # Should not retry on specific exceptions
        assert handler._should_retry(ValueError("Bad input")) == False
        assert handler._should_retry(TypeError("Type error")) == False
        assert handler._should_retry(KeyError("Missing key")) == False


class TestErrorMessageManager:
    """Test ErrorMessageManager functionality"""
    
    def test_error_message_manager_initialization(self):
        """Test error message manager initialization"""
        manager = ErrorMessageManager()
        
        assert 'timeout' in manager.error_messages
        assert 'service_unavailable' in manager.error_messages
        assert 'rate_limited' in manager.error_messages
        assert 'generic_error' in manager.error_messages
        assert 'all_services_failed' in manager.error_messages
    
    def test_get_user_friendly_message_timeout(self):
        """Test getting timeout messages"""
        manager = ErrorMessageManager()
        
        message = manager.get_user_friendly_message('timeout')
        assert isinstance(message, str)
        assert len(message) > 0
        
        # Should be one of the predefined timeout messages
        assert any(msg in message for msg in manager.error_messages['timeout'])
    
    def test_get_user_friendly_message_service_unavailable(self):
        """Test getting service unavailable messages"""
        manager = ErrorMessageManager()
        
        message = manager.get_user_friendly_message('service_unavailable')
        assert isinstance(message, str)
        assert len(message) > 0
        
        # Should be one of the predefined service unavailable messages
        assert any(msg in message for msg in manager.error_messages['service_unavailable'])
    
    def test_get_user_friendly_message_with_context(self):
        """Test getting messages with context"""
        manager = ErrorMessageManager()
        
        # Test with fast service context
        message = manager.get_user_friendly_message(
            'service_unavailable',
            {'service_name': 'fast_service'}
        )
        assert "(Quick response mode unavailable)" in message
        
        # Test with deep service context
        message = manager.get_user_friendly_message(
            'service_unavailable',
            {'service_name': 'deep_analysis_service'}
        )
        assert "(Detailed analysis mode unavailable)" in message
    
    def test_get_user_friendly_message_unknown_type(self):
        """Test getting messages for unknown error types"""
        manager = ErrorMessageManager()
        
        message = manager.get_user_friendly_message('unknown_error_type')
        assert isinstance(message, str)
        assert len(message) > 0
        
        # Should fallback to generic error messages
        assert any(msg in message for msg in manager.error_messages['generic_error'])
    
    def test_message_variety(self):
        """Test that different calls return different messages"""
        manager = ErrorMessageManager()
        
        # Get multiple messages of same type
        messages = [
            manager.get_user_friendly_message('timeout')
            for _ in range(10)
        ]
        
        # Should have some variety (not all identical)
        unique_messages = set(messages)
        assert len(unique_messages) > 1


class TestFallbackManager:
    """Test FallbackManager comprehensive functionality"""
    
    def test_fallback_manager_initialization(self):
        """Test fallback manager initialization"""
        manager = FallbackManager()
        
        assert manager.config.enable_fallback == True
        assert manager.config.fallback_chain == ['fast', 'deep', 'backup']
        assert 'fast' in manager.circuit_breakers
        assert 'deep' in manager.circuit_breakers
        assert 'backup' in manager.circuit_breakers
        assert manager.stats['total_requests'] == 0
    
    def test_fallback_manager_custom_config(self):
        """Test fallback manager with custom configuration"""
        fallback_config = FallbackConfig(
            fallback_chain=['primary', 'secondary'],
            max_fallback_attempts=2
        )
        manager = FallbackManager(config=fallback_config)
        
        assert manager.config.fallback_chain == ['primary', 'secondary']
        assert manager.config.max_fallback_attempts == 2
        assert 'primary' in manager.circuit_breakers
        assert 'secondary' in manager.circuit_breakers
    
    @pytest.mark.asyncio
    async def test_successful_primary_execution(self):
        """Test successful execution on primary service"""
        manager = FallbackManager()
        
        async def primary_func():
            return "primary_result"
        
        service_functions = {
            'fast': primary_func,
            'deep': AsyncMock(return_value="deep_result"),
            'backup': AsyncMock(return_value="backup_result")
        }
        
        result = await manager.execute_with_fallback(
            primary_service='fast',
            service_functions=service_functions
        )
        
        assert result == "primary_result"
        assert manager.stats['successful_requests'] == 1
        assert manager.stats['fallback_used'] == 0
    
    @pytest.mark.asyncio
    async def test_fallback_on_primary_failure(self):
        """Test fallback when primary service fails"""
        manager = FallbackManager()
        
        async def failing_primary():
            raise RuntimeError("Primary service failed")
        
        async def working_fallback():
            return "fallback_result"
        
        service_functions = {
            'fast': failing_primary,
            'deep': working_fallback,
            'backup': AsyncMock(return_value="backup_result")
        }
        
        status_updates = []
        async def status_callback(message):
            status_updates.append(message)
        
        result = await manager.execute_with_fallback(
            primary_service='fast',
            service_functions=service_functions,
            status_callback=status_callback
        )
        
        assert result == "fallback_result"
        assert manager.stats['successful_requests'] == 1
        assert manager.stats['fallback_used'] == 1
        assert len(status_updates) > 0
    
    @pytest.mark.asyncio
    async def test_all_services_fail(self):
        """Test behavior when all services fail"""
        manager = FallbackManager()
        
        async def failing_func():
            raise RuntimeError("Service failed")
        
        service_functions = {
            'fast': failing_func,
            'deep': failing_func,
            'backup': failing_func
        }
        
        status_updates = []
        async def status_callback(message):
            status_updates.append(message)
        
        with pytest.raises(AllServicesFailed) as exc_info:
            await manager.execute_with_fallback(
                primary_service='fast',
                service_functions=service_functions,
                status_callback=status_callback
            )
        
        assert 'fast' in exc_info.value.attempted_services
        assert 'deep' in exc_info.value.attempted_services
        assert 'backup' in exc_info.value.attempted_services
        assert manager.stats['all_services_failed'] == 1
        assert len(status_updates) > 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test integration with circuit breaker"""
        # Create manager with low failure threshold
        circuit_config = CircuitBreakerConfig(failure_threshold=1)
        manager = FallbackManager(circuit_breaker_config=circuit_config)
        
        # Force circuit breaker open by failing fast service
        async def failing_fast():
            raise RuntimeError("Fast service failed")
        
        async def working_deep():
            return "deep_result"
        
        service_functions = {
            'fast': failing_fast,
            'deep': working_deep,
            'backup': AsyncMock(return_value="backup_result")
        }
        
        # First call should fail fast service and open circuit
        result = await manager.execute_with_fallback(
            primary_service='fast',
            service_functions=service_functions
        )
        assert result == "deep_result"
        
        # Second call should skip fast service due to open circuit
        result = await manager.execute_with_fallback(
            primary_service='fast',
            service_functions=service_functions
        )
        assert result == "deep_result"
        
        # Fast service should have been skipped
        fast_cb = manager.circuit_breakers['fast']
        assert fast_cb.state == ServiceState.OPEN
    
    @pytest.mark.asyncio
    async def test_service_health_monitoring(self):
        """Test service health monitoring"""
        manager = FallbackManager()
        
        # Execute some requests to generate health data
        async def mock_func():
            return "result"
        
        service_functions = {'fast': mock_func, 'deep': mock_func, 'backup': mock_func}
        
        await manager.execute_with_fallback('fast', service_functions)
        
        health = manager.get_service_health()
        
        assert 'services' in health
        assert 'overall_stats' in health
        assert 'fast' in health['services']
        assert health['services']['fast']['metrics']['total_requests'] > 0
    
    def test_reset_circuit_breaker(self):
        """Test manual circuit breaker reset"""
        manager = FallbackManager()
        
        # Open circuit breaker manually
        cb = manager.circuit_breakers['fast']
        cb._transition_to_open()
        assert cb.state == ServiceState.OPEN
        
        # Reset circuit breaker
        result = manager.reset_circuit_breaker('fast')
        assert result == True
        assert cb.state == ServiceState.CLOSED
        
        # Try to reset non-existent service
        result = manager.reset_circuit_breaker('nonexistent')
        assert result == False


class TestFactoryFunctions:
    """Test factory functions"""
    
    def test_create_fallback_manager(self):
        """Test create_fallback_manager factory function"""
        manager = create_fallback_manager(
            fallback_chain=['primary', 'secondary'],
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=3),
            retry_config=RetryConfig(max_attempts=5)
        )
        
        assert isinstance(manager, FallbackManager)
        assert manager.config.fallback_chain == ['primary', 'secondary']
        assert manager.circuit_breakers['primary'].config.failure_threshold == 3
        assert manager.retry_handler.config.max_attempts == 5
    
    def test_create_circuit_breaker(self):
        """Test create_circuit_breaker factory function"""
        cb = create_circuit_breaker(
            service_name='test_service',
            failure_threshold=10,
            timeout_seconds=120.0
        )
        
        assert isinstance(cb, CircuitBreaker)
        assert cb.service_name == 'test_service'
        assert cb.config.failure_threshold == 10
        assert cb.config.timeout_seconds == 120.0


class TestIntegrationScenarios:
    """Test complex integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_mixed_success_failure_scenario(self):
        """Test complex scenario with mixed success/failure patterns"""
        manager = FallbackManager()
        
        call_count = {'fast': 0, 'deep': 0, 'backup': 0}
        
        async def intermittent_fast():
            call_count['fast'] += 1
            if call_count['fast'] <= 2:
                raise RuntimeError("Fast service overloaded")
            return "fast_result"
        
        async def reliable_deep():
            call_count['deep'] += 1
            return "deep_result"
        
        async def backup_func():
            call_count['backup'] += 1
            return "backup_result"
        
        service_functions = {
            'fast': intermittent_fast,
            'deep': reliable_deep,
            'backup': backup_func
        }
        
        # First two calls should fail fast, succeed on deep
        result1 = await manager.execute_with_fallback('fast', service_functions)
        assert result1 == "deep_result"
        
        result2 = await manager.execute_with_fallback('fast', service_functions)
        assert result2 == "deep_result"
        
        # Third call should succeed on fast (if circuit hasn't opened)
        # This depends on circuit breaker threshold
        result3 = await manager.execute_with_fallback('fast', service_functions)
        assert result3 in ["fast_result", "deep_result"]
        
        # Verify call counts
        assert call_count['fast'] >= 2
        assert call_count['deep'] >= 2
    
    @pytest.mark.asyncio
    async def test_performance_degradation_scenario(self):
        """Test scenario where service performance degrades over time"""
        manager = FallbackManager()
        
        slow_call_count = 0
        
        async def increasingly_slow_service():
            nonlocal slow_call_count
            slow_call_count += 1
            
            # Simulate increasing slowness
            await asyncio.sleep(0.1 * slow_call_count)
            return f"slow_result_{slow_call_count}"
        
        async def fast_backup():
            return "fast_backup_result"
        
        service_functions = {
            'slow': increasingly_slow_service,
            'fast': fast_backup,
            'backup': fast_backup
        }
        
        start_time = time.time()
        
        # Make several calls
        results = []
        for i in range(3):
            result = await manager.execute_with_fallback('slow', service_functions)
            results.append(result)
        
        total_time = time.time() - start_time
        
        # Should have some results from slow service and some from fallback
        assert len(results) == 3
        assert slow_call_count > 0 