"""
Comprehensive tests for the error handling system
Tests exception hierarchy, retry logic, circuit breakers, graceful degradation, structured logging, and health monitoring
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import asdict

from assistant.error_handling import (
    ErrorCategory, ErrorContext, VoiceAIException, STTException, LLMException, TTSException,
    AudioException, CircuitBreakerOpenError, AllTiersFailedError, BackoffConfig,
    AdvancedBackoff, retry_with_backoff, CircuitState, HealthMetrics, CircuitBreakerConfig,
    ModernCircuitBreaker, ServiceTier, ServiceCapability, GracefulDegradationManager
)
from assistant.structured_logging import (
    VoiceAILogContext, VoiceAILogger, voice_ai_request_context,
    log_audio_processing, log_model_inference, get_voice_ai_logger
)
from assistant.health_monitoring import (
    HealthStatus, ServiceHealthMetrics, SystemHealthMetrics, HealthChecker,
    SystemHealthMonitor, get_health_monitor, health_monitored_operation
)

class TestErrorHierarchy:
    """Test structured exception hierarchy"""
    
    def test_voice_ai_exception_base(self):
        """Test base VoiceAIException"""
        context = ErrorContext(
            service_name="test_service",
            operation="test_operation",
            timestamp=time.time(),
            request_id="test-123"
        )
        
        exc = VoiceAIException(
            "Test error",
            ErrorCategory.TRANSIENT,
            context=context,
            retryable=True
        )
        
        assert exc.category == ErrorCategory.TRANSIENT
        assert exc.retryable is True
        assert exc.context == context
        assert "Network connection issue" in exc.user_message
    
    def test_specific_exceptions(self):
        """Test specific exception types"""
        stt_exc = STTException("STT failed", ErrorCategory.TIMEOUT, retryable=True)
        assert isinstance(stt_exc, VoiceAIException)
        assert stt_exc.category == ErrorCategory.TIMEOUT
        
        llm_exc = LLMException("LLM failed", ErrorCategory.RATE_LIMIT, retryable=False)
        assert isinstance(llm_exc, VoiceAIException)
        assert llm_exc.category == ErrorCategory.RATE_LIMIT
        
        tts_exc = TTSException("TTS failed", ErrorCategory.NETWORK, retryable=True)
        assert isinstance(tts_exc, VoiceAIException)
        assert tts_exc.category == ErrorCategory.NETWORK
        
        audio_exc = AudioException("Audio failed", ErrorCategory.AUDIO, retryable=True)
        assert isinstance(audio_exc, VoiceAIException)
        assert audio_exc.category == ErrorCategory.AUDIO
    
    def test_circuit_breaker_open_error(self):
        """Test CircuitBreakerOpenError"""
        exc = CircuitBreakerOpenError("test_service", 0.25)
        
        assert exc.service_name == "test_service"
        assert exc.health_score == 0.25
        assert exc.category == ErrorCategory.TRANSIENT
        assert exc.retryable is False
    
    def test_all_tiers_failed_error(self):
        """Test AllTiersFailedError"""
        attempted_tiers = ["primary", "fast", "offline"]
        original_exc = Exception("Original error")
        
        exc = AllTiersFailedError("test_operation", attempted_tiers, original_exc)
        
        assert exc.operation == "test_operation"
        assert exc.attempted_tiers == attempted_tiers
        assert exc.category == ErrorCategory.PERMANENT
        assert exc.retryable is False
        assert exc.original_exception == original_exc

class TestAdvancedBackoff:
    """Test exponential backoff with jitter"""
    
    def test_backoff_config(self):
        """Test backoff configuration"""
        config = BackoffConfig(
            base_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter_type="full",
            max_attempts=5
        )
        
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter_type == "full"
        assert config.max_attempts == 5
    
    def test_calculate_delay_no_jitter(self):
        """Test delay calculation without jitter"""
        config = BackoffConfig(base_delay=1.0, jitter_type="none")
        backoff = AdvancedBackoff(config)
        
        assert backoff.calculate_delay(0) == 1.0
        assert backoff.calculate_delay(1) == 2.0
        assert backoff.calculate_delay(2) == 4.0
        assert backoff.calculate_delay(3) == 8.0
    
    def test_calculate_delay_with_jitter(self):
        """Test delay calculation with jitter"""
        config = BackoffConfig(base_delay=1.0, jitter_type="full", max_delay=10.0)
        backoff = AdvancedBackoff(config)
        
        # With full jitter, delay should be between 0 and base delay
        for attempt in range(5):
            delay = backoff.calculate_delay(attempt)
            expected_base = min(1.0 * (2 ** attempt), 10.0)
            assert 0 <= delay <= expected_base
    
    def test_should_retry(self):
        """Test retry decision logic"""
        config = BackoffConfig(max_attempts=3)
        backoff = AdvancedBackoff(config)
        
        # Should retry for retryable exceptions
        retryable_exc = VoiceAIException("Test", ErrorCategory.TRANSIENT, retryable=True)
        assert backoff.should_retry(0, retryable_exc) is True
        assert backoff.should_retry(1, retryable_exc) is True
        assert backoff.should_retry(2, retryable_exc) is True
        assert backoff.should_retry(3, retryable_exc) is False
        
        # Should not retry for non-retryable exceptions
        non_retryable_exc = VoiceAIException("Test", ErrorCategory.PERMANENT, retryable=False)
        assert backoff.should_retry(0, non_retryable_exc) is False
    
    def test_get_timeout(self):
        """Test adaptive timeout calculation"""
        config = BackoffConfig(timeout_multiplier=1.5)
        backoff = AdvancedBackoff(config)
        
        assert backoff.get_timeout(0, 10.0) == 10.0
        assert backoff.get_timeout(1, 10.0) == 15.0
        assert backoff.get_timeout(2, 10.0) == 22.5

@pytest.mark.asyncio
class TestRetryWithBackoff:
    """Test retry with backoff functionality"""
    
    async def test_successful_operation(self):
        """Test successful operation without retries"""
        config = BackoffConfig(max_attempts=3)
        call_count = 0
        
        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await retry_with_backoff(successful_func, config)
        assert result == "success"
        assert call_count == 1
    
    async def test_retry_on_failure(self):
        """Test retry on transient failure"""
        config = BackoffConfig(max_attempts=3, base_delay=0.1)
        call_count = 0
        
        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise VoiceAIException("Temporary failure", ErrorCategory.TRANSIENT, retryable=True)
            return "success"
        
        result = await retry_with_backoff(failing_func, config)
        assert result == "success"
        assert call_count == 3
    
    async def test_max_attempts_exceeded(self):
        """Test failure when max attempts exceeded"""
        config = BackoffConfig(max_attempts=3, base_delay=0.1)
        call_count = 0
        
        async def always_failing_func():
            nonlocal call_count
            call_count += 1
            raise VoiceAIException("Persistent failure", ErrorCategory.TRANSIENT, retryable=True)
        
        with pytest.raises(VoiceAIException):
            await retry_with_backoff(always_failing_func, config)
        
        assert call_count == 3
    
    async def test_non_retryable_exception(self):
        """Test immediate failure for non-retryable exceptions"""
        config = BackoffConfig(max_attempts=3, base_delay=0.1)
        call_count = 0
        
        async def non_retryable_func():
            nonlocal call_count
            call_count += 1
            raise VoiceAIException("Permanent failure", ErrorCategory.PERMANENT, retryable=False)
        
        with pytest.raises(VoiceAIException):
            await retry_with_backoff(non_retryable_func, config)
        
        assert call_count == 1
    
    async def test_status_callback(self):
        """Test status callback during retries"""
        config = BackoffConfig(max_attempts=3, base_delay=0.1)
        callback_calls = []
        call_count = 0
        
        async def status_callback(message, attempt):
            callback_calls.append((message, attempt))
        
        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise VoiceAIException("Temporary failure", ErrorCategory.TRANSIENT, retryable=True)
            return "success"
        
        result = await retry_with_backoff(failing_func, config, status_callback=status_callback)
        assert result == "success"
        assert len(callback_calls) == 1  # One retry
        assert callback_calls[0][1] == 1  # Attempt number

class TestCircuitBreaker:
    """Test circuit breaker functionality"""
    
    def test_circuit_breaker_config(self):
        """Test circuit breaker configuration"""
        config = CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=3,
            timeout_duration=60.0,
            failure_rate_threshold=0.5
        )
        
        assert config.failure_threshold == 5
        assert config.success_threshold == 3
        assert config.timeout_duration == 60.0
        assert config.failure_rate_threshold == 0.5
    
    def test_health_metrics(self):
        """Test health metrics calculations"""
        metrics = HealthMetrics()
        
        # Test initial state
        assert metrics.failure_rate == 0.0
        assert metrics.success_rate == 1.0
        assert metrics.health_score == 1.0
        
        # Add some requests
        metrics.total_requests = 10
        metrics.successful_requests = 8
        metrics.failed_requests = 2
        metrics.avg_response_time = 2.0
        metrics.consecutive_failures = 1
        
        assert metrics.failure_rate == 0.2
        assert metrics.success_rate == 0.8
        assert 0.5 < metrics.health_score < 1.0  # Should be reasonable
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state"""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = ModernCircuitBreaker("test", config)
        
        assert cb.state == CircuitState.CLOSED
        
        # Successful calls should work
        call_count = 0
        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await cb.call(successful_func)
        assert result == "success"
        assert call_count == 1
        assert cb.metrics.successful_requests == 1
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opening(self):
        """Test circuit breaker opening on failures"""
        config = CircuitBreakerConfig(failure_threshold=3, min_request_threshold=1)
        cb = ModernCircuitBreaker("test", config)
        
        # Cause failures to open circuit
        async def failing_func():
            raise Exception("Test failure")
        
        # Should fail and eventually open
        for i in range(3):
            with pytest.raises(Exception):
                await cb.call(failing_func)
        
        assert cb.state == CircuitState.OPEN
        assert cb.metrics.consecutive_failures == 3
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_open_state(self):
        """Test circuit breaker in open state"""
        config = CircuitBreakerConfig(failure_threshold=1, timeout_duration=0.1)
        cb = ModernCircuitBreaker("test", config)
        
        # Force to open state
        async def failing_func():
            raise Exception("Test failure")
        
        with pytest.raises(Exception):
            await cb.call(failing_func)
        
        assert cb.state == CircuitState.OPEN
        
        # Should raise CircuitBreakerOpenError
        async def any_func():
            return "success"
        
        with pytest.raises(CircuitBreakerOpenError):
            await cb.call(any_func)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker half-open and recovery"""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=2,
            timeout_duration=0.1
        )
        cb = ModernCircuitBreaker("test", config)
        
        # Force to open state
        async def failing_func():
            raise Exception("Test failure")
        
        with pytest.raises(Exception):
            await cb.call(failing_func)
        
        assert cb.state == CircuitState.OPEN
        
        # Wait for timeout
        await asyncio.sleep(0.2)
        
        # Should transition to half-open
        async def successful_func():
            return "success"
        
        # First call should transition to half-open
        result = await cb.call(successful_func)
        assert result == "success"
        assert cb.state == CircuitState.HALF_OPEN
        
        # Second successful call should close circuit
        result = await cb.call(successful_func)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED

class TestGracefulDegradation:
    """Test graceful degradation manager"""
    
    def test_service_capability(self):
        """Test service capability definition"""
        capability = ServiceCapability(
            name="test-service",
            tier=ServiceTier.PRIMARY,
            latency_target=2.0,
            quality_score=0.95,
            reliability_score=0.90,
            cost_per_request=1.0
        )
        
        assert capability.name == "test-service"
        assert capability.tier == ServiceTier.PRIMARY
        assert capability.latency_target == 2.0
        assert capability.quality_score == 0.95
    
    def test_service_registration(self):
        """Test service registration"""
        manager = GracefulDegradationManager()
        capability = ServiceCapability(
            "test-service", ServiceTier.PRIMARY, 2.0, 0.95, 0.90, 1.0
        )
        
        manager.register_service("test_operation", ServiceTier.PRIMARY, capability)
        
        assert "test_operation" in manager.service_registry
        assert ServiceTier.PRIMARY in manager.service_registry["test_operation"]
        assert manager.service_registry["test_operation"][ServiceTier.PRIMARY] == capability
    
    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Test successful execution without degradation"""
        manager = GracefulDegradationManager()
        capability = ServiceCapability(
            "test-service", ServiceTier.PRIMARY, 2.0, 0.95, 0.90, 1.0
        )
        manager.register_service("test_operation", ServiceTier.PRIMARY, capability)
        
        call_count = 0
        async def primary_func():
            nonlocal call_count
            call_count += 1
            return "primary_result"
        
        service_functions = {ServiceTier.PRIMARY: primary_func}
        
        result = await manager.execute_with_degradation("test_operation", service_functions)
        assert result == "primary_result"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_degradation_fallback(self):
        """Test degradation to fallback tier"""
        manager = GracefulDegradationManager()
        
        # Register multiple tiers
        primary_cap = ServiceCapability("primary", ServiceTier.PRIMARY, 2.0, 0.95, 0.90, 1.0)
        backup_cap = ServiceCapability("backup", ServiceTier.BACKUP, 3.0, 0.85, 0.95, 0.5)
        
        manager.register_service("test_operation", ServiceTier.PRIMARY, primary_cap)
        manager.register_service("test_operation", ServiceTier.BACKUP, backup_cap)
        
        call_counts = {"primary": 0, "backup": 0}
        
        async def primary_func():
            call_counts["primary"] += 1
            raise Exception("Primary failed")
        
        async def backup_func():
            call_counts["backup"] += 1
            return "backup_result"
        
        service_functions = {
            ServiceTier.PRIMARY: primary_func,
            ServiceTier.BACKUP: backup_func
        }
        
        result = await manager.execute_with_degradation("test_operation", service_functions)
        assert result == "backup_result"
        assert call_counts["primary"] == 1
        assert call_counts["backup"] == 1
    
    @pytest.mark.asyncio
    async def test_all_tiers_failed(self):
        """Test all tiers failing"""
        manager = GracefulDegradationManager()
        
        primary_cap = ServiceCapability("primary", ServiceTier.PRIMARY, 2.0, 0.95, 0.90, 1.0)
        backup_cap = ServiceCapability("backup", ServiceTier.BACKUP, 3.0, 0.85, 0.95, 0.5)
        
        manager.register_service("test_operation", ServiceTier.PRIMARY, primary_cap)
        manager.register_service("test_operation", ServiceTier.BACKUP, backup_cap)
        
        async def failing_func():
            raise Exception("Service failed")
        
        service_functions = {
            ServiceTier.PRIMARY: failing_func,
            ServiceTier.BACKUP: failing_func
        }
        
        with pytest.raises(AllTiersFailedError):
            await manager.execute_with_degradation("test_operation", service_functions)
    
    @pytest.mark.asyncio
    async def test_user_feedback_callback(self):
        """Test user feedback during degradation"""
        manager = GracefulDegradationManager()
        
        primary_cap = ServiceCapability("primary", ServiceTier.PRIMARY, 2.0, 0.95, 0.90, 1.0)
        backup_cap = ServiceCapability("backup", ServiceTier.BACKUP, 3.0, 0.85, 0.95, 0.5)
        
        manager.register_service("test_operation", ServiceTier.PRIMARY, primary_cap)
        manager.register_service("test_operation", ServiceTier.BACKUP, backup_cap)
        
        feedback_messages = []
        
        async def feedback_callback(message):
            feedback_messages.append(message)
        
        async def primary_func():
            raise Exception("Primary failed")
        
        async def backup_func():
            return "backup_result"
        
        service_functions = {
            ServiceTier.PRIMARY: primary_func,
            ServiceTier.BACKUP: backup_func
        }
        
        result = await manager.execute_with_degradation(
            "test_operation", 
            service_functions,
            user_feedback_callback=feedback_callback
        )
        
        assert result == "backup_result"
        assert len(feedback_messages) > 0
        assert "backup" in feedback_messages[0].lower()

class TestStructuredLogging:
    """Test structured logging system"""
    
    def test_voice_ai_log_context(self):
        """Test VoiceAILogContext dataclass"""
        context = VoiceAILogContext(
            request_id="test-123",
            user_session="session-456",
            operation="test_operation",
            service_name="test_service",
            timestamp=time.time(),
            duration_ms=150.0,
            model_used="test-model"
        )
        
        assert context.request_id == "test-123"
        assert context.user_session == "session-456"
        assert context.operation == "test_operation"
        assert context.duration_ms == 150.0
        assert context.model_used == "test-model"
        
        # Test conversion to dict
        context_dict = asdict(context)
        assert isinstance(context_dict, dict)
        assert context_dict["request_id"] == "test-123"
    
    @patch('assistant.structured_logging.logging.FileHandler')
    def test_voice_ai_logger_initialization(self, mock_file_handler):
        """Test VoiceAILogger initialization"""
        with patch('assistant.structured_logging.Path') as mock_path:
            # Create a mock path instance that supports the / operator
            mock_path_instance = Mock()
            mock_path_instance.__truediv__ = Mock(return_value="test_logger.jsonl")
            mock_path.return_value = mock_path_instance
            
            logger = VoiceAILogger("test_logger")
            
            assert logger.name == "test_logger"
            mock_path_instance.mkdir.assert_called_once_with(exist_ok=True)
    
    @patch('assistant.structured_logging.uuid.uuid4')
    def test_log_request_start(self, mock_uuid):
        """Test request start logging"""
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="test-request-123")
        
        with patch('assistant.structured_logging.Path') as mock_path:
            # Create a mock path instance that supports the / operator
            mock_path_instance = Mock()
            mock_path_instance.__truediv__ = Mock(return_value="test_logger.jsonl")
            mock_path.return_value = mock_path_instance
            
            with patch('assistant.structured_logging.logging.FileHandler'):
                logger = VoiceAILogger("test_logger")
                
                # Mock the underlying logger
                logger.logger = Mock()
                
                request_id = logger.log_request_start(
                    operation="test_operation",
                    service_name="test_service",
                    audio_duration_ms=1000.0
                )
                
                assert request_id == "test-request-123"
                logger.logger.info.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_voice_ai_request_context(self):
        """Test voice AI request context manager"""
        with patch('assistant.structured_logging.Path') as mock_path:
            # Create a mock path instance that supports the / operator
            mock_path_instance = Mock()
            mock_path_instance.__truediv__ = Mock(return_value="test_logger.jsonl")
            mock_path.return_value = mock_path_instance
            
            with patch('assistant.structured_logging.logging.FileHandler'):
                logger = VoiceAILogger("test_logger")
                logger.logger = Mock()
                
                async with voice_ai_request_context(
                    "test_operation",
                    "test_service", 
                    logger
                ) as request_id:
                    assert isinstance(request_id, str)
                    # Should log start
                    assert logger.logger.info.call_count >= 1
                
                # Should log success after context exit
                assert logger.logger.info.call_count >= 2
    
    def test_log_audio_processing(self):
        """Test audio processing logging helper"""
        with patch('assistant.structured_logging.Path') as mock_path:
            # Create a mock path instance that supports the / operator
            mock_path_instance = Mock()
            mock_path_instance.__truediv__ = Mock(return_value="test_logger.jsonl")
            mock_path.return_value = mock_path_instance
            
            with patch('assistant.structured_logging.logging.FileHandler'):
                logger = VoiceAILogger("test_logger")
                logger.logger = Mock()
                
                log_audio_processing(logger, 1000.0, 150.0, 0.95)
                
                logger.logger.info.assert_called_once()
                call_args = logger.logger.info.call_args
                assert "structured_data" in call_args[1]
    
    def test_log_model_inference(self):
        """Test model inference logging helper"""
        with patch('assistant.structured_logging.Path'):
            logger = VoiceAILogger("test_logger")
            logger.logger = Mock()
            
            log_model_inference(
                logger, 
                "gpt-4o", 
                100, 
                50, 
                1500.0,
                {"quality": 0.9}
            )
            
            logger.logger.info.assert_called_once()
            call_args = logger.logger.info.call_args
            structured_data = call_args[1]["structured_data"]
            assert structured_data["metrics"]["input_tokens"] == 100
            assert structured_data["metrics"]["output_tokens"] == 50

class TestHealthMonitoring:
    """Test health monitoring system"""
    
    def test_service_health_metrics(self):
        """Test ServiceHealthMetrics dataclass"""
        metrics = ServiceHealthMetrics(
            service_name="test_service",
            status=HealthStatus.HEALTHY,
            response_time_ms=150.0,
            success_rate=0.95,
            error_count=2,
            last_check_time=time.time(),
            last_success_time=time.time(),
            last_error_time=time.time() - 60
        )
        
        assert metrics.service_name == "test_service"
        assert metrics.status == HealthStatus.HEALTHY
        assert metrics.response_time_ms == 150.0
        assert metrics.success_rate == 0.95
        assert metrics.error_count == 2
    
    def test_system_health_metrics(self):
        """Test SystemHealthMetrics dataclass"""
        service_metrics = {
            "stt": ServiceHealthMetrics(
                "stt", HealthStatus.HEALTHY, 100.0, 0.98, 1, 
                time.time(), time.time(), time.time() - 120
            )
        }
        
        metrics = SystemHealthMetrics(
            overall_status=HealthStatus.HEALTHY,
            timestamp=time.time(),
            uptime_seconds=3600.0,
            total_requests=1000,
            error_rate=0.02,
            avg_response_time_ms=200.0,
            memory_usage_mb=512.0,
            cpu_usage_percent=25.0,
            disk_usage_percent=60.0,
            network_connectivity=True,
            services=service_metrics,
            alerts=[]
        )
        
        assert metrics.overall_status == HealthStatus.HEALTHY
        assert metrics.total_requests == 1000
        assert metrics.error_rate == 0.02
        assert len(metrics.services) == 1
        assert len(metrics.alerts) == 0
    
    @pytest.mark.asyncio
    async def test_health_checker(self):
        """Test HealthChecker functionality"""
        call_count = 0
        
        async def mock_health_check():
            nonlocal call_count
            call_count += 1
            return True
        
        with patch('assistant.health_monitoring.get_voice_ai_logger'):
            checker = HealthChecker(
                "test_service",
                mock_health_check,
                check_interval=0.1,
                timeout=1.0
            )
            
            # Test initial state
            assert checker.metrics.status == HealthStatus.UNKNOWN
            assert checker.running is False
            
            # Test single health check
            await checker._perform_health_check()
            assert call_count == 1
            assert checker.metrics.status == HealthStatus.HEALTHY
            assert checker.metrics.success_rate == 1.0
    
    @pytest.mark.asyncio
    async def test_health_checker_failure(self):
        """Test HealthChecker with failures"""
        failure_count = 0
        
        async def failing_health_check():
            nonlocal failure_count
            failure_count += 1
            return False
        
        with patch('assistant.health_monitoring.get_voice_ai_logger'):
            checker = HealthChecker(
                "test_service",
                failing_health_check,
                unhealthy_threshold=0.5
            )
            
            # Perform multiple failed checks
            for _ in range(3):
                await checker._perform_health_check()
            
            assert failure_count == 3
            assert checker.metrics.status == HealthStatus.UNHEALTHY
            assert checker.metrics.success_rate == 0.0
            assert checker.metrics.error_count == 3
    
    def test_system_health_monitor_initialization(self):
        """Test SystemHealthMonitor initialization"""
        monitor = SystemHealthMonitor()
        
        assert len(monitor.service_checkers) == 0
        assert len(monitor.circuit_breakers) == 0
        assert monitor.running is False
        assert monitor.total_requests == 0
        assert monitor.total_errors == 0
    
    def test_system_health_monitor_service_registration(self):
        """Test service registration with health monitor"""
        monitor = SystemHealthMonitor()
        
        async def mock_health_check():
            return True
        
        circuit_breaker = Mock()
        
        monitor.register_service(
            "test_service",
            mock_health_check,
            circuit_breaker
        )
        
        assert "test_service" in monitor.service_checkers
        assert "test_service" in monitor.circuit_breakers
        assert monitor.circuit_breakers["test_service"] == circuit_breaker
    
    def test_record_request_metrics(self):
        """Test request metrics recording"""
        monitor = SystemHealthMonitor()
        
        # Record successful request
        monitor.record_request(150.0, True)
        assert monitor.total_requests == 1
        assert monitor.total_errors == 0
        assert len(monitor.response_times) == 1
        assert monitor.response_times[0] == 150.0
        
        # Record failed request
        monitor.record_request(300.0, False)
        assert monitor.total_requests == 2
        assert monitor.total_errors == 1
        assert len(monitor.response_times) == 2
    
    @pytest.mark.asyncio
    async def test_health_monitored_operation(self):
        """Test health monitored operation context manager"""
        monitor = SystemHealthMonitor()
        
        async with health_monitored_operation("test_operation", monitor):
            await asyncio.sleep(0.1)  # Simulate operation
        
        assert monitor.total_requests == 1
        assert monitor.total_errors == 0
        assert len(monitor.response_times) == 1
        assert monitor.response_times[0] > 50  # Should be > 50ms
    
    @pytest.mark.asyncio
    async def test_health_monitored_operation_failure(self):
        """Test health monitored operation with failure"""
        monitor = SystemHealthMonitor()
        
        with pytest.raises(Exception):
            async with health_monitored_operation("test_operation", monitor):
                raise Exception("Test failure")
        
        assert monitor.total_requests == 1
        assert monitor.total_errors == 1
        assert len(monitor.response_times) == 1

class TestIntegration:
    """Integration tests for complete error handling system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_error_handling(self):
        """Test complete error handling flow"""
        # Setup components
        degradation_manager = GracefulDegradationManager()
        health_monitor = SystemHealthMonitor()
        
        # Setup circuit breaker
        cb_config = CircuitBreakerConfig(failure_threshold=2)
        circuit_breaker = ModernCircuitBreaker("test_service", cb_config)
        
        # Register service with all components
        capability = ServiceCapability(
            "test-service", ServiceTier.PRIMARY, 2.0, 0.95, 0.90, 1.0
        )
        degradation_manager.register_service(
            "test_operation", ServiceTier.PRIMARY, capability, circuit_breaker
        )
        
        backup_capability = ServiceCapability(
            "backup-service", ServiceTier.BACKUP, 3.0, 0.85, 0.95, 0.5
        )
        degradation_manager.register_service(
            "test_operation", ServiceTier.BACKUP, backup_capability
        )
        
        # Test successful operation
        call_counts = {"primary": 0, "backup": 0}
        
        async def primary_func():
            call_counts["primary"] += 1
            return "primary_success"
        
        async def backup_func():
            call_counts["backup"] += 1
            return "backup_success"
        
        service_functions = {
            ServiceTier.PRIMARY: primary_func,
            ServiceTier.BACKUP: backup_func
        }
        
        # Should use primary
        result = await degradation_manager.execute_with_degradation(
            "test_operation", service_functions
        )
        assert result == "primary_success"
        assert call_counts["primary"] == 1
        assert call_counts["backup"] == 0
        
        # Test degradation after circuit breaker opens
        async def failing_primary():
            call_counts["primary"] += 1
            raise Exception("Primary failure")
        
        service_functions[ServiceTier.PRIMARY] = failing_primary
        
        # Cause circuit breaker to open
        for _ in range(2):
            try:
                await circuit_breaker.call(failing_primary)
            except:
                pass
        
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Now degradation should skip primary and use backup
        result = await degradation_manager.execute_with_degradation(
            "test_operation", service_functions
        )
        assert result == "backup_success"
        # Primary shouldn't be called because circuit is open
        assert call_counts["backup"] == 1
    
    @pytest.mark.asyncio
    async def test_structured_logging_with_error_handling(self):
        """Test structured logging integration with error handling"""
        with patch('assistant.structured_logging.Path'):
            logger = VoiceAILogger("test_logger")
            logger.logger = Mock()
            
            # Test logging with error context
            error_context = ErrorContext(
                service_name="test_service",
                operation="test_operation",
                timestamp=time.time(),
                retry_count=2
            )
            
            error = VoiceAIException(
                "Test error",
                ErrorCategory.TRANSIENT,
                context=error_context,
                retryable=True
            )
            
            logger.log_request_error(
                error,
                duration_ms=250.0,
                retry_count=2,
                circuit_breaker_state="open",
                fallback_tier="backup"
            )
            
            logger.logger.error.assert_called_once()
            call_args = logger.logger.error.call_args
            structured_data = call_args[1]["structured_data"]
            
            assert structured_data["retry_count"] == 2
            assert structured_data["circuit_breaker_state"] == "open"
            assert structured_data["fallback_tier"] == "backup"
            assert structured_data["error_category"] == "transient" 