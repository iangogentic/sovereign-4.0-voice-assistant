"""
Unit tests for FallbackDetector

Tests all detection criteria, circuit breaker patterns, health checks,
and sensitivity levels for the intelligent fallback system.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from collections import deque
import aiohttp
from aioresponses import aioresponses

from assistant.fallback_detector import (
    FallbackDetector, FallbackConfig, FallbackMetrics, 
    FallbackTrigger, SensitivityLevel, CircuitBreakerState,
    create_fallback_detector
)


class TestFallbackConfig:
    """Test FallbackConfig dataclass and sensitivity adjustments"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = FallbackConfig()
        
        assert config.max_network_latency_ms == 500
        assert config.max_connection_failures_per_window == 3
        assert config.connection_failure_window_minutes == 5
        assert config.min_audio_snr_db == 10.0
        assert config.sensitivity_level == SensitivityLevel.BALANCED
    
    def test_conservative_sensitivity_adjustments(self):
        """Test conservative sensitivity adjustments"""
        config = FallbackConfig(sensitivity_level=SensitivityLevel.CONSERVATIVE)
        detector = FallbackDetector(config)
        
        # Conservative should have higher thresholds (slower fallback)
        assert detector.config.max_network_latency_ms == 750  # 500 * 1.5
        assert detector.config.max_connection_failures_per_window == 5  # 3 + 2
        assert detector.config.min_audio_snr_db == 7.0  # 10.0 * 0.7
        assert detector.config.max_api_errors_per_window == 7  # 5 + 2
    
    def test_aggressive_sensitivity_adjustments(self):
        """Test aggressive sensitivity adjustments"""
        config = FallbackConfig(sensitivity_level=SensitivityLevel.AGGRESSIVE)
        detector = FallbackDetector(config)
        
        # Aggressive should have lower thresholds (faster fallback)
        assert detector.config.max_network_latency_ms == 350  # 500 * 0.7
        assert detector.config.max_connection_failures_per_window == 2  # max(1, 3-1)
        assert detector.config.min_audio_snr_db == 13.0  # 10.0 * 1.3
        assert detector.config.max_api_errors_per_window == 3  # max(2, 5-2)


class TestFallbackMetrics:
    """Test FallbackMetrics tracking"""
    
    def test_metrics_initialization(self):
        """Test metrics are properly initialized"""
        metrics = FallbackMetrics()
        
        assert len(metrics.network_latency_samples) == 0
        assert len(metrics.connection_failures) == 0
        assert len(metrics.api_errors) == 0
        assert len(metrics.audio_quality_samples) == 0
        assert metrics.complex_queries_detected == 0
        assert metrics.total_fallbacks_triggered == 0
        assert metrics.current_streak_successful == 0
        assert metrics.realtime_api_success_rate == 1.0
        assert metrics.traditional_pipeline_success_rate == 1.0
    
    def test_deque_max_lengths(self):
        """Test that deques respect max lengths"""
        metrics = FallbackMetrics()
        
        # Test network latency samples
        for i in range(60):  # More than maxlen=50
            metrics.network_latency_samples.append(i)
        assert len(metrics.network_latency_samples) == 50
        assert list(metrics.network_latency_samples) == list(range(10, 60))
        
        # Test connection failures
        for i in range(25):  # More than maxlen=20
            metrics.connection_failures.append(i)
        assert len(metrics.connection_failures) == 20
        assert list(metrics.connection_failures) == list(range(5, 25))


class TestFallbackDetector:
    """Test main FallbackDetector functionality"""
    
    @pytest.fixture
    def detector(self):
        """Create a FallbackDetector instance for testing"""
        config = FallbackConfig()
        return FallbackDetector(config)
    
    @pytest.fixture
    def mock_detector_with_session(self):
        """Create detector with mocked HTTP session"""
        config = FallbackConfig()
        detector = FallbackDetector(config)
        detector.http_session = Mock(spec=aiohttp.ClientSession)
        return detector
    
    def test_initialization(self, detector):
        """Test detector initialization"""
        assert detector.circuit_state == CircuitBreakerState.CLOSED
        assert detector.current_mode == "realtime"
        assert detector.is_monitoring == False
        assert detector.recovery_attempts == 0
        assert detector.current_backoff_delay == 60.0  # base delay
    
    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful initialization"""
        detector = FallbackDetector(FallbackConfig())
        
        with patch('aiohttp.ClientSession') as mock_session:
            result = await detector.initialize()
            assert result == True
            assert detector.http_session is not None
            mock_session.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_failure(self):
        """Test initialization failure"""
        detector = FallbackDetector(FallbackConfig())
        
        with patch('aiohttp.ClientSession', side_effect=Exception("Connection error")):
            result = await detector.initialize()
            assert result == False
            assert detector.http_session is None


class TestDetectionCriteria:
    """Test individual detection criteria"""
    
    @pytest.fixture
    def detector(self):
        config = FallbackConfig()
        return FallbackDetector(config)
    
    def test_network_latency_detection(self, detector):
        """Test network latency threshold detection"""
        # Not enough samples
        assert detector._check_network_latency() == False
        
        # Add samples below threshold
        for latency in [100, 200, 300]:
            detector.metrics.network_latency_samples.append(latency)
        assert detector._check_network_latency() == False
        
        # Add samples that push average above threshold (500ms)
        detector.metrics.network_latency_samples.extend([600, 700, 800])
        assert detector._check_network_latency() == True
    
    def test_connection_failures_detection(self, detector):
        """Test connection failures within time window"""
        current_time = time.time()
        
        # No failures
        assert detector._check_connection_failures() == False
        
        # Add failures outside window (older than 5 minutes)
        old_time = current_time - (6 * 60)  # 6 minutes ago
        for _ in range(5):
            detector.metrics.connection_failures.append(old_time)
        assert detector._check_connection_failures() == False
        
        # Add failures within window (exceeds threshold of 3)
        recent_time = current_time - (2 * 60)  # 2 minutes ago
        for _ in range(4):
            detector.metrics.connection_failures.append(recent_time)
        assert detector._check_connection_failures() == True
    
    def test_api_errors_detection(self, detector):
        """Test API errors within time window"""
        current_time = time.time()
        
        # No errors
        assert detector._check_api_errors() == False
        
        # Add errors outside window
        old_time = current_time - (6 * 60)  # 6 minutes ago
        for _ in range(10):
            detector.metrics.api_errors.append(old_time)
        assert detector._check_api_errors() == False
        
        # Add errors within window (exceeds threshold of 5)
        recent_time = current_time - (2 * 60)  # 2 minutes ago
        for _ in range(6):
            detector.metrics.api_errors.append(recent_time)
        assert detector._check_api_errors() == True
    
    def test_audio_quality_detection(self, detector):
        """Test audio quality threshold detection"""
        # Not enough samples
        assert detector._check_audio_quality() == False
        
        # Add samples above threshold (10dB)
        for snr in [15, 12, 14, 11, 13]:
            detector.metrics.audio_quality_samples.append(snr)
        assert detector._check_audio_quality() == False
        
        # Add samples that push average below threshold
        detector.metrics.audio_quality_samples.extend([5, 6, 7, 8, 4])
        assert detector._check_audio_quality() == True
    
    def test_complex_query_detection(self, detector):
        """Test complex query pattern detection"""
        # Short query
        assert detector.detect_complex_query("hello") == False
        
        # Simple long query
        simple_query = "What's the weather like today? It's a beautiful day outside."
        assert detector.detect_complex_query(simple_query) == False
        
        # Complex query with keywords
        complex_query = "Can you analyze this data and explain the detailed technical implementation?"
        assert detector.detect_complex_query(complex_query) == True
        assert detector.metrics.complex_queries_detected == 1
        
        # Technical query with patterns
        tech_query = "How do I implement a function that uses a specific algorithm for processing?"
        assert detector.detect_complex_query(tech_query) == True
        assert detector.metrics.complex_queries_detected == 2
    
    def test_should_use_fallback_comprehensive(self, detector):
        """Test comprehensive fallback evaluation"""
        # Initial state - no fallback needed
        should_fallback, triggers, metrics = detector.should_use_fallback()
        assert should_fallback == False
        assert len(triggers) == 0
        
        # Trigger network latency
        for latency in [600, 700, 800]:
            detector.metrics.network_latency_samples.append(latency)
        
        should_fallback, triggers, metrics = detector.should_use_fallback()
        assert should_fallback == True
        assert FallbackTrigger.NETWORK_LATENCY in triggers
        assert 'avg_latency' in metrics
        
        # Circuit breaker open state
        detector.circuit_state = CircuitBreakerState.OPEN
        should_fallback, triggers, metrics = detector.should_use_fallback()
        assert should_fallback == True


class TestCircuitBreaker:
    """Test circuit breaker pattern"""
    
    @pytest.fixture
    def detector(self):
        config = FallbackConfig()
        return FallbackDetector(config)
    
    @pytest.mark.asyncio
    async def test_trigger_fallback(self, detector):
        """Test fallback triggering"""
        triggers = [FallbackTrigger.NETWORK_LATENCY]
        metrics = {'avg_latency': 600}
        
        # Mock callbacks
        detector.on_fallback_triggered = AsyncMock()
        detector.on_mode_switched = AsyncMock()
        
        await detector.trigger_fallback(triggers, metrics)
        
        assert detector.circuit_state == CircuitBreakerState.OPEN
        assert detector.current_mode == "traditional"
        assert detector.metrics.total_fallbacks_triggered == 1
        assert detector.metrics.current_streak_successful == 0
        
        # Verify callbacks were called
        detector.on_fallback_triggered.assert_called_once_with(triggers, metrics)
        detector.on_mode_switched.assert_called_once_with("traditional", ["network_latency"])
    
    @pytest.mark.asyncio
    async def test_attempt_recovery_too_early(self, detector):
        """Test recovery attempt before backoff period"""
        detector.circuit_state = CircuitBreakerState.OPEN
        detector.circuit_opened_time = time.time()  # Just opened
        
        result = await detector.attempt_recovery()
        assert result == False
        assert detector.circuit_state == CircuitBreakerState.OPEN
    
    @pytest.mark.asyncio
    async def test_attempt_recovery_success(self, detector):
        """Test successful recovery"""
        detector.circuit_state = CircuitBreakerState.OPEN
        detector.circuit_opened_time = time.time() - 70  # More than 60s ago
        detector.recovery_attempts = 0
        
        # Mock successful health check
        detector._perform_health_check = AsyncMock(return_value=True)
        detector.on_mode_switched = AsyncMock()
        
        result = await detector.attempt_recovery()
        
        assert result == True
        assert detector.circuit_state == CircuitBreakerState.CLOSED
        assert detector.current_mode == "realtime"
        assert detector.recovery_attempts == 0
        assert detector.current_backoff_delay == 60.0  # Reset to base
        
        detector.on_mode_switched.assert_called_once_with("realtime", ["recovery_successful"])
    
    @pytest.mark.asyncio
    async def test_attempt_recovery_failure(self, detector):
        """Test failed recovery with backoff increase"""
        detector.circuit_state = CircuitBreakerState.OPEN
        detector.circuit_opened_time = time.time() - 70
        detector.current_backoff_delay = 60.0
        
        # Mock failed health check
        detector._perform_health_check = AsyncMock(return_value=False)
        
        result = await detector.attempt_recovery()
        
        assert result == False
        assert detector.circuit_state == CircuitBreakerState.OPEN
        assert detector.current_backoff_delay == 120.0  # 60 * 2
    
    @pytest.mark.asyncio
    async def test_max_recovery_attempts(self, detector):
        """Test max recovery attempts limit"""
        detector.circuit_state = CircuitBreakerState.OPEN
        detector.circuit_opened_time = time.time() - 70
        detector.recovery_attempts = 5  # At max
        
        result = await detector.attempt_recovery()
        assert result == False
        assert detector.recovery_attempts == 5  # Unchanged


class TestHealthChecks:
    """Test health check functionality"""
    
    @pytest.mark.asyncio
    async def test_perform_health_check_success(self):
        """Test successful health check"""
        config = FallbackConfig()
        detector = FallbackDetector(config)
        
        with aioresponses() as mock_responses:
            # Mock successful responses for all endpoints
            for endpoint in config.openai_health_endpoints:
                mock_responses.get(endpoint, status=200)
            
            # Create real session for test
            detector.http_session = aiohttp.ClientSession()
            
            try:
                result = await detector._perform_health_check()
                assert result == True
                assert detector.last_health_check_result['overall_healthy'] == True
                
                # Verify latency was recorded
                assert len(detector.metrics.network_latency_samples) > 0
                
            finally:
                await detector.http_session.close()
    
    @pytest.mark.asyncio
    async def test_perform_health_check_failure(self):
        """Test failed health check"""
        config = FallbackConfig()
        detector = FallbackDetector(config)
        
        with aioresponses() as mock_responses:
            # Mock failed responses
            for endpoint in config.openai_health_endpoints:
                mock_responses.get(endpoint, status=500)
            
            detector.http_session = aiohttp.ClientSession()
            detector.on_health_check_failed = AsyncMock()
            
            try:
                result = await detector._perform_health_check()
                assert result == False
                assert detector.last_health_check_result['overall_healthy'] == False
                
                # Verify error was recorded
                assert len(detector.metrics.api_errors) > 0
                
                # Verify callback was called
                detector.on_health_check_failed.assert_called_once()
                
            finally:
                await detector.http_session.close()
    
    @pytest.mark.asyncio 
    async def test_health_check_loop_cancellation(self):
        """Test health check loop can be cancelled"""
        detector = FallbackDetector(FallbackConfig())
        detector.http_session = Mock()
        detector._perform_health_check = AsyncMock()
        
        # Start monitoring
        await detector.start_monitoring()
        assert detector.is_monitoring == True
        assert detector.health_check_task is not None
        
        # Stop monitoring
        await detector.stop_monitoring()
        assert detector.is_monitoring == False
        assert detector.health_check_task.cancelled()


class TestMetricRecording:
    """Test metric recording methods"""
    
    @pytest.fixture
    def detector(self):
        return FallbackDetector(FallbackConfig())
    
    def test_record_network_latency(self, detector):
        """Test network latency recording"""
        detector.record_network_latency(250.5)
        
        assert len(detector.metrics.network_latency_samples) == 1
        assert detector.metrics.network_latency_samples[0] == 250.5
    
    def test_record_connection_failure(self, detector):
        """Test connection failure recording"""
        initial_time = time.time()
        detector.record_connection_failure()
        
        assert len(detector.metrics.connection_failures) == 1
        recorded_time = detector.metrics.connection_failures[0]
        assert abs(recorded_time - initial_time) < 1.0  # Within 1 second
    
    def test_record_api_error(self, detector):
        """Test API error recording"""
        initial_time = time.time()
        detector.record_api_error("rate_limit")
        
        assert len(detector.metrics.api_errors) == 1
        recorded_time = detector.metrics.api_errors[0]
        assert abs(recorded_time - initial_time) < 1.0
    
    def test_record_audio_quality(self, detector):
        """Test audio quality recording"""
        detector.record_audio_quality(12.5)
        
        assert len(detector.metrics.audio_quality_samples) == 1
        assert detector.metrics.audio_quality_samples[0] == 12.5
    
    def test_record_successful_operation(self, detector):
        """Test successful operation recording"""
        # Test realtime operation
        detector.record_successful_operation("realtime", 0.25)
        assert detector.metrics.current_streak_successful == 1
        
        # Test traditional operation
        detector.record_successful_operation("traditional", 0.8)
        assert detector.metrics.current_streak_successful == 2
        assert detector.metrics.average_response_time_traditional == 0.8


class TestUtilityMethods:
    """Test utility and helper methods"""
    
    @pytest.fixture
    def detector(self):
        return FallbackDetector(FallbackConfig())
    
    def test_get_average_latency(self, detector):
        """Test average latency calculation"""
        # Empty samples
        assert detector._get_average_latency() == 0.0
        
        # With samples
        detector.metrics.network_latency_samples.extend([100, 200, 300])
        assert detector._get_average_latency() == 200.0
    
    def test_get_average_audio_quality(self, detector):
        """Test average audio quality calculation"""
        # Empty samples
        assert detector._get_average_audio_quality() == 0.0
        
        # With samples
        detector.metrics.audio_quality_samples.extend([10, 12, 14])
        assert detector._get_average_audio_quality() == 12.0
    
    def test_get_status_summary(self, detector):
        """Test comprehensive status summary"""
        # Add some test data
        detector.metrics.network_latency_samples.extend([100, 200])
        detector.metrics.complex_queries_detected = 3
        detector.circuit_state = CircuitBreakerState.OPEN
        
        summary = detector.get_status_summary()
        
        assert summary['circuit_state'] == 'open'
        assert summary['current_mode'] == 'realtime'
        assert summary['sensitivity_level'] == 'balanced'
        assert summary['metrics']['avg_latency_ms'] == 150.0
        assert summary['metrics']['complex_queries_detected'] == 3
        assert 'thresholds' in summary
        assert 'recovery_info' in summary
    
    @pytest.mark.asyncio
    async def test_cleanup(self, detector):
        """Test cleanup method"""
        # Setup session and monitoring
        detector.http_session = Mock()
        detector.http_session.close = AsyncMock()
        
        await detector.start_monitoring()
        assert detector.is_monitoring == True
        
        # Cleanup
        await detector.cleanup()
        
        assert detector.is_monitoring == False
        detector.http_session.close.assert_called_once()


class TestFactoryFunction:
    """Test factory function"""
    
    def test_create_fallback_detector_default(self):
        """Test factory function with default settings"""
        detector = create_fallback_detector()
        
        assert isinstance(detector, FallbackDetector)
        assert detector.config.sensitivity_level == SensitivityLevel.BALANCED
        assert detector.circuit_state == CircuitBreakerState.CLOSED
    
    def test_create_fallback_detector_aggressive(self):
        """Test factory function with aggressive sensitivity"""
        detector = create_fallback_detector(SensitivityLevel.AGGRESSIVE)
        
        assert detector.config.sensitivity_level == SensitivityLevel.AGGRESSIVE
        # Verify aggressive adjustments were applied
        assert detector.config.max_network_latency_ms == 350  # 500 * 0.7


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_full_fallback_cycle(self):
        """Test complete fallback and recovery cycle"""
        config = FallbackConfig(base_backoff_delay_seconds=1)  # Fast backoff for testing
        detector = FallbackDetector(config)
        
        # Mock callbacks
        fallback_called = False
        recovery_called = False
        mode_switches = []
        
        async def on_fallback(triggers, metrics):
            nonlocal fallback_called
            fallback_called = True
        
        async def on_recovery(attempt):
            nonlocal recovery_called
            recovery_called = True
        
        async def on_mode_switch(mode, triggers):
            mode_switches.append((mode, triggers))
        
        detector.on_fallback_triggered = on_fallback
        detector.on_recovery_attempted = on_recovery
        detector.on_mode_switched = on_mode_switch
        
        # Mock health check for recovery
        detector._perform_health_check = AsyncMock(return_value=True)
        
        # 1. Trigger fallback due to high latency
        for latency in [600, 700, 800]:
            detector.record_network_latency(latency)
        
        should_fallback, triggers, metrics = detector.should_use_fallback()
        assert should_fallback == True
        assert FallbackTrigger.NETWORK_LATENCY in triggers
        
        # 2. Execute fallback
        await detector.trigger_fallback(triggers, metrics)
        assert detector.circuit_state == CircuitBreakerState.OPEN
        assert detector.current_mode == "traditional"
        assert fallback_called == True
        
        # 3. Wait for backoff period and attempt recovery
        await asyncio.sleep(1.1)  # Wait longer than backoff
        
        recovery_result = await detector.attempt_recovery()
        assert recovery_result == True
        assert detector.circuit_state == CircuitBreakerState.CLOSED
        assert detector.current_mode == "realtime"
        assert recovery_called == True
        
        # Verify mode switches
        assert len(mode_switches) == 2
        assert mode_switches[0] == ("traditional", ["network_latency"])
        assert mode_switches[1] == ("realtime", ["recovery_successful"])
    
    @pytest.mark.asyncio
    async def test_multiple_criteria_triggering(self):
        """Test multiple criteria triggering fallback simultaneously"""
        detector = FallbackDetector(FallbackConfig())
        
        # Trigger multiple criteria
        # 1. Network latency
        for latency in [600, 700, 800]:
            detector.record_network_latency(latency)
        
        # 2. Audio quality
        for snr in [5, 6, 7, 8, 4, 3, 2]:
            detector.record_audio_quality(snr)
        
        # 3. Connection failures
        current_time = time.time()
        for _ in range(4):  # Exceeds threshold of 3
            detector.record_connection_failure()
        
        should_fallback, triggers, metrics = detector.should_use_fallback()
        
        assert should_fallback == True
        assert FallbackTrigger.NETWORK_LATENCY in triggers
        assert FallbackTrigger.AUDIO_QUALITY in triggers
        assert FallbackTrigger.CONNECTION_FAILURES in triggers
        assert len(triggers) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 