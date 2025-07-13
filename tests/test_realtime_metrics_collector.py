"""
Comprehensive Test Suite for RealtimeMetricsCollector

Tests all components and functionality of the Realtime API metrics collection system
including latency tracking, connection monitoring, cost calculation, and integration
with existing monitoring infrastructure.
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import statistics
import json

# Import the components to test
from assistant.realtime_metrics_collector import (
    RealtimeMetricsCollector,
    RealtimeLatencyMetrics,
    RealtimeConnectionMetrics,
    RealtimeAudioMetrics, 
    RealtimeCostMetrics,
    ConnectionState,
    RealtimeMetricType,
    create_realtime_metrics_collector
)

# Import supporting components for integration tests
from assistant.metrics_collector import MetricsCollector, ComponentType
from assistant.monitoring import PerformanceMonitor
from assistant.health_monitoring import SystemHealthMonitor


class TestRealtimeLatencyMetrics:
    """Test the RealtimeLatencyMetrics dataclass"""
    
    def test_initialization(self):
        """Test proper initialization of latency metrics"""
        metrics = RealtimeLatencyMetrics()
        
        assert len(metrics.voice_to_voice_latency_ms) == 0
        assert len(metrics.audio_processing_latency_ms) == 0
        assert len(metrics.text_generation_latency_ms) == 0
        assert len(metrics.total_round_trip_ms) == 0
    
    def test_add_latency(self):
        """Test adding latency measurements"""
        metrics = RealtimeLatencyMetrics()
        
        # Add voice-to-voice latency
        metrics.add_latency("voice_to_voice_latency_ms", 250.0)
        metrics.add_latency("voice_to_voice_latency_ms", 300.0)
        
        assert len(metrics.voice_to_voice_latency_ms) == 2
        assert list(metrics.voice_to_voice_latency_ms) == [250.0, 300.0]
    
    def test_add_invalid_latency_type(self):
        """Test handling of invalid latency types"""
        metrics = RealtimeLatencyMetrics()
        
        # Should not crash on invalid attribute
        metrics.add_latency("invalid_metric", 100.0)
        
        # Should still work for valid metrics
        metrics.add_latency("voice_to_voice_latency_ms", 200.0)
        assert len(metrics.voice_to_voice_latency_ms) == 1
    
    def test_percentile_calculation(self):
        """Test percentile calculations"""
        metrics = RealtimeLatencyMetrics()
        
        # Add test data
        test_values = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        for value in test_values:
            metrics.add_latency("voice_to_voice_latency_ms", float(value))
        
        percentiles = metrics.get_percentiles("voice_to_voice_latency_ms")
        
        assert "p50" in percentiles
        assert "p95" in percentiles
        assert "p99" in percentiles
        
        # Check reasonable values
        assert 400 <= percentiles["p50"] <= 600  # Should be around median
        assert percentiles["p95"] >= percentiles["p50"]
        assert percentiles["p99"] >= percentiles["p95"]
    
    def test_percentile_calculation_insufficient_data(self):
        """Test percentile calculation with insufficient data"""
        metrics = RealtimeLatencyMetrics()
        
        # No data
        percentiles = metrics.get_percentiles("voice_to_voice_latency_ms")
        assert percentiles == {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        
        # One data point
        metrics.add_latency("voice_to_voice_latency_ms", 100.0)
        percentiles = metrics.get_percentiles("voice_to_voice_latency_ms")
        assert percentiles == {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    
    def test_deque_maxlen_behavior(self):
        """Test that deques properly limit their size"""
        metrics = RealtimeLatencyMetrics()
        
        # Add more than maxlen items (maxlen=1000)
        for i in range(1100):
            metrics.add_latency("voice_to_voice_latency_ms", float(i))
        
        # Should be limited to 1000
        assert len(metrics.voice_to_voice_latency_ms) == 1000
        
        # Should contain the most recent values
        values = list(metrics.voice_to_voice_latency_ms)
        assert values[0] == 100.0  # First value should be 100 (0-99 were evicted)
        assert values[-1] == 1099.0  # Last value should be 1099


class TestRealtimeConnectionMetrics:
    """Test the RealtimeConnectionMetrics dataclass"""
    
    def test_initialization(self):
        """Test proper initialization"""
        metrics = RealtimeConnectionMetrics()
        
        assert metrics.connection_state == ConnectionState.DISCONNECTED
        assert metrics.connection_start_time is None
        assert metrics.total_connections == 0
        assert metrics.successful_connections == 0
        assert metrics.failed_connections == 0
    
    def test_record_successful_connection(self):
        """Test recording successful connection"""
        metrics = RealtimeConnectionMetrics()
        
        start_time = time.time()
        metrics.record_connection_attempt(True)
        
        assert metrics.total_connections == 1
        assert metrics.successful_connections == 1
        assert metrics.failed_connections == 0
        assert metrics.connection_state == ConnectionState.CONNECTED
        assert metrics.connection_start_time is not None
        assert abs(metrics.connection_start_time - start_time) < 1.0
    
    def test_record_failed_connection(self):
        """Test recording failed connection"""
        metrics = RealtimeConnectionMetrics()
        
        metrics.record_connection_attempt(False)
        
        assert metrics.total_connections == 1
        assert metrics.successful_connections == 0
        assert metrics.failed_connections == 1
        assert metrics.connection_state == ConnectionState.DISCONNECTED
        assert metrics.connection_start_time is None
    
    def test_record_disconnection(self):
        """Test recording disconnection events"""
        metrics = RealtimeConnectionMetrics()
        
        # First establish a connection
        metrics.record_connection_attempt(True)
        initial_uptime = metrics.total_uptime_seconds
        
        # Wait a bit and disconnect
        time.sleep(0.1)
        metrics.record_disconnection("test_reason")
        
        assert metrics.connection_state == ConnectionState.DISCONNECTED
        assert metrics.connection_start_time is None
        assert metrics.total_uptime_seconds > initial_uptime
        assert len(metrics.disconnection_events) == 1
        
        # Check disconnection event details
        event = metrics.disconnection_events[0]
        assert event["reason"] == "test_reason"
        assert event["duration"] > 0
    
    def test_connection_success_rate(self):
        """Test connection success rate calculation"""
        metrics = RealtimeConnectionMetrics()
        
        # No connections yet
        assert metrics.connection_success_rate == 0.0
        
        # Add some successful and failed connections
        metrics.record_connection_attempt(True)   # 1/1 = 100%
        assert metrics.connection_success_rate == 100.0
        
        metrics.record_connection_attempt(False)  # 1/2 = 50%
        assert metrics.connection_success_rate == 50.0
        
        metrics.record_connection_attempt(True)   # 2/3 = 66.67%
        assert abs(metrics.connection_success_rate - 66.67) < 0.1
    
    def test_average_session_duration(self):
        """Test average session duration calculation"""
        metrics = RealtimeConnectionMetrics()
        
        # No sessions yet
        assert metrics.average_session_duration == 0.0
        
        # Simulate a few sessions
        for duration in [10.0, 20.0, 30.0]:
            metrics.connection_start_time = time.time() - duration
            metrics.record_disconnection("test")
        
        # Should be average of 10, 20, 30 = 20
        assert abs(metrics.average_session_duration - 20.0) < 0.1


class TestRealtimeAudioMetrics:
    """Test the RealtimeAudioMetrics dataclass"""
    
    def test_initialization(self):
        """Test proper initialization"""
        metrics = RealtimeAudioMetrics()
        
        assert metrics.audio_samples_processed == 0
        assert len(metrics.audio_quality_scores) == 0
        assert len(metrics.volume_levels) == 0
        assert metrics.sample_rate == 24000
        assert metrics.bit_depth == 16
    
    def test_record_audio_quality(self):
        """Test recording audio quality metrics"""
        metrics = RealtimeAudioMetrics()
        
        metrics.record_audio_quality(0.95, 0.7)
        
        assert metrics.audio_samples_processed == 1
        assert len(metrics.audio_quality_scores) == 1
        assert len(metrics.volume_levels) == 1
        assert metrics.audio_quality_scores[0] == 0.95
        assert metrics.volume_levels[0] == 0.7
    
    def test_average_audio_quality(self):
        """Test average audio quality calculation"""
        metrics = RealtimeAudioMetrics()
        
        # No data
        assert metrics.average_audio_quality == 0.0
        
        # Add some quality scores
        scores = [0.8, 0.9, 0.85, 0.95]
        for score in scores:
            metrics.record_audio_quality(score)
        
        expected_avg = statistics.mean(scores)
        assert abs(metrics.average_audio_quality - expected_avg) < 0.001


class TestRealtimeCostMetrics:
    """Test the RealtimeCostMetrics dataclass"""
    
    def test_initialization(self):
        """Test proper initialization"""
        metrics = RealtimeCostMetrics()
        
        assert metrics.total_input_tokens == 0
        assert metrics.total_output_tokens == 0
        assert metrics.total_api_calls == 0
        assert len(metrics.session_costs) == 0
        assert metrics.daily_budget == 50.0
    
    def test_add_usage(self):
        """Test adding token usage and cost calculation"""
        metrics = RealtimeCostMetrics()
        
        # Add usage: 1000 input, 500 output tokens
        metrics.add_usage(1000, 500)
        
        assert metrics.total_input_tokens == 1000
        assert metrics.total_output_tokens == 500
        assert metrics.total_api_calls == 1
        assert len(metrics.session_costs) == 1
        
        # Check cost calculation
        session = metrics.session_costs[0]
        expected_cost = (1000 / 1000) * 0.006 + (500 / 1000) * 0.024
        assert abs(session["cost"] - expected_cost) < 0.0001
    
    def test_total_cost_calculation(self):
        """Test total cost calculation across sessions"""
        metrics = RealtimeCostMetrics()
        
        # Add multiple sessions
        metrics.add_usage(1000, 500)  # $0.006 + $0.012 = $0.018
        metrics.add_usage(2000, 1000) # $0.012 + $0.024 = $0.036
        
        expected_total = 0.018 + 0.036  # $0.054
        assert abs(metrics.total_cost - expected_total) < 0.0001
    
    def test_tokens_per_dollar(self):
        """Test tokens per dollar calculation"""
        metrics = RealtimeCostMetrics()
        
        # No usage yet
        assert metrics.tokens_per_dollar == 0.0
        
        # Add usage
        metrics.add_usage(1000, 500)  # 1500 total tokens
        
        expected_ratio = 1500 / metrics.total_cost
        assert abs(metrics.tokens_per_dollar - expected_ratio) < 0.1


class TestRealtimeMetricsCollector:
    """Test the main RealtimeMetricsCollector class"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for testing"""
        mock_config = Mock()
        mock_metrics_collector = Mock(spec=MetricsCollector)
        mock_performance_monitor = Mock(spec=PerformanceMonitor)
        mock_logger = Mock()
        
        return {
            "realtime_config": mock_config,
            "metrics_collector": mock_metrics_collector,
            "performance_monitor": mock_performance_monitor,
            "logger": mock_logger
        }
    
    def test_initialization(self, mock_dependencies):
        """Test proper initialization of RealtimeMetricsCollector"""
        collector = RealtimeMetricsCollector(**mock_dependencies)
        
        assert collector.realtime_config == mock_dependencies["realtime_config"]
        assert collector.metrics_collector == mock_dependencies["metrics_collector"]
        assert collector.performance_monitor == mock_dependencies["performance_monitor"]
        assert collector.logger == mock_dependencies["logger"]
        
        # Check that metrics objects are created
        assert isinstance(collector.latency_metrics, RealtimeLatencyMetrics)
        assert isinstance(collector.connection_metrics, RealtimeConnectionMetrics)
        assert isinstance(collector.audio_metrics, RealtimeAudioMetrics)
        assert isinstance(collector.cost_metrics, RealtimeCostMetrics)
        
        # Check initial state
        assert collector.current_session_id is None
        assert collector.session_start_time is None
        assert len(collector.active_sessions) == 0
    
    def test_session_lifecycle(self, mock_dependencies):
        """Test session start and end lifecycle"""
        collector = RealtimeMetricsCollector(**mock_dependencies)
        
        # Start a session
        session_id = "test-session-123"
        collector.start_session(session_id)
        
        assert collector.current_session_id == session_id
        assert collector.session_start_time is not None
        assert session_id in collector.active_sessions
        
        session_data = collector.active_sessions[session_id]
        assert "start_time" in session_data
        assert "input_tokens" in session_data
        assert "latencies" in session_data
        
        # End the session
        metrics = collector.end_session(session_id)
        
        assert collector.current_session_id is None
        assert collector.session_start_time is None
        assert session_id not in collector.active_sessions
        
        # Check returned metrics
        assert metrics["session_id"] == session_id
        assert "duration_seconds" in metrics
        assert "total_tokens" in metrics
    
    def test_record_voice_latency(self, mock_dependencies):
        """Test recording voice latency measurements"""
        collector = RealtimeMetricsCollector(**mock_dependencies)
        
        # Start a session
        collector.start_session("test-session")
        
        # Record some latencies
        latencies = [250.0, 300.0, 280.0]
        for latency in latencies:
            collector.record_voice_latency(latency)
        
        # Check latency metrics
        assert len(collector.latency_metrics.voice_to_voice_latency_ms) == len(latencies)
        
        # Check session data
        session = collector.active_sessions["test-session"]
        assert len(session["latencies"]) == len(latencies)
        assert session["latencies"] == latencies
    
    def test_record_voice_latency_alerts(self, mock_dependencies):
        """Test that high latency triggers alerts"""
        collector = RealtimeMetricsCollector(**mock_dependencies)
        
        # Mock the alert trigger method
        collector._trigger_alert = Mock()
        
        # Record normal latency (should not trigger alert)
        collector.record_voice_latency(300.0)
        collector._trigger_alert.assert_not_called()
        
        # Record high latency (should trigger alert)
        collector.record_voice_latency(600.0)  # Above 500ms threshold
        collector._trigger_alert.assert_called_once()
        
        # Check alert call details
        args, kwargs = collector._trigger_alert.call_args
        assert args[0] == "high_latency"
        assert args[1]["latency_ms"] == 600.0
    
    def test_record_connection_events(self, mock_dependencies):
        """Test recording connection events"""
        collector = RealtimeMetricsCollector(**mock_dependencies)
        
        # Test successful connection
        collector.record_connection_event("connection_attempt", True)
        
        assert collector.connection_metrics.total_connections == 1
        assert collector.connection_metrics.successful_connections == 1
        assert collector.connection_metrics.connection_state == ConnectionState.CONNECTED
        
        # Test disconnection
        collector.record_connection_event("disconnection", reason="network_error")
        
        assert collector.connection_metrics.connection_state == ConnectionState.DISCONNECTED
        assert len(collector.connection_metrics.disconnection_events) == 1
        assert collector.connection_metrics.disconnection_events[0]["reason"] == "network_error"
    
    def test_record_audio_metrics(self, mock_dependencies):
        """Test recording audio metrics"""
        collector = RealtimeMetricsCollector(**mock_dependencies)
        
        # Start a session
        collector.start_session("test-session")
        
        # Record audio metrics
        collector.record_audio_metrics(0.95, 0.8, "processing")
        
        # Check audio metrics
        assert collector.audio_metrics.audio_samples_processed == 1
        assert len(collector.audio_metrics.audio_quality_scores) == 1
        assert collector.audio_metrics.audio_quality_scores[0] == 0.95
        
        # Check session data
        session = collector.active_sessions["test-session"]
        assert len(session["audio_events"]) == 1
        
        event = session["audio_events"][0]
        assert event["quality"] == 0.95
        assert event["volume"] == 0.8
        assert event["type"] == "processing"
    
    def test_record_token_usage(self, mock_dependencies):
        """Test recording token usage and cost calculation"""
        collector = RealtimeMetricsCollector(**mock_dependencies)
        
        # Start a session
        collector.start_session("test-session")
        
        # Mock alert trigger
        collector._trigger_alert = Mock()
        
        # Record token usage
        collector.record_token_usage(1000, 500)
        
        # Check cost metrics
        assert collector.cost_metrics.total_input_tokens == 1000
        assert collector.cost_metrics.total_output_tokens == 500
        assert collector.cost_metrics.total_api_calls == 1
        
        # Check session data
        session = collector.active_sessions["test-session"]
        assert session["input_tokens"] == 1000
        assert session["output_tokens"] == 500
    
    def test_record_api_error(self, mock_dependencies):
        """Test recording API errors"""
        collector = RealtimeMetricsCollector(**mock_dependencies)
        
        # Start a session
        collector.start_session("test-session")
        
        # Record an error
        collector.record_api_error("connection_error", "502", "Bad Gateway")
        
        # Check session data
        session = collector.active_sessions["test-session"]
        assert len(session["errors"]) == 1
        
        error = session["errors"][0]
        assert error["type"] == "connection_error"
        assert error["code"] == "502"
        assert error["message"] == "Bad Gateway"
        
        # Check integration with existing metrics collector
        mock_dependencies["metrics_collector"].record_request.assert_called_once()
    
    def test_get_metrics_summary(self, mock_dependencies):
        """Test getting comprehensive metrics summary"""
        collector = RealtimeMetricsCollector(**mock_dependencies)
        
        # Add some test data
        collector.start_session("test-session")
        collector.record_voice_latency(250.0)
        collector.record_voice_latency(300.0)
        collector.record_connection_event("connection_attempt", True)
        collector.record_audio_metrics(0.95)
        collector.record_token_usage(1000, 500)
        
        # Get summary
        summary = collector.get_metrics_summary()
        
        # Check structure and content
        assert "timestamp" in summary
        assert "session" in summary
        assert "latency" in summary
        assert "connection" in summary
        assert "audio" in summary
        assert "cost" in summary
        
        # Check specific values
        assert summary["session"]["current_session_id"] == "test-session"
        assert summary["session"]["active_sessions"] == 1
        assert summary["latency"]["sample_count"] == 2
        assert summary["connection"]["state"] == ConnectionState.CONNECTED.value
        assert summary["cost"]["total_tokens"] == 1500
    
    def test_export_metrics_for_dashboard(self, mock_dependencies):
        """Test exporting metrics in dashboard format"""
        collector = RealtimeMetricsCollector(**mock_dependencies)
        
        # Add some test data
        collector.record_voice_latency(250.0)
        collector.record_connection_event("connection_attempt", True)
        collector.record_audio_metrics(0.95)
        
        # Export for dashboard
        dashboard_data = collector.export_metrics_for_dashboard()
        
        # Check structure
        assert "realtime_api" in dashboard_data
        assert "detailed_metrics" in dashboard_data
        
        realtime_data = dashboard_data["realtime_api"]
        assert "latency_ms" in realtime_data
        assert "connection_health" in realtime_data
        assert "cost_per_hour" in realtime_data
        assert "active_sessions" in realtime_data
        assert "audio_quality" in realtime_data
        assert "error_rate" in realtime_data
    
    def test_cleanup(self, mock_dependencies):
        """Test cleanup functionality"""
        collector = RealtimeMetricsCollector(**mock_dependencies)
        
        # Start multiple sessions
        collector.start_session("session-1")
        collector.start_session("session-2")
        
        assert len(collector.active_sessions) == 2
        
        # Cleanup
        collector.cleanup()
        
        assert collector.is_running == False
        assert len(collector.active_sessions) == 0


class TestFactoryFunction:
    """Test the factory function"""
    
    def test_create_realtime_metrics_collector(self):
        """Test the factory function creates a proper instance"""
        collector = create_realtime_metrics_collector()
        
        assert isinstance(collector, RealtimeMetricsCollector)
        assert collector.realtime_config is None  # Default
        
        # Test with custom parameters
        mock_config = Mock()
        collector = create_realtime_metrics_collector(realtime_config=mock_config)
        
        assert collector.realtime_config == mock_config


class TestIntegrationWithExistingMonitoring:
    """Test integration with existing monitoring systems"""
    
    @patch('assistant.realtime_metrics_collector.get_metrics_collector')
    @patch('assistant.realtime_metrics_collector.get_monitor')
    @patch('assistant.realtime_metrics_collector.get_health_monitor')
    def test_integration_initialization(self, mock_health, mock_monitor, mock_metrics):
        """Test that collector integrates with existing monitoring systems"""
        mock_metrics_instance = Mock(spec=MetricsCollector)
        mock_monitor_instance = Mock(spec=PerformanceMonitor)
        mock_health_instance = Mock(spec=SystemHealthMonitor)
        
        mock_metrics.return_value = mock_metrics_instance
        mock_monitor.return_value = mock_monitor_instance
        mock_health.return_value = mock_health_instance
        
        # Create collector without explicit dependencies
        collector = RealtimeMetricsCollector()
        
        # Should use the global instances
        assert collector.metrics_collector == mock_metrics_instance
        assert collector.performance_monitor == mock_monitor_instance
        
        # Verify the get functions were called
        mock_metrics.assert_called_once()
        mock_monitor.assert_called_once()


class TestPrometheusIntegration:
    """Test Prometheus metrics integration (if available)"""
    
    @patch('assistant.realtime_metrics_collector.HAS_PROMETHEUS', True)
    @patch('assistant.realtime_metrics_collector.Counter')
    @patch('assistant.realtime_metrics_collector.Histogram')
    @patch('assistant.realtime_metrics_collector.Gauge')
    def test_prometheus_metrics_initialization(self, mock_gauge, mock_histogram, mock_counter):
        """Test Prometheus metrics are initialized when available"""
        collector = RealtimeMetricsCollector()
        
        # Should have initialized Prometheus metrics
        assert "voice_latency" in collector.prometheus_metrics
        assert "requests_total" in collector.prometheus_metrics
        assert "errors_total" in collector.prometheus_metrics
        assert "cost_dollars" in collector.prometheus_metrics
        
        # Verify calls to create metrics
        mock_histogram.assert_called()
        mock_counter.assert_called()
        mock_gauge.assert_called()
    
    @patch('assistant.realtime_metrics_collector.HAS_PROMETHEUS', False)
    def test_no_prometheus_metrics_when_unavailable(self):
        """Test that Prometheus metrics are disabled when library is unavailable"""
        collector = RealtimeMetricsCollector()
        
        # Should have empty prometheus_metrics
        assert len(collector.prometheus_metrics) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 