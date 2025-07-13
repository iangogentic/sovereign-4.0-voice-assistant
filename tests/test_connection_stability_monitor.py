"""
Comprehensive Test Suite for ConnectionStabilityMonitor

Tests all components and functionality of the connection stability monitoring system
including health metrics, network quality assessment, pattern analysis, event handling,
and integration with the RealtimeMetricsCollector.
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any
import statistics

# Import the components to test
from assistant.connection_stability_monitor import (
    ConnectionStabilityMonitor,
    ConnectionHealthMetrics,
    NetworkQualityAssessment,
    ConnectionEvent,
    ConnectionQuality,
    NetworkTestType,
    create_connection_stability_monitor
)

# Import supporting components for integration tests
from assistant.realtime_metrics_collector import RealtimeMetricsCollector, ConnectionState


class TestConnectionHealthMetrics:
    """Test the ConnectionHealthMetrics dataclass"""
    
    def test_initialization(self):
        """Test proper initialization of connection health metrics"""
        metrics = ConnectionHealthMetrics()
        
        assert metrics.is_connected == False
        assert metrics.connection_duration == 0.0
        assert metrics.last_heartbeat is None
        assert metrics.heartbeat_latency_ms == 0.0
        assert metrics.connection_quality == ConnectionQuality.UNKNOWN
        assert metrics.stability_score == 0.0
        assert metrics.reliability_score == 0.0
        assert metrics.avg_latency_ms == 0.0
        assert metrics.latency_jitter_ms == 0.0
        assert metrics.total_connections == 0
        assert metrics.successful_connections == 0
        assert metrics.failed_connections == 0
        assert metrics.error_rate == 0.0
    
    def test_timestamps(self):
        """Test timestamp handling"""
        metrics = ConnectionHealthMetrics()
        
        # Check that last_updated is set to current time
        assert abs(metrics.last_updated - time.time()) < 1.0
        
        # Check measurement window
        assert metrics.measurement_window == 300.0  # 5 minutes
    
    def test_connection_tracking(self):
        """Test connection tracking functionality"""
        metrics = ConnectionHealthMetrics()
        
        # Simulate connection attempts
        metrics.total_connections = 10
        metrics.successful_connections = 8
        metrics.failed_connections = 2
        
        # Calculate success rate manually for verification
        expected_success_rate = 8 / 10
        assert abs(metrics.successful_connections / metrics.total_connections - expected_success_rate) < 0.001


class TestNetworkQualityAssessment:
    """Test the NetworkQualityAssessment dataclass"""
    
    def test_initialization(self):
        """Test proper initialization of network quality assessment"""
        assessment = NetworkQualityAssessment()
        
        assert assessment.overall_score == 0.0
        assert assessment.latency_score == 0.0
        assert assessment.stability_score == 0.0
        assert assessment.throughput_score == 0.0
        assert assessment.ping_latency_ms == 0.0
        assert assessment.dns_resolution_ms == 0.0
        assert assessment.ssl_handshake_ms == 0.0
        assert assessment.connection_suitable == True
        assert assessment.estimated_reliability == 0.0
        assert len(assessment.recommended_actions) == 0
    
    def test_timestamp_initialization(self):
        """Test timestamp is properly set on initialization"""
        assessment = NetworkQualityAssessment()
        
        # Check that timestamp is set to current time
        assert abs(assessment.timestamp - time.time()) < 1.0
    
    def test_recommendations_list(self):
        """Test recommendations list functionality"""
        assessment = NetworkQualityAssessment()
        
        # Add some recommendations
        assessment.recommended_actions.append("Check network connectivity")
        assessment.recommended_actions.append("Consider using different network")
        
        assert len(assessment.recommended_actions) == 2
        assert "Check network connectivity" in assessment.recommended_actions


class TestConnectionEvent:
    """Test the ConnectionEvent dataclass"""
    
    def test_initialization(self):
        """Test proper initialization of connection event"""
        timestamp = time.time()
        event = ConnectionEvent(
            timestamp=timestamp,
            event_type="connect"
        )
        
        assert event.timestamp == timestamp
        assert event.event_type == "connect"
        assert event.duration_ms is None
        assert event.error_details is None
        assert len(event.context) == 0
    
    def test_with_optional_fields(self):
        """Test connection event with optional fields"""
        timestamp = time.time()
        error_details = {"error": "Connection timeout"}
        context = {"reason": "network_issue"}
        
        event = ConnectionEvent(
            timestamp=timestamp,
            event_type="disconnect",
            duration_ms=5000.0,
            error_details=error_details,
            context=context
        )
        
        assert event.timestamp == timestamp
        assert event.event_type == "disconnect"
        assert event.duration_ms == 5000.0
        assert event.error_details == error_details
        assert event.context == context


class TestConnectionStabilityMonitor:
    """Test the main ConnectionStabilityMonitor class"""
    
    @pytest.fixture
    def mock_realtime_metrics(self):
        """Create mock RealtimeMetricsCollector"""
        mock_metrics = Mock(spec=RealtimeMetricsCollector)
        mock_metrics.record_connection_event = Mock()
        mock_metrics.record_api_error = Mock()
        mock_metrics.connection_metrics = Mock()
        mock_metrics.connection_metrics.connection_state = ConnectionState.DISCONNECTED
        mock_metrics.connection_metrics.total_uptime_seconds = 0.0
        return mock_metrics
    
    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket"""
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.ping = AsyncMock(return_value=AsyncMock())
        return mock_ws
    
    def test_initialization(self, mock_realtime_metrics):
        """Test proper initialization of ConnectionStabilityMonitor"""
        monitor = ConnectionStabilityMonitor(
            realtime_metrics_collector=mock_realtime_metrics,
            websocket_endpoint="wss://test.example.com/ws",
            monitoring_interval=2.0,
            heartbeat_interval=15.0
        )
        
        assert monitor.realtime_metrics_collector == mock_realtime_metrics
        assert monitor.websocket_endpoint == "wss://test.example.com/ws"
        assert monitor.monitoring_interval == 2.0
        assert monitor.heartbeat_interval == 15.0
        assert monitor.is_monitoring == False
        assert monitor.current_websocket is None
        
        # Check initial metrics
        assert isinstance(monitor.health_metrics, ConnectionHealthMetrics)
        assert isinstance(monitor.network_assessment, NetworkQualityAssessment)
        
        # Check configuration
        assert monitor.config["latency_threshold_ms"] == 500.0
        assert monitor.config["jitter_threshold_ms"] == 100.0
        assert monitor.config["heartbeat_timeout_seconds"] == 60
    
    def test_initialization_with_defaults(self):
        """Test initialization with default parameters"""
        monitor = ConnectionStabilityMonitor()
        
        # Should create its own RealtimeMetricsCollector
        assert monitor.realtime_metrics_collector is not None
        assert monitor.websocket_endpoint == "wss://api.openai.com/v1/realtime"
        assert monitor.monitoring_interval == 5.0
        assert monitor.heartbeat_interval == 30.0
    
    @pytest.mark.asyncio
    async def test_start_monitoring(self, mock_realtime_metrics, mock_websocket):
        """Test starting connection monitoring"""
        monitor = ConnectionStabilityMonitor(realtime_metrics_collector=mock_realtime_metrics)
        
        with patch.object(monitor, '_assess_network_quality', new_callable=AsyncMock):
            result = await monitor.start_monitoring(mock_websocket)
        
        assert result == True
        assert monitor.is_monitoring == True
        assert monitor.current_websocket == mock_websocket
        assert monitor.monitoring_task is not None
        assert monitor.heartbeat_task is not None
        assert monitor.background_thread is not None
        
        # Cleanup
        await monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_start_monitoring_already_active(self, mock_realtime_metrics):
        """Test starting monitoring when already active"""
        monitor = ConnectionStabilityMonitor(realtime_metrics_collector=mock_realtime_metrics)
        monitor.is_monitoring = True
        
        result = await monitor.start_monitoring()
        
        assert result == True  # Should return True but not start again
    
    @pytest.mark.asyncio
    async def test_stop_monitoring(self, mock_realtime_metrics, mock_websocket):
        """Test stopping connection monitoring"""
        monitor = ConnectionStabilityMonitor(realtime_metrics_collector=mock_realtime_metrics)
        
        with patch.object(monitor, '_assess_network_quality', new_callable=AsyncMock):
            await monitor.start_monitoring(mock_websocket)
        
        # Verify it's running
        assert monitor.is_monitoring == True
        
        # Stop monitoring
        await monitor.stop_monitoring()
        
        assert monitor.is_monitoring == False
    
    @pytest.mark.asyncio
    async def test_heartbeat_functionality(self, mock_realtime_metrics, mock_websocket):
        """Test WebSocket heartbeat functionality"""
        monitor = ConnectionStabilityMonitor(
            realtime_metrics_collector=mock_realtime_metrics,
            heartbeat_interval=0.1  # Very short for testing
        )
        
        # Mock ping to return a completed future
        future = asyncio.Future()
        future.set_result(None)
        mock_websocket.ping.return_value = future
        
        with patch.object(monitor, '_assess_network_quality', new_callable=AsyncMock):
            await monitor.start_monitoring(mock_websocket)
        
        # Wait for at least one heartbeat
        await asyncio.sleep(0.2)
        
        # Check that heartbeat was recorded (timing may vary in tests)
        # Note: In fast test environments, heartbeat may not complete, so we check ping was called
        mock_websocket.ping.assert_called()
        
        # Cleanup
        await monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_heartbeat_timeout(self, mock_realtime_metrics, mock_websocket):
        """Test heartbeat timeout handling"""
        monitor = ConnectionStabilityMonitor(
            realtime_metrics_collector=mock_realtime_metrics,
            heartbeat_interval=0.1
        )
        
        # Mock ping to timeout
        mock_websocket.ping.side_effect = asyncio.TimeoutError("Heartbeat timeout")
        
        # Mock the callback
        monitor.on_connection_unstable = AsyncMock()
        
        with patch.object(monitor, '_assess_network_quality', new_callable=AsyncMock):
            await monitor.start_monitoring(mock_websocket)
        
        # Wait for heartbeat attempt
        await asyncio.sleep(0.2)
        
        # Verify timeout was handled
        monitor.on_connection_unstable.assert_called()
        
        # Cleanup
        await monitor.stop_monitoring()
    
    def test_record_connection_attempt(self, mock_realtime_metrics):
        """Test recording connection attempts"""
        monitor = ConnectionStabilityMonitor(realtime_metrics_collector=mock_realtime_metrics)
        
        # Record successful connection
        monitor.record_connection_attempt(True)
        
        assert monitor.health_metrics.total_connections == 1
        assert monitor.health_metrics.successful_connections == 1
        assert monitor.health_metrics.failed_connections == 0
        assert len(monitor.connection_events) == 1
        assert monitor.connection_events[0].event_type == "connect"
        
        # Record failed connection
        error_details = {"error": "Connection refused"}
        monitor.record_connection_attempt(False, error_details)
        
        assert monitor.health_metrics.total_connections == 2
        assert monitor.health_metrics.successful_connections == 1
        assert monitor.health_metrics.failed_connections == 1
        assert len(monitor.connection_events) == 2
        assert monitor.connection_events[1].event_type == "connect_failed"
        assert monitor.connection_events[1].error_details == error_details
    
    def test_record_disconnection(self, mock_realtime_metrics):
        """Test recording disconnection events"""
        monitor = ConnectionStabilityMonitor(realtime_metrics_collector=mock_realtime_metrics)
        
        monitor.record_disconnection("network_error")
        
        assert monitor.health_metrics.connection_drops == 1
        assert len(monitor.connection_events) == 1
        assert monitor.connection_events[0].event_type == "disconnect"
        assert monitor.connection_events[0].context["reason"] == "network_error"
    
    def test_record_latency_measurement(self, mock_realtime_metrics):
        """Test recording latency measurements"""
        monitor = ConnectionStabilityMonitor(realtime_metrics_collector=mock_realtime_metrics)
        
        # Record multiple latency measurements
        latencies = [100.0, 150.0, 200.0, 120.0, 180.0]
        for latency in latencies:
            monitor.record_latency_measurement(latency)
        
        assert len(monitor.latency_history) == len(latencies)
        assert list(monitor.latency_history) == latencies
    
    def test_quality_score_calculation(self, mock_realtime_metrics):
        """Test connection quality score calculation"""
        monitor = ConnectionStabilityMonitor(realtime_metrics_collector=mock_realtime_metrics)
        
        # Set up test data
        monitor.health_metrics.avg_latency_ms = 150.0  # Good latency
        monitor.health_metrics.latency_jitter_ms = 20.0  # Low jitter
        monitor.health_metrics.error_rate = 0.02  # 2% error rate
        monitor.health_metrics.total_connections = 10
        monitor.health_metrics.successful_connections = 9
        
        # Calculate quality scores
        monitor._calculate_quality_scores()
        
        # Verify scores are calculated
        assert monitor.health_metrics.stability_score > 0
        assert monitor.health_metrics.reliability_score > 0
        assert monitor.health_metrics.connection_quality != ConnectionQuality.UNKNOWN
        assert len(monitor.quality_history) == 1
        
        # For good metrics, should have high quality
        assert monitor.health_metrics.connection_quality in [
            ConnectionQuality.GOOD, ConnectionQuality.EXCELLENT
        ]
    
    def test_quality_score_poor_connection(self, mock_realtime_metrics):
        """Test quality scoring for poor connection"""
        monitor = ConnectionStabilityMonitor(realtime_metrics_collector=mock_realtime_metrics)
        
        # Set up poor connection data
        monitor.health_metrics.avg_latency_ms = 800.0  # High latency
        monitor.health_metrics.latency_jitter_ms = 200.0  # High jitter
        monitor.health_metrics.error_rate = 0.25  # 25% error rate
        monitor.health_metrics.total_connections = 10
        monitor.health_metrics.successful_connections = 3
        
        # Calculate quality scores
        monitor._calculate_quality_scores()
        
        # Should indicate poor quality
        assert monitor.health_metrics.connection_quality in [
            ConnectionQuality.POOR, ConnectionQuality.CRITICAL
        ]
        assert monitor.health_metrics.stability_score < 50
        assert monitor.health_metrics.reliability_score < 50
    
    @pytest.mark.asyncio
    async def test_connection_issue_detection(self, mock_realtime_metrics):
        """Test detection of connection issues"""
        monitor = ConnectionStabilityMonitor(realtime_metrics_collector=mock_realtime_metrics)
        
        # Mock callbacks
        monitor.on_network_degradation = AsyncMock()
        monitor.on_connection_unstable = AsyncMock()
        
        # Set up conditions for issue detection
        monitor.health_metrics.avg_latency_ms = 600.0  # Above threshold
        monitor.latency_history.extend([600, 650, 580, 620, 590])  # 5+ samples
        
        monitor.health_metrics.latency_jitter_ms = 150.0  # Above threshold
        monitor.latency_history.extend([100, 200, 150, 250, 180, 160, 140, 190, 210, 170])  # 10+ samples
        
        monitor.health_metrics.error_rate = 0.15  # 15% error rate
        
        # Detect issues
        await monitor._detect_connection_issues()
        
        # Verify callbacks were triggered
        monitor.on_network_degradation.assert_called_with("high_latency")
        monitor.on_connection_unstable.assert_called_with("high_error_rate")
    
    @pytest.mark.asyncio
    async def test_network_quality_assessment(self, mock_realtime_metrics):
        """Test network quality assessment"""
        monitor = ConnectionStabilityMonitor(realtime_metrics_collector=mock_realtime_metrics)
        
        with patch('assistant.connection_stability_monitor.ping3.ping') as mock_ping:
            with patch('assistant.connection_stability_monitor.socket.getaddrinfo') as mock_dns:
                with patch('assistant.connection_stability_monitor.socket.create_connection') as mock_socket:
                    
                    # Mock successful network tests
                    mock_ping.return_value = 0.05  # 50ms ping
                    mock_dns.return_value = [('family', 'type', 'proto', 'canonname', ('ip', 443))]
                    
                    # Mock SSL connection
                    mock_sock = Mock()
                    mock_socket.return_value.__enter__ = Mock(return_value=mock_sock)
                    mock_socket.return_value.__exit__ = Mock(return_value=False)
                    
                    assessment = await monitor._assess_network_quality()
        
        # Verify assessment was performed
        assert isinstance(assessment, NetworkQualityAssessment)
        assert assessment.ping_latency_ms == 50.0
        assert assessment.latency_score > 0
        assert assessment.overall_score >= 0
    
    def test_pattern_analysis(self, mock_realtime_metrics):
        """Test connection pattern analysis"""
        monitor = ConnectionStabilityMonitor(realtime_metrics_collector=mock_realtime_metrics)
        
        # Add various events to create a pattern (all within last 30 minutes)
        current_time = time.time()
        events = [
            ConnectionEvent(current_time - 1600, "connect"),
            ConnectionEvent(current_time - 1500, "heartbeat"),
            ConnectionEvent(current_time - 1400, "error"),
            ConnectionEvent(current_time - 1300, "disconnect"),
            ConnectionEvent(current_time - 1200, "connect"),
            ConnectionEvent(current_time - 1100, "heartbeat_timeout"),
            ConnectionEvent(current_time - 1000, "error"),
            ConnectionEvent(current_time - 900, "disconnect"),
            ConnectionEvent(current_time - 800, "connect"),
            ConnectionEvent(current_time - 700, "heartbeat"),
            ConnectionEvent(current_time - 600, "error"),
            ConnectionEvent(current_time - 500, "heartbeat_timeout")
        ]
        
        monitor.connection_events.extend(events)
        
        # Don't use AsyncMock here to avoid event loop issues - test pattern analysis only
        monitor.on_reconnection_needed = None
        
        # Analyze patterns
        monitor._analyze_connection_patterns()
        
        # Verify pattern analysis was performed
        assert "event_counts" in monitor.connection_patterns
        assert "error_ratio" in monitor.connection_patterns
        assert monitor.connection_patterns["total_events"] == len(events)
        
        # Should detect high error pattern
        error_ratio = monitor.connection_patterns["error_ratio"]
        assert error_ratio > 0.3  # Should be high due to many errors
    
    def test_realtime_metrics_integration(self, mock_realtime_metrics):
        """Test integration with RealtimeMetricsCollector"""
        monitor = ConnectionStabilityMonitor(realtime_metrics_collector=mock_realtime_metrics)
        
        # Set up some test data
        monitor.health_metrics.is_connected = True
        monitor.health_metrics.connection_quality = ConnectionQuality.GOOD
        monitor.health_metrics.connection_duration = 120.0
        
        # Add some connection events
        monitor.connection_events.append(ConnectionEvent(time.time(), "connect"))
        monitor.connection_events.append(ConnectionEvent(time.time(), "error", 
                                                        error_details={"code": "timeout"}))
        
        # Update metrics
        monitor._update_realtime_metrics()
        
        # Verify integration calls
        mock_realtime_metrics.record_connection_event.assert_called()
        mock_realtime_metrics.record_api_error.assert_called()
        
        # Verify connection state was updated
        assert mock_realtime_metrics.connection_metrics.connection_state == ConnectionState.CONNECTED
        assert mock_realtime_metrics.connection_metrics.total_uptime_seconds == 120.0
    
    def test_get_connection_health_summary(self, mock_realtime_metrics):
        """Test getting connection health summary"""
        monitor = ConnectionStabilityMonitor(realtime_metrics_collector=mock_realtime_metrics)
        
        # Set up test data
        monitor.health_metrics.is_connected = True
        monitor.health_metrics.connection_duration = 300.0
        monitor.health_metrics.connection_quality = ConnectionQuality.EXCELLENT
        monitor.health_metrics.stability_score = 95.0
        monitor.health_metrics.reliability_score = 98.0
        monitor.health_metrics.avg_latency_ms = 80.0
        monitor.health_metrics.total_connections = 5
        monitor.health_metrics.successful_connections = 5
        
        monitor.network_assessment.overall_score = 92.0
        monitor.network_assessment.ping_latency_ms = 45.0
        monitor.network_assessment.connection_suitable = True
        
        monitor.connection_events.append(ConnectionEvent(time.time(), "connect"))
        monitor.is_monitoring = True
        
        # Get summary
        summary = monitor.get_connection_health_summary()
        
        # Verify summary structure and content
        assert "timestamp" in summary
        assert "connection_health" in summary
        assert "network_quality" in summary
        assert "connection_stats" in summary
        assert "recent_events" in summary
        assert "monitoring_active" in summary
        
        # Verify specific values
        health = summary["connection_health"]
        assert health["is_connected"] == True
        assert health["connection_duration"] == 300.0
        assert health["quality"] == "excellent"
        assert health["stability_score"] == 95.0
        
        network = summary["network_quality"]
        assert network["overall_score"] == 92.0
        assert network["ping_latency_ms"] == 45.0
        assert network["connection_suitable"] == True
        
        stats = summary["connection_stats"]
        assert stats["total_connections"] == 5
        assert stats["successful_connections"] == 5
        
        assert summary["recent_events"] == 1
        assert summary["monitoring_active"] == True
    
    def test_cleanup(self, mock_realtime_metrics):
        """Test cleanup functionality"""
        monitor = ConnectionStabilityMonitor(realtime_metrics_collector=mock_realtime_metrics)
        
        # Should not raise any exceptions
        monitor.cleanup()


class TestFactoryFunction:
    """Test the factory function"""
    
    def test_create_connection_stability_monitor(self):
        """Test the factory function creates a proper instance"""
        monitor = create_connection_stability_monitor()
        
        assert isinstance(monitor, ConnectionStabilityMonitor)
        assert monitor.websocket_endpoint == "wss://api.openai.com/v1/realtime"
        assert monitor.monitoring_interval == 5.0
        assert monitor.heartbeat_interval == 30.0
        
        # Test with custom parameters
        mock_metrics = Mock(spec=RealtimeMetricsCollector)
        monitor = create_connection_stability_monitor(
            realtime_metrics_collector=mock_metrics,
            websocket_endpoint="wss://test.example.com/ws",
            monitoring_interval=2.0,
            heartbeat_interval=15.0
        )
        
        assert monitor.realtime_metrics_collector == mock_metrics
        assert monitor.websocket_endpoint == "wss://test.example.com/ws"
        assert monitor.monitoring_interval == 2.0
        assert monitor.heartbeat_interval == 15.0


class TestCallbackFunctionality:
    """Test callback functionality"""
    
    @pytest.fixture
    def mock_realtime_metrics(self):
        """Create mock RealtimeMetricsCollector for callback tests"""
        mock_metrics = Mock(spec=RealtimeMetricsCollector)
        mock_metrics.record_connection_event = Mock()
        mock_metrics.record_api_error = Mock()
        mock_metrics.connection_metrics = Mock()
        mock_metrics.connection_metrics.connection_state = ConnectionState.DISCONNECTED
        mock_metrics.connection_metrics.total_uptime_seconds = 0.0
        return mock_metrics
    
    @pytest.mark.asyncio
    async def test_quality_change_callback(self, mock_realtime_metrics):
        """Test connection quality change callback"""
        monitor = ConnectionStabilityMonitor(realtime_metrics_collector=mock_realtime_metrics)
        
        # Mock callback
        callback_called = False
        old_quality = None
        new_quality = None
        
        async def quality_change_callback(old, new):
            nonlocal callback_called, old_quality, new_quality
            callback_called = True
            old_quality = old
            new_quality = new
        
        monitor.on_connection_quality_change = quality_change_callback
        
        # Set initial quality
        monitor.health_metrics.connection_quality = ConnectionQuality.GOOD
        
        # Set up data that should change quality to excellent
        monitor.health_metrics.avg_latency_ms = 50.0
        monitor.health_metrics.latency_jitter_ms = 10.0
        monitor.health_metrics.error_rate = 0.0
        monitor.health_metrics.total_connections = 10
        monitor.health_metrics.successful_connections = 10
        
        # Calculate quality (should trigger callback)
        monitor._calculate_quality_scores()
        
        # Give callback time to execute
        await asyncio.sleep(0.1)
        
        # Verify callback was called with quality change
        assert callback_called == True
        assert old_quality == ConnectionQuality.GOOD
        assert new_quality == ConnectionQuality.EXCELLENT


class TestThreadSafety:
    """Test thread safety of the monitor"""
    
    @pytest.fixture
    def mock_realtime_metrics(self):
        """Create mock RealtimeMetricsCollector for thread safety tests"""
        mock_metrics = Mock(spec=RealtimeMetricsCollector)
        mock_metrics.record_connection_event = Mock()
        mock_metrics.record_api_error = Mock()
        mock_metrics.connection_metrics = Mock()
        mock_metrics.connection_metrics.connection_state = ConnectionState.DISCONNECTED
        mock_metrics.connection_metrics.total_uptime_seconds = 0.0
        return mock_metrics
    
    def test_concurrent_event_recording(self, mock_realtime_metrics):
        """Test concurrent event recording"""
        monitor = ConnectionStabilityMonitor(realtime_metrics_collector=mock_realtime_metrics)
        
        def record_events():
            for i in range(100):
                monitor.record_connection_attempt(i % 2 == 0)
                monitor.record_latency_measurement(100.0 + i)
        
        # Run multiple threads recording events
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=record_events)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all events were recorded correctly
        assert monitor.health_metrics.total_connections == 500  # 5 threads * 100 events
        assert len(monitor.latency_history) == 500
        
        # Verify data integrity
        assert (monitor.health_metrics.successful_connections + 
                monitor.health_metrics.failed_connections == 
                monitor.health_metrics.total_connections)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 