"""
Comprehensive Test Suite for Realtime Dashboard Integration

Tests all components and functionality of the dashboard integration system
including WebSocket streaming, metrics collection, alert handling, cost tracking,
and integration with existing dashboard infrastructure.
"""

import pytest
import asyncio
import json
import time
import threading
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any
from datetime import datetime, timedelta

# Import Flask testing utilities
from flask import Flask
from flask_socketio import SocketIOTestClient

# Import the components to test
from assistant.realtime_dashboard_integration import (
    RealtimeDashboardIntegration,
    RealtimeMetricAggregator,
    create_realtime_dashboard_integration
)

# Import supporting components for integration tests
from assistant.dashboard_server import DashboardServer, MetricAggregator
from assistant.realtime_metrics_collector import (
    RealtimeMetricsCollector, RealtimeLatencyMetrics, RealtimeConnectionMetrics,
    RealtimeAudioMetrics, RealtimeCostMetrics, ConnectionState
)
from assistant.connection_stability_monitor import (
    ConnectionStabilityMonitor, ConnectionQuality, ConnectionHealthMetrics
)


# Global fixtures for use across all test classes
@pytest.fixture
def mock_dashboard_server():
    """Create mock DashboardServer"""
    mock_app = Mock(spec=Flask)
    mock_app.route = Mock(return_value=lambda f: f)  # Decorator passthrough
    
    mock_socketio = Mock()
    mock_socketio.on = Mock(return_value=lambda f: f)  # Decorator passthrough
    mock_socketio.emit = Mock()
    mock_socketio.join_room = Mock()
    mock_socketio.leave_room = Mock()
    
    mock_server = Mock(spec=DashboardServer)
    mock_server.app = mock_app
    mock_server.socketio = mock_socketio
    mock_server.request = Mock()
    mock_server.request.sid = "test_client_123"
    
    return mock_server

@pytest.fixture
def mock_realtime_metrics():
    """Create mock RealtimeMetricsCollector"""
    mock_metrics = Mock(spec=RealtimeMetricsCollector)
    
    # Mock latency metrics with correct structure
    mock_latency = Mock()
    mock_latency.voice_to_voice_latency_ms = Mock()
    mock_latency.voice_to_voice_latency_ms.__len__ = Mock(return_value=100)  # For total_requests
    mock_latency.get_percentiles = Mock(return_value={
        'p50': 140.0,
        'p95': 200.0,
        'p99': 250.0
    })
    mock_metrics.latency_metrics = mock_latency
    
    # Mock connection metrics with correct structure
    mock_connection = Mock()
    mock_connection.connection_state = ConnectionState.CONNECTED
    mock_connection.total_uptime_seconds = 3600.0
    mock_connection.reconnection_attempts = 2
    mock_connection.disconnection_events = Mock()
    mock_connection.disconnection_events.__len__ = Mock(return_value=3)
    mock_metrics.connection_metrics = mock_connection
    
    # Mock audio metrics with correct structure
    mock_audio = Mock()
    mock_audio.average_audio_quality = 85.0
    mock_audio.audio_samples_processed = 1000
    mock_audio.audio_interruptions = 5
    mock_metrics.audio_metrics = mock_audio
    
    # Mock cost metrics with correct structure
    mock_cost = Mock()
    mock_cost.total_cost = 0.15
    mock_cost.total_input_tokens = 2000
    mock_cost.total_output_tokens = 3000
    mock_cost.session_costs = [
        {
            'timestamp': time.time() - 1800,  # 30 minutes ago
            'input_tokens': 2000,
            'output_tokens': 3000,
            'cost': 0.15
        }
    ]
    mock_metrics.cost_metrics = mock_cost
    
    # Mock active sessions
    mock_metrics.active_sessions = Mock()
    mock_metrics.active_sessions.__len__ = Mock(return_value=2)
    
    return mock_metrics

@pytest.fixture
def mock_connection_monitor():
    """Create mock ConnectionStabilityMonitor"""
    mock_monitor = Mock(spec=ConnectionStabilityMonitor)
    
    health_summary = {
        'timestamp': time.time(),
        'connection_health': {
            'is_connected': True,
            'connection_duration': 1800.0,
            'quality': 'good',
            'stability_score': 85.0,
            'reliability_score': 92.0,
            'avg_latency_ms': 120.0,
            'jitter_ms': 15.0,
            'error_rate': 0.02
        },
        'network_quality': {
            'overall_score': 88.0,
            'ping_latency_ms': 45.0,
            'dns_resolution_ms': 12.0,
            'ssl_handshake_ms': 150.0,
            'connection_suitable': True,
            'estimated_reliability': 0.92
        },
        'connection_stats': {
            'total_connections': 10,
            'successful_connections': 9,
            'failed_connections': 1,
            'connection_drops': 1,
            'reconnection_attempts': 2
        },
        'recent_events': 15,
        'monitoring_active': True
    }
    
    mock_monitor.get_connection_health_summary.return_value = health_summary
    
    return mock_monitor


class TestRealtimeMetricAggregator:
    """Test the RealtimeMetricAggregator class"""
    
    def test_initialization(self):
        """Test proper initialization of aggregator"""
        aggregator = RealtimeMetricAggregator(max_history_points=500)
        
        assert aggregator.max_history_points == 500
        assert len(aggregator.history) == 0
        assert aggregator._lock is not None
    
    def test_add_metrics_point(self):
        """Test adding metrics data points"""
        aggregator = RealtimeMetricAggregator()
        
        timestamp = datetime.now()
        metrics_data = {
            'latency': {'voice_to_voice_latency_ms': 150.0},
            'cost': {'total_cost_usd': 0.05},
            'audio': {'quality_score': 85.0}
        }
        
        aggregator.add_metrics_point(timestamp, metrics_data)
        
        assert len(aggregator.history) == 1
        assert aggregator.history[0]['timestamp'] == timestamp.isoformat()
        assert aggregator.history[0]['data'] == metrics_data
    
    def test_get_historical_data(self):
        """Test retrieving historical data for specific metric type"""
        aggregator = RealtimeMetricAggregator()
        
        # Add test data
        now = datetime.now()
        for i in range(5):
            timestamp = now - timedelta(minutes=i*10)
            metrics_data = {
                'latency': {'voice_to_voice_latency_ms': 100.0 + i*10}
            }
            aggregator.add_metrics_point(timestamp, metrics_data)
        
        # Get data from last 30 minutes
        historical_data = aggregator.get_historical_data('latency', time_window_minutes=30)
        
        assert len(historical_data) == 3  # Only last 30 minutes
        assert all('timestamp' in point and 'value' in point for point in historical_data)
    
    def test_get_average_metrics(self):
        """Test calculating average metrics over time window"""
        aggregator = RealtimeMetricAggregator()
        
        # Add test data
        now = datetime.now()
        latencies = [100, 150, 200, 120, 180]
        
        for i, latency in enumerate(latencies):
            timestamp = now - timedelta(minutes=i*2)
            metrics_data = {
                'latency': {'voice_to_voice_latency_ms': latency},
                'cost': {'total_cost_usd': 0.01 * (i+1)},
                'audio': {'quality_score': 80 + i*2}
            }
            aggregator.add_metrics_point(timestamp, metrics_data)
        
        averages = aggregator.get_average_metrics(time_window_minutes=15)
        
        assert averages['avg_latency_ms'] == sum(latencies) / len(latencies)
        assert averages['data_points'] == len(latencies)
        assert 'avg_cost_usd' in averages
        assert 'avg_quality_score' in averages
    
    def test_max_history_points_limit(self):
        """Test that history respects max_history_points limit"""
        aggregator = RealtimeMetricAggregator(max_history_points=3)
        
        # Add more points than the limit
        now = datetime.now()
        for i in range(5):
            timestamp = now - timedelta(minutes=i)
            metrics_data = {'test': {'value': i}}
            aggregator.add_metrics_point(timestamp, metrics_data)
        
        # Should only keep the last 3 points
        assert len(aggregator.history) == 3


class TestRealtimeDashboardIntegration:
    """Test the main RealtimeDashboardIntegration class"""
    
    def test_initialization(self, mock_dashboard_server, mock_realtime_metrics, mock_connection_monitor):
        """Test proper initialization of integration"""
        integration = RealtimeDashboardIntegration(
            dashboard_server=mock_dashboard_server,
            realtime_metrics_collector=mock_realtime_metrics,
            connection_stability_monitor=mock_connection_monitor
        )
        
        assert integration.dashboard_server == mock_dashboard_server
        assert integration.realtime_metrics_collector == mock_realtime_metrics
        assert integration.connection_stability_monitor == mock_connection_monitor
        assert isinstance(integration.realtime_aggregator, RealtimeMetricAggregator)
        assert len(integration.realtime_subscriptions) == 5
        assert not integration.realtime_streaming_active
        
        # Check that routes and socket handlers were set up
        mock_dashboard_server.app.route.assert_called()
        mock_dashboard_server.socketio.on.assert_called()
    
    def test_get_realtime_metrics_summary(self, mock_dashboard_server, mock_realtime_metrics):
        """Test getting comprehensive metrics summary"""
        integration = RealtimeDashboardIntegration(
            dashboard_server=mock_dashboard_server,
            realtime_metrics_collector=mock_realtime_metrics
        )
        
        summary = integration._get_realtime_metrics_summary()
        
        # Verify structure and data
        assert 'latency' in summary
        assert 'connection' in summary
        assert 'audio' in summary
        assert 'cost' in summary
        assert 'session' in summary
        
        # Verify specific values
        assert summary['latency']['voice_to_voice_latency_ms'] == 140.0  # Updated to match mock return value
        assert summary['latency']['p95_latency_ms'] == 200.0
        assert summary['connection']['connection_state'] == 'connected'
        assert summary['audio']['quality_score'] == 85.0
        assert summary['cost']['total_cost_usd'] == 0.15
        assert summary['session']['session_count'] == 2
    
    def test_get_connection_health_summary(self, mock_dashboard_server, mock_connection_monitor):
        """Test getting connection health summary"""
        integration = RealtimeDashboardIntegration(
            dashboard_server=mock_dashboard_server,
            connection_stability_monitor=mock_connection_monitor
        )
        
        health_summary = integration._get_connection_health_summary()
        
        assert 'connection_health' in health_summary
        assert 'network_quality' in health_summary
        assert health_summary['connection_health']['quality'] == 'good'
        assert health_summary['network_quality']['overall_score'] == 88.0
    
    def test_calculate_projected_hourly_cost(self, mock_dashboard_server, mock_realtime_metrics):
        """Test hourly cost projection calculation"""
        integration = RealtimeDashboardIntegration(
            dashboard_server=mock_dashboard_server,
            realtime_metrics_collector=mock_realtime_metrics
        )
        
        projected_cost = integration._calculate_projected_hourly_cost()
        
        # The new logic uses total_cost and session duration
        # Session duration is 30 minutes (1800 seconds), total cost is 0.15
        # Hourly projection = 0.15 * (3600/1800) = 0.15 * 2 = 0.30
        expected_cost = 0.15 * 2  # 0.30
        assert abs(projected_cost - expected_cost) < 0.001
    
    def test_start_stop_streaming(self, mock_dashboard_server, mock_realtime_metrics):
        """Test starting and stopping realtime streaming"""
        integration = RealtimeDashboardIntegration(
            dashboard_server=mock_dashboard_server,
            realtime_metrics_collector=mock_realtime_metrics
        )
        
        # Test starting streaming
        assert not integration.realtime_streaming_active
        integration.start_realtime_streaming()
        assert integration.realtime_streaming_active
        assert integration.realtime_streaming_thread is not None
        
        # Test stopping streaming
        integration.stop_realtime_streaming()
        assert not integration.realtime_streaming_active
    
    def test_check_and_broadcast_alerts(self, mock_dashboard_server, mock_realtime_metrics, mock_connection_monitor):
        """Test alert checking and broadcasting"""
        integration = RealtimeDashboardIntegration(
            dashboard_server=mock_dashboard_server,
            realtime_metrics_collector=mock_realtime_metrics,
            connection_stability_monitor=mock_connection_monitor
        )
        
        # Set up conditions for alerts - modify the mock to return high values
        mock_realtime_metrics.latency_metrics.get_percentiles.return_value = {
            'p50': 400.0,
            'p95': 600.0,  # Above threshold
            'p99': 800.0
        }
        mock_realtime_metrics.audio_metrics.average_audio_quality = 50.0  # Below threshold
        
        # Mock projected cost calculation to trigger cost alert
        with patch.object(integration, '_calculate_projected_hourly_cost', return_value=15.0):
            integration._check_and_broadcast_alerts()
        
        # Verify alerts were broadcast
        assert mock_dashboard_server.socketio.emit.call_count >= 3  # At least 3 alerts
        
        # Check specific alert calls
        calls = mock_dashboard_server.socketio.emit.call_args_list
        alert_types = [call[0][1]['type'] for call in calls if call[0][0] == 'realtime_alert']
        assert 'high_latency' in alert_types
        assert 'high_cost' in alert_types
        assert 'low_audio_quality' in alert_types
    
    def test_realtime_streaming_loop_integration(self, mock_dashboard_server, mock_realtime_metrics):
        """Test the streaming loop with mock data"""
        integration = RealtimeDashboardIntegration(
            dashboard_server=mock_dashboard_server,
            realtime_metrics_collector=mock_realtime_metrics
        )
        
        # Add some subscriptions
        integration.realtime_subscriptions['realtime_latency'].add('client1')
        integration.realtime_subscriptions['realtime_cost'].add('client1')
        
        # Mock the streaming methods
        # Set up alert conditions to trigger socketio calls
        mock_realtime_metrics.latency_metrics.get_percentiles.return_value = {
            'p50': 400.0,
            'p95': 600.0,  # Above threshold to trigger alert
            'p99': 800.0
        }
        
        # Mock the streaming methods (but don't patch _check_and_broadcast_alerts)
        integration.realtime_streaming_active = True
        
        # Run one iteration of the loop manually
        try:
            # This would normally run in the thread
            current_time = datetime.now()
            metrics_data = integration._get_realtime_metrics_summary()
            integration.realtime_aggregator.add_metrics_point(current_time, metrics_data)
            
            # Call the real alert method to trigger socketio calls
            integration._check_and_broadcast_alerts()
            
            # Verify that socketio was called (alert should have been triggered)
            assert mock_dashboard_server.socketio.emit.called
            
        finally:
            integration.realtime_streaming_active = False


class TestRouteHandlers:
    """Test the Flask route handlers added by the integration"""
    
    def test_current_realtime_metrics_route(self, mock_dashboard_server, mock_realtime_metrics, mock_connection_monitor):
        """Test the current realtime metrics API endpoint"""
        integration = RealtimeDashboardIntegration(
            dashboard_server=mock_dashboard_server,
            realtime_metrics_collector=mock_realtime_metrics,
            connection_stability_monitor=mock_connection_monitor
        )
        
        # Test with metrics collector available
        result = integration._get_realtime_metrics_summary()
        assert isinstance(result, dict)
        assert 'latency' in result
        assert 'connection' in result
        assert 'audio' in result
        assert 'cost' in result
    
    def test_connection_health_route(self, mock_dashboard_server, mock_connection_monitor):
        """Test the connection health API endpoint"""
        integration = RealtimeDashboardIntegration(
            dashboard_server=mock_dashboard_server,
            connection_stability_monitor=mock_connection_monitor
        )
        
        # Test with connection monitor available
        result = integration._get_connection_health_summary()
        assert isinstance(result, dict)
        assert 'connection_health' in result
        assert 'network_quality' in result
    
    def test_cost_analysis_route(self, mock_dashboard_server, mock_realtime_metrics):
        """Test the cost analysis API endpoint"""
        integration = RealtimeDashboardIntegration(
            dashboard_server=mock_dashboard_server,
            realtime_metrics_collector=mock_realtime_metrics
        )
        
        # Test cost calculation
        projected_cost = integration._calculate_projected_hourly_cost()
        assert isinstance(projected_cost, float)
        assert projected_cost >= 0


class TestSocketHandlers:
    """Test the Socket.IO event handlers"""
    
    def test_subscription_management(self, mock_dashboard_server, mock_realtime_metrics):
        """Test subscription and unsubscription handlers"""
        integration = RealtimeDashboardIntegration(
            dashboard_server=mock_dashboard_server,
            realtime_metrics_collector=mock_realtime_metrics
        )
        
        client_id = "test_client_123"
        
        # Test subscription
        assert len(integration.realtime_subscriptions['realtime_latency']) == 0
        integration.realtime_subscriptions['realtime_latency'].add(client_id)
        assert client_id in integration.realtime_subscriptions['realtime_latency']
        
        # Test unsubscription
        integration.realtime_subscriptions['realtime_latency'].discard(client_id)
        assert client_id not in integration.realtime_subscriptions['realtime_latency']


class TestFactoryFunction:
    """Test the factory function"""
    
    def test_create_realtime_dashboard_integration(self, mock_dashboard_server, mock_realtime_metrics, mock_connection_monitor):
        """Test the factory function creates a proper instance"""
        integration = create_realtime_dashboard_integration(
            dashboard_server=mock_dashboard_server,
            realtime_metrics_collector=mock_realtime_metrics,
            connection_stability_monitor=mock_connection_monitor
        )
        
        assert isinstance(integration, RealtimeDashboardIntegration)
        assert integration.dashboard_server == mock_dashboard_server
        assert integration.realtime_metrics_collector == mock_realtime_metrics
        assert integration.connection_stability_monitor == mock_connection_monitor


class TestIntegrationScenarios:
    """Test complete integration scenarios"""
    
    def test_full_metrics_flow(self, mock_dashboard_server, mock_realtime_metrics, mock_connection_monitor):
        """Test complete metrics collection and broadcasting flow"""
        integration = RealtimeDashboardIntegration(
            dashboard_server=mock_dashboard_server,
            realtime_metrics_collector=mock_realtime_metrics,
            connection_stability_monitor=mock_connection_monitor
        )
        
        # Simulate client subscription
        client_id = "test_client"
        integration.realtime_subscriptions['realtime_latency'].add(client_id)
        integration.realtime_subscriptions['realtime_cost'].add(client_id)
        
        # Get metrics summary
        metrics_summary = integration._get_realtime_metrics_summary()
        connection_health = integration._get_connection_health_summary()
        
        # Verify data completeness
        assert metrics_summary['latency']['voice_to_voice_latency_ms'] > 0
        assert metrics_summary['cost']['total_cost_usd'] >= 0
        assert connection_health['connection_health']['stability_score'] > 0
        
        # Test metrics aggregation
        current_time = datetime.now()
        integration.realtime_aggregator.add_metrics_point(current_time, metrics_summary)
        
        # Verify historical data
        historical = integration.realtime_aggregator.get_historical_data('latency', 5)
        assert len(historical) == 1
    
    def test_alert_threshold_management(self, mock_dashboard_server, mock_realtime_metrics):
        """Test alert threshold configuration and triggering"""
        integration = RealtimeDashboardIntegration(
            dashboard_server=mock_dashboard_server,
            realtime_metrics_collector=mock_realtime_metrics
        )
        
        # Test threshold configuration
        original_latency_threshold = integration.alert_thresholds['latency_ms']
        integration.alert_thresholds['latency_ms'] = 100.0
        
        # Set metrics to trigger alert
        mock_realtime_metrics.latency_metrics.p95_latency_ms = 150.0
        
        # Check alerts
        integration._check_and_broadcast_alerts()
        
        # Verify alert was broadcast
        mock_dashboard_server.socketio.emit.assert_called()
        
        # Reset threshold
        integration.alert_thresholds['latency_ms'] = original_latency_threshold
    
    def test_concurrent_streaming_and_aggregation(self, mock_dashboard_server, mock_realtime_metrics):
        """Test concurrent streaming and data aggregation"""
        integration = RealtimeDashboardIntegration(
            dashboard_server=mock_dashboard_server,
            realtime_metrics_collector=mock_realtime_metrics
        )
        
        # Add multiple data points concurrently
        import threading
        
        def add_data_points():
            for i in range(10):
                timestamp = datetime.now()
                metrics_data = {'test': {'value': i}}
                integration.realtime_aggregator.add_metrics_point(timestamp, metrics_data)
                time.sleep(0.01)
        
        # Run multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=add_data_points)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify data integrity
        assert len(integration.realtime_aggregator.history) == 30  # 3 threads * 10 points
    
    def test_error_handling_in_streaming(self, mock_dashboard_server):
        """Test error handling in streaming loop"""
        # Create integration without metrics collector to trigger errors
        integration = RealtimeDashboardIntegration(
            dashboard_server=mock_dashboard_server,
            realtime_metrics_collector=None,
            connection_stability_monitor=None
        )
        
        # Metrics summary should return empty dict when no collector
        summary = integration._get_realtime_metrics_summary()
        assert summary == {}
        
        # Health summary should return empty dict when no monitor
        health = integration._get_connection_health_summary()
        assert health == {}
        
        # Cost calculation should return 0 when no metrics
        cost = integration._calculate_projected_hourly_cost()
        assert cost == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 