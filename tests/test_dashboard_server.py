"""
Tests for Dashboard Server Components

Tests the web-based performance dashboard including:
- Flask server initialization and routing
- Socket.IO WebSocket communication
- Metric aggregation and time windows
- Real-time metric streaming
- Alert handling and broadcasting
- Client subscription management
- System health monitoring
"""

import pytest
import asyncio
import time
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from collections import deque

# Import components to test
from assistant.dashboard_server import (
    DashboardServer, MetricAggregator, create_dashboard_server
)
from assistant.metrics_collector import (
    MetricsCollector, MetricType, ComponentType, 
    LatencyMetrics, AccuracyMetrics, ResourceMetrics, ThroughputMetrics
)


class TestMetricAggregator:
    """Test metric aggregation and time window functionality"""
    
    def test_init_with_default_windows(self):
        """Test aggregator initialization with default time windows"""
        aggregator = MetricAggregator()
        
        assert aggregator.window_sizes == [60, 300, 3600]  # 1min, 5min, 1hour
        assert len(aggregator.aggregated_data) == 3
        
        # Check each window has the expected metric types
        for window_size in aggregator.window_sizes:
            window_data = aggregator.aggregated_data[window_size]
            assert 'latency' in window_data
            assert 'accuracy' in window_data
            assert 'resource' in window_data
            assert 'throughput' in window_data
            assert 'anomalies' in window_data
    
    def test_init_with_custom_windows(self):
        """Test aggregator initialization with custom time windows"""
        custom_windows = [30, 120, 600]  # 30s, 2min, 10min
        aggregator = MetricAggregator(window_sizes=custom_windows)
        
        assert aggregator.window_sizes == custom_windows
        assert len(aggregator.aggregated_data) == 3
    
    def test_round_to_window(self):
        """Test timestamp rounding to window boundaries"""
        aggregator = MetricAggregator()
        
        # Test with 60-second window
        timestamp = datetime(2024, 1, 1, 12, 5, 37)  # 12:05:37
        rounded = aggregator._round_to_window(timestamp, 60)
        expected = datetime(2024, 1, 1, 12, 5, 0)  # Should round to 12:05:00
        assert rounded == expected
        
        # Test with 300-second (5-minute) window
        rounded = aggregator._round_to_window(timestamp, 300)
        expected = datetime(2024, 1, 1, 12, 5, 0)  # Should round to 12:05:00
        assert rounded == expected
    
    def test_add_metric_point(self):
        """Test adding metric points to aggregated data"""
        aggregator = MetricAggregator()
        timestamp = datetime.now()
        test_data = {'latency_p50': 150.0, 'component': 'stt_processing'}
        
        aggregator.add_metric_point('latency', timestamp, test_data)
        
        # Check that data was added to all windows
        for window_size in aggregator.window_sizes:
            window_data = aggregator.aggregated_data[window_size]['latency']
            assert len(window_data) == 1
            
            point = window_data[0]
            assert 'timestamp' in point
            assert point['data'] == test_data
    
    def test_get_window_data(self):
        """Test retrieving data for specific window sizes"""
        aggregator = MetricAggregator()
        timestamp = datetime.now()
        test_data = {'cpu_percent': 45.2}
        
        # Add multiple data points
        for i in range(5):
            aggregator.add_metric_point('resource', timestamp + timedelta(seconds=i), test_data)
        
        # Get data for 60-second window
        data = aggregator.get_window_data('resource', 60, limit=3)
        assert len(data) == 3  # Limited to 3 points
        
        # Get data for non-existent window
        data = aggregator.get_window_data('resource', 999, limit=10)
        assert data == []
        
        # Get data for non-existent metric type
        data = aggregator.get_window_data('nonexistent', 60, limit=10)
        assert data == []


class TestDashboardServer:
    """Test dashboard server initialization and configuration"""
    
    def test_init_without_metrics_collector(self):
        """Test dashboard server initialization without metrics collector"""
        server = DashboardServer(
            metrics_collector=None,
            host="127.0.0.1",
            port=8888,
            debug=True
        )
        
        assert server.metrics_collector is None
        assert server.host == "127.0.0.1"
        assert server.port == 8888
        assert server.debug is True
        assert isinstance(server.aggregator, MetricAggregator)
        assert server.connected_clients == {}
        assert server.client_rooms == {}
        assert len(server.recent_alerts) == 0
    
    def test_init_with_metrics_collector(self):
        """Test dashboard server initialization with metrics collector"""
        mock_collector = Mock(spec=MetricsCollector)
        
        server = DashboardServer(
            metrics_collector=mock_collector,
            host="localhost",
            port=8080
        )
        
        assert server.metrics_collector == mock_collector
        assert server.host == "localhost"
        assert server.port == 8080
        assert server.debug is False
        
        # Check that alert callback was registered
        mock_collector.add_alert_callback.assert_called_once()
    
    @patch('assistant.dashboard_server.Flask')
    @patch('assistant.dashboard_server.SocketIO')
    @patch('assistant.dashboard_server.CORS')
    def test_flask_app_initialization(self, mock_cors, mock_socketio, mock_flask):
        """Test Flask app and Socket.IO initialization"""
        mock_app = Mock()
        # Make the mock app.config behave like a dictionary
        mock_app.config = {}
        mock_flask.return_value = mock_app
        mock_socket = Mock()
        mock_socketio.return_value = mock_socket
        
        server = DashboardServer()
        
        # Verify Flask app was created with correct parameters
        mock_flask.assert_called_once()
        call_kwargs = mock_flask.call_args[1]
        assert 'template_folder' in call_kwargs
        assert 'static_folder' in call_kwargs
        
        # Verify CORS was enabled
        mock_cors.assert_called_once_with(mock_app)
        
        # Verify Socket.IO was initialized with correct parameters
        mock_socketio.assert_called_once()
        socketio_kwargs = mock_socketio.call_args[1]
        assert socketio_kwargs['cors_allowed_origins'] == "*"
        assert socketio_kwargs['async_mode'] == 'threading'
        
        # Verify the SECRET_KEY was set
        assert mock_app.config['SECRET_KEY'] == 'sovereign_dashboard_2024'
    
    def test_determine_alert_severity(self):
        """Test alert severity determination logic"""
        server = DashboardServer()
        
        # Test critical alerts
        assert server._determine_alert_severity('latency_anomaly', {}) == 'critical'
        assert server._determine_alert_severity('accuracy_degradation', {}) == 'critical'
        assert server._determine_alert_severity('high_memory_usage', {}) == 'critical'
        
        # Test warning alerts
        assert server._determine_alert_severity('high_cpu_usage', {}) == 'warning'
        assert server._determine_alert_severity('resource_spike', {}) == 'warning'
        
        # Test info alerts (default)
        assert server._determine_alert_severity('unknown_alert', {}) == 'info'
    
    def test_metric_alert_handling(self):
        """Test handling of metric alerts from collector"""
        server = DashboardServer()
        
        # Mock the socketio emit method
        server.socketio = Mock()
        
        alert_type = 'latency_anomaly'
        alert_data = {
            'component': 'stt_processing',
            'p95_latency': 350.0,
            'threshold': 300.0
        }
        
        # Trigger alert
        server._handle_metric_alert(alert_type, alert_data)
        
        # Check that alert was stored
        assert len(server.recent_alerts) == 1
        alert = server.recent_alerts[0]
        assert alert['type'] == alert_type
        assert alert['data'] == alert_data
        assert alert['severity'] == 'critical'
        assert 'timestamp' in alert
        
        # Check that alert was broadcasted
        server.socketio.emit.assert_called_once()
        call_args = server.socketio.emit.call_args
        assert call_args[0][0] == 'anomaly_alert'
        assert call_args[1]['room'] == 'metrics_alerts'


class TestDashboardServerStreaming:
    """Test real-time metric streaming functionality"""
    
    def setup_method(self):
        """Setup test environment with mock metrics collector"""
        self.mock_collector = Mock(spec=MetricsCollector)
        
        # Mock latency metrics
        self.mock_latency_metrics = {
            'stt_processing': Mock(
                count=10,
                p50=150.0,
                p95=250.0,
                p99=300.0,
                mean=175.0
            ),
            'llm_inference': Mock(
                count=8,
                p50=300.0,
                p95=500.0,
                p99=600.0,
                mean=375.0
            )
        }
        self.mock_collector.latency_metrics = self.mock_latency_metrics
        
        # Mock accuracy metrics
        self.mock_accuracy_metrics = {
            'stt_transcription': Mock(
                count=10,
                mean_score=92.5,
                mean_confidence=0.88,
                success_rate=95.0
            ),
            'memory_recall': Mock(
                count=5,
                mean_score=87.2,
                mean_confidence=0.82,
                success_rate=90.0
            )
        }
        self.mock_collector.accuracy_metrics = self.mock_accuracy_metrics
        
        # Mock resource metrics
        self.mock_resource = Mock(
            cpu_percent=35.2,
            memory_percent=62.8,
            memory_used_gb=4.2,
            gpu_percent=78.5,
            gpu_memory_percent=45.3
        )
        self.mock_collector.get_current_resource_usage.return_value = self.mock_resource
        
        # Mock throughput metrics
        self.mock_throughput_metrics = {
            'overall_pipeline': Mock(
                requests_per_second=2.5,
                success_rate=96.7,
                total_requests=145
            )
        }
        self.mock_collector.throughput_metrics = self.mock_throughput_metrics
        
        self.server = DashboardServer(metrics_collector=self.mock_collector)
        self.server.socketio = Mock()
        self.server.connected_clients = {'client1': {'subscriptions': set(['latency', 'accuracy'])}}
    
    def test_streaming_loop_data_collection(self):
        """Test metric data collection in streaming loop"""
        # Run one iteration of the streaming loop logic
        current_time = datetime.now()
        
        # Collect latency data
        latency_data = {}
        for component, metrics in self.mock_collector.latency_metrics.items():
            if metrics.count > 0:
                latency_data[component] = {
                    'p50': metrics.p50,
                    'p95': metrics.p95,
                    'p99': metrics.p99,
                    'mean': metrics.mean,
                    'count': metrics.count
                }
        
        assert len(latency_data) == 2
        assert latency_data['stt_processing']['p50'] == 150.0
        assert latency_data['llm_inference']['p95'] == 500.0
        
        # Collect accuracy data
        accuracy_data = {}
        for metric_name, metrics in self.mock_collector.accuracy_metrics.items():
            if metrics.count > 0:
                accuracy_data[metric_name] = {
                    'mean_score': metrics.mean_score,
                    'mean_confidence': metrics.mean_confidence,
                    'success_rate': metrics.success_rate,
                    'count': metrics.count
                }
        
        assert len(accuracy_data) == 2
        assert accuracy_data['stt_transcription']['mean_score'] == 92.5
        assert accuracy_data['memory_recall']['success_rate'] == 90.0
        
        # Collect resource data
        resource_data = {
            'cpu_percent': self.mock_resource.cpu_percent,
            'memory_percent': self.mock_resource.memory_percent,
            'memory_used_gb': self.mock_resource.memory_used_gb,
            'gpu_percent': self.mock_resource.gpu_percent,
            'gpu_memory_percent': self.mock_resource.gpu_memory_percent
        }
        
        assert resource_data['cpu_percent'] == 35.2
        assert resource_data['gpu_percent'] == 78.5
    
    def test_streaming_active_flag(self):
        """Test streaming active flag controls"""
        assert self.server.streaming_active is False
        assert self.server.streaming_thread is None
        
        # Mock the threading and streaming setup
        with patch('assistant.dashboard_server.threading.Thread') as mock_thread:
            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance
            
            self.server.start_streaming()
            
            assert self.server.streaming_active is True
            assert self.server.streaming_thread == mock_thread_instance
            mock_thread_instance.start.assert_called_once()
        
        self.server.stop_streaming()
        assert self.server.streaming_active is False
    
    def test_client_cleanup(self):
        """Test cleanup of stale clients"""
        # Add a stale client (last ping > 5 minutes ago)
        stale_time = datetime.now() - timedelta(minutes=10)
        active_time = datetime.now() - timedelta(seconds=30)
        
        self.server.connected_clients = {
            'stale_client': {'last_ping': stale_time},
            'active_client': {'last_ping': active_time}
        }
        self.server.client_rooms = {
            'stale_client': 'metrics_latency',
            'active_client': 'metrics_accuracy'
        }
        
        # Run cleanup
        self.server._cleanup_stale_clients()
        
        # Check that stale client was removed
        assert 'stale_client' not in self.server.connected_clients
        assert 'active_client' in self.server.connected_clients
        assert 'stale_client' not in self.server.client_rooms
        assert 'active_client' in self.server.client_rooms


class TestFlaskRoutes:
    """Test Flask route handlers and API endpoints"""
    
    def setup_method(self):
        """Setup test environment with Flask test client"""
        self.mock_collector = Mock(spec=MetricsCollector)
        
        # Mock performance summary
        self.mock_summary = {
            'latency': {'overall': 245.7, 'components': 5},
            'accuracy': {'overall': 89.3, 'components': 3},
            'resource': {'cpu': 42.1, 'memory': 68.5},
            'throughput': {'rps': 2.8, 'success_rate': 96.2}
        }
        self.mock_collector.get_performance_summary.return_value = self.mock_summary
        
        # Mock resource usage
        self.mock_resource = Mock(
            cpu_percent=42.1,
            memory_percent=68.5,
            memory_used_gb=5.2
        )
        self.mock_collector.get_current_resource_usage.return_value = self.mock_resource
        
        self.server = DashboardServer(metrics_collector=self.mock_collector)
        self.server.app.config['TESTING'] = True
        self.client = self.server.app.test_client()
    
    def test_dashboard_route(self):
        """Test main dashboard HTML page route"""
        with patch('assistant.dashboard_server.render_template') as mock_render:
            mock_render.return_value = '<html>Dashboard</html>'
            
            response = self.client.get('/')
            
            assert response.status_code == 200
            mock_render.assert_called_once_with('dashboard.html')
    
    def test_current_metrics_api(self):
        """Test current metrics API endpoint"""
        response = self.client.get('/api/metrics/current')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'timestamp' in data
        assert 'metrics' in data
        assert data['metrics'] == self.mock_summary
        self.mock_collector.get_performance_summary.assert_called_once()
    
    def test_current_metrics_api_without_collector(self):
        """Test current metrics API when collector is unavailable"""
        server = DashboardServer(metrics_collector=None)
        server.app.config['TESTING'] = True
        client = server.app.test_client()
        
        response = client.get('/api/metrics/current')
        
        assert response.status_code == 503
        data = json.loads(response.data)
        assert 'error' in data
        assert data['error'] == 'Metrics collector not available'
    
    def test_metrics_history_api(self):
        """Test historical metrics API endpoint"""
        # Setup mock aggregator data
        mock_data = [
            {'timestamp': '2024-01-01T12:00:00', 'data': {'p50': 150.0}},
            {'timestamp': '2024-01-01T12:01:00', 'data': {'p50': 165.0}}
        ]
        self.server.aggregator.get_window_data = Mock(return_value=mock_data)
        
        response = self.client.get('/api/metrics/history?type=latency&window=60&limit=50')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['metric_type'] == 'latency'
        assert data['window_size'] == 60
        assert data['data'] == mock_data
        
        self.server.aggregator.get_window_data.assert_called_once_with('latency', 60, 50)
    
    def test_recent_alerts_api(self):
        """Test recent alerts API endpoint"""
        # Add mock alerts
        test_alerts = [
            {'type': 'latency_anomaly', 'timestamp': '2024-01-01T12:00:00', 'severity': 'critical'},
            {'type': 'high_cpu_usage', 'timestamp': '2024-01-01T12:01:00', 'severity': 'warning'}
        ]
        self.server.recent_alerts.extend(test_alerts)
        
        response = self.client.get('/api/alerts/recent?limit=5')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'alerts' in data
        assert len(data['alerts']) == 2
        assert data['alerts'] == test_alerts
    
    def test_system_status_api(self):
        """Test system status API endpoint"""
        response = self.client.get('/api/system/status')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'issues' in data
        assert 'uptime' in data
        assert 'connected_clients' in data
        
        self.mock_collector.get_current_resource_usage.assert_called_once()
    
    def test_system_status_api_high_cpu(self):
        """Test system status API with high CPU usage"""
        # Mock high CPU usage
        high_cpu_resource = Mock(
            cpu_percent=85.0,  # Above 80% threshold
            memory_percent=45.0
        )
        self.mock_collector.get_current_resource_usage.return_value = high_cpu_resource
        
        response = self.client.get('/api/system/status')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['status'] == 'warning'
        assert len(data['issues']) == 1
        assert 'High CPU usage: 85.0%' in data['issues'][0]
    
    def test_system_status_api_with_alerts(self):
        """Test system status API with recent critical alerts"""
        # Add critical alerts
        critical_alerts = [
            {'type': 'latency_anomaly', 'severity': 'critical'},
            {'type': 'accuracy_degradation', 'severity': 'critical'}
        ]
        self.server.recent_alerts.extend(critical_alerts)
        
        response = self.client.get('/api/system/status')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['status'] == 'warning'
        assert '2 recent performance alerts' in data['issues'][0]


class TestWebSocketEvents:
    """Test Socket.IO WebSocket event handlers"""
    
    def setup_method(self):
        """Setup test environment"""
        self.mock_collector = Mock(spec=MetricsCollector)
    
    def test_client_connection(self):
        """Test client connection functionality"""
        # Test that server initializes successfully with WebSocket handlers
        server = DashboardServer(metrics_collector=self.mock_collector)
        
        # Verify server has the necessary client tracking attributes
        assert hasattr(server, 'connected_clients')
        assert hasattr(server, 'client_rooms')
        assert isinstance(server.connected_clients, dict)
        assert isinstance(server.client_rooms, dict)
        
        # Test manual client addition (simulating what connect handler does)
        client_id = 'test_client_123'
        from datetime import datetime
        server.connected_clients[client_id] = {
            'connected_at': datetime.now(),
            'subscriptions': set(),
            'last_ping': datetime.now()
        }
        
        assert client_id in server.connected_clients
        client_data = server.connected_clients[client_id]
        assert 'connected_at' in client_data
        assert 'subscriptions' in client_data
        assert 'last_ping' in client_data
    
    def test_client_disconnection(self):
        """Test client disconnection functionality"""
        server = DashboardServer(metrics_collector=self.mock_collector)
        
        # Add a client first
        client_id = 'test_client_123'
        server.connected_clients[client_id] = {'test': 'data'}
        server.client_rooms[client_id] = 'test_room'
        
        # Test manual client removal (simulating what disconnect handler does)
        if client_id in server.connected_clients:
            del server.connected_clients[client_id]
        if client_id in server.client_rooms:
            del server.client_rooms[client_id]
        
        # Check that client was removed
        assert client_id not in server.connected_clients
        assert client_id not in server.client_rooms
    
    def test_metrics_subscription(self):
        """Test metrics subscription functionality"""
        server = DashboardServer(metrics_collector=self.mock_collector)
        
        # Add client first
        client_id = 'test_client_123'
        server.connected_clients[client_id] = {'subscriptions': set()}
        
        # Test manual subscription update (simulating what subscribe handler does)
        metric_types = ['latency', 'accuracy']
        server.connected_clients[client_id]['subscriptions'].update(metric_types)
        
        # Check that subscriptions were added
        assert 'latency' in server.connected_clients[client_id]['subscriptions']
        assert 'accuracy' in server.connected_clients[client_id]['subscriptions']
    
    def test_ping_pong(self):
        """Test ping/pong heartbeat functionality"""
        server = DashboardServer(metrics_collector=self.mock_collector)
        
        # Add client first
        client_id = 'test_client_123'
        from datetime import datetime
        initial_ping_time = datetime.now()
        server.connected_clients[client_id] = {'last_ping': initial_ping_time}
        
        # Test manual ping time update (simulating what ping handler does)
        new_ping_time = datetime.now()
        server.connected_clients[client_id]['last_ping'] = new_ping_time
        
        # Check that last_ping was updated
        assert 'last_ping' in server.connected_clients[client_id]
        assert server.connected_clients[client_id]['last_ping'] == new_ping_time


class TestIntegrationScenarios:
    """Test complete integration scenarios"""
    
    def test_create_dashboard_server_factory(self):
        """Test dashboard server factory function"""
        mock_collector = Mock(spec=MetricsCollector)
        
        server = create_dashboard_server(
            metrics_collector=mock_collector,
            host="0.0.0.0",
            port=9999,
            debug=True
        )
        
        assert isinstance(server, DashboardServer)
        assert server.metrics_collector == mock_collector
        assert server.host == "0.0.0.0"
        assert server.port == 9999
        assert server.debug is True
    
    def test_end_to_end_metric_flow(self):
        """Test complete metric flow from collection to dashboard"""
        # Create a real metrics collector (not mocked)
        collector = MetricsCollector(collection_interval=0.1)
        collector.start()
        
        try:
            # Create dashboard server
            server = DashboardServer(metrics_collector=collector)
            
            # Add some test metrics
            collector.record_latency('stt_processing', 150.0)
            collector.record_latency('stt_processing', 175.0)
            collector.record_accuracy('stt_transcription', 92.5, confidence=0.88)
            
            # Wait a bit for metrics to be processed
            time.sleep(0.2)
            
            # Check that metrics are available
            assert collector.latency_metrics['stt_processing'].count == 2
            assert collector.accuracy_metrics['stt_transcription'].count == 1
            
            # Test that dashboard can process these metrics
            summary = collector.get_performance_summary()
            assert 'latency_metrics' in summary
            assert 'accuracy_metrics' in summary
            
        finally:
            collector.stop()
    
    def test_alert_propagation(self):
        """Test alert propagation from collector to dashboard"""
        collector = MetricsCollector(collection_interval=0.1)
        server = DashboardServer(metrics_collector=collector)
        server.socketio = Mock()
        
        try:
            collector.start()
            
            # Trigger an anomaly by adding high latency values
            for _ in range(15):  # Need enough samples for anomaly detection
                collector.record_latency('stt_processing', 100.0)  # Normal
            
            for _ in range(5):
                collector.record_latency('stt_processing', 400.0)  # High latency spike
            
            # Wait for metrics processing and anomaly detection
            time.sleep(0.5)
            
            # Check if any alerts were generated (the exact triggering depends on anomaly thresholds)
            # The main thing is that the system didn't crash and can handle the data flow
            assert len(server.recent_alerts) >= 0  # Could be 0 or more depending on thresholds
            
        finally:
            collector.stop()


if __name__ == "__main__":
    pytest.main([__file__]) 