"""
Sovereign 4.0 Voice Assistant - Real-Time Performance Dashboard Server

Provides web-based dashboard for monitoring voice assistant performance metrics:
- Real-time WebSocket streaming of metrics data
- RESTful API for historical data access
- Mobile-responsive dashboard interface
- Anomaly detection visualization
- Integration with MetricsCollector system

Features:
- Socket.IO for WebSocket communication with fallback support
- Metric aggregation windows (1min, 5min, 1hour)
- Progressive disclosure for mobile devices
- Chart.js integration for high-performance visualization
- Redis buffering for high-throughput periods (optional)

Usage:
    server = DashboardServer(metrics_collector, port=8080)
    await server.start()
"""

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import asdict
from collections import defaultdict, deque

import flask
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS

# Import our metrics collector
from .metrics_collector import MetricsCollector, MetricType, ComponentType

logger = logging.getLogger(__name__)

class MetricAggregator:
    """Aggregates metrics into time windows for efficient dashboard display"""
    
    def __init__(self, window_sizes: List[int] = None):
        """
        Initialize aggregator with time windows in seconds
        Default windows: 1min, 5min, 1hour
        """
        self.window_sizes = window_sizes or [60, 300, 3600]  # 1min, 5min, 1hour
        self.aggregated_data: Dict[int, Dict[str, deque]] = {}
        
        # Initialize storage for each window size
        for window_size in self.window_sizes:
            max_points = min(1440, 86400 // window_size)  # Max 24 hours of data
            self.aggregated_data[window_size] = {
                'latency': deque(maxlen=max_points),
                'accuracy': deque(maxlen=max_points),
                'resource': deque(maxlen=max_points),
                'throughput': deque(maxlen=max_points),
                'anomalies': deque(maxlen=max_points)
            }
    
    def add_metric_point(self, metric_type: str, timestamp: datetime, data: Dict[str, Any]):
        """Add a metric point to all appropriate time windows"""
        for window_size in self.window_sizes:
            window_data = self.aggregated_data[window_size]
            
            # Round timestamp to window boundary
            window_timestamp = self._round_to_window(timestamp, window_size)
            
            # Add to appropriate metric type
            if metric_type in window_data:
                point = {
                    'timestamp': window_timestamp.isoformat(),
                    'data': data
                }
                window_data[metric_type].append(point)
    
    def _round_to_window(self, timestamp: datetime, window_size: int) -> datetime:
        """Round timestamp to the nearest window boundary"""
        epoch = timestamp.timestamp()
        rounded_epoch = (epoch // window_size) * window_size
        return datetime.fromtimestamp(rounded_epoch)
    
    def get_window_data(self, metric_type: str, window_size: int, limit: int = 100) -> List[Dict]:
        """Get aggregated data for a specific metric type and window size"""
        if window_size not in self.aggregated_data:
            return []
        
        window_data = self.aggregated_data[window_size].get(metric_type, deque())
        return list(window_data)[-limit:]

class DashboardServer:
    """
    Real-time performance dashboard server for Sovereign 4.0 Voice Assistant
    
    Provides:
    - WebSocket streaming of live metrics
    - REST API for historical data
    - Responsive dashboard interface
    - Anomaly detection visualization
    """
    
    def __init__(self, 
                 metrics_collector: Optional[MetricsCollector] = None,
                 host: str = "localhost",
                 port: int = 8080,
                 debug: bool = False):
        self.metrics_collector = metrics_collector
        self.host = host
        self.port = port
        self.debug = debug
        
        # Initialize Flask app and Socket.IO
        self.app = Flask(__name__, 
                        template_folder='../dashboard/templates',
                        static_folder='../dashboard/static')
        self.app.config['SECRET_KEY'] = 'sovereign_dashboard_2024'
        
        # Enable CORS for development
        CORS(self.app)
        
        # Initialize Socket.IO with async support
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins="*",
            async_mode='threading',
            ping_timeout=60,
            ping_interval=25
        )
        
        # Metric aggregation
        self.aggregator = MetricAggregator()
        
        # Connected clients tracking
        self.connected_clients: Dict[str, Dict[str, Any]] = {}
        self.client_rooms: Dict[str, str] = {}  # client_id -> room_name
        
        # Background streaming
        self.streaming_thread: Optional[threading.Thread] = None
        self.streaming_active = False
        
        # Anomaly alert tracking
        self.recent_alerts: deque = deque(maxlen=100)
        
        # Setup routes and socket handlers
        self._setup_routes()
        self._setup_socket_handlers()
        
        # Register with metrics collector for alerts
        if self.metrics_collector:
            self.metrics_collector.add_alert_callback(self._handle_metric_alert)
        
        logger.info(f"Dashboard server initialized on {host}:{port}")
    
    def _setup_routes(self):
        """Setup Flask routes for the dashboard"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/metrics/current')
        def get_current_metrics():
            """Get current real-time metrics snapshot"""
            if not self.metrics_collector:
                return jsonify({'error': 'Metrics collector not available'}), 503
            
            try:
                summary = self.metrics_collector.get_performance_summary()
                return jsonify({
                    'timestamp': datetime.now().isoformat(),
                    'metrics': summary
                })
            except Exception as e:
                logger.error(f"Error getting current metrics: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/metrics/history')
        def get_metrics_history():
            """Get historical metrics data with filtering"""
            metric_type = request.args.get('type', 'latency')
            window_size = int(request.args.get('window', 60))  # seconds
            limit = int(request.args.get('limit', 100))
            
            try:
                data = self.aggregator.get_window_data(metric_type, window_size, limit)
                return jsonify({
                    'metric_type': metric_type,
                    'window_size': window_size,
                    'data': data
                })
            except Exception as e:
                logger.error(f"Error getting metrics history: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/alerts/recent')
        def get_recent_alerts():
            """Get recent anomaly alerts"""
            limit = int(request.args.get('limit', 20))
            alerts = list(self.recent_alerts)[-limit:]
            return jsonify({'alerts': alerts})
        
        @self.app.route('/api/system/status')
        def get_system_status():
            """Get overall system health status"""
            if not self.metrics_collector:
                return jsonify({'status': 'unknown', 'reason': 'No metrics collector'})
            
            try:
                # Get current resource usage
                resource_usage = self.metrics_collector.get_current_resource_usage()
                
                # Determine overall status
                status = 'healthy'
                issues = []
                
                if resource_usage:
                    if resource_usage.cpu_percent > 80:
                        status = 'warning'
                        issues.append(f'High CPU usage: {resource_usage.cpu_percent:.1f}%')
                    
                    if resource_usage.memory_percent > 85:
                        status = 'warning'
                        issues.append(f'High memory usage: {resource_usage.memory_percent:.1f}%')
                
                # Check recent alerts
                recent_critical_alerts = [
                    alert for alert in list(self.recent_alerts)[-10:]
                    if alert.get('type') in ['latency_anomaly', 'accuracy_degradation']
                ]
                
                if recent_critical_alerts:
                    status = 'warning'
                    issues.append(f'{len(recent_critical_alerts)} recent performance alerts')
                
                return jsonify({
                    'status': status,
                    'timestamp': datetime.now().isoformat(),
                    'issues': issues,
                    'uptime': time.time() - getattr(self, '_start_time', time.time()),
                    'connected_clients': len(self.connected_clients)
                })
                
            except Exception as e:
                logger.error(f"Error getting system status: {e}")
                return jsonify({
                    'status': 'error',
                    'reason': str(e)
                })
    
    def _setup_socket_handlers(self):
        """Setup Socket.IO event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            client_id = request.sid
            self.connected_clients[client_id] = {
                'connected_at': datetime.now(),
                'subscriptions': set(),
                'last_ping': datetime.now()
            }
            
            logger.info(f"🔌 Dashboard client connected: {client_id}")
            emit('connection_status', {'status': 'connected', 'client_id': client_id})
            
            # Send current metrics snapshot
            if self.metrics_collector:
                try:
                    summary = self.metrics_collector.get_performance_summary()
                    emit('metrics_snapshot', {
                        'timestamp': datetime.now().isoformat(),
                        'metrics': summary
                    })
                except Exception as e:
                    logger.error(f"Error sending initial metrics: {e}")
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            client_id = request.sid
            
            # Clean up client data
            if client_id in self.connected_clients:
                del self.connected_clients[client_id]
            
            if client_id in self.client_rooms:
                del self.client_rooms[client_id]
            
            logger.info(f"🔌 Dashboard client disconnected: {client_id}")
        
        @self.socketio.on('subscribe_metrics')
        def handle_subscribe(data):
            """Handle metric subscription requests"""
            client_id = request.sid
            metric_types = data.get('metrics', ['latency', 'accuracy', 'resource'])
            
            if client_id in self.connected_clients:
                self.connected_clients[client_id]['subscriptions'].update(metric_types)
                
                # Join appropriate rooms for targeted updates
                for metric_type in metric_types:
                    join_room(f"metrics_{metric_type}")
                    self.client_rooms[client_id] = f"metrics_{metric_type}"
                
                emit('subscription_confirmed', {
                    'subscribed_metrics': list(self.connected_clients[client_id]['subscriptions'])
                })
                
                logger.debug(f"Client {client_id} subscribed to: {metric_types}")
        
        @self.socketio.on('unsubscribe_metrics')
        def handle_unsubscribe(data):
            """Handle metric unsubscription requests"""
            client_id = request.sid
            metric_types = data.get('metrics', [])
            
            if client_id in self.connected_clients:
                for metric_type in metric_types:
                    self.connected_clients[client_id]['subscriptions'].discard(metric_type)
                    leave_room(f"metrics_{metric_type}")
                
                emit('unsubscription_confirmed', {
                    'unsubscribed_metrics': metric_types,
                    'remaining_subscriptions': list(self.connected_clients[client_id]['subscriptions'])
                })
        
        @self.socketio.on('ping')
        def handle_ping():
            """Handle client ping for connection health"""
            client_id = request.sid
            if client_id in self.connected_clients:
                self.connected_clients[client_id]['last_ping'] = datetime.now()
            emit('pong', {'timestamp': datetime.now().isoformat()})
    
    def _handle_metric_alert(self, alert_type: str, alert_data: Dict[str, Any]):
        """Handle alerts from the metrics collector"""
        alert = {
            'type': alert_type,
            'timestamp': datetime.now().isoformat(),
            'data': alert_data,
            'severity': self._determine_alert_severity(alert_type, alert_data)
        }
        
        # Store alert
        self.recent_alerts.append(alert)
        
        # Broadcast to connected clients
        self.socketio.emit('anomaly_alert', alert, room='metrics_alerts')
        
        logger.info(f"📊 Dashboard alert broadcast: {alert_type}")
    
    def _determine_alert_severity(self, alert_type: str, alert_data: Dict[str, Any]) -> str:
        """Determine alert severity level"""
        critical_alerts = ['latency_anomaly', 'accuracy_degradation', 'high_memory_usage']
        warning_alerts = ['high_cpu_usage', 'resource_spike']
        
        if alert_type in critical_alerts:
            return 'critical'
        elif alert_type in warning_alerts:
            return 'warning'
        else:
            return 'info'
    
    def start_streaming(self):
        """Start background thread for streaming metrics to connected clients"""
        if self.streaming_active:
            return
        
        self.streaming_active = True
        self.streaming_thread = threading.Thread(
            target=self._streaming_loop,
            name="dashboard_streaming",
            daemon=True
        )
        self.streaming_thread.start()
        logger.info("📊 Dashboard streaming started")
    
    def stop_streaming(self):
        """Stop background streaming"""
        self.streaming_active = False
        if self.streaming_thread and self.streaming_thread.is_alive():
            self.streaming_thread.join(timeout=2.0)
        logger.info("📊 Dashboard streaming stopped")
    
    def _streaming_loop(self):
        """Background loop for streaming metrics to clients"""
        while self.streaming_active:
            try:
                if not self.connected_clients or not self.metrics_collector:
                    time.sleep(1.0)
                    continue
                
                # Get current metrics
                current_time = datetime.now()
                
                # Collect latency metrics
                latency_data = {}
                for component, metrics in self.metrics_collector.latency_metrics.items():
                    if metrics.count > 0:
                        latency_data[component] = {
                            'p50': metrics.p50,
                            'p95': metrics.p95,
                            'p99': metrics.p99,
                            'mean': metrics.mean,
                            'count': metrics.count
                        }
                
                # Collect accuracy metrics
                accuracy_data = {}
                for metric_name, metrics in self.metrics_collector.accuracy_metrics.items():
                    if metrics.count > 0:
                        accuracy_data[metric_name] = {
                            'mean_score': metrics.mean_score,
                            'mean_confidence': metrics.mean_confidence,
                            'success_rate': metrics.success_rate,
                            'count': metrics.count
                        }
                
                # Collect resource metrics
                resource_data = None
                current_resource = self.metrics_collector.get_current_resource_usage()
                if current_resource:
                    resource_data = {
                        'cpu_percent': current_resource.cpu_percent,
                        'memory_percent': current_resource.memory_percent,
                        'memory_used_gb': current_resource.memory_used_gb,
                        'gpu_percent': current_resource.gpu_percent,
                        'gpu_memory_percent': current_resource.gpu_memory_percent
                    }
                
                # Collect throughput metrics
                throughput_data = {}
                for component, metrics in self.metrics_collector.throughput_metrics.items():
                    if metrics.total_requests > 0:
                        throughput_data[component] = {
                            'requests_per_second': metrics.requests_per_second,
                            'success_rate': metrics.success_rate,
                            'total_requests': metrics.total_requests
                        }
                
                # Aggregate data for storage
                if latency_data:
                    self.aggregator.add_metric_point('latency', current_time, latency_data)
                if accuracy_data:
                    self.aggregator.add_metric_point('accuracy', current_time, accuracy_data)
                if resource_data:
                    self.aggregator.add_metric_point('resource', current_time, resource_data)
                if throughput_data:
                    self.aggregator.add_metric_point('throughput', current_time, throughput_data)
                
                # Stream to subscribed clients
                stream_data = {
                    'timestamp': current_time.isoformat(),
                    'latency': latency_data,
                    'accuracy': accuracy_data,
                    'resource': resource_data,
                    'throughput': throughput_data
                }
                
                # Broadcast to different metric type rooms
                if latency_data:
                    self.socketio.emit('metrics_update', {
                        'type': 'latency',
                        'data': stream_data
                    }, room='metrics_latency')
                
                if accuracy_data:
                    self.socketio.emit('metrics_update', {
                        'type': 'accuracy', 
                        'data': stream_data
                    }, room='metrics_accuracy')
                
                if resource_data:
                    self.socketio.emit('metrics_update', {
                        'type': 'resource',
                        'data': stream_data  
                    }, room='metrics_resource')
                
                # Clean up disconnected clients
                self._cleanup_stale_clients()
                
                # Stream every 500ms for smooth updates
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error in streaming loop: {e}")
                time.sleep(1.0)
    
    def _cleanup_stale_clients(self):
        """Remove clients that haven't pinged recently"""
        current_time = datetime.now()
        stale_timeout = timedelta(minutes=5)
        
        stale_clients = []
        for client_id, client_data in self.connected_clients.items():
            if current_time - client_data['last_ping'] > stale_timeout:
                stale_clients.append(client_id)
        
        for client_id in stale_clients:
            logger.warning(f"Removing stale client: {client_id}")
            del self.connected_clients[client_id]
            if client_id in self.client_rooms:
                del self.client_rooms[client_id]
    
    async def start(self):
        """Start the dashboard server"""
        self._start_time = time.time()
        
        # Start metrics streaming
        self.start_streaming()
        
        logger.info(f"🌐 Starting dashboard server on {self.host}:{self.port}")
        
        # Run the server
        self.socketio.run(
            self.app,
            host=self.host,
            port=self.port,
            debug=self.debug,
            use_reloader=False  # Disable reloader to avoid threading issues
        )
    
    async def stop(self):
        """Stop the dashboard server"""
        self.stop_streaming()
        logger.info("🌐 Dashboard server stopped")

def create_dashboard_server(metrics_collector: Optional[MetricsCollector] = None,
                          host: str = "localhost",
                          port: int = 8080,
                          debug: bool = False) -> DashboardServer:
    """Factory function to create dashboard server"""
    return DashboardServer(metrics_collector, host, port, debug) 