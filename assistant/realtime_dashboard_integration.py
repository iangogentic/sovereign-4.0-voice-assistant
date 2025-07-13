"""
Realtime API Dashboard Integration for Sovereign 4.0

Extends the existing DashboardServer infrastructure to support real-time visualization
of OpenAI Realtime API metrics collected by RealtimeMetricsCollector and 
ConnectionStabilityMonitor.

Features:
- Real-time Realtime API metrics streaming via WebSocket
- Connection stability visualization and health monitoring  
- Cost tracking and usage analytics with alerts
- Audio quality metrics and session analytics
- Integration with existing dashboard infrastructure
- New metric subscription channels for Realtime API data
"""

import asyncio
import json
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import asdict
from collections import defaultdict, deque

from flask_socketio import emit

# Import existing dashboard infrastructure
from .dashboard_server import DashboardServer, MetricAggregator

# Import Realtime API monitoring components
from .realtime_metrics_collector import (
    RealtimeMetricsCollector, RealtimeLatencyMetrics, RealtimeConnectionMetrics,
    RealtimeAudioMetrics, RealtimeCostMetrics, ConnectionState, RealtimeMetricType
)
from .connection_stability_monitor import (
    ConnectionStabilityMonitor, ConnectionQuality, NetworkQualityAssessment
)

logger = logging.getLogger(__name__)


class RealtimeDashboardIntegration:
    """
    Integration layer that extends DashboardServer with Realtime API metrics
    
    Provides seamless integration of RealtimeMetricsCollector and 
    ConnectionStabilityMonitor data into the existing dashboard infrastructure.
    """
    
    def __init__(self,
                 dashboard_server: DashboardServer,
                 realtime_metrics_collector: Optional[RealtimeMetricsCollector] = None,
                 connection_stability_monitor: Optional[ConnectionStabilityMonitor] = None,
                 logger: Optional[logging.Logger] = None):
        
        self.dashboard_server = dashboard_server
        self.realtime_metrics_collector = realtime_metrics_collector
        self.connection_stability_monitor = connection_stability_monitor
        self.logger = logger or logging.getLogger(__name__)
        
        # Realtime-specific metric aggregation
        self.realtime_aggregator = RealtimeMetricAggregator()
        
        # Subscription tracking for Realtime API metrics
        self.realtime_subscriptions = {
            'realtime_latency': set(),
            'realtime_connection': set(), 
            'realtime_audio': set(),
            'realtime_cost': set(),
            'connection_stability': set()
        }
        
        # Real-time streaming state
        self.realtime_streaming_active = False
        self.realtime_streaming_thread: Optional[threading.Thread] = None
        
        # Alert thresholds (configurable)
        self.alert_thresholds = {
            'latency_ms': 500.0,
            'cost_per_hour_usd': 10.0,
            'error_rate_percent': 5.0,
            'connection_quality_min': 60,  # 0-100 scale
            'audio_quality_min': 70       # 0-100 scale
        }
        
        # Initialize integration
        self._setup_realtime_routes()
        self._setup_realtime_socket_handlers()
        
        self.logger.info("RealtimeDashboardIntegration initialized")
    
    def _setup_realtime_routes(self):
        """Add Realtime API specific routes to the dashboard server"""
        
        @self.dashboard_server.app.route('/api/realtime/metrics/current')
        def get_current_realtime_metrics():
            """Get current Realtime API metrics snapshot"""
            if not self.realtime_metrics_collector:
                return {'error': 'Realtime metrics collector not available'}, 503
            
            try:
                metrics_summary = self._get_realtime_metrics_summary()
                connection_health = self._get_connection_health_summary()
                
                return {
                    'timestamp': datetime.now().isoformat(),
                    'realtime_metrics': metrics_summary,
                    'connection_health': connection_health
                }
            except Exception as e:
                self.logger.error(f"Error getting current realtime metrics: {e}")
                return {'error': str(e)}, 500
        
        @self.dashboard_server.app.route('/api/realtime/connection/health')
        def get_connection_health():
            """Get detailed connection health information"""
            if not self.connection_stability_monitor:
                return {'error': 'Connection stability monitor not available'}, 503
            
            try:
                health_summary = self.connection_stability_monitor.get_connection_health_summary()
                return {
                    'timestamp': datetime.now().isoformat(),
                    **health_summary
                }
            except Exception as e:
                self.logger.error(f"Error getting connection health: {e}")
                return {'error': str(e)}, 500
        
        @self.dashboard_server.app.route('/api/realtime/cost/analysis')
        def get_cost_analysis():
            """Get cost analysis and usage trends"""
            if not self.realtime_metrics_collector:
                return {'error': 'Realtime metrics collector not available'}, 503
            
            try:
                cost_metrics = self.realtime_metrics_collector.cost_metrics
                
                # Calculate hourly cost projection
                current_tokens_per_hour = cost_metrics.total_tokens * (3600 / max(1, time.time() - cost_metrics.session_start_time))
                projected_hourly_cost = (
                    (cost_metrics.input_tokens * (3600 / max(1, time.time() - cost_metrics.session_start_time)) * 0.006 / 1000) +
                    (cost_metrics.output_tokens * (3600 / max(1, time.time() - cost_metrics.session_start_time)) * 0.024 / 1000)
                )
                
                return {
                    'timestamp': datetime.now().isoformat(),
                    'current_session_cost': cost_metrics.total_cost_usd,
                    'projected_hourly_cost': projected_hourly_cost,
                    'total_tokens': cost_metrics.total_tokens,
                    'input_tokens': cost_metrics.input_tokens,
                    'output_tokens': cost_metrics.output_tokens,
                    'session_duration_minutes': (time.time() - cost_metrics.session_start_time) / 60,
                    'cost_per_1k_tokens': cost_metrics.total_cost_usd / max(1, cost_metrics.total_tokens / 1000),
                    'alert_triggered': projected_hourly_cost > self.alert_thresholds['cost_per_hour_usd']
                }
            except Exception as e:
                self.logger.error(f"Error getting cost analysis: {e}")
                return {'error': str(e)}, 500
    
    def _setup_realtime_socket_handlers(self):
        """Add Realtime API specific Socket.IO handlers"""
        
        @self.dashboard_server.socketio.on('subscribe_realtime_metrics')
        def handle_realtime_subscribe(data):
            """Handle Realtime API metric subscription requests"""
            client_id = self.dashboard_server.request.sid
            metric_types = data.get('metrics', ['realtime_latency', 'realtime_connection'])
            
            # Add client to appropriate subscription sets
            for metric_type in metric_types:
                if metric_type in self.realtime_subscriptions:
                    self.realtime_subscriptions[metric_type].add(client_id)
                    # Join Socket.IO room for targeted updates
                    self.dashboard_server.socketio.join_room(f"realtime_{metric_type}")
            
            emit('realtime_subscription_confirmed', {
                'subscribed_metrics': metric_types,
                'timestamp': datetime.now().isoformat()
            })
            
            # Send initial snapshot
            if self.realtime_metrics_collector:
                initial_data = self._get_realtime_metrics_summary()
                emit('realtime_metrics_snapshot', {
                    'timestamp': datetime.now().isoformat(),
                    'data': initial_data
                })
            
            self.logger.debug(f"Client {client_id} subscribed to realtime metrics: {metric_types}")
        
        @self.dashboard_server.socketio.on('unsubscribe_realtime_metrics')
        def handle_realtime_unsubscribe(data):
            """Handle Realtime API metric unsubscription"""
            client_id = self.dashboard_server.request.sid
            metric_types = data.get('metrics', [])
            
            for metric_type in metric_types:
                if metric_type in self.realtime_subscriptions:
                    self.realtime_subscriptions[metric_type].discard(client_id)
                    self.dashboard_server.socketio.leave_room(f"realtime_{metric_type}")
            
            emit('realtime_unsubscription_confirmed', {
                'unsubscribed_metrics': metric_types
            })
        
        @self.dashboard_server.socketio.on('get_realtime_config')
        def handle_get_realtime_config():
            """Send current Realtime API configuration and thresholds"""
            emit('realtime_config', {
                'alert_thresholds': self.alert_thresholds,
                'monitoring_intervals': {
                    'metrics_update_ms': 1000,  # 1 second for Realtime API
                    'connection_health_ms': 5000,  # 5 seconds
                    'cost_analysis_ms': 30000   # 30 seconds
                },
                'features_enabled': {
                    'connection_monitoring': self.connection_stability_monitor is not None,
                    'cost_tracking': self.realtime_metrics_collector is not None,
                    'audio_quality': self.realtime_metrics_collector is not None
                }
            })
    
    def start_realtime_streaming(self):
        """Start background thread for streaming Realtime API metrics"""
        if self.realtime_streaming_active:
            return
        
        self.realtime_streaming_active = True
        self.realtime_streaming_thread = threading.Thread(
            target=self._realtime_streaming_loop,
            name="realtime_dashboard_streaming",
            daemon=True
        )
        self.realtime_streaming_thread.start()
        self.logger.info("📊 Realtime dashboard streaming started")
    
    def stop_realtime_streaming(self):
        """Stop Realtime API metrics streaming"""
        self.realtime_streaming_active = False
        if self.realtime_streaming_thread and self.realtime_streaming_thread.is_alive():
            self.realtime_streaming_thread.join(timeout=2.0)
        self.logger.info("📊 Realtime dashboard streaming stopped")
    
    def _realtime_streaming_loop(self):
        """Background loop for streaming Realtime API metrics"""
        while self.realtime_streaming_active:
            try:
                if not any(self.realtime_subscriptions.values()):
                    time.sleep(1.0)
                    continue
                
                current_time = datetime.now()
                
                # Collect and stream Realtime API metrics
                if self.realtime_metrics_collector:
                    metrics_data = self._get_realtime_metrics_summary()
                    
                    # Stream latency metrics
                    if self.realtime_subscriptions['realtime_latency']:
                        latency_data = {
                            'timestamp': current_time.isoformat(),
                            'voice_to_voice_ms': metrics_data.get('latency', {}).get('voice_to_voice_latency_ms', 0),
                            'api_response_ms': metrics_data.get('latency', {}).get('api_response_latency_ms', 0),
                            'p50': metrics_data.get('latency', {}).get('p50_latency_ms', 0),
                            'p95': metrics_data.get('latency', {}).get('p95_latency_ms', 0),
                            'p99': metrics_data.get('latency', {}).get('p99_latency_ms', 0)
                        }
                        
                        self.dashboard_server.socketio.emit('realtime_metrics_update', {
                            'type': 'realtime_latency',
                            'data': latency_data
                        }, room='realtime_realtime_latency')
                    
                    # Stream connection metrics
                    if self.realtime_subscriptions['realtime_connection']:
                        connection_data = {
                            'timestamp': current_time.isoformat(),
                            'connection_state': metrics_data.get('connection', {}).get('connection_state', 'unknown'),
                            'uptime_seconds': metrics_data.get('connection', {}).get('uptime_seconds', 0),
                            'reconnection_count': metrics_data.get('connection', {}).get('reconnection_count', 0),
                            'last_error': metrics_data.get('connection', {}).get('last_error', None)
                        }
                        
                        self.dashboard_server.socketio.emit('realtime_metrics_update', {
                            'type': 'realtime_connection',
                            'data': connection_data
                        }, room='realtime_realtime_connection')
                    
                    # Stream audio quality metrics
                    if self.realtime_subscriptions['realtime_audio']:
                        audio_data = {
                            'timestamp': current_time.isoformat(),
                            'quality_score': metrics_data.get('audio', {}).get('quality_score', 0),
                            'samples_processed': metrics_data.get('audio', {}).get('samples_processed', 0),
                            'processing_time_ms': metrics_data.get('audio', {}).get('avg_processing_time_ms', 0),
                            'buffer_health': metrics_data.get('audio', {}).get('buffer_health_percent', 0)
                        }
                        
                        self.dashboard_server.socketio.emit('realtime_metrics_update', {
                            'type': 'realtime_audio',
                            'data': audio_data
                        }, room='realtime_realtime_audio')
                    
                    # Stream cost metrics
                    if self.realtime_subscriptions['realtime_cost']:
                        cost_data = {
                            'timestamp': current_time.isoformat(),
                            'session_cost_usd': metrics_data.get('cost', {}).get('total_cost_usd', 0),
                            'total_tokens': metrics_data.get('cost', {}).get('total_tokens', 0),
                            'input_tokens': metrics_data.get('cost', {}).get('input_tokens', 0),
                            'output_tokens': metrics_data.get('cost', {}).get('output_tokens', 0),
                            'projected_hourly_cost': self._calculate_projected_hourly_cost()
                        }
                        
                        self.dashboard_server.socketio.emit('realtime_metrics_update', {
                            'type': 'realtime_cost',
                            'data': cost_data
                        }, room='realtime_realtime_cost')
                
                # Collect and stream connection stability metrics
                if (self.connection_stability_monitor and 
                    self.realtime_subscriptions['connection_stability']):
                    
                    health_summary = self.connection_stability_monitor.get_connection_health_summary()
                    
                    stability_data = {
                        'timestamp': current_time.isoformat(),
                        'connection_quality': health_summary['connection_health']['quality'],
                        'stability_score': health_summary['connection_health']['stability_score'],
                        'reliability_score': health_summary['connection_health']['reliability_score'],
                        'avg_latency_ms': health_summary['connection_health']['avg_latency_ms'],
                        'jitter_ms': health_summary['connection_health']['jitter_ms'],
                        'network_score': health_summary['network_quality']['overall_score'],
                        'connection_suitable': health_summary['network_quality']['connection_suitable']
                    }
                    
                    self.dashboard_server.socketio.emit('realtime_metrics_update', {
                        'type': 'connection_stability',
                        'data': stability_data
                    }, room='realtime_connection_stability')
                
                # Check for alerts and broadcast
                self._check_and_broadcast_alerts()
                
                # Store metrics for historical analysis
                self.realtime_aggregator.add_metrics_point(current_time, metrics_data)
                
                # Update every 1 second for real-time feel
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Error in realtime streaming loop: {e}")
                time.sleep(2.0)
    
    def _get_realtime_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive Realtime API metrics summary"""
        if not self.realtime_metrics_collector:
            return {}
        
        metrics = self.realtime_metrics_collector
        
        # Get latency percentiles from the metric collections
        voice_percentiles = metrics.latency_metrics.get_percentiles('voice_to_voice_latency_ms')
        audio_percentiles = metrics.latency_metrics.get_percentiles('audio_processing_latency_ms')
        
        return {
            'latency': {
                'voice_to_voice_latency_ms': voice_percentiles.get('p50', 0.0),
                'audio_processing_latency_ms': audio_percentiles.get('p50', 0.0),
                'text_generation_latency_ms': metrics.latency_metrics.get_percentiles('text_generation_latency_ms').get('p50', 0.0),
                'p50_latency_ms': voice_percentiles.get('p50', 0.0),
                'p95_latency_ms': voice_percentiles.get('p95', 0.0),
                'p99_latency_ms': voice_percentiles.get('p99', 0.0),
                'total_requests': len(metrics.latency_metrics.voice_to_voice_latency_ms)
            },
            'connection': {
                'connection_state': metrics.connection_metrics.connection_state.value,
                'uptime_seconds': metrics.connection_metrics.total_uptime_seconds,
                'reconnection_count': metrics.connection_metrics.reconnection_attempts,
                'disconnection_count': len(metrics.connection_metrics.disconnection_events),
                'last_error': getattr(metrics.connection_metrics, 'last_error_type', None)
            },
            'audio': {
                'quality_score': metrics.audio_metrics.average_audio_quality,
                'samples_processed': metrics.audio_metrics.audio_samples_processed,
                'avg_processing_time_ms': 0.0,  # This would need to be calculated from processing times
                'buffer_health_percent': 100.0,  # Default - could be calculated
                'interruption_count': metrics.audio_metrics.audio_interruptions
            },
            'cost': {
                'total_cost_usd': metrics.cost_metrics.total_cost,
                'total_tokens': metrics.cost_metrics.total_input_tokens + metrics.cost_metrics.total_output_tokens,
                'input_tokens': metrics.cost_metrics.total_input_tokens,
                'output_tokens': metrics.cost_metrics.total_output_tokens,
                'session_start_time': getattr(metrics.cost_metrics, 'session_start_time', time.time())
            },
            'session': {
                'session_count': len(metrics.active_sessions),
                'active_sessions': len(metrics.active_sessions),
                'total_processing_time_ms': 0.0  # Would need to be calculated from session data
            }
        }
    
    def _get_connection_health_summary(self) -> Dict[str, Any]:
        """Get connection health summary from ConnectionStabilityMonitor"""
        if not self.connection_stability_monitor:
            return {}
        
        return self.connection_stability_monitor.get_connection_health_summary()
    
    def _calculate_projected_hourly_cost(self) -> float:
        """Calculate projected hourly cost based on current usage"""
        if not self.realtime_metrics_collector:
            return 0.0
        
        cost_metrics = self.realtime_metrics_collector.cost_metrics
        
        # If we have session costs, calculate based on the time elapsed
        if cost_metrics.session_costs:
            # Get the earliest session timestamp
            earliest_session = min(cost_metrics.session_costs, key=lambda x: x['timestamp'])
            session_duration = time.time() - earliest_session['timestamp']
            
            if session_duration <= 0:
                return 0.0
            
            # Project current rate to hourly
            total_cost = cost_metrics.total_cost
            hourly_projection = total_cost * (3600 / session_duration)
            
            return hourly_projection
        
        # Fallback: calculate based on token usage if available
        total_input_tokens = cost_metrics.total_input_tokens
        total_output_tokens = cost_metrics.total_output_tokens
        
        if total_input_tokens > 0 or total_output_tokens > 0:
            # Estimate based on current session (assume it's been running for at least 1 minute)
            session_duration = max(60, getattr(cost_metrics, 'session_duration', 60))
            
            # Project to hourly
            hourly_input_tokens = total_input_tokens * (3600 / session_duration)
            hourly_output_tokens = total_output_tokens * (3600 / session_duration)
            
            return (hourly_input_tokens * 0.006 / 1000) + (hourly_output_tokens * 0.024 / 1000)
        
        return 0.0
    
    def _check_and_broadcast_alerts(self):
        """Check metrics against thresholds and broadcast alerts"""
        try:
            alerts = []
            
            if self.realtime_metrics_collector:
                metrics = self.realtime_metrics_collector
                
                # Check latency alerts
                if metrics.latency_metrics.get_percentiles('voice_to_voice_latency_ms')['p95'] > self.alert_thresholds['latency_ms']:
                    p95_latency = metrics.latency_metrics.get_percentiles('voice_to_voice_latency_ms')['p95']
                    alerts.append({
                        'type': 'high_latency',
                        'severity': 'warning',
                        'message': f"P95 latency ({p95_latency:.1f}ms) exceeds threshold ({self.alert_thresholds['latency_ms']:.1f}ms)",
                        'value': p95_latency,
                        'threshold': self.alert_thresholds['latency_ms']
                    })
                
                # Check cost alerts
                projected_cost = self._calculate_projected_hourly_cost()
                if projected_cost > self.alert_thresholds['cost_per_hour_usd']:
                    alerts.append({
                        'type': 'high_cost',
                        'severity': 'critical',
                        'message': f"Projected hourly cost (${projected_cost:.2f}) exceeds threshold (${self.alert_thresholds['cost_per_hour_usd']:.2f})",
                        'value': projected_cost,
                        'threshold': self.alert_thresholds['cost_per_hour_usd']
                    })
                
                # Check audio quality alerts
                if metrics.audio_metrics.average_audio_quality < self.alert_thresholds['audio_quality_min']:
                    alerts.append({
                        'type': 'low_audio_quality',
                        'severity': 'warning',
                        'message': f"Audio quality score ({metrics.audio_metrics.average_audio_quality:.1f}) below threshold ({self.alert_thresholds['audio_quality_min']})",
                        'value': metrics.audio_metrics.average_audio_quality,
                        'threshold': self.alert_thresholds['audio_quality_min']
                    })
            
            # Check connection quality alerts
            if self.connection_stability_monitor:
                health = self.connection_stability_monitor.get_connection_health_summary()
                
                stability_score = health['connection_health']['stability_score']
                if stability_score < self.alert_thresholds['connection_quality_min']:
                    alerts.append({
                        'type': 'low_connection_quality',
                        'severity': 'warning',
                        'message': f"Connection stability ({stability_score:.1f}) below threshold ({self.alert_thresholds['connection_quality_min']})",
                        'value': stability_score,
                        'threshold': self.alert_thresholds['connection_quality_min']
                    })
            
            # Broadcast alerts
            for alert in alerts:
                alert['timestamp'] = datetime.now().isoformat()
                self.dashboard_server.socketio.emit('realtime_alert', alert)
                self.logger.warning(f"🚨 Realtime alert: {alert['type']} - {alert['message']}")
                
        except Exception as e:
            self.logger.error(f"Error checking alerts: {e}")


class RealtimeMetricAggregator:
    """Aggregates Realtime API metrics for historical analysis"""
    
    def __init__(self, max_history_points: int = 1000):
        self.max_history_points = max_history_points
        self.history: deque = deque(maxlen=max_history_points)
        self._lock = threading.RLock()
    
    def add_metrics_point(self, timestamp: datetime, metrics_data: Dict[str, Any]):
        """Add a metrics data point for historical tracking"""
        with self._lock:
            self.history.append({
                'timestamp': timestamp.isoformat(),
                'data': metrics_data
            })
    
    def get_historical_data(self, 
                          metric_type: str, 
                          time_window_minutes: int = 60) -> List[Dict[str, Any]]:
        """Get historical data for a specific metric type within time window"""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
            
            filtered_data = []
            for point in self.history:
                point_time = datetime.fromisoformat(point['timestamp'])
                if point_time >= cutoff_time:
                    if metric_type in point['data']:
                        filtered_data.append({
                            'timestamp': point['timestamp'],
                            'value': point['data'][metric_type]
                        })
            
            return filtered_data
    
    def get_average_metrics(self, time_window_minutes: int = 15) -> Dict[str, float]:
        """Calculate average metrics over specified time window"""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
            
            # Collect values for averaging
            latencies = []
            costs = []
            quality_scores = []
            
            for point in self.history:
                point_time = datetime.fromisoformat(point['timestamp'])
                if point_time >= cutoff_time:
                    data = point['data']
                    
                    if 'latency' in data:
                        latencies.append(data['latency'].get('voice_to_voice_latency_ms', 0))
                    
                    if 'cost' in data:
                        costs.append(data['cost'].get('total_cost_usd', 0))
                    
                    if 'audio' in data:
                        quality_scores.append(data['audio'].get('quality_score', 0))
            
            return {
                'avg_latency_ms': sum(latencies) / len(latencies) if latencies else 0,
                'avg_cost_usd': sum(costs) / len(costs) if costs else 0,
                'avg_quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                'data_points': len(latencies)
            }


# Factory function for easy integration
def create_realtime_dashboard_integration(
    dashboard_server: DashboardServer,
    realtime_metrics_collector: Optional[RealtimeMetricsCollector] = None,
    connection_stability_monitor: Optional[ConnectionStabilityMonitor] = None,
    logger: Optional[logging.Logger] = None
) -> RealtimeDashboardIntegration:
    """Create and configure a RealtimeDashboardIntegration instance"""
    
    integration = RealtimeDashboardIntegration(
        dashboard_server=dashboard_server,
        realtime_metrics_collector=realtime_metrics_collector,
        connection_stability_monitor=connection_stability_monitor,
        logger=logger
    )
    
    return integration


# Export main components
__all__ = [
    'RealtimeDashboardIntegration',
    'RealtimeMetricAggregator', 
    'create_realtime_dashboard_integration'
] 