"""
Realtime API Health Monitor for Sovereign 4.0

Provides comprehensive health monitoring for OpenAI Realtime API integration:
- Health check endpoints (/health/realtime, /metrics/realtime)
- Service discovery integration
- Connection status monitoring
- Performance metrics collection
- Prometheus metrics export
- Deployment rollback triggers

Integrates with production deployment infrastructure for:
- Docker health checks
- Kubernetes readiness/liveness probes
- Load balancer health verification
- Monitoring dashboards
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import psutil
import websockets

# Integration with existing monitoring systems
from .realtime_metrics_collector import RealtimeMetricsCollector
from .realtime_session_manager import RealtimeSessionManager
from .connection_stability_monitor import ConnectionStabilityMonitor
from .cost_optimization_manager import CostOptimizationManager
from .config_manager import RealtimeAPIConfig

# Prometheus metrics (optional)
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


class HealthStatus(Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class ServiceState(Enum):
    """Service operational states"""
    STARTING = "starting"
    READY = "ready"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class HealthCheckResult:
    """Individual health check result"""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    duration_ms: float
    critical: bool = False


@dataclass
class ServiceStatus:
    """Overall service status"""
    service_name: str
    status: HealthStatus
    state: ServiceState
    uptime_seconds: float
    version: str
    checks: List[HealthCheckResult]
    timestamp: datetime
    endpoint_urls: Dict[str, str]


@dataclass
class RealtimeMetrics:
    """Realtime API performance metrics"""
    total_connections: int
    active_sessions: int
    average_latency_ms: float
    p95_latency_ms: float
    success_rate: float
    error_rate: float
    total_cost_usd: float
    tokens_per_second: float
    connection_pool_utilization: float


class RealtimeHealthMonitor:
    """
    Comprehensive health monitoring for Realtime API services
    
    Provides HTTP endpoints for health checks, metrics collection,
    and service discovery integration for production deployment.
    """
    
    def __init__(self,
                 config: RealtimeAPIConfig,
                 metrics_collector: Optional[RealtimeMetricsCollector] = None,
                 session_manager: Optional[RealtimeSessionManager] = None,
                 connection_pool: Optional[ConnectionStabilityMonitor] = None,
                 cost_manager: Optional[CostOptimizationManager] = None,
                 logger: Optional[logging.Logger] = None):
        
        self.config = config
        self.metrics_collector = metrics_collector
        self.session_manager = session_manager
        self.connection_pool = connection_pool
        self.cost_manager = cost_manager
        self.logger = logger or logging.getLogger(__name__)
        
        # Flask app for health endpoints
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Service state tracking
        self.service_state = ServiceState.STARTING
        self.start_time = time.time()
        
        # Health check configuration
        self.health_checks = {}
        self.critical_checks = set()
        self.check_intervals = {}
        self.last_check_results = {}
        
        # Metrics tracking
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0.0
        
        # Prometheus metrics
        if HAS_PROMETHEUS:
            self._setup_prometheus_metrics()
        
        # Thread management
        self.monitoring_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        # Setup health endpoints
        self._setup_health_endpoints()
        self._register_default_health_checks()
        
        self.logger.info("🏥 Realtime Health Monitor initialized")
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics for export"""
        try:
            self.prometheus_metrics = {
                'health_check_duration': Histogram(
                    'realtime_health_check_duration_seconds',
                    'Time spent on health checks',
                    ['check_name']
                ),
                'health_check_status': Gauge(
                    'realtime_health_check_status',
                    'Health check status (1=healthy, 0=unhealthy)',
                    ['check_name']
                ),
                'service_uptime': Gauge(
                    'realtime_service_uptime_seconds',
                    'Service uptime in seconds'
                ),
                'active_connections': Gauge(
                    'realtime_active_connections',
                    'Number of active WebSocket connections'
                ),
                'request_total': Counter(
                    'realtime_requests_total',
                    'Total HTTP requests',
                    ['endpoint', 'status']
                ),
                'error_total': Counter(
                    'realtime_errors_total',
                    'Total errors',
                    ['error_type']
                )
            }
            self.logger.info("📊 Prometheus metrics configured")
        except Exception as e:
            self.logger.error(f"❌ Failed to setup Prometheus metrics: {e}")
    
    def _setup_health_endpoints(self):
        """Setup Flask endpoints for health monitoring"""
        
        @self.app.route('/health', methods=['GET'])
        def basic_health():
            """Basic health check endpoint"""
            try:
                self.request_count += 1
                
                if HAS_PROMETHEUS:
                    self.prometheus_metrics['request_total'].labels(
                        endpoint='health', status='success'
                    ).inc()
                
                status = self._get_overall_health_status()
                return jsonify({
                    'status': status.value,
                    'timestamp': datetime.now().isoformat(),
                    'service': 'sovereign-realtime-api',
                    'uptime_seconds': time.time() - self.start_time
                }), 200 if status == HealthStatus.HEALTHY else 503
                
            except Exception as e:
                self.error_count += 1
                if HAS_PROMETHEUS:
                    self.prometheus_metrics['request_total'].labels(
                        endpoint='health', status='error'
                    ).inc()
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/health/realtime', methods=['GET'])
        def realtime_health():
            """Detailed Realtime API health check"""
            try:
                self.request_count += 1
                start_time = time.time()
                
                # Run all health checks
                results = self._run_all_health_checks()
                
                # Create service status
                service_status = ServiceStatus(
                    service_name="sovereign-realtime-api",
                    status=self._get_overall_health_status(),
                    state=self.service_state,
                    uptime_seconds=time.time() - self.start_time,
                    version=getattr(self.config, 'version', '4.0.0'),
                    checks=results,
                    timestamp=datetime.now(),
                    endpoint_urls={
                        'health': '/health',
                        'health_detailed': '/health/realtime',
                        'metrics': '/metrics/realtime',
                        'prometheus': '/metrics'
                    }
                )
                
                duration = (time.time() - start_time) * 1000
                
                if HAS_PROMETHEUS:
                    self.prometheus_metrics['request_total'].labels(
                        endpoint='health_realtime', status='success'
                    ).inc()
                
                response_code = 200 if service_status.status == HealthStatus.HEALTHY else 503
                
                return jsonify(asdict(service_status)), response_code
                
            except Exception as e:
                self.error_count += 1
                if HAS_PROMETHEUS:
                    self.prometheus_metrics['request_total'].labels(
                        endpoint='health_realtime', status='error'
                    ).inc()
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/health/ready', methods=['GET'])
        def readiness_probe():
            """Kubernetes readiness probe"""
            try:
                if self.service_state in [ServiceState.READY, ServiceState.RUNNING]:
                    return jsonify({
                        'ready': True,
                        'state': self.service_state.value,
                        'timestamp': datetime.now().isoformat()
                    }), 200
                else:
                    return jsonify({
                        'ready': False,
                        'state': self.service_state.value,
                        'timestamp': datetime.now().isoformat()
                    }), 503
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/health/live', methods=['GET'])
        def liveness_probe():
            """Kubernetes liveness probe"""
            try:
                if self.service_state != ServiceState.ERROR:
                    return jsonify({
                        'alive': True,
                        'state': self.service_state.value,
                        'timestamp': datetime.now().isoformat()
                    }), 200
                else:
                    return jsonify({
                        'alive': False,
                        'state': self.service_state.value,
                        'timestamp': datetime.now().isoformat()
                    }), 503
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/metrics/realtime', methods=['GET'])
        def realtime_metrics():
            """Detailed Realtime API metrics"""
            try:
                self.request_count += 1
                
                metrics = self._collect_realtime_metrics()
                
                if HAS_PROMETHEUS:
                    self.prometheus_metrics['request_total'].labels(
                        endpoint='metrics_realtime', status='success'
                    ).inc()
                
                return jsonify(asdict(metrics)), 200
                
            except Exception as e:
                self.error_count += 1
                if HAS_PROMETHEUS:
                    self.prometheus_metrics['request_total'].labels(
                        endpoint='metrics_realtime', status='error'
                    ).inc()
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/metrics', methods=['GET'])
        def prometheus_metrics():
            """Prometheus metrics endpoint"""
            try:
                if not HAS_PROMETHEUS:
                    return jsonify({'error': 'Prometheus client not available'}), 503
                
                # Update current metrics
                self._update_prometheus_metrics()
                
                # Return Prometheus format
                return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/service/info', methods=['GET'])
        def service_info():
            """Service discovery information"""
            try:
                return jsonify({
                    'service_name': 'sovereign-realtime-api',
                    'version': getattr(self.config, 'version', '4.0.0'),
                    'description': 'OpenAI Realtime API integration service',
                    'endpoints': {
                        'health': '/health',
                        'health_detailed': '/health/realtime',
                        'readiness': '/health/ready',
                        'liveness': '/health/live',
                        'metrics': '/metrics/realtime',
                        'prometheus': '/metrics',
                        'service_info': '/service/info'
                    },
                    'capabilities': [
                        'realtime_voice',
                        'low_latency_response',
                        'websocket_connection',
                        'session_management',
                        'cost_optimization',
                        'performance_monitoring'
                    ],
                    'protocols': ['http', 'websocket'],
                    'ports': {
                        'http': 8080,
                        'websocket': 5000,
                        'metrics': 9091
                    },
                    'dependencies': ['chroma-db', 'openai-api'],
                    'timestamp': datetime.now().isoformat()
                }), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def _register_default_health_checks(self):
        """Register default health checks"""
        
        # Critical health checks
        self.register_health_check(
            'openai_api_connectivity',
            self._check_openai_api,
            critical=True,
            interval=60
        )
        
        self.register_health_check(
            'database_connectivity',
            self._check_database,
            critical=True,
            interval=30
        )
        
        self.register_health_check(
            'memory_usage',
            self._check_memory_usage,
            critical=False,
            interval=30
        )
        
        self.register_health_check(
            'websocket_pool',
            self._check_websocket_pool,
            critical=True,
            interval=60
        )
        
        self.register_health_check(
            'session_manager',
            self._check_session_manager,
            critical=False,
            interval=45
        )
        
        self.register_health_check(
            'cost_manager',
            self._check_cost_manager,
            critical=False,
            interval=120
        )
    
    def register_health_check(self, name: str, check_func: callable, 
                            critical: bool = False, interval: int = 60):
        """Register a health check function"""
        self.health_checks[name] = check_func
        self.check_intervals[name] = interval
        
        if critical:
            self.critical_checks.add(name)
        
        self.logger.debug(f"🔍 Registered health check: {name} (critical: {critical})")
    
    async def _check_openai_api(self) -> HealthCheckResult:
        """Check OpenAI API connectivity"""
        start_time = time.time()
        
        try:
            # Simple API test
            import openai
            client = openai.OpenAI()
            
            # Test with a simple completion
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            
            duration = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name="openai_api_connectivity",
                status=HealthStatus.HEALTHY,
                message="OpenAI API is accessible",
                details={
                    "response_time_ms": duration,
                    "model_tested": "gpt-3.5-turbo"
                },
                timestamp=datetime.now(),
                duration_ms=duration,
                critical=True
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name="openai_api_connectivity",
                status=HealthStatus.CRITICAL,
                message=f"OpenAI API connectivity failed: {str(e)}",
                details={
                    "error": str(e),
                    "response_time_ms": duration
                },
                timestamp=datetime.now(),
                duration_ms=duration,
                critical=True
            )
    
    async def _check_database(self) -> HealthCheckResult:
        """Check database connectivity"""
        start_time = time.time()
        
        try:
            # Test Chroma database
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://{self.config.chroma_host or 'localhost'}:8000/api/v1/heartbeat",
                    timeout=5.0
                )
                response.raise_for_status()
            
            duration = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name="database_connectivity",
                status=HealthStatus.HEALTHY,
                message="Database is accessible",
                details={
                    "response_time_ms": duration,
                    "database": "chroma"
                },
                timestamp=datetime.now(),
                duration_ms=duration,
                critical=True
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name="database_connectivity",
                status=HealthStatus.CRITICAL,
                message=f"Database connectivity failed: {str(e)}",
                details={
                    "error": str(e),
                    "response_time_ms": duration
                },
                timestamp=datetime.now(),
                duration_ms=duration,
                critical=True
            )
    
    async def _check_memory_usage(self) -> HealthCheckResult:
        """Check system memory usage"""
        start_time = time.time()
        
        try:
            memory = psutil.virtual_memory()
            
            if memory.percent > 90:
                status = HealthStatus.CRITICAL
                message = f"Critical memory usage: {memory.percent}%"
            elif memory.percent > 80:
                status = HealthStatus.DEGRADED
                message = f"High memory usage: {memory.percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory.percent}%"
            
            duration = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name="memory_usage",
                status=status,
                message=message,
                details={
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "memory_total_gb": memory.total / (1024**3)
                },
                timestamp=datetime.now(),
                duration_ms=duration
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name="memory_usage",
                status=HealthStatus.UNHEALTHY,
                message=f"Memory check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now(),
                duration_ms=duration
            )
    
    async def _check_websocket_pool(self) -> HealthCheckResult:
        """Check WebSocket connection pool health"""
        start_time = time.time()
        
        try:
            if not self.connection_pool:
                return HealthCheckResult(
                    name="websocket_pool",
                    status=HealthStatus.DEGRADED,
                    message="WebSocket pool not configured",
                    details={},
                    timestamp=datetime.now(),
                    duration_ms=(time.time() - start_time) * 1000
                )
            
            # Get pool statistics  
            stats = self.connection_pool.get_connection_health_summary()
            
            connection_stats = stats.get('connection_stats', {})
            active_connections = connection_stats.get('active_connections', 0)
            total_connections = connection_stats.get('total_connections', 0)
            
            if active_connections == 0:
                status = HealthStatus.DEGRADED
                message = "No active WebSocket connections"
            elif total_connections > 0:
                status = HealthStatus.HEALTHY
                message = f"WebSocket pool healthy: {active_connections}/{total_connections}"
            else:
                status = HealthStatus.UNHEALTHY
                message = "WebSocket pool not initialized"
            
            duration = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name="websocket_pool",
                status=status,
                message=message,
                details=stats,
                timestamp=datetime.now(),
                duration_ms=duration,
                critical=True
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name="websocket_pool",
                status=HealthStatus.CRITICAL,
                message=f"WebSocket pool check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now(),
                duration_ms=duration,
                critical=True
            )
    
    async def _check_session_manager(self) -> HealthCheckResult:
        """Check session manager health"""
        start_time = time.time()
        
        try:
            if not self.session_manager:
                return HealthCheckResult(
                    name="session_manager",
                    status=HealthStatus.DEGRADED,
                    message="Session manager not configured",
                    details={},
                    timestamp=datetime.now(),
                    duration_ms=(time.time() - start_time) * 1000
                )
            
            # Get session statistics
            active_sessions = self.session_manager.get_active_sessions()
            session_count = len(active_sessions)
            
            if session_count > 100:
                status = HealthStatus.DEGRADED
                message = f"High session count: {session_count}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Session manager healthy: {session_count} active sessions"
            
            duration = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name="session_manager",
                status=status,
                message=message,
                details={
                    "active_sessions": session_count,
                    "session_details": {sid: s.status.value for sid, s in active_sessions.items()}
                },
                timestamp=datetime.now(),
                duration_ms=duration
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name="session_manager",
                status=HealthStatus.UNHEALTHY,
                message=f"Session manager check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now(),
                duration_ms=duration
            )
    
    async def _check_cost_manager(self) -> HealthCheckResult:
        """Check cost optimization manager health"""
        start_time = time.time()
        
        try:
            if not self.cost_manager:
                return HealthCheckResult(
                    name="cost_manager",
                    status=HealthStatus.DEGRADED,
                    message="Cost manager not configured",
                    details={},
                    timestamp=datetime.now(),
                    duration_ms=(time.time() - start_time) * 1000
                )
            
            # Get cost optimization statistics
            stats = await self.cost_manager.get_optimization_stats()
            
            sessions_terminated = stats.get('optimization_stats', {}).get('sessions_terminated', 0)
            cost_savings = stats.get('optimization_stats', {}).get('cost_savings', 0.0)
            
            status = HealthStatus.HEALTHY
            message = f"Cost manager healthy: ${cost_savings:.2f} saved, {sessions_terminated} sessions managed"
            
            duration = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name="cost_manager",
                status=status,
                message=message,
                details=stats,
                timestamp=datetime.now(),
                duration_ms=duration
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name="cost_manager",
                status=HealthStatus.UNHEALTHY,
                message=f"Cost manager check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now(),
                duration_ms=duration
            )
    
    def _run_all_health_checks(self) -> List[HealthCheckResult]:
        """Run all registered health checks"""
        results = []
        
        for name, check_func in self.health_checks.items():
            try:
                # Run the check
                if asyncio.iscoroutinefunction(check_func):
                    # Async check
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(check_func())
                    loop.close()
                else:
                    # Sync check
                    result = check_func()
                
                results.append(result)
                self.last_check_results[name] = result
                
                # Update Prometheus metrics
                if HAS_PROMETHEUS and hasattr(self, 'prometheus_metrics'):
                    self.prometheus_metrics['health_check_duration'].labels(
                        check_name=name
                    ).observe(result.duration_ms / 1000)
                    
                    self.prometheus_metrics['health_check_status'].labels(
                        check_name=name
                    ).set(1 if result.status == HealthStatus.HEALTHY else 0)
                
            except Exception as e:
                # Handle check failures
                error_result = HealthCheckResult(
                    name=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}",
                    details={"error": str(e)},
                    timestamp=datetime.now(),
                    duration_ms=0.0,
                    critical=name in self.critical_checks
                )
                results.append(error_result)
                self.last_check_results[name] = error_result
        
        return results
    
    def _get_overall_health_status(self) -> HealthStatus:
        """Determine overall health status from individual checks"""
        if not self.last_check_results:
            return HealthStatus.DEGRADED
        
        critical_failures = 0
        degraded_count = 0
        total_checks = len(self.last_check_results)
        
        for name, result in self.last_check_results.items():
            if result.status == HealthStatus.CRITICAL:
                if name in self.critical_checks:
                    critical_failures += 1
                else:
                    degraded_count += 1
            elif result.status in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]:
                degraded_count += 1
        
        # Determine overall status
        if critical_failures > 0:
            return HealthStatus.CRITICAL
        elif degraded_count > total_checks * 0.5:  # More than 50% degraded
            return HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def _collect_realtime_metrics(self) -> RealtimeMetrics:
        """Collect comprehensive Realtime API metrics"""
        try:
            # Default values
            metrics = RealtimeMetrics(
                total_connections=0,
                active_sessions=0,
                average_latency_ms=0.0,
                p95_latency_ms=0.0,
                success_rate=0.0,
                error_rate=0.0,
                total_cost_usd=0.0,
                tokens_per_second=0.0,
                connection_pool_utilization=0.0
            )
            
            # Collect from metrics collector
            if self.metrics_collector:
                summary = self.metrics_collector.get_metrics_summary()
                
                metrics.average_latency_ms = summary.get('latency', {}).get('voice_to_voice_avg', 0.0)
                metrics.p95_latency_ms = summary.get('latency', {}).get('voice_to_voice_p95', 0.0)
                metrics.success_rate = summary.get('connection', {}).get('success_rate', 0.0)
                metrics.error_rate = 100.0 - metrics.success_rate
                metrics.total_cost_usd = summary.get('cost', {}).get('total_cost', 0.0)
            
            # Collect from session manager
            if self.session_manager:
                active_sessions = self.session_manager.get_active_sessions()
                metrics.active_sessions = len(active_sessions)
            
            # Collect from connection pool
            if self.connection_pool:
                try:
                    pool_stats = self.connection_pool.get_connection_health_summary()
                    connection_stats = pool_stats.get('connection_stats', {})
                    
                    metrics.total_connections = connection_stats.get('total_connections', 0)
                    # Calculate utilization from connection health data
                    active = connection_stats.get('active_connections', 0)
                    total = connection_stats.get('total_connections', 1)
                    metrics.connection_pool_utilization = (active / total) * 100 if total > 0 else 0.0
                except Exception:
                    pass
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Failed to collect realtime metrics: {e}")
            return RealtimeMetrics(
                total_connections=0, active_sessions=0, average_latency_ms=0.0,
                p95_latency_ms=0.0, success_rate=0.0, error_rate=100.0,
                total_cost_usd=0.0, tokens_per_second=0.0, connection_pool_utilization=0.0
            )
    
    def _update_prometheus_metrics(self):
        """Update Prometheus metrics with current values"""
        if not HAS_PROMETHEUS or not hasattr(self, 'prometheus_metrics'):
            return
        
        try:
            # Update uptime
            self.prometheus_metrics['service_uptime'].set(time.time() - self.start_time)
            
            # Update active connections
            if self.session_manager:
                active_sessions = len(self.session_manager.get_active_sessions())
                self.prometheus_metrics['active_connections'].set(active_sessions)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update Prometheus metrics: {e}")
    
    def set_service_state(self, state: ServiceState):
        """Update service state"""
        old_state = self.service_state
        self.service_state = state
        
        self.logger.info(f"🔄 Service state changed: {old_state.value} → {state.value}")
    
    def start_monitoring(self, host: str = '0.0.0.0', port: int = 8080):
        """Start the health monitoring HTTP server"""
        try:
            self.is_running = True
            self.set_service_state(ServiceState.READY)
            
            self.logger.info(f"🏥 Starting health monitor on http://{host}:{port}")
            
            # Start Flask app
            self.app.run(
                host=host,
                port=port,
                debug=False,
                threaded=True,
                use_reloader=False
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start health monitor: {e}")
            self.set_service_state(ServiceState.ERROR)
            raise
    
    def stop_monitoring(self):
        """Stop the health monitoring system"""
        try:
            self.set_service_state(ServiceState.STOPPING)
            self.is_running = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            
            self.set_service_state(ServiceState.STOPPED)
            self.logger.info("🏥 Health monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping health monitor: {e}")


# Factory function for easy integration
def create_realtime_health_monitor(
    config: RealtimeAPIConfig,
    **integrations
) -> RealtimeHealthMonitor:
    """Create and configure a RealtimeHealthMonitor instance"""
    return RealtimeHealthMonitor(
        config=config,
        **integrations
    ) 