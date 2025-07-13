"""
Health Monitoring System for Sovereign Voice Assistant
Provides health check endpoints, service status tracking, and real-time health metrics
"""

import asyncio
import time
import json
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor

from .error_handling import (
    ModernCircuitBreaker, 
    CircuitState, 
    VoiceAIException, 
    ErrorCategory
)
from .structured_logging import VoiceAILogger, get_voice_ai_logger

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class ServiceHealthMetrics:
    """Individual service health metrics"""
    service_name: str
    status: HealthStatus
    response_time_ms: float
    success_rate: float
    error_count: int
    last_check_time: float
    last_success_time: float
    last_error_time: float
    circuit_breaker_state: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None

@dataclass
class SystemHealthMetrics:
    """Overall system health metrics"""
    overall_status: HealthStatus
    timestamp: float
    uptime_seconds: float
    total_requests: int
    error_rate: float
    avg_response_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    disk_usage_percent: float
    network_connectivity: bool
    services: Dict[str, ServiceHealthMetrics]
    alerts: List[str]

class HealthChecker:
    """Individual service health checker"""
    
    def __init__(
        self,
        service_name: str,
        health_check_func: Callable[[], Awaitable[bool]],
        check_interval: float = 30.0,
        timeout: float = 10.0,
        degraded_threshold: float = 0.8,
        unhealthy_threshold: float = 0.5
    ):
        self.service_name = service_name
        self.health_check_func = health_check_func
        self.check_interval = check_interval
        self.timeout = timeout
        self.degraded_threshold = degraded_threshold
        self.unhealthy_threshold = unhealthy_threshold
        
        # Health metrics
        self.metrics = ServiceHealthMetrics(
            service_name=service_name,
            status=HealthStatus.UNKNOWN,
            response_time_ms=0.0,
            success_rate=0.0,
            error_count=0,
            last_check_time=0.0,
            last_success_time=0.0,
            last_error_time=0.0
        )
        
        # Historical data for success rate calculation
        self.recent_results = []  # List of (timestamp, success) tuples
        self.max_history_size = 100
        
        # Control variables
        self.running = False
        self.check_task = None
        
        # Logger
        self.logger = get_voice_ai_logger(f"health_checker.{service_name}")
    
    async def start(self):
        """Start health checking"""
        if self.running:
            return
        
        self.running = True
        self.check_task = asyncio.create_task(self._check_loop())
        self.logger.info(f"Started health checking for {self.service_name}")
    
    async def stop(self):
        """Stop health checking"""
        if not self.running:
            return
        
        self.running = False
        if self.check_task:
            self.check_task.cancel()
            try:
                await self.check_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info(f"Stopped health checking for {self.service_name}")
    
    async def _check_loop(self):
        """Main health checking loop"""
        while self.running:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop for {self.service_name}: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _perform_health_check(self):
        """Perform a single health check"""
        start_time = time.time()
        
        try:
            # Execute health check with timeout
            result = await asyncio.wait_for(
                self.health_check_func(),
                timeout=self.timeout
            )
            
            response_time_ms = (time.time() - start_time) * 1000
            current_time = time.time()
            
            # Update metrics
            self.metrics.response_time_ms = response_time_ms
            self.metrics.last_check_time = current_time
            
            if result:
                self.metrics.last_success_time = current_time
                self.recent_results.append((current_time, True))
            else:
                self.metrics.error_count += 1
                self.metrics.last_error_time = current_time
                self.recent_results.append((current_time, False))
            
            # Trim history
            self._trim_history()
            
            # Update success rate and status
            self._update_success_rate()
            self._update_status()
            
            # Log result
            status_emoji = "✅" if result else "❌"
            self.logger.info(
                f"{status_emoji} Health check {self.service_name}: "
                f"{self.metrics.status.value} ({response_time_ms:.1f}ms)"
            )
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Health check timeout for {self.service_name}")
            self._handle_check_failure("timeout")
        except Exception as e:
            self.logger.error(f"Health check failed for {self.service_name}: {e}")
            self._handle_check_failure("error")
    
    def _handle_check_failure(self, reason: str):
        """Handle health check failure"""
        current_time = time.time()
        self.metrics.error_count += 1
        self.metrics.last_error_time = current_time
        self.metrics.last_check_time = current_time
        self.recent_results.append((current_time, False))
        
        self._trim_history()
        self._update_success_rate()
        self._update_status()
    
    def _trim_history(self):
        """Trim historical results to max size"""
        if len(self.recent_results) > self.max_history_size:
            self.recent_results = self.recent_results[-self.max_history_size:]
    
    def _update_success_rate(self):
        """Update success rate based on recent results"""
        if not self.recent_results:
            self.metrics.success_rate = 0.0
            return
        
        # Calculate success rate from recent results
        successes = sum(1 for _, success in self.recent_results if success)
        self.metrics.success_rate = successes / len(self.recent_results)
    
    def _update_status(self):
        """Update health status based on success rate"""
        if self.metrics.success_rate >= self.degraded_threshold:
            self.metrics.status = HealthStatus.HEALTHY
        elif self.metrics.success_rate >= self.unhealthy_threshold:
            self.metrics.status = HealthStatus.DEGRADED
        else:
            self.metrics.status = HealthStatus.UNHEALTHY
    
    def get_metrics(self) -> ServiceHealthMetrics:
        """Get current health metrics"""
        return self.metrics

class SystemHealthMonitor:
    """Overall system health monitoring"""
    
    def __init__(self):
        self.service_checkers: Dict[str, HealthChecker] = {}
        self.circuit_breakers: Dict[str, ModernCircuitBreaker] = {}
        self.start_time = time.time()
        self.total_requests = 0
        self.total_errors = 0
        self.response_times = []
        self.max_response_time_history = 1000
        
        # System monitoring
        self.system_metrics_interval = 60.0  # 1 minute
        self.system_metrics_task = None
        self.current_system_metrics = {}
        
        # Health check server
        self.health_server = None
        self.health_server_port = 8080
        
        # Logger
        self.logger = get_voice_ai_logger("system_health_monitor")
        
        # Running state
        self.running = False
    
    def register_service(
        self,
        service_name: str,
        health_check_func: Callable[[], Awaitable[bool]],
        circuit_breaker: Optional[ModernCircuitBreaker] = None,
        **kwargs
    ):
        """Register a service for health monitoring"""
        checker = HealthChecker(service_name, health_check_func, **kwargs)
        self.service_checkers[service_name] = checker
        
        if circuit_breaker:
            self.circuit_breakers[service_name] = circuit_breaker
        
        self.logger.info(f"Registered service for health monitoring: {service_name}")
    
    async def start(self, health_server_port: int = 8080):
        """Start health monitoring"""
        if self.running:
            return
        
        self.running = True
        self.health_server_port = health_server_port
        
        # Start service checkers
        for checker in self.service_checkers.values():
            await checker.start()
        
        # Start system metrics collection
        self.system_metrics_task = asyncio.create_task(self._collect_system_metrics_loop())
        
        # Start health check server
        await self._start_health_server()
        
        self.logger.info(f"Started system health monitoring on port {health_server_port}")
    
    async def stop(self):
        """Stop health monitoring"""
        if not self.running:
            return
        
        self.running = False
        
        # Stop service checkers
        for checker in self.service_checkers.values():
            await checker.stop()
        
        # Stop system metrics collection
        if self.system_metrics_task:
            self.system_metrics_task.cancel()
            try:
                await self.system_metrics_task
            except asyncio.CancelledError:
                pass
        
        # Stop health server
        if self.health_server:
            await self.health_server.stop()
        
        self.logger.info("Stopped system health monitoring")
    
    async def _collect_system_metrics_loop(self):
        """Collect system metrics periodically"""
        while self.running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.system_metrics_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(self.system_metrics_interval)
    
    async def _collect_system_metrics(self):
        """Collect current system metrics"""
        def get_metrics():
            return {
                'memory_usage_mb': psutil.virtual_memory().used / (1024 * 1024),
                'cpu_usage_percent': psutil.cpu_percent(interval=1),
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'network_connectivity': self._check_network_connectivity()
            }
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            self.current_system_metrics = await loop.run_in_executor(executor, get_metrics)
    
    def _check_network_connectivity(self) -> bool:
        """Check basic network connectivity"""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False
    
    async def _start_health_server(self):
        """Start simple HTTP health check server"""
        from aiohttp import web
        
        app = web.Application()
        app.router.add_get('/health', self._health_endpoint)
        app.router.add_get('/health/detailed', self._detailed_health_endpoint)
        app.router.add_get('/health/services', self._services_health_endpoint)
        app.router.add_get('/metrics', self._metrics_endpoint)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', self.health_server_port)
        await site.start()
        
        self.health_server = runner
    
    async def _health_endpoint(self, request):
        """Simple health check endpoint"""
        from aiohttp import web
        
        metrics = await self.get_system_health()
        
        if metrics.overall_status == HealthStatus.HEALTHY:
            return web.json_response({"status": "healthy"}, status=200)
        elif metrics.overall_status == HealthStatus.DEGRADED:
            return web.json_response({"status": "degraded"}, status=200)
        else:
            return web.json_response({"status": "unhealthy"}, status=503)
    
    async def _detailed_health_endpoint(self, request):
        """Detailed health check endpoint"""
        from aiohttp import web
        
        metrics = await self.get_system_health()
        
        # Convert to dict for JSON serialization
        response_data = asdict(metrics)
        
        # Convert enum values to strings
        response_data['overall_status'] = metrics.overall_status.value
        for service_name, service_metrics in response_data['services'].items():
            service_metrics['status'] = service_metrics['status'].value if hasattr(service_metrics['status'], 'value') else service_metrics['status']
        
        status_code = 200 if metrics.overall_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED] else 503
        
        return web.json_response(response_data, status=status_code)
    
    async def _services_health_endpoint(self, request):
        """Services health endpoint"""
        from aiohttp import web
        
        services_health = {}
        for service_name, checker in self.service_checkers.items():
            metrics = checker.get_metrics()
            services_health[service_name] = {
                'status': metrics.status.value,
                'response_time_ms': metrics.response_time_ms,
                'success_rate': metrics.success_rate,
                'error_count': metrics.error_count,
                'last_check_time': metrics.last_check_time
            }
        
        return web.json_response(services_health)
    
    async def _metrics_endpoint(self, request):
        """Prometheus-style metrics endpoint"""
        from aiohttp import web
        
        metrics = await self.get_system_health()
        
        # Generate Prometheus-style metrics
        lines = [
            f"# HELP system_uptime_seconds System uptime in seconds",
            f"# TYPE system_uptime_seconds gauge",
            f"system_uptime_seconds {metrics.uptime_seconds}",
            "",
            f"# HELP system_memory_usage_mb Memory usage in megabytes",
            f"# TYPE system_memory_usage_mb gauge",
            f"system_memory_usage_mb {metrics.memory_usage_mb}",
            "",
            f"# HELP system_cpu_usage_percent CPU usage percentage",
            f"# TYPE system_cpu_usage_percent gauge",
            f"system_cpu_usage_percent {metrics.cpu_usage_percent}",
            "",
            f"# HELP system_requests_total Total number of requests",
            f"# TYPE system_requests_total counter",
            f"system_requests_total {metrics.total_requests}",
            "",
            f"# HELP system_error_rate Error rate (0.0 to 1.0)",
            f"# TYPE system_error_rate gauge",
            f"system_error_rate {metrics.error_rate}",
            ""
        ]
        
        # Add service-specific metrics
        for service_name, service_metrics in metrics.services.items():
            lines.extend([
                f"# HELP service_response_time_ms Response time in milliseconds",
                f"# TYPE service_response_time_ms gauge",
                f"service_response_time_ms{{service=\"{service_name}\"}} {service_metrics.response_time_ms}",
                "",
                f"# HELP service_success_rate Success rate (0.0 to 1.0)",
                f"# TYPE service_success_rate gauge",
                f"service_success_rate{{service=\"{service_name}\"}} {service_metrics.success_rate}",
                "",
                f"# HELP service_error_count_total Total error count",
                f"# TYPE service_error_count_total counter",
                f"service_error_count_total{{service=\"{service_name}\"}} {service_metrics.error_count}",
                ""
            ])
        
        return web.Response(text="\n".join(lines), content_type="text/plain")
    
    async def get_system_health(self) -> SystemHealthMetrics:
        """Get comprehensive system health metrics"""
        current_time = time.time()
        
        # Collect service metrics
        service_metrics = {}
        for service_name, checker in self.service_checkers.items():
            metrics = checker.get_metrics()
            
            # Add circuit breaker state if available
            if service_name in self.circuit_breakers:
                cb = self.circuit_breakers[service_name]
                metrics.circuit_breaker_state = cb.state.value
            
            service_metrics[service_name] = metrics
        
        # Determine overall status
        overall_status = self._determine_overall_status(service_metrics)
        
        # Calculate system metrics
        uptime_seconds = current_time - self.start_time
        error_rate = self.total_errors / self.total_requests if self.total_requests > 0 else 0.0
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0.0
        
        # Generate alerts
        alerts = self._generate_alerts(service_metrics, overall_status)
        
        return SystemHealthMetrics(
            overall_status=overall_status,
            timestamp=current_time,
            uptime_seconds=uptime_seconds,
            total_requests=self.total_requests,
            error_rate=error_rate,
            avg_response_time_ms=avg_response_time,
            memory_usage_mb=self.current_system_metrics.get('memory_usage_mb', 0),
            cpu_usage_percent=self.current_system_metrics.get('cpu_usage_percent', 0),
            disk_usage_percent=self.current_system_metrics.get('disk_usage_percent', 0),
            network_connectivity=self.current_system_metrics.get('network_connectivity', False),
            services=service_metrics,
            alerts=alerts
        )
    
    def _determine_overall_status(self, service_metrics: Dict[str, ServiceHealthMetrics]) -> HealthStatus:
        """Determine overall system health status"""
        if not service_metrics:
            return HealthStatus.UNKNOWN
        
        statuses = [metrics.status for metrics in service_metrics.values()]
        
        # If any service is unhealthy, system is unhealthy
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        
        # If any service is degraded, system is degraded
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        
        # If all services are healthy, system is healthy
        if all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        
        return HealthStatus.UNKNOWN
    
    def _generate_alerts(self, service_metrics: Dict[str, ServiceHealthMetrics], overall_status: HealthStatus) -> List[str]:
        """Generate alerts based on health status"""
        alerts = []
        
        # System-level alerts
        if overall_status == HealthStatus.UNHEALTHY:
            alerts.append("System is in unhealthy state")
        elif overall_status == HealthStatus.DEGRADED:
            alerts.append("System is in degraded state")
        
        # Service-level alerts
        for service_name, metrics in service_metrics.items():
            if metrics.status == HealthStatus.UNHEALTHY:
                alerts.append(f"Service {service_name} is unhealthy (success rate: {metrics.success_rate:.2%})")
            elif metrics.status == HealthStatus.DEGRADED:
                alerts.append(f"Service {service_name} is degraded (success rate: {metrics.success_rate:.2%})")
            
            if metrics.response_time_ms > 5000:  # 5 second threshold
                alerts.append(f"Service {service_name} has high response time ({metrics.response_time_ms:.1f}ms)")
        
        # System resource alerts
        memory_usage = self.current_system_metrics.get('memory_usage_mb', 0)
        if memory_usage > 8000:  # 8GB threshold
            alerts.append(f"High memory usage: {memory_usage:.1f}MB")
        
        cpu_usage = self.current_system_metrics.get('cpu_usage_percent', 0)
        if cpu_usage > 80:  # 80% threshold
            alerts.append(f"High CPU usage: {cpu_usage:.1f}%")
        
        disk_usage = self.current_system_metrics.get('disk_usage_percent', 0)
        if disk_usage > 90:  # 90% threshold
            alerts.append(f"High disk usage: {disk_usage:.1f}%")
        
        if not self.current_system_metrics.get('network_connectivity', True):
            alerts.append("Network connectivity issues detected")
        
        return alerts
    
    def record_request(self, response_time_ms: float, success: bool):
        """Record request metrics"""
        self.total_requests += 1
        if not success:
            self.total_errors += 1
        
        self.response_times.append(response_time_ms)
        if len(self.response_times) > self.max_response_time_history:
            self.response_times = self.response_times[-self.max_response_time_history:]
    
    def get_service_health(self, service_name: str) -> Optional[ServiceHealthMetrics]:
        """Get health metrics for a specific service"""
        if service_name in self.service_checkers:
            return self.service_checkers[service_name].get_metrics()
        return None

# Global system health monitor
_global_health_monitor: Optional[SystemHealthMonitor] = None

def get_health_monitor() -> SystemHealthMonitor:
    """Get or create global health monitor"""
    global _global_health_monitor
    if _global_health_monitor is None:
        _global_health_monitor = SystemHealthMonitor()
    return _global_health_monitor

def set_health_monitor(monitor: SystemHealthMonitor):
    """Set global health monitor"""
    global _global_health_monitor
    _global_health_monitor = monitor

# Context manager for health monitoring
@asynccontextmanager
async def health_monitored_operation(
    operation_name: str,
    health_monitor: Optional[SystemHealthMonitor] = None
):
    """Context manager for health-monitored operations"""
    monitor = health_monitor or get_health_monitor()
    start_time = time.time()
    
    try:
        yield
        
        # Record successful operation
        response_time_ms = (time.time() - start_time) * 1000
        monitor.record_request(response_time_ms, True)
        
    except Exception as e:
        # Record failed operation
        response_time_ms = (time.time() - start_time) * 1000
        monitor.record_request(response_time_ms, False)
        raise

# Fast internal health check functions
async def stt_health_check() -> bool:
    """Health check for STT service - fast internal check"""
    try:
        # Check if OpenAI API key is available (faster than external call)
        import os
        from dotenv import load_dotenv
        load_dotenv()
        return bool(os.getenv('OPENAI_API_KEY'))
    except:
        return False

async def llm_health_check() -> bool:
    """Health check for LLM service - fast internal check"""
    try:
        # Check if OpenRouter API key is available (faster than external call)
        import os
        from dotenv import load_dotenv
        load_dotenv()
        return bool(os.getenv('OPENROUTER_API_KEY'))
    except:
        return False

async def tts_health_check() -> bool:
    """Health check for TTS service - fast internal check"""
    try:
        # Check if OpenAI API key is available (faster than external call)
        import os
        from dotenv import load_dotenv
        load_dotenv()
        return bool(os.getenv('OPENAI_API_KEY'))
    except:
        return False

async def offline_system_health_check() -> bool:
    """Health check for offline system"""
    try:
        # Always return True for now - offline system is optional
        return True
    except:
        return False 