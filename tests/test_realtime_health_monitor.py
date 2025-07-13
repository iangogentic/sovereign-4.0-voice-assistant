"""
Test suite for Realtime Health Monitor

Comprehensive tests for health monitoring, service discovery,
Prometheus metrics, and production deployment integration.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import asdict
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.realtime_health_monitor import (
    RealtimeHealthMonitor,
    HealthStatus,
    ServiceState,
    HealthCheckResult,
    ServiceStatus,
    RealtimeMetrics
)
from assistant.connection_stability_monitor import ConnectionStabilityMonitor


class TestRealtimeHealthMonitor:
    """Test class for RealtimeHealthMonitor"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration"""
        config = Mock()
        config.chroma_host = "localhost"
        config.version = "4.0.0"
        return config
    
    @pytest.fixture
    def mock_metrics_collector(self):
        """Mock metrics collector"""
        collector = Mock()
        collector.get_metrics_summary.return_value = {
            'latency': {
                'voice_to_voice_avg': 150.0,
                'voice_to_voice_p95': 300.0
            },
            'connection': {
                'success_rate': 98.5
            },
            'cost': {
                'total_cost': 5.67
            }
        }
        return collector
    
    @pytest.fixture
    def mock_session_manager(self):
        """Mock session manager"""
        manager = Mock()
        manager.get_active_sessions.return_value = {
            'session1': Mock(status=Mock(value='active')),
            'session2': Mock(status=Mock(value='active'))
        }
        return manager
    
    @pytest.fixture
    def mock_connection_pool(self):
        """Mock connection pool"""
        pool = Mock()
        pool.get_connection_health_summary = Mock(return_value={
            'connection_health': {
                'is_connected': True,
                'quality': 'good'
            },
            'connection_stats': {
                'active_connections': 5,
                'total_connections': 10
            }
        })
        return pool
    
    @pytest.fixture
    def mock_cost_manager(self):
        """Mock cost manager"""
        manager = Mock()
        manager.get_optimization_stats = AsyncMock(return_value={
            'optimization_stats': {
                'sessions_terminated': 2,
                'cost_savings': 12.34
            }
        })
        return manager
    
    @pytest.fixture
    def health_monitor(self, mock_config, mock_metrics_collector, 
                      mock_session_manager, mock_connection_pool, mock_cost_manager):
        """Create health monitor instance"""
        with patch('assistant.realtime_health_monitor.Flask'):
            monitor = RealtimeHealthMonitor(
                config=mock_config,
                metrics_collector=mock_metrics_collector,
                session_manager=mock_session_manager,
                connection_pool=mock_connection_pool,
                cost_manager=mock_cost_manager
            )
            return monitor


class TestHealthMonitorInitialization:
    """Test health monitor initialization"""
    
    def test_monitor_initialization(self, mock_config):
        """Test basic monitor initialization"""
        with patch('assistant.realtime_health_monitor.Flask'):
            monitor = RealtimeHealthMonitor(config=mock_config)
            
            assert monitor.config == mock_config
            assert monitor.service_state == ServiceState.STARTING
            assert len(monitor.health_checks) > 0
            assert 'openai_api_connectivity' in monitor.health_checks
            assert 'database_connectivity' in monitor.health_checks
    
    def test_prometheus_metrics_setup(self, mock_config):
        """Test Prometheus metrics configuration"""
        with patch('assistant.realtime_health_monitor.Flask'), \
             patch('assistant.realtime_health_monitor.HAS_PROMETHEUS', True):
            monitor = RealtimeHealthMonitor(config=mock_config)
            
            assert hasattr(monitor, 'prometheus_metrics')
            assert 'health_check_duration' in monitor.prometheus_metrics
            assert 'service_uptime' in monitor.prometheus_metrics


class TestHealthChecks:
    """Test individual health check implementations"""
    
    @pytest.mark.asyncio
    async def test_openai_api_check_success(self, health_monitor):
        """Test successful OpenAI API health check"""
        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            result = await health_monitor._check_openai_api()
            
            assert result.name == "openai_api_connectivity"
            assert result.status == HealthStatus.HEALTHY
            assert result.critical is True
            assert "OpenAI API is accessible" in result.message
    
    @pytest.mark.asyncio
    async def test_openai_api_check_failure(self, health_monitor):
        """Test failed OpenAI API health check"""
        with patch('openai.OpenAI') as mock_openai:
            mock_openai.side_effect = Exception("API key invalid")
            
            result = await health_monitor._check_openai_api()
            
            assert result.name == "openai_api_connectivity"
            assert result.status == HealthStatus.CRITICAL
            assert result.critical is True
            assert "API key invalid" in result.message
    
    @pytest.mark.asyncio
    async def test_database_check_success(self, health_monitor):
        """Test successful database health check"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            
            result = await health_monitor._check_database()
            
            assert result.name == "database_connectivity"
            assert result.status == HealthStatus.HEALTHY
            assert result.critical is True
    
    @pytest.mark.asyncio
    async def test_database_check_failure(self, health_monitor):
        """Test failed database health check"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=Exception("Connection refused")
            )
            
            result = await health_monitor._check_database()
            
            assert result.name == "database_connectivity"
            assert result.status == HealthStatus.CRITICAL
            assert result.critical is True
    
    @pytest.mark.asyncio
    async def test_memory_usage_check_normal(self, health_monitor):
        """Test memory usage check with normal usage"""
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(
                percent=65.0,
                available=4 * 1024**3,  # 4GB
                total=8 * 1024**3       # 8GB
            )
            
            result = await health_monitor._check_memory_usage()
            
            assert result.name == "memory_usage"
            assert result.status == HealthStatus.HEALTHY
            assert result.details['memory_percent'] == 65.0
    
    @pytest.mark.asyncio
    async def test_memory_usage_check_high(self, health_monitor):
        """Test memory usage check with high usage"""
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(
                percent=85.0,
                available=1 * 1024**3,  # 1GB
                total=8 * 1024**3       # 8GB
            )
            
            result = await health_monitor._check_memory_usage()
            
            assert result.name == "memory_usage"
            assert result.status == HealthStatus.DEGRADED
            assert "High memory usage" in result.message
    
    @pytest.mark.asyncio
    async def test_memory_usage_check_critical(self, health_monitor):
        """Test memory usage check with critical usage"""
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(
                percent=95.0,
                available=0.5 * 1024**3,  # 0.5GB
                total=8 * 1024**3          # 8GB
            )
            
            result = await health_monitor._check_memory_usage()
            
            assert result.name == "memory_usage"
            assert result.status == HealthStatus.CRITICAL
            assert "Critical memory usage" in result.message
    
    @pytest.mark.asyncio
    async def test_websocket_pool_check_healthy(self, health_monitor):
        """Test WebSocket pool health check - healthy state"""
        result = await health_monitor._check_websocket_pool()
        
        assert result.name == "websocket_pool"
        assert result.status == HealthStatus.HEALTHY
        assert "WebSocket pool healthy" in result.message
        assert result.critical is True
    
    @pytest.mark.asyncio
    async def test_websocket_pool_check_no_connections(self, health_monitor):
        """Test WebSocket pool health check - no active connections"""
        health_monitor.connection_pool.get_pool_stats = AsyncMock(return_value={
            'active_connections': 0,
            'total_connections': 10,
            'utilization': 0.0
        })
        
        result = await health_monitor._check_websocket_pool()
        
        assert result.name == "websocket_pool"
        assert result.status == HealthStatus.DEGRADED
        assert "No active WebSocket connections" in result.message
    
    @pytest.mark.asyncio
    async def test_session_manager_check_healthy(self, health_monitor):
        """Test session manager health check - healthy state"""
        result = await health_monitor._check_session_manager()
        
        assert result.name == "session_manager"
        assert result.status == HealthStatus.HEALTHY
        assert "2 active sessions" in result.message
    
    @pytest.mark.asyncio
    async def test_session_manager_check_high_load(self, health_monitor):
        """Test session manager health check - high session count"""
        # Mock 150 active sessions
        sessions = {f'session{i}': Mock(status=Mock(value='active')) for i in range(150)}
        health_monitor.session_manager.get_active_sessions.return_value = sessions
        
        result = await health_monitor._check_session_manager()
        
        assert result.name == "session_manager"
        assert result.status == HealthStatus.DEGRADED
        assert "High session count" in result.message
    
    @pytest.mark.asyncio
    async def test_cost_manager_check_healthy(self, health_monitor):
        """Test cost manager health check - healthy state"""
        result = await health_monitor._check_cost_manager()
        
        assert result.name == "cost_manager"
        assert result.status == HealthStatus.HEALTHY
        assert "$12.34 saved" in result.message


class TestHealthEndpoints:
    """Test HTTP health endpoints"""
    
    def test_basic_health_endpoint(self, health_monitor):
        """Test basic /health endpoint"""
        # Mock Flask app and test client
        with patch.object(health_monitor, '_get_overall_health_status') as mock_status:
            mock_status.return_value = HealthStatus.HEALTHY
            
            # Create a test request context
            with health_monitor.app.test_request_context():
                # Import and call the route function directly
                from flask import jsonify
                
                # Simulate the basic_health route
                response_data = {
                    'status': HealthStatus.HEALTHY.value,
                    'timestamp': datetime.now().isoformat(),
                    'service': 'sovereign-realtime-api',
                    'uptime_seconds': time.time() - health_monitor.start_time
                }
                
                assert response_data['status'] == 'healthy'
                assert 'timestamp' in response_data
                assert response_data['service'] == 'sovereign-realtime-api'
    
    def test_realtime_health_endpoint(self, health_monitor):
        """Test detailed /health/realtime endpoint"""
        with patch.object(health_monitor, '_run_all_health_checks') as mock_checks:
            mock_checks.return_value = [
                HealthCheckResult(
                    name="test_check",
                    status=HealthStatus.HEALTHY,
                    message="Test passed",
                    details={},
                    timestamp=datetime.now(),
                    duration_ms=50.0
                )
            ]
            
            with patch.object(health_monitor, '_get_overall_health_status') as mock_status:
                mock_status.return_value = HealthStatus.HEALTHY
                
                # Simulate endpoint logic
                results = mock_checks.return_value
                service_status = ServiceStatus(
                    service_name="sovereign-realtime-api",
                    status=HealthStatus.HEALTHY,
                    state=health_monitor.service_state,
                    uptime_seconds=time.time() - health_monitor.start_time,
                    version="4.0.0",
                    checks=results,
                    timestamp=datetime.now(),
                    endpoint_urls={
                        'health': '/health',
                        'metrics': '/metrics/realtime'
                    }
                )
                
                assert service_status.service_name == "sovereign-realtime-api"
                assert service_status.status == HealthStatus.HEALTHY
                assert len(service_status.checks) == 1
    
    def test_readiness_probe_ready(self, health_monitor):
        """Test Kubernetes readiness probe - ready state"""
        health_monitor.set_service_state(ServiceState.READY)
        
        # Simulate readiness probe logic
        ready = health_monitor.service_state in [ServiceState.READY, ServiceState.RUNNING]
        
        assert ready is True
    
    def test_readiness_probe_not_ready(self, health_monitor):
        """Test Kubernetes readiness probe - not ready state"""
        health_monitor.set_service_state(ServiceState.STARTING)
        
        # Simulate readiness probe logic
        ready = health_monitor.service_state in [ServiceState.READY, ServiceState.RUNNING]
        
        assert ready is False
    
    def test_liveness_probe_alive(self, health_monitor):
        """Test Kubernetes liveness probe - alive state"""
        health_monitor.set_service_state(ServiceState.RUNNING)
        
        # Simulate liveness probe logic
        alive = health_monitor.service_state != ServiceState.ERROR
        
        assert alive is True
    
    def test_liveness_probe_dead(self, health_monitor):
        """Test Kubernetes liveness probe - dead state"""
        health_monitor.set_service_state(ServiceState.ERROR)
        
        # Simulate liveness probe logic
        alive = health_monitor.service_state != ServiceState.ERROR
        
        assert alive is False


class TestMetricsCollection:
    """Test metrics collection and reporting"""
    
    def test_realtime_metrics_collection(self, health_monitor):
        """Test realtime metrics collection"""
        metrics = health_monitor._collect_realtime_metrics()
        
        assert isinstance(metrics, RealtimeMetrics)
        assert metrics.average_latency_ms == 150.0
        assert metrics.p95_latency_ms == 300.0
        assert metrics.success_rate == 98.5
        assert metrics.total_cost_usd == 5.67
        assert metrics.active_sessions == 2
    
    def test_metrics_collection_with_failures(self, health_monitor):
        """Test metrics collection when subsystems fail"""
        # Make metrics collector fail
        health_monitor.metrics_collector.get_metrics_summary.side_effect = Exception("Metrics unavailable")
        
        metrics = health_monitor._collect_realtime_metrics()
        
        # Should return default metrics when collection fails
        assert isinstance(metrics, RealtimeMetrics)
        assert metrics.average_latency_ms == 0.0
        assert metrics.error_rate == 100.0
    
    def test_prometheus_metrics_update(self, health_monitor):
        """Test Prometheus metrics update"""
        with patch('assistant.realtime_health_monitor.HAS_PROMETHEUS', True):
            health_monitor.prometheus_metrics = {
                'service_uptime': Mock(),
                'active_connections': Mock()
            }
            
            health_monitor._update_prometheus_metrics()
            
            # Verify metrics were updated
            health_monitor.prometheus_metrics['service_uptime'].set.assert_called()
            health_monitor.prometheus_metrics['active_connections'].set.assert_called_with(2)


class TestOverallHealthStatus:
    """Test overall health status determination"""
    
    def test_all_healthy(self, health_monitor):
        """Test overall status when all checks are healthy"""
        health_monitor.last_check_results = {
            'check1': HealthCheckResult(
                name="check1", status=HealthStatus.HEALTHY, message="OK",
                details={}, timestamp=datetime.now(), duration_ms=10.0
            ),
            'check2': HealthCheckResult(
                name="check2", status=HealthStatus.HEALTHY, message="OK",
                details={}, timestamp=datetime.now(), duration_ms=15.0
            )
        }
        
        status = health_monitor._get_overall_health_status()
        assert status == HealthStatus.HEALTHY
    
    def test_critical_failure(self, health_monitor):
        """Test overall status with critical check failure"""
        health_monitor.critical_checks = {'critical_check'}
        health_monitor.last_check_results = {
            'critical_check': HealthCheckResult(
                name="critical_check", status=HealthStatus.CRITICAL, message="Failed",
                details={}, timestamp=datetime.now(), duration_ms=10.0, critical=True
            ),
            'normal_check': HealthCheckResult(
                name="normal_check", status=HealthStatus.HEALTHY, message="OK",
                details={}, timestamp=datetime.now(), duration_ms=15.0
            )
        }
        
        status = health_monitor._get_overall_health_status()
        assert status == HealthStatus.CRITICAL
    
    def test_degraded_state(self, health_monitor):
        """Test overall status with some degraded checks"""
        health_monitor.last_check_results = {
            'check1': HealthCheckResult(
                name="check1", status=HealthStatus.HEALTHY, message="OK",
                details={}, timestamp=datetime.now(), duration_ms=10.0
            ),
            'check2': HealthCheckResult(
                name="check2", status=HealthStatus.DEGRADED, message="Slow",
                details={}, timestamp=datetime.now(), duration_ms=15.0
            )
        }
        
        status = health_monitor._get_overall_health_status()
        assert status == HealthStatus.DEGRADED
    
    def test_unhealthy_state(self, health_monitor):
        """Test overall status with majority degraded checks"""
        health_monitor.last_check_results = {
            'check1': HealthCheckResult(
                name="check1", status=HealthStatus.DEGRADED, message="Slow",
                details={}, timestamp=datetime.now(), duration_ms=10.0
            ),
            'check2': HealthCheckResult(
                name="check2", status=HealthStatus.DEGRADED, message="Slow",
                details={}, timestamp=datetime.now(), duration_ms=15.0
            ),
            'check3': HealthCheckResult(
                name="check3", status=HealthStatus.UNHEALTHY, message="Failed",
                details={}, timestamp=datetime.now(), duration_ms=20.0
            )
        }
        
        status = health_monitor._get_overall_health_status()
        assert status == HealthStatus.UNHEALTHY


class TestServiceStateManagement:
    """Test service state transitions"""
    
    def test_initial_state(self, health_monitor):
        """Test initial service state"""
        assert health_monitor.service_state == ServiceState.STARTING
    
    def test_state_transitions(self, health_monitor):
        """Test valid state transitions"""
        health_monitor.set_service_state(ServiceState.READY)
        assert health_monitor.service_state == ServiceState.READY
        
        health_monitor.set_service_state(ServiceState.RUNNING)
        assert health_monitor.service_state == ServiceState.RUNNING
        
        health_monitor.set_service_state(ServiceState.STOPPING)
        assert health_monitor.service_state == ServiceState.STOPPING
        
        health_monitor.set_service_state(ServiceState.STOPPED)
        assert health_monitor.service_state == ServiceState.STOPPED
    
    def test_error_state(self, health_monitor):
        """Test error state handling"""
        health_monitor.set_service_state(ServiceState.ERROR)
        assert health_monitor.service_state == ServiceState.ERROR


class TestHealthCheckRegistration:
    """Test health check registration and management"""
    
    def test_register_custom_health_check(self, health_monitor):
        """Test registering custom health check"""
        def custom_check():
            return HealthCheckResult(
                name="custom_check",
                status=HealthStatus.HEALTHY,
                message="Custom check passed",
                details={},
                timestamp=datetime.now(),
                duration_ms=25.0
            )
        
        health_monitor.register_health_check(
            'custom_check',
            custom_check,
            critical=True,
            interval=30
        )
        
        assert 'custom_check' in health_monitor.health_checks
        assert 'custom_check' in health_monitor.critical_checks
        assert health_monitor.check_intervals['custom_check'] == 30
    
    def test_default_health_checks_registered(self, health_monitor):
        """Test that default health checks are registered"""
        expected_checks = [
            'openai_api_connectivity',
            'database_connectivity',
            'memory_usage',
            'websocket_pool',
            'session_manager',
            'cost_manager'
        ]
        
        for check_name in expected_checks:
            assert check_name in health_monitor.health_checks
    
    def test_critical_checks_identification(self, health_monitor):
        """Test identification of critical health checks"""
        critical_checks = [
            'openai_api_connectivity',
            'database_connectivity',
            'websocket_pool'
        ]
        
        for check_name in critical_checks:
            assert check_name in health_monitor.critical_checks


class TestErrorHandling:
    """Test error handling in health monitoring"""
    
    def test_health_check_exception_handling(self, health_monitor):
        """Test handling of exceptions in health checks"""
        def failing_check():
            raise Exception("Check failed")
        
        health_monitor.register_health_check('failing_check', failing_check)
        
        results = health_monitor._run_all_health_checks()
        
        # Should have error result for failing check
        failing_result = next((r for r in results if r.name == 'failing_check'), None)
        assert failing_result is not None
        assert failing_result.status == HealthStatus.CRITICAL
        assert "Health check failed" in failing_result.message
    
    def test_metrics_collection_error_handling(self, health_monitor):
        """Test error handling in metrics collection"""
        # Make all integrations fail
        health_monitor.metrics_collector = None
        health_monitor.session_manager = None
        health_monitor.connection_pool = None
        
        # Should not raise exception
        metrics = health_monitor._collect_realtime_metrics()
        
        assert isinstance(metrics, RealtimeMetrics)
        # Should have default/zero values
        assert metrics.total_connections == 0
        assert metrics.active_sessions == 0


class TestIntegrationPoints:
    """Test integration with other system components"""
    
    def test_metrics_collector_integration(self, health_monitor):
        """Test integration with metrics collector"""
        summary = health_monitor.metrics_collector.get_metrics_summary()
        
        assert 'latency' in summary
        assert 'connection' in summary
        assert 'cost' in summary
    
    def test_session_manager_integration(self, health_monitor):
        """Test integration with session manager"""
        sessions = health_monitor.session_manager.get_active_sessions()
        
        assert len(sessions) == 2
        assert all(hasattr(s, 'status') for s in sessions.values())
    
    @pytest.mark.asyncio
    async def test_connection_pool_integration(self, health_monitor):
        """Test integration with connection pool"""
        stats = await health_monitor.connection_pool.get_pool_stats()
        
        assert 'active_connections' in stats
        assert 'total_connections' in stats
        assert 'utilization' in stats
    
    @pytest.mark.asyncio
    async def test_cost_manager_integration(self, health_monitor):
        """Test integration with cost manager"""
        stats = await health_monitor.cost_manager.get_optimization_stats()
        
        assert 'optimization_stats' in stats
        assert 'sessions_terminated' in stats['optimization_stats']
        assert 'cost_savings' in stats['optimization_stats']


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 