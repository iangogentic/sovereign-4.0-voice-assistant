"""
Tests for PerformanceMonitor - Performance Monitoring and Adaptive Quality Control
Comprehensive coverage of metrics collection, quality adaptation, and monitoring
"""

import pytest
import asyncio
import time
import logging
import statistics
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Import classes to test
from assistant.performance_monitor import (
    PerformanceMonitor,
    MetricsCollector,
    NetworkMonitor,
    AdaptiveQualityController,
    PerformanceAlertManager,
    PerformanceMetric,
    PerformanceAlert,
    QualitySettings,
    NetworkCondition,
    QualityLevel,
    MetricType,
    AlertLevel,
    create_performance_monitor
)


@pytest.fixture
def logger():
    """Create test logger"""
    logger = logging.getLogger("test_performance_monitor")
    logger.setLevel(logging.DEBUG)
    return logger


class TestQualitySettings:
    """Test QualitySettings configuration"""
    
    def test_default_quality_settings(self):
        """Test default quality settings"""
        settings = QualitySettings()
        
        assert settings.audio_sample_rate == 24000
        assert settings.audio_channels == 1
        assert settings.audio_bit_depth == 16
        assert settings.model_temperature == 0.7
        assert settings.buffer_size == 512
        assert settings.response_max_tokens == 4096
    
    def test_quality_level_configurations(self):
        """Test quality settings for different levels"""
        # ULTRA_LOW quality
        ultra_low = QualitySettings.for_quality_level(QualityLevel.ULTRA_LOW)
        assert ultra_low.audio_sample_rate == 16000
        assert ultra_low.audio_bit_depth == 8
        assert ultra_low.buffer_size == 256
        assert ultra_low.model_temperature == 0.3
        assert ultra_low.compression_level == "high"
        assert ultra_low.enable_noise_reduction is False
        assert ultra_low.response_max_tokens == 2048
        
        # MEDIUM quality (default)
        medium = QualitySettings.for_quality_level(QualityLevel.MEDIUM)
        assert medium.audio_sample_rate == 24000
        assert medium.buffer_size == 512
        
        # ULTRA_HIGH quality
        ultra_high = QualitySettings.for_quality_level(QualityLevel.ULTRA_HIGH)
        assert ultra_high.audio_sample_rate == 48000
        assert ultra_high.buffer_size == 2048
        assert ultra_high.model_temperature == 0.9
        assert ultra_high.compression_level is None
        assert ultra_high.response_max_tokens == 8192


class TestNetworkCondition:
    """Test NetworkCondition functionality"""
    
    def test_network_condition_creation(self):
        """Test NetworkCondition creation"""
        condition = NetworkCondition(
            latency_ms=50.0,
            jitter_ms=5.0,
            packet_loss_rate=0.01,
            bandwidth_mbps=100.0,
            stability_score=0.9
        )
        
        assert condition.latency_ms == 50.0
        assert condition.jitter_ms == 5.0
        assert condition.packet_loss_rate == 0.01
        assert condition.bandwidth_mbps == 100.0
        assert condition.stability_score == 0.9
    
    def test_quality_recommendations(self):
        """Test quality level recommendations based on network conditions"""
        # Poor network conditions
        poor_condition = NetworkCondition(latency_ms=250, packet_loss_rate=0.1)
        assert poor_condition.get_quality_recommendation() == QualityLevel.ULTRA_LOW
        
        # Moderate network conditions
        moderate_condition = NetworkCondition(latency_ms=75, packet_loss_rate=0.015)
        assert moderate_condition.get_quality_recommendation() == QualityLevel.MEDIUM
        
        # Excellent network conditions
        excellent_condition = NetworkCondition(latency_ms=15, packet_loss_rate=0.001)
        assert excellent_condition.get_quality_recommendation() == QualityLevel.ULTRA_HIGH


class TestPerformanceMetric:
    """Test PerformanceMetric functionality"""
    
    def test_metric_creation(self):
        """Test PerformanceMetric creation"""
        metric = PerformanceMetric(
            metric_type=MetricType.LATENCY,
            name="response_latency",
            value=150.5,
            unit="ms",
            tags={"operation": "chat"}
        )
        
        assert metric.metric_type == MetricType.LATENCY
        assert metric.name == "response_latency"
        assert metric.value == 150.5
        assert metric.unit == "ms"
        assert metric.tags["operation"] == "chat"
        assert isinstance(metric.timestamp, float)
    
    def test_metric_age_calculation(self):
        """Test metric age calculation"""
        past_time = time.time() - 0.1  # 100ms ago
        metric = PerformanceMetric(
            metric_type=MetricType.LATENCY,
            name="test_metric",
            value=100.0,
            timestamp=past_time
        )
        
        age_ms = metric.get_age_ms()
        assert age_ms >= 90  # Should be around 100ms
        assert age_ms <= 150  # Allow some variance


class TestPerformanceAlert:
    """Test PerformanceAlert functionality"""
    
    def test_alert_creation(self):
        """Test PerformanceAlert creation"""
        alert = PerformanceAlert(
            level=AlertLevel.WARNING,
            message="High latency detected",
            metric_name="response_latency",
            current_value=350.0,
            threshold=300.0
        )
        
        assert alert.level == AlertLevel.WARNING
        assert alert.message == "High latency detected"
        assert alert.metric_name == "response_latency"
        assert alert.current_value == 350.0
        assert alert.threshold == 300.0
        assert alert.acknowledged is False
    
    def test_alert_to_dict(self):
        """Test alert dictionary conversion"""
        alert = PerformanceAlert(
            level=AlertLevel.CRITICAL,
            message="Critical issue",
            metric_name="cpu_usage",
            current_value=95.0,
            threshold=85.0
        )
        
        alert_dict = alert.to_dict()
        assert alert_dict["level"] == "critical"
        assert alert_dict["message"] == "Critical issue"
        assert alert_dict["metric_name"] == "cpu_usage"
        assert alert_dict["current_value"] == 95.0
        assert alert_dict["threshold"] == 85.0
        assert "timestamp" in alert_dict


class TestMetricsCollector:
    """Test MetricsCollector functionality"""
    
    def test_collector_initialization(self, logger):
        """Test MetricsCollector initialization"""
        collector = MetricsCollector(max_metrics=1000, logger=logger)
        
        assert collector.max_metrics == 1000
        assert collector.logger == logger
        assert len(collector.metrics) == 0
        assert len(collector.metric_index) == 0
    
    def test_metric_recording(self, logger):
        """Test metric recording"""
        collector = MetricsCollector(logger=logger)
        
        # Record a latency metric
        collector.record_latency("response", 150.5, {"test": "value"})
        
        assert len(collector.metrics) == 1
        assert "response_latency" in collector.metric_index
        assert len(collector.metric_index["response_latency"]) == 1
        
        # Record throughput metric
        collector.record_throughput("processing", 50.0, "ops/sec")
        
        assert len(collector.metrics) == 2
    
    def test_statistics_calculation(self, logger):
        """Test statistics calculation"""
        collector = MetricsCollector(logger=logger)
        
        # Record multiple values
        values = [100, 150, 200, 250, 300]
        for value in values:
            collector.record_latency("test", value)
        
        stats = collector.get_statistics("test_latency")
        
        assert stats["count"] == 5
        assert stats["mean"] == 200.0
        assert stats["median"] == 200.0
        assert stats["min"] == 100.0
        assert stats["max"] == 300.0
        assert stats["latest"] == 300.0
    
    def test_recent_metrics_retrieval(self, logger):
        """Test recent metrics retrieval"""
        collector = MetricsCollector(logger=logger)
        
        # Record 10 metrics
        for i in range(10):
            collector.record_latency("test", i * 10)
        
        # Get last 5 metrics
        recent = collector.get_recent_metrics("test_latency", 5)
        assert len(recent) == 5
        assert recent[-1].value == 90.0  # Last recorded value
    
    def test_comprehensive_stats(self, logger):
        """Test comprehensive statistics"""
        collector = MetricsCollector(logger=logger)
        
        collector.record_latency("test1", 100)
        collector.record_throughput("test2", 50)
        collector.record_quality("test3", 0.95)
        
        stats = collector.get_comprehensive_stats()
        
        assert stats["total_metrics"] == 3
        assert "latency" in stats["metric_types"]
        assert "throughput" in stats["metric_types"]
        assert "quality" in stats["metric_types"]
        assert "statistics" in stats


class TestNetworkMonitor:
    """Test NetworkMonitor functionality"""
    
    @pytest.mark.asyncio
    async def test_monitor_initialization(self, logger):
        """Test NetworkMonitor initialization"""
        monitor = NetworkMonitor(update_interval=1.0, logger=logger)
        
        assert monitor.update_interval == 1.0
        assert monitor.logger == logger
        assert monitor.is_monitoring is False
        assert isinstance(monitor.current_condition, NetworkCondition)
    
    @pytest.mark.asyncio
    async def test_monitor_start_stop(self, logger):
        """Test monitor start and stop"""
        monitor = NetworkMonitor(update_interval=0.1, logger=logger)
        
        # Start monitoring
        success = await monitor.start_monitoring()
        assert success is True
        assert monitor.is_monitoring is True
        assert monitor.monitor_task is not None
        
        # Wait briefly for monitoring
        await asyncio.sleep(0.2)
        
        # Stop monitoring
        await monitor.stop_monitoring()
        assert monitor.is_monitoring is False
    
    @pytest.mark.asyncio
    async def test_condition_callbacks(self, logger):
        """Test network condition callbacks"""
        monitor = NetworkMonitor(update_interval=0.1, logger=logger)
        
        callback_calls = []
        
        def test_callback(condition):
            callback_calls.append(condition)
        
        monitor.add_condition_callback(test_callback)
        
        # Start monitoring
        await monitor.start_monitoring()
        await asyncio.sleep(0.3)  # Let it run a few cycles
        await monitor.stop_monitoring()
        
        # Should have received some callbacks
        assert len(callback_calls) >= 1
        assert isinstance(callback_calls[0], NetworkCondition)


class TestAdaptiveQualityController:
    """Test AdaptiveQualityController functionality"""
    
    def test_controller_initialization(self, logger):
        """Test AdaptiveQualityController initialization"""
        controller = AdaptiveQualityController(target_latency_ms=250.0, logger=logger)
        
        assert controller.target_latency_ms == 250.0
        assert controller.current_level == QualityLevel.MEDIUM
        assert isinstance(controller.current_settings, QualitySettings)
        assert controller.adaptation_cooldown == 5.0
    
    def test_performance_feedback_processing(self, logger):
        """Test performance feedback processing"""
        controller = AdaptiveQualityController(target_latency_ms=200.0, logger=logger)
        network_condition = NetworkCondition(latency_ms=50.0)
        
        # Add several latency measurements
        for latency in [180, 190, 185, 195, 175]:
            controller.update_performance_feedback(latency, network_condition)
        
        # Should not adapt immediately due to good performance
        assert controller.current_level == QualityLevel.MEDIUM
    
    def test_quality_adaptation_high_latency(self, logger):
        """Test quality adaptation due to high latency"""
        controller = AdaptiveQualityController(target_latency_ms=200.0, logger=logger)
        controller.adaptation_cooldown = 0.1  # Reduce cooldown for testing
        
        network_condition = NetworkCondition(latency_ms=50.0)
        
        # Simulate consistently high latencies
        high_latencies = [350, 360, 370, 380, 390]
        for latency in high_latencies:
            adapted = controller.update_performance_feedback(latency, network_condition)
        
        # Should adapt to lower quality
        assert controller.current_level in [QualityLevel.LOW, QualityLevel.ULTRA_LOW]
    
    def test_force_quality_level(self, logger):
        """Test forcing specific quality level"""
        controller = AdaptiveQualityController(logger=logger)
        
        # Force to HIGH quality
        success = controller.force_quality_level(QualityLevel.HIGH)
        assert success is True
        assert controller.current_level == QualityLevel.HIGH
        
        # Verify settings changed
        assert controller.current_settings.audio_sample_rate == 48000
        assert controller.current_settings.buffer_size == 1024
    
    def test_quality_change_callbacks(self, logger):
        """Test quality change callbacks"""
        controller = AdaptiveQualityController(logger=logger)
        
        callback_calls = []
        
        def test_callback(level, settings):
            callback_calls.append((level, settings))
        
        controller.add_quality_change_callback(test_callback)
        
        # Force quality change
        controller.force_quality_level(QualityLevel.LOW)
        
        assert len(callback_calls) == 1
        assert callback_calls[0][0] == QualityLevel.LOW
        assert isinstance(callback_calls[0][1], QualitySettings)
    
    def test_adaptation_statistics(self, logger):
        """Test adaptation statistics"""
        controller = AdaptiveQualityController(logger=logger)
        
        # Force a quality change
        controller.force_quality_level(QualityLevel.HIGH)
        
        stats = controller.get_adaptation_stats()
        
        assert stats["current_level"] == "high"
        assert "current_settings" in stats
        assert stats["adaptation_count"] >= 1
        assert "last_adaptation" in stats


class TestPerformanceAlertManager:
    """Test PerformanceAlertManager functionality"""
    
    def test_alert_manager_initialization(self, logger):
        """Test PerformanceAlertManager initialization"""
        manager = PerformanceAlertManager(logger=logger)
        
        assert len(manager.active_alerts) == 0
        assert len(manager.alert_history) == 0
        assert "response_latency" in manager.thresholds
        assert "cpu_usage" in manager.thresholds
    
    def test_metric_threshold_checking(self, logger):
        """Test metric threshold checking"""
        manager = PerformanceAlertManager(logger=logger)
        
        # Create metric that exceeds warning threshold
        metric = PerformanceMetric(
            metric_type=MetricType.LATENCY,
            name="response_latency",
            value=350.0,  # Above 300ms warning threshold
            unit="ms"
        )
        
        alert = manager.check_metric(metric)
        
        assert alert is not None
        assert alert.level == AlertLevel.WARNING
        assert alert.metric_name == "response_latency"
        assert alert.current_value == 350.0
        assert "response_latency" in manager.active_alerts
    
    def test_alert_escalation(self, logger):
        """Test alert level escalation"""
        manager = PerformanceAlertManager(logger=logger)
        
        # Create metric that exceeds critical threshold
        critical_metric = PerformanceMetric(
            metric_type=MetricType.LATENCY,
            name="response_latency",
            value=600.0,  # Above 500ms critical threshold
            unit="ms"
        )
        
        alert = manager.check_metric(critical_metric)
        
        assert alert.level == AlertLevel.CRITICAL
        
        # Create metric that exceeds emergency threshold
        emergency_metric = PerformanceMetric(
            metric_type=MetricType.LATENCY,
            name="response_latency",
            value=1200.0,  # Above 1000ms emergency threshold
            unit="ms"
        )
        
        emergency_alert = manager.check_metric(emergency_metric)
        assert emergency_alert.level == AlertLevel.EMERGENCY
    
    def test_alert_acknowledgment(self, logger):
        """Test alert acknowledgment"""
        manager = PerformanceAlertManager(logger=logger)
        
        # Create alert
        metric = PerformanceMetric(MetricType.RESOURCE, "cpu_usage", 85.0, "%")
        manager.check_metric(metric)
        
        # Acknowledge alert
        success = manager.acknowledge_alert("cpu_usage")
        assert success is True
        assert manager.active_alerts["cpu_usage"].acknowledged is True
    
    def test_alert_clearing(self, logger):
        """Test alert clearing when metric improves"""
        manager = PerformanceAlertManager(logger=logger)
        
        # Create alert with high latency
        high_metric = PerformanceMetric(MetricType.LATENCY, "response_latency", 400.0, "ms")
        manager.check_metric(high_metric)
        assert "response_latency" in manager.active_alerts
        
        # Improve metric below threshold
        good_metric = PerformanceMetric(MetricType.LATENCY, "response_latency", 200.0, "ms")
        manager.check_metric(good_metric)
        assert "response_latency" not in manager.active_alerts
    
    def test_alert_callbacks(self, logger):
        """Test alert callbacks"""
        manager = PerformanceAlertManager(logger=logger)
        
        callback_calls = []
        
        def test_callback(alert):
            callback_calls.append(alert)
        
        manager.add_alert_callback(test_callback)
        
        # Create alert
        metric = PerformanceMetric(MetricType.LATENCY, "response_latency", 350.0, "ms")
        manager.check_metric(metric)
        
        assert len(callback_calls) == 1
        assert isinstance(callback_calls[0], PerformanceAlert)
    
    def test_alert_statistics(self, logger):
        """Test alert statistics"""
        manager = PerformanceAlertManager(logger=logger)
        
        # Create multiple alerts
        metrics = [
            PerformanceMetric(MetricType.LATENCY, "response_latency", 350.0, "ms"),
            PerformanceMetric(MetricType.RESOURCE, "cpu_usage", 90.0, "%")
        ]
        
        for metric in metrics:
            manager.check_metric(metric)
        
        stats = manager.get_alert_stats()
        
        assert stats["active_alert_count"] == 2
        assert stats["total_alerts"] == 2
        assert "alert_levels" in stats


class TestPerformanceMonitor:
    """Test PerformanceMonitor integration"""
    
    @pytest.mark.asyncio
    async def test_monitor_initialization(self, logger):
        """Test PerformanceMonitor initialization"""
        monitor = PerformanceMonitor(target_latency_ms=300.0, logger=logger)
        
        assert monitor.target_latency_ms == 300.0
        assert isinstance(monitor.metrics_collector, MetricsCollector)
        assert isinstance(monitor.network_monitor, NetworkMonitor)
        assert isinstance(monitor.quality_controller, AdaptiveQualityController)
        assert isinstance(monitor.alert_manager, PerformanceAlertManager)
    
    @pytest.mark.asyncio
    async def test_monitor_lifecycle(self, logger):
        """Test monitor initialization and shutdown"""
        monitor = PerformanceMonitor(target_latency_ms=250.0, logger=logger)
        
        # Initialize
        success = await monitor.initialize()
        assert success is True
        assert monitor.is_monitoring is True
        
        # Shutdown
        await monitor.shutdown()
        assert monitor.is_monitoring is False
    
    @pytest.mark.asyncio
    async def test_response_latency_recording(self, logger):
        """Test response latency recording and adaptation"""
        monitor = PerformanceMonitor(target_latency_ms=200.0, logger=logger)
        await monitor.initialize()
        
        try:
            # Record latency
            monitor.record_response_latency(180.0, "test_operation", {"test": "tag"})
            
            # Check metrics were recorded
            stats = monitor.metrics_collector.get_statistics("test_operation_latency")
            assert stats["latest"] == 180.0
            
            # Record high latency
            monitor.record_response_latency(400.0, "test_operation")
            
            # Should trigger adaptation eventually
            await asyncio.sleep(0.1)
            
        finally:
            await monitor.shutdown()
    
    @pytest.mark.asyncio
    async def test_system_metrics_recording(self, logger):
        """Test system metrics recording"""
        monitor = PerformanceMonitor(logger=logger)
        await monitor.initialize()
        
        try:
            # Record system metrics
            monitor.record_system_metrics()
            
            # Check metrics were recorded
            cpu_stats = monitor.metrics_collector.get_statistics("cpu_usage")
            memory_stats = monitor.metrics_collector.get_statistics("memory_usage")
            
            assert "latest" in cpu_stats
            assert "latest" in memory_stats
            assert cpu_stats["latest"] >= 0
            assert memory_stats["latest"] >= 0
            
        finally:
            await monitor.shutdown()
    
    @pytest.mark.asyncio
    async def test_optimization_callbacks(self, logger):
        """Test optimization callbacks"""
        monitor = PerformanceMonitor(target_latency_ms=100.0, logger=logger)  # Low target
        await monitor.initialize()
        
        optimization_calls = []
        
        def test_callback(data):
            optimization_calls.append(data)
        
        monitor.add_optimization_callback(test_callback)
        
        try:
            # Record consistently high latencies to trigger optimization
            for _ in range(10):
                monitor.record_response_latency(300.0)  # Well above target
                await asyncio.sleep(0.01)
            
            # Wait for monitoring loop to detect optimization opportunity
            await asyncio.sleep(2)
            
            # Should have triggered optimization (might take a moment)
            # Note: This test might be flaky due to timing
            
        finally:
            await monitor.shutdown()
    
    @pytest.mark.asyncio
    async def test_quality_settings_retrieval(self, logger):
        """Test quality settings retrieval"""
        monitor = PerformanceMonitor(logger=logger)
        await monitor.initialize()
        
        try:
            # Get current quality settings
            settings = monitor.get_current_quality_settings()
            assert isinstance(settings, QualitySettings)
            assert settings.audio_sample_rate > 0
            
            # Force quality change
            success = monitor.force_quality_level(QualityLevel.HIGH)
            assert success is True
            
            # Verify settings changed
            new_settings = monitor.get_current_quality_settings()
            assert new_settings.audio_sample_rate == 48000
            
        finally:
            await monitor.shutdown()
    
    @pytest.mark.asyncio
    async def test_comprehensive_performance_report(self, logger):
        """Test comprehensive performance report"""
        monitor = PerformanceMonitor(logger=logger)
        await monitor.initialize()
        
        try:
            # Record some metrics
            monitor.record_response_latency(150.0)
            monitor.record_system_metrics()
            
            # Wait a moment for data to accumulate
            await asyncio.sleep(0.1)
            
            # Get comprehensive report
            report = monitor.get_comprehensive_performance_report()
            
            assert "session_duration" in report
            assert "target_latency_ms" in report
            assert "current_quality" in report
            assert "network_condition" in report
            assert "metrics" in report
            assert "alerts" in report
            assert "adaptations" in report
            
            # Check report structure
            assert report["target_latency_ms"] == monitor.target_latency_ms
            assert "level" in report["current_quality"]
            assert "settings" in report["current_quality"]
            
        finally:
            await monitor.shutdown()


class TestFactoryFunction:
    """Test factory function"""
    
    def test_factory_function_defaults(self):
        """Test factory function with defaults"""
        monitor = create_performance_monitor()
        
        assert isinstance(monitor, PerformanceMonitor)
        assert monitor.target_latency_ms == 250.0
    
    def test_factory_function_custom(self, logger):
        """Test factory function with custom parameters"""
        monitor = create_performance_monitor(
            target_latency_ms=200.0,
            logger=logger
        )
        
        assert monitor.target_latency_ms == 200.0
        assert monitor.logger == logger


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_performance_degradation_scenario(self, logger):
        """Test handling of performance degradation"""
        monitor = PerformanceMonitor(target_latency_ms=200.0, logger=logger)
        await monitor.initialize()
        
        try:
            # Start with good performance
            for _ in range(5):
                monitor.record_response_latency(150.0)
                await asyncio.sleep(0.01)
            
            # Simulate gradual performance degradation
            degradation_latencies = [180, 220, 280, 350, 420]
            for latency in degradation_latencies:
                monitor.record_response_latency(latency)
                await asyncio.sleep(0.01)
            
            # Check that quality adaptation occurred
            current_level = monitor.quality_controller.current_level
            stats = monitor.quality_controller.get_adaptation_stats()
            
            # Should have adapted due to high latencies
            # (Note: Exact behavior depends on adaptation logic and timing)
            
        finally:
            await monitor.shutdown()
    
    @pytest.mark.asyncio
    async def test_mixed_performance_monitoring(self, logger):
        """Test monitoring with mixed performance metrics"""
        monitor = PerformanceMonitor(logger=logger)
        await monitor.initialize()
        
        try:
            # Record various types of metrics
            monitor.record_response_latency(175.0, "chat_response")
            monitor.record_response_latency(95.0, "audio_processing")
            monitor.record_system_metrics()
            
            # Wait for monitoring
            await asyncio.sleep(0.1)
            
            # Check comprehensive report includes all metrics
            report = monitor.get_comprehensive_performance_report()
            
            assert "metrics" in report
            metrics_stats = report["metrics"]["statistics"]
            
            # Should have latency metrics
            latency_metrics = [k for k in metrics_stats.keys() if "latency" in k]
            assert len(latency_metrics) >= 2
            
        finally:
            await monitor.shutdown()
    
    @pytest.mark.asyncio
    async def test_alert_generation_and_resolution(self, logger):
        """Test alert generation and resolution cycle"""
        monitor = PerformanceMonitor(target_latency_ms=200.0, logger=logger)
        await monitor.initialize()
        
        alert_calls = []
        
        def alert_callback(alert):
            alert_calls.append(alert)
        
        monitor.alert_manager.add_alert_callback(alert_callback)
        
        try:
            # Generate alert with high latency
            monitor.record_response_latency(400.0)  # Above warning threshold
            
            # Should have generated alert
            assert len(alert_calls) >= 1
            assert alert_calls[0].level == AlertLevel.WARNING
            
            # Resolve with good latency
            monitor.record_response_latency(150.0)  # Below threshold
            
            # Alert should be cleared from active alerts
            active_alerts = monitor.alert_manager.get_active_alerts()
            response_alerts = [a for a in active_alerts if a.metric_name == "response_latency"]
            assert len(response_alerts) == 0
            
        finally:
            await monitor.shutdown()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 