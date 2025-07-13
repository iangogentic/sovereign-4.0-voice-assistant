"""
Performance Monitor for Sovereign 4.0
Comprehensive real-time performance monitoring and adaptive quality control
Ensures consistent sub-300ms response times with intelligent adaptation
"""

import asyncio
import time
import logging
import threading
import statistics
import queue
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import json
import math
import psutil
import weakref


class QualityLevel(Enum):
    """Quality levels for adaptive control"""
    ULTRA_LOW = "ultra_low"      # Minimum quality for maximum speed
    LOW = "low"                  # Low quality for good speed
    MEDIUM = "medium"            # Balanced quality/speed
    HIGH = "high"               # High quality, acceptable latency
    ULTRA_HIGH = "ultra_high"   # Maximum quality, higher latency


class MetricType(Enum):
    """Types of performance metrics"""
    LATENCY = "latency"          # Response time measurements
    THROUGHPUT = "throughput"    # Processing rate measurements
    QUALITY = "quality"          # Audio/output quality metrics
    RESOURCE = "resource"        # System resource utilization
    NETWORK = "network"          # Network condition metrics
    ERROR = "error"              # Error rate and type tracking


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"                # Informational alerts
    WARNING = "warning"          # Performance degradation
    CRITICAL = "critical"        # Severe performance issues
    EMERGENCY = "emergency"      # System failure imminent


@dataclass
class PerformanceMetric:
    """Individual performance metric measurement"""
    metric_type: MetricType
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    unit: str = "ms"
    tags: Dict[str, str] = field(default_factory=dict)
    
    def get_age_ms(self) -> float:
        """Get metric age in milliseconds"""
        return (time.time() - self.timestamp) * 1000


@dataclass
class PerformanceAlert:
    """Performance alert information"""
    level: AlertLevel
    message: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            "level": self.level.value,
            "message": self.message,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
            "acknowledged": self.acknowledged
        }


@dataclass
class QualitySettings:
    """Quality configuration for different levels"""
    audio_sample_rate: int = 24000
    audio_channels: int = 1
    audio_bit_depth: int = 16
    model_temperature: float = 0.7
    buffer_size: int = 512
    compression_level: Optional[str] = None
    enable_noise_reduction: bool = True
    response_max_tokens: int = 4096
    
    @classmethod
    def for_quality_level(cls, level: QualityLevel) -> 'QualitySettings':
        """Create quality settings for specific level"""
        if level == QualityLevel.ULTRA_LOW:
            return cls(
                audio_sample_rate=16000,
                audio_bit_depth=8,
                buffer_size=256,
                model_temperature=0.3,
                compression_level="high",
                enable_noise_reduction=False,
                response_max_tokens=2048
            )
        elif level == QualityLevel.LOW:
            return cls(
                audio_sample_rate=22050,
                audio_bit_depth=16,
                buffer_size=512,
                model_temperature=0.5,
                compression_level="medium",
                enable_noise_reduction=True,
                response_max_tokens=3072
            )
        elif level == QualityLevel.MEDIUM:
            return cls()  # Default settings
        elif level == QualityLevel.HIGH:
            return cls(
                audio_sample_rate=48000,
                buffer_size=1024,
                model_temperature=0.8,
                compression_level=None,
                response_max_tokens=6144
            )
        else:  # ULTRA_HIGH
            return cls(
                audio_sample_rate=48000,
                buffer_size=2048,
                model_temperature=0.9,
                compression_level=None,
                response_max_tokens=8192
            )


@dataclass
class NetworkCondition:
    """Network condition assessment"""
    latency_ms: float = 0.0
    jitter_ms: float = 0.0
    packet_loss_rate: float = 0.0
    bandwidth_mbps: float = 0.0
    stability_score: float = 1.0  # 0.0 to 1.0
    
    def get_quality_recommendation(self) -> QualityLevel:
        """Recommend quality level based on network conditions"""
        if self.latency_ms > 200 or self.packet_loss_rate > 0.05:
            return QualityLevel.ULTRA_LOW
        elif self.latency_ms > 100 or self.packet_loss_rate > 0.02:
            return QualityLevel.LOW
        elif self.latency_ms > 50 or self.packet_loss_rate > 0.01:
            return QualityLevel.MEDIUM
        elif self.latency_ms > 20:
            return QualityLevel.HIGH
        else:
            return QualityLevel.ULTRA_HIGH


class MetricsCollector:
    """Collects and manages performance metrics"""
    
    def __init__(self, max_metrics: int = 10000, logger: Optional[logging.Logger] = None):
        self.max_metrics = max_metrics
        self.logger = logger or logging.getLogger(__name__)
        
        # Metric storage
        self.metrics: deque = deque(maxlen=max_metrics)
        self.metric_index: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Statistics tracking
        self.stats: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Threading
        self.lock = threading.RLock()
        
        # Timing precision
        self.start_time = time.perf_counter()
        
    def record_metric(self, metric_type: MetricType, name: str, value: float, unit: str = "ms", tags: Optional[Dict[str, str]] = None) -> None:
        """Record a performance metric"""
        try:
            metric = PerformanceMetric(
                metric_type=metric_type,
                name=name,
                value=value,
                unit=unit,
                tags=tags or {}
            )
            
            with self.lock:
                self.metrics.append(metric)
                self.metric_index[name].append(metric)
                self._update_statistics(name, value)
                
        except Exception as e:
            self.logger.error(f"❌ Error recording metric {name}: {e}")
    
    def record_latency(self, operation: str, latency_ms: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record latency metric"""
        self.record_metric(MetricType.LATENCY, f"{operation}_latency", latency_ms, "ms", tags)
    
    def record_throughput(self, operation: str, rate: float, unit: str = "ops/sec", tags: Optional[Dict[str, str]] = None) -> None:
        """Record throughput metric"""
        self.record_metric(MetricType.THROUGHPUT, f"{operation}_throughput", rate, unit, tags)
    
    def record_quality(self, metric_name: str, score: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record quality metric (0.0 to 1.0)"""
        self.record_metric(MetricType.QUALITY, metric_name, score, "score", tags)
    
    def get_recent_metrics(self, name: str, count: int = 100) -> List[PerformanceMetric]:
        """Get recent metrics for a specific name"""
        with self.lock:
            metrics = list(self.metric_index[name])
            return metrics[-count:] if count < len(metrics) else metrics
    
    def get_statistics(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        with self.lock:
            return self.stats.get(name, {}).copy()
    
    def _update_statistics(self, name: str, value: float) -> None:
        """Update rolling statistics for a metric"""
        metrics = list(self.metric_index[name])
        if not metrics:
            return
        
        values = [m.value for m in metrics[-100:]]  # Last 100 values
        
        self.stats[name] = {
            "count": len(values),
            "latest": value,
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p95": self._percentile(values, 0.95),
            "p99": self._percentile(values, 0.99)
        }
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        with self.lock:
            return {
                "total_metrics": len(self.metrics),
                "metric_types": list(set(m.metric_type.value for m in self.metrics)),
                "statistics": dict(self.stats),
                "collection_duration_seconds": time.perf_counter() - self.start_time
            }


class NetworkMonitor:
    """Monitors network conditions for adaptive quality control"""
    
    def __init__(self, update_interval: float = 5.0, logger: Optional[logging.Logger] = None):
        self.update_interval = update_interval
        self.logger = logger or logging.getLogger(__name__)
        
        # Network state
        self.current_condition = NetworkCondition()
        self.condition_history: deque = deque(maxlen=100)
        
        # Monitoring
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.condition_callbacks: List[Callable[[NetworkCondition], None]] = []
        
    async def start_monitoring(self) -> bool:
        """Start network monitoring"""
        try:
            self.is_monitoring = True
            self.monitor_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info("🌐 Network monitoring started")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to start network monitoring: {e}")
            return False
    
    async def stop_monitoring(self):
        """Stop network monitoring"""
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("🌐 Network monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                await self._assess_network_condition()
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Network monitoring error: {e}")
                await asyncio.sleep(1)
    
    async def _assess_network_condition(self):
        """Assess current network conditions"""
        try:
            # Simulate network assessment (in real implementation, this would use actual network probes)
            import random
            
            # Simulate network metrics with some variability
            base_latency = 30 + random.uniform(-10, 20)
            jitter = random.uniform(0, 5)
            packet_loss = random.uniform(0, 0.01)
            bandwidth = 100 + random.uniform(-20, 50)
            
            # Calculate stability score based on recent history
            stability = self._calculate_stability_score()
            
            condition = NetworkCondition(
                latency_ms=base_latency,
                jitter_ms=jitter,
                packet_loss_rate=packet_loss,
                bandwidth_mbps=bandwidth,
                stability_score=stability
            )
            
            # Update state
            self.current_condition = condition
            self.condition_history.append(condition)
            
            # Notify callbacks
            for callback in self.condition_callbacks:
                try:
                    callback(condition)
                except Exception as e:
                    self.logger.error(f"❌ Network condition callback error: {e}")
                    
        except Exception as e:
            self.logger.error(f"❌ Error assessing network condition: {e}")
    
    def _calculate_stability_score(self) -> float:
        """Calculate network stability score based on recent history"""
        if len(self.condition_history) < 3:
            return 1.0
        
        recent_latencies = [c.latency_ms for c in list(self.condition_history)[-10:]]
        if not recent_latencies:
            return 1.0
        
        # Lower variance = higher stability
        variance = statistics.variance(recent_latencies) if len(recent_latencies) > 1 else 0
        stability = max(0.0, 1.0 - (variance / 1000))  # Normalize variance
        return min(1.0, stability)
    
    def add_condition_callback(self, callback: Callable[[NetworkCondition], None]):
        """Add callback for network condition changes"""
        self.condition_callbacks.append(callback)
    
    def get_current_condition(self) -> NetworkCondition:
        """Get current network condition"""
        return self.current_condition


class AdaptiveQualityController:
    """Controls quality settings based on performance feedback"""
    
    def __init__(self, target_latency_ms: float = 250.0, logger: Optional[logging.Logger] = None):
        self.target_latency_ms = target_latency_ms
        self.logger = logger or logging.getLogger(__name__)
        
        # Quality state
        self.current_level = QualityLevel.MEDIUM
        self.current_settings = QualitySettings.for_quality_level(self.current_level)
        
        # Adaptation tracking
        self.adaptation_history: deque = deque(maxlen=100)
        self.last_adaptation = time.time()
        self.adaptation_cooldown = 5.0  # seconds
        
        # Performance tracking
        self.recent_latencies: deque = deque(maxlen=20)
        
        # Callbacks
        self.quality_change_callbacks: List[Callable[[QualityLevel, QualitySettings], None]] = []
        
    def update_performance_feedback(self, latency_ms: float, network_condition: NetworkCondition) -> bool:
        """Update with performance feedback and adapt if needed"""
        try:
            self.recent_latencies.append(latency_ms)
            
            # Check if adaptation is needed
            if self._should_adapt(latency_ms, network_condition):
                new_level = self._determine_optimal_level(latency_ms, network_condition)
                if new_level != self.current_level:
                    return self._adapt_quality(new_level)
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error updating performance feedback: {e}")
            return False
    
    def _should_adapt(self, latency_ms: float, network_condition: NetworkCondition) -> bool:
        """Determine if quality adaptation is needed"""
        # Respect cooldown period
        if time.time() - self.last_adaptation < self.adaptation_cooldown:
            return False
        
        # Check if latency is consistently above/below target
        if len(self.recent_latencies) < 5:
            return False
        
        recent_avg = statistics.mean(list(self.recent_latencies)[-5:])
        
        # Significant deviation from target
        if abs(recent_avg - self.target_latency_ms) > 50:
            return True
        
        # Network conditions changed significantly
        recommended_level = network_condition.get_quality_recommendation()
        if abs(recommended_level.value.split('_')[-1] == 'high') != abs(self.current_level.value.split('_')[-1] == 'high'):
            return True
        
        return False
    
    def _determine_optimal_level(self, latency_ms: float, network_condition: NetworkCondition) -> QualityLevel:
        """Determine optimal quality level"""
        # Start with network-based recommendation
        network_recommendation = network_condition.get_quality_recommendation()
        
        # Adjust based on recent performance
        recent_avg = statistics.mean(list(self.recent_latencies)[-5:]) if self.recent_latencies else latency_ms
        
        if recent_avg > self.target_latency_ms + 100:  # Significantly over target
            # Need to reduce quality
            if self.current_level == QualityLevel.ULTRA_HIGH:
                return QualityLevel.HIGH
            elif self.current_level == QualityLevel.HIGH:
                return QualityLevel.MEDIUM
            elif self.current_level == QualityLevel.MEDIUM:
                return QualityLevel.LOW
            else:
                return QualityLevel.ULTRA_LOW
        elif recent_avg < self.target_latency_ms - 50:  # Well under target
            # Can increase quality
            if self.current_level == QualityLevel.ULTRA_LOW:
                return QualityLevel.LOW
            elif self.current_level == QualityLevel.LOW:
                return QualityLevel.MEDIUM
            elif self.current_level == QualityLevel.MEDIUM:
                return QualityLevel.HIGH
            else:
                return QualityLevel.ULTRA_HIGH
        
        # Use network recommendation if performance is acceptable
        return network_recommendation
    
    def _adapt_quality(self, new_level: QualityLevel) -> bool:
        """Adapt to new quality level"""
        try:
            old_level = self.current_level
            old_settings = self.current_settings
            
            self.current_level = new_level
            self.current_settings = QualitySettings.for_quality_level(new_level)
            self.last_adaptation = time.time()
            
            # Record adaptation
            self.adaptation_history.append({
                "timestamp": self.last_adaptation,
                "from_level": old_level.value,
                "to_level": new_level.value,
                "recent_latency": statistics.mean(list(self.recent_latencies)[-5:]) if self.recent_latencies else 0
            })
            
            self.logger.info(f"🎛️ Quality adapted: {old_level.value} → {new_level.value}")
            
            # Notify callbacks
            for callback in self.quality_change_callbacks:
                try:
                    callback(new_level, self.current_settings)
                except Exception as e:
                    self.logger.error(f"❌ Quality change callback error: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error adapting quality: {e}")
            return False
    
    def force_quality_level(self, level: QualityLevel) -> bool:
        """Force specific quality level"""
        return self._adapt_quality(level)
    
    def add_quality_change_callback(self, callback: Callable[[QualityLevel, QualitySettings], None]):
        """Add callback for quality changes"""
        self.quality_change_callbacks.append(callback)
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get adaptation statistics"""
        return {
            "current_level": self.current_level.value,
            "current_settings": self.current_settings.__dict__,
            "adaptation_count": len(self.adaptation_history),
            "last_adaptation": self.last_adaptation,
            "recent_latencies": list(self.recent_latencies),
            "average_recent_latency": statistics.mean(list(self.recent_latencies)) if self.recent_latencies else 0
        }


class PerformanceAlertManager:
    """Manages performance alerts and notifications"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Alert storage
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # Thresholds
        self.thresholds = {
            "response_latency": {"warning": 300, "critical": 500, "emergency": 1000},
            "connection_latency": {"warning": 100, "critical": 200, "emergency": 500},
            "audio_processing_latency": {"warning": 50, "critical": 100, "emergency": 200},
            "error_rate": {"warning": 0.01, "critical": 0.05, "emergency": 0.1},
            "cpu_usage": {"warning": 70, "critical": 85, "emergency": 95},
            "memory_usage": {"warning": 80, "critical": 90, "emergency": 95}
        }
        
        # Callbacks
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        
    def check_metric(self, metric: PerformanceMetric) -> Optional[PerformanceAlert]:
        """Check metric against thresholds and generate alerts"""
        try:
            thresholds = self.thresholds.get(metric.name)
            if not thresholds:
                return None
            
            alert_level = None
            threshold_value = None
            
            # Determine alert level
            if metric.value >= thresholds.get("emergency", float('inf')):
                alert_level = AlertLevel.EMERGENCY
                threshold_value = thresholds["emergency"]
            elif metric.value >= thresholds.get("critical", float('inf')):
                alert_level = AlertLevel.CRITICAL
                threshold_value = thresholds["critical"]
            elif metric.value >= thresholds.get("warning", float('inf')):
                alert_level = AlertLevel.WARNING
                threshold_value = thresholds["warning"]
            
            if alert_level:
                return self._create_alert(alert_level, metric, threshold_value)
            
            # Clear existing alert if metric is now within thresholds
            if metric.name in self.active_alerts:
                self._clear_alert(metric.name)
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error checking metric {metric.name}: {e}")
            return None
    
    def _create_alert(self, level: AlertLevel, metric: PerformanceMetric, threshold: float) -> PerformanceAlert:
        """Create performance alert"""
        alert = PerformanceAlert(
            level=level,
            message=f"{metric.name} is {metric.value:.1f}{metric.unit} (threshold: {threshold:.1f}{metric.unit})",
            metric_name=metric.name,
            current_value=metric.value,
            threshold=threshold
        )
        
        # Store active alert
        self.active_alerts[metric.name] = alert
        self.alert_history.append(alert)
        
        self.logger.warning(f"🚨 {level.value.upper()} Alert: {alert.message}")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"❌ Alert callback error: {e}")
        
        return alert
    
    def _clear_alert(self, metric_name: str):
        """Clear active alert"""
        if metric_name in self.active_alerts:
            alert = self.active_alerts.pop(metric_name)
            self.logger.info(f"✅ Alert cleared: {metric_name}")
    
    def acknowledge_alert(self, metric_name: str) -> bool:
        """Acknowledge active alert"""
        if metric_name in self.active_alerts:
            self.active_alerts[metric_name].acknowledged = True
            self.logger.info(f"👍 Alert acknowledged: {metric_name}")
            return True
        return False
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add callback for alerts"""
        self.alert_callbacks.append(callback)
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics"""
        return {
            "active_alert_count": len(self.active_alerts),
            "total_alerts": len(self.alert_history),
            "alert_levels": {
                level.value: sum(1 for a in self.alert_history if a.level == level)
                for level in AlertLevel
            }
        }


class PerformanceMonitor:
    """Central performance monitoring system"""
    
    def __init__(self, target_latency_ms: float = 250.0, logger: Optional[logging.Logger] = None):
        self.target_latency_ms = target_latency_ms
        self.logger = logger or logging.getLogger(__name__)
        
        # Core components
        self.metrics_collector = MetricsCollector(logger=logger)
        self.network_monitor = NetworkMonitor(logger=logger)
        self.quality_controller = AdaptiveQualityController(target_latency_ms, logger)
        self.alert_manager = PerformanceAlertManager(logger)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.session_start = time.time()
        self.last_optimization = time.time()
        
        # Integration callbacks
        self.optimization_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
    async def initialize(self) -> bool:
        """Initialize performance monitoring"""
        try:
            # Start network monitoring
            if not await self.network_monitor.start_monitoring():
                return False
            
            # Set up callbacks
            self.network_monitor.add_condition_callback(self._on_network_condition_change)
            self.quality_controller.add_quality_change_callback(self._on_quality_change)
            self.alert_manager.add_alert_callback(self._on_alert)
            
            # Start monitoring loop
            self.is_monitoring = True
            self.monitor_task = asyncio.create_task(self._monitoring_loop())
            
            self.logger.info("✅ Performance monitoring initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize performance monitoring: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown performance monitoring"""
        try:
            self.is_monitoring = False
            
            # Stop network monitoring
            await self.network_monitor.stop_monitoring()
            
            # Cancel monitoring task
            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info("✅ Performance monitoring shutdown completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error during performance monitoring shutdown: {e}")
    
    def record_response_latency(self, latency_ms: float, operation: str = "response", tags: Optional[Dict[str, str]] = None) -> None:
        """Record response latency and trigger adaptation"""
        try:
            # Record metric
            self.metrics_collector.record_latency(operation, latency_ms, tags)
            
            # Check for alerts
            metric = PerformanceMetric(MetricType.LATENCY, f"{operation}_latency", latency_ms)
            alert = self.alert_manager.check_metric(metric)
            
            # Update quality controller
            network_condition = self.network_monitor.get_current_condition()
            adapted = self.quality_controller.update_performance_feedback(latency_ms, network_condition)
            
            if adapted:
                self.logger.info(f"🎛️ Quality adapted based on {latency_ms:.1f}ms latency")
                
        except Exception as e:
            self.logger.error(f"❌ Error recording response latency: {e}")
    
    def record_system_metrics(self) -> None:
        """Record current system resource metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.metrics_collector.record_metric(MetricType.RESOURCE, "cpu_usage", cpu_percent, "%")
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.metrics_collector.record_metric(MetricType.RESOURCE, "memory_usage", memory_percent, "%")
            
            # Check for alerts
            cpu_metric = PerformanceMetric(MetricType.RESOURCE, "cpu_usage", cpu_percent)
            memory_metric = PerformanceMetric(MetricType.RESOURCE, "memory_usage", memory_percent)
            
            self.alert_manager.check_metric(cpu_metric)
            self.alert_manager.check_metric(memory_metric)
            
        except Exception as e:
            self.logger.error(f"❌ Error recording system metrics: {e}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Record system metrics
                self.record_system_metrics()
                
                # Check for optimization opportunities
                await self._check_optimization_opportunities()
                
                # Sleep
                await asyncio.sleep(1.0)  # Monitor every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Monitoring loop error: {e}")
                await asyncio.sleep(1)
    
    async def _check_optimization_opportunities(self):
        """Check for optimization opportunities"""
        try:
            # Get current performance stats
            stats = self.get_comprehensive_performance_report()
            
            # Check if optimization is needed
            optimization_needed = False
            optimization_reasons = []
            
            # Check response latency trends
            response_stats = self.metrics_collector.get_statistics("response_latency")
            if response_stats and response_stats.get("p95", 0) > self.target_latency_ms * 1.2:
                optimization_needed = True
                optimization_reasons.append("High P95 response latency")
            
            # Check error rates
            error_stats = self.metrics_collector.get_statistics("error_rate")
            if error_stats and error_stats.get("latest", 0) > 0.02:
                optimization_needed = True
                optimization_reasons.append("High error rate")
            
            # Trigger optimization if needed
            if optimization_needed and time.time() - self.last_optimization > 30:  # 30 second cooldown
                await self._trigger_optimization(optimization_reasons, stats)
                
        except Exception as e:
            self.logger.error(f"❌ Error checking optimization opportunities: {e}")
    
    async def _trigger_optimization(self, reasons: List[str], stats: Dict[str, Any]):
        """Trigger performance optimization"""
        try:
            self.last_optimization = time.time()
            
            optimization_data = {
                "reasons": reasons,
                "timestamp": self.last_optimization,
                "current_stats": stats,
                "quality_level": self.quality_controller.current_level.value
            }
            
            self.logger.info(f"🔧 Triggering optimization: {', '.join(reasons)}")
            
            # Notify callbacks
            for callback in self.optimization_callbacks:
                try:
                    callback(optimization_data)
                except Exception as e:
                    self.logger.error(f"❌ Optimization callback error: {e}")
                    
        except Exception as e:
            self.logger.error(f"❌ Error triggering optimization: {e}")
    
    def _on_network_condition_change(self, condition: NetworkCondition):
        """Handle network condition changes"""
        self.logger.debug(f"🌐 Network condition: {condition.latency_ms:.1f}ms latency, {condition.stability_score:.2f} stability")
    
    def _on_quality_change(self, level: QualityLevel, settings: QualitySettings):
        """Handle quality level changes"""
        self.logger.info(f"🎛️ Quality changed to {level.value}: {settings.audio_sample_rate}Hz, {settings.buffer_size} buffer")
    
    def _on_alert(self, alert: PerformanceAlert):
        """Handle performance alerts"""
        self.logger.warning(f"🚨 Performance Alert: {alert.message}")
    
    def add_optimization_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for optimization triggers"""
        self.optimization_callbacks.append(callback)
    
    def get_current_quality_settings(self) -> QualitySettings:
        """Get current quality settings"""
        return self.quality_controller.current_settings
    
    def force_quality_level(self, level: QualityLevel) -> bool:
        """Force specific quality level"""
        return self.quality_controller.force_quality_level(level)
    
    def get_comprehensive_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        try:
            return {
                "session_duration": time.time() - self.session_start,
                "target_latency_ms": self.target_latency_ms,
                "current_quality": {
                    "level": self.quality_controller.current_level.value,
                    "settings": self.quality_controller.current_settings.__dict__
                },
                "network_condition": {
                    "latency_ms": self.network_monitor.current_condition.latency_ms,
                    "stability_score": self.network_monitor.current_condition.stability_score,
                    "quality_recommendation": self.network_monitor.current_condition.get_quality_recommendation().value
                },
                "metrics": self.metrics_collector.get_comprehensive_stats(),
                "alerts": {
                    "active": [alert.to_dict() for alert in self.alert_manager.get_active_alerts()],
                    "stats": self.alert_manager.get_alert_stats()
                },
                "adaptations": self.quality_controller.get_adaptation_stats()
            }
        except Exception as e:
            self.logger.error(f"❌ Error generating performance report: {e}")
            return {}


# Factory function for easy creation
def create_performance_monitor(
    target_latency_ms: float = 250.0,
    logger: Optional[logging.Logger] = None
) -> PerformanceMonitor:
    """Create performance monitor with standard configuration"""
    return PerformanceMonitor(target_latency_ms, logger) 