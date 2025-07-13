"""
Performance Monitoring and Latency Tracking Module

This module provides comprehensive monitoring for the VoiceAssistantPipeline:
- Stage-specific latency measurement
- Real-time performance tracking
- Threshold alerting and warnings
- Performance metrics aggregation
- Detailed timing analysis
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import statistics
import json

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline stages for latency measurement"""
    AUDIO_CAPTURE = "audio_capture"
    STT_PROCESSING = "stt_processing"
    TTS_GENERATION = "tts_generation"
    AUDIO_PLAYBACK = "audio_playback"
    TOTAL_ROUND_TRIP = "total_round_trip"


class AlertLevel(Enum):
    """Alert levels for performance monitoring"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class StageMetrics:
    """Metrics for a specific pipeline stage"""
    stage: PipelineStage
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    recent_durations: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add_measurement(self, duration: float, success: bool = True):
        """Add a new measurement to the metrics"""
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
            
        if success and duration > 0:
            self.total_duration += duration
            self.min_duration = min(self.min_duration, duration)
            self.max_duration = max(self.max_duration, duration)
            self.recent_durations.append(duration)
    
    @property
    def average_duration(self) -> float:
        """Calculate average duration for successful calls"""
        if self.successful_calls == 0:
            return 0.0
        return self.total_duration / self.successful_calls
    
    @property
    def recent_average(self) -> float:
        """Calculate average of recent measurements"""
        if not self.recent_durations:
            return 0.0
        return statistics.mean(self.recent_durations)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        if self.total_calls == 0:
            return 0.0
        return (self.successful_calls / self.total_calls) * 100
    
    @property
    def p95_duration(self) -> float:
        """Calculate 95th percentile duration"""
        if len(self.recent_durations) < 2:
            return self.recent_average
        return statistics.quantiles(list(self.recent_durations), n=20)[18]  # 95th percentile


@dataclass
class PerformanceThresholds:
    """Performance thresholds for alerting"""
    # Latency thresholds in seconds
    audio_capture_warning: float = 0.05  # 50ms
    audio_capture_critical: float = 0.1   # 100ms
    
    stt_processing_warning: float = 0.2   # 200ms
    stt_processing_critical: float = 0.3  # 300ms
    
    tts_generation_warning: float = 0.3   # 300ms
    tts_generation_critical: float = 0.4  # 400ms
    
    audio_playback_warning: float = 0.05  # 50ms
    audio_playback_critical: float = 0.1  # 100ms
    
    total_round_trip_warning: float = 0.6  # 600ms
    total_round_trip_critical: float = 0.8 # 800ms
    
    # Success rate thresholds (percentage)
    success_rate_warning: float = 95.0
    success_rate_critical: float = 90.0
    
    def get_stage_thresholds(self, stage: PipelineStage) -> tuple[float, float]:
        """Get warning and critical thresholds for a stage"""
        threshold_map = {
            PipelineStage.AUDIO_CAPTURE: (self.audio_capture_warning, self.audio_capture_critical),
            PipelineStage.STT_PROCESSING: (self.stt_processing_warning, self.stt_processing_critical),
            PipelineStage.TTS_GENERATION: (self.tts_generation_warning, self.tts_generation_critical),
            PipelineStage.AUDIO_PLAYBACK: (self.audio_playback_warning, self.audio_playback_critical),
            PipelineStage.TOTAL_ROUND_TRIP: (self.total_round_trip_warning, self.total_round_trip_critical),
        }
        return threshold_map.get(stage, (0.5, 1.0))


@dataclass
class PerformanceAlert:
    """Alert for performance threshold violations"""
    timestamp: float
    stage: PipelineStage
    level: AlertLevel
    message: str
    value: float
    threshold: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'stage': self.stage.value,
            'level': self.level.value,
            'message': self.message,
            'value': self.value,
            'threshold': self.threshold
        }


class TimingContext:
    """Context manager for measuring execution time"""
    
    def __init__(self, monitor: 'PerformanceMonitor', stage: PipelineStage, 
                 metadata: Optional[Dict[str, Any]] = None):
        self.monitor = monitor
        self.stage = stage
        self.metadata = metadata or {}
        self.start_time = None
        self.end_time = None
        self.success = True
        
    def __enter__(self):
        self.start_time = time.time()
        logger.debug(f"Started timing {self.stage.value}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        # Mark as failure if exception occurred
        if exc_type is not None:
            self.success = False
            logger.error(f"Stage {self.stage.value} failed after {duration:.3f}s: {exc_val}")
        else:
            logger.debug(f"Completed timing {self.stage.value}: {duration:.3f}s")
        
        # Record the measurement
        self.monitor.record_measurement(self.stage, duration, self.success, self.metadata)
        
        # Don't suppress exceptions
        return False
    
    def mark_failure(self, error_message: str = ""):
        """Mark the timing as a failure"""
        self.success = False
        if error_message:
            self.metadata['error'] = error_message


class PerformanceMonitor:
    """Main performance monitoring class"""
    
    def __init__(self, thresholds: Optional[PerformanceThresholds] = None):
        self.thresholds = thresholds or PerformanceThresholds()
        self.metrics: Dict[PipelineStage, StageMetrics] = {}
        self.alerts: deque = deque(maxlen=1000)  # Store last 1000 alerts
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        self._lock = threading.Lock()
        
        # Initialize metrics for all stages
        for stage in PipelineStage:
            self.metrics[stage] = StageMetrics(stage)
        
        logger.info("Performance monitor initialized")
    
    def timing_context(self, stage: PipelineStage, 
                      metadata: Optional[Dict[str, Any]] = None) -> TimingContext:
        """Create a timing context for measuring a stage"""
        return TimingContext(self, stage, metadata)
    
    def record_measurement(self, stage: PipelineStage, duration: float, 
                          success: bool = True, metadata: Optional[Dict[str, Any]] = None):
        """Record a performance measurement"""
        with self._lock:
            # Add measurement to metrics
            self.metrics[stage].add_measurement(duration, success)
            
            # Check for threshold violations
            if success:
                self._check_thresholds(stage, duration)
            
            # Log the measurement
            status = "success" if success else "failure"
            logger.info(f"Stage {stage.value}: {duration:.3f}s ({status})")
            
            if metadata:
                logger.debug(f"Stage {stage.value} metadata: {metadata}")
    
    def _check_thresholds(self, stage: PipelineStage, duration: float):
        """Check if duration exceeds thresholds and create alerts"""
        warning_threshold, critical_threshold = self.thresholds.get_stage_thresholds(stage)
        
        alert_level = None
        threshold = None
        
        if duration >= critical_threshold:
            alert_level = AlertLevel.CRITICAL
            threshold = critical_threshold
        elif duration >= warning_threshold:
            alert_level = AlertLevel.WARNING
            threshold = warning_threshold
        
        if alert_level:
            message = f"{stage.value} latency {duration:.3f}s exceeds {alert_level.value} threshold {threshold:.3f}s"
            
            alert = PerformanceAlert(
                timestamp=time.time(),
                stage=stage,
                level=alert_level,
                message=message,
                value=duration,
                threshold=threshold
            )
            
            self._add_alert(alert)
    
    def _add_alert(self, alert: PerformanceAlert):
        """Add an alert and notify callbacks"""
        with self._lock:
            self.alerts.append(alert)
            
            # Log the alert
            log_func = logger.critical if alert.level == AlertLevel.CRITICAL else logger.warning
            log_func(alert.message)
            
            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add a callback for alerts"""
        self.alert_callbacks.append(callback)
    
    def get_metrics(self, stage: Optional[PipelineStage] = None) -> Dict[str, Any]:
        """Get performance metrics"""
        with self._lock:
            if stage:
                return self._stage_metrics_to_dict(self.metrics[stage])
            
            return {
                stage.value: self._stage_metrics_to_dict(metrics)
                for stage, metrics in self.metrics.items()
            }
    
    def _stage_metrics_to_dict(self, metrics: StageMetrics) -> Dict[str, Any]:
        """Convert stage metrics to dictionary"""
        return {
            'stage': metrics.stage.value,
            'total_calls': metrics.total_calls,
            'successful_calls': metrics.successful_calls,
            'failed_calls': metrics.failed_calls,
            'success_rate': round(metrics.success_rate, 2),
            'average_duration': round(metrics.average_duration * 1000, 2),  # ms
            'recent_average': round(metrics.recent_average * 1000, 2),  # ms
            'min_duration': round(metrics.min_duration * 1000, 2) if metrics.min_duration != float('inf') else 0,  # ms
            'max_duration': round(metrics.max_duration * 1000, 2),  # ms
            'p95_duration': round(metrics.p95_duration * 1000, 2),  # ms
        }
    
    def get_recent_alerts(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        with self._lock:
            alerts = list(self.alerts)
            if limit:
                alerts = alerts[-limit:]
            return [alert.to_dict() for alert in alerts]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        with self._lock:
            total_metrics = self.metrics[PipelineStage.TOTAL_ROUND_TRIP]
            
            # Count recent alerts by level
            recent_time = time.time() - 300  # Last 5 minutes
            recent_alerts = [a for a in self.alerts if a.timestamp >= recent_time]
            alert_counts = {
                'critical': len([a for a in recent_alerts if a.level == AlertLevel.CRITICAL]),
                'warning': len([a for a in recent_alerts if a.level == AlertLevel.WARNING]),
                'info': len([a for a in recent_alerts if a.level == AlertLevel.INFO]),
            }
            
            # Performance status
            avg_latency = total_metrics.recent_average
            target_latency = self.thresholds.total_round_trip_critical
            
            if avg_latency >= target_latency:
                performance_status = "critical"
            elif avg_latency >= self.thresholds.total_round_trip_warning:
                performance_status = "warning"
            else:
                performance_status = "good"
            
            return {
                'performance_status': performance_status,
                'average_latency_ms': round(avg_latency * 1000, 2),
                'target_latency_ms': round(target_latency * 1000, 2),
                'total_sessions': total_metrics.total_calls,
                'success_rate': round(total_metrics.success_rate, 2),
                'recent_alerts': alert_counts,
                'uptime_seconds': time.time() - getattr(self, '_start_time', time.time())
            }
    
    def reset_metrics(self):
        """Reset all metrics"""
        with self._lock:
            for stage in PipelineStage:
                self.metrics[stage] = StageMetrics(stage)
            self.alerts.clear()
            logger.info("Performance metrics reset")
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format"""
        if format.lower() != 'json':
            raise ValueError("Only JSON format is currently supported")
        
        data = {
            'timestamp': time.time(),
            'summary': self.get_summary(),
            'metrics': self.get_metrics(),
            'recent_alerts': self.get_recent_alerts(limit=50),
            'thresholds': {
                'audio_capture_warning': self.thresholds.audio_capture_warning,
                'audio_capture_critical': self.thresholds.audio_capture_critical,
                'stt_processing_warning': self.thresholds.stt_processing_warning,
                'stt_processing_critical': self.thresholds.stt_processing_critical,
                'tts_generation_warning': self.thresholds.tts_generation_warning,
                'tts_generation_critical': self.thresholds.tts_generation_critical,
                'audio_playback_warning': self.thresholds.audio_playback_warning,
                'audio_playback_critical': self.thresholds.audio_playback_critical,
                'total_round_trip_warning': self.thresholds.total_round_trip_warning,
                'total_round_trip_critical': self.thresholds.total_round_trip_critical,
            }
        }
        
        return json.dumps(data, indent=2)


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
        _global_monitor._start_time = time.time()
    return _global_monitor


def set_monitor(monitor: PerformanceMonitor):
    """Set the global performance monitor instance"""
    global _global_monitor
    _global_monitor = monitor


def timing(stage: PipelineStage, metadata: Optional[Dict[str, Any]] = None):
    """Decorator for timing function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_monitor()
            with monitor.timing_context(stage, metadata):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def async_timing(stage: PipelineStage, metadata: Optional[Dict[str, Any]] = None):
    """Decorator for timing async function execution"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            monitor = get_monitor()
            with monitor.timing_context(stage, metadata):
                return await func(*args, **kwargs)
        return wrapper
    return decorator 