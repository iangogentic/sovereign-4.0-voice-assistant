"""
Realtime API Metrics Collection System for Sovereign 4.0

Extends the existing monitoring infrastructure to provide specialized metrics
for OpenAI Realtime API performance, cost tracking, and connection stability.

Key Features:
- Response latency percentiles (P50, P95, P99) for voice-to-voice interactions
- Connection stability monitoring with WebSocket health tracking
- Cost calculation and usage monitoring with OpenAI pricing
- Audio quality metrics and session analytics
- Integration with existing MetricsCollector and dashboard systems
- Prometheus metrics export for external monitoring
"""

import asyncio
import logging
import time
import threading
import json
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import statistics
import numpy as np
from pathlib import Path

# Prometheus metrics (optional dependency)
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server, CollectorRegistry
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    logging.warning("prometheus_client not available. Prometheus metrics disabled.")

# Integration with existing monitoring system
from .metrics_collector import MetricsCollector, MetricType, ComponentType, get_metrics_collector
from .monitoring import PerformanceMonitor, get_monitor
from .health_monitoring import SystemHealthMonitor, get_health_monitor
from .config_manager import RealtimeAPIConfig


class RealtimeMetricType(Enum):
    """Realtime API specific metric types"""
    VOICE_LATENCY = "voice_latency"
    CONNECTION_HEALTH = "connection_health"
    AUDIO_QUALITY = "audio_quality"
    SESSION_ANALYTICS = "session_analytics"
    COST_TRACKING = "cost_tracking"
    API_ERRORS = "api_errors"


class ConnectionState(Enum):
    """WebSocket connection states for monitoring"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ACTIVE = "active"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class RealtimeLatencyMetrics:
    """Latency metrics specific to Realtime API voice interactions"""
    voice_to_voice_latency_ms: deque = field(default_factory=lambda: deque(maxlen=1000))
    audio_processing_latency_ms: deque = field(default_factory=lambda: deque(maxlen=1000))
    text_generation_latency_ms: deque = field(default_factory=lambda: deque(maxlen=1000))
    total_round_trip_ms: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add_latency(self, metric_type: str, latency_ms: float):
        """Add a latency measurement"""
        if hasattr(self, metric_type):
            getattr(self, metric_type).append(latency_ms)
    
    def get_percentiles(self, metric_type: str) -> Dict[str, float]:
        """Calculate percentiles for a latency metric"""
        if not hasattr(self, metric_type):
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        
        data = list(getattr(self, metric_type))
        if len(data) < 2:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        
        return {
            "p50": np.percentile(data, 50),
            "p95": np.percentile(data, 95),
            "p99": np.percentile(data, 99)
        }


@dataclass 
class RealtimeConnectionMetrics:
    """Connection stability and health metrics"""
    connection_state: ConnectionState = ConnectionState.DISCONNECTED
    connection_start_time: Optional[float] = None
    total_connections: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    reconnection_attempts: int = 0
    total_uptime_seconds: float = 0.0
    disconnection_events: deque = field(default_factory=lambda: deque(maxlen=100))
    heartbeat_responses: deque = field(default_factory=lambda: deque(maxlen=50))
    
    def record_connection_attempt(self, successful: bool):
        """Record a connection attempt"""
        self.total_connections += 1
        if successful:
            self.successful_connections += 1
            self.connection_start_time = time.time()
            self.connection_state = ConnectionState.CONNECTED
        else:
            self.failed_connections += 1
    
    def record_disconnection(self, reason: str = "unknown"):
        """Record a disconnection event"""
        if self.connection_start_time:
            duration = time.time() - self.connection_start_time
            self.total_uptime_seconds += duration
            self.disconnection_events.append({
                "timestamp": time.time(),
                "duration": duration,
                "reason": reason
            })
        self.connection_state = ConnectionState.DISCONNECTED
        self.connection_start_time = None
    
    @property
    def connection_success_rate(self) -> float:
        """Calculate connection success rate"""
        if self.total_connections == 0:
            return 0.0
        return (self.successful_connections / self.total_connections) * 100
    
    @property
    def average_session_duration(self) -> float:
        """Calculate average session duration"""
        if not self.disconnection_events:
            return 0.0
        durations = [event["duration"] for event in self.disconnection_events]
        return statistics.mean(durations)


@dataclass
class RealtimeAudioMetrics:
    """Audio quality and processing metrics"""
    audio_samples_processed: int = 0
    audio_quality_scores: deque = field(default_factory=lambda: deque(maxlen=100))
    volume_levels: deque = field(default_factory=lambda: deque(maxlen=200))
    silence_detection_events: int = 0
    audio_interruptions: int = 0
    sample_rate: int = 24000
    bit_depth: int = 16
    
    def record_audio_quality(self, quality_score: float, volume_level: float = 0.0):
        """Record audio quality metrics"""
        self.audio_quality_scores.append(quality_score)
        self.volume_levels.append(volume_level)
        self.audio_samples_processed += 1
    
    @property
    def average_audio_quality(self) -> float:
        """Calculate average audio quality"""
        if not self.audio_quality_scores:
            return 0.0
        return statistics.mean(self.audio_quality_scores)


@dataclass
class RealtimeCostMetrics:
    """Cost tracking and usage monitoring"""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_api_calls: int = 0
    session_costs: deque = field(default_factory=lambda: deque(maxlen=1000))
    hourly_costs: Dict[str, float] = field(default_factory=dict)
    daily_budget: float = 50.0  # Default $50/day budget
    
    # OpenAI Realtime API pricing (as of 2024)
    INPUT_TOKEN_COST_PER_1K = 0.006  # $0.006 per 1k input tokens
    OUTPUT_TOKEN_COST_PER_1K = 0.024  # $0.024 per 1k output tokens
    
    def add_usage(self, input_tokens: int, output_tokens: int):
        """Add token usage and calculate cost"""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_api_calls += 1
        
        # Calculate session cost
        input_cost = (input_tokens / 1000) * self.INPUT_TOKEN_COST_PER_1K
        output_cost = (output_tokens / 1000) * self.OUTPUT_TOKEN_COST_PER_1K
        session_cost = input_cost + output_cost
        
        self.session_costs.append({
            "timestamp": time.time(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": session_cost
        })
        
        # Track hourly costs
        hour_key = datetime.now().strftime("%Y-%m-%d-%H")
        self.hourly_costs[hour_key] = self.hourly_costs.get(hour_key, 0.0) + session_cost
    
    @property
    def total_cost(self) -> float:
        """Calculate total cost"""
        return sum(session["cost"] for session in self.session_costs)
    
    @property
    def current_hour_cost(self) -> float:
        """Get current hour cost"""
        hour_key = datetime.now().strftime("%Y-%m-%d-%H")
        return self.hourly_costs.get(hour_key, 0.0)
    
    @property
    def tokens_per_dollar(self) -> float:
        """Calculate average tokens per dollar"""
        total_cost = self.total_cost
        if total_cost == 0:
            return 0.0
        return (self.total_input_tokens + self.total_output_tokens) / total_cost


class RealtimeMetricsCollector:
    """
    Specialized metrics collector for OpenAI Realtime API
    
    Extends existing monitoring infrastructure with Realtime API specific
    metrics including voice latency, connection stability, and cost tracking.
    """
    
    def __init__(self,
                 realtime_config: Optional[RealtimeAPIConfig] = None,
                 metrics_collector: Optional[MetricsCollector] = None,
                 performance_monitor: Optional[PerformanceMonitor] = None,
                 logger: Optional[logging.Logger] = None):
        
        self.realtime_config = realtime_config
        self.metrics_collector = metrics_collector or get_metrics_collector()
        self.performance_monitor = performance_monitor or get_monitor()
        self.logger = logger or logging.getLogger(__name__)
        
        # Metrics storage
        self.latency_metrics = RealtimeLatencyMetrics()
        self.connection_metrics = RealtimeConnectionMetrics()
        self.audio_metrics = RealtimeAudioMetrics()
        self.cost_metrics = RealtimeCostMetrics()
        
        # Session tracking
        self.current_session_id: Optional[str] = None
        self.session_start_time: Optional[float] = None
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Prometheus metrics (if available)
        self.prometheus_metrics = {}
        if HAS_PROMETHEUS:
            self._initialize_prometheus_metrics()
        
        # Alert thresholds
        self.alert_thresholds = {
            "latency_ms": 500.0,
            "error_rate_percent": 5.0,
            "cost_per_hour": 10.0,
            "connection_failure_rate": 10.0
        }
        
        # Threading for async operations
        self._lock = threading.RLock()
        self.collection_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        self.logger.info("RealtimeMetricsCollector initialized")
    
    def _initialize_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        if not HAS_PROMETHEUS:
            return
        
        try:
            # Latency histograms
            self.prometheus_metrics['voice_latency'] = Histogram(
                'realtime_api_voice_latency_seconds',
                'Voice-to-voice response latency',
                buckets=[0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]
            )
            
            # Request counters
            self.prometheus_metrics['requests_total'] = Counter(
                'realtime_api_requests_total',
                'Total Realtime API requests',
                ['status', 'endpoint']
            )
            
            # Error counter
            self.prometheus_metrics['errors_total'] = Counter(
                'realtime_api_errors_total',
                'Total Realtime API errors',
                ['error_type', 'error_code']
            )
            
            # Cost gauge
            self.prometheus_metrics['cost_dollars'] = Gauge(
                'realtime_api_cost_dollars',
                'Current Realtime API cost',
                ['period']
            )
            
            # Connection state
            self.prometheus_metrics['connection_state'] = Gauge(
                'realtime_api_connection_state',
                'Connection state (0=disconnected, 1=connected, 2=active)',
            )
            
            # Token usage
            self.prometheus_metrics['tokens_total'] = Counter(
                'realtime_api_tokens_total',
                'Total tokens processed',
                ['token_type']
            )
            
            self.logger.info("Prometheus metrics initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Prometheus metrics: {e}")
    
    def start_session(self, session_id: str) -> None:
        """Start a new session tracking"""
        with self._lock:
            self.current_session_id = session_id
            self.session_start_time = time.time()
            
            self.active_sessions[session_id] = {
                "start_time": self.session_start_time,
                "input_tokens": 0,
                "output_tokens": 0,
                "latencies": [],
                "audio_events": [],
                "errors": []
            }
            
            self.logger.debug(f"Started session tracking: {session_id}")
    
    def end_session(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """End session tracking and return session metrics"""
        with self._lock:
            target_session = session_id or self.current_session_id
            
            if not target_session or target_session not in self.active_sessions:
                return {}
            
            session_data = self.active_sessions.pop(target_session)
            duration = time.time() - session_data["start_time"]
            
            # Calculate session metrics
            session_metrics = {
                "session_id": target_session,
                "duration_seconds": duration,
                "total_tokens": session_data["input_tokens"] + session_data["output_tokens"],
                "average_latency_ms": statistics.mean(session_data["latencies"]) if session_data["latencies"] else 0,
                "error_count": len(session_data["errors"]),
                "audio_events": len(session_data["audio_events"])
            }
            
            if target_session == self.current_session_id:
                self.current_session_id = None
                self.session_start_time = None
            
            self.logger.info(f"Session {target_session} ended: {duration:.1f}s, "
                           f"{session_metrics['total_tokens']} tokens, "
                           f"{session_metrics['average_latency_ms']:.0f}ms avg latency")
            
            return session_metrics
    
    def record_voice_latency(self, latency_ms: float, latency_type: str = "voice_to_voice_latency_ms") -> None:
        """Record voice interaction latency"""
        with self._lock:
            # Add to latency metrics
            self.latency_metrics.add_latency(latency_type, latency_ms)
            
            # Update current session
            if self.current_session_id and self.current_session_id in self.active_sessions:
                self.active_sessions[self.current_session_id]["latencies"].append(latency_ms)
            
            # Update Prometheus metrics
            if HAS_PROMETHEUS and 'voice_latency' in self.prometheus_metrics:
                self.prometheus_metrics['voice_latency'].observe(latency_ms / 1000.0)
            
            # Check alert thresholds
            if latency_ms > self.alert_thresholds["latency_ms"]:
                self._trigger_alert("high_latency", {
                    "latency_ms": latency_ms,
                    "threshold": self.alert_thresholds["latency_ms"],
                    "session_id": self.current_session_id
                })
            
            self.logger.debug(f"Recorded {latency_type}: {latency_ms:.1f}ms")
    
    def record_connection_event(self, event_type: str, success: bool = True, reason: str = "unknown") -> None:
        """Record connection-related events"""
        with self._lock:
            if event_type == "connection_attempt":
                self.connection_metrics.record_connection_attempt(success)
                
                # Update Prometheus connection state
                if HAS_PROMETHEUS and 'connection_state' in self.prometheus_metrics:
                    state_value = 1 if success else 0
                    self.prometheus_metrics['connection_state'].set(state_value)
                
            elif event_type == "disconnection":
                self.connection_metrics.record_disconnection(reason)
                
                # Update Prometheus connection state
                if HAS_PROMETHEUS and 'connection_state' in self.prometheus_metrics:
                    self.prometheus_metrics['connection_state'].set(0)
            
            elif event_type == "heartbeat":
                self.connection_metrics.heartbeat_responses.append(time.time())
            
            self.logger.debug(f"Connection event: {event_type}, success: {success}, reason: {reason}")
    
    def record_audio_metrics(self, quality_score: float, volume_level: float = 0.0, 
                           event_type: str = "processing") -> None:
        """Record audio quality and processing metrics"""
        with self._lock:
            self.audio_metrics.record_audio_quality(quality_score, volume_level)
            
            # Update current session
            if self.current_session_id and self.current_session_id in self.active_sessions:
                self.active_sessions[self.current_session_id]["audio_events"].append({
                    "timestamp": time.time(),
                    "quality": quality_score,
                    "volume": volume_level,
                    "type": event_type
                })
            
            self.logger.debug(f"Audio metrics: quality={quality_score:.2f}, volume={volume_level:.2f}")
    
    def record_token_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage and calculate costs"""
        with self._lock:
            # Add to cost metrics
            self.cost_metrics.add_usage(input_tokens, output_tokens)
            
            # Update current session
            if self.current_session_id and self.current_session_id in self.active_sessions:
                session = self.active_sessions[self.current_session_id]
                session["input_tokens"] += input_tokens
                session["output_tokens"] += output_tokens
            
            # Update Prometheus metrics
            if HAS_PROMETHEUS:
                if 'tokens_total' in self.prometheus_metrics:
                    self.prometheus_metrics['tokens_total'].labels(token_type='input').inc(input_tokens)
                    self.prometheus_metrics['tokens_total'].labels(token_type='output').inc(output_tokens)
                
                if 'cost_dollars' in self.prometheus_metrics:
                    self.prometheus_metrics['cost_dollars'].labels(period='current_hour').set(
                        self.cost_metrics.current_hour_cost
                    )
                    self.prometheus_metrics['cost_dollars'].labels(period='total').set(
                        self.cost_metrics.total_cost
                    )
            
            # Check cost alerts
            if self.cost_metrics.current_hour_cost > self.alert_thresholds["cost_per_hour"]:
                self._trigger_alert("high_cost", {
                    "current_hour_cost": self.cost_metrics.current_hour_cost,
                    "threshold": self.alert_thresholds["cost_per_hour"]
                })
            
            self.logger.debug(f"Token usage: {input_tokens} input, {output_tokens} output")
    
    def record_api_error(self, error_type: str, error_code: str = "unknown", 
                        error_message: str = "") -> None:
        """Record API errors and failures"""
        with self._lock:
            # Update current session
            if self.current_session_id and self.current_session_id in self.active_sessions:
                self.active_sessions[self.current_session_id]["errors"].append({
                    "timestamp": time.time(),
                    "type": error_type,
                    "code": error_code,
                    "message": error_message
                })
            
            # Update Prometheus metrics
            if HAS_PROMETHEUS and 'errors_total' in self.prometheus_metrics:
                self.prometheus_metrics['errors_total'].labels(
                    error_type=error_type, 
                    error_code=error_code
                ).inc()
            
            # Integrate with existing metrics collector
            if self.metrics_collector:
                self.metrics_collector.record_request(
                    component=ComponentType.LLM_INFERENCE.value,
                    success=False
                )
            
            self.logger.warning(f"API error: {error_type} ({error_code}): {error_message}")
    
    def _trigger_alert(self, alert_type: str, context: Dict[str, Any]) -> None:
        """Trigger alerts for threshold violations"""
        alert_data = {
            "type": alert_type,
            "timestamp": time.time(),
            "context": context,
            "session_id": self.current_session_id
        }
        
        # Log the alert
        self.logger.warning(f"ALERT: {alert_type} - {context}")
        
        # Integrate with health monitoring system
        health_monitor = get_health_monitor()
        if health_monitor:
            # Could trigger health monitor alerts here
            pass
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        with self._lock:
            # Calculate latency percentiles
            voice_latency_percentiles = self.latency_metrics.get_percentiles("voice_to_voice_latency_ms")
            
            return {
                "timestamp": time.time(),
                "session": {
                    "current_session_id": self.current_session_id,
                    "active_sessions": len(self.active_sessions),
                    "session_duration": time.time() - self.session_start_time if self.session_start_time else 0
                },
                "latency": {
                    "voice_to_voice_p50": voice_latency_percentiles["p50"],
                    "voice_to_voice_p95": voice_latency_percentiles["p95"],
                    "voice_to_voice_p99": voice_latency_percentiles["p99"],
                    "sample_count": len(self.latency_metrics.voice_to_voice_latency_ms)
                },
                "connection": {
                    "state": self.connection_metrics.connection_state.value,
                    "success_rate": self.connection_metrics.connection_success_rate,
                    "total_uptime": self.connection_metrics.total_uptime_seconds,
                    "reconnections": self.connection_metrics.reconnection_attempts
                },
                "audio": {
                    "average_quality": self.audio_metrics.average_audio_quality,
                    "samples_processed": self.audio_metrics.audio_samples_processed,
                    "interruptions": self.audio_metrics.audio_interruptions
                },
                "cost": {
                    "total_cost": self.cost_metrics.total_cost,
                    "current_hour_cost": self.cost_metrics.current_hour_cost,
                    "total_tokens": self.cost_metrics.total_input_tokens + self.cost_metrics.total_output_tokens,
                    "tokens_per_dollar": self.cost_metrics.tokens_per_dollar
                }
            }
    
    def export_metrics_for_dashboard(self) -> Dict[str, Any]:
        """Export metrics in dashboard-compatible format"""
        summary = self.get_metrics_summary()
        
        # Format for dashboard consumption
        return {
            "realtime_api": {
                "latency_ms": summary["latency"]["voice_to_voice_p95"],
                "connection_health": summary["connection"]["success_rate"],
                "cost_per_hour": summary["cost"]["current_hour_cost"],
                "active_sessions": summary["session"]["active_sessions"],
                "audio_quality": summary["audio"]["average_quality"],
                "error_rate": 0.0  # TODO: Calculate from error metrics
            },
            "detailed_metrics": summary
        }
    
    def cleanup(self) -> None:
        """Cleanup resources and stop collection"""
        with self._lock:
            self.is_running = False
            
            # End any active sessions
            for session_id in list(self.active_sessions.keys()):
                self.end_session(session_id)
            
            self.logger.info("RealtimeMetricsCollector cleaned up")


# Factory function for easy creation
def create_realtime_metrics_collector(
    realtime_config: Optional[RealtimeAPIConfig] = None,
    metrics_collector: Optional[MetricsCollector] = None,
    performance_monitor: Optional[PerformanceMonitor] = None,
    logger: Optional[logging.Logger] = None
) -> RealtimeMetricsCollector:
    """Create a RealtimeMetricsCollector instance"""
    return RealtimeMetricsCollector(
        realtime_config=realtime_config,
        metrics_collector=metrics_collector,
        performance_monitor=performance_monitor,
        logger=logger
    )


# Export main components
__all__ = [
    'RealtimeMetricsCollector',
    'RealtimeLatencyMetrics',
    'RealtimeConnectionMetrics', 
    'RealtimeAudioMetrics',
    'RealtimeCostMetrics',
    'RealtimeMetricType',
    'ConnectionState',
    'create_realtime_metrics_collector'
] 