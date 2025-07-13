"""
Connection Stability Monitor for OpenAI Realtime API

Provides comprehensive monitoring of WebSocket connection stability, network quality,
and connection health metrics specifically for the Realtime API. Integrates with
the existing RealtimeMetricsCollector for centralized metrics collection.

Features:
- Real-time connection health monitoring
- Network quality assessment and scoring
- Connection drop prediction using pattern analysis
- Automatic reconnection strategy optimization
- Latency spike detection and analysis
- WebSocket heartbeat monitoring
- Connection resilience scoring
"""

import asyncio
import logging
import time
import threading
import statistics
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import json
try:
    import ping3
    HAS_PING3 = True
except ImportError:
    HAS_PING3 = False
    ping3 = None
import socket
import ssl
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

# Integration with existing metrics system
from .realtime_metrics_collector import (
    RealtimeMetricsCollector, ConnectionState, 
    RealtimeConnectionMetrics, create_realtime_metrics_collector
)


class ConnectionQuality(Enum):
    """Connection quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class NetworkTestType(Enum):
    """Types of network tests"""
    PING = "ping"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    JITTER = "jitter"
    PACKET_LOSS = "packet_loss"


@dataclass
class ConnectionHealthMetrics:
    """Comprehensive connection health metrics"""
    # Basic connection state
    is_connected: bool = False
    connection_duration: float = 0.0
    last_heartbeat: Optional[float] = None
    heartbeat_latency_ms: float = 0.0
    
    # Quality metrics
    connection_quality: ConnectionQuality = ConnectionQuality.UNKNOWN
    stability_score: float = 0.0  # 0-100 scale
    reliability_score: float = 0.0  # 0-100 scale
    
    # Network performance
    avg_latency_ms: float = 0.0
    latency_jitter_ms: float = 0.0
    packet_loss_rate: float = 0.0
    bandwidth_mbps: float = 0.0
    
    # Connection patterns
    total_connections: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    reconnection_attempts: int = 0
    connection_drops: int = 0
    
    # Error analysis
    error_rate: float = 0.0
    last_error_time: Optional[float] = None
    error_types: Dict[str, int] = field(default_factory=dict)
    
    # Timestamps
    last_updated: float = field(default_factory=time.time)
    measurement_window: float = 300.0  # 5 minutes


@dataclass
class NetworkQualityAssessment:
    """Network quality assessment results"""
    timestamp: float = field(default_factory=time.time)
    overall_score: float = 0.0  # 0-100
    latency_score: float = 0.0
    stability_score: float = 0.0
    throughput_score: float = 0.0
    
    # Detailed measurements
    ping_latency_ms: float = 0.0
    dns_resolution_ms: float = 0.0
    ssl_handshake_ms: float = 0.0
    websocket_connect_ms: float = 0.0
    
    # Quality indicators
    jitter_ms: float = 0.0
    packet_loss_percentage: float = 0.0
    connection_stability: str = "unknown"
    
    # Recommendations
    recommended_actions: List[str] = field(default_factory=list)
    connection_suitable: bool = True
    estimated_reliability: float = 0.0


@dataclass
class ConnectionEvent:
    """Individual connection event for pattern analysis"""
    timestamp: float
    event_type: str  # connect, disconnect, error, heartbeat, latency_spike
    duration_ms: Optional[float] = None
    error_details: Optional[Dict[str, Any]] = None
    context: Dict[str, Any] = field(default_factory=dict)


class ConnectionStabilityMonitor:
    """
    Advanced connection stability monitoring system for Realtime API
    
    Monitors WebSocket health, predicts connection issues, and provides
    intelligent reconnection strategies with comprehensive metrics collection.
    """
    
    def __init__(self,
                 realtime_metrics_collector: Optional[RealtimeMetricsCollector] = None,
                 websocket_endpoint: str = "wss://api.openai.com/v1/realtime",
                 monitoring_interval: float = 5.0,
                 heartbeat_interval: float = 30.0,
                 logger: Optional[logging.Logger] = None):
        
        self.realtime_metrics_collector = realtime_metrics_collector or create_realtime_metrics_collector()
        self.websocket_endpoint = websocket_endpoint
        self.monitoring_interval = monitoring_interval
        self.heartbeat_interval = heartbeat_interval
        self.logger = logger or logging.getLogger(__name__)
        
        # Connection health tracking
        self.health_metrics = ConnectionHealthMetrics()
        self.network_assessment = NetworkQualityAssessment()
        
        # Historical data for pattern analysis
        self.connection_events: deque = deque(maxlen=1000)
        self.latency_history: deque = deque(maxlen=500)
        self.heartbeat_history: deque = deque(maxlen=100)
        self.quality_history: deque = deque(maxlen=200)
        
        # Active monitoring state
        self.is_monitoring = False
        self.current_websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.monitoring_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        
        # Threading for background tasks
        self._lock = threading.RLock()
        self.background_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Event callbacks
        self.on_connection_quality_change: Optional[Callable] = None
        self.on_connection_unstable: Optional[Callable] = None
        self.on_reconnection_needed: Optional[Callable] = None
        self.on_network_degradation: Optional[Callable] = None
        
        # Prediction models (simple pattern-based for now)
        self.connection_patterns: Dict[str, Any] = {}
        self.failure_predictors: List[Dict[str, Any]] = []
        
        # Configuration
        self.config = {
            "latency_threshold_ms": 500.0,
            "jitter_threshold_ms": 100.0,
            "packet_loss_threshold": 0.05,  # 5%
            "stability_window_minutes": 5,
            "heartbeat_timeout_seconds": 60,
            "quality_assessment_interval": 60,  # seconds
            "reconnection_backoff_base": 2.0,
            "max_reconnection_attempts": 5
        }
        
        self.logger.info("ConnectionStabilityMonitor initialized")
    
    async def start_monitoring(self, websocket: Optional[websockets.WebSocketServerProtocol] = None) -> bool:
        """Start connection stability monitoring"""
        try:
            if self.is_monitoring:
                self.logger.warning("Connection monitoring already active")
                return True
            
            self.current_websocket = websocket
            self.is_monitoring = True
            self.stop_event.clear()
            
            # Start monitoring tasks
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            # Start background thread for network quality assessment
            self.background_thread = threading.Thread(
                target=self._background_monitoring_loop,
                daemon=True
            )
            self.background_thread.start()
            
            # Initial network quality assessment
            await self._assess_network_quality()
            
            # Record monitoring start event
            self._record_connection_event("monitoring_started")
            
            self.logger.info("🔍 Connection stability monitoring started")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start connection monitoring: {e}")
            self.is_monitoring = False
            return False
    
    async def stop_monitoring(self) -> None:
        """Stop connection stability monitoring"""
        try:
            self.is_monitoring = False
            self.stop_event.set()
            
            # Cancel monitoring tasks
            if self.monitoring_task and not self.monitoring_task.done():
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            if self.heartbeat_task and not self.heartbeat_task.done():
                self.heartbeat_task.cancel()
                try:
                    await self.heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            # Wait for background thread
            if self.background_thread and self.background_thread.is_alive():
                self.background_thread.join(timeout=2.0)
            
            self._record_connection_event("monitoring_stopped")
            self.logger.info("🔍 Connection stability monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping connection monitoring: {e}")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Update connection health metrics
                await self._update_health_metrics()
                
                # Check for connection issues
                await self._detect_connection_issues()
                
                # Update quality scoring
                self._calculate_quality_scores()
                
                # Check for prediction patterns
                self._analyze_connection_patterns()
                
                # Update metrics collector
                self._update_realtime_metrics()
                
                # Sleep until next monitoring cycle
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Error in monitoring loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _heartbeat_loop(self) -> None:
        """WebSocket heartbeat monitoring loop"""
        while self.is_monitoring:
            try:
                if self.current_websocket:
                    # Send heartbeat ping
                    heartbeat_start = time.time()
                    
                    try:
                        pong_waiter = await self.current_websocket.ping()
                        await asyncio.wait_for(pong_waiter, timeout=10.0)
                        
                        # Calculate heartbeat latency
                        heartbeat_latency = (time.time() - heartbeat_start) * 1000
                        
                        # Update metrics
                        with self._lock:
                            self.health_metrics.last_heartbeat = time.time()
                            self.health_metrics.heartbeat_latency_ms = heartbeat_latency
                            self.heartbeat_history.append(heartbeat_latency)
                        
                        # Record successful heartbeat
                        self._record_connection_event("heartbeat", duration_ms=heartbeat_latency)
                        
                        self.logger.debug(f"💓 Heartbeat: {heartbeat_latency:.1f}ms")
                        
                    except asyncio.TimeoutError:
                        self.logger.warning("⚠️ Heartbeat timeout")
                        self._record_connection_event("heartbeat_timeout")
                        
                        # Trigger connection quality degradation
                        if self.on_connection_unstable:
                            await self.on_connection_unstable("heartbeat_timeout")
                    
                    except Exception as e:
                        self.logger.error(f"❌ Heartbeat failed: {e}")
                        self._record_connection_event("heartbeat_error", error_details={"error": str(e)})
                
                # Wait for next heartbeat
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Error in heartbeat loop: {e}")
                await asyncio.sleep(5.0)
    
    def _background_monitoring_loop(self) -> None:
        """Background thread for network quality monitoring"""
        while not self.stop_event.is_set():
            try:
                # Perform network quality assessment
                asyncio.run(self._assess_network_quality())
                
                # Wait for next assessment
                self.stop_event.wait(self.config["quality_assessment_interval"])
                
            except Exception as e:
                self.logger.error(f"❌ Error in background monitoring: {e}")
                self.stop_event.wait(10.0)
    
    async def _update_health_metrics(self) -> None:
        """Update connection health metrics"""
        try:
            with self._lock:
                current_time = time.time()
                
                # Update basic connection state
                if self.current_websocket:
                    self.health_metrics.is_connected = not self.current_websocket.closed
                else:
                    self.health_metrics.is_connected = False
                
                # Calculate connection duration
                if hasattr(self, '_connection_start_time'):
                    self.health_metrics.connection_duration = current_time - self._connection_start_time
                
                # Calculate average latency
                if self.latency_history:
                    self.health_metrics.avg_latency_ms = statistics.mean(self.latency_history)
                    
                    # Calculate jitter (latency standard deviation)
                    if len(self.latency_history) > 1:
                        self.health_metrics.latency_jitter_ms = statistics.stdev(self.latency_history)
                
                # Update error rate
                recent_events = [e for e in self.connection_events 
                               if current_time - e.timestamp <= 300]  # Last 5 minutes
                
                if recent_events:
                    error_events = [e for e in recent_events if 'error' in e.event_type]
                    self.health_metrics.error_rate = len(error_events) / len(recent_events)
                
                self.health_metrics.last_updated = current_time
                
        except Exception as e:
            self.logger.error(f"❌ Error updating health metrics: {e}")
    
    async def _detect_connection_issues(self) -> None:
        """Detect potential connection issues"""
        try:
            # Check for latency spikes
            if (self.health_metrics.avg_latency_ms > self.config["latency_threshold_ms"] and
                len(self.latency_history) >= 5):
                
                self.logger.warning(f"⚠️ High latency detected: {self.health_metrics.avg_latency_ms:.1f}ms")
                self._record_connection_event("latency_spike", 
                                            duration_ms=self.health_metrics.avg_latency_ms)
                
                if self.on_network_degradation:
                    await self.on_network_degradation("high_latency")
            
            # Check for jitter issues
            if (self.health_metrics.latency_jitter_ms > self.config["jitter_threshold_ms"] and
                len(self.latency_history) >= 10):
                
                self.logger.warning(f"⚠️ High jitter detected: {self.health_metrics.latency_jitter_ms:.1f}ms")
                self._record_connection_event("high_jitter", 
                                            duration_ms=self.health_metrics.latency_jitter_ms)
            
            # Check heartbeat health
            if (self.health_metrics.last_heartbeat and 
                time.time() - self.health_metrics.last_heartbeat > self.config["heartbeat_timeout_seconds"]):
                
                self.logger.warning("⚠️ Heartbeat timeout - connection may be unstable")
                self._record_connection_event("heartbeat_stale")
                
                if self.on_connection_unstable:
                    await self.on_connection_unstable("heartbeat_stale")
            
            # Check error rate
            if self.health_metrics.error_rate > 0.1:  # 10% error rate
                self.logger.warning(f"⚠️ High error rate: {self.health_metrics.error_rate:.1%}")
                
                if self.on_connection_unstable:
                    await self.on_connection_unstable("high_error_rate")
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting connection issues: {e}")
    
    def _calculate_quality_scores(self) -> None:
        """Calculate connection quality scores"""
        try:
            with self._lock:
                # Latency score (0-100, where 100 is best)
                if self.health_metrics.avg_latency_ms <= 100:
                    latency_score = 100
                elif self.health_metrics.avg_latency_ms <= 300:
                    latency_score = 100 - ((self.health_metrics.avg_latency_ms - 100) / 200 * 50)
                else:
                    latency_score = max(0, 50 - ((self.health_metrics.avg_latency_ms - 300) / 500 * 50))
                
                # Stability score based on jitter and error rate
                jitter_penalty = min(50, self.health_metrics.latency_jitter_ms / 2)
                error_penalty = self.health_metrics.error_rate * 100
                stability_score = max(0, 100 - jitter_penalty - error_penalty)
                
                # Reliability score based on connection history
                if self.health_metrics.total_connections > 0:
                    success_rate = (self.health_metrics.successful_connections / 
                                  self.health_metrics.total_connections)
                    reliability_score = success_rate * 100
                else:
                    reliability_score = 100
                
                # Overall connection quality
                overall_score = (latency_score * 0.4 + stability_score * 0.4 + reliability_score * 0.2)
                
                # Update metrics
                self.health_metrics.stability_score = stability_score
                self.health_metrics.reliability_score = reliability_score
                
                # Determine quality level
                if overall_score >= 90:
                    quality = ConnectionQuality.EXCELLENT
                elif overall_score >= 75:
                    quality = ConnectionQuality.GOOD
                elif overall_score >= 60:
                    quality = ConnectionQuality.FAIR
                elif overall_score >= 40:
                    quality = ConnectionQuality.POOR
                else:
                    quality = ConnectionQuality.CRITICAL
                
                # Check for quality changes
                old_quality = self.health_metrics.connection_quality
                self.health_metrics.connection_quality = quality
                
                if old_quality != quality and self.on_connection_quality_change:
                    asyncio.create_task(self.on_connection_quality_change(old_quality, quality))
                
                # Store in history
                self.quality_history.append({
                    "timestamp": time.time(),
                    "overall_score": overall_score,
                    "latency_score": latency_score,
                    "stability_score": stability_score,
                    "reliability_score": reliability_score,
                    "quality": quality.value
                })
                
        except Exception as e:
            self.logger.error(f"❌ Error calculating quality scores: {e}")
    
    async def _assess_network_quality(self) -> NetworkQualityAssessment:
        """Perform comprehensive network quality assessment"""
        try:
            assessment = NetworkQualityAssessment()
            
            # Extract hostname from WebSocket endpoint
            import urllib.parse
            parsed_url = urllib.parse.urlparse(self.websocket_endpoint)
            hostname = parsed_url.hostname or "api.openai.com"
            
            # Ping test
            ping_start = time.time()
            if HAS_PING3 and ping3:
                ping_result = ping3.ping(hostname, timeout=5)
                if ping_result:
                    assessment.ping_latency_ms = ping_result * 1000
                    assessment.latency_score = max(0, 100 - (ping_result * 1000 / 10))
                else:
                    assessment.ping_latency_ms = float('inf')
                    assessment.latency_score = 0
            else:
                assessment.ping_latency_ms = float('inf')
                assessment.latency_score = 0
            
            # DNS resolution test
            dns_start = time.time()
            try:
                socket.getaddrinfo(hostname, 443)
                assessment.dns_resolution_ms = (time.time() - dns_start) * 1000
            except Exception:
                assessment.dns_resolution_ms = float('inf')
            
            # SSL handshake test
            ssl_start = time.time()
            try:
                context = ssl.create_default_context()
                with socket.create_connection((hostname, 443), timeout=10) as sock:
                    with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                        assessment.ssl_handshake_ms = (time.time() - ssl_start) * 1000
            except Exception:
                assessment.ssl_handshake_ms = float('inf')
            
            # Calculate overall score
            scores = []
            if assessment.latency_score > 0:
                scores.append(assessment.latency_score)
            if assessment.dns_resolution_ms < 1000:
                scores.append(max(0, 100 - assessment.dns_resolution_ms / 10))
            if assessment.ssl_handshake_ms < 5000:
                scores.append(max(0, 100 - assessment.ssl_handshake_ms / 50))
            
            assessment.overall_score = statistics.mean(scores) if scores else 0
            
            # Generate recommendations
            if assessment.overall_score < 50:
                assessment.recommended_actions.append("Check network connectivity")
                assessment.recommended_actions.append("Consider using cellular/different network")
                assessment.connection_suitable = False
            elif assessment.overall_score < 70:
                assessment.recommended_actions.append("Network performance may affect voice quality")
                assessment.recommended_actions.append("Consider enabling fallback mode")
            
            assessment.estimated_reliability = min(100, assessment.overall_score + 10) / 100
            
            # Update stored assessment
            with self._lock:
                self.network_assessment = assessment
            
            self.logger.debug(f"🌐 Network assessment: {assessment.overall_score:.1f}/100")
            return assessment
            
        except Exception as e:
            self.logger.error(f"❌ Error assessing network quality: {e}")
            return NetworkQualityAssessment()
    
    def _analyze_connection_patterns(self) -> None:
        """Analyze connection patterns for failure prediction"""
        try:
            # Simple pattern analysis - can be enhanced with ML
            recent_events = [e for e in self.connection_events 
                           if time.time() - e.timestamp <= 1800]  # Last 30 minutes
            
            if len(recent_events) < 10:
                return
            
            # Count event types
            event_counts = {}
            for event in recent_events:
                event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
            
            # Check for concerning patterns
            total_events = len(recent_events)
            error_ratio = (event_counts.get('error', 0) + 
                          event_counts.get('heartbeat_timeout', 0) + 
                          event_counts.get('disconnect', 0)) / total_events
            
            if error_ratio > 0.3:  # 30% error rate
                self.logger.warning(f"⚠️ High error pattern detected: {error_ratio:.1%}")
                
                # Suggest reconnection
                if self.on_reconnection_needed:
                    try:
                        # Check if we're in an async context
                        loop = asyncio.get_running_loop()
                        loop.create_task(self.on_reconnection_needed("high_error_pattern"))
                    except RuntimeError:
                        # No running loop, call sync if possible
                        if asyncio.iscoroutinefunction(self.on_reconnection_needed):
                            # Can't call async function without event loop
                            self.logger.debug("Skipping async callback - no event loop")
                        else:
                            # Call sync function
                            self.on_reconnection_needed("high_error_pattern")
                    except Exception as e:
                        self.logger.error(f"Error calling reconnection callback: {e}")
            
            # Store pattern analysis
            self.connection_patterns = {
                "event_counts": event_counts,
                "error_ratio": error_ratio,
                "analysis_time": time.time(),
                "total_events": total_events
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing connection patterns: {e}")
    
    def _update_realtime_metrics(self) -> None:
        """Update the RealtimeMetricsCollector with connection data"""
        try:
            if not self.realtime_metrics_collector:
                return
            
            # Update connection state
            if self.health_metrics.is_connected:
                state = ConnectionState.CONNECTED
                if self.health_metrics.connection_quality == ConnectionQuality.EXCELLENT:
                    state = ConnectionState.ACTIVE
                elif self.health_metrics.connection_quality in [ConnectionQuality.POOR, ConnectionQuality.CRITICAL]:
                    state = ConnectionState.ERROR
            else:
                state = ConnectionState.DISCONNECTED
            
            # Update connection metrics in the metrics collector
            self.realtime_metrics_collector.connection_metrics.connection_state = state
            self.realtime_metrics_collector.connection_metrics.total_uptime_seconds = self.health_metrics.connection_duration
            
            # Record connection events
            for event in list(self.connection_events)[-10:]:  # Last 10 events
                if event.event_type == "connect":
                    self.realtime_metrics_collector.record_connection_event("connection_attempt", True)
                elif event.event_type == "disconnect":
                    self.realtime_metrics_collector.record_connection_event("disconnection", 
                                                                           reason=event.context.get("reason", "unknown"))
                elif "error" in event.event_type:
                    self.realtime_metrics_collector.record_api_error(
                        error_type="connection_error",
                        error_code=event.context.get("error_code", "unknown"),
                        error_message=str(event.error_details) if event.error_details else event.event_type
                    )
            
            # Clear processed events to avoid duplicates
            self.connection_events.clear()
            
        except Exception as e:
            self.logger.error(f"❌ Error updating realtime metrics: {e}")
    
    def _record_connection_event(self, event_type: str, duration_ms: Optional[float] = None, 
                                error_details: Optional[Dict[str, Any]] = None,
                                context: Optional[Dict[str, Any]] = None) -> None:
        """Record a connection event for analysis"""
        try:
            event = ConnectionEvent(
                timestamp=time.time(),
                event_type=event_type,
                duration_ms=duration_ms,
                error_details=error_details,
                context=context or {}
            )
            
            with self._lock:
                self.connection_events.append(event)
            
            self.logger.debug(f"📝 Connection event: {event_type}")
            
        except Exception as e:
            self.logger.error(f"❌ Error recording connection event: {e}")
    
    def record_connection_attempt(self, success: bool, error_details: Optional[Dict] = None) -> None:
        """Record a connection attempt (called by external systems)"""
        with self._lock:
            self.health_metrics.total_connections += 1
            
            if success:
                self.health_metrics.successful_connections += 1
                self._connection_start_time = time.time()
                self._record_connection_event("connect")
            else:
                self.health_metrics.failed_connections += 1
                self._record_connection_event("connect_failed", error_details=error_details)
    
    def record_disconnection(self, reason: str = "unknown") -> None:
        """Record a disconnection event (called by external systems)"""
        with self._lock:
            self.health_metrics.connection_drops += 1
            self._record_connection_event("disconnect", context={"reason": reason})
    
    def record_latency_measurement(self, latency_ms: float) -> None:
        """Record a latency measurement (called by external systems)"""
        with self._lock:
            self.latency_history.append(latency_ms)
    
    def get_connection_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive connection health summary"""
        with self._lock:
            return {
                "timestamp": time.time(),
                "connection_health": {
                    "is_connected": self.health_metrics.is_connected,
                    "connection_duration": self.health_metrics.connection_duration,
                    "quality": self.health_metrics.connection_quality.value,
                    "stability_score": self.health_metrics.stability_score,
                    "reliability_score": self.health_metrics.reliability_score,
                    "avg_latency_ms": self.health_metrics.avg_latency_ms,
                    "jitter_ms": self.health_metrics.latency_jitter_ms,
                    "error_rate": self.health_metrics.error_rate
                },
                "network_quality": {
                    "overall_score": self.network_assessment.overall_score,
                    "ping_latency_ms": self.network_assessment.ping_latency_ms,
                    "dns_resolution_ms": self.network_assessment.dns_resolution_ms,
                    "ssl_handshake_ms": self.network_assessment.ssl_handshake_ms,
                    "connection_suitable": self.network_assessment.connection_suitable,
                    "estimated_reliability": self.network_assessment.estimated_reliability
                },
                "connection_stats": {
                    "total_connections": self.health_metrics.total_connections,
                    "successful_connections": self.health_metrics.successful_connections,
                    "failed_connections": self.health_metrics.failed_connections,
                    "connection_drops": self.health_metrics.connection_drops,
                    "reconnection_attempts": self.health_metrics.reconnection_attempts
                },
                "recent_events": len(self.connection_events),
                "monitoring_active": self.is_monitoring
            }
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            # Check if we're in an async context
            try:
                loop = asyncio.get_running_loop()
                # If we have a running loop, create a task
                loop.create_task(self.stop_monitoring())
            except RuntimeError:
                # No running loop, create a new one for cleanup
                asyncio.run(self.stop_monitoring())
            self.logger.info("ConnectionStabilityMonitor cleaned up")
        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")


# Factory function for easy creation
def create_connection_stability_monitor(
    realtime_metrics_collector: Optional[RealtimeMetricsCollector] = None,
    websocket_endpoint: str = "wss://api.openai.com/v1/realtime",
    monitoring_interval: float = 5.0,
    heartbeat_interval: float = 30.0,
    logger: Optional[logging.Logger] = None
) -> ConnectionStabilityMonitor:
    """Create a ConnectionStabilityMonitor instance"""
    return ConnectionStabilityMonitor(
        realtime_metrics_collector=realtime_metrics_collector,
        websocket_endpoint=websocket_endpoint,
        monitoring_interval=monitoring_interval,
        heartbeat_interval=heartbeat_interval,
        logger=logger
    )


# Export main components
__all__ = [
    'ConnectionStabilityMonitor',
    'ConnectionHealthMetrics',
    'NetworkQualityAssessment',
    'ConnectionEvent',
    'ConnectionQuality',
    'NetworkTestType',
    'create_connection_stability_monitor'
] 