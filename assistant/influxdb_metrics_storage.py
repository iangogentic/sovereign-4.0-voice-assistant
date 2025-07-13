"""
InfluxDB Time-series Metrics Storage for Sovereign 4.0 Realtime API

Provides long-term storage and historical analysis capabilities for Realtime API
metrics collected by RealtimeMetricsCollector. Integrates with InfluxDB 2.x for
scalable time-series data storage with configurable retention policies.

Key Features:
- Automatic schema creation with appropriate tags and fields
- Configurable retention policies for different metric types
- Batch writing for performance optimization
- Query interfaces for dashboard integration
- Connection pooling and error recovery
- Data aggregation and downsampling capabilities
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
from pathlib import Path
import statistics

# InfluxDB client (optional dependency)
try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS
    from influxdb_client.client.exceptions import InfluxDBError
    from influxdb_client.domain.bucket import Bucket
    from influxdb_client.domain.retention_rule import RetentionRule
    HAS_INFLUXDB = True
except ImportError:
    HAS_INFLUXDB = False
    logging.warning("influxdb-client not available. InfluxDB integration disabled.")

# Integration with existing monitoring system
from .realtime_metrics_collector import (
    RealtimeMetricsCollector, RealtimeLatencyMetrics, RealtimeConnectionMetrics,
    RealtimeAudioMetrics, RealtimeCostMetrics, RealtimeMetricType
)
from .config_manager import RealtimeAPIConfig


class InfluxDBConfig:
    """Configuration for InfluxDB connection and storage"""
    
    def __init__(self,
                 url: str = "http://localhost:8086",
                 token: Optional[str] = None,
                 org: str = "sovereign",
                 bucket: str = "realtime_metrics",
                 timeout: int = 10000,
                 batch_size: int = 1000,
                 flush_interval: int = 1000,  # milliseconds
                 retry_attempts: int = 3,
                 enable_debug: bool = False):
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.timeout = timeout
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.retry_attempts = retry_attempts
        self.enable_debug = enable_debug


class RetentionPolicy(Enum):
    """Retention policies for different types of metrics"""
    RAW_METRICS = "7d"      # Raw metrics: 7 days
    HOURLY_AGGREGATES = "30d"  # Hourly aggregates: 30 days  
    DAILY_AGGREGATES = "1y"    # Daily aggregates: 1 year
    COST_TRACKING = "2y"       # Cost data: 2 years
    ERROR_LOGS = "90d"         # Error logs: 90 days


@dataclass
class InfluxDBWriteStats:
    """Statistics for InfluxDB write operations"""
    total_points_written: int = 0
    successful_writes: int = 0
    failed_writes: int = 0
    last_write_time: Optional[float] = None
    write_durations: List[float] = field(default_factory=list)
    connection_errors: int = 0
    batch_queue_size: int = 0
    
    @property
    def average_write_duration(self) -> float:
        """Calculate average write duration"""
        if not self.write_durations:
            return 0.0
        return statistics.mean(self.write_durations[-100:])  # Last 100 writes
    
    @property
    def write_success_rate(self) -> float:
        """Calculate write success rate"""
        total_attempts = self.successful_writes + self.failed_writes
        if total_attempts == 0:
            return 100.0
        return (self.successful_writes / total_attempts) * 100


class InfluxDBMetricsStorage:
    """
    InfluxDB integration for storing Realtime API metrics with time-series capabilities.
    
    Provides automatic batching, retention policies, and query interfaces for
    historical analysis of voice assistant performance metrics.
    """
    
    def __init__(self,
                 config: InfluxDBConfig,
                 realtime_metrics: Optional[RealtimeMetricsCollector] = None,
                 logger: Optional[logging.Logger] = None):
        self.config = config
        self.realtime_metrics = realtime_metrics
        self.logger = logger or logging.getLogger(__name__)
        
        # InfluxDB client and APIs
        self._client: Optional[InfluxDBClient] = None
        self._write_api = None
        self._query_api = None
        self._buckets_api = None
        
        # Write statistics and monitoring
        self.write_stats = InfluxDBWriteStats()
        self._write_lock = threading.Lock()
        self._is_connected = False
        self._last_connection_attempt = 0
        self._connection_retry_delay = 5.0  # seconds
        
        # Batch writing configuration
        self._write_queue: List[Point] = []
        self._last_flush_time = time.time()
        
        # Initialize connection if InfluxDB is available
        if HAS_INFLUXDB:
            self._initialize_connection()
        else:
            self.logger.warning("InfluxDB client not available. Storage disabled.")
    
    def _initialize_connection(self) -> bool:
        """Initialize InfluxDB connection and APIs"""
        try:
            if not self.config.token:
                self.logger.warning("InfluxDB token not provided. Using default authentication.")
            
            self._client = InfluxDBClient(
                url=self.config.url,
                token=self.config.token,
                org=self.config.org,
                timeout=self.config.timeout,
                debug=self.config.enable_debug
            )
            
            # Test connection
            health = self._client.health()
            if health.status != "pass":
                self.logger.error(f"InfluxDB health check failed: {health.message}")
                return False
            
            # Initialize APIs
            self._write_api = self._client.write_api(
                write_options=ASYNCHRONOUS,
                batch_size=self.config.batch_size,
                flush_interval=self.config.flush_interval
            )
            self._query_api = self._client.query_api()
            self._buckets_api = self._client.buckets_api()
            
            # Setup bucket and retention policies
            self._setup_bucket_and_retention()
            
            self._is_connected = True
            self.logger.info(f"InfluxDB connection established: {self.config.url}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to InfluxDB: {e}")
            self._is_connected = False
            self.write_stats.connection_errors += 1
            return False
    
    def _setup_bucket_and_retention(self) -> None:
        """Create bucket with appropriate retention policies if it doesn't exist"""
        try:
            # Check if bucket exists
            existing_bucket = None
            try:
                existing_bucket = self._buckets_api.find_bucket_by_name(self.config.bucket)
            except InfluxDBError:
                pass  # Bucket doesn't exist
            
            if not existing_bucket:
                # Create bucket with default retention (7 days for raw metrics)
                retention_rules = [
                    RetentionRule(
                        type="expire",
                        every_seconds=int(timedelta(days=7).total_seconds())
                    )
                ]
                
                bucket = Bucket(
                    name=self.config.bucket,
                    org_id=self.config.org,
                    retention_rules=retention_rules,
                    description="Sovereign 4.0 Realtime API Metrics Storage"
                )
                
                self._buckets_api.create_bucket(bucket=bucket)
                self.logger.info(f"Created InfluxDB bucket: {self.config.bucket}")
            else:
                self.logger.info(f"Using existing InfluxDB bucket: {self.config.bucket}")
                
        except Exception as e:
            self.logger.error(f"Failed to setup InfluxDB bucket: {e}")
    
    def _ensure_connection(self) -> bool:
        """Ensure InfluxDB connection is active, reconnect if necessary"""
        if not HAS_INFLUXDB:
            return False
        
        if self._is_connected:
            return True
        
        # Throttle connection attempts
        current_time = time.time()
        if current_time - self._last_connection_attempt < self._connection_retry_delay:
            return False
        
        self._last_connection_attempt = current_time
        return self._initialize_connection()
    
    def write_latency_metrics(self, session_id: str, metrics: RealtimeLatencyMetrics) -> None:
        """Write latency metrics to InfluxDB"""
        if not self._ensure_connection():
            return
        
        try:
            points = []
            timestamp = datetime.utcnow()
            
            # Write voice-to-voice latency percentiles
            if metrics.voice_to_voice_latency_ms:
                latency_data = list(metrics.voice_to_voice_latency_ms)
                percentiles = metrics.get_percentiles("voice_to_voice_latency_ms")
                
                point = (Point("voice_latency")
                        .tag("session_id", session_id)
                        .tag("metric_type", "voice_to_voice")
                        .field("p50", percentiles["p50"])
                        .field("p95", percentiles["p95"])
                        .field("p99", percentiles["p99"])
                        .field("sample_count", len(latency_data))
                        .field("average", statistics.mean(latency_data) if latency_data else 0.0)
                        .time(timestamp, WritePrecision.MS))
                points.append(point)
            
            # Write audio processing latency
            if metrics.audio_processing_latency_ms:
                latency_data = list(metrics.audio_processing_latency_ms)
                percentiles = metrics.get_percentiles("audio_processing_latency_ms")
                
                point = (Point("voice_latency")
                        .tag("session_id", session_id)
                        .tag("metric_type", "audio_processing")
                        .field("p50", percentiles["p50"])
                        .field("p95", percentiles["p95"])
                        .field("p99", percentiles["p99"])
                        .field("sample_count", len(latency_data))
                        .field("average", statistics.mean(latency_data) if latency_data else 0.0)
                        .time(timestamp, WritePrecision.MS))
                points.append(point)
            
            # Write total round trip latency
            if metrics.total_round_trip_ms:
                latency_data = list(metrics.total_round_trip_ms)
                percentiles = metrics.get_percentiles("total_round_trip_ms")
                
                point = (Point("voice_latency")
                        .tag("session_id", session_id)
                        .tag("metric_type", "round_trip")
                        .field("p50", percentiles["p50"])
                        .field("p95", percentiles["p95"])
                        .field("p99", percentiles["p99"])
                        .field("sample_count", len(latency_data))
                        .field("average", statistics.mean(latency_data) if latency_data else 0.0)
                        .time(timestamp, WritePrecision.MS))
                points.append(point)
            
            self._queue_points_for_write(points)
            
        except Exception as e:
            self.logger.error(f"Failed to write latency metrics: {e}")
            self.write_stats.failed_writes += 1
    
    def write_connection_metrics(self, session_id: str, metrics: RealtimeConnectionMetrics) -> None:
        """Write connection metrics to InfluxDB"""
        if not self._ensure_connection():
            return
        
        try:
            timestamp = datetime.utcnow()
            
            point = (Point("connection_metrics")
                    .tag("session_id", session_id)
                    .tag("connection_state", metrics.connection_state.value)
                    .field("total_connections", metrics.total_connections)
                    .field("successful_connections", metrics.successful_connections)
                    .field("failed_connections", metrics.failed_connections)
                    .field("reconnection_attempts", metrics.reconnection_attempts)
                    .field("total_uptime_seconds", metrics.total_uptime_seconds)
                    .field("connection_success_rate", metrics.connection_success_rate)
                    .field("average_session_duration", metrics.average_session_duration)
                    .field("heartbeat_responses", len(metrics.heartbeat_responses))
                    .time(timestamp, WritePrecision.MS))
            
            self._queue_points_for_write([point])
            
        except Exception as e:
            self.logger.error(f"Failed to write connection metrics: {e}")
            self.write_stats.failed_writes += 1
    
    def write_audio_metrics(self, session_id: str, metrics: RealtimeAudioMetrics) -> None:
        """Write audio quality metrics to InfluxDB"""
        if not self._ensure_connection():
            return
        
        try:
            timestamp = datetime.utcnow()
            
            point = (Point("audio_metrics")
                    .tag("session_id", session_id)
                    .field("samples_processed", metrics.audio_samples_processed)
                    .field("average_quality", metrics.average_audio_quality)
                    .field("silence_detection_events", metrics.silence_detection_events)
                    .field("audio_interruptions", metrics.audio_interruptions)
                    .field("sample_rate", metrics.sample_rate)
                    .field("bit_depth", metrics.bit_depth)
                    .time(timestamp, WritePrecision.MS))
            
            # Add recent quality scores if available
            if metrics.audio_quality_scores:
                recent_scores = list(metrics.audio_quality_scores)[-10:]  # Last 10 scores
                if recent_scores:
                    point.field("recent_max_quality", max(recent_scores))
                    point.field("recent_min_quality", min(recent_scores))
                    point.field("recent_avg_quality", statistics.mean(recent_scores))
            
            # Add volume level statistics
            if metrics.volume_levels:
                recent_volumes = list(metrics.volume_levels)[-20:]  # Last 20 volume readings
                if recent_volumes:
                    point.field("average_volume", statistics.mean(recent_volumes))
                    point.field("max_volume", max(recent_volumes))
            
            self._queue_points_for_write([point])
            
        except Exception as e:
            self.logger.error(f"Failed to write audio metrics: {e}")
            self.write_stats.failed_writes += 1
    
    def write_cost_metrics(self, session_id: str, metrics: RealtimeCostMetrics) -> None:
        """Write cost tracking metrics to InfluxDB"""
        if not self._ensure_connection():
            return
        
        try:
            timestamp = datetime.utcnow()
            
            point = (Point("cost_metrics")
                    .tag("session_id", session_id)
                    .field("total_input_tokens", metrics.total_input_tokens)
                    .field("total_output_tokens", metrics.total_output_tokens)
                    .field("total_api_calls", metrics.total_api_calls)
                    .field("total_cost", metrics.total_cost)
                    .field("current_hour_cost", metrics.current_hour_cost)
                    .field("tokens_per_dollar", metrics.tokens_per_dollar)
                    .field("daily_budget", metrics.daily_budget)
                    .time(timestamp, WritePrecision.MS))
            
            # Add hourly cost breakdown for the last 24 hours
            if metrics.hourly_costs:
                recent_hours = sorted(metrics.hourly_costs.keys())[-24:]  # Last 24 hours
                for i, hour_key in enumerate(recent_hours):
                    point.field(f"hour_{i}_cost", metrics.hourly_costs[hour_key])
            
            self._queue_points_for_write([point])
            
        except Exception as e:
            self.logger.error(f"Failed to write cost metrics: {e}")
            self.write_stats.failed_writes += 1
    
    def _queue_points_for_write(self, points: List[Point]) -> None:
        """Queue points for batch writing"""
        with self._write_lock:
            self._write_queue.extend(points)
            self.write_stats.batch_queue_size = len(self._write_queue)
            
            # Check if we should flush based on queue size or time
            should_flush = (
                len(self._write_queue) >= self.config.batch_size or
                time.time() - self._last_flush_time > (self.config.flush_interval / 1000)
            )
            
            if should_flush:
                self._flush_write_queue()
    
    def _flush_write_queue(self) -> None:
        """Flush queued points to InfluxDB"""
        if not self._write_queue or not self._write_api:
            return
        
        points_to_write = self._write_queue.copy()
        self._write_queue.clear()
        self._last_flush_time = time.time()
        
        try:
            start_time = time.time()
            
            self._write_api.write(
                bucket=self.config.bucket,
                org=self.config.org,
                record=points_to_write
            )
            
            write_duration = time.time() - start_time
            
            # Update statistics
            self.write_stats.total_points_written += len(points_to_write)
            self.write_stats.successful_writes += 1
            self.write_stats.last_write_time = time.time()
            self.write_stats.write_durations.append(write_duration)
            self.write_stats.batch_queue_size = len(self._write_queue)
            
            # Keep only recent write durations
            if len(self.write_stats.write_durations) > 1000:
                self.write_stats.write_durations = self.write_stats.write_durations[-500:]
            
            self.logger.debug(f"Wrote {len(points_to_write)} points to InfluxDB in {write_duration:.3f}s")
            
        except Exception as e:
            self.logger.error(f"Failed to write {len(points_to_write)} points to InfluxDB: {e}")
            self.write_stats.failed_writes += 1
            
            # Re-queue points for retry (with backoff)
            if self.write_stats.failed_writes < self.config.retry_attempts:
                with self._write_lock:
                    self._write_queue = points_to_write + self._write_queue
    
    def query_latency_history(self, 
                             hours_back: int = 24,
                             session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query latency metrics history"""
        if not self._ensure_connection() or not self._query_api:
            return []
        
        try:
            session_filter = f'|> filter(fn: (r) => r.session_id == "{session_id}")' if session_id else ''
            
            query = f'''
                from(bucket: "{self.config.bucket}")
                |> range(start: -{hours_back}h)
                |> filter(fn: (r) => r._measurement == "voice_latency")
                {session_filter}
                |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
                |> yield(name: "latency_hourly")
            '''
            
            result = self._query_api.query(query=query, org=self.config.org)
            
            metrics = []
            for table in result:
                for record in table.records:
                    metrics.append({
                        "time": record.get_time(),
                        "session_id": record.values.get("session_id"),
                        "metric_type": record.values.get("metric_type"),
                        "field": record.get_field(),
                        "value": record.get_value()
                    })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to query latency history: {e}")
            return []
    
    def query_cost_analysis(self, days_back: int = 7) -> Dict[str, Any]:
        """Query cost analysis for the specified time period"""
        if not self._ensure_connection() or not self._query_api:
            return {}
        
        try:
            query = f'''
                from(bucket: "{self.config.bucket}")
                |> range(start: -{days_back}d)
                |> filter(fn: (r) => r._measurement == "cost_metrics")
                |> filter(fn: (r) => r._field == "total_cost" or r._field == "current_hour_cost")
                |> aggregateWindow(every: 1d, fn: last, createEmpty: false)
                |> yield(name: "daily_costs")
            '''
            
            result = self._query_api.query(query=query, org=self.config.org)
            
            daily_costs = []
            for table in result:
                for record in table.records:
                    if record.get_field() == "total_cost":
                        daily_costs.append({
                            "date": record.get_time().date(),
                            "total_cost": record.get_value()
                        })
            
            # Calculate summary statistics
            costs = [entry["total_cost"] for entry in daily_costs if entry["total_cost"]]
            
            return {
                "daily_costs": daily_costs,
                "total_cost": sum(costs) if costs else 0.0,
                "average_daily_cost": statistics.mean(costs) if costs else 0.0,
                "max_daily_cost": max(costs) if costs else 0.0,
                "min_daily_cost": min(costs) if costs else 0.0,
                "days_analyzed": len(daily_costs)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to query cost analysis: {e}")
            return {}
    
    def query_connection_stability(self, hours_back: int = 24) -> Dict[str, Any]:
        """Query connection stability metrics"""
        if not self._ensure_connection() or not self._query_api:
            return {}
        
        try:
            query = f'''
                from(bucket: "{self.config.bucket}")
                |> range(start: -{hours_back}h)
                |> filter(fn: (r) => r._measurement == "connection_metrics")
                |> filter(fn: (r) => r._field == "connection_success_rate" or r._field == "average_session_duration")
                |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
                |> yield(name: "connection_hourly")
            '''
            
            result = self._query_api.query(query=query, org=self.config.org)
            
            success_rates = []
            session_durations = []
            
            for table in result:
                for record in table.records:
                    if record.get_field() == "connection_success_rate":
                        success_rates.append(record.get_value())
                    elif record.get_field() == "average_session_duration":
                        session_durations.append(record.get_value())
            
            return {
                "average_success_rate": statistics.mean(success_rates) if success_rates else 0.0,
                "min_success_rate": min(success_rates) if success_rates else 0.0,
                "average_session_duration": statistics.mean(session_durations) if session_durations else 0.0,
                "hours_analyzed": hours_back
            }
            
        except Exception as e:
            self.logger.error(f"Failed to query connection stability: {e}")
            return {}
    
    def get_write_statistics(self) -> Dict[str, Any]:
        """Get InfluxDB write statistics"""
        return {
            "total_points_written": self.write_stats.total_points_written,
            "successful_writes": self.write_stats.successful_writes,
            "failed_writes": self.write_stats.failed_writes,
            "write_success_rate": self.write_stats.write_success_rate,
            "average_write_duration": self.write_stats.average_write_duration,
            "connection_errors": self.write_stats.connection_errors,
            "batch_queue_size": self.write_stats.batch_queue_size,
            "is_connected": self._is_connected,
            "last_write_time": self.write_stats.last_write_time
        }
    
    def flush_pending_writes(self) -> None:
        """Force flush any pending writes"""
        with self._write_lock:
            if self._write_queue:
                self._flush_write_queue()
    
    def cleanup(self) -> None:
        """Clean up InfluxDB connection and flush pending writes"""
        try:
            # Flush any pending writes
            self.flush_pending_writes()
            
            # Close write API
            if self._write_api:
                self._write_api.close()
            
            # Close client connection
            if self._client:
                self._client.close()
            
            self._is_connected = False
            self.logger.info("InfluxDB connection closed")
            
        except Exception as e:
            self.logger.error(f"Error during InfluxDB cleanup: {e}")


def create_influxdb_storage(
    influx_config: InfluxDBConfig,
    realtime_metrics: Optional[RealtimeMetricsCollector] = None,
    logger: Optional[logging.Logger] = None
) -> Optional[InfluxDBMetricsStorage]:
    """
    Factory function to create InfluxDB metrics storage.
    
    Returns None if InfluxDB client is not available.
    """
    if not HAS_INFLUXDB:
        if logger:
            logger.warning("InfluxDB client not available. Storage creation skipped.")
        return None
    
    return InfluxDBMetricsStorage(
        config=influx_config,
        realtime_metrics=realtime_metrics,
        logger=logger
    )


# Integration helper for automatic metric writing
class RealtimeMetricsWithInfluxDB:
    """
    Wrapper that automatically writes metrics to both in-memory storage
    and InfluxDB for historical persistence.
    """
    
    def __init__(self,
                 realtime_metrics: RealtimeMetricsCollector,
                 influx_storage: InfluxDBMetricsStorage):
        self.realtime_metrics = realtime_metrics
        self.influx_storage = influx_storage
        self.logger = logging.getLogger(__name__)
    
    def record_voice_latency(self, latency_ms: float, latency_type: str = "voice_to_voice_latency_ms") -> None:
        """Record latency to both in-memory and InfluxDB storage"""
        # Record to in-memory storage
        self.realtime_metrics.record_voice_latency(latency_ms, latency_type)
        
        # Write to InfluxDB
        session_id = getattr(self.realtime_metrics, '_current_session_id', 'default')
        self.influx_storage.write_latency_metrics(session_id, self.realtime_metrics.latency_metrics)
    
    def record_connection_event(self, event_type: str, success: bool = True, reason: str = "unknown") -> None:
        """Record connection event to both storages"""
        self.realtime_metrics.record_connection_event(event_type, success, reason)
        
        session_id = getattr(self.realtime_metrics, '_current_session_id', 'default')
        self.influx_storage.write_connection_metrics(session_id, self.realtime_metrics.connection_metrics)
    
    def record_audio_metrics(self, quality_score: float, volume_level: float = 0.0, event_type: str = "processing") -> None:
        """Record audio metrics to both storages"""
        self.realtime_metrics.record_audio_metrics(quality_score, volume_level, event_type)
        
        session_id = getattr(self.realtime_metrics, '_current_session_id', 'default')
        self.influx_storage.write_audio_metrics(session_id, self.realtime_metrics.audio_metrics)
    
    def record_token_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage to both storages"""
        self.realtime_metrics.record_token_usage(input_tokens, output_tokens)
        
        session_id = getattr(self.realtime_metrics, '_current_session_id', 'default')
        self.influx_storage.write_cost_metrics(session_id, self.realtime_metrics.cost_metrics)
    
    def __getattr__(self, name):
        """Delegate other method calls to the underlying realtime_metrics"""
        return getattr(self.realtime_metrics, name) 