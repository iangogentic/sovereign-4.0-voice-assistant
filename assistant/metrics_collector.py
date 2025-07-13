"""
Sovereign 4.0 Voice Assistant - Advanced Metrics Collection System

Implements comprehensive performance monitoring following 2024-2025 industry standards:
- Percentile-based latency tracking (P50, P95, P99)
- Multi-dimensional accuracy metrics with confidence scoring
- Predictive resource monitoring with trend analysis
- Thread-safe metrics aggregation with real-time updates
- Integration with error handling and configuration systems

Usage:
    collector = MetricsCollector()
    collector.start()
    
    # Track latency with automatic percentile calculation
    with collector.track_latency('stt_processing'):
        result = await stt_service.transcribe(audio)
    
    # Record accuracy metrics
    collector.record_accuracy('memory_recall', bleu_score, confidence)
    
    # Monitor resource usage
    collector.record_resource_usage()
"""

import time
import asyncio
import threading
import psutil
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import logging
import json
import GPUtil
from enum import Enum
import statistics
from contextlib import asynccontextmanager, contextmanager

# BLEU score calculation for memory recall accuracy
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Optional imports for advanced features
try:
    from sentence_transformers import util
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics for categorization and processing"""
    LATENCY = "latency"
    ACCURACY = "accuracy"
    THROUGHPUT = "throughput"
    RESOURCE = "resource"
    ERROR = "error"
    USER_EXPERIENCE = "user_experience"

class ComponentType(Enum):
    """System components for detailed tracking"""
    AUDIO_CAPTURE = "audio_capture"
    STT_PROCESSING = "stt_processing"
    LLM_ROUTING = "llm_routing"
    LLM_INFERENCE = "llm_inference" 
    MEMORY_RETRIEVAL = "memory_retrieval"
    TTS_GENERATION = "tts_generation"
    AUDIO_PLAYBACK = "audio_playback"
    OCR_PROCESSING = "ocr_processing"
    SCREEN_MONITORING = "screen_monitoring"
    ERROR_HANDLING = "error_handling"
    OVERALL_PIPELINE = "overall_pipeline"

@dataclass
class LatencyMetrics:
    """Latency measurements with percentile calculations"""
    samples: List[float] = field(default_factory=list)
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    mean: float = 0.0
    min: float = float('inf')
    max: float = 0.0
    count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def add_sample(self, latency: float) -> None:
        """Add a latency sample and update percentiles"""
        self.samples.append(latency)
        self.count += 1
        self.last_updated = datetime.now()
        
        # Keep only recent samples (last 1000 for efficiency)
        if len(self.samples) > 1000:
            self.samples = self.samples[-1000:]
        
        # Update statistics
        self.min = min(self.min, latency)
        self.max = max(self.max, latency)
        self.mean = statistics.mean(self.samples)
        
        # Calculate percentiles
        if len(self.samples) >= 2:
            sorted_samples = sorted(self.samples)
            self.p50 = np.percentile(sorted_samples, 50)
            self.p95 = np.percentile(sorted_samples, 95)
            self.p99 = np.percentile(sorted_samples, 99)

@dataclass
class AccuracyMetrics:
    """Accuracy measurements with confidence tracking"""
    scores: List[float] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    mean_score: float = 0.0
    mean_confidence: float = 0.0
    min_score: float = float('inf')
    max_score: float = 0.0
    count: int = 0
    target_threshold: float = 0.0
    success_rate: float = 0.0  # Percentage above threshold
    last_updated: datetime = field(default_factory=datetime.now)
    
    def add_measurement(self, score: float, confidence: float = 1.0, threshold: float = 0.0) -> None:
        """Add an accuracy measurement"""
        self.scores.append(score)
        self.confidences.append(confidence)
        self.count += 1
        self.target_threshold = threshold
        self.last_updated = datetime.now()
        
        # Keep only recent measurements
        if len(self.scores) > 500:
            self.scores = self.scores[-500:]
            self.confidences = self.confidences[-500:]
        
        # Update statistics
        self.mean_score = statistics.mean(self.scores)
        self.mean_confidence = statistics.mean(self.confidences)
        self.min_score = min(self.scores)
        self.max_score = max(self.scores)
        
        # Calculate success rate
        if threshold > 0:
            above_threshold = [s for s in self.scores if s >= threshold]
            self.success_rate = len(above_threshold) / len(self.scores) * 100

@dataclass
class ResourceMetrics:
    """System resource usage metrics"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_available_gb: float = 0.0
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_bytes_sent: float = 0.0
    network_bytes_recv: float = 0.0
    open_files: int = 0
    thread_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def update_system_metrics(self) -> None:
        """Update all system resource metrics"""
        self.timestamp = datetime.now()
        
        # CPU and Memory
        self.cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        self.memory_percent = memory.percent
        self.memory_used_gb = memory.used / (1024**3)
        self.memory_available_gb = memory.available / (1024**3)
        
        # GPU metrics (if available)
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                self.gpu_percent = gpu.load * 100
                self.gpu_memory_percent = gpu.memoryUtil * 100
        except Exception:
            self.gpu_percent = 0.0
            self.gpu_memory_percent = 0.0
        
        # Disk I/O
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                self.disk_io_read_mb = disk_io.read_bytes / (1024**2)
                self.disk_io_write_mb = disk_io.write_bytes / (1024**2)
        except Exception:
            pass
        
        # Network I/O
        try:
            net_io = psutil.net_io_counters()
            if net_io:
                self.network_bytes_sent = net_io.bytes_sent
                self.network_bytes_recv = net_io.bytes_recv
        except Exception:
            pass
        
        # Process metrics
        try:
            process = psutil.Process()
            self.open_files = len(process.open_files())
            self.thread_count = process.num_threads()
        except Exception:
            pass

@dataclass
class ThroughputMetrics:
    """Throughput and rate measurements"""
    requests_per_second: float = 0.0
    successful_requests: int = 0
    failed_requests: int = 0
    total_requests: int = 0
    success_rate: float = 0.0
    request_timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))
    last_updated: datetime = field(default_factory=datetime.now)
    
    def record_request(self, success: bool = True) -> None:
        """Record a request and update throughput metrics"""
        now = datetime.now()
        self.request_timestamps.append(now)
        self.total_requests += 1
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.success_rate = (self.successful_requests / self.total_requests) * 100
        self.last_updated = now
        
        # Calculate requests per second (last 60 seconds)
        cutoff_time = now - timedelta(seconds=60)
        recent_requests = [ts for ts in self.request_timestamps if ts > cutoff_time]
        self.requests_per_second = len(recent_requests) / 60.0

class MetricsCollector:
    """
    Advanced metrics collection system for Sovereign 4.0 Voice Assistant
    
    Features:
    - Thread-safe metrics aggregation
    - Percentile-based latency tracking
    - Multi-dimensional accuracy metrics
    - Resource usage monitoring with trends
    - Real-time metric updates
    - Configurable collection intervals
    """
    
    def __init__(self, collection_interval: float = 1.0, max_history_hours: int = 24):
        self.collection_interval = collection_interval
        self.max_history_hours = max_history_hours
        self.running = False
        
        # Thread-safe metrics storage
        self._lock = threading.RLock()
        self._metrics_thread: Optional[threading.Thread] = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="metrics")
        
        # Metrics collections
        self.latency_metrics: Dict[str, LatencyMetrics] = defaultdict(LatencyMetrics)
        self.accuracy_metrics: Dict[str, AccuracyMetrics] = defaultdict(AccuracyMetrics)
        self.resource_metrics: List[ResourceMetrics] = []
        self.throughput_metrics: Dict[str, ThroughputMetrics] = defaultdict(ThroughputMetrics)
        
        # Performance targets and thresholds
        self.performance_targets = {
            'voice_pipeline_latency': 800.0,  # ms - industry standard
            'stt_latency': 200.0,  # ms
            'llm_latency': 400.0,  # ms
            'tts_latency': 100.0,  # ms
            'memory_recall_bleu': 85.0,  # percentage
            'ocr_accuracy': 90.0,  # percentage
            'system_uptime': 99.9,  # percentage
            'cpu_usage_threshold': 70.0,  # percentage
            'memory_usage_threshold': 80.0,  # percentage
        }
        
        # Anomaly detection parameters
        self.anomaly_thresholds = {
            'latency_spike_multiplier': 2.0,  # Alert if latency > 2x normal
            'accuracy_drop_threshold': 10.0,  # Alert if accuracy drops > 10%
            'resource_spike_threshold': 90.0,  # Alert if resource usage > 90%
        }
        
        # Callback system for real-time alerts
        self.alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
        # Cache for expensive calculations
        self._calculation_cache = {}
        self._cache_timeout = 5.0  # seconds
        
        logger.info("MetricsCollector initialized with advanced monitoring capabilities")
    
    def start(self) -> None:
        """Start the metrics collection system"""
        if self.running:
            return
        
        self.running = True
        self._metrics_thread = threading.Thread(
            target=self._collection_loop,
            name="metrics_collector",
            daemon=True
        )
        self._metrics_thread.start()
        logger.info("🔍 Advanced metrics collection system started")
    
    def stop(self) -> None:
        """Stop the metrics collection system"""
        self.running = False
        if self._metrics_thread and self._metrics_thread.is_alive():
            self._metrics_thread.join(timeout=2.0)
        
        self._executor.shutdown(wait=True)
        logger.info("🔍 Metrics collection system stopped")
    
    def _collection_loop(self) -> None:
        """Main collection loop running in background thread"""
        while self.running:
            try:
                self._collect_resource_metrics()
                self._check_anomalies()
                self._cleanup_old_data()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_resource_metrics(self) -> None:
        """Collect system resource metrics"""
        with self._lock:
            resource_metric = ResourceMetrics()
            resource_metric.update_system_metrics()
            self.resource_metrics.append(resource_metric)
            
            # Keep only recent data
            cutoff_time = datetime.now() - timedelta(hours=self.max_history_hours)
            self.resource_metrics = [
                rm for rm in self.resource_metrics 
                if rm.timestamp > cutoff_time
            ]
    
    def _check_anomalies(self) -> None:
        """Check for performance anomalies and trigger alerts"""
        try:
            # Check latency anomalies
            for component, metrics in self.latency_metrics.items():
                if metrics.count > 10:  # Need sufficient data
                    if metrics.p95 > metrics.mean * self.anomaly_thresholds['latency_spike_multiplier']:
                        self._trigger_alert(
                            'latency_anomaly',
                            {
                                'component': component,
                                'p95_latency': metrics.p95,
                                'mean_latency': metrics.mean,
                                'threshold_multiplier': self.anomaly_thresholds['latency_spike_multiplier']
                            }
                        )
            
            # Check accuracy drops
            for metric_name, metrics in self.accuracy_metrics.items():
                if metrics.count > 5:
                    recent_scores = metrics.scores[-5:]  # Last 5 measurements
                    if len(recent_scores) >= 5:
                        recent_mean = statistics.mean(recent_scores)
                        overall_mean = metrics.mean_score
                        
                        if (overall_mean - recent_mean) > self.anomaly_thresholds['accuracy_drop_threshold']:
                            self._trigger_alert(
                                'accuracy_degradation',
                                {
                                    'metric': metric_name,
                                    'recent_accuracy': recent_mean,
                                    'overall_accuracy': overall_mean,
                                    'drop_amount': overall_mean - recent_mean
                                }
                            )
            
            # Check resource usage spikes
            if self.resource_metrics:
                latest_resource = self.resource_metrics[-1]
                
                if latest_resource.cpu_percent > self.anomaly_thresholds['resource_spike_threshold']:
                    self._trigger_alert(
                        'high_cpu_usage',
                        {
                            'cpu_percent': latest_resource.cpu_percent,
                            'threshold': self.anomaly_thresholds['resource_spike_threshold']
                        }
                    )
                
                if latest_resource.memory_percent > self.anomaly_thresholds['resource_spike_threshold']:
                    self._trigger_alert(
                        'high_memory_usage',
                        {
                            'memory_percent': latest_resource.memory_percent,
                            'memory_used_gb': latest_resource.memory_used_gb,
                            'threshold': self.anomaly_thresholds['resource_spike_threshold']
                        }
                    )
                    
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
    
    def _trigger_alert(self, alert_type: str, data: Dict[str, Any]) -> None:
        """Trigger an alert and notify registered callbacks"""
        alert_data = {
            'type': alert_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        logger.warning(f"🚨 Performance Alert: {alert_type} - {data}")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, alert_data)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def _cleanup_old_data(self) -> None:
        """Clean up old metric data to prevent memory leaks"""
        cutoff_time = datetime.now() - timedelta(hours=self.max_history_hours)
        
        with self._lock:
            # Clean latency metrics (keep samples but limit them)
            for metrics in self.latency_metrics.values():
                if len(metrics.samples) > 1000:
                    metrics.samples = metrics.samples[-1000:]
            
            # Clean accuracy metrics
            for metrics in self.accuracy_metrics.values():
                if len(metrics.scores) > 500:
                    metrics.scores = metrics.scores[-500:]
                    metrics.confidences = metrics.confidences[-500:]
            
            # Clean throughput metrics
            for metrics in self.throughput_metrics.values():
                # Deque already has maxlen, but clean old timestamps
                if metrics.request_timestamps:
                    recent_timestamps = [
                        ts for ts in metrics.request_timestamps 
                        if ts > cutoff_time
                    ]
                    metrics.request_timestamps.clear()
                    metrics.request_timestamps.extend(recent_timestamps)
    
    # Latency Tracking Methods
    @contextmanager
    def track_latency(self, component: Union[str, ComponentType]):
        """Context manager for tracking latency of operations"""
        component_name = component.value if isinstance(component, ComponentType) else component
        start_time = time.time()
        
        try:
            yield
        finally:
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            self.record_latency(component_name, latency_ms)
    
    @asynccontextmanager
    async def track_async_latency(self, component: Union[str, ComponentType]):
        """Async context manager for tracking latency of async operations"""
        component_name = component.value if isinstance(component, ComponentType) else component
        start_time = time.time()
        
        try:
            yield
        finally:
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            self.record_latency(component_name, latency_ms)
    
    def record_latency(self, component: str, latency_ms: float) -> None:
        """Record a latency measurement"""
        with self._lock:
            self.latency_metrics[component].add_sample(latency_ms)
            
            # Check against performance targets
            target = self.performance_targets.get(f"{component}_latency")
            if target and latency_ms > target:
                logger.warning(
                    f"⚠️ Latency target exceeded: {component} = {latency_ms:.2f}ms "
                    f"(target: {target}ms)"
                )
    
    # Accuracy Tracking Methods
    def record_accuracy(
        self, 
        metric_name: str, 
        score: float, 
        confidence: float = 1.0,
        threshold: float = 0.0
    ) -> None:
        """Record an accuracy measurement"""
        with self._lock:
            self.accuracy_metrics[metric_name].add_measurement(score, confidence, threshold)
            
            # Check against performance targets
            target = self.performance_targets.get(metric_name)
            if target and score < target:
                logger.warning(
                    f"⚠️ Accuracy target not met: {metric_name} = {score:.2f} "
                    f"(target: {target})"
                )
    
    def calculate_bleu_score(
        self, 
        reference: str, 
        candidate: str, 
        smoothing: bool = True
    ) -> float:
        """Calculate BLEU score for memory recall accuracy"""
        try:
            # Tokenize sentences
            reference_tokens = reference.lower().split()
            candidate_tokens = candidate.lower().split()
            
            # Use smoothing to handle edge cases
            if smoothing:
                smoothing_function = SmoothingFunction().method1
                score = sentence_bleu(
                    [reference_tokens], 
                    candidate_tokens,
                    smoothing_function=smoothing_function
                )
            else:
                score = sentence_bleu([reference_tokens], candidate_tokens)
            
            return score * 100  # Convert to percentage
            
        except Exception as e:
            logger.error(f"Error calculating BLEU score: {e}")
            return 0.0
    
    def calculate_semantic_similarity(
        self, 
        text1: str, 
        text2: str, 
        model_name: str = "all-MiniLM-L6-v2"
    ) -> float:
        """Calculate semantic similarity using sentence transformers or word overlap"""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                # Use sentence transformers for better semantic similarity
                # Note: This is a placeholder - full implementation would require
                # loading the model and computing embeddings
                pass
            
            # Fallback to simple word overlap score
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union) * 100
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    # Throughput Tracking Methods
    def record_request(self, component: str, success: bool = True) -> None:
        """Record a request for throughput tracking"""
        with self._lock:
            self.throughput_metrics[component].record_request(success)
    
    # Resource Monitoring Methods
    def get_current_resource_usage(self) -> Optional[ResourceMetrics]:
        """Get the most recent resource usage metrics"""
        with self._lock:
            if self.resource_metrics:
                return self.resource_metrics[-1]
            return None
    
    def get_resource_trends(self, hours: int = 1) -> Dict[str, List[float]]:
        """Get resource usage trends over specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            recent_metrics = [
                rm for rm in self.resource_metrics 
                if rm.timestamp > cutoff_time
            ]
            
            return {
                'cpu_percent': [rm.cpu_percent for rm in recent_metrics],
                'memory_percent': [rm.memory_percent for rm in recent_metrics],
                'gpu_percent': [rm.gpu_percent for rm in recent_metrics],
                'timestamps': [rm.timestamp.isoformat() for rm in recent_metrics]
            }
    
    # Data Export Methods
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        with self._lock:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'latency_metrics': {},
                'accuracy_metrics': {},
                'throughput_metrics': {},
                'resource_metrics': {},
                'performance_targets': self.performance_targets,
                'target_compliance': {}
            }
            
            # Latency summary
            for component, metrics in self.latency_metrics.items():
                if metrics.count > 0:
                    summary['latency_metrics'][component] = {
                        'p50': metrics.p50,
                        'p95': metrics.p95,
                        'p99': metrics.p99,
                        'mean': metrics.mean,
                        'min': metrics.min,
                        'max': metrics.max,
                        'count': metrics.count,
                        'last_updated': metrics.last_updated.isoformat()
                    }
                    
                    # Check target compliance
                    target = self.performance_targets.get(f"{component}_latency")
                    if target:
                        compliance = (metrics.p95 <= target)
                        summary['target_compliance'][f"{component}_latency"] = {
                            'target': target,
                            'actual_p95': metrics.p95,
                            'compliant': compliance
                        }
            
            # Accuracy summary
            for metric_name, metrics in self.accuracy_metrics.items():
                if metrics.count > 0:
                    summary['accuracy_metrics'][metric_name] = {
                        'mean_score': metrics.mean_score,
                        'mean_confidence': metrics.mean_confidence,
                        'min_score': metrics.min_score,
                        'max_score': metrics.max_score,
                        'success_rate': metrics.success_rate,
                        'count': metrics.count,
                        'last_updated': metrics.last_updated.isoformat()
                    }
                    
                    # Check target compliance
                    target = self.performance_targets.get(metric_name)
                    if target:
                        compliance = (metrics.mean_score >= target)
                        summary['target_compliance'][metric_name] = {
                            'target': target,
                            'actual': metrics.mean_score,
                            'compliant': compliance
                        }
            
            # Throughput summary
            for component, metrics in self.throughput_metrics.items():
                if metrics.total_requests > 0:
                    summary['throughput_metrics'][component] = {
                        'requests_per_second': metrics.requests_per_second,
                        'success_rate': metrics.success_rate,
                        'total_requests': metrics.total_requests,
                        'successful_requests': metrics.successful_requests,
                        'failed_requests': metrics.failed_requests,
                        'last_updated': metrics.last_updated.isoformat()
                    }
            
            # Resource summary
            if self.resource_metrics:
                latest_resource = self.resource_metrics[-1]
                summary['resource_metrics'] = {
                    'cpu_percent': latest_resource.cpu_percent,
                    'memory_percent': latest_resource.memory_percent,
                    'memory_used_gb': latest_resource.memory_used_gb,
                    'gpu_percent': latest_resource.gpu_percent,
                    'gpu_memory_percent': latest_resource.gpu_memory_percent,
                    'thread_count': latest_resource.thread_count,
                    'timestamp': latest_resource.timestamp.isoformat()
                }
                
                # Check resource target compliance
                summary['target_compliance']['cpu_usage'] = {
                    'target': self.performance_targets['cpu_usage_threshold'],
                    'actual': latest_resource.cpu_percent,
                    'compliant': latest_resource.cpu_percent <= self.performance_targets['cpu_usage_threshold']
                }
                
                summary['target_compliance']['memory_usage'] = {
                    'target': self.performance_targets['memory_usage_threshold'],
                    'actual': latest_resource.memory_percent,
                    'compliant': latest_resource.memory_percent <= self.performance_targets['memory_usage_threshold']
                }
            
            return summary
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add a callback for performance alerts"""
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Remove an alert callback"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)

# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None

def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector

def set_metrics_collector(collector: MetricsCollector) -> None:
    """Set the global metrics collector instance"""
    global _metrics_collector
    _metrics_collector = collector

# Export main components
__all__ = [
    'MetricsCollector',
    'MetricType',
    'ComponentType', 
    'LatencyMetrics',
    'AccuracyMetrics',
    'ResourceMetrics',
    'ThroughputMetrics',
    'get_metrics_collector',
    'set_metrics_collector'
] 