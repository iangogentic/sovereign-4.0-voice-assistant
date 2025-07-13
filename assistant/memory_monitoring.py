"""
Memory Leak Detection and Monitoring Suite for Sovereign 4.0 Voice Assistant

This module provides comprehensive memory monitoring capabilities including:
- Continuous memory leak detection with ML-based anomaly detection
- GPU memory monitoring for AI workloads
- Python object tracking and profiling
- Resource exhaustion prevention
- Memory usage trend analysis

Implements modern 2024-2025 memory monitoring best practices.
"""

import asyncio
import time
import logging
import psutil
import tracemalloc
import gc
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json
import sys
from pathlib import Path

# ML libraries for anomaly detection
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    import joblib
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logging.warning("scikit-learn not available. ML-based memory anomaly detection disabled.")

# GPU monitoring
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .performance_testing import TestResult, PerformanceTestConfig

logger = logging.getLogger(__name__)

@dataclass
class MemorySnapshot:
    """Memory usage snapshot at a point in time"""
    timestamp: datetime
    rss_mb: float
    vms_mb: float
    percent: float
    available_mb: float
    gpu_allocated_mb: Optional[float] = None
    gpu_reserved_mb: Optional[float] = None
    python_objects: Optional[Dict[str, int]] = None
    top_allocations: Optional[List[Tuple[str, int]]] = None

@dataclass
class MemoryLeak:
    """Detected memory leak information"""
    leak_id: str
    detection_time: datetime
    leak_rate_mb_per_minute: float
    confidence_score: float
    affected_components: List[str]
    growth_pattern: str  # 'linear', 'exponential', 'step'
    severity: str  # 'low', 'medium', 'high', 'critical'

class MemoryLeakDetector:
    """
    Advanced memory leak detector with ML-based anomaly detection
    
    Provides continuous monitoring, trend analysis, and early warning
    for memory leaks in the voice assistant system.
    """
    
    def __init__(self, config: PerformanceTestConfig):
        self.config = config
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Memory tracking
        self.memory_history: deque = deque(maxlen=2000)  # Keep 2000 samples
        self.memory_snapshots: List[MemorySnapshot] = []
        self.detected_leaks: List[MemoryLeak] = []
        
        # ML-based anomaly detection
        self.anomaly_detector: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.baseline_trained = False
        
        # Thresholds
        self.leak_threshold_percent = config.stress_test_config.get('memory_leak_threshold_percent', 10)
        self.monitoring_interval = 30  # seconds
        self.trend_analysis_window = 10  # number of samples for trend analysis
        
        # Python object tracking
        self.object_tracker = PythonObjectTracker()
        
        logger.info("Memory Leak Detector initialized")
    
    async def start_monitoring(self) -> None:
        """Start continuous memory monitoring"""
        if self.is_monitoring:
            logger.warning("Memory monitoring already active")
            return
        
        self.is_monitoring = True
        logger.info("Starting continuous memory monitoring")
        
        # Start tracemalloc for Python object tracking
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        
        # Start monitoring in background thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="memory_monitor",
            daemon=True
        )
        self.monitoring_thread.start()
    
    async def stop_monitoring(self) -> None:
        """Stop continuous memory monitoring"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        logger.info("Stopping memory monitoring")
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
    
    async def stop_monitoring_and_analyze(self) -> List[TestResult]:
        """Stop monitoring and return analysis results"""
        await self.stop_monitoring()
        
        # Analyze collected data
        results = []
        
        # Memory leak detection
        results.append(await self.analyze_memory_leaks())
        
        # Memory usage analysis
        results.append(await self.analyze_memory_usage_patterns())
        
        # Python object growth analysis
        results.append(await self.analyze_python_object_growth())
        
        # GPU memory analysis (if available)
        if HAS_TORCH and torch.cuda.is_available():
            results.append(await self.analyze_gpu_memory_usage())
        
        return results
    
    async def continuous_monitoring(self, duration_seconds: int) -> List[TestResult]:
        """Run continuous monitoring for specified duration"""
        await self.start_monitoring()
        
        try:
            # Monitor for specified duration
            await asyncio.sleep(duration_seconds)
        finally:
            # Analyze results
            results = await self.stop_monitoring_and_analyze()
        
        return results
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop (runs in background thread)"""
        while self.is_monitoring:
            try:
                # Collect memory snapshot
                snapshot = self._collect_memory_snapshot()
                self.memory_snapshots.append(snapshot)
                self.memory_history.append((snapshot.timestamp, snapshot.rss_mb))
                
                # Detect leaks in real-time
                self._detect_memory_leaks_realtime(snapshot)
                
                # Train or update anomaly detection model
                if len(self.memory_history) >= 50 and not self.baseline_trained:
                    self._train_baseline_model()
                elif len(self.memory_history) >= 100 and self.baseline_trained:
                    self._detect_anomalies(snapshot)
                
                # Check for memory pressure
                self._check_memory_pressure(snapshot)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_memory_snapshot(self) -> MemorySnapshot:
        """Collect comprehensive memory usage snapshot"""
        # System memory
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        system_memory = psutil.virtual_memory()
        
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            rss_mb=memory_info.rss / 1024**2,
            vms_mb=memory_info.vms / 1024**2,
            percent=memory_percent,
            available_mb=system_memory.available / 1024**2
        )
        
        # GPU memory (if available)
        if HAS_TORCH and torch.cuda.is_available():
            try:
                snapshot.gpu_allocated_mb = torch.cuda.memory_allocated() / 1024**2
                snapshot.gpu_reserved_mb = torch.cuda.memory_reserved() / 1024**2
            except Exception as e:
                logger.debug(f"Could not collect GPU memory: {e}")
        
        # Python object tracking
        if tracemalloc.is_tracing():
            try:
                current_trace = tracemalloc.take_snapshot()
                top_stats = current_trace.statistics('lineno')
                
                # Get top allocations
                top_allocations = [
                    (str(stat.traceback), stat.size)
                    for stat in top_stats[:10]
                ]
                snapshot.top_allocations = top_allocations
                
                # Count objects by type
                snapshot.python_objects = self.object_tracker.count_objects()
                
            except Exception as e:
                logger.debug(f"Could not collect Python object stats: {e}")
        
        return snapshot
    
    def _detect_memory_leaks_realtime(self, snapshot: MemorySnapshot) -> None:
        """Detect memory leaks in real-time using trend analysis"""
        if len(self.memory_history) < self.trend_analysis_window:
            return
        
        # Get recent memory usage
        recent_data = list(self.memory_history)[-self.trend_analysis_window:]
        timestamps = [data[0] for data in recent_data]
        memory_values = [data[1] for data in recent_data]
        
        # Calculate trend
        time_diffs = [(t - timestamps[0]).total_seconds() / 60 for t in timestamps]  # minutes
        
        if len(time_diffs) >= 3:
            # Linear regression for trend
            trend_slope = np.polyfit(time_diffs, memory_values, 1)[0]  # MB per minute
            
            # Detect leak based on growth rate
            if trend_slope > 1.0:  # Growing by more than 1MB per minute
                confidence = min(1.0, trend_slope / 5.0)  # Confidence based on growth rate
                
                # Determine severity
                if trend_slope > 10:
                    severity = 'critical'
                elif trend_slope > 5:
                    severity = 'high'
                elif trend_slope > 2:
                    severity = 'medium'
                else:
                    severity = 'low'
                
                # Check if this is a new leak or existing one
                recent_leaks = [leak for leak in self.detected_leaks if 
                              (datetime.now() - leak.detection_time).total_seconds() < 300]
                
                if not recent_leaks or all(abs(leak.leak_rate_mb_per_minute - trend_slope) > 1.0 for leak in recent_leaks):
                    # New leak detected
                    leak = MemoryLeak(
                        leak_id=f"leak_{len(self.detected_leaks) + 1}",
                        detection_time=datetime.now(),
                        leak_rate_mb_per_minute=trend_slope,
                        confidence_score=confidence,
                        affected_components=self._identify_affected_components(snapshot),
                        growth_pattern=self._classify_growth_pattern(memory_values),
                        severity=severity
                    )
                    
                    self.detected_leaks.append(leak)
                    logger.warning(f"Memory leak detected: {leak.leak_rate_mb_per_minute:.2f} MB/min, severity: {leak.severity}")
    
    def _train_baseline_model(self) -> None:
        """Train ML baseline model for anomaly detection"""
        if not HAS_SKLEARN:
            return
        
        try:
            # Prepare training data
            memory_data = np.array([data[1] for data in self.memory_history])
            
            # Calculate features
            features = []
            for i in range(5, len(memory_data)):  # Need at least 5 points for features
                window = memory_data[i-5:i]
                feature_vector = [
                    np.mean(window),
                    np.std(window),
                    np.max(window) - np.min(window),
                    np.polyfit(range(len(window)), window, 1)[0],  # trend
                    memory_data[i]  # current value
                ]
                features.append(feature_vector)
            
            if len(features) < 10:
                return
            
            features_array = np.array(features)
            
            # Train models
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features_array)
            
            self.anomaly_detector = IsolationForest(
                contamination=0.1,  # Expect 10% anomalies
                random_state=42
            )
            self.anomaly_detector.fit(features_scaled)
            
            self.baseline_trained = True
            logger.info("Memory anomaly detection baseline model trained")
            
        except Exception as e:
            logger.error(f"Failed to train memory baseline model: {e}")
    
    def _detect_anomalies(self, snapshot: MemorySnapshot) -> None:
        """Detect memory usage anomalies using ML model"""
        if not self.baseline_trained or not HAS_SKLEARN:
            return
        
        try:
            # Prepare current features
            memory_data = np.array([data[1] for data in list(self.memory_history)[-10:]])
            
            if len(memory_data) < 5:
                return
            
            window = memory_data[-5:]
            feature_vector = np.array([[
                np.mean(window),
                np.std(window),
                np.max(window) - np.min(window),
                np.polyfit(range(len(window)), window, 1)[0],
                memory_data[-1]
            ]])
            
            # Scale features
            features_scaled = self.scaler.transform(feature_vector)
            
            # Detect anomaly
            anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
            is_anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1
            
            if is_anomaly:
                logger.warning(f"Memory usage anomaly detected (score: {anomaly_score:.3f})")
                
                # Create anomaly-based leak detection
                leak = MemoryLeak(
                    leak_id=f"anomaly_{int(time.time())}",
                    detection_time=datetime.now(),
                    leak_rate_mb_per_minute=0,  # Will be calculated
                    confidence_score=abs(anomaly_score),
                    affected_components=self._identify_affected_components(snapshot),
                    growth_pattern='anomalous',
                    severity='medium'
                )
                
                self.detected_leaks.append(leak)
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
    
    def _check_memory_pressure(self, snapshot: MemorySnapshot) -> None:
        """Check for memory pressure conditions"""
        # System memory pressure
        if snapshot.percent > 90:
            logger.critical(f"Critical memory usage: {snapshot.percent:.1f}%")
        elif snapshot.percent > 80:
            logger.warning(f"High memory usage: {snapshot.percent:.1f}%")
        
        # Available memory pressure
        if snapshot.available_mb < 500:  # Less than 500MB available
            logger.critical(f"Low available memory: {snapshot.available_mb:.1f}MB")
        
        # GPU memory pressure (if applicable)
        if snapshot.gpu_allocated_mb and snapshot.gpu_allocated_mb > 1000:  # > 1GB GPU usage
            logger.info(f"GPU memory usage: {snapshot.gpu_allocated_mb:.1f}MB")
    
    def _identify_affected_components(self, snapshot: MemorySnapshot) -> List[str]:
        """Identify components that might be causing memory issues"""
        components = []
        
        if snapshot.top_allocations:
            # Analyze top allocations for patterns
            for allocation, size in snapshot.top_allocations[:5]:
                if 'torch' in allocation.lower():
                    components.append('pytorch')
                elif 'whisper' in allocation.lower() or 'stt' in allocation.lower():
                    components.append('speech_to_text')
                elif 'tts' in allocation.lower() or 'openai' in allocation.lower():
                    components.append('text_to_speech')
                elif 'llm' in allocation.lower() or 'router' in allocation.lower():
                    components.append('llm_processing')
                elif 'audio' in allocation.lower():
                    components.append('audio_processing')
        
        # If no specific components identified, add general categories
        if not components:
            if snapshot.gpu_allocated_mb and snapshot.gpu_allocated_mb > 100:
                components.append('gpu_processing')
            components.append('general_memory')
        
        return list(set(components))  # Remove duplicates
    
    def _classify_growth_pattern(self, memory_values: List[float]) -> str:
        """Classify memory growth pattern"""
        if len(memory_values) < 3:
            return 'insufficient_data'
        
        # Calculate differences between consecutive values
        diffs = np.diff(memory_values)
        
        # Check for patterns
        if all(d >= 0 for d in diffs):
            # All increasing
            if np.std(diffs) < np.mean(diffs) * 0.1:
                return 'linear'
            else:
                return 'exponential'
        elif all(d <= 0 for d in diffs):
            return 'decreasing'
        elif np.std(diffs) > np.mean(np.abs(diffs)) * 2:
            return 'volatile'
        else:
            return 'step'
    
    async def analyze_memory_leaks(self) -> TestResult:
        """Analyze detected memory leaks"""
        start_time = time.time()
        test_name = "memory_leak_analysis"
        
        try:
            # Calculate memory growth over entire monitoring period
            if len(self.memory_history) < 2:
                return TestResult(
                    test_name=test_name,
                    status="skipped",
                    metrics={},
                    execution_time=time.time() - start_time,
                    timestamp=datetime.now(),
                    details="Insufficient memory data"
                )
            
            initial_memory = self.memory_history[0][1]
            final_memory = self.memory_history[-1][1]
            memory_growth_mb = final_memory - initial_memory
            memory_growth_percent = (memory_growth_mb / initial_memory) * 100 if initial_memory > 0 else 0
            
            # Analyze leak patterns
            leak_metrics = {
                'total_memory_growth_mb': memory_growth_mb,
                'memory_growth_percent': memory_growth_percent,
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'peak_memory_mb': max(data[1] for data in self.memory_history),
                'leaks_detected': len(self.detected_leaks),
                'monitoring_duration_minutes': len(self.memory_history) * (self.monitoring_interval / 60)
            }
            
            # Leak severity analysis
            if self.detected_leaks:
                severity_counts = {}
                for leak in self.detected_leaks:
                    severity_counts[leak.severity] = severity_counts.get(leak.severity, 0) + 1
                
                leak_metrics.update({
                    'critical_leaks': severity_counts.get('critical', 0),
                    'high_leaks': severity_counts.get('high', 0),
                    'medium_leaks': severity_counts.get('medium', 0),
                    'low_leaks': severity_counts.get('low', 0),
                    'max_leak_rate_mb_per_min': max(leak.leak_rate_mb_per_minute for leak in self.detected_leaks),
                    'avg_leak_confidence': np.mean([leak.confidence_score for leak in self.detected_leaks])
                })
            
            # Determine test status
            status = 'passed'
            if memory_growth_percent > self.leak_threshold_percent:
                status = 'failed'
            elif len(self.detected_leaks) > 0 and any(leak.severity in ['critical', 'high'] for leak in self.detected_leaks):
                status = 'warning'
            
            details = f"Memory growth: {memory_growth_percent:.1f}%, Leaks detected: {len(self.detected_leaks)}"
            if self.detected_leaks:
                critical_leaks = sum(1 for leak in self.detected_leaks if leak.severity == 'critical')
                if critical_leaks > 0:
                    details += f", Critical leaks: {critical_leaks}"
            
            return TestResult(
                test_name=test_name,
                status=status,
                metrics=leak_metrics,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                details=details
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_name,
                status="failed",
                metrics={},
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                errors=[str(e)]
            )
    
    async def analyze_memory_usage_patterns(self) -> TestResult:
        """Analyze memory usage patterns and trends"""
        start_time = time.time()
        test_name = "memory_usage_patterns"
        
        try:
            if len(self.memory_snapshots) < 10:
                return TestResult(
                    test_name=test_name,
                    status="skipped",
                    metrics={},
                    execution_time=time.time() - start_time,
                    timestamp=datetime.now(),
                    details="Insufficient data for pattern analysis"
                )
            
            # Extract memory values
            rss_values = [s.rss_mb for s in self.memory_snapshots]
            percent_values = [s.percent for s in self.memory_snapshots]
            
            # Statistical analysis
            pattern_metrics = {
                'mean_memory_mb': np.mean(rss_values),
                'median_memory_mb': np.median(rss_values),
                'std_memory_mb': np.std(rss_values),
                'min_memory_mb': np.min(rss_values),
                'max_memory_mb': np.max(rss_values),
                'memory_volatility': np.std(rss_values) / np.mean(rss_values) if np.mean(rss_values) > 0 else 0,
                'mean_memory_percent': np.mean(percent_values),
                'max_memory_percent': np.max(percent_values)
            }
            
            # Trend analysis
            time_points = range(len(rss_values))
            trend_slope = np.polyfit(time_points, rss_values, 1)[0]
            pattern_metrics['trend_slope_mb_per_sample'] = trend_slope
            
            # Pattern classification
            if abs(trend_slope) < 0.1:
                memory_pattern = 'stable'
            elif trend_slope > 0.5:
                memory_pattern = 'increasing'
            elif trend_slope < -0.5:
                memory_pattern = 'decreasing'
            else:
                memory_pattern = 'gradual_change'
            
            pattern_metrics['memory_pattern'] = memory_pattern
            
            # Determine status
            status = 'passed'
            if pattern_metrics['max_memory_percent'] > 95:
                status = 'failed'
            elif pattern_metrics['memory_volatility'] > 0.3 or pattern_metrics['max_memory_percent'] > 85:
                status = 'warning'
            
            return TestResult(
                test_name=test_name,
                status=status,
                metrics=pattern_metrics,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                details=f"Pattern: {memory_pattern}, Volatility: {pattern_metrics['memory_volatility']:.3f}, Peak: {pattern_metrics['max_memory_percent']:.1f}%"
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_name,
                status="failed",
                metrics={},
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                errors=[str(e)]
            )
    
    async def analyze_python_object_growth(self) -> TestResult:
        """Analyze Python object growth patterns"""
        start_time = time.time()
        test_name = "python_object_growth"
        
        try:
            if not tracemalloc.is_tracing():
                return TestResult(
                    test_name=test_name,
                    status="skipped",
                    metrics={},
                    execution_time=time.time() - start_time,
                    timestamp=datetime.now(),
                    details="tracemalloc not active"
                )
            
            # Get current snapshot
            current_snapshot = tracemalloc.take_snapshot()
            top_stats = current_snapshot.statistics('lineno')
            
            # Analyze object counts
            object_counts = self.object_tracker.count_objects()
            
            object_metrics = {
                'total_python_objects': sum(object_counts.values()),
                'unique_object_types': len(object_counts),
                'top_object_type': max(object_counts.items(), key=lambda x: x[1])[0] if object_counts else 'none',
                'top_object_count': max(object_counts.values()) if object_counts else 0,
                'memory_allocations_tracked': len(top_stats),
                'total_allocated_size_mb': sum(stat.size for stat in top_stats) / 1024**2
            }
            
            # Check for problematic patterns
            problematic_types = []
            for obj_type, count in object_counts.items():
                if count > 10000:  # More than 10k objects of same type
                    problematic_types.append(f"{obj_type}: {count}")
            
            object_metrics['problematic_object_types'] = len(problematic_types)
            
            # Determine status
            status = 'passed'
            if object_metrics['problematic_object_types'] > 0:
                status = 'warning'
            if object_metrics['total_allocated_size_mb'] > 1000:  # > 1GB allocated
                status = 'warning'
            
            details = f"Objects: {object_metrics['total_python_objects']}, Allocated: {object_metrics['total_allocated_size_mb']:.1f}MB"
            if problematic_types:
                details += f", Issues: {problematic_types[:3]}"
            
            return TestResult(
                test_name=test_name,
                status=status,
                metrics=object_metrics,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                details=details
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_name,
                status="failed",
                metrics={},
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                errors=[str(e)]
            )
    
    async def analyze_gpu_memory_usage(self) -> TestResult:
        """Analyze GPU memory usage patterns"""
        start_time = time.time()
        test_name = "gpu_memory_analysis"
        
        try:
            if not HAS_TORCH or not torch.cuda.is_available():
                return TestResult(
                    test_name=test_name,
                    status="skipped",
                    metrics={},
                    execution_time=time.time() - start_time,
                    timestamp=datetime.now(),
                    details="GPU not available"
                )
            
            # Get GPU memory snapshots
            gpu_snapshots = [s for s in self.memory_snapshots if s.gpu_allocated_mb is not None]
            
            if len(gpu_snapshots) < 5:
                return TestResult(
                    test_name=test_name,
                    status="skipped",
                    metrics={},
                    execution_time=time.time() - start_time,
                    timestamp=datetime.now(),
                    details="Insufficient GPU memory data"
                )
            
            allocated_values = [s.gpu_allocated_mb for s in gpu_snapshots]
            reserved_values = [s.gpu_reserved_mb for s in gpu_snapshots if s.gpu_reserved_mb is not None]
            
            gpu_metrics = {
                'mean_gpu_allocated_mb': np.mean(allocated_values),
                'max_gpu_allocated_mb': np.max(allocated_values),
                'min_gpu_allocated_mb': np.min(allocated_values),
                'gpu_memory_volatility': np.std(allocated_values) / np.mean(allocated_values) if np.mean(allocated_values) > 0 else 0,
                'current_gpu_allocated_mb': allocated_values[-1],
                'gpu_total_memory_mb': torch.cuda.get_device_properties(0).total_memory / 1024**2
            }
            
            if reserved_values:
                gpu_metrics.update({
                    'mean_gpu_reserved_mb': np.mean(reserved_values),
                    'max_gpu_reserved_mb': np.max(reserved_values)
                })
            
            # GPU utilization analysis
            gpu_utilization = (gpu_metrics['max_gpu_allocated_mb'] / gpu_metrics['gpu_total_memory_mb']) * 100
            gpu_metrics['peak_gpu_utilization_percent'] = gpu_utilization
            
            # Determine status
            status = 'passed'
            if gpu_utilization > 90:
                status = 'warning'
            elif gpu_metrics['gpu_memory_volatility'] > 0.5:
                status = 'warning'
            
            return TestResult(
                test_name=test_name,
                status=status,
                metrics=gpu_metrics,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                details=f"Peak GPU: {gpu_utilization:.1f}%, Current: {gpu_metrics['current_gpu_allocated_mb']:.1f}MB"
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_name,
                status="failed",
                metrics={},
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                errors=[str(e)]
            )


class PythonObjectTracker:
    """Track Python object creation and deletion patterns"""
    
    def __init__(self):
        self.object_history: List[Dict[str, int]] = []
        self.tracking_enabled = True
    
    def count_objects(self) -> Dict[str, int]:
        """Count objects by type"""
        if not self.tracking_enabled:
            return {}
        
        try:
            # Force garbage collection before counting
            gc.collect()
            
            # Count objects by type
            object_counts = defaultdict(int)
            
            for obj in gc.get_objects():
                obj_type = type(obj).__name__
                object_counts[obj_type] += 1
            
            # Store in history
            current_counts = dict(object_counts)
            self.object_history.append(current_counts)
            
            # Keep only last 100 snapshots
            if len(self.object_history) > 100:
                self.object_history = self.object_history[-100:]
            
            return current_counts
            
        except Exception as e:
            logger.error(f"Error counting objects: {e}")
            return {}
    
    def get_object_growth_rate(self, object_type: str) -> float:
        """Get growth rate for specific object type"""
        if len(self.object_history) < 2:
            return 0.0
        
        recent_counts = [
            snapshot.get(object_type, 0) 
            for snapshot in self.object_history[-10:]
        ]
        
        if len(recent_counts) < 2:
            return 0.0
        
        # Calculate linear growth rate
        time_points = list(range(len(recent_counts)))
        slope = np.polyfit(time_points, recent_counts, 1)[0]
        
        return slope


def create_memory_leak_detector(config: Optional[PerformanceTestConfig] = None) -> MemoryLeakDetector:
    """Factory function to create memory leak detector"""
    if config is None:
        config = PerformanceTestConfig()
    return MemoryLeakDetector(config) 