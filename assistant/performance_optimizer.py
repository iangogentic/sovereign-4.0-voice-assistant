"""
Central Performance Optimizer for Sovereign 4.0
Coordinates all optimization strategies for sub-300ms response times
Integrates connection pooling, audio optimization, context pre-loading, and intelligent resource allocation
"""

import asyncio
import time
import logging
import json
import psutil
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import statistics
import pickle
import hashlib
from pathlib import Path

# Import our optimization components
from .connection_pool_manager import ConnectionPoolManager, ConnectionPoolConfig, create_connection_pool_manager
from .audio_optimizer import AudioStreamOptimizer, AudioConfig, ProcessingMode, create_audio_optimizer


class OptimizationLevel(Enum):
    """Performance optimization levels"""
    MINIMAL = "minimal"           # Basic optimizations only
    STANDARD = "standard"         # Balanced optimization 
    AGGRESSIVE = "aggressive"     # Maximum optimization
    ADAPTIVE = "adaptive"         # AI-driven optimization


class ResourceType(Enum):
    """Types of system resources to monitor"""
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    DISK = "disk"
    AUDIO = "audio"


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization"""
    # Optimization settings
    optimization_level: OptimizationLevel = OptimizationLevel.ADAPTIVE
    target_response_time_ms: float = 250.0  # Target under 300ms
    max_acceptable_latency_ms: float = 300.0
    
    # Context management
    context_cache_size: int = 1000
    context_preload_enabled: bool = True
    predictive_loading: bool = True
    context_compression: bool = True
    
    # Resource allocation
    max_cpu_usage: float = 0.8  # 80% max CPU
    max_memory_usage: float = 0.7  # 70% max memory
    adaptive_resource_scaling: bool = True
    resource_monitoring_interval: float = 1.0  # seconds
    
    # Predictive features
    pattern_learning: bool = True
    conversation_prediction: bool = True
    user_behavior_analysis: bool = True
    
    # Performance monitoring
    metrics_collection_enabled: bool = True
    performance_profiling: bool = True
    latency_precision_monitoring: bool = True


@dataclass
class SystemMetrics:
    """System resource metrics"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_available: float = 0.0
    network_latency: float = 0.0
    disk_io_rate: float = 0.0
    audio_buffer_health: float = 100.0
    
    # Performance indicators
    response_time_avg: float = 0.0
    response_time_p95: float = 0.0
    response_time_p99: float = 0.0
    error_rate: float = 0.0
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_resource_pressure(self) -> float:
        """Calculate overall resource pressure (0-1)"""
        cpu_pressure = min(self.cpu_usage, 1.0)
        memory_pressure = min(self.memory_usage, 1.0)
        return (cpu_pressure + memory_pressure) / 2


@dataclass
class ContextPattern:
    """Pattern for predictive context loading"""
    pattern_id: str
    context_keys: List[str]
    access_frequency: int = 1
    last_accessed: datetime = field(default_factory=datetime.now)
    prediction_accuracy: float = 0.0
    context_size: int = 0
    
    def update_access(self):
        """Update access statistics"""
        self.access_frequency += 1
        self.last_accessed = datetime.now()
    
    def calculate_priority(self) -> float:
        """Calculate loading priority based on frequency and recency"""
        time_weight = 1.0 / max((datetime.now() - self.last_accessed).total_seconds(), 1.0)
        frequency_weight = self.access_frequency
        accuracy_weight = self.prediction_accuracy
        
        return (time_weight * 0.4 + frequency_weight * 0.4 + accuracy_weight * 0.2)


class ContextPreloader:
    """Intelligent context pre-loading system"""
    
    def __init__(self, config: PerformanceConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Pattern storage
        self.patterns: Dict[str, ContextPattern] = {}
        self.pattern_lock = threading.Lock()
        
        # Cache management
        self.context_cache: Dict[str, Tuple[Any, datetime, int]] = {}  # key -> (data, timestamp, access_count)
        self.cache_lock = threading.Lock()
        
        # Prediction models
        self.user_patterns: Dict[str, Dict[str, Any]] = {}  # user_id -> patterns
        self.conversation_contexts: List[Dict[str, Any]] = []
        
        # Pre-loading tasks
        self.preload_tasks: List[asyncio.Task] = []
        self.is_running = False
        
    async def initialize(self) -> bool:
        """Initialize context pre-loader"""
        try:
            if self.config.context_preload_enabled:
                # Load saved patterns
                await self._load_patterns()
                
                # Start background pre-loading
                if self.config.predictive_loading:
                    self.is_running = True
                    preload_task = asyncio.create_task(self._preload_loop())
                    self.preload_tasks.append(preload_task)
            
            self.logger.info("✅ Context pre-loader initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize context pre-loader: {e}")
            return False
    
    async def shutdown(self):
        """Graceful shutdown"""
        try:
            self.is_running = False
            
            # Cancel pre-loading tasks
            for task in self.preload_tasks:
                task.cancel()
            
            if self.preload_tasks:
                await asyncio.gather(*self.preload_tasks, return_exceptions=True)
            
            # Save patterns
            await self._save_patterns()
            
            self.logger.info("✅ Context pre-loader shutdown completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error during context pre-loader shutdown: {e}")
    
    async def predict_and_preload(self, context_hint: Dict[str, Any]) -> List[str]:
        """Predict and pre-load contexts based on hints"""
        try:
            predicted_contexts = []
            
            if self.config.pattern_learning:
                # Analyze current context for patterns
                pattern_id = self._generate_pattern_id(context_hint)
                
                with self.pattern_lock:
                    if pattern_id in self.patterns:
                        pattern = self.patterns[pattern_id]
                        pattern.update_access()
                        
                        # Pre-load related contexts
                        for context_key in pattern.context_keys:
                            if context_key not in self.context_cache:
                                predicted_contexts.append(context_key)
                                await self._preload_context(context_key)
                    else:
                        # Create new pattern
                        self.patterns[pattern_id] = ContextPattern(
                            pattern_id=pattern_id,
                            context_keys=list(context_hint.keys())
                        )
            
            return predicted_contexts
            
        except Exception as e:
            self.logger.error(f"❌ Error in predict_and_preload: {e}")
            return []
    
    async def get_cached_context(self, context_key: str) -> Optional[Any]:
        """Get context from cache if available"""
        with self.cache_lock:
            if context_key in self.context_cache:
                data, timestamp, access_count = self.context_cache[context_key]
                
                # Update access statistics
                self.context_cache[context_key] = (data, timestamp, access_count + 1)
                
                # Check if data is still fresh (within 5 minutes)
                if (datetime.now() - timestamp).total_seconds() < 300:
                    return data
                else:
                    # Remove stale data
                    del self.context_cache[context_key]
        
        return None
    
    async def cache_context(self, context_key: str, data: Any):
        """Cache context data"""
        with self.cache_lock:
            # Manage cache size
            if len(self.context_cache) >= self.config.context_cache_size:
                # Remove least recently used items
                lru_items = sorted(
                    self.context_cache.items(),
                    key=lambda x: x[1][1]  # Sort by timestamp
                )
                for key, _ in lru_items[:len(lru_items) // 4]:  # Remove 25%
                    del self.context_cache[key]
            
            self.context_cache[context_key] = (data, datetime.now(), 1)
    
    async def _preload_loop(self):
        """Background pre-loading loop"""
        while self.is_running:
            try:
                # Identify high-priority patterns for pre-loading
                with self.pattern_lock:
                    patterns_by_priority = sorted(
                        self.patterns.values(),
                        key=lambda p: p.calculate_priority(),
                        reverse=True
                    )
                
                # Pre-load top priority contexts
                for pattern in patterns_by_priority[:5]:  # Top 5 patterns
                    for context_key in pattern.context_keys:
                        if context_key not in self.context_cache:
                            await self._preload_context(context_key)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Pre-load loop error: {e}")
    
    async def _preload_context(self, context_key: str):
        """Pre-load specific context"""
        try:
            # Simulate context loading (replace with actual context loading logic)
            # This would integrate with memory_manager, screen_provider, etc.
            
            if context_key.startswith("memory:"):
                # Load memory context
                context_data = f"Preloaded memory context for {context_key}"
            elif context_key.startswith("screen:"):
                # Load screen context
                context_data = f"Preloaded screen context for {context_key}"
            else:
                # Generic context
                context_data = f"Preloaded context for {context_key}"
            
            await self.cache_context(context_key, context_data)
            self.logger.debug(f"📋 Pre-loaded context: {context_key}")
            
        except Exception as e:
            self.logger.error(f"❌ Error pre-loading context {context_key}: {e}")
    
    def _generate_pattern_id(self, context_hint: Dict[str, Any]) -> str:
        """Generate unique pattern ID from context"""
        # Create stable hash from context keys and types
        context_signature = {
            "keys": sorted(context_hint.keys()),
            "types": [type(v).__name__ for v in context_hint.values()]
        }
        
        signature_str = json.dumps(context_signature, sort_keys=True)
        return hashlib.md5(signature_str.encode()).hexdigest()[:16]
    
    async def _load_patterns(self):
        """Load saved patterns from disk"""
        try:
            patterns_file = Path(".cache/performance/patterns.json")
            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    patterns_data = json.load(f)
                
                for pattern_data in patterns_data:
                    pattern = ContextPattern(
                        pattern_id=pattern_data["pattern_id"],
                        context_keys=pattern_data["context_keys"],
                        access_frequency=pattern_data.get("access_frequency", 1),
                        prediction_accuracy=pattern_data.get("prediction_accuracy", 0.0)
                    )
                    self.patterns[pattern.pattern_id] = pattern
                
                self.logger.info(f"📋 Loaded {len(self.patterns)} context patterns")
                
        except Exception as e:
            self.logger.error(f"❌ Error loading patterns: {e}")
    
    async def _save_patterns(self):
        """Save patterns to disk"""
        try:
            patterns_file = Path(".cache/performance/patterns.json")
            patterns_file.parent.mkdir(parents=True, exist_ok=True)
            
            patterns_data = []
            for pattern in self.patterns.values():
                patterns_data.append({
                    "pattern_id": pattern.pattern_id,
                    "context_keys": pattern.context_keys,
                    "access_frequency": pattern.access_frequency,
                    "prediction_accuracy": pattern.prediction_accuracy
                })
            
            with open(patterns_file, 'w') as f:
                json.dump(patterns_data, f, indent=2)
            
            self.logger.info(f"💾 Saved {len(patterns_data)} context patterns")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving patterns: {e}")


class ResourceManager:
    """Intelligent resource allocation and monitoring"""
    
    def __init__(self, config: PerformanceConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Resource monitoring
        self.metrics_history: List[SystemMetrics] = []
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        # Resource allocation callbacks
        self.resource_callbacks: List[Callable[[SystemMetrics], None]] = []
        
        # Performance thresholds
        self.performance_thresholds = {
            "cpu_warning": 0.7,
            "cpu_critical": 0.9,
            "memory_warning": 0.6,
            "memory_critical": 0.8,
            "response_time_warning": 200.0,
            "response_time_critical": 250.0
        }
    
    async def initialize(self) -> bool:
        """Initialize resource manager"""
        try:
            if self.config.adaptive_resource_scaling:
                self.is_monitoring = True
                self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            self.logger.info("✅ Resource manager initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize resource manager: {e}")
            return False
    
    async def shutdown(self):
        """Graceful shutdown"""
        try:
            self.is_monitoring = False
            
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info("✅ Resource manager shutdown completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error during resource manager shutdown: {e}")
    
    async def _monitoring_loop(self):
        """Background resource monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect system metrics
                metrics = await self._collect_system_metrics()
                
                # Store metrics
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > 1000:  # Keep last 1000 samples
                    self.metrics_history.pop(0)
                
                # Check thresholds and trigger callbacks
                await self._check_thresholds(metrics)
                
                # Notify callbacks
                for callback in self.resource_callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        self.logger.error(f"❌ Resource callback error: {e}")
                
                await asyncio.sleep(self.config.resource_monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Resource monitoring error: {e}")
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics"""
        try:
            # CPU metrics
            cpu_usage = psutil.cpu_percent(interval=0.1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100.0
            memory_available = memory.available / (1024**3)  # GB
            
            # Network latency (simplified)
            network_latency = 0.0  # Would measure actual network latency
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_io_rate = 0.0
            if disk_io:
                disk_io_rate = disk_io.read_bytes + disk_io.write_bytes
            
            # Calculate response time metrics
            response_times = [m.response_time_avg for m in self.metrics_history[-10:] if m.response_time_avg > 0]
            response_time_avg = statistics.mean(response_times) if response_times else 0.0
            response_time_p95 = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else response_time_avg
            response_time_p99 = statistics.quantiles(response_times, n=100)[98] if len(response_times) >= 100 else response_time_avg
            
            return SystemMetrics(
                cpu_usage=cpu_usage / 100.0,
                memory_usage=memory_usage,
                memory_available=memory_available,
                network_latency=network_latency,
                disk_io_rate=disk_io_rate,
                response_time_avg=response_time_avg,
                response_time_p95=response_time_p95,
                response_time_p99=response_time_p99
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error collecting system metrics: {e}")
            return SystemMetrics()
    
    async def _check_thresholds(self, metrics: SystemMetrics):
        """Check performance thresholds and trigger alerts"""
        # CPU threshold checks
        if metrics.cpu_usage > self.performance_thresholds["cpu_critical"]:
            self.logger.warning(f"🚨 Critical CPU usage: {metrics.cpu_usage:.1%}")
        elif metrics.cpu_usage > self.performance_thresholds["cpu_warning"]:
            self.logger.warning(f"⚠️ High CPU usage: {metrics.cpu_usage:.1%}")
        
        # Memory threshold checks
        if metrics.memory_usage > self.performance_thresholds["memory_critical"]:
            self.logger.warning(f"🚨 Critical memory usage: {metrics.memory_usage:.1%}")
        elif metrics.memory_usage > self.performance_thresholds["memory_warning"]:
            self.logger.warning(f"⚠️ High memory usage: {metrics.memory_usage:.1%}")
        
        # Response time threshold checks
        if metrics.response_time_avg > self.performance_thresholds["response_time_critical"]:
            self.logger.warning(f"🚨 Critical response time: {metrics.response_time_avg:.1f}ms")
        elif metrics.response_time_avg > self.performance_thresholds["response_time_warning"]:
            self.logger.warning(f"⚠️ High response time: {metrics.response_time_avg:.1f}ms")
    
    def add_resource_callback(self, callback: Callable[[SystemMetrics], None]):
        """Add callback for resource updates"""
        self.resource_callbacks.append(callback)
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get most recent metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_resource_recommendations(self) -> Dict[str, Any]:
        """Get resource optimization recommendations"""
        if not self.metrics_history:
            return {}
        
        current = self.metrics_history[-1]
        recommendations = {}
        
        # CPU recommendations
        if current.cpu_usage > 0.8:
            recommendations["cpu"] = "Consider reducing parallel processing workers"
        elif current.cpu_usage < 0.3:
            recommendations["cpu"] = "Can increase parallel processing workers"
        
        # Memory recommendations
        if current.memory_usage > 0.7:
            recommendations["memory"] = "Consider reducing cache sizes or context buffer"
        elif current.memory_usage < 0.4:
            recommendations["memory"] = "Can increase cache sizes for better performance"
        
        # Response time recommendations
        if current.response_time_avg > 200:
            recommendations["latency"] = "Consider switching to more aggressive optimization mode"
        
        return recommendations


class PerformanceOptimizer:
    """Central performance optimization coordinator"""
    
    def __init__(self, 
                 api_key: str,
                 config: Optional[PerformanceConfig] = None,
                 logger: Optional[logging.Logger] = None):
        self.api_key = api_key
        self.config = config or PerformanceConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Core optimization components
        self.connection_manager: Optional[ConnectionPoolManager] = None
        self.audio_optimizer: Optional[AudioStreamOptimizer] = None
        self.context_preloader: Optional[ContextPreloader] = None
        self.resource_manager: Optional[ResourceManager] = None
        
        # Performance monitoring
        self.is_initialized = False
        self.optimization_active = False
        self.performance_metrics: Dict[str, Any] = {}
        
        # Latency tracking with sub-millisecond precision
        self.latency_measurements: List[float] = []
        self.response_time_targets = {
            "connection": 50.0,     # 50ms connection overhead
            "audio_processing": 25.0,  # 25ms audio processing
            "context_loading": 30.0,   # 30ms context loading
            "llm_processing": 145.0    # 145ms LLM processing (remaining budget)
        }
        
        # Adaptive optimization
        self.current_optimization_level = self.config.optimization_level
        self.optimization_history: List[Tuple[datetime, OptimizationLevel, float]] = []
    
    async def initialize(self) -> bool:
        """Initialize all optimization components"""
        try:
            self.logger.info("🚀 Initializing Performance Optimizer...")
            
            # Initialize connection pooling
            connection_config = ConnectionPoolConfig(
                max_connections=100,
                max_pool_size=10,
                pre_warm_enabled=True,
                predictive_scaling=True
            )
            self.connection_manager = ConnectionPoolManager(self.api_key, connection_config, self.logger)
            if not await self.connection_manager.initialize():
                return False
            
            # Initialize audio optimization
            audio_config = AudioConfig(
                buffer_size=512,
                processing_mode=ProcessingMode.ULTRA_LOW_LATENCY,
                parallel_processing=True,
                max_workers=4,
                adaptive_quality=True
            )
            self.audio_optimizer = AudioStreamOptimizer(audio_config, self.logger)
            if not await self.audio_optimizer.initialize():
                return False
            
            # Initialize context pre-loading
            self.context_preloader = ContextPreloader(self.config, self.logger)
            if not await self.context_preloader.initialize():
                return False
            
            # Initialize resource management
            self.resource_manager = ResourceManager(self.config, self.logger)
            if not await self.resource_manager.initialize():
                return False
            
            # Setup resource monitoring callbacks
            self.resource_manager.add_resource_callback(self._on_resource_update)
            
            # Start adaptive optimization
            if self.config.optimization_level == OptimizationLevel.ADAPTIVE:
                asyncio.create_task(self._adaptive_optimization_loop())
            
            self.is_initialized = True
            self.optimization_active = True
            
            self.logger.info("✅ Performance Optimizer initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Performance Optimizer: {e}")
            return False
    
    async def shutdown(self):
        """Graceful shutdown of all components"""
        try:
            self.optimization_active = False
            
            # Shutdown components
            if self.connection_manager:
                await self.connection_manager.shutdown()
            
            if self.audio_optimizer:
                await self.audio_optimizer.shutdown()
            
            if self.context_preloader:
                await self.context_preloader.shutdown()
            
            if self.resource_manager:
                await self.resource_manager.shutdown()
            
            self.logger.info("✅ Performance Optimizer shutdown completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error during Performance Optimizer shutdown: {e}")
    
    async def optimize_request(self, 
                             context_hint: Optional[Dict[str, Any]] = None,
                             user_id: Optional[str] = None,
                             session_id: Optional[str] = None) -> Dict[str, Any]:
        """Optimize a single request with all available strategies"""
        start_time = time.perf_counter()
        optimizations_applied = []
        
        try:
            # 1. Context pre-loading
            if self.context_preloader and context_hint:
                preloaded_contexts = await self.context_preloader.predict_and_preload(context_hint)
                if preloaded_contexts:
                    optimizations_applied.append(f"preloaded_{len(preloaded_contexts)}_contexts")
            
            # 2. Connection optimization
            connection_start = time.perf_counter()
            if self.connection_manager:
                connection_info = await self.connection_manager.get_realtime_connection(
                    session_id or "default", user_id
                )
                if connection_info:
                    connection_time = (time.perf_counter() - connection_start) * 1000
                    optimizations_applied.append(f"connection_pooled_{connection_time:.1f}ms")
            
            # 3. Audio optimization setup
            if self.audio_optimizer and not self.audio_optimizer.is_streaming:
                await self.audio_optimizer.start_streaming()
                optimizations_applied.append("audio_streaming_enabled")
            
            # 4. Resource optimization
            if self.resource_manager:
                current_metrics = self.resource_manager.get_current_metrics()
                if current_metrics:
                    recommendations = self.resource_manager.get_resource_recommendations()
                    if recommendations:
                        optimizations_applied.append(f"resource_recommendations_{len(recommendations)}")
            
            total_time = (time.perf_counter() - start_time) * 1000
            
            return {
                "optimization_time_ms": total_time,
                "optimizations_applied": optimizations_applied,
                "target_response_time_ms": self.config.target_response_time_ms,
                "estimated_savings_ms": self._estimate_latency_savings(optimizations_applied)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error during request optimization: {e}")
            return {"error": str(e), "optimization_time_ms": 0}
    
    async def measure_end_to_end_latency(self, 
                                       operation: Callable[[], Any],
                                       operation_name: str = "operation") -> Tuple[Any, float]:
        """Measure operation latency with sub-millisecond precision"""
        start_time = time.perf_counter()
        
        try:
            result = await operation() if asyncio.iscoroutinefunction(operation) else operation()
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            self.latency_measurements.append(latency_ms)
            
            # Keep only recent measurements
            if len(self.latency_measurements) > 1000:
                self.latency_measurements = self.latency_measurements[-1000:]
            
            self.logger.debug(f"⏱️ {operation_name}: {latency_ms:.3f}ms")
            
            return result, latency_ms
            
        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            self.logger.error(f"❌ {operation_name} failed after {latency_ms:.3f}ms: {e}")
            raise
    
    def _estimate_latency_savings(self, optimizations: List[str]) -> float:
        """Estimate latency savings from applied optimizations"""
        savings = 0.0
        
        for optimization in optimizations:
            if "connection_pooled" in optimization:
                savings += 100.0  # Connection pooling saves ~100ms
            elif "preloaded" in optimization:
                savings += 30.0   # Context pre-loading saves ~30ms
            elif "audio_streaming" in optimization:
                savings += 25.0   # Audio optimization saves ~25ms
        
        return savings
    
    async def _adaptive_optimization_loop(self):
        """Adaptive optimization based on performance feedback"""
        while self.optimization_active:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Analyze recent performance
                if len(self.latency_measurements) >= 10:
                    recent_avg = statistics.mean(self.latency_measurements[-10:])
                    
                    # Adjust optimization level based on performance
                    if recent_avg > self.config.max_acceptable_latency_ms:
                        # Performance is poor, increase optimization
                        if self.current_optimization_level == OptimizationLevel.STANDARD:
                            await self._switch_optimization_level(OptimizationLevel.AGGRESSIVE)
                        elif self.current_optimization_level == OptimizationLevel.MINIMAL:
                            await self._switch_optimization_level(OptimizationLevel.STANDARD)
                    
                    elif recent_avg < self.config.target_response_time_ms * 0.8:
                        # Performance is excellent, can reduce optimization overhead
                        if self.current_optimization_level == OptimizationLevel.AGGRESSIVE:
                            await self._switch_optimization_level(OptimizationLevel.STANDARD)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Adaptive optimization error: {e}")
    
    async def _switch_optimization_level(self, new_level: OptimizationLevel):
        """Switch to a different optimization level"""
        if new_level == self.current_optimization_level:
            return
        
        old_level = self.current_optimization_level
        self.current_optimization_level = new_level
        
        # Record optimization level change
        recent_avg = statistics.mean(self.latency_measurements[-10:]) if len(self.latency_measurements) >= 10 else 0.0
        self.optimization_history.append((datetime.now(), new_level, recent_avg))
        
        # Apply optimization level changes
        if self.audio_optimizer:
            if new_level == OptimizationLevel.AGGRESSIVE:
                self.audio_optimizer.config.processing_mode = ProcessingMode.ULTRA_LOW_LATENCY
                self.audio_optimizer.config.max_workers = 6
            elif new_level == OptimizationLevel.STANDARD:
                self.audio_optimizer.config.processing_mode = ProcessingMode.LOW_LATENCY
                self.audio_optimizer.config.max_workers = 4
            elif new_level == OptimizationLevel.MINIMAL:
                self.audio_optimizer.config.processing_mode = ProcessingMode.BALANCED
                self.audio_optimizer.config.max_workers = 2
        
        self.logger.info(f"🔧 Optimization level changed: {old_level.value} → {new_level.value}")
    
    def _on_resource_update(self, metrics: SystemMetrics):
        """Handle resource metric updates"""
        # Store in performance metrics
        self.performance_metrics.update({
            "cpu_usage": metrics.cpu_usage,
            "memory_usage": metrics.memory_usage,
            "response_time_avg": metrics.response_time_avg,
            "resource_pressure": metrics.get_resource_pressure()
        })
        
        # Trigger emergency optimization if needed
        if metrics.get_resource_pressure() > 0.9:
            asyncio.create_task(self._emergency_optimization(metrics))
    
    async def _emergency_optimization(self, metrics: SystemMetrics):
        """Emergency optimization when resources are critically low"""
        self.logger.warning("🚨 Emergency optimization triggered")
        
        # Reduce audio processing workers
        if self.audio_optimizer and metrics.cpu_usage > 0.9:
            self.audio_optimizer.config.max_workers = max(1, self.audio_optimizer.config.max_workers - 1)
            self.logger.warning(f"⚠️ Reduced audio workers to {self.audio_optimizer.config.max_workers}")
        
        # Clear context cache if memory is low
        if self.context_preloader and metrics.memory_usage > 0.85:
            with self.context_preloader.cache_lock:
                cache_size_before = len(self.context_preloader.context_cache)
                self.context_preloader.context_cache.clear()
                self.logger.warning(f"⚠️ Cleared context cache ({cache_size_before} items)")
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics from all components"""
        metrics = {
            "optimizer": {
                "is_initialized": self.is_initialized,
                "optimization_active": self.optimization_active,
                "current_optimization_level": self.current_optimization_level.value,
                "target_response_time_ms": self.config.target_response_time_ms,
                "recent_latency_avg": statistics.mean(self.latency_measurements[-10:]) if len(self.latency_measurements) >= 10 else 0.0,
                "latency_measurements_count": len(self.latency_measurements),
                "optimization_level_changes": len(self.optimization_history)
            }
        }
        
        # Add component metrics
        if self.connection_manager:
            metrics["connection_manager"] = self.connection_manager.get_comprehensive_metrics()
        
        if self.audio_optimizer:
            metrics["audio_optimizer"] = self.audio_optimizer.get_comprehensive_metrics()
        
        if self.resource_manager:
            current_metrics = self.resource_manager.get_current_metrics()
            if current_metrics:
                metrics["system_resources"] = {
                    "cpu_usage": current_metrics.cpu_usage,
                    "memory_usage": current_metrics.memory_usage,
                    "response_time_avg": current_metrics.response_time_avg,
                    "resource_pressure": current_metrics.get_resource_pressure()
                }
        
        return metrics


# Factory function for easy creation
def create_performance_optimizer(
    api_key: str,
    target_response_time_ms: float = 250.0,
    optimization_level: OptimizationLevel = OptimizationLevel.ADAPTIVE,
    logger: Optional[logging.Logger] = None
) -> PerformanceOptimizer:
    """Create optimized performance coordinator with common configuration"""
    
    config = PerformanceConfig(
        optimization_level=optimization_level,
        target_response_time_ms=target_response_time_ms,
        context_preload_enabled=True,
        adaptive_resource_scaling=True,
        pattern_learning=True
    )
    
    return PerformanceOptimizer(api_key, config, logger) 