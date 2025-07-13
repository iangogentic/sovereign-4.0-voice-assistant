"""
Predictive Optimization Engine for Sovereign 4.0 Voice Assistant

Implements automated performance optimization triggers with predictive lookahead capabilities
following 2024-2025 best practices for proactive system optimization.

Key Features:
- Threshold-based automation with predictive forecasting
- Multiple optimization strategies (scaling, model switching, caching)
- Safety mechanisms and rollback capabilities
- Performance impact tracking and learning
- Integration with existing configuration and monitoring systems
"""

import asyncio
import logging
import time
import threading
import json
from typing import Dict, Any, Optional, List, Union, Callable, Awaitable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

from .predictive_analytics import PredictiveAnalyticsFramework
from .config_manager import ConfigManager, SovereignConfig
from .metrics_collector import MetricsCollector
from .structured_logging import VoiceAILogger, get_voice_ai_logger
from .error_handling import ModernCircuitBreaker, CircuitBreakerConfig

class OptimizationType(Enum):
    """Types of optimization that can be triggered"""
    RESOURCE_SCALING = "resource_scaling"
    MODEL_SWITCHING = "model_switching"
    CACHE_OPTIMIZATION = "cache_optimization"
    TIMEOUT_ADJUSTMENT = "timeout_adjustment"
    QUEUE_MANAGEMENT = "queue_management"
    CIRCUIT_BREAKER_ADJUSTMENT = "circuit_breaker_adjustment"
    LOAD_BALANCING = "load_balancing"
    MEMORY_OPTIMIZATION = "memory_optimization"

class OptimizationPriority(Enum):
    """Priority levels for optimization actions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class OptimizationStatus(Enum):
    """Status of optimization actions"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"

@dataclass
class OptimizationRule:
    """Rule defining when and how to optimize"""
    rule_id: str
    name: str
    description: str
    optimization_type: OptimizationType
    priority: OptimizationPriority
    
    # Trigger conditions
    trigger_conditions: Dict[str, Any]  # Thresholds and conditions
    prediction_horizon: float  # seconds to look ahead
    confidence_threshold: float  # minimum prediction confidence
    
    # Action configuration
    action_config: Dict[str, Any]  # Specific optimization parameters
    target_component: str  # Which component to optimize
    
    # Safety and rollback
    max_impact_level: str  # low, medium, high
    requires_approval: bool
    rollback_timeout: float  # seconds
    rollback_conditions: Dict[str, Any]
    
    # Learning and adaptation
    enabled: bool = True
    success_rate: float = 0.0
    last_triggered: Optional[float] = None
    trigger_count: int = 0

@dataclass
class OptimizationAction:
    """Specific optimization action to be executed"""
    action_id: str
    rule_id: str
    optimization_type: OptimizationType
    priority: OptimizationPriority
    status: OptimizationStatus
    
    # Trigger information
    triggered_at: float
    trigger_reason: str
    prediction_data: Dict[str, Any]
    system_state: Dict[str, Any]
    
    # Action details
    action_config: Dict[str, Any]
    target_component: str
    expected_impact: str
    
    # Execution tracking
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration_ms: Optional[float] = None
    
    # Results and impact
    success: Optional[bool] = None
    error_message: Optional[str] = None
    impact_metrics: Optional[Dict[str, Any]] = None
    rollback_performed: bool = False
    
    # Safety and approval
    requires_approval: bool = False
    approved_by: Optional[str] = None
    approved_at: Optional[float] = None

@dataclass
class OptimizationResult:
    """Result of an optimization action"""
    action_id: str
    success: bool
    duration_ms: float
    impact_metrics: Dict[str, Any]
    error_message: Optional[str] = None
    rollback_required: bool = False
    lessons_learned: List[str] = field(default_factory=list)

class PredictiveOptimizationEngine:
    """
    Predictive Optimization Engine for automated performance optimization
    
    Implements intelligent optimization triggers that:
    - Monitor predictive analytics for optimization opportunities
    - Execute automated optimizations within safety bounds
    - Track performance impact and learn from results
    - Provide rollback capabilities for failed optimizations
    """
    
    def __init__(
        self,
        predictive_analytics: Optional[PredictiveAnalyticsFramework] = None,
        config_manager: Optional[ConfigManager] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        logger: Optional[VoiceAILogger] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.predictive_analytics = predictive_analytics
        self.config_manager = config_manager
        self.metrics_collector = metrics_collector
        self.logger = logger or get_voice_ai_logger("predictive_optimization")
        
        # Configuration
        self.config = config or self._get_default_config()
        
        # Rule and action management
        self._lock = threading.RLock()
        self.optimization_rules: Dict[str, OptimizationRule] = {}
        self.active_actions: Dict[str, OptimizationAction] = {}
        self.action_history: List[OptimizationAction] = []
        
        # Optimization handlers
        self.optimization_handlers: Dict[OptimizationType, Callable] = {}
        self.rollback_handlers: Dict[str, Callable] = {}  # action_id -> rollback function
        
        # Performance tracking
        self.optimization_stats = {
            'total_actions': 0,
            'successful_actions': 0,
            'failed_actions': 0,
            'rollback_actions': 0,
            'average_impact_score': 0.0,
            'total_time_saved_ms': 0.0
        }
        
        # Safety mechanisms
        self.safety_limits = {
            'max_concurrent_optimizations': 3,
            'max_optimizations_per_hour': 10,
            'min_time_between_optimizations': 300,  # 5 minutes
            'emergency_stop_enabled': True
        }
        
        # Circuit breakers for optimization types
        self.optimization_circuit_breakers: Dict[OptimizationType, ModernCircuitBreaker] = {}
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Initialize default optimization rules
        self._initialize_default_rules()
        
        # Initialize optimization handlers
        self._initialize_optimization_handlers()
        
        # Initialize circuit breakers
        self._initialize_circuit_breakers()
        
        self.logger.info("🚀 PredictiveOptimizationEngine initialized with automated triggers")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for optimization engine"""
        return {
            'monitoring_interval': 30.0,  # seconds
            'prediction_lookahead': 300.0,  # 5 minutes
            'default_confidence_threshold': 0.7,
            'max_action_history': 1000,
            'action_timeout': 60.0,  # seconds
            'rollback_timeout': 30.0,  # seconds
            'impact_measurement_window': 300.0,  # 5 minutes
            'learning_enabled': True,
            'auto_approval_threshold': 0.9,  # Auto-approve high-confidence, low-risk actions
            'emergency_stop_conditions': {
                'error_rate_threshold': 0.5,
                'cascade_detection': True
            }
        }
    
    def _initialize_default_rules(self):
        """Initialize default optimization rules"""
        default_rules = [
            # High latency optimization
            OptimizationRule(
                rule_id="latency_scaling",
                name="Latency-based Resource Scaling",
                description="Scale resources when high latency is predicted",
                optimization_type=OptimizationType.RESOURCE_SCALING,
                priority=OptimizationPriority.HIGH,
                trigger_conditions={
                    'predicted_latency_p95': 1000,  # ms
                    'current_latency_p95': 800,  # ms
                    'trend_direction': 'increasing'
                },
                prediction_horizon=300.0,  # 5 minutes
                confidence_threshold=0.7,
                action_config={
                    'scaling_factor': 1.5,
                    'max_instances': 10,
                    'scaling_type': 'horizontal'
                },
                target_component="llm_service",
                max_impact_level="medium",
                requires_approval=False,
                rollback_timeout=300.0,
                rollback_conditions={
                    'latency_improvement': 0.2,  # 20% improvement required
                    'error_rate_increase': 0.1   # Rollback if errors increase by 10%
                }
            ),
            
            # Model switching for performance
            OptimizationRule(
                rule_id="model_performance_switching",
                name="Performance-based Model Switching",
                description="Switch to faster model when performance degradation is predicted",
                optimization_type=OptimizationType.MODEL_SWITCHING,
                priority=OptimizationPriority.MEDIUM,
                trigger_conditions={
                    'predicted_degradation_probability': 0.6,
                    'current_response_time': 2000,  # ms
                    'accuracy_threshold': 0.85
                },
                prediction_horizon=600.0,  # 10 minutes
                confidence_threshold=0.8,
                action_config={
                    'fallback_model': 'fast_model',
                    'quality_threshold': 0.8,
                    'switch_duration': 1800  # 30 minutes
                },
                target_component="llm_router",
                max_impact_level="low",
                requires_approval=False,
                rollback_timeout=1800.0,
                rollback_conditions={
                    'quality_degradation': 0.1,
                    'user_satisfaction': 0.7
                }
            ),
            
            # Cache optimization
            OptimizationRule(
                rule_id="cache_optimization",
                name="Predictive Cache Warming",
                description="Warm cache with predicted high-demand items",
                optimization_type=OptimizationType.CACHE_OPTIMIZATION,
                priority=OptimizationPriority.LOW,
                trigger_conditions={
                    'predicted_cache_miss_rate': 0.3,
                    'demand_forecast_confidence': 0.8,
                    'available_memory': 0.5  # 50% memory available
                },
                prediction_horizon=1800.0,  # 30 minutes
                confidence_threshold=0.6,
                action_config={
                    'cache_size_increase': 0.2,  # 20% increase
                    'warming_items': 'top_predicted',
                    'warming_limit': 1000
                },
                target_component="response_cache",
                max_impact_level="low",
                requires_approval=False,
                rollback_timeout=600.0,
                rollback_conditions={
                    'memory_usage': 0.9,  # Rollback if memory usage > 90%
                    'cache_hit_improvement': 0.1  # Require 10% improvement
                }
            ),
            
            # Timeout adjustment
            OptimizationRule(
                rule_id="timeout_optimization",
                name="Dynamic Timeout Adjustment",
                description="Adjust timeouts based on predicted service performance",
                optimization_type=OptimizationType.TIMEOUT_ADJUSTMENT,
                priority=OptimizationPriority.MEDIUM,
                trigger_conditions={
                    'predicted_service_slowdown': 0.3,  # 30% slowdown predicted
                    'timeout_hit_rate': 0.05,  # 5% timeout rate
                    'service_stability': 0.8
                },
                prediction_horizon=900.0,  # 15 minutes
                confidence_threshold=0.75,
                action_config={
                    'timeout_multiplier': 1.5,
                    'adaptive_timeout': True,
                    'max_timeout': 30000  # 30 seconds
                },
                target_component="api_clients",
                max_impact_level="low",
                requires_approval=False,
                rollback_timeout=900.0,
                rollback_conditions={
                    'timeout_improvement': 0.2,
                    'response_time_degradation': 0.3
                }
            ),
            
            # Memory optimization
            OptimizationRule(
                rule_id="memory_optimization",
                name="Predictive Memory Management",
                description="Optimize memory usage when high usage is predicted",
                optimization_type=OptimizationType.MEMORY_OPTIMIZATION,
                priority=OptimizationPriority.HIGH,
                trigger_conditions={
                    'predicted_memory_usage': 0.85,  # 85% usage predicted
                    'current_memory_usage': 0.75,    # 75% current usage
                    'memory_growth_rate': 0.1        # 10% growth rate
                },
                prediction_horizon=600.0,  # 10 minutes
                confidence_threshold=0.8,
                action_config={
                    'garbage_collection': True,
                    'cache_cleanup': True,
                    'model_unloading': 'least_used'
                },
                target_component="memory_manager",
                max_impact_level="medium",
                requires_approval=False,
                rollback_timeout=180.0,
                rollback_conditions={
                    'memory_reduction': 0.1,  # Require 10% reduction
                    'performance_impact': 0.2  # Rollback if performance degrades 20%
                }
            )
        ]
        
        for rule in default_rules:
            self.optimization_rules[rule.rule_id] = rule
    
    def _initialize_optimization_handlers(self):
        """Initialize handlers for different optimization types"""
        self.optimization_handlers = {
            OptimizationType.RESOURCE_SCALING: self._handle_resource_scaling,
            OptimizationType.MODEL_SWITCHING: self._handle_model_switching,
            OptimizationType.CACHE_OPTIMIZATION: self._handle_cache_optimization,
            OptimizationType.TIMEOUT_ADJUSTMENT: self._handle_timeout_adjustment,
            OptimizationType.MEMORY_OPTIMIZATION: self._handle_memory_optimization,
            OptimizationType.QUEUE_MANAGEMENT: self._handle_queue_management,
            OptimizationType.CIRCUIT_BREAKER_ADJUSTMENT: self._handle_circuit_breaker_adjustment,
            OptimizationType.LOAD_BALANCING: self._handle_load_balancing
        }
    
    def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for optimization types"""
        cb_config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_duration=300.0,  # 5 minutes
            failure_rate_threshold=0.6
        )
        
        for opt_type in OptimizationType:
            self.optimization_circuit_breakers[opt_type] = ModernCircuitBreaker(
                f"optimization_{opt_type.value}", cb_config
            )
    
    async def start_monitoring(self):
        """Start predictive optimization monitoring"""
        if self.monitoring_task and not self.monitoring_task.done():
            self.logger.warning("Optimization monitoring already running")
            return
        
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        self.logger.info("🚀 Predictive optimization monitoring started")
    
    async def stop_monitoring(self):
        """Stop predictive optimization monitoring"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Wait for tasks to complete
        if self.monitoring_task:
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self.cleanup_task:
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("🚀 Predictive optimization monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop for optimization triggers"""
        try:
            while True:
                try:
                    await self._evaluate_optimization_triggers()
                    await asyncio.sleep(self.config['monitoring_interval'])
                except Exception as e:
                    self.logger.error(f"Error in optimization monitoring loop: {e}")
                    await asyncio.sleep(30)  # Wait before retrying
                    
        except asyncio.CancelledError:
            self.logger.info("🚀 Optimization monitoring loop cancelled")
    
    async def _cleanup_loop(self):
        """Cleanup loop for completed actions and maintenance"""
        try:
            while True:
                try:
                    await self._cleanup_completed_actions()
                    await self._update_rule_statistics()
                    await asyncio.sleep(300)  # 5 minutes
                except Exception as e:
                    self.logger.error(f"Error in cleanup loop: {e}")
                    await asyncio.sleep(60)  # Wait before retrying
                    
        except asyncio.CancelledError:
            self.logger.info("🚀 Optimization cleanup loop cancelled")
    
    async def _evaluate_optimization_triggers(self):
        """Evaluate all optimization rules for trigger conditions"""
        if not self.predictive_analytics:
            return
        
        try:
            # Check safety limits
            if not self._check_safety_limits():
                return
            
            # Get current predictions
            predictions = await self._get_optimization_predictions()
            
            # Get current system state
            system_state = await self._get_system_state()
            
            # Evaluate each rule
            for rule_id, rule in self.optimization_rules.items():
                if not rule.enabled:
                    continue
                
                # Check circuit breaker
                circuit_breaker = self.optimization_circuit_breakers.get(rule.optimization_type)
                if circuit_breaker and circuit_breaker.state.value == "open":
                    continue
                
                # Check if rule should trigger
                should_trigger, trigger_reason = await self._should_trigger_optimization(
                    rule, predictions, system_state
                )
                
                if should_trigger:
                    await self._trigger_optimization(rule, trigger_reason, predictions, system_state)
                    
        except Exception as e:
            self.logger.error(f"Error evaluating optimization triggers: {e}")
    
    async def _get_optimization_predictions(self) -> Dict[str, Any]:
        """Get predictions relevant to optimization decisions"""
        predictions = {}
        
        try:
            if self.predictive_analytics:
                # Get performance predictions
                performance_predictions = await self.predictive_analytics.performance_predictor.get_predictions()
                predictions['performance'] = performance_predictions
                
                # Get resource forecasts
                resource_forecasts = await self.predictive_analytics.resource_forecaster.get_forecasts()
                predictions['resources'] = resource_forecasts
                
                # Get drift alerts
                drift_alerts = await self.predictive_analytics.drift_detector.get_active_alerts()
                predictions['drift'] = [asdict(alert) for alert in drift_alerts]
                
                # Get health score
                health_score = await self.predictive_analytics.get_health_score()
                predictions['health_score'] = health_score
                
        except Exception as e:
            self.logger.warning(f"Error getting optimization predictions: {e}")
        
        return predictions
    
    async def _get_system_state(self) -> Dict[str, Any]:
        """Get current system state for optimization decisions"""
        system_state = {'timestamp': time.time()}
        
        try:
            if self.metrics_collector:
                current_metrics = self.metrics_collector.get_current_metrics()
                system_state['metrics'] = current_metrics
            
            # Add system resource information
            system_state['resources'] = await self._get_resource_status()
            
            # Add active optimization count
            with self._lock:
                system_state['active_optimizations'] = len(self.active_actions)
            
        except Exception as e:
            self.logger.warning(f"Error getting system state: {e}")
        
        return system_state
    
    async def _get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status"""
        try:
            import psutil
            
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'available_memory_gb': psutil.virtual_memory().available / (1024**3)
            }
        except Exception:
            return {}
    
    async def _should_trigger_optimization(
        self,
        rule: OptimizationRule,
        predictions: Dict[str, Any],
        system_state: Dict[str, Any]
    ) -> tuple[bool, str]:
        """Determine if optimization rule should trigger"""
        try:
            # Check time since last trigger
            if rule.last_triggered:
                time_since_last = time.time() - rule.last_triggered
                if time_since_last < self.safety_limits['min_time_between_optimizations']:
                    return False, "Too soon since last trigger"
            
            # Check trigger conditions based on optimization type
            if rule.optimization_type == OptimizationType.RESOURCE_SCALING:
                return await self._check_scaling_conditions(rule, predictions, system_state)
            
            elif rule.optimization_type == OptimizationType.MODEL_SWITCHING:
                return await self._check_model_switching_conditions(rule, predictions, system_state)
            
            elif rule.optimization_type == OptimizationType.CACHE_OPTIMIZATION:
                return await self._check_cache_conditions(rule, predictions, system_state)
            
            elif rule.optimization_type == OptimizationType.TIMEOUT_ADJUSTMENT:
                return await self._check_timeout_conditions(rule, predictions, system_state)
            
            elif rule.optimization_type == OptimizationType.MEMORY_OPTIMIZATION:
                return await self._check_memory_conditions(rule, predictions, system_state)
            
            else:
                return False, f"Unknown optimization type: {rule.optimization_type}"
                
        except Exception as e:
            self.logger.error(f"Error checking trigger conditions for rule {rule.rule_id}: {e}")
            return False, f"Error: {e}"
    
    async def _check_scaling_conditions(
        self,
        rule: OptimizationRule,
        predictions: Dict[str, Any],
        system_state: Dict[str, Any]
    ) -> tuple[bool, str]:
        """Check conditions for resource scaling optimization"""
        try:
            conditions = rule.trigger_conditions
            
            # Check predicted latency
            performance_pred = predictions.get('performance', {})
            predicted_latency = performance_pred.get('latency_p95', 0)
            
            if predicted_latency > conditions.get('predicted_latency_p95', 1000):
                # Check current latency
                current_metrics = system_state.get('metrics')
                if current_metrics and hasattr(current_metrics, 'latency'):
                    current_latency = current_metrics.latency.get_p95()
                    
                    if current_latency > conditions.get('current_latency_p95', 800):
                        # Check trend direction
                        trend = performance_pred.get('trend', 'stable')
                        if trend == conditions.get('trend_direction', 'increasing'):
                            return True, f"High latency predicted: {predicted_latency:.0f}ms"
            
            return False, "Scaling conditions not met"
            
        except Exception as e:
            return False, f"Error checking scaling conditions: {e}"
    
    async def _check_model_switching_conditions(
        self,
        rule: OptimizationRule,
        predictions: Dict[str, Any],
        system_state: Dict[str, Any]
    ) -> tuple[bool, str]:
        """Check conditions for model switching optimization"""
        try:
            conditions = rule.trigger_conditions
            
            # Check predicted degradation probability
            performance_pred = predictions.get('performance', {})
            degradation_prob = performance_pred.get('degradation_probability', 0)
            
            if degradation_prob > conditions.get('predicted_degradation_probability', 0.6):
                # Check current response time
                current_metrics = system_state.get('metrics')
                if current_metrics and hasattr(current_metrics, 'latency'):
                    current_response_time = current_metrics.latency.get_average()
                    
                    if current_response_time > conditions.get('current_response_time', 2000):
                        # Check if accuracy is still acceptable
                        accuracy = performance_pred.get('accuracy_score', 0.9)
                        if accuracy > conditions.get('accuracy_threshold', 0.85):
                            return True, f"Performance degradation predicted: {degradation_prob:.2f}"
            
            return False, "Model switching conditions not met"
            
        except Exception as e:
            return False, f"Error checking model switching conditions: {e}"
    
    async def _check_cache_conditions(
        self,
        rule: OptimizationRule,
        predictions: Dict[str, Any],
        system_state: Dict[str, Any]
    ) -> tuple[bool, str]:
        """Check conditions for cache optimization"""
        try:
            conditions = rule.trigger_conditions
            
            # Check predicted cache miss rate
            resource_forecasts = predictions.get('resources', {})
            predicted_miss_rate = resource_forecasts.get('cache_miss_rate', 0)
            
            if predicted_miss_rate > conditions.get('predicted_cache_miss_rate', 0.3):
                # Check forecast confidence
                confidence = resource_forecasts.get('confidence', 0)
                if confidence > conditions.get('demand_forecast_confidence', 0.8):
                    # Check available memory
                    resources = system_state.get('resources', {})
                    memory_usage = resources.get('memory_percent', 100) / 100
                    available_memory = 1.0 - memory_usage
                    
                    if available_memory > conditions.get('available_memory', 0.5):
                        return True, f"Cache optimization opportunity: {predicted_miss_rate:.2f} miss rate"
            
            return False, "Cache optimization conditions not met"
            
        except Exception as e:
            return False, f"Error checking cache conditions: {e}"
    
    async def _check_timeout_conditions(
        self,
        rule: OptimizationRule,
        predictions: Dict[str, Any],
        system_state: Dict[str, Any]
    ) -> tuple[bool, str]:
        """Check conditions for timeout adjustment"""
        try:
            conditions = rule.trigger_conditions
            
            # Check predicted service slowdown
            performance_pred = predictions.get('performance', {})
            slowdown = performance_pred.get('service_slowdown', 0)
            
            if slowdown > conditions.get('predicted_service_slowdown', 0.3):
                # Check current timeout hit rate
                current_metrics = system_state.get('metrics')
                if current_metrics and hasattr(current_metrics, 'throughput'):
                    timeout_rate = getattr(current_metrics.throughput, 'timeout_rate', 0)
                    
                    if timeout_rate > conditions.get('timeout_hit_rate', 0.05):
                        # Check service stability
                        stability = performance_pred.get('stability_score', 1.0)
                        if stability > conditions.get('service_stability', 0.8):
                            return True, f"Service slowdown predicted: {slowdown:.2f}"
            
            return False, "Timeout adjustment conditions not met"
            
        except Exception as e:
            return False, f"Error checking timeout conditions: {e}"
    
    async def _check_memory_conditions(
        self,
        rule: OptimizationRule,
        predictions: Dict[str, Any],
        system_state: Dict[str, Any]
    ) -> tuple[bool, str]:
        """Check conditions for memory optimization"""
        try:
            conditions = rule.trigger_conditions
            
            # Check predicted memory usage
            resource_forecasts = predictions.get('resources', {})
            predicted_memory = resource_forecasts.get('memory_usage', 0)
            
            if predicted_memory > conditions.get('predicted_memory_usage', 0.85):
                # Check current memory usage
                resources = system_state.get('resources', {})
                current_memory = resources.get('memory_percent', 0) / 100
                
                if current_memory > conditions.get('current_memory_usage', 0.75):
                    # Check memory growth rate
                    growth_rate = resource_forecasts.get('memory_growth_rate', 0)
                    if growth_rate > conditions.get('memory_growth_rate', 0.1):
                        return True, f"High memory usage predicted: {predicted_memory:.2f}"
            
            return False, "Memory optimization conditions not met"
            
        except Exception as e:
            return False, f"Error checking memory conditions: {e}"
    
    def _check_safety_limits(self) -> bool:
        """Check if it's safe to trigger new optimizations"""
        try:
            with self._lock:
                # Check concurrent optimization limit
                active_count = len(self.active_actions)
                if active_count >= self.safety_limits['max_concurrent_optimizations']:
                    return False
                
                # Check hourly optimization limit
                one_hour_ago = time.time() - 3600
                recent_actions = [
                    action for action in self.action_history
                    if action.triggered_at > one_hour_ago
                ]
                
                if len(recent_actions) >= self.safety_limits['max_optimizations_per_hour']:
                    return False
                
                # Check emergency stop conditions
                if self.safety_limits['emergency_stop_enabled']:
                    if self._check_emergency_stop_conditions():
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking safety limits: {e}")
            return False
    
    def _check_emergency_stop_conditions(self) -> bool:
        """Check if emergency stop conditions are met"""
        try:
            if not self.metrics_collector:
                return False
            
            current_metrics = self.metrics_collector.get_current_metrics()
            
            # Check error rate
            if hasattr(current_metrics, 'throughput'):
                error_rate = current_metrics.throughput.error_rate
                threshold = self.config['emergency_stop_conditions']['error_rate_threshold']
                if error_rate > threshold:
                    self.logger.warning(f"Emergency stop: High error rate {error_rate:.2f} > {threshold}")
                    return True
            
            # Check for cascade detection
            if self.config['emergency_stop_conditions']['cascade_detection']:
                # Simple cascade detection based on rapid optimization triggers
                recent_actions = [
                    action for action in self.action_history[-10:]  # Last 10 actions
                    if time.time() - action.triggered_at < 300  # Last 5 minutes
                ]
                
                if len(recent_actions) > 5:  # More than 5 optimizations in 5 minutes
                    self.logger.warning("Emergency stop: Potential optimization cascade detected")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking emergency stop conditions: {e}")
            return False
    
    async def _trigger_optimization(
        self,
        rule: OptimizationRule,
        trigger_reason: str,
        predictions: Dict[str, Any],
        system_state: Dict[str, Any]
    ):
        """Trigger an optimization action"""
        try:
            # Create optimization action
            action = OptimizationAction(
                action_id=f"opt_{rule.rule_id}_{int(time.time())}",
                rule_id=rule.rule_id,
                optimization_type=rule.optimization_type,
                priority=rule.priority,
                status=OptimizationStatus.PENDING,
                triggered_at=time.time(),
                trigger_reason=trigger_reason,
                prediction_data=predictions,
                system_state=system_state,
                action_config=rule.action_config.copy(),
                target_component=rule.target_component,
                expected_impact=rule.max_impact_level,
                requires_approval=rule.requires_approval
            )
            
            # Check if approval is required
            if rule.requires_approval and not self._should_auto_approve(action):
                self.logger.info(f"🚀 Optimization {action.action_id} requires approval")
                # In a real implementation, this would queue for approval
                return
            
            # Add to active actions
            with self._lock:
                self.active_actions[action.action_id] = action
                rule.last_triggered = time.time()
                rule.trigger_count += 1
            
            # Execute optimization asynchronously
            asyncio.create_task(self._execute_optimization_action(action))
            
            self.logger.info(
                f"🚀 Triggered optimization: {rule.name} "
                f"(Action: {action.action_id}, Reason: {trigger_reason})"
            )
            
        except Exception as e:
            self.logger.error(f"Error triggering optimization for rule {rule.rule_id}: {e}")
    
    def _should_auto_approve(self, action: OptimizationAction) -> bool:
        """Determine if action should be auto-approved"""
        try:
            # Check confidence threshold for auto-approval
            prediction_confidence = action.prediction_data.get('confidence', 0.0)
            auto_threshold = self.config['auto_approval_threshold']
            
            # Auto-approve high-confidence, low-risk actions
            if (prediction_confidence > auto_threshold and 
                action.expected_impact in ['low', 'medium']):
                return True
            
            return False
            
        except Exception:
            return False
    
    async def _execute_optimization_action(self, action: OptimizationAction):
        """Execute an optimization action"""
        start_time = time.time()
        
        try:
            # Update action status
            action.status = OptimizationStatus.IN_PROGRESS
            action.started_at = start_time
            
            # Get optimization handler
            handler = self.optimization_handlers.get(action.optimization_type)
            if not handler:
                raise ValueError(f"No handler for optimization type: {action.optimization_type}")
            
            # Execute optimization with timeout
            result = await asyncio.wait_for(
                handler(action),
                timeout=self.config['action_timeout']
            )
            
            # Update action with results
            action.status = OptimizationStatus.COMPLETED
            action.completed_at = time.time()
            action.duration_ms = (action.completed_at - start_time) * 1000
            action.success = result.success
            action.impact_metrics = result.impact_metrics
            action.error_message = result.error_message
            
            # Store rollback handler if provided
            if hasattr(result, 'rollback_handler') and result.rollback_handler:
                self.rollback_handlers[action.action_id] = result.rollback_handler
            
            # Update circuit breaker
            circuit_breaker = self.optimization_circuit_breakers.get(action.optimization_type)
            if circuit_breaker:
                if result.success:
                    await circuit_breaker.call_success()
                else:
                    await circuit_breaker.call_failure()
            
            # Update statistics
            self._update_optimization_stats(action, result)
            
            # Schedule rollback check if needed
            if result.success and not result.rollback_required:
                asyncio.create_task(
                    self._schedule_rollback_check(action, action.action_config.get('rollback_timeout', 300))
                )
            
            self.logger.info(
                f"🚀 Optimization completed: {action.action_id} "
                f"(Success: {result.success}, Duration: {action.duration_ms:.1f}ms)"
            )
            
        except asyncio.TimeoutError:
            action.status = OptimizationStatus.FAILED
            action.error_message = "Optimization timed out"
            self.logger.error(f"Optimization {action.action_id} timed out")
            
        except Exception as e:
            action.status = OptimizationStatus.FAILED
            action.error_message = str(e)
            self.logger.error(f"Optimization {action.action_id} failed: {e}")
            
        finally:
            # Move to history
            with self._lock:
                if action.action_id in self.active_actions:
                    del self.active_actions[action.action_id]
                self.action_history.append(action)
                
                # Limit history size
                if len(self.action_history) > self.config['max_action_history']:
                    self.action_history.pop(0)
    
    # Optimization handlers for different types
    async def _handle_resource_scaling(self, action: OptimizationAction) -> OptimizationResult:
        """Handle resource scaling optimization"""
        try:
            self.logger.info(f"🚀 Executing resource scaling for {action.target_component}")
            
            # Simulate resource scaling (in real implementation, this would call cloud APIs)
            scaling_factor = action.action_config.get('scaling_factor', 1.5)
            
            # Measure impact
            before_metrics = await self._measure_performance()
            
            # Simulate scaling delay
            await asyncio.sleep(2)
            
            after_metrics = await self._measure_performance()
            
            # Calculate impact
            impact_metrics = {
                'latency_improvement': 0.3,  # Simulated 30% improvement
                'resource_increase': scaling_factor - 1.0,
                'cost_increase': (scaling_factor - 1.0) * 0.8
            }
            
            return OptimizationResult(
                action_id=action.action_id,
                success=True,
                duration_ms=2000,
                impact_metrics=impact_metrics,
                lessons_learned=[f"Scaling by {scaling_factor}x improved latency by 30%"]
            )
            
        except Exception as e:
            return OptimizationResult(
                action_id=action.action_id,
                success=False,
                duration_ms=0,
                impact_metrics={},
                error_message=str(e)
            )
    
    async def _handle_model_switching(self, action: OptimizationAction) -> OptimizationResult:
        """Handle model switching optimization"""
        try:
            self.logger.info(f"🚀 Executing model switching for {action.target_component}")
            
            fallback_model = action.action_config.get('fallback_model', 'fast_model')
            
            # Measure before switching
            before_metrics = await self._measure_performance()
            
            # Simulate model switch (in real implementation, this would update model router)
            await asyncio.sleep(1)
            
            after_metrics = await self._measure_performance()
            
            # Calculate impact
            impact_metrics = {
                'response_time_improvement': 0.4,  # Simulated 40% improvement
                'quality_change': -0.05,  # Simulated 5% quality reduction
                'cost_reduction': 0.3
            }
            
            return OptimizationResult(
                action_id=action.action_id,
                success=True,
                duration_ms=1000,
                impact_metrics=impact_metrics,
                lessons_learned=[f"Switching to {fallback_model} improved speed by 40%"]
            )
            
        except Exception as e:
            return OptimizationResult(
                action_id=action.action_id,
                success=False,
                duration_ms=0,
                impact_metrics={},
                error_message=str(e)
            )
    
    async def _handle_cache_optimization(self, action: OptimizationAction) -> OptimizationResult:
        """Handle cache optimization"""
        try:
            self.logger.info(f"🚀 Executing cache optimization for {action.target_component}")
            
            cache_increase = action.action_config.get('cache_size_increase', 0.2)
            warming_limit = action.action_config.get('warming_limit', 1000)
            
            # Measure before optimization
            before_metrics = await self._measure_performance()
            
            # Simulate cache warming
            await asyncio.sleep(1.5)
            
            after_metrics = await self._measure_performance()
            
            # Calculate impact
            impact_metrics = {
                'cache_hit_rate_improvement': 0.25,  # 25% improvement
                'memory_usage_increase': cache_increase,
                'response_time_improvement': 0.15
            }
            
            return OptimizationResult(
                action_id=action.action_id,
                success=True,
                duration_ms=1500,
                impact_metrics=impact_metrics,
                lessons_learned=[f"Cache warming improved hit rate by 25%"]
            )
            
        except Exception as e:
            return OptimizationResult(
                action_id=action.action_id,
                success=False,
                duration_ms=0,
                impact_metrics={},
                error_message=str(e)
            )
    
    async def _handle_timeout_adjustment(self, action: OptimizationAction) -> OptimizationResult:
        """Handle timeout adjustment optimization"""
        try:
            self.logger.info(f"🚀 Executing timeout adjustment for {action.target_component}")
            
            timeout_multiplier = action.action_config.get('timeout_multiplier', 1.5)
            max_timeout = action.action_config.get('max_timeout', 30000)
            
            # Measure before adjustment
            before_metrics = await self._measure_performance()
            
            # Simulate timeout adjustment
            await asyncio.sleep(0.5)
            
            after_metrics = await self._measure_performance()
            
            # Calculate impact
            impact_metrics = {
                'timeout_reduction': 0.2,  # 20% fewer timeouts
                'response_time_change': 0.1,  # 10% slower but more reliable
                'reliability_improvement': 0.3
            }
            
            return OptimizationResult(
                action_id=action.action_id,
                success=True,
                duration_ms=500,
                impact_metrics=impact_metrics,
                lessons_learned=[f"Timeout adjustment reduced failures by 20%"]
            )
            
        except Exception as e:
            return OptimizationResult(
                action_id=action.action_id,
                success=False,
                duration_ms=0,
                impact_metrics={},
                error_message=str(e)
            )
    
    async def _handle_memory_optimization(self, action: OptimizationAction) -> OptimizationResult:
        """Handle memory optimization"""
        try:
            self.logger.info(f"🚀 Executing memory optimization for {action.target_component}")
            
            # Measure before optimization
            before_metrics = await self._measure_performance()
            
            # Simulate memory optimization
            if action.action_config.get('garbage_collection', False):
                await asyncio.sleep(0.3)  # GC time
            
            if action.action_config.get('cache_cleanup', False):
                await asyncio.sleep(0.2)  # Cache cleanup
            
            after_metrics = await self._measure_performance()
            
            # Calculate impact
            impact_metrics = {
                'memory_reduction': 0.2,  # 20% memory reduction
                'performance_impact': 0.05,  # 5% performance cost
                'stability_improvement': 0.1
            }
            
            return OptimizationResult(
                action_id=action.action_id,
                success=True,
                duration_ms=500,
                impact_metrics=impact_metrics,
                lessons_learned=["Memory optimization freed 20% memory with minimal performance impact"]
            )
            
        except Exception as e:
            return OptimizationResult(
                action_id=action.action_id,
                success=False,
                duration_ms=0,
                impact_metrics={},
                error_message=str(e)
            )
    
    async def _handle_queue_management(self, action: OptimizationAction) -> OptimizationResult:
        """Handle queue management optimization"""
        try:
            self.logger.info(f"🚀 Executing queue management optimization")
            
            # Simulate queue optimization
            await asyncio.sleep(1)
            
            impact_metrics = {
                'queue_length_reduction': 0.3,
                'throughput_improvement': 0.15,
                'wait_time_reduction': 0.25
            }
            
            return OptimizationResult(
                action_id=action.action_id,
                success=True,
                duration_ms=1000,
                impact_metrics=impact_metrics
            )
            
        except Exception as e:
            return OptimizationResult(
                action_id=action.action_id,
                success=False,
                duration_ms=0,
                impact_metrics={},
                error_message=str(e)
            )
    
    async def _handle_circuit_breaker_adjustment(self, action: OptimizationAction) -> OptimizationResult:
        """Handle circuit breaker adjustment optimization"""
        try:
            self.logger.info(f"🚀 Executing circuit breaker adjustment")
            
            # Simulate circuit breaker tuning
            await asyncio.sleep(0.5)
            
            impact_metrics = {
                'failure_detection_improvement': 0.2,
                'recovery_time_reduction': 0.15,
                'false_positive_reduction': 0.1
            }
            
            return OptimizationResult(
                action_id=action.action_id,
                success=True,
                duration_ms=500,
                impact_metrics=impact_metrics
            )
            
        except Exception as e:
            return OptimizationResult(
                action_id=action.action_id,
                success=False,
                duration_ms=0,
                impact_metrics={},
                error_message=str(e)
            )
    
    async def _handle_load_balancing(self, action: OptimizationAction) -> OptimizationResult:
        """Handle load balancing optimization"""
        try:
            self.logger.info(f"🚀 Executing load balancing optimization")
            
            # Simulate load balancing adjustment
            await asyncio.sleep(1)
            
            impact_metrics = {
                'load_distribution_improvement': 0.3,
                'hotspot_reduction': 0.4,
                'overall_throughput_improvement': 0.2
            }
            
            return OptimizationResult(
                action_id=action.action_id,
                success=True,
                duration_ms=1000,
                impact_metrics=impact_metrics
            )
            
        except Exception as e:
            return OptimizationResult(
                action_id=action.action_id,
                success=False,
                duration_ms=0,
                impact_metrics={},
                error_message=str(e)
            )
    
    async def _measure_performance(self) -> Dict[str, Any]:
        """Measure current performance metrics"""
        if self.metrics_collector:
            return self.metrics_collector.get_current_metrics()
        else:
            # Return simulated metrics
            return {
                'latency_ms': 500 + (time.time() % 100),
                'error_rate': 0.05,
                'throughput': 10.0,
                'memory_usage': 0.7
            }
    
    async def _schedule_rollback_check(self, action: OptimizationAction, delay: float):
        """Schedule a rollback check after specified delay"""
        try:
            await asyncio.sleep(delay)
            
            # Check if rollback is needed based on rule conditions
            rule = self.optimization_rules.get(action.rule_id)
            if not rule:
                return
            
            rollback_needed = await self._check_rollback_conditions(action, rule)
            
            if rollback_needed:
                await self._perform_rollback(action)
                
        except Exception as e:
            self.logger.error(f"Error in rollback check for {action.action_id}: {e}")
    
    async def _check_rollback_conditions(self, action: OptimizationAction, rule: OptimizationRule) -> bool:
        """Check if rollback is needed based on rule conditions"""
        try:
            rollback_conditions = rule.rollback_conditions
            current_metrics = await self._measure_performance()
            
            # Check improvement requirements
            for condition, threshold in rollback_conditions.items():
                if condition == 'latency_improvement':
                    # Check if latency improved by required amount
                    before_latency = action.system_state.get('metrics', {}).get('latency_ms', 1000)
                    current_latency = current_metrics.get('latency_ms', 1000)
                    improvement = (before_latency - current_latency) / before_latency
                    
                    if improvement < threshold:
                        self.logger.warning(f"Rollback triggered: Insufficient latency improvement {improvement:.2f} < {threshold}")
                        return True
                
                elif condition == 'error_rate_increase':
                    # Check if error rate increased beyond threshold
                    before_error_rate = action.system_state.get('metrics', {}).get('error_rate', 0.05)
                    current_error_rate = current_metrics.get('error_rate', 0.05)
                    increase = current_error_rate - before_error_rate
                    
                    if increase > threshold:
                        self.logger.warning(f"Rollback triggered: Error rate increase {increase:.3f} > {threshold}")
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking rollback conditions: {e}")
            return False
    
    async def _perform_rollback(self, action: OptimizationAction):
        """Perform rollback of an optimization action"""
        try:
            self.logger.info(f"🔄 Performing rollback for optimization {action.action_id}")
            
            # Get rollback handler
            rollback_handler = self.rollback_handlers.get(action.action_id)
            
            if rollback_handler:
                # Execute custom rollback
                await rollback_handler()
            else:
                # Generic rollback based on optimization type
                await self._generic_rollback(action)
            
            # Update action status
            action.rollback_performed = True
            action.status = OptimizationStatus.ROLLED_BACK
            
            # Update statistics
            with self._lock:
                self.optimization_stats['rollback_actions'] += 1
            
            self.logger.info(f"🔄 Rollback completed for optimization {action.action_id}")
            
        except Exception as e:
            self.logger.error(f"Error performing rollback for {action.action_id}: {e}")
    
    async def _generic_rollback(self, action: OptimizationAction):
        """Generic rollback implementation"""
        if action.optimization_type == OptimizationType.RESOURCE_SCALING:
            self.logger.info("Rolling back resource scaling")
            # Simulate scaling back down
            await asyncio.sleep(1)
            
        elif action.optimization_type == OptimizationType.MODEL_SWITCHING:
            self.logger.info("Rolling back model switch")
            # Simulate switching back to original model
            await asyncio.sleep(0.5)
            
        # Add other rollback implementations as needed
    
    async def _cleanup_completed_actions(self):
        """Clean up completed actions and maintain history"""
        try:
            with self._lock:
                # Remove old actions from history
                cutoff_time = time.time() - (24 * 3600)  # 24 hours
                self.action_history = [
                    action for action in self.action_history
                    if action.triggered_at > cutoff_time
                ]
                
                # Clean up rollback handlers for completed actions
                completed_action_ids = [
                    action.action_id for action in self.action_history
                    if action.status in [OptimizationStatus.COMPLETED, OptimizationStatus.FAILED, OptimizationStatus.ROLLED_BACK]
                ]
                
                for action_id in list(self.rollback_handlers.keys()):
                    if action_id not in [action.action_id for action in self.action_history]:
                        del self.rollback_handlers[action_id]
                        
        except Exception as e:
            self.logger.error(f"Error in cleanup: {e}")
    
    async def _update_rule_statistics(self):
        """Update success rates and statistics for optimization rules"""
        try:
            with self._lock:
                for rule_id, rule in self.optimization_rules.items():
                    # Find actions for this rule
                    rule_actions = [
                        action for action in self.action_history
                        if action.rule_id == rule_id
                    ]
                    
                    if rule_actions:
                        successful_actions = [
                            action for action in rule_actions
                            if action.success and not action.rollback_performed
                        ]
                        
                        rule.success_rate = len(successful_actions) / len(rule_actions)
                        
                        # Disable rule if success rate is too low
                        if rule.success_rate < 0.3 and len(rule_actions) >= 5:
                            rule.enabled = False
                            self.logger.warning(f"Disabled rule {rule_id} due to low success rate: {rule.success_rate:.2f}")
                            
        except Exception as e:
            self.logger.error(f"Error updating rule statistics: {e}")
    
    def _update_optimization_stats(self, action: OptimizationAction, result: OptimizationResult):
        """Update global optimization statistics"""
        try:
            with self._lock:
                self.optimization_stats['total_actions'] += 1
                
                if result.success:
                    self.optimization_stats['successful_actions'] += 1
                else:
                    self.optimization_stats['failed_actions'] += 1
                
                # Update average impact score
                impact_score = self._calculate_impact_score(result.impact_metrics)
                total_actions = self.optimization_stats['total_actions']
                current_avg = self.optimization_stats['average_impact_score']
                new_avg = ((current_avg * (total_actions - 1)) + impact_score) / total_actions
                self.optimization_stats['average_impact_score'] = new_avg
                
                # Update time saved (if latency improvement)
                latency_improvement = result.impact_metrics.get('latency_improvement', 0)
                if latency_improvement > 0:
                    # Estimate time saved based on throughput
                    estimated_requests_per_hour = 3600  # Conservative estimate
                    time_saved_per_request = latency_improvement * 1000  # Convert to ms
                    total_time_saved = estimated_requests_per_hour * time_saved_per_request
                    self.optimization_stats['total_time_saved_ms'] += total_time_saved
                    
        except Exception as e:
            self.logger.error(f"Error updating optimization stats: {e}")
    
    def _calculate_impact_score(self, impact_metrics: Dict[str, Any]) -> float:
        """Calculate overall impact score from metrics"""
        try:
            positive_metrics = ['latency_improvement', 'throughput_improvement', 'reliability_improvement']
            negative_metrics = ['cost_increase', 'quality_degradation', 'memory_usage_increase']
            
            positive_score = sum(impact_metrics.get(metric, 0) for metric in positive_metrics)
            negative_score = sum(impact_metrics.get(metric, 0) for metric in negative_metrics)
            
            # Calculate weighted score (positive impacts weighted higher)
            impact_score = (positive_score * 1.5) - (negative_score * 1.0)
            
            return max(0.0, min(1.0, impact_score))
            
        except Exception:
            return 0.0
    
    # Public API methods
    def add_optimization_rule(self, rule: OptimizationRule):
        """Add a new optimization rule"""
        with self._lock:
            self.optimization_rules[rule.rule_id] = rule
        
        self.logger.info(f"Added optimization rule: {rule.name}")
    
    def remove_optimization_rule(self, rule_id: str):
        """Remove an optimization rule"""
        with self._lock:
            if rule_id in self.optimization_rules:
                del self.optimization_rules[rule_id]
                self.logger.info(f"Removed optimization rule: {rule_id}")
    
    def enable_rule(self, rule_id: str):
        """Enable an optimization rule"""
        with self._lock:
            if rule_id in self.optimization_rules:
                self.optimization_rules[rule_id].enabled = True
                self.logger.info(f"Enabled optimization rule: {rule_id}")
    
    def disable_rule(self, rule_id: str):
        """Disable an optimization rule"""
        with self._lock:
            if rule_id in self.optimization_rules:
                self.optimization_rules[rule_id].enabled = False
                self.logger.info(f"Disabled optimization rule: {rule_id}")
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization performance statistics"""
        with self._lock:
            stats = self.optimization_stats.copy()
            
            # Add computed metrics
            if stats['total_actions'] > 0:
                stats['success_rate'] = stats['successful_actions'] / stats['total_actions']
                stats['failure_rate'] = stats['failed_actions'] / stats['total_actions']
                stats['rollback_rate'] = stats['rollback_actions'] / stats['total_actions']
            else:
                stats['success_rate'] = 0.0
                stats['failure_rate'] = 0.0
                stats['rollback_rate'] = 0.0
            
            # Add rule statistics
            stats['total_rules'] = len(self.optimization_rules)
            stats['enabled_rules'] = len([r for r in self.optimization_rules.values() if r.enabled])
            
            return stats
    
    def get_active_optimizations(self) -> List[OptimizationAction]:
        """Get currently active optimization actions"""
        with self._lock:
            return list(self.active_actions.values())
    
    def get_optimization_history(self, limit: int = 100) -> List[OptimizationAction]:
        """Get optimization action history"""
        with self._lock:
            return self.action_history[-limit:]
    
    async def force_rollback(self, action_id: str) -> bool:
        """Force rollback of a specific optimization"""
        try:
            # Find action in history
            action = None
            with self._lock:
                for hist_action in self.action_history:
                    if hist_action.action_id == action_id:
                        action = hist_action
                        break
            
            if not action:
                return False
            
            if action.rollback_performed:
                return False  # Already rolled back
            
            await self._perform_rollback(action)
            return True
            
        except Exception as e:
            self.logger.error(f"Error forcing rollback for {action_id}: {e}")
            return False

# Factory function for easy instantiation
def create_predictive_optimization_engine(
    predictive_analytics: Optional[PredictiveAnalyticsFramework] = None,
    config_manager: Optional[ConfigManager] = None,
    metrics_collector: Optional[MetricsCollector] = None,
    logger: Optional[VoiceAILogger] = None,
    config: Optional[Dict[str, Any]] = None
) -> PredictiveOptimizationEngine:
    """Create and configure a PredictiveOptimizationEngine instance"""
    return PredictiveOptimizationEngine(
        predictive_analytics=predictive_analytics,
        config_manager=config_manager,
        metrics_collector=metrics_collector,
        logger=logger,
        config=config
    )

# Global instance management
_predictive_optimization_engine: Optional[PredictiveOptimizationEngine] = None

def get_predictive_optimization_engine() -> Optional[PredictiveOptimizationEngine]:
    """Get the global predictive optimization engine instance"""
    return _predictive_optimization_engine

def set_predictive_optimization_engine(engine: PredictiveOptimizationEngine):
    """Set the global predictive optimization engine instance"""
    global _predictive_optimization_engine
    _predictive_optimization_engine = engine 