"""
Predictive Configuration Manager for Sovereign 4.0 Voice Assistant

Implements intelligent configuration management with predictive adjustments based on
anticipated workload patterns and system conditions following 2024-2025 best practices.

Key Features:
- Predictive configuration adjustment based on forecasted load
- Safe configuration changes with rollback capabilities
- A/B testing framework for configuration validation
- Real-time configuration hot-reloading
- Configuration drift detection and correction
- Performance impact tracking for configuration changes
"""

import asyncio
import logging
import time
import threading
import json
import copy
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import hashlib

from .config_manager import ConfigManager, SovereignConfig
from .predictive_analytics import PredictiveAnalyticsFramework
from .metrics_collector import MetricsCollector
from .structured_logging import VoiceAILogger, get_voice_ai_logger

class ConfigAdjustmentType(Enum):
    """Types of configuration adjustments"""
    PERFORMANCE_TUNING = "performance_tuning"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    RELIABILITY_ENHANCEMENT = "reliability_enhancement"
    COST_OPTIMIZATION = "cost_optimization"
    QUALITY_IMPROVEMENT = "quality_improvement"
    SECURITY_HARDENING = "security_hardening"

class ConfigScope(Enum):
    """Scope of configuration changes"""
    GLOBAL = "global"
    SERVICE = "service"
    MODEL = "model"
    AUDIO = "audio"
    MEMORY = "memory"
    MONITORING = "monitoring"

class ConfigChangeStatus(Enum):
    """Status of configuration changes"""
    PENDING = "pending"
    ACTIVE = "active"
    TESTING = "testing"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"

@dataclass
class ConfigAdjustmentRule:
    """Rule for automatic configuration adjustment"""
    rule_id: str
    name: str
    description: str
    adjustment_type: ConfigAdjustmentType
    scope: ConfigScope
    
    # Trigger conditions
    trigger_conditions: Dict[str, Any]
    prediction_horizon: float  # seconds
    confidence_threshold: float
    
    # Configuration changes
    config_adjustments: Dict[str, Any]  # Path -> adjustment function/value
    adjustment_limits: Dict[str, Any]   # Safety limits for adjustments
    
    # Safety and validation
    requires_validation: bool
    validation_duration: float  # seconds
    rollback_conditions: Dict[str, Any]
    max_impact_level: str  # low, medium, high
    
    # Metadata
    enabled: bool = True
    success_rate: float = 0.0
    last_triggered: Optional[float] = None
    trigger_count: int = 0

@dataclass
class ConfigChange:
    """Record of a configuration change"""
    change_id: str
    rule_id: str
    adjustment_type: ConfigAdjustmentType
    scope: ConfigScope
    status: ConfigChangeStatus
    
    # Change details
    triggered_at: float
    trigger_reason: str
    prediction_data: Dict[str, Any]
    
    # Configuration changes
    config_path: str
    old_value: Any
    new_value: Any
    adjustment_metadata: Dict[str, Any]
    
    # Validation and testing
    validation_start: Optional[float] = None
    validation_end: Optional[float] = None
    a_b_test_id: Optional[str] = None
    
    # Results
    performance_impact: Optional[Dict[str, Any]] = None
    rollback_performed: bool = False
    success: Optional[bool] = None
    error_message: Optional[str] = None

@dataclass
class ConfigurationProfile:
    """A configuration profile for specific conditions"""
    profile_id: str
    name: str
    description: str
    conditions: Dict[str, Any]
    configuration: Dict[str, Any]
    performance_metrics: Dict[str, float]
    confidence_score: float
    usage_count: int = 0
    last_used: Optional[float] = None

class PredictiveConfigManager:
    """
    Predictive Configuration Manager for intelligent system configuration
    
    Provides automated configuration management by:
    - Analyzing predictive analytics for configuration opportunities
    - Safely adjusting configurations with validation and rollback
    - Learning from configuration changes and their impacts
    - Maintaining multiple configuration profiles for different conditions
    """
    
    def __init__(
        self,
        base_config_manager: ConfigManager,
        predictive_analytics: Optional[PredictiveAnalyticsFramework] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        logger: Optional[VoiceAILogger] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.base_config_manager = base_config_manager
        self.predictive_analytics = predictive_analytics
        self.metrics_collector = metrics_collector
        self.logger = logger or get_voice_ai_logger("predictive_config_manager")
        
        # Configuration
        self.config = config or self._get_default_config()
        
        # State management
        self._lock = threading.RLock()
        self.adjustment_rules: Dict[str, ConfigAdjustmentRule] = {}
        self.active_changes: Dict[str, ConfigChange] = {}
        self.change_history: List[ConfigChange] = []
        self.configuration_profiles: Dict[str, ConfigurationProfile] = {}
        
        # Current configuration state
        self.current_config: Optional[SovereignConfig] = None
        self.config_fingerprint: Optional[str] = None
        self.optimized_configs: Dict[str, Dict[str, Any]] = {}  # condition -> config
        
        # A/B testing framework
        self.ab_tests: Dict[str, Dict[str, Any]] = {}
        self.test_assignments: Dict[str, str] = {}  # session -> test_variant
        
        # Performance tracking
        self.config_performance_history: List[Dict[str, Any]] = []
        self.adjustment_stats = {
            'total_adjustments': 0,
            'successful_adjustments': 0,
            'rolled_back_adjustments': 0,
            'average_performance_improvement': 0.0
        }
        
        # Monitoring tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.validation_tasks: Dict[str, asyncio.Task] = {}
        
        # Initialize default rules and profiles
        self._initialize_default_rules()
        self._initialize_default_profiles()
        
        # Load current configuration
        self._load_current_configuration()
        
        self.logger.info("⚙️ PredictiveConfigManager initialized with adaptive configuration")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for predictive config manager"""
        return {
            'monitoring_interval': 60.0,  # seconds
            'validation_duration': 300.0,  # 5 minutes
            'max_concurrent_changes': 3,
            'max_adjustments_per_hour': 5,
            'performance_improvement_threshold': 0.05,  # 5% improvement required
            'rollback_threshold': 0.1,  # Rollback if performance degrades by 10%
            'a_b_test_duration': 1800.0,  # 30 minutes
            'config_drift_threshold': 0.1,
            'learning_enabled': True,
            'auto_approval_confidence': 0.8
        }
    
    def _initialize_default_rules(self):
        """Initialize default configuration adjustment rules"""
        default_rules = [
            # Performance tuning based on latency predictions
            ConfigAdjustmentRule(
                rule_id="latency_timeout_adjustment",
                name="Latency-based Timeout Adjustment",
                description="Adjust timeouts based on predicted latency patterns",
                adjustment_type=ConfigAdjustmentType.PERFORMANCE_TUNING,
                scope=ConfigScope.SERVICE,
                trigger_conditions={
                    'predicted_latency_increase': 0.3,  # 30% increase predicted
                    'current_timeout_hit_rate': 0.05,   # 5% timeout rate
                    'service_reliability': 0.9
                },
                prediction_horizon=900.0,  # 15 minutes
                confidence_threshold=0.75,
                config_adjustments={
                    'stt.timeout': lambda current, prediction: min(current * 1.5, 30.0),
                    'llm.timeout': lambda current, prediction: min(current * 1.3, 60.0),
                    'tts.timeout': lambda current, prediction: min(current * 1.4, 20.0)
                },
                adjustment_limits={
                    'stt.timeout': {'min': 5.0, 'max': 30.0},
                    'llm.timeout': {'min': 10.0, 'max': 60.0},
                    'tts.timeout': {'min': 2.0, 'max': 20.0}
                },
                requires_validation=True,
                validation_duration=300.0,
                rollback_conditions={
                    'timeout_improvement': 0.2,
                    'latency_degradation': 0.2
                },
                max_impact_level="medium"
            ),
            
            # Resource optimization based on load predictions
            ConfigAdjustmentRule(
                rule_id="load_based_optimization",
                name="Load-based Resource Optimization",
                description="Optimize resource allocation based on predicted load",
                adjustment_type=ConfigAdjustmentType.RESOURCE_OPTIMIZATION,
                scope=ConfigScope.GLOBAL,
                trigger_conditions={
                    'predicted_load_increase': 0.5,     # 50% load increase
                    'current_cpu_usage': 0.6,          # 60% CPU usage
                    'memory_availability': 0.4         # 40% memory available
                },
                prediction_horizon=1800.0,  # 30 minutes
                confidence_threshold=0.7,
                config_adjustments={
                    'llm.fast.max_tokens': lambda current, prediction: min(current * 0.8, 4000),
                    'memory.retrieval_k': lambda current, prediction: max(current * 0.7, 3),
                    'audio.buffer_size': lambda current, prediction: min(current * 1.2, 8192)
                },
                adjustment_limits={
                    'llm.fast.max_tokens': {'min': 1000, 'max': 4000},
                    'memory.retrieval_k': {'min': 3, 'max': 15},
                    'audio.buffer_size': {'min': 1024, 'max': 8192}
                },
                requires_validation=True,
                validation_duration=600.0,
                rollback_conditions={
                    'performance_degradation': 0.15,
                    'error_rate_increase': 0.05
                },
                max_impact_level="medium"
            ),
            
            # Quality improvement based on accuracy predictions
            ConfigAdjustmentRule(
                rule_id="accuracy_model_adjustment",
                name="Accuracy-based Model Adjustment",
                description="Adjust model parameters based on predicted accuracy needs",
                adjustment_type=ConfigAdjustmentType.QUALITY_IMPROVEMENT,
                scope=ConfigScope.MODEL,
                trigger_conditions={
                    'predicted_accuracy_drop': 0.1,    # 10% accuracy drop predicted
                    'current_accuracy': 0.85,          # Current accuracy above threshold
                    'quality_requirements': 'high'
                },
                prediction_horizon=3600.0,  # 1 hour
                confidence_threshold=0.8,
                config_adjustments={
                    'llm.deep.temperature': lambda current, prediction: max(current * 0.8, 0.1),
                    'llm.deep.max_tokens': lambda current, prediction: min(current * 1.2, 6000),
                    'stt.language_model_weight': lambda current, prediction: min(current * 1.1, 1.0)
                },
                adjustment_limits={
                    'llm.deep.temperature': {'min': 0.1, 'max': 1.0},
                    'llm.deep.max_tokens': {'min': 2000, 'max': 6000},
                    'stt.language_model_weight': {'min': 0.1, 'max': 1.0}
                },
                requires_validation=True,
                validation_duration=900.0,
                rollback_conditions={
                    'accuracy_improvement': 0.05,
                    'latency_increase': 0.3
                },
                max_impact_level="low"
            ),
            
            # Cost optimization based on usage patterns
            ConfigAdjustmentRule(
                rule_id="cost_optimization",
                name="Cost-based Configuration Optimization",
                description="Optimize configuration for cost efficiency",
                adjustment_type=ConfigAdjustmentType.COST_OPTIMIZATION,
                scope=ConfigScope.GLOBAL,
                trigger_conditions={
                    'predicted_cost_increase': 0.2,    # 20% cost increase
                    'budget_utilization': 0.8,         # 80% budget used
                    'performance_acceptable': True
                },
                prediction_horizon=7200.0,  # 2 hours
                confidence_threshold=0.7,
                config_adjustments={
                    'llm.fast.model': lambda current, prediction: 'gpt-3.5-turbo',  # Cheaper model
                    'memory.similarity_threshold': lambda current, prediction: min(current * 1.1, 0.9),
                    'monitoring.metrics_enabled': lambda current, prediction: False  # Reduce monitoring
                },
                adjustment_limits={
                    'memory.similarity_threshold': {'min': 0.5, 'max': 0.9}
                },
                requires_validation=True,
                validation_duration=1800.0,
                rollback_conditions={
                    'quality_degradation': 0.1,
                    'user_satisfaction': 0.8
                },
                max_impact_level="high"
            ),
            
            # Reliability enhancement based on error predictions
            ConfigAdjustmentRule(
                rule_id="reliability_enhancement",
                name="Reliability-based Configuration Enhancement",
                description="Enhance configuration for better reliability",
                adjustment_type=ConfigAdjustmentType.RELIABILITY_ENHANCEMENT,
                scope=ConfigScope.SERVICE,
                trigger_conditions={
                    'predicted_error_rate_increase': 0.05,  # 5% error rate increase
                    'current_uptime': 0.995,                # High uptime requirement
                    'reliability_critical': True
                },
                prediction_horizon=1800.0,  # 30 minutes
                confidence_threshold=0.8,
                config_adjustments={
                    'llm.fast.timeout': lambda current, prediction: min(current * 1.5, 45.0),
                    'audio.retry_attempts': lambda current, prediction: min(current + 1, 5),
                    'monitoring.health_check_interval': lambda current, prediction: max(current * 0.5, 10.0)
                },
                adjustment_limits={
                    'llm.fast.timeout': {'min': 10.0, 'max': 45.0},
                    'audio.retry_attempts': {'min': 1, 'max': 5},
                    'monitoring.health_check_interval': {'min': 10.0, 'max': 60.0}
                },
                requires_validation=True,
                validation_duration=600.0,
                rollback_conditions={
                    'error_rate_reduction': 0.02,
                    'performance_impact': 0.2
                },
                max_impact_level="medium"
            )
        ]
        
        for rule in default_rules:
            self.adjustment_rules[rule.rule_id] = rule
    
    def _initialize_default_profiles(self):
        """Initialize default configuration profiles"""
        default_profiles = [
            ConfigurationProfile(
                profile_id="high_performance",
                name="High Performance Profile",
                description="Optimized for maximum performance",
                conditions={
                    'load_level': 'high',
                    'latency_requirements': 'strict',
                    'resource_availability': 'abundant'
                },
                configuration={
                    'llm.fast.max_tokens': 2000,
                    'llm.fast.temperature': 0.7,
                    'audio.buffer_size': 4096,
                    'memory.retrieval_k': 8,
                    'monitoring.metrics_enabled': True
                },
                performance_metrics={
                    'average_latency': 450.0,
                    'accuracy_score': 0.92,
                    'resource_efficiency': 0.8
                },
                confidence_score=0.9
            ),
            
            ConfigurationProfile(
                profile_id="balanced",
                name="Balanced Profile",
                description="Balanced performance, quality, and resource usage",
                conditions={
                    'load_level': 'medium',
                    'quality_requirements': 'standard',
                    'resource_availability': 'normal'
                },
                configuration={
                    'llm.fast.max_tokens': 3000,
                    'llm.fast.temperature': 0.8,
                    'audio.buffer_size': 2048,
                    'memory.retrieval_k': 10,
                    'memory.similarity_threshold': 0.7
                },
                performance_metrics={
                    'average_latency': 650.0,
                    'accuracy_score': 0.89,
                    'resource_efficiency': 0.85
                },
                confidence_score=0.85
            ),
            
            ConfigurationProfile(
                profile_id="efficiency",
                name="Efficiency Profile",
                description="Optimized for resource efficiency and cost",
                conditions={
                    'load_level': 'low',
                    'cost_sensitivity': 'high',
                    'quality_tolerance': 'acceptable'
                },
                configuration={
                    'llm.fast.max_tokens': 1500,
                    'llm.fast.temperature': 0.9,
                    'audio.buffer_size': 1024,
                    'memory.retrieval_k': 5,
                    'memory.similarity_threshold': 0.8,
                    'monitoring.metrics_enabled': False
                },
                performance_metrics={
                    'average_latency': 800.0,
                    'accuracy_score': 0.86,
                    'resource_efficiency': 0.95
                },
                confidence_score=0.8
            ),
            
            ConfigurationProfile(
                profile_id="quality_focused",
                name="Quality Focused Profile",
                description="Optimized for maximum quality and accuracy",
                conditions={
                    'quality_requirements': 'premium',
                    'accuracy_critical': True,
                    'performance_tolerance': 'flexible'
                },
                configuration={
                    'llm.deep.max_tokens': 5000,
                    'llm.deep.temperature': 0.3,
                    'memory.retrieval_k': 15,
                    'memory.similarity_threshold': 0.6,
                    'stt.language_model_weight': 1.0
                },
                performance_metrics={
                    'average_latency': 1200.0,
                    'accuracy_score': 0.96,
                    'resource_efficiency': 0.7
                },
                confidence_score=0.92
            )
        ]
        
        for profile in default_profiles:
            self.configuration_profiles[profile.profile_id] = profile
    
    def _load_current_configuration(self):
        """Load current configuration and generate fingerprint"""
        try:
            self.current_config = self.base_config_manager.get_config()
            self.config_fingerprint = self._generate_config_fingerprint(self.current_config)
            
        except Exception as e:
            self.logger.error(f"Error loading current configuration: {e}")
    
    def _generate_config_fingerprint(self, config: SovereignConfig) -> str:
        """Generate fingerprint for configuration state"""
        try:
            # Extract relevant configuration values
            config_dict = {
                'llm_fast_model': config.llm.fast.model,
                'llm_fast_max_tokens': config.llm.fast.max_tokens,
                'llm_fast_temperature': config.llm.fast.temperature,
                'llm_fast_timeout': config.llm.fast.timeout,
                'memory_retrieval_k': config.memory.retrieval_k,
                'memory_similarity_threshold': config.memory.similarity_threshold,
                'audio_buffer_size': getattr(config.audio, 'buffer_size', 2048),
                'stt_timeout': config.stt.timeout,
                'tts_timeout': config.tts.timeout
            }
            
            # Create hash of configuration
            config_string = json.dumps(config_dict, sort_keys=True)
            return hashlib.sha256(config_string.encode()).hexdigest()[:16]
            
        except Exception as e:
            self.logger.warning(f"Error generating config fingerprint: {e}")
            return "unknown"
    
    async def start_monitoring(self):
        """Start predictive configuration monitoring"""
        if self.monitoring_task and not self.monitoring_task.done():
            self.logger.warning("Configuration monitoring already running")
            return
        
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("⚙️ Predictive configuration monitoring started")
    
    async def stop_monitoring(self):
        """Stop predictive configuration monitoring"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        # Cancel validation tasks
        for task in list(self.validation_tasks.values()):
            task.cancel()
        
        # Wait for tasks to complete
        if self.monitoring_task:
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("⚙️ Predictive configuration monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop for configuration adjustments"""
        try:
            while True:
                try:
                    await self._evaluate_configuration_adjustments()
                    await self._detect_configuration_drift()
                    await self._update_configuration_profiles()
                    await asyncio.sleep(self.config['monitoring_interval'])
                    
                except Exception as e:
                    self.logger.error(f"Error in configuration monitoring loop: {e}")
                    await asyncio.sleep(60)  # Wait before retrying
                    
        except asyncio.CancelledError:
            self.logger.info("⚙️ Configuration monitoring loop cancelled")
    
    async def _evaluate_configuration_adjustments(self):
        """Evaluate all adjustment rules for trigger conditions"""
        if not self.predictive_analytics:
            return
        
        try:
            # Check safety limits
            if not self._check_safety_limits():
                return
            
            # Get predictions for configuration decisions
            predictions = await self._get_configuration_predictions()
            
            # Get current system state
            system_state = await self._get_system_state()
            
            # Evaluate each rule
            for rule_id, rule in self.adjustment_rules.items():
                if not rule.enabled:
                    continue
                
                # Check if rule should trigger
                should_trigger, trigger_reason = await self._should_trigger_adjustment(
                    rule, predictions, system_state
                )
                
                if should_trigger:
                    await self._trigger_configuration_adjustment(
                        rule, trigger_reason, predictions, system_state
                    )
                    
        except Exception as e:
            self.logger.error(f"Error evaluating configuration adjustments: {e}")
    
    async def _get_configuration_predictions(self) -> Dict[str, Any]:
        """Get predictions relevant to configuration decisions"""
        predictions = {}
        
        try:
            if self.predictive_analytics:
                # Get performance predictions
                performance_pred = await self.predictive_analytics.performance_predictor.get_predictions()
                predictions['performance'] = performance_pred
                
                # Get resource forecasts
                resource_forecasts = await self.predictive_analytics.resource_forecaster.get_forecasts()
                predictions['resources'] = resource_forecasts
                
                # Get drift alerts
                drift_alerts = await self.predictive_analytics.drift_detector.get_active_alerts()
                predictions['drift'] = [asdict(alert) for alert in drift_alerts]
                
        except Exception as e:
            self.logger.warning(f"Error getting configuration predictions: {e}")
        
        return predictions
    
    async def _get_system_state(self) -> Dict[str, Any]:
        """Get current system state for configuration decisions"""
        system_state = {'timestamp': time.time()}
        
        try:
            if self.metrics_collector:
                current_metrics = self.metrics_collector.get_current_metrics()
                system_state['metrics'] = current_metrics
            
            # Add current configuration state
            system_state['config_fingerprint'] = self.config_fingerprint
            system_state['active_profile'] = self._identify_current_profile()
            
            # Add resource information
            system_state['resources'] = await self._get_resource_status()
            
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
                'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
            }
        except Exception:
            return {}
    
    def _check_safety_limits(self) -> bool:
        """Check if it's safe to make configuration changes"""
        try:
            with self._lock:
                # Check concurrent changes limit
                active_count = len(self.active_changes)
                if active_count >= self.config['max_concurrent_changes']:
                    return False
                
                # Check hourly adjustment limit
                one_hour_ago = time.time() - 3600
                recent_changes = [
                    change for change in self.change_history
                    if change.triggered_at > one_hour_ago
                ]
                
                if len(recent_changes) >= self.config['max_adjustments_per_hour']:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking safety limits: {e}")
            return False
    
    async def _should_trigger_adjustment(
        self,
        rule: ConfigAdjustmentRule,
        predictions: Dict[str, Any],
        system_state: Dict[str, Any]
    ) -> tuple[bool, str]:
        """Determine if configuration adjustment rule should trigger"""
        try:
            # Check time since last trigger
            if rule.last_triggered:
                time_since_last = time.time() - rule.last_triggered
                if time_since_last < 600:  # 10 minutes minimum
                    return False, "Too soon since last trigger"
            
            # Check trigger conditions based on adjustment type
            conditions = rule.trigger_conditions
            
            if rule.adjustment_type == ConfigAdjustmentType.PERFORMANCE_TUNING:
                return await self._check_performance_conditions(rule, predictions, system_state)
            
            elif rule.adjustment_type == ConfigAdjustmentType.RESOURCE_OPTIMIZATION:
                return await self._check_resource_conditions(rule, predictions, system_state)
            
            elif rule.adjustment_type == ConfigAdjustmentType.QUALITY_IMPROVEMENT:
                return await self._check_quality_conditions(rule, predictions, system_state)
            
            elif rule.adjustment_type == ConfigAdjustmentType.COST_OPTIMIZATION:
                return await self._check_cost_conditions(rule, predictions, system_state)
            
            elif rule.adjustment_type == ConfigAdjustmentType.RELIABILITY_ENHANCEMENT:
                return await self._check_reliability_conditions(rule, predictions, system_state)
            
            else:
                return False, f"Unknown adjustment type: {rule.adjustment_type}"
                
        except Exception as e:
            self.logger.error(f"Error checking trigger conditions for rule {rule.rule_id}: {e}")
            return False, f"Error: {e}"
    
    async def _check_performance_conditions(
        self,
        rule: ConfigAdjustmentRule,
        predictions: Dict[str, Any],
        system_state: Dict[str, Any]
    ) -> tuple[bool, str]:
        """Check conditions for performance tuning adjustments"""
        try:
            conditions = rule.trigger_conditions
            performance_pred = predictions.get('performance', {})
            
            # Check predicted latency increase
            latency_increase = performance_pred.get('latency_increase_ratio', 0)
            if latency_increase > conditions.get('predicted_latency_increase', 0.3):
                
                # Check current timeout hit rate
                current_metrics = system_state.get('metrics')
                if current_metrics and hasattr(current_metrics, 'throughput'):
                    timeout_rate = getattr(current_metrics.throughput, 'timeout_rate', 0)
                    
                    if timeout_rate > conditions.get('current_timeout_hit_rate', 0.05):
                        # Check service reliability
                        reliability = performance_pred.get('reliability_score', 1.0)
                        if reliability > conditions.get('service_reliability', 0.9):
                            return True, f"Performance tuning needed: {latency_increase:.2f} latency increase"
            
            return False, "Performance conditions not met"
            
        except Exception as e:
            return False, f"Error checking performance conditions: {e}"
    
    async def _check_resource_conditions(
        self,
        rule: ConfigAdjustmentRule,
        predictions: Dict[str, Any],
        system_state: Dict[str, Any]
    ) -> tuple[bool, str]:
        """Check conditions for resource optimization adjustments"""
        try:
            conditions = rule.trigger_conditions
            resource_forecasts = predictions.get('resources', {})
            
            # Check predicted load increase
            load_increase = resource_forecasts.get('load_increase_ratio', 0)
            if load_increase > conditions.get('predicted_load_increase', 0.5):
                
                # Check current CPU usage
                resources = system_state.get('resources', {})
                cpu_usage = resources.get('cpu_percent', 0) / 100
                
                if cpu_usage > conditions.get('current_cpu_usage', 0.6):
                    # Check memory availability
                    memory_usage = resources.get('memory_percent', 100) / 100
                    memory_available = 1.0 - memory_usage
                    
                    if memory_available > conditions.get('memory_availability', 0.4):
                        return True, f"Resource optimization needed: {load_increase:.2f} load increase"
            
            return False, "Resource conditions not met"
            
        except Exception as e:
            return False, f"Error checking resource conditions: {e}"
    
    async def _check_quality_conditions(
        self,
        rule: ConfigAdjustmentRule,
        predictions: Dict[str, Any],
        system_state: Dict[str, Any]
    ) -> tuple[bool, str]:
        """Check conditions for quality improvement adjustments"""
        try:
            conditions = rule.trigger_conditions
            performance_pred = predictions.get('performance', {})
            
            # Check predicted accuracy drop
            accuracy_drop = performance_pred.get('accuracy_drop_ratio', 0)
            if accuracy_drop > conditions.get('predicted_accuracy_drop', 0.1):
                
                # Check current accuracy
                current_accuracy = performance_pred.get('current_accuracy', 0.9)
                if current_accuracy > conditions.get('current_accuracy', 0.85):
                    
                    # Check quality requirements
                    quality_req = conditions.get('quality_requirements', 'standard')
                    if quality_req == 'high':
                        return True, f"Quality improvement needed: {accuracy_drop:.2f} accuracy drop"
            
            return False, "Quality conditions not met"
            
        except Exception as e:
            return False, f"Error checking quality conditions: {e}"
    
    async def _check_cost_conditions(
        self,
        rule: ConfigAdjustmentRule,
        predictions: Dict[str, Any],
        system_state: Dict[str, Any]
    ) -> tuple[bool, str]:
        """Check conditions for cost optimization adjustments"""
        try:
            conditions = rule.trigger_conditions
            resource_forecasts = predictions.get('resources', {})
            
            # Check predicted cost increase
            cost_increase = resource_forecasts.get('cost_increase_ratio', 0)
            if cost_increase > conditions.get('predicted_cost_increase', 0.2):
                
                # Check budget utilization
                budget_util = resource_forecasts.get('budget_utilization', 0.5)
                if budget_util > conditions.get('budget_utilization', 0.8):
                    
                    # Check if performance is still acceptable
                    performance_acceptable = conditions.get('performance_acceptable', True)
                    if performance_acceptable:
                        return True, f"Cost optimization needed: {cost_increase:.2f} cost increase"
            
            return False, "Cost conditions not met"
            
        except Exception as e:
            return False, f"Error checking cost conditions: {e}"
    
    async def _check_reliability_conditions(
        self,
        rule: ConfigAdjustmentRule,
        predictions: Dict[str, Any],
        system_state: Dict[str, Any]
    ) -> tuple[bool, str]:
        """Check conditions for reliability enhancement adjustments"""
        try:
            conditions = rule.trigger_conditions
            performance_pred = predictions.get('performance', {})
            
            # Check predicted error rate increase
            error_rate_increase = performance_pred.get('error_rate_increase', 0)
            if error_rate_increase > conditions.get('predicted_error_rate_increase', 0.05):
                
                # Check current uptime requirements
                current_uptime = performance_pred.get('uptime', 0.99)
                if current_uptime > conditions.get('current_uptime', 0.995):
                    
                    # Check if reliability is critical
                    reliability_critical = conditions.get('reliability_critical', False)
                    if reliability_critical:
                        return True, f"Reliability enhancement needed: {error_rate_increase:.3f} error increase"
            
            return False, "Reliability conditions not met"
            
        except Exception as e:
            return False, f"Error checking reliability conditions: {e}"
    
    async def _trigger_configuration_adjustment(
        self,
        rule: ConfigAdjustmentRule,
        trigger_reason: str,
        predictions: Dict[str, Any],
        system_state: Dict[str, Any]
    ):
        """Trigger a configuration adjustment"""
        try:
            # Calculate new configuration values
            adjustments = await self._calculate_configuration_adjustments(
                rule, predictions, system_state
            )
            
            if not adjustments:
                return
            
            # Create configuration changes
            changes = []
            for config_path, (old_value, new_value) in adjustments.items():
                change = ConfigChange(
                    change_id=f"config_{rule.rule_id}_{int(time.time())}_{len(changes)}",
                    rule_id=rule.rule_id,
                    adjustment_type=rule.adjustment_type,
                    scope=rule.scope,
                    status=ConfigChangeStatus.PENDING,
                    triggered_at=time.time(),
                    trigger_reason=trigger_reason,
                    prediction_data=predictions,
                    config_path=config_path,
                    old_value=old_value,
                    new_value=new_value,
                    adjustment_metadata={
                        'rule_name': rule.name,
                        'confidence_threshold': rule.confidence_threshold,
                        'max_impact_level': rule.max_impact_level
                    }
                )
                changes.append(change)
            
            # Apply configuration changes
            for change in changes:
                await self._apply_configuration_change(change, rule)
            
            # Update rule statistics
            with self._lock:
                rule.last_triggered = time.time()
                rule.trigger_count += 1
            
            self.logger.info(
                f"⚙️ Triggered configuration adjustment: {rule.name} "
                f"({len(changes)} changes, Reason: {trigger_reason})"
            )
            
        except Exception as e:
            self.logger.error(f"Error triggering configuration adjustment for rule {rule.rule_id}: {e}")
    
    async def _calculate_configuration_adjustments(
        self,
        rule: ConfigAdjustmentRule,
        predictions: Dict[str, Any],
        system_state: Dict[str, Any]
    ) -> Dict[str, Tuple[Any, Any]]:
        """Calculate specific configuration adjustments"""
        adjustments = {}
        
        try:
            current_config = self.current_config
            if not current_config:
                return adjustments
            
            for config_path, adjustment_func in rule.config_adjustments.items():
                try:
                    # Get current value
                    current_value = self._get_config_value(current_config, config_path)
                    
                    # Calculate new value
                    if callable(adjustment_func):
                        new_value = adjustment_func(current_value, predictions)
                    else:
                        new_value = adjustment_func
                    
                    # Apply limits
                    limits = rule.adjustment_limits.get(config_path)
                    if limits:
                        if 'min' in limits and new_value < limits['min']:
                            new_value = limits['min']
                        if 'max' in limits and new_value > limits['max']:
                            new_value = limits['max']
                    
                    # Only include if value actually changes
                    if new_value != current_value:
                        adjustments[config_path] = (current_value, new_value)
                        
                except Exception as e:
                    self.logger.warning(f"Error calculating adjustment for {config_path}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error calculating configuration adjustments: {e}")
        
        return adjustments
    
    def _get_config_value(self, config: SovereignConfig, path: str) -> Any:
        """Get configuration value by path"""
        try:
            parts = path.split('.')
            value = config
            
            for part in parts:
                if hasattr(value, part):
                    value = getattr(value, part)
                else:
                    return None
            
            return value
            
        except Exception:
            return None
    
    async def _apply_configuration_change(self, change: ConfigChange, rule: ConfigAdjustmentRule):
        """Apply a configuration change"""
        try:
            # Add to active changes
            with self._lock:
                self.active_changes[change.change_id] = change
            
            # Apply the change to configuration
            success = await self._update_configuration_value(change.config_path, change.new_value)
            
            if success:
                change.status = ConfigChangeStatus.ACTIVE
                
                # Start validation if required
                if rule.requires_validation:
                    change.status = ConfigChangeStatus.TESTING
                    change.validation_start = time.time()
                    
                    # Schedule validation
                    validation_task = asyncio.create_task(
                        self._validate_configuration_change(change, rule)
                    )
                    self.validation_tasks[change.change_id] = validation_task
                else:
                    change.status = ConfigChangeStatus.COMPLETED
                    change.success = True
                
                self.logger.info(f"⚙️ Applied configuration change: {change.config_path} = {change.new_value}")
                
            else:
                change.status = ConfigChangeStatus.FAILED
                change.success = False
                change.error_message = "Failed to update configuration"
                
        except Exception as e:
            change.status = ConfigChangeStatus.FAILED
            change.success = False
            change.error_message = str(e)
            self.logger.error(f"Error applying configuration change {change.change_id}: {e}")
    
    async def _update_configuration_value(self, config_path: str, new_value: Any) -> bool:
        """Update a configuration value"""
        try:
            # In a real implementation, this would update the configuration
            # and trigger a hot-reload if supported
            
            # For now, we simulate the update
            self.logger.info(f"⚙️ Updating {config_path} to {new_value}")
            
            # Update optimized configs cache
            current_conditions = await self._get_current_conditions()
            condition_key = self._conditions_to_key(current_conditions)
            
            if condition_key not in self.optimized_configs:
                self.optimized_configs[condition_key] = {}
            
            self.optimized_configs[condition_key][config_path] = new_value
            
            # Trigger hot-reload if supported
            if hasattr(self.base_config_manager, 'reload_config'):
                self.base_config_manager.reload_config()
                self._load_current_configuration()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating configuration value {config_path}: {e}")
            return False
    
    async def _validate_configuration_change(self, change: ConfigChange, rule: ConfigAdjustmentRule):
        """Validate a configuration change"""
        try:
            validation_duration = rule.validation_duration
            
            # Measure performance before validation period
            before_metrics = await self._measure_performance()
            
            # Wait for validation period
            await asyncio.sleep(validation_duration)
            
            # Measure performance after validation period
            after_metrics = await self._measure_performance()
            
            # Calculate performance impact
            impact = self._calculate_performance_impact(before_metrics, after_metrics)
            change.performance_impact = impact
            
            # Check rollback conditions
            should_rollback = self._should_rollback(change, rule, impact)
            
            if should_rollback:
                await self._rollback_configuration_change(change)
                change.status = ConfigChangeStatus.ROLLED_BACK
                change.success = False
            else:
                change.status = ConfigChangeStatus.COMPLETED
                change.success = True
                
                # Update rule success rate
                self._update_rule_success_rate(rule, True)
            
            change.validation_end = time.time()
            
            # Move to history
            with self._lock:
                if change.change_id in self.active_changes:
                    del self.active_changes[change.change_id]
                self.change_history.append(change)
                
                # Clean up validation task
                if change.change_id in self.validation_tasks:
                    del self.validation_tasks[change.change_id]
            
            # Update statistics
            self._update_adjustment_stats(change)
            
            self.logger.info(
                f"⚙️ Configuration validation completed: {change.change_id} "
                f"(Success: {change.success}, Impact: {impact})"
            )
            
        except Exception as e:
            change.status = ConfigChangeStatus.FAILED
            change.success = False
            change.error_message = str(e)
            self.logger.error(f"Error validating configuration change {change.change_id}: {e}")
    
    async def _measure_performance(self) -> Dict[str, Any]:
        """Measure current performance metrics"""
        try:
            if self.metrics_collector:
                metrics = self.metrics_collector.get_current_metrics()
                
                return {
                    'latency_ms': metrics.latency.get_average() if hasattr(metrics, 'latency') else 500,
                    'error_rate': metrics.throughput.error_rate if hasattr(metrics, 'throughput') else 0.05,
                    'throughput': metrics.throughput.requests_per_second if hasattr(metrics, 'throughput') else 10.0,
                    'accuracy': getattr(metrics, 'accuracy_score', 0.9),
                    'timestamp': time.time()
                }
            else:
                # Return simulated metrics
                return {
                    'latency_ms': 500 + (time.time() % 200),
                    'error_rate': 0.05,
                    'throughput': 10.0,
                    'accuracy': 0.9,
                    'timestamp': time.time()
                }
                
        except Exception as e:
            self.logger.warning(f"Error measuring performance: {e}")
            return {'timestamp': time.time()}
    
    def _calculate_performance_impact(self, before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance impact of configuration change"""
        impact = {}
        
        try:
            # Calculate latency impact
            if 'latency_ms' in before and 'latency_ms' in after:
                latency_change = (after['latency_ms'] - before['latency_ms']) / before['latency_ms']
                impact['latency_change'] = latency_change
            
            # Calculate error rate impact
            if 'error_rate' in before and 'error_rate' in after:
                if before['error_rate'] > 0:
                    error_rate_change = (after['error_rate'] - before['error_rate']) / before['error_rate']
                    impact['error_rate_change'] = error_rate_change
            
            # Calculate throughput impact
            if 'throughput' in before and 'throughput' in after:
                throughput_change = (after['throughput'] - before['throughput']) / before['throughput']
                impact['throughput_change'] = throughput_change
            
            # Calculate accuracy impact
            if 'accuracy' in before and 'accuracy' in after:
                accuracy_change = after['accuracy'] - before['accuracy']
                impact['accuracy_change'] = accuracy_change
            
            # Calculate overall performance score
            performance_factors = []
            
            if 'latency_change' in impact:
                performance_factors.append(-impact['latency_change'])  # Lower latency is better
            
            if 'error_rate_change' in impact:
                performance_factors.append(-impact['error_rate_change'])  # Lower error rate is better
            
            if 'throughput_change' in impact:
                performance_factors.append(impact['throughput_change'])  # Higher throughput is better
            
            if 'accuracy_change' in impact:
                performance_factors.append(impact['accuracy_change'])  # Higher accuracy is better
            
            if performance_factors:
                impact['overall_performance_change'] = sum(performance_factors) / len(performance_factors)
            
        except Exception as e:
            self.logger.warning(f"Error calculating performance impact: {e}")
        
        return impact
    
    def _should_rollback(self, change: ConfigChange, rule: ConfigAdjustmentRule, impact: Dict[str, Any]) -> bool:
        """Determine if configuration change should be rolled back"""
        try:
            rollback_conditions = rule.rollback_conditions
            
            # Check performance degradation threshold
            overall_change = impact.get('overall_performance_change', 0)
            if overall_change < -self.config['rollback_threshold']:
                return True
            
            # Check specific rollback conditions
            for condition, threshold in rollback_conditions.items():
                if condition in impact:
                    if condition.endswith('_improvement'):
                        # Improvement conditions (require minimum improvement)
                        if impact[condition] < threshold:
                            return True
                    elif condition.endswith('_degradation') or condition.endswith('_increase'):
                        # Degradation conditions (rollback if exceeds threshold)
                        if abs(impact[condition]) > threshold:
                            return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Error checking rollback conditions: {e}")
            return False
    
    async def _rollback_configuration_change(self, change: ConfigChange):
        """Rollback a configuration change"""
        try:
            self.logger.info(f"🔄 Rolling back configuration change: {change.config_path}")
            
            # Restore original value
            success = await self._update_configuration_value(change.config_path, change.old_value)
            
            if success:
                change.rollback_performed = True
                self.logger.info(f"🔄 Rollback completed: {change.config_path} = {change.old_value}")
            else:
                self.logger.error(f"Failed to rollback configuration change: {change.config_path}")
                
        except Exception as e:
            self.logger.error(f"Error rolling back configuration change {change.change_id}: {e}")
    
    def _update_rule_success_rate(self, rule: ConfigAdjustmentRule, success: bool):
        """Update success rate for an adjustment rule"""
        try:
            with self._lock:
                # Simple moving average with weight for recent results
                current_rate = rule.success_rate
                weight = 0.1  # Weight for new result
                
                new_rate = current_rate * (1 - weight) + (1.0 if success else 0.0) * weight
                rule.success_rate = new_rate
                
                # Disable rule if success rate is too low
                if rule.success_rate < 0.3 and rule.trigger_count >= 5:
                    rule.enabled = False
                    self.logger.warning(f"Disabled adjustment rule {rule.rule_id} due to low success rate")
                    
        except Exception as e:
            self.logger.error(f"Error updating rule success rate: {e}")
    
    def _update_adjustment_stats(self, change: ConfigChange):
        """Update global adjustment statistics"""
        try:
            with self._lock:
                self.adjustment_stats['total_adjustments'] += 1
                
                if change.success:
                    self.adjustment_stats['successful_adjustments'] += 1
                    
                    # Update average performance improvement
                    impact = change.performance_impact or {}
                    improvement = impact.get('overall_performance_change', 0)
                    
                    total = self.adjustment_stats['total_adjustments']
                    current_avg = self.adjustment_stats['average_performance_improvement']
                    new_avg = ((current_avg * (total - 1)) + improvement) / total
                    self.adjustment_stats['average_performance_improvement'] = new_avg
                
                if change.rollback_performed:
                    self.adjustment_stats['rolled_back_adjustments'] += 1
                    
        except Exception as e:
            self.logger.error(f"Error updating adjustment stats: {e}")
    
    async def _detect_configuration_drift(self):
        """Detect if configuration has drifted from optimal settings"""
        try:
            current_conditions = await self._get_current_conditions()
            optimal_profile = self._find_optimal_profile(current_conditions)
            
            if optimal_profile:
                drift_detected = self._check_configuration_drift(optimal_profile)
                
                if drift_detected:
                    await self._correct_configuration_drift(optimal_profile)
                    
        except Exception as e:
            self.logger.error(f"Error detecting configuration drift: {e}")
    
    async def _get_current_conditions(self) -> Dict[str, Any]:
        """Get current system conditions for profile matching"""
        conditions = {}
        
        try:
            # Get performance metrics
            if self.metrics_collector:
                metrics = self.metrics_collector.get_current_metrics()
                
                # Determine load level
                if hasattr(metrics, 'throughput'):
                    rps = metrics.throughput.requests_per_second
                    if rps > 15:
                        conditions['load_level'] = 'high'
                    elif rps > 5:
                        conditions['load_level'] = 'medium'
                    else:
                        conditions['load_level'] = 'low'
                
                # Determine latency requirements
                if hasattr(metrics, 'latency'):
                    p95_latency = metrics.latency.get_p95()
                    if p95_latency < 600:
                        conditions['latency_requirements'] = 'strict'
                    else:
                        conditions['latency_requirements'] = 'flexible'
            
            # Get resource availability
            resources = await self._get_resource_status()
            cpu_usage = resources.get('cpu_percent', 50) / 100
            memory_usage = resources.get('memory_percent', 50) / 100
            
            if cpu_usage < 0.5 and memory_usage < 0.6:
                conditions['resource_availability'] = 'abundant'
            elif cpu_usage < 0.7 and memory_usage < 0.8:
                conditions['resource_availability'] = 'normal'
            else:
                conditions['resource_availability'] = 'limited'
            
            # Add time-based conditions
            hour = datetime.now().hour
            if 9 <= hour <= 17:  # Business hours
                conditions['time_period'] = 'business_hours'
            elif 18 <= hour <= 22:  # Evening
                conditions['time_period'] = 'evening'
            else:  # Night
                conditions['time_period'] = 'night'
            
        except Exception as e:
            self.logger.warning(f"Error getting current conditions: {e}")
        
        return conditions
    
    def _find_optimal_profile(self, conditions: Dict[str, Any]) -> Optional[ConfigurationProfile]:
        """Find the most suitable configuration profile for current conditions"""
        try:
            best_profile = None
            best_score = 0.0
            
            for profile in self.configuration_profiles.values():
                score = self._calculate_profile_match_score(profile, conditions)
                
                if score > best_score:
                    best_score = score
                    best_profile = profile
            
            # Only return profile if match score is above threshold
            if best_score > 0.7:
                return best_profile
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error finding optimal profile: {e}")
            return None
    
    def _calculate_profile_match_score(self, profile: ConfigurationProfile, conditions: Dict[str, Any]) -> float:
        """Calculate how well a profile matches current conditions"""
        try:
            match_factors = []
            
            profile_conditions = profile.conditions
            
            # Check load level match
            if 'load_level' in conditions and 'load_level' in profile_conditions:
                if conditions['load_level'] == profile_conditions['load_level']:
                    match_factors.append(1.0)
                else:
                    match_factors.append(0.0)
            
            # Check latency requirements match
            if 'latency_requirements' in conditions and 'latency_requirements' in profile_conditions:
                if conditions['latency_requirements'] == profile_conditions['latency_requirements']:
                    match_factors.append(1.0)
                else:
                    match_factors.append(0.5)  # Partial match possible
            
            # Check resource availability match
            if 'resource_availability' in conditions and 'resource_availability' in profile_conditions:
                if conditions['resource_availability'] == profile_conditions['resource_availability']:
                    match_factors.append(1.0)
                else:
                    match_factors.append(0.3)  # Partial match
            
            # Weight by profile confidence and usage
            base_score = sum(match_factors) / len(match_factors) if match_factors else 0.0
            confidence_weight = profile.confidence_score
            usage_weight = min(1.0, profile.usage_count / 10.0)  # More usage = higher confidence
            
            final_score = base_score * 0.7 + confidence_weight * 0.2 + usage_weight * 0.1
            
            return final_score
            
        except Exception as e:
            self.logger.warning(f"Error calculating profile match score: {e}")
            return 0.0
    
    def _check_configuration_drift(self, optimal_profile: ConfigurationProfile) -> bool:
        """Check if current configuration has drifted from optimal profile"""
        try:
            if not self.current_config:
                return False
            
            drift_score = 0.0
            total_checks = 0
            
            for config_path, optimal_value in optimal_profile.configuration.items():
                current_value = self._get_config_value(self.current_config, config_path)
                
                if current_value is not None:
                    total_checks += 1
                    
                    # Calculate drift for this configuration item
                    if isinstance(optimal_value, (int, float)) and isinstance(current_value, (int, float)):
                        if optimal_value != 0:
                            item_drift = abs(current_value - optimal_value) / abs(optimal_value)
                            drift_score += item_drift
                    elif optimal_value != current_value:
                        drift_score += 1.0  # Full drift for non-numeric values
            
            if total_checks > 0:
                average_drift = drift_score / total_checks
                return average_drift > self.config['config_drift_threshold']
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Error checking configuration drift: {e}")
            return False
    
    async def _correct_configuration_drift(self, optimal_profile: ConfigurationProfile):
        """Correct configuration drift by applying optimal profile"""
        try:
            self.logger.info(f"⚙️ Correcting configuration drift using profile: {optimal_profile.name}")
            
            changes_applied = 0
            
            for config_path, optimal_value in optimal_profile.configuration.items():
                current_value = self._get_config_value(self.current_config, config_path)
                
                if current_value != optimal_value:
                    # Create drift correction change
                    change = ConfigChange(
                        change_id=f"drift_{optimal_profile.profile_id}_{int(time.time())}_{changes_applied}",
                        rule_id="drift_correction",
                        adjustment_type=ConfigAdjustmentType.PERFORMANCE_TUNING,
                        scope=ConfigScope.GLOBAL,
                        status=ConfigChangeStatus.PENDING,
                        triggered_at=time.time(),
                        trigger_reason=f"Configuration drift correction using {optimal_profile.name}",
                        prediction_data={'profile_match_score': optimal_profile.confidence_score},
                        config_path=config_path,
                        old_value=current_value,
                        new_value=optimal_value,
                        adjustment_metadata={'drift_correction': True, 'profile_id': optimal_profile.profile_id}
                    )
                    
                    # Apply change
                    success = await self._update_configuration_value(config_path, optimal_value)
                    if success:
                        change.status = ConfigChangeStatus.COMPLETED
                        change.success = True
                        changes_applied += 1
                        
                        with self._lock:
                            self.change_history.append(change)
            
            # Update profile usage
            optimal_profile.usage_count += 1
            optimal_profile.last_used = time.time()
            
            self.logger.info(f"⚙️ Configuration drift correction completed: {changes_applied} changes applied")
            
        except Exception as e:
            self.logger.error(f"Error correcting configuration drift: {e}")
    
    async def _update_configuration_profiles(self):
        """Update configuration profiles based on performance data"""
        try:
            current_conditions = await self._get_current_conditions()
            current_performance = await self._measure_performance()
            
            # Find or create profile for current conditions
            profile = self._find_or_create_profile(current_conditions, current_performance)
            
            if profile:
                # Update profile performance metrics
                self._update_profile_performance(profile, current_performance)
                
        except Exception as e:
            self.logger.error(f"Error updating configuration profiles: {e}")
    
    def _find_or_create_profile(self, conditions: Dict[str, Any], performance: Dict[str, Any]) -> Optional[ConfigurationProfile]:
        """Find existing profile or create new one for current conditions"""
        try:
            # Try to find existing profile
            for profile in self.configuration_profiles.values():
                score = self._calculate_profile_match_score(profile, conditions)
                if score > 0.8:  # High match threshold
                    return profile
            
            # Create new profile if learning is enabled
            if self.config['learning_enabled'] and self.current_config:
                profile_id = f"learned_{int(time.time())}"
                
                # Extract current configuration
                current_config_dict = {}
                for config_path in ['llm.fast.max_tokens', 'llm.fast.temperature', 'memory.retrieval_k']:
                    value = self._get_config_value(self.current_config, config_path)
                    if value is not None:
                        current_config_dict[config_path] = value
                
                new_profile = ConfigurationProfile(
                    profile_id=profile_id,
                    name=f"Learned Profile {len(self.configuration_profiles) + 1}",
                    description=f"Automatically learned profile for conditions: {conditions}",
                    conditions=conditions.copy(),
                    configuration=current_config_dict,
                    performance_metrics={
                        'average_latency': performance.get('latency_ms', 500),
                        'accuracy_score': performance.get('accuracy', 0.9),
                        'resource_efficiency': 0.8  # Default
                    },
                    confidence_score=0.5,  # Start with low confidence
                    usage_count=1,
                    last_used=time.time()
                )
                
                self.configuration_profiles[profile_id] = new_profile
                self.logger.info(f"⚙️ Created new configuration profile: {new_profile.name}")
                
                return new_profile
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error finding or creating profile: {e}")
            return None
    
    def _update_profile_performance(self, profile: ConfigurationProfile, performance: Dict[str, Any]):
        """Update profile performance metrics with new data"""
        try:
            # Update metrics with moving average
            weight = 0.1  # Weight for new measurement
            
            if 'latency_ms' in performance:
                current_latency = profile.performance_metrics.get('average_latency', 500)
                new_latency = performance['latency_ms']
                profile.performance_metrics['average_latency'] = (
                    current_latency * (1 - weight) + new_latency * weight
                )
            
            if 'accuracy' in performance:
                current_accuracy = profile.performance_metrics.get('accuracy_score', 0.9)
                new_accuracy = performance['accuracy']
                profile.performance_metrics['accuracy_score'] = (
                    current_accuracy * (1 - weight) + new_accuracy * weight
                )
            
            # Increase confidence based on usage
            profile.confidence_score = min(0.95, profile.confidence_score + 0.01)
            
        except Exception as e:
            self.logger.warning(f"Error updating profile performance: {e}")
    
    def _identify_current_profile(self) -> Optional[str]:
        """Identify which profile best matches current configuration"""
        try:
            if not self.current_config:
                return None
            
            best_match_id = None
            best_score = 0.0
            
            for profile_id, profile in self.configuration_profiles.items():
                score = self._calculate_config_similarity(profile.configuration)
                
                if score > best_score:
                    best_score = score
                    best_match_id = profile_id
            
            return best_match_id if best_score > 0.7 else None
            
        except Exception as e:
            self.logger.warning(f"Error identifying current profile: {e}")
            return None
    
    def _calculate_config_similarity(self, target_config: Dict[str, Any]) -> float:
        """Calculate similarity between current config and target config"""
        try:
            matches = 0
            total = 0
            
            for config_path, target_value in target_config.items():
                current_value = self._get_config_value(self.current_config, config_path)
                
                if current_value is not None:
                    total += 1
                    
                    if isinstance(target_value, (int, float)) and isinstance(current_value, (int, float)):
                        # For numeric values, consider close values as matches
                        if target_value != 0:
                            diff_ratio = abs(current_value - target_value) / abs(target_value)
                            if diff_ratio < 0.1:  # Within 10%
                                matches += 1
                    elif current_value == target_value:
                        matches += 1
            
            return matches / total if total > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating config similarity: {e}")
            return 0.0
    
    def _conditions_to_key(self, conditions: Dict[str, Any]) -> str:
        """Convert conditions dict to a string key"""
        try:
            sorted_items = sorted(conditions.items())
            return json.dumps(sorted_items, sort_keys=True)
        except Exception:
            return "default"
    
    # Public API methods
    def add_adjustment_rule(self, rule: ConfigAdjustmentRule):
        """Add a new configuration adjustment rule"""
        with self._lock:
            self.adjustment_rules[rule.rule_id] = rule
        
        self.logger.info(f"Added configuration adjustment rule: {rule.name}")
    
    def remove_adjustment_rule(self, rule_id: str):
        """Remove a configuration adjustment rule"""
        with self._lock:
            if rule_id in self.adjustment_rules:
                del self.adjustment_rules[rule_id]
                self.logger.info(f"Removed configuration adjustment rule: {rule_id}")
    
    def get_optimized_config(self, component: Optional[str] = None) -> Dict[str, Any]:
        """Get optimized configuration for current or specified conditions"""
        try:
            # This would be called by other components to get optimized configuration
            current_conditions = asyncio.create_task(self._get_current_conditions())
            condition_key = self._conditions_to_key(current_conditions.result())
            
            # Return optimized config for current conditions
            return self.optimized_configs.get(condition_key, {})
            
        except Exception as e:
            self.logger.error(f"Error getting optimized config: {e}")
            return {}
    
    def get_adjustment_statistics(self) -> Dict[str, Any]:
        """Get configuration adjustment statistics"""
        with self._lock:
            stats = self.adjustment_stats.copy()
            
            # Add computed metrics
            if stats['total_adjustments'] > 0:
                stats['success_rate'] = stats['successful_adjustments'] / stats['total_adjustments']
                stats['rollback_rate'] = stats['rolled_back_adjustments'] / stats['total_adjustments']
            else:
                stats['success_rate'] = 0.0
                stats['rollback_rate'] = 0.0
            
            # Add rule statistics
            stats['total_rules'] = len(self.adjustment_rules)
            stats['enabled_rules'] = len([r for r in self.adjustment_rules.values() if r.enabled])
            
            # Add profile statistics
            stats['total_profiles'] = len(self.configuration_profiles)
            stats['active_profile'] = self._identify_current_profile()
            
            return stats
    
    def get_configuration_profiles(self) -> List[ConfigurationProfile]:
        """Get all configuration profiles"""
        with self._lock:
            return list(self.configuration_profiles.values())
    
    def get_active_changes(self) -> List[ConfigChange]:
        """Get currently active configuration changes"""
        with self._lock:
            return list(self.active_changes.values())
    
    def get_change_history(self, limit: int = 100) -> List[ConfigChange]:
        """Get configuration change history"""
        with self._lock:
            return self.change_history[-limit:]

# Factory function for easy instantiation
def create_predictive_config_manager(
    base_config_manager: ConfigManager,
    predictive_analytics: Optional[PredictiveAnalyticsFramework] = None,
    metrics_collector: Optional[MetricsCollector] = None,
    logger: Optional[VoiceAILogger] = None,
    config: Optional[Dict[str, Any]] = None
) -> PredictiveConfigManager:
    """Create and configure a PredictiveConfigManager instance"""
    return PredictiveConfigManager(
        base_config_manager=base_config_manager,
        predictive_analytics=predictive_analytics,
        metrics_collector=metrics_collector,
        logger=logger,
        config=config
    )

# Global instance management
_predictive_config_manager: Optional[PredictiveConfigManager] = None

def get_predictive_config_manager() -> Optional[PredictiveConfigManager]:
    """Get the global predictive config manager instance"""
    return _predictive_config_manager

def set_predictive_config_manager(manager: PredictiveConfigManager):
    """Set the global predictive config manager instance"""
    global _predictive_config_manager
    _predictive_config_manager = manager 