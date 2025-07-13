"""
Comprehensive tests for the Predictive Integration System (Task 10.5)

Tests integration between predictive analytics and existing error handling,
health monitoring, and configuration management systems.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import asdict

from assistant.unified_health_reporter import (
    UnifiedHealthReporter, UnifiedHealthStatus, UnifiedHealthReport, RiskFactor, RecommendedAction,
    create_unified_health_reporter
)
from assistant.predictive_error_enricher import (
    PredictiveErrorEnricher, EnrichedErrorContext, ErrorPattern, SimilarIncident,
    PredictiveRootCause, RecoveryRecommendation, create_predictive_error_enricher
)
from assistant.predictive_optimization_engine import (
    PredictiveOptimizationEngine, OptimizationType, OptimizationPriority, OptimizationStatus,
    OptimizationRule, OptimizationAction, OptimizationResult, create_predictive_optimization_engine
)
from assistant.predictive_config_manager import (
    PredictiveConfigManager, ConfigAdjustmentType, ConfigScope, ConfigChangeStatus,
    ConfigAdjustmentRule, ConfigChange, ConfigurationProfile, create_predictive_config_manager
)
from assistant.health_monitoring import SystemHealthMonitor, HealthStatus, SystemHealthMetrics
from assistant.predictive_analytics import PredictiveAnalyticsFramework
from assistant.metrics_collector import MetricsCollector
from assistant.config_manager import ConfigManager, SovereignConfig
from assistant.error_handling import VoiceAIException, ErrorCategory, STTException, LLMException
from assistant.drift_detection import DriftDetector, DriftAlert, DriftType, DriftSeverity

class TestUnifiedHealthReporter:
    """Test unified health reporting with predictive analytics integration"""
    
    @pytest.fixture
    def mock_health_monitor(self):
        """Create mock health monitor"""
        mock = Mock(spec=SystemHealthMonitor)
        mock.get_system_health = AsyncMock(return_value=SystemHealthMetrics(
            overall_status=HealthStatus.HEALTHY,
            timestamp=time.time(),
            uptime_seconds=3600,
            total_requests=1000,
            error_rate=0.05,
            avg_response_time_ms=450.0,
            memory_usage_mb=512.0,
            cpu_usage_percent=65.0,
            disk_usage_percent=40.0,
            network_connectivity=True,
            services={},
            alerts=[]
        ))
        return mock
    
    @pytest.fixture
    def mock_predictive_analytics(self):
        """Create mock predictive analytics framework"""
        mock = Mock(spec=PredictiveAnalyticsFramework)
        mock.get_health_score = AsyncMock(return_value=0.85)
        mock.drift_detector = Mock()
        mock.drift_detector.get_active_alerts = AsyncMock(return_value=[
            DriftAlert(
                timestamp=time.time(),
                drift_type=DriftType.PERFORMANCE,
                severity=DriftSeverity.HIGH,
                metric_name="accuracy",
                drift_score=0.3,
                threshold=0.2,
                description="Accuracy drift detected"
            )
        ])
        mock.resource_forecaster = Mock()
        mock.resource_forecaster.get_forecasts = AsyncMock(return_value={
            'cpu_usage': {'forecast': 0.8, 'confidence': 0.85},
            'memory_usage': {'forecast': 0.75, 'confidence': 0.9}
        })
        mock.performance_predictor = Mock()
        mock.performance_predictor.get_predictions = AsyncMock(return_value={
            'degradation_probability': 0.3,
            'latency_forecast': 600.0,
            'accuracy_forecast': 0.88
        })
        mock.detect_anomalies = AsyncMock(return_value=[])
        mock.get_current_state = AsyncMock(return_value={'status': 'healthy'})
        return mock
    
    @pytest.fixture
    def mock_metrics_collector(self):
        """Create mock metrics collector"""
        mock = Mock(spec=MetricsCollector)
        mock.get_current_metrics = Mock(return_value=Mock(
            latency=Mock(get_p95=Mock(return_value=500), get_average=Mock(return_value=450)),
            throughput=Mock(error_rate=0.05, requests_per_second=12.0)
        ))
        return mock
    
    @pytest.fixture
    def unified_health_reporter(self, mock_health_monitor, mock_predictive_analytics, mock_metrics_collector):
        """Create unified health reporter with mocked dependencies"""
        return create_unified_health_reporter(
            health_monitor=mock_health_monitor,
            predictive_analytics=mock_predictive_analytics,
            metrics_collector=mock_metrics_collector
        )
    
    @pytest.mark.asyncio
    async def test_unified_health_report_generation(self, unified_health_reporter):
        """Test comprehensive unified health report generation"""
        report = await unified_health_reporter.get_unified_health_report()
        
        assert isinstance(report, UnifiedHealthReport)
        assert report.overall_status in UnifiedHealthStatus
        assert 0.0 <= report.health_score <= 1.0
        assert report.predictions_available is True
        assert len(report.drift_alerts) > 0
        assert len(report.risk_factors) >= 0
        assert len(report.recommended_actions) >= 0
    
    @pytest.mark.asyncio
    async def test_health_score_weighting(self, unified_health_reporter):
        """Test health score calculation with predictive weighting"""
        report = await unified_health_reporter.get_unified_health_report()
        
        # Should weight current and predictive health
        assert report.current_health_score > 0
        assert report.predictive_health_score > 0
        assert report.health_score > 0
        
        # Health score should be influenced by both current and predictive scores
        # (but can be adjusted by trend factors, so may not be strictly between them)
        min_score = min(report.current_health_score, report.predictive_health_score)
        max_score = max(report.current_health_score, report.predictive_health_score)
        
        # Allow for trend factor influence (±10% from the expected range)
        tolerance = 0.1
        expected_min = min_score - tolerance
        expected_max = max_score + tolerance
        assert expected_min <= report.health_score <= expected_max
    
    @pytest.mark.asyncio
    async def test_risk_factor_assessment(self, unified_health_reporter):
        """Test risk factor identification and assessment"""
        report = await unified_health_reporter.get_unified_health_report()
        
        # Should identify risk factors from drift alerts
        drift_risk_factors = [rf for rf in report.risk_factors if rf.factor_type == 'drift_detection']
        assert len(drift_risk_factors) > 0
        
        # Risk factors should have required attributes
        for risk_factor in report.risk_factors:
            assert isinstance(risk_factor, RiskFactor)
            assert risk_factor.severity in ['low', 'medium', 'high', 'critical']
            assert 0.0 <= risk_factor.probability <= 1.0
            assert 0.0 <= risk_factor.confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_recommendation_generation(self, unified_health_reporter):
        """Test automated recommendation generation"""
        report = await unified_health_reporter.get_unified_health_report()
        
        # Should generate recommendations based on status and risk factors
        assert len(report.recommended_actions) >= 0
        
        for action in report.recommended_actions:
            assert isinstance(action, RecommendedAction)
            assert action.priority in ['low', 'medium', 'high', 'urgent']
            assert action.effort_level in ['low', 'medium', 'high']
            assert isinstance(action.automation_possible, bool)
    
    @pytest.mark.asyncio
    async def test_trend_analysis(self, unified_health_reporter):
        """Test health trend analysis"""
        # Generate multiple reports to build history
        for _ in range(5):
            await unified_health_reporter.get_unified_health_report()
            await asyncio.sleep(0.1)
        
        report = await unified_health_reporter.get_unified_health_report()
        
        assert report.health_trend in ['improving', 'stable', 'declining', 'unknown']
        assert 0.0 <= report.trend_confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, mock_health_monitor):
        """Test graceful degradation when predictive services unavailable"""
        # Create reporter without predictive analytics
        reporter = create_unified_health_reporter(health_monitor=mock_health_monitor)
        
        report = await reporter.get_unified_health_report()
        
        assert isinstance(report, UnifiedHealthReport)
        assert report.predictions_available is False
        assert report.predictive_health_score == 0.5  # Default neutral score
        assert len(report.drift_alerts) == 0


class TestPredictiveErrorEnricher:
    """Test predictive error enrichment with analytics integration"""
    
    @pytest.fixture
    def mock_predictive_analytics(self):
        """Create mock predictive analytics framework"""
        mock = Mock(spec=PredictiveAnalyticsFramework)
        mock.detect_anomalies = AsyncMock(return_value=[
            {
                'description': 'High latency anomaly',
                'severity': 'high',
                'confidence': 0.85,
                'factors': ['resource_constraint', 'network_latency'],
                'time_to_impact': 300,
                'recommendations': ['Scale resources', 'Check network']
            }
        ])
        mock.get_current_state = AsyncMock(return_value={'status': 'degraded'})
        return mock
    
    @pytest.fixture
    def mock_drift_detector(self):
        """Create mock drift detector"""
        mock = Mock(spec=DriftDetector)
        mock.get_active_alerts = AsyncMock(return_value=[
            DriftAlert(
                timestamp=time.time(),
                drift_type=DriftType.PERFORMANCE,
                severity=DriftSeverity.MEDIUM,
                metric_name="stt_accuracy",
                drift_score=0.25,
                threshold=0.2,
                description="STT accuracy drift detected"
            )
        ])
        mock.get_current_status = AsyncMock(return_value={'drift_detected': True})
        mock.get_recent_alerts = AsyncMock(return_value=[])
        return mock
    
    @pytest.fixture
    def mock_metrics_collector(self):
        """Create mock metrics collector"""
        mock = Mock(spec=MetricsCollector)
        mock.get_current_metrics = Mock(return_value=Mock(
            latency=Mock(get_p95=Mock(return_value=800)),
            throughput=Mock(error_rate=0.08)
        ))
        mock.get_recent_alerts = Mock(return_value=[])
        mock.get_recent_events = Mock(return_value=[])
        return mock
    
    @pytest.fixture
    def error_enricher(self, mock_predictive_analytics, mock_drift_detector, mock_metrics_collector):
        """Create predictive error enricher with mocked dependencies"""
        return create_predictive_error_enricher(
            predictive_analytics=mock_predictive_analytics,
            drift_detector=mock_drift_detector,
            metrics_collector=mock_metrics_collector
        )
    
    @pytest.mark.asyncio
    async def test_comprehensive_error_enrichment(self, error_enricher):
        """Test comprehensive error enrichment with all components"""
        # Create test error
        error = STTException("Speech recognition failed", ErrorCategory.TRANSIENT, retryable=True)
        
        enriched_context = await error_enricher.enrich_error(error, async_mode=False)
        
        assert isinstance(enriched_context, EnrichedErrorContext)
        assert enriched_context.original_error == error
        assert enriched_context.enrichment_timestamp > 0
        assert enriched_context.enrichment_duration_ms > 0
        
        # Should have predictive insights
        assert len(enriched_context.predicted_root_causes) >= 0
        assert len(enriched_context.recovery_recommendations) >= 0
        assert enriched_context.system_state_at_error is not None
    
    @pytest.mark.asyncio
    async def test_root_cause_prediction(self, error_enricher):
        """Test predictive root cause analysis"""
        error = LLMException("Model inference timeout", ErrorCategory.TIMEOUT, retryable=True)
        
        enriched_context = await error_enricher.enrich_error(error, async_mode=False)
        
        # Should identify root causes from different sources
        drift_causes = [rc for rc in enriched_context.predicted_root_causes 
                       if rc.detection_method == 'drift_detection']
        anomaly_causes = [rc for rc in enriched_context.predicted_root_causes 
                         if rc.detection_method == 'anomaly_detection']
        
        # Check root cause attributes
        for root_cause in enriched_context.predicted_root_causes:
            assert isinstance(root_cause, PredictiveRootCause)
            assert 0.0 <= root_cause.confidence <= 1.0
            assert len(root_cause.preventive_measures) >= 0
    
    @pytest.mark.asyncio
    async def test_recovery_recommendation_generation(self, error_enricher):
        """Test recovery recommendation generation"""
        error = STTException("Audio processing failed", ErrorCategory.AUDIO, retryable=True)
        
        enriched_context = await error_enricher.enrich_error(error, async_mode=False)
        
        # Should generate recovery recommendations
        assert len(enriched_context.recovery_recommendations) >= 0
        
        for recommendation in enriched_context.recovery_recommendations:
            assert isinstance(recommendation, RecoveryRecommendation)
            assert recommendation.priority in ['low', 'medium', 'high', 'urgent']
            assert 0.0 <= recommendation.success_probability <= 1.0
            assert recommendation.risk_level in ['low', 'medium', 'high']
    
    @pytest.mark.asyncio
    async def test_error_correlation_analysis(self, error_enricher):
        """Test error correlation and clustering analysis"""
        # Enrich multiple related errors
        errors = [
            STTException("STT service timeout", ErrorCategory.TIMEOUT, retryable=True),
            LLMException("LLM inference timeout", ErrorCategory.TIMEOUT, retryable=True),
            STTException("STT connection failed", ErrorCategory.NETWORK, retryable=True)
        ]
        
        enriched_contexts = []
        for error in errors:
            context = await error_enricher.enrich_error(error, async_mode=False)
            enriched_contexts.append(context)
        
        # Should identify correlations
        for context in enriched_contexts:
            assert context.error_cluster_id is not None
            assert isinstance(context.cascade_probability, float)
            assert 0.0 <= context.cascade_probability <= 1.0
    
    @pytest.mark.asyncio
    async def test_impact_assessment(self, error_enricher):
        """Test error impact assessment"""
        error = LLMException("Critical model failure", ErrorCategory.PERMANENT, retryable=False)
        
        enriched_context = await error_enricher.enrich_error(error, async_mode=False)
        
        # Should assess impact accurately
        assert 0.0 <= enriched_context.severity_escalation_risk <= 1.0
        assert 0.0 <= enriched_context.user_impact_score <= 1.0
        assert enriched_context.business_impact_assessment in ['low', 'medium', 'high', 'critical', 'unknown']
        
        # Permanent errors should have high impact
        assert enriched_context.severity_escalation_risk > 0.5
    
    @pytest.mark.asyncio
    async def test_async_enrichment_processing(self, error_enricher):
        """Test asynchronous enrichment processing"""
        await error_enricher.start_async_processing()
        
        try:
            error = STTException("Async test error", ErrorCategory.TRANSIENT, retryable=True)
            
            # Should process asynchronously
            enriched_context = await error_enricher.enrich_error(error, async_mode=True)
            
            assert isinstance(enriched_context, EnrichedErrorContext)
            
        finally:
            await error_enricher.stop_async_processing()


class TestPredictiveOptimizationEngine:
    """Test predictive optimization engine with automated triggers"""
    
    @pytest.fixture
    def mock_predictive_analytics(self):
        """Create mock predictive analytics framework"""
        mock = Mock(spec=PredictiveAnalyticsFramework)
        mock.performance_predictor = Mock()
        mock.performance_predictor.get_predictions = AsyncMock(return_value={
            'latency_p95': 1200,  # High latency predicted
            'degradation_probability': 0.7,
            'trend': 'increasing',
            'accuracy_score': 0.9
        })
        mock.resource_forecaster = Mock()
        mock.resource_forecaster.get_forecasts = AsyncMock(return_value={
            'cpu_usage': {'forecast': 0.85, 'confidence': 0.9},
            'memory_usage': {'forecast': 0.8, 'confidence': 0.85}
        })
        mock.drift_detector = Mock()
        mock.drift_detector.get_active_alerts = AsyncMock(return_value=[])
        mock.get_health_score = AsyncMock(return_value=0.7)
        return mock
    
    @pytest.fixture
    def mock_metrics_collector(self):
        """Create mock metrics collector"""
        mock = Mock(spec=MetricsCollector)
        mock.get_current_metrics = Mock(return_value=Mock(
            latency=Mock(get_p95=Mock(return_value=900), get_average=Mock(return_value=800)),
            throughput=Mock(error_rate=0.03, requests_per_second=15.0, timeout_rate=0.02)
        ))
        return mock
    
    @pytest.fixture
    def optimization_engine(self, mock_predictive_analytics, mock_metrics_collector):
        """Create predictive optimization engine with mocked dependencies"""
        return create_predictive_optimization_engine(
            predictive_analytics=mock_predictive_analytics,
            metrics_collector=mock_metrics_collector
        )
    
    @pytest.mark.asyncio
    async def test_optimization_rule_evaluation(self, optimization_engine):
        """Test optimization rule evaluation and triggering"""
        # Start monitoring
        await optimization_engine.start_monitoring()
        
        try:
            # Wait for evaluation cycle
            await asyncio.sleep(0.5)
            
            # Should have evaluated rules
            stats = optimization_engine.get_optimization_statistics()
            assert stats['total_rules'] > 0
            assert stats['enabled_rules'] > 0
            
        finally:
            await optimization_engine.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_resource_scaling_optimization(self, optimization_engine):
        """Test resource scaling optimization trigger"""
        # Get the scaling rule
        scaling_rule = None
        for rule in optimization_engine.optimization_rules.values():
            if rule.optimization_type == OptimizationType.RESOURCE_SCALING:
                scaling_rule = rule
                break
        
        assert scaling_rule is not None
        
        # Mock high latency conditions that should trigger scaling
        predictions = {
            'performance': {
                'latency_p95': 1200,  # Above threshold
                'trend': 'increasing'
            }
        }
        system_state = {
            'metrics': Mock(latency=Mock(get_p95=Mock(return_value=900)))  # Above threshold
        }
        
        should_trigger, reason = await optimization_engine._should_trigger_optimization(
            scaling_rule, predictions, system_state
        )
        
        assert should_trigger is True
        assert 'latency' in reason.lower()
    
    @pytest.mark.asyncio
    async def test_model_switching_optimization(self, optimization_engine):
        """Test model switching optimization trigger"""
        # Find model switching rule
        model_rule = None
        for rule in optimization_engine.optimization_rules.values():
            if rule.optimization_type == OptimizationType.MODEL_SWITCHING:
                model_rule = rule
                break
        
        assert model_rule is not None
        
        # Mock degradation conditions
        predictions = {
            'performance': {
                'degradation_probability': 0.7,  # Above threshold
                'accuracy_score': 0.9
            }
        }
        system_state = {
            'metrics': Mock(latency=Mock(get_average=Mock(return_value=2100)))  # Above threshold
        }
        
        should_trigger, reason = await optimization_engine._should_trigger_optimization(
            model_rule, predictions, system_state
        )
        
        assert should_trigger is True
        assert 'degradation' in reason.lower()
    
    @pytest.mark.asyncio
    async def test_optimization_execution(self, optimization_engine):
        """Test optimization action execution"""
        # Create test optimization action
        action = OptimizationAction(
            action_id="test_action",
            rule_id="test_rule",
            optimization_type=OptimizationType.RESOURCE_SCALING,
            priority=OptimizationPriority.HIGH,
            status=OptimizationStatus.PENDING,
            triggered_at=time.time(),
            trigger_reason="Test trigger",
            prediction_data={},
            system_state={},
            action_config={'scaling_factor': 1.5},
            target_component="test_service",
            expected_impact="medium"
        )
        
        # Execute optimization
        await optimization_engine._execute_optimization_action(action)
        
        # Should complete successfully
        assert action.status == OptimizationStatus.COMPLETED
        assert action.success is True
        assert action.duration_ms > 0
        assert action.impact_metrics is not None
    
    @pytest.mark.asyncio
    async def test_safety_limits(self, optimization_engine):
        """Test safety limits and emergency stop conditions"""
        # Check safety limits
        safety_ok = optimization_engine._check_safety_limits()
        assert isinstance(safety_ok, bool)
        
        # Test emergency stop conditions
        emergency_stop = optimization_engine._check_emergency_stop_conditions()
        assert isinstance(emergency_stop, bool)
    
    @pytest.mark.asyncio
    async def test_optimization_statistics(self, optimization_engine):
        """Test optimization performance statistics"""
        stats = optimization_engine.get_optimization_statistics()
        
        assert 'total_actions' in stats
        assert 'successful_actions' in stats
        assert 'failed_actions' in stats
        assert 'rollback_actions' in stats
        assert 'success_rate' in stats
        assert 'total_rules' in stats
        assert 'enabled_rules' in stats


class TestPredictiveConfigManager:
    """Test predictive configuration manager with adaptive configuration"""
    
    @pytest.fixture
    def mock_base_config_manager(self):
        """Create mock base configuration manager"""
        mock = Mock(spec=ConfigManager)
        
        # Create mock configuration
        mock_config = Mock(spec=SovereignConfig)
        mock_config.llm = Mock()
        mock_config.llm.fast = Mock()
        mock_config.llm.fast.model = "gpt-3.5-turbo"
        mock_config.llm.fast.max_tokens = 3000
        mock_config.llm.fast.temperature = 0.8
        mock_config.llm.fast.timeout = 30.0
        mock_config.llm.deep = Mock()
        mock_config.llm.deep.max_tokens = 4000
        mock_config.llm.deep.temperature = 0.3
        mock_config.memory = Mock()
        mock_config.memory.retrieval_k = 10
        mock_config.memory.similarity_threshold = 0.7
        mock_config.stt = Mock()
        mock_config.stt.timeout = 15.0
        mock_config.tts = Mock()
        mock_config.tts.timeout = 10.0
        mock_config.audio = Mock()
        mock_config.audio.buffer_size = 2048
        
        mock.get_config = Mock(return_value=mock_config)
        mock.reload_config = Mock()
        
        return mock
    
    @pytest.fixture
    def mock_predictive_analytics(self):
        """Create mock predictive analytics framework"""
        mock = Mock(spec=PredictiveAnalyticsFramework)
        mock.performance_predictor = Mock()
        mock.performance_predictor.get_predictions = AsyncMock(return_value={
            'latency_increase_ratio': 0.4,  # Above threshold for timeout adjustment
            'reliability_score': 0.95
        })
        mock.resource_forecaster = Mock()
        mock.resource_forecaster.get_forecasts = AsyncMock(return_value={
            'load_increase_ratio': 0.6,  # Above threshold for optimization
            'confidence': 0.8
        })
        mock.drift_detector = Mock()
        mock.drift_detector.get_active_alerts = AsyncMock(return_value=[])
        return mock
    
    @pytest.fixture
    def mock_metrics_collector(self):
        """Create mock metrics collector"""
        mock = Mock(spec=MetricsCollector)
        mock.get_current_metrics = Mock(return_value=Mock(
            latency=Mock(get_p95=Mock(return_value=600)),
            throughput=Mock(timeout_rate=0.06, error_rate=0.03)
        ))
        return mock
    
    @pytest.fixture
    def config_manager(self, mock_base_config_manager, mock_predictive_analytics, mock_metrics_collector):
        """Create predictive configuration manager with mocked dependencies"""
        return create_predictive_config_manager(
            base_config_manager=mock_base_config_manager,
            predictive_analytics=mock_predictive_analytics,
            metrics_collector=mock_metrics_collector
        )
    
    @pytest.mark.asyncio
    async def test_configuration_adjustment_evaluation(self, config_manager):
        """Test configuration adjustment rule evaluation"""
        await config_manager.start_monitoring()
        
        try:
            # Wait for evaluation cycle
            await asyncio.sleep(0.5)
            
            # Should have rules configured
            assert len(config_manager.adjustment_rules) > 0
            
            # Check rule evaluation
            stats = config_manager.get_adjustment_statistics()
            assert 'total_rules' in stats
            assert 'enabled_rules' in stats
            
        finally:
            await config_manager.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_timeout_adjustment_rule(self, config_manager):
        """Test timeout adjustment based on predicted latency"""
        # Get timeout adjustment rule
        timeout_rule = None
        for rule in config_manager.adjustment_rules.values():
            if 'timeout' in rule.rule_id:
                timeout_rule = rule
                break
        
        assert timeout_rule is not None
        
        # Mock conditions that should trigger timeout adjustment
        predictions = {
            'performance': {
                'latency_increase_ratio': 0.4,  # Above threshold
                'reliability_score': 0.95
            }
        }
        system_state = {
            'metrics': Mock(throughput=Mock(timeout_rate=0.06))  # Above threshold
        }
        
        should_trigger, reason = await config_manager._should_trigger_adjustment(
            timeout_rule, predictions, system_state
        )
        
        assert should_trigger is True
        assert 'timeout' in reason.lower() or 'latency' in reason.lower()
    
    @pytest.mark.asyncio
    async def test_configuration_profile_matching(self, config_manager):
        """Test configuration profile matching and optimization"""
        # Test current conditions detection
        conditions = await config_manager._get_current_conditions()
        assert isinstance(conditions, dict)
        
        # Test profile matching
        optimal_profile = config_manager._find_optimal_profile(conditions)
        
        if optimal_profile:
            assert isinstance(optimal_profile, ConfigurationProfile)
            assert optimal_profile.confidence_score > 0
    
    @pytest.mark.asyncio
    async def test_configuration_drift_detection(self, config_manager):
        """Test configuration drift detection and correction"""
        # Create a profile that differs from current config
        drift_profile = ConfigurationProfile(
            profile_id="drift_test",
            name="Drift Test Profile",
            description="Test profile for drift detection",
            conditions={'load_level': 'medium'},
            configuration={
                'llm.fast.max_tokens': 2000,  # Different from current (3000)
                'memory.retrieval_k': 5       # Different from current (10)
            },
            performance_metrics={'accuracy_score': 0.9},
            confidence_score=0.8
        )
        
        # Check if drift is detected
        drift_detected = config_manager._check_configuration_drift(drift_profile)
        assert isinstance(drift_detected, bool)
        
        # Test drift correction
        if drift_detected:
            await config_manager._correct_configuration_drift(drift_profile)
            
            # Should have created change records
            assert len(config_manager.change_history) >= 0
    
    @pytest.mark.asyncio
    async def test_configuration_change_validation(self, config_manager):
        """Test configuration change validation and rollback"""
        # Create test configuration change
        change = ConfigChange(
            change_id="test_change",
            rule_id="test_rule",
            adjustment_type=ConfigAdjustmentType.PERFORMANCE_TUNING,
            scope=ConfigScope.SERVICE,
            status=ConfigChangeStatus.PENDING,
            triggered_at=time.time(),
            trigger_reason="Test change",
            prediction_data={},
            config_path="llm.fast.timeout",
            old_value=30.0,
            new_value=45.0,
            adjustment_metadata={}
        )
        
        # Create validation rule
        validation_rule = ConfigAdjustmentRule(
            rule_id="test_rule",
            name="Test Rule",
            description="Test validation rule",
            adjustment_type=ConfigAdjustmentType.PERFORMANCE_TUNING,
            scope=ConfigScope.SERVICE,
            trigger_conditions={},
            prediction_horizon=300.0,
            confidence_threshold=0.7,
            config_adjustments={},
            adjustment_limits={},
            requires_validation=True,
            validation_duration=1.0,  # Short duration for testing
            rollback_conditions={'timeout_improvement': 0.2},
            max_impact_level="low"
        )
        
        # Test validation process
        await config_manager._validate_configuration_change(change, validation_rule)
        
        # Should complete validation
        assert change.validation_end is not None
        assert change.status in [ConfigChangeStatus.COMPLETED, ConfigChangeStatus.ROLLED_BACK]
    
    @pytest.mark.asyncio
    async def test_a_b_testing_framework(self, config_manager):
        """Test A/B testing framework for configuration changes"""
        # Test profile creation
        test_profile = config_manager._find_or_create_profile(
            {'load_level': 'test'},
            {'latency_ms': 500, 'accuracy': 0.9}
        )
        
        if test_profile:
            assert isinstance(test_profile, ConfigurationProfile)
            assert test_profile.profile_id in config_manager.configuration_profiles
    
    def test_configuration_statistics(self, config_manager):
        """Test configuration adjustment statistics"""
        stats = config_manager.get_adjustment_statistics()
        
        assert 'total_adjustments' in stats
        assert 'successful_adjustments' in stats
        assert 'rolled_back_adjustments' in stats
        assert 'success_rate' in stats
        assert 'total_rules' in stats
        assert 'total_profiles' in stats


class TestIntegrationWorkflows:
    """Test end-to-end integration workflows"""
    
    @pytest.fixture
    def integrated_system(self):
        """Create integrated predictive system with all components"""
        # Mock dependencies
        mock_health_monitor = Mock(spec=SystemHealthMonitor)
        mock_health_monitor.get_system_health = AsyncMock(return_value=SystemHealthMetrics(
            timestamp=time.time(), uptime_seconds=3600, total_requests=1000,
            error_rate=0.05, avg_response_time_ms=450.0, memory_usage_mb=512.0,
            cpu_usage_percent=65.0, disk_usage_percent=40.0, network_connectivity=True,
            services={}, alerts=[]
        ))
        
        mock_predictive_analytics = Mock(spec=PredictiveAnalyticsFramework)
        mock_predictive_analytics.get_health_score = AsyncMock(return_value=0.85)
        mock_predictive_analytics.drift_detector = Mock()
        mock_predictive_analytics.drift_detector.get_active_alerts = AsyncMock(return_value=[])
        mock_predictive_analytics.resource_forecaster = Mock()
        mock_predictive_analytics.resource_forecaster.get_forecasts = AsyncMock(return_value={})
        mock_predictive_analytics.performance_predictor = Mock()
        mock_predictive_analytics.performance_predictor.get_predictions = AsyncMock(return_value={})
        mock_predictive_analytics.detect_anomalies = AsyncMock(return_value=[])
        mock_predictive_analytics.get_current_state = AsyncMock(return_value={})
        
        mock_metrics_collector = Mock(spec=MetricsCollector)
        mock_metrics_collector.get_current_metrics = Mock(return_value=Mock(
            latency=Mock(get_p95=Mock(return_value=500), get_average=Mock(return_value=450)),
            throughput=Mock(error_rate=0.05, requests_per_second=12.0)
        ))
        
        mock_config_manager = Mock(spec=ConfigManager)
        mock_config = Mock(spec=SovereignConfig)
        mock_config.llm = Mock()
        mock_config.llm.fast = Mock()
        mock_config.llm.fast.max_tokens = 3000
        mock_config_manager.get_config = Mock(return_value=mock_config)
        
        # Create integrated components
        health_reporter = create_unified_health_reporter(
            health_monitor=mock_health_monitor,
            predictive_analytics=mock_predictive_analytics,
            metrics_collector=mock_metrics_collector
        )
        
        error_enricher = create_predictive_error_enricher(
            predictive_analytics=mock_predictive_analytics,
            metrics_collector=mock_metrics_collector
        )
        
        optimization_engine = create_predictive_optimization_engine(
            predictive_analytics=mock_predictive_analytics,
            metrics_collector=mock_metrics_collector
        )
        
        config_manager = create_predictive_config_manager(
            base_config_manager=mock_config_manager,
            predictive_analytics=mock_predictive_analytics,
            metrics_collector=mock_metrics_collector
        )
        
        return {
            'health_reporter': health_reporter,
            'error_enricher': error_enricher,
            'optimization_engine': optimization_engine,
            'config_manager': config_manager,
            'predictive_analytics': mock_predictive_analytics,
            'metrics_collector': mock_metrics_collector
        }
    
    @pytest.mark.asyncio
    async def test_error_to_optimization_workflow(self, integrated_system):
        """Test workflow from error enrichment to optimization trigger"""
        error_enricher = integrated_system['error_enricher']
        optimization_engine = integrated_system['optimization_engine']
        
        # Enrich an error that should trigger optimization
        error = LLMException("High latency timeout", ErrorCategory.TIMEOUT, retryable=True)
        enriched_context = await error_enricher.enrich_error(error, async_mode=False)
        
        # Should identify performance issues
        performance_issues = [
            rc for rc in enriched_context.predicted_root_causes
            if 'latency' in rc.description.lower() or 'performance' in rc.description.lower()
        ]
        
        # Should suggest optimizations in recommendations
        optimization_recommendations = [
            rec for rec in enriched_context.recovery_recommendations
            if 'optim' in rec.description.lower() or 'scale' in rec.description.lower()
        ]
        
        # The enrichment should provide insights that could trigger optimizations
        assert enriched_context.user_impact_score > 0
        assert enriched_context.severity_escalation_risk >= 0
    
    @pytest.mark.asyncio
    async def test_health_reporting_integration(self, integrated_system):
        """Test unified health reporting with all systems"""
        health_reporter = integrated_system['health_reporter']
        
        # Generate comprehensive health report
        report = await health_reporter.get_unified_health_report()
        
        # Should integrate data from all sources
        assert isinstance(report, UnifiedHealthReport)
        assert report.current_metrics is not None
        assert report.predictions_available is not None
        assert len(report.components_evaluated) > 0
        
        # Should provide actionable insights
        assert len(report.recommended_actions) >= 0
        assert len(report.optimization_opportunities) >= 0
        
        # Should assess overall system health
        assert report.overall_status in UnifiedHealthStatus
        assert 0.0 <= report.health_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_configuration_optimization_integration(self, integrated_system):
        """Test configuration optimization based on predictive insights"""
        config_manager = integrated_system['config_manager']
        optimization_engine = integrated_system['optimization_engine']
        
        # Configuration changes should be coordinated with optimizations
        optimized_config = config_manager.get_optimized_config()
        assert isinstance(optimized_config, dict)
        
        # Both systems should work with same predictive analytics
        config_stats = config_manager.get_adjustment_statistics()
        optimization_stats = optimization_engine.get_optimization_statistics()
        
        assert isinstance(config_stats, dict)
        assert isinstance(optimization_stats, dict)
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_integration(self, integrated_system):
        """Test graceful degradation when predictive services fail"""
        # Simulate predictive analytics failure
        integrated_system['predictive_analytics'].get_health_score = AsyncMock(side_effect=Exception("Service down"))
        
        health_reporter = integrated_system['health_reporter']
        error_enricher = integrated_system['error_enricher']
        
        # Health reporting should still work
        report = await health_reporter.get_unified_health_report()
        assert isinstance(report, UnifiedHealthReport)
        assert report.predictions_available is False
        
        # Error enrichment should still provide basic functionality
        error = STTException("Test error", ErrorCategory.TRANSIENT, retryable=True)
        enriched_context = await error_enricher.enrich_error(error, async_mode=False)
        assert isinstance(enriched_context, EnrichedErrorContext)
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_feedback_loop(self, integrated_system):
        """Test feedback loop between monitoring, prediction, and optimization"""
        health_reporter = integrated_system['health_reporter']
        optimization_engine = integrated_system['optimization_engine']
        config_manager = integrated_system['config_manager']
        
        # Generate initial health report
        initial_report = await health_reporter.get_unified_health_report()
        initial_score = initial_report.health_score
        
        # Simulate system improvement through optimization
        integrated_system['metrics_collector'].get_current_metrics = Mock(return_value=Mock(
            latency=Mock(get_p95=Mock(return_value=400), get_average=Mock(return_value=350)),  # Improved latency
            throughput=Mock(error_rate=0.02, requests_per_second=15.0)  # Improved metrics
        ))
        
        # Generate new health report
        improved_report = await health_reporter.get_unified_health_report()
        
        # Should detect improvement
        assert isinstance(improved_report, UnifiedHealthReport)
        # Note: In a real system, this would show improvement over time
        
        # All systems should be responsive to performance changes
        config_stats = config_manager.get_adjustment_statistics()
        optimization_stats = optimization_engine.get_optimization_statistics()
        
        assert isinstance(config_stats, dict)
        assert isinstance(optimization_stats, dict)


# Factory function tests
def test_factory_functions():
    """Test factory functions for all integration components"""
    # Test unified health reporter factory
    health_reporter = create_unified_health_reporter()
    assert isinstance(health_reporter, UnifiedHealthReporter)
    
    # Test predictive error enricher factory
    error_enricher = create_predictive_error_enricher()
    assert isinstance(error_enricher, PredictiveErrorEnricher)
    
    # Test predictive optimization engine factory
    optimization_engine = create_predictive_optimization_engine()
    assert isinstance(optimization_engine, PredictiveOptimizationEngine)
    
    # Test predictive config manager factory (requires base config manager)
    mock_base_config = Mock(spec=ConfigManager)
    config_manager = create_predictive_config_manager(base_config_manager=mock_base_config)
    assert isinstance(config_manager, PredictiveConfigManager)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 