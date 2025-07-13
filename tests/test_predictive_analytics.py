"""
Tests for Predictive Analytics and Alerting System (Task 10.4)

Comprehensive test suite covering:
- ML Model Drift Detection
- Resource Usage Forecasting
- Multi-Channel Alerting System
- Performance Degradation Prediction
- Predictive Analytics Framework
"""

import asyncio
import json
import numpy as np
import pandas as pd
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from assistant.drift_detection import (
    DriftDetector, DriftConfig, DriftAlert, DriftType, DriftSeverity,
    create_drift_detector
)
from assistant.resource_forecasting import (
    ResourceForecaster, ResourceForecastConfig, ForecastResult, ResourceType,
    ForecastType, create_resource_forecaster
)
from assistant.alerting_system import (
    AlertingSystem, AlertingConfig, Alert, AlertSeverity, AlertChannel,
    AlertDestination, create_alerting_system, create_email_destination,
    create_webhook_destination
)
from assistant.performance_prediction import (
    PerformancePredictor, PerformancePredictionConfig, PerformancePrediction,
    RiskLevel, create_performance_predictor
)
from assistant.predictive_analytics import (
    PredictiveAnalyticsFramework, PredictiveAnalyticsConfig,
    create_predictive_analytics
)


class TestDriftDetection:
    """Test suite for drift detection system"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir
    
    @pytest.fixture
    def drift_config(self, temp_dir):
        """Create drift detection config with temp directory"""
        return DriftConfig(
            baseline_dir=str(Path(temp_dir) / "baselines"),
            min_samples=10,
            baseline_window=50
        )
    
    @pytest.fixture
    def drift_detector(self, drift_config):
        """Create drift detector instance"""
        return DriftDetector(drift_config)
    
    def test_drift_detector_initialization(self, drift_detector):
        """Test drift detector initialization"""
        assert drift_detector is not None
        assert isinstance(drift_detector.config, DriftConfig)
        assert drift_detector.baselines == {}
        assert drift_detector.baseline_stats == {}
    
    def test_establish_baseline(self, drift_detector):
        """Test establishing baseline for drift detection"""
        # Create baseline data
        baseline_data = np.random.normal(0.8, 0.1, 100)
        
        # Establish baseline
        success = drift_detector.establish_baseline("test_metric", baseline_data)
        
        assert success is True
        assert "test_metric" in drift_detector.baselines
        assert "test_metric" in drift_detector.baseline_stats
        
        stats = drift_detector.baseline_stats["test_metric"]
        assert "mean" in stats
        assert "std" in stats
        assert stats["samples"] == 100
    
    def test_establish_baseline_insufficient_data(self, drift_detector):
        """Test baseline establishment with insufficient data"""
        # Not enough data points
        baseline_data = np.random.normal(0.8, 0.1, 5)
        
        success = drift_detector.establish_baseline("test_metric", baseline_data)
        
        assert success is False
        assert "test_metric" not in drift_detector.baselines
    
    def test_psi_calculation(self, drift_detector):
        """Test Population Stability Index calculation"""
        # Create expected and actual distributions
        expected = np.random.normal(0.8, 0.1, 1000)
        actual_stable = np.random.normal(0.8, 0.1, 500)  # Similar distribution
        actual_drifted = np.random.normal(0.6, 0.15, 500)  # Drifted distribution
        
        # Calculate PSI for stable data
        psi_stable = drift_detector.calculate_psi(expected, actual_stable)
        
        # Calculate PSI for drifted data
        psi_drifted = drift_detector.calculate_psi(expected, actual_drifted)
        
        # Drifted data should have higher PSI
        assert psi_drifted > psi_stable
        assert psi_stable < 0.1  # Should be low for similar distributions
        assert psi_drifted > 0.1  # Should be higher for different distributions
    
    def test_ks_statistic_calculation(self, drift_detector):
        """Test Kolmogorov-Smirnov test calculation"""
        # Create distributions
        expected = np.random.normal(0.8, 0.1, 1000)
        actual_stable = np.random.normal(0.8, 0.1, 500)
        actual_drifted = np.random.normal(0.6, 0.1, 500)
        
        # Calculate KS statistics
        ks_stable, p_stable = drift_detector.calculate_ks_statistic(expected, actual_stable)
        ks_drifted, p_drifted = drift_detector.calculate_ks_statistic(expected, actual_drifted)
        
        # Drifted data should have higher KS statistic and lower p-value
        assert ks_drifted > ks_stable
        assert p_drifted < p_stable
    
    def test_js_divergence_calculation(self, drift_detector):
        """Test Jensen-Shannon divergence calculation"""
        # Create distributions
        expected = np.random.normal(0.8, 0.1, 1000)
        actual_stable = np.random.normal(0.8, 0.1, 500)
        actual_drifted = np.random.normal(0.5, 0.2, 500)
        
        # Calculate JS divergence
        js_stable = drift_detector.calculate_js_divergence(expected, actual_stable)
        js_drifted = drift_detector.calculate_js_divergence(expected, actual_drifted)
        
        # Drifted data should have higher JS divergence
        assert js_drifted > js_stable
        assert 0 <= js_stable <= 1
        assert 0 <= js_drifted <= 1
    
    def test_drift_detection_no_drift(self, drift_detector):
        """Test drift detection with stable data"""
        # Establish baseline
        baseline_data = np.random.normal(0.8, 0.1, 100)
        drift_detector.establish_baseline("test_metric", baseline_data)
        
        # Test with similar data
        current_data = np.random.normal(0.8, 0.1, 50)
        alerts = drift_detector.detect_drift("test_metric", current_data)
        
        # Should have minimal or no alerts
        high_severity_alerts = [a for a in alerts if a.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]]
        assert len(high_severity_alerts) == 0
    
    def test_drift_detection_with_drift(self, drift_detector):
        """Test drift detection with drifted data"""
        # Establish baseline
        baseline_data = np.random.normal(0.8, 0.1, 100)
        drift_detector.establish_baseline("test_metric", baseline_data)
        
        # Test with significantly different data
        current_data = np.random.normal(0.5, 0.2, 50)
        alerts = drift_detector.detect_drift("test_metric", current_data)
        
        # Should detect drift
        assert len(alerts) > 0
        assert any(alert.severity in [DriftSeverity.MEDIUM, DriftSeverity.HIGH, DriftSeverity.CRITICAL] 
                  for alert in alerts)
    
    def test_performance_drift_detection(self, drift_detector):
        """Test performance-based drift detection"""
        # Establish baseline with good performance
        baseline_data = np.random.normal(0.9, 0.05, 100)
        drift_detector.establish_baseline("accuracy_score", baseline_data)
        
        # Add performance history
        for _ in range(50):
            perf = np.random.normal(0.9, 0.05)
            drift_detector.detect_drift("accuracy_score", np.array([perf]), current_performance=perf)
        
        # Test with degraded performance
        current_data = np.array([0.75])  # Significant drop
        alerts = drift_detector.detect_drift("accuracy_score", current_data, current_performance=0.75)
        
        # Should detect performance drift
        performance_alerts = [a for a in alerts if a.drift_type == DriftType.PERFORMANCE]
        assert len(performance_alerts) > 0
    
    def test_drift_summary(self, drift_detector):
        """Test drift detection summary generation"""
        # Establish baselines for multiple metrics
        metrics = ["latency", "accuracy", "memory_usage"]
        for metric in metrics:
            baseline_data = np.random.normal(0.5, 0.1, 100)
            drift_detector.establish_baseline(metric, baseline_data)
        
        # Generate some alerts
        current_data = np.random.normal(0.3, 0.2, 50)
        drift_detector.detect_drift("latency", current_data)
        
        # Get summary
        summary = drift_detector.get_drift_summary()
        
        assert "timestamp" in summary
        assert "total_metrics_monitored" in summary
        assert summary["total_metrics_monitored"] == 3
        assert "metrics" in summary
        assert len(summary["metrics"]) <= 3


class TestResourceForecasting:
    """Test suite for resource forecasting system"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir
    
    @pytest.fixture
    def forecast_config(self, temp_dir):
        """Create forecasting config with temp directory"""
        return ResourceForecastConfig(
            model_dir=str(Path(temp_dir) / "models"),
            min_training_samples=50,
            save_models=False  # Disable to avoid TensorFlow/Prophet dependency issues in tests
        )
    
    @pytest.fixture
    def resource_forecaster(self, forecast_config):
        """Create resource forecaster instance"""
        return ResourceForecaster(forecast_config)
    
    def test_forecaster_initialization(self, resource_forecaster):
        """Test resource forecaster initialization"""
        assert resource_forecaster is not None
        assert isinstance(resource_forecaster.config, ResourceForecastConfig)
        assert resource_forecaster.prophet_models == {}
        assert resource_forecaster.lstm_models == {}
    
    @pytest.fixture
    def sample_data(self):
        """Create sample resource usage data"""
        # Generate 200 hours of synthetic data
        timestamps = pd.date_range(start='2024-01-01', periods=200, freq='H')
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'cpu_usage': np.random.beta(2, 5, 200) * 0.8 + 0.1,  # Realistic CPU usage
            'memory_usage': np.random.beta(3, 4, 200) * 0.7 + 0.2,  # Realistic memory usage
            'gpu_usage': np.random.beta(2, 8, 200) * 0.6 + 0.1,  # Realistic GPU usage
        })
        
        # Add some trend and seasonality
        hourly_cycle = np.sin(2 * np.pi * np.arange(200) / 24) * 0.1
        data['cpu_usage'] += hourly_cycle
        data['memory_usage'] += hourly_cycle * 0.5
        
        return data
    
    @patch('assistant.resource_forecasting.PROPHET_AVAILABLE', False)
    @patch('assistant.resource_forecasting.TENSORFLOW_AVAILABLE', False)
    def test_training_without_dependencies(self, resource_forecaster, sample_data):
        """Test training when Prophet/TensorFlow are not available"""
        results = resource_forecaster.train_models(sample_data, ['cpu_usage'])
        
        # Should handle missing dependencies gracefully
        assert isinstance(results, dict)
        assert 'cpu_usage' in results
        assert results['cpu_usage'] is False  # Training should fail without dependencies
    
    def test_anomaly_detector_training(self, resource_forecaster, sample_data):
        """Test anomaly detector training (doesn't require Prophet/TensorFlow)"""
        success = resource_forecaster.train_anomaly_detector(sample_data, 'cpu_usage')
        
        assert success is True
        assert 'cpu_usage' in resource_forecaster.anomaly_detectors
    
    def test_anomaly_detection(self, resource_forecaster, sample_data):
        """Test anomaly detection functionality"""
        # Train anomaly detector
        resource_forecaster.train_anomaly_detector(sample_data, 'cpu_usage')
        
        # Create test data with anomalies
        test_data = sample_data.tail(20).copy()
        test_data.loc[test_data.index[-1], 'cpu_usage'] = 0.95  # Add anomaly
        
        # Detect anomalies
        anomalies = resource_forecaster.detect_anomalies(test_data, 'cpu_usage')
        
        # Should detect the anomaly
        assert len(anomalies) > 0
        assert any(anomaly.is_anomaly for anomaly in anomalies)
    
    def test_forecast_summary(self, resource_forecaster):
        """Test forecast summary generation"""
        summary = resource_forecaster.get_forecast_summary()
        
        assert "timestamp" in summary
        assert "models_available" in summary
        assert "resource_types" in summary
        assert "last_training" in summary
        
        # Check model counts
        models = summary["models_available"]
        assert "prophet" in models
        assert "lstm" in models
        assert "anomaly_detectors" in models


class TestAlertingSystem:
    """Test suite for multi-channel alerting system"""
    
    @pytest.fixture
    def alerting_config(self):
        """Create alerting system config"""
        return AlertingConfig(
            persistence_enabled=False,
            global_rate_limit_per_hour=10
        )
    
    @pytest.fixture
    def alerting_system(self, alerting_config):
        """Create alerting system instance"""
        return AlertingSystem(alerting_config)
    
    def test_alerting_system_initialization(self, alerting_system):
        """Test alerting system initialization"""
        assert alerting_system is not None
        assert isinstance(alerting_system.config, AlertingConfig)
        assert alerting_system.destinations == {}
        assert len(alerting_system.correlation_rules) > 0  # Default rules loaded
    
    def test_add_destination(self, alerting_system):
        """Test adding alert destinations"""
        # Add email destination
        email_dest = create_email_destination("test_email", "test@example.com")
        alerting_system.add_destination(email_dest)
        
        assert "test_email" in alerting_system.destinations
        assert alerting_system.destinations["test_email"].channel == AlertChannel.EMAIL
        
        # Add webhook destination
        webhook_dest = create_webhook_destination("test_webhook", "http://example.com/webhook")
        alerting_system.add_destination(webhook_dest)
        
        assert "test_webhook" in alerting_system.destinations
        assert alerting_system.destinations["test_webhook"].channel == AlertChannel.WEBHOOK
    
    def test_alert_creation(self, alerting_system):
        """Test alert creation"""
        alert = alerting_system.create_alert(
            title="Test Alert",
            description="This is a test alert",
            source="test_system",
            severity=AlertSeverity.WARNING,
            metric_name="test_metric",
            metric_value=0.8,
            threshold=0.7
        )
        
        assert alert.title == "Test Alert"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.metric_name == "test_metric"
        assert alert.metric_value == 0.8
        assert alert.threshold == 0.7
        assert alert.id is not None
    
    def test_rate_limiting(self, alerting_system):
        """Test rate limiting functionality"""
        # Add destination with low rate limit
        dest = create_email_destination("rate_test", "test@example.com")
        dest.rate_limit_per_hour = 2
        alerting_system.add_destination(dest)
        
        # Test rate limiting
        assert alerting_system._check_rate_limit("rate_test") is True
        assert alerting_system._check_rate_limit("rate_test") is True
        assert alerting_system._check_rate_limit("rate_test") is False  # Should be rate limited
    
    def test_correlation_rules(self, alerting_system):
        """Test alert correlation functionality"""
        # Create related alerts
        alert1 = alerting_system.create_alert(
            title="High CPU Usage",
            description="CPU usage is high",
            source="cpu_usage_high",
            severity=AlertSeverity.WARNING
        )
        
        alert2 = alerting_system.create_alert(
            title="High Memory Usage", 
            description="Memory usage is high",
            source="memory_usage_high",
            severity=AlertSeverity.WARNING
        )
        
        # Add to history to enable correlation
        alerting_system.alert_history.append(alert1)
        
        # Find correlations for second alert
        matching_rules = alerting_system._find_correlations(alert2)
        
        # Should find resource exhaustion correlation rule
        assert len(matching_rules) > 0
        rule_names = [rule.name for rule in matching_rules]
        assert "resource_exhaustion" in rule_names
    
    @pytest.mark.asyncio
    async def test_webhook_sending(self, alerting_system):
        """Test webhook alert sending"""
        # Mock aiohttp session
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value="OK")
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            # Add webhook destination
            webhook_dest = create_webhook_destination("test_webhook", "http://example.com/webhook")
            alerting_system.add_destination(webhook_dest)
            
            # Create and send alert
            alert = alerting_system.create_alert(
                title="Test Webhook Alert",
                description="Testing webhook functionality",
                source="test_webhook_system",
                severity=AlertSeverity.CRITICAL
            )
            
            result = await alerting_system.send_alert(alert)
            
            assert result["status"] == "sent"
            assert "test_webhook" in result["results"]
            assert result["results"]["test_webhook"] == "sent"
    
    def test_alert_enrichment(self, alerting_system):
        """Test alert enrichment functionality"""
        # Add enrichment callback
        def test_enrichment(alert):
            alert.context["enriched"] = True
            alert.remediation.append("Test remediation")
            return alert
        
        alerting_system.add_enrichment_callback(test_enrichment)
        
        # Create alert
        alert = alerting_system.create_alert(
            title="Test Enrichment",
            description="Testing enrichment",
            source="test_enrichment"
        )
        
        # Enrich alert
        enriched_alert = alerting_system._enrich_alert(alert)
        
        assert enriched_alert.context.get("enriched") is True
        assert "Test remediation" in enriched_alert.remediation
    
    def test_alert_summary(self, alerting_system):
        """Test alert summary generation"""
        # Create some test alerts
        for i in range(5):
            alert = alerting_system.create_alert(
                title=f"Test Alert {i}",
                description=f"Test alert {i}",
                source=f"test_source_{i % 2}",
                severity=AlertSeverity.WARNING if i % 2 == 0 else AlertSeverity.CRITICAL
            )
            alerting_system.alert_history.append(alert)
        
        # Get summary
        summary = alerting_system.get_alert_summary()
        
        assert "total_alerts" in summary
        assert "by_severity" in summary
        assert "by_source" in summary
        assert summary["total_alerts"] == 5
        assert summary["by_severity"]["warning"] == 3
        assert summary["by_severity"]["critical"] == 2


class TestPerformancePrediction:
    """Test suite for performance prediction system"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir
    
    @pytest.fixture
    def prediction_config(self, temp_dir):
        """Create prediction config with temp directory"""
        return PerformancePredictionConfig(
            model_dir=str(Path(temp_dir) / "models"),
            min_training_samples=50,
            save_models=False
        )
    
    @pytest.fixture
    def performance_predictor(self, prediction_config):
        """Create performance predictor instance"""
        return PerformancePredictor(prediction_config)
    
    @pytest.fixture
    def sample_performance_data(self):
        """Create sample performance data"""
        # Generate 200 hours of synthetic performance data
        timestamps = pd.date_range(start='2024-01-01', periods=200, freq='H')
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'latency_p95': np.random.gamma(2, 100) + 200,  # Realistic latency values
            'accuracy_score': np.random.beta(8, 2) * 0.2 + 0.8,  # High accuracy
            'bleu_score': np.random.beta(6, 3) * 0.3 + 0.7,  # Good BLEU scores
            'cpu_usage': np.random.beta(3, 7) * 0.6 + 0.2,  # Moderate CPU usage
            'memory_usage': np.random.beta(4, 6) * 0.7 + 0.2,  # Moderate memory usage
            'error_rate': np.random.beta(1, 20) * 0.1,  # Low error rate
            'request_rate': np.random.poisson(50) + 20  # Request rate
        })
        
        # Add some degradation over time
        degradation_trend = np.linspace(0, 0.1, 200)
        data['latency_p95'] += degradation_trend * 200
        data['accuracy_score'] -= degradation_trend * 0.1
        data['cpu_usage'] += degradation_trend * 0.2
        
        return data
    
    def test_predictor_initialization(self, performance_predictor):
        """Test performance predictor initialization"""
        assert performance_predictor is not None
        assert isinstance(performance_predictor.config, PerformancePredictionConfig)
        assert performance_predictor.is_trained is False
        assert len(performance_predictor.models) == 3  # RF, GB, LR
    
    def test_feature_engineering(self, performance_predictor, sample_performance_data):
        """Test feature engineering functionality"""
        features_df = performance_predictor._engineer_features(sample_performance_data)
        
        # Check that features were created
        assert len(features_df) > 0
        assert 'hour_of_day' in features_df.columns
        assert 'day_of_week' in features_df.columns
        assert 'is_weekend' in features_df.columns
        
        # Check rolling features
        cpu_features = [col for col in features_df.columns if 'cpu_usage' in col and 'avg' in col]
        assert len(cpu_features) > 0
        
        # Check trend features
        trend_features = [col for col in features_df.columns if 'trend' in col]
        assert len(trend_features) > 0
    
    def test_target_variable_creation(self, performance_predictor, sample_performance_data):
        """Test target variable creation for degradation prediction"""
        target = performance_predictor._create_target_variable(sample_performance_data)
        
        assert len(target) == len(sample_performance_data)
        assert all(0 <= score <= 1 for score in target)
        
        # Later data points should have higher degradation scores due to trend
        early_scores = target[:50]
        late_scores = target[-50:]
        assert np.mean(late_scores) > np.mean(early_scores)
    
    def test_model_training(self, performance_predictor, sample_performance_data):
        """Test model training functionality"""
        success = performance_predictor.train_models(sample_performance_data)
        
        assert success is True
        assert performance_predictor.is_trained is True
        assert performance_predictor.last_training_time is not None
        assert len(performance_predictor.feature_names) > 0
        assert len(performance_predictor.model_performance) > 0
    
    def test_prediction_without_training(self, performance_predictor, sample_performance_data):
        """Test prediction without training (should fail gracefully)"""
        prediction = performance_predictor.predict_degradation(sample_performance_data.tail(24))
        
        assert prediction is None
    
    def test_prediction_with_training(self, performance_predictor, sample_performance_data):
        """Test prediction after training"""
        # Train models
        performance_predictor.train_models(sample_performance_data)
        
        # Make prediction
        prediction = performance_predictor.predict_degradation(sample_performance_data.tail(24))
        
        assert prediction is not None
        assert isinstance(prediction, PerformancePrediction)
        assert 0 <= prediction.degradation_probability <= 1
        assert prediction.risk_level in list(RiskLevel)
        assert len(prediction.contributing_factors) >= 0
        assert len(prediction.recommended_actions) > 0
    
    def test_risk_level_assessment(self, performance_predictor):
        """Test risk level assessment"""
        # Test different probability levels
        assert performance_predictor._assess_risk_level(0.1) == RiskLevel.LOW
        assert performance_predictor._assess_risk_level(0.5) == RiskLevel.MEDIUM
        assert performance_predictor._assess_risk_level(0.8) == RiskLevel.HIGH
        assert performance_predictor._assess_risk_level(0.9) == RiskLevel.CRITICAL
    
    def test_early_warnings(self, performance_predictor, sample_performance_data):
        """Test early warning system"""
        # Create data with concerning trends
        warning_data = sample_performance_data.tail(24).copy()
        
        # Add rapid latency increase
        warning_data.loc[warning_data.index[-4:], 'latency_p95'] *= 2
        
        warnings = performance_predictor.check_early_warnings(warning_data)
        
        # Should detect latency trend warning
        latency_warnings = [w for w in warnings if 'latency' in w.warning_type]
        assert len(latency_warnings) > 0
    
    def test_prediction_summary(self, performance_predictor, sample_performance_data):
        """Test prediction summary generation"""
        # Train models first
        performance_predictor.train_models(sample_performance_data)
        
        summary = performance_predictor.get_prediction_summary()
        
        assert "timestamp" in summary
        assert "is_trained" in summary
        assert "model_performance" in summary
        assert "feature_count" in summary
        assert summary["is_trained"] is True
        assert summary["feature_count"] > 0


class TestPredictiveAnalyticsFramework:
    """Test suite for the main predictive analytics framework"""
    
    @pytest.fixture
    def analytics_config(self):
        """Create analytics framework config"""
        return PredictiveAnalyticsConfig(
            collection_interval=10,  # Short intervals for testing
            analysis_interval=20,
            training_interval=3600,
            enable_drift_detection=True,
            enable_resource_forecasting=True,
            enable_performance_prediction=True,
            enable_early_warnings=True,
            enable_predictive_alerts=True
        )
    
    @pytest.fixture
    def mock_metrics_collector(self):
        """Create mock metrics collector"""
        mock_collector = Mock()
        mock_collector.get_current_metrics.return_value = {
            'latency_p95': 500.0,
            'accuracy_score': 0.85,
            'cpu_usage': 0.6,
            'memory_usage': 0.7,
            'gpu_usage': 0.4
        }
        return mock_collector
    
    @pytest.fixture
    def analytics_framework(self, analytics_config):
        """Create analytics framework instance"""
        return PredictiveAnalyticsFramework(analytics_config)
    
    def test_framework_initialization(self, analytics_framework):
        """Test framework initialization"""
        assert analytics_framework is not None
        assert isinstance(analytics_framework.config, PredictiveAnalyticsConfig)
        assert analytics_framework.is_running is False
        assert analytics_framework.collected_data.empty
    
    @patch('assistant.predictive_analytics.get_metrics_collector')
    def test_data_collection(self, mock_get_collector, analytics_framework, mock_metrics_collector):
        """Test data collection functionality"""
        mock_get_collector.return_value = mock_metrics_collector
        analytics_framework.metrics_collector = mock_metrics_collector
        
        # Collect data
        asyncio.run(analytics_framework._collect_data())
        
        # Check that data was collected
        assert len(analytics_framework.collected_data) == 1
        assert 'timestamp' in analytics_framework.collected_data.columns
        assert 'latency_p95' in analytics_framework.collected_data.columns
        assert 'cpu_usage' in analytics_framework.collected_data.columns
    
    def test_health_score_calculation(self, analytics_framework):
        """Test system health score calculation"""
        # Test with no issues
        health_score = analytics_framework._calculate_health_score([], None, [])
        assert health_score == 1.0
        
        # Test with drift alerts
        from assistant.drift_detection import DriftAlert, DriftType, DriftSeverity
        
        drift_alert = DriftAlert(
            timestamp=time.time(),
            drift_type=DriftType.STATISTICAL,
            severity=DriftSeverity.HIGH,
            metric_name="test_metric",
            drift_score=0.3,
            threshold=0.2,
            description="Test drift alert"
        )
        
        health_score = analytics_framework._calculate_health_score([drift_alert], None, [])
        assert health_score < 1.0  # Should be reduced due to drift alert
    
    def test_recommendations_generation(self, analytics_framework):
        """Test recommendations generation"""
        recommendations = analytics_framework._generate_recommendations([], {}, None, [])
        assert isinstance(recommendations, list)
        
        # Test with drift alerts
        from assistant.drift_detection import DriftAlert, DriftType, DriftSeverity
        
        drift_alert = DriftAlert(
            timestamp=time.time(),
            drift_type=DriftType.STATISTICAL,
            severity=DriftSeverity.HIGH,
            metric_name="test_metric",
            drift_score=0.3,
            threshold=0.2,
            description="Test drift alert"
        )
        
        recommendations = analytics_framework._generate_recommendations([drift_alert], {}, None, [])
        assert len(recommendations) > 0
        assert any("model drift" in rec.lower() for rec in recommendations)
    
    def test_analytics_summary(self, analytics_framework):
        """Test analytics summary generation"""
        summary = analytics_framework.get_analytics_summary()
        
        assert "timestamp" in summary
        assert "mode" in summary
        assert "is_running" in summary
        assert "data_points" in summary
        assert "components_enabled" in summary
        
        components = summary["components_enabled"]
        assert "drift_detection" in components
        assert "resource_forecasting" in components
        assert "performance_prediction" in components
    
    def test_force_analysis(self, analytics_framework):
        """Test forced analysis execution"""
        result = analytics_framework.force_analysis()
        
        assert "status" in result
        # Should handle gracefully even without data
        assert result["status"] in ["analysis_scheduled", "analysis_completed", "error"]


# Integration tests
class TestPredictiveAnalyticsIntegration:
    """Integration tests for the complete predictive analytics system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_analytics_flow(self):
        """Test complete end-to-end analytics workflow"""
        # Create framework with all components enabled
        config = PredictiveAnalyticsConfig(
            collection_interval=1,
            analysis_interval=2,
            min_data_points=10
        )
        
        framework = PredictiveAnalyticsFramework(config)
        
        # Mock metrics collector
        mock_collector = Mock()
        mock_collector.get_current_metrics.return_value = {
            'latency_p95': 500.0,
            'accuracy_score': 0.85,
            'cpu_usage': 0.6,
            'memory_usage': 0.7
        }
        
        framework.metrics_collector = mock_collector
        
        # Collect enough data for analysis
        for _ in range(15):
            await framework._collect_data()
            await asyncio.sleep(0.01)  # Small delay
        
        # Run analysis
        await framework._run_analysis()
        
        # Check that analysis was performed
        assert len(framework.analysis_history) > 0
        
        # Get summary
        summary = framework.get_analytics_summary()
        assert summary["data_points"] >= 10
    
    def test_factory_functions(self):
        """Test factory functions for all components"""
        # Test all factory functions
        drift_detector = create_drift_detector()
        assert isinstance(drift_detector, DriftDetector)
        
        forecaster = create_resource_forecaster()
        assert isinstance(forecaster, ResourceForecaster)
        
        alerting_system = create_alerting_system()
        assert isinstance(alerting_system, AlertingSystem)
        
        predictor = create_performance_predictor()
        assert isinstance(predictor, PerformancePredictor)
        
        analytics = create_predictive_analytics()
        assert isinstance(analytics, PredictiveAnalyticsFramework)
    
    def test_component_integration(self):
        """Test integration between different components"""
        # Create all components
        drift_detector = create_drift_detector()
        forecaster = create_resource_forecaster()
        alerting_system = create_alerting_system()
        predictor = create_performance_predictor()
        
        # Test that they can work together
        analytics = PredictiveAnalyticsFramework()
        analytics.drift_detector = drift_detector
        analytics.resource_forecaster = forecaster
        analytics.alerting_system = alerting_system
        analytics.performance_predictor = predictor
        
        # Should initialize without errors
        assert analytics.drift_detector is not None
        assert analytics.resource_forecaster is not None
        assert analytics.alerting_system is not None
        assert analytics.performance_predictor is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 