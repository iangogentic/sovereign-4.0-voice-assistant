"""
Predictive Analytics Framework

Main orchestrator for the predictive analytics and alerting system.
Integrates all components:
- ML Model Drift Detection
- Resource Usage Forecasting 
- Multi-Channel Alerting System
- Performance Degradation Prediction
- Early Warning System
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

from .drift_detection import DriftDetector, DriftConfig, DriftAlert, get_drift_detector
from .resource_forecasting import ResourceForecaster, ResourceForecastConfig, ForecastResult, get_resource_forecaster
from .alerting_system import AlertingSystem, AlertingConfig, Alert, AlertSeverity, get_alerting_system
from .performance_prediction import PerformancePredictor, PerformancePredictionConfig, PerformancePrediction, get_performance_predictor
from .metrics_collector import MetricsCollector, get_metrics_collector


class AnalyticsMode(Enum):
    """Analytics operating modes"""
    MONITORING = "monitoring"
    TRAINING = "training"
    PREDICTION = "prediction"
    MAINTENANCE = "maintenance"


@dataclass
class PredictiveAnalyticsConfig:
    """Configuration for predictive analytics framework"""
    # Collection settings
    collection_interval: int = 300  # 5 minutes
    analysis_interval: int = 900    # 15 minutes
    training_interval: int = 86400  # 24 hours
    
    # Data requirements
    min_data_points: int = 100
    max_data_age_hours: int = 168   # 1 week
    
    # Analysis settings
    enable_drift_detection: bool = True
    enable_resource_forecasting: bool = True
    enable_performance_prediction: bool = True
    enable_early_warnings: bool = True
    
    # Alert settings
    enable_predictive_alerts: bool = True
    alert_cooldown_minutes: int = 30
    
    # Component configs
    drift_config: Optional[DriftConfig] = None
    forecast_config: Optional[ResourceForecastConfig] = None
    alerting_config: Optional[AlertingConfig] = None
    prediction_config: Optional[PerformancePredictionConfig] = None


@dataclass
class AnalyticsReport:
    """Comprehensive analytics report"""
    timestamp: float
    data_points_analyzed: int
    drift_alerts: List[DriftAlert]
    resource_forecasts: Dict[str, ForecastResult]
    performance_prediction: Optional[PerformancePrediction]
    early_warnings: List[Any]
    system_health_score: float
    recommendations: List[str]
    next_training_due: float
    analysis_duration: float


class PredictiveAnalyticsFramework:
    """
    Main predictive analytics framework
    
    Orchestrates all predictive analytics components to provide
    comprehensive monitoring, forecasting, and alerting capabilities.
    """
    
    def __init__(self, config: Optional[PredictiveAnalyticsConfig] = None):
        self.config = config or PredictiveAnalyticsConfig()
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
        
        # Operating state
        self.mode = AnalyticsMode.MONITORING
        self.is_running = False
        self.last_collection_time = 0
        self.last_analysis_time = 0
        self.last_training_time = 0
        
        # Components
        self.metrics_collector: Optional[MetricsCollector] = None
        self.drift_detector: Optional[DriftDetector] = None
        self.resource_forecaster: Optional[ResourceForecaster] = None
        self.alerting_system: Optional[AlertingSystem] = None
        self.performance_predictor: Optional[PerformancePredictor] = None
        
        # Data storage
        self.collected_data = pd.DataFrame()
        self.analysis_history = []
        self.alert_cooldowns = {}
        
        # Monitoring tasks
        self.monitoring_task = None
        self.training_task = None
        
        # Initialize components
        self._initialize_components()
        
        # Setup alert enrichment
        self._setup_alert_enrichment()
    
    def _initialize_components(self):
        """Initialize all analytics components"""
        try:
            # Get or create metrics collector
            self.metrics_collector = get_metrics_collector()
            
            # Initialize drift detector
            if self.config.enable_drift_detection:
                drift_config = self.config.drift_config or DriftConfig()
                self.drift_detector = DriftDetector(drift_config)
                self.logger.info("Initialized drift detector")
            
            # Initialize resource forecaster
            if self.config.enable_resource_forecasting:
                forecast_config = self.config.forecast_config or ResourceForecastConfig()
                self.resource_forecaster = ResourceForecaster(forecast_config)
                self.logger.info("Initialized resource forecaster")
            
            # Initialize alerting system
            if self.config.enable_predictive_alerts:
                alerting_config = self.config.alerting_config or AlertingConfig()
                self.alerting_system = AlertingSystem(alerting_config)
                self.logger.info("Initialized alerting system")
            
            # Initialize performance predictor
            if self.config.enable_performance_prediction:
                prediction_config = self.config.prediction_config or PerformancePredictionConfig()
                self.performance_predictor = PerformancePredictor(prediction_config)
                self.logger.info("Initialized performance predictor")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise
    
    def _setup_alert_enrichment(self):
        """Setup alert enrichment callbacks"""
        if not self.alerting_system:
            return
        
        def enrich_with_analytics(alert: Alert) -> Alert:
            """Enrich alerts with predictive analytics context"""
            try:
                # Add current system state
                if self.metrics_collector:
                    current_metrics = self.metrics_collector.get_current_metrics()
                    alert.context.update({
                        'current_cpu': current_metrics.get('cpu_usage'),
                        'current_memory': current_metrics.get('memory_usage'),
                        'current_latency_p95': current_metrics.get('latency_p95')
                    })
                
                # Add prediction context
                if self.performance_predictor and len(self.collected_data) > 0:
                    prediction = self.performance_predictor.predict_degradation(
                        self.collected_data.tail(24)  # Last 24 hours
                    )
                    if prediction:
                        alert.context.update({
                            'degradation_risk': prediction.risk_level.value,
                            'degradation_probability': prediction.degradation_probability,
                            'predicted_factors': prediction.contributing_factors[:3]
                        })
                
                # Add trend information
                if len(self.collected_data) >= 12:
                    recent_data = self.collected_data.tail(12)
                    if alert.metric_name and alert.metric_name in recent_data.columns:
                        values = recent_data[alert.metric_name].values
                        if len(values) >= 2:
                            trend = np.polyfit(range(len(values)), values, 1)[0]
                            trend_direction = "increasing" if trend > 0 else "decreasing"
                            alert.context['trend_direction'] = trend_direction
                            alert.context['trend_magnitude'] = abs(trend)
                
                return alert
                
            except Exception as e:
                self.logger.error(f"Error enriching alert: {e}")
                return alert
        
        self.alerting_system.add_enrichment_callback(enrich_with_analytics)
    
    async def start_monitoring(self):
        """Start the predictive analytics monitoring"""
        if self.is_running:
            self.logger.warning("Analytics framework already running")
            return
        
        self.is_running = True
        self.logger.info("Starting predictive analytics framework")
        
        # Start monitoring tasks
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.training_task = asyncio.create_task(self._training_loop())
        
        await asyncio.gather(self.monitoring_task, self.training_task)
    
    async def stop_monitoring(self):
        """Stop the predictive analytics monitoring"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.logger.info("Stopping predictive analytics framework")
        
        # Cancel tasks
        if self.monitoring_task:
            self.monitoring_task.cancel()
        if self.training_task:
            self.training_task.cancel()
        
        # Wait for cleanup
        try:
            await asyncio.gather(self.monitoring_task, self.training_task, return_exceptions=True)
        except Exception:
            pass
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Check if it's time to collect data
                if current_time - self.last_collection_time >= self.config.collection_interval:
                    await self._collect_data()
                    self.last_collection_time = current_time
                
                # Check if it's time to run analysis
                if current_time - self.last_analysis_time >= self.config.analysis_interval:
                    await self._run_analysis()
                    self.last_analysis_time = current_time
                
                # Sleep until next check
                await asyncio.sleep(min(60, self.config.collection_interval // 4))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _training_loop(self):
        """Training loop for model updates"""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Check if it's time to retrain models
                if current_time - self.last_training_time >= self.config.training_interval:
                    await self._retrain_models()
                    self.last_training_time = current_time
                
                # Sleep for 1 hour between checks
                await asyncio.sleep(3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in training loop: {e}")
                await asyncio.sleep(3600)
    
    async def _collect_data(self):
        """Collect current metrics data"""
        try:
            if not self.metrics_collector:
                return
            
            # Get current metrics
            metrics = self.metrics_collector.get_current_metrics()
            
            if not metrics:
                return
            
            # Convert to DataFrame row
            row_data = {
                'timestamp': time.time(),
                **metrics
            }
            
            # Add to collected data
            new_row = pd.DataFrame([row_data])
            self.collected_data = pd.concat([self.collected_data, new_row], ignore_index=True)
            
            # Limit data size
            max_rows = self.config.max_data_age_hours * 12  # 5-minute intervals
            if len(self.collected_data) > max_rows:
                self.collected_data = self.collected_data.tail(max_rows)
            
            self.logger.debug(f"Collected metrics: {len(self.collected_data)} total rows")
            
        except Exception as e:
            self.logger.error(f"Error collecting data: {e}")
    
    async def _run_analysis(self):
        """Run comprehensive predictive analysis"""
        analysis_start = time.time()
        
        try:
            if len(self.collected_data) < self.config.min_data_points:
                self.logger.debug(f"Insufficient data for analysis: {len(self.collected_data)}")
                return
            
            # Initialize analysis result
            drift_alerts = []
            resource_forecasts = {}
            performance_prediction = None
            early_warnings = []
            recommendations = []
            
            # 1. Drift Detection
            if self.config.enable_drift_detection and self.drift_detector:
                drift_alerts = await self._run_drift_detection()
            
            # 2. Resource Forecasting
            if self.config.enable_resource_forecasting and self.resource_forecaster:
                resource_forecasts = await self._run_resource_forecasting()
            
            # 3. Performance Prediction
            if self.config.enable_performance_prediction and self.performance_predictor:
                performance_prediction = await self._run_performance_prediction()
            
            # 4. Early Warning System
            if self.config.enable_early_warnings and self.performance_predictor:
                early_warnings = await self._run_early_warnings()
            
            # 5. Generate alerts
            if self.config.enable_predictive_alerts and self.alerting_system:
                await self._process_predictive_alerts(
                    drift_alerts, resource_forecasts, performance_prediction, early_warnings
                )
            
            # 6. Calculate system health score
            health_score = self._calculate_health_score(
                drift_alerts, performance_prediction, early_warnings
            )
            
            # 7. Generate recommendations
            recommendations = self._generate_recommendations(
                drift_alerts, resource_forecasts, performance_prediction, early_warnings
            )
            
            # Create analysis report
            report = AnalyticsReport(
                timestamp=time.time(),
                data_points_analyzed=len(self.collected_data),
                drift_alerts=drift_alerts,
                resource_forecasts=resource_forecasts,
                performance_prediction=performance_prediction,
                early_warnings=early_warnings,
                system_health_score=health_score,
                recommendations=recommendations,
                next_training_due=self.last_training_time + self.config.training_interval,
                analysis_duration=time.time() - analysis_start
            )
            
            # Store in history
            self.analysis_history.append(report)
            if len(self.analysis_history) > 100:
                self.analysis_history = self.analysis_history[-100:]
            
            self.logger.info(f"Analysis complete: health={health_score:.2f}, alerts={len(drift_alerts)}")
            
        except Exception as e:
            self.logger.error(f"Error running analysis: {e}")
    
    async def _run_drift_detection(self) -> List[DriftAlert]:
        """Run drift detection analysis"""
        alerts = []
        
        try:
            # Check drift for key metrics
            metrics_to_check = ['latency_p95', 'accuracy_score', 'bleu_score', 'cpu_usage', 'memory_usage']
            
            for metric in metrics_to_check:
                if metric not in self.collected_data.columns:
                    continue
                
                # Get recent data
                recent_data = self.collected_data[metric].tail(50).values
                
                if len(recent_data) < 20:
                    continue
                
                # Check for drift
                metric_alerts = self.drift_detector.detect_drift(
                    metric_name=metric,
                    current_data=recent_data,
                    current_performance=recent_data[-1] if metric in ['accuracy_score', 'bleu_score'] else None
                )
                
                alerts.extend(metric_alerts)
            
        except Exception as e:
            self.logger.error(f"Error in drift detection: {e}")
        
        return alerts
    
    async def _run_resource_forecasting(self) -> Dict[str, ForecastResult]:
        """Run resource usage forecasting"""
        forecasts = {}
        
        try:
            # Resource types to forecast
            resource_types = ['cpu_usage', 'memory_usage', 'gpu_usage']
            
            for resource_type in resource_types:
                if resource_type not in self.collected_data.columns:
                    continue
                
                # Generate forecast
                forecast = self.resource_forecaster.forecast(
                    resource_type=resource_type,
                    periods=24  # 24-hour forecast
                )
                
                if forecast:
                    forecasts[resource_type] = forecast
            
        except Exception as e:
            self.logger.error(f"Error in resource forecasting: {e}")
        
        return forecasts
    
    async def _run_performance_prediction(self) -> Optional[PerformancePrediction]:
        """Run performance degradation prediction"""
        try:
            # Use recent data for prediction
            recent_data = self.collected_data.tail(48)  # Last 48 hours
            
            if len(recent_data) < 24:
                return None
            
            prediction = self.performance_predictor.predict_degradation(recent_data)
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error in performance prediction: {e}")
            return None
    
    async def _run_early_warnings(self) -> List[Any]:
        """Run early warning system"""
        warnings = []
        
        try:
            # Check for early warning conditions
            recent_data = self.collected_data.tail(24)  # Last 24 hours
            
            if len(recent_data) >= 4:
                warnings = self.performance_predictor.check_early_warnings(recent_data)
            
        except Exception as e:
            self.logger.error(f"Error in early warnings: {e}")
        
        return warnings
    
    async def _process_predictive_alerts(self, drift_alerts: List[DriftAlert],
                                       resource_forecasts: Dict[str, ForecastResult],
                                       performance_prediction: Optional[PerformancePrediction],
                                       early_warnings: List[Any]):
        """Process and send predictive alerts"""
        try:
            current_time = time.time()
            
            # Process drift alerts
            for drift_alert in drift_alerts:
                alert_key = f"drift_{drift_alert.metric_name}"
                
                # Check cooldown
                if self._is_alert_in_cooldown(alert_key):
                    continue
                
                # Create alert
                alert = self.alerting_system.create_alert(
                    title=f"Model Drift Detected: {drift_alert.metric_name}",
                    description=drift_alert.description,
                    source="drift_detection",
                    severity=self._map_drift_severity(drift_alert.severity),
                    metric_name=drift_alert.metric_name,
                    metric_value=drift_alert.drift_score,
                    threshold=drift_alert.threshold,
                    tags=["drift", "prediction"],
                    metadata={
                        "drift_type": drift_alert.drift_type.value,
                        "recommendations": drift_alert.recommendations
                    }
                )
                
                await self.alerting_system.send_alert(alert)
                self.alert_cooldowns[alert_key] = current_time
            
            # Process performance prediction alerts
            if performance_prediction and performance_prediction.risk_level.value in ['high', 'critical']:
                alert_key = "performance_degradation"
                
                if not self._is_alert_in_cooldown(alert_key):
                    alert = self.alerting_system.create_alert(
                        title="Performance Degradation Risk Detected",
                        description=f"System at {performance_prediction.risk_level.value} risk of performance degradation",
                        source="performance_prediction",
                        severity=AlertSeverity.CRITICAL if performance_prediction.risk_level.value == 'critical' else AlertSeverity.WARNING,
                        metric_name="degradation_probability",
                        metric_value=performance_prediction.degradation_probability,
                        threshold=0.5,
                        tags=["prediction", "performance"],
                        metadata={
                            "contributing_factors": performance_prediction.contributing_factors,
                            "recommended_actions": performance_prediction.recommended_actions,
                            "confidence": performance_prediction.model_confidence
                        }
                    )
                    
                    await self.alerting_system.send_alert(alert)
                    self.alert_cooldowns[alert_key] = current_time
            
            # Process resource forecast alerts
            for resource_type, forecast in resource_forecasts.items():
                # Check if forecast indicates resource exhaustion
                max_predicted = max(forecast.predicted_values)
                
                if max_predicted > 0.9:  # 90% utilization predicted
                    alert_key = f"resource_forecast_{resource_type}"
                    
                    if not self._is_alert_in_cooldown(alert_key):
                        alert = self.alerting_system.create_alert(
                            title=f"High {resource_type} Usage Predicted",
                            description=f"Forecast indicates {resource_type} will reach {max_predicted:.1%} utilization",
                            source="resource_forecasting",
                            severity=AlertSeverity.WARNING,
                            metric_name=resource_type,
                            metric_value=max_predicted,
                            threshold=0.9,
                            tags=["forecast", "resource"],
                            metadata={
                                "forecast_horizon": forecast.forecast_horizon,
                                "model_type": forecast.model_type.value
                            }
                        )
                        
                        await self.alerting_system.send_alert(alert)
                        self.alert_cooldowns[alert_key] = current_time
            
            # Process early warnings
            for warning in early_warnings:
                alert_key = f"early_warning_{warning.warning_type}"
                
                if not self._is_alert_in_cooldown(alert_key):
                    alert = self.alerting_system.create_alert(
                        title=f"Early Warning: {warning.warning_type}",
                        description=f"Early warning detected with {warning.probability:.1%} confidence",
                        source="early_warning",
                        severity=self._map_warning_severity(warning.risk_level),
                        metric_name=warning.warning_type,
                        metric_value=warning.probability,
                        threshold=0.3,
                        tags=["early_warning", "prediction"],
                        metadata={
                            "root_causes": warning.root_causes,
                            "preventive_actions": warning.preventive_actions,
                            "affected_metrics": warning.affected_metrics
                        }
                    )
                    
                    await self.alerting_system.send_alert(alert)
                    self.alert_cooldowns[alert_key] = current_time
            
        except Exception as e:
            self.logger.error(f"Error processing predictive alerts: {e}")
    
    def _is_alert_in_cooldown(self, alert_key: str) -> bool:
        """Check if alert is in cooldown period"""
        if alert_key not in self.alert_cooldowns:
            return False
        
        cooldown_seconds = self.config.alert_cooldown_minutes * 60
        return time.time() - self.alert_cooldowns[alert_key] < cooldown_seconds
    
    def _map_drift_severity(self, drift_severity) -> AlertSeverity:
        """Map drift severity to alert severity"""
        mapping = {
            'low': AlertSeverity.INFO,
            'medium': AlertSeverity.WARNING,
            'high': AlertSeverity.CRITICAL,
            'critical': AlertSeverity.EMERGENCY
        }
        return mapping.get(drift_severity.value, AlertSeverity.WARNING)
    
    def _map_warning_severity(self, risk_level) -> AlertSeverity:
        """Map warning risk level to alert severity"""
        mapping = {
            'low': AlertSeverity.INFO,
            'medium': AlertSeverity.WARNING,
            'high': AlertSeverity.CRITICAL,
            'critical': AlertSeverity.EMERGENCY
        }
        return mapping.get(risk_level.value, AlertSeverity.WARNING)
    
    def _calculate_health_score(self, drift_alerts: List[DriftAlert],
                              performance_prediction: Optional[PerformancePrediction],
                              early_warnings: List[Any]) -> float:
        """Calculate overall system health score (0-1)"""
        score = 1.0
        
        # Penalize for drift alerts
        for alert in drift_alerts:
            if alert.severity.value == 'critical':
                score -= 0.3
            elif alert.severity.value == 'high':
                score -= 0.2
            elif alert.severity.value == 'medium':
                score -= 0.1
        
        # Penalize for performance prediction risk
        if performance_prediction:
            if performance_prediction.risk_level.value == 'critical':
                score -= 0.4
            elif performance_prediction.risk_level.value == 'high':
                score -= 0.3
            elif performance_prediction.risk_level.value == 'medium':
                score -= 0.2
        
        # Penalize for early warnings
        high_warnings = [w for w in early_warnings if hasattr(w, 'risk_level') and w.risk_level.value in ['high', 'critical']]
        score -= len(high_warnings) * 0.1
        
        return max(0.0, score)
    
    def _generate_recommendations(self, drift_alerts: List[DriftAlert],
                                resource_forecasts: Dict[str, ForecastResult],
                                performance_prediction: Optional[PerformancePrediction],
                                early_warnings: List[Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = set()
        
        # Drift-based recommendations
        if drift_alerts:
            recommendations.add("Run model drift analysis and consider retraining")
            recommendations.add("Review recent data quality and preprocessing changes")
        
        # Resource-based recommendations
        for resource_type, forecast in resource_forecasts.items():
            max_usage = max(forecast.predicted_values)
            if max_usage > 0.8:
                recommendations.add(f"Plan to scale {resource_type} resources - forecast shows {max_usage:.1%} peak usage")
        
        # Performance-based recommendations
        if performance_prediction and performance_prediction.risk_level.value in ['high', 'critical']:
            recommendations.update(performance_prediction.recommended_actions[:3])
        
        # Early warning recommendations
        for warning in early_warnings:
            if hasattr(warning, 'preventive_actions'):
                recommendations.update(warning.preventive_actions[:2])
        
        return list(recommendations)[:10]  # Top 10 recommendations
    
    async def _retrain_models(self):
        """Retrain all models with current data"""
        try:
            if len(self.collected_data) < self.config.min_data_points * 2:
                self.logger.warning("Insufficient data for model retraining")
                return
            
            self.mode = AnalyticsMode.TRAINING
            self.logger.info("Starting model retraining")
            
            training_data = self.collected_data.copy()
            
            # Train resource forecaster
            if self.resource_forecaster:
                resource_types = ['cpu_usage', 'memory_usage', 'gpu_usage']
                available_types = [rt for rt in resource_types if rt in training_data.columns]
                
                if available_types:
                    results = self.resource_forecaster.train_models(training_data, available_types)
                    self.logger.info(f"Resource forecaster training results: {results}")
            
            # Train performance predictor
            if self.performance_predictor:
                success = self.performance_predictor.train_models(training_data)
                self.logger.info(f"Performance predictor training: {'success' if success else 'failed'}")
            
            # Establish drift baselines
            if self.drift_detector:
                metrics = ['latency_p95', 'accuracy_score', 'bleu_score', 'cpu_usage', 'memory_usage']
                for metric in metrics:
                    if metric in training_data.columns:
                        data = training_data[metric].values
                        if len(data) >= 100:
                            self.drift_detector.establish_baseline(metric, data)
            
            self.mode = AnalyticsMode.MONITORING
            self.logger.info("Model retraining completed")
            
        except Exception as e:
            self.logger.error(f"Error retraining models: {e}")
            self.mode = AnalyticsMode.MONITORING
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics framework summary"""
        with self._lock:
            summary = {
                'timestamp': time.time(),
                'mode': self.mode.value,
                'is_running': self.is_running,
                'data_points': len(self.collected_data),
                'last_analysis': self.last_analysis_time,
                'last_training': self.last_training_time,
                'components_enabled': {
                    'drift_detection': self.config.enable_drift_detection,
                    'resource_forecasting': self.config.enable_resource_forecasting,
                    'performance_prediction': self.config.enable_performance_prediction,
                    'early_warnings': self.config.enable_early_warnings,
                    'alerting': self.config.enable_predictive_alerts
                },
                'recent_analyses': len(self.analysis_history),
                'active_cooldowns': len([k for k, v in self.alert_cooldowns.items() 
                                       if time.time() - v < self.config.alert_cooldown_minutes * 60])
            }
            
            # Add latest analysis results
            if self.analysis_history:
                latest = self.analysis_history[-1]
                summary['latest_analysis'] = {
                    'timestamp': latest.timestamp,
                    'health_score': latest.system_health_score,
                    'drift_alerts': len(latest.drift_alerts),
                    'forecasts': len(latest.resource_forecasts),
                    'has_performance_prediction': latest.performance_prediction is not None,
                    'early_warnings': len(latest.early_warnings),
                    'recommendations': len(latest.recommendations)
                }
            
            return summary
    
    def force_analysis(self) -> Dict[str, Any]:
        """Force immediate analysis run"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create task for running analysis
                task = loop.create_task(self._run_analysis())
                return {"status": "analysis_scheduled", "task_id": id(task)}
            else:
                # Run synchronously
                loop.run_until_complete(self._run_analysis())
                return {"status": "analysis_completed"}
        except Exception as e:
            self.logger.error(f"Error forcing analysis: {e}")
            return {"status": "error", "message": str(e)}


# Factory functions
def create_predictive_analytics(config: Optional[PredictiveAnalyticsConfig] = None) -> PredictiveAnalyticsFramework:
    """Create predictive analytics framework with optional configuration"""
    return PredictiveAnalyticsFramework(config)


# Global framework instance
_global_analytics: Optional[PredictiveAnalyticsFramework] = None


def get_predictive_analytics() -> PredictiveAnalyticsFramework:
    """Get global predictive analytics framework instance"""
    global _global_analytics
    if _global_analytics is None:
        _global_analytics = create_predictive_analytics()
    return _global_analytics


def set_predictive_analytics(analytics: PredictiveAnalyticsFramework):
    """Set global predictive analytics framework instance"""
    global _global_analytics
    _global_analytics = analytics 