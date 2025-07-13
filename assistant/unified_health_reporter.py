"""
Unified Health Reporting System for Sovereign 4.0 Voice Assistant

Integrates existing health monitoring with predictive analytics to provide comprehensive
system health insights following 2024-2025 best practices for health aggregation.

Key Features:
- Unified health status combining real-time and predictive metrics
- Composite health scoring with weighted predictive insights
- Risk factor identification and recommended actions
- Graceful degradation when predictive services unavailable
"""

import asyncio
import logging
import time
import threading
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import json

from .health_monitoring import SystemHealthMonitor, HealthStatus, SystemHealthMetrics
from .predictive_analytics import PredictiveAnalyticsFramework
from .metrics_collector import MetricsCollector
from .config_manager import ConfigManager
from .structured_logging import VoiceAILogger, get_voice_ai_logger

class UnifiedHealthStatus(Enum):
    """Enhanced health status levels with predictive insights"""
    OPTIMAL = "optimal"          # All systems healthy, positive trends
    HEALTHY = "healthy"          # Normal operation, no issues detected
    DEGRADED = "degraded"        # Performance issues, some risk factors
    AT_RISK = "at_risk"         # Predictive warnings, intervention needed
    UNHEALTHY = "unhealthy"      # Active issues, immediate attention required
    CRITICAL = "critical"        # System failing, emergency response needed
    UNKNOWN = "unknown"          # Insufficient data or monitoring unavailable

@dataclass
class RiskFactor:
    """Individual risk factor identified by predictive analytics"""
    factor_type: str
    severity: str  # low, medium, high, critical
    description: str
    probability: float  # 0.0 to 1.0
    time_to_impact: Optional[float]  # seconds until predicted impact
    affected_components: List[str]
    confidence: float  # 0.0 to 1.0
    mitigation_actions: List[str]

@dataclass
class RecommendedAction:
    """Recommended action for system optimization or issue resolution"""
    action_type: str
    priority: str  # low, medium, high, urgent
    description: str
    expected_impact: str
    effort_level: str  # low, medium, high
    automation_possible: bool
    dependencies: List[str]
    estimated_duration: Optional[float]  # seconds
    rollback_plan: Optional[str] = None

@dataclass
class UnifiedHealthReport:
    """Comprehensive health report combining real-time and predictive insights"""
    timestamp: float
    overall_status: UnifiedHealthStatus
    health_score: float  # 0.0 to 1.0, weighted composite score
    
    # Current state
    current_metrics: Optional[SystemHealthMetrics]
    current_health_score: float
    
    # Predictive insights
    predictive_health_score: float
    predictions_available: bool
    drift_alerts: List[Dict[str, Any]]
    resource_forecasts: Dict[str, Any]
    performance_predictions: Dict[str, Any]
    
    # Risk assessment
    risk_factors: List[RiskFactor]
    overall_risk_level: str  # low, medium, high, critical
    
    # Recommendations
    recommended_actions: List[RecommendedAction]
    optimization_opportunities: List[str]
    
    # Trends and insights
    health_trend: str  # improving, stable, declining
    trend_confidence: float
    next_evaluation: float
    
    # Metadata
    evaluation_duration_ms: float
    components_evaluated: List[str]
    predictive_models_used: List[str]

class UnifiedHealthReporter:
    """
    Unified Health Reporter integrating existing health monitoring with predictive analytics
    
    Provides comprehensive system health insights by combining:
    - Real-time health metrics from SystemHealthMonitor
    - Predictive analytics from PredictiveAnalyticsFramework
    - Risk assessment and trend analysis
    - Automated recommendations and optimization opportunities
    """
    
    def __init__(
        self,
        health_monitor: Optional[SystemHealthMonitor] = None,
        predictive_analytics: Optional[PredictiveAnalyticsFramework] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        config_manager: Optional[ConfigManager] = None,
        logger: Optional[VoiceAILogger] = None
    ):
        self.health_monitor = health_monitor
        self.predictive_analytics = predictive_analytics
        self.metrics_collector = metrics_collector
        self.config_manager = config_manager
        self.logger = logger or get_voice_ai_logger("unified_health")
        
        # Configuration
        self.config = self._load_configuration()
        
        # Health scoring weights
        self.health_weights = {
            'current_health': self.config.get('current_health_weight', 0.6),
            'predictive_health': self.config.get('predictive_health_weight', 0.4),
            'trend_factor': self.config.get('trend_factor_weight', 0.1)
        }
        
        # Status mapping thresholds
        self.status_thresholds = {
            'optimal': 0.95,
            'healthy': 0.85,
            'degraded': 0.70,
            'at_risk': 0.50,
            'unhealthy': 0.30,
            'critical': 0.10
        }
        
        # Risk assessment configuration
        self.risk_thresholds = {
            'low': 0.25,
            'medium': 0.50,
            'high': 0.75,
            'critical': 0.90
        }
        
        # State tracking
        self._lock = threading.RLock()
        self._last_report: Optional[UnifiedHealthReport] = None
        self._health_history: List[float] = []
        self._max_history_size = 100
        
        # Performance tracking
        self._evaluation_times: List[float] = []
        self._max_eval_times = 50
        
        self.logger.info("🏥 UnifiedHealthReporter initialized with predictive integration")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load configuration for unified health reporting"""
        default_config = {
            'evaluation_interval': 30.0,  # seconds
            'current_health_weight': 0.6,
            'predictive_health_weight': 0.4,
            'trend_factor_weight': 0.1,
            'risk_assessment_enabled': True,
            'recommendations_enabled': True,
            'trend_analysis_window': 300.0,  # 5 minutes
            'min_predictions_for_weighting': 3
        }
        
        if self.config_manager:
            try:
                config = self.config_manager.get_config()
                health_config = getattr(config, 'unified_health', {})
                return {**default_config, **health_config}
            except Exception as e:
                self.logger.warning(f"Failed to load unified health config: {e}")
        
        return default_config
    
    async def get_unified_health_report(self) -> UnifiedHealthReport:
        """
        Generate comprehensive unified health report
        """
        start_time = time.time()
        
        try:
            # Collect current health metrics
            current_metrics = None
            current_health_score = 0.5  # default neutral score
            
            if self.health_monitor:
                try:
                    current_metrics = await self.health_monitor.get_system_health()
                    current_health_score = self._calculate_current_health_score(current_metrics)
                except Exception as e:
                    self.logger.warning(f"Failed to get current health metrics: {e}")
            
            # Collect predictive insights
            predictive_health_score = 0.5  # default neutral score
            predictions_available = False
            drift_alerts = []
            resource_forecasts = {}
            performance_predictions = {}
            
            if self.predictive_analytics:
                try:
                    predictive_data = await self._collect_predictive_insights()
                    predictive_health_score = predictive_data['health_score']
                    predictions_available = predictive_data['available']
                    drift_alerts = predictive_data['drift_alerts']
                    resource_forecasts = predictive_data['resource_forecasts']
                    performance_predictions = predictive_data['performance_predictions']
                except Exception as e:
                    self.logger.warning(f"Failed to get predictive insights: {e}")
            
            # Calculate composite health score
            health_score = self._calculate_composite_health_score(
                current_health_score, 
                predictive_health_score,
                predictions_available
            )
            
            # Determine overall status
            overall_status = self._determine_overall_status(health_score, drift_alerts)
            
            # Assess risks
            risk_factors = await self._assess_risk_factors(
                current_metrics, drift_alerts, resource_forecasts, performance_predictions
            )
            overall_risk_level = self._calculate_overall_risk_level(risk_factors)
            
            # Generate recommendations
            recommended_actions = await self._generate_recommendations(
                current_metrics, risk_factors, overall_status
            )
            optimization_opportunities = self._identify_optimization_opportunities(
                current_metrics, performance_predictions
            )
            
            # Analyze trends
            health_trend, trend_confidence = self._analyze_health_trend(health_score)
            
            # Calculate next evaluation time
            next_evaluation = time.time() + self.config['evaluation_interval']
            
            # Track evaluation performance
            evaluation_duration_ms = (time.time() - start_time) * 1000
            self._track_evaluation_performance(evaluation_duration_ms)
            
            # Create comprehensive report
            report = UnifiedHealthReport(
                timestamp=time.time(),
                overall_status=overall_status,
                health_score=health_score,
                current_metrics=current_metrics,
                current_health_score=current_health_score,
                predictive_health_score=predictive_health_score,
                predictions_available=predictions_available,
                drift_alerts=drift_alerts,
                resource_forecasts=resource_forecasts,
                performance_predictions=performance_predictions,
                risk_factors=risk_factors,
                overall_risk_level=overall_risk_level,
                recommended_actions=recommended_actions,
                optimization_opportunities=optimization_opportunities,
                health_trend=health_trend,
                trend_confidence=trend_confidence,
                next_evaluation=next_evaluation,
                evaluation_duration_ms=evaluation_duration_ms,
                components_evaluated=self._get_evaluated_components(),
                predictive_models_used=self._get_predictive_models_used()
            )
            
            # Store report for trend analysis
            with self._lock:
                self._last_report = report
                self._health_history.append(health_score)
                if len(self._health_history) > self._max_history_size:
                    self._health_history.pop(0)
            
            self.logger.info(
                f"🏥 Unified health evaluation completed: {overall_status.value} "
                f"(score: {health_score:.3f}, {evaluation_duration_ms:.1f}ms)"
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate unified health report: {e}")
            # Return minimal report with error status
            return UnifiedHealthReport(
                timestamp=time.time(),
                overall_status=UnifiedHealthStatus.UNKNOWN,
                health_score=0.0,
                current_metrics=None,
                current_health_score=0.0,
                predictive_health_score=0.0,
                predictions_available=False,
                drift_alerts=[],
                resource_forecasts={},
                performance_predictions={},
                risk_factors=[],
                overall_risk_level="unknown",
                recommended_actions=[],
                optimization_opportunities=[],
                health_trend="unknown",
                trend_confidence=0.0,
                next_evaluation=time.time() + 60.0,
                evaluation_duration_ms=(time.time() - start_time) * 1000,
                components_evaluated=[],
                predictive_models_used=[]
            )
    
    async def _collect_predictive_insights(self) -> Dict[str, Any]:
        """Collect insights from predictive analytics framework"""
        try:
            # Get overall health score from predictive analytics
            health_score = await self.predictive_analytics.get_health_score()
            
            # Get drift alerts
            drift_alerts = []
            if hasattr(self.predictive_analytics, 'drift_detector'):
                alerts = await self.predictive_analytics.drift_detector.get_active_alerts()
                drift_alerts = []
                for alert in alerts:
                    alert_dict = asdict(alert)
                    # Convert enum values to strings
                    if 'drift_type' in alert_dict and hasattr(alert_dict['drift_type'], 'value'):
                        alert_dict['drift_type'] = alert_dict['drift_type'].value
                    if 'severity' in alert_dict and hasattr(alert_dict['severity'], 'value'):
                        alert_dict['severity'] = alert_dict['severity'].value
                    drift_alerts.append(alert_dict)
            
            # Get resource forecasts
            resource_forecasts = {}
            if hasattr(self.predictive_analytics, 'resource_forecaster'):
                forecasts = await self.predictive_analytics.resource_forecaster.get_forecasts()
                resource_forecasts = forecasts
            
            # Get performance predictions
            performance_predictions = {}
            if hasattr(self.predictive_analytics, 'performance_predictor'):
                predictions = await self.predictive_analytics.performance_predictor.get_predictions()
                performance_predictions = predictions
            
            return {
                'health_score': health_score,
                'available': True,
                'drift_alerts': drift_alerts,
                'resource_forecasts': resource_forecasts,
                'performance_predictions': performance_predictions
            }
            
        except Exception as e:
            self.logger.warning(f"Error collecting predictive insights: {e}")
            return {
                'health_score': 0.5,
                'available': False,
                'drift_alerts': [],
                'resource_forecasts': {},
                'performance_predictions': {}
            }
    
    def _calculate_current_health_score(self, metrics: SystemHealthMetrics) -> float:
        """Calculate normalized health score from current metrics"""
        if not metrics:
            return 0.5
        
        try:
            # Weight different health factors
            factors = {
                'error_rate': 1.0 - min(metrics.error_rate, 1.0),  # Lower error rate = better
                'response_time': max(0.0, 1.0 - (metrics.avg_response_time_ms / 1000.0)),  # <1s target
                'memory_usage': max(0.0, 1.0 - (metrics.memory_usage_mb / 1000.0)),  # <1GB target
                'cpu_usage': max(0.0, 1.0 - (metrics.cpu_usage_percent / 100.0)),  # <100% target
                'connectivity': 1.0 if metrics.network_connectivity else 0.0
            }
            
            # Calculate weighted average
            weights = {'error_rate': 0.3, 'response_time': 0.3, 'memory_usage': 0.2, 'cpu_usage': 0.15, 'connectivity': 0.05}
            score = sum(factors[key] * weights[key] for key in factors.keys())
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.warning(f"Error calculating current health score: {e}")
            return 0.5
    
    def _calculate_composite_health_score(
        self, 
        current_score: float, 
        predictive_score: float, 
        predictions_available: bool
    ) -> float:
        """Calculate weighted composite health score"""
        if not predictions_available or len(self._health_history) < self.config['min_predictions_for_weighting']:
            # If predictions not available or insufficient history, use mainly current health
            return current_score * 0.9 + 0.5 * 0.1
        
        # Calculate trend factor
        trend_factor = self._calculate_trend_factor()
        
        # Weighted combination
        base_score = (
            current_score * self.health_weights['current_health'] +
            predictive_score * self.health_weights['predictive_health']
        )
        
        # Apply trend factor
        composite_score = base_score + (trend_factor * self.health_weights['trend_factor'])
        
        return max(0.0, min(1.0, composite_score))
    
    def _calculate_trend_factor(self) -> float:
        """Calculate trend factor from health history"""
        if len(self._health_history) < 3:
            return 0.0
        
        try:
            recent_scores = self._health_history[-5:]  # Last 5 scores
            
            # Simple linear trend calculation
            x = list(range(len(recent_scores)))
            y = recent_scores
            
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            # Calculate slope (trend)
            if n * sum_x2 - sum_x ** 2 != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
                return max(-0.1, min(0.1, slope))  # Limit trend impact
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _determine_overall_status(self, health_score: float, drift_alerts: List[Dict]) -> UnifiedHealthStatus:
        """Determine overall status based on health score and alerts"""
        # Check for critical drift alerts
        critical_alerts = [alert for alert in drift_alerts if alert.get('severity') == 'critical']
        if critical_alerts:
            return UnifiedHealthStatus.CRITICAL
        
        # Check for high-severity alerts
        high_alerts = [alert for alert in drift_alerts if alert.get('severity') == 'high']
        if high_alerts and health_score < 0.7:
            return UnifiedHealthStatus.AT_RISK
        
        # Use score-based thresholds
        if health_score >= self.status_thresholds['optimal']:
            return UnifiedHealthStatus.OPTIMAL
        elif health_score >= self.status_thresholds['healthy']:
            return UnifiedHealthStatus.HEALTHY
        elif health_score >= self.status_thresholds['degraded']:
            return UnifiedHealthStatus.DEGRADED
        elif health_score >= self.status_thresholds['at_risk']:
            return UnifiedHealthStatus.AT_RISK
        elif health_score >= self.status_thresholds['unhealthy']:
            return UnifiedHealthStatus.UNHEALTHY
        elif health_score >= self.status_thresholds['critical']:
            return UnifiedHealthStatus.CRITICAL
        else:
            return UnifiedHealthStatus.CRITICAL
    
    async def _assess_risk_factors(
        self, 
        current_metrics: Optional[SystemHealthMetrics],
        drift_alerts: List[Dict],
        resource_forecasts: Dict[str, Any],
        performance_predictions: Dict[str, Any]
    ) -> List[RiskFactor]:
        """Assess risk factors from all available data"""
        risk_factors = []
        
        try:
            # Risk factors from drift alerts
            for alert in drift_alerts:
                risk_factors.append(RiskFactor(
                    factor_type="drift_detection",
                    severity=alert.get('severity', 'medium'),
                    description=f"Drift detected in {alert.get('metric_name', 'unknown metric')}",
                    probability=alert.get('confidence', 0.5),
                    time_to_impact=None,
                    affected_components=[alert.get('component', 'unknown')],
                    confidence=alert.get('confidence', 0.5),
                    mitigation_actions=alert.get('recommendations', [])
                ))
            
            # Risk factors from resource forecasts
            if resource_forecasts:
                for resource, forecast in resource_forecasts.items():
                    if isinstance(forecast, dict) and forecast.get('risk_level', 'low') in ['high', 'critical']:
                        risk_factors.append(RiskFactor(
                            factor_type="resource_forecast",
                            severity=forecast.get('risk_level', 'medium'),
                            description=f"Predicted {resource} constraint",
                            probability=forecast.get('confidence', 0.5),
                            time_to_impact=forecast.get('time_to_threshold', None),
                            affected_components=[resource],
                            confidence=forecast.get('confidence', 0.5),
                            mitigation_actions=forecast.get('recommendations', [])
                        ))
            
            # Risk factors from performance predictions
            if performance_predictions:
                degradation_risk = performance_predictions.get('degradation_probability', 0.0)
                if degradation_risk > 0.5:
                    severity = 'critical' if degradation_risk > 0.8 else 'high' if degradation_risk > 0.6 else 'medium'
                    risk_factors.append(RiskFactor(
                        factor_type="performance_degradation",
                        severity=severity,
                        description="Predicted performance degradation",
                        probability=degradation_risk,
                        time_to_impact=performance_predictions.get('time_to_degradation', None),
                        affected_components=performance_predictions.get('affected_components', ['system']),
                        confidence=performance_predictions.get('confidence', 0.5),
                        mitigation_actions=performance_predictions.get('mitigation_strategies', [])
                    ))
            
            # Risk factors from current metrics
            if current_metrics:
                if current_metrics.error_rate > 0.1:  # >10% error rate
                    risk_factors.append(RiskFactor(
                        factor_type="high_error_rate",
                        severity="high" if current_metrics.error_rate > 0.2 else "medium",
                        description=f"High error rate: {current_metrics.error_rate:.1%}",
                        probability=1.0,  # Current issue
                        time_to_impact=0,
                        affected_components=["error_handling"],
                        confidence=1.0,
                        mitigation_actions=["Review error logs", "Check service dependencies"]
                    ))
                
                if current_metrics.avg_response_time_ms > 1000:  # >1s response time
                    risk_factors.append(RiskFactor(
                        factor_type="high_latency",
                        severity="high" if current_metrics.avg_response_time_ms > 2000 else "medium",
                        description=f"High response time: {current_metrics.avg_response_time_ms:.0f}ms",
                        probability=1.0,
                        time_to_impact=0,
                        affected_components=["performance"],
                        confidence=1.0,
                        mitigation_actions=["Optimize queries", "Scale resources"]
                    ))
            
        except Exception as e:
            self.logger.warning(f"Error assessing risk factors: {e}")
        
        return risk_factors
    
    def _calculate_overall_risk_level(self, risk_factors: List[RiskFactor]) -> str:
        """Calculate overall risk level from individual risk factors"""
        if not risk_factors:
            return "low"
        
        # Count risk factors by severity
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        total_risk_score = 0.0
        
        for factor in risk_factors:
            severity_counts[factor.severity] += 1
            # Weight by severity and probability
            severity_weights = {"low": 0.1, "medium": 0.3, "high": 0.7, "critical": 1.0}
            total_risk_score += severity_weights[factor.severity] * factor.probability
        
        # Determine overall risk level
        if severity_counts["critical"] > 0 or total_risk_score > self.risk_thresholds['critical']:
            return "critical"
        elif severity_counts["high"] > 1 or total_risk_score > self.risk_thresholds['high']:
            return "high"
        elif severity_counts["high"] > 0 or severity_counts["medium"] > 2 or total_risk_score > self.risk_thresholds['medium']:
            return "medium"
        else:
            return "low"
    
    async def _generate_recommendations(
        self, 
        current_metrics: Optional[SystemHealthMetrics],
        risk_factors: List[RiskFactor],
        overall_status: UnifiedHealthStatus
    ) -> List[RecommendedAction]:
        """Generate recommendations based on current state and risk factors"""
        recommendations = []
        
        try:
            # Recommendations based on overall status
            if overall_status in [UnifiedHealthStatus.CRITICAL, UnifiedHealthStatus.UNHEALTHY]:
                recommendations.append(RecommendedAction(
                    action_type="emergency_response",
                    priority="urgent",
                    description="Initiate emergency response protocol",
                    expected_impact="Restore basic system functionality",
                    effort_level="high",
                    automation_possible=True,
                    dependencies=["on_call_team"],
                    estimated_duration=300  # 5 minutes
                ))
            
            elif overall_status == UnifiedHealthStatus.AT_RISK:
                recommendations.append(RecommendedAction(
                    action_type="preventive_scaling",
                    priority="high",
                    description="Scale resources preemptively",
                    expected_impact="Prevent predicted performance degradation",
                    effort_level="medium",
                    automation_possible=True,
                    dependencies=["resource_manager"],
                    estimated_duration=60  # 1 minute
                ))
            
            # Recommendations from risk factors
            for factor in risk_factors:
                if factor.severity in ["high", "critical"]:
                    for action in factor.mitigation_actions:
                        recommendations.append(RecommendedAction(
                            action_type="risk_mitigation",
                            priority="high" if factor.severity == "critical" else "medium",
                            description=action,
                            expected_impact=f"Mitigate {factor.factor_type} risk",
                            effort_level="medium",
                            automation_possible=False,
                            dependencies=[],
                            estimated_duration=None
                        ))
            
            # Performance optimization recommendations
            if current_metrics:
                if current_metrics.avg_response_time_ms > 800:  # Above target
                    recommendations.append(RecommendedAction(
                        action_type="performance_optimization",
                        priority="medium",
                        description="Optimize response time",
                        expected_impact="Reduce latency to target <800ms",
                        effort_level="medium",
                        automation_possible=True,
                        dependencies=["load_balancer"],
                        estimated_duration=120,
                        rollback_plan="Revert to previous configuration"
                    ))
                
                if current_metrics.memory_usage_mb > 800:  # High memory usage
                    recommendations.append(RecommendedAction(
                        action_type="memory_optimization",
                        priority="medium",
                        description="Optimize memory usage",
                        expected_impact="Reduce memory consumption",
                        effort_level="low",
                        automation_possible=True,
                        dependencies=["garbage_collector"],
                        estimated_duration=30
                    ))
        
        except Exception as e:
            self.logger.warning(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def _identify_optimization_opportunities(
        self, 
        current_metrics: Optional[SystemHealthMetrics],
        performance_predictions: Dict[str, Any]
    ) -> List[str]:
        """Identify optimization opportunities"""
        opportunities = []
        
        try:
            # From current metrics
            if current_metrics:
                if current_metrics.cpu_usage_percent < 30:
                    opportunities.append("CPU resources underutilized - consider scaling down")
                
                if current_metrics.error_rate < 0.01 and current_metrics.avg_response_time_ms < 500:
                    opportunities.append("System performing well - consider increasing capacity")
            
            # From performance predictions
            if performance_predictions:
                if performance_predictions.get('efficiency_score', 0.5) < 0.7:
                    opportunities.append("Model efficiency could be improved")
                
                cache_hit_rate = performance_predictions.get('cache_hit_rate', 0.5)
                if cache_hit_rate < 0.8:
                    opportunities.append("Cache hit rate could be improved")
        
        except Exception as e:
            self.logger.warning(f"Error identifying optimization opportunities: {e}")
        
        return opportunities
    
    def _analyze_health_trend(self, current_score: float) -> tuple[str, float]:
        """Analyze health trend and confidence"""
        if len(self._health_history) < 3:
            return "unknown", 0.0
        
        try:
            recent_scores = self._health_history[-5:]
            trend_slope = self._calculate_trend_factor()
            
            # Determine trend direction
            if abs(trend_slope) < 0.01:
                trend = "stable"
                confidence = 0.8
            elif trend_slope > 0.02:
                trend = "improving"
                confidence = min(0.9, abs(trend_slope) * 10)
            elif trend_slope < -0.02:
                trend = "declining"
                confidence = min(0.9, abs(trend_slope) * 10)
            else:
                trend = "stable"
                confidence = 0.6
            
            return trend, confidence
            
        except Exception:
            return "unknown", 0.0
    
    def _get_evaluated_components(self) -> List[str]:
        """Get list of components that were evaluated"""
        components = ["health_monitor"]
        
        if self.predictive_analytics:
            components.extend(["drift_detector", "resource_forecaster", "performance_predictor", "alerting_system"])
        
        if self.metrics_collector:
            components.append("metrics_collector")
        
        return components
    
    def _get_predictive_models_used(self) -> List[str]:
        """Get list of predictive models that were used"""
        models = []
        
        if self.predictive_analytics:
            if hasattr(self.predictive_analytics, 'drift_detector'):
                models.append("drift_detection")
            if hasattr(self.predictive_analytics, 'resource_forecaster'):
                models.extend(["prophet_forecaster", "lstm_forecaster"])
            if hasattr(self.predictive_analytics, 'performance_predictor'):
                models.extend(["random_forest", "gradient_boosting", "linear_regression"])
        
        return models
    
    def _track_evaluation_performance(self, duration_ms: float):
        """Track evaluation performance for optimization"""
        with self._lock:
            self._evaluation_times.append(duration_ms)
            if len(self._evaluation_times) > self._max_eval_times:
                self._evaluation_times.pop(0)
    
    def get_last_report(self) -> Optional[UnifiedHealthReport]:
        """Get the last generated health report"""
        with self._lock:
            return self._last_report
    
    def get_average_evaluation_time(self) -> float:
        """Get average evaluation time in milliseconds"""
        with self._lock:
            if not self._evaluation_times:
                return 0.0
            return sum(self._evaluation_times) / len(self._evaluation_times)
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get a quick health summary without full evaluation"""
        try:
            if self._last_report and (time.time() - self._last_report.timestamp) < 60:
                # Use cached report if recent
                report = self._last_report
            else:
                # Generate new report
                report = await self.get_unified_health_report()
            
            return {
                'status': report.overall_status.value,
                'score': report.health_score,
                'risk_level': report.overall_risk_level,
                'trend': report.health_trend,
                'critical_issues': len([r for r in report.risk_factors if r.severity == 'critical']),
                'urgent_actions': len([a for a in report.recommended_actions if a.priority == 'urgent']),
                'last_updated': report.timestamp
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get health summary: {e}")
            return {
                'status': 'unknown',
                'score': 0.0,
                'risk_level': 'unknown',
                'trend': 'unknown',
                'critical_issues': 0,
                'urgent_actions': 0,
                'last_updated': time.time()
            }

# Factory function for easy instantiation
def create_unified_health_reporter(
    health_monitor: Optional[SystemHealthMonitor] = None,
    predictive_analytics: Optional[PredictiveAnalyticsFramework] = None,
    metrics_collector: Optional[MetricsCollector] = None,
    config_manager: Optional[ConfigManager] = None,
    logger: Optional[VoiceAILogger] = None
) -> UnifiedHealthReporter:
    """Create and configure a UnifiedHealthReporter instance"""
    return UnifiedHealthReporter(
        health_monitor=health_monitor,
        predictive_analytics=predictive_analytics,
        metrics_collector=metrics_collector,
        config_manager=config_manager,
        logger=logger
    )

# Global instance management
_unified_health_reporter: Optional[UnifiedHealthReporter] = None

def get_unified_health_reporter() -> Optional[UnifiedHealthReporter]:
    """Get the global unified health reporter instance"""
    return _unified_health_reporter

def set_unified_health_reporter(reporter: UnifiedHealthReporter):
    """Set the global unified health reporter instance"""
    global _unified_health_reporter
    _unified_health_reporter = reporter 