"""
Performance Regression Detection for Sovereign 4.0 Voice Assistant

This module provides ML-based performance regression detection including:
- Statistical and ML-based regression detection
- Baseline comparison and drift detection
- Performance trend analysis
- Automated alerting for performance degradation

Implements modern 2024-2025 regression detection methodologies.
"""

import asyncio
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle

# ML libraries for regression detection
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    import joblib
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logging.warning("scikit-learn not available. ML-based regression detection disabled.")

from .performance_testing import TestResult, PerformanceTestConfig

logger = logging.getLogger(__name__)

@dataclass
class PerformanceBaseline:
    """Performance baseline for regression comparison"""
    baseline_id: str
    creation_date: datetime
    metrics: Dict[str, float]
    thresholds: Dict[str, float]
    sample_count: int
    confidence_interval: Dict[str, Tuple[float, float]]

@dataclass
class RegressionAlert:
    """Performance regression alert"""
    alert_id: str
    detection_time: datetime
    affected_metrics: List[str]
    severity: str  # 'low', 'medium', 'high', 'critical'
    regression_magnitude: float  # Percentage degradation
    confidence_score: float
    root_cause_hints: List[str]

class RegressionDetector:
    """
    Advanced performance regression detector
    
    Uses statistical analysis and ML algorithms to detect performance
    degradation compared to established baselines.
    """
    
    def __init__(self, config: PerformanceTestConfig):
        self.config = config
        
        # Baseline storage
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.performance_history: List[Dict[str, float]] = []
        
        # ML models for regression detection
        self.anomaly_detector: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        
        # Configuration
        self.regression_threshold = config.regression_config.get('z_score_threshold', 2.5)
        self.minimum_samples = config.regression_config.get('minimum_samples', 50)
        self.contamination_rate = config.regression_config.get('anomaly_contamination', 0.1)
        
        # Alert tracking
        self.detected_regressions: List[RegressionAlert] = []
        
        # Storage paths
        self.baseline_storage_path = Path('.performance_baselines')
        self.baseline_storage_path.mkdir(exist_ok=True)
        
        logger.info("Regression Detector initialized")
        
        # Load existing baselines
        self._load_baselines()
    
    async def detect_performance_regressions(self) -> List[TestResult]:
        """Detect performance regressions using multiple approaches"""
        results = []
        
        # Statistical regression detection
        results.append(await self.statistical_regression_detection())
        
        # ML-based anomaly detection
        if HAS_SKLEARN:
            results.append(await self.ml_based_regression_detection())
        
        # Trend analysis
        results.append(await self.trend_based_regression_detection())
        
        # Baseline comparison
        results.append(await self.baseline_comparison_analysis())
        
        return results
    
    async def check_for_regression(self) -> TestResult:
        """Quick regression check for periodic monitoring"""
        start_time = time.time()
        test_name = "quick_regression_check"
        
        try:
            if len(self.performance_history) < 10:
                return TestResult(
                    test_name=test_name,
                    status="skipped",
                    metrics={},
                    execution_time=time.time() - start_time,
                    timestamp=datetime.now(),
                    details="Insufficient performance history"
                )
            
            # Get recent performance data
            recent_data = self.performance_history[-10:]
            current_metrics = recent_data[-1] if recent_data else {}
            
            # Compare against baseline
            regression_detected = False
            affected_metrics = []
            
            for baseline_name, baseline in self.baselines.items():
                for metric_name, baseline_value in baseline.metrics.items():
                    if metric_name in current_metrics:
                        current_value = current_metrics[metric_name]
                        
                        # Calculate percentage change
                        if baseline_value > 0:
                            change_percent = ((current_value - baseline_value) / baseline_value) * 100
                            
                            # Check for regression (performance degradation)
                            if self._is_regression(metric_name, change_percent):
                                regression_detected = True
                                affected_metrics.append(f"{metric_name}: {change_percent:+.1f}%")
            
            metrics = {
                'regression_detected': regression_detected,
                'affected_metrics_count': len(affected_metrics),
                'baselines_compared': len(self.baselines)
            }
            
            status = 'passed' if not regression_detected else 'warning'
            
            return TestResult(
                test_name=test_name,
                status=status,
                metrics=metrics,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                details=f"Regression: {regression_detected}, Affected: {affected_metrics[:3]}"
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
    
    async def statistical_regression_detection(self) -> TestResult:
        """Detect regressions using statistical analysis"""
        start_time = time.time()
        test_name = "statistical_regression_detection"
        
        try:
            if len(self.performance_history) < self.minimum_samples:
                return TestResult(
                    test_name=test_name,
                    status="skipped",
                    metrics={},
                    execution_time=time.time() - start_time,
                    timestamp=datetime.now(),
                    details=f"Need {self.minimum_samples} samples, have {len(self.performance_history)}"
                )
            
            # Convert history to DataFrame for analysis
            df = pd.DataFrame(self.performance_history)
            
            regression_metrics = {}
            detected_regressions = []
            
            for column in df.columns:
                if df[column].dtype in ['float64', 'int64']:
                    values = df[column].dropna()
                    
                    if len(values) < 10:
                        continue
                    
                    # Statistical analysis
                    mean_value = values.mean()
                    std_value = values.std()
                    recent_values = values.tail(10)
                    recent_mean = recent_values.mean()
                    
                    # Z-score based detection
                    if std_value > 0:
                        z_score = abs((recent_mean - mean_value) / std_value)
                        regression_metrics[f"{column}_z_score"] = z_score
                        
                        if z_score > self.regression_threshold:
                            # Determine if this is a regression (performance degradation)
                            if self._is_performance_degradation(column, recent_mean, mean_value):
                                detected_regressions.append({
                                    'metric': column,
                                    'z_score': z_score,
                                    'baseline_mean': mean_value,
                                    'recent_mean': recent_mean,
                                    'degradation_percent': ((recent_mean - mean_value) / mean_value) * 100
                                })
                    
                    # Trend analysis
                    time_indices = range(len(values))
                    trend_slope = np.polyfit(time_indices, values, 1)[0]
                    regression_metrics[f"{column}_trend_slope"] = trend_slope
                    
                    # Check for concerning trends
                    if self._is_concerning_trend(column, trend_slope):
                        detected_regressions.append({
                            'metric': column,
                            'trend_slope': trend_slope,
                            'trend_type': 'degrading'
                        })
            
            # Overall assessment
            regression_metrics.update({
                'total_regressions_detected': len(detected_regressions),
                'metrics_analyzed': len([col for col in df.columns if df[col].dtype in ['float64', 'int64']]),
                'sample_size': len(df)
            })
            
            # Create regression alerts
            for regression in detected_regressions:
                alert = RegressionAlert(
                    alert_id=f"stat_regr_{int(time.time())}_{regression['metric']}",
                    detection_time=datetime.now(),
                    affected_metrics=[regression['metric']],
                    severity=self._determine_regression_severity(regression),
                    regression_magnitude=regression.get('degradation_percent', 0),
                    confidence_score=min(1.0, regression.get('z_score', 0) / self.regression_threshold),
                    root_cause_hints=self._generate_root_cause_hints(regression)
                )
                self.detected_regressions.append(alert)
            
            # Determine status
            status = 'passed'
            if len(detected_regressions) > 0:
                critical_regressions = [r for r in detected_regressions if 
                                     r.get('degradation_percent', 0) > 50 or r.get('z_score', 0) > 5]
                if critical_regressions:
                    status = 'failed'
                else:
                    status = 'warning'
            
            return TestResult(
                test_name=test_name,
                status=status,
                metrics=regression_metrics,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                details=f"Regressions: {len(detected_regressions)}, Metrics analyzed: {regression_metrics['metrics_analyzed']}"
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
    
    async def ml_based_regression_detection(self) -> TestResult:
        """Detect regressions using ML-based anomaly detection"""
        start_time = time.time()
        test_name = "ml_regression_detection"
        
        try:
            if not HAS_SKLEARN:
                return TestResult(
                    test_name=test_name,
                    status="skipped",
                    metrics={},
                    execution_time=time.time() - start_time,
                    timestamp=datetime.now(),
                    details="scikit-learn not available"
                )
            
            if len(self.performance_history) < self.minimum_samples:
                return TestResult(
                    test_name=test_name,
                    status="skipped",
                    metrics={},
                    execution_time=time.time() - start_time,
                    timestamp=datetime.now(),
                    details=f"Need {self.minimum_samples} samples for ML detection"
                )
            
            # Prepare data for ML analysis
            df = pd.DataFrame(self.performance_history)
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) == 0:
                return TestResult(
                    test_name=test_name,
                    status="skipped",
                    metrics={},
                    execution_time=time.time() - start_time,
                    timestamp=datetime.now(),
                    details="No numeric metrics for ML analysis"
                )
            
            # Prepare feature matrix
            X = df[numeric_columns].fillna(0).values
            
            # Train or update anomaly detection model
            if self.anomaly_detector is None:
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
                
                self.anomaly_detector = IsolationForest(
                    contamination=self.contamination_rate,
                    random_state=42
                )
                self.anomaly_detector.fit(X_scaled)
            else:
                X_scaled = self.scaler.transform(X)
            
            # Detect anomalies in recent data
            recent_data = X_scaled[-10:]  # Last 10 samples
            anomaly_scores = self.anomaly_detector.decision_function(recent_data)
            anomaly_predictions = self.anomaly_detector.predict(recent_data)
            
            # Identify anomalous samples
            anomalies = []
            for i, (score, prediction) in enumerate(zip(anomaly_scores, anomaly_predictions)):
                if prediction == -1:  # Anomaly detected
                    sample_idx = len(X_scaled) - 10 + i
                    anomalies.append({
                        'sample_index': sample_idx,
                        'anomaly_score': score,
                        'timestamp': datetime.now() - timedelta(minutes=(10-i)*5)  # Approximate timing
                    })
            
            ml_metrics = {
                'anomalies_detected': len(anomalies),
                'samples_analyzed': len(recent_data),
                'min_anomaly_score': min(anomaly_scores) if anomaly_scores.size > 0 else 0,
                'mean_anomaly_score': np.mean(anomaly_scores) if anomaly_scores.size > 0 else 0,
                'model_contamination_rate': self.contamination_rate
            }
            
            # Create ML-based regression alerts
            for anomaly in anomalies:
                alert = RegressionAlert(
                    alert_id=f"ml_regr_{int(time.time())}_{anomaly['sample_index']}",
                    detection_time=anomaly['timestamp'],
                    affected_metrics=['performance_anomaly'],
                    severity=self._determine_ml_severity(anomaly['anomaly_score']),
                    regression_magnitude=abs(anomaly['anomaly_score']) * 100,
                    confidence_score=min(1.0, abs(anomaly['anomaly_score'])),
                    root_cause_hints=['ML anomaly detection', 'Unusual performance pattern']
                )
                self.detected_regressions.append(alert)
            
            # Determine status
            status = 'passed'
            if len(anomalies) > 0:
                severe_anomalies = [a for a in anomalies if a['anomaly_score'] < -0.5]
                if severe_anomalies:
                    status = 'failed'
                else:
                    status = 'warning'
            
            return TestResult(
                test_name=test_name,
                status=status,
                metrics=ml_metrics,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                details=f"ML anomalies: {len(anomalies)}, Min score: {ml_metrics['min_anomaly_score']:.3f}"
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
    
    async def trend_based_regression_detection(self) -> TestResult:
        """Detect regressions based on performance trends"""
        start_time = time.time()
        test_name = "trend_regression_detection"
        
        try:
            if len(self.performance_history) < 20:  # Need at least 20 samples for trend analysis
                return TestResult(
                    test_name=test_name,
                    status="skipped",
                    metrics={},
                    execution_time=time.time() - start_time,
                    timestamp=datetime.now(),
                    details="Insufficient data for trend analysis"
                )
            
            df = pd.DataFrame(self.performance_history)
            trend_metrics = {}
            trend_regressions = []
            
            for column in df.columns:
                if df[column].dtype in ['float64', 'int64']:
                    values = df[column].dropna()
                    
                    if len(values) < 10:
                        continue
                    
                    # Short-term trend (last 25% of samples)
                    short_term_size = max(5, len(values) // 4)
                    short_term_values = values.tail(short_term_size)
                    short_term_indices = range(len(short_term_values))
                    short_trend_slope = np.polyfit(short_term_indices, short_term_values, 1)[0]
                    
                    # Long-term trend (all samples)
                    long_term_indices = range(len(values))
                    long_trend_slope = np.polyfit(long_term_indices, values, 1)[0]
                    
                    # Calculate trend acceleration
                    trend_acceleration = short_trend_slope - long_trend_slope
                    
                    trend_metrics.update({
                        f"{column}_short_trend": short_trend_slope,
                        f"{column}_long_trend": long_trend_slope,
                        f"{column}_trend_acceleration": trend_acceleration
                    })
                    
                    # Detect concerning trend changes
                    if self._is_trend_regression(column, short_trend_slope, long_trend_slope, trend_acceleration):
                        trend_regressions.append({
                            'metric': column,
                            'short_trend': short_trend_slope,
                            'long_trend': long_trend_slope,
                            'acceleration': trend_acceleration,
                            'trend_change_magnitude': abs(trend_acceleration)
                        })
            
            # Overall trend assessment
            trend_metrics.update({
                'trend_regressions_detected': len(trend_regressions),
                'metrics_with_trends': len([k for k in trend_metrics.keys() if '_short_trend' in k])
            })
            
            # Create trend-based alerts
            for regression in trend_regressions:
                alert = RegressionAlert(
                    alert_id=f"trend_regr_{int(time.time())}_{regression['metric']}",
                    detection_time=datetime.now(),
                    affected_metrics=[regression['metric']],
                    severity=self._determine_trend_severity(regression),
                    regression_magnitude=regression['trend_change_magnitude'],
                    confidence_score=min(1.0, abs(regression['acceleration']) / 10),
                    root_cause_hints=self._generate_trend_hints(regression)
                )
                self.detected_regressions.append(alert)
            
            # Determine status
            status = 'passed'
            if len(trend_regressions) > 0:
                severe_trends = [r for r in trend_regressions if r['trend_change_magnitude'] > 5]
                if severe_trends:
                    status = 'failed'
                else:
                    status = 'warning'
            
            return TestResult(
                test_name=test_name,
                status=status,
                metrics=trend_metrics,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                details=f"Trend regressions: {len(trend_regressions)}, Metrics analyzed: {trend_metrics['metrics_with_trends']}"
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
    
    async def baseline_comparison_analysis(self) -> TestResult:
        """Compare current performance against established baselines"""
        start_time = time.time()
        test_name = "baseline_comparison"
        
        try:
            if not self.baselines:
                return TestResult(
                    test_name=test_name,
                    status="skipped",
                    metrics={},
                    execution_time=time.time() - start_time,
                    timestamp=datetime.now(),
                    details="No baselines available for comparison"
                )
            
            if not self.performance_history:
                return TestResult(
                    test_name=test_name,
                    status="skipped",
                    metrics={},
                    execution_time=time.time() - start_time,
                    timestamp=datetime.now(),
                    details="No performance history for comparison"
                )
            
            # Get recent performance data
            recent_performance = self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history
            current_avg_metrics = {}
            
            # Calculate current averages
            if recent_performance:
                df_recent = pd.DataFrame(recent_performance)
                for column in df_recent.columns:
                    if df_recent[column].dtype in ['float64', 'int64']:
                        current_avg_metrics[column] = df_recent[column].mean()
            
            baseline_metrics = {}
            baseline_violations = []
            
            for baseline_name, baseline in self.baselines.items():
                violations_for_baseline = []
                
                for metric_name, baseline_value in baseline.metrics.items():
                    if metric_name in current_avg_metrics:
                        current_value = current_avg_metrics[metric_name]
                        
                        # Calculate percentage difference
                        if baseline_value != 0:
                            percent_diff = ((current_value - baseline_value) / baseline_value) * 100
                        else:
                            percent_diff = 0
                        
                        baseline_metrics[f"{baseline_name}_{metric_name}_percent_diff"] = percent_diff
                        
                        # Check against thresholds
                        threshold = baseline.thresholds.get(metric_name, 10)  # Default 10% threshold
                        
                        if self._violates_baseline_threshold(metric_name, percent_diff, threshold):
                            violations_for_baseline.append({
                                'metric': metric_name,
                                'baseline_value': baseline_value,
                                'current_value': current_value,
                                'percent_diff': percent_diff,
                                'threshold': threshold
                            })
                
                baseline_violations.extend(violations_for_baseline)
            
            baseline_metrics.update({
                'baselines_compared': len(self.baselines),
                'total_violations': len(baseline_violations),
                'metrics_compared': len(current_avg_metrics),
                'recent_samples_used': len(recent_performance)
            })
            
            # Create baseline violation alerts
            for violation in baseline_violations:
                alert = RegressionAlert(
                    alert_id=f"baseline_regr_{int(time.time())}_{violation['metric']}",
                    detection_time=datetime.now(),
                    affected_metrics=[violation['metric']],
                    severity=self._determine_baseline_severity(violation),
                    regression_magnitude=abs(violation['percent_diff']),
                    confidence_score=min(1.0, abs(violation['percent_diff']) / 100),
                    root_cause_hints=self._generate_baseline_hints(violation)
                )
                self.detected_regressions.append(alert)
            
            # Determine status
            status = 'passed'
            if baseline_violations:
                critical_violations = [v for v in baseline_violations if abs(v['percent_diff']) > 50]
                if critical_violations:
                    status = 'failed'
                else:
                    status = 'warning'
            
            return TestResult(
                test_name=test_name,
                status=status,
                metrics=baseline_metrics,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                details=f"Baseline violations: {len(baseline_violations)}, Baselines: {len(self.baselines)}"
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
    
    def create_performance_baseline(self, baseline_name: str, metrics: Dict[str, float], 
                                  thresholds: Optional[Dict[str, float]] = None) -> None:
        """Create a new performance baseline"""
        if thresholds is None:
            # Generate default thresholds (10% for latency metrics, 5% for accuracy)
            thresholds = {}
            for metric_name in metrics.keys():
                if 'latency' in metric_name.lower() or 'time' in metric_name.lower():
                    thresholds[metric_name] = 15  # 15% threshold for latency
                elif 'accuracy' in metric_name.lower() or 'score' in metric_name.lower():
                    thresholds[metric_name] = 5   # 5% threshold for accuracy
                else:
                    thresholds[metric_name] = 10  # 10% default threshold
        
        # Calculate confidence intervals (assuming normal distribution)
        confidence_interval = {}
        for metric_name, value in metrics.items():
            # Simple confidence interval estimation (would be better with actual sample data)
            std_estimate = value * 0.1  # Assume 10% standard deviation
            confidence_interval[metric_name] = (
                value - 1.96 * std_estimate,  # 95% CI lower bound
                value + 1.96 * std_estimate   # 95% CI upper bound
            )
        
        baseline = PerformanceBaseline(
            baseline_id=baseline_name,
            creation_date=datetime.now(),
            metrics=metrics.copy(),
            thresholds=thresholds.copy(),
            sample_count=1,  # Would be updated with actual sample count
            confidence_interval=confidence_interval
        )
        
        self.baselines[baseline_name] = baseline
        
        # Save baseline to storage
        self._save_baseline(baseline)
        
        logger.info(f"Performance baseline '{baseline_name}' created with {len(metrics)} metrics")
    
    def update_performance_history(self, metrics: Dict[str, float]) -> None:
        """Update performance history with new metrics"""
        # Add timestamp to metrics
        timestamped_metrics = metrics.copy()
        timestamped_metrics['timestamp'] = time.time()
        
        self.performance_history.append(timestamped_metrics)
        
        # Keep only recent history (last 1000 samples)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def _save_baseline(self, baseline: PerformanceBaseline) -> None:
        """Save baseline to persistent storage"""
        try:
            baseline_file = self.baseline_storage_path / f"{baseline.baseline_id}.json"
            
            baseline_data = {
                'baseline_id': baseline.baseline_id,
                'creation_date': baseline.creation_date.isoformat(),
                'metrics': baseline.metrics,
                'thresholds': baseline.thresholds,
                'sample_count': baseline.sample_count,
                'confidence_interval': baseline.confidence_interval
            }
            
            with open(baseline_file, 'w') as f:
                json.dump(baseline_data, f, indent=2)
                
            logger.debug(f"Baseline '{baseline.baseline_id}' saved to {baseline_file}")
            
        except Exception as e:
            logger.error(f"Failed to save baseline '{baseline.baseline_id}': {e}")
    
    def _load_baselines(self) -> None:
        """Load baselines from persistent storage"""
        try:
            for baseline_file in self.baseline_storage_path.glob("*.json"):
                with open(baseline_file, 'r') as f:
                    baseline_data = json.load(f)
                
                baseline = PerformanceBaseline(
                    baseline_id=baseline_data['baseline_id'],
                    creation_date=datetime.fromisoformat(baseline_data['creation_date']),
                    metrics=baseline_data['metrics'],
                    thresholds=baseline_data['thresholds'],
                    sample_count=baseline_data['sample_count'],
                    confidence_interval=baseline_data['confidence_interval']
                )
                
                self.baselines[baseline.baseline_id] = baseline
            
            logger.info(f"Loaded {len(self.baselines)} performance baselines")
            
        except Exception as e:
            logger.error(f"Failed to load baselines: {e}")
    
    # Helper methods for regression detection logic
    def _is_regression(self, metric_name: str, change_percent: float) -> bool:
        """Determine if a metric change represents a regression"""
        # Latency metrics: increase is bad
        if 'latency' in metric_name.lower() or 'time' in metric_name.lower():
            return change_percent > 10  # 10% increase threshold
        
        # Accuracy metrics: decrease is bad
        elif 'accuracy' in metric_name.lower() or 'score' in metric_name.lower():
            return change_percent < -5  # 5% decrease threshold
        
        # Memory metrics: increase is bad
        elif 'memory' in metric_name.lower():
            return change_percent > 15  # 15% increase threshold
        
        # Error rate metrics: increase is bad
        elif 'error' in metric_name.lower():
            return change_percent > 20  # 20% increase threshold
        
        # Throughput metrics: decrease is bad
        elif 'throughput' in metric_name.lower() or 'rps' in metric_name.lower():
            return change_percent < -10  # 10% decrease threshold
        
        # Default: any significant change could be concerning
        else:
            return abs(change_percent) > 20  # 20% change threshold
    
    def _is_performance_degradation(self, metric_name: str, recent_value: float, baseline_value: float) -> bool:
        """Check if recent value represents performance degradation"""
        if baseline_value == 0:
            return False
        
        # Similar logic to _is_regression but with actual values
        change_percent = ((recent_value - baseline_value) / baseline_value) * 100
        return self._is_regression(metric_name, change_percent)
    
    def _is_concerning_trend(self, metric_name: str, trend_slope: float) -> bool:
        """Check if trend slope is concerning for the metric"""
        # Latency metrics: positive slope is bad
        if 'latency' in metric_name.lower() or 'time' in metric_name.lower():
            return trend_slope > 0.1
        
        # Accuracy metrics: negative slope is bad
        elif 'accuracy' in metric_name.lower() or 'score' in metric_name.lower():
            return trend_slope < -0.05
        
        # Memory metrics: positive slope is bad
        elif 'memory' in metric_name.lower():
            return trend_slope > 0.5
        
        # Error metrics: positive slope is bad
        elif 'error' in metric_name.lower():
            return trend_slope > 0.01
        
        return False
    
    def _is_trend_regression(self, metric_name: str, short_trend: float, long_trend: float, acceleration: float) -> bool:
        """Check if trend acceleration indicates regression"""
        # Look for acceleration in the wrong direction
        if 'latency' in metric_name.lower() or 'time' in metric_name.lower():
            # Latency getting worse faster
            return acceleration > 0.2 and short_trend > long_trend
        
        elif 'accuracy' in metric_name.lower() or 'score' in metric_name.lower():
            # Accuracy getting worse faster
            return acceleration < -0.1 and short_trend < long_trend
        
        elif 'memory' in metric_name.lower():
            # Memory usage increasing faster
            return acceleration > 1.0 and short_trend > long_trend
        
        return abs(acceleration) > 1.0  # Significant trend change
    
    def _violates_baseline_threshold(self, metric_name: str, percent_diff: float, threshold: float) -> bool:
        """Check if metric violates baseline threshold"""
        return self._is_regression(metric_name, percent_diff) and abs(percent_diff) > threshold
    
    def _determine_regression_severity(self, regression: Dict) -> str:
        """Determine severity of statistical regression"""
        degradation = abs(regression.get('degradation_percent', 0))
        z_score = regression.get('z_score', 0)
        
        if degradation > 50 or z_score > 5:
            return 'critical'
        elif degradation > 25 or z_score > 3:
            return 'high'
        elif degradation > 10 or z_score > 2:
            return 'medium'
        else:
            return 'low'
    
    def _determine_ml_severity(self, anomaly_score: float) -> str:
        """Determine severity of ML-detected anomaly"""
        score = abs(anomaly_score)
        
        if score > 0.8:
            return 'critical'
        elif score > 0.6:
            return 'high'
        elif score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _determine_trend_severity(self, regression: Dict) -> str:
        """Determine severity of trend regression"""
        magnitude = regression['trend_change_magnitude']
        
        if magnitude > 10:
            return 'critical'
        elif magnitude > 5:
            return 'high'
        elif magnitude > 2:
            return 'medium'
        else:
            return 'low'
    
    def _determine_baseline_severity(self, violation: Dict) -> str:
        """Determine severity of baseline violation"""
        percent_diff = abs(violation['percent_diff'])
        
        if percent_diff > 100:
            return 'critical'
        elif percent_diff > 50:
            return 'high'
        elif percent_diff > 25:
            return 'medium'
        else:
            return 'low'
    
    def _generate_root_cause_hints(self, regression: Dict) -> List[str]:
        """Generate root cause hints for statistical regression"""
        hints = []
        metric = regression['metric']
        
        if 'latency' in metric.lower():
            hints.extend([
                'Check for increased API response times',
                'Verify network conditions',
                'Review recent code changes affecting performance'
            ])
        elif 'memory' in metric.lower():
            hints.extend([
                'Look for memory leaks',
                'Check for increased data processing loads',
                'Review recent changes in data structures'
            ])
        elif 'accuracy' in metric.lower():
            hints.extend([
                'Verify model performance',
                'Check training data quality',
                'Review recent model updates'
            ])
        
        return hints
    
    def _generate_trend_hints(self, regression: Dict) -> List[str]:
        """Generate hints for trend-based regression"""
        return [
            'Gradual performance degradation detected',
            'Monitor system resource utilization',
            'Check for accumulating system state issues'
        ]
    
    def _generate_baseline_hints(self, violation: Dict) -> List[str]:
        """Generate hints for baseline violations"""
        metric = violation['metric']
        percent_diff = violation['percent_diff']
        
        hints = [f"Performance deviated {percent_diff:.1f}% from baseline"]
        
        if percent_diff > 0:
            hints.append(f"{metric} has degraded significantly")
        else:
            hints.append(f"{metric} has improved unexpectedly")
        
        return hints


def create_regression_detector(config: Optional[PerformanceTestConfig] = None) -> RegressionDetector:
    """Factory function to create regression detector"""
    if config is None:
        config = PerformanceTestConfig()
    return RegressionDetector(config) 