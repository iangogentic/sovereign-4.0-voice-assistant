"""
ML Model Drift Detection System

This module implements advanced drift detection algorithms for monitoring
AI model performance degradation over time, including:
- Population Stability Index (PSI) for distribution changes
- Kolmogorov-Smirnov tests for continuous features
- Jensen-Shannon Divergence for high-dimensional embeddings
- Performance-based drift detection using BLEU scores
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import accuracy_score
import threading
import time
import json
from pathlib import Path


class DriftType(Enum):
    """Types of drift detection"""
    STATISTICAL = "statistical"
    PERFORMANCE = "performance"
    DISTRIBUTION = "distribution"


class DriftSeverity(Enum):
    """Drift severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftAlert:
    """Drift detection alert"""
    timestamp: float
    drift_type: DriftType
    severity: DriftSeverity
    metric_name: str
    drift_score: float
    threshold: float
    description: str
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftConfig:
    """Configuration for drift detection"""
    # PSI thresholds
    psi_low_threshold: float = 0.1
    psi_medium_threshold: float = 0.2
    psi_high_threshold: float = 0.25
    
    # KS test thresholds
    ks_low_threshold: float = 0.1
    ks_medium_threshold: float = 0.2
    ks_high_threshold: float = 0.3
    
    # Performance thresholds
    accuracy_drop_low: float = 0.02  # 2% drop
    accuracy_drop_medium: float = 0.05  # 5% drop
    accuracy_drop_high: float = 0.10  # 10% drop
    
    # BLEU score thresholds
    bleu_drop_low: float = 0.03
    bleu_drop_medium: float = 0.05
    bleu_drop_high: float = 0.08
    
    # Jensen-Shannon Divergence thresholds
    js_low_threshold: float = 0.1
    js_medium_threshold: float = 0.2
    js_high_threshold: float = 0.3
    
    # Monitoring settings
    min_samples: int = 50
    baseline_window: int = 1000
    monitoring_window: int = 100
    
    # Persistence settings
    save_baselines: bool = True
    baseline_dir: str = ".taskmaster/drift_baselines"


class DriftDetector:
    """
    Advanced ML model drift detection system
    
    Implements multiple drift detection algorithms:
    - PSI for categorical and binned continuous distributions
    - KS test for continuous distribution changes
    - JS divergence for high-dimensional embeddings
    - Performance-based monitoring for accuracy/BLEU scores
    """
    
    def __init__(self, config: Optional[DriftConfig] = None):
        self.config = config or DriftConfig()
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
        
        # Baseline data storage
        self.baselines = {}
        self.baseline_stats = {}
        
        # Recent data windows for comparison
        self.recent_data = {}
        self.performance_history = {}
        
        # Alert history
        self.alert_history = []
        
        # Initialize baseline directory
        self._init_baseline_dir()
        
        # Load existing baselines
        self._load_baselines()
    
    def _init_baseline_dir(self):
        """Initialize baseline storage directory"""
        baseline_path = Path(self.config.baseline_dir)
        baseline_path.mkdir(parents=True, exist_ok=True)
    
    def _load_baselines(self):
        """Load existing baselines from disk"""
        if not self.config.save_baselines:
            return
            
        baseline_path = Path(self.config.baseline_dir)
        
        for baseline_file in baseline_path.glob("*.json"):
            try:
                with open(baseline_file, 'r') as f:
                    data = json.load(f)
                    metric_name = baseline_file.stem
                    self.baselines[metric_name] = np.array(data['baseline_data'])
                    self.baseline_stats[metric_name] = data['stats']
                    self.logger.info(f"Loaded baseline for {metric_name}")
            except Exception as e:
                self.logger.warning(f"Failed to load baseline {baseline_file}: {e}")
    
    def _save_baseline(self, metric_name: str):
        """Save baseline to disk"""
        if not self.config.save_baselines or metric_name not in self.baselines:
            return
            
        baseline_path = Path(self.config.baseline_dir) / f"{metric_name}.json"
        
        try:
            data = {
                'baseline_data': self.baselines[metric_name].tolist(),
                'stats': self.baseline_stats[metric_name],
                'created_at': time.time(),
                'config': {
                    'min_samples': self.config.min_samples,
                    'baseline_window': self.config.baseline_window
                }
            }
            
            with open(baseline_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.logger.info(f"Saved baseline for {metric_name}")
        except Exception as e:
            self.logger.error(f"Failed to save baseline {metric_name}: {e}")
    
    def establish_baseline(self, metric_name: str, data: np.ndarray) -> bool:
        """
        Establish baseline distribution for drift detection
        
        Args:
            metric_name: Name of the metric
            data: Baseline data array
            
        Returns:
            True if baseline was established successfully
        """
        with self._lock:
            if len(data) < self.config.min_samples:
                self.logger.warning(
                    f"Insufficient data for baseline {metric_name}: "
                    f"{len(data)} < {self.config.min_samples}"
                )
                return False
            
            # Store baseline data (limited to baseline_window size)
            if len(data) > self.config.baseline_window:
                data = data[-self.config.baseline_window:]
            
            self.baselines[metric_name] = data.copy()
            
            # Calculate baseline statistics
            self.baseline_stats[metric_name] = {
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'median': float(np.median(data)),
                'q25': float(np.percentile(data, 25)),
                'q75': float(np.percentile(data, 75)),
                'samples': len(data),
                'timestamp': time.time()
            }
            
            # Initialize recent data tracking
            self.recent_data[metric_name] = []
            
            # Save baseline
            self._save_baseline(metric_name)
            
            self.logger.info(f"Established baseline for {metric_name} with {len(data)} samples")
            return True
    
    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, 
                     buckets: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI)
        
        Args:
            expected: Baseline distribution
            actual: Current distribution
            buckets: Number of buckets for binning
            
        Returns:
            PSI score (higher values indicate more drift)
        """
        try:
            # Create bins based on expected distribution
            bin_edges = np.percentile(expected, np.linspace(0, 100, buckets + 1))
            
            # Handle edge case where all values are the same
            if len(np.unique(bin_edges)) < 2:
                return 0.0
            
            # Calculate expected percentages
            expected_counts, _ = np.histogram(expected, bins=bin_edges)
            expected_percents = expected_counts / len(expected)
            
            # Calculate actual percentages
            actual_counts, _ = np.histogram(actual, bins=bin_edges)
            actual_percents = actual_counts / len(actual)
            
            # Avoid division by zero
            expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
            actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
            
            # Calculate PSI
            psi = np.sum((actual_percents - expected_percents) * 
                        np.log(actual_percents / expected_percents))
            
            return float(psi)
            
        except Exception as e:
            self.logger.error(f"Error calculating PSI: {e}")
            return 0.0
    
    def calculate_ks_statistic(self, expected: np.ndarray, 
                              actual: np.ndarray) -> Tuple[float, float]:
        """
        Calculate Kolmogorov-Smirnov test statistic
        
        Args:
            expected: Baseline distribution
            actual: Current distribution
            
        Returns:
            Tuple of (KS statistic, p-value)
        """
        try:
            ks_stat, p_value = stats.ks_2samp(expected, actual)
            return float(ks_stat), float(p_value)
        except Exception as e:
            self.logger.error(f"Error calculating KS statistic: {e}")
            return 0.0, 1.0
    
    def calculate_js_divergence(self, expected: np.ndarray, 
                               actual: np.ndarray) -> float:
        """
        Calculate Jensen-Shannon divergence
        
        Args:
            expected: Baseline distribution
            actual: Current distribution
            
        Returns:
            JS divergence score
        """
        try:
            # For 1D data, create histograms
            if expected.ndim == 1:
                bins = min(50, len(np.unique(expected)))
                range_min = min(np.min(expected), np.min(actual))
                range_max = max(np.max(expected), np.max(actual))
                
                hist_expected, _ = np.histogram(expected, bins=bins, 
                                              range=(range_min, range_max))
                hist_actual, _ = np.histogram(actual, bins=bins, 
                                            range=(range_min, range_max))
                
                # Normalize to probabilities
                p = hist_expected / np.sum(hist_expected)
                q = hist_actual / np.sum(hist_actual)
                
                # Add small epsilon to avoid log(0)
                epsilon = 1e-10
                p = p + epsilon
                q = q + epsilon
                
                # Renormalize
                p = p / np.sum(p)
                q = q / np.sum(q)
                
                return float(jensenshannon(p, q))
            
            else:
                # For multi-dimensional data, flatten and use directly
                p = expected.flatten()
                q = actual.flatten()
                
                # Normalize
                p = p / np.sum(p) if np.sum(p) > 0 else p
                q = q / np.sum(q) if np.sum(q) > 0 else q
                
                return float(jensenshannon(p, q))
                
        except Exception as e:
            self.logger.error(f"Error calculating JS divergence: {e}")
            return 0.0
    
    def _assess_drift_severity(self, drift_score: float, thresholds: Dict[str, float],
                              drift_type: DriftType) -> DriftSeverity:
        """Assess drift severity based on score and thresholds"""
        if drift_score >= thresholds.get('high', 0.3):
            return DriftSeverity.CRITICAL
        elif drift_score >= thresholds.get('medium', 0.2):
            return DriftSeverity.HIGH
        elif drift_score >= thresholds.get('low', 0.1):
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.LOW
    
    def _generate_recommendations(self, metric_name: str, drift_type: DriftType,
                                 severity: DriftSeverity) -> List[str]:
        """Generate recommendations based on drift detection"""
        recommendations = []
        
        if severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
            recommendations.extend([
                f"Investigate recent changes in {metric_name} data pipeline",
                "Consider retraining the model with recent data",
                "Review data preprocessing steps for inconsistencies",
                "Check for external factors affecting data distribution"
            ])
            
            if drift_type == DriftType.PERFORMANCE:
                recommendations.extend([
                    "Analyze error patterns for recurring issues",
                    "Review recent model deployments or configuration changes",
                    "Consider temporary rollback to previous model version"
                ])
            
            elif drift_type == DriftType.DISTRIBUTION:
                recommendations.extend([
                    "Examine feature engineering pipeline",
                    "Check for data quality issues in recent inputs",
                    "Review data collection methodology"
                ])
        
        elif severity == DriftSeverity.MEDIUM:
            recommendations.extend([
                f"Monitor {metric_name} closely for continued drift",
                "Schedule model performance review",
                "Prepare contingency plans for model retraining"
            ])
        
        return recommendations
    
    def detect_drift(self, metric_name: str, current_data: np.ndarray,
                    current_performance: Optional[float] = None) -> List[DriftAlert]:
        """
        Detect drift in current data compared to baseline
        
        Args:
            metric_name: Name of the metric being monitored
            current_data: Current data to compare against baseline
            current_performance: Current performance metric (e.g., accuracy, BLEU)
            
        Returns:
            List of drift alerts
        """
        alerts = []
        
        with self._lock:
            # Check if baseline exists
            if metric_name not in self.baselines:
                self.logger.warning(f"No baseline established for {metric_name}")
                return alerts
            
            # Update recent data
            if metric_name not in self.recent_data:
                self.recent_data[metric_name] = []
            
            self.recent_data[metric_name].extend(current_data.flatten())
            
            # Keep only recent monitoring window
            if len(self.recent_data[metric_name]) > self.config.monitoring_window:
                self.recent_data[metric_name] = \
                    self.recent_data[metric_name][-self.config.monitoring_window:]
            
            # Need sufficient data for comparison
            if len(self.recent_data[metric_name]) < self.config.min_samples:
                return alerts
            
            baseline = self.baselines[metric_name]
            recent = np.array(self.recent_data[metric_name])
            
            # 1. PSI-based drift detection
            try:
                psi_score = self.calculate_psi(baseline, recent)
                psi_thresholds = {
                    'low': self.config.psi_low_threshold,
                    'medium': self.config.psi_medium_threshold,
                    'high': self.config.psi_high_threshold
                }
                
                psi_severity = self._assess_drift_severity(psi_score, psi_thresholds, 
                                                          DriftType.STATISTICAL)
                
                if psi_severity != DriftSeverity.LOW:
                    alerts.append(DriftAlert(
                        timestamp=time.time(),
                        drift_type=DriftType.STATISTICAL,
                        severity=psi_severity,
                        metric_name=metric_name,
                        drift_score=psi_score,
                        threshold=psi_thresholds['medium'],
                        description=f"PSI drift detected in {metric_name}: {psi_score:.3f}",
                        recommendations=self._generate_recommendations(
                            metric_name, DriftType.STATISTICAL, psi_severity
                        ),
                        metadata={'test_type': 'PSI', 'baseline_size': len(baseline)}
                    ))
                
            except Exception as e:
                self.logger.error(f"Error in PSI drift detection for {metric_name}: {e}")
            
            # 2. KS test drift detection
            try:
                ks_stat, p_value = self.calculate_ks_statistic(baseline, recent)
                ks_thresholds = {
                    'low': self.config.ks_low_threshold,
                    'medium': self.config.ks_medium_threshold,
                    'high': self.config.ks_high_threshold
                }
                
                ks_severity = self._assess_drift_severity(ks_stat, ks_thresholds,
                                                         DriftType.DISTRIBUTION)
                
                if ks_severity != DriftSeverity.LOW:
                    alerts.append(DriftAlert(
                        timestamp=time.time(),
                        drift_type=DriftType.DISTRIBUTION,
                        severity=ks_severity,
                        metric_name=metric_name,
                        drift_score=ks_stat,
                        threshold=ks_thresholds['medium'],
                        description=f"KS drift detected in {metric_name}: {ks_stat:.3f} (p={p_value:.3f})",
                        recommendations=self._generate_recommendations(
                            metric_name, DriftType.DISTRIBUTION, ks_severity
                        ),
                        metadata={'test_type': 'KS', 'p_value': p_value}
                    ))
                
            except Exception as e:
                self.logger.error(f"Error in KS drift detection for {metric_name}: {e}")
            
            # 3. Jensen-Shannon divergence
            try:
                js_score = self.calculate_js_divergence(baseline, recent)
                js_thresholds = {
                    'low': self.config.js_low_threshold,
                    'medium': self.config.js_medium_threshold,
                    'high': self.config.js_high_threshold
                }
                
                js_severity = self._assess_drift_severity(js_score, js_thresholds,
                                                         DriftType.DISTRIBUTION)
                
                if js_severity != DriftSeverity.LOW:
                    alerts.append(DriftAlert(
                        timestamp=time.time(),
                        drift_type=DriftType.DISTRIBUTION,
                        severity=js_severity,
                        metric_name=metric_name,
                        drift_score=js_score,
                        threshold=js_thresholds['medium'],
                        description=f"JS divergence drift detected in {metric_name}: {js_score:.3f}",
                        recommendations=self._generate_recommendations(
                            metric_name, DriftType.DISTRIBUTION, js_severity
                        ),
                        metadata={'test_type': 'Jensen-Shannon'}
                    ))
                
            except Exception as e:
                self.logger.error(f"Error in JS drift detection for {metric_name}: {e}")
            
            # 4. Performance-based drift detection
            if current_performance is not None:
                try:
                    if metric_name not in self.performance_history:
                        self.performance_history[metric_name] = []
                    
                    self.performance_history[metric_name].append(current_performance)
                    
                    # Keep performance history limited
                    if len(self.performance_history[metric_name]) > self.config.baseline_window:
                        self.performance_history[metric_name] = \
                            self.performance_history[metric_name][-self.config.baseline_window:]
                    
                    # Calculate baseline performance
                    if len(self.performance_history[metric_name]) >= self.config.min_samples:
                        baseline_perf = np.mean(
                            self.performance_history[metric_name][:self.config.baseline_window//2]
                        )
                        current_perf = current_performance
                        performance_drop = baseline_perf - current_perf
                        
                        # Determine thresholds based on metric type
                        if 'bleu' in metric_name.lower():
                            perf_thresholds = {
                                'low': self.config.bleu_drop_low,
                                'medium': self.config.bleu_drop_medium,
                                'high': self.config.bleu_drop_high
                            }
                        else:
                            perf_thresholds = {
                                'low': self.config.accuracy_drop_low,
                                'medium': self.config.accuracy_drop_medium,
                                'high': self.config.accuracy_drop_high
                            }
                        
                        perf_severity = self._assess_drift_severity(
                            performance_drop, perf_thresholds, DriftType.PERFORMANCE
                        )
                        
                        if perf_severity != DriftSeverity.LOW:
                            alerts.append(DriftAlert(
                                timestamp=time.time(),
                                drift_type=DriftType.PERFORMANCE,
                                severity=perf_severity,
                                metric_name=metric_name,
                                drift_score=performance_drop,
                                threshold=perf_thresholds['medium'],
                                description=f"Performance drift detected in {metric_name}: "
                                          f"{performance_drop:.3f} drop from baseline",
                                recommendations=self._generate_recommendations(
                                    metric_name, DriftType.PERFORMANCE, perf_severity
                                ),
                                metadata={
                                    'baseline_performance': baseline_perf,
                                    'current_performance': current_perf,
                                    'performance_drop': performance_drop
                                }
                            ))
                
                except Exception as e:
                    self.logger.error(f"Error in performance drift detection for {metric_name}: {e}")
            
            # Store alerts in history
            self.alert_history.extend(alerts)
            
            # Keep alert history limited
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-1000:]
        
        return alerts
    
    def get_drift_summary(self, metric_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary of drift detection status
        
        Args:
            metric_name: Specific metric to summarize (None for all)
            
        Returns:
            Drift detection summary
        """
        with self._lock:
            summary = {
                'timestamp': time.time(),
                'total_metrics_monitored': len(self.baselines),
                'total_alerts': len(self.alert_history),
                'metrics': {}
            }
            
            # Filter metrics
            metrics_to_summarize = [metric_name] if metric_name else list(self.baselines.keys())
            
            for metric in metrics_to_summarize:
                if metric not in self.baselines:
                    continue
                
                # Get recent alerts for this metric
                recent_alerts = [
                    alert for alert in self.alert_history[-100:]
                    if alert.metric_name == metric
                ]
                
                metric_summary = {
                    'baseline_established': True,
                    'baseline_samples': len(self.baselines[metric]),
                    'recent_samples': len(self.recent_data.get(metric, [])),
                    'recent_alerts': len(recent_alerts),
                    'last_alert': None,
                    'current_status': 'stable'
                }
                
                if recent_alerts:
                    latest_alert = max(recent_alerts, key=lambda x: x.timestamp)
                    metric_summary['last_alert'] = {
                        'timestamp': latest_alert.timestamp,
                        'severity': latest_alert.severity.value,
                        'drift_type': latest_alert.drift_type.value,
                        'description': latest_alert.description
                    }
                    
                    # Determine current status based on recent alerts
                    high_severity_alerts = [
                        a for a in recent_alerts[-10:]
                        if a.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]
                    ]
                    
                    if high_severity_alerts:
                        metric_summary['current_status'] = 'drifting'
                    else:
                        metric_summary['current_status'] = 'monitoring'
                
                summary['metrics'][metric] = metric_summary
            
            return summary
    
    def reset_baseline(self, metric_name: str) -> bool:
        """
        Reset baseline for a metric
        
        Args:
            metric_name: Name of metric to reset
            
        Returns:
            True if reset was successful
        """
        with self._lock:
            try:
                if metric_name in self.baselines:
                    del self.baselines[metric_name]
                if metric_name in self.baseline_stats:
                    del self.baseline_stats[metric_name]
                if metric_name in self.recent_data:
                    del self.recent_data[metric_name]
                if metric_name in self.performance_history:
                    del self.performance_history[metric_name]
                
                # Remove saved baseline file
                if self.config.save_baselines:
                    baseline_file = Path(self.config.baseline_dir) / f"{metric_name}.json"
                    if baseline_file.exists():
                        baseline_file.unlink()
                
                self.logger.info(f"Reset baseline for {metric_name}")
                return True
                
            except Exception as e:
                self.logger.error(f"Error resetting baseline for {metric_name}: {e}")
                return False


# Factory function for easy instantiation
def create_drift_detector(config: Optional[DriftConfig] = None) -> DriftDetector:
    """Create a drift detector with optional configuration"""
    return DriftDetector(config)


# Global drift detector instance
_global_drift_detector: Optional[DriftDetector] = None


def get_drift_detector() -> DriftDetector:
    """Get global drift detector instance"""
    global _global_drift_detector
    if _global_drift_detector is None:
        _global_drift_detector = create_drift_detector()
    return _global_drift_detector


def set_drift_detector(detector: DriftDetector):
    """Set global drift detector instance"""
    global _global_drift_detector
    _global_drift_detector = detector 