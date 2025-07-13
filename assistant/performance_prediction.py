"""
Performance Degradation Prediction System

This module implements advanced performance degradation prediction for AI voice assistants:
- Ensemble ML models for robust prediction
- Early warning system with proactive alerts
- Feature engineering for comprehensive performance indicators
- Confidence scoring and uncertainty quantification
- Integration with existing metrics collection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
import time
import json
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)


class RiskLevel(Enum):
    """Performance degradation risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PredictionHorizon(Enum):
    """Prediction time horizons"""
    HOURS_1 = 1
    HOURS_4 = 4
    HOURS_12 = 12
    HOURS_24 = 24
    HOURS_48 = 48


@dataclass
class PerformancePrediction:
    """Performance degradation prediction result"""
    timestamp: float
    horizon_hours: int
    degradation_probability: float
    confidence_interval: Tuple[float, float]
    risk_level: RiskLevel
    contributing_factors: List[str]
    feature_importance: Dict[str, float]
    model_confidence: float
    predicted_metrics: Dict[str, float]
    recommended_actions: List[str] = field(default_factory=list)
    uncertainty_score: float = 0.0


@dataclass
class EarlyWarning:
    """Early warning alert for performance degradation"""
    timestamp: float
    warning_type: str
    risk_level: RiskLevel
    probability: float
    time_to_degradation: Optional[float]  # hours
    affected_metrics: List[str]
    root_causes: List[str]
    preventive_actions: List[str]
    confidence: float


@dataclass
class PerformancePredictionConfig:
    """Configuration for performance prediction system"""
    # Model settings
    random_forest_n_estimators: int = 100
    random_forest_max_depth: Optional[int] = None
    gradient_boosting_n_estimators: int = 100
    gradient_boosting_learning_rate: float = 0.1
    gradient_boosting_max_depth: int = 6
    
    # Feature engineering
    feature_window_hours: int = 24
    trend_calculation_windows: List[int] = field(default_factory=lambda: [1, 4, 12, 24])
    seasonality_features: bool = True
    lag_features: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 12])
    
    # Prediction settings
    default_horizon_hours: int = 24
    min_training_samples: int = 168  # 1 week
    retrain_interval_hours: int = 24
    cross_validation_folds: int = 5
    
    # Risk thresholds
    low_risk_threshold: float = 0.2
    medium_risk_threshold: float = 0.4
    high_risk_threshold: float = 0.7
    critical_risk_threshold: float = 0.85
    
    # Early warning settings
    warning_lookback_hours: int = 4
    trend_warning_threshold: float = 0.15  # 15% trend change
    anomaly_warning_threshold: float = 2.0  # 2 standard deviations
    
    # Model persistence
    save_models: bool = True
    model_dir: str = ".taskmaster/prediction_models"
    
    # Performance targets (from Task 10.1)
    target_latency_p95: float = 800.0  # ms
    target_accuracy: float = 0.85
    target_bleu_score: float = 0.80
    target_cpu_usage: float = 0.70
    target_memory_usage: float = 0.80


class PerformancePredictor:
    """
    Advanced performance degradation prediction system
    
    Uses ensemble machine learning models to predict performance degradation
    and provide early warnings with actionable recommendations.
    """
    
    def __init__(self, config: Optional[PerformancePredictionConfig] = None):
        self.config = config or PerformancePredictionConfig()
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
        
        # Models
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=self.config.random_forest_n_estimators,
                max_depth=self.config.random_forest_max_depth,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=self.config.gradient_boosting_n_estimators,
                learning_rate=self.config.gradient_boosting_learning_rate,
                max_depth=self.config.gradient_boosting_max_depth,
                random_state=42
            ),
            'linear_regression': LinearRegression()
        }
        
        # Model metadata
        self.model_weights = {'random_forest': 0.4, 'gradient_boosting': 0.4, 'linear_regression': 0.2}
        self.model_performance = {}
        self.feature_scaler = StandardScaler()
        self.is_trained = False
        self.last_training_time = None
        
        # Feature importance tracking
        self.feature_names = []
        self.feature_importance = {}
        
        # Historical data for prediction
        self.training_data = None
        
        # Early warning system
        self.warning_history = []
        self.baseline_metrics = {}
        
        # Initialize storage
        self._init_model_dir()
        
        # Load existing models
        self._load_models()
    
    def _init_model_dir(self):
        """Initialize model storage directory"""
        model_path = Path(self.config.model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
    
    def _load_models(self):
        """Load existing models from disk"""
        if not self.config.save_models:
            return
            
        model_path = Path(self.config.model_dir)
        
        try:
            # Load models
            for model_name in self.models.keys():
                model_file = model_path / f"{model_name}.joblib"
                if model_file.exists():
                    self.models[model_name] = joblib.load(model_file)
                    self.logger.info(f"Loaded {model_name} model")
            
            # Load scaler
            scaler_file = model_path / "feature_scaler.joblib"
            if scaler_file.exists():
                self.feature_scaler = joblib.load(scaler_file)
                self.logger.info("Loaded feature scaler")
            
            # Load metadata
            metadata_file = model_path / "model_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.model_weights = metadata.get('weights', self.model_weights)
                    self.model_performance = metadata.get('performance', {})
                    self.feature_names = metadata.get('feature_names', [])
                    self.is_trained = metadata.get('is_trained', False)
                    self.last_training_time = metadata.get('last_training_time')
                    self.baseline_metrics = metadata.get('baseline_metrics', {})
                self.logger.info("Loaded model metadata")
                
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
    
    def _save_models(self):
        """Save models to disk"""
        if not self.config.save_models:
            return
            
        model_path = Path(self.config.model_dir)
        
        try:
            # Save models
            for model_name, model in self.models.items():
                model_file = model_path / f"{model_name}.joblib"
                joblib.dump(model, model_file)
            
            # Save scaler
            scaler_file = model_path / "feature_scaler.joblib"
            joblib.dump(self.feature_scaler, scaler_file)
            
            # Save metadata
            metadata = {
                'weights': self.model_weights,
                'performance': self.model_performance,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained,
                'last_training_time': self.last_training_time,
                'baseline_metrics': self.baseline_metrics,
                'config': {
                    'horizon_hours': self.config.default_horizon_hours,
                    'min_samples': self.config.min_training_samples
                }
            }
            
            metadata_file = model_path / "model_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info("Saved models and metadata")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for performance prediction"""
        features_df = data.copy()
        
        # Time-based features
        if 'timestamp' in features_df.columns:
            features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
            features_df.set_index('timestamp', inplace=True)
        
        features_df['hour_of_day'] = features_df.index.hour
        features_df['day_of_week'] = features_df.index.dayofweek
        features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
        features_df['is_business_hours'] = ((features_df['hour_of_day'] >= 9) & 
                                           (features_df['hour_of_day'] <= 17)).astype(int)
        
        # Resource utilization features
        resource_columns = ['cpu_usage', 'memory_usage', 'gpu_usage']
        for col in resource_columns:
            if col in features_df.columns:
                # Rolling averages
                for window in self.config.trend_calculation_windows:
                    features_df[f'{col}_avg_{window}h'] = features_df[col].rolling(
                        window=window, min_periods=1
                    ).mean()
                    features_df[f'{col}_max_{window}h'] = features_df[col].rolling(
                        window=window, min_periods=1
                    ).max()
                    features_df[f'{col}_std_{window}h'] = features_df[col].rolling(
                        window=window, min_periods=1
                    ).std()
                
                # Trends
                features_df[f'{col}_trend_1h'] = features_df[col].pct_change(periods=1)
                features_df[f'{col}_trend_4h'] = features_df[col].pct_change(periods=4)
                
                # Lag features
                for lag in self.config.lag_features:
                    features_df[f'{col}_lag_{lag}'] = features_df[col].shift(lag)
        
        # Performance metrics features
        performance_columns = ['latency_p95', 'accuracy_score', 'bleu_score', 'error_rate']
        for col in performance_columns:
            if col in features_df.columns:
                # Rolling statistics
                for window in self.config.trend_calculation_windows:
                    features_df[f'{col}_avg_{window}h'] = features_df[col].rolling(
                        window=window, min_periods=1
                    ).mean()
                    features_df[f'{col}_trend_{window}h'] = features_df[col].pct_change(periods=window)
                
                # Deviation from baseline
                if col in self.baseline_metrics:
                    baseline = self.baseline_metrics[col]
                    features_df[f'{col}_deviation'] = (features_df[col] - baseline) / baseline
        
        # System load features
        if 'request_rate' in features_df.columns:
            features_df['request_rate_avg_1h'] = features_df['request_rate'].rolling(
                window=1, min_periods=1
            ).mean()
            features_df['request_rate_peak_4h'] = features_df['request_rate'].rolling(
                window=4, min_periods=1
            ).max()
        
        # External factors
        if 'api_response_time' in features_df.columns:
            features_df['api_response_time_avg_1h'] = features_df['api_response_time'].rolling(
                window=1, min_periods=1
            ).mean()
        
        # Drop original timestamp column if it was added
        numeric_features = features_df.select_dtypes(include=[np.number])
        
        # Fill NaN values
        numeric_features = numeric_features.fillna(method='ffill').fillna(0)
        
        return numeric_features
    
    def _create_target_variable(self, data: pd.DataFrame) -> np.ndarray:
        """Create target variable for degradation prediction"""
        # Create composite degradation score
        degradation_scores = []
        
        for i in range(len(data)):
            score = 0.0
            weight_sum = 0.0
            
            # Latency degradation
            if 'latency_p95' in data.columns:
                latency = data['latency_p95'].iloc[i]
                if latency > self.config.target_latency_p95:
                    score += (latency / self.config.target_latency_p95 - 1) * 0.3
                weight_sum += 0.3
            
            # Accuracy degradation
            if 'accuracy_score' in data.columns:
                accuracy = data['accuracy_score'].iloc[i]
                if accuracy < self.config.target_accuracy:
                    score += (self.config.target_accuracy - accuracy) * 0.25
                weight_sum += 0.25
            
            # BLEU score degradation
            if 'bleu_score' in data.columns:
                bleu = data['bleu_score'].iloc[i]
                if bleu < self.config.target_bleu_score:
                    score += (self.config.target_bleu_score - bleu) * 0.2
                weight_sum += 0.2
            
            # Resource usage
            if 'cpu_usage' in data.columns:
                cpu = data['cpu_usage'].iloc[i]
                if cpu > self.config.target_cpu_usage:
                    score += (cpu - self.config.target_cpu_usage) * 0.15
                weight_sum += 0.15
            
            if 'memory_usage' in data.columns:
                memory = data['memory_usage'].iloc[i]
                if memory > self.config.target_memory_usage:
                    score += (memory - self.config.target_memory_usage) * 0.1
                weight_sum += 0.1
            
            # Normalize score
            if weight_sum > 0:
                score = score / weight_sum
            
            degradation_scores.append(min(score, 1.0))  # Cap at 1.0
        
        return np.array(degradation_scores)
    
    def train_models(self, data: pd.DataFrame) -> bool:
        """Train performance prediction models"""
        try:
            with self._lock:
                if len(data) < self.config.min_training_samples:
                    self.logger.warning(
                        f"Insufficient data for training: {len(data)} < {self.config.min_training_samples}"
                    )
                    return False
                
                # Engineer features
                features_df = self._engineer_features(data)
                
                # Create target variable
                target = self._create_target_variable(data)
                
                # Align features and target
                min_length = min(len(features_df), len(target))
                features_df = features_df.iloc[:min_length]
                target = target[:min_length]
                
                # Store feature names
                self.feature_names = list(features_df.columns)
                
                # Scale features
                X = self.feature_scaler.fit_transform(features_df)
                y = target
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Train models
                for model_name, model in self.models.items():
                    try:
                        model.fit(X_train, y_train)
                        
                        # Evaluate model
                        y_pred = model.predict(X_test)
                        
                        # Calculate metrics
                        mae = mean_absolute_error(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        # Cross-validation score
                        cv_scores = cross_val_score(
                            model, X_train, y_train, 
                            cv=self.config.cross_validation_folds,
                            scoring='neg_mean_absolute_error'
                        )
                        cv_mean = -cv_scores.mean()
                        
                        self.model_performance[model_name] = {
                            'mae': mae,
                            'mse': mse,
                            'r2': r2,
                            'cv_score': cv_mean,
                            'training_samples': len(X_train)
                        }
                        
                        # Update model weights based on performance
                        if cv_mean < 0.1:  # Good performance
                            self.model_weights[model_name] *= 1.1
                        elif cv_mean > 0.3:  # Poor performance
                            self.model_weights[model_name] *= 0.9
                        
                        self.logger.info(
                            f"Trained {model_name}: MAE={mae:.3f}, R²={r2:.3f}, CV={cv_mean:.3f}"
                        )
                        
                    except Exception as e:
                        self.logger.error(f"Error training {model_name}: {e}")
                        continue
                
                # Normalize weights
                total_weight = sum(self.model_weights.values())
                if total_weight > 0:
                    self.model_weights = {
                        k: v / total_weight for k, v in self.model_weights.items()
                    }
                
                # Calculate feature importance
                self._calculate_feature_importance()
                
                # Store training data and baseline metrics
                self.training_data = data.copy()
                self._calculate_baseline_metrics(data)
                
                self.is_trained = True
                self.last_training_time = time.time()
                
                # Save models
                self._save_models()
                
                self.logger.info("Performance prediction models trained successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Error training models: {e}")
            return False
    
    def _calculate_feature_importance(self):
        """Calculate aggregate feature importance from all models"""
        if not self.feature_names:
            return
            
        importance_dict = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                model_importance = model.feature_importances_
                weight = self.model_weights.get(model_name, 0)
                
                for i, feature in enumerate(self.feature_names):
                    if feature not in importance_dict:
                        importance_dict[feature] = 0
                    importance_dict[feature] += model_importance[i] * weight
        
        # Normalize
        total_importance = sum(importance_dict.values())
        if total_importance > 0:
            self.feature_importance = {
                k: v / total_importance for k, v in importance_dict.items()
            }
    
    def _calculate_baseline_metrics(self, data: pd.DataFrame):
        """Calculate baseline metrics for deviation calculations"""
        baseline_window = min(len(data), 168)  # 1 week
        recent_data = data.tail(baseline_window)
        
        metrics = ['latency_p95', 'accuracy_score', 'bleu_score', 'cpu_usage', 'memory_usage']
        
        for metric in metrics:
            if metric in recent_data.columns:
                self.baseline_metrics[metric] = recent_data[metric].median()
    
    def predict_degradation(self, current_data: pd.DataFrame,
                          horizon_hours: Optional[int] = None) -> Optional[PerformancePrediction]:
        """
        Predict performance degradation probability
        
        Args:
            current_data: Recent performance data
            horizon_hours: Prediction horizon in hours
            
        Returns:
            Performance prediction result
        """
        if not self.is_trained:
            self.logger.warning("Models not trained yet")
            return None
        
        if horizon_hours is None:
            horizon_hours = self.config.default_horizon_hours
        
        try:
            with self._lock:
                # Engineer features
                features_df = self._engineer_features(current_data)
                
                if len(features_df) == 0:
                    self.logger.warning("No features available for prediction")
                    return None
                
                # Use latest data point for prediction
                latest_features = features_df.iloc[-1:][self.feature_names]
                X = self.feature_scaler.transform(latest_features)
                
                # Get predictions from all models
                predictions = {}
                for model_name, model in self.models.items():
                    try:
                        pred = model.predict(X)[0]
                        predictions[model_name] = pred
                    except Exception as e:
                        self.logger.error(f"Error predicting with {model_name}: {e}")
                        continue
                
                if not predictions:
                    self.logger.error("No model predictions available")
                    return None
                
                # Calculate ensemble prediction
                ensemble_pred = sum(
                    pred * self.model_weights.get(model_name, 0)
                    for model_name, pred in predictions.items()
                )
                
                # Calculate confidence intervals
                pred_values = list(predictions.values())
                pred_std = np.std(pred_values) if len(pred_values) > 1 else 0.1
                confidence_interval = (
                    max(0, ensemble_pred - 1.96 * pred_std),
                    min(1, ensemble_pred + 1.96 * pred_std)
                )
                
                # Assess risk level
                risk_level = self._assess_risk_level(ensemble_pred)
                
                # Identify contributing factors
                contributing_factors = self._identify_contributing_factors(latest_features.iloc[0])
                
                # Generate recommendations
                recommendations = self._generate_recommendations(
                    ensemble_pred, risk_level, contributing_factors
                )
                
                # Predict specific metrics
                predicted_metrics = self._predict_specific_metrics(current_data, horizon_hours)
                
                # Calculate model confidence
                model_confidence = 1.0 - pred_std
                uncertainty_score = pred_std
                
                return PerformancePrediction(
                    timestamp=time.time(),
                    horizon_hours=horizon_hours,
                    degradation_probability=ensemble_pred,
                    confidence_interval=confidence_interval,
                    risk_level=risk_level,
                    contributing_factors=contributing_factors,
                    feature_importance=dict(list(self.feature_importance.items())[:10]),  # Top 10
                    model_confidence=model_confidence,
                    predicted_metrics=predicted_metrics,
                    recommended_actions=recommendations,
                    uncertainty_score=uncertainty_score
                )
                
        except Exception as e:
            self.logger.error(f"Error predicting degradation: {e}")
            return None
    
    def _assess_risk_level(self, probability: float) -> RiskLevel:
        """Assess risk level based on degradation probability"""
        if probability >= self.config.critical_risk_threshold:
            return RiskLevel.CRITICAL
        elif probability >= self.config.high_risk_threshold:
            return RiskLevel.HIGH
        elif probability >= self.config.medium_risk_threshold:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _identify_contributing_factors(self, features: pd.Series) -> List[str]:
        """Identify factors contributing to degradation risk"""
        factors = []
        
        # Check high-importance features
        for feature, importance in self.feature_importance.items():
            if importance > 0.05 and feature in features:  # Top 5% importance
                value = features[feature]
                
                if 'cpu_usage' in feature and value > 0.8:
                    factors.append(f"High CPU usage ({value:.1%})")
                elif 'memory_usage' in feature and value > 0.8:
                    factors.append(f"High memory usage ({value:.1%})")
                elif 'latency' in feature and value > 1000:
                    factors.append(f"High latency ({value:.0f}ms)")
                elif 'error_rate' in feature and value > 0.05:
                    factors.append(f"Elevated error rate ({value:.1%})")
                elif 'trend' in feature and abs(value) > 0.2:
                    direction = "increasing" if value > 0 else "decreasing"
                    factors.append(f"Significant {direction} trend in {feature.replace('_trend', '')}")
        
        return factors[:5]  # Top 5 factors
    
    def _generate_recommendations(self, probability: float, risk_level: RiskLevel,
                                 factors: List[str]) -> List[str]:
        """Generate recommendations based on prediction"""
        recommendations = []
        
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.extend([
                "Immediate attention required - monitor system closely",
                "Consider scaling resources proactively",
                "Enable offline fallback mode if available",
                "Review recent configuration changes"
            ])
            
            if any("cpu" in factor.lower() for factor in factors):
                recommendations.append("Optimize CPU-intensive operations")
            
            if any("memory" in factor.lower() for factor in factors):
                recommendations.append("Clear caches and restart memory-intensive services")
            
            if any("latency" in factor.lower() for factor in factors):
                recommendations.append("Check network connectivity and API response times")
            
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.extend([
                "Monitor system for signs of degradation",
                "Prepare contingency plans",
                "Review system metrics and trends"
            ])
        
        else:
            recommendations.extend([
                "Continue normal monitoring",
                "System performance appears stable"
            ])
        
        return recommendations
    
    def _predict_specific_metrics(self, data: pd.DataFrame, 
                                 horizon_hours: int) -> Dict[str, float]:
        """Predict specific performance metrics"""
        predictions = {}
        
        # Simple trend-based predictions for specific metrics
        metrics = ['latency_p95', 'accuracy_score', 'bleu_score', 'cpu_usage', 'memory_usage']
        
        for metric in metrics:
            if metric in data.columns and len(data) >= 4:
                # Use simple linear trend
                recent_values = data[metric].tail(4).values
                if len(recent_values) >= 2:
                    trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                    latest_value = recent_values[-1]
                    predicted_value = latest_value + (trend * horizon_hours)
                    predictions[metric] = float(predicted_value)
        
        return predictions
    
    def check_early_warnings(self, current_data: pd.DataFrame) -> List[EarlyWarning]:
        """Check for early warning conditions"""
        warnings = []
        
        if len(current_data) < 4:
            return warnings
        
        try:
            # Check trend-based warnings
            metrics = ['latency_p95', 'accuracy_score', 'cpu_usage', 'memory_usage']
            
            for metric in metrics:
                if metric not in current_data.columns:
                    continue
                
                recent_values = current_data[metric].tail(4).values
                if len(recent_values) >= 2:
                    # Calculate trend
                    trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                    current_value = recent_values[-1]
                    
                    # Check for concerning trends
                    if metric == 'latency_p95' and trend > 50:  # ms per hour
                        warnings.append(EarlyWarning(
                            timestamp=time.time(),
                            warning_type="latency_trend",
                            risk_level=RiskLevel.MEDIUM,
                            probability=min(abs(trend) / 100, 1.0),
                            time_to_degradation=max(1, (self.config.target_latency_p95 - current_value) / trend),
                            affected_metrics=[metric],
                            root_causes=[f"Increasing latency trend: {trend:.1f}ms/hour"],
                            preventive_actions=[
                                "Check API response times",
                                "Review system load",
                                "Consider resource scaling"
                            ],
                            confidence=0.7
                        ))
                    
                    elif metric == 'accuracy_score' and trend < -0.01:  # 1% per hour drop
                        warnings.append(EarlyWarning(
                            timestamp=time.time(),
                            warning_type="accuracy_degradation",
                            risk_level=RiskLevel.HIGH,
                            probability=min(abs(trend) * 100, 1.0),
                            time_to_degradation=max(1, (current_value - self.config.target_accuracy) / abs(trend)),
                            affected_metrics=[metric],
                            root_causes=[f"Declining accuracy trend: {trend:.3f}/hour"],
                            preventive_actions=[
                                "Run drift detection analysis",
                                "Check model performance",
                                "Review input data quality"
                            ],
                            confidence=0.8
                        ))
            
            # Check for anomaly-based warnings
            if len(current_data) >= 24:  # Need at least 24 hours for baseline
                baseline_data = current_data.tail(24)
                
                for metric in metrics:
                    if metric not in baseline_data.columns:
                        continue
                    
                    baseline_mean = baseline_data[metric].mean()
                    baseline_std = baseline_data[metric].std()
                    current_value = baseline_data[metric].iloc[-1]
                    
                    # Check for anomalies (2+ standard deviations)
                    if baseline_std > 0:
                        z_score = abs(current_value - baseline_mean) / baseline_std
                        
                        if z_score >= self.config.anomaly_warning_threshold:
                            severity = RiskLevel.HIGH if z_score >= 3 else RiskLevel.MEDIUM
                            
                            warnings.append(EarlyWarning(
                                timestamp=time.time(),
                                warning_type="anomaly_detection",
                                risk_level=severity,
                                probability=min(z_score / 3, 1.0),
                                time_to_degradation=None,
                                affected_metrics=[metric],
                                root_causes=[f"Anomalous {metric}: {z_score:.1f}σ from baseline"],
                                preventive_actions=[
                                    f"Investigate {metric} spike",
                                    "Check for system anomalies",
                                    "Review recent changes"
                                ],
                                confidence=0.9
                            ))
            
        except Exception as e:
            self.logger.error(f"Error checking early warnings: {e}")
        
        # Store warnings in history
        self.warning_history.extend(warnings)
        if len(self.warning_history) > 1000:
            self.warning_history = self.warning_history[-1000:]
        
        return warnings
    
    def get_prediction_summary(self) -> Dict[str, Any]:
        """Get summary of prediction system status"""
        with self._lock:
            summary = {
                'timestamp': time.time(),
                'is_trained': self.is_trained,
                'last_training_time': self.last_training_time,
                'model_performance': self.model_performance,
                'model_weights': self.model_weights,
                'feature_count': len(self.feature_names),
                'top_features': dict(list(self.feature_importance.items())[:10]) if self.feature_importance else {},
                'recent_warnings': len([w for w in self.warning_history if time.time() - w.timestamp < 3600]),
                'baseline_metrics': self.baseline_metrics
            }
            
            if self.last_training_time:
                summary['hours_since_training'] = (time.time() - self.last_training_time) / 3600
                summary['needs_retraining'] = summary['hours_since_training'] > self.config.retrain_interval_hours
            
            return summary


# Factory functions
def create_performance_predictor(config: Optional[PerformancePredictionConfig] = None) -> PerformancePredictor:
    """Create a performance predictor with optional configuration"""
    return PerformancePredictor(config)


# Global predictor instance
_global_predictor: Optional[PerformancePredictor] = None


def get_performance_predictor() -> PerformancePredictor:
    """Get global performance predictor instance"""
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = create_performance_predictor()
    return _global_predictor


def set_performance_predictor(predictor: PerformancePredictor):
    """Set global performance predictor instance"""
    global _global_predictor
    _global_predictor = predictor 