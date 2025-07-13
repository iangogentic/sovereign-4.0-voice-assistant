"""
Resource Usage Forecasting System

This module implements advanced resource usage forecasting for AI voice assistants,
combining multiple approaches:
- Facebook Prophet for trend and seasonality analysis
- LSTM neural networks for complex pattern recognition
- Isolation Forest for anomaly detection
- CUSUM for change point detection
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
import warnings

# Suppress Prophet warnings
warnings.filterwarnings('ignore', message='.*The package urllib3.contrib.pyopenssl.*')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    Prophet = None

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import joblib


class ForecastType(Enum):
    """Types of forecasting models"""
    PROPHET = "prophet"
    LSTM = "lstm"
    ENSEMBLE = "ensemble"


class ResourceType(Enum):
    """Types of resources to forecast"""
    CPU = "cpu_usage"
    MEMORY = "memory_usage"
    GPU = "gpu_usage"
    NETWORK = "network_io"
    DISK = "disk_io"
    LATENCY = "latency"
    THROUGHPUT = "throughput"


@dataclass
class ForecastResult:
    """Resource usage forecast result"""
    timestamp: float
    resource_type: ResourceType
    forecast_horizon: int  # hours
    predicted_values: List[float]
    confidence_lower: List[float]
    confidence_upper: List[float]
    forecast_timestamps: List[float]
    model_type: ForecastType
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)


@dataclass
class AnomalyResult:
    """Resource anomaly detection result"""
    timestamp: float
    resource_type: ResourceType
    is_anomaly: bool
    anomaly_score: float
    threshold: float
    severity: str
    description: str


@dataclass
class ResourceForecastConfig:
    """Configuration for resource forecasting"""
    # Prophet settings
    prophet_daily_seasonality: bool = True
    prophet_weekly_seasonality: bool = True
    prophet_yearly_seasonality: bool = False
    prophet_changepoint_prior_scale: float = 0.05
    prophet_seasonality_prior_scale: float = 10.0
    
    # LSTM settings
    lstm_sequence_length: int = 60  # 60 time steps
    lstm_features: int = 4  # CPU, Memory, GPU, Network
    lstm_hidden_units: int = 50
    lstm_dropout: float = 0.2
    lstm_epochs: int = 50
    lstm_batch_size: int = 32
    
    # Forecasting settings
    default_forecast_horizon: int = 24  # hours
    min_training_samples: int = 168  # 1 week of hourly data
    retrain_interval: int = 24  # retrain every 24 hours
    
    # Anomaly detection settings
    anomaly_contamination: float = 0.1
    anomaly_n_estimators: int = 100
    
    # Data processing
    data_frequency: str = 'H'  # hourly
    smoothing_window: int = 3
    
    # Persistence settings
    save_models: bool = True
    model_dir: str = ".taskmaster/forecast_models"


class ResourceForecaster:
    """
    Advanced resource usage forecasting system
    
    Combines multiple forecasting approaches for robust prediction:
    - Prophet for capturing trends and seasonality
    - LSTM networks for complex non-linear patterns
    - Ensemble methods for improved accuracy
    """
    
    def __init__(self, config: Optional[ResourceForecastConfig] = None):
        self.config = config or ResourceForecastConfig()
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
        
        # Model storage
        self.prophet_models = {}
        self.lstm_models = {}
        self.scalers = {}
        self.anomaly_detectors = {}
        
        # Training data storage
        self.training_data = {}
        self.last_training_time = {}
        
        # Model performance tracking
        self.model_performance = {}
        
        # Initialize model directory
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
        
        # Load Prophet models
        for model_file in model_path.glob("prophet_*.json"):
            try:
                if PROPHET_AVAILABLE:
                    resource_type = model_file.stem.replace("prophet_", "")
                    model = Prophet()
                    with open(model_file, 'r') as f:
                        model = model.from_json(f.read())
                    self.prophet_models[resource_type] = model
                    self.logger.info(f"Loaded Prophet model for {resource_type}")
            except Exception as e:
                self.logger.warning(f"Failed to load Prophet model {model_file}: {e}")
        
        # Load LSTM models
        for model_file in model_path.glob("lstm_*.h5"):
            try:
                if TENSORFLOW_AVAILABLE:
                    resource_type = model_file.stem.replace("lstm_", "")
                    model = tf.keras.models.load_model(model_file)
                    self.lstm_models[resource_type] = model
                    self.logger.info(f"Loaded LSTM model for {resource_type}")
            except Exception as e:
                self.logger.warning(f"Failed to load LSTM model {model_file}: {e}")
        
        # Load scalers
        for scaler_file in model_path.glob("scaler_*.joblib"):
            try:
                resource_type = scaler_file.stem.replace("scaler_", "")
                scaler = joblib.load(scaler_file)
                self.scalers[resource_type] = scaler
                self.logger.info(f"Loaded scaler for {resource_type}")
            except Exception as e:
                self.logger.warning(f"Failed to load scaler {scaler_file}: {e}")
        
        # Load anomaly detectors
        for detector_file in model_path.glob("anomaly_*.joblib"):
            try:
                resource_type = detector_file.stem.replace("anomaly_", "")
                detector = joblib.load(detector_file)
                self.anomaly_detectors[resource_type] = detector
                self.logger.info(f"Loaded anomaly detector for {resource_type}")
            except Exception as e:
                self.logger.warning(f"Failed to load anomaly detector {detector_file}: {e}")
    
    def _save_models(self, resource_type: str):
        """Save models to disk"""
        if not self.config.save_models:
            return
            
        model_path = Path(self.config.model_dir)
        
        try:
            # Save Prophet model
            if resource_type in self.prophet_models and PROPHET_AVAILABLE:
                prophet_file = model_path / f"prophet_{resource_type}.json"
                with open(prophet_file, 'w') as f:
                    f.write(self.prophet_models[resource_type].to_json())
            
            # Save LSTM model
            if resource_type in self.lstm_models and TENSORFLOW_AVAILABLE:
                lstm_file = model_path / f"lstm_{resource_type}.h5"
                self.lstm_models[resource_type].save(lstm_file)
            
            # Save scaler
            if resource_type in self.scalers:
                scaler_file = model_path / f"scaler_{resource_type}.joblib"
                joblib.dump(self.scalers[resource_type], scaler_file)
            
            # Save anomaly detector
            if resource_type in self.anomaly_detectors:
                detector_file = model_path / f"anomaly_{resource_type}.joblib"
                joblib.dump(self.anomaly_detectors[resource_type], detector_file)
            
            self.logger.info(f"Saved models for {resource_type}")
            
        except Exception as e:
            self.logger.error(f"Error saving models for {resource_type}: {e}")
    
    def _prepare_prophet_data(self, data: pd.DataFrame, 
                             resource_type: str) -> pd.DataFrame:
        """Prepare data for Prophet forecasting"""
        df = data.copy()
        
        # Prophet requires 'ds' (datestamp) and 'y' (value) columns
        if 'timestamp' in df.columns:
            df['ds'] = pd.to_datetime(df['timestamp'])
        else:
            df['ds'] = df.index
        
        df['y'] = df[resource_type]
        
        # Remove any missing values
        df = df.dropna(subset=['ds', 'y'])
        
        # Sort by date
        df = df.sort_values('ds')
        
        return df[['ds', 'y']]
    
    def _create_lstm_sequences(self, data: np.ndarray, 
                              sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i])
        
        return np.array(X), np.array(y)
    
    def _build_lstm_model(self, input_shape: Tuple[int, int]) -> Any:
        """Build LSTM model for time series forecasting"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM forecasting")
        
        model = Sequential([
            LSTM(self.config.lstm_hidden_units, 
                 return_sequences=True, 
                 input_shape=input_shape),
            Dropout(self.config.lstm_dropout),
            
            LSTM(self.config.lstm_hidden_units, 
                 return_sequences=True),
            Dropout(self.config.lstm_dropout),
            
            LSTM(self.config.lstm_hidden_units),
            Dropout(self.config.lstm_dropout),
            
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_prophet_model(self, data: pd.DataFrame, 
                           resource_type: str) -> bool:
        """Train Prophet model for resource forecasting"""
        if not PROPHET_AVAILABLE:
            self.logger.warning("Prophet not available, skipping Prophet training")
            return False
        
        try:
            with self._lock:
                # Prepare data
                prophet_data = self._prepare_prophet_data(data, resource_type)
                
                if len(prophet_data) < self.config.min_training_samples:
                    self.logger.warning(
                        f"Insufficient data for Prophet training: "
                        f"{len(prophet_data)} < {self.config.min_training_samples}"
                    )
                    return False
                
                # Create and configure Prophet model
                model = Prophet(
                    daily_seasonality=self.config.prophet_daily_seasonality,
                    weekly_seasonality=self.config.prophet_weekly_seasonality,
                    yearly_seasonality=self.config.prophet_yearly_seasonality,
                    changepoint_prior_scale=self.config.prophet_changepoint_prior_scale,
                    seasonality_prior_scale=self.config.prophet_seasonality_prior_scale
                )
                
                # Fit model
                model.fit(prophet_data)
                
                # Store model
                self.prophet_models[resource_type] = model
                self.last_training_time[f"prophet_{resource_type}"] = time.time()
                
                self.logger.info(f"Successfully trained Prophet model for {resource_type}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error training Prophet model for {resource_type}: {e}")
            return False
    
    def train_lstm_model(self, data: pd.DataFrame, 
                        resource_type: str) -> bool:
        """Train LSTM model for resource forecasting"""
        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("TensorFlow not available, skipping LSTM training")
            return False
        
        try:
            with self._lock:
                # Prepare data
                if len(data) < self.config.min_training_samples:
                    self.logger.warning(
                        f"Insufficient data for LSTM training: "
                        f"{len(data)} < {self.config.min_training_samples}"
                    )
                    return False
                
                # Extract values and scale
                values = data[resource_type].values.reshape(-1, 1)
                
                # Create scaler if not exists
                if resource_type not in self.scalers:
                    self.scalers[resource_type] = MinMaxScaler()
                
                scaled_data = self.scalers[resource_type].fit_transform(values)
                
                # Create sequences
                X, y = self._create_lstm_sequences(
                    scaled_data.flatten(), 
                    self.config.lstm_sequence_length
                )
                
                if len(X) == 0:
                    self.logger.warning(f"No sequences created for LSTM training")
                    return False
                
                # Reshape for LSTM
                X = X.reshape((X.shape[0], X.shape[1], 1))
                
                # Build model
                model = self._build_lstm_model((self.config.lstm_sequence_length, 1))
                
                # Train model
                model.fit(
                    X, y,
                    epochs=self.config.lstm_epochs,
                    batch_size=self.config.lstm_batch_size,
                    verbose=0,
                    validation_split=0.2
                )
                
                # Store model
                self.lstm_models[resource_type] = model
                self.last_training_time[f"lstm_{resource_type}"] = time.time()
                
                self.logger.info(f"Successfully trained LSTM model for {resource_type}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error training LSTM model for {resource_type}: {e}")
            return False
    
    def train_anomaly_detector(self, data: pd.DataFrame, 
                              resource_type: str) -> bool:
        """Train anomaly detector for resource usage"""
        try:
            with self._lock:
                if len(data) < self.config.min_training_samples:
                    self.logger.warning(
                        f"Insufficient data for anomaly detector training: "
                        f"{len(data)} < {self.config.min_training_samples}"
                    )
                    return False
                
                # Prepare features
                features = data[resource_type].values.reshape(-1, 1)
                
                # Create and train isolation forest
                detector = IsolationForest(
                    contamination=self.config.anomaly_contamination,
                    n_estimators=self.config.anomaly_n_estimators,
                    random_state=42
                )
                
                detector.fit(features)
                
                # Store detector
                self.anomaly_detectors[resource_type] = detector
                
                self.logger.info(f"Successfully trained anomaly detector for {resource_type}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error training anomaly detector for {resource_type}: {e}")
            return False
    
    def train_models(self, data: pd.DataFrame, 
                    resource_types: Optional[List[str]] = None) -> Dict[str, bool]:
        """Train all forecasting models for specified resource types"""
        if resource_types is None:
            resource_types = [rt.value for rt in ResourceType]
        
        results = {}
        
        for resource_type in resource_types:
            if resource_type not in data.columns:
                self.logger.warning(f"Resource type {resource_type} not found in data")
                results[resource_type] = False
                continue
            
            # Store training data
            self.training_data[resource_type] = data.copy()
            
            # Train Prophet model
            prophet_success = self.train_prophet_model(data, resource_type)
            
            # Train LSTM model
            lstm_success = self.train_lstm_model(data, resource_type)
            
            # Train anomaly detector
            anomaly_success = self.train_anomaly_detector(data, resource_type)
            
            # Save models
            self._save_models(resource_type)
            
            results[resource_type] = prophet_success or lstm_success
            
            self.logger.info(
                f"Training complete for {resource_type}: "
                f"Prophet={prophet_success}, LSTM={lstm_success}, "
                f"Anomaly={anomaly_success}"
            )
        
        return results
    
    def forecast_prophet(self, resource_type: str, 
                        periods: int) -> Optional[ForecastResult]:
        """Generate forecast using Prophet model"""
        if not PROPHET_AVAILABLE or resource_type not in self.prophet_models:
            return None
        
        try:
            model = self.prophet_models[resource_type]
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=periods, freq='H')
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Extract forecast values (only future predictions)
            future_forecast = forecast.tail(periods)
            
            return ForecastResult(
                timestamp=time.time(),
                resource_type=ResourceType(resource_type),
                forecast_horizon=periods,
                predicted_values=future_forecast['yhat'].tolist(),
                confidence_lower=future_forecast['yhat_lower'].tolist(),
                confidence_upper=future_forecast['yhat_upper'].tolist(),
                forecast_timestamps=future_forecast['ds'].astype(int).tolist(),
                model_type=ForecastType.PROPHET
            )
            
        except Exception as e:
            self.logger.error(f"Error in Prophet forecast for {resource_type}: {e}")
            return None
    
    def forecast_lstm(self, resource_type: str, 
                     periods: int) -> Optional[ForecastResult]:
        """Generate forecast using LSTM model"""
        if not TENSORFLOW_AVAILABLE or resource_type not in self.lstm_models:
            return None
        
        try:
            model = self.lstm_models[resource_type]
            scaler = self.scalers.get(resource_type)
            
            if scaler is None:
                self.logger.error(f"No scaler found for {resource_type}")
                return None
            
            # Get recent data for initialization
            if resource_type not in self.training_data:
                self.logger.error(f"No training data found for {resource_type}")
                return None
            
            recent_data = self.training_data[resource_type][resource_type].tail(
                self.config.lstm_sequence_length
            ).values
            
            # Scale recent data
            scaled_recent = scaler.transform(recent_data.reshape(-1, 1)).flatten()
            
            # Generate predictions
            predictions = []
            current_sequence = scaled_recent.copy()
            
            for _ in range(periods):
                # Reshape for prediction
                X = current_sequence[-self.config.lstm_sequence_length:].reshape(1, -1, 1)
                
                # Predict next value
                pred_scaled = model.predict(X, verbose=0)[0, 0]
                predictions.append(pred_scaled)
                
                # Update sequence
                current_sequence = np.append(current_sequence[1:], pred_scaled)
            
            # Inverse transform predictions
            predictions_scaled = np.array(predictions).reshape(-1, 1)
            predictions_original = scaler.inverse_transform(predictions_scaled).flatten()
            
            # Generate timestamps
            current_time = time.time()
            timestamps = [current_time + (i + 1) * 3600 for i in range(periods)]
            
            return ForecastResult(
                timestamp=current_time,
                resource_type=ResourceType(resource_type),
                forecast_horizon=periods,
                predicted_values=predictions_original.tolist(),
                confidence_lower=[p * 0.95 for p in predictions_original],  # Simple confidence bands
                confidence_upper=[p * 1.05 for p in predictions_original],
                forecast_timestamps=timestamps,
                model_type=ForecastType.LSTM
            )
            
        except Exception as e:
            self.logger.error(f"Error in LSTM forecast for {resource_type}: {e}")
            return None
    
    def forecast_ensemble(self, resource_type: str, 
                         periods: int) -> Optional[ForecastResult]:
        """Generate ensemble forecast combining Prophet and LSTM"""
        prophet_forecast = self.forecast_prophet(resource_type, periods)
        lstm_forecast = self.forecast_lstm(resource_type, periods)
        
        if prophet_forecast is None and lstm_forecast is None:
            return None
        
        # If only one model available, return that
        if prophet_forecast is None:
            return lstm_forecast
        if lstm_forecast is None:
            return prophet_forecast
        
        # Combine forecasts with weighted average
        prophet_weight = 0.6  # Prophet generally better for trends/seasonality
        lstm_weight = 0.4     # LSTM better for complex patterns
        
        combined_predictions = [
            prophet_weight * p + lstm_weight * l
            for p, l in zip(prophet_forecast.predicted_values, 
                          lstm_forecast.predicted_values)
        ]
        
        # Conservative confidence bounds
        combined_lower = [
            min(p, l) for p, l in zip(prophet_forecast.confidence_lower,
                                    lstm_forecast.confidence_lower)
        ]
        combined_upper = [
            max(p, l) for p, l in zip(prophet_forecast.confidence_upper,
                                    lstm_forecast.confidence_upper)
        ]
        
        return ForecastResult(
            timestamp=time.time(),
            resource_type=ResourceType(resource_type),
            forecast_horizon=periods,
            predicted_values=combined_predictions,
            confidence_lower=combined_lower,
            confidence_upper=combined_upper,
            forecast_timestamps=prophet_forecast.forecast_timestamps,
            model_type=ForecastType.ENSEMBLE,
            accuracy_metrics={
                'prophet_weight': prophet_weight,
                'lstm_weight': lstm_weight
            }
        )
    
    def detect_anomalies(self, data: pd.DataFrame, 
                        resource_type: str) -> List[AnomalyResult]:
        """Detect anomalies in resource usage"""
        anomalies = []
        
        if resource_type not in self.anomaly_detectors:
            self.logger.warning(f"No anomaly detector for {resource_type}")
            return anomalies
        
        try:
            detector = self.anomaly_detectors[resource_type]
            
            # Prepare features
            features = data[resource_type].values.reshape(-1, 1)
            
            # Detect anomalies
            anomaly_predictions = detector.predict(features)
            anomaly_scores = detector.decision_function(features)
            
            # Calculate threshold (10th percentile of scores)
            threshold = np.percentile(anomaly_scores, 10)
            
            # Process results
            for i, (pred, score) in enumerate(zip(anomaly_predictions, anomaly_scores)):
                if pred == -1:  # Anomaly detected
                    severity = "high" if score < threshold * 0.5 else "medium"
                    
                    anomalies.append(AnomalyResult(
                        timestamp=data.index[i] if hasattr(data.index[i], 'timestamp') else time.time(),
                        resource_type=ResourceType(resource_type),
                        is_anomaly=True,
                        anomaly_score=score,
                        threshold=threshold,
                        severity=severity,
                        description=f"Anomalous {resource_type} detected: score {score:.3f}"
                    ))
        
        except Exception as e:
            self.logger.error(f"Error detecting anomalies for {resource_type}: {e}")
        
        return anomalies
    
    def forecast(self, resource_type: str, 
                periods: Optional[int] = None,
                model_type: ForecastType = ForecastType.ENSEMBLE) -> Optional[ForecastResult]:
        """
        Generate resource usage forecast
        
        Args:
            resource_type: Type of resource to forecast
            periods: Number of periods to forecast (default: config value)
            model_type: Type of model to use for forecasting
            
        Returns:
            Forecast result or None if forecasting fails
        """
        if periods is None:
            periods = self.config.default_forecast_horizon
        
        with self._lock:
            if model_type == ForecastType.PROPHET:
                return self.forecast_prophet(resource_type, periods)
            elif model_type == ForecastType.LSTM:
                return self.forecast_lstm(resource_type, periods)
            elif model_type == ForecastType.ENSEMBLE:
                return self.forecast_ensemble(resource_type, periods)
            else:
                self.logger.error(f"Unknown model type: {model_type}")
                return None
    
    def get_forecast_summary(self) -> Dict[str, Any]:
        """Get summary of forecasting capabilities and status"""
        with self._lock:
            summary = {
                'timestamp': time.time(),
                'models_available': {
                    'prophet': len(self.prophet_models),
                    'lstm': len(self.lstm_models),
                    'anomaly_detectors': len(self.anomaly_detectors)
                },
                'resource_types': [],
                'last_training': {}
            }
            
            # Get all resource types with models
            all_resources = set()
            all_resources.update(self.prophet_models.keys())
            all_resources.update(self.lstm_models.keys())
            all_resources.update(self.anomaly_detectors.keys())
            
            summary['resource_types'] = list(all_resources)
            
            # Get last training times
            for key, timestamp in self.last_training_time.items():
                summary['last_training'][key] = {
                    'timestamp': timestamp,
                    'hours_ago': (time.time() - timestamp) / 3600
                }
            
            return summary


# Factory function for easy instantiation
def create_resource_forecaster(config: Optional[ResourceForecastConfig] = None) -> ResourceForecaster:
    """Create a resource forecaster with optional configuration"""
    return ResourceForecaster(config)


# Global forecaster instance
_global_forecaster: Optional[ResourceForecaster] = None


def get_resource_forecaster() -> ResourceForecaster:
    """Get global resource forecaster instance"""
    global _global_forecaster
    if _global_forecaster is None:
        _global_forecaster = create_resource_forecaster()
    return _global_forecaster


def set_resource_forecaster(forecaster: ResourceForecaster):
    """Set global resource forecaster instance"""
    global _global_forecaster
    _global_forecaster = forecaster 