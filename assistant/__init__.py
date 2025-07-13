"""
Jarvis-Pipecat Voice Assistant

A real-time voice assistant built with Pipecat framework that provides:
- Real-time speech-to-text and text-to-speech
- Multi-model LLM routing (fast vs deep responses)
- Long-term memory via vector database
- Screen awareness through OCR
- Code agent integration
- Offline fallback capabilities
"""

__version__ = "0.1.0"
__author__ = "Ian Greenberg"

# Core components will be imported here as they're implemented
from .audio import AudioManager, AudioConfig, create_audio_manager
from .stt import WhisperSTTService, STTConfig, STTResult, create_whisper_stt_service
from .tts import OpenAITTSService, TTSConfig, TTSResult, create_openai_tts_service
from .pipeline import VoiceAssistantPipeline, PipelineConfig, PipelineState, create_pipeline_from_config
from .monitoring import PerformanceMonitor, PipelineStage, get_monitor, set_monitor
from .dashboard import ConsoleDashboard, PerformanceReporter, create_dashboard, create_reporter
from .router_block import RouterBlock, RouterState, RouterMetrics, create_router_block
from .async_processor import AsyncProcessor, ProcessingConfig, ProcessingState
from .fallback_manager import (
    FallbackManager, CircuitBreaker, RetryHandler, ErrorMessageManager,
    FallbackConfig, CircuitBreakerConfig, RetryConfig, ServiceState, RetryStrategy,
    ServiceUnavailableError, AllServicesFailed, create_fallback_manager, create_circuit_breaker
)
from .memory import (
    MemoryManager, MemoryConfig, ConversationMemory, ScreenMemory, MemoryRetriever,
    create_memory_manager, get_default_memory_config
)
from .screen_watcher import (
    ScreenWatcher, ScreenWatcherConfig, ScreenCapture, WindowInfo, 
    ActiveWindowDetector, ScreenImageProcessor, OCRProcessor,
    create_screen_watcher, get_default_screen_config
)
from .kimi_agent import (
    KimiK2Agent, KimiConfig, CodeContext, CodeResponse, 
    CodeContextExtractor, DiffGenerator, create_kimi_agent
)
from .offline_system import (
    OfflineSystem, OfflineConfig, NetworkDetector, ModelManager, MemoryMonitor,
    OfflineSTTService, OfflineTTSService, OfflineLLMService,
    ConnectivityStatus, ModelStatus, create_offline_system
)
from .error_handling import (
    ErrorCategory, ErrorContext, VoiceAIException, STTException, LLMException, TTSException,
    AudioException, CircuitBreakerOpenError, AllTiersFailedError, BackoffConfig,
    AdvancedBackoff, retry_with_backoff, CircuitState, HealthMetrics, CircuitBreakerConfig,
    ModernCircuitBreaker, ServiceTier, ServiceCapability, GracefulDegradationManager
)
from .structured_logging import (
    VoiceAILogContext, VoiceAILogger, StructuredFormatter, ColoredConsoleFormatter,
    PerformanceMetricsFilter, voice_ai_request_context, log_audio_processing,
    log_model_inference, log_network_request, get_voice_ai_logger, set_voice_ai_logger
)
from .health_monitoring import (
    HealthStatus, ServiceHealthMetrics, SystemHealthMetrics, HealthChecker,
    SystemHealthMonitor, get_health_monitor, set_health_monitor, health_monitored_operation,
    stt_health_check, llm_health_check, tts_health_check, offline_system_health_check
)
from .config_manager import (
    ConfigManager, SovereignConfig, APIConfig, AudioConfig, STTConfig, TTSConfig,
    LLMConfig, MemoryConfig, ScreenConfig, CodeAgentConfig, SecurityConfig,
    MonitoringConfig, DevelopmentConfig, EnvironmentType, ConfigurationError,
    get_config_manager, get_config, reload_config, create_config_template
)
from .metrics_collector import (
    MetricsCollector, MetricType, ComponentType, LatencyMetrics, AccuracyMetrics,
    ResourceMetrics, ThroughputMetrics, get_metrics_collector, set_metrics_collector
)
from .dashboard_server import (
    DashboardServer, MetricAggregator, create_dashboard_server
)

# Predictive Analytics and Alerting System (Task 10.4)
from .drift_detection import (
    DriftDetector, DriftConfig, DriftAlert, DriftType, DriftSeverity,
    get_drift_detector, set_drift_detector, create_drift_detector
)
from .resource_forecasting import (
    ResourceForecaster, ResourceForecastConfig, ForecastResult, AnomalyResult,
    ForecastType, ResourceType, get_resource_forecaster, set_resource_forecaster,
    create_resource_forecaster
)
from .alerting_system import (
    AlertingSystem, AlertingConfig, Alert, AlertSeverity, AlertChannel,
    AlertDestination, AlertCorrelationRule, get_alerting_system, set_alerting_system,
    create_alerting_system, create_email_destination, create_webhook_destination, create_slack_destination
)
from .performance_prediction import (
    PerformancePredictor, PerformancePredictionConfig, PerformancePrediction,
    EarlyWarning, RiskLevel, PredictionHorizon, get_performance_predictor,
    set_performance_predictor, create_performance_predictor
)
from .predictive_analytics import (
    PredictiveAnalyticsFramework, PredictiveAnalyticsConfig, AnalyticsReport,
    AnalyticsMode, get_predictive_analytics, set_predictive_analytics,
    create_predictive_analytics
)

# Predictive Integration System (Task 10.5)
from .unified_health_reporter import (
    UnifiedHealthReporter, UnifiedHealthStatus, UnifiedHealthReport, RiskFactor, RecommendedAction,
    get_unified_health_reporter, set_unified_health_reporter, create_unified_health_reporter
)
from .predictive_error_enricher import (
    PredictiveErrorEnricher, EnrichedErrorContext, ErrorPattern, SimilarIncident,
    PredictiveRootCause, RecoveryRecommendation,
    get_predictive_error_enricher, set_predictive_error_enricher, create_predictive_error_enricher
)
from .predictive_optimization_engine import (
    PredictiveOptimizationEngine, OptimizationType, OptimizationPriority, OptimizationStatus,
    OptimizationRule, OptimizationAction, OptimizationResult,
    get_predictive_optimization_engine, set_predictive_optimization_engine, create_predictive_optimization_engine
)
from .predictive_config_manager import (
    PredictiveConfigManager, ConfigAdjustmentType, ConfigScope, ConfigChangeStatus,
    ConfigAdjustmentRule, ConfigChange, ConfigurationProfile,
    get_predictive_config_manager, set_predictive_config_manager, create_predictive_config_manager
)

# from .core import JarvisAssistant
# from .router import LLMRouter
# from .code_agent import CodeAgent 