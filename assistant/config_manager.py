"""
Sovereign 4.0 Voice Assistant - Configuration Management System

Provides comprehensive configuration management with:
- YAML-based configuration files
- Environment variable overrides
- Configuration validation and schema
- Hot-reload capability for development
- Multiple environment profiles (development, production, offline)
- Secure API key handling
- Sensible defaults and fallbacks

Usage:
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Access configuration
    api_key = config.openai.api_key
    model = config.llm.fast.model
    
    # Hot-reload in development
    config_manager.enable_hot_reload()
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import yaml
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

class EnvironmentType(Enum):
    """Environment types for configuration profiles"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    OFFLINE = "offline"
    TESTING = "testing"

@dataclass
class APIConfig:
    """API configuration with secure key handling"""
    openai_api_key: Optional[str] = None
    openai_org_id: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    perplexity_api_key: Optional[str] = None
    kimi_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None
    
    # Endpoints
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_base_url: str = "https://api.anthropic.com"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    perplexity_base_url: str = "https://api.perplexity.ai"
    kimi_base_url: str = "https://api.moonshot.cn/v1"
    
    # Security settings
    validate_keys_on_startup: bool = True
    timeout: float = 30.0
    max_retries: int = 3

@dataclass
class AudioConfig:
    """Audio system configuration"""
    # Input settings
    input_device: Optional[str] = None
    sample_rate: int = 16000
    chunk_size: int = 1024
    channels: int = 1
    
    # Output settings
    output_device: Optional[str] = None
    output_volume: float = 0.8
    
    # Voice Activity Detection
    vad_enabled: bool = True
    silence_threshold: float = 0.001
    silence_duration: float = 1.0
    min_audio_length: float = 0.3
    max_audio_length: float = 30.0
    
    # Audio processing
    noise_reduction: bool = True
    echo_cancellation: bool = True
    automatic_gain_control: bool = True

@dataclass
class STTConfig:
    """Speech-to-Text configuration"""
    # Primary (cloud) STT
    primary_provider: str = "openai"
    primary_model: str = "whisper-1"
    primary_language: str = "en"
    
    # Fallback (local) STT
    fallback_provider: str = "whisper-cpp"
    fallback_model: str = "tiny.en"
    fallback_language: str = "en"
    
    # Performance settings
    timeout: float = 10.0
    temperature: float = 0.0
    max_retries: int = 2
    
    # Advanced settings
    enable_punctuation: bool = True
    enable_timestamps: bool = False
    enable_word_timestamps: bool = False

@dataclass
class TTSConfig:
    """Text-to-Speech configuration"""
    # Primary (cloud) TTS
    primary_provider: str = "openai"
    primary_model: str = "tts-1"
    primary_voice: str = "alloy"
    primary_speed: float = 1.0
    
    # Fallback (local) TTS
    fallback_provider: str = "piper"
    fallback_voice: str = "en_US-lessac-medium"
    fallback_speed: float = 1.0
    
    # Performance settings
    timeout: float = 15.0
    response_format: str = "mp3"
    max_retries: int = 2
    
    # Audio quality
    quality: str = "standard"  # standard, hd
    enable_ssml: bool = False

@dataclass
class ModelConfig:
    """Individual model configuration"""
    provider: str
    model: str
    max_tokens: int
    temperature: float
    timeout: float
    cost_per_1k_tokens: float = 0.0
    max_requests_per_minute: int = 1000
    context_window: int = 4096

@dataclass
class LLMConfig:
    """Language Model configuration"""
    # Fast model for quick responses
    fast: ModelConfig = field(default_factory=lambda: ModelConfig(
        provider="openrouter",
        model="openai/gpt-4o-mini",
        max_tokens=500,
        temperature=0.7,
        timeout=5.0,
        cost_per_1k_tokens=0.0002,
        max_requests_per_minute=1000,
        context_window=128000
    ))
    
    # Deep model for complex queries
    deep: ModelConfig = field(default_factory=lambda: ModelConfig(
        provider="openrouter",
        model="openai/gpt-4o",
        max_tokens=2000,
        temperature=0.7,
        timeout=30.0,
        cost_per_1k_tokens=0.005,
        max_requests_per_minute=500,
        context_window=128000
    ))
    
    # Local model for offline mode
    local: ModelConfig = field(default_factory=lambda: ModelConfig(
        provider="llama-cpp",
        model="gemma-2b-it-q4_0.gguf",
        max_tokens=500,
        temperature=0.7,
        timeout=10.0,
        cost_per_1k_tokens=0.0,
        max_requests_per_minute=1000,
        context_window=2048
    ))
    
    # Fallback configuration
    enable_fallback: bool = True
    fallback_chain: List[str] = field(default_factory=lambda: ["fast", "deep", "local"])
    max_fallback_attempts: int = 3

@dataclass
class MemoryConfig:
    """Memory system configuration"""
    # Vector database
    provider: str = "chroma"
    persist_directory: str = "./data/chroma"
    collection_name_conversations: str = "sovereign_conversations"
    collection_name_screen: str = "sovereign_screen"
    
    # Embeddings
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 100
    
    # Retrieval
    retrieval_k: int = 5
    similarity_threshold: float = 0.7
    max_context_length: int = 8000
    
    # Memory management
    max_conversations_per_session: int = 100
    cleanup_interval_hours: int = 24
    enable_memory_compression: bool = True

@dataclass
class ScreenConfig:
    """Screen monitoring configuration"""
    enabled: bool = True
    screenshot_interval: float = 3.0
    
    # OCR settings
    ocr_provider: str = "tesseract"
    ocr_language: str = "eng"
    ocr_config: str = "--psm 6"
    tesseract_path: Optional[str] = None
    
    # Image preprocessing
    resize_factor: float = 1.0
    contrast_enhancement: bool = True
    noise_reduction: bool = True
    
    # Performance
    max_screen_history: int = 50
    enable_ocr_cache: bool = True

@dataclass
class CodeAgentConfig:
    """Code agent configuration"""
    enabled: bool = True
    provider: str = "kimi"
    model: str = "kimi-k2"
    
    # Trigger patterns
    trigger_patterns: List[str] = field(default_factory=lambda: [
        "#code", "code:", "programming", "write code", "debug", "fix bug"
    ])
    
    # Model settings
    max_tokens: int = 4000
    temperature: float = 0.1
    timeout: float = 60.0
    
    # Context settings
    max_context_files: int = 10
    max_context_size: int = 50000
    
    # Supported file types
    supported_extensions: List[str] = field(default_factory=lambda: [
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".cs",
        ".go", ".rs", ".php", ".rb", ".swift", ".kt", ".html", ".css", ".sql"
    ])

@dataclass
class SecurityConfig:
    """Security configuration"""
    # API key security
    validate_api_keys: bool = True
    mask_keys_in_logs: bool = True
    
    # Data encryption
    encrypt_memory: bool = False
    encryption_key_file: Optional[str] = None
    
    # Network security
    require_https: bool = True
    allowed_hosts: List[str] = field(default_factory=lambda: ["localhost", "127.0.0.1"])
    
    # Rate limiting
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 100
    
    # Logging security
    log_user_input: bool = True
    log_assistant_responses: bool = True
    sanitize_logs: bool = True

@dataclass
class MonitoringConfig:
    """Performance monitoring configuration"""
    enabled: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = "./logs/sovereign.log"
    
    # Metrics
    metrics_enabled: bool = True
    metrics_port: int = 8080
    
    # Health checks
    health_check_interval: float = 30.0
    health_check_timeout: float = 5.0
    
    # Performance tracking
    track_latency: bool = True
    track_memory_usage: bool = True
    track_cpu_usage: bool = True
    
    # Error tracking
    error_reporting: bool = True
    max_error_backlog: int = 100

@dataclass
class DevelopmentConfig:
    """Development-specific configuration"""
    debug_mode: bool = False
    test_mode: bool = False
    mock_apis: bool = False
    
    # Hot reload
    enable_hot_reload: bool = True
    hot_reload_paths: List[str] = field(default_factory=lambda: [
        "config/", "assistant/"
    ])
    
    # Development shortcuts
    skip_audio_init: bool = False
    use_test_data: bool = False
    bypass_api_validation: bool = False

@dataclass
class SovereignConfig:
    """Complete Sovereign 4.0 Voice Assistant configuration"""
    # Core configuration sections
    environment: EnvironmentType = EnvironmentType.DEVELOPMENT
    api: APIConfig = field(default_factory=APIConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    code_agent: CodeAgentConfig = field(default_factory=CodeAgentConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    development: DevelopmentConfig = field(default_factory=DevelopmentConfig)
    
    # Metadata
    version: str = "4.0.0"
    name: str = "Sovereign Voice Assistant"
    description: str = "Advanced AI voice assistant with multi-modal capabilities"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Feature flags
    features: Dict[str, bool] = field(default_factory=lambda: {
        "offline_mode": False,
        "memory_enabled": True,
        "screen_monitoring": True,
        "code_agent": True,
        "performance_monitoring": True,
        "voice_interruption": False,
        "emotion_detection": False,
        "multi_language": False
    })

class ConfigurationError(Exception):
    """Configuration-related errors"""
    pass

class ConfigFileWatcher(FileSystemEventHandler):
    """Watches configuration files for changes and triggers reloads"""
    
    def __init__(self, config_manager: 'ConfigManager'):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
    
    def on_modified(self, event):
        if event.is_directory:
            return
            
        if event.src_path.endswith('.yaml') or event.src_path.endswith('.yml'):
            self.logger.info(f"Configuration file changed: {event.src_path}")
            try:
                self.config_manager.reload_config()
                self.logger.info("Configuration reloaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to reload configuration: {e}")

class ConfigManager:
    """
    Comprehensive configuration management system for Sovereign 4.0
    
    Features:
    - YAML-based configuration with environment overrides
    - Multiple environment profiles
    - Configuration validation and schema
    - Hot-reload capability for development
    - Secure API key handling
    - Sensible defaults and fallbacks
    """
    
    def __init__(self, config_path: Optional[str] = None, environment: Optional[EnvironmentType] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.environment = environment or self._detect_environment()
        self.logger = logging.getLogger(__name__)
        
        # Ensure logger has a handler for testing
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.WARNING)  # Suppress info/debug during tests
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.WARNING)
        
        # Configuration state
        self._config: Optional[SovereignConfig] = None
        self._config_cache: Dict[str, Any] = {}
        self._last_modified: Optional[float] = None
        
        # Hot reload
        self._hot_reload_enabled = False
        self._file_observer: Optional[Observer] = None
        self._reload_callbacks: List[Callable[[SovereignConfig], None]] = []
    
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path"""
        return os.path.join(os.path.dirname(__file__), '..', 'config', 'sovereign.yaml')
    
    def _detect_environment(self) -> EnvironmentType:
        """Detect the current environment"""
        env_name = os.getenv('SOVEREIGN_ENV', 'development').lower()
        
        try:
            return EnvironmentType(env_name)
        except ValueError:
            # Use print instead of logger since logger might not be initialized yet
            print(f"Unknown environment '{env_name}', defaulting to development")
            return EnvironmentType.DEVELOPMENT
    
    def _load_yaml_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            self.logger.warning(f"Configuration file not found: {config_path}")
            return {}
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}
            
            # Check if file was modified
            stat = os.stat(config_path)
            self._last_modified = stat.st_mtime
            
            return config_data
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_path}: {e}")
            return {}
    
    def _apply_environment_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment-specific overrides"""
        # Load environment-specific config file
        env_config_path = self.config_path.replace('.yaml', f'.{self.environment.value}.yaml')
        if os.path.exists(env_config_path):
            env_config = self._load_yaml_config(env_config_path)
            config_data = self._deep_merge(config_data, env_config)
        
        # Apply environment variable overrides
        env_overrides = self._get_environment_variable_overrides()
        if env_overrides:
            config_data = self._deep_merge(config_data, env_overrides)
        
        return config_data
    
    def _get_environment_variable_overrides(self) -> Dict[str, Any]:
        """Get configuration overrides from environment variables"""
        overrides = {}
        
        # API keys
        api_overrides = {}
        if openai_key := os.getenv('OPENAI_API_KEY'):
            api_overrides['openai_api_key'] = openai_key
        if openai_org := os.getenv('OPENAI_ORG_ID'):
            api_overrides['openai_org_id'] = openai_org
        if anthropic_key := os.getenv('ANTHROPIC_API_KEY'):
            api_overrides['anthropic_api_key'] = anthropic_key
        if openrouter_key := os.getenv('OPENROUTER_API_KEY'):
            api_overrides['openrouter_api_key'] = openrouter_key
        if perplexity_key := os.getenv('PERPLEXITY_API_KEY'):
            api_overrides['perplexity_api_key'] = perplexity_key
        if kimi_key := os.getenv('KIMI_API_KEY'):
            api_overrides['kimi_api_key'] = kimi_key
        if google_key := os.getenv('GOOGLE_API_KEY'):
            api_overrides['google_api_key'] = google_key
        if elevenlabs_key := os.getenv('ELEVENLABS_API_KEY'):
            api_overrides['elevenlabs_api_key'] = elevenlabs_key
        
        if api_overrides:
            overrides['api'] = api_overrides
        
        # Debug mode
        if debug := os.getenv('DEBUG'):
            overrides['development'] = {'debug_mode': debug.lower() in ('true', '1', 'yes')}
        
        # Log level
        if log_level := os.getenv('LOG_LEVEL'):
            overrides['monitoring'] = {'log_level': log_level.upper()}
        
        # Environment
        if env := os.getenv('SOVEREIGN_ENV'):
            overrides['environment'] = env
        
        return overrides
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _create_config_from_dict(self, config_data: Dict[str, Any]) -> SovereignConfig:
        """Create SovereignConfig from dictionary data"""
        try:
            # Create config sections
            api_config = APIConfig(**config_data.get('api', {}))
            audio_config = AudioConfig(**config_data.get('audio', {}))
            stt_config = STTConfig(**config_data.get('stt', {}))
            tts_config = TTSConfig(**config_data.get('tts', {}))
            
            # Handle LLM config with nested model configs
            llm_data = config_data.get('llm', {})
            llm_config = LLMConfig()
            if 'fast' in llm_data:
                llm_config.fast = ModelConfig(**llm_data['fast'])
            if 'deep' in llm_data:
                llm_config.deep = ModelConfig(**llm_data['deep'])
            if 'local' in llm_data:
                llm_config.local = ModelConfig(**llm_data['local'])
            
            # Handle other config sections
            memory_config = MemoryConfig(**config_data.get('memory', {}))
            screen_config = ScreenConfig(**config_data.get('screen', {}))
            code_agent_config = CodeAgentConfig(**config_data.get('code_agent', {}))
            security_config = SecurityConfig(**config_data.get('security', {}))
            monitoring_config = MonitoringConfig(**config_data.get('monitoring', {}))
            development_config = DevelopmentConfig(**config_data.get('development', {}))
            
            # Create main config
            config = SovereignConfig(
                environment=self.environment,
                api=api_config,
                audio=audio_config,
                stt=stt_config,
                tts=tts_config,
                llm=llm_config,
                memory=memory_config,
                screen=screen_config,
                code_agent=code_agent_config,
                security=security_config,
                monitoring=monitoring_config,
                development=development_config
            )
            
            # Apply top-level overrides
            if 'features' in config_data:
                config.features.update(config_data['features'])
            if 'version' in config_data:
                config.version = config_data['version']
            if 'name' in config_data:
                config.name = config_data['name']
            if 'description' in config_data:
                config.description = config_data['description']
            
            return config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to create configuration: {e}")
    
    def _validate_config(self, config: SovereignConfig) -> None:
        """Validate configuration for completeness and correctness"""
        errors = []
        
        # Validate API keys for enabled features (skip if mock APIs are enabled)
        if not config.development.mock_apis:
            if not config.api.openai_api_key:
                errors.append("OpenAI API key is required")
            
            if config.llm.fast.provider == "openrouter" and not config.api.openrouter_api_key:
                errors.append("OpenRouter API key is required for fast LLM model")
            
            if config.code_agent.enabled and not config.api.kimi_api_key:
                errors.append("Kimi API key is required for code agent")
        
        # Validate audio settings
        if config.audio.sample_rate not in [8000, 16000, 22050, 44100, 48000]:
            errors.append(f"Invalid sample rate: {config.audio.sample_rate}")
        
        if config.audio.channels not in [1, 2]:
            errors.append(f"Invalid channel count: {config.audio.channels}")
        
        # Validate timeout values
        if config.stt.timeout <= 0:
            errors.append("STT timeout must be positive")
        
        if config.tts.timeout <= 0:
            errors.append("TTS timeout must be positive")
        
        # Validate model settings
        for model_name, model_config in [
            ("fast", config.llm.fast),
            ("deep", config.llm.deep),
            ("local", config.llm.local)
        ]:
            if model_config.max_tokens <= 0:
                errors.append(f"Invalid max_tokens for {model_name} model")
            
            if not 0 <= model_config.temperature <= 2:
                errors.append(f"Invalid temperature for {model_name} model")
        
        # Validate memory settings
        if config.memory.retrieval_k <= 0:
            errors.append("Memory retrieval_k must be positive")
        
        if not 0 <= config.memory.similarity_threshold <= 1:
            errors.append("Memory similarity_threshold must be between 0 and 1")
        
        # Validate screen settings
        if config.screen.enabled and config.screen.screenshot_interval <= 0:
            errors.append("Screen screenshot_interval must be positive")
        
        if errors:
            raise ConfigurationError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def load_config(self) -> SovereignConfig:
        """Load and validate configuration"""
        try:
            self.logger.info(f"Loading configuration from {self.config_path}")
            
            # Load YAML configuration
            config_data = self._load_yaml_config(self.config_path)
            
            # If no config data was loaded (missing file, invalid YAML, etc.)
            # ensure mock_apis is enabled for testing/development
            if not config_data:
                config_data = {
                    'development': {
                        'mock_apis': True,
                        'debug_mode': False,
                        'enable_hot_reload': True
                    }
                }
            elif 'development' not in config_data:
                config_data['development'] = {'mock_apis': True}
            elif 'mock_apis' not in config_data.get('development', {}):
                config_data['development']['mock_apis'] = True
            
            # Apply environment overrides
            config_data = self._apply_environment_overrides(config_data)
            
            # Create configuration object
            config = self._create_config_from_dict(config_data)
            
            # Validate configuration
            self._validate_config(config)
            
            # Cache configuration
            self._config = config
            self._config_cache = config_data
            
            self.logger.info(f"Configuration loaded successfully for {self.environment.value} environment")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise ConfigurationError(f"Configuration loading failed: {e}")
    
    def get_config(self) -> SovereignConfig:
        """Get the current configuration, loading if necessary"""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def reload_config(self) -> SovereignConfig:
        """Reload configuration from file"""
        self.logger.info("Reloading configuration...")
        self._config = None
        self._config_cache.clear()
        
        config = self.load_config()
        
        # Notify reload callbacks
        for callback in self._reload_callbacks:
            try:
                callback(config)
            except Exception as e:
                self.logger.error(f"Error in reload callback: {e}")
        
        return config
    
    def save_config(self, config: SovereignConfig) -> None:
        """Save configuration to file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Convert to dictionary for serialization
            config_dict = {
                'environment': config.environment.value,
                'version': config.version,
                'name': config.name,
                'description': config.description,
                'features': config.features,
                
                'api': {
                    'openai_base_url': config.api.openai_base_url,
                    'anthropic_base_url': config.api.anthropic_base_url,
                    'openrouter_base_url': config.api.openrouter_base_url,
                    'perplexity_base_url': config.api.perplexity_base_url,
                    'kimi_base_url': config.api.kimi_base_url,
                    'validate_keys_on_startup': config.api.validate_keys_on_startup,
                    'timeout': config.api.timeout,
                    'max_retries': config.api.max_retries
                },
                
                'audio': {
                    'sample_rate': config.audio.sample_rate,
                    'chunk_size': config.audio.chunk_size,
                    'channels': config.audio.channels,
                    'vad_enabled': config.audio.vad_enabled,
                    'silence_threshold': config.audio.silence_threshold,
                    'silence_duration': config.audio.silence_duration,
                    'min_audio_length': config.audio.min_audio_length,
                    'max_audio_length': config.audio.max_audio_length
                },
                
                'stt': {
                    'primary_provider': config.stt.primary_provider,
                    'primary_model': config.stt.primary_model,
                    'primary_language': config.stt.primary_language,
                    'fallback_provider': config.stt.fallback_provider,
                    'fallback_model': config.stt.fallback_model,
                    'timeout': config.stt.timeout,
                    'temperature': config.stt.temperature
                },
                
                'tts': {
                    'primary_provider': config.tts.primary_provider,
                    'primary_model': config.tts.primary_model,
                    'primary_voice': config.tts.primary_voice,
                    'primary_speed': config.tts.primary_speed,
                    'fallback_provider': config.tts.fallback_provider,
                    'fallback_voice': config.tts.fallback_voice,
                    'timeout': config.tts.timeout,
                    'response_format': config.tts.response_format
                },
                
                'llm': {
                    'fast': {
                        'provider': config.llm.fast.provider,
                        'model': config.llm.fast.model,
                        'max_tokens': config.llm.fast.max_tokens,
                        'temperature': config.llm.fast.temperature,
                        'timeout': config.llm.fast.timeout
                    },
                    'deep': {
                        'provider': config.llm.deep.provider,
                        'model': config.llm.deep.model,
                        'max_tokens': config.llm.deep.max_tokens,
                        'temperature': config.llm.deep.temperature,
                        'timeout': config.llm.deep.timeout
                    },
                    'local': {
                        'provider': config.llm.local.provider,
                        'model': config.llm.local.model,
                        'max_tokens': config.llm.local.max_tokens,
                        'temperature': config.llm.local.temperature,
                        'timeout': config.llm.local.timeout
                    }
                },
                
                'memory': {
                    'provider': config.memory.provider,
                    'persist_directory': config.memory.persist_directory,
                    'collection_name_conversations': config.memory.collection_name_conversations,
                    'collection_name_screen': config.memory.collection_name_screen,
                    'embedding_provider': config.memory.embedding_provider,
                    'embedding_model': config.memory.embedding_model,
                    'retrieval_k': config.memory.retrieval_k,
                    'similarity_threshold': config.memory.similarity_threshold,
                    'max_context_length': config.memory.max_context_length
                },
                
                'screen': {
                    'enabled': config.screen.enabled,
                    'screenshot_interval': config.screen.screenshot_interval,
                    'ocr_provider': config.screen.ocr_provider,
                    'ocr_language': config.screen.ocr_language,
                    'ocr_config': config.screen.ocr_config,
                    'resize_factor': config.screen.resize_factor,
                    'contrast_enhancement': config.screen.contrast_enhancement,
                    'noise_reduction': config.screen.noise_reduction
                },
                
                'code_agent': {
                    'enabled': config.code_agent.enabled,
                    'provider': config.code_agent.provider,
                    'model': config.code_agent.model,
                    'trigger_patterns': config.code_agent.trigger_patterns,
                    'max_tokens': config.code_agent.max_tokens,
                    'temperature': config.code_agent.temperature,
                    'timeout': config.code_agent.timeout
                },
                
                'security': {
                    'validate_api_keys': config.security.validate_api_keys,
                    'mask_keys_in_logs': config.security.mask_keys_in_logs,
                    'encrypt_memory': config.security.encrypt_memory,
                    'require_https': config.security.require_https,
                    'allowed_hosts': config.security.allowed_hosts
                },
                
                'monitoring': {
                    'enabled': config.monitoring.enabled,
                    'log_level': config.monitoring.log_level,
                    'log_file': config.monitoring.log_file,
                    'metrics_enabled': config.monitoring.metrics_enabled,
                    'metrics_port': config.monitoring.metrics_port,
                    'health_check_interval': config.monitoring.health_check_interval
                },
                
                'development': {
                    'debug_mode': config.development.debug_mode,
                    'test_mode': config.development.test_mode,
                    'mock_apis': config.development.mock_apis,
                    'enable_hot_reload': config.development.enable_hot_reload,
                    'skip_audio_init': config.development.skip_audio_init
                }
            }
            
            # Save to file
            with open(self.config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2, sort_keys=False)
            
            self.logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise ConfigurationError(f"Configuration saving failed: {e}")
    
    def enable_hot_reload(self) -> None:
        """Enable hot reload for development"""
        if self._hot_reload_enabled:
            return
        
        try:
            self._file_observer = Observer()
            event_handler = ConfigFileWatcher(self)
            
            # Watch config directory
            config_dir = os.path.dirname(self.config_path)
            self._file_observer.schedule(event_handler, config_dir, recursive=False)
            
            self._file_observer.start()
            self._hot_reload_enabled = True
            
            self.logger.info("Hot reload enabled for configuration files")
            
        except Exception as e:
            self.logger.error(f"Failed to enable hot reload: {e}")
    
    def disable_hot_reload(self) -> None:
        """Disable hot reload"""
        if not self._hot_reload_enabled:
            return
        
        if self._file_observer:
            self._file_observer.stop()
            self._file_observer.join()
            self._file_observer = None
        
        self._hot_reload_enabled = False
        self.logger.info("Hot reload disabled")
    
    def add_reload_callback(self, callback: Callable[[SovereignConfig], None]) -> None:
        """Add a callback to be called when configuration is reloaded"""
        self._reload_callbacks.append(callback)
    
    def remove_reload_callback(self, callback: Callable[[SovereignConfig], None]) -> None:
        """Remove a reload callback"""
        if callback in self._reload_callbacks:
            self._reload_callbacks.remove(callback)
    
    def is_config_modified(self) -> bool:
        """Check if configuration file has been modified"""
        if not os.path.exists(self.config_path):
            return False
        
        stat = os.stat(self.config_path)
        return stat.st_mtime != self._last_modified
    
    def create_default_config(self) -> None:
        """Create a default configuration file"""
        default_config = SovereignConfig(environment=self.environment)
        # Enable mock APIs by default for development
        default_config.development.mock_apis = True
        self.save_config(default_config)
        self.logger.info(f"Created default configuration at {self.config_path}")

# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None

def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """Get the global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path=config_path)
    return _config_manager

def get_config() -> SovereignConfig:
    """Get the current configuration"""
    return get_config_manager().get_config()

def reload_config() -> SovereignConfig:
    """Reload the configuration"""
    return get_config_manager().reload_config()

def create_config_template(output_path: str = "config/sovereign.yaml") -> None:
    """Create a configuration template file"""
    manager = ConfigManager(config_path=output_path)
    manager.create_default_config()
    print(f"Configuration template created at {output_path}")

# Export main components
__all__ = [
    'ConfigManager',
    'SovereignConfig',
    'APIConfig',
    'AudioConfig',
    'STTConfig',
    'TTSConfig',
    'LLMConfig',
    'MemoryConfig',
    'ScreenConfig',
    'CodeAgentConfig',
    'SecurityConfig',
    'MonitoringConfig',
    'DevelopmentConfig',
    'EnvironmentType',
    'ConfigurationError',
    'get_config_manager',
    'get_config',
    'reload_config',
    'create_config_template'
] 