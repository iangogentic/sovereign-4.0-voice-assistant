import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for an individual model"""
    id: str
    name: str
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout: float = 30.0
    cost_per_1k_tokens: float = 0.001
    max_requests_per_minute: int = 60
    max_tokens_per_minute: int = 60000

@dataclass
class RouterConfig:
    """Configuration for the LLM router"""
    # API Configuration
    openrouter_api_key: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_site_url: str = "https://github.com/user/project"
    openrouter_app_name: str = "Sovereign Voice Assistant"
    
    # Model Configurations
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    
    # Rate Limiting Configuration
    enable_rate_limiting: bool = True
    rate_limit_strategy: str = "token_bucket"  # or "sliding_window"
    default_retry_attempts: int = 3
    retry_backoff_factor: float = 2.0
    max_retry_delay: float = 60.0
    
    # Connection Configuration
    connection_timeout: float = 30.0
    read_timeout: float = 60.0
    max_connections: int = 10
    max_keepalive_connections: int = 5
    keepalive_expiry: float = 30.0
    
    # Logging Configuration
    log_requests: bool = True
    log_responses: bool = False  # Set to True for debugging
    log_rate_limits: bool = True
    
    # Conversation Configuration
    max_conversation_history: int = 10
    include_conversation_context: bool = True
    max_context_messages: int = 8
    
    def __post_init__(self):
        """Initialize default models if not provided"""
        if not self.models:
            self.models = {
                'fast': ModelConfig(
                    id='openai/gpt-4o-mini',
                    name='GPT-4o-mini',
                    max_tokens=1000,
                    temperature=0.7,
                    timeout=10.0,
                    cost_per_1k_tokens=0.0015,
                    max_requests_per_minute=100,
                    max_tokens_per_minute=100000
                ),
                'deep': ModelConfig(
                    id='openai/gpt-4o',
                    name='GPT-4o',
                    max_tokens=2000,
                    temperature=0.8,
                    timeout=60.0,
                    cost_per_1k_tokens=0.015,
                    max_requests_per_minute=20,
                    max_tokens_per_minute=20000
                )
            }

class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self._config: Optional[RouterConfig] = None
    
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path"""
        return os.path.join(os.path.dirname(__file__), '..', 'config', 'router_config.yaml')
    
    def load_config(self) -> RouterConfig:
        """Load configuration from file and environment variables"""
        if self._config is not None:
            return self._config
        
        # Start with default configuration
        config_data = {}
        
        # Load from YAML file if it exists
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
                logger.info(f"Loaded router configuration from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config file {self.config_path}: {e}")
        
        # Override with environment variables
        env_overrides = self._get_env_overrides()
        config_data.update(env_overrides)
        
        # Create RouterConfig instance
        self._config = RouterConfig(**config_data)
        
        # Validate configuration
        self._validate_config()
        
        return self._config
    
    def _get_env_overrides(self) -> Dict[str, Any]:
        """Get configuration overrides from environment variables"""
        overrides = {}
        
        # API Key
        if openrouter_key := os.getenv('OPENROUTER_API_KEY'):
            overrides['openrouter_api_key'] = openrouter_key
        
        # Base URL
        if base_url := os.getenv('OPENROUTER_BASE_URL'):
            overrides['openrouter_base_url'] = base_url
        
        # App identification
        if site_url := os.getenv('OPENROUTER_SITE_URL'):
            overrides['openrouter_site_url'] = site_url
        
        if app_name := os.getenv('OPENROUTER_APP_NAME'):
            overrides['openrouter_app_name'] = app_name
        
        # Rate limiting
        if rate_limit := os.getenv('OPENROUTER_ENABLE_RATE_LIMITING'):
            overrides['enable_rate_limiting'] = rate_limit.lower() in ('true', '1', 'yes')
        
        # Timeouts
        if conn_timeout := os.getenv('OPENROUTER_CONNECTION_TIMEOUT'):
            try:
                overrides['connection_timeout'] = float(conn_timeout)
            except ValueError:
                logger.warning(f"Invalid connection timeout: {conn_timeout}")
        
        if read_timeout := os.getenv('OPENROUTER_READ_TIMEOUT'):
            try:
                overrides['read_timeout'] = float(read_timeout)
            except ValueError:
                logger.warning(f"Invalid read timeout: {read_timeout}")
        
        # Logging
        if log_requests := os.getenv('OPENROUTER_LOG_REQUESTS'):
            overrides['log_requests'] = log_requests.lower() in ('true', '1', 'yes')
        
        if log_responses := os.getenv('OPENROUTER_LOG_RESPONSES'):
            overrides['log_responses'] = log_responses.lower() in ('true', '1', 'yes')
        
        return overrides
    
    def _validate_config(self) -> None:
        """Validate the configuration"""
        if not self._config:
            return
        
        # Check for required API key
        if not self._config.openrouter_api_key:
            logger.warning("No OpenRouter API key provided. Set OPENROUTER_API_KEY environment variable.")
        
        # Validate model configurations
        for model_type, model_config in self._config.models.items():
            if not model_config.id:
                raise ValueError(f"Model ID is required for {model_type}")
            
            if model_config.timeout <= 0:
                raise ValueError(f"Timeout must be positive for {model_type}")
            
            if model_config.max_tokens <= 0:
                raise ValueError(f"Max tokens must be positive for {model_type}")
        
        # Validate rate limiting configuration
        if self._config.enable_rate_limiting:
            if self._config.default_retry_attempts < 0:
                raise ValueError("Retry attempts must be non-negative")
            
            if self._config.retry_backoff_factor <= 0:
                raise ValueError("Backoff factor must be positive")
        
        # Validate connection configuration
        if self._config.connection_timeout <= 0:
            raise ValueError("Connection timeout must be positive")
        
        if self._config.read_timeout <= 0:
            raise ValueError("Read timeout must be positive")
        
        if self._config.max_connections <= 0:
            raise ValueError("Max connections must be positive")
        
        logger.info("Router configuration validated successfully")
    
    def save_config(self, config: RouterConfig) -> None:
        """Save configuration to file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Convert to dictionary for serialization
        config_dict = {
            'openrouter_base_url': config.openrouter_base_url,
            'openrouter_site_url': config.openrouter_site_url,
            'openrouter_app_name': config.openrouter_app_name,
            'enable_rate_limiting': config.enable_rate_limiting,
            'rate_limit_strategy': config.rate_limit_strategy,
            'default_retry_attempts': config.default_retry_attempts,
            'retry_backoff_factor': config.retry_backoff_factor,
            'max_retry_delay': config.max_retry_delay,
            'connection_timeout': config.connection_timeout,
            'read_timeout': config.read_timeout,
            'max_connections': config.max_connections,
            'max_keepalive_connections': config.max_keepalive_connections,
            'keepalive_expiry': config.keepalive_expiry,
            'log_requests': config.log_requests,
            'log_responses': config.log_responses,
            'log_rate_limits': config.log_rate_limits,
            'max_conversation_history': config.max_conversation_history,
            'include_conversation_context': config.include_conversation_context,
            'max_context_messages': config.max_context_messages,
            'models': {
                model_type: {
                    'id': model_config.id,
                    'name': model_config.name,
                    'max_tokens': model_config.max_tokens,
                    'temperature': model_config.temperature,
                    'timeout': model_config.timeout,
                    'cost_per_1k_tokens': model_config.cost_per_1k_tokens,
                    'max_requests_per_minute': model_config.max_requests_per_minute,
                    'max_tokens_per_minute': model_config.max_tokens_per_minute
                }
                for model_type, model_config in config.models.items()
            }
        }
        
        # Save to file
        with open(self.config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved router configuration to {self.config_path}")
    
    def reload_config(self) -> RouterConfig:
        """Reload configuration from file"""
        self._config = None
        return self.load_config()
    
    def get_config(self) -> RouterConfig:
        """Get the current configuration"""
        return self.load_config()

# Global configuration manager instance
config_manager = ConfigManager()

def get_router_config() -> RouterConfig:
    """Get the router configuration"""
    return config_manager.get_config() 