"""
Tests for Configuration Management System

Tests the comprehensive configuration management system including:
- Configuration loading and validation
- Environment variable overrides
- Multiple environment profiles
- Hot-reload functionality
- Error handling and recovery
- Schema validation
"""

import os
import sys
import tempfile
import shutil
import yaml
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from assistant.config_manager import (
    ConfigManager, SovereignConfig, APIConfig, AudioConfig, STTConfig, TTSConfig,
    LLMConfig, MemoryConfig, ScreenConfig, CodeAgentConfig, SecurityConfig,
    MonitoringConfig, DevelopmentConfig, EnvironmentType, OperationMode, ConfigurationError,
    get_config_manager, get_config, reload_config, create_config_template
)

class TestConfigManager:
    """Test the ConfigManager class"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
        self.config_manager = None
        
    def teardown_method(self):
        """Clean up test environment"""
        if self.config_manager:
            self.config_manager.disable_hot_reload()
        shutil.rmtree(self.temp_dir)
        
    def create_basic_config(self) -> dict:
        """Create a basic configuration dictionary"""
        return {
            "environment": "development",
            "version": "4.0.0",
            "name": "Test Assistant",
            "description": "Test configuration",
            "features": {
                "memory_enabled": True,
                "screen_monitoring": True,
                "code_agent": True
            },
            "api": {
                "openai_base_url": "https://api.openai.com/v1",
                "timeout": 30.0
            },
            "audio": {
                "sample_rate": 16000,
                "chunk_size": 1024,
                "vad_enabled": True
            },
            "stt": {
                "primary_provider": "openai",
                "primary_model": "whisper-1",
                "timeout": 10.0
            },
            "tts": {
                "primary_provider": "openai",
                "primary_model": "tts-1",
                "timeout": 15.0
            },
            "llm": {
                "fast": {
                    "provider": "openrouter",
                    "model": "openai/gpt-4o-mini",
                    "max_tokens": 500,
                    "temperature": 0.7,
                    "timeout": 5.0
                },
                "deep": {
                    "provider": "openrouter",
                    "model": "openai/gpt-4o",
                    "max_tokens": 2000,
                    "temperature": 0.7,
                    "timeout": 30.0
                }
            },
            "memory": {
                "provider": "chroma",
                "retrieval_k": 5,
                "similarity_threshold": 0.7
            },
            "screen": {
                "enabled": True,
                "screenshot_interval": 3.0
            },
            "code_agent": {
                "enabled": True,
                "provider": "kimi",
                "model": "kimi-k2"
            },
            "security": {
                "validate_api_keys": True,
                "mask_keys_in_logs": True
            },
            "monitoring": {
                "enabled": True,
                "log_level": "INFO",
                "metrics_enabled": True
            },
            "development": {
                "debug_mode": False,
                "enable_hot_reload": True,
                "mock_apis": True  # Enable mock APIs for testing
            }
        }
    
    def write_config_file(self, config_data: dict, path: str = None):
        """Write configuration data to file"""
        if path is None:
            path = self.config_path
        
        with open(path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    def test_config_manager_initialization(self):
        """Test ConfigManager initialization"""
        self.config_manager = ConfigManager(config_path=self.config_path)
        
        assert self.config_manager.config_path == self.config_path
        assert self.config_manager.environment == EnvironmentType.DEVELOPMENT
        assert self.config_manager._config is None
        assert not self.config_manager._hot_reload_enabled
    
    def test_load_config_success(self):
        """Test successful configuration loading"""
        config_data = self.create_basic_config()
        self.write_config_file(config_data)
        
        self.config_manager = ConfigManager(config_path=self.config_path)
        config = self.config_manager.load_config()
        
        assert isinstance(config, SovereignConfig)
        assert config.environment == EnvironmentType.DEVELOPMENT
        assert config.version == "4.0.0"
        assert config.name == "Test Assistant"
        assert config.features["memory_enabled"] is True
        assert config.api.timeout == 30.0
        assert config.audio.sample_rate == 16000
        assert config.stt.primary_provider == "openai"
        assert config.tts.primary_provider == "openai"
        assert config.llm.fast.provider == "openrouter"
        assert config.memory.retrieval_k == 5
        assert config.screen.enabled is True
        assert config.code_agent.enabled is True
        assert config.security.validate_api_keys is True
        assert config.monitoring.enabled is True
        assert config.development.debug_mode is False
    
    def test_load_config_file_not_found(self):
        """Test loading configuration when file doesn't exist"""
        self.config_manager = ConfigManager(config_path=self.config_path)
        
        # Should not raise an exception, should use defaults
        config = self.config_manager.load_config()
        
        assert isinstance(config, SovereignConfig)
        assert config.environment == EnvironmentType.DEVELOPMENT
        assert config.version == "4.0.0"
        assert config.name == "Sovereign Voice Assistant"
    
    def test_load_config_invalid_yaml(self):
        """Test loading configuration with invalid YAML"""
        with open(self.config_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        self.config_manager = ConfigManager(config_path=self.config_path)
        
        # Should not raise an exception, should use defaults
        config = self.config_manager.load_config()
        
        assert isinstance(config, SovereignConfig)
        assert config.environment == EnvironmentType.DEVELOPMENT
    
    def test_environment_variable_overrides(self):
        """Test environment variable overrides"""
        config_data = self.create_basic_config()
        self.write_config_file(config_data)
        
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_openai_key',
            'OPENROUTER_API_KEY': 'test_openrouter_key',
            'DEBUG': 'true',
            'LOG_LEVEL': 'DEBUG'
        }):
            self.config_manager = ConfigManager(config_path=self.config_path)
            config = self.config_manager.load_config()
            
            assert config.api.openai_api_key == 'test_openai_key'
            assert config.api.openrouter_api_key == 'test_openrouter_key'
            assert config.development.debug_mode is True
            assert config.monitoring.log_level == 'DEBUG'
    
    def test_environment_detection(self):
        """Test environment detection from environment variables"""
        # Create basic config to avoid validation errors
        config_data = self.create_basic_config()
        self.write_config_file(config_data)
        
        with patch.dict(os.environ, {'SOVEREIGN_ENV': 'production'}):
            self.config_manager = ConfigManager(config_path=self.config_path)
            assert self.config_manager.environment == EnvironmentType.PRODUCTION
        
        with patch.dict(os.environ, {'SOVEREIGN_ENV': 'offline'}):
            self.config_manager = ConfigManager(config_path=self.config_path)
            assert self.config_manager.environment == EnvironmentType.OFFLINE
        
        with patch.dict(os.environ, {'SOVEREIGN_ENV': 'invalid'}):
            self.config_manager = ConfigManager(config_path=self.config_path)
            assert self.config_manager.environment == EnvironmentType.DEVELOPMENT
    
    def test_environment_specific_overrides(self):
        """Test environment-specific configuration overrides"""
        config_data = self.create_basic_config()
        self.write_config_file(config_data)
        
        # Create environment-specific config
        env_config_path = self.config_path.replace('.yaml', '.production.yaml')
        env_config = {
            "monitoring": {
                "log_level": "WARNING"
            },
            "security": {
                "validate_api_keys": True,
                "encrypt_memory": True
            }
        }
        self.write_config_file(env_config, env_config_path)
        
        with patch.dict(os.environ, {'SOVEREIGN_ENV': 'production'}):
            self.config_manager = ConfigManager(config_path=self.config_path)
            config = self.config_manager.load_config()
            
            assert config.monitoring.log_level == "WARNING"
            assert config.security.encrypt_memory is True
    
    def test_config_validation_success(self):
        """Test successful configuration validation"""
        config_data = self.create_basic_config()
        self.write_config_file(config_data)
        
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_key',
            'OPENROUTER_API_KEY': 'test_key',
            'KIMI_API_KEY': 'test_key'
        }):
            self.config_manager = ConfigManager(config_path=self.config_path)
            config = self.config_manager.load_config()
            
            # Should not raise an exception
            assert isinstance(config, SovereignConfig)
    
    def test_config_validation_failures(self):
        """Test configuration validation failures"""
        config_data = self.create_basic_config()
        
        # Test invalid sample rate
        config_data['audio']['sample_rate'] = 12345
        self.write_config_file(config_data)
        
        self.config_manager = ConfigManager(config_path=self.config_path)
        
        with pytest.raises(ConfigurationError):
            self.config_manager.load_config()
    
    def test_save_config(self):
        """Test saving configuration to file"""
        config_data = self.create_basic_config()
        self.write_config_file(config_data)
        
        self.config_manager = ConfigManager(config_path=self.config_path)
        config = self.config_manager.load_config()
        
        # Modify configuration
        config.version = "4.1.0"
        config.name = "Modified Assistant"
        config.audio.sample_rate = 48000
        
        # Save configuration
        self.config_manager.save_config(config)
        
        # Verify the file was updated
        with open(self.config_path, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data['version'] == "4.1.0"
        assert saved_data['name'] == "Modified Assistant"
        assert saved_data['audio']['sample_rate'] == 48000
    
    def test_reload_config(self):
        """Test configuration reloading"""
        config_data = self.create_basic_config()
        self.write_config_file(config_data)
        
        self.config_manager = ConfigManager(config_path=self.config_path)
        config1 = self.config_manager.load_config()
        
        # Modify file
        config_data['version'] = "4.2.0"
        self.write_config_file(config_data)
        
        # Reload configuration
        config2 = self.config_manager.reload_config()
        
        assert config1.version == "4.0.0"
        assert config2.version == "4.2.0"
    
    def test_get_config_caching(self):
        """Test configuration caching"""
        config_data = self.create_basic_config()
        self.write_config_file(config_data)
        
        self.config_manager = ConfigManager(config_path=self.config_path)
        
        config1 = self.config_manager.get_config()
        config2 = self.config_manager.get_config()
        
        # Should return the same cached instance
        assert config1 is config2
    
    def test_hot_reload_functionality(self):
        """Test hot reload functionality"""
        config_data = self.create_basic_config()
        self.write_config_file(config_data)
        
        self.config_manager = ConfigManager(config_path=self.config_path)
        self.config_manager.load_config()
        
        # Test enabling hot reload
        self.config_manager.enable_hot_reload()
        assert self.config_manager._hot_reload_enabled is True
        assert self.config_manager._file_observer is not None
        
        # Test disabling hot reload
        self.config_manager.disable_hot_reload()
        assert self.config_manager._hot_reload_enabled is False
        assert self.config_manager._file_observer is None
    
    def test_reload_callbacks(self):
        """Test reload callback functionality"""
        config_data = self.create_basic_config()
        self.write_config_file(config_data)
        
        self.config_manager = ConfigManager(config_path=self.config_path)
        self.config_manager.load_config()
        
        # Create mock callback
        callback = Mock()
        
        # Add callback
        self.config_manager.add_reload_callback(callback)
        
        # Reload configuration
        new_config = self.config_manager.reload_config()
        
        # Verify callback was called
        callback.assert_called_once_with(new_config)
        
        # Remove callback
        self.config_manager.remove_reload_callback(callback)
        
        # Reload again
        self.config_manager.reload_config()
        
        # Callback should not be called again
        callback.assert_called_once()
    
    def test_is_config_modified(self):
        """Test configuration modification detection"""
        config_data = self.create_basic_config()
        self.write_config_file(config_data)
        
        self.config_manager = ConfigManager(config_path=self.config_path)
        self.config_manager.load_config()
        
        # Should not be modified initially
        assert not self.config_manager.is_config_modified()
        
        # Modify file
        import time
        time.sleep(0.1)  # Ensure different modification time
        config_data['version'] = "4.3.0"
        self.write_config_file(config_data)
        
        # Should detect modification
        assert self.config_manager.is_config_modified()
    
    def test_create_default_config(self):
        """Test creating default configuration"""
        self.config_manager = ConfigManager(config_path=self.config_path)
        self.config_manager.create_default_config()
        
        # Verify file was created
        assert os.path.exists(self.config_path)
        
        # Verify it can be loaded
        config = self.config_manager.load_config()
        assert isinstance(config, SovereignConfig)
        assert config.version == "4.0.0"
        assert config.name == "Sovereign Voice Assistant"

class TestConfigurationDataClasses:
    """Test configuration data classes"""
    
    def test_api_config_defaults(self):
        """Test APIConfig default values"""
        config = APIConfig()
        
        assert config.openai_api_key is None
        assert config.openai_base_url == "https://api.openai.com/v1"
        assert config.anthropic_base_url == "https://api.anthropic.com"
        assert config.openrouter_base_url == "https://openrouter.ai/api/v1"
        assert config.validate_keys_on_startup is True
        assert config.timeout == 30.0
        assert config.max_retries == 3
    
    def test_audio_config_defaults(self):
        """Test AudioConfig default values"""
        config = AudioConfig()
        
        assert config.sample_rate == 16000
        assert config.chunk_size == 1024
        assert config.channels == 1
        assert config.vad_enabled is True
        assert config.silence_threshold == 0.001
        assert config.silence_duration == 1.0
        assert config.noise_reduction is True
    
    def test_stt_config_defaults(self):
        """Test STTConfig default values"""
        config = STTConfig()
        
        assert config.primary_provider == "openai"
        assert config.primary_model == "whisper-1"
        assert config.primary_language == "en"
        assert config.fallback_provider == "whisper-cpp"
        assert config.fallback_model == "tiny.en"
        assert config.timeout == 10.0
        assert config.temperature == 0.0
        assert config.max_retries == 2
    
    def test_tts_config_defaults(self):
        """Test TTSConfig default values"""
        config = TTSConfig()
        
        assert config.primary_provider == "openai"
        assert config.primary_model == "tts-1"
        assert config.primary_voice == "alloy"
        assert config.primary_speed == 1.0
        assert config.fallback_provider == "piper"
        assert config.timeout == 15.0
        assert config.response_format == "mp3"
        assert config.quality == "standard"
    
    def test_llm_config_defaults(self):
        """Test LLMConfig default values"""
        config = LLMConfig()
        
        assert config.fast.provider == "openrouter"
        assert config.fast.model == "openai/gpt-4o-mini"
        assert config.fast.max_tokens == 500
        assert config.fast.timeout == 5.0
        
        assert config.deep.provider == "openrouter"
        assert config.deep.model == "openai/gpt-4o"
        assert config.deep.max_tokens == 2000
        assert config.deep.timeout == 30.0
        
        assert config.local.provider == "llama-cpp"
        assert config.local.model == "gemma-2b-it-q4_0.gguf"
        assert config.local.max_tokens == 500
        assert config.local.timeout == 10.0
        
        assert config.enable_fallback is True
        assert config.fallback_chain == ["fast", "deep", "local"]
        assert config.max_fallback_attempts == 3
    
    def test_memory_config_defaults(self):
        """Test MemoryConfig default values"""
        config = MemoryConfig()
        
        assert config.provider == "chroma"
        assert config.persist_directory == "./data/chroma"
        assert config.collection_name_conversations == "sovereign_conversations"
        assert config.collection_name_screen == "sovereign_screen"
        assert config.embedding_provider == "openai"
        assert config.embedding_model == "text-embedding-3-small"
        assert config.retrieval_k == 5
        assert config.similarity_threshold == 0.7
        assert config.max_context_length == 8000
    
    def test_sovereign_config_defaults(self):
        """Test SovereignConfig default values"""
        config = SovereignConfig()
        
        assert config.environment == EnvironmentType.DEVELOPMENT
        assert config.version == "4.0.0"
        assert config.name == "Sovereign Voice Assistant"
        assert config.description == "Advanced AI voice assistant with multi-modal capabilities"
        assert config.features["memory_enabled"] is True
        assert config.features["screen_monitoring"] is True
        assert config.features["code_agent"] is True
        assert config.features["offline_mode"] is False
        
        assert isinstance(config.api, APIConfig)
        assert isinstance(config.audio, AudioConfig)
        assert isinstance(config.stt, STTConfig)
        assert isinstance(config.tts, TTSConfig)
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.memory, MemoryConfig)
        assert isinstance(config.screen, ScreenConfig)
        assert isinstance(config.code_agent, CodeAgentConfig)
        assert isinstance(config.security, SecurityConfig)
        assert isinstance(config.monitoring, MonitoringConfig)
        assert isinstance(config.development, DevelopmentConfig)

class TestGlobalConfigFunctions:
    """Test global configuration functions"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
        
        # Clear any existing global config manager
        import assistant.config_manager as config_module
        config_module._config_manager = None
        
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
        
        # Clear global config manager
        import assistant.config_manager as config_module
        config_module._config_manager = None
    
    def test_get_config_manager(self):
        """Test get_config_manager function"""
        manager1 = get_config_manager(self.config_path)
        manager2 = get_config_manager()
        
        # Should return the same instance
        assert manager1 is manager2
        assert manager1.config_path == self.config_path
    
    def test_get_config(self):
        """Test get_config function"""
        config_data = {
            "version": "4.0.0",
            "name": "Test Assistant"
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Set up global config manager
        get_config_manager(self.config_path)
        
        config = get_config()
        
        assert isinstance(config, SovereignConfig)
        assert config.version == "4.0.0"
        assert config.name == "Test Assistant"
    
    def test_reload_config(self):
        """Test reload_config function"""
        config_data = {
            "version": "4.0.0",
            "name": "Test Assistant"
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Set up global config manager
        get_config_manager(self.config_path)
        
        config1 = get_config()
        
        # Modify file
        config_data["version"] = "4.1.0"
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        config2 = reload_config()
        
        assert config1.version == "4.0.0"
        assert config2.version == "4.1.0"
    
    def test_create_config_template(self):
        """Test create_config_template function"""
        create_config_template(self.config_path)
        
        # Verify file was created
        assert os.path.exists(self.config_path)
        
        # Verify it can be loaded
        manager = ConfigManager(config_path=self.config_path)
        config = manager.load_config()
        
        assert isinstance(config, SovereignConfig)
        assert config.version == "4.0.0"
        assert config.name == "Sovereign Voice Assistant"

        config_path = os.path.join(self.temp_dir, "test_template.yaml")
        create_config_template(config_path)
        
        # Verify the template file was created
        assert os.path.exists(config_path)
        
        # Load and verify the content
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        assert 'environment' in config_data
        assert 'api' in config_data
        assert 'audio' in config_data


class TestOperationMode:
    """Test Operation Mode functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
        self.config_manager = None

    def teardown_method(self):
        """Clean up test environment"""
        if self.config_manager:
            self.config_manager.disable_hot_reload()
        shutil.rmtree(self.temp_dir)

    def create_config_with_operation_mode(self, operation_mode: str, realtime_enabled: bool = False) -> dict:
        """Create a configuration with specified operation mode"""
        return {
            "environment": "development",
            "operation_mode": operation_mode,
            "api": {
                "openai_api_key": "test-key" if not realtime_enabled or operation_mode != "realtime_only" else "test-realtime-key"
            },
            "stt": {
                "primary_provider": "openai",
                "primary_model": "whisper-1"
            },
            "tts": {
                "primary_provider": "openai",
                "primary_model": "tts-1"
            },
            "realtime_api": {
                "enabled": realtime_enabled
            },
            "development": {
                "mock_apis": False
            }
        }

    def write_config_file(self, config_data: dict, path: str = None):
        """Write configuration data to YAML file"""
        path = path or self.config_path
        with open(path, 'w') as f:
            yaml.dump(config_data, f)

    def test_operation_mode_enum_values(self):
        """Test OperationMode enum has correct values"""
        assert OperationMode.REALTIME_ONLY.value == "realtime_only"
        assert OperationMode.TRADITIONAL_ONLY.value == "traditional_only"
        assert OperationMode.HYBRID_AUTO.value == "hybrid_auto"

    def test_operation_mode_default_value(self):
        """Test SovereignConfig has correct default operation mode"""
        config = SovereignConfig()
        assert config.operation_mode == OperationMode.HYBRID_AUTO

    def test_load_config_with_operation_mode_hybrid_auto(self):
        """Test loading configuration with hybrid_auto operation mode"""
        config_data = self.create_config_with_operation_mode("hybrid_auto", realtime_enabled=True)
        self.write_config_file(config_data)
        
        self.config_manager = ConfigManager(self.config_path)
        config = self.config_manager.load_config()
        
        assert config.operation_mode == OperationMode.HYBRID_AUTO
        assert config.realtime_api.enabled is True

    def test_load_config_with_operation_mode_realtime_only(self):
        """Test loading configuration with realtime_only operation mode"""
        config_data = self.create_config_with_operation_mode("realtime_only", realtime_enabled=True)
        self.write_config_file(config_data)
        
        self.config_manager = ConfigManager(self.config_path)
        config = self.config_manager.load_config()
        
        assert config.operation_mode == OperationMode.REALTIME_ONLY
        assert config.realtime_api.enabled is True

    def test_load_config_with_operation_mode_traditional_only(self):
        """Test loading configuration with traditional_only operation mode"""
        config_data = self.create_config_with_operation_mode("traditional_only", realtime_enabled=False)
        self.write_config_file(config_data)
        
        self.config_manager = ConfigManager(self.config_path)
        config = self.config_manager.load_config()
        
        assert config.operation_mode == OperationMode.TRADITIONAL_ONLY
        assert config.realtime_api.enabled is False

    def test_validation_realtime_only_requires_realtime_api_enabled(self):
        """Test validation fails when REALTIME_ONLY mode has realtime API disabled"""
        config_data = self.create_config_with_operation_mode("realtime_only", realtime_enabled=False)
        self.write_config_file(config_data)
        
        self.config_manager = ConfigManager(self.config_path)
        
        with pytest.raises(ConfigurationError) as exc_info:
            self.config_manager.load_config()
        
        assert "REALTIME_ONLY operation mode requires realtime_api.enabled to be true" in str(exc_info.value)

    def test_validation_realtime_only_requires_openai_api_key(self):
        """Test validation fails when REALTIME_ONLY mode lacks OpenAI API key"""
        config_data = self.create_config_with_operation_mode("realtime_only", realtime_enabled=True)
        config_data["api"]["openai_api_key"] = None  # Remove API key
        self.write_config_file(config_data)
        
        self.config_manager = ConfigManager(self.config_path)
        
        with pytest.raises(ConfigurationError) as exc_info:
            self.config_manager.load_config()
        
        assert "REALTIME_ONLY operation mode requires OpenAI API key" in str(exc_info.value)

    def test_validation_traditional_only_requires_stt_tts_providers(self):
        """Test validation fails when TRADITIONAL_ONLY mode lacks STT/TTS providers"""
        config_data = self.create_config_with_operation_mode("traditional_only", realtime_enabled=False)
        config_data["stt"]["primary_provider"] = None  # Remove STT provider
        self.write_config_file(config_data)
        
        self.config_manager = ConfigManager(self.config_path)
        
        with pytest.raises(ConfigurationError) as exc_info:
            self.config_manager.load_config()
        
        assert "TRADITIONAL_ONLY operation mode requires STT and TTS providers to be configured" in str(exc_info.value)

    def test_validation_hybrid_auto_with_realtime_requires_openai_key(self):
        """Test validation fails when HYBRID_AUTO mode with realtime enabled lacks OpenAI API key"""
        config_data = self.create_config_with_operation_mode("hybrid_auto", realtime_enabled=True)
        config_data["api"]["openai_api_key"] = None  # Remove API key
        self.write_config_file(config_data)
        
        self.config_manager = ConfigManager(self.config_path)
        
        with pytest.raises(ConfigurationError) as exc_info:
            self.config_manager.load_config()
        
        assert "HYBRID_AUTO operation mode with Realtime API enabled requires OpenAI API key" in str(exc_info.value)

    def test_validation_hybrid_auto_requires_fallback_providers(self):
        """Test validation fails when HYBRID_AUTO mode lacks fallback STT/TTS providers"""
        config_data = self.create_config_with_operation_mode("hybrid_auto", realtime_enabled=False)
        config_data["tts"]["primary_provider"] = None  # Remove TTS provider
        self.write_config_file(config_data)
        
        self.config_manager = ConfigManager(self.config_path)
        
        with pytest.raises(ConfigurationError) as exc_info:
            self.config_manager.load_config()
        
        assert "HYBRID_AUTO operation mode requires STT and TTS providers for fallback functionality" in str(exc_info.value)

    @patch('assistant.config_manager.logger')
    def test_validation_traditional_only_with_realtime_enabled_warns(self, mock_logger):
        """Test warning when TRADITIONAL_ONLY mode has realtime API enabled"""
        config_data = self.create_config_with_operation_mode("traditional_only", realtime_enabled=True)
        self.write_config_file(config_data)
        
        self.config_manager = ConfigManager(self.config_path)
        config = self.config_manager.load_config()
        
        # Verify config loads successfully
        assert config.operation_mode == OperationMode.TRADITIONAL_ONLY
        assert config.realtime_api.enabled is True
        
        # Verify warning was logged
        mock_logger.warning.assert_called_with(
            "Realtime API is enabled but operation mode is TRADITIONAL_ONLY - Realtime API will be ignored"
        )

    def test_operation_mode_mock_apis_bypass_validation(self):
        """Test that mock APIs bypass operation mode API key validation"""
        config_data = self.create_config_with_operation_mode("realtime_only", realtime_enabled=True)
        config_data["api"]["openai_api_key"] = None  # Remove API key
        config_data["development"]["mock_apis"] = True  # Enable mock APIs
        self.write_config_file(config_data)
        
        self.config_manager = ConfigManager(self.config_path)
        config = self.config_manager.load_config()
        
        # Should load successfully with mock APIs enabled
        assert config.operation_mode == OperationMode.REALTIME_ONLY
        assert config.development.mock_apis is True

    def test_invalid_operation_mode_value(self):
        """Test handling of invalid operation mode value"""
        config_data = self.create_config_with_operation_mode("invalid_mode")
        self.write_config_file(config_data)
        
        self.config_manager = ConfigManager(self.config_path)
        
        with pytest.raises(Exception):  # Should raise during enum conversion
            self.config_manager.load_config()


class TestConfigurationErrors:
    """Test configuration error handling"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
        
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_configuration_error_creation(self):
        """Test ConfigurationError creation"""
        error = ConfigurationError("Test error message")
        
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
    
    def test_config_validation_error_handling(self):
        """Test error handling during config validation"""
        config_data = {
            "audio": {
                "sample_rate": -1,  # Invalid sample rate
                "channels": 5       # Invalid channel count
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        config_manager = ConfigManager(config_path=self.config_path)
        
        with pytest.raises(ConfigurationError) as exc_info:
            config_manager.load_config()
        
        assert "Configuration validation failed" in str(exc_info.value)
    
    def test_config_creation_error_handling(self):
        """Test error handling during config creation"""
        config_data = {
            "development": {
                "mock_apis": True  # Enable mock APIs to bypass validation
            },
            "llm": {
                "fast": {
                    "provider": "openrouter",
                    "model": "openai/gpt-4o-mini",
                    "max_tokens": "invalid_number",  # Invalid type - will cause TypeError
                    "temperature": 0.7,
                    "timeout": 5.0
                }
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        config_manager = ConfigManager(config_path=self.config_path)
        
        with pytest.raises(ConfigurationError) as exc_info:
            config_manager.load_config()
        
        assert "Configuration loading failed" in str(exc_info.value)

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 