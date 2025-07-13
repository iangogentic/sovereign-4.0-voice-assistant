"""
Unit tests for ConfigManager with Realtime API Extensions

Tests the extended configuration system that includes OpenAI Realtime API configuration,
environment variable overrides, validation, and hot-reload functionality.
"""

import pytest
import os
import tempfile
import yaml
import time
from unittest.mock import patch, Mock
from typing import Dict, Any

from assistant.config_manager import (
    ConfigManager, SovereignConfig, RealtimeAPIConfig, 
    EnvironmentType, ConfigurationError,
    get_config_manager, get_config, reload_config
)


class TestRealtimeAPIConfig:
    """Test RealtimeAPIConfig dataclass and default values"""
    
    def test_default_config(self):
        """Test default RealtimeAPIConfig values"""
        config = RealtimeAPIConfig()
        
        # Core settings
        assert config.enabled is False
        assert config.model == "gpt-4o-realtime-preview-2024-10-01"
        assert config.voice == "alloy"
        assert config.modalities == ["text", "audio"]
        assert "Sovereign" in config.instructions
        
        # Audio format settings
        assert config.input_audio_format == "pcm16"
        assert config.output_audio_format == "pcm16"
        assert config.sample_rate == 24000
        
        # VAD settings
        assert config.turn_detection["type"] == "server_vad"
        assert config.turn_detection["threshold"] == 0.5
        
        # Performance settings
        assert config.temperature == 0.8
        assert config.connection_timeout == 30.0
        assert config.max_reconnect_attempts == 5
        
        # Session settings
        assert config.session_timeout_minutes == 30
        assert config.max_concurrent_sessions == 10
        
        # Cost optimization
        assert config.enable_cost_monitoring is True
        assert config.max_cost_per_session == 1.0
        assert config.cost_alert_threshold == 0.8
        
        # Fallback settings
        assert config.fallback_on_errors is True
        assert config.max_latency_threshold_ms == 500.0

    def test_config_serialization(self):
        """Test RealtimeAPIConfig can be serialized to dict"""
        config = RealtimeAPIConfig(
            enabled=True,
            voice="nova",
            temperature=0.9,
            max_cost_per_session=2.0
        )
        
        # Test that the config can be converted to dict-like format
        assert config.enabled is True
        assert config.voice == "nova"
        assert config.temperature == 0.9
        assert config.max_cost_per_session == 2.0


class TestConfigManagerRealtimeAPI:
    """Test ConfigManager with Realtime API configuration"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
    
    def teardown_method(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_config(self, realtime_config: Dict[str, Any] = None) -> None:
        """Create a test configuration file"""
        config_data = {
            "environment": "development",
            "features": {
                "realtime_api": True if realtime_config else False
            },
            "development": {
                "mock_apis": True
            }
        }
        
        if realtime_config:
            config_data["realtime_api"] = realtime_config
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f)
    
    def test_load_config_with_realtime_api(self):
        """Test loading configuration with Realtime API section"""
        realtime_config = {
            "enabled": True,
            "model": "gpt-4o-realtime-preview-2024-10-01",
            "voice": "echo",
            "temperature": 0.7,
            "sample_rate": 24000,
            "max_cost_per_session": 1.5,
            "fallback_on_high_latency": False
        }
        
        self.create_test_config(realtime_config)
        
        manager = ConfigManager(config_path=self.config_path)
        config = manager.load_config()
        
        # Verify RealtimeAPIConfig is loaded correctly
        assert isinstance(config.realtime_api, RealtimeAPIConfig)
        assert config.realtime_api.enabled is True
        assert config.realtime_api.voice == "echo"
        assert config.realtime_api.temperature == 0.7
        assert config.realtime_api.max_cost_per_session == 1.5
        assert config.realtime_api.fallback_on_high_latency is False
        
        # Verify defaults are preserved for non-specified fields
        assert config.realtime_api.sample_rate == 24000
        assert config.realtime_api.max_concurrent_sessions == 10
    
    def test_load_config_without_realtime_api(self):
        """Test loading configuration without Realtime API section"""
        self.create_test_config()
        
        manager = ConfigManager(config_path=self.config_path)
        config = manager.load_config()
        
        # Verify RealtimeAPIConfig uses defaults
        assert isinstance(config.realtime_api, RealtimeAPIConfig)
        assert config.realtime_api.enabled is False
        assert config.realtime_api.model == "gpt-4o-realtime-preview-2024-10-01"
        assert config.realtime_api.voice == "alloy"
    
    def test_environment_variable_overrides(self):
        """Test Realtime API environment variable overrides"""
        self.create_test_config({"enabled": False})
        
        env_vars = {
            "REALTIME_API_ENABLED": "true",
            "REALTIME_API_VOICE": "nova",
            "REALTIME_API_TEMPERATURE": "0.9",
            "REALTIME_API_MAX_COST_PER_SESSION": "2.5",
            "REALTIME_API_FALLBACK_ON_ERRORS": "false",
            "REALTIME_API_MAX_LATENCY_THRESHOLD_MS": "750.0"
        }
        
        with patch.dict(os.environ, env_vars):
            manager = ConfigManager(config_path=self.config_path)
            config = manager.load_config()
        
        # Verify environment variables override config file
        assert config.realtime_api.enabled is True
        assert config.realtime_api.voice == "nova"
        assert config.realtime_api.temperature == 0.9
        assert config.realtime_api.max_cost_per_session == 2.5
        assert config.realtime_api.fallback_on_errors is False
        assert config.realtime_api.max_latency_threshold_ms == 750.0
    
    def test_invalid_environment_variables(self):
        """Test handling of invalid environment variable values"""
        self.create_test_config()
        
        env_vars = {
            "REALTIME_API_TEMPERATURE": "invalid_float",
            "REALTIME_API_MAX_RECONNECT_ATTEMPTS": "not_an_int",
            "REALTIME_API_MAX_COST_PER_SESSION": "NaN"
        }
        
        with patch.dict(os.environ, env_vars):
            manager = ConfigManager(config_path=self.config_path)
            # Should load successfully and ignore invalid values
            config = manager.load_config()
            
            # Verify defaults are used when invalid values are provided
            assert config.realtime_api.temperature == 0.8  # default
            assert config.realtime_api.max_reconnect_attempts == 5  # default
            assert config.realtime_api.max_cost_per_session == 1.0  # default


class TestRealtimeAPIValidation:
    """Test validation for Realtime API configuration"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
    
    def teardown_method(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_config_with_realtime_api(self, realtime_config: Dict[str, Any]) -> ConfigManager:
        """Create a config with Realtime API enabled and specified settings"""
        config_data = {
            "environment": "development",
            "development": {"mock_apis": True},
            "features": {"realtime_api": True},
            "realtime_api": realtime_config
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        return ConfigManager(config_path=self.config_path)
    
    def test_valid_realtime_api_config(self):
        """Test that valid Realtime API configuration passes validation"""
        realtime_config = {
            "enabled": True,
            "model": "gpt-4o-realtime-preview-2024-10-01",
            "voice": "alloy",
            "temperature": 0.8,
            "sample_rate": 24000,
            "max_cost_per_session": 1.0,
            "turn_detection": {
                "threshold": 0.5
            }
        }
        
        manager = self.create_config_with_realtime_api(realtime_config)
        # Should not raise any exceptions
        config = manager.load_config()
        assert config.realtime_api.enabled is True
    
    def test_invalid_voice_validation(self):
        """Test validation of invalid voice selection"""
        realtime_config = {
            "enabled": True,
            "voice": "invalid_voice"
        }
        
        manager = self.create_config_with_realtime_api(realtime_config)
        
        with pytest.raises(ConfigurationError) as exc_info:
            manager.load_config()
        
        assert "Invalid Realtime API voice: invalid_voice" in str(exc_info.value)
    
    def test_invalid_audio_format_validation(self):
        """Test validation of invalid audio formats"""
        realtime_config = {
            "enabled": True,
            "input_audio_format": "mp3"  # Invalid, should be pcm16
        }
        
        manager = self.create_config_with_realtime_api(realtime_config)
        
        with pytest.raises(ConfigurationError) as exc_info:
            manager.load_config()
        
        assert "Invalid input audio format: mp3" in str(exc_info.value)
    
    def test_invalid_sample_rate_validation(self):
        """Test validation of invalid sample rate"""
        realtime_config = {
            "enabled": True,
            "sample_rate": 16000  # Invalid, should be 24000 for Realtime API
        }
        
        manager = self.create_config_with_realtime_api(realtime_config)
        
        with pytest.raises(ConfigurationError) as exc_info:
            manager.load_config()
        
        assert "Realtime API requires 24kHz sample rate" in str(exc_info.value)
    
    def test_invalid_temperature_validation(self):
        """Test validation of invalid temperature values"""
        realtime_config = {
            "enabled": True,
            "temperature": 3.0  # Invalid, should be 0.0-2.0
        }
        
        manager = self.create_config_with_realtime_api(realtime_config)
        
        with pytest.raises(ConfigurationError) as exc_info:
            manager.load_config()
        
        assert "Realtime API temperature must be between 0.0 and 2.0" in str(exc_info.value)
    
    def test_invalid_cost_threshold_validation(self):
        """Test validation of invalid cost alert threshold"""
        realtime_config = {
            "enabled": True,
            "cost_alert_threshold": 1.5  # Invalid, should be 0.0-1.0
        }
        
        manager = self.create_config_with_realtime_api(realtime_config)
        
        with pytest.raises(ConfigurationError) as exc_info:
            manager.load_config()
        
        assert "Realtime API cost alert threshold must be between 0.0 and 1.0" in str(exc_info.value)
    
    def test_invalid_vad_threshold_validation(self):
        """Test validation of invalid VAD threshold"""
        realtime_config = {
            "enabled": True,
            "turn_detection": {
                "threshold": 1.5  # Invalid, should be 0.0-1.0
            }
        }
        
        manager = self.create_config_with_realtime_api(realtime_config)
        
        with pytest.raises(ConfigurationError) as exc_info:
            manager.load_config()
        
        assert "Realtime API VAD threshold must be between 0.0 and 1.0" in str(exc_info.value)


class TestConfigManagerSaveLoad:
    """Test configuration saving and loading with Realtime API"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
    
    def teardown_method(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_load_realtime_config(self):
        """Test saving and loading configuration with Realtime API settings"""
        # Create a config with custom Realtime API settings
        manager = ConfigManager(config_path=self.config_path)
        config = manager.load_config()
        
        # Modify Realtime API settings
        config.realtime_api.enabled = True
        config.realtime_api.voice = "nova"
        config.realtime_api.temperature = 0.9
        config.realtime_api.max_cost_per_session = 2.0
        config.features["realtime_api"] = True
        
        # Save the configuration
        manager.save_config(config)
        
        # Load with a new manager instance
        new_manager = ConfigManager(config_path=self.config_path)
        loaded_config = new_manager.load_config()
        
        # Verify Realtime API settings are preserved
        assert loaded_config.realtime_api.enabled is True
        assert loaded_config.realtime_api.voice == "nova"
        assert loaded_config.realtime_api.temperature == 0.9
        assert loaded_config.realtime_api.max_cost_per_session == 2.0
        assert loaded_config.features["realtime_api"] is True


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    def test_production_realtime_api_setup(self):
        """Test production-like Realtime API configuration"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                "environment": "production",
                "features": {"realtime_api": True},
                "development": {"mock_apis": False},
                "realtime_api": {
                    "enabled": True,
                    "voice": "alloy",
                    "temperature": 0.7,
                    "max_cost_per_session": 0.5,
                    "enable_cost_monitoring": True,
                    "fallback_on_high_latency": True,
                    "max_latency_threshold_ms": 300.0,
                    "session_timeout_minutes": 15
                }
            }
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            # Test with OpenAI API key
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                manager = ConfigManager(config_path=config_path, 
                                      environment=EnvironmentType.PRODUCTION)
                config = manager.load_config()
                
                # Verify production settings
                assert config.environment == EnvironmentType.PRODUCTION
                assert config.realtime_api.enabled is True
                assert config.realtime_api.max_cost_per_session == 0.5
                assert config.realtime_api.max_latency_threshold_ms == 300.0
                assert config.development.mock_apis is False
        finally:
            os.unlink(config_path)
    
    def test_hot_reload_with_realtime_api(self):
        """Test hot reload functionality with Realtime API configuration"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            initial_config = {
                "development": {"mock_apis": True},
                "realtime_api": {"enabled": False, "voice": "alloy"}
            }
            yaml.dump(initial_config, f)
            config_path = f.name
        
        try:
            manager = ConfigManager(config_path=config_path)
            config = manager.load_config()
            
            # Verify initial settings
            assert config.realtime_api.enabled is False
            assert config.realtime_api.voice == "alloy"
            
            # Modify the config file
            time.sleep(0.1)  # Ensure file modification time changes
            updated_config = {
                "development": {"mock_apis": True},
                "realtime_api": {"enabled": True, "voice": "nova"}
            }
            with open(config_path, 'w') as f:
                yaml.dump(updated_config, f)
            
            # Reload configuration
            reloaded_config = manager.reload_config()
            
            # Verify changes are applied
            assert reloaded_config.realtime_api.enabled is True
            assert reloaded_config.realtime_api.voice == "nova"
        finally:
            os.unlink(config_path)


if __name__ == "__main__":
    pytest.main([__file__]) 