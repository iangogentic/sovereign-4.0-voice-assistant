#!/usr/bin/env python3
"""
Integration Tests for Main Program
Tests the complete voice assistant integration in assistant/main.py
"""

import unittest
import asyncio
import tempfile
import yaml
import os
from pathlib import Path

# Import the main application components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assistant.main import SovereignAssistant, load_config

class TestMainIntegration(unittest.TestCase):
    """Test the main voice assistant application integration"""

    def setUp(self):
        """Set up test environment"""
        # Create temporary config
        self.test_config_dir = tempfile.mkdtemp()
        self.test_config_path = os.path.join(self.test_config_dir, "test_config.yaml")
        
        # Basic test configuration
        test_config = {
            'audio': {
                'sample_rate': 16000,
                'chunk_size': 1024,
                'channels': 1,
                'device_id': None
            },
            'stt': {
                'model': 'whisper-1',
                'language': 'en',
                'min_audio_length': 0.3,
                'max_audio_length': 30.0
            },
            'tts': {
                'voice': 'nova',
                'model': 'tts-1',
                'speed': 1.0
            },
            'pipeline': {
                'trigger_key': None,
                'max_recording_duration': 30.0,
                'latency_target': 0.8
            },
            'logging': {
                'level': 'WARNING'
            }
        }
        
        with open(self.test_config_path, 'w') as f:
            yaml.dump(test_config, f)

    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.test_config_dir)

    def test_config_loading(self):
        """Test configuration loading works correctly"""
        config = load_config(self.test_config_path)
        
        # Verify key configuration sections exist
        self.assertIn('audio', config)
        self.assertIn('stt', config)
        self.assertIn('tts', config)
        self.assertIn('pipeline', config)

    def test_app_initialization(self):
        """Test voice assistant app initializes correctly"""
        app = SovereignAssistant(config_path=self.test_config_path)
        
        # Verify app properties
        self.assertIsNotNone(app.config_path)
        self.assertEqual(app.config_path, self.test_config_path)
        self.assertIsNone(app.pipeline)  # Not initialized yet
        self.assertFalse(app.running)

    def test_component_creation_methods(self):
        """Test that component creation methods exist and work"""
        app = SovereignAssistant(config_path=self.test_config_path)
        app._load_config()
        
        # Test audio manager creation
        audio_manager = app._create_audio_manager()
        self.assertIsNotNone(audio_manager)
        
        # Test that methods exist (even if they fail without API keys)
        try:
            stt_service = app._create_stt_service()
            # If we get here, the method worked (API key might be missing)
        except Exception as e:
            # Method exists but might fail due to missing API key
            self.assertIn("api_key", str(e).lower())
        
        try:
            tts_service = app._create_tts_service()
            # If we get here, the method worked (API key might be missing)
        except Exception as e:
            # Method exists but might fail due to missing API key
            self.assertIn("api_key", str(e).lower())

    def test_text_processing(self):
        """Test the text processing functionality"""
        app = SovereignAssistant(config_path=self.test_config_path)
        
        # Test various responses
        response = app._process_user_text("hello")
        self.assertIn("Hello", response)
        
        response = app._process_user_text("how are you")
        self.assertIn("great", response)
        
        response = app._process_user_text("test")
        self.assertIn("Test successful", response)

    def test_signal_handlers_setup(self):
        """Test signal handlers can be set up"""
        app = SovereignAssistant(config_path=self.test_config_path)
        
        # Should not raise any errors
        try:
            app._setup_signal_handlers()
            signal_setup_successful = True
        except Exception as e:
            self.fail(f"Signal handler setup failed: {e}")
        
        self.assertTrue(signal_setup_successful)


class TestRegressionPrevention(unittest.TestCase):
    """Tests to ensure we didn't break existing functionality"""

    def test_critical_vad_threshold_fix_preserved(self):
        """Ensure the critical VAD threshold fix is preserved in core system"""
        from assistant.stt import STTConfig
        
        # Create default config
        config = STTConfig()
        
        # Verify the fix is in place - check silence_threshold (the actual attribute)
        self.assertEqual(config.silence_threshold, 0.001, 
                        "VAD threshold fix not preserved in core system!")

    def test_audio_config_intact(self):
        """Ensure audio configuration is not broken"""
        from assistant.audio import AudioConfig
        
        config = AudioConfig()
        
        # Verify key audio settings
        self.assertEqual(config.sample_rate, 16000)
        self.assertEqual(config.channels, 1)
        self.assertIsInstance(config.chunk_size, int)
        self.assertGreater(config.chunk_size, 0)

    def test_pipeline_components_available(self):
        """Ensure all pipeline components are still available"""
        try:
            from assistant.pipeline import VoiceAssistantPipeline, PipelineConfig
            from assistant.audio import AudioManager, AudioConfig
            from assistant.stt import WhisperSTTService, STTConfig
            from assistant.tts import OpenAITTSService, TTSConfig
            from assistant.monitoring import PerformanceMonitor
            from assistant.dashboard import ConsoleDashboard
            
            # All imports successful
            imports_successful = True
        except ImportError as e:
            self.fail(f"Core component import failed: {e}")
        
        self.assertTrue(imports_successful)

    def test_main_entry_points_available(self):
        """Test that main entry points are available"""
        from assistant.main import SovereignAssistant, load_config, parse_arguments
        
        # Test function availability
        self.assertTrue(callable(load_config))
        self.assertTrue(callable(parse_arguments))
        
        # Test class availability
        app = SovereignAssistant()
        self.assertIsNotNone(app)


class TestMainProgramFunctionality(unittest.TestCase):
    """Test core functionality without requiring API keys"""

    def test_config_validation(self):
        """Test configuration validation works"""
        from assistant.main import validate_environment
        
        # Should return boolean
        result = validate_environment()
        self.assertIsInstance(result, bool)

    def test_argument_parsing_function_exists(self):
        """Test that parse_arguments function exists and works"""
        import argparse
        from unittest.mock import patch
        
        # Mock sys.argv to provide empty arguments
        with patch('sys.argv', ['test']):
            from assistant.main import parse_arguments
            
            # Should be callable
            args = parse_arguments()
            self.assertIsNotNone(args)
            
            # Should have expected attributes
            self.assertTrue(hasattr(args, 'config'))
            self.assertTrue(hasattr(args, 'debug'))

    def test_logging_setup_function(self):
        """Test logging setup function works"""
        from assistant.main import setup_logging
        
        # Should not raise errors
        try:
            setup_logging(debug=True)
            setup_logging(debug=False)
            logging_setup_successful = True
        except Exception as e:
            self.fail(f"Logging setup failed: {e}")
        
        self.assertTrue(logging_setup_successful)


class TestAsyncFunctionality(unittest.TestCase):
    """Test async methods work correctly"""

    def setUp(self):
        """Set up test environment"""
        # Create temporary config
        self.test_config_dir = tempfile.mkdtemp()
        self.test_config_path = os.path.join(self.test_config_dir, "test_config.yaml")
        
        test_config = {
            'audio': {'sample_rate': 16000, 'chunk_size': 1024, 'channels': 1},
            'stt': {'model': 'whisper-1', 'language': 'en'},
            'tts': {'voice': 'nova', 'model': 'tts-1', 'speed': 1.0},
            'pipeline': {'trigger_key': None, 'max_recording_duration': 30.0},
            'logging': {'level': 'WARNING'}
        }
        
        with open(self.test_config_path, 'w') as f:
            yaml.dump(test_config, f)

    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.test_config_dir)

    def test_initialize_method_is_async(self):
        """Test that initialize method exists and is async"""
        app = SovereignAssistant(config_path=self.test_config_path)
        
        # Method should exist
        self.assertTrue(hasattr(app, 'initialize'))
        self.assertTrue(callable(app.initialize))
        
        # Method should be a coroutine function
        import asyncio
        self.assertTrue(asyncio.iscoroutinefunction(app.initialize))

    def test_async_methods_exist(self):
        """Test that async methods exist"""
        app = SovereignAssistant(config_path=self.test_config_path)
        
        # Check for async methods
        async_methods = ['initialize', 'start', 'stop', 'run']
        for method_name in async_methods:
            self.assertTrue(hasattr(app, method_name), f"Missing method: {method_name}")
            method = getattr(app, method_name)
            self.assertTrue(callable(method), f"Method not callable: {method_name}")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2) 