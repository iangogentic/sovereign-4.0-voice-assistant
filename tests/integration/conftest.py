#!/usr/bin/env python3
"""
Integration Test Configuration
Provides fixtures and utilities for comprehensive end-to-end testing
"""

import asyncio
import os
import pytest
import pytest_asyncio
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# Test environment setup
os.environ['TESTING'] = 'true'
os.environ['LOG_LEVEL'] = 'DEBUG'

# Import system components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from assistant.main import SovereignAssistant
from assistant.llm_router import LLMRouter
from assistant.audio import AudioManager
from assistant.stt import WhisperSTTService
from assistant.tts import OpenAITTSService
from assistant.monitoring import PerformanceMonitor
from assistant.shutdown_manager import ShutdownManager

# Pytest asyncio configuration
pytest_plugins = ('pytest_asyncio',)

class TestContext:
    """Test context manager for integration tests"""
    
    def __init__(self):
        self.assistant: Optional[SovereignAssistant] = None
        self.temp_dir: Optional[Path] = None
        self.mock_audio_data: Dict[str, bytes] = {}
        self.test_results: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, float] = {}
        
    def setup(self):
        """Setup test environment"""
        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="sovereign_test_"))
        
        # Setup test audio data
        self._setup_test_audio()
        
        # Initialize performance tracking
        self.performance_metrics = {
            'voice_latency': [],
            'memory_recall_accuracy': [],
            'ocr_accuracy': [],
            'error_recovery_time': []
        }
    
    def teardown(self):
        """Cleanup test environment"""
        if self.assistant:
            try:
                # Use asyncio.create_task for cleanup
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule cleanup for later
                    loop.create_task(self.assistant.stop())
                else:
                    loop.run_until_complete(self.assistant.stop())
            except:
                pass
        
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _setup_test_audio(self):
        """Setup mock audio data for testing"""
        # Simple sine wave for testing (440Hz, 1 second)
        import numpy as np
        
        sample_rate = 16000
        duration = 1.0
        frequency = 440
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * frequency * t)
        
        # Convert to 16-bit PCM
        audio_16bit = (audio_data * 32767).astype(np.int16)
        
        self.mock_audio_data = {
            'test_tone': audio_16bit.tobytes(),
            'silence': np.zeros(int(sample_rate * 0.5), dtype=np.int16).tobytes()
        }


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session"""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_context():
    """Global test context for integration tests"""
    context = TestContext()
    context.setup()
    yield context
    context.teardown()


@pytest_asyncio.fixture
async def sovereign_assistant(test_context):
    """Fixture providing a configured Sovereign Assistant instance"""
    # Mock audio input/output for testing
    with patch('sounddevice.rec') as mock_rec, \
         patch('sounddevice.play') as mock_play, \
         patch('sounddevice.wait') as mock_wait:
        
        # Configure mocks
        mock_rec.return_value = np.frombuffer(
            test_context.mock_audio_data['test_tone'], 
            dtype=np.int16
        ).reshape(-1, 1)
        mock_play.return_value = None
        mock_wait.return_value = None
        
        # Create assistant instance
        assistant = SovereignAssistant()
        
        # Initialize for testing
        await assistant.initialize()
        
        test_context.assistant = assistant
        yield assistant
        
        # Cleanup
        await assistant.stop()


@pytest_asyncio.fixture
async def llm_router():
    """Fixture providing an LLM router for testing"""
    router = LLMRouter()
    yield router
    await router.cleanup()


@pytest.fixture
def mock_whisper_stt():
    """Mock Whisper STT service for testing"""
    mock = AsyncMock()
    mock.transcribe_audio.return_value = "Hello, this is a test"
    return mock


@pytest.fixture
def mock_openai_tts():
    """Mock OpenAI TTS service for testing"""
    mock = AsyncMock()
    mock.synthesize_speech.return_value = b"mock_audio_data"
    return mock


@pytest.fixture
def mock_audio_manager():
    """Mock audio manager for testing"""
    mock = MagicMock()
    mock.record_audio = AsyncMock(return_value=b"mock_recorded_audio")
    mock.play_audio = AsyncMock()
    mock.is_available.return_value = True
    return mock


@pytest.fixture
def performance_monitor():
    """Performance monitor for testing"""
    monitor = PerformanceMonitor()
    yield monitor
    monitor.reset()


@pytest.fixture
def test_data_manager():
    """Test data manager for consistent test scenarios"""
    class TestDataManager:
        def __init__(self):
            self.voice_commands = {
                'simple': [
                    "Hello",
                    "What time is it?",
                    "How are you?",
                    "Thank you"
                ],
                'complex': [
                    "Can you help me debug this Python error about undefined variables?",
                    "Explain the differences between async and sync programming",
                    "Help me optimize this SQL query for better performance",
                    "What are the best practices for error handling in microservices?"
                ],
                'ide_related': [
                    "There's a syntax error on line 42",
                    "The module numpy is not found",
                    "Import error: cannot import name from module",
                    "Type error: expected string but got integer"
                ]
            }
            
            self.memory_queries = [
                "What did we discuss about Python yesterday?",
                "Remind me about the API endpoint we configured",
                "What was that error message from earlier?",
                "Show me the previous conversation about databases"
            ]
            
            self.error_scenarios = [
                {
                    'type': 'network_failure',
                    'description': 'Simulate network connectivity loss',
                    'expected_behavior': 'Fallback to offline mode'
                },
                {
                    'type': 'api_timeout',
                    'description': 'Simulate API timeout',
                    'expected_behavior': 'Retry with exponential backoff'
                },
                {
                    'type': 'audio_device_failure',
                    'description': 'Simulate audio device unavailable',
                    'expected_behavior': 'Graceful error message and fallback'
                }
            ]
        
        def get_test_audio_file(self, duration: float = 1.0) -> bytes:
            """Generate test audio data"""
            import numpy as np
            sample_rate = 16000
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = np.sin(2 * np.pi * 440 * t)  # 440Hz tone
            return (audio_data * 32767).astype(np.int16).tobytes()
        
        def get_mock_ide_error_image(self) -> bytes:
            """Generate mock IDE error dialog image"""
            # Create a simple test image (1x1 pixel PNG)
            import base64
            # Minimal PNG: 1x1 red pixel
            png_data = base64.b64decode(
                'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAGA8UkdVwAAAABJRU5ErkJggg=='
            )
            return png_data
    
    return TestDataManager()


@pytest.fixture
def test_environment():
    """Test environment configuration"""
    return {
        'voice_latency_threshold_cloud': 0.8,  # 800ms
        'voice_latency_threshold_offline': 1.5,  # 1.5s
        'memory_recall_accuracy_threshold': 0.85,  # 85%
        'ocr_accuracy_threshold': 0.80,  # 80%
        'stability_test_duration': 300,  # 5 minutes for testing (8 hours in production)
        'concurrent_users': 3,  # For load testing
        'error_recovery_timeout': 30  # seconds
    }


class TestMetrics:
    """Test metrics collection and validation"""
    
    def __init__(self):
        self.metrics = {}
        self.thresholds = {}
    
    def record_metric(self, name: str, value: float):
        """Record a test metric"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def set_threshold(self, name: str, threshold: float):
        """Set threshold for metric validation"""
        self.thresholds[name] = threshold
    
    def validate_metric(self, name: str) -> bool:
        """Validate metric against threshold"""
        if name not in self.metrics or name not in self.thresholds:
            return False
        
        values = self.metrics[name]
        threshold = self.thresholds[name]
        
        # For latency metrics: all values should be below threshold
        if 'latency' in name:
            return all(v <= threshold for v in values)
        
        # For accuracy metrics: average should be above threshold
        if 'accuracy' in name:
            return sum(values) / len(values) >= threshold
        
        return True
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        summary = {}
        for name, values in self.metrics.items():
            summary[name] = {
                'count': len(values),
                'min': min(values) if values else 0,
                'max': max(values) if values else 0,
                'avg': sum(values) / len(values) if values else 0,
                'threshold': self.thresholds.get(name),
                'passed': self.validate_metric(name)
            }
        return summary


@pytest.fixture
def test_metrics():
    """Test metrics collector"""
    return TestMetrics()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    # Register custom markers
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "stability: mark test as stability test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as security test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Add markers to tests based on path
    for item in items:
        # Mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Mark performance tests
        if "performance" in item.name or "latency" in item.name:
            item.add_marker(pytest.mark.performance)
        
        # Mark stability tests
        if "stability" in item.name or "endurance" in item.name:
            item.add_marker(pytest.mark.stability)
            item.add_marker(pytest.mark.slow)
        
        # Mark security tests
        if "security" in item.name or "privacy" in item.name:
            item.add_marker(pytest.mark.security) 