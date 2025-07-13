"""
Comprehensive Test Fixtures for Sovereign 4.0 Realtime API Testing

Provides essential test fixtures for:
- WebSocket mocking for Realtime API testing
- Audio stream simulation for voice processing
- API response mocking for OpenAI services
- Test data generation for conversation patterns
- Performance testing utilities
- Session management mocking

These fixtures support all test categories in the comprehensive test suite.
"""

import asyncio
import pytest
import pytest_asyncio
import json
import base64
import time
import tempfile
import wave
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncGenerator, Generator, Callable
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np

# Test data generation
from faker import Faker
import factory

# Audio simulation
try:
    import soundfile as sf
    import librosa
    from pydub import AudioSegment
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False

# WebSocket mocking
import websockets
from aioresponses import aioresponses
import responses

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from assistant.realtime_voice import RealtimeConfig, RealtimeVoiceService
from assistant.realtime_session_manager import SessionConfig, RealtimeSessionManager
from assistant.audio_stream_manager import AudioStreamManager, RealtimeAudioConfig
from assistant.config_manager import SovereignConfig, RealtimeAPIConfig


# =============================================================================
# Audio Test Fixtures
# =============================================================================

@dataclass
class AudioTestData:
    """Test audio data container"""
    audio_data: np.ndarray
    sample_rate: int
    duration_seconds: float
    format: str = "pcm16"
    channels: int = 1


class AudioDataGenerator:
    """Generate synthetic audio data for testing"""
    
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        
    def generate_silence(self, duration_seconds: float) -> AudioTestData:
        """Generate silent audio data"""
        samples = int(duration_seconds * self.sample_rate)
        audio_data = np.zeros(samples, dtype=np.float32)
        
        return AudioTestData(
            audio_data=audio_data,
            sample_rate=self.sample_rate,
            duration_seconds=duration_seconds
        )
    
    def generate_sine_wave(self, frequency: float, duration_seconds: float, amplitude: float = 0.5) -> AudioTestData:
        """Generate sine wave audio data"""
        samples = int(duration_seconds * self.sample_rate)
        t = np.linspace(0, duration_seconds, samples)
        audio_data = amplitude * np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        return AudioTestData(
            audio_data=audio_data,
            sample_rate=self.sample_rate,
            duration_seconds=duration_seconds
        )
    
    def generate_white_noise(self, duration_seconds: float, amplitude: float = 0.1) -> AudioTestData:
        """Generate white noise audio data"""
        samples = int(duration_seconds * self.sample_rate)
        audio_data = amplitude * np.random.randn(samples).astype(np.float32)
        
        return AudioTestData(
            audio_data=audio_data,
            sample_rate=self.sample_rate,
            duration_seconds=duration_seconds
        )
    
    def generate_speech_like(self, duration_seconds: float) -> AudioTestData:
        """Generate speech-like audio patterns"""
        samples = int(duration_seconds * self.sample_rate)
        t = np.linspace(0, duration_seconds, samples)
        
        # Combine multiple frequencies to simulate speech
        frequencies = [200, 300, 500, 800, 1200]  # Speech-like frequencies
        audio_data = np.zeros(samples)
        
        for freq in frequencies:
            amplitude = 0.1 + 0.05 * np.random.randn()
            audio_data += amplitude * np.sin(2 * np.pi * freq * t)
        
        # Add envelope to simulate speech patterns
        envelope = np.exp(-t / (duration_seconds * 0.3))
        audio_data *= envelope
        
        # Add some noise
        audio_data += 0.02 * np.random.randn(samples)
        
        return AudioTestData(
            audio_data=audio_data.astype(np.float32),
            sample_rate=self.sample_rate,
            duration_seconds=duration_seconds
        )
    
    def audio_to_base64(self, audio_data: AudioTestData) -> str:
        """Convert audio data to base64 string for API simulation"""
        # Convert float32 to int16 PCM
        audio_int16 = (audio_data.audio_data * 32767).astype(np.int16)
        return base64.b64encode(audio_int16.tobytes()).decode()


@pytest.fixture
def audio_generator():
    """Audio data generator fixture"""
    return AudioDataGenerator()


@pytest.fixture
def sample_audio_silence(audio_generator):
    """Generate 1 second of silence"""
    return audio_generator.generate_silence(1.0)


@pytest.fixture
def sample_audio_speech(audio_generator):
    """Generate 2 seconds of speech-like audio"""
    return audio_generator.generate_speech_like(2.0)


@pytest.fixture
def sample_audio_noise(audio_generator):
    """Generate 0.5 seconds of white noise"""
    return audio_generator.generate_white_noise(0.5)


# =============================================================================
# WebSocket Test Fixtures
# =============================================================================

class MockWebSocket:
    """Mock WebSocket connection for testing"""
    
    def __init__(self, responses: List[Dict[str, Any]] = None):
        self.responses = responses or []
        self.sent_messages = []
        self.is_open = True
        self.response_index = 0
        
    async def send(self, message: str):
        """Mock send method"""
        self.sent_messages.append(json.loads(message) if isinstance(message, str) else message)
        
    async def recv(self) -> str:
        """Mock receive method"""
        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            return json.dumps(response)
        else:
            # Simulate waiting for more messages
            await asyncio.sleep(0.1)
            return json.dumps({"type": "heartbeat"})
    
    async def close(self):
        """Mock close method"""
        self.is_open = False
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class RealtimeWebSocketMocker:
    """Mock OpenAI Realtime WebSocket responses"""
    
    def __init__(self):
        self.session_config = None
        self.conversation_items = []
        
    def create_session_created_response(self) -> Dict[str, Any]:
        """Create session.created response"""
        return {
            "type": "session.created",
            "session": {
                "id": f"sess_test_{int(time.time())}",
                "object": "realtime.session",
                "model": "gpt-4o-realtime-preview-2024-10-01",
                "modalities": ["text", "audio"],
                "instructions": "You are a helpful assistant.",
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"enabled": True},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 200
                },
                "tools": [],
                "tool_choice": "auto",
                "temperature": 0.8,
                "max_response_output_tokens": "inf"
            }
        }
    
    def create_audio_transcript_response(self, transcript: str) -> Dict[str, Any]:
        """Create input_audio_buffer.speech_started and transcript response"""
        return {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": f"item_test_{int(time.time())}",
            "content_index": 0,
            "transcript": transcript
        }
    
    def create_text_response(self, text: str) -> Dict[str, Any]:
        """Create text response from assistant"""
        return {
            "type": "response.content_part.added",
            "response_id": f"resp_test_{int(time.time())}",
            "part": {
                "type": "text",
                "text": text
            }
        }
    
    def create_audio_response(self, audio_base64: str) -> Dict[str, Any]:
        """Create audio response from assistant"""
        return {
            "type": "response.content_part.added",
            "response_id": f"resp_test_{int(time.time())}",
            "part": {
                "type": "audio",
                "audio": audio_base64
            }
        }
    
    def create_error_response(self, error_type: str, message: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "type": "error",
            "error": {
                "type": error_type,
                "code": "test_error",
                "message": message
            }
        }


@pytest.fixture
def websocket_mocker():
    """WebSocket response mocker fixture"""
    return RealtimeWebSocketMocker()


@pytest.fixture
def mock_websocket_connection(websocket_mocker):
    """Mock WebSocket connection with typical responses"""
    responses = [
        websocket_mocker.create_session_created_response(),
        websocket_mocker.create_audio_transcript_response("Hello, how can I help you?"),
        websocket_mocker.create_text_response("I'm here to assist you with any questions you might have."),
    ]
    return MockWebSocket(responses)


# =============================================================================
# API Response Test Fixtures
# =============================================================================

class OpenAIAPIMocker:
    """Mock OpenAI API responses"""
    
    @staticmethod
    def mock_stt_response(transcript: str, confidence: float = 0.95) -> Dict[str, Any]:
        """Mock Whisper STT response"""
        return {
            "text": transcript,
            "segments": [
                {
                    "id": 0,
                    "seek": 0,
                    "start": 0.0,
                    "end": len(transcript.split()) * 0.5,
                    "text": transcript,
                    "tokens": list(range(len(transcript.split()))),
                    "temperature": 0.0,
                    "avg_logprob": -0.5,
                    "compression_ratio": 1.2,
                    "no_speech_prob": 1.0 - confidence
                }
            ],
            "language": "en"
        }
    
    @staticmethod
    def mock_chat_completion_response(content: str, model: str = "gpt-4") -> Dict[str, Any]:
        """Mock Chat Completion response"""
        return {
            "id": f"chatcmpl-test{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": len(content.split()),
                "total_tokens": 20 + len(content.split())
            }
        }
    
    @staticmethod
    def mock_tts_response() -> bytes:
        """Mock TTS audio response"""
        # Generate simple sine wave as mock audio
        duration = 2.0
        sample_rate = 22050
        frequency = 440.0
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        # Convert to int16 PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()


@pytest.fixture
def openai_api_mocker():
    """OpenAI API response mocker fixture"""
    return OpenAIAPIMocker()


@pytest.fixture
def mock_openai_responses():
    """Setup mock responses for OpenAI API calls"""
    with responses.RequestsMock() as rsps:
        # Mock Whisper STT
        rsps.add(
            responses.POST,
            "https://api.openai.com/v1/audio/transcriptions",
            json=OpenAIAPIMocker.mock_stt_response("Hello, this is a test transcription."),
            status=200
        )
        
        # Mock Chat Completion
        rsps.add(
            responses.POST,
            "https://api.openai.com/v1/chat/completions",
            json=OpenAIAPIMocker.mock_chat_completion_response("Hello! I'm an AI assistant. How can I help you today?"),
            status=200
        )
        
        # Mock TTS
        rsps.add(
            responses.POST,
            "https://api.openai.com/v1/audio/speech",
            body=OpenAIAPIMocker.mock_tts_response(),
            status=200,
            content_type="audio/mpeg"
        )
        
        yield rsps


# =============================================================================
# Test Data Generation
# =============================================================================

class ConversationPatternFactory(factory.Factory):
    """Factory for generating conversation test patterns"""
    
    class Meta:
        model = dict
    
    user_message = factory.Faker('text', max_nb_chars=100)
    assistant_response = factory.Faker('text', max_nb_chars=200)
    timestamp = factory.Faker('date_time_this_year')
    confidence = factory.Faker('pyfloat', min_value=0.8, max_value=1.0)
    latency_ms = factory.Faker('pyfloat', min_value=100, max_value=500)


class TestDataGenerator:
    """Generate comprehensive test data patterns"""
    
    def __init__(self):
        self.fake = Faker()
        
    def generate_conversation_flows(self, count: int = 10) -> List[Dict[str, Any]]:
        """Generate realistic conversation flows"""
        conversations = []
        
        conversation_types = [
            "technical_support",
            "general_questions",
            "coding_help",
            "creative_writing",
            "data_analysis"
        ]
        
        for _ in range(count):
            conversation_type = self.fake.random_element(conversation_types)
            
            conversation = {
                "id": f"conv_{self.fake.uuid4()}",
                "type": conversation_type,
                "turns": self._generate_conversation_turns(conversation_type),
                "start_time": self.fake.date_time_this_year(),
                "duration_seconds": self.fake.random_int(30, 300),
                "user_satisfaction": self.fake.random_int(1, 5)
            }
            
            conversations.append(conversation)
        
        return conversations
    
    def _generate_conversation_turns(self, conversation_type: str) -> List[Dict[str, Any]]:
        """Generate conversation turns based on type"""
        turn_count = self.fake.random_int(2, 8)
        turns = []
        
        for i in range(turn_count):
            user_message = self._generate_user_message(conversation_type)
            assistant_response = self._generate_assistant_response(conversation_type, user_message)
            
            turn = {
                "turn_id": i + 1,
                "user_message": user_message,
                "assistant_response": assistant_response,
                "user_audio_duration": self.fake.pyfloat(min_value=1.0, max_value=10.0),
                "response_latency_ms": self.fake.pyfloat(min_value=200, max_value=800),
                "stt_confidence": self.fake.pyfloat(min_value=0.85, max_value=0.99),
                "conversation_quality": self.fake.pyfloat(min_value=0.7, max_value=0.95)
            }
            
            turns.append(turn)
        
        return turns
    
    def _generate_user_message(self, conversation_type: str) -> str:
        """Generate user message based on conversation type"""
        message_templates = {
            "technical_support": [
                "I'm having trouble with my computer, it keeps crashing.",
                "How do I fix this error message?",
                "My application won't start properly.",
                "Can you help me troubleshoot this network issue?"
            ],
            "general_questions": [
                "What's the weather like today?",
                "Can you explain how photosynthesis works?",
                "What are some good restaurants nearby?",
                "How do I cook pasta properly?"
            ],
            "coding_help": [
                "How do I fix this Python error?",
                "What's the best way to optimize this algorithm?",
                "Can you explain how async/await works?",
                "Help me debug this JavaScript function."
            ],
            "creative_writing": [
                "Help me write a story about space exploration.",
                "Can you suggest some character names?",
                "What's a good plot twist for my novel?",
                "How do I improve my writing style?"
            ],
            "data_analysis": [
                "How do I analyze this dataset?",
                "What's the best visualization for this data?",
                "Can you help me interpret these statistics?",
                "How do I clean this messy data?"
            ]
        }
        
        templates = message_templates.get(conversation_type, message_templates["general_questions"])
        return self.fake.random_element(templates)
    
    def _generate_assistant_response(self, conversation_type: str, user_message: str) -> str:
        """Generate assistant response based on context"""
        response_templates = {
            "technical_support": [
                "Let me help you troubleshoot that issue. First, let's check...",
                "That error usually occurs when... Here's how to fix it:",
                "I can help you resolve this. Try these steps:",
                "This is a common problem. Here's the solution:"
            ],
            "general_questions": [
                "Based on what I know, here's the answer...",
                "That's an interesting question. Let me explain...",
                "I'd be happy to help with that. Here's what you need to know:",
                "Great question! The answer is..."
            ],
            "coding_help": [
                "Looking at your code, I can see the issue. Here's the fix:",
                "That's a common programming pattern. Let me show you:",
                "For that error, you'll want to try this approach:",
                "I can help you optimize that. Here's a better way:"
            ],
            "creative_writing": [
                "For your story, I suggest exploring this theme...",
                "Here are some creative ideas you might consider:",
                "That's a fascinating concept. You could develop it by...",
                "For your writing, try this technique:"
            ],
            "data_analysis": [
                "For that dataset, I recommend this analysis approach:",
                "The best way to visualize that data would be...",
                "Looking at those statistics, the key insight is...",
                "To clean that data, start with these steps:"
            ]
        }
        
        templates = response_templates.get(conversation_type, response_templates["general_questions"])
        return self.fake.random_element(templates)
    
    def generate_edge_cases(self) -> List[Dict[str, Any]]:
        """Generate edge case scenarios for testing"""
        edge_cases = [
            {
                "scenario": "very_short_input",
                "user_input": "Hi",
                "expected_behavior": "handle_gracefully"
            },
            {
                "scenario": "very_long_input",
                "user_input": " ".join(self.fake.words(200)),
                "expected_behavior": "truncate_or_chunk"
            },
            {
                "scenario": "silence",
                "user_input": "",
                "expected_behavior": "prompt_for_input"
            },
            {
                "scenario": "background_noise",
                "user_input": "Hello" + "".join(self.fake.random_letters(50)),
                "expected_behavior": "filter_noise"
            },
            {
                "scenario": "multiple_languages",
                "user_input": "Hello, comment allez-vous? ¿Cómo estás?",
                "expected_behavior": "detect_language"
            },
            {
                "scenario": "technical_jargon",
                "user_input": "Implement OAuth2 JWT authentication with RBAC",
                "expected_behavior": "understand_technical_terms"
            }
        ]
        
        return edge_cases


@pytest.fixture
def test_data_generator():
    """Test data generator fixture"""
    return TestDataGenerator()


@pytest.fixture
def sample_conversation_flows(test_data_generator):
    """Generate sample conversation flows for testing"""
    return test_data_generator.generate_conversation_flows(5)


@pytest.fixture
def edge_case_scenarios(test_data_generator):
    """Generate edge case scenarios"""
    return test_data_generator.generate_edge_cases()


# =============================================================================
# Configuration Test Fixtures
# =============================================================================

@pytest.fixture
def test_realtime_config():
    """Standard Realtime API configuration for testing"""
    return RealtimeConfig(
        api_key="test_key_12345",
        model="gpt-4o-realtime-preview-2024-10-01",
        voice="alloy",
        sample_rate=24000,
        max_response_tokens=1000,
        temperature=0.8,
        enable_vad=True,
        vad_threshold=0.5,
        silence_duration_ms=200,
        context_window_tokens=8000
    )


@pytest.fixture
def test_audio_config():
    """Standard audio configuration for testing"""
    return RealtimeAudioConfig(
        sample_rate=24000,
        input_chunk_size=1024,
        output_chunk_size=1024,
        buffer_duration=0.1,
        target_latency_ms=50.0
    )


@pytest.fixture
def test_session_config():
    """Standard session configuration for testing"""
    return SessionConfig(
        database_path=":memory:",  # Use in-memory database for tests
        session_timeout_minutes=30,
        max_concurrent_sessions=10,
        max_recovery_attempts=3
    )


# =============================================================================
# Performance Testing Fixtures
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Container for performance test metrics"""
    latency_ms: float
    throughput_rps: float
    memory_usage_mb: float
    cpu_usage_percent: float
    error_count: int
    success_count: int


class PerformanceCollector:
    """Collect performance metrics during testing"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.start_time: Optional[float] = None
        
    def start_collection(self):
        """Start collecting performance metrics"""
        self.start_time = time.time()
        self.metrics.clear()
        
    def record_metric(self, latency_ms: float, success: bool = True):
        """Record a single performance metric"""
        import psutil
        
        metric = PerformanceMetrics(
            latency_ms=latency_ms,
            throughput_rps=1.0,  # Will be calculated later
            memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
            cpu_usage_percent=psutil.cpu_percent(),
            error_count=0 if success else 1,
            success_count=1 if success else 0
        )
        
        self.metrics.append(metric)
        
    def get_summary(self) -> Dict[str, float]:
        """Get performance summary statistics"""
        if not self.metrics:
            return {}
        
        latencies = [m.latency_ms for m in self.metrics]
        
        total_duration = time.time() - self.start_time if self.start_time else 1.0
        total_requests = len(self.metrics)
        
        return {
            "avg_latency_ms": np.mean(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "throughput_rps": total_requests / total_duration,
            "error_rate": sum(m.error_count for m in self.metrics) / total_requests,
            "avg_memory_mb": np.mean([m.memory_usage_mb for m in self.metrics]),
            "avg_cpu_percent": np.mean([m.cpu_usage_percent for m in self.metrics])
        }


@pytest.fixture
def performance_collector():
    """Performance metrics collector fixture"""
    return PerformanceCollector()


# =============================================================================
# Session Management Test Fixtures  
# =============================================================================

@pytest.fixture
async def mock_session_manager():
    """Mock session manager for testing"""
    mock_manager = AsyncMock(spec=RealtimeSessionManager)
    
    # Setup common mock responses
    mock_manager.create_session.return_value = {
        "session_id": "test_session_123",
        "created_at": datetime.now(),
        "status": "active"
    }
    
    mock_manager.get_session.return_value = {
        "session_id": "test_session_123",
        "status": "active",
        "conversation_items": []
    }
    
    mock_manager.close_session.return_value = True
    
    return mock_manager


# =============================================================================
# Cleanup Fixtures
# =============================================================================

@pytest.fixture
def temp_test_dir():
    """Temporary directory for test files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(autouse=True)
def cleanup_test_environment():
    """Automatically cleanup test environment after each test"""
    yield
    
    # Cleanup any running asyncio tasks
    try:
        loop = asyncio.get_event_loop()
        tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
        for task in tasks:
            task.cancel()
    except RuntimeError:
        pass  # No event loop running
    
    # Force garbage collection
    import gc
    gc.collect()


# =============================================================================
# Export commonly used fixtures
# =============================================================================

__all__ = [
    'audio_generator', 'sample_audio_silence', 'sample_audio_speech', 'sample_audio_noise',
    'websocket_mocker', 'mock_websocket_connection',
    'openai_api_mocker', 'mock_openai_responses', 
    'test_data_generator', 'sample_conversation_flows', 'edge_case_scenarios',
    'test_realtime_config', 'test_audio_config', 'test_session_config',
    'performance_collector', 'mock_session_manager',
    'temp_test_dir', 'cleanup_test_environment'
] 