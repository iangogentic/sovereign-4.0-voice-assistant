"""
Tests for the Speech-to-Text (STT) module
"""

import pytest
import asyncio
import numpy as np
import sys
import io
import wave
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock

# Add the parent directory to sys.path to import assistant
sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant.stt import (
    STTConfig,
    STTResult,
    AudioProcessor,
    WhisperSTTService,
    create_whisper_stt_service
)


class TestSTTConfig:
    """Test STTConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = STTConfig()
        
        assert config.model == "whisper-1"
        assert config.language == "en"
        assert config.temperature == 0.0
        assert config.response_format == "json"
        assert config.min_audio_length == 0.5
        assert config.max_audio_length == 30.0
        assert config.silence_threshold == 0.01
        assert config.timeout == 10
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.vad_enabled is True
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = STTConfig(
            model="whisper-large",
            language="es",
            temperature=0.5,
            min_audio_length=1.0,
            max_retries=5,
            vad_enabled=False
        )
        
        assert config.model == "whisper-large"
        assert config.language == "es"
        assert config.temperature == 0.5
        assert config.min_audio_length == 1.0
        assert config.max_retries == 5
        assert config.vad_enabled is False


class TestSTTResult:
    """Test STTResult dataclass"""
    
    def test_stt_result_creation(self):
        """Test STTResult creation"""
        result = STTResult(
            text="Hello world",
            confidence=0.95,
            language="en",
            duration=2.5,
            processing_time=0.8
        )
        
        assert result.text == "Hello world"
        assert result.confidence == 0.95
        assert result.language == "en"
        assert result.duration == 2.5
        assert result.processing_time == 0.8
        assert result.segments == []
    
    def test_stt_result_with_segments(self):
        """Test STTResult with segments"""
        segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]
        result = STTResult(
            text="Hello world",
            confidence=0.95,
            language="en",
            duration=2.5,
            processing_time=0.8,
            segments=segments
        )
        
        assert result.segments == segments


class TestAudioProcessor:
    """Test AudioProcessor class"""
    
    def test_audio_processor_creation(self):
        """Test AudioProcessor creation"""
        config = STTConfig()
        processor = AudioProcessor(config)
        
        assert processor.config == config
    
    def test_convert_to_wav(self):
        """Test audio conversion to WAV format"""
        config = STTConfig()
        processor = AudioProcessor(config)
        
        # Create test audio data (1 second of silence)
        sample_rate = 16000
        duration = 1.0
        samples = int(sample_rate * duration)
        audio_data = np.zeros(samples, dtype=np.int16).tobytes()
        
        wav_data = processor.convert_to_wav(audio_data, sample_rate)
        
        assert wav_data is not None
        assert len(wav_data) > len(audio_data)  # WAV has headers
        
        # Verify it's valid WAV data
        wav_buffer = io.BytesIO(wav_data)
        with wave.open(wav_buffer, 'rb') as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getsampwidth() == 2
            assert wav_file.getframerate() == sample_rate
    
    def test_calculate_audio_duration(self):
        """Test audio duration calculation"""
        config = STTConfig()
        processor = AudioProcessor(config)
        
        # Create 2 seconds of audio data
        sample_rate = 16000
        duration = 2.0
        samples = int(sample_rate * duration)
        audio_data = np.zeros(samples, dtype=np.int16).tobytes()
        
        calculated_duration = processor.calculate_audio_duration(audio_data, sample_rate)
        
        assert abs(calculated_duration - duration) < 0.01
    
    def test_detect_voice_activity_silence(self):
        """Test voice activity detection with silence"""
        config = STTConfig(vad_enabled=True, silence_threshold=0.01)
        processor = AudioProcessor(config)
        
        # Create silence
        sample_rate = 16000
        samples = int(sample_rate * 0.5)
        audio_data = np.zeros(samples, dtype=np.int16).tobytes()
        
        has_voice = processor.detect_voice_activity(audio_data)
        
        assert has_voice == False
    
    def test_detect_voice_activity_with_sound(self):
        """Test voice activity detection with sound"""
        config = STTConfig(vad_enabled=True, silence_threshold=0.01)
        processor = AudioProcessor(config)
        
        # Create audio with sound
        sample_rate = 16000
        samples = int(sample_rate * 0.5)
        audio_data = np.ones(samples, dtype=np.int16) * 1000  # Amplitude 1000
        
        has_voice = processor.detect_voice_activity(audio_data.tobytes())
        
        assert has_voice == True
    
    def test_detect_voice_activity_disabled(self):
        """Test voice activity detection when disabled"""
        config = STTConfig(vad_enabled=False)
        processor = AudioProcessor(config)
        
        # Create silence
        sample_rate = 16000
        samples = int(sample_rate * 0.5)
        audio_data = np.zeros(samples, dtype=np.int16).tobytes()
        
        has_voice = processor.detect_voice_activity(audio_data)
        
        assert has_voice is True  # Should return True when disabled
    
    def test_trim_silence(self):
        """Test silence trimming"""
        config = STTConfig(silence_threshold=0.01)
        processor = AudioProcessor(config)
        
        # Create audio: silence + sound + silence
        sample_rate = 16000
        silence_samples = int(sample_rate * 0.5)
        sound_samples = int(sample_rate * 1.0)
        
        silence = np.zeros(silence_samples, dtype=np.int16)
        sound = np.ones(sound_samples, dtype=np.int16) * 1000
        
        audio_data = np.concatenate([silence, sound, silence]).tobytes()
        
        trimmed = processor.trim_silence(audio_data, sample_rate)
        
        # Trimmed should be shorter than original
        assert len(trimmed) < len(audio_data)
        
        # Calculate expected length (should be close to sound_samples * 2 bytes)
        expected_length = sound_samples * 2
        assert abs(len(trimmed) - expected_length) < 100  # Allow small difference


class TestWhisperSTTService:
    """Test WhisperSTTService class"""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for testing"""
        with patch('assistant.stt.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # Mock transcription response
            mock_response = Mock()
            mock_response.text = "Hello world"
            mock_response.language = "en"
            mock_response.segments = []
            
            mock_client.audio.transcriptions.create.return_value = mock_response
            
            yield mock_client
    
    def test_service_creation(self, mock_openai_client):
        """Test WhisperSTTService creation"""
        config = STTConfig()
        service = WhisperSTTService(config, "test-api-key")
        
        assert service.config == config
        assert service.is_initialized is False
        assert service.total_requests == 0
        assert service.successful_requests == 0
        assert service.failed_requests == 0
    
    def test_service_initialization(self, mock_openai_client):
        """Test service initialization"""
        config = STTConfig()
        service = WhisperSTTService(config, "test-api-key")
        
        result = service.initialize()
        
        assert result is True
        assert service.is_initialized is True
    
    def test_service_initialization_failure(self):
        """Test service initialization failure"""
        config = STTConfig()
        service = WhisperSTTService(config, "test-api-key")
        
        # Mock the client to fail during initialization
        with patch.object(service.client.audio.transcriptions, 'create') as mock_create:
            mock_create.side_effect = Exception("API Error")
            
            result = service.initialize()
            
            # Service should still initialize successfully even if API test fails
            assert result is True
            assert service.is_initialized is True
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_success(self, mock_openai_client):
        """Test successful audio transcription"""
        config = STTConfig()
        service = WhisperSTTService(config, "test-api-key")
        service.initialize()
        
        # Create test audio data (1 second of sound)
        sample_rate = 16000
        samples = int(sample_rate * 1.0)
        audio_data = np.ones(samples, dtype=np.int16) * 1000
        
        result = await service.transcribe_audio(audio_data.tobytes(), sample_rate)
        
        assert result is not None
        assert isinstance(result, STTResult)
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.duration == 1.0
        assert result.processing_time > 0
        assert service.successful_requests == 1
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_too_short(self, mock_openai_client):
        """Test transcription with audio too short"""
        config = STTConfig(min_audio_length=1.0)
        service = WhisperSTTService(config, "test-api-key")
        service.initialize()
        
        # Create short audio data (0.2 seconds)
        sample_rate = 16000
        samples = int(sample_rate * 0.2)
        audio_data = np.ones(samples, dtype=np.int16) * 1000
        
        result = await service.transcribe_audio(audio_data.tobytes(), sample_rate)
        
        assert result is None
        assert service.total_requests == 1
        assert service.successful_requests == 0
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_no_voice(self, mock_openai_client):
        """Test transcription with no voice activity"""
        config = STTConfig(vad_enabled=True)
        service = WhisperSTTService(config, "test-api-key")
        service.initialize()
        
        # Create silence
        sample_rate = 16000
        samples = int(sample_rate * 1.0)
        audio_data = np.zeros(samples, dtype=np.int16)
        
        result = await service.transcribe_audio(audio_data.tobytes(), sample_rate)
        
        assert result is None
        assert service.total_requests == 1
        assert service.successful_requests == 0
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_api_error(self, mock_openai_client):
        """Test transcription with API error"""
        config = STTConfig(max_retries=1)
        service = WhisperSTTService(config, "test-api-key")
        service.initialize()
        
        # Mock API error
        mock_openai_client.audio.transcriptions.create.side_effect = Exception("API Error")
        
        # Create test audio data
        sample_rate = 16000
        samples = int(sample_rate * 1.0)
        audio_data = np.ones(samples, dtype=np.int16) * 1000
        
        result = await service.transcribe_audio(audio_data.tobytes(), sample_rate)
        
        assert result is None
        assert service.failed_requests == 1
    
    def test_estimate_confidence(self, mock_openai_client):
        """Test confidence estimation"""
        config = STTConfig()
        service = WhisperSTTService(config, "test-api-key")
        
        # Test with good text
        confidence = service._estimate_confidence("Hello world.", [])
        assert confidence > 0.5
        
        # Test with empty text
        confidence = service._estimate_confidence("", [])
        assert confidence == 0.0
        
        # Test with short text
        confidence = service._estimate_confidence("Hi", [])
        assert confidence < 0.8
    
    def test_get_statistics(self, mock_openai_client):
        """Test statistics retrieval"""
        config = STTConfig()
        service = WhisperSTTService(config, "test-api-key")
        
        # Set some test statistics
        service.total_requests = 10
        service.successful_requests = 8
        service.failed_requests = 2
        service.total_processing_time = 16.0
        
        stats = service.get_statistics()
        
        assert stats['total_requests'] == 10
        assert stats['successful_requests'] == 8
        assert stats['failed_requests'] == 2
        assert stats['success_rate'] == 0.8
        assert stats['average_processing_time'] == 2.0
        assert stats['total_processing_time'] == 16.0
    
    def test_reset_statistics(self, mock_openai_client):
        """Test statistics reset"""
        config = STTConfig()
        service = WhisperSTTService(config, "test-api-key")
        
        # Set some test statistics
        service.total_requests = 10
        service.successful_requests = 8
        service.failed_requests = 2
        service.total_processing_time = 16.0
        
        service.reset_statistics()
        
        assert service.total_requests == 0
        assert service.successful_requests == 0
        assert service.failed_requests == 0
        assert service.total_processing_time == 0.0


class TestSTTServiceFactory:
    """Test STT service factory function"""
    
    @patch('assistant.stt.OpenAI')
    def test_create_whisper_stt_service_default(self, mock_openai):
        """Test creating STT service with default config"""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock successful initialization
        mock_response = Mock()
        mock_response.text = "test"
        mock_client.audio.transcriptions.create.return_value = mock_response
        
        service = create_whisper_stt_service("test-api-key")
        
        assert isinstance(service, WhisperSTTService)
        assert service.config.model == "whisper-1"
        assert service.is_initialized is True
    
    @patch('assistant.stt.OpenAI')
    def test_create_whisper_stt_service_custom_config(self, mock_openai):
        """Test creating STT service with custom config"""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock successful initialization
        mock_response = Mock()
        mock_response.text = "test"
        mock_client.audio.transcriptions.create.return_value = mock_response
        
        custom_config = STTConfig(model="whisper-large", language="es")
        service = create_whisper_stt_service("test-api-key", custom_config)
        
        assert isinstance(service, WhisperSTTService)
        assert service.config.model == "whisper-large"
        assert service.config.language == "es"
        assert service.is_initialized is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 