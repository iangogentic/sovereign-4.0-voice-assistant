"""
Tests for the Text-to-Speech (TTS) module
"""

import pytest
import asyncio
import time
import hashlib
import io
import sys
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock, PropertyMock

# Add the parent directory to sys.path to import assistant
sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant.tts import (
    TTSConfig,
    TTSResult,
    AudioProcessor,
    TTSCache,
    OpenAITTSService,
    create_openai_tts_service
)


class TestTTSConfig:
    """Test TTSConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = TTSConfig()
        
        assert config.model == "tts-1"
        assert config.voice == "alloy"
        assert config.speed == 1.0
        assert config.response_format == "mp3"
        assert config.max_text_length == 4096
        assert config.chunk_size == 1000
        assert config.sample_rate == 16000
        assert config.normalize_audio is True
        assert config.timeout == 15
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.cache_enabled is True
        assert config.cache_max_size == 100
        assert config.cache_ttl == 3600
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = TTSConfig(
            model="tts-1-hd",
            voice="nova",
            speed=1.5,
            response_format="wav",
            max_text_length=2000,
            chunk_size=500,
            sample_rate=22050,
            normalize_audio=False,
            timeout=30,
            max_retries=5,
            retry_delay=2.0,
            cache_enabled=False,
            cache_max_size=200,
            cache_ttl=7200
        )
        
        assert config.model == "tts-1-hd"
        assert config.voice == "nova"
        assert config.speed == 1.5
        assert config.response_format == "wav"
        assert config.max_text_length == 2000
        assert config.chunk_size == 500
        assert config.sample_rate == 22050
        assert config.normalize_audio is False
        assert config.timeout == 30
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.cache_enabled is False
        assert config.cache_max_size == 200
        assert config.cache_ttl == 7200


class TestTTSResult:
    """Test TTSResult dataclass"""
    
    def test_tts_result_creation(self):
        """Test TTSResult creation"""
        audio_data = b"fake_audio_data"
        result = TTSResult(
            audio_data=audio_data,
            text="Hello world",
            duration=2.5,
            processing_time=0.8,
            format="mp3",
            voice="alloy",
            speed=1.0,
            sample_rate=16000
        )
        
        assert result.audio_data == audio_data
        assert result.text == "Hello world"
        assert result.duration == 2.5
        assert result.processing_time == 0.8
        assert result.format == "mp3"
        assert result.voice == "alloy"
        assert result.speed == 1.0
        assert result.sample_rate == 16000
        assert result.cached is False
    
    def test_tts_result_cached(self):
        """Test TTSResult with cached flag"""
        audio_data = b"fake_audio_data"
        result = TTSResult(
            audio_data=audio_data,
            text="Hello world",
            duration=2.5,
            processing_time=0.8,
            format="mp3",
            voice="alloy",
            speed=1.0,
            sample_rate=16000,
            cached=True
        )
        
        assert result.cached is True
    
    @patch('builtins.open', new_callable=Mock)
    def test_save_to_file_success(self, mock_open):
        """Test successful file saving"""
        audio_data = b"fake_audio_data"
        result = TTSResult(
            audio_data=audio_data,
            text="Hello world",
            duration=2.5,
            processing_time=0.8,
            format="mp3",
            voice="alloy",
            speed=1.0,
            sample_rate=16000
        )
        
        mock_file = Mock()
        mock_open.return_value.__enter__ = Mock(return_value=mock_file)
        mock_open.return_value.__exit__ = Mock(return_value=None)
        
        success = result.save_to_file("test.mp3")
        
        assert success is True
        mock_open.assert_called_once_with("test.mp3", 'wb')
        mock_file.write.assert_called_once_with(audio_data)
    
    @patch('builtins.open', side_effect=Exception("File error"))
    def test_save_to_file_failure(self, mock_open):
        """Test file saving failure"""
        audio_data = b"fake_audio_data"
        result = TTSResult(
            audio_data=audio_data,
            text="Hello world",
            duration=2.5,
            processing_time=0.8,
            format="mp3",
            voice="alloy",
            speed=1.0,
            sample_rate=16000
        )
        
        success = result.save_to_file("test.mp3")
        
        assert success is False


class TestAudioProcessor:
    """Test AudioProcessor class"""
    
    def test_audio_processor_creation(self):
        """Test AudioProcessor creation"""
        config = TTSConfig()
        processor = AudioProcessor(config)
        
        assert processor.config == config
    
    def test_chunk_text_short(self):
        """Test text chunking with short text"""
        config = TTSConfig(chunk_size=100)
        processor = AudioProcessor(config)
        
        text = "Hello world"
        chunks = processor.chunk_text(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_text_long(self):
        """Test text chunking with long text"""
        config = TTSConfig(chunk_size=50)
        processor = AudioProcessor(config)
        
        text = "This is a long sentence. This is another sentence. This is yet another sentence."
        chunks = processor.chunk_text(text)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 50 for chunk in chunks)
        assert ". ".join(chunks).replace(". ", ". ") in text or text in ". ".join(chunks)
    
    def test_convert_mp3_to_wav_error_handling(self):
        """Test MP3 to WAV conversion error handling"""
        config = TTSConfig()
        processor = AudioProcessor(config)
        
        # Test with None input
        result = processor.convert_mp3_to_wav(None)
        assert result is None
        
        # Test with empty bytes
        result = processor.convert_mp3_to_wav(b"")
        assert result is None
    
    @patch('assistant.tts.AudioSegment')
    @patch('tempfile.NamedTemporaryFile')
    @patch('os.unlink')
    def test_convert_mp3_to_wav_success_path(self, mock_unlink, mock_temp_file, mock_audio_segment):
        """Test MP3 to WAV conversion success path (simplified)"""
        config = TTSConfig(sample_rate=16000, normalize_audio=True)
        processor = AudioProcessor(config)
        
        # Mock temporary file
        mock_temp = Mock()
        mock_temp.name = "temp_file.mp3"
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        # Create a simpler mock that doesn't cause comparison issues
        mock_audio = Mock()
        # Set frame_rate as a simple attribute, not a property
        mock_audio.frame_rate = 16000  # Same as config to avoid resampling
        mock_audio.channels = 1        # Same as config to avoid conversion
        mock_audio_segment.from_mp3.return_value = mock_audio
        
        # Since frame_rate and channels match config, no processing methods should be called
        # Just mock normalize and export
        mock_audio_normalized = Mock()
        mock_audio.normalize.return_value = mock_audio_normalized
        
        # Mock export
        mock_buffer = Mock()
        mock_buffer.getvalue.return_value = b"wav_data"
        
        with patch('io.BytesIO', return_value=mock_buffer):
            mp3_data = b"fake_mp3_data"
            wav_data = processor.convert_mp3_to_wav(mp3_data)
        
        assert wav_data == b"wav_data"
        mock_temp.write.assert_called_once_with(mp3_data)
        mock_audio_segment.from_mp3.assert_called_once_with("temp_file.mp3")
        # Should not call set_frame_rate or set_channels since they match
        mock_audio.set_frame_rate.assert_not_called()
        mock_audio.set_channels.assert_not_called()
        # Should call normalize since normalize_audio=True
        mock_audio.normalize.assert_called_once()
        mock_audio_normalized.export.assert_called_once()
        mock_unlink.assert_called_once_with("temp_file.mp3")
    
    @patch('assistant.tts.AudioSegment')
    @patch('tempfile.NamedTemporaryFile')
    @patch('os.unlink')
    def test_calculate_audio_duration(self, mock_unlink, mock_temp_file, mock_audio_segment):
        """Test audio duration calculation"""
        config = TTSConfig()
        processor = AudioProcessor(config)
        
        # Mock temporary file
        mock_temp = Mock()
        mock_temp.name = "temp_file.mp3"
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        # Mock AudioSegment - 2500ms duration
        mock_audio = Mock()
        mock_audio.__len__ = Mock(return_value=2500)
        mock_audio_segment.from_mp3.return_value = mock_audio
        
        audio_data = b"fake_audio_data"
        duration = processor.calculate_audio_duration(audio_data, "mp3")
        
        assert duration == 2.5  # 2500ms = 2.5s
        mock_temp.write.assert_called_once_with(audio_data)
        mock_audio_segment.from_mp3.assert_called_once_with("temp_file.mp3")
        mock_unlink.assert_called_once_with("temp_file.mp3")
    
    @patch('assistant.tts.AudioSegment')
    @patch('tempfile.NamedTemporaryFile')
    @patch('os.unlink')
    def test_merge_audio_chunks_single(self, mock_unlink, mock_temp_file, mock_audio_segment):
        """Test merging single audio chunk"""
        config = TTSConfig()
        processor = AudioProcessor(config)
        
        chunks = [b"audio_chunk"]
        result = processor.merge_audio_chunks(chunks)
        
        assert result == b"audio_chunk"
        # Should not call any file operations for single chunk
        mock_temp_file.assert_not_called()
    
    @patch('assistant.tts.AudioSegment')
    @patch('tempfile.NamedTemporaryFile')
    @patch('os.unlink')
    @patch('io.BytesIO')
    def test_merge_audio_chunks_multiple(self, mock_io, mock_unlink, mock_temp_file, mock_audio_segment):
        """Test merging multiple audio chunks"""
        config = TTSConfig()
        processor = AudioProcessor(config)
        
        # Mock temporary files
        mock_temp = Mock()
        mock_temp.name = "temp_file.mp3"
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        # Mock AudioSegment
        mock_audio1 = Mock()
        mock_audio2 = Mock()
        mock_merged = Mock()
        mock_audio1.__add__ = Mock(return_value=mock_merged)
        mock_audio_segment.from_mp3.side_effect = [mock_audio1, mock_audio2]
        
        # Mock buffer
        mock_buffer = Mock()
        mock_buffer.getvalue.return_value = b"merged_audio"
        mock_io.return_value = mock_buffer
        
        chunks = [b"audio_chunk1", b"audio_chunk2"]
        result = processor.merge_audio_chunks(chunks)
        
        assert result == b"merged_audio"
        mock_merged.export.assert_called_once_with(mock_buffer, format="mp3")


class TestTTSCache:
    """Test TTSCache class"""
    
    def test_cache_creation(self):
        """Test cache creation"""
        cache = TTSCache(max_size=50, ttl=1800)
        
        assert cache.max_size == 50
        assert cache.ttl == 1800
        assert len(cache.cache) == 0
        assert len(cache.access_times) == 0
    
    def test_generate_key(self):
        """Test cache key generation"""
        cache = TTSCache()
        
        key1 = cache._generate_key("Hello", "alloy", 1.0)
        key2 = cache._generate_key("Hello", "alloy", 1.0)
        key3 = cache._generate_key("Hello", "nova", 1.0)
        
        assert key1 == key2  # Same parameters should generate same key
        assert key1 != key3  # Different parameters should generate different keys
        assert len(key1) == 32  # MD5 hash length
    
    def test_cache_put_and_get(self):
        """Test cache put and get operations"""
        cache = TTSCache()
        
        result = TTSResult(
            audio_data=b"test_audio",
            text="Hello",
            duration=1.0,
            processing_time=0.5,
            format="mp3",
            voice="alloy",
            speed=1.0,
            sample_rate=16000
        )
        
        # Put result in cache
        cache.put("Hello", "alloy", 1.0, result)
        
        # Get result from cache
        cached_result = cache.get("Hello", "alloy", 1.0)
        
        assert cached_result is not None
        assert cached_result.text == "Hello"
        assert cached_result.cached is True
    
    def test_cache_miss(self):
        """Test cache miss"""
        cache = TTSCache()
        
        # Try to get non-existent entry
        cached_result = cache.get("Hello", "alloy", 1.0)
        
        assert cached_result is None
    
    def test_cache_expiration(self):
        """Test cache expiration"""
        cache = TTSCache(ttl=1)  # 1 second TTL
        
        result = TTSResult(
            audio_data=b"test_audio",
            text="Hello",
            duration=1.0,
            processing_time=0.5,
            format="mp3",
            voice="alloy",
            speed=1.0,
            sample_rate=16000
        )
        
        # Put result in cache
        cache.put("Hello", "alloy", 1.0, result)
        
        # Should be available immediately
        cached_result = cache.get("Hello", "alloy", 1.0)
        assert cached_result is not None
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired now
        cached_result = cache.get("Hello", "alloy", 1.0)
        assert cached_result is None
    
    def test_cache_max_size(self):
        """Test cache max size limit"""
        cache = TTSCache(max_size=2)
        
        result1 = TTSResult(b"audio1", "Hello1", 1.0, 0.5, "mp3", "alloy", 1.0, 16000)
        result2 = TTSResult(b"audio2", "Hello2", 1.0, 0.5, "mp3", "alloy", 1.0, 16000)
        result3 = TTSResult(b"audio3", "Hello3", 1.0, 0.5, "mp3", "alloy", 1.0, 16000)
        
        # Add first two results
        cache.put("Hello1", "alloy", 1.0, result1)
        cache.put("Hello2", "alloy", 1.0, result2)
        
        # Both should be available
        assert cache.get("Hello1", "alloy", 1.0) is not None
        assert cache.get("Hello2", "alloy", 1.0) is not None
        
        # Add third result (should evict oldest)
        cache.put("Hello3", "alloy", 1.0, result3)
        
        # First should be evicted, second and third should be available
        assert cache.get("Hello1", "alloy", 1.0) is None
        assert cache.get("Hello2", "alloy", 1.0) is not None
        assert cache.get("Hello3", "alloy", 1.0) is not None
    
    def test_cache_clear(self):
        """Test cache clear operation"""
        cache = TTSCache()
        
        result = TTSResult(b"audio", "Hello", 1.0, 0.5, "mp3", "alloy", 1.0, 16000)
        cache.put("Hello", "alloy", 1.0, result)
        
        # Should be available
        assert cache.get("Hello", "alloy", 1.0) is not None
        
        # Clear cache
        cache.clear()
        
        # Should be empty
        assert cache.get("Hello", "alloy", 1.0) is None
        assert len(cache.cache) == 0
        assert len(cache.access_times) == 0


class TestOpenAITTSService:
    """Test OpenAITTSService class"""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client"""
        with patch('assistant.tts.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # Mock successful response
            mock_response = Mock()
            mock_response.content = b"fake_mp3_data"
            mock_client.audio.speech.create.return_value = mock_response
            
            yield mock_client
    
    def test_service_creation(self, mock_openai_client):
        """Test TTS service creation"""
        config = TTSConfig()
        service = OpenAITTSService(config, "test_api_key")
        
        assert service.config == config
        assert service.audio_processor is not None
        assert service.cache is not None
        assert service.is_initialized is False
        assert service.total_requests == 0
        assert service.successful_requests == 0
        assert service.failed_requests == 0
    
    def test_service_creation_no_cache(self, mock_openai_client):
        """Test TTS service creation without cache"""
        config = TTSConfig(cache_enabled=False)
        service = OpenAITTSService(config, "test_api_key")
        
        assert service.cache is None
    
    def test_service_initialization_success(self, mock_openai_client):
        """Test successful service initialization"""
        config = TTSConfig()
        service = OpenAITTSService(config, "test_api_key")
        
        success = service.initialize()
        
        assert success is True
        assert service.is_initialized is True
        mock_openai_client.audio.speech.create.assert_called_once()
    
    def test_service_initialization_failure(self):
        """Test failed service initialization"""
        with patch('assistant.tts.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            mock_client.audio.speech.create.side_effect = Exception("API Error")
            
            config = TTSConfig()
            service = OpenAITTSService(config, "test_api_key")
            
            success = service.initialize()
            
            assert success is False
            assert service.is_initialized is False
    
    def test_validate_config_success(self, mock_openai_client):
        """Test successful configuration validation"""
        config = TTSConfig()
        service = OpenAITTSService(config, "test_api_key")
        
        assert service._validate_config() is True
    
    def test_validate_config_invalid_voice(self, mock_openai_client):
        """Test configuration validation with invalid voice"""
        config = TTSConfig(voice="invalid_voice")
        service = OpenAITTSService(config, "test_api_key")
        
        assert service._validate_config() is False
    
    def test_validate_config_invalid_model(self, mock_openai_client):
        """Test configuration validation with invalid model"""
        config = TTSConfig(model="invalid_model")
        service = OpenAITTSService(config, "test_api_key")
        
        assert service._validate_config() is False
    
    def test_validate_config_invalid_speed(self, mock_openai_client):
        """Test configuration validation with invalid speed"""
        config = TTSConfig(speed=5.0)  # Too fast
        service = OpenAITTSService(config, "test_api_key")
        
        assert service._validate_config() is False
    
    @pytest.mark.asyncio
    async def test_synthesize_speech_not_initialized(self, mock_openai_client):
        """Test speech synthesis when service not initialized"""
        config = TTSConfig()
        service = OpenAITTSService(config, "test_api_key")
        
        result = await service.synthesize_speech("Hello world")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_synthesize_speech_empty_text(self, mock_openai_client):
        """Test speech synthesis with empty text"""
        config = TTSConfig()
        service = OpenAITTSService(config, "test_api_key")
        service.is_initialized = True
        
        result = await service.synthesize_speech("")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_synthesize_speech_success(self, mock_openai_client):
        """Test successful speech synthesis"""
        config = TTSConfig()
        service = OpenAITTSService(config, "test_api_key")
        service.is_initialized = True
        
        # Mock audio processor
        with patch.object(service.audio_processor, 'calculate_audio_duration', return_value=2.5):
            result = await service.synthesize_speech("Hello world")
        
        assert result is not None
        assert result.text == "Hello world"
        assert result.audio_data == b"fake_mp3_data"
        assert result.duration == 2.5
        assert result.format == "mp3"
        assert result.voice == "alloy"
        assert service.successful_requests == 1
    
    @pytest.mark.asyncio
    async def test_synthesize_speech_cache_hit(self, mock_openai_client):
        """Test speech synthesis with cache hit"""
        config = TTSConfig()
        service = OpenAITTSService(config, "test_api_key")
        service.is_initialized = True
        
        # Create cached result
        cached_result = TTSResult(
            audio_data=b"cached_audio",
            text="Hello world",
            duration=2.0,
            processing_time=0.5,
            format="mp3",
            voice="alloy",
            speed=1.0,
            sample_rate=16000,
            cached=True  # Set the cached flag
        )
        
        # Mock cache hit - the cache.get method should return the cached result
        with patch.object(service.cache, 'get', return_value=cached_result):
            result = await service.synthesize_speech("Hello world")
        
        assert result is not None
        assert result.text == "Hello world"
        assert result.audio_data == b"cached_audio"
        assert result.cached is True
        assert service.cache_hits == 1
        assert service.total_requests == 0  # No API call made
    
    @pytest.mark.asyncio
    async def test_synthesize_speech_text_too_long(self, mock_openai_client):
        """Test speech synthesis with text too long"""
        config = TTSConfig(max_text_length=10)
        service = OpenAITTSService(config, "test_api_key")
        service.is_initialized = True
        
        result = await service.synthesize_speech("This text is too long for the configuration")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_synthesize_speech_api_error(self, mock_openai_client):
        """Test speech synthesis with API error"""
        config = TTSConfig()
        service = OpenAITTSService(config, "test_api_key")
        service.is_initialized = True
        
        # Mock API error
        mock_openai_client.audio.speech.create.side_effect = Exception("API Error")
        
        result = await service.synthesize_speech("Hello world")
        
        assert result is None
        # The failed_requests counter is incremented during synthesis_speech, not _synthesize_with_retry
        assert service.failed_requests == 1
    
    def test_get_wav_audio_mp3_input(self, mock_openai_client):
        """Test WAV audio conversion from MP3"""
        config = TTSConfig()
        service = OpenAITTSService(config, "test_api_key")
        
        result = TTSResult(
            audio_data=b"mp3_data",
            text="Hello",
            duration=1.0,
            processing_time=0.5,
            format="mp3",
            voice="alloy",
            speed=1.0,
            sample_rate=16000
        )
        
        # Mock audio processor
        with patch.object(service.audio_processor, 'convert_mp3_to_wav', return_value=b"wav_data"):
            wav_data = service.get_wav_audio(result)
        
        assert wav_data == b"wav_data"
    
    def test_get_wav_audio_wav_input(self, mock_openai_client):
        """Test WAV audio with WAV input"""
        config = TTSConfig()
        service = OpenAITTSService(config, "test_api_key")
        
        result = TTSResult(
            audio_data=b"wav_data",
            text="Hello",
            duration=1.0,
            processing_time=0.5,
            format="wav",
            voice="alloy",
            speed=1.0,
            sample_rate=16000
        )
        
        wav_data = service.get_wav_audio(result)
        
        assert wav_data == b"wav_data"
    
    def test_get_statistics(self, mock_openai_client):
        """Test getting service statistics"""
        config = TTSConfig()
        service = OpenAITTSService(config, "test_api_key")
        
        # Set some statistics
        service.total_requests = 10
        service.successful_requests = 8
        service.failed_requests = 2
        service.total_processing_time = 5.0
        service.cache_hits = 3
        service.cache_misses = 5
        
        stats = service.get_statistics()
        
        assert stats["total_requests"] == 10
        assert stats["successful_requests"] == 8
        assert stats["failed_requests"] == 2
        assert stats["success_rate"] == 0.8
        assert stats["total_processing_time"] == 5.0
        assert stats["average_processing_time"] == 0.625
        assert stats["cache_hits"] == 3
        assert stats["cache_misses"] == 5
        assert stats["cache_hit_rate"] == 0.375
    
    def test_reset_statistics(self, mock_openai_client):
        """Test resetting service statistics"""
        config = TTSConfig()
        service = OpenAITTSService(config, "test_api_key")
        
        # Set some statistics
        service.total_requests = 10
        service.successful_requests = 8
        service.failed_requests = 2
        service.total_processing_time = 5.0
        service.cache_hits = 3
        service.cache_misses = 5
        
        service.reset_statistics()
        
        assert service.total_requests == 0
        assert service.successful_requests == 0
        assert service.failed_requests == 0
        assert service.total_processing_time == 0.0
        assert service.cache_hits == 0
        assert service.cache_misses == 0
    
    def test_clear_cache(self, mock_openai_client):
        """Test clearing service cache"""
        config = TTSConfig()
        service = OpenAITTSService(config, "test_api_key")
        
        # Mock cache
        with patch.object(service.cache, 'clear') as mock_clear:
            service.clear_cache()
        
        mock_clear.assert_called_once()


class TestTTSServiceFactory:
    """Test TTS service factory function"""
    
    @patch('assistant.tts.OpenAI')
    def test_create_openai_tts_service_default(self, mock_openai):
        """Test creating TTS service with default config"""
        # Mock successful initialization
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        mock_response.content = b"fake_mp3_data"
        mock_client.audio.speech.create.return_value = mock_response
        
        service = create_openai_tts_service("test_api_key")
        
        assert service is not None
        assert service.is_initialized is True
        assert isinstance(service.config, TTSConfig)
        assert service.config.model == "tts-1"
        assert service.config.voice == "alloy"
    
    @patch('assistant.tts.OpenAI')
    def test_create_openai_tts_service_custom_config(self, mock_openai):
        """Test creating TTS service with custom config"""
        # Mock successful initialization
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        mock_response.content = b"fake_mp3_data"
        mock_client.audio.speech.create.return_value = mock_response
        
        custom_config = TTSConfig(model="tts-1-hd", voice="nova")
        service = create_openai_tts_service("test_api_key", custom_config)
        
        assert service is not None
        assert service.is_initialized is True
        assert service.config.model == "tts-1-hd"
        assert service.config.voice == "nova"
    
    @patch('assistant.tts.OpenAI')
    def test_create_openai_tts_service_initialization_failure(self, mock_openai):
        """Test creating TTS service with initialization failure"""
        # Mock failed initialization
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_client.audio.speech.create.side_effect = Exception("API Error")
        
        with pytest.raises(RuntimeError, match="Failed to initialize OpenAI TTS service"):
            create_openai_tts_service("test_api_key") 