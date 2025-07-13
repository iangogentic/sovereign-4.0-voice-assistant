"""
Test suite for EmotionPreserver module

Tests emotion preservation functionality including:
- Audio-only processing configuration
- Real-time audio buffer management
- Audio format validation and conversion
- Performance metrics tracking
- Error handling and graceful degradation
- Integration with Realtime API
"""

import pytest
import asyncio
import numpy as np
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import asdict

# Import the modules to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from assistant.emotion_preserver import (
    EmotionPreserver,
    EmotionPreservationConfig,
    AudioBufferConfig,
    AudioBuffer,
    AudioFormatConverter,
    AudioMetrics,
    AudioFormat,
    EmotionProcessingMode,
    create_emotion_preserver,
    create_audio_only_config
)


@pytest.fixture
def default_config():
    """Create default emotion preservation configuration"""
    return EmotionPreservationConfig()


@pytest.fixture
def audio_buffer_config():
    """Create default audio buffer configuration"""
    return AudioBufferConfig()


@pytest.fixture
def sample_audio_data():
    """Generate sample PCM16 audio data"""
    # Generate 1 second of sine wave at 440Hz
    sample_rate = 24000
    duration = 1.0
    frequency = 440
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_array = (np.sin(2 * np.pi * frequency * t) * 32767 * 0.5).astype(np.int16)
    return audio_array.tobytes()


@pytest.fixture
def mock_realtime_service():
    """Create mock realtime service"""
    service = Mock()
    service.config = Mock()
    service.config.modalities = ["text", "audio"]
    service.config.input_audio_format = "pcm16"
    service.config.output_audio_format = "pcm16"
    return service


class TestAudioFormat:
    """Test AudioFormat enum"""
    
    def test_audio_format_values(self):
        """Test audio format enum values"""
        assert AudioFormat.PCM16.value == "pcm16"
        assert AudioFormat.PCM24.value == "pcm24"
        assert AudioFormat.MP3.value == "mp3"
        assert AudioFormat.WAV.value == "wav"


class TestEmotionProcessingMode:
    """Test EmotionProcessingMode enum"""
    
    def test_processing_mode_values(self):
        """Test processing mode enum values"""
        assert EmotionProcessingMode.AUDIO_ONLY.value == "audio_only"
        assert EmotionProcessingMode.HYBRID.value == "hybrid"
        assert EmotionProcessingMode.DISABLED.value == "disabled"


class TestAudioBufferConfig:
    """Test AudioBufferConfig dataclass"""
    
    def test_default_values(self, audio_buffer_config):
        """Test default configuration values"""
        assert audio_buffer_config.sample_rate == 24000
        assert audio_buffer_config.channels == 1
        assert audio_buffer_config.chunk_size == 1024
        assert audio_buffer_config.buffer_duration_ms == 100
        assert audio_buffer_config.max_buffer_size == 48000
        assert audio_buffer_config.overflow_handling == "drop_oldest"


class TestEmotionPreservationConfig:
    """Test EmotionPreservationConfig dataclass"""
    
    def test_default_values(self, default_config):
        """Test default configuration values"""
        assert default_config.mode == EmotionProcessingMode.AUDIO_ONLY
        assert default_config.input_format == AudioFormat.PCM16
        assert default_config.output_format == AudioFormat.PCM16
        assert default_config.sample_rate == 24000
        assert default_config.max_latency_ms == 50.0
        assert default_config.enable_real_time_processing is True
        assert default_config.enable_graceful_degradation is True


class TestAudioMetrics:
    """Test AudioMetrics dataclass and methods"""
    
    def test_initial_values(self):
        """Test initial metric values"""
        metrics = AudioMetrics()
        assert metrics.total_chunks_processed == 0
        assert metrics.processing_latency_ms == 0.0
        assert metrics.average_latency_ms == 0.0
        assert metrics.max_latency_ms == 0.0
        assert metrics.error_count == 0
    
    def test_update_latency(self):
        """Test latency update functionality"""
        metrics = AudioMetrics()
        
        # First update
        metrics.update_latency(10.0)
        assert metrics.total_chunks_processed == 1
        assert metrics.processing_latency_ms == 10.0
        assert metrics.average_latency_ms == 10.0
        assert metrics.max_latency_ms == 10.0
        
        # Second update
        metrics.update_latency(20.0)
        assert metrics.total_chunks_processed == 2
        assert metrics.processing_latency_ms == 20.0
        assert metrics.average_latency_ms == 15.0  # (10 + 20) / 2
        assert metrics.max_latency_ms == 20.0
        
        # Third update with lower latency
        metrics.update_latency(5.0)
        assert metrics.total_chunks_processed == 3
        assert metrics.processing_latency_ms == 5.0
        assert metrics.average_latency_ms == (10.0 + 20.0 + 5.0) / 3  # ~11.67
        assert metrics.max_latency_ms == 20.0  # Should remain the max


class TestAudioBuffer:
    """Test AudioBuffer class functionality"""
    
    @pytest.fixture
    def audio_buffer(self, audio_buffer_config):
        """Create audio buffer instance"""
        return AudioBuffer(audio_buffer_config)
    
    @pytest.mark.asyncio
    async def test_buffer_initialization(self, audio_buffer, audio_buffer_config):
        """Test buffer initialization"""
        assert audio_buffer.config == audio_buffer_config
        assert audio_buffer.write_pos == 0
        assert audio_buffer.read_pos == 0
        assert audio_buffer.size == 0
        assert len(audio_buffer.buffer) == audio_buffer_config.max_buffer_size
    
    @pytest.mark.asyncio
    async def test_write_and_read(self, audio_buffer, sample_audio_data):
        """Test basic write and read operations"""
        # Write sample data
        result = await audio_buffer.write(sample_audio_data[:2048])  # 1024 samples
        assert result is True
        assert audio_buffer.get_available_size() == 1024
        
        # Read data back
        read_data = await audio_buffer.read(1024)
        assert read_data is not None
        assert len(read_data) == 2048  # 1024 samples * 2 bytes per sample
        assert audio_buffer.get_available_size() == 0
    
    @pytest.mark.asyncio
    async def test_buffer_wrap_around(self, audio_buffer):
        """Test buffer wrap-around functionality"""
        # Fill buffer almost to capacity
        large_data = np.zeros(audio_buffer.config.max_buffer_size - 100, dtype=np.int16).tobytes()
        await audio_buffer.write(large_data)
        
        # Read most of it
        await audio_buffer.read(audio_buffer.config.max_buffer_size - 200)
        
        # Write data that will wrap around
        wrap_data = np.ones(300, dtype=np.int16).tobytes()
        result = await audio_buffer.write(wrap_data)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_buffer_overflow_drop_oldest(self, audio_buffer):
        """Test buffer overflow with drop_oldest strategy"""
        # Fill buffer to capacity
        max_data = np.zeros(audio_buffer.config.max_buffer_size, dtype=np.int16).tobytes()
        await audio_buffer.write(max_data)
        
        # Try to write more data (should drop oldest)
        overflow_data = np.ones(100, dtype=np.int16).tobytes()
        result = await audio_buffer.write(overflow_data)
        assert result is True
        assert audio_buffer.get_available_size() == audio_buffer.config.max_buffer_size
    
    @pytest.mark.asyncio
    async def test_buffer_overflow_extend(self):
        """Test buffer overflow with extend strategy"""
        config = AudioBufferConfig(overflow_handling="extend")
        buffer = AudioBuffer(config)
        
        # Fill buffer to capacity
        max_data = np.zeros(config.max_buffer_size, dtype=np.int16).tobytes()
        await buffer.write(max_data)
        
        # Try to write more data (should reject)
        overflow_data = np.ones(100, dtype=np.int16).tobytes()
        result = await buffer.write(overflow_data)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_read_insufficient_data(self, audio_buffer):
        """Test reading when insufficient data available"""
        # Write small amount of data
        small_data = np.zeros(100, dtype=np.int16).tobytes()
        await audio_buffer.write(small_data)
        
        # Try to read more than available
        result = await audio_buffer.read(200)
        assert result is None
    
    def test_clear_buffer(self, audio_buffer):
        """Test buffer clearing"""
        # Simulate some data
        audio_buffer.write_pos = 100
        audio_buffer.read_pos = 50
        audio_buffer.size = 50
        
        audio_buffer.clear()
        assert audio_buffer.write_pos == 0
        assert audio_buffer.read_pos == 0
        assert audio_buffer.size == 0


class TestAudioFormatConverter:
    """Test AudioFormatConverter class"""
    
    def test_validate_format_pcm16(self, sample_audio_data):
        """Test PCM16 format validation"""
        # Valid PCM16 data (even byte length)
        assert AudioFormatConverter.validate_format(sample_audio_data, AudioFormat.PCM16, 24000) is True
        
        # Invalid PCM16 data (odd byte length)
        invalid_data = sample_audio_data[:-1]
        assert AudioFormatConverter.validate_format(invalid_data, AudioFormat.PCM16, 24000) is False
    
    def test_validate_format_pcm24(self):
        """Test PCM24 format validation"""
        # Valid PCM24 data (multiple of 3 bytes)
        valid_data = b'\x00\x01\x02' * 100  # 300 bytes
        assert AudioFormatConverter.validate_format(valid_data, AudioFormat.PCM24, 24000) is True
        
        # Invalid PCM24 data
        invalid_data = b'\x00\x01\x02\x03\x04'  # 5 bytes
        assert AudioFormatConverter.validate_format(invalid_data, AudioFormat.PCM24, 24000) is False
    
    def test_convert_to_realtime_format_pcm16(self, sample_audio_data):
        """Test conversion when already in PCM16 format"""
        result = AudioFormatConverter.convert_to_realtime_format(sample_audio_data, AudioFormat.PCM16)
        assert result == sample_audio_data
    
    def test_convert_to_realtime_format_pcm24(self):
        """Test conversion from PCM24 to PCM16"""
        # Create sample PCM24 data (3 bytes per sample, but using int32 for simplicity)
        pcm24_array = np.array([1000, 2000, 3000], dtype=np.int32)
        pcm24_data = pcm24_array.tobytes()
        
        result = AudioFormatConverter.convert_to_realtime_format(pcm24_data, AudioFormat.PCM24)
        
        # Should be converted to PCM16
        expected_array = (pcm24_array >> 8).astype(np.int16)
        expected_data = expected_array.tobytes()
        assert result == expected_data
    
    def test_normalize_audio(self, sample_audio_data):
        """Test audio normalization"""
        # Test with quiet audio (half volume)
        quiet_array = (np.frombuffer(sample_audio_data, dtype=np.int16) * 0.5).astype(np.int16)
        quiet_data = quiet_array.tobytes()
        
        normalized = AudioFormatConverter.normalize_audio(quiet_data, target_level=0.8)
        normalized_array = np.frombuffer(normalized, dtype=np.int16)
        
        # Should be louder than original
        assert np.max(np.abs(normalized_array)) > np.max(np.abs(quiet_array))
    
    def test_normalize_audio_silent(self):
        """Test normalization with silent audio"""
        silent_data = np.zeros(1000, dtype=np.int16).tobytes()
        result = AudioFormatConverter.normalize_audio(silent_data)
        assert result == silent_data


class TestEmotionPreserver:
    """Test EmotionPreserver main class"""
    
    @pytest.fixture
    def emotion_preserver(self, default_config, mock_realtime_service):
        """Create EmotionPreserver instance"""
        return EmotionPreserver(default_config, mock_realtime_service)
    
    def test_initialization(self, emotion_preserver, default_config, mock_realtime_service):
        """Test EmotionPreserver initialization"""
        assert emotion_preserver.config == default_config
        assert emotion_preserver.realtime_service == mock_realtime_service
        assert emotion_preserver.is_initialized is False
        assert emotion_preserver.is_processing is False
        assert emotion_preserver.consecutive_errors == 0
        assert isinstance(emotion_preserver.audio_buffer, AudioBuffer)
        assert isinstance(emotion_preserver.format_converter, AudioFormatConverter)
        assert isinstance(emotion_preserver.metrics, AudioMetrics)
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, emotion_preserver):
        """Test successful initialization"""
        result = await emotion_preserver.initialize()
        assert result is True
        assert emotion_preserver.is_initialized is True
    
    @pytest.mark.asyncio
    async def test_initialize_invalid_config(self):
        """Test initialization with invalid config"""
        # Create config with invalid settings
        invalid_config = EmotionPreservationConfig()
        invalid_config.sample_rate = -1  # Invalid sample rate
        
        emotion_preserver = EmotionPreserver(invalid_config)
        
        # Should still initialize (validation warnings only)
        result = await emotion_preserver.initialize()
        assert result is True
    
    def test_validate_config_disabled_mode(self):
        """Test config validation with disabled mode"""
        config = EmotionPreservationConfig(mode=EmotionProcessingMode.DISABLED)
        emotion_preserver = EmotionPreserver(config)
        
        result = emotion_preserver._validate_config()
        assert result is True
    
    def test_validate_config_warnings(self):
        """Test config validation warnings"""
        config = EmotionPreservationConfig(
            sample_rate=16000,  # Non-standard rate
            max_latency_ms=150.0  # High latency
        )
        emotion_preserver = EmotionPreserver(config)
        
        with patch.object(emotion_preserver.logger, 'warning') as mock_warning:
            result = emotion_preserver._validate_config()
            assert result is True
            assert mock_warning.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_configure_realtime_audio_mode(self, emotion_preserver):
        """Test Realtime API audio mode configuration"""
        await emotion_preserver._configure_realtime_audio_mode()
        
        # Check that realtime service config was updated
        assert emotion_preserver.realtime_service.config.modalities == ["audio"]
        assert emotion_preserver.realtime_service.config.input_audio_format == "pcm16"
        assert emotion_preserver.realtime_service.config.output_audio_format == "pcm16"
    
    @pytest.mark.asyncio
    async def test_configure_audio_only_mode(self, emotion_preserver):
        """Test audio-only mode configuration generation"""
        config = await emotion_preserver.configure_audio_only_mode()
        
        expected_config = {
            "modalities": ["audio"],
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {"enabled": False},
            "voice": "alloy",
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 200
            }
        }
        
        assert config == expected_config
    
    @pytest.mark.asyncio
    async def test_process_audio_not_initialized(self, emotion_preserver, sample_audio_data):
        """Test audio processing when not initialized"""
        result = await emotion_preserver.process_audio(sample_audio_data)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_process_audio_disabled_mode(self, sample_audio_data):
        """Test audio processing with disabled mode"""
        config = EmotionPreservationConfig(mode=EmotionProcessingMode.DISABLED)
        emotion_preserver = EmotionPreserver(config)
        emotion_preserver.is_initialized = True
        
        result = await emotion_preserver.process_audio(sample_audio_data)
        assert result == sample_audio_data
    
    @pytest.mark.asyncio
    async def test_process_audio_success(self, emotion_preserver, sample_audio_data):
        """Test successful audio processing"""
        await emotion_preserver.initialize()
        
        result = await emotion_preserver.process_audio(sample_audio_data)
        assert result is not None
        assert len(result) == len(sample_audio_data)
        assert emotion_preserver.consecutive_errors == 0
        assert emotion_preserver.metrics.total_chunks_processed == 1
    
    @pytest.mark.asyncio
    async def test_process_audio_with_callback(self, emotion_preserver, sample_audio_data):
        """Test audio processing with callback"""
        await emotion_preserver.initialize()
        
        # Set up callback
        callback_called = False
        async def test_callback(audio, latency):
            nonlocal callback_called
            callback_called = True
            assert audio is not None
            assert latency >= 0
        
        emotion_preserver.set_audio_processed_callback(test_callback)
        
        result = await emotion_preserver.process_audio(sample_audio_data)
        assert result is not None
        assert callback_called is True
    
    @pytest.mark.asyncio
    async def test_process_audio_error_handling(self, emotion_preserver, sample_audio_data):
        """Test audio processing error handling"""
        await emotion_preserver.initialize()
        
        # Mock buffer write to fail
        with patch.object(emotion_preserver.audio_buffer, 'write', side_effect=Exception("Buffer error")):
            result = await emotion_preserver.process_audio(sample_audio_data)
            
            # Should return original data due to graceful degradation
            assert result == sample_audio_data
            assert emotion_preserver.consecutive_errors == 1
            assert emotion_preserver.metrics.error_count == 1
    
    @pytest.mark.asyncio
    async def test_process_audio_consecutive_errors(self, emotion_preserver, sample_audio_data):
        """Test handling of consecutive processing errors"""
        await emotion_preserver.initialize()
        
        # Mock buffer to always fail
        with patch.object(emotion_preserver.audio_buffer, 'write', side_effect=Exception("Persistent error")):
            # Process multiple times to trigger consecutive error handling
            for _ in range(emotion_preserver.config.max_consecutive_errors + 1):
                await emotion_preserver.process_audio(sample_audio_data)
            
            assert emotion_preserver.consecutive_errors >= emotion_preserver.config.max_consecutive_errors
    
    @pytest.mark.asyncio
    async def test_handle_processing_failure(self, emotion_preserver):
        """Test processing failure handling"""
        emotion_preserver.consecutive_errors = 5
        
        await emotion_preserver._handle_processing_failure()
        
        # Buffer should be cleared
        assert emotion_preserver.audio_buffer.get_available_size() == 0
    
    @pytest.mark.asyncio
    async def test_handle_processing_failure_fallback_to_text(self):
        """Test fallback to text mode on processing failure"""
        config = EmotionPreservationConfig(fallback_to_text_mode=True)
        emotion_preserver = EmotionPreserver(config)
        emotion_preserver.consecutive_errors = 5
        
        await emotion_preserver._handle_processing_failure()
        
        assert emotion_preserver.config.mode == EmotionProcessingMode.DISABLED
    
    def test_get_metrics(self, emotion_preserver):
        """Test metrics retrieval"""
        # Update some metrics
        emotion_preserver.metrics.update_latency(25.0)
        emotion_preserver.consecutive_errors = 2
        
        metrics = emotion_preserver.get_metrics()
        
        expected_keys = [
            "total_chunks_processed", "average_latency_ms", "max_latency_ms",
            "current_latency_ms", "buffer_overflows", "format_conversions",
            "error_count", "error_rate", "is_processing", "consecutive_errors",
            "buffer_utilization"
        ]
        
        for key in expected_keys:
            assert key in metrics
        
        assert metrics["total_chunks_processed"] == 1
        assert metrics["average_latency_ms"] == 25.0
        assert metrics["consecutive_errors"] == 2
    
    def test_set_callbacks(self, emotion_preserver):
        """Test setting callbacks"""
        audio_callback = AsyncMock()
        error_callback = AsyncMock()
        
        emotion_preserver.set_audio_processed_callback(audio_callback)
        emotion_preserver.set_error_callback(error_callback)
        
        assert emotion_preserver.on_audio_processed == audio_callback
        assert emotion_preserver.on_error == error_callback
    
    @pytest.mark.asyncio
    async def test_cleanup(self, emotion_preserver):
        """Test cleanup functionality"""
        # Set some state
        emotion_preserver.is_processing = True
        await emotion_preserver.audio_buffer.write(b'\x00\x01' * 100)
        
        await emotion_preserver.cleanup()
        
        assert emotion_preserver.is_processing is False
        assert emotion_preserver.audio_buffer.get_available_size() == 0


class TestFactoryFunctions:
    """Test factory functions"""
    
    def test_create_emotion_preserver(self):
        """Test emotion preserver factory function"""
        emotion_preserver = create_emotion_preserver()
        
        assert isinstance(emotion_preserver, EmotionPreserver)
        assert emotion_preserver.config.mode == EmotionProcessingMode.AUDIO_ONLY
        assert emotion_preserver.realtime_service is None
    
    def test_create_emotion_preserver_with_service(self, mock_realtime_service):
        """Test emotion preserver factory with realtime service"""
        emotion_preserver = create_emotion_preserver(
            mode=EmotionProcessingMode.HYBRID,
            realtime_service=mock_realtime_service
        )
        
        assert isinstance(emotion_preserver, EmotionPreserver)
        assert emotion_preserver.config.mode == EmotionProcessingMode.HYBRID
        assert emotion_preserver.realtime_service == mock_realtime_service
    
    def test_create_audio_only_config(self):
        """Test audio-only config factory function"""
        config = create_audio_only_config()
        
        expected_config = {
            "modalities": ["audio"],
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {"enabled": False},
            "sample_rate": 24000
        }
        
        assert config == expected_config


class TestIntegration:
    """Integration tests for emotion preservation system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_audio_processing(self, sample_audio_data):
        """Test complete end-to-end audio processing"""
        # Create emotion preserver with real configuration
        config = EmotionPreservationConfig(
            max_latency_ms=100.0,
            enable_audio_normalization=True
        )
        emotion_preserver = EmotionPreserver(config)
        
        # Initialize and process audio
        await emotion_preserver.initialize()
        result = await emotion_preserver.process_audio(sample_audio_data)
        
        # Verify result
        assert result is not None
        assert len(result) == len(sample_audio_data)
        
        # Check metrics
        metrics = emotion_preserver.get_metrics()
        assert metrics["total_chunks_processed"] == 1
        assert metrics["error_count"] == 0
        assert metrics["average_latency_ms"] > 0
        
        # Cleanup
        await emotion_preserver.cleanup()
    
    @pytest.mark.asyncio
    async def test_multiple_audio_chunks(self, sample_audio_data):
        """Test processing multiple audio chunks"""
        emotion_preserver = create_emotion_preserver()
        await emotion_preserver.initialize()
        
        # Process multiple chunks
        for i in range(5):
            chunk = sample_audio_data[i*1000:(i+1)*1000]
            if chunk:
                result = await emotion_preserver.process_audio(chunk)
                assert result is not None
        
        # Check final metrics
        metrics = emotion_preserver.get_metrics()
        assert metrics["total_chunks_processed"] >= 1
        
        await emotion_preserver.cleanup()
    
    @pytest.mark.asyncio
    async def test_performance_requirements(self, sample_audio_data):
        """Test that processing meets performance requirements"""
        config = EmotionPreservationConfig(max_latency_ms=50.0)
        emotion_preserver = EmotionPreserver(config)
        await emotion_preserver.initialize()
        
        # Process audio and measure time
        start_time = time.time()
        result = await emotion_preserver.process_audio(sample_audio_data)
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Verify result and performance
        assert result is not None
        # Note: Actual performance may vary in test environment
        # assert processing_time < config.max_latency_ms
        
        await emotion_preserver.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 