"""
Tests for the audio module
"""

import pytest
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add the parent directory to sys.path to import assistant
sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant.audio import (
    AudioManager, 
    AudioConfig, 
    AudioDevice, 
    AudioDeviceType,
    create_audio_manager
)


class TestAudioConfig:
    """Test AudioConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = AudioConfig()
        
        assert config.sample_rate == 16000  # 16kHz for Whisper
        assert config.channels == 1  # Mono
        assert config.chunk_size == 1024
        assert config.input_device_index is None
        assert config.output_device_index is None
        assert config.enable_noise_suppression is True
        assert config.enable_agc is True
        assert config.input_volume == 1.0
        assert config.output_volume == 1.0
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = AudioConfig(
            sample_rate=44100,
            channels=2,
            chunk_size=2048,
            input_device_index=1,
            output_device_index=2,
            input_volume=0.8,
            output_volume=1.2
        )
        
        assert config.sample_rate == 44100
        assert config.channels == 2
        assert config.chunk_size == 2048
        assert config.input_device_index == 1
        assert config.output_device_index == 2
        assert config.input_volume == 0.8
        assert config.output_volume == 1.2


class TestAudioDevice:
    """Test AudioDevice dataclass"""
    
    def test_audio_device_creation(self):
        """Test AudioDevice creation"""
        device = AudioDevice(
            index=0,
            name="Test Microphone",
            device_type=AudioDeviceType.INPUT,
            max_input_channels=2,
            max_output_channels=0,
            default_sample_rate=44100.0,
            is_default=True
        )
        
        assert device.index == 0
        assert device.name == "Test Microphone"
        assert device.device_type == AudioDeviceType.INPUT
        assert device.max_input_channels == 2
        assert device.max_output_channels == 0
        assert device.default_sample_rate == 44100.0
        assert device.is_default is True


class TestAudioManager:
    """Test AudioManager class"""
    
    @pytest.fixture
    def mock_pyaudio(self):
        """Mock PyAudio for testing"""
        with patch('assistant.audio.pyaudio.PyAudio') as mock_pa:
            mock_instance = Mock()
            mock_pa.return_value = mock_instance
            
            # Mock device info
            mock_instance.get_device_count.return_value = 3
            mock_instance.get_default_input_device_info.return_value = {'index': 0}
            mock_instance.get_default_output_device_info.return_value = {'index': 1}
            
            # Mock device info for each device
            device_infos = [
                {
                    'index': 0,
                    'name': 'Test Microphone',
                    'maxInputChannels': 2,
                    'maxOutputChannels': 0,
                    'defaultSampleRate': 44100.0
                },
                {
                    'index': 1,
                    'name': 'Test Speaker',
                    'maxInputChannels': 0,
                    'maxOutputChannels': 2,
                    'defaultSampleRate': 44100.0
                },
                {
                    'index': 2,
                    'name': 'Test Headset',
                    'maxInputChannels': 1,
                    'maxOutputChannels': 2,
                    'defaultSampleRate': 48000.0
                }
            ]
            
            mock_instance.get_device_info_by_index.side_effect = lambda i: device_infos[i]
            
            yield mock_instance
    
    def test_audio_manager_initialization(self, mock_pyaudio):
        """Test AudioManager initialization"""
        config = AudioConfig()
        manager = AudioManager(config)
        
        assert manager.config == config
        assert manager.pyaudio_instance is None
        assert manager.input_stream is None
        assert manager.output_stream is None
        assert manager.is_recording is False
        assert manager.is_playing is False
    
    def test_audio_manager_initialize(self, mock_pyaudio):
        """Test AudioManager initialize method"""
        config = AudioConfig()
        manager = AudioManager(config)
        
        manager.initialize()
        
        assert manager.pyaudio_instance is not None
        assert manager.config.input_device_index == 0  # Default input
        assert manager.config.output_device_index == 1  # Default output
        
        # Verify device detection was called
        mock_pyaudio.get_device_count.assert_called_once()
        # Note: These methods are called twice (once for detection, once for default assignment)
        assert mock_pyaudio.get_default_input_device_info.call_count >= 1
        assert mock_pyaudio.get_default_output_device_info.call_count >= 1
    
    def test_detect_devices(self, mock_pyaudio):
        """Test device detection"""
        config = AudioConfig()
        manager = AudioManager(config)
        manager.initialize()
        
        devices = manager._detect_devices()
        
        # Should find 3 devices based on our mock
        assert len(devices) == 3
        
        # Check first device (microphone)
        assert devices[0].name == 'Test Microphone'
        assert devices[0].device_type == AudioDeviceType.INPUT
        assert devices[0].is_default is True
        
        # Check second device (speaker)
        assert devices[1].name == 'Test Speaker'
        assert devices[1].device_type == AudioDeviceType.OUTPUT
        assert devices[1].is_default is True
    
    def test_setup_input_stream(self, mock_pyaudio):
        """Test input stream setup"""
        config = AudioConfig(input_device_index=0)
        manager = AudioManager(config)
        manager.initialize()
        
        # Mock the stream
        mock_stream = Mock()
        mock_pyaudio.open.return_value = mock_stream
        
        result = manager.setup_input_stream()
        
        assert result is True
        assert manager.input_stream == mock_stream
        
        # Verify stream was opened with correct parameters
        mock_pyaudio.open.assert_called_with(
            format=config.format,
            channels=config.channels,
            rate=config.sample_rate,
            input=True,
            input_device_index=0,
            frames_per_buffer=config.chunk_size,
            stream_callback=None
        )
    
    def test_setup_output_stream(self, mock_pyaudio):
        """Test output stream setup"""
        config = AudioConfig(output_device_index=1)
        manager = AudioManager(config)
        manager.initialize()
        
        # Mock the stream
        mock_stream = Mock()
        mock_pyaudio.open.return_value = mock_stream
        
        result = manager.setup_output_stream()
        
        assert result is True
        assert manager.output_stream == mock_stream
        
        # Verify stream was opened with correct parameters
        mock_pyaudio.open.assert_called_with(
            format=config.format,
            channels=config.channels,
            rate=config.sample_rate,
            output=True,
            output_device_index=1,
            frames_per_buffer=config.chunk_size
        )
    
    def test_recording_lifecycle(self, mock_pyaudio):
        """Test recording start/stop lifecycle"""
        config = AudioConfig(input_device_index=0)
        manager = AudioManager(config)
        manager.initialize()
        
        # Mock the stream
        mock_stream = Mock()
        mock_pyaudio.open.return_value = mock_stream
        
        # Start recording
        result = manager.start_recording()
        assert result is True
        assert manager.is_recording is True
        mock_stream.start_stream.assert_called_once()
        
        # Stop recording
        result = manager.stop_recording()
        assert result is True
        assert manager.is_recording is False
        mock_stream.stop_stream.assert_called_once()
    
    def test_cleanup(self, mock_pyaudio):
        """Test cleanup method"""
        config = AudioConfig()
        manager = AudioManager(config)
        manager.initialize()
        
        # Setup mock streams
        mock_input_stream = Mock()
        mock_output_stream = Mock()
        manager.input_stream = mock_input_stream
        manager.output_stream = mock_output_stream
        
        manager.cleanup()
        
        # Verify streams were closed
        mock_input_stream.stop_stream.assert_called_once()
        mock_input_stream.close.assert_called_once()
        mock_output_stream.stop_stream.assert_called_once()
        mock_output_stream.close.assert_called_once()
        
        # Verify PyAudio was terminated
        mock_pyaudio.terminate.assert_called_once()
        
        assert manager.input_stream is None
        assert manager.output_stream is None
        assert manager.pyaudio_instance is None
    
    def test_read_audio_chunk(self, mock_pyaudio):
        """Test reading audio chunk"""
        config = AudioConfig(input_device_index=0)
        manager = AudioManager(config)
        manager.initialize()
        
        # Mock the stream
        mock_stream = Mock()
        mock_pyaudio.open.return_value = mock_stream
        
        # Setup recording
        manager.start_recording()
        
        # Mock audio data
        test_data = b'test_audio_data'
        mock_stream.read.return_value = test_data
        
        result = manager.read_audio_chunk()
        
        assert result == test_data
        mock_stream.read.assert_called_with(config.chunk_size, exception_on_overflow=False)
    
    def test_play_audio_chunk(self, mock_pyaudio):
        """Test playing audio chunk"""
        config = AudioConfig(output_device_index=1)
        manager = AudioManager(config)
        manager.initialize()
        
        # Mock the stream
        mock_stream = Mock()
        mock_pyaudio.open.return_value = mock_stream
        
        # Setup output stream
        manager.setup_output_stream()
        
        # Test audio data
        test_data = b'test_audio_data'
        
        result = manager.play_audio_chunk(test_data)
        
        assert result is True
        mock_stream.write.assert_called_with(test_data)


class TestAudioManagerFactory:
    """Test audio manager factory function"""
    
    @patch('assistant.audio.pyaudio.PyAudio')
    def test_create_audio_manager_default(self, mock_pyaudio):
        """Test creating AudioManager with default config"""
        mock_instance = Mock()
        mock_pyaudio.return_value = mock_instance
        
        # Mock device info
        mock_instance.get_device_count.return_value = 1
        mock_instance.get_default_input_device_info.return_value = {'index': 0}
        mock_instance.get_default_output_device_info.return_value = {'index': 0}
        mock_instance.get_device_info_by_index.return_value = {
            'index': 0,
            'name': 'Test Device',
            'maxInputChannels': 2,
            'maxOutputChannels': 2,
            'defaultSampleRate': 44100.0
        }
        
        manager = create_audio_manager()
        
        assert isinstance(manager, AudioManager)
        assert manager.config.sample_rate == 16000
        assert manager.pyaudio_instance is not None
    
    @patch('assistant.audio.pyaudio.PyAudio')
    def test_create_audio_manager_custom(self, mock_pyaudio):
        """Test creating AudioManager with custom config"""
        mock_instance = Mock()
        mock_pyaudio.return_value = mock_instance
        
        # Mock device info
        mock_instance.get_device_count.return_value = 1
        mock_instance.get_default_input_device_info.return_value = {'index': 0}
        mock_instance.get_default_output_device_info.return_value = {'index': 0}
        mock_instance.get_device_info_by_index.return_value = {
            'index': 0,
            'name': 'Test Device',
            'maxInputChannels': 2,
            'maxOutputChannels': 2,
            'defaultSampleRate': 44100.0
        }
        
        custom_config = AudioConfig(sample_rate=48000, channels=2)
        manager = create_audio_manager(custom_config)
        
        assert isinstance(manager, AudioManager)
        assert manager.config.sample_rate == 48000
        assert manager.config.channels == 2
        assert manager.pyaudio_instance is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 