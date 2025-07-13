"""
Test suite for VoiceAssistantPipeline module

This test suite covers:
- Pipeline initialization and configuration
- State management and transitions
- Push-to-talk functionality
- Audio processing pipeline
- Integration with STT and TTS services
- Error handling and recovery
- Statistics and monitoring
"""

import pytest
import asyncio
import threading
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass
from typing import Optional, Dict, Any

from assistant.pipeline import (
    VoiceAssistantPipeline,
    PipelineConfig,
    PipelineState,
    PipelineStatistics,
    load_config_from_yaml,
    create_pipeline_from_config
)
from assistant.audio import AudioManager, AudioConfig
from assistant.stt import WhisperSTTService, STTConfig, STTResult
from assistant.tts import OpenAITTSService, TTSConfig, TTSResult


class TestPipelineConfig:
    """Test PipelineConfig dataclass"""
    
    def test_pipeline_config_defaults(self):
        """Test PipelineConfig default values"""
        config = PipelineConfig()
        
        assert config.trigger_key == "space"
        assert config.trigger_key_name == "spacebar"
        assert config.recording_chunk_size == 1024
        assert config.max_recording_duration == 30.0
        assert config.min_recording_duration == 0.5
        assert config.process_timeout == 30.0
        assert config.concurrent_processing is True
        assert config.playback_volume == 1.0
        assert config.interrupt_on_new_recording is True
        assert config.latency_target == 0.8
        assert config.max_retries == 3
        assert config.log_state_changes is True
        assert config.collect_statistics is True
    
    def test_pipeline_config_custom_values(self):
        """Test PipelineConfig with custom values"""
        config = PipelineConfig(
            trigger_key="ctrl",
            trigger_key_name="control",
            max_recording_duration=60.0,
            min_recording_duration=1.0,
            latency_target=0.5,
            log_state_changes=False
        )
        
        assert config.trigger_key == "ctrl"
        assert config.trigger_key_name == "control"
        assert config.max_recording_duration == 60.0
        assert config.min_recording_duration == 1.0
        assert config.latency_target == 0.5
        assert config.log_state_changes is False


class TestPipelineStatistics:
    """Test PipelineStatistics dataclass"""
    
    def test_pipeline_statistics_defaults(self):
        """Test PipelineStatistics default values"""
        stats = PipelineStatistics()
        
        assert stats.total_sessions == 0
        assert stats.successful_sessions == 0
        assert stats.failed_sessions == 0
        assert stats.total_recording_time == 0.0
        assert stats.total_processing_time == 0.0
        assert stats.total_playback_time == 0.0
        assert stats.average_latency == 0.0
        assert stats.min_latency == float('inf')
        assert stats.max_latency == 0.0
        assert stats.stt_success_rate == 0.0
        assert stats.tts_success_rate == 0.0
    
    def test_update_session_success(self):
        """Test updating statistics for successful session"""
        stats = PipelineStatistics()
        
        stats.update_session(
            success=True,
            recording_time=2.0,
            processing_time=1.5,
            playback_time=3.0
        )
        
        assert stats.total_sessions == 1
        assert stats.successful_sessions == 1
        assert stats.failed_sessions == 0
        assert stats.total_recording_time == 2.0
        assert stats.total_processing_time == 1.5
        assert stats.total_playback_time == 3.0
        assert stats.average_latency == 4.5  # processing + playback
        assert stats.min_latency == 4.5
        assert stats.max_latency == 4.5
    
    def test_update_session_failure(self):
        """Test updating statistics for failed session"""
        stats = PipelineStatistics()
        
        stats.update_session(
            success=False,
            recording_time=1.0,
            processing_time=0.5,
            playback_time=0.0
        )
        
        assert stats.total_sessions == 1
        assert stats.successful_sessions == 0
        assert stats.failed_sessions == 1
        assert stats.total_recording_time == 1.0
        assert stats.total_processing_time == 0.5
        assert stats.total_playback_time == 0.0
    
    def test_update_multiple_sessions(self):
        """Test updating statistics for multiple sessions"""
        stats = PipelineStatistics()
        
        # First session
        stats.update_session(True, 1.0, 1.0, 2.0)
        assert stats.average_latency == 3.0
        
        # Second session  
        stats.update_session(True, 1.5, 2.0, 1.5)
        assert stats.average_latency == 3.25  # (3.0 + 3.5) / 2
        assert stats.min_latency == 3.0
        assert stats.max_latency == 3.5


class TestVoiceAssistantPipeline:
    """Test VoiceAssistantPipeline class"""
    
    @pytest.fixture
    def mock_audio_manager(self):
        """Mock AudioManager for testing"""
        mock = Mock(spec=AudioManager)
        mock.initialize = Mock(return_value=True)
        mock.setup_input_stream = Mock(return_value=True)
        mock.setup_output_stream = Mock(return_value=True)
        mock.start_recording = Mock(return_value=True)
        mock.stop_recording = Mock(return_value=True)
        mock.read_audio_chunk = Mock(return_value=b'test_audio_data')
        mock.play_audio_chunk = Mock(return_value=True)
        mock.cleanup = Mock()
        mock.is_recording = False
        return mock
    
    @pytest.fixture
    def mock_stt_service(self):
        """Mock WhisperSTTService for testing"""
        mock = Mock(spec=WhisperSTTService)
        mock.initialize = Mock(return_value=True)
        mock.transcribe_audio = AsyncMock(return_value=STTResult(
            text="test transcription",
            confidence=0.95,
            language="en",
            duration=2.0,
            processing_time=0.5
        ))
        mock.get_statistics = Mock(return_value={})
        mock.reset_statistics = Mock()
        return mock
    
    @pytest.fixture
    def mock_tts_service(self):
        """Mock OpenAITTSService for testing"""
        mock = Mock(spec=OpenAITTSService)
        mock.initialize = Mock(return_value=True)
        mock.synthesize_speech = AsyncMock(return_value=TTSResult(
            audio_data=b'test_audio_data',
            text="test response",
            duration=2.0,
            processing_time=0.8,
            format="mp3",
            voice="alloy",
            speed=1.0,
            sample_rate=16000
        ))
        mock.get_wav_audio = Mock(return_value=b'test_wav_audio')
        mock.get_statistics = Mock(return_value={})
        mock.reset_statistics = Mock()
        return mock
    
    @pytest.fixture
    def pipeline_config(self):
        """Pipeline configuration for testing"""
        return PipelineConfig(
            trigger_key="space",
            max_recording_duration=5.0,
            min_recording_duration=0.1,
            collect_statistics=True
        )
    
    @pytest.fixture
    def pipeline(self, pipeline_config, mock_audio_manager, mock_stt_service, mock_tts_service):
        """VoiceAssistantPipeline instance for testing"""
        return VoiceAssistantPipeline(
            config=pipeline_config,
            audio_manager=mock_audio_manager,
            stt_service=mock_stt_service,
            tts_service=mock_tts_service
        )
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization"""
        assert pipeline.state == PipelineState.IDLE
        assert pipeline.is_running is False
        assert pipeline.is_push_to_talk_active is False
        assert pipeline.statistics is not None
        assert pipeline.response_callback is not None
    
    def test_default_response_callback(self, pipeline):
        """Test default response callback"""
        text = "hello world"
        response = pipeline._default_response_callback(text)
        assert response == "I heard you say: hello world"
    
    def test_custom_response_callback(self, pipeline_config, mock_audio_manager, mock_stt_service, mock_tts_service):
        """Test custom response callback"""
        def custom_callback(text):
            return f"Custom response to: {text}"
        
        pipeline = VoiceAssistantPipeline(
            config=pipeline_config,
            audio_manager=mock_audio_manager,
            stt_service=mock_stt_service,
            tts_service=mock_tts_service,
            response_callback=custom_callback
        )
        
        response = pipeline.response_callback("test")
        assert response == "Custom response to: test"
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, pipeline, mock_audio_manager, mock_stt_service, mock_tts_service):
        """Test successful pipeline initialization"""
        result = await pipeline.initialize()
        
        assert result is True
        mock_audio_manager.initialize.assert_called_once()
        mock_stt_service.initialize.assert_called_once()
        mock_tts_service.initialize.assert_called_once()
        mock_audio_manager.setup_input_stream.assert_called_once()
        mock_audio_manager.setup_output_stream.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_stt_failure(self, pipeline, mock_audio_manager, mock_stt_service, mock_tts_service):
        """Test pipeline initialization with STT service failure"""
        mock_stt_service.initialize.return_value = False
        
        result = await pipeline.initialize()
        
        assert result is False
        mock_audio_manager.initialize.assert_called_once()
        mock_stt_service.initialize.assert_called_once()
        mock_tts_service.initialize.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_initialize_tts_failure(self, pipeline, mock_audio_manager, mock_stt_service, mock_tts_service):
        """Test pipeline initialization with TTS service failure"""
        mock_tts_service.initialize.return_value = False
        
        result = await pipeline.initialize()
        
        assert result is False
        mock_audio_manager.initialize.assert_called_once()
        mock_stt_service.initialize.assert_called_once()
        mock_tts_service.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_audio_setup_failure(self, pipeline, mock_audio_manager, mock_stt_service, mock_tts_service):
        """Test pipeline initialization with audio setup failure"""
        mock_audio_manager.setup_input_stream.return_value = False
        
        result = await pipeline.initialize()
        
        assert result is False
        mock_audio_manager.setup_input_stream.assert_called_once()
        mock_audio_manager.setup_output_stream.assert_not_called()
    
    def test_get_state(self, pipeline):
        """Test getting pipeline state"""
        assert pipeline.get_state() == PipelineState.IDLE
        
        pipeline.state = PipelineState.RECORDING
        assert pipeline.get_state() == PipelineState.RECORDING
    
    def test_set_state(self, pipeline):
        """Test setting pipeline state"""
        # Test state change callback
        state_changes = []
        def on_state_changed(old_state, new_state):
            state_changes.append((old_state, new_state))
        
        pipeline.on_state_changed = on_state_changed
        
        pipeline._set_state(PipelineState.RECORDING)
        
        assert pipeline.state == PipelineState.RECORDING
        assert len(state_changes) == 1
        assert state_changes[0] == (PipelineState.IDLE, PipelineState.RECORDING)
    
    def test_push_to_talk_press_from_idle(self, pipeline):
        """Test push-to-talk press from idle state"""
        pipeline.state = PipelineState.IDLE
        
        pipeline._handle_push_to_talk_press()
        
        assert pipeline.is_push_to_talk_active is True
        assert pipeline.state == PipelineState.RECORDING
    
    def test_push_to_talk_press_during_playback(self, pipeline):
        """Test push-to-talk press during playback (interrupt)"""
        pipeline.state = PipelineState.PLAYING
        pipeline.config.interrupt_on_new_recording = True
        
        pipeline._handle_push_to_talk_press()
        
        assert pipeline.is_push_to_talk_active is True
        assert pipeline.state == PipelineState.RECORDING
    
    def test_push_to_talk_press_no_interrupt(self, pipeline):
        """Test push-to-talk press during playback without interruption"""
        pipeline.state = PipelineState.PLAYING
        pipeline.config.interrupt_on_new_recording = False
        
        pipeline._handle_push_to_talk_press()
        
        assert pipeline.is_push_to_talk_active is False
        assert pipeline.state == PipelineState.PLAYING
    
    def test_push_to_talk_release(self, pipeline):
        """Test push-to-talk release"""
        pipeline.state = PipelineState.RECORDING
        
        pipeline._handle_push_to_talk_release()
        
        assert pipeline.is_push_to_talk_active is False
        assert pipeline.state == PipelineState.PROCESSING
    
    def test_push_to_talk_release_wrong_state(self, pipeline):
        """Test push-to-talk release from wrong state"""
        pipeline.state = PipelineState.IDLE
        
        pipeline._handle_push_to_talk_release()
        
        assert pipeline.is_push_to_talk_active is False
        assert pipeline.state == PipelineState.IDLE
    
    @pytest.mark.asyncio
    async def test_handle_recording_start(self, pipeline, mock_audio_manager):
        """Test handling recording start"""
        pipeline.state = PipelineState.RECORDING
        pipeline.is_push_to_talk_active = True
        pipeline.is_running = True
        
        # Mock audio manager behavior
        mock_audio_manager.is_recording = False
        mock_audio_manager.read_audio_chunk.return_value = b'chunk1'
        
        # Run recording for a short time
        task = asyncio.create_task(pipeline._handle_recording())
        await asyncio.sleep(0.05)  # Let it start
        
        pipeline.is_push_to_talk_active = False  # Stop recording
        await task
        
        mock_audio_manager.start_recording.assert_called_once()
        mock_audio_manager.stop_recording.assert_called_once()
        assert len(pipeline._audio_buffer) > 0
    
    @pytest.mark.asyncio
    async def test_handle_recording_max_duration(self, pipeline, mock_audio_manager):
        """Test handling recording maximum duration"""
        pipeline.state = PipelineState.RECORDING
        pipeline.is_push_to_talk_active = True
        pipeline.is_running = True
        pipeline.config.max_recording_duration = 0.1  # Very short for testing
        
        # Mock audio manager behavior
        mock_audio_manager.is_recording = False
        mock_audio_manager.read_audio_chunk.return_value = b'chunk1'
        
        # Run recording until max duration
        await pipeline._handle_recording()
        
        mock_audio_manager.start_recording.assert_called_once()
        mock_audio_manager.stop_recording.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_processing_success(self, pipeline, mock_stt_service, mock_tts_service):
        """Test successful processing"""
        pipeline.state = PipelineState.PROCESSING
        pipeline._audio_buffer = [b'chunk1', b'chunk2']
        pipeline._recording_start_time = time.time() - 1.0  # 1 second ago
        
        await pipeline._handle_processing()
        
        assert pipeline.state == PipelineState.PLAYING
        mock_stt_service.transcribe_audio.assert_called_once()
        mock_tts_service.synthesize_speech.assert_called_once()
        mock_tts_service.get_wav_audio.assert_called_once()
        assert hasattr(pipeline, '_playback_audio')
    
    @pytest.mark.asyncio
    async def test_handle_processing_no_audio(self, pipeline):
        """Test processing with no audio data"""
        pipeline.state = PipelineState.PROCESSING
        pipeline._audio_buffer = []
        
        await pipeline._handle_processing()
        
        assert pipeline.state == PipelineState.IDLE
    
    @pytest.mark.asyncio
    async def test_handle_processing_short_recording(self, pipeline):
        """Test processing with recording too short"""
        pipeline.state = PipelineState.PROCESSING
        pipeline._audio_buffer = [b'chunk1']
        pipeline._recording_start_time = time.time() - 0.05  # Very short recording
        pipeline.config.min_recording_duration = 0.1
        
        await pipeline._handle_processing()
        
        assert pipeline.state == PipelineState.IDLE
    
    @pytest.mark.asyncio
    async def test_handle_processing_no_transcription(self, pipeline, mock_stt_service):
        """Test processing with no transcription result"""
        pipeline.state = PipelineState.PROCESSING
        pipeline._audio_buffer = [b'chunk1']
        pipeline._recording_start_time = time.time() - 1.0
        
        mock_stt_service.transcribe_audio.return_value = None
        
        await pipeline._handle_processing()
        
        assert pipeline.state == PipelineState.IDLE
    
    @pytest.mark.asyncio
    async def test_handle_processing_empty_text(self, pipeline, mock_stt_service):
        """Test processing with empty transcription text"""
        pipeline.state = PipelineState.PROCESSING
        pipeline._audio_buffer = [b'chunk1']
        pipeline._recording_start_time = time.time() - 1.0
        
        mock_stt_service.transcribe_audio.return_value = STTResult(
            text="",
            confidence=0.95,
            language="en",
            duration=1.0,
            processing_time=0.5
        )
        
        await pipeline._handle_processing()
        
        assert pipeline.state == PipelineState.IDLE
    
    @pytest.mark.asyncio
    async def test_handle_processing_tts_failure(self, pipeline, mock_tts_service):
        """Test processing with TTS failure"""
        pipeline.state = PipelineState.PROCESSING
        pipeline._audio_buffer = [b'chunk1']
        pipeline._recording_start_time = time.time() - 1.0
        
        mock_tts_service.synthesize_speech.return_value = None
        
        await pipeline._handle_processing()
        
        assert pipeline.state == PipelineState.IDLE
    
    @pytest.mark.asyncio
    async def test_handle_processing_wav_conversion_failure(self, pipeline, mock_tts_service):
        """Test processing with WAV conversion failure"""
        pipeline.state = PipelineState.PROCESSING
        pipeline._audio_buffer = [b'chunk1']
        pipeline._recording_start_time = time.time() - 1.0
        
        mock_tts_service.get_wav_audio.return_value = None
        
        await pipeline._handle_processing()
        
        assert pipeline.state == PipelineState.IDLE
    
    @pytest.mark.asyncio
    async def test_handle_processing_callbacks(self, pipeline):
        """Test processing callbacks"""
        pipeline.state = PipelineState.PROCESSING
        pipeline._audio_buffer = [b'chunk1']
        pipeline._recording_start_time = time.time() - 1.0
        
        transcription_received = Mock()
        response_generated = Mock()
        
        pipeline.on_transcription_received = transcription_received
        pipeline.on_response_generated = response_generated
        
        await pipeline._handle_processing()
        
        transcription_received.assert_called_once_with("test transcription")
        response_generated.assert_called_once_with("I heard you say: test transcription")
    
    @pytest.mark.asyncio
    async def test_handle_playing_success(self, pipeline, mock_audio_manager):
        """Test successful audio playback"""
        pipeline.state = PipelineState.PLAYING
        pipeline._playback_audio = b'test_wav_audio'
        pipeline._session_start_time = time.time() - 2.0
        pipeline._processing_time = 1.0
        
        await pipeline._handle_playing()
        
        assert pipeline.state == PipelineState.IDLE
        mock_audio_manager.play_audio_chunk.assert_called_once_with(b'test_wav_audio')
        assert not hasattr(pipeline, '_playback_audio')
        assert pipeline._audio_buffer == []
    
    @pytest.mark.asyncio
    async def test_handle_playing_no_audio(self, pipeline):
        """Test playback with no audio"""
        pipeline.state = PipelineState.PLAYING
        
        await pipeline._handle_playing()
        
        assert pipeline.state == PipelineState.IDLE
    
    @pytest.mark.asyncio
    async def test_handle_playing_failure(self, pipeline, mock_audio_manager):
        """Test playback failure"""
        pipeline.state = PipelineState.PLAYING
        pipeline._playback_audio = b'test_wav_audio'
        
        mock_audio_manager.play_audio_chunk.return_value = False
        
        await pipeline._handle_playing()
        
        assert pipeline.state == PipelineState.IDLE
    
    def test_get_statistics(self, pipeline, mock_audio_manager, mock_stt_service, mock_tts_service):
        """Test getting pipeline statistics"""
        # Mock statistics returns
        mock_stt_service.get_statistics.return_value = {'stt_stat': 'value'}
        mock_tts_service.get_statistics.return_value = {'tts_stat': 'value'}
        
        stats = pipeline.get_statistics()
        
        assert 'pipeline' in stats
        assert 'audio' in stats
        assert 'stt' in stats
        assert 'tts' in stats
        
        assert stats['pipeline']['state'] == 'idle'
        assert stats['pipeline']['is_running'] is False
        assert stats['stt'] == {'stt_stat': 'value'}
        assert stats['tts'] == {'tts_stat': 'value'}
    
    def test_reset_statistics(self, pipeline, mock_stt_service, mock_tts_service):
        """Test resetting pipeline statistics"""
        pipeline.reset_statistics()
        
        assert pipeline.statistics.total_sessions == 0
        mock_stt_service.reset_statistics.assert_called_once()
        mock_tts_service.reset_statistics.assert_called_once()
    
    def test_cleanup_audio(self, pipeline, mock_audio_manager):
        """Test audio cleanup"""
        mock_audio_manager.is_recording = True
        
        pipeline._cleanup_audio()
        
        mock_audio_manager.stop_recording.assert_called_once()
        mock_audio_manager.cleanup.assert_called_once()
    
    def test_cleanup_audio_error(self, pipeline, mock_audio_manager):
        """Test audio cleanup with error"""
        mock_audio_manager.cleanup.side_effect = Exception("Cleanup error")
        
        # Should not raise exception
        pipeline._cleanup_audio()
        
        mock_audio_manager.cleanup.assert_called_once()


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_load_config_from_yaml_success(self, tmp_path):
        """Test loading configuration from YAML file"""
        config_data = {
            'audio': {'sample_rate': 16000},
            'stt': {'primary': {'model': 'whisper-1'}},
            'tts': {'primary': {'voice': 'alloy'}}
        }
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            import yaml
            yaml.dump(config_data, f)
        
        config = load_config_from_yaml(str(config_file))
        
        assert config == config_data
        assert config['audio']['sample_rate'] == 16000
        assert config['stt']['primary']['model'] == 'whisper-1'
        assert config['tts']['primary']['voice'] == 'alloy'
    
    def test_load_config_from_yaml_file_not_found(self):
        """Test loading configuration from non-existent file"""
        config = load_config_from_yaml("nonexistent.yaml")
        
        assert config == {}
    
    def test_load_config_from_yaml_invalid_yaml(self, tmp_path):
        """Test loading configuration from invalid YAML file"""
        config_file = tmp_path / "invalid.yaml"
        with open(config_file, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        config = load_config_from_yaml(str(config_file))
        
        assert config == {}
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'})
    @patch('assistant.pipeline.create_audio_manager')
    @patch('assistant.pipeline.create_whisper_stt_service')
    @patch('assistant.pipeline.create_openai_tts_service')
    def test_create_pipeline_from_config_success(self, mock_tts_factory, mock_stt_factory, mock_audio_factory, tmp_path):
        """Test creating pipeline from configuration file"""
        # Create test config file
        config_data = {
            'audio': {'sample_rate': 16000, 'chunk_size': 1024},
            'stt': {'primary': {'model': 'whisper-1', 'language': 'en'}},
            'tts': {'primary': {'model': 'tts-1', 'voice': 'alloy', 'speed': 1.0}}
        }
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            import yaml
            yaml.dump(config_data, f)
        
        # Mock factory returns
        mock_audio_manager = Mock()
        mock_stt_service = Mock()
        mock_tts_service = Mock()
        
        mock_audio_factory.return_value = mock_audio_manager
        mock_stt_factory.return_value = mock_stt_service
        mock_tts_factory.return_value = mock_tts_service
        
        pipeline = create_pipeline_from_config(str(config_file))
        
        assert pipeline is not None
        assert isinstance(pipeline, VoiceAssistantPipeline)
        assert pipeline.audio_manager == mock_audio_manager
        assert pipeline.stt_service == mock_stt_service
        assert pipeline.tts_service == mock_tts_service
        
        # Verify factory calls
        mock_audio_factory.assert_called_once()
        mock_stt_factory.assert_called_once()
        mock_tts_factory.assert_called_once()
    
    def test_create_pipeline_from_config_no_config(self):
        """Test creating pipeline from non-existent config file"""
        pipeline = create_pipeline_from_config("nonexistent.yaml")
        
        assert pipeline is None
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'})
    @patch('assistant.pipeline.create_audio_manager')
    def test_create_pipeline_from_config_factory_error(self, mock_audio_factory):
        """Test creating pipeline with factory error"""
        mock_audio_factory.side_effect = Exception("Factory error")
        
        pipeline = create_pipeline_from_config()
        
        assert pipeline is None


if __name__ == "__main__":
    pytest.main([__file__]) 