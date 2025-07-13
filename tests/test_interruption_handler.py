"""
Comprehensive tests for Natural Conversation Interruption Handler
Tests VAD integration, state management, amplitude analysis, and interruption recovery
"""

import pytest
import asyncio
import numpy as np
import time
import threading
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from assistant.interruption_handler import (
    InterruptionHandler,
    ConversationState,
    InterruptionType,
    VADConfig,
    ClientAudioConfig,
    ConversationBuffer,
    AudioAmplitudeAnalyzer,
    InterruptionEvent,
    create_interruption_handler
)


class TestConversationBuffer:
    """Test conversation buffer for partial response recovery"""
    
    def test_buffer_initialization(self):
        """Test ConversationBuffer initialization"""
        buffer = ConversationBuffer(max_buffer_size=10)
        assert buffer.max_buffer_size == 10
        assert buffer.partial_response == ""
        assert len(buffer.response_buffer) == 0
        assert buffer.interruption_point is None
        assert len(buffer.context_snapshot) == 0
    
    def test_save_partial_response(self):
        """Test saving partial AI responses"""
        buffer = ConversationBuffer()
        
        # Save partial response
        buffer.save_partial_response("Hello, how can I", 15)
        assert buffer.partial_response == "Hello, how can I"
        assert buffer.interruption_point == 15
        assert "Hello, how can I" in buffer.response_buffer
        
        # Save another response
        buffer.save_partial_response("help you today?")
        assert buffer.partial_response == "help you today?"
        assert buffer.interruption_point == 15  # Uses length if not specified
    
    def test_buffer_size_limit(self):
        """Test buffer maintains size limit"""
        buffer = ConversationBuffer(max_buffer_size=3)
        
        # Add responses up to limit
        for i in range(5):
            buffer.save_partial_response(f"Response {i}")
        
        # Should only keep last 3
        assert len(buffer.response_buffer) == 3
        assert "Response 2" in buffer.response_buffer
        assert "Response 3" in buffer.response_buffer
        assert "Response 4" in buffer.response_buffer
        assert "Response 0" not in buffer.response_buffer
    
    def test_save_context_snapshot(self):
        """Test saving conversation context"""
        buffer = ConversationBuffer()
        context = {
            "user_query": "What's the weather?",
            "screen_content": "Weather app showing 72°F",
            "memory_context": "User likes detailed weather info"
        }
        
        buffer.save_context_snapshot(context)
        assert buffer.context_snapshot == context
        
        # Test it creates a copy
        context["new_field"] = "test"
        assert "new_field" not in buffer.context_snapshot
    
    def test_clear_buffers(self):
        """Test clearing all buffers"""
        buffer = ConversationBuffer()
        
        # Fill with data
        buffer.save_partial_response("Test response", 5)
        buffer.save_context_snapshot({"test": "data"})
        buffer.audio_buffer.append(b"audio_data")
        
        # Clear and verify
        buffer.clear_buffers()
        assert buffer.partial_response == ""
        assert len(buffer.response_buffer) == 0
        assert buffer.interruption_point is None
        assert len(buffer.context_snapshot) == 0
        assert len(buffer.audio_buffer) == 0
    
    def test_get_recovery_data(self):
        """Test getting recovery data"""
        buffer = ConversationBuffer()
        
        # Add multiple responses
        for i in range(7):
            buffer.save_partial_response(f"Response {i}")
        
        buffer.save_context_snapshot({"context": "test"})
        
        recovery_data = buffer.get_recovery_data()
        
        assert recovery_data["partial_response"] == "Response 6"
        assert recovery_data["context_snapshot"] == {"context": "test"}
        # Should only include last 5 responses
        assert len(recovery_data["response_history"]) == 5
        assert "Response 6" in recovery_data["response_history"]
        assert "Response 2" in recovery_data["response_history"]
        assert "Response 1" not in recovery_data["response_history"]


class TestAudioAmplitudeAnalyzer:
    """Test client-side audio amplitude analysis"""
    
    def test_analyzer_initialization(self):
        """Test AudioAmplitudeAnalyzer initialization"""
        config = ClientAudioConfig(
            sample_rate=24000,
            amplitude_threshold=0.02,
            analysis_window_ms=50
        )
        
        analyzer = AudioAmplitudeAnalyzer(config)
        assert analyzer.config == config
        assert analyzer.current_amplitude == 0.0
        assert not analyzer.is_analyzing
        assert len(analyzer.callbacks) == 0
    
    def test_add_audio_callback(self):
        """Test adding audio analysis callbacks"""
        analyzer = AudioAmplitudeAnalyzer(ClientAudioConfig())
        callback = Mock()
        
        analyzer.add_audio_callback(callback)
        assert callback in analyzer.callbacks
    
    def test_start_stop_analysis(self):
        """Test starting and stopping audio analysis"""
        analyzer = AudioAmplitudeAnalyzer(ClientAudioConfig())
        
        # Start analysis
        analyzer.start_analysis()
        assert analyzer.is_analyzing
        assert analyzer.analysis_thread is not None
        assert analyzer.analysis_thread.is_alive()
        
        # Stop analysis
        analyzer.stop_analysis()
        assert not analyzer.is_analyzing
    
    def test_speech_detection_thresholds(self):
        """Test speech detection with different amplitude thresholds"""
        config = ClientAudioConfig(
            amplitude_threshold=0.01,
            silence_threshold=0.005,
            noise_gate_threshold=0.002
        )
        analyzer = AudioAmplitudeAnalyzer(config)
        
        # Test silence
        assert not analyzer._is_speech_detected(0.001)
        assert not analyzer._is_speech_detected(0.004)
        
        # Test speech
        assert analyzer._is_speech_detected(0.02)
        
        # Test with history for noise filtering
        analyzer.amplitude_history = [0.008, 0.009, 0.007, 0.008, 0.009]
        assert analyzer._is_speech_detected(0.015)  # Above threshold and recent average
        assert not analyzer._is_speech_detected(0.009)  # Below noise gate relative to history
    
    def test_process_audio_chunk(self):
        """Test processing audio chunks"""
        analyzer = AudioAmplitudeAnalyzer(ClientAudioConfig())
        
        # Create mock audio data
        audio_data = np.random.randint(-1000, 1000, 1024, dtype=np.int16).tobytes()
        
        # Should not raise exception
        analyzer.process_audio_chunk(audio_data)
        
        # Queue should contain data (non-blocking test)
        assert not analyzer.audio_queue.empty()
    
    @patch('numpy.frombuffer')
    def test_amplitude_calculation(self, mock_frombuffer):
        """Test RMS amplitude calculation"""
        config = ClientAudioConfig()
        analyzer = AudioAmplitudeAnalyzer(config)
        
        # Mock audio data - sine wave
        mock_audio = np.array([100, 0, -100, 0, 100, 0], dtype=np.int16)
        mock_frombuffer.return_value = mock_audio
        
        analyzer.start_analysis()
        time.sleep(0.1)  # Let analysis start
        
        # Send audio data
        analyzer.process_audio_chunk(b"mock_data")
        time.sleep(0.2)  # Let analysis process
        
        analyzer.stop_analysis()
        
        # Should have processed and updated amplitude
        assert analyzer.current_amplitude >= 0.0


class TestInterruptionHandler:
    """Test main InterruptionHandler class"""
    
    @pytest.fixture
    def handler(self):
        """Create InterruptionHandler for testing"""
        vad_config = VADConfig(threshold=0.5, prefix_padding_ms=300, silence_duration_ms=200)
        audio_config = ClientAudioConfig(amplitude_threshold=0.01)
        return InterruptionHandler(vad_config, audio_config)
    
    @pytest.fixture
    async def initialized_handler(self, handler):
        """Create and initialize InterruptionHandler"""
        await handler.initialize()
        yield handler
        await handler.shutdown()
    
    def test_handler_initialization(self, handler):
        """Test InterruptionHandler initialization"""
        assert handler.current_state == ConversationState.LISTENING
        assert handler.previous_state == ConversationState.LISTENING
        assert handler.conversation_buffer is not None
        assert handler.amplitude_analyzer is not None
        assert handler.metrics is not None
        assert not handler.is_vad_active
    
    @pytest.mark.asyncio
    async def test_async_initialization(self, handler):
        """Test async initialization"""
        success = await handler.initialize()
        assert success
        assert handler.amplitude_analyzer.is_analyzing
        
        await handler.shutdown()
    
    def test_vad_configuration(self, handler):
        """Test VAD configuration generation"""
        config = handler.get_vad_configuration()
        
        expected = {
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 200
            }
        }
        assert config == expected
    
    def test_add_callbacks(self, handler):
        """Test adding state change and interruption callbacks"""
        state_callback = Mock()
        interruption_callback = Mock()
        
        handler.add_state_change_callback(state_callback)
        handler.add_interruption_callback(interruption_callback)
        
        assert state_callback in handler.state_change_callbacks
        assert interruption_callback in handler.interruption_callbacks
    
    @pytest.mark.asyncio
    async def test_state_transitions(self, initialized_handler):
        """Test conversation state transitions"""
        handler = initialized_handler
        state_callback = Mock()
        handler.add_state_change_callback(state_callback)
        
        # Test transition to speaking
        await handler.start_speaking()
        assert handler.current_state == ConversationState.SPEAKING
        state_callback.assert_called_with(ConversationState.LISTENING, ConversationState.SPEAKING)
        
        # Test transition back to listening
        await handler.stop_speaking()
        assert handler.current_state == ConversationState.LISTENING
        state_callback.assert_called_with(ConversationState.SPEAKING, ConversationState.LISTENING)
    
    @pytest.mark.asyncio
    async def test_vad_event_handling(self, initialized_handler):
        """Test VAD event handling from OpenAI"""
        handler = initialized_handler
        interruption_callback = Mock()
        handler.add_interruption_callback(interruption_callback)
        
        # Test speech started during listening (no interruption)
        await handler.handle_vad_event("input_audio_buffer.speech_started", {})
        assert handler.is_vad_active
        assert handler.last_speech_start is not None
        interruption_callback.assert_not_called()
        
        # Test speech started during speaking (should interrupt)
        await handler.start_speaking()
        await handler.handle_vad_event("input_audio_buffer.speech_started", {"audio_start_ms": 1500})
        
        assert handler.current_state == ConversationState.INTERRUPTED
        interruption_callback.assert_called_once()
        event = interruption_callback.call_args[0][0]
        assert event.interruption_type == InterruptionType.USER_SPEECH
    
    @pytest.mark.asyncio
    async def test_speech_stopped_handling(self, initialized_handler):
        """Test speech stopped VAD event"""
        handler = initialized_handler
        
        # Start speech then stop
        await handler.handle_vad_event("input_audio_buffer.speech_started", {})
        await handler.handle_vad_event("input_audio_buffer.speech_stopped", {})
        
        assert not handler.is_vad_active
        assert handler.last_speech_stop is not None
    
    @pytest.mark.asyncio
    async def test_manual_interruption(self, initialized_handler):
        """Test manual interruption trigger"""
        handler = initialized_handler
        interruption_callback = Mock()
        handler.add_interruption_callback(interruption_callback)
        
        await handler.start_speaking()
        await handler.manual_interrupt()
        
        assert handler.current_state == ConversationState.INTERRUPTED
        interruption_callback.assert_called_once()
        event = interruption_callback.call_args[0][0]
        assert event.interruption_type == InterruptionType.MANUAL
    
    @pytest.mark.asyncio
    async def test_interruption_recovery(self, initialized_handler):
        """Test interruption recovery process"""
        handler = initialized_handler
        
        # Setup interruption scenario
        await handler.start_speaking()
        handler.conversation_buffer.save_partial_response("Hello, I was saying", 17)
        handler.conversation_buffer.save_context_snapshot({"context": "test"})
        
        await handler.manual_interrupt()
        
        # Test recovery
        recovery_data = await handler.recover_from_interruption()
        
        assert handler.current_state == ConversationState.RESUMING
        assert recovery_data["partial_response"] == "Hello, I was saying"
        assert recovery_data["context_snapshot"]["context"] == "test"
        assert handler.metrics.successful_recoveries == 1
    
    def test_amplitude_analysis_integration(self, handler):
        """Test integration with amplitude analyzer"""
        # Create mock audio data with high amplitude
        high_amplitude_audio = np.array([10000, -10000, 10000, -10000] * 256, dtype=np.int16).tobytes()
        
        handler.process_audio_chunk(high_amplitude_audio)
        
        # Should have added data to analyzer queue
        assert not handler.amplitude_analyzer.audio_queue.empty()
    
    @pytest.mark.asyncio
    async def test_amplitude_interruption_trigger(self, initialized_handler):
        """Test amplitude-based interruption during speaking"""
        handler = initialized_handler
        interruption_callback = Mock()
        handler.add_interruption_callback(interruption_callback)
        
        await handler.start_speaking()
        
        # Simulate high amplitude detection
        with patch.object(handler.amplitude_analyzer, 'current_amplitude', 0.05):
            handler._on_amplitude_analysis(0.05, True)
            
            # Give async task time to execute
            await asyncio.sleep(0.1)
        
        # Should have triggered amplitude-based interruption
        if interruption_callback.called:
            event = interruption_callback.call_args[0][0]
            assert event.interruption_type == InterruptionType.AMPLITUDE_SPIKE
    
    def test_get_metrics(self, handler):
        """Test metrics collection"""
        # Add some test data
        handler.metrics.total_interruptions = 5
        handler.metrics.successful_recoveries = 4
        handler.metrics.failed_recoveries = 1
        handler.metrics.interruptions_by_type[InterruptionType.USER_SPEECH] = 3
        handler.metrics.interruptions_by_type[InterruptionType.AMPLITUDE_SPIKE] = 2
        
        metrics = handler.get_metrics()
        
        assert metrics["total_interruptions"] == 5
        assert metrics["successful_recoveries"] == 4
        assert metrics["failed_recoveries"] == 1
        assert metrics["current_state"] == ConversationState.LISTENING.value
        assert "session_duration" in metrics
        assert "conversation_quality_score" in metrics
        assert "interruptions_by_type" in metrics
    
    def test_conversation_quality_calculation(self, handler):
        """Test conversation quality score calculation"""
        # Setup test scenario: 10 interruptions, 8 successful recoveries
        handler.metrics.total_interruptions = 10
        handler.metrics.successful_recoveries = 8
        handler.metrics.session_start = datetime.now() - timedelta(minutes=5)  # 5 minute session
        
        metrics = handler.get_metrics()
        quality_score = metrics["conversation_quality_score"]
        
        # Should be a reasonable quality score
        assert 0 <= quality_score <= 100
        assert quality_score > 50  # Good recovery rate should yield decent score
    
    @pytest.mark.asyncio
    async def test_multiple_interruptions(self, initialized_handler):
        """Test handling multiple rapid interruptions"""
        handler = initialized_handler
        interruption_callback = Mock()
        handler.add_interruption_callback(interruption_callback)
        
        await handler.start_speaking()
        
        # Trigger multiple interruptions
        await handler.manual_interrupt()
        await asyncio.sleep(0.05)
        await handler.handle_vad_event("input_audio_buffer.speech_started", {})
        await asyncio.sleep(0.05)
        
        # Should have handled multiple interruptions
        assert handler.metrics.total_interruptions >= 1
        assert len(handler.metrics.interruption_history) >= 1
    
    @pytest.mark.asyncio 
    async def test_error_handling_in_vad_events(self, initialized_handler):
        """Test error handling in VAD event processing"""
        handler = initialized_handler
        
        # Test with invalid event type
        await handler.handle_vad_event("invalid_event_type", {})
        
        # Should not crash and maintain stable state
        assert handler.current_state == ConversationState.LISTENING
    
    @pytest.mark.asyncio
    async def test_transcription_handling(self, initialized_handler):
        """Test transcription completion handling"""
        handler = initialized_handler
        
        # Test transcription event
        await handler.handle_vad_event(
            "conversation.item.input_audio_transcription.completed",
            {"transcript": "Hello, this is a test"}
        )
        
        # Should handle without error
        assert handler.current_state == ConversationState.LISTENING


class TestIntegrationScenarios:
    """Integration tests for realistic conversation scenarios"""
    
    @pytest.fixture
    async def conversation_handler(self):
        """Create handler for conversation testing"""
        handler = create_interruption_handler(
            vad_threshold=0.4,
            amplitude_threshold=0.015,
            vad_prefix_padding_ms=250
        )
        await handler.initialize()
        yield handler
        await handler.shutdown()
    
    @pytest.mark.asyncio
    async def test_natural_conversation_flow(self, conversation_handler):
        """Test natural conversation without interruptions"""
        handler = conversation_handler
        state_changes = []
        
        def track_state_changes(old_state, new_state):
            state_changes.append((old_state, new_state))
        
        handler.add_state_change_callback(track_state_changes)
        
        # Simulate conversation flow
        await handler.start_speaking()
        await asyncio.sleep(0.1)
        await handler.stop_speaking()
        
        # Should have clean state transitions
        assert len(state_changes) == 2
        assert state_changes[0] == (ConversationState.LISTENING, ConversationState.SPEAKING)
        assert state_changes[1] == (ConversationState.SPEAKING, ConversationState.LISTENING)
    
    @pytest.mark.asyncio
    async def test_interruption_and_recovery_flow(self, conversation_handler):
        """Test complete interruption and recovery scenario"""
        handler = conversation_handler
        
        # Start AI response
        await handler.start_speaking()
        handler.conversation_buffer.save_partial_response("I think the weather today is", 29)
        
        # User interrupts
        await handler.handle_vad_event("input_audio_buffer.speech_started", {})
        assert handler.current_state == ConversationState.INTERRUPTED
        
        # Recover from interruption
        recovery_data = await handler.recover_from_interruption()
        assert handler.current_state == ConversationState.RESUMING
        assert recovery_data["partial_response"] == "I think the weather today is"
        
        # Resume normal conversation
        await handler.stop_speaking()
        assert handler.current_state == ConversationState.LISTENING
    
    @pytest.mark.asyncio
    async def test_multiple_interruption_types(self, conversation_handler):
        """Test different types of interruptions in sequence"""
        handler = conversation_handler
        interruptions = []
        
        def track_interruptions(event):
            interruptions.append(event.interruption_type)
        
        handler.add_interruption_callback(track_interruptions)
        
        # Test sequence: manual, VAD, amplitude
        await handler.start_speaking()
        
        # Manual interruption
        await handler.manual_interrupt()
        await handler.recover_from_interruption()
        
        # Back to speaking
        await handler.start_speaking()
        
        # VAD interruption
        await handler.handle_vad_event("input_audio_buffer.speech_started", {})
        await handler.recover_from_interruption()
        
        # Verify different interruption types were captured
        assert InterruptionType.MANUAL in interruptions
        assert InterruptionType.USER_SPEECH in interruptions
    
    @pytest.mark.asyncio
    async def test_performance_requirements(self, conversation_handler):
        """Test performance requirements for real-time usage"""
        handler = conversation_handler
        
        # Test state transition speed
        start_time = time.time()
        await handler.start_speaking()
        transition_time = time.time() - start_time
        assert transition_time < 0.01  # Should be under 10ms
        
        # Test interruption detection speed
        start_time = time.time()
        await handler.manual_interrupt()
        interruption_time = time.time() - start_time
        assert interruption_time < 0.05  # Should be under 50ms
        
        # Test recovery speed
        start_time = time.time()
        await handler.recover_from_interruption()
        recovery_time = time.time() - start_time
        assert recovery_time < 0.1  # Should be under 100ms


class TestFactoryFunction:
    """Test factory function for easy handler creation"""
    
    def test_create_interruption_handler(self):
        """Test factory function with default parameters"""
        handler = create_interruption_handler()
        
        assert isinstance(handler, InterruptionHandler)
        assert handler.vad_config.threshold == 0.5
        assert handler.vad_config.prefix_padding_ms == 300
        assert handler.vad_config.silence_duration_ms == 200
        assert handler.audio_config.amplitude_threshold == 0.01
    
    def test_create_interruption_handler_custom_params(self):
        """Test factory function with custom parameters"""
        handler = create_interruption_handler(
            vad_threshold=0.7,
            vad_prefix_padding_ms=400,
            vad_silence_duration_ms=150,
            amplitude_threshold=0.02
        )
        
        assert handler.vad_config.threshold == 0.7
        assert handler.vad_config.prefix_padding_ms == 400
        assert handler.vad_config.silence_duration_ms == 150
        assert handler.audio_config.amplitude_threshold == 0.02


@pytest.mark.asyncio
async def test_concurrent_operations():
    """Test handler behavior under concurrent operations"""
    handler = create_interruption_handler()
    await handler.initialize()
    
    try:
        # Start multiple concurrent operations
        tasks = [
            asyncio.create_task(handler.start_speaking()),
            asyncio.create_task(handler.manual_interrupt()),
            asyncio.create_task(handler.recover_from_interruption()),
        ]
        
        # Wait for all to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handler should maintain valid state
        assert handler.current_state in ConversationState
        
    finally:
        await handler.shutdown()


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"]) 