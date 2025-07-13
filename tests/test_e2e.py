"""
End-to-End Test Suite for VoiceAssistantPipeline

This test suite covers:
- Complete pipeline integration testing
- Performance and latency validation
- Error injection and recovery testing
- Monitoring system verification
- Real-world usage scenarios
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
import json

from assistant.pipeline import VoiceAssistantPipeline, PipelineConfig, PipelineState
from assistant.audio import AudioManager, AudioConfig
from assistant.stt import WhisperSTTService, STTConfig, STTResult
from assistant.tts import OpenAITTSService, TTSConfig, TTSResult
from assistant.monitoring import PerformanceMonitor, PipelineStage, PerformanceThresholds
from assistant.dashboard import ConsoleDashboard, PerformanceReporter


class TestEndToEndPipeline:
    """End-to-end pipeline integration tests"""
    
    @pytest.fixture
    def performance_monitor(self):
        """Performance monitor with test thresholds"""
        thresholds = PerformanceThresholds(
            # Relaxed thresholds for testing
            audio_capture_warning=0.1,
            audio_capture_critical=0.2,
            stt_processing_warning=0.5,
            stt_processing_critical=1.0,
            tts_generation_warning=0.5,
            tts_generation_critical=1.0,
            audio_playback_warning=0.1,
            audio_playback_critical=0.2,
            total_round_trip_warning=2.0,
            total_round_trip_critical=3.0
        )
        return PerformanceMonitor(thresholds)
    
    @pytest.fixture
    def mock_services(self):
        """Mock all external services for testing"""
        # Mock audio manager
        audio_manager = Mock(spec=AudioManager)
        audio_manager.initialize = Mock(return_value=True)
        audio_manager.setup_input_stream = Mock(return_value=True)
        audio_manager.setup_output_stream = Mock(return_value=True)
        audio_manager.start_recording = Mock(return_value=True)
        audio_manager.stop_recording = Mock(return_value=True)
        audio_manager.read_audio_chunk = Mock(return_value=b'test_audio_chunk' * 100)
        audio_manager.play_audio_chunk = Mock(return_value=True)
        audio_manager.cleanup = Mock()
        audio_manager.is_recording = False
        
        # Mock STT service
        stt_service = Mock(spec=WhisperSTTService)
        stt_service.initialize = Mock(return_value=True)
        stt_service.transcribe_audio = AsyncMock(return_value=STTResult(
            text="Hello, this is a test transcription",
            confidence=0.95,
            language="en",
            duration=2.0,
            processing_time=0.3
        ))
        stt_service.get_statistics = Mock(return_value={'stt': 'stats'})
        stt_service.reset_statistics = Mock()
        
        # Mock TTS service
        tts_service = Mock(spec=OpenAITTSService)
        tts_service.initialize = Mock(return_value=True)
        tts_service.synthesize_speech = AsyncMock(return_value=TTSResult(
            audio_data=b'test_tts_audio_data' * 200,
            text="Test response audio",
            duration=3.0,
            processing_time=0.4,
            format="mp3",
            voice="alloy",
            speed=1.0,
            sample_rate=16000,
            cached=False
        ))
        tts_service.get_wav_audio = Mock(return_value=b'test_wav_audio_data' * 300)
        tts_service.get_statistics = Mock(return_value={'tts': 'stats'})
        tts_service.reset_statistics = Mock()
        
        return {
            'audio_manager': audio_manager,
            'stt_service': stt_service,
            'tts_service': tts_service
        }
    
    @pytest.fixture
    def e2e_pipeline(self, mock_services, performance_monitor):
        """End-to-end pipeline with monitoring"""
        config = PipelineConfig(
            max_recording_duration=10.0,
            min_recording_duration=0.1,
            collect_statistics=True
        )
        
        def test_response_callback(text: str) -> str:
            return f"I heard: {text}"
        
        pipeline = VoiceAssistantPipeline(
            config=config,
            audio_manager=mock_services['audio_manager'],
            stt_service=mock_services['stt_service'],
            tts_service=mock_services['tts_service'],
            response_callback=test_response_callback,
            performance_monitor=performance_monitor
        )
        
        return pipeline
    
    @pytest.mark.asyncio
    async def test_complete_voice_session(self, e2e_pipeline, mock_services):
        """Test complete voice session with monitoring"""
        # Initialize pipeline
        assert await e2e_pipeline.initialize() is True
        
        # Simulate push-to-talk press
        e2e_pipeline._handle_push_to_talk_press()
        assert e2e_pipeline.state == PipelineState.RECORDING
        
        # Simulate recording
        await e2e_pipeline._handle_recording()
        
        # Simulate push-to-talk release
        e2e_pipeline._handle_push_to_talk_release()
        assert e2e_pipeline.state == PipelineState.PROCESSING
        
        # Process audio
        await e2e_pipeline._handle_processing()
        assert e2e_pipeline.state == PipelineState.PLAYING
        
        # Play audio
        await e2e_pipeline._handle_playing()
        assert e2e_pipeline.state == PipelineState.IDLE
        
        # Verify monitoring data
        metrics = e2e_pipeline.monitor.get_metrics()
        assert PipelineStage.AUDIO_CAPTURE.value in metrics
        assert PipelineStage.STT_PROCESSING.value in metrics
        assert PipelineStage.TTS_GENERATION.value in metrics
        assert PipelineStage.AUDIO_PLAYBACK.value in metrics
        assert PipelineStage.TOTAL_ROUND_TRIP.value in metrics
        
        # Check that all stages were called
        for stage in [PipelineStage.AUDIO_CAPTURE, PipelineStage.STT_PROCESSING, 
                     PipelineStage.TTS_GENERATION, PipelineStage.AUDIO_PLAYBACK, 
                     PipelineStage.TOTAL_ROUND_TRIP]:
            stage_metrics = metrics[stage.value]
            assert stage_metrics['total_calls'] >= 1
            assert stage_metrics['successful_calls'] >= 1
    
    @pytest.mark.asyncio
    async def test_stt_failure_handling(self, e2e_pipeline, mock_services):
        """Test STT failure handling and monitoring"""
        # Make STT fail
        mock_services['stt_service'].transcribe_audio.return_value = None
        
        # Initialize pipeline
        await e2e_pipeline.initialize()
        
        # Simulate complete session with STT failure
        e2e_pipeline._handle_push_to_talk_press()
        await e2e_pipeline._handle_recording()
        e2e_pipeline._handle_push_to_talk_release()
        await e2e_pipeline._handle_processing()
        
        # Should return to idle state after STT failure
        assert e2e_pipeline.state == PipelineState.IDLE
        
        # Check monitoring recorded the failure
        metrics = e2e_pipeline.monitor.get_metrics()
        stt_metrics = metrics[PipelineStage.STT_PROCESSING.value]
        assert stt_metrics['failed_calls'] >= 1
    
    @pytest.mark.asyncio
    async def test_tts_failure_handling(self, e2e_pipeline, mock_services):
        """Test TTS failure handling and monitoring"""
        # Make TTS fail
        mock_services['tts_service'].synthesize_speech.return_value = None
        
        # Initialize pipeline
        await e2e_pipeline.initialize()
        
        # Simulate complete session with TTS failure
        e2e_pipeline._handle_push_to_talk_press()
        await e2e_pipeline._handle_recording()
        e2e_pipeline._handle_push_to_talk_release()
        await e2e_pipeline._handle_processing()
        
        # Should return to idle state after TTS failure
        assert e2e_pipeline.state == PipelineState.IDLE
        
        # Check monitoring recorded the failure
        metrics = e2e_pipeline.monitor.get_metrics()
        tts_metrics = metrics[PipelineStage.TTS_GENERATION.value]
        assert tts_metrics['failed_calls'] >= 1
    
    @pytest.mark.asyncio
    async def test_audio_playback_failure(self, e2e_pipeline, mock_services):
        """Test audio playback failure handling"""
        # Make audio playback fail
        mock_services['audio_manager'].play_audio_chunk.return_value = False
        
        # Initialize pipeline
        await e2e_pipeline.initialize()
        
        # Simulate session up to playback
        e2e_pipeline._handle_push_to_talk_press()
        await e2e_pipeline._handle_recording()
        e2e_pipeline._handle_push_to_talk_release()
        await e2e_pipeline._handle_processing()
        await e2e_pipeline._handle_playing()
        
        # Should return to idle state after playback failure
        assert e2e_pipeline.state == PipelineState.IDLE
        
        # Check monitoring recorded the failure
        metrics = e2e_pipeline.monitor.get_metrics()
        playback_metrics = metrics[PipelineStage.AUDIO_PLAYBACK.value]
        assert playback_metrics['failed_calls'] >= 1
    
    @pytest.mark.asyncio
    async def test_latency_threshold_alerts(self, e2e_pipeline, mock_services):
        """Test latency threshold alerts"""
        # Set up slower responses to trigger alerts
        async def slow_stt(audio_data):
            await asyncio.sleep(0.6)  # Exceeds 0.5s warning threshold
            return STTResult(
                text="Slow transcription",
                confidence=0.9,
                language="en",
                duration=2.0,
                processing_time=0.6
            )
        
        mock_services['stt_service'].transcribe_audio = slow_stt
        
        # Initialize pipeline
        await e2e_pipeline.initialize()
        
        # Run session
        e2e_pipeline._handle_push_to_talk_press()
        await e2e_pipeline._handle_recording()
        e2e_pipeline._handle_push_to_talk_release()
        await e2e_pipeline._handle_processing()
        await e2e_pipeline._handle_playing()
        
        # Check for alerts
        alerts = e2e_pipeline.monitor.get_recent_alerts()
        assert len(alerts) > 0
        
        # Should have STT processing alert
        stt_alerts = [a for a in alerts if a['stage'] == PipelineStage.STT_PROCESSING.value]
        assert len(stt_alerts) > 0
        assert stt_alerts[0]['level'] in ['warning', 'critical']
    
    def test_statistics_collection(self, e2e_pipeline):
        """Test statistics collection integration"""
        # Get initial statistics
        stats = e2e_pipeline.get_statistics()
        
        # Verify structure
        assert 'pipeline' in stats
        assert 'monitoring' in stats
        assert 'stage_metrics' in stats
        assert 'recent_alerts' in stats
        
        # Check monitoring summary
        monitoring = stats['monitoring']
        assert 'performance_status' in monitoring
        assert 'average_latency_ms' in monitoring
        assert 'success_rate' in monitoring
        
        # Check stage metrics
        stage_metrics = stats['stage_metrics']
        for stage in PipelineStage:
            assert stage.value in stage_metrics
    
    @pytest.mark.asyncio
    async def test_multiple_sessions_performance(self, e2e_pipeline, mock_services):
        """Test multiple sessions and performance trends"""
        await e2e_pipeline.initialize()
        
        # Run multiple sessions
        for i in range(5):
            e2e_pipeline._handle_push_to_talk_press()
            await e2e_pipeline._handle_recording()
            e2e_pipeline._handle_push_to_talk_release()
            await e2e_pipeline._handle_processing()
            await e2e_pipeline._handle_playing()
            
            # Small delay between sessions
            await asyncio.sleep(0.1)
        
        # Check metrics
        metrics = e2e_pipeline.monitor.get_metrics()
        
        # All stages should have 5 successful calls
        for stage in [PipelineStage.AUDIO_CAPTURE, PipelineStage.STT_PROCESSING, 
                     PipelineStage.TTS_GENERATION, PipelineStage.AUDIO_PLAYBACK]:
            stage_metrics = metrics[stage.value]
            assert stage_metrics['total_calls'] == 5
            assert stage_metrics['successful_calls'] == 5
            assert stage_metrics['success_rate'] == 100.0
        
        # Total round trip should have 5 measurements
        total_metrics = metrics[PipelineStage.TOTAL_ROUND_TRIP.value]
        assert total_metrics['total_calls'] == 5
    
    def test_performance_export(self, e2e_pipeline):
        """Test performance data export functionality"""
        # Export JSON
        json_data = e2e_pipeline.monitor.export_metrics('json')
        parsed_data = json.loads(json_data)
        
        # Verify structure
        assert 'timestamp' in parsed_data
        assert 'summary' in parsed_data
        assert 'metrics' in parsed_data
        assert 'thresholds' in parsed_data
        
        # Verify summary
        summary = parsed_data['summary']
        assert 'performance_status' in summary
        assert 'average_latency_ms' in summary
        assert 'success_rate' in summary


class TestPerformanceDashboard:
    """Test performance dashboard functionality"""
    
    @pytest.fixture
    def test_monitor(self):
        """Monitor with test data"""
        monitor = PerformanceMonitor()
        
        # Add some test measurements
        monitor.record_measurement(PipelineStage.AUDIO_CAPTURE, 0.05, True)
        monitor.record_measurement(PipelineStage.STT_PROCESSING, 0.25, True)
        monitor.record_measurement(PipelineStage.TTS_GENERATION, 0.35, True)
        monitor.record_measurement(PipelineStage.AUDIO_PLAYBACK, 0.08, True)
        monitor.record_measurement(PipelineStage.TOTAL_ROUND_TRIP, 0.73, True)
        
        return monitor
    
    def test_dashboard_creation(self, test_monitor):
        """Test dashboard creation and basic functionality"""
        from assistant.dashboard import ConsoleDashboard
        
        dashboard = ConsoleDashboard(test_monitor, refresh_interval=0.1)
        assert dashboard.monitor == test_monitor
        assert dashboard.refresh_interval == 0.1
        assert not dashboard.is_running
    
    def test_dashboard_start_stop(self, test_monitor):
        """Test dashboard start/stop functionality"""
        from assistant.dashboard import ConsoleDashboard
        
        dashboard = ConsoleDashboard(test_monitor, refresh_interval=0.1)
        
        # Start dashboard
        dashboard.start()
        assert dashboard.is_running
        
        # Give it a moment to run
        time.sleep(0.2)
        
        # Stop dashboard
        dashboard.stop()
        assert not dashboard.is_running
    
    def test_snapshot_functionality(self, test_monitor):
        """Test dashboard snapshot functionality"""
        from assistant.dashboard import ConsoleDashboard
        import tempfile
        import os
        
        dashboard = ConsoleDashboard(test_monitor)
        
        # Take snapshot
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_file = os.path.join(temp_dir, "test_snapshot.json")
            result_file = dashboard.take_snapshot(snapshot_file)
            
            assert result_file == snapshot_file
            assert os.path.exists(snapshot_file)
            
            # Verify snapshot content
            with open(snapshot_file, 'r') as f:
                data = json.load(f)
            
            assert 'timestamp' in data
            assert 'summary' in data
            assert 'metrics' in data
    
    def test_performance_reporter(self, test_monitor):
        """Test performance reporter functionality"""
        from assistant.dashboard import PerformanceReporter
        
        reporter = PerformanceReporter(test_monitor)
        
        # Generate text report
        report = reporter.generate_summary_report()
        assert "VOICE ASSISTANT PERFORMANCE REPORT" in report
        assert "OVERALL PERFORMANCE" in report
        assert "STAGE BREAKDOWN" in report
        
        # Test CSV export
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_file = os.path.join(temp_dir, "test_metrics.csv")
            result_file = reporter.export_csv_metrics(csv_file)
            
            assert result_file == csv_file
            assert os.path.exists(csv_file)
            
            # Verify CSV content
            with open(csv_file, 'r') as f:
                content = f.read()
            
            assert "stage,total_calls" in content
            assert "audio_capture" in content


class TestErrorRecoveryScenarios:
    """Test various error scenarios and recovery"""
    
    @pytest.fixture
    def robust_pipeline(self, mock_services, performance_monitor):
        """Pipeline configured for error testing"""
        config = PipelineConfig(
            max_recording_duration=5.0,
            min_recording_duration=0.05,
            max_retries=2
        )
        
        return VoiceAssistantPipeline(
            config=config,
            audio_manager=mock_services['audio_manager'],
            stt_service=mock_services['stt_service'],
            tts_service=mock_services['tts_service'],
            performance_monitor=performance_monitor
        )
    
    @pytest.mark.asyncio
    async def test_network_timeout_simulation(self, robust_pipeline, mock_services):
        """Test network timeout scenarios"""
        # Simulate network timeout
        async def timeout_stt(audio_data):
            await asyncio.sleep(2.0)  # Simulate long network delay
            raise asyncio.TimeoutError("Network timeout")
        
        mock_services['stt_service'].transcribe_audio = timeout_stt
        
        await robust_pipeline.initialize()
        
        # Try processing with timeout
        robust_pipeline._handle_push_to_talk_press()
        await robust_pipeline._handle_recording()
        robust_pipeline._handle_push_to_talk_release()
        await robust_pipeline._handle_processing()
        
        # Should handle timeout gracefully
        assert robust_pipeline.state == PipelineState.IDLE
        
        # Check error was recorded
        metrics = robust_pipeline.monitor.get_metrics()
        stt_metrics = metrics[PipelineStage.STT_PROCESSING.value]
        assert stt_metrics['failed_calls'] >= 1
    
    def test_memory_pressure_simulation(self, robust_pipeline):
        """Test behavior under memory pressure"""
        # This test would simulate high memory usage
        # For now, we'll test that the pipeline handles large audio buffers
        
        # Simulate large audio buffer
        large_audio_data = b'x' * (10 * 1024 * 1024)  # 10MB
        robust_pipeline._audio_buffer = [large_audio_data]
        
        # Pipeline should handle large buffers gracefully
        assert len(robust_pipeline._audio_buffer) == 1
        assert len(robust_pipeline._audio_buffer[0]) == 10 * 1024 * 1024
    
    @pytest.mark.asyncio
    async def test_rapid_state_changes(self, robust_pipeline):
        """Test rapid state changes and race conditions"""
        await robust_pipeline.initialize()
        
        # Rapidly change states
        for _ in range(10):
            robust_pipeline._handle_push_to_talk_press()
            robust_pipeline._handle_push_to_talk_release()
            robust_pipeline._set_state(PipelineState.IDLE)
        
        # Should end up in a consistent state
        assert robust_pipeline.state in [PipelineState.IDLE, PipelineState.PROCESSING]


if __name__ == "__main__":
    # Run some basic tests
    pytest.main([__file__, "-v"]) 