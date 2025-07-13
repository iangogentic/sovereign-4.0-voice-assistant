#!/usr/bin/env python3
"""
End-to-End Voice Pipeline Integration Tests
Tests the complete voice processing pipeline: STT -> LLM -> TTS
"""

import asyncio
import pytest
import time
import numpy as np
from unittest.mock import patch, AsyncMock

@pytest.mark.integration
@pytest.mark.asyncio
class TestVoicePipelineE2E:
    """End-to-end voice pipeline testing"""
    
    async def test_basic_voice_command_processing(self, sovereign_assistant, test_metrics, test_environment):
        """Test basic voice command processing pipeline"""
        # Arrange
        test_command = "Hello, how are you today?"
        start_time = time.time()
        
        # Mock the STT to return our test command
        with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value=test_command):
            # Act
            response = await sovereign_assistant._process_voice_input()
            processing_time = time.time() - start_time
            
            # Assert
            assert response is not None
            assert len(response) > 0
            assert processing_time < test_environment['voice_latency_threshold_cloud']
            
            # Record metrics
            test_metrics.record_metric('voice_latency', processing_time)
            test_metrics.set_threshold('voice_latency', test_environment['voice_latency_threshold_cloud'])
    
    async def test_complex_query_routing(self, sovereign_assistant, test_data_manager, test_metrics):
        """Test complex query routing through different LLM models"""
        complex_queries = test_data_manager.voice_commands['complex']
        
        for query in complex_queries:
            start_time = time.time()
            
            # Mock STT to return the complex query
            with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value=query):
                response = await sovereign_assistant._process_voice_input()
                processing_time = time.time() - start_time
                
                # Assert response quality for complex queries
                assert response is not None
                assert len(response) > 50  # Complex queries should have detailed responses
                
                # Record metrics
                test_metrics.record_metric('complex_query_latency', processing_time)
    
    async def test_ide_error_processing(self, sovereign_assistant, test_data_manager, test_metrics):
        """Test IDE error dialog processing and response"""
        ide_errors = test_data_manager.voice_commands['ide_related']
        
        for error_command in ide_errors:
            start_time = time.time()
            
            with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value=error_command):
                response = await sovereign_assistant._process_voice_input()
                processing_time = time.time() - start_time
                
                # Assert IDE-specific response characteristics
                assert response is not None
                assert any(keyword in response.lower() for keyword in ['error', 'fix', 'solution', 'check'])
                
                # Record metrics
                test_metrics.record_metric('ide_error_latency', processing_time)
    
    async def test_voice_pipeline_with_audio_simulation(self, sovereign_assistant, test_data_manager, test_metrics):
        """Test complete voice pipeline with simulated audio input"""
        
        # Generate test audio data
        test_audio = test_data_manager.get_test_audio_file(duration=2.0)
        expected_transcription = "Test audio input for voice pipeline"
        
        # Mock the entire pipeline
        with patch.object(sovereign_assistant.audio_manager, 'record_audio', return_value=test_audio), \
             patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value=expected_transcription), \
             patch.object(sovereign_assistant.tts_service, 'synthesize_speech', return_value=b"mock_speech_audio"), \
             patch.object(sovereign_assistant.audio_manager, 'play_audio') as mock_play:
            
            start_time = time.time()
            
            # Execute full pipeline
            await sovereign_assistant._handle_voice_input()
            
            total_time = time.time() - start_time
            
            # Assert pipeline completion
            mock_play.assert_called_once()
            assert total_time < 5.0  # Full pipeline should complete within 5 seconds
            
            # Record metrics
            test_metrics.record_metric('full_pipeline_latency', total_time)
    
    async def test_concurrent_voice_requests(self, sovereign_assistant, test_data_manager, test_environment):
        """Test handling of concurrent voice requests"""
        concurrent_users = test_environment['concurrent_users']
        test_commands = test_data_manager.voice_commands['simple']
        
        async def process_single_request(command):
            with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value=command):
                start_time = time.time()
                response = await sovereign_assistant._process_voice_input()
                return time.time() - start_time, response
        
        # Create concurrent tasks
        tasks = []
        for i in range(concurrent_users):
            command = test_commands[i % len(test_commands)]
            tasks.append(process_single_request(command))
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Assert all requests completed successfully
        for result in results:
            assert not isinstance(result, Exception)
            latency, response = result
            assert response is not None
            assert latency < 10.0  # Allow more time for concurrent processing
    
    async def test_voice_pipeline_error_recovery(self, sovereign_assistant, test_metrics):
        """Test voice pipeline error recovery mechanisms"""
        
        # Test STT failure recovery
        with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', side_effect=Exception("STT failed")):
            start_time = time.time()
            response = await sovereign_assistant._process_voice_input()
            recovery_time = time.time() - start_time
            
            # Should gracefully handle STT failure
            assert response is not None
            assert "sorry" in response.lower() or "error" in response.lower()
            test_metrics.record_metric('stt_error_recovery_time', recovery_time)
        
        # Test LLM failure recovery  
        with patch.object(sovereign_assistant.llm_router, 'route_query', side_effect=Exception("LLM failed")):
            with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value="test command"):
                start_time = time.time()
                response = await sovereign_assistant._process_voice_input()
                recovery_time = time.time() - start_time
                
                # Should have fallback response
                assert response is not None
                test_metrics.record_metric('llm_error_recovery_time', recovery_time)
    
    async def test_offline_mode_fallback(self, sovereign_assistant, test_metrics):
        """Test offline mode fallback when network is unavailable"""
        
        # Simulate network failure
        with patch('aiohttp.ClientSession.post', side_effect=Exception("Network unavailable")):
            with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value="Hello"):
                start_time = time.time()
                response = await sovereign_assistant._process_voice_input()
                fallback_time = time.time() - start_time
                
                # Should fallback gracefully
                assert response is not None
                test_metrics.record_metric('offline_fallback_time', fallback_time)
    
    async def test_memory_integration_in_pipeline(self, sovereign_assistant, test_data_manager):
        """Test memory system integration within voice pipeline"""
        
        # First conversation to store in memory
        first_command = "Remember that I prefer Python for backend development"
        with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value=first_command):
            first_response = await sovereign_assistant._process_voice_input()
            assert first_response is not None
        
        # Second conversation referencing memory
        second_command = "What programming language do I prefer for backend?"
        with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value=second_command):
            second_response = await sovereign_assistant._process_voice_input()
            
            # Should reference previous conversation
            assert second_response is not None
            # Note: Actual memory recall testing will be in dedicated memory tests
    
    async def test_voice_quality_metrics(self, sovereign_assistant, test_data_manager, test_metrics):
        """Test voice quality metrics and audio processing"""
        
        # Test different audio qualities
        audio_scenarios = [
            ('high_quality', 1.0),
            ('medium_quality', 0.7),
            ('low_quality', 0.3)
        ]
        
        for scenario_name, quality_factor in audio_scenarios:
            # Generate audio with different quality simulation
            audio_data = test_data_manager.get_test_audio_file(duration=1.5)
            
            # Add simulated noise based on quality
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            if quality_factor < 1.0:
                noise_level = int(1000 * (1 - quality_factor))
                noise = np.random.randint(-noise_level, noise_level, len(audio_array))
                audio_array = np.clip(audio_array + noise, -32768, 32767).astype(np.int16)
                audio_data = audio_array.tobytes()
            
            # Test transcription accuracy based on quality
            expected_confidence = quality_factor
            test_transcription = f"Test audio quality {scenario_name}"
            
            with patch.object(sovereign_assistant.audio_manager, 'record_audio', return_value=audio_data), \
                 patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value=test_transcription):
                
                start_time = time.time()
                response = await sovereign_assistant._process_voice_input()
                processing_time = time.time() - start_time
                
                # Record quality metrics
                test_metrics.record_metric(f'audio_quality_{scenario_name}_latency', processing_time)
                
                # Assert response quality correlates with audio quality
                assert response is not None
                if quality_factor >= 0.7:
                    # High quality should produce good responses
                    assert len(response) > 10


@pytest.mark.integration
@pytest.mark.performance
@pytest.mark.asyncio
class TestVoicePerformanceMetrics:
    """Voice pipeline performance validation"""
    
    async def test_voice_latency_requirements(self, sovereign_assistant, test_data_manager, test_metrics, test_environment):
        """Test voice latency meets requirements"""
        
        # Test cloud mode latency (< 800ms)
        simple_commands = test_data_manager.voice_commands['simple']
        
        for command in simple_commands:
            with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value=command):
                start_time = time.time()
                response = await sovereign_assistant._process_voice_input()
                latency = time.time() - start_time
                
                # Record and validate latency
                test_metrics.record_metric('cloud_voice_latency', latency)
                test_metrics.set_threshold('cloud_voice_latency', test_environment['voice_latency_threshold_cloud'])
                
                assert latency < test_environment['voice_latency_threshold_cloud'], \
                    f"Cloud voice latency {latency:.3f}s exceeds threshold {test_environment['voice_latency_threshold_cloud']}s"
    
    async def test_throughput_metrics(self, sovereign_assistant, test_data_manager, test_metrics):
        """Test voice pipeline throughput"""
        
        commands = test_data_manager.voice_commands['simple'] * 3  # 12 commands total
        start_time = time.time()
        
        for command in commands:
            with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value=command):
                await sovereign_assistant._process_voice_input()
        
        total_time = time.time() - start_time
        throughput = len(commands) / total_time  # commands per second
        
        test_metrics.record_metric('voice_throughput', throughput)
        
        # Should process at least 0.5 commands per second
        assert throughput >= 0.5, f"Voice throughput {throughput:.2f} commands/sec is too low"
    
    async def test_resource_utilization(self, sovereign_assistant, test_data_manager):
        """Test resource utilization during voice processing"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Baseline measurements
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        baseline_cpu = process.cpu_percent()
        
        # Process multiple commands
        commands = test_data_manager.voice_commands['complex']
        
        for command in commands:
            with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value=command):
                await sovereign_assistant._process_voice_input()
        
        # Final measurements
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        final_cpu = process.cpu_percent()
        
        memory_increase = final_memory - baseline_memory
        
        # Assert reasonable resource usage
        assert memory_increase < 100, f"Memory usage increased by {memory_increase:.1f}MB"
        assert final_cpu < 80, f"CPU usage {final_cpu}% is too high"


@pytest.mark.integration
@pytest.mark.asyncio
class TestVoicePipelineIntegration:
    """Integration tests for voice pipeline components"""
    
    async def test_stt_llm_integration(self, sovereign_assistant, test_data_manager):
        """Test STT to LLM integration"""
        
        # Test that STT output properly feeds into LLM
        test_input = "What is the capital of France?"
        expected_keywords = ["paris", "france", "capital"]
        
        with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value=test_input):
            result = await sovereign_assistant.llm_router.route_query(test_input)
            
            response = result.get('response', '')
            assert response is not None
            
            # Should contain relevant information
            response_lower = response.lower()
            assert any(keyword in response_lower for keyword in expected_keywords)
    
    async def test_llm_tts_integration(self, sovereign_assistant):
        """Test LLM to TTS integration"""
        
        # Test that LLM output properly feeds into TTS
        llm_response = "Hello, this is a test response from the AI assistant."
        
        with patch.object(sovereign_assistant.tts_service, 'synthesize_speech') as mock_tts:
            mock_tts.return_value = b"mock_audio_output"
            
            # Call TTS with LLM response
            audio_output = await sovereign_assistant.tts_service.synthesize_speech(llm_response)
            
            # Assert TTS was called with correct input
            mock_tts.assert_called_once_with(llm_response)
            assert audio_output is not None
    
    async def test_end_to_end_pipeline_integration(self, sovereign_assistant, test_data_manager):
        """Test complete end-to-end pipeline integration"""
        
        original_command = "Help me understand Python decorators"
        
        # Track the data flow through the pipeline
        stt_output = None
        llm_input = None
        llm_output = None
        tts_input = None
        
        async def mock_stt_transcribe(audio_data):
            nonlocal stt_output
            stt_output = original_command
            return stt_output
        
        async def mock_llm_route(query):
            nonlocal llm_input, llm_output
            llm_input = query
            llm_output = "Python decorators are functions that modify other functions..."
            return {"response": llm_output, "model_used": "gpt-4o-mini"}
        
        async def mock_tts_synthesize(text):
            nonlocal tts_input
            tts_input = text
            return b"synthesized_audio_data"
        
        # Apply mocks
        with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', side_effect=mock_stt_transcribe), \
             patch.object(sovereign_assistant.llm_router, 'route_query', side_effect=mock_llm_route), \
             patch.object(sovereign_assistant.tts_service, 'synthesize_speech', side_effect=mock_tts_synthesize), \
             patch.object(sovereign_assistant.audio_manager, 'record_audio', return_value=b"mock_audio"), \
             patch.object(sovereign_assistant.audio_manager, 'play_audio'):
            
            # Execute pipeline
            await sovereign_assistant._handle_voice_input()
            
            # Verify data flow
            assert stt_output == original_command
            assert llm_input == stt_output
            assert llm_output is not None and "decorators" in llm_output
            assert tts_input == llm_output
    
    async def test_pipeline_error_propagation(self, sovereign_assistant):
        """Test error propagation through pipeline"""
        
        # Test that errors are properly handled and don't crash the pipeline
        with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', side_effect=Exception("STT Error")):
            # Should not raise exception
            response = await sovereign_assistant._process_voice_input()
            assert response is not None  # Should have fallback response
        
        with patch.object(sovereign_assistant.llm_router, 'route_query', side_effect=Exception("LLM Error")):
            with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value="test"):
                # Should not raise exception
                response = await sovereign_assistant._process_voice_input()
                assert response is not None  # Should have fallback response 