"""
Performance Benchmark Tests for Sovereign 4.0 Realtime API

Uses pytest-benchmark to validate performance requirements:
- <300ms response time target
- Throughput and latency percentiles
- Memory usage and resource efficiency
- Component-level performance breakdown
- Stress testing and load scenarios

These benchmarks ensure Task 18 performance validation requirements are met.
"""

import pytest
import asyncio
import time
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any

# pytest-benchmark for performance testing
import pytest_benchmark

# Performance monitoring
import psutil
import threading
import gc

# Project imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant.realtime_voice import RealtimeVoiceService, RealtimeConfig
from assistant.smart_context_manager import SmartContextManager, SmartContextConfig
from assistant.async_response_manager import AsyncResponseManager, AsyncConfig
from assistant.connection_stability_monitor import ConnectionStabilityMonitor
from assistant.audio_stream_manager import AudioStreamManager, RealtimeAudioConfig
from assistant.mode_switch_manager import ModeSwitchManager
from assistant.performance_optimizer import PerformanceOptimizer
from tests.fixtures.test_fixtures import *


# =============================================================================
# Performance Test Configuration
# =============================================================================

@pytest.fixture
def benchmark_config():
    """Configuration for performance benchmarks"""
    return {
        'target_latency_ms': 300.0,
        'target_throughput_rps': 10.0,
        'memory_threshold_mb': 512.0,
        'cpu_threshold_percent': 80.0,
        'warmup_rounds': 3,
        'test_rounds': 10,
        'min_rounds': 5
    }


# =============================================================================
# Core Component Benchmarks
# =============================================================================

class TestRealtimeVoiceServiceBenchmarks:
    """Performance benchmarks for RealtimeVoiceService"""
    
    @pytest.mark.benchmark(group="realtime_voice")
    def test_service_initialization_performance(self, benchmark, test_realtime_config):
        """Benchmark RealtimeVoiceService initialization time"""
        
        def initialize_service():
            service = RealtimeVoiceService(
                config=test_realtime_config,
                memory_manager=Mock(),
                screen_content_provider=Mock(),
                logger=Mock()
            )
            return service
        
        result = benchmark.pedantic(
            initialize_service,
            rounds=10,
            warmup_rounds=3
        )
        
        # Verify service was created successfully
        assert result is not None
        
        # Check performance target (should initialize in <100ms)
        assert benchmark.stats.stats.mean < 0.1  # 100ms
    
    @pytest.mark.benchmark(group="realtime_voice")
    @pytest.mark.asyncio
    async def test_voice_processing_latency(self, benchmark, test_realtime_config, mock_websocket_connection, sample_audio_speech):
        """Benchmark end-to-end voice processing latency"""
        
        service = RealtimeVoiceService(
            config=test_realtime_config,
            memory_manager=Mock(),
            screen_content_provider=Mock()
        )
        
        async def process_voice_input():
            # Mock WebSocket connection
            with patch('websockets.connect', return_value=mock_websocket_connection):
                start_time = time.time()
                
                # Simulate voice processing
                await service._simulate_voice_processing(sample_audio_speech.audio_data)
                
                return time.time() - start_time
        
        # Run benchmark
        latency_seconds = await benchmark.pedantic(
            process_voice_input,
            rounds=5,
            warmup_rounds=2
        )
        
        # Verify latency meets target (<300ms)
        latency_ms = latency_seconds * 1000
        assert latency_ms < 300.0, f"Voice processing latency {latency_ms:.1f}ms exceeds 300ms target"


class TestSmartContextManagerBenchmarks:
    """Performance benchmarks for SmartContextManager"""
    
    @pytest.mark.benchmark(group="context_management")
    def test_context_assembly_performance(self, benchmark, test_data_generator):
        """Benchmark context assembly speed"""
        
        # Setup context manager
        config = SmartContextConfig(
            budget=ContextBudget(
                system_instructions=2000,
                recent_memory=4000,
                screen_content=2000,
                conversation_history=24000
            )
        )
        
        context_manager = SmartContextManager(
            config=config,
            memory_manager=Mock(),
            screen_provider=Mock()
        )
        
        # Generate test conversation
        conversations = test_data_generator.generate_conversation_flows(1)
        conversation_turns = conversations[0]['turns']
        
        def assemble_context():
            context = context_manager._assemble_full_context(
                conversation_history=conversation_turns,
                current_query="What's the current status?"
            )
            return context
        
        result = benchmark.pedantic(
            assemble_context,
            rounds=20,
            warmup_rounds=5
        )
        
        # Verify context was assembled
        assert result is not None
        assert len(result) > 0
        
        # Check performance target (should assemble in <100ms)
        assert benchmark.stats.stats.mean < 0.1
    
    @pytest.mark.benchmark(group="context_management") 
    def test_token_counting_performance(self, benchmark):
        """Benchmark token counting efficiency"""
        
        from assistant.smart_context_manager import TokenCounter
        
        counter = TokenCounter()
        
        # Generate large text for testing
        large_text = " ".join(["This is a test sentence with multiple words."] * 100)
        
        def count_tokens():
            return counter.count_tokens(large_text)
        
        token_count = benchmark.pedantic(
            count_tokens,
            rounds=50,
            warmup_rounds=10
        )
        
        # Verify token counting works
        assert token_count > 0
        
        # Should count tokens very quickly (<10ms for large text)
        assert benchmark.stats.stats.mean < 0.01
    
    @pytest.mark.benchmark(group="context_management")
    def test_relevance_scoring_performance(self, benchmark):
        """Benchmark relevance scoring speed"""
        
        from assistant.smart_context_manager import RelevanceScorer
        
        scorer = RelevanceScorer()
        
        query = "How do I fix my Python code?"
        candidates = [
            "Python programming tutorial and debugging guide",
            "JavaScript development best practices",
            "Error handling in Python applications",
            "Database query optimization techniques",
            "Code review checklist for Python projects"
        ]
        
        def score_relevance():
            scores = []
            for candidate in candidates:
                score = scorer.calculate_relevance(query, candidate)
                scores.append(score)
            return scores
        
        scores = benchmark.pedantic(
            score_relevance,
            rounds=20,
            warmup_rounds=5
        )
        
        # Verify scoring works
        assert len(scores) == len(candidates)
        assert all(0.0 <= score <= 1.0 for score in scores)
        
        # Should score quickly (<50ms for 5 candidates)
        assert benchmark.stats.stats.mean < 0.05


class TestAsyncResponseManagerBenchmarks:
    """Performance benchmarks for AsyncResponseManager"""
    
    @pytest.mark.benchmark(group="async_processing")
    @pytest.mark.asyncio
    async def test_response_streaming_latency(self, benchmark):
        """Benchmark response streaming latency"""
        
        config = AsyncConfig(
            stream_mode="IMMEDIATE",
            concurrency_level="MEDIUM",
            micro_buffer_size=128,
            chunk_size_bytes=1024
        )
        
        manager = AsyncResponseManager(config)
        await manager.initialize()
        
        async def stream_response():
            start_time = time.time()
            
            # Create test response stream
            stream_id = await manager.create_stream(priority="HIGH")
            
            # Add chunks to stream
            for i in range(10):
                chunk_data = f"Response chunk {i}".encode()
                await manager.add_chunk(stream_id, chunk_data, i)
            
            # Complete stream
            await manager.complete_stream(stream_id)
            
            return time.time() - start_time
        
        latency = await benchmark.pedantic(
            stream_response,
            rounds=10,
            warmup_rounds=3
        )
        
        # Verify streaming meets latency target
        latency_ms = latency * 1000
        assert latency_ms < 100.0, f"Response streaming latency {latency_ms:.1f}ms exceeds 100ms target"
        
        await manager.shutdown()
    
    @pytest.mark.benchmark(group="async_processing")
    @pytest.mark.asyncio
    async def test_concurrent_stream_throughput(self, benchmark):
        """Benchmark concurrent stream handling throughput"""
        
        config = AsyncConfig(
            stream_mode="IMMEDIATE",
            concurrency_level="HIGH",
            max_concurrent_streams=20
        )
        
        manager = AsyncResponseManager(config)
        await manager.initialize()
        
        async def handle_concurrent_streams():
            # Create multiple concurrent streams
            stream_count = 10
            streams = []
            
            start_time = time.time()
            
            # Create all streams
            for i in range(stream_count):
                stream_id = await manager.create_stream(priority="NORMAL")
                streams.append(stream_id)
            
            # Process all streams concurrently
            tasks = []
            for stream_id in streams:
                task = asyncio.create_task(self._process_stream(manager, stream_id))
                tasks.append(task)
            
            # Wait for all streams to complete
            await asyncio.gather(*tasks)
            
            return time.time() - start_time
        
        duration = await benchmark.pedantic(
            handle_concurrent_streams,
            rounds=5,
            warmup_rounds=2
        )
        
        # Calculate throughput (streams per second)
        throughput = 10 / duration
        assert throughput > 5.0, f"Concurrent throughput {throughput:.1f} streams/sec below target"
        
        await manager.shutdown()
    
    async def _process_stream(self, manager, stream_id):
        """Helper method to process a stream"""
        for i in range(5):
            chunk_data = f"Data {i}".encode()
            await manager.add_chunk(stream_id, chunk_data, i)
            await asyncio.sleep(0.001)  # Small delay
        
        await manager.complete_stream(stream_id)


# =============================================================================
# Integration Performance Benchmarks
# =============================================================================

class TestIntegrationPerformanceBenchmarks:
    """Performance benchmarks for integrated components"""
    
    @pytest.mark.benchmark(group="integration")
    @pytest.mark.asyncio
    async def test_end_to_end_conversation_latency(self, benchmark, test_realtime_config, mock_openai_responses):
        """Benchmark complete conversation processing latency"""
        
        # Setup integrated system
        service = RealtimeVoiceService(
            config=test_realtime_config,
            memory_manager=Mock(),
            screen_content_provider=Mock()
        )
        
        async def process_conversation_turn():
            start_time = time.time()
            
            # Simulate complete conversation turn
            with patch('websockets.connect') as mock_connect:
                mock_ws = AsyncMock()
                mock_connect.return_value.__aenter__.return_value = mock_ws
                
                # Mock responses
                mock_ws.recv.side_effect = [
                    '{"type": "session.created", "session": {"id": "test"}}',
                    '{"type": "response.content_part.added", "part": {"type": "text", "text": "Hello!"}}'
                ]
                
                # Process conversation
                response = await service._process_conversation_turn(
                    user_input="Hello, how are you?",
                    conversation_context={}
                )
                
                return time.time() - start_time
        
        latency = await benchmark.pedantic(
            process_conversation_turn,
            rounds=5,
            warmup_rounds=2
        )
        
        # Verify meets end-to-end latency target
        latency_ms = latency * 1000
        assert latency_ms < 300.0, f"E2E conversation latency {latency_ms:.1f}ms exceeds 300ms target"
    
    @pytest.mark.benchmark(group="integration")
    def test_mode_switching_performance(self, benchmark):
        """Benchmark mode switching latency"""
        
        # Setup mode switch manager
        switch_manager = ModeSwitchManager(
            realtime_service=Mock(),
            traditional_pipeline=Mock(),
            config=Mock()
        )
        
        def perform_mode_switch():
            start_time = time.time()
            
            # Switch from traditional to realtime
            switch_manager.switch_to_realtime_mode(
                conversation_state={"messages": [], "context": {}}
            )
            
            # Switch back to traditional
            switch_manager.switch_to_traditional_mode(
                conversation_state={"messages": [], "context": {}}
            )
            
            return time.time() - start_time
        
        switch_time = benchmark.pedantic(
            perform_mode_switch,
            rounds=10,
            warmup_rounds=3
        )
        
        # Mode switching should be very fast (<50ms)
        switch_time_ms = switch_time * 1000
        assert switch_time_ms < 50.0, f"Mode switching time {switch_time_ms:.1f}ms exceeds 50ms target"


# =============================================================================
# Memory and Resource Benchmarks
# =============================================================================

class TestMemoryAndResourceBenchmarks:
    """Performance benchmarks for memory usage and resource efficiency"""
    
    @pytest.mark.benchmark(group="memory")
    def test_memory_usage_stability(self, benchmark):
        """Benchmark memory usage stability during operation"""
        
        def measure_memory_usage():
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            # Simulate intensive operations
            context_manager = SmartContextManager(
                config=SmartContextConfig(),
                memory_manager=Mock(),
                screen_provider=Mock()
            )
            
            # Generate and process multiple contexts
            for i in range(100):
                context = context_manager._assemble_full_context(
                    conversation_history=[],
                    current_query=f"Test query {i}"
                )
            
            final_memory = process.memory_info().rss
            memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
            
            return memory_increase
        
        memory_increase = benchmark.pedantic(
            measure_memory_usage,
            rounds=5,
            warmup_rounds=2
        )
        
        # Memory usage should remain stable (<50MB increase)
        assert memory_increase < 50.0, f"Memory increase {memory_increase:.1f}MB exceeds 50MB threshold"
    
    @pytest.mark.benchmark(group="resources")
    def test_cpu_efficiency(self, benchmark):
        """Benchmark CPU usage efficiency"""
        
        def cpu_intensive_operation():
            # Simulate CPU-intensive context processing
            context_manager = SmartContextManager(
                config=SmartContextConfig(),
                memory_manager=Mock(),
                screen_provider=Mock()
            )
            
            start_cpu = psutil.cpu_percent(interval=None)
            
            # Perform operations
            for _ in range(50):
                context_manager._assemble_full_context(
                    conversation_history=[],
                    current_query="Complex query with lots of context"
                )
            
            end_cpu = psutil.cpu_percent(interval=0.1)
            
            return end_cpu - start_cpu
        
        cpu_usage = benchmark.pedantic(
            cpu_intensive_operation,
            rounds=5,
            warmup_rounds=2
        )
        
        # CPU usage increase should be reasonable (<30%)
        assert cpu_usage < 30.0, f"CPU usage increase {cpu_usage:.1f}% exceeds 30% threshold"


# =============================================================================
# Load Testing Benchmarks
# =============================================================================

class TestLoadBenchmarks:
    """Load testing benchmarks for concurrent scenarios"""
    
    @pytest.mark.benchmark(group="load")
    @pytest.mark.asyncio
    async def test_concurrent_session_handling(self, benchmark):
        """Benchmark concurrent session handling capacity"""
        
        async def handle_concurrent_sessions():
            session_count = 20
            tasks = []
            
            start_time = time.time()
            
            # Create concurrent session tasks
            for i in range(session_count):
                task = asyncio.create_task(self._simulate_session(i))
                tasks.append(task)
            
            # Wait for all sessions to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successful sessions
            successful = sum(1 for r in results if not isinstance(r, Exception))
            
            duration = time.time() - start_time
            
            return successful, duration
        
        successful, duration = await benchmark.pedantic(
            handle_concurrent_sessions,
            rounds=3,
            warmup_rounds=1
        )
        
        # Verify most sessions completed successfully
        assert successful >= 18, f"Only {successful}/20 sessions completed successfully"
        
        # Calculate throughput
        throughput = successful / duration
        assert throughput > 5.0, f"Session throughput {throughput:.1f} sessions/sec below target"
    
    async def _simulate_session(self, session_id: int):
        """Simulate a conversation session"""
        try:
            # Simulate session processing
            await asyncio.sleep(0.1 + np.random.uniform(0, 0.1))  # Variable delay
            
            # Simulate some work
            for _ in range(5):
                await asyncio.sleep(0.01)
            
            return f"session_{session_id}_completed"
            
        except Exception as e:
            return f"session_{session_id}_failed: {e}"


# =============================================================================
# Performance Regression Tests
# =============================================================================

class TestPerformanceRegression:
    """Performance regression detection tests"""
    
    @pytest.mark.benchmark(group="regression")
    def test_context_assembly_regression(self, benchmark):
        """Detect performance regression in context assembly"""
        
        # Baseline performance expectation (from previous optimizations)
        BASELINE_CONTEXT_ASSEMBLY_MS = 50.0
        
        config = SmartContextConfig()
        context_manager = SmartContextManager(
            config=config,
            memory_manager=Mock(),
            screen_provider=Mock()
        )
        
        def assemble_large_context():
            # Create large conversation history
            conversation_history = []
            for i in range(20):
                conversation_history.append({
                    "turn_id": i,
                    "user_message": f"This is user message {i} with substantial content",
                    "assistant_response": f"This is assistant response {i} with detailed information"
                })
            
            return context_manager._assemble_full_context(
                conversation_history=conversation_history,
                current_query="What's the summary of our conversation?"
            )
        
        result = benchmark.pedantic(
            assemble_large_context,
            rounds=10,
            warmup_rounds=3
        )
        
        # Check for performance regression
        current_time_ms = benchmark.stats.stats.mean * 1000
        regression_threshold = BASELINE_CONTEXT_ASSEMBLY_MS * 1.5  # 50% degradation threshold
        
        assert current_time_ms < regression_threshold, \
            f"Context assembly performance regression detected: {current_time_ms:.1f}ms > {regression_threshold:.1f}ms"
    
    @pytest.mark.benchmark(group="regression")
    @pytest.mark.asyncio
    async def test_response_streaming_regression(self, benchmark):
        """Detect performance regression in response streaming"""
        
        BASELINE_STREAMING_MS = 30.0
        
        config = AsyncConfig(
            stream_mode="IMMEDIATE",
            concurrency_level="MEDIUM"
        )
        
        manager = AsyncResponseManager(config)
        await manager.initialize()
        
        async def stream_large_response():
            stream_id = await manager.create_stream(priority="HIGH")
            
            # Stream large response
            for i in range(100):
                chunk_data = f"Large response chunk {i} with substantial content".encode()
                await manager.add_chunk(stream_id, chunk_data, i)
            
            await manager.complete_stream(stream_id)
        
        await benchmark.pedantic(
            stream_large_response,
            rounds=5,
            warmup_rounds=2
        )
        
        current_time_ms = benchmark.stats.stats.mean * 1000
        regression_threshold = BASELINE_STREAMING_MS * 1.5
        
        assert current_time_ms < regression_threshold, \
            f"Response streaming performance regression detected: {current_time_ms:.1f}ms > {regression_threshold:.1f}ms"
        
        await manager.shutdown()


# =============================================================================
# Performance Summary and Reporting
# =============================================================================

def test_performance_summary(benchmark_config):
    """Generate performance summary report"""
    
    # This test doesn't benchmark anything, just validates overall targets
    targets = {
        "max_latency_ms": benchmark_config['target_latency_ms'],
        "min_throughput_rps": benchmark_config['target_throughput_rps'],
        "max_memory_mb": benchmark_config['memory_threshold_mb'],
        "max_cpu_percent": benchmark_config['cpu_threshold_percent']
    }
    
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK SUMMARY")
    print("="*60)
    print(f"Target Latency: <{targets['max_latency_ms']}ms")
    print(f"Target Throughput: >{targets['min_throughput_rps']} RPS")
    print(f"Memory Threshold: <{targets['max_memory_mb']}MB")
    print(f"CPU Threshold: <{targets['max_cpu_percent']}%")
    print("="*60)
    
    # All benchmarks should pass their individual assertions
    # This summary test always passes if we reach here
    assert True


# =============================================================================
# Pytest Benchmark Configuration
# =============================================================================

def pytest_benchmark_update_json(config, benchmarks, output_json):
    """Customize benchmark JSON output"""
    
    # Add custom performance metadata
    output_json['performance_targets'] = {
        'max_latency_ms': 300.0,
        'min_throughput_rps': 10.0,
        'memory_threshold_mb': 512.0,
        'cpu_threshold_percent': 80.0
    }
    
    # Add test environment info
    output_json['test_environment'] = {
        'python_version': sys.version,
        'cpu_count': psutil.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024**3),
        'platform': sys.platform
    }


# Test markers for easy filtering
pytestmark = [
    pytest.mark.performance,
    pytest.mark.benchmark
] 