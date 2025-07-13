"""
Tests for AsyncResponseManager - Async Processing and Response Streaming
Comprehensive coverage of all async components and streaming functionality
"""

import pytest
import asyncio
import time
import logging
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Import classes to test
from assistant.async_response_manager import (
    AsyncResponseManager,
    AsyncResponseStream,
    ConcurrentAudioProcessor,
    ResponseChunk,
    StreamingMetrics,
    AsyncConfig,
    StreamMode,
    ConcurrencyLevel,
    ResponsePriority,
    create_async_response_manager
)


@pytest.fixture
def async_config():
    """Create test configuration"""
    return AsyncConfig(
        stream_mode=StreamMode.IMMEDIATE,
        concurrency_level=ConcurrencyLevel.LOW,  # Use fewer workers for testing
        max_concurrent_streams=5,
        worker_pool_size=2,
        micro_buffer_size=64,
        chunk_size_bytes=512,
        max_queue_size=10,
        adaptive_concurrency=False  # Disable for predictable testing
    )


@pytest.fixture
def logger():
    """Create test logger"""
    logger = logging.getLogger("test_async_response_manager")
    logger.setLevel(logging.DEBUG)
    return logger


class TestAsyncConfig:
    """Test AsyncConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = AsyncConfig()
        
        assert config.stream_mode == StreamMode.IMMEDIATE
        assert config.concurrency_level == ConcurrencyLevel.MEDIUM
        assert config.first_chunk_target_ms == 50.0
        assert config.chunk_size_bytes == 1024
        assert config.micro_buffer_size == 128
        assert config.max_concurrent_streams == 10
        assert config.enable_progressive_audio is True
        assert config.adaptive_concurrency is True
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = AsyncConfig(
            stream_mode=StreamMode.CHUNKED,
            concurrency_level=ConcurrencyLevel.HIGH,
            chunk_size_bytes=2048,
            max_concurrent_streams=20
        )
        
        assert config.stream_mode == StreamMode.CHUNKED
        assert config.concurrency_level == ConcurrencyLevel.HIGH
        assert config.chunk_size_bytes == 2048
        assert config.max_concurrent_streams == 20


class TestStreamingMetrics:
    """Test StreamingMetrics functionality"""
    
    def test_default_metrics(self):
        """Test default metrics values"""
        metrics = StreamingMetrics()
        
        assert metrics.first_chunk_latency == 0.0
        assert metrics.chunks_streamed == 0
        assert metrics.bytes_streamed == 0
        assert metrics.active_streams == 0
        assert metrics.stream_interruptions == 0
        assert isinstance(metrics.session_start, datetime)
    
    def test_streaming_efficiency_calculation(self):
        """Test streaming efficiency calculation"""
        metrics = StreamingMetrics()
        
        # No chunks streamed - should return 100%
        assert metrics.get_streaming_efficiency() == 100.0
        
        # Perfect streaming
        metrics.chunks_streamed = 100
        metrics.chunk_retransmissions = 0
        assert metrics.get_streaming_efficiency() == 100.0
        
        # 95% efficiency
        metrics.chunk_retransmissions = 5
        assert metrics.get_streaming_efficiency() == 95.0
        
        # 50% efficiency
        metrics.chunk_retransmissions = 50
        assert metrics.get_streaming_efficiency() == 50.0


class TestResponseChunk:
    """Test ResponseChunk functionality"""
    
    def test_response_chunk_creation(self):
        """Test ResponseChunk creation"""
        test_data = b"test_audio_data"
        chunk = ResponseChunk(
            sequence_id=1,
            data=test_data,
            chunk_type="audio",
            timestamp=time.time(),
            priority=ResponsePriority.HIGH,
            is_final=False
        )
        
        assert chunk.sequence_id == 1
        assert chunk.data == test_data
        assert chunk.chunk_type == "audio"
        assert chunk.priority == ResponsePriority.HIGH
        assert chunk.is_final is False
        assert isinstance(chunk.metadata, dict)
    
    def test_chunk_age_calculation(self):
        """Test chunk age calculation"""
        past_time = time.time() - 0.1  # 100ms ago
        chunk = ResponseChunk(
            sequence_id=1,
            data=b"test",
            chunk_type="audio",
            timestamp=past_time
        )
        
        age_ms = chunk.get_age_ms()
        assert age_ms >= 90  # Should be around 100ms
        assert age_ms <= 150  # Allow some variance


class TestAsyncResponseStream:
    """Test AsyncResponseStream functionality"""
    
    @pytest.mark.asyncio
    async def test_stream_initialization(self, async_config, logger):
        """Test stream initialization"""
        stream = AsyncResponseStream("test_stream", async_config, logger)
        
        # Initialize stream
        success = await stream.initialize()
        assert success is True
        assert stream.is_active is True
        assert stream.streaming_task is not None
        assert stream.chunk_processor_task is not None
        
        # Shutdown stream
        await stream.shutdown()
        assert stream.is_active is False
    
    @pytest.mark.asyncio
    async def test_chunk_queuing(self, async_config, logger):
        """Test chunk queuing and processing"""
        stream = AsyncResponseStream("test_stream", async_config, logger)
        await stream.initialize()
        
        try:
            # Queue a chunk
            test_data = b"test_audio_chunk"
            success = await stream.queue_chunk(
                data=test_data,
                chunk_type="audio",
                priority=ResponsePriority.NORMAL
            )
            
            assert success is True
            assert stream.is_streaming is True
            assert stream.sequence_counter == 1
            
            # Queue final chunk
            await stream.queue_chunk(
                data=b"final_chunk",
                chunk_type="audio",
                is_final=True
            )
            
            # Wait a bit for processing
            await asyncio.sleep(0.1)
            
            # Check metrics
            assert stream.metrics.chunks_streamed >= 1
            
        finally:
            await stream.shutdown()
    
    @pytest.mark.asyncio
    async def test_priority_processing(self, async_config, logger):
        """Test priority processing"""
        async_config.enable_priority_processing = True
        stream = AsyncResponseStream("test_stream", async_config, logger)
        await stream.initialize()
        
        try:
            # Queue high priority chunk
            await stream.queue_chunk(
                data=b"high_priority",
                priority=ResponsePriority.HIGH
            )
            
            # Queue normal priority chunk
            await stream.queue_chunk(
                data=b"normal_priority",
                priority=ResponsePriority.NORMAL
            )
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            assert stream.metrics.chunks_streamed >= 2
            
        finally:
            await stream.shutdown()
    
    @pytest.mark.asyncio
    async def test_callback_system(self, async_config, logger):
        """Test callback system"""
        stream = AsyncResponseStream("test_stream", async_config, logger)
        await stream.initialize()
        
        # Track callbacks
        chunk_callbacks = []
        completion_callbacks = []
        
        def chunk_callback(chunk):
            chunk_callbacks.append(chunk)
        
        def completion_callback(stream_id):
            completion_callbacks.append(stream_id)
        
        stream.add_chunk_callback(chunk_callback)
        stream.add_completion_callback(completion_callback)
        
        try:
            # Queue chunks
            await stream.queue_chunk(data=b"chunk1")
            await stream.queue_chunk(data=b"chunk2", is_final=True)
            
            # Wait for processing
            await asyncio.sleep(0.2)
            
            # Check callbacks were called
            assert len(chunk_callbacks) >= 1
            assert len(completion_callbacks) >= 1
            
        finally:
            await stream.shutdown()


class TestConcurrentAudioProcessor:
    """Test ConcurrentAudioProcessor functionality"""
    
    @pytest.mark.asyncio
    async def test_processor_initialization(self, async_config, logger):
        """Test audio processor initialization"""
        processor = ConcurrentAudioProcessor(async_config, logger)
        
        # Initialize
        success = await processor.initialize()
        assert success is True
        assert processor.is_processing is True
        assert len(processor.worker_tasks) >= 2  # At least 2 workers for LOW concurrency
        
        # Shutdown
        await processor.shutdown()
        assert processor.is_processing is False
    
    @pytest.mark.asyncio
    async def test_worker_count_configuration(self, logger):
        """Test worker count based on concurrency level"""
        # Test LOW concurrency
        config_low = AsyncConfig(concurrency_level=ConcurrencyLevel.LOW)
        processor_low = ConcurrentAudioProcessor(config_low, logger)
        assert processor_low._get_worker_count() == 2
        
        # Test MEDIUM concurrency
        config_medium = AsyncConfig(concurrency_level=ConcurrencyLevel.MEDIUM)
        processor_medium = ConcurrentAudioProcessor(config_medium, logger)
        assert processor_medium._get_worker_count() == 4
        
        # Test HIGH concurrency
        config_high = AsyncConfig(concurrency_level=ConcurrencyLevel.HIGH)
        processor_high = ConcurrentAudioProcessor(config_high, logger)
        assert processor_high._get_worker_count() == 8
    
    @pytest.mark.asyncio
    async def test_audio_processing(self, async_config, logger):
        """Test audio processing functionality"""
        processor = ConcurrentAudioProcessor(async_config, logger)
        await processor.initialize()
        
        try:
            # Process audio data
            test_audio = b"test_audio_data_12345"
            success = await processor.process_audio_async(test_audio)
            assert success is True
            
            # Get processed audio
            processed_audio = await processor.get_processed_audio(timeout=1.0)
            assert processed_audio is not None
            assert processed_audio == test_audio  # Should be unchanged in simulation
            
            # Check metrics
            metrics = processor.get_processing_metrics()
            assert metrics["processed_chunks"] >= 1
            assert metrics["active_workers"] >= 1
            
        finally:
            await processor.shutdown()
    
    @pytest.mark.asyncio
    async def test_priority_audio_processing(self, async_config, logger):
        """Test priority audio processing"""
        processor = ConcurrentAudioProcessor(async_config, logger)
        await processor.initialize()
        
        try:
            # Process high priority audio
            high_priority_audio = b"high_priority_audio"
            success = await processor.process_audio_async(
                high_priority_audio,
                priority=ResponsePriority.HIGH
            )
            assert success is True
            
            # Process normal priority audio
            normal_audio = b"normal_audio"
            await processor.process_audio_async(normal_audio)
            
            # Get processed audio (high priority should be processed first)
            processed_audio = await processor.get_processed_audio(timeout=1.0)
            assert processed_audio is not None
            
        finally:
            await processor.shutdown()
    
    @pytest.mark.asyncio
    async def test_queue_overflow_handling(self, logger):
        """Test queue overflow handling"""
        config = AsyncConfig(max_queue_size=2)  # Very small queue
        processor = ConcurrentAudioProcessor(config, logger)
        await processor.initialize()
        
        try:
            # Try to overflow the queue
            results = []
            for i in range(5):
                result = await processor.process_audio_async(f"audio_{i}".encode())
                results.append(result)
            
            # Some should succeed, some might fail due to queue limits
            assert any(results)  # At least some should succeed
            
        finally:
            await processor.shutdown()


class TestAsyncResponseManager:
    """Test AsyncResponseManager functionality"""
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self, async_config, logger):
        """Test async response manager initialization"""
        manager = AsyncResponseManager(async_config, logger)
        
        # Initialize
        success = await manager.initialize()
        assert success is True
        assert manager.is_running is True
        assert manager.audio_processor is not None
        assert manager.manager_task is not None
        
        # Shutdown
        await manager.shutdown()
        assert manager.is_running is False
    
    @pytest.mark.asyncio
    async def test_stream_creation_and_management(self, async_config, logger):
        """Test stream creation and management"""
        manager = AsyncResponseManager(async_config, logger)
        await manager.initialize()
        
        try:
            # Create stream
            stream_id = await manager.create_response_stream()
            assert stream_id is not None
            assert stream_id in manager.active_streams
            assert len(manager.active_streams) == 1
            
            # Create named stream
            named_stream_id = await manager.create_response_stream("custom_stream")
            assert named_stream_id == "custom_stream"
            assert len(manager.active_streams) == 2
            
            # Close stream
            success = await manager.close_response_stream(stream_id)
            assert success is True
            assert len(manager.active_streams) == 1
            
        finally:
            await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_response_streaming(self, async_config, logger):
        """Test response streaming functionality"""
        manager = AsyncResponseManager(async_config, logger)
        await manager.initialize()
        
        try:
            # Create stream
            stream_id = await manager.create_response_stream()
            
            # Stream chunks
            chunk1 = b"audio_chunk_1"
            chunk2 = b"audio_chunk_2"
            final_chunk = b"final_audio_chunk"
            
            success1 = await manager.stream_response_chunk(stream_id, chunk1, "audio")
            success2 = await manager.stream_response_chunk(stream_id, chunk2, "audio", ResponsePriority.HIGH)
            success3 = await manager.stream_response_chunk(stream_id, final_chunk, "audio", is_final=True)
            
            assert success1 is True
            assert success2 is True
            assert success3 is True
            
            # Wait for processing
            await asyncio.sleep(0.2)
            
            # Check metrics
            metrics = manager.get_comprehensive_metrics()
            assert metrics["manager"]["total_chunks_delivered"] >= 3
            
        finally:
            await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_concurrent_audio_processing(self, async_config, logger):
        """Test concurrent audio processing integration"""
        manager = AsyncResponseManager(async_config, logger)
        await manager.initialize()
        
        try:
            # Process audio concurrently
            test_audio = b"concurrent_test_audio"
            success = await manager.process_audio_concurrent(test_audio)
            assert success is True
            
            # Get processed audio
            processed_audio = await manager.get_processed_audio(timeout=1.0)
            assert processed_audio is not None
            assert processed_audio == test_audio
            
        finally:
            await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_max_concurrent_streams_limit(self, async_config, logger):
        """Test maximum concurrent streams limit"""
        async_config.max_concurrent_streams = 2  # Set low limit
        manager = AsyncResponseManager(async_config, logger)
        await manager.initialize()
        
        try:
            # Create streams up to limit
            stream1 = await manager.create_response_stream()
            stream2 = await manager.create_response_stream()
            
            # Try to exceed limit
            with pytest.raises(RuntimeError, match="Maximum concurrent streams"):
                await manager.create_response_stream()
            
            assert len(manager.active_streams) == 2
            
        finally:
            await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, async_config, logger):
        """Test comprehensive metrics collection"""
        manager = AsyncResponseManager(async_config, logger)
        await manager.initialize()
        
        try:
            # Create stream and process data
            stream_id = await manager.create_response_stream()
            await manager.stream_response_chunk(stream_id, b"test_data", "audio")
            await manager.process_audio_concurrent(b"audio_data")
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            # Get metrics
            metrics = manager.get_comprehensive_metrics()
            
            # Check manager metrics
            assert "manager" in metrics
            assert metrics["manager"]["active_streams"] >= 1
            assert metrics["manager"]["total_streams_created"] >= 1
            
            # Check audio processor metrics
            assert "audio_processor" in metrics
            assert "active_workers" in metrics["audio_processor"]
            
            # Check stream metrics
            assert "streams" in metrics
            assert stream_id in metrics["streams"]
            
            # Check global streaming metrics
            assert "global_streaming" in metrics
            assert "chunks_streamed" in metrics["global_streaming"]
            
        finally:
            await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_stream_cleanup(self, async_config, logger):
        """Test automatic stream cleanup"""
        manager = AsyncResponseManager(async_config, logger)
        await manager.initialize()
        
        try:
            # Create and immediately close stream
            stream_id = await manager.create_response_stream()
            stream = manager.active_streams[stream_id]
            
            # Manually set stream as inactive to simulate completion
            stream.is_active = False
            
            # Wait for cleanup cycle
            await asyncio.sleep(1.5)  # Wait for management loop
            
            # Stream should be cleaned up
            assert stream_id not in manager.active_streams
            
        finally:
            await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, async_config, logger):
        """Test error handling in various scenarios"""
        manager = AsyncResponseManager(async_config, logger)
        await manager.initialize()
        
        try:
            # Test streaming to non-existent stream
            success = await manager.stream_response_chunk("non_existent", b"data")
            assert success is False
            
            # Test closing non-existent stream
            success = await manager.close_response_stream("non_existent")
            assert success is False
            
        finally:
            await manager.shutdown()


class TestFactoryFunction:
    """Test factory function for easy creation"""
    
    def test_factory_function_defaults(self):
        """Test factory function with defaults"""
        manager = create_async_response_manager()
        
        assert isinstance(manager, AsyncResponseManager)
        assert manager.config.stream_mode == StreamMode.IMMEDIATE
        assert manager.config.concurrency_level == ConcurrencyLevel.MEDIUM
        assert manager.config.enable_progressive_audio is True
        assert manager.config.max_concurrent_streams == 10
    
    def test_factory_function_custom(self, logger):
        """Test factory function with custom parameters"""
        manager = create_async_response_manager(
            stream_mode=StreamMode.CHUNKED,
            concurrency_level=ConcurrencyLevel.HIGH,
            enable_progressive_audio=False,
            max_concurrent_streams=20,
            logger=logger
        )
        
        assert manager.config.stream_mode == StreamMode.CHUNKED
        assert manager.config.concurrency_level == ConcurrencyLevel.HIGH
        assert manager.config.enable_progressive_audio is False
        assert manager.config.max_concurrent_streams == 20
        assert manager.logger == logger


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_streaming_scenario(self, async_config, logger):
        """Test complete end-to-end streaming scenario"""
        manager = AsyncResponseManager(async_config, logger)
        await manager.initialize()
        
        try:
            # Simulate realistic voice response scenario
            stream_id = await manager.create_response_stream()
            
            # Stream multiple audio chunks as they would arrive
            audio_chunks = [
                b"hello_audio_chunk_1",
                b"hello_audio_chunk_2", 
                b"hello_audio_chunk_3",
                b"final_audio_chunk"
            ]
            
            # Stream chunks with timing simulation
            for i, chunk in enumerate(audio_chunks):
                is_final = (i == len(audio_chunks) - 1)
                priority = ResponsePriority.HIGH if i == 0 else ResponsePriority.NORMAL
                
                success = await manager.stream_response_chunk(
                    stream_id, chunk, "audio", priority, is_final
                )
                assert success is True
                
                # Small delay between chunks to simulate real-time
                await asyncio.sleep(0.01)
            
            # Process concurrent audio input
            input_audio = b"user_input_audio_data"
            await manager.process_audio_concurrent(input_audio, ResponsePriority.HIGH)
            
            # Wait for all processing to complete
            await asyncio.sleep(0.3)
            
            # Verify results
            metrics = manager.get_comprehensive_metrics()
            assert metrics["manager"]["total_chunks_delivered"] >= len(audio_chunks)
            assert metrics["manager"]["active_streams"] >= 1
            
            # Get processed audio
            processed = await manager.get_processed_audio()
            assert processed is not None
            
        finally:
            await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_streams(self, async_config, logger):
        """Test multiple concurrent streams"""
        manager = AsyncResponseManager(async_config, logger)
        await manager.initialize()
        
        try:
            # Create multiple streams
            stream_ids = []
            for i in range(3):
                stream_id = await manager.create_response_stream(f"stream_{i}")
                stream_ids.append(stream_id)
            
            # Stream data to all streams concurrently
            tasks = []
            for i, stream_id in enumerate(stream_ids):
                task = manager.stream_response_chunk(
                    stream_id, 
                    f"audio_data_for_stream_{i}".encode(),
                    "audio",
                    ResponsePriority.NORMAL
                )
                tasks.append(task)
            
            # Wait for all streams to process
            results = await asyncio.gather(*tasks)
            assert all(results)  # All should succeed
            
            # Wait for processing
            await asyncio.sleep(0.2)
            
            # Check metrics
            metrics = manager.get_comprehensive_metrics()
            assert metrics["manager"]["active_streams"] == 3
            assert len(metrics["streams"]) == 3
            
        finally:
            await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, async_config, logger):
        """Test performance under load"""
        # Configure for higher load testing
        async_config.max_concurrent_streams = 10
        async_config.concurrency_level = ConcurrencyLevel.HIGH
        
        manager = AsyncResponseManager(async_config, logger)
        await manager.initialize()
        
        try:
            start_time = time.time()
            
            # Create many streams and process data
            stream_count = 5
            chunks_per_stream = 10
            
            # Create streams
            streams = []
            for i in range(stream_count):
                stream_id = await manager.create_response_stream()
                streams.append(stream_id)
            
            # Stream many chunks concurrently
            tasks = []
            for stream_id in streams:
                for chunk_idx in range(chunks_per_stream):
                    task = manager.stream_response_chunk(
                        stream_id,
                        f"chunk_{chunk_idx}".encode(),
                        "audio"
                    )
                    tasks.append(task)
            
            # Process all chunks
            await asyncio.gather(*tasks)
            
            # Wait for processing
            await asyncio.sleep(0.5)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Verify performance
            assert processing_time < 2.0  # Should complete within 2 seconds
            
            metrics = manager.get_comprehensive_metrics()
            expected_chunks = stream_count * chunks_per_stream
            assert metrics["manager"]["total_chunks_delivered"] >= expected_chunks
            
        finally:
            await manager.shutdown()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 