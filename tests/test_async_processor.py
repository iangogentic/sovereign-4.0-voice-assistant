"""
Tests for AsyncProcessor

Test suite for the enhanced async processing system including:
- Progressive status updates
- Timeout handling and graceful degradation
- Task cancellation and management
- Performance metrics and statistics
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, Mock
from typing import Dict, Any

from assistant.async_processor import (
    AsyncProcessor, 
    ProcessingConfig, 
    ProcessingState, 
    ProcessingTask
)


class TestAsyncProcessor:
    """Test AsyncProcessor functionality"""
    
    @pytest.fixture
    def processing_config(self):
        """Create test processing configuration"""
        return ProcessingConfig(
            fast_timeout=2.0,
            deep_timeout=5.0,
            initial_status_delay=0.1,
            progress_update_interval=0.5,
            timeout_warning_threshold=0.7,
            enable_degradation=True,
            max_retries=1
        )
    
    @pytest.fixture
    def async_processor(self, processing_config):
        """Create AsyncProcessor instance for testing"""
        return AsyncProcessor(processing_config)
    
    @pytest.fixture
    def mock_status_callback(self):
        """Create mock status update callback"""
        return AsyncMock()
    
    def test_async_processor_initialization(self, async_processor):
        """Test AsyncProcessor initialization"""
        assert isinstance(async_processor.config, ProcessingConfig)
        assert len(async_processor.active_tasks) == 0
        assert async_processor.stats['total_tasks'] == 0
        assert 'fast' in async_processor.processing_semaphores
        assert 'deep' in async_processor.processing_semaphores
    
    def test_processing_config_defaults(self):
        """Test ProcessingConfig default values"""
        config = ProcessingConfig()
        assert config.fast_timeout == 5.0
        assert config.deep_timeout == 30.0
        assert config.enable_degradation is True
        assert config.max_retries == 2
        assert len(config.status_messages['initializing']) > 0
    
    def test_processing_task_properties(self):
        """Test ProcessingTask property calculations"""
        task = ProcessingTask(
            task_id="test",
            query="test query",
            model_type="fast",
            start_time=time.time() - 2.0,
            timeout=5.0
        )
        
        assert task.elapsed_time >= 2.0
        assert task.remaining_time <= 3.0
        assert task.timeout_percentage >= 0.4
        assert not task.is_cancelled()
    
    def test_processing_task_cancellation(self):
        """Test ProcessingTask cancellation"""
        task = ProcessingTask(
            task_id="test",
            query="test query",
            model_type="fast",
            start_time=time.time(),
            timeout=5.0
        )
        
        # Cancel the task
        task.cancel()
        
        assert task.is_cancelled()
        assert task.state == ProcessingState.CANCELLED
    
    @pytest.mark.asyncio
    async def test_fast_processing_success(self, async_processor, mock_status_callback):
        """Test successful fast processing"""
        async_processor.set_status_update_callback(mock_status_callback)
        
        async def mock_processor(query: str) -> str:
            await asyncio.sleep(0.1)  # Quick processing
            return f"Result for: {query}"
        
        result = await async_processor.process_async(
            query="fast test query",
            model_type="fast",
            processor_func=mock_processor,
            request_id="test_fast"
        )
        
        assert result == "Result for: fast test query"
        assert async_processor.stats['completed_tasks'] == 1
        assert async_processor.stats['total_tasks'] == 1
        
        # Should have minimal status updates for fast processing
        assert mock_status_callback.call_count <= 2
    
    @pytest.mark.asyncio
    async def test_deep_processing_with_status_updates(self, async_processor, mock_status_callback):
        """Test deep processing with progressive status updates"""
        async_processor.set_status_update_callback(mock_status_callback)
        
        async def mock_processor(query: str) -> str:
            await asyncio.sleep(1.0)  # Longer processing
            return f"Deep result for: {query}"
        
        result = await async_processor.process_async(
            query="complex deep query",
            model_type="deep", 
            processor_func=mock_processor,
            request_id="test_deep"
        )
        
        assert result == "Deep result for: complex deep query"
        assert async_processor.stats['completed_tasks'] == 1
        
        # Should have multiple status updates
        assert mock_status_callback.call_count >= 2
        
        # Verify status update calls
        calls = mock_status_callback.call_args_list
        assert len(calls) >= 1
        
        # First call should be initializing message
        task_id, message = calls[0][0]
        assert task_id == "test_deep"
        assert isinstance(message, str)
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, async_processor, mock_status_callback):
        """Test timeout handling"""
        async_processor.set_status_update_callback(mock_status_callback)
        
        async def slow_processor(query: str) -> str:
            await asyncio.sleep(10.0)  # Longer than timeout
            return f"Should not reach here: {query}"
        
        with pytest.raises(asyncio.TimeoutError):
            await async_processor.process_async(
                query="timeout test query",
                model_type="fast",  # 2.0s timeout
                processor_func=slow_processor,
                request_id="test_timeout"
            )
        
        assert async_processor.stats['failed_tasks'] == 1
        assert async_processor.stats['timeout_rate'] == 1.0
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, async_processor, mock_status_callback):
        """Test graceful degradation on timeout"""
        async_processor.set_status_update_callback(mock_status_callback)
        
        call_count = 0
        
        async def degradation_processor(query: str) -> str:
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                # First call times out
                await asyncio.sleep(10.0)
                return "Should not reach here"
            else:
                # Second call (degraded) succeeds
                await asyncio.sleep(0.1)
                return f"Degraded result for: {query}"
        
        result = await async_processor.process_async(
            query="degradation test query",
            model_type="deep",  # 5.0s timeout
            processor_func=degradation_processor,
            request_id="test_degradation"
        )
        
        assert result == "Degraded result for: degradation test query"
        assert async_processor.stats['degraded_tasks'] == 1
        assert async_processor.stats['completed_tasks'] == 1
        assert call_count == 2
        
        # Should have degradation status update
        degradation_calls = [call for call in mock_status_callback.call_args_list 
                           if 'faster approach' in call[0][1] or 'quicker method' in call[0][1]]
        assert len(degradation_calls) > 0
    
    @pytest.mark.asyncio
    async def test_task_cancellation(self, async_processor, mock_status_callback):
        """Test task cancellation"""
        async_processor.set_status_update_callback(mock_status_callback)
        
        async def cancellable_processor(query: str) -> str:
            await asyncio.sleep(5.0)  # Long enough to be cancelled
            return f"Should be cancelled: {query}"
        
        # Start processing
        task = asyncio.create_task(
            async_processor.process_async(
                query="cancellation test",
                model_type="deep",
                processor_func=cancellable_processor,
                request_id="test_cancel"
            )
        )
        
        # Let it start
        await asyncio.sleep(0.2)
        
        # Cancel the task
        success = async_processor.cancel_task("test_cancel")
        assert success
        
        # Wait for cancellation to complete
        with pytest.raises(asyncio.CancelledError):
            await task
        
        assert async_processor.stats['cancelled_tasks'] == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, async_processor, mock_status_callback):
        """Test concurrent task processing"""
        async_processor.set_status_update_callback(mock_status_callback)
        
        async def concurrent_processor(query: str) -> str:
            await asyncio.sleep(0.2)
            return f"Concurrent result: {query}"
        
        # Start multiple tasks concurrently
        tasks = []
        for i in range(5):
            task = asyncio.create_task(
                async_processor.process_async(
                    query=f"concurrent query {i}",
                    model_type="fast",
                    processor_func=concurrent_processor,
                    request_id=f"concurrent_{i}"
                )
            )
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all("Concurrent result:" in result for result in results)
        assert async_processor.stats['completed_tasks'] == 5
        assert async_processor.stats['total_tasks'] == 5
    
    @pytest.mark.asyncio
    async def test_error_handling(self, async_processor, mock_status_callback):
        """Test error handling in processing"""
        async_processor.set_status_update_callback(mock_status_callback)
        
        async def error_processor(query: str) -> str:
            raise ValueError("Processing error")
        
        with pytest.raises(ValueError, match="Processing error"):
            await async_processor.process_async(
                query="error test query",
                model_type="fast",
                processor_func=error_processor,
                request_id="test_error"
            )
        
        assert async_processor.stats['failed_tasks'] == 1
    
    def test_task_status_retrieval(self, async_processor):
        """Test task status retrieval"""
        # No active tasks initially
        status = async_processor.get_task_status("nonexistent")
        assert status is None
        
        # Would need to test active task status during processing
        # This is tested implicitly in other async tests
    
    def test_statistics_calculation(self, async_processor):
        """Test statistics calculation"""
        # Initial statistics
        stats = async_processor.get_statistics()
        assert stats['total_tasks'] == 0
        assert stats['success_rate'] == 0.0
        assert stats['timeout_rate'] == 0.0
        
        # Manually update stats for testing
        async_processor.stats.update({
            'total_tasks': 10,
            'completed_tasks': 8,
            'failed_tasks': 1,
            'degraded_tasks': 1,
            'avg_processing_time': 2.5
        })
        async_processor._update_timeout_rate()
        
        stats = async_processor.get_statistics()
        assert stats['total_tasks'] == 10
        assert stats['completed_tasks'] == 8
        assert stats['success_rate'] == 0.8
        assert stats['timeout_rate'] == 0.1
        assert stats['avg_processing_time'] == 2.5
    
    @pytest.mark.asyncio
    async def test_status_message_selection(self, async_processor, mock_status_callback):
        """Test status message selection and variety"""
        async_processor.set_status_update_callback(mock_status_callback)
        
        async def test_processor(query: str) -> str:
            await asyncio.sleep(0.8)  # Long enough for status updates
            return f"Result: {query}"
        
        await async_processor.process_async(
            query="status message test",
            model_type="deep",
            processor_func=test_processor,
            request_id="test_status"
        )
        
        # Verify status messages were sent
        assert mock_status_callback.call_count >= 1
        
        # Check that messages are from expected categories
        calls = mock_status_callback.call_args_list
        messages = [call[0][1] for call in calls]
        
        # Should have at least one initializing message
        initializing_messages = async_processor.config.status_messages['initializing']
        assert any(msg in initializing_messages for msg in messages)
    
    @pytest.mark.asyncio
    async def test_semaphore_concurrency_limits(self, processing_config):
        """Test that semaphores properly limit concurrency"""
        # Set very low limits for testing
        processing_config.max_concurrent_fast = 2
        processing_config.max_concurrent_deep = 1
        
        async_processor = AsyncProcessor(processing_config)
        
        async def slow_processor(query: str) -> str:
            await asyncio.sleep(0.5)
            return f"Result: {query}"
        
        # Start more tasks than the semaphore allows
        tasks = []
        start_time = time.time()
        
        for i in range(4):  # More than max_concurrent_fast (2)
            task = asyncio.create_task(
                async_processor.process_async(
                    query=f"concurrent test {i}",
                    model_type="fast",
                    processor_func=slow_processor,
                    request_id=f"semaphore_test_{i}"
                )
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Should take longer than single task due to concurrency limits
        assert total_time >= 1.0  # At least 2 batches of 0.5s each
        assert len(results) == 4
        assert async_processor.stats['completed_tasks'] == 4
    
    def test_cancel_all_tasks(self, async_processor):
        """Test cancelling all active tasks"""
        # Manually add some mock tasks to test cancellation
        for i in range(3):
            task = ProcessingTask(
                task_id=f"test_{i}",
                query=f"query {i}",
                model_type="fast",
                start_time=time.time(),
                timeout=5.0
            )
            async_processor.active_tasks[f"test_{i}"] = task
        
        # Cancel all tasks
        async_processor.cancel_all_tasks()
        
        # All tasks should be cancelled
        for task in async_processor.active_tasks.values():
            assert task.is_cancelled()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 