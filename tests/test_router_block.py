"""
Tests for RouterBlock Pipecat Integration

Test suite for the RouterBlock class that integrates with Pipecat framework
to route queries between fast and deep LLM services.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

from assistant.router_block import RouterBlock, RouterState, RouterMetrics, create_router_block
from assistant.llm_router import QueryComplexity, QueryClassification
from assistant.router_config import RouterConfig, ModelConfig

# Mock Pipecat components
class MockFrame:
    """Mock Frame for testing"""
    pass

class MockTextFrame(MockFrame):
    """Mock TextFrame for testing"""
    def __init__(self, text: str):
        self.text = text

class MockLLMMessagesFrame(MockFrame):
    """Mock LLMMessagesFrame for testing"""
    def __init__(self, messages: List[Dict[str, str]]):
        self.messages = messages

class MockSystemFrame(MockFrame):
    """Mock SystemFrame for testing"""
    def __init__(self, message: str):
        self.message = message

class MockErrorFrame(MockFrame):
    """Mock ErrorFrame for testing"""
    def __init__(self, error: str):
        self.error = error

class MockBotSpeakingFrame(MockFrame):
    """Mock BotSpeakingFrame for testing"""
    pass

class MockOpenRouterLLMService:
    """Mock OpenRouterLLMService for testing"""
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.setup_called = False
        self.cleanup_called = False
        self.push_frame_calls = []
        
    async def setup(self):
        self.setup_called = True
        
    async def cleanup(self):
        self.cleanup_called = True
        
    async def push_frame(self, frame, direction):
        self.push_frame_calls.append((frame, direction))


@pytest.fixture
def router_config():
    """Create test router configuration"""
    return RouterConfig(
        openrouter_api_key="test_key",
        openrouter_base_url="https://openrouter.ai/api/v1",
        models={
            'fast': ModelConfig(
                id="openai/gpt-4o-mini",
                name="GPT-4o-mini",
                max_tokens=500,
                temperature=0.7,
                timeout=5.0,
                cost_per_1k_tokens=0.15
            ),
            'deep': ModelConfig(
                id="openai/gpt-4o",
                name="GPT-4o",
                max_tokens=2000,
                temperature=0.7,
                timeout=30.0,
                cost_per_1k_tokens=1.50
            )
        },
        max_conversation_history=10
    )

@pytest.fixture
def mock_services():
    """Create mock LLM services"""
    fast_service = MockOpenRouterLLMService("openai/gpt-4o-mini")
    deep_service = MockOpenRouterLLMService("openai/gpt-4o")
    return fast_service, deep_service

@pytest.fixture
def router_block(router_config, mock_services):
    """Create RouterBlock instance for testing"""
    fast_service, deep_service = mock_services
    return RouterBlock(
        config=router_config,
        fast_llm_service=fast_service,
        deep_llm_service=deep_service
    )


class TestRouterBlock:
    """Test RouterBlock functionality"""
    
    def test_router_block_initialization(self, router_block):
        """Test RouterBlock initialization"""
        assert router_block.state == RouterState.IDLE
        assert router_block.current_request_id is None
        assert len(router_block.pending_requests) == 0
        assert len(router_block.conversation_history) == 0
        assert isinstance(router_block.metrics, RouterMetrics)
        assert router_block.classifier is not None
        assert router_block.llm_router is not None
        
    def test_create_router_block_factory(self, router_config):
        """Test factory function"""
        router = create_router_block(config=router_config)
        assert isinstance(router, RouterBlock)
        assert router.config == router_config
        
    @pytest.mark.asyncio
    async def test_router_setup_and_cleanup(self, router_block):
        """Test RouterBlock setup and cleanup"""
        # Test setup
        await router_block.setup()
        assert router_block.fast_llm_service.setup_called
        assert router_block.deep_llm_service.setup_called
        
        # Test cleanup
        await router_block.cleanup()
        assert router_block.fast_llm_service.cleanup_called
        assert router_block.deep_llm_service.cleanup_called
        
    @pytest.mark.asyncio
    async def test_simple_query_routing(self, router_block):
        """Test routing of simple queries to fast service"""
        # Mock the push_frame method
        router_block.push_frame = AsyncMock()
        
        # Mock the classification to return simple complexity
        with patch.object(router_block.classifier, 'classify_query') as mock_classify:
            mock_classify.return_value = QueryClassification(
                complexity=QueryComplexity.SIMPLE,
                confidence=0.95,
                reasoning="Short greeting query",
                factors={'length': 12, 'complex_keywords': 0, 'simple_keywords': 1}
            )
            
            # Create test frame
            frame = MockTextFrame("Hello there")
            
            # Process the frame
            await router_block.process_frame(frame, "downstream")
            
            # Check that fast service was used
            assert len(router_block.fast_llm_service.push_frame_calls) == 1
            assert len(router_block.deep_llm_service.push_frame_calls) == 0
            
            # Check metrics
            assert router_block.metrics.total_requests == 1
            assert router_block.metrics.fast_requests == 1
            assert router_block.metrics.deep_requests == 0
            
    @pytest.mark.asyncio
    async def test_complex_query_routing(self, router_block):
        """Test routing of complex queries to deep service"""
        # Mock the push_frame method
        router_block.push_frame = AsyncMock()
        
        # Mock the classification to return complex complexity
        with patch.object(router_block.classifier, 'classify_query') as mock_classify:
            mock_classify.return_value = QueryClassification(
                complexity=QueryComplexity.COMPLEX,
                confidence=0.90,
                reasoning="Complex technical analysis request",
                factors={'length': 85, 'complex_keywords': 3, 'simple_keywords': 0}
            )
            
            # Create test frame
            frame = MockTextFrame("Analyze the performance implications of using microservices architecture")
            
            # Process the frame
            await router_block.process_frame(frame, "downstream")
            
            # Check that deep service was used
            assert len(router_block.fast_llm_service.push_frame_calls) == 0
            assert len(router_block.deep_llm_service.push_frame_calls) == 1
            
            # Check metrics
            assert router_block.metrics.total_requests == 1
            assert router_block.metrics.fast_requests == 0
            assert router_block.metrics.deep_requests == 1
            
            # Check that status update was sent
            assert router_block.push_frame.call_count >= 1
            
    @pytest.mark.asyncio
    async def test_fallback_routing_on_error(self, router_block):
        """Test fallback routing when primary service fails"""
        # Mock the push_frame method
        router_block.push_frame = AsyncMock()
        
        # Mock the classification
        with patch.object(router_block.classifier, 'classify_query') as mock_classify:
            mock_classify.return_value = QueryClassification(
                complexity=QueryComplexity.COMPLEX,
                confidence=0.90,
                reasoning="Complex query",
                factors={'length': 50, 'complex_keywords': 2, 'simple_keywords': 0}
            )
            
            # Make the deep service fail
            router_block.deep_llm_service.push_frame = AsyncMock(side_effect=Exception("Service unavailable"))
            
            # Create test frame
            frame = MockTextFrame("Complex query that should fail")
            
            # Process the frame
            await router_block.process_frame(frame, "downstream")
            
            # Check that fallback to fast service occurred
            assert len(router_block.fast_llm_service.push_frame_calls) == 1
            assert router_block.metrics.fallback_routes == 1
            
    @pytest.mark.asyncio
    async def test_conversation_history_management(self, router_block):
        """Test conversation history tracking"""
        # Mock the push_frame method
        router_block.push_frame = AsyncMock()
        
        # Mock classification
        with patch.object(router_block.classifier, 'classify_query') as mock_classify:
            mock_classify.return_value = QueryClassification(
                complexity=QueryComplexity.SIMPLE,
                confidence=0.95,
                reasoning="Simple query",
                factors={'length': 10, 'complex_keywords': 0, 'simple_keywords': 1}
            )
            
            # Process multiple frames
            for i in range(3):
                frame = MockTextFrame(f"Query {i}")
                await router_block.process_frame(frame, "downstream")
            
            # Check conversation history
            assert len(router_block.conversation_history) == 3
            assert router_block.conversation_history[0]['content'] == "Query 0"
            assert router_block.conversation_history[2]['content'] == "Query 2"
            
    @pytest.mark.asyncio
    async def test_conversation_history_limit(self, router_block):
        """Test conversation history length limit"""
        # Mock the push_frame method
        router_block.push_frame = AsyncMock()
        
        # Set a small history limit
        router_block.max_context_length = 2
        
        # Mock classification
        with patch.object(router_block.classifier, 'classify_query') as mock_classify:
            mock_classify.return_value = QueryClassification(
                complexity=QueryComplexity.SIMPLE,
                confidence=0.95,
                reasoning="Simple query",
                factors={'length': 10, 'complex_keywords': 0, 'simple_keywords': 1}
            )
            
            # Process more frames than the limit
            for i in range(5):
                frame = MockTextFrame(f"Query {i}")
                await router_block.process_frame(frame, "downstream")
            
            # Check that history is limited
            assert len(router_block.conversation_history) == 2
            assert router_block.conversation_history[0]['content'] == "Query 3"
            assert router_block.conversation_history[1]['content'] == "Query 4"
            
    @pytest.mark.asyncio
    async def test_message_building_simple(self, router_block):
        """Test message building for simple queries"""
        classification = QueryClassification(
            complexity=QueryComplexity.SIMPLE,
            confidence=0.95,
            reasoning="Simple query",
            factors={'length': 10, 'complex_keywords': 0, 'simple_keywords': 1}
        )
        
        messages = router_block._build_messages("Hello", classification)
        
        assert len(messages) == 2
        assert messages[0]['role'] == 'system'
        assert 'concise' in messages[0]['content']
        assert messages[1]['role'] == 'user'
        assert messages[1]['content'] == "Hello"
        
    @pytest.mark.asyncio
    async def test_message_building_complex(self, router_block):
        """Test message building for complex queries"""
        classification = QueryClassification(
            complexity=QueryComplexity.COMPLEX,
            confidence=0.90,
            reasoning="Complex query",
            factors={'length': 50, 'complex_keywords': 2, 'simple_keywords': 0}
        )
        
        messages = router_block._build_messages("Analyze system architecture", classification)
        
        assert len(messages) == 2
        assert messages[0]['role'] == 'system'
        assert 'comprehensive' in messages[0]['content']
        assert messages[1]['role'] == 'user'
        assert messages[1]['content'] == "Analyze system architecture"
        
    @pytest.mark.asyncio
    async def test_message_building_with_history(self, router_block):
        """Test message building with conversation history"""
        # Add some conversation history
        router_block.conversation_history = [
            {'role': 'user', 'content': 'Previous question'},
            {'role': 'assistant', 'content': 'Previous answer'}
        ]
        
        classification = QueryClassification(
            complexity=QueryComplexity.SIMPLE,
            confidence=0.95,
            reasoning="Simple query",
            factors={'length': 10, 'complex_keywords': 0, 'simple_keywords': 1}
        )
        
        messages = router_block._build_messages("Current question", classification)
        
        assert len(messages) == 4
        assert messages[0]['role'] == 'system'
        assert messages[1]['role'] == 'user'
        assert messages[1]['content'] == 'Previous question'
        assert messages[2]['role'] == 'assistant'
        assert messages[2]['content'] == 'Previous answer'
        assert messages[3]['role'] == 'user'
        assert messages[3]['content'] == 'Current question'
        
    @pytest.mark.asyncio
    async def test_error_handling_in_classification(self, router_block):
        """Test error handling during classification"""
        # Mock the push_frame method
        router_block.push_frame = AsyncMock()
        
        # Make classification fail
        with patch.object(router_block.classifier, 'classify_query') as mock_classify:
            mock_classify.side_effect = Exception("Classification failed")
            
            # Create test frame
            frame = MockTextFrame("Test query")
            
            # Process the frame
            await router_block.process_frame(frame, "downstream")
            
            # Check that error was handled
            assert router_block.state == RouterState.ERROR
            assert router_block.metrics.failed_routes == 1
            
            # Check that error frame was pushed
            assert router_block.push_frame.called
            
    @pytest.mark.asyncio
    async def test_frame_passthrough(self, router_block):
        """Test that non-text frames are passed through unchanged"""
        # Mock the push_frame method
        router_block.push_frame = AsyncMock()
        
        # Create non-text frames
        system_frame = MockSystemFrame("System message")
        error_frame = MockErrorFrame("Error message")
        
        # Process the frames
        await router_block.process_frame(system_frame, "downstream")
        await router_block.process_frame(error_frame, "downstream")
        
        # Check that frames were passed through
        assert router_block.push_frame.call_count == 2
        
    def test_metrics_calculation(self, router_block):
        """Test metrics calculation and updates"""
        # Update metrics
        router_block.metrics.total_requests = 10
        router_block.metrics.fast_requests = 6
        router_block.metrics.deep_requests = 4
        router_block.metrics.successful_routes = 9
        router_block.metrics.failed_routes = 1
        
        # Test metrics retrieval
        metrics = router_block.get_metrics()
        assert metrics['total_requests'] == 10
        assert metrics['fast_requests'] == 6
        assert metrics['deep_requests'] == 4
        assert metrics['successful_routes'] == 9
        assert metrics['failed_routes'] == 1
        
        # Test routing stats
        stats = router_block.get_routing_stats()
        assert stats['fast_percentage'] == 60.0
        assert stats['deep_percentage'] == 40.0
        assert stats['success_rate'] == 90.0
        assert stats['failure_rate'] == 10.0
        
    def test_metrics_with_no_requests(self, router_block):
        """Test metrics when no requests have been processed"""
        stats = router_block.get_routing_stats()
        assert stats['no_requests'] is True
        
    def test_metrics_time_updates(self, router_block):
        """Test metrics time updates"""
        # Test classification time update
        router_block.metrics.update_classification_time(100.0)
        assert router_block.metrics.avg_classification_time == 100.0
        
        router_block.metrics.total_requests = 1
        router_block.metrics.update_classification_time(50.0)
        assert router_block.metrics.avg_classification_time == 75.0
        
        # Test routing time update
        router_block.metrics.update_routing_time(200.0)
        assert router_block.metrics.avg_routing_time == 200.0
        
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, router_block):
        """Test performance benchmarks for routing"""
        # Mock the push_frame method
        router_block.push_frame = AsyncMock()
        
        # Mock classification to be fast
        with patch.object(router_block.classifier, 'classify_query') as mock_classify:
            mock_classify.return_value = QueryClassification(
                complexity=QueryComplexity.SIMPLE,
                confidence=0.95,
                reasoning="Simple query",
                factors={'length': 10, 'complex_keywords': 0, 'simple_keywords': 1}
            )
            
            # Process multiple queries and measure time
            start_time = time.time()
            for i in range(10):
                frame = MockTextFrame(f"Query {i}")
                await router_block.process_frame(frame, "downstream")
            
            total_time = time.time() - start_time
            avg_time_per_query = total_time / 10
            
            # Check that processing is fast (< 50ms per query)
            assert avg_time_per_query < 0.05, f"Average time per query: {avg_time_per_query:.3f}s"
            
            # Check that all queries were processed
            assert router_block.metrics.total_requests == 10
            assert router_block.metrics.fast_requests == 10
            
    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, router_block):
        """Test handling of concurrent requests"""
        # Mock the push_frame method
        router_block.push_frame = AsyncMock()
        
        # Mock classification
        with patch.object(router_block.classifier, 'classify_query') as mock_classify:
            mock_classify.return_value = QueryClassification(
                complexity=QueryComplexity.SIMPLE,
                confidence=0.95,
                reasoning="Simple query",
                factors={'length': 10, 'complex_keywords': 0, 'simple_keywords': 1}
            )
            
            # Create multiple concurrent requests
            tasks = []
            for i in range(5):
                frame = MockTextFrame(f"Concurrent query {i}")
                task = asyncio.create_task(router_block.process_frame(frame, "downstream"))
                tasks.append(task)
            
            # Wait for all tasks to complete
            await asyncio.gather(*tasks)
            
            # Check that all requests were processed
            assert router_block.metrics.total_requests == 5
            assert router_block.metrics.fast_requests == 5
            
    @pytest.mark.asyncio
    async def test_status_update_for_deep_queries(self, router_block):
        """Test that status updates are sent for deep queries"""
        # Mock the push_frame method
        router_block.push_frame = AsyncMock()
        
        # Mock classification for complex query
        with patch.object(router_block.classifier, 'classify_query') as mock_classify:
            mock_classify.return_value = QueryClassification(
                complexity=QueryComplexity.COMPLEX,
                confidence=0.90,
                reasoning="Complex query requiring deep analysis",
                factors={'length': 80, 'complex_keywords': 3, 'simple_keywords': 0}
            )
            
            # Create test frame
            frame = MockTextFrame("Analyze the quantum computing implications")
            
            # Process the frame
            await router_block.process_frame(frame, "downstream")
            
            # Check that status update was sent (at least 2 calls: status text + speaking indicator)
            assert router_block.push_frame.call_count >= 2
            
    @pytest.mark.asyncio
    async def test_router_state_transitions(self, router_block):
        """Test router state transitions during processing"""
        # Mock the push_frame method
        router_block.push_frame = AsyncMock()
        
        # Mock classification
        with patch.object(router_block.classifier, 'classify_query') as mock_classify:
            mock_classify.return_value = QueryClassification(
                complexity=QueryComplexity.SIMPLE,
                confidence=0.95,
                reasoning="Simple query",
                factors={'length': 10, 'complex_keywords': 0, 'simple_keywords': 1}
            )
            
            # Initial state should be IDLE
            assert router_block.state == RouterState.IDLE
            
            # Process a frame
            frame = MockTextFrame("Test query")
            await router_block.process_frame(frame, "downstream")
            
            # State should transition through CLASSIFYING -> ROUTING -> PROCESSING_FAST
            # (final state depends on timing, but should not be IDLE)
            assert router_block.state != RouterState.IDLE


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 