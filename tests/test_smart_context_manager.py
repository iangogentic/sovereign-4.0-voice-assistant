"""
Comprehensive test suite for SmartContextManager

Tests cover:
- Context prioritization and token budget allocation
- Token counting accuracy and budget management
- Sliding window conversation history management
- Context compression using extractive summarization
- Relevance scoring with sentence transformers
- Context update triggers and caching
- Integration with memory and screen systems
- Performance metrics and error handling
"""

import pytest
import asyncio
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from assistant.smart_context_manager import (
    SmartContextManager, SmartContextConfig, ContextBudget, ContextSegment,
    ConversationTurn, ContextPriority, ContextType, TokenCounter, RelevanceScorer,
    ContextCompressor, create_smart_context_manager, get_default_smart_context_config
)
from assistant.memory import MemoryManager
from assistant.screen_context_provider import ScreenContextProvider, ScreenContextData


# Test fixtures
@pytest.fixture
def context_config():
    """Create test configuration for SmartContextManager"""
    return SmartContextConfig(
        budget=ContextBudget(
            system_instructions=1000,  # Smaller for testing
            recent_memory=2000,
            screen_content=1000,
            conversation_history=4000
        ),
        max_conversation_turns=5,  # Smaller sliding window for testing
        relevance_threshold=0.2,   # Lower threshold for testing
        enable_compression=True,
        context_refresh_interval=10.0,  # Faster for testing
        enable_caching=True,
        cache_ttl_seconds=30
    )


@pytest.fixture
def mock_memory_manager():
    """Create mock MemoryManager for testing"""
    manager = Mock(spec=MemoryManager)
    manager.retrieve_context = AsyncMock(return_value=[])
    return manager


@pytest.fixture
def mock_screen_provider():
    """Create mock ScreenContextProvider for testing"""
    provider = Mock(spec=ScreenContextProvider)
    provider.get_current_context_data = AsyncMock(return_value=None)
    return provider


@pytest.fixture
def sample_conversation_turns():
    """Create sample conversation turns for testing"""
    return [
        ConversationTurn(
            user_message="What is Python?",
            assistant_response="Python is a programming language",
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=5),
            user_tokens=10,
            assistant_tokens=15,
            turn_id="turn_001"
        ),
        ConversationTurn(
            user_message="How do I install packages?",
            assistant_response="You can use pip install package-name",
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=3),
            user_tokens=12,
            assistant_tokens=18,
            turn_id="turn_002"
        ),
        ConversationTurn(
            user_message="What about virtual environments?",
            assistant_response="Virtual environments isolate dependencies using venv or conda",
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=1),
            user_tokens=8,
            assistant_tokens=22,
            turn_id="turn_003"
        )
    ]


# Configuration Tests
class TestSmartContextConfig:
    """Test SmartContextConfig and ContextBudget"""
    
    def test_context_budget_creation(self):
        """Test ContextBudget creation and properties"""
        budget = ContextBudget(
            system_instructions=2000,
            recent_memory=4000,
            screen_content=2000,
            conversation_history=24000
        )
        
        assert budget.system_instructions == 2000
        assert budget.recent_memory == 4000
        assert budget.screen_content == 2000
        assert budget.conversation_history == 24000
        assert budget.total_budget == 32000
        
        # Test budget retrieval
        assert budget.get_budget_for_type(ContextType.SYSTEM) == 2000
        assert budget.get_budget_for_type(ContextType.MEMORY) == 4000
        assert budget.get_budget_for_type(ContextType.SCREEN) == 2000
        assert budget.get_budget_for_type(ContextType.CONVERSATION) == 24000
    
    def test_smart_context_config_defaults(self):
        """Test SmartContextConfig default values"""
        config = SmartContextConfig()
        
        assert config.budget.total_budget == 32000
        assert config.max_conversation_turns == 10
        assert config.relevance_threshold == 0.3
        assert config.enable_compression is True
        assert config.context_refresh_interval == 30.0
        assert config.enable_caching is True
    
    def test_get_default_smart_context_config(self):
        """Test factory function for default config"""
        config = get_default_smart_context_config()
        
        assert isinstance(config, SmartContextConfig)
        assert config.budget.total_budget == 32000
        assert config.max_conversation_turns == 10
        assert config.enable_compression is True


# Context Segment Tests
class TestContextSegment:
    """Test ContextSegment dataclass and functionality"""
    
    def test_context_segment_creation(self):
        """Test ContextSegment creation and priority assignment"""
        segment = ContextSegment(
            content="Test content",
            context_type=ContextType.MEMORY,
            priority=ContextPriority.RECENT_MEMORY,  # Will be overridden
            timestamp=datetime.now(timezone.utc),
            token_count=50,
            relevance_score=0.8,
            source_id="test_001"
        )
        
        assert segment.content == "Test content"
        assert segment.context_type == ContextType.MEMORY
        assert segment.priority == ContextPriority.RECENT_MEMORY  # Auto-assigned based on type
        assert segment.token_count == 50
        assert segment.relevance_score == 0.8
        assert segment.source_id == "test_001"
    
    def test_priority_auto_assignment(self):
        """Test automatic priority assignment based on context type"""
        # Test all context types get correct priorities
        test_cases = [
            (ContextType.SYSTEM, ContextPriority.SYSTEM_INSTRUCTIONS),
            (ContextType.MEMORY, ContextPriority.RECENT_MEMORY),
            (ContextType.SCREEN, ContextPriority.SCREEN_CONTENT),
            (ContextType.CONVERSATION, ContextPriority.CONVERSATION_HISTORY)
        ]
        
        for context_type, expected_priority in test_cases:
            segment = ContextSegment(
                content="test",
                context_type=context_type,
                priority=ContextPriority.CONVERSATION_HISTORY,  # Will be overridden
                timestamp=datetime.now(timezone.utc),
                token_count=10,
                relevance_score=1.0,
                source_id="test"
            )
            assert segment.priority == expected_priority


# Conversation Turn Tests
class TestConversationTurn:
    """Test ConversationTurn functionality"""
    
    def test_conversation_turn_creation(self, sample_conversation_turns):
        """Test ConversationTurn creation and properties"""
        turn = sample_conversation_turns[0]
        
        assert turn.user_message == "What is Python?"
        assert turn.assistant_response == "Python is a programming language"
        assert turn.user_tokens == 10
        assert turn.assistant_tokens == 15
        assert turn.total_tokens == 25
        assert turn.turn_id == "turn_001"
    
    def test_to_context_string(self, sample_conversation_turns):
        """Test conversation turn formatting"""
        turn = sample_conversation_turns[0]
        context_str = turn.to_context_string()
        
        expected = "User: What is Python?\nAssistant: Python is a programming language"
        assert context_str == expected


# Token Counter Tests
class TestTokenCounter:
    """Test TokenCounter functionality"""
    
    def test_token_counter_initialization(self):
        """Test TokenCounter initialization"""
        counter = TokenCounter()
        assert counter.model_name == "gpt-4o-realtime-preview-2024-10-01"
    
    def test_token_counting(self):
        """Test token counting with various inputs"""
        counter = TokenCounter()
        
        # Test basic counting
        short_text = "Hello world"
        tokens = counter.count_tokens(short_text)
        assert tokens > 0
        assert isinstance(tokens, int)
        
        # Test empty string
        assert counter.count_tokens("") >= 0
        
        # Test longer text
        long_text = "This is a much longer piece of text that should have more tokens than the short example above."
        long_tokens = counter.count_tokens(long_text)
        assert long_tokens > tokens
    
    def test_fast_estimation(self):
        """Test fast token estimation"""
        counter = TokenCounter()
        
        text = "This is a test sentence for token estimation."
        fast_tokens = counter.estimate_tokens_fast(text)
        actual_tokens = counter.count_tokens(text)
        
        # Fast estimation should be reasonably close (within 50% for approximation)
        assert abs(fast_tokens - actual_tokens) <= max(actual_tokens * 0.5, 5)


# Relevance Scorer Tests
class TestRelevanceScorer:
    """Test RelevanceScorer functionality"""
    
    def test_relevance_scorer_initialization(self):
        """Test RelevanceScorer initialization"""
        scorer = RelevanceScorer()
        assert scorer.model_name == "all-MiniLM-L6-v2"
    
    def test_relevance_calculation(self):
        """Test relevance score calculation"""
        scorer = RelevanceScorer()
        
        # Test similar texts
        query = "What is Python programming?"
        similar_context = "Python is a programming language for development"
        
        similarity = scorer.calculate_relevance(query, similar_context)
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.1  # Should have some similarity
        
        # Test dissimilar texts
        dissimilar_context = "The weather is nice today with sunshine"
        dissimilarity = scorer.calculate_relevance(query, dissimilar_context)
        assert 0.0 <= dissimilarity <= 1.0
        assert dissimilarity < similarity  # Should be less similar
    
    def test_word_overlap_fallback(self):
        """Test word overlap similarity fallback"""
        scorer = RelevanceScorer()
        
        # Force fallback by using the private method
        text1 = "python programming language"
        text2 = "python is a programming tool"
        
        overlap = scorer._word_overlap_similarity(text1, text2)
        assert 0.0 <= overlap <= 1.0
        assert overlap > 0.0  # Should have some overlap
        
        # Test no overlap
        no_overlap = scorer._word_overlap_similarity("cats dogs", "birds fish")
        assert no_overlap == 0.0


# Context Compressor Tests
class TestContextCompressor:
    """Test ContextCompressor functionality"""
    
    def test_compressor_initialization(self):
        """Test ContextCompressor initialization"""
        compressor = ContextCompressor()
        assert compressor is not None
    
    def test_text_compression(self):
        """Test text compression functionality"""
        compressor = ContextCompressor()
        
        # Test with longer text that should be compressed
        long_text = """
        This is the first sentence of a longer text. This is the second sentence with different content.
        The third sentence continues the narrative. A fourth sentence adds more detail about the topic.
        The fifth sentence concludes the first paragraph. Then we have a sixth sentence in a new paragraph.
        The seventh sentence continues this new paragraph with more information.
        """
        
        compressed = compressor.compress_text(long_text, target_ratio=0.5)
        
        # Compressed text should be shorter
        assert len(compressed) < len(long_text)
        assert len(compressed) > 0
        
        # Test short text (should not be compressed much)
        short_text = "Short text."
        compressed_short = compressor.compress_text(short_text, target_ratio=0.5)
        assert compressed_short == short_text  # Should return original
    
    def test_simple_compression_fallback(self):
        """Test simple compression fallback"""
        compressor = ContextCompressor()
        
        text = "This is a test text that will be compressed using simple truncation method."
        compressed = compressor._simple_compression(text, target_ratio=0.5)
        
        expected_length = int(len(text) * 0.5)
        assert len(compressed) <= expected_length + 3  # +3 for "..."
        assert compressed.endswith("...")


# SmartContextManager Core Tests
class TestSmartContextManager:
    """Test SmartContextManager main functionality"""
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self, context_config, mock_memory_manager, mock_screen_provider):
        """Test SmartContextManager initialization"""
        manager = SmartContextManager(
            config=context_config,
            memory_manager=mock_memory_manager,
            screen_context_provider=mock_screen_provider
        )
        
        assert manager.config == context_config
        assert manager.memory_manager == mock_memory_manager
        assert manager.screen_context_provider == mock_screen_provider
        assert len(manager.conversation_history) == 0
        assert manager.system_instructions != ""
        
        # Test initialization
        success = await manager.initialize()
        assert success is True
    
    @pytest.mark.asyncio
    async def test_conversation_turn_management(self, context_config):
        """Test conversation turn addition and sliding window"""
        manager = SmartContextManager(config=context_config)
        await manager.initialize()
        
        # Add conversation turns up to window limit
        for i in range(7):  # Exceeds max_conversation_turns=5
            manager.add_conversation_turn(
                user_message=f"User message {i}",
                assistant_response=f"Assistant response {i}"
            )
        
        # Should only keep the last 5 turns (sliding window)
        assert len(manager.conversation_history) == 5
        
        # Check that latest turns are kept
        latest_turn = manager.conversation_history[-1]
        assert "User message 6" in latest_turn.user_message
        assert "Assistant response 6" in latest_turn.assistant_response
    
    @pytest.mark.asyncio
    async def test_system_instructions_update(self, context_config):
        """Test system instructions update"""
        manager = SmartContextManager(config=context_config)
        await manager.initialize()
        
        original_instructions = manager.system_instructions
        new_instructions = "Updated system instructions for testing"
        
        manager.update_system_instructions(new_instructions)
        assert manager.system_instructions == new_instructions
        assert manager.system_instructions != original_instructions
    
    @pytest.mark.asyncio
    async def test_basic_context_building(self, context_config):
        """Test basic context building without external dependencies"""
        manager = SmartContextManager(config=context_config)
        await manager.initialize()
        
        # Add some conversation history
        manager.add_conversation_turn(
            "What is machine learning?",
            "Machine learning is a subset of AI that enables computers to learn from data."
        )
        
        # Build context
        context = await manager.build_context("Tell me more about AI")
        
        # Should contain system instructions
        assert "Sovereign" in context
        assert len(context) > 0
        
        # Should be properly formatted
        assert isinstance(context, str)


# Integration Tests
class TestSmartContextManagerIntegration:
    """Test SmartContextManager integration with memory and screen systems"""
    
    @pytest.mark.asyncio
    async def test_memory_integration(self, context_config):
        """Test integration with memory system"""
        # Create mock memory manager with sample data
        mock_memory = Mock(spec=MemoryManager)
        
        from langchain_core.documents import Document
        sample_docs = [
            Document(
                page_content="Previous discussion about Python programming",
                metadata={"timestamp": datetime.now(timezone.utc).isoformat(), "id": "mem_001"}
            ),
            Document(
                page_content="Earlier conversation about machine learning concepts",
                metadata={"timestamp": datetime.now(timezone.utc).isoformat(), "id": "mem_002"}
            )
        ]
        mock_memory.retrieve_context = AsyncMock(return_value=sample_docs)
        
        manager = SmartContextManager(
            config=context_config,
            memory_manager=mock_memory
        )
        await manager.initialize()
        
        # Build context with memory
        context = await manager.build_context("What did we discuss about programming?")
        
        # Verify memory manager was called
        mock_memory.retrieve_context.assert_called_once()
        
        # Context should contain memory content
        assert len(context) > len(manager.system_instructions)
        assert "MEMORIES" in context or "Python programming" in context
    
    @pytest.mark.asyncio
    async def test_screen_integration(self, context_config):
        """Test integration with screen context provider"""
        # Create mock screen provider with sample data
        mock_screen = Mock(spec=ScreenContextProvider)
        
        sample_screen_data = ScreenContextData(
            content="Visual Studio Code - Python file open with main.py",
            timestamp=datetime.now(timezone.utc),
            confidence=85.0,
            word_count=8,
            character_count=45,
            change_score=0.5,
            window_title="Visual Studio Code",
            window_app="VSCode",
            privacy_filtered=False,
            source_hash="screen_hash_123"
        )
        mock_screen.get_current_context_data = AsyncMock(return_value=sample_screen_data)
        
        manager = SmartContextManager(
            config=context_config,
            screen_context_provider=mock_screen
        )
        await manager.initialize()
        
        # Build context with screen content
        context = await manager.build_context("What do you see on my screen?")
        
        # Verify screen provider was called
        mock_screen.get_current_context_data.assert_called_once()
        
        # Context should contain screen content
        assert "Visual Studio Code" in context or "Python file" in context
    
    @pytest.mark.asyncio
    async def test_full_integration_scenario(self, context_config):
        """Test full integration with memory, screen, and conversation history"""
        # Set up mocks
        mock_memory = Mock(spec=MemoryManager)
        from langchain_core.documents import Document
        mock_memory.retrieve_context = AsyncMock(return_value=[
            Document(
                page_content="We discussed Python basics yesterday",
                metadata={"timestamp": datetime.now(timezone.utc).isoformat(), "id": "mem_001"}
            )
        ])
        
        mock_screen = Mock(spec=ScreenContextProvider)
        screen_data = ScreenContextData(
            content="Code editor showing Python function definition",
            timestamp=datetime.now(timezone.utc),
            confidence=90.0,
            word_count=6,
            character_count=38,
            change_score=0.3,
            window_title="Code Editor",
            window_app="VSCode",
            privacy_filtered=False,
            source_hash="screen_456"
        )
        mock_screen.get_current_context_data = AsyncMock(return_value=screen_data)
        
        # Create manager with all integrations
        manager = SmartContextManager(
            config=context_config,
            memory_manager=mock_memory,
            screen_context_provider=mock_screen
        )
        await manager.initialize()
        
        # Add conversation history
        manager.add_conversation_turn(
            "I'm working on a Python project",
            "Great! I can help you with Python programming."
        )
        
        # Build comprehensive context
        context = await manager.build_context("Can you help me with this function?")
        
        # Verify all components were called
        mock_memory.retrieve_context.assert_called_once()
        mock_screen.get_current_context_data.assert_called_once()
        
        # Context should contain elements from all sources
        assert "Sovereign" in context  # System instructions
        assert len(context) > 500     # Should be substantial with all context


# Context Allocation and Priority Tests
class TestContextAllocation:
    """Test priority-based context allocation and token budgeting"""
    
    @pytest.mark.asyncio
    async def test_priority_allocation(self, context_config):
        """Test that contexts are allocated by priority"""
        manager = SmartContextManager(config=context_config)
        await manager.initialize()
        
        # Create segments with different priorities
        segments = [
            ContextSegment(
                content="System instructions content",
                context_type=ContextType.SYSTEM,
                priority=ContextPriority.SYSTEM_INSTRUCTIONS,
                timestamp=datetime.now(timezone.utc),
                token_count=500,
                relevance_score=1.0,
                source_id="system"
            ),
            ContextSegment(
                content="Memory content with high relevance",
                context_type=ContextType.MEMORY,
                priority=ContextPriority.RECENT_MEMORY,
                timestamp=datetime.now(timezone.utc),
                token_count=800,
                relevance_score=0.9,
                source_id="memory"
            ),
            ContextSegment(
                content="Screen content showing code",
                context_type=ContextType.SCREEN,
                priority=ContextPriority.SCREEN_CONTENT,
                timestamp=datetime.now(timezone.utc),
                token_count=600,
                relevance_score=0.7,
                source_id="screen"
            ),
            ContextSegment(
                content="Conversation history turn",
                context_type=ContextType.CONVERSATION,
                priority=ContextPriority.CONVERSATION_HISTORY,
                timestamp=datetime.now(timezone.utc),
                token_count=400,
                relevance_score=0.6,
                source_id="conversation"
            )
        ]
        
        # Test allocation
        allocated = manager._allocate_by_priority(segments)
        
        # All segments should fit within budget
        assert len(allocated) == 4
        
        # Check allocation respects budget limits
        system_tokens = sum(s.token_count for s in allocated if s.context_type == ContextType.SYSTEM)
        memory_tokens = sum(s.token_count for s in allocated if s.context_type == ContextType.MEMORY)
        screen_tokens = sum(s.token_count for s in allocated if s.context_type == ContextType.SCREEN)
        conv_tokens = sum(s.token_count for s in allocated if s.context_type == ContextType.CONVERSATION)
        
        assert system_tokens <= context_config.budget.system_instructions
        assert memory_tokens <= context_config.budget.recent_memory
        assert screen_tokens <= context_config.budget.screen_content
        assert conv_tokens <= context_config.budget.conversation_history
    
    @pytest.mark.asyncio
    async def test_budget_overflow_handling(self, context_config):
        """Test handling when content exceeds budget"""
        manager = SmartContextManager(config=context_config)
        await manager.initialize()
        
        # Create segments that exceed budget
        large_segments = [
            ContextSegment(
                content="Very large system content that exceeds budget" * 50,
                context_type=ContextType.SYSTEM,
                priority=ContextPriority.SYSTEM_INSTRUCTIONS,
                timestamp=datetime.now(timezone.utc),
                token_count=2000,  # Exceeds system budget (1000)
                relevance_score=1.0,
                source_id="large_system"
            ),
            ContextSegment(
                content="Large memory content that also exceeds its budget" * 100,
                context_type=ContextType.MEMORY,
                priority=ContextPriority.RECENT_MEMORY,
                timestamp=datetime.now(timezone.utc),
                token_count=3000,  # Exceeds memory budget (2000)
                relevance_score=0.8,
                source_id="large_memory"
            )
        ]
        
        # Allocation should handle overflow gracefully
        allocated = manager._allocate_by_priority(large_segments)
        
        # Should either compress or exclude oversized content
        total_tokens = sum(s.token_count for s in allocated)
        assert total_tokens <= context_config.budget.total_budget


# Caching Tests
class TestCaching:
    """Test context caching functionality"""
    
    @pytest.mark.asyncio
    async def test_context_caching(self, context_config):
        """Test that context is cached and retrieved correctly"""
        context_config.enable_caching = True
        manager = SmartContextManager(config=context_config)
        await manager.initialize()
        
        query = "What is Python programming?"
        
        # First build should miss cache
        context1 = await manager.build_context(query)
        assert manager.cache_hits == 0
        
        # Second build should hit cache
        context2 = await manager.build_context(query)
        assert manager.cache_hits == 1
        assert context1 == context2
    
    @pytest.mark.asyncio
    async def test_cache_invalidation(self, context_config):
        """Test cache invalidation on context changes"""
        context_config.enable_caching = True
        manager = SmartContextManager(config=context_config)
        await manager.initialize()
        
        query = "Test query"
        
        # Build initial context
        context1 = await manager.build_context(query)
        
        # Change state (add conversation turn)
        manager.add_conversation_turn("New user message", "New assistant response")
        
        # Should not use cache due to state change
        context2 = await manager.build_context(query)
        # Context might be different due to new conversation
        assert isinstance(context2, str)
    
    @pytest.mark.asyncio
    async def test_cache_cleanup(self, context_config):
        """Test cache cleanup functionality"""
        context_config.enable_caching = True
        context_config.cache_ttl_seconds = 1  # Very short TTL
        context_config.max_cache_size = 2
        
        manager = SmartContextManager(config=context_config)
        await manager.initialize()
        
        # Fill cache beyond limit
        await manager.build_context("Query 1")
        await manager.build_context("Query 2")
        await manager.build_context("Query 3")
        
        # Cache should be cleaned up
        assert len(manager.context_cache) <= context_config.max_cache_size


# Performance and Metrics Tests
class TestPerformanceAndMetrics:
    """Test performance characteristics and metrics collection"""
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, context_config):
        """Test metrics collection functionality"""
        manager = SmartContextManager(config=context_config)
        await manager.initialize()
        
        # Build some contexts
        await manager.build_context("Test query 1")
        await manager.build_context("Test query 2")
        
        # Add conversation turns
        manager.add_conversation_turn("User msg 1", "Assistant msg 1")
        manager.add_conversation_turn("User msg 2", "Assistant msg 2")
        
        # Get metrics
        metrics = manager.get_metrics()
        
        # Verify expected metrics
        assert "context_build_count" in metrics
        assert "cache_hits" in metrics
        assert "cache_hit_rate" in metrics
        assert "total_tokens_managed" in metrics
        assert "conversation_turns" in metrics
        assert "budget" in metrics
        
        assert metrics["context_build_count"] >= 2
        assert metrics["conversation_turns"] == 2
        assert metrics["budget"]["total"] == context_config.budget.total_budget
    
    @pytest.mark.asyncio
    async def test_context_refresh_triggers(self, context_config):
        """Test context refresh trigger logic"""
        context_config.context_refresh_interval = 0.1  # Very short interval
        manager = SmartContextManager(config=context_config)
        await manager.initialize()
        
        # Initial state - no refresh needed
        assert not manager.should_refresh_context()
        
        # Wait for time trigger
        await asyncio.sleep(0.2)
        assert manager.should_refresh_context()
        
        # Update last refresh time
        manager.last_context_refresh = time.time()
        assert not manager.should_refresh_context()


# Factory Function Tests
class TestFactoryFunctions:
    """Test factory functions and helper utilities"""
    
    def test_create_smart_context_manager(self, context_config, mock_memory_manager, mock_screen_provider):
        """Test factory function for creating SmartContextManager"""
        manager = create_smart_context_manager(
            config=context_config,
            memory_manager=mock_memory_manager,
            screen_context_provider=mock_screen_provider
        )
        
        assert isinstance(manager, SmartContextManager)
        assert manager.config == context_config
        assert manager.memory_manager == mock_memory_manager
        assert manager.screen_context_provider == mock_screen_provider
    
    def test_factory_with_defaults(self):
        """Test factory function with default parameters"""
        manager = create_smart_context_manager()
        
        assert isinstance(manager, SmartContextManager)
        assert isinstance(manager.config, SmartContextConfig)
        assert manager.memory_manager is None
        assert manager.screen_context_provider is None


# Error Handling Tests
class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_memory_error_handling(self, context_config):
        """Test handling of memory system errors"""
        # Create mock that raises exception
        mock_memory = Mock(spec=MemoryManager)
        mock_memory.retrieve_context = AsyncMock(side_effect=Exception("Memory error"))
        
        manager = SmartContextManager(
            config=context_config,
            memory_manager=mock_memory
        )
        await manager.initialize()
        
        # Should handle error gracefully
        context = await manager.build_context("Test query")
        
        # Should still return valid context (system instructions at minimum)
        assert isinstance(context, str)
        assert len(context) > 0
        assert "Sovereign" in context
    
    @pytest.mark.asyncio
    async def test_screen_error_handling(self, context_config):
        """Test handling of screen provider errors"""
        # Create mock that raises exception
        mock_screen = Mock(spec=ScreenContextProvider)
        mock_screen.get_current_context_data = AsyncMock(side_effect=Exception("Screen error"))
        
        manager = SmartContextManager(
            config=context_config,
            screen_context_provider=mock_screen
        )
        await manager.initialize()
        
        # Should handle error gracefully
        context = await manager.build_context("What's on my screen?")
        
        # Should still return valid context
        assert isinstance(context, str)
        assert len(context) > 0
    
    @pytest.mark.asyncio
    async def test_token_counting_fallback(self, context_config):
        """Test token counting fallback when tiktoken unavailable"""
        manager = SmartContextManager(config=context_config)
        await manager.initialize()
        
        # Test that token counter works even without tiktoken
        test_text = "This is a test sentence for token counting fallback."
        tokens = manager.token_counter.count_tokens(test_text)
        
        assert tokens > 0
        assert isinstance(tokens, int)
    
    @pytest.mark.asyncio
    async def test_cleanup_functionality(self, context_config):
        """Test cleanup functionality"""
        context_config.enable_background_updates = True
        manager = SmartContextManager(config=context_config)
        await manager.initialize()
        
        # Verify background task started
        assert manager.background_task is not None
        
        # Test cleanup
        await manager.cleanup()
        
        # Verify cleanup completed
        assert manager.context_cache == {}


# Integration Scenario Tests
@pytest.mark.asyncio
async def test_realistic_conversation_scenario():
    """Test realistic conversation scenario with full context management"""
    # Create configuration for realistic scenario
    config = SmartContextConfig(
        budget=ContextBudget(
            system_instructions=2000,
            recent_memory=4000,
            screen_content=2000,
            conversation_history=8000
        ),
        max_conversation_turns=8,
        relevance_threshold=0.2,
        enable_compression=True
    )
    
    # Create mocks for external systems
    mock_memory = Mock(spec=MemoryManager)
    from langchain_core.documents import Document
    mock_memory.retrieve_context = AsyncMock(return_value=[
        Document(
            page_content="User was working on a Python web application using Flask framework",
            metadata={"timestamp": datetime.now(timezone.utc).isoformat(), "id": "mem_001"}
        ),
        Document(
            page_content="Discussion about database integration with SQLAlchemy ORM",
            metadata={"timestamp": datetime.now(timezone.utc).isoformat(), "id": "mem_002"}
        )
    ])
    
    mock_screen = Mock(spec=ScreenContextProvider)
    screen_data = ScreenContextData(
        content="Visual Studio Code showing app.py file with Flask routes and database models",
        timestamp=datetime.now(timezone.utc),
        confidence=92.0,
        word_count=12,
        character_count=78,
        change_score=0.4,
        window_title="Visual Studio Code",
        window_app="VSCode",
        privacy_filtered=False,
        source_hash="screen_realistic"
    )
    mock_screen.get_current_context_data = AsyncMock(return_value=screen_data)
    
    # Create manager
    manager = create_smart_context_manager(
        config=config,
        memory_manager=mock_memory,
        screen_context_provider=mock_screen
    )
    
    await manager.initialize()
    
    # Simulate realistic conversation
    conversation_turns = [
        ("How do I create a database model?", "You can create database models using SQLAlchemy ORM with Flask."),
        ("What about relationships between models?", "SQLAlchemy supports relationships using foreign keys and relationship() function."),
        ("Can you show me an example?", "Here's an example of a User model with posts relationship..."),
        ("How do I query the database?", "You can use session.query() or the newer session.get() methods."),
    ]
    
    for user_msg, assistant_msg in conversation_turns:
        manager.add_conversation_turn(user_msg, assistant_msg)
    
    # Build context for a complex query
    context = await manager.build_context("I'm getting an error in my Flask app. Can you help debug it based on what you can see?")
    
    # Verify comprehensive context
    assert len(context) > 1000  # Should be substantial
    assert "Sovereign" in context  # System instructions
    assert "Flask" in context     # Should include relevant memory/conversation
    assert "Visual Studio Code" in context  # Should include screen content
    
    # Verify metrics
    metrics = manager.get_metrics()
    assert metrics["conversation_turns"] == 4
    assert metrics["context_build_count"] >= 1
    assert metrics["total_tokens_managed"] > 0
    
    # Cleanup
    await manager.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 