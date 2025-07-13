"""
Unit tests for MemoryContextProvider

Tests the specialized context provider for OpenAI Realtime API integration,
including semantic search, token management, caching, and performance optimization.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import List, Dict, Any
import hashlib

# Mock tiktoken if not available
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

from langchain_core.documents import Document

from assistant.memory_context_provider import (
    MemoryContextProvider, ContextProviderConfig, ContextResult, 
    ContextEntry, ContextCache, TokenCounter, create_memory_context_provider
)
from assistant.memory import MemoryManager, MemoryConfig
from assistant.config_manager import RealtimeAPIConfig


class TestContextEntry:
    """Test ContextEntry dataclass"""
    
    def test_context_entry_creation(self):
        """Test basic ContextEntry creation"""
        timestamp = datetime.now()
        entry = ContextEntry(
            content="Test content",
            relevance_score=0.85,
            timestamp=timestamp,
            entry_type="conversation",
            token_count=10,
            metadata={"source": "test"}
        )
        
        assert entry.content == "Test content"
        assert entry.relevance_score == 0.85
        assert entry.timestamp == timestamp
        assert entry.entry_type == "conversation"
        assert entry.token_count == 10
        assert entry.metadata["source"] == "test"


class TestContextResult:
    """Test ContextResult dataclass"""
    
    def test_context_result_creation(self):
        """Test basic ContextResult creation"""
        result = ContextResult(
            formatted_context="Formatted context",
            total_tokens=100,
            entries_included=3,
            entries_filtered=1,
            relevance_threshold_used=0.7,
            cache_hit=True,
            processing_time_ms=25.5
        )
        
        assert result.formatted_context == "Formatted context"
        assert result.total_tokens == 100
        assert result.entries_included == 3
        assert result.entries_filtered == 1
        assert result.relevance_threshold_used == 0.7
        assert result.cache_hit is True
        assert result.processing_time_ms == 25.5
        assert len(result.context_entries) == 0  # Default empty list


class TestContextProviderConfig:
    """Test ContextProviderConfig dataclass and defaults"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ContextProviderConfig()
        
        assert config.default_k == 5
        assert config.max_k == 10
        assert config.relevance_threshold == 0.7
        assert config.min_relevance_threshold == 0.5
        assert config.max_context_tokens == 32000
        assert config.reserved_tokens == 2000
        assert config.target_context_tokens == 6000
        assert config.cache_ttl_seconds == 30
        assert config.max_cache_size == 100
        assert config.async_timeout_seconds == 5.0
        assert "Previous conversation context:" in config.context_template
        assert "You have access to" in config.system_prompt_prefix
        assert config.include_timestamps is True
        assert config.include_relevance_scores is False
        assert config.refresh_every_n_turns == 10
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = ContextProviderConfig(
            default_k=8,
            relevance_threshold=0.8,
            max_context_tokens=16000,
            target_context_tokens=4000,
            cache_ttl_seconds=60,
            include_relevance_scores=True
        )
        
        assert config.default_k == 8
        assert config.relevance_threshold == 0.8
        assert config.max_context_tokens == 16000
        assert config.target_context_tokens == 4000
        assert config.cache_ttl_seconds == 60
        assert config.include_relevance_scores is True


class TestContextCache:
    """Test ContextCache functionality"""
    
    @pytest.fixture
    def cache(self):
        """Create a test cache instance"""
        return ContextCache(max_size=3, ttl_seconds=1)
    
    @pytest.fixture
    def sample_result(self):
        """Create a sample ContextResult for testing"""
        return ContextResult(
            formatted_context="Test context",
            total_tokens=50,
            entries_included=2,
            entries_filtered=0,
            relevance_threshold_used=0.7,
            cache_hit=False,
            processing_time_ms=10.0
        )
    
    def test_cache_key_generation(self, cache):
        """Test cache key generation"""
        key1 = cache._generate_cache_key("query1", 5, 0.7)
        key2 = cache._generate_cache_key("query1", 5, 0.7)
        key3 = cache._generate_cache_key("query2", 5, 0.7)
        
        assert key1 == key2  # Same inputs = same key
        assert key1 != key3  # Different inputs = different key
        assert len(key1) == 32  # MD5 hash length
    
    @pytest.mark.asyncio
    async def test_cache_miss(self, cache):
        """Test cache miss scenario"""
        result = await cache.get("test_query", 5, 0.7)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_hit(self, cache, sample_result):
        """Test cache hit scenario"""
        # Set cache entry
        await cache.set("test_query", 5, 0.7, sample_result)
        
        # Retrieve from cache
        result = await cache.get("test_query", 5, 0.7)
        
        assert result is not None
        assert result.formatted_context == "Test context"
        assert result.cache_hit is True
    
    @pytest.mark.asyncio
    async def test_cache_expiry(self, sample_result):
        """Test cache expiry functionality"""
        cache = ContextCache(max_size=10, ttl_seconds=0.1)  # 100ms TTL
        
        # Set cache entry
        await cache.set("test_query", 5, 0.7, sample_result)
        
        # Should hit immediately
        result = await cache.get("test_query", 5, 0.7)
        assert result is not None
        
        # Wait for expiry
        await asyncio.sleep(0.2)
        
        # Should miss after expiry
        result = await cache.get("test_query", 5, 0.7)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_size_limit(self, cache, sample_result):
        """Test cache size limit enforcement"""
        # Fill cache to capacity (max_size=3)
        for i in range(3):
            await cache.set(f"query_{i}", 5, 0.7, sample_result)
        
        assert len(cache.cache) == 3
        
        # Add one more to trigger eviction
        await cache.set("query_3", 5, 0.7, sample_result)
        
        assert len(cache.cache) == 3  # Should still be 3
        
        # First entry should be evicted
        result = await cache.get("query_0", 5, 0.7)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_clear(self, cache, sample_result):
        """Test cache clearing"""
        await cache.set("test_query", 5, 0.7, sample_result)
        assert len(cache.cache) == 1
        
        await cache.clear()
        assert len(cache.cache) == 0


class TestTokenCounter:
    """Test TokenCounter functionality"""
    
    def test_token_counter_initialization(self):
        """Test TokenCounter initialization"""
        counter = TokenCounter()
        assert counter.model_name == "gpt-4o-realtime-preview-2024-10-01"
        
        custom_counter = TokenCounter("gpt-4")
        assert custom_counter.model_name == "gpt-4"
    
    def test_token_counting_without_tiktoken(self):
        """Test token counting fallback when tiktoken unavailable"""
        counter = TokenCounter()
        
        # Mock tiktoken unavailable
        with patch('assistant.memory_context_provider.HAS_TIKTOKEN', False):
            counter.encoding = None
            tokens = counter.count_tokens("This is a test string")
            
            # Should use approximation (length // 4)
            expected = len("This is a test string") // 4
            assert tokens == expected
    
    @pytest.mark.skipif(not HAS_TIKTOKEN, reason="tiktoken not available")
    def test_token_counting_with_tiktoken(self):
        """Test accurate token counting with tiktoken"""
        counter = TokenCounter()
        
        if counter.encoding:
            tokens = counter.count_tokens("Hello world!")
            assert isinstance(tokens, int)
            assert tokens > 0
    
    def test_token_estimation(self):
        """Test fast token estimation"""
        counter = TokenCounter()
        
        text = "This is a test for token estimation"
        estimated = counter.estimate_tokens(text)
        
        assert estimated == len(text) // 4
        assert isinstance(estimated, int)


class TestMemoryContextProvider:
    """Test MemoryContextProvider main functionality"""
    
    @pytest.fixture
    def mock_memory_manager(self):
        """Create a mock MemoryManager"""
        manager = AsyncMock(spec=MemoryManager)
        manager.retrieve_context = AsyncMock()
        return manager
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return ContextProviderConfig(
            default_k=3,
            target_context_tokens=1000,
            cache_ttl_seconds=1,
            async_timeout_seconds=1.0
        )
    
    @pytest.fixture
    def realtime_config(self):
        """Create test Realtime API configuration"""
        return RealtimeAPIConfig(model="gpt-4o-realtime-preview-2024-10-01")
    
    @pytest.fixture
    def provider(self, mock_memory_manager, config, realtime_config):
        """Create test MemoryContextProvider"""
        return MemoryContextProvider(
            memory_manager=mock_memory_manager,
            config=config,
            realtime_config=realtime_config
        )
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        return [
            Document(
                page_content="User asked about weather. Assistant provided forecast.",
                metadata={
                    "type": "conversation",
                    "timestamp": "2024-01-01T10:00:00",
                    "similarity": 0.85
                }
            ),
            Document(
                page_content="Screen shows calendar application with meeting at 2pm.",
                metadata={
                    "type": "screen_content", 
                    "timestamp": "2024-01-01T10:01:00",
                    "similarity": 0.75
                }
            ),
            Document(
                page_content="Previous discussion about project deadline next week.",
                metadata={
                    "type": "conversation",
                    "timestamp": "2024-01-01T09:30:00", 
                    "similarity": 0.65
                }
            )
        ]
    
    def test_provider_initialization(self, provider, mock_memory_manager, config):
        """Test MemoryContextProvider initialization"""
        assert provider.memory_manager == mock_memory_manager
        assert provider.config == config
        assert provider.token_counter is not None
        assert provider.cache is not None
        assert provider.conversation_turn_count == 0
        assert provider.total_requests == 0
        assert provider.cache_hits == 0
    
    @pytest.mark.asyncio
    async def test_get_context_for_query_success(self, provider, mock_memory_manager, sample_documents):
        """Test successful context retrieval"""
        # Mock memory manager response
        mock_memory_manager.retrieve_context.return_value = sample_documents
        
        # Get context
        result = await provider.get_context_for_query("What's on my calendar?")
        
        # Verify results
        assert isinstance(result, ContextResult)
        assert result.entries_included > 0
        assert result.total_tokens > 0
        assert result.cache_hit is False
        assert result.processing_time_ms > 0
        assert len(result.context_entries) > 0
        
        # Verify memory manager was called
        mock_memory_manager.retrieve_context.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_context_cache_hit(self, provider, mock_memory_manager, sample_documents):
        """Test cache hit functionality"""
        # Mock memory manager response
        mock_memory_manager.retrieve_context.return_value = sample_documents
        
        # First call (cache miss)
        result1 = await provider.get_context_for_query("Test query")
        assert result1.cache_hit is False
        
        # Second call (cache hit)
        result2 = await provider.get_context_for_query("Test query")
        assert result2.cache_hit is True
        
        # Memory manager should only be called once
        assert mock_memory_manager.retrieve_context.call_count == 1
    
    @pytest.mark.asyncio
    async def test_get_context_relevance_filtering(self, provider, mock_memory_manager):
        """Test relevance score filtering"""
        # Create documents with different relevance scores
        docs_with_low_relevance = [
            Document(
                page_content="High relevance content",
                metadata={"similarity": 0.85, "type": "conversation", "timestamp": "2024-01-01T10:00:00"}
            ),
            Document(
                page_content="Low relevance content", 
                metadata={"similarity": 0.3, "type": "conversation", "timestamp": "2024-01-01T10:00:00"}
            )
        ]
        
        mock_memory_manager.retrieve_context.return_value = docs_with_low_relevance
        
        result = await provider.get_context_for_query("Test query")
        
        # Only high relevance entry should be included (threshold = 0.7)
        assert result.entries_included == 1
        assert result.entries_filtered == 1
        assert "High relevance content" in result.formatted_context
        assert "Low relevance content" not in result.formatted_context
    
    @pytest.mark.asyncio
    async def test_get_context_token_limit(self, provider, mock_memory_manager):
        """Test token limit enforcement"""
        # Create documents that would exceed token limit
        large_docs = []
        for i in range(10):
            content = "This is a very long document content " * 50  # Large content
            large_docs.append(Document(
                page_content=content,
                metadata={"similarity": 0.8, "type": "conversation", "timestamp": "2024-01-01T10:00:00"}
            ))
        
        mock_memory_manager.retrieve_context.return_value = large_docs
        
        result = await provider.get_context_for_query("Test query")
        
        # Should respect token limits
        assert result.total_tokens <= provider.config.target_context_tokens
        assert result.entries_included < len(large_docs)  # Not all entries included
    
    @pytest.mark.asyncio
    async def test_get_context_timeout_handling(self, provider, mock_memory_manager):
        """Test timeout handling during context retrieval"""
        # Mock timeout
        async def slow_retrieve(*args, **kwargs):
            await asyncio.sleep(2.0)  # Longer than timeout (1.0s)
            return []
        
        mock_memory_manager.retrieve_context.side_effect = slow_retrieve
        
        result = await provider.get_context_for_query("Test query")
        
        # Should return empty context on timeout
        assert result.entries_included == 0
        assert result.formatted_context == ""
    
    @pytest.mark.asyncio
    async def test_get_context_error_handling(self, provider, mock_memory_manager):
        """Test error handling during context retrieval"""
        # Mock error
        mock_memory_manager.retrieve_context.side_effect = Exception("Test error")
        
        result = await provider.get_context_for_query("Test query")
        
        # Should return empty context on error
        assert result.entries_included == 0
        assert result.formatted_context == ""
        assert result.processing_time_ms > 0
    
    def test_context_formatting(self, provider):
        """Test context entry formatting"""
        timestamp = datetime(2024, 1, 1, 10, 0, 0)
        entry = ContextEntry(
            content="Test content",
            relevance_score=0.85,
            timestamp=timestamp,
            entry_type="conversation",
            token_count=10
        )
        
        formatted = provider._format_single_entry(entry)
        
        assert "[2024-01-01 10:00]" in formatted
        assert "(CONVERSATION)" in formatted
        assert "Test content" in formatted
    
    def test_context_formatting_with_relevance_scores(self, provider):
        """Test context formatting with relevance scores enabled"""
        provider.config.include_relevance_scores = True
        
        timestamp = datetime(2024, 1, 1, 10, 0, 0)
        entry = ContextEntry(
            content="Test content",
            relevance_score=0.85,
            timestamp=timestamp,
            entry_type="conversation",
            token_count=10
        )
        
        formatted = provider._format_single_entry(entry)
        
        assert "(relevance: 0.85)" in formatted
    
    @pytest.mark.asyncio
    async def test_refresh_context_turn_based(self, provider):
        """Test turn-based context refresh"""
        # Set refresh every 2 turns for testing
        provider.config.refresh_every_n_turns = 2
        
        # First turn - no refresh
        should_refresh = await provider.refresh_context_if_needed("Query 1")
        assert should_refresh is False
        assert provider.conversation_turn_count == 1
        
        # Second turn - should refresh
        should_refresh = await provider.refresh_context_if_needed("Query 2")
        assert should_refresh is True
        assert provider.conversation_turn_count == 2
    
    @pytest.mark.asyncio
    async def test_refresh_context_time_based(self, provider):
        """Test time-based context refresh"""
        # Set last refresh to past
        provider.last_context_refresh = time.time() - 100
        
        should_refresh = await provider.refresh_context_if_needed("Query")
        assert should_refresh is True
    
    def test_performance_stats(self, provider):
        """Test performance statistics"""
        # Simulate some activity
        provider.total_requests = 10
        provider.cache_hits = 3
        provider.avg_processing_time = 25.5
        provider.conversation_turn_count = 5
        
        stats = provider.get_performance_stats()
        
        assert stats["total_requests"] == 10
        assert stats["cache_hits"] == 3
        assert stats["cache_hit_rate_percent"] == 30.0
        assert stats["avg_processing_time_ms"] == 25.5
        assert stats["conversation_turns"] == 5
    
    @pytest.mark.asyncio
    async def test_cleanup(self, provider):
        """Test cleanup functionality"""
        # Add some data to cache
        await provider.cache.set("test", 5, 0.7, ContextResult(
            formatted_context="", total_tokens=0, entries_included=0,
            entries_filtered=0, relevance_threshold_used=0.7,
            cache_hit=False, processing_time_ms=0.0
        ))
        
        assert len(provider.cache.cache) == 1
        
        await provider.cleanup()
        
        assert len(provider.cache.cache) == 0


class TestCreateMemoryContextProvider:
    """Test factory function"""
    
    def test_create_function(self):
        """Test create_memory_context_provider factory function"""
        mock_manager = Mock(spec=MemoryManager)
        config = ContextProviderConfig()
        realtime_config = RealtimeAPIConfig()
        
        provider = create_memory_context_provider(
            memory_manager=mock_manager,
            config=config,
            realtime_config=realtime_config
        )
        
        assert isinstance(provider, MemoryContextProvider)
        assert provider.memory_manager == mock_manager
        assert provider.config == config
        assert provider.realtime_config == realtime_config
    
    def test_create_function_with_defaults(self):
        """Test factory function with default parameters"""
        mock_manager = Mock(spec=MemoryManager)
        
        provider = create_memory_context_provider(memory_manager=mock_manager)
        
        assert isinstance(provider, MemoryContextProvider)
        assert provider.memory_manager == mock_manager
        assert isinstance(provider.config, ContextProviderConfig)
        assert provider.realtime_config is None


class TestIntegrationScenarios:
    """Test integration scenarios and end-to-end functionality"""
    
    @pytest.mark.asyncio
    async def test_full_context_retrieval_workflow(self):
        """Test complete context retrieval workflow"""
        # Create real-ish components
        mock_manager = AsyncMock(spec=MemoryManager)
        config = ContextProviderConfig(default_k=2, target_context_tokens=500)
        
        # Mock realistic documents
        docs = [
            Document(
                page_content="User: What's the weather like? Assistant: It's sunny and 75°F today.",
                metadata={"type": "conversation", "timestamp": "2024-01-01T10:00:00", "similarity": 0.9}
            ),
            Document(
                page_content="Screen shows weather app with current temperature display.",
                metadata={"type": "screen_content", "timestamp": "2024-01-01T10:01:00", "similarity": 0.8}
            )
        ]
        mock_manager.retrieve_context.return_value = docs
        
        provider = MemoryContextProvider(mock_manager, config)
        
        # Test query
        result = await provider.get_context_for_query("How's the weather today?")
        
        # Verify comprehensive result
        assert result.entries_included == 2
        assert result.entries_filtered == 0
        assert result.total_tokens > 0
        assert "weather" in result.formatted_context.lower()
        assert "conversation" in result.formatted_context.upper()
        assert "screen_content" in result.formatted_context.upper()
        assert result.processing_time_ms > 0
        
        # Test caching on second call
        result2 = await provider.get_context_for_query("How's the weather today?")
        assert result2.cache_hit is True
        
        # Verify metrics
        stats = provider.get_performance_stats()
        assert stats["total_requests"] == 2
        assert stats["cache_hits"] == 1
        assert stats["cache_hit_rate_percent"] == 50.0
    
    @pytest.mark.asyncio
    async def test_large_context_truncation(self):
        """Test handling of large context sets with truncation"""
        mock_manager = AsyncMock(spec=MemoryManager)
        config = ContextProviderConfig(
            default_k=20, 
            target_context_tokens=200,  # Very small limit for testing
            relevance_threshold=0.5
        )
        
        # Create many large documents
        docs = []
        for i in range(15):
            content = f"Document {i}: " + "This is a long conversation about various topics. " * 10
            docs.append(Document(
                page_content=content,
                metadata={"type": "conversation", "timestamp": "2024-01-01T10:00:00", "similarity": 0.8}
            ))
        
        mock_manager.retrieve_context.return_value = docs
        provider = MemoryContextProvider(mock_manager, config)
        
        result = await provider.get_context_for_query("Tell me about our discussions")
        
        # Should include some but not all entries due to token limits
        assert result.entries_included < len(docs)
        assert result.total_tokens <= config.target_context_tokens
        assert result.entries_included > 0  # Should include at least one
        
        # Should have formatted context
        assert len(result.formatted_context) > 0
        assert len(result.context_entries) == result.entries_included 