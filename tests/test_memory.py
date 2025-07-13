"""
Tests for the Memory System Module

Tests cover:
- MemoryManager initialization and configuration
- Conversation storage and retrieval
- Screen content storage
- Context injection and similarity search
- LangChain integration
- ChromaDB operations
"""

import pytest
import asyncio
import tempfile
import shutil
import os
from datetime import datetime, timezone
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock

from assistant.memory import (
    MemoryManager,
    MemoryConfig,
    ConversationMemory,
    ScreenMemory,
    MemoryRetriever,
    create_memory_manager,
    get_default_memory_config
)


class TestMemoryConfig:
    """Test memory configuration"""
    
    def test_default_config(self):
        """Test default memory configuration"""
        config = MemoryConfig()
        
        assert config.persist_directory == "./data/chroma"
        assert config.collection_name_conversations == "conversation_history"
        assert config.collection_name_screen == "screen_content"
        assert config.embedding_model == "text-embedding-3-small"
        assert config.retrieval_k == 5
        assert config.similarity_threshold == 0.7
        assert config.max_context_length == 8000
    
    def test_custom_config(self):
        """Test custom memory configuration"""
        config = MemoryConfig(
            persist_directory="/tmp/test_chroma",
            retrieval_k=10,
            similarity_threshold=0.8
        )
        
        assert config.persist_directory == "/tmp/test_chroma"
        assert config.retrieval_k == 10
        assert config.similarity_threshold == 0.8


class TestMemoryDataClasses:
    """Test memory data classes"""
    
    def test_conversation_memory(self):
        """Test ConversationMemory data class"""
        now = datetime.now(timezone.utc)
        memory = ConversationMemory(
            user_query="Hello",
            assistant_response="Hi there!",
            timestamp=now,
            session_id="test_session",
            metadata={"test": "value"}
        )
        
        assert memory.user_query == "Hello"
        assert memory.assistant_response == "Hi there!"
        assert memory.timestamp == now
        assert memory.session_id == "test_session"
        assert memory.metadata == {"test": "value"}
    
    def test_screen_memory(self):
        """Test ScreenMemory data class"""
        now = datetime.now(timezone.utc)
        memory = ScreenMemory(
            content="Test screen content",
            timestamp=now,
            source="screenshot",
            metadata={"test": "value"}
        )
        
        assert memory.content == "Test screen content"
        assert memory.timestamp == now
        assert memory.source == "screenshot"
        assert memory.metadata == {"test": "value"}


class TestMemoryManager:
    """Test MemoryManager functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def memory_config(self, temp_dir):
        """Create test memory configuration"""
        return MemoryConfig(
            persist_directory=temp_dir,
            retrieval_k=3,
            similarity_threshold=0.5
        )
    
    @pytest.fixture
    def memory_manager(self, memory_config):
        """Create test memory manager"""
        return MemoryManager(memory_config)
    
    def test_memory_manager_creation(self, memory_manager):
        """Test memory manager creation"""
        assert memory_manager is not None
        assert memory_manager._initialized is False
        assert memory_manager.current_session_id.startswith("session_")
        assert memory_manager.conversation_count == 0
    
    @pytest.mark.asyncio
    async def test_memory_manager_initialization_no_api_key(self, memory_manager):
        """Test memory manager initialization without API key"""
        # Should fail without OpenAI API key
        result = await memory_manager.initialize()
        assert result is False
        assert memory_manager._initialized is False
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('assistant.memory.OpenAIEmbeddings')
    @patch('assistant.memory.chromadb.PersistentClient')
    async def test_memory_manager_initialization_success(self, mock_client, mock_embeddings, memory_manager):
        """Test successful memory manager initialization"""
        # Mock ChromaDB client
        mock_collection = Mock()
        mock_collection.count.return_value = 0
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        # Mock embeddings
        mock_embeddings.return_value = Mock()
        
        result = await memory_manager.initialize()
        assert result is True
        assert memory_manager._initialized is True
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('assistant.memory.OpenAIEmbeddings')
    @patch('assistant.memory.chromadb.PersistentClient')
    @patch('assistant.memory.Chroma')
    async def test_store_conversation(self, mock_chroma, mock_client, mock_embeddings, memory_manager):
        """Test conversation storage"""
        # Setup mocks
        mock_collection = Mock()
        mock_collection.count.return_value = 0
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        mock_vectorstore = Mock()
        mock_vectorstore.add_documents = AsyncMock()
        mock_chroma.return_value = mock_vectorstore
        
        # Initialize memory manager
        await memory_manager.initialize()
        
        # Store conversation
        result = await memory_manager.store_conversation(
            user_query="Hello, how are you?",
            assistant_response="I'm doing well, thank you!",
            metadata={"test": "value"}
        )
        
        assert result is True
        assert memory_manager.conversation_count == 1
        
        # Verify add_documents was called
        mock_vectorstore.add_documents.assert_called_once()
        
        # Check the document that was added
        call_args = mock_vectorstore.add_documents.call_args
        documents = call_args[0][0]
        assert len(documents) == 1
        
        doc = documents[0]
        assert "User: Hello, how are you?" in doc.page_content
        assert "Assistant: I'm doing well, thank you!" in doc.page_content
        assert doc.metadata["type"] == "conversation"
        assert doc.metadata["user_query"] == "Hello, how are you?"
        assert doc.metadata["assistant_response"] == "I'm doing well, thank you!"
        assert doc.metadata["test"] == "value"
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('assistant.memory.OpenAIEmbeddings')
    @patch('assistant.memory.chromadb.PersistentClient')
    @patch('assistant.memory.Chroma')
    async def test_store_screen_content(self, mock_chroma, mock_client, mock_embeddings, memory_manager):
        """Test screen content storage"""
        # Setup mocks
        mock_collection = Mock()
        mock_collection.count.return_value = 0
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        mock_vectorstore = Mock()
        mock_vectorstore.add_documents = AsyncMock()
        mock_chroma.return_value = mock_vectorstore
        
        # Initialize memory manager
        await memory_manager.initialize()
        
        # Store screen content
        result = await memory_manager.store_screen_content(
            content="This is test screen content",
            source="screenshot",
            metadata={"resolution": "1920x1080"}
        )
        
        assert result is True
        
        # Verify add_documents was called
        mock_vectorstore.add_documents.assert_called_once()
        
        # Check the document that was added
        call_args = mock_vectorstore.add_documents.call_args
        documents = call_args[0][0]
        assert len(documents) == 1
        
        doc = documents[0]
        assert doc.page_content == "This is test screen content"
        assert doc.metadata["type"] == "screen_content"
        assert doc.metadata["source"] == "screenshot"
        assert doc.metadata["resolution"] == "1920x1080"
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('assistant.memory.OpenAIEmbeddings')
    @patch('assistant.memory.chromadb.PersistentClient')
    @patch('assistant.memory.Chroma')
    async def test_retrieve_context(self, mock_chroma, mock_client, mock_embeddings, memory_manager):
        """Test context retrieval"""
        # Setup mocks
        mock_collection = Mock()
        mock_collection.count.return_value = 5
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        # Mock documents to return
        from langchain_core.documents import Document
        
        mock_conv_doc = Document(
            page_content="User: Hello\nAssistant: Hi there!",
            metadata={"type": "conversation", "timestamp": "2024-01-01T10:00:00"}
        )
        mock_screen_doc = Document(
            page_content="Screen shows weather app",
            metadata={"type": "screen_content", "source": "screenshot"}
        )
        
        mock_conv_vectorstore = Mock()
        mock_conv_vectorstore.similarity_search_with_score = AsyncMock(
            return_value=[(mock_conv_doc, 0.8)]
        )
        
        mock_screen_vectorstore = Mock()
        mock_screen_vectorstore.similarity_search_with_score = AsyncMock(
            return_value=[(mock_screen_doc, 0.7)]
        )
        
        # Mock the vectorstore creation to return different stores
        def mock_chroma_side_effect(*args, **kwargs):
            collection_name = kwargs.get('collection_name', '')
            if 'conversation' in collection_name:
                return mock_conv_vectorstore
            else:
                return mock_screen_vectorstore
        
        mock_chroma.side_effect = mock_chroma_side_effect
        
        # Initialize memory manager
        await memory_manager.initialize()
        
        # Retrieve context
        docs = await memory_manager.retrieve_context("Hello")
        
        assert len(docs) == 2
        
        # Check that both conversation and screen content were retrieved
        doc_types = [doc.metadata.get("type") for doc in docs]
        assert "conversation" in doc_types
        assert "screen_content" in doc_types
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('assistant.memory.OpenAIEmbeddings')
    @patch('assistant.memory.chromadb.PersistentClient')
    @patch('assistant.memory.Chroma')
    async def test_inject_context(self, mock_chroma, mock_client, mock_embeddings, memory_manager):
        """Test context injection"""
        # Setup mocks similar to test_retrieve_context
        mock_collection = Mock()
        mock_collection.count.return_value = 2
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        from langchain_core.documents import Document
        
        mock_doc = Document(
            page_content="User: What's the weather?\nAssistant: It's sunny today!",
            metadata={"type": "conversation", "timestamp": "2024-01-01T10:00:00"}
        )
        
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search_with_score = AsyncMock(
            return_value=[(mock_doc, 0.8)]
        )
        mock_chroma.return_value = mock_vectorstore
        
        # Initialize memory manager
        await memory_manager.initialize()
        
        # Inject context
        context = await memory_manager.inject_context("How's the weather?")
        
        assert "RELEVANT MEMORY CONTEXT" in context
        assert "Past Conversation" in context
        assert "What's the weather?" in context
        assert "It's sunny today!" in context
        assert "END MEMORY CONTEXT" in context
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('assistant.memory.OpenAIEmbeddings')
    @patch('assistant.memory.chromadb.PersistentClient')
    async def test_get_session_stats(self, mock_client, mock_embeddings, memory_manager):
        """Test session statistics"""
        # Setup mocks
        mock_collection = Mock()
        mock_collection.count.return_value = 10
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        # Initialize memory manager
        await memory_manager.initialize()
        
        # Get session stats
        stats = await memory_manager.get_session_stats()
        
        assert "session_id" in stats
        assert stats["conversations_stored"] == 0
        assert stats["total_conversations"] == 10
        assert stats["total_screen_content"] == 10
        assert stats["memory_initialized"] is True
    
    def test_memory_manager_not_initialized(self, memory_manager):
        """Test operations on non-initialized memory manager"""
        # These should fail gracefully when not initialized
        assert memory_manager._initialized is False
    
    @pytest.mark.asyncio
    async def test_store_conversation_not_initialized(self, memory_manager):
        """Test storing conversation when not initialized"""
        result = await memory_manager.store_conversation("test", "test")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_retrieve_context_not_initialized(self, memory_manager):
        """Test retrieving context when not initialized"""
        docs = await memory_manager.retrieve_context("test")
        assert docs == []
    
    @pytest.mark.asyncio
    async def test_cleanup(self, memory_manager):
        """Test memory manager cleanup"""
        # Set as initialized to test cleanup
        memory_manager._initialized = True
        
        await memory_manager.cleanup()
        assert memory_manager._initialized is False


class TestMemoryRetriever:
    """Test MemoryRetriever functionality"""
    
    def test_memory_retriever_creation(self):
        """Test memory retriever creation"""
        memory_manager = Mock()
        retriever = MemoryRetriever(memory_manager)
        
        assert retriever.memory_manager == memory_manager
    
    @patch('asyncio.run')
    def test_get_relevant_documents(self, mock_run):
        """Test getting relevant documents"""
        memory_manager = Mock()
        memory_manager.retrieve_context = Mock(return_value=[])
        
        retriever = MemoryRetriever(memory_manager)
        
        # Mock the run_manager parameter
        run_manager = Mock()
        
        retriever._get_relevant_documents("test query", run_manager=run_manager)
        
        mock_run.assert_called_once()


class TestFactoryFunctions:
    """Test factory functions"""
    
    def test_create_memory_manager(self):
        """Test create_memory_manager factory function"""
        manager = create_memory_manager()
        assert isinstance(manager, MemoryManager)
        assert isinstance(manager.config, MemoryConfig)
    
    def test_create_memory_manager_with_config(self):
        """Test create_memory_manager with custom config"""
        config = MemoryConfig(retrieval_k=10)
        manager = create_memory_manager(config)
        assert manager.config.retrieval_k == 10
    
    def test_get_default_memory_config(self):
        """Test get_default_memory_config function"""
        config = get_default_memory_config()
        assert isinstance(config, MemoryConfig)
        assert config.retrieval_k == 5


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('assistant.memory.OpenAIEmbeddings')
    @patch('assistant.memory.chromadb.PersistentClient')
    @patch('assistant.memory.Chroma')
    async def test_conversation_flow_with_memory(self, mock_chroma, mock_client, mock_embeddings, temp_dir):
        """Test a complete conversation flow with memory"""
        # Create memory manager
        config = MemoryConfig(persist_directory=temp_dir)
        memory_manager = MemoryManager(config)
        
        # Setup mocks
        mock_collection = Mock()
        mock_collection.count.return_value = 0
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        mock_vectorstore = Mock()
        mock_vectorstore.add_documents = AsyncMock()
        mock_vectorstore.similarity_search_with_score = AsyncMock(return_value=[])
        mock_chroma.return_value = mock_vectorstore
        
        # Initialize
        await memory_manager.initialize()
        
        # Simulate a conversation
        user_queries = [
            "What's the weather like?",
            "Do I need an umbrella?",
            "What about tomorrow?"
        ]
        
        assistant_responses = [
            "It's sunny today with a high of 75°F.",
            "No, you don't need an umbrella today.",
            "Tomorrow looks cloudy with possible rain."
        ]
        
        # Store conversations
        for i, (query, response) in enumerate(zip(user_queries, assistant_responses)):
            result = await memory_manager.store_conversation(
                user_query=query,
                assistant_response=response,
                metadata={"conversation_turn": i}
            )
            assert result is True
        
        # Check conversation count
        assert memory_manager.conversation_count == 3
        
        # Verify all conversations were stored
        assert mock_vectorstore.add_documents.call_count == 3
        
        # Test context injection (would normally find relevant memories)
        context = await memory_manager.inject_context("What did we discuss about weather?")
        # Since we mocked to return no documents, context should be empty
        assert context == ""
        
        # Cleanup
        await memory_manager.cleanup()
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('assistant.memory.OpenAIEmbeddings')
    @patch('assistant.memory.chromadb.PersistentClient')
    @patch('assistant.memory.Chroma')
    async def test_memory_with_screen_content(self, mock_chroma, mock_client, mock_embeddings, temp_dir):
        """Test memory with both conversation and screen content"""
        # Create memory manager
        config = MemoryConfig(persist_directory=temp_dir)
        memory_manager = MemoryManager(config)
        
        # Setup mocks
        mock_collection = Mock()
        mock_collection.count.return_value = 0
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        mock_vectorstore = Mock()
        mock_vectorstore.add_documents = AsyncMock()
        mock_chroma.return_value = mock_vectorstore
        
        # Initialize
        await memory_manager.initialize()
        
        # Store conversation
        await memory_manager.store_conversation(
            user_query="Can you see what's on my screen?",
            assistant_response="I can see you have a weather app open."
        )
        
        # Store screen content
        await memory_manager.store_screen_content(
            content="Weather App - Current: 72°F, Sunny",
            source="screenshot"
        )
        
        # Verify both were stored
        assert mock_vectorstore.add_documents.call_count == 2
        
        # Cleanup
        await memory_manager.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 