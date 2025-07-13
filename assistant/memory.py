"""
Sovereign Voice Assistant - Memory System

Implements long-term memory using ChromaDB for storing and retrieving:
- Conversation history (user queries, assistant responses)
- Screen content (OCR data, context)
- Memory-enhanced context injection for LLM calls

Uses LangChain integration with OpenAI embeddings for semantic search.
"""

import os
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun


@dataclass
class MemoryConfig:
    """Configuration for the memory system"""
    
    # ChromaDB settings
    persist_directory: str = "./data/chroma"
    collection_name_conversations: str = "conversation_history"
    collection_name_screen: str = "screen_content"
    
    # Embedding settings
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 100
    
    # Retrieval settings
    retrieval_k: int = 5
    similarity_threshold: float = 0.7
    max_context_length: int = 8000
    
    # Memory management
    max_conversations_per_session: int = 100
    cleanup_interval_hours: int = 24


@dataclass
class ConversationMemory:
    """Represents a stored conversation entry"""
    
    user_query: str
    assistant_response: str
    timestamp: datetime
    session_id: str
    metadata: Dict[str, Any]


@dataclass
class ScreenMemory:
    """Represents stored screen content"""
    
    content: str
    timestamp: datetime
    source: str  # e.g., "screenshot", "ocr", "clipboard"
    metadata: Dict[str, Any]


class MemoryRetriever(BaseRetriever):
    """Custom LangChain retriever for memory-enhanced context"""
    
    def __init__(self, memory_manager: 'MemoryManager'):
        super().__init__()
        self._memory_manager = memory_manager
        
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve relevant memories for the query"""
        return asyncio.run(self._memory_manager.retrieve_context(query))
    
    @property
    def memory_manager(self) -> 'MemoryManager':
        """Get the memory manager"""
        return self._memory_manager


class MemoryManager:
    """
    Core memory management system using ChromaDB and LangChain
    
    Handles:
    - Conversation history storage and retrieval
    - Screen content memory
    - Context injection for LLM calls
    - Similarity search and ranking
    """
    
    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        self.logger = logging.getLogger(__name__)
        
        # ChromaDB components
        self.chroma_client: Optional[chromadb.Client] = None
        self.conversation_collection = None
        self.screen_collection = None
        
        # LangChain components
        self.embeddings: Optional[OpenAIEmbeddings] = None
        self.conversation_vectorstore: Optional[Chroma] = None
        self.screen_vectorstore: Optional[Chroma] = None
        self.retriever: Optional[MemoryRetriever] = None
        
        # Session tracking
        self.current_session_id = self._generate_session_id()
        self.conversation_count = 0
        
        # Initialize flag
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the memory system"""
        try:
            self.logger.info("🧠 Initializing Memory System...")
            
            # Create data directory
            os.makedirs(self.config.persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client
            await self._initialize_chromadb()
            
            # Initialize LangChain components
            await self._initialize_langchain()
            
            # Create retriever
            self.retriever = MemoryRetriever(self)
            
            self._initialized = True
            self.logger.info(f"✅ Memory System initialized with session: {self.current_session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize memory system: {e}")
            return False
    
    async def _initialize_chromadb(self):
        """Initialize ChromaDB client and collections"""
        
        # Create persistent ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=self.config.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get conversation collection
        self.conversation_collection = self.chroma_client.get_or_create_collection(
            name=self.config.collection_name_conversations,
            metadata={"description": "User conversations and assistant responses"}
        )
        
        # Create or get screen content collection
        self.screen_collection = self.chroma_client.get_or_create_collection(
            name=self.config.collection_name_screen,
            metadata={"description": "Screen content, OCR data, and visual context"}
        )
        
        self.logger.info(f"📚 ChromaDB collections initialized: "
                        f"conversations={self.conversation_collection.count()}, "
                        f"screen={self.screen_collection.count()}")
    
    async def _initialize_langchain(self):
        """Initialize LangChain embeddings and vector stores"""
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.config.embedding_model,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create LangChain Chroma vector stores
        self.conversation_vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=self.config.collection_name_conversations,
            embedding_function=self.embeddings
        )
        
        self.screen_vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=self.config.collection_name_screen,
            embedding_function=self.embeddings
        )
        
        self.logger.info("🔗 LangChain integration initialized")
    
    async def store_conversation(
        self,
        user_query: str,
        assistant_response: str,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Store a conversation exchange in memory"""
        
        if not self._initialized:
            self.logger.warning("Memory system not initialized, skipping storage")
            return False
        
        try:
            timestamp = datetime.now(timezone.utc)
            
            # Create conversation memory object
            conversation = ConversationMemory(
                user_query=user_query,
                assistant_response=assistant_response,
                timestamp=timestamp,
                session_id=self.current_session_id,
                metadata=metadata or {}
            )
            
            # Create document for vector storage
            combined_text = f"User: {user_query}\nAssistant: {assistant_response}"
            
            document = Document(
                page_content=combined_text,
                metadata={
                    "type": "conversation",
                    "user_query": user_query,
                    "assistant_response": assistant_response,
                    "timestamp": timestamp.isoformat(),
                    "session_id": self.current_session_id,
                    "conversation_id": f"{self.current_session_id}_{self.conversation_count}",
                    **conversation.metadata
                }
            )
            
            # Add to vector store
            await asyncio.to_thread(
                self.conversation_vectorstore.add_documents,
                [document]
            )
            
            self.conversation_count += 1
            
            self.logger.debug(f"💾 Stored conversation: {user_query[:50]}...")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store conversation: {e}")
            return False
    
    async def store_screen_content(
        self,
        content: str,
        source: str = "screenshot",
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Store screen content in memory"""
        
        if not self._initialized:
            return False
        
        try:
            timestamp = datetime.now(timezone.utc)
            
            # Create screen memory object
            screen_memory = ScreenMemory(
                content=content,
                timestamp=timestamp,
                source=source,
                metadata=metadata or {}
            )
            
            # Create document for vector storage
            document = Document(
                page_content=content,
                metadata={
                    "type": "screen_content",
                    "source": source,
                    "timestamp": timestamp.isoformat(),
                    "session_id": self.current_session_id,
                    **screen_memory.metadata
                }
            )
            
            # Add to vector store
            await asyncio.to_thread(
                self.screen_vectorstore.add_documents,
                [document]
            )
            
            self.logger.debug(f"🖥️ Stored screen content from {source}: {content[:50]}...")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store screen content: {e}")
            return False
    
    async def retrieve_context(
        self,
        query: str,
        include_conversations: bool = True,
        include_screen: bool = True,
        max_results: int = None
    ) -> List[Document]:
        """Retrieve relevant context for a query using similarity search"""
        
        if not self._initialized:
            return []
        
        try:
            max_results = max_results or self.config.retrieval_k
            retrieved_docs = []
            
            # Search conversation history
            if include_conversations and self.conversation_vectorstore:
                conv_docs = await asyncio.to_thread(
                    self.conversation_vectorstore.similarity_search_with_score,
                    query,
                    k=max_results // 2 if include_screen else max_results
                )
                
                # Filter by similarity threshold
                filtered_conv_docs = [
                    doc for doc, score in conv_docs 
                    if score >= self.config.similarity_threshold
                ]
                retrieved_docs.extend(filtered_conv_docs)
            
            # Search screen content
            if include_screen and self.screen_vectorstore:
                screen_docs = await asyncio.to_thread(
                    self.screen_vectorstore.similarity_search_with_score,
                    query,
                    k=max_results // 2 if include_conversations else max_results
                )
                
                # Filter by similarity threshold
                filtered_screen_docs = [
                    doc for doc, score in screen_docs 
                    if score >= self.config.similarity_threshold
                ]
                retrieved_docs.extend(filtered_screen_docs)
            
            # Sort by relevance and limit results
            retrieved_docs = retrieved_docs[:max_results]
            
            self.logger.debug(f"🔍 Retrieved {len(retrieved_docs)} relevant memories for: {query[:50]}...")
            return retrieved_docs
            
        except Exception as e:
            self.logger.error(f"❌ Failed to retrieve context: {e}")
            return []
    
    async def inject_context(
        self,
        current_query: str,
        max_context_length: int = None
    ) -> str:
        """Generate memory-enhanced context for LLM calls"""
        
        max_length = max_context_length or self.config.max_context_length
        
        try:
            # Retrieve relevant memories
            relevant_docs = await self.retrieve_context(current_query)
            
            if not relevant_docs:
                return ""
            
            # Build context string
            context_parts = []
            current_length = 0
            
            context_parts.append("=== RELEVANT MEMORY CONTEXT ===")
            
            for doc in relevant_docs:
                doc_type = doc.metadata.get("type", "unknown")
                timestamp = doc.metadata.get("timestamp", "")
                
                if doc_type == "conversation":
                    context_text = f"\n[Past Conversation - {timestamp}]\n{doc.page_content}"
                elif doc_type == "screen_content":
                    source = doc.metadata.get("source", "unknown")
                    # Provide more detailed screen context and longer content
                    screen_content = doc.page_content[:1500]  # Increased from 500 to 1500 chars
                    context_text = f"\n[CURRENT SCREEN CONTENT - {source} - {timestamp}]\nVISIBLE ON USER'S SCREEN:\n{screen_content}\n[END SCREEN CONTENT]"
                else:
                    context_text = f"\n[Memory - {timestamp}]\n{doc.page_content}"
                
                # Check if adding this context would exceed max length
                if current_length + len(context_text) > max_length:
                    context_parts.append("\n[Additional context truncated...]")
                    break
                
                context_parts.append(context_text)
                current_length += len(context_text)
            
            context_parts.append("\n=== END MEMORY CONTEXT ===\n")
            
            context = "".join(context_parts)
            
            self.logger.debug(f"💭 Injected {len(relevant_docs)} memories ({len(context)} chars)")
            return context
            
        except Exception as e:
            self.logger.error(f"❌ Failed to inject context: {e}")
            return ""
    
    async def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about the current session"""
        
        if not self._initialized:
            return {}
        
        try:
            conv_count = self.conversation_collection.count() if self.conversation_collection else 0
            screen_count = self.screen_collection.count() if self.screen_collection else 0
            
            return {
                "session_id": self.current_session_id,
                "conversations_stored": self.conversation_count,
                "total_conversations": conv_count,
                "total_screen_content": screen_count,
                "memory_initialized": self._initialized
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get session stats: {e}")
            return {}
    
    async def cleanup(self):
        """Clean up memory system resources"""
        try:
            if self.chroma_client:
                # ChromaDB client cleanup happens automatically
                pass
            
            self._initialized = False
            self.logger.info("🧠 Memory system cleaned up")
            
        except Exception as e:
            self.logger.error(f"❌ Error during memory cleanup: {e}")
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"session_{timestamp}"


# Factory functions
def create_memory_manager(config: MemoryConfig = None) -> MemoryManager:
    """Create and return a memory manager instance"""
    return MemoryManager(config)


def get_default_memory_config() -> MemoryConfig:
    """Get default memory configuration"""
    return MemoryConfig()


# Export main classes
__all__ = [
    "MemoryManager",
    "MemoryConfig", 
    "ConversationMemory",
    "ScreenMemory",
    "MemoryRetriever",
    "create_memory_manager",
    "get_default_memory_config"
] 