"""
Memory Context Provider for OpenAI Realtime API Integration

Provides intelligent context injection for Realtime API sessions by leveraging
the existing ChromaDB memory system with optimizations for real-time performance.

Key Features:
- Semantic search using existing OpenAI embeddings
- Context formatting optimized for Realtime API system instructions
- Token counting with tiktoken for precise limit management
- Relevance scoring with cosine similarity thresholds
- Performance optimizations with caching and async operations
- Integration with existing MemoryManager infrastructure
"""

import asyncio
import logging
import time
import json
from typing import Optional, Dict, List, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import hashlib

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    logging.warning("tiktoken not available. Token counting will use approximation.")

import numpy as np
from langchain_core.documents import Document

from .memory import MemoryManager, MemoryConfig
from .config_manager import RealtimeAPIConfig


@dataclass
class ContextEntry:
    """Represents a single context entry for Realtime API"""
    content: str
    relevance_score: float
    timestamp: datetime
    entry_type: str  # "conversation", "screen_content", "memory"
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextResult:
    """Result of context retrieval and formatting"""
    formatted_context: str
    total_tokens: int
    entries_included: int
    entries_filtered: int
    relevance_threshold_used: float
    cache_hit: bool
    processing_time_ms: float
    context_entries: List[ContextEntry] = field(default_factory=list)


@dataclass 
class ContextProviderConfig:
    """Configuration for MemoryContextProvider"""
    # Retrieval settings
    default_k: int = 5
    max_k: int = 10
    relevance_threshold: float = 0.7
    min_relevance_threshold: float = 0.5
    
    # Token management
    max_context_tokens: int = 32000  # Realtime API max tokens
    reserved_tokens: int = 2000  # Reserve for response generation
    target_context_tokens: int = 6000  # Target context size
    
    # Performance settings
    cache_ttl_seconds: int = 30
    max_cache_size: int = 100
    async_timeout_seconds: float = 5.0
    
    # Context formatting
    context_template: str = "Previous conversation context:\n{context_content}\n"
    system_prompt_prefix: str = "You have access to previous conversation history and screen content. "
    include_timestamps: bool = True
    include_relevance_scores: bool = False
    
    # Context refresh
    refresh_every_n_turns: int = 10
    refresh_on_topic_change: bool = True
    topic_change_threshold: float = 0.3


class ContextCache:
    """High-performance caching for context queries"""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 30):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[ContextResult, float]] = {}
        self.access_times: deque = deque()
        self._lock = asyncio.Lock()
    
    def _generate_cache_key(self, query: str, k: int, threshold: float) -> str:
        """Generate cache key for query parameters"""
        key_data = f"{query}:{k}:{threshold}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, query: str, k: int, threshold: float) -> Optional[ContextResult]:
        """Get cached result if available and not expired"""
        async with self._lock:
            cache_key = self._generate_cache_key(query, k, threshold)
            
            if cache_key in self.cache:
                result, timestamp = self.cache[cache_key]
                
                # Check if expired
                if time.time() - timestamp < self.ttl_seconds:
                    # Update access time and mark as cache hit
                    result.cache_hit = True
                    return result
                else:
                    # Remove expired entry
                    del self.cache[cache_key]
            
            return None
    
    async def set(self, query: str, k: int, threshold: float, result: ContextResult):
        """Cache the result"""
        async with self._lock:
            cache_key = self._generate_cache_key(query, k, threshold)
            
            # Ensure cache size limit
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            # Store with current timestamp
            self.cache[cache_key] = (result, time.time())
            result.cache_hit = False
    
    async def clear(self):
        """Clear all cached entries"""
        async with self._lock:
            self.cache.clear()
            self.access_times.clear()


class TokenCounter:
    """Accurate token counting for Realtime API context"""
    
    def __init__(self, model_name: str = "gpt-4o-realtime-preview-2024-10-01"):
        self.model_name = model_name
        self.encoding = None
        
        if HAS_TIKTOKEN:
            try:
                # Try to get encoding for the specific model
                self.encoding = tiktoken.encoding_for_model(model_name)
            except KeyError:
                # Fallback to a compatible encoding
                self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text string"""
        if self.encoding and HAS_TIKTOKEN:
            return len(self.encoding.encode(text))
        else:
            # Rough approximation: 1 token ≈ 4 characters
            return len(text) // 4
    
    def estimate_tokens(self, text: str) -> int:
        """Faster token estimation for preliminary filtering"""
        # Quick approximation for performance
        return len(text) // 4


class MemoryContextProvider:
    """
    Specialized context provider for OpenAI Realtime API integration
    
    Builds on existing MemoryManager infrastructure to provide optimized
    context injection with performance enhancements for real-time usage.
    """
    
    def __init__(self, 
                 memory_manager: MemoryManager,
                 config: Optional[ContextProviderConfig] = None,
                 realtime_config: Optional[RealtimeAPIConfig] = None,
                 logger: Optional[logging.Logger] = None):
        
        self.memory_manager = memory_manager
        self.config = config or ContextProviderConfig()
        self.realtime_config = realtime_config
        self.logger = logger or logging.getLogger(__name__)
        
        # Token management
        model_name = realtime_config.model if realtime_config else "gpt-4o-realtime-preview-2024-10-01"
        self.token_counter = TokenCounter(model_name)
        
        # Performance optimization
        self.cache = ContextCache(
            max_size=self.config.max_cache_size,
            ttl_seconds=self.config.cache_ttl_seconds
        )
        
        # Context tracking
        self.conversation_turn_count = 0
        self.last_context_refresh = time.time()
        self.current_topic_embedding: Optional[np.ndarray] = None
        
        # Performance metrics
        self.total_requests = 0
        self.cache_hits = 0
        self.avg_processing_time = 0.0
        self.total_tokens_provided = 0
        
        self.logger.info("MemoryContextProvider initialized for Realtime API")
    
    async def get_context_for_query(self, 
                                   query: str,
                                   session_id: Optional[str] = None,
                                   k: Optional[int] = None,
                                   force_refresh: bool = False) -> ContextResult:
        """
        Get formatted context for a Realtime API query
        
        Args:
            query: Current user query or conversation context
            session_id: Optional session ID for session-specific context
            k: Number of context entries to retrieve (uses config default if None)
            force_refresh: Bypass cache and force fresh retrieval
            
        Returns:
            ContextResult with formatted context and metadata
        """
        start_time = time.time()
        self.total_requests += 1
        
        try:
            # Determine retrieval parameters
            k = k or self.config.default_k
            threshold = self.config.relevance_threshold
            
            # Check cache first (unless forced refresh)
            if not force_refresh:
                cached_result = await self.cache.get(query, k, threshold)
                if cached_result:
                    self.cache_hits += 1
                    self.logger.debug(f"Cache hit for query: {query[:50]}...")
                    return cached_result
            
            # Perform semantic retrieval
            context_result = await self._retrieve_and_format_context(
                query=query,
                session_id=session_id,
                k=k,
                threshold=threshold
            )
            
            # Cache the result
            await self.cache.set(query, k, threshold, context_result)
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            context_result.processing_time_ms = processing_time
            self._update_performance_metrics(processing_time)
            
            self.logger.debug(f"Context retrieved: {context_result.entries_included} entries, "
                            f"{context_result.total_tokens} tokens, {processing_time:.1f}ms")
            
            return context_result
            
        except Exception as e:
            self.logger.error(f"Error retrieving context: {e}")
            # Return empty context on error
            return ContextResult(
                formatted_context="",
                total_tokens=0,
                entries_included=0,
                entries_filtered=0,
                relevance_threshold_used=threshold,
                cache_hit=False,
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _retrieve_and_format_context(self,
                                         query: str,
                                         session_id: Optional[str],
                                         k: int,
                                         threshold: float) -> ContextResult:
        """Core context retrieval and formatting logic"""
        
        # Retrieve relevant documents using existing memory manager
        try:
            timeout = self.config.async_timeout_seconds
            relevant_docs = await asyncio.wait_for(
                self.memory_manager.retrieve_context(query, k=k),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self.logger.warning(f"Context retrieval timeout after {timeout}s")
            relevant_docs = []
        
        # Convert documents to context entries with relevance scoring
        context_entries = []
        entries_filtered = 0
        
        for doc in relevant_docs:
            # Calculate relevance score (ChromaDB provides similarity in metadata)
            relevance_score = doc.metadata.get('similarity', 0.0)
            
            # Apply relevance threshold
            if relevance_score < threshold:
                entries_filtered += 1
                continue
            
            # Count tokens for this entry
            token_count = self.token_counter.count_tokens(doc.page_content)
            
            # Create context entry
            entry = ContextEntry(
                content=doc.page_content,
                relevance_score=relevance_score,
                timestamp=datetime.fromisoformat(doc.metadata.get('timestamp', datetime.now().isoformat())),
                entry_type=doc.metadata.get('type', 'memory'),
                token_count=token_count,
                metadata=doc.metadata
            )
            
            context_entries.append(entry)
        
        # Sort by relevance score (highest first)
        context_entries.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Apply token limit and format context
        formatted_context, total_tokens, final_entries = self._format_context_with_token_limit(
            context_entries
        )
        
        return ContextResult(
            formatted_context=formatted_context,
            total_tokens=total_tokens,
            entries_included=len(final_entries),
            entries_filtered=entries_filtered,
            relevance_threshold_used=threshold,
            cache_hit=False,
            processing_time_ms=0.0,  # Will be set by caller
            context_entries=final_entries
        )
    
    def _format_context_with_token_limit(self, 
                                       context_entries: List[ContextEntry]) -> Tuple[str, int, List[ContextEntry]]:
        """Format context entries while respecting token limits"""
        
        max_tokens = self.config.max_context_tokens - self.config.reserved_tokens
        target_tokens = min(self.config.target_context_tokens, max_tokens)
        
        formatted_parts = []
        current_tokens = 0
        included_entries = []
        
        # Add system prompt prefix
        prefix_tokens = self.token_counter.count_tokens(self.config.system_prompt_prefix)
        current_tokens += prefix_tokens
        
        # Add context entries until token limit
        for entry in context_entries:
            # Estimate tokens for this entry including formatting
            entry_text = self._format_single_entry(entry)
            entry_tokens = self.token_counter.count_tokens(entry_text)
            
            # Check if adding this entry would exceed limits
            if current_tokens + entry_tokens > target_tokens:
                # If we haven't included any entries yet, try to fit at least one
                if not included_entries and current_tokens + entry_tokens <= max_tokens:
                    pass  # Include this entry even if over target
                else:
                    break  # Stop adding entries
            
            formatted_parts.append(entry_text)
            current_tokens += entry_tokens
            included_entries.append(entry)
        
        # Combine all parts
        if formatted_parts:
            context_content = "\n".join(formatted_parts)
            formatted_context = self.config.context_template.format(
                context_content=context_content
            )
        else:
            formatted_context = ""
        
        # Add system prompt prefix if we have context
        if formatted_context and self.config.system_prompt_prefix:
            formatted_context = self.config.system_prompt_prefix + formatted_context
        
        # Final token count
        final_tokens = self.token_counter.count_tokens(formatted_context)
        
        return formatted_context, final_tokens, included_entries
    
    def _format_single_entry(self, entry: ContextEntry) -> str:
        """Format a single context entry"""
        parts = []
        
        # Add timestamp if configured
        if self.config.include_timestamps:
            timestamp_str = entry.timestamp.strftime("%Y-%m-%d %H:%M")
            parts.append(f"[{timestamp_str}]")
        
        # Add entry type
        parts.append(f"({entry.entry_type.upper()})")
        
        # Add relevance score if configured
        if self.config.include_relevance_scores:
            parts.append(f"(relevance: {entry.relevance_score:.2f})")
        
        # Add content
        content = entry.content.strip()
        if parts:
            return f"{' '.join(parts)} {content}"
        else:
            return content
    
    def _update_performance_metrics(self, processing_time_ms: float):
        """Update running performance metrics"""
        if self.total_requests == 1:
            self.avg_processing_time = processing_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.avg_processing_time = (alpha * processing_time_ms + 
                                      (1 - alpha) * self.avg_processing_time)
    
    async def refresh_context_if_needed(self, current_query: str) -> bool:
        """Check if context should be refreshed and clear cache if so"""
        should_refresh = False
        
        # Check turn-based refresh
        self.conversation_turn_count += 1
        if self.conversation_turn_count % self.config.refresh_every_n_turns == 0:
            should_refresh = True
            self.logger.debug(f"Context refresh triggered by turn count: {self.conversation_turn_count}")
        
        # Check time-based refresh
        time_since_refresh = time.time() - self.last_context_refresh
        if time_since_refresh > self.config.cache_ttl_seconds * 2:
            should_refresh = True
            self.logger.debug("Context refresh triggered by time threshold")
        
        # TODO: Add topic change detection using embeddings
        # This would require generating embeddings for the current query
        # and comparing with stored topic embedding
        
        if should_refresh:
            await self.cache.clear()
            self.last_context_refresh = time.time()
            self.logger.info("Context cache refreshed")
        
        return should_refresh
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        cache_hit_rate = (self.cache_hits / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_hit_rate_percent": cache_hit_rate,
            "avg_processing_time_ms": self.avg_processing_time,
            "total_tokens_provided": self.total_tokens_provided,
            "conversation_turns": self.conversation_turn_count
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.cache.clear()
        self.logger.info("MemoryContextProvider cleaned up")


# Factory function for easy creation
def create_memory_context_provider(
    memory_manager: MemoryManager,
    config: Optional[ContextProviderConfig] = None,
    realtime_config: Optional[RealtimeAPIConfig] = None,
    logger: Optional[logging.Logger] = None
) -> MemoryContextProvider:
    """Create a MemoryContextProvider instance"""
    return MemoryContextProvider(
        memory_manager=memory_manager,
        config=config,
        realtime_config=realtime_config,
        logger=logger
    )


# Export main components
__all__ = [
    'MemoryContextProvider',
    'ContextProviderConfig', 
    'ContextResult',
    'ContextEntry',
    'ContextCache',
    'TokenCounter',
    'create_memory_context_provider'
] 