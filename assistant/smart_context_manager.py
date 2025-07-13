"""
Sovereign Voice Assistant - Smart Context Management for Realtime API

Implements intelligent context prioritization and truncation system designed to work 
within Realtime API token limits while maintaining relevance and conversation quality.

Key Features:
- Priority-based context allocation with 4 tiers
- Accurate token counting using tiktoken for precise limit management
- Sliding window conversation history (last 10 exchanges)
- Context compression using extractive summarization
- Relevance scoring with sentence-transformers
- Dynamic context update triggers
- Performance optimization with caching and async operations

Priority System:
1. System Instructions (2k tokens) - Highest priority
2. Recent Memory (4k tokens) - Critical conversation context
3. Screen Content (2k tokens) - Current screen awareness  
4. Conversation History (24k tokens) - Sliding window of exchanges

Total Budget: 32k tokens for optimal Realtime API performance
"""

import asyncio
import logging
import time
import re
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, IntEnum
import hashlib
import threading
from difflib import SequenceMatcher

# ML and NLP imports
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    logging.warning("tiktoken not available. Token counting will use approximation.")

try:
    from sentence_transformers import SentenceTransformer, util
    import numpy as np
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logging.warning("sentence-transformers not available. Relevance scoring will use fallback.")

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    from nltk.corpus import stopwords
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    logging.warning("NLTK not available. Text summarization will use fallback.")

from .memory import MemoryManager
from .screen_context_provider import ScreenContextProvider, ScreenContextData


class ContextPriority(IntEnum):
    """Context priority levels (lower number = higher priority)"""
    SYSTEM_INSTRUCTIONS = 1
    RECENT_MEMORY = 2
    SCREEN_CONTENT = 3
    CONVERSATION_HISTORY = 4


class ContextType(Enum):
    """Types of context content"""
    SYSTEM = "system_instructions"
    MEMORY = "recent_memory"  
    SCREEN = "screen_content"
    CONVERSATION = "conversation_history"


@dataclass
class ContextBudget:
    """Token budget allocation for different context types"""
    system_instructions: int = 2000      # Core system prompt
    recent_memory: int = 4000           # Critical memory context
    screen_content: int = 2000          # Current screen awareness
    conversation_history: int = 24000   # Chat history sliding window
    
    @property
    def total_budget(self) -> int:
        """Total token budget"""
        return (self.system_instructions + self.recent_memory + 
                self.screen_content + self.conversation_history)
    
    def get_budget_for_type(self, context_type: ContextType) -> int:
        """Get token budget for specific context type"""
        budget_map = {
            ContextType.SYSTEM: self.system_instructions,
            ContextType.MEMORY: self.recent_memory,
            ContextType.SCREEN: self.screen_content,
            ContextType.CONVERSATION: self.conversation_history
        }
        return budget_map.get(context_type, 0)


@dataclass
class ContextSegment:
    """Individual context segment with metadata"""
    content: str
    context_type: ContextType
    priority: ContextPriority
    timestamp: datetime
    token_count: int
    relevance_score: float
    source_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set priority based on context type"""
        priority_map = {
            ContextType.SYSTEM: ContextPriority.SYSTEM_INSTRUCTIONS,
            ContextType.MEMORY: ContextPriority.RECENT_MEMORY,
            ContextType.SCREEN: ContextPriority.SCREEN_CONTENT,
            ContextType.CONVERSATION: ContextPriority.CONVERSATION_HISTORY
        }
        self.priority = priority_map.get(self.context_type, ContextPriority.CONVERSATION_HISTORY)


@dataclass
class ConversationTurn:
    """Individual conversation exchange"""
    user_message: str
    assistant_response: str
    timestamp: datetime
    user_tokens: int
    assistant_tokens: int
    turn_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_tokens(self) -> int:
        """Total tokens for this conversation turn"""
        return self.user_tokens + self.assistant_tokens
    
    def to_context_string(self) -> str:
        """Format as context string"""
        return f"User: {self.user_message}\nAssistant: {self.assistant_response}"


@dataclass
class SmartContextConfig:
    """Configuration for Smart Context Manager"""
    
    # Token budget settings
    budget: ContextBudget = field(default_factory=ContextBudget)
    
    # Conversation history settings
    max_conversation_turns: int = 10  # Sliding window size
    min_conversation_turns: int = 3   # Minimum to keep for context
    
    # Relevance scoring settings
    relevance_threshold: float = 0.3  # Minimum relevance to include
    enable_relevance_scoring: bool = True
    semantic_model_name: str = "all-MiniLM-L6-v2"
    
    # Context compression settings
    enable_compression: bool = True
    compression_ratio: float = 0.6    # Target compression ratio
    min_compression_length: int = 500 # Minimum length to trigger compression
    
    # Update triggers
    context_refresh_interval: float = 30.0  # Seconds between auto-refresh
    screen_change_threshold: float = 0.2    # Screen content change trigger
    memory_relevance_threshold: float = 0.4 # Memory relevance change trigger
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 60
    max_cache_size: int = 100
    enable_background_updates: bool = True
    
    # Model settings
    tiktoken_model: str = "gpt-4o-realtime-preview-2024-10-01"


class TokenCounter:
    """Accurate token counting with tiktoken"""
    
    def __init__(self, model_name: str = "gpt-4o-realtime-preview-2024-10-01"):
        self.model_name = model_name
        self.encoding = None
        
        if HAS_TIKTOKEN:
            try:
                # Try model-specific encoding
                self.encoding = tiktoken.encoding_for_model(model_name)
            except KeyError:
                # Fallback to compatible encoding
                self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.encoding and HAS_TIKTOKEN:
            return len(self.encoding.encode(text))
        else:
            # Approximation: 1 token ≈ 4 characters
            return max(1, len(text) // 4)
    
    def estimate_tokens_fast(self, text: str) -> int:
        """Fast token estimation for filtering"""
        return max(1, len(text) // 4)


class RelevanceScorer:
    """Context relevance scoring using sentence transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.logger = logging.getLogger(__name__)
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(model_name)
                self.logger.info(f"Loaded sentence transformer model: {model_name}")
            except Exception as e:
                self.logger.warning(f"Failed to load sentence transformer: {e}")
                self.model = None
    
    def calculate_relevance(self, query: str, context: str) -> float:
        """Calculate relevance score between query and context"""
        if self.model and HAS_SENTENCE_TRANSFORMERS:
            try:
                # Encode texts
                query_embedding = self.model.encode(query, convert_to_tensor=True)
                context_embedding = self.model.encode(context, convert_to_tensor=True)
                
                # Calculate cosine similarity
                similarity = util.cos_sim(query_embedding, context_embedding)
                return float(similarity.item())
                
            except Exception as e:
                self.logger.error(f"Error calculating semantic similarity: {e}")
        
        # Fallback to word overlap similarity
        return self._word_overlap_similarity(query, context)
    
    def _word_overlap_similarity(self, text1: str, text2: str) -> float:
        """Fallback similarity using word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0


class ContextCompressor:
    """Context compression using extractive summarization"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLTK components if available
        if HAS_NLTK:
            try:
                import ssl
                try:
                    _create_unverified_https_context = ssl._create_unverified_context
                except AttributeError:
                    pass
                else:
                    ssl._create_default_https_context = _create_unverified_https_context
                
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
            except Exception as e:
                self.logger.warning(f"NLTK setup warning: {e}")
    
    def compress_text(self, text: str, target_ratio: float = 0.6) -> str:
        """Compress text using extractive summarization"""
        if len(text) < 200:  # Don't compress very short text
            return text
        
        if HAS_NLTK:
            return self._extractive_summarization(text, target_ratio)
        else:
            return self._simple_compression(text, target_ratio)
    
    def _extractive_summarization(self, text: str, target_ratio: float) -> str:
        """NLTK-based extractive summarization"""
        try:
            # Tokenize into sentences
            sentences = sent_tokenize(text)
            
            if len(sentences) <= 2:
                return text
            
            # Calculate target sentence count
            target_count = max(1, int(len(sentences) * target_ratio))
            
            # Score sentences by word frequency
            word_freq = self._calculate_word_frequencies(text)
            sentence_scores = []
            
            for i, sentence in enumerate(sentences):
                score = sum(word_freq.get(word.lower(), 0) 
                           for word in sentence.split() 
                           if word.lower() not in stopwords.words('english'))
                sentence_scores.append((score, i, sentence))
            
            # Select top sentences
            sentence_scores.sort(reverse=True)
            selected_indices = sorted([idx for _, idx, _ in sentence_scores[:target_count]])
            
            # Reconstruct text maintaining order
            compressed = ' '.join(sentences[i] for i in selected_indices)
            return compressed
            
        except Exception as e:
            self.logger.error(f"Extractive summarization failed: {e}")
            return self._simple_compression(text, target_ratio)
    
    def _calculate_word_frequencies(self, text: str) -> Dict[str, int]:
        """Calculate word frequencies for sentence scoring"""
        words = text.lower().split()
        if HAS_NLTK:
            stop_words = set(stopwords.words('english'))
            words = [word for word in words if word not in stop_words]
        
        freq = {}
        for word in words:
            freq[word] = freq.get(word, 0) + 1
        
        return freq
    
    def _simple_compression(self, text: str, target_ratio: float) -> str:
        """Simple compression by truncating"""
        target_length = int(len(text) * target_ratio)
        return text[:target_length] + "..." if len(text) > target_length else text


class SmartContextManager:
    """
    Intelligent context manager for Realtime API with priority-based allocation,
    token counting, compression, and relevance scoring
    """
    
    def __init__(self,
                 config: Optional[SmartContextConfig] = None,
                 memory_manager: Optional[MemoryManager] = None,
                 screen_context_provider: Optional[ScreenContextProvider] = None,
                 logger: Optional[logging.Logger] = None):
        
        self.config = config or SmartContextConfig()
        self.memory_manager = memory_manager
        self.screen_context_provider = screen_context_provider
        self.logger = logger or logging.getLogger(__name__)
        
        # Core components
        self.token_counter = TokenCounter(self.config.tiktoken_model)
        self.relevance_scorer = RelevanceScorer(self.config.semantic_model_name)
        self.compressor = ContextCompressor()
        
        # Context storage
        self.conversation_history: deque[ConversationTurn] = deque(maxlen=self.config.max_conversation_turns)
        self.system_instructions: str = """You are Sovereign, an advanced AI voice assistant with comprehensive capabilities:

CORE CAPABILITIES:
- Screen Awareness: You can see and discuss the user's current screen content via OCR
- Memory System: You have access to conversation history and context from previous interactions
- Voice Interface: You communicate naturally through speech-to-speech conversation
- Real-time Processing: You provide immediate responses with minimal latency

BEHAVIOR GUIDELINES:
- Reference screen content when relevant to user queries
- Use conversation history to maintain context and continuity
- Provide concise, helpful responses optimized for voice interaction
- Ask clarifying questions when screen content or context is unclear

SCREEN CONTENT: When screen content is provided, analyze it and reference specific elements the user might be asking about."""
        self.current_screen_content: Optional[ScreenContextData] = None
        self.recent_memory_context: List[str] = []
        
        # State management
        self.last_context_refresh: float = time.time()  # Initialize to current time
        self.last_screen_hash: str = ""
        self.last_memory_relevance: float = 0.0
        self.context_cache: Dict[str, Tuple[str, float]] = {}  # hash -> (context, timestamp)
        
        # Performance tracking
        self.context_build_count: int = 0
        self.cache_hits: int = 0
        self.compression_count: int = 0
        self.total_tokens_managed: int = 0
        
        # Threading
        self.lock = threading.RLock()
        self.background_task: Optional[asyncio.Task] = None
        
        self.logger.info("Smart Context Manager initialized with priority-based allocation")
    
    async def initialize(self) -> bool:
        """Initialize the context manager"""
        try:
            self.logger.info("🧠 Initializing Smart Context Manager...")
            
            # Set default system instructions
            self.system_instructions = """You are Sovereign, an advanced AI voice assistant with comprehensive capabilities:

CORE CAPABILITIES:
- Screen Awareness: You can see and discuss the user's current screen content via OCR
- Memory System: You have access to conversation history and context from previous interactions
- Voice Interface: You communicate naturally through speech-to-speech conversation
- Real-time Processing: You provide immediate responses with minimal latency

BEHAVIOR GUIDELINES:
- Reference screen content when relevant to user queries
- Use conversation history to maintain context and continuity
- Provide concise, helpful responses optimized for voice interaction
- Ask clarifying questions when screen content or context is unclear

SCREEN CONTENT: When screen content is provided, analyze it and reference specific elements the user might be asking about."""
            
            # Start background updates if enabled
            if self.config.enable_background_updates:
                self.background_task = asyncio.create_task(self._background_update_loop())
            
            self.logger.info("✅ Smart Context Manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Smart Context Manager: {e}")
            return False
    
    async def build_context(self, current_query: str, force_refresh: bool = False) -> str:
        """Build optimized context for Realtime API with priority-based allocation"""
        start_time = time.time()
        
        try:
            with self.lock:
                self.context_build_count += 1
                
                # Check cache first
                if not force_refresh and self.config.enable_caching:
                    cache_key = self._generate_cache_key(current_query)
                    cached = self.context_cache.get(cache_key)
                    if cached and (time.time() - cached[1]) < self.config.cache_ttl_seconds:
                        self.cache_hits += 1
                        return cached[0]
                
                # Build context segments
                segments = await self._collect_context_segments(current_query)
                
                # Apply priority-based allocation
                allocated_segments = self._allocate_by_priority(segments)
                
                # Build final context string
                context = self._format_context(allocated_segments)
                
                # Cache result
                if self.config.enable_caching:
                    cache_key = self._generate_cache_key(current_query)
                    self.context_cache[cache_key] = (context, time.time())
                    
                    # Cleanup old cache entries
                    self._cleanup_cache()
                
                # Update metrics
                processing_time = (time.time() - start_time) * 1000
                total_tokens = self.token_counter.count_tokens(context)
                self.total_tokens_managed += total_tokens
                
                self.logger.debug(f"Context built: {len(allocated_segments)} segments, "
                                f"{total_tokens} tokens, {processing_time:.1f}ms")
                
                return context
                
        except Exception as e:
            self.logger.error(f"❌ Failed to build context: {e}")
            return self.system_instructions  # Fallback to system instructions only
    
    async def _collect_context_segments(self, current_query: str) -> List[ContextSegment]:
        """Collect all available context segments"""
        segments = []
        
        # 1. System Instructions (Priority 1)
        if self.system_instructions:
            token_count = self.token_counter.count_tokens(self.system_instructions)
            segments.append(ContextSegment(
                content=self.system_instructions,
                context_type=ContextType.SYSTEM,
                priority=ContextPriority.SYSTEM_INSTRUCTIONS,
                timestamp=datetime.now(timezone.utc),
                token_count=token_count,
                relevance_score=1.0,  # Always fully relevant
                source_id="system_instructions"
            ))
        
        # 2. Recent Memory (Priority 2)
        if self.memory_manager:
            memory_segments = await self._get_memory_segments(current_query)
            segments.extend(memory_segments)
        
        # 3. Screen Content (Priority 3)
        if self.screen_context_provider:
            screen_segment = await self._get_screen_segment(current_query)
            if screen_segment:
                segments.append(screen_segment)
        
        # 4. Conversation History (Priority 4)
        conversation_segments = self._get_conversation_segments(current_query)
        segments.extend(conversation_segments)
        
        return segments
    
    async def _get_memory_segments(self, query: str) -> List[ContextSegment]:
        """Get relevant memory context segments"""
        segments = []
        
        try:
            # Retrieve relevant memories
            memories = await self.memory_manager.retrieve_context(
                query=query,
                include_conversations=True,
                include_screen=False,  # Screen handled separately
                max_results=5
            )
            
            for doc in memories:
                content = doc.page_content
                timestamp_str = doc.metadata.get("timestamp", "")
                
                # Parse timestamp
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                except:
                    timestamp = datetime.now(timezone.utc)
                
                # Calculate relevance
                relevance = self.relevance_scorer.calculate_relevance(query, content)
                
                # Be more permissive with memory content since it's already filtered by the memory system
                if relevance >= max(0.1, self.config.relevance_threshold * 0.5):
                    token_count = self.token_counter.count_tokens(content)
                    
                    segments.append(ContextSegment(
                        content=content,
                        context_type=ContextType.MEMORY,
                        priority=ContextPriority.RECENT_MEMORY,
                        timestamp=timestamp,
                        token_count=token_count,
                        relevance_score=relevance,
                        source_id=f"memory_{doc.metadata.get('id', 'unknown')}"
                    ))
            
            # Sort by relevance
            segments.sort(key=lambda x: x.relevance_score, reverse=True)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get memory segments: {e}")
        
        return segments
    
    async def _get_screen_segment(self, query: str) -> Optional[ContextSegment]:
        """Get current screen content segment"""
        try:
            screen_data = await self.screen_context_provider.get_current_context_data()
            
            if screen_data and screen_data.content:
                # Calculate relevance to query
                relevance = self.relevance_scorer.calculate_relevance(query, screen_data.content)
                
                # Special handling for screen-related queries
                screen_keywords = ['screen', 'see', 'display', 'showing', 'visible', 'view', 'window', 'app', 'code', 'editor']
                query_lower = query.lower()
                is_screen_query = any(keyword in query_lower for keyword in screen_keywords)
                
                # Be more permissive with screen content, especially for screen-related queries
                min_threshold = 0.05 if is_screen_query else max(0.1, self.config.relevance_threshold * 0.5)
                
                if relevance >= min_threshold or is_screen_query:
                    # Use formatted context string
                    formatted_content = screen_data.to_context_string()
                    token_count = self.token_counter.count_tokens(formatted_content)
                    
                    return ContextSegment(
                        content=formatted_content,
                        context_type=ContextType.SCREEN,
                        priority=ContextPriority.SCREEN_CONTENT,
                        timestamp=screen_data.timestamp,
                        token_count=token_count,
                        relevance_score=max(relevance, 0.8 if is_screen_query else relevance),  # Boost relevance for screen queries
                        source_id=f"screen_{screen_data.source_hash}"
                    )
        
        except Exception as e:
            self.logger.error(f"❌ Failed to get screen segment: {e}")
        
        return None
    
    def _get_conversation_segments(self, query: str) -> List[ContextSegment]:
        """Get conversation history segments"""
        segments = []
        
        try:
            # Convert conversation turns to segments
            for turn in self.conversation_history:
                content = turn.to_context_string()
                
                # Calculate relevance to current query
                relevance = self.relevance_scorer.calculate_relevance(query, content)
                
                # Be more permissive with conversation history since it's recent context
                if relevance >= max(0.05, self.config.relevance_threshold * 0.3):
                    segments.append(ContextSegment(
                        content=content,
                        context_type=ContextType.CONVERSATION,
                        priority=ContextPriority.CONVERSATION_HISTORY,
                        timestamp=turn.timestamp,
                        token_count=turn.total_tokens,
                        relevance_score=relevance,
                        source_id=turn.turn_id
                    ))
            
            # Sort by timestamp (most recent first)
            segments.sort(key=lambda x: x.timestamp, reverse=True)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get conversation segments: {e}")
        
        return segments
    
    def _allocate_by_priority(self, segments: List[ContextSegment]) -> List[ContextSegment]:
        """Allocate context segments by priority within token budgets"""
        allocated = []
        budget = self.config.budget
        
        # Group segments by priority
        priority_groups = {}
        for segment in segments:
            priority = segment.priority
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(segment)
        
        # Allocate by priority (lower number = higher priority)
        for priority in sorted(priority_groups.keys()):
            context_type = priority_groups[priority][0].context_type
            available_budget = budget.get_budget_for_type(context_type)
            used_tokens = 0
            
            # Sort by relevance within priority group
            priority_segments = sorted(priority_groups[priority], 
                                     key=lambda x: x.relevance_score, reverse=True)
            
            for segment in priority_segments:
                if used_tokens + segment.token_count <= available_budget:
                    # Segment fits within budget
                    allocated.append(segment)
                    used_tokens += segment.token_count
                else:
                    # Try compression for large segments
                    if (self.config.enable_compression and 
                        segment.token_count > self.config.min_compression_length):
                        
                        compressed_content = self.compressor.compress_text(
                            segment.content, 
                            self.config.compression_ratio
                        )
                        compressed_tokens = self.token_counter.count_tokens(compressed_content)
                        
                        if used_tokens + compressed_tokens <= available_budget:
                            # Compressed segment fits
                            compressed_segment = ContextSegment(
                                content=compressed_content,
                                context_type=segment.context_type,
                                priority=segment.priority,
                                timestamp=segment.timestamp,
                                token_count=compressed_tokens,
                                relevance_score=segment.relevance_score * 0.8,  # Slight penalty for compression
                                source_id=segment.source_id + "_compressed",
                                metadata={**segment.metadata, "compressed": True}
                            )
                            allocated.append(compressed_segment)
                            used_tokens += compressed_tokens
                            self.compression_count += 1
                    
                    # If still doesn't fit, skip this segment
                    break
        
        return allocated
    
    def _format_context(self, segments: List[ContextSegment]) -> str:
        """Format allocated segments into final context string"""
        if not segments:
            return self.system_instructions
        
        # Group by context type for organized output
        type_groups = {
            ContextType.SYSTEM: [],
            ContextType.MEMORY: [],
            ContextType.SCREEN: [],
            ContextType.CONVERSATION: []
        }
        
        for segment in segments:
            type_groups[segment.context_type].append(segment)
        
        # Build formatted context
        context_parts = []
        
        # System instructions (always first)
        if type_groups[ContextType.SYSTEM]:
            context_parts.append(type_groups[ContextType.SYSTEM][0].content)
        
        # Memory context
        if type_groups[ContextType.MEMORY]:
            context_parts.append("\n=== RELEVANT MEMORIES ===")
            for segment in type_groups[ContextType.MEMORY]:
                context_parts.append(f"\n{segment.content}")
            context_parts.append("\n=== END MEMORIES ===")
        
        # Screen content
        if type_groups[ContextType.SCREEN]:
            context_parts.append("\n")
            context_parts.append(type_groups[ContextType.SCREEN][0].content)
        
        # Conversation history
        if type_groups[ContextType.CONVERSATION]:
            context_parts.append("\n=== RECENT CONVERSATION ===")
            # Sort by timestamp for chronological order
            conversation_segments = sorted(type_groups[ContextType.CONVERSATION], 
                                         key=lambda x: x.timestamp)
            for segment in conversation_segments:
                context_parts.append(f"\n{segment.content}")
            context_parts.append("\n=== END CONVERSATION ===")
        
        return "".join(context_parts)
    
    def add_conversation_turn(self, user_message: str, assistant_response: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a conversation turn to the sliding window"""
        with self.lock:
            # Count tokens
            user_tokens = self.token_counter.count_tokens(user_message)
            assistant_tokens = self.token_counter.count_tokens(assistant_response)
            
            # Create turn
            turn = ConversationTurn(
                user_message=user_message,
                assistant_response=assistant_response,
                timestamp=datetime.now(timezone.utc),
                user_tokens=user_tokens,
                assistant_tokens=assistant_tokens,
                turn_id=f"turn_{int(time.time())}_{len(self.conversation_history)}",
                metadata=metadata or {}
            )
            
            # Add to sliding window deque (automatically maintains max size)
            self.conversation_history.append(turn)
            
            self.logger.debug(f"Added conversation turn: {user_tokens + assistant_tokens} tokens, "
                            f"sliding window size: {len(self.conversation_history)}")
    
    def update_system_instructions(self, instructions: str):
        """Update system instructions"""
        with self.lock:
            self.system_instructions = instructions
            self._invalidate_cache()
            self.logger.debug("System instructions updated")
    
    def should_refresh_context(self) -> bool:
        """Check if context should be refreshed based on triggers"""
        current_time = time.time()
        
        # Time-based refresh
        if current_time - self.last_context_refresh > self.config.context_refresh_interval:
            return True
        
        # Screen content change
        if self.screen_context_provider and self.current_screen_content:
            try:
                new_screen_data = asyncio.run(
                    self.screen_context_provider.get_current_context_data()
                )
                if new_screen_data and new_screen_data.source_hash != self.last_screen_hash:
                    if new_screen_data.change_score >= self.config.screen_change_threshold:
                        return True
            except:
                pass
        
        return False
    
    async def _background_update_loop(self):
        """Background task for periodic context updates"""
        while True:
            try:
                await asyncio.sleep(self.config.context_refresh_interval)
                
                if self.should_refresh_context():
                    self.last_context_refresh = time.time()
                    self._invalidate_cache()
                    self.logger.debug("Background context refresh triggered")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Background update error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    def _generate_cache_key(self, query: str) -> str:
        """Generate cache key for context"""
        # Include relevant state in cache key
        state_data = {
            "query": query,
            "conversation_count": len(self.conversation_history),
            "last_turn_id": self.conversation_history[-1].turn_id if self.conversation_history else "",
            "screen_hash": self.last_screen_hash,
            "system_hash": hashlib.md5(self.system_instructions.encode()).hexdigest()[:8]
        }
        state_str = str(sorted(state_data.items()))
        return hashlib.md5(state_str.encode()).hexdigest()
    
    def _invalidate_cache(self):
        """Invalidate context cache"""
        with self.lock:
            self.context_cache.clear()
    
    def _cleanup_cache(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.context_cache.items()
            if current_time - timestamp > self.config.cache_ttl_seconds
        ]
        
        for key in expired_keys:
            del self.context_cache[key]
        
        # Limit cache size
        if len(self.context_cache) > self.config.max_cache_size:
            # Remove oldest entries
            sorted_items = sorted(self.context_cache.items(), key=lambda x: x[1][1])
            excess_count = len(self.context_cache) - self.config.max_cache_size
            
            for key, _ in sorted_items[:excess_count]:
                del self.context_cache[key]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "context_build_count": self.context_build_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(1, self.context_build_count),
            "compression_count": self.compression_count,
            "total_tokens_managed": self.total_tokens_managed,
            "conversation_turns": len(self.conversation_history),
            "cache_size": len(self.context_cache),
            "budget": {
                "system": self.config.budget.system_instructions,
                "memory": self.config.budget.recent_memory,
                "screen": self.config.budget.screen_content,
                "conversation": self.config.budget.conversation_history,
                "total": self.config.budget.total_budget
            }
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.background_task:
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
        
        self._invalidate_cache()
        self.logger.info("Smart Context Manager cleaned up")


# Factory functions
def create_smart_context_manager(
    config: Optional[SmartContextConfig] = None,
    memory_manager: Optional[MemoryManager] = None,
    screen_context_provider: Optional[ScreenContextProvider] = None,
    logger: Optional[logging.Logger] = None
) -> SmartContextManager:
    """Factory function to create SmartContextManager"""
    return SmartContextManager(
        config=config,
        memory_manager=memory_manager,
        screen_context_provider=screen_context_provider,
        logger=logger
    )


def get_default_smart_context_config() -> SmartContextConfig:
    """Get default configuration optimized for Realtime API"""
    return SmartContextConfig(
        budget=ContextBudget(
            system_instructions=2000,
            recent_memory=4000,
            screen_content=2000,
            conversation_history=24000
        ),
        max_conversation_turns=10,
        relevance_threshold=0.3,
        enable_compression=True,
        context_refresh_interval=30.0,
        enable_caching=True
    ) 