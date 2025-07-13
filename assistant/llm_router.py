import re
import asyncio
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from openai import AsyncOpenAI
import time
import httpx
from .router_config import get_router_config, RouterConfig, ModelConfig
from .rate_limiter import ModelRateLimiter, RateLimitConfig, RateLimitStrategy, estimate_token_count, estimate_response_tokens

logger = logging.getLogger(__name__)

class QueryComplexity(Enum):
    """Enum for query complexity levels"""
    SIMPLE = "simple"
    COMPLEX = "complex"

@dataclass
class QueryClassification:
    """Result of query classification analysis"""
    complexity: QueryComplexity
    confidence: float
    reasoning: str
    factors: Dict[str, Any]

class QueryClassifier:
    """Intelligent query classification system for LLM routing"""
    
    def __init__(self):
        # Complex query indicators
        self.complex_keywords = {
            'analysis', 'analyze', 'explain', 'research', 'investigate', 
            'comprehensive', 'detailed', 'compare', 'contrast', 'pros and cons',
            'strategy', 'plan', 'design', 'architect', 'implement', 'develop',
            'optimize', 'performance', 'security', 'scalability', 'architecture',
            'algorithm', 'complex', 'sophisticated', 'advanced', 'enterprise',
            'integration', 'migration', 'refactor', 'debug', 'troubleshoot',
            'workflow', 'process', 'methodology', 'framework', 'best practices',
            'recommendations', 'considerations', 'implications', 'trade-offs',
            'requirements', 'specifications', 'documentation', 'technical',
            'machine learning', 'ai', 'neural network', 'deep learning',
            'database', 'sql', 'nosql', 'cloud', 'microservices', 'api',
            'kubernetes', 'docker', 'devops', 'ci/cd', 'deployment',
            'monitoring', 'logging', 'metrics', 'observability', 'testing',
            'unit test', 'integration test', 'end-to-end', 'automation'
        }
        
        # Simple query indicators
        self.simple_keywords = {
            'what', 'when', 'where', 'who', 'which', 'how many', 'how much',
            'yes', 'no', 'maybe', 'ok', 'okay', 'sure', 'thanks', 'thank you',
            'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
            'bye', 'goodbye', 'see you', 'later', 'stop', 'pause', 'resume',
            'start', 'begin', 'end', 'finish', 'next', 'previous', 'back',
            'up', 'down', 'left', 'right', 'open', 'close', 'save', 'load',
            'play', 'pause', 'stop', 'volume', 'mute', 'unmute', 'louder', 'quieter'
        }
        
        # Question patterns that typically indicate complexity
        self.complex_patterns = [
            r'how\s+(?:do|can|should|would|could)\s+(?:i|we)\s+(?:implement|build|create|design|develop)',
            r'what\s+(?:are\s+the|is\s+the)\s+(?:best|recommended|optimal)\s+(?:way|approach|method|strategy)',
            r'explain\s+(?:how|why|what|when|where)',
            r'(?:compare|contrast)\s+(?:between|with)',
            r'(?:pros\s+and\s+cons|advantages\s+and\s+disadvantages)',
            r'step\s+by\s+step',
            r'detailed\s+(?:explanation|analysis|guide|instructions)',
            r'(?:troubleshoot|debug|fix|resolve|solve)\s+(?:the|this|my)',
            r'(?:optimize|improve|enhance|upgrade)\s+(?:the|this|my)',
            r'(?:integrate|connect|link|combine)\s+(?:with|to)',
            r'(?:migrate|move|transfer|convert)\s+(?:from|to)',
            r'(?:security|performance|scalability)\s+(?:concerns|issues|considerations)',
            r'(?:architecture|design|structure)\s+(?:for|of)',
            r'(?:requirements|specifications)\s+(?:for|of)',
            r'(?:best\s+practices|recommendations|guidelines)\s+(?:for|of)',
        ]
        
        # Configuration thresholds
        self.config = {
            'min_complex_length': 40,  # Minimum characters for complex queries
            'max_simple_length': 15,   # Maximum characters for simple queries
            'complex_keyword_threshold': 1,  # Minimum complex keywords for classification
            'simple_keyword_threshold': 1,   # Minimum simple keywords for classification
            'pattern_match_weight': 0.4,     # Weight for pattern matching
            'keyword_weight': 0.3,           # Weight for keyword analysis
            'length_weight': 0.2,            # Weight for length analysis
            'context_weight': 0.1,           # Weight for context analysis
            'confidence_threshold': 0.7,     # Minimum confidence for classification
        }
    
    def classify_query(self, query: str, conversation_context: Optional[List[str]] = None) -> QueryClassification:
        """
        Classify a query as simple or complex based on multiple factors
        
        Args:
            query: The user's query to classify
            conversation_context: Previous conversation messages for context
            
        Returns:
            QueryClassification with complexity, confidence, and reasoning
        """
        query_lower = query.lower().strip()
        
        # Initialize scoring factors
        factors = {
            'length': len(query),
            'complex_keywords': 0,
            'simple_keywords': 0,
            'pattern_matches': 0,
            'context_indicators': 0
        }
        
        # Length analysis
        length_score = self._analyze_length(query, factors)
        
        # Keyword analysis
        keyword_score = self._analyze_keywords(query_lower, factors)
        
        # Pattern matching
        pattern_score = self._analyze_patterns(query_lower, factors)
        
        # Context analysis
        context_score = self._analyze_context(conversation_context, factors)
        
        # Calculate weighted score
        total_score = (
            length_score * self.config['length_weight'] +
            keyword_score * self.config['keyword_weight'] +
            pattern_score * self.config['pattern_match_weight'] +
            context_score * self.config['context_weight']
        )
        
        # Determine complexity and confidence
        if total_score > 0.55:
            complexity = QueryComplexity.COMPLEX
            confidence = min(0.95, 0.6 + (total_score * 0.35))
        elif total_score < 0.35:
            complexity = QueryComplexity.SIMPLE
            confidence = min(0.95, 0.6 + ((1 - total_score) * 0.35))
        else:
            # Borderline case - default to simple for faster response
            complexity = QueryComplexity.SIMPLE
            confidence = 0.5 + (total_score * 0.1)  # Slightly higher confidence for borderline
        
        # Generate reasoning
        reasoning = self._generate_reasoning(complexity, factors, total_score)
        
        return QueryClassification(
            complexity=complexity,
            confidence=confidence,
            reasoning=reasoning,
            factors=factors
        )
    
    def _analyze_length(self, query: str, factors: Dict[str, Any]) -> float:
        """Analyze query length for complexity indicators"""
        length = len(query)
        factors['length'] = length
        
        if length > self.config['min_complex_length']:
            return 0.8  # Long queries tend to be complex
        elif length < self.config['max_simple_length']:
            return 0.2  # Short queries tend to be simple
        else:
            return 0.5  # Medium length is neutral
    
    def _analyze_keywords(self, query_lower: str, factors: Dict[str, Any]) -> float:
        """Analyze keywords for complexity indicators"""
        # Count complex keywords
        complex_count = sum(1 for keyword in self.complex_keywords if keyword in query_lower)
        factors['complex_keywords'] = complex_count
        
        # Count simple keywords
        simple_count = sum(1 for keyword in self.simple_keywords if keyword in query_lower)
        factors['simple_keywords'] = simple_count
        
        # Calculate keyword score
        if complex_count >= self.config['complex_keyword_threshold']:
            return 0.9
        elif simple_count >= self.config['simple_keyword_threshold'] and complex_count == 0:
            return 0.1
        else:
            return 0.5
    
    def _analyze_patterns(self, query_lower: str, factors: Dict[str, Any]) -> float:
        """Analyze query patterns for complexity indicators"""
        pattern_matches = 0
        
        for pattern in self.complex_patterns:
            if re.search(pattern, query_lower):
                pattern_matches += 1
        
        factors['pattern_matches'] = pattern_matches
        
        if pattern_matches > 0:
            return min(0.95, 0.7 + (pattern_matches * 0.1))
        else:
            return 0.3
    
    def _analyze_context(self, conversation_context: Optional[List[str]], factors: Dict[str, Any]) -> float:
        """Analyze conversation context for complexity indicators"""
        if not conversation_context:
            factors['context_indicators'] = 0
            return 0.5
        
        # Look for context indicators in recent messages
        recent_messages = conversation_context[-3:] if len(conversation_context) > 3 else conversation_context
        context_indicators = 0
        
        for message in recent_messages:
            message_lower = message.lower()
            # Check if previous messages were complex
            if any(keyword in message_lower for keyword in self.complex_keywords):
                context_indicators += 1
        
        factors['context_indicators'] = context_indicators
        
        if context_indicators > 1:
            return 0.7  # Complex context suggests complex follow-up
        elif context_indicators == 1:
            return 0.6
        else:
            return 0.4
    
    def _generate_reasoning(self, complexity: QueryComplexity, factors: Dict[str, Any], score: float) -> str:
        """Generate human-readable reasoning for the classification"""
        reasons = []
        
        if factors['length'] > self.config['min_complex_length']:
            reasons.append(f"Long query ({factors['length']} chars)")
        elif factors['length'] < self.config['max_simple_length']:
            reasons.append(f"Short query ({factors['length']} chars)")
        
        if factors['complex_keywords'] > 0:
            reasons.append(f"{factors['complex_keywords']} technical keywords")
        
        if factors['simple_keywords'] > 0:
            reasons.append(f"{factors['simple_keywords']} simple keywords")
        
        if factors['pattern_matches'] > 0:
            reasons.append(f"{factors['pattern_matches']} complex patterns")
        
        if factors['context_indicators'] > 0:
            reasons.append(f"Complex context ({factors['context_indicators']} indicators)")
        
        base_reasoning = f"Classified as {complexity.value} (score: {score:.2f})"
        
        if reasons:
            return f"{base_reasoning}: {', '.join(reasons)}"
        else:
            return base_reasoning

class LLMRouter:
    """Multi-model LLM router with intelligent query classification"""
    
    def __init__(self, config: Optional[RouterConfig] = None):
        self.config = config or get_router_config()
        self.classifier = QueryClassifier()
        
        # Initialize OpenRouter client with proper configuration
        self._initialize_client()
        
        # Initialize rate limiter
        self._initialize_rate_limiter()
        
        # Conversation context
        self.conversation_history = []
        self.max_context_length = self.config.max_conversation_history
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'model_usage': {},
            'avg_response_time': 0.0,
            'total_response_time': 0.0
        }
    
    def _initialize_client(self) -> None:
        """Initialize the OpenRouter client with proper configuration"""
        if not self.config.openrouter_api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable.")
        
        # Create custom HTTP client with connection pooling and timeouts
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=self.config.connection_timeout,
                read=self.config.read_timeout,
                write=self.config.connection_timeout,
                pool=self.config.connection_timeout
            ),
            limits=httpx.Limits(
                max_connections=self.config.max_connections,
                max_keepalive_connections=self.config.max_keepalive_connections,
                keepalive_expiry=self.config.keepalive_expiry
            )
        )
        
        # Initialize OpenAI client with custom headers
        self.openrouter_client = AsyncOpenAI(
            api_key=self.config.openrouter_api_key,
            base_url=self.config.openrouter_base_url,
            http_client=http_client,
            default_headers={
                "HTTP-Referer": self.config.openrouter_site_url,
                "X-Title": self.config.openrouter_app_name,
            }
        )
        
        logger.info("OpenRouter client initialized with custom configuration")
    
    def _initialize_rate_limiter(self) -> None:
        """Initialize rate limiter for each model"""
        if not self.config.enable_rate_limiting:
            self.rate_limiter = None
            logger.info("Rate limiting disabled")
            return
        
        self.rate_limiter = ModelRateLimiter()
        
        # Add rate limiter for each model
        for model_type, model_config in self.config.models.items():
            rate_config = RateLimitConfig(
                requests_per_minute=model_config.max_requests_per_minute,
                tokens_per_minute=model_config.max_tokens_per_minute,
                burst_requests=min(10, model_config.max_requests_per_minute // 6),
                burst_tokens=min(10000, model_config.max_tokens_per_minute // 6),
                strategy=RateLimitStrategy.TOKEN_BUCKET
            )
            self.rate_limiter.add_model_limiter(model_config.id, rate_config)
        
        logger.info(f"Rate limiter initialized for {len(self.config.models)} models")
    
    async def route_query(self, query: str, system_prompt: str = None) -> Dict[str, Any]:
        """
        Route a query to the appropriate LLM model
        
        Args:
            query: User's query
            system_prompt: Optional system prompt for context
            
        Returns:
            Dict containing response, model used, timing, and classification info
        """
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        # Classify the query
        classification = self.classifier.classify_query(query, self.conversation_history)
        
        # Select model based on classification
        model_type = 'deep' if classification.complexity == QueryComplexity.COMPLEX else 'fast'
        model_config = self.config.models[model_type]
        
        # Log the routing decision
        logger.info(f"Query classified as {classification.complexity.value} "
                   f"(confidence: {classification.confidence:.2f}) -> {model_config.name}")
        logger.debug(f"Classification reasoning: {classification.reasoning}")
        
        # Apply rate limiting
        if self.rate_limiter:
            estimated_tokens = estimate_token_count(query) + estimate_response_tokens(model_config.max_tokens)
            try:
                await self.rate_limiter.acquire(model_config.id, estimated_tokens)
            except Exception as e:
                logger.warning(f"Rate limiting error: {e}")
        
        try:
            # Generate response
            response = await self._generate_response(query, model_config, system_prompt)
            
            # Update conversation history
            if self.config.include_conversation_context:
                self.conversation_history.append(query)
                self.conversation_history.append(response)
                
                # Trim history if too long
                if len(self.conversation_history) > self.max_context_length:
                    self.conversation_history = self.conversation_history[-self.max_context_length:]
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Update statistics
            self.stats['successful_requests'] += 1
            self.stats['total_response_time'] += total_time
            self.stats['avg_response_time'] = self.stats['total_response_time'] / self.stats['successful_requests']
            
            # Update model usage stats
            if model_config.id not in self.stats['model_usage']:
                self.stats['model_usage'][model_config.id] = 0
            self.stats['model_usage'][model_config.id] += 1
            
            # Estimate cost
            estimated_tokens = estimate_token_count(query + response)
            estimated_cost = (estimated_tokens / 1000) * model_config.cost_per_1k_tokens
            self.stats['total_tokens'] += estimated_tokens
            self.stats['total_cost'] += estimated_cost
            
            return {
                'response': response,
                'model_used': model_config.name,
                'model_id': model_config.id,
                'model_type': model_type,
                'classification': classification,
                'timing': {
                    'total_time': total_time,
                    'classification_time': 0.01,  # Minimal overhead
                    'generation_time': total_time - 0.01
                },
                'tokens': {
                    'estimated_total': estimated_tokens,
                    'estimated_cost': estimated_cost
                },
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error routing query with {model_config.name}: {str(e)}")
            self.stats['failed_requests'] += 1
            
            return {
                'response': f"I apologize, but I encountered an error processing your request: {str(e)}",
                'model_used': model_config.name,
                'model_id': model_config.id,
                'model_type': model_type,
                'classification': classification,
                'timing': {
                    'total_time': time.time() - start_time,
                    'classification_time': 0.01,
                    'generation_time': 0
                },
                'success': False,
                'error': str(e)
            }
    
    async def _generate_response(self, query: str, model_config: ModelConfig, system_prompt: str = None) -> str:
        """Generate response using the specified model"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation context
        if self.config.include_conversation_context:
            context_messages = []
            for i in range(0, len(self.conversation_history), 2):
                if i + 1 < len(self.conversation_history):
                    context_messages.append({"role": "user", "content": self.conversation_history[i]})
                    context_messages.append({"role": "assistant", "content": self.conversation_history[i + 1]})
            
            # Use configured max context messages
            max_context = self.config.max_context_messages
            messages.extend(context_messages[-max_context:])
        
        messages.append({"role": "user", "content": query})
        
        # Log request if enabled
        if self.config.log_requests:
            logger.debug(f"Making request to {model_config.name} with {len(messages)} messages")
        
        # Make API call with timeout
        response = await asyncio.wait_for(
            self.openrouter_client.chat.completions.create(
                model=model_config.id,
                messages=messages,
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature
            ),
            timeout=model_config.timeout
        )
        
        # Log response if enabled
        if self.config.log_responses:
            logger.debug(f"Received response from {model_config.name}: {len(response.choices[0].message.content)} chars")
        
        return response.choices[0].message.content
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive router statistics"""
        stats = {
            'conversation_length': len(self.conversation_history),
            'models_configured': len(self.config.models),
            'classifier_config': self.classifier.config,
            'performance': self.stats.copy(),
            'configuration': {
                'rate_limiting_enabled': self.config.enable_rate_limiting,
                'max_conversation_history': self.config.max_conversation_history,
                'include_conversation_context': self.config.include_conversation_context,
                'base_url': self.config.openrouter_base_url,
                'app_name': self.config.openrouter_app_name
            },
            'models': {
                model_type: {
                    'id': model_config.id,
                    'name': model_config.name,
                    'max_tokens': model_config.max_tokens,
                    'timeout': model_config.timeout,
                    'cost_per_1k_tokens': model_config.cost_per_1k_tokens,
                    'max_requests_per_minute': model_config.max_requests_per_minute,
                    'max_tokens_per_minute': model_config.max_tokens_per_minute
                }
                for model_type, model_config in self.config.models.items()
            }
        }
        
        # Add rate limiter stats if enabled
        if self.rate_limiter:
            stats['rate_limiting'] = self.rate_limiter.get_stats()
        
        return stats
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        try:
            if hasattr(self.openrouter_client, 'http_client'):
                await self.openrouter_client.http_client.aclose()
            logger.info("Router cleanup completed")
        except Exception as e:
            logger.warning(f"Error during router cleanup: {e}")
    
    def reset_stats(self) -> None:
        """Reset performance statistics"""
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'model_usage': {},
            'avg_response_time': 0.0,
            'total_response_time': 0.0
        }
        logger.info("Router statistics reset") 