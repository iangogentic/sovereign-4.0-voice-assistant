import asyncio
import time
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    requests_per_minute: int = 60
    tokens_per_minute: int = 60000
    burst_requests: int = 10
    burst_tokens: int = 10000
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET

@dataclass
class TokenBucket:
    """Token bucket for rate limiting"""
    capacity: int
    tokens: float
    refill_rate: float  # tokens per second
    last_refill: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Initialize tokens to capacity"""
        if self.tokens is None:
            self.tokens = self.capacity
    
    def refill(self) -> None:
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def can_consume(self, tokens: int) -> bool:
        """Check if we can consume the specified number of tokens"""
        self.refill()
        return self.tokens >= tokens
    
    def consume(self, tokens: int) -> bool:
        """Consume tokens if available"""
        if self.can_consume(tokens):
            self.tokens -= tokens
            return True
        return False
    
    def time_until_available(self, tokens: int) -> float:
        """Calculate time until specified tokens are available"""
        self.refill()
        if self.tokens >= tokens:
            return 0.0
        
        needed_tokens = tokens - self.tokens
        return needed_tokens / self.refill_rate

@dataclass
class SlidingWindow:
    """Sliding window for rate limiting"""
    window_size: int = 60  # seconds
    max_requests: int = 60
    max_tokens: int = 60000
    requests: list = field(default_factory=list)
    tokens: list = field(default_factory=list)
    
    def _cleanup_old_entries(self) -> None:
        """Remove entries older than the window size"""
        now = time.time()
        cutoff = now - self.window_size
        
        self.requests = [req for req in self.requests if req > cutoff]
        self.tokens = [(timestamp, token_count) for timestamp, token_count in self.tokens if timestamp > cutoff]
    
    def can_make_request(self, token_count: int = 0) -> bool:
        """Check if we can make a request"""
        self._cleanup_old_entries()
        
        # Check request limit
        if len(self.requests) >= self.max_requests:
            return False
        
        # Check token limit
        total_tokens = sum(token_count for _, token_count in self.tokens)
        if total_tokens + token_count > self.max_tokens:
            return False
        
        return True
    
    def record_request(self, token_count: int = 0) -> None:
        """Record a request"""
        now = time.time()
        self.requests.append(now)
        if token_count > 0:
            self.tokens.append((now, token_count))
    
    def time_until_available(self, token_count: int = 0) -> float:
        """Calculate time until request is available"""
        self._cleanup_old_entries()
        
        if self.can_make_request(token_count):
            return 0.0
        
        # Calculate time until oldest request/token expires
        now = time.time()
        times_to_wait = []
        
        # Check request limit
        if len(self.requests) >= self.max_requests:
            oldest_request = min(self.requests)
            times_to_wait.append(oldest_request + self.window_size - now)
        
        # Check token limit
        total_tokens = sum(token_count for _, token_count in self.tokens)
        if total_tokens + token_count > self.max_tokens:
            # Find the oldest tokens that need to expire
            sorted_tokens = sorted(self.tokens, key=lambda x: x[0])
            tokens_to_free = total_tokens + token_count - self.max_tokens
            
            for timestamp, token_count in sorted_tokens:
                tokens_to_free -= token_count
                if tokens_to_free <= 0:
                    times_to_wait.append(timestamp + self.window_size - now)
                    break
        
        return max(times_to_wait) if times_to_wait else 0.0

class RateLimiter:
    """Rate limiter for API requests"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.strategy = config.strategy
        
        if self.strategy == RateLimitStrategy.TOKEN_BUCKET:
            self.request_bucket = TokenBucket(
                capacity=config.burst_requests,
                tokens=config.burst_requests,
                refill_rate=config.requests_per_minute / 60.0
            )
            self.token_bucket = TokenBucket(
                capacity=config.burst_tokens,
                tokens=config.burst_tokens,
                refill_rate=config.tokens_per_minute / 60.0
            )
        else:
            self.sliding_window = SlidingWindow(
                window_size=60,
                max_requests=config.requests_per_minute,
                max_tokens=config.tokens_per_minute
            )
    
    async def acquire(self, token_count: int = 0) -> None:
        """Acquire permission to make a request"""
        if self.strategy == RateLimitStrategy.TOKEN_BUCKET:
            await self._acquire_token_bucket(token_count)
        else:
            await self._acquire_sliding_window(token_count)
    
    async def _acquire_token_bucket(self, token_count: int) -> None:
        """Acquire using token bucket strategy"""
        while True:
            # Check if we can make the request
            can_request = self.request_bucket.can_consume(1)
            can_tokens = token_count == 0 or self.token_bucket.can_consume(token_count)
            
            if can_request and can_tokens:
                # Consume tokens
                self.request_bucket.consume(1)
                if token_count > 0:
                    self.token_bucket.consume(token_count)
                logger.debug(f"Rate limit acquired: 1 request, {token_count} tokens")
                return
            
            # Calculate wait time
            request_wait = self.request_bucket.time_until_available(1)
            token_wait = self.token_bucket.time_until_available(token_count) if token_count > 0 else 0
            wait_time = max(request_wait, token_wait)
            
            logger.debug(f"Rate limit hit, waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
    
    async def _acquire_sliding_window(self, token_count: int) -> None:
        """Acquire using sliding window strategy"""
        while True:
            if self.sliding_window.can_make_request(token_count):
                self.sliding_window.record_request(token_count)
                logger.debug(f"Rate limit acquired: 1 request, {token_count} tokens")
                return
            
            wait_time = self.sliding_window.time_until_available(token_count)
            logger.debug(f"Rate limit hit, waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        if self.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return {
                'strategy': 'token_bucket',
                'request_tokens': self.request_bucket.tokens,
                'request_capacity': self.request_bucket.capacity,
                'token_tokens': self.token_bucket.tokens,
                'token_capacity': self.token_bucket.capacity,
                'config': {
                    'requests_per_minute': self.config.requests_per_minute,
                    'tokens_per_minute': self.config.tokens_per_minute,
                    'burst_requests': self.config.burst_requests,
                    'burst_tokens': self.config.burst_tokens
                }
            }
        else:
            return {
                'strategy': 'sliding_window',
                'current_requests': len(self.sliding_window.requests),
                'current_tokens': sum(token_count for _, token_count in self.sliding_window.tokens),
                'max_requests': self.sliding_window.max_requests,
                'max_tokens': self.sliding_window.max_tokens,
                'config': {
                    'requests_per_minute': self.config.requests_per_minute,
                    'tokens_per_minute': self.config.tokens_per_minute,
                    'window_size': self.sliding_window.window_size
                }
            }

class ModelRateLimiter:
    """Rate limiter that manages multiple models"""
    
    def __init__(self):
        self.limiters: Dict[str, RateLimiter] = {}
        self.global_limiter: Optional[RateLimiter] = None
    
    def add_model_limiter(self, model_id: str, config: RateLimitConfig) -> None:
        """Add a rate limiter for a specific model"""
        self.limiters[model_id] = RateLimiter(config)
        logger.info(f"Added rate limiter for model {model_id}")
    
    def set_global_limiter(self, config: RateLimitConfig) -> None:
        """Set a global rate limiter that applies to all models"""
        self.global_limiter = RateLimiter(config)
        logger.info("Set global rate limiter")
    
    async def acquire(self, model_id: str, token_count: int = 0) -> None:
        """Acquire permission to make a request for a specific model"""
        # Apply global limit first
        if self.global_limiter:
            await self.global_limiter.acquire(token_count)
        
        # Apply model-specific limit
        if model_id in self.limiters:
            await self.limiters[model_id].acquire(token_count)
        
        logger.debug(f"Rate limit acquired for model {model_id}: {token_count} tokens")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all rate limiters"""
        stats = {
            'global': self.global_limiter.get_stats() if self.global_limiter else None,
            'models': {
                model_id: limiter.get_stats()
                for model_id, limiter in self.limiters.items()
            }
        }
        return stats

# Estimate token count for rate limiting
def estimate_token_count(text: str) -> int:
    """Estimate token count for text (rough approximation)"""
    # Simple estimation: ~4 characters per token
    # This is a rough approximation, real tokenization is more complex
    return len(text) // 4

def estimate_response_tokens(max_tokens: int) -> int:
    """Estimate response tokens based on max_tokens parameter"""
    # Assume we'll use most of the available tokens
    return int(max_tokens * 0.8) 