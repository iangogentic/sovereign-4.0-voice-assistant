"""
Multi-Model LLM RouterBlock for Pipecat Integration

This module provides a RouterBlock that integrates with Pipecat framework to route
queries between different LLM models based on complexity analysis.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.frames.frames import (
    Frame,
    TextFrame,
    LLMMessagesFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    UserImageRequestFrame,
    BotSpeakingFrame,
    BotStoppedSpeakingFrame,
    SystemFrame,
    ErrorFrame
)

from .llm_router import LLMRouter, QueryClassifier, QueryComplexity, QueryClassification
from .router_config import RouterConfig, get_router_config
from .async_processor import AsyncProcessor, ProcessingConfig, ProcessingState
from .fallback_manager import (
    FallbackManager, CircuitBreakerConfig, RetryConfig, FallbackConfig,
    ServiceUnavailableError, AllServicesFailed
)

logger = logging.getLogger(__name__)


class RouterState(Enum):
    """RouterBlock processing states"""
    IDLE = "idle"
    CLASSIFYING = "classifying"
    ROUTING = "routing"
    PROCESSING_FAST = "processing_fast"
    PROCESSING_DEEP = "processing_deep"
    ERROR = "error"


@dataclass
class RouterMetrics:
    """Metrics for router performance tracking"""
    total_requests: int = 0
    fast_requests: int = 0
    deep_requests: int = 0
    avg_classification_time: float = 0.0
    avg_routing_time: float = 0.0
    successful_routes: int = 0
    failed_routes: int = 0
    fallback_routes: int = 0
    classification_count: int = 0
    routing_count: int = 0
    
    def update_classification_time(self, time_ms: float):
        """Update average classification time"""
        self.avg_classification_time = (
            (self.avg_classification_time * self.classification_count + time_ms) /
            (self.classification_count + 1)
        )
        self.classification_count += 1
    
    def update_routing_time(self, time_ms: float):
        """Update average routing time"""
        self.avg_routing_time = (
            (self.avg_routing_time * self.routing_count + time_ms) /
            (self.routing_count + 1)
        )
        self.routing_count += 1


class RouterBlock(FrameProcessor):
    """
    RouterBlock integrates with Pipecat to route queries between fast and deep LLMs
    
    This processor:
    - Intercepts text input frames
    - Classifies queries using QueryClassifier
    - Routes to appropriate LLM service based on complexity
    - Manages multiple LLM service instances
    - Provides routing metrics and monitoring
    """
    
    def __init__(
        self,
        config: Optional[RouterConfig] = None,
        processing_config: Optional[ProcessingConfig] = None,
        fallback_config: Optional[FallbackConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        fast_llm_service: Optional[OpenRouterLLMService] = None,
        deep_llm_service: Optional[OpenRouterLLMService] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Configuration
        self.config = config or get_router_config()
        self.processing_config = processing_config or ProcessingConfig()
        
        # Initialize routing components
        self.classifier = QueryClassifier()
        self.llm_router = LLMRouter(self.config)
        
        # Initialize async processor
        self.async_processor = AsyncProcessor(self.processing_config)
        self.async_processor.set_status_update_callback(self._handle_async_status_update)
        
        # Initialize comprehensive fallback manager
        self.fallback_manager = FallbackManager(
            config=fallback_config,
            circuit_breaker_config=circuit_breaker_config,
            retry_config=retry_config
        )
        
        # LLM Services
        self.fast_llm_service = fast_llm_service or self._create_fast_llm_service()
        self.deep_llm_service = deep_llm_service or self._create_deep_llm_service()
        
        # State management
        self.state = RouterState.IDLE
        self.current_request_id = None
        self.pending_requests: Dict[str, Dict[str, Any]] = {}
        
        # Metrics
        self.metrics = RouterMetrics()
        
        # Conversation context
        self.conversation_history: List[Dict[str, str]] = []
        self.max_context_length = self.config.max_conversation_history
        
        logger.info("RouterBlock initialized with comprehensive fallback management and async processing capabilities")
    
    def _create_fast_llm_service(self) -> OpenRouterLLMService:
        """Create fast LLM service for simple queries"""
        fast_config = self.config.models['fast']
        return OpenRouterLLMService(
            model=fast_config.id,
            api_key=self.config.openrouter_api_key,
            base_url=self.config.openrouter_base_url
        )
    
    def _create_deep_llm_service(self) -> OpenRouterLLMService:
        """Create deep LLM service for complex queries"""
        deep_config = self.config.models['deep']
        return OpenRouterLLMService(
            model=deep_config.id,
            api_key=self.config.openrouter_api_key,
            base_url=self.config.openrouter_base_url
        )
    
    async def setup(self):
        """Setup the RouterBlock and initialize LLM services"""
        logger.info("Setting up RouterBlock...")
        
        # Initialize LLM services
        if hasattr(self.fast_llm_service, 'setup'):
            await self.fast_llm_service.setup()
        if hasattr(self.deep_llm_service, 'setup'):
            await self.deep_llm_service.setup()
        
        logger.info("RouterBlock setup complete")
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up RouterBlock...")
        
        # Cancel all async tasks
        self.async_processor.cancel_all_tasks()
        
        # Cleanup LLM services
        if hasattr(self.fast_llm_service, 'cleanup'):
            await self.fast_llm_service.cleanup()
        if hasattr(self.deep_llm_service, 'cleanup'):
            await self.deep_llm_service.cleanup()
        
        # Cleanup router resources
        if hasattr(self.llm_router, 'cleanup'):
            await self.llm_router.cleanup()
        
        logger.info("RouterBlock cleanup complete")
    
    async def process_frame(self, frame: Frame, direction: str):
        """Process frames through the routing logic"""
        # Handle text frames for routing
        if isinstance(frame, TextFrame):
            await self._handle_text_frame(frame, direction)
        
        # Handle LLM message frames
        elif isinstance(frame, LLMMessagesFrame):
            await self._handle_llm_messages_frame(frame, direction)
        
        # Handle system frames
        elif isinstance(frame, SystemFrame):
            await self._handle_system_frame(frame, direction)
        
        # Handle error frames
        elif isinstance(frame, ErrorFrame):
            await self._handle_error_frame(frame, direction)
        
        # Pass through other frames unchanged
        else:
            await self.push_frame(frame, direction)
    
    async def _handle_text_frame(self, frame: TextFrame, direction: str):
        """Handle text input frames for routing"""
        logger.debug(f"RouterBlock received text frame: {frame.text}")
        
        # Update state
        self.state = RouterState.CLASSIFYING
        self.metrics.total_requests += 1
        
        # Generate request ID
        request_id = f"req_{int(time.time() * 1000)}"
        self.current_request_id = request_id
        
        try:
            # Classify the query
            start_time = time.time()
            classification = self.classifier.classify_query(
                frame.text, 
                self.conversation_history
            )
            classification_time = (time.time() - start_time) * 1000
            self.metrics.update_classification_time(classification_time)
            
            # Store request context
            self.pending_requests[request_id] = {
                'text': frame.text,
                'classification': classification,
                'start_time': time.time(),
                'direction': direction
            }
            
            # Route to appropriate service
            await self._route_to_service(request_id, frame.text, classification, direction)
            
        except Exception as e:
            logger.error(f"Error in RouterBlock text processing: {e}")
            self.state = RouterState.ERROR
            self.metrics.failed_routes += 1
            
            # Push error frame
            error_frame = ErrorFrame(f"RouterBlock processing error: {str(e)}")
            await self.push_frame(error_frame, direction)
    
    async def _route_to_service(
        self, 
        request_id: str, 
        text: str, 
        classification: QueryClassification, 
        direction: str
    ):
        """Route query to appropriate LLM service with advanced async processing"""
        logger.info(f"Routing query (ID: {request_id}) - Complexity: {classification.complexity.value}, "
                   f"Confidence: {classification.confidence:.2f}")
        
        # Update routing state
        self.state = RouterState.ROUTING
        routing_start = time.time()
        
        # Determine model type and service
        model_type = 'deep' if classification.complexity == QueryComplexity.COMPLEX else 'fast'
        selected_service = self.deep_llm_service if model_type == 'deep' else self.fast_llm_service
        
        # Update state and metrics
        if model_type == 'deep':
            self.state = RouterState.PROCESSING_DEEP
            self.metrics.deep_requests += 1
        else:
            self.state = RouterState.PROCESSING_FAST
            self.metrics.fast_requests += 1
        
        # Update routing metrics
        routing_time = (time.time() - routing_start) * 1000
        self.metrics.update_routing_time(routing_time)
        
        # Store request context for callbacks
        self.pending_requests[request_id]['direction'] = direction
        self.pending_requests[request_id]['selected_service'] = selected_service
        
        try:
            # Create service functions for FallbackManager
            service_functions = {
                'fast': lambda: self._execute_with_service(text, classification, self.fast_llm_service, direction),
                'deep': lambda: self._execute_with_service(text, classification, self.deep_llm_service, direction),
                'backup': lambda: self._execute_with_service(text, classification, self.fast_llm_service, direction)  # Backup uses fast service
            }
            
            # Use FallbackManager for comprehensive fallback logic
            result = await self.fallback_manager.execute_with_fallback(
                primary_service=model_type,
                service_functions=service_functions,
                context={'request_id': request_id, 'classification': classification},
                status_callback=lambda msg: self._send_status_update(msg, direction)
            )
            
            # Update conversation history
            self.conversation_history.append({
                'role': 'user',
                'content': text,
                'timestamp': time.time(),
                'classification': classification.complexity.value
            })
            
            # Add response to conversation history
            if result and isinstance(result, str):
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': result,
                    'timestamp': time.time(),
                    'model_type': model_type
                })
            
            # Limit conversation history length
            if len(self.conversation_history) > self.max_context_length:
                self.conversation_history = self.conversation_history[-self.max_context_length:]
            
            self.metrics.successful_routes += 1
            self.state = RouterState.IDLE
            
            logger.info(f"Successfully processed request {request_id} with fallback-managed execution")
            
        except AllServicesFailed as e:
            logger.error(f"All services failed for request {request_id}: {e}")
            self.state = RouterState.ERROR
            self.metrics.failed_routes += 1
            
            # Send user-friendly error message
            await self._send_status_update(
                "I'm experiencing technical difficulties. Please try again in a moment.",
                direction
            )
            
        except ServiceUnavailableError as e:
            logger.error(f"Service unavailable for request {request_id}: {e}")
            self.state = RouterState.ERROR
            self.metrics.failed_routes += 1
            
            # Send user-friendly error message
            await self._send_status_update(
                "The service is temporarily unavailable. Please try again shortly.",
                direction
            )
            
        except Exception as e:
            logger.error(f"Unexpected error in routing for request {request_id}: {e}")
            self.state = RouterState.ERROR
            self.metrics.failed_routes += 1
            
            # Send generic error message
            await self._send_status_update(
                "I encountered an unexpected issue. Let me try again.",
                direction
            )
    
    async def _execute_with_service(self, text: str, classification: QueryClassification, service: OpenRouterLLMService, direction: str) -> str:
        """Execute request with specific LLM service"""
        # Create LLM messages frame
        messages = self._build_messages(text, classification)
        llm_frame = LLMMessagesFrame(messages)
        
        # Link the selected service to process the request
        self.link(service)
        
        # Push the frame to the selected service
        await service.push_frame(llm_frame, direction)
        
        # For now, return a mock response - in real implementation,
        # this would capture the actual LLM response
        return f"Processed: {text[:50]}..." if len(text) > 50 else f"Processed: {text}"
    
    def _build_messages(self, text: str, classification: QueryClassification) -> List[Dict[str, str]]:
        """Build messages array for LLM service"""
        messages = []
        
        # Add system prompt based on complexity
        if classification.complexity == QueryComplexity.COMPLEX:
            system_prompt = (
                "You are a knowledgeable assistant capable of deep analysis and reasoning. "
                "Provide comprehensive, detailed responses with examples and explanations. "
                "Take time to think through complex problems step by step."
            )
        else:
            system_prompt = (
                "You are a helpful assistant. Provide clear, concise responses. "
                "Be direct and to the point while remaining friendly and helpful."
            )
        
        messages.append({"role": "system", "content": system_prompt})
        
        # Add relevant conversation history
        recent_history = self.conversation_history[-3:]  # Last 3 exchanges
        for msg in recent_history:
            if msg['role'] in ['user', 'assistant']:
                messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
        
        # Add current user message
        messages.append({"role": "user", "content": text})
        
        return messages
    
    def _create_processor_func(self, text: str, classification: QueryClassification, service: OpenRouterLLMService, direction: str):
        """Create processor function for AsyncProcessor"""
        async def processor(query: str) -> str:
            # Create LLM messages frame
            messages = self._build_messages(text, classification)
            llm_frame = LLMMessagesFrame(messages)
            
            # Link the selected service to process the request
            self.link(service)
            
            # Push the frame to the selected service
            await service.push_frame(llm_frame, direction)
            
            # For now, return a mock response - in real implementation,
            # this would capture the actual LLM response
            return f"Processed: {query[:50]}..." if len(query) > 50 else f"Processed: {query}"
        
        return processor
    
    async def _handle_async_status_update(self, task_id: str, message: str):
        """Handle status updates from AsyncProcessor"""
        if task_id in self.pending_requests:
            direction = self.pending_requests[task_id].get('direction', 'downstream')
            await self._send_status_update(message, direction)
    
    async def _send_status_update(self, message: str, direction: str):
        """Send status update to user"""
        status_frame = TextFrame(message)
        await self.push_frame(status_frame, direction)
        
        # Also send speaking indicator
        speaking_frame = BotSpeakingFrame()
        await self.push_frame(speaking_frame, direction)
    
    async def _handle_llm_messages_frame(self, frame: LLMMessagesFrame, direction: str):
        """Handle LLM messages frame"""
        # This would typically come from LLM service responses
        # Pass through for now, but could add response post-processing here
        await self.push_frame(frame, direction)
    
    async def _handle_system_frame(self, frame: SystemFrame, direction: str):
        """Handle system frames"""
        # Pass through system frames
        await self.push_frame(frame, direction)
    
    async def _handle_error_frame(self, frame: ErrorFrame, direction: str):
        """Handle error frames"""
        logger.error(f"RouterBlock received error frame: {frame.error}")
        self.state = RouterState.ERROR
        await self.push_frame(frame, direction)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get routing metrics including async processing and fallback statistics"""
        async_stats = self.async_processor.get_statistics()
        fallback_health = self.fallback_manager.get_service_health()
        
        return {
            'state': self.state.value,
            'total_requests': self.metrics.total_requests,
            'fast_requests': self.metrics.fast_requests,
            'deep_requests': self.metrics.deep_requests,
            'successful_routes': self.metrics.successful_routes,
            'failed_routes': self.metrics.failed_routes,
            'fallback_routes': self.metrics.fallback_routes,
            'avg_classification_time_ms': self.metrics.avg_classification_time,
            'avg_routing_time_ms': self.metrics.avg_routing_time,
            'conversation_history_length': len(self.conversation_history),
            'async_processing': {
                'active_tasks': async_stats['active_tasks'],
                'completed_tasks': async_stats['completed_tasks'],
                'cancelled_tasks': async_stats['cancelled_tasks'],
                'failed_tasks': async_stats['failed_tasks'],
                'degraded_tasks': async_stats['degraded_tasks'],
                'avg_processing_time': async_stats['avg_processing_time'],
                'timeout_rate': async_stats['timeout_rate'],
                'success_rate': async_stats['success_rate']
            },
            'fallback_management': {
                'services': fallback_health['services'],
                'overall_stats': fallback_health['overall_stats']
            }
        }
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get detailed routing statistics"""
        total = self.metrics.total_requests
        if total == 0:
            return {'no_requests': True}
        
        return {
            'total_requests': total,
            'fast_percentage': (self.metrics.fast_requests / total) * 100,
            'deep_percentage': (self.metrics.deep_requests / total) * 100,
            'success_rate': (self.metrics.successful_routes / total) * 100,
            'failure_rate': (self.metrics.failed_routes / total) * 100,
            'fallback_rate': (self.metrics.fallback_routes / total) * 100,
            'avg_classification_time_ms': self.metrics.avg_classification_time,
            'avg_routing_time_ms': self.metrics.avg_routing_time
        }


def create_router_block(
    config: Optional[RouterConfig] = None, 
    processing_config: Optional[ProcessingConfig] = None,
    fallback_config: Optional[FallbackConfig] = None,
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    retry_config: Optional[RetryConfig] = None,
    **kwargs
) -> RouterBlock:
    """Factory function to create RouterBlock instance with comprehensive fallback management"""
    return RouterBlock(
        config=config, 
        processing_config=processing_config,
        fallback_config=fallback_config,
        circuit_breaker_config=circuit_breaker_config,
        retry_config=retry_config,
        **kwargs
    ) 