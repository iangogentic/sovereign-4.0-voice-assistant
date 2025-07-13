"""
Structured Logging System for Sovereign Voice Assistant
Optimized for voice AI applications with context tracking and performance metrics
"""

import logging
import json
import time
import uuid
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from contextvars import ContextVar
from contextlib import asynccontextmanager
import asyncio
from pathlib import Path

# Import from error_handling for context variables
from .error_handling import request_id_var, user_session_var, operation_var

@dataclass
class VoiceAILogContext:
    """Structured context for voice AI logging"""
    request_id: str
    user_session: str
    operation: str
    service_name: str
    timestamp: float
    duration_ms: Optional[float] = None
    audio_duration_ms: Optional[float] = None
    text_length: Optional[int] = None
    model_used: Optional[str] = None
    quality_metrics: Optional[Dict[str, float]] = None
    error_category: Optional[str] = None
    retry_count: int = 0
    circuit_breaker_state: Optional[str] = None
    fallback_tier: Optional[str] = None
    correlation_id: Optional[str] = None
    additional_context: Optional[Dict[str, Any]] = None

class VoiceAILogger:
    """Structured logger optimized for voice AI applications"""
    
    def __init__(self, name: str, level: int = logging.INFO, log_dir: str = "logs"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Configure structured JSON formatter
        formatter = StructuredFormatter()
        
        # Console handler with colored output
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColoredConsoleFormatter())
        self.logger.addHandler(console_handler)
        
        # File handler for persistent structured logging
        file_handler = logging.FileHandler(log_path / f'{name}.jsonl')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Separate error log file
        error_handler = logging.FileHandler(log_path / f'{name}_errors.jsonl')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)
        
        # Performance metrics handler
        perf_handler = logging.FileHandler(log_path / f'{name}_performance.jsonl')
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(formatter)
        perf_handler.addFilter(PerformanceMetricsFilter())
        self.logger.addHandler(perf_handler)
    
    def log_request_start(
        self,
        operation: str,
        service_name: str,
        audio_duration_ms: Optional[float] = None,
        user_session: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log the start of a voice AI request"""
        request_id = str(uuid.uuid4())
        request_id_var.set(request_id)
        operation_var.set(operation)
        
        if user_session:
            user_session_var.set(user_session)
        
        context = VoiceAILogContext(
            request_id=request_id,
            user_session=user_session or user_session_var.get(),
            operation=operation,
            service_name=service_name,
            timestamp=time.time(),
            audio_duration_ms=audio_duration_ms,
            correlation_id=str(uuid.uuid4()),
            additional_context=additional_context
        )
        
        log_data = asdict(context)
        log_data['event'] = 'request_start'
        
        self.logger.info("🚀 Request started", extra={"structured_data": log_data})
        return request_id
    
    def log_request_success(
        self,
        duration_ms: float,
        text_length: Optional[int] = None,
        model_used: Optional[str] = None,
        quality_metrics: Optional[Dict[str, float]] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ):
        """Log successful request completion"""
        context = VoiceAILogContext(
            request_id=request_id_var.get(),
            user_session=user_session_var.get(),
            operation=operation_var.get(),
            service_name="",  # Will be filled by service
            timestamp=time.time(),
            duration_ms=duration_ms,
            text_length=text_length,
            model_used=model_used,
            quality_metrics=quality_metrics,
            additional_context=additional_context
        )
        
        log_data = asdict(context)
        log_data['event'] = 'request_success'
        
        self.logger.info("✅ Request completed successfully", extra={"structured_data": log_data})
    
    def log_request_error(
        self,
        error: Exception,
        duration_ms: float,
        retry_count: int = 0,
        circuit_breaker_state: Optional[str] = None,
        fallback_tier: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ):
        """Log request error with full context"""
        from .error_handling import VoiceAIException
        
        error_category = None
        if isinstance(error, VoiceAIException):
            error_category = error.category.value
        
        context = VoiceAILogContext(
            request_id=request_id_var.get(),
            user_session=user_session_var.get(),
            operation=operation_var.get(),
            service_name="",
            timestamp=time.time(),
            duration_ms=duration_ms,
            error_category=error_category,
            retry_count=retry_count,
            circuit_breaker_state=circuit_breaker_state,
            fallback_tier=fallback_tier,
            additional_context=additional_context
        )
        
        log_data = asdict(context)
        log_data.update({
            'event': 'request_error',
            'error_message': str(error),
            'error_type': type(error).__name__,
            'traceback': logging.Formatter().formatException((type(error), error, error.__traceback__))
        })
        
        self.logger.error("❌ Request failed", extra={"structured_data": log_data})
    
    def log_performance_metrics(
        self,
        operation: str,
        metrics: Dict[str, float],
        additional_context: Optional[Dict[str, Any]] = None
    ):
        """Log performance metrics"""
        log_data = {
            'event': 'performance_metrics',
            'request_id': request_id_var.get(),
            'user_session': user_session_var.get(),
            'operation': operation,
            'timestamp': time.time(),
            'metrics': metrics
        }
        
        if additional_context:
            log_data.update(additional_context)
        
        self.logger.info("📊 Performance metrics", extra={"structured_data": log_data})
    
    def log_user_feedback(
        self,
        message: str,
        feedback_type: str = "info",
        additional_context: Optional[Dict[str, Any]] = None
    ):
        """Log user feedback messages"""
        log_data = {
            'event': 'user_feedback',
            'request_id': request_id_var.get(),
            'user_session': user_session_var.get(),
            'operation': operation_var.get(),
            'timestamp': time.time(),
            'message': message,
            'feedback_type': feedback_type
        }
        
        if additional_context:
            log_data.update(additional_context)
        
        emoji = "🔄" if feedback_type == "degradation" else "ℹ️"
        self.logger.info(f"{emoji} {message}", extra={"structured_data": log_data})
    
    def log_circuit_breaker_event(
        self,
        service_name: str,
        event_type: str,
        health_score: float,
        additional_context: Optional[Dict[str, Any]] = None
    ):
        """Log circuit breaker state changes"""
        log_data = {
            'event': 'circuit_breaker_event',
            'request_id': request_id_var.get(),
            'user_session': user_session_var.get(),
            'operation': operation_var.get(),
            'timestamp': time.time(),
            'service_name': service_name,
            'event_type': event_type,
            'health_score': health_score
        }
        
        if additional_context:
            log_data.update(additional_context)
        
        emoji = "🔓" if event_type == "opened" else "🔒" if event_type == "closed" else "🔄"
        self.logger.warning(f"{emoji} Circuit breaker {event_type}: {service_name}", extra={"structured_data": log_data})
    
    def log_fallback_event(
        self,
        operation: str,
        from_tier: str,
        to_tier: str,
        reason: str,
        additional_context: Optional[Dict[str, Any]] = None
    ):
        """Log fallback events"""
        log_data = {
            'event': 'fallback_event',
            'request_id': request_id_var.get(),
            'user_session': user_session_var.get(),
            'operation': operation,
            'timestamp': time.time(),
            'from_tier': from_tier,
            'to_tier': to_tier,
            'reason': reason
        }
        
        if additional_context:
            log_data.update(additional_context)
        
        self.logger.info(f"🔄 Fallback: {operation} {from_tier} → {to_tier}", extra={"structured_data": log_data})
    
    # Standard logger interface methods for compatibility
    def debug(self, message: str, *args, **kwargs):
        """Standard debug logging"""
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Standard info logging"""
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Standard warning logging"""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Standard error logging"""
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Standard critical logging"""
        self.logger.critical(message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs):
        """Standard exception logging"""
        self.logger.exception(message, *args, **kwargs)
    
    def log(self, level: int, message: str, *args, **kwargs):
        """Standard log method"""
        self.logger.log(level, message, *args, **kwargs)

class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": record.created,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process
        }
        
        # Add structured data if present
        if hasattr(record, 'structured_data'):
            log_data.update(record.structured_data)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)

class ColoredConsoleFormatter(logging.Formatter):
    """Colored console formatter for better readability"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format timestamp
        timestamp = time.strftime('%H:%M:%S', time.localtime(record.created))
        
        # Extract key information from structured data
        context_info = ""
        if hasattr(record, 'structured_data'):
            data = record.structured_data
            
            # Add request ID if present
            if 'request_id' in data:
                context_info += f" [{data['request_id'][:8]}]"
            
            # Add operation if present
            if 'operation' in data:
                context_info += f" {data['operation']}"
            
            # Add duration if present
            if 'duration_ms' in data:
                context_info += f" ({data['duration_ms']:.1f}ms)"
            
            # Add model if present
            if 'model_used' in data:
                context_info += f" [{data['model_used']}]"
        
        return f"{color}{timestamp}{reset} {record.getMessage()}{context_info}"

class PerformanceMetricsFilter(logging.Filter):
    """Filter to only log performance metrics"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, 'structured_data'):
            return record.structured_data.get('event') == 'performance_metrics'
        return False

# Context manager for request tracking
@asynccontextmanager
async def voice_ai_request_context(
    operation: str,
    service_name: str,
    logger: VoiceAILogger,
    user_session: Optional[str] = None,
    audio_duration_ms: Optional[float] = None,
    additional_context: Optional[Dict[str, Any]] = None
):
    """Context manager for tracking voice AI requests"""
    request_id = logger.log_request_start(
        operation, 
        service_name, 
        audio_duration_ms,
        user_session,
        additional_context
    )
    start_time = time.time()
    
    try:
        yield request_id
        
        duration_ms = (time.time() - start_time) * 1000
        logger.log_request_success(duration_ms, additional_context=additional_context)
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.log_request_error(e, duration_ms, additional_context=additional_context)
        raise

# Helper functions for common logging patterns
def log_audio_processing(
    logger: VoiceAILogger,
    audio_duration_ms: float,
    processing_time_ms: float,
    quality_score: Optional[float] = None
):
    """Log audio processing metrics"""
    metrics = {
        'audio_duration_ms': audio_duration_ms,
        'processing_time_ms': processing_time_ms,
        'processing_ratio': processing_time_ms / audio_duration_ms if audio_duration_ms > 0 else 0
    }
    
    if quality_score is not None:
        metrics['quality_score'] = quality_score
    
    logger.log_performance_metrics('audio_processing', metrics)

def log_model_inference(
    logger: VoiceAILogger,
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    inference_time_ms: float,
    quality_metrics: Optional[Dict[str, float]] = None
):
    """Log model inference metrics"""
    metrics = {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': input_tokens + output_tokens,
        'inference_time_ms': inference_time_ms,
        'tokens_per_second': (input_tokens + output_tokens) / (inference_time_ms / 1000) if inference_time_ms > 0 else 0
    }
    
    if quality_metrics:
        metrics.update(quality_metrics)
    
    logger.log_performance_metrics('model_inference', metrics, {'model_name': model_name})

def log_network_request(
    logger: VoiceAILogger,
    service_name: str,
    endpoint: str,
    method: str,
    status_code: int,
    response_time_ms: float,
    request_size_bytes: Optional[int] = None,
    response_size_bytes: Optional[int] = None
):
    """Log network request metrics"""
    metrics = {
        'status_code': status_code,
        'response_time_ms': response_time_ms,
        'success': 200 <= status_code < 300
    }
    
    if request_size_bytes is not None:
        metrics['request_size_bytes'] = request_size_bytes
    
    if response_size_bytes is not None:
        metrics['response_size_bytes'] = response_size_bytes
    
    additional_context = {
        'service_name': service_name,
        'endpoint': endpoint,
        'method': method
    }
    
    logger.log_performance_metrics('network_request', metrics, additional_context)

# Global logger instance
_global_logger: Optional[VoiceAILogger] = None

def get_voice_ai_logger(name: str = "sovereign_assistant") -> VoiceAILogger:
    """Get or create global voice AI logger"""
    global _global_logger
    if _global_logger is None:
        _global_logger = VoiceAILogger(name)
    return _global_logger

def set_voice_ai_logger(logger: VoiceAILogger):
    """Set global voice AI logger"""
    global _global_logger
    _global_logger = logger 