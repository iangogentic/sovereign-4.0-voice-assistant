"""
Realtime API Structured Logging for Sovereign 4.0

Provides comprehensive structured logging for OpenAI Realtime API integration:
- JSON-formatted log entries for machine parsing
- Event correlation and tracing
- Performance metrics logging
- Error context capture
- Cost tracking events
- Security audit trails

Integrates with log aggregation systems:
- ELK Stack (Elasticsearch, Logstash, Kibana)
- Splunk
- Fluentd
- AWS CloudWatch
- Google Cloud Logging

Log formats designed for production deployment infrastructure.
"""

import json
import logging
import time
import uuid
import traceback
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
import threading
from contextlib import contextmanager

# Standard library logging
import logging.handlers


class LogLevel(Enum):
    """Structured log levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EventType(Enum):
    """Types of realtime events to log"""
    CONNECTION_ESTABLISHED = "connection_established"
    CONNECTION_CLOSED = "connection_closed"
    CONNECTION_ERROR = "connection_error"
    VOICE_INPUT_RECEIVED = "voice_input_received"
    VOICE_OUTPUT_GENERATED = "voice_output_generated"
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    API_REQUEST = "api_request"
    API_RESPONSE = "api_response"
    API_ERROR = "api_error"
    COST_ALERT = "cost_alert"
    PERFORMANCE_ALERT = "performance_alert"
    HEALTH_CHECK = "health_check"
    SECURITY_EVENT = "security_event"
    SYSTEM_EVENT = "system_event"


@dataclass
class LogContext:
    """Context information for log correlation"""
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    correlation_id: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance-related metrics for logging"""
    latency_ms: Optional[float] = None
    tokens_processed: Optional[int] = None
    cost_usd: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None


@dataclass
class StructuredLogEntry:
    """Structured log entry format"""
    timestamp: str
    level: str
    event_type: str
    message: str
    service: str = "sovereign-realtime-api"
    version: str = "4.0.0"
    environment: str = "production"
    
    # Context and correlation
    context: Optional[LogContext] = None
    
    # Performance metrics
    performance: Optional[PerformanceMetrics] = None
    
    # Additional structured data
    data: Optional[Dict[str, Any]] = None
    
    # Error information
    error: Optional[Dict[str, Any]] = None
    
    # Source location
    source: Optional[Dict[str, str]] = None


class RealtimeStructuredLogger:
    """
    Structured logger for Realtime API events
    
    Provides JSON-formatted logging with correlation IDs, performance metrics,
    and structured data for integration with log aggregation systems.
    """
    
    def __init__(self,
                 logger_name: str = "sovereign.realtime",
                 log_level: LogLevel = LogLevel.INFO,
                 service_name: str = "sovereign-realtime-api",
                 service_version: str = "4.0.0",
                 environment: str = "production",
                 enable_console: bool = True,
                 enable_file: bool = True,
                 log_file_path: str = "/app/logs/realtime-structured.log",
                 enable_syslog: bool = False,
                 syslog_address: str = "localhost:514",
                 enable_json_stdout: bool = True):
        
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        
        # Thread-local storage for context
        self._local = threading.local()
        
        # Create logger
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(getattr(logging, log_level.value.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup formatters and handlers
        self._setup_formatters()
        self._setup_handlers(
            enable_console=enable_console,
            enable_file=enable_file,
            log_file_path=log_file_path,
            enable_syslog=enable_syslog,
            syslog_address=syslog_address,
            enable_json_stdout=enable_json_stdout
        )
        
        # Initialize correlation tracking
        self._init_correlation_tracking()
        
        # Log initialization
        self.info(
            event_type=EventType.SYSTEM_EVENT,
            message="Structured logger initialized",
            data={"logger_name": logger_name, "log_level": log_level.value}
        )
    
    def _setup_formatters(self):
        """Setup log formatters"""
        # JSON formatter for structured logs
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                # Extract structured data from record
                if hasattr(record, 'structured_data'):
                    return json.dumps(record.structured_data, default=str)
                else:
                    # Fallback to standard JSON format
                    return json.dumps({
                        'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                        'level': record.levelname.lower(),
                        'message': record.getMessage(),
                        'logger': record.name,
                        'module': record.module,
                        'function': record.funcName,
                        'line': record.lineno
                    })
        
        # Human-readable formatter for console
        class HumanFormatter(logging.Formatter):
            def format(self, record):
                if hasattr(record, 'structured_data'):
                    data = record.structured_data
                    timestamp = data.get('timestamp', '')
                    level = data.get('level', '').upper()
                    event_type = data.get('event_type', '')
                    message = data.get('message', '')
                    
                    # Add context if available
                    context_str = ""
                    if data.get('context') and data['context'].get('session_id'):
                        context_str = f" [session:{data['context']['session_id'][:8]}]"
                    
                    return f"{timestamp} {level:8} {event_type:20} {message}{context_str}"
                else:
                    return super().format(record)
        
        self.json_formatter = JSONFormatter()
        self.human_formatter = HumanFormatter()
    
    def _setup_handlers(self,
                       enable_console: bool,
                       enable_file: bool,
                       log_file_path: str,
                       enable_syslog: bool,
                       syslog_address: str,
                       enable_json_stdout: bool):
        """Setup log handlers"""
        
        # Console handler (human-readable)
        if enable_console and not enable_json_stdout:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self.human_formatter)
            self.logger.addHandler(console_handler)
        
        # JSON stdout handler (for container environments)
        if enable_json_stdout:
            json_handler = logging.StreamHandler()
            json_handler.setFormatter(self.json_formatter)
            self.logger.addHandler(json_handler)
        
        # File handler with rotation
        if enable_file:
            try:
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file_path,
                    maxBytes=100 * 1024 * 1024,  # 100MB
                    backupCount=10
                )
                file_handler.setFormatter(self.json_formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                # If file logging fails, log to stderr
                self.logger.error(f"Failed to setup file logging: {e}")
        
        # Syslog handler (for centralized logging)
        if enable_syslog:
            try:
                if ':' in syslog_address:
                    host, port = syslog_address.split(':')
                    syslog_handler = logging.handlers.SysLogHandler(
                        address=(host, int(port))
                    )
                else:
                    syslog_handler = logging.handlers.SysLogHandler(
                        address=syslog_address
                    )
                syslog_handler.setFormatter(self.json_formatter)
                self.logger.addHandler(syslog_handler)
            except Exception as e:
                self.logger.error(f"Failed to setup syslog: {e}")
    
    def _init_correlation_tracking(self):
        """Initialize correlation tracking"""
        self._correlation_stack = []
    
    def _get_context(self) -> Optional[LogContext]:
        """Get current thread-local context"""
        return getattr(self._local, 'context', None)
    
    def _set_context(self, context: LogContext):
        """Set thread-local context"""
        self._local.context = context
    
    @contextmanager
    def context(self, **context_kwargs):
        """Context manager for log correlation"""
        # Save current context
        old_context = self._get_context()
        
        # Create new context
        new_context = LogContext(**context_kwargs)
        self._set_context(new_context)
        
        try:
            yield new_context
        finally:
            # Restore old context
            self._set_context(old_context)
    
    def _create_log_entry(self,
                         level: LogLevel,
                         event_type: EventType,
                         message: str,
                         performance: Optional[PerformanceMetrics] = None,
                         data: Optional[Dict[str, Any]] = None,
                         error: Optional[Exception] = None,
                         include_source: bool = True) -> StructuredLogEntry:
        """Create a structured log entry"""
        
        # Get current context
        context = self._get_context()
        
        # Create error info if exception provided
        error_info = None
        if error:
            error_info = {
                'type': type(error).__name__,
                'message': str(error),
                'traceback': traceback.format_exc() if level in [LogLevel.ERROR, LogLevel.CRITICAL] else None
            }
        
        # Get source information
        source_info = None
        if include_source:
            frame = traceback.extract_stack()[-3]  # Skip this method and the calling log method
            source_info = {
                'file': frame.filename,
                'function': frame.name,
                'line': str(frame.lineno)
            }
        
        return StructuredLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=level.value,
            event_type=event_type.value,
            message=message,
            service=self.service_name,
            version=self.service_version,
            environment=self.environment,
            context=context,
            performance=performance,
            data=data,
            error=error_info,
            source=source_info
        )
    
    def _log_entry(self, log_entry: StructuredLogEntry):
        """Log a structured entry"""
        # Convert to dict for JSON serialization
        entry_dict = asdict(log_entry)
        
        # Remove None values to keep logs clean
        entry_dict = {k: v for k, v in entry_dict.items() if v is not None}
        
        # Create log record
        log_record = logging.LogRecord(
            name=self.logger.name,
            level=getattr(logging, log_entry.level.upper()),
            pathname="",
            lineno=0,
            msg=log_entry.message,
            args=(),
            exc_info=None
        )
        
        # Attach structured data
        log_record.structured_data = entry_dict
        
        # Log through standard logging system
        self.logger.handle(log_record)
    
    # Public logging methods
    def debug(self,
              event_type: EventType,
              message: str,
              performance: Optional[PerformanceMetrics] = None,
              data: Optional[Dict[str, Any]] = None,
              error: Optional[Exception] = None):
        """Log debug message"""
        entry = self._create_log_entry(LogLevel.DEBUG, event_type, message, performance, data, error)
        self._log_entry(entry)
    
    def info(self,
             event_type: EventType,
             message: str,
             performance: Optional[PerformanceMetrics] = None,
             data: Optional[Dict[str, Any]] = None,
             error: Optional[Exception] = None):
        """Log info message"""
        entry = self._create_log_entry(LogLevel.INFO, event_type, message, performance, data, error)
        self._log_entry(entry)
    
    def warning(self,
                event_type: EventType,
                message: str,
                performance: Optional[PerformanceMetrics] = None,
                data: Optional[Dict[str, Any]] = None,
                error: Optional[Exception] = None):
        """Log warning message"""
        entry = self._create_log_entry(LogLevel.WARNING, event_type, message, performance, data, error)
        self._log_entry(entry)
    
    def error(self,
              event_type: EventType,
              message: str,
              performance: Optional[PerformanceMetrics] = None,
              data: Optional[Dict[str, Any]] = None,
              error: Optional[Exception] = None):
        """Log error message"""
        entry = self._create_log_entry(LogLevel.ERROR, event_type, message, performance, data, error)
        self._log_entry(entry)
    
    def critical(self,
                 event_type: EventType,
                 message: str,
                 performance: Optional[PerformanceMetrics] = None,
                 data: Optional[Dict[str, Any]] = None,
                 error: Optional[Exception] = None):
        """Log critical message"""
        entry = self._create_log_entry(LogLevel.CRITICAL, event_type, message, performance, data, error)
        self._log_entry(entry)
    
    # Convenience methods for common events
    def log_connection_event(self,
                           event_type: EventType,
                           session_id: str,
                           user_id: Optional[str] = None,
                           details: Optional[Dict[str, Any]] = None):
        """Log connection-related events"""
        with self.context(session_id=session_id, user_id=user_id):
            self.info(
                event_type=event_type,
                message=f"WebSocket {event_type.value}",
                data=details
            )
    
    def log_voice_event(self,
                       event_type: EventType,
                       session_id: str,
                       user_id: Optional[str] = None,
                       duration_ms: Optional[float] = None,
                       tokens: Optional[int] = None,
                       cost: Optional[float] = None,
                       details: Optional[Dict[str, Any]] = None):
        """Log voice processing events"""
        performance = PerformanceMetrics(
            latency_ms=duration_ms,
            tokens_processed=tokens,
            cost_usd=cost
        )
        
        with self.context(session_id=session_id, user_id=user_id):
            self.info(
                event_type=event_type,
                message=f"Voice {event_type.value}",
                performance=performance,
                data=details
            )
    
    def log_api_event(self,
                     event_type: EventType,
                     endpoint: str,
                     method: str,
                     status_code: Optional[int] = None,
                     duration_ms: Optional[float] = None,
                     request_id: Optional[str] = None,
                     error: Optional[Exception] = None,
                     details: Optional[Dict[str, Any]] = None):
        """Log API request/response events"""
        performance = PerformanceMetrics(latency_ms=duration_ms)
        
        api_data = {
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            **(details or {})
        }
        
        with self.context(request_id=request_id):
            if error:
                self.error(
                    event_type=event_type,
                    message=f"API {method} {endpoint} failed",
                    performance=performance,
                    data=api_data,
                    error=error
                )
            else:
                self.info(
                    event_type=event_type,
                    message=f"API {method} {endpoint} completed",
                    performance=performance,
                    data=api_data
                )
    
    def log_cost_event(self,
                      session_id: str,
                      user_id: Optional[str] = None,
                      cost_usd: float = 0.0,
                      tokens_used: int = 0,
                      alert_threshold: Optional[float] = None,
                      details: Optional[Dict[str, Any]] = None):
        """Log cost tracking events"""
        performance = PerformanceMetrics(
            tokens_processed=tokens_used,
            cost_usd=cost_usd
        )
        
        cost_data = {
            'cost_usd': cost_usd,
            'tokens_used': tokens_used,
            'alert_threshold': alert_threshold,
            **(details or {})
        }
        
        event_type = EventType.COST_ALERT if alert_threshold and cost_usd > alert_threshold else EventType.SYSTEM_EVENT
        level = LogLevel.WARNING if event_type == EventType.COST_ALERT else LogLevel.INFO
        
        with self.context(session_id=session_id, user_id=user_id):
            if level == LogLevel.WARNING:
                self.warning(
                    event_type=event_type,
                    message=f"Cost alert: ${cost_usd:.2f} exceeds threshold ${alert_threshold:.2f}",
                    performance=performance,
                    data=cost_data
                )
            else:
                self.info(
                    event_type=event_type,
                    message=f"Cost tracking update: ${cost_usd:.2f}",
                    performance=performance,
                    data=cost_data
                )
    
    def log_performance_event(self,
                            session_id: str,
                            latency_ms: float,
                            threshold_ms: Optional[float] = None,
                            user_id: Optional[str] = None,
                            operation: str = "voice_processing",
                            details: Optional[Dict[str, Any]] = None):
        """Log performance monitoring events"""
        performance = PerformanceMetrics(latency_ms=latency_ms)
        
        perf_data = {
            'operation': operation,
            'latency_ms': latency_ms,
            'threshold_ms': threshold_ms,
            **(details or {})
        }
        
        is_alert = threshold_ms and latency_ms > threshold_ms
        event_type = EventType.PERFORMANCE_ALERT if is_alert else EventType.SYSTEM_EVENT
        level = LogLevel.WARNING if is_alert else LogLevel.INFO
        
        with self.context(session_id=session_id, user_id=user_id):
            if is_alert:
                self.warning(
                    event_type=event_type,
                    message=f"Performance alert: {operation} took {latency_ms:.2f}ms (threshold: {threshold_ms:.2f}ms)",
                    performance=performance,
                    data=perf_data
                )
            else:
                self.debug(
                    event_type=event_type,
                    message=f"Performance tracking: {operation} completed in {latency_ms:.2f}ms",
                    performance=performance,
                    data=perf_data
                )
    
    def log_health_check(self,
                        check_name: str,
                        status: str,
                        duration_ms: float,
                        details: Optional[Dict[str, Any]] = None):
        """Log health check events"""
        performance = PerformanceMetrics(latency_ms=duration_ms)
        
        health_data = {
            'check_name': check_name,
            'status': status,
            **(details or {})
        }
        
        level = LogLevel.ERROR if status != "healthy" else LogLevel.DEBUG
        
        if level == LogLevel.ERROR:
            self.error(
                event_type=EventType.HEALTH_CHECK,
                message=f"Health check failed: {check_name} - {status}",
                performance=performance,
                data=health_data
            )
        else:
            self.debug(
                event_type=EventType.HEALTH_CHECK,
                message=f"Health check passed: {check_name}",
                performance=performance,
                data=health_data
            )
    
    def log_security_event(self,
                          event_description: str,
                          severity: LogLevel = LogLevel.WARNING,
                          user_id: Optional[str] = None,
                          ip_address: Optional[str] = None,
                          details: Optional[Dict[str, Any]] = None):
        """Log security-related events"""
        security_data = {
            'user_id': user_id,
            'ip_address': ip_address,
            'severity': severity.value,
            **(details or {})
        }
        
        with self.context(user_id=user_id):
            if severity == LogLevel.CRITICAL:
                self.critical(
                    event_type=EventType.SECURITY_EVENT,
                    message=f"Critical security event: {event_description}",
                    data=security_data
                )
            elif severity == LogLevel.ERROR:
                self.error(
                    event_type=EventType.SECURITY_EVENT,
                    message=f"Security event: {event_description}",
                    data=security_data
                )
            else:
                self.warning(
                    event_type=EventType.SECURITY_EVENT,
                    message=f"Security notice: {event_description}",
                    data=security_data
                )


# Global logger instance
_realtime_logger: Optional[RealtimeStructuredLogger] = None


def get_realtime_logger(
    environment: str = "production",
    enable_json_stdout: bool = True
) -> RealtimeStructuredLogger:
    """Get or create the global realtime logger instance"""
    global _realtime_logger
    
    if _realtime_logger is None:
        _realtime_logger = RealtimeStructuredLogger(
            environment=environment,
            enable_json_stdout=enable_json_stdout
        )
    
    return _realtime_logger


# Convenience functions for common logging patterns
def log_session_start(session_id: str, user_id: Optional[str] = None, **details):
    """Log session start event"""
    logger = get_realtime_logger()
    logger.log_connection_event(
        EventType.SESSION_STARTED,
        session_id=session_id,
        user_id=user_id,
        details=details
    )


def log_session_end(session_id: str, user_id: Optional[str] = None, **details):
    """Log session end event"""
    logger = get_realtime_logger()
    logger.log_connection_event(
        EventType.SESSION_ENDED,
        session_id=session_id,
        user_id=user_id,
        details=details
    )


def log_voice_processing(session_id: str, duration_ms: float, tokens: int, cost: float, **details):
    """Log voice processing event"""
    logger = get_realtime_logger()
    logger.log_voice_event(
        EventType.VOICE_OUTPUT_GENERATED,
        session_id=session_id,
        duration_ms=duration_ms,
        tokens=tokens,
        cost=cost,
        details=details
    )


def log_api_call(endpoint: str, method: str, duration_ms: float, status_code: int, **details):
    """Log API call event"""
    logger = get_realtime_logger()
    logger.log_api_event(
        EventType.API_RESPONSE,
        endpoint=endpoint,
        method=method,
        duration_ms=duration_ms,
        status_code=status_code,
        details=details
    ) 