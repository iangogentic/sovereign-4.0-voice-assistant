"""
Advanced Connection Pool Manager for Sovereign 4.0
Implements connection pooling, pre-warming, and keep-alive strategies for OpenAI Realtime API
Optimized for sub-300ms response times with intelligent connection management
"""

import asyncio
import ssl
import time
import json
import logging
import httpx
import websockets
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
import statistics
import threading
from queue import Queue, Empty


class CircuitState(Enum):
    """Circuit breaker states for connection health management"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests due to failures
    HALF_OPEN = "half_open"  # Testing recovery


class ConnectionType(Enum):
    """Types of connections managed"""
    WEBSOCKET_REALTIME = "websocket_realtime"
    HTTP_API = "http_api"
    HTTP_EMBEDDINGS = "http_embeddings"


@dataclass
class ConnectionPoolConfig:
    """Configuration for connection pool optimization"""
    # Pool sizing
    max_connections: int = 100
    max_keepalive_connections: int = 20
    min_pool_size: int = 2
    max_pool_size: int = 10
    
    # Timing configuration
    keepalive_expiry: float = 30.0
    connection_timeout: float = 30.0
    ping_interval: int = 20
    ping_timeout: int = 10
    
    # Retry and circuit breaker
    retries: int = 3
    backoff_factor: float = 0.5
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    
    # Pre-warming configuration
    pre_warm_enabled: bool = True
    pre_warm_size: int = 3
    predictive_scaling: bool = True
    context_aware_warming: bool = True


@dataclass
class ConnectionMetrics:
    """Comprehensive connection performance metrics"""
    # Pool metrics
    pool_hits: int = 0
    pool_misses: int = 0
    connections_created: int = 0
    connections_reused: int = 0
    connections_closed: int = 0
    
    # Performance metrics
    average_connection_time: float = 0.0
    average_response_time: float = 0.0
    connection_success_rate: float = 100.0
    keep_alive_success_rate: float = 100.0
    
    # Circuit breaker metrics
    circuit_breaker_trips: int = 0
    circuit_recovery_count: int = 0
    
    # Usage patterns
    peak_concurrent_connections: int = 0
    total_requests: int = 0
    session_start: datetime = field(default_factory=datetime.now)
    
    def get_reuse_rate(self) -> float:
        """Calculate connection reuse rate"""
        if self.connections_created == 0:
            return 0.0
        return (self.connections_reused / max(self.connections_created, 1)) * 100
    
    def get_pool_efficiency(self) -> float:
        """Calculate pool efficiency (hits vs total requests)"""
        total_attempts = self.pool_hits + self.pool_misses
        if total_attempts == 0:
            return 0.0
        return (self.pool_hits / total_attempts) * 100


class ConnectionCircuitBreaker:
    """Circuit breaker pattern for connection health management"""
    
    def __init__(self, config: ConnectionPoolConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = CircuitState.CLOSED
        self.state_lock = threading.Lock()
        
    async def call_with_circuit_breaker(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self.state_lock:
            current_state = self.state
            
        if current_state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.config.recovery_timeout:
                with self.state_lock:
                    self.state = CircuitState.HALF_OPEN
                self.logger.info("🔄 Circuit breaker transitioning to HALF_OPEN")
            else:
                raise ConnectionError("Circuit breaker is OPEN - connection unavailable")
        
        try:
            result = await func(*args, **kwargs)
            
            # Success - reset failure count if in HALF_OPEN
            if current_state == CircuitState.HALF_OPEN:
                with self.state_lock:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                self.logger.info("✅ Circuit breaker reset to CLOSED")
            
            return result
            
        except Exception as e:
            with self.state_lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
                    self.logger.warning(f"🚨 Circuit breaker opened after {self.failure_count} failures")
            
            raise e
    
    def get_state(self) -> CircuitState:
        """Get current circuit breaker state"""
        with self.state_lock:
            return self.state
    
    def get_failure_count(self) -> int:
        """Get current failure count"""
        with self.state_lock:
            return self.failure_count


class WebSocketKeepAlive:
    """WebSocket keep-alive mechanism with health monitoring"""
    
    def __init__(self, websocket: websockets.WebSocketServerProtocol, config: ConnectionPoolConfig, logger: Optional[logging.Logger] = None):
        self.websocket = websocket
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.last_pong = time.time()
        self.health_callbacks: List[Callable[[bool], None]] = []
        
    def add_health_callback(self, callback: Callable[[bool], None]):
        """Add callback for health status changes"""
        self.health_callbacks.append(callback)
        
    async def start_heartbeat(self):
        """Start heartbeat monitoring"""
        if self.heartbeat_task is not None:
            return
        
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.logger.debug("💓 WebSocket heartbeat started")
        
    async def stop_heartbeat(self):
        """Stop heartbeat monitoring"""
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
            self.heartbeat_task = None
            self.logger.debug("💓 WebSocket heartbeat stopped")
    
    async def _heartbeat_loop(self):
        """Continuous heartbeat monitoring loop"""
        while not self.websocket.closed:
            try:
                # Send ping
                pong_waiter = await self.websocket.ping()
                
                # Wait for pong with timeout
                try:
                    await asyncio.wait_for(pong_waiter, timeout=self.config.ping_timeout)
                    self.last_pong = time.time()
                    
                    # Notify health callbacks of healthy connection
                    for callback in self.health_callbacks:
                        try:
                            callback(True)
                        except Exception as e:
                            self.logger.error(f"❌ Health callback error: {e}")
                            
                except asyncio.TimeoutError:
                    self.logger.warning("⚠️ WebSocket ping timeout - connection may be stale")
                    
                    # Notify health callbacks of unhealthy connection
                    for callback in self.health_callbacks:
                        try:
                            callback(False)
                        except Exception as e:
                            self.logger.error(f"❌ Health callback error: {e}")
                    break
                
                await asyncio.sleep(self.config.ping_interval)
                
            except websockets.exceptions.ConnectionClosed:
                self.logger.debug("🔌 WebSocket connection closed during heartbeat")
                break
            except Exception as e:
                self.logger.error(f"❌ Heartbeat error: {e}")
                break
    
    def is_connection_healthy(self) -> bool:
        """Check if connection is healthy based on heartbeat"""
        return (time.time() - self.last_pong) < (self.config.ping_interval * 2)


class RealtimeConnectionPool:
    """Advanced WebSocket connection pool for OpenAI Realtime API"""
    
    def __init__(self, api_key: str, config: ConnectionPoolConfig, logger: Optional[logging.Logger] = None):
        self.api_key = api_key
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Connection storage
        self.available_connections: List[websockets.WebSocketServerProtocol] = []
        self.active_connections: Dict[str, Tuple[websockets.WebSocketServerProtocol, WebSocketKeepAlive]] = {}
        self.connection_lock = asyncio.Lock()
        
        # Management tasks
        self.pre_warm_task: Optional[asyncio.Task] = None
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Metrics and monitoring
        self.metrics = ConnectionMetrics()
        self.circuit_breaker = ConnectionCircuitBreaker(config, logger)
        
        # SSL context optimization
        self.ssl_context = self._create_optimized_ssl_context()
        
        # Connection affinity for session stickiness
        self.user_affinity: Dict[str, str] = {}  # user_id -> session_id
        
    def _create_optimized_ssl_context(self) -> ssl.SSLContext:
        """Create optimized SSL context for performance"""
        context = ssl.create_default_context()
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        
        # Performance optimizations
        context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
        context.options |= ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3 | ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1
        
        return context
    
    async def initialize(self) -> bool:
        """Initialize the connection pool with pre-warming"""
        try:
            self.is_running = True
            
            if self.config.pre_warm_enabled:
                self.pre_warm_task = asyncio.create_task(self._pre_warm_connections())
            
            self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
            
            self.logger.info(f"✅ Connection pool initialized (pre-warm: {self.config.pre_warm_enabled})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize connection pool: {e}")
            return False
    
    async def shutdown(self):
        """Graceful shutdown of connection pool"""
        try:
            self.is_running = False
            
            # Cancel management tasks
            if self.pre_warm_task:
                self.pre_warm_task.cancel()
            if self.health_monitor_task:
                self.health_monitor_task.cancel()
            
            # Close all connections
            async with self.connection_lock:
                # Close available connections
                for connection in self.available_connections:
                    if not connection.closed:
                        await connection.close()
                
                # Close active connections
                for session_id, (connection, keep_alive) in self.active_connections.items():
                    await keep_alive.stop_heartbeat()
                    if not connection.closed:
                        await connection.close()
                
                self.available_connections.clear()
                self.active_connections.clear()
            
            self.logger.info("✅ Connection pool shutdown completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error during connection pool shutdown: {e}")
    
    async def _pre_warm_connections(self):
        """Pre-warm connection pool with ready connections"""
        while self.is_running:
            try:
                async with self.connection_lock:
                    current_available = len(self.available_connections)
                    
                if current_available < self.config.pre_warm_size:
                    needed = self.config.pre_warm_size - current_available
                    self.logger.debug(f"🔥 Pre-warming {needed} connections")
                    
                    # Create connections in parallel
                    tasks = [self._create_connection() for _ in range(needed)]
                    connections = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    async with self.connection_lock:
                        for conn in connections:
                            if isinstance(conn, websockets.WebSocketServerProtocol) and not conn.closed:
                                self.available_connections.append(conn)
                                self.metrics.connections_created += 1
                
                # Wait before next pre-warm check
                await asyncio.sleep(10)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Pre-warm error: {e}")
                await asyncio.sleep(5)
    
    async def _create_connection(self) -> Optional[websockets.WebSocketServerProtocol]:
        """Create optimized WebSocket connection"""
        start_time = time.time()
        
        try:
            url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1",
                "User-Agent": "Sovereign-4.0-Realtime/1.0"
            }
            
            # Optimized connection parameters for low latency
            connection = await websockets.connect(
                url,
                extra_headers=headers,
                ssl=self.ssl_context,
                ping_interval=self.config.ping_interval,
                ping_timeout=self.config.ping_timeout,
                close_timeout=5,
                max_size=2**20,  # 1MB max message size
                compression=None,  # Disable compression for lower latency
                write_limit=2**16,  # 64KB write buffer
                read_limit=2**16   # 64KB read buffer
            )
            
            connection_time = time.time() - start_time
            self._update_average_connection_time(connection_time)
            
            self.logger.debug(f"🔗 Connection created in {connection_time:.3f}s")
            return connection
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create connection: {e}")
            return None
    
    async def get_connection(self, session_id: str, user_id: Optional[str] = None) -> Optional[Tuple[websockets.WebSocketServerProtocol, WebSocketKeepAlive]]:
        """Get connection from pool with session affinity"""
        try:
            return await self.circuit_breaker.call_with_circuit_breaker(
                self._get_connection_internal, session_id, user_id
            )
        except Exception as e:
            self.logger.error(f"❌ Failed to get connection: {e}")
            return None
    
    async def _get_connection_internal(self, session_id: str, user_id: Optional[str] = None) -> Optional[Tuple[websockets.WebSocketServerProtocol, WebSocketKeepAlive]]:
        """Internal connection retrieval with pooling"""
        async with self.connection_lock:
            # Check for existing session connection
            if session_id in self.active_connections:
                connection, keep_alive = self.active_connections[session_id]
                if keep_alive.is_connection_healthy() and not connection.closed:
                    self.metrics.connections_reused += 1
                    return connection, keep_alive
                else:
                    # Clean up unhealthy connection
                    await self._cleanup_connection(session_id)
            
            # Try to get from available pool
            if self.available_connections:
                connection = self.available_connections.pop()
                
                # Verify connection health
                if not connection.closed:
                    keep_alive = WebSocketKeepAlive(connection, self.config, self.logger)
                    await keep_alive.start_heartbeat()
                    
                    self.active_connections[session_id] = (connection, keep_alive)
                    self.metrics.pool_hits += 1
                    self.metrics.connections_reused += 1
                    
                    # Update affinity if user_id provided
                    if user_id:
                        self.user_affinity[user_id] = session_id
                    
                    self.logger.debug(f"🎯 Connection from pool for session {session_id}")
                    return connection, keep_alive
                else:
                    await connection.close()
            
            # Create new connection if pool is empty
            self.metrics.pool_misses += 1
            connection = await self._create_connection()
            
            if connection:
                keep_alive = WebSocketKeepAlive(connection, self.config, self.logger)
                await keep_alive.start_heartbeat()
                
                self.active_connections[session_id] = (connection, keep_alive)
                
                # Update affinity if user_id provided
                if user_id:
                    self.user_affinity[user_id] = session_id
                
                self.logger.debug(f"🆕 New connection created for session {session_id}")
                return connection, keep_alive
            
            return None
    
    async def return_connection(self, session_id: str):
        """Return connection to pool for reuse"""
        async with self.connection_lock:
            if session_id in self.active_connections:
                connection, keep_alive = self.active_connections.pop(session_id)
                
                # Check if connection is healthy and pool has space
                if (keep_alive.is_connection_healthy() and 
                    not connection.closed and 
                    len(self.available_connections) < self.config.max_pool_size):
                    
                    # Stop heartbeat but keep connection alive
                    await keep_alive.stop_heartbeat()
                    self.available_connections.append(connection)
                    self.logger.debug(f"♻️ Connection returned to pool from session {session_id}")
                else:
                    await self._cleanup_connection_objects(connection, keep_alive)
                    self.logger.debug(f"🗑️ Connection closed from session {session_id}")
    
    async def _cleanup_connection(self, session_id: str):
        """Clean up connection and remove from active connections"""
        if session_id in self.active_connections:
            connection, keep_alive = self.active_connections.pop(session_id)
            await self._cleanup_connection_objects(connection, keep_alive)
    
    async def _cleanup_connection_objects(self, connection: websockets.WebSocketServerProtocol, keep_alive: WebSocketKeepAlive):
        """Clean up connection and keep-alive objects"""
        try:
            await keep_alive.stop_heartbeat()
            if not connection.closed:
                await connection.close()
            self.metrics.connections_closed += 1
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up connection: {e}")
    
    async def _health_monitor_loop(self):
        """Monitor connection pool health and cleanup stale connections"""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                async with self.connection_lock:
                    # Check available connections
                    healthy_available = []
                    for connection in self.available_connections:
                        if not connection.closed:
                            healthy_available.append(connection)
                        else:
                            self.metrics.connections_closed += 1
                    
                    self.available_connections = healthy_available
                    
                    # Check active connections
                    stale_sessions = []
                    for session_id, (connection, keep_alive) in self.active_connections.items():
                        if connection.closed or not keep_alive.is_connection_healthy():
                            stale_sessions.append(session_id)
                    
                    # Clean up stale connections
                    for session_id in stale_sessions:
                        await self._cleanup_connection(session_id)
                
                # Update peak concurrent connections
                current_active = len(self.active_connections)
                if current_active > self.metrics.peak_concurrent_connections:
                    self.metrics.peak_concurrent_connections = current_active
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Health monitor error: {e}")
    
    def _update_average_connection_time(self, connection_time: float):
        """Update average connection establishment time"""
        if self.metrics.connections_created == 0:
            self.metrics.average_connection_time = connection_time
        else:
            self.metrics.average_connection_time = (
                (self.metrics.average_connection_time * (self.metrics.connections_created - 1) + connection_time) 
                / self.metrics.connections_created
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive connection pool metrics"""
        return {
            "pool_efficiency": self.metrics.get_pool_efficiency(),
            "reuse_rate": self.metrics.get_reuse_rate(),
            "average_connection_time": self.metrics.average_connection_time,
            "active_connections": len(self.active_connections),
            "available_connections": len(self.available_connections),
            "peak_concurrent": self.metrics.peak_concurrent_connections,
            "circuit_breaker_state": self.circuit_breaker.get_state().value,
            "circuit_failure_count": self.circuit_breaker.get_failure_count(),
            "total_created": self.metrics.connections_created,
            "total_reused": self.metrics.connections_reused,
            "total_closed": self.metrics.connections_closed,
            "pool_hits": self.metrics.pool_hits,
            "pool_misses": self.metrics.pool_misses,
            "uptime_seconds": (datetime.now() - self.metrics.session_start).total_seconds()
        }


class HTTPConnectionPool:
    """HTTP connection pool for traditional OpenAI API endpoints"""
    
    def __init__(self, api_key: str, config: ConnectionPoolConfig, logger: Optional[logging.Logger] = None):
        self.api_key = api_key
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Create persistent HTTP client with optimized settings
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=config.max_connections,
                max_keepalive_connections=config.max_keepalive_connections,
                keepalive_expiry=config.keepalive_expiry
            ),
            timeout=httpx.Timeout(config.connection_timeout),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Sovereign-4.0-HTTP/1.0"
            },
            http2=True  # Enable HTTP/2 for multiplexing
        )
        
        self.metrics = ConnectionMetrics()
        self.circuit_breaker = ConnectionCircuitBreaker(config, logger)
        
    async def make_request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make HTTP request with connection pooling and circuit breaker"""
        return await self.circuit_breaker.call_with_circuit_breaker(
            self._make_request_internal, method, url, **kwargs
        )
    
    async def _make_request_internal(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Internal request method with metrics tracking"""
        start_time = time.time()
        
        try:
            # Add keep-alive headers
            headers = kwargs.get('headers', {})
            headers.update({
                'Connection': 'keep-alive',
                'Keep-Alive': f'timeout={int(self.config.keepalive_expiry)}, max=100'
            })
            kwargs['headers'] = headers
            
            response = await self.client.request(method, url, **kwargs)
            
            # Update metrics
            response_time = time.time() - start_time
            self.metrics.total_requests += 1
            self._update_average_response_time(response_time)
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ HTTP request failed: {e}")
            raise
    
    def _update_average_response_time(self, response_time: float):
        """Update average response time"""
        if self.metrics.total_requests == 1:
            self.metrics.average_response_time = response_time
        else:
            self.metrics.average_response_time = (
                (self.metrics.average_response_time * (self.metrics.total_requests - 1) + response_time) 
                / self.metrics.total_requests
            )
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get HTTP connection metrics"""
        return {
            "total_requests": self.metrics.total_requests,
            "average_response_time": self.metrics.average_response_time,
            "circuit_breaker_state": self.circuit_breaker.get_state().value,
            "circuit_failure_count": self.circuit_breaker.get_failure_count()
        }


class ConnectionPoolManager:
    """Main connection pool manager coordinating all connection types"""
    
    def __init__(self, api_key: str, config: Optional[ConnectionPoolConfig] = None, logger: Optional[logging.Logger] = None):
        self.api_key = api_key
        self.config = config or ConnectionPoolConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Connection pools
        self.realtime_pool: Optional[RealtimeConnectionPool] = None
        self.http_pool: Optional[HTTPConnectionPool] = None
        
        # Management
        self.is_initialized = False
        self.metrics_aggregator_task: Optional[asyncio.Task] = None
        
    async def initialize(self) -> bool:
        """Initialize all connection pools"""
        try:
            # Initialize Realtime WebSocket pool
            self.realtime_pool = RealtimeConnectionPool(self.api_key, self.config, self.logger)
            if not await self.realtime_pool.initialize():
                return False
            
            # Initialize HTTP pool
            self.http_pool = HTTPConnectionPool(self.api_key, self.config, self.logger)
            
            # Start metrics aggregation
            self.metrics_aggregator_task = asyncio.create_task(self._metrics_aggregation_loop())
            
            self.is_initialized = True
            self.logger.info("✅ Connection pool manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize connection pool manager: {e}")
            return False
    
    async def shutdown(self):
        """Graceful shutdown of all connection pools"""
        try:
            self.is_initialized = False
            
            # Stop metrics aggregation
            if self.metrics_aggregator_task:
                self.metrics_aggregator_task.cancel()
            
            # Shutdown pools
            if self.realtime_pool:
                await self.realtime_pool.shutdown()
            
            if self.http_pool:
                await self.http_pool.close()
            
            self.logger.info("✅ Connection pool manager shutdown completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error during connection pool manager shutdown: {e}")
    
    async def get_realtime_connection(self, session_id: str, user_id: Optional[str] = None) -> Optional[Tuple[websockets.WebSocketServerProtocol, WebSocketKeepAlive]]:
        """Get WebSocket connection for Realtime API"""
        if not self.realtime_pool:
            raise RuntimeError("Realtime connection pool not initialized")
        
        return await self.realtime_pool.get_connection(session_id, user_id)
    
    async def return_realtime_connection(self, session_id: str):
        """Return WebSocket connection to pool"""
        if self.realtime_pool:
            await self.realtime_pool.return_connection(session_id)
    
    async def make_http_request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make HTTP request using connection pool"""
        if not self.http_pool:
            raise RuntimeError("HTTP connection pool not initialized")
        
        return await self.http_pool.make_request(method, url, **kwargs)
    
    async def _metrics_aggregation_loop(self):
        """Aggregate and log metrics from all pools"""
        while self.is_initialized:
            try:
                await asyncio.sleep(60)  # Log metrics every minute
                
                metrics = self.get_comprehensive_metrics()
                self.logger.info(f"📊 Connection Metrics: {json.dumps(metrics, indent=2)}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Metrics aggregation error: {e}")
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from all connection pools"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "realtime_pool": {},
            "http_pool": {},
            "overall": {
                "pools_active": 0,
                "total_connections": 0
            }
        }
        
        if self.realtime_pool:
            metrics["realtime_pool"] = self.realtime_pool.get_metrics()
            metrics["overall"]["pools_active"] += 1
            metrics["overall"]["total_connections"] += metrics["realtime_pool"]["active_connections"]
        
        if self.http_pool:
            metrics["http_pool"] = self.http_pool.get_metrics()
            metrics["overall"]["pools_active"] += 1
        
        return metrics


# Factory function for easy creation
def create_connection_pool_manager(
    api_key: str,
    max_connections: int = 100,
    max_pool_size: int = 10,
    pre_warm_enabled: bool = True,
    logger: Optional[logging.Logger] = None
) -> ConnectionPoolManager:
    """Create optimized connection pool manager with common configuration"""
    
    config = ConnectionPoolConfig(
        max_connections=max_connections,
        max_pool_size=max_pool_size,
        pre_warm_enabled=pre_warm_enabled,
        predictive_scaling=True,
        context_aware_warming=True
    )
    
    return ConnectionPoolManager(api_key, config, logger) 