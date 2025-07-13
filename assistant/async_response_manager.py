"""
Async Response Manager for Sovereign 4.0
Implements immediate response streaming and concurrent audio processing
Optimized for zero-wait architecture with progressive audio delivery
"""

import asyncio
import time
import logging
import threading
import queue
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple, Union, AsyncGenerator
from datetime import datetime, timedelta
from enum import Enum
import weakref
from concurrent.futures import ThreadPoolExecutor
import json


class StreamMode(Enum):
    """Response streaming modes"""
    IMMEDIATE = "immediate"        # Start playback immediately with first chunk
    CHUNKED = "chunked"           # Stream in optimized chunks
    COMPLETE = "complete"         # Wait for complete response
    ADAPTIVE = "adaptive"         # Dynamically switch based on conditions


class ConcurrencyLevel(Enum):
    """Concurrency configuration levels"""
    LOW = "low"           # 2 workers, minimal resources
    MEDIUM = "medium"     # 4 workers, balanced performance
    HIGH = "high"         # 8+ workers, maximum throughput
    ADAPTIVE = "adaptive" # Dynamic adjustment


class ResponsePriority(Enum):
    """Response processing priorities"""
    CRITICAL = 0    # Immediate processing required
    HIGH = 1        # High priority processing
    NORMAL = 2      # Standard processing
    LOW = 3         # Background processing
    BATCH = 4       # Batch processing when resources available


@dataclass
class AsyncConfig:
    """Configuration for async processing and streaming"""
    # Streaming configuration
    stream_mode: StreamMode = StreamMode.IMMEDIATE
    first_chunk_target_ms: float = 50.0  # Target time for first chunk
    chunk_size_bytes: int = 1024         # Base chunk size
    adaptive_chunk_sizing: bool = True
    
    # Concurrency configuration
    concurrency_level: ConcurrencyLevel = ConcurrencyLevel.MEDIUM
    max_concurrent_streams: int = 10
    worker_pool_size: int = 4
    task_timeout_seconds: float = 30.0
    
    # Buffer configuration
    micro_buffer_size: int = 128         # Ultra-small buffers for immediate response
    input_buffer_size: int = 512
    output_buffer_size: int = 512
    max_queue_size: int = 100
    
    # Performance tuning
    enable_progressive_audio: bool = True
    enable_response_prediction: bool = True
    enable_priority_processing: bool = True
    adaptive_concurrency: bool = True
    
    # Network optimization
    websocket_compression: Optional[str] = None  # Disable for lower latency
    tcp_nodelay: bool = True
    socket_keepalive: bool = True


@dataclass
class StreamingMetrics:
    """Metrics for streaming performance"""
    # Latency metrics
    first_chunk_latency: float = 0.0
    avg_chunk_latency: float = 0.0
    end_to_end_latency: float = 0.0
    
    # Throughput metrics
    chunks_streamed: int = 0
    bytes_streamed: int = 0
    stream_rate_mbps: float = 0.0
    
    # Concurrency metrics
    active_streams: int = 0
    max_concurrent_reached: int = 0
    worker_utilization: float = 0.0
    
    # Quality metrics
    stream_interruptions: int = 0
    chunk_retransmissions: int = 0
    adaptive_adjustments: int = 0
    
    session_start: datetime = field(default_factory=datetime.now)
    
    def get_streaming_efficiency(self) -> float:
        """Calculate streaming efficiency percentage"""
        if self.chunks_streamed == 0:
            return 100.0
        
        success_rate = (self.chunks_streamed - self.chunk_retransmissions) / self.chunks_streamed
        return success_rate * 100.0


@dataclass
class ResponseChunk:
    """Individual response chunk for streaming"""
    sequence_id: int
    data: bytes
    chunk_type: str  # "audio", "text", "metadata"
    timestamp: float
    priority: ResponsePriority = ResponsePriority.NORMAL
    is_final: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_age_ms(self) -> float:
        """Get chunk age in milliseconds"""
        return (time.time() - self.timestamp) * 1000
    
    def __lt__(self, other):
        """Make ResponseChunk comparable for PriorityQueue"""
        if not isinstance(other, ResponseChunk):
            return NotImplemented
        # Compare by sequence_id for stable ordering when priorities are equal
        return self.sequence_id < other.sequence_id


class AsyncResponseStream:
    """Individual async response stream handler"""
    
    def __init__(self, stream_id: str, config: AsyncConfig, logger: Optional[logging.Logger] = None):
        self.stream_id = stream_id
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Stream state
        self.is_active = False
        self.is_streaming = False
        self.start_time = 0.0
        self.first_chunk_sent = False
        
        # Queues and buffers
        self.output_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.priority_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        
        # Metrics
        self.metrics = StreamingMetrics()
        self.sequence_counter = 0
        
        # Streaming tasks
        self.streaming_task: Optional[asyncio.Task] = None
        self.chunk_processor_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.chunk_callbacks: List[Callable[[ResponseChunk], None]] = []
        self.completion_callbacks: List[Callable[[str], None]] = []
        
    async def initialize(self) -> bool:
        """Initialize the response stream"""
        try:
            self.is_active = True
            
            # Start streaming tasks
            self.streaming_task = asyncio.create_task(self._streaming_loop())
            self.chunk_processor_task = asyncio.create_task(self._chunk_processor_loop())
            
            self.logger.debug(f"📡 Response stream {self.stream_id} initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize response stream {self.stream_id}: {e}")
            return False
    
    async def shutdown(self):
        """Graceful shutdown of response stream"""
        try:
            self.is_active = False
            self.is_streaming = False
            
            # Cancel streaming tasks
            if self.streaming_task:
                self.streaming_task.cancel()
            if self.chunk_processor_task:
                self.chunk_processor_task.cancel()
            
            # Wait for tasks to complete
            tasks = [task for task in [self.streaming_task, self.chunk_processor_task] if task]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            self.logger.debug(f"📡 Response stream {self.stream_id} shutdown completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error during response stream shutdown: {e}")
    
    async def queue_chunk(self, data: bytes, chunk_type: str = "audio", priority: ResponsePriority = ResponsePriority.NORMAL, is_final: bool = False) -> bool:
        """Queue chunk for streaming"""
        try:
            chunk = ResponseChunk(
                sequence_id=self.sequence_counter,
                data=data,
                chunk_type=chunk_type,
                timestamp=time.time(),
                priority=priority,
                is_final=is_final
            )
            self.sequence_counter += 1
            
            # Add to priority queue if priority processing enabled
            if self.config.enable_priority_processing:
                await self.priority_queue.put((priority.value, chunk))
            else:
                # Add to regular queue
                try:
                    self.output_queue.put_nowait(chunk)
                except asyncio.QueueFull:
                    self.logger.warning(f"⚠️ Output queue full for stream {self.stream_id}")
                    return False
            
            # Start streaming if this is the first chunk
            if not self.is_streaming:
                self.is_streaming = True
                self.start_time = time.time()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error queuing chunk for stream {self.stream_id}: {e}")
            return False
    
    async def _streaming_loop(self):
        """Main streaming loop for immediate delivery"""
        while self.is_active:
            try:
                chunk = None
                
                # Get chunk from priority queue or regular queue
                if self.config.enable_priority_processing and not self.priority_queue.empty():
                    try:
                        priority, chunk = await asyncio.wait_for(self.priority_queue.get(), timeout=0.001)
                    except asyncio.TimeoutError:
                        pass
                
                # Fallback to regular queue
                if chunk is None:
                    try:
                        chunk = await asyncio.wait_for(self.output_queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue
                
                # Stream the chunk immediately
                await self._stream_chunk(chunk)
                
                # Handle final chunk
                if chunk.is_final:
                    await self._handle_stream_completion()
                    break
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Streaming loop error for {self.stream_id}: {e}")
    
    async def _chunk_processor_loop(self):
        """Process chunks for optimization and adaptation"""
        while self.is_active:
            try:
                await asyncio.sleep(0.01)  # 10ms processing interval
                
                # Adaptive chunk size adjustment
                if self.config.adaptive_chunk_sizing:
                    await self._adjust_chunk_size()
                
                # Monitor queue health
                await self._monitor_queue_health()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Chunk processor error for {self.stream_id}: {e}")
    
    async def _stream_chunk(self, chunk: ResponseChunk):
        """Stream individual chunk with immediate delivery"""
        try:
            delivery_start = time.time()
            
            # Mark first chunk timing
            if not self.first_chunk_sent:
                self.first_chunk_sent = True
                self.metrics.first_chunk_latency = (delivery_start - self.start_time) * 1000
                self.logger.debug(f"🚀 First chunk delivered in {self.metrics.first_chunk_latency:.1f}ms")
            
            # Simulate immediate streaming delivery
            # In real implementation, this would send via WebSocket or similar
            await self._deliver_chunk(chunk)
            
            # Update metrics
            delivery_time = (time.time() - delivery_start) * 1000
            self.metrics.chunks_streamed += 1
            self.metrics.bytes_streamed += len(chunk.data)
            
            # Update average chunk latency
            if self.metrics.chunks_streamed == 1:
                self.metrics.avg_chunk_latency = delivery_time
            else:
                self.metrics.avg_chunk_latency = (
                    (self.metrics.avg_chunk_latency * (self.metrics.chunks_streamed - 1) + delivery_time) 
                    / self.metrics.chunks_streamed
                )
            
            # Notify callbacks
            for callback in self.chunk_callbacks:
                try:
                    callback(chunk)
                except Exception as e:
                    self.logger.error(f"❌ Chunk callback error: {e}")
            
        except Exception as e:
            self.logger.error(f"❌ Error streaming chunk: {e}")
            self.metrics.stream_interruptions += 1
    
    async def _deliver_chunk(self, chunk: ResponseChunk):
        """Deliver chunk via transport (WebSocket, etc.)"""
        # Simulate delivery latency based on chunk size and network conditions
        simulated_latency = len(chunk.data) / 100000  # Simulate network delivery
        await asyncio.sleep(min(simulated_latency, 0.001))  # Max 1ms simulated latency
        
        self.logger.debug(f"📦 Chunk {chunk.sequence_id} delivered ({len(chunk.data)} bytes)")
    
    async def _adjust_chunk_size(self):
        """Dynamically adjust chunk size based on performance"""
        if self.metrics.chunks_streamed < 10:
            return  # Need some data to make decisions
        
        # Adjust based on average latency
        if self.metrics.avg_chunk_latency > 10.0:  # Above 10ms average
            # Reduce chunk size for lower latency
            self.config.chunk_size_bytes = max(512, int(self.config.chunk_size_bytes * 0.9))
            self.metrics.adaptive_adjustments += 1
            self.logger.debug(f"📏 Reduced chunk size to {self.config.chunk_size_bytes} bytes")
        elif self.metrics.avg_chunk_latency < 2.0:  # Below 2ms average
            # Increase chunk size for efficiency
            self.config.chunk_size_bytes = min(4096, int(self.config.chunk_size_bytes * 1.1))
            self.metrics.adaptive_adjustments += 1
            self.logger.debug(f"📏 Increased chunk size to {self.config.chunk_size_bytes} bytes")
    
    async def _monitor_queue_health(self):
        """Monitor queue health and handle issues"""
        output_size = self.output_queue.qsize()
        priority_size = self.priority_queue.qsize()
        
        # Warning if queues are getting full
        if output_size > self.config.max_queue_size * 0.8:
            self.logger.warning(f"⚠️ Output queue {output_size}/{self.config.max_queue_size} for {self.stream_id}")
        
        if priority_size > 50:  # Priority queue should stay small
            self.logger.warning(f"⚠️ Priority queue backing up: {priority_size} items")
    
    async def _handle_stream_completion(self):
        """Handle stream completion"""
        self.is_streaming = False
        self.metrics.end_to_end_latency = (time.time() - self.start_time) * 1000
        
        # Calculate stream rate
        session_duration = (datetime.now() - self.metrics.session_start).total_seconds()
        if session_duration > 0:
            self.metrics.stream_rate_mbps = (self.metrics.bytes_streamed * 8) / (session_duration * 1_000_000)
        
        # Notify completion callbacks
        for callback in self.completion_callbacks:
            try:
                callback(self.stream_id)
            except Exception as e:
                self.logger.error(f"❌ Completion callback error: {e}")
        
        self.logger.info(f"✅ Stream {self.stream_id} completed: {self.metrics.chunks_streamed} chunks, {self.metrics.end_to_end_latency:.1f}ms")
    
    def add_chunk_callback(self, callback: Callable[[ResponseChunk], None]):
        """Add callback for chunk delivery"""
        self.chunk_callbacks.append(callback)
    
    def add_completion_callback(self, callback: Callable[[str], None]):
        """Add callback for stream completion"""
        self.completion_callbacks.append(callback)


class ConcurrentAudioProcessor:
    """Concurrent audio processing with full asyncio implementation"""
    
    def __init__(self, config: AsyncConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Processing state
        self.is_processing = False
        self.worker_tasks: List[asyncio.Task] = []
        
        # Audio queues
        self.input_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.output_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.priority_audio_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        
        # Worker management
        self.active_workers = 0
        self.max_workers = self._get_worker_count()
        
        # Performance tracking
        self.processed_chunks = 0
        self.processing_times: List[float] = []
        
    def _get_worker_count(self) -> int:
        """Get optimal worker count based on concurrency level"""
        if self.config.concurrency_level == ConcurrencyLevel.LOW:
            return 2
        elif self.config.concurrency_level == ConcurrencyLevel.MEDIUM:
            return 4
        elif self.config.concurrency_level == ConcurrencyLevel.HIGH:
            return 8
        else:  # ADAPTIVE
            return min(8, max(2, self.config.worker_pool_size))
    
    async def initialize(self) -> bool:
        """Initialize concurrent audio processor"""
        try:
            self.is_processing = True
            
            # Start worker tasks
            for i in range(self.max_workers):
                task = asyncio.create_task(self._audio_worker_loop(i))
                self.worker_tasks.append(task)
            
            # Start adaptive worker management if enabled
            if self.config.adaptive_concurrency:
                adaptive_task = asyncio.create_task(self._adaptive_worker_management())
                self.worker_tasks.append(adaptive_task)
            
            self.logger.info(f"🎧 Concurrent audio processor initialized with {self.max_workers} workers")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize concurrent audio processor: {e}")
            return False
    
    async def shutdown(self):
        """Graceful shutdown of audio processor"""
        try:
            self.is_processing = False
            
            # Cancel all worker tasks
            for task in self.worker_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self.worker_tasks:
                await asyncio.gather(*self.worker_tasks, return_exceptions=True)
            
            self.logger.info("✅ Concurrent audio processor shutdown completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error during concurrent audio processor shutdown: {e}")
    
    async def process_audio_async(self, audio_data: bytes, priority: ResponsePriority = ResponsePriority.NORMAL) -> bool:
        """Process audio data asynchronously"""
        try:
            # Add to appropriate queue based on priority
            if priority in [ResponsePriority.CRITICAL, ResponsePriority.HIGH]:
                await self.priority_audio_queue.put((priority.value, audio_data))
            else:
                try:
                    self.input_queue.put_nowait(audio_data)
                except asyncio.QueueFull:
                    self.logger.warning("⚠️ Audio input queue full - dropping chunk")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error processing audio async: {e}")
            return False
    
    async def get_processed_audio(self, timeout: float = 0.1) -> Optional[bytes]:
        """Get processed audio with timeout"""
        try:
            return await asyncio.wait_for(self.output_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            self.logger.error(f"❌ Error getting processed audio: {e}")
            return None
    
    async def _audio_worker_loop(self, worker_id: int):
        """Main audio processing worker loop"""
        self.logger.debug(f"🎧 Audio worker {worker_id} started")
        self.active_workers += 1
        
        try:
            while self.is_processing:
                audio_data = None
                
                # Check priority queue first
                if not self.priority_audio_queue.empty():
                    try:
                        priority, audio_data = await asyncio.wait_for(
                            self.priority_audio_queue.get(), timeout=0.001
                        )
                    except asyncio.TimeoutError:
                        pass
                
                # Fallback to regular queue
                if audio_data is None:
                    try:
                        audio_data = await asyncio.wait_for(
                            self.input_queue.get(), timeout=0.1
                        )
                    except asyncio.TimeoutError:
                        continue
                
                # Process audio data
                processed_audio = await self._process_audio_chunk(audio_data)
                
                if processed_audio:
                    # Add to output queue
                    try:
                        self.output_queue.put_nowait(processed_audio)
                        self.processed_chunks += 1
                    except asyncio.QueueFull:
                        self.logger.warning(f"⚠️ Audio output queue full in worker {worker_id}")
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"❌ Audio worker {worker_id} error: {e}")
        finally:
            self.active_workers -= 1
            self.logger.debug(f"🎧 Audio worker {worker_id} stopped")
    
    async def _process_audio_chunk(self, audio_data: bytes) -> Optional[bytes]:
        """Process individual audio chunk"""
        start_time = time.time()
        
        try:
            # Simulate audio processing
            # In real implementation, this would do actual audio processing
            # For now, we'll just add a small delay to simulate processing time
            processing_delay = len(audio_data) / 1000000  # Simulate processing based on data size
            await asyncio.sleep(min(processing_delay, 0.001))  # Max 1ms processing time
            
            # Track processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 1000:
                self.processing_times.pop(0)
            
            return audio_data  # Return processed audio (unchanged for simulation)
            
        except Exception as e:
            self.logger.error(f"❌ Error processing audio chunk: {e}")
            return None
    
    async def _adaptive_worker_management(self):
        """Adaptive worker count management based on load"""
        while self.is_processing:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                # Calculate current load
                input_load = self.input_queue.qsize() / self.config.max_queue_size
                priority_load = self.priority_audio_queue.qsize() / 50  # Priority queue target
                
                # Adjust worker count based on load
                if input_load > 0.7 or priority_load > 0.5:  # High load
                    if len(self.worker_tasks) < 12:  # Max workers
                        # Add worker
                        new_worker = asyncio.create_task(self._audio_worker_loop(len(self.worker_tasks)))
                        self.worker_tasks.append(new_worker)
                        self.logger.info(f"📈 Added audio worker (total: {len(self.worker_tasks)})")
                
                elif input_load < 0.2 and priority_load < 0.1:  # Low load
                    if len(self.worker_tasks) > self.max_workers:
                        # Remove worker (let it finish naturally)
                        self.logger.info(f"📉 Reducing audio workers due to low load")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Adaptive worker management error: {e}")
    
    def get_processing_metrics(self) -> Dict[str, Any]:
        """Get audio processing metrics"""
        avg_processing_time = sum(self.processing_times) / max(len(self.processing_times), 1)
        
        return {
            "active_workers": self.active_workers,
            "max_workers": self.max_workers,
            "processed_chunks": self.processed_chunks,
            "avg_processing_time_ms": avg_processing_time * 1000,
            "input_queue_size": self.input_queue.qsize(),
            "output_queue_size": self.output_queue.qsize(),
            "priority_queue_size": self.priority_audio_queue.qsize()
        }


class AsyncResponseManager:
    """Central coordinator for async processing and response streaming"""
    
    def __init__(self, config: Optional[AsyncConfig] = None, logger: Optional[logging.Logger] = None):
        self.config = config or AsyncConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Core components
        self.audio_processor: Optional[ConcurrentAudioProcessor] = None
        
        # Stream management
        self.active_streams: Dict[str, AsyncResponseStream] = {}
        self.stream_counter = 0
        
        # Management tasks
        self.manager_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Performance monitoring
        self.total_streams_created = 0
        self.total_chunks_delivered = 0
        self.global_metrics = StreamingMetrics()
        
    async def initialize(self) -> bool:
        """Initialize async response manager"""
        try:
            # Initialize audio processor
            self.audio_processor = ConcurrentAudioProcessor(self.config, self.logger)
            if not await self.audio_processor.initialize():
                return False
            
            # Start management task
            self.is_running = True
            self.manager_task = asyncio.create_task(self._management_loop())
            
            self.logger.info("✅ Async response manager initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize async response manager: {e}")
            return False
    
    async def shutdown(self):
        """Graceful shutdown of async response manager"""
        try:
            self.is_running = False
            
            # Shutdown all active streams
            shutdown_tasks = []
            for stream in self.active_streams.values():
                shutdown_tasks.append(stream.shutdown())
            
            if shutdown_tasks:
                await asyncio.gather(*shutdown_tasks, return_exceptions=True)
            
            # Shutdown audio processor
            if self.audio_processor:
                await self.audio_processor.shutdown()
            
            # Cancel management task
            if self.manager_task:
                self.manager_task.cancel()
                try:
                    await self.manager_task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info("✅ Async response manager shutdown completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error during async response manager shutdown: {e}")
    
    async def create_response_stream(self, stream_id: Optional[str] = None) -> str:
        """Create new response stream"""
        try:
            if stream_id is None:
                stream_id = f"stream_{self.stream_counter}"
                self.stream_counter += 1
            
            # Check if we've hit max concurrent streams
            if len(self.active_streams) >= self.config.max_concurrent_streams:
                raise RuntimeError(f"Maximum concurrent streams ({self.config.max_concurrent_streams}) reached")
            
            # Create and initialize stream
            stream = AsyncResponseStream(stream_id, self.config, self.logger)
            if not await stream.initialize():
                raise RuntimeError(f"Failed to initialize stream {stream_id}")
            
            self.active_streams[stream_id] = stream
            self.total_streams_created += 1
            
            # Update global metrics
            self.global_metrics.active_streams = len(self.active_streams)
            if self.global_metrics.active_streams > self.global_metrics.max_concurrent_reached:
                self.global_metrics.max_concurrent_reached = self.global_metrics.active_streams
            
            self.logger.info(f"📡 Created response stream: {stream_id}")
            return stream_id
            
        except Exception as e:
            self.logger.error(f"❌ Error creating response stream: {e}")
            raise
    
    async def stream_response_chunk(self, stream_id: str, data: bytes, chunk_type: str = "audio", priority: ResponsePriority = ResponsePriority.NORMAL, is_final: bool = False) -> bool:
        """Stream response chunk to specific stream"""
        try:
            if stream_id not in self.active_streams:
                self.logger.error(f"❌ Stream {stream_id} not found")
                return False
            
            stream = self.active_streams[stream_id]
            success = await stream.queue_chunk(data, chunk_type, priority, is_final)
            
            if success:
                self.total_chunks_delivered += 1
                self.global_metrics.chunks_streamed += 1
                self.global_metrics.bytes_streamed += len(data)
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error streaming chunk to {stream_id}: {e}")
            return False
    
    async def close_response_stream(self, stream_id: str) -> bool:
        """Close and cleanup response stream"""
        try:
            if stream_id not in self.active_streams:
                return False
            
            stream = self.active_streams[stream_id]
            await stream.shutdown()
            del self.active_streams[stream_id]
            
            # Update global metrics
            self.global_metrics.active_streams = len(self.active_streams)
            
            self.logger.info(f"📡 Closed response stream: {stream_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error closing stream {stream_id}: {e}")
            return False
    
    async def process_audio_concurrent(self, audio_data: bytes, priority: ResponsePriority = ResponsePriority.NORMAL) -> bool:
        """Process audio concurrently"""
        if self.audio_processor:
            return await self.audio_processor.process_audio_async(audio_data, priority)
        return False
    
    async def get_processed_audio(self, timeout: float = 0.1) -> Optional[bytes]:
        """Get processed audio"""
        if self.audio_processor:
            return await self.audio_processor.get_processed_audio(timeout)
        return None
    
    async def _management_loop(self):
        """Background management and optimization loop"""
        while self.is_running:
            try:
                await asyncio.sleep(1)  # Run every second
                
                # Cleanup completed streams
                completed_streams = []
                for stream_id, stream in self.active_streams.items():
                    if not stream.is_active:
                        completed_streams.append(stream_id)
                
                for stream_id in completed_streams:
                    await self.close_response_stream(stream_id)
                
                # Update global metrics
                self._update_global_metrics()
                
                # Adaptive optimization
                await self._adaptive_optimization()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Management loop error: {e}")
    
    def _update_global_metrics(self):
        """Update global metrics from all streams"""
        total_efficiency = 0.0
        active_count = 0
        
        for stream in self.active_streams.values():
            if stream.is_active:
                efficiency = stream.metrics.get_streaming_efficiency()
                total_efficiency += efficiency
                active_count += 1
        
        # Update worker utilization from audio processor
        if self.audio_processor:
            self.global_metrics.worker_utilization = (
                self.audio_processor.active_workers / max(self.audio_processor.max_workers, 1)
            ) * 100.0
    
    async def _adaptive_optimization(self):
        """Adaptive optimization based on current performance"""
        # Adjust concurrency level based on load
        if self.config.adaptive_concurrency:
            current_load = len(self.active_streams) / self.config.max_concurrent_streams
            
            if current_load > 0.8 and self.config.concurrency_level != ConcurrencyLevel.HIGH:
                self.config.concurrency_level = ConcurrencyLevel.HIGH
                self.logger.info("📈 Increased concurrency level to HIGH")
            elif current_load < 0.3 and self.config.concurrency_level != ConcurrencyLevel.LOW:
                self.config.concurrency_level = ConcurrencyLevel.LOW
                self.logger.info("📉 Decreased concurrency level to LOW")
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from all components"""
        metrics = {
            "manager": {
                "active_streams": len(self.active_streams),
                "total_streams_created": self.total_streams_created,
                "total_chunks_delivered": self.total_chunks_delivered,
                "max_concurrent_reached": self.global_metrics.max_concurrent_reached,
                "concurrency_level": self.config.concurrency_level.value,
                "stream_mode": self.config.stream_mode.value
            },
            "global_streaming": {
                "chunks_streamed": self.global_metrics.chunks_streamed,
                "bytes_streamed": self.global_metrics.bytes_streamed,
                "stream_interruptions": self.global_metrics.stream_interruptions,
                "worker_utilization": self.global_metrics.worker_utilization
            }
        }
        
        # Add audio processor metrics
        if self.audio_processor:
            metrics["audio_processor"] = self.audio_processor.get_processing_metrics()
        
        # Add individual stream metrics
        stream_metrics = {}
        for stream_id, stream in self.active_streams.items():
            stream_metrics[stream_id] = {
                "is_streaming": stream.is_streaming,
                "chunks_streamed": stream.metrics.chunks_streamed,
                "first_chunk_latency": stream.metrics.first_chunk_latency,
                "avg_chunk_latency": stream.metrics.avg_chunk_latency,
                "streaming_efficiency": stream.metrics.get_streaming_efficiency()
            }
        
        metrics["streams"] = stream_metrics
        return metrics


# Factory function for easy creation
def create_async_response_manager(
    stream_mode: StreamMode = StreamMode.IMMEDIATE,
    concurrency_level: ConcurrencyLevel = ConcurrencyLevel.MEDIUM,
    enable_progressive_audio: bool = True,
    max_concurrent_streams: int = 10,
    logger: Optional[logging.Logger] = None
) -> AsyncResponseManager:
    """Create async response manager with common configuration"""
    
    config = AsyncConfig(
        stream_mode=stream_mode,
        concurrency_level=concurrency_level,
        enable_progressive_audio=enable_progressive_audio,
        max_concurrent_streams=max_concurrent_streams,
        adaptive_concurrency=True,
        enable_priority_processing=True
    )
    
    return AsyncResponseManager(config, logger) 