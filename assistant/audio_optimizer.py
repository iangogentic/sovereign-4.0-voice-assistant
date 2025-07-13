"""
Advanced Audio Processing Optimizer for Sovereign 4.0
Implements low-latency audio processing with 512-sample buffers and parallel processing
Optimized for sub-300ms response times with real-time streaming capabilities
"""

import asyncio
import numpy as np
import time
import logging
import threading
import multiprocessing as mp
import queue
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Tuple, Union
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import io
import wave
import struct
from enum import Enum


class AudioFormat(Enum):
    """Supported audio formats for processing"""
    PCM16 = "pcm16"
    PCM24 = "pcm24"
    FLOAT32 = "float32"


class ProcessingMode(Enum):
    """Audio processing modes for different latency requirements"""
    ULTRA_LOW_LATENCY = "ultra_low_latency"    # 512 samples, minimal processing
    LOW_LATENCY = "low_latency"                # 1024 samples, basic processing
    BALANCED = "balanced"                       # 2048 samples, full processing
    HIGH_QUALITY = "high_quality"              # 4096 samples, enhanced processing


@dataclass
class AudioConfig:
    """Optimized audio configuration for low-latency processing"""
    # Core audio settings
    sample_rate: int = 24000  # OpenAI Realtime API requirement
    channels: int = 1         # Mono for voice
    bit_depth: int = 16       # 16-bit for balance of quality and speed
    audio_format: AudioFormat = AudioFormat.PCM16
    
    # Buffer configuration for ultra-low latency
    buffer_size: int = 512    # Ultra-small buffer for minimal latency
    chunk_size: int = 256     # Even smaller chunks for streaming
    max_buffer_count: int = 8 # Maximum buffered chunks
    
    # Processing configuration
    processing_mode: ProcessingMode = ProcessingMode.ULTRA_LOW_LATENCY
    parallel_processing: bool = True
    max_workers: int = 4      # Number of parallel workers
    
    # Quality settings
    noise_reduction: bool = True
    auto_gain_control: bool = True
    echo_cancellation: bool = False  # Disabled for lowest latency
    adaptive_quality: bool = True
    
    # Streaming configuration
    stream_immediately: bool = True
    overlap_processing: bool = True
    predictive_buffering: bool = True


@dataclass
class AudioMetrics:
    """Audio processing performance metrics"""
    # Latency metrics
    avg_processing_latency: float = 0.0
    avg_buffer_latency: float = 0.0
    avg_streaming_latency: float = 0.0
    total_processing_time: float = 0.0
    
    # Throughput metrics
    samples_processed: int = 0
    chunks_processed: int = 0
    bytes_processed: int = 0
    processing_rate: float = 0.0  # samples per second
    
    # Quality metrics
    buffer_underruns: int = 0
    buffer_overruns: int = 0
    dropped_samples: int = 0
    noise_reduction_applied: int = 0
    
    # Parallel processing metrics
    worker_utilization: float = 0.0
    parallel_efficiency: float = 0.0
    queue_depth: int = 0
    
    # Session tracking
    session_start: datetime = field(default_factory=datetime.now)
    
    def get_processing_efficiency(self) -> float:
        """Calculate processing efficiency as percentage of real-time"""
        if self.total_processing_time == 0:
            return 100.0
        
        session_duration = (datetime.now() - self.session_start).total_seconds()
        return min(100.0, (session_duration / self.total_processing_time) * 100)
    
    def get_average_latency(self) -> float:
        """Get overall average latency"""
        return (self.avg_processing_latency + self.avg_buffer_latency + self.avg_streaming_latency) / 3


class AudioChunk:
    """Optimized audio chunk for efficient processing"""
    
    def __init__(self, data: bytes, timestamp: float, sequence_id: int, sample_rate: int = 24000):
        self.data = data
        self.timestamp = timestamp
        self.sequence_id = sequence_id
        self.sample_rate = sample_rate
        self.processed_timestamp: Optional[float] = None
        self.metadata: Dict[str, Any] = {}
        
        # Calculate derived properties
        self.sample_count = len(data) // 2  # 16-bit samples
        self.duration_ms = (self.sample_count / sample_rate) * 1000
        
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array for processing"""
        return np.frombuffer(self.data, dtype=np.int16)
    
    def from_numpy(self, array: np.ndarray) -> 'AudioChunk':
        """Create new chunk from numpy array"""
        return AudioChunk(
            data=array.astype(np.int16).tobytes(),
            timestamp=self.timestamp,
            sequence_id=self.sequence_id,
            sample_rate=self.sample_rate
        )
    
    def get_processing_latency(self) -> float:
        """Get processing latency in milliseconds"""
        if self.processed_timestamp is None:
            return 0.0
        return (self.processed_timestamp - self.timestamp) * 1000


class ParallelAudioProcessor:
    """Parallel audio processing engine for ultra-low latency"""
    
    def __init__(self, config: AudioConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Worker management
        self.executor: Optional[ThreadPoolExecutor] = None
        self.worker_tasks: List[asyncio.Task] = []
        self.is_processing = False
        
        # Queue management
        self.input_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_buffer_count)
        self.output_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_buffer_count)
        self.priority_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        
        # Metrics and monitoring
        self.metrics = AudioMetrics()
        self.sequence_counter = 0
        self.last_sequence_output = -1
        
        # Processing callbacks
        self.chunk_callbacks: List[Callable[[AudioChunk], None]] = []
        self.latency_callbacks: List[Callable[[float], None]] = []
        
    async def initialize(self) -> bool:
        """Initialize parallel processing system"""
        try:
            if self.config.parallel_processing:
                self.executor = ThreadPoolExecutor(
                    max_workers=self.config.max_workers,
                    thread_name_prefix="AudioProcessor"
                )
                
                # Start worker tasks
                for i in range(self.config.max_workers):
                    task = asyncio.create_task(self._worker_loop(i))
                    self.worker_tasks.append(task)
            
            # Start output sequencer
            sequencer_task = asyncio.create_task(self._output_sequencer())
            self.worker_tasks.append(sequencer_task)
            
            self.is_processing = True
            self.logger.info(f"✅ Audio processor initialized with {self.config.max_workers} workers")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize audio processor: {e}")
            return False
    
    async def shutdown(self):
        """Graceful shutdown of processing system"""
        try:
            self.is_processing = False
            
            # Cancel worker tasks
            for task in self.worker_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self.worker_tasks:
                await asyncio.gather(*self.worker_tasks, return_exceptions=True)
            
            # Shutdown executor
            if self.executor:
                self.executor.shutdown(wait=True)
            
            self.logger.info("✅ Audio processor shutdown completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error during audio processor shutdown: {e}")
    
    async def process_chunk(self, audio_data: bytes) -> Optional[AudioChunk]:
        """Process audio chunk with parallel processing"""
        if not self.is_processing:
            return None
        
        try:
            # Create chunk with sequence ID
            chunk = AudioChunk(
                data=audio_data,
                timestamp=time.time(),
                sequence_id=self.sequence_counter,
                sample_rate=self.config.sample_rate
            )
            self.sequence_counter += 1
            
            # Add to processing queue (non-blocking for ultra-low latency)
            try:
                self.input_queue.put_nowait(chunk)
            except asyncio.QueueFull:
                self.metrics.buffer_overruns += 1
                self.logger.warning("⚠️ Audio processing queue full - dropping chunk")
                return None
            
            # Return immediately if streaming enabled
            if self.config.stream_immediately:
                return chunk
            
            # Wait for processed chunk if not streaming
            try:
                processed_chunk = await asyncio.wait_for(
                    self.output_queue.get(), 
                    timeout=0.1  # 100ms max wait
                )
                return processed_chunk
            except asyncio.TimeoutError:
                self.metrics.buffer_underruns += 1
                return chunk  # Return original if processing too slow
                
        except Exception as e:
            self.logger.error(f"❌ Error processing audio chunk: {e}")
            return None
    
    async def _worker_loop(self, worker_id: int):
        """Main worker loop for parallel processing"""
        self.logger.debug(f"🔧 Audio worker {worker_id} started")
        
        while self.is_processing:
            try:
                # Get chunk from input queue
                chunk = await asyncio.wait_for(self.input_queue.get(), timeout=1.0)
                
                # Process chunk in thread pool
                if self.executor:
                    loop = asyncio.get_event_loop()
                    processed_chunk = await loop.run_in_executor(
                        self.executor, 
                        self._process_chunk_sync, 
                        chunk
                    )
                else:
                    processed_chunk = self._process_chunk_sync(chunk)
                
                # Add to priority queue for sequencing
                await self.priority_queue.put((chunk.sequence_id, processed_chunk))
                
                # Update metrics
                self.metrics.chunks_processed += 1
                self.metrics.samples_processed += chunk.sample_count
                self.metrics.bytes_processed += len(chunk.data)
                
            except asyncio.TimeoutError:
                continue  # No chunks to process
            except Exception as e:
                self.logger.error(f"❌ Worker {worker_id} error: {e}")
    
    def _process_chunk_sync(self, chunk: AudioChunk) -> AudioChunk:
        """Synchronous chunk processing for thread execution"""
        start_time = time.time()
        
        try:
            # Convert to numpy for processing
            audio_array = chunk.to_numpy()
            
            # Apply processing based on mode
            if self.config.processing_mode == ProcessingMode.ULTRA_LOW_LATENCY:
                # Minimal processing for lowest latency
                processed_array = self._minimal_processing(audio_array)
            elif self.config.processing_mode == ProcessingMode.LOW_LATENCY:
                # Basic processing
                processed_array = self._basic_processing(audio_array)
            else:
                # Full processing
                processed_array = self._full_processing(audio_array)
            
            # Create processed chunk
            processed_chunk = chunk.from_numpy(processed_array)
            processed_chunk.processed_timestamp = time.time()
            
            # Update latency metrics
            processing_time = processed_chunk.processed_timestamp - start_time
            self._update_processing_metrics(processing_time)
            
            return processed_chunk
            
        except Exception as e:
            self.logger.error(f"❌ Sync processing error: {e}")
            return chunk  # Return original on error
    
    def _minimal_processing(self, audio_array: np.ndarray) -> np.ndarray:
        """Minimal processing for ultra-low latency"""
        # Only essential processing
        if self.config.auto_gain_control:
            # Simple AGC
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            if rms > 0:
                target_rms = 0.1 * 32767  # Target 10% of max amplitude
                gain = target_rms / rms
                gain = np.clip(gain, 0.1, 3.0)  # Limit gain range
                audio_array = (audio_array.astype(np.float32) * gain).astype(np.int16)
        
        return audio_array
    
    def _basic_processing(self, audio_array: np.ndarray) -> np.ndarray:
        """Basic processing for low latency"""
        # Start with minimal processing
        processed = self._minimal_processing(audio_array)
        
        # Add noise gate
        if self.config.noise_reduction:
            noise_floor = 0.01 * 32767  # 1% of max amplitude
            mask = np.abs(processed) > noise_floor
            processed = processed * mask
            
            if np.any(mask):
                self.metrics.noise_reduction_applied += 1
        
        return processed
    
    def _full_processing(self, audio_array: np.ndarray) -> np.ndarray:
        """Full processing for highest quality"""
        # Start with basic processing
        processed = self._basic_processing(audio_array)
        
        # Add more sophisticated processing if needed
        # (Keep minimal for now to maintain performance)
        
        return processed
    
    async def _output_sequencer(self):
        """Ensure output chunks are delivered in sequence"""
        while self.is_processing:
            try:
                # Get next chunk in sequence
                sequence_id, processed_chunk = await asyncio.wait_for(
                    self.priority_queue.get(), 
                    timeout=1.0
                )
                
                # Check if this is the next expected sequence
                if sequence_id == self.last_sequence_output + 1:
                    # Output immediately
                    await self.output_queue.put(processed_chunk)
                    self.last_sequence_output = sequence_id
                    
                    # Notify callbacks
                    for callback in self.chunk_callbacks:
                        try:
                            callback(processed_chunk)
                        except Exception as e:
                            self.logger.error(f"❌ Chunk callback error: {e}")
                else:
                    # Out of sequence - put back in queue
                    await self.priority_queue.put((sequence_id, processed_chunk))
                    await asyncio.sleep(0.001)  # Short delay before retry
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"❌ Output sequencer error: {e}")
    
    def _update_processing_metrics(self, processing_time: float):
        """Update processing performance metrics"""
        # Update average processing latency
        if self.metrics.chunks_processed == 0:
            self.metrics.avg_processing_latency = processing_time * 1000  # Convert to ms
        else:
            self.metrics.avg_processing_latency = (
                (self.metrics.avg_processing_latency * (self.metrics.chunks_processed - 1) + 
                 processing_time * 1000) / self.metrics.chunks_processed
            )
        
        # Update total processing time
        self.metrics.total_processing_time += processing_time
        
        # Update processing rate
        session_duration = (datetime.now() - self.metrics.session_start).total_seconds()
        if session_duration > 0:
            self.metrics.processing_rate = self.metrics.samples_processed / session_duration
    
    def add_chunk_callback(self, callback: Callable[[AudioChunk], None]):
        """Add callback for processed chunks"""
        self.chunk_callbacks.append(callback)
    
    def add_latency_callback(self, callback: Callable[[float], None]):
        """Add callback for latency monitoring"""
        self.latency_callbacks.append(callback)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive processing metrics"""
        return {
            "processing_efficiency": self.metrics.get_processing_efficiency(),
            "average_latency": self.metrics.get_average_latency(),
            "avg_processing_latency": self.metrics.avg_processing_latency,
            "avg_buffer_latency": self.metrics.avg_buffer_latency,
            "avg_streaming_latency": self.metrics.avg_streaming_latency,
            "samples_processed": self.metrics.samples_processed,
            "chunks_processed": self.metrics.chunks_processed,
            "processing_rate": self.metrics.processing_rate,
            "buffer_underruns": self.metrics.buffer_underruns,
            "buffer_overruns": self.metrics.buffer_overruns,
            "dropped_samples": self.metrics.dropped_samples,
            "queue_depth": self.input_queue.qsize(),
            "output_queue_depth": self.output_queue.qsize(),
            "worker_count": len(self.worker_tasks),
            "session_duration": (datetime.now() - self.metrics.session_start).total_seconds()
        }


class AudioStreamOptimizer:
    """Real-time audio streaming optimizer with adaptive quality"""
    
    def __init__(self, config: AudioConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Processing components
        self.processor = ParallelAudioProcessor(config, logger)
        
        # Streaming state
        self.is_streaming = False
        self.stream_task: Optional[asyncio.Task] = None
        
        # Adaptive quality
        self.current_quality = config.processing_mode
        self.quality_adjustment_history: List[Tuple[datetime, ProcessingMode]] = []
        
        # Buffer management
        self.input_buffer: List[AudioChunk] = []
        self.output_buffer: List[AudioChunk] = []
        self.buffer_lock = asyncio.Lock()
        
        # Performance monitoring
        self.latency_history: List[float] = []
        self.quality_metrics: Dict[str, float] = {}
        
    async def initialize(self) -> bool:
        """Initialize streaming optimizer"""
        try:
            if not await self.processor.initialize():
                return False
            
            # Setup callbacks
            self.processor.add_chunk_callback(self._on_chunk_processed)
            self.processor.add_latency_callback(self._on_latency_update)
            
            self.logger.info("✅ Audio stream optimizer initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize stream optimizer: {e}")
            return False
    
    async def start_streaming(self):
        """Start real-time audio streaming"""
        if self.is_streaming:
            return
        
        self.is_streaming = True
        self.stream_task = asyncio.create_task(self._streaming_loop())
        self.logger.info("🎵 Audio streaming started")
    
    async def stop_streaming(self):
        """Stop audio streaming"""
        self.is_streaming = False
        
        if self.stream_task:
            self.stream_task.cancel()
            try:
                await self.stream_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("🎵 Audio streaming stopped")
    
    async def process_input(self, audio_data: bytes) -> Optional[bytes]:
        """Process input audio with streaming optimization"""
        try:
            # Process through parallel processor
            chunk = await self.processor.process_chunk(audio_data)
            
            if chunk and self.config.stream_immediately:
                return chunk.data
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error processing input: {e}")
            return None
    
    async def _streaming_loop(self):
        """Main streaming loop for continuous processing"""
        while self.is_streaming:
            try:
                # Monitor performance and adjust quality if needed
                await self._monitor_performance()
                
                # Process any buffered input
                async with self.buffer_lock:
                    if self.input_buffer:
                        chunk = self.input_buffer.pop(0)
                        processed = await self.processor.process_chunk(chunk.data)
                        
                        if processed:
                            self.output_buffer.append(processed)
                
                await asyncio.sleep(0.001)  # Minimal delay for cooperative multitasking
                
            except Exception as e:
                self.logger.error(f"❌ Streaming loop error: {e}")
    
    async def _monitor_performance(self):
        """Monitor performance and adjust quality adaptively"""
        if not self.config.adaptive_quality:
            return
        
        try:
            metrics = self.processor.get_metrics()
            current_latency = metrics.get("average_latency", 0.0)
            
            # Track latency history
            self.latency_history.append(current_latency)
            if len(self.latency_history) > 100:  # Keep last 100 samples
                self.latency_history.pop(0)
            
            # Adaptive quality adjustment
            if len(self.latency_history) >= 10:
                avg_latency = sum(self.latency_history[-10:]) / 10
                
                # Increase quality if latency is consistently low
                if avg_latency < 5.0 and self.current_quality == ProcessingMode.ULTRA_LOW_LATENCY:
                    await self._adjust_quality(ProcessingMode.LOW_LATENCY)
                elif avg_latency < 10.0 and self.current_quality == ProcessingMode.LOW_LATENCY:
                    await self._adjust_quality(ProcessingMode.BALANCED)
                
                # Decrease quality if latency is too high
                elif avg_latency > 20.0 and self.current_quality == ProcessingMode.BALANCED:
                    await self._adjust_quality(ProcessingMode.LOW_LATENCY)
                elif avg_latency > 15.0 and self.current_quality == ProcessingMode.LOW_LATENCY:
                    await self._adjust_quality(ProcessingMode.ULTRA_LOW_LATENCY)
                    
        except Exception as e:
            self.logger.error(f"❌ Performance monitoring error: {e}")
    
    async def _adjust_quality(self, new_quality: ProcessingMode):
        """Adjust processing quality dynamically"""
        if new_quality == self.current_quality:
            return
        
        old_quality = self.current_quality
        self.current_quality = new_quality
        self.processor.config.processing_mode = new_quality
        
        # Log quality change
        self.quality_adjustment_history.append((datetime.now(), new_quality))
        self.logger.info(f"🔧 Quality adjusted: {old_quality.value} → {new_quality.value}")
    
    def _on_chunk_processed(self, chunk: AudioChunk):
        """Callback for processed audio chunks"""
        # Calculate and track latency
        latency = chunk.get_processing_latency()
        if latency > 0:
            self.latency_history.append(latency)
    
    def _on_latency_update(self, latency: float):
        """Callback for latency updates"""
        self.latency_history.append(latency)
    
    async def shutdown(self):
        """Graceful shutdown"""
        await self.stop_streaming()
        await self.processor.shutdown()
        self.logger.info("✅ Audio stream optimizer shutdown completed")
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from all components"""
        processor_metrics = self.processor.get_metrics()
        
        return {
            **processor_metrics,
            "current_quality": self.current_quality.value,
            "quality_adjustments": len(self.quality_adjustment_history),
            "recent_latency_avg": sum(self.latency_history[-10:]) / max(len(self.latency_history[-10:]), 1),
            "latency_history_size": len(self.latency_history),
            "streaming_active": self.is_streaming,
            "input_buffer_size": len(self.input_buffer),
            "output_buffer_size": len(self.output_buffer)
        }


# Factory function for easy creation
def create_audio_optimizer(
    sample_rate: int = 24000,
    buffer_size: int = 512,
    processing_mode: ProcessingMode = ProcessingMode.ULTRA_LOW_LATENCY,
    parallel_processing: bool = True,
    max_workers: int = 4,
    logger: Optional[logging.Logger] = None
) -> AudioStreamOptimizer:
    """Create optimized audio stream processor with common configuration"""
    
    config = AudioConfig(
        sample_rate=sample_rate,
        buffer_size=buffer_size,
        processing_mode=processing_mode,
        parallel_processing=parallel_processing,
        max_workers=max_workers,
        stream_immediately=True,
        adaptive_quality=True
    )
    
    return AudioStreamOptimizer(config, logger) 