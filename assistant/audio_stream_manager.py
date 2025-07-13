"""
Real-time Audio Stream Manager for OpenAI Realtime API

This module provides high-performance audio streaming capabilities specifically
optimized for OpenAI's Realtime API requirements:
- 24kHz PCM16 format
- Ultra-low latency streaming
- Circular buffer management
- Non-blocking audio processing with threading
- Audio format conversion utilities
"""

import asyncio
import logging
import threading
import time
from typing import Optional, Dict, List, Callable, Any
import pyaudio
import numpy as np
from collections import deque
from dataclasses import dataclass
from enum import Enum
import wave
import io

from .audio import AudioManager, AudioConfig, AudioDevice, AudioDeviceType

logger = logging.getLogger(__name__)


@dataclass
class RealtimeAudioConfig:
    """Configuration for real-time audio streaming (OpenAI Realtime API)"""
    # OpenAI Realtime API requirements
    sample_rate: int = 24000  # 24kHz required by Realtime API
    channels: int = 1  # Mono audio
    format: int = pyaudio.paInt16  # PCM16 format
    
    # Streaming configuration
    input_chunk_size: int = 1024  # Input buffer size
    output_chunk_size: int = 1024  # Output buffer size
    buffer_duration: float = 0.1  # Buffer duration in seconds
    
    # Threading configuration
    input_thread_priority: int = 1  # High priority for input
    output_thread_priority: int = 1  # High priority for output
    
    # Device selection
    input_device_index: Optional[int] = None
    output_device_index: Optional[int] = None
    
    # Performance tuning
    enable_latency_optimization: bool = True
    target_latency_ms: float = 50.0  # Target latency in milliseconds


class AudioStreamState(Enum):
    """Audio stream states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class AudioStreamManager:
    """
    High-performance audio stream manager for OpenAI Realtime API
    
    Provides real-time audio input/output with circular buffers,
    threading, and format conversion utilities.
    """
    
    def __init__(self, config: RealtimeAudioConfig, audio_manager: AudioManager = None):
        self.config = config
        self.audio_manager = audio_manager
        self.pyaudio = pyaudio.PyAudio()
        
        # Stream objects
        self.input_stream: Optional[pyaudio.Stream] = None
        self.output_stream: Optional[pyaudio.Stream] = None
        
        # Threading
        self.input_thread: Optional[threading.Thread] = None
        self.output_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Circular buffers for real-time processing
        buffer_size = int(config.sample_rate * config.buffer_duration * 2)  # Double buffer
        self.input_buffer = deque(maxlen=buffer_size)
        self.output_buffer = deque(maxlen=buffer_size)
        
        # Buffer locks for thread safety
        self.input_lock = threading.Lock()
        self.output_lock = threading.Lock()
        
        # State management
        self.input_state = AudioStreamState.STOPPED
        self.output_state = AudioStreamState.STOPPED
        
        # Callbacks for audio data handling
        self.on_audio_input: Optional[Callable[[bytes], None]] = None
        self.on_audio_output_request: Optional[Callable[[], Optional[bytes]]] = None
        
        # Performance metrics
        self.input_samples_processed = 0
        self.output_samples_processed = 0
        self.stream_start_time = 0.0
        
        self.logger = logging.getLogger(f"{__name__}.AudioStreamManager")
    
    def initialize(self) -> bool:
        """Initialize the audio stream manager"""
        try:
            self.logger.info("🎵 Initializing Real-time Audio Stream Manager...")
            
            # Verify device compatibility
            if not self._verify_device_compatibility():
                return False
            
            # Initialize performance monitoring
            self.stream_start_time = time.time()
            
            self.logger.info("✅ Audio Stream Manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Audio Stream Manager: {e}")
            return False
    
    def _verify_device_compatibility(self) -> bool:
        """Verify that selected devices support real-time streaming"""
        try:
            # Check input device
            if self.config.input_device_index is not None:
                input_info = self.pyaudio.get_device_info_by_index(self.config.input_device_index)
                if input_info['maxInputChannels'] < self.config.channels:
                    self.logger.error(f"Input device doesn't support {self.config.channels} channels")
                    return False
            
            # Check output device  
            if self.config.output_device_index is not None:
                output_info = self.pyaudio.get_device_info_by_index(self.config.output_device_index)
                if output_info['maxOutputChannels'] < self.config.channels:
                    self.logger.error(f"Output device doesn't support {self.config.channels} channels")
                    return False
            
            # Test sample rate compatibility
            if not self._test_sample_rate_support():
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Device compatibility check failed: {e}")
            return False
    
    def _test_sample_rate_support(self) -> bool:
        """Test if devices support the required 24kHz sample rate"""
        try:
            # Test input device
            if self.config.input_device_index is not None:
                if not self.pyaudio.is_format_supported(
                    rate=self.config.sample_rate,
                    input_device=self.config.input_device_index,
                    input_channels=self.config.channels,
                    input_format=self.config.format
                ):
                    self.logger.warning(f"Input device may not support 24kHz - will attempt anyway")
            
            # Test output device
            if self.config.output_device_index is not None:
                if not self.pyaudio.is_format_supported(
                    rate=self.config.sample_rate,
                    output_device=self.config.output_device_index,
                    output_channels=self.config.channels,
                    output_format=self.config.format
                ):
                    self.logger.warning(f"Output device may not support 24kHz - will attempt anyway")
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Sample rate test failed: {e} - will attempt anyway")
            return True
    
    def start_input_stream(self, callback: Callable[[bytes], None]) -> bool:
        """Start real-time audio input stream"""
        try:
            if self.input_state != AudioStreamState.STOPPED:
                self.logger.warning("Input stream already running")
                return False
            
            self.input_state = AudioStreamState.STARTING
            self.on_audio_input = callback
            
            # Create input stream
            self.input_stream = self.pyaudio.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=self.config.input_device_index,
                frames_per_buffer=self.config.input_chunk_size,
                stream_callback=self._input_stream_callback,
                start=False
            )
            
            # Start input processing thread
            self.input_thread = threading.Thread(
                target=self._input_thread_loop,
                name="AudioInputStream",
                daemon=True
            )
            
            # Start stream and thread
            self.input_stream.start_stream()
            self.input_thread.start()
            
            self.input_state = AudioStreamState.RUNNING
            self.logger.info("🎤 Real-time audio input stream started")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start input stream: {e}")
            self.input_state = AudioStreamState.ERROR
            return False
    
    def start_output_stream(self, callback: Callable[[], Optional[bytes]]) -> bool:
        """Start real-time audio output stream"""
        try:
            if self.output_state != AudioStreamState.STOPPED:
                self.logger.warning("Output stream already running")
                return False
            
            self.output_state = AudioStreamState.STARTING
            self.on_audio_output_request = callback
            
            # Create output stream
            self.output_stream = self.pyaudio.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                output=True,
                output_device_index=self.config.output_device_index,
                frames_per_buffer=self.config.output_chunk_size,
                stream_callback=self._output_stream_callback,
                start=False
            )
            
            # Start output processing thread
            self.output_thread = threading.Thread(
                target=self._output_thread_loop,
                name="AudioOutputStream",
                daemon=True
            )
            
            # Start stream and thread
            self.output_stream.start_stream()
            self.output_thread.start()
            
            self.output_state = AudioStreamState.RUNNING
            self.logger.info("🔊 Real-time audio output stream started")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start output stream: {e}")
            self.output_state = AudioStreamState.ERROR
            return False
    
    def _input_stream_callback(self, in_data, frame_count, time_info, status):
        """PyAudio input stream callback"""
        if status:
            self.logger.warning(f"Input stream status: {status}")
        
        # Add audio data to circular buffer
        with self.input_lock:
            self.input_buffer.extend(in_data)
        
        return (None, pyaudio.paContinue)
    
    def _output_stream_callback(self, in_data, frame_count, time_info, status):
        """PyAudio output stream callback"""
        if status:
            self.logger.warning(f"Output stream status: {status}")
        
        # Get audio data from circular buffer
        with self.output_lock:
            if len(self.output_buffer) >= frame_count * 2:  # 2 bytes per sample for PCM16
                audio_data = bytes([self.output_buffer.popleft() for _ in range(frame_count * 2)])
                return (audio_data, pyaudio.paContinue)
            else:
                # Return silence if no data available
                silence = b'\x00' * (frame_count * 2)
                return (silence, pyaudio.paContinue)
    
    def _input_thread_loop(self):
        """Input processing thread loop"""
        self.logger.debug("Input thread started")
        
        while not self.stop_event.is_set() and self.input_state == AudioStreamState.RUNNING:
            try:
                # Process input buffer
                with self.input_lock:
                    if len(self.input_buffer) >= self.config.input_chunk_size * 2:
                        # Extract audio chunk
                        chunk_data = bytes([self.input_buffer.popleft() 
                                          for _ in range(self.config.input_chunk_size * 2)])
                        
                        # Call callback with audio data
                        if self.on_audio_input:
                            self.on_audio_input(chunk_data)
                        
                        self.input_samples_processed += self.config.input_chunk_size
                
                # Small sleep to prevent busy waiting
                time.sleep(0.001)  # 1ms
                
            except Exception as e:
                self.logger.error(f"Input thread error: {e}")
                break
        
        self.logger.debug("Input thread stopped")
    
    def _output_thread_loop(self):
        """Output processing thread loop"""
        self.logger.debug("Output thread started")
        
        while not self.stop_event.is_set() and self.output_state == AudioStreamState.RUNNING:
            try:
                # Request audio data from callback
                if self.on_audio_output_request:
                    audio_data = self.on_audio_output_request()
                    
                    if audio_data:
                        # Add to output buffer
                        with self.output_lock:
                            self.output_buffer.extend(audio_data)
                        
                        self.output_samples_processed += len(audio_data) // 2  # 2 bytes per sample
                
                # Small sleep to prevent busy waiting
                time.sleep(0.001)  # 1ms
                
            except Exception as e:
                self.logger.error(f"Output thread error: {e}")
                break
        
        self.logger.debug("Output thread stopped")
    
    def handle_audio_chunk(self, audio_data: bytes):
        """Handle incoming audio chunk for output"""
        with self.output_lock:
            self.output_buffer.extend(audio_data)
    
    def stop_input_stream(self):
        """Stop audio input stream"""
        if self.input_state == AudioStreamState.RUNNING:
            self.input_state = AudioStreamState.STOPPING
            
            if self.input_stream:
                self.input_stream.stop_stream()
                self.input_stream.close()
                self.input_stream = None
            
            self.input_state = AudioStreamState.STOPPED
            self.logger.info("🎤 Audio input stream stopped")
    
    def stop_output_stream(self):
        """Stop audio output stream"""
        if self.output_state == AudioStreamState.RUNNING:
            self.output_state = AudioStreamState.STOPPING
            
            if self.output_stream:
                self.output_stream.stop_stream()
                self.output_stream.close()
                self.output_stream = None
            
            self.output_state = AudioStreamState.STOPPED
            self.logger.info("🔊 Audio output stream stopped")
    
    def stop_all_streams(self):
        """Stop all audio streams"""
        self.stop_event.set()
        self.stop_input_stream()
        self.stop_output_stream()
        
        # Wait for threads to finish
        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=1.0)
        
        if self.output_thread and self.output_thread.is_alive():
            self.output_thread.join(timeout=1.0)
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_all_streams()
        
        if self.pyaudio:
            self.pyaudio.terminate()
        
        self.logger.info("🧹 Audio Stream Manager cleaned up")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        runtime = time.time() - self.stream_start_time
        
        return {
            'runtime_seconds': runtime,
            'input_samples_processed': self.input_samples_processed,
            'output_samples_processed': self.output_samples_processed,
            'input_sample_rate': self.input_samples_processed / runtime if runtime > 0 else 0,
            'output_sample_rate': self.output_samples_processed / runtime if runtime > 0 else 0,
            'input_buffer_size': len(self.input_buffer),
            'output_buffer_size': len(self.output_buffer),
            'input_state': self.input_state.value,
            'output_state': self.output_state.value
        }


# Audio format conversion utilities
class AudioFormatConverter:
    """Utilities for audio format conversion"""
    
    @staticmethod
    def pcm16_to_float32(pcm16_data: bytes) -> np.ndarray:
        """Convert PCM16 bytes to float32 numpy array"""
        # Convert bytes to int16 array
        int16_array = np.frombuffer(pcm16_data, dtype=np.int16)
        # Normalize to float32 [-1.0, 1.0]
        float32_array = int16_array.astype(np.float32) / 32768.0
        return float32_array
    
    @staticmethod
    def float32_to_pcm16(float32_array: np.ndarray) -> bytes:
        """Convert float32 numpy array to PCM16 bytes"""
        # Clip to valid range and convert to int16
        clipped = np.clip(float32_array * 32768.0, -32768, 32767)
        int16_array = clipped.astype(np.int16)
        # Convert to bytes
        return int16_array.tobytes()
    
    @staticmethod
    def resample_audio(audio_data: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Simple audio resampling (for small rate differences)"""
        if from_rate == to_rate:
            return audio_data
        
        # Simple linear interpolation resampling
        ratio = to_rate / from_rate
        new_length = int(len(audio_data) * ratio)
        
        # Create new time indices
        old_indices = np.arange(len(audio_data))
        new_indices = np.linspace(0, len(audio_data) - 1, new_length)
        
        # Interpolate
        resampled = np.interp(new_indices, old_indices, audio_data)
        return resampled
    
    @staticmethod
    def convert_sample_rate(audio_bytes: bytes, from_rate: int, to_rate: int) -> bytes:
        """Convert audio sample rate while maintaining PCM16 format"""
        if from_rate == to_rate:
            return audio_bytes
        
        # Convert to float32 for processing
        float32_data = AudioFormatConverter.pcm16_to_float32(audio_bytes)
        
        # Resample
        resampled_data = AudioFormatConverter.resample_audio(float32_data, from_rate, to_rate)
        
        # Convert back to PCM16
        return AudioFormatConverter.float32_to_pcm16(resampled_data)


def create_realtime_audio_manager(config: Optional[RealtimeAudioConfig] = None, 
                                 audio_manager: Optional[AudioManager] = None) -> AudioStreamManager:
    """Factory function to create AudioStreamManager"""
    if config is None:
        config = RealtimeAudioConfig()
    
    return AudioStreamManager(config, audio_manager) 