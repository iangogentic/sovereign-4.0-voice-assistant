"""
Emotion Preserver Module for Sovereign 4.0 Voice Assistant

Maintains voice tone and emotion throughout conversations without text intermediary flattening.
Implements direct audio-to-audio processing using OpenAI Realtime API with audio-only modalities.

Key Features:
- Audio-only processing (modalities=["audio"])
- Real-time audio buffer management
- Audio format validation and conversion
- Voice characteristic preservation
- Error handling and graceful degradation
- Integration with existing Realtime API infrastructure

Usage:
    emotion_preserver = EmotionPreserver(config)
    await emotion_preserver.initialize()
    await emotion_preserver.configure_audio_only_mode()
    
    # Process audio while preserving emotion
    preserved_audio = await emotion_preserver.process_audio(input_audio)
"""

import asyncio
import logging
import time
import numpy as np
from typing import Optional, Dict, Any, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pyaudio
import io
import wave
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class AudioFormat(Enum):
    """Supported audio formats for emotion preservation"""
    PCM16 = "pcm16"
    PCM24 = "pcm24" 
    MP3 = "mp3"
    WAV = "wav"


class EmotionProcessingMode(Enum):
    """Processing modes for emotion preservation"""
    AUDIO_ONLY = "audio_only"
    HYBRID = "hybrid"
    DISABLED = "disabled"


@dataclass
class AudioBufferConfig:
    """Configuration for audio buffer management"""
    sample_rate: int = 24000  # Required by Realtime API
    channels: int = 1  # Mono
    chunk_size: int = 1024
    format: int = pyaudio.paInt16
    buffer_duration_ms: int = 100  # 100ms buffer for real-time processing
    max_buffer_size: int = 48000  # 2 seconds at 24kHz
    overflow_handling: str = "drop_oldest"  # or "extend"


@dataclass
class EmotionPreservationConfig:
    """Configuration for emotion preservation system"""
    # Processing mode
    mode: EmotionProcessingMode = EmotionProcessingMode.AUDIO_ONLY
    
    # Audio settings
    input_format: AudioFormat = AudioFormat.PCM16
    output_format: AudioFormat = AudioFormat.PCM16
    sample_rate: int = 24000
    preserve_sample_rate: bool = True
    
    # Buffer management
    buffer_config: AudioBufferConfig = field(default_factory=AudioBufferConfig)
    
    # Performance settings
    processing_timeout: float = 0.1  # 100ms max processing time
    max_latency_ms: float = 50.0  # Target: < 50ms processing latency
    enable_real_time_processing: bool = True
    
    # Quality settings
    maintain_bit_depth: bool = True
    preserve_dynamic_range: bool = True
    enable_audio_normalization: bool = False
    
    # Error handling
    enable_graceful_degradation: bool = True
    fallback_to_text_mode: bool = False
    max_consecutive_errors: int = 3
    error_recovery_delay: float = 1.0


@dataclass
class AudioMetrics:
    """Track audio processing metrics for emotion preservation"""
    total_chunks_processed: int = 0
    processing_latency_ms: float = 0.0
    average_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    buffer_overflows: int = 0
    format_conversions: int = 0
    error_count: int = 0
    last_processing_time: float = 0.0
    
    def update_latency(self, latency_ms: float):
        """Update latency metrics"""
        self.last_processing_time = time.time()
        self.processing_latency_ms = latency_ms
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        
        # Calculate rolling average
        total_latency = self.average_latency_ms * self.total_chunks_processed + latency_ms
        self.total_chunks_processed += 1
        self.average_latency_ms = total_latency / self.total_chunks_processed


class AudioBuffer:
    """High-performance audio buffer for real-time processing"""
    
    def __init__(self, config: AudioBufferConfig):
        self.config = config
        self.buffer = np.zeros(config.max_buffer_size, dtype=np.int16)
        self.write_pos = 0
        self.read_pos = 0
        self.size = 0
        self.lock = asyncio.Lock()
        
    async def write(self, audio_data: bytes) -> bool:
        """Write audio data to buffer"""
        async with self.lock:
            try:
                # Convert bytes to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                data_length = len(audio_array)
                
                # Check for overflow
                if self.size + data_length > self.config.max_buffer_size:
                    if self.config.overflow_handling == "drop_oldest":
                        # Drop oldest data to make room
                        drop_amount = (self.size + data_length) - self.config.max_buffer_size
                        self.read_pos = (self.read_pos + drop_amount) % self.config.max_buffer_size
                        self.size -= drop_amount
                    else:
                        # Reject new data
                        logger.warning("Audio buffer overflow, dropping new data")
                        return False
                
                # Write data to buffer (handle wrap-around)
                end_pos = self.write_pos + data_length
                if end_pos <= self.config.max_buffer_size:
                    self.buffer[self.write_pos:end_pos] = audio_array
                else:
                    # Handle wrap-around
                    first_part = self.config.max_buffer_size - self.write_pos
                    self.buffer[self.write_pos:] = audio_array[:first_part]
                    self.buffer[:end_pos - self.config.max_buffer_size] = audio_array[first_part:]
                
                self.write_pos = end_pos % self.config.max_buffer_size
                self.size += data_length
                return True
                
            except Exception as e:
                logger.error(f"Error writing to audio buffer: {e}")
                return False
    
    async def read(self, chunk_size: int) -> Optional[bytes]:
        """Read audio data from buffer"""
        async with self.lock:
            try:
                if self.size < chunk_size:
                    return None
                
                # Read data from buffer (handle wrap-around)
                end_pos = self.read_pos + chunk_size
                if end_pos <= self.config.max_buffer_size:
                    audio_data = self.buffer[self.read_pos:end_pos]
                else:
                    # Handle wrap-around
                    first_part = self.config.max_buffer_size - self.read_pos
                    audio_data = np.concatenate([
                        self.buffer[self.read_pos:],
                        self.buffer[:end_pos - self.config.max_buffer_size]
                    ])
                
                self.read_pos = end_pos % self.config.max_buffer_size
                self.size -= chunk_size
                
                return audio_data.tobytes()
                
            except Exception as e:
                logger.error(f"Error reading from audio buffer: {e}")
                return None
    
    def get_available_size(self) -> int:
        """Get amount of data available in buffer"""
        return self.size
    
    def clear(self):
        """Clear the buffer"""
        self.write_pos = 0
        self.read_pos = 0
        self.size = 0


class AudioFormatConverter:
    """Handle audio format conversions for emotion preservation"""
    
    @staticmethod
    def validate_format(audio_data: bytes, expected_format: AudioFormat, sample_rate: int) -> bool:
        """Validate audio format matches expectations"""
        try:
            if expected_format == AudioFormat.PCM16:
                # Check if data length is appropriate for 16-bit samples
                return len(audio_data) % 2 == 0
            elif expected_format == AudioFormat.PCM24:
                return len(audio_data) % 3 == 0
            else:
                return True  # For other formats, assume valid
        except Exception:
            return False
    
    @staticmethod
    def convert_to_realtime_format(audio_data: bytes, source_format: AudioFormat) -> bytes:
        """Convert audio to Realtime API required format (PCM16, 24kHz)"""
        try:
            if source_format == AudioFormat.PCM16:
                # Already in correct format
                return audio_data
            elif source_format == AudioFormat.PCM24:
                # Convert 24-bit to 16-bit
                audio_array = np.frombuffer(audio_data, dtype=np.int32)
                # Scale down from 24-bit to 16-bit
                audio_16bit = (audio_array >> 8).astype(np.int16)
                return audio_16bit.tobytes()
            else:
                # For other formats, return as-is and log warning
                logger.warning(f"Unsupported audio format conversion: {source_format}")
                return audio_data
        except Exception as e:
            logger.error(f"Audio format conversion failed: {e}")
            return audio_data
    
    @staticmethod
    def normalize_audio(audio_data: bytes, target_level: float = 0.8) -> bytes:
        """Normalize audio levels while preserving dynamic range"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Calculate current peak level
            peak = np.max(np.abs(audio_array))
            if peak == 0:
                return audio_data
            
            # Calculate normalization factor
            max_int16 = 32767
            target_peak = int(max_int16 * target_level)
            normalization_factor = target_peak / peak
            
            # Apply normalization (avoid clipping)
            normalized = np.clip(audio_array * normalization_factor, -max_int16, max_int16)
            return normalized.astype(np.int16).tobytes()
            
        except Exception as e:
            logger.error(f"Audio normalization failed: {e}")
            return audio_data


class EmotionPreserver:
    """
    Core class for preserving voice emotion and tone through direct audio-to-audio processing.
    
    Integrates with OpenAI Realtime API using audio-only modalities to bypass text conversion
    that can flatten emotional content in voice.
    """
    
    def __init__(self, config: EmotionPreservationConfig, realtime_service=None, logger=None):
        self.config = config
        self.realtime_service = realtime_service
        self.logger = logger or logging.getLogger(__name__)
        
        # Audio processing components
        self.audio_buffer = AudioBuffer(config.buffer_config)
        self.format_converter = AudioFormatConverter()
        self.metrics = AudioMetrics()
        
        # State management
        self.is_initialized = False
        self.is_processing = False
        self.consecutive_errors = 0
        
        # Audio-only mode configuration
        self.audio_only_config = {
            "modalities": ["audio"],  # No text modality
            "input_audio_format": config.input_format.value,
            "output_audio_format": config.output_format.value,
            "sample_rate": config.sample_rate
        }
        
        # Callbacks
        self.on_audio_processed: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
    async def initialize(self) -> bool:
        """Initialize the emotion preservation system"""
        try:
            self.logger.info("Initializing EmotionPreserver...")
            
            # Validate configuration
            if not self._validate_config():
                raise ValueError("Invalid emotion preservation configuration")
            
            # Configure audio-only mode if realtime service is available
            if self.realtime_service:
                await self._configure_realtime_audio_mode()
            
            self.is_initialized = True
            self.logger.info("EmotionPreserver initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize EmotionPreserver: {e}")
            return False
    
    def _validate_config(self) -> bool:
        """Validate emotion preservation configuration"""
        try:
            # Check mode
            if self.config.mode == EmotionProcessingMode.DISABLED:
                self.logger.warning("Emotion preservation is disabled")
                return True
            
            # Check sample rate
            if self.config.sample_rate != 24000:
                self.logger.warning("Sample rate should be 24000 for Realtime API compatibility")
            
            # Check latency requirements
            if self.config.max_latency_ms > 100:
                self.logger.warning("Target latency > 100ms may affect real-time performance")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    async def _configure_realtime_audio_mode(self):
        """Configure Realtime API for audio-only processing"""
        try:
            if hasattr(self.realtime_service, 'config'):
                # Update realtime service config for audio-only mode
                self.realtime_service.config.modalities = ["audio"]
                self.realtime_service.config.input_audio_format = self.config.input_format.value
                self.realtime_service.config.output_audio_format = self.config.output_format.value
                
                self.logger.info("Configured Realtime API for audio-only emotion preservation")
        except Exception as e:
            self.logger.error(f"Failed to configure audio-only mode: {e}")
            raise
    
    async def configure_audio_only_mode(self) -> Dict[str, Any]:
        """Get audio-only configuration for Realtime API"""
        return {
            "modalities": ["audio"],
            "input_audio_format": self.config.input_format.value,
            "output_audio_format": self.config.output_format.value,
            "input_audio_transcription": {"enabled": False},  # Disable text transcription
            "voice": "alloy",  # Preserve voice setting
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 200
            }
        }
    
    async def process_audio(self, audio_data: bytes) -> Optional[bytes]:
        """
        Process audio data while preserving emotional characteristics.
        Returns processed audio or None if processing fails.
        """
        if not self.is_initialized:
            self.logger.error("EmotionPreserver not initialized")
            return None
        
        if self.config.mode == EmotionProcessingMode.DISABLED:
            return audio_data
        
        start_time = time.time()
        
        try:
            self.is_processing = True
            
            # Validate input format
            if not self.format_converter.validate_format(
                audio_data, self.config.input_format, self.config.sample_rate
            ):
                self.logger.warning("Invalid audio format detected")
                self.metrics.error_count += 1
            
            # Convert to Realtime API format if needed
            processed_audio = self.format_converter.convert_to_realtime_format(
                audio_data, self.config.input_format
            )
            
            # Apply normalization if enabled
            if not self.config.enable_audio_normalization:
                normalized_audio = processed_audio
            else:
                normalized_audio = self.format_converter.normalize_audio(processed_audio)
            
            # Buffer management for real-time processing
            await self.audio_buffer.write(normalized_audio)
            
            # Process through buffer to ensure consistent timing
            output_audio = await self.audio_buffer.read(len(normalized_audio))
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self.metrics.update_latency(processing_time)
            
            # Check latency requirements
            if processing_time > self.config.max_latency_ms:
                self.logger.warning(f"Processing latency ({processing_time:.1f}ms) exceeds target ({self.config.max_latency_ms}ms)")
            
            # Reset error counter on success
            self.consecutive_errors = 0
            
            # Trigger callback if set
            if self.on_audio_processed:
                await self.on_audio_processed(output_audio, processing_time)
            
            return output_audio or normalized_audio
            
        except Exception as e:
            self.logger.error(f"Audio processing failed: {e}")
            self.consecutive_errors += 1
            self.metrics.error_count += 1
            
            # Handle consecutive errors
            if self.consecutive_errors >= self.config.max_consecutive_errors:
                await self._handle_processing_failure()
            
            # Trigger error callback
            if self.on_error:
                await self.on_error(e, self.consecutive_errors)
            
            # Return original audio as fallback
            return audio_data if self.config.enable_graceful_degradation else None
            
        finally:
            self.is_processing = False
    
    async def _handle_processing_failure(self):
        """Handle repeated processing failures"""
        self.logger.error(f"Multiple consecutive processing failures ({self.consecutive_errors})")
        
        if self.config.fallback_to_text_mode:
            self.logger.info("Falling back to text mode due to audio processing failures")
            self.config.mode = EmotionProcessingMode.DISABLED
        
        # Clear buffer to prevent cascading failures
        self.audio_buffer.clear()
        
        # Add recovery delay
        await asyncio.sleep(self.config.error_recovery_delay)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics"""
        return {
            "total_chunks_processed": self.metrics.total_chunks_processed,
            "average_latency_ms": self.metrics.average_latency_ms,
            "max_latency_ms": self.metrics.max_latency_ms,
            "current_latency_ms": self.metrics.processing_latency_ms,
            "buffer_overflows": self.metrics.buffer_overflows,
            "format_conversions": self.metrics.format_conversions,
            "error_count": self.metrics.error_count,
            "error_rate": self.metrics.error_count / max(self.metrics.total_chunks_processed, 1),
            "is_processing": self.is_processing,
            "consecutive_errors": self.consecutive_errors,
            "buffer_utilization": self.audio_buffer.get_available_size() / self.config.buffer_config.max_buffer_size
        }
    
    def set_audio_processed_callback(self, callback: Callable):
        """Set callback for when audio is processed"""
        self.on_audio_processed = callback
    
    def set_error_callback(self, callback: Callable):
        """Set callback for processing errors"""
        self.on_error = callback
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            self.is_processing = False
            self.audio_buffer.clear()
            self.logger.info("EmotionPreserver cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


def create_emotion_preserver(
    mode: EmotionProcessingMode = EmotionProcessingMode.AUDIO_ONLY,
    realtime_service=None,
    logger=None
) -> EmotionPreserver:
    """Factory function to create EmotionPreserver with default configuration"""
    
    config = EmotionPreservationConfig(mode=mode)
    return EmotionPreserver(config, realtime_service, logger)


def create_audio_only_config() -> Dict[str, Any]:
    """Helper function to create audio-only Realtime API configuration"""
    return {
        "modalities": ["audio"],
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "input_audio_transcription": {"enabled": False},
        "sample_rate": 24000
    } 