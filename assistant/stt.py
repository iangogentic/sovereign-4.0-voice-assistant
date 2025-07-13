"""
Speech-to-Text (STT) Service Module

This module provides speech-to-text functionality using OpenAI's Whisper API.
Features include:
- Real-time audio transcription
- Voice activity detection
- Retry logic with exponential backoff
- Audio preprocessing and chunk management
- Language detection and accuracy settings
"""

import asyncio
import logging
import time
import io
import wave
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
import openai
import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class STTConfig:
    """Configuration for Speech-to-Text service"""
    model: str = "whisper-1"
    language: str = "en"
    temperature: float = 0.0
    response_format: str = "json"
    
    # Audio processing settings
    min_audio_length: float = 0.3  # Minimum seconds of audio to process
    max_audio_length: float = 30.0  # Maximum seconds (Whisper API limit)
    silence_threshold: float = 0.001  # Silence detection threshold (lowered for better voice detection)
    
    # API settings
    timeout: int = 10  # API timeout in seconds
    max_retries: int = 3
    retry_delay: float = 1.0  # Base delay for exponential backoff
    
    # Voice Activity Detection
    vad_enabled: bool = True
    vad_frame_duration: float = 0.02  # 20ms frames
    vad_window_size: int = 30  # Number of frames to analyze


@dataclass
class STTResult:
    """Result from speech-to-text conversion"""
    text: str
    confidence: float
    language: str
    duration: float
    processing_time: float
    segments: List[Dict] = None
    
    def __post_init__(self):
        if self.segments is None:
            self.segments = []


class AudioProcessor:
    """Handles audio preprocessing and format conversion"""
    
    def __init__(self, config: STTConfig):
        self.config = config
        
    def convert_to_wav(self, audio_data: bytes, sample_rate: int = 16000, 
                      channels: int = 1) -> bytes:
        """Convert raw audio data to WAV format"""
        try:
            # Create a BytesIO object to store WAV data
            wav_buffer = io.BytesIO()
            
            # Create WAV file
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)  # 16-bit audio
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data)
            
            # Get WAV data
            wav_data = wav_buffer.getvalue()
            wav_buffer.close()
            
            return wav_data
            
        except Exception as e:
            logger.error(f"Error converting audio to WAV: {e}")
            return None
    
    def detect_voice_activity(self, audio_data: bytes) -> bool:
        """Detect if audio contains voice activity"""
        if not self.config.vad_enabled:
            return True
            
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Calculate RMS energy
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            
            # Normalize to 0-1 range
            normalized_rms = rms / 32768.0
            
            # Check against threshold
            has_voice = normalized_rms > self.config.silence_threshold
            
            logger.debug(f"Voice activity detection: RMS={normalized_rms:.4f}, "
                        f"Threshold={self.config.silence_threshold}, "
                        f"HasVoice={has_voice}")
            
            return has_voice
            
        except Exception as e:
            logger.error(f"Error in voice activity detection: {e}")
            return True  # Default to processing if VAD fails
    
    def calculate_audio_duration(self, audio_data: bytes, sample_rate: int = 16000) -> float:
        """Calculate duration of audio data in seconds"""
        try:
            # Each sample is 2 bytes (16-bit), mono
            num_samples = len(audio_data) // 2
            duration = num_samples / sample_rate
            return duration
        except Exception as e:
            logger.error(f"Error calculating audio duration: {e}")
            return 0.0
    
    def trim_silence(self, audio_data: bytes, sample_rate: int = 16000) -> bytes:
        """Remove silence from beginning and end of audio"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Find non-silent regions
            threshold = int(32768 * self.config.silence_threshold)
            non_silent = np.abs(audio_array) > threshold
            
            if not np.any(non_silent):
                return audio_data  # All silence, return as-is
            
            # Find start and end of non-silent regions
            non_silent_indices = np.where(non_silent)[0]
            start_idx = non_silent_indices[0]
            end_idx = non_silent_indices[-1]
            
            # Trim the audio
            trimmed_array = audio_array[start_idx:end_idx + 1]
            
            return trimmed_array.tobytes()
            
        except Exception as e:
            logger.error(f"Error trimming silence: {e}")
            return audio_data


class WhisperSTTService:
    """OpenAI Whisper Speech-to-Text service"""
    
    def __init__(self, config: STTConfig, api_key: str, org_id: Optional[str] = None):
        self.config = config
        # Initialize OpenAI client with organization ID if provided
        client_kwargs = {"api_key": api_key}
        if org_id:
            client_kwargs["organization"] = org_id
        self.client = OpenAI(**client_kwargs)
        self.audio_processor = AudioProcessor(config)
        self.is_initialized = False
        
        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_processing_time = 0.0
        
    def initialize(self) -> bool:
        """Initialize the STT service"""
        try:
            logger.info("Initializing Whisper STT service...")
            
            # Skip the API test call to avoid blocking during startup
            # API connectivity will be tested during the first actual transcription
            logger.info("Whisper STT service initialized (API test skipped for faster startup)")
            self.is_initialized = True
            return True
                
        except Exception as e:
            logger.error(f"Failed to initialize Whisper STT service: {e}")
            return False
    
    def _create_test_audio(self) -> bytes:
        """Create a minimal test audio file for API testing"""
        # Create 0.5 seconds of silence
        sample_rate = 16000
        duration = 0.5
        samples = int(sample_rate * duration)
        
        # Generate silence
        audio_data = np.zeros(samples, dtype=np.int16)
        
        # Convert to WAV
        return self.audio_processor.convert_to_wav(audio_data.tobytes(), sample_rate)
    
    async def transcribe_audio(self, audio_data: bytes, sample_rate: int = 16000) -> Optional[STTResult]:
        """Transcribe audio data to text"""
        if not self.is_initialized:
            logger.error("STT service not initialized")
            return None
        
        start_time = time.time()
        
        try:
            # Calculate audio duration
            duration = self.audio_processor.calculate_audio_duration(audio_data, sample_rate)
            
            # Check minimum duration
            if duration < self.config.min_audio_length:
                logger.debug(f"Audio too short: {duration:.2f}s < {self.config.min_audio_length}s")
                return None
            
            # Check maximum duration
            if duration > self.config.max_audio_length:
                logger.warning(f"Audio too long: {duration:.2f}s > {self.config.max_audio_length}s")
                return None
            
            # Voice activity detection
            if not self.audio_processor.detect_voice_activity(audio_data):
                logger.debug("No voice activity detected")
                return None
            
            # Trim silence
            trimmed_audio = self.audio_processor.trim_silence(audio_data, sample_rate)
            
            # Convert to WAV format
            wav_data = self.audio_processor.convert_to_wav(trimmed_audio, sample_rate)
            if not wav_data:
                logger.error("Failed to convert audio to WAV format")
                return None
            
            # Perform transcription with retry logic
            result = await self._transcribe_with_retry(wav_data, duration)
            
            if result:
                processing_time = time.time() - start_time
                result.processing_time = processing_time
                
                self.successful_requests += 1
                self.total_processing_time += processing_time
                
                logger.info(f"Transcription successful: '{result.text}' "
                           f"(Duration: {duration:.2f}s, "
                           f"Processing: {processing_time:.2f}s)")
                
                return result
            else:
                self.failed_requests += 1
                return None
                
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            self.failed_requests += 1
            return None
        finally:
            self.total_requests += 1
    
    async def _transcribe_with_retry(self, wav_data: bytes, duration: float) -> Optional[STTResult]:
        """Perform transcription with exponential backoff retry"""
        for attempt in range(self.config.max_retries + 1):
            try:
                # Create file-like object for API
                audio_file = io.BytesIO(wav_data)
                audio_file.name = "audio.wav"
                
                # Make API request
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.audio.transcriptions.create(
                        model=self.config.model,
                        file=audio_file,
                        language=self.config.language,
                        response_format=self.config.response_format,
                        temperature=self.config.temperature
                    )
                )
                
                # Process response
                if hasattr(response, 'text'):
                    text = response.text.strip()
                    
                    # Extract additional information if available
                    language = getattr(response, 'language', self.config.language)
                    segments = getattr(response, 'segments', [])
                    
                    # Estimate confidence (Whisper API doesn't provide confidence scores)
                    confidence = self._estimate_confidence(text, segments)
                    
                    return STTResult(
                        text=text,
                        confidence=confidence,
                        language=language,
                        duration=duration,
                        processing_time=0.0,  # Will be set by caller
                        segments=segments
                    )
                else:
                    logger.error("Invalid response format from Whisper API")
                    return None
                    
            except Exception as e:
                logger.warning(f"Transcription attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.max_retries:
                    # Calculate delay with exponential backoff
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All transcription attempts failed after {self.config.max_retries + 1} tries")
                    return None
        
        return None
    
    def _estimate_confidence(self, text: str, segments: List[Dict]) -> float:
        """Estimate confidence score based on text characteristics"""
        if not text:
            return 0.0
        
        # Basic confidence estimation based on text length and characteristics
        confidence = 0.5  # Base confidence
        
        # Longer text generally indicates better transcription
        if len(text) > 10:
            confidence += 0.2
        
        # Check for common transcription artifacts
        if not any(char in text.lower() for char in ['[', ']', '(', ')', 'inaudible', 'unclear']):
            confidence += 0.2
        
        # Text with proper capitalization and punctuation
        if text[0].isupper() and text.endswith('.'):
            confidence += 0.1
        
        # Ensure confidence is in valid range
        return min(max(confidence, 0.0), 1.0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        avg_processing_time = (self.total_processing_time / self.successful_requests 
                             if self.successful_requests > 0 else 0.0)
        
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': (self.successful_requests / self.total_requests 
                           if self.total_requests > 0 else 0.0),
            'average_processing_time': avg_processing_time,
            'total_processing_time': self.total_processing_time
        }
    
    def reset_statistics(self):
        """Reset service statistics"""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_processing_time = 0.0
        logger.info("STT service statistics reset")


def create_whisper_stt_service(api_key: str, config: Optional[STTConfig] = None, org_id: Optional[str] = None) -> WhisperSTTService:
    """Factory function to create WhisperSTTService with default configuration"""
    if config is None:
        config = STTConfig()
    
    service = WhisperSTTService(config, api_key, org_id)
    
    # Initialize the service
    if service.initialize():
        logger.info("Whisper STT service created and initialized successfully")
    else:
        logger.error("Failed to initialize Whisper STT service")
        
    return service 