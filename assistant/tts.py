"""
Text-to-Speech (TTS) Service Module

This module provides text-to-speech functionality using OpenAI's TTS API.
Features include:
- High-quality speech synthesis
- Multiple voice options and speed control
- Audio format conversion and optimization
- Audio caching for repeated phrases
- Retry logic with exponential backoff
- Performance monitoring and statistics
"""

import asyncio
import hashlib
import logging
import time
import io
import os
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import openai
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play
import tempfile

logger = logging.getLogger(__name__)


@dataclass
class TTSConfig:
    """Configuration for Text-to-Speech service"""
    model: str = "tts-1"
    voice: str = "alloy"  # alloy, echo, fable, onyx, nova, shimmer
    speed: float = 1.0  # 0.25 to 4.0
    response_format: str = "mp3"  # mp3, opus, aac, flac
    
    # Text processing settings
    max_text_length: int = 4096  # Maximum characters per request
    chunk_size: int = 1000  # Characters per chunk for long text
    
    # Audio settings
    sample_rate: int = 16000  # Output sample rate for compatibility
    normalize_audio: bool = True  # Normalize audio levels
    
    # API settings
    timeout: int = 15  # API timeout in seconds
    max_retries: int = 3
    retry_delay: float = 1.0  # Base delay for exponential backoff
    
    # Caching settings
    cache_enabled: bool = True
    cache_max_size: int = 100  # Maximum cached items
    cache_ttl: int = 3600  # Time to live in seconds


@dataclass
class TTSResult:
    """Result from text-to-speech conversion"""
    audio_data: bytes
    text: str
    duration: float
    processing_time: float
    format: str
    voice: str
    speed: float
    sample_rate: int
    cached: bool = False
    
    def save_to_file(self, filename: str) -> bool:
        """Save audio data to file"""
        try:
            with open(filename, 'wb') as f:
                f.write(self.audio_data)
            return True
        except Exception as e:
            logger.error(f"Error saving audio to file: {e}")
            return False


class AudioProcessor:
    """Handles audio processing and format conversion for TTS"""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        
    def convert_mp3_to_wav(self, mp3_data: bytes) -> bytes:
        """Convert MP3 audio data to WAV format"""
        try:
            # Create temporary file for MP3 data
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_mp3:
                temp_mp3.write(mp3_data)
                temp_mp3_path = temp_mp3.name
            
            try:
                # Load MP3 and convert to WAV
                audio = AudioSegment.from_mp3(temp_mp3_path)
                
                # Resample to target sample rate
                if audio.frame_rate != self.config.sample_rate:
                    audio = audio.set_frame_rate(self.config.sample_rate)
                
                # Convert to mono if needed
                if audio.channels > 1:
                    audio = audio.set_channels(1)
                
                # Normalize audio levels
                if self.config.normalize_audio:
                    audio = audio.normalize()
                
                # Export to WAV bytes
                wav_buffer = io.BytesIO()
                audio.export(wav_buffer, format="wav")
                wav_data = wav_buffer.getvalue()
                wav_buffer.close()
                
                return wav_data
                
            finally:
                # Clean up temporary file
                os.unlink(temp_mp3_path)
                
        except Exception as e:
            logger.error(f"Error converting MP3 to WAV: {e}")
            return None
    
    def calculate_audio_duration(self, audio_data: bytes, format: str = "mp3") -> float:
        """Calculate duration of audio data in seconds"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=f'.{format}', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                # Load audio and get duration
                if format == "mp3":
                    audio = AudioSegment.from_mp3(temp_file_path)
                elif format == "wav":
                    audio = AudioSegment.from_wav(temp_file_path)
                else:
                    audio = AudioSegment.from_file(temp_file_path)
                
                duration = len(audio) / 1000.0  # Convert ms to seconds
                return duration
                
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)
                
        except Exception as e:
            logger.error(f"Error calculating audio duration: {e}")
            return 0.0
    
    def chunk_text(self, text: str) -> List[str]:
        """Split long text into chunks for processing"""
        if len(text) <= self.config.chunk_size:
            return [text]
        
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size, start new chunk
            if current_chunk and len(current_chunk) + len(sentence) + 2 > self.config.chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def merge_audio_chunks(self, audio_chunks: List[bytes]) -> bytes:
        """Merge multiple audio chunks into a single audio file"""
        try:
            if not audio_chunks:
                return b""
            
            if len(audio_chunks) == 1:
                return audio_chunks[0]
            
            # Load first chunk
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_file.write(audio_chunks[0])
                temp_file_path = temp_file.name
            
            try:
                merged_audio = AudioSegment.from_mp3(temp_file_path)
                
                # Add remaining chunks
                for chunk in audio_chunks[1:]:
                    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_chunk:
                        temp_chunk.write(chunk)
                        temp_chunk_path = temp_chunk.name
                    
                    try:
                        chunk_audio = AudioSegment.from_mp3(temp_chunk_path)
                        merged_audio += chunk_audio
                    finally:
                        os.unlink(temp_chunk_path)
                
                # Export merged audio
                merged_buffer = io.BytesIO()
                merged_audio.export(merged_buffer, format="mp3")
                merged_data = merged_buffer.getvalue()
                merged_buffer.close()
                
                return merged_data
                
            finally:
                os.unlink(temp_file_path)
                
        except Exception as e:
            logger.error(f"Error merging audio chunks: {e}")
            return b""


class TTSCache:
    """Simple in-memory cache for TTS results"""
    
    def __init__(self, max_size: int = 100, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
    
    def _generate_key(self, text: str, voice: str, speed: float) -> str:
        """Generate cache key from text and voice parameters"""
        key_string = f"{text}_{voice}_{speed}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, text: str, voice: str, speed: float) -> Optional[TTSResult]:
        """Get cached TTS result"""
        key = self._generate_key(text, voice, speed)
        
        if key in self.cache:
            # Check if cache entry is still valid
            if time.time() - self.access_times[key] < self.ttl:
                self.access_times[key] = time.time()  # Update access time
                result = self.cache[key]
                result.cached = True
                return result
            else:
                # Remove expired entry
                del self.cache[key]
                del self.access_times[key]
        
        return None
    
    def put(self, text: str, voice: str, speed: float, result: TTSResult):
        """Store TTS result in cache"""
        key = self._generate_key(text, voice, speed)
        
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            # Remove the oldest entry
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = result
        self.access_times[key] = time.time()
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.access_times.clear()


class OpenAITTSService:
    """OpenAI Text-to-Speech service"""
    
    def __init__(self, config: TTSConfig, api_key: str, org_id: Optional[str] = None):
        self.config = config
        # Initialize OpenAI client with organization ID if provided
        client_kwargs = {"api_key": api_key}
        if org_id:
            client_kwargs["organization"] = org_id
        self.client = OpenAI(**client_kwargs)
        self.audio_processor = AudioProcessor(config)
        self.cache = TTSCache(config.cache_max_size, config.cache_ttl) if config.cache_enabled else None
        self.is_initialized = False
        
        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        
    def initialize(self) -> bool:
        """Initialize the TTS service"""
        try:
            logger.info("Initializing OpenAI TTS service...")
            
            # Validate configuration
            if not self._validate_config():
                return False
            
            # Mark as initialized - API connectivity will be tested on first use
            logger.info("OpenAI TTS service initialized successfully")
            self.is_initialized = True
            return True
                
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI TTS service: {e}")
            return False
    
    def _validate_config(self) -> bool:
        """Validate TTS configuration"""
        valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        valid_models = ["tts-1", "tts-1-hd"]
        valid_formats = ["mp3", "opus", "aac", "flac"]
        
        if self.config.voice not in valid_voices:
            logger.error(f"Invalid voice: {self.config.voice}. Must be one of: {valid_voices}")
            return False
        
        if self.config.model not in valid_models:
            logger.error(f"Invalid model: {self.config.model}. Must be one of: {valid_models}")
            return False
        
        if self.config.response_format not in valid_formats:
            logger.error(f"Invalid format: {self.config.response_format}. Must be one of: {valid_formats}")
            return False
        
        if not (0.25 <= self.config.speed <= 4.0):
            logger.error(f"Invalid speed: {self.config.speed}. Must be between 0.25 and 4.0")
            return False
        
        return True
    
    async def synthesize_speech(self, text: str) -> Optional[TTSResult]:
        """Convert text to speech"""
        if not self.is_initialized:
            logger.error("TTS service not initialized")
            return None
        
        if not text or not text.strip():
            logger.warning("Empty text provided for TTS")
            return None
        
        text = text.strip()
        start_time = time.time()
        
        try:
            # Check cache first
            if self.cache:
                cached_result = self.cache.get(text, self.config.voice, self.config.speed)
                if cached_result:
                    logger.debug(f"Cache hit for text: {text[:50]}...")
                    self.cache_hits += 1
                    return cached_result
                else:
                    self.cache_misses += 1
            
            # Process text
            if len(text) > self.config.max_text_length:
                logger.warning(f"Text too long: {len(text)} > {self.config.max_text_length}")
                return None
            
            # Handle long text by chunking
            if len(text) > self.config.chunk_size:
                return await self._synthesize_long_text(text, start_time)
            else:
                return await self._synthesize_single_chunk(text, start_time)
                
        except Exception as e:
            logger.error(f"Error in speech synthesis: {e}")
            self.failed_requests += 1
            return None
    
    async def _synthesize_single_chunk(self, text: str, start_time: float) -> Optional[TTSResult]:
        """Synthesize speech for a single text chunk"""
        try:
            # Make API request with retry
            audio_data = await self._synthesize_with_retry(text)
            if not audio_data:
                self.failed_requests += 1
                return None
            
            # Calculate duration
            duration = self.audio_processor.calculate_audio_duration(audio_data, self.config.response_format)
            
            # Create result
            processing_time = time.time() - start_time
            result = TTSResult(
                audio_data=audio_data,
                text=text,
                duration=duration,
                processing_time=processing_time,
                format=self.config.response_format,
                voice=self.config.voice,
                speed=self.config.speed,
                sample_rate=self.config.sample_rate
            )
            
            # Cache result
            if self.cache:
                self.cache.put(text, self.config.voice, self.config.speed, result)
            
            # Update statistics
            self.successful_requests += 1
            self.total_processing_time += processing_time
            
            logger.info(f"Successfully synthesized speech: {len(text)} chars, {duration:.2f}s audio, "
                       f"{processing_time:.2f}s processing")
            
            return result
            
        except Exception as e:
            logger.error(f"Error synthesizing single chunk: {e}")
            self.failed_requests += 1
            return None
    
    async def _synthesize_long_text(self, text: str, start_time: float) -> Optional[TTSResult]:
        """Synthesize speech for long text by chunking"""
        try:
            # Split text into chunks
            chunks = self.audio_processor.chunk_text(text)
            logger.info(f"Processing long text: {len(text)} chars in {len(chunks)} chunks")
            
            # Synthesize each chunk
            audio_chunks = []
            total_duration = 0.0
            
            for i, chunk in enumerate(chunks):
                logger.debug(f"Processing chunk {i+1}/{len(chunks)}: {len(chunk)} chars")
                
                chunk_audio = await self._synthesize_with_retry(chunk)
                if not chunk_audio:
                    logger.error(f"Failed to synthesize chunk {i+1}")
                    self.failed_requests += 1
                    return None
                
                audio_chunks.append(chunk_audio)
                chunk_duration = self.audio_processor.calculate_audio_duration(chunk_audio, self.config.response_format)
                total_duration += chunk_duration
            
            # Merge audio chunks
            merged_audio = self.audio_processor.merge_audio_chunks(audio_chunks)
            if not merged_audio:
                logger.error("Failed to merge audio chunks")
                return None
            
            # Create result
            processing_time = time.time() - start_time
            result = TTSResult(
                audio_data=merged_audio,
                text=text,
                duration=total_duration,
                processing_time=processing_time,
                format=self.config.response_format,
                voice=self.config.voice,
                speed=self.config.speed,
                sample_rate=self.config.sample_rate
            )
            
            # Update statistics
            self.successful_requests += 1
            self.total_processing_time += processing_time
            
            logger.info(f"Successfully synthesized long text: {len(text)} chars, {total_duration:.2f}s audio, "
                       f"{processing_time:.2f}s processing")
            
            return result
            
        except Exception as e:
            logger.error(f"Error synthesizing long text: {e}")
            self.failed_requests += 1
            return None
    
    async def _synthesize_with_retry(self, text: str) -> Optional[bytes]:
        """Synthesize speech with retry logic"""
        for attempt in range(self.config.max_retries + 1):
            try:
                self.total_requests += 1
                
                # Make API request
                response = self.client.audio.speech.create(
                    model=self.config.model,
                    voice=self.config.voice,
                    input=text,
                    speed=self.config.speed,
                    response_format=self.config.response_format
                )
                
                # Get audio data
                audio_data = response.content
                
                if audio_data:
                    logger.debug(f"Successfully synthesized {len(text)} chars (attempt {attempt + 1})")
                    return audio_data
                else:
                    logger.warning(f"Empty audio response (attempt {attempt + 1})")
                    
            except Exception as e:
                logger.warning(f"TTS API request failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.config.max_retries:
                    # Exponential backoff
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.config.max_retries + 1} attempts failed")
        
        return None
    
    def get_wav_audio(self, result: TTSResult) -> Optional[bytes]:
        """Convert TTS result to WAV format for playback"""
        if not result:
            return None
        
        if result.format == "wav":
            return result.audio_data
        
        # Convert to WAV
        wav_data = self.audio_processor.convert_mp3_to_wav(result.audio_data)
        return wav_data
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.successful_requests / max(self.total_requests, 1),
            "total_processing_time": self.total_processing_time,
            "average_processing_time": self.total_processing_time / max(self.successful_requests, 1),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
        }
    
    def reset_statistics(self):
        """Reset service statistics"""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def clear_cache(self):
        """Clear TTS cache"""
        if self.cache:
            self.cache.clear()
            logger.info("TTS cache cleared")


def create_openai_tts_service(api_key: str, config: Optional[TTSConfig] = None, org_id: Optional[str] = None) -> OpenAITTSService:
    """Factory function to create OpenAI TTS service"""
    if config is None:
        config = TTSConfig()
    
    service = OpenAITTSService(config, api_key, org_id)
    
    if not service.initialize():
        raise RuntimeError("Failed to initialize OpenAI TTS service")
    
    return service 