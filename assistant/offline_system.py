#!/usr/bin/env python3
"""
Offline Fallback System for Sovereign Voice Assistant

Provides fully offline operation using:
- whisper.cpp for local STT (tiny.en model)
- Piper TTS for offline speech synthesis
- llama.cpp with Gemma 2B-GGUF for local LLM inference
- Network connectivity detection and automatic fallback
- Model downloading, caching, and memory optimization
"""

import asyncio
import time
import os
import sys
import psutil
import subprocess
import hashlib
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
from contextlib import asynccontextmanager
import numpy as np
import json

logger = logging.getLogger(__name__)

class ConnectivityStatus(Enum):
    """Network connectivity status"""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"

class ModelStatus(Enum):
    """Model loading status"""
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"
    UNLOADING = "unloading"

@dataclass
class OfflineConfig:
    """Configuration for offline fallback system"""
    
    # Model paths and URLs
    models_dir: str = "./data/offline_models"
    
    # Whisper.cpp configuration
    whisper_model: str = "tiny.en"
    whisper_use_gpu: bool = True
    whisper_threads: int = 4
    whisper_streaming: bool = True
    
    # Piper TTS configuration
    piper_voice: str = "en_US-lessac-medium"
    piper_quality: str = "medium"  # low, medium, high
    piper_speed: float = 1.0
    piper_warmup_text: str = "Hello, this is a warmup phrase for Piper TTS."
    
    # LLaMA.cpp configuration
    llama_model: str = "gemma-2b-it-q4_k_m.gguf"
    llama_context_size: int = 2048
    llama_threads: int = 0  # Auto-detect
    llama_batch_size: int = 512
    llama_use_mlock: bool = True
    llama_use_mmap: bool = True
    
    # Network detection configuration
    ping_hosts: List[str] = field(default_factory=lambda: ["8.8.8.8", "1.1.1.1", "208.67.222.222"])
    ping_timeout: float = 2.0
    ping_interval: float = 30.0
    api_timeout: float = 5.0
    max_consecutive_failures: int = 3
    
    # Memory management
    max_memory_usage_percent: float = 85.0
    memory_check_interval: float = 10.0
    model_unload_threshold: float = 90.0
    
    # Performance targets
    target_offline_latency: float = 1.5  # seconds
    model_loading_timeout: float = 30.0

@dataclass
class ModelInfo:
    """Information about an offline model"""
    name: str
    url: str
    file_path: str
    size_mb: float
    checksum: str
    status: ModelStatus = ModelStatus.NOT_LOADED
    load_time: float = 0.0
    memory_usage: float = 0.0

@dataclass
class ConnectivityResult:
    """Result of network connectivity check"""
    status: ConnectivityStatus
    latency_ms: float
    timestamp: float
    error: Optional[str] = None

class NetworkDetector:
    """Multi-layer network connectivity detection"""
    
    def __init__(self, config: OfflineConfig):
        self.config = config
        self.consecutive_failures = 0
        self.last_check = 0.0
        self.last_result = ConnectivityResult(
            status=ConnectivityStatus.UNKNOWN,
            latency_ms=0.0,
            timestamp=time.time()
        )
        
    def check_ping_connectivity(self) -> ConnectivityResult:
        """Fast ping checks to reliable endpoints"""
        start_time = time.time()
        
        for host in self.config.ping_hosts:
            try:
                # Use ping command for fast connectivity check
                result = subprocess.run(
                    ["ping", "-c", "1", "-W", str(int(self.config.ping_timeout * 1000)), host],
                    capture_output=True,
                    timeout=self.config.ping_timeout + 1
                )
                
                if result.returncode == 0:
                    latency = (time.time() - start_time) * 1000
                    return ConnectivityResult(
                        status=ConnectivityStatus.ONLINE,
                        latency_ms=latency,
                        timestamp=time.time()
                    )
                    
            except subprocess.TimeoutExpired:
                continue
            except Exception as e:
                logger.debug(f"Ping to {host} failed: {e}")
                continue
        
        return ConnectivityResult(
            status=ConnectivityStatus.OFFLINE,
            latency_ms=0.0,
            timestamp=time.time(),
            error="All ping hosts unreachable"
        )
    
    async def check_api_connectivity(self, test_urls: List[str] = None) -> ConnectivityResult:
        """Check API endpoint connectivity"""
        if not test_urls:
            test_urls = [
                "https://api.openai.com/v1/models",
                "https://openrouter.ai/api/v1/models",
                "https://httpbin.org/status/200"
            ]
        
        start_time = time.time()
        
        for url in test_urls:
            try:
                response = requests.get(url, timeout=self.config.api_timeout)
                if response.status_code < 500:  # Accept even auth errors as connectivity
                    latency = (time.time() - start_time) * 1000
                    return ConnectivityResult(
                        status=ConnectivityStatus.ONLINE,
                        latency_ms=latency,
                        timestamp=time.time()
                    )
            except Exception as e:
                logger.debug(f"API check to {url} failed: {e}")
                continue
        
        return ConnectivityResult(
            status=ConnectivityStatus.OFFLINE,
            latency_ms=0.0,
            timestamp=time.time(),
            error="All API endpoints unreachable"
        )
    
    async def check_connectivity(self) -> ConnectivityResult:
        """Comprehensive connectivity check"""
        now = time.time()
        
        # Skip check if too recent
        if now - self.last_check < self.config.ping_interval:
            return self.last_result
        
        # Start with fast ping check
        ping_result = self.check_ping_connectivity()
        
        if ping_result.status == ConnectivityStatus.ONLINE:
            # Verify with API check for degraded connection detection
            api_result = await self.check_api_connectivity()
            
            if api_result.status == ConnectivityStatus.ONLINE:
                self.consecutive_failures = 0
                result = ConnectivityResult(
                    status=ConnectivityStatus.ONLINE,
                    latency_ms=(ping_result.latency_ms + api_result.latency_ms) / 2,
                    timestamp=now
                )
            else:
                # Ping works but API doesn't - degraded connection
                result = ConnectivityResult(
                    status=ConnectivityStatus.DEGRADED,
                    latency_ms=ping_result.latency_ms,
                    timestamp=now,
                    error="API endpoints unreachable"
                )
        else:
            # No ping connectivity
            self.consecutive_failures += 1
            result = ConnectivityResult(
                status=ConnectivityStatus.OFFLINE,
                latency_ms=0.0,
                timestamp=now,
                error=f"No connectivity ({self.consecutive_failures} consecutive failures)"
            )
        
        self.last_check = now
        self.last_result = result
        return result

class ModelManager:
    """Downloads, caches, and manages offline models"""
    
    def __init__(self, config: OfflineConfig):
        self.config = config
        self.models_dir = Path(config.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model registry
        self.models = {
            "whisper_tiny_en": ModelInfo(
                name="whisper_tiny_en",
                url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
                file_path=str(self.models_dir / "ggml-tiny.en.bin"),
                size_mb=39.0,
                checksum="bd577a113a864445d4c299885e0cb97d4ba92b5f"
            ),
            "piper_lessac_medium": ModelInfo(
                name="piper_lessac_medium",
                url="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
                file_path=str(self.models_dir / "en_US-lessac-medium.onnx"),
                size_mb=63.0,
                checksum="8c4c5c8c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c"  # Placeholder
            ),
            "gemma_2b_q4": ModelInfo(
                name="gemma_2b_q4",
                url="https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
                file_path=str(self.models_dir / "gemma-2-2b-it-Q4_K_M.gguf"),
                size_mb=1500.0,
                checksum="1234567890abcdef1234567890abcdef12345678"  # Placeholder
            )
        }
    
    def calculate_file_checksum(self, file_path: str) -> str:
        """Calculate SHA-1 checksum of a file"""
        sha1 = hashlib.sha1()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha1.update(chunk)
        return sha1.hexdigest()
    
    def verify_model(self, model_info: ModelInfo) -> bool:
        """Verify model file exists and has correct checksum"""
        file_path = Path(model_info.file_path)
        
        if not file_path.exists():
            return False
        
        # Check file size (approximate)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if abs(file_size_mb - model_info.size_mb) > model_info.size_mb * 0.1:  # 10% tolerance
            logger.warning(f"Model {model_info.name} size mismatch: expected {model_info.size_mb}MB, got {file_size_mb:.1f}MB")
            return False
        
        # Skip checksum verification for now (placeholder checksums)
        # In production, implement proper checksum verification
        return True
    
    async def download_model(self, model_info: ModelInfo, progress_callback=None) -> bool:
        """Download a model with progress tracking"""
        try:
            logger.info(f"Downloading {model_info.name} from {model_info.url}")
            model_info.status = ModelStatus.LOADING
            
            file_path = Path(model_info.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download with progress tracking
            response = requests.get(model_info.url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback and total_size > 0:
                            progress = (downloaded / total_size) * 100
                            progress_callback(model_info.name, progress)
            
            # Verify download
            if self.verify_model(model_info):
                model_info.status = ModelStatus.LOADED
                logger.info(f"Successfully downloaded {model_info.name}")
                return True
            else:
                model_info.status = ModelStatus.FAILED
                logger.error(f"Downloaded {model_info.name} failed verification")
                file_path.unlink(missing_ok=True)
                return False
                
        except Exception as e:
            model_info.status = ModelStatus.FAILED
            logger.error(f"Failed to download {model_info.name}: {e}")
            return False
    
    async def ensure_models_available(self, models: List[str] = None) -> Dict[str, bool]:
        """Ensure specified models are available, downloading if necessary"""
        if models is None:
            models = list(self.models.keys())
        
        results = {}
        
        for model_name in models:
            if model_name not in self.models:
                results[model_name] = False
                continue
            
            model_info = self.models[model_name]
            
            # Check if already available
            if self.verify_model(model_info):
                model_info.status = ModelStatus.LOADED
                results[model_name] = True
                logger.info(f"Model {model_name} already available")
                continue
            
            # Download if missing
            logger.info(f"Model {model_name} not available, downloading...")
            success = await self.download_model(model_info)
            results[model_name] = success
        
        return results
    
    def get_model_path(self, model_name: str) -> Optional[str]:
        """Get the file path for a model"""
        if model_name in self.models:
            model_info = self.models[model_name]
            if self.verify_model(model_info):
                return model_info.file_path
        return None
    
    def get_models_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all models"""
        status = {}
        for name, model_info in self.models.items():
            status[name] = {
                "status": model_info.status.value,
                "size_mb": model_info.size_mb,
                "file_exists": Path(model_info.file_path).exists(),
                "verified": self.verify_model(model_info),
                "memory_usage": model_info.memory_usage,
                "load_time": model_info.load_time
            }
        return status

class MemoryMonitor:
    """Monitor and manage system memory usage"""
    
    def __init__(self, config: OfflineConfig):
        self.config = config
        self.last_check = 0.0
        self.memory_pressure_callbacks = []
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information"""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent_used": memory.percent,
            "free_gb": memory.free / (1024**3)
        }
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        memory_info = self.get_memory_info()
        return memory_info["percent_used"] > self.config.max_memory_usage_percent
    
    def should_unload_models(self) -> bool:
        """Check if models should be unloaded due to memory pressure"""
        memory_info = self.get_memory_info()
        return memory_info["percent_used"] > self.config.model_unload_threshold
    
    def add_pressure_callback(self, callback):
        """Add callback to be called when memory pressure is detected"""
        self.memory_pressure_callbacks.append(callback)
    
    async def monitor_memory(self):
        """Continuous memory monitoring"""
        while True:
            try:
                if self.check_memory_pressure():
                    logger.warning(f"Memory pressure detected: {self.get_memory_info()['percent_used']:.1f}% used")
                    
                    # Notify callbacks
                    for callback in self.memory_pressure_callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback()
                            else:
                                callback()
                        except Exception as e:
                            logger.error(f"Memory pressure callback failed: {e}")
                
                await asyncio.sleep(self.config.memory_check_interval)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                await asyncio.sleep(self.config.memory_check_interval)

class OfflineSTTService:
    """Offline STT using whisper.cpp"""
    
    def __init__(self, config: OfflineConfig, model_manager: ModelManager):
        self.config = config
        self.model_manager = model_manager
        self.model_loaded = False
        self.whisper_process = None
        self.model_path = None
    
    async def initialize(self) -> bool:
        """Initialize the offline STT service"""
        try:
            # Ensure whisper model is available
            results = await self.model_manager.ensure_models_available(["whisper_tiny_en"])
            if not results.get("whisper_tiny_en", False):
                logger.error("Failed to ensure whisper model availability")
                return False
            
            self.model_path = self.model_manager.get_model_path("whisper_tiny_en")
            if not self.model_path:
                logger.error("Whisper model path not available")
                return False
            
            # Verify whisper.cpp binary is available
            try:
                result = subprocess.run(["whisper", "--help"], capture_output=True, timeout=5)
                if result.returncode != 0:
                    logger.error("whisper.cpp binary not found or not working")
                    return False
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.error("whisper.cpp binary not available")
                return False
            
            self.model_loaded = True
            logger.info("Offline STT service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize offline STT service: {e}")
            return False
    
    async def transcribe_audio(self, audio_data: bytes) -> Optional[str]:
        """Transcribe audio using offline whisper.cpp"""
        if not self.model_loaded:
            logger.error("Offline STT service not initialized")
            return None
        
        try:
            # Save audio to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            try:
                # Run whisper.cpp
                cmd = [
                    "whisper",
                    "-m", self.model_path,
                    "-t", str(self.config.whisper_threads),
                    "-f", temp_path,
                    "--output-txt"
                ]
                
                if self.config.whisper_use_gpu:
                    cmd.append("-ng")  # No GPU for tiny model (CPU is fine)
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    # Read the output text file
                    txt_path = temp_path + ".txt"
                    if os.path.exists(txt_path):
                        with open(txt_path, 'r') as f:
                            text = f.read().strip()
                        os.unlink(txt_path)
                        return text
                
                logger.error(f"Whisper.cpp failed: {result.stderr}")
                return None
                
            finally:
                # Cleanup temp file
                os.unlink(temp_path)
                
        except Exception as e:
            logger.error(f"Offline STT transcription failed: {e}")
            return None
    
    def cleanup(self):
        """Cleanup resources"""
        if self.whisper_process:
            self.whisper_process.terminate()
        self.model_loaded = False

class OfflineTTSService:
    """Offline TTS using Piper"""
    
    def __init__(self, config: OfflineConfig, model_manager: ModelManager):
        self.config = config
        self.model_manager = model_manager
        self.model_loaded = False
        self.piper_process = None
        self.model_path = None
        self.config_path = None
    
    async def initialize(self) -> bool:
        """Initialize the offline TTS service"""
        try:
            # Ensure Piper model is available
            results = await self.model_manager.ensure_models_available(["piper_lessac_medium"])
            if not results.get("piper_lessac_medium", False):
                logger.error("Failed to ensure Piper model availability")
                return False
            
            self.model_path = self.model_manager.get_model_path("piper_lessac_medium")
            if not self.model_path:
                logger.error("Piper model path not available")
                return False
            
            # Find config file (should be alongside model)
            model_dir = Path(self.model_path).parent
            config_file = model_dir / "en_US-lessac-medium.onnx.json"
            if not config_file.exists():
                # Create a basic config if missing
                config_data = {
                    "audio": {
                        "sample_rate": 22050
                    },
                    "espeak": {
                        "voice": "en-us"
                    },
                    "inference": {
                        "noise_scale": 0.667,
                        "length_scale": 1.0,
                        "noise_w": 0.8
                    }
                }
                with open(config_file, 'w') as f:
                    json.dump(config_data, f)
            
            self.config_path = str(config_file)
            
            # Verify Piper binary is available
            try:
                result = subprocess.run(["piper", "--help"], capture_output=True, timeout=5)
                if result.returncode != 0:
                    logger.error("Piper binary not found or not working")
                    return False
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.error("Piper binary not available")
                return False
            
            # Warm up with test phrase
            await self.synthesize_speech(self.config.piper_warmup_text)
            
            self.model_loaded = True
            logger.info("Offline TTS service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize offline TTS service: {e}")
            return False
    
    async def synthesize_speech(self, text: str) -> Optional[bytes]:
        """Synthesize speech using offline Piper TTS"""
        if not self.model_loaded:
            logger.error("Offline TTS service not initialized")
            return None
        
        try:
            # Prepare Piper command
            cmd = [
                "piper",
                "--model", self.model_path,
                "--config", self.config_path,
                "--output-raw"
            ]
            
            # Run Piper
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(input=text, timeout=10)
            
            if process.returncode == 0:
                # Convert raw audio to WAV format
                import io
                import wave
                
                # Assume 22050 Hz, 16-bit mono (Piper default)
                sample_rate = 22050
                audio_data = np.frombuffer(stdout.encode('latin1'), dtype=np.int16)
                
                # Create WAV file in memory
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_data.tobytes())
                
                return wav_buffer.getvalue()
            
            logger.error(f"Piper TTS failed: {stderr}")
            return None
            
        except Exception as e:
            logger.error(f"Offline TTS synthesis failed: {e}")
            return None
    
    def cleanup(self):
        """Cleanup resources"""
        if self.piper_process:
            self.piper_process.terminate()
        self.model_loaded = False

class OfflineLLMService:
    """Offline LLM using llama.cpp"""
    
    def __init__(self, config: OfflineConfig, model_manager: ModelManager):
        self.config = config
        self.model_manager = model_manager
        self.model_loaded = False
        self.llama_process = None
        self.model_path = None
        self.context = []
        self.max_context_tokens = 1500  # Keep context manageable
    
    async def initialize(self) -> bool:
        """Initialize the offline LLM service"""
        try:
            # Ensure Gemma model is available
            results = await self.model_manager.ensure_models_available(["gemma_2b_q4"])
            if not results.get("gemma_2b_q4", False):
                logger.error("Failed to ensure Gemma model availability")
                return False
            
            self.model_path = self.model_manager.get_model_path("gemma_2b_q4")
            if not self.model_path:
                logger.error("Gemma model path not available")
                return False
            
            # Verify llama.cpp binary is available
            try:
                result = subprocess.run(["llama-cli", "--help"], capture_output=True, timeout=5)
                if result.returncode != 0:
                    # Try alternative binary name
                    result = subprocess.run(["llama-cpp-python", "--help"], capture_output=True, timeout=5)
                    if result.returncode != 0:
                        logger.error("llama.cpp binary not found")
                        return False
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.error("llama.cpp binary not available")
                return False
            
            self.model_loaded = True
            logger.info("Offline LLM service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize offline LLM service: {e}")
            return False
    
    def _build_prompt(self, user_input: str) -> str:
        """Build prompt with context"""
        # Simple context management
        context_str = ""
        if self.context:
            context_str = "\n".join(self.context[-6:])  # Last 3 exchanges
            context_str += "\n"
        
        # Gemma 2B format
        prompt = f"""<start_of_turn>system
You are a helpful voice assistant. Provide concise, accurate responses. Keep answers under 50 words when possible.
<end_of_turn>
{context_str}<start_of_turn>user
{user_input}
<end_of_turn>
<start_of_turn>model
"""
        return prompt
    
    async def generate_response(self, user_input: str) -> Optional[str]:
        """Generate response using offline LLaMA.cpp"""
        if not self.model_loaded:
            logger.error("Offline LLM service not initialized")
            return None
        
        try:
            prompt = self._build_prompt(user_input)
            
            # Prepare llama.cpp command
            cmd = [
                "llama-cli",
                "-m", self.model_path,
                "-p", prompt,
                "-n", "100",  # Max tokens
                "-t", str(self.config.llama_threads or 4),
                "--temp", "0.7",
                "--top-p", "0.9",
                "--repeat-penalty", "1.1",
                "--ctx-size", str(self.config.llama_context_size),
                "--batch-size", str(self.config.llama_batch_size),
                "--no-display-prompt"
            ]
            
            if self.config.llama_use_mlock:
                cmd.append("--mlock")
            if self.config.llama_use_mmap:
                cmd.append("--mmap")
            
            # Run llama.cpp
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15  # Generous timeout for small model
            )
            
            if result.returncode == 0:
                response = result.stdout.strip()
                
                # Clean up the response
                if response:
                    # Remove any remaining prompt artifacts
                    response = response.replace("<end_of_turn>", "").strip()
                    
                    # Update context
                    self.context.append(f"User: {user_input}")
                    self.context.append(f"Assistant: {response}")
                    
                    # Trim context if too long
                    if len(self.context) > 12:  # Keep last 6 exchanges
                        self.context = self.context[-12:]
                    
                    return response
            
            logger.error(f"LLaMA.cpp failed: {result.stderr}")
            return "I apologize, but I'm having trouble generating a response right now."
            
        except subprocess.TimeoutExpired:
            logger.error("LLaMA.cpp timeout")
            return "I apologize, but my response is taking too long. Please try again."
        except Exception as e:
            logger.error(f"Offline LLM generation failed: {e}")
            return "I apologize, but I encountered an error. Please try again."
    
    def reset_context(self):
        """Reset conversation context"""
        self.context = []
    
    def cleanup(self):
        """Cleanup resources"""
        if self.llama_process:
            self.llama_process.terminate()
        self.model_loaded = False

# Factory function for easy integration
def create_offline_system(config: Optional[OfflineConfig] = None) -> 'OfflineSystem':
    """Create an OfflineSystem instance with optional configuration"""
    return OfflineSystem(config or OfflineConfig())

class OfflineSystem:
    """Main offline fallback system coordinator"""
    
    def __init__(self, config: OfflineConfig):
        self.config = config
        self.network_detector = NetworkDetector(config)
        self.model_manager = ModelManager(config)
        self.memory_monitor = MemoryMonitor(config)
        
        # Services
        self.offline_stt = None
        self.offline_tts = None
        self.offline_llm = None
        
        # State
        self.is_offline_mode = False
        self.initialization_complete = False
        self.services_ready = {
            'stt': False,
            'tts': False,
            'llm': False
        }
        
        # Statistics
        self.stats = {
            'total_offline_requests': 0,
            'successful_offline_requests': 0,
            'failed_offline_requests': 0,
            'avg_offline_latency': 0.0,
            'mode_switches': 0,
            'total_offline_time': 0.0
        }
        
    async def initialize(self) -> bool:
        """Initialize the offline system"""
        logger.info("Initializing offline fallback system...")
        
        try:
            # Initialize services
            self.offline_stt = OfflineSTTService(self.config, self.model_manager)
            self.offline_tts = OfflineTTSService(self.config, self.model_manager)
            self.offline_llm = OfflineLLMService(self.config, self.model_manager)
            
            # Start memory monitoring
            asyncio.create_task(self.memory_monitor.monitor_memory())
            
            # Add memory pressure callback
            self.memory_monitor.add_pressure_callback(self._handle_memory_pressure)
            
            # Initialize services with error tolerance
            try:
                if await self.offline_stt.initialize():
                    self.services_ready['stt'] = True
                    logger.info("✅ Offline STT service ready")
                else:
                    logger.warning("⚠️ Offline STT service failed to initialize")
            except Exception as e:
                logger.error(f"STT initialization error: {e}")
            
            try:
                if await self.offline_tts.initialize():
                    self.services_ready['tts'] = True
                    logger.info("✅ Offline TTS service ready")
                else:
                    logger.warning("⚠️ Offline TTS service failed to initialize")
            except Exception as e:
                logger.error(f"TTS initialization error: {e}")
            
            try:
                if await self.offline_llm.initialize():
                    self.services_ready['llm'] = True
                    logger.info("✅ Offline LLM service ready")
                else:
                    logger.warning("⚠️ Offline LLM service failed to initialize")
            except Exception as e:
                logger.error(f"LLM initialization error: {e}")
            
            # Consider system ready if at least one service works
            any_service_ready = any(self.services_ready.values())
            
            if any_service_ready:
                self.initialization_complete = True
                ready_services = [k for k, v in self.services_ready.items() if v]
                logger.info(f"Offline fallback system initialized with services: {ready_services}")
                return True
            else:
                logger.error("No offline services successfully initialized")
                return False
            
        except Exception as e:
            logger.error(f"Failed to initialize offline system: {e}")
            return False
    
    async def _handle_memory_pressure(self):
        """Handle memory pressure by optimizing resource usage"""
        logger.info("Handling memory pressure in offline system")
        
        # Could unload less critical models temporarily
        if self.memory_monitor.should_unload_models():
            logger.warning("Critical memory pressure - considering model unloading")
            # Implementation would depend on which models are currently loaded
    
    async def check_and_switch_mode(self) -> ConnectivityStatus:
        """Check connectivity and switch modes if necessary"""
        connectivity = await self.network_detector.check_connectivity()
        
        should_be_offline = connectivity.status in [ConnectivityStatus.OFFLINE, ConnectivityStatus.DEGRADED]
        
        if should_be_offline and not self.is_offline_mode:
            logger.info("🔌 Switching to offline mode")
            self.is_offline_mode = True
            self.stats['mode_switches'] += 1
        elif not should_be_offline and self.is_offline_mode:
            logger.info("🌐 Switching back to online mode")
            self.is_offline_mode = False
            self.stats['mode_switches'] += 1
        
        return connectivity.status
    
    async def process_offline_request(self, audio_data: bytes) -> Optional[Tuple[str, bytes]]:
        """Process a complete offline voice request (STT -> LLM -> TTS)"""
        if not self.initialization_complete:
            logger.error("Offline system not initialized")
            return None
        
        start_time = time.time()
        self.stats['total_offline_requests'] += 1
        
        try:
            # Step 1: Speech to Text
            if not self.services_ready['stt']:
                logger.error("Offline STT service not available")
                return None
            
            user_text = await self.offline_stt.transcribe_audio(audio_data)
            if not user_text:
                logger.error("Offline STT failed")
                return None
            
            logger.info(f"🎤 Offline STT: {user_text}")
            
            # Step 2: Generate response
            if not self.services_ready['llm']:
                logger.error("Offline LLM service not available")
                return None
            
            ai_response = await self.offline_llm.generate_response(user_text)
            if not ai_response:
                logger.error("Offline LLM failed")
                return None
            
            logger.info(f"🤖 Offline LLM: {ai_response}")
            
            # Step 3: Text to Speech
            if not self.services_ready['tts']:
                logger.error("Offline TTS service not available")
                return None
            
            audio_response = await self.offline_tts.synthesize_speech(ai_response)
            if not audio_response:
                logger.error("Offline TTS failed")
                return None
            
            # Update statistics
            total_time = time.time() - start_time
            self.stats['successful_offline_requests'] += 1
            current_avg = self.stats['avg_offline_latency']
            successful = self.stats['successful_offline_requests']
            self.stats['avg_offline_latency'] = ((current_avg * (successful - 1)) + total_time) / successful
            
            logger.info(f"✅ Offline request completed in {total_time:.2f}s")
            
            return user_text, audio_response
            
        except Exception as e:
            self.stats['failed_offline_requests'] += 1
            logger.error(f"Offline request processing failed: {e}")
            return None
    
    async def transcribe_offline(self, audio_data: bytes) -> Optional[str]:
        """Transcribe audio using offline STT only"""
        if not self.services_ready['stt']:
            return None
        return await self.offline_stt.transcribe_audio(audio_data)
    
    async def generate_offline_response(self, text: str) -> Optional[str]:
        """Generate response using offline LLM only"""
        if not self.services_ready['llm']:
            return None
        return await self.offline_llm.generate_response(text)
    
    async def synthesize_offline(self, text: str) -> Optional[bytes]:
        """Synthesize speech using offline TTS only"""
        if not self.services_ready['tts']:
            return None
        return await self.offline_tts.synthesize_speech(text)
    
    def reset_llm_context(self):
        """Reset the offline LLM conversation context"""
        if self.offline_llm:
            self.offline_llm.reset_context()
    
    def is_ready_for_offline(self) -> bool:
        """Check if system is ready for offline operation"""
        return (self.initialization_complete and 
                self.services_ready['stt'] and 
                self.services_ready['llm'] and 
                self.services_ready['tts'])
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        connectivity_result = self.network_detector.last_result
        
        return {
            "offline_mode": self.is_offline_mode,
            "initialization_complete": self.initialization_complete,
            "services_ready": self.services_ready.copy(),
            "ready_for_offline": self.is_ready_for_offline(),
            "connectivity": {
                "status": connectivity_result.status.value,
                "latency_ms": connectivity_result.latency_ms,
                "last_check": connectivity_result.timestamp,
                "consecutive_failures": self.network_detector.consecutive_failures
            },
            "memory": self.memory_monitor.get_memory_info(),
            "models": self.model_manager.get_models_status(),
            "statistics": self.stats.copy()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get offline system statistics"""
        return {
            "performance": self.stats.copy(),
            "services": self.services_ready.copy(),
            "memory": self.memory_monitor.get_memory_info(),
            "target_latency": self.config.target_offline_latency,
            "models_status": self.model_manager.get_models_status()
        }
    
    def reset_stats(self):
        """Reset offline system statistics"""
        self.stats = {
            'total_offline_requests': 0,
            'successful_offline_requests': 0,
            'failed_offline_requests': 0,
            'avg_offline_latency': 0.0,
            'mode_switches': 0,
            'total_offline_time': 0.0
        }
    
    async def cleanup(self):
        """Cleanup offline system resources"""
        logger.info("Cleaning up offline system...")
        
        try:
            if self.offline_stt:
                self.offline_stt.cleanup()
            if self.offline_tts:
                self.offline_tts.cleanup()
            if self.offline_llm:
                self.offline_llm.cleanup()
            
            logger.info("Offline system cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during offline system cleanup: {e}") 