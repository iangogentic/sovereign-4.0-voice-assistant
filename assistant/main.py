#!/usr/bin/env python3
"""
Sovereign 4.0 Voice Assistant Main Entry Point

This is the main entry point for the voice assistant. It handles:
- Configuration loading
- System initialization
- Voice pipeline integration
- Main event loop
- Graceful shutdown
"""

import asyncio
import argparse
import logging
import signal
import sys
import os
import threading
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from dotenv import load_dotenv

# Import our core components
from .audio import AudioManager, AudioConfig, create_audio_manager
from .stt import WhisperSTTService, STTConfig, create_whisper_stt_service
from .tts import OpenAITTSService, TTSConfig, create_openai_tts_service
from .monitoring import PerformanceMonitor, get_monitor, set_monitor
from .dashboard import ConsoleDashboard, create_dashboard
# Import LLM Router components for intelligent AI responses
from .llm_router import LLMRouter
from .router_config import get_router_config

# Import Memory System components for conversation memory
from .memory import MemoryManager, MemoryConfig, create_memory_manager

# Import Screen Watcher components for screen awareness
from .screen_watcher import ScreenWatcher, ScreenWatcherConfig, create_screen_watcher

# Import Kimi K2 Code Agent for code operations
from .kimi_agent import KimiK2Agent, KimiConfig, create_kimi_agent

# Import Offline Fallback System for network-independent operation
from .offline_system import OfflineSystem, OfflineConfig, ConnectivityStatus, create_offline_system

# Import Hybrid Voice System for ultra-low latency Realtime API
from .hybrid_voice_system import HybridVoiceSystem, HybridConfig, VoiceMode
from .realtime_voice import RealtimeConfig

# Import comprehensive error handling system
from .error_handling import (
    ErrorCategory, ErrorContext, VoiceAIException, STTException, LLMException, TTSException,
    AudioException, BackoffConfig, retry_with_backoff, CircuitBreakerConfig, ModernCircuitBreaker,
    ServiceTier, ServiceCapability, GracefulDegradationManager, AllTiersFailedError
)
from .structured_logging import (
    VoiceAILogger, voice_ai_request_context, get_voice_ai_logger, 
    log_audio_processing, log_model_inference
)
from .health_monitoring import (
    SystemHealthMonitor, get_health_monitor, stt_health_check, llm_health_check,
    tts_health_check, offline_system_health_check
)
# Import the new configuration management system
from .config_manager import ConfigManager, get_config_manager, SovereignConfig
# Import the new metrics collection system
from .metrics_collector import MetricsCollector, get_metrics_collector, ComponentType

# Import the web dashboard server
from .dashboard_server import DashboardServer, create_dashboard_server
import sounddevice as sd
import numpy as np
import io
import wave
import time
import threading
import uuid


def setup_logging(debug: bool = False, log_file: Optional[str] = None) -> None:
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Setup file handler if specified
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logging.warning(f"Configuration file not found: {config_path}, using defaults")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        return {}


def validate_environment(config: Optional[SovereignConfig] = None) -> bool:
    """Validate required environment variables are set based on configuration"""
    # If no config is provided, load it
    if config is None:
        try:
            config_manager = get_config_manager()
            config = config_manager.get_config()
        except Exception as e:
            logging.error(f"Failed to load configuration for validation: {e}")
            return False
    
    # Skip validation if configured to do so
    if config.development.bypass_api_validation:
        logging.info("API validation bypassed (development mode)")
        return True
    
    missing_vars = []
    
    # Check OpenAI API key (required for STT/TTS/embeddings)
    if not config.api.openai_api_key and not os.getenv('OPENAI_API_KEY'):
        missing_vars.append("OPENAI_API_KEY")
    
    # Check OpenRouter API key (required for LLM routing)
    if not config.api.openrouter_api_key and not os.getenv('OPENROUTER_API_KEY'):
        missing_vars.append("OPENROUTER_API_KEY")
    
    # Check Kimi API key (required if code agent is enabled)
    if config.code_agent.enabled and not config.api.kimi_api_key and not os.getenv('KIMI_API_KEY'):
        missing_vars.append("KIMI_API_KEY")
    
    # Check other optional API keys based on configuration
    if config.memory.embedding_provider == "openai" and not config.api.openai_api_key and not os.getenv('OPENAI_API_KEY'):
        if "OPENAI_API_KEY" not in missing_vars:
            missing_vars.append("OPENAI_API_KEY")
    
    if missing_vars:
        logging.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logging.error("Please check your .env file and ensure all required API keys are set")
        return False
    
    return True


class SovereignAssistant:
    """Main Sovereign Voice Assistant class - using direct service calls like the working demo"""
    
    def __init__(self, config_path: str = "config/sovereign.yaml", debug: bool = False):
        self.config_path = config_path
        self.debug = debug
        self.config: Optional[SovereignConfig] = None
        self.config_manager: Optional[ConfigManager] = None
        self.running = False
        
        # Initialize structured logging
        self.logger = get_voice_ai_logger("sovereign_assistant")
        
        # Initialize metrics collection
        self.metrics_collector: Optional[MetricsCollector] = None
        
        # Core components (direct services, no pipeline)
        self.stt_service: Optional[WhisperSTTService] = None
        self.tts_service: Optional[OpenAITTSService] = None
        self.monitor: Optional[PerformanceMonitor] = None
        self.dashboard: Optional[ConsoleDashboard] = None
        self.dashboard_server: Optional[DashboardServer] = None
        self.llm_router: Optional[LLMRouter] = None
        self.memory_manager = None
        self.screen_watcher = None
        self.kimi_agent: Optional[KimiK2Agent] = None
        self.offline_system: Optional[OfflineSystem] = None
        
        # Hybrid Voice System for ultra-low latency Realtime API
        self.hybrid_voice_system: Optional[HybridVoiceSystem] = None
        self.use_realtime_api: bool = False  # Flag to enable Realtime API mode
        
        # Error handling and monitoring components
        self.health_monitor: Optional[SystemHealthMonitor] = None
        self.degradation_manager: Optional[GracefulDegradationManager] = None
        self.circuit_breakers: Dict[str, ModernCircuitBreaker] = {}
        
        # Shutdown flag
        self.shutdown_requested = False
        self.services_ready = False
        
    def _load_config(self) -> bool:
        """Load configuration using the new ConfigManager"""
        try:
            self.config_manager = ConfigManager(config_path=self.config_path)
            self.config = self.config_manager.load_config()
            
            # Enable hot reload in development mode
            if self.config.development.enable_hot_reload:
                self.config_manager.enable_hot_reload()
                self.logger.info("🔄 Hot reload enabled for configuration files")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return False
    
    def _initialize_services(self):
        """Initialize STT and TTS services using the new configuration system"""
        self.logger.info("🚀 Initializing Sovereign Voice Assistant services...")
        
        # Initialize STT service with configuration from SovereignConfig
        stt_config = STTConfig(
            model=self.config.stt.primary_model,
            language=self.config.stt.primary_language,
            silence_threshold=self.config.audio.silence_threshold,
            min_audio_length=self.config.audio.min_audio_length,
            vad_enabled=self.config.audio.vad_enabled,
            temperature=self.config.stt.temperature
        )
        self.stt_service = WhisperSTTService(
            config=stt_config,
            api_key=self.config.api.openai_api_key or os.getenv('OPENAI_API_KEY')
        )
        self.stt_service.initialize()
        
        # Initialize TTS service with configuration from SovereignConfig
        tts_config = TTSConfig(
            model=self.config.tts.primary_model,
            voice=self.config.tts.primary_voice,
            speed=self.config.tts.primary_speed,
            response_format=self.config.tts.response_format
        )
        self.tts_service = OpenAITTSService(
            config=tts_config,
            api_key=self.config.api.openai_api_key or os.getenv('OPENAI_API_KEY')
        )
        self.tts_service.initialize()

        # Initialize LLM Router
        router_config = get_router_config()
        self.llm_router = LLMRouter(config=router_config)
        
        # Initialize Memory System with configuration
        memory_config = MemoryConfig(
            persist_directory=self.config.memory.persist_directory,
            collection_name_conversations=self.config.memory.collection_name_conversations,
            collection_name_screen=self.config.memory.collection_name_screen,
            embedding_model=self.config.memory.embedding_model,
            retrieval_k=self.config.memory.retrieval_k,
            similarity_threshold=self.config.memory.similarity_threshold,
            max_context_length=self.config.memory.max_context_length
        )
        self.memory_manager = create_memory_manager(memory_config)
        
        # Initialize Screen Watcher with configuration
        screen_config = ScreenWatcherConfig(
            monitor_interval=getattr(self.config.screen, 'screenshot_interval', 3.0),
            language=getattr(self.config.screen, 'ocr_language', 'eng'),
            tesseract_config=getattr(self.config.screen, 'ocr_config', '--psm 6'),
            focus_active_window=getattr(self.config.screen, 'focus_active_window', True),
            enable_preprocessing=True
        )
        self.screen_watcher = create_screen_watcher(screen_config, self.memory_manager)
        
        # Initialize Kimi K2 Code Agent with configuration
        kimi_config = KimiConfig(
            api_key=self.config.api.kimi_api_key,
            model_id=self.config.code_agent.model,
            max_tokens=self.config.code_agent.max_tokens,
            temperature=self.config.code_agent.temperature,
            timeout=self.config.code_agent.timeout
        )
        self.kimi_agent = create_kimi_agent(kimi_config) if self.config.code_agent.enabled else None
        
        # Initialize Offline Fallback System with configuration
        offline_config = OfflineConfig(
            models_dir="./data/offline_models",
            whisper_model=self.config.stt.fallback_model,
            piper_voice=self.config.tts.fallback_voice,
            llama_model=self.config.llm.local.model
        )
        self.offline_system = create_offline_system(offline_config)
        
        # Initialize Hybrid Voice System for Realtime API (optional)
        self._initialize_hybrid_voice_system()
        
        # Initialize error handling and monitoring components
        self._setup_error_handling()
        
        # Initialize advanced metrics collection
        self._setup_metrics_collection()
        
        self.services_ready = True
        self.logger.info("✅ Sovereign Voice Assistant services ready!")
    
    def _initialize_hybrid_voice_system(self):
        """Initialize the Hybrid Voice System for Realtime API support"""
        try:
            # Check if we should enable Realtime API
            enable_realtime = os.getenv('ENABLE_REALTIME_API', 'false').lower() == 'true'
            
            if not enable_realtime:
                self.logger.info("🔄 Realtime API disabled - using traditional voice pipeline")
                return
            
            # Configure Realtime API
            realtime_config = RealtimeConfig(
                api_key=self.config.api.openai_api_key or os.getenv('OPENAI_API_KEY'),
                model="gpt-4o-realtime-preview-2024-10-01",
                voice=self.config.tts.primary_voice or "alloy",
                instructions="You are Sovereign, an advanced AI voice assistant with screen awareness and memory capabilities. Provide helpful, natural responses.",
                temperature=0.8,
                max_context_length=8000,
                include_screen_content=True,
                include_memory_context=True
            )
            
            # Configure Hybrid System 
            hybrid_config = HybridConfig(
                voice_mode=VoiceMode.HYBRID_AUTO,  # Automatically choose best system
                max_realtime_failures=3,
                fallback_on_high_latency=True,
                max_acceptable_latency=2.0,
                monitor_performance=True,
                preserve_context_on_switch=True
            )
            
            # Create Hybrid Voice System
            self.hybrid_voice_system = HybridVoiceSystem(
                hybrid_config=hybrid_config,
                realtime_config=realtime_config,
                openai_api_key=self.config.api.openai_api_key or os.getenv('OPENAI_API_KEY'),
                openrouter_api_key=self.config.api.openrouter_api_key or os.getenv('OPENROUTER_API_KEY'),
                memory_manager=self.memory_manager,
                screen_watcher=self.screen_watcher,
                logger=self.logger
            )
            
            self.use_realtime_api = True
            self.logger.info("✅ Hybrid Voice System initialized - Realtime API available")
            
        except Exception as e:
            self.logger.warning(f"⚠️  Could not initialize Hybrid Voice System: {e}")
            self.logger.info("🔄 Falling back to traditional voice pipeline")
            self.hybrid_voice_system = None
            self.use_realtime_api = False
    
    def _setup_error_handling(self):
        """Set up comprehensive error handling system"""
        self.logger.info("🛡️  Setting up error handling and monitoring...")
        
        # Initialize health monitor
        self.health_monitor = get_health_monitor()
        
        # Initialize graceful degradation manager
        self.degradation_manager = GracefulDegradationManager()
        
        # Create circuit breakers for each service
        cb_config = CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=3,
            timeout_duration=60.0,
            failure_rate_threshold=0.5
        )
        
        # STT Circuit Breaker
        self.circuit_breakers['stt'] = ModernCircuitBreaker('stt', cb_config)
        
        # LLM Circuit Breaker
        self.circuit_breakers['llm'] = ModernCircuitBreaker('llm', cb_config)
        
        # TTS Circuit Breaker
        self.circuit_breakers['tts'] = ModernCircuitBreaker('tts', cb_config)
        
        # Offline System Circuit Breaker
        self.circuit_breakers['offline'] = ModernCircuitBreaker('offline', cb_config)
        
        # Register services with health monitor
        self.health_monitor.register_service(
            'stt',
            stt_health_check,
            self.circuit_breakers['stt'],
            check_interval=30.0
        )
        
        self.health_monitor.register_service(
            'llm',
            llm_health_check,
            self.circuit_breakers['llm'],
            check_interval=30.0
        )
        
        self.health_monitor.register_service(
            'tts',
            tts_health_check,
            self.circuit_breakers['tts'],
            check_interval=30.0
        )
        
        self.health_monitor.register_service(
            'offline',
            offline_system_health_check,
            self.circuit_breakers['offline'],
            check_interval=60.0
        )
        
        # Register service tiers for graceful degradation
        # STT Service Tiers
        self.degradation_manager.register_service(
            'stt',
            ServiceTier.PRIMARY,
            ServiceCapability('openai-whisper', ServiceTier.PRIMARY, 2.0, 0.95, 0.90, 1.0),
            self.circuit_breakers['stt']
        )
        
        self.degradation_manager.register_service(
            'stt',
            ServiceTier.OFFLINE,
            ServiceCapability('whisper-local', ServiceTier.OFFLINE, 3.0, 0.85, 0.95, 0.0),
            self.circuit_breakers['offline']
        )
        
        # LLM Service Tiers
        self.degradation_manager.register_service(
            'llm',
            ServiceTier.PRIMARY,
            ServiceCapability('openrouter-gpt4o', ServiceTier.PRIMARY, 3.0, 0.95, 0.90, 2.0),
            self.circuit_breakers['llm']
        )
        
        self.degradation_manager.register_service(
            'llm',
            ServiceTier.FAST,
            ServiceCapability('openrouter-gpt4o-mini', ServiceTier.FAST, 1.5, 0.85, 0.95, 0.5),
            self.circuit_breakers['llm']
        )
        
        self.degradation_manager.register_service(
            'llm',
            ServiceTier.OFFLINE,
            ServiceCapability('gemma-local', ServiceTier.OFFLINE, 5.0, 0.70, 0.95, 0.0),
            self.circuit_breakers['offline']
        )
        
        # TTS Service Tiers
        self.degradation_manager.register_service(
            'tts',
            ServiceTier.PRIMARY,
            ServiceCapability('openai-tts', ServiceTier.PRIMARY, 2.0, 0.95, 0.90, 1.0),
            self.circuit_breakers['tts']
        )
        
        self.degradation_manager.register_service(
            'tts',
            ServiceTier.OFFLINE,
            ServiceCapability('piper-local', ServiceTier.OFFLINE, 3.0, 0.85, 0.95, 0.0),
            self.circuit_breakers['offline']
        )
        
        self.logger.info("✅ Error handling and monitoring setup complete!")
    
    def _setup_metrics_collection(self):
        """Initialize and configure the advanced metrics collection system"""
        setup_start = time.time()
        
        try:
            # Initialize metrics collector with configuration
            collection_interval = getattr(self.config.monitoring, 'metrics_collection_interval', 1.0)
            self.metrics_collector = MetricsCollector(
                collection_interval=collection_interval,
                max_history_hours=24
            )
            
            # Set performance targets based on configuration
            if hasattr(self.config.monitoring, 'performance_targets'):
                self.metrics_collector.performance_targets.update(self.config.monitoring.performance_targets)
            
            # Add alert callback for integration with error handling
            def metrics_alert_handler(alert_type: str, alert_data: Dict[str, Any]):
                """Handle metrics alerts by logging and potentially triggering error handling"""
                self.logger.warning(f"📊 Metrics Alert: {alert_type}", extra={
                    'alert_type': alert_type,
                    'alert_data': alert_data,
                    'component': 'metrics_collector'
                })
                
                # Trigger circuit breakers for severe performance issues
                if alert_type in ['latency_anomaly', 'high_cpu_usage', 'high_memory_usage']:
                    component_map = {
                        'stt_processing': 'stt',
                        'llm_inference': 'llm', 
                        'tts_generation': 'tts'
                    }
                    
                    component_name = alert_data.get('data', {}).get('component')
                    if component_name in component_map:
                        circuit_breaker = self.circuit_breakers.get(component_map[component_name])
                        if circuit_breaker:
                            # Record failure to potentially open circuit breaker
                            circuit_breaker.record_failure()
            
            self.metrics_collector.add_alert_callback(metrics_alert_handler)
            
            # Start the metrics collection
            self.metrics_collector.start()
            
            # Set as global instance
            from .metrics_collector import set_metrics_collector
            set_metrics_collector(self.metrics_collector)
            
            self.logger.info(f"📊 Advanced metrics collection system initialized and started")
            
        except Exception as e:
            self.logger.error(f"Failed to setup metrics collection: {e}")
            # Continue without metrics collection - not critical for core functionality
            self.metrics_collector = None
        
        # Log setup performance
        setup_time = time.time() - setup_start
        self.logger.log_performance_metrics(
            'metrics_collection_setup',
            {
                'setup_duration': setup_time,
                'collector_initialized': self.metrics_collector is not None
            }
        )
    
    async def record_with_realtime_vad(self, max_duration=6.0):
        """Record audio with real-time voice activity detection (from working demo)"""
        self.logger.info("🎤 Listening... (I'll stop when you stop talking)")
        
        # Recording parameters
        sample_rate = 16000
        chunk_size = 256  # Small chunks for responsiveness
        silence_threshold = 0.001  # Fixed VAD threshold that works
        silence_duration = 0.8  # Stop after 0.8s of silence
        
        # Recording state
        audio_chunks = []
        silence_chunks = 0
        max_silence_chunks = int(silence_duration * sample_rate / chunk_size)
        max_chunks = int(max_duration * sample_rate / chunk_size)
        chunk_count = 0
        
        # Start recording
        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=chunk_size
        )
        
        with stream:
            while chunk_count < max_chunks:
                # Read audio chunk
                audio_chunk, overflowed = stream.read(chunk_size)
                
                if overflowed:
                    continue
                
                # Convert to int16 and store
                audio_int16 = (audio_chunk * 32767).astype(np.int16)
                audio_chunks.append(audio_int16)
                
                # Calculate RMS for voice activity detection
                rms = np.sqrt(np.mean(audio_chunk ** 2))
                
                if rms < silence_threshold:
                    silence_chunks += 1
                    if silence_chunks >= max_silence_chunks:
                        # Extended silence detected - stop recording
                        self.logger.info(f"🔇 Silence detected, stopping recording ({chunk_count * chunk_size / sample_rate:.1f}s)")
                        break
                else:
                    # Voice detected - reset silence counter
                    silence_chunks = 0
                
                chunk_count += 1
        
        # Convert to bytes
        if audio_chunks:
            audio_array = np.concatenate(audio_chunks)
            return audio_array.tobytes()
        else:
            return b''
    
    async def play_audio_fast(self, wav_audio: bytes):
        """Play WAV audio with minimal delay (from working demo)"""
        if not wav_audio:
            return
            
        try:
            # Play the audio using the same method as the working demo
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(24000)
                wav_file.writeframes(wav_audio)
            
            wav_buffer.seek(0)
            with wave.open(wav_buffer, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                audio_array = np.frombuffer(frames, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32767.0
                
                # Play audio
                sd.play(audio_float, 24000)
                sd.wait()
                
        except Exception as e:
            self.logger.error(f"⚠️  Audio playback error: {e}")
    
    async def process_conversation_turn(self):
        """Process one conversation turn with comprehensive error handling"""
        if not self.services_ready:
            self.logger.warning("⚠️  Services not ready")
            return True
        
        # Generate unique session ID for this conversation turn
        user_session = str(uuid.uuid4())
        
        # Use structured logging context
        async with voice_ai_request_context(
            "conversation_turn",
            "sovereign_assistant",
            self.logger,
            user_session=user_session
        ) as request_id:
            total_start = time.time()
            
            try:
                # Step 0: Check connectivity and determine mode
                connectivity_status = None
                if self.offline_system:
                    connectivity_status = await self.offline_system.check_and_switch_mode()
                    is_offline_mode = self.offline_system.is_offline_mode
                else:
                    is_offline_mode = False
                
                # Step 1: Record with real-time VAD using error handling
                record_start = time.time()
                try:
                    # Track audio capture latency
                    if self.metrics_collector:
                        with self.metrics_collector.track_latency(ComponentType.AUDIO_CAPTURE):
                            audio_data = await self._record_with_error_handling()
                    else:
                        audio_data = await self._record_with_error_handling()
                    
                    record_time = time.time() - record_start
                    
                    if not audio_data:
                        self.logger.warning("⚠️  No audio recorded")
                        if self.metrics_collector:
                            self.metrics_collector.record_request(ComponentType.AUDIO_CAPTURE.value, success=False)
                        return True
                    
                    # Record successful audio capture
                    if self.metrics_collector:
                        self.metrics_collector.record_request(ComponentType.AUDIO_CAPTURE.value, success=True)
                    
                    # Log audio processing metrics
                    audio_duration_ms = len(audio_data) / 32  # Approximate duration
                    log_audio_processing(self.logger, audio_duration_ms, record_time * 1000)
                    
                except AudioException as e:
                    self.logger.error(f"❌ Audio recording failed: {e.user_message}")
                    if self.metrics_collector:
                        self.metrics_collector.record_request(ComponentType.AUDIO_CAPTURE.value, success=False)
                    return True
                
                # Step 2: Try Hybrid Voice System first (if available and enabled)
                if self.use_realtime_api and self.hybrid_voice_system:
                    try:
                        self.logger.info("⚡ Processing with Hybrid Voice System (Realtime API)")
                        hybrid_start = time.time()
                        
                        # Process with hybrid system
                        result = await self.hybrid_voice_system.process_voice_input(audio_data=audio_data)
                        
                        if result and result.get('success'):
                            hybrid_time = time.time() - hybrid_start
                            total_time = time.time() - total_start
                            
                            self.logger.info(f"⚡ Hybrid Response: {total_time:.2f}s total")
                            self.logger.info(f"   🎤 Record: {record_time:.2f}s")
                            self.logger.info(f"   🔄 Hybrid: {hybrid_time:.2f}s")
                            
                            # Metrics
                            if self.metrics_collector:
                                self.metrics_collector.record_request("hybrid_voice", success=True)
                            
                            return True
                        else:
                            self.logger.warning("⚠️ Hybrid Voice System failed, falling back to traditional pipeline")
                            if self.metrics_collector:
                                self.metrics_collector.record_request("hybrid_voice", success=False)
                    
                    except Exception as e:
                        self.logger.warning(f"⚠️ Hybrid Voice System error: {e}, falling back to traditional pipeline")
                        if self.metrics_collector:
                            self.metrics_collector.record_request("hybrid_voice", success=False)
                
                # Step 3: Process with graceful degradation (traditional pipeline)
                if is_offline_mode and self.offline_system and self.offline_system.is_ready_for_offline():
                    self.logger.info("🔌 Processing request in offline mode")
                    
                    result = await self.offline_system.process_offline_request(audio_data)
                    if result:
                        user_text, audio_response = result
                        
                        # Play offline response
                        play_start = time.time()
                        if audio_response:
                            await self.play_audio_fast(audio_response)
                        play_time = time.time() - play_start
                        
                        # Performance metrics
                        total_time = time.time() - total_start
                        
                        self.logger.info(f"⚡ Offline Response: {total_time:.2f}s total")
                        self.logger.info(f"   🎤 Record: {record_time:.2f}s")
                        self.logger.info(f"   🔊 Play: {play_time:.2f}s")
                        
                        return True
                    else:
                        self.logger.error("❌ Offline processing failed, falling back to online if possible")
                        if connectivity_status == ConnectivityStatus.OFFLINE:
                            self.logger.error("No connectivity available for online fallback")
                            return True
                
                # Continue with online processing with graceful degradation
                connectivity_info = ""
                if connectivity_status:
                    connectivity_info = f" ({connectivity_status.value})"
                self.logger.info(f"🌐 Processing request in online mode{connectivity_info}")
                
                # Step 3: STT with fallback
                stt_start = time.time()
                
                # Track STT processing with metrics
                async def stt_with_metrics(tier_func):
                    if self.metrics_collector:
                        async with self.metrics_collector.track_async_latency(ComponentType.STT_PROCESSING):
                            result = await tier_func()
                        self.metrics_collector.record_request(ComponentType.STT_PROCESSING.value, success=bool(result))
                        return result
                    else:
                        return await tier_func()
                
                stt_functions = {
                    ServiceTier.PRIMARY: lambda: stt_with_metrics(lambda: self._transcribe_with_primary_stt(audio_data)),
                    ServiceTier.OFFLINE: lambda: stt_with_metrics(lambda: self._transcribe_with_offline_stt(audio_data))
                }
                
                stt_result = await self.degradation_manager.execute_with_degradation(
                    'stt',
                    stt_functions,
                    user_feedback_callback=self._provide_user_feedback
                )
                
                stt_time = time.time() - stt_start
                
                if not stt_result or not stt_result.text.strip():
                    self.logger.warning("⚠️  No speech detected")
                    if self.metrics_collector:
                        self.metrics_collector.record_accuracy('stt_detection', 0.0, confidence=0.0, threshold=0.5)
                    return True
                
                user_text = stt_result.text.strip()
                self.logger.info(f"🗣️  You: {user_text}")
                
                # Record STT accuracy metrics (confidence from STT service if available)
                if self.metrics_collector:
                    stt_confidence = getattr(stt_result, 'confidence', 0.8)  # Default confidence
                    self.metrics_collector.record_accuracy('stt_transcription', stt_confidence, confidence=stt_confidence, threshold=0.7)
                
                # Check for exit
                if user_text.lower() in ['goodbye', 'bye', 'exit']:
                    self.logger.info("👋 Goodbye!")
                    return False
                
                # Step 4: Enhance query with memory context
                enhanced_query = user_text
                if self.memory_manager:
                    try:
                        # Inject conversation and screen context
                        context = await self.memory_manager.inject_context(
                            current_query=user_text,
                            max_context_length=2000  # Reasonable limit for context
                        )
                        
                        if context:
                            # Create system-aware prompt that tells LLM about its capabilities
                            system_prompt = """SYSTEM: You are a voice assistant with SCREEN AWARENESS capabilities. You can see the user's screen content through OCR. When screen content is provided in the context below, you should reference and discuss what you can see on their screen. You have access to their current screen content and conversation history."""
                            
                            enhanced_query = f"{system_prompt}\n\n{context}\n\nUSER QUERY: {user_text}\n\nRESPOND: Based on the screen content and context above, provide a helpful response."
                            self.logger.info(f"🧠 Enhanced query with memory context ({len(context)} chars)")
                        else:
                            enhanced_query = user_text
                    except Exception as e:
                        self.logger.warning(f"⚠️ Memory context injection failed: {e}")
                        enhanced_query = user_text
                
                # Step 4: LLM processing with comprehensive fallback
                llm_start = time.time()
                
                # Check if this is a code-related request
                is_code_request = self.kimi_agent and self.kimi_agent.detect_code_request(user_text)
                
                if is_code_request:
                    self.logger.info(f"🔧 Code request detected, routing to Kimi K2 Agent")
                    
                    # Track code agent processing
                    if self.metrics_collector:
                        async with self.metrics_collector.track_async_latency('code_agent_processing'):
                            # Get current screen content for code context
                            screen_content = None
                            if self.screen_watcher:
                                recent_content = await self.screen_watcher.get_recent_content()
                                if recent_content:
                                    screen_content = recent_content[-1].get('text', '')
                            
                            # Process with Kimi agent
                            code_response = await self.kimi_agent.process_code_request(
                                message=user_text,
                                current_file=None,
                                error_context=screen_content
                            )
                        
                        # Record code agent success
                        self.metrics_collector.record_request('code_agent', success=bool(code_response.content))
                        
                        # Record code generation accuracy (based on response quality)
                        if code_response.content:
                            response_quality = min(len(code_response.content) / 100, 1.0)  # Simple quality metric
                            self.metrics_collector.record_accuracy('code_generation', response_quality * 100, confidence=0.9)
                    else:
                        # Get current screen content for code context
                        screen_content = None
                        if self.screen_watcher:
                            recent_content = await self.screen_watcher.get_recent_content()
                            if recent_content:
                                screen_content = recent_content[-1].get('text', '')
                        
                        # Process with Kimi agent
                        code_response = await self.kimi_agent.process_code_request(
                            message=user_text,
                            current_file=None,
                            error_context=screen_content
                        )
                    
                    ai_response = code_response.content
                    model_used = f"Kimi-K2 ({code_response.operation_type})"
                    
                    # Log code operation details
                    self.logger.info(f"🔧 Operation: {code_response.operation_type}")
                    if code_response.language:
                        self.logger.info(f"💻 Language: {code_response.language}")
                    if code_response.diff:
                        self.logger.info(f"📝 Diff generated: {len(code_response.diff)} chars")
                    
                else:
                    # Regular LLM routing with fallback
                    # Get memory context for the query
                    memory_start = time.time()
                    if self.metrics_collector:
                        async with self.metrics_collector.track_async_latency(ComponentType.MEMORY_RETRIEVAL):
                            memory_context = await self.memory_manager.inject_context(user_text)
                    else:
                        memory_context = await self.memory_manager.inject_context(user_text)
                    
                    memory_time = time.time() - memory_start
                    
                    # Record memory recall metrics
                    if self.metrics_collector and memory_context:
                        # Calculate BLEU score for memory relevance (simplified)
                        memory_relevance = self.metrics_collector.calculate_semantic_similarity(user_text, memory_context)
                        self.metrics_collector.record_accuracy('memory_recall', memory_relevance, confidence=0.8, threshold=self.metrics_collector.performance_targets.get('memory_recall_bleu', 85.0))
                        self.metrics_collector.record_request(ComponentType.MEMORY_RETRIEVAL.value, success=True)
                    elif self.metrics_collector:
                        self.metrics_collector.record_accuracy('memory_recall', 0.0, confidence=0.0)
                        self.metrics_collector.record_request(ComponentType.MEMORY_RETRIEVAL.value, success=False)
                    
                    # Add memory context to the query if available
                    if memory_context:
                        enhanced_query = f"{memory_context}\n\nCurrent Query: {user_text}"
                        self.logger.debug(f"💭 Enhanced query with memory context ({len(memory_context)} chars)")
                    else:
                        enhanced_query = user_text
                    
                    # Track LLM processing with metrics
                    async def llm_with_metrics(tier_func):
                        if self.metrics_collector:
                            async with self.metrics_collector.track_async_latency(ComponentType.LLM_INFERENCE):
                                result = await tier_func()
                            
                            # Record LLM success and response quality
                            if result and result.get('response'):
                                self.metrics_collector.record_request(ComponentType.LLM_INFERENCE.value, success=True)
                                # Simple response quality metric based on length and content
                                response_length = len(result.get('response', ''))
                                quality_score = min(response_length / 50, 1.0) * 100  # Normalize to 0-100
                                classification = result.get('classification')
                                confidence = classification.confidence if classification else 0.8
                                self.metrics_collector.record_accuracy('llm_response_quality', quality_score, confidence=confidence)
                            else:
                                self.metrics_collector.record_request(ComponentType.LLM_INFERENCE.value, success=False)
                            
                            return result
                        else:
                            return await tier_func()
                    
                    llm_functions = {
                        ServiceTier.PRIMARY: lambda: llm_with_metrics(lambda: self._process_with_primary_llm(enhanced_query)),
                        ServiceTier.FAST: lambda: llm_with_metrics(lambda: self._process_with_fast_llm(enhanced_query)),
                        ServiceTier.OFFLINE: lambda: llm_with_metrics(lambda: self._process_with_offline_llm(enhanced_query))
                    }
                    
                    llm_result = await self.degradation_manager.execute_with_degradation(
                        'llm',
                        llm_functions,
                        user_feedback_callback=self._provide_user_feedback
                    )
                    
                    ai_response = llm_result.get('response', 'I apologize, but I could not generate a response.')
                    model_used = llm_result.get('model_used', 'unknown')
                    
                    # Log routing information
                    classification = llm_result.get('classification')
                    if classification:
                        self.logger.info(f"🤖 LLM: {model_used} ({classification.complexity.value}, {classification.confidence:.2f})")
                    else:
                        self.logger.info(f"🤖 LLM: {model_used}")
                
                llm_time = time.time() - llm_start
                self.logger.info(f"🤖 Response: {ai_response}")
                
                # Step 5: TTS with fallback
                tts_start = time.time()
                
                # Track TTS processing with metrics
                async def tts_with_metrics(tier_func):
                    if self.metrics_collector:
                        async with self.metrics_collector.track_async_latency(ComponentType.TTS_GENERATION):
                            result = await tier_func()
                        self.metrics_collector.record_request(ComponentType.TTS_GENERATION.value, success=bool(result))
                        
                        # Record TTS quality metrics (based on successful generation)
                        if result:
                            # Simple quality metric based on audio data size
                            audio_quality = min(len(result) / 1000, 1.0) * 100 if hasattr(result, '__len__') else 90.0
                            self.metrics_collector.record_accuracy('tts_generation', audio_quality, confidence=0.9)
                        return result
                    else:
                        return await tier_func()
                
                tts_functions = {
                    ServiceTier.PRIMARY: lambda: tts_with_metrics(lambda: self._synthesize_with_primary_tts(ai_response)),
                    ServiceTier.OFFLINE: lambda: tts_with_metrics(lambda: self._synthesize_with_offline_tts(ai_response))
                }
                
                tts_result = await self.degradation_manager.execute_with_degradation(
                    'tts',
                    tts_functions,
                    user_feedback_callback=self._provide_user_feedback
                )
                
                tts_time = time.time() - tts_start
                
                if not tts_result:
                    self.logger.warning("⚠️  TTS generation failed")
                    if self.metrics_collector:
                        self.metrics_collector.record_request(ComponentType.TTS_GENERATION.value, success=False)
                    return True
                
                # Step 6: Play response
                play_start = time.time()
                if self.metrics_collector:
                    with self.metrics_collector.track_latency(ComponentType.AUDIO_PLAYBACK):
                        wav_audio = self.tts_service.get_wav_audio(tts_result)
                        if wav_audio:
                            await self.play_audio_fast(wav_audio)
                        else:
                            self.logger.warning("⚠️  Failed to convert audio to playable format")
                else:
                    wav_audio = self.tts_service.get_wav_audio(tts_result)
                    if wav_audio:
                        await self.play_audio_fast(wav_audio)
                    else:
                        self.logger.warning("⚠️  Failed to convert audio to playable format")
                
                play_time = time.time() - play_start
                
                # Record playback success
                if self.metrics_collector:
                    self.metrics_collector.record_request(ComponentType.AUDIO_PLAYBACK.value, success=bool(wav_audio))
                
                # Step 7: Store conversation in memory
                await self.memory_manager.store_conversation(
                    user_query=user_text,
                    assistant_response=ai_response,
                    metadata={
                        "model_used": model_used,
                        "response_time": total_time,
                        "llm_time": llm_time,
                        "memory_enhanced": bool(memory_context),
                        "request_id": request_id
                    }
                )
                
                # Step 6: Store conversation in memory for future context
                if self.memory_manager and ai_response:
                    try:
                        await self.memory_manager.store_conversation(
                            user_query=user_text,  # Store original user input
                            assistant_response=ai_response,
                            metadata={
                                "model_used": model_used,
                                "session_id": user_session,
                                "timestamp": time.time(),
                                "latency_ms": llm_time * 1000,
                                "classification": classification.complexity.value if classification else "unknown"
                            }
                        )
                        self.logger.debug("💾 Conversation stored in memory")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to store conversation: {e}")
                
                # Performance metrics
                total_time = time.time() - total_start
                
                # Record overall pipeline latency
                if self.metrics_collector:
                    self.metrics_collector.record_latency(ComponentType.OVERALL_PIPELINE.value, total_time * 1000)
                    self.metrics_collector.record_request(ComponentType.OVERALL_PIPELINE.value, success=True)
                    
                    # Check if we met the overall latency target
                    target = self.metrics_collector.performance_targets.get('voice_pipeline_latency', 800.0)
                    if total_time * 1000 <= target:
                        self.metrics_collector.record_accuracy('pipeline_latency_target', 100.0, confidence=1.0, threshold=100.0)
                    else:
                        overage_pct = ((total_time * 1000 - target) / target) * 100
                        score = max(0, 100 - overage_pct)
                        self.metrics_collector.record_accuracy('pipeline_latency_target', score, confidence=1.0, threshold=100.0)
                
                self.logger.log_performance_metrics(
                    'conversation_turn',
                    {
                        'total_time': total_time,
                        'record_time': record_time,
                        'stt_time': stt_time,
                        'memory_time': memory_time,
                        'llm_time': llm_time,
                        'tts_time': tts_time,
                        'play_time': play_time,
                        'audio_duration': audio_duration_ms
                    }
                )
                
                self.logger.info(f"⚡ Response: {total_time:.2f}s total")
                self.logger.info(f"   🎤 Record: {record_time:.2f}s")
                self.logger.info(f"   🗣️  STT: {stt_time:.2f}s")
                self.logger.info(f"   💭 Memory: {memory_time:.2f}s")
                self.logger.info(f"   🤖 LLM: {llm_time:.2f}s")
                self.logger.info(f"   🎵 TTS: {tts_time:.2f}s")
                self.logger.info(f"   🔊 Play: {play_time:.2f}s")
                
                return True
                
            except AllTiersFailedError as e:
                await self._handle_complete_failure(e)
                return True
            except VoiceAIException as e:
                self.logger.error(f"❌ Voice AI Error: {e.user_message}")
                return True
            except Exception as e:
                self.logger.error(f"❌ Unexpected error: {e}")
                return True
    
    async def _record_with_error_handling(self):
        """Audio recording with comprehensive error handling"""
        backoff_config = BackoffConfig(
            max_attempts=3,
            base_delay=0.5,
            jitter_type="full"
        )
        
        async def record_attempt():
            try:
                return await self.record_with_realtime_vad()
            except Exception as e:
                raise AudioException(
                    f"Audio recording failed: {str(e)}",
                    ErrorCategory.AUDIO,
                    retryable=True,
                    original_exception=e
                )
        
        return await retry_with_backoff(
            record_attempt,
            backoff_config,
            status_callback=self._provide_status_update
        )
    
    async def _transcribe_with_primary_stt(self, audio_data):
        """Transcribe with primary STT service"""
        return await self.stt_service.transcribe_audio(audio_data)
    
    async def _transcribe_with_offline_stt(self, audio_data):
        """Transcribe with offline STT service"""
        if self.offline_system:
            return await self.offline_system.stt_service.transcribe_audio(audio_data)
        else:
            raise STTException("Offline STT not available", ErrorCategory.PERMANENT, retryable=False)
    
    async def _process_with_primary_llm(self, query):
        """Process with primary LLM service"""
        return await self.llm_router.route_query(query)
    
    async def _process_with_fast_llm(self, query):
        """Process with fast LLM service"""
        # Route to fast model specifically
        return await self.llm_router.route_query(query, force_fast=True)
    
    async def _process_with_offline_llm(self, query):
        """Process with offline LLM service"""
        if self.offline_system:
            response = await self.offline_system.llm_service.generate_response(query)
            return {'response': response, 'model_used': 'gemma-local'}
        else:
            raise LLMException("Offline LLM not available", ErrorCategory.PERMANENT, retryable=False)
    
    async def _synthesize_with_primary_tts(self, text):
        """Synthesize with primary TTS service"""
        return await self.tts_service.synthesize_speech(text)
    
    async def _synthesize_with_offline_tts(self, text):
        """Synthesize with offline TTS service"""
        if self.offline_system:
            return await self.offline_system.tts_service.synthesize_speech(text)
        else:
            raise TTSException("Offline TTS not available", ErrorCategory.PERMANENT, retryable=False)
    
    async def _provide_user_feedback(self, message: str):
        """Provide user feedback during fallback operations"""
        self.logger.log_user_feedback(message, "degradation")
    
    async def _provide_status_update(self, message: str, attempt: int):
        """Provide status update during retry operations"""
        self.logger.info(f"🔄 {message} (attempt {attempt})")
    
    async def _handle_complete_failure(self, error: AllTiersFailedError):
        """Handle complete service failure gracefully"""
        self.logger.error(f"❌ Complete service failure: {error.user_message}")
        
        # Provide user feedback
        await self._provide_user_feedback(
            "All services are currently unavailable. Please try again in a moment."
        )
        
        # Could implement additional recovery strategies here
        # such as restarting services, clearing caches, etc.
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down...")
            self.shutdown_requested = True
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _start_dashboard(self) -> None:
        """Start the dashboard in a separate thread"""
        if self.dashboard:
            self.dashboard.start()
            self.logger.info("📊 Console dashboard started")
    
    def _start_web_dashboard(self) -> None:
        """Start the web dashboard server in a separate thread"""
        if not self.dashboard_server:
            return
            
        def run_dashboard_server():
            """Run dashboard server in background thread"""
            try:
                # Run the server (this blocks until stopped)
                asyncio.new_event_loop().run_until_complete(
                    self.dashboard_server.start()
                )
            except Exception as e:
                self.logger.error(f"Web dashboard server error: {e}")
        
        # Start dashboard server in background thread
        dashboard_thread = threading.Thread(
            target=run_dashboard_server,
            name="dashboard_server",
            daemon=True
        )
        dashboard_thread.start()
        self.logger.info("🌐 Web dashboard server started on http://localhost:8080")
    
    async def initialize(self) -> bool:
        """Initialize all assistant components (direct services, no pipeline)"""
        self.logger.info("Initializing Sovereign Voice Assistant...")
        
        try:
            # Load configuration
            if not self._load_config():
                return False
            
            # Initialize performance monitoring
            self.monitor = PerformanceMonitor()
            set_monitor(self.monitor)
            
            # Initialize console dashboard
            self.dashboard = create_dashboard(self.monitor)
            
            # Initialize services directly (no pipeline)
            try:
                self._initialize_services()
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize services: {e}")
                return False
            
            # Initialize web dashboard server with metrics collector integration (after services)
            if self.metrics_collector:
                self.dashboard_server = create_dashboard_server(
                    metrics_collector=self.metrics_collector,
                    host="localhost",
                    port=8080,
                    debug=self.debug
                )
                self.logger.info("🌐 Web dashboard server initialized")
            else:
                self.logger.warning("⚠️ Web dashboard server not initialized - metrics collector unavailable")
            
            # Initialize Memory System
            if not await self.memory_manager.initialize():
                self.logger.error("❌ Failed to initialize memory system")
                return False
            
            # Initialize Screen Watcher
            if not await self.screen_watcher.initialize():
                self.logger.error("❌ Failed to initialize screen watcher")
                return False
            
            # Initialize Kimi K2 Code Agent
            try:
                await self.kimi_agent.initialize()
                self.logger.info("🔧 Kimi K2 Code Agent initialized")
            except Exception as e:
                # Kimi agent is optional - continue without it if it fails
                self.logger.warning(f"⚠️ Kimi K2 Code Agent initialization failed: {e}")
                self.kimi_agent = None
            
            # Initialize Offline Fallback System
            try:
                if await self.offline_system.initialize():
                    self.logger.info("🔌 Offline fallback system initialized")
                else:
                    self.logger.warning("⚠️ Offline fallback system initialization failed")
            except Exception as e:
                # Offline system is optional - continue without it if it fails
                self.logger.warning(f"⚠️ Offline system initialization failed: {e}")
                self.offline_system = None
            
            # Initialize Hybrid Voice System (if enabled)
            if self.hybrid_voice_system:
                try:
                    if await self.hybrid_voice_system.initialize():
                        self.logger.info("⚡ Hybrid Voice System initialized - Realtime API ready")
                    else:
                        self.logger.warning("⚠️ Hybrid Voice System initialization failed - using traditional pipeline")
                        self.hybrid_voice_system = None
                        self.use_realtime_api = False
                except Exception as e:
                    self.logger.warning(f"⚠️ Hybrid Voice System initialization failed: {e} - using traditional pipeline")
                    self.hybrid_voice_system = None
                    self.use_realtime_api = False
            
            # Start health monitoring
            if self.health_monitor:
                await self.health_monitor.start(health_server_port=8080)
                self.logger.info("🏥 Health monitoring started on port 8080")
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            self.logger.info("Sovereign Voice Assistant initialized successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}")
            return False
    
    async def start(self) -> None:
        """Start the assistant"""
        self.logger.info("Starting Sovereign Voice Assistant...")
        self.running = True
        
        try:
            # Start console dashboard
            self._start_dashboard()
            
            # Start web dashboard server
            self._start_web_dashboard()
            
            # Start screen monitoring
            if self.screen_watcher.start_monitoring():
                self.logger.info("🖥️ Screen monitoring started")
            else:
                self.logger.warning("⚠️ Screen monitoring failed to start")
            
            self.logger.info("✅ Sovereign Voice Assistant started successfully!")
            self.logger.info("🎤 Voice assistant is now running...")
            self.logger.info("💬 Start speaking... I'll respond automatically!")
            self.logger.info("🖥️ Screen awareness active - I can see what's on your screen!")
            self.logger.info("📊 Performance dashboard is running")
            self.logger.info("⏹️  Press Ctrl+C to exit")
            
        except Exception as e:
            self.logger.error(f"Failed to start: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the assistant gracefully"""
        self.logger.info("Stopping Sovereign Voice Assistant...")
        self.running = False
        
        try:
            # Stop console dashboard
            if self.dashboard:
                self.dashboard.stop()
            
            # Stop web dashboard server
            if self.dashboard_server:
                await self.dashboard_server.stop()
                self.logger.info("🌐 Web dashboard server stopped")
            
            # Cleanup LLM router
            if self.llm_router:
                await self.llm_router.cleanup()
            
            # Cleanup Memory System
            if self.memory_manager:
                await self.memory_manager.cleanup()
            
            # Stop Screen Watcher
            if self.screen_watcher:
                self.screen_watcher.stop_monitoring()
            
            # Cleanup Kimi K2 Code Agent
            if self.kimi_agent:
                await self.kimi_agent.cleanup()
            
            # Cleanup Offline System
            if self.offline_system:
                await self.offline_system.cleanup()
            
            # Cleanup Hybrid Voice System
            if self.hybrid_voice_system:
                await self.hybrid_voice_system.cleanup()
                self.logger.info("⚡ Hybrid Voice System stopped")
            
            # Stop health monitoring
            if self.health_monitor:
                await self.health_monitor.stop()
                self.logger.info("🏥 Health monitoring stopped")
            
            # Stop metrics collection
            if self.metrics_collector:
                self.metrics_collector.stop()
                self.logger.info("📊 Metrics collection stopped")
            
            self.logger.info("Sovereign Voice Assistant stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def run(self) -> None:
        """Main event loop (like working demo)"""
        try:
            # Initialize
            if not await self.initialize():
                self.logger.error("Failed to initialize, exiting")
                return
            
            # Start
            await self.start()
            
            # Main conversation loop
            while self.running and not self.shutdown_requested:
                try:
                    should_continue = await self.process_conversation_turn()
                    if not should_continue:
                        break
                        
                except KeyboardInterrupt:
                    self.logger.info("Received keyboard interrupt")
                    break
                except Exception as e:
                    self.logger.error(f"❌ Error: {e}")
                    await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
        finally:
            await self.stop()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Sovereign 4.0 Voice Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config", 
        default="config/sovereign.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--log-file",
        help="Path to log file"
    )
    
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Enable OpenAI Realtime API for ultra-low latency responses"
    )
    
    return parser.parse_args()


async def main() -> None:
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Load environment variables
    load_dotenv()
    
    # Set realtime API flag if requested
    if args.realtime:
        os.environ['ENABLE_REALTIME_API'] = 'true'
    
    # Setup logging
    setup_logging(debug=args.debug, log_file=args.log_file)
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Create and run assistant
    assistant = SovereignAssistant(
        config_path=args.config,
        debug=args.debug
    )
    
    await assistant.run()


if __name__ == "__main__":
    asyncio.run(main()) 