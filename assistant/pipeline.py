"""
Voice Assistant Pipeline Module

This module provides the main VoiceAssistantPipeline class that integrates:
- AudioManager for audio input/output
- WhisperSTTService for speech-to-text
- OpenAITTSService for text-to-speech
- Push-to-talk functionality with keyboard controls
- Pipeline state management and async processing
"""

import asyncio
import logging
import os
import threading
import time
from enum import Enum
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
import keyboard
import yaml
from pathlib import Path

from .audio import AudioManager, AudioConfig, create_audio_manager
from .stt import WhisperSTTService, STTConfig, STTResult, create_whisper_stt_service
from .tts import OpenAITTSService, TTSConfig, TTSResult, create_openai_tts_service
from .monitoring import PerformanceMonitor, PipelineStage, get_monitor

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    """Voice Assistant Pipeline States"""
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    PLAYING = "playing"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class PipelineConfig:
    """Configuration for VoiceAssistantPipeline"""
    # Push-to-talk settings
    trigger_key: str = "space"
    trigger_key_name: str = "spacebar"
    
    # Audio settings
    recording_chunk_size: int = 1024
    max_recording_duration: float = 30.0  # seconds
    min_recording_duration: float = 0.5  # seconds
    
    # Processing settings
    process_timeout: float = 30.0  # seconds
    concurrent_processing: bool = True
    
    # Playback settings
    playback_volume: float = 1.0
    interrupt_on_new_recording: bool = True
    
    # Performance settings
    latency_target: float = 0.8  # seconds
    max_retries: int = 3
    
    # Logging and monitoring
    log_state_changes: bool = True
    collect_statistics: bool = True


@dataclass
class PipelineStatistics:
    """Pipeline performance statistics"""
    total_sessions: int = 0
    successful_sessions: int = 0
    failed_sessions: int = 0
    
    total_recording_time: float = 0.0
    total_processing_time: float = 0.0
    total_playback_time: float = 0.0
    
    average_latency: float = 0.0
    min_latency: float = float('inf')
    max_latency: float = 0.0
    
    stt_success_rate: float = 0.0
    tts_success_rate: float = 0.0
    
    def update_session(self, success: bool, recording_time: float, 
                      processing_time: float, playback_time: float):
        """Update statistics for a completed session"""
        self.total_sessions += 1
        if success:
            self.successful_sessions += 1
        else:
            self.failed_sessions += 1
            
        self.total_recording_time += recording_time
        self.total_processing_time += processing_time
        self.total_playback_time += playback_time
        
        # Update latency statistics
        session_latency = processing_time + playback_time
        if session_latency > 0:
            self.min_latency = min(self.min_latency, session_latency)
            self.max_latency = max(self.max_latency, session_latency)
            
            # Calculate rolling average
            if self.successful_sessions > 0:
                self.average_latency = ((self.average_latency * (self.successful_sessions - 1)) 
                                     + session_latency) / self.successful_sessions


class VoiceAssistantPipeline:
    """Main Voice Assistant Pipeline coordinating all services"""
    
    def __init__(self, config: PipelineConfig, 
                 audio_manager: AudioManager,
                 stt_service: WhisperSTTService,
                 tts_service: OpenAITTSService,
                 response_callback: Optional[Callable[[str], str]] = None,
                 performance_monitor: Optional[PerformanceMonitor] = None):
        """
        Initialize the Voice Assistant Pipeline
        
        Args:
            config: Pipeline configuration
            audio_manager: Audio input/output manager
            stt_service: Speech-to-text service
            tts_service: Text-to-speech service
            response_callback: Optional callback function for processing transcribed text
            performance_monitor: Optional performance monitor (uses global if not provided)
        """
        self.config = config
        self.audio_manager = audio_manager
        self.stt_service = stt_service
        self.tts_service = tts_service
        self.response_callback = response_callback or self._default_response_callback
        self.monitor = performance_monitor or get_monitor()
        
        # Pipeline state
        self.state = PipelineState.IDLE
        self.is_running = False
        self.is_push_to_talk_active = False
        
        # Threading and async
        self._state_lock = threading.Lock()
        self._keyboard_thread = None
        self._pipeline_task = None
        self._event_loop = None
        
        # Audio data buffer
        self._audio_buffer = []
        self._recording_start_time = None
        self._session_start_time = None
        
        # Statistics
        self.statistics = PipelineStatistics()
        
        # Event callbacks
        self.on_state_changed: Optional[Callable[[PipelineState, PipelineState], None]] = None
        self.on_transcription_received: Optional[Callable[[str], None]] = None
        self.on_response_generated: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        
    def _default_response_callback(self, text: str) -> str:
        """Default response callback that echoes the input"""
        return f"I heard you say: {text}"
    
    async def initialize(self) -> bool:
        """Initialize the pipeline and all services"""
        try:
            logger.info("Initializing Voice Assistant Pipeline...")
            
            # Initialize audio manager
            self.audio_manager.initialize()
            
            # Initialize STT service
            if not self.stt_service.initialize():
                logger.error("Failed to initialize STT service")
                return False
            
            # Initialize TTS service
            if not self.tts_service.initialize():
                logger.error("Failed to initialize TTS service")
                return False
            
            # Skip audio stream setup during initialization - streams will be setup on-demand
            logger.info("Voice Assistant Pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            return False
    
    def start(self) -> bool:
        """Start the pipeline and begin listening for push-to-talk"""
        try:
            if self.is_running:
                logger.warning("Pipeline is already running")
                return True
            
            logger.info("Starting Voice Assistant Pipeline...")
            
            # Start keyboard listener thread only if trigger_key is configured
            if self.config.trigger_key is not None:
                self._keyboard_thread = threading.Thread(
                    target=self._keyboard_listener_worker,
                    daemon=True
                )
                self._keyboard_thread.start()
                logger.info(f"Keyboard listener started for {self.config.trigger_key_name}")
            else:
                logger.info("Keyboard integration disabled - no trigger key configured")
            
            # Start pipeline event loop
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
            
            self._pipeline_task = self._event_loop.create_task(self._pipeline_worker())
            
            self.is_running = True
            self._set_state(PipelineState.IDLE)
            
            logger.info("Voice Assistant Pipeline started successfully")
            if self.config.trigger_key is not None:
                logger.info(f"Press and hold {self.config.trigger_key_name} to start recording")
            else:
                logger.info("Voice pipeline running without keyboard controls")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start pipeline: {e}")
            return False
    
    def stop(self):
        """Stop the pipeline and cleanup resources"""
        try:
            logger.info("Stopping Voice Assistant Pipeline...")
            
            self.is_running = False
            self._set_state(PipelineState.SHUTDOWN)
            
            # Stop keyboard listener
            if self._keyboard_thread and self._keyboard_thread.is_alive():
                # Note: keyboard.unhook_all() will be called in the worker
                self._keyboard_thread.join(timeout=2.0)
            
            # Stop pipeline task
            if self._pipeline_task and not self._pipeline_task.done():
                self._pipeline_task.cancel()
                
            # Stop event loop
            if self._event_loop and self._event_loop.is_running():
                self._event_loop.stop()
            
            # Cleanup audio resources
            self._cleanup_audio()
            
            logger.info("Voice Assistant Pipeline stopped")
            
        except Exception as e:
            logger.error(f"Error stopping pipeline: {e}")
    
    def run(self):
        """Run the pipeline (blocking call)"""
        try:
            if not self.start():
                return
                
            # Run the event loop
            self._event_loop.run_until_complete(self._pipeline_task)
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, stopping pipeline...")
        except Exception as e:
            logger.error(f"Error running pipeline: {e}")
        finally:
            self.stop()
    
    def _keyboard_listener_worker(self):
        """Worker thread for keyboard listener"""
        try:
            logger.debug("Keyboard listener thread started")
            
            # Safety check - don't set up keyboard if trigger_key is None
            if self.config.trigger_key is None:
                logger.warning("Keyboard listener called but trigger_key is None")
                return
            
            def on_press(event):
                if event.name == self.config.trigger_key:
                    self._handle_push_to_talk_press()
            
            def on_release(event):
                if event.name == self.config.trigger_key:
                    self._handle_push_to_talk_release()
            
            # Set up keyboard hooks
            keyboard.on_press(on_press)
            keyboard.on_release(on_release)
            
            # Keep thread alive while pipeline is running
            while self.is_running:
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in keyboard listener: {e}")
        finally:
            keyboard.unhook_all()
            logger.debug("Keyboard listener thread stopped")
    
    def _handle_push_to_talk_press(self):
        """Handle push-to-talk key press"""
        with self._state_lock:
            if self.state == PipelineState.IDLE:
                self.is_push_to_talk_active = True
                self._set_state(PipelineState.RECORDING)
                logger.info("Push-to-talk activated - recording started")
            elif self.state == PipelineState.PLAYING and self.config.interrupt_on_new_recording:
                # Interrupt current playback
                self.is_push_to_talk_active = True
                self._set_state(PipelineState.RECORDING)
                logger.info("Push-to-talk activated - interrupting playback")
    
    def _handle_push_to_talk_release(self):
        """Handle push-to-talk key release"""
        with self._state_lock:
            if self.state == PipelineState.RECORDING:
                self.is_push_to_talk_active = False
                self._set_state(PipelineState.PROCESSING)
                logger.info("Push-to-talk released - processing audio")
    
    async def _pipeline_worker(self):
        """Main pipeline worker coroutine"""
        try:
            logger.debug("Pipeline worker started")
            
            while self.is_running:
                try:
                    if self.state == PipelineState.RECORDING:
                        await self._handle_recording()
                    elif self.state == PipelineState.PROCESSING:
                        await self._handle_processing()
                    elif self.state == PipelineState.PLAYING:
                        await self._handle_playing()
                    elif self.state == PipelineState.IDLE:
                        # Auto-start recording if keyboard integration is disabled
                        if self.config.trigger_key is None:
                            logger.info("🎤 Auto-starting recording (keyboard disabled)")
                            self.is_push_to_talk_active = True
                            self._set_state(PipelineState.RECORDING)
                        else:
                            await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                    else:
                        await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                        
                except Exception as e:
                    logger.error(f"Error in pipeline worker: {e}")
                    if self.on_error:
                        self.on_error(e)
                    self._set_state(PipelineState.ERROR)
                    await asyncio.sleep(1.0)  # Wait before retrying
                    if self.state == PipelineState.ERROR:
                        self._set_state(PipelineState.IDLE)
                        
        except asyncio.CancelledError:
            logger.debug("Pipeline worker cancelled")
        except Exception as e:
            logger.error(f"Fatal error in pipeline worker: {e}")
        finally:
            logger.debug("Pipeline worker stopped")
    
    async def _handle_recording(self):
        """Handle recording state"""
        # Use monitoring context for audio capture
        with self.monitor.timing_context(PipelineStage.AUDIO_CAPTURE, 
                                       {'push_to_talk': self.is_push_to_talk_active}) as timing:
            
            if not self.audio_manager.is_recording:
                # Start recording
                self.audio_manager.start_recording()
                self._audio_buffer = []
                self._recording_start_time = time.time()
                self._session_start_time = time.time()
                
            # Voice activity detection for auto-recording mode
            silence_threshold = 0.001  # Fixed VAD threshold that works
            silence_duration = 0.8     # Stop after 0.8s of silence
            silence_samples = int(silence_duration * 16000)  # Assuming 16kHz sample rate
            silence_buffer = []
            
            # Read audio chunks while recording
            while self.is_push_to_talk_active and self.is_running:
                chunk = self.audio_manager.read_audio_chunk()
                if chunk:
                    self._audio_buffer.append(chunk)
                    
                    # Voice activity detection for auto-recording mode
                    if self.config.trigger_key is None:  # Auto-recording mode
                        # Calculate RMS for voice activity detection
                        import numpy as np
                        audio_data = np.frombuffer(chunk, dtype=np.int16)
                        rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2) / (32767 ** 2))
                        
                        if rms < silence_threshold:
                            silence_buffer.append(chunk)
                            if len(silence_buffer) * len(chunk) > silence_samples:
                                # Extended silence detected in auto-mode, stop recording
                                logger.info(f"🔇 Silence detected, stopping auto-recording ({time.time() - self._recording_start_time:.1f}s)")
                                self.is_push_to_talk_active = False
                                break
                        else:
                            # Voice detected, clear silence buffer
                            silence_buffer = []
                    
                    # Check for maximum recording duration
                    if (time.time() - self._recording_start_time) > self.config.max_recording_duration:
                        logger.warning(f"Recording exceeded maximum duration ({self.config.max_recording_duration}s)")
                        timing.mark_failure("Max recording duration exceeded")
                        break
                        
                await asyncio.sleep(0.01)  # Small delay to yield control
            
            # Stop recording
            if self.audio_manager.is_recording:
                self.audio_manager.stop_recording()
                
            # Add metadata about recording
            recording_duration = time.time() - self._recording_start_time if self._recording_start_time else 0
            audio_chunks = len(self._audio_buffer)
            timing.metadata.update({
                'recording_duration': recording_duration,
                'audio_chunks': audio_chunks,
                'buffer_size': sum(len(chunk) for chunk in self._audio_buffer)
            })
    
    async def _handle_processing(self):
        """Handle processing state"""
        try:
            if not self._audio_buffer:
                logger.warning("No audio data to process")
                self._set_state(PipelineState.IDLE)
                return
            
            # Calculate recording duration
            recording_duration = time.time() - self._recording_start_time
            
            # Check minimum recording duration
            if recording_duration < self.config.min_recording_duration:
                logger.info(f"Recording too short ({recording_duration:.2f}s), ignoring")
                self._set_state(PipelineState.IDLE)
                return
            
            # Combine audio chunks
            audio_data = b''.join(self._audio_buffer)
            
            # Process STT with monitoring
            processing_start_time = time.time()
            
            with self.monitor.timing_context(PipelineStage.STT_PROCESSING, 
                                           {'audio_size': len(audio_data), 
                                            'recording_duration': recording_duration}) as stt_timing:
                stt_result = await self.stt_service.transcribe_audio(audio_data)
                
                if not stt_result or not stt_result.text.strip():
                    stt_timing.mark_failure("No text transcribed")
                    logger.warning("No text transcribed from audio")
                    self._set_state(PipelineState.IDLE)
                    return
                
                # Add metadata about transcription
                stt_timing.metadata.update({
                    'text_length': len(stt_result.text),
                    'confidence': stt_result.confidence,
                    'language': stt_result.language
                })
            
            logger.info(f"Transcribed: {stt_result.text}")
            
            # Callback for transcription
            if self.on_transcription_received:
                self.on_transcription_received(stt_result.text)
            
            # Generate response (not monitored as it's user-defined)
            response_text = self.response_callback(stt_result.text)
            
            if not response_text:
                logger.warning("No response generated")
                self._set_state(PipelineState.IDLE)
                return
            
            logger.info(f"Response: {response_text}")
            
            # Callback for response generation
            if self.on_response_generated:
                self.on_response_generated(response_text)
            
            # Process TTS with monitoring
            with self.monitor.timing_context(PipelineStage.TTS_GENERATION, 
                                           {'text_length': len(response_text),
                                            'response_text': response_text[:100]}) as tts_timing:
                tts_result = await self.tts_service.synthesize_speech(response_text)
                
                if not tts_result:
                    tts_timing.mark_failure("TTS synthesis failed")
                    logger.error("Failed to generate speech")
                    self._set_state(PipelineState.IDLE)
                    return
                
                # Add metadata about TTS
                tts_timing.metadata.update({
                    'audio_duration': tts_result.duration,
                    'voice': tts_result.voice,
                    'speed': tts_result.speed,
                    'cached': tts_result.cached
                })
            
            # Convert to WAV for playback
            wav_audio = self.tts_service.get_wav_audio(tts_result)
            
            if not wav_audio:
                logger.error("Failed to convert audio to WAV")
                self._set_state(PipelineState.IDLE)
                return
            
            # Store audio for playback
            self._playback_audio = wav_audio
            self._processing_time = time.time() - processing_start_time
            
            # Move to playback state
            self._set_state(PipelineState.PLAYING)
            
        except Exception as e:
            logger.error(f"Error in processing: {e}")
            self._set_state(PipelineState.ERROR)
    
    async def _handle_playing(self):
        """Handle playing state"""
        try:
            if not hasattr(self, '_playback_audio'):
                logger.error("No audio to play")
                self._set_state(PipelineState.IDLE)
                return
            
            # Monitor audio playback
            with self.monitor.timing_context(PipelineStage.AUDIO_PLAYBACK, 
                                           {'audio_size': len(self._playback_audio)}) as playback_timing:
                
                # Play audio
                success = self.audio_manager.play_audio_chunk(self._playback_audio)
                
                if not success:
                    playback_timing.mark_failure("Audio playback failed")
                    logger.error("Failed to play audio")
                    self._set_state(PipelineState.IDLE)
                    return
                
                # Wait for playback to complete
                # Note: This is a simplified implementation
                # In a real system, you'd want to track playback progress
                await asyncio.sleep(2.0)  # Estimated playback duration
                
                playback_timing.metadata.update({
                    'estimated_duration': 2.0,
                    'audio_format': 'WAV'
                })
            
            # Record total round-trip time
            if self._session_start_time:
                total_session_time = time.time() - self._session_start_time
                
                self.monitor.record_measurement(
                    PipelineStage.TOTAL_ROUND_TRIP, 
                    total_session_time, 
                    True,
                    {
                        'processing_time': self._processing_time,
                        'session_complete': True
                    }
                )
            
            # Update statistics
            playback_time = time.time() - (time.time() - 2.0)  # Simplified calculation
            total_time = time.time() - self._session_start_time
            
            if self.config.collect_statistics:
                self.statistics.update_session(
                    success=True,
                    recording_time=total_time - self._processing_time - playback_time,
                    processing_time=self._processing_time,
                    playback_time=playback_time
                )
            
            # Cleanup
            delattr(self, '_playback_audio')
            self._audio_buffer = []
            
            # Return to idle state to continue the cycle (especially for auto-recording mode)
            self._set_state(PipelineState.IDLE)
            
        except Exception as e:
            logger.error(f"Error in playback: {e}")
            self._set_state(PipelineState.ERROR)
    
    def _set_state(self, new_state: PipelineState):
        """Set pipeline state and trigger callbacks"""
        with self._state_lock:
            old_state = self.state
            self.state = new_state
            
            if self.config.log_state_changes:
                logger.info(f"Pipeline state: {old_state.value} -> {new_state.value}")
            
            if self.on_state_changed:
                self.on_state_changed(old_state, new_state)
    
    def _cleanup_audio(self):
        """Clean up audio resources"""
        try:
            if self.audio_manager.is_recording:
                self.audio_manager.stop_recording()
            
            self.audio_manager.cleanup()
            
        except Exception as e:
            logger.error(f"Error cleaning up audio: {e}")
    
    def get_state(self) -> PipelineState:
        """Get current pipeline state"""
        with self._state_lock:
            return self.state
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        stats = {
            'pipeline': {
                'state': self.state.value,
                'is_running': self.is_running,
                'is_push_to_talk_active': self.is_push_to_talk_active,
                'total_sessions': self.statistics.total_sessions,
                'successful_sessions': self.statistics.successful_sessions,
                'failed_sessions': self.statistics.failed_sessions,
                'average_latency': self.statistics.average_latency,
                'min_latency': self.statistics.min_latency,
                'max_latency': self.statistics.max_latency,
            },
            'audio': self.audio_manager.get_statistics() if hasattr(self.audio_manager, 'get_statistics') else {},
            'stt': self.stt_service.get_statistics(),
            'tts': self.tts_service.get_statistics(),
            'monitoring': self.monitor.get_summary(),
            'stage_metrics': self.monitor.get_metrics(),
            'recent_alerts': self.monitor.get_recent_alerts(limit=10),
        }
        return stats
    
    def reset_statistics(self):
        """Reset all statistics"""
        self.statistics = PipelineStatistics()
        if hasattr(self.audio_manager, 'reset_statistics'):
            self.audio_manager.reset_statistics()
        self.stt_service.reset_statistics()
        self.tts_service.reset_statistics()


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return {}


def create_pipeline_from_config(config_path: str = "config/config.yaml",
                              response_callback: Optional[Callable[[str], str]] = None) -> Optional[VoiceAssistantPipeline]:
    """Create a VoiceAssistantPipeline from configuration file"""
    try:
        # Load configuration
        config = load_config_from_yaml(config_path)
        
        if not config:
            logger.error("Failed to load configuration")
            return None
        
        # Create pipeline configuration
        pipeline_config = PipelineConfig()
        
        # Create audio manager
        audio_config = AudioConfig(
            sample_rate=config.get('audio', {}).get('sample_rate', 16000),
            channels=1,
            chunk_size=config.get('audio', {}).get('chunk_size', 1024)
        )
        audio_manager = create_audio_manager(audio_config)
        
        # Create STT service
        stt_config = STTConfig(
            model=config.get('stt', {}).get('primary', {}).get('model', 'whisper-1'),
            language=config.get('stt', {}).get('primary', {}).get('language', 'en')
        )
        stt_service = create_whisper_stt_service(
            api_key=os.environ.get('OPENAI_API_KEY'),
            config=stt_config
        )
        
        # Create TTS service
        tts_config = TTSConfig(
            model=config.get('tts', {}).get('primary', {}).get('model', 'tts-1'),
            voice=config.get('tts', {}).get('primary', {}).get('voice', 'alloy'),
            speed=config.get('tts', {}).get('primary', {}).get('speed', 1.0)
        )
        tts_service = create_openai_tts_service(
            api_key=os.environ.get('OPENAI_API_KEY'),
            config=tts_config
        )
        
        # Create pipeline
        pipeline = VoiceAssistantPipeline(
            config=pipeline_config,
            audio_manager=audio_manager,
            stt_service=stt_service,
            tts_service=tts_service,
            response_callback=response_callback
        )
        
        return pipeline
        
    except Exception as e:
        logger.error(f"Failed to create pipeline from config: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    import os
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create pipeline
    pipeline = create_pipeline_from_config()
    
    if pipeline:
        # Initialize and run
        async def main():
            if await pipeline.initialize():
                pipeline.run()
        
        asyncio.run(main())
    else:
        logger.error("Failed to create pipeline") 