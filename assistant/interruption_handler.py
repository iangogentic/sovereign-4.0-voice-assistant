"""
Natural Conversation Interruption Handler for Sovereign 4.0
Implements intelligent turn-taking and interruption management using OpenAI's server-side VAD
Enhanced with client-side amplitude analysis and conversation state management
"""

import asyncio
import json
import time
import logging
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, Callable, List, Tuple
from datetime import datetime, timedelta
import threading
from queue import Queue, Empty
import statistics


class ConversationState(Enum):
    """Conversation flow states for natural turn-taking"""
    LISTENING = "listening"        # Actively listening for user input
    SPEAKING = "speaking"          # AI is speaking/responding
    INTERRUPTED = "interrupted"    # User interrupted AI response
    RESUMING = "resuming"         # Recovering from interruption


class InterruptionType(Enum):
    """Types of interruptions detected"""
    USER_SPEECH = "user_speech"           # User started speaking
    SILENCE_TIMEOUT = "silence_timeout"   # Extended silence detected
    AMPLITUDE_SPIKE = "amplitude_spike"   # Client-side audio amplitude detection
    MANUAL = "manual"                     # Manual interruption trigger


@dataclass
class VADConfig:
    """OpenAI Server-side VAD configuration"""
    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 200
    turn_detection_type: str = "server_vad"


@dataclass
class ClientAudioConfig:
    """Client-side audio amplitude analysis configuration"""
    sample_rate: int = 24000
    chunk_size: int = 1024
    amplitude_threshold: float = 0.01
    silence_threshold: float = 0.005
    analysis_window_ms: int = 100
    noise_gate_threshold: float = 0.002


@dataclass
class InterruptionEvent:
    """Represents a single interruption event"""
    timestamp: datetime = field(default_factory=datetime.now)
    interruption_type: InterruptionType = InterruptionType.USER_SPEECH
    conversation_state: ConversationState = ConversationState.SPEAKING
    audio_amplitude: Optional[float] = None
    recovery_time: Optional[float] = None
    context_preserved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InterruptionMetrics:
    """Analytics for conversation interruption patterns"""
    session_start: datetime = field(default_factory=datetime.now)
    total_interruptions: int = 0
    interruptions_by_type: Dict[InterruptionType, int] = field(default_factory=dict)
    average_recovery_time: float = 0.0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    conversation_quality_score: float = 0.0
    turn_taking_efficiency: float = 0.0
    interruption_history: List[InterruptionEvent] = field(default_factory=list)


class ConversationBuffer:
    """Manages conversation state and partial response recovery"""
    
    def __init__(self, max_buffer_size: int = 1000):
        self.max_buffer_size = max_buffer_size
        self.partial_response: str = ""
        self.response_buffer: List[str] = []
        self.interruption_point: Optional[int] = None
        self.context_snapshot: Dict[str, Any] = {}
        self.audio_buffer: List[bytes] = []
        self._lock = threading.Lock()
    
    def save_partial_response(self, content: str, interruption_point: Optional[int] = None):
        """Save partial AI response for recovery"""
        with self._lock:
            self.partial_response = content
            self.interruption_point = interruption_point or len(content)
            self.response_buffer.append(content)
            
            # Maintain buffer size
            if len(self.response_buffer) > self.max_buffer_size:
                self.response_buffer.pop(0)
    
    def save_context_snapshot(self, context: Dict[str, Any]):
        """Save conversation context at interruption point"""
        with self._lock:
            self.context_snapshot = context.copy()
    
    def clear_buffers(self):
        """Clear all buffers for new conversation turn"""
        with self._lock:
            self.partial_response = ""
            self.response_buffer.clear()
            self.interruption_point = None
            self.context_snapshot.clear()
            self.audio_buffer.clear()
    
    def get_recovery_data(self) -> Dict[str, Any]:
        """Get data needed for interruption recovery"""
        with self._lock:
            return {
                "partial_response": self.partial_response,
                "interruption_point": self.interruption_point,
                "context_snapshot": self.context_snapshot.copy(),
                "response_history": self.response_buffer.copy()[-5:]  # Last 5 responses
            }


class AudioAmplitudeAnalyzer:
    """Real-time audio amplitude analysis for client-side interruption detection"""
    
    def __init__(self, config: ClientAudioConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.audio_queue = Queue()
        self.is_analyzing = False
        self.current_amplitude = 0.0
        self.amplitude_history: List[float] = []
        self.analysis_thread: Optional[threading.Thread] = None
        self.callbacks: List[Callable] = []
        
        # Analysis parameters
        self.samples_per_window = int(
            self.config.sample_rate * self.config.analysis_window_ms / 1000
        )
    
    def add_audio_callback(self, callback: Callable[[float, bool], None]):
        """Add callback for amplitude analysis results
        
        Args:
            callback: Function called with (amplitude, is_speech) parameters
        """
        self.callbacks.append(callback)
    
    def start_analysis(self):
        """Start real-time audio amplitude analysis"""
        if self.is_analyzing:
            return
        
        self.is_analyzing = True
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()
        self.logger.info("🔊 Audio amplitude analysis started")
    
    def stop_analysis(self):
        """Stop audio amplitude analysis"""
        self.is_analyzing = False
        if self.analysis_thread:
            self.analysis_thread.join(timeout=1.0)
        self.logger.info("🔊 Audio amplitude analysis stopped")
    
    def process_audio_chunk(self, audio_data: bytes):
        """Add audio chunk for analysis"""
        try:
            self.audio_queue.put(audio_data, block=False)
        except:
            pass  # Skip if queue is full
    
    def _analysis_loop(self):
        """Main analysis loop running in separate thread"""
        while self.is_analyzing:
            try:
                # Get audio data with timeout
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # Convert bytes to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                if len(audio_array) == 0:
                    continue
                
                # Normalize to [-1, 1] range
                normalized_audio = audio_array.astype(np.float32) / 32768.0
                
                # Calculate RMS amplitude
                amplitude = np.sqrt(np.mean(normalized_audio ** 2))
                self.current_amplitude = amplitude
                
                # Update amplitude history
                self.amplitude_history.append(amplitude)
                if len(self.amplitude_history) > 100:  # Keep last 100 samples
                    self.amplitude_history.pop(0)
                
                # Determine if this is speech
                is_speech = self._is_speech_detected(amplitude)
                
                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        callback(amplitude, is_speech)
                    except Exception as e:
                        self.logger.error(f"❌ Amplitude callback error: {e}")
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"❌ Audio analysis error: {e}")
    
    def _is_speech_detected(self, amplitude: float) -> bool:
        """Determine if current amplitude indicates speech"""
        # Basic threshold check
        if amplitude < self.config.silence_threshold:
            return False
        
        if amplitude > self.config.amplitude_threshold:
            # Additional noise filtering using recent history
            if len(self.amplitude_history) >= 5:
                recent_avg = statistics.mean(self.amplitude_history[-5:])
                return amplitude > recent_avg * 1.5 and amplitude > self.config.noise_gate_threshold
            return True
        
        return False


class InterruptionHandler:
    """
    Comprehensive interruption handling system for natural conversation flow
    Integrates OpenAI server-side VAD with client-side amplitude analysis
    """
    
    def __init__(
        self,
        vad_config: Optional[VADConfig] = None,
        audio_config: Optional[ClientAudioConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.vad_config = vad_config or VADConfig()
        self.audio_config = audio_config or ClientAudioConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # State management
        self.current_state = ConversationState.LISTENING
        self.previous_state = ConversationState.LISTENING
        self.state_change_time = datetime.now()
        self.state_lock = threading.Lock()
        
        # Components
        self.conversation_buffer = ConversationBuffer()
        self.amplitude_analyzer = AudioAmplitudeAnalyzer(self.audio_config, self.logger)
        self.metrics = InterruptionMetrics()
        
        # Callbacks
        self.state_change_callbacks: List[Callable] = []
        self.interruption_callbacks: List[Callable] = []
        
        # VAD event tracking
        self.last_speech_start: Optional[datetime] = None
        self.last_speech_stop: Optional[datetime] = None
        self.is_vad_active = False
        
        # Setup amplitude analyzer callbacks
        self.amplitude_analyzer.add_audio_callback(self._on_amplitude_analysis)
        
        self.logger.info("✅ InterruptionHandler initialized")
    
    async def initialize(self) -> bool:
        """Initialize the interruption handler"""
        try:
            # Start amplitude analysis
            self.amplitude_analyzer.start_analysis()
            
            # Initialize state
            await self._transition_to_state(ConversationState.LISTENING)
            
            self.logger.info("✅ InterruptionHandler initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize InterruptionHandler: {e}")
            return False
    
    def add_state_change_callback(self, callback: Callable[[ConversationState, ConversationState], None]):
        """Add callback for conversation state changes"""
        self.state_change_callbacks.append(callback)
    
    def add_interruption_callback(self, callback: Callable[[InterruptionEvent], None]):
        """Add callback for interruption events"""
        self.interruption_callbacks.append(callback)
    
    def get_vad_configuration(self) -> Dict[str, Any]:
        """Get VAD configuration for OpenAI Realtime API session setup"""
        return {
            "turn_detection": {
                "type": self.vad_config.turn_detection_type,
                "threshold": self.vad_config.threshold,
                "prefix_padding_ms": self.vad_config.prefix_padding_ms,
                "silence_duration_ms": self.vad_config.silence_duration_ms
            }
        }
    
    async def handle_vad_event(self, event_type: str, event_data: Dict[str, Any]):
        """Handle VAD events from OpenAI Realtime API"""
        try:
            if event_type == "input_audio_buffer.speech_started":
                await self._on_speech_started(event_data)
            elif event_type == "input_audio_buffer.speech_stopped":
                await self._on_speech_stopped(event_data)
            elif event_type == "conversation.item.input_audio_transcription.completed":
                await self._on_transcription_completed(event_data)
                
        except Exception as e:
            self.logger.error(f"❌ Error handling VAD event {event_type}: {e}")
    
    async def _on_speech_started(self, event_data: Dict[str, Any]):
        """Handle speech start detection from server VAD"""
        self.last_speech_start = datetime.now()
        self.is_vad_active = True
        
        self.logger.debug("🎤 Server VAD: Speech started")
        
        # Check if we need to interrupt current AI response
        if self.current_state == ConversationState.SPEAKING:
            await self._trigger_interruption(
                InterruptionType.USER_SPEECH,
                metadata={"vad_event": "speech_started", **event_data}
            )
    
    async def _on_speech_stopped(self, event_data: Dict[str, Any]):
        """Handle speech stop detection from server VAD"""
        self.last_speech_stop = datetime.now()
        self.is_vad_active = False
        
        self.logger.debug("🎤 Server VAD: Speech stopped")
        
        # Calculate speech duration if we have start time
        if self.last_speech_start:
            duration = (self.last_speech_stop - self.last_speech_start).total_seconds()
            self.logger.debug(f"🎤 Speech duration: {duration:.2f}s")
    
    async def _on_transcription_completed(self, event_data: Dict[str, Any]):
        """Handle completed transcription from VAD"""
        transcript = event_data.get("transcript", "")
        if transcript:
            self.logger.debug(f"🎤 Transcription: {transcript}")
    
    def _on_amplitude_analysis(self, amplitude: float, is_speech: bool):
        """Handle client-side amplitude analysis results"""
        # Check for amplitude-based interruption during AI speaking
        if (self.current_state == ConversationState.SPEAKING and 
            is_speech and 
            amplitude > self.audio_config.amplitude_threshold * 2):  # Higher threshold for interruption
            
            # Trigger interruption asynchronously
            asyncio.create_task(self._trigger_interruption(
                InterruptionType.AMPLITUDE_SPIKE,
                metadata={"amplitude": amplitude, "threshold": self.audio_config.amplitude_threshold}
            ))
    
    async def _trigger_interruption(self, interruption_type: InterruptionType, metadata: Dict[str, Any] = None):
        """Trigger an interruption event"""
        try:
            # Create interruption event
            event = InterruptionEvent(
                interruption_type=interruption_type,
                conversation_state=self.current_state,
                audio_amplitude=self.amplitude_analyzer.current_amplitude,
                metadata=metadata or {}
            )
            
            # Save current response state for recovery
            if self.current_state == ConversationState.SPEAKING:
                self.conversation_buffer.save_context_snapshot({
                    "interruption_time": event.timestamp.isoformat(),
                    "state": self.current_state.value,
                    "metadata": metadata
                })
            
            # Transition to interrupted state
            await self._transition_to_state(ConversationState.INTERRUPTED)
            
            # Update metrics
            self.metrics.total_interruptions += 1
            if interruption_type not in self.metrics.interruptions_by_type:
                self.metrics.interruptions_by_type[interruption_type] = 0
            self.metrics.interruptions_by_type[interruption_type] += 1
            self.metrics.interruption_history.append(event)
            
            # Notify callbacks
            for callback in self.interruption_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error(f"❌ Interruption callback error: {e}")
            
            self.logger.info(f"🚫 Interruption triggered: {interruption_type.value}")
            
        except Exception as e:
            self.logger.error(f"❌ Error triggering interruption: {e}")
    
    async def _transition_to_state(self, new_state: ConversationState):
        """Transition to a new conversation state"""
        with self.state_lock:
            old_state = self.current_state
            self.previous_state = old_state
            self.current_state = new_state
            self.state_change_time = datetime.now()
        
        # Handle state-specific logic
        if new_state == ConversationState.LISTENING:
            self.conversation_buffer.clear_buffers()
        elif new_state == ConversationState.INTERRUPTED:
            # Prepare for recovery
            pass
        
        # Notify state change callbacks
        for callback in self.state_change_callbacks:
            try:
                callback(old_state, new_state)
            except Exception as e:
                self.logger.error(f"❌ State change callback error: {e}")
        
        self.logger.debug(f"🔄 State transition: {old_state.value} → {new_state.value}")
    
    async def start_speaking(self):
        """Indicate AI has started speaking"""
        await self._transition_to_state(ConversationState.SPEAKING)
    
    async def stop_speaking(self):
        """Indicate AI has finished speaking"""
        await self._transition_to_state(ConversationState.LISTENING)
    
    async def manual_interrupt(self):
        """Manually trigger an interruption"""
        await self._trigger_interruption(
            InterruptionType.MANUAL,
            metadata={"trigger": "manual", "user_initiated": True}
        )
    
    async def recover_from_interruption(self) -> Dict[str, Any]:
        """Attempt to recover from interruption and resume conversation"""
        try:
            recovery_start = time.time()
            
            # Get recovery data
            recovery_data = self.conversation_buffer.get_recovery_data()
            
            # Transition to resuming state
            await self._transition_to_state(ConversationState.RESUMING)
            
            # Calculate recovery time
            recovery_time = time.time() - recovery_start
            
            # Update metrics
            self.metrics.successful_recoveries += 1
            if self.metrics.interruption_history:
                self.metrics.interruption_history[-1].recovery_time = recovery_time
                self.metrics.interruption_history[-1].context_preserved = bool(recovery_data["partial_response"])
            
            # Update average recovery time
            total_recoveries = self.metrics.successful_recoveries + self.metrics.failed_recoveries
            if total_recoveries > 0:
                self.metrics.average_recovery_time = (
                    self.metrics.average_recovery_time * (total_recoveries - 1) + recovery_time
                ) / total_recoveries
            
            self.logger.info(f"✅ Interruption recovery completed in {recovery_time:.3f}s")
            return recovery_data
            
        except Exception as e:
            self.logger.error(f"❌ Interruption recovery failed: {e}")
            self.metrics.failed_recoveries += 1
            return {}
    
    def process_audio_chunk(self, audio_data: bytes):
        """Process audio chunk for client-side analysis"""
        self.amplitude_analyzer.process_audio_chunk(audio_data)
    
    def get_current_state(self) -> ConversationState:
        """Get current conversation state"""
        return self.current_state
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive interruption metrics"""
        # Calculate conversation quality score
        total_interruptions = self.metrics.total_interruptions
        successful_recoveries = self.metrics.successful_recoveries
        
        if total_interruptions > 0:
            recovery_rate = successful_recoveries / total_interruptions
            # Quality score based on recovery rate and interruption frequency
            session_duration = (datetime.now() - self.metrics.session_start).total_seconds()
            interruption_rate = total_interruptions / max(session_duration / 60, 1)  # per minute
            
            # Score from 0-100: penalize high interruption rate, reward good recovery
            quality_score = max(0, min(100, 
                (recovery_rate * 70) +                    # 70% weight for recovery
                (max(0, 5 - interruption_rate) * 6)       # 30% weight for low interruption rate
            ))
            self.metrics.conversation_quality_score = quality_score
        
        return {
            "current_state": self.current_state.value,
            "session_duration": (datetime.now() - self.metrics.session_start).total_seconds(),
            "total_interruptions": self.metrics.total_interruptions,
            "interruptions_by_type": {k.value: v for k, v in self.metrics.interruptions_by_type.items()},
            "successful_recoveries": self.metrics.successful_recoveries,
            "failed_recoveries": self.metrics.failed_recoveries,
            "average_recovery_time": self.metrics.average_recovery_time,
            "conversation_quality_score": self.metrics.conversation_quality_score,
            "current_amplitude": self.amplitude_analyzer.current_amplitude,
            "vad_active": self.is_vad_active,
            "recent_interruptions": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "type": event.interruption_type.value,
                    "recovery_time": event.recovery_time,
                    "context_preserved": event.context_preserved
                }
                for event in self.metrics.interruption_history[-10:]  # Last 10 interruptions
            ]
        }
    
    async def shutdown(self):
        """Clean shutdown of interruption handler"""
        try:
            self.amplitude_analyzer.stop_analysis()
            await self._transition_to_state(ConversationState.LISTENING)
            self.logger.info("✅ InterruptionHandler shutdown completed")
        except Exception as e:
            self.logger.error(f"❌ Error during InterruptionHandler shutdown: {e}")


# Factory function for easy creation
def create_interruption_handler(
    vad_threshold: float = 0.5,
    vad_prefix_padding_ms: int = 300,
    vad_silence_duration_ms: int = 200,
    amplitude_threshold: float = 0.01,
    logger: Optional[logging.Logger] = None
) -> InterruptionHandler:
    """Create InterruptionHandler with common configuration"""
    
    vad_config = VADConfig(
        threshold=vad_threshold,
        prefix_padding_ms=vad_prefix_padding_ms,
        silence_duration_ms=vad_silence_duration_ms
    )
    
    audio_config = ClientAudioConfig(
        amplitude_threshold=amplitude_threshold
    )
    
    return InterruptionHandler(vad_config, audio_config, logger) 