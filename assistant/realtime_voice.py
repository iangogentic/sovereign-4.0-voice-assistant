"""
OpenAI Realtime API integration for Sovereign 4.0
Provides ultra-low latency speech-to-speech conversation
Enhanced for seamless integration with existing systems
"""

import asyncio
import base64
import json
import os
import time
import logging
from typing import Optional, Dict, Any, Callable, AsyncGenerator, List
import websockets
from dataclasses import dataclass, field
import pyaudio
import wave
import threading
from queue import Queue
import io
import ssl
from datetime import datetime

from .audio_stream_manager import AudioStreamManager, RealtimeAudioConfig, create_realtime_audio_manager
from .realtime_session_manager import (
    RealtimeSessionManager, SessionConfig, SessionState, 
    ConversationItemType, create_session_manager
)
from .realtime_error_handler import (
    RealtimeErrorHandler, RecoveryConfig, ErrorCategory,
    ErrorSeverity, RecoveryStrategy, create_error_handler
)


@dataclass
class RealtimeConfig:
    """Enhanced configuration for OpenAI Realtime API"""
    api_key: str
    model: str = "gpt-4o-realtime-preview"  # or gpt-4o-mini-realtime-preview for speed
    voice: str = "alloy"  # alloy, echo, fable, onyx, nova, shimmer
    instructions: str = "You are Sovereign, an advanced AI voice assistant with screen awareness and memory capabilities. Provide helpful, natural responses."
    temperature: float = 0.8
    max_response_output_tokens: str = "inf"
    
    # Audio settings
    input_audio_format: str = "pcm16"
    output_audio_format: str = "pcm16"
    sample_rate: int = 24000  # Required by Realtime API
    
    # VAD settings
    turn_detection_type: str = "server_vad"  # or "none" for push-to-talk
    vad_threshold: float = 0.5
    vad_prefix_padding_ms: int = 300
    vad_silence_duration_ms: int = 200
    
    # Connection settings
    connection_timeout: int = 30
    max_reconnect_attempts: int = 5
    initial_retry_delay: float = 1.0
    max_retry_delay: float = 30.0
    
    # Context settings
    max_context_length: int = 8000  # Token limit for context injection
    include_screen_content: bool = True
    include_memory_context: bool = True
    context_refresh_interval: float = 5.0  # Seconds

@dataclass
class ConversationMetrics:
    """Track conversation performance metrics"""
    session_start: datetime = field(default_factory=datetime.now)
    total_turns: int = 0
    total_audio_sent: int = 0
    total_audio_received: int = 0
    average_response_time: float = 0.0
    connection_count: int = 0
    error_count: int = 0
    fallback_count: int = 0
    last_response_time: float = 0.0


class RealtimeVoiceService:
    """
    Enhanced OpenAI Realtime API service for ultra-low latency voice conversations
    Replaces the traditional STT→LLM→TTS pipeline with direct speech-to-speech
    Seamlessly integrates with existing Sovereign 4.0 infrastructure
    """
    
    def __init__(self, config: RealtimeConfig, memory_manager=None, screen_content_provider=None, logger=None):
        self.config = config
        self.memory_manager = memory_manager
        self.screen_content_provider = screen_content_provider
        self.logger = logger or logging.getLogger(__name__)
        
        # Connection state
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.session_id: Optional[str] = None
        self.is_connected: bool = False
        self.is_running: bool = False
        self.connection_lock = asyncio.Lock()
        
        # Audio streaming with AudioStreamManager
        self.audio_stream_manager: Optional[AudioStreamManager] = None
        self.audio_config = RealtimeAudioConfig(
            sample_rate=config.sample_rate,
            input_chunk_size=1024,
            output_chunk_size=1024,
            buffer_duration=0.1,
            target_latency_ms=50.0
        )
        
        # Session management
        self.session_manager: Optional[RealtimeSessionManager] = None
        self.current_session_id: Optional[str] = None
        self.session_config = SessionConfig(
            database_path="data/realtime_sessions.db",
            session_timeout_minutes=30,
            max_concurrent_sessions=10,
            max_recovery_attempts=3
        )
        
        # Error handling
        self.error_handler: Optional[RealtimeErrorHandler] = None
        self.recovery_config = RecoveryConfig(
            max_connection_retries=5,
            max_audio_retries=3,
            max_api_retries=3,
            enable_graceful_degradation=True,
            enable_fallback_to_text=True,
            error_notification_threshold=5
        )
        
        # Context management
        self.current_context: Optional[str] = None
        self.last_context_update: float = 0
        
        # Metrics and monitoring
        self.metrics = ConversationMetrics()
        self.response_start_time: Optional[float] = None
        
        # Event handlers
        self.on_response_generated: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        self.on_connection_status: Optional[Callable] = None
        
    async def initialize(self) -> bool:
        """Initialize the Realtime service"""
        try:
            self.logger.info("🚀 Initializing OpenAI Realtime API service...")
            
            # Initialize Audio Stream Manager for real-time streaming
            self.audio_stream_manager = create_realtime_audio_manager(self.audio_config)
            if not self.audio_stream_manager.initialize():
                raise RuntimeError("Failed to initialize Audio Stream Manager")
            
            # Initialize Session Manager
            self.session_manager = create_session_manager(self.session_config)
            if not await self.session_manager.initialize():
                raise RuntimeError("Failed to initialize Session Manager")
            
            # Set up session event callbacks
            self.session_manager.on_session_created = self._on_session_created
            self.session_manager.on_session_expired = self._on_session_expired
            self.session_manager.on_session_recovered = self._on_session_recovered
            self.session_manager.on_session_error = self._on_session_error
            
            # Initialize Error Handler
            self.error_handler = create_error_handler(self.recovery_config)
            
            # Set up error event callbacks
            self.error_handler.on_error = self._on_error_occurred
            self.error_handler.on_recovery_success = self._on_recovery_success
            self.error_handler.on_recovery_failure = self._on_recovery_failure
            self.error_handler.on_degradation = self._on_degradation_event
            self.error_handler.on_critical_error = self._on_critical_error
            
            # Test API key
            if not self.config.api_key:
                raise ValueError("OpenAI API key is required for Realtime API")
                
            self.logger.info("✅ Realtime service initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Realtime service: {e}")
            return False
    
    async def _on_session_created(self, session_id: str, session_metadata):
        """Handle session creation event"""
        self.logger.info(f"🆕 Session created: {session_id}")
        self.current_session_id = session_id
        
    async def _on_session_expired(self, session_id: str):
        """Handle session expiration event"""
        self.logger.warning(f"⏰ Session expired: {session_id}")
        if self.current_session_id == session_id:
            self.current_session_id = None
            
    async def _on_session_recovered(self, session_id: str):
        """Handle session recovery event"""
        self.logger.info(f"🔄 Session recovered: {session_id}")
        
    async def _on_session_error(self, session_id: str, error_info):
        """Handle session error event"""
        self.logger.error(f"❌ Session error in {session_id}: {error_info}")
        if self.current_session_id == session_id:
            # Attempt recovery or fallback
            if self.session_manager:
                recovery_success = await self.session_manager.attempt_session_recovery(session_id)
                if not recovery_success:
                    self.current_session_id = None
    
    async def _on_error_occurred(self, error_info):
        """Handle error occurrence event"""
        self.logger.warning(f"🚨 Error occurred: {error_info.category.value} - {error_info.user_message}")
        
        # Update metrics
        self.metrics.error_count += 1
        
        # Notify user of error if user-friendly message exists
        if error_info.user_message and self.on_error:
            await self.on_error(error_info.user_message)
    
    async def _on_recovery_success(self, error_info):
        """Handle successful error recovery"""
        self.logger.info(f"✅ Recovery successful for {error_info.category.value}")
        
        # Reset error count on successful recovery
        if self.error_handler:
            await self.error_handler.reset_error_state(error_info.category)
    
    async def _on_recovery_failure(self, error_info, failure_reason):
        """Handle failed error recovery"""
        self.logger.error(f"❌ Recovery failed for {error_info.category.value}: {failure_reason}")
        
        # Increment fallback count
        self.metrics.fallback_count += 1
        
        # Consider more drastic measures
        if error_info.category in [ErrorCategory.CONNECTION_ERROR, ErrorCategory.AUTHENTICATION_ERROR]:
            await self.disconnect()
    
    async def _on_degradation_event(self, degradation_type, message):
        """Handle degradation event"""
        self.logger.warning(f"📉 Degradation activated: {degradation_type} - {message}")
        
        # Update degradation state
        if degradation_type == "audio_quality":
            # Reduce audio quality in audio stream manager
            if self.audio_stream_manager:
                # Audio quality reduction would be implemented in audio stream manager
                pass
        elif degradation_type == "text_mode":
            # Switch to text-only mode
            self.logger.info("💬 Switching to text-only mode due to persistent issues")
    
    async def _on_critical_error(self, error_info):
        """Handle critical error event"""
        self.logger.error(f"🚨 CRITICAL ERROR: {error_info.message}")
        
        # For critical errors, we may need to terminate the session
        if error_info.category == ErrorCategory.AUTHENTICATION_ERROR:
            self.logger.critical("🔒 Authentication failed - terminating session")
            await self.disconnect()
        elif error_info.user_message == "System experiencing high error rate. Please try again later.":
            self.logger.critical("⚡ High error rate detected - considering service restart")
            # Could trigger service restart or maintenance mode
    
    async def connect(self) -> bool:
        """Connect to OpenAI Realtime API via WebSocket"""
        try:
            self.logger.info("🔌 Connecting to OpenAI Realtime API...")
            
            # WebSocket URL for Realtime API
            url = "wss://api.openai.com/v1/realtime?model=" + self.config.model
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "OpenAI-Beta": "realtime=v1"
            }
            
            # Create SSL context for testing (bypass certificate verification)
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Connect to WebSocket
            self.websocket = await websockets.connect(url, extra_headers=headers, ssl=ssl_context)
            self.is_connected = True
            
            # Create new session in session manager
            if self.session_manager:
                self.current_session_id = await self.session_manager.create_session(
                    model=self.config.model,
                    voice=self.config.voice,
                    instructions=self.config.instructions,
                    temperature=self.config.temperature,
                    max_response_output_tokens=self.config.max_response_output_tokens
                )
                await self.session_manager.update_session_state(self.current_session_id, SessionState.CONNECTED)
            
            # Start message handling
            asyncio.create_task(self._handle_messages())
            
            # Configure session
            await self._configure_session()
            
            self.logger.info("✅ Connected to OpenAI Realtime API")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Realtime API: {e}")
            self.is_connected = False
            
            # Handle error through error handler
            if self.error_handler:
                await self.error_handler.handle_error(e, {
                    "session_id": self.current_session_id,
                    "operation": "connect",
                    "url": "wss://api.openai.com/v1/realtime"
                })
            
            return False
    
    async def _configure_session(self):
        """Configure the Realtime API session"""
        try:
            # Prepare context from memory and screen
            context_instructions = self.config.instructions
            
            # Add screen awareness context
            if self.screen_content_provider:
                screen_content = await self._get_current_screen_content()
                if screen_content:
                    context_instructions += f"\n\nCURRENT SCREEN CONTENT:\n{screen_content}"
            
            # Add memory context  
            if self.memory_manager:
                memory_context = await self._get_memory_context()
                if memory_context:
                    context_instructions += f"\n\nCONVERSATION HISTORY:\n{memory_context}"
            
            session_config = {
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "instructions": context_instructions,
                    "voice": self.config.voice,
                    "input_audio_format": self.config.input_audio_format,
                    "output_audio_format": self.config.output_audio_format,
                    "input_audio_transcription": {
                        "model": "whisper-1"
                    },
                    "turn_detection": {
                        "type": self.config.turn_detection_type,
                        "threshold": self.config.vad_threshold,
                        "prefix_padding_ms": self.config.vad_prefix_padding_ms,
                        "silence_duration_ms": self.config.vad_silence_duration_ms,
                        "create_response": True
                    },
                    "temperature": self.config.temperature,
                    "max_response_output_tokens": self.config.max_response_output_tokens
                }
            }
            
            await self._send_message(session_config)
            self.logger.info("✅ Session configured with enhanced context")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to configure session: {e}")
    
    async def start_conversation(self):
        """Start real-time conversation mode"""
        try:
            self.logger.info("🎤 Starting real-time conversation mode...")
            
            # Start real-time audio streams with AudioStreamManager
            await self._start_realtime_audio_streams()
            
            # Start message handling
            asyncio.create_task(self._handle_messages())
            
            self.is_running = True
            self.logger.info("✅ Real-time conversation active!")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start conversation: {e}")
    
    async def _start_realtime_audio_streams(self):
        """Start real-time audio streams using AudioStreamManager"""
        try:
            if not self.audio_stream_manager:
                raise RuntimeError("AudioStreamManager not initialized")
            
            # Start input stream with callback
            if not self.audio_stream_manager.start_input_stream(self._on_audio_input):
                raise RuntimeError("Failed to start input stream")
            
            # Start output stream with callback
            if not self.audio_stream_manager.start_output_stream(self._on_audio_output_request):
                raise RuntimeError("Failed to start output stream")
            
            self.logger.info("🎵 Real-time audio streams started successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start real-time audio streams: {e}")
            raise
    
    def _on_audio_input(self, audio_data: bytes):
        """Callback for real-time audio input from AudioStreamManager"""
        try:
            if self.is_running and self.websocket:
                # Convert to base64 and send to Realtime API
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                
                message = {
                    "type": "input_audio_buffer.append",
                    "audio": audio_base64
                }
                
                # Schedule async message sending
                asyncio.create_task(self._send_message(message))
                
                # Update metrics
                self.metrics.total_audio_sent += len(audio_data)
                
        except Exception as e:
            self.logger.error(f"❌ Error in audio input callback: {e}")
    
    def _on_audio_output_request(self) -> Optional[bytes]:
        """Callback for real-time audio output request from AudioStreamManager"""
        try:
            # This will be populated by _handle_audio_delta when receiving audio from API
            # For now, return None (silence will be handled by AudioStreamManager)
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error in audio output callback: {e}")
            return None
    
    # Audio processing methods removed - now handled by AudioStreamManager callbacks
    
    async def _handle_messages(self):
        """Handle incoming messages from Realtime API"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._process_message(data)
                except json.JSONDecodeError as e:
                    self.logger.error(f"❌ Failed to parse message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.info("🔌 WebSocket connection closed")
            self.is_connected = False
            
            # Handle connection closed through error handler
            if self.error_handler:
                await self.error_handler.handle_error(
                    ConnectionError("WebSocket connection closed"),
                    {"session_id": self.current_session_id, "operation": "message_handling"}
                )
        except Exception as e:
            self.logger.error(f"❌ Error handling messages: {e}")
            self.is_connected = False
            
            # Handle error through error handler
            if self.error_handler:
                await self.error_handler.handle_error(e, {
                    "session_id": self.current_session_id,
                    "operation": "message_handling"
                })
    
    async def _process_message(self, data: Dict[str, Any]):
        """Process individual message from Realtime API"""
        message_type = data.get("type")
        
        if message_type == "session.created":
            self.session_id = data.get("session", {}).get("id")
            self.logger.info(f"✅ Session created: {self.session_id}")
        
        elif message_type == "session.updated":
            self.logger.info("✅ Session updated")
        
        elif message_type == "response.audio.delta":
            # Receive audio response and send to AudioStreamManager for playback
            audio_base64 = data.get("delta", "")
            if audio_base64:
                audio_bytes = base64.b64decode(audio_base64)
                if self.audio_stream_manager:
                    self.audio_stream_manager.handle_audio_chunk(audio_bytes)
                
                # Store audio response in session manager
                if self.session_manager and self.current_session_id:
                    await self.session_manager.add_conversation_item(
                        session_id=self.current_session_id,
                        item_type=ConversationItemType.AUDIO,
                        role="assistant",
                        content=f"Audio response delta ({len(audio_bytes)} bytes)",
                        audio_data=audio_bytes
                    )
                
                # Update metrics
                self.metrics.total_audio_received += len(audio_bytes)
        
        elif message_type == "response.audio_transcript.delta":
            # Log transcript for debugging
            transcript = data.get("delta", "")
            if transcript:
                self.logger.debug(f"🗣️ AI: {transcript}")
        
        elif message_type == "input_audio_buffer.speech_started":
            self.logger.debug("🎤 Speech detected")
        
        elif message_type == "input_audio_buffer.speech_stopped":
            self.logger.debug("🎤 Speech ended")
        
        elif message_type == "response.done":
            # Response completed
            response_data = data.get("response", {})
            
            # Store conversation in memory if available
            if self.memory_manager:
                await self._store_conversation_turn(response_data)
            
            if self.on_response_generated:
                await self.on_response_generated(response_data)
        
        elif message_type == "error":
            error_info = data.get("error", {})
            self.logger.error(f"❌ Realtime API error: {error_info}")
            if self.on_error:
                await self.on_error(error_info)
    
    async def _send_message(self, message: Dict[str, Any]):
        """Send message to Realtime API"""
        if self.websocket and self.is_connected:
            try:
                await self.websocket.send(json.dumps(message))
            except Exception as e:
                self.logger.error(f"❌ Failed to send message: {e}")
    
    async def _get_current_screen_content(self) -> Optional[str]:
        """Get current screen content for context"""
        if self.screen_content_provider:
            try:
                # This would interface with your existing screen_watcher
                return await self.screen_content_provider.get_latest_content()
            except Exception as e:
                self.logger.error(f"❌ Failed to get screen content: {e}")
        return None
    
    async def _get_memory_context(self) -> Optional[str]:
        """Get conversation memory context"""
        if self.memory_manager:
            try:
                context = await self.memory_manager.inject_context(
                    current_query="voice conversation",
                    max_context_length=self.config.max_context_length
                )
                return context
            except Exception as e:
                self.logger.error(f"❌ Failed to get memory context: {e}")
        return None
    
    async def _store_conversation_turn(self, response_data: Dict[str, Any]):
        """Store conversation turn in memory"""
        if self.memory_manager:
            try:
                # Extract transcript from response
                output_items = response_data.get("output", [])
                for item in output_items:
                    if item.get("type") == "message":
                        content = item.get("content", [])
                        for content_part in content:
                            if content_part.get("type") == "audio":
                                transcript = content_part.get("transcript", "")
                                if transcript:
                                    await self.memory_manager.store_conversation(
                                        user_query="[Voice Input]",
                                        assistant_response=transcript,
                                        metadata={"source": "realtime_api", "timestamp": time.time()}
                                    )
                                    break
            except Exception as e:
                self.logger.error(f"❌ Failed to store conversation: {e}")
    
    async def send_text_message(self, text: str):
        """Send a text message (for testing or hybrid mode)"""
        message = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}]
            }
        }
        await self._send_message(message)
        
        # Trigger response
        await self._send_message({"type": "response.create"})
    
    async def interrupt_response(self):
        """Interrupt the current AI response"""
        await self._send_message({"type": "response.cancel"})
    
    async def stop_conversation(self):
        """Stop the real-time conversation"""
        try:
            self.is_running = False
            
            # Stop AudioStreamManager
            if self.audio_stream_manager:
                self.audio_stream_manager.stop_all_streams()
            
            self.logger.info("✅ Conversation stopped")
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping conversation: {e}")
    
    async def disconnect(self):
        """Disconnect from Realtime API"""
        try:
            self.is_connected = False
            await self.stop_conversation()
            
            if self.websocket:
                await self.websocket.close()
            
            # Cleanup AudioStreamManager
            if self.audio_stream_manager:
                self.audio_stream_manager.cleanup()
            
            # Terminate current session in session manager
            if self.session_manager and self.current_session_id:
                await self.session_manager.terminate_session(self.current_session_id, "disconnect")
                self.current_session_id = None
            
            self.logger.info("✅ Disconnected from Realtime API")
            
        except Exception as e:
            self.logger.error(f"❌ Error disconnecting: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        base_metrics = {
            "is_connected": self.is_connected,
            "is_running": self.is_running,
            "session_id": self.session_id,
            "current_session_id": self.current_session_id,
            "total_audio_sent": self.metrics.total_audio_sent,
            "total_audio_received": self.metrics.total_audio_received,
            "total_turns": self.metrics.total_turns,
            "connection_count": self.metrics.connection_count,
            "error_count": self.metrics.error_count,
            "fallback_count": self.metrics.fallback_count,
            "average_response_time": self.metrics.average_response_time,
            "last_response_time": self.metrics.last_response_time
        }
        
        # Add AudioStreamManager metrics if available
        if self.audio_stream_manager:
            try:
                audio_metrics = self.audio_stream_manager.get_performance_metrics()
                base_metrics.update({
                    "audio_stream_metrics": audio_metrics
                })
            except Exception:
                base_metrics["audio_stream_metrics"] = {"status": "unavailable"}
        
        # Add Session Manager metrics if available
        if self.session_manager:
            try:
                session_metrics = self.session_manager.get_metrics()
                base_metrics.update({
                    "session_manager_metrics": session_metrics
                })
            except Exception:
                base_metrics["session_manager_metrics"] = {"status": "unavailable"}
        
        # Add Error Handler metrics if available
        if self.error_handler:
            try:
                error_metrics = self.error_handler.get_error_metrics()
                base_metrics.update({
                    "error_handler_metrics": error_metrics
                })
            except Exception:
                base_metrics["error_handler_metrics"] = {"status": "unavailable"}
        
        return base_metrics


# Example usage
async def test_realtime_service():
    """Test the Realtime API service"""
    config = RealtimeConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini-realtime-preview",  # Faster for testing
        voice="alloy",
        instructions="You are a helpful AI assistant. Be conversational and engaging."
    )
    
    service = RealtimeVoiceService(config)
    
    try:
        await service.initialize()
        await service.connect()
        await service.start_conversation()
        
        # Let it run for a bit
        await asyncio.sleep(30)
        
    finally:
        await service.disconnect()


if __name__ == "__main__":
    asyncio.run(test_realtime_service()) 