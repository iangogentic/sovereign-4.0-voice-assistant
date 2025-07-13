"""
OpenAI Realtime API integration for Sovereign 4.0
Provides ultra-low latency speech-to-speech conversation
"""

import asyncio
import base64
import json
import os
import time
import logging
from typing import Optional, Dict, Any, Callable, AsyncGenerator
import websockets
from dataclasses import dataclass
import pyaudio
import wave
import threading
from queue import Queue
import io
import ssl

@dataclass
class RealtimeConfig:
    """Configuration for OpenAI Realtime API"""
    api_key: str
    model: str = "gpt-4o-realtime-preview"  # or gpt-4o-mini-realtime-preview for speed
    voice: str = "alloy"  # alloy, echo, fable, onyx, nova, shimmer
    instructions: str = "You are a helpful AI assistant with screen awareness capabilities."
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


class RealtimeVoiceService:
    """
    OpenAI Realtime API service for ultra-low latency voice conversations
    Replaces the traditional STT→LLM→TTS pipeline with direct speech-to-speech
    """
    
    def __init__(self, config: RealtimeConfig, memory_manager=None, screen_content_provider=None):
        self.config = config
        self.memory_manager = memory_manager
        self.screen_content_provider = screen_content_provider
        self.logger = logging.getLogger(__name__)
        
        # Connection state
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.session_id: Optional[str] = None
        self.is_connected = False
        self.is_recording = False
        
        # Audio handling
        self.audio_in_queue = Queue()
        self.audio_out_queue = Queue()
        self.pyaudio_instance = None
        self.input_stream = None
        self.output_stream = None
        
        # Event handlers
        self.on_response_received: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
    async def initialize(self) -> bool:
        """Initialize the Realtime service"""
        try:
            self.logger.info("🚀 Initializing OpenAI Realtime API service...")
            
            # Initialize PyAudio for local audio handling
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # Test API key
            if not self.config.api_key:
                raise ValueError("OpenAI API key is required for Realtime API")
                
            self.logger.info("✅ Realtime service initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Realtime service: {e}")
            return False
    
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
            
            # Start message handling
            asyncio.create_task(self._handle_messages())
            
            # Configure session
            await self._configure_session()
            
            self.logger.info("✅ Connected to OpenAI Realtime API")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Realtime API: {e}")
            self.is_connected = False
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
            
            # Start audio input/output streams
            await self._start_audio_streams()
            
            # Start audio processing tasks
            asyncio.create_task(self._process_audio_input())
            asyncio.create_task(self._process_audio_output())
            
            self.is_recording = True
            self.logger.info("✅ Real-time conversation active!")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start conversation: {e}")
    
    async def _start_audio_streams(self):
        """Start PyAudio input and output streams"""
        try:
            # Input stream (microphone)
            self.input_stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=1024,
                stream_callback=self._audio_input_callback
            )
            
            # Output stream (speakers)
            self.output_stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.config.sample_rate,
                output=True,
                frames_per_buffer=1024,
                stream_callback=self._audio_output_callback
            )
            
            self.input_stream.start_stream()
            self.output_stream.start_stream()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start audio streams: {e}")
            raise
    
    def _audio_input_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio input stream"""
        if self.is_recording and in_data:
            self.audio_in_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    def _audio_output_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio output stream"""
        try:
            if not self.audio_out_queue.empty():
                return (self.audio_out_queue.get(), pyaudio.paContinue)
            else:
                # Return silence if no audio available
                silence = b'\x00' * (frame_count * 2)  # 16-bit mono
                return (silence, pyaudio.paContinue)
        except:
            silence = b'\x00' * (frame_count * 2)
            return (silence, pyaudio.paContinue)
    
    async def _process_audio_input(self):
        """Process microphone input and send to Realtime API"""
        while self.is_connected and self.is_recording:
            try:
                if not self.audio_in_queue.empty():
                    audio_chunk = self.audio_in_queue.get()
                    
                    # Convert to base64 and send to API
                    audio_base64 = base64.b64encode(audio_chunk).decode('utf-8')
                    
                    message = {
                        "type": "input_audio_buffer.append",
                        "audio": audio_base64
                    }
                    
                    await self._send_message(message)
                
                await asyncio.sleep(0.01)  # Small delay to prevent overwhelming
                
            except Exception as e:
                self.logger.error(f"❌ Error processing audio input: {e}")
    
    async def _process_audio_output(self):
        """Process audio output from Realtime API"""
        # Audio output is handled in the message handler
        pass
    
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
        except Exception as e:
            self.logger.error(f"❌ Error handling messages: {e}")
            self.is_connected = False
    
    async def _process_message(self, data: Dict[str, Any]):
        """Process individual message from Realtime API"""
        message_type = data.get("type")
        
        if message_type == "session.created":
            self.session_id = data.get("session", {}).get("id")
            self.logger.info(f"✅ Session created: {self.session_id}")
        
        elif message_type == "session.updated":
            self.logger.info("✅ Session updated")
        
        elif message_type == "response.audio.delta":
            # Receive audio response and queue for playback
            audio_base64 = data.get("delta", "")
            if audio_base64:
                audio_bytes = base64.b64decode(audio_base64)
                self.audio_out_queue.put(audio_bytes)
        
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
            
            if self.on_response_received:
                await self.on_response_received(response_data)
        
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
                    max_context_length=1000
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
            self.is_recording = False
            
            if self.input_stream:
                self.input_stream.stop_stream()
                self.input_stream.close()
            
            if self.output_stream:
                self.output_stream.stop_stream()
                self.output_stream.close()
            
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
            
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
            
            self.logger.info("✅ Disconnected from Realtime API")
            
        except Exception as e:
            self.logger.error(f"❌ Error disconnecting: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "is_connected": self.is_connected,
            "is_recording": self.is_recording,
            "session_id": self.session_id,
            "audio_in_queue_size": self.audio_in_queue.qsize(),
            "audio_out_queue_size": self.audio_out_queue.qsize()
        }


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