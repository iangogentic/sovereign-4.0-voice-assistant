#!/usr/bin/env python3
"""
Voice Assistant Demo with Fixed Voice Activity Detection
"""

import asyncio
import logging
import os
import time
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import our components
from assistant.audio import AudioManager, AudioConfig
from assistant.stt import WhisperSTTService, STTConfig
from assistant.tts import OpenAITTSService, TTSConfig
from assistant.monitoring import PerformanceMonitor, PipelineStage

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FixedVoiceAssistant:
    """Voice assistant with fixed voice activity detection"""
    
    def __init__(self):
        self.monitor = None
        self.audio_manager = None
        self.stt_service = None
        self.tts_service = None
    
    def process_text(self, text: str) -> str:
        """Process transcribed text and return response"""
        text = text.strip().lower()
        
        if "hello" in text or "hi" in text:
            return "Hello! Nice to meet you. The voice assistant is working!"
        elif "test" in text:
            return "Test successful! The voice pipeline is working correctly."
        elif "weather" in text:
            return "I don't have access to weather data, but the voice system is responding!"
        else:
            return f"I heard you say: {text}. This is just a simple echo response."
    
    async def initialize(self):
        """Initialize the voice assistant components"""
        logger.info("Initializing voice assistant...")
        
        # Check API key
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY not found in environment variables")
            return False
        
        try:
            # Initialize performance monitoring
            self.monitor = PerformanceMonitor()
            
            # Initialize audio components
            audio_config = AudioConfig(
                sample_rate=16000,
                chunk_size=1024,
                channels=1
            )
            self.audio_manager = AudioManager(config=audio_config)
            self.audio_manager.initialize()
            
            # Initialize STT service with LOWER silence threshold
            stt_config = STTConfig(
                model="whisper-1",
                language="en",
                silence_threshold=0.001,  # Much lower threshold!
                vad_enabled=True,
                min_audio_length=0.5
            )
            self.stt_service = WhisperSTTService(
                config=stt_config,
                api_key=os.getenv("OPENAI_API_KEY"),
                org_id=os.getenv("OPENAI_ORG_ID")
            )
            
            # Initialize TTS service
            tts_config = TTSConfig(
                model="tts-1",
                voice="alloy",
                speed=1.0
            )
            self.tts_service = OpenAITTSService(
                config=tts_config,
                api_key=os.getenv("OPENAI_API_KEY"),
                org_id=os.getenv("OPENAI_ORG_ID")
            )
            
            # Initialize services
            try:
                self.stt_service.initialize()
                logger.info("STT service initialized")
            except Exception as e:
                logger.warning(f"STT service initialization had issues: {e}, but continuing...")
            
            try:
                self.tts_service.initialize()
                logger.info("TTS service initialized")
            except Exception as e:
                logger.warning(f"TTS service initialization had issues: {e}, but continuing...")
            
            # Set up audio streams
            if not self.audio_manager.setup_input_stream():
                logger.error("Failed to setup audio input stream")
                return False
            
            if not self.audio_manager.setup_output_stream():
                logger.error("Failed to setup audio output stream")
                return False
            
            logger.info("Voice assistant initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize voice assistant: {e}")
            return False
    
    async def record_audio(self, duration: float = 5.0):
        """Record audio for a specified duration"""
        logger.info(f"Recording for {duration} seconds...")
        
        # Start recording
        if not self.audio_manager.start_recording():
            logger.error("Failed to start recording")
            return None
        
        # Collect audio data
        audio_chunks = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            chunk = self.audio_manager.read_audio_chunk()
            if chunk:
                audio_chunks.append(chunk)
            await asyncio.sleep(0.01)  # Small delay
        
        # Stop recording
        self.audio_manager.stop_recording()
        
        # Combine audio chunks
        if audio_chunks:
            audio_data = b''.join(audio_chunks)
            logger.info(f"Recorded {len(audio_data)} bytes of audio")
            return audio_data
        else:
            logger.warning("No audio data recorded")
            return None
    
    async def process_voice_interaction(self):
        """Process a single voice interaction"""
        try:
            # Record audio
            audio_data = await self.record_audio(duration=5.0)
            if not audio_data:
                return
            
            # Convert speech to text
            logger.info("Converting speech to text...")
            with self.monitor.timing_context(PipelineStage.STT_PROCESSING):
                stt_result = await self.stt_service.transcribe_audio(audio_data)
            
            if not stt_result or not stt_result.text.strip():
                logger.warning("No speech detected or transcription failed")
                logger.info("💡 Try speaking louder or closer to the microphone")
                return
            
            logger.info(f"✅ Transcribed: '{stt_result.text}'")
            
            # Process the text
            response_text = self.process_text(stt_result.text)
            logger.info(f"💬 Response: '{response_text}'")
            
            # Convert response to speech
            logger.info("Converting response to speech...")
            with self.monitor.timing_context(PipelineStage.TTS_GENERATION):
                tts_result = await self.tts_service.synthesize_speech(response_text)
            
            if not tts_result:
                logger.error("Failed to synthesize speech")
                return
            
            # Play the response
            logger.info("Playing response...")
            with self.monitor.timing_context(PipelineStage.AUDIO_PLAYBACK):
                wav_audio = self.tts_service.get_wav_audio(tts_result)
                if wav_audio:
                    logger.info(f"🎵 Generated {len(wav_audio)} bytes of audio response")
                    logger.info("🔊 (Audio playback not implemented yet, but TTS is working!)")
                else:
                    logger.error("Failed to get WAV audio")
                    
        except Exception as e:
            logger.error(f"Error in voice interaction: {e}")
    
    async def run(self):
        """Run the voice assistant"""
        if not await self.initialize():
            return
        
        try:
            logger.info("🎤 Voice Assistant is ready!")
            logger.info("🔊 Fixed Voice Activity Detection - lower threshold")
            logger.info("📊 Performance monitoring enabled")
            logger.info("⏹️  Press Ctrl+C to exit")
            
            while True:
                try:
                    user_input = input("\nPress Enter to start 5-second recording (or 'q' to quit): ")
                    
                    if user_input.lower() == 'q':
                        break
                    
                    # Process voice interaction
                    await self.process_voice_interaction()
                    
                    # Show performance stats
                    stats = self.monitor.get_summary()
                    if stats:
                        logger.info(f"📊 Performance: {stats}")
                    
                except KeyboardInterrupt:
                    break
                except EOFError:
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    
        except KeyboardInterrupt:
            logger.info("Shutting down voice assistant...")
        finally:
            # Clean shutdown
            if self.audio_manager:
                self.audio_manager.cleanup()
            
            logger.info("Voice assistant shut down complete")

def main():
    """Main function"""
    assistant = FixedVoiceAssistant()
    asyncio.run(assistant.run())

if __name__ == "__main__":
    main() 