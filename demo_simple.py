#!/usr/bin/env python3
"""
Simple conversational voice assistant - starts up, greets user, then listens/responds
"""

import asyncio
import logging
import os
import time
import io
import wave
import numpy as np
import sounddevice as sd

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

class SimpleConversationalAssistant:
    """Simple conversational voice assistant"""
    
    def __init__(self):
        self.audio_manager = None
        self.stt_service = None
        self.tts_service = None
        self.monitor = None
        self.running = False
        
    def process_text(self, text: str) -> str:
        """Process user input and generate response"""
        text = text.lower().strip()
        
        if "hello" in text or "hi" in text:
            return "Hello! Nice to hear from you. How can I help you today?"
        elif "how are you" in text:
            return "I'm doing great, thanks for asking! How are you doing?"
        elif "what's your name" in text:
            return "I'm your AI voice assistant. You can call me Assistant."
        elif "goodbye" in text or "bye" in text:
            return "Goodbye! Have a great day!"
        elif "thank you" in text or "thanks" in text:
            return "You're welcome! Happy to help."
        elif "weather" in text:
            return "I don't have access to current weather data, but I hope it's nice where you are!"
        elif "time" in text:
            return f"I don't have access to the current time, but I'm here whenever you need me!"
        else:
            return f"I heard you say: {text}. That's interesting! Tell me more."
    
    async def initialize(self):
        """Initialize the voice assistant"""
        logger.info("Starting up voice assistant...")
        
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY not found")
            return False
        
        try:
            # Initialize components
            self.monitor = PerformanceMonitor()
            
            # Audio setup
            audio_config = AudioConfig(sample_rate=16000, chunk_size=1024, channels=1)
            self.audio_manager = AudioManager(config=audio_config)
            self.audio_manager.initialize()
            
            # STT setup
            stt_config = STTConfig(model="whisper-1", language="en")
            self.stt_service = WhisperSTTService(
                config=stt_config,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            self.stt_service.initialize()
            
            # TTS setup
            tts_config = TTSConfig(model="tts-1", voice="alloy", speed=1.0)
            self.tts_service = OpenAITTSService(
                config=tts_config,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            self.tts_service.initialize()
            
            # Setup audio streams
            self.audio_manager.setup_input_stream()
            self.audio_manager.setup_output_stream()
            
            logger.info("Voice assistant initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return False
    
    async def speak(self, text: str):
        """Convert text to speech and play it"""
        try:
            logger.info(f"Speaking: {text}")
            
            # Generate speech
            tts_result = await self.tts_service.synthesize_speech(text)
            if not tts_result:
                logger.error("Failed to generate speech")
                return
            
            # Get audio data
            wav_audio = self.tts_service.get_wav_audio(tts_result)
            if not wav_audio:
                logger.error("Failed to get audio data")
                return
            
            # Play the audio
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
                
            logger.info("Finished speaking")
            
        except Exception as e:
            logger.error(f"Error speaking: {e}")
    
    async def listen(self, duration: float = 5.0):
        """Listen for user input"""
        try:
            logger.info(f"Listening for {duration} seconds...")
            
            # Start recording
            if not self.audio_manager.start_recording():
                return None
            
            # Collect audio
            audio_chunks = []
            start_time = time.time()
            
            while time.time() - start_time < duration:
                chunk = self.audio_manager.read_audio_chunk()
                if chunk:
                    audio_chunks.append(chunk)
                await asyncio.sleep(0.01)
            
            # Stop recording
            self.audio_manager.stop_recording()
            
            if not audio_chunks:
                return None
            
            # Combine audio
            audio_data = b''.join(audio_chunks)
            
            # Convert to text
            stt_result = await self.stt_service.transcribe_audio(audio_data)
            if stt_result and stt_result.text.strip():
                return stt_result.text.strip()
            
            return None
            
        except Exception as e:
            logger.error(f"Error listening: {e}")
            return None
    
    async def conversation_loop(self):
        """Main conversation loop"""
        logger.info("Starting conversation...")
        
        # Initial greeting
        await self.speak("Hello Ian! I'm your voice assistant. I'm ready to chat with you.")
        
        conversation_count = 0
        
        while self.running:
            try:
                # Listen for user input
                print("\n🎤 Listening... (speak now)")
                user_text = await self.listen(duration=6.0)
                
                if user_text:
                    print(f"You said: {user_text}")
                    
                    # Check for exit commands
                    if "goodbye" in user_text.lower() or "bye" in user_text.lower():
                        await self.speak("Goodbye Ian! It was nice talking with you!")
                        break
                    
                    # Generate and speak response
                    response = self.process_text(user_text)
                    await self.speak(response)
                    
                    conversation_count += 1
                    
                else:
                    # No speech detected, give a gentle prompt
                    if conversation_count == 0:
                        await self.speak("I'm here when you're ready to talk. Just say something!")
                    elif conversation_count % 3 == 0:  # Every 3rd silent cycle
                        await self.speak("I'm still listening if you'd like to chat.")
                
                # Short pause between interactions
                await asyncio.sleep(1.0)
                
            except KeyboardInterrupt:
                logger.info("Stopping conversation...")
                break
            except Exception as e:
                logger.error(f"Error in conversation loop: {e}")
                await asyncio.sleep(2.0)
    
    async def run(self):
        """Run the voice assistant"""
        if not await self.initialize():
            logger.error("Failed to initialize voice assistant")
            return
        
        self.running = True
        
        try:
            print("🎤 Voice Assistant Started!")
            print("💬 I'll greet you and then we can have a conversation")
            print("🛑 Press Ctrl+C to stop")
            
            await self.conversation_loop()
            
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.running = False
            if self.audio_manager:
                self.audio_manager.cleanup()
            print("👋 Voice assistant stopped")

def main():
    """Main function"""
    assistant = SimpleConversationalAssistant()
    asyncio.run(assistant.run())

if __name__ == "__main__":
    main() 