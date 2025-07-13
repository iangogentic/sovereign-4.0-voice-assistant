#!/usr/bin/env python3
"""
Optimized Voice Assistant Demo
Takes the best of both worlds: working demo's simplicity + real speed optimizations
"""

import asyncio
import logging
import os
import time
import io
import wave
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
load_dotenv()

from assistant.audio import AudioManager, AudioConfig
from assistant.stt import WhisperSTTService, STTConfig
from assistant.tts import OpenAITTSService, TTSConfig

# Minimal logging for speed
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class OptimizedVoiceAssistant:
    """Optimized voice assistant - fast but reliable"""
    
    def __init__(self):
        self.audio_manager = None
        self.stt_service = None
        self.tts_service = None
        self.services_ready = False
        
    async def initialize_services(self):
        """Initialize all services"""
        print("🚀 Initializing Optimized Voice Assistant...")
        
        # Audio setup - use working demo's proven config
        audio_config = AudioConfig(
            sample_rate=16000,
            chunk_size=1024,  # Proven chunk size from working demo
            channels=1
        )
        self.audio_manager = AudioManager(config=audio_config)
        self.audio_manager.initialize()
        self.audio_manager.setup_input_stream()
        
        # STT setup - use the FIXED VAD threshold that works
        stt_config = STTConfig(
            model="whisper-1",
            language="en",
            silence_threshold=0.001,  # 🔧 FIXED threshold that works
            vad_enabled=True,
            min_audio_length=0.3,     # 🔧 FIXED minimum length
            temperature=0.0           # Deterministic = faster
        )
        self.stt_service = WhisperSTTService(
            config=stt_config,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.stt_service.initialize()
        
        # TTS setup - optimized for speed
        tts_config = TTSConfig(
            model="tts-1",            # Fastest TTS model
            voice="nova",             # Fast, clear voice
            speed=1.15,               # 15% faster speech
            response_format="mp3"
        )
        self.tts_service = OpenAITTSService(
            config=tts_config,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.tts_service.initialize()
        
        self.services_ready = True
        
        print("✅ Optimized Voice Assistant Ready!")
        print("🎯 Optimizations:")
        print("   • Proven VAD threshold (0.001)")
        print("   • Shorter recording duration (3.5s)")
        print("   • 15% faster TTS speech")
        print("   • Optimized audio processing")
        print("   • Target: 3-4 second total response")
        print()
        
    async def listen_optimized(self, duration: float = 3.5):
        """Listen with optimized duration - shorter than original but reliable"""
        try:
            print(f"🎤 Listening for {duration}s...")
            
            # Start recording (using proven method)
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
            else:
                return None
            
        except Exception as e:
            logger.error(f"Error listening: {e}")
            return None
    
    async def speak_optimized(self, text: str):
        """Speak with optimized audio playback"""
        try:
            # Generate speech
            tts_result = await self.tts_service.synthesize_speech(text)
            if not tts_result:
                return
            
            # Get audio data and play (using proven method)
            wav_audio = self.tts_service.get_wav_audio(tts_result)
            if not wav_audio:
                return
            
            # Play audio using the same method as working demo
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
            logger.error(f"Error speaking: {e}")
    
    def process_text(self, text: str) -> str:
        """Process user input and generate response (same as working demo)"""
        text = text.lower().strip()
        
        if "hello" in text or "hi" in text:
            return "Hello! Nice to hear from you. How can I help you today?"
        elif "how are you" in text:
            return "I'm doing great, thanks for asking! How are you doing?"
        elif "what's your name" in text:
            return "I'm your optimized AI voice assistant. Ready to help!"
        elif "goodbye" in text or "bye" in text:
            return "Goodbye! Have a great day!"
        elif "thank you" in text or "thanks" in text:
            return "You're welcome! Happy to help."
        elif "weather" in text:
            return "I don't have access to current weather data, but I hope it's nice where you are!"
        elif "time" in text:
            return "I don't have access to the current time, but I'm here whenever you need me!"
        else:
            return f"I heard you say: {text}. That's interesting! Tell me more."
    
    async def conversation_turn(self):
        """Handle one optimized conversation turn"""
        if not self.services_ready:
            return False
            
        total_start = time.time()
        
        # Step 1: Listen (optimized duration)
        listen_start = time.time()
        user_text = await self.listen_optimized()
        listen_time = time.time() - listen_start
        
        if not user_text:
            print("⚠️  No speech detected, try again.")
            return True
        
        print(f"🗣️  You: {user_text}")
        
        # Check for exit
        if user_text.lower() in ['goodbye', 'bye', 'exit']:
            await self.speak_optimized("Goodbye! Have a great day!")
            return False
        
        # Step 2: Process text (instant)
        ai_response = self.process_text(user_text)
        
        # Step 3: Speak response
        speak_start = time.time()
        await self.speak_optimized(ai_response)
        speak_time = time.time() - speak_start
        
        # Performance metrics
        total_time = time.time() - total_start
        
        print(f"⚡ Response Time: {total_time:.2f}s total")
        print(f"   🎤 Listen: {listen_time:.2f}s")
        print(f"   🔊 Speak: {speak_time:.2f}s")
        
        # Show improvement vs original
        old_time = 5.0 + speak_time  # Old 5s recording + speak time
        if total_time < old_time:
            improvement = ((old_time - total_time) / old_time) * 100
            print(f"   📈 {improvement:.0f}% faster than original")
        
        print()
        return True
    
    async def run(self):
        """Run the optimized voice assistant"""
        await self.initialize_services()
        
        print("🎤 Optimized Voice Assistant Active!")
        print("💬 Start speaking... I'll respond quickly!")
        print()
        
        while True:
            try:
                should_continue = await self.conversation_turn()
                if not should_continue:
                    break
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(1)

async def main():
    assistant = OptimizedVoiceAssistant()
    await assistant.run()

if __name__ == "__main__":
    asyncio.run(main()) 