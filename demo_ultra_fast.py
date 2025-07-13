#!/usr/bin/env python3
"""
Ultra-Fast Voice Assistant Demo
Optimized for minimal lag and maximum responsiveness
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

class UltraFastVoiceAssistant:
    """Ultra-fast voice assistant with all optimizations"""
    
    def __init__(self):
        # Ultra-optimized configurations
        self.audio_config = AudioConfig(
            sample_rate=16000,
            chunk_size=256,     # Smaller chunks = lower latency
            channels=1
        )
        
        self.stt_config = STTConfig(
            model="whisper-1",
            silence_threshold=0.0005,  # More sensitive than default
            min_audio_length=0.1,      # Very responsive
            max_audio_length=8.0,      # Shorter max to prevent long waits
            temperature=0.0            # Deterministic = faster
        )
        
        self.tts_config = TTSConfig(
            model="tts-1",            # Fastest model
            voice="nova",             # Fast, clear voice
            speed=1.2,                # 20% faster speech
            response_format="mp3"
        )
        
        self.services_ready = False
        
    async def initialize_services(self):
        """Initialize all services for speed"""
        print("🚀 Initializing Ultra-Fast Voice Assistant...")
        
        # Initialize services
        self.audio_manager = AudioManager(config=self.audio_config)
        self.stt_service = WhisperSTTService(
            config=self.stt_config,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.tts_service = OpenAITTSService(
            config=self.tts_config,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize (synchronous calls)
        self.audio_manager.initialize()
        self.stt_service.initialize()
        self.tts_service.initialize()
        
        self.services_ready = True
        
        print("✅ Ultra-Fast Voice Assistant Ready!")
        print("🎯 Optimizations Active:")
        print("   • Real-time voice activity detection")
        print("   • Ultra-sensitive speech detection (0.0005)")
        print("   • 20% faster TTS speech")
        print("   • Minimal audio processing")
        print("   • Optimized for <2 second responses")
        print()
        
    async def record_with_realtime_vad(self, max_duration=6.0):
        """Record audio with real-time voice activity detection"""
        print("🎤 Listening... (I'll stop when you stop talking)")
        
        # Recording parameters
        sample_rate = 16000
        chunk_size = 256  # Small chunks for responsiveness
        silence_threshold = 0.0005
        silence_duration = 0.5  # Stop after 0.5s of silence
        
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
                        print(f"🔇 Silence detected, stopping recording ({chunk_count * chunk_size / sample_rate:.1f}s)")
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
        """Play WAV audio with minimal delay (same method as working demo)"""
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
            print(f"⚠️  Audio playback error: {e}")
    
    async def process_ultra_fast_turn(self):
        """Process one conversation turn with maximum speed"""
        if not self.services_ready:
            print("⚠️  Services not ready")
            return True
            
        total_start = time.time()
        
        try:
            # Step 1: Record with real-time VAD
            record_start = time.time()
            audio_data = await self.record_with_realtime_vad()
            record_time = time.time() - record_start
            
            if not audio_data:
                print("⚠️  No audio recorded")
                return True
            
            # Step 2: Transcribe speech
            stt_start = time.time()
            stt_result = await self.stt_service.transcribe_audio(audio_data)
            stt_time = time.time() - stt_start
            
            if not stt_result or not stt_result.text.strip():
                print("⚠️  No speech detected")
                return True
            
            user_text = stt_result.text.strip()
            print(f"🗣️  You: {user_text}")
            
            # Check for exit
            if user_text.lower() in ['goodbye', 'bye', 'exit']:
                print("👋 Goodbye!")
                return False
            
            # Step 3: Generate response (simple for speed)
            ai_response = f"I heard you say: {user_text}. Is there anything specific you'd like to know about that?"
            
            # Step 4: Generate TTS
            tts_start = time.time()
            tts_result = await self.tts_service.synthesize_speech(ai_response)
            tts_time = time.time() - tts_start
            
            if not tts_result:
                print("⚠️  TTS generation failed")
                return True
            
            # Step 5: Play response (convert MP3 to WAV first)
            play_start = time.time()
            wav_audio = self.tts_service.get_wav_audio(tts_result)
            if wav_audio:
                await self.play_audio_fast(wav_audio)
            else:
                print("⚠️  Failed to convert audio to playable format")
            play_time = time.time() - play_start
            
            # Performance metrics
            total_time = time.time() - total_start
            
            print(f"⚡ Ultra-Fast Response: {total_time:.2f}s total")
            print(f"   🎤 Record: {record_time:.2f}s")
            print(f"   🗣️  STT: {stt_time:.2f}s")
            print(f"   🎵 TTS: {tts_time:.2f}s")
            print(f"   🔊 Play: {play_time:.2f}s")
            
            # Show improvement
            old_time = 6.0 + stt_time + tts_time + play_time  # Old fixed recording
            improvement = ((old_time - total_time) / old_time) * 100
            print(f"   📈 Speed improvement: {improvement:.0f}% faster than before")
            print()
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return True
    
    async def run(self):
        """Run the ultra-fast voice assistant"""
        await self.initialize_services()
        
        print("🎤 Ultra-Fast Voice Assistant Active!")
        print("💬 Start speaking... I'll respond as fast as possible!")
        print("🎯 Target: <2 seconds total response time")
        print()
        
        while True:
            try:
                should_continue = await self.process_ultra_fast_turn()
                if not should_continue:
                    break
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(1)

async def main():
    assistant = UltraFastVoiceAssistant()
    await assistant.run()

if __name__ == "__main__":
    asyncio.run(main())
