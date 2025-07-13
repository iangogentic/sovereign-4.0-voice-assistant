#!/usr/bin/env python3
"""
Improved demo script with variable recording duration and actual voice playback
"""

import asyncio
import logging
import os
import time
from pathlib import Path
import threading

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

class ImprovedVoiceAssistant:
    """Improved voice assistant with better recording and playback"""
    
    def __init__(self):
        self.audio_manager = None
        self.stt_service = None
        self.tts_service = None
        self.monitor = None
        self.recording = False
        
    def process_text(self, text: str) -> str:
        """Simple text processing - just echo back what was said"""
        logger.info(f"Processing text: {text}")
        
        # Simple responses for testing
        if "hello" in text.lower():
            return "Hello! I'm your voice assistant. I can hear you clearly and now I can talk back to you!"
        elif "how are you" in text.lower():
            return "I'm doing well, thank you for asking! The voice pipeline is working perfectly."
        elif "goodbye" in text.lower() or "bye" in text.lower():
            return "Goodbye! It was nice talking with you."
        elif "test" in text.lower():
            return "Test successful! I can now hear you and speak back to you with full voice responses."
        else:
            return f"I heard you say: {text}. This is my voice response back to you!"
    
    async def initialize(self):
        """Initialize the voice assistant components"""
        logger.info("Initializing improved voice assistant...")
        
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
            
            # Initialize STT service
            stt_config = STTConfig(
                model="whisper-1",
                language="en"
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
            
            logger.info("Improved voice assistant initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize voice assistant: {e}")
            return False
    
    async def record_audio_until_silence(self, max_duration: float = 30.0):
        """Record audio until user stops speaking or max duration reached"""
        logger.info(f"Recording until silence (max {max_duration} seconds)...")
        logger.info("🎤 Speak now! Recording will stop when you're done...")
        
        # Start recording
        if not self.audio_manager.start_recording():
            logger.error("Failed to start recording")
            return None
        
        # Collect audio data
        audio_chunks = []
        start_time = time.time()
        silence_threshold = 0.001  # Adjust based on your microphone
        silence_duration = 0.0
        silence_limit = 2.0  # Stop after 2 seconds of silence
        
        while time.time() - start_time < max_duration:
            chunk = self.audio_manager.read_audio_chunk()
            if chunk:
                audio_chunks.append(chunk)
                
                # Simple silence detection (very basic)
                import numpy as np
                chunk_array = np.frombuffer(chunk, dtype=np.int16)
                volume = np.sqrt(np.mean(chunk_array**2))
                
                if volume < silence_threshold:
                    silence_duration += 0.01  # Roughly chunk duration
                    if silence_duration >= silence_limit:
                        logger.info("Silence detected - stopping recording")
                        break
                else:
                    silence_duration = 0.0  # Reset silence counter
                    
            await asyncio.sleep(0.01)  # Small delay
        
        # Stop recording
        self.audio_manager.stop_recording()
        
        # Combine audio chunks
        if audio_chunks:
            audio_data = b''.join(audio_chunks)
            duration = time.time() - start_time
            logger.info(f"Recorded {len(audio_data)} bytes of audio in {duration:.2f} seconds")
            return audio_data
        else:
            logger.warning("No audio data recorded")
            return None
    
    async def record_audio_timed(self, duration: float):
        """Record audio for a specific duration"""
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
    
    async def play_audio_response(self, audio_data: bytes):
        """Actually play the audio response through speakers"""
        try:
            logger.info("🔊 Playing voice response...")
            
            # Convert to playable format and play
            import io
            import wave
            import numpy as np
            
            # Create a temporary WAV file in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(24000)  # 24kHz (TTS output rate)
                wav_file.writeframes(audio_data)
            
            # Reset buffer position
            wav_buffer.seek(0)
            
            # Play using sounddevice
            import sounddevice as sd
            
            # Read the WAV data
            with wave.open(wav_buffer, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())
                audio_array = np.frombuffer(frames, dtype=np.int16)
                
                # Convert to float and normalize
                audio_float = audio_array.astype(np.float32) / 32767.0
                
                # Play the audio
                sd.play(audio_float, sample_rate)
                sd.wait()  # Wait until playback is finished
                
            logger.info("✅ Voice response played successfully!")
            
        except Exception as e:
            logger.error(f"Failed to play audio response: {e}")
            logger.info("Falling back to text-only response")
    
    async def process_voice_interaction(self, recording_mode: str = "auto"):
        """Process a single voice interaction"""
        try:
            # Record audio based on mode
            if recording_mode == "auto":
                audio_data = await self.record_audio_until_silence()
            elif recording_mode == "timed":
                duration = float(input("Enter recording duration in seconds (or press Enter for 10s): ") or "10")
                audio_data = await self.record_audio_timed(duration)
            else:
                # Default to 5 seconds
                audio_data = await self.record_audio_timed(5.0)
                
            if not audio_data:
                return
            
            # Convert speech to text
            logger.info("Converting speech to text...")
            with self.monitor.timing_context(PipelineStage.STT_PROCESSING):
                stt_result = await self.stt_service.transcribe_audio(audio_data)
            
            if not stt_result or not stt_result.text.strip():
                logger.warning("No speech detected or transcription failed")
                return
            
            logger.info(f"Transcribed: '{stt_result.text}'")
            
            # Process the text
            response_text = self.process_text(stt_result.text)
            logger.info(f"Response: '{response_text}'")
            
            # Convert response to speech
            logger.info("Converting response to speech...")
            with self.monitor.timing_context(PipelineStage.TTS_GENERATION):
                tts_result = await self.tts_service.synthesize_speech(response_text)
            
            if not tts_result:
                logger.error("Failed to synthesize speech")
                return
            
            # Play the response (ACTUALLY PLAY IT!)
            logger.info("Playing response...")
            with self.monitor.timing_context(PipelineStage.AUDIO_PLAYBACK):
                wav_audio = self.tts_service.get_wav_audio(tts_result)
                if wav_audio:
                    await self.play_audio_response(wav_audio)
                else:
                    logger.error("Failed to get WAV audio")
                    
        except Exception as e:
            logger.error(f"Error in voice interaction: {e}")
    
    async def run(self):
        """Run the improved voice assistant"""
        if not await self.initialize():
            return
        
        try:
            logger.info("🎤 Improved Voice Assistant is ready!")
            logger.info("🔊 Full voice conversation capability enabled")
            logger.info("📊 Performance monitoring enabled")
            logger.info("⏹️  Press Ctrl+C to exit")
            
            # Ask user for recording preference
            print("\nRecording options:")
            print("1. Auto-stop (records until you stop speaking)")
            print("2. Timed (you specify duration)")
            print("3. Fixed 5s (classic mode)")
            
            while True:
                try:
                    choice = input("\nChoose recording mode (1/2/3) or 'q' to quit: ").strip()
                    
                    if choice.lower() == 'q':
                        break
                    
                    recording_mode = {"1": "auto", "2": "timed", "3": "fixed"}.get(choice, "auto")
                    
                    print(f"\nUsing {recording_mode} recording mode...")
                    input("Press Enter to start recording...")
                    
                    # Process voice interaction
                    await self.process_voice_interaction(recording_mode)
                    
                    # Show performance stats
                    stats = self.monitor.get_summary()
                    if stats:
                        logger.info(f"Performance: {stats}")
                    
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
    assistant = ImprovedVoiceAssistant()
    asyncio.run(assistant.run())

if __name__ == "__main__":
    main() 