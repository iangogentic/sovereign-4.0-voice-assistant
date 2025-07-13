#!/usr/bin/env python3
"""
Voice Activity Detection Test
"""

import asyncio
import logging
import os
import time
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from assistant.audio import AudioManager, AudioConfig
from assistant.stt import WhisperSTTService, STTConfig

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,  # More verbose logging
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_vad_settings():
    """Test different VAD settings"""
    print("🔍 Testing Voice Activity Detection Settings")
    print("=" * 50)
    
    # Initialize audio manager
    audio_config = AudioConfig(sample_rate=16000, chunk_size=1024, channels=1)
    audio_manager = AudioManager(config=audio_config)
    audio_manager.initialize()
    audio_manager.setup_input_stream()
    
    # Record audio
    print("🎤 Recording 3 seconds of audio...")
    if not audio_manager.start_recording():
        print("❌ Failed to start recording")
        return
    
    audio_chunks = []
    start_time = time.time()
    
    while time.time() - start_time < 3.0:
        chunk = audio_manager.read_audio_chunk()
        if chunk:
            audio_chunks.append(chunk)
        await asyncio.sleep(0.01)
    
    audio_manager.stop_recording()
    
    if not audio_chunks:
        print("❌ No audio recorded")
        return
    
    total_audio = b''.join(audio_chunks)
    print(f"✅ Recorded {len(total_audio)} bytes of audio")
    
    # Test different VAD thresholds
    thresholds = [0.001, 0.01, 0.05, 0.1, 0.2]
    
    for threshold in thresholds:
        print(f"\n🔬 Testing VAD threshold: {threshold}")
        
        # Create STT config with this threshold
        stt_config = STTConfig(
            model="whisper-1",
            language="en",
            silence_threshold=threshold,
            vad_enabled=True,
            min_audio_length=0.1  # Lower minimum for testing
        )
        
        # Initialize STT service
        stt_service = WhisperSTTService(
            config=stt_config,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        stt_service.initialize()
        
        # Test voice activity detection
        has_voice = stt_service.audio_processor.detect_voice_activity(total_audio)
        
        # Calculate audio statistics
        audio_array = np.frombuffer(total_audio, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
        normalized_rms = rms / 32768.0
        max_amplitude = np.max(np.abs(audio_array))
        
        print(f"   Audio RMS: {normalized_rms:.4f}")
        print(f"   Max amplitude: {max_amplitude}")
        print(f"   Voice detected: {has_voice}")
        
        if has_voice:
            print("   ✅ This threshold would allow transcription")
            
            # Try actual transcription
            result = await stt_service.transcribe_audio(total_audio)
            if result:
                print(f"   🎯 Transcription: '{result.text}'")
                print(f"   📊 Confidence: {result.confidence}")
                break
            else:
                print("   ❌ Transcription failed despite VAD pass")
        else:
            print("   ❌ This threshold would reject the audio")
    
    # Cleanup
    audio_manager.cleanup()

def main():
    """Run the VAD test"""
    asyncio.run(test_vad_settings())

if __name__ == "__main__":
    main() 