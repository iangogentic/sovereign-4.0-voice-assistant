#!/usr/bin/env python3
"""
Test to analyze recorded audio content
"""

import time
import numpy as np
import wave
import tempfile
import os
from assistant.audio import AudioManager, AudioConfig

def test_audio_content():
    """Test audio recording and analyze content"""
    print("🎤 Testing Audio Content Analysis...")
    
    # Create audio manager
    config = AudioConfig(
        sample_rate=16000,
        chunk_size=1024,
        channels=1
    )
    
    audio_manager = AudioManager(config)
    
    # Initialize
    try:
        audio_manager.initialize()
        print("✅ Audio manager initialized")
    except Exception as e:
        print(f"❌ Failed to initialize audio manager: {e}")
        return False
    
    # Setup input stream
    if not audio_manager.setup_input_stream():
        print("❌ Failed to setup input stream")
        return False
    
    # Start recording
    if not audio_manager.start_recording():
        print("❌ Failed to start recording")
        return False
    
    print("🟢 Recording for 3 seconds - SPEAK NOW!")
    print("   (Say something loud and clear)")
    
    # Record for 3 seconds
    audio_chunks = []
    start_time = time.time()
    
    while time.time() - start_time < 3.0:
        chunk = audio_manager.read_audio_chunk()
        if chunk:
            audio_chunks.append(chunk)
        time.sleep(0.01)
    
    # Stop recording
    audio_manager.stop_recording()
    
    # Analyze audio content
    if not audio_chunks:
        print("❌ No audio data recorded")
        return False
    
    # Combine audio data
    audio_data = b''.join(audio_chunks)
    print(f"📊 Total audio data: {len(audio_data)} bytes")
    
    # Convert to numpy array for analysis
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    print(f"📊 Audio samples: {len(audio_array)}")
    
    # Calculate audio statistics
    max_amplitude = np.max(np.abs(audio_array))
    rms_amplitude = np.sqrt(np.mean(audio_array.astype(np.float32)**2))
    
    print(f"📊 Max amplitude: {max_amplitude} (out of 32767)")
    print(f"📊 RMS amplitude: {rms_amplitude:.2f}")
    
    # Check if audio contains significant sound
    if max_amplitude < 100:
        print("⚠️  Audio seems very quiet or silent")
        print("   - Check your microphone settings")
        print("   - Make sure microphone is not muted")
        print("   - Try speaking louder")
    elif max_amplitude < 1000:
        print("⚠️  Audio is quite quiet")
        print("   - Try speaking louder or closer to microphone")
    else:
        print("✅ Audio contains significant sound!")
    
    # Save audio to file for testing
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_filename = f.name
        
        # Write WAV file
        with wave.open(temp_filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)  # 16kHz
            wav_file.writeframes(audio_data)
        
        print(f"💾 Audio saved to: {temp_filename}")
        print("   You can play this file to hear what was recorded")
        
        # Clean up
        os.unlink(temp_filename)
        
    except Exception as e:
        print(f"❌ Failed to save audio file: {e}")
    
    # Clean up
    audio_manager.cleanup()
    
    return max_amplitude > 100

if __name__ == "__main__":
    test_audio_content() 