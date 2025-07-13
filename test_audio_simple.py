#!/usr/bin/env python3
"""
Simple test to debug audio recording issues
"""

import time
import logging
from assistant.audio import AudioManager, AudioConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_audio_recording():
    """Test basic audio recording functionality"""
    print("🎤 Testing Audio Recording...")
    
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
    
    print("🟢 Recording for 3 seconds - make some noise!")
    
    # Record for 3 seconds
    audio_chunks = []
    start_time = time.time()
    chunks_read = 0
    
    while time.time() - start_time < 3.0:
        chunk = audio_manager.read_audio_chunk()
        if chunk:
            audio_chunks.append(chunk)
            chunks_read += 1
            if chunks_read % 10 == 0:  # Log every 10 chunks
                print(f"📊 Read {chunks_read} chunks so far...")
        else:
            print("⚠️  Got None chunk")
        
        time.sleep(0.01)  # Small delay
    
    # Stop recording
    audio_manager.stop_recording()
    
    # Results
    total_data = b''.join(audio_chunks) if audio_chunks else b''
    print(f"✅ Recording complete!")
    print(f"📊 Total chunks: {len(audio_chunks)}")
    print(f"📊 Total bytes: {len(total_data)}")
    print(f"📊 Chunks read: {chunks_read}")
    
    # Calculate expected data
    expected_bytes = 3.0 * 16000 * 2  # 3 seconds * 16kHz * 2 bytes per sample
    print(f"📊 Expected bytes: {expected_bytes}")
    
    # Clean up
    audio_manager.cleanup()
    
    if len(total_data) > 0:
        print("🎉 Audio recording working!")
        return True
    else:
        print("❌ No audio data recorded")
        return False

if __name__ == "__main__":
    test_audio_recording() 