#!/usr/bin/env python3
"""
Audio System Diagnostic Script
Tests each component of the voice assistant to identify issues
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

# Import our components
from assistant.audio import AudioManager, AudioConfig
from assistant.stt import WhisperSTTService, STTConfig
from assistant.tts import OpenAITTSService, TTSConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioDiagnostic:
    """Comprehensive audio system diagnostic"""
    
    def __init__(self):
        self.audio_manager = None
        self.stt_service = None
        self.tts_service = None
        
    def test_basic_audio_devices(self):
        """Test 1: Basic audio device detection"""
        print("\n🔧 TEST 1: Audio Device Detection")
        print("=" * 50)
        
        try:
            devices = sd.query_devices()
            print(f"Found {len(devices)} audio devices:")
            
            for i, device in enumerate(devices):
                device_type = []
                if device['max_input_channels'] > 0:
                    device_type.append("input")
                if device['max_output_channels'] > 0:
                    device_type.append("output")
                
                print(f"  Device {i}: {device['name']} ({'/'.join(device_type)}) - {device['default_samplerate']}Hz")
            
            # Test default devices
            default_input = sd.query_devices(kind='input')
            default_output = sd.query_devices(kind='output')
            
            print(f"\nDefault Input: {default_input['name']}")
            print(f"Default Output: {default_output['name']}")
            
            print("✅ Audio device detection: PASSED")
            return True
            
        except Exception as e:
            print(f"❌ Audio device detection: FAILED - {e}")
            return False
    
    def test_microphone_recording(self):
        """Test 2: Microphone recording"""
        print("\n🔧 TEST 2: Microphone Recording")
        print("=" * 50)
        
        try:
            duration = 3.0
            sample_rate = 16000
            
            print(f"Recording for {duration} seconds...")
            print("🎤 Say something now!")
            
            # Record audio
            audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
            sd.wait()
            
            # Analyze the recording
            audio_flat = audio_data.flatten()
            max_amplitude = np.max(np.abs(audio_flat))
            rms_level = np.sqrt(np.mean(audio_flat.astype(np.float32) ** 2))
            
            print(f"Recording completed:")
            print(f"  Samples recorded: {len(audio_flat)}")
            print(f"  Max amplitude: {max_amplitude}")
            print(f"  RMS level: {rms_level:.2f}")
            
            if max_amplitude > 100:  # Reasonable threshold for speech
                print("✅ Microphone recording: PASSED - Audio detected")
                return True, audio_data
            else:
                print("⚠️  Microphone recording: WEAK - Very low audio level")
                return False, audio_data
                
        except Exception as e:
            print(f"❌ Microphone recording: FAILED - {e}")
            return False, None
    
    def test_speaker_playback(self):
        """Test 3: Speaker playback"""
        print("\n🔧 TEST 3: Speaker Playback")
        print("=" * 50)
        
        try:
            # Generate test tone
            duration = 2.0
            sample_rate = 24000
            frequency = 440  # A4 note
            
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = np.sin(2 * np.pi * frequency * t) * 0.3
            
            print(f"Playing {frequency}Hz test tone for {duration} seconds...")
            print("🔊 You should hear a tone now!")
            
            # Play the tone
            sd.play(tone, sample_rate)
            sd.wait()
            
            print("✅ Speaker playback: COMPLETED")
            return True
            
        except Exception as e:
            print(f"❌ Speaker playback: FAILED - {e}")
            return False
    
    async def test_audio_manager(self):
        """Test 4: Audio Manager"""
        print("\n🔧 TEST 4: Audio Manager")
        print("=" * 50)
        
        try:
            # Initialize audio manager
            audio_config = AudioConfig(sample_rate=16000, chunk_size=1024, channels=1)
            self.audio_manager = AudioManager(config=audio_config)
            
            print("Initializing audio manager...")
            self.audio_manager.initialize()
            
            print("Setting up audio streams...")
            self.audio_manager.setup_input_stream()
            self.audio_manager.setup_output_stream()
            
            print("Testing audio recording...")
            if not self.audio_manager.start_recording():
                print("❌ Audio Manager: FAILED - Could not start recording")
                return False
            
            # Record for 3 seconds
            print("🎤 Recording for 3 seconds... speak now!")
            audio_chunks = []
            start_time = time.time()
            
            while time.time() - start_time < 3.0:
                chunk = self.audio_manager.read_audio_chunk()
                if chunk:
                    audio_chunks.append(chunk)
                await asyncio.sleep(0.01)
            
            self.audio_manager.stop_recording()
            
            if audio_chunks:
                total_audio = b''.join(audio_chunks)
                print(f"Recorded {len(total_audio)} bytes of audio")
                print("✅ Audio Manager: PASSED")
                return True, total_audio
            else:
                print("❌ Audio Manager: FAILED - No audio chunks recorded")
                return False, None
                
        except Exception as e:
            print(f"❌ Audio Manager: FAILED - {e}")
            return False, None
    
    async def test_stt_service(self, audio_data):
        """Test 5: Speech-to-Text"""
        print("\n🔧 TEST 5: Speech-to-Text Service")
        print("=" * 50)
        
        try:
            if not os.getenv("OPENAI_API_KEY"):
                print("❌ STT Service: FAILED - No OpenAI API key")
                return False
            
            # Initialize STT service
            stt_config = STTConfig(model="whisper-1", language="en")
            self.stt_service = WhisperSTTService(
                config=stt_config,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            
            print("Initializing STT service...")
            self.stt_service.initialize()
            
            print("Testing transcription...")
            result = await self.stt_service.transcribe_audio(audio_data)
            
            if result and result.text:
                print(f"Transcription: '{result.text}'")
                print(f"Confidence: {result.confidence}")
                print("✅ STT Service: PASSED")
                return True
            else:
                print("❌ STT Service: FAILED - No transcription result")
                return False
                
        except Exception as e:
            print(f"❌ STT Service: FAILED - {e}")
            return False
    
    async def test_tts_service(self):
        """Test 6: Text-to-Speech"""
        print("\n🔧 TEST 6: Text-to-Speech Service")
        print("=" * 50)
        
        try:
            if not os.getenv("OPENAI_API_KEY"):
                print("❌ TTS Service: FAILED - No OpenAI API key")
                return False
            
            # Initialize TTS service
            tts_config = TTSConfig(model="tts-1", voice="alloy", speed=1.0)
            self.tts_service = OpenAITTSService(
                config=tts_config,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            
            print("Initializing TTS service...")
            self.tts_service.initialize()
            
            print("Testing speech synthesis...")
            test_text = "Hello, this is a test of the text to speech system."
            result = await self.tts_service.synthesize_speech(test_text)
            
            if result:
                print("Getting audio data...")
                wav_audio = self.tts_service.get_wav_audio(result)
                
                if wav_audio:
                    print(f"Generated {len(wav_audio)} bytes of audio")
                    
                    # Play the audio
                    print("🔊 Playing generated speech...")
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
                        
                        sd.play(audio_float, 24000)
                        sd.wait()
                    
                    print("✅ TTS Service: PASSED")
                    return True
                else:
                    print("❌ TTS Service: FAILED - Could not get audio data")
                    return False
            else:
                print("❌ TTS Service: FAILED - No synthesis result")
                return False
                
        except Exception as e:
            print(f"❌ TTS Service: FAILED - {e}")
            return False
    
    async def run_full_diagnostic(self):
        """Run complete diagnostic suite"""
        print("🔍 VOICE ASSISTANT DIAGNOSTIC SUITE")
        print("=" * 60)
        
        results = {}
        
        # Test 1: Audio devices
        results['devices'] = self.test_basic_audio_devices()
        
        # Test 2: Microphone
        results['microphone'], recorded_audio = self.test_microphone_recording()
        
        # Test 3: Speakers
        results['speakers'] = self.test_speaker_playback()
        
        # Test 4: Audio Manager
        results['audio_manager'], manager_audio = await self.test_audio_manager()
        
        # Test 5: STT (use manager audio if available, otherwise recorded audio)
        test_audio = manager_audio if manager_audio else recorded_audio
        if test_audio:
            results['stt'] = await self.test_stt_service(test_audio)
        else:
            results['stt'] = False
        
        # Test 6: TTS
        results['tts'] = await self.test_tts_service()
        
        # Summary
        print("\n🏁 DIAGNOSTIC SUMMARY")
        print("=" * 60)
        
        for test_name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{test_name.upper()}: {status}")
        
        total_tests = len(results)
        passed_tests = sum(results.values())
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 All systems operational!")
        else:
            print("⚠️  Some systems need attention")
        
        # Cleanup
        if self.audio_manager:
            self.audio_manager.cleanup()

def main():
    """Run the diagnostic"""
    diagnostic = AudioDiagnostic()
    asyncio.run(diagnostic.run_full_diagnostic())

if __name__ == "__main__":
    main() 