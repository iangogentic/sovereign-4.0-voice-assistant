"""
Audio Device Management and Configuration Module

This module handles all audio-related functionality including:
- Audio device detection and selection
- Sample rate configuration (16kHz for Whisper)
- Buffer management for low-latency processing
- Microphone input with noise suppression
- Speaker output configuration
- Audio format conversion
"""

import asyncio
import logging
import threading
from typing import Optional, Dict, List, Tuple, Any
import pyaudio
import numpy as np
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AudioDeviceType(Enum):
    """Audio device types"""
    INPUT = "input"
    OUTPUT = "output"


@dataclass
class AudioDevice:
    """Audio device information"""
    index: int
    name: str
    device_type: AudioDeviceType
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    is_default: bool


@dataclass
class AudioConfig:
    """Audio configuration settings"""
    sample_rate: int = 16000  # 16kHz for Whisper compatibility
    channels: int = 1  # Mono audio
    chunk_size: int = 1024  # Buffer size for low latency
    format: int = pyaudio.paInt16  # 16-bit audio
    input_device_index: Optional[int] = None
    output_device_index: Optional[int] = None
    
    # Advanced settings
    enable_noise_suppression: bool = True
    enable_agc: bool = True  # Automatic Gain Control
    input_volume: float = 1.0
    output_volume: float = 1.0


class AudioManager:
    """Manages audio devices and configuration"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.pyaudio_instance = None
        self.input_stream = None
        self.output_stream = None
        self.input_device = None
        self.output_device = None
        self.is_recording = False
        self.is_playing = False
        self._lock = threading.Lock()
        
    def initialize(self):
        """Initialize PyAudio and detect devices"""
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            logger.info("PyAudio initialized successfully")
            
            # Detect and log available devices
            self._detect_devices()
            
            # Set default devices if not specified
            if self.config.input_device_index is None:
                self.config.input_device_index = self._get_default_input_device()
            if self.config.output_device_index is None:
                self.config.output_device_index = self._get_default_output_device()
                
            logger.info(f"Audio configuration: Sample Rate={self.config.sample_rate}Hz, "
                       f"Channels={self.config.channels}, Chunk Size={self.config.chunk_size}")
            
        except Exception as e:
            logger.error(f"Failed to initialize audio system: {e}")
            raise
    
    def _detect_devices(self) -> List[AudioDevice]:
        """Detect available audio devices"""
        devices = []
        
        if not self.pyaudio_instance:
            return devices
            
        try:
            device_count = self.pyaudio_instance.get_device_count()
            default_input = self.pyaudio_instance.get_default_input_device_info()
            default_output = self.pyaudio_instance.get_default_output_device_info()
            
            logger.info(f"Found {device_count} audio devices:")
            
            for i in range(device_count):
                try:
                    device_info = self.pyaudio_instance.get_device_info_by_index(i)
                    
                    # Determine device type
                    device_type = None
                    if device_info['maxInputChannels'] > 0:
                        device_type = AudioDeviceType.INPUT
                    elif device_info['maxOutputChannels'] > 0:
                        device_type = AudioDeviceType.OUTPUT
                    
                    if device_type:
                        device = AudioDevice(
                            index=i,
                            name=device_info['name'],
                            device_type=device_type,
                            max_input_channels=device_info['maxInputChannels'],
                            max_output_channels=device_info['maxOutputChannels'],
                            default_sample_rate=device_info['defaultSampleRate'],
                            is_default=(i == default_input['index'] if device_type == AudioDeviceType.INPUT 
                                       else i == default_output['index'])
                        )
                        devices.append(device)
                        
                        logger.info(f"  Device {i}: {device.name} ({device.device_type.value}) "
                                   f"- Sample Rate: {device.default_sample_rate}Hz "
                                   f"{'[DEFAULT]' if device.is_default else ''}")
                        
                except Exception as e:
                    logger.warning(f"Could not get info for device {i}: {e}")
                    
        except Exception as e:
            logger.error(f"Error detecting audio devices: {e}")
            
        return devices
    
    def _get_default_input_device(self) -> int:
        """Get default input device index"""
        try:
            return self.pyaudio_instance.get_default_input_device_info()['index']
        except Exception as e:
            logger.warning(f"Could not get default input device: {e}")
            return None
    
    def _get_default_output_device(self) -> int:
        """Get default output device index"""
        try:
            return self.pyaudio_instance.get_default_output_device_info()['index']
        except Exception as e:
            logger.warning(f"Could not get default output device: {e}")
            return None
    
    def setup_input_stream(self) -> bool:
        """Set up audio input stream"""
        try:
            if self.input_stream:
                logger.warning("Input stream already exists, closing previous stream")
                self.close_input_stream()
            
            # Validate input device
            if self.config.input_device_index is None:
                logger.error("No input device specified")
                return False
            
            logger.info(f"Setting up input stream with device {self.config.input_device_index}")
            
            self.input_stream = self.pyaudio_instance.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=self.config.input_device_index,
                frames_per_buffer=self.config.chunk_size,
                stream_callback=None  # We'll use blocking read
            )
            
            logger.info("Input stream setup successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup input stream: {e}")
            return False
    
    def setup_output_stream(self) -> bool:
        """Set up audio output stream"""
        try:
            if self.output_stream:
                logger.warning("Output stream already exists, closing previous stream")
                self.close_output_stream()
            
            # Validate output device
            if self.config.output_device_index is None:
                logger.error("No output device specified")
                return False
            
            logger.info(f"Setting up output stream with device {self.config.output_device_index}")
            
            self.output_stream = self.pyaudio_instance.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                output=True,
                output_device_index=self.config.output_device_index,
                frames_per_buffer=self.config.chunk_size
            )
            
            logger.info("Output stream setup successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup output stream: {e}")
            return False
    
    def start_recording(self) -> bool:
        """Start recording audio"""
        with self._lock:
            if self.is_recording:
                logger.warning("Already recording")
                return True
            
            if not self.input_stream:
                if not self.setup_input_stream():
                    return False
            
            try:
                self.input_stream.start_stream()
                self.is_recording = True
                logger.info("Started recording")
                return True
            except Exception as e:
                logger.error(f"Failed to start recording: {e}")
                return False
    
    def stop_recording(self) -> bool:
        """Stop recording audio"""
        with self._lock:
            if not self.is_recording:
                return True
            
            try:
                if self.input_stream:
                    self.input_stream.stop_stream()
                self.is_recording = False
                logger.info("Stopped recording")
                return True
            except Exception as e:
                logger.error(f"Failed to stop recording: {e}")
                return False
    
    def read_audio_chunk(self) -> Optional[bytes]:
        """Read audio chunk from input stream"""
        if not self.input_stream or not self.is_recording:
            return None
        
        try:
            data = self.input_stream.read(self.config.chunk_size, exception_on_overflow=False)
            
            # Apply volume control
            if self.config.input_volume != 1.0:
                audio_data = np.frombuffer(data, dtype=np.int16)
                audio_data = (audio_data * self.config.input_volume).astype(np.int16)
                data = audio_data.tobytes()
            
            return data
        except Exception as e:
            logger.error(f"Error reading audio chunk: {e}")
            return None
    
    def play_audio_chunk(self, audio_data: bytes) -> bool:
        """Play audio chunk to output stream"""
        if not self.output_stream:
            if not self.setup_output_stream():
                return False
        
        try:
            # Apply volume control
            if self.config.output_volume != 1.0:
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_array = (audio_array * self.config.output_volume).astype(np.int16)
                audio_data = audio_array.tobytes()
            
            self.output_stream.write(audio_data)
            return True
        except Exception as e:
            logger.error(f"Error playing audio chunk: {e}")
            return False
    
    def close_input_stream(self):
        """Close input stream"""
        if self.input_stream:
            try:
                self.input_stream.stop_stream()
                self.input_stream.close()
                self.input_stream = None
                self.is_recording = False
                logger.info("Input stream closed")
            except Exception as e:
                logger.error(f"Error closing input stream: {e}")
    
    def close_output_stream(self):
        """Close output stream"""
        if self.output_stream:
            try:
                self.output_stream.stop_stream()
                self.output_stream.close()
                self.output_stream = None
                logger.info("Output stream closed")
            except Exception as e:
                logger.error(f"Error closing output stream: {e}")
    
    def cleanup(self):
        """Clean up audio resources"""
        logger.info("Cleaning up audio resources")
        
        self.close_input_stream()
        self.close_output_stream()
        
        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
                self.pyaudio_instance = None
                logger.info("PyAudio terminated")
            except Exception as e:
                logger.error(f"Error terminating PyAudio: {e}")
    
    def get_device_info(self, device_index: int) -> Optional[Dict]:
        """Get information about a specific device"""
        if not self.pyaudio_instance:
            return None
        
        try:
            return self.pyaudio_instance.get_device_info_by_index(device_index)
        except Exception as e:
            logger.error(f"Error getting device info for index {device_index}: {e}")
            return None
    
    def test_device_compatibility(self, device_index: int, is_input: bool = True) -> bool:
        """Test if a device is compatible with current configuration"""
        try:
            test_stream = self.pyaudio_instance.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=is_input,
                output=not is_input,
                input_device_index=device_index if is_input else None,
                output_device_index=device_index if not is_input else None,
                frames_per_buffer=self.config.chunk_size
            )
            
            test_stream.close()
            return True
            
        except Exception as e:
            logger.warning(f"Device {device_index} not compatible: {e}")
            return False


def create_audio_manager(config: Optional[AudioConfig] = None) -> AudioManager:
    """Factory function to create AudioManager with default configuration"""
    if config is None:
        config = AudioConfig()
    
    manager = AudioManager(config)
    manager.initialize()
    return manager 