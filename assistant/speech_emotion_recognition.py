"""
Speech Emotion Recognition System for Sovereign 4.0 Voice Assistant

Implements real-time emotion detection using SpeechBrain's wav2vec2-based models with
advanced features for voice emotion preservation and consistency tracking.

Key Features:
- Real-time emotion recognition using speechbrain/emotion-recognition-wav2vec2-IEMOCAP
- Sliding window processing with temporal smoothing
- Multi-modal feature extraction using librosa
- Emotion trajectory tracking for consistency
- Performance optimization with dynamic quantization
- Integration with EmotionPreserver system

Usage:
    emotion_recognizer = SpeechEmotionRecognizer()
    await emotion_recognizer.initialize()
    
    # Process audio chunk
    emotion_result = await emotion_recognizer.process_audio_chunk(audio_data)
    print(f"Detected emotion: {emotion_result.primary_emotion}")
"""

import asyncio
import logging
import time
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import torch
import io
from pathlib import Path

logger = logging.getLogger(__name__)


class EmotionLabel(Enum):
    """Standard emotion labels for classification"""
    ANGRY = "angry"
    HAPPY = "happy"
    SAD = "sad"
    NEUTRAL = "neutral"
    FEAR = "fear"
    DISGUST = "disgust"
    SURPRISE = "surprise"


class ProcessingMode(Enum):
    """Processing modes for emotion recognition"""
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"


@dataclass
class EmotionResult:
    """Result of emotion recognition analysis"""
    primary_emotion: EmotionLabel
    confidence: float
    emotion_scores: Dict[EmotionLabel, float]
    arousal: float  # 0.0 (calm) to 1.0 (excited)
    valence: float  # 0.0 (negative) to 1.0 (positive)
    processing_time_ms: float
    audio_quality_score: float
    features: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class AudioFeatures:
    """Extracted audio features for emotion analysis"""
    f0_mean: float  # Fundamental frequency mean
    f0_std: float   # Fundamental frequency standard deviation
    energy_mean: float  # RMS energy mean
    energy_std: float   # RMS energy standard deviation
    spectral_centroid_mean: float
    spectral_rolloff_mean: float
    zero_crossing_rate_mean: float
    mfcc: np.ndarray  # MFCC features
    chroma: np.ndarray  # Chroma features
    tempo: float
    spectral_contrast: np.ndarray


@dataclass
class EmotionRecognitionConfig:
    """Configuration for speech emotion recognition"""
    # Model settings
    model_name: str = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
    device: str = "auto"  # auto, cpu, cuda
    
    # Audio processing
    sample_rate: int = 16000  # Required by wav2vec2
    chunk_duration_ms: int = 2000  # 2 second chunks
    overlap_ratio: float = 0.5  # 50% overlap between chunks
    min_chunk_duration_ms: int = 1000  # Minimum chunk size
    
    # Processing settings
    processing_mode: ProcessingMode = ProcessingMode.REAL_TIME
    batch_size: int = 1
    max_buffer_size: int = 160000  # 10 seconds at 16kHz
    
    # Confidence and filtering
    min_confidence_threshold: float = 0.6
    emotion_smoothing_factor: float = 0.7
    enable_temporal_smoothing: bool = True
    
    # Feature extraction
    enable_multi_modal_features: bool = True
    n_mfcc: int = 13
    n_chroma: int = 12
    
    # Performance optimization
    enable_quantization: bool = True
    enable_model_caching: bool = True
    max_cache_size: int = 100
    
    # Quality control
    min_audio_quality_score: float = 0.5
    noise_reduction_enabled: bool = True


class AudioQualityAnalyzer:
    """Analyze audio quality for emotion recognition"""
    
    @staticmethod
    def calculate_quality_score(audio: np.ndarray, sample_rate: int) -> float:
        """Calculate overall audio quality score (0.0 to 1.0)"""
        try:
            # Signal-to-noise ratio estimation
            signal_power = np.mean(audio ** 2)
            if signal_power == 0:
                return 0.0
            
            # Calculate SNR approximation
            noise_floor = np.percentile(np.abs(audio), 10)
            signal_peak = np.percentile(np.abs(audio), 90)
            
            if noise_floor == 0:
                snr_estimate = 40  # Very clean signal
            else:
                snr_estimate = 20 * np.log10(signal_peak / noise_floor)
            
            # Normalize SNR to 0-1 range (assume 0-40 dB range)
            snr_score = np.clip(snr_estimate / 40.0, 0.0, 1.0)
            
            # Check for clipping
            clipping_ratio = np.sum(np.abs(audio) > 0.95 * np.max(np.abs(audio))) / len(audio)
            clipping_penalty = max(0, 1.0 - clipping_ratio * 10)
            
            # Dynamic range check
            dynamic_range = np.max(audio) - np.min(audio)
            range_score = min(1.0, dynamic_range / 0.5)  # Expect some dynamic range
            
            # Combine scores
            quality_score = (snr_score * 0.5 + clipping_penalty * 0.3 + range_score * 0.2)
            return float(np.clip(quality_score, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Audio quality analysis failed: {e}")
            return 0.5  # Default moderate quality


class FeatureExtractor:
    """Extract multi-modal audio features using librosa"""
    
    def __init__(self, config: EmotionRecognitionConfig):
        self.config = config
        self.sample_rate = config.sample_rate
        
    def extract_features(self, audio: np.ndarray) -> AudioFeatures:
        """Extract comprehensive audio features"""
        try:
            # Import librosa here to avoid dependency issues if not needed
            import librosa
            
            # Ensure audio is 1D
            if audio.ndim > 1:
                audio = np.mean(audio, axis=0)
            
            # Normalize audio
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            # Fundamental frequency (F0) analysis
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=80, fmax=400, sr=self.sample_rate, frame_length=2048
            )
            f0_clean = f0[voiced_flag]
            f0_mean = np.nanmean(f0_clean) if len(f0_clean) > 0 else 0.0
            f0_std = np.nanstd(f0_clean) if len(f0_clean) > 0 else 0.0
            
            # Energy features (RMS)
            rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
            energy_mean = np.mean(rms)
            energy_std = np.std(rms)
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
            
            # MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio, sr=self.sample_rate, n_mfcc=self.config.n_mfcc
            )
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
            
            # Tempo estimation
            tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            
            # Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
            
            return AudioFeatures(
                f0_mean=float(f0_mean),
                f0_std=float(f0_std),
                energy_mean=float(energy_mean),
                energy_std=float(energy_std),
                spectral_centroid_mean=float(np.mean(spectral_centroid)),
                spectral_rolloff_mean=float(np.mean(spectral_rolloff)),
                zero_crossing_rate_mean=float(np.mean(zero_crossing_rate)),
                mfcc=mfcc,
                chroma=chroma,
                tempo=float(tempo),
                spectral_contrast=spectral_contrast
            )
            
        except ImportError:
            logger.warning("librosa not available, using basic features")
            return self._extract_basic_features(audio)
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return self._extract_basic_features(audio)
    
    def _extract_basic_features(self, audio: np.ndarray) -> AudioFeatures:
        """Extract basic features without librosa"""
        # Basic energy calculation
        energy_mean = float(np.mean(audio ** 2))
        energy_std = float(np.std(audio ** 2))
        
        # Simple zero crossing rate
        zero_crossings = np.where(np.diff(np.signbit(audio)))[0]
        zcr = len(zero_crossings) / len(audio)
        
        # Basic spectral centroid approximation using FFT
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.fftfreq(len(fft), 1/self.sample_rate)
        spectral_centroid = float(np.sum(freqs[:len(freqs)//2] * magnitude[:len(magnitude)//2]) / 
                                 np.sum(magnitude[:len(magnitude)//2]))
        
        return AudioFeatures(
            f0_mean=0.0,
            f0_std=0.0,
            energy_mean=energy_mean,
            energy_std=energy_std,
            spectral_centroid_mean=abs(spectral_centroid),
            spectral_rolloff_mean=0.0,
            zero_crossing_rate_mean=float(zcr),
            mfcc=np.zeros((self.config.n_mfcc, 1)),
            chroma=np.zeros((self.config.n_chroma, 1)),
            tempo=120.0,  # Default tempo
            spectral_contrast=np.zeros((7, 1))
        )


class EmotionTrajectoryTracker:
    """Track emotion trajectories for temporal consistency"""
    
    def __init__(self, smoothing_factor: float = 0.7, history_length: int = 10):
        self.smoothing_factor = smoothing_factor
        self.emotion_history = deque(maxlen=history_length)
        self.confidence_history = deque(maxlen=history_length)
        self.previous_emotion = None
        
    def update_emotion(self, emotion_scores: Dict[EmotionLabel, float], confidence: float) -> Dict[EmotionLabel, float]:
        """Update emotion with temporal smoothing"""
        
        # Store current results
        self.emotion_history.append(emotion_scores.copy())
        self.confidence_history.append(confidence)
        
        # If this is the first emotion or high confidence, use as-is
        if self.previous_emotion is None or confidence > 0.8:
            self.previous_emotion = emotion_scores.copy()
            return emotion_scores
        
        # Apply temporal smoothing
        smoothed_scores = {}
        for emotion in EmotionLabel:
            current_score = emotion_scores.get(emotion, 0.0)
            previous_score = self.previous_emotion.get(emotion, 0.0)
            
            # Smooth based on confidence
            if confidence > 0.6:
                # Medium confidence - moderate smoothing
                smoothed_score = (self.smoothing_factor * previous_score + 
                                (1 - self.smoothing_factor) * current_score)
            else:
                # Low confidence - heavy smoothing
                smoothed_score = (0.9 * previous_score + 0.1 * current_score)
            
            smoothed_scores[emotion] = smoothed_score
        
        # Normalize scores
        total_score = sum(smoothed_scores.values())
        if total_score > 0:
            smoothed_scores = {k: v/total_score for k, v in smoothed_scores.items()}
        
        self.previous_emotion = smoothed_scores.copy()
        return smoothed_scores
    
    def get_emotion_trend(self) -> Optional[EmotionLabel]:
        """Get the trending emotion over recent history"""
        if len(self.emotion_history) < 3:
            return None
        
        # Calculate average scores over recent history
        emotion_averages = {}
        for emotion in EmotionLabel:
            scores = [hist.get(emotion, 0.0) for hist in self.emotion_history]
            emotion_averages[emotion] = np.mean(scores)
        
        # Return the most frequent emotion
        return max(emotion_averages, key=emotion_averages.get)


class SpeechEmotionRecognizer:
    """
    Main class for speech emotion recognition using SpeechBrain wav2vec2 models.
    
    Provides real-time emotion detection with temporal consistency and multi-modal
    feature extraction for enhanced accuracy.
    """
    
    def __init__(self, config: Optional[EmotionRecognitionConfig] = None):
        self.config = config or EmotionRecognitionConfig()
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.classifier = None
        self.is_initialized = False
        self.device = None
        
        # Processing components
        self.feature_extractor = FeatureExtractor(self.config)
        self.quality_analyzer = AudioQualityAnalyzer()
        self.trajectory_tracker = EmotionTrajectoryTracker(
            self.config.emotion_smoothing_factor
        )
        
        # Audio buffer for streaming
        self.audio_buffer = deque(maxlen=self.config.max_buffer_size)
        self.processing_lock = asyncio.Lock()
        
        # Performance metrics
        self.total_predictions = 0
        self.average_processing_time = 0.0
        self.last_prediction_time = 0.0
        
        # Emotion mapping for IEMOCAP model
        self.emotion_mapping = {
            0: EmotionLabel.ANGRY,
            1: EmotionLabel.HAPPY, 
            2: EmotionLabel.SAD,
            3: EmotionLabel.NEUTRAL
        }
        
    async def initialize(self) -> bool:
        """Initialize the emotion recognition system"""
        try:
            self.logger.info("Initializing SpeechEmotionRecognizer...")
            
            # Determine device
            if self.config.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = self.config.device
            
            self.logger.info(f"Using device: {self.device}")
            
            # Load speechbrain model
            await self._load_model()
            
            # Apply optimizations
            if self.config.enable_quantization and self.device == "cpu":
                await self._apply_quantization()
            
            self.is_initialized = True
            self.logger.info("SpeechEmotionRecognizer initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SpeechEmotionRecognizer: {e}")
            return False
    
    async def _load_model(self):
        """Load the SpeechBrain emotion recognition model"""
        try:
            # Import speechbrain here to handle optional dependency
            from speechbrain.pretrained import EncoderClassifier
            
            self.classifier = EncoderClassifier.from_hparams(
                source=self.config.model_name,
                run_opts={"device": self.device}
            )
            
        except ImportError:
            self.logger.error("SpeechBrain not installed. Please install with: pip install speechbrain")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load SpeechBrain model: {e}")
            raise
    
    async def _apply_quantization(self):
        """Apply dynamic quantization for performance optimization"""
        try:
            if hasattr(self.classifier, 'mods') and hasattr(self.classifier.mods, 'encoder'):
                # Apply quantization to encoder
                encoder = self.classifier.mods.encoder
                encoder = torch.quantization.quantize_dynamic(
                    encoder, {torch.nn.Linear}, dtype=torch.qint8
                )
                self.classifier.mods.encoder = encoder
                self.logger.info("Applied dynamic quantization to emotion recognition model")
                
        except Exception as e:
            self.logger.warning(f"Quantization failed, continuing without optimization: {e}")
    
    async def process_audio_chunk(self, audio_data: bytes) -> Optional[EmotionResult]:
        """
        Process audio chunk and return emotion recognition results.
        
        Args:
            audio_data: Raw audio bytes (PCM16, 16kHz expected)
            
        Returns:
            EmotionResult with detected emotion and confidence
        """
        if not self.is_initialized:
            self.logger.error("SpeechEmotionRecognizer not initialized")
            return None
        
        start_time = time.time()
        
        async with self.processing_lock:
            try:
                # Convert bytes to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                
                # Normalize to [-1, 1] range
                if np.max(np.abs(audio_array)) > 0:
                    audio_array = audio_array / 32768.0
                
                # Check audio quality
                quality_score = self.quality_analyzer.calculate_quality_score(
                    audio_array, self.config.sample_rate
                )
                
                if quality_score < self.config.min_audio_quality_score:
                    self.logger.warning(f"Low audio quality detected: {quality_score:.2f}")
                    return None
                
                # Add to buffer for streaming processing
                self.audio_buffer.extend(audio_array)
                
                # Process if we have enough audio
                if len(self.audio_buffer) >= self.config.min_chunk_duration_ms * self.config.sample_rate // 1000:
                    # Extract chunk for processing
                    chunk_size = self.config.chunk_duration_ms * self.config.sample_rate // 1000
                    chunk_size = min(chunk_size, len(self.audio_buffer))
                    
                    audio_chunk = np.array(list(self.audio_buffer)[-chunk_size:])
                    
                    # Run emotion recognition
                    emotion_result = await self._predict_emotion(audio_chunk, quality_score)
                    
                    # Update performance metrics
                    processing_time = (time.time() - start_time) * 1000
                    self.total_predictions += 1
                    self.average_processing_time = (
                        (self.average_processing_time * (self.total_predictions - 1) + processing_time) /
                        self.total_predictions
                    )
                    self.last_prediction_time = processing_time
                    
                    if emotion_result:
                        emotion_result.processing_time_ms = processing_time
                    
                    return emotion_result
                
                return None
                
            except Exception as e:
                self.logger.error(f"Audio processing failed: {e}")
                return None
    
    async def _predict_emotion(self, audio: np.ndarray, quality_score: float) -> Optional[EmotionResult]:
        """Predict emotion from audio chunk"""
        try:
            # Convert to tensor
            audio_tensor = torch.tensor(audio).float()
            
            # Ensure correct sample rate (resample if needed)
            if self.config.sample_rate != 16000:
                # Simple resampling - in production, use proper resampling
                target_length = int(len(audio) * 16000 / self.config.sample_rate)
                audio_tensor = torch.nn.functional.interpolate(
                    audio_tensor.unsqueeze(0).unsqueeze(0),
                    size=target_length,
                    mode='linear'
                ).squeeze()
            
            # Normalize
            if torch.max(torch.abs(audio_tensor)) > 0:
                audio_tensor = audio_tensor / torch.max(torch.abs(audio_tensor))
            
            # Predict emotion
            prediction = self.classifier.classify_batch(audio_tensor.unsqueeze(0))
            emotion_logits = prediction[0].squeeze()
            emotion_probs = torch.nn.functional.softmax(emotion_logits, dim=-1)
            
            # Convert to emotion scores dictionary
            emotion_scores = {}
            for idx, prob in enumerate(emotion_probs):
                if idx in self.emotion_mapping:
                    emotion_scores[self.emotion_mapping[idx]] = float(prob)
            
            # Get primary emotion and confidence
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[primary_emotion]
            
            # Apply temporal smoothing if enabled
            if self.config.enable_temporal_smoothing:
                smoothed_scores = self.trajectory_tracker.update_emotion(emotion_scores, confidence)
                primary_emotion = max(smoothed_scores, key=smoothed_scores.get)
                confidence = smoothed_scores[primary_emotion]
                emotion_scores = smoothed_scores
            
            # Calculate arousal and valence from emotion scores
            arousal, valence = self._calculate_arousal_valence(emotion_scores)
            
            # Extract additional features if enabled
            features = None
            if self.config.enable_multi_modal_features:
                try:
                    audio_features = self.feature_extractor.extract_features(audio)
                    features = {
                        'f0_mean': audio_features.f0_mean,
                        'f0_std': audio_features.f0_std,
                        'energy_mean': audio_features.energy_mean,
                        'spectral_centroid_mean': audio_features.spectral_centroid_mean,
                        'tempo': audio_features.tempo
                    }
                except Exception as e:
                    self.logger.warning(f"Feature extraction failed: {e}")
            
            return EmotionResult(
                primary_emotion=primary_emotion,
                confidence=confidence,
                emotion_scores=emotion_scores,
                arousal=arousal,
                valence=valence,
                processing_time_ms=0.0,  # Will be set by caller
                audio_quality_score=quality_score,
                features=features
            )
            
        except Exception as e:
            self.logger.error(f"Emotion prediction failed: {e}")
            return None
    
    def _calculate_arousal_valence(self, emotion_scores: Dict[EmotionLabel, float]) -> Tuple[float, float]:
        """Calculate arousal and valence from emotion scores"""
        # Simplified arousal-valence mapping based on research
        arousal_map = {
            EmotionLabel.ANGRY: 0.8,
            EmotionLabel.HAPPY: 0.7,
            EmotionLabel.SAD: 0.3,
            EmotionLabel.NEUTRAL: 0.5,
            EmotionLabel.FEAR: 0.9,
            EmotionLabel.DISGUST: 0.6,
            EmotionLabel.SURPRISE: 0.8
        }
        
        valence_map = {
            EmotionLabel.ANGRY: 0.2,
            EmotionLabel.HAPPY: 0.9,
            EmotionLabel.SAD: 0.1,
            EmotionLabel.NEUTRAL: 0.5,
            EmotionLabel.FEAR: 0.2,
            EmotionLabel.DISGUST: 0.1,
            EmotionLabel.SURPRISE: 0.6
        }
        
        arousal = sum(emotion_scores.get(emotion, 0.0) * arousal_map.get(emotion, 0.5) 
                     for emotion in arousal_map)
        valence = sum(emotion_scores.get(emotion, 0.0) * valence_map.get(emotion, 0.5) 
                     for emotion in valence_map)
        
        return float(arousal), float(valence)
    
    def get_processing_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "total_predictions": self.total_predictions,
            "average_processing_time_ms": self.average_processing_time,
            "last_processing_time_ms": self.last_prediction_time,
            "buffer_size": len(self.audio_buffer),
            "is_initialized": self.is_initialized,
            "device": self.device,
            "emotion_trend": self.trajectory_tracker.get_emotion_trend()
        }
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            self.audio_buffer.clear()
            if hasattr(self, 'classifier'):
                # Clear GPU memory if using CUDA
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            self.logger.info("SpeechEmotionRecognizer cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


def create_speech_emotion_recognizer(
    config: Optional[EmotionRecognitionConfig] = None
) -> SpeechEmotionRecognizer:
    """Factory function to create SpeechEmotionRecognizer"""
    return SpeechEmotionRecognizer(config)


def create_default_config() -> EmotionRecognitionConfig:
    """Create default configuration for emotion recognition"""
    return EmotionRecognitionConfig() 