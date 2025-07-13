"""
Voice Consistency Tracker for Sovereign 4.0 Voice Assistant

Implements voice consistency tracking using librosa audio feature analysis to maintain
voice characteristics across conversation turns. Ensures emotional voice preservation
without drift or inconsistency in the Realtime API audio-only processing pipeline.

Key Features:
- Voice fingerprinting with spectral feature analysis
- Voice similarity scoring across conversation turns
- Voice drift detection and correction algorithms
- Voice profile baseline establishment and maintenance
- Integration with EmotionPreserver and SpeechEmotionRecognizer
- Real-time voice characteristic monitoring

Usage:
    voice_tracker = VoiceConsistencyTracker()
    await voice_tracker.initialize()
    
    # Establish baseline voice profile
    await voice_tracker.establish_baseline(audio_samples)
    
    # Track consistency during conversation
    consistency_result = await voice_tracker.analyze_consistency(audio_chunk)
    
    # Get voice adaptation recommendations
    recommendations = voice_tracker.get_adaptation_recommendations()
"""

import asyncio
import logging
import time
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import pickle
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)


class VoiceConsistencyLevel(Enum):
    """Voice consistency levels for classification"""
    EXCELLENT = "excellent"    # >95% similarity
    GOOD = "good"             # 85-95% similarity
    MODERATE = "moderate"     # 70-85% similarity
    POOR = "poor"            # 50-70% similarity
    INCONSISTENT = "inconsistent"  # <50% similarity


class VoiceDriftType(Enum):
    """Types of voice drift detection"""
    PITCH_DRIFT = "pitch_drift"
    TIMBRE_DRIFT = "timbre_drift" 
    ENERGY_DRIFT = "energy_drift"
    SPECTRAL_DRIFT = "spectral_drift"
    TEMPORAL_DRIFT = "temporal_drift"


@dataclass
class VoiceFingerprint:
    """Voice fingerprint containing characteristic features"""
    # Fundamental frequency characteristics
    f0_mean: float
    f0_std: float
    f0_range: float
    
    # Spectral characteristics
    spectral_centroid_mean: float
    spectral_centroid_std: float
    spectral_rolloff_mean: float
    spectral_bandwidth_mean: float
    
    # MFCC characteristics (first 13 coefficients)
    mfcc_mean: np.ndarray
    mfcc_std: np.ndarray
    
    # Chroma and spectral contrast
    chroma_mean: np.ndarray
    spectral_contrast_mean: np.ndarray
    
    # Energy and temporal characteristics
    rms_energy_mean: float
    rms_energy_std: float
    zero_crossing_rate_mean: float
    tempo: float
    
    # Quality and metadata
    quality_score: float
    sample_count: int
    timestamp: float = field(default_factory=time.time)
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert fingerprint to numerical feature vector for comparison"""
        features = [
            self.f0_mean, self.f0_std, self.f0_range,
            self.spectral_centroid_mean, self.spectral_centroid_std,
            self.spectral_rolloff_mean, self.spectral_bandwidth_mean,
            self.rms_energy_mean, self.rms_energy_std,
            self.zero_crossing_rate_mean, self.tempo,
            self.quality_score
        ]
        
        # Add MFCC means (flatten to 1D)
        if self.mfcc_mean is not None:
            features.extend(self.mfcc_mean.flatten()[:13])  # First 13 MFCC coefficients
        else:
            features.extend([0.0] * 13)
            
        # Add chroma means (12 coefficients)
        if self.chroma_mean is not None:
            features.extend(self.chroma_mean.flatten()[:12])
        else:
            features.extend([0.0] * 12)
            
        # Add spectral contrast means (7 coefficients)
        if self.spectral_contrast_mean is not None:
            features.extend(self.spectral_contrast_mean.flatten()[:7])
        else:
            features.extend([0.0] * 7)
            
        return np.array(features, dtype=np.float32)


@dataclass
class VoiceConsistencyResult:
    """Result of voice consistency analysis"""
    consistency_score: float  # 0.0 to 1.0
    consistency_level: VoiceConsistencyLevel
    similarity_to_baseline: float
    detected_drifts: List[VoiceDriftType]
    feature_differences: Dict[str, float]
    recommendations: List[str]
    processing_time_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class VoiceProfile:
    """Voice profile containing baseline and historical data"""
    baseline_fingerprint: VoiceFingerprint
    recent_fingerprints: deque  # Recent voice samples for trend analysis
    drift_history: List[Tuple[float, VoiceDriftType]]  # (timestamp, drift_type)
    adaptation_parameters: Dict[str, Any]
    creation_timestamp: float
    last_updated: float
    sample_count: int = 0
    
    def __post_init__(self):
        if not hasattr(self, 'recent_fingerprints') or self.recent_fingerprints is None:
            self.recent_fingerprints = deque(maxlen=20)  # Keep last 20 samples


@dataclass
class VoiceConsistencyConfig:
    """Configuration for voice consistency tracking"""
    # Audio processing
    sample_rate: int = 16000
    chunk_duration_ms: int = 2000  # 2 second chunks for analysis
    overlap_ratio: float = 0.5
    
    # Consistency thresholds
    excellent_threshold: float = 0.95
    good_threshold: float = 0.85
    moderate_threshold: float = 0.70
    poor_threshold: float = 0.50
    
    # Drift detection settings
    pitch_drift_threshold: float = 0.15  # 15% change in F0
    timbre_drift_threshold: float = 0.20  # 20% change in spectral features
    energy_drift_threshold: float = 0.25  # 25% change in RMS energy
    
    # Profile management
    baseline_sample_count: int = 5  # Samples needed to establish baseline
    profile_update_frequency: int = 10  # Update profile every N samples
    max_profile_history: int = 100  # Maximum historical samples to keep
    
    # Feature extraction
    n_mfcc: int = 13
    n_chroma: int = 12
    n_spectral_contrast: int = 7
    
    # Quality control
    min_audio_quality: float = 0.6
    min_audio_length_ms: int = 1000  # Minimum 1 second for analysis
    
    # Persistence
    enable_profile_persistence: bool = True
    profile_save_path: str = ".voice_profiles"


class VoiceFeatureExtractor:
    """Extract voice features using librosa for consistency tracking"""
    
    def __init__(self, config: VoiceConsistencyConfig):
        self.config = config
        self.sample_rate = config.sample_rate
        
    def extract_voice_fingerprint(self, audio: np.ndarray, quality_score: float = 1.0) -> VoiceFingerprint:
        """Extract comprehensive voice fingerprint from audio"""
        try:
            # Import librosa here to handle optional dependency
            import librosa
            
            # Ensure audio is 1D and normalized
            if audio.ndim > 1:
                audio = np.mean(audio, axis=0)
            
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            # Fundamental frequency analysis
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=80, fmax=400, sr=self.sample_rate, frame_length=2048
            )
            f0_clean = f0[voiced_flag]
            f0_mean = np.nanmean(f0_clean) if len(f0_clean) > 0 else 0.0
            f0_std = np.nanstd(f0_clean) if len(f0_clean) > 0 else 0.0
            f0_range = np.nanmax(f0_clean) - np.nanmin(f0_clean) if len(f0_clean) > 0 else 0.0
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
            
            # MFCC features
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.config.n_mfcc)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
            chroma_mean = np.mean(chroma, axis=1)
            
            # Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
            spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
            
            # Energy and temporal features
            rms_energy = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
            tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            
            return VoiceFingerprint(
                f0_mean=float(f0_mean),
                f0_std=float(f0_std),
                f0_range=float(f0_range),
                spectral_centroid_mean=float(np.mean(spectral_centroid)),
                spectral_centroid_std=float(np.std(spectral_centroid)),
                spectral_rolloff_mean=float(np.mean(spectral_rolloff)),
                spectral_bandwidth_mean=float(np.mean(spectral_bandwidth)),
                mfcc_mean=mfcc_mean,
                mfcc_std=mfcc_std,
                chroma_mean=chroma_mean,
                spectral_contrast_mean=spectral_contrast_mean,
                rms_energy_mean=float(np.mean(rms_energy)),
                rms_energy_std=float(np.std(rms_energy)),
                zero_crossing_rate_mean=float(np.mean(zero_crossing_rate)),
                tempo=float(tempo),
                quality_score=quality_score,
                sample_count=1
            )
            
        except ImportError:
            logger.warning("librosa not available, using basic voice fingerprint")
            return self._extract_basic_fingerprint(audio, quality_score)
        except Exception as e:
            logger.error(f"Voice fingerprint extraction failed: {e}")
            return self._extract_basic_fingerprint(audio, quality_score)
    
    def _extract_basic_fingerprint(self, audio: np.ndarray, quality_score: float) -> VoiceFingerprint:
        """Extract basic voice fingerprint without librosa"""
        # Basic energy and frequency analysis
        rms_energy = np.sqrt(np.mean(audio ** 2))
        zero_crossings = np.where(np.diff(np.signbit(audio)))[0]
        zcr = len(zero_crossings) / len(audio)
        
        # Basic spectral analysis using FFT
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.fftfreq(len(fft), 1/self.sample_rate)
        
        # Spectral centroid approximation
        spectral_centroid = np.sum(freqs[:len(freqs)//2] * magnitude[:len(magnitude)//2]) / np.sum(magnitude[:len(magnitude)//2])
        
        return VoiceFingerprint(
            f0_mean=0.0,
            f0_std=0.0,
            f0_range=0.0,
            spectral_centroid_mean=abs(float(spectral_centroid)),
            spectral_centroid_std=0.0,
            spectral_rolloff_mean=0.0,
            spectral_bandwidth_mean=0.0,
            mfcc_mean=np.zeros(self.config.n_mfcc),
            mfcc_std=np.zeros(self.config.n_mfcc),
            chroma_mean=np.zeros(self.config.n_chroma),
            spectral_contrast_mean=np.zeros(self.config.n_spectral_contrast),
            rms_energy_mean=float(rms_energy),
            rms_energy_std=0.0,
            zero_crossing_rate_mean=float(zcr),
            tempo=120.0,  # Default tempo
            quality_score=quality_score,
            sample_count=1
        )


class VoiceSimilarityCalculator:
    """Calculate similarity between voice fingerprints"""
    
    @staticmethod
    def calculate_similarity(fingerprint1: VoiceFingerprint, fingerprint2: VoiceFingerprint) -> float:
        """
        Calculate similarity score between two voice fingerprints (0.0 to 1.0)
        Uses weighted combination of different feature similarities
        """
        try:
            # Convert fingerprints to feature vectors
            features1 = fingerprint1.to_feature_vector()
            features2 = fingerprint2.to_feature_vector()
            
            # Ensure same length
            min_length = min(len(features1), len(features2))
            features1 = features1[:min_length]
            features2 = features2[:min_length]
            
            # Calculate normalized differences for each feature group
            similarities = []
            
            # Fundamental frequency similarity (indices 0-2)
            f0_sim = VoiceSimilarityCalculator._calculate_feature_similarity(
                features1[0:3], features2[0:3], tolerance=0.2
            )
            similarities.append(f0_sim * 0.25)  # 25% weight
            
            # Spectral similarity (indices 3-6)
            spectral_sim = VoiceSimilarityCalculator._calculate_feature_similarity(
                features1[3:7], features2[3:7], tolerance=0.15
            )
            similarities.append(spectral_sim * 0.30)  # 30% weight
            
            # Energy/temporal similarity (indices 7-11)
            energy_sim = VoiceSimilarityCalculator._calculate_feature_similarity(
                features1[7:12], features2[7:12], tolerance=0.25
            )
            similarities.append(energy_sim * 0.15)  # 15% weight
            
            # MFCC similarity (indices 12-24)
            mfcc_sim = VoiceSimilarityCalculator._calculate_feature_similarity(
                features1[12:25], features2[12:25], tolerance=0.20
            )
            similarities.append(mfcc_sim * 0.30)  # 30% weight
            
            # Overall similarity score
            overall_similarity = sum(similarities)
            return float(np.clip(overall_similarity, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Similarity calculation failed: {e}")
            return 0.5  # Default moderate similarity
    
    @staticmethod
    def _calculate_feature_similarity(features1: np.ndarray, features2: np.ndarray, tolerance: float = 0.2) -> float:
        """Calculate similarity for a specific feature group"""
        # Handle zero values to avoid division by zero
        features1 = features1 + 1e-8
        features2 = features2 + 1e-8
        
        # Calculate relative differences
        relative_diffs = np.abs(features1 - features2) / (np.abs(features1) + np.abs(features2))
        
        # Convert to similarity scores (1.0 - normalized_difference)
        similarities = 1.0 - np.clip(relative_diffs / tolerance, 0.0, 1.0)
        
        # Return average similarity for the feature group
        return float(np.mean(similarities))


class VoiceDriftDetector:
    """Detect different types of voice drift"""
    
    def __init__(self, config: VoiceConsistencyConfig):
        self.config = config
        
    def detect_drifts(self, baseline: VoiceFingerprint, current: VoiceFingerprint) -> List[VoiceDriftType]:
        """Detect various types of voice drift"""
        detected_drifts = []
        
        # Pitch drift detection
        if self._detect_pitch_drift(baseline, current):
            detected_drifts.append(VoiceDriftType.PITCH_DRIFT)
            
        # Timbre drift detection
        if self._detect_timbre_drift(baseline, current):
            detected_drifts.append(VoiceDriftType.TIMBRE_DRIFT)
            
        # Energy drift detection
        if self._detect_energy_drift(baseline, current):
            detected_drifts.append(VoiceDriftType.ENERGY_DRIFT)
            
        # Spectral drift detection
        if self._detect_spectral_drift(baseline, current):
            detected_drifts.append(VoiceDriftType.SPECTRAL_DRIFT)
            
        return detected_drifts
    
    def _detect_pitch_drift(self, baseline: VoiceFingerprint, current: VoiceFingerprint) -> bool:
        """Detect significant pitch drift"""
        if baseline.f0_mean == 0 or current.f0_mean == 0:
            return False
            
        relative_change = abs(current.f0_mean - baseline.f0_mean) / baseline.f0_mean
        return relative_change > self.config.pitch_drift_threshold
    
    def _detect_timbre_drift(self, baseline: VoiceFingerprint, current: VoiceFingerprint) -> bool:
        """Detect timbre drift using MFCC comparison"""
        if baseline.mfcc_mean is None or current.mfcc_mean is None:
            return False
            
        # Compare first few MFCC coefficients (most important for timbre)
        mfcc_diff = np.mean(np.abs(current.mfcc_mean[:5] - baseline.mfcc_mean[:5]))
        return mfcc_diff > self.config.timbre_drift_threshold
    
    def _detect_energy_drift(self, baseline: VoiceFingerprint, current: VoiceFingerprint) -> bool:
        """Detect significant energy drift"""
        if baseline.rms_energy_mean == 0:
            return False
            
        relative_change = abs(current.rms_energy_mean - baseline.rms_energy_mean) / baseline.rms_energy_mean
        return relative_change > self.config.energy_drift_threshold
    
    def _detect_spectral_drift(self, baseline: VoiceFingerprint, current: VoiceFingerprint) -> bool:
        """Detect spectral characteristic drift"""
        if baseline.spectral_centroid_mean == 0:
            return False
            
        relative_change = abs(current.spectral_centroid_mean - baseline.spectral_centroid_mean) / baseline.spectral_centroid_mean
        return relative_change > self.config.timbre_drift_threshold


class VoiceConsistencyTracker:
    """
    Main class for voice consistency tracking and management.
    
    Provides voice fingerprinting, consistency analysis, drift detection,
    and voice profile management for maintaining consistent voice characteristics
    across conversation turns.
    """
    
    def __init__(self, config: Optional[VoiceConsistencyConfig] = None):
        self.config = config or VoiceConsistencyConfig()
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.feature_extractor = VoiceFeatureExtractor(self.config)
        self.similarity_calculator = VoiceSimilarityCalculator()
        self.drift_detector = VoiceDriftDetector(self.config)
        
        # Voice profile management
        self.voice_profile: Optional[VoiceProfile] = None
        self.is_baseline_established = False
        self.baseline_samples = []
        
        # Processing state
        self.is_initialized = False
        self.processing_lock = asyncio.Lock()
        
        # Performance metrics
        self.total_analyses = 0
        self.average_processing_time = 0.0
        self.consistency_history = deque(maxlen=50)
        
    async def initialize(self) -> bool:
        """Initialize the voice consistency tracker"""
        try:
            self.logger.info("Initializing VoiceConsistencyTracker...")
            
            # Create profile save directory if needed
            if self.config.enable_profile_persistence:
                profile_path = Path(self.config.profile_save_path)
                profile_path.mkdir(exist_ok=True)
            
            # Try to load existing voice profile
            if self.config.enable_profile_persistence:
                await self._load_voice_profile()
            
            self.is_initialized = True
            self.logger.info("VoiceConsistencyTracker initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize VoiceConsistencyTracker: {e}")
            return False
    
    async def establish_baseline(self, audio_samples: List[bytes]) -> bool:
        """Establish baseline voice profile from multiple audio samples"""
        if not self.is_initialized:
            self.logger.error("VoiceConsistencyTracker not initialized")
            return False
        
        async with self.processing_lock:
            try:
                self.logger.info(f"Establishing voice baseline from {len(audio_samples)} samples")
                
                fingerprints = []
                for audio_data in audio_samples:
                    # Convert bytes to numpy array
                    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                    audio_array = audio_array / 32768.0  # Normalize
                    
                    # Extract fingerprint
                    fingerprint = self.feature_extractor.extract_voice_fingerprint(audio_array)
                    if fingerprint.quality_score >= self.config.min_audio_quality:
                        fingerprints.append(fingerprint)
                
                if len(fingerprints) < self.config.baseline_sample_count:
                    self.logger.warning(f"Insufficient quality samples for baseline: {len(fingerprints)}")
                    return False
                
                # Create averaged baseline fingerprint
                baseline_fingerprint = self._create_averaged_fingerprint(fingerprints)
                
                # Create voice profile
                self.voice_profile = VoiceProfile(
                    baseline_fingerprint=baseline_fingerprint,
                    recent_fingerprints=deque(fingerprints[-10:], maxlen=20),
                    drift_history=[],
                    adaptation_parameters={},
                    creation_timestamp=time.time(),
                    last_updated=time.time(),
                    sample_count=len(fingerprints)
                )
                
                self.is_baseline_established = True
                
                # Save profile if persistence enabled
                if self.config.enable_profile_persistence:
                    await self._save_voice_profile()
                
                self.logger.info("Voice baseline established successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to establish baseline: {e}")
                return False
    
    async def analyze_consistency(self, audio_data: bytes) -> Optional[VoiceConsistencyResult]:
        """Analyze voice consistency for current audio chunk"""
        if not self.is_initialized or not self.is_baseline_established:
            self.logger.warning("Voice consistency tracker not ready for analysis")
            return None
        
        start_time = time.time()
        
        async with self.processing_lock:
            try:
                # Convert audio to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                audio_array = audio_array / 32768.0  # Normalize
                
                # Check audio length
                audio_length_ms = len(audio_array) / self.config.sample_rate * 1000
                if audio_length_ms < self.config.min_audio_length_ms:
                    return None
                
                # Extract current fingerprint
                current_fingerprint = self.feature_extractor.extract_voice_fingerprint(audio_array)
                
                # Check audio quality
                if current_fingerprint.quality_score < self.config.min_audio_quality:
                    self.logger.warning(f"Low audio quality: {current_fingerprint.quality_score}")
                    return None
                
                # Calculate similarity to baseline
                similarity_score = self.similarity_calculator.calculate_similarity(
                    self.voice_profile.baseline_fingerprint, current_fingerprint
                )
                
                # Determine consistency level
                consistency_level = self._determine_consistency_level(similarity_score)
                
                # Detect drifts
                detected_drifts = self.drift_detector.detect_drifts(
                    self.voice_profile.baseline_fingerprint, current_fingerprint
                )
                
                # Calculate feature differences
                feature_differences = self._calculate_feature_differences(
                    self.voice_profile.baseline_fingerprint, current_fingerprint
                )
                
                # Generate recommendations
                recommendations = self._generate_recommendations(consistency_level, detected_drifts)
                
                # Update voice profile
                self.voice_profile.recent_fingerprints.append(current_fingerprint)
                self.voice_profile.sample_count += 1
                self.voice_profile.last_updated = time.time()
                
                # Record drift history
                for drift_type in detected_drifts:
                    self.voice_profile.drift_history.append((time.time(), drift_type))
                
                # Update performance metrics
                processing_time = (time.time() - start_time) * 1000
                self.total_analyses += 1
                self.average_processing_time = (
                    (self.average_processing_time * (self.total_analyses - 1) + processing_time) /
                    self.total_analyses
                )
                
                # Create result
                result = VoiceConsistencyResult(
                    consistency_score=similarity_score,
                    consistency_level=consistency_level,
                    similarity_to_baseline=similarity_score,
                    detected_drifts=detected_drifts,
                    feature_differences=feature_differences,
                    recommendations=recommendations,
                    processing_time_ms=processing_time
                )
                
                # Store in consistency history
                self.consistency_history.append(result)
                
                return result
                
            except Exception as e:
                self.logger.error(f"Voice consistency analysis failed: {e}")
                return None
    
    def _create_averaged_fingerprint(self, fingerprints: List[VoiceFingerprint]) -> VoiceFingerprint:
        """Create an averaged fingerprint from multiple samples"""
        if not fingerprints:
            raise ValueError("No fingerprints provided for averaging")
        
        # Average scalar values
        f0_means = [fp.f0_mean for fp in fingerprints if fp.f0_mean > 0]
        f0_mean = np.mean(f0_means) if f0_means else 0.0
        
        f0_stds = [fp.f0_std for fp in fingerprints if fp.f0_std > 0]
        f0_std = np.mean(f0_stds) if f0_stds else 0.0
        
        # Average other scalar features
        scalar_features = [
            'f0_range', 'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_rolloff_mean', 'spectral_bandwidth_mean',
            'rms_energy_mean', 'rms_energy_std', 'zero_crossing_rate_mean', 'tempo'
        ]
        
        averaged_values = {}
        for feature in scalar_features:
            values = [getattr(fp, feature) for fp in fingerprints]
            averaged_values[feature] = float(np.mean(values))
        
        # Average array features
        mfcc_means = [fp.mfcc_mean for fp in fingerprints if fp.mfcc_mean is not None]
        mfcc_mean = np.mean(mfcc_means, axis=0) if mfcc_means else np.zeros(self.config.n_mfcc)
        
        mfcc_stds = [fp.mfcc_std for fp in fingerprints if fp.mfcc_std is not None]
        mfcc_std = np.mean(mfcc_stds, axis=0) if mfcc_stds else np.zeros(self.config.n_mfcc)
        
        chroma_means = [fp.chroma_mean for fp in fingerprints if fp.chroma_mean is not None]
        chroma_mean = np.mean(chroma_means, axis=0) if chroma_means else np.zeros(self.config.n_chroma)
        
        spectral_contrast_means = [fp.spectral_contrast_mean for fp in fingerprints if fp.spectral_contrast_mean is not None]
        spectral_contrast_mean = np.mean(spectral_contrast_means, axis=0) if spectral_contrast_means else np.zeros(self.config.n_spectral_contrast)
        
        # Average quality scores
        quality_scores = [fp.quality_score for fp in fingerprints]
        quality_score = float(np.mean(quality_scores))
        
        return VoiceFingerprint(
            f0_mean=f0_mean,
            f0_std=f0_std,
            mfcc_mean=mfcc_mean,
            mfcc_std=mfcc_std,
            chroma_mean=chroma_mean,
            spectral_contrast_mean=spectral_contrast_mean,
            quality_score=quality_score,
            sample_count=len(fingerprints),
            **averaged_values
        )
    
    def _determine_consistency_level(self, similarity_score: float) -> VoiceConsistencyLevel:
        """Determine consistency level based on similarity score"""
        if similarity_score >= self.config.excellent_threshold:
            return VoiceConsistencyLevel.EXCELLENT
        elif similarity_score >= self.config.good_threshold:
            return VoiceConsistencyLevel.GOOD
        elif similarity_score >= self.config.moderate_threshold:
            return VoiceConsistencyLevel.MODERATE
        elif similarity_score >= self.config.poor_threshold:
            return VoiceConsistencyLevel.POOR
        else:
            return VoiceConsistencyLevel.INCONSISTENT
    
    def _calculate_feature_differences(self, baseline: VoiceFingerprint, current: VoiceFingerprint) -> Dict[str, float]:
        """Calculate specific feature differences"""
        differences = {}
        
        # Pitch differences
        if baseline.f0_mean > 0:
            differences['pitch_change_percent'] = abs(current.f0_mean - baseline.f0_mean) / baseline.f0_mean * 100
        
        # Energy differences
        if baseline.rms_energy_mean > 0:
            differences['energy_change_percent'] = abs(current.rms_energy_mean - baseline.rms_energy_mean) / baseline.rms_energy_mean * 100
        
        # Spectral differences
        if baseline.spectral_centroid_mean > 0:
            differences['spectral_change_percent'] = abs(current.spectral_centroid_mean - baseline.spectral_centroid_mean) / baseline.spectral_centroid_mean * 100
        
        return differences
    
    def _generate_recommendations(self, consistency_level: VoiceConsistencyLevel, detected_drifts: List[VoiceDriftType]) -> List[str]:
        """Generate recommendations based on consistency analysis"""
        recommendations = []
        
        if consistency_level == VoiceConsistencyLevel.INCONSISTENT:
            recommendations.append("Voice consistency is very poor - consider re-establishing baseline")
        elif consistency_level == VoiceConsistencyLevel.POOR:
            recommendations.append("Voice consistency is poor - voice adaptation may be needed")
        elif consistency_level == VoiceConsistencyLevel.MODERATE:
            recommendations.append("Voice consistency is moderate - minor adjustments recommended")
        
        for drift_type in detected_drifts:
            if drift_type == VoiceDriftType.PITCH_DRIFT:
                recommendations.append("Significant pitch drift detected - adjust pitch normalization")
            elif drift_type == VoiceDriftType.TIMBRE_DRIFT:
                recommendations.append("Timbre drift detected - voice characteristic adaptation needed")
            elif drift_type == VoiceDriftType.ENERGY_DRIFT:
                recommendations.append("Energy drift detected - adjust audio levels")
            elif drift_type == VoiceDriftType.SPECTRAL_DRIFT:
                recommendations.append("Spectral drift detected - spectral normalization recommended")
        
        return recommendations
    
    async def _save_voice_profile(self):
        """Save voice profile to disk"""
        try:
            if not self.voice_profile:
                return
            
            profile_path = Path(self.config.profile_save_path) / "voice_profile.pkl"
            with open(profile_path, 'wb') as f:
                pickle.dump(self.voice_profile, f)
            
            self.logger.info(f"Voice profile saved to {profile_path}")
        except Exception as e:
            self.logger.error(f"Failed to save voice profile: {e}")
    
    async def _load_voice_profile(self):
        """Load voice profile from disk"""
        try:
            profile_path = Path(self.config.profile_save_path) / "voice_profile.pkl"
            if profile_path.exists():
                with open(profile_path, 'rb') as f:
                    self.voice_profile = pickle.load(f)
                
                self.is_baseline_established = True
                self.logger.info(f"Voice profile loaded from {profile_path}")
        except Exception as e:
            self.logger.warning(f"Failed to load voice profile: {e}")
    
    def get_adaptation_recommendations(self) -> Dict[str, Any]:
        """Get voice adaptation recommendations based on recent analysis"""
        if not self.voice_profile or not self.consistency_history:
            return {}
        
        # Analyze recent consistency trends
        recent_scores = [result.consistency_score for result in list(self.consistency_history)[-10:]]
        avg_recent_score = np.mean(recent_scores) if recent_scores else 0.0
        
        # Analyze drift patterns
        recent_drifts = {}
        for result in list(self.consistency_history)[-5:]:
            for drift_type in result.detected_drifts:
                drift_name = drift_type.value
                recent_drifts[drift_name] = recent_drifts.get(drift_name, 0) + 1
        
        return {
            'average_recent_consistency': avg_recent_score,
            'consistency_trend': 'improving' if len(recent_scores) > 1 and recent_scores[-1] > recent_scores[0] else 'declining',
            'frequent_drifts': {k: v for k, v in recent_drifts.items() if v >= 2},
            'total_analyses': self.total_analyses,
            'average_processing_time_ms': self.average_processing_time,
            'baseline_age_hours': (time.time() - self.voice_profile.creation_timestamp) / 3600 if self.voice_profile else 0
        }
    
    def get_processing_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'total_analyses': self.total_analyses,
            'average_processing_time_ms': self.average_processing_time,
            'is_initialized': self.is_initialized,
            'is_baseline_established': self.is_baseline_established,
            'profile_sample_count': self.voice_profile.sample_count if self.voice_profile else 0,
            'recent_consistency_scores': [r.consistency_score for r in list(self.consistency_history)[-5:]],
            'baseline_age_hours': (time.time() - self.voice_profile.creation_timestamp) / 3600 if self.voice_profile else 0
        }
    
    async def cleanup(self):
        """Clean up resources and save state"""
        try:
            if self.config.enable_profile_persistence and self.voice_profile:
                await self._save_voice_profile()
            
            self.logger.info("VoiceConsistencyTracker cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


def create_voice_consistency_tracker(config: Optional[VoiceConsistencyConfig] = None) -> VoiceConsistencyTracker:
    """Factory function to create VoiceConsistencyTracker"""
    return VoiceConsistencyTracker(config)


def create_default_voice_config() -> VoiceConsistencyConfig:
    """Create default configuration for voice consistency tracking"""
    return VoiceConsistencyConfig() 