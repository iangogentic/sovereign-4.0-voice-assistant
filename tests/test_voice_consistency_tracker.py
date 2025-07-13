"""
Test suite for Voice Consistency Tracker module

Tests voice consistency tracking functionality including:
- VoiceConsistencyTracker initialization and baseline establishment
- Voice fingerprinting and feature extraction
- Voice similarity calculation and drift detection
- Voice profile management and persistence
- Performance optimization and error handling
- Integration with existing emotion preservation systems
"""

import pytest
import asyncio
import numpy as np
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import asdict
import tempfile
import os
from pathlib import Path

# Import the modules to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from assistant.voice_consistency_tracker import (
    VoiceConsistencyTracker,
    VoiceConsistencyConfig,
    VoiceFingerprint,
    VoiceConsistencyResult,
    VoiceProfile,
    VoiceConsistencyLevel,
    VoiceDriftType,
    VoiceFeatureExtractor,
    VoiceSimilarityCalculator,
    VoiceDriftDetector,
    create_voice_consistency_tracker,
    create_default_voice_config
)


class TestVoiceConsistencyConfig:
    """Test voice consistency configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = VoiceConsistencyConfig()
        
        assert config.sample_rate == 16000
        assert config.chunk_duration_ms == 2000
        assert config.overlap_ratio == 0.5
        assert config.excellent_threshold == 0.95
        assert config.good_threshold == 0.85
        assert config.moderate_threshold == 0.70
        assert config.poor_threshold == 0.50
        assert config.pitch_drift_threshold == 0.15
        assert config.timbre_drift_threshold == 0.20
        assert config.energy_drift_threshold == 0.25
        assert config.baseline_sample_count == 5
        assert config.n_mfcc == 13
        assert config.n_chroma == 12
        assert config.n_spectral_contrast == 7
        assert config.min_audio_quality == 0.6
        assert config.enable_profile_persistence == True
        
    def test_custom_config(self):
        """Test custom configuration creation"""
        config = VoiceConsistencyConfig(
            sample_rate=22050,
            chunk_duration_ms=3000,
            excellent_threshold=0.98,
            baseline_sample_count=10,
            enable_profile_persistence=False
        )
        
        assert config.sample_rate == 22050
        assert config.chunk_duration_ms == 3000
        assert config.excellent_threshold == 0.98
        assert config.baseline_sample_count == 10
        assert config.enable_profile_persistence == False


class TestVoiceFingerprint:
    """Test voice fingerprint data structure"""
    
    def test_voice_fingerprint_creation(self):
        """Test voice fingerprint creation"""
        mfcc_mean = np.random.rand(13)
        chroma_mean = np.random.rand(12)
        spectral_contrast_mean = np.random.rand(7)
        
        fingerprint = VoiceFingerprint(
            f0_mean=220.0,
            f0_std=15.0,
            f0_range=50.0,
            spectral_centroid_mean=1500.0,
            spectral_centroid_std=200.0,
            spectral_rolloff_mean=3000.0,
            spectral_bandwidth_mean=800.0,
            mfcc_mean=mfcc_mean,
            mfcc_std=np.random.rand(13),
            chroma_mean=chroma_mean,
            spectral_contrast_mean=spectral_contrast_mean,
            rms_energy_mean=0.15,
            rms_energy_std=0.02,
            zero_crossing_rate_mean=0.05,
            tempo=120.0,
            quality_score=0.85,
            sample_count=1
        )
        
        assert fingerprint.f0_mean == 220.0
        assert fingerprint.f0_std == 15.0
        assert fingerprint.spectral_centroid_mean == 1500.0
        assert fingerprint.quality_score == 0.85
        assert isinstance(fingerprint.timestamp, float)
        
    def test_feature_vector_conversion(self):
        """Test conversion to feature vector"""
        mfcc_mean = np.random.rand(13)
        chroma_mean = np.random.rand(12)
        spectral_contrast_mean = np.random.rand(7)
        
        fingerprint = VoiceFingerprint(
            f0_mean=220.0,
            f0_std=15.0,
            f0_range=50.0,
            spectral_centroid_mean=1500.0,
            spectral_centroid_std=200.0,
            spectral_rolloff_mean=3000.0,
            spectral_bandwidth_mean=800.0,
            mfcc_mean=mfcc_mean,
            mfcc_std=np.random.rand(13),
            chroma_mean=chroma_mean,
            spectral_contrast_mean=spectral_contrast_mean,
            rms_energy_mean=0.15,
            rms_energy_std=0.02,
            zero_crossing_rate_mean=0.05,
            tempo=120.0,
            quality_score=0.85,
            sample_count=1
        )
        
        feature_vector = fingerprint.to_feature_vector()
        
        # Should have: 12 scalar + 13 MFCC + 12 chroma + 7 spectral_contrast = 44 features
        assert len(feature_vector) == 44
        assert isinstance(feature_vector, np.ndarray)
        assert feature_vector.dtype == np.float32


class TestVoiceConsistencyResult:
    """Test voice consistency result data structure"""
    
    def test_consistency_result_creation(self):
        """Test consistency result creation"""
        result = VoiceConsistencyResult(
            consistency_score=0.87,
            consistency_level=VoiceConsistencyLevel.GOOD,
            similarity_to_baseline=0.87,
            detected_drifts=[VoiceDriftType.PITCH_DRIFT],
            feature_differences={'pitch_change_percent': 12.5},
            recommendations=['Adjust pitch normalization'],
            processing_time_ms=45.2
        )
        
        assert result.consistency_score == 0.87
        assert result.consistency_level == VoiceConsistencyLevel.GOOD
        assert result.similarity_to_baseline == 0.87
        assert VoiceDriftType.PITCH_DRIFT in result.detected_drifts
        assert 'pitch_change_percent' in result.feature_differences
        assert len(result.recommendations) == 1
        assert result.processing_time_ms == 45.2
        assert isinstance(result.timestamp, float)


class TestVoiceFeatureExtractor:
    """Test voice feature extraction"""
    
    def test_extractor_initialization(self):
        """Test feature extractor initialization"""
        config = VoiceConsistencyConfig()
        extractor = VoiceFeatureExtractor(config)
        
        assert extractor.config == config
        assert extractor.sample_rate == config.sample_rate
        
    @patch('assistant.voice_consistency_tracker.librosa')
    def test_feature_extraction_with_librosa(self, mock_librosa):
        """Test feature extraction with librosa"""
        config = VoiceConsistencyConfig()
        extractor = VoiceFeatureExtractor(config)
        
        # Mock librosa functions
        mock_librosa.pyin.return_value = (
            np.array([220.0, 220.0, 220.0]), 
            np.array([True, True, True]),
            np.array([0.9, 0.9, 0.9])
        )
        mock_librosa.feature.spectral_centroid.return_value = np.array([[1000, 1100, 1050]])
        mock_librosa.feature.spectral_rolloff.return_value = np.array([[2000, 2100, 2050]])
        mock_librosa.feature.spectral_bandwidth.return_value = np.array([[800, 850, 825]])
        mock_librosa.feature.mfcc.return_value = np.random.rand(13, 10)
        mock_librosa.feature.chroma_stft.return_value = np.random.rand(12, 10)
        mock_librosa.feature.spectral_contrast.return_value = np.random.rand(7, 10)
        mock_librosa.feature.rms.return_value = np.array([[0.1, 0.15, 0.12]])
        mock_librosa.feature.zero_crossing_rate.return_value = np.array([[0.05, 0.06, 0.055]])
        mock_librosa.beat.beat_track.return_value = (120.0, np.array([0, 1, 2]))
        
        # Generate test audio
        audio = np.random.rand(16000)
        
        fingerprint = extractor.extract_voice_fingerprint(audio, quality_score=0.8)
        
        assert isinstance(fingerprint, VoiceFingerprint)
        assert fingerprint.f0_mean == 220.0
        assert fingerprint.spectral_centroid_mean > 0
        assert fingerprint.quality_score == 0.8
        assert fingerprint.mfcc_mean.shape[0] == 13
        assert fingerprint.chroma_mean.shape[0] == 12
        assert fingerprint.tempo == 120.0
        
    def test_basic_feature_extraction_fallback(self):
        """Test basic feature extraction without librosa"""
        config = VoiceConsistencyConfig()
        extractor = VoiceFeatureExtractor(config)
        
        # Generate test audio
        audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        
        with patch('assistant.voice_consistency_tracker.librosa', side_effect=ImportError):
            fingerprint = extractor._extract_basic_fingerprint(audio, quality_score=0.7)
            
            assert isinstance(fingerprint, VoiceFingerprint)
            assert fingerprint.f0_mean == 0.0  # Not calculated in basic mode
            assert fingerprint.rms_energy_mean > 0
            assert fingerprint.spectral_centroid_mean != 0
            assert fingerprint.zero_crossing_rate_mean > 0
            assert fingerprint.quality_score == 0.7
            assert fingerprint.mfcc_mean.shape == (config.n_mfcc,)


class TestVoiceSimilarityCalculator:
    """Test voice similarity calculation"""
    
    def test_identical_fingerprints(self):
        """Test similarity calculation for identical fingerprints"""
        fingerprint = VoiceFingerprint(
            f0_mean=220.0,
            f0_std=15.0,
            f0_range=50.0,
            spectral_centroid_mean=1500.0,
            spectral_centroid_std=200.0,
            spectral_rolloff_mean=3000.0,
            spectral_bandwidth_mean=800.0,
            mfcc_mean=np.random.rand(13),
            mfcc_std=np.random.rand(13),
            chroma_mean=np.random.rand(12),
            spectral_contrast_mean=np.random.rand(7),
            rms_energy_mean=0.15,
            rms_energy_std=0.02,
            zero_crossing_rate_mean=0.05,
            tempo=120.0,
            quality_score=0.85,
            sample_count=1
        )
        
        similarity = VoiceSimilarityCalculator.calculate_similarity(fingerprint, fingerprint)
        
        assert similarity == 1.0
        
    def test_different_fingerprints(self):
        """Test similarity calculation for different fingerprints"""
        fingerprint1 = VoiceFingerprint(
            f0_mean=220.0,
            f0_std=15.0,
            f0_range=50.0,
            spectral_centroid_mean=1500.0,
            spectral_centroid_std=200.0,
            spectral_rolloff_mean=3000.0,
            spectral_bandwidth_mean=800.0,
            mfcc_mean=np.ones(13),
            mfcc_std=np.ones(13),
            chroma_mean=np.ones(12),
            spectral_contrast_mean=np.ones(7),
            rms_energy_mean=0.15,
            rms_energy_std=0.02,
            zero_crossing_rate_mean=0.05,
            tempo=120.0,
            quality_score=0.85,
            sample_count=1
        )
        
        fingerprint2 = VoiceFingerprint(
            f0_mean=180.0,  # Different pitch
            f0_std=20.0,
            f0_range=60.0,
            spectral_centroid_mean=1800.0,  # Different spectral centroid
            spectral_centroid_std=250.0,
            spectral_rolloff_mean=3500.0,
            spectral_bandwidth_mean=900.0,
            mfcc_mean=np.zeros(13),  # Different MFCC
            mfcc_std=np.zeros(13),
            chroma_mean=np.zeros(12),
            spectral_contrast_mean=np.zeros(7),
            rms_energy_mean=0.25,  # Different energy
            rms_energy_std=0.03,
            zero_crossing_rate_mean=0.08,
            tempo=100.0,  # Different tempo
            quality_score=0.75,
            sample_count=1
        )
        
        similarity = VoiceSimilarityCalculator.calculate_similarity(fingerprint1, fingerprint2)
        
        assert 0.0 <= similarity <= 1.0
        assert similarity < 0.9  # Should be significantly different


class TestVoiceDriftDetector:
    """Test voice drift detection"""
    
    def test_drift_detector_initialization(self):
        """Test drift detector initialization"""
        config = VoiceConsistencyConfig()
        detector = VoiceDriftDetector(config)
        
        assert detector.config == config
        
    def test_pitch_drift_detection(self):
        """Test pitch drift detection"""
        config = VoiceConsistencyConfig(pitch_drift_threshold=0.1)  # 10% threshold
        detector = VoiceDriftDetector(config)
        
        baseline = VoiceFingerprint(
            f0_mean=200.0,
            f0_std=10.0,
            f0_range=40.0,
            spectral_centroid_mean=1500.0,
            spectral_centroid_std=200.0,
            spectral_rolloff_mean=3000.0,
            spectral_bandwidth_mean=800.0,
            mfcc_mean=np.ones(13),
            mfcc_std=np.ones(13),
            chroma_mean=np.ones(12),
            spectral_contrast_mean=np.ones(7),
            rms_energy_mean=0.15,
            rms_energy_std=0.02,
            zero_crossing_rate_mean=0.05,
            tempo=120.0,
            quality_score=0.85,
            sample_count=1
        )
        
        # Test significant pitch change (>10%)
        current_high_pitch = VoiceFingerprint(
            f0_mean=250.0,  # 25% increase
            f0_std=10.0,
            f0_range=40.0,
            spectral_centroid_mean=1500.0,
            spectral_centroid_std=200.0,
            spectral_rolloff_mean=3000.0,
            spectral_bandwidth_mean=800.0,
            mfcc_mean=np.ones(13),
            mfcc_std=np.ones(13),
            chroma_mean=np.ones(12),
            spectral_contrast_mean=np.ones(7),
            rms_energy_mean=0.15,
            rms_energy_std=0.02,
            zero_crossing_rate_mean=0.05,
            tempo=120.0,
            quality_score=0.85,
            sample_count=1
        )
        
        # Should detect pitch drift
        assert detector._detect_pitch_drift(baseline, current_high_pitch) == True
        
        # Test small pitch change (<10%)
        current_small_change = VoiceFingerprint(
            f0_mean=205.0,  # 2.5% increase
            f0_std=10.0,
            f0_range=40.0,
            spectral_centroid_mean=1500.0,
            spectral_centroid_std=200.0,
            spectral_rolloff_mean=3000.0,
            spectral_bandwidth_mean=800.0,
            mfcc_mean=np.ones(13),
            mfcc_std=np.ones(13),
            chroma_mean=np.ones(12),
            spectral_contrast_mean=np.ones(7),
            rms_energy_mean=0.15,
            rms_energy_std=0.02,
            zero_crossing_rate_mean=0.05,
            tempo=120.0,
            quality_score=0.85,
            sample_count=1
        )
        
        # Should not detect pitch drift
        assert detector._detect_pitch_drift(baseline, current_small_change) == False
        
    def test_energy_drift_detection(self):
        """Test energy drift detection"""
        config = VoiceConsistencyConfig(energy_drift_threshold=0.2)  # 20% threshold
        detector = VoiceDriftDetector(config)
        
        baseline = VoiceFingerprint(
            f0_mean=200.0,
            f0_std=10.0,
            f0_range=40.0,
            spectral_centroid_mean=1500.0,
            spectral_centroid_std=200.0,
            spectral_rolloff_mean=3000.0,
            spectral_bandwidth_mean=800.0,
            mfcc_mean=np.ones(13),
            mfcc_std=np.ones(13),
            chroma_mean=np.ones(12),
            spectral_contrast_mean=np.ones(7),
            rms_energy_mean=0.10,
            rms_energy_std=0.02,
            zero_crossing_rate_mean=0.05,
            tempo=120.0,
            quality_score=0.85,
            sample_count=1
        )
        
        # Test significant energy change (>20%)
        current_high_energy = VoiceFingerprint(
            f0_mean=200.0,
            f0_std=10.0,
            f0_range=40.0,
            spectral_centroid_mean=1500.0,
            spectral_centroid_std=200.0,
            spectral_rolloff_mean=3000.0,
            spectral_bandwidth_mean=800.0,
            mfcc_mean=np.ones(13),
            mfcc_std=np.ones(13),
            chroma_mean=np.ones(12),
            spectral_contrast_mean=np.ones(7),
            rms_energy_mean=0.25,  # 150% increase
            rms_energy_std=0.02,
            zero_crossing_rate_mean=0.05,
            tempo=120.0,
            quality_score=0.85,
            sample_count=1
        )
        
        # Should detect energy drift
        assert detector._detect_energy_drift(baseline, current_high_energy) == True


class TestVoiceConsistencyTracker:
    """Test main voice consistency tracker"""
    
    def test_tracker_initialization(self):
        """Test tracker initialization"""
        config = VoiceConsistencyConfig(enable_profile_persistence=False)
        tracker = VoiceConsistencyTracker(config)
        
        assert tracker.config == config
        assert not tracker.is_initialized
        assert tracker.voice_profile is None
        assert not tracker.is_baseline_established
        assert len(tracker.baseline_samples) == 0
        
    @pytest.mark.asyncio
    async def test_successful_initialization(self):
        """Test successful initialization"""
        config = VoiceConsistencyConfig(enable_profile_persistence=False)
        tracker = VoiceConsistencyTracker(config)
        
        result = await tracker.initialize()
        
        assert result == True
        assert tracker.is_initialized == True
        
    @pytest.mark.asyncio
    async def test_baseline_establishment(self):
        """Test baseline establishment"""
        config = VoiceConsistencyConfig(
            enable_profile_persistence=False,
            baseline_sample_count=3,
            min_audio_quality=0.5
        )
        tracker = VoiceConsistencyTracker(config)
        await tracker.initialize()
        
        # Create mock audio samples
        audio_samples = []
        for i in range(5):  # More than minimum required
            # Generate different but similar audio
            audio = 0.5 * np.sin(2 * np.pi * (440 + i*10) * np.linspace(0, 2, 32000))
            audio_data = (audio * 32767).astype(np.int16).tobytes()
            audio_samples.append(audio_data)
        
        # Mock the feature extractor to return consistent quality
        with patch.object(tracker.feature_extractor, 'extract_voice_fingerprint') as mock_extract:
            mock_fingerprints = []
            for i in range(5):
                fingerprint = VoiceFingerprint(
                    f0_mean=220.0 + i,
                    f0_std=15.0,
                    f0_range=50.0,
                    spectral_centroid_mean=1500.0,
                    spectral_centroid_std=200.0,
                    spectral_rolloff_mean=3000.0,
                    spectral_bandwidth_mean=800.0,
                    mfcc_mean=np.ones(13),
                    mfcc_std=np.ones(13),
                    chroma_mean=np.ones(12),
                    spectral_contrast_mean=np.ones(7),
                    rms_energy_mean=0.15,
                    rms_energy_std=0.02,
                    zero_crossing_rate_mean=0.05,
                    tempo=120.0,
                    quality_score=0.8,  # Above minimum
                    sample_count=1
                )
                mock_fingerprints.append(fingerprint)
            
            mock_extract.side_effect = mock_fingerprints
            
            result = await tracker.establish_baseline(audio_samples)
            
            assert result == True
            assert tracker.is_baseline_established == True
            assert tracker.voice_profile is not None
            assert tracker.voice_profile.sample_count == 5
            
    @pytest.mark.asyncio
    async def test_baseline_establishment_insufficient_quality(self):
        """Test baseline establishment with insufficient quality samples"""
        config = VoiceConsistencyConfig(
            enable_profile_persistence=False,
            baseline_sample_count=3,
            min_audio_quality=0.8  # High threshold
        )
        tracker = VoiceConsistencyTracker(config)
        await tracker.initialize()
        
        # Create mock audio samples
        audio_samples = [b"mock_audio"] * 5
        
        # Mock the feature extractor to return low quality
        with patch.object(tracker.feature_extractor, 'extract_voice_fingerprint') as mock_extract:
            low_quality_fingerprint = VoiceFingerprint(
                f0_mean=220.0,
                f0_std=15.0,
                f0_range=50.0,
                spectral_centroid_mean=1500.0,
                spectral_centroid_std=200.0,
                spectral_rolloff_mean=3000.0,
                spectral_bandwidth_mean=800.0,
                mfcc_mean=np.ones(13),
                mfcc_std=np.ones(13),
                chroma_mean=np.ones(12),
                spectral_contrast_mean=np.ones(7),
                rms_energy_mean=0.15,
                rms_energy_std=0.02,
                zero_crossing_rate_mean=0.05,
                tempo=120.0,
                quality_score=0.5,  # Below minimum
                sample_count=1
            )
            
            mock_extract.return_value = low_quality_fingerprint
            
            result = await tracker.establish_baseline(audio_samples)
            
            assert result == False
            assert tracker.is_baseline_established == False
            
    @pytest.mark.asyncio
    async def test_consistency_analysis_not_ready(self):
        """Test consistency analysis when not ready"""
        tracker = VoiceConsistencyTracker()
        
        audio_data = b"mock_audio"
        result = await tracker.analyze_consistency(audio_data)
        
        assert result is None
        
    @pytest.mark.asyncio
    async def test_consistency_analysis_short_audio(self):
        """Test consistency analysis with short audio"""
        config = VoiceConsistencyConfig(
            enable_profile_persistence=False,
            min_audio_length_ms=2000  # 2 seconds minimum
        )
        tracker = VoiceConsistencyTracker(config)
        await tracker.initialize()
        tracker.is_baseline_established = True
        
        # Create short audio (1 second)
        short_audio = np.random.rand(16000).astype(np.int16).tobytes()
        
        result = await tracker.analyze_consistency(short_audio)
        
        assert result is None
        
    @pytest.mark.asyncio
    async def test_processing_metrics(self):
        """Test processing metrics collection"""
        tracker = VoiceConsistencyTracker()
        tracker.is_initialized = True
        tracker.is_baseline_established = True
        tracker.total_analyses = 10
        tracker.average_processing_time = 45.5
        
        # Mock voice profile
        mock_profile = Mock()
        mock_profile.sample_count = 25
        mock_profile.creation_timestamp = time.time() - 3600  # 1 hour ago
        tracker.voice_profile = mock_profile
        
        metrics = tracker.get_processing_metrics()
        
        assert metrics["total_analyses"] == 10
        assert metrics["average_processing_time_ms"] == 45.5
        assert metrics["is_initialized"] == True
        assert metrics["is_baseline_established"] == True
        assert metrics["profile_sample_count"] == 25
        assert "baseline_age_hours" in metrics
        
    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test cleanup functionality"""
        config = VoiceConsistencyConfig(enable_profile_persistence=False)
        tracker = VoiceConsistencyTracker(config)
        
        await tracker.cleanup()
        
        # Should not raise any exceptions


class TestFactoryFunctions:
    """Test factory functions"""
    
    def test_create_voice_consistency_tracker(self):
        """Test factory function for voice consistency tracker"""
        config = VoiceConsistencyConfig(sample_rate=22050)
        tracker = create_voice_consistency_tracker(config)
        
        assert isinstance(tracker, VoiceConsistencyTracker)
        assert tracker.config == config
        
    def test_create_voice_consistency_tracker_default(self):
        """Test factory function with default config"""
        tracker = create_voice_consistency_tracker()
        
        assert isinstance(tracker, VoiceConsistencyTracker)
        assert isinstance(tracker.config, VoiceConsistencyConfig)
        
    def test_create_default_voice_config(self):
        """Test default config factory function"""
        config = create_default_voice_config()
        
        assert isinstance(config, VoiceConsistencyConfig)
        assert config.sample_rate == 16000
        assert config.chunk_duration_ms == 2000


class TestIntegrationScenarios:
    """Test integration scenarios"""
    
    def test_consistency_level_determination(self):
        """Test consistency level determination"""
        config = VoiceConsistencyConfig()
        tracker = VoiceConsistencyTracker(config)
        
        # Test different similarity scores
        assert tracker._determine_consistency_level(0.98) == VoiceConsistencyLevel.EXCELLENT
        assert tracker._determine_consistency_level(0.90) == VoiceConsistencyLevel.GOOD
        assert tracker._determine_consistency_level(0.75) == VoiceConsistencyLevel.MODERATE
        assert tracker._determine_consistency_level(0.60) == VoiceConsistencyLevel.POOR
        assert tracker._determine_consistency_level(0.30) == VoiceConsistencyLevel.INCONSISTENT
        
    def test_feature_differences_calculation(self):
        """Test feature differences calculation"""
        tracker = VoiceConsistencyTracker()
        
        baseline = VoiceFingerprint(
            f0_mean=200.0,
            f0_std=10.0,
            f0_range=40.0,
            spectral_centroid_mean=1500.0,
            spectral_centroid_std=200.0,
            spectral_rolloff_mean=3000.0,
            spectral_bandwidth_mean=800.0,
            mfcc_mean=np.ones(13),
            mfcc_std=np.ones(13),
            chroma_mean=np.ones(12),
            spectral_contrast_mean=np.ones(7),
            rms_energy_mean=0.10,
            rms_energy_std=0.02,
            zero_crossing_rate_mean=0.05,
            tempo=120.0,
            quality_score=0.85,
            sample_count=1
        )
        
        current = VoiceFingerprint(
            f0_mean=220.0,  # 10% increase
            f0_std=10.0,
            f0_range=40.0,
            spectral_centroid_mean=1650.0,  # 10% increase
            spectral_centroid_std=200.0,
            spectral_rolloff_mean=3000.0,
            spectral_bandwidth_mean=800.0,
            mfcc_mean=np.ones(13),
            mfcc_std=np.ones(13),
            chroma_mean=np.ones(12),
            spectral_contrast_mean=np.ones(7),
            rms_energy_mean=0.12,  # 20% increase
            rms_energy_std=0.02,
            zero_crossing_rate_mean=0.05,
            tempo=120.0,
            quality_score=0.85,
            sample_count=1
        )
        
        differences = tracker._calculate_feature_differences(baseline, current)
        
        assert 'pitch_change_percent' in differences
        assert 'energy_change_percent' in differences
        assert 'spectral_change_percent' in differences
        assert differences['pitch_change_percent'] == 10.0
        assert differences['energy_change_percent'] == 20.0
        assert differences['spectral_change_percent'] == 10.0
        
    def test_recommendations_generation(self):
        """Test recommendations generation"""
        tracker = VoiceConsistencyTracker()
        
        # Test poor consistency with multiple drifts
        recommendations = tracker._generate_recommendations(
            VoiceConsistencyLevel.POOR,
            [VoiceDriftType.PITCH_DRIFT, VoiceDriftType.ENERGY_DRIFT]
        )
        
        assert len(recommendations) == 3  # One for poor consistency + two for drifts
        assert any("poor" in rec.lower() for rec in recommendations)
        assert any("pitch" in rec.lower() for rec in recommendations)
        assert any("energy" in rec.lower() for rec in recommendations)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 