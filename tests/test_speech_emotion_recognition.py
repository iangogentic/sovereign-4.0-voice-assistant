"""
Test suite for Speech Emotion Recognition module

Tests emotion recognition functionality including:
- SpeechEmotionRecognizer initialization and model loading
- Real-time audio processing and emotion detection
- Emotion trajectory tracking and temporal smoothing
- Audio quality analysis and feature extraction
- Performance optimization and error handling
- Integration with EmotionPreserver system
"""

import pytest
import asyncio
import numpy as np
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import asdict

# Import the modules to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from assistant.speech_emotion_recognition import (
    SpeechEmotionRecognizer,
    EmotionRecognitionConfig,
    EmotionResult,
    EmotionLabel,
    ProcessingMode,
    AudioFeatures,
    AudioQualityAnalyzer,
    FeatureExtractor,
    EmotionTrajectoryTracker,
    create_speech_emotion_recognizer,
    create_default_config
)


class TestEmotionRecognitionConfig:
    """Test emotion recognition configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = EmotionRecognitionConfig()
        
        assert config.model_name == "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
        assert config.device == "auto"
        assert config.sample_rate == 16000
        assert config.chunk_duration_ms == 2000
        assert config.overlap_ratio == 0.5
        assert config.processing_mode == ProcessingMode.REAL_TIME
        assert config.min_confidence_threshold == 0.6
        assert config.emotion_smoothing_factor == 0.7
        assert config.enable_temporal_smoothing == True
        assert config.enable_multi_modal_features == True
        assert config.enable_quantization == True
        
    def test_custom_config(self):
        """Test custom configuration creation"""
        config = EmotionRecognitionConfig(
            device="cpu",
            chunk_duration_ms=3000,
            min_confidence_threshold=0.8,
            enable_temporal_smoothing=False
        )
        
        assert config.device == "cpu"
        assert config.chunk_duration_ms == 3000
        assert config.min_confidence_threshold == 0.8
        assert config.enable_temporal_smoothing == False


class TestEmotionResult:
    """Test emotion result data structure"""
    
    def test_emotion_result_creation(self):
        """Test emotion result creation"""
        emotion_scores = {
            EmotionLabel.HAPPY: 0.7,
            EmotionLabel.NEUTRAL: 0.2,
            EmotionLabel.SAD: 0.1
        }
        
        result = EmotionResult(
            primary_emotion=EmotionLabel.HAPPY,
            confidence=0.7,
            emotion_scores=emotion_scores,
            arousal=0.6,
            valence=0.8,
            processing_time_ms=45.2,
            audio_quality_score=0.85
        )
        
        assert result.primary_emotion == EmotionLabel.HAPPY
        assert result.confidence == 0.7
        assert result.arousal == 0.6
        assert result.valence == 0.8
        assert result.processing_time_ms == 45.2
        assert result.audio_quality_score == 0.85
        assert isinstance(result.timestamp, float)


class TestAudioQualityAnalyzer:
    """Test audio quality analysis"""
    
    def test_high_quality_audio(self):
        """Test high quality audio detection"""
        # Generate clean sine wave
        sample_rate = 16000
        duration = 1.0
        frequency = 440  # A4 note
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        quality_score = AudioQualityAnalyzer.calculate_quality_score(audio, sample_rate)
        
        assert 0.0 <= quality_score <= 1.0
        assert quality_score > 0.5  # Should be reasonably high quality
        
    def test_low_quality_audio(self):
        """Test low quality audio detection"""
        # Generate noisy audio
        sample_rate = 16000
        duration = 1.0
        audio = np.random.normal(0, 0.1, int(sample_rate * duration))
        
        quality_score = AudioQualityAnalyzer.calculate_quality_score(audio, sample_rate)
        
        assert 0.0 <= quality_score <= 1.0
        
    def test_silent_audio(self):
        """Test silent audio handling"""
        audio = np.zeros(16000)  # 1 second of silence
        
        quality_score = AudioQualityAnalyzer.calculate_quality_score(audio, 16000)
        
        assert quality_score == 0.0
        
    def test_clipped_audio(self):
        """Test clipped audio detection"""
        # Create clipped audio
        audio = np.ones(16000)  # Fully clipped
        
        quality_score = AudioQualityAnalyzer.calculate_quality_score(audio, 16000)
        
        assert quality_score < 0.5  # Should penalize clipping


class TestFeatureExtractor:
    """Test audio feature extraction"""
    
    def test_feature_extractor_initialization(self):
        """Test feature extractor initialization"""
        config = EmotionRecognitionConfig()
        extractor = FeatureExtractor(config)
        
        assert extractor.config == config
        assert extractor.sample_rate == config.sample_rate
        
    @patch('assistant.speech_emotion_recognition.librosa')
    def test_feature_extraction_with_librosa(self, mock_librosa):
        """Test feature extraction with librosa"""
        config = EmotionRecognitionConfig()
        extractor = FeatureExtractor(config)
        
        # Mock librosa functions
        mock_librosa.pyin.return_value = (
            np.array([220.0, 220.0, 220.0]), 
            np.array([True, True, True]),
            np.array([0.9, 0.9, 0.9])
        )
        mock_librosa.feature.rms.return_value = np.array([[0.1, 0.15, 0.12]])
        mock_librosa.feature.spectral_centroid.return_value = np.array([[1000, 1100, 1050]])
        mock_librosa.feature.spectral_rolloff.return_value = np.array([[2000, 2100, 2050]])
        mock_librosa.feature.zero_crossing_rate.return_value = np.array([[0.05, 0.06, 0.055]])
        mock_librosa.feature.mfcc.return_value = np.random.rand(13, 10)
        mock_librosa.feature.chroma_stft.return_value = np.random.rand(12, 10)
        mock_librosa.beat.beat_track.return_value = (120.0, np.array([0, 1, 2]))
        mock_librosa.feature.spectral_contrast.return_value = np.random.rand(7, 10)
        
        # Generate test audio
        audio = np.random.rand(16000)
        
        features = extractor.extract_features(audio)
        
        assert isinstance(features, AudioFeatures)
        assert features.f0_mean == 220.0
        assert features.energy_mean > 0
        assert features.spectral_centroid_mean > 0
        assert features.mfcc.shape[0] == 13
        assert features.chroma.shape[0] == 12
        assert features.tempo == 120.0
        
    def test_basic_feature_extraction_fallback(self):
        """Test basic feature extraction without librosa"""
        config = EmotionRecognitionConfig()
        extractor = FeatureExtractor(config)
        
        # Generate test audio
        audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        
        with patch('assistant.speech_emotion_recognition.librosa', side_effect=ImportError):
            features = extractor._extract_basic_features(audio)
            
            assert isinstance(features, AudioFeatures)
            assert features.f0_mean == 0.0  # Not calculated in basic mode
            assert features.energy_mean > 0
            assert features.spectral_centroid_mean != 0
            assert features.zero_crossing_rate_mean > 0
            assert features.tempo == 120.0  # Default
            assert features.mfcc.shape == (config.n_mfcc, 1)


class TestEmotionTrajectoryTracker:
    """Test emotion trajectory tracking"""
    
    def test_tracker_initialization(self):
        """Test tracker initialization"""
        tracker = EmotionTrajectoryTracker(smoothing_factor=0.8, history_length=5)
        
        assert tracker.smoothing_factor == 0.8
        assert tracker.emotion_history.maxlen == 5
        assert tracker.previous_emotion is None
        
    def test_first_emotion_update(self):
        """Test first emotion update"""
        tracker = EmotionTrajectoryTracker()
        
        emotion_scores = {
            EmotionLabel.HAPPY: 0.7,
            EmotionLabel.NEUTRAL: 0.2,
            EmotionLabel.SAD: 0.1
        }
        
        smoothed = tracker.update_emotion(emotion_scores, 0.7)
        
        assert smoothed == emotion_scores
        assert tracker.previous_emotion == emotion_scores
        
    def test_high_confidence_emotion_update(self):
        """Test high confidence emotion update"""
        tracker = EmotionTrajectoryTracker()
        
        # Set initial emotion
        initial_emotion = {
            EmotionLabel.HAPPY: 0.7,
            EmotionLabel.NEUTRAL: 0.2,
            EmotionLabel.SAD: 0.1
        }
        tracker.update_emotion(initial_emotion, 0.7)
        
        # Update with high confidence
        new_emotion = {
            EmotionLabel.ANGRY: 0.8,
            EmotionLabel.NEUTRAL: 0.1,
            EmotionLabel.SAD: 0.1
        }
        
        smoothed = tracker.update_emotion(new_emotion, 0.9)
        
        # High confidence should use new emotion as-is
        assert smoothed == new_emotion
        
    def test_low_confidence_emotion_smoothing(self):
        """Test low confidence emotion smoothing"""
        tracker = EmotionTrajectoryTracker(smoothing_factor=0.7)
        
        # Set initial emotion
        initial_emotion = {
            EmotionLabel.HAPPY: 0.8,
            EmotionLabel.NEUTRAL: 0.1,
            EmotionLabel.SAD: 0.1
        }
        tracker.update_emotion(initial_emotion, 0.8)
        
        # Update with low confidence
        new_emotion = {
            EmotionLabel.ANGRY: 0.6,
            EmotionLabel.HAPPY: 0.2,
            EmotionLabel.NEUTRAL: 0.1,
            EmotionLabel.SAD: 0.1
        }
        
        smoothed = tracker.update_emotion(new_emotion, 0.4)
        
        # Should be heavily smoothed toward previous emotion
        assert smoothed[EmotionLabel.HAPPY] > new_emotion[EmotionLabel.HAPPY]
        assert smoothed[EmotionLabel.ANGRY] < new_emotion[EmotionLabel.ANGRY]
        
    def test_emotion_trend_detection(self):
        """Test emotion trend detection"""
        tracker = EmotionTrajectoryTracker()
        
        # Add multiple emotions with happy trend
        for _ in range(5):
            emotion_scores = {
                EmotionLabel.HAPPY: 0.6,
                EmotionLabel.NEUTRAL: 0.3,
                EmotionLabel.SAD: 0.1
            }
            tracker.update_emotion(emotion_scores, 0.7)
        
        trend = tracker.get_emotion_trend()
        assert trend == EmotionLabel.HAPPY
        
    def test_emotion_trend_insufficient_data(self):
        """Test emotion trend with insufficient data"""
        tracker = EmotionTrajectoryTracker()
        
        # Add only one emotion
        emotion_scores = {EmotionLabel.HAPPY: 0.8, EmotionLabel.NEUTRAL: 0.2}
        tracker.update_emotion(emotion_scores, 0.7)
        
        trend = tracker.get_emotion_trend()
        assert trend is None


class TestSpeechEmotionRecognizer:
    """Test main speech emotion recognizer"""
    
    def test_recognizer_initialization(self):
        """Test recognizer initialization"""
        config = EmotionRecognitionConfig(device="cpu")
        recognizer = SpeechEmotionRecognizer(config)
        
        assert recognizer.config == config
        assert not recognizer.is_initialized
        assert recognizer.classifier is None
        assert len(recognizer.audio_buffer) == 0
        
    def test_default_config_initialization(self):
        """Test initialization with default config"""
        recognizer = SpeechEmotionRecognizer()
        
        assert isinstance(recognizer.config, EmotionRecognitionConfig)
        
    @pytest.mark.asyncio
    @patch('assistant.speech_emotion_recognition.EncoderClassifier')
    async def test_successful_initialization(self, mock_encoder_classifier):
        """Test successful initialization"""
        config = EmotionRecognitionConfig(device="cpu", enable_quantization=False)
        recognizer = SpeechEmotionRecognizer(config)
        
        # Mock the classifier
        mock_classifier = Mock()
        mock_encoder_classifier.from_hparams.return_value = mock_classifier
        
        result = await recognizer.initialize()
        
        assert result == True
        assert recognizer.is_initialized == True
        assert recognizer.device == "cpu"
        assert recognizer.classifier == mock_classifier
        
    @pytest.mark.asyncio
    async def test_initialization_failure(self):
        """Test initialization failure"""
        recognizer = SpeechEmotionRecognizer()
        
        with patch('assistant.speech_emotion_recognition.EncoderClassifier', side_effect=ImportError("speechbrain not found")):
            result = await recognizer.initialize()
            
            assert result == False
            assert recognizer.is_initialized == False
            
    @pytest.mark.asyncio
    @patch('torch.cuda.is_available')
    async def test_device_auto_detection(self, mock_cuda_available):
        """Test automatic device detection"""
        mock_cuda_available.return_value = True
        
        config = EmotionRecognitionConfig(device="auto")
        recognizer = SpeechEmotionRecognizer(config)
        
        with patch('assistant.speech_emotion_recognition.EncoderClassifier'):
            await recognizer.initialize()
            
            assert recognizer.device == "cuda"
            
    @pytest.mark.asyncio
    async def test_quantization_application(self):
        """Test quantization application"""
        config = EmotionRecognitionConfig(device="cpu", enable_quantization=True)
        recognizer = SpeechEmotionRecognizer(config)
        
        # Mock classifier with encoder
        mock_encoder = Mock()
        mock_classifier = Mock()
        mock_classifier.mods.encoder = mock_encoder
        
        with patch('assistant.speech_emotion_recognition.EncoderClassifier.from_hparams', return_value=mock_classifier):
            with patch('torch.quantization.quantize_dynamic') as mock_quantize:
                mock_quantize.return_value = mock_encoder
                
                await recognizer.initialize()
                
                mock_quantize.assert_called_once()
                
    @pytest.mark.asyncio
    async def test_audio_processing_not_initialized(self):
        """Test audio processing when not initialized"""
        recognizer = SpeechEmotionRecognizer()
        
        audio_data = np.random.randint(-32768, 32767, 16000, dtype=np.int16).tobytes()
        
        result = await recognizer.process_audio_chunk(audio_data)
        
        assert result is None
        
    @pytest.mark.asyncio
    async def test_audio_processing_insufficient_data(self):
        """Test audio processing with insufficient data"""
        recognizer = SpeechEmotionRecognizer()
        recognizer.is_initialized = True
        
        # Small audio chunk (less than minimum)
        audio_data = np.random.randint(-32768, 32767, 8000, dtype=np.int16).tobytes()
        
        result = await recognizer.process_audio_chunk(audio_data)
        
        assert result is None
        
    @pytest.mark.asyncio
    async def test_audio_processing_low_quality(self):
        """Test audio processing with low quality audio"""
        recognizer = SpeechEmotionRecognizer()
        recognizer.is_initialized = True
        
        # Mock quality analyzer to return low quality
        with patch.object(recognizer.quality_analyzer, 'calculate_quality_score', return_value=0.3):
            audio_data = np.random.randint(-32768, 32767, 32000, dtype=np.int16).tobytes()
            
            result = await recognizer.process_audio_chunk(audio_data)
            
            assert result is None
            
    @pytest.mark.asyncio
    @patch('torch.tensor')
    @patch('torch.nn.functional.softmax')
    async def test_successful_audio_processing(self, mock_softmax, mock_tensor):
        """Test successful audio processing"""
        config = EmotionRecognitionConfig(enable_temporal_smoothing=False, enable_multi_modal_features=False)
        recognizer = SpeechEmotionRecognizer(config)
        recognizer.is_initialized = True
        
        # Mock classifier
        mock_classifier = Mock()
        mock_prediction = Mock()
        mock_prediction[0].squeeze.return_value = Mock()
        mock_classifier.classify_batch.return_value = mock_prediction
        recognizer.classifier = mock_classifier
        
        # Mock tensor operations
        mock_audio_tensor = Mock()
        mock_audio_tensor.unsqueeze.return_value = mock_audio_tensor
        mock_audio_tensor.__truediv__ = Mock(return_value=mock_audio_tensor)
        mock_tensor.return_value = mock_audio_tensor
        
        # Mock softmax output
        mock_probs = [0.1, 0.7, 0.1, 0.1]  # Happy emotion (index 1)
        mock_softmax.return_value = mock_probs
        
        # Mock quality analyzer
        with patch.object(recognizer.quality_analyzer, 'calculate_quality_score', return_value=0.8):
            audio_data = np.random.randint(-32768, 32767, 32000, dtype=np.int16).tobytes()
            
            result = await recognizer.process_audio_chunk(audio_data)
            
            assert result is not None
            assert isinstance(result, EmotionResult)
            assert result.primary_emotion == EmotionLabel.HAPPY
            assert result.confidence == 0.7
            assert result.audio_quality_score == 0.8
            
    @pytest.mark.asyncio
    async def test_processing_metrics(self):
        """Test processing metrics collection"""
        recognizer = SpeechEmotionRecognizer()
        recognizer.is_initialized = True
        recognizer.device = "cpu"
        recognizer.total_predictions = 5
        recognizer.average_processing_time = 45.2
        
        metrics = recognizer.get_processing_metrics()
        
        assert metrics["total_predictions"] == 5
        assert metrics["average_processing_time_ms"] == 45.2
        assert metrics["device"] == "cpu"
        assert metrics["is_initialized"] == True
        assert "buffer_size" in metrics
        
    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test cleanup functionality"""
        recognizer = SpeechEmotionRecognizer()
        recognizer.audio_buffer.extend([1, 2, 3, 4, 5])
        
        await recognizer.cleanup()
        
        assert len(recognizer.audio_buffer) == 0
        
    def test_arousal_valence_calculation(self):
        """Test arousal and valence calculation"""
        recognizer = SpeechEmotionRecognizer()
        
        emotion_scores = {
            EmotionLabel.HAPPY: 0.8,
            EmotionLabel.NEUTRAL: 0.2
        }
        
        arousal, valence = recognizer._calculate_arousal_valence(emotion_scores)
        
        assert 0.0 <= arousal <= 1.0
        assert 0.0 <= valence <= 1.0
        # Happy emotion should have high valence
        assert valence > 0.5


class TestFactoryFunctions:
    """Test factory functions"""
    
    def test_create_speech_emotion_recognizer(self):
        """Test factory function for speech emotion recognizer"""
        config = EmotionRecognitionConfig(device="cpu")
        recognizer = create_speech_emotion_recognizer(config)
        
        assert isinstance(recognizer, SpeechEmotionRecognizer)
        assert recognizer.config == config
        
    def test_create_speech_emotion_recognizer_default(self):
        """Test factory function with default config"""
        recognizer = create_speech_emotion_recognizer()
        
        assert isinstance(recognizer, SpeechEmotionRecognizer)
        assert isinstance(recognizer.config, EmotionRecognitionConfig)
        
    def test_create_default_config(self):
        """Test default config factory function"""
        config = create_default_config()
        
        assert isinstance(config, EmotionRecognitionConfig)
        assert config.device == "auto"
        assert config.sample_rate == 16000


class TestErrorHandling:
    """Test error handling scenarios"""
    
    @pytest.mark.asyncio
    async def test_audio_processing_exception(self):
        """Test audio processing exception handling"""
        recognizer = SpeechEmotionRecognizer()
        recognizer.is_initialized = True
        
        # Mock quality analyzer to raise exception
        with patch.object(recognizer.quality_analyzer, 'calculate_quality_score', side_effect=Exception("Test error")):
            audio_data = b"invalid audio data"
            
            result = await recognizer.process_audio_chunk(audio_data)
            
            assert result is None
            
    @pytest.mark.asyncio
    async def test_emotion_prediction_exception(self):
        """Test emotion prediction exception handling"""
        recognizer = SpeechEmotionRecognizer()
        recognizer.is_initialized = True
        
        # Mock classifier to raise exception
        mock_classifier = Mock()
        mock_classifier.classify_batch.side_effect = Exception("Prediction failed")
        recognizer.classifier = mock_classifier
        
        with patch.object(recognizer.quality_analyzer, 'calculate_quality_score', return_value=0.8):
            audio_data = np.random.randint(-32768, 32767, 32000, dtype=np.int16).tobytes()
            
            result = await recognizer.process_audio_chunk(audio_data)
            
            assert result is None


class TestIntegrationScenarios:
    """Test integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_real_time_processing_simulation(self):
        """Test simulated real-time processing"""
        config = EmotionRecognitionConfig(
            chunk_duration_ms=1000,
            enable_temporal_smoothing=True
        )
        recognizer = SpeechEmotionRecognizer(config)
        recognizer.is_initialized = True
        
        # Mock successful processing
        mock_result = EmotionResult(
            primary_emotion=EmotionLabel.HAPPY,
            confidence=0.8,
            emotion_scores={EmotionLabel.HAPPY: 0.8, EmotionLabel.NEUTRAL: 0.2},
            arousal=0.7,
            valence=0.9,
            processing_time_ms=50.0,
            audio_quality_score=0.8
        )
        
        with patch.object(recognizer, '_predict_emotion', return_value=mock_result):
            with patch.object(recognizer.quality_analyzer, 'calculate_quality_score', return_value=0.8):
                
                # Process multiple chunks
                for i in range(3):
                    audio_data = np.random.randint(-32768, 32767, 16000, dtype=np.int16).tobytes()
                    result = await recognizer.process_audio_chunk(audio_data)
                    
                    if result:  # May be None for first few chunks
                        assert isinstance(result, EmotionResult)
                        assert result.primary_emotion == EmotionLabel.HAPPY
                        
    def test_emotion_consistency_tracking(self):
        """Test emotion consistency across multiple predictions"""
        tracker = EmotionTrajectoryTracker(smoothing_factor=0.8)
        
        # Simulate consistent happy emotion
        emotions = []
        for confidence in [0.9, 0.8, 0.7, 0.85, 0.9]:
            emotion_scores = {
                EmotionLabel.HAPPY: confidence,
                EmotionLabel.NEUTRAL: 1 - confidence
            }
            smoothed = tracker.update_emotion(emotion_scores, confidence)
            emotions.append(smoothed)
        
        # All predictions should maintain happy as primary emotion
        for emotion in emotions:
            primary = max(emotion, key=emotion.get)
            assert primary == EmotionLabel.HAPPY
            
    @pytest.mark.asyncio
    async def test_performance_metrics_tracking(self):
        """Test performance metrics tracking"""
        recognizer = SpeechEmotionRecognizer()
        recognizer.is_initialized = True
        
        # Simulate processing
        recognizer.total_predictions = 0
        recognizer.average_processing_time = 0.0
        
        # Mock processing times
        processing_times = [45.0, 52.0, 38.0, 41.0, 49.0]
        
        for i, processing_time in enumerate(processing_times):
            recognizer.total_predictions += 1
            recognizer.average_processing_time = (
                (recognizer.average_processing_time * i + processing_time) / (i + 1)
            )
        
        metrics = recognizer.get_processing_metrics()
        
        assert metrics["total_predictions"] == 5
        assert abs(metrics["average_processing_time_ms"] - np.mean(processing_times)) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 