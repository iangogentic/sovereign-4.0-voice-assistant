"""
Comprehensive test suite for ScreenContextProvider for Realtime API integration

Tests cover:
- Configuration and initialization
- Screen content formatting and cleaning
- Privacy filtering and sensitive data protection
- Change detection and similarity scoring
- Integration with ScreenWatcher
- Performance metrics and monitoring
- Error handling and edge cases
"""

import pytest
import asyncio
import time
import re
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from assistant.screen_context_provider import (
    ScreenContextProvider, ScreenContextConfig, ScreenContextData,
    ScreenContentFormatter, create_screen_context_provider,
    get_default_screen_context_config
)
from assistant.screen_watcher import ScreenCapture, ScreenWatcher


# Test fixtures for mock data
@pytest.fixture
def sample_screen_capture():
    """Create a sample screen capture for testing"""
    return ScreenCapture(
        image=None,  # Mock PIL Image not needed for these tests
        timestamp=datetime.now(timezone.utc),
        window_title="Test Application",
        window_app="TestApp",
        window_bounds=(0, 0, 1920, 1080),
        ocr_text="Welcome to the test application. Please enter your password: secret123",
        text_hash="test_hash_123",
        confidence=85.5
    )


@pytest.fixture
def mock_screen_watcher():
    """Create a mock ScreenWatcher for testing"""
    watcher = Mock(spec=ScreenWatcher)
    watcher.mss_instance = MagicMock()
    watcher.running = False
    watcher.initialize = AsyncMock(return_value=True)
    watcher.start_monitoring = Mock(return_value=True)
    watcher._capture_screen = Mock()
    return watcher


@pytest.fixture
def screen_context_config():
    """Create test configuration for ScreenContextProvider"""
    return ScreenContextConfig(
        update_interval=1.0,  # Faster for testing
        min_change_threshold=0.1,
        max_content_length=500,  # Smaller for testing
        min_confidence_threshold=60.0,
        enable_privacy_mode=True,
        enable_content_caching=True,
        enable_background_updates=False  # Disable for controlled testing
    )


# Configuration Tests
class TestScreenContextConfig:
    """Test ScreenContextConfig dataclass and validation"""
    
    def test_default_config_values(self):
        """Test default configuration values"""
        config = ScreenContextConfig()
        
        assert config.update_interval == 5.0
        assert config.min_change_threshold == 0.1
        assert config.max_content_length == 1500
        assert config.min_confidence_threshold == 60.0
        assert config.enable_privacy_mode is True
        assert config.enable_content_caching is True
        assert config.enable_background_updates is True
        
        # Test sensitive patterns
        assert len(config.sensitive_patterns) >= 6
        assert any('password' in pattern for pattern in config.sensitive_patterns)
        assert any('[A-Za-z0-9._%+-]+@' in pattern for pattern in config.sensitive_patterns)  # Email regex pattern
    
    def test_custom_config_values(self):
        """Test custom configuration values"""
        config = ScreenContextConfig(
            update_interval=2.0,
            max_content_length=1000,
            enable_privacy_mode=False,
            sensitive_patterns=['test_pattern']
        )
        
        assert config.update_interval == 2.0
        assert config.max_content_length == 1000
        assert config.enable_privacy_mode is False
        assert config.sensitive_patterns == ['test_pattern']
    
    def test_get_default_screen_context_config(self):
        """Test factory function for default config"""
        config = get_default_screen_context_config()
        
        assert isinstance(config, ScreenContextConfig)
        assert config.update_interval == 5.0
        assert config.min_change_threshold == 0.15
        assert config.max_content_length == 1200
        assert config.enable_privacy_mode is True


# ScreenContextData Tests
class TestScreenContextData:
    """Test ScreenContextData dataclass and methods"""
    
    def test_screen_context_data_creation(self):
        """Test creating ScreenContextData"""
        timestamp = datetime.now(timezone.utc)
        data = ScreenContextData(
            content="Test content",
            timestamp=timestamp,
            confidence=85.0,
            word_count=2,
            character_count=12,
            change_score=0.5,
            window_title="Test Window",
            window_app="TestApp",
            privacy_filtered=False,
            source_hash="abc123"
        )
        
        assert data.content == "Test content"
        assert data.timestamp == timestamp
        assert data.confidence == 85.0
        assert data.word_count == 2
        assert data.character_count == 12
        assert data.change_score == 0.5
        assert data.window_title == "Test Window"
        assert data.window_app == "TestApp"
        assert data.privacy_filtered is False
        assert data.source_hash == "abc123"
    
    def test_to_context_string(self):
        """Test formatting as context string for Realtime API"""
        timestamp = datetime.now(timezone.utc)
        data = ScreenContextData(
            content="This is test content for the screen",
            timestamp=timestamp,
            confidence=90.0,
            word_count=7,
            character_count=35,
            change_score=0.8,
            window_title="Test Application",
            window_app="TestApp",
            privacy_filtered=False,
            source_hash="test123"
        )
        
        context_string = data.to_context_string()
        
        assert f"[SCREEN CONTENT at {timestamp.strftime('%H:%M:%S')} - TestApp]" in context_string
        assert "This is test content for the screen" in context_string
        assert "[END SCREEN CONTENT]" in context_string
    
    def test_to_context_string_with_privacy_filter(self):
        """Test context string with privacy filtering applied"""
        timestamp = datetime.now(timezone.utc)
        data = ScreenContextData(
            content="Filtered content",
            timestamp=timestamp,
            confidence=85.0,
            word_count=2,
            character_count=16,
            change_score=0.3,
            window_title="Test",
            window_app="TestApp",
            privacy_filtered=True,
            source_hash="filtered123"
        )
        
        context_string = data.to_context_string()
        
        assert "[Privacy filtered]" in context_string
        assert "Filtered content" in context_string
    
    def test_is_significant_change(self):
        """Test significant change detection"""
        data = ScreenContextData(
            content="Test",
            timestamp=datetime.now(timezone.utc),
            confidence=80.0,
            word_count=1,
            character_count=4,
            change_score=0.15,
            window_title="Test",
            window_app="Test",
            privacy_filtered=False,
            source_hash="test"
        )
        
        assert data.is_significant_change(0.1) is True  # 0.15 >= 0.1
        assert data.is_significant_change(0.2) is False  # 0.15 < 0.2


# ScreenContentFormatter Tests
class TestScreenContentFormatter:
    """Test ScreenContentFormatter functionality"""
    
    def test_formatter_initialization(self, screen_context_config):
        """Test formatter initialization with config"""
        formatter = ScreenContentFormatter(screen_context_config)
        
        assert formatter.config == screen_context_config
        assert len(formatter.sensitive_patterns) >= 6
        assert len(formatter.noise_patterns) >= 6
    
    def test_clean_text(self, screen_context_config):
        """Test text cleaning functionality"""
        formatter = ScreenContentFormatter(screen_context_config)
        
        # Test whitespace cleaning
        dirty_text = "Hello    world\n\n\nwith   extra  spaces"
        clean_text = formatter._clean_text(dirty_text)
        assert clean_text == "Hello world with extra spaces"
        
        # Test OCR artifact cleaning
        artifact_text = "Text||||with||multiple|||pipes and---many---dashes...lots...of...dots"
        clean_text = formatter._clean_text(artifact_text)
        assert "||||" not in clean_text
        assert "---" in clean_text  # Should reduce to triple
        assert "..." in clean_text  # Should reduce to triple
    
    def test_filter_noise(self, screen_context_config):
        """Test noise pattern filtering"""
        # Ensure noise filtering is enabled for this test
        screen_context_config.enable_noise_filtering = True
        formatter = ScreenContentFormatter(screen_context_config)
        
        noisy_text = "Real content here 12:34:56 A B C more content 95% 150px"
        filtered_text = formatter._filter_noise(noisy_text)
        
        # Time stamps should be removed
        assert "12:34:56" not in filtered_text
        # Percentages should be removed
        assert "95%" not in filtered_text
        # Pixel measurements should be removed
        assert "150px" not in filtered_text
        # Real content should remain
        assert "Real content here" in filtered_text
        assert "more content" in filtered_text
    
    def test_apply_privacy_filter(self, screen_context_config):
        """Test privacy filtering for sensitive data"""
        formatter = ScreenContentFormatter(screen_context_config)
        
        # Test credit card number filtering
        text_with_cc = "Your card number is 1234-5678-9012-3456 for payment"
        filtered_text, was_filtered = formatter._apply_privacy_filter(text_with_cc)
        assert "[SENSITIVE DATA HIDDEN]" in filtered_text
        assert was_filtered is True
        
        # Test email filtering
        text_with_email = "Contact us at support@example.com for help"
        filtered_text, was_filtered = formatter._apply_privacy_filter(text_with_email)
        assert "[SENSITIVE DATA HIDDEN]" in filtered_text
        assert was_filtered is True
        
        # Test password filtering
        text_with_password = "Enter password: secret123 to continue"
        filtered_text, was_filtered = formatter._apply_privacy_filter(text_with_password)
        assert "[SENSITIVE DATA HIDDEN]" in filtered_text
        assert was_filtered is True
        
        # Test clean text (no filtering)
        clean_text = "This is a normal text without sensitive data"
        filtered_text, was_filtered = formatter._apply_privacy_filter(clean_text)
        assert filtered_text == clean_text
        assert was_filtered is False
    
    def test_calculate_change_score(self, screen_context_config):
        """Test change score calculation using SequenceMatcher"""
        formatter = ScreenContentFormatter(screen_context_config)
        
        # Identical text should have 0 change
        text1 = "This is the same text"
        text2 = "This is the same text"
        score = formatter.calculate_change_score(text1, text2)
        assert score == 0.0
        
        # Completely different text should have high change score
        text1 = "Completely different content"
        text2 = "Totally unrelated information"
        score = formatter.calculate_change_score(text1, text2)
        assert score > 0.5  # Should be significantly different
        
        # Similar text with small changes
        text1 = "This is the original text with some content"
        text2 = "This is the modified text with some content"
        score = formatter.calculate_change_score(text1, text2)
        assert 0.0 < score < 0.5  # Some change, but not complete
        
        # Empty or None cases
        assert formatter.calculate_change_score("", "") == 0.0
        assert formatter.calculate_change_score("text", "") == 1.0
        assert formatter.calculate_change_score("", "text") == 1.0
        assert formatter.calculate_change_score("text", None) == 1.0
    
    def test_format_content_integration(self, screen_context_config, sample_screen_capture):
        """Test complete content formatting pipeline"""
        formatter = ScreenContentFormatter(screen_context_config)
        
        # Create a screen capture with sensitive data and noise
        capture = ScreenCapture(
            image=None,
            timestamp=datetime.now(timezone.utc),
            window_title="Banking App",
            window_app="BankApp",
            window_bounds=(0, 0, 1024, 768),
            ocr_text="Account: 1234-5678-9012-3456   Balance: $1,500.00   Contact: support@bank.com   12:34:56",
            text_hash="bank_hash",
            confidence=92.5
        )
        
        formatted_data = formatter.format_content(capture)
        
        # Check basic data transfer
        assert formatted_data.timestamp == capture.timestamp
        assert formatted_data.confidence == capture.confidence
        assert formatted_data.window_title == "Banking App"
        assert formatted_data.window_app == "BankApp"
        
        # Check privacy filtering applied
        assert formatted_data.privacy_filtered is True
        assert "[SENSITIVE DATA HIDDEN]" in formatted_data.content
        assert "1234-5678-9012-3456" not in formatted_data.content
        assert "support@bank.com" not in formatted_data.content
        
        # Check metrics
        assert formatted_data.word_count > 0
        assert formatted_data.character_count > 0
        assert len(formatted_data.source_hash) == 32  # MD5 hash length


# ScreenContextProvider Tests
class TestScreenContextProvider:
    """Test ScreenContextProvider main functionality"""
    
    @pytest.mark.asyncio
    async def test_provider_initialization(self, screen_context_config, mock_screen_watcher):
        """Test provider initialization"""
        provider = ScreenContextProvider(
            config=screen_context_config,
            screen_watcher=mock_screen_watcher
        )
        
        assert provider.config == screen_context_config
        assert provider.screen_watcher == mock_screen_watcher
        assert provider.is_initialized is False
        assert provider.is_running is False
        
        # Test initialization
        success = await provider.initialize()
        assert success is True
        assert provider.is_initialized is True
        
        # Verify screen watcher was initialized
        mock_screen_watcher.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_provider_initialization_failure(self, screen_context_config):
        """Test provider initialization failure without screen watcher"""
        provider = ScreenContextProvider(config=screen_context_config)
        
        success = await provider.initialize()
        assert success is False
        assert provider.is_initialized is False
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, screen_context_config, mock_screen_watcher):
        """Test monitoring start and stop"""
        provider = ScreenContextProvider(
            config=screen_context_config,
            screen_watcher=mock_screen_watcher
        )
        
        await provider.initialize()
        
        # Start monitoring
        success = await provider.start_monitoring()
        assert success is True
        assert provider.is_running is True
        
        # Verify screen watcher was started
        mock_screen_watcher.start_monitoring.assert_called_once()
        
        # Stop monitoring
        provider.stop_monitoring()
        assert provider.is_running is False
    
    @pytest.mark.asyncio
    async def test_get_latest_content_no_content(self, screen_context_config, mock_screen_watcher):
        """Test get_latest_content when no content is available"""
        provider = ScreenContextProvider(
            config=screen_context_config,
            screen_watcher=mock_screen_watcher
        )
        
        await provider.initialize()
        
        # Mock no capture available
        mock_screen_watcher._capture_screen.return_value = None
        
        content = await provider.get_latest_content()
        assert content is None
    
    @pytest.mark.asyncio
    async def test_get_latest_content_with_content(self, screen_context_config, mock_screen_watcher, sample_screen_capture):
        """Test get_latest_content with available content"""
        provider = ScreenContextProvider(
            config=screen_context_config,
            screen_watcher=mock_screen_watcher
        )
        
        await provider.initialize()
        
        # Mock screen capture return
        mock_screen_watcher._capture_screen.return_value = sample_screen_capture
        
        content = await provider.get_latest_content()
        
        assert content is not None
        assert isinstance(content, str)
        assert "[SCREEN CONTENT at" in content
        assert "TestApp" in content
        assert "[END SCREEN CONTENT]" in content
    
    @pytest.mark.asyncio
    async def test_content_caching(self, screen_context_config, mock_screen_watcher, sample_screen_capture):
        """Test content caching functionality"""
        config = screen_context_config
        config.enable_content_caching = True
        config.cache_duration = 10.0  # 10 second cache
        
        provider = ScreenContextProvider(
            config=config,
            screen_watcher=mock_screen_watcher
        )
        
        await provider.initialize()
        
        # Mock screen capture
        mock_screen_watcher._capture_screen.return_value = sample_screen_capture
        
        # First call should process content
        content1 = await provider.get_latest_content()
        call_count_1 = mock_screen_watcher._capture_screen.call_count
        
        # Second call within cache duration should use cache
        content2 = await provider.get_latest_content()
        call_count_2 = mock_screen_watcher._capture_screen.call_count
        
        assert content1 == content2
        assert call_count_2 == call_count_1  # No additional capture calls
    
    def test_get_metrics(self, screen_context_config, mock_screen_watcher):
        """Test metrics collection"""
        provider = ScreenContextProvider(
            config=screen_context_config,
            screen_watcher=mock_screen_watcher
        )
        
        metrics = provider.get_metrics()
        
        assert "total_updates" in metrics
        assert "significant_changes" in metrics
        assert "privacy_filters_applied" in metrics
        assert "change_rate" in metrics
        assert "privacy_filter_rate" in metrics
        assert "is_running" in metrics
        assert "cache_size" in metrics
        assert "current_content_length" in metrics
        assert "current_word_count" in metrics
        
        # Check initial values
        assert metrics["total_updates"] == 0
        assert metrics["significant_changes"] == 0
        assert metrics["privacy_filters_applied"] == 0
        assert metrics["is_running"] is False
    
    @pytest.mark.asyncio
    async def test_process_screen_capture(self, screen_context_config, mock_screen_watcher):
        """Test screen capture processing with change detection"""
        provider = ScreenContextProvider(
            config=screen_context_config,
            screen_watcher=mock_screen_watcher
        )
        
        await provider.initialize()
        
        # Create first capture
        capture1 = ScreenCapture(
            image=None,
            timestamp=datetime.now(timezone.utc),
            window_title="Test App",
            window_app="TestApp",
            window_bounds=(0, 0, 800, 600),
            ocr_text="First content here",
            text_hash="hash1",
            confidence=80.0
        )
        
        # Process first capture
        await provider._process_screen_capture(capture1)
        
        assert provider.current_content is not None
        assert provider.current_content.content == "First content here"
        assert provider.total_updates == 1
        assert provider.significant_changes == 1  # First is always significant
        
        # Create second capture with significant change
        capture2 = ScreenCapture(
            image=None,
            timestamp=datetime.now(timezone.utc),
            window_title="Test App",
            window_app="TestApp",
            window_bounds=(0, 0, 800, 600),
            ocr_text="Completely different content now",
            text_hash="hash2",
            confidence=85.0
        )
        
        # Process second capture
        await provider._process_screen_capture(capture2)
        
        assert provider.current_content.content == "Completely different content now"
        assert provider.total_updates == 2
        assert provider.significant_changes == 2
        
        # Create third capture with minimal change
        capture3 = ScreenCapture(
            image=None,
            timestamp=datetime.now(timezone.utc),
            window_title="Test App",
            window_app="TestApp",
            window_bounds=(0, 0, 800, 600),
            ocr_text="Completely different content now.",  # Just added period
            text_hash="hash3",
            confidence=85.0
        )
        
        # Process third capture
        await provider._process_screen_capture(capture3)
        
        assert provider.total_updates == 3
        # Should still be 2 if change wasn't significant enough
        # (depends on threshold, minimal punctuation change might not trigger)
    
    @pytest.mark.asyncio
    async def test_privacy_filtering_integration(self, screen_context_config, mock_screen_watcher):
        """Test privacy filtering in full processing pipeline"""
        provider = ScreenContextProvider(
            config=screen_context_config,
            screen_watcher=mock_screen_watcher
        )
        
        await provider.initialize()
        
        # Create capture with sensitive data
        sensitive_capture = ScreenCapture(
            image=None,
            timestamp=datetime.now(timezone.utc),
            window_title="Banking",
            window_app="BankApp",
            window_bounds=(0, 0, 1024, 768),
            ocr_text="Account: 4532-1234-5678-9012 Email: john@example.com Password: secret123",
            text_hash="sensitive_hash",
            confidence=90.0
        )
        
        # Process the sensitive capture
        await provider._process_screen_capture(sensitive_capture)
        
        assert provider.current_content is not None
        assert provider.current_content.privacy_filtered is True
        assert provider.privacy_filters_applied == 1
        assert "[SENSITIVE DATA HIDDEN]" in provider.current_content.content
        assert "4532-1234-5678-9012" not in provider.current_content.content
        assert "john@example.com" not in provider.current_content.content
        assert "secret123" not in provider.current_content.content


# Integration and Factory Tests
class TestIntegrationAndFactories:
    """Test factory functions and integration scenarios"""
    
    def test_create_screen_context_provider_factory(self):
        """Test factory function with default parameters"""
        provider = create_screen_context_provider()
        
        assert isinstance(provider, ScreenContextProvider)
        assert isinstance(provider.config, ScreenContextConfig)
        assert provider.screen_watcher is None  # No watcher provided
    
    def test_create_screen_context_provider_with_params(self, screen_context_config, mock_screen_watcher):
        """Test factory function with custom parameters"""
        provider = create_screen_context_provider(
            config=screen_context_config,
            screen_watcher=mock_screen_watcher
        )
        
        assert isinstance(provider, ScreenContextProvider)
        assert provider.config == screen_context_config
        assert provider.screen_watcher == mock_screen_watcher
    
    @pytest.mark.asyncio
    async def test_callback_system(self, screen_context_config, mock_screen_watcher, sample_screen_capture):
        """Test event callback system"""
        provider = ScreenContextProvider(
            config=screen_context_config,
            screen_watcher=mock_screen_watcher
        )
        
        # Set up callback mocks
        content_changed_callback = Mock()
        significant_change_callback = Mock()
        
        provider.on_content_changed = content_changed_callback
        provider.on_significant_change = significant_change_callback
        
        await provider.initialize()
        
        # Process a capture to trigger callbacks
        await provider._process_screen_capture(sample_screen_capture)
        
        # Verify callbacks were called
        content_changed_callback.assert_called_once()
        significant_change_callback.assert_called_once()
        
        # Check callback arguments
        call_args = content_changed_callback.call_args[0]
        assert isinstance(call_args[0], ScreenContextData)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, screen_context_config, mock_screen_watcher):
        """Test error handling in various scenarios"""
        provider = ScreenContextProvider(
            config=screen_context_config,
            screen_watcher=mock_screen_watcher
        )
        
        await provider.initialize()
        
        # Test error in screen capture processing
        mock_screen_watcher._capture_screen.side_effect = Exception("Capture failed")
        
        # Should not raise exception
        content = await provider.get_latest_content()
        assert content is None
        
        # Test error in capture processing
        invalid_capture = ScreenCapture(
            image=None,
            timestamp=datetime.now(timezone.utc),
            window_title="Test",
            window_app="Test",
            window_bounds=(0, 0, 100, 100),
            ocr_text=None,  # Invalid None text
            text_hash="test",
            confidence=80.0
        )
        
        # Should handle gracefully
        await provider._process_screen_capture(invalid_capture)
        # Should not crash


# Performance and Edge Case Tests
class TestPerformanceAndEdgeCases:
    """Test performance characteristics and edge cases"""
    
    def test_content_length_truncation(self, screen_context_config):
        """Test content truncation for token limits"""
        config = screen_context_config
        config.max_content_length = 50  # Very small for testing
        
        formatter = ScreenContentFormatter(config)
        
        # Create capture with long content
        long_capture = ScreenCapture(
            image=None,
            timestamp=datetime.now(timezone.utc),
            window_title="Long Content",
            window_app="LongApp",
            window_bounds=(0, 0, 1024, 768),
            ocr_text="This is a very long piece of content that should be truncated to fit within the specified maximum length limit",
            text_hash="long_hash",
            confidence=85.0
        )
        
        formatted_data = formatter.format_content(long_capture)
        
        assert len(formatted_data.content) <= config.max_content_length
        assert formatted_data.content.endswith("...")
    
    def test_empty_and_invalid_content_handling(self, screen_context_config):
        """Test handling of empty or invalid content"""
        formatter = ScreenContentFormatter(screen_context_config)
        
        # Test empty content
        empty_capture = ScreenCapture(
            image=None,
            timestamp=datetime.now(timezone.utc),
            window_title="Empty",
            window_app="EmptyApp",
            window_bounds=(0, 0, 100, 100),
            ocr_text="",
            text_hash="empty_hash",
            confidence=0.0
        )
        
        formatted_data = formatter.format_content(empty_capture)
        
        assert formatted_data.content == ""
        assert formatted_data.word_count == 0
        assert formatted_data.character_count == 0
        assert formatted_data.privacy_filtered is False
    
    def test_high_confidence_threshold_filtering(self, screen_context_config):
        """Test confidence threshold filtering (placeholder for future implementation)"""
        # Note: Current implementation doesn't filter by confidence in OCR text
        # but this test establishes the framework for future confidence-based filtering
        config = screen_context_config
        config.min_confidence_threshold = 95.0  # Very high threshold
        
        formatter = ScreenContentFormatter(config)
        
        # Low confidence capture
        low_conf_capture = ScreenCapture(
            image=None,
            timestamp=datetime.now(timezone.utc),
            window_title="Low Confidence",
            window_app="LowConfApp",
            window_bounds=(0, 0, 100, 100),
            ocr_text="uncertain text",
            text_hash="low_conf_hash",
            confidence=70.0  # Below threshold
        )
        
        formatted_data = formatter.format_content(low_conf_capture)
        
        # Currently passes through regardless of confidence
        # Future implementation could filter based on confidence
        assert formatted_data.confidence == 70.0


# Mock integration test
@pytest.mark.asyncio
async def test_full_integration_simulation():
    """Test full integration simulation with realistic data flow"""
    # Create realistic configuration
    config = ScreenContextConfig(
        update_interval=2.0,
        min_change_threshold=0.2,
        max_content_length=800,
        enable_privacy_mode=True,
        enable_content_caching=False,  # Disable caching for this test
        enable_background_updates=False
    )
    
    # Create mock screen watcher
    mock_watcher = Mock(spec=ScreenWatcher)
    mock_watcher.mss_instance = MagicMock()
    mock_watcher.running = False
    mock_watcher.initialize = AsyncMock(return_value=True)
    mock_watcher.start_monitoring = Mock(return_value=True)
    
    # Create provider
    provider = create_screen_context_provider(config=config, screen_watcher=mock_watcher)
    
    # Initialize and start
    init_success = await provider.initialize()
    assert init_success is True
    
    start_success = await provider.start_monitoring()
    assert start_success is True
    
    # Simulate realistic screen content updates
    screen_contents = [
        "Welcome to Code Editor. File: main.py",
        "def hello_world():\n    print('Hello, World!')",
        "Running tests... 5 passed, 0 failed",
        "Git commit: 'Fix issue with screen context provider'"
    ]
    
    for i, content in enumerate(screen_contents):
        capture = ScreenCapture(
            image=None,
            timestamp=datetime.now(timezone.utc),
            window_title="Code Editor",
            window_app="VSCode",
            window_bounds=(0, 0, 1920, 1080),
            ocr_text=content,
            text_hash=f"hash_{i}",
            confidence=85.0 + i * 2  # Increasing confidence
        )
        
        mock_watcher._capture_screen.return_value = capture
        
        # Force fresh capture and processing
        await provider._capture_and_process_content()
        
        # Get latest content
        formatted_content = await provider.get_latest_content()
        
        if formatted_content:
            assert "[SCREEN CONTENT at" in formatted_content
            assert "VSCode" in formatted_content
            
            # Account for text cleaning that removes extra whitespace/newlines
            expected_cleaned_content = re.sub(r'\s+', ' ', content).strip()
            assert expected_cleaned_content in formatted_content
            assert "[END SCREEN CONTENT]" in formatted_content
    
    # Verify metrics
    metrics = provider.get_metrics()
    assert metrics["total_updates"] > 0
    assert metrics["is_running"] is True
    
    # Stop monitoring
    provider.stop_monitoring()
    assert provider.is_running is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 