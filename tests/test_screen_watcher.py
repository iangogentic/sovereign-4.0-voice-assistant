"""
Tests for the Screen Watcher Module

Tests cover:
- ScreenWatcher initialization and configuration
- Screenshot capture and processing
- Active window detection (platform-specific)
- OCR text extraction with Tesseract
- Image preprocessing for better accuracy
- Memory system integration
- Change detection algorithms
"""

import pytest
import asyncio
import tempfile
import platform
from unittest.mock import Mock, patch, AsyncMock
from PIL import Image
import numpy as np

from assistant.screen_watcher import (
    ScreenWatcher,
    ScreenWatcherConfig,
    ScreenCapture,
    WindowInfo,
    ActiveWindowDetector,
    ScreenImageProcessor,
    OCRProcessor,
    create_screen_watcher,
    get_default_screen_config
)


class TestScreenWatcherConfig:
    """Test screen watcher configuration"""
    
    def test_default_config(self):
        """Test default screen watcher configuration"""
        config = ScreenWatcherConfig()
        
        assert config.monitor_interval == 3.0
        assert config.change_threshold == 0.05
        assert config.max_text_length == 10000
        assert config.tesseract_config.startswith('--psm 6')
        assert config.language == 'eng'
        assert config.enable_preprocessing is True
        assert config.focus_active_window is True
        assert config.max_screenshot_size == (1920, 1080)
    
    def test_custom_config(self):
        """Test custom screen watcher configuration"""
        config = ScreenWatcherConfig(
            monitor_interval=5.0,
            change_threshold=0.1,
            max_text_length=5000,
            language='fra'
        )
        
        assert config.monitor_interval == 5.0
        assert config.change_threshold == 0.1
        assert config.max_text_length == 5000
        assert config.language == 'fra'


class TestScreenWatcherDataClasses:
    """Test screen watcher data classes"""
    
    def test_window_info(self):
        """Test WindowInfo data class"""
        window = WindowInfo(
            title="Test Window",
            app_name="Test App",
            bounds=(100, 100, 800, 600),
            window_id=12345
        )
        
        assert window.title == "Test Window"
        assert window.app_name == "Test App"
        assert window.bounds == (100, 100, 800, 600)
        assert window.window_id == 12345
    
    def test_screen_capture(self):
        """Test ScreenCapture data class"""
        # Create a simple test image
        test_image = Image.new('RGB', (100, 100), 'white')
        
        capture = ScreenCapture(
            image=test_image,
            timestamp=None,  # Will be set automatically
            window_title="Test",
            window_app="TestApp",
            window_bounds=(0, 0, 100, 100),
            ocr_text="Test text",
            text_hash="abc123",
            confidence=95.5
        )
        
        assert capture.image == test_image
        assert capture.window_title == "Test"
        assert capture.ocr_text == "Test text"
        assert capture.confidence == 95.5


class TestActiveWindowDetector:
    """Test active window detection"""
    
    def test_detector_creation(self):
        """Test window detector creation"""
        detector = ActiveWindowDetector()
        assert detector.platform == platform.system()
    
    def test_get_active_window_fallback(self):
        """Test fallback window detection"""
        detector = ActiveWindowDetector()
        
        # Force fallback mode
        with patch.object(detector, 'platform', 'Unknown'):
            window = detector.get_active_window()
            
            assert window is not None
            assert window.title == "Unknown Window"
            assert window.app_name == "Unknown App"
            assert window.bounds == (0, 0, 1920, 1080)
    
    @pytest.mark.skipif(platform.system() != "Darwin", reason="macOS specific test")
    def test_macos_window_detection(self):
        """Test macOS window detection"""
        detector = ActiveWindowDetector()
        
        # This will test actual window detection on macOS
        # Skip if running in CI or Quartz not available
        try:
            window = detector.get_active_window()
            if window:
                assert isinstance(window.title, str)
                assert isinstance(window.app_name, str)
                assert len(window.bounds) == 4
        except Exception:
            pytest.skip("Quartz window detection not available")


class TestScreenImageProcessor:
    """Test image preprocessing functionality"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return ScreenWatcherConfig()
    
    @pytest.fixture
    def processor(self, config):
        """Create image processor"""
        return ScreenImageProcessor(config)
    
    @pytest.fixture
    def test_image(self):
        """Create test image"""
        return Image.new('RGB', (800, 600), 'white')
    
    def test_processor_creation(self, processor):
        """Test image processor creation"""
        assert processor is not None
        assert processor.config is not None
    
    def test_preprocess_image_disabled(self, processor, test_image):
        """Test preprocessing when disabled"""
        processor.config.enable_preprocessing = False
        result = processor.preprocess_image(test_image)
        assert result == test_image
    
    def test_preprocess_image_basic(self, processor, test_image):
        """Test basic image preprocessing"""
        result = processor.preprocess_image(test_image)
        
        # Should have been processed
        assert result is not test_image  # Different object
        assert result.size == test_image.size  # Same size for small image
    
    def test_preprocess_image_resize(self, processor):
        """Test image resizing for large images"""
        # Create oversized image
        large_image = Image.new('RGB', (3000, 2000), 'white')
        result = processor.preprocess_image(large_image)
        
        # Should be resized
        assert result.width <= processor.config.max_screenshot_size[0]
        assert result.height <= processor.config.max_screenshot_size[1]
    
    def test_dark_theme_detection(self, processor):
        """Test dark theme detection"""
        # Create dark image
        dark_image = Image.new('RGB', (100, 100), (30, 30, 30))
        is_dark = processor._is_dark_theme(dark_image.convert('L'))
        assert is_dark is True
        
        # Create light image
        light_image = Image.new('RGB', (100, 100), (200, 200, 200))
        is_light = processor._is_dark_theme(light_image.convert('L'))
        assert is_light is False


class TestOCRProcessor:
    """Test OCR text extraction"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return ScreenWatcherConfig()
    
    @pytest.fixture
    def ocr_processor(self, config):
        """Create OCR processor"""
        return OCRProcessor(config)
    
    def test_ocr_processor_creation(self, ocr_processor):
        """Test OCR processor creation"""
        assert ocr_processor is not None
        assert hasattr(ocr_processor, 'tesseract_available')
        # Note: May be True or False depending on system setup
    
    @patch('pytesseract.get_tesseract_version')
    def test_ocr_availability_check(self, mock_version, config):
        """Test Tesseract availability checking"""
        # Test when available
        mock_version.return_value = "5.0.0"
        processor = OCRProcessor(config)
        assert processor.tesseract_available is True
        
        # Test when not available
        mock_version.side_effect = Exception("Not found")
        processor = OCRProcessor(config)
        assert processor.tesseract_available is False
    
    @patch('pytesseract.image_to_data')
    def test_extract_text_success(self, mock_image_to_data, ocr_processor):
        """Test successful text extraction"""
        # Mock OCR data
        mock_image_to_data.return_value = {
            'text': ['Hello', 'World', ''],
            'conf': [95, 90, 0]
        }
        
        # Force OCR availability
        ocr_processor.tesseract_available = True
        
        test_image = Image.new('RGB', (100, 50), 'white')
        text, confidence = ocr_processor.extract_text(test_image)
        
        assert text == "Hello World"
        assert confidence == 92.5  # Average of 95 and 90
    
    def test_extract_text_unavailable(self, ocr_processor):
        """Test text extraction when OCR unavailable"""
        ocr_processor.tesseract_available = False
        
        test_image = Image.new('RGB', (100, 50), 'white')
        text, confidence = ocr_processor.extract_text(test_image)
        
        assert text == ""
        assert confidence == 0.0
    
    @patch('pytesseract.image_to_data')
    def test_extract_text_long_content(self, mock_image_to_data, ocr_processor):
        """Test text extraction with length limiting"""
        # Create very long text
        long_text = ['word'] * 5000  # Much longer than max_text_length
        confidences = [80] * 5000
        
        mock_image_to_data.return_value = {
            'text': long_text,
            'conf': confidences
        }
        
        ocr_processor.tesseract_available = True
        ocr_processor.config.max_text_length = 100
        
        test_image = Image.new('RGB', (100, 50), 'white')
        text, confidence = ocr_processor.extract_text(test_image)
        
        assert len(text) <= 104  # 100 + "..."
        assert text.endswith("...")


class TestScreenWatcher:
    """Test main ScreenWatcher functionality"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return ScreenWatcherConfig(monitor_interval=1.0)  # Faster for testing
    
    @pytest.fixture
    def mock_memory_manager(self):
        """Create mock memory manager"""
        mock_manager = Mock()
        mock_manager.store_screen_content = AsyncMock(return_value=True)
        return mock_manager
    
    @pytest.fixture
    def screen_watcher(self, config, mock_memory_manager):
        """Create screen watcher"""
        return ScreenWatcher(config, mock_memory_manager)
    
    def test_screen_watcher_creation(self, screen_watcher):
        """Test screen watcher creation"""
        assert screen_watcher is not None
        assert screen_watcher.running is False
        assert screen_watcher.capture_count == 0
        assert screen_watcher.last_text_hash == ""
    
    @pytest.mark.asyncio
    @patch('mss.mss')
    async def test_screen_watcher_initialization(self, mock_mss_class, screen_watcher):
        """Test screen watcher initialization"""
        # Mock MSS
        mock_mss = Mock()
        mock_screenshot = Mock()
        mock_screenshot.size = (1920, 1080)
        mock_screenshot.bgra = b'\x00' * (1920 * 1080 * 4)
        
        mock_mss.grab.return_value = mock_screenshot
        mock_mss.monitors = [{'top': 0, 'left': 0, 'width': 1920, 'height': 1080}]
        mock_mss_class.return_value = mock_mss
        
        # Mock window detection
        mock_window = WindowInfo("Test Window", "Test App", (0, 0, 800, 600))
        with patch.object(screen_watcher.window_detector, 'get_active_window', return_value=mock_window):
            # Mock OCR availability
            screen_watcher.ocr_processor.tesseract_available = True
            
            result = await screen_watcher.initialize()
            assert result is True
    
    def test_screen_watcher_start_stop(self, screen_watcher):
        """Test starting and stopping screen monitoring"""
        # Mock initialization
        screen_watcher.mss_instance = Mock()
        
        # Test start
        result = screen_watcher.start_monitoring()
        assert result is True
        assert screen_watcher.running is True
        assert screen_watcher.monitor_thread is not None
        
        # Test stop
        screen_watcher.stop_monitoring()
        assert screen_watcher.running is False
    
    def test_has_significant_change(self, screen_watcher):
        """Test change detection"""
        # First hash should always be significant
        assert screen_watcher._has_significant_change("hash1") is True
        
        # Set last hash
        screen_watcher.last_text_hash = "hash1"
        
        # Same hash should not be significant
        assert screen_watcher._has_significant_change("hash1") is False
        
        # Different hash should be significant
        assert screen_watcher._has_significant_change("hash2") is True
    
    def test_get_stats(self, screen_watcher):
        """Test statistics gathering"""
        stats = screen_watcher.get_stats()
        
        assert "running" in stats
        assert "capture_count" in stats
        assert "monitor_interval" in stats
        assert "ocr_available" in stats
        assert "window_detection_available" in stats
        
        assert stats["running"] is False
        assert stats["capture_count"] == 0
    
    @pytest.mark.asyncio
    async def test_store_screen_content(self, screen_watcher, mock_memory_manager):
        """Test storing screen content in memory"""
        test_image = Image.new('RGB', (100, 100), 'white')
        capture = ScreenCapture(
            image=test_image,
            timestamp=None,
            window_title="Test Window",
            window_app="Test App",
            window_bounds=(0, 0, 100, 100),
            ocr_text="Test text",
            text_hash="abc123",
            confidence=95.0
        )
        
        await screen_watcher._store_screen_content(capture)
        
        # Verify memory manager was called
        mock_memory_manager.store_screen_content.assert_called_once()
        call_args = mock_memory_manager.store_screen_content.call_args
        
        assert call_args[1]["content"] == "Test text"
        assert call_args[1]["source"] == "screen_ocr"
        assert "window_title" in call_args[1]["metadata"]


class TestFactoryFunctions:
    """Test factory functions"""
    
    def test_create_screen_watcher(self):
        """Test create_screen_watcher factory function"""
        watcher = create_screen_watcher()
        assert isinstance(watcher, ScreenWatcher)
        assert isinstance(watcher.config, ScreenWatcherConfig)
    
    def test_create_screen_watcher_with_config(self):
        """Test create_screen_watcher with custom config"""
        config = ScreenWatcherConfig(monitor_interval=5.0)
        watcher = create_screen_watcher(config)
        assert watcher.config.monitor_interval == 5.0
    
    def test_get_default_screen_config(self):
        """Test get_default_screen_config function"""
        config = get_default_screen_config()
        assert isinstance(config, ScreenWatcherConfig)
        assert config.monitor_interval == 3.0


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    @pytest.mark.asyncio
    @patch('mss.mss')
    @patch('pytesseract.image_to_data')
    async def test_complete_capture_cycle(self, mock_ocr, mock_mss_class):
        """Test a complete screen capture and OCR cycle"""
        # Setup mocks
        mock_mss = Mock()
        mock_screenshot = Mock()
        mock_screenshot.size = (800, 600)
        mock_screenshot.bgra = b'\x00' * (800 * 600 * 4)
        
        mock_mss.grab.return_value = mock_screenshot
        mock_mss.monitors = [{'top': 0, 'left': 0, 'width': 800, 'height': 600}]
        mock_mss_class.return_value = mock_mss
        
        mock_ocr.return_value = {
            'text': ['Hello', 'World', 'from', 'screen'],
            'conf': [95, 90, 85, 92]
        }
        
        # Create screen watcher
        config = ScreenWatcherConfig()
        mock_memory = Mock()
        mock_memory.store_screen_content = AsyncMock(return_value=True)
        
        watcher = ScreenWatcher(config, mock_memory)
        
        # Force OCR availability and window detection
        watcher.ocr_processor.tesseract_available = True
        mock_window = WindowInfo("Test Window", "Test App", (0, 0, 800, 600))
        watcher.window_detector.get_active_window = Mock(return_value=mock_window)
        
        # Initialize and capture
        await watcher.initialize()
        capture = watcher._capture_screen()
        
        # Verify capture
        assert capture is not None
        assert capture.ocr_text == "Hello World from screen"
        assert capture.window_title == "Test Window"
        assert capture.confidence > 0
    
    @pytest.mark.asyncio
    async def test_memory_integration(self):
        """Test integration with memory system"""
        # Create actual memory manager (mocked to avoid DB calls)
        mock_memory = Mock()
        mock_memory.store_screen_content = AsyncMock(return_value=True)
        
        config = ScreenWatcherConfig()
        watcher = ScreenWatcher(config, mock_memory)
        
        # Create test capture
        test_image = Image.new('RGB', (100, 100), 'white')
        capture = ScreenCapture(
            image=test_image,
            timestamp=None,
            window_title="Memory Test",
            window_app="Test App",
            window_bounds=(0, 0, 100, 100),
            ocr_text="Test content for memory",
            text_hash="test_hash",
            confidence=88.5
        )
        
        # Store in memory
        await watcher._store_screen_content(capture)
        
        # Verify memory call
        mock_memory.store_screen_content.assert_called_once()
        call_args = mock_memory.store_screen_content.call_args
        
        assert call_args[1]["content"] == "Test content for memory"
        assert call_args[1]["source"] == "screen_ocr"
        metadata = call_args[1]["metadata"]
        assert metadata["window_title"] == "Memory Test"
        assert metadata["confidence"] == 88.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 