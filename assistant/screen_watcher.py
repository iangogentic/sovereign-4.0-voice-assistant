"""
Sovereign Voice Assistant - Screen OCR Monitoring System

Implements screen awareness capability that:
- Captures screenshots using mss (cross-platform)
- Detects active windows (macOS: Quartz, Windows: win32gui, Linux: xdotool) 
- Performs OCR text extraction using Tesseract with optimized settings
- Preprocesses images for better OCR accuracy
- Stores results in ChromaDB memory system
- Implements intelligent change detection to avoid redundant processing

Integrates with the memory system built in Task 4.
"""

import os
import sys
import asyncio
import logging
import time
import hashlib
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
import threading
import platform

# Image processing and OCR
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import pytesseract
import mss
import numpy as np

# Platform-specific imports
if platform.system() == "Darwin":  # macOS
    try:
        from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGNullWindowID
        QUARTZ_AVAILABLE = True
    except ImportError:
        QUARTZ_AVAILABLE = False
        logging.warning("Quartz not available for macOS window detection")
elif platform.system() == "Windows":
    try:
        import win32gui
        WIN32_AVAILABLE = True
    except ImportError:
        WIN32_AVAILABLE = False
        logging.warning("win32gui not available for Windows window detection")
else:  # Linux
    WIN32_AVAILABLE = False
    QUARTZ_AVAILABLE = False


@dataclass
class ScreenWatcherConfig:
    """Configuration for the screen monitoring system"""
    
    # Monitoring settings
    monitor_interval: float = 3.0  # seconds between captures
    change_threshold: float = 0.05  # minimum text change to trigger storage
    max_text_length: int = 10000  # maximum OCR text length to store
    
    # OCR settings
    tesseract_config: str = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?;: \n\t'
    language: str = 'eng'
    
    # Image preprocessing
    enable_preprocessing: bool = True
    grayscale_conversion: bool = True
    contrast_enhancement: float = 1.5
    sharpening_enabled: bool = True
    dark_theme_detection: bool = True
    
    # Active window detection
    focus_active_window: bool = True
    capture_full_screen: bool = False  # If False, capture only active window
    
    # Performance settings
    max_screenshot_size: Tuple[int, int] = (1920, 1080)
    compression_quality: int = 85


@dataclass
class ScreenCapture:
    """Represents a screen capture with metadata"""
    
    image: Image.Image
    timestamp: datetime
    window_title: str = ""
    window_app: str = ""
    window_bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)
    ocr_text: str = ""
    text_hash: str = ""
    confidence: float = 0.0


@dataclass
class WindowInfo:
    """Information about an active window"""
    
    title: str
    app_name: str
    bounds: Tuple[int, int, int, int]  # (x, y, width, height)
    window_id: int = 0


class ActiveWindowDetector:
    """Platform-specific active window detection"""
    
    def __init__(self):
        self.platform = platform.system()
        self.logger = logging.getLogger(__name__)
        
    def get_active_window(self) -> Optional[WindowInfo]:
        """Get information about the currently active window"""
        try:
            if self.platform == "Darwin" and QUARTZ_AVAILABLE:
                return self._get_active_window_macos()
            elif self.platform == "Windows" and WIN32_AVAILABLE:
                return self._get_active_window_windows()
            else:
                return self._get_active_window_fallback()
        except Exception as e:
            self.logger.error(f"❌ Failed to get active window: {e}")
            return None
    
    def _get_active_window_macos(self) -> Optional[WindowInfo]:
        """Get active window on macOS using Quartz"""
        try:
            # Get list of all windows
            window_list = CGWindowListCopyWindowInfo(
                kCGWindowListOptionOnScreenOnly, 
                kCGNullWindowID
            )
            
            # Find the frontmost window
            for window in window_list:
                if window.get('kCGWindowLayer', 0) == 0:  # Main layer
                    window_name = window.get('kCGWindowName', '')
                    owner_name = window.get('kCGWindowOwnerName', '')
                    bounds = window.get('kCGWindowBounds', {})
                    
                    if window_name and bounds:
                        return WindowInfo(
                            title=window_name,
                            app_name=owner_name,
                            bounds=(
                                int(bounds.get('X', 0)),
                                int(bounds.get('Y', 0)), 
                                int(bounds.get('Width', 0)),
                                int(bounds.get('Height', 0))
                            ),
                            window_id=window.get('kCGWindowNumber', 0)
                        )
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ macOS window detection failed: {e}")
            return None
    
    def _get_active_window_windows(self) -> Optional[WindowInfo]:
        """Get active window on Windows using win32gui"""
        try:
            hwnd = win32gui.GetForegroundWindow()
            if hwnd:
                window_title = win32gui.GetWindowText(hwnd)
                rect = win32gui.GetWindowRect(hwnd)
                
                return WindowInfo(
                    title=window_title,
                    app_name="Unknown",  # Would need additional API calls
                    bounds=(rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]),
                    window_id=hwnd
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Windows window detection failed: {e}")
            return None
    
    def _get_active_window_fallback(self) -> Optional[WindowInfo]:
        """Fallback when platform-specific detection is unavailable"""
        return WindowInfo(
            title="Unknown Window",
            app_name="Unknown App",
            bounds=(0, 0, 1920, 1080),  # Default to full screen
            window_id=0
        )


class ScreenImageProcessor:
    """Handles image preprocessing for better OCR accuracy"""
    
    def __init__(self, config: ScreenWatcherConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply preprocessing to improve OCR accuracy"""
        if not self.config.enable_preprocessing:
            return image
        
        try:
            processed_image = image.copy()
            
            # Resize if too large
            if (processed_image.width > self.config.max_screenshot_size[0] or 
                processed_image.height > self.config.max_screenshot_size[1]):
                processed_image.thumbnail(self.config.max_screenshot_size, Image.Resampling.LANCZOS)
            
            # Convert to grayscale
            if self.config.grayscale_conversion:
                processed_image = processed_image.convert('L')
            
            # Detect and handle dark themes
            if self.config.dark_theme_detection and self._is_dark_theme(processed_image):
                processed_image = ImageOps.invert(processed_image)
                self.logger.debug("🌙 Dark theme detected - inverted image")
            
            # Enhance contrast
            if self.config.contrast_enhancement != 1.0:
                enhancer = ImageEnhance.Contrast(processed_image)
                processed_image = enhancer.enhance(self.config.contrast_enhancement)
            
            # Apply sharpening
            if self.config.sharpening_enabled:
                processed_image = processed_image.filter(ImageFilter.SHARPEN)
            
            return processed_image
            
        except Exception as e:
            self.logger.error(f"❌ Image preprocessing failed: {e}")
            return image  # Return original on error
    
    def _is_dark_theme(self, image: Image.Image) -> bool:
        """Detect if image likely shows a dark theme interface"""
        try:
            # Convert to grayscale if not already
            gray_image = image.convert('L') if image.mode != 'L' else image
            
            # Sample pixels from the image
            pixels = np.array(gray_image)
            mean_brightness = np.mean(pixels)
            
            # Consider dark theme if average brightness is below threshold
            return mean_brightness < 128  # Middle of 0-255 range
            
        except Exception:
            return False


class OCRProcessor:
    """Handles OCR text extraction with optimized settings"""
    
    def __init__(self, config: ScreenWatcherConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Test Tesseract availability
        try:
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
            self.logger.info("✅ Tesseract OCR available")
        except Exception as e:
            self.tesseract_available = False
            self.logger.error(f"❌ Tesseract OCR not available: {e}")
    
    def extract_text(self, image: Image.Image) -> Tuple[str, float]:
        """Extract text from image using OCR"""
        if not self.tesseract_available:
            return "", 0.0
        
        try:
            # Extract text with confidence data
            ocr_data = pytesseract.image_to_data(
                image,
                config=self.config.tesseract_config,
                lang=self.config.language,
                output_type=pytesseract.Output.DICT
            )
            
            # Combine text and calculate average confidence
            text_parts = []
            confidences = []
            
            for i, word in enumerate(ocr_data['text']):
                confidence = ocr_data['conf'][i]
                if confidence > 0 and word.strip():  # Valid word
                    text_parts.append(word)
                    confidences.append(confidence)
            
            text = ' '.join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Limit text length
            if len(text) > self.config.max_text_length:
                text = text[:self.config.max_text_length] + "..."
            
            self.logger.debug(f"🔍 OCR extracted {len(text)} chars with {avg_confidence:.1f}% confidence")
            return text, avg_confidence
            
        except Exception as e:
            self.logger.error(f"❌ OCR text extraction failed: {e}")
            return "", 0.0


class ScreenWatcher:
    """
    Main screen monitoring system that captures screenshots, performs OCR,
    and stores results in the memory system
    """
    
    def __init__(self, config: ScreenWatcherConfig = None, memory_manager = None):
        self.config = config or ScreenWatcherConfig()
        self.memory_manager = memory_manager
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.window_detector = ActiveWindowDetector()
        self.image_processor = ScreenImageProcessor(self.config)
        self.ocr_processor = OCRProcessor(self.config)
        
        # State management
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.last_text_hash = ""
        self.capture_count = 0
        
        # Screenshot capture
        self.mss_instance = None
        
        # Callbacks
        self.on_text_changed: Optional[Callable[[str, WindowInfo], None]] = None
        self.on_capture_complete: Optional[Callable[[ScreenCapture], None]] = None
        
    async def initialize(self) -> bool:
        """Initialize the screen watcher"""
        try:
            self.logger.info("🖥️ Initializing Screen OCR Monitoring System...")
            
            # Initialize MSS for screenshot capture
            self.mss_instance = mss.mss()
            
            # Test screenshot capability
            try:
                screenshot = self.mss_instance.grab(self.mss_instance.monitors[0])
                test_image = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                
                if test_image.size[0] > 0 and test_image.size[1] > 0:
                    self.logger.info(f"✅ Screenshot capability verified: {test_image.size}")
                else:
                    raise Exception("Invalid screenshot size")
            except Exception as e:
                if "CoreGraphics.CGWindowListCreateImage() failed" in str(e) or "CGDisplayCreateImage" in str(e):
                    self.logger.error("❌ SCREEN RECORDING PERMISSION REQUIRED!")
                    self.logger.error("🔧 TO FIX: Go to System Preferences > Security & Privacy > Privacy > Screen Recording")
                    self.logger.error("🔧 Add Terminal (or your IDE) to the allowed applications")
                    self.logger.error("🔧 Restart the application after granting permission")
                    return False
                else:
                    raise e
            
            # Test window detection
            window_info = self.window_detector.get_active_window()
            if window_info:
                self.logger.info(f"✅ Window detection working: {window_info.title}")
            else:
                self.logger.warning("⚠️ Window detection may not be fully functional")
            
            # Test OCR
            if self.ocr_processor.tesseract_available:
                self.logger.info("✅ OCR processor ready")
            else:
                self.logger.warning("⚠️ OCR not available - text extraction disabled")
            
            self.logger.info("✅ Screen OCR Monitoring System initialized successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize screen watcher: {e}")
            return False
    
    def start_monitoring(self) -> bool:
        """Start the screen monitoring thread"""
        if self.running:
            self.logger.warning("Screen monitoring already running")
            return True
        
        try:
            self.running = True
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True,
                name="ScreenWatcher"
            )
            self.monitor_thread.start()
            
            self.logger.info(f"🖥️ Screen monitoring started (interval: {self.config.monitor_interval}s)")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start screen monitoring: {e}")
            self.running = False
            return False
    
    def stop_monitoring(self):
        """Stop the screen monitoring"""
        if not self.running:
            return
        
        self.logger.info("🛑 Stopping screen monitoring...")
        self.running = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        if self.mss_instance:
            self.mss_instance.close()
            
        self.logger.info("✅ Screen monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop that runs in a separate thread"""
        self.logger.info("🔄 Screen monitoring loop started")
        
        while self.running:
            try:
                # Capture and process screen
                capture = self._capture_screen()
                
                if capture and capture.ocr_text:
                    # Check for significant changes
                    if self._has_significant_change(capture.text_hash):
                        # Store in memory system if available
                        if self.memory_manager:
                            asyncio.run(self._store_screen_content(capture))
                        
                        # Trigger callbacks
                        if self.on_text_changed:
                            window_info = WindowInfo(
                                title=capture.window_title,
                                app_name=capture.window_app,
                                bounds=capture.window_bounds
                            )
                            self.on_text_changed(capture.ocr_text, window_info)
                        
                        if self.on_capture_complete:
                            self.on_capture_complete(capture)
                        
                        self.last_text_hash = capture.text_hash
                        self.logger.debug(f"📝 Screen content changed: {len(capture.ocr_text)} chars")
                
                self.capture_count += 1
                
                # Wait for next interval
                time.sleep(self.config.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Error in monitoring loop: {e}")
                time.sleep(1.0)  # Brief pause on error
    
    def _capture_screen(self) -> Optional[ScreenCapture]:
        """Capture and process the current screen"""
        try:
            # Get active window info
            window_info = self.window_detector.get_active_window()
            
            # Determine capture area
            if self.config.focus_active_window and window_info and window_info.bounds[2] > 0:
                # Capture specific window
                monitor = {
                    "top": window_info.bounds[1],
                    "left": window_info.bounds[0], 
                    "width": window_info.bounds[2],
                    "height": window_info.bounds[3]
                }
            else:
                # Capture full screen
                monitor = self.mss_instance.monitors[0]
            
            # Take screenshot
            screenshot = self.mss_instance.grab(monitor)
            image = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            
            # Preprocess image
            processed_image = self.image_processor.preprocess_image(image)
            
            # Extract text via OCR
            ocr_text, confidence = self.ocr_processor.extract_text(processed_image)
            
            # Create text hash for change detection
            text_hash = hashlib.md5(ocr_text.encode('utf-8')).hexdigest()
            
            # Create capture object
            capture = ScreenCapture(
                image=processed_image,
                timestamp=datetime.now(timezone.utc),
                window_title=window_info.title if window_info else "Unknown",
                window_app=window_info.app_name if window_info else "Unknown",
                window_bounds=window_info.bounds if window_info else (0, 0, 0, 0),
                ocr_text=ocr_text,
                text_hash=text_hash,
                confidence=confidence
            )
            
            return capture
            
        except Exception as e:
            self.logger.error(f"❌ Screen capture failed: {e}")
            return None
    
    def _has_significant_change(self, new_text_hash: str) -> bool:
        """Check if the screen content has changed significantly"""
        if not self.last_text_hash:
            return True  # First capture
        
        return new_text_hash != self.last_text_hash
    
    async def _store_screen_content(self, capture: ScreenCapture):
        """Store screen content in the memory system"""
        try:
            metadata = {
                "window_title": capture.window_title,
                "window_app": capture.window_app,
                "window_bounds": capture.window_bounds,
                "confidence": capture.confidence,
                "capture_count": self.capture_count
            }
            
            success = await self.memory_manager.store_screen_content(
                content=capture.ocr_text,
                source="screen_ocr",
                metadata=metadata
            )
            
            if success:
                self.logger.debug(f"💾 Stored screen content: {capture.window_title}")
            else:
                self.logger.warning("⚠️ Failed to store screen content in memory")
                
        except Exception as e:
            self.logger.error(f"❌ Error storing screen content: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return {
            "running": self.running,
            "capture_count": self.capture_count,
            "monitor_interval": self.config.monitor_interval,
            "last_capture_hash": self.last_text_hash,
            "ocr_available": self.ocr_processor.tesseract_available,
            "window_detection_available": (
                QUARTZ_AVAILABLE if platform.system() == "Darwin" 
                else WIN32_AVAILABLE if platform.system() == "Windows"
                else False
            )
        }


# Factory functions
def create_screen_watcher(config: ScreenWatcherConfig = None, memory_manager = None) -> ScreenWatcher:
    """Create and return a screen watcher instance"""
    return ScreenWatcher(config, memory_manager)


def get_default_screen_config() -> ScreenWatcherConfig:
    """Get default screen watcher configuration"""
    return ScreenWatcherConfig()


# Export main classes
__all__ = [
    "ScreenWatcher",
    "ScreenWatcherConfig", 
    "ScreenCapture",
    "WindowInfo",
    "ActiveWindowDetector",
    "ScreenImageProcessor",
    "OCRProcessor",
    "create_screen_watcher",
    "get_default_screen_config"
] 