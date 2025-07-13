"""
Sovereign Voice Assistant - Screen Context Provider for Realtime API

Implements specialized screen content integration for OpenAI Realtime API that:
- Extends existing ScreenWatcher functionality for Realtime API sessions
- Provides formatted screen content for session instructions
- Implements intelligent change detection and privacy filtering  
- Manages periodic content updates with configurable intervals
- Optimizes content for Realtime API token limits and relevance

Integrates seamlessly with RealtimeVoiceService for screen-aware conversations.
"""

import asyncio
import logging
import time
import re
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Callable, Set
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import hashlib

from .screen_watcher import ScreenWatcher, ScreenWatcherConfig, ScreenCapture


@dataclass
class ScreenContextConfig:
    """Configuration for screen context provider optimized for Realtime API"""
    
    # Content update settings
    update_interval: float = 5.0  # seconds between content updates
    min_change_threshold: float = 0.1  # minimum text similarity change to trigger update
    max_content_length: int = 1500  # maximum screen content length for Realtime API
    
    # OCR quality settings  
    min_confidence_threshold: float = 60.0  # minimum OCR confidence to include text
    enable_text_cleaning: bool = True
    enable_noise_filtering: bool = True
    
    # Privacy settings
    enable_privacy_mode: bool = True
    sensitive_patterns: List[str] = field(default_factory=lambda: [
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card numbers
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN format
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email addresses
        r'\bpassword[:\s]*[^\s]+',  # Password fields
        r'\btoken[:\s]*[^\s]+',  # API tokens
        r'\bkey[:\s]*[^\s]+',  # API keys
    ])
    
    # Performance settings
    enable_content_caching: bool = True
    cache_duration: float = 30.0  # seconds to cache formatted content
    enable_background_updates: bool = True


@dataclass
class ScreenContextData:
    """Processed screen content data for Realtime API"""
    
    content: str
    timestamp: datetime
    confidence: float
    word_count: int
    character_count: int
    change_score: float  # 0.0 = no change, 1.0 = completely different
    window_title: str
    window_app: str
    privacy_filtered: bool
    source_hash: str
    
    def to_context_string(self) -> str:
        """Format as context string for Realtime API session instructions"""
        timestamp_str = self.timestamp.strftime("%H:%M:%S")
        context_header = f"[SCREEN CONTENT at {timestamp_str} - {self.window_app}]"
        
        if self.privacy_filtered:
            context_header += " [Privacy filtered]"
            
        return f"{context_header}\n{self.content}\n[END SCREEN CONTENT]"
    
    def is_significant_change(self, threshold: float = 0.1) -> bool:
        """Check if this represents a significant change from previous content"""
        return self.change_score >= threshold


class ScreenContentFormatter:
    """Handles screen content formatting and cleaning for Realtime API"""
    
    def __init__(self, config: ScreenContextConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Compile regex patterns for performance
        self.sensitive_patterns = [re.compile(pattern, re.IGNORECASE) 
                                 for pattern in config.sensitive_patterns]
        
        # Common noise patterns to filter out
        self.noise_patterns = [
            re.compile(r'\b[A-Z]{1,3}\b(?:\s+[A-Z]{1,3}\b)*'),  # Single letter sequences
            re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\b'),  # Time stamps
            re.compile(r'\b\d+%'),  # Percentages  
            re.compile(r'\b\d+px\b'),  # Pixel measurements
            re.compile(r'\|\s*\|'),  # Table separators
            re.compile(r'^[\s\-_=]+$', re.MULTILINE),  # Lines of separators
        ]
    
    def format_content(self, screen_capture: ScreenCapture) -> ScreenContextData:
        """Format screen capture into structured context data"""
        
        # Start with raw OCR text
        content = screen_capture.ocr_text
        
        # Apply text cleaning
        if self.config.enable_text_cleaning:
            content = self._clean_text(content)
        
        # Apply noise filtering
        if self.config.enable_noise_filtering:
            content = self._filter_noise(content)
        
        # Apply privacy filtering
        privacy_filtered = False
        if self.config.enable_privacy_mode:
            content, privacy_filtered = self._apply_privacy_filter(content)
        
        # Truncate to max length
        if len(content) > self.config.max_content_length:
            # Account for the ellipsis length
            truncate_length = self.config.max_content_length - 3
            content = content[:truncate_length] + "..."
        
        # Calculate content hash for change detection
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        return ScreenContextData(
            content=content,
            timestamp=screen_capture.timestamp,
            confidence=screen_capture.confidence,
            word_count=len(content.split()) if content else 0,
            character_count=len(content),
            change_score=0.0,  # Will be calculated by provider
            window_title=screen_capture.window_title,
            window_app=screen_capture.window_app,
            privacy_filtered=privacy_filtered,
            source_hash=content_hash
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean OCR text for better readability"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove lone characters scattered in text
        text = re.sub(r'\b[a-zA-Z]\s+(?=[a-zA-Z]\b)', '', text)
        
        # Clean up common OCR artifacts
        text = re.sub(r'[|]{2,}', '|', text)  # Multiple pipes
        text = re.sub(r'[-]{3,}', '---', text)  # Multiple dashes
        text = re.sub(r'[.]{3,}', '...', text)  # Multiple dots
        
        # Remove excessive punctuation
        text = re.sub(r'([!?]){2,}', r'\1', text)
        
        return text.strip()
    
    def _filter_noise(self, text: str) -> str:
        """Filter out noise patterns that don't add context value"""
        for pattern in self.noise_patterns:
            text = pattern.sub('', text)
        
        # Remove empty lines and excessive spacing
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def _apply_privacy_filter(self, text: str) -> tuple[str, bool]:
        """Apply privacy filtering to remove sensitive information"""
        original_text = text
        privacy_filtered = False
        
        for pattern in self.sensitive_patterns:
            if pattern.search(text):
                text = pattern.sub('[SENSITIVE DATA HIDDEN]', text)
                privacy_filtered = True
        
        return text, privacy_filtered
    
    def calculate_change_score(self, current_content: str, previous_content: str) -> float:
        """Calculate similarity score between current and previous content"""
        if not previous_content and not current_content:
            return 0.0  # Both empty = no change
        
        if not previous_content:
            return 1.0  # Complete change if no previous content
        
        if not current_content:
            return 1.0  # Complete change if current is empty but previous had content
        
        # Use SequenceMatcher for text similarity
        matcher = SequenceMatcher(None, previous_content, current_content)
        similarity = matcher.ratio()
        
        # Return change score (1.0 - similarity)
        return 1.0 - similarity


class ScreenContextProvider:
    """
    Specialized screen content provider for Realtime API integration
    
    Extends ScreenWatcher functionality to provide optimized screen content
    for OpenAI Realtime API sessions with privacy filtering and change detection.
    """
    
    def __init__(self, 
                 config: ScreenContextConfig = None,
                 screen_watcher: Optional[ScreenWatcher] = None,
                 logger: Optional[logging.Logger] = None):
        
        self.config = config or ScreenContextConfig()
        self.screen_watcher = screen_watcher
        self.logger = logger or logging.getLogger(__name__)
        
        # Content processing
        self.formatter = ScreenContentFormatter(self.config)
        
        # State management
        self.is_initialized = False
        self.is_running = False
        self.update_timer: Optional[threading.Timer] = None
        
        # Content tracking
        self.current_content: Optional[ScreenContextData] = None
        self.previous_content_hash: str = ""
        self.content_cache: Dict[str, ScreenContextData] = {}
        self.last_update_time: float = 0.0
        
        # Performance metrics
        self.total_updates = 0
        self.significant_changes = 0
        self.privacy_filters_applied = 0
        
        # Event callbacks
        self.on_content_changed: Optional[Callable[[ScreenContextData], None]] = None
        self.on_significant_change: Optional[Callable[[ScreenContextData], None]] = None
        self.on_privacy_filter_applied: Optional[Callable[[str], None]] = None
    
    async def initialize(self) -> bool:
        """Initialize the screen context provider"""
        try:
            self.logger.info("🖥️ Initializing Screen Context Provider for Realtime API...")
            
            # Verify screen watcher is available
            if not self.screen_watcher:
                self.logger.error("❌ ScreenWatcher instance required")
                return False
            
            # Ensure screen watcher is initialized
            if not getattr(self.screen_watcher, 'is_initialized', False):
                screen_init_success = await self.screen_watcher.initialize()
                if not screen_init_success:
                    self.logger.error("❌ Failed to initialize ScreenWatcher")
                    return False
            
            # Set up screen content change callback
            self.screen_watcher.on_capture_complete = self._on_screen_capture
            
            self.is_initialized = True
            self.logger.info("✅ Screen Context Provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Screen Context Provider: {e}")
            return False
    
    async def start_monitoring(self) -> bool:
        """Start periodic screen content monitoring"""
        if not self.is_initialized:
            self.logger.error("❌ Provider not initialized")
            return False
        
        if self.is_running:
            self.logger.warning("⚠️ Provider already running")
            return True
        
        try:
            self.is_running = True
            
            # Start screen watcher if not already running
            if not self.screen_watcher.running:
                self.screen_watcher.start_monitoring()
            
            # Start background content updates if enabled
            if self.config.enable_background_updates:
                self._schedule_next_update()
            
            self.logger.info(f"🔄 Screen context monitoring started (interval: {self.config.update_interval}s)")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start monitoring: {e}")
            self.is_running = False
            return False
    
    def stop_monitoring(self):
        """Stop screen content monitoring"""
        if not self.is_running:
            return
        
        self.logger.info("🛑 Stopping screen context monitoring...")
        self.is_running = False
        
        # Cancel update timer
        if self.update_timer:
            self.update_timer.cancel()
            self.update_timer = None
        
        self.logger.info("✅ Screen context monitoring stopped")
    
    async def get_latest_content(self) -> Optional[str]:
        """
        Get the latest formatted screen content for Realtime API
        This is the main interface method expected by RealtimeVoiceService
        """
        try:
            # Check if we have recent cached content
            if (self.current_content and 
                self.config.enable_content_caching and
                time.time() - self.last_update_time < self.config.cache_duration):
                
                return self.current_content.to_context_string()
            
            # Force a fresh capture if needed
            await self._capture_and_process_content()
            
            if self.current_content:
                return self.current_content.to_context_string()
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get latest content: {e}")
            return None
    
    async def get_current_context_data(self) -> Optional[ScreenContextData]:
        """Get the current screen context data object"""
        if not self.current_content:
            await self._capture_and_process_content()
        return self.current_content
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        return {
            "total_updates": self.total_updates,
            "significant_changes": self.significant_changes,
            "privacy_filters_applied": self.privacy_filters_applied,
            "change_rate": self.significant_changes / max(1, self.total_updates),
            "privacy_filter_rate": self.privacy_filters_applied / max(1, self.total_updates),
            "last_update_time": self.last_update_time,
            "is_running": self.is_running,
            "cache_size": len(self.content_cache),
            "current_content_length": len(self.current_content.content) if self.current_content else 0,
            "current_word_count": self.current_content.word_count if self.current_content else 0
        }
    
    def _on_screen_capture(self, capture: ScreenCapture):
        """Handle screen capture events from ScreenWatcher"""
        try:
            # Process capture asynchronously
            asyncio.create_task(self._process_screen_capture(capture))
        except Exception as e:
            self.logger.error(f"❌ Error handling screen capture: {e}")
    
    async def _process_screen_capture(self, capture: ScreenCapture):
        """Process a screen capture into context data"""
        try:
            # Format the content
            context_data = self.formatter.format_content(capture)
            
            # Calculate change score if we have previous content
            if self.current_content:
                context_data.change_score = self.formatter.calculate_change_score(
                    context_data.content, 
                    self.current_content.content
                )
            else:
                context_data.change_score = 1.0  # First content is always a complete change
            
            # Update metrics
            self.total_updates += 1
            if context_data.privacy_filtered:
                self.privacy_filters_applied += 1
            
            # Check for significant changes
            if context_data.is_significant_change(self.config.min_change_threshold):
                self.significant_changes += 1
                self.current_content = context_data
                self.last_update_time = time.time()
                
                # Trigger callbacks
                if self.on_significant_change:
                    self.on_significant_change(context_data)
                
                if self.on_content_changed:
                    self.on_content_changed(context_data)
                
                # Update cache
                if self.config.enable_content_caching:
                    self.content_cache[context_data.source_hash] = context_data
                
                self.logger.debug(f"📱 Screen content updated: {context_data.word_count} words, "
                                f"change: {context_data.change_score:.2f}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process screen capture: {e}")
    
    async def _capture_and_process_content(self):
        """Force capture and process screen content"""
        try:
            if not self.screen_watcher:
                return
            
            # Get latest capture from screen watcher
            capture = self.screen_watcher._capture_screen()
            if capture:
                await self._process_screen_capture(capture)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to capture and process content: {e}")
    
    def _schedule_next_update(self):
        """Schedule the next background content update"""
        if not self.is_running:
            return
        
        def update_callback():
            if self.is_running:
                try:
                    asyncio.create_task(self._capture_and_process_content())
                except Exception as e:
                    self.logger.error(f"❌ Error in background update: {e}")
                finally:
                    # Schedule next update
                    self._schedule_next_update()
        
        self.update_timer = threading.Timer(self.config.update_interval, update_callback)
        self.update_timer.daemon = True
        self.update_timer.start()


def create_screen_context_provider(
    config: Optional[ScreenContextConfig] = None,
    screen_watcher: Optional[ScreenWatcher] = None,
    logger: Optional[logging.Logger] = None
) -> ScreenContextProvider:
    """Factory function to create ScreenContextProvider with proper configuration"""
    
    if not config:
        config = ScreenContextConfig()
    
    provider = ScreenContextProvider(
        config=config,
        screen_watcher=screen_watcher,
        logger=logger
    )
    
    return provider


def get_default_screen_context_config() -> ScreenContextConfig:
    """Get default configuration optimized for Realtime API usage"""
    return ScreenContextConfig(
        update_interval=5.0,
        min_change_threshold=0.15,
        max_content_length=1200,
        min_confidence_threshold=65.0,
        enable_privacy_mode=True,
        enable_content_caching=True,
        enable_background_updates=True
    ) 