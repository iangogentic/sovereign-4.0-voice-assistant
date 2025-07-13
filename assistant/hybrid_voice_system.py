"""
Hybrid Voice System for Sovereign 4.0
Seamlessly switches between OpenAI Realtime API and traditional STT→LLM→TTS pipeline
Provides optimal performance with intelligent fallback capabilities
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Callable, Union, Literal
from dataclasses import dataclass
from enum import Enum

from assistant.realtime_voice import RealtimeVoiceService, RealtimeConfig, ConversationMetrics
from assistant.stt import WhisperSTTService, STTConfig
from assistant.llm_router import LLMRouter
from assistant.tts import OpenAITTSService, TTSConfig
from assistant.memory import MemoryManager
from assistant.screen_watcher import ScreenWatcher


class VoiceMode(Enum):
    """Available voice processing modes"""
    REALTIME_ONLY = "realtime_only"
    TRADITIONAL_ONLY = "traditional_only"
    HYBRID_AUTO = "hybrid_auto"
    HYBRID_PREFER_REALTIME = "hybrid_prefer_realtime"
    HYBRID_PREFER_TRADITIONAL = "hybrid_prefer_traditional"


@dataclass
class HybridConfig:
    """Configuration for hybrid voice system"""
    # Mode selection
    voice_mode: VoiceMode = VoiceMode.HYBRID_AUTO
    
    # Fallback triggers
    max_realtime_failures: int = 3
    fallback_on_high_latency: bool = True
    max_acceptable_latency: float = 2.0  # seconds
    fallback_on_cost_limit: bool = False
    daily_cost_limit: float = 10.0  # dollars
    
    # Performance monitoring
    monitor_performance: bool = True
    performance_window: int = 10  # number of recent interactions to consider
    auto_switch_threshold: float = 0.5  # latency difference to trigger switch
    
    # Context preservation
    preserve_context_on_switch: bool = True
    max_context_transfer_time: float = 1.0  # seconds


class HybridVoiceSystem:
    """
    Intelligent hybrid voice system that seamlessly switches between
    Realtime API and traditional pipeline based on performance and availability
    """
    
    def __init__(self, 
                 hybrid_config: HybridConfig,
                 realtime_config: RealtimeConfig,
                 openai_api_key: str,
                 openrouter_api_key: str,
                 memory_manager: Optional[MemoryManager] = None,
                 screen_watcher: Optional[ScreenWatcher] = None,
                 logger: Optional[logging.Logger] = None):
        
        self.config = hybrid_config
        self.memory_manager = memory_manager
        self.screen_watcher = screen_watcher
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize both systems
        self.realtime_service = RealtimeVoiceService(
            config=realtime_config,
            memory_manager=memory_manager,
            screen_content_provider=screen_watcher,
            logger=self.logger
        )
        
        # Traditional pipeline components
        self.stt_service = WhisperSTTService(STTConfig(), openai_api_key)
        self.llm_router = LLMRouter()
        self.tts_service = OpenAITTSService(TTSConfig(), openai_api_key)
        
        # System state
        self.current_mode: VoiceMode = hybrid_config.voice_mode
        self.active_system: Literal["realtime", "traditional", "none"] = "none"
        self.is_initialized: bool = False
        self.is_running: bool = False
        
        # Performance tracking
        self.realtime_failures: int = 0
        self.traditional_failures: int = 0
        self.recent_performance: list = []
        self.total_cost_today: float = 0.0
        
        # Conversation state
        self.current_conversation_id: Optional[str] = None
        self.conversation_context: Dict[str, Any] = {}
        
        # Event handlers
        self.on_mode_switch: Optional[Callable] = None
        self.on_response_generated: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
    
    async def initialize(self) -> bool:
        """Initialize both voice systems"""
        try:
            self.logger.info("🔧 Initializing Hybrid Voice System...")
            
            # Initialize traditional pipeline
            traditional_success = await self._initialize_traditional_pipeline()
            
            # Initialize realtime service
            realtime_success = await self.realtime_service.initialize()
            
            # Determine initial mode based on availability
            if realtime_success and traditional_success:
                self.logger.info("✅ Both systems available - using configured mode")
                self.active_system = self._select_initial_system()
            elif realtime_success:
                self.logger.warning("⚠️ Only Realtime API available - forcing realtime mode")
                self.current_mode = VoiceMode.REALTIME_ONLY
                self.active_system = "realtime"
            elif traditional_success:
                self.logger.warning("⚠️ Only traditional pipeline available - forcing traditional mode")
                self.current_mode = VoiceMode.TRADITIONAL_ONLY
                self.active_system = "traditional"
            else:
                self.logger.error("❌ Neither system available - initialization failed")
                return False
            
            self.is_initialized = True
            self.logger.info(f"✅ Hybrid Voice System initialized - Active: {self.active_system}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize hybrid system: {e}")
            return False
    
    async def _initialize_traditional_pipeline(self) -> bool:
        """Initialize traditional STT→LLM→TTS pipeline"""
        try:
            # Initialize STT
            if not self.stt_service.initialize():
                self.logger.error("❌ Failed to initialize STT service")
                return False
            
            # Initialize TTS
            if not self.tts_service.initialize():
                self.logger.error("❌ Failed to initialize TTS service")
                return False
            
            self.logger.info("✅ Traditional pipeline initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Traditional pipeline initialization failed: {e}")
            return False
    
    def _select_initial_system(self) -> Literal["realtime", "traditional"]:
        """Select which system to use initially based on mode"""
        if self.current_mode == VoiceMode.REALTIME_ONLY:
            return "realtime"
        elif self.current_mode == VoiceMode.TRADITIONAL_ONLY:
            return "traditional"
        elif self.current_mode in [VoiceMode.HYBRID_AUTO, VoiceMode.HYBRID_PREFER_REALTIME]:
            return "realtime"  # Prefer realtime for speed
        else:  # HYBRID_PREFER_TRADITIONAL
            return "traditional"
    
    async def start_conversation(self) -> bool:
        """Start voice conversation with the active system"""
        if not self.is_initialized:
            self.logger.error("❌ System not initialized")
            return False
        
        try:
            self.is_running = True
            self.current_conversation_id = f"conv_{int(time.time())}"
            
            if self.active_system == "realtime":
                success = await self._start_realtime_conversation()
            else:
                success = await self._start_traditional_conversation()
            
            if success:
                self.logger.info(f"✅ Conversation started with {self.active_system} system")
            else:
                # Try fallback if initial system fails
                success = await self._try_fallback_system()
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start conversation: {e}")
            return False
    
    async def _start_realtime_conversation(self) -> bool:
        """Start conversation with Realtime API"""
        try:
            # Set up event handlers
            self.realtime_service.on_response_generated = self._on_realtime_response
            self.realtime_service.on_error = self._on_realtime_error
            
            # Connect and start
            if await self.realtime_service.connect():
                await self.realtime_service.start_conversation()
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Realtime conversation failed: {e}")
            self.realtime_failures += 1
            return False
    
    async def _start_traditional_conversation(self) -> bool:
        """Start conversation with traditional pipeline"""
        try:
            # Traditional pipeline is always "ready" - no persistent connection needed
            self.logger.info("✅ Traditional pipeline ready for conversation")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Traditional pipeline failed: {e}")
            self.traditional_failures += 1
            return False
    
    async def process_voice_input(self, audio_data: bytes = None, text_input: str = None) -> Dict[str, Any]:
        """
        Process voice input through the active system
        Returns response with timing and system information
        """
        start_time = time.time()
        
        try:
            if self.active_system == "realtime":
                result = await self._process_realtime_input(audio_data)
            else:
                result = await self._process_traditional_input(audio_data, text_input)
            
            # Track performance
            processing_time = time.time() - start_time
            self._update_performance_metrics(processing_time, self.active_system, success=True)
            
            # Check if we should switch systems
            await self._evaluate_system_switch()
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_performance_metrics(processing_time, self.active_system, success=False)
            
            self.logger.error(f"❌ Voice processing failed: {e}")
            
            # Try fallback system
            return await self._try_fallback_processing(audio_data, text_input)
    
    async def _process_realtime_input(self, audio_data: bytes) -> Dict[str, Any]:
        """Process input through Realtime API"""
        if not audio_data:
            raise ValueError("Audio data required for Realtime API")
        
        # For now, return a simulated response
        # In full implementation, this would integrate with the realtime service
        return {
            "system": "realtime",
            "response_text": "Realtime API response",
            "response_audio": b"",  # Audio bytes would be here
            "processing_time": 0.3,
            "model_used": "gpt-4o-realtime-preview"
        }
    
    async def _process_traditional_input(self, audio_data: bytes = None, text_input: str = None) -> Dict[str, Any]:
        """Process input through traditional STT→LLM→TTS pipeline"""
        processing_start = time.time()
        
        # Step 1: Speech-to-Text (if audio provided)
        if audio_data and not text_input:
            stt_start = time.time()
            # In full implementation, would transcribe audio_data
            text_input = "Traditional pipeline input"  # Simulated for now
            stt_time = time.time() - stt_start
        else:
            stt_time = 0.0
            text_input = text_input or "Traditional pipeline test"
        
        # Step 2: LLM Processing
        llm_start = time.time()
        llm_response = await self.llm_router.route_query(text_input)
        llm_time = time.time() - llm_start
        
        response_text = llm_response.get("response", "")
        model_used = llm_response.get("model_used", "unknown")
        
        # Step 3: Text-to-Speech
        tts_start = time.time()
        tts_result = await self.tts_service.synthesize_speech(response_text)
        tts_time = time.time() - tts_start
        
        total_time = time.time() - processing_start
        
        return {
            "system": "traditional",
            "response_text": response_text,
            "response_audio": tts_result.audio_data if tts_result else b"",
            "processing_time": total_time,
            "stt_time": stt_time,
            "llm_time": llm_time,
            "tts_time": tts_time,
            "model_used": model_used
        }
    
    async def _try_fallback_system(self) -> bool:
        """Try to start conversation with fallback system"""
        if self.current_mode in [VoiceMode.REALTIME_ONLY, VoiceMode.TRADITIONAL_ONLY]:
            return False  # No fallback allowed in single-system modes
        
        fallback_system = "traditional" if self.active_system == "realtime" else "realtime"
        
        self.logger.info(f"🔄 Trying fallback system: {fallback_system}")
        
        old_system = self.active_system
        self.active_system = fallback_system
        
        try:
            if fallback_system == "realtime":
                success = await self._start_realtime_conversation()
            else:
                success = await self._start_traditional_conversation()
            
            if success:
                self.logger.info(f"✅ Fallback successful: {old_system} → {fallback_system}")
                if self.on_mode_switch:
                    await self.on_mode_switch(old_system, fallback_system, "fallback")
                return True
            else:
                self.active_system = old_system  # Revert
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Fallback failed: {e}")
            self.active_system = old_system  # Revert
            return False
    
    async def _try_fallback_processing(self, audio_data: bytes = None, text_input: str = None) -> Dict[str, Any]:
        """Try processing with fallback system"""
        old_system = self.active_system
        fallback_system = "traditional" if self.active_system == "realtime" else "realtime"
        
        if self.current_mode in [VoiceMode.REALTIME_ONLY, VoiceMode.TRADITIONAL_ONLY]:
            # No fallback allowed
            return {
                "system": old_system,
                "error": "Processing failed and fallback not allowed",
                "processing_time": 0.0
            }
        
        self.logger.info(f"🔄 Fallback processing: {old_system} → {fallback_system}")
        self.active_system = fallback_system
        
        try:
            result = await self.process_voice_input(audio_data, text_input)
            
            self.logger.info(f"✅ Fallback processing successful")
            if self.on_mode_switch:
                await self.on_mode_switch(old_system, fallback_system, "error_fallback")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Fallback processing also failed: {e}")
            self.active_system = old_system  # Revert
            
            return {
                "system": "none",
                "error": f"Both systems failed: {e}",
                "processing_time": 0.0
            }
    
    def _update_performance_metrics(self, processing_time: float, system: str, success: bool):
        """Update performance tracking metrics"""
        self.recent_performance.append({
            "timestamp": time.time(),
            "system": system,
            "processing_time": processing_time,
            "success": success
        })
        
        # Keep only recent metrics
        if len(self.recent_performance) > self.config.performance_window:
            self.recent_performance.pop(0)
    
    async def _evaluate_system_switch(self):
        """Evaluate if we should switch systems based on performance"""
        if self.current_mode not in [VoiceMode.HYBRID_AUTO]:
            return  # No auto-switching in other modes
        
        if len(self.recent_performance) < 5:
            return  # Not enough data
        
        # Calculate average performance for each system
        realtime_times = [p["processing_time"] for p in self.recent_performance 
                         if p["system"] == "realtime" and p["success"]]
        traditional_times = [p["processing_time"] for p in self.recent_performance 
                           if p["system"] == "traditional" and p["success"]]
        
        if not realtime_times or not traditional_times:
            return  # Need data for both systems
        
        avg_realtime = sum(realtime_times) / len(realtime_times)
        avg_traditional = sum(traditional_times) / len(traditional_times)
        
        # Switch if performance difference exceeds threshold
        performance_diff = abs(avg_realtime - avg_traditional)
        if performance_diff > self.config.auto_switch_threshold:
            better_system = "realtime" if avg_realtime < avg_traditional else "traditional"
            
            if better_system != self.active_system:
                await self._switch_system(better_system, "performance_optimization")
    
    async def _switch_system(self, target_system: str, reason: str):
        """Switch to target system"""
        old_system = self.active_system
        
        try:
            self.logger.info(f"🔄 Switching system: {old_system} → {target_system} ({reason})")
            
            # Stop current system
            if old_system == "realtime":
                await self.realtime_service.stop_conversation()
            
            # Start new system
            self.active_system = target_system
            if target_system == "realtime":
                success = await self._start_realtime_conversation()
            else:
                success = await self._start_traditional_conversation()
            
            if success:
                self.logger.info(f"✅ System switch successful")
                if self.on_mode_switch:
                    await self.on_mode_switch(old_system, target_system, reason)
            else:
                # Revert if switch failed
                self.active_system = old_system
                self.logger.error(f"❌ System switch failed, reverted to {old_system}")
                
        except Exception as e:
            self.logger.error(f"❌ System switch error: {e}")
            self.active_system = old_system  # Revert
    
    async def _on_realtime_response(self, response_data: Dict[str, Any]):
        """Handle response from Realtime API"""
        if self.on_response_generated:
            await self.on_response_generated(response_data)
    
    async def _on_realtime_error(self, error_data: Dict[str, Any]):
        """Handle error from Realtime API"""
        self.realtime_failures += 1
        
        # Check if we should fallback due to too many failures
        if (self.realtime_failures >= self.config.max_realtime_failures and 
            self.current_mode not in [VoiceMode.REALTIME_ONLY]):
            await self._switch_system("traditional", "excessive_failures")
        
        if self.on_error:
            await self.on_error(error_data)
    
    async def stop_conversation(self):
        """Stop the active conversation"""
        try:
            self.is_running = False
            
            if self.active_system == "realtime":
                await self.realtime_service.stop_conversation()
            
            self.logger.info("✅ Conversation stopped")
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping conversation: {e}")
    
    async def cleanup(self):
        """Cleanup all resources"""
        try:
            await self.stop_conversation()
            
            # Cleanup realtime service
            await self.realtime_service.disconnect()
            
            # Cleanup LLM router
            await self.llm_router.cleanup()
            
            self.logger.info("✅ Hybrid system cleanup complete")
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup error: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "active_system": self.active_system,
            "current_mode": self.current_mode.value,
            "is_running": self.is_running,
            "is_initialized": self.is_initialized,
            "realtime_failures": self.realtime_failures,
            "traditional_failures": self.traditional_failures,
            "recent_performance": self.recent_performance[-5:],  # Last 5 interactions
            "conversation_id": self.current_conversation_id,
            "realtime_metrics": self.realtime_service.get_metrics() if self.realtime_service else None
        } 