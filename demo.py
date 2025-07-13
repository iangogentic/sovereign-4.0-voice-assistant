#!/usr/bin/env python3
"""
Simple demo script to test the Voice Assistant Pipeline
"""

import asyncio
import logging
import os
import threading
import time
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import our components
from assistant.audio import AudioManager, AudioConfig
from assistant.stt import WhisperSTTService, STTConfig
from assistant.tts import OpenAITTSService, TTSConfig
from assistant.pipeline import VoiceAssistantPipeline, PipelineConfig
from assistant.monitoring import PerformanceMonitor
from assistant.dashboard import ConsoleDashboard

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleVoiceAssistant:
    """Simple voice assistant for testing"""
    
    def __init__(self):
        self.pipeline = None
        self.dashboard = None
        self.monitor = None
        self.dashboard_thread = None
        self.running = False
        
    def process_text(self, text: str) -> str:
        """Simple text processing - just echo back what was said"""
        logger.info(f"Processing text: {text}")
        
        # Simple responses for testing
        if "hello" in text.lower():
            return "Hello! I'm your voice assistant. I can hear you clearly."
        elif "how are you" in text.lower():
            return "I'm doing well, thank you for asking!"
        elif "goodbye" in text.lower() or "bye" in text.lower():
            return "Goodbye! It was nice talking with you."
        elif "test" in text.lower():
            return "Test successful! The voice pipeline is working correctly."
        else:
            return f"I heard you say: {text}. This is just a simple echo response."
    
    async def initialize(self):
        """Initialize the voice assistant components"""
        logger.info("Initializing voice assistant...")
        
        # Check API key
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY not found in environment variables")
            return False
        
        try:
            # Initialize performance monitoring
            self.monitor = PerformanceMonitor()
            
            # Initialize dashboard
            self.dashboard = ConsoleDashboard(self.monitor)
            
            # Initialize audio components
            audio_config = AudioConfig(
                sample_rate=16000,
                chunk_size=1024,
                channels=1
            )
            audio_manager = AudioManager(config=audio_config)
            audio_manager.initialize()
            
            # Initialize STT service
            stt_config = STTConfig(
                model="whisper-1",  # Use OpenAI's Whisper API
                language="en"
            )
            stt_service = WhisperSTTService(
                config=stt_config,
                api_key=os.getenv("OPENAI_API_KEY"),
                org_id=os.getenv("OPENAI_ORG_ID")
            )
            
            # Initialize TTS service
            tts_config = TTSConfig(
                model="tts-1",
                voice="alloy",
                speed=1.0
            )
            tts_service = OpenAITTSService(
                config=tts_config,
                api_key=os.getenv("OPENAI_API_KEY"),
                org_id=os.getenv("OPENAI_ORG_ID")
            )
            
            # Initialize pipeline
            pipeline_config = PipelineConfig(
                trigger_key="space",
                trigger_key_name="spacebar"
            )
            self.pipeline = VoiceAssistantPipeline(
                config=pipeline_config,
                audio_manager=audio_manager,
                stt_service=stt_service,
                tts_service=tts_service,
                response_callback=self.process_text,
                performance_monitor=self.monitor
            )
            
            # Initialize the pipeline
            await self.pipeline.initialize()
            
            logger.info("Voice assistant initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize voice assistant: {e}")
            return False
    
    def start_dashboard(self):
        """Start the dashboard in a separate thread"""
        try:
            asyncio.run(self.dashboard.run())
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
    
    def run(self):
        """Run the voice assistant"""
        try:
            # Initialize first
            if not asyncio.run(self.initialize()):
                return
            
            # Start dashboard in background thread
            self.dashboard_thread = threading.Thread(
                target=self.start_dashboard,
                daemon=True
            )
            self.dashboard_thread.start()
            
            logger.info("🎤 Voice Assistant is running!")
            logger.info("📋 Dashboard is showing performance metrics")
            logger.info("🎯 Press SPACEBAR to start recording, release to stop")
            logger.info("⏹️  Press Ctrl+C to exit")
            
            # Check for keyboard permissions
            try:
                import keyboard
                keyboard.is_pressed('space')  # Test if we can access keyboard
                logger.info("✅ Keyboard access available - push-to-talk will work")
            except Exception as e:
                logger.warning("⚠️  Keyboard access not available (this is common on macOS)")
                logger.warning("⚠️  You may need to grant accessibility permissions or run with sudo")
                logger.warning("⚠️  The pipeline will still work but push-to-talk may not function")
            
            # Run the pipeline (this blocks until stopped)
            self.pipeline.run()
            
        except KeyboardInterrupt:
            logger.info("Shutting down voice assistant...")
        except Exception as e:
            logger.error(f"Error during operation: {e}")
        finally:
            # Clean shutdown
            if self.pipeline:
                self.pipeline.stop()
            
            # Show final statistics
            if self.pipeline:
                stats = self.pipeline.get_statistics()
                logger.info("Final Statistics:")
                logger.info(f"  Total Sessions: {stats.get('total_sessions', 0)}")
                logger.info(f"  Successful Sessions: {stats.get('successful_sessions', 0)}")
                logger.info(f"  Average Session Duration: {stats.get('average_session_duration', 0):.2f}s")

def main():
    """Main function"""
    assistant = SimpleVoiceAssistant()
    assistant.run()

if __name__ == "__main__":
    main() 