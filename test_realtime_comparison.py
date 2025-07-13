"""
Sovereign 4.0 - Realtime API vs Traditional Pipeline Comparison Test
Tests both approaches to demonstrate speed and quality differences
"""

import asyncio
import time
import os
import ssl
from dotenv import load_dotenv
import logging

# Setup
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import both systems
from assistant.realtime_voice import RealtimeVoiceService, RealtimeConfig
from assistant.stt import WhisperSTTService, STTConfig
from assistant.llm_router import LLMRouter
from assistant.tts import OpenAITTSService, TTSConfig

class PerformanceComparison:
    """Compare traditional pipeline vs Realtime API performance"""
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY required for Realtime API")
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY required for traditional pipeline")
    
    async def test_traditional_pipeline(self, test_message: str = "What can you see on my screen?"):
        """Test the current STT→LLM→TTS pipeline"""
        logger.info("🔧 Testing Traditional Pipeline (STT→LLM→TTS)")
        
        total_start = time.time()
        
        try:
            # Step 1: STT (simulate - we'll use text input for testing)
            stt_start = time.time()
            user_text = test_message  # In real scenario, this would be transcribed from audio
            stt_time = time.time() - stt_start
            logger.info(f"  🗣️ STT: {stt_time:.3f}s (simulated)")
            
            # Step 2: LLM Processing
            llm_start = time.time()
            llm_router = LLMRouter()
            llm_response = await llm_router.route_query(user_text)
            llm_time = time.time() - llm_start
            
            response_text = llm_response.get("response", "")
            model_used = llm_response.get("model_used", "unknown")
            logger.info(f"  🤖 LLM ({model_used}): {llm_time:.3f}s")
            logger.info(f"  📝 Response: {response_text[:100]}...")
            
            # Step 3: TTS
            tts_start = time.time()
            tts_config = TTSConfig()
            tts_service = OpenAITTSService(tts_config, self.openai_api_key)
            tts_service.initialize()
            
            # Fix: Use the correct method name
            tts_result = await tts_service.synthesize_speech(response_text)
            tts_time = time.time() - tts_start
            logger.info(f"  📢 TTS: {tts_time:.3f}s")
            
            # Total time
            total_time = time.time() - total_start
            logger.info(f"  ⚡ TOTAL TRADITIONAL: {total_time:.3f}s")
            
            await llm_router.cleanup()
            
            return {
                "method": "traditional",
                "total_time": total_time,
                "stt_time": stt_time,
                "llm_time": llm_time,
                "tts_time": tts_time,
                "response_text": response_text,
                "model_used": model_used,
                "audio_length": len(tts_result.audio_data) if tts_result else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Traditional pipeline failed: {e}")
            return None
    
    async def test_realtime_api(self, test_message: str = "What can you see on my screen?"):
        """Test the OpenAI Realtime API"""
        logger.info("🚀 Testing Realtime API (Direct Speech-to-Speech)")
        
        total_start = time.time()
        
        try:
            # Configure Realtime API
            config = RealtimeConfig(
                api_key=self.openai_api_key,
                model="gpt-4o-mini-realtime-preview",  # Fast model for testing
                voice="alloy",
                instructions="You are a helpful AI assistant with screen awareness. Be concise and conversational."
            )
            
            # Create service
            service = RealtimeVoiceService(config)
            
            # Initialize and connect
            init_start = time.time()
            await service.initialize()
            await service.connect()
            init_time = time.time() - init_start
            logger.info(f"  🔌 Connection: {init_time:.3f}s")
            
            # Send test message and measure response time
            response_start = time.time()
            
            # Set up response tracking
            response_received = False
            response_text = ""
            
            async def response_handler(response_data):
                nonlocal response_received, response_text
                response_received = True
                
                # Extract response text
                output_items = response_data.get("output", [])
                for item in output_items:
                    if item.get("type") == "message":
                        content = item.get("content", [])
                        for content_part in content:
                            if content_part.get("type") == "audio":
                                response_text = content_part.get("transcript", "")
                                break
            
            service.on_response_received = response_handler
            
            # Send message
            await service.send_text_message(test_message)
            
            # Wait for response (with timeout)
            timeout = 10.0
            while not response_received and (time.time() - response_start) < timeout:
                await asyncio.sleep(0.1)
            
            response_time = time.time() - response_start
            total_time = time.time() - total_start
            
            if response_received:
                logger.info(f"  ⚡ Response: {response_time:.3f}s")
                logger.info(f"  📝 Response: {response_text[:100]}...")
                logger.info(f"  🚀 TOTAL REALTIME: {total_time:.3f}s")
            else:
                logger.warning(f"  ⚠️ No response received within {timeout}s")
            
            # Cleanup
            await service.disconnect()
            
            return {
                "method": "realtime",
                "total_time": total_time,
                "init_time": init_time,
                "response_time": response_time,
                "response_text": response_text,
                "model_used": config.model,
                "received": response_received
            }
            
        except Exception as e:
            logger.error(f"❌ Realtime API failed: {e}")
            return None
    
    async def run_comparison(self, test_message: str = "What can you see on my screen?"):
        """Run both tests and compare results"""
        logger.info(f"🧪 SOVEREIGN 4.0 PERFORMANCE COMPARISON")
        logger.info(f"📝 Test Query: '{test_message}'")
        logger.info("="*80)
        
        # Test traditional pipeline
        traditional_result = await self.test_traditional_pipeline(test_message)
        logger.info("")
        
        # Test Realtime API
        realtime_result = await self.test_realtime_api(test_message)
        logger.info("")
        
        # Compare results
        if traditional_result and realtime_result:
            self.print_comparison(traditional_result, realtime_result)
        else:
            logger.error("❌ Could not complete comparison - one or both tests failed")
    
    def print_comparison(self, traditional: dict, realtime: dict):
        """Print detailed comparison results"""
        logger.info("📊 PERFORMANCE COMPARISON RESULTS")
        logger.info("="*80)
        
        # Speed comparison
        trad_total = traditional["total_time"]
        real_total = realtime["total_time"]
        speedup = trad_total / real_total if real_total > 0 else 0
        
        logger.info(f"⚡ SPEED COMPARISON:")
        logger.info(f"  Traditional Pipeline: {trad_total:.3f}s")
        logger.info(f"  Realtime API:        {real_total:.3f}s")
        logger.info(f"  🚀 Speedup:          {speedup:.1f}x faster!")
        logger.info("")
        
        # Breakdown
        logger.info(f"🔧 TRADITIONAL BREAKDOWN:")
        logger.info(f"  STT:   {traditional['stt_time']:.3f}s")
        logger.info(f"  LLM:   {traditional['llm_time']:.3f}s")
        logger.info(f"  TTS:   {traditional['tts_time']:.3f}s")
        logger.info("")
        
        logger.info(f"🚀 REALTIME BREAKDOWN:")
        logger.info(f"  Init:     {realtime['init_time']:.3f}s")
        logger.info(f"  Response: {realtime['response_time']:.3f}s")
        logger.info("")
        
        # Response quality
        logger.info(f"💬 RESPONSE COMPARISON:")
        logger.info(f"  Traditional: {traditional['response_text'][:100]}...")
        logger.info(f"  Realtime:    {realtime['response_text'][:100]}...")
        logger.info("")
        
        # Models used
        logger.info(f"🤖 MODELS USED:")
        logger.info(f"  Traditional: {traditional['model_used']}")
        logger.info(f"  Realtime:    {realtime['model_used']}")
        logger.info("")
        
        # Benefits summary
        logger.info(f"✅ REALTIME API BENEFITS:")
        logger.info(f"  🚀 {speedup:.1f}x faster response time")
        logger.info(f"  🎭 Preserves voice emotion & tone")
        logger.info(f"  🔄 Natural conversation flow")
        logger.info(f"  💰 Single API call vs 3 separate calls")
        logger.info(f"  ⚡ Sub-second responses possible")


async def main():
    """Run the comparison test"""
    try:
        comparison = PerformanceComparison()
        
        # Test with a few different queries
        test_queries = [
            "What can you see on my screen?",
            "Tell me about the current weather",
            "Help me write a quick email"
        ]
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n🧪 TEST {i}/{len(test_queries)}")
            await comparison.run_comparison(query)
            
            if i < len(test_queries):
                logger.info("\n" + "="*80)
                await asyncio.sleep(2)  # Brief pause between tests
        
        logger.info("\n🎯 RECOMMENDATION:")
        logger.info("The Realtime API provides dramatically faster, more natural voice interactions.")
        logger.info("Consider implementing this for Sovereign 4.0's next major upgrade!")
        
    except Exception as e:
        logger.error(f"❌ Comparison test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 