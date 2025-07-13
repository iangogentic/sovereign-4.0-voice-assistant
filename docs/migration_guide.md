# Migration Guide: Traditional to Hybrid Mode

Complete guide for migrating from traditional STT→LLM→TTS pipeline to hybrid Realtime API mode.

## Table of Contents

1. [Migration Overview](#migration-overview)
2. [Pre-Migration Assessment](#pre-migration-assessment)
3. [Backup and Preparation](#backup-and-preparation)
4. [Configuration Migration](#configuration-migration)
5. [Code Migration](#code-migration)
6. [Testing and Validation](#testing-and-validation)
7. [Performance Optimization](#performance-optimization)
8. [Rollback Procedures](#rollback-procedures)
9. [Common Migration Issues](#common-migration-issues)
10. [Post-Migration Optimization](#post-migration-optimization)

---

## Migration Overview

### What's Changing

**From Traditional Pipeline:**
```
Audio Input → STT (Whisper) → LLM (GPT) → TTS → Audio Output
Average Latency: 800-1200ms
```

**To Hybrid Mode:**
```
Audio Input → Realtime API (Primary) → Audio Output
             ↓ (Fallback when needed)
Traditional Pipeline (Backup)
Average Latency: <300ms
```

### Breaking Changes

⚠️ **Important Changes in v4.0:**

- **Configuration Structure**: YAML configuration requires new `realtime_api` section
- **API Requirements**: OpenAI Realtime API access required (contact OpenAI if needed)
- **Dependencies**: New Python packages including `websockets`, `pyaudio`, `sentence-transformers`
- **Environment Variables**: New required API keys (`OPENAI_API_KEY` for Realtime API)
- **Audio Permissions**: System audio permissions required for real-time processing
- **Screen Context**: Screen recording permissions required (macOS/Linux) for screen awareness
- **Memory System**: ChromaDB database structure updated - existing databases may need recreation

### Key Benefits

- **3x Faster Response Times**: <300ms vs 800ms+
- **Better Audio Quality**: 24kHz vs 16kHz
- **Seamless Fallback**: Automatic degradation handling  
- **Enhanced Context**: Screen awareness and memory integration
- **Production Ready**: Comprehensive monitoring and reliability

### Migration Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| Assessment | 1-2 days | Evaluate current system, plan migration |
| Preparation | 1 day | Backup, environment setup |
| Migration | 2-3 days | Code changes, configuration updates |
| Testing | 2-3 days | Comprehensive testing and validation |
| Optimization | 1-2 days | Performance tuning |
| **Total** | **7-11 days** | **Complete migration** |

---

## Pre-Migration Assessment

### Current System Analysis

1. **Inventory Current Components:**
   ```bash
   # Check existing configuration
   find . -name "*.yaml" -o -name "*.yml" -o -name "*.json" | head -10
   
   # List current dependencies
   pip freeze > current_requirements.txt
   
   # Analyze current code structure
   find assistant/ -name "*.py" | wc -l
   ```

2. **Performance Baseline:**
   ```bash
   # Run performance assessment
   python -c "
   from assistant.monitoring import PerformanceMonitor
   monitor = PerformanceMonitor()
   baseline = monitor.get_baseline_metrics()
   print(f'Current average latency: {baseline[\"avg_latency\"]}ms')
   print(f'Success rate: {baseline[\"success_rate\"]}%')
   "
   ```

3. **Custom Modifications Audit:**
   ```bash
   # Identify custom changes
   git diff HEAD~10 --name-only
   
   # Check for custom modules
   find . -name "*custom*" -o -name "*local*" -o -name "*override*"
   ```

### Requirements Check

**System Requirements:**
- Python 3.11+ (current: check with `python --version`)
- OpenAI API key with Realtime API access
- Tesseract OCR for screen awareness
- ChromaDB for memory management
- Sufficient system resources (2GB+ RAM, 2+ CPU cores)

**API Access Verification:**
```bash
# Test OpenAI API access
python -c "
import openai
try:
    # Test general API access
    client = openai.OpenAI()
    models = client.models.list()
    print('✓ OpenAI API access confirmed')
    
    # Check for Realtime API access (this will be validated during setup)
    print('Note: Realtime API access will be validated during migration')
except Exception as e:
    print(f'✗ API access issue: {e}')
"
```

---

## Backup and Preparation

### 1. Create Complete Backup

```bash
# Create backup directory
mkdir -p backups/$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"

# Backup configuration files
cp -r config/ $BACKUP_DIR/
cp .env $BACKUP_DIR/ 2>/dev/null || echo "No .env file found"

# Backup current code
cp -r assistant/ $BACKUP_DIR/

# Backup data directory
cp -r data/ $BACKUP_DIR/ 2>/dev/null || echo "No data directory found"

# Backup current requirements
pip freeze > $BACKUP_DIR/requirements_backup.txt

# Create backup manifest
echo "Backup created on $(date)" > $BACKUP_DIR/BACKUP_MANIFEST.txt
echo "Git commit: $(git rev-parse HEAD)" >> $BACKUP_DIR/BACKUP_MANIFEST.txt
echo "Python version: $(python --version)" >> $BACKUP_DIR/BACKUP_MANIFEST.txt
```

### 2. Environment Preparation

```bash
# Update Python packages
pip install --upgrade pip setuptools wheel

# Install additional dependencies for Realtime API
pip install websockets==12.0
pip install asyncio-mqtt==0.16.1
pip install prometheus-client==0.19.0
pip install sentence-transformers
pip install chromadb

# Install optional dependencies for enhanced features
pip install pytesseract Pillow  # Screen awareness
pip install numpy scipy        # Audio processing
pip install psutil            # System monitoring
```

### 3. Validate Dependencies

```bash
# Test critical dependencies
python -c "
import websockets
import asyncio
import chromadb
import sentence_transformers
print('✓ All critical dependencies available')
"

# Test Tesseract OCR
tesseract --version && echo "✓ Tesseract OCR available" || echo "✗ Tesseract OCR missing"

# Test audio system
python -c "
import pyaudio
p = pyaudio.PyAudio()
print(f'✓ Audio system: {p.get_device_count()} devices available')
p.terminate()
"
```

---

## Configuration Migration

### 1. Update Main Configuration

Create new configuration structure for hybrid mode:

```bash
# Create new configuration from template
cp config/sovereign.yaml config/sovereign.yaml.backup
```

**OLD (v3.x) Configuration:**
```yaml
# Traditional pipeline configuration
app:
  name: "Sovereign Voice Assistant"
  version: "3.x"
  mode: "traditional"

# Old STT configuration
stt:
  provider: "whisper"
  model: "base"

# Old LLM configuration  
llm:
  provider: "openai"
  model: "gpt-4"

# Old TTS configuration
tts:
  provider: "openai"
  voice: "alloy"
```

**NEW (v4.0) Configuration:**
```yaml
# Application Configuration
app:
  name: "Sovereign Voice Assistant"
  version: "4.0.0"
  mode: "hybrid"  # NEW: Enable hybrid mode

# NEW: Realtime API Configuration
realtime_api:
  model: "gpt-4o-realtime-preview"
  voice: "alloy"
  sample_rate: 24000
  channels: 1
  format: "pcm16"
  max_tokens: 4096
  temperature: 0.8
  
  # Connection settings
  timeout: 30.0
  max_retries: 3
  retry_delay: 2.0

# NEW: Smart Context Management
smart_context:
  token_budget: 32000
  priorities:
    system: 2000
    memory: 4000
    screen: 2000
    conversation: 24000
  
  relevance:
    enabled: true
    threshold: 0.6
    model: "sentence-transformers/all-MiniLM-L6-v2"
  
  cache:
    enabled: true
    ttl_seconds: 300
    max_size: 100

# NEW: Screen Context Provider
screen_context:
  enabled: true
  capture_interval: 5.0
  ocr:
    engine: "tesseract"
    confidence_threshold: 60
  privacy:
    enabled: true
    filter_sensitive: true

# NEW: Memory Context Provider
memory_context:
  enabled: true
  vector_db:
    type: "chromadb"
    persist_directory: "./data/memory"
    collection_name: "conversations"
  embeddings:
    model: "text-embedding-3-small"
    dimensions: 1536

# UPDATED: Audio Configuration
audio:
  sample_rate: 24000  # CHANGED: from 16000 to 24000
  channels: 1
  chunk_size: 1024
  format: "int16"
  
  # NEW: Enhanced processing
  processing:
    noise_reduction: true
    echo_cancellation: true
    auto_gain_control: true

# NEW: Performance Configuration
performance:
  targets:
    latency_ms: 300
    uptime_percent: 99.9
  
  fallback:
    latency_threshold_ms: 500
    error_rate_threshold: 0.1
    connection_failures: 3

# NEW: Mode Switching
mode_switching:
  enabled: true
  evaluation:
    window_seconds: 60
    min_samples: 10
  conditions:
    latency_degradation: 0.5
    error_rate_increase: 0.1

# UPDATED: Monitoring
monitoring:
  enabled: true
  metrics:
    - "latency"
    - "throughput"
    - "error_rate"
    - "connection_health"
    - "resource_usage"
    - "context_performance"  # NEW
  
  export:
    prometheus:
      enabled: true
      port: 9090

# Preserve existing settings
# ... (keep any existing custom configurations)
```

### 2. Environment Variables Update

Update `.env` file with new requirements:

```bash
# Copy existing .env
cp .env .env.backup

# Add new environment variables
cat >> .env << 'EOF'

# Realtime API Configuration
SOVEREIGN_MODE=hybrid
REALTIME_MODEL=gpt-4o-realtime-preview

# Enhanced Features (Optional)
ANTHROPIC_API_KEY=your-anthropic-key-here
PERPLEXITY_API_KEY=your-perplexity-key-here

# System Configuration  
SOVEREIGN_LOG_LEVEL=INFO
SOVEREIGN_PERFORMANCE_MODE=balanced

# ChromaDB Configuration
CHROMA_DB_PATH=./data/memory
CHROMA_COLLECTION_NAME=conversations

# Tesseract Configuration
TESSERACT_CMD=/usr/local/bin/tesseract
EOF
```

### 3. Update Requirements

```bash
# Update requirements.txt
cat >> requirements.txt << 'EOF'

# Realtime API Dependencies
websockets==12.0
asyncio-mqtt==0.16.1

# Enhanced Context Management
sentence-transformers>=2.2.0
chromadb>=0.4.0

# Screen Context
pytesseract>=0.3.10
Pillow>=9.0.0

# Performance Monitoring
prometheus-client>=0.19.0
psutil>=5.9.0

# Audio Enhancement
numpy>=1.21.0
scipy>=1.9.0
EOF

# Install updated requirements
pip install -r requirements.txt
```

---

## Code Migration

### 1. Update Main Application Entry Point

**Update `assistant/main.py`:**

```python
"""
Sovereign 4.0 Voice Assistant - Hybrid Mode Main Entry Point
"""
import asyncio
import logging
import signal
import sys
from pathlib import Path

# NEW imports for hybrid mode
from assistant.config_manager import ConfigManager
from assistant.hybrid_voice_system import HybridVoiceSystem
from assistant.realtime_dashboard_integration import RealtimeDashboard
from assistant.shutdown_manager import ShutdownManager

# Traditional imports (preserved for fallback)
from assistant.pipeline import VoicePipeline
from assistant.monitoring import PerformanceMonitor

logger = logging.getLogger(__name__)

class SovereignAssistant:
    """Main application class with hybrid mode support."""
    
    def __init__(self, config_path: str = None):
        self.config_manager = ConfigManager(config_path)
        self.shutdown_manager = ShutdownManager()
        
        # NEW: Choose system based on mode
        mode = self.config_manager.get('app.mode', 'hybrid')
        
        if mode == 'hybrid':
            self.voice_system = HybridVoiceSystem(self.config_manager)
            logger.info("Initialized in hybrid mode (Realtime API + fallback)")
        elif mode == 'realtime':
            from assistant.realtime_voice import RealtimeVoiceService
            self.voice_system = RealtimeVoiceService(self.config_manager)
            logger.info("Initialized in realtime-only mode")
        else:
            # Traditional mode (preserved)
            self.voice_system = VoicePipeline(self.config_manager)
            logger.info("Initialized in traditional mode")
        
        # NEW: Dashboard integration
        self.dashboard = RealtimeDashboard(self.config_manager)
        
        # Performance monitoring (enhanced)
        self.monitor = PerformanceMonitor(self.config_manager)
    
    async def start(self):
        """Start the voice assistant system."""
        try:
            logger.info("Starting Sovereign Voice Assistant 4.0...")
            
            # Start monitoring
            await self.monitor.start()
            
            # Start dashboard
            await self.dashboard.start()
            
            # Start voice system
            await self.voice_system.start()
            
            logger.info("System started successfully!")
            logger.info(f"Dashboard available at: http://localhost:8080/dashboard")
            
            # Setup shutdown handlers
            self.shutdown_manager.register_signal_handlers(self.stop)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            await self.stop()
            return False
    
    async def stop(self):
        """Stop the voice assistant system."""
        logger.info("Shutting down Sovereign Voice Assistant...")
        
        try:
            if hasattr(self, 'voice_system'):
                await self.voice_system.stop()
            
            if hasattr(self, 'dashboard'):
                await self.dashboard.stop()
                
            if hasattr(self, 'monitor'):
                await self.monitor.stop()
                
            logger.info("Shutdown completed successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

async def main():
    """Main application entry point."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start assistant
    assistant = SovereignAssistant()
    
    try:
        success = await assistant.start()
        if not success:
            sys.exit(1)
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await assistant.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Create Hybrid Voice System

**Create `assistant/hybrid_voice_system.py`:**

```python
"""
Hybrid Voice System - Orchestrates Realtime API with Traditional Fallback
"""
import asyncio
import logging
from typing import Optional, Dict, Any

from assistant.realtime_voice import RealtimeVoiceService
from assistant.mode_switch_manager import ModeSwitchManager
from assistant.fallback_detector import FallbackDetector
from assistant.smart_context_manager import SmartContextManager

# Traditional components (preserved for fallback)
from assistant.pipeline import VoicePipeline

logger = logging.getLogger(__name__)

class HybridVoiceSystem:
    """
    Hybrid voice system that intelligently switches between 
    Realtime API and traditional pipeline based on performance.
    """
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.current_mode = "realtime"
        
        # Initialize Realtime API system (primary)
        self.realtime_service = RealtimeVoiceService(config_manager)
        
        # Initialize traditional pipeline (fallback)
        self.traditional_pipeline = VoicePipeline(config_manager)
        
        # Smart systems
        self.context_manager = SmartContextManager(config_manager)
        self.mode_switcher = ModeSwitchManager(config_manager)
        self.fallback_detector = FallbackDetector(config_manager)
        
        # State tracking
        self.is_running = False
        self.performance_metrics = {}
    
    async def start(self):
        """Start the hybrid voice system."""
        try:
            logger.info("Starting hybrid voice system...")
            
            # Start context management
            await self.context_manager.start()
            
            # Start mode switching and fallback detection
            await self.mode_switcher.start()
            await self.fallback_detector.start()
            
            # Start with Realtime API
            await self._start_realtime_mode()
            
            self.is_running = True
            logger.info("Hybrid voice system started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start hybrid system: {e}")
            await self.stop()
            raise
    
    async def process_voice_input(self, audio_data: bytes) -> Optional[bytes]:
        """
        Process voice input through the appropriate pipeline.
        """
        if not self.is_running:
            return None
        
        try:
            # Build smart context
            context = await self.context_manager.build_context(
                query="",  # Will be updated with transcription
                include_memory=True,
                include_screen=True
            )
            
            # Process through current mode
            if self.current_mode == "realtime":
                response = await self._process_realtime(audio_data, context)
            else:
                response = await self._process_traditional(audio_data, context)
            
            # Check if mode switch is needed
            await self._evaluate_mode_switch()
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing voice input: {e}")
            
            # Try fallback if primary mode fails
            if self.current_mode == "realtime":
                logger.info("Attempting fallback to traditional mode")
                await self._switch_to_traditional()
                return await self._process_traditional(audio_data, context)
            
            return None
    
    async def _process_realtime(self, audio_data: bytes, context: Dict) -> Optional[bytes]:
        """Process through Realtime API."""
        try:
            # Update Realtime API context
            await self.realtime_service.update_context(context)
            
            # Send audio and get response
            response = await self.realtime_service.process_audio(audio_data)
            
            # Record successful processing
            self.fallback_detector.record_success()
            
            return response
            
        except Exception as e:
            logger.error(f"Realtime API processing failed: {e}")
            self.fallback_detector.record_failure("realtime_api_error")
            raise
    
    async def _process_traditional(self, audio_data: bytes, context: Dict) -> Optional[bytes]:
        """Process through traditional pipeline."""
        try:
            # Update traditional pipeline context
            self.traditional_pipeline.update_context(context)
            
            # Process through STT -> LLM -> TTS
            response = await self.traditional_pipeline.process_audio(audio_data)
            
            return response
            
        except Exception as e:
            logger.error(f"Traditional pipeline processing failed: {e}")
            raise
    
    async def _evaluate_mode_switch(self):
        """Evaluate if mode switch is needed."""
        try:
            should_switch = await self.mode_switcher.should_switch_mode()
            
            if should_switch:
                if self.current_mode == "realtime":
                    await self._switch_to_traditional()
                else:
                    await self._switch_to_realtime()
                    
        except Exception as e:
            logger.error(f"Mode evaluation failed: {e}")
    
    async def _switch_to_realtime(self):
        """Switch to Realtime API mode."""
        if self.current_mode == "realtime":
            return
        
        try:
            logger.info("Switching to Realtime API mode")
            await self._start_realtime_mode()
            await self._stop_traditional_mode()
            self.current_mode = "realtime"
            
        except Exception as e:
            logger.error(f"Failed to switch to Realtime mode: {e}")
    
    async def _switch_to_traditional(self):
        """Switch to traditional pipeline mode."""
        if self.current_mode == "traditional":
            return
        
        try:
            logger.info("Switching to traditional pipeline mode")
            await self._start_traditional_mode()
            await self._stop_realtime_mode()
            self.current_mode = "traditional"
            
        except Exception as e:
            logger.error(f"Failed to switch to traditional mode: {e}")
    
    async def _start_realtime_mode(self):
        """Start Realtime API service."""
        await self.realtime_service.start()
        logger.info("Realtime API mode active")
    
    async def _stop_realtime_mode(self):
        """Stop Realtime API service."""
        await self.realtime_service.stop()
    
    async def _start_traditional_mode(self):
        """Start traditional pipeline."""
        await self.traditional_pipeline.start()
        logger.info("Traditional pipeline mode active")
    
    async def _stop_traditional_mode(self):
        """Stop traditional pipeline."""
        await self.traditional_pipeline.stop()
    
    async def stop(self):
        """Stop the hybrid voice system."""
        logger.info("Stopping hybrid voice system...")
        
        self.is_running = False
        
        try:
            # Stop both systems
            await self._stop_realtime_mode()
            await self._stop_traditional_mode()
            
            # Stop smart systems
            await self.context_manager.stop()
            await self.mode_switcher.stop()
            await self.fallback_detector.stop()
            
            logger.info("Hybrid voice system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping hybrid system: {e}")
```

### 3. Preserve Existing Functionality

**Ensure backward compatibility:**

```python
# assistant/pipeline.py - Enhanced traditional pipeline
class VoicePipeline:
    """Enhanced traditional pipeline with context support."""
    
    def __init__(self, config_manager):
        # Preserve existing initialization
        self.config = config_manager
        self.stt = STTService(config_manager)
        self.tts = TTSService(config_manager)
        self.llm = LLMService(config_manager)
        
        # NEW: Add context support
        self.context = {}
    
    def update_context(self, context: Dict):
        """Update pipeline context (NEW method)."""
        self.context = context
    
    async def process_audio(self, audio_data: bytes) -> bytes:
        """Process audio through traditional pipeline."""
        # Existing STT processing
        transcription = await self.stt.transcribe(audio_data)
        
        # NEW: Include context in LLM request
        llm_request = {
            "message": transcription,
            "context": self.context
        }
        
        # Existing LLM processing with context
        response_text = await self.llm.generate_response(llm_request)
        
        # Existing TTS processing
        audio_response = await self.tts.generate_speech(response_text)
        
        return audio_response
```

---

## Testing and Validation

### 1. Pre-Migration Tests

```bash
# Run comprehensive test suite
python -m pytest tests/ -v --cov=assistant

# Test traditional pipeline functionality
python -c "
from assistant.pipeline import VoicePipeline
from assistant.config_manager import ConfigManager

config = ConfigManager()
pipeline = VoicePipeline(config)
print('✓ Traditional pipeline loads successfully')
"
```

### 2. Migration Tests

```bash
# Test hybrid system initialization
python -c "
from assistant.hybrid_voice_system import HybridVoiceSystem
from assistant.config_manager import ConfigManager

config = ConfigManager()
hybrid = HybridVoiceSystem(config)
print('✓ Hybrid system initializes successfully')
"

# Test Realtime API connection
python assistant/realtime_voice.py --test-connection

# Test smart context management
python assistant/smart_context_manager.py --test

# Test screen context
python assistant/screen_context_provider.py --test-ocr

# Test memory context
python assistant/memory_context_provider.py --test-memory
```

### 3. Integration Tests

```bash
# Run hybrid mode integration tests
python -m pytest tests/test_hybrid_integration.py -v

# Test mode switching functionality
python -m pytest tests/test_mode_switch_manager.py -v

# Test fallback scenarios
python -c "
import asyncio
from assistant.hybrid_voice_system import HybridVoiceSystem

async def test_fallback():
    hybrid = HybridVoiceSystem(ConfigManager())
    await hybrid.start()
    
    # Simulate Realtime API failure
    hybrid.realtime_service._simulate_failure = True
    
    # Test fallback
    test_audio = b'test audio data'
    result = await hybrid.process_voice_input(test_audio)
    
    print('✓ Fallback mechanism working')
    await hybrid.stop()

asyncio.run(test_fallback())
"
```

### 4. Performance Validation

```bash
# Compare performance before and after migration
python -c "
from assistant.performance_testing import PerformanceTester

tester = PerformanceTester()

# Test hybrid mode performance
hybrid_metrics = tester.test_hybrid_mode()
print(f'Hybrid mode latency: {hybrid_metrics[\"avg_latency\"]}ms')

# Compare with baseline
baseline_metrics = tester.load_baseline()
improvement = baseline_metrics['avg_latency'] / hybrid_metrics['avg_latency']
print(f'Performance improvement: {improvement:.2f}x faster')
"
```

---

## Performance Optimization

### 1. Initial Tuning

```yaml
# config/sovereign.yaml - Optimize for your environment
realtime_api:
  timeout: 45.0          # Increase if network is slow
  max_retries: 5         # More retries for stability

smart_context:
  token_budget: 24000    # Reduce if experiencing latency
  cache:
    ttl_seconds: 600     # Longer cache for better performance

audio:
  chunk_size: 512        # Smaller chunks for lower latency
  sample_rate: 24000     # Keep high for quality

performance:
  targets:
    latency_ms: 250      # Aggressive target
  fallback:
    latency_threshold_ms: 400  # Switch earlier if needed
```

### 2. Resource Optimization

```bash
# Monitor resource usage during operation
python -c "
import psutil
import time

process = psutil.Process()
for i in range(30):  # Monitor for 30 seconds
    cpu = process.cpu_percent()
    memory = process.memory_info().rss / 1024 / 1024
    print(f'CPU: {cpu:5.1f}% Memory: {memory:6.1f}MB')
    time.sleep(1)
"
```

### 3. Network Optimization

```bash
# Test network performance to OpenAI
ping -c 10 api.openai.com

# Optimize DNS resolution
echo "1.1.1.1 api.openai.com" | sudo tee -a /etc/hosts  # Optional

# Check bandwidth
speedtest-cli  # Install with: pip install speedtest-cli
```

---

## Rollback Procedures

### Emergency Rollback

If issues occur during migration, quickly rollback:

```bash
# Stop current system
pkill -f "python.*sovereign"

# Restore from backup
BACKUP_DIR="backups/$(ls backups/ | tail -1)"
cp -r $BACKUP_DIR/config/ ./
cp -r $BACKUP_DIR/assistant/ ./
cp $BACKUP_DIR/.env ./

# Restore requirements
pip install -r $BACKUP_DIR/requirements_backup.txt

# Start traditional system
python assistant/main.py --mode=traditional
```

### Gradual Rollback

```yaml
# config/sovereign.yaml - Switch to traditional mode
app:
  mode: "traditional"  # Change from "hybrid"

# Disable new features temporarily
screen_context:
  enabled: false

memory_context:
  enabled: false

monitoring:
  export:
    prometheus:
      enabled: false
```