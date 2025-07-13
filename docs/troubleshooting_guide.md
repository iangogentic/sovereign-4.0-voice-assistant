# Troubleshooting Guide

Comprehensive troubleshooting guide for Sovereign 4.0 Voice Assistant Realtime API issues.

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Realtime API Issues](#realtime-api-issues)
3. [Connection Problems](#connection-problems)
4. [Audio Issues](#audio-issues)
5. [Performance Problems](#performance-problems)
6. [Context Management Issues](#context-management-issues)
7. [Screen Context Problems](#screen-context-problems)
8. [Memory Issues](#memory-issues)
9. [Configuration Errors](#configuration-errors)
10. [System-Specific Issues](#system-specific-issues)

---

## Quick Diagnostics

### Health Check Commands

Run these commands to quickly diagnose system health:

```bash
# Check overall system health
python assistant/main.py --health-check

# Test Realtime API connection
python assistant/realtime_voice.py --test-connection

# Test audio system
python assistant/audio_stream_manager.py --test-audio

# Test screen context
python assistant/screen_context_provider.py --test-ocr

# Test memory context
python assistant/memory_context_provider.py --test-memory

# Run comprehensive diagnostics
python -m assistant.diagnostics --full-check
```

### System Status Dashboard

Access the live dashboard for real-time diagnostics:
```
http://localhost:8080/dashboard
```

### Log Analysis

Check logs for errors and warnings:
```bash
# View recent logs
tail -f logs/sovereign.log

# Search for specific errors
grep -i "error" logs/sovereign.log | tail -20

# Check Realtime API specific logs
grep "realtime" logs/sovereign.log | tail -10

# Analyze performance metrics
grep "latency\|performance" logs/sovereign.log | tail -15
```

---

## Realtime API Issues

### Connection Failed

**Symptoms:**
```
[ERROR] Realtime WebSocket connection failed
[ERROR] Failed to connect to wss://api.openai.com/v1/realtime
```

**Diagnosis:**
```bash
# Test API key validity
python -c "
import openai
openai.api_key = 'your-api-key'
try:
    openai.models.list()
    print('API key valid')
except Exception as e:
    print(f'API key invalid: {e}')
"

# Test network connectivity
curl -I https://api.openai.com/v1/models

# Check Realtime API access
python assistant/realtime_voice.py --test-connection --verbose
```

**Solutions:**

1. **Verify API Key:**
   ```bash
   # Check .env file
   cat .env | grep OPENAI_API_KEY
   
   # Ensure key has Realtime API access
   # Contact OpenAI support if needed
   ```

2. **Check Network Configuration:**
   ```bash
   # Test proxy settings
   echo $HTTP_PROXY
   echo $HTTPS_PROXY
   
   # Test firewall rules
   telnet api.openai.com 443
   ```

3. **Update Configuration:**
   ```yaml
   # config/sovereign.yaml
   realtime_api:
     timeout: 60.0        # Increase timeout
     max_retries: 5       # More retry attempts
     retry_delay: 5.0     # Longer retry delay
   ```

### High Latency

**Symptoms:**
```
[WARN] Realtime latency >500ms, considering fallback
[INFO] Average response time: 800ms (target: <300ms)
```

**Diagnosis:**
```bash
# Monitor real-time latency
python -c "
from assistant.realtime_metrics_collector import RealtimeMetricsCollector
metrics = RealtimeMetricsCollector()
print(metrics.get_latency_stats())
"

# Check network latency to OpenAI
ping -c 10 api.openai.com

# Test system resources
python -c "
import psutil
print(f'CPU: {psutil.cpu_percent()}%')
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'Disk I/O: {psutil.disk_io_counters()}')
"
```

**Solutions:**

1. **Optimize Context Size:**
   ```yaml
   # config/sovereign.yaml
   smart_context:
     token_budget: 16000  # Reduce from 32000
     priorities:
       conversation: 12000  # Reduce conversation history
   ```

2. **Improve Network:**
   ```bash
   # Use wired connection instead of WiFi
   # Close bandwidth-heavy applications
   # Consider CDN or regional API endpoints
   ```

3. **System Optimization:**
   ```yaml
   # config/sovereign.yaml
   performance:
     optimization:
       parallel_processing: true
       async_operations: true
       low_latency_mode: true
   ```

### Audio Quality Issues

**Symptoms:**
```
[ERROR] Audio chunk dropped due to buffer overflow
[WARN] Audio quality degraded: noise detected
```

**Solutions:**

1. **Optimize Audio Settings:**
   ```yaml
   # config/sovereign.yaml
   audio:
     sample_rate: 24000     # Ensure high quality
     chunk_size: 512        # Smaller chunks for lower latency
     processing:
       noise_reduction: true
       echo_cancellation: true
   ```

2. **Check Audio Hardware:**
   ```bash
   # List audio devices
   python -c "
   import pyaudio
   p = pyaudio.PyAudio()
   for i in range(p.get_device_count()):
       print(p.get_device_info_by_index(i))
   "
   
   # Test microphone quality
   python assistant/audio.py --test-recording
   ```

---

## Connection Problems

### WebSocket Disconnections

**Symptoms:**
```
[ERROR] WebSocket connection lost
[INFO] Attempting reconnection (attempt 3/10)
```

**Diagnosis:**
```bash
# Monitor connection stability
python assistant/connection_stability_monitor.py --monitor

# Check connection metrics
python -c "
from assistant.connection_stability_monitor import ConnectionStabilityMonitor
monitor = ConnectionStabilityMonitor()
print(monitor.get_stability_metrics())
"
```

**Solutions:**

1. **Improve Connection Stability:**
   ```yaml
   # config/sovereign.yaml
   connection_stability:
     ping_interval: 15.0      # More frequent pings
     timeout: 20.0            # Longer timeout
     max_ping_failures: 5     # More tolerant of failures
     
     reconnection:
       max_attempts: 20       # More reconnection attempts
       initial_delay: 2.0     # Faster initial reconnection
   ```

2. **Network Diagnostics:**
   ```bash
   # Check for packet loss
   ping -c 100 api.openai.com | tail -1
   
   # Monitor network interface
   netstat -i
   
   # Check for network congestion
   iftop  # or nethogs
   ```

### Circuit Breaker Activation

**Symptoms:**
```
[WARN] Circuit breaker OPEN - too many failures
[INFO] Blocking requests for 60 seconds
```

**Solutions:**

1. **Adjust Circuit Breaker Settings:**
   ```yaml
   # config/sovereign.yaml
   connection_stability:
     circuit_breaker:
       failure_threshold: 10    # More tolerant
       timeout: 30.0           # Shorter recovery time
   ```

2. **Address Root Cause:**
   ```bash
   # Check for API rate limiting
   curl -I -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
   
   # Monitor error patterns
   grep "circuit_breaker" logs/sovereign.log | tail -20
   ```

---

## Audio Issues

### No Audio Input Detected

**Symptoms:**
```
[ERROR] No audio input detected from microphone
[WARN] Voice activity detection timeout
```

**Diagnosis:**
```bash
# Test audio input
python -c "
import pyaudio
import numpy as np

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True)
data = stream.read(1024)
print(f'Audio level: {np.abs(np.frombuffer(data, dtype=np.int16)).max()}')
"

# Check system audio settings
# macOS: System Preferences > Sound > Input
# Linux: alsamixer or pavucontrol
# Windows: Sound Settings > Input
```

**Solutions:**

1. **Check Microphone Permissions:**
   ```bash
   # macOS: Grant microphone permissions in System Preferences
   # Linux: Check PulseAudio/ALSA configuration
   # Windows: Check Privacy settings > Microphone
   ```

2. **Adjust Audio Settings:**
   ```yaml
   # config/sovereign.yaml
   audio:
     vad:
       sensitivity: 0.3        # Lower sensitivity
       energy_threshold: 100   # Lower threshold
       min_speech_duration: 0.3
   ```

3. **Test Different Input Device:**
   ```python
   # List available devices
   import pyaudio
   p = pyaudio.PyAudio()
   for i in range(p.get_device_count()):
       info = p.get_device_info_by_index(i)
       if info['maxInputChannels'] > 0:
           print(f"Device {i}: {info['name']}")
   ```

### Audio Output Problems

**Symptoms:**
```
[ERROR] Audio playback failed
[WARN] Speaker output unavailable
```

**Solutions:**

1. **Check Output Device:**
   ```yaml
   # config/sovereign.yaml
   audio:
     output_device: 0  # Specify device index
     # or
     output_device: "Built-in Output"  # Device name
   ```

2. **Test Audio Playback:**
   ```bash
   # Test system audio
   speaker-test -t wav -c 2  # Linux
   afplay /System/Library/Sounds/Ping.aiff  # macOS
   ```

### Audio Latency Issues

**Symptoms:**
```
[WARN] Audio latency >100ms detected
[INFO] Audio buffer underrun
```

**Solutions:**

1. **Optimize Buffer Settings:**
   ```yaml
   # config/sovereign.yaml
   audio:
     chunk_size: 512           # Smaller chunks
     buffering:
       input_buffer_size: 2048
       output_buffer_size: 2048
       max_latency_ms: 50      # Strict latency limit
   ```

2. **Use ASIO Drivers (Windows):**
   ```bash
   # Install ASIO4ALL or manufacturer ASIO drivers
   # Configure in audio settings
   ```

---

## Performance Problems

### High CPU Usage

**Symptoms:**
```
[WARN] CPU usage >90% for extended period
[ERROR] Performance degraded due to resource constraints
```

**Diagnosis:**
```bash
# Monitor CPU usage by process
top -p $(pgrep -f "python.*sovereign")

# Check Python profiling
python -m cProfile -o profile.out assistant/main.py
python -c "
import pstats
p = pstats.Stats('profile.out')
p.sort_stats('cumulative').print_stats(20)
"
```

**Solutions:**

1. **Optimize Processing:**
   ```yaml
   # config/sovereign.yaml
   performance:
     resources:
       max_cpu_percent: 70     # Limit CPU usage
       thread_pool_size: 2     # Reduce worker threads
   
   smart_context:
     background_refresh:
       enabled: false          # Disable background processing
   ```

2. **Reduce Audio Processing:**
   ```yaml
   # config/sovereign.yaml
   audio:
     processing:
       noise_reduction: false  # Disable CPU-intensive features
       echo_cancellation: false
   ```

### Memory Leaks

**Symptoms:**
```
[WARN] Memory usage increasing over time
[ERROR] Out of memory error
```

**Diagnosis:**
```bash
# Monitor memory usage
python -c "
import psutil
import time
process = psutil.Process()
for i in range(10):
    print(f'Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB')
    time.sleep(5)
"

# Use memory profiler
pip install memory-profiler
python -m memory_profiler assistant/main.py
```

**Solutions:**

1. **Configure Memory Management:**
   ```yaml
   # config/sovereign.yaml
   performance:
     resources:
       max_memory_mb: 512      # Limit memory usage
       garbage_collection: "aggressive"
   
   smart_context:
     cache:
       max_size: 50            # Reduce cache size
       ttl_seconds: 180        # Shorter TTL
   ```

2. **Clear Caches Periodically:**
   ```python
   # Add to application code
   import gc
   import threading
   
   def cleanup_memory():
       gc.collect()
       # Clear application caches
   
   # Run cleanup every 10 minutes
   threading.Timer(600, cleanup_memory).start()
   ```

---

## Context Management Issues

### Context Build Failures

**Symptoms:**
```
[ERROR] Failed to build smart context
[ERROR] Token budget exceeded
```

**Diagnosis:**
```bash
# Check context manager status
python -c "
from assistant.smart_context_manager import SmartContextManager
manager = SmartContextManager()
print(manager.get_token_usage())
print(manager.get_cache_stats())
"
```

**Solutions:**

1. **Adjust Token Budget:**
   ```yaml
   # config/sovereign.yaml
   smart_context:
     token_budget: 16000       # Reduce from 32000
     priorities:
       system: 1000
       memory: 2000
       screen: 1000
       conversation: 12000
   ```

2. **Enable Context Compression:**
   ```yaml
   # config/sovereign.yaml
   smart_context:
     compression:
       enabled: true
       ratio: 0.5              # More aggressive compression
       method: "extractive"
   ```

### Relevance Scoring Issues

**Symptoms:**
```
[ERROR] Relevance model failed to load
[WARN] Relevance scoring disabled
```

**Solutions:**

1. **Install Required Models:**
   ```bash
   # Install sentence transformers
   pip install sentence-transformers
   
   # Download model manually
   python -c "
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('all-MiniLM-L6-v2')
   print('Model downloaded successfully')
   "
   ```

2. **Use Fallback Scoring:**
   ```yaml
   # config/sovereign.yaml
   smart_context:
     relevance:
       enabled: false          # Disable if causing issues
   ```

---

## Screen Context Problems

### OCR Not Working

**Symptoms:**
```
[ERROR] Tesseract OCR not found
[ERROR] Screen capture failed
```

**Diagnosis:**
```bash
# Test Tesseract installation
tesseract --version

# Test OCR functionality
python -c "
from assistant.screen_context_provider import ScreenContextProvider
provider = ScreenContextProvider()
result = provider.test_ocr()
print(result)
"
```

**Solutions:**

1. **Install Tesseract:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr
   
   # macOS
   brew install tesseract
   
   # Windows
   # Download from: https://github.com/UB-Mannheim/tesseract/wiki
   ```

2. **Configure Tesseract Path:**
   ```bash
   # Add to .env file
   TESSERACT_CMD=/usr/local/bin/tesseract
   
   # Or in config
   export TESSERACT_CMD=/path/to/tesseract
   ```

### Screen Permissions

**Symptoms:**
```
[ERROR] Screen capture permission denied
[ERROR] Unable to access screen content
```

**Solutions:**

1. **Grant Screen Recording Permissions:**
   ```bash
   # macOS: System Preferences > Security & Privacy > Screen Recording
   # Add Python or Terminal app
   
   # Linux: May need to run with appropriate permissions
   # Check X11 or Wayland permissions
   
   # Windows: Usually not required, check UAC settings
   ```

2. **Test Screen Capture:**
   ```python
   import PIL.ImageGrab as ImageGrab
   screenshot = ImageGrab.grab()
   screenshot.save("test_screenshot.png")
   print("Screen capture successful")
   ```

---

## Memory Issues

### ChromaDB Connection Failed

**Symptoms:**
```
[ERROR] Failed to connect to ChromaDB
[ERROR] Memory context unavailable
```

**Solutions:**

1. **Check ChromaDB Installation:**
   ```bash
   pip install chromadb
   
   # Test ChromaDB
   python -c "
   import chromadb
   client = chromadb.Client()
   print('ChromaDB working')
   "
   ```

2. **Reset Database:**
   ```bash
   # Remove corrupted database
   rm -rf ./data/memory
   
   # Restart application to recreate
   python assistant/main.py
   ```

### Embedding Model Issues

**Symptoms:**
```
[ERROR] Failed to load embedding model
[WARN] Using fallback embedding method
```

**Solutions:**

1. **Install Required Packages:**
   ```bash
   pip install sentence-transformers
   pip install torch torchvision  # For PyTorch backend
   ```

2. **Download Model Manually:**
   ```python
   from sentence_transformers import SentenceTransformer
   
   # Download and cache model
   model = SentenceTransformer('text-embedding-3-small')
   print("Model downloaded and cached")
   ```

---

## Configuration Errors

### Invalid YAML Syntax

**Symptoms:**
```
[ERROR] Failed to parse configuration file
[ERROR] YAML syntax error at line 45
```

**Solutions:**

1. **Validate YAML Syntax:**
   ```bash
   # Use YAML validator
   python -c "
   import yaml
   with open('config/sovereign.yaml') as f:
       try:
           yaml.safe_load(f)
           print('YAML syntax is valid')
       except yaml.YAMLError as e:
           print(f'YAML error: {e}')
   "
   ```

2. **Common YAML Issues:**
   ```yaml
   # Incorrect indentation
   realtime_api:
   model: "gpt-4o"  # Should be indented
   
   # Correct indentation
   realtime_api:
     model: "gpt-4o"
   
   # Missing quotes for special characters
   voice: nova:special  # Should be "nova:special"
   ```

### Missing Environment Variables

**Symptoms:**
```
[ERROR] OPENAI_API_KEY environment variable not set
[ERROR] Configuration validation failed
```

**Solutions:**

1. **Check Environment Variables:**
   ```bash
   # List all environment variables
   env | grep -i openai
   
   # Check .env file
   cat .env
   
   # Source .env file manually
   export $(cat .env | xargs)
   ```

2. **Validate Configuration:**
   ```bash
   python -c "
   from assistant.config_manager import ConfigManager
   config = ConfigManager()
   if config.validate_config():
       print('Configuration is valid')
   else:
       print('Configuration has errors')
   "
   ```

---

## System-Specific Issues

### macOS Issues

**Permission Problems:**
```bash
# Grant microphone permissions
# System Preferences > Security & Privacy > Microphone

# Grant screen recording permissions  
# System Preferences > Security & Privacy > Screen Recording

# Check code signing for distribution
codesign -dv --verbose=4 /path/to/python
```

**Audio Device Issues:**
```bash
# Reset Core Audio
sudo killall coreaudiod

# Check audio devices
system_profiler SPAudioDataType
```

### Linux Issues

**Audio System Problems:**
```bash
# Check ALSA
aplay -l
arecord -l

# Check PulseAudio
pactl info
pactl list sources
pactl list sinks

# Restart audio system
pulseaudio -k
pulseaudio --start
```

**Permission Issues:**
```bash
# Add user to audio group
sudo usermod -a -G audio $USER

# Check device permissions
ls -la /dev/snd/
```

### Windows Issues

**Audio Driver Problems:**
```bash
# Install/update audio drivers
# Use Device Manager to check for issues

# Install Microsoft Visual C++ Redistributables
# Download from Microsoft website
```

**Firewall/Antivirus:**
```bash
# Add Python to firewall exceptions
# Configure antivirus to exclude project directory
# Check Windows Defender settings
```

---

## Getting Help

### Collect Diagnostic Information

Before seeking help, collect this information:

```bash
# System information
python -c "
import platform
import sys
print(f'OS: {platform.system()} {platform.release()}')
print(f'Python: {sys.version}')
print(f'Architecture: {platform.machine()}')
"

# Package versions
pip freeze | grep -E "(openai|chromadb|sentence|torch|numpy|scipy)"

# Configuration summary
python -c "
from assistant.config_manager import ConfigManager
config = ConfigManager()
print(config.get_system_summary())
"

# Recent logs
tail -100 logs/sovereign.log > diagnostic_logs.txt

# Performance metrics
python -c "
from assistant.realtime_metrics_collector import RealtimeMetricsCollector
metrics = RealtimeMetricsCollector()
print(metrics.export_diagnostic_report())
"
```

### Support Channels

1. **GitHub Issues**: Report bugs and feature requests
2. **Documentation**: Check `/docs` directory for detailed guides
3. **Community Forum**: Discuss issues with other users
4. **Stack Overflow**: Tag questions with `sovereign-voice-assistant`

### Enterprise Support

For production deployments:
- 24/7 technical support
- Custom troubleshooting assistance
- Performance optimization consulting
- Integration support

---

This troubleshooting guide covers the most common issues encountered with Sovereign 4.0. For issues not covered here, please collect diagnostic information and reach out through the appropriate support channels. 