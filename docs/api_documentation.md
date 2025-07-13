# Sovereign 4.0 - API Documentation

Complete API reference for all Realtime API components, classes, and methods.

## Table of Contents

1. [Core Components](#core-components)
2. [Realtime API Service](#realtime-api-service)
3. [Smart Context Management](#smart-context-management)
4. [Screen Context Provider](#screen-context-provider)
5. [Memory Context Provider](#memory-context-provider)
6. [Audio Stream Management](#audio-stream-management)
7. [Connection Stability](#connection-stability)
8. [Mode Switching](#mode-switching)
9. [Performance Monitoring](#performance-monitoring)
10. [Configuration Management](#configuration-management)

---

## Core Components

### HybridVoiceSystem

The main orchestrator for Realtime API and fallback functionality.

```python
from assistant.hybrid_voice_system import HybridVoiceSystem

# Initialize the hybrid system
voice_system = HybridVoiceSystem(config_manager)

# Start voice processing
await voice_system.start()

# Process voice input
response = await voice_system.process_voice_input(audio_data)

# Stop the system
await voice_system.stop()
```

#### Methods

- **`__init__(config_manager: ConfigManager)`**
  - Initializes hybrid voice system with configuration
  - Sets up Realtime API and traditional pipeline components

- **`async start()`**
  - Starts the voice processing system
  - Establishes WebSocket connections
  - Initializes audio streams

- **`async process_voice_input(audio_data: bytes) -> str`**
  - Processes voice input through appropriate pipeline
  - Returns transcribed and processed response
  - Handles automatic fallback if needed

- **`async stop()`**
  - Cleanly stops all voice processing
  - Closes connections and releases resources

---

## Realtime API Service

### RealtimeVoiceService

Core service for OpenAI Realtime API integration.

```python
from assistant.realtime_voice import RealtimeVoiceService

# Initialize service
realtime_service = RealtimeVoiceService(config_manager)

# Connect to Realtime API
await realtime_service.connect()

# Send audio data
await realtime_service.send_audio(audio_chunk)

# Receive response
response = await realtime_service.receive_response()
```

#### Configuration

```yaml
realtime_api:
  model: "gpt-4o-realtime-preview"
  voice: "alloy"  # alloy, echo, fable, onyx, nova, shimmer
  sample_rate: 24000
  channels: 1
  format: "pcm16"
  max_tokens: 4096
  temperature: 0.8
```

#### Methods

- **`__init__(config_manager: ConfigManager)`**
  - Initializes Realtime API service
  - Sets up WebSocket configuration

- **`async connect() -> bool`**
  - Establishes WebSocket connection to OpenAI
  - Returns True if successful, False otherwise

- **`async send_audio(audio_data: bytes)`**
  - Sends audio data to Realtime API
  - Handles chunking and formatting

- **`async receive_response() -> Dict[str, Any]`**
  - Receives and processes API response
  - Returns parsed response data

- **`async disconnect()`**
  - Closes WebSocket connection
  - Cleans up resources

#### Events

The service handles various WebSocket events:

- **`session.created`**: Session initialization
- **`input_audio_buffer.speech_started`**: Speech detection
- **`input_audio_buffer.speech_stopped`**: End of speech
- **`response.audio.delta`**: Audio response chunks
- **`response.done`**: Complete response received
- **`error`**: Error handling

---

## Smart Context Management

### SmartContextManager

Intelligent context prioritization and token budget management.

```python
from assistant.smart_context_manager import SmartContextManager

# Initialize context manager
context_manager = SmartContextManager(config_manager)

# Build context for request
context = await context_manager.build_context(
    query="What's on my screen?",
    include_memory=True,
    include_screen=True
)

# Update context cache
await context_manager.update_cache()
```

#### Configuration

```yaml
smart_context:
  token_budget: 32000
  priorities:
    system: 2000      # System instructions
    memory: 4000      # Recent memory
    screen: 2000      # Screen content
    conversation: 24000  # Conversation history
  
  cache:
    enabled: true
    ttl_seconds: 300
    max_size: 100
  
  relevance:
    enabled: true
    threshold: 0.6
    model: "sentence-transformers/all-MiniLM-L6-v2"
```

#### Methods

- **`__init__(config_manager: ConfigManager)`**
  - Initializes smart context manager
  - Sets up token counter and relevance scorer

- **`async build_context(query: str, **options) -> Dict[str, Any]`**
  - Builds optimized context for query
  - Applies priority-based token allocation
  - Returns structured context data

- **`async update_cache()`**
  - Updates context cache with fresh data
  - Manages cache expiration and cleanup

- **`get_token_usage() -> Dict[str, int]`**
  - Returns current token usage by category
  - Useful for monitoring and optimization

#### Context Structure

```python
{
    "system_instructions": "...",
    "memory_context": ["...", "..."],
    "screen_context": "...",
    "conversation_history": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ],
    "metadata": {
        "total_tokens": 28000,
        "token_usage": {
            "system": 2000,
            "memory": 3500,
            "screen": 1800,
            "conversation": 20700
        }
    }
}
```

---

## Screen Context Provider

### ScreenContextProvider

Real-time screen content extraction and analysis.

```python
from assistant.screen_context_provider import ScreenContextProvider

# Initialize screen provider
screen_provider = ScreenContextProvider(config_manager)

# Capture screen content
screen_data = await screen_provider.get_screen_context()

# Check for sensitive content
is_safe = screen_provider.is_content_safe(screen_data["text"])
```

#### Configuration

```yaml
screen_context:
  enabled: true
  capture_interval: 5.0  # seconds
  ocr_engine: "tesseract"
  privacy_filter: true
  
  privacy:
    filter_patterns:
      - "password"
      - "credit card"
      - "ssn"
      - "api key"
    
  ocr_settings:
    confidence_threshold: 60
    language: "eng"
    preserve_layout: true
```

#### Methods

- **`__init__(config_manager: ConfigManager)`**
  - Initializes screen context provider
  - Sets up OCR engine and privacy filters

- **`async get_screen_context() -> Dict[str, Any]`**
  - Captures current screen content
  - Returns extracted text and metadata

- **`is_content_safe(text: str) -> bool`**
  - Checks content for sensitive information
  - Returns True if safe to include in context

- **`async start_monitoring()`**
  - Starts background screen monitoring
  - Updates context cache periodically

- **`async stop_monitoring()`**
  - Stops background monitoring
  - Cleans up resources

#### Response Format

```python
{
    "text": "Extracted screen text...",
    "timestamp": "2024-01-15T10:30:00Z",
    "confidence": 85.2,
    "safe_content": true,
    "metadata": {
        "screen_resolution": "1920x1080",
        "window_title": "Code Editor",
        "application": "VS Code"
    }
}
```

---

## Memory Context Provider

### MemoryContextProvider

Conversation memory and context retrieval using ChromaDB.

```python
from assistant.memory_context_provider import MemoryContextProvider

# Initialize memory provider
memory_provider = MemoryContextProvider(config_manager)

# Store conversation
await memory_provider.store_conversation(
    user_message="Hello",
    assistant_response="Hi there!",
    metadata={"timestamp": "2024-01-15T10:30:00Z"}
)

# Retrieve relevant context
context = await memory_provider.get_relevant_context(
    query="What did we talk about earlier?"
)
```

#### Configuration

```yaml
memory_context:
  enabled: true
  vector_db: "chromadb"
  collection_name: "conversations"
  max_results: 10
  similarity_threshold: 0.7
  
  embedding:
    model: "text-embedding-3-small"
    dimensions: 1536
  
  storage:
    persist_directory: "./data/memory"
    cleanup_interval: 86400  # 24 hours
```

#### Methods

- **`__init__(config_manager: ConfigManager)`**
  - Initializes memory provider with ChromaDB
  - Sets up embedding model and storage

- **`async store_conversation(user_message: str, assistant_response: str, metadata: Dict)`**
  - Stores conversation in vector database
  - Creates embeddings for semantic search

- **`async get_relevant_context(query: str, max_results: int = 10) -> List[Dict]`**
  - Retrieves semantically similar conversations
  - Returns ranked results with similarity scores

- **`async clear_old_memories(days: int = 30)`**
  - Cleans up old conversation data
  - Maintains database performance

#### Memory Format

```python
{
    "id": "conv_123456",
    "user_message": "What's the weather like?",
    "assistant_response": "I don't have access to real-time weather...",
    "timestamp": "2024-01-15T10:30:00Z",
    "similarity_score": 0.85,
    "metadata": {
        "conversation_id": "sess_789",
        "context_type": "weather_query"
    }
}
```

---

## Audio Stream Management

### AudioStreamManager

Optimized WebSocket audio streaming for Realtime API.

```python
from assistant.audio_stream_manager import AudioStreamManager

# Initialize audio manager
audio_manager = AudioStreamManager(config_manager)

# Start audio streaming
await audio_manager.start_streaming()

# Send audio chunk
await audio_manager.send_audio_chunk(audio_data)

# Receive audio response
response_audio = await audio_manager.receive_audio()
```

#### Configuration

```yaml
audio_streaming:
  sample_rate: 24000
  channels: 1
  chunk_size: 1024
  format: "pcm16"
  
  buffering:
    input_buffer_size: 4096
    output_buffer_size: 4096
    max_latency_ms: 100
  
  quality:
    noise_reduction: true
    echo_cancellation: true
    auto_gain_control: true
```

#### Methods

- **`__init__(config_manager: ConfigManager)`**
  - Initializes audio stream manager
  - Sets up audio processing pipeline

- **`async start_streaming()`**
  - Starts audio input/output streams
  - Initializes WebSocket connection

- **`async send_audio_chunk(audio_data: bytes)`**
  - Sends audio chunk to Realtime API
  - Handles buffering and rate limiting

- **`async receive_audio() -> bytes`**
  - Receives audio response from API
  - Returns processed audio data

- **`get_stream_metrics() -> Dict[str, float]`**
  - Returns real-time streaming metrics
  - Latency, buffer levels, quality metrics

---

## Connection Stability

### ConnectionStabilityMonitor

Monitors and maintains WebSocket connection health.

```python
from assistant.connection_stability_monitor import ConnectionStabilityMonitor

# Initialize monitor
monitor = ConnectionStabilityMonitor(config_manager)

# Start monitoring
await monitor.start_monitoring()

# Check connection health
health = monitor.get_connection_health()

# Get stability metrics
metrics = monitor.get_stability_metrics()
```

#### Configuration

```yaml
connection_stability:
  ping_interval: 30.0  # seconds
  timeout: 10.0
  max_retries: 3
  backoff_factor: 2.0
  
  health_checks:
    latency_threshold: 500  # ms
    packet_loss_threshold: 0.05  # 5%
    jitter_threshold: 50  # ms
  
  reconnection:
    auto_reconnect: true
    reconnect_delay: 5.0
    max_reconnect_attempts: 10
```

#### Methods

- **`__init__(config_manager: ConfigManager)`**
  - Initializes connection monitor
  - Sets up health check parameters

- **`async start_monitoring()`**
  - Starts connection health monitoring
  - Begins periodic ping/pong checks

- **`get_connection_health() -> Dict[str, Any]`**
  - Returns current connection status
  - Health metrics and stability indicators

- **`async handle_disconnection()`**
  - Handles unexpected disconnections
  - Implements reconnection logic

#### Health Metrics

```python
{
    "status": "healthy",  # healthy, degraded, unhealthy
    "latency_ms": 150,
    "packet_loss": 0.02,
    "jitter_ms": 25,
    "uptime_seconds": 3600,
    "reconnections": 0,
    "last_ping": "2024-01-15T10:30:00Z"
}
```

---

## Mode Switching

### ModeSwitchManager

Intelligent switching between Realtime API and traditional pipeline.

```python
from assistant.mode_switch_manager import ModeSwitchManager

# Initialize mode switcher
mode_switcher = ModeSwitchManager(config_manager)

# Check if mode switch is needed
should_switch = await mode_switcher.should_switch_mode()

# Switch to fallback mode
await mode_switcher.switch_to_fallback()

# Switch back to realtime mode
await mode_switcher.switch_to_realtime()
```

#### Configuration

```yaml
mode_switching:
  enabled: true
  switch_threshold:
    latency_ms: 500
    error_rate: 0.1
    connection_failures: 3
  
  fallback:
    mode: "traditional"  # traditional, offline
    switch_delay: 2.0
    recovery_delay: 30.0
  
  monitoring:
    evaluation_window: 60  # seconds
    min_samples: 10
```

#### Methods

- **`__init__(config_manager: ConfigManager)`**
  - Initializes mode switch manager
  - Sets up performance thresholds

- **`async should_switch_mode() -> bool`**
  - Evaluates current performance metrics
  - Returns True if mode switch recommended

- **`async switch_to_fallback()`**
  - Switches to traditional pipeline
  - Maintains conversation continuity

- **`async switch_to_realtime()`**
  - Returns to Realtime API mode
  - Verifies connection stability

- **`get_current_mode() -> str`**
  - Returns current operating mode
  - "realtime", "traditional", or "offline"

---

## Performance Monitoring

### RealtimeMetricsCollector

Comprehensive performance metrics collection and analysis.

```python
from assistant.realtime_metrics_collector import RealtimeMetricsCollector

# Initialize metrics collector
metrics = RealtimeMetricsCollector(config_manager)

# Record performance metric
metrics.record_latency("realtime_response", 250)

# Get performance summary
summary = metrics.get_performance_summary()

# Export metrics for monitoring
prometheus_metrics = metrics.export_prometheus_metrics()
```

#### Configuration

```yaml
metrics_collection:
  enabled: true
  collection_interval: 1.0  # seconds
  retention_days: 30
  
  metrics:
    - latency
    - throughput
    - error_rate
    - connection_health
    - resource_usage
  
  export:
    prometheus: true
    influxdb: false
    file: true
```

#### Methods

- **`__init__(config_manager: ConfigManager)`**
  - Initializes metrics collector
  - Sets up metric storage and export

- **`record_latency(operation: str, latency_ms: float)`**
  - Records operation latency
  - Updates performance statistics

- **`record_error(error_type: str, details: Dict)`**
  - Records error occurrences
  - Tracks error patterns and trends

- **`get_performance_summary() -> Dict[str, Any]`**
  - Returns comprehensive performance summary
  - Includes trends and recommendations

#### Metrics Format

```python
{
    "timestamp": "2024-01-15T10:30:00Z",
    "metrics": {
        "realtime_latency": {
            "avg": 245.6,
            "p50": 230,
            "p95": 320,
            "p99": 450
        },
        "connection_health": {
            "uptime": 0.999,
            "reconnections": 0,
            "packet_loss": 0.001
        },
        "resource_usage": {
            "cpu_percent": 15.2,
            "memory_mb": 256,
            "gpu_utilization": 45.3
        }
    }
}
```

---

## Configuration Management

### ConfigManager

Enhanced configuration management with Realtime API support.

```python
from assistant.config_manager import ConfigManager

# Initialize configuration
config = ConfigManager("config/sovereign.yaml")

# Get Realtime API settings
realtime_config = config.get_realtime_config()

# Update configuration
config.update_config("realtime_api.voice", "nova")

# Validate configuration
is_valid = config.validate_config()
```

#### Configuration Structure

```yaml
# Realtime API Configuration
realtime_api:
  model: "gpt-4o-realtime-preview"
  voice: "alloy"
  sample_rate: 24000
  channels: 1
  instructions: "You are a helpful voice assistant..."
  
# Smart Context Configuration
smart_context:
  token_budget: 32000
  priorities:
    system: 2000
    memory: 4000
    screen: 2000
    conversation: 24000
  
# Performance Configuration
performance:
  target_latency_ms: 300
  fallback_threshold_ms: 500
  monitoring_enabled: true
  
# Audio Configuration
audio:
  input_device: null  # Auto-detect
  output_device: null  # Auto-detect
  sample_rate: 24000
  channels: 1
  chunk_size: 1024
```

#### Methods

- **`__init__(config_path: str)`**
  - Loads configuration from YAML file
  - Validates configuration structure

- **`get_realtime_config() -> Dict[str, Any]`**
  - Returns Realtime API configuration
  - Includes all necessary parameters

- **`update_config(path: str, value: Any)`**
  - Updates configuration value
  - Saves changes to file

- **`validate_config() -> bool`**
  - Validates configuration completeness
  - Checks for required fields and types

---

## Error Handling

All API components include comprehensive error handling:

### Common Exceptions

- **`RealtimeAPIError`**: Realtime API connection or processing errors
- **`ContextBuildError`**: Smart context management errors
- **`ScreenCaptureError`**: Screen context extraction errors
- **`AudioStreamError`**: Audio streaming and processing errors
- **`ConfigurationError`**: Configuration validation errors

### Error Response Format

```python
{
    "error": {
        "type": "RealtimeAPIError",
        "message": "WebSocket connection failed",
        "code": "CONNECTION_FAILED",
        "details": {
            "url": "wss://api.openai.com/v1/realtime",
            "status_code": 401,
            "retry_after": 30
        },
        "timestamp": "2024-01-15T10:30:00Z"
    }
}
```

### Best Practices

1. **Always handle exceptions** when calling API methods
2. **Use try-catch blocks** for network operations
3. **Implement retry logic** for transient failures
4. **Log errors appropriately** for debugging
5. **Provide user feedback** for recoverable errors

---

## Examples

### Complete Voice Assistant Setup

```python
import asyncio
from assistant.config_manager import ConfigManager
from assistant.hybrid_voice_system import HybridVoiceSystem

async def main():
    # Initialize configuration
    config = ConfigManager("config/sovereign.yaml")
    
    # Create hybrid voice system
    voice_system = HybridVoiceSystem(config)
    
    try:
        # Start the system
        await voice_system.start()
        print("Voice assistant ready!")
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        # Clean shutdown
        await voice_system.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Custom Context Provider

```python
from assistant.smart_context_manager import SmartContextManager

class CustomContextProvider:
    def __init__(self, config_manager):
        self.config = config_manager
        self.context_manager = SmartContextManager(config_manager)
    
    async def get_enhanced_context(self, query: str):
        # Build base context
        context = await self.context_manager.build_context(query)
        
        # Add custom context
        if "code" in query.lower():
            context["code_context"] = await self.get_code_context()
        
        return context
    
    async def get_code_context(self):
        # Custom code context logic
        return {"files": [], "functions": []}
```

---

This documentation provides complete coverage of all Realtime API components and their usage. For more examples and tutorials, see the `/docs/examples` directory. 