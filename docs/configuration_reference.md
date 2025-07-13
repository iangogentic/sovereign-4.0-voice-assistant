# Configuration Reference Guide

Complete configuration reference for Sovereign 4.0 Voice Assistant with Realtime API support.

## Table of Contents

1. [Overview](#overview)
2. [Environment Variables](#environment-variables)
3. [YAML Configuration](#yaml-configuration)
4. [Realtime API Settings](#realtime-api-settings)
5. [Smart Context Configuration](#smart-context-configuration)
6. [Audio Configuration](#audio-configuration)
7. [Performance Tuning](#performance-tuning)
8. [Monitoring & Logging](#monitoring--logging)
9. [Security Settings](#security-settings)
10. [Deployment Configuration](#deployment-configuration)

---

## Overview

Sovereign 4.0 uses a hierarchical configuration system:

1. **Environment Variables** (`.env`) - API keys and sensitive data
2. **YAML Configuration** (`config/sovereign.yaml`) - Main settings
3. **Runtime Overrides** - Command-line arguments and programmatic changes

### Configuration Priority

**Description:** The configuration system follows a strict priority hierarchy to ensure predictable behavior:

1. Command-line arguments (highest priority)
2. Environment variables
3. YAML configuration files  
4. Default values (lowest priority)

**Important:** API keys should always be stored in environment variables, never in YAML files for security reasons.

---

## Environment Variables

**Description:** Environment variables store sensitive configuration data like API keys and override system defaults.

### Required API Keys

**Important:** These API keys are required for Realtime API functionality. Store them securely in your `.env` file.

Create a `.env` file in your project root:

```bash
# OpenAI API (Required for Realtime API)
OPENAI_API_KEY=sk-proj-your-realtime-api-key-here

# Optional AI Providers
ANTHROPIC_API_KEY=your-anthropic-key-here
PERPLEXITY_API_KEY=your-perplexity-key-here
GOOGLE_API_KEY=your-google-api-key-here
ELEVENLABS_API_KEY=your-elevenlabs-key-here

# Optional for enhanced features
OPENROUTER_API_KEY=your-openrouter-key-here
XAI_API_KEY=your-xai-key-here
```

### System Configuration

```bash
# Application Mode
SOVEREIGN_MODE=hybrid              # hybrid|realtime|traditional
SOVEREIGN_ENV=production           # development|production|testing

# Logging
SOVEREIGN_LOG_LEVEL=INFO           # DEBUG|INFO|WARN|ERROR
SOVEREIGN_LOG_FILE=logs/sovereign.log

# Performance
SOVEREIGN_PERFORMANCE_MODE=balanced # fast|balanced|quality
SOVEREIGN_MAX_WORKERS=4

# Security
SOVEREIGN_API_RATE_LIMIT=60        # requests per minute
SOVEREIGN_ENCRYPTION_ENABLED=true
```

### Database Configuration

```bash
# ChromaDB (Memory Context)
CHROMA_DB_PATH=./data/memory
CHROMA_COLLECTION_NAME=conversations

# InfluxDB (Optional - Metrics)
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your-influxdb-token
INFLUXDB_ORG=your-org
INFLUXDB_BUCKET=sovereign-metrics
```

### Audio System

```bash
# Audio Devices
AUDIO_INPUT_DEVICE=default
AUDIO_OUTPUT_DEVICE=default
AUDIO_SAMPLE_RATE=24000

# Tesseract OCR
TESSERACT_CMD=/usr/local/bin/tesseract
TESSERACT_DATA_PATH=/usr/local/share/tessdata
```

---

## YAML Configuration

### Main Configuration File (`config/sovereign.yaml`)

```yaml
# Application Information
app:
  name: "Sovereign Voice Assistant"
  version: "4.0.0"
  description: "Realtime API Voice Assistant"
  
# Realtime API Configuration
realtime_api:
  # Model Settings
  model: "gpt-4o-realtime-preview"
  voice: "alloy"                    # alloy|echo|fable|onyx|nova|shimmer
  language: "en"
  
  # Audio Settings
  sample_rate: 24000                # 24kHz for high quality
  channels: 1                       # Mono audio
  format: "pcm16"                   # 16-bit PCM
  
  # Connection Settings
  url: "wss://api.openai.com/v1/realtime"
  timeout: 30.0                     # Connection timeout in seconds
  max_retries: 3
  retry_delay: 2.0
  
  # Response Settings
  max_tokens: 4096
  temperature: 0.8
  top_p: 0.9
  
  # Instructions
  instructions: |
    You are Sovereign, a helpful voice assistant with access to screen content 
    and conversation memory. Respond naturally and conversationally. When users 
    ask about their screen, use the provided screen context. Remember previous 
    conversations to provide better assistance.

# Smart Context Management
smart_context:
  # Token Budget (OpenAI limit: 32k)
  token_budget: 32000
  
  # Priority-based allocation
  priorities:
    system: 2000                    # System instructions
    memory: 4000                    # Recent conversation memory
    screen: 2000                    # Screen content
    conversation: 24000             # Current conversation
  
  # Relevance Scoring
  relevance:
    enabled: true
    threshold: 0.6                  # Minimum relevance score
    model: "sentence-transformers/all-MiniLM-L6-v2"
    
  # Context Compression
  compression:
    enabled: true
    ratio: 0.7                      # Compress to 70% of original
    method: "extractive"            # extractive|abstractive
    
  # Caching
  cache:
    enabled: true
    ttl_seconds: 300                # 5 minutes
    max_size: 100                   # Max cached contexts
    
  # Background Updates
  background_refresh:
    enabled: true
    interval_seconds: 60            # Update every minute
    batch_size: 10

# Screen Context Provider
screen_context:
  enabled: true
  
  # Capture Settings
  capture_interval: 5.0             # Seconds between captures
  auto_capture: true                # Automatic background capture
  
  # OCR Configuration
  ocr:
    engine: "tesseract"
    language: "eng"
    confidence_threshold: 60        # Minimum OCR confidence
    preserve_layout: true
    
  # Privacy Protection
  privacy:
    enabled: true
    filter_sensitive: true
    
    # Sensitive data patterns
    patterns:
      - "(?i)password"
      - "(?i)credit.?card"
      - "(?i)ssn|social.?security"
      - "(?i)api.?key"
      - "(?i)token"
      - "\\b\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}\\b"  # Credit card
      - "\\b\\d{3}-\\d{2}-\\d{4}\\b"                            # SSN
    
    # Replacement text
    replacement: "[REDACTED]"
    
  # Content Filtering
  content_filter:
    min_text_length: 10             # Minimum text length to include
    max_text_length: 2000           # Maximum text to capture
    exclude_applications:
      - "Keychain Access"
      - "1Password"
      - "LastPass"

# Memory Context Provider
memory_context:
  enabled: true
  
  # Vector Database
  vector_db:
    type: "chromadb"
    persist_directory: "./data/memory"
    collection_name: "conversations"
    
  # Embedding Configuration
  embeddings:
    model: "text-embedding-3-small"
    dimensions: 1536
    batch_size: 100
    
  # Retrieval Settings
  retrieval:
    max_results: 10                 # Max memories to retrieve
    similarity_threshold: 0.7       # Minimum similarity score
    include_metadata: true
    
  # Memory Management
  management:
    max_memories: 10000             # Maximum stored memories
    cleanup_interval: 86400         # 24 hours in seconds
    retention_days: 90              # Keep memories for 90 days

# Audio Configuration
audio:
  # Device Settings
  input_device: null                # null = auto-detect
  output_device: null               # null = auto-detect
  
  # Audio Quality
  sample_rate: 24000                # 24kHz for Realtime API
  channels: 1                       # Mono
  chunk_size: 1024                  # Samples per chunk
  format: "int16"                   # 16-bit integers
  
  # Processing
  processing:
    noise_reduction: true
    echo_cancellation: true
    auto_gain_control: true
    volume_normalization: true
    
  # Voice Activity Detection
  vad:
    enabled: true
    sensitivity: 0.6                # 0.0 (least) to 1.0 (most sensitive)
    min_speech_duration: 0.5        # Seconds
    min_silence_duration: 0.8       # Seconds
    
  # Buffering
  buffering:
    input_buffer_size: 4096         # Input buffer size
    output_buffer_size: 4096        # Output buffer size
    max_latency_ms: 100             # Maximum acceptable latency

# Performance Configuration
performance:
  # Target Metrics
  targets:
    latency_ms: 300                 # Target response latency
    uptime_percent: 99.9            # Target uptime
    accuracy_percent: 95.0          # Target accuracy
    
  # Fallback Thresholds
  fallback:
    latency_threshold_ms: 500       # Switch to fallback above this
    error_rate_threshold: 0.1       # 10% error rate threshold
    connection_failures: 3          # Max consecutive failures
    
  # Resource Management
  resources:
    max_cpu_percent: 80             # Maximum CPU usage
    max_memory_mb: 1024             # Maximum memory usage
    max_gpu_percent: 70             # Maximum GPU usage
    
  # Optimization
  optimization:
    parallel_processing: true
    async_operations: true
    connection_pooling: true
    request_batching: true

# Mode Switching
mode_switching:
  enabled: true
  
  # Evaluation Window
  evaluation:
    window_seconds: 60              # Performance evaluation window
    min_samples: 10                 # Minimum samples for decision
    
  # Switch Conditions
  conditions:
    latency_degradation: 0.5        # 50% latency increase
    error_rate_increase: 0.1        # 10% error rate increase
    connection_instability: 3       # Failed connection attempts
    
  # Fallback Configuration
  fallback:
    mode: "traditional"             # traditional|offline
    switch_delay: 2.0               # Delay before switching
    recovery_delay: 30.0            # Delay before trying to recover
    
  # Recovery Conditions
  recovery:
    stable_duration: 60             # Seconds of stable performance
    success_rate: 0.95              # 95% success rate required
    latency_improvement: 0.3        # 30% latency improvement

# Connection Stability
connection_stability:
  # Health Monitoring
  monitoring:
    ping_interval: 30.0             # Ping every 30 seconds
    timeout: 10.0                   # Ping timeout
    max_ping_failures: 3            # Max consecutive ping failures
    
  # Connection Health Thresholds
  health_thresholds:
    latency_ms: 500                 # Unhealthy above 500ms
    packet_loss_percent: 5.0        # Unhealthy above 5% loss
    jitter_ms: 50                   # Unhealthy above 50ms jitter
    
  # Reconnection Strategy
  reconnection:
    enabled: true
    max_attempts: 10                # Maximum reconnection attempts
    initial_delay: 1.0              # Initial delay between attempts
    max_delay: 60.0                 # Maximum delay between attempts
    backoff_factor: 2.0             # Exponential backoff multiplier
    
  # Circuit Breaker
  circuit_breaker:
    enabled: true
    failure_threshold: 5            # Failures before opening circuit
    success_threshold: 3            # Successes before closing circuit
    timeout: 60.0                   # Circuit breaker timeout

# Monitoring & Metrics
monitoring:
  enabled: true
  
  # Collection Settings
  collection:
    interval_seconds: 1.0           # Metrics collection interval
    retention_days: 30              # How long to keep metrics
    
  # Metrics to Collect
  metrics:
    - "latency"
    - "throughput" 
    - "error_rate"
    - "connection_health"
    - "resource_usage"
    - "context_performance"
    
  # Export Configuration
  export:
    prometheus:
      enabled: true
      port: 9090
      path: "/metrics"
      
    influxdb:
      enabled: false
      url: "http://localhost:8086"
      database: "sovereign_metrics"
      
    file:
      enabled: true
      path: "./logs/metrics.json"
      rotation: "daily"

# Dashboard Configuration
dashboard:
  enabled: true
  
  # Server Settings
  server:
    host: "localhost"
    port: 8080
    debug: false
    
  # Features
  features:
    realtime_metrics: true
    performance_graphs: true
    connection_status: true
    error_tracking: true
    
  # Update Intervals
  updates:
    metrics_refresh_ms: 1000        # 1 second
    graph_refresh_ms: 5000          # 5 seconds
    status_refresh_ms: 2000         # 2 seconds

# Logging Configuration  
logging:
  # Log Levels
  level: "INFO"                     # DEBUG|INFO|WARN|ERROR
  
  # Log Formats
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  date_format: "%Y-%m-%d %H:%M:%S"
  
  # File Logging
  file:
    enabled: true
    path: "./logs/sovereign.log"
    max_size_mb: 100                # Maximum log file size
    backup_count: 5                 # Number of backup files
    rotation: "size"                # size|time
    
  # Console Logging
  console:
    enabled: true
    colored: true                   # Colored console output
    
  # Structured Logging
  structured:
    enabled: true
    format: "json"                  # json|logfmt
    
  # Component-specific Levels
  components:
    "assistant.realtime_voice": "DEBUG"
    "assistant.smart_context": "INFO"
    "assistant.screen_context": "WARN"

# Security Configuration
security:
  # API Security
  api:
    rate_limiting:
      enabled: true
      requests_per_minute: 60
      burst_size: 10
      
    authentication:
      enabled: false                # For future enterprise features
      method: "api_key"             # api_key|oauth|jwt
      
  # Data Protection
  data:
    encryption_at_rest: true
    encryption_in_transit: true
    
    # PII Detection
    pii_detection:
      enabled: true
      confidence_threshold: 0.8
      redaction_method: "mask"      # mask|remove|encrypt
      
  # Privacy Settings
  privacy:
    data_retention_days: 90
    anonymize_logs: true
    gdpr_compliance: true
    
    # Data Minimization
    collection:
      minimal_data: true
      explicit_consent: true
      purpose_limitation: true

# Development Settings
development:
  # Debug Features
  debug:
    enabled: false
    mock_apis: false
    simulation_mode: false
    
  # Testing
  testing:
    test_mode: false
    mock_responses: false
    deterministic_behavior: false
    
  # Hot Reload
  hot_reload:
    enabled: false
    watch_paths:
      - "./assistant"
      - "./config"
```

---

## Realtime API Settings

### Model Configuration

```yaml
realtime_api:
  # Available Models
  model: "gpt-4o-realtime-preview"   # Primary Realtime model
  
  # Voice Options
  voice: "alloy"                     # alloy, echo, fable, onyx, nova, shimmer
  
  # Quality Settings
  sample_rate: 24000                 # 24kHz recommended for quality
  format: "pcm16"                    # 16-bit PCM audio format
  
  # Response Configuration
  max_tokens: 4096                   # Maximum response length
  temperature: 0.8                   # Response creativity (0.0-2.0)
  top_p: 0.9                        # Response focus (0.0-1.0)
```

### Voice Characteristics

| Voice | Characteristics | Best For |
|-------|----------------|----------|
| `alloy` | Balanced, neutral | General conversation |
| `echo` | Warm, engaging | Friendly interactions |
| `fable` | Clear, articulate | Educational content |
| `onyx` | Deep, authoritative | Professional contexts |
| `nova` | Bright, energetic | Creative tasks |
| `shimmer` | Calm, soothing | Meditation, relaxation |

### Connection Settings

```yaml
realtime_api:
  # WebSocket Configuration
  url: "wss://api.openai.com/v1/realtime"
  timeout: 30.0                      # Connection timeout
  max_retries: 3                     # Retry attempts
  retry_delay: 2.0                   # Delay between retries
  
  # Keep-alive Settings
  ping_interval: 30.0                # Ping every 30 seconds
  ping_timeout: 10.0                 # Ping response timeout
  
  # Buffer Configuration
  send_buffer_size: 8192             # WebSocket send buffer
  receive_buffer_size: 8192          # WebSocket receive buffer
```

---

## Smart Context Configuration

### Token Budget Management

```yaml
smart_context:
  # Total budget (OpenAI Realtime API limit)
  token_budget: 32000
  
  # Priority-based allocation
  priorities:
    system: 2000        # 6.25% - System instructions
    memory: 4000        # 12.5% - Conversation memory
    screen: 2000        # 6.25% - Screen content
    conversation: 24000 # 75%   - Current conversation
  
  # Dynamic allocation
  dynamic_allocation:
    enabled: true
    min_allocation_percent: 50       # Minimum guaranteed allocation
    reallocation_threshold: 0.9      # Reallocate when 90% full
```

### Relevance Scoring

```yaml
smart_context:
  relevance:
    enabled: true
    
    # Scoring Model
    model: "sentence-transformers/all-MiniLM-L6-v2"
    device: "auto"                   # auto|cpu|cuda
    
    # Thresholds
    threshold: 0.6                   # Minimum relevance score
    memory_threshold: 0.5            # Lower threshold for memory
    screen_threshold: 0.4            # Lower threshold for screen
    
    # Batch Processing
    batch_size: 32                   # Sentences per batch
    max_workers: 4                   # Parallel workers
```

### Context Compression

```yaml
smart_context:
  compression:
    enabled: true
    
    # Compression Methods
    method: "extractive"             # extractive|abstractive|hybrid
    ratio: 0.7                       # Target compression ratio
    
    # Extractive Settings
    extractive:
      algorithm: "textrank"          # textrank|luhn|lsa
      sentence_count: 3              # Sentences to extract
      
    # Abstractive Settings (requires additional models)
    abstractive:
      model: "facebook/bart-large-cnn"
      max_length: 150
      min_length: 30
```

---

## Audio Configuration

### Device Selection

```yaml
audio:
  # Automatic Device Detection
  input_device: null                 # null = auto-detect best input
  output_device: null                # null = auto-detect best output
  
  # Manual Device Selection (use device index or name)
  # input_device: 0                  # Device index
  # input_device: "MacBook Pro Microphone"  # Device name
  
  # Device Requirements
  device_requirements:
    min_sample_rate: 16000           # Minimum supported sample rate
    preferred_sample_rate: 24000     # Preferred sample rate
    min_channels: 1                  # Minimum channels required
    max_latency_ms: 50               # Maximum acceptable device latency
```

### Audio Quality Settings

```yaml
audio:
  # Core Settings
  sample_rate: 24000                 # 24kHz for high quality
  channels: 1                        # Mono audio
  chunk_size: 1024                   # Samples per processing chunk
  format: "int16"                    # 16-bit signed integers
  
  # Quality Enhancement
  processing:
    noise_reduction:
      enabled: true
      method: "spectral_subtraction" # spectral_subtraction|wiener
      strength: 0.5                  # 0.0 (none) to 1.0 (aggressive)
      
    echo_cancellation:
      enabled: true
      filter_length: 1024            # Echo cancellation filter length
      adaptation_rate: 0.01          # Learning rate
      
    auto_gain_control:
      enabled: true
      target_level: -20              # Target level in dB
      max_gain: 20                   # Maximum gain in dB
      
    volume_normalization:
      enabled: true
      target_lufs: -23               # Target loudness (LUFS)
      max_peak: -3                   # Maximum peak level (dB)
```

### Voice Activity Detection

```yaml
audio:
  vad:
    enabled: true
    
    # Sensitivity Settings
    sensitivity: 0.6                 # 0.0 (least) to 1.0 (most sensitive)
    energy_threshold: 300            # Energy threshold for speech
    
    # Timing Settings
    min_speech_duration: 0.5         # Minimum speech duration (seconds)
    min_silence_duration: 0.8        # Minimum silence for end-of-speech
    speech_timeout: 30.0             # Maximum speech duration
    
    # Advanced Settings
    lookahead_ms: 100                # Lookahead buffer for speech start
    lookback_ms: 200                 # Lookback buffer for speech end
    
    # Model-based VAD (optional)
    model_vad:
      enabled: false
      model: "silero-vad"            # silero-vad|webrtc-vad
      confidence_threshold: 0.7       # Model confidence threshold
```

---

## Performance Tuning

### Response Time Optimization

```yaml
performance:
  # Target Metrics
  targets:
    latency_ms: 300                  # Ultra-fast response target
    jitter_ms: 50                    # Response time consistency
    throughput_rps: 10               # Requests per second capacity
    
  # Optimization Strategies
  optimization:
    # Connection Optimization
    connection_pooling: true         # Reuse WebSocket connections
    persistent_connections: true     # Keep connections alive
    
    # Processing Optimization
    parallel_processing: true        # Parallel audio processing
    async_operations: true           # Asynchronous operations
    
    # Context Optimization
    context_caching: true            # Cache built contexts
    incremental_updates: true        # Update contexts incrementally
    
    # Audio Optimization
    audio_buffering: true            # Buffer audio for smooth playback
    low_latency_mode: true           # Optimize for low latency
```

### Resource Management

```yaml
performance:
  resources:
    # CPU Management
    max_cpu_percent: 80              # Maximum CPU usage
    cpu_affinity: []                 # CPU cores to use (empty = all)
    thread_pool_size: 4              # Worker thread pool size
    
    # Memory Management
    max_memory_mb: 1024              # Maximum memory usage
    memory_limit_percent: 80         # Memory usage threshold
    garbage_collection: "auto"       # auto|aggressive|conservative
    
    # GPU Management (if available)
    gpu_enabled: true                # Enable GPU acceleration
    max_gpu_percent: 70              # Maximum GPU usage
    gpu_memory_fraction: 0.8         # GPU memory allocation
    
    # Disk I/O
    async_file_operations: true      # Asynchronous file operations
    buffer_size_kb: 64               # File buffer size
```

### Caching Strategy

```yaml
performance:
  caching:
    # Context Caching
    context_cache:
      enabled: true
      max_size: 100                  # Maximum cached contexts
      ttl_seconds: 300               # Time to live
      
    # Audio Caching
    audio_cache:
      enabled: true
      max_size_mb: 50                # Maximum cache size
      compression: true              # Compress cached audio
      
    # Model Caching
    model_cache:
      enabled: true
      preload_models: true           # Preload ML models
      memory_mapping: true           # Use memory mapping
```

---

## Monitoring & Logging

### Metrics Collection

```yaml
monitoring:
  # Performance Metrics
  metrics:
    latency:
      buckets: [50, 100, 200, 300, 500, 1000, 2000, 5000]  # Histogram buckets (ms)
      percentiles: [0.5, 0.9, 0.95, 0.99]                  # Percentiles to track
      
    throughput:
      window_seconds: 60             # Moving average window
      
    error_rate:
      window_seconds: 300            # Error rate calculation window
      
    resource_usage:
      collection_interval: 5         # Collect every 5 seconds
      
  # Health Checks
  health_checks:
    - name: "realtime_api"
      interval: 30
      timeout: 10
      
    - name: "context_manager"
      interval: 60
      timeout: 5
      
    - name: "audio_system"
      interval: 15
      timeout: 3
```

### Alerting Configuration

```yaml
monitoring:
  alerting:
    enabled: true
    
    # Alert Rules
    rules:
      - name: "high_latency"
        condition: "latency_p95 > 500"
        severity: "warning"
        for: "2m"
        
      - name: "connection_failure"
        condition: "connection_errors > 3"
        severity: "critical"
        for: "30s"
        
      - name: "high_error_rate"
        condition: "error_rate > 0.1"
        severity: "warning"
        for: "5m"
    
    # Notification Channels
    channels:
      - type: "webhook"
        url: "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
        
      - type: "email"
        smtp_server: "smtp.gmail.com"
        recipients: ["admin@example.com"]
```

---

## Security Settings

### API Security

```yaml
security:
  api:
    # Rate Limiting
    rate_limiting:
      enabled: true
      algorithm: "token_bucket"      # token_bucket|sliding_window
      requests_per_minute: 60
      burst_size: 10
      
    # Request Validation
    validation:
      max_request_size_mb: 10        # Maximum request size
      allowed_audio_formats: ["pcm16", "wav", "mp3"]
      max_audio_duration: 60         # Maximum audio length (seconds)
      
    # CORS Settings
    cors:
      enabled: true
      allowed_origins: ["http://localhost:3000"]
      allowed_methods: ["GET", "POST", "OPTIONS"]
      allow_credentials: true
```

### Data Protection

```yaml
security:
  data:
    # Encryption
    encryption:
      at_rest:
        enabled: true
        algorithm: "AES-256-GCM"
        key_rotation_days: 90
        
      in_transit:
        enabled: true
        tls_version: "1.3"
        cipher_suites: ["TLS_AES_256_GCM_SHA384"]
        
    # Data Sanitization
    sanitization:
      enabled: true
      remove_pii: true               # Remove personally identifiable information
      anonymize_logs: true           # Anonymize sensitive log data
      
    # Access Control
    access_control:
      file_permissions: "600"        # Configuration file permissions
      log_permissions: "640"         # Log file permissions
      data_permissions: "700"        # Data directory permissions
```

---

## Deployment Configuration

### Docker Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  sovereign-assistant:
    image: sovereign:4.0
    ports:
      - "8080:8080"    # Dashboard
      - "9090:9090"    # Metrics
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - SOVEREIGN_MODE=hybrid
      - SOVEREIGN_LOG_LEVEL=INFO
    volumes:
      - ./config:/app/config
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
```

### Kubernetes Configuration

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sovereign-assistant
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sovereign-assistant
  template:
    metadata:
      labels:
        app: sovereign-assistant
    spec:
      containers:
      - name: sovereign
        image: sovereign:4.0
        ports:
        - containerPort: 8080
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai-api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

### Production Environment Variables

```bash
# Production Settings
SOVEREIGN_ENV=production
SOVEREIGN_MODE=hybrid
SOVEREIGN_LOG_LEVEL=INFO
SOVEREIGN_PERFORMANCE_MODE=balanced

# Scaling Settings
SOVEREIGN_MAX_WORKERS=8
SOVEREIGN_CONNECTION_POOL_SIZE=20
SOVEREIGN_CACHE_SIZE=200

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
METRICS_EXPORT_INTERVAL=10

# Security
ENCRYPTION_ENABLED=true
RATE_LIMITING_ENABLED=true
PII_DETECTION_ENABLED=true
```

---

## Best Practices

### Configuration Management

1. **Use Environment-Specific Configs**
   ```
   config/
   ├── sovereign.yaml          # Base configuration
   ├── sovereign.dev.yaml      # Development overrides
   ├── sovereign.prod.yaml     # Production overrides
   └── sovereign.test.yaml     # Testing overrides
   ```

2. **Secure Sensitive Data**
   - Store API keys in environment variables
   - Use encrypted configuration for production
   - Rotate API keys regularly
   - Implement proper access controls

3. **Monitor Configuration Changes**
   - Version control all configuration files
   - Log configuration changes
   - Validate configurations before deployment
   - Use infrastructure as code (IaC) tools

### Performance Optimization

1. **Tune for Your Hardware**
   - Adjust worker counts based on CPU cores
   - Optimize memory settings for available RAM
   - Configure GPU settings if available
   - Monitor resource usage and adjust accordingly

2. **Network Optimization**
   - Use connection pooling for better performance
   - Configure appropriate timeouts
   - Implement retry strategies with exponential backoff
   - Monitor network latency and adjust thresholds

3. **Audio Quality vs Latency**
   - Use 24kHz for best quality, 16kHz for lower latency
   - Adjust chunk sizes based on latency requirements
   - Enable audio processing features based on environment
   - Monitor audio quality metrics

---

This configuration reference provides comprehensive coverage of all available settings for optimal Sovereign 4.0 deployment and operation. 