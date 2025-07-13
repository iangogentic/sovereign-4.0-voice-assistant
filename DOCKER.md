# Sovereign Voice Assistant - Docker Deployment Guide

## Quick Start

### Production Deployment
```bash
# Clone and build
git clone <repository-url> sovereign-assistant
cd sovereign-assistant

# Create environment file
cp .env.example .env
# Edit .env with your API keys

# Start production stack
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs -f sovereign-assistant
```

### Development with Hot Reload
```bash
# Start development environment
docker-compose --profile dev up -d

# View logs
docker-compose logs -f sovereign-dev
```

## Multi-Stage Docker Architecture

### Available Build Targets

#### 1. **Production** (Default)
- Optimized runtime image
- Non-root user for security
- Health checks enabled
- Volume mounts for persistence

```bash
docker build --target production -t sovereign-assistant:latest .
```

#### 2. **Development**
- Hot reload capabilities
- Development tools included
- Debug logging enabled
- Jupyter notebook support

```bash
docker build --target development -t sovereign-assistant:dev .
```

#### 3. **Testing**
- CI/CD pipeline ready
- Testing framework included
- Code coverage reporting

```bash
docker build --target testing -t sovereign-assistant:test .
docker run --rm sovereign-assistant:test
```

#### 4. **Model Server**
- Dedicated model inference
- Minimal dependencies
- Optimized for ML workloads

```bash
docker build --target model-server -t sovereign-models:latest .
```

## Docker Compose Profiles

### Core Services (Default)
```bash
docker-compose up -d
```
- `sovereign-assistant` - Main application
- `chroma-db` - Vector database

### Development Profile
```bash
docker-compose --profile dev up -d
```
- `sovereign-dev` - Development version with hot reload
- `chroma-db` - Vector database
- Source code mounted for live editing

### Full Stack Profile
```bash
docker-compose --profile full up -d
```
Includes all services:
- Main application + database
- Model server (dedicated inference)
- Redis cache
- Nginx reverse proxy
- Prometheus + Grafana monitoring

### Individual Profiles
```bash
# Add caching layer
docker-compose --profile cache up -d redis

# Add monitoring
docker-compose --profile monitoring up -d prometheus grafana

# Add reverse proxy
docker-compose --profile proxy up -d nginx

# Add model server
docker-compose --profile models up -d model-server
```

## Environment Configuration

### Required Environment Variables
Create a `.env` file in the project root:

```env
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here

# Optional API Keys (enables additional features)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
PERPLEXITY_API_KEY=your_perplexity_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here

# Build Configuration
BUILD_VERSION=1.0.0
BUILD_DATE=2024-01-01T00:00:00Z
VCS_REF=abc123

# Monitoring (optional)
GRAFANA_PASSWORD=secure_password_here
```

### Advanced Configuration
```env
# Application Settings
SOVEREIGN_ENV=production
LOG_LEVEL=INFO
DEBUG=false

# Performance Tuning
MAX_WORKERS=4
MEMORY_LIMIT=4096
CACHE_SIZE=1000

# Database Settings
CHROMA_HOST=chroma-db
CHROMA_PORT=8000

# Redis Settings (if using cache profile)
REDIS_HOST=redis
REDIS_PORT=6379

# Feature Flags
OFFLINE_MODE=false
OCR_ENABLED=true
MEMORY_RECORDING=true
```

## Volume Management

### Persistent Data
The following volumes store persistent data:

- **`sovereign_data`** - Application data, models, user files
- **`sovereign_logs`** - Application logs
- **`sovereign_config`** - Configuration files
- **`chroma_data`** - Vector database storage

### Backup and Restore
```bash
# Backup data volumes
docker run --rm -v sovereign_data:/data -v $(pwd):/backup alpine tar czf /backup/sovereign-data-backup.tar.gz -C /data .
docker run --rm -v chroma_data:/data -v $(pwd):/backup alpine tar czf /backup/chroma-backup.tar.gz -C /data .

# Restore data volumes
docker run --rm -v sovereign_data:/data -v $(pwd):/backup alpine tar xzf /backup/sovereign-data-backup.tar.gz -C /data
docker run --rm -v chroma_data:/data -v $(pwd):/backup alpine tar xzf /backup/chroma-backup.tar.gz -C /data
```

## Audio Device Configuration

### Linux Host
Audio devices are automatically mapped in docker-compose:
```yaml
volumes:
  - /dev/snd:/dev/snd
devices:
  - /dev/snd
```

### macOS Host
Audio in Docker requires additional setup:
```bash
# Install PulseAudio (for audio forwarding)
brew install pulseaudio

# Start PulseAudio in system mode
pulseaudio --system --disallow-exit --disallow-module-loading

# Run container with audio
docker run -e PULSE_RUNTIME_PATH=/var/run/pulse sovereign-assistant:latest
```

### Windows Host (WSL2)
```bash
# In WSL2, audio devices should be accessible
# Ensure WSL2 integration is enabled in Docker Desktop
docker-compose up -d
```

## Multi-Architecture Support

### Build for Multiple Platforms
```bash
# Enable buildx
docker buildx create --use

# Build for multiple architectures
docker buildx build \
  --platform linux/amd64,linux/arm64,linux/arm/v7 \
  --target production \
  -t sovereign-assistant:multi \
  --push .
```

### Platform-Specific Builds
```bash
# Intel/AMD x86_64
docker build --platform linux/amd64 -t sovereign-assistant:amd64 .

# Apple Silicon / ARM64
docker build --platform linux/arm64 -t sovereign-assistant:arm64 .

# Raspberry Pi / ARM32
docker build --platform linux/arm/v7 -t sovereign-assistant:armv7 .
```

## Development Workflow

### Hot Reload Development
```bash
# Start development environment
docker-compose --profile dev up -d

# View logs in real-time
docker-compose logs -f sovereign-dev

# Make changes to source code (automatically reloaded)
# Access Jupyter at http://localhost:8888 (if enabled)
```

### Testing in Docker
```bash
# Run tests
docker-compose -f docker-compose.yml -f docker-compose.test.yml up --build testing

# Run specific test
docker run --rm sovereign-assistant:test python -m pytest tests/test_specific.py -v

# Run with coverage
docker run --rm sovereign-assistant:test python -m pytest tests/ --cov=assistant --cov-report=html
```

### Debugging
```bash
# Access container shell
docker-compose exec sovereign-assistant bash

# Check application status
docker-compose exec sovereign-assistant python -c "
import requests
response = requests.get('http://localhost:8080/health')
print(response.json())
"

# View container resource usage
docker stats sovereign-assistant

# Inspect volumes
docker volume inspect sovereign_data
```

## Production Deployment

### Resource Requirements
- **Minimum**: 4GB RAM, 2 CPU cores, 10GB storage
- **Recommended**: 8GB RAM, 4 CPU cores, 50GB storage
- **Audio devices**: Microphone and speaker access required

### Resource Limits
Configure in docker-compose.yml:
```yaml
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2.0'
    reservations:
      memory: 2G
      cpus: '1.0'
```

### Health Monitoring
```bash
# Check service health
docker-compose ps

# View health check logs
docker inspect sovereign-assistant | jq '.[0].State.Health'

# Manual health check
curl http://localhost:8080/health
```

### Log Management
```bash
# View logs
docker-compose logs -f sovereign-assistant

# Limit log size
docker-compose logs --tail=100 sovereign-assistant

# Export logs
docker-compose logs --no-color sovereign-assistant > assistant.log
```

## Scaling and Load Balancing

### Horizontal Scaling
```bash
# Scale main service
docker-compose up -d --scale sovereign-assistant=3

# With load balancer
docker-compose --profile full up -d --scale sovereign-assistant=3
```

### Custom Load Balancer Configuration
Create `docker/nginx.conf`:
```nginx
upstream sovereign_backend {
    server sovereign-assistant:8080;
    server sovereign-assistant:8080;
    server sovereign-assistant:8080;
}

server {
    listen 80;
    location / {
        proxy_pass http://sovereign_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Monitoring and Observability

### Prometheus Metrics
Access at `http://localhost:9090`

### Grafana Dashboards
Access at `http://localhost:3000`
- Default credentials: admin/admin (change via GRAFANA_PASSWORD)

### Custom Monitoring
```bash
# Add custom metrics endpoint
curl http://localhost:8080/metrics

# Monitor container metrics
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  cadvisor/cadvisor:latest
```

## Security Considerations

### Container Security
- Runs as non-root user (`sovereign`)
- Minimal base image (python:3.11-slim)
- No unnecessary packages in production
- Environment variables for sensitive data

### Network Security
```yaml
networks:
  sovereign-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### Secrets Management
```bash
# Use Docker secrets (Swarm mode)
echo "your_api_key" | docker secret create openai_api_key -

# Or use external secret management
# - HashiCorp Vault
# - AWS Secrets Manager
# - Azure Key Vault
```

## Troubleshooting

### Common Issues

#### Build Failures
```bash
# Clear build cache
docker builder prune

# Rebuild without cache
docker-compose build --no-cache

# Check build context size
du -sh .
```

#### Audio Issues
```bash
# Check audio devices
docker run --rm --device /dev/snd sovereign-assistant:latest \
  python -c "import pyaudio; p = pyaudio.PyAudio(); print(f'Devices: {p.get_device_count()}')"

# Test audio permissions
ls -la /dev/snd
```

#### Memory Issues
```bash
# Check container memory usage
docker stats --no-stream

# Increase memory limits
# Edit docker-compose.yml deploy.resources.limits.memory
```

#### Database Connection
```bash
# Check Chroma database
curl http://localhost:8000/api/v1/heartbeat

# Reset database
docker-compose down
docker volume rm sovereign_chroma_data
docker-compose up -d
```

### Debug Commands
```bash
# Container shell access
docker-compose exec sovereign-assistant bash

# Check Python environment
docker-compose exec sovereign-assistant python --version
docker-compose exec sovereign-assistant pip list

# Check environment variables
docker-compose exec sovereign-assistant env | grep SOVEREIGN

# Check network connectivity
docker-compose exec sovereign-assistant curl http://chroma-db:8000/api/v1/heartbeat

# Check disk usage
docker system df
docker images
docker volume ls
```

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Build and Deploy
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Build and test
      run: |
        docker build --target testing -t sovereign-test .
        docker run --rm sovereign-test
    - name: Build production
      run: |
        docker build --target production -t sovereign-assistant:${{ github.sha }} .
```

### GitLab CI Example
```yaml
stages:
  - build
  - test
  - deploy

build:
  stage: build
  script:
    - docker build --target production -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
```

This Docker deployment provides enterprise-grade containerization with multi-stage builds, comprehensive orchestration, and production-ready configurations for the Sovereign Voice Assistant. 