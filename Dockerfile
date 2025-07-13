# Sovereign Voice Assistant - Multi-Stage Docker Build
# Optimized for AI voice assistant with offline model support

# =============================================================================
# Base Stage - Common system dependencies
# =============================================================================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Core system tools
    curl \
    wget \
    unzip \
    git \
    # OCR dependencies
    tesseract-ocr \
    libtesseract-dev \
    # Audio dependencies
    portaudio19-dev \
    libasound2-dev \
    libpulse-dev \
    libsndfile1-dev \
    # Video/Image processing
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Build dependencies for Python packages
    build-essential \
    gcc \
    g++ \
    cmake \
    pkg-config \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app user for security
RUN groupadd -r sovereign && useradd -r -g sovereign sovereign

# Create necessary directories
RUN mkdir -p /app /app/data /app/logs /app/config \
    && chown -R sovereign:sovereign /app

# =============================================================================
# Dependencies Stage - Python package installation
# =============================================================================
FROM base as dependencies

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CPU support (smaller size for containers)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
RUN pip install -r requirements.txt

# =============================================================================
# Model Download Stage - Offline models
# =============================================================================
FROM dependencies as models

# Set working directory for models
WORKDIR /app/data/offline_models

# Copy model download script
COPY scripts/download_models.py /tmp/download_models.py

# Download offline models (if available)
RUN python /tmp/download_models.py --models whisper-tiny,piper-en-us || \
    echo "Model download script not available or failed - will download at runtime"

# Create model directory structure
RUN mkdir -p whisper piper tts ocr \
    && echo "Model directories created"

# =============================================================================
# Development Stage - Hot reload and debugging
# =============================================================================
FROM dependencies as development

# Install development dependencies
RUN pip install \
    watchdog \
    black \
    pytest \
    mypy \
    ipython \
    jupyter

# Set development environment
ENV SOVEREIGN_ENV=development \
    LOG_LEVEL=DEBUG \
    DEBUG=true

# Copy source code
COPY --chown=sovereign:sovereign . /app

# Copy models from model stage
COPY --from=models --chown=sovereign:sovereign /app/data/offline_models /app/data/offline_models

# Switch to app user
USER sovereign

# Expose development ports
EXPOSE 8080 8000 5000

# Development entry point with hot reload
CMD ["python", "-m", "assistant.main", "--dev"]

# =============================================================================
# Production Stage - Optimized runtime
# =============================================================================
FROM base as production

# Install only runtime Python dependencies
COPY --from=dependencies /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=sovereign:sovereign . /app

# Copy offline models
COPY --from=models --chown=sovereign:sovereign /app/data/offline_models /app/data/offline_models

# Set production environment
ENV SOVEREIGN_ENV=production \
    LOG_LEVEL=INFO \
    DEBUG=false

# Create volume mount points
VOLUME ["/app/data", "/app/logs", "/app/config"]

# Switch to app user
USER sovereign

# Set working directory
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Expose application port
EXPOSE 8080

# Production entry point
CMD ["python", "-m", "assistant.main"]

# =============================================================================
# Testing Stage - CI/CD pipeline testing
# =============================================================================
FROM dependencies as testing

# Install testing dependencies
RUN pip install \
    pytest \
    pytest-cov \
    pytest-asyncio \
    black \
    mypy \
    flake8

# Copy source code and tests
COPY --chown=sovereign:sovereign . /app

# Copy models for testing
COPY --from=models --chown=sovereign:sovereign /app/data/offline_models /app/data/offline_models

# Switch to app user
USER sovereign

# Set working directory
WORKDIR /app

# Set testing environment
ENV SOVEREIGN_ENV=testing \
    LOG_LEVEL=DEBUG

# Run tests by default
CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=assistant"]

# =============================================================================
# Model Serving Stage - Dedicated model inference
# =============================================================================
FROM base as model-server

# Install minimal dependencies for model serving
COPY --from=dependencies /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only model serving components
COPY --chown=sovereign:sovereign assistant/offline_system.py /app/assistant/
COPY --chown=sovereign:sovereign assistant/models/ /app/assistant/models/
COPY --chown=sovereign:sovereign assistant/__init__.py /app/assistant/

# Copy offline models
COPY --from=models --chown=sovereign:sovereign /app/data/offline_models /app/data/offline_models

# Switch to app user
USER sovereign

# Set working directory
WORKDIR /app

# Expose model serving port
EXPOSE 9000

# Model server entry point
CMD ["python", "-m", "assistant.offline_system", "--serve"]

# =============================================================================
# Multi-architecture support
# =============================================================================
# This Dockerfile supports multiple architectures:
# - linux/amd64 (Intel/AMD x86_64)
# - linux/arm64 (Apple Silicon, ARM64)
# - linux/arm/v7 (Raspberry Pi, ARM32)
#
# Build examples:
# docker build --target production -t sovereign-assistant:latest .
# docker build --target development -t sovereign-assistant:dev .
# docker build --target testing -t sovereign-assistant:test .
# docker buildx build --platform linux/amd64,linux/arm64 -t sovereign-assistant:multi .

# =============================================================================
# Build Arguments and Labels
# =============================================================================
ARG BUILD_VERSION=latest
ARG BUILD_DATE
ARG VCS_REF

LABEL org.opencontainers.image.title="Sovereign Voice Assistant" \
      org.opencontainers.image.description="AI Voice Assistant with offline capabilities" \
      org.opencontainers.image.version=${BUILD_VERSION} \
      org.opencontainers.image.created=${BUILD_DATE} \
      org.opencontainers.image.revision=${VCS_REF} \
      org.opencontainers.image.vendor="Sovereign AI" \
      org.opencontainers.image.source="https://github.com/sovereign-ai/voice-assistant" \
      org.opencontainers.image.documentation="https://github.com/sovereign-ai/voice-assistant/blob/main/README.md" \
      org.opencontainers.image.licenses="MIT" 