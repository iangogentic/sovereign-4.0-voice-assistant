#!/bin/bash
# scripts/docker-build.sh - Docker build automation script
# Supports multi-stage builds, versioning, and multi-architecture

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="sovereign-assistant"
REGISTRY=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    if [[ "${DEBUG:-}" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] <target>

Build Docker images for Sovereign Voice Assistant

TARGETS:
    production      Build production image (default)
    development     Build development image with hot reload
    testing         Build testing image for CI/CD
    model-server    Build dedicated model server
    all             Build all targets

OPTIONS:
    --registry <url>     Docker registry URL (e.g., docker.io/myuser)
    --version <tag>      Version tag (default: auto-generated)
    --push               Push to registry after build
    --no-cache           Build without using cache
    --platform <list>    Build for specific platforms (e.g., linux/amd64,linux/arm64)
    --buildx             Use Docker buildx for multi-platform builds
    --help               Show this help message

EXAMPLES:
    $0 production
    $0 --version v1.0.0 --push production
    $0 --platform linux/amd64,linux/arm64 --buildx all
    $0 --registry ghcr.io/myuser --push production

EOF
}

# Generate version tag
generate_version() {
    if [[ -n "${VERSION:-}" ]]; then
        echo "$VERSION"
        return
    fi
    
    # Try to get version from git
    if command -v git &> /dev/null && git rev-parse --git-dir > /dev/null 2>&1; then
        local git_hash=$(git rev-parse --short HEAD)
        local git_tag=$(git describe --tags --exact-match 2>/dev/null || echo "")
        
        if [[ -n "$git_tag" ]]; then
            echo "$git_tag"
        else
            echo "dev-$git_hash"
        fi
    else
        echo "latest"
    fi
}

# Get build metadata
get_build_metadata() {
    export BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    export VCS_REF=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
    export BUILD_VERSION=$(generate_version)
    
    log_info "Build metadata:"
    log_info "  Version: $BUILD_VERSION"
    log_info "  Date: $BUILD_DATE"
    log_info "  VCS Ref: $VCS_REF"
}

# Build single target
build_target() {
    local target=$1
    local tag_suffix=${2:-$target}
    
    log_info "Building target: $target"
    
    local image_tag="${REGISTRY:+$REGISTRY/}$IMAGE_NAME:$tag_suffix"
    if [[ "$tag_suffix" != "latest" ]]; then
        image_tag="${image_tag}-${BUILD_VERSION}"
    fi
    
    local build_args=(
        "--target" "$target"
        "--tag" "$image_tag"
        "--build-arg" "BUILD_VERSION=$BUILD_VERSION"
        "--build-arg" "BUILD_DATE=$BUILD_DATE"
        "--build-arg" "VCS_REF=$VCS_REF"
    )
    
    # Add cache options
    if [[ "${NO_CACHE:-}" == "true" ]]; then
        build_args+=("--no-cache")
    fi
    
    # Add platform options
    if [[ -n "${PLATFORM:-}" ]]; then
        build_args+=("--platform" "$PLATFORM")
    fi
    
    # Choose build command
    local build_cmd="docker build"
    if [[ "${USE_BUILDX:-}" == "true" ]]; then
        build_cmd="docker buildx build"
        if [[ "${PUSH:-}" == "true" ]]; then
            build_args+=("--push")
        fi
    fi
    
    # Execute build
    log_info "Executing: $build_cmd ${build_args[*]} $PROJECT_ROOT"
    $build_cmd "${build_args[@]}" "$PROJECT_ROOT"
    
    # Push if requested and not using buildx
    if [[ "${PUSH:-}" == "true" && "${USE_BUILDX:-}" != "true" ]]; then
        log_info "Pushing $image_tag"
        docker push "$image_tag"
    fi
    
    log_info "Successfully built: $image_tag"
}

# Build all targets
build_all_targets() {
    log_info "Building all targets"
    
    build_target "production" "latest"
    build_target "development" "dev"
    build_target "testing" "test"
    build_target "model-server" "models"
    
    log_info "All targets built successfully"
}

# Validate Docker environment
validate_environment() {
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker."
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon not running. Please start Docker."
        exit 1
    fi
    
    # Check buildx if requested
    if [[ "${USE_BUILDX:-}" == "true" ]]; then
        if ! docker buildx version &> /dev/null; then
            log_error "Docker buildx not available. Please install buildx."
            exit 1
        fi
        
        # Create builder if needed
        if ! docker buildx inspect sovereign-builder &> /dev/null; then
            log_info "Creating buildx builder: sovereign-builder"
            docker buildx create --name sovereign-builder --use
        else
            docker buildx use sovereign-builder
        fi
    fi
    
    log_info "Docker environment validated"
}

# Check project structure
validate_project() {
    cd "$PROJECT_ROOT"
    
    # Check required files
    local required_files=("Dockerfile" "requirements.txt" "docker-compose.yml")
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "Required file not found: $file"
            exit 1
        fi
    done
    
    # Check .dockerignore
    if [[ ! -f ".dockerignore" ]]; then
        log_warn ".dockerignore not found - build context may be large"
    fi
    
    log_info "Project structure validated"
}

# Main execution
main() {
    local target=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --registry)
                REGISTRY="$2"
                shift 2
                ;;
            --version)
                VERSION="$2"
                shift 2
                ;;
            --push)
                PUSH="true"
                shift
                ;;
            --no-cache)
                NO_CACHE="true"
                shift
                ;;
            --platform)
                PLATFORM="$2"
                USE_BUILDX="true"
                shift 2
                ;;
            --buildx)
                USE_BUILDX="true"
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            --debug)
                DEBUG="true"
                shift
                ;;
            -*)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
            *)
                target="$1"
                shift
                ;;
        esac
    done
    
    # Default target
    if [[ -z "$target" ]]; then
        target="production"
    fi
    
    # Validate target
    case $target in
        production|development|testing|model-server|all)
            ;;
        *)
            log_error "Invalid target: $target"
            show_usage
            exit 1
            ;;
    esac
    
    log_info "Starting Docker build for target: $target"
    
    # Validation
    validate_environment
    validate_project
    
    # Get build metadata
    get_build_metadata
    
    # Build
    case $target in
        all)
            build_all_targets
            ;;
        *)
            build_target "$target" "$target"
            ;;
    esac
    
    log_info "Build completed successfully!"
}

# Run main function
main "$@" 