#!/bin/bash
# run.sh - Universal launcher for Sovereign Voice Assistant
# Compatible with Linux, macOS, and other Unix-like systems

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"
CONFIG_DIR="${SCRIPT_DIR}/config"
DATA_DIR="${SCRIPT_DIR}/data"
LOG_DIR="${SCRIPT_DIR}/logs"
MODELS_DIR="${DATA_DIR}/offline_models"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
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
        echo -e "${CYAN}[DEBUG]${NC} $1"
    fi
}

log_header() {
    echo -e "${PURPLE}=== $1 ===${NC}"
}

# Platform detection
detect_platform() {
    case "$(uname -s)" in
        Linux*)
            PLATFORM="linux"
            if command -v apt-get &> /dev/null; then
                PACKAGE_MANAGER="apt"
            elif command -v yum &> /dev/null; then
                PACKAGE_MANAGER="yum"
            elif command -v dnf &> /dev/null; then
                PACKAGE_MANAGER="dnf"
            elif command -v pacman &> /dev/null; then
                PACKAGE_MANAGER="pacman"
            elif command -v zypper &> /dev/null; then
                PACKAGE_MANAGER="zypper"
            else
                PACKAGE_MANAGER="unknown"
            fi
            ;;
        Darwin*)
            PLATFORM="macos"
            PACKAGE_MANAGER="brew"
            ;;
        CYGWIN*|MINGW*|MSYS*)
            PLATFORM="windows"
            PACKAGE_MANAGER="choco"
            ;;
        *)
            PLATFORM="unknown"
            PACKAGE_MANAGER="unknown"
            ;;
    esac
    
    # Detect architecture
    ARCH="$(uname -m)"
    case "$ARCH" in
        x86_64|amd64)
            ARCH="x64"
            ;;
        arm64|aarch64)
            ARCH="arm64"
            ;;
        armv7l)
            ARCH="arm"
            ;;
    esac
    
    log_info "Detected platform: $PLATFORM ($ARCH) with package manager: $PACKAGE_MANAGER"
}

# Check if running as root (should not for security)
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "Do not run as root for security reasons"
        log_info "Please run as a regular user"
        exit 1
    fi
}

# Python version checking
check_python() {
    log_header "Checking Python Installation"
    
    # Check if python3 exists
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found"
        install_python
        return
    fi
    
    # Check Python version
    local python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    local required_version="3.11"
    
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
        log_info "Python $python_version found (>= $required_version required) ✓"
    else
        log_warn "Python $python_version found, but >= $required_version required"
        install_python
    fi
}

# Install Python
install_python() {
    log_info "Installing Python..."
    
    case $PLATFORM in
        linux)
            case $PACKAGE_MANAGER in
                apt)
                    sudo apt-get update
                    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip
                    ;;
                yum|dnf)
                    sudo $PACKAGE_MANAGER install -y python311 python311-pip python311-devel
                    ;;
                pacman)
                    sudo pacman -S --noconfirm python python-pip
                    ;;
                zypper)
                    sudo zypper install -y python311 python311-pip python311-devel
                    ;;
                *)
                    log_error "Unsupported package manager: $PACKAGE_MANAGER"
                    log_info "Please install Python 3.11+ manually"
                    exit 1
                    ;;
            esac
            ;;
        macos)
            if command -v brew &> /dev/null; then
                brew install python@3.11
            else
                log_error "Homebrew not found. Please install Python 3.11+ manually or install Homebrew first"
                log_info "Install Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
                exit 1
            fi
            ;;
        *)
            log_error "Unsupported platform for automatic Python installation"
            log_info "Please install Python 3.11+ manually"
            exit 1
            ;;
    esac
}

# Check system dependencies
check_system_dependencies() {
    log_header "Checking System Dependencies"
    
    local missing_deps=()
    
    # Check Tesseract OCR
    if ! command -v tesseract &> /dev/null; then
        log_warn "Tesseract OCR not found"
        missing_deps+=("tesseract")
    else
        local tess_version=$(tesseract --version 2>&1 | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")
        log_info "Tesseract $tess_version found ✓"
    fi
    
    # Check audio dependencies based on platform
    case $PLATFORM in
        linux)
            # Check for portaudio development headers
            if ! pkg-config --exists portaudio-2.0 2>/dev/null; then
                missing_deps+=("portaudio-dev")
            fi
            
            # Check for ALSA development headers
            if ! pkg-config --exists alsa 2>/dev/null; then
                missing_deps+=("alsa-dev")
            fi
            
            # Check for PulseAudio development headers
            if ! pkg-config --exists libpulse 2>/dev/null; then
                missing_deps+=("pulseaudio-dev")
            fi
            
            # Check for FFmpeg
            if ! command -v ffmpeg &> /dev/null; then
                missing_deps+=("ffmpeg")
            fi
            ;;
        macos)
            # Check for portaudio via Homebrew
            if ! brew list portaudio &> /dev/null; then
                missing_deps+=("portaudio")
            fi
            
            # Check for FFmpeg
            if ! command -v ffmpeg &> /dev/null; then
                missing_deps+=("ffmpeg")
            fi
            ;;
    esac
    
    # Install missing dependencies
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_warn "Missing dependencies: ${missing_deps[*]}"
        install_system_dependencies "${missing_deps[@]}"
    else
        log_info "All system dependencies found ✓"
    fi
}

# Install system dependencies
install_system_dependencies() {
    local deps=("$@")
    log_info "Installing system dependencies: ${deps[*]}"
    
    case $PLATFORM in
        linux)
            case $PACKAGE_MANAGER in
                apt)
                    # Map generic names to apt package names
                    local apt_deps=()
                    for dep in "${deps[@]}"; do
                        case $dep in
                            tesseract) apt_deps+=("tesseract-ocr" "libtesseract-dev") ;;
                            portaudio-dev) apt_deps+=("portaudio19-dev") ;;
                            alsa-dev) apt_deps+=("libasound2-dev") ;;
                            pulseaudio-dev) apt_deps+=("libpulse-dev") ;;
                            ffmpeg) apt_deps+=("ffmpeg") ;;
                        esac
                    done
                    
                    sudo apt-get update
                    sudo apt-get install -y "${apt_deps[@]}"
                    ;;
                yum|dnf)
                    # Map generic names to RPM package names
                    local rpm_deps=()
                    for dep in "${deps[@]}"; do
                        case $dep in
                            tesseract) rpm_deps+=("tesseract" "tesseract-devel") ;;
                            portaudio-dev) rpm_deps+=("portaudio-devel") ;;
                            alsa-dev) rpm_deps+=("alsa-lib-devel") ;;
                            pulseaudio-dev) rpm_deps+=("pulseaudio-libs-devel") ;;
                            ffmpeg) rpm_deps+=("ffmpeg") ;;
                        esac
                    done
                    
                    sudo $PACKAGE_MANAGER install -y "${rpm_deps[@]}"
                    ;;
                pacman)
                    # Map generic names to Arch package names
                    local arch_deps=()
                    for dep in "${deps[@]}"; do
                        case $dep in
                            tesseract) arch_deps+=("tesseract") ;;
                            portaudio-dev) arch_deps+=("portaudio") ;;
                            alsa-dev) arch_deps+=("alsa-lib") ;;
                            pulseaudio-dev) arch_deps+=("libpulse") ;;
                            ffmpeg) arch_deps+=("ffmpeg") ;;
                        esac
                    done
                    
                    sudo pacman -S --noconfirm "${arch_deps[@]}"
                    ;;
                *)
                    log_error "Unsupported package manager for dependency installation"
                    log_info "Please install these dependencies manually: ${deps[*]}"
                    ;;
            esac
            ;;
        macos)
            if command -v brew &> /dev/null; then
                # Map generic names to Homebrew package names
                local brew_deps=()
                for dep in "${deps[@]}"; do
                    case $dep in
                        tesseract) brew_deps+=("tesseract") ;;
                        portaudio) brew_deps+=("portaudio") ;;
                        ffmpeg) brew_deps+=("ffmpeg") ;;
                    esac
                done
                
                brew install "${brew_deps[@]}"
            else
                log_error "Homebrew not available for dependency installation"
                log_info "Please install these dependencies manually: ${deps[*]}"
            fi
            ;;
        *)
            log_error "Unsupported platform for automatic dependency installation"
            log_info "Please install these dependencies manually: ${deps[*]}"
            ;;
    esac
}

# Virtual environment management
setup_virtual_environment() {
    log_header "Setting Up Virtual Environment"
    
    if [ ! -d "$VENV_DIR" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    else
        log_info "Virtual environment already exists ✓"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    log_info "Virtual environment activated ✓"
    
    # Upgrade pip
    log_info "Upgrading pip..."
    python -m pip install --upgrade pip
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        log_info "Installing Python requirements..."
        # Use timeout if available, otherwise install normally
        if command -v timeout &> /dev/null; then
            timeout 600 pip install -r requirements.txt || {
                log_error "Failed to install requirements (timeout or error)"
                log_info "You may need to install some dependencies manually"
                return 1
            }
        else
            # Fallback for systems without timeout (like macOS)
            pip install -r requirements.txt || {
                log_error "Failed to install requirements"
                log_info "You may need to install some dependencies manually"
                return 1
            }
        fi
        log_info "Python requirements installed ✓"
    else
        log_warn "requirements.txt not found"
    fi
}

# Environment validation
validate_environment() {
    log_header "Validating Environment"
    
    # Required environment variables
    local required_vars=("OPENAI_API_KEY")
    local optional_vars=("ANTHROPIC_API_KEY" "OPENROUTER_API_KEY" "PERPLEXITY_API_KEY" "GOOGLE_API_KEY" "ELEVENLABS_API_KEY")
    
    # Load .env file if it exists
    if [ -f ".env" ]; then
        log_info "Loading environment from .env file"
        set -a  # Automatically export all variables
        source .env
        set +a
    fi
    
    # Check required variables
    local missing_required=()
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            missing_required+=("$var")
        else
            log_info "$var is set ✓"
        fi
    done
    
    # Check optional variables
    local missing_optional=()
    for var in "${optional_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            missing_optional+=("$var")
        else
            log_info "$var is set ✓"
        fi
    done
    
    if [ ${#missing_required[@]} -ne 0 ]; then
        log_error "Missing required environment variables: ${missing_required[*]}"
        log_info "Please create a .env file or set these variables:"
        for var in "${missing_required[@]}"; do
            echo "  export $var=your_api_key_here"
        done
        return 1
    fi
    
    if [ ${#missing_optional[@]} -ne 0 ]; then
        log_warn "Optional environment variables not set: ${missing_optional[*]}"
        log_info "Some features may be limited without these API keys"
    fi
    
    log_info "Environment validation completed ✓"
}

# Health checks
perform_health_checks() {
    log_header "Performing Health Checks"
    
    # Check disk space (require at least 2GB free)
    local available_space=$(df "$SCRIPT_DIR" | awk 'NR==2 {print $4}')
    local required_space=2097152  # 2GB in KB
    
    if [[ "$available_space" -lt "$required_space" ]]; then
        log_warn "Low disk space: $(( available_space / 1024 ))MB available, 2GB recommended"
    else
        log_info "Disk space: $(( available_space / 1024 ))MB available ✓"
    fi
    
    # Check memory (require at least 4GB available)
    if command -v free &> /dev/null; then
        local available_memory=$(free -m | awk 'NR==2{printf "%d", $7}')
        if [[ "$available_memory" -lt 4096 ]]; then
            log_warn "Low available memory: ${available_memory}MB available, 4GB recommended"
        else
            log_info "Available memory: ${available_memory}MB ✓"
        fi
    elif [[ "$PLATFORM" == "macos" ]]; then
        # macOS memory check
        local total_memory=$(sysctl -n hw.memsize)
        local memory_gb=$(( total_memory / 1024 / 1024 / 1024 ))
        if [[ "$memory_gb" -lt 8 ]]; then
            log_warn "Total memory: ${memory_gb}GB, 8GB+ recommended"
        else
            log_info "Total memory: ${memory_gb}GB ✓"
        fi
    fi
    
    # Test audio devices
    python3 -c "
import pyaudio
import sys

try:
    p = pyaudio.PyAudio()
    device_count = p.get_device_count()
    if device_count == 0:
        print('Warning: No audio devices found')
        sys.exit(1)
    else:
        print(f'Audio devices: {device_count} found')
        sys.exit(0)
except Exception as e:
    print(f'Audio system error: {e}')
    sys.exit(1)
finally:
    try:
        p.terminate()
    except:
        pass
" && log_info "Audio system ✓" || log_warn "Audio system issues detected"
    
    log_info "Health checks completed"
}

# Create necessary directories
create_directories() {
    log_header "Creating Directories"
    
    local dirs=("$LOG_DIR" "$DATA_DIR" "$MODELS_DIR" "$CONFIG_DIR")
    
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        fi
    done
    
    log_info "Directory structure ready ✓"
}

# Model setup
setup_models() {
    log_header "Setting Up Models"
    
    if [ ! -d "$MODELS_DIR" ] || [ -z "$(ls -A "$MODELS_DIR" 2>/dev/null)" ]; then
        log_info "Setting up offline models..."
        python3 -c "
try:
    from assistant.offline_system import OfflineSystem
    offline = OfflineSystem()
    offline.download_models()
    print('Offline models setup completed')
except ImportError as e:
    print(f'Offline system not available: {e}')
except Exception as e:
    print(f'Error setting up models: {e}')
"
    else
        log_info "Offline models already present ✓"
    fi
}

# Graceful shutdown handler
cleanup() {
    log_info "Shutting down Sovereign Assistant..."
    
    if [[ -n "${ASSISTANT_PID:-}" ]]; then
        log_info "Stopping main process (PID: $ASSISTANT_PID)..."
        kill -TERM "$ASSISTANT_PID" 2>/dev/null || true
        
        # Wait for graceful shutdown
        local timeout=30
        while kill -0 "$ASSISTANT_PID" 2>/dev/null && [[ $timeout -gt 0 ]]; do
            sleep 1
            ((timeout--))
        done
        
        # Force kill if still running
        if kill -0 "$ASSISTANT_PID" 2>/dev/null; then
            log_warn "Force killing process..."
            kill -KILL "$ASSISTANT_PID" 2>/dev/null || true
        fi
    fi
    
    log_info "Shutdown complete"
}

# Signal handlers
trap cleanup SIGTERM SIGINT

# Usage information
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    --dev           Start in development mode with debug logging and hot reload
    --no-deps       Skip dependency checking and installation
    --no-health     Skip health checks
    --no-models     Skip model setup
    --help          Show this help message

ENVIRONMENT VARIABLES:
    DEBUG=true          Enable debug logging
    LOG_LEVEL=DEBUG     Set logging level (DEBUG, INFO, WARN, ERROR)
    SOVEREIGN_ENV=dev   Set environment mode

EXAMPLES:
    $0                  Start normally with all checks
    $0 --dev            Start in development mode
    $0 --no-deps        Start without checking dependencies (faster)

EOF
}

# Parse command line arguments
parse_arguments() {
    SKIP_DEPS=false
    SKIP_HEALTH=false
    SKIP_MODELS=false
    DEV_MODE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dev)
                DEV_MODE=true
                export SOVEREIGN_ENV=development
                export LOG_LEVEL=DEBUG
                export DEBUG=true
                shift
                ;;
            --no-deps)
                SKIP_DEPS=true
                shift
                ;;
            --no-health)
                SKIP_HEALTH=true
                shift
                ;;
            --no-models)
                SKIP_MODELS=true
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Main execution
main() {
    log_header "Sovereign Voice Assistant Launcher"
    log_info "Starting initialization..."
    
    # Platform detection
    detect_platform
    
    # Security check
    check_root
    
    # Parse arguments
    parse_arguments "$@"
    
    # Conditional setup steps
    if [[ "$SKIP_DEPS" != "true" ]]; then
        check_python
        check_system_dependencies
    fi
    
    # Always do these steps
    create_directories
    setup_virtual_environment
    validate_environment
    
    if [[ "$SKIP_HEALTH" != "true" ]]; then
        perform_health_checks
    fi
    
    if [[ "$SKIP_MODELS" != "true" ]]; then
        setup_models
    fi
    
    # Start the assistant
    log_header "Starting Sovereign Assistant"
    
    if [[ "$DEV_MODE" == "true" ]]; then
        log_info "Starting in development mode with hot reload..."
        python3 -m assistant.main --dev &
    else
        log_info "Starting in production mode..."
        python3 -m assistant.main &
    fi
    
    ASSISTANT_PID=$!
    log_info "Assistant started with PID: $ASSISTANT_PID"
    
    # Wait for the process to complete
    wait "$ASSISTANT_PID"
    
    log_info "Sovereign Assistant has stopped"
}

# Run main function with all arguments
main "$@" 