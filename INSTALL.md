# Sovereign Voice Assistant - Installation Guide

## Quick Start

### Unix/Linux/macOS
```bash
# Make executable and run
chmod +x run.sh
./run.sh
```

### Windows
```cmd
# Run from Command Prompt or PowerShell
run.bat
```

## Installation Options

### Standard Installation
```bash
# Unix/Linux/macOS
./run.sh

# Windows
run.bat
```
This performs all checks, installs dependencies, and starts the assistant.

### Development Mode
```bash
# Unix/Linux/macOS
./run.sh --dev

# Windows
run.bat --dev
```
Enables debug logging, development environment, and hot reload capabilities.

### Quick Start (Skip Checks)
```bash
# Unix/Linux/macOS - Skip dependency and health checks
./run.sh --no-deps --no-health

# Windows
run.bat --no-deps --no-health
```
For subsequent runs when dependencies are already installed.

## Prerequisites

### Required Dependencies
1. **Python 3.11+** - The core runtime environment
2. **Tesseract OCR** - For screen text recognition
3. **Audio System** - Microphone and speaker access

### API Keys (Required)
Create a `.env` file in the project root with:
```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (enables additional features)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
PERPLEXITY_API_KEY=your_perplexity_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

## Platform-Specific Setup

### Ubuntu/Debian Linux
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip
sudo apt-get install -y tesseract-ocr libtesseract-dev
sudo apt-get install -y portaudio19-dev libasound2-dev libpulse-dev
sudo apt-get install -y ffmpeg

# Run the assistant
./run.sh
```

### CentOS/RHEL/Fedora
```bash
# Install system dependencies
sudo dnf install -y python311 python311-pip python311-devel
sudo dnf install -y tesseract tesseract-devel
sudo dnf install -y portaudio-devel alsa-lib-devel pulseaudio-libs-devel
sudo dnf install -y ffmpeg

# Run the assistant
./run.sh
```

### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install python@3.11 tesseract portaudio ffmpeg

# Run the assistant
./run.sh
```

### Windows 10/11
1. **Install Python 3.11+** from [python.org](https://www.python.org/downloads/)
2. **Install Tesseract OCR** from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - Add Tesseract to your PATH
3. **Install Visual C++ Redistributable** from [Microsoft](https://aka.ms/vs/17/release/vc_redist.x64.exe)

```cmd
# Run the assistant
run.bat
```

#### Alternative Windows Installation (Chocolatey)
```powershell
# Install Chocolatey if not already installed
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install dependencies
choco install python tesseract

# Run the assistant
run.bat
```

## Troubleshooting

### Common Issues

#### Python Version Error
```
[ERROR] Python 3.11+ required, found 3.9.x
```
**Solution:** Install Python 3.11 or later from [python.org](https://www.python.org/downloads/)

#### Tesseract Not Found
```
[WARN] Tesseract OCR not found
```
**Solutions:**
- **Linux:** `sudo apt-get install tesseract-ocr` (Ubuntu/Debian) or `sudo dnf install tesseract` (CentOS/Fedora)
- **macOS:** `brew install tesseract`
- **Windows:** Download from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)

#### Audio System Issues
```
[WARN] Audio system issues detected
```
**Solutions:**
- **Linux:** Install audio development libraries: `sudo apt-get install portaudio19-dev libasound2-dev`
- **macOS:** Install portaudio: `brew install portaudio`
- **Windows:** Ensure audio drivers are installed and microphone permissions are granted

#### Missing API Keys
```
[ERROR] Missing required environment variables: OPENAI_API_KEY
```
**Solution:** Create a `.env` file with your API keys (see API Keys section above)

#### Virtual Environment Creation Failed
```
[ERROR] Failed to create virtual environment
```
**Solutions:**
- Ensure Python venv module is available: `python3 -m venv --help`
- On Ubuntu/Debian: `sudo apt-get install python3.11-venv`
- Check disk space and permissions

#### Package Installation Timeout/Failure
```
[ERROR] Failed to install requirements (timeout or error)
```
**Solutions:**
- Run with `--no-deps` to skip automatic installation: `./run.sh --no-deps`
- Manually install requirements: `pip install -r requirements.txt`
- Check internet connection and Python package index accessibility
- On slow connections, manually install large packages first: `pip install torch tensorflow`

### Performance Issues

#### Low Memory Warning
```
[WARN] Low available memory: 2048MB available, 4GB recommended
```
**Solution:** Close other applications or add more RAM. The assistant needs at least 4GB of available memory for optimal performance.

#### Low Disk Space Warning
```
[WARN] Low disk space: 1024MB available, 2GB+ recommended
```
**Solution:** Free up disk space. The assistant needs at least 2GB for models, logs, and temporary files.

### Audio Troubleshooting

#### No Audio Devices Found
```
[WARN] No audio devices found
```
**Solutions:**
- Check audio device connections
- On Linux: `pulseaudio --check -v` and restart if needed
- On macOS: Check System Preferences > Sound
- On Windows: Check Device Manager > Audio inputs and outputs

#### Audio Permission Issues
- **macOS:** Grant microphone permission in System Preferences > Security & Privacy > Privacy > Microphone
- **Windows:** Grant microphone permission in Settings > Privacy > Microphone
- **Linux:** Add user to audio group: `sudo usermod -a -G audio $USER`

## Advanced Configuration

### Environment Variables
```env
# Development mode
SOVEREIGN_ENV=development
LOG_LEVEL=DEBUG
DEBUG=true

# Performance tuning
MAX_WORKERS=4
CACHE_SIZE=1000
MEMORY_LIMIT=4096

# Feature flags
OFFLINE_MODE=false
OCR_ENABLED=true
MEMORY_RECORDING=true
```

### Command Line Options
```bash
# All available options
./run.sh --help

# Specific use cases
./run.sh --dev                    # Development mode with hot reload
./run.sh --no-deps                # Skip dependency checking (faster)
./run.sh --no-health              # Skip health checks
./run.sh --no-models              # Skip model setup
./run.sh --no-deps --no-health    # Quick start for development
```

### Custom Installation Paths
You can modify the script variables at the top of `run.sh` or `run.bat`:
```bash
# Custom paths (edit in script)
VENV_DIR="/custom/path/venv"
DATA_DIR="/custom/path/data"
LOG_DIR="/custom/path/logs"
```

## Security Considerations

### API Key Security
- Never commit `.env` files to version control
- Use environment-specific API keys
- Rotate API keys regularly
- Consider using API key management services for production

### File Permissions
- The launcher scripts will refuse to run as root/administrator
- Ensure proper file permissions: `chmod 755 run.sh`
- Keep the virtual environment directory secure

### Network Security
- The assistant makes API calls to AI services
- Consider firewall rules for production deployments
- Use HTTPS-only endpoints for API communications

## Getting Help

### Log Files
Check the log files for detailed error information:
- **Unix/Linux/macOS:** `logs/sovereign.log`
- **Windows:** `logs\sovereign.log`

### Debug Mode
Run in debug mode for verbose output:
```bash
./run.sh --dev        # Unix/Linux/macOS
run.bat --dev         # Windows
```

### Common Commands
```bash
# Check Python installation
python3 --version

# Check virtual environment
source venv/bin/activate && python --version

# Manual package installation
pip install -r requirements.txt

# Test audio devices
python3 -c "import pyaudio; p = pyaudio.PyAudio(); print(f'Audio devices: {p.get_device_count()}')"

# Test API keys
python3 -c "import os; print('OpenAI key:', bool(os.getenv('OPENAI_API_KEY')))"
```

## Support

For additional help:
1. Check the log files for detailed error messages
2. Run in debug mode: `./run.sh --dev`
3. Verify all prerequisites are installed
4. Check the project documentation
5. Ensure API keys are properly configured

The installation scripts are designed to handle most common scenarios automatically. If you encounter issues not covered here, the debug output should provide additional clues for troubleshooting. 