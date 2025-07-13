@echo off
:: run.bat - Windows launcher for Sovereign Voice Assistant
:: Compatible with Windows 10/11 with PowerShell support

setlocal enabledelayedexpansion

:: Configuration
set SCRIPT_DIR=%~dp0
set VENV_DIR=%SCRIPT_DIR%venv
set CONFIG_DIR=%SCRIPT_DIR%config
set DATA_DIR=%SCRIPT_DIR%data
set LOG_DIR=%SCRIPT_DIR%logs
set MODELS_DIR=%DATA_DIR%\offline_models

:: Colors for output (requires Windows 10 version 1607 or later)
for /f %%A in ('"prompt $H &echo on &for %%B in (1) do rem"') do set BS=%%A

:: Initialize variables
set SKIP_DEPS=false
set SKIP_HEALTH=false
set SKIP_MODELS=false
set DEV_MODE=false
set PYTHON_CMD=python

:: Header
echo.
echo ===============================================
echo    Sovereign Voice Assistant Launcher
echo ===============================================
echo Starting initialization...
echo.

:: Parse command line arguments
:parse_args
if "%~1"=="" goto :start_checks
if /i "%~1"=="--dev" (
    set DEV_MODE=true
    set SOVEREIGN_ENV=development
    set LOG_LEVEL=DEBUG
    set DEBUG=true
    shift
    goto :parse_args
)
if /i "%~1"=="--no-deps" (
    set SKIP_DEPS=true
    shift
    goto :parse_args
)
if /i "%~1"=="--no-health" (
    set SKIP_HEALTH=true
    shift
    goto :parse_args
)
if /i "%~1"=="--no-models" (
    set SKIP_MODELS=true
    shift
    goto :parse_args
)
if /i "%~1"=="--help" goto :show_help
if /i "%~1"=="-h" goto :show_help

echo [ERROR] Unknown option: %~1
goto :show_help

:show_help
echo Usage: %~nx0 [OPTIONS]
echo.
echo OPTIONS:
echo     --dev           Start in development mode with debug logging
echo     --no-deps       Skip dependency checking and installation
echo     --no-health     Skip health checks
echo     --no-models     Skip model setup
echo     --help          Show this help message
echo.
echo ENVIRONMENT VARIABLES:
echo     DEBUG=true          Enable debug logging
echo     LOG_LEVEL=DEBUG     Set logging level
echo     SOVEREIGN_ENV=dev   Set environment mode
echo.
echo EXAMPLES:
echo     %~nx0                  Start normally with all checks
echo     %~nx0 --dev            Start in development mode
echo     %~nx0 --no-deps        Start without checking dependencies
echo.
goto :end

:start_checks
:: Check if running as administrator (not recommended)
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [WARN] Running as administrator. Consider running as regular user for security.
    echo.
)

:: Platform detection
echo [INFO] Detected platform: Windows %PROCESSOR_ARCHITECTURE%
echo.

:: Check Python installation
if "%SKIP_DEPS%"=="false" (
    call :check_python
    if errorlevel 1 goto :error
)

:: Create directories
call :create_directories

:: Setup virtual environment
call :setup_virtual_environment
if errorlevel 1 goto :error

:: Load and validate environment
call :validate_environment
if errorlevel 1 goto :error

:: Check system dependencies
if "%SKIP_DEPS%"=="false" (
    call :check_system_dependencies
    if errorlevel 1 goto :error
)

:: Perform health checks
if "%SKIP_HEALTH%"=="false" (
    call :perform_health_checks
)

:: Setup models
if "%SKIP_MODELS%"=="false" (
    call :setup_models
)

:: Start the assistant
echo.
echo ===============================================
echo    Starting Sovereign Assistant
echo ===============================================
echo.

if "%DEV_MODE%"=="true" (
    echo [INFO] Starting in development mode...
    %PYTHON_CMD% -m assistant.main --dev
) else (
    echo [INFO] Starting in production mode...
    %PYTHON_CMD% -m assistant.main
)

if errorlevel 1 (
    echo [ERROR] Failed to start Sovereign Assistant
    goto :error
)

echo [INFO] Sovereign Assistant has stopped
goto :end

:: Function: Check Python installation
:check_python
echo === Checking Python Installation ===
echo.

:: Check if python is available
%PYTHON_CMD% --version >nul 2>&1
if errorlevel 1 (
    echo [WARN] Python not found with 'python' command, trying 'py'...
    set PYTHON_CMD=py
    py --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python not found. Please install Python 3.11 or later.
        echo Download from: https://www.python.org/downloads/
        exit /b 1
    )
)

:: Check Python version
for /f "tokens=2" %%a in ('%PYTHON_CMD% --version 2^>^&1') do set PYTHON_VERSION=%%a
echo [INFO] Found Python %PYTHON_VERSION%

:: Verify minimum version (3.11)
%PYTHON_CMD% -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.11+ required, found %PYTHON_VERSION%
    echo Please upgrade Python from: https://www.python.org/downloads/
    exit /b 1
)

echo [INFO] Python version check passed ✓
echo.
exit /b 0

:: Function: Create directories
:create_directories
echo === Creating Directories ===
echo.

if not exist "%LOG_DIR%" (
    mkdir "%LOG_DIR%"
    echo [INFO] Created directory: %LOG_DIR%
)

if not exist "%DATA_DIR%" (
    mkdir "%DATA_DIR%"
    echo [INFO] Created directory: %DATA_DIR%
)

if not exist "%MODELS_DIR%" (
    mkdir "%MODELS_DIR%"
    echo [INFO] Created directory: %MODELS_DIR%
)

if not exist "%CONFIG_DIR%" (
    mkdir "%CONFIG_DIR%"
    echo [INFO] Created directory: %CONFIG_DIR%
)

echo [INFO] Directory structure ready ✓
echo.
exit /b 0

:: Function: Setup virtual environment
:setup_virtual_environment
echo === Setting Up Virtual Environment ===
echo.

if not exist "%VENV_DIR%" (
    echo [INFO] Creating virtual environment...
    %PYTHON_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        exit /b 1
    )
) else (
    echo [INFO] Virtual environment already exists ✓
)

:: Activate virtual environment
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    exit /b 1
)

echo [INFO] Virtual environment activated ✓

:: Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo [WARN] Failed to upgrade pip, continuing anyway...
)

:: Install requirements
if exist "requirements.txt" (
    echo [INFO] Installing Python requirements...
    echo This may take several minutes for first-time setup...
    python -m pip install -r requirements.txt --quiet
    if errorlevel 1 (
        echo [ERROR] Failed to install requirements
        echo Try running: pip install -r requirements.txt
        exit /b 1
    )
    echo [INFO] Python requirements installed ✓
) else (
    echo [WARN] requirements.txt not found
)

echo.
exit /b 0

:: Function: Validate environment
:validate_environment
echo === Validating Environment ===
echo.

:: Load .env file if it exists
if exist ".env" (
    echo [INFO] Loading environment from .env file
    for /f "usebackq tokens=1,2 delims==" %%a in (".env") do (
        if not "%%a"=="" if not "%%a:~0,1%"=="#" (
            set "%%a=%%b"
        )
    )
)

:: Check required environment variables
set MISSING_REQUIRED=

if "%OPENAI_API_KEY%"=="" (
    set MISSING_REQUIRED=!MISSING_REQUIRED! OPENAI_API_KEY
) else (
    echo [INFO] OPENAI_API_KEY is set ✓
)

:: Check optional environment variables
set MISSING_OPTIONAL=

if "%ANTHROPIC_API_KEY%"=="" (
    set MISSING_OPTIONAL=!MISSING_OPTIONAL! ANTHROPIC_API_KEY
) else (
    echo [INFO] ANTHROPIC_API_KEY is set ✓
)

if "%OPENROUTER_API_KEY%"=="" (
    set MISSING_OPTIONAL=!MISSING_OPTIONAL! OPENROUTER_API_KEY
) else (
    echo [INFO] OPENROUTER_API_KEY is set ✓
)

if "%PERPLEXITY_API_KEY%"=="" (
    set MISSING_OPTIONAL=!MISSING_OPTIONAL! PERPLEXITY_API_KEY
) else (
    echo [INFO] PERPLEXITY_API_KEY is set ✓
)

if "%GOOGLE_API_KEY%"=="" (
    set MISSING_OPTIONAL=!MISSING_OPTIONAL! GOOGLE_API_KEY
) else (
    echo [INFO] GOOGLE_API_KEY is set ✓
)

if "%ELEVENLABS_API_KEY%"=="" (
    set MISSING_OPTIONAL=!MISSING_OPTIONAL! ELEVENLABS_API_KEY
) else (
    echo [INFO] ELEVENLABS_API_KEY is set ✓
)

:: Handle missing required variables
if not "%MISSING_REQUIRED%"=="" (
    echo [ERROR] Missing required environment variables:%MISSING_REQUIRED%
    echo.
    echo Please create a .env file or set these variables:
    for %%a in (%MISSING_REQUIRED%) do (
        echo   set %%a=your_api_key_here
    )
    echo.
    exit /b 1
)

:: Handle missing optional variables
if not "%MISSING_OPTIONAL%"=="" (
    echo [WARN] Optional environment variables not set:%MISSING_OPTIONAL%
    echo [INFO] Some features may be limited without these API keys
)

echo [INFO] Environment validation completed ✓
echo.
exit /b 0

:: Function: Check system dependencies
:check_system_dependencies
echo === Checking System Dependencies ===
echo.

:: Check Tesseract OCR
tesseract --version >nul 2>&1
if errorlevel 1 (
    echo [WARN] Tesseract OCR not found
    echo [INFO] Please install Tesseract from:
    echo   https://github.com/UB-Mannheim/tesseract/wiki
    echo   Or use chocolatey: choco install tesseract
    echo.
) else (
    for /f "tokens=2" %%a in ('tesseract --version 2^>^&1 ^| findstr "tesseract"') do set TESS_VERSION=%%a
    echo [INFO] Tesseract !TESS_VERSION! found ✓
)

:: Check if Visual C++ Redistributable is installed (needed for some Python packages)
reg query "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" >nul 2>&1
if errorlevel 1 (
    echo [WARN] Visual C++ Redistributable may be missing
    echo [INFO] If you encounter build errors, install from:
    echo   https://aka.ms/vs/17/release/vc_redist.x64.exe
    echo.
)

:: Check Windows version for audio support
for /f "tokens=4-5 delims=. " %%i in ('ver') do set VERSION=%%i.%%j
echo [INFO] Windows version: %VERSION%

echo [INFO] System dependency check completed
echo.
exit /b 0

:: Function: Perform health checks
:perform_health_checks
echo === Performing Health Checks ===
echo.

:: Check disk space (require at least 2GB free)
for /f "tokens=3" %%a in ('dir /-c "%SCRIPT_DIR%" ^| findstr "bytes free"') do set FREE_SPACE=%%a
set /a FREE_SPACE_GB=%FREE_SPACE% / 1073741824
if %FREE_SPACE_GB% LSS 2 (
    echo [WARN] Low disk space: %FREE_SPACE_GB%GB available, 2GB+ recommended
) else (
    echo [INFO] Disk space: %FREE_SPACE_GB%GB available ✓
)

:: Check memory
for /f "skip=1" %%p in ('wmic computersystem get TotalPhysicalMemory') do set TOTAL_MEMORY=%%p & goto :got_memory
:got_memory
set /a MEMORY_GB=%TOTAL_MEMORY% / 1073741824
if %MEMORY_GB% LSS 8 (
    echo [WARN] Total memory: %MEMORY_GB%GB, 8GB+ recommended
) else (
    echo [INFO] Total memory: %MEMORY_GB%GB ✓
)

:: Test audio devices
python -c "
import pyaudio
import sys
try:
    p = pyaudio.PyAudio()
    device_count = p.get_device_count()
    if device_count == 0:
        print('[WARN] No audio devices found')
        sys.exit(1)
    else:
        print(f'[INFO] Audio devices: {device_count} found ✓')
        sys.exit(0)
except Exception as e:
    print(f'[WARN] Audio system error: {e}')
    sys.exit(1)
finally:
    try:
        p.terminate()
    except:
        pass
" 2>nul

echo [INFO] Health checks completed
echo.
exit /b 0

:: Function: Setup models
:setup_models
echo === Setting Up Models ===
echo.

:: Check if models directory exists and has content
dir /b "%MODELS_DIR%" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Setting up offline models...
    python -c "
try:
    from assistant.offline_system import OfflineSystem
    offline = OfflineSystem()
    offline.download_models()
    print('[INFO] Offline models setup completed ✓')
except ImportError as e:
    print(f'[WARN] Offline system not available: {e}')
except Exception as e:
    print(f'[ERROR] Error setting up models: {e}')
"
) else (
    echo [INFO] Offline models already present ✓
)

echo.
exit /b 0

:error
echo.
echo [ERROR] Setup failed. Please check the error messages above.
echo.
pause
exit /b 1

:end
if not "%DEV_MODE%"=="true" (
    echo.
    echo Press any key to exit...
    pause >nul
)
exit /b 0 