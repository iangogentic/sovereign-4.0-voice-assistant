@echo off
:: services/scripts/service-manager.bat - Windows service management
:: Provides Windows service operations for Sovereign Voice Assistant

setlocal enabledelayedexpansion

:: Configuration
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%\..\..\
set SERVICE_NAME=SovereignAssistant
set SERVICE_DISPLAY_NAME=Sovereign Voice Assistant

:: Initialize variables
set COMMAND=
set DEBUG=false

:: Header
echo.
echo ===============================================
echo    Sovereign Voice Assistant Service Manager
echo ===============================================
echo.

:: Parse command line arguments
:parse_args
if "%~1"=="" goto :validate_command
if /i "%~1"=="--debug" (
    set DEBUG=true
    shift
    goto :parse_args
)
if /i "%~1"=="--help" goto :show_help
if /i "%~1"=="-h" goto :show_help

:: Check if this is a command
if /i "%~1"=="install" (
    set COMMAND=install
    shift
    goto :parse_args
)
if /i "%~1"=="uninstall" (
    set COMMAND=uninstall
    shift
    goto :parse_args
)
if /i "%~1"=="start" (
    set COMMAND=start
    shift
    goto :parse_args
)
if /i "%~1"=="stop" (
    set COMMAND=stop
    shift
    goto :parse_args
)
if /i "%~1"=="restart" (
    set COMMAND=restart
    shift
    goto :parse_args
)
if /i "%~1"=="status" (
    set COMMAND=status
    shift
    goto :parse_args
)
if /i "%~1"=="logs" (
    set COMMAND=logs
    shift
    goto :parse_args
)

echo [ERROR] Unknown option: %~1
goto :show_help

:validate_command
if "%COMMAND%"=="" (
    echo [ERROR] No command specified
    goto :show_help
)

:: Check if running as administrator
call :check_admin
if errorlevel 1 (
    echo [WARN] Some operations may require administrator privileges
    echo.
)

:: Execute command
if /i "%COMMAND%"=="install" goto :install_service
if /i "%COMMAND%"=="uninstall" goto :uninstall_service
if /i "%COMMAND%"=="start" goto :start_service
if /i "%COMMAND%"=="stop" goto :stop_service
if /i "%COMMAND%"=="restart" goto :restart_service
if /i "%COMMAND%"=="status" goto :status_service
if /i "%COMMAND%"=="logs" goto :logs_service

echo [ERROR] Invalid command: %COMMAND%
goto :show_help

:check_admin
:: Check if running as administrator
net session >nul 2>&1
exit /b %errorlevel%

:install_service
echo [INFO] Installing Windows service...
echo.

:: Change to project directory
cd /d "%PROJECT_ROOT%"

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please ensure Python is installed and in PATH.
    goto :error
)

:: Check if pywin32 is installed
python -c "import win32serviceutil" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pywin32 not installed. Installing...
    pip install pywin32
    if errorlevel 1 (
        echo [ERROR] Failed to install pywin32
        goto :error
    )
)

:: Install the service
echo [INFO] Installing service: %SERVICE_DISPLAY_NAME%
python services\windows\sovereign_service.py install
if errorlevel 1 (
    echo [ERROR] Failed to install service
    goto :error
)

echo [INFO] Service installed successfully
echo [INFO] Use 'sc start %SERVICE_NAME%' or this script to start the service
goto :end

:uninstall_service
echo [INFO] Uninstalling Windows service...
echo.

cd /d "%PROJECT_ROOT%"

:: Stop service first
call :stop_service_internal

:: Uninstall the service
echo [INFO] Uninstalling service: %SERVICE_DISPLAY_NAME%
python services\windows\sovereign_service.py uninstall
if errorlevel 1 (
    echo [ERROR] Failed to uninstall service
    goto :error
)

echo [INFO] Service uninstalled successfully
goto :end

:start_service
echo [INFO] Starting Windows service...
echo.

cd /d "%PROJECT_ROOT%"

python services\windows\sovereign_service.py start
if errorlevel 1 (
    echo [ERROR] Failed to start service
    goto :error
)

echo [INFO] Service started successfully
goto :end

:stop_service
echo [INFO] Stopping Windows service...
echo.
call :stop_service_internal
goto :end

:stop_service_internal
cd /d "%PROJECT_ROOT%"

python services\windows\sovereign_service.py stop
if errorlevel 1 (
    echo [ERROR] Failed to stop service
    goto :error
)

echo [INFO] Service stopped successfully
exit /b 0

:restart_service
echo [INFO] Restarting Windows service...
echo.

cd /d "%PROJECT_ROOT%"

python services\windows\sovereign_service.py restart
if errorlevel 1 (
    echo [ERROR] Failed to restart service
    goto :error
)

echo [INFO] Service restarted successfully
goto :end

:status_service
echo [INFO] Checking Windows service status...
echo.

cd /d "%PROJECT_ROOT%"

python services\windows\sovereign_service.py status
if errorlevel 1 (
    echo [ERROR] Failed to get service status
    goto :error
)

goto :end

:logs_service
echo [INFO] Showing service logs...
echo.

:: Check if log file exists
if exist "%PROJECT_ROOT%\logs\sovereign-service.log" (
    echo [INFO] Recent log entries:
    echo.
    type "%PROJECT_ROOT%\logs\sovereign-service.log" | more
) else (
    echo [WARN] Log file not found: %PROJECT_ROOT%\logs\sovereign-service.log
)

:: Also check Windows Event Log
echo.
echo [INFO] Checking Windows Event Log for service events...
powershell -Command "Get-EventLog -LogName Application -Source '*Sovereign*' -Newest 10 -ErrorAction SilentlyContinue | Format-Table TimeGenerated, EntryType, Message -Wrap"

goto :end

:show_help
echo Usage: %~nx0 [OPTIONS] COMMAND
echo.
echo Service management for Sovereign Voice Assistant
echo.
echo COMMANDS:
echo     install     Install the Windows service
echo     uninstall   Uninstall the Windows service
echo     start       Start the service
echo     stop        Stop the service
echo     restart     Restart the service
echo     status      Show service status
echo     logs        Show service logs
echo.
echo OPTIONS:
echo     --debug     Enable debug logging
echo     --help      Show this help message
echo.
echo EXAMPLES:
echo     %~nx0 install                Install service
echo     %~nx0 start                  Start service
echo     %~nx0 logs                   Show service logs
echo.
echo NOTES:
echo     - Some operations require administrator privileges
echo     - Service will be installed as '%SERVICE_DISPLAY_NAME%'
echo     - Service name for sc commands: '%SERVICE_NAME%'
echo.
echo ALTERNATIVE COMMANDS:
echo     sc start %SERVICE_NAME%         Start service (Windows built-in)
echo     sc stop %SERVICE_NAME%          Stop service (Windows built-in)
echo     sc query %SERVICE_NAME%         Query service status (Windows built-in)
echo.
goto :end

:error
echo.
echo [ERROR] Operation failed. Please check the error messages above.
echo.
pause
exit /b 1

:end
echo.
echo [INFO] Operation completed
if not "%DEBUG%"=="true" (
    echo.
    echo Press any key to exit...
    pause >nul
)
exit /b 0 