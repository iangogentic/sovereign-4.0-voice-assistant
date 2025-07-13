#!/bin/bash
# services/scripts/service-manager.sh - Cross-platform service management
# Handles SystemD (Linux), launchd (macOS), and Windows service operations

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
SERVICE_NAME="sovereign-assistant"
SERVICE_DISPLAY_NAME="Sovereign Voice Assistant"

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

# Platform detection
detect_platform() {
    case "$(uname -s)" in
        Linux*)
            PLATFORM="linux"
            SERVICE_MANAGER="systemd"
            if ! command -v systemctl &> /dev/null; then
                SERVICE_MANAGER="sysv"
            fi
            ;;
        Darwin*)
            PLATFORM="macos"
            SERVICE_MANAGER="launchd"
            ;;
        CYGWIN*|MINGW*|MSYS*)
            PLATFORM="windows"
            SERVICE_MANAGER="windows"
            ;;
        *)
            PLATFORM="unknown"
            SERVICE_MANAGER="unknown"
            ;;
    esac
    
    log_info "Detected platform: $PLATFORM with service manager: $SERVICE_MANAGER"
}

# Check if running as appropriate user
check_permissions() {
    case $SERVICE_MANAGER in
        systemd|sysv)
            if [[ $EUID -ne 0 ]]; then
                log_error "This operation requires root privileges"
                log_info "Please run with sudo: sudo $0 $*"
                exit 1
            fi
            ;;
        launchd)
            # macOS can install user or system services
            if [[ "$1" == "install" && "${SYSTEM_SERVICE:-}" == "true" ]]; then
                if [[ $EUID -ne 0 ]]; then
                    log_error "System service installation requires root privileges"
                    log_info "Please run with sudo for system service, or without sudo for user service"
                    exit 1
                fi
            fi
            ;;
        windows)
            # Windows service operations typically require admin
            log_warn "Windows service operations may require administrator privileges"
            ;;
    esac
}

# SystemD service management
systemd_install() {
    log_info "Installing SystemD service..."
    
    # Create service user
    if ! id "sovereign" &>/dev/null; then
        log_info "Creating service user: sovereign"
        useradd --system --home /opt/sovereign --shell /bin/false sovereign
    fi
    
    # Create directories
    mkdir -p /opt/sovereign
    mkdir -p /var/log/sovereign
    mkdir -p /run/sovereign
    mkdir -p /etc/sovereign
    
    # Set permissions
    chown -R sovereign:sovereign /opt/sovereign
    chown sovereign:sovereign /var/log/sovereign
    chown sovereign:sovereign /run/sovereign
    
    # Copy application files
    cp -r "$PROJECT_ROOT"/* /opt/sovereign/
    chown -R sovereign:sovereign /opt/sovereign
    
    # Copy service file
    cp "$PROJECT_ROOT/services/systemd/sovereign-assistant.service" /etc/systemd/system/
    
    # Create service scripts
    mkdir -p /opt/sovereign/services/scripts
    cat > /opt/sovereign/services/scripts/pre-start.sh << 'EOF'
#!/bin/bash
# Pre-start script
echo "Starting Sovereign Voice Assistant pre-checks..."
# Check audio devices
if [ ! -d /dev/snd ]; then
    echo "Warning: No audio devices found"
fi
mkdir -p /run/sovereign
chown sovereign:sovereign /run/sovereign
EOF
    
    cat > /opt/sovereign/services/scripts/post-start.sh << 'EOF'
#!/bin/bash
# Post-start script
echo "Sovereign Voice Assistant started successfully"
# Create PID file if needed
echo "Service is running with PID: $(systemctl show --property MainPID --value sovereign-assistant)"
EOF
    
    cat > /opt/sovereign/services/scripts/post-stop.sh << 'EOF'
#!/bin/bash
# Post-stop script
echo "Sovereign Voice Assistant stopped"
# Cleanup PID file
rm -f /run/sovereign/sovereign-assistant.pid
EOF
    
    chmod +x /opt/sovereign/services/scripts/*.sh
    
    # Reload systemd and enable service
    systemctl daemon-reload
    systemctl enable sovereign-assistant.service
    
    log_info "SystemD service installed successfully"
    log_info "Use 'systemctl start sovereign-assistant' to start the service"
}

systemd_uninstall() {
    log_info "Uninstalling SystemD service..."
    
    # Stop and disable service
    systemctl stop sovereign-assistant.service 2>/dev/null || true
    systemctl disable sovereign-assistant.service 2>/dev/null || true
    
    # Remove service file
    rm -f /etc/systemd/system/sovereign-assistant.service
    
    # Reload systemd
    systemctl daemon-reload
    
    log_info "SystemD service uninstalled successfully"
    log_warn "Application files remain in /opt/sovereign (remove manually if desired)"
}

systemd_start() {
    log_info "Starting SystemD service..."
    systemctl start sovereign-assistant.service
    log_info "Service started successfully"
}

systemd_stop() {
    log_info "Stopping SystemD service..."
    systemctl stop sovereign-assistant.service
    log_info "Service stopped successfully"
}

systemd_restart() {
    log_info "Restarting SystemD service..."
    systemctl restart sovereign-assistant.service
    log_info "Service restarted successfully"
}

systemd_status() {
    log_info "Checking SystemD service status..."
    systemctl status sovereign-assistant.service --no-pager -l
}

systemd_logs() {
    log_info "Showing SystemD service logs..."
    journalctl -u sovereign-assistant.service --no-pager -l ${1:--f}
}

# macOS launchd service management
launchd_install() {
    log_info "Installing launchd service..."
    
    # Determine service location
    if [[ "${SYSTEM_SERVICE:-}" == "true" ]]; then
        PLIST_DIR="/Library/LaunchDaemons"
        SERVICE_LABEL="com.sovereign.assistant"
    else
        PLIST_DIR="$HOME/Library/LaunchAgents"
        SERVICE_LABEL="com.sovereign.assistant.user"
    fi
    
    # Create plist file
    cat > "$PLIST_DIR/com.sovereign.assistant.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$SERVICE_LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PROJECT_ROOT/venv/bin/python</string>
        <string>-m</string>
        <string>assistant.main</string>
        <string>--service</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$PROJECT_ROOT</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$PROJECT_ROOT/logs/sovereign-service.log</string>
    <key>StandardErrorPath</key>
    <string>$PROJECT_ROOT/logs/sovereign-service-error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>SOVEREIGN_ENV</key>
        <string>production</string>
        <key>LOG_LEVEL</key>
        <string>INFO</string>
    </dict>
</dict>
</plist>
EOF
    
    # Set permissions
    chmod 644 "$PLIST_DIR/com.sovereign.assistant.plist"
    
    # Load service
    if [[ "${SYSTEM_SERVICE:-}" == "true" ]]; then
        launchctl load -w "$PLIST_DIR/com.sovereign.assistant.plist"
    else
        launchctl load -w "$PLIST_DIR/com.sovereign.assistant.plist"
    fi
    
    log_info "launchd service installed successfully"
}

launchd_uninstall() {
    log_info "Uninstalling launchd service..."
    
    # Determine service location
    if [[ "${SYSTEM_SERVICE:-}" == "true" ]]; then
        PLIST_DIR="/Library/LaunchDaemons"
    else
        PLIST_DIR="$HOME/Library/LaunchAgents"
    fi
    
    # Unload and remove service
    launchctl unload -w "$PLIST_DIR/com.sovereign.assistant.plist" 2>/dev/null || true
    rm -f "$PLIST_DIR/com.sovereign.assistant.plist"
    
    log_info "launchd service uninstalled successfully"
}

launchd_start() {
    log_info "Starting launchd service..."
    launchctl start com.sovereign.assistant
    log_info "Service started successfully"
}

launchd_stop() {
    log_info "Stopping launchd service..."
    launchctl stop com.sovereign.assistant
    log_info "Service stopped successfully"
}

launchd_restart() {
    launchd_stop
    sleep 2
    launchd_start
}

launchd_status() {
    log_info "Checking launchd service status..."
    launchctl list | grep sovereign || log_warn "Service not found or not running"
}

launchd_logs() {
    log_info "Showing service logs..."
    if [[ -f "$PROJECT_ROOT/logs/sovereign-service.log" ]]; then
        tail ${1:--f} "$PROJECT_ROOT/logs/sovereign-service.log"
    else
        log_warn "Log file not found: $PROJECT_ROOT/logs/sovereign-service.log"
    fi
}

# Windows service management
windows_install() {
    log_info "Installing Windows service..."
    
    # Use Python to install service
    cd "$PROJECT_ROOT"
    python services/windows/sovereign_service.py install
    
    log_info "Windows service installed successfully"
}

windows_uninstall() {
    log_info "Uninstalling Windows service..."
    
    cd "$PROJECT_ROOT"
    python services/windows/sovereign_service.py uninstall
    
    log_info "Windows service uninstalled successfully"
}

windows_start() {
    log_info "Starting Windows service..."
    
    cd "$PROJECT_ROOT"
    python services/windows/sovereign_service.py start
}

windows_stop() {
    log_info "Stopping Windows service..."
    
    cd "$PROJECT_ROOT"
    python services/windows/sovereign_service.py stop
}

windows_restart() {
    log_info "Restarting Windows service..."
    
    cd "$PROJECT_ROOT"
    python services/windows/sovereign_service.py restart
}

windows_status() {
    log_info "Checking Windows service status..."
    
    cd "$PROJECT_ROOT"
    python services/windows/sovereign_service.py status
}

windows_logs() {
    log_info "Showing Windows service logs..."
    if [[ -f "$PROJECT_ROOT/logs/sovereign-service.log" ]]; then
        tail ${1:--f} "$PROJECT_ROOT/logs/sovereign-service.log"
    else
        log_warn "Log file not found: $PROJECT_ROOT/logs/sovereign-service.log"
    fi
}

# Generic service operations
service_install() {
    case $SERVICE_MANAGER in
        systemd) systemd_install ;;
        launchd) launchd_install ;;
        windows) windows_install ;;
        *) log_error "Unsupported service manager: $SERVICE_MANAGER" ;;
    esac
}

service_uninstall() {
    case $SERVICE_MANAGER in
        systemd) systemd_uninstall ;;
        launchd) launchd_uninstall ;;
        windows) windows_uninstall ;;
        *) log_error "Unsupported service manager: $SERVICE_MANAGER" ;;
    esac
}

service_start() {
    case $SERVICE_MANAGER in
        systemd) systemd_start ;;
        launchd) launchd_start ;;
        windows) windows_start ;;
        *) log_error "Unsupported service manager: $SERVICE_MANAGER" ;;
    esac
}

service_stop() {
    case $SERVICE_MANAGER in
        systemd) systemd_stop ;;
        launchd) launchd_stop ;;
        windows) windows_stop ;;
        *) log_error "Unsupported service manager: $SERVICE_MANAGER" ;;
    esac
}

service_restart() {
    case $SERVICE_MANAGER in
        systemd) systemd_restart ;;
        launchd) launchd_restart ;;
        windows) windows_restart ;;
        *) log_error "Unsupported service manager: $SERVICE_MANAGER" ;;
    esac
}

service_status() {
    case $SERVICE_MANAGER in
        systemd) systemd_status ;;
        launchd) launchd_status ;;
        windows) windows_status ;;
        *) log_error "Unsupported service manager: $SERVICE_MANAGER" ;;
    esac
}

service_logs() {
    case $SERVICE_MANAGER in
        systemd) systemd_logs "$@" ;;
        launchd) launchd_logs "$@" ;;
        windows) windows_logs "$@" ;;
        *) log_error "Unsupported service manager: $SERVICE_MANAGER" ;;
    esac
}

# Show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] <command>

Service management for Sovereign Voice Assistant

COMMANDS:
    install     Install the service
    uninstall   Uninstall the service
    start       Start the service
    stop        Stop the service
    restart     Restart the service
    status      Show service status
    logs        Show service logs (use -f for follow)

OPTIONS:
    --system    Install as system service (macOS only)
    --user      Install as user service (macOS only, default)
    --debug     Enable debug logging
    --help      Show this help message

EXAMPLES:
    $0 install                  Install service
    $0 start                    Start service
    $0 logs -f                  Follow service logs
    $0 --system install         Install system service (macOS)

PLATFORM SUPPORT:
    Linux       SystemD service
    macOS       launchd service (user or system)
    Windows     Windows service

EOF
}

# Main execution
main() {
    local command=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --system)
                SYSTEM_SERVICE="true"
                shift
                ;;
            --user)
                SYSTEM_SERVICE="false"
                shift
                ;;
            --debug)
                DEBUG="true"
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            -*)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
            *)
                command="$1"
                shift
                break
                ;;
        esac
    done
    
    # Validate command
    if [[ -z "$command" ]]; then
        log_error "No command specified"
        show_usage
        exit 1
    fi
    
    case $command in
        install|uninstall|start|stop|restart|status|logs)
            ;;
        *)
            log_error "Invalid command: $command"
            show_usage
            exit 1
            ;;
    esac
    
    # Platform detection
    detect_platform
    
    # Permission check
    check_permissions "$command"
    
    # Execute command
    case $command in
        install) service_install ;;
        uninstall) service_uninstall ;;
        start) service_start ;;
        stop) service_stop ;;
        restart) service_restart ;;
        status) service_status ;;
        logs) service_logs "$@" ;;
    esac
    
    log_info "Operation completed successfully"
}

# Run main function
main "$@" 