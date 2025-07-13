#!/usr/bin/env python3
"""
Sovereign Voice Assistant - Windows Service Implementation
Provides Windows service management for the voice assistant
"""

import sys
import os
import time
import logging
import asyncio
import threading
from pathlib import Path

try:
    import win32serviceutil
    import win32service
    import win32event
    import servicemanager
    import win32api
    import win32con
except ImportError as e:
    print(f"Windows service modules not available: {e}")
    print("Install with: pip install pywin32")
    sys.exit(1)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class SovereignService(win32serviceutil.ServiceFramework):
    """Windows service wrapper for Sovereign Voice Assistant"""
    
    # Service configuration
    _svc_name_ = "SovereignAssistant"
    _svc_display_name_ = "Sovereign Voice Assistant"
    _svc_description_ = "AI-powered voice assistant with offline capabilities and memory management"
    _svc_deps_ = ["AudioSrv", "AudioEndpointBuilder"]  # Audio service dependencies
    
    # Service behavior
    _exe_name_ = sys.executable
    _exe_args_ = f'"{__file__}"'
    
    def __init__(self, args):
        """Initialize the service"""
        win32serviceutil.ServiceFramework.__init__(self, args)
        
        # Create stop event
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        
        # Service state
        self.assistant = None
        self.assistant_thread = None
        self.stop_requested = False
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info(f"Service initialized: {self._svc_name_}")
    
    def _setup_logging(self):
        """Setup logging for the service"""
        # Create logs directory
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        log_file = log_dir / "sovereign-service.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(self._svc_name_)
        self.logger.info(f"Logging initialized: {log_file}")
    
    def SvcStop(self):
        """Handle service stop request"""
        self.logger.info("Service stop requested")
        
        # Report stop pending
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        
        # Set stop flag
        self.stop_requested = True
        
        # Stop the assistant
        self._stop_assistant()
        
        # Signal stop event
        win32event.SetEvent(self.hWaitStop)
        
        self.logger.info("Service stop completed")
    
    def SvcPause(self):
        """Handle service pause request"""
        self.logger.info("Service pause requested")
        self.ReportServiceStatus(win32service.SERVICE_PAUSE_PENDING)
        
        # Pause the assistant
        if self.assistant and hasattr(self.assistant, 'pause'):
            self.assistant.pause()
        
        self.ReportServiceStatus(win32service.SERVICE_PAUSED)
        self.logger.info("Service paused")
    
    def SvcContinue(self):
        """Handle service continue request"""
        self.logger.info("Service continue requested")
        self.ReportServiceStatus(win32service.SERVICE_CONTINUE_PENDING)
        
        # Resume the assistant
        if self.assistant and hasattr(self.assistant, 'resume'):
            self.assistant.resume()
        
        self.ReportServiceStatus(win32service.SERVICE_RUNNING)
        self.logger.info("Service resumed")
    
    def SvcDoRun(self):
        """Main service execution"""
        try:
            # Log service start
            servicemanager.LogMsg(
                servicemanager.EVENTLOG_INFORMATION_TYPE,
                servicemanager.PYS_SERVICE_STARTED,
                (self._svc_name_, '')
            )
            self.logger.info(f"Service started: {self._svc_name_}")
            
            # Report running status
            self.ReportServiceStatus(win32service.SERVICE_RUNNING)
            
            # Start the assistant
            self._start_assistant()
            
            # Wait for stop event
            self._main_loop()
            
        except Exception as e:
            self.logger.error(f"Service error: {e}", exc_info=True)
            
            # Log error to Windows event log
            servicemanager.LogErrorMsg(f"Service error: {e}")
            
            # Report stopped status
            self.ReportServiceStatus(win32service.SERVICE_STOPPED)
        
        finally:
            self.logger.info("Service execution completed")
    
    def _start_assistant(self):
        """Start the voice assistant in a separate thread"""
        try:
            self.logger.info("Starting voice assistant...")
            
            # Import and initialize the assistant
            from assistant.main import SovereignAssistant
            
            # Create assistant instance
            self.assistant = SovereignAssistant()
            
            # Start in separate thread
            self.assistant_thread = threading.Thread(
                target=self._run_assistant,
                daemon=True
            )
            self.assistant_thread.start()
            
            self.logger.info("Voice assistant started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start assistant: {e}", exc_info=True)
            raise
    
    def _run_assistant(self):
        """Run the assistant in async context"""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the assistant
            loop.run_until_complete(self.assistant.run())
            
        except Exception as e:
            self.logger.error(f"Assistant runtime error: {e}", exc_info=True)
        finally:
            try:
                loop.close()
            except:
                pass
    
    def _stop_assistant(self):
        """Stop the voice assistant gracefully"""
        try:
            if self.assistant:
                self.logger.info("Stopping voice assistant...")
                
                # Signal assistant to stop
                if hasattr(self.assistant, 'stop'):
                    self.assistant.stop()
                
                # Wait for thread to finish (with timeout)
                if self.assistant_thread and self.assistant_thread.is_alive():
                    self.assistant_thread.join(timeout=30)
                    
                    if self.assistant_thread.is_alive():
                        self.logger.warning("Assistant thread did not stop gracefully")
                
                self.logger.info("Voice assistant stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping assistant: {e}", exc_info=True)
    
    def _main_loop(self):
        """Main service loop"""
        self.logger.info("Service main loop started")
        
        try:
            while not self.stop_requested:
                # Wait for stop event or timeout
                rc = win32event.WaitForSingleObject(self.hWaitStop, 5000)
                
                if rc == win32event.WAIT_OBJECT_0:
                    # Stop event was signaled
                    break
                elif rc == win32event.WAIT_TIMEOUT:
                    # Timeout - check if assistant is still running
                    if self.assistant_thread and not self.assistant_thread.is_alive():
                        self.logger.warning("Assistant thread stopped unexpectedly")
                        if not self.stop_requested:
                            # Restart assistant if not stopping
                            self._restart_assistant()
                else:
                    # Error
                    self.logger.error(f"Wait error: {rc}")
                    break
        
        except Exception as e:
            self.logger.error(f"Main loop error: {e}", exc_info=True)
        
        self.logger.info("Service main loop ended")
    
    def _restart_assistant(self):
        """Restart the assistant after failure"""
        try:
            self.logger.info("Restarting voice assistant...")
            
            # Stop current instance
            self._stop_assistant()
            
            # Wait a moment
            time.sleep(5)
            
            # Start new instance
            self._start_assistant()
            
            self.logger.info("Voice assistant restarted")
            
        except Exception as e:
            self.logger.error(f"Failed to restart assistant: {e}", exc_info=True)


class SovereignServiceManager:
    """Service management utilities"""
    
    @staticmethod
    def install():
        """Install the service"""
        try:
            win32serviceutil.InstallService(
                SovereignService._exe_name_,
                SovereignService._svc_name_,
                SovereignService._svc_display_name_,
                exeArgs=SovereignService._exe_args_,
                description=SovereignService._svc_description_,
                startType=win32service.SERVICE_AUTO_START,
                serviceDeps=SovereignService._svc_deps_
            )
            print(f"Service '{SovereignService._svc_display_name_}' installed successfully")
            return True
        except Exception as e:
            print(f"Failed to install service: {e}")
            return False
    
    @staticmethod
    def uninstall():
        """Uninstall the service"""
        try:
            win32serviceutil.RemoveService(SovereignService._svc_name_)
            print(f"Service '{SovereignService._svc_display_name_}' uninstalled successfully")
            return True
        except Exception as e:
            print(f"Failed to uninstall service: {e}")
            return False
    
    @staticmethod
    def start():
        """Start the service"""
        try:
            win32serviceutil.StartService(SovereignService._svc_name_)
            print(f"Service '{SovereignService._svc_display_name_}' started successfully")
            return True
        except Exception as e:
            print(f"Failed to start service: {e}")
            return False
    
    @staticmethod
    def stop():
        """Stop the service"""
        try:
            win32serviceutil.StopService(SovereignService._svc_name_)
            print(f"Service '{SovereignService._svc_display_name_}' stopped successfully")
            return True
        except Exception as e:
            print(f"Failed to stop service: {e}")
            return False
    
    @staticmethod
    def restart():
        """Restart the service"""
        try:
            win32serviceutil.RestartService(SovereignService._svc_name_)
            print(f"Service '{SovereignService._svc_display_name_}' restarted successfully")
            return True
        except Exception as e:
            print(f"Failed to restart service: {e}")
            return False
    
    @staticmethod
    def status():
        """Get service status"""
        try:
            status = win32serviceutil.QueryServiceStatus(SovereignService._svc_name_)
            state = status[1]
            
            states = {
                win32service.SERVICE_STOPPED: "Stopped",
                win32service.SERVICE_START_PENDING: "Starting",
                win32service.SERVICE_STOP_PENDING: "Stopping",
                win32service.SERVICE_RUNNING: "Running",
                win32service.SERVICE_CONTINUE_PENDING: "Resuming",
                win32service.SERVICE_PAUSE_PENDING: "Pausing",
                win32service.SERVICE_PAUSED: "Paused"
            }
            
            state_name = states.get(state, f"Unknown ({state})")
            print(f"Service '{SovereignService._svc_display_name_}' status: {state_name}")
            return state
        except Exception as e:
            print(f"Failed to get service status: {e}")
            return None


def main():
    """Main entry point"""
    if len(sys.argv) == 1:
        # Run as service
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(SovereignService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        # Handle command line arguments
        command = sys.argv[1].lower()
        
        if command == 'install':
            SovereignServiceManager.install()
        elif command == 'uninstall':
            SovereignServiceManager.uninstall()
        elif command == 'start':
            SovereignServiceManager.start()
        elif command == 'stop':
            SovereignServiceManager.stop()
        elif command == 'restart':
            SovereignServiceManager.restart()
        elif command == 'status':
            SovereignServiceManager.status()
        elif command == 'debug':
            # Run in debug mode
            service = SovereignService([])
            service.SvcDoRun()
        else:
            # Use standard service utility handling
            win32serviceutil.HandleCommandLine(SovereignService)


if __name__ == '__main__':
    main() 