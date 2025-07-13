#!/usr/bin/env python3
"""
Sovereign Voice Assistant - Debug Server
Provides web interface for development monitoring and control
"""

import asyncio
import json
import logging
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse
import os
import gc


class DebugHandler(BaseHTTPRequestHandler):
    """HTTP handler for debug server requests"""
    
    def __init__(self, *args, dev_manager=None, **kwargs):
        self.dev_manager = dev_manager
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        query = parse_qs(parsed_path.query)
        
        try:
            if path == '/':
                self._serve_dashboard()
            elif path == '/api/stats':
                self._serve_stats()
            elif path == '/api/modules':
                self._serve_modules()
            elif path == '/api/logs':
                self._serve_logs(query)
            elif path == '/api/memory':
                self._serve_memory_info()
            elif path == '/api/gc':
                self._trigger_gc()
            elif path.startswith('/static/'):
                self._serve_static(path)
            else:
                self._send_404()
                
        except Exception as e:
            logging.error(f"Debug server error: {e}")
            self._send_error(str(e))
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8')) if post_data else {}
            
            if path == '/api/reload':
                self._trigger_reload(data)
            elif path == '/api/config':
                self._update_config(data)
            elif path == '/api/debug':
                self._trigger_debug_action(data)
            else:
                self._send_404()
                
        except Exception as e:
            logging.error(f"Debug server POST error: {e}")
            self._send_error(str(e))
    
    def _serve_dashboard(self):
        """Serve main dashboard HTML"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Sovereign Assistant - Debug Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: #333; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .section {{ background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .stat-box {{ background: #f8f9fa; padding: 15px; border-radius: 6px; text-align: center; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
        .stat-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
        .btn {{ background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; margin: 5px; }}
        .btn:hover {{ background: #0056b3; }}
        .btn-danger {{ background: #dc3545; }}
        .btn-danger:hover {{ background: #c82333; }}
        .btn-success {{ background: #28a745; }}
        .btn-success:hover {{ background: #218838; }}
        .log-area {{ background: #1e1e1e; color: #fff; padding: 15px; font-family: monospace; height: 300px; overflow-y: auto; border-radius: 4px; }}
        .status-enabled {{ color: #28a745; font-weight: bold; }}
        .status-disabled {{ color: #dc3545; font-weight: bold; }}
        #autoRefresh {{ margin-left: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Sovereign Assistant - Debug Dashboard</h1>
            <p>Development Mode Monitoring & Control</p>
            <label>
                <input type="checkbox" id="autoRefresh" checked> Auto-refresh (5s)
            </label>
        </div>
        
        <div class="section">
            <h2>System Status</h2>
            <div class="stats-grid" id="statsGrid">
                <!-- Stats will be loaded here -->
            </div>
        </div>
        
        <div class="section">
            <h2>Quick Actions</h2>
            <button class="btn" onclick="reloadModules()">🔄 Reload All Modules</button>
            <button class="btn btn-success" onclick="triggerGC()">🗑️ Garbage Collect</button>
            <button class="btn" onclick="memorySnapshot()">📊 Memory Snapshot</button>
            <button class="btn btn-danger" onclick="saveDebugReport()">💾 Save Debug Report</button>
        </div>
        
        <div class="section">
            <h2>Recent Logs</h2>
            <div class="log-area" id="logArea">
                <!-- Logs will be loaded here -->
            </div>
        </div>
        
        <div class="section">
            <h2>Module Information</h2>
            <div id="moduleInfo">
                <!-- Module info will be loaded here -->
            </div>
        </div>
    </div>
    
    <script>
        let autoRefreshInterval;
        
        function updateStats() {{
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {{
                    const grid = document.getElementById('statsGrid');
                    grid.innerHTML = `
                        <div class="stat-box">
                            <div class="stat-value">${{data.enabled ? '<span class="status-enabled">ON</span>' : '<span class="status-disabled">OFF</span>'}}</div>
                            <div class="stat-label">Dev Mode</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">${{Math.round(data.uptime_seconds / 60)}}m</div>
                            <div class="stat-label">Uptime</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">${{data.reload_count}}</div>
                            <div class="stat-label">Reloads</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">${{data.memory_usage_mb ? data.memory_usage_mb.toFixed(1) + 'MB' : 'N/A'}}</div>
                            <div class="stat-label">Memory</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">${{data.cpu_usage_percent ? data.cpu_usage_percent.toFixed(1) + '%' : 'N/A'}}</div>
                            <div class="stat-label">CPU</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">${{data.debug_calls}}</div>
                            <div class="stat-label">Debug Calls</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">${{data.errors_count}}</div>
                            <div class="stat-label">Errors</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">${{data.profiling_active ? '<span class="status-enabled">ON</span>' : '<span class="status-disabled">OFF</span>'}}</div>
                            <div class="stat-label">Profiling</div>
                        </div>
                    `;
                }})
                .catch(err => console.error('Failed to update stats:', err));
        }}
        
        function updateLogs() {{
            fetch('/api/logs?lines=20')
                .then(response => response.text())
                .then(data => {{
                    document.getElementById('logArea').textContent = data;
                }})
                .catch(err => console.error('Failed to update logs:', err));
        }}
        
        function updateModules() {{
            fetch('/api/modules')
                .then(response => response.json())
                .then(data => {{
                    const moduleInfo = document.getElementById('moduleInfo');
                    const recentModules = data.recent_modules.slice(0, 10);
                    moduleInfo.innerHTML = `
                        <p><strong>Total modules:</strong> ${{data.total_modules}}</p>
                        <p><strong>Recent modules:</strong></p>
                        <ul>${{recentModules.map(m => `<li>${{m}}</li>`).join('')}}</ul>
                    `;
                }})
                .catch(err => console.error('Failed to update modules:', err));
        }}
        
        function reloadModules() {{
            fetch('/api/reload', {{method: 'POST'}})
                .then(response => response.json())
                .then(data => alert('Reload triggered: ' + data.message))
                .catch(err => alert('Reload failed: ' + err));
        }}
        
        function triggerGC() {{
            fetch('/api/gc')
                .then(response => response.json())
                .then(data => alert('Garbage collection: ' + data.collected + ' objects collected'))
                .catch(err => alert('GC failed: ' + err));
        }}
        
        function memorySnapshot() {{
            fetch('/api/debug', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{'action': 'memory_snapshot', 'label': 'dashboard_trigger'}})
            }})
            .then(response => response.json())
            .then(data => alert('Memory snapshot taken: ' + data.message))
            .catch(err => alert('Memory snapshot failed: ' + err));
        }}
        
        function saveDebugReport() {{
            fetch('/api/debug', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{'action': 'save_report'}})
            }})
            .then(response => response.json())
            .then(data => alert('Debug report saved: ' + data.message))
            .catch(err => alert('Save failed: ' + err));
        }}
        
        function toggleAutoRefresh() {{
            const checkbox = document.getElementById('autoRefresh');
            if (checkbox.checked) {{
                autoRefreshInterval = setInterval(() => {{
                    updateStats();
                    updateLogs();
                    updateModules();
                }}, 5000);
            }} else {{
                clearInterval(autoRefreshInterval);
            }}
        }}
        
        // Initial load
        updateStats();
        updateLogs();
        updateModules();
        
        // Setup auto-refresh
        document.getElementById('autoRefresh').addEventListener('change', toggleAutoRefresh);
        toggleAutoRefresh();
    </script>
</body>
</html>
        """
        
        self._send_response(html, 'text/html')
    
    def _serve_stats(self):
        """Serve development statistics"""
        if self.dev_manager:
            stats = self.dev_manager.get_stats()
        else:
            stats = {'error': 'Dev manager not available'}
        
        self._send_json(stats)
    
    def _serve_modules(self):
        """Serve module information"""
        import sys
        modules = list(sys.modules.keys())
        
        data = {
            'total_modules': len(modules),
            'recent_modules': modules[-20:],  # Last 20 modules
            'python_modules': [m for m in modules if not m.startswith('_')],
        }
        
        self._send_json(data)
    
    def _serve_logs(self, query):
        """Serve recent log entries"""
        try:
            lines = int(query.get('lines', ['50'])[0])
            log_file = Path.cwd() / "logs" / "dev_mode.log"
            
            if log_file.exists():
                with open(log_file, 'r') as f:
                    all_lines = f.readlines()
                    recent_lines = all_lines[-lines:]
                    content = ''.join(recent_lines)
            else:
                content = "Log file not found"
                
        except Exception as e:
            content = f"Error reading logs: {e}"
        
        self._send_response(content, 'text/plain')
    
    def _serve_memory_info(self):
        """Serve memory information"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            data = {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent(),
                'gc_counts': gc.get_count()
            }
        except Exception as e:
            data = {'error': str(e)}
        
        self._send_json(data)
    
    def _trigger_gc(self):
        """Trigger garbage collection"""
        try:
            collected = gc.collect()
            data = {'collected': collected, 'counts': gc.get_count()}
        except Exception as e:
            data = {'error': str(e)}
        
        self._send_json(data)
    
    def _trigger_reload(self, data):
        """Trigger module reload"""
        try:
            if self.dev_manager:
                # Trigger a manual reload
                result = "Manual reload triggered"
                self.dev_manager.stats.reload_count += 1
            else:
                result = "Dev manager not available"
            
            self._send_json({'message': result})
        except Exception as e:
            self._send_json({'error': str(e)})
    
    def _trigger_debug_action(self, data):
        """Trigger debug actions"""
        try:
            action = data.get('action')
            
            if action == 'memory_snapshot':
                if self.dev_manager:
                    label = data.get('label', 'api_trigger')
                    self.dev_manager.memory_snapshot(label)
                    message = f"Memory snapshot taken: {label}"
                else:
                    message = "Dev manager not available"
            
            elif action == 'save_report':
                if self.dev_manager:
                    report_file = self.dev_manager.save_debug_report()
                    message = f"Debug report saved: {report_file}"
                else:
                    message = "Dev manager not available"
            
            else:
                message = f"Unknown action: {action}"
            
            self._send_json({'message': message})
            
        except Exception as e:
            self._send_json({'error': str(e)})
    
    def _send_response(self, content, content_type):
        """Send HTTP response"""
        self.send_response(200)
        self.send_header('Content-type', content_type)
        self.send_header('Content-length', len(content.encode('utf-8')))
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))
    
    def _send_json(self, data):
        """Send JSON response"""
        content = json.dumps(data, indent=2, default=str)
        self._send_response(content, 'application/json')
    
    def _send_404(self):
        """Send 404 response"""
        self.send_response(404)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'404 Not Found')
    
    def _send_error(self, error_msg):
        """Send error response"""
        self.send_response(500)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        error_data = json.dumps({'error': error_msg})
        self.wfile.write(error_data.encode('utf-8'))
    
    def log_message(self, format, *args):
        """Override to reduce log noise"""
        pass


class DebugServer:
    """Simple debug server for development monitoring"""
    
    def __init__(self, dev_manager, host: str = 'localhost', port: int = 8888):
        self.dev_manager = dev_manager
        self.host = host
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None
        self.running = False
        
        self.logger = logging.getLogger("debug_server")
    
    def start(self):
        """Start the debug server"""
        if self.running:
            self.logger.warning("Debug server already running")
            return
        
        try:
            # Create handler with dev_manager reference
            def handler(*args, **kwargs):
                return DebugHandler(*args, dev_manager=self.dev_manager, **kwargs)
            
            self.server = HTTPServer((self.host, self.port), handler)
            
            def serve_forever():
                self.logger.info(f"Debug server starting on http://{self.host}:{self.port}")
                self.server.serve_forever()
            
            self.server_thread = threading.Thread(target=serve_forever, daemon=True)
            self.server_thread.start()
            
            self.running = True
            self.logger.info(f"Debug server started on http://{self.host}:{self.port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start debug server: {e}")
            raise
    
    def stop(self):
        """Stop the debug server"""
        if not self.running:
            return
        
        try:
            if self.server:
                self.server.shutdown()
                self.server.server_close()
            
            if self.server_thread:
                self.server_thread.join(timeout=5)
            
            self.running = False
            self.logger.info("Debug server stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping debug server: {e}")
    
    def is_running(self) -> bool:
        """Check if server is running"""
        return self.running 