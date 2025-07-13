"""
Real-Time Performance Dashboard Module

This module provides real-time visualization and reporting for the VoiceAssistantPipeline:
- Console-based performance display
- Real-time latency monitoring
- Alert status and history
- Performance trend analysis
- Export capabilities
"""

import time
import threading
import os
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

from .monitoring import PerformanceMonitor, PipelineStage, AlertLevel, get_monitor


class ConsoleDashboard:
    """Console-based real-time performance dashboard"""
    
    def __init__(self, monitor: Optional[PerformanceMonitor] = None, refresh_interval: float = 1.0):
        self.monitor = monitor or get_monitor()
        self.refresh_interval = refresh_interval
        self.is_running = False
        self.dashboard_thread = None
        self._should_stop = threading.Event()
        
        # Terminal control
        self.clear_command = 'cls' if os.name == 'nt' else 'clear'
        
    def start(self):
        """Start the dashboard display"""
        if self.is_running:
            return
            
        self.is_running = True
        self._should_stop.clear()
        self.dashboard_thread = threading.Thread(target=self._dashboard_loop, daemon=True)
        self.dashboard_thread.start()
        
    def stop(self):
        """Stop the dashboard display"""
        if not self.is_running:
            return
            
        self.is_running = False
        self._should_stop.set()
        
        if self.dashboard_thread and self.dashboard_thread.is_alive():
            self.dashboard_thread.join(timeout=2.0)
    
    def _dashboard_loop(self):
        """Main dashboard display loop"""
        try:
            while self.is_running and not self._should_stop.is_set():
                self._refresh_display()
                self._should_stop.wait(self.refresh_interval)
        except KeyboardInterrupt:
            pass
        finally:
            self.is_running = False
    
    def _refresh_display(self):
        """Refresh the dashboard display"""
        try:
            # Clear screen
            os.system(self.clear_command)
            
            # Get current data
            summary = self.monitor.get_summary()
            metrics = self.monitor.get_metrics()
            recent_alerts = self.monitor.get_recent_alerts(limit=5)
            
            # Display header
            print("=" * 80)
            print("🎙️  VOICE ASSISTANT PERFORMANCE DASHBOARD")
            print("=" * 80)
            print(f"📊 Status: {self._format_status(summary['performance_status'])}")
            print(f"⏱️  Average Latency: {summary['average_latency_ms']:.1f}ms (Target: {summary['target_latency_ms']:.0f}ms)")
            print(f"📈 Success Rate: {summary['success_rate']:.1f}%")
            print(f"🔄 Total Sessions: {summary['total_sessions']}")
            print(f"⏰ Uptime: {self._format_uptime(summary['uptime_seconds'])}")
            print()
            
            # Display stage metrics
            print("🎯 PIPELINE STAGE PERFORMANCE")
            print("-" * 80)
            print(f"{'Stage':<20} {'Calls':<8} {'Success':<8} {'Avg (ms)':<10} {'P95 (ms)':<10} {'Status':<10}")
            print("-" * 80)
            
            stage_order = [
                PipelineStage.AUDIO_CAPTURE,
                PipelineStage.STT_PROCESSING,
                PipelineStage.TTS_GENERATION,
                PipelineStage.AUDIO_PLAYBACK,
                PipelineStage.TOTAL_ROUND_TRIP
            ]
            
            for stage in stage_order:
                if stage.value in metrics:
                    metric = metrics[stage.value]
                    status = self._get_stage_status(stage, metric['recent_average'])
                    
                    print(f"{self._format_stage_name(stage):<20} "
                          f"{metric['total_calls']:<8} "
                          f"{metric['success_rate']:>6.1f}% "
                          f"{metric['recent_average']:>8.1f} "
                          f"{metric['p95_duration']:>8.1f} "
                          f"{self._format_status(status):<10}")
            
            print()
            
            # Display recent alerts
            if recent_alerts:
                print("🚨 RECENT ALERTS")
                print("-" * 80)
                for alert in reversed(recent_alerts[-5:]):  # Show last 5, most recent first
                    timestamp = datetime.fromtimestamp(alert['timestamp']).strftime('%H:%M:%S')
                    level_icon = self._get_alert_icon(alert['level'])
                    print(f"{timestamp} {level_icon} {alert['message']}")
                print()
            
            # Display alert summary
            alert_counts = summary['recent_alerts']
            if any(alert_counts.values()):
                print("📋 ALERT SUMMARY (Last 5 minutes)")
                print("-" * 80)
                print(f"🔴 Critical: {alert_counts['critical']}")
                print(f"🟡 Warning: {alert_counts['warning']}")
                print(f"🔵 Info: {alert_counts['info']}")
                print()
            
            # Display controls
            print("⌨️  CONTROLS: Press Ctrl+C to stop dashboard")
            print("=" * 80)
            
        except Exception as e:
            print(f"Dashboard error: {e}")
            time.sleep(1)
    
    def _format_status(self, status: str) -> str:
        """Format status with color indicators"""
        status_map = {
            'good': '🟢 Good',
            'warning': '🟡 Warning',
            'critical': '🔴 Critical'
        }
        return status_map.get(status, f'❓ {status.title()}')
    
    def _format_stage_name(self, stage: PipelineStage) -> str:
        """Format stage name for display"""
        name_map = {
            PipelineStage.AUDIO_CAPTURE: '🎤 Audio Capture',
            PipelineStage.STT_PROCESSING: '🗣️  STT Processing',
            PipelineStage.TTS_GENERATION: '🔊 TTS Generation',
            PipelineStage.AUDIO_PLAYBACK: '📢 Audio Playback',
            PipelineStage.TOTAL_ROUND_TRIP: '🔄 Total Round-Trip'
        }
        return name_map.get(stage, stage.value.replace('_', ' ').title())
    
    def _get_stage_status(self, stage: PipelineStage, recent_avg_ms: float) -> str:
        """Get status for a stage based on recent average"""
        warning_threshold, critical_threshold = self.monitor.thresholds.get_stage_thresholds(stage)
        
        recent_avg_s = recent_avg_ms / 1000.0
        
        if recent_avg_s >= critical_threshold:
            return 'critical'
        elif recent_avg_s >= warning_threshold:
            return 'warning'
        else:
            return 'good'
    
    def _get_alert_icon(self, level: str) -> str:
        """Get icon for alert level"""
        icons = {
            'critical': '🔴',
            'warning': '🟡',
            'info': '🔵'
        }
        return icons.get(level, '❓')
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human readable form"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def take_snapshot(self, filename: Optional[str] = None) -> str:
        """Take a snapshot of current dashboard state"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"dashboard_snapshot_{timestamp}.json"
        
        data = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'summary': self.monitor.get_summary(),
            'metrics': self.monitor.get_metrics(),
            'alerts': self.monitor.get_recent_alerts(limit=50)
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filename


class PerformanceReporter:
    """Generate performance reports and exports"""
    
    def __init__(self, monitor: Optional[PerformanceMonitor] = None):
        self.monitor = monitor or get_monitor()
    
    def generate_summary_report(self) -> str:
        """Generate a text summary report"""
        summary = self.monitor.get_summary()
        metrics = self.monitor.get_metrics()
        alerts = self.monitor.get_recent_alerts(limit=10)
        
        report = []
        report.append("VOICE ASSISTANT PERFORMANCE REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall performance
        report.append("OVERALL PERFORMANCE")
        report.append("-" * 30)
        report.append(f"Status: {summary['performance_status'].upper()}")
        report.append(f"Average Latency: {summary['average_latency_ms']:.1f}ms")
        report.append(f"Target Latency: {summary['target_latency_ms']:.0f}ms")
        report.append(f"Success Rate: {summary['success_rate']:.1f}%")
        report.append(f"Total Sessions: {summary['total_sessions']}")
        report.append(f"Uptime: {summary['uptime_seconds']:.0f} seconds")
        report.append("")
        
        # Stage breakdown
        report.append("STAGE BREAKDOWN")
        report.append("-" * 30)
        for stage_name, metric in metrics.items():
            if metric['total_calls'] > 0:
                report.append(f"{stage_name}:")
                report.append(f"  Calls: {metric['total_calls']}")
                report.append(f"  Success Rate: {metric['success_rate']:.1f}%")
                report.append(f"  Average: {metric['recent_average']:.1f}ms")
                report.append(f"  P95: {metric['p95_duration']:.1f}ms")
                report.append("")
        
        # Recent alerts
        if alerts:
            report.append("RECENT ALERTS")
            report.append("-" * 30)
            for alert in reversed(alerts[-5:]):
                timestamp = datetime.fromtimestamp(alert['timestamp']).strftime('%H:%M:%S')
                report.append(f"{timestamp} [{alert['level'].upper()}] {alert['message']}")
            report.append("")
        
        return "\n".join(report)
    
    def export_csv_metrics(self, filename: Optional[str] = None) -> str:
        """Export metrics to CSV format"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"performance_metrics_{timestamp}.csv"
        
        metrics = self.monitor.get_metrics()
        
        # CSV header
        csv_lines = [
            "stage,total_calls,successful_calls,failed_calls,success_rate,avg_duration_ms,recent_avg_ms,min_duration_ms,max_duration_ms,p95_duration_ms"
        ]
        
        # CSV data
        for stage_name, metric in metrics.items():
            line = (
                f"{stage_name},"
                f"{metric['total_calls']},"
                f"{metric['successful_calls']},"
                f"{metric['failed_calls']},"
                f"{metric['success_rate']:.2f},"
                f"{metric['average_duration']:.2f},"
                f"{metric['recent_average']:.2f},"
                f"{metric['min_duration']:.2f},"
                f"{metric['max_duration']:.2f},"
                f"{metric['p95_duration']:.2f}"
            )
            csv_lines.append(line)
        
        with open(filename, 'w') as f:
            f.write("\n".join(csv_lines))
        
        return filename
    
    def export_alerts_log(self, filename: Optional[str] = None, limit: int = 100) -> str:
        """Export alerts to log format"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"alerts_log_{timestamp}.txt"
        
        alerts = self.monitor.get_recent_alerts(limit=limit)
        
        log_lines = []
        log_lines.append(f"VOICE ASSISTANT ALERTS LOG")
        log_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_lines.append(f"Total alerts: {len(alerts)}")
        log_lines.append("")
        
        for alert in alerts:
            timestamp = datetime.fromtimestamp(alert['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            log_lines.append(
                f"{timestamp} [{alert['level'].upper()}] {alert['stage']}: {alert['message']} "
                f"(value: {alert['value']:.3f}s, threshold: {alert['threshold']:.3f}s)"
            )
        
        with open(filename, 'w') as f:
            f.write("\n".join(log_lines))
        
        return filename


def create_dashboard(monitor: Optional[PerformanceMonitor] = None) -> ConsoleDashboard:
    """Create a console dashboard instance"""
    return ConsoleDashboard(monitor)


def create_reporter(monitor: Optional[PerformanceMonitor] = None) -> PerformanceReporter:
    """Create a performance reporter instance"""
    return PerformanceReporter(monitor)


if __name__ == "__main__":
    # Example usage - run a dashboard with mock data
    import random
    import asyncio
    
    # Create a monitor with some test data
    monitor = get_monitor()
    
    # Simulate some measurements
    stages = [PipelineStage.AUDIO_CAPTURE, PipelineStage.STT_PROCESSING, 
              PipelineStage.TTS_GENERATION, PipelineStage.AUDIO_PLAYBACK]
    
    for _ in range(10):
        for stage in stages:
            duration = random.uniform(0.05, 0.5)  # Random duration between 50ms and 500ms
            success = random.random() > 0.1  # 90% success rate
            monitor.record_measurement(stage, duration, success)
        
        # Record total round trip
        total_duration = random.uniform(0.4, 1.2)
        monitor.record_measurement(PipelineStage.TOTAL_ROUND_TRIP, total_duration, True)
    
    # Create and start dashboard
    dashboard = create_dashboard(monitor)
    
    print("Starting dashboard... Press Ctrl+C to stop")
    try:
        dashboard.start()
        
        # Keep main thread alive
        while dashboard.is_running:
            time.sleep(1)
            
            # Add some random measurements
            stage = random.choice(stages)
            duration = random.uniform(0.05, 0.8)
            success = random.random() > 0.05
            monitor.record_measurement(stage, duration, success)
            
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
        dashboard.stop()
        
        # Generate final report
        reporter = create_reporter(monitor)
        report = reporter.generate_summary_report()
        print("\nFinal Report:")
        print(report) 