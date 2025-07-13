#!/usr/bin/env python3
"""
Stability and Endurance Testing Framework
8-hour continuous operation testing with comprehensive monitoring
"""

import asyncio
import time
import sys
import json
import csv
import statistics
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

@dataclass
class SystemMetrics:
    """System metrics snapshot"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_rss_mb: float
    memory_available_gb: float
    disk_usage_percent: float
    active_threads: int
    open_files: int = 0
    network_connections: int = 0

@dataclass
class PerformanceSnapshot:
    """Performance metrics snapshot"""
    timestamp: float
    request_count: int
    avg_latency: float
    error_count: int
    success_rate: float
    throughput: float

@dataclass
class StabilityResults:
    """Stability test results"""
    start_time: float
    end_time: float
    duration_hours: float
    total_requests: int
    total_errors: int
    avg_latency: float
    memory_leak_detected: bool
    performance_degradation: bool
    system_metrics: List[SystemMetrics]
    performance_snapshots: List[PerformanceSnapshot]

class StabilityMonitor:
    """Comprehensive stability and resource monitoring"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("reports/stability")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.system_metrics: List[SystemMetrics] = []
        self.performance_snapshots: List[PerformanceSnapshot] = []
        self.start_time = None
        self.monitoring = False
        
        # Import psutil for system monitoring
        try:
            import psutil
            self.psutil = psutil
            self.psutil_available = True
        except ImportError:
            print("⚠️  Warning: psutil not available. Limited system monitoring.")
            self.psutil_available = False
    
    def start_monitoring(self):
        """Start continuous system monitoring"""
        self.start_time = time.time()
        self.monitoring = True
        print(f"📊 Starting stability monitoring at {datetime.now().strftime('%H:%M:%S')}")
    
    def stop_monitoring(self):
        """Stop monitoring and generate reports"""
        self.monitoring = False
        duration = time.time() - self.start_time if self.start_time else 0
        print(f"📊 Stopped monitoring after {duration/3600:.2f} hours")
        return self.generate_stability_report()
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        if not self.psutil_available:
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_rss_mb=0.0,
                memory_available_gb=0.0,
                disk_usage_percent=0.0,
                active_threads=0
            )
        
        try:
            process = self.psutil.Process()
            memory_info = self.psutil.virtual_memory()
            disk_usage = self.psutil.disk_usage('/')
            
            # Get process-specific metrics
            proc_memory = process.memory_info()
            
            metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=self.psutil.cpu_percent(interval=0.1),
                memory_percent=memory_info.percent,
                memory_rss_mb=proc_memory.rss / 1024 / 1024,
                memory_available_gb=memory_info.available / (1024**3),
                disk_usage_percent=disk_usage.percent,
                active_threads=process.num_threads(),
                open_files=len(process.open_files()),
                network_connections=len(process.connections())
            )
            
            self.system_metrics.append(metrics)
            return metrics
            
        except Exception as e:
            print(f"⚠️  Error collecting system metrics: {e}")
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_rss_mb=0.0,
                memory_available_gb=0.0,
                disk_usage_percent=0.0,
                active_threads=0
            )
    
    def record_performance(self, request_count: int, avg_latency: float, 
                          error_count: int, success_rate: float, throughput: float):
        """Record performance metrics"""
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            request_count=request_count,
            avg_latency=avg_latency,
            error_count=error_count,
            success_rate=success_rate,
            throughput=throughput
        )
        self.performance_snapshots.append(snapshot)
    
    def detect_memory_leak(self) -> bool:
        """Detect memory leaks by analyzing memory usage trends"""
        if len(self.system_metrics) < 10:
            return False
        
        # Analyze memory usage trend over time
        memory_usage = [m.memory_rss_mb for m in self.system_metrics[-10:]]
        
        # Simple linear regression to detect upward trend
        n = len(memory_usage)
        x = list(range(n))
        
        # Calculate slope
        x_mean = sum(x) / n
        y_mean = sum(memory_usage) / n
        
        numerator = sum((x[i] - x_mean) * (memory_usage[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return False
        
        slope = numerator / denominator
        
        # If memory is consistently increasing at > 1MB per sample, flag as potential leak
        return slope > 1.0
    
    def detect_performance_degradation(self) -> bool:
        """Detect performance degradation over time"""
        if len(self.performance_snapshots) < 10:
            return False
        
        # Compare recent performance to initial performance
        recent_latencies = [p.avg_latency for p in self.performance_snapshots[-5:]]
        initial_latencies = [p.avg_latency for p in self.performance_snapshots[:5]]
        
        if not recent_latencies or not initial_latencies:
            return False
        
        recent_avg = sum(recent_latencies) / len(recent_latencies)
        initial_avg = sum(initial_latencies) / len(initial_latencies)
        
        # Flag if recent performance is >50% worse than initial
        degradation_threshold = 1.5
        return recent_avg > initial_avg * degradation_threshold
    
    def generate_stability_report(self) -> StabilityResults:
        """Generate comprehensive stability report"""
        if not self.start_time or not self.system_metrics:
            return StabilityResults(
                start_time=0, end_time=0, duration_hours=0,
                total_requests=0, total_errors=0, avg_latency=0.0,
                memory_leak_detected=False, performance_degradation=False,
                system_metrics=[], performance_snapshots=[]
            )
        
        end_time = time.time()
        duration_hours = (end_time - self.start_time) / 3600
        
        # Calculate aggregate metrics
        total_requests = sum(p.request_count for p in self.performance_snapshots)
        total_errors = sum(p.error_count for p in self.performance_snapshots)
        
        if self.performance_snapshots:
            avg_latency = statistics.mean(p.avg_latency for p in self.performance_snapshots)
        else:
            avg_latency = 0.0
        
        # Detect issues
        memory_leak = self.detect_memory_leak()
        perf_degradation = self.detect_performance_degradation()
        
        results = StabilityResults(
            start_time=self.start_time,
            end_time=end_time,
            duration_hours=duration_hours,
            total_requests=total_requests,
            total_errors=total_errors,
            avg_latency=avg_latency,
            memory_leak_detected=memory_leak,
            performance_degradation=perf_degradation,
            system_metrics=self.system_metrics.copy(),
            performance_snapshots=self.performance_snapshots.copy()
        )
        
        # Save detailed reports
        self.save_detailed_reports(results)
        
        return results
    
    def save_detailed_reports(self, results: StabilityResults):
        """Save detailed CSV and JSON reports"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save system metrics to CSV
        metrics_file = self.output_dir / f"system_metrics_{timestamp}.csv"
        with open(metrics_file, 'w', newline='') as f:
            if results.system_metrics:
                writer = csv.DictWriter(f, fieldnames=asdict(results.system_metrics[0]).keys())
                writer.writeheader()
                for metric in results.system_metrics:
                    writer.writerow(asdict(metric))
        
        # Save performance snapshots to CSV
        perf_file = self.output_dir / f"performance_{timestamp}.csv"
        with open(perf_file, 'w', newline='') as f:
            if results.performance_snapshots:
                writer = csv.DictWriter(f, fieldnames=asdict(results.performance_snapshots[0]).keys())
                writer.writeheader()
                for snapshot in results.performance_snapshots:
                    writer.writerow(asdict(snapshot))
        
        # Save summary to JSON
        summary_file = self.output_dir / f"stability_summary_{timestamp}.json"
        summary_data = {
            'start_time': results.start_time,
            'end_time': results.end_time,
            'duration_hours': results.duration_hours,
            'total_requests': results.total_requests,
            'total_errors': results.total_errors,
            'avg_latency': results.avg_latency,
            'memory_leak_detected': results.memory_leak_detected,
            'performance_degradation': results.performance_degradation,
            'metrics_file': str(metrics_file),
            'performance_file': str(perf_file)
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"📁 Detailed reports saved:")
        print(f"   System metrics: {metrics_file}")
        print(f"   Performance: {perf_file}")
        print(f"   Summary: {summary_file}")

class EnduranceTestRunner:
    """Run endurance tests with various scenarios"""
    
    def __init__(self, monitor: StabilityMonitor):
        self.monitor = monitor
        self.running = False
        
    async def run_endurance_test(self, duration_hours: float = 8.0, 
                                accelerated: bool = False):
        """Run endurance test for specified duration"""
        
        if accelerated:
            print(f"🚀 Starting ACCELERATED endurance test (simulating {duration_hours}h)")
            # Accelerated test: 1 minute = 1 hour simulation
            actual_duration = duration_hours * 60  # seconds
            cycle_interval = 1.0  # Check every second
        else:
            print(f"🚀 Starting FULL endurance test ({duration_hours}h continuous operation)")
            actual_duration = duration_hours * 3600  # seconds  
            cycle_interval = 60.0  # Check every minute
        
        self.monitor.start_monitoring()
        self.running = True
        
        start_time = time.time()
        end_time = start_time + actual_duration
        cycle_count = 0
        
        try:
            while time.time() < end_time and self.running:
                cycle_count += 1
                
                # Perform test operations
                await self._run_test_cycle(cycle_count)
                
                # Log progress
                elapsed = time.time() - start_time
                if accelerated:
                    simulated_hours = elapsed / 60
                    print(f"⏱️  Cycle {cycle_count}: {simulated_hours:.1f}h simulated "
                          f"({elapsed:.0f}s elapsed)")
                else:
                    hours_elapsed = elapsed / 3600
                    print(f"⏱️  Cycle {cycle_count}: {hours_elapsed:.2f}h elapsed")
                
                # Wait for next cycle
                await asyncio.sleep(cycle_interval)
                
        except KeyboardInterrupt:
            print(f"\n⚠️  Endurance test interrupted by user")
        except Exception as e:
            print(f"\n❌ Endurance test failed: {e}")
        finally:
            self.running = False
            results = self.monitor.stop_monitoring()
            return results
    
    async def _run_test_cycle(self, cycle: int):
        """Run a single test cycle"""
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            from assistant.llm_router import LLMRouter
            
            # Simulate voice assistant operations
            router = LLMRouter()
            
            # Test scenarios rotating through different types
            scenarios = [
                "Hello, how are you?",
                "What time is it?", 
                "Help me debug this error",
                "Explain machine learning",
                "What's the weather like?",
                "Can you help me with coding?",
                "Tell me a joke",
                "What are your capabilities?"
            ]
            
            cycle_start = time.time()
            requests_this_cycle = 3  # 3 requests per cycle
            errors_this_cycle = 0
            latencies = []
            
            for i in range(requests_this_cycle):
                query = scenarios[(cycle + i) % len(scenarios)]
                
                try:
                    request_start = time.time()
                    result = await router.route_query(f"Cycle {cycle}: {query}")
                    latency = time.time() - request_start
                    latencies.append(latency)
                    
                    # Verify response quality
                    if not result.get('response') or len(result.get('response', '')) < 10:
                        errors_this_cycle += 1
                        
                except Exception as e:
                    errors_this_cycle += 1
                    latencies.append(5.0)  # Default high latency for errors
            
            # Calculate cycle metrics
            cycle_duration = time.time() - cycle_start
            avg_latency = statistics.mean(latencies) if latencies else 0.0
            success_rate = (requests_this_cycle - errors_this_cycle) / requests_this_cycle
            throughput = requests_this_cycle / cycle_duration
            
            # Record metrics
            self.monitor.collect_system_metrics()
            self.monitor.record_performance(
                requests_this_cycle, avg_latency, errors_this_cycle, 
                success_rate, throughput
            )
            
            await router.cleanup()
            
        except Exception as e:
            print(f"⚠️  Error in test cycle {cycle}: {e}")
            # Record failed cycle
            self.monitor.collect_system_metrics()
            self.monitor.record_performance(0, 5.0, 1, 0.0, 0.0)

class StabilityAnalyzer:
    """Analyze stability test results"""
    
    @staticmethod
    def analyze_results(results: StabilityResults) -> Dict[str, Any]:
        """Perform comprehensive analysis of stability results"""
        
        analysis = {
            'overall_status': 'UNKNOWN',
            'duration_analysis': {},
            'performance_analysis': {},
            'resource_analysis': {},
            'stability_issues': [],
            'recommendations': []
        }
        
        # Duration analysis
        analysis['duration_analysis'] = {
            'target_hours': 8.0,
            'actual_hours': results.duration_hours,
            'completion_rate': results.duration_hours / 8.0 if results.duration_hours > 0 else 0,
            'passed': results.duration_hours >= 7.5  # Allow 6.25% tolerance
        }
        
        # Performance analysis
        if results.performance_snapshots:
            error_rate = results.total_errors / max(results.total_requests, 1)
            analysis['performance_analysis'] = {
                'total_requests': results.total_requests,
                'total_errors': results.total_errors,
                'error_rate': error_rate,
                'avg_latency': results.avg_latency,
                'latency_acceptable': results.avg_latency < 5.0,
                'error_rate_acceptable': error_rate < 0.05,  # <5% error rate
                'performance_stable': not results.performance_degradation
            }
        
        # Resource analysis
        if results.system_metrics:
            cpu_values = [m.cpu_percent for m in results.system_metrics]
            memory_values = [m.memory_rss_mb for m in results.system_metrics]
            
            analysis['resource_analysis'] = {
                'avg_cpu_percent': statistics.mean(cpu_values),
                'max_cpu_percent': max(cpu_values),
                'avg_memory_mb': statistics.mean(memory_values),
                'max_memory_mb': max(memory_values),
                'memory_leak_detected': results.memory_leak_detected,
                'cpu_acceptable': max(cpu_values) < 80,
                'memory_stable': not results.memory_leak_detected
            }
        
        # Identify stability issues
        if results.memory_leak_detected:
            analysis['stability_issues'].append("Memory leak detected")
            analysis['recommendations'].append("Investigate memory cleanup in long-running processes")
        
        if results.performance_degradation:
            analysis['stability_issues'].append("Performance degradation over time")
            analysis['recommendations'].append("Profile performance bottlenecks in extended operation")
        
        if analysis['performance_analysis'].get('error_rate', 0) > 0.05:
            analysis['stability_issues'].append("High error rate")
            analysis['recommendations'].append("Improve error handling and recovery mechanisms")
        
        # Overall status determination
        critical_issues = len([issue for issue in analysis['stability_issues'] 
                             if 'leak' in issue.lower() or 'degradation' in issue.lower()])
        
        if critical_issues == 0 and analysis['duration_analysis']['passed']:
            analysis['overall_status'] = 'PASSED'
        elif critical_issues == 0:
            analysis['overall_status'] = 'PARTIAL'
        else:
            analysis['overall_status'] = 'FAILED'
        
        if not analysis['recommendations']:
            analysis['recommendations'].append("System demonstrates good stability characteristics")
        
        return analysis

async def main():
    """Main endurance testing entry point"""
    parser = argparse.ArgumentParser(description="Sovereign 4.0 Stability & Endurance Testing")
    parser.add_argument('--duration', '-d', type=float, default=8.0, 
                       help='Test duration in hours (default: 8.0)')
    parser.add_argument('--accelerated', '-a', action='store_true',
                       help='Run accelerated test (1 minute = 1 hour simulation)')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Run quick stability test (10 minutes)')
    parser.add_argument('--output-dir', '-o', type=str, 
                       help='Output directory for reports')
    
    args = parser.parse_args()
    
    if args.quick:
        duration = 0.17  # ~10 minutes
        accelerated = True
    else:
        duration = args.duration
        accelerated = args.accelerated
    
    # Setup monitoring
    output_dir = Path(args.output_dir) if args.output_dir else None
    monitor = StabilityMonitor(output_dir)
    runner = EnduranceTestRunner(monitor)
    
    print("🚀 Starting Sovereign 4.0 Stability & Endurance Testing")
    print("=" * 60)
    print(f"Duration: {duration} hours {'(accelerated)' if accelerated else '(real-time)'}")
    print(f"Output: {monitor.output_dir}")
    print("=" * 60)
    
    try:
        # Run endurance test
        results = await runner.run_endurance_test(duration, accelerated)
        
        # Analyze results
        analysis = StabilityAnalyzer.analyze_results(results)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 STABILITY & ENDURANCE TEST RESULTS")
        print("=" * 60)
        
        status_emoji = {"PASSED": "✅", "PARTIAL": "⚠️", "FAILED": "❌"}[analysis['overall_status']]
        print(f"{status_emoji} Overall Status: {analysis['overall_status']}")
        
        print(f"\n⏱️  Duration Analysis:")
        duration_info = analysis['duration_analysis']
        print(f"   Target: {duration_info['target_hours']} hours")
        print(f"   Actual: {duration_info['actual_hours']:.2f} hours")
        print(f"   Completion: {duration_info['completion_rate']:.1%}")
        
        if 'performance_analysis' in analysis and analysis['performance_analysis']:
            perf_info = analysis['performance_analysis']
            print(f"\n🚀 Performance Analysis:")
            print(f"   Requests: {perf_info['total_requests']}")
            print(f"   Errors: {perf_info['total_errors']}")
            print(f"   Error Rate: {perf_info['error_rate']:.1%}")
            print(f"   Avg Latency: {perf_info['avg_latency']:.3f}s")
        
        if 'resource_analysis' in analysis and analysis['resource_analysis']:
            resource_info = analysis['resource_analysis']
            print(f"\n💻 Resource Analysis:")
            print(f"   Avg CPU: {resource_info['avg_cpu_percent']:.1f}%")
            print(f"   Max CPU: {resource_info['max_cpu_percent']:.1f}%")
            print(f"   Avg Memory: {resource_info['avg_memory_mb']:.1f} MB")
            print(f"   Max Memory: {resource_info['max_memory_mb']:.1f} MB")
        
        if analysis['stability_issues']:
            print(f"\n⚠️  Stability Issues:")
            for issue in analysis['stability_issues']:
                print(f"   - {issue}")
        
        print(f"\n💡 Recommendations:")
        for rec in analysis['recommendations']:
            print(f"   - {rec}")
        
        # Return appropriate exit code
        if analysis['overall_status'] == 'PASSED':
            print(f"\n🎯 ENDURANCE TEST SUCCESS: System demonstrates excellent stability")
            return True
        else:
            print(f"\n⚠️  ENDURANCE TEST NEEDS ATTENTION: See issues above")
            return False
            
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Endurance testing interrupted by user")
        return False
    except Exception as e:
        print(f"\n\n❌ Endurance testing failed: {e}")
        return False

if __name__ == '__main__':
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Testing failed: {e}")
        sys.exit(1) 