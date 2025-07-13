"""
Automated Performance Testing Suite for Sovereign 4.0 Voice Assistant

This module provides comprehensive performance testing capabilities including:
- End-to-end latency testing with percentile tracking
- Accuracy benchmarking with multiple metrics (BLEU, semantic similarity)
- Stress testing with synthetic workload generation
- Memory leak detection with continuous monitoring
- Performance regression detection using ML models
- Automated report generation with pass/fail criteria

Based on 2024-2025 best practices for AI voice assistant performance testing.
"""

import asyncio
import time
import logging
import psutil
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, AsyncGenerator
from collections import deque, defaultdict
from pathlib import Path
import json
import tracemalloc
import gc
import threading
from concurrent.futures import ThreadPoolExecutor

# Performance monitoring integration
from .metrics_collector import MetricsCollector, MetricType, ComponentType

# Import all the test suite components
from .accuracy_testing import AccuracyBenchmarkSuite, create_accuracy_benchmark_suite
from .stress_testing import StressTestSuite, create_stress_test_suite
from .memory_monitoring import MemoryLeakDetector, create_memory_leak_detector
from .regression_detection import RegressionDetector, create_regression_detector
from .report_generation import ReportGenerator, create_report_generator

logger = logging.getLogger(__name__)

@dataclass
class PerformanceTestConfig:
    """Configuration for performance testing suite"""
    # Latency thresholds (milliseconds)
    latency_targets: Dict[str, float] = field(default_factory=lambda: {
        'cloud_p95': 800.0,
        'offline_p95': 1500.0,
        'stt_p95': 200.0,
        'llm_p95': 400.0,
        'tts_p95': 100.0
    })
    
    # Accuracy thresholds
    accuracy_targets: Dict[str, float] = field(default_factory=lambda: {
        'stt_accuracy': 0.95,
        'memory_recall_bleu': 0.8,
        'memory_recall_semantic': 0.85,
        'ocr_accuracy': 0.92
    })
    
    # Stress testing parameters
    stress_test_config: Dict[str, Any] = field(default_factory=lambda: {
        'max_concurrent_users': 100,
        'test_duration_seconds': 300,
        'ramp_up_duration_seconds': 60,
        'memory_leak_threshold_percent': 10
    })
    
    # Regression detection settings
    regression_config: Dict[str, Any] = field(default_factory=lambda: {
        'anomaly_contamination': 0.1,
        'z_score_threshold': 2.5,
        'minimum_samples': 50
    })

@dataclass
class TestResult:
    """Individual test result"""
    test_name: str
    status: str  # 'passed', 'failed', 'warning'
    metrics: Dict[str, float]
    execution_time: float
    timestamp: datetime
    details: Optional[str] = None
    errors: List[str] = field(default_factory=list)

@dataclass
class PerformanceTestReport:
    """Comprehensive performance test report"""
    test_session_id: str
    timestamp: datetime
    overall_status: str  # 'passed', 'failed', 'partial'
    test_results: List[TestResult]
    summary_metrics: Dict[str, float]
    recommendations: List[str]
    execution_time: float

class PerformanceTestFramework:
    """
    Main performance testing framework orchestrating all test components
    
    Integrates with existing MetricsCollector from Task 10.1 and provides
    comprehensive automated testing capabilities for the voice assistant system.
    """
    
    def __init__(self, 
                 metrics_collector: Optional[MetricsCollector] = None,
                 config: Optional[PerformanceTestConfig] = None):
        self.metrics_collector = metrics_collector
        self.config = config or PerformanceTestConfig()
        
        # Initialize test components
        self.latency_tester = LatencyTestSuite(self.config, self.metrics_collector)
        self.accuracy_tester = AccuracyBenchmarkSuite(self.config)
        self.stress_tester = StressTestSuite(self.config)
        self.memory_monitor = MemoryLeakDetector(self.config)
        self.regression_detector = RegressionDetector(self.config)
        self.report_generator = ReportGenerator(self.config)
        
        # Test execution state
        self.is_running = False
        self.current_session_id = None
        self.test_results: List[TestResult] = []
        
        logger.info("Performance Testing Framework initialized")
    
    async def run_comprehensive_test_suite(self, 
                                         test_types: Optional[List[str]] = None) -> PerformanceTestReport:
        """
        Run comprehensive performance test suite
        
        Args:
            test_types: Optional list of test types to run. If None, runs all tests.
                       Available: ['latency', 'accuracy', 'stress', 'memory', 'regression']
        
        Returns:
            PerformanceTestReport with complete results and recommendations
        """
        if self.is_running:
            raise RuntimeError("Performance test suite is already running")
        
        self.is_running = True
        start_time = time.time()
        self.current_session_id = f"perf_test_{int(start_time)}"
        self.test_results = []
        
        # Default to all test types if none specified
        if test_types is None:
            test_types = ['latency', 'accuracy', 'stress', 'memory', 'regression']
        
        try:
            logger.info(f"Starting comprehensive performance test suite: {self.current_session_id}")
            
            # Run test suites in parallel where possible
            test_tasks = []
            
            if 'latency' in test_types:
                test_tasks.append(self._run_latency_tests())
            
            if 'accuracy' in test_types:
                test_tasks.append(self._run_accuracy_tests())
            
            if 'memory' in test_types:
                # Start memory monitoring first (continuous)
                asyncio.create_task(self.memory_monitor.start_monitoring())
            
            if 'stress' in test_types:
                test_tasks.append(self._run_stress_tests())
            
            if 'regression' in test_types:
                test_tasks.append(self._run_regression_tests())
            
            # Execute all test suites
            if test_tasks:
                await asyncio.gather(*test_tasks, return_exceptions=True)
            
            # Stop memory monitoring and collect results
            if 'memory' in test_types:
                memory_results = await self.memory_monitor.stop_monitoring_and_analyze()
                self.test_results.extend(memory_results)
            
            # Generate comprehensive report
            execution_time = time.time() - start_time
            report = await self.report_generator.generate_report(
                session_id=self.current_session_id,
                test_results=self.test_results,
                execution_time=execution_time
            )
            
            logger.info(f"Performance test suite completed in {execution_time:.2f}s")
            return report
            
        except Exception as e:
            logger.error(f"Performance test suite failed: {e}")
            raise
        finally:
            self.is_running = False
    
    async def _run_latency_tests(self) -> None:
        """Run comprehensive latency testing suite"""
        try:
            logger.info("Starting latency test suite")
            results = await self.latency_tester.run_all_latency_tests()
            self.test_results.extend(results)
            logger.info(f"Latency tests completed: {len(results)} tests")
        except Exception as e:
            logger.error(f"Latency tests failed: {e}")
            self.test_results.append(TestResult(
                test_name="latency_suite",
                status="failed",
                metrics={},
                execution_time=0,
                timestamp=datetime.now(),
                errors=[str(e)]
            ))
    
    async def _run_accuracy_tests(self) -> None:
        """Run comprehensive accuracy benchmarking suite"""
        try:
            logger.info("Starting accuracy test suite")
            results = await self.accuracy_tester.run_all_accuracy_tests()
            self.test_results.extend(results)
            logger.info(f"Accuracy tests completed: {len(results)} tests")
        except Exception as e:
            logger.error(f"Accuracy tests failed: {e}")
            self.test_results.append(TestResult(
                test_name="accuracy_suite",
                status="failed",
                metrics={},
                execution_time=0,
                timestamp=datetime.now(),
                errors=[str(e)]
            ))
    
    async def _run_stress_tests(self) -> None:
        """Run comprehensive stress testing suite"""
        try:
            logger.info("Starting stress test suite")
            results = await self.stress_tester.run_all_stress_tests()
            self.test_results.extend(results)
            logger.info(f"Stress tests completed: {len(results)} tests")
        except Exception as e:
            logger.error(f"Stress tests failed: {e}")
            self.test_results.append(TestResult(
                test_name="stress_suite",
                status="failed",
                metrics={},
                execution_time=0,
                timestamp=datetime.now(),
                errors=[str(e)]
            ))
    
    async def _run_regression_tests(self) -> None:
        """Run performance regression detection"""
        try:
            logger.info("Starting regression detection")
            results = await self.regression_detector.detect_performance_regressions()
            self.test_results.extend(results)
            logger.info(f"Regression tests completed: {len(results)} tests")
        except Exception as e:
            logger.error(f"Regression tests failed: {e}")
            self.test_results.append(TestResult(
                test_name="regression_suite",
                status="failed",
                metrics={},
                execution_time=0,
                timestamp=datetime.now(),
                errors=[str(e)]
            ))
    
    async def run_continuous_monitoring(self, duration_hours: int = 8) -> PerformanceTestReport:
        """
        Run continuous performance monitoring for extended periods
        
        This is designed for 8-hour continuous operation testing as specified
        in the task requirements. Monitors for memory leaks, performance degradation,
        and system stability under sustained load.
        
        Args:
            duration_hours: Duration to run continuous monitoring (default: 8 hours)
            
        Returns:
            PerformanceTestReport with extended monitoring results
        """
        if self.is_running:
            raise RuntimeError("Performance test suite is already running")
        
        self.is_running = True
        start_time = time.time()
        self.current_session_id = f"continuous_monitor_{int(start_time)}"
        self.test_results = []
        
        duration_seconds = duration_hours * 3600
        
        try:
            logger.info(f"Starting {duration_hours}h continuous monitoring: {self.current_session_id}")
            
            # Start continuous memory monitoring
            memory_task = asyncio.create_task(
                self.memory_monitor.continuous_monitoring(duration_seconds)
            )
            
            # Run periodic performance validation
            validation_task = asyncio.create_task(
                self._run_periodic_validation(duration_seconds, interval_minutes=30)
            )
            
            # Run sustained load testing
            load_task = asyncio.create_task(
                self.stress_tester.run_sustained_load_test(duration_seconds)
            )
            
            # Wait for all monitoring tasks to complete
            await asyncio.gather(memory_task, validation_task, load_task)
            
            # Generate comprehensive monitoring report
            execution_time = time.time() - start_time
            report = await self.report_generator.generate_continuous_monitoring_report(
                session_id=self.current_session_id,
                test_results=self.test_results,
                execution_time=execution_time,
                duration_hours=duration_hours
            )
            
            logger.info(f"Continuous monitoring completed after {execution_time/3600:.2f}h")
            return report
            
        except Exception as e:
            logger.error(f"Continuous monitoring failed: {e}")
            raise
        finally:
            self.is_running = False
    
    async def _run_periodic_validation(self, total_duration: int, interval_minutes: int = 30) -> None:
        """Run periodic performance validation during continuous monitoring"""
        interval_seconds = interval_minutes * 60
        iterations = int(total_duration / interval_seconds)
        
        for i in range(iterations):
            try:
                logger.info(f"Running periodic validation {i+1}/{iterations}")
                
                # Run quick performance checks
                latency_result = await self.latency_tester.run_quick_latency_check()
                accuracy_result = await self.accuracy_tester.run_quick_accuracy_check()
                regression_result = await self.regression_detector.check_for_regression()
                
                self.test_results.extend([latency_result, accuracy_result, regression_result])
                
                # Wait for next interval
                if i < iterations - 1:  # Don't wait after last iteration
                    await asyncio.sleep(interval_seconds)
                    
            except Exception as e:
                logger.error(f"Periodic validation {i+1} failed: {e}")
                self.test_results.append(TestResult(
                    test_name=f"periodic_validation_{i+1}",
                    status="failed",
                    metrics={},
                    execution_time=0,
                    timestamp=datetime.now(),
                    errors=[str(e)]
                ))
    
    def get_test_status(self) -> Dict[str, Any]:
        """Get current test execution status"""
        return {
            'is_running': self.is_running,
            'current_session_id': self.current_session_id,
            'tests_completed': len(self.test_results),
            'last_update': datetime.now().isoformat()
        }
    
    async def stop_all_tests(self) -> None:
        """Emergency stop for all running tests"""
        logger.warning("Emergency stop requested for all performance tests")
        self.is_running = False
        
        # Stop individual test components
        await self.memory_monitor.stop_monitoring()
        await self.stress_tester.stop_all_tests()
        
        logger.info("All performance tests stopped")


class LatencyTestSuite:
    """
    Comprehensive latency testing suite with percentile tracking
    
    Implements end-to-end latency measurement with component-level breakdown
    as recommended in the research findings.
    """
    
    def __init__(self, config: PerformanceTestConfig, metrics_collector: Optional[MetricsCollector] = None):
        self.config = config
        self.metrics_collector = metrics_collector
        self.latency_history = deque(maxlen=1000)
    
    async def run_all_latency_tests(self) -> List[TestResult]:
        """Run comprehensive latency test suite"""
        results = []
        
        # End-to-end latency test
        results.append(await self.test_end_to_end_latency())
        
        # Component-level latency tests
        results.append(await self.test_component_latencies())
        
        # Percentile analysis
        results.append(await self.test_latency_percentiles())
        
        # Network latency simulation
        results.append(await self.test_network_conditions())
        
        return results
    
    async def test_end_to_end_latency(self) -> TestResult:
        """Test complete voice pipeline latency"""
        start_time = time.time()
        test_name = "end_to_end_latency"
        
        try:
            latencies = []
            
            # Simulate 50 voice interactions
            for i in range(50):
                interaction_start = time.time()
                
                # Simulate voice pipeline (would integrate with actual pipeline)
                await self._simulate_voice_interaction()
                
                interaction_latency = (time.time() - interaction_start) * 1000  # ms
                latencies.append(interaction_latency)
                self.latency_history.append(interaction_latency)
            
            # Calculate metrics
            metrics = {
                'mean_latency_ms': np.mean(latencies),
                'p50_latency_ms': np.percentile(latencies, 50),
                'p95_latency_ms': np.percentile(latencies, 95),
                'p99_latency_ms': np.percentile(latencies, 99),
                'max_latency_ms': np.max(latencies),
                'std_dev_ms': np.std(latencies)
            }
            
            # Determine test status based on targets
            p95_target = self.config.latency_targets['cloud_p95']
            status = 'passed' if metrics['p95_latency_ms'] <= p95_target else 'failed'
            
            return TestResult(
                test_name=test_name,
                status=status,
                metrics=metrics,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                details=f"P95 latency: {metrics['p95_latency_ms']:.2f}ms (target: {p95_target}ms)"
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_name,
                status="failed",
                metrics={},
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                errors=[str(e)]
            )
    
    async def test_component_latencies(self) -> TestResult:
        """Test individual component latencies (STT, LLM, TTS)"""
        start_time = time.time()
        test_name = "component_latencies"
        
        try:
            component_metrics = {}
            
            # Test STT latency
            stt_latencies = []
            for _ in range(20):
                stt_start = time.time()
                await self._simulate_stt_processing()
                stt_latencies.append((time.time() - stt_start) * 1000)
            
            component_metrics.update({
                'stt_mean_ms': np.mean(stt_latencies),
                'stt_p95_ms': np.percentile(stt_latencies, 95)
            })
            
            # Test LLM latency
            llm_latencies = []
            for _ in range(20):
                llm_start = time.time()
                await self._simulate_llm_processing()
                llm_latencies.append((time.time() - llm_start) * 1000)
            
            component_metrics.update({
                'llm_mean_ms': np.mean(llm_latencies),
                'llm_p95_ms': np.percentile(llm_latencies, 95)
            })
            
            # Test TTS latency
            tts_latencies = []
            for _ in range(20):
                tts_start = time.time()
                await self._simulate_tts_processing()
                tts_latencies.append((time.time() - tts_start) * 1000)
            
            component_metrics.update({
                'tts_mean_ms': np.mean(tts_latencies),
                'tts_p95_ms': np.percentile(tts_latencies, 95)
            })
            
            # Check against component targets
            failures = []
            if component_metrics['stt_p95_ms'] > self.config.latency_targets['stt_p95']:
                failures.append(f"STT P95: {component_metrics['stt_p95_ms']:.2f}ms")
            if component_metrics['llm_p95_ms'] > self.config.latency_targets['llm_p95']:
                failures.append(f"LLM P95: {component_metrics['llm_p95_ms']:.2f}ms")
            if component_metrics['tts_p95_ms'] > self.config.latency_targets['tts_p95']:
                failures.append(f"TTS P95: {component_metrics['tts_p95_ms']:.2f}ms")
            
            status = 'passed' if not failures else 'failed'
            
            return TestResult(
                test_name=test_name,
                status=status,
                metrics=component_metrics,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                details=f"Component failures: {failures}" if failures else "All components within targets"
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_name,
                status="failed",
                metrics={},
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                errors=[str(e)]
            )
    
    async def test_latency_percentiles(self) -> TestResult:
        """Comprehensive percentile analysis of latency distribution"""
        start_time = time.time()
        test_name = "latency_percentiles"
        
        try:
            if len(self.latency_history) < 50:
                # Generate test data if insufficient history
                for _ in range(100):
                    await self._simulate_voice_interaction()
            
            latencies = list(self.latency_history)
            
            # Calculate comprehensive percentile metrics
            percentiles = [50, 75, 90, 95, 99, 99.9]
            percentile_values = np.percentile(latencies, percentiles)
            
            metrics = {
                f'p{p}_latency_ms': value 
                for p, value in zip(percentiles, percentile_values)
            }
            
            # Add distribution metrics
            metrics.update({
                'latency_variance': np.var(latencies),
                'latency_skewness': self._calculate_skewness(latencies),
                'sample_count': len(latencies)
            })
            
            # Validate against industry benchmarks
            status = 'passed'
            if metrics['p95_latency_ms'] > self.config.latency_targets['cloud_p95']:
                status = 'failed'
            elif metrics['p99_latency_ms'] > self.config.latency_targets['cloud_p95'] * 1.5:
                status = 'warning'
            
            return TestResult(
                test_name=test_name,
                status=status,
                metrics=metrics,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                details=f"Latency distribution analysis on {len(latencies)} samples"
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_name,
                status="failed",
                metrics={},
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                errors=[str(e)]
            )
    
    async def test_network_conditions(self) -> TestResult:
        """Test latency under various network conditions"""
        start_time = time.time()
        test_name = "network_conditions"
        
        try:
            network_scenarios = [
                {'name': 'optimal', 'delay_ms': 0, 'jitter_ms': 0},
                {'name': 'good_wifi', 'delay_ms': 20, 'jitter_ms': 5},
                {'name': 'poor_wifi', 'delay_ms': 100, 'jitter_ms': 20},
                {'name': 'mobile_4g', 'delay_ms': 50, 'jitter_ms': 15}
            ]
            
            scenario_metrics = {}
            
            for scenario in network_scenarios:
                scenario_latencies = []
                
                for _ in range(20):
                    # Simulate network conditions
                    network_delay = scenario['delay_ms'] + np.random.normal(0, scenario['jitter_ms'])
                    
                    interaction_start = time.time()
                    await self._simulate_voice_interaction_with_network_delay(network_delay)
                    
                    total_latency = (time.time() - interaction_start) * 1000
                    scenario_latencies.append(total_latency)
                
                scenario_metrics[f"{scenario['name']}_p95_ms"] = np.percentile(scenario_latencies, 95)
                scenario_metrics[f"{scenario['name']}_mean_ms"] = np.mean(scenario_latencies)
            
            # Determine status based on worst-case performance
            worst_p95 = max([v for k, v in scenario_metrics.items() if 'p95' in k])
            status = 'passed' if worst_p95 <= self.config.latency_targets['cloud_p95'] * 1.2 else 'failed'
            
            return TestResult(
                test_name=test_name,
                status=status,
                metrics=scenario_metrics,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                details=f"Network condition testing completed. Worst P95: {worst_p95:.2f}ms"
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_name,
                status="failed",
                metrics={},
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                errors=[str(e)]
            )
    
    async def run_quick_latency_check(self) -> TestResult:
        """Quick latency check for periodic monitoring"""
        start_time = time.time()
        test_name = "quick_latency_check"
        
        try:
            # Run 10 quick interactions
            latencies = []
            for _ in range(10):
                interaction_start = time.time()
                await self._simulate_voice_interaction()
                latencies.append((time.time() - interaction_start) * 1000)
            
            metrics = {
                'mean_latency_ms': np.mean(latencies),
                'p95_latency_ms': np.percentile(latencies, 95)
            }
            
            status = 'passed' if metrics['p95_latency_ms'] <= self.config.latency_targets['cloud_p95'] else 'failed'
            
            return TestResult(
                test_name=test_name,
                status=status,
                metrics=metrics,
                execution_time=time.time() - start_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_name,
                status="failed",
                metrics={},
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                errors=[str(e)]
            )
    
    # Simulation methods (would integrate with actual components)
    async def _simulate_voice_interaction(self) -> None:
        """Simulate a complete voice interaction"""
        # Simulate processing time
        await asyncio.sleep(np.random.uniform(0.2, 0.6))  # 200-600ms
    
    async def _simulate_stt_processing(self) -> None:
        """Simulate STT processing"""
        await asyncio.sleep(np.random.uniform(0.1, 0.3))  # 100-300ms
    
    async def _simulate_llm_processing(self) -> None:
        """Simulate LLM processing"""
        await asyncio.sleep(np.random.uniform(0.2, 0.5))  # 200-500ms
    
    async def _simulate_tts_processing(self) -> None:
        """Simulate TTS processing"""
        await asyncio.sleep(np.random.uniform(0.05, 0.15))  # 50-150ms
    
    async def _simulate_voice_interaction_with_network_delay(self, delay_ms: float) -> None:
        """Simulate voice interaction with network delay"""
        await asyncio.sleep(delay_ms / 1000.0)  # Add network delay
        await self._simulate_voice_interaction()
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of latency distribution"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean([(x - mean) ** 3 for x in data]) / (std ** 3)


# We'll continue with the other test suites in the next part...
# This is the first part of the comprehensive performance testing framework

def create_performance_test_framework(metrics_collector: Optional[MetricsCollector] = None,
                                    config: Optional[PerformanceTestConfig] = None) -> PerformanceTestFramework:
    """Factory function to create performance test framework"""
    return PerformanceTestFramework(metrics_collector, config) 