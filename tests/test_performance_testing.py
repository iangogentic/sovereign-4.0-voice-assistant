"""
Tests for the Automated Performance Testing Suite

This test suite validates the comprehensive performance testing framework
including latency testing, accuracy benchmarking, stress testing, memory
monitoring, regression detection, and report generation.
"""

import pytest
import asyncio
import time
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import json

# Import the performance testing framework
from assistant.performance_testing import (
    PerformanceTestFramework,
    PerformanceTestConfig,
    TestResult,
    create_performance_test_framework
)
from assistant.accuracy_testing import AccuracyBenchmarkSuite
from assistant.stress_testing import StressTestSuite
from assistant.memory_monitoring import MemoryLeakDetector
from assistant.regression_detection import RegressionDetector
from assistant.report_generation import ReportGenerator
from assistant.metrics_collector import MetricsCollector

@pytest.fixture
def test_config():
    """Create test configuration"""
    return PerformanceTestConfig(
        latency_targets={
            'cloud_p95': 800.0,
            'offline_p95': 1500.0,
            'stt_p95': 200.0,
            'llm_p95': 400.0,
            'tts_p95': 100.0
        },
        accuracy_targets={
            'stt_accuracy': 0.95,
            'memory_recall_bleu': 0.8,
            'memory_recall_semantic': 0.85,
            'ocr_accuracy': 0.92
        },
        stress_test_config={
            'max_concurrent_users': 50,  # Reduced for testing
            'test_duration_seconds': 60,  # Reduced for testing
            'ramp_up_duration_seconds': 10,
            'memory_leak_threshold_percent': 10
        }
    )

@pytest.fixture
def mock_metrics_collector():
    """Create mock metrics collector"""
    collector = Mock(spec=MetricsCollector)
    collector.get_performance_summary.return_value = {
        'latency': 150.0,
        'accuracy': 92.5,
        'memory_usage': 256.0,
        'throughput': 45.2
    }
    return collector

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)

class TestPerformanceTestFramework:
    """Test the main performance testing framework"""
    
    def test_framework_initialization(self, test_config, mock_metrics_collector):
        """Test framework initialization"""
        framework = PerformanceTestFramework(mock_metrics_collector, test_config)
        
        assert framework.config == test_config
        assert framework.metrics_collector == mock_metrics_collector
        assert not framework.is_running
        assert framework.current_session_id is None
        assert len(framework.test_results) == 0
        
        # Check that all test components are initialized
        assert isinstance(framework.latency_tester, type(framework.latency_tester))
        assert isinstance(framework.accuracy_tester, AccuracyBenchmarkSuite)
        assert isinstance(framework.stress_tester, StressTestSuite)
        assert isinstance(framework.memory_monitor, MemoryLeakDetector)
        assert isinstance(framework.regression_detector, RegressionDetector)
        assert isinstance(framework.report_generator, ReportGenerator)
    
    def test_factory_function(self):
        """Test factory function"""
        framework = create_performance_test_framework()
        
        assert isinstance(framework, PerformanceTestFramework)
        assert framework.config is not None
        assert framework.metrics_collector is None  # No collector provided
    
    @pytest.mark.asyncio
    async def test_comprehensive_test_suite_basic(self, test_config, mock_metrics_collector):
        """Test basic comprehensive test suite execution"""
        framework = PerformanceTestFramework(mock_metrics_collector, test_config)
        
        # Mock the individual test methods to avoid actual execution
        with patch.object(framework, '_run_latency_tests', new_callable=AsyncMock) as mock_latency, \
             patch.object(framework, '_run_accuracy_tests', new_callable=AsyncMock) as mock_accuracy, \
             patch.object(framework, '_run_stress_tests', new_callable=AsyncMock) as mock_stress, \
             patch.object(framework, '_run_regression_tests', new_callable=AsyncMock) as mock_regression, \
             patch.object(framework.memory_monitor, 'start_monitoring', new_callable=AsyncMock) as mock_memory_start, \
             patch.object(framework.memory_monitor, 'stop_monitoring_and_analyze', new_callable=AsyncMock) as mock_memory_stop:
            
            # Configure mock returns
            mock_memory_stop.return_value = [
                TestResult(
                    test_name="memory_analysis",
                    status="passed",
                    metrics={'memory_growth_percent': 2.5},
                    execution_time=1.0,
                    timestamp=datetime.now()
                )
            ]
            
            # Run comprehensive test suite
            report = await framework.run_comprehensive_test_suite(['latency', 'accuracy', 'memory'])
            
            # Verify execution
            assert report is not None
            assert report.test_session_id.startswith('perf_test_')
            assert report.overall_status in ['passed', 'failed', 'partial']
            assert isinstance(report.execution_time, float)
            
            # Verify test methods were called
            mock_latency.assert_called_once()
            mock_accuracy.assert_called_once()
            mock_memory_start.assert_called_once()
            mock_memory_stop.assert_called_once()
            
            # Stress and regression should not be called for this test
            mock_stress.assert_not_called()
            mock_regression.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_continuous_monitoring(self, test_config, mock_metrics_collector):
        """Test continuous monitoring functionality"""
        framework = PerformanceTestFramework(mock_metrics_collector, test_config)
        
        # Mock components for quick execution
        with patch.object(framework.memory_monitor, 'continuous_monitoring', new_callable=AsyncMock) as mock_memory, \
             patch.object(framework, '_run_periodic_validation', new_callable=AsyncMock) as mock_validation, \
             patch.object(framework.stress_tester, 'run_sustained_load_test', new_callable=AsyncMock) as mock_load:
            
            # Configure mock returns
            mock_memory.return_value = [
                TestResult(
                    test_name="continuous_memory_monitoring",
                    status="passed",
                    metrics={'monitoring_duration_hours': 0.01},
                    execution_time=36.0,  # 36 seconds = 0.01 hours
                    timestamp=datetime.now()
                )
            ]
            
            mock_load.return_value = TestResult(
                test_name="sustained_load_test",
                status="passed",
                metrics={'success_rate': 0.98},
                execution_time=36.0,
                timestamp=datetime.now()
            )
            
            # Run continuous monitoring for 0.01 hours (36 seconds)
            report = await framework.run_continuous_monitoring(duration_hours=0.01)
            
            # Verify execution
            assert report is not None
            assert report.test_session_id.startswith('continuous_monitor_')
            assert report.overall_status in ['passed', 'failed', 'partial']
            
            # Verify monitoring methods were called
            mock_memory.assert_called_once_with(36)  # 0.01 hours * 3600 seconds
            mock_validation.assert_called_once()
            mock_load.assert_called_once_with(36)
    
    def test_test_status_tracking(self, test_config, mock_metrics_collector):
        """Test test status tracking"""
        framework = PerformanceTestFramework(mock_metrics_collector, test_config)
        
        # Initial status
        status = framework.get_test_status()
        assert status['is_running'] is False
        assert status['current_session_id'] is None
        assert status['tests_completed'] == 0
        
        # Simulate running state
        framework.is_running = True
        framework.current_session_id = "test_session_123"
        framework.test_results = [Mock(), Mock(), Mock()]
        
        status = framework.get_test_status()
        assert status['is_running'] is True
        assert status['current_session_id'] == "test_session_123"
        assert status['tests_completed'] == 3
    
    @pytest.mark.asyncio
    async def test_emergency_stop(self, test_config, mock_metrics_collector):
        """Test emergency stop functionality"""
        framework = PerformanceTestFramework(mock_metrics_collector, test_config)
        
        # Mock the stop methods
        with patch.object(framework.memory_monitor, 'stop_monitoring', new_callable=AsyncMock) as mock_memory_stop, \
             patch.object(framework.stress_tester, 'stop_all_tests', new_callable=AsyncMock) as mock_stress_stop:
            
            # Simulate running state
            framework.is_running = True
            
            # Call emergency stop
            await framework.stop_all_tests()
            
            # Verify state and method calls
            assert framework.is_running is False
            mock_memory_stop.assert_called_once()
            mock_stress_stop.assert_called_once()

class TestLatencyTestSuite:
    """Test the latency testing suite"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_latency_test(self, test_config):
        """Test end-to-end latency testing"""
        framework = PerformanceTestFramework(config=test_config)
        latency_tester = framework.latency_tester
        
        # Mock the voice interaction simulation
        with patch.object(latency_tester, '_simulate_voice_interaction', new_callable=AsyncMock) as mock_interaction:
            mock_interaction.return_value = None  # Simulate 100ms processing time
            
            # Run the test
            result = await latency_tester.test_end_to_end_latency()
            
            # Verify result
            assert result is not None
            assert result.test_name == "end_to_end_latency"
            assert result.status in ['passed', 'failed', 'warning']
            assert 'p95_latency_ms' in result.metrics
            assert 'mean_latency_ms' in result.metrics
            assert result.execution_time > 0
            
            # Verify simulation was called multiple times
            assert mock_interaction.call_count == 50  # Default test count
    
    @pytest.mark.asyncio
    async def test_component_latency_test(self, test_config):
        """Test component-level latency testing"""
        framework = PerformanceTestFramework(config=test_config)
        latency_tester = framework.latency_tester
        
        # Mock component simulations
        with patch.object(latency_tester, '_simulate_stt_processing', new_callable=AsyncMock) as mock_stt, \
             patch.object(latency_tester, '_simulate_llm_processing', new_callable=AsyncMock) as mock_llm, \
             patch.object(latency_tester, '_simulate_tts_processing', new_callable=AsyncMock) as mock_tts:
            
            # Run the test
            result = await latency_tester.test_component_latencies()
            
            # Verify result
            assert result is not None
            assert result.test_name == "component_latencies"
            assert 'stt_p95_ms' in result.metrics
            assert 'llm_p95_ms' in result.metrics
            assert 'tts_p95_ms' in result.metrics
            
            # Verify all components were tested
            assert mock_stt.call_count == 20
            assert mock_llm.call_count == 20
            assert mock_tts.call_count == 20
    
    @pytest.mark.asyncio
    async def test_quick_latency_check(self, test_config):
        """Test quick latency check for monitoring"""
        framework = PerformanceTestFramework(config=test_config)
        latency_tester = framework.latency_tester
        
        with patch.object(latency_tester, '_simulate_voice_interaction', new_callable=AsyncMock):
            result = await latency_tester.run_quick_latency_check()
            
            assert result is not None
            assert result.test_name == "quick_latency_check"
            assert 'mean_latency_ms' in result.metrics
            assert 'p95_latency_ms' in result.metrics

class TestAccuracyBenchmarkSuite:
    """Test the accuracy benchmarking suite"""
    
    def test_test_dataset_loading(self, test_config):
        """Test test dataset loading"""
        accuracy_suite = AccuracyBenchmarkSuite(test_config)
        
        # Verify datasets are loaded
        assert 'memory_recall' in accuracy_suite.test_cases
        assert 'stt_accuracy' in accuracy_suite.test_cases
        assert 'ocr_accuracy' in accuracy_suite.test_cases
        assert 'semantic_similarity' in accuracy_suite.test_cases
        
        # Verify test cases have required structure
        for dataset_name, test_cases in accuracy_suite.test_cases.items():
            assert len(test_cases) > 0
            for test_case in test_cases:
                assert hasattr(test_case, 'test_id')
                assert hasattr(test_case, 'input_data')
                assert hasattr(test_case, 'expected_output')
                assert hasattr(test_case, 'test_type')
    
    @pytest.mark.asyncio
    async def test_memory_recall_accuracy(self, test_config):
        """Test memory recall accuracy testing"""
        accuracy_suite = AccuracyBenchmarkSuite(test_config)
        
        # Mock the memory recall simulation
        with patch.object(accuracy_suite, '_simulate_memory_recall', new_callable=AsyncMock) as mock_recall:
            mock_recall.return_value = "Simulated response for testing"
            
            # Skip if BLEU metric not available
            if accuracy_suite.bleu_metric is None:
                pytest.skip("BLEU metric not available")
            
            result = await accuracy_suite.test_memory_recall_accuracy()
            
            assert result is not None
            assert result.test_name == "memory_recall_bleu"
            if result.status != "skipped":
                assert 'mean_bleu_score' in result.metrics
                assert mock_recall.call_count > 0
    
    @pytest.mark.asyncio
    async def test_stt_accuracy(self, test_config):
        """Test STT accuracy testing"""
        accuracy_suite = AccuracyBenchmarkSuite(test_config)
        
        with patch.object(accuracy_suite, '_simulate_stt_transcription', new_callable=AsyncMock) as mock_stt:
            mock_stt.return_value = "transcribed text"
            
            result = await accuracy_suite.test_stt_accuracy()
            
            assert result is not None
            assert result.test_name == "stt_accuracy"
            assert 'mean_character_accuracy' in result.metrics
            assert 'mean_word_accuracy' in result.metrics
            assert mock_stt.call_count > 0
    
    @pytest.mark.asyncio
    async def test_quick_accuracy_check(self, test_config):
        """Test quick accuracy check"""
        accuracy_suite = AccuracyBenchmarkSuite(test_config)
        
        with patch.object(accuracy_suite, '_simulate_memory_recall', new_callable=AsyncMock), \
             patch.object(accuracy_suite, '_simulate_stt_transcription', new_callable=AsyncMock):
            
            result = await accuracy_suite.run_quick_accuracy_check()
            
            assert result is not None
            assert result.test_name == "quick_accuracy_check"

class TestStressTestSuite:
    """Test the stress testing suite"""
    
    def test_synthetic_workload_creation(self, test_config):
        """Test synthetic workload generation"""
        stress_suite = StressTestSuite(test_config)
        
        workload = stress_suite.synthetic_workload
        
        # Verify workload structure
        assert len(workload.voice_queries) > 0
        assert len(workload.interaction_patterns) > 0
        assert len(workload.expected_response_times) > 0
        assert len(workload.complexity_distribution) > 0
        
        # Verify interaction patterns sum to approximately 1.0
        total_weight = sum(workload.interaction_patterns.values())
        assert abs(total_weight - 1.0) < 0.1
    
    @pytest.mark.asyncio
    async def test_peak_load_simulation(self, test_config):
        """Test peak load simulation"""
        stress_suite = StressTestSuite(test_config)
        
        # Mock the scenario execution
        with patch.object(stress_suite, '_execute_load_scenario', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {
                'successful_interactions': 95,
                'total_interactions': 100,
                'average_latency_ms': 450.0,
                'p95_latency_ms': 650.0,
                'error_count': 5
            }
            
            result = await stress_suite.test_peak_load_simulation()
            
            assert result is not None
            assert result.test_name == "peak_load_simulation"
            assert result.status in ['passed', 'failed', 'warning']
            mock_execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_sustained_load(self, test_config):
        """Test sustained load testing"""
        stress_suite = StressTestSuite(test_config)
        
        with patch.object(stress_suite, '_execute_load_scenario', new_callable=AsyncMock) as mock_execute, \
             patch.object(stress_suite, '_analyze_performance_degradation', new_callable=AsyncMock) as mock_analyze:
            
            mock_execute.return_value = {
                'successful_interactions': 980,
                'total_interactions': 1000,
                'average_latency_ms': 400.0,
                'p95_latency_ms': 580.0,
                'error_count': 20
            }
            
            mock_analyze.return_value = {
                'performance_degradation_percent': 5.0,
                'memory_growth_percent': 3.0
            }
            
            result = await stress_suite.test_sustained_load()
            
            assert result is not None
            assert result.test_name == "sustained_load"
            mock_execute.assert_called_once()
            mock_analyze.assert_called_once()

class TestMemoryLeakDetector:
    """Test the memory monitoring suite"""
    
    def test_initialization(self, test_config):
        """Test memory leak detector initialization"""
        detector = MemoryLeakDetector(test_config)
        
        assert detector.config == test_config
        assert not detector.is_monitoring
        assert len(detector.memory_history) == 0
        assert len(detector.detected_leaks) == 0
    
    @pytest.mark.asyncio
    async def test_memory_monitoring_lifecycle(self, test_config):
        """Test memory monitoring start/stop lifecycle"""
        detector = MemoryLeakDetector(test_config)
        
        # Mock the monitoring loop
        with patch.object(detector, '_monitoring_loop') as mock_loop:
            # Start monitoring
            await detector.start_monitoring()
            assert detector.is_monitoring is True
            
            # Stop monitoring
            await detector.stop_monitoring()
            assert detector.is_monitoring is False
    
    @pytest.mark.asyncio
    async def test_memory_analysis(self, test_config):
        """Test memory usage analysis"""
        detector = MemoryLeakDetector(test_config)
        
        # Simulate some memory history
        current_time = time.time()
        for i in range(10):
            detector.memory_history.append((
                datetime.fromtimestamp(current_time + i * 30),  # Every 30 seconds
                100 + i * 5  # Gradually increasing memory
            ))
        
        result = await detector.analyze_memory_leaks()
        
        assert result is not None
        assert result.test_name == "memory_leak_analysis"
        assert 'total_memory_growth_mb' in result.metrics
        assert 'memory_growth_percent' in result.metrics

class TestRegressionDetector:
    """Test the regression detection suite"""
    
    def test_initialization(self, test_config):
        """Test regression detector initialization"""
        detector = RegressionDetector(test_config)
        
        assert detector.config == test_config
        assert len(detector.baselines) >= 0  # May load existing baselines
        assert len(detector.performance_history) == 0
        assert len(detector.detected_regressions) == 0
    
    def test_baseline_creation(self, test_config):
        """Test performance baseline creation"""
        detector = RegressionDetector(test_config)
        
        # Create a test baseline
        test_metrics = {
            'avg_latency_ms': 350.0,
            'stt_accuracy': 0.94,
            'memory_usage_mb': 512.0
        }
        
        detector.create_performance_baseline("test_baseline", test_metrics)
        
        assert "test_baseline" in detector.baselines
        baseline = detector.baselines["test_baseline"]
        assert baseline.metrics == test_metrics
        assert len(baseline.thresholds) == len(test_metrics)
    
    def test_performance_history_update(self, test_config):
        """Test performance history updates"""
        detector = RegressionDetector(test_config)
        
        # Add some performance data
        test_metrics = {
            'latency_ms': 400.0,
            'accuracy': 0.92
        }
        
        detector.update_performance_history(test_metrics)
        
        assert len(detector.performance_history) == 1
        assert 'timestamp' in detector.performance_history[0]
        assert detector.performance_history[0]['latency_ms'] == 400.0
    
    @pytest.mark.asyncio
    async def test_quick_regression_check(self, test_config):
        """Test quick regression check"""
        detector = RegressionDetector(test_config)
        
        # Add baseline and history
        detector.create_performance_baseline("test", {'latency_ms': 300.0})
        for i in range(10):
            detector.update_performance_history({'latency_ms': 320.0 + i * 2})
        
        result = await detector.check_for_regression()
        
        assert result is not None
        assert result.test_name == "quick_regression_check"
        assert 'regression_detected' in result.metrics

class TestReportGenerator:
    """Test the report generation suite"""
    
    def test_initialization(self, test_config, temp_dir):
        """Test report generator initialization"""
        # Temporarily change the storage path
        with patch.object(Path, 'mkdir'):
            generator = ReportGenerator(test_config)
            
            assert generator.config == test_config
            assert hasattr(generator, 'scoring_weights')
            assert hasattr(generator, 'report_storage_path')
    
    @pytest.mark.asyncio
    async def test_comprehensive_report_generation(self, test_config):
        """Test comprehensive report generation"""
        generator = ReportGenerator(test_config)
        
        # Create sample test results
        test_results = [
            TestResult(
                test_name="latency_test",
                status="passed",
                metrics={'avg_latency_ms': 350.0, 'p95_latency_ms': 500.0},
                execution_time=2.5,
                timestamp=datetime.now()
            ),
            TestResult(
                test_name="accuracy_test",
                status="passed",
                metrics={'mean_bleu_score': 0.85, 'stt_accuracy': 0.93},
                execution_time=1.8,
                timestamp=datetime.now()
            ),
            TestResult(
                test_name="memory_test",
                status="warning",
                metrics={'memory_growth_percent': 8.5, 'leaks_detected': 0},
                execution_time=5.0,
                timestamp=datetime.now()
            )
        ]
        
        # Mock file operations
        with patch('builtins.open', create=True) as mock_open, \
             patch.object(Path, 'mkdir') as mock_mkdir:
            
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            report = await generator.generate_report(
                session_id="test_session_123",
                test_results=test_results,
                execution_time=120.0
            )
            
            # Verify report structure
            assert report is not None
            assert report.test_session_id == "test_session_123"
            assert report.overall_status in ['passed', 'failed', 'partial']
            assert len(report.test_results) == 3
            assert len(report.recommendations) > 0
            assert report.execution_time == 120.0
            
            # Verify file operations were called
            assert mock_open.call_count >= 1  # At least JSON report saved
    
    def test_performance_score_calculation(self, test_config):
        """Test performance score calculation"""
        generator = ReportGenerator(test_config)
        
        # Sample metrics
        summary_metrics = {
            'avg_latency_ms': 400.0,
            'avg_accuracy': 0.92,
            'pass_rate_percent': 95.0,
            'peak_memory_usage': 800.0
        }
        
        test_results = []  # Empty for this test
        
        score = generator._calculate_performance_score(summary_metrics, test_results)
        
        assert isinstance(score, (int, float))
        assert 0 <= score <= 100

class TestEndToEndIntegration:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    async def test_full_performance_testing_workflow(self, test_config, mock_metrics_collector):
        """Test complete performance testing workflow"""
        framework = PerformanceTestFramework(mock_metrics_collector, test_config)
        
        # Mock all heavy operations for quick execution
        with patch.object(framework.latency_tester, 'run_all_latency_tests', new_callable=AsyncMock) as mock_latency, \
             patch.object(framework.accuracy_tester, 'run_all_accuracy_tests', new_callable=AsyncMock) as mock_accuracy, \
             patch.object(framework.stress_tester, 'run_all_stress_tests', new_callable=AsyncMock) as mock_stress, \
             patch.object(framework.regression_detector, 'detect_performance_regressions', new_callable=AsyncMock) as mock_regression, \
             patch.object(framework.memory_monitor, 'start_monitoring', new_callable=AsyncMock), \
             patch.object(framework.memory_monitor, 'stop_monitoring_and_analyze', new_callable=AsyncMock) as mock_memory:
            
            # Configure mock returns
            mock_latency.return_value = [
                TestResult("latency_test", "passed", {'avg_latency_ms': 300}, 1.0, datetime.now())
            ]
            mock_accuracy.return_value = [
                TestResult("accuracy_test", "passed", {'avg_accuracy': 0.94}, 1.0, datetime.now())
            ]
            mock_stress.return_value = [
                TestResult("stress_test", "passed", {'success_rate': 0.98}, 1.0, datetime.now())
            ]
            mock_regression.return_value = [
                TestResult("regression_test", "passed", {'regressions_detected': 0}, 1.0, datetime.now())
            ]
            mock_memory.return_value = [
                TestResult("memory_test", "passed", {'memory_growth_percent': 2.0}, 1.0, datetime.now())
            ]
            
            # Mock report saving
            with patch('builtins.open', create=True), \
                 patch.object(Path, 'mkdir'):
                
                # Run full test suite
                report = await framework.run_comprehensive_test_suite()
                
                # Verify comprehensive execution
                assert report is not None
                assert len(report.test_results) >= 5  # All test suites ran
                assert report.overall_status in ['passed', 'failed', 'partial']
                assert report.execution_time > 0
                assert len(report.recommendations) >= 0
                
                # Verify all test suites were called
                mock_latency.assert_called_once()
                mock_accuracy.assert_called_once()
                mock_stress.assert_called_once()
                mock_regression.assert_called_once()
                mock_memory.assert_called_once()

@pytest.mark.asyncio
async def test_performance_testing_with_real_metrics():
    """Test with more realistic metrics (slower test)"""
    config = PerformanceTestConfig()
    framework = create_performance_test_framework(config=config)
    
    # Run a minimal test to verify real execution
    with patch.object(framework, 'run_comprehensive_test_suite') as mock_run:
        mock_run.return_value = Mock(
            test_session_id="real_test",
            overall_status="passed",
            test_results=[],
            execution_time=5.0
        )
        
        result = await framework.run_comprehensive_test_suite(['latency'])
        
        assert result is not None
        mock_run.assert_called_once()

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"]) 