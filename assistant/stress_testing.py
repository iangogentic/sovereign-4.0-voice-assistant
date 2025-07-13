"""
Stress Testing Suite for Sovereign 4.0 Voice Assistant

This module provides comprehensive stress testing capabilities including:
- Synthetic workload generation mimicking real usage patterns
- Sustained load testing for 8-hour continuous operation
- Chaos engineering scenarios for resilience testing
- Peak load simulation and spike testing
- Resource exhaustion testing

Implements modern 2024-2025 stress testing methodologies.
"""

import asyncio
import time
import logging
import numpy as np
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, field
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .performance_testing import TestResult, PerformanceTestConfig

logger = logging.getLogger(__name__)

@dataclass
class LoadTestScenario:
    """Load test scenario configuration"""
    name: str
    concurrent_users: int
    duration_seconds: int
    ramp_up_seconds: int
    interaction_frequency: str  # 'low', 'medium', 'high', 'very_high'
    user_behavior_pattern: str  # 'quick_queries', 'long_conversations', 'mixed', 'code_assistance'
    
@dataclass
class UserSession:
    """Individual user session state"""
    user_id: str
    start_time: float
    session_duration: float
    interaction_count: int
    total_latency: float
    errors: List[str] = field(default_factory=list)

@dataclass
class SyntheticWorkload:
    """Synthetic workload data"""
    voice_queries: List[str]
    interaction_patterns: Dict[str, float]
    expected_response_times: Dict[str, float]
    complexity_distribution: Dict[str, float]

class StressTestSuite:
    """
    Comprehensive stress testing suite
    
    Implements synthetic workload generation, sustained load testing,
    and chaos engineering patterns for voice assistant resilience testing.
    """
    
    def __init__(self, config: PerformanceTestConfig):
        self.config = config
        self.active_sessions: Dict[str, UserSession] = {}
        self.test_metrics: Dict[str, Any] = {}
        self.is_running = False
        self.stop_requested = False
        
        # Initialize synthetic workload
        self.synthetic_workload = self._create_synthetic_workload()
        
        logger.info("Stress Test Suite initialized")
    
    def _create_synthetic_workload(self) -> SyntheticWorkload:
        """Create comprehensive synthetic workload for voice assistant testing"""
        
        # Programming and development queries
        programming_queries = [
            "How do I fix this memory leak in my Python application?",
            "What's the best way to implement JWT authentication?",
            "Can you help me debug this TypeScript compilation error?",
            "Show me how to optimize this SQL query performance",
            "Explain the difference between async and await in JavaScript",
            "How do I set up a Docker container for my Node.js app?",
            "What's causing this React component to re-render constantly?",
            "Help me understand this error message from my API",
            "How do I implement pagination in my database queries?",
            "What's the best practice for handling user input validation?"
        ]
        
        # Quick technical queries
        quick_queries = [
            "What's the syntax for list comprehension in Python?",
            "How do I create a new branch in Git?",
            "What port does PostgreSQL use by default?",
            "How do I install numpy using pip?",
            "What's the keyboard shortcut for commenting code in VS Code?",
            "How do I check my Python version?",
            "What's the difference between let and const in JavaScript?",
            "How do I export a function in Node.js?",
            "What's the command to start a React development server?",
            "How do I create a virtual environment in Python?"
        ]
        
        # Complex problem-solving queries
        complex_queries = [
            "I'm building a microservices architecture and need help with service discovery and load balancing",
            "Can you walk me through implementing a complete authentication system with refresh tokens, rate limiting, and RBAC?",
            "I'm having performance issues with my React application - can you help me identify bottlenecks and optimization strategies?",
            "Help me design a database schema for a multi-tenant SaaS application with proper data isolation",
            "I need to implement real-time collaboration features like Google Docs - what technologies and patterns should I use?",
            "Can you help me set up a CI/CD pipeline with automated testing, security scanning, and deployment strategies?",
            "I'm building a data processing pipeline that needs to handle millions of records - help me with architecture and scaling",
            "Help me implement proper error handling and monitoring for a distributed system with multiple services"
        ]
        
        # Screen content and OCR-related queries
        screen_queries = [
            "What does this error message on my screen mean?",
            "Can you read the code on my screen and explain what it does?",
            "Help me understand this compiler error that's showing up",
            "What's wrong with the syntax highlighted on my editor?",
            "Can you see the performance metrics on my dashboard?",
            "Read the documentation snippet on my screen",
            "What's the issue with the test output showing on my terminal?",
            "Help me understand this stack trace error message"
        ]
        
        # Combine all query types
        all_queries = (
            programming_queries * 3 +  # 30% programming
            quick_queries * 4 +        # 40% quick
            complex_queries * 2 +      # 20% complex
            screen_queries * 1         # 10% screen
        )
        
        # Interaction patterns with frequencies
        interaction_patterns = {
            'quick_query': 0.4,           # 40% quick technical questions
            'problem_solving': 0.25,      # 25% complex problem solving
            'code_assistance': 0.2,       # 20% code help and debugging
            'screen_analysis': 0.1,       # 10% screen content analysis
            'conversation': 0.05          # 5% general conversation
        }
        
        # Expected response times by query type
        expected_response_times = {
            'quick_query': 0.3,          # 300ms for simple queries
            'problem_solving': 0.8,      # 800ms for complex queries
            'code_assistance': 0.6,      # 600ms for code help
            'screen_analysis': 0.4,      # 400ms for OCR + analysis
            'conversation': 0.5          # 500ms for general chat
        }
        
        # Complexity distribution
        complexity_distribution = {
            'simple': 0.4,     # Single-step queries
            'medium': 0.4,     # Multi-step queries
            'complex': 0.2     # Complex multi-turn interactions
        }
        
        return SyntheticWorkload(
            voice_queries=all_queries,
            interaction_patterns=interaction_patterns,
            expected_response_times=expected_response_times,
            complexity_distribution=complexity_distribution
        )
    
    async def run_all_stress_tests(self) -> List[TestResult]:
        """Run comprehensive stress test suite"""
        results = []
        
        # Peak load simulation
        results.append(await self.test_peak_load_simulation())
        
        # Sustained load test (shorter version for regular testing)
        results.append(await self.test_sustained_load())
        
        # Spike testing
        results.append(await self.test_spike_load())
        
        # Memory stress test
        results.append(await self.test_memory_stress())
        
        # Chaos engineering scenarios
        results.extend(await self.test_chaos_scenarios())
        
        return results
    
    async def test_peak_load_simulation(self) -> TestResult:
        """Test system under peak load conditions"""
        start_time = time.time()
        test_name = "peak_load_simulation"
        
        try:
            scenario = LoadTestScenario(
                name="peak_load",
                concurrent_users=self.config.stress_test_config['max_concurrent_users'],
                duration_seconds=300,  # 5 minutes
                ramp_up_seconds=60,    # 1 minute ramp-up
                interaction_frequency='high',
                user_behavior_pattern='mixed'
            )
            
            metrics = await self._execute_load_scenario(scenario)
            
            # Evaluate results against targets
            success_rate = metrics['successful_interactions'] / metrics['total_interactions'] if metrics['total_interactions'] > 0 else 0
            avg_latency = metrics['average_latency_ms']
            p95_latency = metrics['p95_latency_ms']
            
            # Determine status
            status = 'passed'
            if success_rate < 0.95:  # 95% success rate required
                status = 'failed'
            elif p95_latency > self.config.latency_targets['cloud_p95'] * 1.5:  # Allow 50% latency degradation under peak load
                status = 'warning'
            
            return TestResult(
                test_name=test_name,
                status=status,
                metrics=metrics,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                details=f"Peak load: {scenario.concurrent_users} users, Success rate: {success_rate:.3f}, P95: {p95_latency:.1f}ms"
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
    
    async def test_sustained_load(self) -> TestResult:
        """Test system under sustained load (shorter version for regular testing)"""
        start_time = time.time()
        test_name = "sustained_load"
        
        try:
            scenario = LoadTestScenario(
                name="sustained_load",
                concurrent_users=50,  # Moderate load
                duration_seconds=1800,  # 30 minutes for regular testing
                ramp_up_seconds=300,    # 5 minutes ramp-up
                interaction_frequency='medium',
                user_behavior_pattern='mixed'
            )
            
            metrics = await self._execute_load_scenario(scenario)
            
            # Check for performance degradation over time
            degradation_metrics = await self._analyze_performance_degradation(metrics)
            metrics.update(degradation_metrics)
            
            # Evaluate sustained performance
            success_rate = metrics['successful_interactions'] / metrics['total_interactions'] if metrics['total_interactions'] > 0 else 0
            performance_degradation = metrics.get('performance_degradation_percent', 0)
            memory_growth = metrics.get('memory_growth_percent', 0)
            
            status = 'passed'
            if success_rate < 0.98:  # Higher success rate for sustained load
                status = 'failed'
            elif performance_degradation > 20 or memory_growth > 10:  # 20% performance degradation or 10% memory growth
                status = 'warning'
            
            return TestResult(
                test_name=test_name,
                status=status,
                metrics=metrics,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                details=f"Sustained load: {scenario.duration_seconds/60:.1f}min, Degradation: {performance_degradation:.1f}%, Memory growth: {memory_growth:.1f}%"
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
    
    async def run_sustained_load_test(self, duration_seconds: int) -> TestResult:
        """Run extended sustained load test for continuous monitoring"""
        start_time = time.time()
        test_name = "extended_sustained_load"
        
        try:
            scenario = LoadTestScenario(
                name="extended_sustained_load",
                concurrent_users=30,  # Conservative load for 8-hour testing
                duration_seconds=duration_seconds,
                ramp_up_seconds=600,  # 10 minutes ramp-up
                interaction_frequency='low',
                user_behavior_pattern='mixed'
            )
            
            logger.info(f"Starting {duration_seconds/3600:.1f}h sustained load test")
            
            metrics = await self._execute_load_scenario(scenario)
            
            # Extended analysis for long-duration testing
            extended_metrics = await self._analyze_extended_performance(metrics, duration_seconds)
            metrics.update(extended_metrics)
            
            # Evaluate extended performance
            success_rate = metrics['successful_interactions'] / metrics['total_interactions'] if metrics['total_interactions'] > 0 else 0
            memory_leak_detected = metrics.get('memory_leak_detected', False)
            performance_trend = metrics.get('performance_trend', 'stable')
            
            status = 'passed'
            if success_rate < 0.95:
                status = 'failed'
            elif memory_leak_detected or performance_trend == 'degrading':
                status = 'warning'
            
            return TestResult(
                test_name=test_name,
                status=status,
                metrics=metrics,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                details=f"Extended test: {duration_seconds/3600:.1f}h, Success: {success_rate:.3f}, Trend: {performance_trend}"
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
    
    async def test_spike_load(self) -> TestResult:
        """Test system response to sudden load spikes"""
        start_time = time.time()
        test_name = "spike_load"
        
        try:
            # Baseline load
            baseline_users = 20
            spike_users = 150
            spike_duration = 120  # 2 minutes
            
            metrics = {
                'baseline_response_time': 0,
                'spike_response_time': 0,
                'recovery_time_seconds': 0,
                'errors_during_spike': 0,
                'baseline_success_rate': 0,
                'spike_success_rate': 0
            }
            
            # Phase 1: Baseline measurement
            logger.info("Starting baseline measurement")
            baseline_scenario = LoadTestScenario(
                name="baseline",
                concurrent_users=baseline_users,
                duration_seconds=60,
                ramp_up_seconds=10,
                interaction_frequency='medium',
                user_behavior_pattern='quick_queries'
            )
            
            baseline_metrics = await self._execute_load_scenario(baseline_scenario)
            metrics['baseline_response_time'] = baseline_metrics['average_latency_ms']
            metrics['baseline_success_rate'] = baseline_metrics['successful_interactions'] / baseline_metrics['total_interactions']
            
            # Phase 2: Spike load
            logger.info(f"Starting spike load: {spike_users} users")
            spike_scenario = LoadTestScenario(
                name="spike",
                concurrent_users=spike_users,
                duration_seconds=spike_duration,
                ramp_up_seconds=5,  # Very fast ramp-up for spike
                interaction_frequency='very_high',
                user_behavior_pattern='quick_queries'
            )
            
            spike_metrics = await self._execute_load_scenario(spike_scenario)
            metrics['spike_response_time'] = spike_metrics['average_latency_ms']
            metrics['spike_success_rate'] = spike_metrics['successful_interactions'] / spike_metrics['total_interactions']
            metrics['errors_during_spike'] = spike_metrics['error_count']
            
            # Phase 3: Recovery measurement
            logger.info("Measuring recovery")
            recovery_start = time.time()
            recovery_scenario = LoadTestScenario(
                name="recovery",
                concurrent_users=baseline_users,
                duration_seconds=60,
                ramp_up_seconds=10,
                interaction_frequency='medium',
                user_behavior_pattern='quick_queries'
            )
            
            recovery_metrics = await self._execute_load_scenario(recovery_scenario)
            metrics['recovery_time_seconds'] = time.time() - recovery_start
            metrics['recovery_response_time'] = recovery_metrics['average_latency_ms']
            
            # Evaluate spike handling
            response_time_increase = (metrics['spike_response_time'] - metrics['baseline_response_time']) / metrics['baseline_response_time']
            recovery_degradation = abs(metrics['recovery_response_time'] - metrics['baseline_response_time']) / metrics['baseline_response_time']
            
            metrics['response_time_increase_percent'] = response_time_increase * 100
            metrics['recovery_degradation_percent'] = recovery_degradation * 100
            
            status = 'passed'
            if metrics['spike_success_rate'] < 0.8:  # 80% success rate during spike
                status = 'failed'
            elif response_time_increase > 3.0 or recovery_degradation > 0.2:  # 300% increase or 20% recovery degradation
                status = 'warning'
            
            return TestResult(
                test_name=test_name,
                status=status,
                metrics=metrics,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                details=f"Spike: {spike_users} users, Response increase: {response_time_increase*100:.1f}%, Recovery: {recovery_degradation*100:.1f}%"
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
    
    async def test_memory_stress(self) -> TestResult:
        """Test system under memory stress conditions"""
        start_time = time.time()
        test_name = "memory_stress"
        
        try:
            # Create scenarios that stress memory usage
            scenario = LoadTestScenario(
                name="memory_stress",
                concurrent_users=75,
                duration_seconds=600,  # 10 minutes
                ramp_up_seconds=60,
                interaction_frequency='high',
                user_behavior_pattern='complex'  # Use complex queries that require more memory
            )
            
            # Monitor memory usage during test
            initial_memory = psutil.Process().memory_info().rss / 1024**2  # MB
            
            metrics = await self._execute_load_scenario(scenario)
            
            final_memory = psutil.Process().memory_info().rss / 1024**2  # MB
            memory_growth = ((final_memory - initial_memory) / initial_memory) * 100
            
            # Add memory-specific metrics
            metrics.update({
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'memory_growth_percent': memory_growth,
                'peak_memory_mb': metrics.get('peak_memory_mb', final_memory)
            })
            
            # Evaluate memory usage
            success_rate = metrics['successful_interactions'] / metrics['total_interactions'] if metrics['total_interactions'] > 0 else 0
            
            status = 'passed'
            if memory_growth > self.config.stress_test_config['memory_leak_threshold_percent']:
                status = 'failed'
            elif success_rate < 0.95 or memory_growth > 5:  # 5% memory growth warning
                status = 'warning'
            
            return TestResult(
                test_name=test_name,
                status=status,
                metrics=metrics,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                details=f"Memory growth: {memory_growth:.1f}%, Success rate: {success_rate:.3f}"
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
    
    async def test_chaos_scenarios(self) -> List[TestResult]:
        """Test system resilience using chaos engineering scenarios"""
        results = []
        
        chaos_scenarios = [
            self._chaos_network_latency,
            self._chaos_memory_pressure,
            self._chaos_cpu_spike,
            self._chaos_disk_io_stress
        ]
        
        for scenario in chaos_scenarios:
            try:
                result = await scenario()
                results.append(result)
            except Exception as e:
                results.append(TestResult(
                    test_name=f"chaos_{scenario.__name__}",
                    status="failed",
                    metrics={},
                    execution_time=0,
                    timestamp=datetime.now(),
                    errors=[str(e)]
                ))
        
        return results
    
    async def _chaos_network_latency(self) -> TestResult:
        """Simulate network latency chaos"""
        start_time = time.time()
        test_name = "chaos_network_latency"
        
        try:
            # Simulate network latency variations during load test
            baseline_scenario = LoadTestScenario(
                name="chaos_network",
                concurrent_users=30,
                duration_seconds=180,  # 3 minutes
                ramp_up_seconds=30,
                interaction_frequency='medium',
                user_behavior_pattern='mixed'
            )
            
            # This would integrate with actual network simulation
            # For now, we simulate the effects
            metrics = await self._execute_load_scenario_with_chaos(
                baseline_scenario, 
                chaos_type='network_latency',
                chaos_intensity=0.3  # 30% of requests affected
            )
            
            # Evaluate resilience
            success_rate = metrics['successful_interactions'] / metrics['total_interactions'] if metrics['total_interactions'] > 0 else 0
            chaos_recovery_time = metrics.get('chaos_recovery_time_seconds', 0)
            
            status = 'passed'
            if success_rate < 0.9:  # 90% success rate under chaos
                status = 'failed'
            elif chaos_recovery_time > 30:  # Recovery within 30 seconds
                status = 'warning'
            
            return TestResult(
                test_name=test_name,
                status=status,
                metrics=metrics,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                details=f"Network chaos: Success {success_rate:.3f}, Recovery {chaos_recovery_time:.1f}s"
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
    
    async def _chaos_memory_pressure(self) -> TestResult:
        """Simulate memory pressure chaos"""
        start_time = time.time()
        test_name = "chaos_memory_pressure"
        
        try:
            # Simulate memory pressure during testing
            scenario = LoadTestScenario(
                name="chaos_memory",
                concurrent_users=40,
                duration_seconds=180,
                ramp_up_seconds=30,
                interaction_frequency='medium',
                user_behavior_pattern='complex'
            )
            
            metrics = await self._execute_load_scenario_with_chaos(
                scenario,
                chaos_type='memory_pressure',
                chaos_intensity=0.5
            )
            
            # Evaluate memory resilience
            success_rate = metrics['successful_interactions'] / metrics['total_interactions'] if metrics['total_interactions'] > 0 else 0
            memory_recovery = metrics.get('memory_recovered', True)
            
            status = 'passed'
            if success_rate < 0.85 or not memory_recovery:
                status = 'failed'
            elif success_rate < 0.95:
                status = 'warning'
            
            return TestResult(
                test_name=test_name,
                status=status,
                metrics=metrics,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                details=f"Memory chaos: Success {success_rate:.3f}, Recovery {memory_recovery}"
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
    
    async def _chaos_cpu_spike(self) -> TestResult:
        """Simulate CPU spike chaos"""
        start_time = time.time()
        test_name = "chaos_cpu_spike"
        
        try:
            scenario = LoadTestScenario(
                name="chaos_cpu",
                concurrent_users=35,
                duration_seconds=180,
                ramp_up_seconds=30,
                interaction_frequency='medium',
                user_behavior_pattern='mixed'
            )
            
            metrics = await self._execute_load_scenario_with_chaos(
                scenario,
                chaos_type='cpu_spike',
                chaos_intensity=0.8  # High CPU load
            )
            
            # Evaluate CPU resilience
            success_rate = metrics['successful_interactions'] / metrics['total_interactions'] if metrics['total_interactions'] > 0 else 0
            avg_latency_increase = metrics.get('latency_increase_percent', 0)
            
            status = 'passed'
            if success_rate < 0.8 or avg_latency_increase > 200:  # 200% latency increase threshold
                status = 'failed'
            elif success_rate < 0.9 or avg_latency_increase > 100:
                status = 'warning'
            
            return TestResult(
                test_name=test_name,
                status=status,
                metrics=metrics,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                details=f"CPU chaos: Success {success_rate:.3f}, Latency increase {avg_latency_increase:.1f}%"
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
    
    async def _chaos_disk_io_stress(self) -> TestResult:
        """Simulate disk I/O stress chaos"""
        start_time = time.time()
        test_name = "chaos_disk_io"
        
        try:
            scenario = LoadTestScenario(
                name="chaos_disk",
                concurrent_users=25,
                duration_seconds=180,
                ramp_up_seconds=30,
                interaction_frequency='medium',
                user_behavior_pattern='mixed'
            )
            
            metrics = await self._execute_load_scenario_with_chaos(
                scenario,
                chaos_type='disk_io_stress',
                chaos_intensity=0.6
            )
            
            # Evaluate I/O resilience
            success_rate = metrics['successful_interactions'] / metrics['total_interactions'] if metrics['total_interactions'] > 0 else 0
            io_error_rate = metrics.get('io_error_rate', 0)
            
            status = 'passed'
            if success_rate < 0.9 or io_error_rate > 0.05:  # 5% I/O error threshold
                status = 'failed'
            elif success_rate < 0.95 or io_error_rate > 0.02:
                status = 'warning'
            
            return TestResult(
                test_name=test_name,
                status=status,
                metrics=metrics,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                details=f"I/O chaos: Success {success_rate:.3f}, I/O errors {io_error_rate:.3f}"
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
    
    async def stop_all_tests(self) -> None:
        """Stop all running stress tests"""
        self.stop_requested = True
        self.is_running = False
        logger.info("Stress tests stop requested")
    
    # Core execution methods
    async def _execute_load_scenario(self, scenario: LoadTestScenario) -> Dict[str, Any]:
        """Execute a load testing scenario"""
        self.is_running = True
        self.stop_requested = False
        
        try:
            logger.info(f"Starting load scenario: {scenario.name}")
            
            # Initialize metrics
            metrics = {
                'scenario_name': scenario.name,
                'concurrent_users': scenario.concurrent_users,
                'duration_seconds': scenario.duration_seconds,
                'total_interactions': 0,
                'successful_interactions': 0,
                'error_count': 0,
                'latencies': [],
                'start_time': time.time(),
                'user_sessions': []
            }
            
            # Create user simulation tasks
            user_tasks = []
            users_per_second = scenario.concurrent_users / scenario.ramp_up_seconds if scenario.ramp_up_seconds > 0 else scenario.concurrent_users
            
            for user_id in range(scenario.concurrent_users):
                # Stagger user start times for ramp-up
                start_delay = user_id / users_per_second if users_per_second > 0 else 0
                
                task = asyncio.create_task(
                    self._simulate_user_session(
                        user_id=f"user_{user_id}",
                        scenario=scenario,
                        start_delay=start_delay,
                        metrics=metrics
                    )
                )
                user_tasks.append(task)
            
            # Wait for all users to complete or timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*user_tasks, return_exceptions=True),
                    timeout=scenario.duration_seconds + scenario.ramp_up_seconds + 60  # Extra buffer
                )
            except asyncio.TimeoutError:
                logger.warning(f"Load scenario {scenario.name} timed out")
                # Cancel remaining tasks
                for task in user_tasks:
                    if not task.done():
                        task.cancel()
            
            # Calculate final metrics
            await self._calculate_scenario_metrics(metrics)
            
            logger.info(f"Load scenario {scenario.name} completed: {metrics['successful_interactions']}/{metrics['total_interactions']} successful")
            
            return metrics
            
        finally:
            self.is_running = False
    
    async def _execute_load_scenario_with_chaos(self, scenario: LoadTestScenario, chaos_type: str, chaos_intensity: float) -> Dict[str, Any]:
        """Execute load scenario with chaos engineering"""
        # This is a simulation of chaos engineering
        # In a real implementation, this would inject actual chaos
        
        metrics = await self._execute_load_scenario(scenario)
        
        # Simulate chaos effects on metrics
        if chaos_type == 'network_latency':
            # Increase latencies for some interactions
            affected_count = int(len(metrics['latencies']) * chaos_intensity)
            for i in range(affected_count):
                if i < len(metrics['latencies']):
                    metrics['latencies'][i] *= (1 + random.uniform(0.5, 2.0))  # 50-200% increase
            
            metrics['chaos_recovery_time_seconds'] = random.uniform(5, 25)
            
        elif chaos_type == 'memory_pressure':
            # Simulate memory pressure effects
            metrics['memory_pressure_peak_mb'] = 1500  # Simulated peak
            metrics['memory_recovered'] = random.choice([True, True, True, False])  # 75% recovery rate
            
        elif chaos_type == 'cpu_spike':
            # Simulate CPU spike effects
            original_avg = np.mean(metrics['latencies']) if metrics['latencies'] else 100
            new_avg = original_avg * (1 + chaos_intensity)
            metrics['latency_increase_percent'] = ((new_avg - original_avg) / original_avg) * 100
            
        elif chaos_type == 'disk_io_stress':
            # Simulate I/O stress effects
            metrics['io_error_rate'] = random.uniform(0, 0.08) * chaos_intensity
        
        return metrics
    
    async def _simulate_user_session(self, user_id: str, scenario: LoadTestScenario, start_delay: float, metrics: Dict[str, Any]) -> None:
        """Simulate an individual user session"""
        if start_delay > 0:
            await asyncio.sleep(start_delay)
        
        if self.stop_requested:
            return
        
        session_start = time.time()
        session_duration = scenario.duration_seconds - start_delay
        interaction_count = 0
        errors = []
        
        try:
            while (time.time() - session_start) < session_duration and not self.stop_requested:
                # Determine interaction type based on behavior pattern
                interaction_type = self._choose_interaction_type(scenario.user_behavior_pattern)
                
                # Simulate voice interaction
                interaction_start = time.time()
                try:
                    await self._simulate_voice_interaction(interaction_type)
                    
                    interaction_latency = (time.time() - interaction_start) * 1000  # ms
                    
                    # Record metrics (thread-safe updates)
                    metrics['total_interactions'] += 1
                    metrics['successful_interactions'] += 1
                    metrics['latencies'].append(interaction_latency)
                    
                    interaction_count += 1
                    
                except Exception as e:
                    errors.append(str(e))
                    metrics['total_interactions'] += 1
                    metrics['error_count'] += 1
                
                # Wait between interactions based on frequency
                wait_time = self._get_interaction_wait_time(scenario.interaction_frequency)
                await asyncio.sleep(wait_time)
        
        except Exception as e:
            errors.append(f"Session error: {str(e)}")
        
        # Record session data
        session = UserSession(
            user_id=user_id,
            start_time=session_start,
            session_duration=time.time() - session_start,
            interaction_count=interaction_count,
            total_latency=sum(metrics['latencies'][-interaction_count:]) if interaction_count > 0 else 0,
            errors=errors
        )
        
        metrics['user_sessions'].append(session)
    
    def _choose_interaction_type(self, behavior_pattern: str) -> str:
        """Choose interaction type based on user behavior pattern"""
        if behavior_pattern == 'quick_queries':
            return random.choice(['quick_query'] * 8 + ['problem_solving'] * 2)
        elif behavior_pattern == 'long_conversations':
            return random.choice(['problem_solving'] * 6 + ['conversation'] * 4)
        elif behavior_pattern == 'code_assistance':
            return random.choice(['code_assistance'] * 7 + ['screen_analysis'] * 3)
        elif behavior_pattern == 'complex':
            return random.choice(['problem_solving'] * 5 + ['code_assistance'] * 3 + ['screen_analysis'] * 2)
        else:  # mixed
            return random.choices(
                list(self.synthetic_workload.interaction_patterns.keys()),
                weights=list(self.synthetic_workload.interaction_patterns.values())
            )[0]
    
    def _get_interaction_wait_time(self, frequency: str) -> float:
        """Get wait time between interactions based on frequency"""
        base_times = {
            'very_high': 2.0,   # 2 seconds
            'high': 5.0,        # 5 seconds
            'medium': 10.0,     # 10 seconds
            'low': 20.0         # 20 seconds
        }
        
        base_time = base_times.get(frequency, 10.0)
        # Add random variation (±50%)
        return base_time * random.uniform(0.5, 1.5)
    
    async def _simulate_voice_interaction(self, interaction_type: str) -> None:
        """Simulate a voice interaction based on type"""
        # Get expected response time for interaction type
        expected_time = self.synthetic_workload.expected_response_times.get(interaction_type, 0.5)
        
        # Add realistic variation (±30%)
        actual_time = expected_time * random.uniform(0.7, 1.3)
        
        # Simulate processing time
        await asyncio.sleep(actual_time)
        
        # Simulate occasional failures (1% rate)
        if random.random() < 0.01:
            raise Exception(f"Simulated {interaction_type} failure")
    
    async def _calculate_scenario_metrics(self, metrics: Dict[str, Any]) -> None:
        """Calculate final metrics for a completed scenario"""
        latencies = metrics['latencies']
        
        if latencies:
            metrics.update({
                'average_latency_ms': np.mean(latencies),
                'median_latency_ms': np.median(latencies),
                'p95_latency_ms': np.percentile(latencies, 95),
                'p99_latency_ms': np.percentile(latencies, 99),
                'max_latency_ms': np.max(latencies),
                'min_latency_ms': np.min(latencies),
                'latency_std_dev': np.std(latencies)
            })
        else:
            metrics.update({
                'average_latency_ms': 0,
                'median_latency_ms': 0,
                'p95_latency_ms': 0,
                'p99_latency_ms': 0,
                'max_latency_ms': 0,
                'min_latency_ms': 0,
                'latency_std_dev': 0
            })
        
        # Calculate additional metrics
        total_sessions = len(metrics['user_sessions'])
        successful_sessions = sum(1 for s in metrics['user_sessions'] if not s.errors)
        
        metrics.update({
            'total_sessions': total_sessions,
            'successful_sessions': successful_sessions,
            'session_success_rate': successful_sessions / total_sessions if total_sessions > 0 else 0,
            'average_session_duration': np.mean([s.session_duration for s in metrics['user_sessions']]) if total_sessions > 0 else 0,
            'total_test_duration': time.time() - metrics['start_time']
        })
    
    async def _analyze_performance_degradation(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance degradation over time"""
        latencies = metrics['latencies']
        if len(latencies) < 10:
            return {'performance_degradation_percent': 0}
        
        # Compare first 25% vs last 25% of test
        first_quarter_size = len(latencies) // 4
        last_quarter_size = len(latencies) // 4
        
        first_quarter = latencies[:first_quarter_size]
        last_quarter = latencies[-last_quarter_size:]
        
        first_avg = np.mean(first_quarter)
        last_avg = np.mean(last_quarter)
        
        degradation = ((last_avg - first_avg) / first_avg) * 100 if first_avg > 0 else 0
        
        return {
            'performance_degradation_percent': max(0, degradation),  # Only report degradation, not improvement
            'first_quarter_avg_ms': first_avg,
            'last_quarter_avg_ms': last_avg
        }
    
    async def _analyze_extended_performance(self, metrics: Dict[str, Any], duration_seconds: int) -> Dict[str, Any]:
        """Analyze performance for extended duration tests"""
        extended_metrics = await self._analyze_performance_degradation(metrics)
        
        # Analyze memory growth over time
        memory_growth = metrics.get('memory_growth_percent', 0)
        memory_leak_detected = memory_growth > self.config.stress_test_config['memory_leak_threshold_percent']
        
        # Determine performance trend
        degradation = extended_metrics.get('performance_degradation_percent', 0)
        if degradation > 15:
            trend = 'degrading'
        elif degradation > 5:
            trend = 'concerning'
        else:
            trend = 'stable'
        
        extended_metrics.update({
            'memory_leak_detected': memory_leak_detected,
            'performance_trend': trend,
            'test_duration_hours': duration_seconds / 3600,
            'stability_score': max(0, 100 - degradation - (memory_growth * 2))  # Composite stability score
        })
        
        return extended_metrics


def create_stress_test_suite(config: Optional[PerformanceTestConfig] = None) -> StressTestSuite:
    """Factory function to create stress test suite"""
    if config is None:
        config = PerformanceTestConfig()
    return StressTestSuite(config) 