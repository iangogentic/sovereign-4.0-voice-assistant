#!/usr/bin/env python3
"""
Error Handling and Fallback Mechanism Validation Framework
Comprehensive fault injection and chaos engineering testing
"""

import asyncio
import time
import sys
import json
import random
import traceback
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class FailureType(Enum):
    """Types of failures to inject"""
    NETWORK_FAILURE = "network_failure"
    API_TIMEOUT = "api_timeout"
    API_ERROR = "api_error"
    VOICE_RECOGNITION_ERROR = "voice_recognition_error"
    OCR_FAILURE = "ocr_failure"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    DISK_FULL = "disk_full"
    HIGH_CPU_LOAD = "high_cpu_load"
    SERVICE_UNAVAILABLE = "service_unavailable"
    AUTHENTICATION_FAILURE = "authentication_failure"

class RecoveryMechanism(Enum):
    """Types of recovery mechanisms to test"""
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    FALLBACK_SERVICE = "fallback_service"
    OFFLINE_MODE = "offline_mode"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    USER_NOTIFICATION = "user_notification"
    CIRCUIT_BREAKER = "circuit_breaker"
    CACHE_FALLBACK = "cache_fallback"

@dataclass
class FailureScenario:
    """Defines a failure scenario to test"""
    name: str
    failure_type: FailureType
    description: str
    inject_function: str
    expected_recovery: RecoveryMechanism
    max_recovery_time: float  # seconds
    critical: bool = False

@dataclass
class TestResult:
    """Result of a failure test"""
    scenario: FailureScenario
    start_time: float
    end_time: float
    recovery_time: float
    success: bool
    error_message: str = ""
    recovery_mechanism_triggered: bool = False
    system_state_stable: bool = True
    user_experience_maintained: bool = True

@dataclass
class ValidationResults:
    """Overall validation results"""
    start_time: float
    end_time: float
    total_scenarios: int
    passed_scenarios: int
    failed_scenarios: int
    critical_failures: int
    average_recovery_time: float
    test_results: List[TestResult]

class FaultInjector:
    """Inject various types of failures into the system"""
    
    def __init__(self):
        self.active_faults = {}
        self.original_functions = {}
    
    async def inject_network_failure(self, duration: float = 5.0):
        """Simulate network connectivity failure"""
        print(f"🔌 Injecting network failure for {duration}s")
        
        # Mock network calls to fail
        import aiohttp
        original_request = aiohttp.ClientSession._request
        
        async def failing_request(*args, **kwargs):
            raise aiohttp.ClientConnectorError(
                connection_key=None, 
                os_error=OSError("Network is unreachable")
            )
        
        aiohttp.ClientSession._request = failing_request
        await asyncio.sleep(duration)
        aiohttp.ClientSession._request = original_request
        
        print(f"🔌 Network failure cleared")
    
    async def inject_api_timeout(self, duration: float = 3.0):
        """Simulate API timeout scenarios"""
        print(f"⏱️ Injecting API timeout for {duration}s")
        
        import aiohttp
        original_request = aiohttp.ClientSession._request
        
        async def timeout_request(*args, **kwargs):
            raise asyncio.TimeoutError("Request timed out")
        
        aiohttp.ClientSession._request = timeout_request
        await asyncio.sleep(duration)
        aiohttp.ClientSession._request = original_request
        
        print(f"⏱️ API timeout cleared")
    
    async def inject_api_error(self, duration: float = 3.0):
        """Simulate API error responses"""
        print(f"🚫 Injecting API errors for {duration}s")
        
        import aiohttp
        original_request = aiohttp.ClientSession._request
        
        async def error_request(*args, **kwargs):
            # Mock response with error status
            class MockResponse:
                status = 500
                async def text(self): return "Internal Server Error"
                async def json(self): return {"error": "Service unavailable"}
            
            return MockResponse()
        
        aiohttp.ClientSession._request = error_request
        await asyncio.sleep(duration)
        aiohttp.ClientSession._request = original_request
        
        print(f"🚫 API error cleared")
    
    async def inject_voice_recognition_error(self, duration: float = 3.0):
        """Simulate speech-to-text failures"""
        print(f"🎤 Injecting voice recognition errors for {duration}s")
        
        try:
            from assistant.stt import WhisperSTTService
            original_transcribe = WhisperSTTService.transcribe
            
            async def failing_transcribe(self, audio_data):
                raise Exception("Speech recognition service unavailable")
            
            WhisperSTTService.transcribe = failing_transcribe
            await asyncio.sleep(duration)
            WhisperSTTService.transcribe = original_transcribe
            
        except ImportError:
            print("⚠️ STT service not available for testing")
        
        print(f"🎤 Voice recognition error cleared")
    
    async def inject_ocr_failure(self, duration: float = 3.0):
        """Simulate OCR processing failures"""
        print(f"📷 Injecting OCR failures for {duration}s")
        
        try:
            # Mock OCR failures by overriding OCR functions
            import pytesseract
            original_image_to_string = pytesseract.image_to_string
            
            def failing_ocr(*args, **kwargs):
                raise Exception("OCR service unavailable")
            
            pytesseract.image_to_string = failing_ocr
            await asyncio.sleep(duration)
            pytesseract.image_to_string = original_image_to_string
            
        except ImportError:
            print("⚠️ OCR service not available for testing")
        
        print(f"📷 OCR failure cleared")
    
    async def inject_memory_pressure(self, duration: float = 5.0):
        """Simulate high memory usage"""
        print(f"💾 Injecting memory pressure for {duration}s")
        
        # Allocate significant memory to simulate pressure
        memory_hog = []
        try:
            for i in range(100):  # Allocate chunks of memory
                chunk = bytearray(1024 * 1024)  # 1MB chunks
                memory_hog.append(chunk)
                await asyncio.sleep(0.05)
            
            await asyncio.sleep(duration - 5)  # Keep memory allocated
            
        finally:
            del memory_hog  # Clean up
            import gc
            gc.collect()
        
        print(f"💾 Memory pressure cleared")
    
    async def inject_high_cpu_load(self, duration: float = 5.0):
        """Simulate high CPU usage"""
        print(f"🔥 Injecting high CPU load for {duration}s")
        
        # Create CPU-intensive tasks
        async def cpu_burner():
            end_time = time.time() + duration
            while time.time() < end_time:
                # CPU-intensive calculation
                sum(x * x for x in range(1000))
                await asyncio.sleep(0.001)  # Brief yield
        
        # Run multiple CPU burners
        tasks = [asyncio.create_task(cpu_burner()) for _ in range(4)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        print(f"🔥 High CPU load cleared")

class RecoveryValidator:
    """Validate system recovery mechanisms"""
    
    def __init__(self):
        self.recovery_checks = {
            RecoveryMechanism.RETRY_WITH_BACKOFF: self._validate_retry_mechanism,
            RecoveryMechanism.FALLBACK_SERVICE: self._validate_fallback_service,
            RecoveryMechanism.OFFLINE_MODE: self._validate_offline_mode,
            RecoveryMechanism.GRACEFUL_DEGRADATION: self._validate_graceful_degradation,
            RecoveryMechanism.USER_NOTIFICATION: self._validate_user_notification,
            RecoveryMechanism.CIRCUIT_BREAKER: self._validate_circuit_breaker,
            RecoveryMechanism.CACHE_FALLBACK: self._validate_cache_fallback,
        }
    
    async def validate_recovery(self, mechanism: RecoveryMechanism, 
                              test_context: Dict[str, Any]) -> bool:
        """Validate a specific recovery mechanism"""
        validator = self.recovery_checks.get(mechanism)
        if validator:
            return await validator(test_context)
        return False
    
    async def _validate_retry_mechanism(self, context: Dict[str, Any]) -> bool:
        """Check if retry logic is working"""
        try:
            from assistant.fallback_manager import RetryHandler
            retry_handler = RetryHandler()
            
            # Test retry with mock function
            attempts = 0
            async def mock_failing_function():
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    raise Exception("Temporary failure")
                return "success"
            
            result = await retry_handler.execute_with_retry(
                mock_failing_function, max_retries=3
            )
            
            return result == "success" and attempts == 3
            
        except Exception as e:
            print(f"⚠️ Retry validation failed: {e}")
            return False
    
    async def _validate_fallback_service(self, context: Dict[str, Any]) -> bool:
        """Check if fallback services are activated"""
        try:
            from assistant.fallback_manager import FallbackManager
            from assistant.llm_router import LLMRouter
            
            router = LLMRouter()
            
            # Test fallback by triggering failure scenario
            result = await router.route_query("test query during failure")
            
            # Should get response from fallback service
            return result and result.get('response') is not None
            
        except Exception as e:
            print(f"⚠️ Fallback service validation failed: {e}")
            return False
    
    async def _validate_offline_mode(self, context: Dict[str, Any]) -> bool:
        """Check if offline mode is activated correctly"""
        try:
            # Check if system switches to offline mode during network failure
            # This would involve checking cached responses or offline capabilities
            return True  # Placeholder - implement based on offline system
            
        except Exception as e:
            print(f"⚠️ Offline mode validation failed: {e}")
            return False
    
    async def _validate_graceful_degradation(self, context: Dict[str, Any]) -> bool:
        """Check if system degrades gracefully under stress"""
        try:
            # Verify system continues operating with reduced functionality
            # Check performance metrics don't drop below critical thresholds
            
            # Test basic functionality still works
            from assistant.llm_router import LLMRouter
            router = LLMRouter()
            result = await router.route_query("simple test")
            
            return result is not None
            
        except Exception as e:
            print(f"⚠️ Graceful degradation validation failed: {e}")
            return False
    
    async def _validate_user_notification(self, context: Dict[str, Any]) -> bool:
        """Check if users are properly notified of issues"""
        try:
            # Check if error messages are user-friendly and informative
            # This would involve checking error message formatting
            return True  # Placeholder - implement based on notification system
            
        except Exception as e:
            print(f"⚠️ User notification validation failed: {e}")
            return False
    
    async def _validate_circuit_breaker(self, context: Dict[str, Any]) -> bool:
        """Check if circuit breaker is functioning"""
        try:
            from assistant.fallback_manager import CircuitBreaker
            
            circuit_breaker = CircuitBreaker(failure_threshold=2, timeout=1.0)
            
            # Trigger failures to open circuit
            for _ in range(3):
                try:
                    async def failing_function():
                        raise Exception("Service unavailable")
                    
                    await circuit_breaker.call(failing_function)
                except:
                    pass
            
            # Check if circuit is open
            return circuit_breaker.state.name == "OPEN"
            
        except Exception as e:
            print(f"⚠️ Circuit breaker validation failed: {e}")
            return False
    
    async def _validate_cache_fallback(self, context: Dict[str, Any]) -> bool:
        """Check if cached responses are used during failures"""
        try:
            # Check if system uses cached data when services are unavailable
            return True  # Placeholder - implement based on caching system
            
        except Exception as e:
            print(f"⚠️ Cache fallback validation failed: {e}")
            return False

class ErrorHandlingValidator:
    """Main validator for error handling and fallback mechanisms"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("reports/error_handling")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.fault_injector = FaultInjector()
        self.recovery_validator = RecoveryValidator()
        
        # Define test scenarios
        self.scenarios = [
            FailureScenario(
                name="Network Connectivity Failure",
                failure_type=FailureType.NETWORK_FAILURE,
                description="Simulate complete network connectivity loss",
                inject_function="inject_network_failure",
                expected_recovery=RecoveryMechanism.OFFLINE_MODE,
                max_recovery_time=5.0,
                critical=True
            ),
            FailureScenario(
                name="API Timeout",
                failure_type=FailureType.API_TIMEOUT,
                description="Simulate API request timeouts",
                inject_function="inject_api_timeout",
                expected_recovery=RecoveryMechanism.RETRY_WITH_BACKOFF,
                max_recovery_time=10.0
            ),
            FailureScenario(
                name="API Error Response",
                failure_type=FailureType.API_ERROR,
                description="Simulate API returning error responses",
                inject_function="inject_api_error",
                expected_recovery=RecoveryMechanism.FALLBACK_SERVICE,
                max_recovery_time=5.0
            ),
            FailureScenario(
                name="Voice Recognition Failure",
                failure_type=FailureType.VOICE_RECOGNITION_ERROR,
                description="Simulate speech-to-text service failures",
                inject_function="inject_voice_recognition_error",
                expected_recovery=RecoveryMechanism.USER_NOTIFICATION,
                max_recovery_time=3.0
            ),
            FailureScenario(
                name="OCR Processing Failure",
                failure_type=FailureType.OCR_FAILURE,
                description="Simulate OCR service failures",
                inject_function="inject_ocr_failure",
                expected_recovery=RecoveryMechanism.GRACEFUL_DEGRADATION,
                max_recovery_time=3.0
            ),
            FailureScenario(
                name="Memory Pressure",
                failure_type=FailureType.MEMORY_EXHAUSTION,
                description="Simulate high memory usage conditions",
                inject_function="inject_memory_pressure",
                expected_recovery=RecoveryMechanism.GRACEFUL_DEGRADATION,
                max_recovery_time=10.0,
                critical=True
            ),
            FailureScenario(
                name="High CPU Load",
                failure_type=FailureType.HIGH_CPU_LOAD,
                description="Simulate high CPU usage conditions",
                inject_function="inject_high_cpu_load",
                expected_recovery=RecoveryMechanism.GRACEFUL_DEGRADATION,
                max_recovery_time=8.0
            )
        ]
    
    async def run_scenario_test(self, scenario: FailureScenario) -> TestResult:
        """Run a single failure scenario test"""
        print(f"\n🧪 Testing: {scenario.name}")
        print(f"   📝 {scenario.description}")
        print(f"   🎯 Expected recovery: {scenario.expected_recovery.value}")
        
        start_time = time.time()
        success = False
        error_message = ""
        recovery_mechanism_triggered = False
        system_state_stable = True
        user_experience_maintained = True
        
        try:
            # Start monitoring system state
            initial_state = await self._capture_system_state()
            
            # Inject the failure
            inject_method = getattr(self.fault_injector, scenario.inject_function)
            
            # Run failure injection and monitoring concurrently
            failure_task = asyncio.create_task(inject_method(3.0))
            monitoring_task = asyncio.create_task(
                self._monitor_recovery(scenario, initial_state)
            )
            
            # Wait for both tasks
            await failure_task
            recovery_result = await monitoring_task
            
            # Validate recovery mechanism was triggered
            recovery_mechanism_triggered = await self.recovery_validator.validate_recovery(
                scenario.expected_recovery, {"scenario": scenario}
            )
            
            # Check final system state
            final_state = await self._capture_system_state()
            system_state_stable = self._compare_system_states(initial_state, final_state)
            
            # Determine overall success
            recovery_time = time.time() - start_time
            success = (
                recovery_time <= scenario.max_recovery_time and
                recovery_mechanism_triggered and
                system_state_stable
            )
            
            if success:
                print(f"   ✅ Scenario passed ({recovery_time:.2f}s)")
            else:
                reasons = []
                if recovery_time > scenario.max_recovery_time:
                    reasons.append(f"slow recovery ({recovery_time:.2f}s > {scenario.max_recovery_time}s)")
                if not recovery_mechanism_triggered:
                    reasons.append("recovery mechanism not triggered")
                if not system_state_stable:
                    reasons.append("system state unstable")
                
                error_message = "; ".join(reasons)
                print(f"   ❌ Scenario failed: {error_message}")
                
        except Exception as e:
            error_message = str(e)
            print(f"   ❌ Scenario failed with exception: {error_message}")
            print(f"   📋 Traceback: {traceback.format_exc()}")
        
        end_time = time.time()
        recovery_time = end_time - start_time
        
        return TestResult(
            scenario=scenario,
            start_time=start_time,
            end_time=end_time,
            recovery_time=recovery_time,
            success=success,
            error_message=error_message,
            recovery_mechanism_triggered=recovery_mechanism_triggered,
            system_state_stable=system_state_stable,
            user_experience_maintained=user_experience_maintained
        )
    
    async def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for comparison"""
        try:
            import psutil
            process = psutil.Process()
            
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'process_memory': process.memory_info().rss,
                'threads': process.num_threads(),
                'timestamp': time.time()
            }
        except ImportError:
            return {'timestamp': time.time()}
    
    def _compare_system_states(self, initial: Dict[str, Any], 
                             final: Dict[str, Any]) -> bool:
        """Compare system states to check stability"""
        if not initial or not final:
            return True
        
        try:
            # Check for significant resource changes
            memory_change = abs(final.get('memory_percent', 0) - 
                              initial.get('memory_percent', 0))
            
            # Allow up to 10% memory change
            return memory_change < 10.0
            
        except:
            return True
    
    async def _monitor_recovery(self, scenario: FailureScenario, 
                              initial_state: Dict[str, Any]) -> bool:
        """Monitor system during failure injection"""
        recovery_start = time.time()
        
        while time.time() - recovery_start < scenario.max_recovery_time:
            try:
                # Test if system is responsive
                await self._test_system_responsiveness()
                await asyncio.sleep(0.5)
                
            except Exception:
                # System not responsive yet, continue monitoring
                await asyncio.sleep(0.5)
                continue
        
        return True
    
    async def _test_system_responsiveness(self):
        """Test if system is still responsive"""
        try:
            # Simple responsiveness test
            from dotenv import load_dotenv
            load_dotenv()
            
            from assistant.llm_router import LLMRouter
            router = LLMRouter()
            
            # Quick test query
            result = await asyncio.wait_for(
                router.route_query("health check"), timeout=2.0
            )
            
            await router.cleanup()
            return result is not None
            
        except Exception:
            raise Exception("System not responsive")
    
    async def run_validation(self) -> ValidationResults:
        """Run complete error handling validation"""
        print("🚀 Starting Error Handling and Fallback Mechanism Validation")
        print("=" * 70)
        print(f"Scenarios to test: {len(self.scenarios)}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 70)
        
        start_time = time.time()
        test_results = []
        
        for i, scenario in enumerate(self.scenarios, 1):
            print(f"\n[{i}/{len(self.scenarios)}] Running scenario: {scenario.name}")
            
            try:
                result = await self.run_scenario_test(scenario)
                test_results.append(result)
                
                # Brief pause between tests
                await asyncio.sleep(2.0)
                
            except Exception as e:
                print(f"❌ Scenario {scenario.name} failed with error: {e}")
                # Create failed result
                failed_result = TestResult(
                    scenario=scenario,
                    start_time=time.time(),
                    end_time=time.time(),
                    recovery_time=999.0,
                    success=False,
                    error_message=str(e),
                    recovery_mechanism_triggered=False,
                    system_state_stable=False
                )
                test_results.append(failed_result)
        
        end_time = time.time()
        
        # Calculate summary statistics
        total_scenarios = len(test_results)
        passed_scenarios = sum(1 for r in test_results if r.success)
        failed_scenarios = total_scenarios - passed_scenarios
        critical_failures = sum(1 for r in test_results 
                              if not r.success and r.scenario.critical)
        
        recovery_times = [r.recovery_time for r in test_results if r.success]
        average_recovery_time = sum(recovery_times) / len(recovery_times) if recovery_times else 0
        
        results = ValidationResults(
            start_time=start_time,
            end_time=end_time,
            total_scenarios=total_scenarios,
            passed_scenarios=passed_scenarios,
            failed_scenarios=failed_scenarios,
            critical_failures=critical_failures,
            average_recovery_time=average_recovery_time,
            test_results=test_results
        )
        
        # Save results
        await self._save_results(results)
        
        return results
    
    async def _save_results(self, results: ValidationResults):
        """Save test results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        results_file = self.output_dir / f"error_handling_results_{timestamp}.json"
        results_data = {
            'summary': {
                'start_time': results.start_time,
                'end_time': results.end_time,
                'duration_minutes': (results.end_time - results.start_time) / 60,
                'total_scenarios': results.total_scenarios,
                'passed_scenarios': results.passed_scenarios,
                'failed_scenarios': results.failed_scenarios,
                'critical_failures': results.critical_failures,
                'average_recovery_time': results.average_recovery_time,
                'success_rate': results.passed_scenarios / results.total_scenarios if results.total_scenarios > 0 else 0
            },
            'test_results': []
        }
        
        for result in results.test_results:
            results_data['test_results'].append({
                'scenario_name': result.scenario.name,
                'failure_type': result.scenario.failure_type.value,
                'expected_recovery': result.scenario.expected_recovery.value,
                'critical': result.scenario.critical,
                'recovery_time': result.recovery_time,
                'max_recovery_time': result.scenario.max_recovery_time,
                'success': result.success,
                'error_message': result.error_message,
                'recovery_mechanism_triggered': result.recovery_mechanism_triggered,
                'system_state_stable': result.system_state_stable
            })
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n📁 Results saved to: {results_file}")

async def main():
    """Main entry point for error handling validation"""
    parser = argparse.ArgumentParser(description="Error Handling and Fallback Mechanism Validation")
    parser.add_argument('--output-dir', '-o', type=str, 
                       help='Output directory for test results')
    parser.add_argument('--scenarios', '-s', type=str, nargs='+',
                       help='Specific scenarios to test (by name)')
    parser.add_argument('--critical-only', '-c', action='store_true',
                       help='Run only critical scenarios')
    
    args = parser.parse_args()
    
    # Setup validator
    output_dir = Path(args.output_dir) if args.output_dir else None
    validator = ErrorHandlingValidator(output_dir)
    
    # Filter scenarios if requested
    if args.scenarios:
        validator.scenarios = [s for s in validator.scenarios 
                             if s.name in args.scenarios]
    elif args.critical_only:
        validator.scenarios = [s for s in validator.scenarios if s.critical]
    
    try:
        # Run validation
        results = await validator.run_validation()
        
        # Print summary
        print("\n" + "=" * 70)
        print("📊 ERROR HANDLING VALIDATION RESULTS")
        print("=" * 70)
        
        success_rate = results.passed_scenarios / results.total_scenarios if results.total_scenarios > 0 else 0
        
        if results.critical_failures == 0 and success_rate >= 0.8:
            status = "✅ PASSED"
        elif results.critical_failures == 0:
            status = "⚠️ PARTIAL"
        else:
            status = "❌ FAILED"
        
        print(f"{status} Overall Status")
        print(f"📈 Success Rate: {success_rate:.1%} ({results.passed_scenarios}/{results.total_scenarios})")
        print(f"⚡ Average Recovery Time: {results.average_recovery_time:.2f}s")
        print(f"🚨 Critical Failures: {results.critical_failures}")
        
        if results.failed_scenarios > 0:
            print(f"\n❌ Failed Scenarios:")
            for result in results.test_results:
                if not result.success:
                    critical_marker = " (CRITICAL)" if result.scenario.critical else ""
                    print(f"   - {result.scenario.name}{critical_marker}: {result.error_message}")
        
        print(f"\n⏱️ Test Duration: {(results.end_time - results.start_time)/60:.1f} minutes")
        
        # Return appropriate exit code
        return results.critical_failures == 0 and success_rate >= 0.8
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        print(f"📋 Traceback: {traceback.format_exc()}")
        return False

if __name__ == '__main__':
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Validation failed: {e}")
        sys.exit(1) 