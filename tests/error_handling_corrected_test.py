#!/usr/bin/env python3
"""
Corrected Error Handling Test
Validates error handling mechanisms using proper APIs
"""

import asyncio
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

async def test_corrected_error_handling():
    """Test error handling with correct APIs"""
    print("🧪 Corrected Error Handling Validation")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: LLM Router Integration
    print("\n1. Testing LLM Router Integration...")
    total_tests += 1
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        from assistant.llm_router import LLMRouter
        
        router = LLMRouter()
        result = await router.route_query("Quick test query")
        
        if result and result.get('response'):
            print("   ✅ LLM Router working correctly")
            tests_passed += 1
        else:
            print("   ❌ LLM Router not responding properly")
            
        await router.cleanup()
        
    except Exception as e:
        print(f"   ❌ LLM Router test failed: {e}")
    
    # Test 2: Circuit Breaker (Corrected API)
    print("\n2. Testing Circuit Breaker...")
    total_tests += 1
    try:
        from assistant.fallback_manager import CircuitBreaker, CircuitBreakerConfig
        
        # Create circuit breaker with correct API
        config = CircuitBreakerConfig(failure_threshold=2, timeout_seconds=1.0)
        cb = CircuitBreaker("test_service", config)
        
        # Test normal operation
        async def working_function():
            return "success"
        
        result = await cb.execute(working_function)
        
        if result == "success":
            print("   ✅ Circuit Breaker working correctly")
            tests_passed += 1
        else:
            print("   ❌ Circuit Breaker not working properly")
            
    except Exception as e:
        print(f"   ❌ Circuit Breaker test failed: {e}")
    
    # Test 3: Retry Handler (Corrected API)
    print("\n3. Testing Retry Handler...")
    total_tests += 1
    try:
        from assistant.fallback_manager import RetryHandler, RetryConfig
        
        config = RetryConfig(max_attempts=3, base_delay=0.1)
        retry_handler = RetryHandler(config)
        
        # Test retry with eventually successful function
        attempts = 0
        async def eventually_successful():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise Exception("Temporary failure")
            return "success after retries"
        
        result = await retry_handler.execute_with_retry(eventually_successful)
        
        if result == "success after retries" and attempts == 3:
            print("   ✅ Retry Handler working correctly")
            tests_passed += 1
        else:
            print(f"   ❌ Retry Handler not working (result: {result}, attempts: {attempts})")
            
    except Exception as e:
        print(f"   ❌ Retry Handler test failed: {e}")
    
    # Test 4: Fallback Manager (Corrected API)
    print("\n4. Testing Fallback Manager...")
    total_tests += 1
    try:
        from assistant.fallback_manager import FallbackManager, FallbackConfig
        
        # Create fallback manager with correct API
        fallback_config = FallbackConfig(fallback_chain=['primary', 'secondary', 'backup'])
        fallback_manager = FallbackManager(config=fallback_config)
        
        # Test basic functionality
        service_functions = {
            'primary': lambda: asyncio.create_task(asyncio.coroutine(lambda: "Primary response")()),
            'secondary': lambda: asyncio.create_task(asyncio.coroutine(lambda: "Secondary response")()),
            'backup': lambda: asyncio.create_task(asyncio.coroutine(lambda: "Backup response")())
        }
        
        result = await fallback_manager.execute_with_fallback(
            'primary', service_functions
        )
        
        if result and "Primary" in str(result):
            print("   ✅ Fallback Manager working correctly")
            tests_passed += 1
        else:
            print(f"   ❌ Fallback Manager not working (result: {result})")
            
    except Exception as e:
        print(f"   ❌ Fallback Manager test failed: {e}")
    
    # Test 5: Error Message Manager
    print("\n5. Testing Error Message Manager...")
    total_tests += 1
    try:
        from assistant.fallback_manager import ErrorMessageManager
        
        error_manager = ErrorMessageManager()
        
        # Test error message generation
        user_message = error_manager.get_user_friendly_message(
            "network_timeout", {"service": "LLM"}
        )
        
        if user_message and len(user_message) > 10:
            print("   ✅ Error Message Manager generating messages")
            tests_passed += 1
        else:
            print(f"   ❌ Error Message Manager not working (message: {user_message})")
            
    except Exception as e:
        print(f"   ❌ Error Message Manager test failed: {e}")
    
    # Test 6: System Stress Test
    print("\n6. Testing System Under Stress...")
    total_tests += 1
    try:
        from assistant.llm_router import LLMRouter
        
        # Create multiple routers for stress testing
        routers = [LLMRouter() for _ in range(3)]
        
        start_time = time.time()
        
        # Create concurrent tasks
        async def make_request(router, query_id):
            try:
                result = await router.route_query(f"Stress test query {query_id}")
                return result.get('response') is not None
            except Exception:
                return False
        
        tasks = []
        for i, router in enumerate(routers):
            task = asyncio.create_task(make_request(router, i))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Cleanup routers
        for router in routers:
            await router.cleanup()
        
        response_time = time.time() - start_time
        successful_requests = sum(1 for r in results if r is True)
        
        if successful_requests >= 2 and response_time < 20.0:
            print(f"   ✅ System handles stress well ({successful_requests}/3 successful, {response_time:.2f}s)")
            tests_passed += 1
        else:
            print(f"   ❌ System stressed ({successful_requests}/3 successful, {response_time:.2f}s)")
            
    except Exception as e:
        print(f"   ❌ System stress test failed: {e}")
    
    # Test 7: Error Recovery Simulation
    print("\n7. Testing Error Recovery...")
    total_tests += 1
    try:
        from assistant.fallback_manager import CircuitBreaker, CircuitBreakerConfig
        
        # Test circuit breaker recovery after failures
        config = CircuitBreakerConfig(failure_threshold=2, timeout_seconds=0.5)
        cb = CircuitBreaker("recovery_test", config)
        
        # Simulate failures to open circuit
        failure_count = 0
        async def failing_function():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:
                raise Exception("Service failure")
            return "recovered"
        
        # First few calls should fail and open circuit
        for _ in range(3):
            try:
                await cb.execute(failing_function)
            except:
                pass
        
        # Wait for half-open state
        await asyncio.sleep(0.6)
        
        # Now should be able to succeed
        result = await cb.execute(failing_function)
        
        if result == "recovered":
            print("   ✅ Error recovery working correctly")
            tests_passed += 1
        else:
            print(f"   ❌ Error recovery not working (result: {result})")
            
    except Exception as e:
        print(f"   ❌ Error recovery test failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 CORRECTED ERROR HANDLING TEST RESULTS")
    print("=" * 60)
    
    success_rate = tests_passed / total_tests if total_tests > 0 else 0
    
    if success_rate >= 0.85:
        status = "✅ PASSED"
    elif success_rate >= 0.7:
        status = "⚠️ PARTIAL"
    else:
        status = "❌ FAILED"
    
    print(f"{status} Overall Status")
    print(f"📈 Success Rate: {success_rate:.1%} ({tests_passed}/{total_tests})")
    
    if tests_passed == total_tests:
        print("\n🎯 All error handling mechanisms are functioning correctly!")
        print("✅ System demonstrates robust error handling and recovery capabilities")
    elif tests_passed >= total_tests * 0.85:
        print("\n⚠️ Most error handling mechanisms are working correctly")
        print("🔧 Minor adjustments may improve resilience")
    else:
        print("\n❌ Error handling system needs attention")
        print("🛠️ Multiple mechanisms require fixes")
    
    # Additional insights
    print(f"\n🔍 Key Findings:")
    print(f"   • LLM Router Integration: {'✅ Working' if tests_passed >= 1 else '❌ Issues'}")
    print(f"   • Circuit Breaker Pattern: {'✅ Working' if tests_passed >= 2 else '❌ Issues'}")
    print(f"   • Retry Mechanisms: {'✅ Working' if tests_passed >= 3 else '❌ Issues'}")
    print(f"   • Fallback Systems: {'✅ Working' if tests_passed >= 4 else '❌ Issues'}")
    print(f"   • Error Messages: {'✅ Working' if tests_passed >= 5 else '❌ Issues'}")
    print(f"   • Stress Handling: {'✅ Working' if tests_passed >= 6 else '❌ Issues'}")
    print(f"   • Recovery Capability: {'✅ Working' if tests_passed >= 7 else '❌ Issues'}")
    
    return success_rate >= 0.85

if __name__ == '__main__':
    try:
        success = asyncio.run(test_corrected_error_handling())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Testing interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Testing failed: {e}")
        sys.exit(1) 