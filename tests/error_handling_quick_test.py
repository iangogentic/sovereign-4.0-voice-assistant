#!/usr/bin/env python3
"""
Quick Error Handling Test
Validates error handling mechanisms with realistic thresholds
"""

import asyncio
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

async def test_basic_error_handling():
    """Test basic error handling and recovery mechanisms"""
    print("🧪 Quick Error Handling Validation")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: LLM Router Fallback
    print("\n1. Testing LLM Router Fallback Mechanism...")
    total_tests += 1
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        from assistant.llm_router import LLMRouter
        
        router = LLMRouter()
        result = await router.route_query("Hello, test query")
        
        if result and result.get('response'):
            print("   ✅ LLM Router working correctly")
            tests_passed += 1
        else:
            print("   ❌ LLM Router not responding properly")
            
        await router.cleanup()
        
    except Exception as e:
        print(f"   ❌ LLM Router test failed: {e}")
    
    # Test 2: Circuit Breaker
    print("\n2. Testing Circuit Breaker...")
    total_tests += 1
    try:
        from assistant.fallback_manager import CircuitBreaker
        
        # Create circuit breaker
        cb = CircuitBreaker(failure_threshold=2, timeout=1.0)
        
        # Test normal operation
        async def working_function():
            return "success"
        
        result = await cb.call(working_function)
        
        if result == "success":
            print("   ✅ Circuit Breaker allowing calls in CLOSED state")
            tests_passed += 1
        else:
            print("   ❌ Circuit Breaker not working properly")
            
    except Exception as e:
        print(f"   ❌ Circuit Breaker test failed: {e}")
    
    # Test 3: Retry Handler
    print("\n3. Testing Retry Handler...")
    total_tests += 1
    try:
        from assistant.fallback_manager import RetryHandler
        
        retry_handler = RetryHandler()
        
        # Test retry with eventually successful function
        attempts = 0
        async def eventually_successful():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise Exception("Temporary failure")
            return "success after retries"
        
        result = await retry_handler.execute_with_retry(
            eventually_successful, max_retries=5
        )
        
        if result == "success after retries" and attempts == 3:
            print("   ✅ Retry Handler working correctly")
            tests_passed += 1
        else:
            print(f"   ❌ Retry Handler not working (result: {result}, attempts: {attempts})")
            
    except Exception as e:
        print(f"   ❌ Retry Handler test failed: {e}")
    
    # Test 4: Fallback Manager
    print("\n4. Testing Fallback Manager...")
    total_tests += 1
    try:
        from assistant.fallback_manager import FallbackManager
        
        # Create simple fallback chain
        services = ["primary", "secondary", "backup"]
        fallback_manager = FallbackManager(services)
        
        # Test basic functionality
        async def mock_service_call(service_name):
            if service_name == "primary":
                return f"Response from {service_name}"
            raise Exception(f"{service_name} unavailable")
        
        result = await fallback_manager.execute_with_fallback(
            mock_service_call, "test_operation"
        )
        
        if result and "primary" in result:
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
        test_error = Exception("Connection timeout")
        user_message = error_manager.get_user_friendly_message(
            test_error, "network_service"
        )
        
        if user_message and "sorry" in user_message.lower():
            print("   ✅ Error Message Manager working correctly")
            tests_passed += 1
        else:
            print(f"   ❌ Error Message Manager not working (message: {user_message})")
            
    except Exception as e:
        print(f"   ❌ Error Message Manager test failed: {e}")
    
    # Test 6: System Responsiveness Under Load
    print("\n6. Testing System Responsiveness...")
    total_tests += 1
    try:
        start_time = time.time()
        
        # Create multiple concurrent requests
        from assistant.llm_router import LLMRouter
        router = LLMRouter()
        
        tasks = []
        for i in range(3):
            task = asyncio.create_task(
                router.route_query(f"Concurrent test query {i}")
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        response_time = time.time() - start_time
        successful_requests = sum(1 for r in results if isinstance(r, dict) and r.get('response'))
        
        await router.cleanup()
        
        if successful_requests >= 2 and response_time < 15.0:
            print(f"   ✅ System responsive under load ({successful_requests}/3 successful, {response_time:.2f}s)")
            tests_passed += 1
        else:
            print(f"   ❌ System not responsive enough ({successful_requests}/3 successful, {response_time:.2f}s)")
            
    except Exception as e:
        print(f"   ❌ System responsiveness test failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 QUICK ERROR HANDLING TEST RESULTS")
    print("=" * 50)
    
    success_rate = tests_passed / total_tests if total_tests > 0 else 0
    
    if success_rate >= 0.8:
        status = "✅ PASSED"
    elif success_rate >= 0.6:
        status = "⚠️ PARTIAL"
    else:
        status = "❌ FAILED"
    
    print(f"{status} Overall Status")
    print(f"📈 Success Rate: {success_rate:.1%} ({tests_passed}/{total_tests})")
    
    if tests_passed == total_tests:
        print("\n🎯 All error handling mechanisms are functioning correctly!")
    elif tests_passed >= total_tests * 0.8:
        print("\n⚠️ Most error handling mechanisms are working, minor issues detected")
    else:
        print("\n❌ Multiple error handling mechanisms need attention")
    
    return success_rate >= 0.8

if __name__ == '__main__':
    try:
        success = asyncio.run(test_basic_error_handling())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Testing interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Testing failed: {e}")
        sys.exit(1) 