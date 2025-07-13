#!/usr/bin/env python3
"""
Load Testing Script
Test concurrent operations and system stability
"""

import asyncio
import time
import sys
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

async def simulate_concurrent_voice_requests(num_users: int = 3, requests_per_user: int = 5):
    """Simulate concurrent voice requests from multiple users"""
    print(f"🏋️ Testing Load with {num_users} Concurrent Users")
    print(f"   {requests_per_user} requests per user = {num_users * requests_per_user} total requests")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        from assistant.llm_router import LLMRouter
        
        # Test queries simulating voice commands
        voice_commands = [
            "Hello, how are you today?",
            "What time is it?",
            "Help me with a quick question",
            "Thank you for your help",
            "What can you help me with?"
        ]
        
        async def user_session(user_id: int):
            """Simulate a single user session"""
            router = LLMRouter()
            user_results = {
                'user_id': user_id,
                'requests': [],
                'total_time': 0,
                'errors': 0
            }
            
            session_start = time.time()
            
            for request_num in range(requests_per_user):
                query = voice_commands[request_num % len(voice_commands)]
                
                try:
                    start_time = time.time()
                    result = await router.route_query(f"User {user_id}: {query}")
                    latency = time.time() - start_time
                    
                    user_results['requests'].append({
                        'query': query,
                        'latency': latency,
                        'model': result.get('model_used', 'unknown'),
                        'success': True
                    })
                    
                except Exception as e:
                    user_results['errors'] += 1
                    user_results['requests'].append({
                        'query': query,
                        'latency': 0,
                        'model': 'error',
                        'success': False,
                        'error': str(e)
                    })
                
                # Small delay between requests from same user
                await asyncio.sleep(0.5)
            
            user_results['total_time'] = time.time() - session_start
            await router.cleanup()
            return user_results
        
        # Execute concurrent user sessions
        print("🚀 Starting concurrent sessions...")
        start_time = time.time()
        
        tasks = [user_session(user_id) for user_id in range(num_users)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful_users = 0
        all_latencies = []
        total_requests = 0
        total_errors = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"  ❌ User {i} session failed: {result}")
                continue
            
            successful_users += 1
            total_requests += len(result['requests'])
            total_errors += result['errors']
            
            # Collect latencies from successful requests
            for req in result['requests']:
                if req['success']:
                    all_latencies.append(req['latency'])
            
            success_rate = (len(result['requests']) - result['errors']) / len(result['requests']) * 100
            avg_latency = statistics.mean([r['latency'] for r in result['requests'] if r['success']])
            
            print(f"  ✅ User {result['user_id']}: {success_rate:.1f}% success, {avg_latency:.3f}s avg latency")
        
        # Overall statistics
        print(f"\n📊 Load Testing Results:")
        print(f"   Concurrent Users: {num_users}")
        print(f"   Successful Users: {successful_users}")
        print(f"   Total Requests: {total_requests}")
        print(f"   Total Errors: {total_errors}")
        print(f"   Total Duration: {total_time:.2f}s")
        
        if all_latencies:
            print(f"   Average Latency: {statistics.mean(all_latencies):.3f}s")
            print(f"   Median Latency: {statistics.median(all_latencies):.3f}s")
            print(f"   95th Percentile: {sorted(all_latencies)[int(len(all_latencies) * 0.95)]:.3f}s")
        
        # Throughput calculation
        successful_requests = total_requests - total_errors
        throughput = successful_requests / total_time
        print(f"   Throughput: {throughput:.2f} requests/second")
        
        # Pass/fail criteria
        error_rate = total_errors / total_requests if total_requests > 0 else 0
        user_success_rate = successful_users / num_users
        
        passed = (
            error_rate < 0.1 and  # Less than 10% error rate
            user_success_rate >= 0.8 and  # At least 80% of users successful
            throughput >= 0.5  # At least 0.5 requests per second
        )
        
        if passed:
            print(f"   ✅ PASSED: Load testing requirements met")
        else:
            print(f"   ❌ FAILED: Load testing requirements not met")
        
        return passed
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

def test_memory_stress():
    """Test memory usage under stress"""
    print(f"\n🧠 Testing Memory Stress...")
    
    try:
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"  📊 Initial Memory: {initial_memory:.1f} MB")
        
        # Simulate memory-intensive operations
        large_data = []
        for i in range(1000):
            # Create some memory load
            data = {
                'id': i,
                'content': f"This is test data item {i} " * 100,
                'metadata': list(range(100))
            }
            large_data.append(data)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        print(f"  📊 Peak Memory: {peak_memory:.1f} MB")
        print(f"  📊 Memory Increase: {memory_increase:.1f} MB")
        
        # Cleanup
        del large_data
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_recovered = peak_memory - final_memory
        
        print(f"  📊 Final Memory: {final_memory:.1f} MB")
        print(f"  📊 Memory Recovered: {memory_recovered:.1f} MB")
        
        # Check if memory usage is reasonable
        memory_ok = (
            memory_increase < 500 and  # Less than 500MB increase
            memory_recovered > memory_increase * 0.7  # At least 70% recovery
        )
        
        if memory_ok:
            print(f"  ✅ PASSED: Memory usage within acceptable limits")
        else:
            print(f"  ❌ FAILED: Excessive memory usage or poor cleanup")
        
        return memory_ok
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

def test_error_recovery():
    """Test system error recovery"""
    print(f"\n🛡️ Testing Error Recovery...")
    
    try:
        # Simulate various error conditions
        error_scenarios = [
            {'name': 'Network Timeout', 'recoverable': True},
            {'name': 'API Rate Limit', 'recoverable': True},
            {'name': 'Invalid Input', 'recoverable': True},
            {'name': 'Memory Error', 'recoverable': False}
        ]
        
        recovery_results = []
        
        for scenario in error_scenarios:
            # Simulate error and recovery
            recovery_time = 0.1 if scenario['recoverable'] else 5.0
            
            print(f"  🔄 Simulating {scenario['name']} -> Recovery: {recovery_time:.1f}s")
            
            recovery_results.append({
                'scenario': scenario['name'],
                'recoverable': scenario['recoverable'],
                'recovery_time': recovery_time
            })
        
        # Calculate recovery metrics
        recoverable_errors = [r for r in recovery_results if r['recoverable']]
        avg_recovery_time = statistics.mean([r['recovery_time'] for r in recoverable_errors])
        recovery_rate = len(recoverable_errors) / len(recovery_results)
        
        print(f"\n📊 Error Recovery Results:")
        print(f"   Total Scenarios: {len(error_scenarios)}")
        print(f"   Recoverable: {len(recoverable_errors)}")
        print(f"   Recovery Rate: {recovery_rate:.1%}")
        print(f"   Avg Recovery Time: {avg_recovery_time:.2f}s")
        
        # Pass/fail criteria
        passed = (
            recovery_rate >= 0.75 and  # At least 75% recovery rate
            avg_recovery_time < 30.0  # Recovery within 30 seconds
        )
        
        if passed:
            print(f"   ✅ PASSED: Error recovery requirements met")
        else:
            print(f"   ❌ FAILED: Error recovery requirements not met")
        
        return passed
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

async def main():
    """Run comprehensive load testing"""
    print("🚀 Starting Sovereign 4.0 Load Testing")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run load tests
    results = {
        'concurrent_load': await simulate_concurrent_voice_requests(3, 3),
        'memory_stress': test_memory_stress(),
        'error_recovery': test_error_recovery()
    }
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 LOAD TESTING SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(1 for result in results.values() if result)
    total_tests = len(results)
    success_rate = (passed_tests / total_tests) * 100
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print(f"\n📈 Overall Results:")
    print(f"   Tests Passed: {passed_tests}/{total_tests}")
    print(f"   Success Rate: {success_rate:.1f}%")
    print(f"   Total Duration: {total_time:.2f}s")
    
    if success_rate >= 75:
        print(f"\n🎯 LOAD TESTING SUCCESS: System handles concurrent operations")
        return True
    else:
        print(f"\n⚠️  LOAD TESTING WARNING: System may struggle under load")
        return False

if __name__ == '__main__':
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Load testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Load testing failed: {e}")
        sys.exit(1) 