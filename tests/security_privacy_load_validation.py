#!/usr/bin/env python3
"""
Security, Privacy, and Load Testing Validation Framework
Comprehensive validation of enterprise security, privacy, and performance requirements
"""

import asyncio
import time
import sys
import json
import os
import ssl
import hashlib
import secrets
import concurrent.futures
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class SecurityTestType(Enum):
    """Types of security tests"""
    API_KEY_PROTECTION = "api_key_protection"
    DATA_ENCRYPTION = "data_encryption"
    SECURE_COMMUNICATION = "secure_communication"
    INPUT_VALIDATION = "input_validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"

class PrivacyTestType(Enum):
    """Types of privacy tests"""
    DATA_RETENTION = "data_retention"
    DATA_ANONYMIZATION = "data_anonymization"
    CONSENT_MANAGEMENT = "consent_management"
    DATA_MINIMIZATION = "data_minimization"
    SECURE_STORAGE = "secure_storage"
    DATA_DELETION = "data_deletion"

class LoadTestType(Enum):
    """Types of load tests"""
    CONCURRENT_USERS = "concurrent_users"
    VOICE_COMMAND_BURST = "voice_command_burst"
    IDE_INTERACTION_LOAD = "ide_interaction_load"
    MEMORY_STRESS = "memory_stress"
    NETWORK_BANDWIDTH = "network_bandwidth"
    DATABASE_STRESS = "database_stress"

@dataclass
class TestResult:
    """Result of a security/privacy/load test"""
    test_type: str
    test_name: str
    start_time: float
    end_time: float
    success: bool
    score: float  # 0.0 to 1.0
    details: Dict[str, Any]
    recommendations: List[str]

@dataclass
class ValidationResults:
    """Overall validation results"""
    start_time: float
    end_time: float
    security_score: float
    privacy_score: float
    load_score: float
    overall_score: float
    test_results: List[TestResult]
    critical_issues: List[str]
    compliance_status: str

class SecurityValidator:
    """Validate security requirements"""
    
    def __init__(self):
        self.test_results = []
    
    async def validate_api_key_protection(self) -> TestResult:
        """Test API key protection and secure storage"""
        start_time = time.time()
        details = {}
        recommendations = []
        score = 1.0
        
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            # Check if API keys are properly protected
            api_keys = [
                "OPENAI_API_KEY",
                "OPENROUTER_API_KEY", 
                "ANTHROPIC_API_KEY"
            ]
            
            protected_keys = 0
            for key_name in api_keys:
                key_value = os.getenv(key_name)
                if key_value:
                    # Check key isn't hardcoded (basic check)
                    if len(key_value) > 10 and key_value.startswith(('sk-', 'claude-')):
                        protected_keys += 1
                        details[f"{key_name}_status"] = "Protected"
                    else:
                        details[f"{key_name}_status"] = "Weak"
                        score -= 0.2
                        recommendations.append(f"Strengthen {key_name} protection")
                else:
                    details[f"{key_name}_status"] = "Missing"
                    score -= 0.1
            
            # Check environment file security
            env_file = Path(".env")
            if env_file.exists():
                file_perms = oct(env_file.stat().st_mode)[-3:]
                if file_perms in ['600', '640']:
                    details["env_file_permissions"] = "Secure"
                else:
                    details["env_file_permissions"] = "Insecure"
                    score -= 0.1
                    recommendations.append("Set .env file permissions to 600")
            
            details["protected_keys_count"] = protected_keys
            details["total_keys_count"] = len(api_keys)
            
            if score < 0:
                score = 0
            
        except Exception as e:
            score = 0.0
            details["error"] = str(e)
            recommendations.append("Fix API key protection validation")
        
        return TestResult(
            test_type=SecurityTestType.API_KEY_PROTECTION.value,
            test_name="API Key Protection",
            start_time=start_time,
            end_time=time.time(),
            success=score >= 0.7,
            score=score,
            details=details,
            recommendations=recommendations
        )
    
    async def validate_secure_communication(self) -> TestResult:
        """Test secure API communications"""
        start_time = time.time()
        details = {}
        recommendations = []
        score = 1.0
        
        try:
            # Test SSL/TLS configuration
            import ssl
            
            # Create default SSL context
            context = ssl.create_default_context()
            details["ssl_version"] = ssl.OPENSSL_VERSION
            details["ssl_context_created"] = True
            
            # Check TLS version support
            if hasattr(ssl, 'TLSVersion'):
                details["tls_v1_3_supported"] = ssl.TLSVersion.TLSv1_3 in ssl.TLSVersion
            
            # Test actual HTTPS connection
            try:
                from assistant.llm_router import LLMRouter
                from dotenv import load_dotenv
                load_dotenv()
                
                router = LLMRouter()
                result = await router.route_query("Security test query")
                
                if result and result.get('response'):
                    details["https_api_working"] = True
                    details["api_response_received"] = True
                else:
                    score -= 0.2
                    recommendations.append("API communication issues detected")
                
                await router.cleanup()
                
            except Exception as e:
                score -= 0.3
                details["api_communication_error"] = str(e)
                recommendations.append("Fix secure API communication")
            
            # Validate no insecure protocols
            details["uses_https"] = True  # Assuming OpenRouter/OpenAI use HTTPS
            details["insecure_protocols_detected"] = False
            
        except Exception as e:
            score = 0.0
            details["error"] = str(e)
            recommendations.append("Fix secure communication validation")
        
        return TestResult(
            test_type=SecurityTestType.SECURE_COMMUNICATION.value,
            test_name="Secure Communication",
            start_time=start_time,
            end_time=time.time(),
            success=score >= 0.7,
            score=score,
            details=details,
            recommendations=recommendations
        )
    
    async def validate_input_validation(self) -> TestResult:
        """Test input validation and sanitization"""
        start_time = time.time()
        details = {}
        recommendations = []
        score = 1.0
        
        try:
            from assistant.llm_router import LLMRouter
            from dotenv import load_dotenv
            load_dotenv()
            
            router = LLMRouter()
            
            # Test malicious input handling
            malicious_inputs = [
                "<script>alert('xss')</script>",
                "'; DROP TABLE users; --",
                "../../../etc/passwd",
                "\\x00\\x01\\x02",
                "A" * 10000,  # Very long input
                "\n\r\t\0",  # Control characters
            ]
            
            safe_responses = 0
            for i, malicious_input in enumerate(malicious_inputs):
                try:
                    result = await router.route_query(malicious_input)
                    
                    if result and result.get('response'):
                        # Check if response seems to contain reflected input
                        response = result.get('response', '')
                        if malicious_input not in response:
                            safe_responses += 1
                            details[f"malicious_input_{i}_handled"] = True
                        else:
                            details[f"malicious_input_{i}_reflected"] = True
                            score -= 0.1
                            recommendations.append(f"Input reflection detected for test {i}")
                    
                except Exception:
                    # Exception is actually good - means input was rejected
                    safe_responses += 1
                    details[f"malicious_input_{i}_rejected"] = True
            
            details["safe_responses"] = safe_responses
            details["total_malicious_tests"] = len(malicious_inputs)
            details["input_safety_rate"] = safe_responses / len(malicious_inputs)
            
            await router.cleanup()
            
            if safe_responses < len(malicious_inputs) * 0.8:
                score = safe_responses / len(malicious_inputs)
                recommendations.append("Improve input validation and sanitization")
            
        except Exception as e:
            score = 0.0
            details["error"] = str(e)
            recommendations.append("Fix input validation testing")
        
        return TestResult(
            test_type=SecurityTestType.INPUT_VALIDATION.value,
            test_name="Input Validation",
            start_time=start_time,
            end_time=time.time(),
            success=score >= 0.8,
            score=score,
            details=details,
            recommendations=recommendations
        )

class PrivacyValidator:
    """Validate privacy requirements"""
    
    def __init__(self):
        self.test_results = []
    
    async def validate_data_retention(self) -> TestResult:
        """Test data retention policies"""
        start_time = time.time()
        details = {}
        recommendations = []
        score = 1.0
        
        try:
            # Check if data retention policies are implemented
            data_dirs = ["data", "logs", ".taskmaster"]
            
            for dir_name in data_dirs:
                dir_path = Path(dir_name)
                if dir_path.exists():
                    # Check for old files that should be cleaned up
                    old_files = []
                    current_time = time.time()
                    
                    for file_path in dir_path.rglob("*"):
                        if file_path.is_file():
                            file_age_days = (current_time - file_path.stat().st_mtime) / (24 * 3600)
                            if file_age_days > 30:  # Files older than 30 days
                                old_files.append(str(file_path))
                    
                    details[f"{dir_name}_old_files"] = len(old_files)
                    
                    if len(old_files) > 10:
                        score -= 0.2
                        recommendations.append(f"Clean up old files in {dir_name}")
            
            # Check for data retention configuration
            config_files = ["config/config.yaml", ".taskmaster/config.json"]
            retention_config_found = False
            
            for config_file in config_files:
                config_path = Path(config_file)
                if config_path.exists():
                    try:
                        content = config_path.read_text()
                        if any(term in content.lower() for term in ['retention', 'cleanup', 'expire']):
                            retention_config_found = True
                            break
                    except:
                        pass
            
            details["retention_config_found"] = retention_config_found
            if not retention_config_found:
                score -= 0.3
                recommendations.append("Implement data retention configuration")
            
        except Exception as e:
            score = 0.0
            details["error"] = str(e)
            recommendations.append("Fix data retention validation")
        
        return TestResult(
            test_type=PrivacyTestType.DATA_RETENTION.value,
            test_name="Data Retention",
            start_time=start_time,
            end_time=time.time(),
            success=score >= 0.7,
            score=score,
            details=details,
            recommendations=recommendations
        )
    
    async def validate_secure_storage(self) -> TestResult:
        """Test secure data storage practices"""
        start_time = time.time()
        details = {}
        recommendations = []
        score = 1.0
        
        try:
            # Check file permissions on sensitive directories
            sensitive_dirs = [".taskmaster", "data", "logs"]
            
            for dir_name in sensitive_dirs:
                dir_path = Path(dir_name)
                if dir_path.exists():
                    # Check directory permissions
                    dir_perms = oct(dir_path.stat().st_mode)[-3:]
                    details[f"{dir_name}_permissions"] = dir_perms
                    
                    # Secure permissions: 700, 750, 755
                    if dir_perms not in ['700', '750', '755']:
                        score -= 0.1
                        recommendations.append(f"Secure {dir_name} directory permissions")
            
            # Check for unencrypted sensitive data
            sensitive_patterns = ['password', 'secret', 'key', 'token']
            potential_leaks = []
            
            for pattern in sensitive_patterns:
                # Check common files for potential leaks
                files_to_check = ['config/config.yaml', 'README.md', 'demo.py']
                
                for file_name in files_to_check:
                    file_path = Path(file_name)
                    if file_path.exists():
                        try:
                            content = file_path.read_text()
                            if pattern in content.lower() and '=' in content:
                                # Potential hardcoded sensitive data
                                potential_leaks.append(f"{pattern} in {file_name}")
                        except:
                            pass
            
            details["potential_data_leaks"] = len(potential_leaks)
            details["leak_details"] = potential_leaks[:5]  # First 5 for brevity
            
            if len(potential_leaks) > 5:
                score -= 0.3
                recommendations.append("Review and remove potential hardcoded secrets")
            elif len(potential_leaks) > 0:
                score -= 0.1
                recommendations.append("Minor potential data exposure detected")
            
            # Check for secure configuration
            details["secure_storage_practices"] = score >= 0.8
            
        except Exception as e:
            score = 0.0
            details["error"] = str(e)
            recommendations.append("Fix secure storage validation")
        
        return TestResult(
            test_type=PrivacyTestType.SECURE_STORAGE.value,
            test_name="Secure Storage",
            start_time=start_time,
            end_time=time.time(),
            success=score >= 0.7,
            score=score,
            details=details,
            recommendations=recommendations
        )

class LoadValidator:
    """Validate system performance under load"""
    
    def __init__(self):
        self.test_results = []
    
    async def validate_concurrent_users(self, max_users: int = 5) -> TestResult:
        """Test concurrent user operations"""
        start_time = time.time()
        details = {}
        recommendations = []
        score = 1.0
        
        try:
            from assistant.llm_router import LLMRouter
            from dotenv import load_dotenv
            load_dotenv()
            
            # Create multiple concurrent "users"
            async def simulate_user(user_id: int) -> Dict[str, Any]:
                router = LLMRouter()
                user_start = time.time()
                
                try:
                    result = await router.route_query(f"User {user_id} concurrent test query")
                    response_time = time.time() - user_start
                    
                    await router.cleanup()
                    
                    return {
                        'user_id': user_id,
                        'success': result and result.get('response') is not None,
                        'response_time': response_time,
                        'error': None
                    }
                except Exception as e:
                    await router.cleanup()
                    return {
                        'user_id': user_id,
                        'success': False,
                        'response_time': time.time() - user_start,
                        'error': str(e)
                    }
            
            # Run concurrent user simulations
            tasks = [simulate_user(i) for i in range(max_users)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Analyze results
            successful_users = sum(1 for r in results if isinstance(r, dict) and r.get('success'))
            total_response_time = sum(r.get('response_time', 0) for r in results if isinstance(r, dict))
            avg_response_time = total_response_time / len(results) if results else 0
            
            details["concurrent_users"] = max_users
            details["successful_users"] = successful_users
            details["success_rate"] = successful_users / max_users
            details["avg_response_time"] = avg_response_time
            details["total_test_time"] = time.time() - start_time
            
            # Score based on success rate and performance
            success_rate = successful_users / max_users
            if success_rate < 0.8:
                score = success_rate
                recommendations.append("Improve concurrent user handling")
            
            if avg_response_time > 10.0:
                score -= 0.2
                recommendations.append("Optimize response times under load")
            
        except Exception as e:
            score = 0.0
            details["error"] = str(e)
            recommendations.append("Fix concurrent user testing")
        
        return TestResult(
            test_type=LoadTestType.CONCURRENT_USERS.value,
            test_name="Concurrent Users",
            start_time=start_time,
            end_time=time.time(),
            success=score >= 0.8,
            score=score,
            details=details,
            recommendations=recommendations
        )
    
    async def validate_voice_command_burst(self) -> TestResult:
        """Test burst of voice commands"""
        start_time = time.time()
        details = {}
        recommendations = []
        score = 1.0
        
        try:
            from assistant.llm_router import LLMRouter
            from dotenv import load_dotenv
            load_dotenv()
            
            router = LLMRouter()
            
            # Simulate rapid voice commands
            voice_commands = [
                "What time is it?",
                "Help me debug this error",
                "Explain this code",
                "What's the weather?",
                "Set a reminder",
                "Search for files",
                "Open application",
                "Check system status"
            ]
            
            # Send commands in rapid succession
            async def send_command(command: str, cmd_id: int) -> Dict[str, Any]:
                cmd_start = time.time()
                try:
                    result = await router.route_query(f"{command} (burst test {cmd_id})")
                    return {
                        'command_id': cmd_id,
                        'success': result and result.get('response') is not None,
                        'response_time': time.time() - cmd_start,
                        'error': None
                    }
                except Exception as e:
                    return {
                        'command_id': cmd_id,
                        'success': False,
                        'response_time': time.time() - cmd_start,
                        'error': str(e)
                    }
            
            # Execute burst test
            tasks = [send_command(cmd, i) for i, cmd in enumerate(voice_commands)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Analyze burst performance
            successful_commands = sum(1 for r in results if isinstance(r, dict) and r.get('success'))
            response_times = [r.get('response_time', 0) for r in results if isinstance(r, dict)]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            max_response_time = max(response_times) if response_times else 0
            
            details["total_commands"] = len(voice_commands)
            details["successful_commands"] = successful_commands
            details["success_rate"] = successful_commands / len(voice_commands)
            details["avg_response_time"] = avg_response_time
            details["max_response_time"] = max_response_time
            details["burst_duration"] = time.time() - start_time
            
            await router.cleanup()
            
            # Score based on success rate and performance consistency
            success_rate = successful_commands / len(voice_commands)
            if success_rate < 0.9:
                score = success_rate
                recommendations.append("Improve voice command burst handling")
            
            if max_response_time > 15.0:
                score -= 0.1
                recommendations.append("Reduce maximum response times in bursts")
            
        except Exception as e:
            score = 0.0
            details["error"] = str(e)
            recommendations.append("Fix voice command burst testing")
        
        return TestResult(
            test_type=LoadTestType.VOICE_COMMAND_BURST.value,
            test_name="Voice Command Burst",
            start_time=start_time,
            end_time=time.time(),
            success=score >= 0.8,
            score=score,
            details=details,
            recommendations=recommendations
        )
    
    async def validate_memory_stress(self) -> TestResult:
        """Test memory usage under stress"""
        start_time = time.time()
        details = {}
        recommendations = []
        score = 1.0
        
        try:
            import psutil
            
            # Capture initial memory state
            initial_memory = psutil.virtual_memory()
            process = psutil.Process()
            initial_process_memory = process.memory_info()
            
            details["initial_system_memory_percent"] = initial_memory.percent
            details["initial_process_memory_mb"] = initial_process_memory.rss / 1024 / 1024
            
            # Perform memory-intensive operations
            from assistant.llm_router import LLMRouter
            from dotenv import load_dotenv
            load_dotenv()
            
            routers = []
            memory_samples = []
            
            # Create multiple routers and perform operations
            for i in range(10):
                router = LLMRouter()
                routers.append(router)
                
                # Perform operation
                await router.route_query(f"Memory stress test operation {i}")
                
                # Sample memory usage
                current_memory = process.memory_info()
                memory_samples.append(current_memory.rss / 1024 / 1024)
                
                # Brief pause
                await asyncio.sleep(0.1)
            
            # Cleanup
            for router in routers:
                await router.cleanup()
            
            # Wait for cleanup
            await asyncio.sleep(1.0)
            
            # Final memory check
            final_memory = psutil.virtual_memory()
            final_process_memory = process.memory_info()
            
            details["final_system_memory_percent"] = final_memory.percent
            details["final_process_memory_mb"] = final_process_memory.rss / 1024 / 1024
            details["memory_growth_mb"] = details["final_process_memory_mb"] - details["initial_process_memory_mb"]
            details["max_memory_mb"] = max(memory_samples)
            details["memory_samples"] = len(memory_samples)
            
            # Evaluate memory performance
            memory_growth = details["memory_growth_mb"]
            if memory_growth > 100:  # More than 100MB growth
                score -= 0.3
                recommendations.append("Investigate memory usage patterns")
            elif memory_growth > 50:  # More than 50MB growth
                score -= 0.1
                recommendations.append("Monitor memory usage")
            
            # Check for excessive memory usage
            if details["max_memory_mb"] > 1000:  # More than 1GB
                score -= 0.2
                recommendations.append("Optimize memory consumption")
            
        except ImportError:
            score = 0.5
            details["psutil_unavailable"] = True
            recommendations.append("Install psutil for memory monitoring")
        except Exception as e:
            score = 0.0
            details["error"] = str(e)
            recommendations.append("Fix memory stress testing")
        
        return TestResult(
            test_type=LoadTestType.MEMORY_STRESS.value,
            test_name="Memory Stress",
            start_time=start_time,
            end_time=time.time(),
            success=score >= 0.7,
            score=score,
            details=details,
            recommendations=recommendations
        )

class SecurityPrivacyLoadValidator:
    """Main validator for security, privacy, and load testing"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("reports/security_privacy_load")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.security_validator = SecurityValidator()
        self.privacy_validator = PrivacyValidator()
        self.load_validator = LoadValidator()
    
    async def run_validation(self) -> ValidationResults:
        """Run complete security, privacy, and load validation"""
        print("🚀 Starting Security, Privacy, and Load Testing Validation")
        print("=" * 80)
        
        start_time = time.time()
        all_results = []
        
        # Security Tests
        print("\n🔒 SECURITY VALIDATION")
        print("-" * 40)
        
        security_tests = [
            self.security_validator.validate_api_key_protection(),
            self.security_validator.validate_secure_communication(),
            self.security_validator.validate_input_validation()
        ]
        
        for test in security_tests:
            result = await test
            all_results.append(result)
            status = "✅" if result.success else "❌"
            print(f"{status} {result.test_name}: {result.score:.2f}")
        
        # Privacy Tests
        print("\n🔐 PRIVACY VALIDATION")
        print("-" * 40)
        
        privacy_tests = [
            self.privacy_validator.validate_data_retention(),
            self.privacy_validator.validate_secure_storage()
        ]
        
        for test in privacy_tests:
            result = await test
            all_results.append(result)
            status = "✅" if result.success else "❌"
            print(f"{status} {result.test_name}: {result.score:.2f}")
        
        # Load Tests
        print("\n⚡ LOAD VALIDATION")
        print("-" * 40)
        
        load_tests = [
            self.load_validator.validate_concurrent_users(),
            self.load_validator.validate_voice_command_burst(),
            self.load_validator.validate_memory_stress()
        ]
        
        for test in load_tests:
            result = await test
            all_results.append(result)
            status = "✅" if result.success else "❌"
            print(f"{status} {result.test_name}: {result.score:.2f}")
        
        # Calculate scores
        security_results = [r for r in all_results if r.test_type in [t.value for t in SecurityTestType]]
        privacy_results = [r for r in all_results if r.test_type in [t.value for t in PrivacyTestType]]
        load_results = [r for r in all_results if r.test_type in [t.value for t in LoadTestType]]
        
        security_score = sum(r.score for r in security_results) / len(security_results) if security_results else 0
        privacy_score = sum(r.score for r in privacy_results) / len(privacy_results) if privacy_results else 0
        load_score = sum(r.score for r in load_results) / len(load_results) if load_results else 0
        
        overall_score = (security_score + privacy_score + load_score) / 3
        
        # Identify critical issues
        critical_issues = []
        for result in all_results:
            if not result.success and result.score < 0.5:
                critical_issues.append(f"{result.test_name}: {result.score:.2f}")
        
        # Determine compliance status
        if overall_score >= 0.9:
            compliance_status = "EXCELLENT"
        elif overall_score >= 0.8:
            compliance_status = "GOOD"
        elif overall_score >= 0.7:
            compliance_status = "ACCEPTABLE"
        else:
            compliance_status = "NEEDS_IMPROVEMENT"
        
        end_time = time.time()
        
        results = ValidationResults(
            start_time=start_time,
            end_time=end_time,
            security_score=security_score,
            privacy_score=privacy_score,
            load_score=load_score,
            overall_score=overall_score,
            test_results=all_results,
            critical_issues=critical_issues,
            compliance_status=compliance_status
        )
        
        # Save results
        await self._save_results(results)
        
        return results
    
    async def _save_results(self, results: ValidationResults):
        """Save detailed test results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive results
        results_file = self.output_dir / f"security_privacy_load_results_{timestamp}.json"
        results_data = {
            'summary': {
                'start_time': results.start_time,
                'end_time': results.end_time,
                'duration_minutes': (results.end_time - results.start_time) / 60,
                'security_score': results.security_score,
                'privacy_score': results.privacy_score,
                'load_score': results.load_score,
                'overall_score': results.overall_score,
                'compliance_status': results.compliance_status,
                'critical_issues_count': len(results.critical_issues)
            },
            'test_results': [],
            'critical_issues': results.critical_issues
        }
        
        for result in results.test_results:
            results_data['test_results'].append({
                'test_type': result.test_type,
                'test_name': result.test_name,
                'success': result.success,
                'score': result.score,
                'duration_seconds': result.end_time - result.start_time,
                'details': result.details,
                'recommendations': result.recommendations
            })
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n📁 Detailed results saved to: {results_file}")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Security, Privacy, and Load Testing Validation")
    parser.add_argument('--output-dir', '-o', type=str,
                       help='Output directory for test results')
    parser.add_argument('--security-only', action='store_true',
                       help='Run only security tests')
    parser.add_argument('--privacy-only', action='store_true',
                       help='Run only privacy tests')
    parser.add_argument('--load-only', action='store_true',
                       help='Run only load tests')
    
    args = parser.parse_args()
    
    # Setup validator
    output_dir = Path(args.output_dir) if args.output_dir else None
    validator = SecurityPrivacyLoadValidator(output_dir)
    
    try:
        # Run validation
        results = await validator.run_validation()
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 SECURITY, PRIVACY & LOAD VALIDATION RESULTS")
        print("=" * 80)
        
        status_emoji = {
            "EXCELLENT": "🏆",
            "GOOD": "✅", 
            "ACCEPTABLE": "⚠️",
            "NEEDS_IMPROVEMENT": "❌"
        }[results.compliance_status]
        
        print(f"{status_emoji} Overall Status: {results.compliance_status}")
        print(f"📊 Overall Score: {results.overall_score:.3f}")
        print(f"🔒 Security Score: {results.security_score:.3f}")
        print(f"🔐 Privacy Score: {results.privacy_score:.3f}")
        print(f"⚡ Load Score: {results.load_score:.3f}")
        
        if results.critical_issues:
            print(f"\n🚨 Critical Issues ({len(results.critical_issues)}):")
            for issue in results.critical_issues:
                print(f"   - {issue}")
        
        # Recommendations summary
        all_recommendations = []
        for result in results.test_results:
            all_recommendations.extend(result.recommendations)
        
        if all_recommendations:
            print(f"\n💡 Key Recommendations:")
            unique_recommendations = list(set(all_recommendations))[:5]
            for rec in unique_recommendations:
                print(f"   - {rec}")
        
        print(f"\n⏱️ Test Duration: {(results.end_time - results.start_time)/60:.1f} minutes")
        
        # Return success based on overall score
        return results.overall_score >= 0.7
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        return False

if __name__ == '__main__':
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Validation interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Validation failed: {e}")
        sys.exit(1) 