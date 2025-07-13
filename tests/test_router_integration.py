"""
Integration test for RouterBlock with FallbackManager

This test verifies that the RouterBlock correctly integrates with the FallbackManager
and that the comprehensive fallback logic is working as expected.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from assistant.router_block import RouterBlock, RouterState, create_router_block
from assistant.router_config import RouterConfig
from assistant.fallback_manager import FallbackConfig, CircuitBreakerConfig, RetryConfig
from assistant.llm_router import QueryClassification, QueryComplexity
from assistant.async_processor import ProcessingConfig


class TestRouterBlockFallbackIntegration:
    """Test RouterBlock integration with FallbackManager"""
    
    def test_router_block_with_fallback_manager_initialization(self):
        """Test RouterBlock initializes correctly with FallbackManager"""
        fallback_config = FallbackConfig(fallback_chain=['fast', 'deep', 'backup'])
        circuit_config = CircuitBreakerConfig(failure_threshold=3)
        retry_config = RetryConfig(max_attempts=2)
        
        router = create_router_block(
            fallback_config=fallback_config,
            circuit_breaker_config=circuit_config,
            retry_config=retry_config
        )
        
        assert router.fallback_manager is not None
        assert router.fallback_manager.config.fallback_chain == ['fast', 'deep', 'backup']
        assert 'fast' in router.fallback_manager.circuit_breakers
        assert 'deep' in router.fallback_manager.circuit_breakers
        assert 'backup' in router.fallback_manager.circuit_breakers
    
    def test_router_metrics_include_fallback_stats(self):
        """Test router metrics include fallback manager statistics"""
        router = create_router_block()
        
        metrics = router.get_metrics()
        
        assert 'fallback_management' in metrics
        assert 'services' in metrics['fallback_management']
        assert 'overall_stats' in metrics['fallback_management']
        assert 'fast' in metrics['fallback_management']['services']
        assert 'deep' in metrics['fallback_management']['services']
        assert 'backup' in metrics['fallback_management']['services']
    
    def test_fallback_manager_service_functions_created(self):
        """Test that service functions are properly created for FallbackManager"""
        router = create_router_block()
        
        # Mock the _execute_with_service method to verify it's called
        router._execute_with_service = AsyncMock(return_value="test_result")
        
        # Test that we can create service functions
        text = "test query"
        classification = QueryClassification(
            complexity=QueryComplexity.SIMPLE,
            confidence=0.95,
            reasoning="Simple test",
            factors={'length': 10}
        )
        
        # This should not raise an error
        service_func = lambda: router._execute_with_service(
            text, classification, router.fast_llm_service, "downstream"
        )
        
        # Verify the service function can be called
        assert asyncio.iscoroutinefunction(service_func)
    
    def test_fallback_manager_error_handling(self):
        """Test that FallbackManager error handling is properly integrated"""
        router = create_router_block()
        
        # Test that the router has the proper error types available
        from assistant.fallback_manager import ServiceUnavailableError, AllServicesFailed
        
        # These should be importable and available
        assert ServiceUnavailableError is not None
        assert AllServicesFailed is not None
        
        # Test that the router can handle these error types
        assert hasattr(router, 'fallback_manager')
        assert hasattr(router.fallback_manager, 'execute_with_fallback')
    
    def test_circuit_breaker_service_names_match(self):
        """Test that circuit breaker service names match expected services"""
        router = create_router_block()
        
        circuit_breakers = router.fallback_manager.circuit_breakers
        
        # Should have circuit breakers for all expected services
        expected_services = ['fast', 'deep', 'backup']
        for service_name in expected_services:
            assert service_name in circuit_breakers
            assert circuit_breakers[service_name].service_name == service_name
    
    def test_fallback_chain_configuration(self):
        """Test fallback chain configuration"""
        custom_chain = ['primary', 'secondary', 'tertiary']
        fallback_config = FallbackConfig(fallback_chain=custom_chain)
        
        router = create_router_block(fallback_config=fallback_config)
        
        assert router.fallback_manager.config.fallback_chain == custom_chain
        
        # Should have circuit breakers for all services in chain
        for service_name in custom_chain:
            assert service_name in router.fallback_manager.circuit_breakers
    
    def test_factory_function_with_fallback_configs(self):
        """Test factory function creates router with fallback configurations"""
        fallback_config = FallbackConfig(max_fallback_attempts=5)
        circuit_config = CircuitBreakerConfig(failure_threshold=10)
        retry_config = RetryConfig(max_attempts=3)
        
        router = create_router_block(
            fallback_config=fallback_config,
            circuit_breaker_config=circuit_config,
            retry_config=retry_config
        )
        
        assert router.fallback_manager.config.max_fallback_attempts == 5
        assert router.fallback_manager.circuit_breakers['fast'].config.failure_threshold == 10
        assert router.fallback_manager.retry_handler.config.max_attempts == 3
    
    def test_health_monitoring_integration(self):
        """Test health monitoring integration"""
        router = create_router_block()
        
        # Get health information
        health_info = router.fallback_manager.get_service_health()
        
        assert 'services' in health_info
        assert 'overall_stats' in health_info
        
        # Check that all expected services are monitored
        for service_name in ['fast', 'deep', 'backup']:
            assert service_name in health_info['services']
            service_info = health_info['services'][service_name]
            assert 'state' in service_info
            assert 'metrics' in service_info
            assert 'time_in_current_state' in service_info
    
    def test_error_message_manager_integration(self):
        """Test error message manager integration"""
        router = create_router_block()
        
        # Test that error message manager is available
        assert hasattr(router.fallback_manager, 'error_manager')
        
        # Test that it can generate user-friendly messages
        message = router.fallback_manager.error_manager.get_user_friendly_message('timeout')
        assert isinstance(message, str)
        assert len(message) > 0
        
        # Test with context
        context_message = router.fallback_manager.error_manager.get_user_friendly_message(
            'service_unavailable',
            {'service_name': 'fast'}
        )
        assert isinstance(context_message, str)
        assert len(context_message) > 0
    
    def test_router_state_management_preserved(self):
        """Test that router state management is preserved"""
        router = create_router_block()
        
        # Should start in IDLE state
        assert router.state == RouterState.IDLE
        
        # Should have state management attributes
        assert hasattr(router, 'current_request_id')
        assert hasattr(router, 'pending_requests')
        assert router.current_request_id is None
        assert len(router.pending_requests) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 