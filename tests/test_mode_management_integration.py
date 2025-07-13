#!/usr/bin/env python3
"""
Comprehensive Integration Tests for Mode Management System

Tests the complete mode management system including:
- ModeManager functionality and state management
- ModeValidator validation and error handling
- Dashboard API endpoints and UI integration
- Mode persistence and configuration handling
- Performance metrics and health monitoring

This test suite ensures all components work together seamlessly.
"""

import pytest
import asyncio
import json
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

# Import the components we're testing
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'assistant'))

from config_manager import SovereignConfig, OperationMode, EnvironmentType
from mode_switch_manager import ModeManager, ModeValidationError, ModeTransitionError
from fallback_detector import ModeValidator, ValidationSeverity, FailureType
from dashboard_server import DashboardServer


class TestModeManagerIntegration:
    """Test ModeManager functionality and integration"""
    
    def setup_method(self):
        """Setup for each test"""
        # Create test configuration
        self.config = SovereignConfig(
            environment=EnvironmentType.TESTING,
            operation_mode=OperationMode.HYBRID_AUTO
        )
        self.config.development.mock_apis = True
        
        # Reset singleton instance
        ModeManager._instance = None
        ModeManager._lock = type(ModeManager._lock)()
        
        self.mode_manager = ModeManager.get_instance()
        
    def test_singleton_pattern(self):
        """Test that ModeManager implements singleton correctly"""
        manager1 = ModeManager.get_instance()
        manager2 = ModeManager.get_instance()
        assert manager1 is manager2
        
    def test_mode_initialization(self):
        """Test mode manager initialization"""
        assert not self.mode_manager.is_initialized()
        
        # Initialize with config
        success = self.mode_manager.initialize(self.config)
        assert success
        assert self.mode_manager.is_initialized()
        assert self.mode_manager.get_current_mode() == OperationMode.HYBRID_AUTO
        
    def test_mode_capabilities(self):
        """Test mode capabilities for each operation mode"""
        self.mode_manager.initialize(self.config)
        
        # Test REALTIME_ONLY capabilities
        capabilities = self.mode_manager.get_capabilities(OperationMode.REALTIME_ONLY)
        assert capabilities.can_use_realtime_api
        assert not capabilities.can_use_traditional_pipeline
        assert capabilities.supports_screen_context
        assert capabilities.supports_memory_injection
        
        # Test TRADITIONAL_ONLY capabilities
        capabilities = self.mode_manager.get_capabilities(OperationMode.TRADITIONAL_ONLY)
        assert not capabilities.can_use_realtime_api
        assert capabilities.can_use_traditional_pipeline
        assert capabilities.supports_screen_context
        assert capabilities.supports_memory_injection
        
        # Test HYBRID_AUTO capabilities
        capabilities = self.mode_manager.get_capabilities(OperationMode.HYBRID_AUTO)
        assert capabilities.can_use_realtime_api
        assert capabilities.can_use_traditional_pipeline
        assert capabilities.supports_screen_context
        assert capabilities.supports_memory_injection
    
    @pytest.mark.asyncio
    async def test_mode_switching(self):
        """Test mode switching functionality"""
        self.mode_manager.initialize(self.config)
        
        # Test switch to REALTIME_ONLY
        success = await self.mode_manager.switch_mode(
            OperationMode.REALTIME_ONLY, 
            "Test switch to realtime"
        )
        assert success
        assert self.mode_manager.get_current_mode() == OperationMode.REALTIME_ONLY
        
        # Test switch to TRADITIONAL_ONLY
        success = await self.mode_manager.switch_mode(
            OperationMode.TRADITIONAL_ONLY, 
            "Test switch to traditional"
        )
        assert success
        assert self.mode_manager.get_current_mode() == OperationMode.TRADITIONAL_ONLY
    
    def test_mode_metrics_tracking(self):
        """Test mode metrics collection and reporting"""
        self.mode_manager.initialize(self.config)
        
        # Record some test metrics
        self.mode_manager.record_session_metrics(
            OperationMode.REALTIME_ONLY,
            success=True,
            response_time=0.25,
            cost=0.05
        )
        
        metrics = self.mode_manager.get_mode_metrics()
        assert OperationMode.REALTIME_ONLY in metrics
        
        mode_metrics = metrics[OperationMode.REALTIME_ONLY]
        assert mode_metrics['total_sessions'] == 1
        assert mode_metrics['success_rate'] == 1.0
        assert mode_metrics['avg_response_time'] == 0.25
        assert mode_metrics['estimated_cost'] == 0.05
    
    def test_mode_transition_history(self):
        """Test mode transition history tracking"""
        self.mode_manager.initialize(self.config)
        
        # Get initial status
        status = self.mode_manager.get_status()
        initial_transitions = len(status['transition_history'])
        
        # Perform mode switches
        asyncio.run(self.mode_manager.switch_mode(
            OperationMode.REALTIME_ONLY, 
            "Test transition 1"
        ))
        asyncio.run(self.mode_manager.switch_mode(
            OperationMode.TRADITIONAL_ONLY, 
            "Test transition 2"
        ))
        
        # Check transition history
        status = self.mode_manager.get_status()
        transitions = status['transition_history']
        assert len(transitions) == initial_transitions + 2
        
        # Check last transition
        last_transition = transitions[-1]
        assert last_transition['to_mode'] == OperationMode.TRADITIONAL_ONLY.value
        assert last_transition['reason'] == "Test transition 2"


class TestModeValidatorIntegration:
    """Test ModeValidator functionality and integration"""
    
    def setup_method(self):
        """Setup for each test"""
        # Create test configuration
        self.config = SovereignConfig(
            environment=EnvironmentType.TESTING,
            operation_mode=OperationMode.HYBRID_AUTO
        )
        self.config.development.mock_apis = True
        
        # Setup mode manager
        ModeManager._instance = None
        ModeManager._lock = type(ModeManager._lock)()
        self.mode_manager = ModeManager.get_instance()
        self.mode_manager.initialize(self.config)
        
        # Create validator
        self.validator = ModeValidator(self.config, self.mode_manager)
    
    @pytest.mark.asyncio
    async def test_mode_validation_realtime_only(self):
        """Test validation for REALTIME_ONLY mode"""
        is_valid, issues = await self.validator.validate_mode(OperationMode.REALTIME_ONLY)
        
        # In test mode with mock APIs, should be valid
        assert is_valid
        
        # Should have no critical issues
        critical_issues = [issue for issue in issues if issue.severity == ValidationSeverity.CRITICAL]
        assert len(critical_issues) == 0
    
    @pytest.mark.asyncio
    async def test_mode_validation_traditional_only(self):
        """Test validation for TRADITIONAL_ONLY mode"""
        is_valid, issues = await self.validator.validate_mode(OperationMode.TRADITIONAL_ONLY)
        
        # May have issues due to missing STT/TTS configuration, but should not be critical
        critical_issues = [issue for issue in issues if issue.severity == ValidationSeverity.CRITICAL]
        assert len(critical_issues) == 0
    
    @pytest.mark.asyncio
    async def test_mode_validation_hybrid_auto(self):
        """Test validation for HYBRID_AUTO mode"""
        is_valid, issues = await self.validator.validate_mode(OperationMode.HYBRID_AUTO)
        
        # Hybrid mode should be more forgiving
        critical_issues = [issue for issue in issues if issue.severity == ValidationSeverity.CRITICAL]
        assert len(critical_issues) == 0
    
    @pytest.mark.asyncio
    async def test_api_availability_checking(self):
        """Test API availability checking"""
        # Test OpenAI API status
        status = await self.validator.check_openai_api_availability()
        assert status.service_name == "openai_api"
        assert status.is_available  # Should be true in mock mode
        assert status.response_time_ms is not None
    
    @pytest.mark.asyncio
    async def test_traditional_pipeline_checking(self):
        """Test traditional pipeline availability checking"""
        availability = await self.validator.check_traditional_pipeline_availability()
        
        assert 'stt_available' in availability
        assert 'tts_available' in availability
        assert 'llm_available' in availability
        
        # In mock mode, should all be available
        assert availability['stt_available']
        assert availability['tts_available']
        assert availability['llm_available']
    
    @pytest.mark.asyncio
    async def test_mode_failure_handling(self):
        """Test mode failure and graceful degradation"""
        # Simulate a failure in REALTIME_ONLY mode
        test_error = Exception("Simulated API failure")
        
        success = await self.validator.handle_mode_failure(
            OperationMode.REALTIME_ONLY,
            test_error,
            FailureType.API_UNAVAILABLE
        )
        
        # Should successfully degrade to traditional mode
        assert success
        
        # Mode manager should have switched to traditional mode
        current_mode = self.mode_manager.get_current_mode()
        assert current_mode == OperationMode.TRADITIONAL_ONLY
    
    def test_health_summary(self):
        """Test mode health summary generation"""
        summary = self.validator.get_mode_health_summary()
        
        # Should have entries for all modes
        for mode in OperationMode:
            assert mode.value in summary
            mode_data = summary[mode.value]
            assert 'mode' in mode_data
            assert 'failure_count' in mode_data
            assert 'in_cooldown' in mode_data


class TestDashboardIntegration:
    """Test Dashboard API integration with mode management"""
    
    def setup_method(self):
        """Setup for each test"""
        # Create test configuration
        self.config = SovereignConfig(
            environment=EnvironmentType.TESTING,
            operation_mode=OperationMode.HYBRID_AUTO
        )
        self.config.development.mock_apis = True
        
        # Setup mode components
        ModeManager._instance = None
        ModeManager._lock = type(ModeManager._lock)()
        self.mode_manager = ModeManager.get_instance()
        self.mode_manager.initialize(self.config)
        
        self.validator = ModeValidator(self.config, self.mode_manager)
        
        # Create dashboard server for testing
        self.dashboard = DashboardServer(
            metrics_collector=None,
            mode_manager=self.mode_manager,
            mode_validator=self.validator,
            host="localhost",
            port=8080,
            debug=True
        )
        
        # Setup Flask test client
        self.dashboard._setup_routes()
        self.client = self.dashboard.app.test_client()
    
    def test_mode_status_endpoint(self):
        """Test GET /api/mode/status endpoint"""
        response = self.client.get('/api/mode/status')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'current_mode' in data
        assert 'capabilities' in data
        assert 'metrics' in data
        assert 'health' in data
        assert 'status' in data
        assert 'timestamp' in data
        
        # Verify current mode
        assert data['current_mode'] == OperationMode.HYBRID_AUTO.value
    
    def test_mode_switch_endpoint(self):
        """Test POST /api/mode/switch endpoint"""
        # Test valid mode switch
        response = self.client.post('/api/mode/switch', 
                                  json={'mode': 'realtime_only', 'reason': 'Test switch'})
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['success'] is True
        assert data['mode'] == 'realtime_only'
        
        # Verify mode actually changed
        current_mode = self.mode_manager.get_current_mode()
        assert current_mode == OperationMode.REALTIME_ONLY
    
    def test_mode_switch_invalid_mode(self):
        """Test mode switch with invalid mode"""
        response = self.client.post('/api/mode/switch', 
                                  json={'mode': 'invalid_mode'})
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data
        assert 'valid_modes' in data
    
    def test_mode_validate_endpoint(self):
        """Test POST /api/mode/validate endpoint"""
        response = self.client.post('/api/mode/validate', 
                                  json={'mode': 'traditional_only'})
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'mode' in data
        assert 'is_valid' in data
        assert 'issues' in data
        assert 'timestamp' in data
        
        assert data['mode'] == 'traditional_only'
        assert isinstance(data['is_valid'], bool)
        assert isinstance(data['issues'], list)
    
    def test_available_modes_endpoint(self):
        """Test GET /api/mode/available endpoint"""
        response = self.client.get('/api/mode/available')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'modes' in data
        assert 'timestamp' in data
        
        modes = data['modes']
        assert len(modes) == 3  # Should have all three operation modes
        
        # Check mode structure
        for mode in modes:
            assert 'value' in mode
            assert 'name' in mode
            assert 'description' in mode
            assert 'is_available' in mode
            assert 'validation_summary' in mode
    
    def test_mode_endpoints_without_manager(self):
        """Test mode endpoints when mode manager is not available"""
        # Create dashboard without mode manager
        dashboard_no_mode = DashboardServer(
            metrics_collector=None,
            mode_manager=None,
            mode_validator=None,
            host="localhost",
            port=8080,
            debug=True
        )
        dashboard_no_mode._setup_routes()
        client = dashboard_no_mode.app.test_client()
        
        # Test endpoints return 503 when manager not available
        response = client.get('/api/mode/status')
        assert response.status_code == 503
        
        response = client.post('/api/mode/switch', json={'mode': 'realtime_only'})
        assert response.status_code == 503
        
        response = client.post('/api/mode/validate', json={'mode': 'realtime_only'})
        assert response.status_code == 503


class TestModePersistence:
    """Test mode persistence and state management"""
    
    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.yaml')
        
    def teardown_method(self):
        """Cleanup after each test"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_mode_configuration_persistence(self):
        """Test that mode preferences persist in configuration"""
        # Create config with specific mode
        config = SovereignConfig(
            environment=EnvironmentType.TESTING,
            operation_mode=OperationMode.REALTIME_ONLY
        )
        
        # Verify mode is set correctly
        assert config.operation_mode == OperationMode.REALTIME_ONLY
        
        # Test serialization/deserialization
        config_dict = {
            'environment': config.environment.value,
            'operation_mode': config.operation_mode.value
        }
        
        # Recreate from dict (simulating config file load)
        new_mode = OperationMode(config_dict['operation_mode'])
        assert new_mode == OperationMode.REALTIME_ONLY
    
    def test_mode_manager_state_persistence(self):
        """Test that mode manager state can be saved and restored"""
        # Create and initialize mode manager
        ModeManager._instance = None
        ModeManager._lock = type(ModeManager._lock)()
        mode_manager = ModeManager.get_instance()
        
        config = SovereignConfig(
            environment=EnvironmentType.TESTING,
            operation_mode=OperationMode.HYBRID_AUTO
        )
        mode_manager.initialize(config)
        
        # Switch mode and record metrics
        asyncio.run(mode_manager.switch_mode(
            OperationMode.REALTIME_ONLY, 
            "Test persistence"
        ))
        
        mode_manager.record_session_metrics(
            OperationMode.REALTIME_ONLY,
            success=True,
            response_time=0.3,
            cost=0.06
        )
        
        # Get current state
        status = mode_manager.get_status()
        metrics = mode_manager.get_mode_metrics()
        
        # Verify state can be serialized
        state_data = {
            'current_mode': mode_manager.get_current_mode().value,
            'transition_history': status['transition_history'],
            'metrics': {
                mode.value: mode_metrics 
                for mode, mode_metrics in metrics.items()
            }
        }
        
        # Serialize to JSON (simulating persistence)
        serialized = json.dumps(state_data, default=str)
        assert serialized is not None
        
        # Deserialize (simulating restoration)
        restored_data = json.loads(serialized)
        assert restored_data['current_mode'] == OperationMode.REALTIME_ONLY.value
        assert len(restored_data['transition_history']) > 0
        assert OperationMode.REALTIME_ONLY.value in restored_data['metrics']


class TestModePerformance:
    """Test mode system performance and metrics"""
    
    def setup_method(self):
        """Setup for each test"""
        self.config = SovereignConfig(
            environment=EnvironmentType.TESTING,
            operation_mode=OperationMode.HYBRID_AUTO
        )
        self.config.development.mock_apis = True
        
        ModeManager._instance = None
        ModeManager._lock = type(ModeManager._lock)()
        self.mode_manager = ModeManager.get_instance()
        self.mode_manager.initialize(self.config)
        
        self.validator = ModeValidator(self.config, self.mode_manager)
    
    @pytest.mark.asyncio
    async def test_mode_switching_performance(self):
        """Test mode switching performance"""
        import time
        
        # Measure mode switch time
        start_time = time.time()
        success = await self.mode_manager.switch_mode(
            OperationMode.REALTIME_ONLY, 
            "Performance test"
        )
        switch_time = time.time() - start_time
        
        assert success
        assert switch_time < 1.0  # Should switch within 1 second
    
    @pytest.mark.asyncio
    async def test_validation_performance(self):
        """Test mode validation performance"""
        import time
        
        # Test validation speed for all modes
        for mode in OperationMode:
            start_time = time.time()
            is_valid, issues = await self.validator.validate_mode(mode)
            validation_time = time.time() - start_time
            
            assert validation_time < 2.0  # Should validate within 2 seconds
    
    def test_metrics_collection_performance(self):
        """Test metrics collection performance"""
        import time
        
        # Record many metrics quickly
        start_time = time.time()
        for i in range(100):
            self.mode_manager.record_session_metrics(
                OperationMode.REALTIME_ONLY,
                success=i % 2 == 0,
                response_time=0.1 + (i * 0.001),
                cost=0.01 + (i * 0.0001)
            )
        collection_time = time.time() - start_time
        
        assert collection_time < 1.0  # Should collect 100 metrics within 1 second
        
        # Verify metrics accuracy
        metrics = self.mode_manager.get_mode_metrics()
        mode_metrics = metrics[OperationMode.REALTIME_ONLY]
        assert mode_metrics['total_sessions'] == 100
        assert 0.4 < mode_metrics['success_rate'] < 0.6  # Should be around 50%


# Utility functions for test setup
def create_test_config(mode: OperationMode = OperationMode.HYBRID_AUTO) -> SovereignConfig:
    """Create a test configuration"""
    config = SovereignConfig(
        environment=EnvironmentType.TESTING,
        operation_mode=mode
    )
    config.development.mock_apis = True
    return config


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 