"""
Test Suite for InfluxDB Metrics Storage

Comprehensive tests for InfluxDB time-series storage integration including
connection management, data writing, querying, and error handling scenarios.
"""

import pytest
import asyncio
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock, call
from collections import deque
import statistics

# Import the module under test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.influxdb_metrics_storage import (
    InfluxDBMetricsStorage, InfluxDBConfig, RetentionPolicy, InfluxDBWriteStats,
    RealtimeMetricsWithInfluxDB, create_influxdb_storage
)

# Mock InfluxDB classes for testing
class MockPoint:
    def __init__(self, measurement):
        self.measurement = measurement
        self._tags = {}
        self._fields = {}
        self._time = None
    
    def tag(self, key, value):
        self._tags[key] = value
        return self
    
    def field(self, key, value):
        self._fields[key] = value
        return self
    
    def time(self, timestamp, precision=None):
        self._time = timestamp
        return self

class MockInfluxDBClient:
    def __init__(self, *args, **kwargs):
        self.health_response = Mock()
        self.health_response.status = "pass"
        self.health_response.message = "ready for queries and writes"
    
    def health(self):
        return self.health_response
    
    def write_api(self, **kwargs):
        return Mock()
    
    def query_api(self):
        return Mock()
    
    def buckets_api(self):
        return Mock()
    
    def close(self):
        pass

class MockBucket:
    def __init__(self, **kwargs):
        self.name = kwargs.get('name')
        self.org_id = kwargs.get('org_id')
        self.retention_rules = kwargs.get('retention_rules', [])
        self.description = kwargs.get('description', '')

class MockRetentionRule:
    def __init__(self, **kwargs):
        self.type = kwargs.get('type')
        self.every_seconds = kwargs.get('every_seconds')

# Mock the realtime metrics classes
class MockRealtimeLatencyMetrics:
    def __init__(self):
        self.voice_to_voice_latency_ms = deque([100, 150, 200, 120, 180], maxlen=1000)
        self.audio_processing_latency_ms = deque([50, 60, 55, 48, 52], maxlen=1000)
        self.total_round_trip_ms = deque([250, 300, 280, 260, 290], maxlen=1000)
    
    def get_percentiles(self, metric_type: str) -> Dict[str, float]:
        if hasattr(self, metric_type):
            data = list(getattr(self, metric_type))
            if len(data) < 2:
                return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
            return {
                "p50": statistics.median(data),
                "p95": sorted(data)[int(len(data) * 0.95)] if len(data) > 1 else data[0],
                "p99": sorted(data)[int(len(data) * 0.99)] if len(data) > 1 else data[0]
            }
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

class MockRealtimeConnectionMetrics:
    def __init__(self):
        self.connection_state = Mock()
        self.connection_state.value = "connected"
        self.total_connections = 10
        self.successful_connections = 9
        self.failed_connections = 1
        self.reconnection_attempts = 2
        self.total_uptime_seconds = 3600
        self.heartbeat_responses = deque([1, 2, 3, 4, 5], maxlen=50)
        
    @property
    def connection_success_rate(self):
        return 90.0
    
    @property
    def average_session_duration(self):
        return 600.0

class MockRealtimeAudioMetrics:
    def __init__(self):
        self.audio_samples_processed = 1000
        self.audio_quality_scores = deque([0.8, 0.9, 0.85, 0.88, 0.92], maxlen=100)
        self.volume_levels = deque([0.5, 0.6, 0.55, 0.58, 0.52], maxlen=200)
        self.silence_detection_events = 5
        self.audio_interruptions = 2
        self.sample_rate = 24000
        self.bit_depth = 16
        
    @property
    def average_audio_quality(self):
        return 0.87

class MockRealtimeCostMetrics:
    def __init__(self):
        self.total_input_tokens = 1000
        self.total_output_tokens = 2000
        self.total_api_calls = 50
        self.hourly_costs = {"2024-07-13-10": 0.50, "2024-07-13-11": 0.75}
        self.daily_budget = 50.0
        
    @property
    def total_cost(self):
        return 1.25
    
    @property
    def current_hour_cost(self):
        return 0.75
    
    @property
    def tokens_per_dollar(self):
        return 2400.0


class TestInfluxDBConfig:
    """Test InfluxDB configuration class"""
    
    def test_default_config_creation(self):
        config = InfluxDBConfig()
        assert config.url == "http://localhost:8086"
        assert config.token is None
        assert config.org == "sovereign"
        assert config.bucket == "realtime_metrics"
        assert config.timeout == 10000
        assert config.batch_size == 1000
        assert config.flush_interval == 1000
        assert config.retry_attempts == 3
        assert config.enable_debug is False
    
    def test_custom_config_creation(self):
        config = InfluxDBConfig(
            url="http://custom:8087",
            token="test-token",
            org="test-org",
            bucket="test-bucket",
            timeout=5000,
            batch_size=500,
            flush_interval=2000,
            retry_attempts=5,
            enable_debug=True
        )
        assert config.url == "http://custom:8087"
        assert config.token == "test-token"
        assert config.org == "test-org"
        assert config.bucket == "test-bucket"
        assert config.timeout == 5000
        assert config.batch_size == 500
        assert config.flush_interval == 2000
        assert config.retry_attempts == 5
        assert config.enable_debug is True


class TestRetentionPolicy:
    """Test retention policy enumeration"""
    
    def test_retention_policy_values(self):
        assert RetentionPolicy.RAW_METRICS.value == "7d"
        assert RetentionPolicy.HOURLY_AGGREGATES.value == "30d"
        assert RetentionPolicy.DAILY_AGGREGATES.value == "1y"
        assert RetentionPolicy.COST_TRACKING.value == "2y"
        assert RetentionPolicy.ERROR_LOGS.value == "90d"


class TestInfluxDBWriteStats:
    """Test InfluxDB write statistics tracking"""
    
    def test_default_stats_creation(self):
        stats = InfluxDBWriteStats()
        assert stats.total_points_written == 0
        assert stats.successful_writes == 0
        assert stats.failed_writes == 0
        assert stats.last_write_time is None
        assert stats.write_durations == []
        assert stats.connection_errors == 0
        assert stats.batch_queue_size == 0
    
    def test_average_write_duration_empty(self):
        stats = InfluxDBWriteStats()
        assert stats.average_write_duration == 0.0
    
    def test_average_write_duration_with_data(self):
        stats = InfluxDBWriteStats()
        stats.write_durations = [0.1, 0.2, 0.3, 0.4, 0.5]
        assert stats.average_write_duration == 0.3
    
    def test_write_success_rate_no_attempts(self):
        stats = InfluxDBWriteStats()
        assert stats.write_success_rate == 100.0
    
    def test_write_success_rate_with_attempts(self):
        stats = InfluxDBWriteStats()
        stats.successful_writes = 8
        stats.failed_writes = 2
        assert stats.write_success_rate == 80.0
    
    def test_write_success_rate_all_successful(self):
        stats = InfluxDBWriteStats()
        stats.successful_writes = 10
        stats.failed_writes = 0
        assert stats.write_success_rate == 100.0


class TestInfluxDBMetricsStorage:
    """Test main InfluxDB metrics storage functionality"""
    
    @pytest.fixture
    def mock_influxdb_modules(self):
        """Mock InfluxDB modules and classes"""
        with patch.multiple(
            'assistant.influxdb_metrics_storage',
            HAS_INFLUXDB=True,
            InfluxDBClient=MockInfluxDBClient,
            Point=MockPoint,
            Bucket=MockBucket,
            RetentionRule=MockRetentionRule
        ):
            yield
    
    @pytest.fixture
    def config(self):
        return InfluxDBConfig(
            url="http://test:8086",
            token="test-token",
            org="test-org",
            bucket="test-bucket",
            batch_size=10,
            flush_interval=500
        )
    
    @pytest.fixture
    def storage(self, config, mock_influxdb_modules):
        with patch('assistant.influxdb_metrics_storage.InfluxDBMetricsStorage._initialize_connection') as mock_init:
            mock_init.return_value = True
            storage = InfluxDBMetricsStorage(config)
            storage._is_connected = True
            storage._write_api = Mock()
            storage._query_api = Mock()
            storage._buckets_api = Mock()
            yield storage
    
    def test_initialization_without_influxdb(self):
        """Test initialization when InfluxDB client is not available"""
        config = InfluxDBConfig()
        with patch('assistant.influxdb_metrics_storage.HAS_INFLUXDB', False):
            storage = InfluxDBMetricsStorage(config)
            assert storage._client is None
            assert not storage._is_connected
    
    def test_initialization_with_influxdb(self, config, mock_influxdb_modules):
        """Test successful initialization with InfluxDB client"""
        with patch.object(InfluxDBMetricsStorage, '_initialize_connection', return_value=True):
            storage = InfluxDBMetricsStorage(config)
            assert storage.config == config
            assert isinstance(storage.write_stats, InfluxDBWriteStats)
    
    def test_write_latency_metrics(self, storage):
        """Test writing latency metrics to InfluxDB"""
        metrics = MockRealtimeLatencyMetrics()
        session_id = "test-session-123"
        
        storage.write_latency_metrics(session_id, metrics)
        
        # Check that points were queued for writing
        assert storage.write_stats.batch_queue_size > 0
    
    def test_write_connection_metrics(self, storage):
        """Test writing connection metrics to InfluxDB"""
        metrics = MockRealtimeConnectionMetrics()
        session_id = "test-session-123"
        
        storage.write_connection_metrics(session_id, metrics)
        
        # Check that points were queued for writing
        assert storage.write_stats.batch_queue_size > 0
    
    def test_write_audio_metrics(self, storage):
        """Test writing audio metrics to InfluxDB"""
        metrics = MockRealtimeAudioMetrics()
        session_id = "test-session-123"
        
        storage.write_audio_metrics(session_id, metrics)
        
        # Check that points were queued for writing
        assert storage.write_stats.batch_queue_size > 0
    
    def test_write_cost_metrics(self, storage):
        """Test writing cost metrics to InfluxDB"""
        metrics = MockRealtimeCostMetrics()
        session_id = "test-session-123"
        
        storage.write_cost_metrics(session_id, metrics)
        
        # Check that points were queued for writing
        assert storage.write_stats.batch_queue_size > 0
    
    def test_batch_writing_threshold(self, storage):
        """Test that batch writing is triggered at threshold"""
        metrics = MockRealtimeLatencyMetrics()
        session_id = "test-session-123"
        
        # Fill queue to batch size threshold
        for i in range(storage.config.batch_size):
            storage.write_latency_metrics(session_id, metrics)
        
        # Should have triggered a flush
        assert storage.write_stats.total_points_written > 0
    
    def test_query_latency_history(self, storage):
        """Test querying latency history"""
        # Mock query API response
        mock_record = Mock()
        mock_record.get_time.return_value = datetime.utcnow()
        mock_record.values = {"session_id": "test-session", "metric_type": "voice_to_voice"}
        mock_record.get_field.return_value = "p50"
        mock_record.get_value.return_value = 150.0
        
        mock_table = Mock()
        mock_table.records = [mock_record]
        
        storage._query_api.query.return_value = [mock_table]
        
        result = storage.query_latency_history(hours_back=24)
        
        assert len(result) == 1
        assert result[0]["session_id"] == "test-session"
        assert result[0]["metric_type"] == "voice_to_voice"
        assert result[0]["field"] == "p50"
        assert result[0]["value"] == 150.0
    
    def test_query_cost_analysis(self, storage):
        """Test querying cost analysis"""
        # Mock query API response
        mock_record = Mock()
        mock_record.get_time.return_value = datetime.utcnow()
        mock_record.get_field.return_value = "total_cost"
        mock_record.get_value.return_value = 5.0
        
        mock_table = Mock()
        mock_table.records = [mock_record]
        
        storage._query_api.query.return_value = [mock_table]
        
        result = storage.query_cost_analysis(days_back=7)
        
        assert "daily_costs" in result
        assert "total_cost" in result
        assert "average_daily_cost" in result
        assert "days_analyzed" in result
    
    def test_query_connection_stability(self, storage):
        """Test querying connection stability"""
        # Mock query API response
        mock_record1 = Mock()
        mock_record1.get_field.return_value = "connection_success_rate"
        mock_record1.get_value.return_value = 95.0
        
        mock_record2 = Mock()
        mock_record2.get_field.return_value = "average_session_duration"
        mock_record2.get_value.return_value = 600.0
        
        mock_table = Mock()
        mock_table.records = [mock_record1, mock_record2]
        
        storage._query_api.query.return_value = [mock_table]
        
        result = storage.query_connection_stability(hours_back=24)
        
        assert "average_success_rate" in result
        assert "average_session_duration" in result
        assert "hours_analyzed" in result
        assert result["average_success_rate"] == 95.0
        assert result["average_session_duration"] == 600.0
    
    def test_get_write_statistics(self, storage):
        """Test getting write statistics"""
        stats = storage.get_write_statistics()
        
        expected_keys = [
            "total_points_written", "successful_writes", "failed_writes",
            "write_success_rate", "average_write_duration", "connection_errors",
            "batch_queue_size", "is_connected", "last_write_time"
        ]
        
        for key in expected_keys:
            assert key in stats
    
    def test_flush_pending_writes(self, storage):
        """Test manual flush of pending writes"""
        # Add some data to queue
        metrics = MockRealtimeLatencyMetrics()
        storage.write_latency_metrics("test-session", metrics)
        
        initial_queue_size = storage.write_stats.batch_queue_size
        storage.flush_pending_writes()
        
        # Queue should be smaller after flush
        assert storage.write_stats.batch_queue_size <= initial_queue_size
    
    def test_cleanup(self, storage):
        """Test cleanup method"""
        storage.cleanup()
        
        assert not storage._is_connected
    
    def test_error_handling_on_write_failure(self, storage):
        """Test error handling when write operation fails"""
        storage._write_api.write.side_effect = Exception("Write failed")
        
        metrics = MockRealtimeLatencyMetrics()
        initial_failed_writes = storage.write_stats.failed_writes
        
        # Fill queue to trigger write
        for i in range(storage.config.batch_size):
            storage.write_latency_metrics("test-session", metrics)
        
        # Should have recorded the failure
        assert storage.write_stats.failed_writes > initial_failed_writes
    
    def test_connection_retry_logic(self, config, mock_influxdb_modules):
        """Test connection retry logic"""
        with patch.object(InfluxDBMetricsStorage, '_initialize_connection') as mock_init:
            # First call fails, second succeeds
            mock_init.side_effect = [False, True]
            
            storage = InfluxDBMetricsStorage(config)
            
            # Should not be connected initially
            assert not storage._is_connected
            
            # Try to ensure connection (should succeed on retry)
            result = storage._ensure_connection()
            assert result is True


class TestRealtimeMetricsWithInfluxDB:
    """Test the integration wrapper class"""
    
    @pytest.fixture
    def mock_realtime_metrics(self):
        return Mock()
    
    @pytest.fixture
    def mock_influx_storage(self):
        storage = Mock()
        storage.write_latency_metrics = Mock()
        storage.write_connection_metrics = Mock()
        storage.write_audio_metrics = Mock()
        storage.write_cost_metrics = Mock()
        return storage
    
    @pytest.fixture
    def wrapper(self, mock_realtime_metrics, mock_influx_storage):
        return RealtimeMetricsWithInfluxDB(mock_realtime_metrics, mock_influx_storage)
    
    def test_record_voice_latency(self, wrapper, mock_realtime_metrics, mock_influx_storage):
        """Test voice latency recording with dual storage"""
        wrapper.record_voice_latency(150.0, "voice_to_voice_latency_ms")
        
        mock_realtime_metrics.record_voice_latency.assert_called_once_with(150.0, "voice_to_voice_latency_ms")
        mock_influx_storage.write_latency_metrics.assert_called_once()
    
    def test_record_connection_event(self, wrapper, mock_realtime_metrics, mock_influx_storage):
        """Test connection event recording with dual storage"""
        wrapper.record_connection_event("connect", True, "successful")
        
        mock_realtime_metrics.record_connection_event.assert_called_once_with("connect", True, "successful")
        mock_influx_storage.write_connection_metrics.assert_called_once()
    
    def test_record_audio_metrics(self, wrapper, mock_realtime_metrics, mock_influx_storage):
        """Test audio metrics recording with dual storage"""
        wrapper.record_audio_metrics(0.85, 0.6, "processing")
        
        mock_realtime_metrics.record_audio_metrics.assert_called_once_with(0.85, 0.6, "processing")
        mock_influx_storage.write_audio_metrics.assert_called_once()
    
    def test_record_token_usage(self, wrapper, mock_realtime_metrics, mock_influx_storage):
        """Test token usage recording with dual storage"""
        wrapper.record_token_usage(100, 200)
        
        mock_realtime_metrics.record_token_usage.assert_called_once_with(100, 200)
        mock_influx_storage.write_cost_metrics.assert_called_once()
    
    def test_attribute_delegation(self, wrapper, mock_realtime_metrics):
        """Test that other attributes are delegated to underlying metrics"""
        mock_realtime_metrics.some_method = Mock(return_value="test_result")
        
        result = wrapper.some_method()
        
        assert result == "test_result"
        mock_realtime_metrics.some_method.assert_called_once()


class TestFactoryFunction:
    """Test the factory function for creating storage instances"""
    
    def test_create_influxdb_storage_without_client(self):
        """Test factory function when InfluxDB client is not available"""
        config = InfluxDBConfig()
        
        with patch('assistant.influxdb_metrics_storage.HAS_INFLUXDB', False):
            result = create_influxdb_storage(config)
            assert result is None
    
    def test_create_influxdb_storage_with_client(self):
        """Test factory function when InfluxDB client is available"""
        config = InfluxDBConfig()
        
        with patch('assistant.influxdb_metrics_storage.HAS_INFLUXDB', True):
            with patch.object(InfluxDBMetricsStorage, '__init__', return_value=None) as mock_init:
                result = create_influxdb_storage(config)
                assert result is not None
                mock_init.assert_called_once()


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    @pytest.fixture
    def full_setup(self, mock_influxdb_modules):
        """Set up a complete test environment"""
        config = InfluxDBConfig(batch_size=5, flush_interval=100)
        
        with patch.object(InfluxDBMetricsStorage, '_initialize_connection', return_value=True):
            storage = InfluxDBMetricsStorage(config)
            storage._is_connected = True
            storage._write_api = Mock()
            storage._query_api = Mock()
            
            return storage
    
    def test_complete_metrics_workflow(self, full_setup):
        """Test a complete workflow with all metric types"""
        storage = full_setup
        session_id = "integration-test-session"
        
        # Write various metrics
        latency_metrics = MockRealtimeLatencyMetrics()
        connection_metrics = MockRealtimeConnectionMetrics()
        audio_metrics = MockRealtimeAudioMetrics()
        cost_metrics = MockRealtimeCostMetrics()
        
        storage.write_latency_metrics(session_id, latency_metrics)
        storage.write_connection_metrics(session_id, connection_metrics)
        storage.write_audio_metrics(session_id, audio_metrics)
        storage.write_cost_metrics(session_id, cost_metrics)
        
        # Force flush
        storage.flush_pending_writes()
        
        # Verify writes occurred
        assert storage.write_stats.total_points_written > 0
        assert storage.write_stats.successful_writes > 0
    
    def test_high_volume_metrics_writing(self, full_setup):
        """Test handling of high-volume metrics writing"""
        storage = full_setup
        session_id = "high-volume-test"
        
        metrics = MockRealtimeLatencyMetrics()
        
        # Write many metrics rapidly
        for i in range(50):
            storage.write_latency_metrics(f"{session_id}-{i}", metrics)
        
        # Should have triggered multiple flushes
        assert storage.write_stats.successful_writes > 1
        assert storage.write_stats.total_points_written > 50
    
    def test_error_recovery_scenario(self, full_setup):
        """Test error recovery and retry behavior"""
        storage = full_setup
        
        # Simulate write failures followed by success
        storage._write_api.write.side_effect = [
            Exception("Network error"),
            Exception("Timeout"),
            None  # Success
        ]
        
        metrics = MockRealtimeLatencyMetrics()
        
        # Try to write enough metrics to trigger multiple flush attempts
        for i in range(15):  # More than batch size
            storage.write_latency_metrics("error-test", metrics)
        
        # Should have recorded failures and eventual success
        assert storage.write_stats.failed_writes > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 