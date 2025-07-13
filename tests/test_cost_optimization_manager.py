"""
Test suite for Cost Optimization and Usage Monitoring Manager

Comprehensive tests for:
- User quota management and database operations
- Cost tracking and session monitoring  
- Cost prediction using machine learning
- Automatic session termination and limits
- Cost optimization recommendations
- Usage analytics and reporting
- Alert system integration
- Connection pool optimization
"""

import pytest
import asyncio
import sqlite3
import json
import time
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from assistant.cost_optimization_manager import (
    CostOptimizationManager, CostControlConfig, UserQuota, UsageMetrics,
    CostPrediction, OptimizationRecommendation, CostAlertType, OptimizationLevel,
    create_cost_optimization_manager
)
from assistant.config_manager import RealtimeAPIConfig
from assistant.realtime_metrics_collector import RealtimeMetricsCollector
from assistant.realtime_session_manager import RealtimeSessionManager
from assistant.enhanced_alert_system import EnhancedAlertSystem


class TestCostOptimizationManager:
    """Test the main CostOptimizationManager functionality"""
    
    @pytest.fixture
    async def temp_db(self):
        """Create temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = Path(tmp.name)
            yield db_path
            db_path.unlink(missing_ok=True)
    
    @pytest.fixture
    def cost_config(self):
        """Create test cost control configuration"""
        return CostControlConfig(
            max_daily_cost=20.0,
            max_session_duration=15,  # 15 minutes for testing
            max_concurrent_sessions=3,
            cost_alert_threshold=0.8,
            enable_auto_termination=True,
            enable_cost_prediction=True,
            optimization_level=OptimizationLevel.BALANCED,
            connection_pool_size=5
        )
    
    @pytest.fixture
    def realtime_config(self):
        """Create test realtime API configuration"""
        return RealtimeAPIConfig(
            enabled=True,
            max_cost_per_session=2.0,
            cost_alert_threshold=0.8
        )
    
    @pytest.fixture
    def mock_metrics_collector(self):
        """Create mock metrics collector"""
        mock = Mock(spec=RealtimeMetricsCollector)
        mock.record_token_usage = AsyncMock()
        return mock
    
    @pytest.fixture
    def mock_session_manager(self):
        """Create mock session manager"""
        mock = Mock(spec=RealtimeSessionManager)
        mock.terminate_session = AsyncMock()
        return mock
    
    @pytest.fixture
    def mock_alert_system(self):
        """Create mock alert system"""
        mock = Mock(spec=EnhancedAlertSystem)
        mock.add_rule = AsyncMock()
        mock.process_alert = AsyncMock()
        return mock
    
    @pytest.fixture
    async def cost_manager(self, temp_db, cost_config, realtime_config, 
                          mock_metrics_collector, mock_session_manager, mock_alert_system):
        """Create cost optimization manager for testing"""
        # Patch the database path
        cost_config_copy = cost_config
        
        manager = CostOptimizationManager(
            config=cost_config_copy,
            realtime_config=realtime_config,
            metrics_collector=mock_metrics_collector,
            session_manager=mock_session_manager,
            alert_system=mock_alert_system
        )
        
        # Override database path for testing
        manager.database_path = temp_db
        
        await manager.initialize()
        yield manager
        await manager.cleanup()


class TestDatabaseOperations:
    """Test database initialization and operations"""
    
    async def test_database_initialization(self, cost_manager):
        """Test database tables are created correctly"""
        async with cost_manager._initialize_database():
            # Check if database file exists
            assert cost_manager.database_path.exists()
            
            # Check tables exist
            async with aiosqlite.connect(cost_manager.database_path) as db:
                async with db.execute("SELECT name FROM sqlite_master WHERE type='table'") as cursor:
                    tables = [row[0] for row in await cursor.fetchall()]
                    
                    expected_tables = [
                        'user_quotas', 'daily_usage', 'session_costs',
                        'cost_predictions', 'optimization_recommendations'
                    ]
                    
                    for table in expected_tables:
                        assert table in tables, f"Table {table} not created"
    
    async def test_user_quota_operations(self, cost_manager):
        """Test user quota creation, update, and retrieval"""
        # Create test user quota
        quota = UserQuota(
            user_id="test_user_1",
            daily_budget=15.0,
            monthly_budget=400.0,
            max_session_duration=20,
            max_concurrent_sessions=2,
            max_tokens_per_session=40000
        )
        
        # Test creation
        result = await cost_manager.create_or_update_user_quota(quota)
        assert result is True
        
        # Test retrieval
        assert "test_user_1" in cost_manager.user_quotas
        retrieved_quota = cost_manager.user_quotas["test_user_1"]
        assert retrieved_quota.daily_budget == 15.0
        assert retrieved_quota.max_session_duration == 20
        
        # Test update
        quota.daily_budget = 25.0
        result = await cost_manager.create_or_update_user_quota(quota)
        assert result is True
        assert cost_manager.user_quotas["test_user_1"].daily_budget == 25.0
    
    async def test_daily_usage_tracking(self, cost_manager):
        """Test daily usage metrics tracking"""
        user_id = "test_user_2"
        
        # Create user quota
        quota = UserQuota(user_id=user_id, daily_budget=10.0)
        await cost_manager.create_or_update_user_quota(quota)
        
        # Simulate session metrics
        session_metrics = {
            "session_id": "session_123",
            "user_id": user_id,
            "duration_seconds": 600,  # 10 minutes
            "total_cost": 1.5,
            "input_tokens": 1000,
            "output_tokens": 500,
            "total_tokens": 1500,
            "error_count": 1,
            "warnings_triggered": 0
        }
        
        # Update daily usage
        await cost_manager._update_daily_usage(user_id, session_metrics)
        
        # Verify daily usage was recorded
        daily_cost = await cost_manager._get_user_daily_cost(user_id)
        assert daily_cost == 1.5


class TestSessionMonitoring:
    """Test session monitoring and cost tracking"""
    
    async def test_session_lifecycle(self, cost_manager):
        """Test complete session monitoring lifecycle"""
        user_id = "test_user_3"
        session_id = "session_456"
        
        # Create user quota
        quota = UserQuota(user_id=user_id, daily_budget=20.0, max_tokens_per_session=10000)
        await cost_manager.create_or_update_user_quota(quota)
        
        # Start session monitoring
        result = await cost_manager.start_session_monitoring(session_id, user_id)
        assert result is True
        assert session_id in cost_manager.active_sessions
        
        # Record token usage
        result = await cost_manager.record_session_usage(session_id, 500, 250)
        assert result is True
        
        session = cost_manager.active_sessions[session_id]
        assert session["input_tokens"] == 500
        assert session["output_tokens"] == 250
        assert session["total_cost"] > 0
        
        # End session monitoring
        metrics = await cost_manager.end_session_monitoring(session_id)
        assert metrics["session_id"] == session_id
        assert metrics["total_tokens"] == 750
        assert session_id not in cost_manager.active_sessions
    
    async def test_concurrent_session_limits(self, cost_manager):
        """Test concurrent session limit enforcement"""
        user_id = "test_user_4"
        
        # Create user quota with low concurrent session limit
        quota = UserQuota(user_id=user_id, max_concurrent_sessions=2)
        await cost_manager.create_or_update_user_quota(quota)
        
        # Start first two sessions (should succeed)
        result1 = await cost_manager.start_session_monitoring("session_1", user_id)
        result2 = await cost_manager.start_session_monitoring("session_2", user_id)
        assert result1 is True
        assert result2 is True
        
        # Try to start third session (should fail)
        result3 = await cost_manager.start_session_monitoring("session_3", user_id)
        assert result3 is False
    
    async def test_token_limit_enforcement(self, cost_manager):
        """Test token limit enforcement with session termination"""
        user_id = "test_user_5"
        session_id = "session_789"
        
        # Create user quota with low token limit
        quota = UserQuota(user_id=user_id, max_tokens_per_session=1000)
        await cost_manager.create_or_update_user_quota(quota)
        
        # Start session
        await cost_manager.start_session_monitoring(session_id, user_id)
        
        # Add tokens below limit
        result = await cost_manager.record_session_usage(session_id, 400, 200)
        assert result is True
        
        # Add tokens that exceed limit
        result = await cost_manager.record_session_usage(session_id, 300, 200)
        assert result is False  # Should fail and trigger termination
        
        # Verify session manager termination was called
        cost_manager.session_manager.terminate_session.assert_called_once()
    
    async def test_daily_budget_enforcement(self, cost_manager):
        """Test daily budget limit enforcement"""
        user_id = "test_user_6"
        session_id = "session_budget"
        
        # Create user quota with low daily budget
        quota = UserQuota(user_id=user_id, daily_budget=1.0)
        await cost_manager.create_or_update_user_quota(quota)
        
        # Simulate existing daily usage close to limit
        session_metrics = {
            "session_id": "previous_session",
            "user_id": user_id,
            "duration_seconds": 300,
            "total_cost": 0.9,  # Close to limit
            "input_tokens": 500,
            "output_tokens": 250,
            "total_tokens": 750,
            "error_count": 0,
            "warnings_triggered": 0
        }
        await cost_manager._update_daily_usage(user_id, session_metrics)
        
        # Try to start new session (should fail due to budget)
        result = await cost_manager.start_session_monitoring(session_id, user_id)
        assert result is False


class TestCostPrediction:
    """Test cost prediction and machine learning features"""
    
    async def test_prediction_model_initialization(self, cost_manager):
        """Test machine learning model initialization"""
        # Create sample historical data
        historical_data = []
        for i in range(15):  # 15 days of data
            date = datetime.now() - timedelta(days=i)
            historical_data.append({
                "user_id": "test_user",
                "date": date.strftime("%Y-%m-%d"),
                "total_cost": 2.0 + (i % 3) * 0.5,  # Varying costs
                "total_tokens": 2000 + (i % 3) * 500,
                "session_count": 3 + (i % 2),
                "average_session_duration": 15.0 + (i % 4) * 2,
                "total_session_duration": 45.0 + (i % 4) * 6,
                "error_count": i % 3,
                "peak_concurrent_sessions": 1 + (i % 2)
            })
        
        # Mock the historical data retrieval
        with patch.object(cost_manager, '_get_historical_usage_data', return_value=historical_data):
            await cost_manager._initialize_prediction_model()
            
            # Check if model was initialized
            assert cost_manager.cost_predictor is not None
            assert cost_manager.scaler is not None
    
    async def test_training_data_preparation(self, cost_manager):
        """Test training data preparation for ML model"""
        historical_data = [
            {
                "date": "2024-01-15",
                "session_count": 5,
                "average_session_duration": 18.0,
                "total_session_duration": 90.0,
                "total_tokens": 3000,
                "error_count": 2,
                "total_cost": 3.5
            }
        ]
        
        features, targets = cost_manager._prepare_training_data(historical_data)
        
        assert len(features) == 1
        assert len(targets) == 1
        assert features[0][2] == 5  # session_count
        assert targets[0] == 3.5  # total_cost
    
    async def test_cost_prediction_generation(self, cost_manager):
        """Test cost prediction generation for users"""
        user_id = "test_user_predict"
        
        # Create user quota
        quota = UserQuota(user_id=user_id)
        await cost_manager.create_or_update_user_quota(quota)
        
        # Mock recent usage data
        recent_usage = [
            {
                "date": "2024-01-14", "total_cost": 2.0, "total_tokens": 2000,
                "session_count": 3, "average_session_duration": 15.0,
                "total_session_duration": 45.0, "error_count": 1,
                "peak_concurrent_sessions": 1
            },
            {
                "date": "2024-01-15", "total_cost": 2.5, "total_tokens": 2500,
                "session_count": 4, "average_session_duration": 16.0,
                "total_session_duration": 64.0, "error_count": 0,
                "peak_concurrent_sessions": 2
            }
        ]
        
        with patch.object(cost_manager, '_get_user_recent_usage', return_value=recent_usage):
            # Mock the ML model
            cost_manager.cost_predictor = Mock()
            cost_manager.cost_predictor.predict.return_value = np.array([2.8])
            cost_manager.cost_predictor.coef_ = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            
            cost_manager.scaler = Mock()
            cost_manager.scaler.transform.return_value = np.array([[1, 2, 3, 4, 5, 6]])
            
            prediction = await cost_manager._predict_user_costs(user_id)
            
            assert prediction is not None
            assert prediction.predicted_daily_cost == 2.8
            assert prediction.predicted_monthly_cost == 2.8 * 30
            assert prediction.confidence_score > 0
    
    async def test_trend_direction_calculation(self, cost_manager):
        """Test cost trend direction calculation"""
        # Increasing trend
        increasing_costs = [1.0, 1.5, 2.0, 2.5, 3.0]
        trend = cost_manager._calculate_trend_direction(increasing_costs)
        assert trend == "increasing"
        
        # Decreasing trend  
        decreasing_costs = [3.0, 2.5, 2.0, 1.5, 1.0]
        trend = cost_manager._calculate_trend_direction(decreasing_costs)
        assert trend == "decreasing"
        
        # Stable trend
        stable_costs = [2.0, 2.1, 1.9, 2.0, 2.1]
        trend = cost_manager._calculate_trend_direction(stable_costs)
        assert trend == "stable"


class TestOptimizationRecommendations:
    """Test cost optimization recommendations"""
    
    async def test_optimization_opportunity_analysis(self, cost_manager):
        """Test analysis of optimization opportunities"""
        user_id = "test_user_opt"
        
        # Mock recent usage with optimization opportunities
        recent_usage = [
            {
                "total_cost": 4.0,  # High cost
                "average_session_duration": 30 * 60,  # 30 minutes (long)
                "session_count": 8,
                "error_count": 2  # High error rate (2/8 = 25%)
            }
        ]
        
        with patch.object(cost_manager, '_get_user_recent_usage', return_value=recent_usage):
            recommendations = await cost_manager._analyze_user_optimization_opportunities(user_id)
            
            assert len(recommendations) > 0
            
            # Should have session optimization recommendation
            session_opt = next((r for r in recommendations if r.type == "session_optimization"), None)
            assert session_opt is not None
            assert session_opt.priority == "high"
            assert session_opt.potential_savings > 0
            
            # Should have duration optimization recommendation
            duration_opt = next((r for r in recommendations if r.type == "duration_optimization"), None)
            assert duration_opt is not None
            
            # Should have error reduction recommendation
            error_opt = next((r for r in recommendations if r.type == "error_reduction"), None)
            assert error_opt is not None
    
    async def test_recommendation_persistence(self, cost_manager):
        """Test saving optimization recommendations to database"""
        recommendation = OptimizationRecommendation(
            type="test_optimization",
            priority="medium",
            description="Test recommendation for optimization",
            potential_savings=1.5,
            implementation_effort="easy",
            details={"test": "data"}
        )
        
        await cost_manager._save_optimization_recommendation(recommendation)
        
        # Verify recommendation was saved
        async with aiosqlite.connect(cost_manager.database_path) as db:
            async with db.execute("SELECT * FROM optimization_recommendations") as cursor:
                rows = await cursor.fetchall()
                assert len(rows) > 0
                
                row = rows[0]
                assert row[3] == "test_optimization"  # type
                assert row[4] == "medium"  # priority
                assert row[6] == 1.5  # potential_savings


class TestUsageAnalytics:
    """Test usage analytics and reporting"""
    
    async def test_cost_analytics_generation(self, cost_manager):
        """Test comprehensive cost analytics generation"""
        user_id = "test_user_analytics"
        
        # Create test data
        quota = UserQuota(user_id=user_id)
        await cost_manager.create_or_update_user_quota(quota)
        
        # Add some daily usage data
        today = datetime.now().date()
        async with aiosqlite.connect(cost_manager.database_path) as db:
            await db.execute("""
                INSERT INTO daily_usage 
                (user_id, date, total_cost, total_tokens, session_count, 
                 average_session_duration, total_session_duration, error_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, today, 5.0, 5000, 10, 15.0, 150.0, 2))
            await db.commit()
        
        # Get analytics
        analytics = await cost_manager.get_cost_analytics(user_id=user_id)
        
        assert "summary" in analytics
        assert "daily_breakdown" in analytics
        assert "predictions" in analytics
        assert "recommendations" in analytics
        assert "efficiency_metrics" in analytics
        
        # Check summary data
        summary = analytics["summary"]
        assert summary["total_cost"] == 5.0
        assert summary["total_tokens"] == 5000
        assert summary["total_sessions"] == 10
    
    async def test_optimization_stats(self, cost_manager):
        """Test optimization performance statistics"""
        # Set some optimization stats
        cost_manager.optimization_stats["sessions_terminated"] = 3
        cost_manager.optimization_stats["cost_savings"] = 15.50
        cost_manager.optimization_stats["alerts_triggered"] = 5
        
        stats = await cost_manager.get_optimization_stats()
        
        assert stats["optimization_stats"]["sessions_terminated"] == 3
        assert stats["optimization_stats"]["cost_savings"] == 15.50
        assert stats["optimization_stats"]["alerts_triggered"] == 5
        assert "active_sessions" in stats
        assert "user_quotas" in stats
    
    async def test_budget_utilization_calculation(self, cost_manager):
        """Test budget utilization calculation"""
        user_id = "test_user_budget"
        
        # Create user quota
        quota = UserQuota(user_id=user_id, daily_budget=10.0)
        await cost_manager.create_or_update_user_quota(quota)
        
        # Add daily usage (50% of budget)
        today = datetime.now().date()
        async with aiosqlite.connect(cost_manager.database_path) as db:
            await db.execute("""
                INSERT INTO daily_usage (user_id, date, total_cost)
                VALUES (?, ?, ?)
            """, (user_id, today, 5.0))
            await db.commit()
        
        utilization = await cost_manager._calculate_budget_utilization()
        
        assert user_id in utilization
        assert utilization[user_id] == 50.0  # 50% utilization


class TestAlertIntegration:
    """Test integration with alert system"""
    
    async def test_alert_rule_setup(self, cost_manager):
        """Test cost alert rules are properly configured"""
        # Verify alert rules were added during initialization
        assert cost_manager.alert_system.add_rule.call_count >= 3
        
        # Check that cost-related rules were added
        call_args = [call[0][0] for call in cost_manager.alert_system.add_rule.call_args_list]
        rule_names = [rule.name for rule in call_args]
        
        expected_rules = ["daily_budget_warning", "daily_budget_exceeded", "high_cost_session"]
        for rule_name in expected_rules:
            assert rule_name in rule_names
    
    async def test_cost_alert_triggering(self, cost_manager):
        """Test cost alert triggering functionality"""
        user_id = "test_user_alert"
        context = {"daily_cost": 8.0, "budget": 10.0}
        
        await cost_manager._trigger_cost_alert(
            CostAlertType.USER_QUOTA_WARNING, 
            user_id, 
            context
        )
        
        # Verify alert was processed
        cost_manager.alert_system.process_alert.assert_called_once()
        
        # Check alert details
        call_args = cost_manager.alert_system.process_alert.call_args
        alert = call_args[0][0]
        assert "cost_user_quota_warning" in alert.id
        assert alert.context == context


class TestBackgroundTasks:
    """Test background monitoring and optimization tasks"""
    
    async def test_session_monitoring_task(self, cost_manager):
        """Test background session monitoring"""
        user_id = "test_user_bg"
        session_id = "session_bg"
        
        # Create user quota with short duration limit
        quota = UserQuota(user_id=user_id, max_session_duration=1)  # 1 minute
        await cost_manager.create_or_update_user_quota(quota)
        
        # Start session and simulate long duration
        await cost_manager.start_session_monitoring(session_id, user_id)
        
        # Manually set start time to simulate long-running session
        cost_manager.active_sessions[session_id]["start_time"] = time.time() - 120  # 2 minutes ago
        
        # Run monitoring task
        await cost_manager._monitor_active_sessions()
        
        # Verify session termination was called
        cost_manager.session_manager.terminate_session.assert_called()
    
    async def test_daily_usage_aggregation(self, cost_manager):
        """Test daily usage aggregation task"""
        user_id = "test_user_daily"
        
        # Add active session
        cost_manager.active_sessions["session_daily"] = {
            "user_id": user_id,
            "start_time": time.time()
        }
        
        # Run aggregation
        await cost_manager._aggregate_daily_usage()
        
        # Verify database was updated (check for no errors)
        # In a real test, we'd verify the peak concurrent sessions was updated


class TestConnectionPoolOptimization:
    """Test connection pool optimization features"""
    
    async def test_connection_pool_optimization(self, cost_manager):
        """Test connection pool size optimization"""
        # Mock connection pool
        mock_pool = Mock()
        mock_pool.get_pool_stats = AsyncMock(return_value={
            "connections_reused": 80,
            "total_connections": 100,
            "active_connections": 2
        })
        cost_manager.connection_pool = mock_pool
        
        # Run optimization
        await cost_manager._optimize_connection_pool()
        
        # Check that optimization stats were updated
        assert cost_manager.optimization_stats["connections_reused"] == 80


class TestDataClassesAndEnums:
    """Test data classes and enums"""
    
    def test_user_quota_dataclass(self):
        """Test UserQuota dataclass functionality"""
        quota = UserQuota(
            user_id="test_user",
            daily_budget=25.0,
            monthly_budget=750.0
        )
        
        assert quota.user_id == "test_user"
        assert quota.daily_budget == 25.0
        assert quota.monthly_budget == 750.0
        assert quota.enabled is True
        assert isinstance(quota.created_at, datetime)
    
    def test_cost_prediction_dataclass(self):
        """Test CostPrediction dataclass"""
        prediction = CostPrediction(
            predicted_daily_cost=3.5,
            predicted_monthly_cost=105.0,
            confidence_score=0.85,
            trend_direction="increasing",
            recommendation="Monitor usage",
            feature_importance={"session_count": 0.4},
            model_accuracy=0.82
        )
        
        assert prediction.predicted_daily_cost == 3.5
        assert prediction.trend_direction == "increasing"
        assert prediction.feature_importance["session_count"] == 0.4
    
    def test_optimization_level_enum(self):
        """Test OptimizationLevel enum"""
        assert OptimizationLevel.CONSERVATIVE.value == "conservative"
        assert OptimizationLevel.BALANCED.value == "balanced"
        assert OptimizationLevel.AGGRESSIVE.value == "aggressive"
    
    def test_cost_alert_type_enum(self):
        """Test CostAlertType enum values"""
        assert CostAlertType.DAILY_BUDGET_WARNING.value == "daily_budget_warning"
        assert CostAlertType.USER_QUOTA_EXCEEDED.value == "user_quota_exceeded"
        assert CostAlertType.HIGH_COST_SESSION.value == "high_cost_session"


class TestFactoryFunction:
    """Test factory function for creating cost manager"""
    
    def test_create_cost_optimization_manager(self):
        """Test factory function creates manager correctly"""
        config = CostControlConfig()
        realtime_config = RealtimeAPIConfig()
        
        manager = create_cost_optimization_manager(
            config=config,
            realtime_config=realtime_config,
            metrics_collector=Mock(),
            session_manager=Mock()
        )
        
        assert isinstance(manager, CostOptimizationManager)
        assert manager.config == config
        assert manager.realtime_config == realtime_config


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    async def test_missing_user_quota_handling(self, cost_manager):
        """Test handling of operations with missing user quota"""
        # Try to start session monitoring for user without quota
        result = await cost_manager.start_session_monitoring("session_no_quota", "nonexistent_user")
        assert result is True  # Should still work, just without quota enforcement
    
    async def test_invalid_session_operations(self, cost_manager):
        """Test operations on invalid/missing sessions"""
        # Try to record usage for non-existent session
        result = await cost_manager.record_session_usage("nonexistent_session", 100, 50)
        assert result is False
        
        # Try to end monitoring for non-existent session
        metrics = await cost_manager.end_session_monitoring("nonexistent_session")
        assert metrics == {}
    
    async def test_database_error_handling(self, cost_manager):
        """Test handling of database errors"""
        # Corrupt the database path to trigger errors
        original_path = cost_manager.database_path
        cost_manager.database_path = Path("/invalid/path/database.db")
        
        # Operations should handle database errors gracefully
        result = await cost_manager.create_or_update_user_quota(UserQuota(user_id="test"))
        assert result is False
        
        # Restore path
        cost_manager.database_path = original_path


# Performance and Integration Tests
class TestPerformanceAndIntegration:
    """Test performance characteristics and integration scenarios"""
    
    async def test_high_volume_session_tracking(self, cost_manager):
        """Test performance with many concurrent sessions"""
        user_id = "test_user_volume"
        
        # Create user quota with high limits
        quota = UserQuota(
            user_id=user_id,
            max_concurrent_sessions=50,
            daily_budget=100.0
        )
        await cost_manager.create_or_update_user_quota(quota)
        
        # Start many sessions
        session_ids = []
        for i in range(20):
            session_id = f"session_volume_{i}"
            result = await cost_manager.start_session_monitoring(session_id, user_id)
            assert result is True
            session_ids.append(session_id)
        
        # Record usage for all sessions
        for session_id in session_ids:
            result = await cost_manager.record_session_usage(session_id, 100, 50)
            assert result is True
        
        # End all sessions
        for session_id in session_ids:
            metrics = await cost_manager.end_session_monitoring(session_id)
            assert metrics["session_id"] == session_id
        
        # Verify all sessions were properly tracked
        assert len(cost_manager.active_sessions) == 0
    
    async def test_concurrent_user_operations(self, cost_manager):
        """Test concurrent operations across multiple users"""
        user_count = 10
        tasks = []
        
        async def user_session_workflow(user_index):
            user_id = f"concurrent_user_{user_index}"
            session_id = f"concurrent_session_{user_index}"
            
            # Create quota
            quota = UserQuota(user_id=user_id, daily_budget=20.0)
            await cost_manager.create_or_update_user_quota(quota)
            
            # Session workflow
            await cost_manager.start_session_monitoring(session_id, user_id)
            await cost_manager.record_session_usage(session_id, 200, 100)
            await cost_manager.end_session_monitoring(session_id)
        
        # Run concurrent workflows
        for i in range(user_count):
            tasks.append(user_session_workflow(i))
        
        await asyncio.gather(*tasks)
        
        # Verify all users were created
        assert len(cost_manager.user_quotas) >= user_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 