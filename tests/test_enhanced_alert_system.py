"""
Test Suite for Enhanced Alert System

Comprehensive tests for the enhanced alert system including alert rules,
escalation policies, notification channels, message templating, and integration scenarios.
"""

import pytest
import asyncio
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
from collections import deque

# Import the module under test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.enhanced_alert_system import (
    EnhancedAlertSystem,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    NotificationChannel,
    NotificationConfig,
    EscalationPolicy,
    Alert,
    AlertSystemStats,
    MessageTemplateEngine,
    NotificationDispatcher,
    create_enhanced_alert_system,
    _add_default_alert_rules
)


class TestAlertRule:
    """Test AlertRule class functionality"""
    
    def test_alert_rule_creation(self):
        """Test alert rule creation and basic properties"""
        rule = AlertRule(
            name="test_rule",
            description="Test rule description",
            metric_conditions={"latency.p95": {"operator": "gt", "threshold": 500}},
            severity=AlertSeverity.WARNING,
            tags=["test", "performance"]
        )
        
        assert rule.name == "test_rule"
        assert rule.description == "Test rule description"
        assert rule.severity == AlertSeverity.WARNING
        assert rule.enabled == True
        assert rule.suppression_window == 300
        assert "test" in rule.tags
        assert "performance" in rule.tags
    
    def test_metric_condition_evaluation_gt(self):
        """Test greater than condition evaluation"""
        rule = AlertRule(
            name="latency_rule",
            description="Latency threshold",
            metric_conditions={"latency.p95": {"operator": "gt", "threshold": 500}},
            severity=AlertSeverity.WARNING
        )
        
        # Should trigger
        metrics = {"latency": {"p95": 600}}
        assert rule.evaluate(metrics) == True
        
        # Should not trigger
        metrics = {"latency": {"p95": 400}}
        assert rule.evaluate(metrics) == False
    
    def test_metric_condition_evaluation_lt(self):
        """Test less than condition evaluation"""
        rule = AlertRule(
            name="quality_rule",
            description="Quality threshold",
            metric_conditions={"audio.quality": {"operator": "lt", "threshold": 0.8}},
            severity=AlertSeverity.WARNING
        )
        
        # Should trigger
        metrics = {"audio": {"quality": 0.6}}
        assert rule.evaluate(metrics) == True
        
        # Should not trigger
        metrics = {"audio": {"quality": 0.9}}
        assert rule.evaluate(metrics) == False
    
    def test_multiple_conditions(self):
        """Test multiple condition evaluation"""
        rule = AlertRule(
            name="multi_rule",
            description="Multiple conditions",
            metric_conditions={
                "latency.p95": {"operator": "gt", "threshold": 500},
                "connection.success_rate": {"operator": "lt", "threshold": 95}
            },
            severity=AlertSeverity.CRITICAL
        )
        
        # Both conditions met
        metrics = {
            "latency": {"p95": 600},
            "connection": {"success_rate": 90}
        }
        assert rule.evaluate(metrics) == True
        
        # Only one condition met
        metrics = {
            "latency": {"p95": 600},
            "connection": {"success_rate": 98}
        }
        assert rule.evaluate(metrics) == False
    
    def test_nested_metric_access(self):
        """Test accessing nested metrics"""
        rule = AlertRule(
            name="nested_rule",
            description="Nested metric access",
            metric_conditions={"deep.nested.value": {"operator": "eq", "threshold": 42}},
            severity=AlertSeverity.INFO
        )
        
        metrics = {"deep": {"nested": {"value": 42}}}
        assert rule.evaluate(metrics) == True
        
        # Missing nested key
        metrics = {"deep": {"other": {"value": 42}}}
        assert rule.evaluate(metrics) == False
    
    def test_condition_operators(self):
        """Test all condition operators"""
        test_cases = [
            ("gt", 100, 150, True),
            ("gt", 100, 50, False),
            ("gte", 100, 100, True),
            ("gte", 100, 50, False),
            ("lt", 100, 50, True),
            ("lt", 100, 150, False),
            ("lte", 100, 100, True),
            ("lte", 100, 150, False),
            ("eq", 100, 100, True),
            ("eq", 100, 50, False),
            ("ne", 100, 50, True),
            ("ne", 100, 100, False),
            ("contains", "error", "error message", True),
            ("contains", "error", "success message", False)
        ]
        
        for operator, threshold, value, expected in test_cases:
            rule = AlertRule(
                name=f"test_{operator}",
                description=f"Test {operator}",
                metric_conditions={"test.value": {"operator": operator, "threshold": threshold}},
                severity=AlertSeverity.INFO
            )
            
            metrics = {"test": {"value": value}}
            assert rule.evaluate(metrics) == expected, f"Failed for {operator}: {threshold} vs {value}"


class TestEscalationPolicy:
    """Test EscalationPolicy class"""
    
    def test_escalation_policy_creation(self):
        """Test escalation policy creation"""
        policy = EscalationPolicy(
            name="test_policy",
            steps=[
                {"channels": ["slack"], "slack_channel": "#alerts"},
                {"channels": ["email"], "emails": ["ops@example.com"]}
            ],
            escalation_delay=600,
            max_escalations=2
        )
        
        assert policy.name == "test_policy"
        assert len(policy.steps) == 2
        assert policy.escalation_delay == 600
        assert policy.max_escalations == 2
    
    def test_get_escalation_step(self):
        """Test getting escalation step by level"""
        policy = EscalationPolicy(
            name="test_policy",
            steps=[
                {"channels": ["slack"]},
                {"channels": ["email"]},
                {"channels": ["webhook"]}
            ]
        )
        
        step0 = policy.get_escalation_step(0)
        assert step0["channels"] == ["slack"]
        
        step1 = policy.get_escalation_step(1)
        assert step1["channels"] == ["email"]
        
        step_invalid = policy.get_escalation_step(5)
        assert step_invalid is None


class TestAlert:
    """Test Alert class functionality"""
    
    def test_alert_creation(self):
        """Test alert creation"""
        alert = Alert(
            id="test_123",
            rule_name="test_rule",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.ACTIVE,
            title="Test Alert",
            message="Test message",
            context={"metric": "value"},
            created_at=time.time(),
            updated_at=time.time(),
            tags=["test"]
        )
        
        assert alert.id == "test_123"
        assert alert.rule_name == "test_rule"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.status == AlertStatus.ACTIVE
        assert alert.escalation_level == 0
        assert alert.occurrence_count == 1
    
    def test_alert_acknowledge(self):
        """Test alert acknowledgment"""
        alert = Alert(
            id="test_123",
            rule_name="test_rule",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.ACTIVE,
            title="Test Alert",
            message="Test message",
            context={},
            created_at=time.time(),
            updated_at=time.time()
        )
        
        initial_updated_at = alert.updated_at
        time.sleep(0.01)
        
        alert.acknowledge("test_user")
        
        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledged_at is not None
        assert alert.updated_at > initial_updated_at
    
    def test_alert_resolve(self):
        """Test alert resolution"""
        alert = Alert(
            id="test_123",
            rule_name="test_rule",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.ACTIVE,
            title="Test Alert",
            message="Test message",
            context={},
            created_at=time.time(),
            updated_at=time.time()
        )
        
        alert.resolve("test_user")
        
        assert alert.status == AlertStatus.RESOLVED
        assert alert.resolved_at is not None
    
    def test_alert_escalate(self):
        """Test alert escalation"""
        alert = Alert(
            id="test_123",
            rule_name="test_rule",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.ACTIVE,
            title="Test Alert",
            message="Test message",
            context={},
            created_at=time.time(),
            updated_at=time.time()
        )
        
        initial_level = alert.escalation_level
        alert.escalate()
        
        assert alert.escalation_level == initial_level + 1
        assert alert.last_escalation is not None


class TestMessageTemplateEngine:
    """Test MessageTemplateEngine class"""
    
    def test_template_engine_creation(self):
        """Test template engine creation"""
        engine = MessageTemplateEngine()
        assert "default" in engine.templates
        assert "slack" in engine.templates
        assert "email_html" in engine.templates
    
    def test_simple_template_rendering(self):
        """Test template rendering without Jinja2"""
        engine = MessageTemplateEngine()
        
        alert = Alert(
            id="test_123",
            rule_name="test_rule",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.ACTIVE,
            title="Test Alert",
            message="Test alert message",
            context={"latency": 600, "threshold": 500},
            created_at=time.time(),
            updated_at=time.time()
        )
        
        # Test with simple template
        simple_template = "Alert: {{ alert.title }} - {{ alert.message }} ({{ alert.id }})"
        
        with patch('assistant.enhanced_alert_system.HAS_JINJA2', False):
            result = engine.render_template("custom", alert, simple_template)
            assert "Test Alert" in result
            assert "Test alert message" in result
            assert "test_123" in result
    
    def test_fallback_rendering(self):
        """Test fallback rendering when template fails"""
        engine = MessageTemplateEngine()
        
        alert = Alert(
            id="test_123",
            rule_name="test_rule",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.ACTIVE,
            title="Test Alert",
            message="Test message",
            context={},
            created_at=time.time(),
            updated_at=time.time()
        )
        
        # Test with invalid template
        invalid_template = "{{ invalid.syntax"
        
        result = engine.render_template("custom", alert, invalid_template)
        assert "Test Alert" in result
        assert "Test message" in result
    
    def test_timestamp_formatting(self):
        """Test timestamp formatting"""
        engine = MessageTemplateEngine()
        timestamp = 1640995200.0  # 2022-01-01 00:00:00
        formatted = engine._format_timestamp(timestamp)
        assert "2022-01-01" in formatted


class TestNotificationDispatcher:
    """Test NotificationDispatcher class"""
    
    def test_dispatcher_creation(self):
        """Test notification dispatcher creation"""
        dispatcher = NotificationDispatcher()
        assert len(dispatcher.channels) == 0
        assert dispatcher.template_engine is not None
        assert len(dispatcher.notification_history) == 0
    
    def test_configure_slack_channel(self):
        """Test Slack channel configuration"""
        dispatcher = NotificationDispatcher()
        
        config = NotificationConfig(
            channel=NotificationChannel.SLACK,
            config={"token": "xoxb-test-token", "channel": "#alerts"},
            enabled=True
        )
        
        dispatcher.configure_channel(NotificationChannel.SLACK, config)
        
        assert NotificationChannel.SLACK in dispatcher.channels
        assert dispatcher.channels[NotificationChannel.SLACK].enabled == True
    
    def test_configure_email_channel(self):
        """Test email channel configuration"""
        dispatcher = NotificationDispatcher()
        
        config = NotificationConfig(
            channel=NotificationChannel.EMAIL,
            config={
                "api_key": "test-key",
                "from_email": "alerts@test.com",
                "to_emails": ["ops@test.com"]
            },
            enabled=True
        )
        
        dispatcher.configure_channel(NotificationChannel.EMAIL, config)
        
        assert NotificationChannel.EMAIL in dispatcher.channels
    
    def test_rate_limiting(self):
        """Test notification rate limiting"""
        config = NotificationConfig(
            channel=NotificationChannel.LOG,
            rate_limit=60
        )
        
        # First notification should not be rate limited
        assert not config.is_rate_limited(None)
        
        # Recent notification should be rate limited
        recent_time = time.time() - 30
        assert config.is_rate_limited(recent_time)
        
        # Old notification should not be rate limited
        old_time = time.time() - 120
        assert not config.is_rate_limited(old_time)
    
    @pytest.mark.asyncio
    async def test_log_notification(self):
        """Test log notification"""
        dispatcher = NotificationDispatcher()
        
        config = NotificationConfig(
            channel=NotificationChannel.LOG,
            config={"log_level": "warning"},
            enabled=True
        )
        
        dispatcher.configure_channel(NotificationChannel.LOG, config)
        
        alert = Alert(
            id="test_123",
            rule_name="test_rule",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.ACTIVE,
            title="Test Alert",
            message="Test message",
            context={},
            created_at=time.time(),
            updated_at=time.time()
        )
        
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            success = await dispatcher.send_notification(alert, NotificationChannel.LOG)
            
            assert success == True
            mock_logger.warning.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_disabled_channel(self):
        """Test sending to disabled channel"""
        dispatcher = NotificationDispatcher()
        
        config = NotificationConfig(
            channel=NotificationChannel.LOG,
            enabled=False
        )
        
        dispatcher.configure_channel(NotificationChannel.LOG, config)
        
        alert = Alert(
            id="test_123",
            rule_name="test_rule",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.ACTIVE,
            title="Test Alert",
            message="Test message",
            context={},
            created_at=time.time(),
            updated_at=time.time()
        )
        
        success = await dispatcher.send_notification(alert, NotificationChannel.LOG)
        assert success == False


class TestEnhancedAlertSystem:
    """Test EnhancedAlertSystem main class"""
    
    def test_alert_system_creation(self):
        """Test alert system creation"""
        system = EnhancedAlertSystem()
        
        assert len(system.alert_rules) == 0
        assert len(system.escalation_policies) == 0
        assert len(system.active_alerts) == 0
        assert system.notification_dispatcher is not None
        assert system.stats is not None
    
    def test_add_alert_rule(self):
        """Test adding alert rule"""
        system = EnhancedAlertSystem()
        
        rule = AlertRule(
            name="test_rule",
            description="Test rule",
            metric_conditions={"test.metric": {"operator": "gt", "threshold": 100}},
            severity=AlertSeverity.WARNING
        )
        
        system.add_alert_rule(rule)
        
        assert "test_rule" in system.alert_rules
        assert system.alert_rules["test_rule"] == rule
    
    def test_add_escalation_policy(self):
        """Test adding escalation policy"""
        system = EnhancedAlertSystem()
        
        policy = EscalationPolicy(
            name="test_policy",
            steps=[{"channels": ["log"]}]
        )
        
        system.add_escalation_policy(policy)
        
        assert "test_policy" in system.escalation_policies
        assert system.escalation_policies["test_policy"] == policy
    
    def test_configure_notification_channel(self):
        """Test configuring notification channel"""
        system = EnhancedAlertSystem()
        
        config = NotificationConfig(
            channel=NotificationChannel.LOG,
            enabled=True
        )
        
        system.configure_notification_channel(NotificationChannel.LOG, config)
        
        assert NotificationChannel.LOG in system.notification_dispatcher.channels
    
    @pytest.mark.asyncio
    async def test_evaluate_metrics_no_rules(self):
        """Test metric evaluation with no rules"""
        system = EnhancedAlertSystem()
        
        metrics = {"test": {"value": 100}}
        alerts = await system.evaluate_metrics(metrics)
        
        assert len(alerts) == 0
    
    @pytest.mark.asyncio
    async def test_evaluate_metrics_with_triggered_rule(self):
        """Test metric evaluation with triggered rule"""
        system = EnhancedAlertSystem()
        
        # Configure log notification
        log_config = NotificationConfig(
            channel=NotificationChannel.LOG,
            enabled=True
        )
        system.configure_notification_channel(NotificationChannel.LOG, log_config)
        
        rule = AlertRule(
            name="test_rule",
            description="Test rule",
            metric_conditions={"test.value": {"operator": "gt", "threshold": 50}},
            severity=AlertSeverity.WARNING
        )
        system.add_alert_rule(rule)
        
        with patch.object(system.notification_dispatcher, 'send_notification', return_value=True):
            metrics = {"test": {"value": 100}}
            alerts = await system.evaluate_metrics(metrics)
            
            assert len(alerts) == 1
            assert alerts[0].rule_name == "test_rule"
            assert alerts[0].severity == AlertSeverity.WARNING
            assert len(system.active_alerts) == 1
    
    @pytest.mark.asyncio
    async def test_evaluate_metrics_duplicate_alert(self):
        """Test metric evaluation with duplicate alert"""
        system = EnhancedAlertSystem()
        
        # Configure log notification
        log_config = NotificationConfig(
            channel=NotificationChannel.LOG,
            enabled=True
        )
        system.configure_notification_channel(NotificationChannel.LOG, log_config)
        
        rule = AlertRule(
            name="test_rule",
            description="Test rule",
            metric_conditions={"test.value": {"operator": "gt", "threshold": 50}},
            severity=AlertSeverity.WARNING
        )
        system.add_alert_rule(rule)
        
        with patch.object(system.notification_dispatcher, 'send_notification', return_value=True):
            metrics = {"test": {"value": 100}}
            
            # First evaluation - should create new alert
            alerts1 = await system.evaluate_metrics(metrics)
            assert len(alerts1) == 1
            assert len(system.active_alerts) == 1
            assert alerts1[0].occurrence_count == 1
            
            # Second evaluation - should update existing alert
            alerts2 = await system.evaluate_metrics(metrics)
            assert len(alerts2) == 1
            assert len(system.active_alerts) == 1  # Still only one active alert
            assert alerts2[0].occurrence_count == 2  # Occurrence count increased
    
    @pytest.mark.asyncio
    async def test_acknowledge_alert(self):
        """Test alert acknowledgment"""
        system = EnhancedAlertSystem()
        
        alert = Alert(
            id="test_123",
            rule_name="test_rule",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.ACTIVE,
            title="Test Alert",
            message="Test message",
            context={},
            created_at=time.time(),
            updated_at=time.time()
        )
        
        system.active_alerts[alert.id] = alert
        
        success = await system.acknowledge_alert(alert.id, "test_user")
        
        assert success == True
        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledged_at is not None
    
    @pytest.mark.asyncio
    async def test_resolve_alert(self):
        """Test alert resolution"""
        system = EnhancedAlertSystem()
        
        alert = Alert(
            id="test_123",
            rule_name="test_rule",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.ACTIVE,
            title="Test Alert",
            message="Test message",
            context={},
            created_at=time.time(),
            updated_at=time.time()
        )
        
        system.active_alerts[alert.id] = alert
        system.stats.active_alerts_count = 1
        
        success = await system.resolve_alert(alert.id, "test_user")
        
        assert success == True
        assert alert.status == AlertStatus.RESOLVED
        assert alert.resolved_at is not None
        assert alert.id not in system.active_alerts
        assert system.stats.active_alerts_count == 0
    
    def test_suppress_alert(self):
        """Test alert suppression"""
        system = EnhancedAlertSystem()
        
        alert_id = "test_123"
        system.suppress_alert(alert_id, duration=1)
        
        assert alert_id in system.suppressed_alerts
        assert system.stats.suppressed_alerts_count == 1
    
    def test_get_active_alerts(self):
        """Test getting active alerts"""
        system = EnhancedAlertSystem()
        
        alert1 = Alert(
            id="test_1",
            rule_name="rule1",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.ACTIVE,
            title="Alert 1",
            message="Message 1",
            context={},
            created_at=time.time(),
            updated_at=time.time()
        )
        
        alert2 = Alert(
            id="test_2",
            rule_name="rule2",
            severity=AlertSeverity.CRITICAL,
            status=AlertStatus.ACTIVE,
            title="Alert 2",
            message="Message 2",
            context={},
            created_at=time.time() + 1,
            updated_at=time.time() + 1
        )
        
        system.active_alerts[alert1.id] = alert1
        system.active_alerts[alert2.id] = alert2
        
        # Get all alerts
        all_alerts = system.get_active_alerts()
        assert len(all_alerts) == 2
        assert all_alerts[0].id == "test_2"  # Newer alert first
        
        # Get alerts by severity
        critical_alerts = system.get_active_alerts(AlertSeverity.CRITICAL)
        assert len(critical_alerts) == 1
        assert critical_alerts[0].id == "test_2"
    
    def test_get_alert_statistics(self):
        """Test getting alert statistics"""
        system = EnhancedAlertSystem()
        
        # Add some test data
        system.stats.total_alerts_generated = 10
        system.stats.alerts_by_severity["WARNING"] = 5
        system.stats.alerts_by_severity["CRITICAL"] = 3
        
        # Add active alert
        alert = Alert(
            id="test_1",
            rule_name="rule1",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.ACTIVE,
            title="Alert 1",
            message="Message 1",
            context={},
            created_at=time.time(),
            updated_at=time.time()
        )
        system.active_alerts[alert.id] = alert
        
        stats = system.get_alert_statistics()
        
        assert stats.total_alerts_generated == 10
        assert stats.alerts_by_severity["WARNING"] == 5
        assert stats.alerts_by_severity["CRITICAL"] == 3
        assert stats.active_alerts_count == 1
    
    def test_export_import_configuration(self):
        """Test configuration export and import"""
        system = EnhancedAlertSystem()
        
        # Add test configuration
        rule = AlertRule(
            name="test_rule",
            description="Test rule",
            metric_conditions={"test.value": {"operator": "gt", "threshold": 100}},
            severity=AlertSeverity.WARNING
        )
        system.add_alert_rule(rule)
        
        policy = EscalationPolicy(
            name="test_policy",
            steps=[{"channels": ["log"]}]
        )
        system.add_escalation_policy(policy)
        
        # Export configuration
        config = system.export_configuration()
        
        assert "alert_rules" in config
        assert "escalation_policies" in config
        assert "test_rule" in config["alert_rules"]
        assert "test_policy" in config["escalation_policies"]
        
        # Import into new system
        new_system = EnhancedAlertSystem()
        new_system.import_configuration(config)
        
        assert "test_rule" in new_system.alert_rules
        assert "test_policy" in new_system.escalation_policies
    
    @pytest.mark.asyncio
    async def test_escalation_detection(self):
        """Test escalation detection logic"""
        system = EnhancedAlertSystem()
        
        # Add escalation policy
        policy = EscalationPolicy(
            name="default",
            escalation_delay=1,  # 1 second for testing
            max_escalations=2
        )
        system.add_escalation_policy(policy)
        
        alert = Alert(
            id="test_123",
            rule_name="test_rule",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.ACTIVE,
            title="Test Alert",
            message="Test message",
            context={},
            created_at=time.time() - 2,  # 2 seconds ago
            updated_at=time.time() - 2
        )
        
        system.active_alerts[alert.id] = alert
        
        # Should need escalation
        should_escalate = system._should_escalate(alert)
        assert should_escalate == True
        
        # After escalation, should not need immediate re-escalation
        alert.escalate()
        should_escalate_again = system._should_escalate(alert)
        assert should_escalate_again == False


class TestFactoryFunctions:
    """Test factory functions and utilities"""
    
    def test_create_enhanced_alert_system(self):
        """Test factory function for creating alert system"""
        system = create_enhanced_alert_system()
        
        assert system is not None
        assert len(system.escalation_policies) >= 1  # Should have default policy
        assert len(system.alert_rules) >= 1  # Should have default rules
    
    def test_add_default_alert_rules(self):
        """Test adding default alert rules"""
        system = EnhancedAlertSystem()
        
        initial_count = len(system.alert_rules)
        _add_default_alert_rules(system)
        
        assert len(system.alert_rules) > initial_count
        
        # Check for specific default rules
        rule_names = list(system.alert_rules.keys())
        assert "high_voice_latency" in rule_names
        assert "low_connection_success_rate" in rule_names
        assert "high_hourly_cost" in rule_names
        assert "poor_audio_quality" in rule_names


class TestIntegrationScenarios:
    """Integration tests for complete alert scenarios"""
    
    @pytest.mark.asyncio
    async def test_complete_alert_lifecycle(self):
        """Test complete alert lifecycle from trigger to resolution"""
        system = create_enhanced_alert_system()
        
        # Configure log notification
        log_config = NotificationConfig(
            channel=NotificationChannel.LOG,
            enabled=True
        )
        system.configure_notification_channel(NotificationChannel.LOG, log_config)
        
        with patch.object(system.notification_dispatcher, 'send_notification', return_value=True) as mock_send:
            # Trigger alert
            metrics = {"latency": {"voice_to_voice_p95": 600}}  # Above 500ms threshold
            alerts = await system.evaluate_metrics(metrics)
            
            assert len(alerts) == 1
            alert = alerts[0]
            assert alert.status == AlertStatus.ACTIVE
            
            # Verify notification was sent
            mock_send.assert_called_once()
            
            # Acknowledge alert
            success = await system.acknowledge_alert(alert.id)
            assert success == True
            assert alert.status == AlertStatus.ACKNOWLEDGED
            
            # Resolve alert
            success = await system.resolve_alert(alert.id)
            assert success == True
            assert alert.status == AlertStatus.RESOLVED
            assert alert.id not in system.active_alerts
    
    @pytest.mark.asyncio
    async def test_multiple_severity_alerts(self):
        """Test handling multiple alerts with different severities"""
        system = create_enhanced_alert_system()
        
        # Configure log notification
        log_config = NotificationConfig(
            channel=NotificationChannel.LOG,
            enabled=True
        )
        system.configure_notification_channel(NotificationChannel.LOG, log_config)
        
        with patch.object(system.notification_dispatcher, 'send_notification', return_value=True):
            # Trigger multiple alerts
            metrics = {
                "latency": {"voice_to_voice_p95": 600},    # WARNING
                "connection": {"success_rate": 80},        # CRITICAL
                "cost": {"current_hour_cost": 15.0}        # WARNING
            }
            
            alerts = await system.evaluate_metrics(metrics)
            
            # Should have multiple alerts
            assert len(alerts) >= 2
            
            # Check severity distribution
            severities = [alert.severity for alert in alerts]
            assert AlertSeverity.WARNING in severities
            assert AlertSeverity.CRITICAL in severities
    
    @pytest.mark.asyncio
    async def test_alert_suppression_and_rate_limiting(self):
        """Test alert suppression and notification rate limiting"""
        system = EnhancedAlertSystem()
        
        # Configure log notification with rate limiting
        log_config = NotificationConfig(
            channel=NotificationChannel.LOG,
            enabled=True,
            rate_limit=60  # 1 minute
        )
        system.configure_notification_channel(NotificationChannel.LOG, log_config)
        
        rule = AlertRule(
            name="test_rule",
            description="Test rule",
            metric_conditions={"test.value": {"operator": "gt", "threshold": 50}},
            severity=AlertSeverity.WARNING,
            suppression_window=300
        )
        system.add_alert_rule(rule)
        
        metrics = {"test": {"value": 100}}
        
        with patch.object(system.notification_dispatcher, 'send_notification', return_value=True) as mock_send:
            # First evaluation - should create alert and send notification
            alerts1 = await system.evaluate_metrics(metrics)
            assert len(alerts1) == 1
            assert mock_send.call_count == 1
            
            # Suppress the alert
            system.suppress_alert(alerts1[0].id)
            
            # Second evaluation - should not create new alert due to suppression
            alerts2 = await system.evaluate_metrics(metrics)
            assert len(alerts2) == 0
            assert mock_send.call_count == 1  # No additional notifications


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 