"""
Enhanced Alert System for Sovereign 4.0 Realtime API Monitoring

Provides advanced alerting capabilities with multiple notification channels,
escalation policies, and custom alert rules for comprehensive monitoring
of Realtime API performance, costs, and stability.

Key Features:
- Multi-channel notifications (Email via SendGrid, Slack)
- Configurable escalation policies with severity levels
- Custom alert rules based on metric combinations
- Template-based alert messages with context injection
- Alert suppression and rate limiting
- Historical alert tracking and analytics
- Integration with existing RealtimeMetricsCollector
"""

import asyncio
import logging
import time
import json
import smtplib
from typing import Dict, List, Optional, Any, Union, Callable, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import threading
import hashlib
from pathlib import Path
import re

# Third-party notification libraries (optional dependencies)
try:
    from slack_sdk import WebClient as SlackClient
    from slack_sdk.errors import SlackApiError
    HAS_SLACK = True
except ImportError:
    HAS_SLACK = False
    SlackClient = None
    SlackApiError = Exception

try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail
    HAS_SENDGRID = True
except ImportError:
    HAS_SENDGRID = False
    SendGridAPIClient = None
    Mail = None

try:
    from jinja2 import Template, Environment, BaseLoader
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False
    Template = None

try:
    from croniter import croniter
    HAS_CRONITER = True
except ImportError:
    HAS_CRONITER = False
    croniter = None


class AlertSeverity(Enum):
    """Alert severity levels with escalation order"""
    INFO = 1
    WARNING = 2
    CRITICAL = 3
    EMERGENCY = 4


class AlertStatus(Enum):
    """Alert lifecycle status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class NotificationChannel(Enum):
    """Available notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    LOG = "log"


@dataclass
class AlertRule:
    """Custom alert rule configuration"""
    name: str
    description: str
    metric_conditions: Dict[str, Any]  # Metric thresholds and conditions
    severity: AlertSeverity
    enabled: bool = True
    suppression_window: int = 300  # 5 minutes in seconds
    evaluation_window: int = 60   # 1 minute in seconds
    min_occurrences: int = 1      # Minimum occurrences before triggering
    custom_message_template: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def evaluate(self, metrics: Dict[str, Any]) -> bool:
        """Evaluate if this rule should trigger based on current metrics"""
        try:
            # Extract metric conditions
            for metric_path, condition in self.metric_conditions.items():
                value = self._get_nested_metric(metrics, metric_path)
                if value is None:
                    continue
                
                # Evaluate condition
                if not self._evaluate_condition(value, condition):
                    return False
            
            return True
        except Exception as e:
            logging.error(f"Error evaluating alert rule {self.name}: {e}")
            return False
    
    def _get_nested_metric(self, metrics: Dict[str, Any], path: str) -> Optional[Any]:
        """Get metric value from nested dictionary using dot notation"""
        keys = path.split('.')
        current = metrics
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def _evaluate_condition(self, value: Any, condition: Dict[str, Any]) -> bool:
        """Evaluate a single condition against a metric value"""
        operator = condition.get('operator', 'gt')
        threshold = condition.get('threshold')
        
        if threshold is None:
            return False
        
        if operator == 'gt':
            return value > threshold
        elif operator == 'gte':
            return value >= threshold
        elif operator == 'lt':
            return value < threshold
        elif operator == 'lte':
            return value <= threshold
        elif operator == 'eq':
            return value == threshold
        elif operator == 'ne':
            return value != threshold
        elif operator == 'contains':
            return str(threshold) in str(value)
        elif operator == 'regex':
            return bool(re.search(str(threshold), str(value)))
        
        return False


@dataclass
class EscalationPolicy:
    """Escalation policy configuration"""
    name: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    escalation_delay: int = 900  # 15 minutes in seconds
    max_escalations: int = 3
    auto_resolve_timeout: int = 3600  # 1 hour in seconds
    
    def get_escalation_step(self, escalation_level: int) -> Optional[Dict[str, Any]]:
        """Get escalation configuration for a specific level"""
        if 0 <= escalation_level < len(self.steps):
            return self.steps[escalation_level]
        return None


@dataclass
class NotificationConfig:
    """Notification channel configuration"""
    channel: NotificationChannel
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    rate_limit: int = 60  # Minimum seconds between notifications
    retry_attempts: int = 3
    retry_delay: int = 30  # Seconds between retries
    
    def is_rate_limited(self, last_notification: Optional[float]) -> bool:
        """Check if notification is rate limited"""
        if last_notification is None:
            return False
        return (time.time() - last_notification) < self.rate_limit


@dataclass
class Alert:
    """Active alert instance"""
    id: str
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    title: str
    message: str
    context: Dict[str, Any]
    created_at: float
    updated_at: float
    acknowledged_at: Optional[float] = None
    resolved_at: Optional[float] = None
    escalation_level: int = 0
    last_escalation: Optional[float] = None
    occurrence_count: int = 1
    tags: List[str] = field(default_factory=list)
    
    def acknowledge(self, acknowledger: str = "system") -> None:
        """Acknowledge the alert"""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = time.time()
        self.updated_at = time.time()
    
    def resolve(self, resolver: str = "system") -> None:
        """Resolve the alert"""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = time.time()
        self.updated_at = time.time()
    
    def escalate(self) -> None:
        """Escalate the alert to next level"""
        self.escalation_level += 1
        self.last_escalation = time.time()
        self.updated_at = time.time()


@dataclass
class AlertSystemStats:
    """Alert system performance statistics"""
    total_alerts_generated: int = 0
    alerts_by_severity: Dict[str, int] = field(default_factory=lambda: {
        "INFO": 0, "WARNING": 0, "CRITICAL": 0, "EMERGENCY": 0
    })
    notifications_sent: Dict[str, int] = field(default_factory=dict)
    notification_failures: Dict[str, int] = field(default_factory=dict)
    average_resolution_time: float = 0.0
    active_alerts_count: int = 0
    suppressed_alerts_count: int = 0


class MessageTemplateEngine:
    """Template engine for alert messages"""
    
    def __init__(self):
        self.templates = {
            'default': """
🚨 **Alert: {{ alert.title }}**

**Severity:** {{ alert.severity.name }}
**Time:** {{ alert.created_at | timestamp }}
**System:** Sovereign 4.0 Realtime API

**Details:**
{{ alert.message }}

**Metrics Context:**
{% for key, value in alert.context.items() %}
• **{{ key }}:** {{ value }}
{% endfor %}

**Alert ID:** {{ alert.id }}
            """.strip(),
            
            'slack': """
:warning: *{{ alert.title }}*

*Severity:* {{ alert.severity.name }}
*Time:* {{ alert.created_at | timestamp }}

{{ alert.message }}

*Metrics:*
{% for key, value in alert.context.items() %}
• *{{ key }}:* {{ value }}
{% endfor %}
            """.strip(),
            
            'email_html': """
<h2 style="color: {% if alert.severity.name == 'EMERGENCY' %}#FF0000{% elif alert.severity.name == 'CRITICAL' %}#FF6600{% elif alert.severity.name == 'WARNING' %}#FFAA00{% else %}#0066CC{% endif %};">
    🚨 Alert: {{ alert.title }}
</h2>

<table style="border-collapse: collapse; width: 100%;">
    <tr><td style="padding: 8px; border: 1px solid #ddd;"><strong>Severity:</strong></td>
        <td style="padding: 8px; border: 1px solid #ddd;">{{ alert.severity.name }}</td></tr>
    <tr><td style="padding: 8px; border: 1px solid #ddd;"><strong>Time:</strong></td>
        <td style="padding: 8px; border: 1px solid #ddd;">{{ alert.created_at | timestamp }}</td></tr>
    <tr><td style="padding: 8px; border: 1px solid #ddd;"><strong>System:</strong></td>
        <td style="padding: 8px; border: 1px solid #ddd;">Sovereign 4.0 Realtime API</td></tr>
</table>

<h3>Details:</h3>
<p>{{ alert.message }}</p>

<h3>Metrics Context:</h3>
<ul>
{% for key, value in alert.context.items() %}
    <li><strong>{{ key }}:</strong> {{ value }}</li>
{% endfor %}
</ul>

<p><small>Alert ID: {{ alert.id }}</small></p>
            """.strip()
        }
        
        if HAS_JINJA2:
            self.env = Environment(loader=BaseLoader())
            self.env.filters['timestamp'] = self._format_timestamp
        
    def render_template(self, template_name: str, alert: Alert, 
                       custom_template: Optional[str] = None) -> str:
        """Render alert message using template"""
        try:
            if custom_template:
                template_content = custom_template
            else:
                template_content = self.templates.get(template_name, self.templates['default'])
            
            if HAS_JINJA2:
                template = self.env.from_string(template_content)
                return template.render(alert=alert)
            else:
                # Fallback simple substitution
                return self._simple_substitution(template_content, alert)
        
        except Exception as e:
            logging.error(f"Error rendering template {template_name}: {e}")
            return f"Alert: {alert.title} - {alert.message}"
    
    def _format_timestamp(self, timestamp: float) -> str:
        """Format timestamp for display"""
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    def _simple_substitution(self, template: str, alert: Alert) -> str:
        """Simple template substitution without Jinja2"""
        result = template
        result = result.replace("{{ alert.title }}", alert.title)
        result = result.replace("{{ alert.severity.name }}", alert.severity.name)
        result = result.replace("{{ alert.message }}", alert.message)
        result = result.replace("{{ alert.id }}", alert.id)
        result = result.replace("{{ alert.created_at | timestamp }}", 
                              self._format_timestamp(alert.created_at))
        
        # Simple context substitution
        context_str = "\n".join([f"• {k}: {v}" for k, v in alert.context.items()])
        result = result.replace("{% for key, value in alert.context.items() %}• **{{ key }}:** {{ value }}\n{% endfor %}", 
                              context_str)
        
        return result


class NotificationDispatcher:
    """Handles dispatching notifications through various channels"""
    
    def __init__(self):
        self.channels: Dict[NotificationChannel, NotificationConfig] = {}
        self.template_engine = MessageTemplateEngine()
        self.notification_history: deque = deque(maxlen=1000)
        self.last_notification_time: Dict[str, float] = {}
        self._lock = threading.Lock()
        
        # Initialize notification clients
        self.slack_client: Optional[SlackClient] = None
        self.sendgrid_client: Optional[SendGridAPIClient] = None
        
    def configure_channel(self, channel: NotificationChannel, config: NotificationConfig) -> None:
        """Configure a notification channel"""
        self.channels[channel] = config
        
        # Initialize client if needed
        if channel == NotificationChannel.SLACK and HAS_SLACK:
            token = config.config.get('token')
            if token:
                self.slack_client = SlackClient(token=token)
        
        elif channel == NotificationChannel.EMAIL and HAS_SENDGRID:
            api_key = config.config.get('api_key')
            if api_key:
                self.sendgrid_client = SendGridAPIClient(api_key=api_key)
    
    async def send_notification(self, alert: Alert, channel: NotificationChannel, 
                              escalation_step: Optional[Dict[str, Any]] = None) -> bool:
        """Send notification through specified channel"""
        try:
            config = self.channels.get(channel)
            if not config or not config.enabled:
                return False
            
            # Check rate limiting
            rate_key = f"{channel.value}_{alert.rule_name}"
            if config.is_rate_limited(self.last_notification_time.get(rate_key)):
                logging.info(f"Notification rate limited for {rate_key}")
                return False
            
            # Attempt notification with retries
            for attempt in range(config.retry_attempts):
                try:
                    success = await self._dispatch_notification(alert, channel, config, escalation_step)
                    if success:
                        with self._lock:
                            self.last_notification_time[rate_key] = time.time()
                            self.notification_history.append({
                                "timestamp": time.time(),
                                "alert_id": alert.id,
                                "channel": channel.value,
                                "success": True,
                                "attempt": attempt + 1
                            })
                        return True
                
                except Exception as e:
                    logging.error(f"Notification attempt {attempt + 1} failed for {channel.value}: {e}")
                    if attempt < config.retry_attempts - 1:
                        await asyncio.sleep(config.retry_delay)
            
            # Record failure
            with self._lock:
                self.notification_history.append({
                    "timestamp": time.time(),
                    "alert_id": alert.id,
                    "channel": channel.value,
                    "success": False,
                    "attempts": config.retry_attempts
                })
            
            return False
        
        except Exception as e:
            logging.error(f"Error sending notification through {channel.value}: {e}")
            return False
    
    async def _dispatch_notification(self, alert: Alert, channel: NotificationChannel,
                                   config: NotificationConfig, 
                                   escalation_step: Optional[Dict[str, Any]]) -> bool:
        """Dispatch notification to specific channel"""
        if channel == NotificationChannel.SLACK:
            return await self._send_slack_notification(alert, config, escalation_step)
        elif channel == NotificationChannel.EMAIL:
            return await self._send_email_notification(alert, config, escalation_step)
        elif channel == NotificationChannel.WEBHOOK:
            return await self._send_webhook_notification(alert, config, escalation_step)
        elif channel == NotificationChannel.LOG:
            return self._send_log_notification(alert, config)
        
        return False
    
    async def _send_slack_notification(self, alert: Alert, config: NotificationConfig,
                                     escalation_step: Optional[Dict[str, Any]]) -> bool:
        """Send Slack notification"""
        if not self.slack_client or not HAS_SLACK:
            return False
        
        try:
            channel = escalation_step.get('slack_channel') if escalation_step else config.config.get('channel', '#alerts')
            message = self.template_engine.render_template('slack', alert)
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.slack_client.chat_postMessage(
                    channel=channel,
                    text=message,
                    username="Sovereign Alert System",
                    icon_emoji=":warning:"
                )
            )
            
            return response.get("ok", False)
        
        except SlackApiError as e:
            logging.error(f"Slack API error: {e}")
            return False
    
    async def _send_email_notification(self, alert: Alert, config: NotificationConfig,
                                     escalation_step: Optional[Dict[str, Any]]) -> bool:
        """Send email notification"""
        if not self.sendgrid_client or not HAS_SENDGRID or not Mail:
            return False
        
        try:
            from_email = config.config.get('from_email', 'alerts@sovereign.ai')
            to_emails = escalation_step.get('emails') if escalation_step else config.config.get('to_emails', [])
            
            if not to_emails:
                return False
            
            subject = f"🚨 Sovereign Alert: {alert.title}"
            html_content = self.template_engine.render_template('email_html', alert)
            plain_content = self.template_engine.render_template('default', alert)
            
            message = Mail(
                from_email=from_email,
                to_emails=to_emails,
                subject=subject,
                html_content=html_content,
                plain_text_content=plain_content
            )
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.sendgrid_client.send(message)
            )
            
            return response.status_code == 202
        
        except Exception as e:
            logging.error(f"Email notification error: {e}")
            return False
    
    async def _send_webhook_notification(self, alert: Alert, config: NotificationConfig,
                                       escalation_step: Optional[Dict[str, Any]]) -> bool:
        """Send webhook notification"""
        try:
            import aiohttp
            
            webhook_url = escalation_step.get('webhook_url') if escalation_step else config.config.get('webhook_url')
            if not webhook_url:
                return False
            
            payload = {
                "alert": asdict(alert),
                "timestamp": time.time(),
                "escalation_level": alert.escalation_level
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload, timeout=30) as response:
                    return response.status < 400
        
        except Exception as e:
            logging.error(f"Webhook notification error: {e}")
            return False
    
    def _send_log_notification(self, alert: Alert, config: NotificationConfig) -> bool:
        """Send log notification"""
        try:
            log_level = config.config.get('log_level', 'warning').upper()
            message = self.template_engine.render_template('default', alert)
            
            logger = logging.getLogger("sovereign.alerts")
            getattr(logger, log_level.lower(), logger.warning)(message)
            
            return True
        
        except Exception as e:
            logging.error(f"Log notification error: {e}")
            return False


class EnhancedAlertSystem:
    """Main enhanced alert system with comprehensive alerting capabilities"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Core components
        self.alert_rules: Dict[str, AlertRule] = {}
        self.escalation_policies: Dict[str, EscalationPolicy] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=5000)
        self.suppressed_alerts: Set[str] = set()
        
        # Notification system
        self.notification_dispatcher = NotificationDispatcher()
        
        # Statistics and monitoring
        self.stats = AlertSystemStats()
        self.rule_evaluation_cache: Dict[str, Dict[str, Any]] = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Background task management
        self._background_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        self.logger.info("Enhanced Alert System initialized")
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add a custom alert rule"""
        with self._lock:
            self.alert_rules[rule.name] = rule
            self.logger.info(f"Added alert rule: {rule.name}")
    
    def add_escalation_policy(self, policy: EscalationPolicy) -> None:
        """Add an escalation policy"""
        with self._lock:
            self.escalation_policies[policy.name] = policy
            self.logger.info(f"Added escalation policy: {policy.name}")
    
    def configure_notification_channel(self, channel: NotificationChannel, 
                                     config: NotificationConfig) -> None:
        """Configure a notification channel"""
        self.notification_dispatcher.configure_channel(channel, config)
        self.logger.info(f"Configured notification channel: {channel.value}")
    
    async def evaluate_metrics(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Evaluate metrics against all alert rules"""
        triggered_alerts = []
        
        with self._lock:
            rules_to_evaluate = list(self.alert_rules.values())
        
        for rule in rules_to_evaluate:
            if not rule.enabled:
                continue
            
            try:
                # Check if rule should trigger
                if rule.evaluate(metrics):
                    alert = await self._process_rule_trigger(rule, metrics)
                    if alert:
                        triggered_alerts.append(alert)
            
            except Exception as e:
                self.logger.error(f"Error evaluating rule {rule.name}: {e}")
        
        return triggered_alerts
    
    async def _process_rule_trigger(self, rule: AlertRule, metrics: Dict[str, Any]) -> Optional[Alert]:
        """Process a triggered alert rule"""
        # Generate alert ID
        alert_id = self._generate_alert_id(rule, metrics)
        
        with self._lock:
            # Check if alert already exists
            if alert_id in self.active_alerts:
                existing_alert = self.active_alerts[alert_id]
                existing_alert.occurrence_count += 1
                existing_alert.updated_at = time.time()
                
                # Check if we should escalate
                if self._should_escalate(existing_alert):
                    await self._escalate_alert(existing_alert)
                
                return existing_alert
            
            # Check suppression
            if alert_id in self.suppressed_alerts:
                return None
            
            # Create new alert
            alert = Alert(
                id=alert_id,
                rule_name=rule.name,
                severity=rule.severity,
                status=AlertStatus.ACTIVE,
                title=self._generate_alert_title(rule, metrics),
                message=self._generate_alert_message(rule, metrics),
                context=self._extract_alert_context(rule, metrics),
                created_at=time.time(),
                updated_at=time.time(),
                tags=rule.tags.copy()
            )
            
            # Store alert
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Update statistics
            self.stats.total_alerts_generated += 1
            self.stats.alerts_by_severity[alert.severity.name] += 1
            self.stats.active_alerts_count += 1
        
        # Send initial notifications
        await self._send_alert_notifications(alert)
        
        self.logger.info(f"Generated new alert: {alert.title} (ID: {alert.id})")
        return alert
    
    def _generate_alert_id(self, rule: AlertRule, metrics: Dict[str, Any]) -> str:
        """Generate unique alert ID based on rule and context"""
        context_str = json.dumps(self._extract_alert_context(rule, metrics), sort_keys=True)
        hash_input = f"{rule.name}:{context_str}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def _generate_alert_title(self, rule: AlertRule, metrics: Dict[str, Any]) -> str:
        """Generate alert title"""
        if rule.custom_message_template:
            try:
                template = Template(rule.custom_message_template) if HAS_JINJA2 else None
                if template:
                    return template.render(rule=rule, metrics=metrics)
            except Exception:
                pass
        
        return f"{rule.name}: {rule.description}"
    
    def _generate_alert_message(self, rule: AlertRule, metrics: Dict[str, Any]) -> str:
        """Generate detailed alert message"""
        context = self._extract_alert_context(rule, metrics)
        conditions_met = []
        
        for metric_path, condition in rule.metric_conditions.items():
            value = rule._get_nested_metric(metrics, metric_path)
            if value is not None:
                operator = condition.get('operator', 'gt')
                threshold = condition.get('threshold')
                conditions_met.append(f"{metric_path} {operator} {threshold} (current: {value})")
        
        return f"Alert conditions met: {', '.join(conditions_met)}"
    
    def _extract_alert_context(self, rule: AlertRule, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant context for alert"""
        context = {}
        
        for metric_path in rule.metric_conditions.keys():
            value = rule._get_nested_metric(metrics, metric_path)
            if value is not None:
                context[metric_path] = value
        
        return context
    
    def _should_escalate(self, alert: Alert) -> bool:
        """Check if alert should be escalated"""
        if alert.status != AlertStatus.ACTIVE:
            return False
        
        # Find escalation policy
        policy_name = f"default"  # Could be configurable per rule
        policy = self.escalation_policies.get(policy_name)
        if not policy:
            return False
        
        # Check escalation timing
        if alert.last_escalation is None:
            time_since_creation = time.time() - alert.created_at
            return time_since_creation >= policy.escalation_delay
        else:
            time_since_escalation = time.time() - alert.last_escalation
            return time_since_escalation >= policy.escalation_delay
    
    async def _escalate_alert(self, alert: Alert) -> None:
        """Escalate an alert to the next level"""
        policy_name = f"default"
        policy = self.escalation_policies.get(policy_name)
        if not policy:
            return
        
        if alert.escalation_level >= policy.max_escalations:
            return
        
        alert.escalate()
        escalation_step = policy.get_escalation_step(alert.escalation_level)
        
        if escalation_step:
            await self._send_escalation_notifications(alert, escalation_step)
        
        self.logger.warning(f"Escalated alert {alert.id} to level {alert.escalation_level}")
    
    async def _send_alert_notifications(self, alert: Alert) -> None:
        """Send notifications for a new alert"""
        # Determine notification channels based on severity
        channels = self._get_notification_channels_for_severity(alert.severity)
        
        for channel in channels:
            success = await self.notification_dispatcher.send_notification(alert, channel)
            if success:
                self.stats.notifications_sent[channel.value] = \
                    self.stats.notifications_sent.get(channel.value, 0) + 1
            else:
                self.stats.notification_failures[channel.value] = \
                    self.stats.notification_failures.get(channel.value, 0) + 1
    
    async def _send_escalation_notifications(self, alert: Alert, 
                                           escalation_step: Dict[str, Any]) -> None:
        """Send escalation notifications"""
        channels = escalation_step.get('channels', [NotificationChannel.EMAIL])
        
        for channel in channels:
            if isinstance(channel, str):
                channel = NotificationChannel(channel)
            
            await self.notification_dispatcher.send_notification(alert, channel, escalation_step)
    
    def _get_notification_channels_for_severity(self, severity: AlertSeverity) -> List[NotificationChannel]:
        """Get notification channels based on alert severity"""
        if severity == AlertSeverity.EMERGENCY:
            return [NotificationChannel.SLACK, NotificationChannel.EMAIL, NotificationChannel.LOG]
        elif severity == AlertSeverity.CRITICAL:
            return [NotificationChannel.SLACK, NotificationChannel.EMAIL]
        elif severity == AlertSeverity.WARNING:
            return [NotificationChannel.SLACK]
        else:
            return [NotificationChannel.LOG]
    
    async def acknowledge_alert(self, alert_id: str, acknowledger: str = "system") -> bool:
        """Acknowledge an active alert"""
        with self._lock:
            alert = self.active_alerts.get(alert_id)
            if alert and alert.status == AlertStatus.ACTIVE:
                alert.acknowledge(acknowledger)
                self.logger.info(f"Alert {alert_id} acknowledged by {acknowledger}")
                return True
        return False
    
    async def resolve_alert(self, alert_id: str, resolver: str = "system") -> bool:
        """Resolve an active alert"""
        with self._lock:
            alert = self.active_alerts.get(alert_id)
            if alert and alert.status in [AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED]:
                alert.resolve(resolver)
                
                # Update statistics
                resolution_time = alert.resolved_at - alert.created_at
                self._update_average_resolution_time(resolution_time)
                self.stats.active_alerts_count = max(0, self.stats.active_alerts_count - 1)
                
                # Remove from active alerts
                del self.active_alerts[alert_id]
                
                self.logger.info(f"Alert {alert_id} resolved by {resolver}")
                return True
        return False
    
    def _update_average_resolution_time(self, resolution_time: float) -> None:
        """Update average resolution time statistic"""
        # Simple moving average approximation
        if self.stats.average_resolution_time == 0:
            self.stats.average_resolution_time = resolution_time
        else:
            self.stats.average_resolution_time = \
                (self.stats.average_resolution_time * 0.9) + (resolution_time * 0.1)
    
    def suppress_alert(self, alert_id: str, duration: int = 3600) -> None:
        """Suppress an alert for a specified duration"""
        with self._lock:
            self.suppressed_alerts.add(alert_id)
            self.stats.suppressed_alerts_count += 1
        
        # Schedule unsuppression
        async def unsuppress():
            await asyncio.sleep(duration)
            with self._lock:
                self.suppressed_alerts.discard(alert_id)
                self.stats.suppressed_alerts_count = max(0, self.stats.suppressed_alerts_count - 1)
        
        asyncio.create_task(unsuppress())
        self.logger.info(f"Alert {alert_id} suppressed for {duration} seconds")
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get list of active alerts, optionally filtered by severity"""
        with self._lock:
            alerts = list(self.active_alerts.values())
            if severity:
                alerts = [alert for alert in alerts if alert.severity == severity]
            return sorted(alerts, key=lambda a: a.created_at, reverse=True)
    
    def get_alert_statistics(self) -> AlertSystemStats:
        """Get alert system statistics"""
        with self._lock:
            # Update active count
            self.stats.active_alerts_count = len(self.active_alerts)
            return self.stats
    
    async def start_background_monitoring(self) -> None:
        """Start background monitoring and maintenance tasks"""
        if self._background_task and not self._background_task.done():
            return
        
        self._shutdown_event.clear()
        self._background_task = asyncio.create_task(self._background_monitoring_loop())
        self.logger.info("Started background alert monitoring")
    
    async def stop_background_monitoring(self) -> None:
        """Stop background monitoring"""
        self._shutdown_event.set()
        if self._background_task:
            await self._background_task
        self.logger.info("Stopped background alert monitoring")
    
    async def _background_monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while not self._shutdown_event.is_set():
            try:
                # Check for escalations
                await self._check_escalations()
                
                # Auto-resolve stale alerts
                await self._auto_resolve_stale_alerts()
                
                # Clean up old alert history
                self._cleanup_alert_history()
                
                # Wait before next iteration
                await asyncio.sleep(60)  # Run every minute
            
            except Exception as e:
                self.logger.error(f"Error in background monitoring: {e}")
                await asyncio.sleep(10)
    
    async def _check_escalations(self) -> None:
        """Check for alerts that need escalation"""
        with self._lock:
            alerts_to_escalate = [
                alert for alert in self.active_alerts.values()
                if self._should_escalate(alert)
            ]
        
        for alert in alerts_to_escalate:
            await self._escalate_alert(alert)
    
    async def _auto_resolve_stale_alerts(self) -> None:
        """Auto-resolve alerts that have been active too long"""
        current_time = time.time()
        alerts_to_resolve = []
        
        with self._lock:
            for alert in self.active_alerts.values():
                # Find associated escalation policy
                policy = self.escalation_policies.get("default")
                if policy and policy.auto_resolve_timeout > 0:
                    if (current_time - alert.created_at) > policy.auto_resolve_timeout:
                        alerts_to_resolve.append(alert.id)
        
        for alert_id in alerts_to_resolve:
            await self.resolve_alert(alert_id, "auto-resolve")
    
    def _cleanup_alert_history(self) -> None:
        """Clean up old alert history"""
        # History cleanup is automatic via deque maxlen
        pass
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export alert system configuration"""
        with self._lock:
            return {
                "alert_rules": {name: asdict(rule) for name, rule in self.alert_rules.items()},
                "escalation_policies": {name: asdict(policy) for name, policy in self.escalation_policies.items()},
                "notification_channels": {
                    channel.value: asdict(config) 
                    for channel, config in self.notification_dispatcher.channels.items()
                }
            }
    
    def import_configuration(self, config: Dict[str, Any]) -> None:
        """Import alert system configuration"""
        # Import alert rules
        for rule_data in config.get("alert_rules", {}).values():
            rule = AlertRule(**rule_data)
            self.add_alert_rule(rule)
        
        # Import escalation policies
        for policy_data in config.get("escalation_policies", {}).values():
            policy = EscalationPolicy(**policy_data)
            self.add_escalation_policy(policy)
        
        # Import notification channels
        for channel_name, config_data in config.get("notification_channels", {}).items():
            channel = NotificationChannel(channel_name)
            notification_config = NotificationConfig(**config_data)
            self.configure_notification_channel(channel, notification_config)


# Factory function for easy instantiation
def create_enhanced_alert_system(
    logger: Optional[logging.Logger] = None,
    config_file: Optional[str] = None
) -> EnhancedAlertSystem:
    """Create and configure enhanced alert system"""
    alert_system = EnhancedAlertSystem(logger)
    
    # Load configuration if provided
    if config_file and Path(config_file).exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            alert_system.import_configuration(config)
        except Exception as e:
            logging.error(f"Error loading alert configuration: {e}")
    
    # Add default escalation policy
    default_policy = EscalationPolicy(
        name="default",
        steps=[
            {"channels": ["slack"], "slack_channel": "#alerts"},
            {"channels": ["email"], "emails": ["ops@sovereign.ai"]},
            {"channels": ["email", "slack"], "emails": ["emergency@sovereign.ai"], "slack_channel": "#emergencies"}
        ],
        escalation_delay=900,  # 15 minutes
        max_escalations=2,
        auto_resolve_timeout=3600  # 1 hour
    )
    alert_system.add_escalation_policy(default_policy)
    
    # Add default alert rules for common Realtime API issues
    _add_default_alert_rules(alert_system)
    
    return alert_system


def _add_default_alert_rules(alert_system: EnhancedAlertSystem) -> None:
    """Add default alert rules for common monitoring scenarios"""
    
    # High latency alert
    high_latency_rule = AlertRule(
        name="high_voice_latency",
        description="Voice-to-voice latency exceeds 500ms",
        metric_conditions={
            "latency.voice_to_voice_p95": {"operator": "gt", "threshold": 500}
        },
        severity=AlertSeverity.WARNING,
        suppression_window=300,
        tags=["latency", "performance"]
    )
    alert_system.add_alert_rule(high_latency_rule)
    
    # Connection failure alert
    connection_failure_rule = AlertRule(
        name="low_connection_success_rate",
        description="Connection success rate below 95%",
        metric_conditions={
            "connection.success_rate": {"operator": "lt", "threshold": 95}
        },
        severity=AlertSeverity.CRITICAL,
        suppression_window=600,
        tags=["connection", "reliability"]
    )
    alert_system.add_alert_rule(connection_failure_rule)
    
    # High cost alert
    high_cost_rule = AlertRule(
        name="high_hourly_cost",
        description="Hourly API costs exceed $10",
        metric_conditions={
            "cost.current_hour_cost": {"operator": "gt", "threshold": 10.0}
        },
        severity=AlertSeverity.WARNING,
        suppression_window=1800,
        tags=["cost", "budget"]
    )
    alert_system.add_alert_rule(high_cost_rule)
    
    # Audio quality degradation
    audio_quality_rule = AlertRule(
        name="poor_audio_quality",
        description="Audio quality score below 0.7",
        metric_conditions={
            "audio.average_quality": {"operator": "lt", "threshold": 0.7}
        },
        severity=AlertSeverity.WARNING,
        suppression_window=300,
        tags=["audio", "quality"]
    )
    alert_system.add_alert_rule(audio_quality_rule) 