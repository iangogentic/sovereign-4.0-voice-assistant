"""
Multi-Channel Alerting System

This module implements an advanced alerting system for AI voice assistant monitoring:
- Intelligent alert routing based on severity and context
- Multi-channel support (email, webhook, Slack, Teams)
- Context-aware alert enrichment with remediation suggestions
- Alert correlation engine for noise reduction
- Rate limiting and escalation policies
"""

import asyncio
import aiohttp
import aiosmtplib
import json
import logging
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
import smtplib
from email.mime.text import MIMEText

try:
    from slack_sdk.web.async_client import AsyncWebClient as SlackAsyncWebClient
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False
    SlackAsyncWebClient = None


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertChannel(Enum):
    """Supported alert channels"""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"
    SMS = "sms"
    DASHBOARD = "dashboard"


class AlertStatus(Enum):
    """Alert processing status"""
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    SUPPRESSED = "suppressed"
    CORRELATED = "correlated"


@dataclass
class AlertDestination:
    """Alert destination configuration"""
    name: str
    channel: AlertChannel
    config: Dict[str, Any]
    severity_threshold: AlertSeverity = AlertSeverity.WARNING
    rate_limit_per_hour: int = 50
    enabled: bool = True
    tags: List[str] = field(default_factory=list)


@dataclass
class Alert:
    """Alert message"""
    id: str
    timestamp: float
    severity: AlertSeverity
    title: str
    description: str
    source: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    remediation: List[str] = field(default_factory=list)
    related_alerts: List[str] = field(default_factory=list)
    status: AlertStatus = AlertStatus.PENDING
    correlation_id: Optional[str] = None


@dataclass
class AlertCorrelationRule:
    """Rule for correlating related alerts"""
    name: str
    primary_pattern: str
    secondary_patterns: List[str]
    time_window: int  # seconds
    correlation_threshold: float  # 0.0 to 1.0
    action: str  # "suppress", "merge", "escalate"
    description: str


@dataclass
class AlertingConfig:
    """Configuration for alerting system"""
    # Rate limiting
    global_rate_limit_per_hour: int = 100
    burst_rate_limit: int = 10
    burst_time_window: int = 300  # 5 minutes
    
    # Correlation settings
    correlation_window: int = 300  # 5 minutes
    max_correlation_size: int = 10
    
    # Escalation settings
    escalation_timeout: int = 1800  # 30 minutes
    escalation_retry_count: int = 3
    
    # Storage settings
    alert_history_size: int = 10000
    correlation_history_size: int = 1000
    
    # Email settings
    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_use_tls: bool = True
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    email_from: str = "noreply@sovereign.ai"
    
    # Webhook settings
    webhook_timeout: int = 10
    webhook_retry_count: int = 3
    
    # Slack settings
    slack_token: Optional[str] = None
    slack_default_channel: str = "#alerts"
    
    # Storage
    persistence_enabled: bool = True
    storage_dir: str = ".taskmaster/alerts"


class AlertingSystem:
    """
    Advanced multi-channel alerting system
    
    Features:
    - Multiple notification channels with intelligent routing
    - Context-aware alert enrichment
    - Alert correlation and noise reduction
    - Rate limiting and escalation policies
    - Persistent storage and history tracking
    """
    
    def __init__(self, config: Optional[AlertingConfig] = None):
        self.config = config or AlertingConfig()
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
        
        # Alert destinations
        self.destinations: Dict[str, AlertDestination] = {}
        
        # Alert storage
        self.alert_history: deque = deque(maxlen=self.config.alert_history_size)
        self.correlation_history: deque = deque(maxlen=self.config.correlation_history_size)
        
        # Rate limiting tracking
        self.rate_limit_counters: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.global_rate_limit_per_hour)
        )
        
        # Correlation engine
        self.correlation_rules: List[AlertCorrelationRule] = []
        self.active_correlations: Dict[str, List[Alert]] = {}
        
        # Alert enrichment callbacks
        self.enrichment_callbacks: List[Callable[[Alert], Alert]] = []
        
        # Initialize storage
        self._init_storage()
        
        # Load default correlation rules
        self._load_default_correlation_rules()
        
        # Initialize clients
        self.slack_client = None
        if SLACK_AVAILABLE and self.config.slack_token:
            self.slack_client = SlackAsyncWebClient(token=self.config.slack_token)
    
    def _init_storage(self):
        """Initialize persistent storage directory"""
        if self.config.persistence_enabled:
            storage_path = Path(self.config.storage_dir)
            storage_path.mkdir(parents=True, exist_ok=True)
    
    def _load_default_correlation_rules(self):
        """Load default correlation rules for common alert patterns"""
        default_rules = [
            AlertCorrelationRule(
                name="latency_cascade",
                primary_pattern="*_latency_high",
                secondary_patterns=["stt_latency_high", "llm_latency_high", "tts_latency_high"],
                time_window=300,
                correlation_threshold=0.8,
                action="merge",
                description="Correlate cascading latency issues across pipeline"
            ),
            AlertCorrelationRule(
                name="resource_exhaustion",
                primary_pattern="memory_usage_high",
                secondary_patterns=["cpu_usage_high", "gpu_usage_high", "disk_usage_high"],
                time_window=600,
                correlation_threshold=0.7,
                action="merge",
                description="Correlate resource exhaustion patterns"
            ),
            AlertCorrelationRule(
                name="model_degradation",
                primary_pattern="accuracy_drop",
                secondary_patterns=["drift_detected", "error_rate_high", "bleu_score_low"],
                time_window=900,
                correlation_threshold=0.9,
                action="escalate",
                description="Correlate model performance degradation signals"
            ),
            AlertCorrelationRule(
                name="api_connectivity",
                primary_pattern="*_api_error",
                secondary_patterns=["openai_api_error", "openrouter_api_error", "network_timeout"],
                time_window=180,
                correlation_threshold=0.6,
                action="suppress",
                description="Correlate API connectivity issues"
            )
        ]
        
        self.correlation_rules.extend(default_rules)
    
    def add_destination(self, destination: AlertDestination):
        """Add an alert destination"""
        with self._lock:
            self.destinations[destination.name] = destination
            self.logger.info(f"Added alert destination: {destination.name}")
    
    def remove_destination(self, name: str):
        """Remove an alert destination"""
        with self._lock:
            if name in self.destinations:
                del self.destinations[name]
                self.logger.info(f"Removed alert destination: {name}")
    
    def add_correlation_rule(self, rule: AlertCorrelationRule):
        """Add a correlation rule"""
        with self._lock:
            self.correlation_rules.append(rule)
            self.logger.info(f"Added correlation rule: {rule.name}")
    
    def add_enrichment_callback(self, callback: Callable[[Alert], Alert]):
        """Add an alert enrichment callback"""
        self.enrichment_callbacks.append(callback)
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        return f"alert_{int(time.time() * 1000)}_{id(object())}"
    
    def _check_rate_limit(self, destination_name: str) -> bool:
        """Check if destination is within rate limits"""
        current_time = time.time()
        hour_ago = current_time - 3600
        
        # Clean old entries
        counter = self.rate_limit_counters[destination_name]
        while counter and counter[0] < hour_ago:
            counter.popleft()
        
        # Check limits
        destination = self.destinations.get(destination_name)
        if destination and len(counter) >= destination.rate_limit_per_hour:
            return False
        
        # Add current timestamp
        counter.append(current_time)
        return True
    
    def _pattern_matches(self, pattern: str, alert_source: str) -> bool:
        """Check if alert source matches correlation pattern"""
        if pattern == "*":
            return True
        if pattern.startswith("*") and pattern.endswith("*"):
            return pattern[1:-1] in alert_source
        if pattern.startswith("*"):
            return alert_source.endswith(pattern[1:])
        if pattern.endswith("*"):
            return alert_source.startswith(pattern[:-1])
        return pattern == alert_source
    
    def _find_correlations(self, alert: Alert) -> List[AlertCorrelationRule]:
        """Find correlation rules that match the given alert"""
        matching_rules = []
        
        for rule in self.correlation_rules:
            # Check if alert matches primary pattern
            if self._pattern_matches(rule.primary_pattern, alert.source):
                matching_rules.append(rule)
                continue
            
            # Check if alert matches any secondary pattern
            for pattern in rule.secondary_patterns:
                if self._pattern_matches(pattern, alert.source):
                    matching_rules.append(rule)
                    break
        
        return matching_rules
    
    def _apply_correlation(self, alert: Alert, 
                          rule: AlertCorrelationRule) -> Optional[str]:
        """Apply correlation rule to alert"""
        current_time = time.time()
        window_start = current_time - rule.time_window
        
        # Find related alerts in time window
        related_alerts = []
        for hist_alert in self.alert_history:
            if hist_alert.timestamp < window_start:
                continue
            
            # Check if historical alert matches rule patterns
            is_primary = self._pattern_matches(rule.primary_pattern, hist_alert.source)
            is_secondary = any(
                self._pattern_matches(pattern, hist_alert.source)
                for pattern in rule.secondary_patterns
            )
            
            if is_primary or is_secondary:
                related_alerts.append(hist_alert)
        
        # Check correlation threshold
        if len(related_alerts) < 2:
            return None
        
        # Calculate correlation strength
        unique_patterns = set()
        for rel_alert in related_alerts:
            if self._pattern_matches(rule.primary_pattern, rel_alert.source):
                unique_patterns.add("primary")
            for pattern in rule.secondary_patterns:
                if self._pattern_matches(pattern, rel_alert.source):
                    unique_patterns.add(pattern)
        
        correlation_strength = len(unique_patterns) / (1 + len(rule.secondary_patterns))
        
        if correlation_strength >= rule.correlation_threshold:
            # Create correlation ID
            correlation_id = f"corr_{rule.name}_{int(current_time)}"
            
            # Apply correlation action
            if rule.action == "suppress":
                alert.status = AlertStatus.SUPPRESSED
            elif rule.action == "merge":
                alert.status = AlertStatus.CORRELATED
                alert.related_alerts = [a.id for a in related_alerts]
            elif rule.action == "escalate":
                alert.severity = AlertSeverity.EMERGENCY
                alert.related_alerts = [a.id for a in related_alerts]
            
            alert.correlation_id = correlation_id
            self.active_correlations[correlation_id] = related_alerts + [alert]
            
            self.logger.info(
                f"Applied correlation rule '{rule.name}' to alert {alert.id}: "
                f"action={rule.action}, strength={correlation_strength:.2f}"
            )
            
            return correlation_id
        
        return None
    
    def _enrich_alert(self, alert: Alert) -> Alert:
        """Enrich alert with additional context and remediation suggestions"""
        # Apply custom enrichment callbacks
        for callback in self.enrichment_callbacks:
            try:
                alert = callback(alert)
            except Exception as e:
                self.logger.error(f"Error in enrichment callback: {e}")
        
        # Add standard enrichment
        alert.context.update({
            'alert_id': alert.id,
            'timestamp_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(alert.timestamp)),
            'severity_level': self._get_severity_level(alert.severity),
            'alert_age_seconds': time.time() - alert.timestamp
        })
        
        # Add remediation suggestions based on alert type
        if not alert.remediation:
            alert.remediation = self._generate_remediation_suggestions(alert)
        
        return alert
    
    def _get_severity_level(self, severity: AlertSeverity) -> int:
        """Get numeric severity level"""
        levels = {
            AlertSeverity.INFO: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.CRITICAL: 3,
            AlertSeverity.EMERGENCY: 4
        }
        return levels.get(severity, 1)
    
    def _generate_remediation_suggestions(self, alert: Alert) -> List[str]:
        """Generate remediation suggestions based on alert type"""
        suggestions = []
        
        if alert.metric_name:
            metric_lower = alert.metric_name.lower()
            
            if 'latency' in metric_lower:
                suggestions.extend([
                    "Check network connectivity and API response times",
                    "Review LLM router configuration for optimal model selection",
                    "Consider scaling to offline fallback mode",
                    "Verify system resource availability (CPU, memory, GPU)"
                ])
            
            elif 'memory' in metric_lower:
                suggestions.extend([
                    "Clear application caches and temporary data",
                    "Restart services if memory usage exceeds 80%",
                    "Check for memory leaks in audio processing pipeline",
                    "Consider increasing available memory resources"
                ])
            
            elif 'accuracy' in metric_lower or 'bleu' in metric_lower:
                suggestions.extend([
                    "Run model drift detection analysis",
                    "Review recent training data quality",
                    "Check for changes in input data distribution",
                    "Consider model retraining or fine-tuning"
                ])
            
            elif 'drift' in metric_lower:
                suggestions.extend([
                    "Investigate recent changes in data pipeline",
                    "Review data preprocessing steps",
                    "Consider updating model baselines",
                    "Check for external factors affecting data quality"
                ])
            
            elif 'error' in metric_lower:
                suggestions.extend([
                    "Review error logs for patterns and root causes",
                    "Check service health and connectivity",
                    "Verify configuration settings",
                    "Consider service restart if error rate is high"
                ])
        
        # Add generic suggestions if no specific ones
        if not suggestions:
            suggestions.extend([
                "Monitor system closely for additional symptoms",
                "Review recent configuration changes",
                "Check system logs for related issues",
                "Contact system administrator if issue persists"
            ])
        
        return suggestions
    
    def _filter_destinations(self, alert: Alert) -> List[AlertDestination]:
        """Filter destinations based on alert severity and configuration"""
        eligible_destinations = []
        
        for dest in self.destinations.values():
            if not dest.enabled:
                continue
            
            # Check severity threshold
            alert_level = self._get_severity_level(alert.severity)
            dest_level = self._get_severity_level(dest.severity_threshold)
            
            if alert_level < dest_level:
                continue
            
            # Check tags
            if dest.tags and alert.tags:
                if not any(tag in alert.tags for tag in dest.tags):
                    continue
            
            # Check rate limits
            if not self._check_rate_limit(dest.name):
                self.logger.warning(f"Rate limit exceeded for destination {dest.name}")
                continue
            
            eligible_destinations.append(dest)
        
        return eligible_destinations
    
    async def _send_email(self, destination: AlertDestination, alert: Alert) -> bool:
        """Send alert via email"""
        try:
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.config.email_from
            msg['To'] = destination.config.get('email', '')
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Create email body
            body_parts = [
                f"Alert: {alert.title}",
                f"Severity: {alert.severity.value.upper()}",
                f"Source: {alert.source}",
                f"Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(alert.timestamp))}",
                "",
                "Description:",
                alert.description,
            ]
            
            if alert.metric_name:
                body_parts.extend([
                    "",
                    f"Metric: {alert.metric_name}",
                    f"Value: {alert.metric_value}",
                    f"Threshold: {alert.threshold}"
                ])
            
            if alert.remediation:
                body_parts.extend([
                    "",
                    "Recommended Actions:",
                    *[f"- {action}" for action in alert.remediation]
                ])
            
            if alert.related_alerts:
                body_parts.extend([
                    "",
                    f"Related Alerts: {', '.join(alert.related_alerts)}"
                ])
            
            body = "\n".join(body_parts)
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            if self.config.smtp_username and self.config.smtp_password:
                await aiosmtplib.send(
                    msg,
                    hostname=self.config.smtp_host,
                    port=self.config.smtp_port,
                    use_tls=self.config.smtp_use_tls,
                    username=self.config.smtp_username,
                    password=self.config.smtp_password
                )
            else:
                # Fallback to standard SMTP without auth
                server = smtplib.SMTP(self.config.smtp_host, self.config.smtp_port)
                if self.config.smtp_use_tls:
                    server.starttls()
                server.send_message(msg)
                server.quit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
            return False
    
    async def _send_webhook(self, destination: AlertDestination, alert: Alert) -> bool:
        """Send alert via webhook"""
        try:
            webhook_url = destination.config.get('url', '')
            if not webhook_url:
                self.logger.error("No webhook URL configured")
                return False
            
            # Prepare payload
            payload = {
                'alert_id': alert.id,
                'timestamp': alert.timestamp,
                'severity': alert.severity.value,
                'title': alert.title,
                'description': alert.description,
                'source': alert.source,
                'metric_name': alert.metric_name,
                'metric_value': alert.metric_value,
                'threshold': alert.threshold,
                'tags': alert.tags,
                'metadata': alert.metadata,
                'context': alert.context,
                'remediation': alert.remediation,
                'related_alerts': alert.related_alerts,
                'correlation_id': alert.correlation_id
            }
            
            # Send webhook
            timeout = aiohttp.ClientTimeout(total=self.config.webhook_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    webhook_url,
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        return True
                    else:
                        self.logger.error(
                            f"Webhook returned status {response.status}: "
                            f"{await response.text()}"
                        )
                        return False
        
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")
            return False
    
    async def _send_slack(self, destination: AlertDestination, alert: Alert) -> bool:
        """Send alert via Slack"""
        if not SLACK_AVAILABLE or not self.slack_client:
            self.logger.error("Slack not available or not configured")
            return False
        
        try:
            channel = destination.config.get('channel', self.config.slack_default_channel)
            
            # Create Slack message
            color_map = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.EMERGENCY: "danger"
            }
            
            attachment = {
                "color": color_map.get(alert.severity, "warning"),
                "title": alert.title,
                "text": alert.description,
                "fields": [
                    {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                    {"title": "Source", "value": alert.source, "short": True},
                    {"title": "Time", "value": time.strftime('%Y-%m-%d %H:%M:%S UTC', 
                                                          time.gmtime(alert.timestamp)), "short": True}
                ]
            }
            
            if alert.metric_name:
                attachment["fields"].extend([
                    {"title": "Metric", "value": alert.metric_name, "short": True},
                    {"title": "Value", "value": str(alert.metric_value), "short": True},
                    {"title": "Threshold", "value": str(alert.threshold), "short": True}
                ])
            
            if alert.remediation:
                remediation_text = "\n".join([f"• {action}" for action in alert.remediation])
                attachment["fields"].append({
                    "title": "Recommended Actions",
                    "value": remediation_text,
                    "short": False
                })
            
            # Send message
            response = await self.slack_client.chat_postMessage(
                channel=channel,
                text=f"Alert: {alert.title}",
                attachments=[attachment]
            )
            
            return response["ok"]
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    async def _send_to_destination(self, destination: AlertDestination, 
                                  alert: Alert) -> bool:
        """Send alert to specific destination"""
        try:
            if destination.channel == AlertChannel.EMAIL:
                return await self._send_email(destination, alert)
            elif destination.channel == AlertChannel.WEBHOOK:
                return await self._send_webhook(destination, alert)
            elif destination.channel == AlertChannel.SLACK:
                return await self._send_slack(destination, alert)
            elif destination.channel == AlertChannel.DASHBOARD:
                # Dashboard alerts are handled separately
                return True
            else:
                self.logger.error(f"Unsupported channel: {destination.channel}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error sending to {destination.name}: {e}")
            return False
    
    async def send_alert(self, alert: Alert) -> Dict[str, Any]:
        """
        Send alert through appropriate channels
        
        Returns:
            Dictionary with sending results
        """
        with self._lock:
            # Generate ID if not set
            if not alert.id:
                alert.id = self._generate_alert_id()
            
            # Add to history
            self.alert_history.append(alert)
            
            # Apply correlation rules
            for rule in self._find_correlations(alert):
                correlation_id = self._apply_correlation(alert, rule)
                if correlation_id and alert.status == AlertStatus.SUPPRESSED:
                    self.logger.info(f"Alert {alert.id} suppressed due to correlation")
                    return {"alert_id": alert.id, "status": "suppressed", "results": {}}
            
            # Enrich alert
            alert = self._enrich_alert(alert)
            
            # Filter destinations
            destinations = self._filter_destinations(alert)
            
            if not destinations:
                self.logger.warning(f"No eligible destinations for alert {alert.id}")
                return {"alert_id": alert.id, "status": "no_destinations", "results": {}}
        
        # Send to destinations (outside lock to avoid blocking)
        results = {}
        tasks = []
        
        for destination in destinations:
            task = asyncio.create_task(
                self._send_to_destination(destination, alert)
            )
            tasks.append((destination.name, task))
        
        # Wait for all sends to complete
        for dest_name, task in tasks:
            try:
                success = await task
                results[dest_name] = "sent" if success else "failed"
            except Exception as e:
                results[dest_name] = f"error: {e}"
                self.logger.error(f"Error sending to {dest_name}: {e}")
        
        # Update alert status
        if any(result == "sent" for result in results.values()):
            alert.status = AlertStatus.SENT
            overall_status = "sent"
        else:
            alert.status = AlertStatus.FAILED
            overall_status = "failed"
        
        return {
            "alert_id": alert.id,
            "status": overall_status,
            "results": results,
            "destinations_attempted": len(destinations)
        }
    
    def create_alert(self, title: str, description: str, source: str,
                    severity: AlertSeverity = AlertSeverity.WARNING,
                    metric_name: Optional[str] = None,
                    metric_value: Optional[float] = None,
                    threshold: Optional[float] = None,
                    tags: Optional[List[str]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> Alert:
        """Create a new alert"""
        return Alert(
            id=self._generate_alert_id(),
            timestamp=time.time(),
            severity=severity,
            title=title,
            description=description,
            source=source,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            tags=tags or [],
            metadata=metadata or {}
        )
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent alerts"""
        cutoff_time = time.time() - (hours * 3600)
        
        recent_alerts = [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff_time
        ]
        
        # Count by severity
        severity_counts = defaultdict(int)
        source_counts = defaultdict(int)
        status_counts = defaultdict(int)
        
        for alert in recent_alerts:
            severity_counts[alert.severity.value] += 1
            source_counts[alert.source] += 1
            status_counts[alert.status.value] += 1
        
        return {
            "time_range_hours": hours,
            "total_alerts": len(recent_alerts),
            "by_severity": dict(severity_counts),
            "by_source": dict(source_counts),
            "by_status": dict(status_counts),
            "active_correlations": len(self.active_correlations),
            "configured_destinations": len(self.destinations),
            "enabled_destinations": len([d for d in self.destinations.values() if d.enabled])
        }


# Factory functions
def create_alerting_system(config: Optional[AlertingConfig] = None) -> AlertingSystem:
    """Create an alerting system with optional configuration"""
    return AlertingSystem(config)


def create_email_destination(name: str, email: str, 
                           severity: AlertSeverity = AlertSeverity.WARNING) -> AlertDestination:
    """Create an email alert destination"""
    return AlertDestination(
        name=name,
        channel=AlertChannel.EMAIL,
        config={"email": email},
        severity_threshold=severity
    )


def create_webhook_destination(name: str, url: str,
                             severity: AlertSeverity = AlertSeverity.WARNING) -> AlertDestination:
    """Create a webhook alert destination"""
    return AlertDestination(
        name=name,
        channel=AlertChannel.WEBHOOK,
        config={"url": url},
        severity_threshold=severity
    )


def create_slack_destination(name: str, channel: str,
                           severity: AlertSeverity = AlertSeverity.WARNING) -> AlertDestination:
    """Create a Slack alert destination"""
    return AlertDestination(
        name=name,
        channel=AlertChannel.SLACK,
        config={"channel": channel},
        severity_threshold=severity
    )


# Global alerting system instance
_global_alerting_system: Optional[AlertingSystem] = None


def get_alerting_system() -> AlertingSystem:
    """Get global alerting system instance"""
    global _global_alerting_system
    if _global_alerting_system is None:
        _global_alerting_system = create_alerting_system()
    return _global_alerting_system


def set_alerting_system(system: AlertingSystem):
    """Set global alerting system instance"""
    global _global_alerting_system
    _global_alerting_system = system 