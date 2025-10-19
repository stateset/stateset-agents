"""
Advanced Alerting System for GRPO Agent Framework

This module provides sophisticated alerting capabilities including custom alert rules,
notification channels, alert aggregation, and integration with external systems.
"""

import asyncio
import json
import smtplib
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import requests

# Optional imports
try:
    import aiohttp

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    import slack_sdk

    HAS_SLACK = True
except ImportError:
    HAS_SLACK = False


class AlertSeverity(Enum):
    """Alert severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status"""

    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"
    SILENCED = "silenced"


class NotificationChannel(Enum):
    """Notification channel types"""

    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"
    DISCORD = "discord"


@dataclass
class AlertRule:
    """Alert rule definition"""

    name: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: AlertSeverity
    message: str
    labels: Dict[str, str] = field(default_factory=dict)
    cooldown_minutes: int = 15
    repeat_interval_minutes: int = 60
    auto_resolve: bool = True
    resolve_condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    enabled: bool = True

    def __post_init__(self):
        if self.resolve_condition is None:
            # Default resolve condition is the inverse of the alert condition
            self.resolve_condition = lambda metrics: not self.condition(metrics)


@dataclass
class Alert:
    """Alert instance"""

    id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    labels: Dict[str, str] = field(default_factory=dict)
    status: AlertStatus = AlertStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    fired_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    last_notification_at: Optional[datetime] = None
    notification_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            "id": self.id,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "message": self.message,
            "labels": self.labels,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "fired_at": self.fired_at.isoformat() if self.fired_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "last_notification_at": self.last_notification_at.isoformat()
            if self.last_notification_at
            else None,
            "notification_count": self.notification_count,
            "metadata": self.metadata,
        }

    def age_minutes(self) -> float:
        """Get alert age in minutes"""
        return (datetime.now() - self.created_at).total_seconds() / 60


@dataclass
class NotificationConfig:
    """Notification configuration"""

    channel: NotificationChannel
    config: Dict[str, Any]
    enabled: bool = True
    severity_filter: List[AlertSeverity] = field(
        default_factory=lambda: list(AlertSeverity)
    )
    label_filters: Dict[str, str] = field(default_factory=dict)


class NotificationHandler:
    """Base notification handler"""

    def __init__(self, config: NotificationConfig):
        self.config = config

    async def send_notification(self, alert: Alert) -> bool:
        """Send notification for alert"""
        raise NotImplementedError

    def should_notify(self, alert: Alert) -> bool:
        """Check if notification should be sent"""
        if not self.config.enabled:
            return False

        if alert.severity not in self.config.severity_filter:
            return False

        # Check label filters
        for key, value in self.config.label_filters.items():
            if key not in alert.labels or alert.labels[key] != value:
                return False

        return True


class EmailNotificationHandler(NotificationHandler):
    """Email notification handler"""

    async def send_notification(self, alert: Alert) -> bool:
        """Send email notification"""
        try:
            smtp_server = self.config.config.get("smtp_server", "localhost")
            smtp_port = self.config.config.get("smtp_port", 587)
            username = self.config.config.get("username")
            password = self.config.config.get("password")
            from_email = self.config.config.get("from_email")
            to_emails = self.config.config.get("to_emails", [])

            if not to_emails:
                return False

            # Create message
            msg = MIMEMultipart()
            msg["From"] = from_email
            msg["To"] = ", ".join(to_emails)
            msg["Subject"] = f"[{alert.severity.value.upper()}] {alert.rule_name}"

            # Email body
            body = f"""
Alert: {alert.rule_name}
Severity: {alert.severity.value.upper()}
Status: {alert.status.value.upper()}
Created: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}

Message:
{alert.message}

Labels:
{json.dumps(alert.labels, indent=2)}

Alert ID: {alert.id}
"""

            msg.attach(MIMEText(body, "plain"))

            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            if username and password:
                server.login(username, password)

            server.send_message(msg)
            server.quit()

            return True

        except Exception as e:
            print(f"Failed to send email notification: {e}")
            return False


class SlackNotificationHandler(NotificationHandler):
    """Slack notification handler"""

    async def send_notification(self, alert: Alert) -> bool:
        """Send Slack notification"""
        if not HAS_SLACK:
            print("Slack SDK not available")
            return False

        try:
            token = self.config.config.get("token")
            channel = self.config.config.get("channel")

            if not token or not channel:
                return False

            client = slack_sdk.WebClient(token=token)

            # Create message
            color = {
                AlertSeverity.LOW: "good",
                AlertSeverity.MEDIUM: "warning",
                AlertSeverity.HIGH: "danger",
                AlertSeverity.CRITICAL: "danger",
            }.get(alert.severity, "warning")

            attachment = {
                "color": color,
                "title": f"{alert.rule_name}",
                "text": alert.message,
                "fields": [
                    {
                        "title": "Severity",
                        "value": alert.severity.value.upper(),
                        "short": True,
                    },
                    {
                        "title": "Status",
                        "value": alert.status.value.upper(),
                        "short": True,
                    },
                    {
                        "title": "Created",
                        "value": alert.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                        "short": True,
                    },
                    {"title": "Alert ID", "value": alert.id, "short": True},
                ],
                "footer": "GRPO Agent Framework",
                "ts": int(alert.created_at.timestamp()),
            }

            # Add labels as fields
            for key, value in alert.labels.items():
                attachment["fields"].append(
                    {"title": key.title(), "value": value, "short": True}
                )

            response = client.chat_postMessage(
                channel=channel, attachments=[attachment]
            )

            return response["ok"]

        except Exception as e:
            print(f"Failed to send Slack notification: {e}")
            return False


class WebhookNotificationHandler(NotificationHandler):
    """Webhook notification handler"""

    async def send_notification(self, alert: Alert) -> bool:
        """Send webhook notification"""
        try:
            url = self.config.config.get("url")
            if not url:
                return False

            payload = {
                "alert": alert.to_dict(),
                "timestamp": datetime.now().isoformat(),
                "source": "stateset_agents",
            }

            headers = self.config.config.get("headers", {})
            headers.setdefault("Content-Type", "application/json")

            if HAS_AIOHTTP:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url, json=payload, headers=headers
                    ) as response:
                        return response.status < 400
            else:
                response = requests.post(url, json=payload, headers=headers, timeout=10)
                return response.status_code < 400

        except Exception as e:
            print(f"Failed to send webhook notification: {e}")
            return False


class PagerDutyNotificationHandler(NotificationHandler):
    """PagerDuty notification handler"""

    async def send_notification(self, alert: Alert) -> bool:
        """Send PagerDuty notification"""
        try:
            routing_key = self.config.config.get("routing_key")
            if not routing_key:
                return False

            # PagerDuty Events API v2
            url = "https://events.pagerduty.com/v2/enqueue"

            payload = {
                "routing_key": routing_key,
                "event_action": "trigger"
                if alert.status == AlertStatus.FIRING
                else "resolve",
                "dedup_key": f"grpo_alert_{alert.rule_name}",
                "payload": {
                    "summary": f"{alert.rule_name}: {alert.message}",
                    "severity": alert.severity.value,
                    "source": "GRPO Agent Framework",
                    "component": alert.labels.get("component", "unknown"),
                    "group": alert.labels.get("service", "grpo"),
                    "class": alert.rule_name,
                    "custom_details": {
                        "alert_id": alert.id,
                        "labels": alert.labels,
                        "metadata": alert.metadata,
                    },
                },
            }

            headers = {"Content-Type": "application/json"}

            if HAS_AIOHTTP:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url, json=payload, headers=headers
                    ) as response:
                        return response.status < 400
            else:
                response = requests.post(url, json=payload, headers=headers, timeout=10)
                return response.status_code < 400

        except Exception as e:
            print(f"Failed to send PagerDuty notification: {e}")
            return False


class AlertManager:
    """Advanced alert manager"""

    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.notification_handlers: List[NotificationHandler] = []
        self.silenced_rules: Dict[str, datetime] = {}
        self.statistics = {
            "total_alerts": 0,
            "alerts_by_severity": defaultdict(int),
            "alerts_by_rule": defaultdict(int),
            "notifications_sent": 0,
            "notifications_failed": 0,
        }

    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.alert_rules[rule.name] = rule

    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule"""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]

    def add_notification_handler(self, handler: NotificationHandler):
        """Add a notification handler"""
        self.notification_handlers.append(handler)

    def silence_rule(self, rule_name: str, duration_minutes: int = 60):
        """Silence an alert rule for a duration"""
        silence_until = datetime.now() + timedelta(minutes=duration_minutes)
        self.silenced_rules[rule_name] = silence_until

    def unsilence_rule(self, rule_name: str):
        """Remove silence from an alert rule"""
        if rule_name in self.silenced_rules:
            del self.silenced_rules[rule_name]

    def is_rule_silenced(self, rule_name: str) -> bool:
        """Check if a rule is silenced"""
        if rule_name not in self.silenced_rules:
            return False

        silence_until = self.silenced_rules[rule_name]
        if datetime.now() > silence_until:
            del self.silenced_rules[rule_name]
            return False

        return True

    async def evaluate_rules(self, metrics: Dict[str, Any]):
        """Evaluate all alert rules against metrics"""
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue

            if self.is_rule_silenced(rule_name):
                continue

            try:
                # Check if alert condition is met
                if rule.condition(metrics):
                    await self._handle_alert_trigger(rule, metrics)
                else:
                    await self._handle_alert_resolve(rule, metrics)

            except Exception as e:
                print(f"Error evaluating alert rule {rule_name}: {e}")

    async def _handle_alert_trigger(self, rule: AlertRule, metrics: Dict[str, Any]):
        """Handle alert trigger"""
        existing_alert = self.active_alerts.get(rule.name)

        if existing_alert:
            # Check if we should send repeat notification
            if self._should_repeat_notification(existing_alert, rule):
                await self._send_notifications(existing_alert)
        else:
            # Create new alert
            alert = Alert(
                id=str(uuid.uuid4()),
                rule_name=rule.name,
                severity=rule.severity,
                message=rule.message,
                labels=rule.labels.copy(),
                status=AlertStatus.FIRING,
                fired_at=datetime.now(),
                metadata={"triggered_by_metrics": metrics},
            )

            self.active_alerts[rule.name] = alert
            self.alert_history.append(alert)

            # Update statistics
            self.statistics["total_alerts"] += 1
            self.statistics["alerts_by_severity"][rule.severity.value] += 1
            self.statistics["alerts_by_rule"][rule.name] += 1

            # Send notifications
            await self._send_notifications(alert)

    async def _handle_alert_resolve(self, rule: AlertRule, metrics: Dict[str, Any]):
        """Handle alert resolution"""
        if not rule.auto_resolve:
            return

        existing_alert = self.active_alerts.get(rule.name)
        if existing_alert and existing_alert.status == AlertStatus.FIRING:
            # Check resolve condition
            if rule.resolve_condition and rule.resolve_condition(metrics):
                existing_alert.status = AlertStatus.RESOLVED
                existing_alert.resolved_at = datetime.now()
                existing_alert.metadata["resolved_by_metrics"] = metrics

                # Send resolution notification
                await self._send_notifications(existing_alert)

                # Remove from active alerts
                del self.active_alerts[rule.name]

    def _should_repeat_notification(self, alert: Alert, rule: AlertRule) -> bool:
        """Check if notification should be repeated"""
        if alert.last_notification_at is None:
            return True

        minutes_since_last = (
            datetime.now() - alert.last_notification_at
        ).total_seconds() / 60
        return minutes_since_last >= rule.repeat_interval_minutes

    async def _send_notifications(self, alert: Alert):
        """Send notifications for an alert"""
        for handler in self.notification_handlers:
            if handler.should_notify(alert):
                try:
                    success = await handler.send_notification(alert)
                    if success:
                        self.statistics["notifications_sent"] += 1
                    else:
                        self.statistics["notifications_failed"] += 1
                except Exception as e:
                    print(f"Notification handler error: {e}")
                    self.statistics["notifications_failed"] += 1

        # Update notification tracking
        alert.last_notification_at = datetime.now()
        alert.notification_count += 1

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())

    def get_alert_by_id(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID"""
        for alert in self.active_alerts.values():
            if alert.id == alert_id:
                return alert

        for alert in self.alert_history:
            if alert.id == alert_id:
                return alert

        return None

    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts by severity"""
        return [
            alert for alert in self.active_alerts.values() if alert.severity == severity
        ]

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        active_alerts = self.get_active_alerts()

        return {
            "total_alerts_ever": self.statistics["total_alerts"],
            "active_alerts": len(active_alerts),
            "alerts_by_severity": dict(self.statistics["alerts_by_severity"]),
            "alerts_by_rule": dict(self.statistics["alerts_by_rule"]),
            "notifications_sent": self.statistics["notifications_sent"],
            "notifications_failed": self.statistics["notifications_failed"],
            "active_by_severity": {
                severity.value: len(
                    [a for a in active_alerts if a.severity == severity]
                )
                for severity in AlertSeverity
            },
            "silenced_rules": len(self.silenced_rules),
            "total_rules": len(self.alert_rules),
            "enabled_rules": len([r for r in self.alert_rules.values() if r.enabled]),
        }

    def export_alert_config(self) -> Dict[str, Any]:
        """Export alert configuration"""
        return {
            "alert_rules": [
                {
                    "name": rule.name,
                    "severity": rule.severity.value,
                    "message": rule.message,
                    "labels": rule.labels,
                    "cooldown_minutes": rule.cooldown_minutes,
                    "repeat_interval_minutes": rule.repeat_interval_minutes,
                    "auto_resolve": rule.auto_resolve,
                    "enabled": rule.enabled,
                }
                for rule in self.alert_rules.values()
            ],
            "notification_handlers": [
                {
                    "channel": handler.config.channel.value,
                    "enabled": handler.config.enabled,
                    "severity_filter": [
                        s.value for s in handler.config.severity_filter
                    ],
                    "label_filters": handler.config.label_filters,
                }
                for handler in self.notification_handlers
            ],
        }


def create_default_alert_rules() -> List[AlertRule]:
    """Create default alert rules for common scenarios"""
    rules = []

    # System resource alerts
    rules.append(
        AlertRule(
            name="high_cpu_usage",
            condition=lambda m: m.get("system_metrics", {})
            .get("cpu_usage", {})
            .get("current", 0)
            > 80,
            severity=AlertSeverity.MEDIUM,
            message="CPU usage is above 80%",
            labels={"category": "system", "resource": "cpu"},
        )
    )

    rules.append(
        AlertRule(
            name="high_memory_usage",
            condition=lambda m: m.get("system_metrics", {})
            .get("memory_usage", {})
            .get("current", 0)
            > 85,
            severity=AlertSeverity.HIGH,
            message="Memory usage is above 85%",
            labels={"category": "system", "resource": "memory"},
        )
    )

    rules.append(
        AlertRule(
            name="high_disk_usage",
            condition=lambda m: m.get("system_metrics", {})
            .get("disk_usage", {})
            .get("current", 0)
            > 90,
            severity=AlertSeverity.HIGH,
            message="Disk usage is above 90%",
            labels={"category": "system", "resource": "disk"},
        )
    )

    # Application alerts
    rules.append(
        AlertRule(
            name="high_error_rate",
            condition=lambda m: m.get("application_metrics", {}).get("error_count", 0)
            > 100,
            severity=AlertSeverity.HIGH,
            message="Error rate is above 100 errors",
            labels={"category": "application", "type": "errors"},
        )
    )

    rules.append(
        AlertRule(
            name="slow_response_time",
            condition=lambda m: m.get("application_metrics", {}).get(
                "average_response_time", 0
            )
            > 5.0,
            severity=AlertSeverity.MEDIUM,
            message="Average response time is above 5 seconds",
            labels={"category": "application", "type": "performance"},
        )
    )

    rules.append(
        AlertRule(
            name="low_training_reward",
            condition=lambda m: m.get("application_metrics", {}).get(
                "average_reward_score", 1.0
            )
            < 0.3,
            severity=AlertSeverity.MEDIUM,
            message="Training reward score is below 0.3",
            labels={"category": "training", "type": "performance"},
        )
    )

    # Critical system alerts
    rules.append(
        AlertRule(
            name="system_overload",
            condition=lambda m: (
                m.get("system_metrics", {}).get("cpu_usage", {}).get("current", 0) > 95
                and m.get("system_metrics", {})
                .get("memory_usage", {})
                .get("current", 0)
                > 95
            ),
            severity=AlertSeverity.CRITICAL,
            message="System is overloaded - both CPU and memory usage are above 95%",
            labels={"category": "system", "type": "critical"},
        )
    )

    return rules


def setup_email_notifications(
    smtp_server: str,
    smtp_port: int,
    username: str,
    password: str,
    from_email: str,
    to_emails: List[str],
    severity_filter: List[AlertSeverity] = None,
) -> EmailNotificationHandler:
    """Setup email notifications"""
    config = NotificationConfig(
        channel=NotificationChannel.EMAIL,
        config={
            "smtp_server": smtp_server,
            "smtp_port": smtp_port,
            "username": username,
            "password": password,
            "from_email": from_email,
            "to_emails": to_emails,
        },
        severity_filter=severity_filter or [AlertSeverity.HIGH, AlertSeverity.CRITICAL],
    )

    return EmailNotificationHandler(config)


def setup_slack_notifications(
    token: str, channel: str, severity_filter: List[AlertSeverity] = None
) -> SlackNotificationHandler:
    """Setup Slack notifications"""
    config = NotificationConfig(
        channel=NotificationChannel.SLACK,
        config={"token": token, "channel": channel},
        severity_filter=severity_filter or list(AlertSeverity),
    )

    return SlackNotificationHandler(config)


def setup_webhook_notifications(
    url: str,
    headers: Dict[str, str] = None,
    severity_filter: List[AlertSeverity] = None,
) -> WebhookNotificationHandler:
    """Setup webhook notifications"""
    config = NotificationConfig(
        channel=NotificationChannel.WEBHOOK,
        config={"url": url, "headers": headers or {}},
        severity_filter=severity_filter or list(AlertSeverity),
    )

    return WebhookNotificationHandler(config)


def setup_pagerduty_notifications(
    routing_key: str, severity_filter: List[AlertSeverity] = None
) -> PagerDutyNotificationHandler:
    """Setup PagerDuty notifications"""
    config = NotificationConfig(
        channel=NotificationChannel.PAGERDUTY,
        config={"routing_key": routing_key},
        severity_filter=severity_filter or [AlertSeverity.HIGH, AlertSeverity.CRITICAL],
    )

    return PagerDutyNotificationHandler(config)
