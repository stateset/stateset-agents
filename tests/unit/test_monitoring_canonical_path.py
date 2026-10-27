"""One canonical monitoring path: ``stateset_agents.utils.monitoring``.

``utils.alerts``, ``utils.observability``, ``utils.logging`` and
``utils.profiler`` had no in-package users and duplicated the canonical
alerting / tracing / logging / profiling paths; they are deprecated. The
time-series classes behind ``AdvancedMonitoringService`` were renamed so they
no longer share the canonical ``AlertManager`` / ``MetricsCollector`` names.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from stateset_agents.utils.monitoring import (
    Alert,
    AlertManager,
    AlertSeverity,
    MetricsCollector,
)

DEPRECATED_UTILS_MODULES = (
    "stateset_agents.utils.alerts",
    "stateset_agents.utils.observability",
    "stateset_agents.utils.logging",
    "stateset_agents.utils.profiler",
)


def _high_cpu(metrics: dict) -> bool:
    return metrics.get("cpu", 0) > 90


def test_alert_lifecycle_creates_dedupes_notifies_and_resolves() -> None:
    manager = AlertManager()
    notified: list[Alert] = []
    manager.add_notification_handler(notified.append)
    manager.add_alert_rule(
        "high_cpu", _high_cpu, AlertSeverity.HIGH, "CPU above 90%", {"team": "ops"}
    )

    manager.check_alerts({"cpu": 95})
    manager.check_alerts({"cpu": 97})  # still firing: no duplicate alert
    active = manager.get_active_alerts()
    assert [alert.name for alert in active] == ["high_cpu"]
    assert active[0].labels == {"team": "ops"}
    assert len(notified) == 1

    manager.check_alerts({"cpu": 10})
    assert manager.get_active_alerts() == []
    resolved = list(manager.alerts)[0]
    assert resolved.resolved and resolved.resolved_at is not None

    summary = manager.get_alert_summary()
    assert summary["total_alerts"] == 1
    assert summary["active_alerts"] == 0


def test_alert_rule_errors_do_not_stop_other_rules() -> None:
    manager = AlertManager()
    manager.add_alert_rule(
        "broken", lambda m: m["missing"] > 1, AlertSeverity.LOW, "never"
    )
    manager.add_alert_rule("high_cpu", _high_cpu, AlertSeverity.HIGH, "CPU")
    manager.check_alerts({"cpu": 99})
    assert [alert.name for alert in manager.get_active_alerts()] == ["high_cpu"]


def test_metrics_collector_records_typed_metrics() -> None:
    collector = MetricsCollector(enable_prometheus=False)
    collector.increment_counter("requests", 2.0, {"route": "/v1/messages"})
    collector.set_gauge("queue_depth", 3.0)
    collector.observe_histogram("latency", 0.25)

    assert [m.value for m in collector.metrics["requests"]] == [2.0]
    assert collector.metrics["requests"][0].labels == {"route": "/v1/messages"}
    assert collector.metrics["queue_depth"][0].type.value == "gauge"
    assert collector.metrics["latency"][0].type.value == "histogram"
    assert "custom_metrics" in collector.get_metrics_summary()


@pytest.mark.parametrize("module_name", DEPRECATED_UTILS_MODULES)
def test_unreferenced_utils_modules_warn_with_removal_version(
    module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    with pytest.warns(DeprecationWarning, match=r"removed in v0\.71\.0"):
        importlib.import_module(module_name)


@pytest.mark.parametrize(
    "name", ["AlertManager", "ObservabilityManager", "get_logger", "PerformanceReport"]
)
def test_utils_package_serves_deprecated_names_with_a_warning(name: str) -> None:
    import stateset_agents.utils as utils

    assert name not in utils.__all__
    with pytest.warns(DeprecationWarning, match=rf"utils\.{name}.*v0\.71\.0"):
        assert getattr(utils, name) is not None


def test_utils_alert_manager_is_not_the_canonical_one() -> None:
    """The deprecated alias keeps its old behaviour until removal."""
    import stateset_agents.utils as utils

    with pytest.warns(DeprecationWarning):
        legacy = utils.AlertManager
    assert legacy is not AlertManager
    assert legacy.__module__ == "stateset_agents.utils.alerts"


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("AlertManager", "ThresholdAlertManager"),
        ("MetricsCollector", "TimeSeriesMetricsCollector"),
    ],
)
def test_advanced_monitoring_old_names_are_deprecated_aliases(
    old: str, new: str
) -> None:
    from stateset_agents.core import advanced_monitoring

    assert old not in advanced_monitoring.__all__
    with pytest.warns(DeprecationWarning, match=rf"{old}.*removed in v0\.71\.0"):
        alias = getattr(advanced_monitoring, old)
    assert alias is getattr(advanced_monitoring, new)
    assert alias.__module__ == "stateset_agents.core.advanced_monitoring"


def test_one_public_alert_manager_and_metrics_collector_outside_deprecated() -> None:
    """Outside modules scheduled for removal, each name has one definition."""
    import ast
    from pathlib import Path

    package = Path(__file__).resolve().parents[2] / "stateset_agents"
    deprecated = {
        package / "utils" / "alerts.py",
        package / "utils" / "observability.py",
        package / "utils" / "logging.py",
        package / "utils" / "profiler.py",
    }
    definitions: dict[str, list[str]] = {"AlertManager": [], "MetricsCollector": []}
    for path in package.rglob("*.py"):
        if path in deprecated:
            continue
        text = path.read_text(encoding="utf-8")
        if "class AlertManager" not in text and "class MetricsCollector" not in text:
            continue
        for node in ast.parse(text).body:
            if isinstance(node, ast.ClassDef) and node.name in definitions:
                definitions[node.name].append(path.relative_to(package).as_posix())
    assert definitions == {
        "AlertManager": ["utils/monitoring.py"],
        "MetricsCollector": ["utils/monitoring.py"],
    }
