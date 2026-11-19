"""Real Prometheus listeners release their ports without unregistering metrics."""

from urllib.request import urlopen

import pytest

from stateset_agents.utils import monitoring

pytestmark = pytest.mark.integration


def test_listener_shutdown_port_reuse_and_shared_metric_registry(monkeypatch):
    client = pytest.importorskip("prometheus_client")
    monkeypatch.setattr(monitoring, "HAS_PSUTIL", False)
    monkeypatch.setattr(
        monitoring,
        "start_http_server",
        lambda port: client.start_http_server(port, addr="127.0.0.1"),
    )
    first = monitoring.MonitoringService(prometheus_port=0)
    second = replacement = None
    try:
        assert first.get_status()["prometheus"]["server_owned"]
        port = first._prometheus_server.server_port
        first_thread = first._prometheus_thread
        with first.track_request("GET", "/owned-monitor"):
            pass
        assert (
            first.metrics_collector.get_metrics_summary()["application_metrics"][
                "requests_total"
            ]
            == 1
        )
        second = monitoring.MonitoringService(prometheus_port=0)
        shared_counter = second.metrics_collector.prometheus_metrics["requests_total"]
        assert (
            first.metrics_collector.prometheus_metrics["requests_total"]
            is shared_counter
        )
        first.close()
        first.close()
        assert not first_thread.is_alive()
        assert not first.get_status()["prometheus"]["server_owned"]
        with second.track_request("GET", "/owned-monitor"):
            pass
        second_port = second._prometheus_server.server_port
        with urlopen(f"http://127.0.0.1:{second_port}/metrics", timeout=2) as response:
            text = response.read().decode()
        assert (
            'grpo_requests_total{endpoint="/owned-monitor",method="GET",status="success"} 2.0'
            in text
        )
        replacement = monitoring.MonitoringService(prometheus_port=port)
        assert replacement.get_status()["prometheus"]["server_owned"]
        assert replacement.get_status()["prometheus"]["error_type"] is None
    finally:
        first.close()
        if second is not None:
            second.close()
        if replacement is not None:
            replacement.close()


def test_global_replacement_reuses_listener_port(monkeypatch):
    client = pytest.importorskip("prometheus_client")
    monkeypatch.setattr(monitoring, "HAS_PSUTIL", False)
    monkeypatch.setattr(monitoring, "_global_monitoring", None)
    monkeypatch.setattr(monitoring, "_global_creation_error", None)
    monkeypatch.setattr(monitoring, "_global_initializing", False)
    monkeypatch.setattr(monitoring, "_global_replacing", False)
    monkeypatch.setattr(
        monitoring,
        "start_http_server",
        lambda port: client.start_http_server(port, addr="127.0.0.1"),
    )
    first = monitoring.setup_monitoring(prometheus_port=0)
    replacement = None
    try:
        port = first._prometheus_server.server_port
        replacement = monitoring.setup_monitoring(prometheus_port=port)
        assert first.get_status()["state"] == "closed"
        assert replacement.get_status()["prometheus"]["server_owned"]
        assert replacement.get_status()["prometheus"]["error_type"] is None
        assert monitoring.get_monitoring_service() is replacement
        with urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2) as response:
            assert response.status == 200
    finally:
        first.close()
        if replacement is not None:
            replacement.close()
