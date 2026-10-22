"""Guard that vLLM endpoints (served without --api-key) are network-isolated."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
K8S_DIR = ROOT / "deployment" / "kubernetes"
CHART = ROOT / "deployment" / "helm" / "stateset-agents"
VLLM_MANIFESTS = sorted(K8S_DIR.glob("*vllm*.yaml"))
GATEWAY_LABELS = {"app": "stateset-agents-api"}


def _docs(text: str) -> list[dict]:
    return [doc for doc in yaml.safe_load_all(text) if doc]


def _gateway_labels() -> dict:
    docs = _docs((K8S_DIR / "production-deployment.yaml").read_text())
    gateway = next(
        d
        for d in docs
        if d["kind"] == "Deployment" and d["metadata"]["name"] == "stateset-agents-api"
    )
    return gateway["spec"]["template"]["metadata"]["labels"]


def test_gateway_label_matches_production_deployment() -> None:
    labels = _gateway_labels()
    assert GATEWAY_LABELS.items() <= labels.items()


def test_vllm_manifests_exist() -> None:
    assert VLLM_MANIFESTS


@pytest.mark.parametrize("path", VLLM_MANIFESTS, ids=lambda p: p.name)
def test_vllm_manifest_ships_restrictive_network_policy(path: Path) -> None:
    docs = _docs(path.read_text())
    deployment = next(d for d in docs if d["kind"] == "Deployment")
    pod_labels = deployment["spec"]["template"]["metadata"]["labels"]
    policies = [d for d in docs if d["kind"] == "NetworkPolicy"]
    assert len(policies) == 1, f"{path.name} must ship one NetworkPolicy"
    spec = policies[0]["spec"]
    assert policies[0]["metadata"]["namespace"] == deployment["metadata"]["namespace"]
    assert spec["podSelector"]["matchLabels"].items() <= pod_labels.items()
    assert spec["policyTypes"] == ["Ingress"]
    (rule,) = spec["ingress"]
    assert {"podSelector": {"matchLabels": GATEWAY_LABELS}} in rule["from"]
    for peer in rule["from"]:
        # No allow-all peers (empty selectors) and no ipBlock escapes.
        assert "ipBlock" not in peer
        assert all(selector for selector in peer.values()), peer
    assert rule["ports"] == [{"protocol": "TCP", "port": 8000}]


@pytest.mark.parametrize("path", VLLM_MANIFESTS, ids=lambda p: p.name)
def test_trust_remote_code_is_documented(path: Path) -> None:
    lines = path.read_text().splitlines()
    index = next(
        i for i, line in enumerate(lines) if line.strip() == "- --trust-remote-code"
    )
    assert "Trust decision" in "\n".join(lines[max(0, index - 4) : index])


def test_chart_values_enable_network_policy_by_default() -> None:
    values = yaml.safe_load((CHART / "values.yaml").read_text())
    assert values["networkPolicy"]["enabled"] is True
    assert values["networkPolicy"]["vllm"]["extraIngressFrom"] == []


def _helm_template(*args: str) -> list[dict]:
    helm = shutil.which("helm")
    if not helm:
        pytest.skip("helm binary not available")
    result = subprocess.run(
        [helm, "template", "rel", str(CHART), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return _docs(result.stdout)


def test_chart_renders_vllm_network_policy_for_api_pods() -> None:
    docs = _helm_template()
    (policy,) = [d for d in docs if d["kind"] == "NetworkPolicy"]
    vllm = next(
        d
        for d in docs
        if d["kind"] == "Deployment" and d["metadata"]["name"] == "rel-vllm"
    )
    api = next(
        d
        for d in docs
        if d["kind"] == "Deployment" and d["metadata"]["name"] == "rel-api"
    )
    spec = policy["spec"]
    assert (
        spec["podSelector"]["matchLabels"].items()
        <= vllm["spec"]["template"]["metadata"]["labels"].items()
    )
    (rule,) = spec["ingress"]
    (peer,) = rule["from"]
    assert (
        peer["podSelector"]["matchLabels"].items()
        <= api["spec"]["template"]["metadata"]["labels"].items()
    )


def test_chart_network_policy_can_be_disabled() -> None:
    docs = _helm_template("--set", "networkPolicy.enabled=false")
    assert not [d for d in docs if d["kind"] == "NetworkPolicy"]
