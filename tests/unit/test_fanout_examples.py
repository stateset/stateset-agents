"""Shipped example manifests stay valid, and the free one really runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from stateset_agents.fanout.local_backend import LocalBackend
from stateset_agents.fanout.manifest import FanoutManifest
from stateset_agents.fanout.orchestrator import FanoutRunner

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "fanout"
MANIFESTS = sorted(EXAMPLES.glob("*.yaml"))


def test_examples_exist() -> None:
    assert MANIFESTS


@pytest.mark.parametrize("path", MANIFESTS, ids=lambda path: path.stem)
def test_example_validates_and_expands(path: Path) -> None:
    assert FanoutManifest.from_file(path).expand()


def test_sandbox_example_passes_sandbox_preflight() -> None:
    from stateset_agents.fanout.sandbox_backend import SandboxBackend

    manifest = FanoutManifest.from_file(EXAMPLES / "sandbox_sweep.yaml")

    SandboxBackend(
        api_key="k", base_url="http://localhost:1", client=object()
    ).validate(manifest)
    assert manifest.worst_case_cost_usd() == pytest.approx(3.00)


def test_command_example_runs_end_to_end(tmp_path: Path) -> None:
    manifest = FanoutManifest.from_file(EXAMPLES / "command_sweep.yaml")
    run_dir = tmp_path / "run"

    summary = FanoutRunner(
        manifest,
        LocalBackend(run_dir / "scratch"),
        run_dir,
        poll_interval_s=0.02,
        forwarded_env={},
    ).run()

    assert summary.ok, summary
    assert summary.total == 6
    assert summary.report["verdict"]["status"] == "resolved"
