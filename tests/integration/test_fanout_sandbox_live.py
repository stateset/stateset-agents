"""Fan-out against a real sandbox controller. Opt-in: this allocates real compute.

    export STATESET_FANOUT_LIVE=1
    export STATESET_API_KEY=...
    export STATESET_BASE_URL=http://localhost:8080   # hosted API by default
    export STATESET_AGENTS_WHEEL=dist/stateset_agents-X.Y.Z-py3-none-any.whl
    pytest tests/integration/test_fanout_sandbox_live.py -n 0 -m integration

Every sandbox is capped at a few cents and stopped afterwards. The learning test
also needs ``STATESET_FANOUT_LIVE_RL=1``: it installs torch and transformers in each
sandbox, so the org's egress policy must allow PyPI and the PyTorch index.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.integration

_MISSING = [
    name
    for name in ("STATESET_FANOUT_LIVE", "STATESET_API_KEY", "STATESET_AGENTS_WHEEL")
    if not os.environ.get(name)
]
if _MISSING:
    pytest.skip(
        f"live sandbox test needs: {', '.join(_MISSING)}", allow_module_level=True
    )

pytest.importorskip("stateset_sandbox")

from stateset_agents.fanout.manifest import FanoutManifest  # noqa: E402
from stateset_agents.fanout.orchestrator import FanoutRunner, load_records  # noqa: E402
from stateset_agents.fanout.sandbox_backend import SandboxBackend  # noqa: E402

WRITE_METRICS = (
    "import json, pathlib, sys; "
    "print('hello from seed', sys.argv[1]); "
    "pathlib.Path(sys.argv[2], 'metrics.json').write_text(json.dumps("
    "{'reward': int(sys.argv[1]) / 10}))"
)


def _wheel_name(tmp_path: Path) -> str:
    source = Path(os.environ["STATESET_AGENTS_WHEEL"]).resolve()
    shutil.copy(source, tmp_path / source.name)
    return source.name


def _run(manifest_data: dict[str, Any], tmp_path: Path) -> Any:
    manifest = FanoutManifest.from_dict(manifest_data, base_dir=tmp_path)
    run_dir = tmp_path / "run"
    summary = FanoutRunner(
        manifest,
        SandboxBackend(),
        run_dir,
        poll_interval_s=2.0,
        forwarded_env={},
        ledger_path=tmp_path / "ledger.jsonl",
    ).run()
    assert not summary.unreleased, f"still running: {summary.unreleased}"
    assert summary.ok, [
        (record.run_id, record.status, record.error) for record in load_records(run_dir)
    ]
    return summary


def test_parallel_runs_on_real_sandboxes(tmp_path: Path) -> None:
    summary = _run(
        {
            "schema_version": 1,
            "name": "live-smoke",
            "seeds": [1, 2],
            "workload": {
                "kind": "command",
                "argv": ["{python}", "-c", WRITE_METRICS, "{seed}", "{output_dir}"],
            },
            "report": {"metric": "reward", "mode": "max"},
            "resources": {
                "cpus": "1",
                "memory": "1Gi",
                "timeout_seconds": 300,
                "max_cost_cents": 5,
            },
            "package": {
                "source": "wheel",
                "wheel": _wheel_name(tmp_path),
                "install_deps": False,
            },
            "concurrency": 2,
            "max_attempts": 1,
        },
        tmp_path,
    )

    assert summary.report["arms"][0]["mean"] == pytest.approx(0.15)


@pytest.mark.skipif(
    not os.environ.get("STATESET_FANOUT_LIVE_RL"),
    reason="set STATESET_FANOUT_LIVE_RL=1 to install torch in real sandboxes",
)
def test_real_policy_updates_on_real_sandboxes(tmp_path: Path) -> None:
    summary = _run(
        {
            "schema_version": 1,
            "name": "live-rl",
            "seeds": [42, 43],
            "workload": {"kind": "workflow_regression", "updates": 10, "device": "cpu"},
            "report": {"metric": "improvement", "mode": "max"},
            "resources": {
                "cpus": "2",
                "memory": "4Gi",
                "ephemeral_storage": "20Gi",
                "timeout_seconds": 1800,
                "max_cost_cents": 50,
            },
            "package": {
                "source": "wheel",
                "wheel": _wheel_name(tmp_path),
                "extras": ["training"],
            },
            "concurrency": 2,
            "max_attempts": 1,
        },
        tmp_path,
    )

    for result in summary.report["run_results"]:
        assert result["metrics"]["optimizer_updates"] == 10
