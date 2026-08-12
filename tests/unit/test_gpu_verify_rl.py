"""In-process CPU tests for the RL GPU verification job.

The live workflow runs ``python -m stateset_agents.training.gpu_verify_rl``
on a rented GPU; these tests run the same ``main()`` on CPU with the
tiniest settings so the code path is exercised in every CI run.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from stateset_agents.training import gpu_verify_rl


def _run(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, dict]:
    code = gpu_verify_rl.main(argv)
    lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(gpu_verify_rl.SUMMARY_PREFIX)
    ]
    assert lines, "summary line missing"
    summary = json.loads(lines[-1].removeprefix(gpu_verify_rl.SUMMARY_PREFIX).strip())
    return code, summary


def test_main_cpu_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    """A short run exits 0 and reports a strictly increased target prob."""
    code, summary = _run(capsys, ["--steps", "8"])
    assert code == 0
    assert summary["job"] == "gspo_gpu_verify"
    assert summary["converged"] is True
    assert summary["final_target_prob"] > summary["initial_target_prob"]
    assert summary["num_steps"] == 8
    assert summary["device"] in ("cpu", "cuda")


@pytest.mark.slow
def test_main_cpu_default_steps(capsys: pytest.CaptureFixture[str]) -> None:
    """The default (workflow) configuration also converges on CPU."""
    code, summary = _run(capsys, [])
    assert code == 0
    assert summary["converged"] is True
    assert summary["num_steps"] == 40
