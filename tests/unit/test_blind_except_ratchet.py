"""Guardrail: blind ``except Exception`` handlers in the package only go down.

ruff's ``BLE`` rules are enabled in ``pyproject.toml``. Files that still had
findings when the rule was switched on are listed below with their count and
are exempted in ``[tool.ruff.lint.per-file-ignores]``; every other package
file is linted normally, so a new blind catch anywhere else fails
``ruff check``. Inside an exempted file this test fails when the count grows.

When you narrow a handler, lower its count here (and drop the file from both
this baseline and ``per-file-ignores`` at zero) in the same commit: the
baseline must match reality, so it can only shrink.

A real process boundary (CLI top level, worker supervisor, callback or tool
isolation, best-effort cleanup) keeps ``except Exception:  # noqa: BLE001``
and says why, on that line or in a comment directly above. Handlers that log
with ``exc_info`` or re-raise are not findings (ruff's own rule).

``core/`` and the training modules owned by this workstream are at zero.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = REPO_ROOT / "stateset_agents"

#: Files exempt from ruff BLE001 because they predate the rule, with their
#: finding count. Only shrinks. (120 findings in 51 files when the rule was
#: written; this pass took core/ and its training modules to zero.)
BLE_BASELINE: dict[str, int] = {
    "stateset_agents/api/distributed_cache.py": 2,
    "stateset_agents/api/main.py": 4,
    "stateset_agents/api/middleware.py": 1,
    "stateset_agents/api/persistence.py": 2,
    "stateset_agents/api/routers/metrics.py": 1,
    "stateset_agents/api/routers/training_lab.py": 2,
    "stateset_agents/api/services/training_service.py": 1,
    "stateset_agents/api/training_worker.py": 2,
    "stateset_agents/cli.py": 2,
    "stateset_agents/cli_advanced.py": 1,
    "stateset_agents/cli_chat.py": 1,
    "stateset_agents/cli_finetune.py": 1,
    "stateset_agents/cli_research.py": 1,
    "stateset_agents/evaluation/industry.py": 1,
    "stateset_agents/evaluation/reward_audit.py": 1,
    "stateset_agents/experimental/ppo_online.py": 1,
    "stateset_agents/finetune_evaluation_tui.py": 1,
    "stateset_agents/finetune_evaluation_worker.py": 3,
    "stateset_agents/finetune_validation.py": 1,
    "stateset_agents/finetune_worker.py": 1,
    "stateset_agents/remote/chat_session.py": 4,
    "stateset_agents/remote/modal.py": 6,
    "stateset_agents/remote/river_rl_runner.py": 2,
    "stateset_agents/remote/river_runtime.py": 3,
    "stateset_agents/remote/runpod.py": 4,
    "stateset_agents/remote/serve_session.py": 5,
    "stateset_agents/rewards/nsr_verifier.py": 1,
    "stateset_agents/rewards/symbolic_physics_reward.py": 10,
    "stateset_agents/training/single_turn_trainer.py": 1,
    "stateset_agents/training/spine/loop.py": 1,
    "stateset_agents/training/spine/update.py": 2,
    "stateset_agents/tui.py": 8,
    "stateset_agents/tui_tasks.py": 1,
    "stateset_agents/utils/async_calls.py": 1,
    "stateset_agents/utils/wandb_integration.py": 1,
}

#: Suppressed BLE001 findings (noqa markers) without a stated reason, per file.
UNREASONED_NOQA_BASELINE: dict[str, int] = {
    "stateset_agents/cli_chat.py": 1,
    "stateset_agents/remote/artifacts.py": 3,
    "stateset_agents/remote/huggingface.py": 4,
    "stateset_agents/remote/prime.py": 1,
    "stateset_agents/remote/serve_session.py": 1,
    "stateset_agents/remote/together.py": 3,
}

#: Directories that must stay at zero blind excepts (and zero unreasoned
#: noqa markers): the core layer and the training modules owned by the
#: error-handling workstream.
ZERO_ZONES: tuple[str, ...] = (
    "stateset_agents/core/",
    "stateset_agents/training/auto_research/",
    "stateset_agents/training/advanced_training_orchestrator.py",
    "stateset_agents/training/base_trainer.py",
    "stateset_agents/training/ddp_control.py",
    "stateset_agents/training/experiment_tracking.py",
    "stateset_agents/training/harvest.py",
    "stateset_agents/training/serving_artifacts.py",
    "stateset_agents/training/sft.py",
    "stateset_agents/training/train.py",
    "stateset_agents/training/trainer_utils.py",
)

#: Non-library trees where blind excepts are allowed (scripts, demos, tests).
EXEMPT_TREES = ("tests", "examples", "benchmarks", "scripts", "deployment")


def _ruff_findings() -> Counter[str]:
    if shutil.which("ruff") is None:
        try:
            import ruff  # noqa: F401
        except ImportError:
            pytest.skip("ruff is not installed")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--isolated",
            "--select",
            "BLE001",
            "--output-format",
            "json",
            "--exit-zero",
            "stateset_agents",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    return Counter(
        Path(item["filename"]).resolve().relative_to(REPO_ROOT).as_posix()
        for item in json.loads(result.stdout)
    )


def _has_reason(lines: list[str], index: int) -> bool:
    after = lines[index].split("noqa: BLE001", 1)[1]
    if after.strip(" \t-—:#,"):
        return True
    previous = lines[index - 1].strip() if index else ""
    return previous.startswith("#")


def unreasoned_noqa_markers(root: Path = PACKAGE) -> Counter[str]:
    """Per file, ``# noqa: BLE001`` lines with no reason on or above them."""
    found: Counter[str] = Counter()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if "noqa: BLE001" in line and not _has_reason(lines, index):
                found[path.relative_to(root.parent).as_posix()] += 1
    return found


def _per_file_ignores() -> dict[str, list[str]]:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
    lint = config["tool"]["ruff"]["lint"]
    return dict(lint.get("per-file-ignores", {}))


def test_ruff_selects_the_blind_except_and_try_rules() -> None:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
    selected = set(config["tool"]["ruff"]["lint"]["select"])
    assert "BLE" in selected
    assert {"TRY002", "TRY201", "TRY203", "TRY401"} <= selected


def test_no_new_blind_excepts() -> None:
    findings = _ruff_findings()
    grown = {
        path: f"{count} > baseline {BLE_BASELINE.get(path, 0)}"
        for path, count in sorted(findings.items())
        if count > BLE_BASELINE.get(path, 0)
    }
    assert not grown, (
        "New blind `except Exception` handlers. Catch the specific exceptions "
        "you expect (stateset_agents/exceptions.py has canonical tuples), or "
        "for a real process boundary add `# noqa: BLE001` with the reason:\n  "
        + "\n  ".join(f"{path}: {why}" for path, why in grown.items())
    )


def test_blind_except_baseline_matches_reality() -> None:
    findings = _ruff_findings()
    stale = {
        path: f"baseline {expected}, now {findings.get(path, 0)}"
        for path, expected in sorted(BLE_BASELINE.items())
        if findings.get(path, 0) < expected
    }
    assert not stale, (
        "Blind excepts were removed; lower BLE_BASELINE (and drop files that "
        "reach zero from it and from pyproject per-file-ignores):\n  "
        + "\n  ".join(f"{path}: {why}" for path, why in stale.items())
    )


def test_lint_exemptions_are_exactly_the_baseline() -> None:
    exempt = {
        pattern
        for pattern, codes in _per_file_ignores().items()
        if pattern.startswith("stateset_agents/")
        and any(code in {"BLE", "BLE001"} for code in codes)
    }
    assert exempt == set(BLE_BASELINE), (
        "pyproject per-file-ignores must exempt from BLE001 exactly the files "
        f"in BLE_BASELINE; extra: {sorted(exempt - set(BLE_BASELINE))}, "
        f"missing: {sorted(set(BLE_BASELINE) - exempt)}"
    )
    trees = {
        pattern.split("/", 1)[0]
        for pattern, codes in _per_file_ignores().items()
        if "BLE001" in codes and not pattern.startswith("stateset_agents/")
    }
    assert trees == set(EXEMPT_TREES)


def test_zero_zones_have_no_blind_excepts() -> None:
    offenders = [
        path
        for path in [*BLE_BASELINE, *UNREASONED_NOQA_BASELINE]
        if any(path.startswith(zone) for zone in ZERO_ZONES)
    ]
    assert offenders == []


def test_noqa_markers_state_a_reason() -> None:
    found = unreasoned_noqa_markers()
    grown = {
        path: count
        for path, count in sorted(found.items())
        if count > UNREASONED_NOQA_BASELINE.get(path, 0)
    }
    assert not grown, (
        "`# noqa: BLE001` needs a reason (`# noqa: BLE001 - why`, or a "
        f"comment on the line above): {grown}"
    )
    stale = {
        path: count
        for path, count in sorted(UNREASONED_NOQA_BASELINE.items())
        if found.get(path, 0) < count
    }
    assert not stale, f"lower UNREASONED_NOQA_BASELINE for {stale}"


def test_reason_detector(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "m.py").write_text(
        "try:\n    pass\nexcept Exception:  # noqa: BLE001\n    pass\n"
        "try:\n    pass\nexcept Exception:  # noqa: BLE001 - worker boundary\n"
        "    pass\n"
        "try:\n    pass\n# Boundary: plugin isolation.\n"
        "except Exception:  # noqa: BLE001\n    pass\n",
        encoding="utf-8",
    )
    assert unreasoned_noqa_markers(package) == Counter({"pkg/m.py": 1})
