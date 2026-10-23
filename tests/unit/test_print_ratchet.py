"""Guardrail: ``print()`` in the package only goes down.

ruff's ``T20`` rules (``print``/``pprint``) are enabled in ``pyproject.toml``:
library code logs through ``logging`` so an embedding application decides what
reaches the console. Files that still printed when the rule was switched on
are listed below with their count and exempted in
``[tool.ruff.lint.per-file-ignores]``; every other package file is linted
normally, so a new ``print`` anywhere else fails ``ruff check``. Inside an
exempted file this test fails when the count grows.

CLI front ends (``cli*.py``) print by design, but they are count-locked here
too; the library files in the baseline should reach zero. When you convert a
print, lower its count here (and drop the file from both this baseline and
``per-file-ignores`` at zero) in the same commit: the baseline must match
reality, so it can only shrink.

A genuine command-line output inside an otherwise-library module (a
``main()`` report line, a dry-run plan whose stdout is the contract) keeps
``print(...)  # noqa: T201`` and says why on that line or in a comment
directly above. Those markers are counted too (:data:`NOQA_CEILING`).

Counting is AST-based (no ruff subprocess): a ``print(...)``/``pprint(...)``
call whose first line carries no ``noqa: T20`` marker.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = REPO_ROOT / "stateset_agents"

#: Package files exempt from ruff T20 because they predate the rule, with
#: their unsuppressed print count. Only shrinks.
PRINT_BASELINE: dict[str, int] = {
    # CLI front ends: printing is their job, but the count is locked.
    "stateset_agents/cli.py": 7,
    "stateset_agents/cli_chat.py": 10,
    "stateset_agents/cli_evaluate.py": 5,
    "stateset_agents/cli_remote.py": 1,
    "stateset_agents/cli_research.py": 1,
    # Library files that should reach zero.
    "stateset_agents/api/grpo/service.py": 13,
    "stateset_agents/core/advanced_monitoring.py": 2,
    "stateset_agents/core/enhanced_state_management.py": 3,
    "stateset_agents/evaluation/capability_regression.py": 3,
    "stateset_agents/finetune_worker.py": 1,
    "stateset_agents/remote/fireworks.py": 1,
    "stateset_agents/remote/river.py": 1,
    "stateset_agents/remote/worker.py": 1,
}

#: Files whose prints are being converted to logging by in-flight training
#: work. They are exempted in pyproject but kept out of the baseline; drop a
#: file from here and from per-file-ignores once it has no prints.
PENDING_CONVERSION: frozenset[str] = frozenset(
    {
        "stateset_agents/training/neural_reward_trainer.py",
        "stateset_agents/training/transformer_reward_model.py",
    }
)

#: Unsuppressed print calls across the baseline files. Only shrinks.
CEILING = 49

#: Print calls suppressed with a T201/T203 noqa marker in the package
#: (reasoned CLI output in library modules). Only shrinks.
NOQA_CEILING = 17

#: Non-library trees where printing is allowed (scripts, demos, tests).
EXEMPT_TREES = ("tests", "examples", "benchmarks", "scripts", "deployment")


def _is_print_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in {"print", "pprint"}
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "pprint"
        and isinstance(func.value, ast.Name)
        and func.value.id == "pprint"
    )


def _package_files(root: Path = PACKAGE) -> list[Path]:
    return [p for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts]


def print_findings(root: Path = PACKAGE) -> tuple[Counter[str], Counter[str]]:
    """Per file: unsuppressed print calls, and ``noqa: T20`` suppressed ones."""
    unsuppressed: Counter[str] = Counter()
    suppressed: Counter[str] = Counter()
    for path in _package_files(root):
        source = path.read_text(encoding="utf-8")
        if "print" not in source:
            continue
        lines = source.splitlines()
        name = path.relative_to(root.parent).as_posix()
        for node in ast.walk(ast.parse(source)):
            if _is_print_call(node):
                line = lines[node.lineno - 1]  # type: ignore[attr-defined]
                if "noqa: T20" in line:
                    suppressed[name] += 1
                else:
                    unsuppressed[name] += 1
    return unsuppressed, suppressed


def _has_reason(lines: list[str], index: int) -> bool:
    marker = lines[index].split("noqa: T20", 1)[1]
    # Drop the rest of the rule code (e.g. "1"), then look for words.
    after = marker.lstrip("0123456789")
    if after.strip(" \t-—:#,"):
        return True
    previous = lines[index - 1].strip() if index else ""
    return previous.startswith("#")


def unreasoned_noqa_markers(root: Path = PACKAGE) -> Counter[str]:
    """Per file, ``# noqa: T20x`` lines with no reason on or above them."""
    found: Counter[str] = Counter()
    for path in _package_files(root):
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if "noqa: T20" in line and not _has_reason(lines, index):
                found[path.relative_to(root.parent).as_posix()] += 1
    return found


def _lint_config() -> dict:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
    return dict(config["tool"]["ruff"]["lint"])


def test_ruff_selects_the_print_rules() -> None:
    assert "T20" in set(_lint_config()["select"])


def test_ceiling_is_the_baseline_total() -> None:
    assert sum(PRINT_BASELINE.values()) == CEILING
    assert not PENDING_CONVERSION & set(PRINT_BASELINE)


def test_no_new_prints() -> None:
    unsuppressed, _ = print_findings()
    grown = {
        path: f"{count} > baseline {PRINT_BASELINE.get(path, 0)}"
        for path, count in sorted(unsuppressed.items())
        if path not in PENDING_CONVERSION and count > PRINT_BASELINE.get(path, 0)
    }
    assert not grown, (
        "New print() calls in the package. Log through a module logger "
        "(logging.getLogger(__name__)), or for genuine CLI output add "
        "`# noqa: T201 - why`:\n  "
        + "\n  ".join(f"{path}: {why}" for path, why in grown.items())
    )
    total = sum(
        count for path, count in unsuppressed.items() if path not in PENDING_CONVERSION
    )
    assert total <= CEILING


def test_print_baseline_matches_reality() -> None:
    unsuppressed, _ = print_findings()
    stale = {
        path: f"baseline {expected}, now {unsuppressed.get(path, 0)}"
        for path, expected in sorted(PRINT_BASELINE.items())
        if unsuppressed.get(path, 0) < expected
    }
    assert not stale, (
        "print() calls were removed; lower PRINT_BASELINE and CEILING (and drop "
        "files that reach zero from them and from pyproject per-file-ignores):\n  "
        + "\n  ".join(f"{path}: {why}" for path, why in stale.items())
    )


def test_lint_exemptions_are_exactly_the_baseline() -> None:
    ignores = dict(_lint_config().get("per-file-ignores", {}))
    exempt = {
        pattern
        for pattern, codes in ignores.items()
        if pattern.startswith("stateset_agents/")
        and any(code in {"T", "T2", "T20", "T201", "T203"} for code in codes)
    }
    expected = set(PRINT_BASELINE) | PENDING_CONVERSION
    assert exempt == expected, (
        "pyproject per-file-ignores must exempt from T20 exactly the files in "
        f"PRINT_BASELINE (plus PENDING_CONVERSION); extra: "
        f"{sorted(exempt - expected)}, missing: {sorted(expected - exempt)}"
    )
    trees = {
        pattern.split("/", 1)[0]
        for pattern, codes in ignores.items()
        if "T20" in codes and not pattern.startswith("stateset_agents/")
    }
    assert trees == set(EXEMPT_TREES)


def test_noqa_markers_state_a_reason_and_only_shrink() -> None:
    assert unreasoned_noqa_markers() == Counter(), (
        "`# noqa: T201` needs a reason (`# noqa: T201 - CLI report`, or a "
        "comment on the line above)"
    )
    _, suppressed = print_findings()
    total = sum(suppressed.values())
    assert total <= NOQA_CEILING, (
        f"{total} suppressed print() calls (ceiling {NOQA_CEILING}): log instead "
        f"of printing in library code. Per file: {dict(suppressed)}"
    )
    assert total >= NOQA_CEILING, (
        f"Only {total} suppressed print() calls remain; lower NOQA_CEILING from "
        f"{NOQA_CEILING} to {total}."
    )


def test_counter_and_reason_detector(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "m.py").write_text(
        "import pprint\n"
        "print('a')\n"
        "print('b')  # noqa: T201\n"
        "print('c')  # noqa: T201 - CLI line\n"
        "# CLI contract: the summary line.\n"
        "print('d')  # noqa: T201\n"
        "pprint.pprint({})\n"
        "logger_print = 'print(x)'\n",
        encoding="utf-8",
    )
    unsuppressed, suppressed = print_findings(package)
    assert unsuppressed == Counter({"pkg/m.py": 2})
    assert suppressed == Counter({"pkg/m.py": 3})
    assert unreasoned_noqa_markers(package) == Counter({"pkg/m.py": 1})
