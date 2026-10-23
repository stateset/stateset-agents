"""Guardrail: the number of oversized functions in the package only goes down.

A function is oversized when it spans more than ``MAX_LINES`` lines or its
branch count (``if``/loops/``try``/``with``/``except``/boolean operators/
comprehensions/conditional expressions/``match``, plus one) exceeds
``MAX_BRANCHES`` -- the measure the code-quality review used. Split new long
functions instead of raising ``CEILING``; when a refactor (or the removal of a
deprecated module) lowers the count, lower ``CEILING`` in the same commit.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "stateset_agents"

MAX_LINES = 150
MAX_BRANCHES = 40

#: 82 when the ratchet was introduced (v0.70.0 development); splitting
#: cli._validate_config, AgentConfig.validate, the /api/v1/conversations
#: handler, finetune_workspace.check_config_values and
#: data.finetuning.validate_finetuning_row brought it to 77; splitting the
#: spine loop's train() into an episode step brought it to 74.
CEILING = 74

_BRANCH_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.ExceptHandler,
    ast.BoolOp,
    ast.IfExp,
    ast.comprehension,
    ast.Match,
)


def oversized_functions(root: Path = PACKAGE) -> list[str]:
    """Return ``path:line name (lines, branches)`` for every oversized function."""
    return list(_oversized_functions(root))


@lru_cache(maxsize=4)
def _oversized_functions(root: Path) -> tuple[str, ...]:
    found: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            assert node.end_lineno is not None
            lines = node.end_lineno - node.lineno + 1
            branches = 1 + sum(
                isinstance(child, _BRANCH_NODES) for child in ast.walk(node)
            )
            if lines > MAX_LINES or branches > MAX_BRANCHES:
                rel = path.relative_to(root.parent).as_posix()
                found.append(
                    f"{rel}:{node.lineno} {node.name} ({lines} lines, "
                    f"{branches} branches)"
                )
    return tuple(found)


def test_oversized_function_count_never_increases() -> None:
    found = oversized_functions()
    assert len(found) <= CEILING, (
        f"{len(found)} functions exceed {MAX_LINES} lines or {MAX_BRANCHES} "
        f"branches (ceiling {CEILING}). Split the new or grown function into "
        "focused helpers instead of raising the ceiling:\n  " + "\n  ".join(found)
    )


def test_ceiling_matches_reality() -> None:
    found = oversized_functions()
    assert len(found) >= CEILING, (
        f"Only {len(found)} oversized functions remain; lower CEILING from "
        f"{CEILING} to {len(found)} to lock in the improvement."
    )


def test_split_functions_stay_split() -> None:
    """The functions split in v0.70.0 must not grow back past the limits."""
    found = oversized_functions()
    oversized = {(entry.split(":", 1)[0], entry.split(" ", 2)[1]) for entry in found}
    for key in (
        ("stateset_agents/cli.py", "_validate_config"),
        ("stateset_agents/core/agent_config.py", "validate"),
        ("stateset_agents/api/routers/v1.py", "chat_v1"),
        ("stateset_agents/finetune_workspace.py", "check_config_values"),
        ("stateset_agents/data/finetuning.py", "validate_finetuning_row"),
    ):
        assert key not in oversized, f"{key} is oversized again"


def test_measure_counts_lines_and_branches(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    long_body = "\n".join(f"    x{i} = {i}" for i in range(MAX_LINES))
    branchy = "\n".join(f"    if a == {i}:\n        return {i}" for i in range(41))
    (package / "m.py").write_text(
        f"def long():\n{long_body}\n\n"
        f"def branchy(a):\n{branchy}\n\n"
        "def fine():\n    return 1\n",
        encoding="utf-8",
    )
    found = oversized_functions(package)
    assert [entry.split(" ")[1] for entry in found] == ["long", "branchy"]
