"""Guardrail: ``Any`` in the RL numerics' signatures only ever goes down.

The code-quality review found the loss modules typed almost entirely as
``Any`` (58 of 61 signatures). They now use real tensor types and the small
Protocols in ``training/typing_protocols.py``; the ``Any`` that remains is
the heterogeneous metrics dicts the public losses return, the duck-typed
assistant turns, and the Protocols' boundary with third-party objects.

A signature counts when any parameter or return annotation mentions ``Any``.
When a refactor removes one, lower the module's cap in the same commit.
"""

from __future__ import annotations

import ast
import configparser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Module -> maximum signatures mentioning ``Any``.
ANY_SIGNATURE_CAPS = {
    "stateset_agents/training/loss_computation.py": 4,
    "stateset_agents/training/rl_losses.py": 0,
    "stateset_agents/training/objectives.py": 0,
    "stateset_agents/training/typing_protocols.py": 2,
}

#: Strict mypy flags every capped module must keep (see mypy.ini).
STRICT_FLAGS = (
    "disallow_untyped_defs",
    "disallow_incomplete_defs",
    "disallow_any_generics",
    "warn_return_any",
)


def _mentions_any(node: ast.AST | None) -> bool:
    if node is None:
        return False
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id == "Any":
            return True
        if isinstance(sub, ast.Attribute) and sub.attr == "Any":
            return True
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            try:
                parsed = ast.parse(sub.value, mode="eval")
            except SyntaxError:
                continue
            if _mentions_any(parsed):
                return True
    return False


def any_signatures(path: Path) -> list[str]:
    """Names of functions in ``path`` whose signature mentions ``Any``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        annotations = [
            arg.annotation for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)
        ]
        annotations += [
            args.vararg.annotation if args.vararg else None,
            args.kwarg.annotation if args.kwarg else None,
            node.returns,
        ]
        if any(_mentions_any(annotation) for annotation in annotations):
            found.append(f"{node.name}:{node.lineno}")
    return found


def test_any_in_numerics_signatures_never_increases() -> None:
    for module, cap in ANY_SIGNATURE_CAPS.items():
        found = any_signatures(ROOT / module)
        assert len(found) <= cap, (
            f"{module}: {len(found)} signatures mention Any (cap {cap}). Use a "
            "tensor type or a Protocol from training/typing_protocols.py: "
            + ", ".join(found)
        )


def test_caps_match_reality() -> None:
    for module, cap in ANY_SIGNATURE_CAPS.items():
        count = len(any_signatures(ROOT / module))
        assert count >= cap, (
            f"{module} now has {count} signatures mentioning Any; lower its cap "
            f"from {cap} to {count} to lock in the improvement."
        )


def test_numerics_modules_stay_strict_in_mypy() -> None:
    parser = configparser.ConfigParser()
    parser.read(ROOT / "mypy.ini")
    for module in ANY_SIGNATURE_CAPS:
        dotted = module.removesuffix(".py").replace("/", ".")
        section = f"mypy-{dotted}"
        assert parser.has_section(section), f"mypy.ini lost [{section}]"
        for flag in STRICT_FLAGS:
            assert parser.getboolean(
                section, flag, fallback=False
            ), f"[{section}] must keep {flag} = true"


def test_measure_counts_parameters_returns_and_string_annotations(
    tmp_path: Path,
) -> None:
    source = tmp_path / "m.py"
    source.write_text(
        "from typing import Any\n"
        "def a(x: Any) -> int: ...\n"
        "def b(x: int) -> 'dict[str, Any]': ...\n"
        "def c(*args: int, **kwargs: Any) -> None: ...\n"
        "def d(x: int) -> int: ...\n",
        encoding="utf-8",
    )
    assert [name.split(":")[0] for name in any_signatures(source)] == ["a", "b", "c"]
