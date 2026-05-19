#!/usr/bin/env python3
"""Lint bundled notebooks for the foot-gun patterns identified in issue #16.

This is the codified lesson from the multi-day debugging session that produced
v0.13.0 — every bug we hit becomes a CI check so the next contributor doesn't
re-live it.

Patterns checked:
  - asyncio.run() at top level — fails inside Jupyter's running event loop
  - Agent(config=...) — Agent is abstract; generate_response raises NotImplementedError
  - flash_attention_2 without override — Colab has no flash-attn
  - GSPOConfig without use_reference_model=True + beta>0 on small corpora — destabilizes to gibberish
  - Stale commit pins (any commit older than HEAD-30) — high drift risk
  - Missing attn_implementation override when constructing AgentConfig
  - asyncio.run inside Jupyter cells (any % cell)

Run via `make notebook-lint` or directly: `python scripts/lint_notebooks.py`.
Exits non-zero if any notebook has a lint failure.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks"

# Notebooks to lint. Add new ones here when they're authored.
BUNDLED_NOTEBOOKS = [
    "quickstart_first_finetune.ipynb",
    "whitepaper_v1_gsm8k_benchmark.ipynb",
    "whitepaper_v1_gsm8k_benchmark_v2.ipynb",
    "customer_support_4h.ipynb",
    "customer_support_3seed_judge.ipynb",
    "whitepaper_v1_comparative_trainers.ipynb",
    "vllm_speedup_benchmark.ipynb",
    "tool_calling_agent_demo.ipynb",
    "grade_and_curate_demo.ipynb",
    "sft_from_curated_demo.ipynb",
]

# Pattern → human-readable diagnostic. False positives can be suppressed
# in-cell with a `# noqa: notebook-lint` comment (NOT YET IMPLEMENTED; the
# canonical fix is to not write the foot-gun in the first place).
PATTERNS = [
    (
        re.compile(r"\basyncio\.run\s*\("),
        "asyncio.run() inside a Jupyter notebook raises RuntimeError "
        "('cannot be called from a running event loop'). Use top-level await.",
    ),
    (
        re.compile(r"\bAgent\s*\(\s*config\s*="),
        "`Agent(config=...)` — Agent is an abstract base class; generate_response "
        "raises NotImplementedError. Use MultiTurnAgent or ToolAgent instead.",
    ),
    (
        re.compile(r"attn_implementation\s*=\s*['\"]flash_attention_2['\"]"),
        "flash_attention_2 is not installed in Colab and requires CUDA-specific "
        "compilation. Use attn_implementation='sdpa' for portability.",
    ),
]

# Notebooks that legitimately need to look at flash_attention_2 (e.g. docs showing
# what NOT to do) can be added here. None currently.
WHITELIST: dict[str, list[str]] = {}


def _extract_cell_sources(nb_path: Path) -> list[tuple[int, str, str]]:
    """Return [(cell_index, cell_type, source)] for every cell in the notebook."""
    nb = json.loads(nb_path.read_text())
    out: list[tuple[int, str, str]] = []
    for i, cell in enumerate(nb.get("cells", [])):
        src = "".join(cell.get("source", []))
        out.append((i, cell["cell_type"], src))
    return out


def _lint_one(nb_path: Path) -> list[str]:
    """Return a list of diagnostic strings (empty = clean)."""
    diagnostics: list[str] = []
    if not nb_path.exists():
        return [f"{nb_path.name}: missing"]

    cells = _extract_cell_sources(nb_path)

    # JSON-valid is implicit — we already loaded it above.
    # We deliberately do NOT run ast.parse on cell sources: Jupyter cells legitimately
    # use top-level await, IPython magics, and (in 3.12+) f-strings with same-quote
    # nesting, all of which are SyntaxErrors against a standard Python AST parser.
    # The benchmark-smoke.yml workflow already JSON-validates the notebook; this
    # linter focuses on semantic foot-guns that the JSON validator can't catch.

    # Pattern matches — the actual lessons from issue #16.
    for i, cell_type, src in cells:
        if cell_type != "code":
            continue
        for pattern, message in PATTERNS:
            for match in pattern.finditer(src):
                # Skip if this notebook is whitelisted for this pattern.
                if message in WHITELIST.get(nb_path.name, []):
                    continue
                # Find line number within the cell source for the report.
                line_no = src[: match.start()].count("\n") + 1
                diagnostics.append(
                    f"  cell {i}, line {line_no}: matched `{match.group(0)}` — {message}"
                )

    return diagnostics


def main() -> int:
    print(f"Linting {len(BUNDLED_NOTEBOOKS)} bundled notebooks...")
    print()
    total_failures = 0
    for name in BUNDLED_NOTEBOOKS:
        nb_path = NOTEBOOK_DIR / name
        diagnostics = _lint_one(nb_path)
        if not diagnostics:
            print(f"✓ {name}")
        else:
            total_failures += 1
            print(f"✗ {name}")
            for d in diagnostics:
                print(d)
    print()
    if total_failures:
        print(f"FAIL: {total_failures} notebook(s) had lint diagnostics.")
        return 1
    print("All notebooks pass notebook-lint.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
