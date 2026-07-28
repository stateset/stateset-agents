"""
One-shot release packager for the v1.0 whitepaper revision.

Aggregates all benchmark results, generates publication figures, regenerates
the whitepaper's §11.7 markdown snippet, copies the figures into ``docs/``, and
produces a release manifest the maintainer can review before publishing.

Run this after all benchmark JSON files are in ``benchmark_results/whitepaper_v1/``.

Usage:

    python scripts/release_v1_whitepaper.py
    python scripts/release_v1_whitepaper.py --dry-run
    python scripts/release_v1_whitepaper.py --strict   # fail if gates not met

The script always exits 0 unless ``--strict`` is set and gates fail. The
generated artifacts are reported on stdout so CI can grep / archive them.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("release_v1")

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "benchmark_results" / "whitepaper_v1"
DOCS_DIR = REPO_ROOT / "docs"
WHITEPAPER_SECTION_PATH = DOCS_DIR / "WHITEPAPER_SECTION_11_7.md"
RELEASE_MANIFEST_PATH = REPO_ROOT / "benchmark_results" / "RELEASE_MANIFEST.json"

WHITEPAPER_SECTION_TEMPLATE = """### 11.7 Empirical Results (v1.0)

The following table is **auto-generated** from `benchmark_results/whitepaper_v1/`
by `scripts/release_v1_whitepaper.py`. Each row is mean ± std across the seeds
recorded for the (trainer, task, model) combination. See
`benchmark_results/SCHEMA.md` for the publication gates.

**Results commit:** `{commit}`
**Generated:** {timestamp}

{table}

**Reproducibility.** Each row links back to schema-compliant JSON files under
`benchmark_results/whitepaper_v1/`. Reproduce a row by checking out the named
commit and running:

```bash
make benchmark-phase0 TRAINER=<trainer> SEED=<seed>
```

The full {n_runs}-run matrix takes ~6 hours on a single A100 80GB and can be
reproduced via `make benchmark-phase0-all`.

**Figures.**

- `docs/figures/fig_pass_at_1_per_trainer.png` — bar chart of mean post-training
  pass@1 per trainer with seed-variance error bars, plus the un-tuned baseline
  reference line.
- `docs/figures/fig_improvement_per_trainer.png` — ranked improvement over
  baseline per trainer with the +0.03 publication-gate line.
"""


def run_step(name: str, cmd: list[str], dry_run: bool = False) -> int:
    """Run a subprocess step with logging."""
    pretty = " ".join(cmd)
    if dry_run:
        logger.info("[dry-run] %s: %s", name, pretty)
        return 0
    logger.info("==> %s", name)
    logger.info("    %s", pretty)
    result = subprocess.run(cmd, check=False, cwd=REPO_ROOT)
    if result.returncode != 0:
        logger.warning("%s exited %d", name, result.returncode)
    return result.returncode


def collect_run_metadata() -> dict[str, Any]:
    """Walk the results directory and summarize what's there."""
    if not RESULTS_DIR.exists():
        return {"runs": 0, "trainers": [], "tasks": [], "seeds": [], "commits": []}

    runs: list[dict[str, Any]] = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.name in {"summary.json", "passes_gates.json"}:
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "trainer" in data and "seed" in data:
            runs.append(data)

    trainers = sorted({r.get("trainer") for r in runs if r.get("trainer")})
    tasks = sorted({r.get("task", "gsm8k") for r in runs})
    seeds = sorted({r.get("seed") for r in runs if r.get("seed") is not None})
    commits = sorted({r.get("commit") for r in runs if r.get("commit")})
    return {
        "runs": len(runs),
        "trainers": trainers,
        "tasks": tasks,
        "seeds": seeds,
        "commits": commits,
    }


def render_whitepaper_section(summary_md: str, metadata: dict[str, Any]) -> str:
    """Slot the aggregated table into the whitepaper §11.7 template."""
    commit = metadata["commits"][0] if len(metadata["commits"]) == 1 else "various"
    return WHITEPAPER_SECTION_TEMPLATE.format(
        commit=commit,
        timestamp=datetime.now(timezone.utc).isoformat(),
        table=summary_md.strip(),
        n_runs=metadata["runs"],
    )


def copy_figures(dry_run: bool = False) -> list[Path]:
    """Copy generated figures into docs/figures/ for the whitepaper."""
    figures_dir = DOCS_DIR / "figures"
    if dry_run:
        logger.info("[dry-run] would create %s and copy figures", figures_dir)
        return []
    figures_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for figure_name in (
        "fig_pass_at_1_per_trainer.png",
        "fig_improvement_per_trainer.png",
    ):
        src = RESULTS_DIR / figure_name
        if not src.exists():
            logger.warning("figure %s not found; skipping", src)
            continue
        dst = figures_dir / figure_name
        shutil.copy2(src, dst)
        copied.append(dst)
        logger.info("copied %s -> %s", src, dst)
    return copied


def write_manifest(metadata: dict[str, Any], artifacts: list[str]) -> None:
    """Persist a manifest describing what this release contains."""
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
        "artifacts": artifacts,
    }
    RELEASE_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    RELEASE_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote %s", RELEASE_MANIFEST_PATH)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print steps without writing."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if benchmark gates aren't met.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip PNG generation (markdown-only).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    # Step 1: snapshot what we have.
    metadata = collect_run_metadata()
    logger.info(
        "Found %d run(s): trainers=%s tasks=%s seeds=%s commits=%s",
        metadata["runs"],
        metadata["trainers"],
        metadata["tasks"],
        metadata["seeds"],
        metadata["commits"],
    )

    if metadata["runs"] == 0:
        logger.warning(
            "No benchmark runs found in %s. Run `make benchmark-phase0-all` first.",
            RESULTS_DIR,
        )
        if args.strict:
            return 1

    # Step 2: aggregate.
    aggregate_cmd = [
        sys.executable,
        "scripts/aggregate_phase0_results.py",
        "--results-dir",
        str(RESULTS_DIR),
    ]
    if args.strict:
        aggregate_cmd.append("--strict")
    rc = run_step("Aggregate benchmark results", aggregate_cmd, dry_run=args.dry_run)
    if rc != 0 and args.strict:
        return rc

    # Step 3: plot.
    if not args.skip_figures:
        run_step(
            "Generate publication figures",
            [
                sys.executable,
                "scripts/plot_phase0_results.py",
                "--results-dir",
                str(RESULTS_DIR),
            ],
            dry_run=args.dry_run,
        )

    # Step 4: render whitepaper §11.7 snippet.
    summary_path = RESULTS_DIR / "summary.md"
    if args.dry_run:
        logger.info("[dry-run] would render whitepaper §11.7 from %s", summary_path)
    elif summary_path.exists():
        summary_md = summary_path.read_text()
        section = render_whitepaper_section(summary_md, metadata)
        WHITEPAPER_SECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
        WHITEPAPER_SECTION_PATH.write_text(section)
        logger.info("Wrote %s", WHITEPAPER_SECTION_PATH)
    else:
        logger.warning(
            "summary.md not found at %s; skipping §11.7 render", summary_path
        )

    # Step 5: copy figures into docs/.
    figures = []
    if not args.skip_figures:
        figures = copy_figures(dry_run=args.dry_run)

    # Step 6: write the manifest.
    artifacts = [
        str(RESULTS_DIR / "summary.md"),
        str(RESULTS_DIR / "summary.csv"),
        str(RESULTS_DIR / "passes_gates.json"),
        str(WHITEPAPER_SECTION_PATH),
    ] + [str(p) for p in figures]
    if not args.dry_run:
        write_manifest(metadata, artifacts)

    logger.info("Release packaging complete.")
    print()
    print("=" * 60)
    print("v1.0 whitepaper release artifacts")
    print("=" * 60)
    for a in artifacts:
        marker = "📄" if a.endswith(".md") else "📊" if a.endswith(".png") else "📋"
        print(f"  {marker} {a}")
    print()
    print(
        f"Runs: {metadata['runs']} across trainers={metadata['trainers']} tasks={metadata['tasks']}"
    )
    print(f"Seeds: {metadata['seeds']}")
    print(f"Commits: {metadata['commits']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
