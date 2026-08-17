"""The flywheel, as one command: harvest → train → eval → repeat.

Productizes the loop that took a Muse-Glimmer-30B agent from 2/12 to 10/12
on out-of-distribution compound requests (``docs/FLYWHEEL_HEADROOM.md``):
each generation, the current model's rare successes are harvested by
best-of-N rejection sampling against objective checks, and the next
generation is trained on nothing but those successes.

The loop is deliberately dumb and auditable:

1. **Harvest** (``job_kind="harvest"``): sample ``best_of`` completions per
   harvest prompt from the current generation, keep the ones passing that
   prompt's ``expect``/``forbid`` checks, and measure the current
   generation on the eval set while the model is loaded.
2. **Stop if there is no way forward** — an empty harvest means the model
   never succeeds at temperature, and more training data cannot come from it.
3. **Train** the next generation on the harvest, with the eval prompts (and
   their assertions) as the job's own eval gate. A gate failure is *data*
   here, not an error: 10/12 fails an all-assertions gate while being a
   huge improvement, so the loop reads ``eval_results.json`` from the
   fetched artifacts either way (the fetch-on-failure contract exists for
   exactly this reason).
4. **Stop on plateau** (no improvement over the previous generation), on
   ``generations`` exhausted, or when the **worst-case cost of the next
   generation** would break ``max_cost_usd`` — checked *before* renting,
   because a budget you can only exceed is not a budget.

Every generation leaves a full audit trail: the harvest set, its summary,
the adapter with its lineage manifest, and ``flywheel_report.json`` tying
them together with pass rates and dollars.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stateset_agents.remote.executor import RemoteExecutor
from stateset_agents.remote.job import JobStatus, RemoteJobSpec

logger = logging.getLogger(__name__)

__all__ = ["FlywheelConfig", "GenerationOutcome", "run_flywheel"]


@dataclass
class FlywheelConfig:
    """Everything one flywheel run needs, provider knobs included."""

    base_model: str
    #: ``{prompt, expect, forbid}`` specs sampled during harvest. The checks
    #: are mandatory — without them every sample "passes" and the harvest is
    #: noise (enforced by the harvest job itself).
    harvest_prompts: list[dict[str, Any]]
    #: ``{prompt, expect, forbid}`` specs that score each generation. Keep
    #: them disjoint from ``harvest_prompts`` or the score measures leakage.
    eval_prompts: list[dict[str, Any]]
    output_root: Path = Path("outputs/flywheel")
    #: Adapter directory of the current generation, if one already exists
    #: (e.g. a persona fine-tune). ``None`` starts from the bare base model.
    initial_adapter: Path | None = None
    #: Maximum NEW generations to train.
    generations: int = 3
    best_of: int = 8
    temperature: float = 0.9
    top_p: float = 0.95
    max_new_tokens: int = 300
    eval_max_new_tokens: int = 300
    #: Training knobs for each generation.
    num_epochs: int = 3
    lora_r: int = 16
    lora_alpha: int = 32
    learning_rate: float = 1e-4
    max_length: int = 512
    #: Hard ceiling for the WHOLE run, enforced before each rental from the
    #: worst case of the next job. ``None`` means uncapped.
    max_cost_usd: float | None = None
    #: Provider resources, passed through to every spec.
    gpu: str | None = None
    gpu_count: int = 1
    container_disk_gb: int | None = None
    cloud_type: str = "SECURE"
    #: Worst-case seconds per job, used with the provider's $/hr for the
    #: pre-rental budget check and as the job timeout.
    timeout_s: int = 3600
    dry_run: bool = False


@dataclass
class GenerationOutcome:
    """What one turn of the wheel produced."""

    generation: int
    adapter_dir: Path | None
    harvest_kept: int
    harvest_samples: int
    eval_passed: int | None
    eval_total: int | None
    cost_usd: float | None
    stopped: str | None = None  # reason the loop ended here, if it did


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _eval_score(output_dir: Path | None) -> tuple[int | None, int | None]:
    """(passed, total) from a training job's ``eval_results.json``."""
    if output_dir is None:
        return None, None
    data = _read_json(Path(output_dir) / "eval_results.json")
    if not isinstance(data, dict):
        return None, None
    results = data.get("results")
    if not isinstance(results, list):
        return None, None
    checked = [r for r in results if isinstance(r, dict) and "passed" in r]
    if not checked:
        return None, None
    return sum(1 for r in checked if r["passed"]), len(checked)


def _spend(results: list[float | None]) -> float:
    return sum(c for c in results if c is not None)


def run_flywheel(
    config: FlywheelConfig,
    executor: RemoteExecutor,
) -> dict[str, Any]:
    """Turn the wheel until plateau, exhaustion, budget, or a dry harvest.

    Returns the report that is also written to
    ``output_root/flywheel_report.json``. Raises only on infrastructure
    errors; "the model stopped improving" is a result, not an exception.
    """
    root = Path(config.output_root)
    root.mkdir(parents=True, exist_ok=True)

    adapter: Path | None = config.initial_adapter
    costs: list[float | None] = []
    outcomes: list[GenerationOutcome] = []
    best_passed: int | None = None
    stop_reason = "generations exhausted"

    for generation in range(1, config.generations + 1):
        if config.max_cost_usd is not None:
            spent = _spend(costs)
            if spent >= config.max_cost_usd:
                stop_reason = (
                    f"budget: ${spent:.2f} of ${config.max_cost_usd:.2f} spent"
                )
                break

        gen_dir = root / f"gen{generation}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        # ---- 1. harvest from the current generation ----------------------
        prompts_file = gen_dir / "harvest_prompts.json"
        prompts_file.write_text(json.dumps(config.harvest_prompts, indent=2))
        harvest_spec = RemoteJobSpec(
            dataset=prompts_file,
            base_model=config.base_model,
            output_dir=gen_dir / "harvest",
            job_kind="harvest",
            harvest={
                "adapter_dir": str(adapter) if adapter else None,
                "best_of": config.best_of,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_new_tokens": config.max_new_tokens,
            },
            eval_prompts=list(config.eval_prompts),
            eval_max_new_tokens=config.eval_max_new_tokens,
            dry_run=config.dry_run,
            gpu=config.gpu,
            gpu_count=config.gpu_count,
            container_disk_gb=config.container_disk_gb,
            cloud_type=config.cloud_type,
            timeout_s=config.timeout_s,
            max_cost_usd=(
                None
                if config.max_cost_usd is None
                else max(0.0, config.max_cost_usd - _spend(costs))
            ),
        )
        logger.info("generation %d: harvesting…", generation)
        harvest_result = executor.wait(executor.submit(harvest_spec))
        costs.append(harvest_result.cost_usd)
        if not harvest_result.succeeded:
            raise RuntimeError(
                f"generation {generation} harvest failed:\n"
                + "\n".join(harvest_result.logs[-20:])
            )
        summary = (
            _read_json(Path(harvest_result.output_dir) / "harvest_summary.json")
            if harvest_result.output_dir
            else None
        ) or {}
        kept = int(summary.get("kept") or 0)
        samples = int(summary.get("samples") or 0)
        current_eval = summary.get("eval") or {}
        if best_passed is None and isinstance(current_eval.get("passed"), int):
            best_passed = int(current_eval["passed"])
            logger.info(
                "current generation scores %d/%s before training",
                best_passed,
                current_eval.get("total"),
            )

        if kept == 0 and not config.dry_run:
            outcomes.append(
                GenerationOutcome(
                    generation=generation,
                    adapter_dir=None,
                    harvest_kept=0,
                    harvest_samples=samples,
                    eval_passed=None,
                    eval_total=None,
                    cost_usd=harvest_result.cost_usd,
                    stopped="dry harvest",
                )
            )
            stop_reason = (
                f"dry harvest: 0/{samples} samples passed — the current "
                "generation never succeeds at temperature, so there is no "
                "signal to train on"
            )
            break

        harvest_dataset = (
            Path(harvest_result.output_dir) / "harvest.jsonl"
            if harvest_result.output_dir
            else gen_dir / "harvest" / "harvest.jsonl"
        )
        if config.dry_run and not harvest_dataset.exists():
            # A dry-run harvest writes no dataset; the train job still needs
            # an existing file to print its plan against.
            harvest_dataset.parent.mkdir(parents=True, exist_ok=True)
            harvest_dataset.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "[dry-run placeholder]"},
                            {"role": "assistant", "content": "[dry-run]"},
                        ]
                    }
                )
                + "\n"
            )

        # ---- 2. train the next generation on the harvest -----------------
        train_spec = RemoteJobSpec(
            dataset=harvest_dataset,
            base_model=config.base_model,
            output_dir=gen_dir / "adapter",
            num_epochs=config.num_epochs,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            learning_rate=config.learning_rate,
            max_length=config.max_length,
            eval_prompts=list(config.eval_prompts),
            eval_max_new_tokens=config.eval_max_new_tokens,
            parent_adapter=str(adapter) if adapter else None,
            dry_run=config.dry_run,
            gpu=config.gpu,
            gpu_count=config.gpu_count,
            container_disk_gb=config.container_disk_gb,
            cloud_type=config.cloud_type,
            timeout_s=config.timeout_s,
            max_cost_usd=(
                None
                if config.max_cost_usd is None
                else max(0.0, config.max_cost_usd - _spend(costs))
            ),
        )
        logger.info("generation %d: training on %d harvested rows…", generation, kept)
        train_result = executor.wait(executor.submit(train_spec))
        costs.append(train_result.cost_usd)
        # A FAILED status with artifacts is the eval gate speaking (e.g.
        # 10/12): read the score either way. FAILED *without* artifacts is a
        # real failure.
        passed, total = _eval_score(train_result.output_dir)
        if (
            train_result.status is not JobStatus.SUCCEEDED
            and passed is None
            and not config.dry_run
        ):
            raise RuntimeError(
                f"generation {generation} training failed with no eval "
                "artifacts:\n" + "\n".join(train_result.logs[-20:])
            )

        outcome = GenerationOutcome(
            generation=generation,
            adapter_dir=train_result.output_dir,
            harvest_kept=kept,
            harvest_samples=samples,
            eval_passed=passed,
            eval_total=total,
            cost_usd=_spend(costs[-2:]),
        )
        outcomes.append(outcome)
        logger.info(
            "generation %d: %s/%s eval prompts passed (cost ~$%.2f)",
            generation,
            passed,
            total,
            outcome.cost_usd or 0.0,
        )

        # ---- 3. plateau check --------------------------------------------
        if passed is not None and best_passed is not None and passed <= best_passed:
            stop_reason = (
                f"plateau: generation {generation} scored {passed}, "
                f"previous best was {best_passed}"
            )
            # The previous adapter stays current; this one is kept on disk
            # for the audit trail but not advanced.
            break
        if passed is not None:
            best_passed = passed
        adapter = train_result.output_dir or adapter
        if total is not None and passed == total:
            stop_reason = f"perfect score: {passed}/{total}"
            break

    report = {
        "base_model": config.base_model,
        "initial_adapter": (
            str(config.initial_adapter) if config.initial_adapter else None
        ),
        "stop_reason": stop_reason,
        "final_adapter": str(adapter) if adapter else None,
        "best_eval_passed": best_passed,
        "total_cost_usd": round(_spend(costs), 4) if costs else 0.0,
        "unpriced_jobs": sum(1 for c in costs if c is None),
        "generations": [
            {
                "generation": o.generation,
                "adapter_dir": str(o.adapter_dir) if o.adapter_dir else None,
                "harvest_kept": o.harvest_kept,
                "harvest_samples": o.harvest_samples,
                "eval_passed": o.eval_passed,
                "eval_total": o.eval_total,
                "cost_usd": o.cost_usd,
                "stopped": o.stopped,
            }
            for o in outcomes
        ],
    }
    (root / "flywheel_report.json").write_text(json.dumps(report, indent=2))
    return report
