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
import math
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any

from stateset_agents.remote.executor import RemoteExecutor
from stateset_agents.remote.job import JobStatus, RemoteJobSpec
from stateset_agents.remote.ledger import remaining_budget_usd

logger = logging.getLogger(__name__)

__all__ = [
    "FlywheelConfig",
    "GenerationOutcome",
    "run_flywheel",
    "run_flywheel_repeats",
]


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
    #: Distillation mode: a FIXED teacher does the harvesting while the
    #: student (``base_model`` + ``initial_adapter``) is what gets trained
    #: and measured. The teacher never advances between generations — its
    #: successes are the curriculum, not the pupil. Motivated by the
    #: capacity study: the 35B clears walls the 9B cannot, but the 9B is
    #: what you want to serve.
    teacher_base_model: str | None = None
    teacher_adapter: Path | None = None
    #: The rarity controller. When set (e.g. 0.6), each generation first
    #: probes a few prompts at several temperatures and harvests at the one
    #: whose pass rate lands nearest this target — keeping the loop inside
    #: its measured operating window (~60% harvest with headroom: lifts at
    #: rungs 3/4, stalls at 65%/83%). None (default) uses ``temperature``.
    target_harvest_rate: float | None = None
    #: Temperatures the controller may choose between.
    probe_temperatures: tuple[float, ...] = (0.7, 0.9, 1.1)
    #: Prompts and samples per probe (kept tiny: the probe is a thermometer,
    #: not a harvest).
    probe_prompts: int = 6
    probe_best_of: int = 4
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
    #: Shared compute budget checked against the next job's projected cost.
    #: ``None`` means uncapped.
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
    """(passed, total) from a training job's ``eval_results.json``.

    The file is what :func:`stateset_agents.training.sft.write_eval_results`
    writes: a bare LIST of rows, each carrying its assertion outcome nested
    as ``checks.passed`` (rows for plain prompts have no ``checks`` and are
    not counted). The first live run shipped a reader for an imagined
    ``{"results": [...]}`` envelope and scored every real file as None.
    """
    if output_dir is None:
        return None, None
    try:
        data = json.loads((Path(output_dir) / "eval_results.json").read_text())
    except (OSError, ValueError):
        return None, None
    rows = data.get("results") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return None, None
    outcomes = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        checks = row.get("checks")
        if isinstance(checks, dict) and "passed" in checks:
            outcomes.append(bool(checks["passed"]))
        elif "passed" in row:
            outcomes.append(bool(row["passed"]))
    if not outcomes:
        return None, None
    return sum(outcomes), len(outcomes)


def _spend(results: list[float | None]) -> float:
    known = [c for c in results if c is not None]
    if any(not math.isfinite(c) or c < 0 for c in known):
        raise RuntimeError("Invalid provider cost. The remaining budget is unknown.")
    return float(sum((Decimal(str(c)) for c in known), Decimal(0)))


def _probe_temperature(
    config: FlywheelConfig,
    executor: RemoteExecutor,
    gen_dir: Path,
    harvest_base: str,
    adapter_dir: str | None,
    costs: list[float | None],
) -> float:
    """Pick the sampling temperature whose pass rate lands nearest target.

    Runs one tiny harvest per candidate temperature over the first few
    prompts. Record each cost before admitting the next probe. A failed
    submission with uncertain charges stops a budgeted campaign.
    """
    assert config.target_harvest_rate is not None
    prompts = config.harvest_prompts[: config.probe_prompts]
    probe_file = gen_dir / "probe_prompts.json"
    probe_file.write_text(json.dumps(prompts))
    best_temp = config.temperature
    best_gap: float | None = None
    for temp in config.probe_temperatures:
        remaining = remaining_budget_usd(config.max_cost_usd, costs)
        if remaining == 0:
            break
        spec = RemoteJobSpec(
            dataset=probe_file,
            base_model=harvest_base,
            output_dir=gen_dir / f"probe_t{temp}",
            job_kind="harvest",
            harvest={
                "adapter_dir": adapter_dir,
                "best_of": config.probe_best_of,
                "temperature": temp,
                "top_p": config.top_p,
                "max_new_tokens": config.max_new_tokens,
            },
            dry_run=config.dry_run,
            gpu=config.gpu,
            gpu_count=config.gpu_count,
            container_disk_gb=config.container_disk_gb,
            cloud_type=config.cloud_type,
            timeout_s=config.timeout_s,
            max_cost_usd=remaining,
        )
        try:
            result = executor.wait(executor.submit(spec))
        except Exception as exc:  # noqa: BLE001 - probe is best-effort
            if remaining is not None:
                raise RuntimeError(
                    f"Budgeted temperature probe {temp} failed: {exc}"
                ) from exc
            logger.warning("temperature probe %.2f failed (%s); skipping", temp, exc)
            continue
        costs.append(result.cost_usd)
        if result.is_preview or not result.succeeded:
            continue
        summary = (
            _read_json(Path(result.output_dir) / "harvest_summary.json")
            if result.output_dir
            else None
        ) or {}
        samples = int(summary.get("samples") or 0)
        if summary.get("dry_run") or not samples:
            continue
        rate = int(summary.get("kept") or 0) / samples
        gap = abs(rate - config.target_harvest_rate)
        logger.info("probe t=%.2f: harvest rate %.0f%%", temp, rate * 100)
        if best_gap is None or gap < best_gap:
            best_gap, best_temp = gap, temp
    logger.info("rarity controller chose temperature %.2f", best_temp)
    return best_temp


def _harvest_job(
    config: FlywheelConfig,
    gen_dir: Path,
    adapter: Path | None,
    harvest_temperature: float,
    budget: float | None,
) -> RemoteJobSpec:
    distilling = config.teacher_base_model is not None
    source_adapter = config.teacher_adapter if distilling else adapter
    return RemoteJobSpec(
        dataset=gen_dir / "harvest_prompts.json",
        base_model=config.teacher_base_model or config.base_model,
        output_dir=gen_dir / "harvest",
        job_kind="harvest",
        harvest={
            # In distillation the teacher is fixed; otherwise the
            # current student generation harvests from itself.
            "adapter_dir": str(source_adapter) if source_adapter else None,
            "best_of": config.best_of,
            "temperature": harvest_temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_new_tokens,
        },
        eval_prompts=None if distilling else list(config.eval_prompts),
        eval_max_new_tokens=config.eval_max_new_tokens,
        _require_dataset=not config.dry_run,
        gpu=config.gpu,
        gpu_count=config.gpu_count,
        container_disk_gb=config.container_disk_gb,
        cloud_type=config.cloud_type,
        timeout_s=config.timeout_s,
        max_cost_usd=budget,
    )


def _training_job(
    config: FlywheelConfig,
    gen_dir: Path,
    adapter: Path | None,
    harvest_dataset: Path,
    budget: float | None,
) -> RemoteJobSpec:
    return RemoteJobSpec(
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
        _require_dataset=not config.dry_run,
        gpu=config.gpu,
        gpu_count=config.gpu_count,
        container_disk_gb=config.container_disk_gb,
        cloud_type=config.cloud_type,
        timeout_s=config.timeout_s,
        max_cost_usd=budget,
    )


def _budget_stop_reason(costs: list[float | None]) -> str:
    if None in costs:
        return "budget: a prior job has an unknown cost"
    return "budget exhausted"


def _plan_flywheel(
    config: FlywheelConfig,
    executor: RemoteExecutor,
    repeats: int,
    first_run: Path,
) -> dict[str, Any]:
    """Describe one generation and the campaign bounds before any submission."""
    if executor.uses_packaged_training:
        from stateset_agents.training.harvest import normalize_harvest_prompts

        normalize_harvest_prompts(list(config.harvest_prompts))
    gen_dir = first_run / "gen1"
    harvest = _harvest_job(
        config, gen_dir, config.initial_adapter, config.temperature, config.max_cost_usd
    )
    train = _training_job(
        config,
        gen_dir,
        config.initial_adapter,
        harvest.output_dir / "harvest.jsonl",
        config.max_cost_usd,
    )
    jobs: list[dict[str, Any]] = []
    if config.target_harvest_rate is not None:
        jobs.extend(
            {
                "job_kind": "harvest",
                "purpose": "temperature probe",
                "base_model": harvest.base_model,
                "temperature": temperature,
                "prompts": len(config.harvest_prompts[: config.probe_prompts]),
                "best_of": config.probe_best_of,
            }
            for temperature in config.probe_temperatures
        )
    for spec, fields in (
        (harvest, ("harvest", "eval_max_new_tokens")),
        (
            train,
            (
                "num_epochs",
                "lora_r",
                "lora_alpha",
                "learning_rate",
                "max_length",
                "parent_adapter",
                "eval_max_new_tokens",
            ),
        ),
    ):
        executor.validate_job_config(spec)
        settings = spec.to_dict()
        job = {
            key: settings[key]
            for key in ("job_kind", "base_model", "dataset", "output_dir", *fields)
        }
        job["eval_prompt_count"] = len(spec.eval_prompts or [])
        jobs.append(job)
    generations = max(config.generations, 0)
    return {
        "dry_run": True,
        "provider": executor.name,
        "base_model": config.base_model,
        "repeats": repeats,
        "generations_per_run": generations,
        "max_jobs": repeats * generations * len(jobs),
        "harvest_prompt_count": len(config.harvest_prompts),
        "target_harvest_rate": config.target_harvest_rate,
        "first_generation_jobs": jobs if generations else [],
        "max_cost_usd": config.max_cost_usd,
        "resources": {
            "gpu": train.gpu,
            "gpu_count": train.gpu_count,
            "cloud_type": train.cloud_type,
            "container_disk_gb": train.container_disk_gb,
            "timeout_s": train.timeout_s,
        },
    }


def run_flywheel(
    config: FlywheelConfig,
    executor: RemoteExecutor,
) -> dict[str, Any]:
    """Turn the wheel until plateau, exhaustion, budget, or a dry harvest.

    A dry run returns an offline job plan. A live run returns the report written to
    ``output_root/flywheel_report.json``. Raises only on infrastructure
    errors; "the model stopped improving" is a result, not an exception.
    """
    root = Path(config.output_root)
    executor.validate_cost_ceiling(config.max_cost_usd)
    if config.dry_run:
        return _plan_flywheel(config, executor, 1, root)
    root.mkdir(parents=True, exist_ok=True)

    adapter: Path | None = config.initial_adapter
    costs: list[float | None] = []
    outcomes: list[GenerationOutcome] = []
    best_passed: int | None = None
    stop_reason = "generations exhausted"

    for generation in range(1, config.generations + 1):
        if remaining_budget_usd(config.max_cost_usd, costs) == 0:
            stop_reason = _budget_stop_reason(costs)
            break
        generation_start = len(costs)

        gen_dir = root / f"gen{generation}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        # ---- 1. harvest from the current generation ----------------------
        prompts_file = gen_dir / "harvest_prompts.json"
        prompts_file.write_text(json.dumps(config.harvest_prompts, indent=2))
        distilling = config.teacher_base_model is not None
        source_adapter = config.teacher_adapter if distilling else adapter
        harvest_temperature = config.temperature
        if config.target_harvest_rate is not None:
            harvest_temperature = _probe_temperature(
                config,
                executor,
                gen_dir,
                config.teacher_base_model or config.base_model,
                str(source_adapter) if source_adapter else None,
                costs,
            )
        remaining = remaining_budget_usd(config.max_cost_usd, costs)
        if remaining == 0:
            stop_reason = _budget_stop_reason(costs)
            break
        harvest_spec = _harvest_job(
            config,
            gen_dir,
            adapter,
            harvest_temperature,
            remaining,
        )
        logger.info(
            "generation %d: harvesting%s…",
            generation,
            " (from the teacher)" if distilling else "",
        )
        harvest_result = executor.wait(executor.submit(harvest_spec))
        costs.append(harvest_result.cost_usd)
        if not harvest_result.succeeded:
            raise RuntimeError(
                f"generation {generation} harvest failed:\n"
                + "\n".join(harvest_result.logs[-20:])
            )
        if harvest_result.is_preview:
            raise RuntimeError(
                f"Generation {generation} harvest returned a preview. Sampling is required."
            )
        if harvest_result.output_dir is None:
            raise RuntimeError(
                f"Generation {generation} harvest returned no artifacts."
            )
        harvest_dir = Path(harvest_result.output_dir)
        summary = _read_json(harvest_dir / "harvest_summary.json")
        if summary is None:
            raise RuntimeError(
                f"Generation {generation} harvest summary is missing or unreadable."
            )
        if summary.get("dry_run"):
            raise RuntimeError(
                f"Generation {generation} harvest summary records a preview."
            )
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

        remaining = remaining_budget_usd(config.max_cost_usd, costs)
        if kept == 0 or remaining == 0:
            stopped = "dry harvest" if kept == 0 else _budget_stop_reason(costs)
            outcomes.append(
                GenerationOutcome(
                    generation=generation,
                    adapter_dir=None,
                    harvest_kept=kept,
                    harvest_samples=samples,
                    eval_passed=None,
                    eval_total=None,
                    cost_usd=_spend(costs[generation_start:]),
                    stopped=stopped,
                )
            )
            stop_reason = (
                stopped
                if kept
                else (
                    f"dry harvest: 0/{samples} samples passed — the current "
                    "generation never succeeds at temperature, so there is no "
                    "signal to train on"
                )
            )
            break

        harvest_dataset = harvest_dir / "harvest.jsonl"
        # ---- 2. train the next generation on the harvest -----------------
        train_spec = _training_job(config, gen_dir, adapter, harvest_dataset, remaining)
        logger.info("generation %d: training on %d harvested rows…", generation, kept)
        train_result = executor.wait(executor.submit(train_spec))
        costs.append(train_result.cost_usd)
        if train_result.is_preview:
            raise RuntimeError(
                f"Generation {generation} training returned a preview. A trained adapter is required."
            )
        if train_result.succeeded and train_result.output_dir is None:
            raise RuntimeError(
                f"Generation {generation} training returned no artifacts."
            )
        # A FAILED status with artifacts is the eval gate speaking (e.g.
        # 10/12): read the score either way. FAILED *without* artifacts is a
        # real failure.
        passed, total = _eval_score(train_result.output_dir)
        if train_result.status is not JobStatus.SUCCEEDED and passed is None:
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
            cost_usd=_spend(costs[generation_start:]),
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
        "total_cost_usd": _spend(costs),
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


def run_flywheel_repeats(
    config: FlywheelConfig,
    executor: RemoteExecutor,
    repeats: int,
) -> dict[str, Any]:
    """Run the flywheel ``repeats`` times; report the score distribution.

    A dry run returns one offline plan for the whole campaign.

    Two runs of the domain-2 experiment scored 7/12 and 11/12 — a spread
    wide enough that any single run over- or under-states the mechanism.
    Repeats turn "reproduced" into a distribution: per-run best scores plus
    min/mean/max, in one report.

    The budget is shared: ``config.max_cost_usd`` caps the WHOLE campaign,
    and each repeat receives what remains — a repeat that would start with
    nothing left is skipped and reported as such. Each repeat writes under
    ``output_root/run<N>/`` with its own full audit trail.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    executor.validate_cost_ceiling(config.max_cost_usd)
    root = Path(config.output_root)
    if config.dry_run:
        return _plan_flywheel(config, executor, repeats, root / "run1")
    root.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    spent = 0.0
    unpriced_jobs = 0
    for index in range(1, repeats + 1):
        remaining = remaining_budget_usd(config.max_cost_usd, [spent])
        if remaining is not None and (remaining <= 0 or unpriced_jobs):
            runs.append(
                {
                    "run": index,
                    "skipped": (
                        "budget: a prior job has an unknown cost"
                        if unpriced_jobs
                        else (
                            f"budget exhausted: ${spent:.2f} of "
                            f"${config.max_cost_usd:.2f} spent"
                        )
                    ),
                }
            )
            continue
        run_config = replace(
            config,
            output_root=root / f"run{index}",
            max_cost_usd=remaining,
        )
        logger.info("repeat %d/%d starting…", index, repeats)
        report = run_flywheel(run_config, executor)
        spent = _spend([spent, report["total_cost_usd"]])
        unpriced_jobs += report["unpriced_jobs"]
        runs.append(
            {
                "run": index,
                "best_eval_passed": report.get("best_eval_passed"),
                "stop_reason": report.get("stop_reason"),
                "final_adapter": report.get("final_adapter"),
                "cost_usd": report.get("total_cost_usd"),
            }
        )

    scores = [
        r["best_eval_passed"]
        for r in runs
        if isinstance(r.get("best_eval_passed"), int)
    ]
    aggregate = {
        "repeats": repeats,
        "completed": len(scores),
        "runs": runs,
        "scores": scores,
        "min": min(scores) if scores else None,
        "max": max(scores) if scores else None,
        "mean": round(sum(scores) / len(scores), 2) if scores else None,
        "total_cost_usd": spent,
        "unpriced_jobs": unpriced_jobs,
    }
    (root / "flywheel_repeats_report.json").write_text(json.dumps(aggregate, indent=2))
    return aggregate
