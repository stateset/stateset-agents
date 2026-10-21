"""Validation-only judge checkpoint selection for a single-process trainer."""

from __future__ import annotations

import copy
import inspect
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.evaluation.case_separation import validate_case_separation
from stateset_agents.evaluation.support_judge import (
    JudgeItem,
    rubric_revision,
)
from stateset_agents.rewards.support_judge_reward import checked_judgement
from stateset_agents.training.promotion import artifact_digest


class _JudgedValidation(list[Any]):
    """List-compatible results carrying a detached raw-evidence reference."""

    def __init__(self, results: Any, evidence: dict[str, Any] | None) -> None:
        super().__init__(results)
        self.evidence = copy.deepcopy(evidence)


@contextmanager
def _greedy_generation(agent: Any) -> Iterator[None]:
    """Temporarily override both engine and cached HF decoding configuration."""
    originals = {}
    try:
        for name in ("config", "generation_config"):
            original = getattr(agent, name, None)
            if original is None:
                continue
            overrides: dict[str, Any] = {
                "do_sample": False,
                "temperature": 0.0 if name == "config" else 1.0,
                "top_p": 1.0,
                "top_k": 0 if name == "config" else 50,
                "repetition_penalty": 1.0,
            }
            if name == "generation_config":
                overrides.update(num_beams=1, num_return_sequences=1)
            temporary = copy.deepcopy(original)
            for key, value in overrides.items():
                setattr(temporary, key, value)
            originals[name] = original
            setattr(agent, name, temporary)
        yield
    finally:
        for name, original in originals.items():
            setattr(agent, name, original)


def support_judge_evaluator(
    judge: Any, environment: Any, *, max_turns: int = 4
) -> Callable[..., Any]:
    """Generate fresh validation episodes with current weights, then batch-judge.

    The judge may use a separate process/endpoint so its model need not share
    learner GPU memory. Environment clones keep validation from resetting the
    training environment. Decoding is greedy; training configuration objects,
    RNGs and model modes are restored, including after a failed generation.
    This evaluator requires exclusive use of the agent during validation.
    """
    if type(max_turns) is not int or max_turns < 1:
        raise ValueError("max_turns must be a positive integer")

    async def evaluate(trainer: Any, cases: Sequence[Mapping[str, Any]]) -> Any:
        from stateset_agents.evaluation.isolation import (
            isolated_evaluation,
            paired_evaluation_seed,
        )

        items = []
        seed = getattr(trainer.config, "seed", 42)
        seed = 42 if seed is None else seed
        for case in cases:
            env = environment.clone()
            if env is environment:
                raise ValueError(
                    "Judge validation requires an independent environment clone"
                )
            case_seed = paired_evaluation_seed(seed, case["id"], content_hash(case))
            with (
                isolated_evaluation(seed=case_seed, model=trainer.agent.model),
                _greedy_generation(trainer.agent),
            ):
                trajectory = await env.run_episode(
                    trainer.agent.generate_response,
                    scenario=copy.deepcopy(dict(case)),
                    max_turns=max_turns,
                )
            transcript = tuple(
                (str(turn.role), turn.content or "")
                for turn in trajectory.turns
                if turn.role in {"user", "assistant"}
            )
            if not any(
                role == "assistant" and text.strip() for role, text in transcript
            ):
                raise ValueError("Judge validation generated no assistant completion")
            items.append(JudgeItem(id=case["id"], scenario=case, transcript=transcript))
        from stateset_agents.evaluation.judge_evidence import RecordedSupportJudge

        if isinstance(judge, RecordedSupportJudge):
            results = judge.score(
                items,
                context={"split": "validation", "optimizer_step": trainer.global_step},
            )
        else:
            results = judge.score(items)
        if inspect.isawaitable(results):
            results = await results
        return _JudgedValidation(results, getattr(judge, "last_evidence", None))

    return evaluate


class JudgeCheckpointSelector:
    """Retain the baseline unless independent validation strictly improves.

    ``evaluate(trainer, validation_cases)`` returns one two-pass JudgeResult per
    case, using current policy weights and a pinned judge. The evaluator owns
    generation and must not substitute cached base-model completions. Test cases
    are never passed to it. Register this object in ``trainer.callbacks`` and
    deploy ``selected_checkpoint()`` after training, rather than the last save.

    This callback starts a fresh selection study and deliberately rejects native
    multi-rank runs and resumed trainers: collective-safe judging and persisted
    callback recovery need separate implementations. The receipt binds saved
    bytes; it is validation evidence, not production promotion approval.
    """

    fail_on_error = True

    def __init__(
        self,
        *,
        evaluate: Callable[..., Any],
        train_cases: Sequence[Mapping[str, Any]],
        validation_cases: Sequence[Mapping[str, Any]],
        test_cases: Sequence[Mapping[str, Any]],
        judge_model: str,
        judge_revision: str,
        rubric: str,
        every_updates: int = 25,
        max_evaluations: int = 12,
        max_mean_disagreement: float = 0.05,
        min_improvement: float = 0.0,
        require_evidence: bool = False,
    ) -> None:
        if type(require_evidence) is not bool:
            raise ValueError("require_evidence must be boolean")
        for name, value in (
            ("every_updates", every_updates),
            ("max_evaluations", max_evaluations),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for threshold_name, threshold in (
            ("max_mean_disagreement", max_mean_disagreement),
            ("min_improvement", min_improvement),
        ):
            if (
                isinstance(threshold, bool)
                or not math.isfinite(threshold)
                or not 0 <= threshold <= 1
            ):
                raise ValueError(f"{threshold_name} must be finite in [0, 1]")
        if not judge_model.strip() or not re.fullmatch(r"[0-9a-f]{40}", judge_revision):
            raise ValueError(
                "Judge model and immutable 40-character revision are required"
            )
        if rubric != rubric_revision():
            raise ValueError("Judge rubric differs from the installed pinned rubric")
        self.split_hashes = validate_case_separation(
            {"train": train_cases, "validation": validation_cases, "test": test_cases}
        )
        self.validation_cases = copy.deepcopy(list(validation_cases))
        self.evaluate = evaluate
        self.every_updates = every_updates
        self.max_evaluations = max_evaluations
        self.max_mean_disagreement = float(max_mean_disagreement)
        self.min_improvement = float(min_improvement)
        self.require_evidence = require_evidence
        self.judge = {
            "model": judge_model,
            "revision": judge_revision,
            "rubric_revision": rubric,
        }
        self._trainer: Any = None
        self._root: Path
        self._receipt: Path
        self._reports: list[dict[str, Any]] = []
        self._selected: dict[str, Any] | None = None
        self._attempts = 0

    async def on_training_start(self, trainer: Any, config: Any) -> None:
        """Measure and save the untrained baseline before the first update."""
        torch = sys.modules.get("torch")
        distributed = getattr(torch, "distributed", None)
        if getattr(trainer, "world_size", 1) != 1 or (
            distributed is not None and distributed.is_initialized()
        ):
            raise ValueError(
                "Judge checkpoint callback supports single-process training only"
            )
        if (
            self._trainer is not None
            or getattr(trainer, "global_step", 0) != 0
            or getattr(config, "resume_from_checkpoint", None)
        ):
            raise ValueError("Judge selection requires a fresh, non-resumed study")
        self._root = Path(config.output_dir).resolve()
        self._receipt = self._root / "judge-selection.json"
        if self._receipt.exists():
            raise ValueError(
                "Judge selection receipt already exists; use a fresh output directory"
            )
        self._trainer = trainer
        await self._measure(0)

    async def on_optimizer_step(self, step: int, metrics: dict[str, Any]) -> None:
        """Use committed update counts, including task/final accumulation flushes."""
        if type(step) is not int or step < 0 or not self._reports:
            raise ValueError("Judge selection received an invalid optimizer step")
        if step != self._trainer.global_step:
            raise ValueError("Judge selection step differs from current policy step")
        previous = self._reports[-1]["step"]
        if step < previous:
            raise ValueError("Judge selection optimizer step moved backwards")
        if step // self.every_updates > previous // self.every_updates:
            await self._measure(step)

    async def on_training_end(self, metrics: dict[str, Any]) -> None:
        """Measure a final partial interval without replacing a better checkpoint."""
        step = metrics["final_step"]
        if (
            type(step) is not int
            or not self._reports
            or step < self._reports[-1]["step"]
        ):
            raise ValueError("Invalid final step for judge selection")
        if step > self._reports[-1]["step"]:
            await self._measure(step)
        self.selected_checkpoint()

    async def _measure(self, step: int) -> None:
        if step != self._trainer.global_step:
            raise ValueError("Judge selection step differs from current policy step")
        if self._attempts >= self.max_evaluations:
            raise RuntimeError("Judge validation evaluation budget exhausted")
        self._attempts += 1
        cases = copy.deepcopy(self.validation_cases)
        # An evaluator cannot silently mutate the selector's held-out inputs.
        from stateset_agents.evaluation.isolation import isolated_evaluation

        seed = getattr(self._trainer.config, "seed", 42)
        with isolated_evaluation(
            seed=42 if seed is None else seed, model=self._trainer.agent.model
        ):
            results = self.evaluate(self._trainer, cases)
            if inspect.isawaitable(results):
                results = await results
        evidence = copy.deepcopy(getattr(results, "evidence", None))
        results = list(results)
        if step != self._trainer.global_step:
            raise ValueError("Policy step changed during judge validation")
        if {case["id"]: content_hash(case) for case in cases} != self.split_hashes[
            "validation"
        ]:
            raise ValueError("Judge evaluator mutated held-out validation cases")
        ids = [result.id for result in results]
        if len(ids) != len(set(ids)) or set(ids) != set(
            self.split_hashes["validation"]
        ):
            raise ValueError(
                "Judge validation must cover each held-out case exactly once"
            )
        scores = [checked_judgement(result, 1.0) for result in results]
        disagreement = math.fsum(result.disagreement for result in results) / len(
            results
        )
        if disagreement > self.max_mean_disagreement:
            raise ValueError(
                "Validation judge exceeds its mean self-disagreement limit"
            )
        mean = math.fsum(scores) / len(scores)
        report = {
            "step": step,
            "mean_score": mean,
            "mean_disagreement": disagreement,
            "items": [asdict(result) for result in results],
        }
        if evidence is not None:
            report["evidence"] = self._verify_evidence(evidence, step, report["items"])
        elif self.require_evidence:
            raise ValueError("Judge selection requires replayable transcript evidence")
        selected = self._selected
        if selected is None or mean > selected["mean_score"] + self.min_improvement:
            name = f"judge-selected-step-{step}"
            path = self._root / name
            if path.exists():
                raise ValueError("Refusing to overwrite a judge-selected checkpoint")
            await self._trainer.save_checkpoint(checkpoint_name=name)
            selected = {
                "step": step,
                "mean_score": mean,
                "path": name,
                "artifact_sha256": artifact_digest(path),
            }
        document = {
            "schema_version": 1,
            "split": "validation",
            "production_approved": False,
            "judge": self.judge,
            "selection_config": {
                "every_updates": self.every_updates,
                "max_evaluations": self.max_evaluations,
                "attempted_evaluations": self._attempts,
                "max_mean_disagreement": self.max_mean_disagreement,
                "min_improvement": self.min_improvement,
                "seed": seed,
                "require_evidence": self.require_evidence,
            },
            "case_hashes": self.split_hashes,
            "selected": selected,
            "evaluations": [*self._reports, report],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", dir=self._root, prefix=".judge-selection-", delete=False
        ) as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write("\n")
            temporary = Path(stream.name)
        try:
            os.replace(temporary, self._receipt)
        finally:
            temporary.unlink(missing_ok=True)
        self._reports.append(report)
        self._selected = selected

    def selected_checkpoint(self) -> Path:
        """Return the selected artifact only while its bytes match the receipt."""
        if self._selected is None:
            raise ValueError("No successfully judged checkpoint has been selected")
        path: Path = self._root / str(self._selected["path"])
        if artifact_digest(path) != self._selected["artifact_sha256"]:
            raise ValueError("Judge-selected checkpoint bytes changed after evaluation")
        selected_report = next(
            report
            for report in self._reports
            if report["step"] == self._selected["step"]
        )
        if "evidence" in selected_report:
            self._verify_evidence(
                selected_report["evidence"],
                selected_report["step"],
                selected_report["items"],
            )
        elif self.require_evidence:
            raise ValueError("Selected checkpoint lacks replayable judge evidence")
        return path

    def _verify_evidence(
        self, reference: dict[str, Any], step: int, results: Any
    ) -> dict[str, Any]:
        from stateset_agents.evaluation.judge_evidence import replay_judge_evidence

        path = Path(reference["path"])
        if not path.is_absolute():
            path = self._root / path
        if not path.resolve().is_relative_to(self._root):
            raise ValueError(
                "Judge evidence must remain inside the study output directory"
            )
        replay = replay_judge_evidence(
            path, expected_manifest_sha256=reference["manifest_sha256"]
        )
        if (
            replay["judge"] != self.judge
            or replay["context"] != {"split": "validation", "optimizer_step": step}
            or content_hash(replay["results"]) != content_hash(results)
            or {row["id"]: content_hash(row["scenario"]) for row in replay["items"]}
            != self.split_hashes["validation"]
        ):
            raise ValueError("Recorded judge evidence differs from validation results")
        return {
            "path": path.relative_to(self._root).as_posix(),
            "manifest_sha256": reference["manifest_sha256"],
        }
