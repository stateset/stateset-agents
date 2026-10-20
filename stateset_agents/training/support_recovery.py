"""Plan and wire a reviewed, recorded support-reward recovery study.

The CLI is a free local planner. It never creates a GPU or calls a judge.
Live execution requires a separately reviewed provider quote, image and runtime.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from stateset_agents import __version__
from stateset_agents.core.agent_config import AgentConfig
from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.evaluation.case_separation import validate_case_separation
from stateset_agents.evaluation.implementation import package_implementation
from stateset_agents.evaluation.support_judge import rubric_revision
from stateset_agents.training.config import TrainingConfig

DEFAULT_PROFILE = (
    Path(__file__).resolve().parents[1]
    / "config/presets/experiments/support_recovery_preflight6.json"
)


def load_recovery_profile(path: Path = DEFAULT_PROFILE) -> dict[str, Any]:
    """Validate an explicit recovery configuration before model allocation."""
    return validate_recovery_profile(json.loads(path.read_text()))


def validate_recovery_profile(profile: Any) -> dict[str, Any]:
    """Apply the same profile checks to file and Python factory entry points."""
    content_hash(profile)
    if (
        not isinstance(profile, dict)
        or profile.get("schema_version") != 1
        or profile.get("kind") != "stateset-support-recovery-profile"
    ):
        raise ValueError("Expected a version-one support recovery profile")
    if profile.get("framework_version") != __version__:
        raise ValueError("Recovery profile framework version differs from installation")
    agent = AgentConfig(**profile["agent"])
    if profile.get("validation_decoding") != {
        "do_sample": False,
        "temperature": 0.0,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }:
        raise ValueError(
            "Recovery validation requires the fixed greedy decoding policy"
        )
    config = TrainingConfig(**profile["training"])
    judge = profile["judge"]
    for pin in (agent.model_revision, judge.get("revision")):
        if (
            not isinstance(pin, str)
            or not re.fullmatch(r"[0-9a-f]{40}", pin)
            or pin == "0" * 40
        ):
            raise ValueError("Recovery models require nonzero immutable commit pins")
    if (agent.model_name, agent.model_revision) != (
        config.model_name,
        config.model_revision,
    ):
        raise ValueError("Agent and learner model identities differ")
    if config.beta <= 0 or config.use_reference_model is not True:
        raise ValueError("Recovery training requires an active reference KL penalty")
    if (
        config.objective != "gspo"
        or config.gradient_accumulation_steps != 1
        or config.num_gradient_updates != 1
    ):
        raise ValueError("Recovery workload requires GSPO and one update per batch")
    if config.continual_strategy != "none" or config.resume_from_checkpoint is not None:
        raise ValueError(
            "Recovery preflight requires a fresh study without continual replay"
        )
    if config.num_episodes < 1 or config.num_generations < 2:
        raise ValueError(
            "Recovery study needs positive batches and multiple generations"
        )
    if (
        type(config.max_steps_per_episode) is not int
        or config.max_steps_per_episode < 1
    ):
        raise ValueError("Recovery episodes require an explicit positive turn limit")
    if config.eval_steps <= config.num_episodes:
        raise ValueError("Recovery checkpoint selection must use the independent judge")
    if (
        agent.peft_config is None
        or agent.peft_config.get("lora_dropout") != 0
        or agent.trust_remote_code
    ):
        raise ValueError("Recovery adapters require zero dropout and native model code")
    workload = profile["workload"]
    if (
        type(workload.get("scenarios_per_update")) is not int
        or workload["scenarios_per_update"] < 1
        or workload.get("reward") != "support_conversation_v3"
    ):
        raise ValueError(
            "Recovery workload must explicitly select v3 and a batch width"
        )
    if (
        judge.get("rubric_revision") != rubric_revision()
        or judge.get("placement") != "separate_endpoint"
    ):
        raise ValueError("Pin the installed rubric and use a separately placed judge")
    if (
        type(judge.get("max_parse_retries")) is not int
        or judge["max_parse_retries"] < 0
    ):
        raise ValueError("Judge retries must be an explicit nonnegative integer")
    selection = profile["selection"]
    if selection.get("require_evidence") is not True:
        raise ValueError("Recovery selection requires replayable judge evidence")
    for name in ("every_updates", "max_evaluations"):
        if type(selection.get(name)) is not int or selection[name] < 1:
            raise ValueError("Selection cadence and budget must be positive integers")
    for name in ("max_mean_disagreement", "min_improvement"):
        value = selection.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("Selection score thresholds must be finite in [0, 1]")
    required = 1 + math.ceil(config.num_episodes / selection["every_updates"])
    if selection["max_evaluations"] < required:
        raise ValueError(
            "Judge evaluation budget cannot cover baseline and final checkpoint"
        )
    if set(profile["minimum_cases"]) != {"train", "validation", "test"}:
        raise ValueError("Recovery case partitions must be train, validation and test")
    for split in ("train", "validation", "test"):
        if (
            type(profile["minimum_cases"].get(split)) is not int
            or profile["minimum_cases"][split] < 32
        ):
            raise ValueError(
                "Every recovery split needs at least 32 distinct reviewed cases"
            )
    compute = profile["proposed_compute"]
    if (
        type(profile.get("model_parameter_count")) is not int
        or profile["model_parameter_count"] < 1
    ):
        raise ValueError(
            "Declare a positive model parameter count for storage planning"
        )
    ceiling = compute.get("proposed_spending_ceiling_usd")
    if (
        isinstance(ceiling, bool)
        or not isinstance(ceiling, (int, float))
        or not math.isfinite(ceiling)
        or ceiling <= 0
        or compute.get("spending_authorized") is not False
    ):
        raise ValueError(
            "Planner requires a positive proposed ceiling and no spending authority"
        )
    return copy.deepcopy(profile)


def validate_recovery_cases(
    data: Mapping[str, Any], profile: Mapping[str, Any]
) -> dict[str, Any]:
    """Check declared review, fresh-test attestation and input/source separation."""
    provenance = data.get("provenance", {})
    if (
        not isinstance(provenance, dict)
        or provenance.get("source") != "reviewed"
        or not isinstance(provenance.get("reviewer"), str)
        or not provenance["reviewer"].strip()
        or provenance.get("test_is_fresh") is not True
    ):
        raise ValueError(
            "Recovery cases need reviewed provenance and a fresh-test attestation"
        )
    for split, minimum in profile["minimum_cases"].items():
        cases = data.get(split)
        if not isinstance(cases, list) or len(cases) < minimum:
            raise ValueError(f"Recovery {split} needs at least {minimum} cases")
        for case in cases:
            if not isinstance(case, dict):
                raise ValueError("Recovery cases must be scenario objects")
            for field in ("id", "user_query", "expected_resolution"):
                if not isinstance(case.get(field), str) or not case[field].strip():
                    raise ValueError(f"Recovery scenario requires {field}")
            for field in ("policy_facts", "must_avoid"):
                if not isinstance(case.get(field), list) or any(
                    not isinstance(value, str) or not value.strip()
                    for value in case[field]
                ):
                    raise ValueError(
                        f"Recovery scenario requires a text list for {field}"
                    )
            if not case["policy_facts"]:
                raise ValueError("Recovery scenario must contain verified policy facts")
    validate_case_separation({split: data[split] for split in profile["minimum_cases"]})
    content_hash(data)
    return copy.deepcopy(dict(data))


def build_recovery_plan(
    profile: dict[str, Any], cases: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Describe workload bounds and remaining gates; never authorize a paid run."""
    profile = validate_recovery_profile(profile)
    data = validate_recovery_cases(cases, profile) if cases is not None else None
    from stateset_agents.training.loss_computation import _token_objective

    config = TrainingConfig(**profile["training"])
    objective = _token_objective(config).with_(
        entropy_coef=config.entropy_coef, kl="k3_token", kl_coef=config.beta
    )
    updates = profile["training"]["num_episodes"]
    validations = 1 + math.ceil(updates / profile["selection"]["every_updates"])
    prompts = (
        None
        if data is None
        else len(data["validation"])
        * validations
        * 2
        * (1 + profile["judge"]["max_parse_retries"])
    )
    # Conservative full policy + reference copies, plus one atomic save in flight.
    # Optimizer/activation/cache/driver overhead is not measured by this estimate.
    checkpoint_gib = (
        profile["model_parameter_count"] * 2 * 2 * (validations + 2) / 2**30
    )
    blockers = (
        []
        if data is not None
        else ["Reviewed train/validation/fresh-test cases are missing"]
    )
    blockers.extend(
        [
            "Current provider quote including storage, judge and full lifetime is missing",
            "Immutable launch image and live model/runtime conformance remain unverified",
            "Paid execution requires a separately reviewed launch contract",
        ]
    )
    return {
        "schema_version": 1,
        "kind": "stateset-support-recovery-plan",
        "protocol": profile["protocol"],
        "profile_sha256": content_hash(profile),
        "package_implementation_sha256": content_hash(package_implementation()),
        "dataset_sha256": content_hash(data) if data is not None else None,
        "case_counts": (
            {split: len(data[split]) for split in ("train", "validation", "test")}
            if data
            else None
        ),
        "optimizer_updates": updates,
        "resolved_objective": asdict(objective),
        "training_rollout_episodes": updates
        * profile["workload"]["scenarios_per_update"]
        * profile["training"]["num_generations"],
        "planned_validation_evaluations": validations,
        "validation_decoding": profile["validation_decoding"],
        "validation_judge_prompt_upper_bound": prompts,
        "test_judge_prompts_included": False,
        "checkpoint_disk_estimate_gib": checkpoint_gib,
        "storage_estimate_measured": False,
        "proposed_compute": profile["proposed_compute"],
        "estimated_total_cost_usd": None,
        "ready_for_paid_execution": False,
        "blockers": blockers,
    }


class RecoveryRuntimeGuard:
    """Reject missing KL anchors, active dropout and drift from the study profile."""

    fail_on_error = True

    def __init__(self, profile: dict[str, Any]) -> None:
        self.profile = validate_recovery_profile(profile)

    def validate(self, trainer: Any) -> None:
        """Check actual learner components before the next rollout/update."""
        import torch

        for name, expected in self.profile["training"].items():
            if getattr(trainer.config, name, None) != expected:
                raise ValueError(f"Recovery learner configuration differs: {name}")
        if asdict(trainer.agent.config) != asdict(AgentConfig(**self.profile["agent"])):
            raise ValueError("Recovery agent configuration drifted from the profile")
        reference = trainer.reference_model
        if reference is None or reference is trainer.agent.model:
            raise ValueError("Recovery KL requires an independent reference model")
        policy = dict(trainer.agent.model.named_parameters())
        anchor = dict(reference.named_parameters())
        if set(policy) != set(anchor) or not policy:
            raise ValueError("Recovery reference architecture differs from the policy")
        policy_storage = {parameter.data_ptr() for parameter in policy.values()}
        if any(module.training for module in reference.modules()):
            raise ValueError("Recovery reference must remain in evaluation mode")
        for name, parameter in anchor.items():
            if (
                parameter.requires_grad
                or parameter.shape != policy[name].shape
                or parameter.data_ptr() in policy_storage
                or parameter.dtype != policy[name].dtype
                or parameter.device != policy[name].device
            ):
                raise ValueError(
                    "Recovery reference must be frozen and separately stored"
                )
        for module in trainer.agent.model.modules():
            if (
                isinstance(module, torch.nn.modules.dropout._DropoutNd)
                and module.p != 0
            ):
                raise ValueError(
                    "Recovery likelihood passes require zero model dropout"
                )
        if getattr(trainer.reward_fn, "name", None) != "support_conversation_v3":
            raise ValueError("Recovery learner must use support reward v3")
        if trainer.optimizer is None or any(
            group.get("initial_lr", group["lr"])
            != self.profile["training"]["learning_rate"]
            for group in trainer.optimizer.param_groups
        ):
            raise ValueError(
                "Recovery optimizer learning rate differs from the profile"
            )

    def on_training_start(self, trainer: Any, config: Any) -> None:
        """Validate the live reference and optimizer before the first update."""
        self.validate(trainer)

    def on_training_end(self, metrics: dict[str, Any]) -> None:
        """Do not call skipped or missing updates a completed recovery run."""
        if metrics.get("final_step") != self.profile["training"]["num_episodes"]:
            raise ValueError(
                "Recovery study did not complete its optimizer-update budget"
            )


def create_recovery_trainer(
    agent: Any,
    environment: Any,
    *,
    profile: dict[str, Any],
    cases: Mapping[str, Any],
    judge: Any,
    output_dir: Path,
) -> Any:
    """Wire the native loop to four-scenario batches and recorded judge selection.

    This constructs a trainer without allocating a model or provisioning compute.
    Call initialize/train explicitly only in a reviewed, bounded execution setup.
    Caller-supplied agent/environment must implement the pinned model and support
    conversation protocol; CPU tests do not certify a full-size GPU backend.
    """
    from stateset_agents.evaluation.judge_evidence import RecordedSupportJudge
    from stateset_agents.rewards.support_conversation_reward import (
        SupportConversationRewardV3,
    )
    from stateset_agents.training.judge_selection import (
        JudgeCheckpointSelector,
        support_judge_evaluator,
    )
    from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

    profile = validate_recovery_profile(profile)
    data = validate_recovery_cases(cases, profile)
    if not isinstance(judge, RecordedSupportJudge) or judge.judge != {
        key: profile["judge"][key] for key in ("model", "revision", "rubric_revision")
    }:
        raise ValueError("Recovery study requires a matching recorded judge")
    if asdict(agent.config) != asdict(AgentConfig(**profile["agent"])):
        raise ValueError("Recovery agent configuration differs from the pinned profile")
    if judge.max_batches < profile["selection"]["max_evaluations"]:
        raise ValueError("Recorded judge batch budget is insufficient")
    if judge.max_parse_retries != profile["judge"]["max_parse_retries"]:
        raise ValueError("Recorded judge retry budget differs from the profile")
    config = TrainingConfig(**profile["training"])
    config.output_dir = str(output_dir)
    max_turns = config.max_steps_per_episode
    if max_turns is None:
        raise ValueError("Recovery episodes require a turn limit")
    guard = RecoveryRuntimeGuard(profile)
    selector = JudgeCheckpointSelector(
        evaluate=support_judge_evaluator(judge, environment, max_turns=max_turns),
        train_cases=data["train"],
        validation_cases=data["validation"],
        test_cases=data["test"],
        judge_model=profile["judge"]["model"],
        judge_revision=profile["judge"]["revision"],
        rubric=profile["judge"]["rubric_revision"],
        **profile["selection"],
    )
    rng = random.Random(config.seed)
    width = profile["workload"]["scenarios_per_update"]
    schedule = []
    remaining: list[dict[str, Any]] = []
    for _ in range(config.num_episodes):
        while len(remaining) < width:
            epoch = copy.deepcopy(data["train"])
            rng.shuffle(epoch)
            remaining.extend(epoch)
        schedule.append(remaining[:width])
        del remaining[:width]

    plan = build_recovery_plan(profile, data)
    plan["training_schedule"] = [[case["id"] for case in batch] for batch in schedule]
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (output_dir / "study-plan.json").open("x") as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
        stream.write("\n")

    class RecoveryTrainer(MultiTurnGRPOTrainer):
        judge_selector: JudgeCheckpointSelector

        def _get_training_scenarios(self) -> list[dict[str, Any]]:
            return [
                {**batch[0], "_support_batch": index}
                for index, batch in enumerate(schedule)
            ]

        def _get_eval_scenarios(self) -> list[dict[str, Any]]:
            return []

        async def _generate_training_rollouts(
            self, scenarios: list[dict[str, Any]]
        ) -> Any:
            guard.validate(self)
            groups = await super()._generate_training_rollouts(
                copy.deepcopy(schedule[scenarios[0]["_support_batch"]])
            )
            if len(groups) != width or any(
                len(group.trajectories) != config.num_generations for group in groups
            ):
                raise ValueError(
                    "Recovery rollout batch did not cover every requested scenario/generation"
                )
            return groups

    trainer = RecoveryTrainer(
        agent,
        environment,
        reward_fn=SupportConversationRewardV3(),
        config=config,
        callbacks=[guard, selector],
    )
    trainer.judge_selector = selector
    return trainer


def main() -> None:
    """Write a free preflight plan, including explicit blockers to paid execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = load_recovery_profile(args.profile)
    cases = json.loads(args.cases.read_text()) if args.cases else None
    plan = build_recovery_plan(profile, cases)
    with args.output.open("x") as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
