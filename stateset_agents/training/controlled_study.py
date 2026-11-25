"""Controlled Multi-Model Frontier RL Study, Sweep & Appendix Harness.

Codifies the controlled closed-universe RL experimental stack:
- **Environments:** Closed-universe :class:`ClientMatterSandbox` workspaces with
  key and peripheral documents, sandboxed search/read/draft tools, execution
  limits, and binary pass/fail expert rubrics.
- **Reward:** :class:`HierarchicalRubricReward` (weighted sum of criteria fraction,
  full legal issue-group fraction, and all-criteria bonus; explicitly zero length
  penalty) graded by an LLM judge (or deterministic offline judge for CI).
- **Optimization:** Rank-64 LoRA adapter trained with **CISPO** per-token clipped
  importance weights for async sampling, using **Dr. GRPO** unscaled group-mean
  advantage (``r_i - mean(r_group)``) and constant-length token loss normalization
  (``dr_cispo`` objective preset).
- **Infrastructure:** Controlled 5-model fleet — ``Muse Spark Open``,
  ``Nemotron 3.5 Super VL``, ``Beam 1.0``, and ``GLM-5.3`` trained on **River**
  (or StateSet distributed trainer); ``Inkling 2 Small`` trained on **Tinker**
  (supporting fine-tuning prior to weight release); all five models executing
  rollouts inside **StateSet Sandboxes** under the default LAB harness.
- **Controls:** Shared environment, reward, judge, and optimizer controls across
  all models; learning-rate & rollout-group-size hyperparameter sweep on
  ``Nemotron 3.5 Super VL`` transferred verbatim to the remaining four models;
  machine-readable manifest + publication-ready Markdown Appendix exporter.

Zero eager ``torch`` imports.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stateset_agents.core.client_matter_sandbox import (
    ClientMatterTask,
    default_client_matter_tasks,
)
from stateset_agents.core.sandbox_pool import MultiDomainSandboxPool, SandboxEpisodeSpec
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.rewards.rubric_reward import RubricRewardConfig

VALID_OBJECTIVE_PRESETS: tuple[str, ...] = (
    "grpo",
    "dr_grpo",
    "bnpo",
    "dapo",
    "gspo",
    "gspo_token",
    "gepo",
    "rloo",
    "reinforce_pp_baseline",
    "cispo",
    "dr_cispo",
    "ppo",
)


@dataclass(frozen=True)
class OptimizationStackConfig:
    """Shared LoRA + Dr. GRPO / CISPO optimization controls."""

    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    algorithm: str = "cispo"
    advantage_normalization: str = "dr_grpo"  # reward - group_mean (no std division)
    loss_normalization: str = "dr_grpo"  # constant-length token sum (no length bias)
    objective_preset: str = "dr_cispo"
    clip_low: float = 0.2
    clip_high: float = 0.2
    async_is_clip_max: float = 6.0
    kl_coef: float = 0.0
    learning_rate: float = 1e-4
    rollout_group_size: int = 8
    max_completion_length: int = 4096
    grad_clip_norm: float = 1.0
    optimizer: str = "adamw"

    def __post_init__(self) -> None:
        if self.lora_rank < 1:
            raise ValueError("lora_rank must be >= 1")
        if self.rollout_group_size < 2:
            raise ValueError("rollout_group_size must be >= 2 for group mean centering")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and > 0")
        if self.objective_preset not in VALID_OBJECTIVE_PRESETS:
            raise ValueError(
                f"Unknown objective_preset {self.objective_preset!r}; "
                f"expected one of {sorted(VALID_OBJECTIVE_PRESETS)}"
            )
        if not math.isfinite(self.async_is_clip_max) or self.async_is_clip_max <= 0.0:
            raise ValueError("async_is_clip_max must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StudyModelSpec:
    """Infrastructure specification for one model in the controlled 5-model study."""

    display_name: str
    model_id: str
    training_backend: str  # e.g. "river" or "tinker"
    rollout_backend: str = "stateset_sandboxes"
    harness: str = "lab_default"
    pre_weight_release: bool = False
    is_sweep_anchor: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HyperparameterSweepGrid:
    """Standard-range sweep conducted exclusively on the anchor model."""

    anchor_model: str = "Nemotron 3.5 Super VL"
    learning_rates: tuple[float, ...] = (3e-5, 1e-4, 3e-4)
    rollout_group_sizes: tuple[int, ...] = (4, 8, 16)

    def __post_init__(self) -> None:
        if not self.learning_rates or not self.rollout_group_sizes:
            raise ValueError(
                "Sweep grid must include learning_rates and rollout_group_sizes"
            )
        if any(lr <= 0.0 or not math.isfinite(lr) for lr in self.learning_rates):
            raise ValueError("All sweep learning_rates must be finite and > 0")
        if any(gs < 2 for gs in self.rollout_group_sizes):
            raise ValueError("All sweep rollout_group_sizes must be >= 2")

    def candidates(self) -> list[tuple[float, int]]:
        return [
            (float(lr), int(gs))
            for lr in self.learning_rates
            for gs in self.rollout_group_sizes
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_model": self.anchor_model,
            "learning_rates": list(self.learning_rates),
            "rollout_group_sizes": list(self.rollout_group_sizes),
        }


@dataclass(frozen=True)
class SweepTrialResult:
    """Outcome of one hyperparameter trial on the anchor model."""

    anchor_model: str
    learning_rate: float
    rollout_group_size: int
    validation_reward: float
    criteria_pass_rate: float = 0.0
    issue_group_pass_rate: float = 0.0
    all_pass_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SweepSelection:
    """Winning hyperparameter configuration transferred to the entire model fleet."""

    anchor_model: str
    selected_learning_rate: float
    selected_rollout_group_size: int
    selected_validation_reward: float
    trials: tuple[SweepTrialResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_model": self.anchor_model,
            "selected_learning_rate": self.selected_learning_rate,
            "selected_rollout_group_size": self.selected_rollout_group_size,
            "selected_validation_reward": self.selected_validation_reward,
            "trials": [t.to_dict() for t in self.trials],
        }


@dataclass(frozen=True)
class ControlledModelRunPlan:
    """Fully resolved, control-verified training plan for one model in the study."""

    model: StudyModelSpec
    optimization: OptimizationStackConfig
    reward: RubricRewardConfig
    judge_model: str
    judge_temperature: float
    matter_ids: tuple[str, ...]
    max_steps: int
    seed: int
    transferred_from_anchor: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "optimization": self.optimization.to_dict(),
            "reward": self.reward.to_dict(),
            "judge_model": self.judge_model,
            "judge_temperature": self.judge_temperature,
            "matter_ids": list(self.matter_ids),
            "max_steps": self.max_steps,
            "seed": self.seed,
            "transferred_from_anchor": self.transferred_from_anchor,
        }


def default_frontier_study_models(
    *,
    primary_training_backend: str = "river",
    pre_release_backend: str = "tinker",
) -> tuple[StudyModelSpec, ...]:
    """Return the exact 5-model controlled study fleet."""
    return (
        StudyModelSpec(
            display_name="Muse Spark Open",
            model_id="meta-models/Muse-Spark-Open-816B-A42B",
            training_backend=primary_training_backend,
            rollout_backend="stateset_sandboxes",
            harness="lab_default",
            notes="Trained via managed RL backend with StateSet Sandboxes rollouts.",
        ),
        StudyModelSpec(
            display_name="Nemotron 3.5 Super VL",
            model_id="nvidia/NVIDIA-Nemotron-3.5-Super-VL",
            training_backend=primary_training_backend,
            rollout_backend="stateset_sandboxes",
            harness="lab_default",
            is_sweep_anchor=True,
            notes="Hyperparameter sweep anchor (LR & rollout group size transferred to fleet).",
        ),
        StudyModelSpec(
            display_name="Beam 1.0",
            model_id="beam/Beam-1.0",
            training_backend=primary_training_backend,
            rollout_backend="stateset_sandboxes",
            harness="lab_default",
            notes="Trained via managed RL backend with transferred hyperparameters.",
        ),
        StudyModelSpec(
            display_name="GLM-5.3",
            model_id="zai-org/GLM-5.3",
            training_backend=primary_training_backend,
            rollout_backend="stateset_sandboxes",
            harness="lab_default",
            notes="Trained via managed RL backend with transferred hyperparameters.",
        ),
        StudyModelSpec(
            display_name="Inkling 2 Small",
            model_id="thinkingmachines/Inkling-2-Small",
            training_backend=pre_release_backend,
            rollout_backend="stateset_sandboxes",
            harness="lab_default",
            pre_weight_release=True,
            notes="Trained via Tinker, supporting fine-tuning prior to public weight release.",
        ),
    )


@dataclass(frozen=True)
class ControlledStudyConfig:
    """Top-level specification of the controlled 5-model RL study stack."""

    study_name: str = "stateset-closed-universe-legal-rl-v1"
    models: tuple[StudyModelSpec, ...] = field(
        default_factory=default_frontier_study_models
    )
    optimization: OptimizationStackConfig = field(
        default_factory=OptimizationStackConfig
    )
    reward: RubricRewardConfig = field(default_factory=RubricRewardConfig)
    sweep_grid: HyperparameterSweepGrid = field(default_factory=HyperparameterSweepGrid)
    judge_model: str = "claude-sonnet-4-6"
    judge_temperature: float = 0.0
    max_steps: int = 12
    seed: int = 42
    tasks: tuple[ClientMatterTask, ...] = field(
        default_factory=lambda: tuple(default_client_matter_tasks())
    )

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("ControlledStudyConfig requires at least one model")
        anchors = [
            m for m in self.models if m.display_name == self.sweep_grid.anchor_model
        ]
        if len(anchors) != 1:
            raise ValueError(
                f"Sweep anchor model {self.sweep_grid.anchor_model!r} must appear "
                "exactly once in models."
            )
        if self.reward.length_penalty != 0.0:
            raise ValueError("Controlled study enforces zero explicit length penalty.")
        if not self.tasks:
            raise ValueError("ControlledStudyConfig requires at least one matter task.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_name": self.study_name,
            "models": [m.to_dict() for m in self.models],
            "optimization": self.optimization.to_dict(),
            "reward": self.reward.to_dict(),
            "sweep_grid": self.sweep_grid.to_dict(),
            "judge_model": self.judge_model,
            "judge_temperature": self.judge_temperature,
            "max_steps": self.max_steps,
            "seed": self.seed,
            "tasks": [t.to_summary() for t in self.tasks],
        }


SweepEvaluatorFn = Callable[[StudyModelSpec, float, int], SweepTrialResult]


def run_anchor_hyperparameter_sweep(
    config: ControlledStudyConfig,
    evaluator: SweepEvaluatorFn,
) -> SweepSelection:
    """Sweep learning rate and rollout group size exclusively on the anchor model.

    Selects the candidate with the highest ``validation_reward`` (deterministic
    tie-break: lower learning rate, then smaller rollout group size).
    """
    anchor = next(
        m for m in config.models if m.display_name == config.sweep_grid.anchor_model
    )
    trials: list[SweepTrialResult] = []
    for lr, group_size in config.sweep_grid.candidates():
        trial = evaluator(anchor, lr, group_size)
        if trial.anchor_model != anchor.display_name:
            raise ValueError(
                f"Sweep trial ran on {trial.anchor_model!r}, expected anchor {anchor.display_name!r}"
            )
        if not math.isfinite(trial.validation_reward):
            raise ValueError("Sweep validation_reward must be finite")
        trials.append(trial)

    best = min(
        trials,
        key=lambda t: (
            -float(t.validation_reward),
            float(t.learning_rate),
            int(t.rollout_group_size),
        ),
    )
    return SweepSelection(
        anchor_model=anchor.display_name,
        selected_learning_rate=best.learning_rate,
        selected_rollout_group_size=best.rollout_group_size,
        selected_validation_reward=best.validation_reward,
        trials=tuple(trials),
    )


def build_controlled_run_plans(
    config: ControlledStudyConfig,
    selection: SweepSelection | None = None,
) -> list[ControlledModelRunPlan]:
    """Transfer anchor-selected hyperparameters verbatim across all models."""
    opt = config.optimization
    if selection is not None:
        opt = replace(
            opt,
            learning_rate=selection.selected_learning_rate,
            rollout_group_size=selection.selected_rollout_group_size,
        )
    matter_ids = tuple(t.matter_id for t in config.tasks)
    plans = [
        ControlledModelRunPlan(
            model=spec,
            optimization=opt,
            reward=config.reward,
            judge_model=config.judge_model,
            judge_temperature=config.judge_temperature,
            matter_ids=matter_ids,
            max_steps=config.max_steps,
            seed=config.seed,
            transferred_from_anchor=config.sweep_grid.anchor_model,
        )
        for spec in config.models
    ]
    validate_study_controls(plans)
    return plans


def validate_study_controls(plans: list[ControlledModelRunPlan]) -> dict[str, Any]:
    """Assert strict control invariance across every model in the study."""
    if not plans:
        raise ValueError("At least one model run plan is required.")
    ref = plans[0]
    for plan in plans:
        if plan.reward.length_penalty != 0.0:
            raise ValueError(
                f"Model {plan.model.display_name!r} violates zero length penalty control."
            )
        if plan.model.rollout_backend != "stateset_sandboxes":
            raise ValueError(
                f"Model {plan.model.display_name!r} must use 'stateset_sandboxes' rollouts."
            )
        if plan.model.harness != ref.model.harness:
            raise ValueError(
                f"Harness mismatch on {plan.model.display_name!r}: "
                f"{plan.model.harness!r} != {ref.model.harness!r}"
            )
        if plan.optimization != ref.optimization:
            raise ValueError(
                f"Optimization hyperparameter drift on {plan.model.display_name!r}."
            )
        if plan.reward != ref.reward:
            raise ValueError(f"Reward weight drift on {plan.model.display_name!r}.")
        if (
            plan.judge_model != ref.judge_model
            or plan.judge_temperature != ref.judge_temperature
        ):
            raise ValueError(
                f"Judge configuration drift on {plan.model.display_name!r}."
            )
        if plan.matter_ids != ref.matter_ids or plan.max_steps != ref.max_steps:
            raise ValueError(
                f"Environment control drift on {plan.model.display_name!r}."
            )

    return {
        "controls_verified": True,
        "model_count": len(plans),
        "shared_lora_rank": ref.optimization.lora_rank,
        "shared_objective": ref.optimization.objective_preset,
        "shared_learning_rate": ref.optimization.learning_rate,
        "shared_rollout_group_size": ref.optimization.rollout_group_size,
        "shared_harness": ref.model.harness,
        "shared_rollout_backend": ref.model.rollout_backend,
        "length_penalty": ref.reward.length_penalty,
        "matter_count": len(ref.matter_ids),
    }


def compute_dr_grpo_group_advantages(rewards: list[float]) -> list[float]:
    """Compute unscaled Dr. GRPO advantages: ``A_i = R_i - mean(R_group)``."""
    if not rewards:
        return []
    mean_r = sum(float(r) for r in rewards) / len(rewards)
    return [round(float(r) - mean_r, 6) for r in rewards]


async def run_offline_sandbox_stack_smoke(
    config: ControlledStudyConfig | None = None,
) -> dict[str, Any]:
    """Run an offline end-to-end closed-universe Client Matter rollout group proof.

    Exercises :class:`MultiDomainSandboxPool` with ``domain="client_matter"`` and
    Dr. GRPO ``group_mean`` advantage centering over a group of rollouts (comparing
    a high-quality multi-doc draft vs. an incomplete draft with missing citations).
    """
    cfg = config or ControlledStudyConfig()
    pool = MultiDomainSandboxPool(max_concurrency=4)
    specs = [
        SandboxEpisodeSpec(
            episode_id="matter-strong-rollout",
            domain="client_matter",
            scenario={"matter_id": "matter_acme_assignment", "variant": "strong"},
            max_turns=6,
        ),
        SandboxEpisodeSpec(
            episode_id="matter-partial-rollout",
            domain="client_matter",
            scenario={"matter_id": "matter_acme_assignment", "variant": "partial"},
            max_turns=6,
        ),
    ]

    async def _scripted_policy(
        obs: str, meta: dict[str, Any], turn_idx: int
    ) -> ConversationTurn:
        del obs
        variant = str((meta.get("scenario") or {}).get("variant") or "strong")
        if turn_idx == 0:
            return ConversationTurn(
                role="assistant",
                content='<tool_call>{"name": "list_matter_files", "arguments": {}}</tool_call>',
            )
        if turn_idx == 1:
            return ConversationTurn(
                role="assistant",
                content='<tool_call>{"name": "read_document", "arguments": {"filename": "first_amendment_2024.txt"}}</tool_call>',
            )
        if turn_idx == 2:
            if variant == "strong":
                draft = (
                    "Under Master Lease Section 12.2 a merger is deemed an assignment "
                    "requiring consent, but First Amendment Section 3 establishes a "
                    "Permitted Transfer carve-out eliminating prior consent because "
                    "Vertex Automation Corp.'s $62M tangible net worth exceeds the $50M "
                    "threshold. Acme must still deliver 15 days prior to closing written "
                    "notice and certified financial statements, and pay the reduced "
                    "$1,500 fee under First Amendment Section 4. Per Harborview counsel's "
                    "October 2026 email, closing without notice triggers Section 18.1 "
                    "default remedies (including termination) subject to a 15 business "
                    "days cure period after written notice."
                )
            else:
                draft = (
                    "The First Amendment creates a Permitted Transfer carve-out so no "
                    "landlord consent is needed for Vertex's $62M net worth (> $50M)."
                )
            payload = json.dumps(
                {
                    "name": "draft_work_product",
                    "arguments": {
                        "title": "Acme Lease Assignment Memorandum",
                        "content": draft,
                    },
                }
            )
            return ConversationTurn(
                role="assistant", content=f"<tool_call>{payload}</tool_call>"
            )
        if turn_idx == 3:
            return ConversationTurn(
                role="assistant",
                content='<tool_call>{"name": "signal_complete", "arguments": {"summary": "Memo finalized."}}</tool_call>',
            )
        return ConversationTurn(role="assistant", content="Completed memorandum.")

    batch = await pool.run_batch(specs, _scripted_policy, advantage_mode="group_mean")
    rewards = [float(ep["reward"]) for ep in batch["episodes"]]
    return {
        "study_name": cfg.study_name,
        "advantage_mode": "group_mean",
        "episode_count": batch["episode_count"],
        "episodes": batch["episodes"],
        "group_advantages": compute_dr_grpo_group_advantages(rewards),
        "strong_beats_partial": rewards[0] > rewards[1],
    }


def render_appendix_markdown(
    config: ControlledStudyConfig,
    plans: list[ControlledModelRunPlan],
    *,
    selection: SweepSelection | None = None,
) -> str:
    """Render the publication-ready Stack overview & Training Configuration Appendix."""
    controls = validate_study_controls(plans)
    opt = plans[0].optimization
    rew = plans[0].reward
    primary_models = [
        p.model.display_name for p in plans if not p.model.pre_weight_release
    ]
    pre_release_models = [
        p.model.display_name for p in plans if p.model.pre_weight_release
    ]
    primary_backend = next(
        (p.model.training_backend for p in plans if not p.model.pre_weight_release),
        "river",
    )
    pre_release_backend = next(
        (p.model.training_backend for p in plans if p.model.pre_weight_release),
        "tinker",
    )

    lines = [
        f"# Training Stack & Configuration Appendix — `{config.study_name}`",
        "",
        "## Stack Summary",
        "",
        "### Environments",
        (
            "Each task places an agent in a closed-universe client matter with a short "
            "instruction requesting work. Each client matter contains both key controlling "
            "documents and peripheral distractor documents, requiring cross-file search "
            "and reading in a confined sandboxed workspace (`list_matter_files`, "
            "`search_matter`, `read_document`, `draft_work_product`, `read_draft`, "
            f"`signal_complete`). Episodes end upon `signal_complete` or the `{config.max_steps}`-step "
            "execution limit. Deliverables are graded against an expert rubric of binary "
            "pass/fail criteria organized into legal issue groups."
        ),
        "",
        "### Reward",
        (
            f"An LLM judge (`{config.judge_model}`, temperature `{config.judge_temperature}`) "
            "marks each rubric criterion pass or fail. Reward is the weighted sum of "
            f"`{rew.criteria_weight:.2f}` × (fraction of criteria passed), "
            f"`{rew.issue_group_weight:.2f}` × (fraction of legal issue groups passed in full), "
            f"and a `{rew.all_pass_bonus:.2f}` bonus for passing every criterion. "
            f"Explicit length penalty: **`{rew.length_penalty}`** (none)."
        ),
        "",
        "### Optimization",
        (
            f"Every model trains a **rank-{opt.lora_rank} LoRA** adapter (`alpha={opt.lora_alpha}`) "
            f"with **{opt.algorithm.upper()}** (`{opt.objective_preset}` preset), using **Dr. GRPO** "
            "unscaled group-mean advantage (`r_i - mean(r_group)`) and constant-length token "
            f"loss normalization (`max_completion_length={opt.max_completion_length}`), plus "
            f"per-token importance weight clipping (`clip=[{1.0 - opt.clip_low:.2f}, "
            f"{1.0 + opt.clip_high:.2f}]`, async cap `{opt.async_is_clip_max}`) to account "
            "for asynchronous rollout sampling."
        ),
        "",
        "### Infrastructure",
        (
            f"We train {', '.join(primary_models)} with **`{primary_backend}`** and "
            f"{', '.join(pre_release_models) or 'N/A'} with **`{pre_release_backend}`** "
            "(supporting fine-tuning prior to public weight release). Across all "
            f"{len(plans)} models, rollouts execute inside **StateSet Sandboxes** "
            f"under the `{controls['shared_harness']}` harness."
        ),
        "",
        "---",
        "",
        "## Appendix A — Model & Infrastructure Matrix",
        "",
        "| Model | Checkpoint ID | Training Backend | Rollout Backend | Harness | LoRA Rank | LR | Group Size | Notes |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for plan in plans:
        lines.append(
            f"| {plan.model.display_name} | `{plan.model.model_id}` | `{plan.model.training_backend}` | "
            f"`{plan.model.rollout_backend}` | `{plan.model.harness}` | `{plan.optimization.lora_rank}` | "
            f"`{plan.optimization.learning_rate:.1e}` | `{plan.optimization.rollout_group_size}` | "
            f"{plan.model.notes} |"
        )

    lines.extend(
        [
            "",
            "## Appendix B — Shared Optimization, Reward & Control Settings",
            "",
            "| Control Parameter | Shared Value Across All Models |",
            "| --- | --- |",
            f"| Objective Preset | `{opt.objective_preset}` (`{opt.algorithm.upper()}` + Dr. GRPO) |",
            f"| Advantage Normalization | `{opt.advantage_normalization}` (`R_i - mean(R_group)`, unscaled) |",
            f"| Loss Normalization | `{opt.loss_normalization}` (`seq_sum_const`, `L_max={opt.max_completion_length}`) |",
            f"| Per-Token IS Clipping | `[{1.0 - opt.clip_low:.2f}, {1.0 + opt.clip_high:.2f}]` (async cap `{opt.async_is_clip_max}`) |",
            f"| LoRA Rank / Alpha / Dropout | `r={opt.lora_rank}` / `alpha={opt.lora_alpha}` / `dropout={opt.lora_dropout}` |",
            f"| Selected Learning Rate | `{opt.learning_rate:.1e}` (swept on `{config.sweep_grid.anchor_model}`) |",
            f"| Selected Rollout Group Size | `{opt.rollout_group_size}` (swept on `{config.sweep_grid.anchor_model}`) |",
            f"| Reward Weights | `criteria={rew.criteria_weight}`, `issue_groups={rew.issue_group_weight}`, `all_pass_bonus={rew.all_pass_bonus}` |",
            f"| Explicit Length Penalty | `{rew.length_penalty}` (disabled) |",
            f"| Judge Model & Temperature | `{config.judge_model}` @ `T={config.judge_temperature}` |",
            f"| Closed-Universe Matters | `{len(config.tasks)}` matters, `max_steps={config.max_steps}` |",
        ]
    )

    if selection is not None:
        lines.extend(
            [
                "",
                f"## Appendix C — Anchor Hyperparameter Sweep (`{selection.anchor_model}`)",
                "",
                "| Learning Rate | Rollout Group Size | Validation Reward | Criteria Pass | Issue-Group Pass | All-Pass | Selected |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
            ]
        )
        for t in selection.trials:
            is_best = (
                t.learning_rate == selection.selected_learning_rate
                and t.rollout_group_size == selection.selected_rollout_group_size
            )
            lines.append(
                f"| `{t.learning_rate:.1e}` | `{t.rollout_group_size}` | `{t.validation_reward:.4f}` | "
                f"`{t.criteria_pass_rate:.3f}` | `{t.issue_group_pass_rate:.3f}` | "
                f"`{t.all_pass_rate:.3f}` | {'✅' if is_best else ''} |"
            )

    lines.append("")
    return "\n".join(lines)


def export_study_bundle(
    output_dir: Path,
    config: ControlledStudyConfig,
    plans: list[ControlledModelRunPlan],
    *,
    selection: SweepSelection | None = None,
) -> dict[str, Any]:
    """Write ``study_manifest.json`` (SHA-256 attested) and ``appendix_training_config.md``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    controls = validate_study_controls(plans)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "study": config.to_dict(),
        "controls": controls,
        "sweep_selection": selection.to_dict() if selection is not None else None,
        "plans": [p.to_dict() for p in plans],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload["manifest_sha256"] = digest

    manifest_path = output_dir / "study_manifest.json"
    appendix_path = output_dir / "appendix_training_config.md"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    appendix_md = render_appendix_markdown(config, plans, selection=selection)
    appendix_path.write_text(appendix_md, encoding="utf-8")
    return {
        "manifest_path": str(manifest_path),
        "appendix_path": str(appendix_path),
        "manifest_sha256": digest,
        "controls": controls,
    }


def run_offline_sandbox_stack_smoke_sync(
    config: ControlledStudyConfig | None = None,
) -> dict[str, Any]:
    """Synchronous wrapper safe inside Jupyter/Colab or plain scripts."""
    from stateset_agents.utils.colab import _run_coro

    return dict(_run_coro(run_offline_sandbox_stack_smoke(config)))


__all__ = [
    "ControlledModelRunPlan",
    "ControlledStudyConfig",
    "HyperparameterSweepGrid",
    "OptimizationStackConfig",
    "StudyModelSpec",
    "SweepEvaluatorFn",
    "SweepSelection",
    "SweepTrialResult",
    "build_controlled_run_plans",
    "compute_dr_grpo_group_advantages",
    "default_frontier_study_models",
    "export_study_bundle",
    "render_appendix_markdown",
    "run_anchor_hyperparameter_sweep",
    "run_offline_sandbox_stack_smoke",
    "run_offline_sandbox_stack_smoke_sync",
    "validate_study_controls",
]
