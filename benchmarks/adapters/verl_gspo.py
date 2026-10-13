#!/usr/bin/env python3
"""Matched-GSPO shootout adapter for upstream verl (volcengine/verl).

verl implements GSPO natively: ``actor_rollout_ref.actor.policy_loss.
loss_mode=gspo`` (``verl/trainer/ppo/core_algos.py::compute_policy_loss_gspo``,
sequence-level geometric-mean importance ratio, clipped with ``clip_ratio_low``
/ ``clip_ratio_high``). Training runs as ``python -m verl.trainer.main_ppo``
with Hydra overrides in verl's own venv; the LoRA weights are exported with
``python -m verl.model_merger`` and evaluated by the shared evaluator.

Verified against the verl 0.9.1 wheel source (config keys, loss mode, merger
CLI, file logger); not yet executed end to end on a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from . import _matched_gsm8k as matched
except ImportError:  # executed as a script: benchmarks/adapters is on sys.path
    import _matched_gsm8k as matched  # type: ignore[no-redef]

VERL_VERSION = "0.9.1"
FRAMEWORK_NAME = "verl"
PYTHON_ENV_VAR = "STATESET_VERL_PYTHON"
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]*")
_REWARD_KEYS = ("critic/score/mean", "critic/rewards/mean")


def hydra_value(value: Any) -> str:
    """Render one value in Hydra override grammar."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(hydra_value(item) for item in value) + "]"
    text = str(value)
    if _IDENTIFIER.fullmatch(text) and text not in {"true", "false", "null"}:
        return text
    escaped = text.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def build_overrides(
    config: Mapping[str, Any],
    *,
    seed: int,
    model_path: Path,
    train_file: Path,
    val_file: Path,
    reward_path: Path,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    """Map the matched protocol onto verl's Hydra config (dotted keys)."""
    max_steps = int(config["max_steps"])
    prompts = matched.prompts_per_step(config)
    return {
        # Group-normalized GRPO advantages, no KL anywhere (beta = 0).
        "algorithm.adv_estimator": "grpo",
        "algorithm.norm_adv_by_std_in_grpo": config["scale_rewards"] == "group",
        "algorithm.use_kl_in_reward": False,
        # Data: raw-text prompts (identity chat template) and the TRL split.
        "data.train_files": str(train_file),
        "data.val_files": str(val_file),
        "data.prompt_key": "prompt",
        "data.train_batch_size": prompts,
        "data.max_prompt_length": int(config["max_prompt_length"]),
        "data.max_response_length": int(config["max_completion_length"]),
        "data.filter_overlong_prompts": False,
        "data.truncation": "error",
        "data.shuffle": True,
        "data.seed": seed,
        # Policy model + LoRA.
        "actor_rollout_ref.model.path": str(model_path),
        "actor_rollout_ref.model.lora_rank": int(config["lora_r"]),
        "actor_rollout_ref.model.lora_alpha": int(config["lora_alpha"]),
        "actor_rollout_ref.model.target_modules": list(config["lora_target_modules"]),
        "actor_rollout_ref.model.enable_gradient_checkpointing": bool(
            config["gradient_checkpointing"]
        ),
        "actor_rollout_ref.model.custom_chat_template": (
            matched.RAW_PROMPT_CHAT_TEMPLATE
        ),
        # GSPO objective: sequence-level ratio, asymmetric clip, seq-mean loss.
        "actor_rollout_ref.actor.policy_loss.loss_mode": "gspo",
        "actor_rollout_ref.actor.loss_agg_mode": "seq-mean-token-mean",
        "actor_rollout_ref.actor.clip_ratio_low": float(config["epsilon"]),
        "actor_rollout_ref.actor.clip_ratio_high": float(config["epsilon_high"]),
        "actor_rollout_ref.actor.use_kl_loss": float(config["beta"]) != 0.0,
        "actor_rollout_ref.actor.kl_loss_coef": float(config["beta"]),
        "actor_rollout_ref.actor.entropy_coeff": 0.0,
        "actor_rollout_ref.actor.ppo_epochs": int(config["num_iterations"]),
        # One optimizer step per rollout batch (mini-batch = whole batch).
        "actor_rollout_ref.actor.ppo_mini_batch_size": prompts,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": int(
            config["per_device_train_batch_size"]
        ),
        "actor_rollout_ref.actor.data_loader_seed": seed,
        "actor_rollout_ref.actor.fsdp_config.seed": seed,
        "actor_rollout_ref.actor.optim.lr": float(config["learning_rate"]),
        "actor_rollout_ref.actor.optim.betas": [
            float(config["adam_beta1"]),
            float(config["adam_beta2"]),
        ],
        "actor_rollout_ref.actor.optim.weight_decay": float(config["weight_decay"]),
        "actor_rollout_ref.actor.optim.clip_grad": float(config["max_grad_norm"]),
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio": float(
            config["warmup_ratio"]
        ),
        "actor_rollout_ref.actor.optim.lr_scheduler_type": str(
            config["lr_scheduler_type"]
        ),
        "actor_rollout_ref.actor.optim.min_lr_ratio": 0.0,
        "actor_rollout_ref.actor.checkpoint.save_contents": ["model", "extra"],
        # Rollout (vLLM, single GPU, colocated with FSDP).
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.n": int(config["num_generations"]),
        "actor_rollout_ref.rollout.temperature": float(config["temperature"]),
        "actor_rollout_ref.rollout.top_p": float(config["top_p"]),
        "actor_rollout_ref.rollout.top_k": -1,
        "actor_rollout_ref.rollout.seed": seed,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.4,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": int(
            config["per_device_train_batch_size"]
        ),
        "actor_rollout_ref.rollout.load_format": "safetensors",
        "actor_rollout_ref.rollout.layered_summon": True,
        # Reward: the TRL adapter's exact-numeric scorer.
        "reward.custom_reward_function.path": str(reward_path),
        "reward.custom_reward_function.name": "compute_score",
        # Trainer: one GPU, exactly max_steps optimizer steps, final save.
        "trainer.nnodes": 1,
        "trainer.n_gpus_per_node": 1,
        "trainer.total_epochs": int(config["num_train_epochs"]),
        "trainer.total_training_steps": max_steps,
        "trainer.save_freq": max_steps,
        "trainer.test_freq": -1,
        "trainer.val_before_train": False,
        "trainer.critic_warmup": 0,
        "trainer.logger": ["console", "file"],
        "trainer.project_name": "stateset-shootout",
        "trainer.experiment_name": f"verl-gspo-seed{seed}",
        "trainer.default_local_dir": str(checkpoint_dir),
        "trainer.resume_mode": "disable",
    }


def render_overrides(overrides: Mapping[str, Any]) -> list[str]:
    """Hydra ``key=value`` arguments in a stable order."""
    return [f"{key}={hydra_value(value)}" for key, value in overrides.items()]


def protocol_deviations(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Protocol settings verl 0.9.1 cannot express on the FSDP path."""
    deviations: list[dict[str, Any]] = []
    if float(config["lora_dropout"]) != 0.0:
        deviations.append(
            matched.deviation(
                "lora_dropout",
                float(config["lora_dropout"]),
                0.0,
                "verl's FSDP engine builds peft.LoraConfig without a dropout "
                "field (workers/engine/fsdp/transformer_impl.py); only the "
                "Megatron path reads model.lora.dropout",
            )
        )
    return deviations


PROTOCOL_NOTES = [
    "verl's GSPO loss uses the GSPO-token form (stop-gradient sequence ratio x "
    "token ratio), which has the same gradient as sequence-level GSPO when the "
    "advantage is constant per sequence (true for outcome rewards).",
    "FSDP keeps fp32 master weights with bf16 compute; TRL holds the frozen "
    "base in bf16. Optimizer state covers only LoRA parameters in both.",
    "Prompt order comes from verl's own seeded sampler, not TRL's.",
    "The verl trainer and vLLM run in verl's venv (pinned torch/vLLM); "
    "evaluation runs in the harness venv like every other framework.",
]


def _rows_to_parquet(rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    from datasets import Dataset

    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(list(rows)).to_parquet(str(path))
    return path


def verl_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """verl RLHFDataset rows (chat-form prompt, rule reward with gold label)."""
    return [
        {
            "data_source": "stateset/gsm8k-matched",
            "prompt": [{"role": "user", "content": row["prompt"]}],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": matched.gold_label(row["gold_answer"]),
            },
            "extra_info": {"index": int(row["index"])},
        }
        for row in rows
    ]


def read_metrics(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """Parse verl's FileLogger JSONL (``{"step": n, "data": {...}}``)."""
    if not path.is_file():
        raise matched.AdapterFailure(f"verl wrote no metrics file: {path}")
    records: list[tuple[int, dict[str, Any]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        data = payload.get("data")
        step = payload.get("step")
        if isinstance(data, dict) and isinstance(step, int):
            records.append((step, data))
    return records


def summarize_metrics(
    records: Sequence[tuple[int, Mapping[str, Any]]],
) -> tuple[int, list[float], float | None, dict[str, float]]:
    """Steps completed, per-step reward means, allocator peak, extras."""
    steps = 0
    rewards: list[float] = []
    peak_gb: float | None = None
    throughput: list[float] = []
    for step, data in sorted(records, key=lambda item: item[0]):
        global_step = data.get("training/global_step", step)
        if isinstance(global_step, (int, float)):
            steps = max(steps, int(global_step))
        for key in _REWARD_KEYS:
            value = data.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                rewards.append(float(value))
                break
        for key, value in data.items():
            if key.endswith("max_memory_allocated_gb") and isinstance(
                value, (int, float)
            ):
                peak_gb = float(value) if peak_gb is None else max(peak_gb, value)
        value = data.get("perf/throughput")
        if isinstance(value, (int, float)):
            throughput.append(float(value))
    extras = (
        {"tokens_per_second_per_gpu_mean": sum(throughput) / len(throughput)}
        if throughput
        else {}
    )
    return steps, rewards, (peak_gb * 1024 if peak_gb is not None else None), extras


class VerlAdapter:
    """verl implementation of the matched-GSPO adapter contract."""

    name = FRAMEWORK_NAME
    pinned_version = VERL_VERSION

    @staticmethod
    def python(args: argparse.Namespace) -> str:
        return str(
            args.framework_python or os.environ.get(PYTHON_ENV_VAR) or sys.executable
        )

    def probe_command(self, args: argparse.Namespace) -> list[str]:
        return [self.python(args), "-c", matched.probe_code("verl")]

    def prepare(self, ctx: matched.LaunchContext) -> matched.LaunchPlan:
        data_dir = ctx.work_dir / "data"
        train_file = _rows_to_parquet(
            verl_rows(ctx.train_rows), data_dir / "train.parquet"
        )
        # verl always builds a validation dataset; it is never evaluated here
        # (test_freq=-1, val_before_train=false).
        val_file = _rows_to_parquet(
            verl_rows(ctx.train_rows[:8]), data_dir / "val.parquet"
        )
        overrides = build_overrides(
            ctx.config,
            seed=int(ctx.args.seed),
            model_path=ctx.model_path,
            train_file=train_file,
            val_file=val_file,
            reward_path=Path(matched.__file__).resolve(),
            checkpoint_dir=ctx.work_dir / "checkpoints",
        )
        env = dict(ctx.env)
        env["VERL_FILE_LOGGER_PATH"] = str(
            ctx.artifact_dir / "framework" / "metrics.jsonl"
        )
        env.setdefault("WANDB_MODE", "disabled")
        return matched.LaunchPlan(
            command=[
                self.python(ctx.args),
                "-m",
                "verl.trainer.main_ppo",
                *render_overrides(overrides),
            ],
            env=env,
            settings=overrides,
            deviations=protocol_deviations(ctx.config),
            notes=list(PROTOCOL_NOTES),
        )

    def collect(
        self,
        ctx: matched.LaunchContext,
        plan: matched.LaunchPlan,
        runtime: matched.Runtime,
    ) -> matched.TrainingOutcome:
        steps, rewards, peak_mb, extras = summarize_metrics(
            read_metrics(Path(plan.env["VERL_FILE_LOGGER_PATH"]))
        )
        matched.require_completed_steps(steps, ctx.config, self.name)
        matched.require_rewards(rewards, self.name)
        step_dir = ctx.work_dir / "checkpoints" / f"global_step_{steps}"
        actor_dir = step_dir / "actor"
        if not actor_dir.is_dir():
            raise matched.AdapterFailure(f"verl wrote no actor checkpoint: {actor_dir}")
        merged = ctx.work_dir / "merged"
        outcome = runtime.run_command(
            [
                self.python(ctx.args),
                "-m",
                "verl.model_merger",
                "merge",
                "--backend",
                "fsdp",
                "--local_dir",
                str(actor_dir),
                "--target_dir",
                str(merged),
            ],
            env=plan.env,
            log_path=ctx.artifact_dir / "framework" / "merge.log",
            timeout_seconds=3600,
        )
        if outcome.returncode != 0:
            raise matched.AdapterFailure(
                f"verl model merger exited {outcome.returncode}"
            )
        return matched.TrainingOutcome(
            steps_completed=steps,
            step_rewards=rewards,
            step_reward_stds=[],
            adapter_dir=matched.find_peft_adapter(merged / "lora_adapter"),
            framework_peak_vram_mb=peak_mb,
            framework_metrics=extras,
        )


def main(
    argv: Sequence[str] | None = None, runtime: matched.Runtime | None = None
) -> int:
    """Train upstream verl and emit a neutral measured result."""
    return matched.run_adapter(VerlAdapter(), argv, runtime, description=__doc__)


if __name__ == "__main__":
    sys.exit(main())
