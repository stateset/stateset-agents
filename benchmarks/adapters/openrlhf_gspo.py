#!/usr/bin/env python3
"""Matched-GSPO shootout adapter for upstream OpenRLHF.

OpenRLHF implements GSPO natively: ``--actor.policy_loss_type gspo``
(``openrlhf/models/loss.py::PolicyLoss``) computes the length-normalized
sequence importance ratio, clips it with ``--actor.eps_clip_low_high`` and
forces the per-sample (sequence-mean) loss. Training runs as ``python -m
openrlhf.cli.train_ppo_ray`` in OpenRLHF's own venv with the GSM8K reward
loaded through ``--reward.remote_url <python file>`` (OpenRLHF's documented
``reward_func(queries, prompts, labels)`` hook), LoRA through
``--ds.lora.*``, and vLLM colocated with DeepSpeed on the single GPU.

Verified against the OpenRLHF 0.11.2 wheel source (CLI flags, GSPO loss,
reward hook, step log line, LoRA save path); not yet executed end to end on a
GPU.
"""

from __future__ import annotations

import argparse
import ast
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

OPENRLHF_VERSION = "0.11.2"
FRAMEWORK_NAME = "openrlhf"
PYTHON_ENV_VAR = "STATESET_OPENRLHF_PYTHON"
_STEP_LINE = re.compile(r"Global step (\d+): (\{.*\})\s*$")
_FLOAT_FIELD = r"'{key}':\s*(?:np\.float\d*\()?(-?[0-9.]+(?:[eE][-+]?\d+)?)"


def build_arguments(
    config: Mapping[str, Any],
    *,
    seed: int,
    model_path: Path,
    prompt_file: Path,
    reward_path: Path,
    output_dir: Path,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    """Map the matched protocol onto ``train_ppo_ray`` flags.

    Values of ``True`` are store-true switches; lists expand to ``nargs``.
    """
    prompts = matched.prompts_per_step(config)
    beta = float(config["beta"])
    arguments: dict[str, Any] = {
        # Single-GPU colocated placement (DeepSpeed actor + one vLLM engine).
        "--actor.num_nodes": 1,
        "--actor.num_gpus_per_node": 1,
        "--ref.num_nodes": 1,
        "--ref.num_gpus_per_node": 1,
        "--vllm.num_engines": 1,
        "--vllm.tensor_parallel_size": 1,
        "--vllm.gpu_memory_utilization": 0.4,
        "--vllm.enable_sleep": True,
        "--ds.enable_sleep": True,
        "--train.colocate_all": True,
        # Model, precision, memory.
        "--actor.model_name_or_path": str(model_path),
        "--ds.param_dtype": "bf16" if bool(config["bf16"]) else "fp16",
        "--ds.zero_stage": 2,
        "--actor.gradient_checkpointing_enable": bool(config["gradient_checkpointing"]),
        "--ds.lora.rank": int(config["lora_r"]),
        "--ds.lora.alpha": int(config["lora_alpha"]),
        "--ds.lora.dropout": float(config["lora_dropout"]),
        "--ds.lora.target_modules": list(config["lora_target_modules"]),
        # GSPO objective: sequence ratio, asymmetric clip, sequence-mean loss.
        "--actor.policy_loss_type": "gspo",
        "--actor.eps_clip_low_high": [
            float(config["epsilon"]),
            float(config["epsilon_high"]),
        ],
        # Group-normalized advantages: (r - mean_group) / std_group.
        "--algo.advantage.estimator": "group_norm",
        # beta = 0: no reference model (init_coef 0) and no KL loss.
        "--algo.kl.init_coef": beta,
        # Batch geometry: prompts x generations per step, one optimizer step.
        "--rollout.batch_size": prompts,
        "--rollout.n_samples_per_prompt": int(config["num_generations"]),
        "--rollout.micro_batch_size": int(config["per_device_train_batch_size"]),
        "--train.batch_size": matched.completions_per_step(config),
        "--train.micro_batch_size": int(config["per_device_train_batch_size"]),
        "--train.max_epochs": int(config["num_iterations"]),
        "--train.num_episodes": int(config["num_train_epochs"]),
        "--train.seed": seed,
        # Sampling.
        "--rollout.temperature": float(config["temperature"]),
        "--rollout.top_p": float(config["top_p"]),
        "--rollout.max_new_tokens": int(config["max_completion_length"]),
        "--data.max_len": int(config["max_prompt_length"])
        + int(config["max_completion_length"]),
        # Optimizer and schedule (cosine to zero after linear warmup).
        "--actor.adam.lr": float(config["learning_rate"]),
        "--actor.adam.betas": [
            float(config["adam_beta1"]),
            float(config["adam_beta2"]),
        ],
        "--actor.adam.weight_decay": float(config["weight_decay"]),
        "--actor.max_norm": float(config["max_grad_norm"]),
        "--actor.lr_scheduler": "cosine_with_min_lr",
        "--actor.lr_warmup_ratio": float(config["warmup_ratio"]),
        "--actor.min_lr_ratio": 0.0,
        # Data: raw prompt text (no chat template), gold answer as label.
        "--data.prompt_dataset": str(prompt_file),
        "--data.input_key": "prompt",
        "--data.label_key": "label",
        "--data.max_samples": int(config["num_train_examples"]),
        "--reward.remote_url": str(reward_path),
        # Outputs.
        "--ckpt.output_dir": str(output_dir),
        "--ckpt.path": str(checkpoint_dir),
        "--ckpt.save_steps": -1,
        "--logger.logging_steps": 1,
    }
    if config["lr_scheduler_type"] != "cosine":
        raise matched.AdapterFailure(
            "the OpenRLHF mapping covers lr_scheduler_type=cosine only"
        )
    return arguments


def render_arguments(arguments: Mapping[str, Any]) -> list[str]:
    """Flatten the flag mapping into argv."""
    argv: list[str] = []
    for flag, value in arguments.items():
        if value is True:
            argv.append(flag)
        elif value is False or value is None:
            continue
        elif isinstance(value, (list, tuple)):
            argv.extend([flag, *(str(item) for item in value)])
        else:
            argv.extend([flag, str(value)])
    return argv


def protocol_deviations(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Protocol settings OpenRLHF 0.11.2 cannot express."""
    return []


PROTOCOL_NOTES = [
    "OpenRLHF's GSPO ratio keeps the gradient through the sequence-mean log "
    "ratio (paper form); TRL computes the same sequence-level ratio.",
    "group_norm whitens with the group std plus 1e-9; TRL uses the group std "
    "plus 1e-4. Both are per-prompt-group normalizations.",
    "The completion is recovered by stripping the prompt from OpenRLHF's "
    "decoded query (special tokens removed) before the shared scorer runs.",
    "Prompt order comes from OpenRLHF's own seeded dataloader, not TRL's.",
]


def openrlhf_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Prompt-dataset rows: raw prompt text plus the gold label."""
    return [
        {
            "prompt": row["prompt"],
            "label": matched.gold_label(row["gold_answer"]),
            "index": int(row["index"]),
        }
        for row in rows
    ]


def _status_value(text: str, key: str) -> float | None:
    match = re.search(_FLOAT_FIELD.format(key=re.escape(key)), text)
    return float(match.group(1)) if match else None


def parse_step_log(
    log_text: str,
) -> tuple[int, list[float], list[float]]:
    """Steps completed and per-step reward mean/std from ``train.log``.

    OpenRLHF prints ``✨ Global step N: {status dict}`` after every step.
    """
    by_step: dict[int, tuple[float | None, float | None]] = {}
    for line in log_text.splitlines():
        match = _STEP_LINE.search(line)
        if match is None:
            continue
        step = int(match.group(1))
        body = match.group(2)
        try:
            status = ast.literal_eval(body)
        except (SyntaxError, ValueError):
            status = None
        if isinstance(status, dict):
            mean = status.get("rollout/reward_mean")
            std = status.get("rollout/reward_std")
        else:
            mean = _status_value(body, "rollout/reward_mean")
            std = _status_value(body, "rollout/reward_std")
        by_step[step] = (
            float(mean) if isinstance(mean, (int, float)) else None,
            float(std) if isinstance(std, (int, float)) else None,
        )
    steps = max(by_step, default=0)
    ordered = [by_step[step] for step in sorted(by_step)]
    rewards = [mean for mean, _ in ordered if mean is not None]
    stds = [std for _, std in ordered if std is not None]
    return steps, rewards, stds


class OpenRLHFAdapter:
    """OpenRLHF implementation of the matched-GSPO adapter contract."""

    name = FRAMEWORK_NAME
    pinned_version = OPENRLHF_VERSION

    @staticmethod
    def python(args: argparse.Namespace) -> str:
        return str(
            args.framework_python or os.environ.get(PYTHON_ENV_VAR) or sys.executable
        )

    def probe_command(self, args: argparse.Namespace) -> list[str]:
        return [self.python(args), "-c", matched.probe_code("openrlhf")]

    def prepare(self, ctx: matched.LaunchContext) -> matched.LaunchPlan:
        prompt_file = matched.write_jsonl(
            ctx.work_dir / "data" / "train.jsonl", openrlhf_rows(ctx.train_rows)
        )
        arguments = build_arguments(
            ctx.config,
            seed=int(ctx.args.seed),
            model_path=ctx.model_path,
            prompt_file=prompt_file,
            reward_path=Path(matched.__file__).resolve(),
            output_dir=ctx.work_dir / "final",
            checkpoint_dir=ctx.work_dir / "checkpoints",
        )
        env = dict(ctx.env)
        env.setdefault("WANDB_MODE", "disabled")
        return matched.LaunchPlan(
            command=[
                self.python(ctx.args),
                "-m",
                "openrlhf.cli.train_ppo_ray",
                *render_arguments(arguments),
            ],
            env=env,
            settings=arguments,
            deviations=protocol_deviations(ctx.config),
            notes=list(PROTOCOL_NOTES),
        )

    def collect(
        self,
        ctx: matched.LaunchContext,
        plan: matched.LaunchPlan,
        runtime: matched.Runtime,
    ) -> matched.TrainingOutcome:
        log_path = ctx.artifact_dir / "framework" / "train.log"
        if not log_path.is_file():
            raise matched.AdapterFailure(f"OpenRLHF wrote no training log: {log_path}")
        steps, rewards, stds = parse_step_log(
            log_path.read_text(encoding="utf-8", errors="replace")
        )
        matched.require_completed_steps(steps, ctx.config, self.name)
        matched.require_rewards(rewards, self.name)
        return matched.TrainingOutcome(
            steps_completed=steps,
            step_rewards=rewards,
            step_reward_stds=stds,
            adapter_dir=matched.find_peft_adapter(ctx.work_dir / "final"),
        )


def main(
    argv: Sequence[str] | None = None, runtime: matched.Runtime | None = None
) -> int:
    """Train upstream OpenRLHF and emit a neutral measured result."""
    return matched.run_adapter(OpenRLHFAdapter(), argv, runtime, description=__doc__)


if __name__ == "__main__":
    sys.exit(main())
