#!/usr/bin/env python3
"""Matched-GSPO shootout adapter for NVIDIA NeMo-RL.

NeMo-RL implements GSPO natively in ``ClippedPGLossFn``
(``nemo_rl/algorithms/loss/loss_functions.py``): set
``loss_fn.sequence_level_importance_ratios=true`` and
``loss_fn.token_level_loss=false`` (its own GSPO recipe,
``examples/configs/recipes/llm/grpo-gspo-deepscaler-1.5b-8K.yaml``, does the
same). The adapter writes a NeMo-RL YAML config that inherits the upstream
``examples/configs/grpo_math_1B.yaml`` and runs ``_nemo_rl_driver.py`` through
``uv run`` inside a NeMo-RL source checkout (NeMo-RL is only distributed as
source; PyPI ``nemo-rl`` is an empty 0.0.0 placeholder). The driver registers
a raw-prompt data processor and a Ray environment that scores completions with
the shared GSM8K scorer, then calls NeMo-RL's own ``setup``/``grpo_train``.

Verified against the NeMo-RL v0.7.0 source tree (config keys, loss options,
environment/processor registries, logger hook); not yet executed end to end
on a GPU. The exact on-disk layout of a DTensor-v2 LoRA checkpoint is the
least-verified part: the adapter fails closed unless it finds exactly one
PEFT adapter under the final checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from . import _matched_gsm8k as matched
except ImportError:  # executed as a script: benchmarks/adapters is on sys.path
    import _matched_gsm8k as matched  # type: ignore[no-redef]

NEMO_RL_VERSION = "0.7.0"
NEMO_RL_TAG = "v0.7.0"
NEMO_RL_COMMIT = "81aa43dda4765b0429cf31dab44441e4e4383911"
FRAMEWORK_NAME = "nemo-rl"
DIR_ENV_VAR = "STATESET_NEMO_RL_DIR"
DEFAULT_DIR = Path("/workspace/frameworks/nemo-rl")
ENV_NAME = "stateset_gsm8k"
PROCESSOR_NAME = "stateset_raw_prompt"
DRIVER = Path(__file__).resolve().parent / "_nemo_rl_driver.py"


def build_config(
    config: Mapping[str, Any],
    *,
    seed: int,
    model_path: Path,
    train_file: Path,
    base_config: Path,
    checkpoint_dir: Path,
    log_dir: Path,
) -> dict[str, Any]:
    """Map the matched protocol onto a NeMo-RL GRPO config (nested)."""
    max_steps = int(config["max_steps"])
    warmup = max(1, matched.warmup_steps(config))
    total_length = int(config["max_prompt_length"]) + int(
        config["max_completion_length"]
    )
    return {
        "defaults": str(base_config),
        "grpo": {
            "num_prompts_per_step": matched.prompts_per_step(config),
            "num_generations_per_prompt": int(config["num_generations"]),
            "max_rollout_turns": 1,
            "max_num_epochs": int(config["num_train_epochs"]),
            "max_num_steps": max_steps,
            # (r - mean_group) / std_group, mean including the sample itself.
            "normalize_rewards": config["scale_rewards"] == "group",
            "use_leave_one_out_baseline": False,
            "overlong_filtering": False,
            "val_period": 0,
            "val_at_start": False,
            "val_at_end": False,
            "seed": seed,
        },
        "loss_fn": {
            # GSPO: sequence-level importance ratio + sequence-mean loss.
            "sequence_level_importance_ratios": True,
            "token_level_loss": False,
            "ratio_clip_min": float(config["epsilon"]),
            "ratio_clip_max": float(config["epsilon_high"]),
            "ratio_clip_c": None,
            "reference_policy_kl_penalty": float(config["beta"]),
            "use_on_policy_kl_approximation": False,
            "use_importance_sampling_correction": False,
        },
        "checkpointing": {
            "enabled": True,
            "checkpoint_dir": str(checkpoint_dir),
            "metric_name": None,
            "keep_top_k": 1,
            "save_period": max_steps,
            "save_optimizer": False,
            "model_save_format": "safetensors",
            "save_consolidated": True,
        },
        "policy": {
            "model_name": str(model_path),
            "tokenizer": {"name": str(model_path)},
            # One optimizer step per rollout batch.
            "train_global_batch_size": matched.completions_per_step(config),
            "train_micro_batch_size": int(config["per_device_train_batch_size"]),
            "logprob_batch_size": int(config["per_device_train_batch_size"]),
            "max_total_sequence_length": total_length,
            "precision": "bfloat16" if bool(config["bf16"]) else "float32",
            "max_grad_norm": float(config["max_grad_norm"]),
            "dtensor_cfg": {
                "activation_checkpointing": bool(config["gradient_checkpointing"]),
                "lora_cfg": {
                    "enabled": True,
                    "target_modules": list(config["lora_target_modules"]),
                    "match_all_linear": False,
                    "dim": int(config["lora_r"]),
                    "alpha": int(config["lora_alpha"]),
                    "dropout": float(config["lora_dropout"]),
                },
            },
            "sequence_packing": {"enabled": False},
            "dynamic_batching": {"enabled": False},
            "optimizer": {
                "name": "torch.optim.AdamW",
                "kwargs": {
                    "lr": float(config["learning_rate"]),
                    "weight_decay": float(config["weight_decay"]),
                    "betas": [float(config["adam_beta1"]), float(config["adam_beta2"])],
                    "eps": 1e-8,
                },
            },
            # Linear warmup from ~0 then cosine to 0, as in TRL's schedule.
            "scheduler": [
                {
                    "name": "torch.optim.lr_scheduler.LinearLR",
                    "kwargs": {
                        "start_factor": 1e-8,
                        "end_factor": 1.0,
                        "total_iters": warmup,
                    },
                },
                {
                    "name": "torch.optim.lr_scheduler.CosineAnnealingLR",
                    "kwargs": {"T_max": max_steps - warmup, "eta_min": 0.0},
                },
                {"milestones": [warmup]},
            ],
            "generation": {
                "max_new_tokens": int(config["max_completion_length"]),
                "temperature": float(config["temperature"]),
                "top_p": float(config["top_p"]),
                "top_k": None,
                "vllm_cfg": {
                    "gpu_memory_utilization": 0.4,
                    "max_model_len": total_length,
                },
            },
        },
        "data": {
            "max_input_seq_length": int(config["max_prompt_length"]),
            "shuffle": True,
            "use_multiple_dataloader": False,
            "train": {
                "dataset_name": "ResponseDataset",
                "data_path": str(train_file),
                "input_key": "prompt",
                "output_key": "label",
                "prompt_file": None,
                "system_prompt_file": None,
                "processor": PROCESSOR_NAME,
                "env_name": ENV_NAME,
                "split_validation_size": 0,
                "seed": seed,
            },
            "validation": None,
            "default": None,
        },
        "env": {ENV_NAME: {"num_workers": 1}},
        "logger": {
            "log_dir": str(log_dir),
            "wandb_enabled": False,
            "tensorboard_enabled": False,
            "mlflow_enabled": False,
            "swanlab_enabled": False,
            "monitor_gpus": False,
        },
        "cluster": {"gpus_per_node": 1, "num_nodes": 1},
    }


def protocol_deviations(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Protocol settings NeMo-RL v0.7.0 cannot express identically."""
    return [
        matched.deviation(
            "lora_init",
            "peft default (A kaiming-uniform, B zero)",
            "xavier A, zero B",
            "NeMo-RL DTensor LoRA (lora_cfg.lora_A_init) offers only "
            "'xavier' or 'uniform' A initialization",
        ),
        matched.deviation(
            "lora_generation_weights",
            "rollouts sample from the exact trained LoRA policy",
            "LoRA merged into base weights for vLLM generation",
            "NeMo-RL's DTensor GRPO LoRA path merges adapter weights for "
            "generation (documented small train/inference mismatch)",
        ),
    ]


PROTOCOL_NOTES = [
    "GSPO is selected with loss_fn.sequence_level_importance_ratios=true and "
    "token_level_loss=false, mirroring NeMo-RL's own GSPO recipe.",
    "NeMo-RL computes the group std unbiased (Bessel) with epsilon 1e-6; TRL "
    "uses torch.std (also unbiased) with epsilon 1e-4.",
    "Prompts are tokenized raw (no chat template) by a registered processor; "
    "the reward runs the shared scorer in a NeMo-RL Ray environment.",
    "Prompt order comes from NeMo-RL's own seeded dataloader, not TRL's.",
]


def nemo_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """ResponseDataset rows: input=raw prompt, output=gold label."""
    return [
        {"prompt": row["prompt"], "label": matched.gold_label(row["gold_answer"])}
        for row in rows
    ]


def write_config(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write the config as JSON (a strict YAML subset NeMo-RL's loader reads)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def read_metrics(
    path: Path,
) -> tuple[int, list[float], dict[str, float]]:
    """Steps, per-step mean reward, and extras from the driver's JSONL tee."""
    if not path.is_file():
        raise matched.AdapterFailure(f"NeMo-RL driver wrote no metrics: {path}")
    rewards: dict[int, float] = {}
    tokens_per_sec: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or record.get("prefix") != "train":
            continue
        step = record.get("step")
        metrics = record.get("metrics")
        if not isinstance(step, int) or not isinstance(metrics, dict):
            continue
        reward = metrics.get("reward")
        if isinstance(reward, (int, float)) and not isinstance(reward, bool):
            rewards[step] = float(reward)
        rate = metrics.get("valid_tokens_per_sec_per_gpu")
        if isinstance(rate, (int, float)):
            tokens_per_sec.append(float(rate))
    extras = (
        {"tokens_per_second_per_gpu_mean": sum(tokens_per_sec) / len(tokens_per_sec)}
        if tokens_per_sec
        else {}
    )
    return max(rewards, default=0), [rewards[k] for k in sorted(rewards)], extras


class NemoRLAdapter:
    """NeMo-RL implementation of the matched-GSPO adapter contract."""

    name = FRAMEWORK_NAME
    pinned_version = NEMO_RL_VERSION

    @staticmethod
    def checkout(args: argparse.Namespace) -> Path:
        return Path(
            args.framework_dir or os.environ.get(DIR_ENV_VAR) or DEFAULT_DIR
        ).resolve()

    def uv_python(self, args: argparse.Namespace) -> list[str]:
        return [
            "uv",
            "run",
            "--locked",
            "--directory",
            str(self.checkout(args)),
            "python",
        ]

    def probe_command(self, args: argparse.Namespace) -> list[str]:
        return [*self.uv_python(args), "-c", matched.probe_code("nemo_rl")]

    def prepare(self, ctx: matched.LaunchContext) -> matched.LaunchPlan:
        checkout = self.checkout(ctx.args)
        base_config = checkout / "examples" / "configs" / "grpo_math_1B.yaml"
        train_file = matched.write_jsonl(
            ctx.work_dir / "data" / "train.jsonl", nemo_rows(ctx.train_rows)
        )
        payload = build_config(
            ctx.config,
            seed=int(ctx.args.seed),
            model_path=ctx.model_path,
            train_file=train_file,
            base_config=base_config,
            checkpoint_dir=ctx.work_dir / "checkpoints",
            log_dir=ctx.work_dir / "logs",
        )
        config_path = write_config(ctx.work_dir / "nemo_rl_gspo.yaml", payload)
        metrics_path = ctx.artifact_dir / "framework" / "metrics.jsonl"
        env = dict(ctx.env)
        env.setdefault("WANDB_MODE", "disabled")
        return matched.LaunchPlan(
            command=[
                *self.uv_python(ctx.args),
                str(DRIVER),
                "--config",
                str(config_path),
                "--metrics-jsonl",
                str(metrics_path),
                "--env-name",
                ENV_NAME,
                "--processor-name",
                PROCESSOR_NAME,
            ],
            env=env,
            settings=payload,
            deviations=protocol_deviations(ctx.config),
            notes=list(PROTOCOL_NOTES),
            cwd=checkout,
        )

    def collect(
        self,
        ctx: matched.LaunchContext,
        plan: matched.LaunchPlan,
        runtime: matched.Runtime,
    ) -> matched.TrainingOutcome:
        steps, rewards, extras = read_metrics(
            ctx.artifact_dir / "framework" / "metrics.jsonl"
        )
        matched.require_completed_steps(steps, ctx.config, self.name)
        matched.require_rewards(rewards, self.name)
        final = ctx.work_dir / "checkpoints" / f"step_{steps}"
        return matched.TrainingOutcome(
            steps_completed=steps,
            step_rewards=rewards,
            step_reward_stds=[],
            adapter_dir=matched.find_peft_adapter(final),
            framework_metrics=extras,
        )


def main(
    argv: Sequence[str] | None = None, runtime: matched.Runtime | None = None
) -> int:
    """Train NeMo-RL from source and emit a neutral measured result."""
    return matched.run_adapter(NemoRLAdapter(), argv, runtime, description=__doc__)


if __name__ == "__main__":
    sys.exit(main())
