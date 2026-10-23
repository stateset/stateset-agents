"""Revision-pinned, distributed text LoRA SFT for Muse Spark Open.

The native Transformers/TRL load-time FSDP and expert-parallel path follows
the upstream recipe. Full-size training requires gated checkpoint access and
cluster resources; imports and previews never load Torch or model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Iterator
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from stateset_agents.data.text_sft import load_text_sft_snapshot
from stateset_agents.training.distributed_preflight import (
    fingerprint_encoded_rows,
    sft_preflight_identity,
    verify_sft_rank_consensus,
)

MUSE_SPARK_MODEL = "meta-models/Muse-Spark-Open-816B-A42B"
MUSE_SPARK_REVISION = "0fb02478b02565b2ce7fc57e6051c91172774ac9"
MUSE_SPARK_LORA_TARGETS = (
    r"model\.language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
)
MUSE_SPARK_REQUIREMENTS = {
    "torch": "2.7.0",
    "transformers": "5.20.0",
    "trl": "1.14.0",
    "peft": "0.21.0",
    "accelerate": "1.15.0",
}


class MuseSparkSFTConfig(BaseModel):
    """Configure assistant-only text adaptation, without inference quantization."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    revision: str = MUSE_SPARK_REVISION
    train_data: str = "train.jsonl"
    validation_data: str | None = None
    output_dir: str = "outputs/muse_spark_sft"
    max_steps: int = Field(100, gt=0, strict=True)
    max_length: int = Field(2048, ge=2, le=1048576, strict=True)
    learning_rate: float = Field(2e-5, gt=0, le=0.01, allow_inf_nan=False)
    gradient_accumulation_steps: int = Field(8, gt=0, strict=True)
    expert_parallel_size: int = Field(8, gt=0, le=256, strict=True)
    lora_rank: int = Field(16, gt=0, le=256, strict=True)
    lora_alpha: int = Field(32, gt=0, strict=True)
    save_steps: int = Field(50, gt=0, strict=True)
    eval_steps: int = Field(50, gt=0, strict=True)
    dataset_num_proc: int = Field(1, gt=0, strict=True)
    seed: int = Field(42, ge=0, le=2**32 - 1, strict=True)
    reasoning_effort: Literal["low", "medium", "high", "xhigh"] = "low"
    current_date: str = "2026-10-15"

    @model_validator(mode="after")
    def validate_identity_and_paths(self) -> MuseSparkSFTConfig:
        """Keep code/weight identity immutable and rendering independent of today."""
        if 256 % self.expert_parallel_size:
            raise ValueError("expert_parallel_size must divide the 256 routed experts")
        if not re.fullmatch(r"[a-fA-F0-9]{40}", self.revision) or set(
            self.revision
        ) == {"0"}:
            raise ValueError("revision must be a nonzero 40-character commit SHA")
        if date.fromisoformat(self.current_date).isoformat() != self.current_date:
            raise ValueError("current_date must be an ISO YYYY-MM-DD date")
        for name in ("train_data", "validation_data", "output_dir"):
            value = getattr(self, name)
            if value is not None and not value.strip():
                raise ValueError(f"{name} must not be blank")
        return self


def describe_muse_spark(
    config: MuseSparkSFTConfig, *, check_data: bool = False
) -> dict[str, Any]:
    """Preview the actual distributed path without importing training dependencies."""
    report: dict[str, Any] = {
        "model_id": MUSE_SPARK_MODEL,
        "model_revision": config.revision,
        "objective": "assistant-only text LoRA SFT",
        "backend": "transformers-trl-load-time-fsdp-ep",
        "lora_target_modules": MUSE_SPARK_LORA_TARGETS,
        "requirements": MUSE_SPARK_REQUIREMENTS.copy(),
        "requires_gated_access": True,
        "gpu_training_verified": False,
        "optimizer_resume_supported": False,
        "base_weight_bytes_lower_bound": 821906490368 * 2 + 130039296 * 4,
        "memory_scope": "Base tensors only, from published safetensors metadata; excludes training overhead.",
        "config": config.model_dump(),
        "notes": [
            "Requires torchrun, BF16 CUDA GPUs, and sufficient aggregate GPU/host memory.",
            "816B total and 42B active do not imply a 42B-weight training footprint.",
            "Only language attention adapters are trained; vision, audio and experts remain frozen.",
            "Load-time distributed sharding saves adapters only; optimizer resume is unsupported.",
            "This path accepts text; native image/audio/video support is not certified here.",
        ],
    }
    if check_data:
        snapshot = load_text_sft_snapshot(config.train_data, config.validation_data)
        report.update(snapshot.describe())
    return report


def _prepare_tokenizer(tokenizer: Any, config: MuseSparkSFTConfig) -> None:
    """Bind the native template's public options and require exact loss targets."""
    if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template:
        raise ValueError("Muse Spark tokenizer has no chat template")
    # Equivalent to passing these kwargs on every native template call. TRL
    # renders internally, so persist the options on this private tokenizer.
    tokenizer.chat_template = (
        "{% set reasoning_effort = "
        + json.dumps(config.reasoning_effort)
        + " %}{% set current_date = "
        + json.dumps(config.current_date)
        + " %}"
        + tokenizer.chat_template
    )


def _validate_encoded_rows(tokenizer: Any, rows: list[Any], max_length: int) -> str:
    def rows_for_hash() -> Iterator[dict[str, Any]]:
        for index, row in enumerate(rows):
            encoded = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_assistant_tokens_mask=True,
            )
            ids = encoded["input_ids"]
            mask = encoded.get("assistant_masks")
            if len(ids) > max_length:
                raise ValueError(
                    f"conversation {index} exceeds max_length; no truncation"
                )
            if mask is None or len(mask) != len(ids) or not any(mask[1:]):
                raise ValueError(
                    f"conversation {index} has no assistant next-token targets"
                )

            yield {
                "input_ids": ids,
                "labels": [
                    token if supervised else -100
                    for token, supervised in zip(ids, mask, strict=True)
                ],
            }

    return fingerprint_encoded_rows(rows_for_hash())


def train_muse_spark(config: MuseSparkSFTConfig) -> dict[str, Any]:
    """Run the upstream load-time FSDP/EP recipe with text-only assistant loss."""
    snapshot = load_text_sft_snapshot(config.train_data, config.validation_data)
    train, validation = snapshot.train, snapshot.validation
    report = describe_muse_spark(config)
    report.update(snapshot.describe())
    from packaging.version import Version

    runtime_versions = {}
    for package, minimum in MUSE_SPARK_REQUIREMENTS.items():
        try:
            installed = version(package)
        except PackageNotFoundError as exc:
            raise RuntimeError(
                "Install stateset-agents[muse-spark] to train Muse Spark"
            ) from exc
        if Version(installed) < Version(minimum):
            raise RuntimeError(
                f"Muse Spark requires {package}>={minimum}; install [muse-spark]"
            )
        runtime_versions[package] = installed

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, set_seed
    from transformers.distributed import DistributedConfig
    from trl import SFTConfig, SFTTrainer

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2 or world_size % config.expert_parallel_size:
        raise ValueError(
            "torchrun WORLD_SIZE must be >=2 and divisible by expert_parallel_size"
        )
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "Muse Spark distributed training requires BF16-capable CUDA GPUs"
        )
    report["distributed"] = {
        "world_size": world_size,
        "expert_parallel_size": config.expert_parallel_size,
        "load_time_sharding": True,
    }
    report["runtime_versions"] = runtime_versions
    tokenizer = AutoTokenizer.from_pretrained(
        MUSE_SPARK_MODEL, revision=config.revision, trust_remote_code=False
    )
    _prepare_tokenizer(tokenizer, config)
    report["chat_template_sha256"] = hashlib.sha256(
        tokenizer.chat_template.encode("utf-8")
    ).hexdigest()
    encoded_data = {
        "protocol": "jsonl-input-labels-v1",
        "train": _validate_encoded_rows(tokenizer, train, config.max_length),
        "validation": (
            _validate_encoded_rows(tokenizer, validation, config.max_length)
            if validation is not None
            else None
        ),
    }
    args = SFTConfig(
        output_dir=config.output_dir,
        model_init_kwargs={
            "revision": config.revision,
            "trust_remote_code": False,
            "dtype": "bfloat16",
            # ``ep_size`` exists from transformers 5.20 (gated above); older
            # installed stubs do not declare it.
            "distributed_config": DistributedConfig(  # type: ignore[call-arg, unused-ignore]
                fsdp_size=world_size, ep_size=config.expert_parallel_size
            ),
        },
        max_steps=config.max_steps,
        max_length=config.max_length,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataset_num_proc=config.dataset_num_proc,
        assistant_only_loss=True,
        eos_token="<|eot|>",
        packing=False,
        bf16=True,
        gradient_checkpointing=True,
        save_only_model=True,
        save_steps=config.save_steps,
        eval_strategy="steps" if validation is not None else "no",
        eval_steps=config.eval_steps,
        prediction_loss_only=True,
        seed=config.seed,
        data_seed=config.seed,
        logging_steps=1,
        report_to=[],
    )

    def identity() -> dict[str, Any]:
        return sft_preflight_identity(
            model_id=MUSE_SPARK_MODEL,
            revision=config.revision,
            training_config=config.model_dump(
                exclude={"train_data", "validation_data", "output_dir"}
            ),
            dataset=snapshot.describe(),
            tokenizer=tokenizer,
            encoded_data=encoded_data,
            runtime_versions=runtime_versions,
            implementation_modules=(__name__, "stateset_agents.data.text_sft"),
        )

    report["rank_consensus"] = verify_sft_rank_consensus(
        identity, world_size=world_size
    )
    set_seed(config.seed)
    trainer = SFTTrainer(
        model=MUSE_SPARK_MODEL,
        args=args,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list(train),
        eval_dataset=Dataset.from_list(validation) if validation is not None else None,
        peft_config=LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=0.0,
            target_modules=MUSE_SPARK_LORA_TARGETS,
            bias="none",
        ),
    )
    if trainer.model is None:
        raise RuntimeError("SFTTrainer did not initialise a model")
    trainable = [
        name
        for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable or any(
        "lora_" not in name or ".language_model." not in name for name in trainable
    ):
        raise RuntimeError("expected language attention LoRA parameters only")
    result = trainer.train()
    if trainer.state.global_step <= 0:
        raise RuntimeError("Muse Spark training finished without optimizer steps")
    trainer.save_model(config.output_dir)
    report.update(global_step=trainer.state.global_step, metrics=result.metrics)
    if trainer.is_world_process_zero():
        output = Path(config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(output)
        (output / "stateset_training.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    return report


def main() -> None:
    """Run or preview through Python/torchrun using a pinned JSON configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = MuseSparkSFTConfig.model_validate_json(Path(args.config).read_text())
    report = (
        train_muse_spark(config)
        if args.execute
        else describe_muse_spark(config, check_data=True)
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(report, indent=2))  # noqa: T201 - CLI report


if __name__ == "__main__":
    main()


__all__ = ["MuseSparkSFTConfig", "describe_muse_spark", "train_muse_spark"]
