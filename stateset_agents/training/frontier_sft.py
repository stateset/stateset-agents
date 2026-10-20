"""Pinned, assistant-only distributed text SFT for audited frontier checkpoints.

The four native Transformers paths use load-time FSDP sharding and BF16
attention LoRA. Unsupported packed layouts fail before any model download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from stateset_agents.core.chat_masking import assistant_token_mask
from stateset_agents.core.frontier_models import (
    get_frontier_model,
    require_frontier_training,
)
from stateset_agents.data.text_sft import load_text_sft_snapshot
from stateset_agents.training.distributed_preflight import (
    fingerprint_encoded_rows,
    sft_preflight_identity,
    verify_sft_rank_consensus,
)

FRONTIER_REQUIREMENTS = {
    "torch": "2.7.0",
    "transformers": "5.20.0",
    "trl": "1.14.0",
    "peft": "0.21.0",
    "accelerate": "1.15.0",
}


class FrontierSFTConfig(BaseModel):
    """Configure text attention LoRA; checkpoint identities come from the catalog."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    model: str
    train_data: str = "train.jsonl"
    validation_data: str | None = None
    output_dir: str = "outputs/frontier_sft"
    max_steps: int = Field(100, gt=0, strict=True)
    max_length: int = Field(2048, ge=2, le=262144, strict=True)
    learning_rate: float = Field(2e-5, gt=0, le=0.01, allow_inf_nan=False)
    gradient_accumulation_steps: int = Field(8, gt=0, strict=True)
    lora_rank: int = Field(16, gt=0, le=256, strict=True)
    lora_alpha: int = Field(32, gt=0, strict=True)
    save_steps: int = Field(50, gt=0, strict=True)
    eval_steps: int = Field(50, gt=0, strict=True)
    seed: int = Field(42, ge=0, le=2**32 - 1, strict=True)

    @model_validator(mode="after")
    def validate_model_and_paths(self) -> FrontierSFTConfig:
        """Reject unsupported identities and blank file paths before runtime imports."""
        get_frontier_model(self.model)
        for name in ("train_data", "validation_data", "output_dir"):
            path = getattr(self, name)
            if path is not None and not path.strip():
                raise ValueError(f"{name} must not be blank")
        return self


def describe_frontier_sft(
    config: FrontierSFTConfig, *, check_data: bool = False
) -> dict[str, Any]:
    """Preview identity, loaders, loss and limitations without importing Torch."""
    spec = get_frontier_model(config.model)
    report = {
        **spec.describe(),
        "config": config.model_dump(),
        "requirements": FRONTIER_REQUIREMENTS.copy(),
        "objective": "assistant-only text attention LoRA SFT",
        "backend": "transformers-trl-load-time-fsdp",
        "dequantize_to_bf16": spec.checkpoint_format == "fp8",
        "expert_parallel_size": 1,
    }
    if check_data:
        snapshot = load_text_sft_snapshot(config.train_data, config.validation_data)
        report.update(snapshot.describe())
    return report


def prepare_frontier_tokenizer(tokenizer: Any, spec: Any) -> None:
    """Bind supported native options without disabling a thinking-only model."""
    if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template:
        raise ValueError("The pinned tokenizer must supply a native chat template")
    prefix = "".join(
        "{% set " + name + " = " + json.dumps(value) + " %}"
        for name, value in spec.chat_template_options
    )
    tokenizer.chat_template = prefix + tokenizer.chat_template
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("The native tokenizer has neither a pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token


def encode_frontier_rows(
    tokenizer: Any, rows: list[dict[str, Any]], max_length: int
) -> list[dict[str, Any]]:
    """Render native chat and build exact assistant labels without truncation.

    Offset/prefix masking supports upstream templates without generation tags.
    Token IDs must agree with the native tokenizing template or training stops.
    """
    encoded_rows = []
    for index, row in enumerate(rows):
        masked = assistant_token_mask(tokenizer, row["messages"])
        if masked is None:
            raise ValueError(
                f"conversation {index}: cannot locate native assistant spans"
            )
        ids, mask = masked
        native_ids = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
        if ids != list(native_ids):
            raise ValueError(
                f"conversation {index}: rendered IDs differ from native template"
            )
        if len(ids) > max_length:
            raise ValueError(f"conversation {index} exceeds max_length; no truncation")
        if len(mask) != len(ids) or not any(mask[1:]):
            raise ValueError(
                f"conversation {index} has no assistant next-token targets"
            )
        encoded_rows.append(
            {
                "input_ids": ids,
                "labels": [
                    token if supervised else -100
                    for token, supervised in zip(ids, mask, strict=True)
                ],
            }
        )
    return encoded_rows


def prepare_frontier_config(spec: Any, model_config: Any) -> Any:
    """Verify checkpoint architecture and supply per-layer FSDP wrapping.

    Some native architectures have only a head plan. Wrapping the entire
    trunk as one FSDP unit would all-gather the complete checkpoint at once.
    """
    if model_config.model_type != spec.model_type:
        raise ValueError("Pinned checkpoint model_type differs from the catalog")
    text_config = model_config.get_text_config()
    plan = dict(getattr(text_config, "base_model_fsdp_plan", None) or {})
    plan.setdefault("embed_tokens", "free_full_weight")
    plan.setdefault("layers.*", "free_full_weight")
    plan.setdefault("norm", "keep_full_weight")
    text_config.base_model_fsdp_plan = plan
    return model_config


def train_frontier_sft(config: FrontierSFTConfig) -> dict[str, Any]:
    """Train pinned native models with load-time sharding and explicit BF16 weights."""
    spec = get_frontier_model(config.model)
    require_frontier_training(spec)
    snapshot = load_text_sft_snapshot(config.train_data, config.validation_data)
    train, validation = snapshot.train, snapshot.validation
    report = describe_frontier_sft(config)
    report.update(snapshot.describe())
    from packaging.version import Version

    runtime_versions = {}
    for package, minimum in FRONTIER_REQUIREMENTS.items():
        try:
            installed = version(package)
        except PackageNotFoundError as exc:
            raise RuntimeError(
                "Install stateset-agents[frontier] to train this model"
            ) from exc
        if Version(installed) < Version(minimum):
            raise RuntimeError(
                f"{package}>={minimum} required; install stateset-agents[frontier]"
            )
        runtime_versions[package] = installed

    import torch
    import transformers
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoConfig, AutoTokenizer, FineGrainedFP8Config, set_seed
    from transformers.distributed import DistributedConfig
    from trl import SFTConfig, SFTTrainer

    if not hasattr(transformers, spec.architecture):
        raise RuntimeError(
            f"Training runtime has no {spec.architecture}; install [frontier]"
        )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        raise ValueError(
            "Use torchrun with WORLD_SIZE>=2 and enough aggregate BF16 memory"
        )
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Frontier distributed SFT requires BF16-capable CUDA GPUs")
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id, revision=spec.revision, trust_remote_code=False
    )
    prepare_frontier_tokenizer(tokenizer, spec)
    encoded_train = encode_frontier_rows(tokenizer, train, config.max_length)
    encoded_validation = (
        encode_frontier_rows(tokenizer, validation, config.max_length)
        if validation is not None
        else None
    )
    model_config = prepare_frontier_config(
        spec,
        AutoConfig.from_pretrained(
            spec.model_id, revision=spec.revision, trust_remote_code=False
        ),
    )
    model_kwargs: dict[str, Any] = {
        "config": model_config,
        "revision": spec.revision,
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "distributed_config": DistributedConfig(fsdp_size=world_size, ep_size=1),
    }
    if spec.checkpoint_format == "fp8":
        model_kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
    args = SFTConfig(
        output_dir=config.output_dir,
        model_init_kwargs=model_kwargs,
        max_steps=config.max_steps,
        max_length=config.max_length,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataset_num_proc=1,
        dataset_kwargs={"skip_prepare_dataset": True},
        router_aux_loss_coef=0.0,
        assistant_only_loss=False,
        completion_only_loss=False,
        packing=False,
        padding_free=False,
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
            model_id=spec.model_id,
            revision=spec.revision,
            training_config=config.model_dump(
                exclude={"model", "train_data", "validation_data", "output_dir"}
            ),
            dataset=snapshot.describe(),
            tokenizer=tokenizer,
            encoded_data={
                "protocol": "jsonl-input-labels-v1",
                "train": fingerprint_encoded_rows(encoded_train),
                "validation": (
                    fingerprint_encoded_rows(encoded_validation)
                    if encoded_validation is not None
                    else None
                ),
            },
            runtime_versions=runtime_versions,
            implementation_modules=(
                __name__,
                "stateset_agents.data.text_sft",
                "stateset_agents.core.chat_masking",
                "stateset_agents.core.frontier_models",
            ),
        )

    report["rank_consensus"] = verify_sft_rank_consensus(
        identity, world_size=world_size
    )
    set_seed(config.seed)
    trainer = SFTTrainer(
        model=spec.model_id,
        args=args,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list(encoded_train),
        eval_dataset=(
            Dataset.from_list(encoded_validation)
            if encoded_validation is not None
            else None
        ),
        peft_config=LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=0.0,
            target_modules=spec.lora_regex,
            bias="none",
        ),
    )
    import re

    trainable = [
        name for name, p in trainer.model.named_parameters() if p.requires_grad
    ]
    if not trainable or any(
        "lora_" not in name
        or re.search(spec.lora_regex + r"\.lora_[AB]\.", name) is None
        for name in trainable
    ):
        raise RuntimeError("Expected only language attention LoRA trainable parameters")
    result = trainer.train()
    if trainer.state.global_step <= 0:
        raise RuntimeError("Training finished without optimizer steps")
    trainer.save_model(config.output_dir)
    report.update(
        global_step=trainer.state.global_step,
        metrics=result.metrics,
        runtime_versions=runtime_versions,
        world_size=world_size,
        chat_template_sha256=hashlib.sha256(
            tokenizer.chat_template.encode()
        ).hexdigest(),
        encoded_training_sha256=hashlib.sha256(
            json.dumps(encoded_train, sort_keys=True).encode()
        ).hexdigest(),
    )
    if trainer.is_world_process_zero():
        output = Path(config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(output)
        (output / "stateset_training.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    return report


def main() -> None:
    """Preview or execute a config with ``python -m`` or ``torchrun -m``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = FrontierSFTConfig.model_validate_json(Path(args.config).read_text())
    report = (
        train_frontier_sft(config)
        if args.execute
        else describe_frontier_sft(config, check_data=True)
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

__all__ = [
    "FrontierSFTConfig",
    "describe_frontier_sft",
    "train_frontier_sft",
    "encode_frontier_rows",
]
