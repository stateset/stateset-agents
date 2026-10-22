"""BF16 LoRA supervised fine-tuning for Nemotron 3.5 Super VL and Bolt.

This is a multimodal Transformers/PEFT path, separate from the Lightning GSPO
starter. Imports remain lightweight until training is explicitly requested.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from stateset_agents.data.nemotron_vl import NemotronVLCollator, NemotronVLDataset

NEMOTRON_VL_MODELS = {
    "super-vl": "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
    "bolt": "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16",
}
NEMOTRON_VL_REVISIONS = {
    "super-vl": "fc067b34f6b514014d2f8bddba855d2e4bfdfba6",
    "bolt": "bca0a8cb74200a2eb7ca56eb8428b821c829bd6e",
}
# Mamba's fused kernels read projection weights directly, bypassing LoRA's
# forward method. Limit adapters to attention modules in the language tower.
NEMOTRON_VL_LORA_TARGETS = r"language_model\..*\.(q_proj|k_proj|v_proj|o_proj)"


class NemotronVLConfig(BaseModel):
    """Validated settings for image-conditioned, assistant-only BF16 LoRA SFT."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model: Literal["super-vl", "bolt"] = "super-vl"
    revision: str | None = None
    train_data: str = "train.jsonl"
    validation_data: str | None = None
    image_root: str | None = None
    output_dir: str = "./outputs/nemotron_vl"
    max_steps: int = Field(100, gt=0, strict=True)
    max_length: int = Field(4096, ge=2, le=262144, strict=True)
    learning_rate: float = Field(2e-4, gt=0, le=0.01, allow_inf_nan=False)
    gradient_accumulation_steps: int = Field(8, gt=0, strict=True)
    lora_rank: int = Field(16, gt=0, le=256, strict=True)
    lora_alpha: int = Field(32, gt=0, strict=True)
    seed: int = Field(42, ge=0, le=2**32 - 1, strict=True)
    save_steps: int = Field(50, gt=0, strict=True)
    eval_steps: int = Field(50, gt=0, strict=True)
    fsdp: bool = False
    gradient_checkpointing: bool = True
    resume_from_checkpoint: str | None = None

    @model_validator(mode="after")
    def validate_paths_and_revision(self) -> NemotronVLConfig:
        """Reject ambiguous revisions and empty filesystem settings."""
        if self.revision is not None and (
            not re.fullmatch(r"[a-fA-F0-9]{40}", self.revision)
            or set(self.revision) == {"0"}
        ):
            raise ValueError(
                "revision must be an immutable 40-character model commit SHA"
            )
        for name in (
            "train_data",
            "output_dir",
            "validation_data",
            "image_root",
            "resume_from_checkpoint",
        ):
            value = getattr(self, name)
            if value is not None and not value.strip():
                raise ValueError(f"{name} must not be blank")
        return self

    @property
    def model_id(self) -> str:
        """The official BF16 customization checkpoint (never a 24GB/GGUF alias)."""
        return NEMOTRON_VL_MODELS[self.model]

    @property
    def model_revision(self) -> str:
        """Resolve the checked-in checkpoint pin unless explicitly overridden."""
        return self.revision or NEMOTRON_VL_REVISIONS[self.model]


def describe_nemotron_vl(
    config: NemotronVLConfig, *, check_data: bool = False
) -> dict[str, Any]:
    """Describe the real training path without importing Torch or downloading weights."""
    report: dict[str, Any] = {
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "backend": "transformers-peft",
        "objective": "assistant-only image-conditioned SFT",
        "precision": "bfloat16",
        "lora_target_modules": NEMOTRON_VL_LORA_TARGETS,
        "per_device_batch_size": 1,
        "trust_remote_code": True,
        "gpu_training_verified": False,
        "config": {**config.model_dump(), "revision": config.model_revision},
        "notes": [
            "Requires a CUDA BF16 training environment and sufficient aggregate VRAM/host RAM.",
            "The Bolt 24GB footprint applies to quantized inference, not BF16 fine-tuning.",
            "FSDP sharding is optional; its hardware footprint has not been measured here.",
            "Only local images and text are supported; video/audio/tool blocks fail validation.",
        ],
    }
    if check_data:
        train = NemotronVLDataset(config.train_data, config.image_root)
        report["train_examples"] = len(train)
        if config.validation_data:
            validation = NemotronVLDataset(config.validation_data, config.image_root)
            if train.path == validation.path:
                raise ValueError(
                    "training and validation must use separate dataset files"
                )
            report["validation_examples"] = len(validation)
    return report


def train_nemotron_vl(config: NemotronVLConfig) -> dict[str, Any]:
    """Train and save LoRA adapters plus the matching processor using Transformers.

    Run under ``torchrun`` with ``fsdp=True`` for fully sharded training. Model
    loading uses host memory before FSDP wrapping; plan host RAM accordingly.
    No inference device map or quantized inference checkpoint is used.
    """
    report = describe_nemotron_vl(config, check_data=True)
    try:
        import torch
        from packaging.version import Version
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoConfig,
            AutoModelForImageTextToText,
            AutoProcessor,
            Trainer,
            TrainingArguments,
            __version__,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install stateset-agents[nemotron-vl] in your CUDA training environment"
        ) from exc
    if Version(__version__) < Version("5.14.1"):
        raise RuntimeError("Nemotron VL requires transformers>=5.14.1")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Nemotron VL BF16 training requires a BF16-capable CUDA GPU")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not config.fsdp:
        raise ValueError("multi-process Nemotron VL training requires fsdp=True")
    if config.fsdp and world_size < 2:
        raise ValueError("FSDP requires torchrun with at least two processes")
    set_seed(config.seed)
    processor = AutoProcessor.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        code_revision=config.model_revision,
        trust_remote_code=True,
    )
    model_config = AutoConfig.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        code_revision=config.model_revision,
        trust_remote_code=True,
    )
    model_config.use_cache = False
    model_config.llm_config.use_cache = False
    model_config.llm_config.num_logits_to_keep = 0
    model_config.video_pruning_rate = 0.0
    model = AutoModelForImageTextToText.from_pretrained(
        config.model_id,
        config=model_config,
        dtype=torch.bfloat16,
        revision=config.model_revision,
        code_revision=config.model_revision,
        trust_remote_code=True,
    )
    if config.gradient_checkpointing and not config.fsdp:
        # The language tower comes from remote code; nn.Module.__getattr__ types
        # submodule access as ``Tensor | Module``, so resolve the hook by name.
        enable_checkpointing = getattr(
            model.language_model, "gradient_checkpointing_enable", None
        )
        if not callable(enable_checkpointing):
            raise RuntimeError(
                "Nemotron VL language model does not support gradient checkpointing"
            )
        enable_checkpointing(gradient_checkpointing_kwargs={"use_reentrant": False})
    # Generic PeftModel preserves the composite multimodal forward signature;
    # a CausalLM-specific wrapper assumes output-embedding APIs this VLM lacks.
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=0.0,
            target_modules=NEMOTRON_VL_LORA_TARGETS,
            bias="none",
        ),
    )
    trainable = [
        name for name, value in peft_model.named_parameters() if value.requires_grad
    ]
    if not trainable or any(
        "lora_" not in name or ".language_model." not in name for name in trainable
    ):
        raise RuntimeError("expected language-tower LoRA parameters only")
    fsdp_options: dict[str, Any] = {}
    if config.fsdp:
        block = "NemotronHPuzzleBlock" if config.model == "bolt" else "NemotronHBlock"
        fsdp_options = {
            "fsdp": "full_shard auto_wrap",
            "fsdp_config": {
                "version": 2,
                "transformer_layer_cls_to_wrap": [block],
                "reshard_after_forward": True,
                "activation_checkpointing": config.gradient_checkpointing,
            },
        }
    arguments = TrainingArguments(
        output_dir=config.output_dir,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        bf16=True,
        seed=config.seed,
        data_seed=config.seed,
        save_strategy="steps",
        save_steps=config.save_steps,
        eval_strategy="steps" if config.validation_data else "no",
        eval_steps=config.eval_steps,
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        logging_steps=1,
        report_to=[],
        **fsdp_options,
    )
    trainer = Trainer(
        model=peft_model,
        args=arguments,
        train_dataset=NemotronVLDataset(config.train_data, config.image_root),
        eval_dataset=(
            NemotronVLDataset(config.validation_data, config.image_root)
            if config.validation_data
            else None
        ),
        data_collator=NemotronVLCollator(processor, config.max_length),
    )
    trainer.model_accepts_loss_kwargs = False
    result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    if trainer.state.global_step <= 0:
        raise RuntimeError("training finished without any optimizer steps")
    trainer.save_model(config.output_dir)
    if trainer.is_world_process_zero():
        processor.save_pretrained(config.output_dir)
        report.update(global_step=trainer.state.global_step, metrics=result.metrics)
        output = Path(config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "stateset_training.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    return report


def main() -> None:
    """Module entrypoint for direct and torchrun-based fine-tuning."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="StateSet Nemotron VL JSON configuration"
    )
    parser.add_argument("--execute", action="store_true", help="Load weights and train")
    args = parser.parse_args()
    config = NemotronVLConfig.model_validate_json(
        Path(args.config).read_text(encoding="utf-8")
    )
    report = (
        train_nemotron_vl(config)
        if args.execute
        else describe_nemotron_vl(config, check_data=True)
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["NemotronVLConfig", "describe_nemotron_vl", "train_nemotron_vl"]
