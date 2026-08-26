"""
Shared runtime scaffolding for the flagship RL trainers (GSPO, DAPO, GEPO, VAPO).

This module consolidates the structural plumbing that was previously duplicated
across the four trainer modules:

- ``SharedModelManager``: a template-method model/tokenizer loader. Each trainer
  module subclasses it and implements the ``_get_transformers`` /
  ``_peft_components`` hooks against its *own module globals*, so tests (and
  users) that patch e.g. ``stateset_agents.training.gspo_trainer.AutoTokenizer``
  keep working exactly as before.
- ``build_group_batch``: pads a group of generated-response dicts into batch
  tensors.
- ``hf_generate_group``: the sequential HuggingFace ``generate`` fallback used
  by DAPO and VAPO group rollouts.
- ``save_checkpoint_artifacts``: model/tokenizer/state/config checkpoint
  writing.

Only structural scaffolding lives here. Loss computation, advantage/ratio math,
and sampling logic remain in the individual trainer modules.

Torch is imported at module scope, matching the four trainer modules that
import this one (they all import torch at module scope themselves); this module
is never imported by torch-optional code paths.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch

from stateset_agents.core.transformers_compat import load_generation_model

logger = logging.getLogger(__name__)


class SharedModelManager:
    """
    Unified model and tokenizer loading pipeline for the flagship trainers.

    The load pipeline is a template method: subclasses customize behavior via
    small hooks instead of re-implementing the whole flow.

    Hooks:
    - ``_get_transformers()``: return ``(AutoTokenizer, AutoModelForCausalLM)``
      after performing the module-specific availability checks. Must read the
      subclass module's globals so test patching keeps working.
    - ``_peft_components()``: return ``(LoraConfig, TaskType, get_peft_model)``
      after module-specific PEFT availability checks.
    - ``_lora_target_modules()``: LoRA target modules (default q/v projections).
    - ``_prepare_model_kwargs(model_kwargs)``: mutate model-loading kwargs
      (e.g. quantization flags).
    - ``_prepare_base_model(base_model)``: post-load adjustments (gradient
      checkpointing, k-bit preparation) before LoRA is applied.
    - ``_load_reference_model(model_cls, model_kwargs)``: optionally load a
      frozen reference model.
    """

    _loaded_message = "Model loaded on {device}"

    def __init__(self, config: Any) -> None:
        self.config = config
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.ref_model: Any | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ hooks

    def _get_transformers(self) -> tuple[Any, Any]:
        """Return (AutoTokenizer, AutoModelForCausalLM) or raise."""
        raise NotImplementedError

    def _peft_components(self) -> tuple[Any, Any, Any]:
        """Return (LoraConfig, TaskType, get_peft_model) or raise."""
        raise NotImplementedError

    def _lora_target_modules(self) -> list[str]:
        return ["q_proj", "v_proj"]

    def _prepare_model_kwargs(self, model_kwargs: dict[str, Any]) -> None:
        return None

    def _prepare_base_model(self, base_model: Any) -> Any:
        return base_model

    def _load_reference_model(
        self, model_cls: Any, model_kwargs: dict[str, Any]
    ) -> None:
        return None

    # --------------------------------------------------------------- pipeline

    def load_model_and_tokenizer(self) -> tuple[Any, Any]:
        """Load model and tokenizer with the configured optimizations."""
        logger.info(f"Loading model: {self.config.model_name}")
        tokenizer_cls, model_cls = self._get_transformers()
        return self._load_impl(tokenizer_cls, model_cls)

    def _load_impl(self, tokenizer_cls: Any, model_cls: Any) -> tuple[Any, Any]:
        # Load tokenizer
        self.tokenizer = tokenizer_cls.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model loading kwargs
        model_kwargs: dict[str, Any] = {
            "torch_dtype": (
                torch.float16
                if self.config.fp16
                else (torch.bfloat16 if self.config.bf16 else torch.float32)
            ),
            "device_map": "auto" if torch.cuda.is_available() else None,
            "trust_remote_code": True,
        }
        self._prepare_model_kwargs(model_kwargs)

        # Load base model
        base_model, resolved_model_cls = load_generation_model(
            model_cls,
            self.config.model_name,
            model_kwargs,
        )
        base_model = self._prepare_base_model(base_model)

        # Add LoRA adapters if configured
        if self.config.use_lora:
            self.model = self._apply_lora(base_model)
        else:
            self.model = base_model

        # Optionally load a frozen reference model
        self._load_reference_model(resolved_model_cls, model_kwargs)

        logger.info(self._loaded_message.format(device=self.device))
        return self.model, self.tokenizer

    def _apply_lora(self, base_model: Any) -> Any:
        lora_config_cls, task_type, peft_factory = self._peft_components()
        lora_config = lora_config_cls(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self._lora_target_modules(),
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=task_type.CAUSAL_LM,
        )
        model = peft_factory(base_model, lora_config)
        if model is not None:
            model.print_trainable_parameters()
        return model


def build_group_batch(
    responses: list[dict[str, Any]],
    device: Any,
    *,
    include_response_mask: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Pad a group of generated-response dicts into batch tensors.

    Each response dict must contain 1-D ``input_ids`` and ``attention_mask``
    tensors (and ``response_mask`` when ``include_response_mask`` is True).

    Returns ``(input_ids, attention_mask, response_mask_or_None)``.
    """
    max_len = max(len(r["input_ids"]) for r in responses)
    batch_size = len(responses)

    batch_input_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    batch_attention_mask = torch.zeros(
        batch_size, max_len, dtype=torch.long, device=device
    )
    batch_response_mask = (
        torch.zeros(batch_size, max_len, device=device)
        if include_response_mask
        else None
    )

    for i, resp in enumerate(responses):
        seq_len = len(resp["input_ids"])
        batch_input_ids[i, :seq_len] = resp["input_ids"]
        batch_attention_mask[i, :seq_len] = resp["attention_mask"]
        if batch_response_mask is not None:
            batch_response_mask[i, :seq_len] = resp["response_mask"]

    return batch_input_ids, batch_attention_mask, batch_response_mask


async def hf_generate_group(
    model: Any,
    tokenizer: Any,
    config: Any,
    device: Any,
    prompt: str,
    group_size: int,
) -> list[dict[str, Any]]:
    """
    Sequential HuggingFace ``generate`` fallback for group rollouts.

    Produces the response-dict schema shared by the DAPO and VAPO trainers:
    ``response``, ``input_ids``, ``attention_mask``, ``response_mask``,
    ``sequence_length``, ``prompt_length``.
    """
    responses = []

    prompt_tokens = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_prompt_length,
    )
    prompt_length = prompt_tokens["input_ids"].shape[1]

    model.eval()
    with torch.no_grad():
        for _ in range(group_size):
            input_ids = prompt_tokens["input_ids"].to(device)
            attention_mask = prompt_tokens["attention_mask"].to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.max_completion_length,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

            full_ids = outputs[0]
            response_length = len(full_ids) - prompt_length

            # Create response mask (1 for response tokens, 0 for prompt)
            response_mask = torch.zeros(len(full_ids), device=device)
            response_mask[prompt_length:] = 1.0

            response_text = tokenizer.decode(
                full_ids[prompt_length:], skip_special_tokens=True
            )

            responses.append(
                {
                    "response": response_text,
                    "input_ids": full_ids,
                    "attention_mask": torch.ones_like(full_ids),
                    "response_mask": response_mask,
                    "sequence_length": response_length,
                    "prompt_length": prompt_length,
                }
            )

    model.train()
    return responses


def save_checkpoint_artifacts(
    model: Any,
    tokenizer: Any,
    output_dir: str,
    *,
    training_state: dict[str, Any] | None = None,
    config_dict: dict[str, Any] | None = None,
    config_filename: str | None = None,
    extra_json: dict[str, Any] | None = None,
    log_label: str = "Checkpoint",
) -> None:
    """
    Save model/tokenizer plus optional training state and JSON artifacts.

    - ``training_state`` is written to ``training_state.pt`` via ``torch.save``.
    - ``config_dict`` is written to ``config_filename`` as indented JSON.
    - ``extra_json`` maps additional filenames to JSON-serializable objects.
    """
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if training_state is not None:
        torch.save(training_state, os.path.join(output_dir, "training_state.pt"))

    if config_dict is not None and config_filename:
        with open(
            os.path.join(output_dir, config_filename), "w", encoding="utf-8"
        ) as f:
            json.dump(config_dict, f, indent=2)

    if extra_json:
        for filename, obj in extra_json.items():
            with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2)

    logger.info(f"{log_label} saved to {output_dir}")


__all__ = [
    "SharedModelManager",
    "build_group_batch",
    "hf_generate_group",
    "save_checkpoint_artifacts",
]
