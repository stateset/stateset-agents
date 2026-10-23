"""Supervised fine-tune from a curated chat-format JSONL.

This is the job at the end of the improvement loop: chat → capture → grade →
curate → **sft** → trained adapter → chat with it → repeat.

The logic lives in the installed package rather than in ``scripts/`` because
``scripts*`` is excluded from the wheel (see ``[tool.setuptools.packages.find]``).
A remote worker provisioned by ``stateset_agents.remote`` installs the
published package and nothing else, so anything it must execute has to be
importable from here. ``scripts/sft_from_curated.py`` re-exports these names
and keeps the standalone CLI.

The implementation deliberately uses ``transformers.Trainer`` directly rather
than a framework-level abstraction:

* The framework's existing trainers (GSPO/GRPO/DAPO) are RL trainers — they
  expect an environment + reward function, not a static dataset.
* SFT is well-served by HF Trainer; reinventing it here would add maintenance
  burden without benefit.
* The minimal HF Trainer call composes cleanly with the framework's
  ``AgentConfig.peft_path`` field — drop the adapter directory at the path
  and ``stateset-agents serve --checkpoint`` picks it up.

Stub-aware: when no CUDA GPU is present or transformers isn't importable, the
job prints the training plan it *would* run and succeeds. That keeps the whole
submit → run → fetch path exercisable on CPU-only CI.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from stateset_agents.core.chat_masking import supervised_turn_sequences
from stateset_agents.core.lora_targets import (
    NON_TEXT_STACK_MARKERS as _NON_TEXT_STACK_MARKERS,
)
from stateset_agents.core.lora_targets import (
    text_lora_targets,
)
from stateset_agents.core.model_presets import PRESETS
from stateset_agents.core.transformers_compat import generation_compat_kwargs
from stateset_agents.data.finetuning import (
    parse_finetuning_json,
    validate_finetuning_split,
)
from stateset_agents.data.jsonl import jsonl_lines
from stateset_agents.data.source import DatasetSourceIdentity, SFTDatasetIdentity
from stateset_agents.exceptions import IMPORT_EXCEPTIONS
from stateset_agents.training.lineage import (
    AdapterManifest,
    write_manifest,
)
from stateset_agents.training.quantization import (
    QuantizationConfig,
    prepare_quantized_model,
    quantized_load_kwargs,
    require_bitsandbytes,
)

logger = logging.getLogger("sft_from_curated")


def _package_version() -> str | None:
    """Installed package version, for the manifest. None if unavailable."""
    try:
        from stateset_agents import __version__

        return str(__version__)
    except ImportError:  # pragma: no cover - defensive
        return None


__all__ = [
    "build_eval_extras",
    "build_training_arguments",
    "cuda_device_count",
    "eval_gate_failures",
    "evaluate_checks",
    "generate_completions",
    "gpu_available",
    "is_placed_at_load",
    "is_sharded_across_devices",
    "judge_completion",
    "load_base_model_for_sft",
    "load_chat_dataset",
    "load_sft_dataset",
    "parse_chat_dataset",
    "log_cuda_memory_per_device",
    "log_device_map_summary",
    "model_load_kwargs",
    "normalize_eval_prompts",
    "prepare_for_lora",
    "print_training_plan",
    "resolve_resume_checkpoint",
    "run_sft",
    "run_sft_job",
    "write_eval_results",
]

#: Keys of a ``RemoteJobSpec`` dict that configure the provider rather than the
#: job. ``run_sft_job`` ignores them so a full spec can be passed straight in.
_PROVIDER_ONLY_KEYS = frozenset(
    {
        "gpu",
        "gpu_count",
        "timeout_s",
        "package_version",
        "container_disk_gb",
        "cloud_type",
        "network_volume_id",
    }
)


def load_chat_dataset(path: Path) -> list[dict[str, Any]]:
    """Load a chat-format JSONL.

    Each row must have a ``messages`` key with a list of ``{role, content}``
    dicts. Tool-calling rows may also contain ``tools`` (JSON schemas) and
    assistant ``tool_calls`` with typed arguments. Legacy malformed JSON and
    missing-message rows are skipped. Ambiguous JSON and non-training split
    declarations fail the whole read, rather than training on a partial input.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return parse_chat_dataset(path.read_bytes())


def parse_chat_dataset(data: bytes) -> list[dict[str, Any]]:
    """Parse captured UTF-8 chat JSONL with the same rules as the file loader.

    Consumers can hash these exact bytes and retain source identity before
    training, rather than hashing the original path after it may have changed.
    """
    rows: list[dict[str, Any]] = []
    for line_num, line in jsonl_lines(data.decode("utf-8")):
        line = line.strip()
        try:
            row = parse_finetuning_json(line)
        except json.JSONDecodeError as e:
            logger.warning(
                "Skipping line %d: invalid JSON at column %d", line_num, e.colno
            )
            continue
        except ValueError as exc:
            raise ValueError(f"line {line_num}: {exc}") from None
        try:
            validate_finetuning_split(row, expected_split="train")
        except ValueError as exc:
            raise ValueError(f"line {line_num}: {exc}") from None
        if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
            logger.warning("Skipping line %d: missing 'messages' list", line_num)
            continue
        rows.append(row)
    return rows


def load_sft_dataset(
    path: Path, *, text_only: bool = False, expected_sha256: str | None = None
) -> tuple[list[dict[str, Any]], SFTDatasetIdentity]:
    """Capture source bytes, parse training rows and bind both identities.

    ``text_only`` uses Studio's strict text normalization. The default preserves
    typed tool arguments and metadata through the general chat loader. An
    expected source hash binds this read to an existing review or manifest.
    """
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Dataset not found: {path}") from exc
    source = DatasetSourceIdentity.from_bytes(path, data)
    if expected_sha256 is not None and source.sha256 != expected_sha256:
        raise ValueError("Training dataset changed from the reviewed source")
    if text_only:
        from stateset_agents.data.text_sft import parse_text_sft_records

        rows, _ = parse_text_sft_records(data, str(path))
    else:
        rows = parse_chat_dataset(data)
    return rows, SFTDatasetIdentity.from_rows(rows, source=source)


def gpu_available() -> bool:
    """Detect whether we have a CUDA GPU + transformers."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def print_training_plan(
    rows: list[dict[str, Any]],
    base_model: str,
    output_dir: Path,
    num_epochs: int,
    lora_r: int,
    learning_rate: float,
    max_length: int,
    seed: int = 42,
    quantization: QuantizationConfig | None = None,
) -> None:
    """Dump a human-readable summary of what training *would* do."""
    print()
    print("=" * 60)
    print("SFT Training Plan (dry run — no GPU detected)")
    print("=" * 60)
    print(f"  Dataset size:     {len(rows)} examples")
    print(f"  Base model:       {base_model}")
    print(f"  Output dir:       {output_dir}")
    print(f"  Epochs:           {num_epochs}")
    print(f"  LoRA r:           {lora_r}")
    print(f"  Learning rate:    {learning_rate}")
    print(f"  Max sequence:     {max_length}")
    print(f"  Training seed:    {seed}")
    if quantization is not None and quantization.enabled:
        settings = ", ".join(f"{k}={v}" for k, v in quantization.to_dict().items())
        print(f"  Quantization:     QLoRA ({settings}; needs CUDA + bitsandbytes)")
    print()
    sample = rows[0] if rows else None
    if sample:
        print("  First example:")
        for msg in sample.get("messages", [])[:2]:
            content = msg.get("content")
            if not isinstance(content, str):  # e.g. a pure tool call
                calls = msg.get("tool_calls") or []
                names = [
                    str((c.get("function") or {}).get("name"))
                    for c in calls
                    if isinstance(c, dict)
                ]
                content = f"<tool call: {', '.join(names) or '?'}>"
            print(f"    [{msg.get('role')}] {content[:60]}…")
    print()
    print("  To execute, run this script on a host with a CUDA-capable GPU.")
    print("=" * 60)


#: Linear-layer names LoRA is normally applied to, across common decoder
#: architectures: separate q/k/v/o projections (Llama, Qwen, Mistral), fused
#: qkv (GPT-2 style ``c_attn``), and the MLP projections.
_LORA_TARGET_CANDIDATES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "c_attn",
    "c_proj",
    "c_fc",
    "query_key_value",
    "qkv_proj",
    "gate_up_proj",
    "input_linear",
    "output_linear",
    "in_proj",
    "w1",
    "w2",
    "w3",
    "dense",
    "fc1",
    "fc2",
    # Mamba-style linear-attention projections. Hybrid models interleave a
    # minority of standard `self_attn` layers with a majority of `linear_attn`
    # layers (verified against the Qwen/Qwen3.8-27B weight map: 432 linear_attn
    # tensors vs 96 self_attn tensors). Without these names LoRA silently
    # adapts only the minority attention layers — no error, just a badly
    # under-adapted model.
    "in_proj_qkv",
    "in_proj_a",
    "in_proj_b",
    "in_proj_z",
    "out_proj",
)


def _trust_remote_code(model_name: str) -> bool:
    """Run repo-shipped model code only for curated presets that require it;
    any other model loads with the native transformers implementation."""
    return next(
        (p.trust_remote_code for p in PRESETS.values() if p.model_id == model_name),
        False,
    )


def _row_tools(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    tools = row.get("tools")
    if tools is not None and (
        not isinstance(tools, list) or not all(isinstance(t, dict) for t in tools)
    ):
        raise ValueError("SFT row 'tools' must be a list of tool schema objects.")
    return tools


def _render_sft_row(tokenizer: Any, row: dict[str, Any]) -> dict[str, str]:
    """Render before Arrow conversion so heterogeneous tool schemas stay intact."""
    kwargs: dict[str, Any] = {}
    tools = _row_tools(row)
    if tools is not None:
        kwargs["tools"] = tools
    return {
        "text": tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False, **kwargs
        )
    }


#: How :func:`tokenize_sft_rows` treated each row, as counted in the manifest
#: (exactly one outcome per row):
#:
#: * ``all_assistant_turns`` — one sequence supervising every target turn;
#: * ``split_per_turn`` — the template renders a target turn differently once
#:   it is history (Qwen3/Qwen3.5), so the row became one sequence per target
#:   turn, each rendered as the final turn of its prefix;
#: * ``dropped_no_target`` — every assistant turn carries ``weight: 0``;
#: * ``dropped_unmaskable`` — a target turn cannot be located in the render;
#: * ``dropped_truncated`` — every supervised token fell past ``max_length``.
#:
#: Kept rows that lost some supervised tokens (or whole later-turn sequences)
#: to ``max_length`` are counted separately as ``partially_truncated``.
SFT_LABEL_OUTCOMES = (
    "all_assistant_turns",
    "split_per_turn",
    "dropped_no_target",
    "dropped_unmaskable",
    "dropped_truncated",
)


def _validate_weights(messages: list[dict[str, Any]]) -> None:
    for index, message in enumerate(messages):
        if "weight" not in message:
            continue
        weight = message["weight"]
        if message.get("role") != "assistant" or weight not in (0, 1):
            raise ValueError(
                f"message {index}: weight must be 0 or 1 on an assistant message"
            )


def _tokenize_sft_row(
    tokenizer: Any, row: dict[str, Any], max_length: int
) -> tuple[list[dict[str, list[int]]], str, bool]:
    """(records, outcome, lost supervised tokens to truncation)."""
    tools = _row_tools(row)
    _validate_weights(row["messages"])
    located = supervised_turn_sequences(tokenizer, row["messages"], tools=tools)
    if located is None:
        return [], "dropped_unmaskable", False
    outcome, sequences = located
    if outcome == "no_target":
        return [], "dropped_no_target", False
    records = []
    truncated = False
    for ids, assistant in sequences:
        truncated = truncated or any(assistant[max_length:])
        ids, assistant = ids[:max_length], assistant[:max_length]
        if not any(assistant):
            continue
        # Real EOS labels survive even when EOS doubles as padding: the
        # collator pads labels with -100 by position, not token id.
        records.append(
            {
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "labels": [
                    t if a else -100 for t, a in zip(ids, assistant, strict=True)
                ],
            }
        )
    if not records:
        return [], "dropped_truncated", truncated
    return records, outcome, truncated


def tokenize_sft_row(
    tokenizer: Any, row: dict[str, Any], max_length: int
) -> dict[str, list[int]] | None:
    """Tokenize one chat row with labels on assistant tokens only.

    System prompts, user turns and tool results are context, not targets
    (training on them teaches the model to write its own tool outputs), and
    assistant messages with ``weight: 0`` are context too. Every target turn
    is trained in exactly the context it is generated in: when the template
    renders a turn differently once it is history (Qwen3/Qwen3.5) the row is
    split per target turn and this returns the last surviving sequence — use
    :func:`tokenize_sft_rows` for all of them. ``None`` when no target turn
    can be located or every supervised token was truncated at ``max_length``.
    """
    records, _outcome, _truncated = _tokenize_sft_row(tokenizer, row, max_length)
    return records[-1] if records else None


def tokenize_sft_rows(
    tokenizer: Any, rows: list[dict[str, Any]], max_length: int
) -> tuple[list[dict[str, list[int]]], dict[str, int]]:
    """Tokenize ``rows`` and count how each was labeled or why it was dropped.

    Returns the training sequences (one per row, or one per target turn for a
    ``split_per_turn`` row) and a count per :data:`SFT_LABEL_OUTCOMES` entry
    plus ``partially_truncated``, so a run's manifest says how many rows were
    split, lost supervised tokens, or were not trained.
    """
    counts = dict.fromkeys((*SFT_LABEL_OUTCOMES, "partially_truncated"), 0)
    records = []
    for row in rows:
        kept, outcome, truncated = _tokenize_sft_row(tokenizer, row, max_length)
        counts[outcome] += 1
        if kept:
            records.extend(kept)
            counts["partially_truncated"] += truncated
    if counts["split_per_turn"]:
        logger.info(
            "Chat template renders history differently from new turns in %d of "
            "%d SFT row(s); each was split into one sequence per assistant turn.",
            counts["split_per_turn"],
            len(rows),
        )
    for outcome, reason in (
        ("dropped_no_target", "have no assistant turn without weight 0"),
        ("dropped_unmaskable", "have an assistant turn the template render hides"),
        ("dropped_truncated", f"lost every target token at max_length={max_length}"),
    ):
        if counts[outcome]:
            logger.warning(
                "Dropped %d of %d SFT row(s) that %s.",
                counts[outcome],
                len(rows),
                reason,
            )
    if counts["partially_truncated"]:
        logger.warning(
            "%d of %d SFT row(s) lost part of their assistant tokens at "
            "max_length=%d; only the surviving prefix is supervised.",
            counts["partially_truncated"],
            len(rows),
            max_length,
        )
    return records, counts


def build_training_arguments(training_arguments_cls: Any, **kwargs: Any) -> Any:
    """Construct ``TrainingArguments`` tolerating removed keyword arguments.

    transformers 5.x dropped arguments the 4.x line accepted (hit for real on
    a RunPod pod resolving transformers 5.15.0: ``warmup_ratio`` no longer
    exists and the job died after a 63GB model download). Filter kwargs
    against the constructor's actual signature and log what was dropped, so
    an optional tuning knob degrades gracefully instead of killing the run.
    """
    import inspect

    try:
        accepted = set(inspect.signature(training_arguments_cls.__init__).parameters)
    except (TypeError, ValueError):  # exotic __init__; pass everything through
        return training_arguments_cls(**kwargs)
    if any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in inspect.signature(
            training_arguments_cls.__init__
        ).parameters.values()
    ):
        return training_arguments_cls(**kwargs)
    dropped = sorted(k for k in kwargs if k not in accepted)
    if dropped:
        logger.warning(
            "TrainingArguments does not accept %s in this transformers "
            "version; continuing without them.",
            ", ".join(dropped),
        )
    return training_arguments_cls(**{k: v for k, v in kwargs.items() if k in accepted})


def cuda_device_count() -> int:
    """Number of visible CUDA devices, 0 when torch/CUDA is absent."""
    try:
        import torch

        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except ImportError:
        return 0


def model_load_kwargs() -> dict[str, Any]:
    """Keyword arguments for ``from_pretrained`` in this job.

    On a multi-GPU host, ``device_map="auto"`` shards the checkpoint across
    every visible GPU at load time — this is what lets a model bigger than
    one card train at all. HF ``Trainer`` cooperates: a model carrying an
    ``hf_device_map`` spanning more than one device is treated as
    model-parallel (``place_model_on_device=False``, ``_n_gpu`` forced to 1),
    so the Trainer neither moves the sharded model nor wraps it in
    DataParallel. On a single-GPU (or CPU) host nothing changes: the model
    loads exactly as before and the Trainer places it itself.
    """
    kwargs: dict[str, Any] = {
        "trust_remote_code": False,
        "torch_dtype": "bfloat16",
    }
    n_gpus = cuda_device_count()
    if n_gpus > 1:
        logger.info(
            "%d CUDA devices visible; loading with device_map='auto' to "
            "shard the checkpoint across them.",
            n_gpus,
        )
        kwargs["device_map"] = "auto"
    return kwargs


def is_sharded_across_devices(model: Any) -> bool:
    """True when ``model`` was loaded sharded across more than one device.

    Such a model must never be ``.to("cuda")``-moved: accelerate placed each
    layer already, and a blanket move would try to collapse the shards onto
    ``cuda:0`` (OOM for any model that needed sharding in the first place).
    """
    device_map = getattr(model, "hf_device_map", None)
    if not device_map:
        return False
    return len(set(device_map.values())) > 1


def is_placed_at_load(model: Any) -> bool:
    """True when ``model`` must stay where loading put it.

    Sharded models (see :func:`is_sharded_across_devices`) and bitsandbytes
    k-bit models are placed by ``from_pretrained``; moving them with
    ``.to("cuda")`` either collapses shards onto one card or is refused.
    """
    return is_sharded_across_devices(model) or any(
        getattr(model, flag, False) is True
        for flag in ("is_loaded_in_4bit", "is_loaded_in_8bit")
    )


def prepare_for_lora(model: Any, quantization: QuantizationConfig) -> Any:
    """Apply QLoRA's k-bit preparation when the base was loaded quantized."""
    if not quantization.enabled:
        return model
    logger.info(
        "Preparing the %s base model for LoRA (k-bit training)", quantization.mode
    )
    return prepare_quantized_model(model, gradient_checkpointing=True)


def quantization_hyperparameters(quantization: QuantizationConfig) -> dict[str, Any]:
    """Manifest entries for a quantized run (none for full-precision runs)."""
    return {"quantization": quantization.to_dict()} if quantization.enabled else {}


def log_device_map_summary(model: Any) -> dict[str, int]:
    """Log how many modules landed on each device, and return the counts.

    Without this a multi-GPU run is indistinguishable from a single-GPU one
    in the logs — renting two cards is not evidence that both were used.
    Verified live: a 63GB checkpoint on two 48GB cards logs
    ``0=24 module(s), 1=36 module(s)``.
    """
    device_map = getattr(model, "hf_device_map", None)
    if not device_map:
        return {}
    counts: dict[str, int] = {}
    for device in device_map.values():
        counts[str(device)] = counts.get(str(device), 0) + 1
    if len(counts) > 1:
        logger.info(
            "Model sharded across devices: %s",
            ", ".join(f"{dev}={n} module(s)" for dev, n in sorted(counts.items())),
        )
    else:
        logger.info("Model loaded on a single device: %s", next(iter(counts)))
    return counts


def log_cuda_memory_per_device() -> None:
    """Log allocated CUDA memory per device — the observed VRAM split."""
    try:
        import torch

        if not torch.cuda.is_available():
            return
        parts = [
            f"cuda:{i}={torch.cuda.memory_allocated(i) / 2**30:.1f}GiB"
            for i in range(torch.cuda.device_count())
        ]
        logger.info("CUDA memory allocated: %s", ", ".join(parts))
    except IMPORT_EXCEPTIONS:  # no torch, or a CUDA error: diagnostics only
        return


def load_base_model_for_sft(
    base_model: str,
    *,
    revision: str | None = None,
    quantization: QuantizationConfig | str | None = None,
) -> Any:
    """Load ``base_model`` for text-only SFT, tolerating multimodal repos.

    Composite multimodal checkpoints (e.g. ``zai-org/GLM-5.3-Flash`` and
    ``meta-models/Muse-Glimmer-30B``) register only under Transformers'
    multimodal/image-text auto mappings, so ``AutoModelForCausalLM`` raises
    ValueError on them. Text-only SFT of the language stack still works: retry
    through the composite conditional-generation model and let LoRA target
    the text-stack projections.

    On a multi-GPU host the checkpoint is sharded across every visible GPU
    via ``device_map="auto"`` (see :func:`model_load_kwargs`).

    ``quantization`` (``"4bit"``/``"8bit"`` or a :class:`QuantizationConfig`)
    loads the frozen base through bitsandbytes for QLoRA; it needs CUDA and
    ``bitsandbytes`` and says so before any download starts.
    """
    quant = QuantizationConfig.coerce(quantization)
    if quant.enabled:
        require_bitsandbytes()
    from transformers import AutoModelForCausalLM

    from stateset_agents.core.transformers_compat import load_generation_model

    kwargs = quantized_load_kwargs(quant, model_load_kwargs())
    kwargs["trust_remote_code"] = _trust_remote_code(base_model)
    if revision is not None:
        kwargs["revision"] = revision
    model, resolved_model_cls = load_generation_model(
        AutoModelForCausalLM,
        base_model,
        kwargs,
        for_training=True,
    )
    if resolved_model_cls is not AutoModelForCausalLM:
        logger.info(
            "AutoModelForCausalLM rejected %s; loaded the composite "
            "conditional-generation model and training its text stack.",
            base_model,
        )
    log_device_map_summary(model)
    return model


def infer_lora_target_modules(model: Any) -> list[str]:
    """Pick the module names LoRA should adapt on ``model``.

    peft can infer these itself, but only for architectures in its built-in
    mapping; for anything else it raises "Please specify ``target_modules``".
    That failure only appears once the real model is loaded on a GPU, so it
    escapes every CPU dry run.

    Returns the recognised names present on this model, or an empty list —
    which means "we did not recognise anything, let peft try" rather than a
    guess that would fail confusingly later.
    """
    # peft matches target_modules by leaf name across the WHOLE model, so a
    # name that only exists in a vision tower/adapter must be dropped from
    # the list entirely — skipping it during the walk is not enough
    # (verified live on meta-models/Muse-Glimmer-30B: fc1/fc2 exist only
    # under model.vision_tower/model.vision_adapter, and listing them
    # adapted the ViT despite text-only SFT sending it no gradient).
    text_found: set[str] = set()
    non_text_found: set[str] = set()
    for name, _module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in _LORA_TARGET_CANDIDATES:
            continue
        if any(part in _NON_TEXT_STACK_MARKERS for part in name.split(".")):
            non_text_found.add(leaf)
        else:
            text_found.add(leaf)
    dropped = non_text_found - text_found
    if dropped:
        logger.info(
            "Skipping LoRA candidates that exist only in non-text stacks: %s",
            ", ".join(sorted(dropped)),
        )
    shared = non_text_found & text_found
    if shared:
        logger.warning(
            "LoRA candidates %s exist in both text and non-text stacks; "
            "using full text paths to exclude encoders.",
            ", ".join(sorted(shared)),
        )
    # The output head is deliberately excluded: adapting it inflates the
    # adapter with a vocab-sized matrix for no benefit on SFT.
    text_found.discard("lm_head")
    return text_lora_targets(model, sorted(text_found))


def encode_chat_prompt(tokenizer: Any, prompt: str, device: Any) -> Any:
    """Render one user prompt with the generation prompt and encode it.

    The single prompt path for evaluation, harvesting and the SFT eval: the
    chat template already supplies the model's special tokens, so encoding
    adds none (a second BOS on Llama/Gemma-style tokenizers otherwise), and
    reasoning models get ``enable_thinking=False`` where the template accepts
    it, so the budget is not spent on a reasoning preamble (hit for real with
    NVIDIA Nemotron 3.5 Lightning on an H100 pod). Templates that reject the
    kwarg (Muse Glimmer's) get the plain call.
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)


def generate_from_prompt(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    num_return_sequences: int = 1,
    temperature: float | None = None,
    top_p: float | None = None,
) -> list[str]:
    """Completions for one prompt: greedy, or sampled when ``temperature`` is set.

    Shared by :func:`generate_completions` (greedy evals) and the harvest's
    best-of-N sampling, so both see the prompt SFT trains on and the same
    generation-compatibility settings.
    """
    import torch

    inputs = encode_chat_prompt(tokenizer, prompt, model.device)
    sampling: dict[str, Any] = {"do_sample": False}
    if temperature is not None:
        sampling = {"do_sample": True, "temperature": temperature}
        if top_p is not None:
            sampling["top_p"] = top_p
    with torch.no_grad():
        output = model.generate(
            **inputs,
            **sampling,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            **generation_compat_kwargs(model),
        )
    prompt_length = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(row[prompt_length:], skip_special_tokens=True)
        for row in output
    ]


def generate_completions(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int = 90,
) -> list[str]:
    """Greedily generate one completion per prompt through ``model``.

    Used for the post-train base-vs-tuned comparison: each prompt is encoded
    by :func:`encode_chat_prompt` and decoded back to just the completion
    text. Greedy decoding keeps the two runs comparable — sampling noise
    would swamp the tuning signal.
    """
    return [
        generate_from_prompt(model, tokenizer, prompt, max_new_tokens=max_new_tokens)[0]
        for prompt in prompts
    ]


def load_tokenizer_for(base_model: str, *, revision: str | None = None) -> Any:
    """The tokenizer SFT, evaluation and harvesting load for ``base_model``.

    Repository code runs only for curated presets that need it.
    """
    from transformers import AutoTokenizer

    revision_kwargs = {"revision": revision} if revision is not None else {}
    return AutoTokenizer.from_pretrained(  # nosec: B615 - caller pins revision
        base_model, trust_remote_code=_trust_remote_code(base_model), **revision_kwargs
    )


#: Keys an eval prompt-spec dict may carry. Anything else is a typo the user
#: should hear about before renting a GPU.
_EVAL_SPEC_KEYS = frozenset(
    {"prompt", "expect", "forbid", "judge", "min_judge_score", "nsr"}
)


def normalize_eval_prompts(
    entries: list[str | dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize eval prompt entries to spec dicts.

    An entry may be a plain string (a bare prompt, no assertions — the
    original format) or a dict ``{"prompt": str, "expect": [substrings],
    "forbid": [substrings], "judge": str, "min_judge_score": float}`` where
    everything but ``prompt`` is optional. Raises ``ValueError`` on anything
    else — this runs before the GPU is rented, so it should be loud.
    """
    specs: list[dict[str, Any]] = []
    for i, entry in enumerate(entries):
        if isinstance(entry, str):
            specs.append({"prompt": entry})
            continue
        if not isinstance(entry, dict):
            raise ValueError(
                f"eval prompt {i} must be a string or an object, got "
                f"{type(entry).__name__}"
            )
        unknown = sorted(set(entry) - _EVAL_SPEC_KEYS)
        if unknown:
            raise ValueError(
                f"eval prompt {i} has unknown key(s): {', '.join(unknown)}"
            )
        if not isinstance(entry.get("prompt"), str) or not entry["prompt"].strip():
            raise ValueError(f"eval prompt {i} needs a non-empty 'prompt' string")
        for key in ("expect", "forbid"):
            value = entry.get(key, [])
            if not isinstance(value, list) or not all(
                isinstance(s, str) for s in value
            ):
                raise ValueError(f"eval prompt {i}: '{key}' must be a list of strings")
        if "judge" in entry and not isinstance(entry["judge"], str):
            raise ValueError(f"eval prompt {i}: 'judge' must be a string")
        if "min_judge_score" in entry and not isinstance(
            entry["min_judge_score"], (int, float)
        ):
            raise ValueError(f"eval prompt {i}: 'min_judge_score' must be a number")
        if "nsr" in entry and not isinstance(entry["nsr"], dict):
            raise ValueError(
                f"eval prompt {i}: 'nsr' must be an object (a /v1/decisions request body)"
            )
        specs.append(dict(entry))
    return specs


#: Words that negate a following verb: "not", "never", "no", "without",
#: "cannot" and any "...n't" contraction.
_NEGATOR = r"(?:not|never|no|without|cannot|\w+n['\u2019]t)"
#: Auxiliaries and aspect allowed between a negator and what it governs
#: ("won't be able to rebook", "not going to rebook", "no need to rebook").
_GOVERNING = (
    r"(?:be\s+able\s+to|able\s+to|be|need\s+to|have\s+to|going\s+to"
    r"|go\s+ahead\s+and|try\s+to|bother\s+to|ever)"
)
#: A negated verb may take the phrase as its object through a few function
#: words: "I can't open a baggage trace", "I won't issue you a refund".
#: Focus adverbs are not verbs: "I didn't just rebook you" still rebooks.
_GOVERNED_VERB = (
    rf"(?!(?:only|just|simply|merely|also|even|wait|hesitate|{_NEGATOR})\b)"
    r"[\w'\u2019-]+"
)
_OBJECT_WORDS = (
    r"(?:you|your|him|her|them|their|it|its|us|our|me|my|a|an|the|any"
    r"|another|that|this|these|those|some)"
)
_NEGATED_BEFORE = re.compile(
    rf"(?:(?P<previous>[\w'\u2019]+)\s+)?\b{_NEGATOR}\s+(?:{_GOVERNING}\s+)?"
    rf"(?:{_GOVERNED_VERB}\s+(?:{_OBJECT_WORDS}\s+){{0,2}})?$",
    re.IGNORECASE,
)
_NEGATOR_WORD = re.compile(rf"{_NEGATOR}", re.IGNORECASE)


def _negation_governs(before: str) -> bool:
    """Whether the text right before a phrase negates it.

    The negator must govern the phrase: adjacent ("won't rebook"), through
    an auxiliary ("won't be able to rebook"), or as the object of the negated
    verb ("can't open a baggage trace"). It must not itself be negated ("I
    can't not rebook you" still rebooks), and "won't hesitate to", "not only
    will I" or "can't wait to" are not negations of what follows.
    """
    match = _NEGATED_BEFORE.search(before)
    if match is None:
        return False
    previous = match.group("previous")
    return not (previous and _NEGATOR_WORD.fullmatch(previous))


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    """``phrase`` at a word start; a trailing number must end there too.

    "rebook" still matches "rebooked", but reference "7701" does not match
    inside "77015".
    """
    start = r"(?<!\w)" if re.match(r"\w", phrase) else ""
    end = r"(?!\d)" if phrase[-1:].isdigit() else ""
    return re.compile(start + re.escape(phrase) + end, re.IGNORECASE)


def _phrase_asserted(completion: str, phrase: str) -> bool:
    """True when ``phrase`` occurs in ``completion`` and is not negated.

    One rule for both sides of a check: a reply that says "I won't rebook
    you" honors a declined remedy (``forbid``), and one that says "I can't
    open a baggage trace" did not deliver it (``expect``). The same check
    filters harvested samples, so a negation it misses becomes training
    data.
    """
    if not phrase:
        return True
    for match in _phrase_pattern(phrase).finditer(completion):
        # The governing window is a few words; bound the regex's input.
        before = completion[max(0, match.start() - 200) : match.start()]
        if not _negation_governs(before):
            return True
    return False


def evaluate_checks(
    completion: str, expect: list[str], forbid: list[str]
) -> dict[str, Any]:
    """Case-insensitively match ``expect`` and ``forbid`` phrases.

    Passes when every ``expect`` phrase is asserted in ``completion`` and no
    ``forbid`` phrase is. A phrase counts where it starts at a word boundary
    (and, when it ends in a digit, ends at a number boundary) and is not
    governed by a negation; see :func:`_phrase_asserted`.
    """
    expect_hits = [s for s in expect if _phrase_asserted(completion, s)]
    forbid_hits = [s for s in forbid if _phrase_asserted(completion, s)]
    return {
        "expect_hits": expect_hits,
        "forbid_hits": forbid_hits,
        "passed": len(expect_hits) == len(expect) and not forbid_hits,
    }


def has_eval_assertions(spec: Any) -> bool:
    """Whether a prompt spec (or episode script) carries checks.

    ``expect``/``forbid`` for single prompts, ``turn_expect`` for episode
    scripts. Only such specs can fail, so only they count toward an eval
    score; a spec without checks would "pass" whatever the model says.
    """
    if not isinstance(spec, dict):
        return False
    turn_expect = spec.get("turn_expect")
    return bool(
        spec.get("expect")
        or spec.get("forbid")
        or (isinstance(turn_expect, list) and any(turn_expect))
    )


def score_eval_rows(rows: Any) -> tuple[int | None, int | None]:
    """``(passed, total)`` over eval rows that carry an assertion outcome.

    The one scorer for every generation of a flywheel: the harvest job's
    "before" eval and each training job's ``eval_results.json`` count the
    same rows — those with ``checks.passed`` (or a flat ``passed``) — so a
    real improvement is never compared against a larger denominator.
    ``(None, None)`` when no row carries an outcome.
    """
    if not isinstance(rows, list):
        return None, None
    outcomes = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        checks = row.get("checks")
        if isinstance(checks, dict) and "passed" in checks:
            outcomes.append(bool(checks["passed"]))
        elif "passed" in row:
            outcomes.append(bool(row["passed"]))
    if not outcomes:
        return None, None
    return sum(outcomes), len(outcomes)


def _create_domain_reward(name: str) -> Any:
    """Import hook for the optional judge — split out so tests can fake it."""
    from stateset_agents.rewards.multi_objective_reward import create_domain_reward

    return create_domain_reward(name)


def judge_completion(judge: str, prompt: str, completion: str) -> float | None:
    """Score ``completion`` with a domain reward, if one is importable here.

    The judge is a nicety on top of the substring checks, and the reward
    stack may simply not be installed on the pod — so a missing or broken
    reward stack, or a scoring failure, degrades to a logged warning and
    ``None``. A bug in building the judge (a ``TypeError`` from the factory)
    propagates.
    """
    try:
        reward = _create_domain_reward(judge)
    except IMPORT_EXCEPTIONS as exc:  # the reward stack is absent or broken here
        logger.warning(
            "Judge %r unavailable on this worker (%s); skipping judge score.",
            judge,
            exc,
        )
        return None
    try:
        import asyncio

        result = asyncio.run(
            reward.compute_reward(
                turns=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ]
            )
        )
        return float(result.score)
    # Fail-closed judge on an unattended worker: any scoring failure records
    # no score, which rejects the sample (see sample_passes).
    except Exception as exc:  # noqa: BLE001
        logger.warning("Judge %r failed to score (%s); skipping.", judge, exc)
        return None


def build_eval_extras(
    specs: list[dict[str, Any]], finetuned: list[str]
) -> list[dict[str, Any]]:
    """Per-row extra fields for ``eval_results.json``.

    Rows for plain prompts stay exactly as before (empty extras); rows whose
    spec carries assertions gain ``checks``, and specs naming a ``judge``
    gain ``judge_score`` when the judge could run and
    ``judge_unavailable: true`` when it could not.
    """
    extras: list[dict[str, Any]] = []
    for spec, completion in zip(specs, finetuned, strict=True):
        extra: dict[str, Any] = {}
        expect = spec.get("expect", [])
        forbid = spec.get("forbid", [])
        if expect or forbid:
            extra["checks"] = evaluate_checks(completion, expect, forbid)
        if spec.get("judge"):
            score = judge_completion(spec["judge"], spec["prompt"], completion)
            if score is not None:
                extra["judge_score"] = score
            else:  # recorded, so a skipped min_judge_score gate is visible
                extra["judge_unavailable"] = True
        if spec.get("nsr"):
            # Lazy import: harvest imports this module at load time.
            from stateset_agents.training import harvest

            extra["nsr_verified"] = harvest.nsr_gate_passes(spec, completion)
        extras.append(extra)
    return extras


def eval_gate_failures(
    specs: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> list[str]:
    """Human-readable assertion failures across the eval rows.

    A row fails when its substring checks did not pass, or when the spec set
    ``min_judge_score`` and a judge score exists below it. A judge that could
    not run never fails the gate — judge failures degrade, by design. NSR
    verification is the opposite: fail-closed, so ``nsr_verified: false``
    (disagreement OR an unreachable verifier) fails the gate — an eval that
    silently skips its strongest check would overstate the model.
    """
    failures: list[str] = []
    for spec, row in zip(specs, rows, strict=True):
        prompt = spec["prompt"]
        checks = row.get("checks")
        if checks and not checks["passed"]:
            missing = [
                s for s in spec.get("expect", []) if s not in checks["expect_hits"]
            ]
            parts = []
            if missing:
                parts.append(f"missing expected {missing!r}")
            if checks["forbid_hits"]:
                parts.append(f"contains forbidden {checks['forbid_hits']!r}")
            failures.append(f"{prompt!r}: {'; '.join(parts)}")
        min_score = spec.get("min_judge_score")
        score = row.get("judge_score")
        if min_score is not None and score is not None and score < min_score:
            failures.append(
                f"{prompt!r}: judge_score {score:.3f} < min_judge_score {min_score}"
            )
        if row.get("nsr_verified") is False:
            failures.append(
                f"{prompt!r}: nsr_verified is false — the completion's verdict "
                "did not agree with a verified NSR decision"
            )
    return failures


def eval_gate_skips(
    specs: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> list[str]:
    """``min_judge_score`` gates that could not be applied (no judge score).

    A judge that cannot run degrades instead of failing the gate (see
    :func:`eval_gate_failures`); this lists those prompts so the run says
    which of its gates were not actually checked.
    """
    return [
        f"{spec['prompt']!r}: min_judge_score {spec['min_judge_score']} not "
        "checked — no judge score (judge unavailable)"
        for spec, row in zip(specs, rows, strict=True)
        if spec.get("min_judge_score") is not None
        and not isinstance(row.get("judge_score"), (int, float))
    ]


def resolve_resume_checkpoint(output_dir: Path, resume: bool) -> bool:
    """Whether training should resume from a checkpoint in ``output_dir``.

    HF Trainer writes ``checkpoint-<N>`` directories there (we run with
    ``save_strategy="epoch"``), and ``trainer.train(resume_from_checkpoint=
    True)`` raises when none exists — so ``--resume`` against a fresh output
    dir must degrade to a logged fresh start, not a crash.
    """
    if not resume:
        return False
    checkpoints = sorted(p for p in Path(output_dir).glob("checkpoint-*") if p.is_dir())
    if checkpoints:
        logger.info(
            "Resuming from the newest checkpoint in %s (found %s).",
            output_dir,
            checkpoints[-1].name,
        )
        return True
    logger.info(
        "--resume requested but no checkpoint-* directory exists in %s; "
        "training from scratch.",
        output_dir,
    )
    return False


def run_sft(
    rows: list[dict[str, Any]],
    base_model: str,
    output_dir: Path,
    num_epochs: int,
    lora_r: int,
    lora_alpha: int,
    learning_rate: float,
    max_length: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    eval_prompts: list[str | dict[str, Any]] | None = None,
    eval_max_new_tokens: int = 90,
    resume: bool = False,
    dataset_path: Path | None = None,
    parent_adapter: str | None = None,
    model_revision: str | None = None,
    callbacks: list[Any] | None = None,
    resume_from_checkpoint: Path | None = None,
    seed: int = 42,
    dataset_identity: SFTDatasetIdentity | None = None,
    quantization: QuantizationConfig | str | None = None,
) -> Path:
    """Run the actual SFT training on GPU.

    ``quantization`` (``"4bit"``, ``"8bit"`` or a :class:`QuantizationConfig`)
    trains the LoRA adapter on a bitsandbytes-quantized frozen base (QLoRA).

    ``resume=True`` continues from the newest ``checkpoint-<N>`` directory
    already in ``output_dir`` when one exists; with none, it logs the fact
    and trains fresh (see ``resolve_resume_checkpoint``).

    ``resume_from_checkpoint`` selects an exact checkpoint and fails if it is
    missing. It cannot be combined with the legacy ``resume`` flag. The studio
    verifies and privately snapshots all optimizer/model/RNG bytes before use.

    When ``eval_prompts`` is given, a completion per prompt is generated with
    the base model *before* LoRA is applied and again through the trained
    adapter afterwards; the pairs land in ``output_dir/eval_results.json``.
    Entries may be plain strings or spec dicts (see
    ``normalize_eval_prompts``); assertion results are recorded per row, and
    the assertion *gate* — exiting non-zero on failure — belongs to
    ``run_sft_job``, so a failed assertion never destroys the adapter this
    function already saved.

    Uses ``transformers.Trainer`` directly with a PEFT LoRA adapter on top of
    the base model. The result is saved as a LoRA adapter directory
    consumable by ``AgentConfig.peft_path``.

    File-based callers should pass ``dataset_identity`` from the same captured
    read as ``rows``. Without it, ``dataset_path`` is read and checked against
    the supplied rows before loading a model. In-memory input without a source
    remains explicitly unbound to a file. No source is hashed after training.

    Raises ``ImportError`` if transformers / peft / datasets aren't available.
    """
    rows = deepcopy(rows)
    for index, row in enumerate(rows, 1):
        try:
            validate_finetuning_split(row, expected_split="train")
        except ValueError as exc:
            raise ValueError(f"row {index}: {exc}") from None
    seed = _training_seed(seed)
    quant = QuantizationConfig.coerce(quantization)
    if dataset_identity is None:
        if dataset_path is not None:
            _, dataset_identity = load_sft_dataset(dataset_path)
        else:
            dataset_identity = SFTDatasetIdentity.from_rows(rows)
    if not isinstance(dataset_identity, SFTDatasetIdentity):
        raise ValueError("Expected a captured SFT dataset identity")
    dataset_identity.verify_rows(rows)
    source = dataset_identity.source
    if dataset_path is not None and (
        source is None or Path(source.path) != Path(dataset_path)
    ):
        raise ValueError("Dataset path differs from the captured SFT source")
    if resume_from_checkpoint is not None:
        if (
            resume
            or not resume_from_checkpoint.is_dir()
            or not (resume_from_checkpoint / "trainer_state.json").is_file()
        ):
            raise ValueError(
                "Explicit SFT recovery requires an existing exact checkpoint; fresh fallback is disabled"
            )

    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(seed)
    logger.info("Loading tokenizer and model: %s", base_model)
    # base_model is a caller-supplied CLI argument (public HF model repo id),
    # not attacker-controlled input; pinning a fixed revision would break
    # support for arbitrary user-chosen base models.
    revision_kwargs = {"revision": model_revision} if model_revision is not None else {}
    tokenizer = load_tokenizer_for(base_model, revision=model_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_base_model_for_sft(base_model, quantization=quant, **revision_kwargs)
    log_device_map_summary(model)
    log_cuda_memory_per_device()

    eval_specs = normalize_eval_prompts(eval_prompts or [])
    eval_prompt_texts = [spec["prompt"] for spec in eval_specs]

    base_completions: list[str] = []
    if eval_prompt_texts:
        logger.info(
            "Generating base-model completions for %d eval prompt(s)…",
            len(eval_prompt_texts),
        )
        # The Trainer moves the model to GPU later; base-eval generation runs
        # BEFORE that, and a 30B generate on CPU takes tens of minutes of
        # billed pod time (hit for real on an H100 pod: GPU idle, CPU
        # grinding). Move it now when a GPU exists — unless accelerate
        # already sharded it across devices, in which case moving it would
        # collapse the shards onto cuda:0 and OOM.
        if gpu_available() and not is_placed_at_load(model):
            model = model.to("cuda")
        base_completions = generate_completions(
            model, tokenizer, eval_prompt_texts, max_new_tokens=eval_max_new_tokens
        )

    model = prepare_for_lora(model, quant)
    target_modules = infer_lora_target_modules(model)
    logger.info(
        "Applying LoRA adapter (r=%d, alpha=%d, targets=%s)",
        lora_r,
        lora_alpha,
        ",".join(target_modules) or "<peft default>",
    )
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        # Passed explicitly: peft only infers these for architectures in its
        # built-in mapping and raises on anything else (hit for real on
        # Qwen3.5). None means "nothing recognised — let peft try".
        target_modules=target_modules or None,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Render Python records directly: Arrow otherwise unifies different tool
    # schemas/argument objects and inserts null fields into the conversations.
    records, label_outcomes = tokenize_sft_rows(tokenizer, rows, max_length)
    if not records:
        raise ValueError(
            "No SFT row keeps a locatable assistant token within "
            f"max_length={max_length}."
        )
    dataset = Dataset.from_list(records)

    args = build_training_arguments(
        TrainingArguments,
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=seed,
        data_seed=seed,
        report_to=[],  # local-only by default
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100),
        callbacks=callbacks,  # ``None`` is Trainer's own default
    )

    logger.info("Starting SFT…")
    if resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint=str(resume_from_checkpoint))
    elif resolve_resume_checkpoint(output_dir, resume):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    logger.info("Training complete. Saving adapter to %s", output_dir)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    # Provenance is written as soon as the adapter exists, so an adapter is
    # never on disk without a record of what made it — even if the eval
    # below fails or the process dies during it.
    manifest = AdapterManifest(
        base_model=base_model,
        dataset_path=source.path if source is not None else None,
        dataset_sha256=source.sha256 if source is not None else None,
        dataset_rows=source.rows if source is not None else dataset_identity.examples,
        hyperparameters={
            "dataset_provenance": (
                "captured_before_training" if source else "in_memory_unbound"
            ),
            "training_content_sha256": dataset_identity.content_sha256,
            "training_input_rows": dataset_identity.examples,
            # Input rows that produced at least one training sequence, the
            # sequences themselves (a split row yields one per target turn),
            # and rows not trained at all, by reason.
            "effective_training_rows": label_outcomes["all_assistant_turns"]
            + label_outcomes["split_per_turn"],
            "training_sequences": len(records),
            "split_per_turn_rows": label_outcomes["split_per_turn"],
            "untargeted_rows_dropped": label_outcomes["dropped_no_target"],
            "unmaskable_rows_dropped": label_outcomes["dropped_unmaskable"],
            "truncated_rows_dropped": label_outcomes["dropped_truncated"],
            "partially_truncated_rows": label_outcomes["partially_truncated"],
            "seed": seed,
            "data_seed": seed,
            "num_epochs": num_epochs,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "per_device_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "resumed": bool(resume or resume_from_checkpoint is not None),
            **(
                {"resume_from_checkpoint": str(resume_from_checkpoint)}
                if resume_from_checkpoint is not None
                else {}
            ),
            "model_revision": model_revision,
            **quantization_hyperparameters(quant),
        },
        parent_adapter=parent_adapter,
        package_version=_package_version(),
    )
    write_manifest(output_dir, manifest)
    from stateset_agents.training.sft_artifacts import verify_sft_adapter

    verify_sft_adapter(
        output_dir,
        base_model=base_model,
        lora_r=lora_r,
        dataset_identity=dataset_identity,
        model_revision=model_revision,
        expected_lineage=manifest.to_dict(),
    )

    if eval_prompt_texts:
        logger.info(
            "Generating fine-tuned completions for %d eval prompt(s)…",
            len(eval_prompt_texts),
        )
        model.eval()
        tuned_completions = generate_completions(
            model, tokenizer, eval_prompt_texts, max_new_tokens=eval_max_new_tokens
        )
        extras = build_eval_extras(eval_specs, tuned_completions)
        results_path = write_eval_results(
            output_dir,
            eval_prompt_texts,
            base_completions,
            tuned_completions,
            extras=extras,
        )
        logger.info("Eval comparison written to %s", results_path)
        checked = [e for e in extras if "checks" in e]
        if checked:
            passed = sum(1 for e in checked if e["checks"]["passed"])
            logger.info("Eval checks: %d/%d prompt(s) passed.", passed, len(checked))
            manifest.eval_passed = passed
            manifest.eval_total = len(checked)
        if any(spec.get("min_judge_score") is not None for spec in eval_specs):
            skipped = eval_gate_skips(eval_specs, extras)
            for skip in skipped:
                logger.warning("Eval judge gate skipped: %s", skip)
            manifest.eval_judge_gates_skipped = len(skipped)
        if checked or manifest.eval_judge_gates_skipped is not None:
            write_manifest(output_dir, manifest)

    return output_dir


def write_eval_results(
    output_dir: Path,
    prompts: list[str],
    base: list[str],
    finetuned: list[str],
    extras: list[dict[str, Any]] | None = None,
) -> Path:
    """Write the base-vs-tuned comparison as ``eval_results.json``.

    ``extras`` (from ``build_eval_extras``) merges per-row assertion fields —
    ``checks``, ``judge_score`` — into the corresponding row; rows for plain
    prompts stay the original three-key shape.
    """
    if extras is None:
        extras = [{} for _ in prompts]
    path = Path(output_dir) / "eval_results.json"
    path.write_text(
        json.dumps(
            [
                {"prompt": p, "base": b, "finetuned": f, **extra}
                for p, b, f, extra in zip(prompts, base, finetuned, extras, strict=True)
            ],
            indent=2,
        )
        + "\n"
    )
    return path


def _apply_eval_gate(
    eval_prompts: list[str | dict[str, Any]] | None, output_dir: Path
) -> int:
    """Return the job exit code the eval assertions call for.

    Runs strictly AFTER ``run_sft`` has saved the adapter and written
    ``eval_results.json`` — a failed assertion turns the job red without
    destroying the training artifacts.
    """
    if not eval_prompts:
        return 0
    specs = normalize_eval_prompts(eval_prompts)
    results_path = output_dir / "eval_results.json"
    if not results_path.exists():  # e.g. eval skipped; nothing to gate on
        return 0
    rows = json.loads(results_path.read_text())
    for skip in eval_gate_skips(specs, rows):
        logger.warning("Eval judge gate skipped: %s", skip)
    failures = eval_gate_failures(specs, rows)
    if not failures:
        return 0
    for failure in failures:
        logger.error("Eval assertion failed: %s", failure)
    logger.error(
        "%d eval assertion(s) failed — exiting non-zero. The adapter and "
        "eval_results.json in %s were saved before this gate ran.",
        len(failures),
        output_dir,
    )
    return 1


def _training_seed(value: Any) -> int:
    """Validate before data/model loading, including dry-run entrypoints."""
    if type(value) is not int or not 0 <= value <= 2**32 - 1:
        raise ValueError("Training seed must be an integer in [0, 4294967295]")
    return value


def run_sft_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one SFT job from a ``RemoteJobSpec``-shaped dict.

    This is the single entrypoint every executor calls — locally in a
    subprocess, or inside a provider's container. It never raises: failures
    are reported as a non-zero ``returncode`` with the captured output, so
    the calling executor has one uniform thing to interpret.

    Returns ``{"returncode": int, "logs": list[str], "output_dir": str}``.
    """
    job = {k: v for k, v in payload.items() if k not in _PROVIDER_ONLY_KEYS}
    output_dir = Path(job["output_dir"])
    buffer = io.StringIO()

    handler = logging.StreamHandler(buffer)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
    previous_level = logger.level
    logger.setLevel(logging.INFO)

    returncode = 0
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            seed = _training_seed(42 if job.get("seed") is None else job["seed"])
            quantization = QuantizationConfig.coerce(job.get("quantization"))
            rows, dataset_identity = load_sft_dataset(Path(job["dataset"]))
            if not rows:
                logger.error("No usable rows in %s", job["dataset"])
                returncode = 1
            else:
                logger.info("Loaded %d examples from %s", len(rows), job["dataset"])
                if job.get("dry_run") or not gpu_available():
                    print_training_plan(
                        rows=rows,
                        base_model=job["base_model"],
                        output_dir=output_dir,
                        num_epochs=job["num_epochs"],
                        lora_r=job["lora_r"],
                        learning_rate=job["learning_rate"],
                        max_length=job["max_length"],
                        seed=seed,
                        quantization=quantization,
                    )
                else:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    run_sft(
                        rows=rows,
                        base_model=job["base_model"],
                        output_dir=output_dir,
                        num_epochs=job["num_epochs"],
                        lora_r=job["lora_r"],
                        lora_alpha=job["lora_alpha"],
                        learning_rate=job["learning_rate"],
                        max_length=job["max_length"],
                        per_device_batch_size=job["per_device_batch_size"],
                        gradient_accumulation_steps=job["gradient_accumulation_steps"],
                        eval_prompts=job.get("eval_prompts"),
                        eval_max_new_tokens=job.get("eval_max_new_tokens", 90),
                        resume=bool(job.get("resume", False)),
                        dataset_path=Path(job["dataset"]),
                        dataset_identity=dataset_identity,
                        parent_adapter=job.get("parent_adapter"),
                        seed=seed,
                        quantization=quantization,
                    )
                    from stateset_agents.training.sft_artifacts import (
                        verify_sft_adapter,
                    )

                    verify_sft_adapter(
                        output_dir,
                        base_model=job["base_model"],
                        lora_r=job["lora_r"],
                        dataset_identity=dataset_identity,
                    )
                    returncode = _apply_eval_gate(job.get("eval_prompts"), output_dir)
    # Process boundary: the job runner reports failure as a return code
    # (see the docstring), never as an exception.
    except Exception as exc:  # noqa: BLE001
        logger.error("SFT job failed: %s", exc)
        returncode = 1
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

    return {
        "returncode": returncode,
        "logs": buffer.getvalue().splitlines(),
        "output_dir": str(output_dir),
    }


def build_parser() -> Any:
    """Argument parser for the job. Shared by the CLI and the ``scripts/`` wrapper."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Supervised fine-tune from curated data."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Chat-format JSONL from prepare_sft_dataset.py.",
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Hugging Face base model name (e.g. Qwen/Qwen3.5-0.8B).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sft_v1"))
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Initialization and training seed (0–4294967295).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the training plan without running it (forced "
        "automatically when no GPU is detected).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the newest checkpoint-* directory in --output-dir "
        "when one exists; with none, log it and train from scratch.",
    )
    parser.add_argument(
        "--parent-adapter",
        default=None,
        help="Adapter this run descends from (recorded in the manifest, so "
        "the improvement loop's generations stay linked).",
    )
    parser.add_argument(
        "--eval-prompts-json",
        default=None,
        help="JSON-encoded list of eval entries for a post-train base-vs-"
        "tuned comparison, written to output_dir/eval_results.json. Each "
        "entry is a plain prompt string, or a spec object {'prompt', "
        "'expect', 'forbid', 'judge', 'min_judge_score'} whose assertions "
        "gate the job's exit code (after artifacts are saved). JSON rather "
        "than a file path so remote workers need no second upload.",
    )
    parser.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=90,
        help="Token budget per eval completion. Raise it for reasoning "
        "models whose answers follow a long preamble.",
    )
    parser.add_argument(
        "--quantization",
        choices=("none", "4bit", "8bit"),
        default="none",
        help="Load the frozen base model quantized and train LoRA on top "
        "(QLoRA). Needs a CUDA GPU and bitsandbytes "
        "(pip install 'stateset-agents[qlora]').",
    )
    parser.add_argument(
        "--quant-type",
        choices=("nf4", "fp4"),
        default="nf4",
        help="4-bit storage type (with --quantization 4bit).",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Matmul dtype for 4-bit layers (float16 on GPUs without bf16).",
    )
    parser.add_argument(
        "--no-double-quant",
        action="store_true",
        help="Disable QLoRA double quantization of the 4-bit constants.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the job from command-line arguments.

    Available as ``python -m stateset_agents.training.sft`` — the only
    invocation a remote worker has, since it holds the wheel and no checkout.
    """
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    eval_prompts: list[str | dict[str, Any]] | None = None
    if args.eval_prompts_json:
        try:
            eval_prompts = json.loads(args.eval_prompts_json)
        except json.JSONDecodeError as exc:
            print(f"--eval-prompts-json is not valid JSON: {exc}")
            return 2
        if not isinstance(eval_prompts, list):
            print(
                "--eval-prompts-json must be a JSON list of strings or "
                "prompt-spec objects"
            )
            return 2
        try:
            normalize_eval_prompts(eval_prompts)
        except ValueError as exc:
            print(f"--eval-prompts-json is invalid: {exc}")
            return 2

    outcome = run_sft_job(
        {
            "dataset": str(args.dataset),
            "base_model": args.base_model,
            "output_dir": str(args.output_dir),
            "num_epochs": args.num_epochs,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "per_device_batch_size": args.per_device_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "dry_run": args.dry_run,
            "resume": args.resume,
            "eval_prompts": eval_prompts,
            "eval_max_new_tokens": args.eval_max_new_tokens,
            "parent_adapter": args.parent_adapter,
            "seed": args.seed,
            "quantization": {
                "mode": args.quantization,
                "quant_type": args.quant_type,
                "double_quant": not args.no_double_quant,
                "compute_dtype": args.compute_dtype,
            },
        }
    )

    for line in outcome["logs"]:
        print(line)

    if outcome["returncode"] != 0:
        return int(outcome["returncode"])

    path = outcome["output_dir"]
    print()
    print("Next steps:")
    print(
        f"  Chat with it:   stateset-agents chat "
        f"--model {args.base_model} --checkpoint {path}"
    )
    print(
        f"  Serve it:       stateset-agents serve "
        f"--checkpoint {path} --base-model {args.base_model}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised via subprocess tests
    import sys

    sys.exit(main())
