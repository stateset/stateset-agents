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
from pathlib import Path
from typing import Any

logger = logging.getLogger("sft_from_curated")

__all__ = [
    "build_eval_extras",
    "build_training_arguments",
    "eval_gate_failures",
    "evaluate_checks",
    "generate_completions",
    "gpu_available",
    "judge_completion",
    "load_base_model_for_sft",
    "load_chat_dataset",
    "normalize_eval_prompts",
    "print_training_plan",
    "resolve_resume_checkpoint",
    "run_sft",
    "run_sft_job",
    "write_eval_results",
]

#: Keys of a ``RemoteJobSpec`` dict that configure the provider rather than the
#: job. ``run_sft_job`` ignores them so a full spec can be passed straight in.
_PROVIDER_ONLY_KEYS = frozenset(
    {"gpu", "timeout_s", "package_version", "container_disk_gb", "cloud_type"}
)


def load_chat_dataset(path: Path) -> list[dict[str, Any]]:
    """Load a chat-format JSONL.

    Each row must have a ``messages`` key with a list of ``{role, content}``
    dicts. Returns the list of rows after validation.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    rows: list[dict[str, Any]] = []
    for line_num, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            logger.warning("Skipping line %d: %s", line_num, e)
            continue
        if "messages" not in row or not isinstance(row["messages"], list):
            logger.warning("Skipping line %d: missing 'messages' list", line_num)
            continue
        rows.append(row)
    return rows


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
    print()
    sample = rows[0] if rows else None
    if sample:
        print("  First example:")
        for msg in sample.get("messages", [])[:2]:
            content = msg.get("content", "")[:60]
            print(f"    [{msg.get('role')}] {content}…")
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
    "dense",
    "fc1",
    "fc2",
)

#: Module-path components that mark the non-text stack of a multimodal
#: composite model. Anything under these gets no gradient from text-only SFT.
_NON_TEXT_STACK_MARKERS = frozenset(
    {
        "vision_tower",
        "vision_model",
        "visual",
        "vision_encoder",
        "image_processor",
        "vision_adapter",
        "vision_projection",
        "vision_projector",
        "perception_encoder",
        "multi_modal_projector",
        "mm_projector",
        "audio_tower",
    }
)


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


def load_base_model_for_sft(base_model: str):
    """Load ``base_model`` for text-only SFT, tolerating multimodal repos.

    Composite multimodal checkpoints (e.g. ``meta-models/Muse-Glimmer-30B``,
    model_type ``muse_glimmer``) register only under transformers'
    image-text-to-text auto-mapping, so ``AutoModelForCausalLM`` raises
    ValueError on them. Text-only SFT of the language stack still works:
    retry via ``AutoModelForImageTextToText`` and let LoRA target the
    text-stack projections.
    """
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(  # nosec: B615
            base_model,
            trust_remote_code=True,
            torch_dtype="bfloat16",
        )
    except ValueError as causal_exc:
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            raise causal_exc from None
        logger.info(
            "AutoModelForCausalLM rejected %s (%s); retrying as an "
            "image-text-to-text checkpoint and training the text stack.",
            base_model,
            causal_exc,
        )
        return AutoModelForImageTextToText.from_pretrained(  # nosec: B615
            base_model,
            trust_remote_code=True,
            torch_dtype="bfloat16",
        )


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
            "peft's leaf-name matching will adapt both.",
            ", ".join(sorted(shared)),
        )
    # The output head is deliberately excluded: adapting it inflates the
    # adapter with a vocab-sized matrix for no benefit on SFT.
    text_found.discard("lm_head")
    return sorted(text_found)


def generate_completions(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int = 90,
) -> list[str]:
    """Greedily generate one completion per prompt through ``model``.

    Used for the post-train base-vs-tuned comparison: each prompt is rendered
    through the model's chat template (with the generation prompt appended)
    and decoded back to just the completion text. Greedy decoding keeps the
    two runs comparable — sampling noise would swamp the tuning signal.

    Reasoning models (e.g. NVIDIA Nemotron 3.5 Lightning) default to thinking
    mode in their chat template, so the whole ``max_new_tokens`` budget goes
    to the reasoning preamble and the comparison is truncated garbage (hit
    for real on an H100 pod). ``enable_thinking=False`` turns that off;
    templates that don't accept the kwarg (e.g. Muse Glimmer's) raise
    TypeError and get the plain call instead.
    """
    import torch

    completions: list[str] = []
    for prompt in prompts:
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
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_length = inputs["input_ids"].shape[1]
        completions.append(
            tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
        )
    return completions


#: Keys an eval prompt-spec dict may carry. Anything else is a typo the user
#: should hear about before renting a GPU.
_EVAL_SPEC_KEYS = frozenset({"prompt", "expect", "forbid", "judge", "min_judge_score"})


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
        specs.append(dict(entry))
    return specs


def evaluate_checks(
    completion: str, expect: list[str], forbid: list[str]
) -> dict[str, Any]:
    """Case-insensitively match ``expect``/``forbid`` substrings.

    Passes when every ``expect`` substring appears in ``completion`` and no
    ``forbid`` substring does.
    """
    haystack = completion.lower()
    expect_hits = [s for s in expect if s.lower() in haystack]
    forbid_hits = [s for s in forbid if s.lower() in haystack]
    return {
        "expect_hits": expect_hits,
        "forbid_hits": forbid_hits,
        "passed": len(expect_hits) == len(expect) and not forbid_hits,
    }


def _create_domain_reward(name: str) -> Any:
    """Import hook for the optional judge — split out so tests can fake it."""
    from stateset_agents.rewards.multi_objective_reward import create_domain_reward

    return create_domain_reward(name)


def judge_completion(judge: str, prompt: str, completion: str) -> float | None:
    """Score ``completion`` with a domain reward, if one is importable here.

    The judge is a nicety on top of the substring checks, and the reward
    stack may simply not be installed on the pod — so every failure mode
    degrades to a logged warning and ``None``, never an exception.
    """
    try:
        reward = _create_domain_reward(judge)
    except Exception as exc:
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
    except Exception as exc:
        logger.warning("Judge %r failed to score (%s); skipping.", judge, exc)
        return None


def build_eval_extras(
    specs: list[dict[str, Any]], finetuned: list[str]
) -> list[dict[str, Any]]:
    """Per-row extra fields for ``eval_results.json``.

    Rows for plain prompts stay exactly as before (empty extras); rows whose
    spec carries assertions gain ``checks``, and specs naming a ``judge``
    gain ``judge_score`` (when the judge could run).
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
        extras.append(extra)
    return extras


def eval_gate_failures(
    specs: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> list[str]:
    """Human-readable assertion failures across the eval rows.

    A row fails when its substring checks did not pass, or when the spec set
    ``min_judge_score`` and a judge score exists below it. A judge that could
    not run never fails the gate — judge failures degrade, by design.
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
    return failures


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
) -> Path:
    """Run the actual SFT training on GPU.

    ``resume=True`` continues from the newest ``checkpoint-<N>`` directory
    already in ``output_dir`` when one exists; with none, it logs the fact
    and trains fresh (see ``resolve_resume_checkpoint``).

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

    Raises ``ImportError`` if transformers / peft / datasets aren't available.
    """
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    logger.info("Loading tokenizer and model: %s", base_model)
    # base_model is a caller-supplied CLI argument (public HF model repo id),
    # not attacker-controlled input; pinning a fixed revision would break
    # support for arbitrary user-chosen base models.
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True
    )  # nosec: B615
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_base_model_for_sft(base_model)

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
        # grinding). Move it now when a GPU exists.
        if gpu_available():
            model = model.to("cuda")
        base_completions = generate_completions(
            model, tokenizer, eval_prompt_texts, max_new_tokens=eval_max_new_tokens
        )

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

    def render(row: dict[str, Any]) -> dict[str, Any]:
        """Apply the model's chat template to each row."""
        text = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = Dataset.from_list(rows).map(render, remove_columns=["messages"])
    dataset = dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        ),
        remove_columns=["text"],
    )

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
        report_to=[],  # local-only by default
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    logger.info("Starting SFT…")
    if resolve_resume_checkpoint(output_dir, resume):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    logger.info("Training complete. Saving adapter to %s", output_dir)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

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
            rows = load_chat_dataset(Path(job["dataset"]))
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
                    )
                    returncode = _apply_eval_gate(job.get("eval_prompts"), output_dir)
    except Exception as exc:  # reported, never raised — see docstring
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
