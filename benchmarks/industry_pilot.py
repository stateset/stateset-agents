#!/usr/bin/env python3
"""GPU smoke benchmark for pinned Qwen3.5 industry SFT and paired evaluation.

Uses explicitly synthetic lookup conversations. Results certify neither domain
quality nor business outcomes. Provisioning and spend limits belong to the caller.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import re
import time
from pathlib import Path
from typing import Any


def file_hash(path: Path) -> str:
    """Fingerprint artifacts without reading large weights into RAM at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Retain one JSON result, refusing accidental overwrite."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def synthetic_lookup_rows(industry: str, count: int = 64) -> list[dict[str, Any]]:
    """Create distinct source groups to exercise tooling, not domain expertise."""
    from stateset_agents.training.industry import get_industry_recipe

    recipe = get_industry_recipe(industry)
    rows = []
    for index in range(count):
        record_id = f"PILOT-{index:04d}"
        status = (
            "pending review",
            "review complete",
            "awaiting information",
            "resolved",
        )[index % 4]
        rows.append(
            {
                "synthetic": True,
                "group_id": record_id,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": recipe.tool_name,
                            "description": f"Read the authorized {recipe.record_label} status.",
                            "parameters": {
                                "type": "object",
                                "properties": {"record_id": {"type": "string"}},
                                "required": ["record_id"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
                "messages": [
                    {
                        "role": "system",
                        "content": recipe.system_prompt
                        + " When reporting a status, reply exactly: '<record_id>: <status>'.",
                    },
                    {
                        "role": "user",
                        "content": f"Check my {recipe.record_label} {record_id}.",
                    },
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "lookup_1",
                                "type": "function",
                                "function": {
                                    "name": recipe.tool_name,
                                    "arguments": {"record_id": record_id},
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "lookup_1",
                        "content": json.dumps(
                            {"record_id": record_id, "status": status}
                        ),
                    },
                    {"role": "assistant", "content": f"{record_id}: {status}"},
                ],
            }
        )
    return rows


def validate_manifest(manifest: Any) -> None:
    """Restrict this harness to the architectures its decoder understands."""
    from stateset_agents.training.industry import get_industry_recipe

    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Expected a schema_version 1 pilot manifest")
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("The pilot needs a nonempty models list")
    seen = set()
    for spec in models:
        if not isinstance(spec, dict) or spec.get("preset") not in (
            "qwen3.5-2b",
            "qwen3.5-4b",
        ):
            raise ValueError("This pilot supports qwen3.5-2b and qwen3.5-4b")
        if spec["preset"] in seen:
            raise ValueError("Duplicate pilot model")
        seen.add(spec["preset"])
        if not isinstance(spec.get("revision"), str) or not re.fullmatch(
            "[0-9a-f]{40}", spec["revision"]
        ):
            raise ValueError("An immutable model revision is required")
        get_industry_recipe(spec.get("industry"))


def parse_qwen_response(text: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Parse Qwen3.5's native XML calls without discarding unparsed content."""
    text = text.strip()
    if "<tool_call>" not in text:
        return {"role": "assistant", "content": text}
    pattern = re.compile(
        r"<tool_call>\s*<function=([^>]+)>\s*(.*?)</function>\s*</tool_call>", re.DOTALL
    )
    parameters = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
    schemas = {
        tool["function"]["name"]: tool["function"]
        .get("parameters", {})
        .get("properties", {})
        for tool in tools
    }
    calls = []
    for index, match in enumerate(pattern.finditer(text)):
        name, body = match.groups()
        if name not in schemas or parameters.sub("", body).strip():
            raise ValueError("Unknown tool or malformed parameter block")
        arguments = {}
        for parameter in parameters.finditer(body):
            key, value = parameter.groups()
            if key in arguments or key not in schemas[name]:
                raise ValueError("Repeated or unknown tool parameter")
            arguments[key] = (
                value
                if schemas[name][key].get("type") == "string"
                else json.loads(value)
            )
        calls.append(
            {
                "id": f"pilot-call-{index}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    content = pattern.sub("", text).strip()
    if not calls or any(
        marker in content
        for marker in ("<tool_call", "</tool_call", "<function=", "<parameter=")
    ):
        raise ValueError("Malformed or incomplete tool call")
    return {"role": "assistant", "content": content or None, "tool_calls": calls}


def verify_model_revision(
    requested: str, source_config: Any, loaded_config: Any
) -> None:
    """Check the source pin, allowing a text subconfig to omit parent metadata.

    Transformers can extract a composite checkpoint's text config when loading
    AutoModelForCausalLM. That subconfig does not inherit the parent commit hash.
    The loader must still receive the immutable revision explicitly.
    """
    if getattr(source_config, "_commit_hash", None) != requested:
        raise ValueError("Source configuration differs from the requested pin")
    if getattr(loaded_config, "_commit_hash", None) not in (None, requested):
        raise ValueError("Loaded model revision differs from the requested pin")


def run_model(spec: dict[str, Any], output: Path) -> dict[str, Any]:
    """Train one full checkpoint and verify its saved adapter on CUDA."""
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoTokenizer, set_seed

    from stateset_agents.core.transformers_compat import generation_compat_kwargs
    from stateset_agents.evaluation.industry import (
        collect_industry_predictions,
        evaluate_industry_project,
        export_industry_evaluation,
    )
    from stateset_agents.training.industry import (
        prepare_industry_training,
        train_industry_project,
    )
    from stateset_agents.training.sft import _render_sft_row, load_base_model_for_sft

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a real CUDA device")
    started = time.perf_counter()
    output.mkdir()
    source = output / "synthetic.jsonl"
    rows = synthetic_lookup_rows(spec["industry"])
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    project = output / "prepared"
    manifest = prepare_industry_training(
        spec["industry"],
        source,
        project,
        model=spec["preset"],
        model_revision=spec["revision"],
        validation_fraction=0.25,
        seed=42,
    )
    model_id = manifest["base_model"]
    settings = {
        "do_sample": False,
        "max_new_tokens": 96,
        "enable_thinking": False,
        "seed": 42,
    }
    identity = {"base_model": model_id, "revision": spec["revision"]}
    native_outputs: list[str | None] = []
    source_config = AutoConfig.from_pretrained(model_id, revision=spec["revision"])
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=spec["revision"])
    lengths = [
        len(
            tokenizer(
                _render_sft_row(tokenizer, row)["text"], add_special_tokens=False
            )["input_ids"]
        )
        for row in rows
    ]
    if max(lengths) > 1024:
        raise ValueError("Pilot examples would be truncated during training")

    def load_base() -> Any:
        loaded = load_base_model_for_sft(model_id, revision=spec["revision"])
        verify_model_revision(spec["revision"], source_config, loaded.config)
        return loaded.to("cuda").eval()

    def callback(active_model: Any) -> Any:
        def predict(
            messages: list[dict[str, Any]], tools: list[dict[str, Any]]
        ) -> dict[str, Any]:
            native_outputs.append(None)
            prompt = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            ).to("cuda")
            with torch.inference_mode():
                tokens = active_model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=settings["max_new_tokens"],
                    pad_token_id=tokenizer.eos_token_id,
                    **generation_compat_kwargs(active_model),
                )
            completion = tokens[0, inputs["input_ids"].shape[1] :]
            text = tokenizer.decode(completion, skip_special_tokens=True)
            native_outputs[-1] = text
            result: dict[str, Any] = {
                "response": {"role": "assistant", "content": text},
                "generated_tokens": len(completion),
                "cost_usd": None,
            }
            try:
                result["response"] = parse_qwen_response(text, tools)
            except ValueError as exc:
                result["error"] = str(exc)
            if (
                len(completion) >= settings["max_new_tokens"]
                and completion[-1].item() != tokenizer.eos_token_id
            ):
                result["error"] = "generation reached the token limit"
            return result

        return predict

    set_seed(42)
    model = load_base()
    baseline = collect_industry_predictions(
        project, callback(model), variant="baseline", model=identity, settings=settings
    )
    for row, raw in zip(baseline["predictions"], native_outputs, strict=True):
        row["raw_response"] = raw
    write_json(output / "baseline.json", baseline)
    native_outputs.clear()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    train_started = time.perf_counter()
    plan = train_industry_project(project, dry_run=False, num_epochs=1, max_length=1024)
    train_seconds = time.perf_counter() - train_started
    peak_training_gib = torch.cuda.max_memory_allocated() / 2**30
    gc.collect()
    torch.cuda.empty_cache()
    adapter = project / "adapter"
    weights_path = adapter / "adapter_model.safetensors"
    weights = load_file(str(weights_path))
    updated = any(
        torch.count_nonzero(value).item() > 0
        for name, value in weights.items()
        if "lora_B" in name
    )
    if not updated:
        raise RuntimeError("Saved adapter has no nonzero LoRA B weights")
    del weights
    states = list(adapter.glob("checkpoint-*/trainer_state.json"))
    optimizer_steps = max(
        (json.loads(path.read_text())["global_step"] for path in states), default=0
    )
    if optimizer_steps < 1:
        raise RuntimeError("No committed training steps were retained")
    model = PeftModel.from_pretrained(load_base(), str(adapter)).eval()
    candidate_identity = {
        **identity,
        "adapter": {"id": str(adapter), "sha256": file_hash(weights_path)},
    }
    candidate = collect_industry_predictions(
        project,
        callback(model),
        variant="candidate",
        model=candidate_identity,
        settings=settings,
    )
    for row, raw in zip(candidate["predictions"], native_outputs, strict=True):
        row["raw_response"] = raw
    write_json(output / "candidate.json", candidate)
    report = evaluate_industry_project(project, baseline, candidate)
    write_json(output / "evaluation.json", report)
    # Prove the reloaded adapter changes the active model on a held-out prefix.
    probe = export_industry_evaluation(project)["cases"][0]
    prompt = tokenizer.apply_chat_template(
        probe["messages"],
        tools=probe["tools"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.inference_mode():
        tuned_logits = model(**inputs, use_cache=False).logits[:, -1].float().cpu()
        with model.disable_adapter():
            base_logits = model(**inputs, use_cache=False).logits[:, -1].float().cpu()
    logit_delta = (tuned_logits - base_logits).abs().max().item()
    if (
        not torch.isfinite(tuned_logits).all()
        or not torch.isfinite(base_logits).all()
        or logit_delta <= 0
    ):
        raise RuntimeError("Reloaded adapter effect was not established")
    result = {
        "model": model_id,
        "revision": spec["revision"],
        "source_config_revision": source_config._commit_hash,
        "loaded_model_config_revision": getattr(model.config, "_commit_hash", None),
        "industry": spec["industry"],
        "status": "training_and_reload_verified",
        "quality_certified": False,
        "synthetic": True,
        "seed": 42,
        "optimizer_steps": optimizer_steps,
        "training_seconds_including_load_and_save": train_seconds,
        "peak_training_allocated_gib": peak_training_gib,
        "total_model_seconds": time.perf_counter() - started,
        "max_training_tokens": max(lengths),
        "reloaded_adapter_max_abs_logit_delta": logit_delta,
        "adapter_sha256": file_hash(weights_path),
        "training_plan": plan,
        "baseline": report["baseline"],
        "candidate": report["candidate"],
        "quality_gate": report["gate"],
    }
    write_json(output / "result.json", result)
    del model, tuned_logits, base_logits, inputs
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    """Run manifest checkpoints sequentially and retain failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    import torch

    report = {
        "schema_version": 1,
        "kind": "stateset-industry-gpu-pilot",
        "manifest_sha256": file_hash(args.manifest),
        "harness_sha256": file_hash(Path(__file__)),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "stateset-agents",
                "torch",
                "transformers",
                "peft",
                "datasets",
                "accelerate",
            )
        },
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "results": [],
        "quality_certified": False,
    }
    failed = False
    for spec in manifest["models"]:
        print(f"Starting {spec['preset']} on {spec['industry']}", flush=True)
        try:
            result = run_model(spec, args.output / spec["preset"])
        except Exception as exc:
            failed = True
            result = {
                "model": spec["preset"],
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            import traceback

            traceback.print_exc()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        report["results"].append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    report["status"] = "failed" if failed else "completed"
    write_json(args.output / "pilot.json", report)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
