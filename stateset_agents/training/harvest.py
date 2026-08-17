"""Best-of-N rejection-sampling harvest — the flywheel's forward step.

This module runs ON the training pod (``python -m
stateset_agents.training.harvest``), exactly like its sibling
:mod:`stateset_agents.training.sft`. It productizes the methodology of
``docs/FLYWHEEL_HEADROOM.md``, where it took a Muse-Glimmer-30B agent from
2/12 to 10/12 on out-of-distribution compound requests:

1. Load the base model (+ the current generation's LoRA adapter, if any).
2. Optionally measure the current generation on an eval set (greedy) —
   the "before" number the next generation must beat.
3. Sample ``best_of`` completions per harvest prompt (temperature > 0), and
   keep only the samples that pass that prompt's objective ``expect`` /
   ``forbid`` checks — the same checker the training job's eval gate uses.
4. Write the survivors as an ingest-ready chat dataset (``harvest.jsonl``)
   plus ``harvest_summary.json`` with the rates that decide whether the
   flywheel has a way forward.

Prompt specs reuse the eval-prompt shape (``{"prompt", "expect",
"forbid"}``) — one vocabulary for "what does success look like" everywhere.

Dry-run/no-GPU behaviour mirrors ``sft``: print the plan, write an empty
summary, exit 0 — so the whole path is exercisable on CPU-only CI.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from stateset_agents.training.sft import (
    evaluate_checks,
    generate_completions,
    gpu_available,
    load_base_model_for_sft,
    normalize_eval_prompts,
)

logger = logging.getLogger(__name__)

__all__ = [
    "build_harvest_rows",
    "main",
    "run_harvest_job",
    "sample_completions",
]


def sample_completions(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    best_of: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> list[str]:
    """``best_of`` sampled completions for one prompt.

    One generate() call with ``num_return_sequences`` — the prompt is
    encoded once and the batch shares its KV cache, which on the headroom
    run was the difference between a 17-minute and a multi-hour harvest.
    """
    import torch

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
            messages, tokenize=False, add_generation_prompt=True
        )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=best_of,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_length = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(row[prompt_length:], skip_special_tokens=True)
        for row in output
    ]


def build_harvest_rows(
    prompt_spec: dict[str, Any], samples: list[str]
) -> list[dict[str, Any]]:
    """Filter ``samples`` by the spec's checks; survivors become chat rows.

    Every passing sample is kept (not just the first): the headroom run's
    58-row training set came from multiple passes on the same prompts, and
    thinning it would have thinned the signal.
    """
    rows: list[dict[str, Any]] = []
    for sample in samples:
        result = evaluate_checks(
            sample,
            prompt_spec.get("expect", []),
            prompt_spec.get("forbid", []),
        )
        if result["passed"]:
            rows.append(
                {
                    "messages": [
                        {"role": "user", "content": prompt_spec["prompt"]},
                        {"role": "assistant", "content": sample.strip()},
                    ]
                }
            )
    return rows


def _load_adapter(model: Any, adapter_dir: str) -> Any:
    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_dir)  # nosec: B615


def _plan(payload: dict[str, Any]) -> None:
    print("Harvest Plan (dry run — no GPU detected)")
    for key in (
        "base_model",
        "adapter_dir",
        "best_of",
        "temperature",
        "top_p",
        "max_new_tokens",
    ):
        print(f"  {key}: {payload.get(key)}")
    print(f"  harvest prompts: {len(payload['harvest_prompts'])}")
    print(f"  eval prompts: {len(payload.get('eval_prompts') or [])}")


def run_harvest_job(payload: dict[str, Any]) -> dict[str, Any]:
    """The whole harvest, as one JSON-in/JSON-out function.

    ``payload`` mirrors the CLI arguments; the return value is the summary
    that also lands in ``harvest_summary.json``.
    """
    output_dir = Path(payload["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    harvest_specs = normalize_eval_prompts(payload["harvest_prompts"])
    for i, spec in enumerate(harvest_specs):
        if not spec.get("expect") and not spec.get("forbid"):
            raise ValueError(
                f"harvest prompt {i} has no expect/forbid checks — without "
                "them every sample passes and the harvest is noise, not "
                "signal"
            )
    eval_specs = normalize_eval_prompts(payload.get("eval_prompts") or [])

    summary: dict[str, Any] = {
        "base_model": payload["base_model"],
        "adapter_dir": payload.get("adapter_dir"),
        "best_of": payload["best_of"],
        "temperature": payload["temperature"],
        "prompts": len(harvest_specs),
        "samples": 0,
        "kept": 0,
        "prompts_with_a_pass": 0,
        "eval": None,
        "dry_run": False,
    }

    if payload.get("dry_run") or not gpu_available():
        _plan(payload)
        summary["dry_run"] = True
        (output_dir / "harvest_summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(payload["base_model"])  # nosec: B615
    model = load_base_model_for_sft(payload["base_model"])
    if payload.get("adapter_dir"):
        model = _load_adapter(model, payload["adapter_dir"])
        logger.info("loaded adapter %s", payload["adapter_dir"])
    model.eval()

    # -- "before" eval of the current generation, greedy ------------------
    if eval_specs:
        completions = generate_completions(
            model,
            tokenizer,
            [spec["prompt"] for spec in eval_specs],
            max_new_tokens=payload.get("eval_max_new_tokens", 300),
        )
        results = []
        for spec, completion in zip(eval_specs, completions):
            checked = evaluate_checks(
                completion, spec.get("expect", []), spec.get("forbid", [])
            )
            results.append(
                {"prompt": spec["prompt"], "completion": completion, **checked}
            )
        summary["eval"] = {
            "passed": sum(1 for r in results if r["passed"]),
            "total": len(results),
            "results": results,
        }
        logger.info(
            "current generation: %d/%d eval prompts passed",
            summary["eval"]["passed"],
            summary["eval"]["total"],
        )

    # -- best-of-N harvest -------------------------------------------------
    rows: list[dict[str, Any]] = []
    for spec in harvest_specs:
        samples = sample_completions(
            model,
            tokenizer,
            spec["prompt"],
            best_of=payload["best_of"],
            temperature=payload["temperature"],
            top_p=payload["top_p"],
            max_new_tokens=payload["max_new_tokens"],
        )
        summary["samples"] += len(samples)
        kept = build_harvest_rows(spec, samples)
        if kept:
            summary["prompts_with_a_pass"] += 1
        rows.extend(kept)

    summary["kept"] = len(rows)
    with (output_dir / "harvest.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    (output_dir / "harvest_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        "harvest: kept %d/%d samples across %d/%d prompts",
        summary["kept"],
        summary["samples"],
        summary["prompts_with_a_pass"],
        summary["prompts"],
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Best-of-N rejection-sampling harvest for the flywheel."
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument(
        "--adapter", default=None, help="current generation's LoRA adapter dir"
    )
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument(
        "--prompts-json",
        help="JSON list of {prompt, expect, forbid} harvest specs",
    )
    prompts.add_argument(
        "--prompts-file",
        type=Path,
        help="path to a JSON file with the same list (what executors upload)",
    )
    parser.add_argument(
        "--eval-prompts-json",
        default=None,
        help="JSON list of eval specs; measures the CURRENT generation first",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/harvest"))
    parser.add_argument("--best-of", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--eval-max-new-tokens", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args(argv)
    if args.prompts_file is not None:
        harvest_prompts = json.loads(args.prompts_file.read_text())
    else:
        harvest_prompts = json.loads(args.prompts_json)
    payload = {
        "base_model": args.base_model,
        "adapter_dir": args.adapter,
        "harvest_prompts": harvest_prompts,
        "eval_prompts": (
            json.loads(args.eval_prompts_json) if args.eval_prompts_json else None
        ),
        "output_dir": str(args.output_dir),
        "best_of": args.best_of,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "eval_max_new_tokens": args.eval_max_new_tokens,
        "dry_run": args.dry_run,
    }
    run_harvest_job(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised via subprocess tests
    raise SystemExit(main())
