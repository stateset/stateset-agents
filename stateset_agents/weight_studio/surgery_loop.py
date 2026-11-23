"""Autonomous Transcript-to-Surgery Self-Healing Loop for StateSet Agents.

Bridges production agent failure logs directly into 3D Weight Studio's surgical
weight editing, Contrastive Activation Steering (CAA/RepE), Multi-Edit A/B
Arena, 95% CI Batch Eval Gate, Surgical Auto-Repair, and verified artifact
export (PEFT Diff-as-LoRA, .safetensors, GGUF v3, HF Model Card, and standalone
HTML PR reports).

No eager ``torch`` import: runs cleanly on CPU in milliseconds or with real
PyTorch hooks when available.
"""

from __future__ import annotations

import difflib
import json
import re
import time
from pathlib import Path
from typing import Any

from .editor import WeightEditEngine
from .inference_engine import InferenceStudioEngine
from .inspector import WeightInspector
from .report_exporter import build_standalone_html_report

_PLACEHOLDER_RE = re.compile(
    r"(\[[A-Z0-9_ ]{2,24}\]|\{\{[a-zA-Z0-9_ .-]{1,24}\}\}|<[A-Z_]{2,20}>|\bTODO\b|\bTBD\b)"
)

_FAILURE_CONTRAST_TEMPLATES: dict[str, tuple[str, str, str]] = {
    "unfilled_placeholder": (
        "Concrete Grounded Values",
        "Always substitute exact verified order IDs, amounts, and customer details with zero template placeholders.",
        "Return generic draft templates containing brackets like [ORDER_ID] and {{customer_name}}.",
    ),
    "repetition_loop": (
        "Concise Non-Repetitive Resolution",
        "Deliver one clear, non-repetitive resolution summarizing verified eligibility and exact next steps.",
        "Repeat the same apology paragraph and refund disclaimer verbatim over and over.",
    ),
    "policy_gap": (
        "Policy-Verified Commerce Resolution",
        "Verify 30-day eligibility, order status, and billing records before approving refunds or escalations.",
        "Approve arbitrary refunds or ignore policy boundaries without verifying eligibility.",
    ),
    "tool_schema_error": (
        "Strict JSON Tool Schema Adherence",
        "Emit strictly valid JSON tool calls with typed arguments matching the ledger schema.",
        "Output malformed prose or broken brackets instead of valid JSON tool arguments.",
    ),
    "terse_or_unresponsive": (
        "Empathetic Actionable Resolution",
        "Acknowledge the customer's exact issue empathetically and provide concrete, verifiable resolution steps.",
        "Give a dismissive one-line reply that ignores the user's specific details.",
    ),
}


def _extract_turn_pair(record: dict[str, Any]) -> tuple[str, str]:
    """Extract (user_prompt, assistant_response) from common transcript JSONL formats."""
    prompt = str(
        record.get("prompt")
        or record.get("instruction")
        or record.get("input")
        or record.get("user")
        or ""
    ).strip()
    response = str(
        record.get("response")
        or record.get("completion")
        or record.get("output")
        or record.get("assistant")
        or ""
    ).strip()

    messages = record.get("messages")
    if isinstance(messages, list):
        user_parts: list[str] = []
        asst_parts: list[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "").lower()
            content = str(msg.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                user_parts.append(content)
            elif role == "assistant":
                asst_parts.append(content)
        if not prompt and user_parts:
            prompt = user_parts[-1]
        if not response and asst_parts:
            response = asst_parts[-1]
    return prompt, response


def _detect_repetition(text: str) -> bool:
    """Detect repeated sentences or clauses typical of RL reward-hacking / decoding loops."""
    chunks = [
        c.strip().lower() for c in re.split(r"[.!?\n]+", text) if len(c.strip()) > 18
    ]
    if len(chunks) < 2:
        return False
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            if difflib.SequenceMatcher(None, chunks[i], chunks[j]).ratio() >= 0.72:
                return True
    return False


def _score_transcript_turn(
    prompt: str, response: str, *, reward_fn: str = "customer_support"
) -> tuple[float, list[str]]:
    """Deterministically grade a single (prompt, response) turn and tag failure modes."""
    flags: list[str] = []
    score = 0.92
    low_p = prompt.lower()
    low_r = response.lower()

    if not response or len(response.split()) < 6:
        flags.append("terse_or_unresponsive")
        score -= 0.42

    if _PLACEHOLDER_RE.search(response):
        flags.append("unfilled_placeholder")
        score -= 0.35

    if _detect_repetition(response):
        flags.append("repetition_loop")
        score -= 0.38

    if reward_fn in {"customer_support", "composite"} and any(
        kw in low_p
        for kw in ("refund", "damaged", "billed", "order", "cancel", "return")
    ):
        if not any(
            kw in low_r
            for kw in (
                "eligib",
                "verify",
                "policy",
                "order",
                "refund",
                "escalat",
                "confirm",
                "30-day",
            )
        ):
            flags.append("policy_gap")
            score -= 0.28

    if reward_fn == "tool_calling" or any(
        kw in low_p for kw in ("json", "sku", "tool", "inventory", "schema")
    ):
        if "{" in response and "}" in response:
            # Verify braces are at least balanced
            if response.count("{") != response.count("}"):
                flags.append("tool_schema_error")
                score -= 0.34
        elif reward_fn == "tool_calling":
            flags.append("tool_schema_error")
            score -= 0.30

    return round(max(0.0, min(1.0, score)), 4), flags


class AutonomousSurgeryLoop:
    """End-to-end transcript diagnosis -> contrastive mining -> CAA surgical healing -> eval-gated export."""

    def __init__(
        self,
        inspector: WeightInspector,
        editor: WeightEditEngine,
        inference: InferenceStudioEngine,
    ) -> None:
        self.inspector = inspector
        self.editor = editor
        self.inference = inference

    def _load_records(
        self,
        *,
        transcripts_path: str | None = None,
        transcripts_jsonl: str | None = None,
        records: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        loaded: list[dict[str, Any]] = []
        if records:
            loaded.extend(r for r in records if isinstance(r, dict))

        if transcripts_jsonl and transcripts_jsonl.strip():
            for line in transcripts_jsonl.strip().splitlines():
                text = line.strip()
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                    if isinstance(obj, dict):
                        loaded.append(obj)
                except json.JSONDecodeError:
                    loaded.append({"prompt": text, "response": ""})

        if transcripts_path and transcripts_path.strip():
            raw_p = Path(transcripts_path.strip())
            resolved = (
                raw_p.resolve()
                if raw_p.is_absolute()
                else (self.inspector.workspace_root / raw_p).resolve()
            )
            if not resolved.is_file():
                raise FileNotFoundError(
                    f"Transcripts file not found: {transcripts_path}"
                )
            with resolved.open("r", encoding="utf-8") as fh:
                for line in fh:
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        obj = json.loads(text)
                        if isinstance(obj, dict):
                            loaded.append(obj)
                    except json.JSONDecodeError:
                        continue

        return loaded

    def diagnose_transcripts(
        self,
        *,
        transcripts_path: str | None = None,
        transcripts_jsonl: str | None = None,
        records: list[dict[str, Any]] | None = None,
        reward_fn: str = "customer_support",
        pass_threshold: float = 0.72,
    ) -> dict[str, Any]:
        """Grade transcripts and identify dominant failure modes + contrastive mining candidates."""
        items = self._load_records(
            transcripts_path=transcripts_path,
            transcripts_jsonl=transcripts_jsonl,
            records=records,
        )
        if not items:
            # Built-in fallback diagnostic batch so one-click UI runs stay informative
            items = [
                {
                    "prompt": "Customer says: My order #77701 arrived damaged after 9 days and I was billed twice.",
                    "response": "Sorry [CUSTOMER_NAME]. Sorry [CUSTOMER_NAME]. Contact support later.",
                },
                {
                    "prompt": "Customer says: My order #77702 arrived cracked within the 30-day window.",
                    "response": "I verified order #77702 is within the 30-day damaged-item refund policy and initiated a full refund.",
                },
            ]

        graded: list[dict[str, Any]] = []
        failure_counts: dict[str, int] = {}
        for idx, rec in enumerate(items[:200]):
            prompt, response = _extract_turn_pair(rec)
            if not prompt:
                continue
            score, flags = _score_transcript_turn(prompt, response, reward_fn=reward_fn)
            for fl in flags:
                failure_counts[fl] = failure_counts.get(fl, 0) + 1
            graded.append(
                {
                    "index": idx,
                    "prompt": prompt[:500],
                    "response": response[:500],
                    "score": score,
                    "passed": score >= pass_threshold and not flags,
                    "failure_modes": flags,
                }
            )

        if not graded:
            raise ValueError("No valid user prompts found in supplied transcripts")

        failing = sorted(
            [g for g in graded if not g["passed"]], key=lambda r: float(r["score"])
        )
        healthy = sorted(
            [g for g in graded if g["passed"]],
            key=lambda r: float(r["score"]),
            reverse=True,
        )
        dominant_failure = (
            max(failure_counts.items(), key=lambda kv: kv[1])[0]
            if failure_counts
            else "policy_gap"
        )
        concept, fallback_pos, fallback_neg = _FAILURE_CONTRAST_TEMPLATES.get(
            dominant_failure, _FAILURE_CONTRAST_TEMPLATES["policy_gap"]
        )

        pos_prompt = (
            f"{healthy[0]['prompt']} -> {healthy[0]['response']}"
            if healthy and healthy[0]["response"]
            else fallback_pos
        )
        neg_prompt = (
            f"{failing[0]['prompt']} -> {failing[0]['response']}"
            if failing and failing[0]["response"]
            else fallback_neg
        )

        mean_score = sum(float(g["score"]) for g in graded) / len(graded)
        return {
            "total_transcripts": len(graded),
            "failing_count": len(failing),
            "healthy_count": len(healthy),
            "pass_rate": round(len(healthy) / len(graded), 4),
            "mean_score": round(mean_score, 4),
            "dominant_failure_mode": dominant_failure,
            "failure_counts": failure_counts,
            "mined_contrast": {
                "concept": concept,
                "positive_prompt": pos_prompt[:600],
                "negative_prompt": neg_prompt[:600],
            },
            "eval_prompts": [g["prompt"] for g in (failing[:4] or graded[:4])],
            "samples": graded[:12],
        }

    def run_healing_loop(
        self,
        model_id: str,
        *,
        checkpoint_id: str | None = None,
        transcripts_path: str | None = None,
        transcripts_jsonl: str | None = None,
        records: list[dict[str, Any]] | None = None,
        reward_fn: str = "customer_support",
        export_format: str = "none",
        output_name: str = "",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        gguf_quant: str = "Q8_0",
        require_go: bool = True,
        auto_repair: bool = True,
    ) -> dict[str, Any]:
        """Run the full autonomous Transcript -> Contrastive CAA -> Arena -> Bake -> Eval Gate -> Export loop."""
        t0 = time.perf_counter()
        reward_key = (
            reward_fn
            if reward_fn in {"composite", "customer_support", "tool_calling", "gsm8k"}
            else "customer_support"
        )
        fmt = export_format.lower().strip()
        if fmt not in {"none", "lora", "safetensors", "gguf", "hf_card"}:
            raise ValueError(f"Unsupported export_format: {export_format}")

        # Stage 1 & 2: Diagnose transcripts & mine contrastive positive/negative pair
        diagnosis = self.diagnose_transcripts(
            transcripts_path=transcripts_path,
            transcripts_jsonl=transcripts_jsonl,
            records=records,
            reward_fn=reward_key,
        )
        contrast = diagnosis["mined_contrast"]

        # Stage 3: Extract CAA steering vector & sweep candidate strengths in Multi-Edit A/B Arena
        vec = self.inference.extract_steering_vector(
            model_id,
            contrast["positive_prompt"],
            contrast["negative_prompt"],
            checkpoint_id=checkpoint_id,
            name=f"AutoHeal: {contrast['concept']}",
        )
        target_layer = int(vec["recommended_layer"])
        layer_vecs = vec.get("layer_vectors", {})
        direction = layer_vecs.get(target_layer) or vec["direction"]

        probe_prompt = diagnosis["eval_prompts"][0]
        arena_candidates = [
            {
                "name": f"CAA Heal α=+0.55 (L{target_layer:02d})",
                "op": "steer_bake",
                "target": f"layer:{target_layer}",
                "params": {
                    "concept": contrast["concept"],
                    "strength": 0.55,
                    "direction": direction,
                },
            },
            {
                "name": f"CAA Heal α=+0.95 (L{target_layer:02d})",
                "op": "steer_bake",
                "target": f"layer:{target_layer}",
                "params": {
                    "concept": contrast["concept"],
                    "strength": 0.95,
                    "direction": direction,
                },
            },
            {
                "name": f"CAA Heal α=+1.30 (L{target_layer:02d})",
                "op": "steer_bake",
                "target": f"layer:{target_layer}",
                "params": {
                    "concept": contrast["concept"],
                    "strength": 1.30,
                    "direction": direction,
                },
            },
        ]
        arena = self.inference.compare_arena_candidates(
            model_id,
            probe_prompt,
            arena_candidates,
            checkpoint_id=checkpoint_id,
            reward_fn=reward_key,
        )
        winning_cand: dict[str, Any] = next(
            (
                c
                for c in arena["candidates"]
                if isinstance(c, dict) and c.get("kind") == "candidate"
            ),
            dict(arena_candidates[1]),
        )
        raw_params = winning_cand.get("params")
        chosen_strength = float(
            raw_params.get("strength", 0.95) if isinstance(raw_params, dict) else 0.95
        )

        # Stage 4: Permanently bake the winning CAA direction into the target layer
        bake_res = self.editor.apply_edit(
            model_id,
            "steer_bake",
            f"layer:{target_layer}",
            {
                "concept": contrast["concept"],
                "strength": chosen_strength,
                "direction": direction,
            },
            checkpoint_id=checkpoint_id,
            label=(
                f"Auto-Heal '{contrast['concept'][:20]}' "
                f"(L{target_layer:02d}, α={chosen_strength:+.2f})"
            ),
        )

        # Stage 5: 95% CI Pre-Export Eval Gate on mined failure prompts + StateSet benchmark
        custom_jsonl = "\n".join(
            json.dumps({"prompt": p, "category": "Mined Failure"})
            for p in diagnosis["eval_prompts"]
        )
        eval_report = self.inference.run_batch_eval(
            model_id,
            checkpoint_id=checkpoint_id,
            custom_jsonl=custom_jsonl,
            reward_fn=reward_key,
            max_cases=6,
        )

        repair_report: dict[str, Any] | None = None
        if auto_repair and eval_report["verdict"] != "GO":
            repair_report = self.editor.auto_repair_edits(
                model_id,
                self.inference,
                checkpoint_id=checkpoint_id,
                reward_fn=reward_key,
                max_steps=3,
            )
            eval_report = repair_report["eval_report"]

        # Stage 6: Optional auditable artifact export (LoRA / .safetensors / GGUF / HF Card + HTML PR report)
        exported_manifest: dict[str, Any] | None = None
        html_report_path: str | None = None
        summary_path: str | None = None
        export_blocked_reason: str | None = None

        if fmt != "none":
            if require_go and eval_report["verdict"] != "GO":
                export_blocked_reason = f"Eval gate verdict is {eval_report['verdict']}; pass require_go=False to override."
            else:
                safe_out = (
                    output_name.strip()
                    or f"healed-{diagnosis['dominant_failure_mode']}-{int(time.time())}"
                )
                if fmt == "lora":
                    exported_manifest = self.editor.export_diff_as_lora(
                        model_id,
                        safe_out,
                        rank=lora_rank,
                        alpha=lora_alpha,
                        checkpoint_id=checkpoint_id,
                    )
                    bundle_dir = (
                        self.inspector.workspace_root / exported_manifest["export_dir"]
                    )
                elif fmt == "gguf":
                    exported_manifest = self.editor.export_gguf_package(
                        model_id,
                        safe_out,
                        quant_type=gguf_quant,
                        checkpoint_id=checkpoint_id,
                    )
                    bundle_dir = (
                        self.inspector.workspace_root / exported_manifest["gguf_file"]
                    ).parent
                elif fmt == "hf_card":
                    exported_manifest = self.editor.export_hf_model_card(
                        model_id,
                        safe_out,
                        eval_report=eval_report,
                        checkpoint_id=checkpoint_id,
                    )
                    bundle_dir = (
                        self.inspector.workspace_root / exported_manifest["export_dir"]
                    )
                else:
                    exported_manifest = self.editor.export_verified_checkpoint(
                        model_id,
                        safe_out,
                        checkpoint_id=checkpoint_id,
                        run_merge_probe=True,
                    )
                    bundle_dir = (
                        self.inspector.workspace_root / exported_manifest["export_file"]
                    ).parent

                # Write standalone interactive HTML PR report + heal_summary.json right inside the bundle
                overrides = self.editor.get_overrides(model_id, checkpoint_id)
                struct = self.inspector.get_model_structure(
                    model_id,
                    checkpoint_id,
                    overrides=overrides if overrides else None,
                )
                recipe = self.editor.export_recipe(model_id, checkpoint_id)
                html_doc = build_standalone_html_report(
                    model_summary=struct["model"],
                    structure_summary=struct["summary"],
                    layers=struct["layers"],
                    edit_history=self.editor.get_history(model_id, checkpoint_id),
                    recipe=recipe,
                    eval_report=eval_report,
                )
                html_file = bundle_dir / "healing_report.html"
                html_file.write_text(html_doc, encoding="utf-8")
                html_report_path = str(
                    html_file.relative_to(self.inspector.workspace_root)
                )

                summary_file = bundle_dir / "heal_summary.json"
                summary_path = str(
                    summary_file.relative_to(self.inspector.workspace_root)
                )
                summary_file.write_text(
                    json.dumps(
                        {
                            "model_id": model_id,
                            "reward_fn": reward_key,
                            "diagnosis": {
                                k: v for k, v in diagnosis.items() if k != "samples"
                            },
                            "chosen_layer": target_layer,
                            "chosen_strength": chosen_strength,
                            "eval_verdict": eval_report["verdict"],
                            "eval_deltas": eval_report["deltas"],
                            "paired_stats": eval_report.get("paired_stats"),
                            "export_format": fmt,
                            "html_report": html_report_path,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )

        return {
            "model_id": model_id,
            "checkpoint_id": checkpoint_id or "default",
            "reward_fn": reward_key,
            "diagnosis": diagnosis,
            "steering_vector": {
                "vector_id": vec["vector_id"],
                "name": vec["name"],
                "target_layer": target_layer,
                "chosen_strength": chosen_strength,
                "peak_separation": vec["peak_separation"],
            },
            "arena_winner": arena["winner"],
            "applied_edit": bake_res["edit"],
            "eval_gate": eval_report,
            "auto_repair": repair_report,
            "exported": exported_manifest is not None,
            "export_format": fmt,
            "export_manifest": exported_manifest,
            "export_blocked_reason": export_blocked_reason,
            "html_report_path": html_report_path,
            "summary_path": summary_path,
            "history": self.editor.get_history(model_id, checkpoint_id),
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }


def run_autonomous_heal(
    model_id: str,
    *,
    workspace_root: Path | None = None,
    checkpoint_id: str | None = None,
    transcripts_path: str | None = None,
    transcripts_jsonl: str | None = None,
    records: list[dict[str, Any]] | None = None,
    reward_fn: str = "customer_support",
    export_format: str = "lora",
    output_name: str = "",
    lora_rank: int = 16,
    lora_alpha: int = 32,
    gguf_quant: str = "Q8_0",
    require_go: bool = True,
) -> dict[str, Any]:
    """Convenience one-call entrypoint for the Autonomous Transcript-to-Surgery Loop."""
    inspector = WeightInspector(workspace_root=workspace_root)
    editor = WeightEditEngine(inspector)
    inference = InferenceStudioEngine(inspector, editor)
    loop = AutonomousSurgeryLoop(inspector, editor, inference)
    return loop.run_healing_loop(
        model_id,
        checkpoint_id=checkpoint_id,
        transcripts_path=transcripts_path,
        transcripts_jsonl=transcripts_jsonl,
        records=records,
        reward_fn=reward_fn,
        export_format=export_format,
        output_name=output_name,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        gguf_quant=gguf_quant,
        require_go=require_go,
    )
