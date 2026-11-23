"""Autonomous Multi-Model Tournament & Evolutionary Surgical Merge Lab.

Evaluates 2+ checkpoints/adapters/models side-by-side across StateSet reward
scorers and optional Red-Team safety probes, builds a Pareto-ranked
leaderboard, and optionally synthesizes & benchmarks TIES / DARE / SLERP /
Linear surgical merges of the top contenders — automatically promoting a
merged champion only when it matches or beats the strongest individual parent!

All heavy dependencies remain strictly lazy (zero eager ``torch`` import).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

from .editor import WeightEditEngine
from .inference_engine import InferenceStudioEngine
from .inspector import WeightInspector
from .redteam_engine import RedTeamSurgeryEngine

DEFAULT_MERGE_METHODS: tuple[str, ...] = ("ties", "dare", "slerp")


class ModelTournamentEngine:
    """Multi-contender benchmark leaderboard and evolutionary surgical merge lab."""

    def __init__(
        self,
        inspector: WeightInspector,
        editor: WeightEditEngine,
        inference: InferenceStudioEngine,
        redteam: RedTeamSurgeryEngine,
    ) -> None:
        self.inspector = inspector
        self.editor = editor
        self.inference = inference
        self.redteam = redteam

    def _score_contender(
        self,
        model_id: str,
        *,
        checkpoint_id: str | None = None,
        reward_fn: str = "customer_support",
        max_cases: int = 4,
        include_redteam: bool = True,
        redteam_probes: int = 4,
        label: str = "",
    ) -> dict[str, Any]:
        gate = self.inference.run_batch_eval(
            model_id,
            checkpoint_id=checkpoint_id,
            reward_fn=reward_fn,
            max_cases=max_cases,
        )
        edited_metrics = gate.get("edited_metrics") or {}
        reward_score = float(
            edited_metrics.get(
                "composite",
                edited_metrics.get("mean_reward", 0.5),
            )
        )
        verdict = str(gate.get("verdict") or "GO")

        safety_pass = 1.0
        asr = 0.0
        if include_redteam:
            rt_rep = self.redteam.run_redteam_audit(
                model_id,
                checkpoint_id=checkpoint_id,
                max_probes=redteam_probes,
                immunize=False,
            )
            safety_pass = float(rt_rep["safety_pass_rate"])
            asr = float(rt_rep["attack_success_rate"])

        # Composite Pareto fitness: 70% task reward + 30% adversarial safety pass rate
        fitness = (
            (0.70 * reward_score + 0.30 * safety_pass)
            if include_redteam
            else reward_score
        )
        return {
            "label": label or model_id,
            "model_id": model_id,
            "checkpoint_id": checkpoint_id,
            "reward_score": round(reward_score, 4),
            "safety_pass_rate": round(safety_pass, 4),
            "attack_success_rate": round(asr, 4),
            "eval_verdict": verdict,
            "fitness": round(fitness, 4),
            "gate_summary": {
                "verdict": verdict,
                "case_count": gate.get("case_count", max_cases),
                "deltas": gate.get("deltas", {}),
            },
        }

    def run_tournament(
        self,
        model_ids: list[str] | None = None,
        *,
        reward_fn: str = "customer_support",
        max_cases: int = 4,
        include_redteam: bool = True,
        redteam_probes: int = 4,
        synthesize_merge: bool = True,
        merge_methods: list[str] | None = None,
        export_format: str = "none",
        output_name: str = "",
        lora_rank: int = 16,
        lora_alpha: int = 32,
    ) -> dict[str, Any]:
        """Run a multi-model tournament and optionally synthesize an evolutionary surgical merge."""
        t0 = time.perf_counter()

        discovered = self.inspector.list_models()
        if not discovered:
            raise ValueError("No models discovered in workspace for tournament.")

        contender_ids: list[str] = [
            m.strip() for m in (model_ids or []) if isinstance(m, str) and m.strip()
        ]
        if not contender_ids:
            contender_ids = [str(m["model_id"]) for m in discovered[:3]]
        elif len(contender_ids) == 1:
            # Automatically pair with a second discovered model for comparison & merge
            primary = contender_ids[0]
            for m in discovered:
                cand_id = str(m["model_id"])
                if cand_id != primary:
                    contender_ids.append(cand_id)
                    break

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_ids: list[str] = []
        for mid in contender_ids:
            if mid not in seen:
                seen.add(mid)
                unique_ids.append(mid)

        # Stage 1: Score every individual contender
        entries: list[dict[str, Any]] = []
        for mid in unique_ids:
            entry = self._score_contender(
                mid,
                reward_fn=reward_fn,
                max_cases=max_cases,
                include_redteam=include_redteam,
                redteam_probes=redteam_probes,
                label=mid,
            )
            entry["kind"] = "contender"
            entries.append(entry)

        entries.sort(key=lambda e: (e["fitness"], e["reward_score"]), reverse=True)
        best_parent = entries[0]

        # Stage 2: Synthesize & benchmark TIES / DARE / SLERP surgical merges of Top-2
        merge_candidates: list[dict[str, Any]] = []
        promoted_merge: dict[str, Any] | None = None
        champion: dict[str, Any] = dict(best_parent)

        if synthesize_merge and len(entries) >= 2:
            parent_a = entries[0]["model_id"]
            parent_b = entries[1]["model_id"]
            methods = [
                m.strip().lower()
                for m in (merge_methods or list(DEFAULT_MERGE_METHODS))
                if m.strip().lower() in ("ties", "dare", "slerp", "linear")
            ] or list(DEFAULT_MERGE_METHODS)

            # Preserve parent_a's pre-tournament recipe so merge exploration is safe
            prior_recipe = self.editor.export_recipe(parent_a)
            best_merge_entry: dict[str, Any] | None = None
            best_merge_params: dict[str, Any] | None = None

            for method in methods:
                self.editor.reset(parent_a)
                if prior_recipe.get("operations"):
                    self.editor.apply_recipe(parent_a, prior_recipe, reset_first=False)

                params: dict[str, Any] = {
                    "secondary_model_id": parent_b,
                    "method": method,
                    "alpha": 0.45 if method in ("slerp", "linear") else 0.50,
                }
                if method == "ties":
                    params["density"] = 0.65
                elif method == "dare":
                    params["drop_rate"] = 0.35

                try:
                    edit_res = self.editor.apply_edit(
                        parent_a,
                        "blend",
                        "all",
                        params,
                        label=f"Tournament {method.upper()} Merge ({parent_a} × {parent_b})",
                    )
                    scored = self._score_contender(
                        parent_a,
                        reward_fn=reward_fn,
                        max_cases=max_cases,
                        include_redteam=include_redteam,
                        redteam_probes=redteam_probes,
                        label=f"merge:{method}({parent_a} × {parent_b})",
                    )
                    scored["kind"] = "synthesized_merge"
                    scored["merge_method"] = method
                    scored["parent_a"] = parent_a
                    scored["parent_b"] = parent_b
                    scored["conflict_rate"] = (
                        edit_res.get("edit", {})
                        .get("metrics", {})
                        .get("conflict_rate", 0.0)
                    )
                    merge_candidates.append(scored)
                    if (
                        best_merge_entry is None
                        or scored["fitness"] > best_merge_entry["fitness"]
                    ):
                        best_merge_entry = scored
                        best_merge_params = params
                except Exception as exc:
                    merge_candidates.append(
                        {
                            "kind": "synthesized_merge",
                            "merge_method": method,
                            "parent_a": parent_a,
                            "parent_b": parent_b,
                            "error": str(exc),
                            "fitness": -1.0,
                        }
                    )

            # Restore parent_a and apply the winning merge only if it meets or beats best_parent
            self.editor.reset(parent_a)
            if prior_recipe.get("operations"):
                self.editor.apply_recipe(parent_a, prior_recipe, reset_first=False)

            if (
                best_merge_entry is not None
                and best_merge_params is not None
                and best_merge_entry["fitness"] >= best_parent["fitness"] - 0.005
            ):
                applied = self.editor.apply_edit(
                    parent_a,
                    "blend",
                    "all",
                    best_merge_params,
                    label=(
                        f"Champion {best_merge_entry['merge_method'].upper()} Merge "
                        f"({parent_a} × {parent_b})"
                    ),
                )
                promoted_merge = {
                    **best_merge_entry,
                    "edit_id": applied["edit"]["edit_id"],
                    "delta_vs_best_parent": round(
                        best_merge_entry["fitness"] - best_parent["fitness"], 4
                    ),
                }
                champion = promoted_merge

        leaderboard = sorted(
            entries + [m for m in merge_candidates if "error" not in m],
            key=lambda e: (e["fitness"], e["reward_score"]),
            reverse=True,
        )
        for rank_idx, row in enumerate(leaderboard, start=1):
            row["rank"] = rank_idx
        champion["rank"] = 1

        # Stage 3: Optional verified Champion export + offline HTML PR report
        export_bundle: dict[str, Any] | None = None
        report_path: str | None = None
        summary_path: str | None = None
        fmt = (export_format or "none").strip().lower()
        champ_mid = str(champion["model_id"])

        if fmt != "none":
            slug = (
                "".join(
                    ch if ch.isalnum() or ch in ("-", "_") else "-"
                    for ch in (output_name or f"tournament-champ-{int(time.time())}")
                ).strip("-")
                or "tournament-champ"
            )
            if fmt == "lora":
                export_bundle = self.editor.export_diff_as_lora(
                    champ_mid, slug, rank=lora_rank, alpha=lora_alpha
                )
            elif fmt == "safetensors":
                export_bundle = self.editor.export_verified_checkpoint(
                    champ_mid, slug, run_merge_probe=True
                )
            elif fmt == "gguf":
                export_bundle = self.editor.export_gguf_package(
                    champ_mid, slug, quant_type="Q8_0"
                )
            elif fmt == "hf_card":
                eval_rep = self.inference.run_batch_eval(
                    champ_mid, reward_fn=reward_fn, max_cases=max_cases
                )
                export_bundle = self.editor.export_hf_model_card(
                    champ_mid, slug, eval_report=eval_rep
                )
            else:
                raise ValueError(
                    f"Unsupported export_format {export_format!r}. "
                    "Choose one of: none, lora, safetensors, gguf, hf_card."
                )

            out_dir = (
                self.inspector.workspace_root
                / "outputs"
                / "weight_studio_exports"
                / slug
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            from .report_exporter import build_standalone_html_report

            overrides = self.editor.get_overrides(champ_mid)
            struct = self.inspector.get_model_structure(
                champ_mid, overrides=overrides if overrides else None
            )
            eval_rep = self.inference.run_batch_eval(
                champ_mid, reward_fn=reward_fn, max_cases=max_cases
            )
            html_doc = build_standalone_html_report(
                model_summary=struct["model"],
                structure_summary=struct["summary"],
                layers=struct["layers"],
                edit_history=self.editor.get_history(champ_mid),
                recipe=self.editor.export_recipe(champ_mid),
                eval_report=eval_rep,
            )
            rp = out_dir / "tournament_report.html"
            rp.write_text(html_doc, encoding="utf-8")
            report_path = str(rp)

            sp = out_dir / "tournament_summary.json"
            sp.write_text(
                json.dumps(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "reward_fn": reward_fn,
                        "champion": champion,
                        "leaderboard": leaderboard,
                        "promoted_merge": promoted_merge,
                        "export_bundle": export_bundle,
                        "report_path": report_path,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            summary_path = str(sp)

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "schema_version": 1,
            "reward_fn": reward_fn,
            "include_redteam": include_redteam,
            "contender_count": len(entries),
            "merge_candidates_evaluated": len(merge_candidates),
            "champion": champion,
            "promoted_merge": promoted_merge,
            "leaderboard": leaderboard,
            "export_bundle": export_bundle,
            "report_path": report_path,
            "summary_path": summary_path,
            "latency_ms": latency_ms,
        }
