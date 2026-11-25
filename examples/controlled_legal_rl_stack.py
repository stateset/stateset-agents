#!/usr/bin/env python3
"""End-to-end runnable demo of the StateSet Controlled Multi-Model Legal RL Stack.

Demonstrates:
1. Closed-universe Client Matter Sandboxes (key + peripheral docs, confined tools)
2. Hierarchical Issue-Group Binary Rubric Reward (zero explicit length penalty)
3. Rank-64 LoRA + CISPO + Dr. GRPO (`dr_cispo`) optimization controls
4. Nemotron 3.5 Super VL anchor hyperparameter sweep transferred to Muse Spark Open,
   Beam 1.0, GLM-5.3 (River) and Inkling 2 Small (Tinker) on StateSet Sandboxes
5. Exporting a SHA-256 study manifest and Markdown Training Configuration Appendix
"""

from __future__ import annotations

import json
from pathlib import Path

from stateset_agents.training.controlled_study import (
    ControlledStudyConfig,
    StudyModelSpec,
    SweepTrialResult,
    build_controlled_run_plans,
    export_study_bundle,
    run_anchor_hyperparameter_sweep,
    run_offline_sandbox_stack_smoke_sync,
)


def _offline_sweep_evaluator(
    anchor: StudyModelSpec, lr: float, group_size: int
) -> SweepTrialResult:
    """Deterministic illustrative sweep evaluator for offline dry runs."""
    lr_bonus = 0.14 if lr == 1e-4 else (0.08 if lr == 3e-5 else 0.05)
    gs_bonus = 0.09 if group_size == 8 else (0.06 if group_size == 16 else 0.02)
    reward = round(0.66 + lr_bonus + gs_bonus, 4)
    return SweepTrialResult(
        anchor_model=anchor.display_name,
        learning_rate=lr,
        rollout_group_size=group_size,
        validation_reward=reward,
        criteria_pass_rate=round(min(1.0, reward + 0.03), 4),
        issue_group_pass_rate=round(max(0.0, reward - 0.04), 4),
        all_pass_rate=round(max(0.0, reward - 0.16), 4),
    )


def main() -> None:
    cfg = ControlledStudyConfig()
    selection = run_anchor_hyperparameter_sweep(cfg, _offline_sweep_evaluator)
    plans = build_controlled_run_plans(cfg, selection)
    smoke = run_offline_sandbox_stack_smoke_sync(cfg)

    out_dir = Path("outputs/controlled_legal_rl_stack")
    bundle = export_study_bundle(out_dir, cfg, plans, selection=selection)

    print("=== Controlled Multi-Model RL Stack Smoke ===")
    print(
        json.dumps(
            {
                "anchor_sweep_winner": {
                    "anchor_model": selection.anchor_model,
                    "learning_rate": selection.selected_learning_rate,
                    "rollout_group_size": selection.selected_rollout_group_size,
                    "validation_reward": selection.selected_validation_reward,
                },
                "sandbox_smoke": {
                    "strong_beats_partial": smoke["strong_beats_partial"],
                    "group_advantages": smoke["group_advantages"],
                },
                "bundle": bundle,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
