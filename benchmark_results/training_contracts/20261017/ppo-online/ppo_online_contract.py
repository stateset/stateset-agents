#!/usr/bin/env python3
"""Retain a CPU PPO learning/continuation contract without model downloads.

The task is a synthetic one-action bandit on a tiny randomly initialized GPT-2.
This verifies optimizer direction and exact continuation, not language quality.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import torch

from stateset_agents.evaluation.implementation import package_implementation
from stateset_agents.training.ppo_trainer import PPOConfig, PPOTrainer
from stateset_agents.training.state_fingerprint import state_fingerprint


class BanditTokenizer:
    """One fixed prompt token and readable action IDs for a synthetic bandit."""

    eos_token_id = None

    def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        ids = torch.ones((len(texts), 1), dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids: list[int], **kwargs: Any) -> str:
        return " ".join(map(str, ids))

    def save_pretrained(self, path: str | Path) -> None:
        (Path(path) / "tokenizer_config.json").write_text(
            json.dumps({"kind": "synthetic-bandit-tokenizer", "prompt_id": 1}) + "\n"
        )


def _learner_state(trainer: PPOTrainer) -> str:
    return str(
        state_fingerprint(
            {
                "policy": trainer.model.state_dict(),
                "critic": trainer.value_head.state_dict(),
                "policy_optimizer": trainer.policy_optimizer.state_dict(),
                "value_optimizer": trainer.value_optimizer.state_dict(),
                "policy_scheduler": trainer.policy_scheduler.state_dict(),
                "value_scheduler": trainer.value_scheduler.state_dict(),
                "step": trainer.global_step,
                "metrics": trainer.metrics_history,
            }
        )
    )


async def run(output: Path) -> dict[str, Any]:
    """Train a real causal transformer, reload midpoint, and compare continuation."""
    from transformers import GPT2Config, GPT2LMHeadModel

    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=16,
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_positions=8,
            bos_token_id=1,
            eos_token_id=None,
            resid_pdrop=0.2,
            embd_pdrop=0.2,
            attn_pdrop=0.2,
        )
    )
    trainer = PPOTrainer(
        PPOConfig(
            output_dir=str(output),
            num_episodes=48,
            num_ppo_epochs=1,
            mini_batch_size=4,
            num_generations=8,
            learning_rate=0.01,
            max_prompt_length=1,
            max_completion_length=1,
            temperature=1.0,
            value_hidden_size=16,
            entropy_coef=0.0,
            use_lora=False,
            bf16=False,
            report_to="none",
        ),
        model,
        BanditTokenizer(),
        lambda _, completion: float(completion == "7"),
    )

    def target_probability() -> float:
        modes = {module: module.training for module in model.modules()}
        try:
            model.eval()
            with torch.no_grad():
                logits = model(input_ids=torch.ones((1, 1), dtype=torch.long)).logits[
                    0, -1
                ]
                return float(torch.softmax(logits, dim=-1)[7].item())
        finally:
            for module, mode in modes.items():
                module.training = mode

    baseline = target_probability()
    first = await trainer.train(["Select the rewarded action."], num_episodes=24)
    checkpoint = trainer.save_checkpoint(str(output / "checkpoint-24"))
    continued = await trainer.train(["Select the rewarded action."], num_episodes=48)
    candidate = target_probability()
    expected_state = _learner_state(trainer)
    trainer.load_runtime_checkpoint(checkpoint)
    resumed = await trainer.train(["Select the rewarded action."], num_episodes=48)
    same_samples = all(
        json.loads(
            (Path(continued["rollout_directory"]) / f"rollout-{step}.json").read_text()
        )["rows"]
        == json.loads(
            (Path(resumed["rollout_directory"]) / f"rollout-{step}.json").read_text()
        )["rows"]
        for step in range(24, 48)
    )
    report = {
        "schema_version": 1,
        "kind": "stateset-ppo-online-cpu-contract",
        "synthetic": True,
        "quality_certified": False,
        "gpu_verified": False,
        "implementation": package_implementation(),
        "baseline_target_probability": baseline,
        "trained_target_probability": candidate,
        "learning_direction_passed": candidate > 0.8 and candidate > baseline + 0.5,
        "exact_continuation_passed": same_samples
        and expected_state == _learner_state(trainer),
        "update_target": 48,
        "replayed_updates": 24,
        "episodes_per_update": 32,
        "torch_version": str(torch.__version__),
        "rollout_directories": [
            str(Path(r["rollout_directory"]).relative_to(output))
            for r in (first, continued, resumed)
        ],
    }
    report["passed"] = (
        report["learning_direction_passed"] and report["exact_continuation_passed"]
    )
    (output / "contract.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> int:
    """Run the bounded local contract; a failed assertion returns nonzero."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(run(args.output))
    print(json.dumps({k: v for k, v in report.items() if k != "implementation"}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
