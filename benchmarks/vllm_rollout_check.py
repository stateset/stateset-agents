#!/usr/bin/env python3
"""Live proof that vLLM rollouts stay on-policy through a real optimizer step.

Runs on one GPU with vLLM installed. It attaches a ``VLLMGenerator`` to a
``MultiTurnAgent`` holding the same Hugging Face model, samples a group of
turns through the engine, measures how far the engine's reported log-probs
are from the policy's own log-probs of those tokens, takes real GRPO
optimizer steps on the token path with ``old_logprobs_source="sampler"``,
measures the gap again (the engine is now stale), calls
``VLLMGenerator.sync_weights`` and measures once more (the engine serves the
updated policy). The JSON it writes records versions, hardware, the three
gaps, and a pass/fail verdict:

* ``after_sync`` must be no worse than ``before_step`` (up to bf16 kernel
  noise: 3x the initial gap or 0.05 nats, whichever is larger), and
* ``after_step_before_sync`` must exceed ``after_sync`` (the step really
  moved the policy and the sync really moved the engine).

    python benchmarks/vllm_rollout_check.py --model Qwen/Qwen2.5-0.5B-Instruct \\
        --revision <sha> --output vllm-rollout-check.json [--peft]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROMPTS = [
    "What is 17 + 26? Answer with the number only.",
    "Name the largest planet in the solar system in one word.",
    "Give one synonym for 'quick'.",
    "What colour do you get by mixing blue and yellow?",
]
REWARDS = [1.0, 0.0, 0.5, 0.25]


def _gap(a: list[float], b: list[float]) -> dict[str, float]:
    diffs = [abs(x - y) for x, y in zip(a, b, strict=True)]
    return {
        "max_abs": max(diffs) if diffs else 0.0,
        "mean_abs": sum(diffs) / len(diffs) if diffs else 0.0,
        "tokens": float(len(diffs)),
    }


def _merge(gaps: list[dict[str, float]]) -> dict[str, float]:
    tokens = sum(g["tokens"] for g in gaps)
    return {
        "max_abs": max(g["max_abs"] for g in gaps),
        "mean_abs": sum(g["mean_abs"] * g["tokens"] for g in gaps) / max(tokens, 1),
        "tokens": tokens,
    }


def _policy_logprobs(agent: Any, turns: list[Any], torch: Any) -> list[list[float]]:
    from stateset_agents.training import loss_computation as lc

    rows = [
        (list(t.metadata["prompt_token_ids"]), list(t.metadata["token_ids"]), i)
        for i, t in enumerate(turns)
    ]
    device = lc._resolve_model_device(agent, torch)
    lp, mask, _ = lc._forward_token_rows(agent.model, rows, device, 4, grad=False)
    out = []
    for row_lp, row_mask in zip(lp, mask, strict=True):
        out.append([float(v) for v in row_lp[row_mask.bool()].tolist()])
    return out


def _engine_logprobs(gen: Any, turns: list[Any]) -> list[list[float]]:
    return [
        gen.token_logprobs_for_ids(
            list(t.metadata["prompt_token_ids"]), list(t.metadata["token_ids"])
        )
        for t in turns
    ]


async def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
    from stateset_agents.core.trajectory import (
        ConversationTurn,
        MultiTurnTrajectory,
        TrajectoryGroup,
    )
    from stateset_agents.training import loss_computation as lc
    from stateset_agents.training.vllm_backend import VLLMConfig, VLLMGenerator

    started = time.monotonic()
    torch.manual_seed(args.seed)
    peft_config = (
        {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "target_modules": ["q_proj", "v_proj"],
            "task_type": "CAUSAL_LM",
        }
        if args.peft
        else None
    )
    agent = MultiTurnAgent(
        AgentConfig(
            model_name=args.model,
            model_revision=args.revision,
            torch_dtype="bfloat16",
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=1.0,
            max_new_tokens=args.max_tokens,
            use_peft=bool(args.peft),
            peft_config=peft_config,
        )
    )
    await agent.initialize()
    gen = VLLMGenerator(
        VLLMConfig(
            model_name=args.model,
            revision=args.revision,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=512,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=args.max_tokens,
            enable_prefix_caching=False,
            dtype="bfloat16",
            seed=args.seed,
        )
    )
    if not await gen.initialize():
        raise RuntimeError("vLLM engine failed to initialise")
    import vllm

    agent.set_rollout_backend(gen)

    turns = [
        await agent.generate_turn([{"role": "user", "content": p}]) for p in PROMPTS
    ]
    for t in turns:
        if t.metadata.get("rollout_backend") != "VLLMGenerator":
            raise RuntimeError(
                f"turn did not come from the engine: {t.metadata.get('rollout_backend_error')}"
            )
        if not t.metadata.get("token_ids") or not t.metadata.get("sampler_log_probs"):
            raise RuntimeError("engine turn carries no token ids / log-probs")
    sampler = [[float(x) for x in t.metadata["sampler_log_probs"]] for t in turns]
    before_step = _merge(
        [
            _gap(s, p)
            for s, p in zip(sampler, _policy_logprobs(agent, turns, torch), strict=True)
        ]
    )

    group = TrajectoryGroup(
        scenario_id="vllm-rollout-check",
        trajectories=[
            MultiTurnTrajectory(
                turns=[ConversationTurn(role="user", content=p), t], total_reward=r
            )
            for p, t, r in zip(PROMPTS, turns, REWARDS, strict=True)
        ],
    )
    config = SimpleNamespace(
        max_prompt_length=256,
        max_completion_length=args.max_tokens,
        clip_ratio=0.2,
        seq_clip_ratio=3e-4,
        entropy_coef=0.0,
        advantage_normalization=True,
        baseline_type="group_mean",
        reward_clip=None,
        generation_batch_size=4,
        objective=None,
        objective_overrides=None,
        old_logprobs_source="sampler",
        bf16=True,
        fp16=False,
    )
    params = [p for p in agent.model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    agent.model.train()
    step_metrics = []
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        out = lc.compute_grpo_loss([group], config, agent, 0.0, 0, lambda m, n: None)
        out["total_loss"].backward()
        optimizer.step()
        step_metrics.append(
            {
                k: (float(v.item()) if hasattr(v, "item") else v)
                for k, v in out.items()
                if k
                in (
                    "total_loss",
                    "ratio_mean",
                    "clip_fraction",
                    "path",
                    "old_logprobs_source",
                    "num_rows",
                )
            }
        )
    agent.model.eval()

    policy_after = _policy_logprobs(agent, turns, torch)
    stale = _merge(
        [
            _gap(e, p)
            for e, p in zip(_engine_logprobs(gen, turns), policy_after, strict=True)
        ]
    )
    synced = agent.sync_rollout_backend()
    after_sync = _merge(
        [
            _gap(e, p)
            for e, p in zip(_engine_logprobs(gen, turns), policy_after, strict=True)
        ]
    )
    post_turn = await agent.generate_turn([{"role": "user", "content": PROMPTS[0]}])

    tolerance = max(3.0 * before_step["max_abs"], 0.05)
    verdict = {
        "sync_succeeded": bool(synced),
        "after_sync_within_tolerance": after_sync["max_abs"] <= tolerance,
        "step_moved_policy_and_sync_moved_engine": stale["max_abs"]
        > after_sync["max_abs"],
        "post_sync_turn_from_engine": post_turn.metadata.get("rollout_backend")
        == "VLLMGenerator"
        and post_turn.metadata.get("rollout_backend_version") == 1,
        "tolerance_max_abs": tolerance,
    }
    verdict["passed"] = all(v for k, v in verdict.items() if k != "tolerance_max_abs")
    import transformers

    import stateset_agents

    return {
        "schema_version": 1,
        "kind": "vllm-rollout-sync-check",
        "measured": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_revision": args.revision,
        "peft": bool(args.peft),
        "dtype": "bfloat16",
        "versions": {
            "vllm": getattr(vllm, "__version__", "?"),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "stateset_agents": getattr(stateset_agents, "__version__", "?"),
            "python": platform.python_version(),
        },
        "hardware": {
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            ),
            "gpu_count": torch.cuda.device_count(),
        },
        "engine_model_path": getattr(gen, "engine_model_path", None),
        "weight_sync_count": gen.weight_sync_count,
        "rollout_backend_error": agent.rollout_backend_error,
        "gaps_nats": {
            "engine_vs_policy_before_step": before_step,
            "engine_vs_policy_after_step_before_sync": stale,
            "engine_vs_policy_after_sync": after_sync,
        },
        "optimizer": {"kind": "adam", "lr": args.lr, "steps": args.steps},
        "step_metrics": step_metrics,
        "verdict": verdict,
        "wall_clock_seconds": time.monotonic() - started,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--peft", action="store_true")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    result = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"verdict": result["verdict"], "gaps_nats": result["gaps_nats"]}, indent=2
        )
    )
    return 0 if result["verdict"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
