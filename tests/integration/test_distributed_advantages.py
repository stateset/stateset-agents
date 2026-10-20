"""Real DDP reward populations match a combined batch and reject peer drift."""

from __future__ import annotations

import asyncio
import copy
import itertools
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.integration


def _worker(rank, directory):
    import torch
    import torch.distributed as dist

    from stateset_agents.training.config import TrainingConfig
    from stateset_agents.training.distributed import (
        DistributedConfig,
        DistributedTrainer,
    )
    from stateset_agents.training.loss_computation import (
        _token_batch_advantages,
        _token_objective,
    )
    from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer
    from tests.unit.test_token_group_aggregation import TinyPolicy, _groups, _reference

    torch.set_num_threads(1)
    if sys.platform.startswith("linux"):
        os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    root = Path(directory)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=(root / "rendezvous").as_uri(),
        timeout=timedelta(seconds=30),
    )
    results, guards = [], []
    try:
        for kind, clip, inner, steps, enhanced in itertools.product(
            ["batch_norm", "global_mean", "global_mean_normalized"],
            [None, 3.0],
            [1, 3],
            [1, 2],
            [False, True],
        ):
            config = TrainingConfig(
                objective="reinforce_pp_baseline" if kind == "batch_norm" else None,
                baseline_type="global_mean",
                advantage_normalization=kind == "global_mean_normalized",
                reward_clip=clip,
                num_gradient_updates=inner,
                gradient_accumulation_steps=steps,
                generation_batch_size=1,
                bf16=False,
                fp16=False,
                max_grad_norm=100.0,
                rollout_sync=True,
                zero_signal_patience=0,
                use_reference_model=enhanced,
                beta=0.2 if enhanced else 0.0,
                output_dir=str(root / f"rank-{rank}"),
                report_to="none",
            )
            trainer = DistributedTrainer(
                SimpleNamespace(model=TinyPolicy()),
                SimpleNamespace(),
                config=config,
                distributed_config=DistributedConfig(
                    backend="gloo", world_size=2, rank=rank, strategy="ddp"
                ),
            )
            reference = MultiTurnGRPOTrainer(
                SimpleNamespace(model=TinyPolicy()),
                SimpleNamespace(),
                config=copy.deepcopy(config),
            )
            for worker in (trainer, reference):
                worker.optimizer = torch.optim.SGD(
                    worker.agent.model.parameters(), lr=0.1
                )
                worker._update_global_stats(0.5, 3)
                if enhanced:
                    worker.reference_model = TinyPolicy(seed=13)
            trainer._wrap_model_ddp()
            errors, advantage_errors = [], []
            for batch in range(2):
                groups = _groups()
                if batch:
                    for index, group in enumerate(groups):
                        for trajectory in group.trajectories:
                            trajectory.total_reward = trajectory.total_reward * 2 + (
                                17 if index else -9
                            )
                rewards = [
                    torch.tensor(
                        [t.total_reward for t in group.trajectories],
                        dtype=torch.float32,
                    )
                    for group in groups
                ]
                if clip is not None:
                    rewards = [r.clamp(-clip, clip) for r in rewards]
                if kind == "batch_norm":
                    std = torch.cat(rewards).std(correction=0)
                    expected = (rewards[rank] - rewards[rank].mean()) / (std + 1e-8)
                else:
                    expected = rewards[rank] - trainer._global_reward_mean
                    if (
                        kind == "global_mean_normalized"
                        and expected.std(correction=0) > 0
                    ):
                        expected = (expected - expected.mean()) / (
                            expected.std(correction=0) + 1e-8
                        )
                actual = _token_batch_advantages(
                    [groups[rank]],
                    config,
                    _token_objective(config),
                    "cpu",
                    model=trainer.agent.model,
                    global_reward_mean=trainer._global_reward_mean,
                    global_reward_count=trainer._global_reward_count,
                    global_reward_sum=trainer._global_reward_sum,
                )[0]
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
                advantage_errors.append((actual - expected).abs().max().item())
                # Independent on-policy token-loss gradient control for batch normalization.
                if kind == "batch_norm" and batch == 0 and not enhanced:
                    independent = TinyPolicy()
                    loss, _ = _reference(groups, independent, config)
                    loss.backward()
                    from stateset_agents.training.loss_computation import (
                        compute_grpo_loss,
                    )

                    canonical = TinyPolicy()
                    out = compute_grpo_loss(
                        groups, config, SimpleNamespace(model=canonical), 0, 0, None
                    )
                    out["total_loss"].backward()
                    torch.testing.assert_close(
                        canonical.lookup.weight.grad, independent.lookup.weight.grad
                    )
                wanted = asyncio.run(reference.training_step(copy.deepcopy(groups)))
                metrics = asyncio.run(
                    trainer.training_step([copy.deepcopy(groups[rank])])
                )
                for key in [
                    "total_loss",
                    "mean_advantage",
                    "advantage_std",
                    "kl_penalty",
                    "ratio_mean",
                ]:
                    errors.append(abs(metrics[key] - wanted[key]))
                    assert metrics[key] == pytest.approx(
                        wanted[key], rel=2e-5, abs=2e-6
                    ), (kind, key)
                assert trainer._global_reward_count == reference._global_reward_count
                assert trainer._global_reward_mean == pytest.approx(
                    reference._global_reward_mean, abs=1e-14
                )
            for worker in (trainer, reference):
                worker._flush_accumulation(torch)
            actual = trainer.agent.model.module.lookup.weight.detach()
            wanted = reference.agent.model.lookup.weight.detach()
            torch.testing.assert_close(actual, wanted, rtol=1e-6, atol=1e-7)
            expected_count = 3 if kind == "batch_norm" else 13
            assert trainer._global_reward_count == expected_count
            results.append(
                {
                    "kind": kind,
                    "clip": clip,
                    "inner": inner,
                    "steps": steps,
                    "enhanced": enhanced,
                    "count": trainer._global_reward_count,
                    "mean": trainer._global_reward_mean,
                    "max_advantage_error": max(advantage_errors),
                    "max_metric_error": max(errors),
                    "max_parameter_error": (actual - wanted).abs().max().item(),
                }
            )
            (root / f"advantages-{rank}.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )

        edge_cases = []
        for case in ["constant", "singleton", "offset", "tiny"]:
            edge_config = copy.deepcopy(config)
            edge_config.objective = "reinforce_pp_baseline"
            edge_config.reward_clip = None
            groups = _groups()
            if case == "singleton":
                groups[0].trajectories = groups[0].trajectories[:1]
            for group in groups:
                for trajectory in group.trajectories:
                    if case == "constant":
                        trajectory.total_reward = 2.0
                    elif case == "offset":
                        trajectory.total_reward = (
                            1e12 + trajectory.total_reward * 131072
                        )
                    elif case == "tiny":
                        trajectory.total_reward *= 1e-10
            rewards = [
                torch.tensor(
                    [t.total_reward for t in g.trajectories], dtype=torch.float32
                )
                for g in groups
            ]
            std = torch.cat(rewards).double().std(correction=0).float()
            expected = (
                (rewards[rank] - rewards[rank].mean()) / (std + 1e-8)
                if std > 1e-8
                else torch.zeros_like(rewards[rank])
            )
            actual = _token_batch_advantages(
                [groups[rank]],
                edge_config,
                _token_objective(edge_config),
                "cpu",
                model=trainer.agent.model,
            )[0]
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
            edge_cases.append(
                {
                    "case": case,
                    "max_advantage_error": (actual - expected).abs().max().item(),
                }
            )
        (root / f"advantage-edge-{rank}.json").write_text(
            json.dumps(edge_cases, indent=2) + "\n"
        )

        # Corrupt only rank 1. Every rejection must leave baseline/model untouched,
        # and a good batch on the same process group must work afterwards.
        for fault in [
            "reward_nan",
            "reward_inf_clipped",
            "reward_fp32_overflow",
            "clip",
            "estimator",
            "normalization",
            "mean",
            "count",
            "sum",
            "update",
            "overflow",
            "computed_overflow",
            "recovery",
        ]:
            config = copy.deepcopy(config)
            config.objective = None
            config.baseline_type = "global_mean"
            config.advantage_normalization = False
            config.reward_clip = 3.0
            groups = _groups()
            seen = []
            kwargs = {
                "global_reward_mean": 0.5,
                "global_reward_count": 3,
                "global_reward_sum": 1.5,
                "update_global_stats": lambda mean, count, observed=seen: observed.append(
                    (mean, count)
                ),
            }
            if fault == "computed_overflow":
                # Finite FP32 rewards and prior mean can still overflow subtraction.
                config.reward_clip = None
                kwargs.update(global_reward_mean=-9e38 / 3, global_reward_sum=-9e38)
                if rank == 1:
                    groups[1].trajectories[0].total_reward = 3e38
            if rank == 1:
                if fault == "reward_nan":
                    groups[1].trajectories[0].total_reward = float("nan")
                elif fault == "reward_inf_clipped":
                    groups[1].trajectories[0].total_reward = float("inf")
                elif fault == "reward_fp32_overflow":
                    groups[1].trajectories[0].total_reward = 1e50
                    config.reward_clip = None
                elif fault == "clip":
                    config.reward_clip = -1.0
                elif fault == "estimator":
                    config.objective = "grpo"
                elif fault == "normalization":
                    config.advantage_normalization = True
                elif fault == "mean":
                    kwargs.update(global_reward_mean=0.75, global_reward_sum=2.25)
                elif fault == "count":
                    kwargs.update(global_reward_count=4, global_reward_sum=2.0)
                elif fault == "sum":
                    kwargs["global_reward_sum"] = float("nan")
                elif fault == "update":
                    kwargs["update_global_stats"] = None
            if fault == "overflow":
                kwargs.update(global_reward_count=2**63 - 1, global_reward_sum=None)
            try:
                _token_batch_advantages(
                    [groups[rank]],
                    config,
                    _token_objective(config),
                    "cpu",
                    model=trainer.agent.model,
                    **kwargs,
                )
            except ValueError:
                rejected = True
            else:
                rejected = False
            assert rejected is (fault != "recovery"), fault
            assert seen == ([(0.4, 5)] if fault == "recovery" else []), (fault, seen)
            guards.append({"case": fault, "rejected": rejected})
            (root / f"advantage-guards-{rank}.json").write_text(
                json.dumps(guards, indent=2) + "\n"
            )
    finally:
        dist.destroy_process_group()


def test_ddp_reward_populations_and_frozen_rollout_baselines(tmp_path):
    torch = pytest.importorskip("torch")
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_gloo_available()
    ):
        pytest.skip("Gloo distributed support required")
    context = torch.multiprocessing.spawn(
        _worker, args=(str(tmp_path.resolve()),), nprocs=2, join=False
    )
    try:
        deadline = time.monotonic() + 90
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("two-rank advantages exceeded 90 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        destination = os.environ.get("STATESET_DISTRIBUTED_CONTRACT_REPORT_DIR")
        if destination:
            output = Path(destination)
            output.mkdir(parents=True, exist_ok=True)
            for path in tmp_path.glob("*.json"):
                (output / path.name).write_bytes(path.read_bytes())
    first = json.loads((tmp_path / "advantages-0.json").read_text())
    second = json.loads((tmp_path / "advantages-1.json").read_text())
    assert len(first) == len(second) == 48
    for left, right in zip(first, second, strict=True):
        assert left["count"] == right["count"]
        assert left["mean"] == right["mean"]
    guards = json.loads((tmp_path / "advantage-guards-0.json").read_text())
    assert guards == json.loads((tmp_path / "advantage-guards-1.json").read_text())
    assert sum(row["rejected"] for row in guards) == 12
    assert guards[-1] == {"case": "recovery", "rejected": False}
