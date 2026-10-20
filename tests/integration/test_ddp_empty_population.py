"""Zero-data peers preserve the combined objective, pending resume and loop flow."""

from __future__ import annotations

import asyncio
import copy
import itertools
import json
import os
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _make(
    root,
    rank=0,
    world=2,
    objective="grpo",
    entropy=0.07,
    inner=1,
    optimizer="sgd",
    scaler=False,
):
    import torch

    from tests.integration.test_replica_consensus import _make as make

    trainer = make(root / f"rank-{rank}", rank, world)
    trainer.config.objective = objective
    trainer.config.entropy_coef = entropy
    trainer.config.beta = 0.2 if entropy else 0.0
    trainer.config.use_reference_model = bool(entropy)
    trainer.config.gradient_accumulation_steps = 3
    trainer.config.num_gradient_updates = inner
    trainer.config.lr_scheduler_type = "constant"
    trainer.distributed_config.checkpoint_dir = str(root / "checkpoints")
    trainer.agent.model.register_parameter(
        "unused", torch.nn.Parameter(torch.tensor([0.2]))
    )
    cls = torch.optim.SGD if optimizer == "sgd" else torch.optim.AdamW
    trainer.optimizer = cls(
        trainer.agent.model.parameters(),
        lr=0.05,
        weight_decay=0.1,
        **({"momentum": 0.9} if optimizer == "sgd" else {}),
    )
    trainer.scaler = torch.amp.GradScaler("cpu", init_scale=128.0) if scaler else None
    if world > 1:
        trainer._wrap_model_ddp()
    return trainer


def _worker(rank, directory):
    import torch
    import torch.distributed as dist

    from stateset_agents.training.ddp_checkpoint_state import capture_payload
    from tests.integration.test_learner_preflight import _same
    from tests.unit.test_token_group_aggregation import _groups

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
    results = []
    try:
        for index, (objective, entropy, inner, pattern, optimizer) in enumerate(
            itertools.product(
                [
                    "grpo",
                    "dr_grpo",
                    "dapo",
                    "cispo",
                    "rloo",
                    "gspo",
                    "reinforce_pp_baseline",
                    None,
                ],
                [0.0, 0.07],
                [1, 2],
                ["fixed", "alternating"],
                ["sgd", "adamw"],
            )
        ):
            folder = root / str(index)
            trainer = _make(
                folder,
                rank,
                objective=objective,
                entropy=entropy,
                inner=inner,
                optimizer=optimizer,
            )
            reference = _make(
                folder / "serial",
                world=1,
                objective=objective,
                entropy=entropy,
                inner=inner,
                optimizer=optimizer,
            )
            resumed = None
            for micro in [0, 1]:
                groups = [_groups()[micro]]
                active = 1 if pattern == "fixed" else micro
                local = groups if rank == active else []
                calls = trainer.agent.model.module.calls
                metrics = asyncio.run(trainer.training_step(local))
                expected_metrics = asyncio.run(reference.training_step(groups))
                assert metrics["num_trajectories"] == len(groups[0].trajectories)
                for name in [
                    "total_loss",
                    "policy_objective_loss",
                    "entropy",
                    "kl_penalty",
                    "ratio_mean",
                    "mean_advantage",
                    "advantage_std",
                ]:
                    assert metrics[name] == pytest.approx(
                        expected_metrics[name], rel=2e-5, abs=2e-6
                    )
                if not local:
                    assert trainer.agent.model.module.calls == calls
                if micro == 0:
                    path = folder / "saved"
                    trainer.save_runtime_checkpoint(str(path))
                    resumed = _make(
                        folder,
                        rank,
                        objective=objective,
                        entropy=entropy,
                        inner=inner,
                        optimizer=optimizer,
                    )
                    assert resumed.load_runtime_checkpoint(str(path))
                    for learner in (trainer, resumed):
                        before = copy.deepcopy(capture_payload(learner))
                        noop = asyncio.run(learner.training_step([]))
                        assert noop["global_empty_batch"] and not noop["optimizer_step"]
                        _same(capture_payload(learner), before, torch)
                else:
                    asyncio.run(resumed.training_step(local))
            for learner in (trainer, resumed, reference):
                learner._flush_accumulation(torch)
            wanted = reference.agent.model.lookup.weight
            torch.testing.assert_close(
                trainer.agent.model.module.lookup.weight, wanted, rtol=2e-5, atol=2e-6
            )
            _same(
                trainer.agent.model.state_dict(),
                resumed.agent.model.state_dict(),
                torch,
            )
            _same(trainer.optimizer.state_dict(), resumed.optimizer.state_dict(), torch)
            assert trainer.global_step == resumed.global_step == reference.global_step
            assert trainer.agent.model.module.unused.item() == torch.tensor(0.2).item()
            results.append(
                {
                    "objective": objective,
                    "entropy": entropy,
                    "inner": inner,
                    "pattern": pattern,
                    "optimizer": optimizer,
                    "combined_objective_equal": True,
                    "empty_peer_no_forward": True,
                    "global_empty_preserves_state": True,
                    "pending_resume_next_update_exact": True,
                    "unused_parameter_unchanged": True,
                }
            )

        # A resumed zero-gradient worker may flush before its next backward.
        trainer = _make(root / "scaler", rank, scaler=True)
        reference = _make(root / "scaler-serial", world=1, scaler=True)
        asyncio.run(trainer.training_step([_groups()[0]] if rank == 1 else []))
        asyncio.run(reference.training_step([_groups()[0]]))
        path = root / "scaler" / "saved"
        trainer.save_runtime_checkpoint(str(path))
        resumed = _make(root / "scaler", rank, scaler=True)
        assert resumed.load_runtime_checkpoint(str(path))
        for learner in (trainer, resumed, reference):
            learner._flush_accumulation(torch)
        _same(trainer.agent.model.state_dict(), resumed.agent.model.state_dict(), torch)
        torch.testing.assert_close(
            trainer.agent.model.module.lookup.weight,
            reference.agent.model.lookup.weight,
            rtol=1e-6,
            atol=1e-7,
        )
        assert (
            trainer.scaler.state_dict()
            == resumed.scaler.state_dict()
            == reference.scaler.state_dict()
        )
        results.append({"case": "scaled_empty_resume_flush", "equal": True})

        # A nonfinite backward remains GradScaler's skipped update, including
        # when only one peer has data and the overflow starts on that peer.
        for entropy in [0.0, 0.07]:
            trainer = _make(
                root / f"overflow-{entropy}", rank, scaler=True, entropy=entropy
            )
            trainer.config.gradient_accumulation_steps = 1
            before = copy.deepcopy(trainer.agent.model.state_dict())
            if rank == 1:
                trainer.agent.model.module.lookup.weight.register_hook(
                    lambda gradient: torch.full_like(gradient, float("inf"))
                )
            metrics = asyncio.run(
                trainer.training_step([_groups()[0]] if rank == 1 else [])
            )
            assert not metrics["optimizer_step"] and trainer.global_step == 0
            assert trainer.scaler.get_scale() == 64.0
            assert not trainer._ddp_restore_failed
            _same(trainer.agent.model.state_dict(), before, torch)
            results.append(
                {
                    "case": "scaled_empty_overflow",
                    "entropy": entropy,
                    "skipped_on_all_peers": True,
                }
            )
    finally:
        (root / f"empty-population-{rank}.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        dist.destroy_process_group()


@pytest.mark.timeout(120)
def test_empty_ddp_populations_match_combined_updates_and_resume(tmp_path):
    torch = pytest.importorskip("torch")
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_gloo_available()
    ):
        pytest.skip("Gloo required")
    context = torch.multiprocessing.spawn(
        _worker, args=(str(tmp_path.resolve()),), nprocs=2, join=False
    )
    try:
        deadline = time.monotonic() + 90
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("empty population contract exceeded 90 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        report = os.environ.get("STATESET_DISTRIBUTED_CONTRACT_REPORT_DIR")
        if report:
            Path(report).mkdir(parents=True, exist_ok=True)
            for p in tmp_path.glob("empty-population-*.json"):
                shutil.copy2(p, Path(report) / p.name)
    for rank in [0, 1]:
        assert (
            len(json.loads((tmp_path / f"empty-population-{rank}.json").read_text()))
            == 131
        )
