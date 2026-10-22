"""Two real workers resume rank-local windows and recover coordinated failures."""

from __future__ import annotations

import asyncio
import copy
import itertools
import json
import os
import random
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path

import pytest

from tests.integration._spawn_budget import spawn_budget

pytestmark = pytest.mark.integration
_ARTIFACT_FAULTS = [
    "rank",
    "generation",
    "topology",
    "shared",
    "gradient",
    "rng",
    "buffer",
]


def _copy_artifacts(source, target):
    for file in source.iterdir():
        if file.name != ".stateset-checkpoint.json":
            shutil.copy2(file, target / file.name)


def _tamper(path, backup, fault):
    import torch

    from stateset_agents.core.checkpoint_io import load_checkpoint_file
    from stateset_agents.training.checkpoint_publication import checkpoint_write

    with checkpoint_write(path) as staging:
        _copy_artifacts(backup, staging)
        metadata_file = staging / "native-ddp.json"
        if fault == "topology":
            metadata = json.loads(metadata_file.read_text())
            metadata["topology"] = [0, 2]
            metadata_file.write_text(json.dumps(metadata))
            return
        name = "training_state.pt" if fault == "shared" else "rank-00001.pt"
        packet = load_checkpoint_file(staging / name)
        if fault == "rank":
            packet["rank"] = 0
        elif fault == "generation":
            packet["generation"] = "0" * 32
        elif fault == "shared":
            packet["state"]["policy"]["lookup.weight"].add_(1)
        elif fault == "gradient":
            packet["state"]["gradient_accumulation"]["gradients"][
                "module.lookup.weight"
            ] = torch.zeros(1)
        elif fault == "rng":
            packet["state"]["rng"]["torch"] = torch.zeros(3, dtype=torch.uint8)
        elif fault == "buffer":
            packet["state"]["buffers"]["scratch"] = torch.zeros(2)
        torch.save(packet, staging / name)


def _cases():
    objectives = [None, "grpo", "dr_grpo", "dapo", "cispo", "rloo", "gspo"]
    return [
        (api, optimizer, objective, pending, inner)
        for api, optimizer, objective, inner in itertools.product(
            ["runtime", "async"], ["sgd", "adamw"], objectives, [1, 2]
        )
        for pending in ([False, True] if inner == 1 else [False])
    ]


def _make(
    root,
    rank,
    optimizer="sgd",
    steps=2,
    objective=None,
    inner=1,
    device="cpu",
    precision="fp32",
):
    import torch

    from tests.integration.test_replica_consensus import _make as make
    from tests.integration.test_replica_consensus import _scheduler

    trainer = make(root / f"rank-{rank}", rank, 2, device=device)
    if torch.device(device).type == "cuda":
        for model in (trainer.agent.model, trainer.reference_model):
            model.register_buffer(
                "precision_projection", torch.eye(16, device=device), persistent=False
            )
    if precision == "fp16":
        assert torch.device(device).type == "cuda"
        trainer.config.fp16 = True
        trainer.scaler = torch.amp.GradScaler("cuda", init_scale=128.0)
    trainer.config.gradient_accumulation_steps = steps
    trainer.config.objective = objective
    trainer.config.num_gradient_updates = inner
    trainer.distributed_config.checkpoint_dir = str(root / "checkpoints")
    trainer.agent.model.register_buffer(
        "scratch", torch.tensor([3.0], device=device), persistent=False
    )
    cls = torch.optim.SGD if optimizer == "sgd" else torch.optim.AdamW
    kwargs = {"momentum": 0.9} if optimizer == "sgd" else {}
    trainer.optimizer = cls(trainer.agent.model.parameters(), lr=0.1, **kwargs)
    trainer.lr_scheduler = _scheduler(trainer.optimizer, "lambda_closure", 0.9)
    trainer._wrap_model_ddp()
    trainer.agent.model.broadcast_buffers = False
    trainer.agent.model.module.local_buffer.fill_(rank + 10)
    trainer.agent.model.module.scratch.fill_(rank + 20)
    return trainer


def _save(trainer, path, api):
    if api == "runtime":
        trainer.save_runtime_checkpoint(str(path))
    else:
        trainer.distributed_config.checkpoint_dir = str(path.parent)
        asyncio.run(trainer.save_checkpoint(checkpoint_name=path.name))


def _load(trainer, path, api):
    return (
        trainer.load_runtime_checkpoint(str(path))
        if api == "runtime"
        else trainer.load_checkpoint(path)
    )


def _worker(rank, directory):
    import numpy as np
    import torch
    import torch.distributed as dist

    from stateset_agents.training.ddp_checkpoint_state import capture_payload
    from tests.integration.test_learner_preflight import _same, _snapshot
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
        for api, optimizer, objective, pending, inner in _cases():
            case = f"{api}-{optimizer}-{objective}-pending-{pending}-inner-{inner}"
            trainer = _make(root, rank, optimizer, objective=objective, inner=inner)
            groups = [_groups()[rank]]
            # First complete window creates real optimizer slots.
            for _ in range(2 + int(pending)):
                asyncio.run(trainer.training_step(groups))
            path = root / "checkpoints" / case
            _save(trainer, path, api)
            expected = copy.deepcopy(capture_payload(trainer))
            draw = (random.random(), np.random.rand(), torch.rand(3))
            resumed = _make(root, rank, optimizer, objective=objective, inner=inner)
            assert _load(resumed, path, api)
            actual_draw = (random.random(), np.random.rand(), torch.rand(3))
            _same(actual_draw, draw, torch)
            actual = capture_payload(resumed)
            _same(actual[0], expected[0], torch)
            for key in [
                "counters",
                "buffers",
                "gradient_accumulation",
                "learning_state",
            ]:
                _same(actual[1][key], expected[1][key], torch)
            assert actual[2] == expected[2]
            for learner in (trainer, resumed):
                asyncio.run(learner.training_step(groups))
                learner._flush_accumulation(torch)
            _same(
                trainer.optimizer.state_dict(),
                resumed.optimizer.state_dict(),
                torch,
            )
            _same(
                trainer.agent.model.state_dict(),
                resumed.agent.model.state_dict(),
                torch,
            )
            assert trainer.global_step == resumed.global_step
            results.append(
                {
                    "case": case,
                    "resumed": True,
                    "next_update_equal": True,
                    "rank_local_rng_equal": True,
                    "pending": pending,
                }
            )

        # Each rank participates even when only one rank experiences the fault.
        trainer = _make(root, rank)
        asyncio.run(trainer.training_step([_groups()[rank]]))
        path = root / "checkpoints" / "faults"
        _save(trainer, path, "runtime")
        marker = (path / ".stateset-checkpoint.json").read_bytes()
        for fault in ["write", "read", "validate", "implementation", "apply"]:
            before = _snapshot(trainer, torch)
            if fault == "write":
                original = torch.save

                def fail_save(*args, original=original, **kwargs):
                    if rank == 1:
                        raise OSError("injected rank-local write failure")
                    return original(*args, **kwargs)

                torch.save = fail_save
                try:
                    with pytest.raises(ValueError, match="rank-write"):
                        _save(trainer, path, "runtime")
                finally:
                    torch.save = original
                assert (path / ".stateset-checkpoint.json").read_bytes() == marker
                _same(_snapshot(trainer, torch), before, torch)
            elif fault == "read":
                from stateset_agents.training import ddp_checkpoint

                original = ddp_checkpoint.load_checkpoint_file

                def fail_read(*args, original=original, **kwargs):
                    if rank == 1:
                        raise OSError("injected rank-local read failure")
                    return original(*args, **kwargs)

                ddp_checkpoint.load_checkpoint_file = fail_read
                try:
                    with pytest.raises(ValueError, match="read"):
                        _load(trainer, path, "runtime")
                finally:
                    ddp_checkpoint.load_checkpoint_file = original
                _same(_snapshot(trainer, torch), before, torch)
            elif fault == "validate":
                # Admission accepts a same-class scheduler; validation must
                # still bind the checkpoint's future callable closure.
                from tests.integration.test_replica_consensus import _scheduler

                trainer.lr_scheduler = _scheduler(
                    trainer.optimizer, "lambda_closure", 0.5 if rank else 0.9
                )
                unchanged = _snapshot(trainer, torch)
                with pytest.raises(ValueError, match="validate"):
                    _load(trainer, path, "runtime")
                _same(_snapshot(trainer, torch), unchanged, torch)
                trainer.lr_scheduler = _scheduler(
                    trainer.optimizer, "lambda_closure", 0.9
                )
            elif fault == "implementation":
                # Both workers agree on the new code, but the saved revision
                # differs. Restore must reject before touching either learner.
                cls = type(trainer.agent.model.module)
                original = cls.forward
                cls.forward = lambda *args, **kwargs: None
                try:
                    with pytest.raises(ValueError, match="validate"):
                        _load(trainer, path, "runtime")
                    _same(_snapshot(trainer, torch), before, torch)
                finally:
                    cls.forward = original
            else:
                original = trainer.optimizer.load_state_dict

                def fail_apply(state, original=original):
                    if rank == 1:
                        raise RuntimeError("injected rank-local apply failure")
                    return original(state)

                trainer.optimizer.load_state_dict = fail_apply
                try:
                    with pytest.raises(ValueError, match="apply"):
                        _load(trainer, path, "runtime")
                finally:
                    trainer.optimizer.load_state_dict = original
                assert trainer._ddp_restore_failed is True
                with pytest.raises(ValueError, match="preflight"):
                    asyncio.run(trainer.training_step([_groups()[rank]]))
                with pytest.raises(ValueError, match="capture"):
                    _save(trainer, path, "runtime")
            assert _load(trainer, path, "runtime")
            assert trainer._ddp_restore_failed is False
            results.append(
                {"case": fault, "coordinated_rejection": True, "reloaded": True}
            )

        # A valid publication inventory alone must not admit a swapped shard,
        # another generation, or malformed local state. Re-sign each tampered
        # directory through the real publisher to exercise internal validation.
        backup = root / "checkpoint-backup"
        if rank == 0:
            shutil.copytree(path, backup)
        dist.barrier()
        for fault in _ARTIFACT_FAULTS:
            if rank == 0:
                _tamper(path, backup, fault)
            dist.barrier()
            unchanged = copy.deepcopy(capture_payload(trainer))
            with pytest.raises(ValueError, match="read|validate"):
                _load(trainer, path, "runtime")
            _same(capture_payload(trainer), unchanged, torch)
            dist.barrier()
            if rank == 0:
                from stateset_agents.training.checkpoint_publication import (
                    checkpoint_write,
                )

                with checkpoint_write(path) as staging:
                    _copy_artifacts(backup, staging)
            dist.barrier()
            assert _load(trainer, path, "runtime")
            results.append(
                {
                    "case": "artifact-" + fault,
                    "coordinated_rejection": True,
                    "state_preserved": True,
                    "reloaded": True,
                }
            )

        for fault in ["path", "operation", "shared-values"]:
            unchanged = copy.deepcopy(capture_payload(trainer))
            original = None
            if fault == "shared-values" and rank == 1:
                slot = next(iter(trainer.optimizer.state.values()), None)
                # This checkpoint predates its first optimizer update; change
                # policy values instead of inventing optimizer state.
                assert slot is None
                original = trainer.agent.model.module.lookup.weight.detach().clone()
                with torch.no_grad():
                    trainer.agent.model.module.lookup.weight.add_(1)
            try:
                with pytest.raises(ValueError, match="differs"):
                    if fault == "path":
                        _load(trainer, path if rank == 0 else root / "wrong", "runtime")
                    elif fault == "operation" and rank == 1:
                        _load(trainer, path, "runtime")
                    else:
                        _save(trainer, path, "runtime")
            finally:
                if original is not None:
                    with torch.no_grad():
                        trainer.agent.model.module.lookup.weight.copy_(original)
            _same(capture_payload(trainer), unchanged, torch)
            assert _load(trainer, path, "runtime")
            results.append(
                {"case": fault, "coordinated_rejection": True, "reloaded": True}
            )

        if rank == 0:
            files = {p.name for p in path.iterdir()}
            assert files == {
                "training_state.pt",
                "native-ddp.json",
                "rank-00000.pt",
                "rank-00001.pt",
                ".stateset-checkpoint.json",
            }
        asyncio.run(trainer.training_step([_groups()[rank]]))
        assert trainer.global_step == 1
    finally:
        (root / f"native-ddp-checkpoint-{rank}.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        dist.destroy_process_group()


@pytest.mark.timeout(spawn_budget(210))
def test_native_ddp_checkpoint_resume_and_coordinated_recovery(tmp_path):
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
        # 84 roundtrips now re-read source provenance at every admission and
        # restore. Keep a bounded harness timeout separate from numerical gates.
        deadline = time.monotonic() + spawn_budget(180)
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail(
                    f"native DDP checkpoint contract exceeded {spawn_budget(180):.0f} seconds"
                )
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        report_dir = os.environ.get("STATESET_DISTRIBUTED_CONTRACT_REPORT_DIR")
        if report_dir:
            destination = Path(report_dir)
            destination.mkdir(parents=True, exist_ok=True)
            for report in tmp_path.glob("native-ddp-checkpoint-*.json"):
                shutil.copy2(report, destination / report.name)
    for rank in [0, 1]:
        report = json.loads(
            (tmp_path / f"native-ddp-checkpoint-{rank}.json").read_text()
        )
        assert len(report) == len(_cases()) + 5 + len(_ARTIFACT_FAULTS) + 3
