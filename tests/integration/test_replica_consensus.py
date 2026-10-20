"""Native DDP startup binds completed checkpoints and future scheduler behavior."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import os
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.integration
GLOBAL_DECAY = 0.9
CASES = [
    "policy",
    "policy_nan",
    "optimizer",
    "optimizer_nan",
    "reference",
    "reference_nan",
    "lambda_closure",
    "lambda_partial",
    "lambda_global",
    "lambda_composite",
    "lambda_source_error",
    "asymmetric_refresh",
    "failed_legacy_restore",
    "failed_runtime_restore",
    "cache",
    "protocol_version",
]


def _closure(gamma):
    return lambda step: gamma**step


def _partial(step, *, gamma):
    return gamma**step


def _global(step):
    return GLOBAL_DECAY**step


def _scheduler(optimizer, case, gamma):
    import torch

    function = (
        functools.partial(_partial, gamma=gamma)
        if case == "lambda_partial"
        else _global if case == "lambda_global" else _closure(gamma)
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, function)
    if case == "lambda_composite":
        return torch.optim.lr_scheduler.ChainedScheduler(
            [scheduler], optimizer=optimizer
        )
    return scheduler


def _config(directory):
    from stateset_agents.training.config import TrainingConfig

    return TrainingConfig(
        objective=None,
        baseline_type="global_mean",
        advantage_normalization=False,
        gradient_accumulation_steps=1,
        num_gradient_updates=1,
        reward_clip=3.0,
        entropy_coef=0.07,
        beta=0.2,
        use_reference_model=True,
        generation_batch_size=1,
        bf16=False,
        fp16=False,
        max_grad_norm=100,
        zero_signal_patience=0,
        output_dir=str(directory),
        report_to="none",
    )


def _policy():
    import torch

    from tests.unit.test_token_group_aggregation import TinyPolicy

    class CountingPolicy(TinyPolicy):
        def __init__(self, seed=7):
            super().__init__(seed=seed)
            self.calls = 0
            self.config = SimpleNamespace(vocab_size=16)
            self.register_buffer("local_buffer", torch.ones(1))

        def forward(self, input_ids, attention_mask=None):
            self.calls += 1
            output = super().forward(input_ids, attention_mask)
            if hasattr(self, "precision_projection"):
                # CUDA contract only: embedding lookup alone is not autocast
                # eligible. Identity GEMM exercises the actual AMP kernel.
                output.logits = torch.nn.functional.linear(
                    output.logits, self.precision_projection
                )
                self._contract_last_output = output.logits.detach().clone()
            return output

        def save_pretrained(self, directory):
            torch.save(self.state_dict(), Path(directory) / "pytorch_model.bin")

    return CountingPolicy


def _make(directory, rank=0, world=1, device="cpu"):
    import torch

    from stateset_agents.training.distributed import (
        DistributedConfig,
        DistributedTrainer,
    )

    policy = _policy()
    trainer = DistributedTrainer(
        SimpleNamespace(model=policy().to(device), tokenizer=None),
        SimpleNamespace(),
        config=_config(directory),
        distributed_config=DistributedConfig(
            backend="nccl" if torch.device(device).type == "cuda" else "gloo",
            rank=rank,
            local_rank=torch.device(device).index or 0,
            world_size=world,
            strategy="ddp",
            checkpoint_parallel=False,
        ),
    )
    trainer.optimizer = torch.optim.SGD(
        trainer.agent.model.parameters(), lr=0.1, momentum=0.9
    )
    trainer.reference_model = policy(seed=13).to(device)
    return trainer


def _prepare(directory):
    from tests.unit.test_token_group_aggregation import _groups

    trainer = _make(directory / "serial")
    trainer._update_global_stats(0.5, 3)
    asyncio.run(trainer.training_step(_groups()))
    assert trainer.global_step == 1 and trainer.optimizer.state
    asyncio.run(trainer.save_checkpoint(checkpoint_name="legacy"))
    trainer.save_runtime_checkpoint(str(directory / "serial" / "runtime"))
    for rank in [0, 1]:
        for api in ["legacy", "runtime"]:
            shutil.copytree(
                directory / "serial" / api, directory / f"staged-{rank}" / api
            )


def _restore_tensor(tensor, saved):
    import torch

    with torch.no_grad():
        tensor.copy_(saved)


def _worker(rank, directory):
    import torch
    import torch.distributed as dist

    from stateset_agents.training import replica_consensus, scheduler_identity
    from tests.integration.test_learner_preflight import _same, _snapshot
    from tests.unit.test_token_group_aggregation import _groups

    global GLOBAL_DECAY
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
        for api in ["legacy", "runtime"]:
            for case in CASES:
                GLOBAL_DECAY = 0.9
                path = root / f"staged-{rank}" / api
                trainer = _make(root / f"rank-{rank}", rank, 2)
                reference = _make(root / f"reference-{rank}")
                for learner in (trainer, reference):
                    loaded = (
                        learner.load_checkpoint(path)
                        if api == "legacy"
                        else learner.load_runtime_checkpoint(str(path))
                    )
                    assert loaded and learner.optimizer.state
                    assert learner._ddp_replica_consensus_pending is True
                trainer._wrap_model_ddp()
                trainer.agent.model.module.local_buffer.fill_(rank + 10)
                groups = _groups()
                counts = [0]
                original_verify = replica_consensus.verify_ddp_replica_state

                def counted(learner, original=original_verify, counter=counts):
                    counter[0] += 1
                    return original(learner)

                with contextlib.ExitStack() as cleanup:
                    cleanup.callback(
                        setattr,
                        replica_consensus,
                        "verify_ddp_replica_state",
                        original_verify,
                    )
                    replica_consensus.verify_ddp_replica_state = counted
                    if case in {
                        "asymmetric_refresh",
                        "failed_legacy_restore",
                        "failed_runtime_restore",
                        "cache",
                    }:
                        asyncio.run(trainer.training_step([groups[rank]]))
                        asyncio.run(reference.training_step(groups))
                        assert (
                            counts[0] == 1
                            and trainer._ddp_replica_consensus_pending is False
                        )
                    trainer.agent.model.module.calls = 0
                    trainer.reference_model.calls = 0
                    if case.startswith("lambda_"):
                        if case == "lambda_global":
                            GLOBAL_DECAY = 0.9 if rank == 0 else 0.5
                        trainer.lr_scheduler = _scheduler(
                            trainer.optimizer, case, 0.9 if rank == 0 else 0.5
                        )
                    rejected = case not in {
                        "failed_legacy_restore",
                        "failed_runtime_restore",
                        "cache",
                    }
                    wire = []
                    original_gather = dist.all_gather

                    def captured(
                        outputs,
                        tensor,
                        *args,
                        original=original_gather,
                        observed=wire,
                        scenario=case,
                        worker_rank=rank,
                        **kwargs,
                    ):
                        if (
                            scenario == "protocol_version"
                            and worker_rank == 1
                            and tensor.numel() == 194
                        ):
                            tensor = tensor.clone()
                            tensor[0] = 1
                        observed.append(
                            {
                                "dtype": str(tensor.dtype),
                                "shape": list(tensor.shape),
                                "min": tensor.min().item(),
                                "max": tensor.max().item(),
                            }
                        )
                        return original(outputs, tensor, *args, **kwargs)

                    with contextlib.ExitStack() as repair:
                        if rank == 1:
                            if case == "lambda_source_error":
                                source_reader = scheduler_identity._source
                                repair.callback(
                                    setattr,
                                    scheduler_identity,
                                    "_source",
                                    source_reader,
                                )

                                def missing_source(_value):
                                    raise FileNotFoundError(
                                        "scheduler implementation source unavailable"
                                    )

                                scheduler_identity._source = missing_source
                            tensor = None
                            if case.startswith("policy"):
                                tensor = trainer.agent.model.module.lookup.weight
                            elif (
                                case.startswith("optimizer")
                                or case == "asymmetric_refresh"
                            ):
                                tensor = next(iter(trainer.optimizer.state.values()))[
                                    "momentum_buffer"
                                ]
                            elif case.startswith("reference"):
                                tensor = trainer.reference_model.lookup.weight
                            if tensor is not None:
                                repair.callback(
                                    _restore_tensor, tensor, tensor.detach().clone()
                                )
                                with torch.no_grad():
                                    tensor.reshape(-1)[0] = (
                                        float("nan")
                                        if case.endswith("nan")
                                        else tensor.reshape(-1)[0] + 1
                                    )
                            if case == "asymmetric_refresh":
                                trainer._ddp_replica_consensus_pending = True
                        # Wrapped DDP restore is now a collective operation.
                        # Both workers reject the missing checkpoint, then the
                        # next admission must recheck their unchanged replicas.
                        if case == "failed_legacy_restore":
                            with pytest.raises(ValueError):
                                trainer.load_checkpoint(root / "missing")
                            assert trainer._ddp_replica_consensus_pending is True
                        elif case == "failed_runtime_restore":
                            with pytest.raises(ValueError):
                                trainer.load_runtime_checkpoint(str(root / "missing"))
                            assert trainer._ddp_replica_consensus_pending is True
                        before = _snapshot(trainer, torch)
                        if rejected:
                            dist.all_gather = captured
                            try:
                                with pytest.raises(
                                    ValueError, match="replica|preflight"
                                ) as error:
                                    asyncio.run(trainer.training_step([groups[rank]]))
                            finally:
                                dist.all_gather = original_gather
                            _same(_snapshot(trainer, torch), before, torch)
                            assert (
                                trainer.agent.model.module.calls
                                == trainer.reference_model.calls
                                == 0
                            )
                            shapes = [row["shape"] for row in wire]
                            assert shapes == (
                                [[194]]
                                if case.startswith("lambda_")
                                or case == "protocol_version"
                                else [[194], [98]]
                            )
                            assert all(
                                row["dtype"] == "torch.int64"
                                and 0 <= row["min"] <= row["max"] <= 255
                                for row in wire
                            )
                            message = str(error.value)
                        else:
                            message = "accepted"
                    GLOBAL_DECAY = 0.9
                    if case.startswith("lambda_"):
                        trainer.lr_scheduler = _scheduler(trainer.optimizer, case, 0.9)
                        reference.lr_scheduler = _scheduler(
                            reference.optimizer, case, 0.9
                        )
                    asyncio.run(trainer.training_step([groups[rank]]))
                    asyncio.run(reference.training_step(groups))
                    trainer._flush_accumulation(torch)
                    reference._flush_accumulation(torch)
                    assert trainer._ddp_replica_consensus_pending is False
                    if case == "asymmetric_refresh":
                        expected_calls = 3
                    elif (
                        case == "cache"
                        or case.startswith("lambda_")
                        or case == "protocol_version"
                    ):
                        expected_calls = 1
                    else:
                        expected_calls = 2
                    assert counts[0] == expected_calls, (case, counts)
                    actual = trainer.agent.model.module.lookup.weight.detach()
                    wanted = reference.agent.model.lookup.weight.detach()
                    torch.testing.assert_close(actual, wanted, rtol=1e-6, atol=1e-7)
                    results.append(
                        {
                            "api": api,
                            "case": case,
                            "rejected": rejected,
                            "message": message,
                            "value_checks": counts[0],
                            "no_scoring_on_rejection": True,
                            "state_preserved_on_rejection": True,
                            "max_parameter_error": (actual - wanted).abs().max().item(),
                            "wire_shapes": [row["shape"] for row in wire],
                        }
                    )
                    (root / f"replica-consensus-{rank}.json").write_text(
                        json.dumps(results, indent=2) + "\n"
                    )
    finally:
        dist.destroy_process_group()


def test_checkpoint_value_consensus_and_scheduler_bindings(tmp_path):
    torch = pytest.importorskip("torch")
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_gloo_available()
    ):
        pytest.skip("Gloo required")
    torch.set_num_threads(1)
    _prepare(tmp_path)
    context = torch.multiprocessing.spawn(
        _worker, args=(str(tmp_path.resolve()),), nprocs=2, join=False
    )
    try:
        deadline = time.monotonic() + 90
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("native replica consensus exceeded 90 seconds")
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
    peers = [
        json.loads((tmp_path / f"replica-consensus-{rank}.json").read_text())
        for rank in [0, 1]
    ]
    assert len(peers[0]) == len(peers[1]) == 32
    assert [
        {k: v for k, v in row.items() if k != "max_parameter_error"} for row in peers[0]
    ] == [
        {k: v for k, v in row.items() if k != "max_parameter_error"} for row in peers[1]
    ]
