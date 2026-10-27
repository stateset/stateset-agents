"""Two actual DDP peers reject unsafe admission before any learner side effect."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import math
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.integration._spawn_budget import spawn_budget

pytestmark = pytest.mark.integration

CASES = [
    "inner_schedule",
    "accumulation_schedule",
    "objective",
    "clip",
    "entropy",
    "dropout_setting",
    "beta",
    "normalization",
    "global_step",
    "grad_counter",
    "learning_rate",
    "optimizer_type",
    "broadcast_preference",
    "model_buffer_layout",
    "trainable_layout",
    "reference_presence",
    "baseline_mean",
    "baseline_sum",
    "sampler_source_config",
    "sampler_path",
    "sampler_required",
    "missing_metadata",
    "mixed_compatibility_path",
    "negative_id",
    "boolean_id",
    "input_vocabulary",
    "output_vocabulary",
    "sampler_length",
    "sampler_positive",
    "sampler_nan",
    "policy_version",
    "metadata_type",
    "reward_nan",
    "reward_inf_clipped",
    "reward_overflow",
    "invalid_group",
    "pending_presence",
    "pending_normalizer",
    "pending_counter",
    "pending_negative_count",
    "pending_measure_switch",
    "policy_implementation",
    "reference_implementation",
    "submodule_implementation",
    "optimizer_implementation",
    "runtime_precision",
    "package_revision",
    "implementation_unreadable",
]


def _same(actual, expected, torch):
    if torch.is_tensor(actual):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _same(actual[key], expected[key], torch)
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            _same(a, b, torch)
    elif isinstance(actual, float) and math.isnan(actual):
        assert math.isnan(expected)
    else:
        assert actual == expected


def _snapshot(trainer, torch):
    model = trainer.agent.model
    return copy.deepcopy(
        {
            "model": model.module.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "gradients": [p.grad for p in model.parameters()],
            "banks": trainer._grpo_accumulation,
            "baseline": [
                trainer._global_reward_mean,
                trainer._global_reward_count,
                trainer._global_reward_sum,
            ],
            "counters": [trainer.global_step, trainer._grad_accum_step],
            "calls": model.module.calls,
            "reference_calls": (
                trainer.reference_model.calls
                if trainer.reference_model is not None
                else None
            ),
            "broadcast_buffers": model.broadcast_buffers,
        }
    )


def _worker(rank, directory):
    import torch
    import torch.distributed as dist

    from stateset_agents.training.config import TrainingConfig
    from stateset_agents.training.distributed import (
        DistributedConfig,
        DistributedTrainer,
    )
    from stateset_agents.training.learner_preflight import verify_ddp_learner_batch
    from tests.unit.test_token_group_aggregation import TinyPolicy, _groups

    class CountingPolicy(TinyPolicy):
        def __init__(self, seed=7):
            super().__init__(seed=seed)
            self.config = SimpleNamespace(vocab_size=16)
            self.calls = 0
            self.register_buffer("sentinel", torch.tensor(1.0))

        def get_input_embeddings(self):
            return self.lookup

        def get_output_embeddings(self):
            return self.lookup

        def forward(self, input_ids, attention_mask=None):
            self.calls += 1
            return super().forward(input_ids, attention_mask)

    def make(case):
        config = TrainingConfig(
            objective=None,
            baseline_type="global_mean",
            advantage_normalization=False,
            gradient_accumulation_steps=4 if case.startswith("pending_") else 2,
            num_gradient_updates=(
                1 if case.startswith("pending_") or case.startswith("sampler_") else 2
            ),
            reward_clip=3.0,
            entropy_coef=0.07,
            beta=0.2,
            generation_batch_size=1,
            bf16=False,
            fp16=False,
            max_grad_norm=100.0,
            use_reference_model=True,
            zero_signal_patience=0,
            output_dir=str(Path(directory) / f"rank-{rank}"),
            report_to="none",
        )
        config.allow_sequence_fallback = case == "mixed_compatibility_path"
        config.old_logprobs_source = (
            "sampler" if case in {"sampler_path", "sampler_required"} else "recompute"
        )
        config.allow_missing_sampler_logprobs = case == "sampler_path"
        trainer = DistributedTrainer(
            SimpleNamespace(model=CountingPolicy()),
            SimpleNamespace(),
            config=config,
            distributed_config=DistributedConfig(
                backend="gloo", rank=rank, world_size=2, strategy="ddp"
            ),
        )
        trainer.optimizer = torch.optim.SGD(
            trainer.agent.model.parameters(), lr=0.1, momentum=0.9
        )
        trainer.reference_model = CountingPolicy(seed=13)
        trainer._update_global_stats(0.5, 3)
        trainer._wrap_model_ddp()
        groups = _groups()
        if case in {"sampler_path", "sampler_required"}:
            for group in groups:
                for trajectory in group.trajectories:
                    for turn in trajectory.turns:
                        if turn.role == "assistant":
                            turn.metadata["sampler_log_probs"] = [-2.0] * len(
                                turn.metadata["token_ids"]
                            )
        # Populate real momentum tensors before testing admission rejection.
        # Flush the completed warmup regime before restoring its target schedule.
        target_steps = trainer.config.gradient_accumulation_steps
        target_inner = trainer.config.num_gradient_updates
        trainer.config.gradient_accumulation_steps = 1
        trainer.config.num_gradient_updates = 1
        asyncio.run(trainer.training_step([copy.deepcopy(groups[rank])]))
        trainer._flush_accumulation(torch)
        assert trainer.optimizer.state
        assert all(
            "momentum_buffer" in state for state in trainer.optimizer.state.values()
        )
        trainer.config.gradient_accumulation_steps = target_steps
        trainer.config.num_gradient_updates = target_inner
        if case.startswith("pending_"):
            asyncio.run(trainer.training_step([copy.deepcopy(groups[rank])]))
            assert trainer._grpo_accumulation is not None
            if case != "pending_measure_switch":
                trainer.config.num_gradient_updates = 2
        trainer.agent.model.module.calls = 0
        trainer.reference_model.calls = 0
        trainer.agent.model.module.sentinel.fill_(rank + 10.0)
        return trainer, groups

    torch.set_num_threads(1)
    if sys.platform.startswith("linux"):
        os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    root = Path(directory)
    results = []

    def progress(phase, case=None):
        # Keep the last completed stage even when native code aborts without a
        # Python exception. Replace atomically so readers never see partial JSON.
        pending = root / f"learner-progress-{rank}.tmp"
        pending.write_text(
            json.dumps(
                {
                    "rank": rank,
                    "pid": os.getpid(),
                    "torch_version": torch.__version__,
                    "phase": phase,
                    "case": case,
                    "completed_cases": [row["case"] for row in results],
                },
                indent=2,
            )
            + "\n"
        )
        pending.replace(root / f"learner-progress-{rank}.json")

    progress("initializing_group")
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=(root / "rendezvous").as_uri(),
        timeout=timedelta(seconds=30),
    )
    try:
        for case in CASES:
            progress("constructing_trainer", case)
            trainer, clean_groups = make(case)
            groups = copy.deepcopy(clean_groups)
            metadata = groups[rank].trajectories[0].turns[0].metadata
            original_config = copy.deepcopy(trainer.config)
            original_optimizer = trainer.optimizer
            optimizer_state = copy.deepcopy(original_optimizer.state_dict())
            with contextlib.ExitStack() as restore:
                restore.callback(setattr, trainer, "config", original_config)
                restore.callback(original_optimizer.load_state_dict, optimizer_state)

                def change(owner, name, value, cleanup=restore):
                    cleanup.callback(setattr, owner, name, getattr(owner, name))
                    setattr(owner, name, value)

                if case == "pending_measure_switch":
                    trainer.config.objective = "dapo"
                if rank == 1:
                    if case in {
                        "policy_implementation",
                        "submodule_implementation",
                        "optimizer_implementation",
                    }:
                        owner, method = {
                            "policy_implementation": (CountingPolicy, "forward"),
                            "submodule_implementation": (torch.nn.Embedding, "forward"),
                            "optimizer_implementation": (torch.optim.SGD, "step"),
                        }[case]

                        def forbidden(*args, **kwargs):
                            raise AssertionError("divergent implementation executed")

                        change(owner, method, forbidden)
                    elif case == "reference_implementation":
                        restore.callback(delattr, trainer.reference_model, "forward")
                        trainer.reference_model.forward = lambda *args, **kwargs: None
                    elif case == "runtime_precision":
                        precision = torch.get_float32_matmul_precision()
                        restore.callback(torch.set_float32_matmul_precision, precision)
                        torch.set_float32_matmul_precision(
                            "high" if precision == "highest" else "highest"
                        )
                    elif case == "package_revision":
                        from stateset_agents.training import implementation_identity

                        change(
                            implementation_identity,
                            "package_revision",
                            lambda: "0" * 64,
                        )
                    elif case == "implementation_unreadable":
                        from stateset_agents.training import implementation_identity

                        def unreadable(value):
                            raise OSError("implementation source unavailable")

                        change(implementation_identity, "_source", unreadable)
                    elif case == "inner_schedule":
                        trainer.config.num_gradient_updates = 3
                    elif case == "accumulation_schedule":
                        trainer.config.gradient_accumulation_steps = 3
                    elif case == "objective":
                        trainer.config.objective_overrides = {"clip_low": 0.05}
                    elif case == "clip":
                        trainer.config.reward_clip = 2.0
                    elif case == "entropy":
                        trainer.config.entropy_coef = 0.1
                    elif case == "dropout_setting":
                        trainer.config.disable_dropout = True
                    elif case == "beta":
                        trainer.config.beta = 0.3
                    elif case == "normalization":
                        trainer.config.advantage_normalization = True
                    elif case == "global_step":
                        change(trainer, "global_step", 2**60 + 1)
                    elif case == "grad_counter":
                        change(trainer, "_grad_accum_step", 1)
                    elif case == "learning_rate":
                        trainer.optimizer.param_groups[0]["lr"] = 0.125
                    elif case == "optimizer_type":
                        change(
                            trainer,
                            "optimizer",
                            torch.optim.Adam(trainer.agent.model.parameters(), lr=0.1),
                        )
                    elif case == "broadcast_preference":
                        change(trainer.agent.model, "broadcast_buffers", False)
                    elif case == "model_buffer_layout":
                        trainer.agent.model.module.register_buffer(
                            "extra", torch.ones(1)
                        )
                        restore.callback(delattr, trainer.agent.model.module, "extra")
                    elif case == "trainable_layout":
                        parameter = trainer.agent.model.module.lookup.weight
                        restore.callback(parameter.requires_grad_, True)
                        parameter.requires_grad_(False)
                    elif case == "reference_presence":
                        change(trainer, "reference_model", None)
                    elif case == "baseline_mean":
                        change(trainer, "_global_reward_mean", 0.75)
                        change(trainer, "_global_reward_sum", 2.25)
                    elif case == "baseline_sum":
                        change(trainer, "_global_reward_sum", float("nan"))
                    elif case == "sampler_source_config":
                        trainer.config.old_logprobs_source = "unknown"
                    elif case in {"sampler_path", "sampler_required"}:
                        metadata.pop("sampler_log_probs")
                    elif case in {"missing_metadata", "mixed_compatibility_path"}:
                        metadata.clear()
                    elif case == "negative_id":
                        metadata["token_ids"][0] = -1
                    elif case == "boolean_id":
                        metadata["token_ids"][0] = True
                    elif case == "input_vocabulary":
                        metadata["prompt_token_ids"][0] = 16
                    elif case == "output_vocabulary":
                        metadata["token_ids"][0] = 16
                    elif case == "sampler_length":
                        metadata["sampler_log_probs"] = [-2.0] * (
                            len(metadata["token_ids"]) + 1
                        )
                    elif case == "sampler_positive":
                        metadata["sampler_log_probs"] = [0.1] * len(
                            metadata["token_ids"]
                        )
                    elif case == "sampler_nan":
                        metadata["sampler_log_probs"] = [float("nan")] * len(
                            metadata["token_ids"]
                        )
                    elif case == "policy_version":
                        metadata["policy_version"] = -1
                    elif case == "metadata_type":
                        groups[rank].trajectories[0].turns[0].metadata = ["invalid"]
                    elif case == "reward_nan":
                        groups[rank].trajectories[0].total_reward = float("nan")
                    elif case == "reward_inf_clipped":
                        groups[rank].trajectories[0].total_reward = float("inf")
                    elif case == "reward_overflow":
                        groups[rank].trajectories[0].total_reward = 1e50
                        trainer.config.reward_clip = None
                    elif case == "invalid_group":
                        groups[rank] = None
                    elif case == "pending_presence":
                        change(trainer, "_grpo_accumulation", None)
                    elif case == "pending_counter":
                        change(trainer, "_grad_accum_step", 2)
                    elif case in {"pending_normalizer", "pending_negative_count"}:
                        pending = copy.deepcopy(trainer._grpo_accumulation)
                        if case == "pending_normalizer":
                            pending["policy_normalizer"] += 1
                        else:
                            pending["policy_count"] = -1
                        change(trainer, "_grpo_accumulation", pending)
                local = [groups[rank]] if groups else []
                before = _snapshot(trainer, torch)
                wire = []
                original_gather = dist.all_gather

                def capture(
                    outputs,
                    tensor,
                    *args,
                    original=original_gather,
                    observed=wire,
                    **kwargs,
                ):
                    observed.append(
                        {
                            "dtype": str(tensor.dtype),
                            "shape": list(tensor.shape),
                            "min": tensor.min().item(),
                            "max": tensor.max().item(),
                        }
                    )
                    return original(outputs, tensor, *args, **kwargs)

                restore.callback(setattr, dist, "all_gather", original_gather)
                dist.all_gather = capture
                progress("rejecting_unsafe_batch", case)
                with pytest.raises(ValueError, match="preflight") as error:
                    asyncio.run(trainer.training_step(local))
                assert len(wire) == 1
                assert wire[0]["dtype"] == "torch.int64" and wire[0]["shape"] == [194]
                assert 0 <= wire[0]["min"] <= wire[0]["max"] <= 255
                _same(_snapshot(trainer, torch), before, torch)
                assert trainer.agent.model.module.calls == 0
                if trainer.reference_model is not None:
                    assert trainer.reference_model.calls == 0
                message = str(error.value)
            progress("recovering_safe_batch", case)
            metrics = asyncio.run(
                trainer.training_step([copy.deepcopy(clean_groups[rank])])
            )
            trainer._flush_accumulation(torch)
            assert trainer.agent.model.module.calls > 0
            assert metrics["num_trajectories"] == 5
            parameters = trainer.agent.model.module.lookup.weight.detach()
            peers = [torch.empty_like(parameters) for _ in range(2)]
            dist.all_gather(peers, parameters)
            torch.testing.assert_close(peers[0], peers[1], rtol=0, atol=0)
            results.append(
                {
                    "case": case,
                    "rejected": True,
                    "side_effect_free": True,
                    "nonempty_optimizer_state": True,
                    "wire": wire,
                    "recovered": True,
                    "message": message,
                    "optimizer_steps_after_recovery": trainer.global_step,
                }
            )
            (root / f"learner-preflight-{rank}.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            progress("case_complete", case)
        # Exact counters and intentionally unequal chunk preferences are valid.
        progress("accepting_uneven_chunks", "uneven_chunks")
        trainer, groups = make("uneven_chunks")
        trainer.global_step = 2**60 + 1
        trainer.config.generation_batch_size = 1 if rank == 0 else 16
        receipt = verify_ddp_learner_batch(trainer, [groups[rank]])
        assert trainer.agent.model.module.calls == 0
        metrics = asyncio.run(trainer.training_step([groups[rank]]))
        assert metrics["global_step"] == 2**60 + 3
        assert type(metrics["global_step"]) is int
        (root / f"learner-accepted-{rank}.json").write_text(
            json.dumps(
                {
                    "receipt": receipt,
                    "global_step": metrics["global_step"],
                    "calls": trainer.agent.model.module.calls,
                },
                indent=2,
            )
            + "\n"
        )
        # Neither peer should destroy shared communication state while the
        # other still checks the final update or writes its acceptance receipt.
        progress("waiting_for_peer", "uneven_chunks")
        dist.barrier()
        progress("closing_group", "uneven_chunks")
    finally:
        dist.destroy_process_group()
    progress("complete", "uneven_chunks")


@pytest.mark.timeout(spawn_budget(120))
def test_real_ddp_admission_rejects_before_forward_and_recovers(tmp_path):
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
        deadline = time.monotonic() + spawn_budget(90)
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail(
                    f"native learner preflight exceeded {spawn_budget(90):.0f} seconds"
                )
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
    first = json.loads((tmp_path / "learner-preflight-0.json").read_text())
    second = json.loads((tmp_path / "learner-preflight-1.json").read_text())
    assert len(first) == len(second) == len(CASES) == 48
    # Different settings intentionally produce different digest bytes. Wire
    # shape/type/range are checked per worker; outcomes and errors must agree.
    assert [{k: v for k, v in row.items() if k != "wire"} for row in first] == [
        {k: v for k, v in row.items() if k != "wire"} for row in second
    ]
    accepted = [
        json.loads((tmp_path / f"learner-accepted-{rank}.json").read_text())
        for rank in [0, 1]
    ]
    assert accepted[0]["receipt"] == accepted[1]["receipt"]
    assert accepted[0]["global_step"] == accepted[1]["global_step"] == 2**60 + 3
    assert accepted[0]["calls"] != accepted[1]["calls"]
    for rank in (0, 1):
        journal = json.loads((tmp_path / f"learner-progress-{rank}.json").read_text())
        assert journal["rank"] == rank
        assert journal["phase"] == "complete"
        assert journal["completed_cases"] == CASES
