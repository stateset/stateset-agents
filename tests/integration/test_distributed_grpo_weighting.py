"""Real DDP updates must match the objective over all workers' trajectories."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import itertools
import json
import os
import sys
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.integration._spawn_budget import spawn_budget

pytestmark = pytest.mark.integration
SHARDS = 4


def _matrix_cases():
    cases = [
        (*case, 0.0)
        for case in itertools.product(
            ["grpo", "dr_grpo", "dapo", "cispo", "rloo", "gspo"],
            [0.0, 0.07],
            [1, 2, 4],
            [1, 16],
            [1, 2],
            [False, True],
        )
    ]
    cases.extend(
        (name, 0.07, steps, chunk, 1, unsynced, 0.2)
        for name in ["grpo", "dr_grpo", "dapo"]
        for steps, chunk, unsynced in [(2, 1, False), (4, 16, True)]
    )
    return cases


def _worker(rank, directory, shard):
    import torch
    import torch.distributed as dist

    from stateset_agents.training import grpo_accumulation
    from stateset_agents.training.config import TrainingConfig
    from stateset_agents.training.distributed import (
        DistributedConfig,
        DistributedTrainer,
    )
    from stateset_agents.training.grpo_accumulation import (
        _average_ddp_banks,
        _ddp_counts,
    )
    from stateset_agents.training.loss_computation import (
        compute_enhanced_grpo_loss,
        compute_grpo_loss,
        compute_token_old_logprobs,
    )
    from stateset_agents.training.metric_reduction import (
        ReducedMetrics,
        reduce_ddp_metrics,
    )
    from tests.unit.test_metric_reduction import _rows as metric_rows
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
    results = []

    class BufferedPolicy(TinyPolicy):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.ones(1))

        def forward(self, input_ids, attention_mask=None):
            output = super().forward(input_ids, attention_mask)
            return SimpleNamespace(logits=output.logits * self.scale)

    try:
        for index, case in enumerate(_matrix_cases()):
            if index % SHARDS != shard:
                continue
            objective, entropy, steps, chunk_size, inner, no_sync, beta = case
            config = TrainingConfig(
                objective=objective,
                entropy_coef=entropy,
                gradient_accumulation_steps=steps,
                generation_batch_size=chunk_size,
                max_completion_length=8,
                num_gradient_updates=inner,
                bf16=False,
                fp16=False,
                beta=beta,
                max_grad_norm=100.0,
                rollout_sync=True,
                zero_signal_patience=0,
                output_dir=str(root / f"rank-{rank}"),
                report_to="none",
            )
            trainer = DistributedTrainer(
                agent=SimpleNamespace(model=BufferedPolicy()),
                environment=SimpleNamespace(),
                config=config,
                distributed_config=DistributedConfig(
                    backend="gloo", world_size=2, rank=rank, strategy="ddp"
                ),
            )
            trainer.optimizer = torch.optim.SGD(
                trainer.agent.model.parameters(), lr=0.1
            )
            trainer._wrap_model_ddp()
            reference_policy = TinyPolicy(seed=13) if beta else None
            trainer.reference_model = reference_policy
            trainer.agent.model.module.scale.fill_(rank + 1)
            groups = _groups()
            microbatches = min(steps, 2) if inner == 1 else 1
            reference = TinyPolicy()
            if inner == 1:
                loss, _ = _reference(
                    groups * microbatches,
                    reference,
                    config,
                    beta=beta,
                    reference_model=reference_policy,
                )
                loss.backward()
                wanted = (
                    reference.lookup.weight.detach()
                    - 0.1 * reference.lookup.weight.grad
                )
                diagnostics = compute_grpo_loss(
                    groups,
                    config,
                    SimpleNamespace(model=TinyPolicy()),
                    0.0,
                    0,
                    lambda *_: None,
                )
                if beta:
                    diagnostics = compute_enhanced_grpo_loss(
                        groups,
                        beta,
                        config,
                        SimpleNamespace(model=TinyPolicy()),
                        reference_model=reference_policy,
                    )
            else:
                reference_agent = SimpleNamespace(model=reference)
                old = compute_token_old_logprobs(groups, config, reference_agent)
                optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
                for _ in range(inner):
                    loss_dict = compute_grpo_loss(
                        groups,
                        config,
                        reference_agent,
                        0.0,
                        0,
                        lambda *_: None,
                        old_logprobs=old,
                    )
                    loss_dict["total_loss"].backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                wanted = reference.lookup.weight.detach()
                diagnostics = loss_dict

            async def run(
                worker=trainer, batches=microbatches, inputs=groups, unsynced=no_sync
            ):
                observed = []
                for _ in range(batches):
                    with (
                        worker.agent.model.no_sync()
                        if unsynced
                        else contextlib.nullcontext()
                    ):
                        observed.append(
                            await worker.training_step([copy.deepcopy(inputs[rank])])
                        )
                worker._flush_accumulation(torch)
                return observed

            reported = asyncio.run(run())
            metric_errors = []
            for metrics in reported:
                for key in [
                    "total_loss",
                    "policy_loss",
                    "policy_objective_loss",
                    "entropy",
                    "kl_penalty",
                    "ratio_mean",
                    "clip_fraction",
                    "mean_advantage",
                    "advantage_std",
                    "nonzero_advantage_fraction",
                ]:
                    value = diagnostics[key]
                    expected_metric = float(
                        value.detach() if torch.is_tensor(value) else value
                    )
                    metric_errors.append(abs(metrics[key] - expected_metric))
                    assert metrics[key] == pytest.approx(
                        expected_metric, rel=1e-5, abs=2e-6
                    ), (key, metrics[key], expected_metric)
                assert metrics["num_trajectories"] == 5
                assert metrics["num_rows"] == 8 and metrics["num_tokens"] == 17
                assert (
                    metrics["policy_denominator"] == diagnostics["policy_denominator"]
                )
                assert metrics["metric_scope"] == "ddp_global_batch"
                assert metrics["entropy_measured"] is (entropy > 0)
                if inner > 1:
                    assert metrics["ratio_mean_last"] == metrics["ratio_mean"]
            actual = trainer.agent.model.module.lookup.weight.detach()
            error = (actual - wanted).abs().max().item()
            results.append(
                {
                    "matrix_index": index,
                    "objective": objective,
                    "entropy": entropy,
                    "steps": steps,
                    "chunk_size": chunk_size,
                    "inner_updates": inner,
                    "no_sync": no_sync,
                    "beta": beta,
                    "microbatches": microbatches,
                    "max_parameter_error": error,
                    "max_metric_error": max(metric_errors),
                    "optimizer_steps": trainer.global_step,
                }
            )
            (root / f"rank-{rank}.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            torch.testing.assert_close(actual, wanted, rtol=1e-6, atol=1e-7)
            assert trainer.global_step == inner
            assert trainer.agent.model.module.scale.item() == 1.0
            assert trainer.agent.model.broadcast_buffers is True
        if shard != 0:
            return
        metric_guards = []
        baseline = metric_rows()[rank]
        huge_step = 2**60 + 1
        for case in [
            "missing_key",
            "number_type",
            "metadata",
            "control",
            "mixed_native",
            "measure",
            "invalid_count",
            "invalid_stat",
            "nonfinite",
            "recovery",
        ]:
            local = ReducedMetrics(
                {
                    "total_loss": baseline.policy_sum / baseline.policy_count,
                    "entropy": baseline.entropy_sum / baseline.token_count,
                    "global_step": huge_step,
                    "ewc_penalty": 1.0 + 2 * rank,
                    "path": "token",
                },
                baseline,
            )
            if rank == 1:
                if case == "missing_key":
                    local.pop("entropy")
                elif case == "number_type":
                    local["entropy"] = int(local["entropy"])
                elif case == "metadata":
                    local["path"] = "DO_NOT_TRANSMIT"
                elif case == "control":
                    local["global_step"] = huge_step + 1
                elif case == "mixed_native":
                    local = dict(local)
                elif case == "measure":
                    local.reduction = replace(
                        baseline, measure="token", policy_count=baseline.token_count
                    )
                elif case == "invalid_count":
                    local.reduction = replace(baseline, policy_count=0)
                elif case == "invalid_stat":
                    local.reduction = replace(baseline, policy_sum=float("nan"))
                elif case == "nonfinite":
                    local["entropy"] = float("inf")
            try:
                reduced = reduce_ddp_metrics(local, trainer.agent.model)
            except ValueError as exc:
                assert case != "recovery" and "DO_NOT_TRANSMIT" not in str(exc)
                metric_guards.append(
                    {"case": case, "rejected": True, "error": str(exc)}
                )
            else:
                assert case == "recovery"
                assert reduced["total_loss"] == pytest.approx(7.25)
                assert reduced["entropy"] == 3.5
                assert reduced["ewc_penalty"] == pytest.approx(2.2)
                assert (
                    type(reduced["global_step"]) is int
                    and reduced["global_step"] == huge_step
                )
                metric_guards.append({"case": case, "rejected": False})
        legacy = reduce_ddp_metrics(
            {"loss": 2.0 + 2 * rank, "label": "legacy"}, trainer.agent.model
        )
        assert legacy == {"loss": 3.0, "label": "legacy"}
        (root / f"metric-guards-{rank}.json").write_text(
            json.dumps(metric_guards, indent=2) + "\n"
        )
        original_compute = trainer.compute_grpo_loss
        original_enhanced = trainer.compute_enhanced_grpo_loss

        def fail_compute(*_args, **_kwargs):
            raise ValueError("intentional local loss failure")

        trainer.compute_grpo_loss = fail_compute
        trainer.compute_enhanced_grpo_loss = fail_compute
        try:
            with pytest.raises(ValueError, match="intentional local loss failure"):
                asyncio.run(trainer.training_step([copy.deepcopy(groups[rank])]))
            assert trainer.agent.model.broadcast_buffers is True
        finally:
            trainer.compute_grpo_loss = original_compute
            trainer.compute_enhanced_grpo_loss = original_enhanced
        (root / f"buffer-sync-{rank}.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "buffer_value": trainer.agent.model.module.scale.item(),
                    "broadcast_preference_restored_after_error": True,
                },
                indent=2,
            )
            + "\n"
        )
        guards = []
        for case in [
            "measure",
            "entropy",
            "invalid_count",
            "overflow_count",
            "recovery",
        ]:
            measure = "token" if case == "measure" and rank == 1 else "trajectory"
            separate = not (case == "entropy" and rank == 1)
            policy = 2 + rank
            if rank == 1 and case in {"invalid_count", "overflow_count"}:
                policy = 0 if case == "invalid_count" else 2**70
            try:
                totals = _ddp_counts(
                    trainer,
                    torch,
                    policy,
                    5 + rank if separate else 0,
                    measure=measure,
                    separate_entropy=separate,
                )
            except ValueError as exc:
                assert case != "recovery"
                guards.append({"case": case, "rejected": True, "error": str(exc)})
            else:
                assert case == "recovery" and totals == (5, 11, 2)
                guards.append({"case": case, "rejected": False})
        # Two parameters/dtypes, a rank-local missing gradient, an entirely
        # unused parameter and a frozen parameter exercise both reduction banks.
        model = torch.nn.ParameterDict(
            {
                "first": torch.nn.Parameter(torch.ones(3, 2)),
                "second": torch.nn.Parameter(torch.ones(2, 3, dtype=torch.float64)),
                "unused": torch.nn.Parameter(torch.ones(1)),
                "frozen": torch.nn.Parameter(torch.ones(1), requires_grad=False),
                "large_float": torch.nn.Parameter(torch.ones(1)),
                "large_half": torch.nn.Parameter(torch.ones(1, dtype=torch.float16)),
                "small_half": torch.nn.Parameter(torch.ones(1, dtype=torch.float16)),
            }
        )
        ddp = torch.nn.parallel.DistributedDataParallel(model)
        bank = {name: None for name, _ in ddp.named_parameters()}
        for name, parameter in ddp.named_parameters():
            if "first" in name:
                parameter.grad = (
                    torch.full_like(parameter, rank + 1.0).t().contiguous().t()
                )
                bank[name] = torch.full_like(parameter, rank + 3.0)
            elif "second" in name and rank == 1:
                parameter.grad = torch.full_like(parameter, 4.0)
                bank[name] = torch.full_like(parameter, 8.0)
            elif "large_" in name or "small_" in name:
                value = (
                    0.75 * torch.finfo(parameter.dtype).max
                    if "large_" in name
                    else 2**-24
                )
                parameter.grad = torch.full_like(parameter, value)
                bank[name] = torch.full_like(parameter, value)
        original_reduce = dist.all_reduce
        original_limit = grpo_accumulation._DDP_BUCKET_BYTES
        payload_sizes = []

        def observed_reduce(tensor, *args, **kwargs):
            if tensor.is_floating_point():
                payload_sizes.append(tensor.numel() * tensor.element_size())
            return original_reduce(tensor, *args, **kwargs)

        dist.all_reduce = observed_reduce
        grpo_accumulation._DDP_BUCKET_BYTES = 16
        try:
            _average_ddp_banks(
                SimpleNamespace(agent=SimpleNamespace(model=ddp)),
                {"entropy_gradients": bank, "entropy_normalizer": 1},
                torch,
            )
        finally:
            dist.all_reduce = original_reduce
            grpo_accumulation._DDP_BUCKET_BYTES = original_limit
        assert len(payload_sizes) > 4 and max(payload_sizes) <= 16
        (root / f"bank-reduction-{rank}.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "test_payload_limit_bytes": 16,
                    "production_payload_limit_bytes": original_limit,
                    "observed_payload_bytes": payload_sizes,
                    "half_overflow_and_subnormal_checked": True,
                },
                indent=2,
            )
            + "\n"
        )
        for name, parameter in ddp.named_parameters():
            if "first" in name or "second" in name:
                torch.testing.assert_close(
                    parameter.grad,
                    torch.full_like(parameter, 1.5 if "first" in name else 2.0),
                )
                torch.testing.assert_close(
                    bank[name],
                    torch.full_like(parameter, 3.5 if "first" in name else 4.0),
                )
            elif "large_" in name or "small_" in name:
                expected = torch.full_like(
                    parameter,
                    (
                        0.75 * torch.finfo(parameter.dtype).max
                        if "large_" in name
                        else 2**-24
                    ),
                )
                assert torch.isfinite(parameter.grad).all()
                assert torch.isfinite(bank[name]).all()
                torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)
                torch.testing.assert_close(bank[name], expected, rtol=0, atol=0)
            else:
                assert parameter.grad is None and bank[name] is None
        (root / f"guards-{rank}.json").write_text(json.dumps(guards, indent=2) + "\n")
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.timeout(spawn_budget(120))
@pytest.mark.parametrize("shard", range(SHARDS), ids=lambda value: f"shard-{value}")
def test_two_rank_uneven_updates_match_combined_reference(tmp_path, shard):
    torch = pytest.importorskip("torch")
    dist = pytest.importorskip("torch.distributed")
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("requires the CPU Gloo backend")
    context = torch.multiprocessing.spawn(
        _worker, args=(str(tmp_path.resolve()), shard), nprocs=2, join=False
    )
    try:
        deadline = time.monotonic() + spawn_budget(90)
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail(
                    f"two-rank RL weighting shard {shard} exceeded {spawn_budget(90):.0f} seconds"
                )
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        destination = os.environ.get("STATESET_DISTRIBUTED_CONTRACT_REPORT_DIR")
        if destination:
            output = Path(destination) / "weighting" / f"shard-{shard}"
            output.mkdir(parents=True, exist_ok=True)
            for path in tmp_path.glob("*.json"):
                (output / path.name).write_bytes(path.read_bytes())
    first = json.loads((tmp_path / "rank-0.json").read_text())
    second = json.loads((tmp_path / "rank-1.json").read_text())
    assert len(_matrix_cases()) == 294
    expected_indices = list(range(shard, 294, SHARDS))
    assert [row["matrix_index"] for row in first] == expected_indices
    assert [row["matrix_index"] for row in second] == expected_indices
    for left, right in zip(first, second, strict=True):
        assert left["max_parameter_error"] == pytest.approx(
            right["max_parameter_error"], abs=1e-7
        )
    if shard != 0:
        return
    metric_guards = json.loads((tmp_path / "metric-guards-0.json").read_text())
    assert metric_guards == json.loads((tmp_path / "metric-guards-1.json").read_text())
    assert sum(row["rejected"] for row in metric_guards) == 9
    assert metric_guards[-1]["case"] == "recovery" and not metric_guards[-1]["rejected"]
    assert json.loads((tmp_path / "guards-0.json").read_text()) == json.loads(
        (tmp_path / "guards-1.json").read_text()
    )
    assert json.loads((tmp_path / "bank-reduction-0.json").read_text()) == json.loads(
        (tmp_path / "bank-reduction-1.json").read_text()
    )
    assert json.loads((tmp_path / "buffer-sync-0.json").read_text()) == json.loads(
        (tmp_path / "buffer-sync-1.json").read_text()
    )
