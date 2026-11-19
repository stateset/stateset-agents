"""Exact subnormal-grid oracles for the native GRPO all-reduce and optimizer path."""

from __future__ import annotations

import itertools
import math
import os
import sys
from datetime import timedelta
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.integration


def _rounding_worker(rank: int, ranks: int, rendezvous: str) -> None:
    import torch
    import torch.distributed as dist

    from stateset_agents.training.ddp_population import token_forward_context
    from stateset_agents.training.grpo_accumulation import (
        NormalizedLoss,
        _average_ddp_banks,
        backward_accumulated_loss,
        finalize_accumulated_gradients,
    )

    if sys.platform.startswith("linux"):
        os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=ranks,
        timeout=timedelta(seconds=30),
    )
    try:
        grid = list(itertools.product(range(-4, 5), repeat=ranks))
        dtypes = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
        params = torch.nn.ParameterDict(
            {
                str(t).split(".")[1]
                + "_weight": torch.nn.Parameter(torch.zeros(len(grid) + 4, dtype=t))
                for t in dtypes
            }
        )
        model = torch.nn.parallel.DistributedDataParallel(params)
        trainer = SimpleNamespace(agent=SimpleNamespace(model=model))
        entropy = {}
        wanted = {}
        for name, p in model.named_parameters():
            info = torch.finfo(p.dtype)
            unit = info.tiny * info.eps
            large = info.max
            signs = [1, -1] if ranks == 2 else [1, 1, -1]
            extra = [
                large,
                -large,
                signs[rank] * large,
                float("inf") if rank == 0 else 0.0,
            ]
            p.grad = torch.tensor(
                [row[rank] * unit for row in grid] + extra, dtype=p.dtype
            )
            entropy[name] = -p.grad
            expected_extra = [
                large,
                -large,
                float(Fraction.from_float(large) * sum(signs) / ranks),
                math.inf,
            ]
            wanted[name] = torch.tensor(
                [round(Fraction(sum(row), ranks)) * unit for row in grid]
                + expected_extra,
                dtype=p.dtype,
            )
        _average_ddp_banks(
            trainer, {"entropy_gradients": entropy, "entropy_normalizer": 1}, torch
        )
        for name, p in model.named_parameters():
            # In the large cancelling three-rank case only the last rounding
            # bit may differ across Gloo's reduction orders.
            torch.testing.assert_close(p.grad[:-2], wanted[name][:-2], atol=0, rtol=0)
            torch.testing.assert_close(
                entropy[name][:-2], -wanted[name][:-2], atol=0, rtol=0
            )
            torch.testing.assert_close(
                p.grad[-2:], wanted[name][-2:], atol=0, rtol=torch.finfo(p.dtype).eps
            )
            torch.testing.assert_close(
                entropy[name][-2:],
                -wanted[name][-2:],
                atol=0,
                rtol=torch.finfo(p.dtype).eps,
            )

        # A restored window has rank-local counts and a common stored scale.
        # Cross two independently normalized banks before the destination cast.
        for name, p in model.named_parameters():
            unit = torch.finfo(p.dtype).tiny * torch.finfo(p.dtype).eps
            p.grad = torch.tensor(
                [row[rank] * unit for row in grid] + [0.0] * 4, dtype=p.dtype
            )
            entropy[name] = torch.tensor(
                [row[(rank + 1) % ranks] * unit for row in grid] + [0.0] * 4,
                dtype=p.dtype,
            )
            wanted[name] = torch.tensor(
                [
                    round(Fraction(sum(row), 2) + Fraction(sum(row), 4)) * unit
                    for row in grid
                ]
                + [0.0] * 4,
                dtype=p.dtype,
            )
        trainer._grpo_accumulation = {
            "measure": "trajectory",
            "policy_count": 2,
            "policy_normalizer": ranks,
            "entropy_count": 4,
            "entropy_normalizer": ranks,
            "entropy_gradients": entropy,
        }
        assert finalize_accumulated_gradients(trainer, torch)
        for name, p in model.named_parameters():
            torch.testing.assert_close(p.grad, wanted[name], atol=0, rtol=0)

        # A restored scale can put each weighted bank above the binary64
        # maximum, even though their combined update is representable.
        p = params["float64_weight"]
        p.grad.zero_()
        q = entropy["module.float64_weight"]
        q.zero_()
        large = torch.finfo(p.dtype).max * 0.75
        p.grad[0], q[0] = large, -large
        # Exact cancellation across ranks in one bank leaves a subnormal
        # from the other bank visible.
        p.grad[1] = (large, -large, 0.0)[rank]
        q[1] = math.ulp(0.0) if rank == 0 else 0.0
        trainer._grpo_accumulation.update(policy_count=1, entropy_count=1)
        finalize_accumulated_gradients(trainer, torch)
        assert p.grad[0].item() == 0 and p.grad[1].item() == math.ulp(0.0)

        # Backpropagate in the same local-forward context as DistributedTrainer,
        # then compare an actual SGD update to a single-process PyTorch update.
        for p in model.parameters():
            p.grad = None
        trainer._grpo_accumulation = None
        trainer._grad_accum_step = 0
        trainer._get_grad_accum_steps = lambda: 1
        p = params["float32_weight"]
        unit = torch.finfo(p.dtype).tiny * torch.finfo(p.dtype).eps
        directions = torch.tensor(
            [row[rank] * unit for row in grid] + [0.0] * 4, dtype=p.dtype
        )
        loss = (p * directions).sum()
        with token_forward_context(model, "test"):
            backward_accumulated_loss(
                trainer,
                loss,
                NormalizedLoss({}, policy_mean=loss, policy_count=1, measure="token"),
                torch,
            )
        finalize_accumulated_gradients(trainer, torch)
        torch.optim.SGD(model.parameters(), lr=1).step()
        reference = torch.nn.Parameter(torch.zeros_like(p))
        batch = (
            torch.tensor(
                [list(row) for row in grid] + [[0] * ranks] * 4, dtype=torch.float64
            )
            * unit
        )
        (reference.double() * batch.mean(dim=1)).sum().backward()
        torch.optim.SGD([reference], lr=1).step()
        torch.testing.assert_close(p, reference, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("ranks", [2, 3])
def test_ddp_bank_rounding_against_rational_oracle(tmp_path: Path, ranks: int) -> None:
    torch = pytest.importorskip("torch")
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_gloo_available()
    ):
        pytest.skip("requires the CPU Gloo backend")
    torch.multiprocessing.spawn(
        _rounding_worker,
        args=(ranks, (tmp_path / "gloo").as_uri()),
        nprocs=ranks,
    )


def test_conjugate_complex_gradients_keep_both_components() -> None:
    torch = pytest.importorskip("torch")
    from stateset_agents.training.grpo_accumulation import (
        finalize_accumulated_gradients,
    )

    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.complex128)
    policy = torch.tensor([[1 + 2j, 3 - 5j]], dtype=torch.complex128).conj()
    entropy = torch.tensor([[5 - 4j, -1 + 3j]], dtype=torch.complex128).conj()
    expected = policy / 2 + entropy / 4
    model.weight.grad = policy
    trainer = SimpleNamespace(
        agent=SimpleNamespace(model=model),
        _grpo_accumulation={
            "measure": "trajectory",
            "policy_count": 2,
            "policy_normalizer": 1,
            "entropy_count": 4,
            "entropy_normalizer": 1,
            "entropy_gradients": {"weight": entropy},
        },
    )
    assert finalize_accumulated_gradients(trainer, torch)
    torch.testing.assert_close(model.weight.grad, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dense_policy", [False, True])
def test_local_sparse_embedding_with_overlapping_banks(dense_policy: bool) -> None:
    torch = pytest.importorskip("torch")
    from stateset_agents.training.grpo_accumulation import (
        finalize_accumulated_gradients,
    )

    model = torch.nn.Embedding(8, 2, sparse=True)
    model(torch.tensor([2, 2, 5])).sum().backward()
    policy = model.weight.grad
    model.weight.grad = None
    (-model(torch.tensor([2, 6])).sum()).backward()
    entropy = model.weight.grad
    model.weight.grad = policy.to_dense() if dense_policy else policy
    trainer = SimpleNamespace(
        agent=SimpleNamespace(model=model),
        _grpo_accumulation={
            "measure": "trajectory",
            "policy_count": 3,
            "policy_normalizer": 1,
            "entropy_count": 5,
            "entropy_normalizer": 1,
            "entropy_gradients": {"weight": entropy},
        },
    )
    assert finalize_accumulated_gradients(trainer, torch)
    grad = model.weight.grad
    assert grad.is_sparse == (not dense_policy)
    expected = (
        policy.to_dense().double() / 3 + entropy.to_dense().double() / 5
    ).float()
    torch.testing.assert_close(grad.to_dense(), expected, atol=0, rtol=0)
