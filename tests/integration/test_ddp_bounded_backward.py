"""Two Gloo peers: the bounded token backward matches the retained graph."""

from __future__ import annotations

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

    from stateset_agents.training import loss_computation as lc
    from stateset_agents.training.ddp_population import token_forward_context
    from tests.unit.test_grpo_token_path import _config, _group, _tiny_model

    torch.set_num_threads(1)
    if sys.platform.startswith("linux"):
        os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    root = Path(directory)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=(root / "rendezvous").as_uri(),
        timeout=timedelta(seconds=25),
    )
    try:
        wrapper = torch.nn.parallel.DistributedDataParallel(_tiny_model())
        agent = SimpleNamespace(model=wrapper, tokenizer=None)
        # Ranks hold different rollouts, as they do in training.
        rewards = (1.0, 0.0, 0.5) if rank == 0 else (0.0, 1.0)
        groups = [_group(rewards=rewards, n_assistant=2 + rank)]

        def gradients(bounded):
            wrapper.zero_grad(set_to_none=True)
            config = _config(generation_batch_size=2, bounded_token_backward=bounded)
            with token_forward_context(wrapper, "test"):
                result = lc.compute_grpo_loss(
                    groups, config, agent, 0.0, 0, lambda mean, count: None
                )
                assert result["path"] == "token"
                result["total_loss"].backward()
            return float(result["total_loss"].detach()), {
                name: p.grad.clone()
                for name, p in wrapper.module.named_parameters()
                if p.grad is not None
            }

        loss, expected = gradients(False)
        bounded_loss, actual = gradients(True)
        report = {
            "rank": rank,
            "loss_matches": abs(loss - bounded_loss) < 1e-5,
            "same_parameters": sorted(actual) == sorted(expected) and bool(expected),
            "nonzero": any(bool(g.count_nonzero()) for g in expected.values()),
            "gradients_match": all(
                torch.allclose(actual[name], grad, rtol=1e-4, atol=1e-6)
                for name, grad in expected.items()
            ),
        }
        (root / f"bounded-{rank}.json").write_text(json.dumps(report))
    finally:
        dist.destroy_process_group()


@pytest.mark.timeout(120)
def test_bounded_backward_matches_on_two_gloo_ranks(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
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
                pytest.fail("bounded backward contract exceeded 90 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    for rank in (0, 1):
        report = json.loads((tmp_path / f"bounded-{rank}.json").read_text())
        assert report == {
            "rank": rank,
            "loss_matches": True,
            "same_parameters": True,
            "nonzero": True,
            "gradients_match": True,
        }
