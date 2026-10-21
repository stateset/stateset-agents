"""Real DDP workers evaluate uneven partitions and enter the same best save."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import random
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.integration
CASES = [
    "uneven",
    "nonimprovement",
    "tie",
    "empty_peer",
    "empty_all",
    "error",
    "cancel",
    "policy",
    "seed",
    "counter",
    "metric",
    "save_error",
    "buffer_error",
    "operation",
]


def _state(trainer):
    from stateset_agents.training.ddp_checkpoint_state import capture_payload

    result = copy.deepcopy(capture_payload(trainer))
    # Successful evaluation intentionally changes only checkpoint selection.
    result[1]["counters"].pop("best_eval_metric")
    result[1]["counters"].pop("steps_without_improvement")
    return result


def _worker(rank, directory):
    import numpy as np
    import torch
    import torch.distributed as dist

    from stateset_agents.core.environment import Environment
    from tests.integration.test_learner_preflight import _same
    from tests.integration.test_native_ddp_checkpoint import _make
    from tests.unit.test_token_group_aggregation import _groups

    class LocalEnvironment(Environment):
        async def reset(self, scenario=None):
            raise NotImplementedError

        async def step(self, state, action):
            raise NotImplementedError

        async def run_episode(self, agent_fn, scenario=None, max_turns=None):
            await agent_fn([], scenario)
            if scenario.get("failure") == "error":
                raise OSError("injected local evaluation failure")
            if scenario.get("failure") == "cancel":
                raise asyncio.CancelledError("injected local evaluation cancellation")
            return SimpleNamespace(
                total_reward=scenario["reward"],
                episode_length=scenario["length"],
                turns=[],
            )

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
        for broadcast in [False, True]:
            for case in CASES:
                folder = root / f"{case}-buffers-{broadcast}"
                trainer = _make(folder, rank)
                trainer.config.eval_num_generations = 1
                trainer.config.rollout_concurrency = 1
                trainer.agent.model.broadcast_buffers = broadcast
                wrapper = trainer.agent.model
                wrapper.module.config.use_cache = False
                wrapper.module.train()
                wrapper.module.lookup.eval()
                modes = [m.training for m in wrapper.modules()]
                trainer.environment = LocalEnvironment()
                observed_buffers = []

                async def generate(
                    history,
                    context,
                    learner=trainer,
                    original=wrapper,
                    observed=observed_buffers,
                ):
                    assert learner.agent.model is original.module
                    assert not torch.is_grad_enabled()
                    assert not learner.agent.model.training
                    assert learner.agent.model.config.use_cache is True
                    observed.append(learner.agent.model.local_buffer.item())
                    random.random(), np.random.rand(), torch.rand(1)
                    learner.agent.model(torch.ones(1, 2, dtype=torch.long))
                    # A custom evaluator's buffer writes must not leak into
                    # pending training state, including on local failure.
                    learner.agent.model.scratch.add_(1)
                    return "done"

                trainer.agent.generate_response = generate
                asyncio.run(trainer.training_step([_groups()[rank]]))
                wrapper.module.lookup.eval()
                modes = [m.training for m in wrapper.modules()]
                assert (
                    trainer._grad_accum_step == 1
                    and trainer._grpo_accumulation is not None
                )
                trainer.best_eval_metric = 0.6
                trainer.steps_without_improvement = 2
                scenarios = [{"reward": float(rank), "length": rank + 1}] * (
                    1 if rank == 0 else 3
                )
                if case == "nonimprovement":
                    trainer.best_eval_metric = 1.0
                elif case == "tie":
                    trainer.best_eval_metric = 0.75
                elif case == "empty_peer" and rank == 0 or case == "empty_all":
                    scenarios = []
                if case in {"error", "cancel"} and rank == 1:
                    scenarios = [{"reward": 1.0, "length": 1, "failure": case}]
                before = _state(trainer)
                selection = (
                    trainer.best_eval_metric,
                    trainer.steps_without_improvement,
                )
                expected_reward = 1.0 if case == "empty_peer" else 0.75
                path = folder / "checkpoints" / "best-checkpoint"
                caught = None
                original_local = trainer._evaluate_local
                original_save = torch.save
                original_copy = torch.Tensor.copy_
                saved_policy = None
                if case == "policy" and rank == 1:
                    saved_policy = wrapper.module.lookup.weight.detach().clone()
                    with torch.no_grad():
                        wrapper.module.lookup.weight.add_(1)
                if case == "seed" and rank == 1:
                    trainer.config.seed += 1
                if case == "counter" and rank == 1:
                    trainer.best_eval_metric += 0.1
                if case == "metric":

                    async def malformed(scenarios, episodes, original=original_local):
                        metrics = await original(scenarios, episodes)
                        if rank == 1:
                            metrics["eval_num_episodes"] += 0.5
                        return metrics

                    trainer._evaluate_local = malformed
                if case == "save_error":
                    first = asyncio.run(trainer.evaluate(scenarios))
                    assert first["eval_reward"] == expected_reward
                    before, selection = _state(trainer), (
                        trainer.best_eval_metric,
                        trainer.steps_without_improvement,
                    )
                    marker = (path / ".stateset-checkpoint.json").read_bytes()
                    scenarios = [{"reward": 1.0, "length": 2}]

                    def failed_save(*args, original=original_save, **kwargs):
                        if rank == 0:
                            raise OSError("injected best-checkpoint save failure")
                        return original(*args, **kwargs)

                    torch.save = failed_save
                if case == "buffer_error":
                    trainer.save_runtime_checkpoint(str(folder / "before-evaluation"))

                    def failed_copy(
                        tensor,
                        *args,
                        original=original_copy,
                        target=wrapper.module.scratch,
                        **kwargs,
                    ):
                        if rank == 1 and tensor is target:
                            raise RuntimeError("injected buffer restoration failure")
                        return original(tensor, *args, **kwargs)

                    torch.Tensor.copy_ = failed_copy
                rejection = case in {
                    "error",
                    "cancel",
                    "policy",
                    "seed",
                    "counter",
                    "metric",
                    "save_error",
                    "buffer_error",
                    "operation",
                }
                original_gather = dist.all_gather
                wire = []

                def captured(
                    outputs,
                    tensor,
                    *args,
                    original=original_gather,
                    observed=wire,
                    **kwargs,
                ):
                    observed.append([str(tensor.dtype), tensor.numel()])
                    return original(outputs, tensor, *args, **kwargs)

                dist.all_gather = captured
                try:
                    if case == "operation" and rank == 1:
                        trainer.save_runtime_checkpoint(str(folder / "checkpoint"))
                    else:
                        metrics = asyncio.run(trainer.evaluate(scenarios))
                except (ValueError, asyncio.CancelledError) as error:
                    caught = type(error).__name__
                    assert rejection, (case, error)
                finally:
                    dist.all_gather = original_gather
                    torch.save = original_save
                    torch.Tensor.copy_ = original_copy
                    trainer._evaluate_local = original_local
                    if saved_policy is not None:
                        with torch.no_grad():
                            wrapper.module.lookup.weight.copy_(saved_policy)
                    if case == "seed" and rank == 1:
                        trainer.config.seed -= 1
                    if case == "counter":
                        trainer.best_eval_metric = selection[0]
                assert trainer.agent.model is wrapper
                if case == "buffer_error":
                    assert trainer._ddp_restore_failed is True
                    with pytest.raises(ValueError, match="preflight"):
                        asyncio.run(trainer.training_step([_groups()[rank]]))
                    with pytest.raises(ValueError, match="admission"):
                        asyncio.run(trainer.evaluate(scenarios))
                    assert trainer.load_runtime_checkpoint(
                        str(folder / "before-evaluation")
                    )
                    assert trainer._ddp_restore_failed is False
                assert modes == [m.training for m in wrapper.modules()]
                assert wrapper.module.config.use_cache is False
                _same(_state(trainer), before, torch)
                if rejection:
                    assert caught is not None
                    assert (
                        trainer.best_eval_metric,
                        trainer.steps_without_improvement,
                    ) == selection
                    if case == "save_error":
                        assert (
                            path / ".stateset-checkpoint.json"
                        ).read_bytes() == marker
                    else:
                        assert not path.exists()
                    if case in {"policy", "seed", "counter", "operation"}:
                        assert observed_buffers == []
                    # Repair and run the same operation on the same live group.
                    scenarios = [{"reward": float(rank), "length": rank + 1}] * (
                        1 if rank == 0 else 3
                    )
                    metrics = asyncio.run(trainer.evaluate(scenarios))
                if case == "empty_all":
                    assert metrics["eval_num_episodes"] == 0
                    assert (
                        trainer.best_eval_metric,
                        trainer.steps_without_improvement,
                    ) == selection
                    assert not path.exists()
                else:
                    assert metrics["eval_reward"] == pytest.approx(expected_reward)
                    assert metrics["eval_reward_std"] == pytest.approx(
                        0 if case == "empty_peer" else math.sqrt(3 / 16)
                    )
                    assert metrics["eval_episode_length"] == pytest.approx(
                        2 if case == "empty_peer" else 1.75
                    )
                    assert metrics["eval_success_rate"] == expected_reward
                    assert metrics["eval_complete"] == 1
                    if case in {"nonimprovement", "tie"}:
                        assert trainer.steps_without_improvement == 3
                        assert not path.exists()
                    else:
                        assert path.is_dir()
                        if case != "save_error":
                            assert trainer.steps_without_improvement == 0
                        resumed = _make(folder, rank)
                        resumed.config.eval_num_generations = 1
                        resumed.config.rollout_concurrency = 1
                        resumed.agent.model.broadcast_buffers = broadcast
                        assert resumed.load_checkpoint(path)
                        assert resumed.best_eval_metric == expected_reward
                        for learner in (trainer, resumed):
                            asyncio.run(learner.training_step([_groups()[rank]]))
                        _same(
                            trainer.agent.model.state_dict(),
                            resumed.agent.model.state_dict(),
                            torch,
                        )
                        _same(
                            trainer.optimizer.state_dict(),
                            resumed.optimizer.state_dict(),
                            torch,
                        )
                assert all(
                    value == (10 if broadcast else rank + 10)
                    for value in observed_buffers
                )
                assert all(
                    row
                    in [["torch.int64", 66], ["torch.int64", 4], ["torch.float64", 3]]
                    for row in wire
                )
                if not rejection:
                    assert ["torch.int64", 4] in wire
                    assert ["torch.float64", 3] in wire
                results.append(
                    {
                        "case": case,
                        "broadcast_buffers": broadcast,
                        "global_metric_correct": True,
                        "coordinated_selection": True,
                        "rng_modes_gradients_buffers_preserved": True,
                        "rejected": rejection,
                        "repaired": rejection,
                        "typed_tensor_wire": wire,
                    }
                )
    finally:
        (root / f"ddp-evaluation-{rank}.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        dist.destroy_process_group()


@pytest.mark.timeout(120)
def test_native_ddp_uneven_evaluation_and_best_checkpoint(tmp_path):
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
                pytest.fail("native DDP evaluation exceeded 90 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        report_dir = os.environ.get("STATESET_DISTRIBUTED_CONTRACT_REPORT_DIR")
        if report_dir:
            destination = Path(report_dir)
            destination.mkdir(parents=True, exist_ok=True)
            for report in tmp_path.glob("ddp-evaluation-*.json"):
                shutil.copy2(report, destination / report.name)
    for rank in [0, 1]:
        rows = json.loads((tmp_path / f"ddp-evaluation-{rank}.json").read_text())
        assert len(rows) == len(CASES) * 2
