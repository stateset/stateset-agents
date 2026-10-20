"""True two-worker rollout reports agree with independent combined populations."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import shutil
import sys
import time
from datetime import timedelta
from decimal import Decimal, localcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.integration


def _groups(values):
    return [
        SimpleNamespace(trajectories=[SimpleNamespace(total_reward=v) for v in group])
        for group in values
    ]


def _reference(groups):
    values = [float(v) for group in groups for v in group]

    def moments(rows):
        if not rows:
            return 0.0, 0.0
        with localcontext() as context:
            context.prec = 1100
            numbers = [Decimal.from_float(float(v)) for v in rows]
            mean = sum(numbers) / len(numbers)
            std = (sum((v - mean) ** 2 for v in numbers) / len(numbers)).sqrt()
            return float(mean), float(std)

    mean, std = moments(values)
    populated = [g for g in groups if g]
    result = {
        "num_groups": len(groups),
        "total_trajectories": len(values),
        "avg_group_size": len(values) / len(populated) if populated else 0.0,
        "reward_mean": mean,
        "reward_std": std,
        "reward_min": min(values, default=0.0),
        "reward_max": max(values, default=0.0),
    }
    diversities = [moments(g)[1] for g in groups if len(g) > 1]
    if diversities:
        result["reward_diversity_mean"], result["reward_diversity_std"] = moments(
            diversities
        )
    ordered = sorted(values)
    with localcontext() as context:
        context.prec = 1100
        result["reward_median"] = (
            float(
                (
                    Decimal.from_float(ordered[(len(values) - 1) // 2])
                    + Decimal.from_float(ordered[len(values) // 2])
                )
                / 2
            )
            if values
            else 0.0
        )
    return result


def _worker(rank, directory):
    import torch
    import torch.distributed as dist

    from stateset_agents.training import rollout_metrics as module
    from stateset_agents.training.ddp_checkpoint_state import capture_payload
    from stateset_agents.utils.wandb_logger import WandBLogger
    from tests.integration.test_ddp_empty_population import _make
    from tests.integration.test_ddp_training_loop import _configure
    from tests.integration.test_learner_preflight import _same
    from tests.unit.test_token_group_aggregation import _groups as token_groups

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
    results = []
    try:
        trainer = _make(root / "populations", rank)
        patterns = {
            "unequal": [[[-2, 5]], [[-10, 1, 9]]],
            "empty_main": [[], [[-2, 5], [7]]],
            "empty_peer": [[[-2, 5], [7]], []],
            "all_empty": [[], []],
            "empty_groups": [[[], [-2, 5]], [[], [-10, 1, 9]]],
            "singleton": [[[7]], []],
            "ties": [[[3, 3]], [[3, 3, 3]]],
            "signed_zero": [[[-0.0]], [[0.0, -0.0]]],
            "extremes": [[[-1e308, 1e308]], [[0.0]]],
            "same_extreme": [[[1e308]], [[1e308]]],
            "subnormal": [[[5e-324]], [[5e-324]]],
            "skewed_medians": [[[100]], [[-10, -9, -8, -7]]],
        }
        for name, populations in patterns.items():
            before = copy.deepcopy(capture_payload(trainer))
            actual = module.summarize_ddp_rollouts(
                trainer, _groups(populations[rank]), phase=name, details=True
            )
            expected = _reference(populations[0] + populations[1])
            assert actual.keys() == expected.keys()
            for key in expected:
                if key in {"reward_median", "num_groups", "total_trajectories"}:
                    assert actual[key] == expected[key]
                else:
                    assert actual[key] == pytest.approx(expected[key], rel=2e-12, abs=0)
            assert all(math.isfinite(v) for v in actual.values())
            _same(capture_payload(trainer), before, torch)
            results.append(
                {
                    "case": name,
                    "population_equal": True,
                    "learner_state_preserved": True,
                }
            )

        for fault in ["missing", "nan", "infinite", "schema", "details"]:
            groups = _groups([[1, 2]])
            if rank == 1 and fault in {"missing", "nan", "infinite"}:
                groups[0].trajectories[0].total_reward = {
                    "missing": None,
                    "nan": float("nan"),
                    "infinite": float("inf"),
                }[fault]
            pack = module._pack

            def invalid_pack(row, original=pack):
                counts, values = original(row)
                if rank == 1:
                    counts[1] = counts[0] + 1
                return counts, values

            if fault == "schema":
                module._pack = invalid_pack
            before = copy.deepcopy(capture_payload(trainer))
            try:
                with pytest.raises(ValueError):
                    module.summarize_ddp_rollouts(
                        trainer,
                        groups,
                        phase="invalid",
                        details=rank == 1 if fault == "details" else True,
                    )
            finally:
                module._pack = pack
            _same(capture_payload(trainer), before, torch)
            assert (
                module.summarize_ddp_rollouts(
                    trainer, _groups([[1, 2]]), phase="recovery", details=True
                )["reward_median"]
                == 1.5
            )
            results.append({"case": fault, "coherent_rejection_and_recovery": True})

        for mode in ["wandb", "legacy", "logger_type_error", "callback_mutation"]:
            folder = root / mode
            learner, events, generated, finished = _configure(folder, rank)
            learner.config.num_episodes = 1
            learner.config.logging_steps = 1
            episodes, logs = [], []

            class Callback:
                fail_on_error = True

                def __call__(self, event, data, sink=episodes):
                    if event == "episode_end":
                        sink.append(copy.deepcopy(data))

            learner.callbacks.append(Callback())
            if mode == "callback_mutation":

                class MutatingCallback:
                    def on_episode_end(self, episode, metrics):
                        metrics["optimizer_updates"] = 0
                        metrics["total_reward"] = -999.0
                        metrics["trajectory_metrics"]["reward_mean"] = -999.0

                learner.callbacks.append(MutatingCallback())

            def mix(groups, task):
                replay = [token_groups()[1]] if rank == 0 else []
                return [*groups, *replay], replay

            learner._maybe_mix_replay = mix
            if rank == 0:
                if mode in {"wandb", "callback_mutation"}:
                    log = WandBLogger(enabled=False)
                    log.enabled = True
                    log.log_metrics = lambda data, sink=logs, **kwargs: sink.append(
                        data
                    )
                else:

                    class LegacyLogger:
                        def log_training_step(
                            self,
                            losses,
                            learning_rate,
                            step,
                            trajectory_groups=None,
                            *,
                            sink=logs,
                            kind=mode,
                        ):
                            assert trajectory_groups is None
                            sink.append(losses)
                            if kind == "logger_type_error":
                                raise TypeError("body failure; must not retry")

                        def finish_run(self, summary):
                            pass

                    log = LegacyLogger()
                learner.wandb_logger = log
            else:
                learner.wandb_logger = None
            _, _, budget = learner._prepare_training_loop()
            learner._setup_scheduler(budget)
            saved = folder / "initial"
            learner.save_runtime_checkpoint(str(saved))
            if mode == "logger_type_error":
                with pytest.raises(ValueError):
                    asyncio.run(learner.train())
                assert learner._ddp_restore_failed
                assert learner.load_runtime_checkpoint(str(saved))
                assert not learner._ddp_restore_failed
            else:
                asyncio.run(learner.train())
                assert learner.global_step == 1
                reference = _make(folder / "reference", world=1)
                reference.config.gradient_accumulation_steps = 1
                asyncio.run(reference.training_step(token_groups()))
                torch.testing.assert_close(
                    learner.agent.model.module.lookup.weight,
                    reference.agent.model.lookup.weight,
                    rtol=2e-5,
                    atol=2e-6,
                )
            assert len(episodes) == 1
            data = episodes[0]
            # Replay on rank zero must not contaminate the new-rollout callback.
            assert data["total_reward"] == data["episode_reward"] == 1.5
            assert data["episode_num_trajectories"] == 2
            assert data["episode_reward_scope"] == "ddp_global_new_rollouts"
            wanted = _reference([[-2, 5], [-10, 1, 9]])
            assert data["trajectory_metrics"]["reward_median"] == 1
            for key, value in wanted.items():
                assert data["trajectory_metrics"][key] == pytest.approx(value)
                if rank == 0:
                    assert len(logs) == 1
                    assert logs[0][
                        ("train/" if mode in {"wandb", "callback_mutation"} else "")
                        + key
                    ] == pytest.approx(value)
            results.append(
                {
                    "case": mode,
                    "episode_and_training_populations_separate": True,
                    "empty_main_logs_global": True,
                }
            )
    finally:
        (root / f"ddp-rollout-metrics-{rank}.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        dist.destroy_process_group()


def test_ddp_rollout_populations_callbacks_and_loggers(tmp_path):
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
                pytest.fail("rollout metrics contract exceeded 90 seconds")
    finally:
        for p in context.processes:
            if p.is_alive():
                p.terminate()
            p.join(timeout=5)
        report = os.environ.get("STATESET_DISTRIBUTED_CONTRACT_REPORT_DIR")
        if report:
            Path(report).mkdir(parents=True, exist_ok=True)
            for p in tmp_path.glob("ddp-rollout-metrics-*.json"):
                shutil.copy2(p, Path(report) / p.name)
    for rank in [0, 1]:
        assert (
            len(json.loads((tmp_path / f"ddp-rollout-metrics-{rank}.json").read_text()))
            == 21
        )
