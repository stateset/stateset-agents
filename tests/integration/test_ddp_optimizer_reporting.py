"""True two-worker receipts cover task/final flushes, AMP skips and recovery."""

from __future__ import annotations

import asyncio
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


def _configure(root, rank, mode, world=2):
    import torch

    from stateset_agents.training.continual_learning import (
        ContinualLearningConfig,
        ContinualLearningManager,
    )
    from tests.integration.test_ddp_empty_population import _make
    from tests.unit.test_token_group_aggregation import _groups

    trainer = _make(
        root, rank, world=world, scaler=mode in {"amp_skip", "amp_boundary_skip"}
    )
    # The population fixture adds an unused policy parameter after constructing
    # its reference. Continual teachers include that parameter, so the initial
    # reference must have the same architecture for a meaningful full reload.
    trainer.reference_model.register_parameter(
        "unused", torch.nn.Parameter(torch.tensor([0.2]), requires_grad=False)
    )
    # Exercise a real scheduler; the shared population fixture uses a constant
    # learning rate and deliberately leaves lr_scheduler unset.
    trainer.config.lr_scheduler_type = "linear"
    tasks = (
        ["A", "B"]
        if mode in {"boundary", "stop", "amp_boundary_skip"}
        else ["A"] * (4 if mode == "batch_final" else 1)
    )
    if mode == "inner":
        trainer.config.num_gradient_updates = 3
    trainer.config.num_episodes = len(tasks)
    trainer.config.logging_steps = trainer.config.eval_steps = (
        trainer.config.save_steps
    ) = 1
    trainer._get_training_scenarios = lambda: [
        {"task_id": task, "id": index} for index, task in enumerate(tasks)
    ]
    trainer._get_eval_scenarios = lambda: []
    if mode in {"boundary", "stop", "amp_boundary_skip"}:
        trainer.continual_manager = ContinualLearningManager(
            ContinualLearningConfig(strategy="lwf", continual_kl_beta=0.2),
            training_config=trainer.config,
            seed=19,
        )
    generated, receipts, logs, evaluations, finished, events = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    async def generate(scenarios):
        generated.append(scenarios[0]["id"])
        return [_groups()[0]] if mode != "empty" and (rank == 1 or world == 1) else []

    async def evaluate(scenarios):
        evaluations.append(trainer.global_step)
        if mode == "stop":
            trainer.steps_without_improvement = 1
        return {"eval_reward": 0.5}

    class Callback:
        fail_on_error = True

        def __call__(self, event, data):
            events.append(event)
            if event == "optimizer_step":
                receipts.append(
                    {
                        key: data[key]
                        for key in (
                            "optimizer_step_start",
                            "optimizer_step_end",
                            "optimizer_updates",
                            "update_source",
                            "update_task_id",
                        )
                    }
                )
                if data["update_source"] != "batch":
                    assert "total_loss" not in data and "total_reward" not in data
                    assert data["trajectory_metrics"] == {}

    class Logger:
        def log_training_step(
            self,
            losses,
            learning_rate,
            step,
            trajectory_groups=None,
            *,
            trajectory_metrics=None,
        ):
            assert trajectory_groups is None
            logs.append(
                {
                    "step": step,
                    "losses": dict(losses),
                    "trajectories": dict(trajectory_metrics),
                }
            )

        def log_evaluation(self, **kwargs):
            pass

        def finish_run(self, summary):
            finished.append(summary)

    trainer.callbacks = [Callback()]
    trainer.generate_trajectories = generate
    trainer.evaluate = evaluate
    trainer.wandb_logger = Logger() if world == 2 and rank == 0 else None
    if world == 1:

        async def save_reference(*args, **kwargs):
            pass

        trainer.save_checkpoint = save_reference
    if mode == "stop":
        trainer.config.early_stopping = True
        trainer.config.patience = 1
    return trainer, SimpleNamespace(
        generated=generated,
        receipts=receipts,
        logs=logs,
        evaluations=evaluations,
        finished=finished,
        events=events,
    )


def _worker(rank, directory):
    import torch
    import torch.distributed as dist

    from tests.integration.test_learner_preflight import _same

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
        for mode in (
            "final",
            "boundary",
            "batch_final",
            "empty",
            "amp_skip",
            "stop",
            "inner",
            "amp_boundary_skip",
        ):
            folder = root / mode
            trainer, report = _configure(folder, rank, mode)
            before = trainer.agent.model.module.lookup.weight.detach().clone()
            if mode in {"amp_skip", "amp_boundary_skip"} and rank == 1:
                trainer.agent.model.module.lookup.weight.register_hook(
                    lambda gradient: torch.full_like(gradient, float("inf"))
                )
            asyncio.run(trainer.train())
            expected_sources = {
                "final": ["final_flush"],
                "boundary": ["task_boundary", "final_flush"],
                "batch_final": ["batch", "final_flush"],
                "empty": [],
                "amp_skip": [],
                "stop": ["task_boundary"],
                "inner": ["batch"],
                "amp_boundary_skip": [],
            }[mode]
            assert [row["update_source"] for row in report.receipts] == expected_sources
            assert trainer.global_step == sum(
                row["optimizer_updates"] for row in report.receipts
            )
            if mode == "inner":
                assert trainer.global_step == 3 and len(report.receipts) == 1
            assert report.evaluations == [
                row["optimizer_step_end"] for row in report.receipts
            ]
            if mode == "stop":
                assert report.generated == [0] and trainer._current_task_id == "A"
                assert trainer.continual_manager._task_boundary["task_id"] == "A"
            if mode in {"amp_skip", "amp_boundary_skip", "empty"}:
                torch.testing.assert_close(
                    trainer.agent.model.module.lookup.weight, before, rtol=0, atol=0
                )
            if mode in {"amp_skip", "amp_boundary_skip"}:
                assert trainer.scaler.get_scale() == (
                    64.0 if mode == "amp_skip" else 32.0
                )
            if rank == 0:
                assert len(report.logs) == len(expected_sources)
                for entry, source in zip(report.logs, expected_sources, strict=True):
                    if source != "batch":
                        assert entry["trajectories"] == {}
                        assert entry["losses"] == {
                            "optimizer_updates": 1,
                            "accumulation_flush_updates": 1,
                            f"{source}_updates": 1,
                        }
                assert len(report.finished) == 1 and not report.finished[0]["errored"]
            if mode in {"final", "boundary", "batch_final", "inner"}:
                reference, _ = _configure(folder / "reference", 0, mode, world=1)
                asyncio.run(reference.train())
                torch.testing.assert_close(
                    trainer.agent.model.module.lookup.weight,
                    reference.agent.model.lookup.weight,
                    rtol=2e-5,
                    atol=2e-6,
                )
                for actual, expected in zip(
                    trainer.optimizer.state.values(),
                    reference.optimizer.state.values(),
                    strict=True,
                ):
                    for name in actual:
                        torch.testing.assert_close(
                            actual[name], expected[name], rtol=2e-5, atol=2e-6
                        )
            # Completed resume may save again, but must emit no additional updates.
            snapshot = folder / "completed"
            trainer.save_runtime_checkpoint(str(snapshot))
            count = len(report.receipts)
            trainer.config.resume_from_checkpoint = str(snapshot)
            if mode != "stop":
                asyncio.run(trainer.train())
                assert len(report.receipts) == count
            results.append({"case": mode, "correct_receipts_and_cadence": True})

        for mode in ("final", "boundary"):
            for phase in (
                "optimizer-step",
                "training-log",
                "evaluation-log",
                "evaluation-end",
            ):
                folder = root / f"{mode}-{phase}"
                trainer, report = _configure(folder, rank, mode)
                _, _, budget = trainer._prepare_training_loop()
                trainer._setup_scheduler(budget)
                saved = folder / "initial"
                trainer.save_runtime_checkpoint(str(saved))

                class Fail:
                    fail_on_error = True

                    def __call__(self, event, data, target=phase):
                        if rank == 1 and event == {
                            "optimizer-step": "optimizer_step",
                            "evaluation-end": "evaluation_end",
                        }.get(target):
                            raise OSError("injected flush callback failure")

                def fail(*args, **kwargs):
                    raise OSError("injected flush logger failure")

                original_train_log = trainer._log_training_metrics
                original_eval_log = trainer._log_eval_metrics
                if phase in {"optimizer-step", "evaluation-end"}:
                    trainer.callbacks.append(Fail())
                elif rank == 1 and phase == "training-log":
                    trainer._log_training_metrics = fail
                elif rank == 1 and phase == "evaluation-log":
                    trainer._log_eval_metrics = fail
                with pytest.raises(ValueError):
                    asyncio.run(trainer.train())
                assert trainer.global_step == 1 and trainer._ddp_restore_failed
                assert "training_end" not in report.events
                if rank == 0:
                    assert report.finished[-1]["errored"]
                with pytest.raises(ValueError):
                    trainer.save_runtime_checkpoint(str(folder / "forbidden"))
                trainer.callbacks = trainer.callbacks[:1]
                trainer._log_training_metrics = original_train_log
                trainer._log_eval_metrics = original_eval_log
                assert trainer.load_runtime_checkpoint(str(saved))
                assert not trainer._ddp_restore_failed
                report.receipts.clear()
                report.evaluations.clear()
                asyncio.run(trainer.train())
                reference, _ = _configure(folder / "reference", 0, mode, world=1)
                asyncio.run(reference.train())
                torch.testing.assert_close(
                    trainer.agent.model.module.lookup.weight,
                    reference.agent.model.lookup.weight,
                    rtol=2e-5,
                    atol=2e-6,
                )
                _same(
                    trainer.lr_scheduler.state_dict(),
                    reference.lr_scheduler.state_dict(),
                    torch,
                )
                results.append(
                    {
                        "case": f"{mode}-{phase}",
                        "coherent_failure_and_full_recovery": True,
                    }
                )
    finally:
        (root / f"ddp-optimizer-reporting-{rank}.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        dist.destroy_process_group()


@pytest.mark.timeout(140)
def test_ddp_flush_reporting_and_recovery(tmp_path):
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
        deadline = time.monotonic() + 110
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("optimizer reporting contract exceeded 110 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        report = os.environ.get("STATESET_DISTRIBUTED_CONTRACT_REPORT_DIR")
        if report:
            Path(report).mkdir(parents=True, exist_ok=True)
            for path in tmp_path.glob("ddp-optimizer-reporting-*.json"):
                shutil.copy2(path, Path(report) / path.name)
    for rank in (0, 1):
        assert (
            len(
                json.loads(
                    (tmp_path / f"ddp-optimizer-reporting-{rank}.json").read_text()
                )
            )
            == 16
        )
