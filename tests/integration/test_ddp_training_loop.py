"""Real Gloo peers finish empty episodes and reject local loop failures together."""

from __future__ import annotations

import asyncio
import copy
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


def _configure(root, rank):
    import torch

    from tests.integration.test_ddp_empty_population import _make
    from tests.unit.test_token_group_aggregation import _groups

    trainer = _make(root, rank)
    trainer.config.num_episodes = 3
    trainer.config.gradient_accumulation_steps = 1
    trainer.config.logging_steps = 100
    trainer.config.eval_steps = 100
    trainer.config.save_steps = 100
    trainer._get_training_scenarios = lambda: [{"id": i} for i in range(3)]
    trainer._get_eval_scenarios = lambda: []
    trainer._maybe_mix_replay = lambda groups, task: (groups, [])
    events, generated, finished = [], [], []

    async def generate(scenarios):
        assert not torch.is_grad_enabled()
        assert not trainer.agent.model.training
        assert not isinstance(
            trainer.agent.model, torch.nn.parallel.DistributedDataParallel
        )
        generated.append(scenarios[0]["id"])
        # Exercise actual local inference while peers have different populations.
        trainer.agent.model(torch.tensor([[1, 2]]))
        return [_groups()[0]] if rank == 1 and scenarios[0]["id"] != 1 else []

    trainer.generate_trajectories = generate

    class Callback:
        fail_on_error = True

        def __call__(self, event, data):
            events.append(event)

    trainer.callbacks = [Callback()]
    trainer.wandb_logger = SimpleNamespace(
        finish_run=lambda summary: finished.append(summary)
    )
    return trainer, events, generated, finished


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
        timeout=timedelta(seconds=25),
    )
    results = []
    try:
        # A checkpoint taken before episode zero must resume all three episodes.
        folder = root / "success"
        trainer, events, generated, finished = _configure(folder, rank)
        _, _, budget = trainer._prepare_training_loop()
        trainer._setup_scheduler(budget)
        initial = folder / "initial"
        trainer.save_runtime_checkpoint(str(initial))
        trainer.config.resume_from_checkpoint = str(initial)
        wrapper = trainer.agent.model
        asyncio.run(trainer.train())
        assert generated == [0, 1, 2]
        assert trainer._next_episode == 3 and trainer.global_step == 2
        assert events.count("episode_end") == 2
        assert events.count("training_end") == 1
        assert len(finished) == 1 and not finished[0]["errored"]
        assert trainer.agent.model is wrapper and wrapper.module.training
        before = copy.deepcopy(capture_payload(trainer))
        trainer.config.resume_from_checkpoint = str(folder / "checkpoints/checkpoint-2")
        asyncio.run(trainer.train())
        assert generated == [0, 1, 2]
        _same(capture_payload(trainer), before, torch)
        results.append(
            {"case": "empty_loop_initial_and_completed_resume", "passed": True}
        )

        # A transition can fail after A committed but before B generated data.
        # Reloading A's checkpoint must execute B, retaining the exact next
        # optimizer update and teacher of an uninterrupted two-task run.
        from tests.integration.test_ddp_continual import _make as continual

        folder = root / "task-cursor"
        task_generated = []

        def task_trainer(path):
            value = continual(path, rank, strategy="lwf")
            value.config.num_episodes = 2
            value.config.gradient_accumulation_steps = 1
            value.config.logging_steps = value.config.eval_steps = 100
            value.config.save_steps = 1
            value._get_training_scenarios = lambda: [{"task_id": "A"}, {"task_id": "B"}]
            value._get_eval_scenarios = lambda: []

            async def generate_task(scenarios):
                task_generated.append(scenarios[0]["task_id"])
                return [_groups()[rank]]

            value.generate_trajectories = generate_task
            return value

        learner = task_trainer(folder)
        manager = learner.continual_manager
        original_snapshot = manager._snapshot_reference_model

        def failed_snapshot(agent):
            if rank == 1:
                raise OSError("injected task teacher failure")
            return original_snapshot(agent)

        try:
            # Inject a transient local failure without changing the class code
            # bound into the checkpoint's implementation identity.
            manager._snapshot_reference_model = failed_snapshot
            with pytest.raises(ValueError):
                asyncio.run(learner.train())
        finally:
            del manager._snapshot_reference_model
        assert task_generated == ["A"]
        assert learner.global_step == 1 and learner._next_episode == 1
        learner.config.resume_from_checkpoint = str(folder / "checkpoints/checkpoint-1")
        asyncio.run(learner.train())
        assert task_generated == ["A", "B"]
        assert learner.global_step == learner._next_episode == 2
        reference = task_trainer(root / "task-reference")
        asyncio.run(reference.train())
        _same(
            learner.agent.model.state_dict(), reference.agent.model.state_dict(), torch
        )
        _same(learner.optimizer.state_dict(), reference.optimizer.state_dict(), torch)
        _same(
            learner.continual_manager.reference_model.state_dict(),
            reference.continual_manager.reference_model.state_dict(),
            torch,
        )
        results.append(
            {
                "case": "interrupted_task_cursor",
                "resume_next_update_and_teacher_exact": True,
            }
        )

        for failure in [
            "prepare",
            "scheduler",
            "training_start",
            "generation",
            "cancel_generation",
            "forward",
            "backward",
            "optimizer",
            "episode_end",
            "training_log",
            "evaluation_log",
            "evaluation_end",
            "stop",
            "training_end",
            "tracking",
        ]:
            folder = root / failure
            trainer, events, generated, finished = _configure(folder, rank)
            wrapper = trainer.agent.model
            _, _, budget = trainer._prepare_training_loop()
            trainer._setup_scheduler(budget)
            saved = folder / "initial"
            trainer.save_runtime_checkpoint(str(saved))

            def fail(*args, reason=failure, **kwargs):
                raise OSError(f"injected {reason}")

            if rank == 1:
                if failure == "prepare":
                    trainer._get_training_scenarios = fail
                elif failure == "scheduler":
                    trainer._setup_scheduler = fail
                elif failure in {"generation", "cancel_generation"}:

                    async def bad_generate(
                        scenarios,
                        learner=trainer,
                        actor=wrapper,
                        reason=failure,
                        fail_fn=fail,
                    ):
                        assert learner.agent.model is actor.module
                        if reason == "cancel_generation":
                            raise asyncio.CancelledError(
                                "injected rollout cancellation"
                            )
                        fail_fn()

                    trainer.generate_trajectories = bad_generate
                elif failure == "forward":
                    original = wrapper.module.forward

                    def bad_forward(*args, forward=original, fail_fn=fail, **kwargs):
                        if torch.is_grad_enabled():
                            fail_fn()
                        return forward(*args, **kwargs)

                    wrapper.module.forward = bad_forward
                elif failure == "backward":
                    wrapper.module.lookup.weight.register_hook(fail)
                elif failure == "optimizer":
                    trainer.optimizer.step = fail
                elif failure == "stop":
                    original_stop = trainer._should_stop_training

                    def different_stop(learner=trainer, stop=original_stop):
                        learner.steps_without_improvement += 1
                        return stop()

                    trainer._should_stop_training = different_stop
                elif failure == "tracking":
                    trainer.wandb_logger.finish_run = fail
                else:

                    class FailingCallback:
                        fail_on_error = True

                        def __call__(self, event, data, reason=failure, fail_fn=fail):
                            if event == reason:
                                fail_fn()

                    trainer.callbacks.append(FailingCallback())

            if failure == "training_log":
                trainer.config.logging_steps = 1
                trainer._log_training_metrics = (
                    fail if rank == 1 else lambda *args: None
                )
            if failure in {"evaluation_log", "evaluation_end"}:
                trainer.config.eval_steps = 1

                async def evaluate(*args, **kwargs):
                    return {"eval_reward": 0.5}

                trainer.evaluate = evaluate
                trainer._log_eval_metrics = (
                    fail
                    if rank == 1 and failure == "evaluation_log"
                    else lambda *args: None
                )

            wanted = (
                asyncio.CancelledError
                if rank == 1 and failure == "cancel_generation"
                else ValueError
            )
            with pytest.raises(wanted):
                asyncio.run(trainer.train())
            assert trainer.agent.model is wrapper
            assert wrapper.module.training
            assert not getattr(wrapper, "_stateset_token_forward", False)
            assert trainer._grad_accum_step == 0
            assert trainer._ddp_restore_failed
            if failure != "tracking":
                assert len(finished) == 1
            if failure not in {"tracking", "training_end"}:
                assert "training_end" not in events
            with pytest.raises(ValueError):
                asyncio.run(trainer.training_step([_groups()[0]] if rank == 1 else []))
            with pytest.raises(ValueError):
                trainer.save_runtime_checkpoint(str(folder / "invalid"))
            if failure == "forward":
                # A checkpoint cannot restore while one peer still runs a
                # different model implementation. Repair the injected code
                # change before testing recovery of the failed runtime state.
                with pytest.raises(ValueError, match="identity"):
                    trainer.load_runtime_checkpoint(str(saved))
                if rank == 1:
                    del wrapper.module.forward
            # Restoring a complete good snapshot repairs the poison marker.
            assert trainer.load_runtime_checkpoint(str(saved))
            assert not trainer._ddp_restore_failed
            assert trainer._next_episode == 0 and trainer.global_step == 0
            results.append({"case": failure, "coherent_failure_and_reload": True})
    finally:
        (root / f"ddp-training-loop-{rank}.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        dist.destroy_process_group()


@pytest.mark.timeout(120)
def test_ddp_training_loop_empty_workers_and_local_failures(tmp_path):
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
                pytest.fail("training loop contract exceeded 90 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        report = os.environ.get("STATESET_DISTRIBUTED_CONTRACT_REPORT_DIR")
        if report:
            Path(report).mkdir(parents=True, exist_ok=True)
            for path in tmp_path.glob("ddp-training-loop-*.json"):
                shutil.copy2(path, Path(report) / path.name)
