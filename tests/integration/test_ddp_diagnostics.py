"""Live peers coordinate post-update diagnostics and resume the exact guard."""

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

import pytest

from tests.integration._spawn_budget import spawn_budget

pytestmark = pytest.mark.integration


def _worker(rank, directory):
    import torch
    import torch.distributed as dist

    from stateset_agents.training.ddp_checkpoint_state import capture_payload
    from stateset_agents.training.diagnostics import capture_diagnostics
    from tests.integration.test_ddp_empty_population import _make
    from tests.integration.test_ddp_training_loop import _configure
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
        # Pending windows still retain their observation count and identity.
        trainer = _make(root / "history", rank)
        trainer.config.zero_signal_patience = 2
        zero = [_groups(zero_rewards=True)[rank]]
        asyncio.run(trainer.training_step(zero))
        assert trainer._diagnostic_observations == 1
        assert trainer.diagnostics.zero_steps == 1
        before = capture_diagnostics(trainer)
        path = root / "history" / "saved"
        trainer.save_runtime_checkpoint(str(path))
        monitor = trainer.diagnostics
        asyncio.run(trainer.training_step([]))
        assert capture_diagnostics(trainer) == before
        assert trainer._diagnostic_observations == 1
        assert trainer.load_runtime_checkpoint(str(path))
        assert trainer.diagnostics is monitor
        assert capture_diagnostics(trainer) == before
        # Same pending optimizer step is not a new guard identity.
        asyncio.run(trainer.training_step(zero))
        assert trainer._diagnostic_observations == 2
        assert monitor.zero_steps == 1
        with pytest.raises(ValueError, match="diagnostics"):
            asyncio.run(trainer.training_step(zero))
        assert trainer._ddp_restore_failed and monitor.failure
        assert trainer.load_runtime_checkpoint(str(path))
        assert not trainer._ddp_restore_failed
        assert capture_diagnostics(trainer) == before
        asyncio.run(trainer.training_step([_groups()[rank]]))
        results.append(
            {"case": "pending_history_noop_threshold_reload", "passed": True}
        )

        for fault in (
            "nonfinite_metric",
            "local_profile_error",
            "finite_signal_disagreement",
        ):
            folder = root / fault
            trainer = _make(folder, rank)
            trainer.config.gradient_accumulation_steps = 1
            path = folder / "saved"
            trainer.save_runtime_checkpoint(str(path))
            sync = trainer._sync_metrics

            def broken(metrics, original=sync, mode=fault, learner=trainer):
                result = original(metrics)
                if rank == 1:
                    if mode == "nonfinite_metric":
                        result["grad_norm"] = float("nan")
                    elif mode == "finite_signal_disagreement":
                        result["nonzero_advantage_fraction"] = 0.0
                    else:
                        learner._runtime_rollout_metrics = {"rollout_seconds": None}
                return result

            trainer._sync_metrics = broken
            with pytest.raises(ValueError, match="diagnostics"):
                asyncio.run(trainer.training_step([_groups()[rank]]))
            assert trainer.global_step == 1 and trainer._ddp_restore_failed
            with pytest.raises(ValueError):
                asyncio.run(trainer.training_step([_groups()[rank]]))
            with pytest.raises(ValueError):
                trainer.save_runtime_checkpoint(str(folder / "forbidden"))
            trainer._sync_metrics = sync
            assert trainer.load_runtime_checkpoint(str(path))
            assert not trainer._ddp_restore_failed
            assert trainer._diagnostic_observations == 0
            assert trainer.diagnostics is None or trainer.diagnostics.failure is None
            asyncio.run(trainer.training_step([_groups()[rank]]))
            reference = _make(folder / "reference", world=1)
            reference.config.gradient_accumulation_steps = 1
            asyncio.run(reference.training_step(_groups()))
            torch.testing.assert_close(
                trainer.agent.model.module.lookup.weight,
                reference.agent.model.lookup.weight,
                rtol=2e-5,
                atol=2e-6,
            )
            results.append({"case": fault, "reloaded_next_update_equal": True})

        for fault in ("patience", "streak", "observations"):
            trainer = _make(root / fault, rank)
            asyncio.run(trainer.training_step([_groups()[rank]]))
            if rank == 1:
                if fault == "patience":
                    trainer.config.zero_signal_patience = 9
                elif fault == "streak":
                    trainer.diagnostics.zero_steps += 1
                else:
                    trainer._diagnostic_observations += 1
            before = (
                copy.deepcopy(capture_payload(trainer))
                if fault != "patience" or rank == 0
                else None
            )
            with pytest.raises(ValueError, match="schedule"):
                asyncio.run(trainer.training_step([_groups()[rank]]))
            with pytest.raises(ValueError):
                trainer.save_runtime_checkpoint(str(root / fault / "forbidden"))
            assert not getattr(trainer, "_ddp_restore_failed", False)
            if before is not None:
                _same(capture_payload(trainer), before, torch)
            if rank == 1:
                if fault == "patience":
                    trainer.config.zero_signal_patience = trainer.diagnostics.patience
                elif fault == "streak":
                    trainer.diagnostics.zero_steps -= 1
                else:
                    trainer._diagnostic_observations -= 1
            asyncio.run(trainer.training_step([_groups()[rank]]))
            results.append({"case": fault, "rejected_before_mutation": True})

        trainer, events, generated, finished = _configure(root / "loop", rank)
        sync = trainer._sync_metrics

        def bad_loop(metrics):
            result = sync(metrics)
            if rank == 1:
                result["grad_norm"] = float("nan")
            return result

        trainer._sync_metrics = bad_loop
        with pytest.raises(ValueError, match="diagnostics"):
            asyncio.run(trainer.train())
        assert trainer._ddp_restore_failed
        assert generated == [0] and "training_end" not in events
        assert len(finished) == 1 and finished[0]["errored"]
        results.append({"case": "loop_cleanup", "all_peers_finished": True})
    finally:
        (root / f"ddp-diagnostics-{rank}.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        dist.destroy_process_group()


@pytest.mark.timeout(spawn_budget(120))
def test_ddp_diagnostics_failure_and_checkpoint_recovery(tmp_path):
    torch = pytest.importorskip("torch")
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_gloo_available()
    ):
        pytest.skip("Gloo required")
    context = torch.multiprocessing.spawn(
        _worker,
        args=(str(tmp_path.resolve()),),
        nprocs=2,
        join=False,
    )
    try:
        deadline = time.monotonic() + spawn_budget(90)
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail(
                    f"diagnostics contract exceeded {spawn_budget(90):.0f} seconds"
                )
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        report = os.environ.get("STATESET_DISTRIBUTED_CONTRACT_REPORT_DIR")
        if report:
            Path(report).mkdir(parents=True, exist_ok=True)
            for path in tmp_path.glob("ddp-diagnostics-*.json"):
                shutil.copy2(path, Path(report) / path.name)
    for rank in (0, 1):
        assert (
            len(json.loads((tmp_path / f"ddp-diagnostics-{rank}.json").read_text()))
            == 8
        )
