"""Real worker task memory matches an independent combined-population Fisher."""

from __future__ import annotations

import asyncio
import copy
import itertools
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
SHARDS = 4


def _matrix_cases():
    return tuple(
        itertools.product(
            ["lwf", "ewc", "replay_lwf_ewc"],
            [False, True],
            ["uneven", "empty_peer", "all_empty", "masked", "cap", "buffer"],
        )
    )


def _policy():
    import torch

    from tests.unit.test_token_group_aggregation import TinyPolicy

    class Policy(TinyPolicy):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(vocab_size=16, use_cache=False)
            self.register_buffer("buffer", torch.ones(1))
            self.calls = 0

        @property
        def device(self):
            return self.lookup.weight.device

        def forward(self, input_ids, attention_mask=None, labels=None):
            self.calls += 1
            logits = self.lookup(input_ids)
            loss = (
                None
                if labels is None
                else torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, 16),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
            )
            return SimpleNamespace(logits=logits, loss=loss)

    return Policy()


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = [
        "<unk>",
        "<pad>",
        "user",
        "assistant",
        "alpha",
        "beta",
        "gamma",
        "delta",
        "one",
        "two",
        "three",
        "four",
        "five",
        "end",
        "system",
        "go",
    ]
    backend = Tokenizer(
        models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>"
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] }} {% if m['role'] == 'assistant' %}{% generation %}{{ m['content'] }} end {% endgeneration %}{% else %}{{ m['content'] }} end {% endif %}{% endfor %}"
    return tokenizer


def _make(root, rank, strategy="replay_lwf_ewc", broadcast=True):
    import torch

    from stateset_agents.training.continual_learning import (
        ContinualLearningConfig,
        ContinualLearningManager,
    )
    from tests.integration.test_replica_consensus import _make as make

    trainer = make(root / f"rank-{rank}", rank, 2)
    trainer.config.beta = 0.0
    trainer.config.use_reference_model = False
    trainer.config.gradient_accumulation_steps = 2
    trainer.config.lr_scheduler_type = "constant"
    trainer.config.max_prompt_length = 32
    trainer.config.max_completion_length = 32
    trainer.distributed_config.checkpoint_dir = str(root / "checkpoints")
    trainer.agent.model = _policy()
    trainer.agent.tokenizer = _tokenizer()
    trainer.reference_model = None
    trainer.optimizer = torch.optim.SGD(
        trainer.agent.model.parameters(), lr=0.1, momentum=0.9
    )
    trainer.continual_manager = ContinualLearningManager(
        ContinualLearningConfig(
            strategy=strategy,
            ewc_lambda=0.2,
            ewc_decay=0.7,
            ewc_num_samples=10,
            continual_kl_beta=0.2,
            replay_min_size=1,
        ),
        training_config=trainer.config,
        seed=19 + rank,
    )
    trainer._wrap_model_ddp()
    trainer.agent.model.broadcast_buffers = broadcast
    trainer.agent.model.module.buffer.fill_(rank + 10)
    trainer._current_task_id = "A"
    for name, p in trainer.agent.model.named_parameters():
        trainer.continual_manager._ewc_fisher[name] = torch.full_like(p, 0.2)
        trainer.continual_manager._ewc_params[name] = p.detach().clone()
    return trainer


def _groups(rank, population="uneven"):
    from stateset_agents.core.trajectory import (
        ConversationTurn,
        MultiTurnTrajectory,
        TrajectoryGroup,
    )

    texts = (
        ["alpha beta"] if rank == 0 else ["gamma delta", "one two three", "alpha five"]
    )
    if population == "empty_peer" and rank == 0 or population == "all_empty":
        texts = []
    trajectories = [
        MultiTurnTrajectory(
            turns=[
                ConversationTurn(role="user", content="go"),
                ConversationTurn(role="assistant", content=text),
            ],
            total_reward=1.0,
        )
        for text in texts
    ]
    if population == "masked" and rank == 1:
        trajectories.append(
            MultiTurnTrajectory(
                turns=[ConversationTurn(role="user", content="go")], total_reward=0.0
            )
        )
    return (
        [TrajectoryGroup(scenario_id=f"rank-{rank}", trajectories=trajectories)]
        if trajectories
        else []
    )


def _reference(trainer, groups):
    import torch
    import torch.distributed as dist

    from stateset_agents.training.loss_computation import _prepare_inputs_and_labels

    policy = trainer.agent.model.module
    value = torch.zeros_like(policy.lookup.weight, dtype=torch.float64)
    count = 0
    for trajectory in [t for g in groups for t in g.trajectories][
        : trainer.continual_manager.config.ewc_num_samples
    ]:
        inputs, labels = _prepare_inputs_and_labels(
            trajectory,
            SimpleNamespace(model=policy, tokenizer=trainer.agent.tokenizer),
            trainer.config,
        )
        if not bool(labels[:, 1:].ne(-100).any()):
            continue
        (gradient,) = torch.autograd.grad(
            policy(**inputs, labels=labels).loss, [policy.lookup.weight]
        )
        value += gradient.detach().double().square()
        count += 1
    encoded = torch.tensor(count, dtype=torch.int64)
    dist.all_reduce(encoded)
    dist.all_reduce(value)
    return value / encoded.item() if encoded.item() else value, encoded.item()


def _worker(rank, directory, shard):
    import torch
    import torch.distributed as dist

    from stateset_agents.training import continual_learning as module
    from stateset_agents.training import ddp_continual
    from stateset_agents.training.ddp_checkpoint_state import capture_payload
    from stateset_agents.training.runtime_state import capture_rng
    from tests.integration.test_learner_preflight import _same
    from tests.unit.test_token_group_aggregation import _groups as rollout_groups

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
        for case_index, (strategy, broadcast, population) in enumerate(_matrix_cases()):
            if case_index % SHARDS != shard:
                continue
            folder = root / f"{strategy}-{broadcast}-{population}"
            trainer = _make(folder, rank, strategy, broadcast)
            manager = trainer.continual_manager
            asyncio.run(trainer.training_step([rollout_groups()[rank]]))
            if population == "cap":
                manager.config.ewc_num_samples = 2
            groups = _groups(rank, population)
            manager.add_trajectory_groups(groups, task_id="A")
            selected = groups
            if population == "buffer":
                shadow = copy.copy(manager.buffer)
                shadow._rng = copy.deepcopy(manager.buffer._rng)
                selected = shadow.sample_groups(
                    manager.config.ewc_num_samples, task_id="A"
                )
            expected, accepted = _reference(trainer, selected)
            wrapper = trainer.agent.model
            wrapper.module.lookup.eval()
            modes = [m.training for m in wrapper.modules()]
            before = copy.deepcopy(capture_payload(trainer))
            before_rng = copy.deepcopy(capture_rng(torch))
            original_manager = manager
            with torch.no_grad():
                manager.on_task_end(
                    trainer.agent, "A", None if population == "buffer" else groups
                )
            assert trainer.continual_manager is original_manager
            _same(capture_rng(torch), before_rng, torch)
            assert modes == [m.training for m in wrapper.modules()]
            actual = capture_payload(trainer)
            _same(actual[0]["policy"], before[0]["policy"], torch)
            _same(actual[0]["optimizer"], before[0]["optimizer"], torch)
            _same(actual[1]["buffers"], before[1]["buffers"], torch)
            _same(
                actual[1]["gradient_accumulation"],
                before[1]["gradient_accumulation"],
                torch,
            )
            if manager.config.uses_ewc():
                assert manager.last_fisher_estimate["accepted"] == accepted
                wanted = 0.14 + expected if accepted else torch.full_like(expected, 0.2)
                torch.testing.assert_close(
                    manager._ewc_fisher["module.lookup.weight"].double(),
                    wanted,
                    rtol=2e-6,
                    atol=1e-8,
                )
            else:
                assert manager.last_fisher_estimate["accepted"] == 0
            if manager.config.uses_lwf():
                teacher = manager.reference_model
                assert not isinstance(
                    teacher, torch.nn.parallel.DistributedDataParallel
                )
                assert not teacher.training and not any(
                    p.requires_grad for p in teacher.parameters()
                )
                torch.testing.assert_close(
                    teacher.lookup.weight, wrapper.module.lookup.weight, rtol=0, atol=0
                )
                assert teacher.buffer.item() == (10 if broadcast else rank + 10)
                trainer.reference_model = teacher
            path = folder / "saved"
            trainer.save_runtime_checkpoint(str(path))
            restored = _make(folder, rank, strategy, broadcast)
            restored.continual_manager.config.ewc_num_samples = (
                manager.config.ewc_num_samples
            )
            assert restored.load_runtime_checkpoint(str(path))
            for learner in (trainer, restored):
                asyncio.run(learner.training_step([rollout_groups()[rank]]))
            _same(
                trainer.agent.model.state_dict(),
                restored.agent.model.state_dict(),
                torch,
            )
            _same(
                trainer.optimizer.state_dict(), restored.optimizer.state_dict(), torch
            )
            results.append(
                {
                    "matrix_index": case_index,
                    "strategy": strategy,
                    "broadcast_buffers": broadcast,
                    "population": population,
                    "global_fisher_matches_reference": True,
                    "pending_state_preserved": True,
                    "plain_frozen_teacher": True,
                    "resume_next_update_equal": True,
                }
            )

        for case_index, fault in enumerate(
            [
                "snapshot",
                "estimate",
                "config",
                "task",
                "commit",
                "window",
                "flush",
                "after_flush",
            ],
            start=len(_matrix_cases()),
        ):
            if case_index % SHARDS != shard:
                continue
            folder = root / fault
            trainer = _make(folder, rank)
            manager = trainer.continual_manager
            manager.add_trajectory_groups(_groups(rank), task_id="A")
            asyncio.run(trainer.training_step([rollout_groups()[rank]]))
            path = folder / "before"
            trainer.save_runtime_checkpoint(str(path))
            before = copy.deepcopy(capture_payload(trainer))
            step = trainer.optimizer.step
            snapshot, estimate, commit = (
                module.ContinualLearningManager._snapshot_reference_model,
                module.ContinualLearningManager._estimate_fisher,
                ddp_continual._commit,
            )

            def fail_snapshot(self, agent, original=snapshot):
                if rank == 1:
                    raise RuntimeError("snapshot failed")
                return original(self, agent)

            def fail_estimate(self, agent, trajectories, original=estimate):
                if rank == 1:
                    raise torch.OutOfMemoryError("estimate failed")
                return original(self, agent, trajectories)

            def fail_commit(target, staged, original=commit):
                if rank == 1:
                    raise RuntimeError("commit failed")
                return original(target, staged)

            def fail_step(*args, original=step, **kwargs):
                if rank == 1:
                    raise RuntimeError("optimizer flush failed")
                return original(*args, **kwargs)

            if fault == "snapshot":
                module.ContinualLearningManager._snapshot_reference_model = (
                    fail_snapshot
                )
            if fault in {"estimate", "after_flush"}:
                module.ContinualLearningManager._estimate_fisher = fail_estimate
            if fault == "commit":
                ddp_continual._commit = fail_commit
            if fault == "config" and rank == 1:
                manager.config.ewc_decay = 0.1
            if fault == "window" and rank == 1:
                trainer._grad_accum_step = 0
            if fault == "flush":
                trainer.optimizer.step = fail_step
            try:
                with pytest.raises(ValueError):
                    if fault in {"window", "flush", "after_flush"}:
                        trainer._maybe_handle_task_switch("B")
                    else:
                        manager.on_task_end(
                            trainer.agent, "B" if fault == "task" and rank else "A"
                        )
            finally:
                (
                    module.ContinualLearningManager._snapshot_reference_model,
                    module.ContinualLearningManager._estimate_fisher,
                    ddp_continual._commit,
                ) = (snapshot, estimate, commit)
                trainer.optimizer.step = step
                if fault == "config":
                    manager.config.ewc_decay = 0.7
                if fault == "window":
                    trainer._grad_accum_step = 1
            if fault in {"commit", "flush"}:
                assert manager._ddp_restore_failed or trainer._ddp_restore_failed
                with pytest.raises(ValueError):
                    asyncio.run(trainer.training_step([rollout_groups()[rank]]))
                with pytest.raises(ValueError):
                    trainer.save_runtime_checkpoint(str(folder / "invalid"))
                assert trainer.load_runtime_checkpoint(str(path))
            elif fault == "after_flush":
                assert trainer.global_step == 1 and trainer._grad_accum_step == 0
                assert trainer._current_task_id == "A"
                _same(
                    manager.state_dict(),
                    before[1]["learning_state"]["continual_state"],
                    torch,
                )
            else:
                _same(capture_payload(trainer), before, torch)
            trainer._maybe_handle_task_switch("B")
            assert (
                trainer.global_step == 1
                and trainer._grad_accum_step == 0
                and trainer._current_task_id == "B"
            )
            assert manager._task_boundary == {
                "task_id": "A",
                "global_step": 1,
                "buffer_sequence": manager.buffer._sequence,
            }
            results.append(
                {
                    "matrix_index": case_index,
                    "fault": fault,
                    "coordinated_rejection": True,
                    "recovered": True,
                    "boundary_flush": True,
                }
            )
        for case_index, inner in enumerate([1, 2], start=len(_matrix_cases()) + 8):
            if case_index % SHARDS != shard:
                continue
            folder = root / f"full-train-{inner}"
            trainer = _make(folder, rank)
            trainer.config.num_episodes = 5
            trainer.config.num_gradient_updates = inner
            trainer.config.logging_steps = trainer.config.eval_steps = (
                trainer.config.save_steps
            ) = 1000
            trainer._get_training_scenarios = lambda: [
                {"task_id": t} for t in ["A", "B", "A", "B", "A"]
            ]
            trainer._get_eval_scenarios = lambda: []

            async def generate(_scenarios):
                return [rollout_groups()[rank]]

            trainer.generate_trajectories = generate
            asyncio.run(trainer.train())
            assert trainer.global_step == 5 * inner and trainer._grad_accum_step == 0
            before = copy.deepcopy(capture_payload(trainer))
            trainer.config.resume_from_checkpoint = str(
                folder / "checkpoints" / f"checkpoint-{5*inner}"
            )
            asyncio.run(trainer.train())
            _same(capture_payload(trainer), before, torch)
            results.append(
                {
                    "matrix_index": case_index,
                    "full_train_inner_updates": inner,
                    "task_window_budget": 5 * inner,
                    "complete_resume_idempotent": True,
                }
            )
    finally:
        (root / f"ddp-continual-{rank}.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        dist.destroy_process_group()


@pytest.mark.timeout(120)
@pytest.mark.parametrize("shard", range(SHARDS))
def test_native_ddp_task_memory_population_and_recovery(tmp_path, shard):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_gloo_available()
    ):
        pytest.skip("Gloo required")
    context = torch.multiprocessing.spawn(
        _worker, args=(str(tmp_path.resolve()), shard), nprocs=2, join=False
    )
    try:
        deadline = time.monotonic() + 90
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("DDP task memory exceeded 90 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        directory = os.environ.get("STATESET_DISTRIBUTED_CONTRACT_REPORT_DIR")
        if directory:
            destination = Path(directory) / "ddp-continual" / f"shard-{shard}"
            destination.mkdir(parents=True, exist_ok=True)
            for p in tmp_path.glob("ddp-continual-*.json"):
                shutil.copy2(p, destination / p.name)
    assert len(_matrix_cases()) == 36
    expected_indices = [index for index in range(46) if index % SHARDS == shard]
    for rank in [0, 1]:
        results = json.loads((tmp_path / f"ddp-continual-{rank}.json").read_text())
        assert [row["matrix_index"] for row in results] == expected_indices
