"""Checkpoints restore everything a resumed run needs, including under DDP."""

from __future__ import annotations

import asyncio
import logging
import random
import socket
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch.distributed as dist  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402

from stateset_agents.training.checkpoint_state import (  # noqa: E402
    load_training_checkpoint,
    save_training_checkpoint,
)
from stateset_agents.training.trainer_utils import require_torch  # noqa: E402


def _tiny_model(seed: int):
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    return GPT2LMHeadModel(
        GPT2Config(n_layer=1, n_embd=16, n_head=2, vocab_size=50, n_positions=16)
    )


class _Tokenizer:
    def save_pretrained(self, path):
        pass


def _trainer(model, **extra):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = SimpleNamespace(
        agent=SimpleNamespace(model=model, tokenizer=None),
        optimizer=optimizer,
        lr_scheduler=None,
        scaler=None,
        _grad_accum_step=0,
        global_step=0,
        current_epoch=0,
        best_eval_metric=float("-inf"),
        steps_without_improvement=0,
        continual_manager=None,
        _current_task_id=None,
        config=SimpleNamespace(gradient_accumulation_steps=1),
    )
    for key, value in extra.items():
        setattr(trainer, key, value)
    return trainer


def _train_one_step(model, optimizer):
    ids = torch.tensor([[1, 2, 3]])
    model(ids, labels=ids).loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def _load(trainer, path) -> bool:
    return load_training_checkpoint(
        trainer,
        path,
        require_torch_fn=require_torch,
        trainer_exceptions=(Exception,),
        logger=logging.getLogger(__name__),
        trusted=True,
    )


@pytest.fixture
def gloo_group():
    if dist.is_initialized():
        pytest.skip("a process group is already initialized")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


# --- DDP-wrapped models ------------------------------------------------------------


def test_training_checkpoint_saves_and_restores_a_ddp_wrapped_model(
    gloo_group, tmp_path
):
    trained = DDP(_tiny_model(1))
    source = _trainer(trained, global_step=7, current_epoch=6)
    _train_one_step(trained, source.optimizer)
    save_training_checkpoint(source, tmp_path / "ck", require_torch_fn=require_torch)

    fresh = DDP(_tiny_model(2))
    target = _trainer(fresh)
    assert _load(target, tmp_path / "ck") is True
    assert torch.equal(
        trained.module.transformer.wte.weight, fresh.module.transformer.wte.weight
    )
    assert len(target.optimizer.state) > 0
    assert target.global_step == 7


def test_distributed_checkpoint_keeps_full_training_state(gloo_group, tmp_path):
    from stateset_agents.training.distributed import (
        DistributedConfig,
        DistributedTrainer,
    )

    trained = DDP(_tiny_model(1))
    trainer = _trainer(
        trained,
        global_step=100,
        current_epoch=99,
        best_eval_metric=0.5,
        is_main_process=True,
        callbacks=[],
        wandb_logger=None,
        distributed_config=DistributedConfig(
            world_size=1, checkpoint_dir=str(tmp_path), strategy="ddp"
        ),
    )
    trainer.agent.tokenizer = _Tokenizer()
    _train_one_step(trained, trainer.optimizer)
    asyncio.run(DistributedTrainer._save_distributed_checkpoint(trainer, False, None))

    fresh = DDP(_tiny_model(2))
    target = _trainer(fresh)
    assert _load(target, tmp_path / "checkpoint-100") is True
    assert torch.equal(
        trained.module.transformer.wte.weight, fresh.module.transformer.wte.weight
    )
    assert len(target.optimizer.state) > 0
    assert (target.global_step, target.current_epoch) == (100, 99)


def test_distributed_best_checkpoint_uses_the_best_name(gloo_group, tmp_path):
    from stateset_agents.training.distributed import (
        DistributedConfig,
        DistributedTrainer,
    )

    trainer = _trainer(
        DDP(_tiny_model(1)),
        global_step=5,
        is_main_process=True,
        callbacks=[],
        wandb_logger=None,
        distributed_config=DistributedConfig(
            world_size=1, checkpoint_dir=str(tmp_path), strategy="ddp"
        ),
    )
    asyncio.run(DistributedTrainer._save_distributed_checkpoint(trainer, True, None))
    assert (tmp_path / "best-checkpoint" / "training_state.pt").is_file()
    assert not (tmp_path / "checkpoint-5").exists()


def test_weights_that_match_no_model_parameter_are_not_a_successful_resume(
    tmp_path,
):
    # A legacy (pre-accumulation-schema) checkpoint whose weight names belong
    # to a differently wrapped model must not report a successful resume.
    saved = _tiny_model(1)
    ck = tmp_path / "legacy"
    ck.mkdir()
    torch.save(
        {f"wrapper.{name}": value for name, value in saved.state_dict().items()},
        ck / "pytorch_model.bin",
    )
    torch.save({"global_step": 3, "current_epoch": 2}, ck / "training_state.pt")

    fresh = _tiny_model(2)
    before = fresh.transformer.wte.weight.detach().clone()
    target = _trainer(fresh)
    assert _load(target, ck) is False
    assert torch.equal(fresh.transformer.wte.weight, before)


# --- running statistics and RNG streams -------------------------------------------


def test_running_reward_statistics_survive_a_resume(tmp_path):
    source = _trainer(
        _tiny_model(1),
        _global_reward_mean=0.25,
        _global_reward_sum=2.5,
        _global_reward_count=10,
    )
    save_training_checkpoint(source, tmp_path / "ck", require_torch_fn=require_torch)

    target = _trainer(_tiny_model(2), _global_reward_mean=0.0, _global_reward_count=0)
    assert _load(target, tmp_path / "ck") is True
    assert target._global_reward_mean == pytest.approx(0.25)
    assert target._global_reward_count == 10
    assert target._global_reward_sum == pytest.approx(2.5)


def test_resume_continues_the_saved_random_streams(tmp_path):
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    source = _trainer(_tiny_model(1))
    save_training_checkpoint(source, tmp_path / "ck", require_torch_fn=require_torch)
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))

    # A resumed process re-seeds from config.seed before loading.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    target = _trainer(_tiny_model(2))
    assert _load(target, tmp_path / "ck") is True
    resumed = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert resumed == pytest.approx(expected)


def test_checkpoints_without_new_fields_still_load(tmp_path):
    source = _trainer(_tiny_model(1), global_step=4)
    save_training_checkpoint(source, tmp_path / "ck", require_torch_fn=require_torch)
    state_path = tmp_path / "ck" / "training_state.pt"
    state = torch.load(state_path, weights_only=False)
    # Reconstruct a legacy writer's schema, rather than remove only obsolete
    # aliases while retaining a current schema's incompatible estimator state.
    for name in (
        "reward_stats",
        "rng_state",
        "learning_state",
        "rng",
        "rollout_state",
        "native_checkpoint_schema",
        "native_component_identities",
    ):
        state.pop(name, None)
    torch.save(state, state_path)
    # Publication metadata lists the file digest; refresh it for the edit.
    from stateset_agents.training import checkpoint_publication as cp

    completion = tmp_path / "ck" / cp.COMPLETION_FILE
    completion.unlink()
    files = cp._inventory(tmp_path / "ck", sync=False)
    cp._write_json(completion, {"schema_version": 1, "files": files})

    target = _trainer(_tiny_model(2), _global_reward_mean=0.0, _global_reward_count=0)
    assert _load(target, tmp_path / "ck") is True
    assert target.global_step == 4
    assert target._global_reward_count == 0


# --- data parallelism: ranks see different data -------------------------------------


def _multi_turn_trainer(seed=5):
    from unittest.mock import AsyncMock, MagicMock

    from stateset_agents.training.config import TrainingConfig
    from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

    agent = SimpleNamespace(model=None, tokenizer=None, initialize=AsyncMock())
    return MultiTurnGRPOTrainer(
        agent=agent,
        environment=MagicMock(),
        reward_fn=MagicMock(),
        config=TrainingConfig(seed=seed, num_episodes=8),
    )


def test_single_process_scenario_order_is_unchanged():
    trainer = _multi_turn_trainer()
    scenarios = [{"id": i} for i in range(8)]
    assert [trainer._episode_scenario(scenarios, e)["id"] for e in range(8)] == list(
        range(8)
    )


def test_ranks_draw_disjoint_scenarios(monkeypatch):
    scenarios = [{"id": i} for i in range(8)]
    picked = {}
    for rank in (0, 1):
        trainer = _multi_turn_trainer()
        monkeypatch.setattr(trainer, "_data_parallel_rank_world", lambda r=rank: (r, 2))
        picked[rank] = [trainer._episode_scenario(scenarios, e)["id"] for e in range(4)]
    assert picked == {0: [0, 2, 4, 6], 1: [1, 3, 5, 7]}


def test_ranks_seed_distinct_random_streams(monkeypatch):
    draws = {}
    for rank in (0, 1):
        trainer = _multi_turn_trainer(seed=5)
        monkeypatch.setattr(trainer, "_data_parallel_rank_world", lambda r=rank: (r, 2))
        trainer._seed_everything(torch)
        draws[rank] = (random.random(), float(torch.rand(())))
    assert draws[0] != draws[1]

    single = _multi_turn_trainer(seed=5)
    single._seed_everything(torch)
    random.seed(5)
    torch.manual_seed(5)
    expected = (random.random(), float(torch.rand(())))
    single._seed_everything(torch)
    assert (random.random(), float(torch.rand(()))) == expected


def test_distributed_trainer_reports_its_configured_rank():
    from stateset_agents.training.distributed import (
        DistributedConfig,
        DistributedTrainer,
    )

    trainer = object.__new__(DistributedTrainer)
    trainer.distributed_config = DistributedConfig(world_size=4, rank=3)
    assert trainer._data_parallel_rank_world() == (3, 4)
    assert trainer._restore_rng_on_resume() is False
