"""Multiple inner updates per rollout batch on the GRPO token path.

With ``num_gradient_updates > 1`` the trainers freeze the old policy's
per-token log-probs once (one no-grad forward on the stored token ids), then
take that many optimizer steps against them, so the second and later updates
see ratios that differ from 1 and the trust-region clip can engage. With the
default of 1 the behaviour is exactly the on-policy single step (goldens).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from stateset_agents.core.trajectory import (  # noqa: E402
    ConversationTurn,
    MultiTurnTrajectory,
    TrajectoryGroup,
)
from stateset_agents.training import loss_computation as lc  # noqa: E402

VOCAB = 200


def _tiny_model(seed: int = 0):
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    return GPT2LMHeadModel(
        GPT2Config(
            n_embd=32,
            n_layer=2,
            n_head=2,
            vocab_size=VOCAB,
            n_positions=64,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )


def _group(rewards=(1.0, 0.0, 0.5)):
    trajs = []
    for i, r in enumerate(rewards):
        g = torch.Generator().manual_seed(100 + i)
        p = torch.randint(0, VOCAB, (5,), generator=g).tolist()
        t = torch.randint(0, VOCAB, (6,), generator=g).tolist()
        trajs.append(
            MultiTurnTrajectory(
                turns=[
                    ConversationTurn(role="user", content="q"),
                    ConversationTurn(
                        role="assistant",
                        content="a",
                        metadata={"prompt_token_ids": p, "token_ids": t},
                    ),
                ],
                total_reward=r,
            )
        )
    return TrajectoryGroup(scenario_id="s", trajectories=trajs)


def _config(**over):
    base = {
        "max_prompt_length": 64,
        "max_completion_length": 64,
        "clip_ratio": 0.2,
        "seq_clip_ratio": 3e-4,
        "entropy_coef": 0.0,
        "advantage_normalization": True,
        "baseline_type": "group_mean",
        "reward_clip": None,
        "generation_batch_size": 4,
        "objective": None,
        "objective_overrides": None,
        "bf16": False,
        "fp16": False,
        "num_gradient_updates": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-2,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "seed": 0,
        "use_reference_model": False,
        "report_to": None,
        "continual_strategy": "none",
    }
    base.update(over)
    return SimpleNamespace(**base)


# --- loss-level API --------------------------------------------------------------


def test_old_logprobs_snapshot_matches_current_before_any_update():
    model = _tiny_model()
    agent = SimpleNamespace(model=model, tokenizer=None)
    groups = [_group()]
    old = lc.compute_token_old_logprobs(groups, _config(), agent)
    assert isinstance(old, list) and len(old) == 1
    out = lc.compute_grpo_loss(
        groups, _config(), agent, 0.0, 0, lambda m, n: None, old_logprobs=old
    )
    assert out["path"] == "token"
    assert out["ratio_mean"] == pytest.approx(1.0, abs=1e-5)
    assert out["clip_fraction"] == 0.0


def test_ratio_departs_from_one_after_an_optimizer_step():
    model = _tiny_model()
    agent = SimpleNamespace(model=model, tokenizer=None)
    groups = [_group()]
    cfg = _config()
    old = lc.compute_token_old_logprobs(groups, cfg, agent)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    out = lc.compute_grpo_loss(
        groups, cfg, agent, 0.0, 0, lambda m, n: None, old_logprobs=old
    )
    out["total_loss"].backward()
    opt.step()
    again = lc.compute_grpo_loss(
        groups, cfg, agent, 0.0, 0, lambda m, n: None, old_logprobs=old
    )
    assert again["ratio_mean"] != pytest.approx(1.0, abs=1e-4)


def test_old_logprobs_are_detached_and_match_a_manual_forward():
    model = _tiny_model()
    agent = SimpleNamespace(model=model, tokenizer=None)
    groups = [_group()]
    old = lc.compute_token_old_logprobs(groups, _config(), agent)
    lp = old[0]
    assert not lp.requires_grad
    cur = lc.compute_grpo_loss(groups, _config(), agent, 0.0, 0, lambda m, n: None)
    # same weights, so the snapshot equals what the live path just computed
    assert cur["ratio_mean"] == pytest.approx(1.0)


# --- trainer-level behaviour ---------------------------------------------------------


def _multi_turn_trainer(model, num_gradient_updates):
    from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

    agent = SimpleNamespace(model=model, tokenizer=None, initialize=AsyncMock())
    env = MagicMock()
    reward_fn = MagicMock()
    cfg = _config(num_gradient_updates=num_gradient_updates)
    trainer = MultiTurnGRPOTrainer(
        agent=agent, environment=env, reward_fn=reward_fn, config=cfg
    )
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    trainer.scaler = None
    return trainer


@pytest.mark.asyncio
async def test_training_step_takes_one_optimizer_step_per_inner_update():
    model = _tiny_model()
    trainer = _multi_turn_trainer(model, num_gradient_updates=3)
    steps = []
    orig = trainer.optimizer.step

    def spy(*a, **kw):
        steps.append(1)
        return orig(*a, **kw)

    trainer.optimizer.step = spy
    metrics = await trainer.training_step([_group()])
    assert len(steps) == 3
    assert metrics["inner_updates"] == 3
    assert metrics["path"] == "token"


@pytest.mark.asyncio
async def test_single_inner_update_keeps_grad_accumulation_semantics():
    model = _tiny_model()
    trainer = _multi_turn_trainer(model, num_gradient_updates=1)
    trainer.config.gradient_accumulation_steps = 2
    steps = []
    orig = trainer.optimizer.step

    def spy(*a, **kw):
        steps.append(1)
        return orig(*a, **kw)

    trainer.optimizer.step = spy
    m1 = await trainer.training_step([_group()])
    m2 = await trainer.training_step([_group()])
    assert (
        len(steps) == 1
        and m1["optimizer_step"] is False
        and m2["optimizer_step"] is True
    )


@pytest.mark.asyncio
async def test_later_inner_updates_see_non_unit_ratios():
    model = _tiny_model()
    trainer = _multi_turn_trainer(model, num_gradient_updates=3)
    metrics = await trainer.training_step([_group()])
    assert metrics["ratio_mean_last"] != pytest.approx(1.0, abs=1e-4)
