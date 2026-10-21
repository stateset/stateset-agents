"""VAPO correctness against the paper (Yue et al., 2025).

- gamma defaults to 1.0 (VC-PPO/VAPO), so a terminal reward is not shrunk
  geometrically toward the first response token;
- length-adaptive lambda is ``1 - 1/(alpha * l)``, floored at 0;
- the value loss trains only the value head, never the policy backbone;
- rollout-time values are taken with value-head dropout off;
- the loss is a token-mean over the whole batch, not a mean of per-prompt
  token-means (which re-weights tokens by their group's length).
"""

import pytest
import torch

pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402

from stateset_agents.training.vapo_trainer import (  # noqa: E402
    LengthAdaptiveGAE,
    VAPOConfig,
    VAPOTrainer,
)


def _trainer(**overrides):
    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(
            n_embd=32,
            n_layer=1,
            n_head=2,
            vocab_size=100,
            n_positions=64,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )
    config = VAPOConfig(
        model_name="gpt2",
        group_size=2,
        per_device_train_batch_size=2,
        **overrides,
    )
    trainer = VAPOTrainer(
        config=config, model=model, tokenizer=None, reward_fn=lambda p, r: 1.0
    )
    trainer.value_warmup_complete = True
    return trainer


def _fake_groups(lengths: dict[str, int], prompt_length: int = 3):
    async def generate(prompt):
        n = lengths[prompt]
        out = []
        for _ in range(2):
            ids = torch.randint(0, 100, (prompt_length + n,))
            mask = torch.zeros(prompt_length + n)
            mask[prompt_length:] = 1.0
            out.append(
                {
                    "input_ids": ids,
                    "attention_mask": torch.ones(prompt_length + n, dtype=torch.long),
                    "response_mask": mask,
                    "prompt_length": prompt_length,
                    "sequence_length": n,
                    "response": "ok",
                }
            )
        return out

    return generate


def test_gamma_defaults_to_one_and_reaches_gae():
    assert VAPOConfig().gamma == 1.0
    assert _trainer().gae_computer.gamma == 1.0
    assert _trainer(gamma=0.9).gae_computer.gamma == 0.9


def test_terminal_reward_is_not_discounted_by_default():
    trainer = _trainer()
    response_mask = torch.zeros(1, 301)
    response_mask[0, 1:] = 1.0
    targets, action_mask = trainer.monte_carlo_value_targets(
        torch.tensor([1.0]), response_mask
    )
    first = int(action_mask[0].nonzero()[0])
    assert targets[0, first].item() == pytest.approx(1.0)


def test_length_adaptive_lambda_matches_paper():
    gae = LengthAdaptiveGAE(lambda_policy_alpha=0.05)
    assert gae.compute_lambda_policy(100) == pytest.approx(1 - 1 / 5)
    assert gae.compute_lambda_policy(20) == pytest.approx(0.0)
    # alpha * l < 1 would give a negative lambda; it is floored at 0.
    assert gae.compute_lambda_policy(10) == 0.0
    assert gae.compute_lambda_policy(0) == 0.0


def test_value_loss_gradient_does_not_reach_policy_backbone():
    trainer = _trainer()
    ids = torch.randint(0, 100, (2, 8))
    values = trainer.compute_values(ids, torch.ones_like(ids))
    values.pow(2).sum().backward()
    assert all(p.grad is None or not p.grad.any() for p in trainer.model.parameters())
    assert any(
        p.grad is not None and p.grad.any() for p in trainer.value_head.parameters()
    )


@pytest.mark.asyncio
async def test_rollout_values_are_computed_without_value_head_dropout(monkeypatch):
    trainer = _trainer()
    monkeypatch.setattr(
        trainer, "generate_group_responses", _fake_groups({"a": 4, "b": 4})
    )
    modes: list[tuple[bool, bool]] = []
    original = trainer.value_head.forward

    def spy(hidden_states):
        modes.append((torch.is_grad_enabled(), trainer.value_head.training))
        return original(hidden_states)

    monkeypatch.setattr(trainer.value_head, "forward", spy)
    await trainer.train_step(["a", "b"])

    rollout_modes = [training for grad, training in modes if not grad]
    update_modes = [training for grad, training in modes if grad]
    assert rollout_modes and not any(rollout_modes)
    assert update_modes and all(update_modes)
    assert trainer.value_head.training


@pytest.mark.asyncio
async def test_losses_are_token_means_over_the_whole_batch(monkeypatch):
    trainer = _trainer()
    lengths = {"short": 2, "long": 8}
    monkeypatch.setattr(trainer, "generate_group_responses", _fake_groups(lengths))
    captured: list[tuple[torch.Tensor, ...]] = []
    original = trainer.compute_vapo_losses

    def spy(*args, **kwargs):
        losses = original(*args, **kwargs)
        for loss in losses:
            if loss.requires_grad:
                loss.retain_grad()
        captured.append(losses)
        return losses

    monkeypatch.setattr(trainer, "compute_vapo_losses", spy)
    await trainer.train_step(["short", "long"])

    tokens = [2 * lengths["short"], 2 * lengths["long"]]
    total = sum(tokens)
    cfg = trainer.config
    for (policy, value, positive), n in zip(captured, tokens, strict=True):
        share = n / total
        assert policy.grad.item() == pytest.approx(share)
        assert value.grad.item() == pytest.approx(cfg.value_loss_coef * share)
        # Every sample is correct, so positive tokens == response tokens.
        assert positive.grad.item() == pytest.approx(cfg.positive_lm_weight * share)


def test_from_training_config_keeps_vapo_gamma_unless_base_chose_one():
    from stateset_agents.training.config import TrainingConfig
    from stateset_agents.training.vapo_config import VAPOConfig

    assert VAPOConfig.from_training_config(TrainingConfig()).gamma == 1.0
    assert VAPOConfig.from_training_config(TrainingConfig(gamma=0.9)).gamma == 0.9
    assert VAPOConfig.from_training_config(TrainingConfig(), gamma=0.95).gamma == 0.95


def test_zero_length_adaptive_lambda_warns_once(caplog):
    import logging

    from stateset_agents.training.vapo_trainer import LengthAdaptiveGAE

    gae = LengthAdaptiveGAE(lambda_policy_alpha=0.05)
    with caplog.at_level(logging.WARNING):
        assert gae.compute_lambda_policy(10) == 0.0
        assert gae.compute_lambda_policy(5) == 0.0
        assert gae.compute_lambda_policy(100) > 0.0
    warnings = [r for r in caplog.records if "lambda_policy_alpha" in r.getMessage()]
    assert len(warnings) == 1
