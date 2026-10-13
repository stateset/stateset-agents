"""Decision Transformer must ignore padded timesteps when given a mask."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.experimental.offline_rl.decision_transformer import (  # noqa: E402
    DecisionTransformer,
    DecisionTransformerConfig,
)


def _model() -> DecisionTransformer:
    torch.manual_seed(0)
    config = DecisionTransformerConfig(
        state_dim=6, action_dim=3, max_context_length=5, n_embd=16, n_layer=2, n_head=2
    )
    model = DecisionTransformer(config)
    model.eval()
    return model


def _batch(seed: int) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator().manual_seed(seed)
    states = torch.randn(1, 5, 6, generator=gen)
    actions = torch.randn(1, 5, 3, generator=gen)
    rtg = torch.randn(1, 5, 1, generator=gen)
    timesteps = torch.arange(5).unsqueeze(0)
    return states, actions, rtg, timesteps


def test_padded_positions_do_not_change_valid_predictions() -> None:
    model = _model()
    mask = torch.tensor([[0, 0, 1, 1, 1]])  # left padding, three real steps
    states, actions, rtg, timesteps = _batch(1)
    other_states, other_actions, other_rtg, _ = _batch(2)
    # Same real steps, different garbage in the two padded steps.
    other_states[:, 2:] = states[:, 2:]
    other_actions[:, 2:] = actions[:, 2:]
    other_rtg[:, 2:] = rtg[:, 2:]

    with torch.no_grad():
        a, s, r = model(states, actions, rtg, timesteps, attention_mask=mask)
        a2, s2, r2 = model(
            other_states, other_actions, other_rtg, timesteps, attention_mask=mask
        )

    assert torch.allclose(a[:, 2:], a2[:, 2:], atol=1e-6)
    assert torch.allclose(s[:, 2:], s2[:, 2:], atol=1e-6)
    assert torch.allclose(r[:, 2:], r2[:, 2:], atol=1e-6)
    assert torch.isfinite(a).all() and torch.isfinite(s).all()


def test_full_mask_matches_no_mask() -> None:
    model = _model()
    states, actions, rtg, timesteps = _batch(3)
    with torch.no_grad():
        plain = model(states, actions, rtg, timesteps)
        masked = model(states, actions, rtg, timesteps, attention_mask=torch.ones(1, 5))
    for x, y in zip(plain, masked, strict=True):
        assert torch.allclose(x, y, atol=1e-6)
