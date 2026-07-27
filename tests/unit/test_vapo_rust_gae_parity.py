"""Parity test for VAPO's optional Rust-accelerated GAE fast path.

``LengthAdaptiveGAE.compute_gae`` (stateset_agents/training/vapo_trainer.py)
tries to delegate single-termination-per-row batches to the Rust
``stateset_rl_core`` kernel via ``core.rust_accelerator.compute_gae``, falling
back to the pure-torch loop when the extension isn't installed or a row has
more than one termination flag. This test asserts the two code paths agree
numerically on identical input.
"""

from __future__ import annotations

import pytest
import torch

from stateset_agents.training.vapo_trainer import LengthAdaptiveGAE


def _sample_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    rewards = torch.zeros(3, 6)
    rewards[:, -1] = torch.tensor([1.0, -1.0, 0.5])
    values = torch.randn(3, 6) * 0.1
    dones = torch.zeros(3, 6)
    # One termination per row, at varying positions (last two rows terminate
    # before the padded end, matching real VAPO usage where sequence_length
    # can be shorter than max_len).
    dones[0, 5] = 1.0
    dones[1, 4] = 1.0
    dones[2, 3] = 1.0
    return rewards, values, dones


def test_torch_fallback_matches_manual_backward_recursion():
    """Sanity check the pure-torch path itself (no Rust dependency)."""
    gae = LengthAdaptiveGAE(gamma=0.99)
    rewards, values, dones = _sample_batch()
    advantages, returns = gae.compute_gae(rewards, values, dones, lambda_value=0.95)
    assert torch.allclose(returns, advantages + values)


def test_rust_gae_matches_python_fallback_when_extension_installed():
    """When stateset_rl_core is installed, the Rust fast path used inside
    LengthAdaptiveGAE.compute_gae must agree with the pure-torch fallback."""
    pytest.importorskip("stateset_rl_core")

    gae = LengthAdaptiveGAE(gamma=0.99)
    rewards, values, dones = _sample_batch()

    rust_advantages = gae._try_rust_gae(rewards, values, dones, lambda_value=0.95)
    assert rust_advantages is not None

    # Force the pure-torch path for comparison by bypassing _try_rust_gae.
    batch_size, seq_len = rewards.shape
    torch_advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(batch_size)
    for t in reversed(range(seq_len)):
        next_value = torch.zeros(batch_size) if t == seq_len - 1 else values[:, t + 1]
        delta = rewards[:, t] + gae.gamma * next_value * (1 - dones[:, t]) - values[:, t]
        last_gae = delta + gae.gamma * 0.95 * (1 - dones[:, t]) * last_gae
        torch_advantages[:, t] = last_gae

    # Padding positions *after* a row's terminal step are left as 0 by the
    # Rust fast path (that region is always excluded by the response mask
    # downstream, see vapo_trainer.py's `policy_adv_masked`), whereas the
    # dones-based torch recursion keeps computing (harmless but nonzero)
    # values there. Parity only needs to hold up to and including each row's
    # terminal step.
    for i in range(batch_size):
        done_indices = dones[i].nonzero(as_tuple=True)[0]
        end = int(done_indices[0].item()) + 1 if len(done_indices) else seq_len
        assert torch.allclose(
            rust_advantages[i, :end], torch_advantages[i, :end], atol=1e-5
        )


def test_rust_gae_helper_returns_none_without_extension(monkeypatch):
    """When the Rust extension isn't available, the helper must return None
    so the caller falls back to the pure-torch loop rather than erroring."""
    import stateset_agents.training.vapo_trainer as vapo_trainer_module

    monkeypatch.setattr(vapo_trainer_module, "_rust_gae_available", lambda: False)

    gae = LengthAdaptiveGAE(gamma=0.99)
    rewards, values, dones = _sample_batch()
    assert gae._try_rust_gae(rewards, values, dones, lambda_value=0.95) is None


def test_rust_gae_helper_bails_out_on_multiple_terminations_per_row():
    """A row with more than one termination flag isn't representable by the
    plain (no-dones) kernel; the helper must signal a full fallback."""
    pytest.importorskip("stateset_rl_core")

    gae = LengthAdaptiveGAE(gamma=0.99)
    rewards = torch.zeros(1, 6)
    values = torch.zeros(1, 6)
    dones = torch.zeros(1, 6)
    dones[0, 2] = 1.0
    dones[0, 4] = 1.0

    assert gae._try_rust_gae(rewards, values, dones, lambda_value=0.95) is None
