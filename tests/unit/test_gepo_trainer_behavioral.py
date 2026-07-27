import torch

from stateset_agents.training.gepo_trainer import GEPOTrainer


def test_gepo_coefficient_no_underflow_long_sequences():
    """Sums of token log probs for realistic sequences (~ -600 nats) must not
    produce 0/NaN coefficients."""
    learner = torch.tensor([-600.0, -610.0, -605.0, -595.0])
    sampler = torch.tensor([-601.0, -609.0, -606.0, -594.0])
    coef = GEPOTrainer.compute_gepo_coefficient_static(learner, sampler)
    assert torch.isfinite(coef).all()
    assert (coef > 0).all()


def test_gepo_coefficient_matches_linear_space_on_small_values():
    """On numerically safe values, log-space result equals the linear formula
    coef_i = p_i / E_qhat[q], E_qhat[q] = sum(q^2)/sum(q)."""
    learner_lp = torch.log(torch.tensor([0.30, 0.20, 0.10]))
    sampler_lp = torch.log(torch.tensor([0.25, 0.25, 0.10]))
    q = sampler_lp.exp()
    expected = learner_lp.exp() / ((q * q).sum() / q.sum())
    got = GEPOTrainer.compute_gepo_coefficient_static(learner_lp, sampler_lp)
    assert torch.allclose(got, expected, rtol=1e-5)


def test_response_mask_offset_matches_gspo_convention():
    """With prompt length P, the shifted-label mask must start at P-1."""
    mask = GEPOTrainer.build_response_mask(
        attention_mask=torch.ones(1, 10, dtype=torch.long), response_start_idx=4
    )
    # shifted axis has length 9; positions 0..2 are prompt-only, 3.. are response
    assert mask.shape == (1, 9)
    assert mask[0, :3].sum() == 0
    assert mask[0, 3:].sum() == 6


def test_response_mask_offset_clamped_at_zero():
    """response_start_idx of 0 should not underflow to a negative index."""
    mask = GEPOTrainer.build_response_mask(
        attention_mask=torch.ones(1, 5, dtype=torch.long), response_start_idx=0
    )
    assert mask.shape == (1, 4)
    assert mask.sum() == 4
