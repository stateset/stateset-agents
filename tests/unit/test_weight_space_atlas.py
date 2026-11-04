"""Tests for the weight-only atlas (which neurons write toward which tokens)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.experimental.weight_space.atlas import (  # noqa: E402
    logit_direction,
    neuron_readout,
    token_neighbours,
    top_writers,
)


def test_logit_direction_applies_the_final_norm_gain() -> None:
    embed = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    norm = torch.tensor([0.5, -0.5])
    # Zero-centred RMSNorm (Qwen3.5, Gemma): output * (1 + weight)
    assert torch.equal(logit_direction(embed, [1], norm), torch.tensor([4.5, 2.0]))
    # Plain RMSNorm: output * weight
    assert torch.equal(
        logit_direction(embed, [1], norm, zero_centered=False),
        torch.tensor([1.5, -2.0]),
    )
    # Several ids: the mean direction
    assert torch.equal(
        logit_direction(embed, [0, 1], torch.zeros(2)), torch.tensor([2.0, 3.0])
    )


def test_top_writers_finds_the_planted_neuron() -> None:
    g = torch.Generator().manual_seed(0)
    target = torch.zeros(8)
    target[0] = 1.0
    layers = {
        3: torch.randn(8, 16, generator=g) * 0.1,
        4: torch.randn(8, 16, generator=g) * 0.1,
    }
    layers[3][:, 5] = 0.0
    layers[3][0, 5] = 3.0  # neuron 5 of layer 3 writes exactly along the target
    best = top_writers(layers, target, k=2)
    assert (best[0].layer, best[0].neuron) == (3, 5)
    assert best[0].alignment == pytest.approx(1.0)
    assert best[0].alignment >= best[1].alignment
    # Negative k-th: neurons that write *against* the target
    worst = top_writers(layers, -target, k=1)
    assert worst[0].alignment > 0


def test_neuron_readout_lists_the_tokens_a_neuron_promotes() -> None:
    embed = torch.eye(5)
    column = torch.tensor([0.0, 0.0, 2.0, 0.0, 1.0])
    assert neuron_readout(column, embed, torch.zeros(5), k=2) == [2, 4]


def test_token_neighbours_excludes_the_query_tokens() -> None:
    embed = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [1.0, 0.05]])
    assert token_neighbours(embed, [0], k=2) == [3, 1]


def _readouts(vocab: int = 12, hidden: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    embed = torch.randn(vocab, hidden, generator=g)
    norm = torch.randn(hidden, generator=g) * 0.3
    return embed, norm, embed * (1.0 + norm)


def test_protected_write_moves_the_target_and_leaves_protected_tokens_alone() -> None:
    from stateset_agents.experimental.weight_space.atlas import protected_write

    embed, norm, readout = _readouts()
    v = protected_write(embed, 3, norm, protect=[0, 5, 7])
    assert float(v.norm()) == pytest.approx(1.0, abs=1e-6)
    assert float(readout[3] @ v) > 0
    for p in (0, 5, 7):
        assert abs(float(readout[p] @ v)) < 1e-5


def test_protected_write_without_protection_is_the_unit_readout() -> None:
    from stateset_agents.experimental.weight_space.atlas import (
        logit_direction,
        protected_write,
    )

    embed, norm, _ = _readouts()
    plain = logit_direction(embed, [3], norm)
    v = protected_write(embed, 3, norm, protect=[])
    assert torch.allclose(v, plain / plain.norm(), atol=1e-6)


@pytest.mark.parametrize("protect", [[3], list(range(8))])
def test_protected_write_refuses_impossible_protection(protect: list[int]) -> None:
    from stateset_agents.experimental.weight_space.atlas import protected_write

    embed, norm, _ = _readouts()
    with pytest.raises(ValueError):
        protected_write(embed, 3, norm, protect=protect)


def test_quiet_direction_avoids_readouts_and_the_residuals_own_use() -> None:
    from stateset_agents.experimental.weight_space.atlas import quiet_direction

    g = torch.Generator().manual_seed(1)
    hidden = 6
    readout = torch.randn(50, hidden, generator=g)
    readout[:, 4:] *= 0.01  # dims 4 and 5 are nearly invisible to the unembedding
    residuals = torch.randn(400, hidden, generator=g)
    residuals[:, 4] *= 5.0  # the model uses dim 4 a lot, dim 5 barely
    residuals[:, 5] *= 0.01
    w = quiet_direction(readout, residuals, k=2)
    assert float(w.norm()) == pytest.approx(1.0, abs=1e-6)
    assert abs(float(w[5])) > 0.99


def test_quiet_direction_refuses_k_outside_the_hidden_size() -> None:
    from stateset_agents.experimental.weight_space.atlas import quiet_direction

    with pytest.raises(ValueError):
        quiet_direction(torch.randn(10, 4), torch.randn(20, 4), k=5)
