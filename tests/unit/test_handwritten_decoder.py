"""Exact decoding, useful parameter count and standard checkpoint inference."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.experimental.weight_space.handwritten_decoder import (  # noqa: E402
    CODE_BITS,
    HIDDEN,
    MAX_ERRORS,
    MIN_DISTANCE,
    PARAMETER_COUNT,
    make_codebook,
    make_decoder,
    message_bits,
    read_messages,
)


def test_parameter_count_and_no_randomness_or_unused_padding() -> None:
    before = torch.get_rng_state().clone()
    model = make_decoder()
    assert torch.equal(before, torch.get_rng_state())
    assert sum(p.numel() for p in model.parameters()) == PARAMETER_COUNT
    assert (
        sum(int(p.count_nonzero()) for p in model.parameters()) == PARAMETER_COUNT - 8
    )
    assert [type(m) for m in model] == [torch.nn.Linear, torch.nn.ReLU, torch.nn.Linear]
    assert all(p.requires_grad for p in model.parameters())


def test_codebook_has_minimum_distance_192_and_represents_every_byte() -> None:
    code = make_codebook()
    distances = (CODE_BITS - code @ code.T) / 2
    distances.fill_diagonal_(CODE_BITS)
    assert int(distances.min()) == MIN_DISTANCE
    assert set(code.flatten().tolist()) == {-1.0, 1.0}
    assert torch.equal(read_messages(make_decoder()(code)), torch.arange(HIDDEN))


@pytest.mark.parametrize("errors", [1, 32, 64, 94, 95])
def test_corrects_random_corruptions_of_every_message(errors: int) -> None:
    code = make_codebook()
    generator = torch.Generator().manual_seed(108 + errors)
    positions = torch.rand(HIDDEN, CODE_BITS, generator=generator).argsort(1)[
        :, :errors
    ]
    received = code.clone()
    received.scatter_(1, positions, -received.gather(1, positions))
    model = make_decoder()
    with torch.no_grad():
        active = model[:2](received)
        assert torch.equal((active > 0).sum(1), torch.ones(HIDDEN, dtype=torch.long))
        assert torch.equal(active.diag(), torch.full((HIDDEN,), 192.0 - 2 * errors))
        assert torch.equal(read_messages(model(received)), torch.arange(HIDDEN))


def test_nearest_rival_boundary_is_correct_at_95_and_erases_at_96() -> None:
    code = make_codebook()
    model = make_decoder()
    for message in range(HIDDEN):
        rival = message ^ 1
        different = torch.where(code[message] != code[rival])[0]
        assert len(different) == MIN_DISTANCE
        received = code[message].clone()
        received[different[:MAX_ERRORS]] *= -1
        assert read_messages(model(received)).item() == message
        received[different[MAX_ERRORS]] *= -1
        assert read_messages(model(received)).item() == -1


def test_all_256_hidden_units_and_all_eight_output_bits_are_used() -> None:
    model = make_decoder()
    code = make_codebook()
    active = model[:2](code)
    assert torch.equal(active, 192 * torch.eye(HIDDEN))
    assert torch.equal(model(code) > 0, message_bits().bool())


def test_random_baseline_is_reproducible_and_does_not_reset_caller_rng() -> None:
    before = torch.get_rng_state().clone()
    a, b, c = (make_decoder(handwritten=False, seed=s) for s in (11, 11, 12))
    assert torch.equal(before, torch.get_rng_state())
    assert sum(p.numel() for p in a.parameters()) == PARAMETER_COUNT
    assert all(
        torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters(), strict=True)
    )
    assert not torch.equal(a[0].weight, c[0].weight)


def test_checkpoint_loads_as_an_ordinary_network(tmp_path: Path) -> None:
    model = make_decoder()
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    loaded = make_decoder(handwritten=False)
    loaded.load_state_dict(torch.load(path, weights_only=True))
    assert torch.equal(read_messages(loaded(make_codebook())), torch.arange(HIDDEN))
