"""State equality hashes are exact, bounded and independent of layout/device."""

import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training.state_fingerprint import state_fingerprint  # noqa: E402


def test_identity_modules_import_without_torch():
    script = "import sys; import stateset_agents.training.state_fingerprint; import stateset_agents.training.scheduler_identity; import stateset_agents.training.replica_consensus; assert 'torch' not in sys.modules"
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int64,
        torch.bool,
        torch.complex64,
    ],
)
def test_chunk_size_and_noncontiguous_layout_do_not_change_identity(dtype):
    values = torch.arange(120).reshape(4, 5, 6).to(dtype).transpose(0, 2)
    state = {"values": values, "step": 2**60 + 1, "nested": [None, (True, "x")]}
    expected = state_fingerprint(state, chunk_bytes=4096)
    for bound in [1, 8, 32, 128]:
        assert state_fingerprint(state, chunk_bytes=bound) == expected
    assert state_fingerprint({**state, "values": values.contiguous()}) == expected


def test_mapping_order_is_irrelevant_but_types_and_exact_values_matter():
    first = {"a": [1, 2.0], 3: torch.tensor([0.0, 1.0])}
    assert state_fingerprint(first) == state_fingerprint({3: first[3], "a": first["a"]})
    for changed in [
        {"a": (1, 2.0), 3: first[3]},
        {"a": [1.0, 2.0], 3: first[3]},
        {"a": first["a"], 3: torch.tensor([-0.0, 1.0])},
        {"a": first["a"], 3: first[3].double()},
        {"a": first["a"], 3: first[3].reshape(1, 2)},
    ]:
        assert state_fingerprint(changed) != state_fingerprint(first)


def test_empty_scalar_and_repeated_views_have_stable_values():
    for tensor in [torch.tensor(3.0), torch.empty(0), torch.arange(3.0).expand(7, -1)]:
        assert state_fingerprint(tensor, chunk_bytes=4) == state_fingerprint(
            tensor.clone(), chunk_bytes=128
        )


def test_chunk_views_bound_temporary_tensor_storage(monkeypatch):
    from stateset_agents.training import grpo_accumulation

    original = grpo_accumulation._gradient_chunks
    sizes = []

    def observed(tensor, maximum):
        for piece in original(tensor, maximum):
            sizes.append(piece.numel() * piece.element_size())
            yield piece

    monkeypatch.setattr(grpo_accumulation, "_gradient_chunks", observed)
    tensor = torch.arange(1000.0).reshape(20, 50).t()
    state_fingerprint(tensor, chunk_bytes=32)
    assert len(sizes) > 1 and max(sizes) <= 32


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        torch.tensor([float("nan")]),
        torch.tensor([float("inf")]),
        object(),
        torch.empty(2, device="meta"),
        torch.eye(2).to_sparse(),
    ],
)
def test_invalid_or_opaque_state_rejects(value):
    with pytest.raises(ValueError):
        state_fingerprint(value)


def test_cyclic_state_rejects():
    value = []
    value.append(value)
    with pytest.raises(ValueError, match="Cyclic"):
        state_fingerprint(value)
