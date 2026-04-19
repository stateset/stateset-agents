"""Regression tests for EMA serialization helpers."""

from __future__ import annotations

import torch

from stateset_agents.training.ema import MultiEMA


def test_multi_ema_state_dict_uses_string_decay_keys() -> None:
    model = torch.nn.Linear(2, 2)

    ema = MultiEMA(model, decays=[0.9, 0.99], cpu_offload=True)

    state = ema.state_dict()

    assert set(state.keys()) == {"0.9", "0.99"}


def test_multi_ema_load_state_dict_accepts_serialized_decay_keys() -> None:
    model = torch.nn.Linear(2, 2)
    source = MultiEMA(model, decays=[0.9], cpu_offload=True)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    source.update()

    restored_model = torch.nn.Linear(2, 2)
    restored = MultiEMA(restored_model, decays=[0.9], cpu_offload=True)
    restored.load_state_dict(source.state_dict())

    source_state = source.get_ema(0.9).state_dict()
    restored_state = restored.get_ema(0.9).state_dict()

    assert restored_state["step"] == source_state["step"]
    assert restored_state["decay"] == source_state["decay"]
    assert restored_state["shadow_params"].keys() == source_state["shadow_params"].keys()
    for name, tensor in source_state["shadow_params"].items():
        assert torch.equal(restored_state["shadow_params"][name], tensor)
