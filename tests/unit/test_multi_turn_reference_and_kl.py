"""Multi-turn KL regularization: memory-safe LoRA reference and objective-true KL."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import torch

from stateset_agents.training import loss_computation
from stateset_agents.training.multi_turn_trainer import (
    AdapterDisabledReference,
    build_reference_model,
)


class _FakePeftModel(torch.nn.Module):
    """Adds ``delta`` to its output unless adapters are disabled."""

    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(2, 2)
        self.delta = torch.nn.Parameter(torch.ones(2))
        self.adapters_enabled = True
        self.disable_calls = 0

    @contextmanager
    def disable_adapter(self):  # noqa: ANN201 - mirrors peft's API
        self.disable_calls += 1
        self.adapters_enabled = False
        try:
            yield
        finally:
            self.adapters_enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        return out + self.delta if self.adapters_enabled else out


def test_peft_policy_reuses_frozen_base_instead_of_copying() -> None:
    policy = _FakePeftModel()

    reference = build_reference_model(policy)

    assert isinstance(reference, AdapterDisabledReference)
    x = torch.ones(1, 2)
    with torch.no_grad():
        assert torch.allclose(reference(x), policy.base(x))
    assert policy.disable_calls == 1
    assert policy.adapters_enabled  # restored for the policy forward
    assert list(reference.parameters()) == []


def test_plain_policy_gets_a_frozen_copy() -> None:
    policy = torch.nn.Linear(2, 2)

    reference = build_reference_model(policy)

    assert reference is not policy
    assert not reference.training
    assert all(not p.requires_grad for p in reference.parameters())
    assert all(p.requires_grad for p in policy.parameters())


def _kl_kind(objective_name: str | None) -> str:
    config = SimpleNamespace(objective=objective_name, objective_overrides=None)
    return loss_computation._token_kl_objective(
        loss_computation._token_objective(config), config, beta=0.05
    ).kl


def test_selected_objective_keeps_its_own_kl_estimator() -> None:
    assert _kl_kind("gspo") == "k3_sequence"


def test_native_token_objective_uses_token_kl() -> None:
    assert _kl_kind(None) == "k3_token"


def test_kl_coefficient_comes_from_beta() -> None:
    config = SimpleNamespace(objective="gspo", objective_overrides=None)
    objective = loss_computation._token_kl_objective(
        loss_computation._token_objective(config), config, beta=0.05
    )
    assert objective.kl_coef == 0.05
