"""On-policy trainers warn when rollouts are not sampled from the policy itself."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from stateset_agents.training.multi_turn_trainer import rollout_sampling_transforms


def _agent(**config: object) -> SimpleNamespace:
    values = {
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
    }
    values.update(config)
    return SimpleNamespace(config=SimpleNamespace(**values))


def test_identity_sampling_has_no_transforms() -> None:
    assert rollout_sampling_transforms(_agent()) == []
    assert rollout_sampling_transforms(_agent(top_k=None)) == []


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("top_p", 0.9, "top_p=0.9"),
        ("top_k", 50, "top_k=50"),
        ("repetition_penalty", 1.1, "repetition_penalty=1.1"),
        ("temperature", 0.7, "temperature=0.7"),
        ("do_sample", False, "greedy decoding"),
    ],
)
def test_each_transform_is_reported(field: str, value: object, expected: str) -> None:
    assert rollout_sampling_transforms(_agent(**{field: value})) == [expected]


def test_agents_without_config_are_not_flagged() -> None:
    assert rollout_sampling_transforms(SimpleNamespace()) == []


def test_trainer_warns_once_at_initialize(caplog: pytest.LogCaptureFixture) -> None:
    from stateset_agents.training.multi_turn_trainer import (
        _warn_on_biased_rollout_sampling,
    )

    agent = _agent(top_p=0.9, repetition_penalty=1.1)
    with caplog.at_level(logging.WARNING):
        _warn_on_biased_rollout_sampling(agent)
    [record] = [r for r in caplog.records if "on-policy" in r.getMessage()]
    assert "top_p=0.9" in record.getMessage()
    assert "repetition_penalty=1.1" in record.getMessage()
