"""Regressions for issues CodeQL flagged on the release-chain PR (#98)."""

from __future__ import annotations

import pytest

from stateset_agents.utils.security import InputValidator


@pytest.mark.parametrize(
    "payload",
    [
        "<script>alert(1)</script>",
        "<script>alert(1)</script >",
        "<SCRIPT src=x>alert(1)</script\t\n>",
        "<script>alert(1)",
    ],
)
def test_script_tags_are_detected_in_every_form(payload):
    assert "xss" in InputValidator.detect_injection_attempts(payload)


def test_bayesian_reward_function_initialises_reward_base():
    pytest.importorskip("torch")
    from stateset_agents.core.reward_base import RewardType
    from stateset_agents.experimental.bayesian_reward_model import (
        BayesianRewardFunction,
    )

    reward = BayesianRewardFunction(input_dim=8, device="cpu")
    assert reward.weight == 1.0
    assert reward.reward_type == RewardType.IMMEDIATE
    assert reward.name
