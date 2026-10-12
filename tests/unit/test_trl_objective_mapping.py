"""``TrainingConfig.objective`` must reach TRL instead of being silently dropped.

The (preset -> loss_type, importance level, clip) pairs are the ones
``test_objectives_trl_pin.py`` verifies numerically against TRL's loss."""

from __future__ import annotations

import logging

import pytest

from stateset_agents.training.trl_grpo_config import TRLGRPOConfig
from stateset_agents.training.trl_grpo_trainer import trl_objective_kwargs


def _cfg(**kwargs) -> TRLGRPOConfig:
    return TRLGRPOConfig(model_name="gpt2", **kwargs)


def test_no_objective_keeps_trl_defaults():
    assert trl_objective_kwargs(_cfg()) == {}


@pytest.mark.parametrize(
    "preset, expected",
    [
        ("grpo", {"loss_type": "grpo", "epsilon": 0.2, "epsilon_high": 0.2}),
        ("bnpo", {"loss_type": "bnpo", "epsilon": 0.2, "epsilon_high": 0.2}),
        ("dapo", {"loss_type": "dapo", "epsilon": 0.2, "epsilon_high": 0.28}),
        ("dr_grpo", {"loss_type": "dr_grpo", "scale_rewards": "none"}),
        ("cispo", {"loss_type": "cispo", "epsilon_high": 5.0}),
        (
            "gspo",
            {
                "loss_type": "grpo",
                "importance_sampling_level": "sequence",
                "epsilon": 3e-4,
                "epsilon_high": 4e-4,
            },
        ),
    ],
)
def test_presets_map_to_their_pinned_trl_settings(preset, expected):
    mapped = trl_objective_kwargs(_cfg(objective=preset))
    for key, value in expected.items():
        if isinstance(value, float):
            assert mapped[key] == pytest.approx(value), (key, mapped)
        else:
            assert mapped[key] == value, (key, mapped)
    assert mapped.get("importance_sampling_level", "token") == expected.get(
        "importance_sampling_level", "token"
    )


def test_overrides_flow_through():
    mapped = trl_objective_kwargs(
        _cfg(objective="dapo", objective_overrides={"clip_high": 0.3})
    )
    assert mapped["epsilon_high"] == pytest.approx(0.3)


@pytest.mark.parametrize("preset", ["gepo", "rloo", "gspo_token", "ppo"])
def test_presets_trl_cannot_express_are_rejected(preset):
    with pytest.raises(ValueError, match="TRL"):
        trl_objective_kwargs(_cfg(objective=preset))


def test_structural_override_that_trl_cannot_express_is_rejected():
    with pytest.raises(ValueError, match="TRL"):
        trl_objective_kwargs(
            _cfg(objective="dapo", objective_overrides={"advantage": "leave_one_out"})
        )


def test_mapped_kwargs_are_accepted_by_real_trl():
    trl = pytest.importorskip("trl")
    if int(trl.__version__.split(".")[0]) != 1:
        pytest.skip("pinned against TRL 1.x")
    from trl import GRPOConfig

    for preset in ("grpo", "bnpo", "dapo", "dr_grpo", "cispo", "gspo"):
        kwargs = trl_objective_kwargs(_cfg(objective=preset))
        config = GRPOConfig(
            output_dir="unused", report_to=[], bf16=False, use_cpu=True, **kwargs
        )
        for key, value in kwargs.items():
            assert getattr(config, key) == value


def test_dropped_config_keys_are_logged(caplog):
    from stateset_agents.training.trl_grpo_trainer import _supported_config_kwargs

    def fake_config(*, output_dir: str, beta: float = 0.0):
        return None

    with caplog.at_level(logging.WARNING):
        kept = _supported_config_kwargs(
            fake_config, {"output_dir": "x", "beta": 0.1, "max_prompt_length": 9}
        )
    assert kept == {"output_dir": "x", "beta": 0.1}
    assert "max_prompt_length" in caplog.text
