"""Selecting a PolicyObjective preset by name from configs and trainers."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training import objectives as O  # noqa: E402
from stateset_agents.training.config import TrainingConfig  # noqa: E402

# --- config ------------------------------------------------------------------


def test_training_config_defaults_to_no_objective():
    cfg = TrainingConfig()
    assert cfg.objective is None
    assert cfg.objective_overrides is None


def test_objective_round_trips_through_dict_and_file(tmp_path):
    cfg = TrainingConfig(objective="cispo", objective_overrides={"is_cap": 3.0})
    again = TrainingConfig.from_dict(cfg.to_dict())
    assert again.objective == "cispo" and again.objective_overrides == {"is_cap": 3.0}
    path = tmp_path / "cfg.json"
    cfg.save(str(path))
    loaded = TrainingConfig.load(str(path))
    assert loaded.objective == "cispo"
    assert json.loads(path.read_text())["objective_overrides"] == {"is_cap": 3.0}


def test_validate_warns_on_unknown_objective():
    warnings = TrainingConfig(objective="not-a-preset").validate()
    assert any("objective" in w and "not-a-preset" in w for w in warnings)
    assert not any(
        "objective" in w for w in TrainingConfig(objective="dapo").validate()
    )


# --- resolve_objective ---------------------------------------------------------


def test_resolve_unset_uses_native_preset_and_fields():
    cfg = TrainingConfig(beta=0.05)
    obj = O.resolve_objective(
        cfg, "gspo", kl_coef=cfg.beta, clip_low=0.1, clip_high=0.3, aggregate="seq_mean"
    )
    assert obj.name == "gspo"
    assert (obj.clip_low, obj.clip_high, obj.aggregate, obj.kl_coef) == (
        0.1,
        0.3,
        "seq_mean",
        0.05,
    )
    # A native preset without a KL estimator ignores kl_coef instead of failing.
    assert O.resolve_objective(cfg, "dapo", kl_coef=cfg.beta).kl_coef == 0.0


def test_resolve_named_preset_ignores_native_clip_fields_but_keeps_kl_and_length():
    cfg = TrainingConfig(objective="cispo", beta=0.02, max_completion_length=64)
    obj = O.resolve_objective(
        cfg,
        "dapo",
        kl_coef=cfg.beta,
        max_completion_length=cfg.max_completion_length,
        clip_low=0.1,
        clip_high=0.3,
        aggregate="seq_mean",
    )
    assert obj.name == "cispo" and obj.clip == "cispo"
    assert obj.aggregate == "token_mean"  # the preset's own, not the native field
    assert obj.kl_coef == 0.0  # cispo preset has kl='none'; beta cannot apply
    assert obj.max_completion_length == 64


def test_resolve_applies_kl_coef_only_when_preset_has_a_kl_estimator():
    cfg = TrainingConfig(objective="grpo", beta=0.04)
    obj = O.resolve_objective(cfg, "dapo", kl_coef=cfg.beta)
    assert obj.kl == "k3_token" and obj.kl_coef == 0.04


def test_resolve_applies_overrides_last():
    cfg = TrainingConfig(objective="dapo", objective_overrides={"clip_high": 0.5})
    obj = O.resolve_objective(cfg, "gspo", clip_low=0.1, clip_high=0.3)
    assert obj.name == "dapo" and obj.clip_high == 0.5 and obj.clip_low == 0.2


def test_resolve_unknown_name_lists_presets():
    cfg = TrainingConfig(objective="bogus")
    with pytest.raises(ValueError, match="bogus") as info:
        O.resolve_objective(cfg, "dapo")
    for name in ("grpo", "dr_grpo", "cispo", "rloo"):
        assert name in str(info.value)


def test_resolve_rejects_unsupported_ratio_with_helpful_message():
    cfg = TrainingConfig(objective="dapo")
    with pytest.raises(ValueError, match="sequence") as info:
        O.resolve_objective(
            cfg, "gspo", supported_ratios=("sequence", "sequence_token")
        )
    assert "gspo" in str(info.value) and "dapo" in str(info.value)


def test_resolve_dr_grpo_gets_max_completion_length_from_config():
    cfg = TrainingConfig(objective="dr_grpo", max_completion_length=128)
    obj = O.resolve_objective(
        cfg, "dapo", max_completion_length=cfg.max_completion_length
    )
    assert obj.aggregate == "seq_sum_const" and obj.max_completion_length == 128


def test_compatible_presets_helper():
    names = O.compatible_presets(("sequence", "sequence_token"))
    assert set(names) == {"gspo", "gspo_token"}


# --- trainers --------------------------------------------------------------------


def _tiny_model():
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    return GPT2LMHeadModel(
        GPT2Config(
            n_embd=32,
            n_layer=2,
            n_head=2,
            vocab_size=200,
            n_positions=64,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )


def _dapo(**cfg):
    from stateset_agents.training.dapo_trainer import DAPOConfig, DAPOTrainer

    return DAPOTrainer(
        config=DAPOConfig(model_name="gpt2", group_size=4, **cfg),
        model=_tiny_model(),
        tokenizer=None,
        reward_fn=lambda p, r: 0.0,
    )


def _gspo(cls_name="GSPOTrainer", **cfg):
    from stateset_agents.training import gspo_token_trainer, gspo_trainer
    from stateset_agents.training.gspo_config import GSPOConfig

    cls = getattr(gspo_trainer, cls_name, None) or getattr(gspo_token_trainer, cls_name)
    return cls(
        config=GSPOConfig(model_name="gpt2", num_generations=4, **cfg),
        model=_tiny_model(),
        tokenizer=None,
        agent=None,
        environment=None,
        reward_model=None,
        ref_model=None,
    )


def _gepo(**cfg):
    from stateset_agents.training.gepo_trainer import GEPOConfig, GEPOTrainer

    return GEPOTrainer(
        config=GEPOConfig(model_name="gpt2", group_size=4, **cfg),
        model=_tiny_model(),
        tokenizer=None,
        reward_fn=lambda p, r: 0.0,
    )


def _vapo(**cfg):
    from stateset_agents.training.vapo_trainer import VAPOConfig, VAPOTrainer

    return VAPOTrainer(
        config=VAPOConfig(model_name="gpt2", group_size=4, **cfg),
        model=_tiny_model(),
        tokenizer=None,
        reward_fn=lambda p, r: 1.0,
    )


def _ppo(**cfg):
    from stateset_agents.training.ppo_trainer import PPOConfig, PPOTrainer

    return PPOTrainer(
        config=PPOConfig(model_name="gpt2", **cfg),
        model=_tiny_model(),
        tokenizer=None,
        reward_fn=lambda p, r: 0.0,
    )


@pytest.mark.parametrize(
    "factory, native, chosen",
    [
        (_dapo, "dapo", "cispo"),
        (_dapo, "dapo", "rloo"),
        (_vapo, "vapo", "bnpo"),
        (_ppo, "ppo", "dapo"),
        (_gspo, "gspo", "gspo_token"),
        (lambda **c: _gspo("GSPOTokenTrainer", **c), "gspo_token", "gspo"),
        (_gepo, "gepo", "gspo"),
    ],
)
def test_trainers_honour_config_objective(factory, native, chosen):
    assert factory()._objective.name == native
    assert factory(objective=chosen)._objective.name == chosen


@pytest.mark.parametrize("factory", [_gspo, _gepo])
def test_sequence_sum_trainers_reject_token_presets(factory):
    with pytest.raises(ValueError, match="sequence"):
        factory(objective="dapo")


@pytest.mark.parametrize("factory", [_dapo, _vapo, _ppo])
def test_per_token_trainers_reject_group_expectation(factory):
    with pytest.raises(ValueError, match="group_expectation"):
        factory(objective="gepo")


def test_dapo_rloo_changes_group_advantages():
    rewards = torch.tensor([1.0, 0.0, 0.5, 0.25])
    native = _dapo().compute_group_advantages(rewards)
    rloo = _dapo(objective="rloo").compute_group_advantages(rewards)
    assert not torch.allclose(native, rloo)
    torch.testing.assert_close(
        rloo,
        O.compute_advantages(
            rewards, torch.zeros(4, dtype=torch.long), O.OBJECTIVES["rloo"]
        ),
    )


def test_gspo_dr_grpo_style_advantages_via_overrides():
    rewards = torch.tensor([1.0, 0.0, 0.5, 0.25])
    tr = _gspo(objective="gspo", objective_overrides={"advantage": "group_mean"})
    adv, _ = tr.compute_group_advantages(rewards)
    torch.testing.assert_close(adv, rewards - rewards.mean())


# --- GRPO loss path -----------------------------------------------------------------


def test_grpo_loss_path_honours_objective(monkeypatch):
    import importlib.util
    from pathlib import Path

    from stateset_agents.training import loss_computation as lc

    spec = importlib.util.spec_from_file_location(
        "tlcb", Path(__file__).with_name("test_loss_computation_behavioral.py")
    )
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)

    # ratio ~ exp(1e-3) > 1 + 3e-4: clipped under the default symmetric
    # trust region, unclipped once clip_high is widened through overrides.
    group = helpers._make_group(8, 4, log_probs=-(0.5 + 1e-3) * 4)
    base = helpers._config(clip_ratio=0.2, seq_clip_ratio=3e-4)
    wide = helpers._config(
        clip_ratio=0.2,
        seq_clip_ratio=3e-4,
        objective="gspo",
        objective_overrides={"clip_high": 0.5},
    )
    loss_base, _ = lc._compute_group_policy_loss(
        group, torch.tensor([1.0]), base, helpers._make_agent(8, 4)
    )
    loss_wide, _ = lc._compute_group_policy_loss(
        group, torch.tensor([1.0]), wide, helpers._make_agent(8, 4)
    )
    assert loss_base.item() != pytest.approx(loss_wide.item())

    token = helpers._config(clip_ratio=0.2, objective="dapo")
    with pytest.raises(ValueError, match="sequence"):
        lc._compute_group_policy_loss(
            group, torch.tensor([1.0]), token, helpers._make_agent(8, 4)
        )
