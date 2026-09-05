"""PolicyObjective validation, presets, and parity with the loop reference."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training import objectives as O  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "objective_reference", Path(__file__).with_name("objective_reference.py")
)
R = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(R)

PRESETS = sorted(O.OBJECTIVES)


def _fixture(seed: int, n: int = 6, t: int = 7, groups: int = 2):
    g = torch.Generator().manual_seed(seed)
    logp_cur = -torch.rand(n, t, generator=g) * 3
    logp_old = logp_cur + 0.05 * torch.randn(n, t, generator=g)
    logp_ref = logp_cur + 0.1 * torch.randn(n, t, generator=g)
    mask = torch.ones(n, t)
    mask[0, 5:] = 0
    mask[1, 2:] = 0
    mask[2, :] = 0  # empty row
    group_ids = torch.arange(n) % groups
    rewards = torch.rand(n, generator=g)
    rewards[group_ids == 1] = 0.5  # one constant-reward group
    return logp_cur, logp_old, logp_ref, mask, group_ids, rewards


def _lists(x):
    return x.detach().tolist()


# --- construction ----------------------------------------------------------


def test_presets_exist_and_are_frozen():
    assert PRESETS == sorted(
        [
            "grpo",
            "dr_grpo",
            "bnpo",
            "dapo",
            "gspo",
            "gspo_token",
            "gepo",
            "rloo",
            "reinforce_pp_baseline",
            "cispo",
            "ppo",
        ]
    )
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        O.OBJECTIVES["grpo"].clip_low = 0.5  # type: ignore[misc]
    with pytest.raises(TypeError):
        O.OBJECTIVES["grpo"] = None  # type: ignore[index]


def test_with_returns_modified_copy():
    base = O.OBJECTIVES["grpo"]
    new = base.with_(clip_high=0.28, name="mine")
    assert new.clip_high == 0.28 and new.name == "mine"
    assert base.clip_high == 0.2


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"advantage": "bogus"}, "advantage"),
        ({"ratio": "bogus"}, "ratio"),
        ({"clip": "bogus"}, "clip"),
        ({"aggregate": "bogus"}, "aggregate"),
        ({"kl": "bogus"}, "kl"),
        ({"clip_low": -0.1}, "clip_low"),
        ({"clip_high": -0.1}, "clip_high"),
        ({"is_cap": 0.0}, "is_cap"),
        ({"ratio_clamp": 0.0}, "ratio_clamp"),
        ({"delta": 0.0}, "delta"),
        ({"kl": "none", "kl_coef": 0.1}, "kl_coef"),
        ({"kl": "k3_sequence", "kl_bias_correction": True}, "kl_bias_correction"),
        ({"max_completion_length": 0}, "max_completion_length"),
    ],
)
def test_invalid_objectives_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        O.PolicyObjective(name="x", **kwargs)


# --- advantages ------------------------------------------------------------


@pytest.mark.parametrize(
    "kind", ["group_norm", "group_mean", "leave_one_out", "batch_norm"]
)
def test_compute_advantages_matches_reference(kind):
    _, _, _, _, group_ids, rewards = _fixture(1)
    obj = O.PolicyObjective(name="t", advantage=kind, advantage_eps=1e-8)
    got = O.compute_advantages(rewards, group_ids, obj)
    want = R.ref_advantages(_lists(rewards), _lists(group_ids), kind, 1e-8)
    torch.testing.assert_close(got, torch.tensor(want), atol=1e-6, rtol=0)


def test_compute_advantages_group_of_one_is_zero():
    obj = O.PolicyObjective(name="t", advantage="group_norm")
    got = O.compute_advantages(
        torch.tensor([0.3, 1.0, 2.0]), torch.tensor([0, 1, 1]), obj
    )
    assert got[0].item() == 0.0 and torch.isfinite(got).all()


def test_compute_advantages_external_raises():
    obj = O.PolicyObjective(name="t", advantage="external")
    with pytest.raises(ValueError, match="external"):
        O.compute_advantages(torch.zeros(2), torch.zeros(2, dtype=torch.long), obj)


def test_group_norm_matches_rl_losses_group_advantages():
    from stateset_agents.training import rl_losses

    rewards = torch.tensor([0.1, 0.9, 0.4, 0.4])
    obj = O.PolicyObjective(name="t", advantage="group_norm")
    got = O.compute_advantages(rewards, torch.zeros(4, dtype=torch.long), obj)
    torch.testing.assert_close(got, rl_losses.group_advantages(rewards))


# --- policy_loss parity ----------------------------------------------------


def _run(obj, seed=0, **overrides):
    logp_cur, logp_old, logp_ref, mask, group_ids, rewards = _fixture(seed)
    logp_cur = logp_cur.clone().requires_grad_(True)
    if obj.aggregate == "seq_sum_const":
        obj = obj.with_(max_completion_length=7)
    if obj.ratio == "group_expectation":
        logp_old = (logp_old * mask).sum(-1)  # sampler sequence sums
    adv_obj = obj if obj.advantage != "external" else obj.with_(advantage="group_norm")
    advantages = O.compute_advantages(rewards, group_ids, adv_obj)
    kwargs = {
        "logp_cur": logp_cur,
        "mask": mask,
        "advantages": advantages,
        "objective": obj,
        "logp_old": logp_old,
        "logp_ref": logp_ref,
        "group_ids": group_ids,
    }
    kwargs.update(overrides)
    res = O.policy_loss(**kwargs)
    want = R.ref_policy_loss(
        obj,
        _lists(logp_cur),
        _lists(mask),
        _lists(advantages),
        logp_old=_lists(logp_old),
        logp_ref=_lists(logp_ref),
        groups=_lists(group_ids),
        kl_ext=_lists(kwargs["kl"]) if kwargs.get("kl") is not None else None,
        entropy=(
            _lists(kwargs["entropy"]) if kwargs.get("entropy") is not None else None
        ),
    )
    return res, want


@pytest.mark.parametrize("name", PRESETS)
def test_preset_matches_reference(name):
    res, want = _run(O.OBJECTIVES[name])
    assert res.loss.item() == pytest.approx(want, abs=1e-6)
    assert torch.isfinite(res.loss)
    res.loss.backward()  # differentiable


@pytest.mark.parametrize("name", ["grpo", "gspo", "gspo_token", "dapo", "cispo"])
def test_preset_with_kl_matches_reference(name):
    obj = O.OBJECTIVES[name]
    kl = "k3_sequence" if obj.ratio in ("sequence", "sequence_token") else "k3_token"
    obj = obj.with_(kl=kl, kl_coef=0.05)
    res, want = _run(obj)
    assert res.loss.item() == pytest.approx(want, abs=1e-6)
    assert res.metrics["kl"] > 0


def test_k3_token_bias_correction_matches_reference():
    obj = O.OBJECTIVES["grpo"].with_(
        kl="k3_token", kl_coef=0.05, kl_bias_correction=True
    )
    res, want = _run(obj)
    assert res.loss.item() == pytest.approx(want, abs=1e-6)


def test_external_kl_matches_reference():
    obj = O.OBJECTIVES["grpo"].with_(kl="external", kl_coef=0.1)
    kl = torch.rand(6, 7, generator=torch.Generator().manual_seed(3))
    res, want = _run(obj, kl=kl)
    assert res.loss.item() == pytest.approx(want, abs=1e-6)


def test_entropy_bonus_matches_reference():
    obj = O.OBJECTIVES["dapo"].with_(entropy_coef=0.01)
    ent = torch.rand(6, 7, generator=torch.Generator().manual_seed(4))
    res, want = _run(obj, entropy=ent)
    assert res.loss.item() == pytest.approx(want, abs=1e-6)
    assert res.metrics["entropy"] > 0


def test_delta_cap_matches_reference():
    obj = O.OBJECTIVES["dapo"].with_(delta=1.1)
    res, want = _run(obj)
    assert res.loss.item() == pytest.approx(want, abs=1e-6)


def test_sequence_ratio_accepts_sum_logp_old():
    logp_cur, logp_old, _, mask, group_ids, rewards = _fixture(2)
    obj = O.OBJECTIVES["gspo"]
    adv = O.compute_advantages(rewards, group_ids, obj)
    per_token = O.policy_loss(
        logp_cur=logp_cur, mask=mask, advantages=adv, objective=obj, logp_old=logp_old
    )
    sums = O.policy_loss(
        logp_cur=logp_cur,
        mask=mask,
        advantages=adv,
        objective=obj,
        logp_old=(logp_old * mask).sum(-1),
    )
    torch.testing.assert_close(per_token.loss, sums.loss)


def test_token_ratio_rejects_sum_logp_old():
    logp_cur, logp_old, _, mask, group_ids, rewards = _fixture(2)
    obj = O.OBJECTIVES["dapo"]
    adv = O.compute_advantages(rewards, group_ids, obj)
    with pytest.raises(ValueError, match="per-token"):
        O.policy_loss(
            logp_cur=logp_cur,
            mask=mask,
            advantages=adv,
            objective=obj,
            logp_old=logp_old.sum(-1),
        )


def test_dr_grpo_requires_max_completion_length():
    logp_cur, logp_old, _, mask, group_ids, rewards = _fixture(2)
    obj = O.OBJECTIVES["dr_grpo"]
    adv = O.compute_advantages(rewards, group_ids, obj)
    with pytest.raises(ValueError, match="max_completion_length"):
        O.policy_loss(
            logp_cur=logp_cur,
            mask=mask,
            advantages=adv,
            objective=obj,
            logp_old=logp_old,
        )


def test_missing_logp_old_defaults_to_detached_current():
    logp_cur, _, _, mask, group_ids, rewards = _fixture(5)
    logp_cur = logp_cur.requires_grad_(True)
    obj = O.OBJECTIVES["grpo"]
    adv = O.compute_advantages(rewards, group_ids, obj)
    a = O.policy_loss(logp_cur=logp_cur, mask=mask, advantages=adv, objective=obj)
    b = O.policy_loss(
        logp_cur=logp_cur,
        mask=mask,
        advantages=adv,
        objective=obj,
        logp_old=logp_cur.detach(),
    )
    torch.testing.assert_close(a.loss, b.loss)
    torch.testing.assert_close(a.ratio, torch.ones_like(a.ratio))
    a.loss.backward()
    assert logp_cur.grad.abs().sum() > 0


def test_metrics_keys_and_types():
    res, _ = _run(O.OBJECTIVES["dapo"])
    for key in (
        "policy_loss",
        "kl",
        "entropy",
        "clip_fraction",
        "ratio_mean",
        "ratio_max",
        "advantage_mean",
        "advantage_std",
    ):
        assert isinstance(res.metrics[key], float), key


def test_per_token_advantages_are_accepted():
    logp_cur, logp_old, _, mask, _, _ = _fixture(6)
    obj = O.OBJECTIVES["ppo"]
    adv = torch.randn(6, 7, generator=torch.Generator().manual_seed(9))
    res = O.policy_loss(
        logp_cur=logp_cur, mask=mask, advantages=adv, objective=obj, logp_old=logp_old
    )
    want = R.ref_policy_loss(
        obj, _lists(logp_cur), _lists(mask), _lists(adv), logp_old=_lists(logp_old)
    )
    assert res.loss.item() == pytest.approx(want, abs=1e-6)


def test_module_imports_without_torch(monkeypatch):
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    sys.modules.pop("stateset_agents.training.objectives", None)
    mod = importlib.import_module("stateset_agents.training.objectives")
    assert mod.OBJECTIVES["grpo"].name == "grpo"
    sys.modules.pop("stateset_agents.training.objectives", None)
