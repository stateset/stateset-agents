"""The token path must honour every legacy ``baseline_type`` the sequence path
does; ``group_median`` and ``global_mean`` used to fall back to group-norm."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training import loss_computation as lc  # noqa: E402

REWARDS = torch.tensor([0.0, 1.0, 5.0])


def _cfg(**kw):
    return SimpleNamespace(objective=None, advantage_normalization=False, **kw)


def test_group_median_baseline():
    adv = lc._token_advantages(REWARDS, _cfg(baseline_type="group_median"), None)
    assert adv.tolist() == [-1.0, 0.0, 4.0]


def test_global_mean_baseline_uses_running_mean_and_updates_it():
    seen = []
    adv = lc._token_advantages(
        REWARDS,
        _cfg(baseline_type="global_mean"),
        None,
        global_reward_mean=0.5,
        update_global_stats=lambda mean, n: seen.append((mean, n)),
    )
    assert adv.tolist() == [-0.5, 0.5, 4.5]
    assert seen == [(2.0, 3)]


@pytest.mark.parametrize("baseline", ["group_median", "global_mean"])
def test_normalized_legacy_baselines_match_sequence_path(baseline):
    cfg = SimpleNamespace(
        objective=None, advantage_normalization=True, baseline_type=baseline
    )
    token = lc._token_advantages(
        REWARDS, cfg, None, global_reward_mean=0.5, update_global_stats=lambda *a: None
    )
    seq = lc._legacy_group_advantages(
        REWARDS, cfg, global_reward_mean=0.5, update_global_stats=lambda *a: None
    )
    torch.testing.assert_close(token, seq)
