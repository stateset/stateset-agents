"""Pin StateSet objectives to TRL's GRPO loss on identical tensors.

TRL's ``GRPOTrainer._compute_loss`` is bound to a bare namespace carrying only
the attributes it reads, with the per-token log-prob helper replaced by a
function that returns our fixture. Any TRL major-version change makes this
module skip loudly rather than silently pass.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import os
import sys
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training import objectives as O  # noqa: E402


def _is_real_module(mod) -> bool:
    """True only for a module actually loaded from a file on disk."""
    if isinstance(mod, MagicMock):
        return False
    spec = getattr(mod, "__spec__", None)
    if not isinstance(spec, importlib.machinery.ModuleSpec):
        return False
    origin = spec.origin
    if origin in (None, "namespace", "built-in", "frozen"):
        return bool(spec.submodule_search_locations)
    return os.path.exists(origin)


def _import_real_trl():
    """Import the real ``trl`` even when another test module has leaked
    stand-ins for ``trl``/``peft``/``vllm`` into ``sys.modules`` (mocks or bare
    ``ModuleType`` stubs; ``tests/unit/test_trl_grpo_trainer.py`` and others do
    so at import time and never restore them). The mocks are put back afterwards so that module still
    sees them; the real classes bound here keep working regardless.
    """
    leaked = {
        name: mod
        for name, mod in list(sys.modules.items())
        if name.split(".")[0] in ("trl", "peft", "vllm") and not _is_real_module(mod)
    }
    for name in leaked:
        del sys.modules[name]
    try:
        trl_mod = importlib.import_module("trl")
        grpo_mod = importlib.import_module("trl.trainer.grpo_trainer")
    finally:
        sys.modules.update(leaked)
    return trl_mod, grpo_mod.GRPOTrainer


try:
    trl, GRPOTrainer = _import_real_trl()
except ImportError as exc:  # pragma: no cover - depends on the environment
    pytest.skip(f"trl is not importable: {exc}", allow_module_level=True)

TRL_MAJOR = 1
if int(trl.__version__.split(".")[0]) != TRL_MAJOR:
    pytest.skip(
        f"objective pin targets trl {TRL_MAJOR}.x, found {trl.__version__}",
        allow_module_level=True,
    )


def _fixture(seed: int, n: int = 6, t: int = 8, prompt: int = 3):
    g = torch.Generator().manual_seed(seed)
    logp = (-torch.rand(n, t, generator=g) * 3).requires_grad_(True)
    old = logp.detach() + 0.05 * torch.randn(n, t, generator=g)
    ref = logp.detach() + 0.1 * torch.randn(n, t, generator=g)
    mask = torch.ones(n, t)
    mask[0, 5:] = 0
    mask[1, 6:] = 0
    entropies = torch.rand(n, t, generator=g)
    rewards = torch.rand(n, generator=g)
    return logp, old, ref, mask, entropies, rewards, prompt


def _trl_loss(
    *,
    loss_type,
    level,
    eps_low,
    eps_high,
    beta,
    bias_correction,
    delta,
    fixture,
    adv,
    max_len,
):
    logp, old, ref, mask, entropies, _, prompt = fixture
    n, t = logp.shape

    def fake_logps(self, model, input_ids, attention_mask, logits_to_keep, **kw):
        return logp, entropies, None

    fake = SimpleNamespace(
        aux_loss_enabled=False,
        top_entropy_quantile=1.0,
        off_policy_mask_threshold=None,
        importance_sampling_level=level,
        beta=beta,
        loss_type=loss_type,
        epsilon_low=eps_low,
        epsilon_high=eps_high,
        use_vllm=False,
        vllm_importance_sampling_correction=False,
        model=SimpleNamespace(training=False),
        current_gradient_accumulation_steps=1,
        max_completion_length=max_len,
        _entropy_bonus_enabled=False,
        _metrics={"eval": defaultdict(list), "train": defaultdict(list)},
        args=SimpleNamespace(
            use_bias_correction_kl=bias_correction, delta=delta, steps_per_generation=1
        ),
        accelerator=SimpleNamespace(
            num_processes=1,
            sync_gradients=True,
            gather=lambda x: x,
            gather_for_metrics=lambda x: x,
            reduce=lambda x, reduction="sum": x,
        ),
    )
    fake._get_per_token_logps_and_entropies = fake_logps.__get__(fake)
    inputs = {
        "prompt_ids": torch.zeros(n, prompt, dtype=torch.long),
        "prompt_mask": torch.ones(n, prompt, dtype=torch.long),
        "completion_ids": torch.zeros(n, t, dtype=torch.long),
        "completion_mask": mask,
        "advantages": adv,
        "old_per_token_logps": old,
        "ref_per_token_logps": ref,
        "num_items_in_batch": mask.sum(),
    }
    return GRPOTrainer._compute_loss(fake, None, inputs)


CASES = [
    # (stateset preset, trl loss_type, importance level, eps_low, eps_high)
    ("grpo", "grpo", "token", 0.2, 0.2),
    ("bnpo", "bnpo", "token", 0.2, 0.2),
    ("dapo", "dapo", "token", 0.2, 0.28),
    ("dr_grpo", "dr_grpo", "token", 0.2, 0.2),
    ("cispo", "cispo", "token", 0.2, 5.0),
    ("gspo", "grpo", "sequence", 3e-4, 4e-4),
]


@pytest.mark.parametrize("preset, loss_type, level, lo, hi", CASES)
@pytest.mark.parametrize("beta", [0.0, 0.04])
@pytest.mark.parametrize("bias_correction", [False, True])
def test_loss_matches_trl(preset, loss_type, level, lo, hi, beta, bias_correction):
    if bias_correction and (beta == 0.0 or level == "sequence"):
        pytest.skip("bias correction only meaningful for token-level k3")
    fixture = _fixture(seed=sum(map(ord, preset)) % 1000)
    logp, old, ref, mask, entropies, rewards, _ = fixture
    max_len = logp.shape[1]
    group_ids = torch.arange(logp.shape[0]) % 2

    obj = O.OBJECTIVES[preset].with_(advantage_eps=1e-4)
    if obj.aggregate == "seq_sum_const":
        obj = obj.with_(max_completion_length=max_len)
    if beta > 0:
        obj = obj.with_(kl="k3_token", kl_coef=beta, kl_bias_correction=bias_correction)
    adv = O.compute_advantages(rewards, group_ids, obj)

    ours = O.policy_loss(
        logp_cur=logp,
        mask=mask,
        advantages=adv,
        objective=obj,
        logp_old=old,
        logp_ref=ref,
    ).loss
    theirs = _trl_loss(
        loss_type=loss_type,
        level=level,
        eps_low=lo,
        eps_high=hi,
        beta=beta,
        bias_correction=bias_correction,
        delta=None,
        fixture=fixture,
        adv=adv,
        max_len=max_len,
    )
    torch.testing.assert_close(ours, theirs, atol=1e-5, rtol=1e-5)


def test_delta_two_sided_clip_matches_trl():
    fixture = _fixture(seed=77)
    logp, old, ref, mask, _, rewards, _ = fixture
    group_ids = torch.arange(logp.shape[0]) % 2
    obj = O.OBJECTIVES["dapo"].with_(advantage_eps=1e-4, delta=1.05)
    adv = O.compute_advantages(rewards, group_ids, obj)
    ours = O.policy_loss(
        logp_cur=logp, mask=mask, advantages=adv, objective=obj, logp_old=old
    ).loss
    theirs = _trl_loss(
        loss_type="dapo",
        level="token",
        eps_low=0.2,
        eps_high=0.28,
        beta=0.0,
        bias_correction=False,
        delta=1.05,
        fixture=fixture,
        adv=adv,
        max_len=logp.shape[1],
    )
    torch.testing.assert_close(ours, theirs, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "scale, kind",
    [("group", "group_norm"), ("none", "group_mean"), ("batch", "batch_norm")],
)
def test_advantages_match_trl_scale_rewards(scale, kind):
    """Transcription of TRL 1.12 ``scale_rewards`` (nanstd, eps 1e-4)."""
    g = torch.Generator().manual_seed(5)
    num_generations = 4
    rewards = torch.rand(8, generator=g)
    grouped = rewards.view(-1, num_generations)
    mean_g = grouped.mean(dim=1).repeat_interleave(num_generations)
    if scale in ("group", "none"):
        std = grouped.std(dim=1, correction=0).repeat_interleave(num_generations)
    else:
        std = rewards.std(correction=0).expand_as(rewards)
    trl_adv = rewards - mean_g
    if scale != "none":
        trl_adv = trl_adv / (std + 1e-4)

    obj = O.PolicyObjective(name="t", advantage=kind, advantage_eps=1e-4)
    ours = O.compute_advantages(rewards, torch.arange(8) // num_generations, obj)
    torch.testing.assert_close(ours, trl_adv, atol=1e-6, rtol=1e-6)
