"""Pin the preference losses to TRL 1.x on identical tensors.

TRL's loss methods are bound to bare namespaces carrying only the attributes
they read, fed the same logits / log-probs as ours. A TRL major-version change
skips loudly rather than silently passing.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import os
import sys
from collections import defaultdict
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training.preference.losses import (  # noqa: E402
    bradley_terry_loss,
    dpo_loss,
    ipo_loss,
    kto_kl_estimate,
    kto_loss,
    orpo_loss,
    sequence_logprobs,
    simpo_loss,
)


def _is_real_module(mod: Any) -> bool:
    if isinstance(mod, MagicMock):
        return False
    spec = getattr(mod, "__spec__", None)
    if not isinstance(spec, importlib.machinery.ModuleSpec):
        return False
    if spec.origin in (None, "namespace", "built-in", "frozen"):
        return bool(spec.submodule_search_locations)
    return os.path.exists(spec.origin)


def _import_real(*names: str) -> list[Any]:
    """Import real TRL modules even if another test leaked stand-ins for
    ``trl``/``peft``/``vllm`` into ``sys.modules`` (restored afterwards)."""
    leaked = {
        name: mod
        for name, mod in list(sys.modules.items())
        if name.split(".")[0] in ("trl", "peft", "vllm") and not _is_real_module(mod)
    }
    for name in leaked:
        del sys.modules[name]
    try:
        return [importlib.import_module(name) for name in names]
    finally:
        sys.modules.update(leaked)


try:
    trl, dpo_module, kto_module, reward_module = _import_real(
        "trl",
        "trl.trainer.dpo_trainer",
        "trl.trainer.kto_trainer",
        "trl.trainer.reward_trainer",
    )
except ImportError as exc:  # pragma: no cover - depends on the environment
    pytest.skip(f"trl is not importable: {exc}", allow_module_level=True)

if int(trl.__version__.split(".")[0]) != 1:
    pytest.skip(
        f"preference pin targets trl 1.x, found {trl.__version__}",
        allow_module_level=True,
    )

CPU = torch.device("cpu")
ACCELERATOR = SimpleNamespace(
    device=CPU,
    gather=lambda value: value,
    gather_for_metrics=lambda value: value,
    unwrap_model=lambda model: model,
)


def _bare(**attributes: Any) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(training=True),
        accelerator=ACCELERATOR,
        aux_loss_enabled=False,
        _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
        _total_train_tokens=0,
        _tp_size=1,
        use_liger_kernel=False,
        **attributes,
    )


def _fixture(
    seed: int, rows: int = 6, width: int = 9, vocab: int = 11
) -> dict[str, Any]:
    """Chosen rows then rejected rows, prompts of 2-3 tokens, right padding."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, width, vocab, generator=g)
    ids = torch.randint(0, vocab, (rows, width), generator=g)
    completion = torch.zeros(rows, width, dtype=torch.long)
    attention = torch.zeros(rows, width, dtype=torch.long)
    for row in range(rows):
        prompt = 2 + row % 2
        length = width - (row % 3)
        completion[row, prompt:length] = 1
        attention[row, :length] = 1
    return {
        "logits": logits,
        "ids": ids,
        "completion": completion,
        "attention": attention,
    }


def _model(logits: torch.Tensor) -> Any:
    """Return ``logits``; with ``fused_lm_head=True`` (TRL >= 1.16) return the
    fused head's per-token fields instead, zeroed where the label is ``-100``."""

    def forward(
        *, labels: Any = None, fused_lm_head: bool = False, **_: Any
    ) -> SimpleNamespace:
        if not fused_lm_head:
            return SimpleNamespace(logits=logits)
        shifted = labels[:, 1:]
        mask = shifted != -100
        head = logits[:, :-1]
        logps = torch.log_softmax(head, dim=-1)
        probs = logps.exp()
        fields = {
            "log_probs": logps.gather(-1, shifted.clamp(min=0).unsqueeze(-1)).squeeze(
                -1
            ),
            "entropy": -(probs * logps).sum(-1),
            "log_sum_sq_probs": probs.square().sum(-1).log(),
            "mean_logits": head.mean(-1),
            "is_top1": head.argmax(-1) == shifted,
        }
        return SimpleNamespace(
            aux_loss=None,
            **{name: value.masked_fill(~mask, 0) for name, value in fields.items()},
        )

    return forward


def _trl_dpo(
    loss_type: str, fixture: dict[str, Any], ref: torch.Tensor, beta: float
) -> float:
    half = fixture["ids"].shape[0] // 2
    bare = _bare(
        ld_alpha=None,
        precompute_ref_logps=True,
        f_divergence_type="reverse_kl",
        loss_types=[loss_type],
        loss_weights=[1.0],
        beta=beta,
        use_weighting=False,
        label_smoothing=0.0,
    )
    inputs = {
        "input_ids": fixture["ids"],
        "attention_mask": fixture["attention"],
        "completion_mask": fixture["completion"],
        "ref_chosen_logps": ref[:half],
        "ref_rejected_logps": ref[half:],
    }
    loss = dpo_module.DPOTrainer._compute_loss(
        bare, _model(fixture["logits"]), inputs, return_outputs=False
    )
    return float(loss)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("beta", [0.1, 0.5])
def test_dpo_sigmoid_and_ipo_match_trl(seed: int, beta: float) -> None:
    fixture = _fixture(seed)
    half = fixture["ids"].shape[0] // 2
    sums, counts = sequence_logprobs(
        fixture["logits"], fixture["ids"], fixture["completion"]
    )
    ref = sums.detach() + 0.3 * torch.randn(
        sums.shape, generator=torch.Generator().manual_seed(seed)
    )
    ours = dpo_loss(sums[:half], sums[half:], ref[:half], ref[half:], beta=beta)
    assert ours.losses.mean().item() == pytest.approx(
        _trl_dpo("sigmoid", fixture, ref, beta), rel=1e-5
    )
    ours_ipo = ipo_loss(
        sums[:half],
        sums[half:],
        ref[:half],
        ref[half:],
        counts[:half],
        counts[half:],
        beta=beta,
    )
    assert ours_ipo.losses.mean().item() == pytest.approx(
        _trl_dpo("ipo", fixture, ref, beta), rel=1e-5
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kto_matches_trl(seed: int) -> None:
    fixture = _fixture(seed)
    labels = torch.tensor([True, False, True, True, False, False])
    sums, _ = sequence_logprobs(
        fixture["logits"], fixture["ids"], fixture["completion"]
    )
    g = torch.Generator().manual_seed(100 + seed)
    ref = sums.detach() + 0.3 * torch.randn(sums.shape, generator=g)
    policy_kl = sums.detach().roll(1) + 0.2 * torch.randn(sums.shape, generator=g)
    ref_kl = sums.detach().roll(1)
    bare = _bare(
        calculate_KL=True,
        precompute_ref_logps=True,
        loss_type="kto",
        beta=0.2,
        desirable_weight=1.0,
        undesirable_weight=1.5,
        _compute_kl_logps=lambda model, batch: policy_kl,
    )
    inputs = {
        "input_ids": fixture["ids"],
        "attention_mask": fixture["attention"],
        "completion_mask": fixture["completion"],
        "label": labels,
        "ref_logps": ref,
        "ref_KL_logps": ref_kl,
    }
    trl_loss = kto_module.KTOTrainer._compute_loss(
        bare, _model(fixture["logits"]), inputs, return_outputs=False
    )
    kl = kto_kl_estimate(policy_kl, ref_kl)
    ours = kto_loss(
        sums, ref, labels, kl, beta=0.2, desirable_weight=1.0, undesirable_weight=1.5
    )
    assert ours.losses.mean().item() == pytest.approx(float(trl_loss), rel=1e-5)


@pytest.mark.parametrize("center", [None, 0.01])
def test_bradley_terry_matches_trl_reward_trainer(center: float | None) -> None:
    g = torch.Generator().manual_seed(7)
    rewards = torch.randn(8, 1, generator=g)
    margin = torch.rand(4, generator=g)
    bare = _bare(args=SimpleNamespace(center_rewards_coefficient=center))
    inputs = {"attention_mask": torch.ones(8, 3, dtype=torch.long), "margin": margin}
    trl_loss = reward_module.RewardTrainer.compute_loss(
        bare, _model(rewards), inputs, return_outputs=False
    )
    ours = bradley_terry_loss(
        rewards[:4, 0], rewards[4:, 0], margin=margin, center_coefficient=center
    )
    assert ours.losses.mean().item() == pytest.approx(float(trl_loss), rel=1e-5)


def _experimental(name: str) -> Any:
    try:
        return _import_real(name)[0]
    except ImportError as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"{name} is not importable: {exc}")


@pytest.mark.parametrize("seed", [0, 1])
def test_simpo_matches_trl_cpo(seed: int) -> None:
    cpo = _experimental("trl.experimental.cpo.cpo_trainer")
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(1, 9, (8,), generator=g).float()
    sums = -torch.rand(8, generator=g) * lengths * 2
    bare = SimpleNamespace(
        alpha=0.0,
        loss_type="simpo",
        simpo_gamma=0.6,
        beta=2.0,
        label_smoothing=0.1,
        accelerator=ACCELERATOR,
    )
    trl_losses, trl_chosen, _ = cpo.CPOTrainer.cpo_loss(
        bare, sums[:4] / lengths[:4], sums[4:] / lengths[4:]
    )
    ours = simpo_loss(
        sums[:4],
        sums[4:],
        lengths[:4],
        lengths[4:],
        beta=2.0,
        gamma=0.6,
        label_smoothing=0.1,
    )
    assert torch.allclose(ours.losses, trl_losses, rtol=1e-5)
    assert torch.allclose(ours.chosen_rewards, trl_chosen, rtol=1e-5)


@pytest.mark.parametrize("seed", [0, 1])
def test_orpo_odds_ratio_matches_trl(seed: int) -> None:
    orpo = _experimental("trl.experimental.orpo.orpo_trainer")
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(1, 9, (8,), generator=g).float()
    sums = -(torch.rand(8, generator=g) + 0.05) * lengths
    bare = SimpleNamespace(beta=0.25, accelerator=ACCELERATOR)
    trl_losses = orpo.ORPOTrainer.odds_ratio_loss(
        bare, sums[:4] / lengths[:4], sums[4:] / lengths[4:]
    )[0]
    ours = orpo_loss(sums[:4], sums[4:], lengths[:4], lengths[4:], beta=0.25)
    chosen_nll = -(sums[:4] / lengths[:4])
    # TRL returns β log σ(log odds); ORPO's loss is NLL minus that term.
    assert torch.allclose(ours.losses, chosen_nll - trl_losses, rtol=1e-5)
