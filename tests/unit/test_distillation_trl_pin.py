"""Pin the distillation losses to TRL's GKD trainer on identical tensors.

``GKDTrainer.generalized_jsd_loss`` (``trl.experimental.gkd``, TRL 1.x) is a
static method, so it is called directly; ``compute_loss`` is bound to a bare
namespace carrying only what it reads, with the trainer's real tiny student
and teacher. A TRL major-version change skips loudly rather than silently
passing. Verified on trl 1.9.1 (the lock file) and 1.16.0.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import os
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from stateset_agents.training.distillation import (  # noqa: E402
    DistillationConfig,
    DistillationDataset,
    DistillationTrainer,
    generalized_jsd_loss,
)
from stateset_agents.training.distillation.trainer import _Sample  # noqa: E402
from stateset_agents.training.preference import _loop  # noqa: E402
from stateset_agents.training.preference.losses import reduce_losses  # noqa: E402
from tests.learning._toy import make_model, make_tokenizer  # noqa: E402


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


os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
try:
    trl, gkd_module = _import_real("trl", "trl.experimental.gkd.gkd_trainer")
except ImportError as exc:  # pragma: no cover - depends on the environment
    pytest.skip(f"trl's GKD trainer is not importable: {exc}", allow_module_level=True)

if int(trl.__version__.split(".")[0]) != 1:
    pytest.skip(
        f"distillation pin targets trl 1.x, found {trl.__version__}",
        allow_module_level=True,
    )

GKD = gkd_module.GKDTrainer


def _fixture(
    seed: int, rows: int = 4, width: int = 9, vocab: int = 13
) -> dict[str, Any]:
    """Shift-aligned logits with prompts of 2-3 positions and right padding."""
    g = torch.Generator().manual_seed(seed)
    student = torch.randn(rows, width, vocab, generator=g) * 2.0
    teacher = torch.randn(rows, width, vocab, generator=g) * 2.0
    labels = torch.randint(0, vocab, (rows, width), generator=g)
    for row in range(rows):
        labels[row, : 2 + row % 2] = -100
        labels[row, width - row % 3 :] = -100
    return {"student": student, "teacher": teacher, "labels": labels}


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("beta", [0.0, 0.1, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("temperature", [1.0, 2.0])
def test_generalized_jsd_matches_trl(
    seed: int, beta: float, temperature: float
) -> None:
    fx = _fixture(seed)
    ours_student = fx["student"].clone().requires_grad_(True)
    trl_student = fx["student"].clone().requires_grad_(True)
    ours = generalized_jsd_loss(
        ours_student,
        fx["teacher"],
        fx["labels"],
        beta=beta,
        temperature=temperature,
    )
    theirs = GKD.generalized_jsd_loss(
        trl_student,
        fx["teacher"],
        labels=fx["labels"],
        beta=beta,
        temperature=temperature,
    )
    assert float(ours.detach()) == pytest.approx(
        float(theirs.detach()), rel=1e-5, abs=1e-7
    )
    ours.backward()
    theirs.backward()
    assert torch.allclose(ours_student.grad, trl_student.grad, rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_sum_reduction_and_unmasked_batchmean_match_trl(beta: float) -> None:
    fx = _fixture(3)
    for labels in (fx["labels"], None):
        for reduction in ("sum", "batchmean"):
            ours = generalized_jsd_loss(
                fx["student"], fx["teacher"], labels, beta=beta, reduction=reduction
            )
            theirs = GKD.generalized_jsd_loss(
                fx["student"],
                fx["teacher"],
                labels=labels,
                beta=beta,
                reduction=reduction,
            )
            if labels is None and reduction == "batchmean":
                # Without labels TRL divides by the batch size, not positions.
                theirs = theirs * fx["student"].shape[0] / fx["labels"].numel()
            assert float(ours) == pytest.approx(float(theirs), rel=1e-5)


# ------------------------------------------------------ the trainer's own path


def _trainer(divergence: str) -> DistillationTrainer:
    tokenizer = make_tokenizer()
    prompts = ["hi", "count to three", "abc", "what now?"]
    completions = ["12", "1 2 3", "xyz9", "ok"]
    data = DistillationDataset.from_records(
        {"prompt": [{"role": "user", "content": p}], "completion": c}
        for p, c in zip(prompts, completions, strict=True)
    )
    config = DistillationConfig(
        divergence=divergence,
        jsd_beta=0.5,
        batch_size=4,
        device="cpu",
        max_length=64,
        max_prompt_length=None,
    )
    return DistillationTrainer(
        make_model(0, n_embd=16), tokenizer, config, data, teacher=make_model(7)
    )


def _trl_window_loss(trainer: DistillationTrainer, batches: list[list[Any]]) -> Any:
    """TRL's ``compute_loss`` per micro-batch with the window's token count as
    ``num_items_in_batch`` (its gradient-accumulation normalization), summed."""
    bare = SimpleNamespace(
        teacher_model=trainer.teacher,
        beta=trainer.config.beta,
        use_liger_gkd_loss=False,
        generalized_jsd_loss=GKD.generalized_jsd_loss,
    )
    total_tokens = sum(len(s.response_ids) for batch in batches for s in batch)
    loss = 0.0
    for batch in batches:
        ids, attention, response = _loop.collate(
            batch, trainer.encoder.pad_token_id, trainer.device
        )
        labels = ids.masked_fill(response == 0, -100)
        inputs = {"input_ids": ids, "attention_mask": attention, "labels": labels}
        loss = loss + GKD.compute_loss(
            bare, trainer.model, inputs, num_items_in_batch=total_tokens
        )
    return loss


@pytest.mark.parametrize("divergence", ["forward_kl", "jsd", "reverse_kl"])
def test_trainer_window_loss_matches_trl_compute_loss(divergence: str) -> None:
    trainer = _trainer(divergence)
    sequences = [item.sequence for item in trainer._train_items]
    batches = [sequences[:2], sequences[2:]]
    total = sum(len(s.response_ids) for s in sequences)
    ours = 0.0
    for batch in batches:
        loss = trainer._compute([_Sample(s, 1.0, False) for s in batch])
        ours = ours + reduce_losses(loss.losses, loss.weights, normalizer=total)
    theirs = _trl_window_loss(trainer, batches)
    assert float(ours.detach()) == pytest.approx(float(theirs.detach()), rel=1e-5)
    trainer.model.zero_grad()
    ours.backward()
    ours_grads = [p.grad.clone() for p in trainer.model.parameters()]
    trainer.model.zero_grad()
    _trl_window_loss(trainer, batches).backward()
    for mine, reference in zip(
        ours_grads, (p.grad for p in trainer.model.parameters()), strict=True
    ):
        assert torch.allclose(mine, reference, rtol=1e-4, atol=1e-8)
