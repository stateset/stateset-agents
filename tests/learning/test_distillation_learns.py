"""Distillation moves a tiny student's token distribution onto a tiny teacher's.

The teacher is a tiny GPT-2 fine-tuned (60 supervised steps) to answer any
prompt with digits, a sharp behaviour: ~0.99 of its first-token mass is on
the ten digit tokens, against ~0.10 for the untrained student. Training data
is the teacher's own samples on the training prompts (offline KD on
sequence-level-KD data); held-out data is its samples on different prompts.

Learning is measured on the held-out prompts as the token-mean forward
KL(teacher ‖ student) on the teacher's completions (teacher-forced) and the
reverse KL(student ‖ teacher) on the student's own samples (on-policy, fixed
seed), plus the student's digit mass. Every mode must cut both KLs by at least
65% and lift the digit mass above 0.6.

Negative control: the same runs against a *shuffled* teacher (its logits
permuted over the vocabulary by a fixed permutation, so it is exactly as
sharp but on the wrong tokens) must not reduce the KL to the real teacher at
all, nor lift the digit mass anywhere near the teacher's (it stays below 0.4).
"""

from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("transformers")

import torch  # noqa: E402

from tests.learning._toy import (  # noqa: E402
    HELDOUT_PROMPTS,
    TRAIN_PROMPTS,
    make_model,
    make_tokenizer,
    seeds,
    target_mass,
)

pytestmark = [pytest.mark.learning, pytest.mark.timeout(300)]

EXTRA_PROMPTS = ["alpha", "beta gamma", "zz top", "quick", "number please", "hey"]
STEPS = 30
KL_RATIO = 0.35  # after <= 35% of before: a drop of at least 65%


def _messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def _teacher(seed: int, tokenizer: Any) -> Any:
    """A tiny GPT-2 fine-tuned to answer every prompt with four digits."""
    from stateset_agents.training.preference import _loop
    from stateset_agents.training.preference.encoding import PreferenceEncoder

    model = make_model(1000 + seed)
    encoder = PreferenceEncoder(tokenizer, max_length=64)
    rng = random.Random(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(60):
        sequences = [
            encoder.encode(
                _messages(p), "".join(rng.choice("0123456789") for _ in "1234")
            )
            for p in TRAIN_PROMPTS + EXTRA_PROMPTS
        ]
        ids, attention, response = _loop.collate(sequences, 0, torch.device("cpu"))
        labels = ids.masked_fill(response == 0, -100)
        loss = model(input_ids=ids, attention_mask=attention, labels=labels).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model


class ShuffledTeacher(torch.nn.Module):
    """The teacher with its logits permuted over the vocabulary."""

    def __init__(self, inner: Any, seed: int) -> None:
        super().__init__()
        self.inner = inner
        generator = torch.Generator().manual_seed(seed)
        size = inner.config.vocab_size
        self.register_buffer("permutation", torch.randperm(size, generator=generator))

    def forward(self, input_ids: Any, attention_mask: Any = None, **_: Any) -> Any:
        logits = self.inner(input_ids=input_ids, attention_mask=attention_mask).logits
        return SimpleNamespace(logits=logits[..., self.permutation])


def _teacher_data(
    teacher: Any, tokenizer: Any, prompts: list[str], n: int, seed: int
) -> Any:
    from stateset_agents.training.distillation import DistillationDataset
    from stateset_agents.training.preference.encoding import PreferenceEncoder
    from stateset_agents.training.preference.sampling import sample_completions

    encoder = PreferenceEncoder(tokenizer, max_length=64)
    torch.manual_seed(seed)
    completions = sample_completions(
        teacher,
        [encoder.encode_prompt(_messages(p)) for p in prompts],
        num_samples=n,
        max_new_tokens=6,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=int(tokenizer.pad_token_id),
        stop_ids=[int(tokenizer.eos_token_id)],
        batch_size=16,
    )
    rows = []
    for prompt, group in zip(prompts, completions, strict=True):
        for tokens in group:
            text = tokenizer.decode(list(tokens), skip_special_tokens=True)
            if text:
                rows.append({"prompt": _messages(prompt), "completion": text})
    return DistillationDataset.from_records(rows)


MODES: dict[str, dict[str, Any]] = {
    "offline_forward_kl": {},
    "offline_jsd": {"divergence": "jsd", "jsd_beta": 0.5},
    "offline_topk_cache": {"teacher_top_k": 8},
    # Pure on-policy JSD(0.5) at the common 5e-3 collapsed on seed 2 (the
    # bounded JSD saturates once the student's samples leave the teacher's
    # support, and its gradient vanishes); 2e-3 is stable on every seed.
    "gkd_on_policy_jsd": {
        "lmbda": 1.0,
        "divergence": "jsd",
        "jsd_beta": 0.5,
        "learning_rate": 2e-3,
    },
    "gkd_on_policy_reverse_kl": {"lmbda": 1.0, "divergence": "reverse_kl"},
    "gkd_mixed": {"lmbda": 0.5},
    "on_policy_sampled_reverse_kl": {
        "lmbda": 1.0,
        "divergence": "reverse_kl",
        "on_policy_loss": "sampled",
    },
}


def _distil(seed: int, mode: str, *, shuffled: bool = False) -> dict[str, float]:
    from stateset_agents.training.distillation import (
        DistillationConfig,
        DistillationTrainer,
    )

    tokenizer = make_tokenizer()
    teacher = _teacher(seed, tokenizer)
    train = _teacher_data(teacher, tokenizer, TRAIN_PROMPTS, 4, seed)
    heldout = _teacher_data(teacher, tokenizer, HELDOUT_PROMPTS, 2, 999)
    common: dict[str, Any] = {
        "learning_rate": 5e-3,
        "batch_size": 8,
        "max_steps": STEPS,
        "lr_scheduler": "constant",
        "max_length": 64,
        "max_prompt_length": None,
        "max_new_tokens": 6,
        "device": "cpu",
        "seed": seed,
    }
    student = make_model(seed)
    # Measured against the real teacher whatever the student trains on.
    probe = DistillationTrainer(
        student,
        tokenizer,
        DistillationConfig(**common),
        train,
        heldout,
        teacher=teacher,
    )
    before = {**probe.evaluate(), **probe.evaluate(on_policy=True)}
    mass_before = target_mass(student, tokenizer)
    used = ShuffledTeacher(teacher, seed) if shuffled else teacher
    trainer = DistillationTrainer(
        student,
        tokenizer,
        DistillationConfig(**{**common, **MODES[mode]}),
        train,
        heldout,
        teacher=used,
    )
    summary = trainer.train()
    assert summary.global_step == STEPS and summary.skipped_steps == 0
    after = {**probe.evaluate(), **probe.evaluate(on_policy=True)}
    return {
        "forward_before": before["eval/forward_kl"],
        "forward_after": after["eval/forward_kl"],
        "reverse_before": before["eval/on_policy/reverse_kl"],
        "reverse_after": after["eval/on_policy/reverse_kl"],
        "mass_before": mass_before,
        "mass_after": target_mass(student, tokenizer),
    }


@pytest.mark.parametrize("mode", sorted(MODES))
@pytest.mark.parametrize("seed", seeds())
def test_student_distribution_moves_to_the_teacher(seed: int, mode: str) -> None:
    # Observed (seeds 0/1/2): see docs/DISTILLATION.md "Evidence".
    result = _distil(seed, mode)
    assert result["forward_after"] <= KL_RATIO * result["forward_before"], result
    assert result["reverse_after"] <= KL_RATIO * result["reverse_before"], result
    assert result["mass_after"] >= 0.6 > result["mass_before"], result


@pytest.mark.parametrize("mode", ["offline_forward_kl", "on_policy_sampled_reverse_kl"])
@pytest.mark.parametrize("seed", seeds())
def test_shuffled_teacher_control_does_not_move_toward_the_teacher(
    seed: int, mode: str
) -> None:
    result = _distil(seed, mode, shuffled=True)
    assert result["forward_after"] >= result["forward_before"], result
    assert result["reverse_after"] >= result["reverse_before"], result
    assert result["mass_after"] < 0.4, result
