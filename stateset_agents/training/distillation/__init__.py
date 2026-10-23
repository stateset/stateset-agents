"""Knowledge distillation from a frozen teacher (offline KD and on-policy GKD).

- Data: :class:`DistillationExample` / :class:`DistillationDataset` (a prompt
  and an optional completion), :func:`load_distillation_jsonl` and the
  prompt-disjoint :func:`split_distillation_dataset`.
- Training: :class:`DistillationTrainer` / :func:`train_distillation`:
  forward KL, reverse KL or GKD's generalized JSD over response tokens, with
  temperature; GKD's ``lmbda`` mix of student samples and dataset
  completions; the sampled-token reverse-KL on-policy objective; a top-k
  teacher cache (:class:`TeacherCache`, :func:`precompute_teacher_cache`).
- Vocabularies: :func:`resolve_vocab_mapping` (``strict`` or token-string
  ``shared`` alignment); mismatches raise :class:`VocabularyMismatchError`.
- Losses: pure functions in :mod:`.losses`, pinned to TRL's
  ``GKDTrainer.generalized_jsd_loss``.

See docs/DISTILLATION.md. Stability: provisional (not part of the v1
contract). Importing this package does not import torch; exports resolve
lazily.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, str] = {
    "DistillationExample": "data",
    "DistillationDataset": "data",
    "load_distillation_jsonl": "data",
    "write_distillation_jsonl": "data",
    "parse_distillation_record": "data",
    "split_distillation_dataset": "data",
    "resolve_distillation_splits": "data",
    "DIVERGENCES": "config",
    "ON_POLICY_LOSSES": "config",
    "VOCAB_ALIGNMENTS": "config",
    "DistillationConfig": "config",
    "DistillationLoss": "losses",
    "VocabularyMismatchError": "losses",
    "divergence_from_logprobs": "losses",
    "generalized_jsd_loss": "losses",
    "sampled_reverse_kl": "losses",
    "token_divergence": "losses",
    "topk_divergence": "losses",
    "topk_logprobs": "losses",
    "VocabMapping": "vocab",
    "resolve_vocab_mapping": "vocab",
    "TeacherCache": "cache",
    "DistillationTrainer": "trainer",
    "DistillationTrainingResult": "trainer",
    "precompute_teacher_cache": "trainer",
    "train_distillation": "trainer",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
