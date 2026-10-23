"""Maintained preference optimization and pairwise reward models.

- Data: :class:`PreferencePair`, :class:`UnpairedExample`,
  :class:`PreferenceDataset`, :func:`load_preference_jsonl` and
  prompt-disjoint :func:`split_preference_dataset`.
- Policy optimization: :class:`PreferenceTrainer` / :func:`train_preference`
  for DPO (and conservative DPO), IPO, SimPO, KTO and ORPO.
- Reward models: :class:`RewardModelTrainer` / :func:`train_reward_model`
  (Bradley-Terry) and :class:`PairwiseRewardFunction`, a ``RewardFunction``
  for the RL trainers.
- Losses: pure functions in :mod:`.losses`.

See docs/PREFERENCE_TRAINING.md. Stability: provisional (not part of the v1
contract in docs/API_COMPATIBILITY.md). Importing this package does not
import torch; exports resolve lazily.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, str] = {
    "PreferencePair": "data",
    "UnpairedExample": "data",
    "PreferenceDataset": "data",
    "load_preference_jsonl": "data",
    "write_preference_jsonl": "data",
    "parse_preference_record": "data",
    "split_preference_dataset": "data",
    "check_preference_overlap": "data",
    "group_preference_records": "data",
    "as_preference_dataset": "data",
    "resolve_splits": "data",
    "PreferenceEncoder": "encoding",
    "EncodedSequence": "encoding",
    "PROMPT_FORMATS": "encoding",
    "PREFERENCE_METHODS": "config",
    "REFERENCE_FREE_METHODS": "config",
    "PreferenceConfig": "config",
    "RewardModelConfig": "config",
    "PreferenceLoss": "losses",
    "NonFiniteLossError": "losses",
    "dpo_loss": "losses",
    "ipo_loss": "losses",
    "simpo_loss": "losses",
    "kto_loss": "losses",
    "kto_kl_estimate": "losses",
    "orpo_loss": "losses",
    "bradley_terry_loss": "losses",
    "sequence_logprobs": "losses",
    "reduce_losses": "losses",
    "pairwise_accuracy": "losses",
    "pairwise_auc": "losses",
    "PreferenceTrainer": "trainer",
    "PreferenceTrainingResult": "trainer",
    "PreferenceTrainingSummary": "trainer",
    "train_preference": "trainer",
    "PairwiseRewardModel": "reward_model",
    "PairwiseRewardFunction": "reward_model",
    "RewardModelTrainer": "reward_model",
    "RewardModelTrainingResult": "reward_model",
    "train_reward_model": "reward_model",
    "save_reward_model": "reward_model",
    "load_reward_model": "reward_model",
    "load_reward_function": "reward_model",
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
