"""Bradley-Terry pairwise reward models and their ``RewardFunction`` export.

:class:`PairwiseRewardModel` puts a scalar head on a transformer backbone's
final hidden state at the last non-padding token. :class:`RewardModelTrainer`
fits it with ``-log σ(r_chosen - r_rejected - margin)`` on preference pairs
and reports pairwise accuracy and ROC AUC. :class:`PairwiseRewardFunction`
scores the newest assistant reply of a conversation for the reward contract
(docs/REWARD_CONTRACT.md): ``σ((r - offset) / temperature)`` in ``[0, 1]``,
finite, and deterministic (eval mode, no gradients).
"""

from __future__ import annotations

import importlib
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from stateset_agents.core.checkpoint_io import (
    load_checkpoint_file,
    validate_local_shard_indexes,
)
from stateset_agents.core.reward_base import RewardFunction, RewardResult, RewardType
from stateset_agents.core.reward_context import (
    assistant_turns_in_scope,
    normalize_turns,
)
from stateset_agents.core.trajectory import ConversationTurn

from ..trainer_utils import disable_dropout_in_model
from . import _loop
from .config import RewardModelConfig
from .data import (
    Message,
    PreferenceDataset,
    check_preference_overlap,
    resolve_splits,
)
from .encoding import EncodedSequence, PreferenceEncoder, render_plain_prompt
from .losses import (
    PreferenceLoss,
    bradley_terry_loss,
    pairwise_accuracy,
    pairwise_auc,
)
from .trainer import PreferenceTrainingSummary, _PairItem, encode_pairs

logger = logging.getLogger(__name__)

__all__ = [
    "REWARD_MODEL_SCHEMA",
    "PairwiseRewardFunction",
    "PairwiseRewardModel",
    "RewardModelTrainer",
    "RewardModelTrainingResult",
    "load_reward_function",
    "load_reward_model",
    "save_reward_model",
    "train_reward_model",
]

REWARD_MODEL_SCHEMA = "stateset-reward-model-v1"
_TRAINER_SCHEMA = "stateset-reward-model-checkpoint-v1"
_MODEL_META = "reward_model.json"
_HEAD_FILE = "reward_head.pt"
_TRAINER_META = "reward_model_checkpoint.json"
_STATE_FILE = "trainer_state.pt"


def _hidden_size(backbone: Any) -> int:
    config = getattr(backbone, "config", None)
    for name in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    get_embeddings = getattr(backbone, "get_input_embeddings", None)
    embeddings = get_embeddings() if callable(get_embeddings) else None
    dimension = getattr(embeddings, "embedding_dim", None)
    if isinstance(dimension, int) and dimension > 0:
        return dimension
    raise ValueError("cannot infer the backbone hidden size; pass hidden_size")


class PairwiseRewardModel(nn.Module):
    """A scalar reward head on a transformer backbone.

    The reward of a right-padded sequence is a linear function of the
    backbone's final hidden state at its last attended token.
    """

    def __init__(self, backbone: Any, *, hidden_size: int | None = None) -> None:
        super().__init__()
        self.backbone = backbone
        size = hidden_size or _hidden_size(backbone)
        self.head = nn.Linear(size, 1)
        nn.init.normal_(self.head.weight, std=1.0 / math.sqrt(size + 1))
        nn.init.zeros_(self.head.bias)

    @property
    def hidden_size(self) -> int:
        return int(self.head.in_features)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Rewards ``[B]`` for right-padded ``input_ids``."""
        outputs = _loop.model_outputs(
            self.backbone, input_ids, attention_mask, output_hidden_states=True
        )
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            states = getattr(outputs, "hidden_states", None)
            if not states:
                raise ValueError("the reward backbone returned no hidden states")
            hidden = states[-1]
        last = (attention_mask.sum(-1) - 1).clamp(min=0)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        pooled = hidden[rows, last.to(hidden.device)]
        rewards: torch.Tensor = self.head(pooled.to(self.head.weight.dtype)).squeeze(-1)
        return rewards


def _sequence_rewards(
    model: PairwiseRewardModel,
    encoder: PreferenceEncoder,
    sequences: Sequence[EncodedSequence],
    device: Any,
    mixed_precision: str = "no",
) -> torch.Tensor:
    ids, attention, _ = _loop.collate(sequences, encoder.pad_token_id, device)
    with _loop.autocast(ids.device, mixed_precision):
        rewards: torch.Tensor = model(ids, attention)
    return rewards.float()


class RewardModelTrainer(_loop.WindowedTrainer):
    """Bradley-Terry training of a :class:`PairwiseRewardModel`.

    ``model`` is a ``PairwiseRewardModel`` or a backbone to wrap (a Hugging
    Face base or causal LM). Pairs' ``margin`` is used when
    ``config.use_margin``; pair ``weight`` scales its loss.
    """

    trainer_kind = "reward_model"

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: RewardModelConfig | None = None,
        train_dataset: PreferenceDataset | None = None,
        eval_dataset: PreferenceDataset | None = None,
    ) -> None:
        self.config = config if config is not None else RewardModelConfig()
        self.config.validate()
        if train_dataset is None or not train_dataset.pairs:
            raise ValueError("train_dataset must contain preference pairs")
        for dataset in (train_dataset, eval_dataset):
            if dataset is not None and dataset.unpaired:
                raise ValueError("reward models train on pairs, not unpaired records")
        if eval_dataset is not None:
            check_preference_overlap(train_dataset, eval_dataset)
        _loop.seed_everything(self.config.seed)
        if not isinstance(model, PairwiseRewardModel):
            model = PairwiseRewardModel(model)
        self.model = model
        self.tokenizer = tokenizer
        self.device = _loop.resolve_device(self.config.device, model)
        if self.config.disable_dropout:
            disable_dropout_in_model(model)
        self.encoder = PreferenceEncoder(
            tokenizer,
            prompt_format=self.config.prompt_format,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_prompt_length,
            append_eos=self.config.append_eos,
        )
        self.train_dataset = train_dataset
        self.data_stats: dict[str, Any] = {"truncated": 0}
        self._train_pairs = encode_pairs(self.encoder, train_dataset, self.data_stats)
        self._eval_pairs: list[_PairItem] = (
            encode_pairs(self.encoder, eval_dataset, self.data_stats)
            if eval_dataset is not None
            else []
        )
        self._init_optimization(len(self._train_pairs))

    def _has_eval_data(self) -> bool:
        return bool(self._eval_pairs)

    def _rewards(self, sequences: Sequence[EncodedSequence]) -> torch.Tensor:
        return _sequence_rewards(
            self.model,
            self.encoder,
            sequences,
            self.device,
            self.config.mixed_precision,
        )

    def _pair_loss(
        self, items: Sequence[_PairItem]
    ) -> tuple[PreferenceLoss, torch.Tensor]:
        count = len(items)
        rewards = self._rewards([i.chosen for i in items] + [i.rejected for i in items])
        margin = None
        if self.config.use_margin and any(item.margin is not None for item in items):
            margin = torch.tensor(
                [item.margin or 0.0 for item in items],
                dtype=rewards.dtype,
                device=rewards.device,
            )
        loss = bradley_terry_loss(
            rewards[:count],
            rewards[count:],
            margin=margin,
            center_coefficient=self.config.center_rewards_coefficient,
        )
        weights = torch.tensor(
            [item.weight for item in items], dtype=rewards.dtype, device=rewards.device
        )
        return loss, weights

    def _train_window(self, window: list[list[int]]) -> dict[str, float] | None:
        lr = _loop.current_lr(self.optimizer)

        def compute(batch: list[int]) -> tuple[PreferenceLoss, Any, Any]:
            loss, weights = self._pair_loss([self._train_pairs[i] for i in batch])
            return loss, weights, None

        results = self._accumulate(
            window, lambda index: self._train_pairs[index].weight, compute
        )
        if results is None:
            return None
        chosen: list[float] = []
        rejected: list[float] = []
        for _, (loss, _extra) in results:
            chosen.extend(float(v) for v in loss.chosen_rewards.tolist())
            rejected.extend(float(v) for v in loss.rejected_rewards.tolist())
        return {
            "step": float(self.global_step),
            "epoch": float(self.epoch),
            "learning_rate": lr,
            "loss": sum(value for value, _ in results),
            "accuracy": pairwise_accuracy(chosen, rejected),
            "margin": _loop.mean(chosen) - _loop.mean(rejected),
            "rewards/chosen": _loop.mean(chosen),
            "rewards/rejected": _loop.mean(rejected),
        }

    def train(
        self, *, resume_from: str | Path | None = None
    ) -> PreferenceTrainingSummary:
        """Run the configured epochs (or ``max_steps`` update attempts)."""
        if resume_from is not None:
            self.load_checkpoint(resume_from)
        self._run()
        return PreferenceTrainingSummary(
            method="rm",
            global_step=self.global_step,
            update_attempts=self.update_attempts,
            skipped_steps=self.skipped_steps,
            epochs_completed=self.epoch,
            history=list(self.history),
            eval_metrics=self.evaluate() if self._eval_pairs else None,
            data={
                "train_pairs": len(self._train_pairs),
                "eval_pairs": len(self._eval_pairs),
                **self.data_stats,
            },
        )

    def evaluate(self, dataset: PreferenceDataset | None = None) -> dict[str, float]:
        """Held-out Bradley-Terry loss, pairwise accuracy, ROC AUC and margins."""
        if dataset is not None:
            check_preference_overlap(self.train_dataset, dataset)
            pairs = encode_pairs(self.encoder, dataset, {})
        else:
            pairs = self._eval_pairs
        if not pairs:
            raise ValueError("no evaluation pairs")
        losses: list[float] = []
        weights: list[float] = []
        chosen: list[float] = []
        rejected: list[float] = []
        with _loop.evaluation_mode(self.model):
            for start in range(0, len(pairs), self.config.batch_size):
                loss, weight = self._pair_loss(
                    pairs[start : start + self.config.batch_size]
                )
                losses.extend(loss.losses.float().tolist())
                weights.extend(weight.float().tolist())
                chosen.extend(loss.chosen_rewards.tolist())
                rejected.extend(loss.rejected_rewards.tolist())
        return {
            "eval/loss": sum(
                value * weight for value, weight in zip(losses, weights, strict=True)
            )
            / sum(weights),
            "eval/accuracy": pairwise_accuracy(chosen, rejected),
            "eval/auc": pairwise_auc(chosen, rejected),
            "eval/margin": _loop.mean(chosen) - _loop.mean(rejected),
            "eval/rewards/chosen": _loop.mean(chosen),
            "eval/rewards/rejected": _loop.mean(rejected),
            "eval/num_pairs": float(len(chosen)),
        }

    def score(self, prompt: Any, response: Any) -> float:
        """Raw reward of one response (eval mode, no gradients)."""
        with _loop.evaluation_mode(self.model):
            reward = self._rewards([self.encoder.encode(prompt, response)])
        return float(reward[0])

    def reward_function(self, **kwargs: Any) -> PairwiseRewardFunction:
        """A :class:`PairwiseRewardFunction` scoring with this model."""
        return PairwiseRewardFunction(
            self.model, self.tokenizer, encoder=self.encoder, **kwargs
        )

    def save_checkpoint(self, directory: str | Path) -> Path:
        """Save the reward model (``reward_model/``), optimizer state and
        progress; ``reward_model/`` alone loads with :func:`load_reward_function`."""
        from stateset_agents import __version__

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        save_reward_model(
            self.model, self.tokenizer, path / "reward_model", self.encoder
        )
        self._save_optimization_state(path / _STATE_FILE)
        _loop.write_json(
            path / _TRAINER_META,
            {
                "schema": _TRAINER_SCHEMA,
                "trainer": type(self).__name__,
                "config": self.config.to_dict(),
                "package_version": __version__,
                **self._progress(),
            },
        )
        return path

    def load_checkpoint(self, directory: str | Path, *, trusted: bool = False) -> None:
        """Restore the reward model, optimizer, scheduler and progress."""
        from ..model_checkpoint import load_model_checkpoint

        path = Path(directory)
        meta = _loop.read_json(path / _TRAINER_META)
        if meta.get("schema") != _TRAINER_SCHEMA:
            raise ValueError(f"{path} is not a reward-model checkpoint")
        model_dir = path / "reward_model"
        _read_model_meta(model_dir)
        if not load_model_checkpoint(
            self.model.backbone, model_dir, torch_module=torch, trusted=trusted
        ):
            raise ValueError(f"{model_dir} holds no backbone weights")
        _load_head(self.model, model_dir, trusted=trusted)
        self._load_optimization_state(path / _STATE_FILE, trusted=trusted)
        self._restore_progress(meta)


# ------------------------------------------------------------------ persistence


def _read_model_meta(directory: Path) -> dict[str, Any]:
    meta = _loop.read_json(directory / _MODEL_META)
    if meta.get("schema") != REWARD_MODEL_SCHEMA:
        raise ValueError(f"{directory} is not a saved pairwise reward model")
    return meta


def _load_head(model: PairwiseRewardModel, directory: Path, *, trusted: bool) -> None:
    state = load_checkpoint_file(
        directory / _HEAD_FILE, trusted=trusted, torch_module=torch
    )
    if not isinstance(state, dict):
        raise ValueError("reward head checkpoint must be a state dictionary")
    for name, tensor in state.items():
        if not torch.is_tensor(tensor) or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"invalid reward head tensor: {name}")
    model.head.load_state_dict(state, strict=True)


def save_reward_model(
    model: PairwiseRewardModel,
    tokenizer: Any,
    directory: str | Path,
    encoder: PreferenceEncoder,
) -> Path:
    """Save the backbone (``save_pretrained``), head, tokenizer and the
    encoder settings scoring must reuse."""
    from stateset_agents import __version__

    path = Path(directory)
    _loop.save_model_dir(model.backbone, tokenizer, path)
    torch.save(model.head.state_dict(), path / _HEAD_FILE)
    _loop.write_json(
        path / _MODEL_META,
        {
            "schema": REWARD_MODEL_SCHEMA,
            "backbone_class": type(model.backbone).__name__,
            "hidden_size": model.hidden_size,
            "encoder": encoder.settings(),
            "package_version": __version__,
        },
    )
    return path


def load_reward_model(
    directory: str | Path, *, trusted: bool = False, device: Any = None
) -> tuple[PairwiseRewardModel, Any, PreferenceEncoder]:
    """Load a saved reward model, its tokenizer and its encoder (local files
    only, no remote code)."""
    transformers = importlib.import_module("transformers")
    path = Path(directory)
    meta = _read_model_meta(path)
    validate_local_shard_indexes(path)
    model_cls = getattr(transformers, str(meta.get("backbone_class")), None)
    if not (isinstance(model_cls, type) and hasattr(model_cls, "from_pretrained")):
        model_cls = transformers.AutoModel
    backbone = model_cls.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False
    )
    model = PairwiseRewardModel(backbone, hidden_size=int(meta["hidden_size"]))
    _load_head(model, path, trusted=trusted)
    model.to(torch.device(device) if device is not None else torch.device("cpu"))
    model.eval()
    encoder = PreferenceEncoder(tokenizer, **meta["encoder"])
    return model, tokenizer, encoder


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


class PairwiseRewardFunction(RewardFunction):
    """Score the newest in-scope assistant reply with a pairwise reward model.

    The reply is scored in the context of every turn before it, rendered
    exactly as in training (the model's encoder). The score is
    ``σ((r - offset) / temperature)`` in ``[0, 1]``; ``metadata["raw_reward"]``
    keeps ``r``. A non-finite model output raises ``ValueError``.
    """

    score_range = (0.0, 1.0)

    def __init__(
        self,
        model: PairwiseRewardModel,
        tokenizer: Any,
        *,
        encoder: PreferenceEncoder | None = None,
        temperature: float = 1.0,
        offset: float = 0.0,
        weight: float = 1.0,
        reward_type: RewardType = RewardType.IMMEDIATE,
        name: str = "PairwiseRewardModel",
    ) -> None:
        super().__init__(weight=weight, reward_type=reward_type, name=name)
        for label, value in (("temperature", temperature), ("offset", offset)):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{label} must be a number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{label} must be finite")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = encoder if encoder is not None else PreferenceEncoder(tokenizer)
        self.temperature = float(temperature)
        self.offset = float(offset)

    def _prompt(self, turns: Sequence[ConversationTurn]) -> str | list[Message]:
        messages: list[Message] = [
            {"role": str(turn.role), "content": str(turn.content or "")}
            for turn in turns
        ]
        rendering = (
            "raw"
            if self.encoder.prompt_format == "raw"
            else self.encoder.resolve_format(messages)
        )
        if rendering in {"plain", "raw"}:
            # MultiTurnAgent renders only these roles without a chat template.
            messages = [
                m for m in messages if m["role"] in {"system", "user", "assistant"}
            ]
            if rendering == "raw":
                return render_plain_prompt(messages)
        return messages

    def raw_score(self, prompt: Any, response: Any) -> float:
        """The model's unnormalized reward for ``response`` to ``prompt``."""
        device = next(self.model.parameters()).device
        sequence = self.encoder.encode(prompt, response)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                reward = _sequence_rewards(self.model, self.encoder, [sequence], device)
        finally:
            self.model.train(was_training)
        value = float(reward[0])
        if not math.isfinite(value):
            raise ValueError("the reward model produced a non-finite reward")
        return value

    def normalize(self, raw: float) -> float:
        """Map a raw reward into ``[0, 1]``."""
        return _sigmoid((raw - self.offset) / self.temperature)

    async def compute_reward(
        self, turns: list[ConversationTurn], context: dict[str, Any] | None = None
    ) -> RewardResult:
        normalized = normalize_turns(turns)
        in_scope = assistant_turns_in_scope(normalized, context)
        if in_scope:
            target = in_scope[-1]
            index = next(i for i, turn in enumerate(normalized) if turn is target)
            prompt = self._prompt(normalized[:index])
            response = str(target.content or "")
        else:
            prompt = self._prompt(normalized)
            response = ""
        raw = self.raw_score(prompt, response)
        score = self.normalize(raw)
        return RewardResult(
            score=score,
            breakdown={"reward_model": score},
            metadata={"raw_reward": raw, "reward_model": self.name},
        )


def load_reward_function(
    directory: str | Path,
    *,
    trusted: bool = False,
    device: Any = None,
    **kwargs: Any,
) -> PairwiseRewardFunction:
    """A :class:`PairwiseRewardFunction` from :func:`save_reward_model` output
    (a trainer checkpoint's ``reward_model/`` directory, or the checkpoint)."""
    path = Path(directory)
    if (
        not (path / _MODEL_META).exists()
        and (path / "reward_model" / _MODEL_META).exists()
    ):
        path = path / "reward_model"
    model, tokenizer, encoder = load_reward_model(path, trusted=trusted, device=device)
    return PairwiseRewardFunction(model, tokenizer, encoder=encoder, **kwargs)


# ------------------------------------------------------------------- Python API


@dataclass
class RewardModelTrainingResult:
    """The trainer, its summary and the final checkpoint (when saved)."""

    trainer: RewardModelTrainer
    summary: PreferenceTrainingSummary
    checkpoint: Path | None = None

    def reward_function(self, **kwargs: Any) -> PairwiseRewardFunction:
        return self.trainer.reward_function(**kwargs)


def train_reward_model(
    model: Any,
    train_data: Any,
    *,
    tokenizer: Any = None,
    eval_data: Any = None,
    config: RewardModelConfig | None = None,
    model_revision: str | None = None,
    **overrides: Any,
) -> RewardModelTrainingResult:
    """Train a Bradley-Terry reward model on preference pairs.

    ``model`` is a backbone, a :class:`PairwiseRewardModel`, or a Hugging Face
    model id (loaded with ``AutoModel``, no remote code). Data handling and
    ``overrides`` follow :func:`train_preference`. With ``config.output_dir``
    the final checkpoint is saved to ``<output_dir>/final`` (its
    ``reward_model/`` loads with :func:`load_reward_function`).
    """
    settings = replace(config or RewardModelConfig(), **overrides)
    settings.validate()
    train, evaluation = resolve_splits(
        train_data, eval_data, settings.validation_fraction, settings.seed
    )
    if isinstance(model, str):
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model, revision=model_revision)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(
            model, revision=model_revision, trust_remote_code=False
        )
    if tokenizer is None:
        raise ValueError("pass the tokenizer for a loaded model")
    trainer = RewardModelTrainer(model, tokenizer, settings, train, evaluation)
    summary = trainer.train()
    checkpoint = None
    if settings.output_dir:
        output = Path(settings.output_dir)
        checkpoint = trainer.save_checkpoint(output / "final")
        _loop.write_json(output / "reward_model_metrics.json", summary.to_dict())
    return RewardModelTrainingResult(trainer, summary, checkpoint)
