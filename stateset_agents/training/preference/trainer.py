"""Maintained offline preference optimization: DPO, cDPO, IPO, SimPO, KTO, ORPO.

:class:`PreferenceTrainer` trains a causal language model on
:class:`~stateset_agents.training.preference.data.PreferenceDataset` records:

- response-token log-probs only (exact masks from :mod:`.encoding`),
  shift-by-one aligned by the shared ``rl_losses.gather_token_logprobs``;
- a frozen reference: a deep copy of the policy, an explicit ``ref_model``,
  or (with LoRA) the policy with its adapters disabled; reference log-probs
  are computed under ``no_grad`` per batch or precomputed once
  (``precompute_reference_logps``); SimPO and ORPO need none;
- AdamW, a warmup + linear/cosine/constant schedule, gradient accumulation
  with exact weighted-mean windows, ``max_grad_norm`` clipping and a
  ``raise``/``skip`` policy for non-finite losses and gradients, with updates
  committed through ``trainer_utils.safe_optimizer_step``;
- evaluation metrics (reward accuracy, margins, chosen/rejected rewards and
  log-probs) and checkpoints loaded through ``core.checkpoint_io``.
"""

from __future__ import annotations

import copy
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..trainer_utils import disable_dropout_in_model, require_torch
from . import _loop
from .config import PreferenceConfig
from .data import (
    PreferenceDataset,
    UnpairedExample,
    check_preference_overlap,
    resolve_splits,
)
from .encoding import EncodedSequence, PreferenceEncoder
from .losses import (
    PreferenceLoss,
    dpo_loss,
    ipo_loss,
    kto_kl_estimate,
    kto_loss,
    orpo_loss,
    pairwise_accuracy,
    sequence_logprobs,
    simpo_loss,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CHECKPOINT_SCHEMA",
    "PreferenceTrainer",
    "PreferenceTrainingResult",
    "PreferenceTrainingSummary",
    "encode_pairs",
    "load_policy_and_tokenizer",
    "train_preference",
]

CHECKPOINT_SCHEMA = "stateset-preference-checkpoint-v1"
_META_FILE = "preference_checkpoint.json"
_STATE_FILE = "trainer_state.pt"


@dataclass
class _PairItem:
    chosen: EncodedSequence
    rejected: EncodedSequence
    weight: float
    margin: float | None = None
    ref_chosen: float | None = None
    ref_rejected: float | None = None


@dataclass
class _UnpairedItem:
    sequence: EncodedSequence
    kl_sequence: EncodedSequence
    label: bool
    weight: float
    ref: float | None = None
    ref_kl: float | None = None


@dataclass
class PreferenceTrainingSummary:
    """What a preference or reward-model training run did."""

    method: str
    global_step: int
    update_attempts: int
    skipped_steps: int
    epochs_completed: int
    history: list[dict[str, float]] = field(default_factory=list)
    eval_metrics: dict[str, float] | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "global_step": self.global_step,
            "update_attempts": self.update_attempts,
            "skipped_steps": self.skipped_steps,
            "epochs_completed": self.epochs_completed,
            "history": [dict(entry) for entry in self.history],
            "eval_metrics": self.eval_metrics,
            "data": dict(self.data),
        }


def encode_pairs(
    encoder: PreferenceEncoder, dataset: PreferenceDataset, stats: dict[str, Any]
) -> list[_PairItem]:
    """Encode every pair of ``dataset`` (``stats["truncated"]`` counts cuts)."""
    items = []
    for index, pair in enumerate(dataset.pairs):
        sides = []
        for name, response in (("chosen", pair.chosen), ("rejected", pair.rejected)):
            try:
                sequence = encoder.encode(pair.prompt_value, response)
            except ValueError as exc:
                raise ValueError(f"pair {index} {name}: {exc}") from exc
            stats["truncated"] = stats.get("truncated", 0) + int(sequence.truncated)
            sides.append(sequence)
        items.append(_PairItem(sides[0], sides[1], pair.weight, pair.margin))
    return items


def _apply_lora(model: Any, config: Any, *, task_type: str = "CAUSAL_LM") -> Any:
    """Wrap ``model`` in a PEFT LoRA adapter on its text stack.

    ``config`` carries ``lora_r``/``lora_alpha``/``lora_dropout``/
    ``lora_target_modules``; ``task_type`` is a ``peft.TaskType`` name
    (``FEATURE_EXTRACTION`` for a reward-model backbone).
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "use_lora needs peft: pip install 'stateset-agents[training]'"
        ) from exc
    from stateset_agents.core.lora_targets import text_lora_targets

    if getattr(model, "peft_config", None):
        return model
    if config.lora_target_modules:
        targets = text_lora_targets(model, list(config.lora_target_modules))
    else:
        from ..sft import infer_lora_target_modules

        targets = infer_lora_target_modules(model)
    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=targets or None,
        bias="none",
        task_type=TaskType[task_type],
    )
    return get_peft_model(model, lora)


class PreferenceTrainer(_loop.WindowedTrainer):
    """Offline preference optimization of a causal LM.

    Args:
        model: The policy (a Hugging Face causal LM or any module returning
            ``.logits`` for ``input_ids``/``attention_mask``).
        tokenizer: Its tokenizer.
        config: :class:`PreferenceConfig`; ``config.method`` picks the loss.
        train_dataset: Training records. KTO also accepts unpaired examples
            and splits pairs into desirable/undesirable examples.
        eval_dataset: Optional held-out records (must not share prompts with
            ``train_dataset``).
        ref_model: Optional frozen reference. Defaults to a frozen deep copy
            of the policy, or the adapter-disabled policy under LoRA.
    """

    trainer_kind = "preference"

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: PreferenceConfig | None = None,
        train_dataset: PreferenceDataset | None = None,
        eval_dataset: PreferenceDataset | None = None,
        *,
        ref_model: Any = None,
    ) -> None:
        require_torch()
        self.config = config if config is not None else PreferenceConfig()
        self.config.validate()
        if train_dataset is None or not len(train_dataset):
            raise ValueError("train_dataset must contain preference records")
        if eval_dataset is not None:
            check_preference_overlap(train_dataset, eval_dataset)
        _loop.seed_everything(self.config.seed)
        if self.config.use_lora:
            model = _apply_lora(model, self.config)
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
        self._logprob_dtype = None
        if self.config.logprob_dtype is not None:
            from ..rl_losses import resolve_logprob_dtype

            self._logprob_dtype = resolve_logprob_dtype(self.config.logprob_dtype)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_stats: dict[str, Any] = {"truncated": 0}
        self._train_pairs, self._train_unpaired = self._encode(train_dataset)
        if self.method == "kto":
            self._train_pairs = []  # KTO trains on the unpaired examples
        self._eval_pairs: list[_PairItem] = []
        self._eval_unpaired: list[_UnpairedItem] = []
        if eval_dataset is not None and len(eval_dataset):
            self._eval_pairs, self._eval_unpaired = self._encode(eval_dataset)
        self.ref_model = self._init_reference(ref_model)
        self._owns_reference = ref_model is None and self.ref_model is not None
        self._reference_released = False
        self._init_optimization(len(self._training_items()))
        if self.config.precompute_reference_logps:
            self._precompute_reference()

    # ------------------------------------------------------------------ data

    @property
    def method(self) -> str:
        return str(self.config.method)

    def _training_items(self) -> Sequence[_PairItem | _UnpairedItem]:
        return self._train_unpaired if self.method == "kto" else self._train_pairs

    def _has_eval_data(self) -> bool:
        return bool(self._eval_pairs or self._eval_unpaired)

    def _encode(
        self, dataset: PreferenceDataset
    ) -> tuple[list[_PairItem], list[_UnpairedItem]]:
        pairs = encode_pairs(self.encoder, dataset, self.data_stats)
        if self.method != "kto":
            if dataset.unpaired:
                raise ValueError(
                    "unpaired (completion/label) records need method='kto'"
                )
            return pairs, []
        examples = dataset.kto_examples()
        if len(examples) < 2:
            raise ValueError("KTO needs at least two examples for its KL estimate")
        labels = {example.label for example in examples}
        if len(labels) < 2:
            logger.warning("KTO data has only one label class")
        return pairs, self._encode_unpaired(examples)

    def _encode_unpaired(
        self, examples: Sequence[UnpairedExample]
    ) -> list[_UnpairedItem]:
        sequences = []
        for index, example in enumerate(examples):
            try:
                sequence = self.encoder.encode(example.prompt_value, example.completion)
            except ValueError as exc:
                raise ValueError(f"example {index}: {exc}") from exc
            self.data_stats["truncated"] += int(sequence.truncated)
            sequences.append(sequence)
        # A fixed derangement pairs each prompt with another example's
        # completion, so reference KL log-probs can be precomputed.
        order = list(range(len(sequences)))
        random.Random(self.config.seed).shuffle(order)
        partner = {order[k]: order[(k + 1) % len(order)] for k in range(len(order))}
        return [
            _UnpairedItem(
                sequence=sequence,
                kl_sequence=sequence.with_response(sequences[partner[index]]),
                label=example.label,
                weight=example.weight,
            )
            for index, (sequence, example) in enumerate(
                zip(sequences, examples, strict=True)
            )
        ]

    # ------------------------------------------------------------- reference

    def _init_reference(self, ref_model: Any) -> Any:
        if self.config.reference_free:
            if ref_model is not None:
                raise ValueError(f"{self.method} is reference-free; drop ref_model")
            return None
        if ref_model is None:
            if getattr(self.model, "peft_config", None) and callable(
                getattr(self.model, "disable_adapter", None)
            ):
                return None  # the adapter-disabled policy is the reference
            ref_model = copy.deepcopy(self.model)
        if not getattr(ref_model, "hf_device_map", None):
            ref_model.to(self.device)
        for parameter in ref_model.parameters():
            parameter.requires_grad_(False)
        ref_model.eval()
        return ref_model

    def _reference_logps(self, sequences: Sequence[EncodedSequence]) -> tuple[Any, Any]:
        torch = require_torch()
        if self._reference_released:
            raise RuntimeError(
                "the reference model was released after precomputing reference "
                "log-probs; pass evaluation data to the trainer's constructor"
            )
        with torch.no_grad():
            if self.ref_model is not None:
                return self._logps(self.ref_model, sequences)
            with self.model.disable_adapter():
                return self._logps(self.model, sequences)

    def _precompute_reference(self) -> None:
        size = self.config.batch_size
        with _loop.evaluation_mode(self.model):
            for pairs in (self._train_pairs, self._eval_pairs):
                for start in range(0, len(pairs), size):
                    chunk = pairs[start : start + size]
                    sums, _ = self._reference_logps(
                        [i.chosen for i in chunk] + [i.rejected for i in chunk]
                    )
                    values = sums.float().tolist()
                    for offset, item in enumerate(chunk):
                        item.ref_chosen = values[offset]
                        item.ref_rejected = values[len(chunk) + offset]
            for unpaired in (self._train_unpaired, self._eval_unpaired):
                for start in range(0, len(unpaired), size):
                    part = unpaired[start : start + size]
                    sums, _ = self._reference_logps(
                        [i.sequence for i in part] + [i.kl_sequence for i in part]
                    )
                    values = sums.float().tolist()
                    for offset, entry in enumerate(part):
                        entry.ref = values[offset]
                        entry.ref_kl = values[len(part) + offset]
        if self._owns_reference:
            self.ref_model = None
            self._reference_released = True

    # ---------------------------------------------------------------- losses

    def _logps(
        self, model: Any, sequences: Sequence[EncodedSequence]
    ) -> tuple[Any, Any]:
        ids, attention, response = _loop.collate(
            sequences, self.encoder.pad_token_id, self.device
        )
        with _loop.autocast(self.device, self.config.mixed_precision):
            outputs = _loop.model_outputs(model, ids, attention)
        return sequence_logprobs(
            _loop.logits_of(outputs), ids, response, dtype=self._logprob_dtype
        )

    def sequence_logps(self, prompt: Any, responses: Sequence[Any]) -> list[float]:
        """Summed response log-probs of ``responses`` to ``prompt`` under the
        policy (eval mode, no gradients)."""
        sequences = [self.encoder.encode(prompt, response) for response in responses]
        with _loop.evaluation_mode(self.model):
            sums, _ = self._logps(self.model, sequences)
        return [float(value) for value in sums.float().tolist()]

    def _pair_loss(self, items: Sequence[_PairItem]) -> tuple[PreferenceLoss, Any]:
        torch = require_torch()
        count = len(items)
        sums, lengths = self._logps(
            self.model, [i.chosen for i in items] + [i.rejected for i in items]
        )
        policy_chosen, policy_rejected = sums[:count], sums[count:]
        chosen_lengths, rejected_lengths = lengths[:count], lengths[count:]
        config = self.config
        if self.method == "simpo":
            loss = simpo_loss(
                policy_chosen,
                policy_rejected,
                chosen_lengths,
                rejected_lengths,
                beta=config.beta,
                gamma=config.simpo_gamma,
                label_smoothing=config.label_smoothing,
            )
        elif self.method == "orpo":
            loss = orpo_loss(
                policy_chosen,
                policy_rejected,
                chosen_lengths,
                rejected_lengths,
                beta=config.beta,
            )
        else:
            if all(item.ref_chosen is not None for item in items):
                reference = torch.tensor(
                    [item.ref_chosen for item in items]
                    + [item.ref_rejected for item in items],
                    dtype=sums.dtype,
                    device=sums.device,
                )
            else:
                reference, _ = self._reference_logps(
                    [i.chosen for i in items] + [i.rejected for i in items]
                )
            reference = reference.to(sums.dtype)
            ref_chosen, ref_rejected = reference[:count], reference[count:]
            if self.method == "ipo":
                loss = ipo_loss(
                    policy_chosen,
                    policy_rejected,
                    ref_chosen,
                    ref_rejected,
                    chosen_lengths,
                    rejected_lengths,
                    beta=config.beta,
                )
            else:  # dpo, and KTO's paired evaluation rewards
                loss = dpo_loss(
                    policy_chosen,
                    policy_rejected,
                    ref_chosen,
                    ref_rejected,
                    beta=config.beta,
                    label_smoothing=config.label_smoothing,
                )
        weights = torch.tensor(
            [item.weight for item in items], dtype=loss.losses.dtype, device=sums.device
        )
        return loss, weights

    def _kto_loss(
        self, items: Sequence[_UnpairedItem]
    ) -> tuple[PreferenceLoss, Any, float]:
        torch = require_torch()
        count = len(items)
        sums, _ = self._logps(self.model, [i.sequence for i in items])
        with torch.no_grad():
            policy_kl, _ = self._logps(self.model, [i.kl_sequence for i in items])
        if all(item.ref is not None for item in items):
            reference = torch.tensor(
                [item.ref for item in items] + [item.ref_kl for item in items],
                dtype=sums.dtype,
                device=sums.device,
            )
        else:
            reference, _ = self._reference_logps(
                [i.sequence for i in items] + [i.kl_sequence for i in items]
            )
        reference = reference.to(sums.dtype)
        kl = kto_kl_estimate(policy_kl.to(sums.dtype), reference[count:])
        labels = torch.tensor([item.label for item in items], device=sums.device)
        loss = kto_loss(
            sums,
            reference[:count],
            labels,
            kl,
            beta=self.config.beta,
            desirable_weight=self.config.kto_desirable_weight,
            undesirable_weight=self.config.kto_undesirable_weight,
        )
        weights = torch.tensor(
            [item.weight for item in items], dtype=loss.losses.dtype, device=sums.device
        )
        return loss, weights, float(kl)

    # -------------------------------------------------------------- training

    def _train_window(self, window: list[list[int]]) -> dict[str, float] | None:
        lr = _loop.current_lr(self.optimizer)
        if self.method == "kto":

            def compute(batch: list[int]) -> tuple[PreferenceLoss, Any, Any]:
                return self._kto_loss([self._train_unpaired[i] for i in batch])

            def weight_of(index: int) -> float:
                return self._train_unpaired[index].weight

        else:

            def compute(batch: list[int]) -> tuple[PreferenceLoss, Any, Any]:
                loss, weights = self._pair_loss([self._train_pairs[i] for i in batch])
                return loss, weights, None

            def weight_of(index: int) -> float:
                return self._train_pairs[index].weight

        results = self._accumulate(window, weight_of, compute)
        if results is None:
            return None
        chosen: list[float] = []
        rejected: list[float] = []
        for _, (loss, _extra) in results:
            chosen.extend(float(v) for v in loss.chosen_rewards.float().tolist())
            rejected.extend(float(v) for v in loss.rejected_rewards.float().tolist())
        metrics = {
            "step": float(self.global_step),
            "epoch": float(self.epoch),
            "learning_rate": lr,
            "loss": sum(value for value, _ in results),
            "rewards/chosen": _loop.mean(chosen),
            "rewards/rejected": _loop.mean(rejected),
        }
        if self.method == "kto":
            metrics["kl"] = _loop.mean([float(extra) for _, (_l, extra) in results])
        else:
            metrics["rewards/margins"] = (
                metrics["rewards/chosen"] - metrics["rewards/rejected"]
            )
            metrics["rewards/accuracies"] = pairwise_accuracy(chosen, rejected)
        return metrics

    def train(
        self, *, resume_from: str | Path | None = None
    ) -> PreferenceTrainingSummary:
        """Run the configured epochs (or ``max_steps`` update attempts)."""
        if resume_from is not None:
            self.load_checkpoint(resume_from)
        self._run()
        return PreferenceTrainingSummary(
            method=self.method,
            global_step=self.global_step,
            update_attempts=self.update_attempts,
            skipped_steps=self.skipped_steps,
            epochs_completed=self.epoch,
            history=list(self.history),
            eval_metrics=self.evaluate() if self._has_eval_data() else None,
            data={
                "train_pairs": len(self._train_pairs),
                "train_examples": len(self._train_unpaired),
                "eval_pairs": len(self._eval_pairs),
                "eval_examples": len(self._eval_unpaired),
                **self.data_stats,
            },
        )

    # ------------------------------------------------------------ evaluation

    def evaluate(self, dataset: PreferenceDataset | None = None) -> dict[str, float]:
        """Held-out loss and implicit-reward metrics (eval mode, no grad).

        Rewards are each method's implicit rewards: ``β(log π - log π_ref)``
        for DPO/IPO/KTO, ``β`` × mean token log-prob for SimPO/ORPO.
        """
        if dataset is not None:
            check_preference_overlap(self.train_dataset, dataset)
            pairs, unpaired = self._encode(dataset)
        else:
            pairs, unpaired = self._eval_pairs, self._eval_unpaired
        if not pairs and not unpaired:
            raise ValueError("no evaluation data")
        size = self.config.batch_size
        losses: list[float] = []
        weights: list[float] = []
        chosen: list[float] = []
        rejected: list[float] = []
        logps_chosen: list[float] = []
        logps_rejected: list[float] = []
        kls: list[float] = []
        with _loop.evaluation_mode(self.model, self.ref_model):
            for start in range(0, len(pairs), size):
                chunk = pairs[start : start + size]
                loss, weight = self._pair_loss(chunk)
                if self.method != "kto":
                    losses.extend(loss.losses.float().tolist())
                    weights.extend(weight.float().tolist())
                chosen.extend(loss.chosen_rewards.float().tolist())
                rejected.extend(loss.rejected_rewards.float().tolist())
                sums, _ = self._logps(
                    self.model, [i.chosen for i in chunk] + [i.rejected for i in chunk]
                )
                values = sums.float().tolist()
                logps_chosen.extend(values[: len(chunk)])
                logps_rejected.extend(values[len(chunk) :])
            for start in range(0, len(unpaired), size):
                loss, weight, kl = self._kto_loss(unpaired[start : start + size])
                losses.extend(loss.losses.float().tolist())
                weights.extend(weight.float().tolist())
                kls.append(kl)
        metrics: dict[str, float] = {}
        if losses:
            metrics["eval/loss"] = sum(
                value * weight for value, weight in zip(losses, weights, strict=True)
            ) / sum(weights)
        if kls:
            metrics["eval/kl"] = _loop.mean(kls)
        if chosen:
            metrics.update(
                {
                    "eval/rewards/chosen": _loop.mean(chosen),
                    "eval/rewards/rejected": _loop.mean(rejected),
                    "eval/rewards/margins": _loop.mean(chosen) - _loop.mean(rejected),
                    "eval/rewards/accuracies": pairwise_accuracy(chosen, rejected),
                    "eval/logps/chosen": _loop.mean(logps_chosen),
                    "eval/logps/rejected": _loop.mean(logps_rejected),
                    "eval/num_pairs": float(len(chosen)),
                }
            )
        return metrics

    # ----------------------------------------------------------- checkpoints

    def save_checkpoint(self, directory: str | Path) -> Path:
        """Save weights (``model/``), optimizer/scheduler state and metadata.

        ``model/`` is a ``save_pretrained`` directory (an adapter under LoRA)
        with the tokenizer. The metadata file is written last.
        """
        from stateset_agents import __version__

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        _loop.save_model_dir(self.model, self.tokenizer, path / "model")
        self._save_optimization_state(path / _STATE_FILE)
        _loop.write_json(
            path / _META_FILE,
            {
                "schema": CHECKPOINT_SCHEMA,
                "trainer": type(self).__name__,
                "method": self.method,
                "config": self.config.to_dict(),
                "encoder": self.encoder.settings(),
                "lora": bool(getattr(self.model, "peft_config", None)),
                "package_version": __version__,
                **self._progress(),
            },
        )
        return path

    def load_checkpoint(self, directory: str | Path, *, trusted: bool = False) -> None:
        """Restore weights, optimizer, scheduler and progress counters.

        Weights load through ``training.model_checkpoint`` and trainer state
        through ``core.checkpoint_io`` (weights-only unless ``trusted=True``).
        Construct the trainer from the original model first: the reference
        (and any precomputed reference log-probs) comes from it.
        """
        from ..model_checkpoint import load_model_checkpoint

        path = Path(directory)
        meta = _loop.read_json(path / _META_FILE)
        if meta.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError(f"{path} is not a preference checkpoint")
        if meta.get("method") != self.method:
            raise ValueError(
                f"checkpoint method {meta.get('method')!r} differs from "
                f"{self.method!r}"
            )
        if not load_model_checkpoint(
            self.model, path / "model", torch_module=require_torch(), trusted=trusted
        ):
            raise ValueError(f"{path / 'model'} holds no model weights")
        self._load_optimization_state(path / _STATE_FILE, trusted=trusted)
        self._restore_progress(meta)


# ------------------------------------------------------------------- Python API


@dataclass
class PreferenceTrainingResult:
    """The trainer, its summary and the final checkpoint (when saved)."""

    trainer: PreferenceTrainer
    summary: PreferenceTrainingSummary
    checkpoint: Path | None = None


def load_policy_and_tokenizer(
    model_name: str, *, revision: str | None = None
) -> tuple[Any, Any]:
    """Load a causal LM and tokenizer for preference training (no remote code)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from stateset_agents.core.transformers_compat import load_generation_model

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    model, _ = load_generation_model(
        AutoModelForCausalLM,
        model_name,
        {"revision": revision, "trust_remote_code": False},
        for_training=True,
    )
    return model, tokenizer


def train_preference(
    model: Any,
    train_data: Any,
    *,
    tokenizer: Any = None,
    eval_data: Any = None,
    config: PreferenceConfig | None = None,
    ref_model: Any = None,
    model_revision: str | None = None,
    **overrides: Any,
) -> PreferenceTrainingResult:
    """Train ``model`` on preference data and return the result.

    ``model`` is a loaded causal LM (pass ``tokenizer``) or a Hugging Face
    model id. ``train_data``/``eval_data`` are datasets, JSONL paths or
    records. Without ``eval_data``, ``config.validation_fraction > 0`` holds
    out prompt-disjoint groups. ``overrides`` replace config fields, e.g.
    ``train_preference(model, "pairs.jsonl", tokenizer=tok, method="simpo")``.
    With ``config.output_dir`` the final checkpoint is saved to
    ``<output_dir>/final`` and metrics to ``preference_metrics.json``.
    """
    settings = replace(config or PreferenceConfig(), **overrides)
    settings.validate()
    train, evaluation = resolve_splits(
        train_data, eval_data, settings.validation_fraction, settings.seed
    )
    if isinstance(model, str):
        model, tokenizer = load_policy_and_tokenizer(model, revision=model_revision)
    if tokenizer is None:
        raise ValueError("pass the tokenizer for a loaded model")
    trainer = PreferenceTrainer(
        model, tokenizer, settings, train, evaluation, ref_model=ref_model
    )
    summary = trainer.train()
    checkpoint = None
    if settings.output_dir:
        output = Path(settings.output_dir)
        checkpoint = trainer.save_checkpoint(output / "final")
        _loop.write_json(output / "preference_metrics.json", summary.to_dict())
    return PreferenceTrainingResult(trainer, summary, checkpoint)
