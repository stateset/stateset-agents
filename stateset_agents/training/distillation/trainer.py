"""Knowledge distillation of a causal LM from a frozen teacher.

:class:`DistillationTrainer` trains a student on
:class:`~stateset_agents.training.distillation.data.DistillationDataset`
examples:

- **offline logit KD**: on each example's completion (teacher-forced), the
  per-token forward KL, reverse KL or generalized JSD between the teacher's
  and the student's ``softmax(logits / temperature)`` over response tokens
  only (exact masks from the shared preference encoder), optionally against
  a precomputed top-k teacher cache so the teacher need not stay in memory;
- **on-policy distillation (GKD)**: with probability ``lmbda`` a micro-batch
  trains on completions the student samples itself (exact sampled token
  ids). The teacher scores those tokens and the loss is the same divergence
  at every sampled position (``on_policy_loss="full"``, GKD) or the
  sampled-token reverse-KL estimator (``"sampled"``).

The teacher is frozen (no gradients, eval mode). Vocabularies must match
(``vocab_alignment="strict"``) or are aligned by token string (``"shared"``);
mismatches raise. Optimization is the preference trainers' loop: AdamW, a
warmup schedule, exact token-weighted gradient accumulation, clipping,
``safe_optimizer_step``, a raise/skip non-finite policy, and checkpoints that
resume exactly.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..preference import _loop
from ..preference.data import Message, check_preference_overlap
from ..preference.encoding import EncodedSequence, PreferenceEncoder
from ..preference.sampling import sample_completions, stop_token_ids
from ..preference.trainer import (
    PreferenceTrainingSummary,
    _apply_lora,
    load_policy_and_tokenizer,
)
from ..trainer_utils import disable_dropout_in_model, require_torch
from .cache import TeacherCache, build_teacher_cache, teacher_response_logits
from .config import DistillationConfig
from .data import DistillationDataset, resolve_distillation_splits
from .losses import (
    DistillationLoss,
    VocabularyMismatchError,
    divergence_from_logprobs,
    per_example_means,
    require_finite,
    sampled_reverse_kl,
    topk_divergence,
)
from .vocab import VocabMapping, logit_width, resolve_vocab_mapping

logger = logging.getLogger(__name__)

__all__ = [
    "CHECKPOINT_SCHEMA",
    "DistillationTrainer",
    "DistillationTrainingResult",
    "precompute_teacher_cache",
    "train_distillation",
]

CHECKPOINT_SCHEMA = "stateset-distillation-checkpoint-v1"
_META_FILE = "distillation_checkpoint.json"
_STATE_FILE = "trainer_state.pt"


@dataclass
class _Item:
    prompt: str | tuple[Message, ...]
    prompt_ids: tuple[int, ...]
    sequence: EncodedSequence | None
    weight: float


@dataclass
class _Sample:
    sequence: EncodedSequence
    weight: float
    on_policy: bool


def _encoder(tokenizer: Any, config: DistillationConfig) -> PreferenceEncoder:
    return PreferenceEncoder(
        tokenizer,
        prompt_format=config.prompt_format,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        append_eos=config.append_eos,
    )


def _encode_items(
    encoder: PreferenceEncoder, dataset: DistillationDataset, stats: dict[str, Any]
) -> list[_Item]:
    items = []
    for index, example in enumerate(dataset.examples):
        try:
            prompt_ids = encoder.encode_prompt(example.prompt_value)
            sequence = None
            if example.completion is not None:
                sequence = encoder.encode(example.prompt_value, example.completion)
                stats["truncated"] = stats.get("truncated", 0) + int(sequence.truncated)
        except ValueError as exc:
            raise ValueError(f"example {index}: {exc}") from exc
        items.append(_Item(example.prompt_value, prompt_ids, sequence, example.weight))
    return items


def _freeze(model: Any, device: Any) -> None:
    if not getattr(model, "hf_device_map", None):
        model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()


def _pad_id(tokenizer: Any, fallback: int) -> int:
    for name in ("pad_token_id", "eos_token_id"):
        value = getattr(tokenizer, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return fallback


class DistillationTrainer(_loop.WindowedTrainer):
    """Distil a frozen ``teacher`` into ``model`` (the student).

    Args:
        model: The student causal LM (a Hugging Face model or any module
            returning ``.logits``; on-policy sampling needs ``generate``).
        tokenizer: The student's tokenizer.
        config: :class:`DistillationConfig`.
        train_dataset: Training examples (completions are required unless
            ``lmbda == 1``).
        eval_dataset: Optional held-out examples (prompt-disjoint).
        teacher: The teacher causal LM. Optional only with ``teacher_top_k``
            and a ``teacher_cache``.
        teacher_tokenizer: The teacher's tokenizer when it differs.
        teacher_cache: A :class:`TeacherCache` (``teacher_top_k`` runs build
            one from ``teacher`` when absent and then release the teacher).
    """

    trainer_kind = "distillation"

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: DistillationConfig | None = None,
        train_dataset: DistillationDataset | None = None,
        eval_dataset: DistillationDataset | None = None,
        *,
        teacher: Any = None,
        teacher_tokenizer: Any = None,
        teacher_cache: TeacherCache | None = None,
    ) -> None:
        require_torch()
        self.config = config if config is not None else DistillationConfig()
        self.config.validate()
        self._check_data(train_dataset, eval_dataset)
        assert train_dataset is not None
        if teacher is not None and teacher is model:
            raise ValueError("the teacher must be a separate (frozen) model")
        _loop.seed_everything(self.config.seed)
        if self.config.use_lora:
            model = _apply_lora(model, self.config)
        self.model = model
        self.tokenizer = tokenizer
        self.device = _loop.resolve_device(self.config.device, model)
        if self.config.disable_dropout:
            disable_dropout_in_model(model)
        self.encoder = _encoder(tokenizer, self.config)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_stats: dict[str, Any] = {"truncated": 0}
        self._train_items = _encode_items(self.encoder, train_dataset, self.data_stats)
        self._eval_items = (
            _encode_items(self.encoder, eval_dataset, self.data_stats)
            if eval_dataset is not None
            else []
        )
        self.teacher_cache: TeacherCache | None = None
        self.teacher = self._init_teacher(teacher, teacher_cache, teacher_tokenizer)
        self._stop_ids = stop_token_ids(model, tokenizer)
        self._suppressed = self.mapping.unsupported_student_ids()
        self._init_optimization(len(self._train_items))

    # ------------------------------------------------------------------ data

    def _check_data(
        self, train: DistillationDataset | None, evaluation: DistillationDataset | None
    ) -> None:
        if train is None or not len(train):
            raise ValueError("train_dataset must contain distillation examples")
        if evaluation is not None:
            check_preference_overlap(train, evaluation)
        if float(self.config.lmbda) < 1.0 and not train.has_completions:
            raise ValueError(
                "every training example needs a completion unless lmbda == 1 "
                "(only on-policy samples)"
            )

    def _has_eval_data(self) -> bool:
        return bool(self._eval_items)

    # --------------------------------------------------------------- teacher

    def _init_teacher(
        self, teacher: Any, cache: TeacherCache | None, teacher_tokenizer: Any
    ) -> Any:
        config = self.config
        if config.teacher_top_k is None and cache is not None:
            raise ValueError("a teacher_cache needs teacher_top_k in the config")
        if teacher is None and cache is None:
            raise ValueError("pass the teacher model (or a teacher_cache)")
        self._teacher_pad_id = _pad_id(
            teacher_tokenizer or self.tokenizer, self.encoder.pad_token_id
        )
        if teacher is not None:
            self.mapping = resolve_vocab_mapping(
                self.model,
                teacher,
                self.tokenizer,
                teacher_tokenizer,
                alignment=config.vocab_alignment,
            )
            _freeze(teacher, self.device)
        else:
            assert cache is not None
            self.mapping = self._mapping_from_cache(cache)
        self.data_stats["vocab"] = self.mapping.stats()
        if config.teacher_top_k is None:
            return teacher
        self.teacher_cache = self._resolve_cache(teacher, cache)
        return None  # the cache replaces the teacher, which can be freed

    def _mapping_from_cache(self, cache: TeacherCache) -> VocabMapping:
        width = logit_width(self.model)
        if not cache.support:
            return VocabMapping.strict(width)
        if width is not None and max(cache.support) >= width:
            raise VocabularyMismatchError(
                "the teacher cache's support has ids beyond the student vocabulary"
            )
        return VocabMapping(cache.support, cache.support, width, None, False)

    def _all_sequences(self) -> list[EncodedSequence]:
        return [
            item.sequence
            for item in (*self._train_items, *self._eval_items)
            if item.sequence is not None
        ]

    def _resolve_cache(self, teacher: Any, cache: TeacherCache | None) -> TeacherCache:
        config = self.config
        assert config.teacher_top_k is not None
        support = () if self.mapping.identity else self.mapping.student_ids
        if cache is None:
            cache = build_teacher_cache(
                teacher,
                self.encoder,
                self.mapping,
                self._all_sequences(),
                top_k=config.teacher_top_k,
                temperature=config.temperature,
                batch_size=config.batch_size,
                device=self.device,
                teacher_pad_id=self._teacher_pad_id,
                mixed_precision=config.mixed_precision,
            )
        cache.check_compatible(
            top_k=config.teacher_top_k,
            temperature=config.temperature,
            encoder=self.encoder.settings(),
            support=support,
        )
        for sequence in self._all_sequences():
            cache.lookup(sequence)  # fail before training, not mid-run
        return cache

    # ----------------------------------------------------------------- loss

    def _teacher_logps(self, sequences: Sequence[EncodedSequence], device: Any) -> Any:
        torch = require_torch()
        logits = teacher_response_logits(
            self.teacher,
            self.mapping,
            sequences,
            pad_id=self.encoder.pad_token_id,
            teacher_pad_id=self._teacher_pad_id,
            device=device,
            mixed_precision=self.config.mixed_precision,
        )
        scaled = logits.float() / float(self.config.temperature)
        return torch.log_softmax(scaled, dim=-1)

    def _cached(self, sequences: Sequence[EncodedSequence], device: Any) -> Any:
        torch = require_torch()
        assert self.teacher_cache is not None
        entries = [self.teacher_cache.lookup(sequence) for sequence in sequences]
        ids = torch.cat([entry[0] for entry in entries]).to(device)
        logps = torch.cat([entry[1] for entry in entries]).to(device)
        return ids, logps

    def _compute(self, samples: Sequence[_Sample]) -> DistillationLoss:
        """Token losses of one micro-batch, reduced to per-example means."""
        torch = require_torch()
        config = self.config
        sequences = [sample.sequence for sample in samples]
        ids, attention, response = _loop.collate(
            sequences, self.encoder.pad_token_id, self.device
        )
        mask = response[:, 1:].bool()
        with _loop.autocast(self.device, config.mixed_precision):
            outputs = _loop.model_outputs(self.model, ids, attention)
        student = self.mapping.student_support(_loop.logits_of(outputs)[:, :-1][mask])
        student_logps = torch.log_softmax(
            student.float() / float(config.temperature), dim=-1
        )
        sampled = samples[0].on_policy and config.on_policy_loss == "sampled"
        sums: dict[str, float] = {}
        if self.teacher_cache is not None:
            teacher_ids, teacher_logps = self._cached(sequences, student.device)
            token_loss = topk_divergence(
                student_logps, teacher_ids, teacher_logps, beta=config.beta
            )
            with torch.no_grad():
                detached = student_logps.detach()
                forward = topk_divergence(detached, teacher_ids, teacher_logps, beta=0)
                reverse = topk_divergence(detached, teacher_ids, teacher_logps, beta=1)
                divergence = token_loss.detach()
        else:
            teacher_logps = self._teacher_logps(sequences, student.device)
            with torch.no_grad():
                detached = student_logps.detach()
                forward = divergence_from_logprobs(detached, teacher_logps, beta=0.0)
                reverse = divergence_from_logprobs(detached, teacher_logps, beta=1.0)
                divergence = divergence_from_logprobs(
                    detached, teacher_logps, beta=config.beta
                )
            if sampled:
                targets = self.mapping.support_index(ids[:, 1:][mask]).unsqueeze(-1)
                token_loss, estimate = sampled_reverse_kl(
                    student_logps.gather(-1, targets).squeeze(-1),
                    teacher_logps.gather(-1, targets).squeeze(-1),
                )
                sums["kl_estimate"] = float(estimate.sum())
            else:
                token_loss = divergence_from_logprobs(
                    student_logps, teacher_logps, beta=config.beta
                )
        require_finite(token_loss, "distillation loss")
        counts = mask.sum(-1)
        rows = mask.nonzero(as_tuple=True)[0]
        losses = per_example_means(token_loss, rows, counts)
        weights = torch.tensor(
            [sample.weight for sample in samples],
            dtype=losses.dtype,
            device=losses.device,
        ) * counts.to(losses.dtype)
        sums.update(
            forward_kl=float(forward.sum()),
            reverse_kl=float(reverse.sum()),
            divergence=float(divergence.sum()),
        )
        return DistillationLoss(losses, weights, sums, int(mask.sum()))

    # -------------------------------------------------------------- sampling

    def _sample(self, items: Sequence[_Item]) -> list[EncodedSequence]:
        """One student completion per item, as exact token ids."""
        config = self.config
        completions = sample_completions(
            self.model,
            [item.prompt_ids for item in items],
            num_samples=1,
            max_new_tokens=config.max_new_tokens,
            temperature=config.sampling_temperature,
            top_p=config.top_p,
            pad_token_id=self.encoder.pad_token_id,
            stop_ids=self._stop_ids,
            batch_size=config.batch_size,
            suppress_tokens=self._suppressed or None,
        )
        sequences = []
        for item, (completion,) in zip(items, completions, strict=True):
            room = config.max_length - len(item.prompt_ids)
            kept = completion[:room]
            sequences.append(
                EncodedSequence(item.prompt_ids, kept, len(kept) < len(completion))
            )
        return sequences

    def _draw_on_policy(self, batch_index: int) -> bool:
        """GKD's λ draw for one micro-batch (deterministic per seed/position)."""
        lmbda = float(self.config.lmbda)
        if lmbda <= 0.0 or lmbda >= 1.0:
            return lmbda >= 1.0
        key = f"{self.config.seed}:{self.update_attempts}:{batch_index}"
        return random.Random(key).random() < lmbda

    def _prepare(
        self, window: list[list[int]]
    ) -> tuple[list[_Sample], list[list[int]]]:
        samples: list[_Sample] = []
        batches: list[list[int]] = []
        for batch_index, batch in enumerate(window):
            items = [self._train_items[i] for i in batch]
            on_policy = self._draw_on_policy(batch_index)
            if on_policy:
                sequences = self._sample(items)
            else:
                sequences = [item.sequence for item in items if item.sequence]
            batches.append(list(range(len(samples), len(samples) + len(items))))
            samples.extend(
                _Sample(sequence, item.weight, on_policy)
                for item, sequence in zip(items, sequences, strict=True)
            )
        return samples, batches

    # -------------------------------------------------------------- training

    def _train_window(self, window: list[list[int]]) -> dict[str, float] | None:
        lr = _loop.current_lr(self.optimizer)
        samples, batches = self._prepare(window)

        def weight_of(index: int) -> float:
            sample = samples[index]
            return sample.weight * len(sample.sequence.response_ids)

        def compute(batch: list[int]) -> tuple[DistillationLoss, Any, Any]:
            loss = self._compute([samples[i] for i in batch])
            return loss, loss.weights, None

        results = self._accumulate(batches, weight_of, compute)
        if results is None:
            return None
        losses = [loss for _, (loss, _extra) in results]
        metrics = {
            "step": float(self.global_step),
            "epoch": float(self.epoch),
            "learning_rate": lr,
            "loss": sum(value for value, _ in results),
            "on_policy_fraction": sum(samples[b[0]].on_policy for b in batches)
            / len(batches),
            **_token_means(losses),
        }
        return metrics

    def train(
        self, *, resume_from: str | Path | None = None
    ) -> PreferenceTrainingSummary:
        """Run the configured epochs (or ``max_steps`` update attempts)."""
        if resume_from is not None:
            self.load_checkpoint(resume_from)
        self._run()
        return PreferenceTrainingSummary(
            method="distillation",
            global_step=self.global_step,
            update_attempts=self.update_attempts,
            skipped_steps=self.skipped_steps,
            epochs_completed=self.epoch,
            history=list(self.history),
            eval_metrics=self._final_eval(),
            data={
                "train_examples": len(self._train_items),
                "eval_examples": len(self._eval_items),
                **self.data_stats,
            },
        )

    def _final_eval(self) -> dict[str, float] | None:
        if not self._eval_items:
            return None
        if any(item.sequence is None for item in self._eval_items):
            return self.evaluate(on_policy=True) if self.teacher is not None else None
        return self.evaluate()

    # ------------------------------------------------------------ evaluation

    def evaluate(
        self, dataset: DistillationDataset | None = None, *, on_policy: bool = False
    ) -> dict[str, float]:
        """Held-out token-mean forward KL, reverse KL and configured
        divergence (eval mode, no gradients).

        Teacher-forced on the examples' completions, or (``on_policy=True``)
        on one completion per prompt sampled from the student with a fixed
        seed (``eval/on_policy/*``; needs the live teacher).
        """
        torch = require_torch()
        if dataset is not None:
            check_preference_overlap(self.train_dataset, dataset)
            items = _encode_items(self.encoder, dataset, {})
        else:
            items = self._eval_items
        if not items:
            raise ValueError("no evaluation data")
        prefix = "eval/on_policy/" if on_policy else "eval/"
        if on_policy:
            if self.teacher is None:
                raise ValueError("on-policy evaluation needs the live teacher")
            devices = [self.device.index or 0] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(self.config.seed))
                sequences = self._sample(items)
        elif any(item.sequence is None for item in items):
            raise ValueError("teacher-forced evaluation needs completions")
        else:
            sequences = [item.sequence for item in items if item.sequence]
        losses = []
        with _loop.evaluation_mode(self.model, self.teacher):
            size = self.config.batch_size
            for start in range(0, len(sequences), size):
                chunk = [
                    _Sample(sequence, 1.0, False)
                    for sequence in sequences[start : start + size]
                ]
                losses.append(self._compute(chunk))
        means = _token_means(losses)
        return {prefix + name: value for name, value in means.items()}

    def divergence_report(self, prompt: Any, response: Any) -> dict[str, float]:
        """Token-mean divergences on one teacher-forced ``(prompt, response)``."""
        sequence = self.encoder.encode(prompt, response)
        with _loop.evaluation_mode(self.model, self.teacher):
            loss = self._compute([_Sample(sequence, 1.0, False)])
        return _token_means([loss])

    # ----------------------------------------------------------- checkpoints

    def save_checkpoint(self, directory: str | Path) -> Path:
        """Save the student (``model/``), optimizer/scheduler/RNG state and
        metadata (written last)."""
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
                "config": self.config.to_dict(),
                "encoder": self.encoder.settings(),
                "vocab": self.mapping.stats(),
                "lora": bool(getattr(self.model, "peft_config", None)),
                "package_version": __version__,
                **self._progress(),
            },
        )
        return path

    def load_checkpoint(self, directory: str | Path, *, trusted: bool = False) -> None:
        """Restore the student, optimizer, scheduler, RNG and progress
        (weights-only loads unless ``trusted=True``)."""
        from ..model_checkpoint import load_model_checkpoint

        path = Path(directory)
        meta = _loop.read_json(path / _META_FILE)
        if meta.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError(f"{path} is not a distillation checkpoint")
        saved = meta.get("config") or {}
        for name in ("divergence", "jsd_beta", "temperature", "lmbda"):
            if saved.get(name) != getattr(self.config, name):
                raise ValueError(
                    f"checkpoint {name} {saved.get(name)!r} differs from "
                    f"{getattr(self.config, name)!r}"
                )
        if not load_model_checkpoint(
            self.model, path / "model", torch_module=require_torch(), trusted=trusted
        ):
            raise ValueError(f"{path / 'model'} holds no model weights")
        self._load_optimization_state(path / _STATE_FILE, trusted=trusted)
        self._restore_progress(meta)


def _token_means(losses: Sequence[DistillationLoss]) -> dict[str, float]:
    """Token means of the summed diagnostics of several micro-batches."""
    totals: dict[str, float] = defaultdict(float)
    tokens = 0
    sampled_tokens = 0
    for loss in losses:
        for name, value in loss.token_sums.items():
            totals[name] += value
        tokens += loss.tokens
        if "kl_estimate" in loss.token_sums:
            sampled_tokens += loss.tokens
    means = {
        name: value / max(tokens, 1)
        for name, value in totals.items()
        if name != "kl_estimate"
    }
    if sampled_tokens:
        means["kl_estimate"] = totals["kl_estimate"] / sampled_tokens
    means["tokens"] = float(tokens)
    return means


# ------------------------------------------------------------------- Python API


@dataclass
class DistillationTrainingResult:
    """The trainer, its summary and the final checkpoint (when saved)."""

    trainer: DistillationTrainer
    summary: PreferenceTrainingSummary
    checkpoint: Path | None = None


def precompute_teacher_cache(
    teacher: Any,
    tokenizer: Any,
    data: Any,
    config: DistillationConfig,
    *,
    student: Any = None,
    teacher_tokenizer: Any = None,
    device: str = "auto",
) -> TeacherCache:
    """Score every completion of ``data`` with ``teacher`` once and return
    its top-``config.teacher_top_k`` cache (save it with ``.save(path)``).

    ``tokenizer`` is the student's; pass ``student`` (or its logit width
    holder) so ``strict`` alignment can check the vocabularies now.
    """
    config.validate()
    if config.teacher_top_k is None:
        raise ValueError("set teacher_top_k in the config")
    dataset = data if isinstance(data, DistillationDataset) else None
    if dataset is None:
        dataset, _ = resolve_distillation_splits(data, None, 0.0, config.seed)
    encoder = _encoder(tokenizer, config)
    items = _encode_items(encoder, dataset, {})
    sequences = [item.sequence for item in items if item.sequence is not None]
    if not sequences:
        raise ValueError("the data holds no completions to cache")
    mapping = resolve_vocab_mapping(
        student, teacher, tokenizer, teacher_tokenizer, alignment=config.vocab_alignment
    )
    resolved = _loop.resolve_device(device, teacher)
    _freeze(teacher, resolved)
    return build_teacher_cache(
        teacher,
        encoder,
        mapping,
        sequences,
        top_k=config.teacher_top_k,
        temperature=config.temperature,
        batch_size=config.batch_size,
        device=resolved,
        teacher_pad_id=_pad_id(teacher_tokenizer or tokenizer, encoder.pad_token_id),
        mixed_precision=config.mixed_precision,
    )


def train_distillation(
    model: Any,
    train_data: Any,
    *,
    teacher: Any = None,
    tokenizer: Any = None,
    teacher_tokenizer: Any = None,
    eval_data: Any = None,
    config: DistillationConfig | None = None,
    teacher_cache: TeacherCache | str | Path | None = None,
    model_revision: str | None = None,
    teacher_revision: str | None = None,
    **overrides: Any,
) -> DistillationTrainingResult:
    """Distil ``teacher`` into ``model`` and return the result.

    ``model`` and ``teacher`` are loaded causal LMs (pass ``tokenizer`` for a
    loaded student) or Hugging Face model ids. ``train_data``/``eval_data``
    are datasets, JSONL paths or records; without ``eval_data``,
    ``config.validation_fraction > 0`` holds out prompt-disjoint groups.
    ``teacher_cache`` may be a :class:`TeacherCache` or a saved cache path
    (with ``teacher_top_k`` set). ``overrides`` replace config fields. With
    ``config.output_dir`` the final checkpoint is saved to
    ``<output_dir>/final`` and metrics to ``distillation_metrics.json``.
    """
    settings = replace(config or DistillationConfig(), **overrides)
    settings.validate()
    train, evaluation = resolve_distillation_splits(
        train_data, eval_data, settings.validation_fraction, settings.seed
    )
    if isinstance(model, str):
        model, tokenizer = load_policy_and_tokenizer(model, revision=model_revision)
    if tokenizer is None:
        raise ValueError("pass the tokenizer for a loaded student model")
    if isinstance(teacher, str):
        teacher, loaded = load_policy_and_tokenizer(teacher, revision=teacher_revision)
        teacher_tokenizer = teacher_tokenizer or loaded
    if isinstance(teacher_cache, str | Path):
        teacher_cache = TeacherCache.load(teacher_cache)
    trainer = DistillationTrainer(
        model,
        tokenizer,
        settings,
        train,
        evaluation,
        teacher=teacher,
        teacher_tokenizer=teacher_tokenizer,
        teacher_cache=teacher_cache,
    )
    summary = trainer.train()
    checkpoint = None
    if settings.output_dir:
        output = Path(settings.output_dir)
        checkpoint = trainer.save_checkpoint(output / "final")
        _loop.write_json(output / "distillation_metrics.json", summary.to_dict())
    return DistillationTrainingResult(trainer, summary, checkpoint)
