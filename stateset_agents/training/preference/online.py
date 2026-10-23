"""Online (iterative) DPO: sample, judge, pair, train; repeat.

Each round of :class:`OnlineDPOTrainer`:

1. the current policy samples ``samples_per_prompt`` completions per prompt
   (exact token ids, :mod:`.sampling`);
2. a judge scores each decoded completion: any ``RewardFunction`` through the
   shared :func:`~stateset_agents.core.reward_context.score_response` path
   (a :class:`~.reward_model.PairwiseRewardFunction` included). Only
   ``RewardUnavailableError`` drops a completion; every other failure, and
   a non-finite score, raises;
3. each prompt's best completion is paired against its worst when their
   scores differ by more than zero and by at least ``min_score_margin``
   (ties and too-small gaps are skipped; among equal scores the earliest
   sample is used, a uniform choice since samples are i.i.d.);
4. the policy is DPO- (or IPO-) trained on that round's pairs by a
   :class:`~.trainer.PreferenceTrainer` on the exact sampled tokens, against
   a reference that is either the initial policy (``fixed``) or the policy
   as the previous round left it (``previous_round``).

Rounds checkpoint (policy, round summaries, the round's pairs as JSONL) and
resume at round granularity.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import logging
import math
import random
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

from stateset_agents.core.reward_context import RewardUnavailableError, score_response
from stateset_agents.data.identity import prompt_identity

from ..trainer_utils import disable_dropout_in_model, require_torch
from . import _loop
from .config import OnlineDPOConfig, PreferenceConfig
from .data import Message, PreferenceDataset, PreferencePair, _identity_text, _prompt
from .encoding import EncodedSequence, PreferenceEncoder
from .sampling import sample_completions, stop_token_ids
from .trainer import (
    PreferenceTrainer,
    _apply_lora,
    _PairItem,
    load_policy_and_tokenizer,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ONLINE_SCHEMA",
    "OnlineDPOResult",
    "OnlineDPOTrainer",
    "OnlinePrompt",
    "OnlineRoundSummary",
    "select_pair",
    "train_online_dpo",
]

ONLINE_SCHEMA = "stateset-online-dpo-checkpoint-v1"
_META_FILE = "online_dpo_checkpoint.json"
_T = TypeVar("_T")


@dataclass(frozen=True)
class OnlinePrompt:
    """A prompt to sample from, with the scenario its reward reads (e.g. a
    ``ground_truth``)."""

    prompt: str | tuple[Message, ...]
    scenario: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        value = _prompt(self.prompt)
        object.__setattr__(
            self, "prompt", value if isinstance(value, str) else tuple(value)
        )
        if self.scenario is not None and not isinstance(self.scenario, Mapping):
            raise ValueError("scenario must be a mapping")

    @property
    def identity(self) -> str:
        return prompt_identity(_identity_text(self.prompt))


def _as_prompts(prompts: Any, name: str) -> list[OnlinePrompt]:
    if isinstance(prompts, str | Mapping):
        raise ValueError(f"{name} must be a sequence of prompts")
    result = []
    for index, prompt in enumerate(prompts):
        try:
            if isinstance(prompt, OnlinePrompt):
                result.append(prompt)
            elif isinstance(prompt, Mapping):
                text = prompt.get("prompt", prompt.get("messages"))
                result.append(OnlinePrompt(text, prompt.get("scenario")))
            else:
                result.append(OnlinePrompt(prompt))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{name}[{index}]: {exc}") from exc
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def select_pair(
    scores: Sequence[float | None], *, min_margin: float = 0.0
) -> tuple[int, int] | str:
    """``(best, worst)`` sample indices, or why the prompt yields no pair.

    ``None`` scores (reward unavailable) are ignored. Reasons:
    ``too_few_scores`` (fewer than two scored samples), ``tie`` (best equals
    worst) and ``below_margin`` (the gap is under ``min_margin``). Among
    equal scores the earliest sample is chosen.
    """
    scored = [(value, index) for index, value in enumerate(scores) if value is not None]
    if len(scored) < 2:
        return "too_few_scores"
    for value, _ in scored:
        if not math.isfinite(value):
            raise ValueError(f"reward score must be finite, got {value}")
    best = max(scored, key=lambda item: (item[0], -item[1]))
    worst = min(scored, key=lambda item: (item[0], item[1]))
    gap = best[0] - worst[0]
    if gap <= 0:
        return "tie"
    if gap < min_margin:
        return "below_margin"
    return best[1], worst[1]


@dataclass
class OnlineRoundSummary:
    """What one online round sampled, scored, paired and trained."""

    round: int
    prompts: int
    samples: int
    scored: int
    unavailable: int
    pairs: int
    skipped: dict[str, int] = field(default_factory=dict)
    mean_score: float | None = None
    mean_chosen_score: float | None = None
    mean_rejected_score: float | None = None
    training: dict[str, Any] | None = None
    eval_metrics: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


class _RoundTrainer(PreferenceTrainer):
    """A :class:`PreferenceTrainer` on pre-encoded (exactly sampled) pairs."""

    def __init__(
        self,
        *args: Any,
        encoded: Sequence[tuple[EncodedSequence, EncodedSequence]],
        **kwargs: Any,
    ) -> None:
        self._encoded = list(encoded)
        super().__init__(*args, **kwargs)

    def _encode(self, dataset: PreferenceDataset) -> tuple[list[_PairItem], list[Any]]:
        if dataset is not self.train_dataset:
            return super()._encode(dataset)
        return (
            [
                _PairItem(chosen, rejected, pair.weight)
                for (chosen, rejected), pair in zip(
                    self._encoded, dataset.pairs, strict=True
                )
            ],
            [],
        )


def _run_sync(factory: Callable[[], Coroutine[Any, Any, _T]]) -> _T:
    """Run ``factory()`` (a coroutine) to completion from synchronous code,
    also when an event loop is already running in this thread."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(factory())).result()


class OnlineDPOTrainer:
    """Iterative on-policy DPO against a judge.

    Args:
        model: The policy (a Hugging Face causal LM with ``generate``).
        tokenizer: Its tokenizer.
        reward_fn: The judge: any ``RewardFunction`` (scored through
            ``score_response``), e.g. a ``PairwiseRewardFunction``.
        prompts: Training prompts: strings, message lists, ``OnlinePrompt``
            or ``{"prompt"|"messages": ..., "scenario": {...}}`` mappings.
        config: :class:`OnlineDPOConfig`.
        ref_model: Optional frozen reference for the first round (the
            ``fixed`` reference throughout). Defaults to a frozen copy of the
            initial policy, or the adapter-disabled policy under LoRA.
        eval_prompts: Held-out prompts (prompt-disjoint) scored after every
            round by :meth:`evaluate`.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        reward_fn: Any,
        prompts: Any,
        config: OnlineDPOConfig | None = None,
        *,
        ref_model: Any = None,
        eval_prompts: Any = None,
    ) -> None:
        require_torch()
        self.config = config if config is not None else OnlineDPOConfig()
        self.config.validate()
        if reward_fn is None or not callable(
            getattr(reward_fn, "compute_reward", None)
        ):
            raise ValueError("reward_fn must be a RewardFunction (compute_reward)")
        self.prompts = _as_prompts(prompts, "prompts")
        self.eval_prompts = (
            _as_prompts(eval_prompts, "eval_prompts")
            if eval_prompts is not None
            else []
        )
        shared = {p.identity for p in self.prompts} & {
            p.identity for p in self.eval_prompts
        }
        if shared:
            raise ValueError(
                f"Train/evaluation overlap: {len(shared)} shared prompt identities"
            )
        _loop.seed_everything(self.config.seed)
        if self.config.use_lora:
            model = _apply_lora(model, self.config)
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
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
        self._prompt_ids = [self.encoder.encode_prompt(p.prompt) for p in self.prompts]
        self._stop_ids = stop_token_ids(model, tokenizer)
        self._initial_reference = self._init_reference(ref_model)
        self.round = 0
        self.history: list[OnlineRoundSummary] = []
        self.last_pairs: PreferenceDataset | None = None

    # ------------------------------------------------------------- reference

    def _frozen_copy(self, model: Any) -> Any:
        reference = copy.deepcopy(model)
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        reference.eval()
        return reference

    def _init_reference(self, ref_model: Any) -> Any:
        if ref_model is not None:
            if not getattr(ref_model, "hf_device_map", None):
                ref_model.to(self.device)
            for parameter in ref_model.parameters():
                parameter.requires_grad_(False)
            ref_model.eval()
            return ref_model
        if getattr(self.model, "peft_config", None) and callable(
            getattr(self.model, "disable_adapter", None)
        ):
            return None  # the adapter-disabled policy is the initial policy
        return self._frozen_copy(self.model)

    def reference_for_round(self) -> Any:
        """The frozen reference the next round trains against (``None``: the
        adapter-disabled policy)."""
        if self.config.reference_update == "previous_round" and self.round > 0:
            return self._frozen_copy(self.model)
        return self._initial_reference

    # -------------------------------------------------------------- sampling

    def _seed_sampling(self, round_index: int, purpose: str) -> None:
        torch = require_torch()
        key = f"{self.config.seed}:{purpose}:{round_index}"
        torch.manual_seed(random.Random(key).getrandbits(31))

    def _sample(
        self, prompt_ids: Sequence[tuple[int, ...]], num_samples: int
    ) -> list[list[tuple[int, ...]]]:
        config = self.config
        return sample_completions(
            self.model,
            prompt_ids,
            num_samples=num_samples,
            max_new_tokens=config.max_new_tokens,
            temperature=config.sampling_temperature,
            top_p=config.top_p,
            pad_token_id=self.encoder.pad_token_id,
            stop_ids=self._stop_ids,
            batch_size=config.generation_batch_size,
        )

    def _decode(self, tokens: Sequence[int]) -> str:
        return str(self.tokenizer.decode(list(tokens), skip_special_tokens=True))

    # --------------------------------------------------------------- scoring

    async def score_completions(
        self, prompts: Sequence[OnlinePrompt], texts: Sequence[Sequence[str]]
    ) -> list[list[float | None]]:
        """Judge scores per prompt and completion (``None``: unavailable).

        Scores go through ``score_response``; only ``RewardUnavailableError``
        is absorbed (unless the reward sets ``fail_on_error``), any other
        error and any non-finite score propagates.
        """
        strict = bool(getattr(self.reward_fn, "fail_on_error", False))

        async def one(prompt: OnlinePrompt, text: str) -> float | None:
            try:
                score = await score_response(
                    self.reward_fn, prompt.prompt, text, prompt.scenario
                )
            except RewardUnavailableError as error:
                if strict:
                    raise
                logger.warning("Dropping a completion: reward unavailable: %s", error)
                return None
            if not math.isfinite(score):
                raise ValueError(f"reward score must be finite, got {score}")
            return float(score)

        flat = await asyncio.gather(
            *(
                one(prompt, text)
                for prompt, group in zip(prompts, texts, strict=True)
                for text in group
            )
        )
        result: list[list[float | None]] = []
        offset = 0
        for group in texts:
            result.append(list(flat[offset : offset + len(group)]))
            offset += len(group)
        return result

    def _score(
        self, prompts: Sequence[OnlinePrompt], texts: Sequence[Sequence[str]]
    ) -> list[list[float | None]]:
        return _run_sync(lambda: self.score_completions(prompts, texts))

    # ----------------------------------------------------------------- pairs

    def _build_pairs(
        self,
        completions: Sequence[Sequence[tuple[int, ...]]],
        texts: Sequence[Sequence[str]],
        scores: Sequence[Sequence[float | None]],
        summary: OnlineRoundSummary,
    ) -> tuple[list[PreferencePair], list[tuple[EncodedSequence, EncodedSequence]]]:
        pairs: list[PreferencePair] = []
        encoded: list[tuple[EncodedSequence, EncodedSequence]] = []
        chosen_scores: list[float] = []
        rejected_scores: list[float] = []
        for index, prompt in enumerate(self.prompts):
            decision = select_pair(
                scores[index], min_margin=self.config.min_score_margin
            )
            if isinstance(decision, str):
                summary.skipped[decision] = summary.skipped.get(decision, 0) + 1
                continue
            best, worst = decision
            best_score, worst_score = scores[index][best], scores[index][worst]
            assert best_score is not None and worst_score is not None
            try:
                pair = PreferencePair(
                    prompt=prompt.prompt,
                    chosen=texts[index][best],
                    rejected=texts[index][worst],
                    metadata={
                        "round": self.round,
                        "chosen_score": best_score,
                        "rejected_score": worst_score,
                    },
                )
            except ValueError:  # e.g. the two decode to the same text
                summary.skipped["identical_text"] = (
                    summary.skipped.get("identical_text", 0) + 1
                )
                continue
            prompt_ids = self._prompt_ids[index]
            encoded.append(
                (
                    self._sequence(prompt_ids, completions[index][best]),
                    self._sequence(prompt_ids, completions[index][worst]),
                )
            )
            pairs.append(pair)
            chosen_scores.append(best_score)
            rejected_scores.append(worst_score)
        summary.pairs = len(pairs)
        summary.mean_chosen_score = _loop.mean(chosen_scores) if pairs else None
        summary.mean_rejected_score = _loop.mean(rejected_scores) if pairs else None
        return pairs, encoded

    def _sequence(
        self, prompt_ids: tuple[int, ...], completion: tuple[int, ...]
    ) -> EncodedSequence:
        room = self.config.max_length - len(prompt_ids)
        kept = completion[:room]
        return EncodedSequence(prompt_ids, kept, len(kept) < len(completion))

    # -------------------------------------------------------------- training

    def _round_config(self) -> PreferenceConfig:
        values = {
            item.name: getattr(self.config, item.name)
            for item in fields(PreferenceConfig)
        }
        values.update(
            seed=int(self.config.seed) + self.round,
            output_dir=None,
            save_steps=None,
            eval_steps=None,
        )
        return PreferenceConfig(**values)

    def run_round(self) -> OnlineRoundSummary:
        """Sample, judge, pair and train one round."""
        self._seed_sampling(self.round, "train")
        completions = self._sample(self._prompt_ids, self.config.samples_per_prompt)
        texts = [[self._decode(tokens) for tokens in group] for group in completions]
        scores = self._score(self.prompts, texts)
        available = [s for group in scores for s in group if s is not None]
        total = sum(len(group) for group in scores)
        if not available:
            raise RewardUnavailableError(
                f"every one of {total} sampled completions failed to get a reward"
            )
        summary = OnlineRoundSummary(
            round=self.round,
            prompts=len(self.prompts),
            samples=total,
            scored=len(available),
            unavailable=total - len(available),
            pairs=0,
            mean_score=_loop.mean(available),
        )
        pairs, encoded = self._build_pairs(completions, texts, scores, summary)
        self.last_pairs = PreferenceDataset(tuple(pairs))
        if pairs:
            trainer = _RoundTrainer(
                self.model,
                self.tokenizer,
                self._round_config(),
                self.last_pairs,
                ref_model=self.reference_for_round(),
                encoded=encoded,
            )
            summary.training = trainer.train().to_dict()
        else:
            logger.warning("Online round %d produced no preference pairs", self.round)
        if self.eval_prompts:
            summary.eval_metrics = self.evaluate()
        self.history.append(summary)
        self.round += 1
        return summary

    def train(
        self, *, resume_from: str | Path | None = None
    ) -> list[OnlineRoundSummary]:
        """Run the remaining rounds (checkpointing each under ``output_dir``)."""
        if resume_from is not None:
            self.load_checkpoint(resume_from)
        while self.round < self.config.rounds:
            self.run_round()
            if self.config.output_dir:
                self.save_checkpoint(
                    Path(self.config.output_dir) / f"round-{self.round - 1}"
                )
        return list(self.history)

    # ------------------------------------------------------------ evaluation

    def evaluate(
        self, prompts: Any = None, *, samples_per_prompt: int | None = None
    ) -> dict[str, float]:
        """Mean judge score of fresh policy samples on held-out prompts
        (a fixed sampling seed per round; unavailable rewards excluded)."""
        torch = require_torch()
        targets = (
            _as_prompts(prompts, "prompts")
            if prompts is not None
            else self.eval_prompts
        )
        if not targets:
            raise ValueError("no evaluation prompts")
        count = samples_per_prompt or self.config.samples_per_prompt
        ids = [self.encoder.encode_prompt(p.prompt) for p in targets]
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            self._seed_sampling(0, "eval")
            completions = self._sample(ids, count)
        texts = [[self._decode(tokens) for tokens in group] for group in completions]
        scores = [
            s for group in self._score(targets, texts) for s in group if s is not None
        ]
        if not scores:
            raise RewardUnavailableError("no evaluation completion got a reward")
        return {
            "eval/mean_score": _loop.mean(scores),
            "eval/scored": float(len(scores)),
        }

    # ----------------------------------------------------------- checkpoints

    def save_checkpoint(self, directory: str | Path) -> Path:
        """Save the policy (``model/``), the last round's pairs
        (``pairs.jsonl``) and the round history (metadata written last)."""
        from stateset_agents import __version__

        from .data import write_preference_jsonl

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        _loop.save_model_dir(self.model, self.tokenizer, path / "model")
        if self.last_pairs is not None and len(self.last_pairs):
            write_preference_jsonl(self.last_pairs, path / "pairs.jsonl")
        _loop.write_json(
            path / _META_FILE,
            {
                "schema": ONLINE_SCHEMA,
                "rounds_completed": self.round,
                "config": self.config.to_dict(),
                "encoder": self.encoder.settings(),
                "history": [summary.to_dict() for summary in self.history],
                "lora": bool(getattr(self.model, "peft_config", None)),
                "package_version": __version__,
            },
        )
        return path

    def load_checkpoint(self, directory: str | Path, *, trusted: bool = False) -> None:
        """Restore the policy and round counter. Construct the trainer from
        the original model first: a ``fixed`` reference comes from it."""
        from ..model_checkpoint import load_model_checkpoint

        path = Path(directory)
        meta = _loop.read_json(path / _META_FILE)
        if meta.get("schema") != ONLINE_SCHEMA:
            raise ValueError(f"{path} is not an online DPO checkpoint")
        if not load_model_checkpoint(
            self.model, path / "model", torch_module=require_torch(), trusted=trusted
        ):
            raise ValueError(f"{path / 'model'} holds no model weights")
        self.round = int(meta["rounds_completed"])
        self.history = [
            OnlineRoundSummary(**entry) for entry in meta.get("history", [])
        ]


# ------------------------------------------------------------------- Python API


@dataclass
class OnlineDPOResult:
    """The trainer, its per-round summaries and the final checkpoint."""

    trainer: OnlineDPOTrainer
    rounds: list[OnlineRoundSummary]
    checkpoint: Path | None = None


def train_online_dpo(
    model: Any,
    reward_fn: Any,
    prompts: Any,
    *,
    tokenizer: Any = None,
    eval_prompts: Any = None,
    config: OnlineDPOConfig | None = None,
    ref_model: Any = None,
    model_revision: str | None = None,
    **overrides: Any,
) -> OnlineDPOResult:
    """Run online DPO rounds and return the result.

    ``model`` is a loaded causal LM (pass ``tokenizer``) or a Hugging Face
    model id; ``overrides`` replace config fields. With ``config.output_dir``
    every round is checkpointed to ``<output_dir>/round-<r>`` and the final
    policy to ``<output_dir>/final``.
    """
    from dataclasses import replace

    settings = replace(config or OnlineDPOConfig(), **overrides)
    settings.validate()
    if isinstance(model, str):
        model, tokenizer = load_policy_and_tokenizer(model, revision=model_revision)
    if tokenizer is None:
        raise ValueError("pass the tokenizer for a loaded model")
    trainer = OnlineDPOTrainer(
        model,
        tokenizer,
        reward_fn,
        prompts,
        settings,
        ref_model=ref_model,
        eval_prompts=eval_prompts,
    )
    rounds = trainer.train()
    checkpoint = None
    if settings.output_dir:
        checkpoint = trainer.save_checkpoint(Path(settings.output_dir) / "final")
    return OnlineDPOResult(trainer, rounds, checkpoint)
