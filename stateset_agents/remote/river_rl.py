"""Validated configuration and learning-signal contracts for River RL.

This module has no River SDK dependency. Token normalization is over retained,
nonzero-advantage spans; excluded groups do not enter the denominator.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RiverRLConfig:
    """Controls for synchronous, group-centered River updates."""

    rounds: int = 4
    best_of: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 300
    seed: int = 0
    loss_fn: str = "cispo"
    normalization: str = "token"
    truncation: str = "drop_group"
    microbatch_size: int = 8
    eps_max: float = 6.0
    clip_low: float = 0.2
    clip_high: float = 0.2
    grad_clip_norm: float = 1.0
    max_generated_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in ("rounds", "best_of", "max_new_tokens", "microbatch_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"RL {name} must be a positive integer")
        if self.best_of < 2:
            raise ValueError("RL best_of must be at least 2 for group advantages")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("RL seed must be a nonnegative integer")
        for name in ("temperature", "top_p", "eps_max", "grad_clip_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"RL {name} must be finite and positive")
        if self.top_p > 1:
            raise ValueError("RL top_p must be <= 1")
        for name in ("clip_low", "clip_high"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"RL {name} must be between 0 and 1")
        if self.loss_fn not in {"cispo", "ppo", "importance_sampling"}:
            raise ValueError("RL loss_fn must be cispo, ppo, or importance_sampling")
        if self.normalization not in {"token", "sequence", "sum"}:
            raise ValueError("RL normalization must be token, sequence, or sum")
        if self.truncation not in {"drop_group", "error"}:
            raise ValueError("RL truncation must be drop_group or error")
        if self.max_generated_tokens is not None and (
            type(self.max_generated_tokens) is not int or self.max_generated_tokens < 1
        ):
            raise ValueError("RL max_generated_tokens must be a positive integer")

    @classmethod
    def from_knobs(cls, knobs: Mapping[str, Any] | None) -> RiverRLConfig:
        """Parse legacy harvest knobs without silently ignoring misspellings."""
        values = dict(knobs or {})
        values.pop("adapter_dir", None)
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown River RL options: {', '.join(sorted(unknown))}")
        return cls(**values)

    def loss_kwargs(self) -> dict[str, float]:
        """Return only arguments supported by the selected loss."""
        if self.loss_fn == "cispo":
            return {"eps_max": self.eps_max}
        if self.loss_fn == "ppo":
            return {"clip_low": self.clip_low, "clip_high": self.clip_high}
        return {}


@dataclass(frozen=True)
class RLScore:
    """A scalar learning signal plus an independently explicit pass decision."""

    reward: float
    passed: bool
    components: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.reward):
            raise ValueError("RL reward must be finite")
        if type(self.passed) is not bool:
            raise ValueError("RL pass decision must be a boolean")


RLScorer = Callable[[dict[str, Any], str | list[str]], RLScore]


def reward_function_scorer(reward_fn: Any, *, threshold: float) -> RLScorer:
    """Adapt a StateSet RewardFunction to River's synchronous scoring boundary.

    Pass the task as context and real ConversationTurn objects to the reward.
    Call synchronous RiverExecutor jobs outside an already-running event loop.
    Exceptions propagate; unavailable rewards never become successful tasks.
    """
    from stateset_agents.core.trajectory import ConversationTurn

    if not math.isfinite(threshold):
        raise ValueError("Reward threshold must be finite")

    def score(task: dict[str, Any], replies: str | list[str]) -> RLScore:
        prompts = task.get("turns", [task.get("prompt", "")])
        answers = [replies] if isinstance(replies, str) else replies
        turns = []
        for prompt, answer in zip(prompts, answers, strict=True):
            turns.extend(
                [
                    ConversationTurn(role="user", content=prompt),
                    ConversationTurn(role="assistant", content=answer),
                ]
            )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "Run RiverExecutor in a worker thread outside the event loop"
            )
        result = asyncio.run(reward_fn.compute_reward(turns, context=task))
        return RLScore(
            float(result.score), result.score >= threshold, dict(result.breakdown)
        )

    return score


def validate_task(task: dict[str, Any], *, custom_scorer: bool = False) -> None:
    """Reject tasks whose configured verifier would otherwise be ignored."""
    for name in ("expect", "forbid", "known_tools"):
        values = task.get(name, [])
        if not isinstance(values, list) or not all(
            isinstance(v, str) and v.strip() for v in values
        ):
            raise ValueError(f"RL {name} must contain nonempty strings")
    if task.get("nsr") is not None and not custom_scorer:
        raise ValueError("NSR is not supported by River RL; supply an explicit scorer")
    if "turns" in task:
        turns = task["turns"]
        if (
            not isinstance(turns, list)
            or not turns
            or not all(isinstance(t, str) and t.strip() for t in turns)
        ):
            raise ValueError("Episode turns must be nonempty strings")
        if custom_scorer:
            return
        expects = task.get("turn_expect")
        tools = task.get("turn_tool", [None] * len(turns))
        if not isinstance(expects, list) or len(expects) != len(turns):
            raise ValueError("Episode turn_expect must align with turns")
        if not isinstance(tools, list) or len(tools) != len(turns):
            raise ValueError("Episode turn_tool must align with turns")
        for tool in tools:
            if tool is not None and (
                not isinstance(tool, dict)
                or not isinstance(tool.get("tool"), str)
                or not tool["tool"].strip()
                or not isinstance(tool.get("args", {}), dict)
            ):
                raise ValueError(
                    "Episode tool assertions require a tool name and args object"
                )
        if not all(
            isinstance(e, list) and all(isinstance(x, str) and x for x in e)
            for e in expects
        ):
            raise ValueError("Episode expectations must be lists of nonempty strings")
        if not any(expects) and not any(tools) and not task.get("forbid"):
            raise ValueError("RL episode needs an objective verifier")
        if task.get("judge") or "min_judge_score" in task:
            raise ValueError("Episode judges require an explicit episode scorer")
    elif not custom_scorer:
        if not task.get("expect") and not task.get("forbid") and not task.get("judge"):
            raise ValueError("RL task needs expect, forbid, judge, or a custom scorer")
        if task.get("judge"):
            threshold = task.get("min_judge_score")
            if not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
                raise ValueError("RL judge requires a finite min_judge_score")
        elif "min_judge_score" in task:
            raise ValueError("min_judge_score requires a judge")


def score_text(task: dict[str, Any], text: str) -> RLScore:
    """Score assertions and optional domain judges, failing closed on errors."""
    from stateset_agents.training.sft import evaluate_checks, judge_completion

    validate_task(task)
    checks = evaluate_checks(text, task.get("expect", []), task.get("forbid", []))
    passed = checks["passed"]
    fraction = (
        (len(checks["expect_hits"]) / len(task["expect"]))
        if task.get("expect")
        else 1.0
    )
    components: dict[str, Any] = {"checks": checks, "coverage": fraction}
    signal = fraction
    if task.get("judge"):
        judged = judge_completion(task["judge"], task["prompt"], text)
        if judged is None or not math.isfinite(judged):
            raise ValueError(f"Required RL judge {task['judge']!r} failed to score")
        components["judge_score"] = judged
        passed = passed and judged >= task["min_judge_score"]
        signal = judged
    return RLScore(
        signal + float(passed) - float(bool(checks["forbid_hits"])), passed, components
    )


def sample_record(sample: Any, prompt_ids: list[int]) -> dict[str, Any]:
    """Preserve and validate the sampler's exact behavior-policy record."""
    record = {
        "text": str(getattr(sample, "text", "")),
        "prompt_ids": list(prompt_ids),
        "tokens": list(sample.tokens),
        "logprobs": list(sample.logprobs),
        "token_data_is_exact": getattr(sample, "token_data_is_exact", None),
        "stop_reason": getattr(sample, "stop_reason", None),
    }
    policy = getattr(sample, "policy_version", None)
    record["policy_id"] = getattr(policy, "id", None)
    validate_record(record)
    return record


def validate_record(record: dict[str, Any]) -> None:
    """Fail before an optimizer update when rollout data is corrupt."""
    if record.get("token_data_is_exact") is not True:
        raise ValueError(
            "River RL requires token_data_is_exact=True; use river-client>=0.11"
        )
    if not record.get("prompt_ids") or not record.get("tokens"):
        raise ValueError("River RL requires nonempty prompt and response tokens")
    if any(
        type(t) is not int or t < 0 for t in record["prompt_ids"] + record["tokens"]
    ):
        raise ValueError("River token IDs must be nonnegative integers")
    if len(record["tokens"]) != len(record["logprobs"]):
        raise ValueError("River token/logprob lengths differ")
    if any(not math.isfinite(p) or p > 1e-6 for p in record["logprobs"]):
        raise ValueError("River logprobs must be finite and nonpositive")


def retain_group(records: list[dict[str, Any]], config: RiverRLConfig) -> bool:
    """Drop whole truncated groups so the retained baseline remains defined."""
    for record in records:
        validate_record(record)
    truncated = any(
        r.get("stop_reason") in {"length", "max_tokens", "max_length"} for r in records
    )
    if truncated and config.truncation == "error":
        raise ValueError("Truncated RL rollout; increase max_new_tokens")
    return not truncated


def normalize_datums(
    data: list[dict[str, Any]], mode: str
) -> tuple[list[dict[str, Any]], int]:
    """Normalize a logical update once, before transport microbatching.

    Sequence mode assigns equal weight to each retained assistant span (each
    turn in an episode). Zero-advantage spans are excluded in all modes.
    """
    counts = [sum(a != 0 for a in d["advantages"]) for d in data]
    total = sum(counts)
    retained = sum(n > 0 for n in counts)
    result = []
    for datum, count in zip(data, counts, strict=True):
        if not count:
            continue
        denominator = (
            total if mode == "token" else retained * count if mode == "sequence" else 1
        )
        result.append(
            {**datum, "advantages": [a / denominator for a in datum["advantages"]]}
        )
    return result, total


def atomic_json(path: Path, value: Any) -> None:
    """Replace a durable JSON artifact without exposing a partial write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextlib.contextmanager
def exclusive_run(output: Path) -> Iterator[None]:
    """Hold an OS-released lock so concurrent drivers cannot overwrite a run."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".rl.lock").open("a+b") as stream:
        if sys.platform == "win32":
            import msvcrt

            stream.write(b"0")
            stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise ValueError(
                    "Another driver owns this RL output directory"
                ) from exc
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise ValueError(
                    "Another driver owns this RL output directory"
                ) from exc
        try:
            yield
        finally:
            if sys.platform == "win32":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run_fingerprint(
    spec: Any,
    config: RiverRLConfig,
    scorer_id: str,
    *,
    initial_checkpoint: str | None,
) -> str:
    """Bind resumable optimizer state to its data, scoring and configuration."""
    payload = {
        "dataset": hashlib.sha256(Path(spec.dataset).read_bytes()).hexdigest(),
        "eval": spec.eval_prompts,
        "model": spec.base_model,
        "rank": spec.lora_r,
        "lr": spec.learning_rate,
        "eval_max_new_tokens": spec.eval_max_new_tokens,
        "initial_adapter": initial_checkpoint,
        "config": asdict(config),
        "scorer": scorer_id,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
