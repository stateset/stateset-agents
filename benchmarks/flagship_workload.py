#!/usr/bin/env python3
"""On-pod workload for the StateSet flagship benchmark (one seed, one GPU).

Invoked by ``flagship_runpod_driver.py`` on the provisioned pod with the same
pinned inputs that ``run_flagship_matrix.py`` hands the driver. The run is:

1. fail-closed preflight: installed ``stateset_agents`` version, config and
   judge JSON digests, judge rubric revision, dataset size;
2. baseline evaluation: every eval scenario is rolled out through the
   framework's ``ConversationEnvironment`` (system prompt with the store
   policy, then the scenario's scripted customer turns, one policy reply per
   customer turn) with the base policy, greedy;
3. on-policy multi-turn GSPO + LoRA with ``MultiTurnGRPOTrainer``
   (``objective="gspo"``): every step samples ``num_generations`` full
   conversations per scheduled train scenario from the *current* policy in
   that same environment, scores each episode with :func:`support_reward`
   (``SupportConversationReward``), and optimizes the length-normalized
   sequence ratio of every assistant turn against the episode's
   group-normalized advantage;
4. final evaluation with the trained adapter: the identical environment
   rollout, prompt rendering and decoding as the baseline;
5. cross-family judge scoring of the baseline and final conversations.

Every GPU phase runs in a fresh child process (``--phase``) so that each
engine's memory is returned to the device before the next phase starts.
Orchestration lives in small pure functions; the heavy imports (torch, vLLM,
transformers, peft) only happen inside the phase implementations.

``--smoke`` wires the same path on a tiny model with tiny ``n`` for local CPU
checks and is always labelled ``status="smoke"``, ``measured=false``.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import functools
import hashlib
import inspect
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("flagship_workload")

RESULT_KIND = "stateset-flagship-workload-result"
TRAIN_SPLIT = "train"
EVAL_SPLIT = "eval"
SMOKE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
SMOKE_OVERRIDES: dict[str, Any] = {
    "num_train_examples": 4,
    "num_eval_examples": 2,
    "max_turns": 2,
    "max_steps": 1,
    "num_generations": 2,
    "scenarios_per_step": 2,
    "max_completion_length": 24,
    "max_prompt_length": 512,
}
REQUIRED_CONFIG_KEYS: tuple[str, ...] = (
    "num_train_examples",
    "num_eval_examples",
    "temperature",
    "max_turns",
    "use_lora",
    "bf16",
    "learning_rate",
    "num_generations",
    "scenarios_per_step",
    "max_steps",
    "lora_r",
    "lora_alpha",
    "max_completion_length",
    "rollout_temperature",
    "beta",
)
POSITIVE_INT_KEYS: tuple[str, ...] = (
    "num_train_examples",
    "num_eval_examples",
    "max_turns",
    "num_generations",
    "scenarios_per_step",
    "max_steps",
    "lora_r",
    "lora_alpha",
    "max_completion_length",
)
#: Every linear projection of a Qwen2/Llama-style decoder block. LoRA on the
#: MLP as well as attention is the stronger choice for RL fine-tuning.
DEFAULT_LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
REQUIRED_JUDGE_KEYS: tuple[str, ...] = ("model", "revision", "rubric_revision")
#: Runtime profiles a config may pin (``config.runtime_profile``). A missing
#: key is the v1 profile, so v1 manifests and their config hashes are unchanged.
RUNTIME_PROFILES: tuple[str, ...] = ("flagship-v1", "flagship-v2")
#: Judge engine loaders (``judge.loader``). ``mistral`` is the v1 path: plain
#: vLLM ``LLM.chat`` with the HF-format Mistral snapshot (mistral_common's
#: tekken.json kept, consolidated weights skipped by ``_default_snapshot``).
#: ``gemma4`` loads ``Gemma4ForConditionalGeneration`` text-only (no image
#: slots, so vLLM does not reserve vision-encoder memory), with thinking off
#: and a higher memory fraction for the ~62 GB of bf16 31B weights.
JUDGE_LOADERS: dict[str, dict[str, Any]] = {
    "mistral": {},
    "gemma4": {
        "gpu_memory_utilization": 0.92,
        "llm_kwargs": {"limit_mm_per_prompt": {"image": 0}},
        "chat_template_kwargs": {"enable_thinking": False},
    },
}
DEFAULT_JUDGE_LOADER = "mistral"
#: ``TrainingConfig`` fields a config may opt into (v2). ``episode_weighting``
#: ``"per_episode"`` weights each assistant-turn row by 1/(turns in its
#: episode) so every episode counts once (GSPO eq. 5); ``disable_dropout``
#: makes the loss pass match the eval-mode policy that sampled the rollouts.
V2_TRAINING_KEYS: tuple[str, ...] = ("episode_weighting", "disable_dropout")
EPISODE_WEIGHTINGS: tuple[str, ...] = ("per_turn", "per_episode")
#: Reasoning markers a non-thinking policy must never emit (Qwen3/3.5 style
#: ``<think>`` blocks and Gemma 4's ``<|think|>`` channel marker).
THINK_MARKERS: tuple[str, ...] = ("<think>", "</think>", "<|think|>")
SYSTEM_PROMPT = (
    "You are a customer-support agent for an online store. Follow the store "
    "policy exactly: never promise anything the policy does not allow, never "
    "ask for full card numbers, CVVs, passwords or one-time codes, and move "
    "the case to a concrete next step. Be concise, warm and specific."
)
#: Each GPU phase runs vLLM's engine in-process so torch's allocator (and
#: therefore ``torch.cuda.max_memory_allocated``) observes its memory.
PHASE_ENV = {"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "TOKENIZERS_PARALLELISM": "false"}

Transcript = list[tuple[str, str]]
Messages = list[dict[str, str]]
#: Batch of conversations (role/content messages ending in a user turn) to the
#: policy's next replies; may be sync or async.
GenerateFn = Callable[[Sequence[Messages]], "Sequence[str] | Awaitable[Sequence[str]]"]
PhaseRunner = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]


class WorkloadError(RuntimeError):
    """Raised when the workload cannot produce honest flagship evidence."""


# --------------------------------------------------------------------------- #
# Pinned-input checks (pure)
# --------------------------------------------------------------------------- #


def canonical_json(value: Mapping[str, Any]) -> str:
    """Serialize exactly as ``run_flagship_matrix.canonical_json`` does."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest_json(value: Mapping[str, Any]) -> str:
    """SHA-256 of canonical JSON."""
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def verify_framework_version(expected: str, installed: str) -> None:
    """Refuse to run a different framework build than the manifest pins."""
    if expected != installed:
        raise WorkloadError(
            f"installed stateset_agents {installed} does not match the pinned "
            f"framework_version {expected}"
        )


def parse_pinned_json(text: str, sha256: str, label: str) -> dict[str, Any]:
    """Parse ``text`` and require its canonical digest to equal ``sha256``."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise WorkloadError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise WorkloadError(f"{label} must be a JSON object")
    observed = digest_json(value)
    if observed != sha256:
        raise WorkloadError(
            f"{label} sha256 mismatch: expected {sha256}, observed {observed}"
        )
    return value


def _is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def validate_config(config: Mapping[str, Any]) -> None:
    """Require every knob the workload reads to be declared and well typed."""
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise WorkloadError("config is missing: " + ", ".join(missing))
    for key in POSITIVE_INT_KEYS:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise WorkloadError(f"config.{key} must be a positive integer")
    for key in ("temperature", "learning_rate", "rollout_temperature", "beta"):
        value = config[key]
        if not _is_number(value) or value < 0:
            raise WorkloadError(f"config.{key} must be a non-negative number")
    if config["learning_rate"] <= 0:
        raise WorkloadError("config.learning_rate must be positive")
    if config["rollout_temperature"] <= 0:
        raise WorkloadError("config.rollout_temperature must be positive (sampling)")
    if config["use_lora"] is not True:
        raise WorkloadError("the flagship protocol requires use_lora=true")
    if config["num_generations"] < 2:
        raise WorkloadError("config.num_generations must be >= 2 for GSPO groups")
    targets = config.get("lora_target_modules")
    if targets is not None and (
        not isinstance(targets, list)
        or not targets
        or not all(isinstance(t, str) and t for t in targets)
    ):
        raise WorkloadError("config.lora_target_modules must be a non-empty list")
    profile = config.get("runtime_profile", "flagship-v1")
    if profile not in RUNTIME_PROFILES:
        raise WorkloadError(
            f"config.runtime_profile must be one of {', '.join(RUNTIME_PROFILES)}"
        )
    template_kwargs = config.get("chat_template_kwargs")
    if template_kwargs is not None and (
        not isinstance(template_kwargs, Mapping)
        or not all(isinstance(key, str) and key for key in template_kwargs)
    ):
        raise WorkloadError("config.chat_template_kwargs must be an object")
    weighting = config.get("episode_weighting", "per_turn")
    if weighting not in EPISODE_WEIGHTINGS:
        raise WorkloadError(
            f"config.episode_weighting must be one of {', '.join(EPISODE_WEIGHTINGS)}"
        )
    for key in (
        "trust_remote_code",
        "forbid_think_blocks",
        "merge_adapter_for_eval",
        "disable_dropout",
    ):
        if key in config and not isinstance(config[key], bool):
            raise WorkloadError(f"config.{key} must be a boolean")
    for key in ("think_violation_window", "max_think_violations"):
        value = config.get(key, THINK_GUARD_DEFAULTS[key])
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise WorkloadError(f"config.{key} must be a positive integer")
    rate = config.get("max_think_violation_rate", 0.05)
    if (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not 0 < rate <= 1
    ):
        raise WorkloadError("config.max_think_violation_rate must be in (0, 1]")


def judge_loader(judge: Mapping[str, Any]) -> str:
    """The pinned judge engine loader (``mistral`` when unpinned, as in v1)."""
    loader = judge.get("loader", DEFAULT_JUDGE_LOADER)
    if loader not in JUDGE_LOADERS:
        raise WorkloadError(
            f"judge.loader must be one of {', '.join(sorted(JUDGE_LOADERS))}"
        )
    return str(loader)


def find_think_blocks(texts: Iterable[str]) -> list[str]:
    """Every text that carries a reasoning marker (see :data:`THINK_MARKERS`)."""
    return [t for t in texts if any(marker in str(t) for marker in THINK_MARKERS)]


#: A leading reasoning block with only whitespace inside, e.g. the
#: ``<think>\n\n</think>\n\n`` that Qwen3/3.5's ``enable_thinking=False``
#: template writes into the prompt and that a policy can learn to repeat.
EMPTY_THINK_PREFIX = re.compile(r"\A\s*(?:<think>\s*</think>\s*)+")
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

#: Training-time think guard defaults (overridable per manifest config).
#: A completion with reasoning content or a marker anywhere but an empty
#: leading block is a *violation*: its episode reward is zeroed and counted.
#: The run aborts only when violations exceed the rolling-window rate (share
#: of assistant turns over the last ``think_violation_window`` steps) or the
#: absolute cap; preflight-4 died on 1 stray tag in ~24k turns.
THINK_GUARD_DEFAULTS: dict[str, Any] = {
    "think_violation_window": 10,
    "max_think_violation_rate": 0.05,
    "max_think_violations": 250,
}
#: Evaluations never abort on think markers; a violation rate above this is
#: flagged in the report (``think_flags``).
EVAL_THINK_FLAG_RATE = 0.01


def strip_empty_think_blocks(text: str) -> tuple[str, bool]:
    """Drop leading whitespace-only ``<think></think>`` blocks from a reply.

    Returns ``(text, stripped)``. Only empty blocks at the very start are
    removed; anything else is left for :func:`sanitize_think`.
    """
    match = EMPTY_THINK_PREFIX.match(text)
    if match is None:
        return text, False
    return text[match.end() :], True


def sanitize_think(text: str) -> tuple[str, str | None]:
    """Remove reasoning markers and content from a non-thinking reply.

    Returns ``(clean_text, kind)`` where ``kind`` is ``None`` (no marker),
    ``"empty"`` (only leading empty blocks, benign) or ``"violation"``
    (reasoning content, or a marker anywhere else). For a violation, complete
    ``<think>...</think>`` blocks are removed; a lone ``</think>`` keeps the
    text after it (the reasoning came first) unless that is empty; an
    unclosed ``<think>`` keeps the text before it unless that is empty; any
    remaining marker is deleted. Best effort: a violating episode's reward is
    zeroed anyway, the cleaning only keeps markers out of history,
    transcripts and the judge.
    """
    clean, empty = strip_empty_think_blocks(str(text))
    if not any(marker in clean for marker in THINK_MARKERS):
        return clean, ("empty" if empty else None)
    clean = _THINK_BLOCK.sub(" ", clean)
    if "</think>" in clean:
        before, _, after = clean.rpartition("</think>")
        clean = after if after.strip() else before
    if "<think>" in clean:
        before, _, after = clean.partition("<think>")
        clean = before if before.strip() else after
    for marker in THINK_MARKERS:
        clean = clean.replace(marker, "")
    return re.sub(r"[ \t]{2,}", " ", clean).strip(), "violation"


@dataclass
class ThinkStats:
    """Think-marker counts for the replies one agent finalized."""

    replies: int = 0
    empty_stripped: int = 0
    violations: int = 0
    pending: list[str | None] = field(default_factory=list, repr=False)

    def record(self, kind: str | None) -> None:
        self.replies += 1
        if kind == "empty":
            self.empty_stripped += 1
        elif kind == "violation":
            self.violations += 1
        self.pending.append(kind)

    def summary(self, flag_rate: float | None = None) -> dict[str, Any]:
        rate = self.violations / self.replies if self.replies else 0.0
        out: dict[str, Any] = {
            "replies": self.replies,
            "empty_stripped": self.empty_stripped,
            "violations": self.violations,
            "violation_rate": rate,
        }
        if flag_rate is not None:
            out["flagged"] = rate > flag_rate
        return out


def install_think_guard(agent: Any) -> ThinkStats:
    """Sanitize think markers in every reply ``agent`` finalizes.

    ``_finalize_response`` is the single point where the agent turns sampled
    text into turn content and history, for training rollouts
    (``generate_turn(s)``/``generate_turns_batch``), the vLLM eval renderer
    and HF ``generate_response`` alike, so the reward, the environment, the
    next-turn prompt, the transcripts and the judge all see the cleaned text.
    The rollout APIs are wrapped too, so each returned turn carries
    ``metadata["think"]`` (``"empty"``/``"violation"``) for the training
    loop. Recorded ``token_ids``/``sampler_log_probs`` stay exactly as
    sampled (as with ``AgentConfig.enable_reasoning``), so GSPO ratios are
    computed on the real sampled sequence. Idempotent per agent.
    """
    existing = getattr(agent, "_flagship_think_stats", None)
    if isinstance(existing, ThinkStats):
        return existing
    stats = ThinkStats()
    original = agent._finalize_response

    @functools.wraps(original)
    def finalize(response: str, *args: Any, **kwargs: Any) -> Any:
        text, kind = sanitize_think(str(response))
        stats.record(kind)
        return original(text, *args, **kwargs)

    agent._finalize_response = finalize

    def tag(turns: list[Any]) -> None:
        # The finalize calls of one rollout call run without an await between
        # them, so the newest len(turns) entries belong to these turns.
        kinds = stats.pending[-len(turns) :] if turns else []
        stats.pending.clear()
        if len(kinds) != len(turns):
            logger.warning("think guard: %d turns, %d replies", len(turns), len(kinds))
            return
        for turn, kind in zip(turns, kinds, strict=True):
            metadata = getattr(turn, "metadata", None)
            if kind and isinstance(metadata, dict):
                metadata["think"] = kind

    for name in ("generate_turn", "generate_turns", "generate_turns_batch"):
        method = getattr(agent, name, None)
        if not inspect.iscoroutinefunction(method):
            continue

        def wrap(method: Any, single: bool) -> Any:
            @functools.wraps(method)
            async def call(*args: Any, **kwargs: Any) -> Any:
                result = await method(*args, **kwargs)
                tag([result] if single else list(result))
                return result

            return call

        setattr(agent, name, wrap(method, name == "generate_turn"))
    agent._flagship_think_stats = stats
    return stats


def think_guard_limits(config: Mapping[str, Any]) -> dict[str, Any]:
    """The configured training think-guard limits (see THINK_GUARD_DEFAULTS)."""
    return {key: config.get(key, value) for key, value in THINK_GUARD_DEFAULTS.items()}


def apply_think_guard(groups: Sequence[Any]) -> dict[str, int]:
    """Zero the reward of every episode with a think violation; count them.

    A turn violates when the think guard tagged it, or when its content still
    carries a marker (then the content is sanitized here). Runs before the
    update, so the zeroed rewards feed the group advantages, like a
    ``must_avoid`` hit.
    """
    counts = {"turns": 0, "violations": 0, "empty_stripped": 0, "episodes": 0}
    for group in groups:
        for trajectory in group.trajectories:
            violated = False
            for turn in trajectory.turns:
                if _turn_field(turn, "role") != "assistant":
                    continue
                counts["turns"] += 1
                metadata = _turn_field(turn, "metadata") or {}
                kind = metadata.get("think") if isinstance(metadata, Mapping) else None
                content = str(_turn_field(turn, "content") or "")
                if any(marker in content for marker in THINK_MARKERS):
                    clean, found = sanitize_think(content)
                    kind = "violation" if found == "violation" else kind or found
                    if isinstance(turn, dict):
                        turn["content"] = clean
                    else:
                        turn.content = clean
                if kind == "violation":
                    counts["violations"] += 1
                    violated = True
                elif kind == "empty":
                    counts["empty_stripped"] += 1
            if violated:
                counts["episodes"] += 1
                trajectory.total_reward = 0.0
                metadata = getattr(trajectory, "metadata", None)
                if isinstance(metadata, dict):
                    breakdown = metadata.setdefault("reward_breakdown", {})
                    if isinstance(breakdown, dict):
                        breakdown["think_violation"] = 1.0
    return counts


def assert_no_think_blocks(texts: Iterable[str], label: str) -> None:
    """Fail closed when a non-thinking policy emitted a reasoning block."""
    found = find_think_blocks(texts)
    if found:
        raise WorkloadError(
            f"{label}: {len(found)} completion(s) contain a <think> block "
            f"although thinking is disabled; first: {found[0][:120]!r}"
        )


def bind_chat_template_kwargs(tokenizer: Any, kwargs: Mapping[str, Any]) -> Any:
    """Make every ``apply_chat_template`` call on ``tokenizer`` pass ``kwargs``.

    The agent, the trainer's token path and the vLLM renderer all render
    through ``tokenizer.apply_chat_template``; binding the pinned template
    kwargs (e.g. ``enable_thinking=False``) on the instance keeps rollouts,
    training rows and evaluation prompts identical. Explicit call-site
    values win. No kwargs leaves the tokenizer untouched (the v1 path).
    """
    if not kwargs or tokenizer is None:
        return tokenizer
    original = getattr(tokenizer, "apply_chat_template", None)
    if not callable(original) or getattr(original, "_flagship_bound", None):
        return tokenizer
    bound = dict(kwargs)

    @functools.wraps(original)
    def apply_chat_template(*args: Any, **call_kwargs: Any) -> Any:
        for key, value in bound.items():
            call_kwargs.setdefault(key, value)
        return original(*args, **call_kwargs)

    apply_chat_template._flagship_bound = bound  # type: ignore[attr-defined]
    tokenizer.apply_chat_template = apply_chat_template
    return tokenizer


def validate_judge(judge: Mapping[str, Any], installed_rubric_revision: str) -> None:
    """Require the judge contract and the installed rubric to agree exactly."""
    missing = [key for key in REQUIRED_JUDGE_KEYS if not judge.get(key)]
    if missing:
        raise WorkloadError("judge is missing: " + ", ".join(missing))
    judge_loader(judge)
    if judge["rubric_revision"] != installed_rubric_revision:
        raise WorkloadError(
            f"installed judge rubric revision {installed_rubric_revision} does not "
            f"match the pinned rubric_revision {judge['rubric_revision']}"
        )


def effective_config(config: Mapping[str, Any], smoke: bool) -> dict[str, Any]:
    """Config actually run: the pinned one, or tiny overrides for ``--smoke``."""
    result = dict(config)
    if smoke:
        result.update(SMOKE_OVERRIDES)
    return result


def select_examples(items: Sequence[Any], count: int, label: str) -> list[Any]:
    """The first ``count`` items in dataset order; fewer is a hard failure."""
    if len(items) < count:
        raise WorkloadError(
            f"{label} split has {len(items)} scenarios; {count} are required"
        )
    selected = list(items[:count])
    ids = [str(getattr(item, "id", "")) for item in selected]
    if any(not item_id for item_id in ids) or len(set(ids)) != len(ids):
        raise WorkloadError(f"{label} scenarios must have unique non-empty ids")
    return selected


# --------------------------------------------------------------------------- #
# Scenarios, environment, rollouts, reward (pure / framework-light)
# --------------------------------------------------------------------------- #


def scenario_record(scenario: Any) -> dict[str, Any]:
    """Flatten a ``SupportDialogueScenario`` into the JSON a phase consumes.

    ``scenario`` is the ``ConversationEnvironment`` scenario: the dataset's
    ``to_scenario()`` (reward context: ``must_acknowledge``, ``must_avoid``,
    ``intent``, ...) plus the scripted ``user_turns`` and the policy facts the
    system prompt shows. Grading keys never reach the policy's prompt.
    """
    user_turns = [str(turn) for turn in getattr(scenario, "user_turns", ())]
    if not user_turns:
        raise WorkloadError(f"scenario {getattr(scenario, 'id', '?')} has no turns")
    environment_scenario = {
        **dict(scenario.to_scenario()),
        "id": str(scenario.id),
        "user_turns": user_turns,
        "product": getattr(scenario, "product", None),
        "policy_facts": list(getattr(scenario, "policy_facts", ()) or ()),
    }
    return {
        "id": str(scenario.id),
        "user_turns": user_turns,
        "scenario": environment_scenario,
        "judge_scenario": {
            "policy_facts": getattr(scenario, "policy_facts", None),
            "expected_resolution": getattr(scenario, "expected_resolution", None),
            "must_avoid": list(getattr(scenario, "must_avoid", ()) or ()),
            "intent": getattr(scenario, "intent", None),
            "product": getattr(scenario, "product", None),
        },
    }


def _render_value(value: Any) -> str:
    if value is None:
        return "(none)"
    if isinstance(value, str):
        return value.strip() or "(none)"
    if isinstance(value, Mapping):
        return "\n".join(
            f"- {key}: {_render_value(item)}" for key, item in value.items()
        )
    if isinstance(value, Sequence):
        return "\n".join(f"- {_render_value(item)}" for item in value) or "(none)"
    return str(value)


def render_system_prompt(scenario: Mapping[str, Any]) -> str:
    """The episode's system prompt: role, product and the authoritative policy.

    Only ``product`` and ``policy_facts`` are shown; the grading keys
    (``must_acknowledge``, ``must_avoid``, ``expected_resolution``) are not.
    """
    return "\n".join(
        [
            SYSTEM_PROMPT,
            "",
            f"Product: {_render_value(scenario.get('product'))}",
            "Store policy (authoritative):",
            _render_value(scenario.get("policy_facts")),
        ]
    )


@functools.lru_cache(maxsize=1)
def support_environment_cls() -> type:
    """``ConversationEnvironment`` for scripted support dialogues.

    Training, baseline and final evaluation all use this class: the system
    prompt carries the scenario's policy, the scripted customer turns drive the
    conversation (framework behaviour), and the episode ends only when the
    script (or ``max_turns``) is exhausted -- never because a reply happened
    to contain a farewell word, which the base class treats as an exit.
    """
    from stateset_agents.core.conversation_environment import (
        ConversationEnvironment,
    )

    class SupportConversationEnvironment(ConversationEnvironment):
        async def get_initial_prompt(
            self, scenario: dict[str, Any] | None = None
        ) -> str:
            return render_system_prompt(scenario or self.current_scenario or {})

        async def _should_end_conversation(self, agent_turn: Any, state: Any) -> bool:
            return False

        def clone(self) -> SupportConversationEnvironment:
            return SupportConversationEnvironment(
                scenarios=list(self.scenarios),
                max_turns=self.max_turns,
                reward_fn=self.reward_fn,
                persona=self.persona,
                timeout_seconds=self.timeout_seconds,
            )

    return SupportConversationEnvironment


def make_support_environment(
    scenarios: Sequence[Mapping[str, Any]], max_turns: int
) -> Any:
    """The shared environment. No env-side reward: the trainer scores episodes."""
    return support_environment_cls()(
        scenarios=[dict(s) for s in scenarios], max_turns=int(max_turns)
    )


async def rollout_episodes(
    environment: Any,
    scenarios: Sequence[Mapping[str, Any]],
    generate: GenerateFn,
) -> dict[str, Transcript]:
    """Roll out every scenario through ``environment``, batching across them.

    Mirrors ``Environment.run_episode`` exactly (``reset`` ->
    ``get_initial_prompt`` system turn -> ``get_initial_turns`` -> per agent
    reply ``step`` and its ``env_response``, until done or ``max_turns``), but
    advances all episodes in lockstep so each turn is one generation batch.
    Returns ``{scenario_id: [(role, content), ...]}`` without the system turn
    and without a trailing customer message nobody answered.
    """
    from stateset_agents.core.environment_base import EpisodeStatus
    from stateset_agents.core.trajectory import ConversationTurn

    episodes: list[dict[str, Any]] = []
    for scenario in scenarios:
        state = await environment.reset(dict(scenario))
        messages: Messages = []
        system = await environment.get_initial_prompt(dict(scenario))
        if system:
            messages.append({"role": "system", "content": str(system)})
        for turn in await environment.get_initial_turns(state):
            messages.append({"role": str(turn.role), "content": str(turn.content)})
        episodes.append(
            {"id": str(scenario["id"]), "state": state, "messages": messages}
        )
    active = list(episodes)
    for _ in range(int(environment.max_turns)):
        if not active:
            break
        replies = generate([list(e["messages"]) for e in active])
        if inspect.isawaitable(replies):
            replies = await replies
        replies = list(replies)
        if len(replies) != len(active):
            raise WorkloadError(
                f"policy returned {len(replies)} replies for {len(active)} prompts"
            )
        still: list[dict[str, Any]] = []
        for episode, reply in zip(active, replies, strict=True):
            text = str(reply)
            episode["messages"].append({"role": "assistant", "content": text})
            new_state, _reward, done, info = await environment.step(
                episode["state"], ConversationTurn(role="assistant", content=text)
            )
            response = info.get("env_response") if isinstance(info, dict) else None
            if isinstance(response, ConversationTurn):
                episode["messages"].append(
                    {"role": str(response.role), "content": str(response.content)}
                )
            episode["state"] = new_state
            if not done and new_state.status == EpisodeStatus.ONGOING:
                still.append(episode)
        active = still
    transcripts: dict[str, Transcript] = {}
    for episode in episodes:
        pairs = [
            (m["role"], m["content"])
            for m in episode["messages"]
            if m["role"] != "system"
        ]
        # A script longer than max_turns leaves the environment's next customer
        # message unanswered; the judged transcript ends at the last reply.
        while pairs and pairs[-1][0] == "user" and len(pairs) > 1:
            pairs.pop()
        transcripts[episode["id"]] = pairs
    return transcripts


def judge_items(
    records: Sequence[Mapping[str, Any]],
    transcripts: Mapping[str, Sequence[Sequence[str]]],
) -> list[dict[str, Any]]:
    """JSON form of ``support_judge.JudgeItem`` for every eval transcript."""
    items: list[dict[str, Any]] = []
    for record in records:
        transcript = [list(pair) for pair in transcripts.get(str(record["id"]), [])]
        if not any(role == "assistant" and text for role, text in transcript):
            raise WorkloadError(f"eval scenario {record['id']} has no assistant reply")
        items.append(
            {
                "id": str(record["id"]),
                "scenario": dict(record["judge_scenario"]),
                "transcript": transcript,
            }
        )
    return items


def _turn_field(turn: Any, name: str) -> Any:
    return turn.get(name) if isinstance(turn, Mapping) else getattr(turn, name, None)


def support_reward() -> Any:
    """The training reward: conversation-level ``SupportConversationReward``.

    v1 averaged ``SupportRewardComposite`` over assistant turns and was
    reward-hacked on the H100 preflight (held-out rule reward 0.707 -> 0.899
    while the judge fell 0.748 -> 0.605): the policy repeated one
    policy-correct paragraph on every turn. v2 multiplies each turn's
    composite score by repetition, placeholder and responsiveness penalties
    and zeroes an episode on a ``must_avoid`` hit; see
    :mod:`stateset_agents.rewards.support_conversation_reward`.
    """
    from stateset_agents.rewards.support_conversation_reward import (
        SupportConversationReward,
    )

    return SupportConversationReward()


#: Train-log key -> reward breakdown key, averaged over a step's episodes so a
#: drift toward a known hack (repeating, placeholders, ...) shows up in the log.
REWARD_COMPONENT_LOG_KEYS: dict[str, str] = {
    "base_reward_mean": "base_mean",
    "repetition_penalty_mean": "repetition_penalty_mean",
    "max_similarity_mean": "max_similarity_mean",
    "placeholder_rate": "placeholder_rate",
    "must_avoid_rate": "must_avoid_rate",
    "unresponsive_rate": "unresponsive_rate",
}


def reward_component_means(groups: Sequence[Any]) -> dict[str, float | None]:
    """Per-step mean of each :data:`REWARD_COMPONENT_LOG_KEYS` component.

    Reads ``trajectory.metadata["reward_breakdown"]`` (set by the trainer);
    a component no episode reported is ``None``.
    """
    sums = dict.fromkeys(REWARD_COMPONENT_LOG_KEYS, 0.0)
    counts = dict.fromkeys(REWARD_COMPONENT_LOG_KEYS, 0)
    for group in groups:
        for trajectory in getattr(group, "trajectories", None) or ():
            metadata = getattr(trajectory, "metadata", None) or {}
            breakdown = metadata.get("reward_breakdown") or {}
            for log_key, key in REWARD_COMPONENT_LOG_KEYS.items():
                value = breakdown.get(key)
                if _is_number(value):
                    sums[log_key] += float(value)
                    counts[log_key] += 1
    return {
        key: (sums[key] / counts[key] if counts[key] else None)
        for key in REWARD_COMPONENT_LOG_KEYS
    }


def rule_reward_mean(
    records: Sequence[Mapping[str, Any]],
    transcripts: Mapping[str, Sequence[Sequence[str]]],
    reward_fn: Any,
) -> float:
    """Mean training reward over the eval conversations (diagnostic only)."""
    from stateset_agents.core.trajectory import ConversationTurn

    async def _score() -> list[float]:
        scores = []
        for record in records:
            turns = [
                ConversationTurn(role=str(role), content=str(text))
                for role, text in transcripts.get(str(record["id"]), [])
            ]
            result = await reward_fn.compute_reward(
                turns=turns, context=dict(record["scenario"])
            )
            scores.append(float(result.score))
        return scores

    values = asyncio.run(_score())
    return sum(values) / len(values) if values else 0.0


# --------------------------------------------------------------------------- #
# Training plan (pure)
# --------------------------------------------------------------------------- #


def training_schedule(
    num_scenarios: int, scenarios_per_step: int, max_steps: int, seed: int
) -> list[list[int]]:
    """Scenario indices for every step: seeded epoch permutations, in order.

    Each epoch is one ``random.Random(seed)`` shuffle of every train scenario,
    so no scenario repeats before all have been used and the order depends
    only on the seed.
    """
    if scenarios_per_step > num_scenarios:
        raise WorkloadError(
            f"scenarios_per_step {scenarios_per_step} exceeds the "
            f"{num_scenarios} train scenarios"
        )
    rng = random.Random(int(seed))
    order: list[int] = []
    needed = scenarios_per_step * max_steps
    while len(order) < needed:
        epoch = list(range(num_scenarios))
        rng.shuffle(epoch)
        order.extend(epoch)
    return [
        order[step * scenarios_per_step : (step + 1) * scenarios_per_step]
        for step in range(max_steps)
    ]


def nonzero_advantage_fraction(
    group_rewards: Sequence[Sequence[float]], eps: float = 1e-6
) -> float:
    """Fraction of episodes whose group-normalized advantage is nonzero.

    An episode has signal only if its group's rewards vary and its own reward
    differs from the group mean -- exactly when GSPO's group-normalized
    advantage is nonzero.
    """
    total = nonzero = 0
    for rewards in group_rewards:
        values = [float(r) for r in rewards]
        total += len(values)
        if len(values) < 2:
            continue
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        if std <= eps:
            continue
        nonzero += sum(1 for v in values if abs(v - mean) > eps)
    return nonzero / total if total else 0.0


def lora_peft_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """``peft.LoraConfig`` kwargs for the pinned LoRA knobs."""
    targets = config.get("lora_target_modules") or list(DEFAULT_LORA_TARGET_MODULES)
    return {
        "r": int(config["lora_r"]),
        "lora_alpha": int(config["lora_alpha"]),
        "lora_dropout": float(config.get("lora_dropout", 0.05)),
        "target_modules": list(targets),
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def agent_config_kwargs(
    payload: Mapping[str, Any],
    *,
    temperature: float,
    max_new_tokens: int,
    top_p: float = 1.0,
    lora: Mapping[str, Any] | None = None,
    adapter_path: str | None = None,
) -> dict[str, Any]:
    """``AgentConfig`` kwargs shared by the training policy and HF evaluation.

    The model is the pinned local snapshot (its directory name is the pinned
    revision). ``system_prompt`` stays unset: the environment supplies the
    per-scenario system turn. Sampling is plain ancestral sampling at
    ``temperature`` (``top_k=0``, ``repetition_penalty=1``), greedy when
    ``temperature == 0``.
    """
    sampling = temperature > 0
    kwargs: dict[str, Any] = {
        "model_name": str(payload["model_path"]),
        "system_prompt": None,
        "use_chat_template": True,
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p) if sampling else 1.0,
        # Disabled in the config itself, so a rebuilt generation config keeps it.
        "top_k": 0,
        "do_sample": sampling,
        "repetition_penalty": 1.0,
        "torch_dtype": "bfloat16" if payload.get("bf16", True) else "float32",
        "attn_implementation": str(payload.get("attn_implementation", "sdpa")),
    }
    if payload.get("trust_remote_code"):
        kwargs["trust_remote_code"] = True
    if lora is not None:
        kwargs.update({"use_peft": True, "peft_config": lora_peft_config(lora)})
    if adapter_path:
        kwargs["peft_path"] = str(adapter_path)
    return kwargs


def training_config_kwargs(
    payload: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """``TrainingConfig`` kwargs: multi-turn GSPO from the pinned config."""
    beta = float(config["beta"])
    overrides = {
        name: float(config[key])
        for key, name in (
            ("clip_range_left", "clip_low"),
            ("clip_range_right", "clip_high"),
        )
        if key in config
    }
    max_steps = int(config["max_steps"])
    return {
        "model_name": str(payload["model_path"]),
        "output_dir": str(payload["output_dir"]),
        "report_to": "none",
        "objective": "gspo",
        "objective_overrides": overrides or None,
        "num_episodes": max_steps,
        "num_generations": int(config["num_generations"]),
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": 0.0,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": float(config.get("warmup_ratio", 0.05)),
        "max_grad_norm": float(config.get("max_grad_norm", 1.0)),
        # One optimizer step per scheduled batch of groups.
        "gradient_accumulation_steps": 1,
        "num_gradient_updates": 1,
        "beta": beta,
        "use_reference_model": beta > 0,
        "entropy_coef": 0.0,
        "max_prompt_length": int(config.get("max_prompt_length", 1536)),
        "max_completion_length": int(config["max_completion_length"]),
        # Rows per chunked forward on the token path.
        "generation_batch_size": int(config.get("train_forward_rows", 8)),
        "gradient_checkpointing": True,
        # A rare non-finite update is discarded and logged with the offending
        # parameters and loss terms (counted in the evidence); repeated ones abort.
        "nonfinite_grad_policy": "skip",
        "max_consecutive_nonfinite_skips": 3,
        "bf16": bool(config["bf16"]),
        "fp16": False,
        "seed": int(payload["seed"]),
        "save_steps": max_steps + 1,
        "eval_steps": max_steps + 1,
        "logging_steps": 1,
        # v2 opt-ins; absent from v1 configs, so v1 keeps the trainer defaults
        # (per-turn row averaging, dropout active in the loss pass).
        **{key: config[key] for key in V2_TRAINING_KEYS if key in config},
    }


def _group_rewards(group: Any) -> list[float]:
    return [float(getattr(t, "total_reward", 0.0)) for t in group.trajectories]


def check_groups(
    groups: Sequence[Any], scenarios: Sequence[Mapping[str, Any]], num_generations: int
) -> None:
    """Fail closed on a missing group or a group with a failed rollout.

    ``generate_trajectories`` logs and skips an episode that raised; training
    on the survivors would silently change the group size and advantages.
    """
    expected = [str(s["id"]) for s in scenarios]
    observed = [str(getattr(g, "scenario_id", "")) for g in groups]
    if observed != expected:
        raise WorkloadError(f"rollout groups {observed} do not match batch {expected}")
    for group in groups:
        size = len(getattr(group, "trajectories", []) or [])
        if size != num_generations:
            raise WorkloadError(
                f"group {group.scenario_id} has {size} of {num_generations} episodes"
            )


async def run_training_loop(
    trainer: Any,
    scenarios: Sequence[Mapping[str, Any]],
    schedule: Sequence[Sequence[int]],
    num_generations: int,
    guard: Any,
    clock: Callable[[], float] = time.monotonic,
    *,
    forbid_think_blocks: bool = False,
    think_limits: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """On-policy multi-turn GSPO: sample groups, then one update, per step.

    ``trainer`` is an initialized ``MultiTurnGRPOTrainer``. Every step must take
    the batched token path (per-assistant-turn sequence ratio); a fallback to
    the sequence-level path is a different objective and fails the run.
    """
    log: dict[str, list[Any]] = {
        key: []
        for key in (
            "step",
            "scenario_ids",
            "reward_mean",
            "reward_std",
            "nonzero_advantage_fraction",
            "assistant_turns",
            "total_loss",
            "kl_penalty",
            "ratio_mean",
            "clip_fraction",
            "learning_rate",
            "optimizer_step",
            "seconds",
            "empty_think_stripped",
            "think_violations",
            "think_violation_episodes",
            "think_violation_rate_window",
            *REWARD_COMPONENT_LOG_KEYS,
        )
    }
    aborted: str | None = None

    limits = {**THINK_GUARD_DEFAULTS, **dict(think_limits or {})}
    window: collections.deque[tuple[int, int]] = collections.deque(
        maxlen=int(limits["think_violation_window"])
    )
    think_totals = {"violations": 0, "episodes": 0, "empty_stripped": 0}
    for step, batch in enumerate(schedule, start=1):
        started = clock()
        chosen = [dict(scenarios[index]) for index in batch]
        groups = await trainer.generate_trajectories(
            chosen, num_generations=num_generations
        )
        check_groups(groups, chosen, num_generations)
        think = {"turns": 0, "violations": 0, "empty_stripped": 0, "episodes": 0}
        window_rate = 0.0
        if forbid_think_blocks:
            think = apply_think_guard(groups)
            for key in think_totals:
                think_totals[key] += think[key]
            window.append((think["violations"], think["turns"]))
            seen = sum(turns for _, turns in window)
            window_rate = sum(v for v, _ in window) / seen if seen else 0.0
            if window_rate > float(limits["max_think_violation_rate"]):
                aborted = (
                    f"think violations at step {step}: rate {window_rate:.3f} over "
                    f"the last {len(window)} steps exceeds "
                    f"{limits['max_think_violation_rate']}"
                )
            elif think_totals["violations"] > int(limits["max_think_violations"]):
                aborted = (
                    f"think violations at step {step}: {think_totals['violations']} "
                    f"total exceeds {limits['max_think_violations']}"
                )
            if aborted is not None:
                logger.error("%s", aborted)
                break
        rewards = [_group_rewards(group) for group in groups]
        flat = [value for values in rewards for value in values]
        metrics = dict(await trainer.training_step(groups))
        if metrics.get("path") != "token":
            raise WorkloadError(
                f"step {step} took the {metrics.get('path')!r} loss path; flagship "
                "GSPO requires per-turn token metadata (the batched token path)"
            )
        fraction = nonzero_advantage_fraction(rewards)
        mean = sum(flat) / len(flat)
        log["step"].append(step)
        log["scenario_ids"].append([str(s["id"]) for s in chosen])
        log["reward_mean"].append(mean)
        log["reward_std"].append(
            math.sqrt(sum((v - mean) ** 2 for v in flat) / len(flat))
        )
        log["nonzero_advantage_fraction"].append(fraction)
        log["assistant_turns"].append(
            sum(
                1
                for group in groups
                for trajectory in group.trajectories
                for turn in trajectory.turns
                if _turn_field(turn, "role") == "assistant"
            )
        )
        for key in (
            "total_loss",
            "kl_penalty",
            "ratio_mean",
            "clip_fraction",
            "learning_rate",
        ):
            value = metrics.get(key)
            log[key].append(float(value) if _is_number(value) else None)
        log["optimizer_step"].append(bool(metrics.get("optimizer_step")))
        components = reward_component_means(groups)
        for key, value in components.items():
            log[key].append(value)
        log["seconds"].append(round(clock() - started, 3))
        log["empty_think_stripped"].append(think["empty_stripped"])
        log["think_violations"].append(think["violations"])
        log["think_violation_episodes"].append(think["episodes"])
        log["think_violation_rate_window"].append(round(window_rate, 6))
        logger.info(
            "step %d/%d reward=%.4f nonzero_adv=%.3f loss=%s %.1fs "
            "repetition=%s placeholders=%s must_avoid=%s unresponsive=%s "
            "empty_think_stripped=%d think_violations=%d think_rate_window=%.4f",
            step,
            len(schedule),
            mean,
            fraction,
            log["total_loss"][-1],
            log["seconds"][-1],
            components["repetition_penalty_mean"],
            components["placeholder_rate"],
            components["must_avoid_rate"],
            components["unresponsive_rate"],
            log["empty_think_stripped"][-1],
            log["think_violations"][-1],
            window_rate,
        )
        guard.on_step_end(step, {"nonzero_advantage_fraction": fraction})
        if guard.should_abort:
            aborted = str(guard.abort_reason)
            break
    optimizer_steps = sum(1 for flag in log["optimizer_step"] if flag)
    if aborted is None and optimizer_steps == 0:
        raise WorkloadError("training applied no optimizer step")
    return {
        "train_log": log,
        "steps": len(log["step"]),
        "optimizer_steps": optimizer_steps,
        "nonfinite_skips": int(getattr(trainer, "nonfinite_skips", 0)),
        "empty_think_stripped": think_totals["empty_stripped"],
        "think_violations": think_totals["violations"],
        "think_violation_episodes": think_totals["episodes"],
        "think_limits": limits if forbid_think_blocks else None,
        "episodes": (
            len(log["step"]) * len(schedule[0]) * num_generations if schedule else 0
        ),
        "aborted": aborted,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _summary_number(summary: Mapping[str, Any], key: str, label: str) -> float:
    value = summary.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkloadError(f"{label}.{key} is missing")
    number = float(value)
    if not math.isfinite(number):
        raise WorkloadError(f"{label}.{key} is not finite")
    return number


def _default_installed_version() -> str:
    from stateset_agents import __version__

    return str(__version__)


def _default_rubric_revision() -> str:
    from stateset_agents.evaluation.support_judge import rubric_revision

    return str(rubric_revision())


def _default_dataset_loader(source: str, revision: str | None, split: str) -> list[Any]:
    from stateset_agents.data.support_multiturn import load_support_dataset

    return list(load_support_dataset(source, revision=revision, split=split))


def _default_dataset_name() -> str:
    from stateset_agents.data.support_multiturn import DATASET_NAME

    return str(DATASET_NAME)


def _default_snapshot(model: str, revision: str | None) -> str:
    """Download the pinned snapshot once; every phase loads that exact path."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=model,
        revision=revision,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"],
        # Mistral repos also ship a Mistral-format copy of the weights
        # (consolidated.safetensors: +47 GB for the 24B judge) described by
        # params.json. Only the HF-format config and shards are loaded, so
        # both are skipped; tekken.json stays so vLLM can use the official
        # mistral_common tokenizer.
        ignore_patterns=["consolidated*.safetensors", "original/*", "params.json"],
    )
    if revision is not None and Path(path).name != revision:
        raise WorkloadError(
            f"resolved snapshot {Path(path).name} is not the pinned revision {revision}"
        )
    return str(path)


def _default_reward_fn() -> Any:
    return support_reward()


def _default_hardware() -> dict[str, Any]:
    """GPU name/count/driver from nvidia-smi, CUDA runtime from torch."""
    hardware: dict[str, Any] = {"gpu": "", "gpu_count": 0, "cuda": "", "driver": ""}
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout
        rows = [
            [part.strip() for part in line.split(",")]
            for line in output.splitlines()
            if line.strip()
        ]
        if rows:
            hardware["gpu"] = rows[0][0]
            hardware["driver"] = rows[0][1] if len(rows[0]) > 1 else ""
            hardware["gpu_count"] = len(rows)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        import torch

        hardware["cuda"] = str(torch.version.cuda or "")
        if not hardware["gpu"] and torch.cuda.is_available():
            hardware["gpu"] = torch.cuda.get_device_name(0)
            hardware["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        pass
    return hardware


class WorkloadDeps:
    """Injectable side effects so orchestration is testable with fakes."""

    def __init__(
        self,
        *,
        run_phase: PhaseRunner,
        installed_version: Callable[[], str] = _default_installed_version,
        rubric_revision: Callable[[], str] = _default_rubric_revision,
        load_dataset: Callable[
            [str, str | None, str], list[Any]
        ] = _default_dataset_loader,
        dataset_name: Callable[[], str] = _default_dataset_name,
        snapshot: Callable[[str, str | None], str] = _default_snapshot,
        reward_fn: Callable[[], Any] = _default_reward_fn,
        hardware: Callable[[], dict[str, Any]] = _default_hardware,
    ) -> None:
        self.run_phase = run_phase
        self.installed_version = installed_version
        self.rubric_revision = rubric_revision
        self.load_dataset = load_dataset
        self.dataset_name = dataset_name
        self.snapshot = snapshot
        self.reward_fn = reward_fn
        self.hardware = hardware


def _phase_peak(output: Mapping[str, Any]) -> float:
    value = output.get("peak_vram_mb", 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def run_workload(args: argparse.Namespace, deps: WorkloadDeps) -> dict[str, Any]:
    """Execute the full flagship seed and return the workload result document.

    Pinned-input violations raise :class:`WorkloadError` before any model is
    loaded. The returned document is ``status="completed"``/``measured=True``
    only for a non-smoke run whose every phase finished.
    """
    started = time.monotonic()
    artifact_dir = Path(args.artifact_dir)
    smoke = bool(args.smoke)

    verify_framework_version(args.framework_version, deps.installed_version())
    config = parse_pinned_json(args.config_json, args.config_sha256, "config")
    judge = parse_pinned_json(args.judge_json, args.judge_sha256, "judge")
    validate_config(config)
    rubric = deps.rubric_revision()
    validate_judge(judge, rubric)
    run_config = effective_config(config, smoke)

    source = args.dataset_source or deps.dataset_name()
    train = select_examples(
        deps.load_dataset(source, args.dataset_revision, TRAIN_SPLIT),
        int(run_config["num_train_examples"]),
        TRAIN_SPLIT,
    )
    evaluation = select_examples(
        deps.load_dataset(source, args.dataset_revision, EVAL_SPLIT),
        int(run_config["num_eval_examples"]),
        EVAL_SPLIT,
    )
    overlap = {s.id for s in train} & {s.id for s in evaluation}
    if overlap:
        raise WorkloadError(f"train/eval scenario ids overlap: {sorted(overlap)[:5]}")
    train_records = [scenario_record(s) for s in train]
    eval_records = [scenario_record(s) for s in evaluation]
    schedule = training_schedule(
        len(train_records),
        int(run_config["scenarios_per_step"]),
        int(run_config["max_steps"]),
        int(args.seed),
    )

    policy_model = args.smoke_model if smoke else args.model
    policy_revision = None if smoke else args.model_revision
    judge_model = args.smoke_model if smoke else str(judge["model"])
    judge_revision = None if smoke else str(judge["revision"])
    # Smoke runs judge with the tiny smoke model, so they take the v1 loader.
    loader = DEFAULT_JUDGE_LOADER if smoke else judge_loader(judge)
    model_path = deps.snapshot(policy_model, policy_revision)
    # Resolve the judge before any GPU phase so a bad pin fails fast; phase 4
    # loads this exact local snapshot.
    judge_path = deps.snapshot(judge_model, judge_revision)
    backend = "hf" if smoke else "auto"
    max_turns = int(run_config["max_turns"])
    common = {
        "model_path": model_path,
        "max_turns": max_turns,
        "seed": int(args.seed),
        "lora_r": int(run_config["lora_r"]),
        "bf16": bool(run_config["bf16"]),
        # HF attention kernel for the policy ("sdpa" default; "eager" avoids
        # SDPA's bf16 backward when diagnosing non-finite gradients).
        "attn_implementation": str(run_config.get("attn_implementation", "sdpa")),
        # v2 knobs; absent from v1 configs, which keep the v1 behaviour.
        "chat_template_kwargs": dict(run_config.get("chat_template_kwargs") or {}),
        "trust_remote_code": bool(run_config.get("trust_remote_code", False)),
        "forbid_think_blocks": bool(run_config.get("forbid_think_blocks", False)),
        "vllm_language_model_only": (
            run_config.get("runtime_profile", "flagship-v1") == "flagship-v2"
        ),
    }
    evaluation_payload = {
        **common,
        "temperature": float(run_config["temperature"]),
        "max_tokens": int(run_config["max_completion_length"]),
        "backend": backend,
        "scenarios": [r["scenario"] for r in eval_records],
    }

    peaks: list[float] = []
    # Phase 1: baseline eval -- the multi-turn environment rollout, base policy.
    baseline_out = deps.run_phase(
        "rollout", {**evaluation_payload, "adapter_path": None}
    )
    peaks.append(_phase_peak(baseline_out))
    baseline = {r["id"]: baseline_out["transcripts"][r["id"]] for r in eval_records}
    _write_jsonl(
        artifact_dir / "transcripts" / "baseline.jsonl",
        [{"id": key, "transcript": value} for key, value in baseline.items()],
    )

    # Phase 2: on-policy multi-turn GSPO + LoRA in the same environment.
    adapter_dir = artifact_dir / "adapter"
    train_out = deps.run_phase(
        "train",
        {
            **common,
            "config": run_config,
            "scenarios": [r["scenario"] for r in train_records],
            "schedule": schedule,
            "output_dir": str(Path(args.scratch_dir) / "gspo"),
            "adapter_dir": str(adapter_dir),
        },
    )
    peaks.append(_phase_peak(train_out))
    _write_json(artifact_dir / "train_log.json", train_out.get("train_log", {}))
    if train_out.get("aborted"):
        raise WorkloadError(f"GSPO training aborted: {train_out['aborted']}")

    # Phase 3: final eval with the trained adapter (identical rollout/decoding).
    final_payload = {**evaluation_payload, "adapter_path": str(adapter_dir)}
    if run_config.get("merge_adapter_for_eval"):
        # Hybrid Qwen3.5 adapters are silently ignored by vLLM LoRA
        # (docs/PROOFS.md), so v2 evaluates merged weights instead.
        final_payload["merge_adapter"] = True
        final_payload["merged_dir"] = str(Path(args.scratch_dir) / "merged-policy")
    final_out = deps.run_phase("rollout", final_payload)
    peaks.append(_phase_peak(final_out))
    final = {r["id"]: final_out["transcripts"][r["id"]] for r in eval_records}
    _write_jsonl(
        artifact_dir / "transcripts" / "final.jsonl",
        [{"id": key, "transcript": value} for key, value in final.items()],
    )

    # Phase 4: judge both transcript sets in one engine session.
    judge_out = deps.run_phase(
        "judge",
        {
            "model": judge_model,
            "revision": judge_revision,
            "model_path": judge_path,
            "backend": backend,
            "loader": loader,
            "groups": {
                "baseline": judge_items(eval_records, baseline),
                "final": judge_items(eval_records, final),
            },
        },
    )
    peaks.append(_phase_peak(judge_out))
    if judge_out.get("rubric_revision") != judge["rubric_revision"]:
        raise WorkloadError(
            "judge phase ran rubric revision "
            f"{judge_out.get('rubric_revision')}, expected {judge['rubric_revision']}"
        )
    groups = judge_out["groups"]
    for name in ("baseline", "final"):
        _write_jsonl(artifact_dir / "judge" / f"{name}.jsonl", groups[name]["results"])
    baseline_summary = groups["baseline"]["summary"]
    final_summary = groups["final"]["summary"]
    for label, summary in (("baseline", baseline_summary), ("final", final_summary)):
        if int(_summary_number(summary, "n", label)) != len(eval_records):
            raise WorkloadError(f"judge scored the wrong number of {label} items")

    think_markers: dict[str, Any] = {
        "baseline_eval": baseline_out.get("think"),
        "train": {
            "empty_stripped": int(train_out.get("empty_think_stripped", 0)),
            "violations": int(train_out.get("think_violations", 0)),
            "violation_episodes": int(train_out.get("think_violation_episodes", 0)),
            "limits": train_out.get("think_limits"),
        },
        "final_eval": final_out.get("think"),
        "eval_flag_rate": EVAL_THINK_FLAG_RATE,
    }
    reward_fn = deps.reward_fn()
    metrics = {
        "baseline_score": _summary_number(baseline_summary, "mean_score", "baseline"),
        "final_score": _summary_number(final_summary, "mean_score", "final"),
        "judge_self_disagreement": max(
            _summary_number(baseline_summary, "self_disagreement", "baseline"),
            _summary_number(final_summary, "self_disagreement", "final"),
        ),
        "train_examples": len(train_records),
        "eval_examples": len(eval_records),
        "peak_vram_mb": max(peaks) if peaks else 0.0,
    }
    diagnostics = {
        "baseline_parse_failure_rate": _summary_number(
            baseline_summary, "parse_failure_rate", "baseline"
        ),
        "final_parse_failure_rate": _summary_number(
            final_summary, "parse_failure_rate", "final"
        ),
        "baseline_self_disagreement": float(baseline_summary["self_disagreement"]),
        "final_self_disagreement": float(final_summary["self_disagreement"]),
        "baseline_rule_reward_mean": rule_reward_mean(
            eval_records, baseline, reward_fn
        ),
        "final_rule_reward_mean": rule_reward_mean(eval_records, final, reward_fn),
        "rule_reward": str(getattr(reward_fn, "name", type(reward_fn).__name__)),
        "training": "on-policy multi-turn GSPO (MultiTurnGRPOTrainer, objective=gspo)",
        "train_steps": train_out.get("steps"),
        "train_optimizer_steps": train_out.get("optimizer_steps"),
        "train_nonfinite_skips": train_out.get("nonfinite_skips", 0),
        # v2 (forbid_think_blocks): think markers sanitized from replies.
        # Training zeroes violating episodes and aborts only past the limits;
        # evaluations never abort and are flagged above EVAL_THINK_FLAG_RATE.
        "think_markers": think_markers,
        "think_flags": [
            name
            for name, value in think_markers.items()
            if isinstance(value, Mapping) and value.get("flagged")
        ],
        "train_episodes": train_out.get("episodes"),
        "train_unique_scenarios": len({i for batch in schedule for i in batch}),
        "policy_backend": baseline_out.get("backend"),
        "judge_backend": judge_out.get("backend"),
        "judge_loader": loader,
        "final_eval_weights": final_out.get("weights", "adapter"),
        "baseline_judge_findings": groups["baseline"].get("findings_summary"),
        "final_judge_findings": groups["final"].get("findings_summary"),
        "phase_peak_vram_mb": peaks,
        "workload_seconds": round(time.monotonic() - started, 3),
    }
    return {
        "status": "smoke" if smoke else "completed",
        "measured": not smoke,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "rubric_revision": rubric,
        "effective_config": run_config,
        "policy_model": policy_model,
        "judge_model": judge_model,
    }


def build_result(
    args: argparse.Namespace,
    outcome: Mapping[str, Any] | None,
    hardware: Mapping[str, Any],
    error: str | None,
) -> dict[str, Any]:
    """The result document the driver downloads and validates."""
    completed = error is None and outcome is not None
    status = str(outcome["status"]) if completed and outcome is not None else "failed"
    measured = bool(completed and outcome and outcome.get("measured"))
    document: dict[str, Any] = {
        "schema_version": 1,
        "kind": RESULT_KIND,
        "status": status,
        "measured": measured and status == "completed",
        "mode": args.mode,
        "preflight": args.mode == "preflight",
        "smoke": bool(args.smoke),
        "seed": int(args.seed),
        "framework_version": args.framework_version,
        "model": args.model,
        "model_revision": args.model_revision,
        "dataset_revision": args.dataset_revision,
        "config_sha256": args.config_sha256,
        "judge_sha256": args.judge_sha256,
        "hardware": dict(hardware),
        "metrics": dict(outcome.get("metrics", {})) if outcome else {},
        "diagnostics": dict(outcome.get("diagnostics", {})) if outcome else {},
        "artifact_dir": str(args.artifact_dir),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    if outcome:
        for key in (
            "rubric_revision",
            "effective_config",
            "policy_model",
            "judge_model",
        ):
            if key in outcome:
                document[key] = outcome[key]
    if error is not None:
        document["error"] = error
    return document


# --------------------------------------------------------------------------- #
# Phase execution (child processes)
# --------------------------------------------------------------------------- #


def subprocess_phase_runner(work_dir: Path, log_dir: Path) -> PhaseRunner:
    """Run each phase as ``python flagship_workload.py --phase NAME`` so every
    GPU engine releases its memory when the child exits."""
    counter = {"n": 0}

    def run(name: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        counter["n"] += 1
        tag = f"{counter['n']:02d}-{name}"
        work_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        source = work_dir / f"phase-{tag}-input.json"
        target = work_dir / f"phase-{tag}-output.json"
        source.write_text(json.dumps(payload, default=str), encoding="utf-8")
        target.unlink(missing_ok=True)
        env = {**os.environ, **PHASE_ENV}
        with (log_dir / f"phase-{tag}.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "--phase",
                    name,
                    "--phase-input",
                    str(source),
                    "--phase-output",
                    str(target),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
            )
        if completed.returncode != 0:
            raise WorkloadError(
                f"phase {tag} exited {completed.returncode}; see logs/phase-{tag}.log"
            )
        try:
            output = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WorkloadError(f"phase {tag} wrote no valid output") from exc
        if not isinstance(output, dict):
            raise WorkloadError(f"phase {tag} output must be an object")
        return output

    return run


def _peak_vram_mb() -> float:
    """``torch.cuda.max_memory_allocated`` for this phase process.

    vLLM runs in-process (:data:`PHASE_ENV`), so its weights and KV cache go
    through torch's allocator and are counted. Only when torch saw nothing
    (an engine that allocated outside this process) does the device's
    ``nvidia-smi`` memory in use stand in.
    """
    peak = 0.0
    try:
        import torch

        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    if peak > 0:
        return round(peak, 1)
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout
        used = [float(line) for line in output.split() if line.strip()]
        if used:
            peak = max(used)
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return round(peak, 1)


@functools.lru_cache(maxsize=1)
def flagship_agent_cls() -> type:
    """``MultiTurnAgent`` whose conversation text is exactly what it sampled.

    The base agent now does this by default (no default stop strings, no
    canned persona text; see ``docs/OBJECTIVES.md``). This subclass pins the
    behaviour for the flagship workload regardless of config: it ends a reply
    only at the chat template's EOS, never appends text, and its
    ``_clean_response`` (used on the vLLM evaluation path) only strips
    whitespace, so the training rows, the conversation history and the
    judged transcripts agree.
    """
    from stateset_agents.core.agent import MultiTurnAgent, StopOnSpecialTokens

    class FlagshipSupportAgent(MultiTurnAgent):
        def _build_stopping_criteria(self) -> Any:
            from transformers import StoppingCriteriaList

            return StoppingCriteriaList([StopOnSpecialTokens([], self.tokenizer)])

        def _clean_response(self, response: str) -> str:
            return str(response).strip()

        def _finalize_response(
            self, response: str, messages: list[dict[str, str]]
        ) -> str:
            self._last_reasoning = None
            self.turn_count += 1
            self.conversation_history.append({"role": "assistant", "content": response})
            return response

    return FlagshipSupportAgent


def disable_top_k(agent: Any) -> None:
    """Plain ancestral sampling: ``top_k=0`` on the built generation config.

    ``AgentConfig(top_k=0)`` now does the same; this stays for agents built
    from configs that keep the default ``top_k=50``."""
    generation_config = getattr(agent, "generation_config", None)
    if generation_config is not None:
        generation_config.top_k = 0


def enable_gradient_checkpointing(model: Any) -> bool:
    """Non-reentrant activation checkpointing on a (PEFT-wrapped) HF model."""
    enable = getattr(model, "gradient_checkpointing_enable", None)
    if not callable(enable):
        return False
    try:
        enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        enable()
    require_grads = getattr(model, "enable_input_require_grads", None)
    if callable(require_grads):
        require_grads()
    return True


def _vllm_policy_generate(payload: Mapping[str, Any]) -> GenerateFn:
    """Greedy/temperature replies from vLLM on the agent's own prompt rendering."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from stateset_agents.core.agent_config import AgentConfig

    adapter = payload.get("adapter_path")
    renderer = flagship_agent_cls()(
        AgentConfig(
            **agent_config_kwargs(
                payload,
                temperature=float(payload["temperature"]),
                max_new_tokens=int(payload["max_tokens"]),
            )
        ),
        memory_window=0,
    )
    think_stats = (
        install_think_guard(renderer)
        if payload.get("forbid_think_blocks")
        else ThinkStats()
    )
    # model_path is the local snapshot directory deps.snapshot() resolved at the
    # pinned model_revision, not a hub ID, so there is no revision to pass here.
    renderer.tokenizer = (
        AutoTokenizer.from_pretrained(  # nosec B615 - local pinned snapshot
            str(payload["model_path"]),
            trust_remote_code=bool(payload.get("trust_remote_code", False)),
        )
    )
    bind_chat_template_kwargs(
        renderer.tokenizer, payload.get("chat_template_kwargs") or {}
    )
    kwargs: dict[str, Any] = {
        "model": payload["model_path"],
        "seed": int(payload["seed"]),
        "dtype": "bfloat16" if payload.get("bf16", True) else "auto",
        "gpu_memory_utilization": 0.85,
        "max_model_len": 6144,
    }
    if payload.get("trust_remote_code"):
        kwargs["trust_remote_code"] = True
    if payload.get("vllm_language_model_only"):
        # Composite (vision + text) checkpoints evaluated on text only.
        kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    if adapter:
        kwargs.update({"enable_lora": True, "max_lora_rank": int(payload["lora_r"])})
    llm = LLM(**kwargs)
    params = SamplingParams(
        temperature=float(payload["temperature"]),
        top_p=1.0,
        top_k=-1,
        repetition_penalty=1.0,
        max_tokens=int(payload["max_tokens"]),
        seed=int(payload["seed"]),
        skip_special_tokens=True,
    )
    lora_request = None
    if adapter:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("flagship-policy", 1, str(adapter))

    def generate(conversations: Sequence[Messages]) -> list[str]:
        rendered = [renderer._render_prompt(list(c)) for c in conversations]
        outputs = llm.generate(
            [prompt for prompt, _ in rendered],
            params,
            use_tqdm=False,
            lora_request=lora_request,
        )
        return [
            renderer._finalize_response(
                renderer._clean_response(output.outputs[0].text), messages
            )
            for output, (_, messages) in zip(outputs, rendered, strict=True)
        ]

    generate.think_stats = think_stats  # type: ignore[attr-defined]
    return generate


def _hf_policy_generate(payload: Mapping[str, Any]) -> GenerateFn:
    """The training agent class itself (sequential), for smoke or no vLLM."""
    from stateset_agents.core.agent_config import AgentConfig

    agent = flagship_agent_cls()(
        AgentConfig(
            **agent_config_kwargs(
                payload,
                temperature=float(payload["temperature"]),
                max_new_tokens=int(payload["max_tokens"]),
                adapter_path=payload.get("adapter_path"),
            )
        ),
        memory_window=0,
    )
    think_stats = (
        install_think_guard(agent)
        if payload.get("forbid_think_blocks")
        else ThinkStats()
    )
    ready = {"done": False}

    async def generate(conversations: Sequence[Messages]) -> list[str]:
        if not ready["done"]:
            await agent.initialize()
            disable_top_k(agent)
            bind_chat_template_kwargs(
                agent.tokenizer, payload.get("chat_template_kwargs") or {}
            )
            ready["done"] = True
        replies = []
        for messages in conversations:
            await agent.reset()
            replies.append(await agent.generate_response(list(messages)))
        return replies

    generate.think_stats = think_stats  # type: ignore[attr-defined]
    return generate


def _eval_weights(payload: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Merge the adapter into full weights when the payload asks for it.

    Returns the payload to evaluate (the merged directory as ``model_path``
    and no ``adapter_path``) and which weights it uses: ``base``, ``adapter``
    (LoRA applied by the engine, v1) or ``merged``. ``merge_adapter`` refuses
    an adapter whose merged greedy probe is identical to the base model.
    """
    adapter = payload.get("adapter_path")
    if not adapter:
        return dict(payload), "base"
    if not payload.get("merge_adapter"):
        return dict(payload), "adapter"
    from stateset_agents.training.merge_adapter import merge_adapter

    merged = merge_adapter(
        str(payload["model_path"]), Path(adapter), Path(payload["merged_dir"])
    )
    return (
        {
            **payload,
            "model_path": str(merged),
            "adapter_path": None,
            "vllm_language_model_only": True,
        },
        "merged",
    )


def phase_rollout(payload: Mapping[str, Any]) -> dict[str, Any]:
    """GPU phase: environment rollouts with the base policy or a LoRA adapter."""
    from stateset_agents.utils.reproducibility import set_all_seeds

    set_all_seeds(int(payload["seed"]))
    payload, weights = _eval_weights(payload)
    backend = str(payload.get("backend", "auto"))
    generate: GenerateFn
    used = "hf"
    if backend in {"auto", "vllm"}:
        try:
            generate = _vllm_policy_generate(payload)
            used = "vllm"
        except ImportError:
            if backend == "vllm":
                raise
            logger.warning("vLLM unavailable; falling back to transformers rollouts")
            generate = _hf_policy_generate(payload)
    else:
        generate = _hf_policy_generate(payload)
    scenarios = list(payload["scenarios"])
    environment = make_support_environment(scenarios, int(payload["max_turns"]))
    transcripts = asyncio.run(rollout_episodes(environment, scenarios, generate))
    return {
        "transcripts": {
            key: [list(p) for p in value] for key, value in transcripts.items()
        },
        "backend": used,
        "weights": weights,
        # Evaluations never abort on think markers: replies are sanitized
        # and counted, and a violation rate above EVAL_THINK_FLAG_RATE is
        # flagged in the report.
        "think": getattr(generate, "think_stats", ThinkStats()).summary(
            EVAL_THINK_FLAG_RATE
        ),
        "peak_vram_mb": _peak_vram_mb(),
    }


def phase_train(payload: Mapping[str, Any]) -> dict[str, Any]:
    """GPU phase: on-policy multi-turn GSPO with LoRA, then save the adapter."""
    import torch

    from stateset_agents.core.agent_config import AgentConfig
    from stateset_agents.training.callbacks import ZeroSignalGuard
    from stateset_agents.training.config import TrainingConfig
    from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer
    from stateset_agents.utils.reproducibility import set_all_seeds

    config = payload["config"]
    seed = int(payload["seed"])
    set_all_seeds(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    scenarios = [dict(s) for s in payload["scenarios"]]
    environment = make_support_environment(scenarios, int(payload["max_turns"]))
    agent = flagship_agent_cls()(
        AgentConfig(
            **agent_config_kwargs(
                payload,
                # Training rollouts sample (exploration); evals stay greedy.
                temperature=float(config["rollout_temperature"]),
                top_p=float(config.get("top_p", 1.0)),
                max_new_tokens=int(config["max_completion_length"]),
                lora=config,
            )
        ),
        memory_window=0,
    )
    if payload.get("forbid_think_blocks"):
        install_think_guard(agent)
    guard = ZeroSignalGuard(max_zero_steps=int(config.get("zero_signal_steps", 10)))
    schedule = [list(batch) for batch in payload["schedule"]]

    async def run() -> tuple[Any, dict[str, Any]]:
        await agent.initialize()
        disable_top_k(agent)
        bind_chat_template_kwargs(
            agent.tokenizer, payload.get("chat_template_kwargs") or {}
        )
        checkpointing = enable_gradient_checkpointing(agent.model)
        trainer = MultiTurnGRPOTrainer(
            agent=agent,
            environment=environment,
            reward_fn=support_reward(),
            config=TrainingConfig(**training_config_kwargs(payload, config)),
        )
        await trainer.initialize()
        # Driven step by step (not trainer.train()), so build the warmup +
        # cosine schedule for exactly the planned optimizer steps.
        trainer._setup_scheduler(len(schedule))
        outcome = await run_training_loop(
            trainer,
            scenarios,
            schedule,
            int(config["num_generations"]),
            guard,
            forbid_think_blocks=bool(payload.get("forbid_think_blocks", False)),
            think_limits=think_guard_limits(config),
        )
        outcome["gradient_checkpointing"] = checkpointing
        outcome["reference_model"] = trainer.reference_model is not None
        return trainer, outcome

    _trainer, outcome = asyncio.run(run())
    adapter_dir = Path(payload["adapter_dir"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    agent.model.save_pretrained(str(adapter_dir))
    agent.tokenizer.save_pretrained(str(adapter_dir))
    if not (adapter_dir / "adapter_config.json").is_file():
        raise WorkloadError("training did not produce a LoRA adapter_config.json")
    return {**outcome, "peak_vram_mb": _peak_vram_mb()}


def phase_judge(payload: Mapping[str, Any]) -> dict[str, Any]:
    """GPU phase: score every transcript group with the pinned judge."""
    from dataclasses import asdict

    from stateset_agents.evaluation.support_judge import (
        JudgeItem,
        SupportJudge,
        hf_generate_fn,
        rubric_revision,
        summarize,
        vllm_generate_fn,
    )

    backend = str(payload.get("backend", "auto"))
    model, revision = str(payload["model"]), payload.get("revision")
    if payload.get("model_path"):
        # The pinned local snapshot resolved by deps.snapshot(); its directory
        # is the pinned revision, so no hub revision is passed with it.
        model, revision = str(payload["model_path"]), None
    settings = dict(JUDGE_LOADERS[judge_loader(payload)])
    used = "hf"
    if backend in {"auto", "vllm"}:
        try:
            generate = vllm_generate_fn(model, revision, **settings)
            used = "vllm"
        except ImportError:
            if backend == "vllm":
                raise
            generate = hf_generate_fn(
                model,
                revision,
                chat_template_kwargs=settings.get("chat_template_kwargs"),
            )
    else:
        generate = hf_generate_fn(
            model, revision, chat_template_kwargs=settings.get("chat_template_kwargs")
        )
    judge = SupportJudge(generate)
    names = list(payload["groups"])
    items: list[Any] = []
    spans: dict[str, tuple[int, int]] = {}
    for name in names:
        start = len(items)
        for raw in payload["groups"][name]:
            items.append(
                JudgeItem(
                    id=f"{name}/{raw['id']}",
                    scenario=raw["scenario"],
                    transcript=tuple((str(r), str(c)) for r, c in raw["transcript"]),
                )
            )
        spans[name] = (start, len(items))
    results = judge.score(items)
    groups: dict[str, Any] = {}
    for name in names:
        start, end = spans[name]
        chosen = results[start:end]
        groups[name] = {
            "results": [asdict(result) for result in chosen],
            "summary": summarize(chosen),
            "findings_summary": {
                kind: sum(1 for result in chosen if result.findings.get(kind))
                for kind in ("must_avoid_hits", "placeholders", "repeated_turns")
            },
        }
    return {
        "groups": groups,
        "rubric_revision": rubric_revision(),
        "backend": used,
        "loader": judge_loader(payload),
        "peak_vram_mb": _peak_vram_mb(),
    }


PHASES: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "rollout": phase_rollout,
    "train": phase_train,
    "judge": phase_judge,
}


def _run_phase_main(name: str, source: Path, target: Path) -> int:
    payload = json.loads(source.read_text(encoding="utf-8"))
    output = PHASES[name](payload)
    target.write_text(json.dumps(output, default=str), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--phase", choices=sorted(PHASES))
    parser.add_argument("--phase-input", type=Path)
    parser.add_argument("--phase-output", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mode", choices=("measured", "preflight"), default="measured")
    parser.add_argument("--framework-version")
    parser.add_argument("--model")
    parser.add_argument("--model-revision")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--dataset-source", default=None)
    parser.add_argument("--config-json")
    parser.add_argument("--config-sha256")
    parser.add_argument("--judge-json")
    parser.add_argument("--judge-sha256")
    parser.add_argument("--output", type=Path, help="workload result JSON path")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--scratch-dir", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-model", default=SMOKE_MODEL)
    return parser


_REQUIRED_ARGS = (
    "seed",
    "framework_version",
    "model",
    "model_revision",
    "dataset_revision",
    "config_json",
    "config_sha256",
    "judge_json",
    "judge_sha256",
    "output",
    "artifact_dir",
)


def main(argv: Sequence[str] | None = None, deps: WorkloadDeps | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.phase:
        if args.phase_input is None or args.phase_output is None:
            parser.error("--phase requires --phase-input and --phase-output")
        return _run_phase_main(args.phase, args.phase_input, args.phase_output)
    missing = [name for name in _REQUIRED_ARGS if getattr(args, name) is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if args.scratch_dir is None:
        args.scratch_dir = Path(args.output).resolve().parent / "scratch"
    if deps is None:
        deps = WorkloadDeps(
            run_phase=subprocess_phase_runner(
                Path(args.scratch_dir) / "phases", artifact_dir / "logs"
            )
        )
    outcome: dict[str, Any] | None = None
    error: str | None = None
    try:
        outcome = run_workload(args, deps)
    except Exception as exc:  # noqa: BLE001 - every failure is recorded, never hidden
        logger.exception("flagship workload failed")
        error = f"{type(exc).__name__}: {exc}"
    try:
        hardware = deps.hardware()
    except Exception as exc:  # noqa: BLE001
        hardware = {}
        error = error or f"hardware detection failed: {exc}"
    document = build_result(args, outcome, hardware, error)
    _write_json(Path(args.output), document)
    _write_json(artifact_dir / "workload_result.json", document)
    if Path(args.scratch_dir).exists() and document["status"] == "completed":
        shutil.rmtree(Path(args.scratch_dir) / "gspo", ignore_errors=True)
    print(
        json.dumps(
            {k: document[k] for k in ("status", "measured", "metrics")}, sort_keys=True
        ),
        flush=True,
    )
    return 0 if error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
