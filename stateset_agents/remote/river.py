"""Run the fine-tune on River AI's remote autograd service.

River (https://docs.river.ai) is unlike the other providers here. Modal and
RunPod rent you a machine and we ship ``stateset_agents.training.sft`` to it.
River rents you *gradients*: the model lives on their infrastructure and you
drive the training loop yourself, call by call —
``create_model`` -> N x (``forward_backward`` -> ``optim_step``) ->
``save_weights``. So this executor does not shell out to the training script;
it *is* the training loop, with the tensor math happening elsewhere.

This integration is live-verified (SFT, rejection-sampling flywheels, and
clipped-importance-sampling RL) and its provider call shapes are pinned by an
injectable-client test suite. See ``docs/GETTING_STARTED_RIVER.md`` and
``docs/PROOFS.md`` for dated evidence and remaining operational limits.

Two structural consequences worth knowing before you use it:

**There are no local weights.** ``save_weights`` returns a ``river://`` URI.
The adapter lives on River's servers, reachable through their sampling API.
``fetch()`` therefore writes a *pointer* (``river_checkpoint.json``) plus the
usual ``stateset_manifest.json``, and does not pretend to have downloaded
safetensors. ``stateset-agents serve --checkpoint`` will not load the result;
sample through River instead.

**Provider-resource spec fields do not apply.** ``gpu``, ``gpu_count``,
``container_disk_gb``, ``cloud_type``, and ``network_volume_id`` describe
rented machines. River exposes no machine, so they are ignored (noted in the
logs) rather than treated as errors — the same spec should be submittable to
any provider.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import random
import threading
import time
from asyncio import CancelledError
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stateset_agents.remote.executor import RemoteExecutionError, RemoteExecutor
from stateset_agents.remote.job import JobHandle, JobStatus, RemoteJobSpec
from stateset_agents.remote.ledger import CostEntry, record_entry
from stateset_agents.remote.river_batches import (
    build_group_rl_datums,
    build_sft_batch,
    validate_base_model,
    validate_lora_rank,
)
from stateset_agents.remote.river_rl import (
    RiverRLConfig,
    RLScore,
    RLScorer,
    atomic_json,
    exclusive_run,
    retain_group,
    sample_record,
    score_text,
    validate_record,
    validate_task,
)
from stateset_agents.remote.river_runtime import require_model_access

__all__ = ["CHECKPOINT_POINTER_NAME", "RiverExecutor"]

#: Environment variable holding the River API key (``rv_...``).
RIVER_API_KEY_ENV = "RIVER_API_KEY"


def _sft_options(spec: RemoteJobSpec) -> dict[str, Any]:
    """Validate reproducibility options before constructing a River client."""
    options = {"seed": 0, "shuffle": True}
    supplied = spec.harvest or {}
    unknown = supplied.keys() - options.keys()
    if unknown:
        raise RemoteExecutionError(
            f"Unknown River SFT options: {sorted(unknown)}", provider="river"
        )
    options.update(supplied)
    if type(options["seed"]) is not int or options["seed"] < 0:
        raise RemoteExecutionError(
            "River SFT seed must be a nonnegative integer", provider="river"
        )
    if type(options["shuffle"]) is not bool:
        raise RemoteExecutionError(
            "River SFT shuffle must be boolean", provider="river"
        )
    return options


#: When set (any non-empty value), executor progress lines are ALSO printed
#: to stderr as they happen. The job log is only rendered after the job
#: resolves, which twice turned a slow River pool into a 45-minute mystery —
#: this is the flashlight.
RIVER_VERBOSE_ENV = "STATESET_RIVER_VERBOSE"

logger = logging.getLogger(__name__)

#: Written by ``fetch()`` in place of adapter weights.
CHECKPOINT_POINTER_NAME = "river_checkpoint.json"

#: Spec fields that describe rented hardware. River has none, so a spec
#: carrying them is accepted and they are reported as ignored.
_IGNORED_SPEC_FIELDS = (
    "gpu",
    "gpu_count",
    "container_disk_gb",
    "cloud_type",
    "network_volume_id",
)


def _verbose_log(message: str) -> None:
    import sys

    if os.environ.get(RIVER_VERBOSE_ENV, "").strip():
        print(f"[river] {message}", file=sys.stderr, flush=True)


class _ProgressLogs(list[str]):
    """A log list that optionally echoes appends live (see RIVER_VERBOSE_ENV)."""

    def append(self, item: str) -> None:
        _verbose_log(item)
        super().append(item)


@dataclass
class _RiverJob:
    """Bookkeeping for one River training run."""

    spec: RemoteJobSpec
    status: JobStatus
    logs: list[str] = field(default_factory=_ProgressLogs)
    checkpoint_uri: str | None = None
    steps: int = 0
    final_loss: float | None = None
    tokens: int | None = None
    duration_s: float | None = None
    cost_usd: float | None = None


@dataclass(frozen=True)
class _RlMode:
    """What differs between the single-turn and multi-turn RL loops.

    Everything else — the rounds loop, the eval bracketing, the optimiser
    call, the report and the failure/ledger handling — is shared in
    :meth:`RiverExecutor._run_rl_rounds`.
    """

    #: Previous generation's ``river://`` weights, or None to start from base.
    checkpoint: str | None
    #: Written to ``rl_report.json`` when the spec is a dry run.
    dry_run_report: dict[str, Any]
    #: (model) -> {"passed", "total", "results"}, or None with no eval set.
    greedy_eval: Callable[[Any], dict[str, Any] | None]
    #: (model) -> (datums to train on, per-group mean rewards).
    collect_round: Callable[
        [Any, int], tuple[list[dict[str, Any]], list[float], list[dict[str, Any]]]
    ]
    #: How a round with nothing to learn from is described in the log.
    zero_variance_note: str
    #: Log line and wrapped-error text when the run fails.
    failure_label: str


@dataclass(frozen=True)
class _HarvestMode:
    """What differs between the single-turn and multi-turn harvests.

    The retry loop, artifact writes and ledger live in
    :meth:`RiverExecutor._run_harvest_attempts`.
    """

    checkpoint: str | None
    #: Seed ``harvest_summary.json``; the runner and ``collect_rows`` fill in
    #: the counters, so this is mutated in place rather than frozen.
    summary: dict[str, Any]
    greedy_eval: Callable[[Any], dict[str, Any] | None]
    #: (model, summary) -> kept rows, counting into summary as it goes.
    collect_rows: Callable[[Any, dict[str, Any]], list[dict[str, Any]]]
    #: Noun for the retry and completion log lines ("harvest"/"episode harvest").
    label: str
    #: What ``prompts_with_a_pass`` counts ("prompts"/"scripts").
    unit: str
    #: Suffix on the "current generation" eval line.
    eval_note: str


class RiverExecutor(RemoteExecutor):
    """Drives River's remote training loop for one :class:`RemoteJobSpec`.

    ``client`` is the seam: pass any object implementing River's surface and
    nothing here touches the network or the SDK. Left as ``None``, the real
    ``river_client`` is imported lazily at submit time, so merely listing
    providers never requires the SDK. Internally created clients are closed
    after each submission; injected clients remain owned by the caller. Use
    a separate executor for each concurrent submission.
    """

    name = "river"
    supported_job_kinds = frozenset({"sft", "harvest", "rl"})
    result_kind = "hosted_pointer"
    compute_model = "managed-remote-autograd"
    verification_status = "live-training-verified"

    #: River's default loss for supervised data.
    SFT_LOSS_FN = "cross_entropy"

    def __init__(
        self,
        client: Any = None,
        *,
        tokenizer: Any = None,
        ledger_path: Path | None = None,
        rl_scorer: RLScorer | None = None,
        rl_scorer_id: str | None = None,
    ) -> None:
        self._client = client
        self._owns_client = client is None
        self._submit_lock = threading.Lock()
        self._tokenizer = tokenizer
        self._tokenizers: dict[str, Any] = {}
        self.ledger_path = ledger_path
        self._jobs: dict[str, _RiverJob] = {}
        self._counter = 0
        if rl_scorer is not None and not rl_scorer_id:
            raise ValueError("A custom rl_scorer requires a versioned rl_scorer_id")
        self._rl_scorer = rl_scorer
        self._rl_scorer_id = rl_scorer_id or "stateset-checks-v2"

    # -- SDK seam ----------------------------------------------------------

    def _get_client(self) -> Any:
        """Return the injected client, or construct a real one."""
        if self._client is not None:
            return self._client

        try:
            import river_client
        except ImportError as exc:
            raise RemoteExecutionError.wrap(
                exc,
                "the River SDK is not installed. Install it with "
                "`pip install river-client`, or pass an explicit client to "
                "RiverExecutor(client=...).",
                provider=self.name,
            ) from exc

        api_key = os.environ.get(RIVER_API_KEY_ENV, "").strip()
        if not api_key:
            raise RemoteExecutionError(
                f"{RIVER_API_KEY_ENV} is not set. Create an API key in the "
                f"River console and export it: "
                f"`export {RIVER_API_KEY_ENV}=rv_...`.",
                provider=self.name,
            )
        self._client = river_client.Client(api_key=api_key)
        return self._client

    def _get_tokenizer(self, base_model: str) -> Any:
        """Tokenizer used to build batches.

        River takes token ids, not text, so *we* tokenize. Injectable because
        loading a real tokenizer needs `transformers` and a network fetch —
        neither of which a unit test should require.
        """
        if self._tokenizer is not None:
            return self._tokenizer
        if base_model in self._tokenizers:
            return self._tokenizers[base_model]
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RemoteExecutionError.wrap(
                exc,
                "building River batches needs a tokenizer; install "
                "`transformers` or pass RiverExecutor(tokenizer=...).",
                provider=self.name,
            ) from exc
        try:
            # base_model is the caller's own model id, not attacker input;
            # pinning a revision would break arbitrary user-chosen models
            # (same rationale as training/sft.py).
            tokenizer = AutoTokenizer.from_pretrained(base_model)  # nosec: B615
        except Exception as exc:  # noqa: BLE001 - hub/network/auth all land here
            raise RemoteExecutionError.wrap(
                exc,
                f"could not load a tokenizer for {base_model!r}",
                provider=self.name,
            ) from exc
        self._tokenizers[base_model] = tokenizer
        return tokenizer

    def _checked_client(self, spec: RemoteJobSpec, job: _RiverJob) -> Any:
        """Reject unavailable models before a session or tokenizer is created."""
        try:
            client = self._get_client()
            require_model_access(client, spec.base_model)
        except Exception:
            job.status = JobStatus.FAILED
            raise
        job.logs.append(f"River account advertises model access: {spec.base_model}")
        return client

    # -- submit ------------------------------------------------------------

    #: River reports account state in an OpenAI-shaped error envelope
    #: (``{"error": {"message": ..., "type": ...}}``). Observed live against
    #: api.river.ai: an unfunded account answers 402 with
    #: "Billing: insufficient_funds", and a missing key answers 401. Both are
    #: states the user must act on, so they are named rather than wrapped in
    #: a generic training failure.
    _ACCOUNT_HINTS: tuple[tuple[str, str], ...] = (
        (
            "insufficient_funds",
            "River reports the account has no credits "
            "(HTTP 402, 'Billing: insufficient_funds'). Add funds at "
            "https://river.ai before running a job — no training call will "
            "succeed until then.",
        ),
        (
            "billing",
            "River rejected the request for a billing reason. Check the "
            "account's credits and plan at https://river.ai.",
        ),
        (
            "unauthorized",
            "River rejected the API key (HTTP 401). Check RIVER_API_KEY.",
        ),
        (
            "invalid_api_key",
            "River rejected the API key. Check RIVER_API_KEY.",
        ),
    )

    def _account_error(self, exc: Exception) -> RemoteExecutionError | None:
        """Translate an account-state failure into an actionable error.

        Returns None when the failure is not one of these, so genuine
        training errors keep their own reporting.
        """
        text = str(exc).lower()
        for needle, guidance in self._ACCOUNT_HINTS:
            if needle in text:
                return RemoteExecutionError(guidance, provider=self.name)
        return None

    def submit(self, spec: RemoteJobSpec) -> JobHandle:
        """Run synchronously, releasing owned clients on every exit path."""
        if not self._submit_lock.acquire(blocking=False):
            raise RemoteExecutionError(
                "RiverExecutor already has an active submission; use a separate "
                "executor for concurrent jobs",
                provider=self.name,
            )
        try:
            return self._submit_with_cleanup(spec)
        finally:
            self._submit_lock.release()

    def _submit_with_cleanup(self, spec: RemoteJobSpec) -> JobHandle:
        self.validate_spec(spec)
        self._counter += 1
        job_id = f"river-{self._counter}"
        handle = JobHandle(provider=self.name, job_id=job_id)
        # _ProgressLogs so appends echo live under STATESET_RIVER_VERBOSE —
        # a plain [] here silently bypassed the echo and hid a mid-run
        # transient retry for an hour (observed live).
        logs: list[str] = _ProgressLogs()
        job = _RiverJob(spec=spec, status=JobStatus.PENDING, logs=logs)
        self._jobs[job_id] = job

        failed = False
        try:
            return self._submit_job(handle, job, spec)
        except BaseException as exc:
            failed = True
            if not job.status.is_terminal:
                job.status = (
                    JobStatus.CANCELLED
                    if isinstance(exc, (KeyboardInterrupt, CancelledError))
                    else JobStatus.FAILED
                )
            raise
        finally:
            if self._owns_client and self._client is not None:
                # Detach before closing: even a partially closed client must
                # never be reused by a subsequent submission.
                client, self._client = self._client, None
                try:
                    client.close()
                except Exception as exc:
                    message = (
                        f"River client cleanup failed ({type(exc).__name__}) "
                        f"for {job_id}; inspect the retained job status and "
                        "artifacts before resubmitting"
                    )
                    # SDK exception text can contain credentials. Keep the
                    # cleanup diagnostic without masking a training failure.
                    logs.append(message)
                    logger.warning(message)
                    if not failed:
                        raise RemoteExecutionError(
                            message,
                            provider=self.name,
                            job_id=job_id,
                            stage="client_cleanup",
                        ) from exc

    def _submit_job(
        self, handle: JobHandle, job: _RiverJob, spec: RemoteJobSpec
    ) -> JobHandle:
        logs = job.logs
        job_id = handle.job_id
        ignored = [f for f in _IGNORED_SPEC_FIELDS if _is_set(spec, f)]
        if ignored:
            logs.append(
                "ignoring machine-shaped spec fields that River has no "
                f"equivalent for: {', '.join(ignored)}"
            )

        # Validate before tokenizing anything — cheapest failure first.
        validate_lora_rank(spec.lora_r)
        validate_base_model(spec.base_model)

        if spec.job_kind == "harvest":
            return self._submit_harvest(handle, job, spec)
        if spec.job_kind == "rl":
            return self._submit_rl(handle, job, spec)

        _sft_options(spec)
        client = None if spec.dry_run else self._checked_client(spec, job)
        tokenizer = self._get_tokenizer(spec.base_model)

        from stateset_agents.training.sft import load_chat_dataset

        rows = load_chat_dataset(Path(spec.dataset))
        data = build_sft_batch(rows, tokenizer, max_length=spec.max_length)
        if not data:
            job.status = JobStatus.FAILED
            logs.append(f"no trainable rows in {spec.dataset}")
            return handle
        logs.append(f"built {len(data)} River SFT data from {len(rows)} rows")

        if spec.dry_run:
            logs.append(
                f"dry run: would train {spec.num_epochs} epoch(s) of "
                f"{len(data)} data on {spec.base_model} "
                f"(LoRA rank {spec.lora_r}) via River"
            )
            job.status = JobStatus.SUCCEEDED
            return handle

        started = time.monotonic()
        job.status = JobStatus.RUNNING
        try:
            self._train_with_recovery(client, spec, data, job)
        except RemoteExecutionError:
            job.status = JobStatus.FAILED
            job.duration_s = time.monotonic() - started
            self._record_cost(job_id, job)
            raise
        except Exception as exc:  # noqa: BLE001 - unknown SDK exception surface
            job.status = JobStatus.FAILED
            job.duration_s = time.monotonic() - started
            logs.append(f"River training failed: {exc}")
            self._record_cost(job_id, job)
            account = self._account_error(exc)
            if account is not None:
                logs.append(str(account))
                raise account from exc
            raise RemoteExecutionError.wrap(
                exc, "River training run failed", provider=self.name
            ) from exc

        job.duration_s = time.monotonic() - started
        job.status = JobStatus.SUCCEEDED
        self._record_cost(job_id, job)
        return handle

    # -- RL: one loop, two modes -------------------------------------------

    def _run_rl_rounds(
        self,
        handle: JobHandle,
        job: _RiverJob,
        spec: RemoteJobSpec,
        mode: _RlMode,
    ) -> JobHandle:
        """Drive a durable RL run shared by single-turn and episode collectors."""
        from stateset_agents.remote.river_rl_runner import run_rl

        with exclusive_run(Path(spec.output_dir)):
            return run_rl(self, handle, job, spec, mode)

    def _submit_episode_rl(
        self,
        handle: JobHandle,
        job: _RiverJob,
        spec: RemoteJobSpec,
        scripts: list[dict[str, Any]],
    ) -> JobHandle:
        """Multi-turn RL: whole conversations, episode-level advantages.

        Built for the wall imitation hit at rung 5: when most sampled
        episodes pass, passing examples carry little signal — but graded
        rewards still separate near-misses, violations, and clean passes.
        Rounds of: branch-rollout every script (capturing per-turn ids and
        sampler logprobs), grade each episode (token fraction +
        completeness bonus − violation), compute group-relative advantages
        per script, broadcast each episode's advantage across all its
        turns' datums, ``train_step`` with the clipped-IS loss. Greedy
        episode eval brackets every round.
        """
        knobs = spec.harvest or {}
        config = RiverRLConfig.from_knobs(knobs)
        branches = config.best_of
        eval_scripts = [
            e for e in (spec.eval_prompts or []) if isinstance(e, dict) and "turns" in e
        ]

        def greedy_eval(model: Any) -> dict[str, Any] | None:
            if not eval_scripts:
                return None
            outs = _rollout_episodes(
                model,
                spec.base_model,
                eval_scripts,
                branches=1,
                temperature=0.0,
                top_p=1.0,
                max_tokens=spec.eval_max_new_tokens,
            )
            results = []
            for script, bouts in zip(eval_scripts, outs, strict=True):
                scored = self._score_rl(script, bouts[0])
                results.append(
                    {
                        "prompt": " / ".join(script["turns"]),
                        "finetuned": " ||| ".join(bouts[0]),
                        "passed": scored.passed,
                        "detail": scored.components,
                    }
                )
            return {
                "passed": sum(1 for r in results if r["passed"]),
                "total": len(results),
                "results": results,
            }

        def collect_round(
            model: Any, rnd: int
        ) -> tuple[list[dict[str, Any]], list[float], list[dict[str, Any]]]:
            episodes = _rollout_episodes(
                model,
                spec.base_model,
                scripts,
                branches=branches,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_new_tokens,
                capture=True,
                seed=config.seed + rnd,
            )
            data: list[dict[str, Any]] = []
            mean_rewards: list[float] = []
            audit: list[dict[str, Any]] = []
            for script, branch_records in zip(scripts, episodes, strict=True):
                flat = [t for branch in branch_records for t in branch]
                record: dict[str, Any] = {
                    "task": script,
                    "records": flat,
                    "skip_reason": None,
                }
                audit.append(record)
                if not retain_group(flat, config):
                    record["skip_reason"] = "truncated"
                    continue
                scores = [
                    self._score_rl(script, [t["text"] for t in turns])
                    for turns in branch_records
                ]
                rewards = [s.reward for s in scores]
                record["rewards"] = rewards
                record["scores"] = [
                    {"passed": s.passed, "components": s.components} for s in scores
                ]
                mean_rewards.append(sum(rewards) / len(rewards))
                if all(r == rewards[0] for r in rewards):
                    record["skip_reason"] = "zero_variance"
                    continue
                mean = sum(rewards) / len(rewards)
                for records, reward in zip(branch_records, rewards, strict=True):
                    data.extend(_episode_rl_datums(records, reward - mean))
            return data, mean_rewards, audit

        return self._run_rl_rounds(
            handle,
            job,
            spec,
            _RlMode(
                checkpoint=_checkpoint_from_pointer(
                    knobs.get("adapter_dir"),
                    base_model=spec.base_model,
                    lora_rank=spec.lora_r if spec.job_kind == "rl" else None,
                ),
                dry_run_report={
                    "rounds": int(knobs.get("rounds", 4)),
                    "episodes": len(scripts),
                    "dry_run": True,
                },
                greedy_eval=greedy_eval,
                collect_round=collect_round,
                zero_variance_note="every episode group zero-variance",
                failure_label="River episode RL failed",
            ),
        )

    def _submit_rl(
        self, handle: JobHandle, job: _RiverJob, spec: RemoteJobSpec
    ) -> JobHandle:
        """GRPO-style RL, zero infrastructure: rounds of sample -> grade ->
        group-relative advantages -> ``train_step(loss_fn=cispo)``.

        The rejection-sampling flywheel imitates winners; this trains on the
        WHOLE sample group, gradient-weighted by graded reward — failures
        push probability mass away, and refusal violations (forbid hits)
        are punished directly instead of merely filtered out. Logprobs come
        from River's own sampler verbatim (never recomputed), prompt ids
        from the echoed tokenization, and the datum layout is their
        pre-shifted RL contract (see ``build_group_rl_datums``).
        """
        from stateset_agents.training.sft import normalize_eval_prompts

        raw_prompts = json.loads(Path(spec.dataset).read_text())
        if not isinstance(raw_prompts, list) or not raw_prompts:
            raise ValueError("River RL dataset must be a nonempty JSON list")
        config = RiverRLConfig.from_knobs(spec.harvest)
        if spec.max_cost_usd is not None:
            raise ValueError(
                "River cannot enforce a dollar ceiling; use max_generated_tokens"
            )
        if not all(isinstance(p, dict) for p in raw_prompts):
            raise ValueError("River RL tasks must be objects with explicit scoring")
        episode_mode = "turns" in raw_prompts[0]
        for task in raw_prompts + list(spec.eval_prompts or []):
            if not isinstance(task, dict) or ("turns" in task) != episode_mode:
                raise ValueError("Do not mix single-turn and episode RL tasks")
            validate_task(task, custom_scorer=self._rl_scorer is not None)
        train_keys = {
            json.dumps(p.get("turns", p.get("prompt")), sort_keys=True)
            for p in raw_prompts
        }
        eval_keys = {
            json.dumps(p.get("turns", p.get("prompt")), sort_keys=True)
            for p in (spec.eval_prompts or [])
            if isinstance(p, dict)
        }
        if train_keys & eval_keys:
            raise ValueError("RL training and validation prompts must be disjoint")
        if (
            raw_prompts
            and isinstance(raw_prompts[0], dict)
            and "turns" in raw_prompts[0]
        ):
            return self._submit_episode_rl(handle, job, spec, raw_prompts)
        prompts = normalize_eval_prompts(raw_prompts)
        eval_specs = normalize_eval_prompts(list(spec.eval_prompts or []))
        knobs = spec.harvest or {}
        num_samples = config.best_of

        eval_texts = [e["prompt"] for e in eval_specs]

        def greedy_eval(model: Any) -> dict[str, Any] | None:
            if not eval_specs:
                return None
            outs = _sample_texts(
                model,
                spec.base_model,
                eval_texts,
                num_samples=1,
                temperature=0.0,
                max_tokens=spec.eval_max_new_tokens,
            )
            results = []
            for espec, completions in zip(eval_specs, outs, strict=True):
                scored = self._score_rl(espec, completions[0])
                results.append(
                    {
                        "prompt": espec["prompt"],
                        "finetuned": completions[0],
                        "checks": {"passed": scored.passed},
                        "reward": scored.reward,
                        "components": scored.components,
                    }
                )
            passed = sum(1 for r in results if r["checks"]["passed"])
            return {"passed": passed, "total": len(results), "results": results}

        def collect_round(
            model: Any, rnd: int
        ) -> tuple[list[dict[str, Any]], list[float], list[dict[str, Any]]]:
            groups, prompt_ids_per = _rl_sample_groups(
                model,
                spec.base_model,
                [p["prompt"] for p in prompts],
                num_samples=num_samples,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_new_tokens,
                seed=config.seed + rnd,
            )
            data: list[dict[str, Any]] = []
            mean_rewards: list[float] = []
            audit: list[dict[str, Any]] = []
            for pspec, group, prompt_ids in zip(
                prompts, groups, prompt_ids_per, strict=True
            ):
                if len(group) != num_samples:
                    raise ValueError("River returned an incomplete RL sample group")
                samples = [sample_record(s, prompt_ids) for s in group]
                record: dict[str, Any] = {
                    "task": pspec,
                    "records": samples,
                    "skip_reason": None,
                }
                audit.append(record)
                if not retain_group(samples, config):
                    record["skip_reason"] = "truncated"
                    continue
                scores = [self._score_rl(pspec, s["text"]) for s in samples]
                rewards = [s.reward for s in scores]
                record["rewards"] = rewards
                record["scores"] = [
                    {"passed": s.passed, "components": s.components} for s in scores
                ]
                mean_rewards.append(sum(rewards) / len(rewards))
                if all(r == rewards[0] for r in rewards):
                    record["skip_reason"] = "zero_variance"
                data.extend(build_group_rl_datums(prompt_ids, samples, rewards))
            return data, mean_rewards, audit

        return self._run_rl_rounds(
            handle,
            job,
            spec,
            _RlMode(
                checkpoint=_checkpoint_from_pointer(
                    knobs.get("adapter_dir"),
                    base_model=spec.base_model,
                    lora_rank=spec.lora_r if spec.job_kind == "rl" else None,
                ),
                dry_run_report={
                    "rounds": int(knobs.get("rounds", 4)),
                    "prompts": len(prompts),
                    "dry_run": True,
                },
                greedy_eval=greedy_eval,
                collect_round=collect_round,
                zero_variance_note="every group zero-variance",
                failure_label="River RL run failed",
            ),
        )

    def _score_rl(self, task: dict[str, Any], replies: str | list[str]) -> RLScore:
        """Use the same explicit verifier for learning and validation."""
        if self._rl_scorer is not None:
            score = self._rl_scorer(task, replies)
            if not isinstance(score, RLScore):
                raise ValueError("Custom River RL scorer must return RLScore")
            return score
        if isinstance(replies, str):
            return score_text(task, replies)
        passed, detail = _score_episode(task, replies)
        return RLScore(_graded_episode_reward(task, replies), passed, detail)

    # -- harvest: one retry loop, two modes --------------------------------

    def _run_harvest_attempts(
        self,
        handle: JobHandle,
        job: _RiverJob,
        spec: RemoteJobSpec,
        mode: _HarvestMode,
    ) -> JobHandle:
        """Sample, keep what passes, write the pod-contract artifacts.

        The retry policy comes from the SDK's own recovery taxonomy: a
        connection or timeout error means back off, rebuild the session and
        try again — observed live, a gen-2 harvest died on 'Server
        unavailable' and took a finished generation's momentum with it.
        ``mode`` supplies only how to score the eval set and how to turn
        samples into kept rows.
        """
        import time as _time

        logs = job.logs
        summary = mode.summary
        output_dir = Path(spec.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if spec.dry_run:
            (output_dir / "harvest_summary.json").write_text(
                json.dumps(summary, indent=2)
            )
            logs.append("dry run: harvest plan only, nothing sampled")
            job.status = JobStatus.SUCCEEDED
            return handle

        client = self._checked_client(spec, job)
        started = _time.monotonic()
        job.status = JobStatus.RUNNING
        transient = self._transient_exceptions(client)
        attempt = 0
        rows: list[dict[str, Any]] = []
        while True:
            attempt += 1
            try:
                with _open_session(
                    client, project=Path(spec.output_dir).name or None
                ) as session:
                    model = session.create_model(
                        base_model=spec.base_model,
                        checkpoint=_inference_checkpoint(client, mode.checkpoint),
                    )
                    logs.append(
                        f"sampling from {mode.checkpoint or spec.base_model} via River"
                    )
                    # Counters are per-attempt: a retry re-samples everything.
                    rows = []
                    summary["samples"] = 0
                    summary["prompts_with_a_pass"] = 0
                    evaluated = mode.greedy_eval(model)
                    if evaluated is not None:
                        summary["eval"] = evaluated
                        logs.append(
                            f"current generation{mode.eval_note}: "
                            f"{evaluated['passed']}/{evaluated['total']}"
                        )
                    rows = mode.collect_rows(model, summary)
                break
            except transient as exc:
                if attempt >= self.MAX_TRANSIENT_ATTEMPTS:
                    self._fail_harvest(job, started, handle, exc)
                delay = self.TRANSIENT_BACKOFF_S * (2 ** (attempt - 1))
                logs.append(
                    f"transient River failure ({type(exc).__name__}): "
                    f"{exc} — retrying {mode.label} in {delay:.0f}s "
                    f"(attempt {attempt}/{self.MAX_TRANSIENT_ATTEMPTS})"
                )
                self._sleep(delay)
            except Exception as exc:  # noqa: BLE001 - unknown SDK surface
                self._fail_harvest(job, started, handle, exc)

        summary["kept"] = len(rows)
        with (output_dir / "harvest.jsonl").open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        (output_dir / "harvest_summary.json").write_text(json.dumps(summary, indent=2))
        logs.append(
            f"{mode.label}: kept {summary['kept']}/{summary['samples']} across "
            f"{summary['prompts_with_a_pass']}/{summary['prompts']} {mode.unit}"
        )
        job.duration_s = _time.monotonic() - started
        job.status = JobStatus.SUCCEEDED
        self._record_cost(handle.job_id, job)
        return handle

    def _submit_episode_harvest(
        self,
        handle: JobHandle,
        job: _RiverJob,
        spec: RemoteJobSpec,
        scripts: list[dict[str, Any]],
    ) -> JobHandle:
        """Multi-turn episode harvest: the flywheel learns conversations.

        Rolls ``best_of`` independent episodes per script (branches batched
        per turn), scores at the EPISODE level (every turn's checks must
        hit — turn 2's reference echo proves context carryover — and no
        forbid anywhere), and writes passing episodes as multi-turn chat
        rows: the SFT batcher already weights every assistant turn. Eval
        scripts are rolled greedily first for the current generation's
        score. Same artifacts, same retry policy as the single-turn
        harvest.
        """
        knobs = spec.harvest or {}
        eval_scripts = [
            e for e in (spec.eval_prompts or []) if isinstance(e, dict) and "turns" in e
        ]
        branches = int(knobs.get("best_of", 8))

        def greedy_eval(model: Any) -> dict[str, Any] | None:
            if not eval_scripts:
                return None
            greedy = _rollout_episodes(
                model,
                spec.base_model,
                eval_scripts,
                branches=1,
                temperature=0.0,
                top_p=1.0,
                max_tokens=spec.eval_max_new_tokens,
            )
            results = []
            for script, branches_out in zip(eval_scripts, greedy, strict=True):
                passed, detail = _score_episode(script, branches_out[0])
                results.append(
                    {
                        "prompt": " / ".join(script["turns"]),
                        "completion": " ||| ".join(branches_out[0]),
                        "passed": passed,
                        "detail": detail,
                    }
                )
            return {
                "passed": sum(1 for r in results if r["passed"]),
                "total": len(results),
                "results": results,
            }

        def collect_rows(model: Any, summary: dict[str, Any]) -> list[dict[str, Any]]:
            episodes = _rollout_episodes(
                model,
                spec.base_model,
                scripts,
                branches=branches,
                temperature=float(knobs.get("temperature", 0.9)),
                top_p=float(knobs.get("top_p", 0.95)),
                max_tokens=int(knobs.get("max_new_tokens", 300)),
            )
            rows: list[dict[str, Any]] = []
            for script, branch_replies in zip(scripts, episodes, strict=True):
                script_pass = False
                for replies in branch_replies:
                    summary["samples"] += 1
                    passed, _detail = _score_episode(script, replies)
                    if not passed:
                        continue
                    script_pass = True
                    messages: list[dict[str, str]] = []
                    for user, assistant in zip(script["turns"], replies, strict=True):
                        messages.append({"role": "user", "content": user})
                        messages.append({"role": "assistant", "content": assistant})
                    rows.append({"messages": messages})
                if script_pass:
                    summary["prompts_with_a_pass"] += 1
            return rows

        return self._run_harvest_attempts(
            handle,
            job,
            spec,
            _HarvestMode(
                checkpoint=_checkpoint_from_pointer(
                    knobs.get("adapter_dir"),
                    base_model=spec.base_model,
                    lora_rank=spec.lora_r if spec.job_kind == "rl" else None,
                ),
                summary={
                    "base_model": spec.base_model,
                    "adapter_dir": knobs.get("adapter_dir"),
                    "best_of": branches,
                    "temperature": knobs.get("temperature", 0.9),
                    "prompts": len(scripts),
                    "samples": 0,
                    "kept": 0,
                    "prompts_with_a_pass": 0,
                    "eval": None,
                    "episodes": True,
                    "dry_run": bool(spec.dry_run),
                },
                greedy_eval=greedy_eval,
                collect_rows=collect_rows,
                label="episode harvest",
                unit="scripts",
                eval_note=" (episodes, greedy)",
            ),
        )

    def _fail_harvest(
        self, job: _RiverJob, started: float, handle: JobHandle, exc: Exception
    ) -> None:
        """Record a harvest failure and raise the caller-facing error."""
        import time as _time

        job.status = JobStatus.FAILED
        job.duration_s = _time.monotonic() - started
        job.logs.append(f"River harvest failed: {exc}")
        self._record_cost(handle.job_id, job)
        account = self._account_error(exc)
        if account is not None:
            raise account from exc
        raise RemoteExecutionError.wrap(
            exc, "River harvest failed", provider=self.name
        ) from exc

    def _submit_harvest(
        self, handle: JobHandle, job: _RiverJob, spec: RemoteJobSpec
    ) -> JobHandle:
        """Best-of-N harvest with ZERO rented machines.

        The flywheel's forward step, entirely through River's sampling API:
        create a model from the previous generation's ``river://``
        checkpoint (or the bare base), ``model.sample`` ``best_of``
        completions per harvest prompt, keep the ones passing their own
        checks, and write ``harvest.jsonl`` + ``harvest_summary.json``
        exactly where the pod-based harvest would — so ``run_flywheel``
        cannot tell the difference. The eval set is measured greedily from
        the same in-memory model, giving the loop its "before" number.
        """
        from stateset_agents.training.harvest import build_harvest_rows
        from stateset_agents.training.sft import evaluate_checks, normalize_eval_prompts

        prompts = json.loads(Path(spec.dataset).read_text())
        if prompts and isinstance(prompts[0], dict) and "turns" in prompts[0]:
            return self._submit_episode_harvest(handle, job, spec, prompts)
        harvest_specs = normalize_eval_prompts(prompts)
        eval_specs = normalize_eval_prompts(list(spec.eval_prompts or []))
        knobs = spec.harvest or {}
        best_of = int(knobs.get("best_of", 8))

        def greedy_eval(model: Any) -> dict[str, Any] | None:
            if not eval_specs:
                return None
            greedy = _sample_texts(
                model,
                spec.base_model,
                [s["prompt"] for s in eval_specs],
                num_samples=1,
                temperature=0.0,
                max_tokens=spec.eval_max_new_tokens,
            )
            results = []
            for espec, completions in zip(eval_specs, greedy, strict=True):
                checked = evaluate_checks(
                    completions[0],
                    espec.get("expect", []),
                    espec.get("forbid", []),
                )
                results.append(
                    {
                        "prompt": espec["prompt"],
                        "completion": completions[0],
                        **checked,
                    }
                )
            return {
                "passed": sum(1 for r in results if r["passed"]),
                "total": len(results),
                "results": results,
            }

        def collect_rows(model: Any, summary: dict[str, Any]) -> list[dict[str, Any]]:
            sampled = _sample_texts(
                model,
                spec.base_model,
                [s["prompt"] for s in harvest_specs],
                num_samples=best_of,
                temperature=float(knobs.get("temperature", 0.9)),
                top_p=float(knobs.get("top_p", 0.95)),
                max_tokens=int(knobs.get("max_new_tokens", 300)),
            )
            rows: list[dict[str, Any]] = []
            for hspec, completions in zip(harvest_specs, sampled, strict=True):
                summary["samples"] += len(completions)
                kept = build_harvest_rows(hspec, completions)
                if kept:
                    summary["prompts_with_a_pass"] += 1
                rows.extend(kept)
            return rows

        return self._run_harvest_attempts(
            handle,
            job,
            spec,
            _HarvestMode(
                checkpoint=_checkpoint_from_pointer(
                    knobs.get("adapter_dir"),
                    base_model=spec.base_model,
                    lora_rank=spec.lora_r if spec.job_kind == "rl" else None,
                ),
                summary={
                    "base_model": spec.base_model,
                    "adapter_dir": knobs.get("adapter_dir"),
                    "best_of": best_of,
                    "temperature": knobs.get("temperature", 0.9),
                    "prompts": len(harvest_specs),
                    "samples": 0,
                    "kept": 0,
                    "prompts_with_a_pass": 0,
                    "eval": None,
                    "dry_run": bool(spec.dry_run),
                },
                greedy_eval=greedy_eval,
                collect_rows=collect_rows,
                label="harvest",
                unit="prompts",
                eval_note=" (greedy)",
            ),
        )

    MAX_TRANSIENT_ATTEMPTS = 3
    TRANSIENT_BACKOFF_S = 10.0

    def _transient_exceptions(self, client: Any) -> tuple[type[Exception], ...]:
        """The SDK's transient error types, if the SDK is importable."""
        try:
            import river_client
        except ImportError:
            river_client = _river_module(client)
        names = ("RiverConnectionError", "RiverTimeoutError")
        return tuple(
            exc
            for name in names
            if isinstance(exc := getattr(river_client, name, None), type)
            and issubclass(exc, Exception)
        )

    def _train_with_recovery(
        self,
        client: Any,
        spec: RemoteJobSpec,
        data: list[dict[str, Any]],
        job: _RiverJob,
    ) -> None:
        """Run ``_train``, retrying transient SDK failures with backoff.

        Each retry rebuilds the session from scratch (sessions are not
        durable — checkpoints are; these short SFT runs restart rather than
        resume). Auth, model-not-found, and data errors propagate on the
        first throw — retrying those fails identically.
        """
        transient = self._transient_exceptions(client)
        attempt = 0
        while True:
            attempt += 1
            try:
                job.steps = 0  # each attempt trains from scratch
                self._train(client, spec, data, job)
                return
            except transient as exc:
                if attempt >= self.MAX_TRANSIENT_ATTEMPTS:
                    raise
                delay = self.TRANSIENT_BACKOFF_S * (2 ** (attempt - 1))
                job.logs.append(
                    f"transient River failure ({type(exc).__name__}): {exc} "
                    f"— rebuilding session and retrying in {delay:.0f}s "
                    f"(attempt {attempt}/{self.MAX_TRANSIENT_ATTEMPTS})"
                )
                self._sleep(delay)

    #: Seam for tests; time.sleep in production.
    _sleep = staticmethod(time.sleep)

    def _train(
        self,
        client: Any,
        spec: RemoteJobSpec,
        data: list[dict[str, Any]],
        job: _RiverJob,
    ) -> None:
        """The training loop River expects the caller to own."""
        logs = job.logs
        river = _river_module(client)
        options = _sft_options(spec)
        rng = random.Random(options["seed"])
        lora = river.LoraConfig(
            rank=spec.lora_r,
            train_attn=True,
            train_mlp=True,
            train_unembed=False,
            seed=options["seed"],
        )
        with _open_session(
            client, project=Path(spec.output_dir).name or None
        ) as session:
            model = session.create_model(base_model=spec.base_model, lora=lora)
            logs.append(
                f"created River model on {spec.base_model} "
                f"(LoRA rank {spec.lora_r}, attn+mlp)"
            )

            size = max(1, spec.per_device_batch_size)
            tokens = 0
            for epoch in range(spec.num_epochs):
                epoch_data = list(data)
                if options["shuffle"]:
                    rng.shuffle(epoch_data)
                for start in range(0, len(data), size):
                    chunk = epoch_data[start : start + size]
                    if hasattr(model, "train_step"):
                        # The SDK's preferred complete step: forward+backward
                        # and the optimizer step pipelined server-side.
                        result, _ = model.train_step(
                            chunk, lr=spec.learning_rate, loss_fn=self.SFT_LOSS_FN
                        )
                    else:
                        result = model.forward_backward(chunk, loss_fn=self.SFT_LOSS_FN)
                    loss = _extract(result, "loss_mean", "loss")
                    if loss is not None:
                        job.final_loss = float(loss)
                        if not math.isfinite(job.final_loss):
                            raise ValueError("River SFT returned a nonfinite loss")
                    if not hasattr(model, "train_step"):
                        model.optim_step(lr=spec.learning_rate)
                    job.steps += 1
                    if job.steps % 10 == 0:
                        _verbose_log(
                            f"step {job.steps}"
                            + (
                                f" loss {job.final_loss:.4f}"
                                if job.final_loss is not None
                                else ""
                            )
                        )
                    counted = _extract(result, "num_tokens", "tokens", "total_tokens")
                    if counted is not None:
                        tokens += int(counted)
                logs.append(
                    f"epoch {epoch + 1}/{spec.num_epochs} complete "
                    f"({job.steps} steps"
                    + (
                        f", loss {job.final_loss:.4f}"
                        if job.final_loss is not None
                        else ""
                    )
                    + ")"
                )
            job.tokens = tokens or None

            name = Path(spec.output_dir).name or "adapter"
            uri = model.save_weights(name, mode="inference")
            job.checkpoint_uri = _as_uri(uri)
            logs.append(f"saved River checkpoint: {job.checkpoint_uri}")

            if spec.eval_prompts:
                # Greedy-score the freshly trained weights in-session and
                # write eval_results.json in the sft writer's shape, so the
                # flywheel reads River generations exactly like pod ones.
                from stateset_agents.training.sft import (
                    evaluate_checks,
                    normalize_eval_prompts,
                )

                episode_scripts = [
                    e for e in spec.eval_prompts if isinstance(e, dict) and "turns" in e
                ]
                if episode_scripts:
                    greedy_eps = _rollout_episodes(
                        model,
                        spec.base_model,
                        episode_scripts,
                        branches=1,
                        temperature=0.0,
                        top_p=1.0,
                        max_tokens=spec.eval_max_new_tokens,
                    )
                    eval_rows = []
                    for script, branches_out in zip(
                        episode_scripts, greedy_eps, strict=True
                    ):
                        passed_ep, detail = _score_episode(script, branches_out[0])
                        eval_rows.append(
                            {
                                "prompt": " / ".join(script["turns"]),
                                "finetuned": " ||| ".join(branches_out[0]),
                                "passed": passed_ep,
                                "detail": detail,
                            }
                        )
                    out_dir = Path(spec.output_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    (out_dir / "eval_results.json").write_text(
                        json.dumps(eval_rows, indent=2)
                    )
                    passed = sum(1 for r in eval_rows if r["passed"])
                    logs.append(
                        f"eval: {passed}/{len(eval_rows)} episode(s) passed "
                        "(greedy, in-session)"
                    )
                    return

                specs_ = normalize_eval_prompts(list(spec.eval_prompts))
                greedy = _sample_texts(
                    model,
                    spec.base_model,
                    [e["prompt"] for e in specs_],
                    num_samples=1,
                    temperature=0.0,
                    max_tokens=spec.eval_max_new_tokens,
                )
                plain_rows = []
                for espec, completions in zip(specs_, greedy, strict=True):
                    checked = evaluate_checks(
                        completions[0],
                        espec.get("expect", []),
                        espec.get("forbid", []),
                    )
                    plain_rows.append(
                        {
                            "prompt": espec["prompt"],
                            "finetuned": completions[0],
                            "checks": checked,
                        }
                    )
                out_dir = Path(spec.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "eval_results.json").write_text(
                    json.dumps(plain_rows, indent=2)
                )
                passed = sum(1 for r in plain_rows if r["checks"]["passed"])
                logs.append(
                    f"eval: {passed}/{len(plain_rows)} prompt(s) passed "
                    "(greedy, in-session)"
                )

    def _record_cost(self, job_id: str, job: _RiverJob) -> None:
        """Append a ledger line.

        River bills per token, and the SDK does not publish a price here, so
        ``cost_usd`` is recorded as **None** — unknown, never zero. A zero
        would quietly make ``stateset-agents costs`` under-report and any
        budget check pass. The token count, when the response exposed one, is
        recorded so the spend can be reconstructed from River's price list.
        """
        record_entry(
            CostEntry(
                provider=self.name,
                job_id=job_id,
                base_model=job.spec.base_model,
                gpu="river-managed",
                gpu_count=0,
                cost_per_hr=None,
                duration_s=(
                    round(job.duration_s, 1) if job.duration_s is not None else None
                ),
                cost_usd=None,
                status=job.status.value,
            ),
            path=self.ledger_path,
        )

    # -- executor interface ------------------------------------------------

    def _job(self, handle: JobHandle) -> _RiverJob:
        try:
            return self._jobs[handle.job_id]
        except KeyError:
            raise RemoteExecutionError(
                f"unknown job: {handle.job_id}", provider=self.name
            ) from None

    def status(self, handle: JobHandle) -> JobStatus:
        return self._job(handle).status

    def logs(self, handle: JobHandle) -> Iterator[str]:
        yield from self._job(handle).logs

    def job_cost(self, handle: JobHandle) -> tuple[float | None, float | None]:
        """Wall-clock duration, and an honestly-unknown cost.

        River prices per token; we have no price list, so reporting a dollar
        figure would be inventing one.
        """
        return (self._job(handle).duration_s, None)

    def fetch(self, handle: JobHandle, dest: Path | None = None) -> Path:
        """Write the checkpoint pointer and manifest; return the directory.

        There is nothing to download: the trained LoRA lives on River. What
        lands on disk is the ``river://`` URI, the configuration that produced
        it, and a standard ``stateset_manifest.json`` so provenance and
        ``stateset-agents adapters`` work exactly as they do for local runs.
        """
        job = self._job(handle)
        if job.status is not JobStatus.SUCCEEDED:
            raise RemoteExecutionError(
                f"job {handle.job_id} is not finished successfully; nothing to fetch",
                provider=self.name,
            )
        return self._write_checkpoint_artifacts(job, dest)

    def _write_checkpoint_artifacts(
        self, job: _RiverJob, dest: Path | None = None
    ) -> Path:
        """Publish a committed checkpoint, including during a recoverable RL run."""
        spec = job.spec
        output_dir = Path(dest) if dest is not None else Path(spec.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pointer = {
            "provider": self.name,
            "checkpoint": job.checkpoint_uri,
            "base_model": spec.base_model,
            "lora": {
                "rank": spec.lora_r,
                "train_attn": True,
                "train_mlp": True,
                "train_unembed": False,
            },
            "steps": job.steps,
            "final_loss": job.final_loss,
            "num_epochs": spec.num_epochs,
            "learning_rate": spec.learning_rate,
            "tokens": job.tokens,
            "note": (
                "River hosts the trained weights; this file is a pointer, not "
                "an adapter. Sample through the River API with this "
                "checkpoint — `stateset-agents serve --checkpoint` cannot "
                "load it."
            ),
        }
        if spec.job_kind == "sft":
            pointer["sft_options"] = _sft_options(spec)
        atomic_json(output_dir / CHECKPOINT_POINTER_NAME, pointer)

        from stateset_agents.training.lineage import (
            AdapterManifest,
            hash_dataset,
            write_manifest,
        )

        digest, rows = hash_dataset(Path(spec.dataset))
        write_manifest(
            output_dir,
            AdapterManifest(
                base_model=spec.base_model,
                dataset_path=str(spec.dataset),
                dataset_sha256=digest,
                dataset_rows=rows,
                hyperparameters={
                    "provider": self.name,
                    "lora_r": spec.lora_r,
                    "num_epochs": spec.num_epochs,
                    "learning_rate": spec.learning_rate,
                    "max_length": spec.max_length,
                    "per_device_batch_size": spec.per_device_batch_size,
                    "river_checkpoint": job.checkpoint_uri,
                    "steps": job.steps,
                    **(_sft_options(spec) if spec.job_kind == "sft" else {}),
                },
                parent_adapter=spec.parent_adapter,
                package_version=spec.package_version,
            ),
        )
        return output_dir

    def cancel(self, handle: JobHandle) -> None:
        """Mark a job cancelled.

        River's training loop runs inside ``submit()``, so by the time a
        caller can cancel, the run has already ended. Kept for interface
        parity, and honest about doing nothing to a finished job.
        """
        job = self._job(handle)
        if not job.status.is_terminal:
            job.status = JobStatus.CANCELLED


def _is_set(spec: RemoteJobSpec, field_name: str) -> bool:
    """True when a machine-shaped spec field carries a non-default value."""
    value = getattr(spec, field_name, None)
    if field_name == "gpu_count":
        return value is not None and int(value) != 1
    if field_name == "cloud_type":
        return str(value).upper() not in ("", "SECURE")
    return value is not None


@contextlib.contextmanager
def _open_session(client: Any, project: str | None = None) -> Iterator[Any]:
    """Yield a River training session, whatever the client's vintage.

    The docs' canonical form is ``with client.session(project=...) as s`` —
    tried first, degrading to a plain ``client.session(...)`` return value if
    it is not a context manager, then to the older ``create_session()``, then
    to the client itself (the seam fakes use). Cleanup runs only when the
    session was actually opened as a context manager.
    """
    if hasattr(client, "session"):
        try:
            opened = client.session(project=project) if project else client.session()
        except TypeError:
            opened = client.session()
        if hasattr(opened, "__enter__"):
            with opened as session:
                yield session
            return
        yield opened
        return
    if hasattr(client, "create_session"):
        yield client.create_session()
        return
    yield client


def _inference_checkpoint(client: Any, uri: str | None) -> Any:
    """Wrap a ``river://`` URI as an inference-mode ``Checkpoint``.

    ``create_model`` given a bare path tries to restore optimizer state,
    which an inference-mode save does not carry — observed live:
    "Cannot load optimizer from an inference checkpoint". Typing the
    checkpoint tells the server weights-only is intended.
    """
    if uri is None:
        return None
    river = _river_module(client)
    checkpoint_cls = getattr(river, "Checkpoint", None)
    if checkpoint_cls is None:
        return uri
    return checkpoint_cls(path=uri, step=0, checkpoint_type="inference")


def _checkpoint_from_pointer(
    adapter_dir: str | Path | None,
    *,
    base_model: str | None = None,
    lora_rank: int | None = None,
) -> str | None:
    """Resolve and validate a River URI or local pointer before provider work.

    Legacy pointers without model/LoRA metadata remain supported. When present,
    metadata must agree with the requested model and training rank. This checks
    local claims only; a URI does not prove remote checkpoint compatibility.
    """
    if adapter_dir is None:
        return None

    def checked_uri(value: Any) -> str:
        if (
            not isinstance(value, str)
            or not value.startswith("river://")
            or len(value) <= len("river://")
            or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value)
        ):
            raise RemoteExecutionError(
                "Checkpoint must be a nonempty river:// URI without whitespace or control characters",
                provider="river",
            )
        return value

    if not isinstance(adapter_dir, (str, Path)) or not str(adapter_dir).strip():
        raise RemoteExecutionError(
            "Checkpoint reference must be a River URI or pointer directory",
            provider="river",
        )
    if str(adapter_dir).startswith("river://"):
        return checked_uri(str(adapter_dir))

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate checkpoint pointer field")
            result[key] = value
        return result

    pointer = Path(adapter_dir) / CHECKPOINT_POINTER_NAME
    try:
        data = json.loads(pointer.read_text(), object_pairs_hook=unique_object)
    except (OSError, ValueError) as exc:
        raise RemoteExecutionError(
            f"{adapter_dir!r} is not a River checkpoint pointer directory "
            f"(no readable, unambiguous {CHECKPOINT_POINTER_NAME})",
            provider="river",
        ) from exc
    if not isinstance(data, dict):
        raise RemoteExecutionError(
            "Checkpoint pointer must be an object", provider="river"
        )
    checkpoint = checked_uri(data.get("checkpoint"))
    if "provider" in data and data["provider"] != "river":
        raise RemoteExecutionError(
            "Checkpoint pointer provider is not River", provider="river"
        )
    if "base_model" in data and (
        not isinstance(data["base_model"], str)
        or not data["base_model"].strip()
        or (base_model is not None and data["base_model"] != base_model)
    ):
        raise RemoteExecutionError(
            "Checkpoint pointer base model differs from request", provider="river"
        )
    if "lora" in data:
        lora = data["lora"]
        if (
            not isinstance(lora, dict)
            or type(lora.get("rank")) is not int
            or lora["rank"] < 1
            or (lora_rank is not None and lora["rank"] != lora_rank)
        ):
            raise RemoteExecutionError(
                "Checkpoint pointer LoRA rank differs from request", provider="river"
            )
    return checkpoint


def _rendered_prompts(
    base_model: str, prompts: list[str]
) -> tuple[list[str], list[str] | None]:
    """Chat-template ``prompts`` with the SDK's renderer, thinking off.

    River's ``model.sample`` takes raw text — the caller renders the chat
    template. The SDK ships per-family renderers for exactly this; thinking
    is disabled so the token budget goes to the answer (the Nemotron
    lesson). Returns (rendered prompts, stop strings). Falls back to the
    raw text when no renderer knows the model.
    """
    try:
        from river_client.renderers import get_renderer

        renderer = get_renderer(base_model, thinking=False)
        rendered = [
            renderer.build_sample_prompt([{"role": "user", "content": p}]).prompt
            for p in prompts
        ]
        stops = list(renderer.get_stop_strings())
        return rendered, stops or None
    except Exception as exc:  # noqa: BLE001 - renderer coverage varies by model
        logger.warning("no River renderer for %s (%s); sampling raw", base_model, exc)
        return list(prompts), None


def _rl_sample_groups(
    model: Any,
    base_model: str,
    prompts: list[str],
    *,
    num_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int = 0,
) -> tuple[list[list[Any]], list[list[int]]]:
    """Sample groups for RL with CLIENT-side prompt token ids.

    River does not echo ``prompt_token_ids`` for text-prompt sampling
    (observed live), and the RL datum layout needs the prompt ids exactly.
    So the renderer's own tokenizer encodes the rendered prompt and the ids
    are PASSED to ``sample`` — the server then generates from precisely the
    ids the datums will carry.
    """
    try:
        from river_client.renderers import get_renderer

        renderer = get_renderer(base_model, thinking=False)
    except Exception as exc:  # noqa: BLE001 - fakes/unknown models land here
        logger.warning(
            "no River renderer for %s (%s); sampling raw text and relying "
            "on echoed prompt ids",
            base_model,
            exc,
        )
        groups = model.sample(
            prompts,
            num_samples=num_samples,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            top_k=-1,
        )
        prompt_ids = [
            [int(t) for t in (getattr(g[0], "prompt_token_ids", None) or [])]
            for g in groups
        ]
        if any(not ids for ids in prompt_ids):
            raise RemoteExecutionError(
                "no renderer available and River echoed no prompt_token_ids; "
                "the RL datum layout needs the prompt ids exactly",
                provider="river",
            ) from exc
        return groups, prompt_ids

    rendered = [
        renderer.build_sample_prompt([{"role": "user", "content": p}]).prompt
        for p in prompts
    ]
    prompt_ids = [
        [int(t) for t in renderer.tokenizer.encode(text)] for text in rendered
    ]
    groups = model.sample(
        prompt_token_ids=prompt_ids,
        num_samples=num_samples,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=list(renderer.get_stop_strings()) or None,
        seed=seed,
        top_k=-1,
    )
    return groups, prompt_ids


def _sample_texts(
    model: Any,
    base_model: str,
    prompts: list[str],
    *,
    num_samples: int,
    temperature: float,
    top_p: float = 1.0,
    max_tokens: int = 300,
) -> list[list[str]]:
    """``model.sample`` for chat prompts -> texts, one list per prompt.

    Prompts are chat-templated client-side via the SDK's renderers (raw
    ``model_input`` message dicts are a multimodal parts format, not chat —
    observed live: "must be a dict with a 'type' field"). The return shape
    is ``list[list[Sample]]`` whose entries expose ``.text``.
    """
    rendered, stops = _rendered_prompts(base_model, prompts)
    groups = model.sample(
        rendered,
        num_samples=num_samples,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stops,
    )
    return [[str(getattr(s, "text", s)) for s in group] for group in groups]


def _river_module(client: Any) -> Any:
    """Where ``LoraConfig`` lives.

    River's docs construct it as ``river.LoraConfig(...)`` from the top-level
    module. A fake client may expose it directly instead, which is the seam
    tests use.
    """
    if hasattr(client, "LoraConfig"):
        return client
    try:
        import river_client

        return river_client
    except ImportError as exc:  # pragma: no cover - unreachable with a real client
        raise RemoteExecutionError.wrap(
            exc, "river-client is required to build a LoraConfig", provider="river"
        ) from exc


def _as_uri(saved: Any) -> str:
    """Normalize whatever ``save_weights`` returns to a ``river://`` string.

    The docs show a path being returned; whether that is a bare string or a
    small object is not stated, so both are accepted.
    """
    if isinstance(saved, str):
        return saved
    for name in ("path", "uri", "checkpoint", "url", "name"):
        value = getattr(saved, name, None)
        if isinstance(value, str):
            return value
    if isinstance(saved, dict):
        for name in ("path", "uri", "checkpoint", "url"):
            if isinstance(saved.get(name), str):
                return str(saved[name])
    return str(saved)


def _extract(result: Any, *names: str) -> Any:
    """Pull the first present field out of an SDK response.

    River's response object shape is undocumented enough that we accept both
    attribute and mapping access, and tolerate its absence — a missing loss
    must not fail a training run that otherwise succeeded.
    """
    for name in names:
        if isinstance(result, dict) and name in result:
            return result[name]
        value = getattr(result, name, None)
        if value is not None:
            return value
    # ForwardResult keeps its scalars in a ``metrics`` mapping — observed
    # live: step ticks printed without loss because loss_mean lives there.
    metrics = getattr(result, "metrics", None)
    if isinstance(metrics, dict):
        for name in names:
            if name in metrics:
                return metrics[name]
    return None


def _graded_episode_reward(script: dict[str, Any], replies: list[str]) -> float:
    """Shaped episode reward: token fraction + completeness bonus − violation.

    Mirrors the single-turn shaping (which Goodharted without the bonus):
    fraction of ALL turn_expect tokens satisfied in their own turns, +1.0
    when the whole episode passes, −1.0 when any forbid appears anywhere.
    """
    from stateset_agents.training.sft import evaluate_checks

    total = 0
    hit = 0
    for expects, reply in zip(script.get("turn_expect", []), replies, strict=True):
        checked = evaluate_checks(reply, list(expects), [])
        total += len(expects)
        hit += len(checked["expect_hits"])
    frac = hit / total if total else 1.0
    passed, detail = _score_episode(script, replies)
    bonus = 1.0 if passed else 0.0
    penalty = 1.0 if detail["forbid_hits"] else 0.0
    return frac + bonus - penalty


def _episode_rl_datums(
    branch_turns: list[dict[str, Any]], advantage: float
) -> list[dict[str, Any]]:
    """One branch's captured turns -> RL datums with a shared advantage.

    Multi-turn credit assignment, v1: the episode-level group-relative
    advantage is broadcast to every assistant turn's response positions —
    every action in a winning conversation is reinforced, every action in
    a losing one pushed away. Layout per datum is River's pre-shifted RL
    contract (see ``build_group_rl_datums``).
    """
    datums: list[dict[str, Any]] = []
    for turn in branch_turns:
        validate_record(turn)
        prompt_ids = turn.get("prompt_ids") or []
        tokens = turn.get("tokens") or []
        logprobs = turn.get("logprobs") or []
        pad = [0.0] * (len(prompt_ids) - 1)
        datums.append(
            {
                "input_ids": list(prompt_ids) + list(tokens),
                "old_logprobs": pad + list(logprobs) + [0.0],
                "advantages": pad + [advantage] * len(tokens) + [0.0],
                "attention_mask": [1] * (len(prompt_ids) + len(tokens)),
            }
        )
    return datums


def check_tool_call(reply: str, expected: dict[str, Any]) -> bool:
    """Deterministically verify a structured action in ``reply``.

    The reply must contain a fenced ```json block whose object names the
    expected ``tool`` and includes every expected ``args`` key with an
    exactly-equal value (extra args are allowed — the check is a subset
    match). No judges, no substrings: the action parses or it does not.
    """
    import re

    blocks = re.findall(r"```json\s*(.*?)```", reply, flags=re.DOTALL)
    for block in blocks:
        try:
            data = json.loads(block)
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        if data.get("tool") != expected.get("tool"):
            continue
        want = expected.get("args") or {}
        have = data.get("args") or {}
        if all(have.get(k) == v for k, v in want.items()):
            return True
    return False


def _tool_blocks_are_clean(reply: str, known_tools: list[str]) -> bool:
    """No junk actions anywhere: every json block parses and names a known tool.

    A turn with no ``turn_tool`` requirement is UNCHECKED, not unconstrained —
    and the flywheel will happily harvest whatever it emits there. Observed
    live: unchecked turns produced invented tool names ("suppress_dispatch",
    "summarize_account") and two concatenated objects in one block; 113
    such episodes entered training and the trained model stopped emitting
    valid actions entirely under greedy decoding. The harvest must reject
    junk in dimensions the per-turn checks do not cover.
    """
    import re

    for block in re.findall(r"```json\s*(.*?)```", reply, flags=re.DOTALL):
        try:
            data = json.loads(block)
        except ValueError:
            return False  # malformed (e.g. two objects concatenated)
        if not isinstance(data, dict):
            return False
        if data.get("tool") not in known_tools:
            return False  # invented tool name
    return True


def _score_episode(
    script: dict[str, Any], assistant_turns: list[str]
) -> tuple[bool, dict[str, Any]]:
    """Episode-level pass: every turn's checks hit, no forbid anywhere.

    ``turn_expect[i]`` is asserted against assistant turn ``i`` alone (a
    reference echoed only in turn 1 must not satisfy turn 2 — carryover is
    the skill under test); ``forbid`` is asserted over the whole episode.
    """
    from stateset_agents.training.sft import evaluate_checks

    per_turn = []
    passed = True
    tool_specs = script.get("turn_tool") or [None] * len(assistant_turns)
    for expects, tool, reply in zip(
        script.get("turn_expect", []), tool_specs, assistant_turns, strict=True
    ):
        checked = evaluate_checks(reply, list(expects), [])
        if tool is not None:
            checked = dict(checked)
            checked["tool_ok"] = check_tool_call(reply, tool)
            if not checked["tool_ok"]:
                checked["passed"] = False
        per_turn.append(checked)
        if not checked["passed"]:
            passed = False
    whole = " ".join(assistant_turns)
    forbid_checked = evaluate_checks(whole, [], list(script.get("forbid", [])))
    if not forbid_checked["passed"]:
        passed = False
    known_tools = script.get("known_tools")
    junk_tools = False
    if known_tools:
        junk_tools = not all(
            _tool_blocks_are_clean(reply, list(known_tools))
            for reply in assistant_turns
        )
        if junk_tools:
            passed = False
    return passed, {
        "per_turn": per_turn,
        "forbid_hits": forbid_checked["forbid_hits"],
        "junk_tools": junk_tools,
        "passed": passed,
    }


def _rollout_episodes(
    model: Any,
    base_model: str,
    scripts: list[dict[str, Any]],
    *,
    branches: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    capture: bool = False,
    seed: int | None = None,
) -> list[list[list[Any]]]:
    """Roll ``branches`` independent episodes per script, batched per turn.

    Returns ``episodes[script][branch] = [assistant_turn_1, ...]`` — plain
    reply strings by default. With ``capture=True`` each turn is instead a
    dict ``{"text", "prompt_ids", "tokens", "logprobs"}`` — everything
    multi-turn RL needs, with prompt ids tokenized client-side and PASSED
    to the sampler so generation used exactly the ids the datums carry.

    Each branch keeps its own conversation history; at every turn all live
    branches across all scripts are sampled in ONE ``model.sample`` call
    (one prompt each), so a T-turn rollout costs T calls, not S*N*T.
    """
    from river_client.renderers import get_renderer

    renderer = get_renderer(base_model, thinking=False)
    stops = list(renderer.get_stop_strings()) or None
    tokenizer = getattr(renderer, "tokenizer", None)
    max_turns = max(len(s["turns"]) for s in scripts)
    # histories[script][branch] = message list
    histories: list[list[list[dict[str, str]]]] = [
        [[] for _ in range(branches)] for _ in scripts
    ]
    replies: list[list[list[Any]]] = [[[] for _ in range(branches)] for _ in scripts]
    for turn in range(max_turns):
        flat: list[tuple[int, int]] = []
        prompts: list[str] = []
        for si, script in enumerate(scripts):
            if turn >= len(script["turns"]):
                continue
            for bi in range(branches):
                histories[si][bi].append(
                    {"role": "user", "content": script["turns"][turn]}
                )
                prompts.append(renderer.build_sample_prompt(histories[si][bi]).prompt)
                flat.append((si, bi))
        if not prompts:
            break
        if capture and tokenizer is not None:
            prompt_ids = [[int(t) for t in tokenizer.encode(text)] for text in prompts]
            groups = model.sample(
                prompt_token_ids=prompt_ids,
                num_samples=1,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stops,
                **({"seed": seed + turn, "top_k": -1} if seed is not None else {}),
            )
        else:
            prompt_ids = [[] for _ in prompts]
            groups = model.sample(
                prompts,
                num_samples=1,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stops,
                **({"seed": seed + turn, "top_k": -1} if seed is not None else {}),
            )
        for (si, bi), group, ids in zip(flat, groups, prompt_ids, strict=True):
            if len(group) != 1:
                raise ValueError("River returned an incomplete episode sample group")
            sample = group[0]
            text = str(getattr(sample, "text", "")).strip()
            histories[si][bi].append({"role": "assistant", "content": text})
            if capture:
                replies[si][bi].append(
                    sample_record(
                        sample,
                        ids or list(getattr(sample, "prompt_token_ids", None) or []),
                    )
                )
            else:
                replies[si][bi].append(text)
    return replies
