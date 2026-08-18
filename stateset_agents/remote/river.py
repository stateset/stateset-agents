"""Run the fine-tune on River AI's remote autograd service.

River (https://docs.river.ai) is unlike the other providers here. Modal and
RunPod rent you a machine and we ship ``stateset_agents.training.sft`` to it.
River rents you *gradients*: the model lives on their infrastructure and you
drive the training loop yourself, call by call —
``create_model`` -> N x (``forward_backward`` -> ``optim_step``) ->
``save_weights``. So this executor does not shell out to the training script;
it *is* the training loop, with the tensor math happening elsewhere.

.. warning::

   **NOT LIVE-VERIFIED.** This adapter was written against River's published
   documentation. No River API key was available and ``river-client`` could
   not be installed, so every call sequence, keyword name, and response shape
   below is an informed guess. It is exercised only against fakes. Treat the
   first real run as an integration test, and see ``docs/RIVER_PROVIDER.md``
   for the specific assumptions most likely to bite.

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
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stateset_agents.remote.executor import RemoteExecutionError, RemoteExecutor
from stateset_agents.remote.job import JobHandle, JobStatus, RemoteJobSpec
from stateset_agents.remote.ledger import CostEntry, record_entry
from stateset_agents.remote.river_batches import (
    build_sft_batch,
    validate_base_model,
    validate_lora_rank,
)

__all__ = ["CHECKPOINT_POINTER_NAME", "RiverExecutor"]

#: Environment variable holding the River API key (``rv_...``).
RIVER_API_KEY_ENV = "RIVER_API_KEY"

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


@dataclass
class _RiverJob:
    """Bookkeeping for one River training run."""

    spec: RemoteJobSpec
    status: JobStatus
    logs: list[str] = field(default_factory=list)
    checkpoint_uri: str | None = None
    steps: int = 0
    final_loss: float | None = None
    tokens: int | None = None
    duration_s: float | None = None
    cost_usd: float | None = None


class RiverExecutor(RemoteExecutor):
    """Drives River's remote training loop for one :class:`RemoteJobSpec`.

    ``client`` is the seam: pass any object implementing River's surface and
    nothing here touches the network or the SDK. Left as ``None``, the real
    ``river_client`` is imported lazily at submit time, so merely listing
    providers never requires the SDK.
    """

    name = "river"

    #: River's default loss for supervised data.
    SFT_LOSS_FN = "cross_entropy"

    def __init__(
        self,
        client: Any = None,
        *,
        tokenizer: Any = None,
        ledger_path: Path | None = None,
    ) -> None:
        self._client = client
        self._tokenizer = tokenizer
        self.ledger_path = ledger_path
        self._jobs: dict[str, _RiverJob] = {}
        self._counter = 0

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
            self._tokenizer = AutoTokenizer.from_pretrained(base_model)  # nosec: B615
        except Exception as exc:  # noqa: BLE001 - hub/network/auth all land here
            raise RemoteExecutionError.wrap(
                exc,
                f"could not load a tokenizer for {base_model!r}",
                provider=self.name,
            ) from exc
        return self._tokenizer

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
        self._counter += 1
        job_id = f"river-{self._counter}"
        handle = JobHandle(provider=self.name, job_id=job_id)
        logs: list[str] = []
        job = _RiverJob(spec=spec, status=JobStatus.PENDING, logs=logs)
        self._jobs[job_id] = job

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

        client = self._get_client()
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
        import time as _time

        from stateset_agents.training.harvest import build_harvest_rows
        from stateset_agents.training.sft import normalize_eval_prompts

        logs = job.logs
        prompts = json.loads(Path(spec.dataset).read_text())
        harvest_specs = normalize_eval_prompts(prompts)
        eval_specs = normalize_eval_prompts(list(spec.eval_prompts or []))
        knobs = spec.harvest or {}
        checkpoint = _checkpoint_from_pointer(knobs.get("adapter_dir"))
        output_dir = Path(spec.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary: dict[str, Any] = {
            "base_model": spec.base_model,
            "adapter_dir": knobs.get("adapter_dir"),
            "best_of": knobs.get("best_of", 8),
            "temperature": knobs.get("temperature", 0.9),
            "prompts": len(harvest_specs),
            "samples": 0,
            "kept": 0,
            "prompts_with_a_pass": 0,
            "eval": None,
            "dry_run": bool(spec.dry_run),
        }
        if spec.dry_run:
            (output_dir / "harvest_summary.json").write_text(
                json.dumps(summary, indent=2)
            )
            logs.append("dry run: harvest plan only, nothing sampled")
            job.status = JobStatus.SUCCEEDED
            return handle

        client = self._get_client()
        started = _time.monotonic()
        job.status = JobStatus.RUNNING
        try:
            with _open_session(
                client, project=Path(spec.output_dir).name or None
            ) as session:
                model = session.create_model(
                    base_model=spec.base_model,
                    checkpoint=_inference_checkpoint(client, checkpoint),
                )
                logs.append(f"sampling from {checkpoint or spec.base_model} via River")
                if eval_specs:
                    greedy = _sample_texts(
                        model,
                        [s["prompt"] for s in eval_specs],
                        num_samples=1,
                        temperature=0.0,
                        max_tokens=spec.eval_max_new_tokens,
                    )
                    from stateset_agents.training.sft import evaluate_checks

                    results = []
                    for espec, completions in zip(eval_specs, greedy):
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
                    summary["eval"] = {
                        "passed": sum(1 for r in results if r["passed"]),
                        "total": len(results),
                        "results": results,
                    }
                sampled = _sample_texts(
                    model,
                    [s["prompt"] for s in harvest_specs],
                    num_samples=int(knobs.get("best_of", 8)),
                    temperature=float(knobs.get("temperature", 0.9)),
                    top_p=float(knobs.get("top_p", 0.95)),
                    max_tokens=int(knobs.get("max_new_tokens", 300)),
                )
                rows: list[dict[str, Any]] = []
                for hspec, completions in zip(harvest_specs, sampled):
                    summary["samples"] += len(completions)
                    kept = build_harvest_rows(hspec, completions)
                    if kept:
                        summary["prompts_with_a_pass"] += 1
                    rows.extend(kept)
        except Exception as exc:  # noqa: BLE001 - unknown SDK exception surface
            job.status = JobStatus.FAILED
            job.duration_s = _time.monotonic() - started
            logs.append(f"River harvest failed: {exc}")
            self._record_cost(handle.job_id, job)
            account = self._account_error(exc)
            if account is not None:
                raise account from exc
            raise RemoteExecutionError.wrap(
                exc, "River harvest failed", provider=self.name
            ) from exc

        summary["kept"] = len(rows)
        with (output_dir / "harvest.jsonl").open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        (output_dir / "harvest_summary.json").write_text(json.dumps(summary, indent=2))
        logs.append(
            f"harvest: kept {summary['kept']}/{summary['samples']} across "
            f"{summary['prompts_with_a_pass']}/{summary['prompts']} prompts"
        )
        job.duration_s = _time.monotonic() - started
        job.status = JobStatus.SUCCEEDED
        self._record_cost(handle.job_id, job)
        return handle

    #: Transient-retry policy, from the SDK's own recovery taxonomy:
    #: RiverConnectionError (which covers SessionHeartbeatError and capacity
    #: squeezes) and RiverTimeoutError mean "back off, rebuild the session,
    #: try again"; auth/model/data errors mean "fail fast". Observed live on
    #: the first real run: a slow create_model timed out client-side, the
    #: retry raced the server-side create, and ALREADY_EXISTS arrived as a
    #: RiverConnectionError.
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
        lora = river.LoraConfig(
            rank=spec.lora_r,
            train_attn=True,
            train_mlp=True,
            train_unembed=False,
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
                for start in range(0, len(data), size):
                    chunk = data[start : start + size]
                    if hasattr(model, "train_step"):
                        # The SDK's preferred complete step: forward+backward
                        # and the optimizer step pipelined server-side.
                        result, _ = model.train_step(
                            chunk, lr=spec.learning_rate, loss_fn=self.SFT_LOSS_FN
                        )
                    else:
                        result = model.forward_backward(chunk, loss_fn=self.SFT_LOSS_FN)
                        model.optim_step(lr=spec.learning_rate)
                    job.steps += 1
                    loss = _extract(result, "loss_mean", "loss")
                    if loss is not None:
                        job.final_loss = float(loss)
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

                specs_ = normalize_eval_prompts(list(spec.eval_prompts))
                greedy = _sample_texts(
                    model,
                    [e["prompt"] for e in specs_],
                    num_samples=1,
                    temperature=0.0,
                    max_tokens=spec.eval_max_new_tokens,
                )
                eval_rows = []
                for espec, completions in zip(specs_, greedy):
                    checked = evaluate_checks(
                        completions[0],
                        espec.get("expect", []),
                        espec.get("forbid", []),
                    )
                    eval_rows.append(
                        {
                            "prompt": espec["prompt"],
                            "finetuned": completions[0],
                            "checks": checked,
                        }
                    )
                out_dir = Path(spec.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "eval_results.json").write_text(
                    json.dumps(eval_rows, indent=2)
                )
                passed = sum(1 for r in eval_rows if r["checks"]["passed"])
                logs.append(
                    f"eval: {passed}/{len(eval_rows)} prompt(s) passed "
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
        (output_dir / CHECKPOINT_POINTER_NAME).write_text(
            json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

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


def _checkpoint_from_pointer(adapter_dir: str | None) -> str | None:
    """Resolve a flywheel adapter reference to a ``river://`` URI.

    Between generations the flywheel passes the previous training job's
    output directory — which for River is a POINTER directory holding
    ``river_checkpoint.json``, not weights. A bare ``river://`` string and
    ``None`` (start from base) pass through unchanged.
    """
    if not adapter_dir:
        return None
    if str(adapter_dir).startswith("river://"):
        return str(adapter_dir)
    pointer = Path(adapter_dir) / CHECKPOINT_POINTER_NAME
    try:
        data = json.loads(pointer.read_text())
    except (OSError, ValueError) as exc:
        raise RemoteExecutionError(
            f"{adapter_dir!r} is not a River checkpoint pointer directory "
            f"(no readable {CHECKPOINT_POINTER_NAME}): {exc}",
            provider="river",
        ) from exc
    checkpoint = data.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise RemoteExecutionError(
            f"{pointer} has no 'checkpoint' field", provider="river"
        )
    return checkpoint


def _sample_texts(
    model: Any,
    prompts: list[str],
    *,
    num_samples: int,
    temperature: float,
    top_p: float = 1.0,
    max_tokens: int = 300,
) -> list[list[str]]:
    """``model.sample`` for chat prompts -> texts, one list per prompt.

    ``model_input`` carries message lists so River renders the model's own
    chat template server-side; the return shape is ``list[list[Sample]]``
    (per prompt, then per sample) whose entries expose ``.text``.
    """
    groups = model.sample(
        model_input=[[{"role": "user", "content": p}] for p in prompts],
        num_samples=num_samples,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
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
    return None
