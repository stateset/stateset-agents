"""The executor contract every compute provider implements.

Deliberately poll-based: a job is submitted, polled, and its artifacts
fetched. State durability and retry policy are provider-specific: Fireworks
owns a durable asynchronous job, while machine executors keep local metadata
and RunPod may retry an interrupted pod within the caller's budget.
"""

from __future__ import annotations

import abc
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from stateset_agents.core.errors import ErrorCode, StateSetError, wrap_exception
from stateset_agents.remote.job import (
    JobHandle,
    JobStatus,
    RemoteJobResult,
    RemoteJobSpec,
)

__all__ = ["RemoteExecutionError", "RemoteExecutor", "SubmissionReceiptError"]


class RemoteExecutionError(StateSetError):
    """Raised when a remote job cannot be submitted, polled, or retrieved."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.NET_API_ERROR,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, code=code)
        if kwargs:
            self.context.details.update(kwargs)

    @classmethod
    def wrap(
        cls, exc: BaseException, message: str, **context: Any
    ) -> RemoteExecutionError:
        """Wrap a provider SDK exception, preserving the ``.cause`` chain."""
        wrapped = wrap_exception(
            exc, cls, message=message, code=ErrorCode.NET_API_ERROR, **context
        )
        assert isinstance(wrapped, cls)  # wrap_exception constructs `cls`
        return wrapped


class SubmissionReceiptError(RemoteExecutionError):
    """A durable allocation observation could not be retained; do not retry."""


class RemoteExecutor(abc.ABC):
    """Runs a :class:`RemoteJobSpec` on some compute provider."""

    #: Provider name, as used by the registry and stamped onto handles.
    name: str = "unknown"

    #: Job modes this provider can execute without changing their meaning.
    #: SFT is the safe default for third-party executors; providers opt into
    #: harvesting and remote-autograd RL explicitly.
    supported_job_kinds: frozenset[str] = frozenset({"sft"})
    durable_handles: bool = False
    managed_deployments: bool = False
    result_kind: str = "local_artifacts"
    compute_model: str = "provider-specific"
    verification_status: str = "unit-tested"
    #: Opt in only when explicit SFT seeds reach initialization and training.
    supports_sft_seed: bool = False
    #: Set when SFT invokes StateSet's packaged Transformers worker. Managed
    #: provider runtimes keep their own exact-model capability checks.
    uses_packaged_training: bool = False
    #: Allocation observations can be durably retained while submit is pending.
    #: This does not imply restartable training or process-independent fetch.
    durable_submission_receipts: bool = False
    _submission_observer_failed: bool = False

    def set_submission_observer(
        self, observer: Callable[[dict[str, Any]], None] | None
    ) -> None:
        """Observe allocation/cleanup synchronously; persistence errors fail closed.

        Only controlled provider identities and lifecycle fields are supplied.
        The observer must retain an event before returning. No observer keeps
        the provider's existing synchronous submission behavior unchanged.
        """
        if observer is not None and not callable(observer):
            raise TypeError("Submission observer must be callable or None")
        self._submission_observer = observer
        self._submission_observer_failed = False

    def _observe_submission(self, event: dict[str, Any]) -> None:
        observer = getattr(self, "_submission_observer", None)
        if observer is None:
            return
        try:
            observer({**event, "schema_version": 1, "provider": self.name})
        except Exception as exc:
            self._submission_observer_failed = True
            raise SubmissionReceiptError(
                "Could not retain provider submission evidence; inspect retained resources before retrying",
                provider=self.name,
                stage="submission_receipt",
            ) from exc

    def supports(self, job_kind: str) -> bool:
        """Whether this provider preserves the requested job mode's meaning."""
        return job_kind.strip().lower() in self.supported_job_kinds

    def capabilities(self) -> dict[str, Any]:
        """Machine-readable provider features for CLIs and orchestrators."""
        return {
            "provider": self.name,
            "job_kinds": sorted(self.supported_job_kinds),
            "durable_handles": self.durable_handles,
            "managed_deployments": self.managed_deployments,
            "result_kind": self.result_kind,
            "compute_model": self.compute_model,
            "verification_status": self.verification_status,
            "sft_seed": self.supports_sft_seed,
            "uses_packaged_training": self.uses_packaged_training,
            "durable_submission_receipts": self.durable_submission_receipts,
        }

    def validate_spec(self, spec: RemoteJobSpec) -> None:
        """Check job mode and native loader readiness before allocating compute."""
        # Specs are mutable. Dispatch uses canonical job kinds, so accepting a
        # normalized variant here could send a harvest request into SFT.
        if (
            not isinstance(spec.job_kind, str)
            or spec.job_kind != spec.job_kind.strip().lower()
        ):
            raise RemoteExecutionError(
                "job_kind must be a canonical mode; construct a new RemoteJobSpec",
                code=ErrorCode.CFG_INVALID,
                provider=self.name,
            )
        if not isinstance(spec.base_model, str) or not spec.base_model.strip():
            raise RemoteExecutionError(
                "base_model must be a non-empty model name",
                code=ErrorCode.CFG_INVALID,
                provider=self.name,
            )
        if spec.seed is not None and not self.supports_sft_seed:
            raise RemoteExecutionError(
                f"provider {self.name!r} does not support an explicit SFT seed",
                provider=self.name,
            )
        if not self.supports(spec.job_kind):
            supported = ", ".join(sorted(self.supported_job_kinds))
            raise RemoteExecutionError(
                f"provider {self.name!r} does not support job_kind "
                f"{spec.job_kind!r}; supported: {supported}",
                provider=self.name,
                job_kind=spec.job_kind,
            )

        # A local training source must be checked before any SDK setup, upload,
        # or paid resource allocation. Recheck staged bytes in packaged workers.
        # Missing files retain each provider's existing path diagnostics.
        if spec.job_kind == "sft" and Path(spec.dataset).is_file():
            from stateset_agents.data.finetuning import validate_finetuning_source_split

            try:
                validate_finetuning_source_split(spec.dataset)
            except (ValueError, OSError) as exc:
                raise RemoteExecutionError(
                    f"Invalid SFT training source: {exc}",
                    code=ErrorCode.CFG_INVALID,
                    provider=self.name,
                ) from exc

        if self.uses_packaged_training and spec.job_kind == "sft":
            from stateset_agents.core.frontier_models import (
                get_frontier_model,
                require_frontier_training,
            )

            try:
                frontier = get_frontier_model(spec.base_model.strip())
            except ValueError:
                return
            try:
                require_frontier_training(frontier, generic_loader=True)
            except RuntimeError as exc:
                raise RemoteExecutionError(
                    str(exc),
                    code=ErrorCode.CFG_INVALID,
                    provider=self.name,
                    model=frontier.model_id,
                    model_revision=frontier.revision,
                    blockers=list(frontier.blockers) or [str(exc)],
                ) from exc

    @abc.abstractmethod
    def submit(self, spec: RemoteJobSpec) -> JobHandle:
        """Start the job and return a handle for polling it."""

    @abc.abstractmethod
    def status(self, handle: JobHandle) -> JobStatus:
        """Return the job's current lifecycle state."""

    @abc.abstractmethod
    def logs(self, handle: JobHandle) -> Iterator[str]:
        """Yield the job's output lines captured so far."""

    @abc.abstractmethod
    def fetch(self, handle: JobHandle, dest: Path | None = None) -> Path:
        """Retrieve the trained adapter directory. Errors if not finished."""

    @abc.abstractmethod
    def cancel(self, handle: JobHandle) -> None:
        """Stop the job if it is still running."""

    def wait(self, handle: JobHandle, poll_interval_s: float = 1.0) -> RemoteJobResult:
        """Poll until the job reaches a terminal state, then collect it.

        Shared by every provider — the per-provider work is only in the four
        primitives above.
        """
        while not (status := self.status(handle)).is_terminal:
            time.sleep(poll_interval_s)

        logs = list(self.logs(handle))
        # Fetch on failure too, best-effort. The eval gate deliberately fails
        # a job AFTER saving the adapter and eval_results.json (a failed
        # assertion must not destroy what was paid for) — but fetching only
        # on success discarded exactly those artifacts with the pod. Observed
        # live: a 10/12 assertion run's adapter was lost and had to be
        # retrained. A failed job with nothing to fetch still returns None.
        if status is JobStatus.SUCCEEDED:
            output_dir = self.fetch(handle)
        else:
            try:
                output_dir = self.fetch(handle)
            except Exception:  # noqa: BLE001 - nothing to fetch is fine
                output_dir = None
        # Providers that meter their own pods expose the measured spend via
        # `job_cost`; those that do not simply report None.
        duration_s, cost_usd = self.job_cost(handle)
        return RemoteJobResult(
            handle=handle,
            status=status,
            output_dir=output_dir,
            logs=logs,
            duration_s=duration_s,
            cost_usd=cost_usd,
        )

    def job_cost(self, handle: JobHandle) -> tuple[float | None, float | None]:
        """Measured (duration_s, cost_usd) for a finished job.

        Default: unknown. Providers that rent metered hardware override it.
        """
        return (None, None)

    def undeploy(self, deployment: str) -> None:
        """Tear down a managed deployment.

        Default: unsupported. Providers that manage deployments (e.g.
        Fireworks on-demand serving) override this.
        """
        raise NotImplementedError(f"{type(self).__name__} does not manage deployments")
