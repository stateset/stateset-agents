import asyncio
import copy
import json
import logging
import uuid
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.core.reward import CompositeReward, HelpfulnessReward, SafetyReward
from stateset_agents.utils.async_calls import drain_owned_operation, run_sync_owned
from stateset_agents.utils.clock import utc_now

from ..errors import ConflictError, ResourceExhaustedError, ServiceUnavailableError
from ..schemas import TrainingRequest
from ..training_queue import (
    MAX_PENDING_JOBS,
    TrainingIdempotencyConflictError,
    TrainingQueue,
    TrainingQueueFullError,
)

logger = logging.getLogger(__name__)

TERMINAL_TRAINING_STATUSES = frozenset({"completed", "failed", "cancelled"})


class JobProgressCallback:
    """Training callback that updates a job dict with episode progress.

    Also monitors a ``cancel_event`` and raises ``asyncio.CancelledError``
    when cancellation is requested so the trainer loop exits cleanly.
    """

    fail_on_error = True

    def __init__(
        self,
        job: dict[str, Any],
        total_episodes: int,
        cancel_event: asyncio.Event,
    ) -> None:
        if type(total_episodes) is not int or total_episodes < 1:
            raise ValueError("total_episodes must be a positive integer")
        self.job = job
        self.total_episodes = total_episodes
        self.cancel_event = cancel_event
        self._last_episode = -1
        self._failure: str | None = None

    @property
    def completion_scope(self) -> str:
        """Describe observed episode progress, without inferring optimizer work."""
        if self._last_episode == self.total_episodes - 1:
            return "final_episode_reported"
        if self._last_episode >= 0:
            return "partial_episode_progress"
        return "no_episode_progress"

    def raise_if_failed(self) -> None:
        """Keep an invalid observation fatal even if a trainer swallowed it."""
        if self._failure is not None:
            raise ValueError(self._failure)

    def on_episode_end(self, episode: int, metrics: dict[str, Any]) -> None:
        """Publish a valid, monotonic progress snapshot after an episode."""
        if self.job.get("status") in TERMINAL_TRAINING_STATUSES:
            return
        self.raise_if_failed()
        try:
            if type(episode) is not int or not 0 <= episode < self.total_episodes:
                raise ValueError(
                    "Training episode must be an integer within the planned run"
                )
            if episode <= self._last_episode:
                raise ValueError(
                    "Training episode progress must increase monotonically"
                )
            if not isinstance(metrics, dict):
                raise ValueError("Training metrics must be a dictionary")
            # Preserve scalar strings and booleans for API diagnostics. Tensor
            # and structured values remain omitted as in the existing API.
            snapshot = {
                k: v
                for k, v in metrics.items()
                if isinstance(v, (int, float, str, bool))
            }
            if any(not isinstance(k, str) or not k.strip() for k in snapshot):
                raise ValueError("Training metric names must be nonempty strings")
            # Match strict JSON/UTF-8 response serialization before mutating the
            # public job. This rejects nonfinite floats and invalid text too.
            json.dumps(snapshot, allow_nan=False, ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            self._failure = f"Invalid training progress: {exc}"
            raise ValueError(self._failure) from exc
        self.job.update(
            current_episode=episode + 1,
            progress=((episode + 1) / self.total_episodes) * 100.0,
            metrics=snapshot,
        )
        self._last_episode = episode

        if self.cancel_event.is_set():
            raise asyncio.CancelledError("Training cancelled by user")


class TrainingService:
    """Service for managing training jobs."""

    def __init__(
        self,
        *,
        max_concurrent_jobs: int = 1,
        queue_path: str | Path | None = None,
        output_root: str | Path | None = None,
    ) -> None:
        if type(max_concurrent_jobs) is not int or max_concurrent_jobs < 1:
            raise ValueError("max_concurrent_jobs must be a positive integer")
        self._max_concurrent_jobs = max_concurrent_jobs
        self._output_root = output_root
        self.training_jobs: dict[str, dict[str, Any]] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._pending_submissions: set[asyncio.Task[str]] = set()
        self._closing = False
        self.queue = (
            None
            if queue_path is None
            else TrainingQueue(queue_path, max_concurrent_jobs=max_concurrent_jobs)
        )

    @property
    def max_concurrent_jobs(self) -> int:
        """Maximum number of workers this service may own concurrently."""
        return self._max_concurrent_jobs

    @property
    def is_closing(self) -> bool:
        """Whether shutdown has stopped admission of new training jobs."""
        return self._closing

    async def aclose(self) -> None:
        """Drain accepted submissions and cancel embedded workers cooperatively.

        Repeated or concurrent calls are safe. Caller cancellation is propagated
        only after workers exit. Unresponsive trainers can delay shutdown; this
        does not forcibly interrupt threads or remote provider work.
        """
        self._closing = True
        tasks = tuple(self._tasks.values())
        submissions = tuple(self._pending_submissions)
        for training_id in tuple(self._tasks):
            self.cancel_training(training_id)

        async def finish() -> None:
            await asyncio.gather(*tasks, *submissions, return_exceptions=True)

        await drain_owned_operation(finish())

    @property
    def jobs(self) -> dict[str, dict[str, Any]]:
        """Compatibility alias used by some router/tests."""
        return self.queue.jobs() if self.queue is not None else self.training_jobs

    @staticmethod
    def _can_access_job(job: dict[str, Any], user_id: str | None) -> bool:
        if user_id is None:
            return True
        owner = job.get("user_id")
        return owner is None or owner == user_id

    def output_root(self) -> Path:
        """Resolved directory that holds every API job's outputs."""
        root = self._output_root
        if root is None:
            from ..config import get_config

            root = get_config().training_output_root
        return Path(root).expanduser().resolve()

    def _resolve_resume(self, value: str, user_id: str | None, root: Path) -> str:
        """Map ``resume_from_checkpoint`` to a path inside ``root``.

        Authenticated callers may use a visible job ID or a path inside that
        job's directory. Internal administrative callers (``user_id=None``)
        may use any path under the root. Resolved symlinks must stay confined.
        """
        job = self.get_training_status(value, user_id)
        if job is not None:
            directory = job.get("metrics", {}).get("artifact_directory") or job.get(
                "output_dir"
            )
            if not directory:
                raise ValueError(f"Training job {value!r} is not resumable")
            candidate = Path(directory)
        else:
            candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(
                "resume_from_checkpoint must be a training job id or a path "
                "under the server's training output root"
            )
        if user_id is not None:
            parts = resolved.relative_to(root).parts
            if not parts or self.get_training_status(parts[0], user_id) is None:
                raise ValueError("Checkpoint is not inside an accessible training job")
        return str(resolved)

    async def start_training(
        self,
        request: TrainingRequest,
        user_id: str | None = None,
        idempotency_key: str | None = None,
        *,
        roles: list[str] | None = None,
    ) -> str:
        """Start a training job."""
        if self._closing:
            raise ServiceUnavailableError("Training service is shutting down")
        request = request.model_copy(deep=True)
        accepted_config = copy.deepcopy(request.model_dump())
        from ..model_policy import client_model_uses_stub

        use_stub_model = client_model_uses_stub(
            request.agent_config.model_name, roles=roles
        )
        root = self.output_root()
        resume_owner = None if roles is not None and "admin" in roles else user_id
        resume = (
            await run_sync_owned(
                self._resolve_resume, request.resume_from_checkpoint, resume_owner, root
            )
            if request.resume_from_checkpoint
            else None
        )
        # Resume lookup may yield to a database thread. Shutdown must close
        # admission before any submission or embedded worker is registered.
        if self._closing:
            raise ServiceUnavailableError("Training service is shutting down")
        from stateset_agents.training.configuration import resolve_training_config

        # Validate accepted overrides before committing work or allocating a model.
        try:
            config = resolve_training_config(
                profile=request.profile,
                num_episodes=request.num_episodes,
                config_overrides=request.training_config_overrides,
                resume_from_checkpoint=resume,
                objective=None,
                objective_overrides=None,
                extra_config={"model_name": request.agent_config.model_name},
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid training configuration: {exc}") from exc
        config.validate_for_training()
        reward_fn = build_reward(request.reward_config)
        if self.queue is not None:
            # Freeze caller-owned nested data before yielding to the database
            # thread. Shutdown owns this operation until commit/rollback ends;
            # disconnecting the HTTP caller cannot detach an unknown write.
            submission = asyncio.create_task(
                run_sync_owned(
                    self.queue.submit,
                    accepted_config,
                    user_id,
                    idempotency_key,
                    output_root=str(root),
                    resume_from_checkpoint=resume,
                    resume_user_id=resume_owner,
                )
            )
            self._pending_submissions.add(submission)

            async def wait_for_submission() -> str:
                return await submission

            try:
                return await drain_owned_operation(wait_for_submission())
            except TrainingIdempotencyConflictError as exc:
                raise ConflictError(str(exc)) from exc
            except TrainingQueueFullError as exc:
                raise ResourceExhaustedError(
                    "training_queue", MAX_PENDING_JOBS
                ) from exc
            finally:
                self._pending_submissions.discard(submission)
        if idempotency_key is not None:
            raise ValueError("Idempotency keys require API_TRAINING_QUEUE_PATH")
        # Submission contains no await between this check and task registration,
        # so admission is atomic for this service's event loop. Count owned tasks,
        # including cancellation cleanup, rather than mutable public status labels.
        if len(self._tasks) >= self.max_concurrent_jobs:
            raise ResourceExhaustedError("training_jobs", self.max_concurrent_jobs)
        # The task can start after this coroutine returns. Own the accepted
        # inputs before constructors or caller code can mutate nested values.
        training_id = str(uuid.uuid4())
        now = utc_now()
        output_dir = root / training_id
        request.training_config_overrides = {
            **(request.training_config_overrides or {}),
            "output_dir": str(output_dir),
        }
        request.resume_from_checkpoint = resume

        # Create training configuration
        agent_config = AgentConfig(
            **request.agent_config.model_dump(), model_revision=config.model_revision
        )
        agent_config = replace(agent_config, use_stub_model=use_stub_model)
        agent = MultiTurnAgent(agent_config)

        # Create environment
        environment = ConversationEnvironment(scenarios=request.environment_scenarios)

        # Create job record
        self.training_jobs[training_id] = {
            "status": "running",
            "user_id": user_id,
            "created_at": now,
            "started_at": now,
            "completed_at": None,
            "completion_scope": None,
            "progress": 0.0,
            "current_episode": 0,
            "total_episodes": request.num_episodes,
            "metrics": {},
            "error": None,
            "config": accepted_config,
            "output_dir": str(output_dir),
        }

        # Create cancellation event and launch background task
        cancel_event = asyncio.Event()
        self._cancel_events[training_id] = cancel_event
        task = asyncio.create_task(
            self._run_training(
                training_id, agent, environment, reward_fn, request, cancel_event
            )
        )
        self._tasks[training_id] = task
        task.add_done_callback(
            lambda finished: self._training_done(training_id, finished)
        )

        return training_id

    def _training_done(self, training_id: str, task: asyncio.Task[None]) -> None:
        """Retrieve task failures and clean up even cancellation before startup."""
        error = None if task.cancelled() else task.exception()
        job = self.training_jobs[training_id]
        if job["status"] not in TERMINAL_TRAINING_STATUSES:
            job.update(
                status="cancelled" if task.cancelled() else "failed",
                completed_at=utc_now(),
                error=(
                    None
                    if task.cancelled()
                    else str(error or "Training exited without a terminal status")
                ),
            )
        if self._tasks.get(training_id) is task:
            self._tasks.pop(training_id, None)
            self._cancel_events.pop(training_id, None)

    async def _run_training(
        self,
        training_id: str,
        agent: MultiTurnAgent,
        environment: ConversationEnvironment,
        reward_fn: Any,
        request: TrainingRequest,
        cancel_event: asyncio.Event,
    ) -> None:
        """Run training job in the background."""
        try:
            if cancel_event.is_set():
                raise asyncio.CancelledError("Training cancelled before startup")
            from stateset_agents.training.train import train

            progress_cb = JobProgressCallback(
                job=self.training_jobs[training_id],
                total_episodes=request.num_episodes,
                cancel_event=cancel_event,
            )
            await train(
                agent=agent,
                environment=environment,
                reward_fn=reward_fn,
                num_episodes=request.num_episodes,
                profile=request.profile,
                config_overrides=request.training_config_overrides,
                resume_from_checkpoint=request.resume_from_checkpoint,
                callbacks=[progress_cb],
            )
            progress_cb.raise_if_failed()
            if cancel_event.is_set():
                raise asyncio.CancelledError("Training cancellation requested")

            self.training_jobs[training_id].update(
                {
                    "status": "completed",
                    "completion_scope": progress_cb.completion_scope,
                    "completed_at": utc_now(),
                }
            )

        except asyncio.CancelledError:
            logger.info("Training cancelled for job %s", training_id)
            self.training_jobs[training_id].update(
                {
                    "status": "cancelled",
                    "completed_at": utc_now(),
                }
            )

        except Exception as e:
            # This boundary owns a background task: every ordinary trainer or
            # import failure must become visible job state and be retrieved.
            logger.error("Training failed for job %s: %s", training_id, e)
            self.training_jobs[training_id].update(
                {
                    "status": "failed",
                    "error": str(e),
                    "completed_at": utc_now(),
                }
            )

        finally:
            self._cancel_events.pop(training_id, None)
            self._tasks.pop(training_id, None)

    def cancel_training(self, training_id: str, user_id: str | None = None) -> bool:
        """Request cooperative cancellation without claiming work has stopped.

        Terminal jobs remain unchanged. Repeated requests are idempotent; a
        trainer that has not reached a cancellation boundary stays cancelling.
        """
        if self.queue is not None:
            return self.queue.cancel(training_id, user_id)
        job = self.training_jobs.get(training_id)
        if not job or not self._can_access_job(job, user_id):
            return False
        if job["status"] in TERMINAL_TRAINING_STATUSES:
            return True

        event = self._cancel_events.get(training_id)
        if event is not None:
            event.set()

        job["status"] = "cancelling"
        return True

    def get_training_status(
        self, training_id: str, user_id: str | None = None
    ) -> dict[str, Any] | None:
        """Return a detached status snapshot, or None if missing/inaccessible."""
        if self.queue is not None:
            return self.queue.get(training_id, user_id)
        job = self.training_jobs.get(training_id)
        if not job or not self._can_access_job(job, user_id):
            return None
        return copy.deepcopy(job)

    async def aget_training_status(
        self, training_id: str, user_id: str | None = None
    ) -> dict[str, Any] | None:
        """Read durable status off-loop while preserving the synchronous API."""
        if self.queue is not None:
            return await run_sync_owned(self.get_training_status, training_id, user_id)
        return self.get_training_status(training_id, user_id)

    async def acancel_training(
        self, training_id: str, user_id: str | None = None
    ) -> bool:
        """Drain a durable cancellation write before propagating caller cancellation."""
        if self.queue is not None:
            return await run_sync_owned(self.cancel_training, training_id, user_id)
        return self.cancel_training(training_id, user_id)

    async def list_training_jobs(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        status: str | None = None,
        user_id: str | None = None,
    ) -> tuple[list[tuple[str, dict[str, Any]]], int]:
        """Return one detached, authorized page and total without loading all payloads."""
        if type(page) is not int or page < 1:
            raise ValueError("page must be a positive integer")
        if type(page_size) is not int or not 1 <= page_size <= 100:
            raise ValueError("page_size must be an integer between 1 and 100")
        if status is not None and not isinstance(status, str):
            raise ValueError("status must be a string or None")
        offset = (page - 1) * page_size
        if self.queue is not None:
            return await run_sync_owned(
                self.queue.page,
                owner=user_id,
                status=status,
                limit=page_size,
                offset=offset,
            )
        jobs = [
            (job_id, job)
            for job_id, job in self.training_jobs.items()
            if self._can_access_job(job, user_id)
            and (not status or job.get("status") == status)
        ]
        jobs.sort(
            key=lambda item: (item[1].get("created_at") or datetime.min, item[0]),
            reverse=True,
        )
        return copy.deepcopy(jobs[offset : offset + page_size]), len(jobs)


def build_reward(config: dict[str, Any]) -> Any:
    """Select outcome rewards by default; heuristics require explicit starter mode."""
    mode = config.get("mode", "outcome")
    if mode == "environment":
        return None
    if mode == "outcome":
        from stateset_agents.rewards.task_outcome import TaskOutcomeReward

        return TaskOutcomeReward()
    if mode == "heuristic":
        return CompositeReward(
            [
                HelpfulnessReward(weight=config.get("helpfulness_weight", 0.7)),
                SafetyReward(weight=config.get("safety_weight", 0.3)),
            ]
        )
    raise ValueError("reward_config.mode must be outcome, environment, or heuristic")
