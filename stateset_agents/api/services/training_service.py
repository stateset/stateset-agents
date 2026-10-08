import asyncio
import copy
import json
import logging
import uuid
from datetime import datetime
from typing import Any

from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.core.reward import CompositeReward, HelpfulnessReward, SafetyReward
from stateset_agents.utils.async_calls import drain_owned_operation

from ..errors import ResourceExhaustedError, ServiceUnavailableError
from ..schemas import TrainingRequest

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

    def __init__(self, *, max_concurrent_jobs: int = 1) -> None:
        if type(max_concurrent_jobs) is not int or max_concurrent_jobs < 1:
            raise ValueError("max_concurrent_jobs must be a positive integer")
        self._max_concurrent_jobs = max_concurrent_jobs
        self.training_jobs: dict[str, dict[str, Any]] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._closing = False

    @property
    def max_concurrent_jobs(self) -> int:
        """Maximum number of workers this service may own concurrently."""
        return self._max_concurrent_jobs

    @property
    def is_closing(self) -> bool:
        """Whether shutdown has stopped admission of new training jobs."""
        return self._closing

    async def aclose(self) -> None:
        """Cancel cooperatively and own all workers until their cleanup finishes.

        Repeated or concurrent calls are safe. Caller cancellation is propagated
        only after workers exit. Unresponsive trainers can delay shutdown; this
        does not forcibly interrupt threads or remote provider work.
        """
        self._closing = True
        tasks = tuple(self._tasks.values())
        for training_id in tuple(self._tasks):
            self.cancel_training(training_id)

        async def finish() -> None:
            await asyncio.gather(*tasks, return_exceptions=True)

        await drain_owned_operation(finish())

    @property
    def jobs(self) -> dict[str, dict[str, Any]]:
        """Compatibility alias used by some router/tests."""
        return self.training_jobs

    @staticmethod
    def _can_access_job(job: dict[str, Any], user_id: str | None) -> bool:
        if user_id is None:
            return True
        owner = job.get("user_id")
        return owner is None or owner == user_id

    async def start_training(
        self, request: TrainingRequest, user_id: str | None = None
    ) -> str:
        """Start a training job."""
        if self._closing:
            raise ServiceUnavailableError("Training service is shutting down")
        # Submission contains no await between this check and task registration,
        # so admission is atomic for this service's event loop. Count owned tasks,
        # including cancellation cleanup, rather than mutable public status labels.
        if len(self._tasks) >= self.max_concurrent_jobs:
            raise ResourceExhaustedError("training_jobs", self.max_concurrent_jobs)
        # The task can start after this coroutine returns. Own the accepted
        # inputs before constructors or caller code can mutate nested values.
        request = request.model_copy(deep=True)
        accepted_config = copy.deepcopy(request.model_dump())
        training_id = str(uuid.uuid4())
        now = datetime.utcnow()

        # Create training configuration
        agent_config = AgentConfig(**request.agent_config.model_dump())
        agent = MultiTurnAgent(agent_config)

        # Create environment
        environment = ConversationEnvironment(scenarios=request.environment_scenarios)

        # Create reward function
        reward_fn = CompositeReward(
            [
                HelpfulnessReward(
                    weight=request.reward_config.get("helpfulness_weight", 0.7)
                ),
                SafetyReward(weight=request.reward_config.get("safety_weight", 0.3)),
            ]
        )

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
                completed_at=datetime.utcnow(),
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
        reward_fn: CompositeReward,
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
                    "completed_at": datetime.utcnow(),
                }
            )

        except asyncio.CancelledError:
            logger.info("Training cancelled for job %s", training_id)
            self.training_jobs[training_id].update(
                {
                    "status": "cancelled",
                    "completed_at": datetime.utcnow(),
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
                    "completed_at": datetime.utcnow(),
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
        job = self.training_jobs.get(training_id)
        if not job or not self._can_access_job(job, user_id):
            return None
        return copy.deepcopy(job)
