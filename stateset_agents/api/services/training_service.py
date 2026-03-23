import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any

from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.core.reward import CompositeReward, HelpfulnessReward, SafetyReward

from ..schemas import TrainingRequest

logger = logging.getLogger(__name__)

TRAINING_SERVICE_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


class JobProgressCallback:
    """Training callback that updates a job dict with episode progress.

    Also monitors a ``cancel_event`` and raises ``asyncio.CancelledError``
    when cancellation is requested so the trainer loop exits cleanly.
    """

    def __init__(
        self,
        job: dict[str, Any],
        total_episodes: int,
        cancel_event: asyncio.Event,
    ) -> None:
        self.job = job
        self.total_episodes = total_episodes
        self.cancel_event = cancel_event

    def on_episode_end(self, episode: int, metrics: dict[str, Any]) -> None:
        """Called by the trainer after each episode."""
        self.job["current_episode"] = episode + 1
        self.job["progress"] = ((episode + 1) / self.total_episodes) * 100.0
        # Only store JSON-serializable scalar metrics
        self.job["metrics"] = {
            k: v
            for k, v in metrics.items()
            if isinstance(v, (int, float, str, bool))
        }

        if self.cancel_event.is_set():
            raise asyncio.CancelledError("Training cancelled by user")


class TrainingService:
    """Service for managing training jobs."""

    def __init__(self) -> None:
        self.training_jobs: dict[str, dict[str, Any]] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}

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

    async def start_training(self, request: TrainingRequest, user_id: str | None = None) -> str:
        """Start a training job."""
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
            "progress": 0.0,
            "current_episode": 0,
            "total_episodes": request.num_episodes,
            "metrics": {},
            "error": None,
            "config": request.model_dump(),
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

        return training_id

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
        from stateset_agents.training.train import train

        progress_cb = JobProgressCallback(
            job=self.training_jobs[training_id],
            total_episodes=request.num_episodes,
            cancel_event=cancel_event,
        )

        try:
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

            self.training_jobs[training_id].update(
                {
                    "status": "completed",
                    "progress": 100.0,
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

        except TRAINING_SERVICE_EXCEPTIONS as e:
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
        """Request cancellation of a running training job."""
        job = self.training_jobs.get(training_id)
        if not job or not self._can_access_job(job, user_id):
            return False

        event = self._cancel_events.get(training_id)
        if event is not None:
            event.set()

        if training_id in self.training_jobs:
            self.training_jobs[training_id]["status"] = "cancelled"
            self.training_jobs[training_id]["completed_at"] = datetime.utcnow()
        return True

    def get_training_status(self, training_id: str, user_id: str | None = None) -> dict[str, Any] | None:
        """Get training job status, or None if not found."""
        job = self.training_jobs.get(training_id)
        if not job or not self._can_access_job(job, user_id):
            return None
        return job
