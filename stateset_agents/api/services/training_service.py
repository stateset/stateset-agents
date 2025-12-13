import uuid
import logging
import asyncio
import os
from datetime import datetime
from typing import Dict, Optional
from fastapi import HTTPException
from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.core.reward import CompositeReward, HelpfulnessReward, SafetyReward
# from training.train import train  <-- Moved inside method
from ..schemas import TrainingRequest

logger = logging.getLogger(__name__)

class TrainingService:
    """Service for managing training jobs."""

    def __init__(self):
        self.training_jobs: Dict[str, Dict] = {}

    @property
    def jobs(self) -> Dict[str, Dict]:
        """Compatibility alias used by some router/tests."""
        return self.training_jobs

    async def start_training(self, request: TrainingRequest) -> str:
        """Start a training job."""
        training_id = str(uuid.uuid4())
        now = datetime.utcnow()

        # Create training configuration
        agent_config = AgentConfig(**request.agent_config.dict())
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

        # Start training in background
        self.training_jobs[training_id] = {
            "status": "running",
            "created_at": now,
            "started_at": now,
            "completed_at": None,
            "progress": 0.0,
            "current_episode": 0,
            "total_episodes": request.num_episodes,
            "metrics": {},
            "error": None,
            "config": request.dict(),
        }

        # In non-production environments we avoid kicking off heavyweight
        # background training loops by default.
        environment_name = os.getenv("API_ENVIRONMENT", "production").lower()
        if environment_name == "production":
            asyncio.create_task(
                self._run_training(training_id, agent, environment, reward_fn, request)
            )

        return training_id

    async def _run_training(
        self,
        training_id: str,
        agent: MultiTurnAgent,
        environment,
        reward_fn,
        request: TrainingRequest,
    ):
        """Run training job."""
        from training.train import train

        try:
            trained_agent = await train(
                agent=agent,
                environment=environment,
                reward_fn=reward_fn,
                num_episodes=request.num_episodes,
                profile=request.profile,
            )

            self.training_jobs[training_id].update(
                {
                    "status": "completed",
                    "completed_at": datetime.utcnow(),
                    "trained_agent": trained_agent,
                }
            )

        except Exception as e:
            logger.error(f"Training failed for job {training_id}: {e}")
            self.training_jobs[training_id].update(
                {
                    "status": "failed",
                    "error": str(e),
                    "completed_at": datetime.utcnow(),
                }
            )

    def get_training_status(self, training_id: str) -> Dict:
        """Get training job status."""
        if training_id not in self.training_jobs:
            raise HTTPException(status_code=404, detail="Training job not found")

        return self.training_jobs[training_id]
