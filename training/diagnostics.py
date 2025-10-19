"""
Diagnostics monitor for training
"""

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class DiagnosticsMonitor:
    """
    Simple diagnostics monitor for training
    """

    def __init__(self, config):
        self.config = config
        self.start_time = time.time()
        self.episode_count = 0
        self.total_rewards = []

    def __call__(self, event: str, data: Optional[Dict[str, Any]] = None):
        """Callback for training events"""
        if event == "episode_end":
            self.episode_count += 1
            if data and "total_reward" in data:
                self.total_rewards.append(data["total_reward"])

            # Log progress every 10 episodes
            if self.episode_count % 10 == 0:
                elapsed = time.time() - self.start_time
                avg_reward = sum(self.total_rewards[-10:]) / min(
                    10, len(self.total_rewards)
                )
                logger.info(
                    f"Episode {self.episode_count}: Avg reward (last 10) = {avg_reward:.3f}, "
                    f"Elapsed: {elapsed:.1f}s"
                )

        elif event == "training_end":
            total_time = time.time() - self.start_time
            if self.total_rewards:
                final_avg = sum(self.total_rewards) / len(self.total_rewards)
                logger.info(
                    f"Training completed: {self.episode_count} episodes, "
                    f"Average reward: {final_avg:.3f}, Total time: {total_time:.1f}s"
                )
