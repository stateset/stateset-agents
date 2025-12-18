"""
Sim-to-Real Transfer Metrics

Provides comprehensive metrics for evaluating the quality of
sim-to-real transfer in conversational agents.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)


@dataclass
class SimToRealMetrics:
    """
    Comprehensive metrics for evaluating sim-to-real transfer quality.

    Attributes:
        response_length_kl: KL divergence of response length distributions
        vocabulary_js_divergence: Jensen-Shannon divergence of vocabularies
        turn_count_mmd: MMD between turn count distributions
        reward_correlation: Correlation between sim and real rewards
        policy_performance_gap: Difference in agent performance
        user_model_likelihood: Likelihood of real responses under user model
        scenario_coverage: Coverage of real scenarios by simulation
    """

    # Distribution metrics
    response_length_kl: float = 0.0
    vocabulary_js_divergence: float = 0.0
    turn_count_mmd: float = 0.0
    response_time_gap: float = 0.0

    # Performance metrics
    reward_correlation: float = 0.0
    policy_performance_gap: float = 0.0
    success_rate_gap: float = 0.0

    # Calibration metrics
    user_model_likelihood: float = 0.0
    scenario_coverage: float = 0.0
    persona_diversity: float = 0.0

    # Overall score
    overall_gap: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary"""
        return {
            "response_length_kl": self.response_length_kl,
            "vocabulary_js_divergence": self.vocabulary_js_divergence,
            "turn_count_mmd": self.turn_count_mmd,
            "response_time_gap": self.response_time_gap,
            "reward_correlation": self.reward_correlation,
            "policy_performance_gap": self.policy_performance_gap,
            "success_rate_gap": self.success_rate_gap,
            "user_model_likelihood": self.user_model_likelihood,
            "scenario_coverage": self.scenario_coverage,
            "persona_diversity": self.persona_diversity,
            "overall_gap": self.overall_gap,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "SimToRealMetrics":
        """Create metrics from dictionary"""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


def compute_kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    epsilon: float = 1e-10,
) -> float:
    """
    Compute KL divergence D_KL(P || Q).

    Args:
        p: True distribution
        q: Approximating distribution
        epsilon: Small value to avoid log(0)

    Returns:
        KL divergence value
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Normalize
    p = p / (p.sum() + epsilon)
    q = q / (q.sum() + epsilon)

    # Add epsilon to avoid log(0)
    p = p + epsilon
    q = q + epsilon

    return float(np.sum(p * np.log(p / q)))


def compute_js_divergence(
    p: np.ndarray,
    q: np.ndarray,
    epsilon: float = 1e-10,
) -> float:
    """
    Compute Jensen-Shannon divergence.

    JS(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
    where M = 0.5 * (P + Q)

    Args:
        p: First distribution
        q: Second distribution
        epsilon: Small value to avoid log(0)

    Returns:
        JS divergence value (0 to 1)
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Normalize
    p = p / (p.sum() + epsilon)
    q = q / (q.sum() + epsilon)

    # Compute mixture
    m = 0.5 * (p + q)

    # JS divergence
    js = 0.5 * compute_kl_divergence(p, m, epsilon) + 0.5 * compute_kl_divergence(q, m, epsilon)

    return float(js)


def compute_mmd(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float = 1.0,
) -> float:
    """
    Compute Maximum Mean Discrepancy with RBF kernel.

    Args:
        x: Samples from first distribution
        y: Samples from second distribution
        sigma: RBF kernel bandwidth

    Returns:
        MMD value
    """
    x = np.asarray(x).reshape(-1, 1) if x.ndim == 1 else x
    y = np.asarray(y).reshape(-1, 1) if y.ndim == 1 else y

    def rbf_kernel(a, b, sigma):
        sq_dist = np.sum((a[:, np.newaxis] - b[np.newaxis, :]) ** 2, axis=-1)
        return np.exp(-sq_dist / (2 * sigma ** 2))

    k_xx = rbf_kernel(x, x, sigma)
    k_yy = rbf_kernel(y, y, sigma)
    k_xy = rbf_kernel(x, y, sigma)

    n_x = x.shape[0]
    n_y = y.shape[0]

    mmd = (k_xx.sum() / (n_x * n_x) +
           k_yy.sum() / (n_y * n_y) -
           2 * k_xy.sum() / (n_x * n_y))

    return float(max(0, mmd))


def compute_distribution_divergence(
    sim_values: List[float],
    real_values: List[float],
    method: str = "kl",
    num_bins: int = 50,
) -> float:
    """
    Compute divergence between two distributions of values.

    Args:
        sim_values: Values from simulation
        real_values: Values from real data
        method: Divergence method ("kl", "js", "mmd")
        num_bins: Number of histogram bins for KL/JS

    Returns:
        Divergence value
    """
    sim_values = np.asarray(sim_values)
    real_values = np.asarray(real_values)

    if len(sim_values) == 0 or len(real_values) == 0:
        return float("inf")

    if method == "mmd":
        return compute_mmd(sim_values, real_values)

    # Create histograms for KL/JS
    all_values = np.concatenate([sim_values, real_values])
    min_val, max_val = all_values.min(), all_values.max()

    if min_val == max_val:
        return 0.0

    bins = np.linspace(min_val, max_val, num_bins + 1)

    sim_hist, _ = np.histogram(sim_values, bins=bins, density=True)
    real_hist, _ = np.histogram(real_values, bins=bins, density=True)

    if method == "kl":
        return compute_kl_divergence(real_hist, sim_hist)
    elif method == "js":
        return compute_js_divergence(sim_hist, real_hist)
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_response_statistics(
    trajectories: List[Any],
    role: str = "assistant",
) -> Dict[str, Any]:
    """
    Compute statistics from conversation trajectories.

    Args:
        trajectories: List of trajectory objects
        role: Role to compute statistics for

    Returns:
        Dictionary of statistics
    """
    response_lengths = []
    turn_counts = []
    vocabulary = set()
    rewards = []

    for traj in trajectories:
        turns = getattr(traj, "turns", [])
        turn_counts.append(len(turns))

        for turn in turns:
            if getattr(turn, "role", None) == role:
                content = getattr(turn, "content", "")
                words = content.lower().split()
                response_lengths.append(len(words))
                vocabulary.update(words)

        if hasattr(traj, "total_reward"):
            rewards.append(traj.total_reward)

    return {
        "response_lengths": response_lengths,
        "turn_counts": turn_counts,
        "vocabulary_size": len(vocabulary),
        "vocabulary": vocabulary,
        "rewards": rewards,
        "mean_response_length": np.mean(response_lengths) if response_lengths else 0,
        "std_response_length": np.std(response_lengths) if response_lengths else 0,
        "mean_turn_count": np.mean(turn_counts) if turn_counts else 0,
        "mean_reward": np.mean(rewards) if rewards else 0,
    }


class SimToRealEvaluator:
    """
    Evaluates quality of sim-to-real transfer.

    Computes comprehensive metrics comparing simulated and real
    conversation distributions and agent performance.

    Example:
        >>> evaluator = SimToRealEvaluator(simulator, real_dataset)
        >>> metrics = await evaluator.full_evaluation(agent, num_episodes=100)
        >>> print(f"Overall gap: {metrics.overall_gap:.4f}")
    """

    def __init__(
        self,
        simulator: Any = None,  # ConversationSimulator
        real_dataset: Any = None,  # ConversationDataset
        user_model: Any = None,  # UserBehaviorModel
    ):
        self.simulator = simulator
        self.real_dataset = real_dataset
        self.user_model = user_model

        # Cached statistics
        self._real_stats: Optional[Dict[str, Any]] = None

    def compute_distribution_metrics(
        self,
        sim_trajectories: List[Any],
        real_trajectories: List[Any],
    ) -> Dict[str, float]:
        """
        Compute distribution-based metrics between sim and real data.

        Args:
            sim_trajectories: Simulated trajectories
            real_trajectories: Real trajectories

        Returns:
            Dictionary of distribution metrics
        """
        sim_stats = compute_response_statistics(sim_trajectories)
        real_stats = compute_response_statistics(real_trajectories)

        metrics = {}

        # Response length divergence
        if sim_stats["response_lengths"] and real_stats["response_lengths"]:
            metrics["response_length_kl"] = compute_distribution_divergence(
                sim_stats["response_lengths"],
                real_stats["response_lengths"],
                method="kl",
            )

        # Turn count MMD
        if sim_stats["turn_counts"] and real_stats["turn_counts"]:
            metrics["turn_count_mmd"] = compute_distribution_divergence(
                sim_stats["turn_counts"],
                real_stats["turn_counts"],
                method="mmd",
            )

        # Vocabulary divergence (Jaccard distance)
        if sim_stats["vocabulary"] and real_stats["vocabulary"]:
            sim_vocab = sim_stats["vocabulary"]
            real_vocab = real_stats["vocabulary"]
            intersection = len(sim_vocab & real_vocab)
            union = len(sim_vocab | real_vocab)
            metrics["vocabulary_js_divergence"] = 1.0 - (intersection / union if union > 0 else 0)

        # Reward correlation
        if sim_stats["rewards"] and real_stats["rewards"]:
            if len(sim_stats["rewards"]) > 1 and len(real_stats["rewards"]) > 1:
                # Sample to same size
                min_len = min(len(sim_stats["rewards"]), len(real_stats["rewards"]))
                sim_rewards = np.random.choice(sim_stats["rewards"], min_len, replace=False)
                real_rewards = np.random.choice(real_stats["rewards"], min_len, replace=False)
                correlation = np.corrcoef(sim_rewards, real_rewards)[0, 1]
                metrics["reward_correlation"] = float(correlation) if not np.isnan(correlation) else 0

        return metrics

    def compute_performance_gap(
        self,
        agent: Any,
        sim_rewards: List[float],
        real_rewards: List[float],
    ) -> float:
        """
        Compute gap in agent performance between sim and real.

        Args:
            agent: The agent being evaluated
            sim_rewards: Rewards from simulation
            real_rewards: Rewards from real environment

        Returns:
            Performance gap (sim - real)
        """
        if not sim_rewards or not real_rewards:
            return 0.0

        sim_mean = np.mean(sim_rewards)
        real_mean = np.mean(real_rewards)

        return float(sim_mean - real_mean)

    def evaluate_user_model(
        self,
        user_model: Any,
        real_data: Any,
        num_samples: int = 100,
    ) -> Dict[str, float]:
        """
        Evaluate user model quality on real data.

        Args:
            user_model: Trained UserBehaviorModel
            real_data: Real conversation dataset
            num_samples: Number of samples to evaluate

        Returns:
            User model evaluation metrics
        """
        if user_model is None or real_data is None:
            return {"user_model_likelihood": 0.0}

        likelihoods = []

        # Sample random conversations
        indices = np.random.choice(
            len(real_data), size=min(num_samples, len(real_data)), replace=False
        )

        for idx in indices:
            traj = real_data[idx]
            turns = getattr(traj, "turns", [])

            for i in range(len(turns) - 1):
                if turns[i].role == "assistant" and turns[i + 1].role == "user":
                    # Compute likelihood (placeholder - would use actual embeddings)
                    likelihood = 0.5  # Placeholder
                    likelihoods.append(likelihood)

        return {
            "user_model_likelihood": np.mean(likelihoods) if likelihoods else 0.0,
            "num_samples": len(likelihoods),
        }

    async def full_evaluation(
        self,
        agent: Any = None,
        num_episodes: int = 100,
    ) -> SimToRealMetrics:
        """
        Perform comprehensive sim-to-real evaluation.

        Args:
            agent: Agent to evaluate (optional)
            num_episodes: Number of episodes to generate for evaluation

        Returns:
            SimToRealMetrics with all evaluation results
        """
        logger.info(f"Running full sim-to-real evaluation with {num_episodes} episodes")

        metrics = SimToRealMetrics()

        # Generate simulated trajectories
        sim_trajectories = []
        if self.simulator is not None:
            for _ in range(num_episodes):
                state = await self.simulator.reset()
                trajectory = MockTrajectory()

                for _ in range(20):  # Max turns
                    # Generate dummy response
                    response = "This is a simulated response for evaluation."
                    from ..core.trajectory import ConversationTurn
                    action = ConversationTurn(role="assistant", content=response)
                    trajectory.turns.append(action)

                    new_state, reward, done, _ = await self.simulator.step(state, action)
                    trajectory.total_reward += reward

                    if done:
                        break
                    state = new_state

                sim_trajectories.append(trajectory)

        # Get real trajectories
        real_trajectories = []
        if self.real_dataset is not None:
            real_trajectories = list(self.real_dataset)[:num_episodes]

        # Compute distribution metrics
        if sim_trajectories and real_trajectories:
            dist_metrics = self.compute_distribution_metrics(
                sim_trajectories, real_trajectories
            )
            metrics.response_length_kl = dist_metrics.get("response_length_kl", 0)
            metrics.turn_count_mmd = dist_metrics.get("turn_count_mmd", 0)
            metrics.vocabulary_js_divergence = dist_metrics.get("vocabulary_js_divergence", 0)
            metrics.reward_correlation = dist_metrics.get("reward_correlation", 0)

        # Evaluate user model
        if self.user_model is not None and self.real_dataset is not None:
            user_metrics = self.evaluate_user_model(
                self.user_model, self.real_dataset, num_episodes
            )
            metrics.user_model_likelihood = user_metrics.get("user_model_likelihood", 0)

        # Compute overall gap (weighted average of normalized metrics)
        gap_components = [
            metrics.response_length_kl,
            metrics.turn_count_mmd,
            metrics.vocabulary_js_divergence,
            1 - abs(metrics.reward_correlation),  # Convert correlation to gap
        ]
        metrics.overall_gap = np.mean([g for g in gap_components if g > 0])

        logger.info(f"Evaluation complete. Overall gap: {metrics.overall_gap:.4f}")
        return metrics

    def get_real_stats(self) -> Dict[str, Any]:
        """Get cached real data statistics"""
        if self._real_stats is None and self.real_dataset is not None:
            self._real_stats = compute_response_statistics(list(self.real_dataset))
        return self._real_stats or {}


class MockTrajectory:
    """Mock trajectory for evaluation"""

    def __init__(self):
        self.turns = []
        self.total_reward = 0.0
