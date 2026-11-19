"""
Sim-to-Real Transfer Metrics

Provides comprehensive metrics for evaluating the quality of
sim-to-real transfer in conversational agents.
"""

import logging
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

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

    def to_dict(self) -> dict[str, float]:
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
    def from_dict(cls, data: dict[str, float]) -> "SimToRealMetrics":
        """Create metrics from dictionary"""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


def _histogram_probabilities(
    p: np.ndarray, q: np.ndarray, epsilon: float
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize each histogram independently, then apply per-bin smoothing."""
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    if np.shape(p) != np.shape(q):
        raise ValueError("histograms must have the same shape")

    def probabilities(values: np.ndarray) -> np.ndarray:
        if np.iscomplexobj(values):
            raise ValueError("histograms require real masses")
        mass = np.asarray(values, dtype=np.longdouble).ravel()
        if not mass.size or not np.all(np.isfinite(mass)) or np.any(mass < 0):
            raise ValueError("histograms require nonempty, finite, nonnegative masses")
        peak = mass.max()
        if peak == 0:
            raise ValueError("each histogram must have positive total mass")
        scaled = mass / peak
        prob: np.ndarray = scaled / scaled.sum()
        if epsilon:
            smoothed = prob + epsilon
            scaled = smoothed / smoothed.max()
            prob = scaled / scaled.sum()
        return prob

    return probabilities(p), probabilities(q)


def compute_kl_divergence(
    p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10
) -> float:
    """Forward KL in nats. Positive epsilon smooths each normalized histogram.

    With zero smoothing a positive source mass outside the target support
    gives infinity. Bins with zero mass in both distributions contribute zero.
    """
    source, target = _histogram_probabilities(p, q, epsilon)
    if np.any((source > 0) & (target == 0)):
        return math.inf
    positive = (source > 0) & (target > 0)
    a, b = source[positive], target[positive]
    delta = a - b
    # The linear correction sums to zero over normalized distributions.
    # Its nonnegative per-bin form avoids cancelling large signed terms.
    close = np.abs(delta) <= 0.001 * b
    u = delta[close] / b[close]
    near = (
        b[close] * u**2 * (1 / 2 + u * (-1 / 6 + u * (1 / 12 + u * (-1 / 20 + u / 30))))
    )
    far = a[~close] * (np.log(a[~close]) - np.log(b[~close])) - delta[~close]
    return float(near.sum() + far.sum() + target[source == 0].sum())


def compute_js_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 0.0) -> float:
    """Balanced Jensen-Shannon divergence in nats, from zero to log(2).

    Positive epsilon adds a pseudocount to each normalized bin before mixing.
    The default includes disjoint supports and ignores shared zero bins.
    """
    source, target = _histogram_probabilities(p, q, epsilon)
    total = source + target
    active = total > 0
    a, b, total = source[active], target[active], total[active]
    u = (a - b) / total
    close = np.abs(u) <= 0.5
    # Each cell is its mixture mass times the divergence of the posterior
    # label probability from chance. This form preserves the quadratic term
    # when the two frequencies are almost equal.
    v = u[close]
    near = (total[close] / 2) * (v * np.arctanh(v) + np.log1p(-(v * v)) / 2)
    mix = total[~close] / 2
    far = np.longdouble(0)
    for mass in (a[~close], b[~close]):
        positive = mass > 0
        far += (mass[positive] * (np.log(mass[positive]) - np.log(mix[positive]))).sum()
    return float(near.sum() + far / 2)


def compute_mmd(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float = 1.0,
) -> float:
    """Biased empirical Gaussian MMD squared, including same-sample diagonals."""
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Gaussian bandwidth must be finite and positive")
    if np.iscomplexobj(x) or np.iscomplexobj(y):
        raise ValueError("Gaussian features must be real")
    x, y = np.asarray(x, dtype=np.longdouble), np.asarray(y, dtype=np.longdouble)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if (
        x.ndim != 2
        or y.ndim != 2
        or not x.size
        or not y.size
        or x.shape[1] != y.shape[1]
    ):
        raise ValueError("MMD needs nonempty sample sets of equal feature width")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Gaussian features must be finite")

    def kernel_means(
        a: np.ndarray, b: np.ndarray
    ) -> tuple[np.longdouble, np.longdouble]:
        # Bound temporary storage in samples x samples x features.
        chunk = max(1, (1 << 18) // b.size)
        anchors = 0
        correction = np.longdouble(0)
        for start in range(0, len(a), chunk):
            scaled = (a[start : start + chunk, None, :] - b[None, :, :]) / sigma
            exponent = -np.square(scaled).sum(axis=-1) / 2
            near = exponent > -math.log(2)
            anchors += int(np.count_nonzero(near))
            correction += np.expm1(exponent[near]).sum() + np.exp(exponent[~near]).sum()
        count = len(a) * len(b)
        return np.longdouble(anchors) / count, correction / count

    xx, yy, xy = (kernel_means(a, b) for a, b in ((x, x), (y, y), (x, y)))
    base = (xx[0] - xy[0]) + (yy[0] - xy[0])
    correction = (xx[1] - xy[1]) + (yy[1] - xy[1])
    return max(0.0, float(base + correction))


def compute_distribution_divergence(
    sim_values: list[float],
    real_values: list[float],
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
    sim_array = np.asarray(sim_values)
    real_array = np.asarray(real_values)

    if len(sim_array) == 0 or len(real_array) == 0:
        return float("inf")

    if method == "mmd":
        return compute_mmd(sim_array, real_array)

    # Create histograms for KL/JS
    all_values = np.concatenate([sim_array, real_array])
    min_val, max_val = all_values.min(), all_values.max()

    if min_val == max_val:
        return 0.0

    bins = np.linspace(min_val, max_val, num_bins + 1)

    sim_hist, _ = np.histogram(sim_array, bins=bins)
    real_hist, _ = np.histogram(real_array, bins=bins)

    if method == "kl":
        return compute_kl_divergence(real_hist, sim_hist)
    elif method == "js":
        return compute_js_divergence(sim_hist, real_hist)
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_response_statistics(
    trajectories: list[Any],
    role: str = "assistant",
) -> dict[str, Any]:
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
    word_counts: Counter[str] = Counter()
    rewards = []

    for traj in trajectories:
        turns = getattr(traj, "turns", [])
        turn_counts.append(len(turns))

        for turn in turns:
            if getattr(turn, "role", None) == role:
                content = getattr(turn, "content", "")
                words = content.lower().split()
                response_lengths.append(len(words))
                word_counts.update(words)

        if hasattr(traj, "total_reward"):
            rewards.append(traj.total_reward)

    return {
        "response_lengths": response_lengths,
        "turn_counts": turn_counts,
        "vocabulary_size": len(word_counts),
        "vocabulary": set(word_counts),
        "word_counts": word_counts,
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
        self._real_stats: dict[str, Any] | None = None

    def compute_distribution_metrics(
        self,
        sim_trajectories: list[Any],
        real_trajectories: list[Any],
    ) -> dict[str, float]:
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

        sim_words, real_words = sim_stats["word_counts"], real_stats["word_counts"]
        if sim_words and real_words:
            words = sorted(sim_words.keys() | real_words.keys())
            metrics["vocabulary_js_divergence"] = compute_js_divergence(
                np.array([sim_words[word] for word in words]),
                np.array([real_words[word] for word in words]),
            )

        # Reward correlation
        if sim_stats["rewards"] and real_stats["rewards"]:
            if len(sim_stats["rewards"]) > 1 and len(real_stats["rewards"]) > 1:
                # Sample to same size
                min_len = min(len(sim_stats["rewards"]), len(real_stats["rewards"]))
                sim_rewards = np.random.choice(
                    sim_stats["rewards"], min_len, replace=False
                )
                real_rewards = np.random.choice(
                    real_stats["rewards"], min_len, replace=False
                )
                correlation = np.corrcoef(sim_rewards, real_rewards)[0, 1]
                metrics["reward_correlation"] = (
                    float(correlation) if not np.isnan(correlation) else 0
                )

        return metrics

    def compute_performance_gap(
        self,
        agent: Any,
        sim_rewards: list[float],
        real_rewards: list[float],
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
    ) -> dict[str, float]:
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
            "user_model_likelihood": (
                float(np.mean(likelihoods)) if likelihoods else 0.0
            ),
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

                    new_state, reward, done, _ = await self.simulator.step(
                        state, action
                    )
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
            metrics.vocabulary_js_divergence = dist_metrics.get(
                "vocabulary_js_divergence", 0
            )
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
        metrics.overall_gap = float(np.mean([g for g in gap_components if g > 0]))

        logger.info(f"Evaluation complete. Overall gap: {metrics.overall_gap:.4f}")
        return metrics

    def get_real_stats(self) -> dict[str, Any]:
        """Get cached real data statistics"""
        if self._real_stats is None and self.real_dataset is not None:
            self._real_stats = compute_response_statistics(list(self.real_dataset))
        return self._real_stats or {}


class MockTrajectory:
    """Mock trajectory for evaluation"""

    def __init__(self):
        self.turns = []
        self.total_reward = 0.0
