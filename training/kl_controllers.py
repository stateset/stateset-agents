"""
KL Divergence Controllers for RL Training

This module provides various KL penalty controllers for managing the trade-off
between policy improvement and staying close to a reference policy.

Controllers included:
- FixedKLController: Constant KL coefficient
- AdaptiveKLController: Adjusts based on measured KL
- LinearKLScheduler: Linearly increases/decreases KL coefficient
- CosineKLScheduler: Cosine annealing of KL coefficient
- WarmupKLScheduler: Gradual warmup of KL penalty

Reference: https://arxiv.org/abs/1909.08593 (Fine-Tuning Language Models from Human Preferences)
"""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class KLStats:
    """Statistics for KL divergence tracking."""
    current_kl: float = 0.0
    mean_kl: float = 0.0
    max_kl: float = 0.0
    min_kl: float = float('inf')
    num_updates: int = 0

    def update(self, kl: float) -> None:
        """Update statistics with new KL measurement."""
        self.current_kl = kl
        self.max_kl = max(self.max_kl, kl)
        self.min_kl = min(self.min_kl, kl)

        # Running mean
        self.num_updates += 1
        self.mean_kl = self.mean_kl + (kl - self.mean_kl) / self.num_updates


class KLController(ABC):
    """Abstract base class for KL controllers."""

    def __init__(self, init_kl_coef: float = 0.1):
        self.kl_coef = init_kl_coef
        self.stats = KLStats()

    @abstractmethod
    def update(self, current_kl: float, step: int) -> float:
        """
        Update KL coefficient based on current KL and step.

        Args:
            current_kl: Measured KL divergence
            step: Current training step

        Returns:
            Updated KL coefficient
        """
        pass

    def get_coef(self) -> float:
        """Get current KL coefficient."""
        return self.kl_coef

    def get_stats(self) -> KLStats:
        """Get KL statistics."""
        return self.stats


class FixedKLController(KLController):
    """
    Fixed KL coefficient controller.

    Simply maintains a constant KL penalty throughout training.
    """

    def __init__(self, kl_coef: float = 0.1):
        super().__init__(kl_coef)

    def update(self, current_kl: float, step: int) -> float:
        """KL coefficient remains fixed."""
        self.stats.update(current_kl)
        return self.kl_coef


class AdaptiveKLController(KLController):
    """
    Adaptive KL penalty controller.

    Adjusts the KL coefficient to keep KL divergence near a target value.
    Based on the InstructGPT paper approach.

    The coefficient is adjusted proportionally to how far the current KL
    is from the target:
    - If KL > target: increase coefficient (penalize divergence more)
    - If KL < target: decrease coefficient (allow more divergence)

    Reference: https://arxiv.org/abs/2203.02155
    """

    def __init__(
        self,
        init_kl_coef: float = 0.1,
        target_kl: float = 0.01,
        horizon: int = 10000,
        min_kl_coef: float = 0.001,
        max_kl_coef: float = 100.0,
    ):
        super().__init__(init_kl_coef)
        self.target_kl = target_kl
        self.horizon = horizon
        self.min_kl_coef = min_kl_coef
        self.max_kl_coef = max_kl_coef

    def update(self, current_kl: float, step: int) -> float:
        """Adjust KL coefficient based on current KL divergence."""
        self.stats.update(current_kl)

        # Proportional error
        proportional_error = (current_kl - self.target_kl) / self.target_kl

        # Adjust coefficient
        # Using a multiplicative update scaled by progress through horizon
        n_steps = min(step, self.horizon)
        mult = 1.0 + proportional_error * n_steps / self.horizon

        self.kl_coef = self.kl_coef * mult

        # Clamp to valid range
        self.kl_coef = max(self.min_kl_coef, min(self.kl_coef, self.max_kl_coef))

        logger.debug(
            f"KL update: current={current_kl:.4f}, target={self.target_kl:.4f}, "
            f"coef={self.kl_coef:.4f}"
        )

        return self.kl_coef


class LinearKLScheduler(KLController):
    """
    Linear KL coefficient scheduler.

    Linearly interpolates the KL coefficient from start to end value
    over a specified number of steps.
    """

    def __init__(
        self,
        init_kl_coef: float = 0.1,
        final_kl_coef: float = 0.01,
        total_steps: int = 10000,
    ):
        super().__init__(init_kl_coef)
        self.init_kl_coef = init_kl_coef
        self.final_kl_coef = final_kl_coef
        self.total_steps = total_steps

    def update(self, current_kl: float, step: int) -> float:
        """Linearly interpolate KL coefficient."""
        self.stats.update(current_kl)

        progress = min(step / self.total_steps, 1.0)
        self.kl_coef = (
            self.init_kl_coef
            + (self.final_kl_coef - self.init_kl_coef) * progress
        )

        return self.kl_coef


class CosineKLScheduler(KLController):
    """
    Cosine annealing KL coefficient scheduler.

    Uses cosine annealing to smoothly transition the KL coefficient.
    This provides a more gradual change than linear scheduling.
    """

    def __init__(
        self,
        init_kl_coef: float = 0.1,
        min_kl_coef: float = 0.01,
        total_steps: int = 10000,
        num_cycles: float = 0.5,
    ):
        super().__init__(init_kl_coef)
        self.init_kl_coef = init_kl_coef
        self.min_kl_coef = min_kl_coef
        self.total_steps = total_steps
        self.num_cycles = num_cycles

    def update(self, current_kl: float, step: int) -> float:
        """Apply cosine annealing to KL coefficient."""
        self.stats.update(current_kl)

        progress = step / self.total_steps
        cosine_value = math.cos(math.pi * self.num_cycles * progress)

        # Map cosine from [-1, 1] to [min_kl_coef, init_kl_coef]
        self.kl_coef = (
            self.min_kl_coef
            + 0.5 * (self.init_kl_coef - self.min_kl_coef) * (1 + cosine_value)
        )

        return self.kl_coef


class WarmupKLScheduler(KLController):
    """
    Warmup KL coefficient scheduler.

    Gradually increases the KL coefficient during a warmup period,
    then maintains or decays it.

    This helps stabilize early training when the policy might diverge
    significantly from the reference.
    """

    def __init__(
        self,
        init_kl_coef: float = 0.0,
        peak_kl_coef: float = 0.1,
        final_kl_coef: float = 0.05,
        warmup_steps: int = 1000,
        total_steps: int = 10000,
    ):
        super().__init__(init_kl_coef)
        self.peak_kl_coef = peak_kl_coef
        self.final_kl_coef = final_kl_coef
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def update(self, current_kl: float, step: int) -> float:
        """Apply warmup + decay schedule to KL coefficient."""
        self.stats.update(current_kl)

        if step < self.warmup_steps:
            # Linear warmup
            progress = step / self.warmup_steps
            self.kl_coef = progress * self.peak_kl_coef
        else:
            # Linear decay from peak to final
            decay_progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            decay_progress = min(decay_progress, 1.0)
            self.kl_coef = (
                self.peak_kl_coef
                + (self.final_kl_coef - self.peak_kl_coef) * decay_progress
            )

        return self.kl_coef


class HybridKLController(KLController):
    """
    Hybrid KL controller combining adaptive and scheduled approaches.

    Uses a scheduled baseline with adaptive adjustments based on
    measured KL divergence.
    """

    def __init__(
        self,
        init_kl_coef: float = 0.1,
        target_kl: float = 0.01,
        min_kl_coef: float = 0.001,
        max_kl_coef: float = 100.0,
        total_steps: int = 10000,
        adaptation_strength: float = 0.5,
    ):
        super().__init__(init_kl_coef)
        self.target_kl = target_kl
        self.min_kl_coef = min_kl_coef
        self.max_kl_coef = max_kl_coef
        self.total_steps = total_steps
        self.adaptation_strength = adaptation_strength

        # Scheduled component (cosine decay)
        self.scheduled_coef = init_kl_coef

    def update(self, current_kl: float, step: int) -> float:
        """Apply hybrid schedule with adaptive adjustment."""
        self.stats.update(current_kl)

        # Scheduled component: cosine decay
        progress = step / self.total_steps
        self.scheduled_coef = (
            self.min_kl_coef
            + 0.5 * (self.kl_coef - self.min_kl_coef) * (1 + math.cos(math.pi * progress))
        )

        # Adaptive component
        if self.target_kl > 0:
            kl_ratio = current_kl / self.target_kl
            adaptive_multiplier = kl_ratio ** self.adaptation_strength
        else:
            adaptive_multiplier = 1.0

        # Combine
        self.kl_coef = self.scheduled_coef * adaptive_multiplier

        # Clamp
        self.kl_coef = max(self.min_kl_coef, min(self.kl_coef, self.max_kl_coef))

        return self.kl_coef


class NoKLController(KLController):
    """
    No KL penalty controller.

    Used when KL penalty is disabled (beta = 0).
    Some algorithms like DAPO explicitly disable KL penalty.
    """

    def __init__(self):
        super().__init__(0.0)

    def update(self, current_kl: float, step: int) -> float:
        """Always return 0 - no KL penalty."""
        self.stats.update(current_kl)
        return 0.0


def create_kl_controller(
    controller_type: str,
    init_kl_coef: float = 0.1,
    target_kl: Optional[float] = None,
    total_steps: int = 10000,
    **kwargs,
) -> KLController:
    """
    Factory function for creating KL controllers.

    Args:
        controller_type: Type of controller ("fixed", "adaptive", "linear", "cosine", "warmup", "hybrid", "none")
        init_kl_coef: Initial KL coefficient
        target_kl: Target KL for adaptive controllers
        total_steps: Total training steps for scheduled controllers
        **kwargs: Additional controller-specific parameters

    Returns:
        KL controller instance
    """
    controller_type = controller_type.lower()

    if controller_type == "fixed":
        return FixedKLController(kl_coef=init_kl_coef)

    elif controller_type == "adaptive":
        return AdaptiveKLController(
            init_kl_coef=init_kl_coef,
            target_kl=target_kl or 0.01,
            horizon=total_steps,
            **kwargs,
        )

    elif controller_type == "linear":
        return LinearKLScheduler(
            init_kl_coef=init_kl_coef,
            final_kl_coef=kwargs.get("final_kl_coef", init_kl_coef * 0.1),
            total_steps=total_steps,
        )

    elif controller_type == "cosine":
        return CosineKLScheduler(
            init_kl_coef=init_kl_coef,
            min_kl_coef=kwargs.get("min_kl_coef", init_kl_coef * 0.1),
            total_steps=total_steps,
            **kwargs,
        )

    elif controller_type == "warmup":
        return WarmupKLScheduler(
            init_kl_coef=0.0,
            peak_kl_coef=init_kl_coef,
            final_kl_coef=kwargs.get("final_kl_coef", init_kl_coef * 0.5),
            warmup_steps=kwargs.get("warmup_steps", total_steps // 10),
            total_steps=total_steps,
        )

    elif controller_type == "hybrid":
        return HybridKLController(
            init_kl_coef=init_kl_coef,
            target_kl=target_kl or 0.01,
            total_steps=total_steps,
            **kwargs,
        )

    elif controller_type == "none" or controller_type == "disabled":
        return NoKLController()

    else:
        raise ValueError(f"Unknown KL controller type: {controller_type}")
