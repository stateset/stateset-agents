"""
Early abort callback for the autonomous research loop.

Monitors training metrics mid-experiment and signals abort if training
is clearly failing — preventing wasted wall-clock time on diverging
loss, NaN gradients, or stalled plateaus.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)


class EarlyAbortCallback:
    """Monitors training steps and signals when to abort early.

    Checks for:
    1. NaN loss — training has blown up
    2. Loss explosion — gradient blow-up (loss jumps >50x in one step)
    3. Plateau — loss hasn't improved for N steps (waste of time)

    Usage:
        callback = EarlyAbortCallback()
        # During training, call on each step:
        callback.on_step_end(step=42, metrics={"loss": 0.5})
        if callback.should_abort:
            raise RuntimeError(callback.abort_reason)
    """

    def __init__(
        self,
        *,
        loss_explosion_factor: float = 50.0,
        plateau_window: int = 30,
        plateau_min_relative_change: float = 0.001,
        plateau_min_steps: int = 50,
        nan_patience: int = 3,
    ):
        self.loss_explosion_factor = loss_explosion_factor
        self.plateau_window = plateau_window
        self.plateau_min_relative_change = plateau_min_relative_change
        self.plateau_min_steps = plateau_min_steps
        self.nan_patience = nan_patience

        self.loss_history: deque[float] = deque(maxlen=max(100, plateau_window * 2))
        self.nan_count = 0
        self.step_count = 0
        self.should_abort = False
        self.abort_reason: str | None = None

    def on_step_end(self, step: int | None = None, metrics: dict[str, Any] | None = None, **_: Any) -> None:
        """Called after each training step. Check for abort conditions."""
        if self.should_abort or metrics is None:
            return

        self.step_count += 1
        loss = metrics.get("loss")
        if loss is None:
            return

        # Check 1: NaN loss
        if math.isnan(loss) or math.isinf(loss):
            self.nan_count += 1
            if self.nan_count >= self.nan_patience:
                self._trigger_abort(
                    f"NaN/Inf loss detected {self.nan_count} times "
                    f"(step {self.step_count})"
                )
            return

        self.nan_count = 0  # Reset on valid loss

        # Check 2: Loss explosion
        if (
            len(self.loss_history) > 0
            and self.loss_history[-1] > 0
            and loss > self.loss_history[-1] * self.loss_explosion_factor
        ):
            self._trigger_abort(
                f"Loss explosion at step {self.step_count}: "
                f"{self.loss_history[-1]:.4f} → {loss:.4f} "
                f"({loss / self.loss_history[-1]:.0f}x increase)"
            )
            return

        self.loss_history.append(loss)

        # Check 3: Plateau (only after enough steps)
        if (
            self.step_count >= self.plateau_min_steps
            and len(self.loss_history) >= self.plateau_window
        ):
            window = list(self.loss_history)[-self.plateau_window :]
            first_half = sum(window[: len(window) // 2]) / (len(window) // 2)
            second_half = sum(window[len(window) // 2 :]) / (
                len(window) - len(window) // 2
            )

            if first_half != 0:
                relative_change = abs(second_half - first_half) / abs(first_half)
                if relative_change < self.plateau_min_relative_change:
                    self._trigger_abort(
                        f"Training plateau at step {self.step_count}: "
                        f"loss change < {self.plateau_min_relative_change:.4f} "
                        f"over last {self.plateau_window} steps"
                    )

    def on_train_start(self, *args: Any, **kwargs: Any) -> None:
        """Reset state for a new training run."""
        self.loss_history.clear()
        self.nan_count = 0
        self.step_count = 0
        self.should_abort = False
        self.abort_reason = None

    def _trigger_abort(self, reason: str) -> None:
        self.should_abort = True
        self.abort_reason = reason
        logger.warning("Early abort triggered: %s", reason)
