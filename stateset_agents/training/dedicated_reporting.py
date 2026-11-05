"""Weights & Biases reporting for the dedicated entry points.

``train_with_gspo``/``gspo_token``/``dapo``/``gepo``/``vapo`` and the TRL
GRPO entry points each initialised, logged to and finished a W&B run with
their own copy of the same few lines, and the copies drifted: GEPO, VAPO and
GSPO-token skipped ``wandb.init`` when no project was set but still called
``wandb.log``, which raises after the first step. :class:`WandbRun` owns the
whole lifecycle:

* requested without a project, it warns once and stays inactive;
* requested with a project, it imports ``wandb`` (``ImportError`` naming the
  algorithm when missing) and initialises the run;
* :meth:`WandbRun.log` is a no-op unless the run is active;
* used as a context manager it always finishes the run, with
  ``exit_code=1`` when the body raised.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from types import TracebackType
from typing import Any

from .dedicated_deps import require_wandb

logger = logging.getLogger(__name__)


def reports_to_wandb(report_to: Any) -> bool:
    """Whether a ``report_to`` value (``"wandb"``, ``"wandb,tensorboard"``,
    a list of targets) includes W&B."""
    if not report_to:
        return False
    targets = report_to.split(",") if isinstance(report_to, str) else report_to
    return "wandb" in {str(target).strip() for target in targets}


def _config_dict(config: Any) -> Any:
    """``config.to_dict()`` for a config object, else ``config`` itself."""
    return config.to_dict() if hasattr(config, "to_dict") else config


class WandbRun:
    """One dedicated run's W&B reporting (see the module docstring)."""

    def __init__(
        self,
        algorithm: str,
        *,
        enabled: bool,
        project: str | None,
        config: Any,
        tags: Sequence[str] = (),
        name: str | None = None,
    ) -> None:
        self.algorithm = algorithm
        self.requested = bool(enabled)
        self.project = project or None
        self.config = config
        self.tags = list(tags)
        self.name = name
        self._wandb: Any = None

    @property
    def active(self) -> bool:
        """True between a successful :meth:`start` and :meth:`finish`."""
        return self._wandb is not None

    def start(self) -> None:
        """Initialise the run when logging was requested with a project."""
        if not self.requested or self.active:
            return
        if self.project is None:
            logger.warning(
                "%s: W&B logging was requested but no wandb_project was set; "
                "disabling W&B logging.",
                self.algorithm,
            )
            self.requested = False
            return
        wandb = require_wandb(self.algorithm.upper())
        wandb.init(
            project=self.project,
            name=self.name
            or f"{self.algorithm}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=_config_dict(self.config),
            tags=list(self.tags),
        )
        self._wandb = wandb

    def log(self, metrics: Mapping[str, Any], *, step: int) -> None:
        """Log ``metrics`` at ``step``; a no-op unless the run is active."""
        if self._wandb is not None:
            self._wandb.log(dict(metrics), step=step)

    def finish(self, *, failed: bool = False) -> None:
        """Close the run (``exit_code=1`` when ``failed``); idempotent."""
        wandb, self._wandb = self._wandb, None
        if wandb is None:
            return
        if failed:
            wandb.finish(exit_code=1)
        else:
            wandb.finish()

    def __enter__(self) -> WandbRun:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.finish(failed=exc_type is not None)


__all__ = ["WandbRun", "reports_to_wandb"]
