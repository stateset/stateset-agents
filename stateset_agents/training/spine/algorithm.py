"""One algorithm = one :class:`AlgorithmSpec` run by the shared spine step.

The prompt-group trainers (GSPO, GSPO-token, DAPO, GEPO) used to each own a
``train_step`` that sampled, scored, built update items, ran the update
passes and assembled metrics. That step is now :func:`run_algorithm_step`;
what differs between algorithms is declared in an :class:`AlgorithmSpec`:

* **scoring** -- ``score(trainer, prompt, response, context)``, one sampled
  reply's reward through a spine scorer;
* **loss / objective** -- the native ``PolicyObjective`` preset
  (``objective`` + ``objective_options``) and ``group_loss``, one update
  item's loss in one pass;
* **sampling** -- ``rollouts``, the step's update items (sampled, scored,
  advantages and frozen old-policy log-probs) plus whatever the metrics need;
* **update** -- ``update_mode``, ``passes`` and ``update_options`` for the
  spine :class:`UpdateLoop`, and the ``counter`` the commits use;
* **metrics** -- ``metrics``, the step's metrics from its rollouts and passes.

A trainer mixes in :class:`AlgorithmTrainer` and names its spec in
``algorithm``; it keeps its constructor (model, optimizer, scheduler) and the
methods the hooks call, so subclasses and tests can still replace them. A new
algorithm is a spec, a small trainer class and a thin entry point on
:func:`~stateset_agents.training.spine.loop.run_dedicated_loop`.
:mod:`~stateset_agents.training.spine.presets` lists the shipped specs.

Importing this module does not import ``torch``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..config import config_value
from .update import (
    OptimizerStep,
    StepCounter,
    UpdateLoop,
    UpdateMode,
    UpdatePasses,
    nonfinite_grad_settings,
)


@dataclass
class StepRollouts:
    """One step's rollouts: the update items and the spec's step data."""

    #: One update item per kept prompt group, in prompt order.
    items: list[Any] = field(default_factory=list)
    #: What the spec's ``group_loss``/``metrics`` share for this step
    #: (rewards, pass statistics, ...); the spec decides its type.
    info: Any = None


def _no_options(trainer: Any, rollouts: StepRollouts) -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class AlgorithmSpec:
    """Everything one prompt-group RL algorithm adds to the spine step."""

    #: Registry name (``spine.presets``).
    name: str
    #: Native ``PolicyObjective`` preset, used when ``config.objective`` is
    #: unset (:meth:`AlgorithmTrainer._resolve_algorithm_objective`).
    objective: str
    #: ``resolve_objective`` keyword arguments from the config (clip bounds,
    #: supported ratios, aggregation, ...).
    objective_options: Callable[[Any], Mapping[str, Any]]
    #: How ``global_step`` advances on a commit (spine ``StepCounter``).
    counter: StepCounter
    #: ``"chunked"`` (one backward per item) or ``"single_graph"``.
    update_mode: UpdateMode
    #: Update passes per rollout batch, from the config.
    passes: Callable[[Any], int]
    #: One reply's reward: ``score(trainer, prompt, response, context)``.
    score: Callable[[Any, str, str, Mapping[str, Any]], Awaitable[float]]
    #: The step's rollouts: ``rollouts(trainer, prompts)``.
    rollouts: Callable[[Any, list[Any]], Awaitable[StepRollouts]]
    #: One item's loss in one pass: ``group_loss(trainer, rollouts, pass,
    #: index, item)``.
    group_loss: Callable[[Any, StepRollouts, int, int, Any], Any]
    #: The step's metrics; ``passes`` is ``None`` when the step had nothing
    #: to update and ``update_empty`` is False.
    metrics: Callable[[Any, StepRollouts, UpdatePasses | None], dict[str, Any]]
    #: Extra :meth:`UpdateLoop.run` arguments (``shares``, ``reduce``,
    #: ``initial_loss``, ``commit``, ``on_pass_start``).
    update_options: Callable[[Any, StepRollouts], dict[str, Any]] = _no_options
    #: Run the update passes even without items (each pass then reports a
    #: zero loss and nothing commits); otherwise such a step skips them.
    update_empty: bool = False
    #: Prefix of the commit's error messages; defaults to ``name.upper()``.
    label: str = ""

    @property
    def commit_label(self) -> str:
        return self.label or self.name.upper()


async def run_algorithm_step(
    trainer: Any, spec: AlgorithmSpec, prompts: list[Any]
) -> dict[str, Any]:
    """One ``train_step`` of ``spec`` on ``trainer``.

    The policy enters train mode, ``spec.rollouts`` samples and scores the
    step, then ``spec.passes(config)`` update passes run through the spine
    :class:`UpdateLoop`, each committed by ``trainer._build_optimizer_step()``
    (called as a bound method, so a subclass may replace it), and
    ``spec.metrics`` reports the step. A step without items skips the passes
    unless ``spec.update_empty``.
    """
    trainer.model.train()
    rollouts = await spec.rollouts(trainer, prompts)
    if not rollouts.items and not spec.update_empty:
        return spec.metrics(trainer, rollouts, None)
    passes = UpdateLoop(
        trainer._build_optimizer_step(),
        passes=spec.passes(trainer.config),
        mode=spec.update_mode,
    ).run(
        rollouts.items,
        lambda pass_index, index, item: spec.group_loss(
            trainer, rollouts, pass_index, index, item
        ),
        **spec.update_options(trainer, rollouts),
    )
    return spec.metrics(trainer, rollouts, passes)


class AlgorithmTrainer:
    """What an :class:`AlgorithmSpec` trainer shares: objective resolution,
    the spine commit, rollout-engine sync and scoring.

    The host provides ``config``, ``model``, ``optimizer`` and ``scheduler``
    and may override :meth:`_rollout_engine`.
    """

    #: The algorithm this trainer runs.
    algorithm: ClassVar[AlgorithmSpec]

    config: Any
    model: Any
    optimizer: Any
    scheduler: Any
    #: Outcome of the last rollout-engine sync: True/False, or ``None`` when
    #: nothing was attempted.
    last_rollout_sync: bool | None = None

    def _resolve_algorithm_objective(self, **overrides: Any) -> Any:
        """The configured ``PolicyObjective`` (``config.objective``, else the
        spec's native preset) with the spec's options."""
        from .. import objectives

        spec = self.algorithm
        options = dict(spec.objective_options(self.config))
        options.update(overrides)
        return objectives.resolve_objective(
            self.config,
            spec.objective,
            max_completion_length=int(self.config.max_completion_length),
            **options,
        )

    def _rollout_engine(self) -> Any:
        """The vLLM engine (``VLLMEngineMixin``) to keep on-policy, if any."""
        return None

    def _build_optimizer_step(self) -> OptimizerStep:
        """The spine commit for one update pass.

        Gradients must be finite (``config.nonfinite_grad_policy`` raises or
        skips the pass otherwise), the policy is clipped to ``max_grad_norm``,
        then the optimizer and the LR scheduler step once and the rollout
        engine is synced (:meth:`_sync_rollout_backend`). ``global_step``
        follows the spec's ``counter``.
        """
        from ..trainer_utils import require_torch

        params = list(self.model.parameters())
        return OptimizerStep(
            self,
            require_torch(),
            optimizers=[self.optimizer],
            schedulers=[self.scheduler],
            clip_groups=[params] if params else [],
            max_grad_norm=self.config.max_grad_norm,
            counter=self.algorithm.counter,
            **nonfinite_grad_settings(self.config),
            named_modules=[self.model],
            on_commit=lambda: self._sync_rollout_backend(),
            label=self.algorithm.commit_label,
        )

    def _sync_rollout_backend(self) -> None:
        """Keep the rollout engine on-policy after a committed update.

        ``last_rollout_sync`` is True/False for the sync outcome and ``None``
        when nothing was attempted (``config.rollout_sync`` off or no engine
        in use); a failed sync leaves the engine stale and training continues
        (``VLLMEngineMixin.sync_vllm_weights``).
        """
        self.last_rollout_sync = None
        if not bool(config_value(self.config, "rollout_sync")):
            return
        engine = self._rollout_engine()
        if engine is not None:
            self.last_rollout_sync = engine.sync_vllm_weights(self.model)

    async def _score_reply(
        self, prompt: str, response: str, context: Mapping[str, Any] | None = None
    ) -> float:
        """One sampled reply's reward through the spec's scorer."""
        return await self.algorithm.score(self, prompt, response, context or {})


__all__ = [
    "AlgorithmSpec",
    "AlgorithmTrainer",
    "StepRollouts",
    "run_algorithm_step",
]
