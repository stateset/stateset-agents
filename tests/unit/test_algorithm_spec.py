"""``spine.AlgorithmSpec``: one algorithm is one spec on the shared step.

GSPO, GSPO-token, DAPO and GEPO each declare a spec (scoring, objective,
sampling, update and metrics hooks) that ``run_algorithm_step`` runs; a new
algorithm needs nothing more than a spec and a small trainer class.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from stateset_agents.training.spine import (  # noqa: E402
    ALGORITHM_NAMES,
    AlgorithmSpec,
    AlgorithmTrainer,
    StepRollouts,
    algorithm_spec,
    run_algorithm_step,
)
from stateset_agents.training.spine.update import UpdatePasses  # noqa: E402
from tests.unit.test_dedicated_trainer_nonfinite_grads import (  # noqa: E402
    _dapo,
    _gepo,
    _gspo,
    _gspo_classes,
    _gspo_step,
)


def test_the_registry_resolves_every_shipped_spec() -> None:
    from stateset_agents.training.dapo_trainer import DAPOTrainer
    from stateset_agents.training.gepo_trainer import GEPOTrainer
    from stateset_agents.training.gspo_token_trainer import GSPOTokenTrainer
    from stateset_agents.training.gspo_trainer import GSPOTrainer

    trainers = {
        "gspo": GSPOTrainer,
        "gspo_token": GSPOTokenTrainer,
        "dapo": DAPOTrainer,
        "gepo": GEPOTrainer,
    }
    assert set(ALGORITHM_NAMES) == set(trainers)
    for name, trainer in trainers.items():
        spec = algorithm_spec(name)
        assert spec.name == name and spec.objective == name
        assert trainer.algorithm is spec
    with pytest.raises(ValueError, match="unknown algorithm 'ppo'"):
        algorithm_spec("ppo")


def test_specs_declare_each_algorithms_update_contract() -> None:
    expected = {
        "gspo": ("none", "single_graph", True, "GSPO"),
        "gspo_token": ("none", "single_graph", True, "GSPO"),
        "dapo": ("train_steps", "chunked", False, "DAPO"),
        "gepo": ("train_steps", "single_graph", False, "GEPO"),
    }
    for name, contract in expected.items():
        spec = algorithm_spec(name)
        assert (
            spec.counter,
            spec.update_mode,
            spec.update_empty,
            spec.commit_label,
        ) == contract


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["dapo", "gepo", "gspo", "gspo_token"])
async def test_train_step_runs_the_trainers_spec(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    from stateset_agents.training import dapo_trainer, gepo_trainer, gspo_trainer

    module = {
        "dapo": dapo_trainer,
        "gepo": gepo_trainer,
        "gspo": gspo_trainer,
        "gspo_token": gspo_trainer,
    }[name]
    seen: list[str] = []
    real = module.run_algorithm_step

    async def spy(trainer: Any, spec: AlgorithmSpec, prompts: list[Any]) -> Any:
        seen.append(spec.name)
        return await real(trainer, spec, prompts)

    monkeypatch.setattr(module, "run_algorithm_step", spy)
    if name == "dapo":
        metrics = await _dapo(num_gradient_updates=1).train_step(["q"])
    elif name == "gepo":
        metrics = await _gepo().train_step(["q"])
    else:
        index = 0 if name == "gspo" else 1
        metrics = await _gspo_step(_gspo(_gspo_classes()[index]))
    assert seen == [name]
    assert metrics["optimizer_updates"] == 1.0


@pytest.mark.asyncio
async def test_the_rollout_engine_is_synced_only_for_spec_trainers_with_one() -> None:
    gepo = _gepo()
    await gepo.train_step(["q"])
    assert gepo._rollout_engine() is None and gepo.last_rollout_sync is None
    dapo = _dapo()
    assert dapo._rollout_engine() is dapo


# --- a new algorithm is a spec and a small trainer --------------------------


class _Toy(AlgorithmTrainer):
    """REINFORCE on a linear "policy" over three canned replies."""

    def __init__(self, spec: AlgorithmSpec) -> None:
        from types import SimpleNamespace

        self.algorithm = spec  # type: ignore[misc]
        self.config = SimpleNamespace(
            max_grad_norm=1.0, max_completion_length=8, rollout_sync=True
        )
        torch.manual_seed(0)
        self.model = torch.nn.Linear(3, 1, bias=False)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.5)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1)
        self.global_step = 0
        self.history: list[float] = []


async def _toy_score(trainer: _Toy, prompt: str, response: str, context: Any) -> float:
    return float(response == "right")


async def _toy_rollouts(trainer: _Toy, prompts: list[Any]) -> StepRollouts:
    replies = ["wrong", "right", "wrong"]
    rewards = [
        await trainer._score_reply(prompt, r) for prompt in prompts for r in replies
    ]
    return StepRollouts(items=[torch.tensor(rewards[:3])], info=rewards)


def _toy_loss(
    trainer: _Toy, rollouts: StepRollouts, _pass: int, _index: int, rewards: Any
) -> Any:
    logits = trainer.model(torch.eye(3)).squeeze(-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    return -((rewards - rewards.mean()) * log_probs).sum()


def _toy_metrics(
    trainer: _Toy, rollouts: StepRollouts, passes: UpdatePasses | None
) -> dict[str, Any]:
    assert passes is not None
    trainer.history.append(passes.losses[-1])
    return {"loss": passes.losses[-1], "optimizer_updates": passes.committed}


TOY_SPEC = AlgorithmSpec(
    name="toy",
    objective="grpo",
    objective_options=lambda config: {},
    counter="train_steps",
    update_mode="chunked",
    passes=lambda config: 2,
    score=_toy_score,
    rollouts=_toy_rollouts,
    group_loss=_toy_loss,
    metrics=_toy_metrics,
)


@pytest.mark.asyncio
async def test_a_new_algorithm_is_one_spec_on_the_shared_step() -> None:
    trainer = _Toy(TOY_SPEC)
    for _ in range(5):
        metrics = await run_algorithm_step(trainer, TOY_SPEC, ["q"])
        assert metrics["optimizer_updates"] == 2
    assert trainer.global_step == 5  # counter="train_steps"
    logits = trainer.model(torch.eye(3)).squeeze(-1)
    assert int(logits.argmax()) == 1  # learned to prefer the rewarded reply
    assert trainer.last_rollout_sync is None  # no rollout engine
    assert trainer._resolve_algorithm_objective().name == "grpo"


@pytest.mark.asyncio
async def test_a_step_without_items_skips_the_passes_unless_update_empty() -> None:
    async def nothing(trainer: Any, prompts: list[Any]) -> StepRollouts:
        return StepRollouts(info="empty")

    reported: list[Any] = []

    def metrics(trainer: Any, rollouts: StepRollouts, passes: Any) -> dict[str, Any]:
        reported.append(passes)
        return {}

    import dataclasses

    spec = dataclasses.replace(TOY_SPEC, rollouts=nothing, metrics=metrics)
    trainer = _Toy(spec)
    await run_algorithm_step(trainer, spec, ["q"])
    assert reported == [None] and trainer.global_step == 0

    spec = dataclasses.replace(spec, update_empty=True)
    trainer = _Toy(spec)
    await run_algorithm_step(trainer, spec, ["q"])
    assert reported[-1].losses == [0.0, 0.0] and reported[-1].committed == 0
    assert trainer.global_step == 1  # the train step still counts
