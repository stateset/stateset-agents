"""Every native trainer's LR schedule spans the optimizer steps it really takes.

The schedulers step once per committed update pass, so a run of ``N`` outer
iterations with ``K`` passes takes ``N * K`` optimizer steps. GEPO and VAPO
sized their cosine over ``num_episodes * num_epochs`` (``N``), so with ``K=2``
Hugging Face's cosine reached zero halfway through the run and then climbed
back to the full learning rate; DAPO ignored ``lr_scheduler_type`` entirely.
The budget now lives in the spine (``spine.schedule``) and every trainer
uses it: the cosine is monotone non-increasing over the whole run and ends
at exactly zero.

The GRPO trainers (multi- and single-turn) are covered by
``test_grpo_scheduler_budget.py``, which runs their real ``train()`` loops.
"""

from __future__ import annotations

import asyncio
import importlib.util
import math
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

_spec = importlib.util.spec_from_file_location(
    "capture_step_goldens_for_schedules",
    Path(__file__).resolve().parents[2] / "scripts" / "capture_step_goldens.py",
)
assert _spec is not None and _spec.loader is not None
capture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(capture)

ITERATIONS = 4
PASSES = 2
LR = 1e-3


def cosine(step: int, total: int) -> float:
    return LR * 0.5 * (1.0 + math.cos(math.pi * min(1.0, step / total)))


class RateRecorder:
    """The learning rate each ``optimizer.step()`` actually used."""

    def __init__(self, optimizer: Any) -> None:
        self.optimizer = optimizer
        self.rates: list[float] = []
        optimizer.register_step_pre_hook(
            lambda opt, args, kwargs: self.rates.append(opt.param_groups[0]["lr"])
        )

    @property
    def final(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])


def assert_full_cosine(recorder: RateRecorder, steps: int, *, first: int = 0) -> None:
    assert len(recorder.rates) == steps - first
    assert recorder.rates == pytest.approx(
        [cosine(step, steps) for step in range(first, steps)], rel=1e-12, abs=1e-15
    )
    assert all(
        later <= earlier
        for earlier, later in zip(recorder.rates, recorder.rates[1:], strict=False)
    ), recorder.rates
    assert recorder.final == pytest.approx(0.0, abs=1e-15)


def _run(trainer: Any, steps: int, call: Any) -> None:
    with capture.deterministic_cpu():
        for _ in range(steps):
            asyncio.run(call())
    del trainer


def _gspo(token_level: bool) -> tuple[Any, Any]:
    from stateset_agents.training.gspo_config import GSPOConfig
    from stateset_agents.training.gspo_token_trainer import GSPOTokenTrainer
    from stateset_agents.training.gspo_trainer import GSPOTrainer

    cls = GSPOTokenTrainer if token_level else GSPOTrainer
    trainer = cls(
        config=GSPOConfig(
            model_name="gpt2",
            num_generations=len(capture.COMPLETIONS),
            num_outer_iterations=ITERATIONS,
            num_iterations=PASSES,
            learning_rate=LR,
            warmup_ratio=0.0,
            max_prompt_length=32,
            max_completion_length=32,
            bf16=False,
            report_to="none",
        ),
        model=capture.tiny_model(),
        tokenizer=capture.tokenizer(),
        agent=None,
        environment=None,
        reward_model=capture._Reward(),
        ref_model=None,
    )

    async def generate(prompt: str, count: int) -> list[tuple[str, float]]:
        return [(text, -5.0) for text, _ in capture.COMPLETIONS]

    trainer.generator.generate_group_responses = generate
    step = trainer.train_step_token_level if token_level else trainer.train_step
    return trainer, lambda: step(list(capture.PROMPTS), num_groups=2)


def _dapo() -> tuple[Any, Any]:
    from stateset_agents.training.dapo_trainer import DAPOConfig, DAPOTrainer

    trainer = DAPOTrainer(
        config=DAPOConfig(
            model_name="gpt2",
            group_size=len(capture.COMPLETIONS),
            num_gradient_updates=PASSES,
            num_episodes=ITERATIONS,
            learning_rate=LR,
            lr_scheduler_type="cosine",
            warmup_ratio=0.0,
            max_prompt_length=32,
            max_completion_length=32,
            bf16=False,
            report_to="none",
        ),
        model=capture.tiny_model(),
        tokenizer=None,
        reward_fn=lambda prompt, response: capture.digit_fraction(response),
    )

    async def generate(prompt: str) -> list[dict[str, Any]]:
        return capture._response_dicts(prompt)

    trainer.generate_group_responses = generate
    return trainer, lambda: trainer.train_step(list(capture.PROMPTS))


def _gepo() -> tuple[Any, Any]:
    from stateset_agents.training.gepo_trainer import GEPOConfig, GEPOTrainer

    trainer = GEPOTrainer(
        config=GEPOConfig(
            model_name="gpt2",
            group_size=len(capture.COMPLETIONS),
            num_gradient_updates=PASSES,
            num_episodes=ITERATIONS,
            learning_rate=LR,
            warmup_ratio=0.0,
            max_prompt_length=32,
            max_completion_length=32,
            bf16=False,
            report_to="none",
        ),
        model=capture.tiny_model(),
        tokenizer=None,
        reward_fn=lambda prompt, response: capture.digit_fraction(response),
    )

    async def generate(prompt: str, group_size: int) -> list[dict[str, Any]]:
        return capture._gepo_response_dicts(prompt)

    trainer.generate_group_responses = generate
    return trainer, lambda: trainer.train_step(list(capture.PROMPTS))


def _vapo(**overrides: Any) -> Any:
    from stateset_agents.training.vapo_trainer import VAPOConfig, VAPOTrainer

    values: dict[str, Any] = {
        "model_name": "gpt2",
        "group_size": len(capture.COMPLETIONS),
        "num_gradient_updates": PASSES,
        "num_episodes": ITERATIONS,
        "actor_learning_rate": LR,
        "critic_learning_rate": LR,
        "warmup_ratio": 0.0,
        "max_prompt_length": 32,
        "max_completion_length": 32,
        "bf16": False,
        "report_to": "none",
    }
    values.update(overrides)
    trainer = VAPOTrainer(
        config=VAPOConfig(**values),
        model=capture.tiny_model(),
        tokenizer=None,
        reward_fn=lambda prompt, response: capture.digit_fraction(response),
    )

    async def generate(prompt: str) -> list[dict[str, Any]]:
        return capture._response_dicts(prompt)

    trainer.generate_group_responses = generate
    return trainer


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: _gspo(False), id="gspo"),
        pytest.param(lambda: _gspo(True), id="gspo_token"),
        pytest.param(_dapo, id="dapo"),
        pytest.param(_gepo, id="gepo"),
    ],
)
def test_spec_trainer_cosine_spans_every_update_pass(build: Any) -> None:
    trainer, step = build()
    recorder = RateRecorder(trainer.optimizer)
    _run(trainer, ITERATIONS, step)
    assert_full_cosine(recorder, ITERATIONS * PASSES)
    assert trainer.scheduler.last_epoch == ITERATIONS * PASSES


def test_vapo_cosine_spans_every_update_pass() -> None:
    trainer = _vapo()
    trainer.value_warmup_complete = True
    actor, critic = RateRecorder(trainer.actor_optimizer), RateRecorder(
        trainer.critic_optimizer
    )
    _run(trainer, ITERATIONS, lambda: trainer.train_step(list(capture.PROMPTS)))
    assert_full_cosine(actor, ITERATIONS * PASSES)
    assert_full_cosine(critic, ITERATIONS * PASSES)


def test_vapo_warmup_iteration_holds_its_place_in_the_schedule() -> None:
    """The value-warmup iteration is one of ``num_episodes``: it advances the
    schedules by its ``K`` positions (as an empty batch always has), so a run
    still ends exactly at the schedule's final value."""
    trainer = _vapo(value_warmup_steps=1)
    actor = RateRecorder(trainer.actor_optimizer)
    _run(trainer, ITERATIONS, lambda: trainer.train_step(list(capture.PROMPTS)))
    assert trainer.value_warmup_complete
    assert_full_cosine(actor, ITERATIONS * PASSES, first=PASSES)
    assert trainer.actor_scheduler.last_epoch == ITERATIONS * PASSES
    assert trainer.critic_scheduler.last_epoch == ITERATIONS * PASSES


def test_vapo_value_warmup_trains_the_critic_at_its_base_rate() -> None:
    """With ``warmup_ratio > 0`` the schedule starts at 0; value warmup used to
    run at that scheduled rate and so never moved the value head."""
    trainer = _vapo(value_warmup_steps=2, warmup_ratio=0.5)
    before = [p.detach().clone() for p in trainer.value_head.parameters()]
    rates = RateRecorder(trainer.critic_optimizer)
    with capture.deterministic_cpu():
        asyncio.run(trainer.train_step(list(capture.PROMPTS)))
    assert rates.rates == [LR, LR]
    moved = any(
        not torch.equal(old, new)
        for old, new in zip(before, trainer.value_head.parameters(), strict=True)
    )
    assert moved


@pytest.mark.parametrize("kind", ["cosine", "linear", "constant"])
@pytest.mark.parametrize("warmup_ratio", [0.0, 0.25])
def test_schedule_is_monotone_after_warmup_and_clamped_past_the_end(
    kind: str, warmup_ratio: float
) -> None:
    from stateset_agents.training.spine.schedule import schedule_factor

    total = 8
    warmup = int(total * warmup_ratio)
    factors = [
        schedule_factor(step, kind=kind, warmup_steps=warmup, total_steps=total)
        for step in range(total + 6)
    ]
    if kind == "constant":
        assert factors == [1.0] * len(factors)
        return
    decay = factors[warmup:]
    assert all(
        later <= earlier for earlier, later in zip(decay, decay[1:], strict=False)
    )
    assert factors[total:] == [0.0] * 6  # held at zero, never climbs back


def test_budget_counts_iterations_times_passes() -> None:
    from stateset_agents.training.spine.schedule import optimizer_step_budget

    assert optimizer_step_budget(10, 2) == 20
    assert optimizer_step_budget(10, 1) == 10
    assert optimizer_step_budget(0, 3) == 3  # never an empty schedule
    with pytest.raises(ValueError):
        optimizer_step_budget(10, 0)


def test_build_lr_scheduler_rejects_unknown_kinds() -> None:
    from stateset_agents.training.spine.schedule import build_lr_scheduler

    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=LR)
    with pytest.raises(ValueError, match="lr_scheduler_type"):
        build_lr_scheduler(
            torch, optimizer, kind="polynomial", total_steps=4, warmup_ratio=0.0
        )
