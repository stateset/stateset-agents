"""Run contracts of the dedicated entry points (``train_with_dapo``/``gepo``/
``vapo``) on the spine's ``run_dedicated_loop``.

A tiny GPT-2 stands in for the model manager and canned rollouts for
generation, so each test runs the real trainers end to end.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from stateset_agents.core.errors import ConfigurationError  # noqa: E402
from stateset_agents.training import (  # noqa: E402
    dapo_entrypoints,
    dapo_trainer,
    gepo_trainer,
    vapo_entrypoints,
    vapo_trainer,
)
from stateset_agents.training.dapo_config import DAPOConfig  # noqa: E402
from stateset_agents.training.gepo_trainer import GEPOConfig  # noqa: E402
from stateset_agents.training.vapo_config import VAPOConfig  # noqa: E402
from tests.unit.test_dedicated_trainer_nonfinite_grads import (  # noqa: E402
    _gepo,
    _gepo_rollouts,
    _model,
    _reward,
    _token_rollouts,
)

# Real (tiny) trainers run end to end: allow for a loaded machine.
pytestmark = pytest.mark.timeout(240)

ALGORITHMS = ("dapo", "gepo", "vapo")
PROMPTS = [f"prompt {i}" for i in range(12)]


def _config(name: str, **overrides: Any) -> Any:
    values: dict[str, Any] = {
        "model_name": "gpt2",
        "group_size": 3,
        "per_device_train_batch_size": 2,
        "num_episodes": 2,
        "num_gradient_updates": 1,
        "save_steps": 100,
        "warmup_ratio": 0.0,
        "bf16": False,
        "report_to": "none",
    }
    if name == "vapo":
        values["value_warmup_steps"] = 1
    values.update(overrides)
    return {"dapo": DAPOConfig, "gepo": GEPOConfig, "vapo": VAPOConfig}[name](**values)


class _Run:
    """Records what one entry-point run generated, saved and loaded."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.saves: list[str] = []
        self.loaded: list[str] = []
        self.configs: list[Any] = []


def _harness(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    rollouts: Callable[[str], list[dict[str, Any]]] | None = None,
) -> tuple[Callable[..., Any], _Run]:
    """Patch ``name``'s model manager, generation and saving; return the
    entry point and the run record."""
    run = _Run()
    module, trainer_cls, manager_name, entry = {
        "dapo": (
            dapo_trainer,
            dapo_trainer.DAPOTrainer,
            "DAPOModelManager",
            dapo_entrypoints.train_with_dapo,
        ),
        "gepo": (
            gepo_trainer,
            gepo_trainer.GEPOTrainer,
            "GEPOModelManager",
            gepo_trainer.train_with_gepo,
        ),
        "vapo": (
            vapo_trainer,
            vapo_trainer.VAPOTrainer,
            "VAPOModelManager",
            vapo_entrypoints.train_with_vapo,
        ),
    }[name]

    class Manager:
        def __init__(self, config: Any) -> None:
            self.config = config
            self.ref_model = None

        def load_model_and_tokenizer(self) -> tuple[Any, Any]:
            run.loaded.append(self.config.model_name)
            run.configs.append(self.config)
            return _model(), None

    def sampled(prompt: str) -> list[dict[str, Any]]:
        run.prompts.append(prompt)
        if rollouts is not None:
            return rollouts(prompt)
        # torch-sampled completions, so torch seeding shows in the metrics
        canned = (
            _gepo_rollouts(prompt, 3) if name == "gepo" else _token_rollouts(prompt)
        )
        order = torch.randperm(len(canned)).tolist()
        return [canned[i] for i in order[: 2 + int(torch.randint(2, ()).item())]]

    async def generate(self: Any, prompt: str, *args: Any) -> list[dict[str, Any]]:
        return sampled(prompt)

    def save_checkpoint(self: Any, path: str) -> None:
        run.saves.append(path)

    monkeypatch.setattr(module, manager_name, Manager)
    monkeypatch.setattr(trainer_cls, "generate_group_responses", generate)
    monkeypatch.setattr(trainer_cls, "save_checkpoint", save_checkpoint)
    return entry, run


@pytest.mark.parametrize("config_cls", [GEPOConfig, VAPOConfig], ids=["gepo", "vapo"])
@pytest.mark.parametrize("group_size", [1, 0, True, 2.0])
def test_group_configs_need_two_responses_per_group(
    config_cls: Any, group_size: Any
) -> None:
    """GEPO and VAPO drop groups of fewer than two responses, so a smaller
    group_size trained nothing, silently."""
    with pytest.raises(ConfigurationError, match="group_size"):
        config_cls(group_size=group_size).validate_for_training()


@pytest.mark.asyncio
async def test_gepo_reports_a_step_without_valid_groups() -> None:
    """The no-valid-group step returns every key a normal step reports."""
    full = await _gepo().train_step(["q"])
    trainer = _gepo()

    async def single(prompt: str, group_size: int) -> list[dict[str, Any]]:
        return _gepo_rollouts(prompt, group_size)[:1]

    trainer.generate_group_responses = single
    empty = await trainer.train_step(["q"])
    assert set(empty) == set(full)
    assert empty["optimizer_updates"] == 0
    assert empty["gepo_coefficient_mean"] == 1.0
    assert empty["global_step"] == trainer.global_step


@pytest.mark.asyncio
async def test_train_with_gepo_survives_a_step_without_valid_groups(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Its report indexed ``gepo_coefficient_mean``, which the no-valid-group
    step did not return: a KeyError ended the run."""
    entry, run = _harness(
        monkeypatch, "gepo", rollouts=lambda p: _gepo_rollouts(p, 3)[:1]
    )
    _model_, _tokenizer, history = await entry(
        model_name="gpt2",
        reward_fn=_reward,
        train_prompts=PROMPTS,
        config=_config("gepo", logging_steps=1),
        output_dir=str(tmp_path),
    )
    assert [os.path.basename(path) for path in run.saves] == ["final"]
    assert len(run.prompts) == 2 * 2


async def _seeded_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, name: str, seed: int
) -> tuple[list[str], dict[str, list[float]]]:
    entry, run = _harness(monkeypatch, name)
    # Leftover global state must not leak into a seeded run.
    torch.manual_seed(1000 + seed)
    np.random.seed(1000 + seed)
    _model_, _tokenizer, history = await entry(
        model_name="gpt2",
        reward_fn=_reward,
        train_prompts=PROMPTS,
        config=_config(name, seed=seed),
        output_dir=str(tmp_path / f"{name}-{seed}"),
    )
    return run.prompts, history


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ALGORITHMS)
async def test_config_seed_fixes_batch_order_and_step_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, name: str
) -> None:
    """``config.seed`` was ignored: batches came from the global NumPy
    generator and torch was never seeded."""
    first = await _seeded_run(monkeypatch, tmp_path, name, seed=3)
    again = await _seeded_run(monkeypatch, tmp_path, name, seed=3)
    other = await _seeded_run(monkeypatch, tmp_path, name, seed=4)
    assert first == again
    assert first[0] != other[0]
    assert first[1] != other[1]


@pytest.mark.asyncio
async def test_vapo_value_warmup_draws_prompts_from_the_config_seed() -> None:
    from tests.unit.test_dedicated_trainer_nonfinite_grads import _vapo

    async def warmup_prompts(global_seed: int, seed: int) -> list[str]:
        trainer = _vapo(seed=seed, value_warmup_steps=6)
        trainer.value_warmup_complete = False
        drawn: list[str] = []

        async def generate(prompt: str) -> list[dict[str, Any]]:
            drawn.append(prompt)
            return _token_rollouts(prompt)

        trainer.generate_group_responses = generate
        np.random.seed(global_seed)
        await trainer.train_step(PROMPTS)
        return drawn

    first = await warmup_prompts(global_seed=0, seed=5)
    assert len(first) == 6
    assert await warmup_prompts(global_seed=1, seed=5) == first
    assert await warmup_prompts(global_seed=0, seed=6) != first


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ALGORITHMS)
async def test_arguments_override_the_config_and_the_run_records_them(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    name: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The model manager loaded ``config.model_name`` while the entry point
    logged (and stub-checked) the ``model_name`` argument, and checkpoints
    went to the ``output_dir`` argument while the config (logged to W&B)
    named another directory."""
    entry, run = _harness(monkeypatch, name)
    config = _config(name, output_dir=str(tmp_path / "from-config"))
    config.model_name = "config-model"
    await entry(
        model_name="argument-model",
        reward_fn=_reward,
        train_prompts=PROMPTS,
        config=config,
        output_dir=str(tmp_path / "from-argument"),
    )
    assert run.loaded == ["argument-model"]
    effective = run.configs[0]
    assert effective.output_dir == str(tmp_path / "from-argument")
    assert {os.path.dirname(path) for path in run.saves} == {
        str(tmp_path / "from-argument")
    }
    # the caller's config is not mutated
    assert (config.model_name, config.output_dir) == (
        "config-model",
        str(tmp_path / "from-config"),
    )
    assert "overrides config.model_name" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ALGORITHMS)
async def test_without_an_output_dir_argument_the_config_decides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, name: str
) -> None:
    entry, run = _harness(monkeypatch, name)
    await entry(
        model_name="gpt2",
        reward_fn=_reward,
        train_prompts=PROMPTS,
        config=_config(name, output_dir=str(tmp_path / "from-config")),
    )
    assert {os.path.dirname(path) for path in run.saves} == {
        str(tmp_path / "from-config")
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ALGORITHMS)
async def test_without_a_config_the_arguments_build_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, name: str
) -> None:
    from stateset_agents.training.spine.loop import DedicatedRun

    entry, run = _harness(monkeypatch, name)
    loops: list[dict[str, Any]] = []

    async def loop(**kwargs: Any) -> DedicatedRun:
        loops.append(kwargs)
        return DedicatedRun()

    module = {"dapo": dapo_entrypoints, "gepo": gepo_trainer, "vapo": vapo_entrypoints}
    monkeypatch.setattr(module[name], "run_dedicated_loop", loop)
    monkeypatch.chdir(tmp_path)
    await entry(model_name="gpt2", reward_fn=_reward, train_prompts=PROMPTS)
    assert run.loaded == ["gpt2"]
    assert run.configs[0].output_dir == loops[0]["output_dir"] == f"./outputs/{name}"
    assert (tmp_path / "outputs" / name).is_dir()


class _AbortAfter:
    """A callback that records ``on_step_end`` and aborts after ``steps``."""

    def __init__(self, steps: int) -> None:
        self.steps = steps
        self.seen: list[tuple[int, dict[str, Any]]] = []
        self.should_abort = False
        self.abort_reason = ""

    def on_step_end(self, step: int, metrics: dict[str, Any]) -> None:
        self.seen.append((step, dict(metrics)))
        if len(self.seen) >= self.steps:
            self.should_abort = True
            self.abort_reason = f"stop after {self.steps}"


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ALGORITHMS)
async def test_callbacks_see_every_step_and_may_abort(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, name: str
) -> None:
    """Only ``train_with_gspo`` accepted callbacks."""
    entry, run = _harness(monkeypatch, name)
    callback = _AbortAfter(1)
    await entry(
        model_name="gpt2",
        reward_fn=_reward,
        train_prompts=PROMPTS,
        config=_config(name, num_episodes=3, save_steps=1),
        output_dir=str(tmp_path),
        callbacks=[callback],
    )
    assert [step for step, _ in callback.seen] == [0]
    assert "policy_loss" in callback.seen[0][1] or "warmup_value_loss" in (
        callback.seen[0][1]
    )
    # the step's checkpoint precedes the callbacks; the final save follows
    assert [os.path.basename(path) for path in run.saves] == ["checkpoint-1", "final"]


@pytest.mark.asyncio
async def test_train_with_gspo_token_takes_callbacks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from types import SimpleNamespace

    from stateset_agents.training import gspo_token_trainer as tok
    from stateset_agents.training import gspo_trainer
    from stateset_agents.training.gspo_config import GSPOConfig

    saves: list[str] = []

    class Manager:
        ref_model = None

        def __init__(self, config: Any) -> None:
            pass

        def load_model_and_tokenizer(self) -> tuple[Any, Any]:
            return object(), object()

    class FakeTokenTrainer:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def train_step_token_level(
            self, queries: Any, num_groups: int = 1
        ) -> dict[str, Any]:
            return {"policy_loss": 0.5}

        def save_model(self, path: str) -> None:
            saves.append(os.path.basename(path))

    monkeypatch.setattr(gspo_trainer, "GSPOModelManager", Manager)
    monkeypatch.setattr(tok, "GSPOTokenTrainer", FakeTokenTrainer)
    callback = _AbortAfter(2)
    agent = SimpleNamespace(model=None, tokenizer=None)
    result = await tok.train_with_gspo_token(
        config=GSPOConfig(
            output_dir=str(tmp_path), report_to="none", num_outer_iterations=5
        ),
        agent=agent,
        environment=SimpleNamespace(scenarios=[]),
        reward_model=SimpleNamespace(),
        train_queries=["q"],
        callbacks=[callback],
    )
    assert [step for step, _ in callback.seen] == [0, 1]
    assert saves == ["final_model"]
    assert result._training_aborted == "stop after 2"
