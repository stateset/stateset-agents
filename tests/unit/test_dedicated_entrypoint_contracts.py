"""Run contracts of the dedicated entry points (``train_with_dapo``/``gepo``/
``vapo``) on the spine's ``run_dedicated_loop``.

A tiny GPT-2 stands in for the model manager and canned rollouts for
generation, so each test runs the real trainers end to end.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

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

ALGORITHMS = ("dapo", "gepo", "vapo")
PROMPTS = [f"prompt {i}" for i in range(12)]


def _config(name: str, **overrides: Any) -> Any:
    values: dict[str, Any] = {
        "model_name": "gpt2",
        "group_size": 3,
        "per_device_train_batch_size": 2,
        "num_episodes": 3,
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
    assert len(run.prompts) == 3 * 2
