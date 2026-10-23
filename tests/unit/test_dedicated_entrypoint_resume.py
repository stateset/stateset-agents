"""Dedicated convenience loops must reject resume before expensive setup."""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

import stateset_agents.training as public_training
from stateset_agents.training import (
    dapo_entrypoints,
    dapo_trainer,
    gepo_trainer,
    gspo_entrypoints,
    gspo_token_trainer,
    gspo_trainer,
    vapo_entrypoints,
    vapo_trainer,
)
from stateset_agents.training.dapo_config import DAPOConfig
from stateset_agents.training.gspo_config import GSPOConfig
from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer
from stateset_agents.training.vapo_config import VAPOConfig

ENTRYPOINTS = (
    "gspo",
    "gspo_token",
    "gepo",
    "dapo",
    "vapo",
    "gspo_customer_service",
    "dapo_reasoning",
)


class _UnexpectedSetup(AssertionError):
    """Fail safely if a rejected request reaches an expensive operation."""


@pytest.fixture
def setup_sentinels(tmp_path, monkeypatch):
    checkpoint = tmp_path / "existing-checkpoint"
    checkpoint.mkdir()
    calls = []

    def forbidden(name):
        def fail(*_args, **_kwargs):
            calls.append(name)
            raise _UnexpectedSetup(name)

        return MagicMock(side_effect=fail)

    guards = []
    for module, names in (
        (gspo_trainer, ("GSPOModelManager", "GSPOTrainer", "_require_wandb")),
        (gspo_token_trainer, ("GSPOTokenTrainer",)),
        (gepo_trainer, ("GEPOModelManager", "GEPOTrainer", "_require_wandb")),
        (dapo_trainer, ("DAPOModelManager", "DAPOTrainer", "_require_wandb")),
        (vapo_trainer, ("VAPOModelManager", "VAPOTrainer", "_require_wandb")),
        (gspo_entrypoints, ("MultiTurnAgent",)),
    ):
        for name in names:
            guard = forbidden(name)
            guards.append(guard)
            monkeypatch.setattr(module, name, guard)

    wandb_init = forbidden("wandb.init")
    guards.append(wandb_init)
    wandb = SimpleNamespace(init=wandb_init)
    for module in (gspo_trainer, gepo_trainer, dapo_trainer, vapo_trainer):
        monkeypatch.setattr(module, "wandb", wandb)
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    mkdir = forbidden("os.makedirs")
    guards.append(mkdir)
    monkeypatch.setattr(gspo_entrypoints.os, "makedirs", mkdir)
    model, tokenizer = object(), object()
    agent = SimpleNamespace(
        model=model,
        tokenizer=tokenizer,
        initialize=AsyncMock(side_effect=_UnexpectedSetup("agent.initialize")),
    )
    return SimpleNamespace(
        output=tmp_path / "must-not-be-created",
        checkpoint=checkpoint,
        missing=tmp_path / "missing-checkpoint",
        calls=calls,
        guards=guards,
        agent=agent,
        model=model,
        tokenizer=tokenizer,
    )


async def _invoke(name, settings, resume, *, omit=False):
    values = {
        "model_name": "fixture/model-with-no-download",
        "output_dir": str(settings.output),
        "report_to": "wandb",
        "wandb_project": "must-not-contact",
        "use_vllm": True,
    }
    if not omit:
        values["resume_from_checkpoint"] = resume
    if name == "gspo_customer_service":
        await gspo_entrypoints.train_customer_service_with_gspo(**values)
        return
    if name == "dapo_reasoning":
        await dapo_entrypoints.train_reasoning_with_dapo(
            math_problems=[{"problem": "one plus one", "answer": "two"}], **values
        )
        return

    config_type = {
        "gspo": GSPOConfig,
        "gspo_token": GSPOConfig,
        "gepo": gepo_trainer.GEPOConfig,
        "dapo": DAPOConfig,
        "vapo": VAPOConfig,
    }[name]
    config = config_type(**values)
    if omit:
        # A legacy duck-typed config really lacks the attribute, rather than
        # inheriting the dataclass's default None through class lookup.
        fields = config.to_dict()
        fields.pop("resume_from_checkpoint", None)
        config = SimpleNamespace(**fields, to_dict=lambda: dict(fields))
    if name in {"gspo", "gspo_token"}:
        function = (
            gspo_entrypoints.train_with_gspo
            if name == "gspo"
            else gspo_token_trainer.train_with_gspo_token
        )
        await function(
            config=config,
            agent=settings.agent,
            environment=SimpleNamespace(scenarios=[{"prompt": "fixture question"}]),
            reward_model=object(),
            train_queries=["fixture question"],
        )
    else:
        function = {
            "gepo": gepo_trainer.train_with_gepo,
            "dapo": dapo_entrypoints.train_with_dapo,
            "vapo": vapo_entrypoints.train_with_vapo,
        }[name]
        await function(
            model_name=values["model_name"],
            reward_fn=lambda *_args: 0.0,
            train_prompts=["fixture question"],
            config=config,
            output_dir=str(settings.output),
            use_wandb=True,
            wandb_project="must-not-contact",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize(
    "resume_value", ["existing", "missing", "", "  ", False, 0, True]
)
async def test_explicit_resume_is_rejected_before_any_setup(
    setup_sentinels, entrypoint, resume_value
):
    resume = resume_value
    if resume_value == "existing":
        resume = str(setup_sentinels.checkpoint)
    elif resume_value == "missing":
        resume = str(setup_sentinels.missing)
    with pytest.raises(ValueError, match="does not support resume_from_checkpoint"):
        await _invoke(entrypoint, setup_sentinels, resume)
    assert setup_sentinels.calls == []
    for guard in setup_sentinels.guards:
        guard.assert_not_called()
    setup_sentinels.agent.initialize.assert_not_awaited()
    assert setup_sentinels.agent.model is setup_sentinels.model
    assert setup_sentinels.agent.tokenizer is setup_sentinels.tokenizer
    assert not setup_sentinels.output.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize("omit", [False, True], ids=["none", "missing"])
async def test_fresh_configuration_still_reaches_existing_setup(
    setup_sentinels, entrypoint, omit
):
    with pytest.raises(_UnexpectedSetup):
        await _invoke(entrypoint, setup_sentinels, None, omit=omit)
    assert setup_sentinels.calls
    assert not setup_sentinels.output.exists()


def test_public_dedicated_aliases_share_the_guarded_entrypoints():
    assert gspo_trainer.train_with_gspo is gspo_entrypoints.train_with_gspo
    assert public_training.train_with_gspo is gspo_entrypoints.train_with_gspo
    assert (
        gspo_trainer.train_customer_service_with_gspo
        is gspo_entrypoints.train_customer_service_with_gspo
    )
    assert dapo_trainer.train_with_dapo is dapo_entrypoints.train_with_dapo
    assert public_training.train_with_dapo is dapo_entrypoints.train_with_dapo
    assert (
        dapo_trainer.train_reasoning_with_dapo
        is dapo_entrypoints.train_reasoning_with_dapo
    )
    assert (
        public_training.train_reasoning_with_dapo
        is dapo_entrypoints.train_reasoning_with_dapo
    )
    assert public_training.train_with_gepo is gepo_trainer.train_with_gepo


@pytest.mark.asyncio
async def test_native_multi_turn_gspo_objective_still_restores_explicit_resume(
    tmp_path,
):
    model = torch.nn.Linear(2, 1)
    agent = SimpleNamespace(model=model, tokenizer=None)
    callback = SimpleNamespace(
        fail_on_error=True,
        on_training_start=AsyncMock(),
        on_training_end=AsyncMock(),
    )
    trainer = MultiTurnGRPOTrainer(
        agent,
        SimpleNamespace(reset=AsyncMock(), step=AsyncMock()),
        config=SimpleNamespace(
            num_episodes=2,
            gradient_accumulation_steps=1,
            objective="gspo",
            resume_from_checkpoint=str(tmp_path / "native-checkpoint"),
        ),
        callbacks=[callback],
    )
    trainer._setup_scheduler = MagicMock()
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer._get_training_scenarios = MagicMock(return_value=[{}, {}])
    trainer._get_eval_scenarios = MagicMock(return_value=[])
    trainer.generate_trajectories = AsyncMock()
    trainer.save_checkpoint = AsyncMock()

    def restore(_path):
        trainer.current_epoch = 1
        if hasattr(trainer, "_next_episode"):
            trainer._next_episode = 2
        trainer.global_step = 7
        return True

    trainer.load_checkpoint = MagicMock(side_effect=restore)
    assert await trainer.train() is agent
    trainer.load_checkpoint.assert_called_once_with(
        trainer.config.resume_from_checkpoint
    )
    assert trainer.global_step == 7
    callback.on_training_start.assert_awaited_once()
    callback.on_training_end.assert_awaited_once()
    trainer.generate_trajectories.assert_not_awaited()
