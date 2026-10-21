"""User-facing config paths must reach the trainer: nothing silently dropped.

Each case drives the real entry point (CLI command, ``train()``, GSPO
entrypoint, starter resolver, API schema) and inspects the config the trainer
would receive.
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import json
import re
from types import SimpleNamespace
from unittest import mock

import pytest
from typer.testing import CliRunner

from stateset_agents.cli import app

T = importlib.import_module("stateset_agents.training.train")


# --- 1. `train --config`: training section and profile precedence -------------------


def _write_config(tmp_path, training, profile="conservative"):
    path = tmp_path / "cfg.json"
    path.write_text(
        json.dumps(
            {
                "profile": profile,
                "agent": {"model_name": "stub://x", "use_stub_model": True},
                "training": training,
            }
        )
    )
    return str(path)


def _invoke_train(args):
    captured: dict = {}

    async def fake_train(**kwargs):
        captured.update(kwargs)
        captured["cfg"] = T._build_training_config(
            profile=kwargs["profile"],
            num_episodes=kwargs["num_episodes"],
            config_overrides=kwargs["config_overrides"],
            resume_from_checkpoint=None,
            objective=None,
            objective_overrides=None,
            extra_config={},
        )

    with (
        mock.patch.object(T, "train", fake_train),
        mock.patch(
            "stateset_agents.core.agent.MultiTurnAgent.initialize", mock.AsyncMock()
        ),
    ):
        result = CliRunner().invoke(app, ["train", *args, "--no-dry-run"])
    return result, captured


def test_config_training_section_reaches_the_trainer(tmp_path):
    path = _write_config(
        tmp_path,
        {
            "num_episodes": 3,
            "learning_rate": 3e-5,
            "num_generations": 4,
            "beta": 0.05,
            "seed": 7,
        },
    )
    result, captured = _invoke_train(["--config", path])
    assert result.exit_code == 0, result.output
    cfg = captured["cfg"]
    assert (cfg.learning_rate, cfg.num_generations, cfg.beta, cfg.seed) == (
        3e-5,
        4,
        0.05,
        7,
    )
    assert captured["num_episodes"] == 3
    assert captured["profile"] == "conservative"  # file profile without a flag


def test_explicit_profile_flag_beats_the_config_file(tmp_path):
    path = _write_config(tmp_path, {"num_episodes": 2})
    result, captured = _invoke_train(["--config", path, "--profile", "aggressive"])
    assert result.exit_code == 0, result.output
    assert captured["profile"] == "aggressive"


def test_unknown_training_key_is_a_validation_error(tmp_path):
    path = _write_config(tmp_path, {"learning_rat": 1e-5})
    result, _ = _invoke_train(["--config", path])
    assert result.exit_code == 2
    assert "learning_rat" in result.output


def test_cli_objective_overrides_file_objective(tmp_path):
    path = _write_config(tmp_path, {"objective": "dapo"})
    result, captured = _invoke_train(["--config", path, "--objective", "cispo"])
    assert result.exit_code == 0, result.output
    assert captured["cfg"].objective == "cispo"


# --- 2. offline / hybrid training ----------------------------------------------------


class _FakeOfflineTrainer:
    instances: list = []

    def __init__(self, config, **kwargs):
        self.config = config
        self.saved: list[str] = []
        _FakeOfflineTrainer.instances.append(self)

    async def pretrain_value_functions(self, *args, **kwargs):
        return {}

    async def train(self, num_epochs=None, dataset=None):
        return {}

    def save_checkpoint(self, path):
        self.saved.append(path)


def _run_offline(tmp_path, **kwargs):
    og = importlib.import_module("stateset_agents.experimental.offline_rl.offline_grpo")
    _FakeOfflineTrainer.instances.clear()
    agent = SimpleNamespace(model=mock.MagicMock(), tokenizer=mock.MagicMock())
    with mock.patch.object(og, "OfflineGRPOTrainer", _FakeOfflineTrainer):
        asyncio.run(
            T.train(
                agent=agent,
                environment=mock.MagicMock(),
                reward_fn=mock.MagicMock(),
                num_episodes=5,
                training_mode="offline",
                dataset=[1, 2],
                **kwargs,
            )
        )
    return agent, _FakeOfflineTrainer.instances[-1]


def test_offline_mode_maps_fields_the_offline_trainer_reads(tmp_path):
    _, trainer = _run_offline(
        tmp_path,
        max_grad_norm=0.3,
        num_epochs=2,
        seed=17,
        adam_beta1=0.8,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=3,
    )
    cfg = trainer.config
    assert cfg.max_grad_norm == 0.3
    assert cfg.seed == 17 and cfg.adam_beta1 == 0.8
    assert cfg.batch_size == 2 and cfg.gradient_accumulation_steps == 3
    assert cfg.num_epochs == 2


@pytest.mark.parametrize(
    "field,value",
    [("objective", "cispo"), ("clip_ratio", 0.1), ("advantage_normalization", False)],
)
def test_offline_mode_rejects_fields_it_would_ignore(tmp_path, field, value):
    with pytest.raises(ValueError, match=field):
        _run_offline(tmp_path, **{field: value})


def test_offline_mode_dispatches_checkpoint_save_to_requested_directory(tmp_path):
    save = str(tmp_path / "ckpt")
    _, trainer = _run_offline(tmp_path, save_path=save)
    assert trainer.saved == [save]


# --- 3. GSPO customer-service entrypoint --------------------------------------------


def _gspo_entry_config(**kwargs):
    entry = importlib.import_module("stateset_agents.training.gspo_entrypoints")
    captured = {}

    async def fake(config, **kw):
        captured["config"] = config

    with (
        mock.patch.object(entry, "train_with_gspo", fake),
        mock.patch.object(entry.MultiTurnAgent, "initialize", mock.AsyncMock()),
    ):
        asyncio.run(
            entry.train_customer_service_with_gspo(model_name="stub://x", **kwargs)
        )
    return captured["config"]


def test_gspo_entrypoint_runs_num_episodes_iterations():
    assert _gspo_entry_config(num_episodes=7).num_outer_iterations == 7


def test_gspo_entrypoint_explicit_outer_iterations_win():
    config = _gspo_entry_config(num_episodes=7, num_outer_iterations=3)
    assert config.num_outer_iterations == 3


def test_gspo_validate_warns_when_num_episodes_cannot_take_effect():
    from stateset_agents.training.gspo_config import GSPOConfig

    warned = [w for w in GSPOConfig(num_episodes=50).validate() if "num_episodes" in w]
    assert warned
    quiet = GSPOConfig(num_episodes=50, num_outer_iterations=50).validate()
    assert not [w for w in quiet if "num_episodes" in w]


# --- 4. starter quantization flags ---------------------------------------------------


def test_explicit_8bit_replaces_profile_4bit():
    from stateset_agents.training.kimi_k3_starter import get_kimi_k3_config

    config = get_kimi_k3_config(use_8bit=True)
    assert (config.use_8bit, config.use_4bit) == (True, False)


def test_explicit_4bit_and_8bit_conflict():
    from stateset_agents.training.kimi_k3_starter import get_kimi_k3_config

    with pytest.raises(ValueError, match="8"):
        get_kimi_k3_config(use_8bit=True, use_4bit=True)


# --- 5. batch_size alias -------------------------------------------------------------


def test_batch_size_alias_applies_once():
    from stateset_agents.training.config import (
        TrainingConfig,
        optimize_config_for_hardware,
    )

    config = TrainingConfig(batch_size=4)
    assert config.per_device_train_batch_size == 4
    assert (
        dataclasses.replace(config, per_device_train_batch_size=1)
    ).per_device_train_batch_size == 1
    assert (
        optimize_config_for_hardware(config, num_gpus=2).per_device_train_batch_size
        == 2
    )
    assert (
        optimize_config_for_hardware(
            config, gpu_memory_gb=6
        ).per_device_train_batch_size
        == 1
    )
    roundtrip = TrainingConfig.from_dict(config.to_dict())
    assert roundtrip.per_device_train_batch_size == 4


# --- 6. API profile validation -------------------------------------------------------


def _request(profile):
    from stateset_agents.api.schemas import TrainingRequest

    return TrainingRequest(
        agent_config={"model_name": "stub://x"},
        environment_scenarios=[{"id": "a"}],
        reward_config={},
        profile=profile,
    )


def test_api_profile_aliases_normalize():
    assert _request("speed").profile == "aggressive"
    assert _request(" Quality ").profile == "conservative"
    assert _request("balanced").profile == "balanced"


def test_api_rejects_unknown_profile():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="profile"):
        _request("turbo")


def test_api_profile_aliases_match_the_cli():
    from stateset_agents.api import schemas
    from stateset_agents.cli import TRAIN_PROFILE_ALIASES, TRAIN_PROFILE_CHOICES

    for alias, canonical in TRAIN_PROFILE_ALIASES.items():
        assert _request(alias).profile == canonical
    for name in TRAIN_PROFILE_CHOICES:
        assert _request(name).profile == name
    from stateset_agents.training.config import TrainingProfile

    assert schemas._TRAINING_PROFILES == {p.value for p in TrainingProfile}
    assert set(TRAIN_PROFILE_CHOICES) == schemas._TRAINING_PROFILES


# --- minor: create_config_from_args and flywheel loss knobs -------------------------


def test_create_config_from_args_honours_profile_for_tasks():
    from stateset_agents.training.config import (
        TrainingConfig,
        TrainingProfile,
        create_config_from_args,
        get_config_for_task,
    )

    tuned = create_config_from_args(
        "gpt2", task_type="tutoring", profile="conservative"
    )
    assert (
        tuned.learning_rate
        == TrainingConfig.from_profile(TrainingProfile.CONSERVATIVE).learning_rate
    )
    plain = create_config_from_args("gpt2", task_type="tutoring")
    assert plain.learning_rate == get_config_for_task("tutoring").learning_rate


@pytest.fixture
def flywheel_args(tmp_path, monkeypatch):
    train, validation = tmp_path / "train.json", tmp_path / "validation.json"
    train.write_text(json.dumps([{"prompt": "train", "expect": ["done"]}]))
    validation.write_text(json.dumps([{"prompt": "validation", "expect": ["done"]}]))

    class Executor:
        supported_job_kinds = {"rl"}

        def supports(self, kind):
            return kind == "rl"

        def submit(self, spec):
            spec.output_dir.mkdir(parents=True, exist_ok=True)
            (spec.output_dir / "rl_report.json").write_text(
                json.dumps({"best_round": 1, "best_eval": {"passed": 1, "total": 1}})
            )
            return spec

        def wait(self, spec):
            return SimpleNamespace(succeeded=True, logs=[])

    monkeypatch.setattr(
        "stateset_agents.cli_flywheel.get_executor", lambda name: Executor()
    )

    def build(algorithm):
        return [
            "flywheel",
            "--provider",
            "river",
            "--algorithm",
            algorithm,
            "--base-model",
            "test/model",
            "--harvest-prompts",
            str(train),
            "--eval-prompts",
            str(validation),
            "--output-root",
            str(tmp_path / f"out-{algorithm}"),
        ]

    return build


_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_PLAIN_TERMINAL = {
    "COLUMNS": "200",
    "NO_COLOR": "1",
    "TERM": "dumb",
    "FORCE_COLOR": "0",
}


@pytest.mark.parametrize(
    "algorithm,flag",
    [
        ("cispo", "--clip-low"),
        ("cispo", "--clip-high"),
        ("ppo", "--eps-max"),
        ("importance_sampling", "--eps-max"),
    ],
)
def test_flywheel_rejects_loss_knobs_for_other_algorithms(
    flywheel_args, algorithm, flag
):
    # CI sets FORCE_COLOR=1, which colours and wraps Rich's error panel.
    result = CliRunner().invoke(
        app, flywheel_args(algorithm) + [flag, "0.1"], env=_PLAIN_TERMINAL
    )
    assert result.exit_code != 0
    assert flag in " ".join(_ANSI.sub("", result.output).split())


@pytest.mark.parametrize(
    "algorithm,flag", [("cispo", "--eps-max"), ("ppo", "--clip-low")]
)
def test_flywheel_accepts_matching_loss_knobs(flywheel_args, algorithm, flag):
    result = CliRunner().invoke(app, flywheel_args(algorithm) + [flag, "0.3"])
    assert result.exit_code == 0, result.output
