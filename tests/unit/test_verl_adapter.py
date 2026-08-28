"""Unit tests for the fail-closed verl training adapter."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import stateset_agents.training.adapters.verl as verl_module
from stateset_agents.training.adapters.verl import (
    VerlConfigError,
    build_verl_command,
    verify_verl_dataset,
    verl_backend,
)
from stateset_agents.training.backends import TrainingExperiment


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _request(tmp_path: Path, **updates: object) -> dict[str, object]:
    reward = tmp_path / "reward.py"
    reward.write_text(
        "def compute_score(*args, **kwargs): return 1.0\n", encoding="utf-8"
    )
    dataset = tmp_path / "train.parquet"
    dataset.write_bytes(b"PAR1-test-fixture")
    request: dict[str, object] = {
        "algorithm": "grpo",
        "model": "Qwen/example",
        "model_revision": "a" * 40,
        "dataset_uri": dataset.as_uri(),
        "dataset_sha256": _digest(dataset),
        "output_dir": str(tmp_path / "output"),
        "seed": 42,
        "config": {
            "learning_rate": 1e-6,
            "rollout_samples": 8,
            "rollout_engine": "vllm",
            "num_nodes": 1,
            "gpus_per_node": 4,
            "deterministic": True,
        },
        "environment": {"type": "single_turn"},
        "reward": {
            "type": "python",
            "path": str(reward),
            "sha256": _digest(reward),
            "function": "compute_score",
        },
        "requirements": ["distributed"],
    }
    request.update(updates)
    return request


def _build(tmp_path: Path, request: dict[str, object]) -> list[str]:
    dataset = verify_verl_dataset(
        str(request["dataset_uri"]), str(request["dataset_sha256"])
    )
    return build_verl_command(
        request, resolved_model=tmp_path / "model", dataset_path=dataset
    )


def _override_value(command: list[str], key: str) -> str:
    prefix = f"{key}="
    return next(value[len(prefix) :] for value in command if value.startswith(prefix))


def test_grpo_translation_pins_inputs_topology_and_checkpoint(tmp_path: Path) -> None:
    request = _request(tmp_path)
    command = _build(tmp_path, request)
    assert command[:3] == [sys.executable, "-m", "verl.trainer.main_ppo"]
    assert _override_value(command, "algorithm.adv_estimator") == '"grpo"'
    assert _override_value(command, "trainer.n_gpus_per_node") == "4"
    assert _override_value(command, "actor_rollout_ref.rollout.n") == "8"
    assert _override_value(command, "actor_rollout_ref.rollout.name") == '"vllm"'
    assert "trainer.save_freq=1" in command
    assert "trainer.resume_mode=disable" in command
    assert "trainer.logger=[console]" in command
    assert _override_value(command, "actor_rollout_ref.actor.fsdp.seed") == "42"
    assert (
        _override_value(command, "actor_rollout_ref.actor.fsdp.full_determinism")
        == "true"
    )


def test_ppo_maps_critic_to_the_same_pinned_model(tmp_path: Path) -> None:
    request = _request(tmp_path, algorithm="ppo")
    command = _build(tmp_path, request)
    assert _override_value(command, "algorithm.adv_estimator") == '"gae"'
    assert _override_value(command, "critic.model.path") == json.dumps(
        str(tmp_path / "model")
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"config": {"rollout_samples": 8}}, "rollout_engine"),
        (
            {"config": {"rollout_samples": 8, "rollout_engine": "mystery"}},
            "vllm.*sglang",
        ),
        (
            {
                "config": {
                    "rollout_samples": 8,
                    "rollout_engine": "vllm",
                    "learning_rate_typo": 1e-6,
                }
            },
            "learning_rate_typo",
        ),
        (
            {
                "config": {"rollout_samples": 1, "rollout_engine": "vllm"},
            },
            "greater than 1",
        ),
    ],
)
def test_invalid_config_is_rejected_before_execution(
    tmp_path: Path, updates: dict[str, object], message: str
) -> None:
    with pytest.raises(VerlConfigError, match=message):
        _build(tmp_path, _request(tmp_path, **updates))


def test_unpinned_reward_and_multiturn_are_rejected(tmp_path: Path) -> None:
    request = _request(
        tmp_path,
        reward={"type": "python", "path": "reward.py"},
    )
    with pytest.raises(VerlConfigError, match="exactly"):
        _build(tmp_path, request)
    request = _request(tmp_path, environment={"type": "multi_turn"})
    with pytest.raises(VerlConfigError, match="single_turn"):
        _build(tmp_path, request)


def test_multimodal_requires_an_explicit_image_key(tmp_path: Path) -> None:
    request = _request(tmp_path, requirements=["distributed", "multimodal"])
    with pytest.raises(VerlConfigError, match="image_key"):
        _build(tmp_path, request)


def test_dataset_must_be_parquet_and_match_digest(tmp_path: Path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    with pytest.raises(VerlConfigError, match="parquet"):
        verify_verl_dataset(dataset.as_uri(), _digest(dataset))
    request = _request(tmp_path)
    dataset = verify_verl_dataset(
        str(request["dataset_uri"]), str(request["dataset_sha256"])
    )
    dataset.write_bytes(b"changed")
    with pytest.raises(VerlConfigError, match="does not match"):
        verify_verl_dataset(str(request["dataset_uri"]), str(request["dataset_sha256"]))


def test_windows_file_uri_drive_prefix_is_normalized() -> None:
    assert (
        verl_module._decode_file_uri_path("/C:/data/train.parquet", platform="win32")
        == "C:/data/train.parquet"
    )


def test_factory_is_lazy_and_version_pinned() -> None:
    backend = verl_backend(version="0.10.0.dev", timeout_seconds=60)
    assert backend.name == "verl"
    assert backend.version == "0.10.0.dev"
    assert backend.capabilities.algorithms == frozenset({"ppo", "grpo"})
    assert backend.capabilities.features == frozenset({"distributed", "multimodal"})
    assert "0.10.0.dev" in backend.command


def _protocol_request(tmp_path: Path) -> dict[str, object]:
    values = _request(tmp_path)
    experiment = TrainingExperiment(
        algorithm=str(values["algorithm"]),
        model=str(values["model"]),
        model_revision=str(values["model_revision"]),
        dataset_uri=str(values["dataset_uri"]),
        dataset_sha256=str(values["dataset_sha256"]),
        output_dir=Path(str(values["output_dir"])),
        seed=int(values["seed"]),
        config=values["config"],  # type: ignore[arg-type]
        task="adapter-test",
        environment=values["environment"],  # type: ignore[arg-type]
        reward=values["reward"],  # type: ignore[arg-type]
        requirements=frozenset(values["requirements"]),  # type: ignore[arg-type]
    )
    return experiment.to_dict()


def test_runner_validates_executes_and_normalizes_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _protocol_request(tmp_path)
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.setattr(
        verl_module.importlib.metadata, "version", lambda _: "0.10.0.dev"
    )
    monkeypatch.setattr(verl_module, "resolve_verl_model", lambda *_: model)

    def fake_run(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        artifact = Path(
            json.loads(_override_value(command, "trainer.default_local_dir"))
        )
        checkpoint = artifact / "global_step_1" / "actor"
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(verl_module.subprocess, "run", fake_run)
    verl_module.run_adapter(
        request_path,
        result_path,
        Path(str(request["output_dir"])),
        "0.10.0.dev",
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["backend"] == "verl"
    assert result["experiment_sha256"] == request["experiment_sha256"]
    assert result["metrics"]["completed"] == 1.0


def test_runner_rejects_version_and_digest_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _protocol_request(tmp_path)
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    monkeypatch.setattr(verl_module.importlib.metadata, "version", lambda _: "wrong")
    with pytest.raises(VerlConfigError, match="expected"):
        verl_module.run_adapter(
            request_path,
            tmp_path / "result.json",
            Path(str(request["output_dir"])),
            "0.10.0.dev",
        )
    monkeypatch.setattr(
        verl_module.importlib.metadata, "version", lambda _: "0.10.0.dev"
    )
    request["seed"] = 7
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(VerlConfigError, match="digest does not match"):
        verl_module.run_adapter(
            request_path,
            tmp_path / "result.json",
            Path(str(request["output_dir"])),
            "0.10.0.dev",
        )
