"""Unit tests for the fail-closed OpenRLHF training adapter."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import stateset_agents.training.adapters.openrlhf as openrlhf_module
from stateset_agents.training.adapters.openrlhf import (
    OpenRLHFConfigError,
    build_openrlhf_command,
    openrlhf_backend,
    verify_dataset,
)
from stateset_agents.training.backends import TrainingExperiment


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _request(tmp_path: Path, **updates: object) -> dict[str, object]:
    reward = tmp_path / "reward.py"
    reward.write_text("def reward_func(*args): return [1.0]\n", encoding="utf-8")
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"input":"question"}\n', encoding="utf-8")
    request: dict[str, object] = {
        "algorithm": "grpo",
        "model": "Qwen/example",
        "model_revision": "a" * 40,
        "dataset_uri": dataset.as_uri(),
        "dataset_sha256": _digest(dataset),
        "output_dir": str(tmp_path / "output"),
        "seed": 42,
        "config": {
            "learning_rate": 5e-7,
            "samples_per_prompt": 4,
            "num_nodes": 1,
            "gpus_per_node": 4,
            "apply_chat_template": True,
        },
        "environment": {"type": "single_turn"},
        "reward": {"type": "python", "path": str(reward), "sha256": _digest(reward)},
        "requirements": ["distributed"],
    }
    request.update(updates)
    return request


def _build(tmp_path: Path, request: dict[str, object]) -> list[str]:
    dataset = verify_dataset(
        str(request["dataset_uri"]), str(request["dataset_sha256"])
    )
    return build_openrlhf_command(
        request, resolved_model=tmp_path / "model", dataset_path=dataset
    )


def _option_values(command: list[str], option: str) -> list[str]:
    return [
        command[index + 1] for index, value in enumerate(command) if value == option
    ]


def test_grpo_translation_pins_inputs_and_topology(tmp_path: Path) -> None:
    request = _request(tmp_path)
    command = _build(tmp_path, request)
    assert command[:3] == [
        sys.executable,
        "-m",
        "openrlhf.cli.train_ppo_ray",
    ]
    assert _option_values(command, "--algo.advantage.estimator") == ["group_norm"]
    assert float(_option_values(command, "--actor.adam.lr")[0]) == 5e-7
    assert _option_values(command, "--actor.num_gpus_per_node") == ["4"]
    assert _option_values(command, "--ref.num_gpus_per_node") == ["4"]
    assert _option_values(command, "--critic.num_gpus_per_node") == ["4"]
    assert "--data.apply_chat_template" in command
    assert str(Path(str(request["reward"]["path"])).resolve()) in command  # type: ignore[index]


def test_gspo_and_async_requirements_are_explicit(tmp_path: Path) -> None:
    request = _request(
        tmp_path,
        algorithm="gspo",
        requirements=["async_rollouts", "distributed"],
    )
    command = _build(tmp_path, request)
    assert _option_values(command, "--actor.policy_loss_type") == ["gspo"]
    assert "--train.async_enable" in command


@pytest.mark.parametrize("algorithm", ["grpo", "gspo"])
def test_group_algorithms_reject_single_sample(tmp_path: Path, algorithm: str) -> None:
    request = _request(
        tmp_path,
        algorithm=algorithm,
        config={"samples_per_prompt": 1},
    )
    with pytest.raises(OpenRLHFConfigError, match="greater than 1"):
        _build(tmp_path, request)


def test_unknown_config_is_never_silently_ignored(tmp_path: Path) -> None:
    request = _request(
        tmp_path,
        config={"samples_per_prompt": 4, "typo_learning_rate": 1e-6},
    )
    with pytest.raises(OpenRLHFConfigError, match="typo_learning_rate"):
        _build(tmp_path, request)


def test_agent_and_reward_code_are_content_pinned(tmp_path: Path) -> None:
    agent = tmp_path / "agent.py"
    agent.write_text("async def agent_func(*args): pass\n", encoding="utf-8")
    request = _request(
        tmp_path,
        environment={
            "type": "agent",
            "function_path": str(agent),
            "sha256": "0" * 64,
        },
        requirements=["multi_turn", "tool_use"],
    )
    with pytest.raises(OpenRLHFConfigError, match="does not match"):
        _build(tmp_path, request)


def test_multimodal_rejects_semantic_downgrade(tmp_path: Path) -> None:
    request = _request(tmp_path, requirements=["multimodal"])
    with pytest.raises(OpenRLHFConfigError, match="max_images_per_prompt"):
        _build(tmp_path, request)


def test_dataset_digest_is_verified(tmp_path: Path) -> None:
    request = _request(tmp_path)
    dataset = verify_dataset(
        str(request["dataset_uri"]), str(request["dataset_sha256"])
    )
    dataset.write_text("changed", encoding="utf-8")
    with pytest.raises(OpenRLHFConfigError, match="does not match"):
        verify_dataset(str(request["dataset_uri"]), str(request["dataset_sha256"]))


def test_windows_file_uri_drive_prefix_is_normalized() -> None:
    path = openrlhf_module._decode_file_uri_path(
        "/C:/data/train.jsonl", platform="win32"
    )
    assert path == "C:/data/train.jsonl"


def test_factory_is_lightweight_and_version_pinned() -> None:
    backend = openrlhf_backend(version="0.10.2", timeout_seconds=60)
    assert backend.name == "openrlhf"
    assert backend.version == "0.10.2"
    assert backend.capabilities.algorithms == frozenset({"ppo", "grpo", "gspo"})
    assert "--expected-version" in backend.command
    assert "0.10.2" in backend.command


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


def test_runner_checks_version_executes_and_writes_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _protocol_request(tmp_path)
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()

    monkeypatch.setattr(
        openrlhf_module.importlib.metadata, "version", lambda _: "0.10.2"
    )
    monkeypatch.setattr(openrlhf_module, "resolve_model", lambda *_: model)

    def fake_run(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        artifact = Path(_option_values(command, "--ckpt.output_dir")[0])
        artifact.mkdir(parents=True)
        (artifact / "config.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(openrlhf_module.subprocess, "run", fake_run)
    openrlhf_module.run_adapter(
        request_path, result_path, Path(str(request["output_dir"])), "0.10.2"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["backend"] == "openrlhf"
    assert result["backend_version"] == "0.10.2"
    assert result["experiment_sha256"] == request["experiment_sha256"]
    assert result["metrics"]["completed"] == 1.0


def test_runner_rejects_request_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _protocol_request(tmp_path)
    request["seed"] = 7
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    monkeypatch.setattr(
        openrlhf_module.importlib.metadata, "version", lambda _: "0.10.2"
    )
    with pytest.raises(OpenRLHFConfigError, match="digest does not match"):
        openrlhf_module.run_adapter(
            request_path,
            tmp_path / "result.json",
            Path(str(request["output_dir"])),
            "0.10.2",
        )


def test_request_fixture_is_json_serializable(tmp_path: Path) -> None:
    json.dumps(_request(tmp_path))
