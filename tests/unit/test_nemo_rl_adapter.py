"""Unit tests for the fail-closed NeMo RL training adapter."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import stateset_agents.training.adapters.nemo_rl as nemo_module
from stateset_agents.training.adapters.nemo_rl import (
    NemoRLConfigError,
    build_nemo_rl_command,
    nemo_rl_backend,
    verify_nemo_rl_dataset,
)
from stateset_agents.training.backends import TrainingExperiment


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_root(tmp_path: Path) -> Path:
    root = tmp_path / "nemo-rl"
    (root / "examples" / "configs").mkdir(parents=True, exist_ok=True)
    (root / "examples" / "run_grpo.py").write_text("# fixture\n", encoding="utf-8")
    (root / "examples" / "configs" / "grpo_math_1B.yaml").write_text(
        "grpo: {}\n", encoding="utf-8"
    )
    return root


def _request(tmp_path: Path, **updates: object) -> dict[str, object]:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"question":"1+1","answer":"2"}\n', encoding="utf-8")
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
            "num_generations_per_prompt": 8,
            "max_num_steps": 10,
            "generation_backend": "vllm",
            "num_nodes": 1,
            "gpus_per_node": 4,
            "tensor_parallel_size": 2,
            "input_key": "question",
            "output_key": "answer",
            "activation_checkpointing": True,
        },
        "environment": {"type": "single_turn", "name": "math"},
        "reward": {
            "type": "nemo_builtin",
            "name": "math",
            "implementation": "hf_math_verify",
        },
        "requirements": ["distributed"],
    }
    request.update(updates)
    return request


def _build(tmp_path: Path, request: dict[str, object]) -> list[str]:
    dataset = verify_nemo_rl_dataset(
        str(request["dataset_uri"]), str(request["dataset_sha256"])
    )
    return build_nemo_rl_command(
        request,
        source_root=_source_root(tmp_path),
        resolved_model=tmp_path / "model",
        dataset_path=dataset,
    )


def _override_value(command: list[str], key: str) -> str:
    prefix = f"{key}="
    return next(value[len(prefix) :] for value in command if value.startswith(prefix))


def test_grpo_translation_pins_inputs_topology_reward_and_checkpoint(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    command = _build(tmp_path, request)
    assert command[:2] == [
        sys.executable,
        str(tmp_path / "nemo-rl" / "examples" / "run_grpo.py"),
    ]
    assert _override_value(command, "policy.model_name") == json.dumps(
        str(tmp_path / "model")
    )
    assert _override_value(command, "+data.train.data_path") == json.dumps(
        str(tmp_path / "train.jsonl")
    )
    assert _override_value(command, "cluster.gpus_per_node") == "4"
    assert _override_value(command, "policy.dtensor_cfg.tensor_parallel_size") == "2"
    assert (
        _override_value(command, "policy.generation.vllm_cfg.tensor_parallel_size")
        == "2"
    )
    assert "env.math.math_verify_impl=hf_math_verify" in command
    assert "checkpointing.save_period=1" in command
    assert "checkpointing.save_consolidated=true" in command
    assert "logger.wandb_enabled=false" in command
    assert _override_value(command, "grpo.seed") == "42"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"algorithm": "ppo"}, "unsupported.*ppo"),
        (
            {"config": {"num_generations_per_prompt": 8, "max_num_steps": 10}},
            "generation_backend",
        ),
        (
            {
                "config": {
                    "num_generations_per_prompt": 8,
                    "generation_backend": "vllm",
                }
            },
            "max_num_steps",
        ),
        (
            {
                "config": {
                    "num_generations_per_prompt": 8,
                    "max_num_steps": 10,
                    "generation_backend": "sglang",
                }
            },
            "vllm",
        ),
        (
            {
                "config": {
                    "num_generations_per_prompt": 8,
                    "max_num_steps": 10,
                    "generation_backend": "vllm",
                    "learning_rate_typo": 1e-6,
                }
            },
            "learning_rate_typo",
        ),
        (
            {
                "config": {
                    "num_generations_per_prompt": 1,
                    "max_num_steps": 10,
                    "generation_backend": "vllm",
                }
            },
            "greater than 1",
        ),
    ],
)
def test_unrepresentable_config_is_rejected_before_execution(
    tmp_path: Path, updates: dict[str, object], message: str
) -> None:
    with pytest.raises(NemoRLConfigError, match=message):
        _build(tmp_path, _request(tmp_path, **updates))


def test_custom_reward_and_multiturn_are_rejected(tmp_path: Path) -> None:
    request = _request(
        tmp_path,
        reward={"type": "python", "path": "reward.py"},
    )
    with pytest.raises(NemoRLConfigError, match="hf_math_verify"):
        _build(tmp_path, request)
    request = _request(tmp_path, environment={"type": "multi_turn", "name": "math"})
    with pytest.raises(NemoRLConfigError, match="single-turn math"):
        _build(tmp_path, request)


def test_dataset_must_be_json_and_match_digest(tmp_path: Path) -> None:
    dataset = tmp_path / "train.parquet"
    dataset.write_bytes(b"PAR1")
    with pytest.raises(NemoRLConfigError, match="json"):
        verify_nemo_rl_dataset(dataset.as_uri(), _digest(dataset))
    request = _request(tmp_path)
    dataset = verify_nemo_rl_dataset(
        str(request["dataset_uri"]), str(request["dataset_sha256"])
    )
    dataset.write_text("changed", encoding="utf-8")
    with pytest.raises(NemoRLConfigError, match="does not match"):
        verify_nemo_rl_dataset(
            str(request["dataset_uri"]), str(request["dataset_sha256"])
        )


def test_windows_file_uri_drive_prefix_is_normalized() -> None:
    assert (
        nemo_module._decode_file_uri_path("/C:/data/train.jsonl", platform="win32")
        == "C:/data/train.jsonl"
    )


def test_factory_is_lazy_and_version_pinned() -> None:
    backend = nemo_rl_backend(version="0.6.0+abcdef0", timeout_seconds=60)
    assert backend.name == "nemo-rl"
    assert backend.version == "0.6.0+abcdef0"
    assert backend.capabilities.algorithms == frozenset({"grpo"})
    assert backend.capabilities.features == frozenset({"distributed"})
    assert "0.6.0+abcdef0" in backend.command


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
    source_root = _source_root(tmp_path)
    monkeypatch.setattr(
        nemo_module.importlib.metadata, "version", lambda _: "0.6.0+abcdef0"
    )
    monkeypatch.setattr(nemo_module, "resolve_nemo_rl_model", lambda *_: model)
    monkeypatch.setattr(nemo_module, "resolve_nemo_rl_source_root", lambda: source_root)

    def fake_run(command: list[str], *, cwd: Path, check: bool) -> SimpleNamespace:
        assert check is False
        assert cwd == source_root
        artifact = Path(
            json.loads(_override_value(command, "checkpointing.checkpoint_dir"))
        )
        checkpoint = artifact / "step_1" / "weights"
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(nemo_module.subprocess, "run", fake_run)
    nemo_module.run_adapter(
        request_path,
        result_path,
        Path(str(request["output_dir"])),
        "0.6.0+abcdef0",
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["backend"] == "nemo-rl"
    assert result["experiment_sha256"] == request["experiment_sha256"]
    assert result["metrics"]["completed"] == 1.0


def test_runner_rejects_version_and_digest_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _protocol_request(tmp_path)
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    monkeypatch.setattr(nemo_module.importlib.metadata, "version", lambda _: "wrong")
    with pytest.raises(NemoRLConfigError, match="expected"):
        nemo_module.run_adapter(
            request_path,
            tmp_path / "result.json",
            Path(str(request["output_dir"])),
            "0.6.0+abcdef0",
        )
    monkeypatch.setattr(
        nemo_module.importlib.metadata, "version", lambda _: "0.6.0+abcdef0"
    )
    request["seed"] = 7
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(NemoRLConfigError, match="digest does not match"):
        nemo_module.run_adapter(
            request_path,
            tmp_path / "result.json",
            Path(str(request["output_dir"])),
            "0.6.0+abcdef0",
        )
