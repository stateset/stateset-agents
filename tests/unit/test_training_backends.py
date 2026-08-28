"""Tests for the versioned, framework-neutral training backend contract."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from stateset_agents.training.backends import (
    BACKEND_PROTOCOL_VERSION,
    BackendCapabilities,
    BackendError,
    BackendExecutionError,
    BackendRegistry,
    CommandTrainingBackend,
    TrainingBackend,
    TrainingExperiment,
)


def _experiment(tmp_path: Path, **overrides: object) -> TrainingExperiment:
    values = {
        "algorithm": "grpo",
        "model": "Qwen/Qwen3.5-0.8B",
        "model_revision": "model-commit",
        "dataset_uri": "file:///data/train.jsonl",
        "dataset_sha256": hashlib.sha256(b"dataset").hexdigest(),
        "output_dir": tmp_path / "run",
        "seed": 42,
        "config": {"learning_rate": 5e-6, "steps": 4},
        "task": "gsm8k",
        "environment": {"name": "single-turn"},
        "reward": {"name": "exact-match"},
        "requirements": frozenset({"distributed"}),
    }
    values.update(overrides)
    return TrainingExperiment(**values)  # type: ignore[arg-type]


def test_experiment_digest_is_stable_and_excludes_output_location(
    tmp_path: Path,
) -> None:
    first = _experiment(tmp_path, config={"steps": 4, "learning_rate": 5e-6})
    second = _experiment(
        tmp_path,
        config={"learning_rate": 5e-6, "steps": 4},
        output_dir=tmp_path / "somewhere-else",
    )
    assert first.sha256 == second.sha256
    assert first.to_dict()["experiment_sha256"] == first.sha256


@pytest.mark.parametrize(
    "key",
    ["token", "api_key", "fireworks_api_key", "hf_token", "aws_secret_access_key"],
)
def test_experiment_rejects_serialized_secrets(tmp_path: Path, key: str) -> None:
    with pytest.raises(ValueError, match="supplied via environment"):
        _experiment(tmp_path, config={key: "do-not-store"})


def test_experiment_rejects_nested_and_uri_credentials(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="supplied via environment"):
        _experiment(tmp_path, reward={"judges": ({"authorization": "secret"},)})
    with pytest.raises(ValueError, match="must not embed credentials"):
        _experiment(tmp_path, dataset_uri="https://user:password@example.test/data")


def test_capabilities_reject_semantic_downgrade(tmp_path: Path) -> None:
    backend = TrainingBackend()
    backend.name = "limited"
    backend.version = "1"
    backend.capabilities = BackendCapabilities(
        algorithms=frozenset({"grpo"}), features=frozenset()
    )
    with pytest.raises(BackendError, match="missing capabilities"):
        backend.validate(_experiment(tmp_path))


class _Backend(TrainingBackend):
    name = "test"
    version = "1"
    capabilities = BackendCapabilities(algorithms=frozenset({"grpo"}))


def test_registry_is_explicit_and_rejects_replacement() -> None:
    registry = BackendRegistry()
    backend = _Backend()
    registry.register(backend)
    assert registry.available() == ["test"]
    assert registry.get(" TEST ") is backend
    with pytest.raises(BackendError, match="already registered"):
        registry.register(_Backend())
    with pytest.raises(BackendError, match="available: test"):
        registry.get("missing")


def _worker_code(*, digest_override: str | None = None) -> str:
    digest_expr = (
        repr(digest_override) if digest_override else "request['experiment_sha256']"
    )
    return (
        "import json,pathlib,sys;"
        "request=json.loads(pathlib.Path(sys.argv[1]).read_text());"
        "result=pathlib.Path(sys.argv[2]);out=pathlib.Path(sys.argv[3]);"
        "artifact=out/'artifact';artifact.mkdir();"
        "(artifact/'weights.bin').write_bytes(b'weights');"
        "result.write_text(json.dumps({"
        "'backend':'verl','backend_version':'1.2.3',"
        f"'experiment_sha256':{digest_expr},"
        "'artifact_uri':str(artifact),'protocol_version':1,"
        "'metrics':{'samples_per_second':1.5,'eval_score_final':0.4}}))"
    )


def _command_backend(tmp_path: Path, code: str) -> CommandTrainingBackend:
    return CommandTrainingBackend(
        name="verl",
        version="1.2.3",
        capabilities=BackendCapabilities(
            algorithms=frozenset({"grpo"}), features=frozenset({"distributed"})
        ),
        command=[
            sys.executable,
            "-c",
            code,
            "{request}",
            "{result}",
            "{output_dir}",
        ],
        cwd=tmp_path,
    )


def test_command_backend_runs_shell_free_and_validates_result(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    result = _command_backend(tmp_path, _worker_code()).run(experiment)
    assert result.backend == "verl"
    assert result.experiment_sha256 == experiment.sha256
    assert result.metrics["samples_per_second"] == 1.5
    request = json.loads(
        (experiment.output_dir / "backend-request.json").read_text(encoding="utf-8")
    )
    assert request["protocol_version"] == BACKEND_PROTOCOL_VERSION
    assert (experiment.output_dir / "backend-stdout.log").is_file()
    assert (experiment.output_dir / "artifact" / "weights.bin").is_file()


def test_command_backend_rejects_adapter_semantic_drift(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    backend = _command_backend(tmp_path, _worker_code(digest_override="0" * 64))
    with pytest.raises(BackendExecutionError, match="digest does not match"):
        backend.run(experiment)


def test_command_backend_rejects_artifact_escape(tmp_path: Path) -> None:
    code = _worker_code().replace(
        "artifact=out/'artifact'", "artifact=out.parent/'escape'"
    )
    with pytest.raises(BackendExecutionError, match="inside output_dir"):
        _command_backend(tmp_path, code).run(_experiment(tmp_path))


def test_command_backend_retains_failure_logs(tmp_path: Path) -> None:
    backend = CommandTrainingBackend(
        name="verl",
        version="1.2.3",
        capabilities=BackendCapabilities(
            algorithms=frozenset({"grpo"}), features=frozenset({"distributed"})
        ),
        command=[
            sys.executable,
            "-c",
            "import sys;print('out');print('err',file=sys.stderr);sys.exit(7)",
            "{request}",
            "{result}",
            "{output_dir}",
        ],
    )
    experiment = _experiment(tmp_path)
    with pytest.raises(BackendExecutionError, match="exited 7"):
        backend.run(experiment)
    assert (experiment.output_dir / "backend-stdout.log").read_text() == "out\n"
    assert (experiment.output_dir / "backend-stderr.log").read_text() == "err\n"


def test_command_backend_requires_complete_argument_contract(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing required placeholders"):
        CommandTrainingBackend(
            name="verl",
            version="1",
            capabilities=BackendCapabilities(algorithms=frozenset({"grpo"})),
            command=[sys.executable, "worker.py", "{request}"],
            cwd=tmp_path,
        )
