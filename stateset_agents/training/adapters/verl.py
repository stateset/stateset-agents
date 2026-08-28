"""Fail-closed verl adapter for the StateSet training-backend protocol."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import numbers
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from ..backends import (
    BACKEND_PROTOCOL_VERSION,
    BackendCapabilities,
    CommandTrainingBackend,
)

_ALGORITHM_ESTIMATORS = {"ppo": "gae", "grpo": "grpo"}
_VALUE_OVERRIDES: dict[str, tuple[tuple[str, ...], type[Any]]] = {
    "learning_rate": (("actor_rollout_ref.actor.optim.lr",), float),
    "train_batch_size": (("data.train_batch_size",), int),
    "ppo_mini_batch_size": (("actor_rollout_ref.actor.ppo_mini_batch_size",), int),
    "ppo_micro_batch_size_per_gpu": (
        ("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",),
        int,
    ),
    "rollout_samples": (("actor_rollout_ref.rollout.n",), int),
    "max_prompt_length": (("data.max_prompt_length",), int),
    "max_response_length": (("data.max_response_length",), int),
    "total_epochs": (("trainer.total_epochs",), int),
    "total_training_steps": (("trainer.total_training_steps",), int),
    "num_nodes": (("trainer.nnodes",), int),
    "gpus_per_node": (("trainer.n_gpus_per_node",), int),
    "tensor_parallel_size": (
        ("actor_rollout_ref.rollout.tensor_model_parallel_size",),
        int,
    ),
    "gpu_memory_utilization": (
        ("actor_rollout_ref.rollout.gpu_memory_utilization",),
        float,
    ),
    "temperature": (("actor_rollout_ref.rollout.temperature",), float),
    "top_p": (("actor_rollout_ref.rollout.top_p",), float),
    "prompt_key": (("data.prompt_key",), str),
    "image_key": (("data.image_key",), str),
    "kl_reward_coefficient": (("algorithm.kl_ctrl.kl_coef",), float),
    "kl_loss_coefficient": (("actor_rollout_ref.actor.kl_loss_coef",), float),
    "rollout_engine": (("actor_rollout_ref.rollout.name",), str),
}
_FLAG_OVERRIDES: dict[str, tuple[str, ...]] = {
    "gradient_checkpointing": (
        "actor_rollout_ref.model.enable_gradient_checkpointing",
    ),
    "remove_padding": ("actor_rollout_ref.model.use_remove_padding",),
    "dynamic_batching": ("actor_rollout_ref.actor.use_dynamic_bsz",),
    "deterministic": (
        "actor_rollout_ref.rollout.full_determinism",
        "actor_rollout_ref.actor.fsdp.full_determinism",
        "actor_rollout_ref.ref.fsdp.full_determinism",
        "critic.fsdp.full_determinism",
    ),
}
_NONNEGATIVE = frozenset({"kl_reward_coefficient", "kl_loss_coefficient"})
_SEMANTIC_KEYS = (
    "protocol",
    "protocol_version",
    "algorithm",
    "model",
    "model_revision",
    "dataset_uri",
    "dataset_sha256",
    "seed",
    "task",
    "config",
    "environment",
    "reward",
    "requirements",
)


class VerlConfigError(ValueError):
    """Raised when verl cannot preserve an experiment's semantics."""


def _require_object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VerlConfigError(f"{name} must be a JSON object")
    return value


def _verified_file(path_value: Any, digest_value: Any, name: str) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise VerlConfigError(f"{name} must be a non-empty path")
    if not isinstance(digest_value, str) or not re.fullmatch(
        r"[0-9a-f]{64}", digest_value
    ):
        raise VerlConfigError(f"{name} SHA-256 must be 64 lowercase hex digits")
    path = Path(path_value).resolve()
    if not path.is_file():
        raise VerlConfigError(f"{name} is not a file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != digest_value:
        raise VerlConfigError(f"{name} SHA-256 does not match experiment")
    return path


def _decode_file_uri_path(value: str, *, platform: str | None = None) -> str:
    decoded = unquote(value)
    current_platform = sys.platform if platform is None else platform
    if current_platform == "win32" and re.match(r"^/[A-Za-z]:/", decoded):
        return decoded[1:]
    return decoded


def verify_verl_dataset(dataset_uri: str, expected_sha256: str) -> Path:
    """Verify a local Parquet dataset before launching verl."""
    parsed = urlparse(dataset_uri)
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise VerlConfigError("file dataset_uri must not name a remote host")
        path = Path(_decode_file_uri_path(parsed.path))
    elif not parsed.scheme:
        path = Path(dataset_uri)
    else:
        raise VerlConfigError("verl adapter requires a local dataset path or file URI")
    path = path.resolve()
    if path.suffix.lower() != ".parquet":
        raise VerlConfigError("verl adapter currently requires a .parquet dataset")
    return _verified_file(str(path), expected_sha256, "dataset")


def resolve_verl_model(model: str, revision: str) -> Path:
    """Resolve a local marker-pinned model or immutable Hub commit."""
    local = Path(model)
    if local.exists():
        resolved = local.resolve()
        marker = resolved / ".stateset-model-revision"
        if (
            not marker.is_file()
            or marker.read_text(encoding="utf-8").strip() != revision
        ):
            raise VerlConfigError(
                "local model must contain .stateset-model-revision matching model_revision"
            )
        return resolved
    if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise VerlConfigError(
            "remote model_revision must be an immutable 40-64 character commit"
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - exercised in engine image
        raise VerlConfigError(
            "huggingface_hub is required to resolve a remote model revision"
        ) from exc
    return Path(snapshot_download(repo_id=model, revision=revision)).resolve()


def _value(value: Any, expected: type[Any], name: str) -> Any:
    valid = isinstance(value, expected)
    if expected is float:
        valid = isinstance(value, numbers.Real)
    if isinstance(value, bool) or not valid:
        raise VerlConfigError(f"config.{name} must be {expected.__name__}")
    if expected in (int, float):
        invalid = value < 0 if name in _NONNEGATIVE else value <= 0
        if invalid:
            qualifier = "non-negative" if name in _NONNEGATIVE else "positive"
            raise VerlConfigError(f"config.{name} must be {qualifier}")
    return value


def _override(key: str, value: Any) -> str:
    if isinstance(value, bool):
        encoded = "true" if value else "false"
    elif isinstance(value, str):
        encoded = json.dumps(value)
    else:
        encoded = json.dumps(value, separators=(",", ":"))
    return f"{key}={encoded}"


def build_verl_command(
    request: Mapping[str, Any], *, resolved_model: Path, dataset_path: Path
) -> list[str]:
    """Translate one StateSet request to a strict verl Hydra command."""
    algorithm = str(request.get("algorithm", "")).lower()
    if algorithm not in _ALGORITHM_ESTIMATORS:
        raise VerlConfigError(f"unsupported verl algorithm: {algorithm!r}")
    config = _require_object(request.get("config"), "config")
    unknown = sorted(set(config) - set(_VALUE_OVERRIDES) - set(_FLAG_OVERRIDES))
    if unknown:
        raise VerlConfigError("unsupported verl config fields: " + ", ".join(unknown))
    if "rollout_engine" not in config:
        raise VerlConfigError("config.rollout_engine must be explicitly set")
    environment = _require_object(request.get("environment", {}), "environment")
    if environment not in ({}, {"type": "single_turn"}):
        raise VerlConfigError("verl adapter currently supports single_turn only")
    reward = _require_object(request.get("reward", {}), "reward")
    if set(reward) != {"type", "path", "sha256", "function"}:
        raise VerlConfigError(
            "verl Python reward requires exactly type, path, sha256, and function"
        )
    if reward.get("type") != "python":
        raise VerlConfigError("verl adapter requires a content-pinned Python reward")
    function_name = reward.get("function")
    if not isinstance(function_name, str) or not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", function_name
    ):
        raise VerlConfigError("reward.function must be a Python identifier")
    reward_path = _verified_file(
        reward.get("path"), reward.get("sha256"), "reward.path"
    )

    requirements = set(request.get("requirements", ()))
    unsupported = requirements - {"distributed", "multimodal"}
    if unsupported:
        raise VerlConfigError(
            "unsupported verl requirements: " + ", ".join(sorted(unsupported))
        )
    if "multimodal" in requirements and "image_key" not in config:
        raise VerlConfigError("multimodal requires config.image_key")

    artifact = Path(str(request["output_dir"])).resolve() / "artifact"
    seed = request["seed"]
    command = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        "--config-name=ppo_trainer",
        "model_engine=dp",
        _override("algorithm.adv_estimator", _ALGORITHM_ESTIMATORS[algorithm]),
        _override("actor_rollout_ref.model.path", str(resolved_model)),
        _override("critic.model.path", str(resolved_model)),
        _override("data.train_files", [str(dataset_path)]),
        _override("data.val_files", [str(dataset_path)]),
        _override("reward.custom_reward_function.path", str(reward_path)),
        _override("reward.custom_reward_function.name", function_name),
        _override("trainer.default_local_dir", str(artifact)),
        "trainer.logger=[console]",
        "trainer.val_before_train=false",
        "trainer.test_freq=-1",
        "trainer.save_freq=1",
        "trainer.resume_mode=disable",
        _override("data.seed", seed),
        _override("actor_rollout_ref.rollout.seed", seed),
        _override("actor_rollout_ref.actor.fsdp.seed", seed),
        _override("actor_rollout_ref.ref.fsdp.seed", seed),
        _override("critic.fsdp.seed", seed),
    ]
    for name, (keys, expected) in _VALUE_OVERRIDES.items():
        if name not in config:
            continue
        value = _value(config[name], expected, name)
        if name in {"top_p", "gpu_memory_utilization"} and float(value) > 1:
            raise VerlConfigError(f"config.{name} must not exceed 1")
        if name == "rollout_engine" and value not in {"vllm", "sglang"}:
            raise VerlConfigError("config.rollout_engine must be 'vllm' or 'sglang'")
        if name in {"prompt_key", "image_key"} and not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_.-]*", str(value)
        ):
            raise VerlConfigError(f"config.{name} contains invalid characters")
        for key in keys:
            command.append(_override(key, value))
    for name, keys in _FLAG_OVERRIDES.items():
        if name in config:
            if not isinstance(config[name], bool):
                raise VerlConfigError(f"config.{name} must be bool")
            for key in keys:
                command.append(_override(key, config[name]))
    if config.get("kl_reward_coefficient", 0) > 0:
        command.append("algorithm.use_kl_in_reward=true")
    if config.get("kl_loss_coefficient", 0) > 0:
        command.append("actor_rollout_ref.actor.use_kl_loss=true")
    if algorithm == "grpo" and config.get("rollout_samples", 1) <= 1:
        raise VerlConfigError("grpo requires config.rollout_samples greater than 1")
    return command


def verl_backend(
    *, version: str, timeout_seconds: int = 14400
) -> CommandTrainingBackend:
    """Create a verl backend without importing the optional engine."""
    return CommandTrainingBackend(
        name="verl",
        version=version,
        capabilities=BackendCapabilities(
            algorithms=frozenset(_ALGORITHM_ESTIMATORS),
            features=frozenset({"distributed", "multimodal"}),
        ),
        command=(
            sys.executable,
            "-m",
            "stateset_agents.training.adapters.verl",
            "--request",
            "{request}",
            "--result",
            "{result}",
            "--output-dir",
            "{output_dir}",
            "--expected-version",
            version,
        ),
        timeout_seconds=timeout_seconds,
    )


def run_adapter(
    request_path: Path, result_path: Path, output_dir: Path, expected_version: str
) -> None:
    """Validate, execute, and normalize one verl run."""
    installed_version = importlib.metadata.version("verl")
    if installed_version != expected_version:
        raise VerlConfigError(
            f"installed verl {installed_version!r} != expected {expected_version!r}"
        )
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("protocol_version") != BACKEND_PROTOCOL_VERSION:
        raise VerlConfigError("unsupported StateSet backend protocol version")
    if Path(str(request.get("output_dir"))).resolve() != output_dir.resolve():
        raise VerlConfigError("request output_dir does not match adapter output_dir")
    semantic_payload = {key: request.get(key) for key in _SEMANTIC_KEYS}
    digest = hashlib.sha256(
        json.dumps(semantic_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if request.get("experiment_sha256") != digest:
        raise VerlConfigError("request experiment digest does not match payload")

    dataset = verify_verl_dataset(
        str(request["dataset_uri"]), str(request["dataset_sha256"])
    )
    model = resolve_verl_model(str(request["model"]), str(request["model_revision"]))
    command = build_verl_command(request, resolved_model=model, dataset_path=dataset)
    start = time.monotonic()
    completed = subprocess.run(command, check=False)
    duration = time.monotonic() - start
    if completed.returncode != 0:
        raise RuntimeError(f"verl exited {completed.returncode}")
    artifact = output_dir.resolve() / "artifact"
    if not artifact.is_dir() or not any(path.is_file() for path in artifact.rglob("*")):
        raise RuntimeError("verl did not produce a reusable checkpoint artifact")
    result = {
        "protocol_version": BACKEND_PROTOCOL_VERSION,
        "backend": "verl",
        "backend_version": installed_version,
        "experiment_sha256": request["experiment_sha256"],
        "artifact_uri": str(artifact),
        "metrics": {"completed": 1.0, "wall_time_seconds": duration},
        "metadata": {"engine_command": command},
    }
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the adapter process used by :class:`CommandTrainingBackend`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args(argv)
    run_adapter(args.request, args.result, args.output_dir, args.expected_version)
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess entrypoint
    raise SystemExit(main())


__all__ = [
    "VerlConfigError",
    "build_verl_command",
    "main",
    "resolve_verl_model",
    "run_adapter",
    "verify_verl_dataset",
    "verl_backend",
]
