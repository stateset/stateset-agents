"""Fail-closed OpenRLHF adapter for the StateSet backend protocol.

The adapter intentionally supports a curated subset of OpenRLHF's CLI.  Every
canonical field is either translated, independently verified, or rejected.
This prevents a configuration typo from silently changing a paid GPU run.
"""

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

_ALGORITHM_ARGS: dict[str, tuple[str, ...]] = {
    "ppo": ("--algo.advantage.estimator", "gae"),
    "grpo": ("--algo.advantage.estimator", "group_norm"),
    "gspo": (
        "--algo.advantage.estimator",
        "group_norm",
        "--actor.policy_loss_type",
        "gspo",
    ),
}

_VALUE_OPTIONS: dict[str, tuple[tuple[str, ...], type[Any]]] = {
    "learning_rate": (("--actor.adam.lr",), float),
    "train_batch_size": (("--train.batch_size",), int),
    "train_micro_batch_size": (("--train.micro_batch_size",), int),
    "rollout_batch_size": (("--rollout.batch_size",), int),
    "rollout_micro_batch_size": (("--rollout.micro_batch_size",), int),
    "samples_per_prompt": (("--rollout.n_samples_per_prompt",), int),
    "max_epochs": (("--train.max_epochs",), int),
    "max_samples": (("--data.max_samples",), int),
    "max_length": (("--data.max_len",), int),
    "max_new_tokens": (("--rollout.max_new_tokens",), int),
    "zero_stage": (("--ds.zero_stage",), int),
    "dtype": (("--ds.param_dtype",), str),
    "num_nodes": (
        ("--actor.num_nodes", "--ref.num_nodes", "--critic.num_nodes"),
        int,
    ),
    "gpus_per_node": (
        (
            "--actor.num_gpus_per_node",
            "--ref.num_gpus_per_node",
            "--critic.num_gpus_per_node",
        ),
        int,
    ),
    "vllm_num_engines": (("--vllm.num_engines",), int),
    "vllm_tensor_parallel_size": (("--vllm.tensor_parallel_size",), int),
    "kl_coefficient": (("--algo.kl.init_coef",), float),
    "temperature": (("--rollout.temperature",), float),
    "top_p": (("--rollout.top_p",), float),
    "input_key": (("--data.input_key",), str),
    "label_key": (("--data.label_key",), str),
    "max_images_per_prompt": (("--data.max_images_per_prompt",), int),
}

_FLAG_OPTIONS = {
    "apply_chat_template": "--data.apply_chat_template",
    "packing_samples": "--ds.packing_samples",
    "gradient_checkpointing": "--actor.gradient_checkpointing_enable",
    "colocate_all": "--train.colocate_all",
    "deterministic": "--train.full_determinism_enable",
}

_ALLOWED_ENVIRONMENT_KEYS = frozenset({"type", "function_path", "sha256"})
_ALLOWED_REWARD_KEYS = frozenset({"type", "path", "sha256"})
_NONNEGATIVE_OPTIONS = frozenset({"zero_stage", "kl_coefficient"})
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


class OpenRLHFConfigError(ValueError):
    """Raised when OpenRLHF cannot preserve an experiment's semantics."""


def _require_object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OpenRLHFConfigError(f"{name} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], allowed: frozenset[str], name: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise OpenRLHFConfigError(
            f"unsupported OpenRLHF {name} fields: {', '.join(unknown)}"
        )


def _positive_number(value: Any, expected: type[Any], name: str) -> str:
    valid = isinstance(value, expected)
    if expected is float:
        valid = isinstance(value, numbers.Real)
    if isinstance(value, bool) or not valid:
        raise OpenRLHFConfigError(f"config.{name} must be {expected.__name__}")
    if expected in (int, float):
        invalid = value < 0 if name in _NONNEGATIVE_OPTIONS else value <= 0
        if invalid:
            qualifier = "non-negative" if name in _NONNEGATIVE_OPTIONS else "positive"
            raise OpenRLHFConfigError(f"config.{name} must be {qualifier}")
    normalized = str(value)
    if expected is float:
        normalized = format(float(value), ".17g")
    return normalized


def _verified_file(path_value: Any, digest_value: Any, name: str) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise OpenRLHFConfigError(f"{name} must be a non-empty path")
    if not isinstance(digest_value, str) or not re.fullmatch(
        r"[0-9a-f]{64}", digest_value
    ):
        raise OpenRLHFConfigError(f"{name} SHA-256 must be 64 lowercase hex digits")
    path = Path(path_value).resolve()
    if not path.is_file():
        raise OpenRLHFConfigError(f"{name} is not a file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != digest_value:
        raise OpenRLHFConfigError(f"{name} SHA-256 does not match experiment")
    return path


def _decode_file_uri_path(value: str, *, platform: str | None = None) -> str:
    """Decode an RFC 8089 path, including Windows drive-letter form."""
    decoded = unquote(value)
    current_platform = sys.platform if platform is None else platform
    if current_platform == "win32" and re.match(r"^/[A-Za-z]:/", decoded):
        return decoded[1:]
    return decoded


def _local_dataset_path(dataset_uri: str) -> Path:
    parsed = urlparse(dataset_uri)
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise OpenRLHFConfigError("file dataset_uri must not name a remote host")
        # RFC 8089 file URIs encode Windows drive paths as ``/C:/...``.
        # WindowsPath treats that leading slash inconsistently across Python
        # patch releases, so normalize it explicitly before constructing Path.
        path = Path(_decode_file_uri_path(parsed.path))
    elif not parsed.scheme:
        path = Path(dataset_uri)
    else:
        raise OpenRLHFConfigError(
            "OpenRLHF adapter currently requires a local dataset path or file URI"
        )
    if not path.is_absolute():
        path = path.resolve()
    return path


def verify_dataset(dataset_uri: str, expected_sha256: str) -> Path:
    """Verify the exact local dataset bytes before allocating an engine."""
    path = _local_dataset_path(dataset_uri)
    if not path.is_file():
        raise OpenRLHFConfigError(f"dataset is not a file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise OpenRLHFConfigError("dataset SHA-256 does not match experiment")
    return path


def resolve_model(model: str, revision: str) -> Path:
    """Resolve an immutable local or Hugging Face model revision."""
    local = Path(model)
    if local.exists():
        resolved = local.resolve()
        marker = resolved / ".stateset-model-revision"
        if (
            not marker.is_file()
            or marker.read_text(encoding="utf-8").strip() != revision
        ):
            raise OpenRLHFConfigError(
                "local model must contain .stateset-model-revision matching model_revision"
            )
        return resolved
    if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise OpenRLHFConfigError(
            "remote model_revision must be an immutable 40-64 character commit"
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - exercised in engine image
        raise OpenRLHFConfigError(
            "huggingface_hub is required to resolve a remote model revision"
        ) from exc
    return Path(snapshot_download(repo_id=model, revision=revision)).resolve()


def build_openrlhf_command(
    request: Mapping[str, Any], *, resolved_model: Path, dataset_path: Path
) -> list[str]:
    """Translate a validated StateSet request into OpenRLHF CLI arguments."""
    algorithm = str(request.get("algorithm", "")).lower()
    if algorithm not in _ALGORITHM_ARGS:
        raise OpenRLHFConfigError(f"unsupported OpenRLHF algorithm: {algorithm!r}")
    config = _require_object(request.get("config"), "config")
    unknown = sorted(set(config) - set(_VALUE_OPTIONS) - set(_FLAG_OPTIONS))
    if unknown:
        raise OpenRLHFConfigError(
            "unsupported OpenRLHF config fields: " + ", ".join(unknown)
        )

    output_dir = Path(str(request["output_dir"])).resolve()
    artifact_dir = output_dir / "artifact"
    command = [
        sys.executable,
        "-m",
        "openrlhf.cli.train_ppo_ray",
        "--actor.model_name_or_path",
        str(resolved_model),
        "--data.prompt_dataset",
        str(dataset_path),
        "--ckpt.output_dir",
        str(artifact_dir),
        "--train.seed",
        str(request["seed"]),
        *_ALGORITHM_ARGS[algorithm],
    ]

    for key, (options, expected) in _VALUE_OPTIONS.items():
        if key in config:
            value = _positive_number(config[key], expected, key)
            if key == "dtype" and value not in {"bf16", "fp16"}:
                raise OpenRLHFConfigError("config.dtype must be 'bf16' or 'fp16'")
            if key == "top_p" and float(value) > 1:
                raise OpenRLHFConfigError("config.top_p must not exceed 1")
            for option in options:
                command.extend((option, value))
    for key, option in _FLAG_OPTIONS.items():
        if key in config:
            if not isinstance(config[key], bool):
                raise OpenRLHFConfigError(f"config.{key} must be bool")
            if config[key]:
                command.append(option)

    environment = _require_object(request.get("environment", {}), "environment")
    _require_exact_keys(environment, _ALLOWED_ENVIRONMENT_KEYS, "environment")
    environment_type = environment.get("type", "single_turn")
    if environment_type == "agent":
        function_path = _verified_file(
            environment.get("function_path"),
            environment.get("sha256"),
            "environment.function_path",
        )
        command.extend(("--train.agent_func_path", str(function_path)))
    elif environment_type != "single_turn":
        raise OpenRLHFConfigError(f"unsupported environment type: {environment_type!r}")
    elif set(environment) - {"type"}:
        raise OpenRLHFConfigError("single_turn environment accepts only the type field")

    requirements = set(request.get("requirements", ()))
    if (
        requirements.intersection({"multi_turn", "tool_use"})
        and environment_type != "agent"
    ):
        raise OpenRLHFConfigError(
            "multi_turn/tool_use requires an agent environment function"
        )
    if "async_rollouts" in requirements:
        command.append("--train.async_enable")
    if "multimodal" in requirements and config.get("max_images_per_prompt", 0) < 1:
        raise OpenRLHFConfigError(
            "multimodal requires config.max_images_per_prompt to be positive"
        )
    if "multimodal" in requirements and config.get("packing_samples", False):
        raise OpenRLHFConfigError("OpenRLHF multimodal runs cannot pack samples")

    reward = _require_object(request.get("reward", {}), "reward")
    _require_exact_keys(reward, _ALLOWED_REWARD_KEYS, "reward")
    reward_type = reward.get("type")
    if reward_type != "python":
        raise OpenRLHFConfigError(
            "OpenRLHF currently requires a content-pinned Python reward"
        )
    if set(reward) != {"type", "path", "sha256"}:
        raise OpenRLHFConfigError(
            "Python reward requires exactly type, path, and sha256"
        )
    reward_path = _verified_file(
        reward.get("path"), reward.get("sha256"), "reward.path"
    )
    command.extend(("--reward.remote_url", str(reward_path)))

    samples = config.get("samples_per_prompt", 1)
    if algorithm in {"grpo", "gspo"} and samples <= 1:
        raise OpenRLHFConfigError(
            f"{algorithm} requires config.samples_per_prompt greater than 1"
        )
    return command


def openrlhf_backend(
    *, version: str, timeout_seconds: int = 14400
) -> CommandTrainingBackend:
    """Create an OpenRLHF command backend without importing OpenRLHF."""
    return CommandTrainingBackend(
        name="openrlhf",
        version=version,
        capabilities=BackendCapabilities(
            algorithms=frozenset(_ALGORITHM_ARGS),
            features=frozenset(
                {
                    "async_rollouts",
                    "distributed",
                    "multi_turn",
                    "multimodal",
                    "tool_use",
                }
            ),
        ),
        command=(
            sys.executable,
            "-m",
            "stateset_agents.training.adapters.openrlhf",
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
    """Validate, execute, and normalize one OpenRLHF training run."""
    installed_version = importlib.metadata.version("openrlhf")
    if installed_version != expected_version:
        raise OpenRLHFConfigError(
            f"installed OpenRLHF {installed_version!r} != expected {expected_version!r}"
        )
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("protocol_version") != BACKEND_PROTOCOL_VERSION:
        raise OpenRLHFConfigError("unsupported StateSet backend protocol version")
    if Path(str(request.get("output_dir"))).resolve() != output_dir.resolve():
        raise OpenRLHFConfigError(
            "request output_dir does not match adapter output_dir"
        )
    semantic_payload = {key: request.get(key) for key in _SEMANTIC_KEYS}
    digest = hashlib.sha256(
        json.dumps(semantic_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if request.get("experiment_sha256") != digest:
        raise OpenRLHFConfigError("request experiment digest does not match payload")

    dataset_path = verify_dataset(
        str(request["dataset_uri"]), str(request["dataset_sha256"])
    )
    model_path = resolve_model(str(request["model"]), str(request["model_revision"]))
    command = build_openrlhf_command(
        request, resolved_model=model_path, dataset_path=dataset_path
    )
    start = time.monotonic()
    completed = subprocess.run(command, check=False)
    duration = time.monotonic() - start
    if completed.returncode != 0:
        raise RuntimeError(f"OpenRLHF exited {completed.returncode}")

    artifact_dir = output_dir.resolve() / "artifact"
    if not artifact_dir.is_dir() or not any(
        path.is_file() for path in artifact_dir.rglob("*")
    ):
        raise RuntimeError("OpenRLHF did not produce a reusable model artifact")
    result = {
        "protocol_version": BACKEND_PROTOCOL_VERSION,
        "backend": "openrlhf",
        "backend_version": installed_version,
        "experiment_sha256": request["experiment_sha256"],
        "artifact_uri": str(artifact_dir),
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
    "OpenRLHFConfigError",
    "build_openrlhf_command",
    "main",
    "openrlhf_backend",
    "resolve_model",
    "run_adapter",
    "verify_dataset",
]
