"""Fail-closed NeMo RL adapter for the StateSet backend protocol.

The adapter intentionally exposes only NeMo RL's built-in, single-turn math
GRPO path.  Custom environments and rewards are rejected until they can be
content-pinned and represented without changing experiment semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
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

_VALUE_OVERRIDES: dict[str, tuple[tuple[str, ...], type[Any]]] = {
    "learning_rate": (("policy.optimizer.kwargs.lr",), float),
    "train_global_batch_size": (("policy.train_global_batch_size",), int),
    "train_micro_batch_size": (("policy.train_micro_batch_size",), int),
    "num_prompts_per_step": (("grpo.num_prompts_per_step",), int),
    "num_generations_per_prompt": (("grpo.num_generations_per_prompt",), int),
    "max_num_steps": (("grpo.max_num_steps",), int),
    "max_num_epochs": (("grpo.max_num_epochs",), int),
    "max_total_sequence_length": (
        (
            "policy.max_total_sequence_length",
            "policy.generation.max_new_tokens",
            "policy.generation.vllm_cfg.max_model_len",
        ),
        int,
    ),
    "num_nodes": (("cluster.num_nodes",), int),
    "gpus_per_node": (("cluster.gpus_per_node",), int),
    "tensor_parallel_size": (
        (
            "policy.dtensor_cfg.tensor_parallel_size",
            "policy.generation.vllm_cfg.tensor_parallel_size",
        ),
        int,
    ),
    "gpu_memory_utilization": (
        ("policy.generation.vllm_cfg.gpu_memory_utilization",),
        float,
    ),
    "temperature": (("policy.generation.temperature",), float),
    "top_p": (("policy.generation.top_p",), float),
    "top_k": (("policy.generation.top_k",), int),
    "precision": (("policy.precision",), str),
    "input_key": (("+data.default.input_key",), str),
    "output_key": (("+data.default.output_key",), str),
    "environment_workers": (("env.math.num_workers",), int),
    "generation_backend": (("policy.generation.backend",), str),
}
_FLAG_OVERRIDES: dict[str, tuple[str, ...]] = {
    "activation_checkpointing": ("policy.dtensor_cfg.activation_checkpointing",),
    "colocated": ("policy.generation.colocated.enabled",),
    "shuffle": ("data.shuffle",),
}
_NONNEGATIVE: frozenset[str] = frozenset()
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
_MATH_ENVIRONMENT = {"type": "single_turn", "name": "math"}
_MATH_REWARD = {
    "type": "nemo_builtin",
    "name": "math",
    "implementation": "hf_math_verify",
}


class NemoRLConfigError(ValueError):
    """Raised when NeMo RL cannot preserve an experiment's semantics."""


def _require_object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NemoRLConfigError(f"{name} must be a JSON object")
    return value


def _decode_file_uri_path(value: str, *, platform: str | None = None) -> str:
    decoded = unquote(value)
    current_platform = sys.platform if platform is None else platform
    if current_platform == "win32" and re.match(r"^/[A-Za-z]:/", decoded):
        return decoded[1:]
    return decoded


def _verified_file(path: Path, expected_sha256: str, name: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise NemoRLConfigError(f"{name} SHA-256 must be 64 lowercase hex digits")
    resolved = path.resolve()
    if not resolved.is_file():
        raise NemoRLConfigError(f"{name} is not a file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise NemoRLConfigError(f"{name} SHA-256 does not match experiment")
    return resolved


def verify_nemo_rl_dataset(dataset_uri: str, expected_sha256: str) -> Path:
    """Verify a local JSON/JSONL response dataset before launching NeMo RL."""
    parsed = urlparse(dataset_uri)
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise NemoRLConfigError("file dataset_uri must not name a remote host")
        path = Path(_decode_file_uri_path(parsed.path))
    elif not parsed.scheme:
        path = Path(dataset_uri)
    else:
        raise NemoRLConfigError(
            "NeMo RL adapter requires a local dataset path or file URI"
        )
    if path.suffix.lower() not in {".json", ".jsonl"}:
        raise NemoRLConfigError("NeMo RL adapter requires a .json or .jsonl dataset")
    return _verified_file(path, expected_sha256, "dataset")


def resolve_nemo_rl_model(model: str, revision: str) -> Path:
    """Resolve a local marker-pinned model or immutable Hub commit."""
    local = Path(model)
    if local.exists():
        resolved = local.resolve()
        marker = resolved / ".stateset-model-revision"
        if (
            not marker.is_file()
            or marker.read_text(encoding="utf-8").strip() != revision
        ):
            raise NemoRLConfigError(
                "local model must contain .stateset-model-revision matching model_revision"
            )
        return resolved
    if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise NemoRLConfigError(
            "remote model_revision must be an immutable 40-64 character commit"
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - exercised in engine image
        raise NemoRLConfigError(
            "huggingface_hub is required to resolve a remote model revision"
        ) from exc
    return Path(snapshot_download(repo_id=model, revision=revision)).resolve()


def resolve_nemo_rl_source_root() -> Path:
    """Locate the version-pinned NeMo RL checkout that owns the engine package."""
    spec = importlib.util.find_spec("nemo_rl")
    if spec is None or spec.origin is None:
        raise NemoRLConfigError("installed nemo_rl package cannot be located")
    root = Path(spec.origin).resolve().parent.parent
    required = (
        root / "examples" / "run_grpo.py",
        root / "examples" / "configs" / "grpo_math_1B.yaml",
    )
    if not all(path.is_file() for path in required):
        raise NemoRLConfigError(
            "NeMo RL adapter requires an installed source checkout with examples"
        )
    return root


def _value(value: Any, expected: type[Any], name: str) -> Any:
    valid = isinstance(value, expected)
    if expected is float:
        valid = isinstance(value, numbers.Real)
    if isinstance(value, bool) or not valid:
        raise NemoRLConfigError(f"config.{name} must be {expected.__name__}")
    if expected in (int, float):
        invalid = value < 0 if name in _NONNEGATIVE else value <= 0
        if invalid:
            qualifier = "non-negative" if name in _NONNEGATIVE else "positive"
            raise NemoRLConfigError(f"config.{name} must be {qualifier}")
    return value


def _override(key: str, value: Any) -> str:
    if isinstance(value, bool):
        encoded = "true" if value else "false"
    else:
        encoded = json.dumps(value, separators=(",", ":"))
    return f"{key}={encoded}"


def build_nemo_rl_command(
    request: Mapping[str, Any],
    *,
    source_root: Path,
    resolved_model: Path,
    dataset_path: Path,
) -> list[str]:
    """Translate one StateSet request to NeMo RL's official GRPO launcher."""
    algorithm = str(request.get("algorithm", "")).lower()
    if algorithm != "grpo":
        raise NemoRLConfigError(f"unsupported NeMo RL algorithm: {algorithm!r}")
    config = _require_object(request.get("config"), "config")
    unknown = sorted(set(config) - set(_VALUE_OVERRIDES) - set(_FLAG_OVERRIDES))
    if unknown:
        raise NemoRLConfigError(
            "unsupported NeMo RL config fields: " + ", ".join(unknown)
        )
    required = {"generation_backend", "max_num_steps", "num_generations_per_prompt"}
    missing = sorted(required - set(config))
    if missing:
        raise NemoRLConfigError(
            "required NeMo RL config fields are missing: " + ", ".join(missing)
        )
    if config.get("generation_backend") != "vllm":
        raise NemoRLConfigError("config.generation_backend must be explicitly 'vllm'")
    environment = _require_object(request.get("environment", {}), "environment")
    if dict(environment) != _MATH_ENVIRONMENT:
        raise NemoRLConfigError(
            "NeMo RL adapter currently requires the single-turn math environment"
        )
    reward = _require_object(request.get("reward", {}), "reward")
    if dict(reward) != _MATH_REWARD:
        raise NemoRLConfigError(
            "NeMo RL adapter currently requires its hf_math_verify built-in reward"
        )
    unsupported = set(request.get("requirements", ())) - {"distributed"}
    if unsupported:
        raise NemoRLConfigError(
            "unsupported NeMo RL requirements: " + ", ".join(sorted(unsupported))
        )

    artifact = Path(str(request["output_dir"])).resolve() / "artifact"
    log_dir = Path(str(request["output_dir"])).resolve() / "logs"
    runner = source_root / "examples" / "run_grpo.py"
    template = source_root / "examples" / "configs" / "grpo_math_1B.yaml"
    seed = request["seed"]
    command = [
        sys.executable,
        str(runner),
        "--config",
        str(template),
        _override("policy.model_name", str(resolved_model)),
        _override("policy.tokenizer.name", str(resolved_model)),
        "data.train.dataset_name=ResponseDataset",
        _override("+data.train.data_path", str(dataset_path)),
        "data.train.split_validation_size=0",
        "data.validation=null",
        "data.default.prompt_file=null",
        "data.default.system_prompt_file=null",
        "data.default.processor=math_hf_data_processor",
        "data.default.env_name=math",
        "env.math.math_verify_impl=hf_math_verify",
        _override("grpo.seed", seed),
        _override("data.train.seed", seed),
        "grpo.max_rollout_turns=1",
        "grpo.val_period=0",
        "grpo.val_at_start=false",
        "grpo.val_at_end=false",
        "checkpointing.enabled=true",
        _override("checkpointing.checkpoint_dir", str(artifact)),
        "checkpointing.save_period=1",
        "checkpointing.keep_top_k=1",
        "checkpointing.save_consolidated=true",
        "checkpointing.save_optimizer=true",
        _override("logger.log_dir", str(log_dir)),
        "logger.wandb_enabled=false",
        "logger.tensorboard_enabled=false",
        "logger.mlflow_enabled=false",
        "logger.swanlab_enabled=false",
        "logger.monitor_gpus=false",
    ]
    for name, (keys, expected) in _VALUE_OVERRIDES.items():
        if name not in config:
            continue
        value = _value(config[name], expected, name)
        if name in {"top_p", "gpu_memory_utilization"} and float(value) > 1:
            raise NemoRLConfigError(f"config.{name} must not exceed 1")
        if name == "temperature" and float(value) > 2:
            raise NemoRLConfigError("config.temperature must not exceed 2")
        if name == "precision" and value not in {"bfloat16", "float16", "float32"}:
            raise NemoRLConfigError(
                "config.precision must be bfloat16, float16, or float32"
            )
        if name in {"input_key", "output_key"} and not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_.-]*", str(value)
        ):
            raise NemoRLConfigError(f"config.{name} contains invalid characters")
        for key in keys:
            command.append(_override(key, value))
    for name, keys in _FLAG_OVERRIDES.items():
        if name not in config:
            continue
        if not isinstance(config[name], bool):
            raise NemoRLConfigError(f"config.{name} must be bool")
        for key in keys:
            command.append(_override(key, config[name]))
    if config.get("num_generations_per_prompt", 1) <= 1:
        raise NemoRLConfigError(
            "grpo requires config.num_generations_per_prompt greater than 1"
        )
    return command


def nemo_rl_backend(
    *, version: str, timeout_seconds: int = 14400
) -> CommandTrainingBackend:
    """Create a NeMo RL backend without importing the optional engine."""
    return CommandTrainingBackend(
        name="nemo-rl",
        version=version,
        capabilities=BackendCapabilities(
            algorithms=frozenset({"grpo"}),
            features=frozenset({"distributed"}),
        ),
        command=(
            sys.executable,
            "-m",
            "stateset_agents.training.adapters.nemo_rl",
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
    """Validate, execute, and normalize one NeMo RL run."""
    installed_version = importlib.metadata.version("nemo-rl")
    if installed_version != expected_version:
        raise NemoRLConfigError(
            f"installed nemo-rl {installed_version!r} != expected {expected_version!r}"
        )
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("protocol_version") != BACKEND_PROTOCOL_VERSION:
        raise NemoRLConfigError("unsupported StateSet backend protocol version")
    if Path(str(request.get("output_dir"))).resolve() != output_dir.resolve():
        raise NemoRLConfigError("request output_dir does not match adapter output_dir")
    semantic_payload = {key: request.get(key) for key in _SEMANTIC_KEYS}
    digest = hashlib.sha256(
        json.dumps(semantic_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if request.get("experiment_sha256") != digest:
        raise NemoRLConfigError("request experiment digest does not match payload")

    dataset = verify_nemo_rl_dataset(
        str(request["dataset_uri"]), str(request["dataset_sha256"])
    )
    model = resolve_nemo_rl_model(str(request["model"]), str(request["model_revision"]))
    source_root = resolve_nemo_rl_source_root()
    command = build_nemo_rl_command(
        request,
        source_root=source_root,
        resolved_model=model,
        dataset_path=dataset,
    )
    start = time.monotonic()
    completed = subprocess.run(command, cwd=source_root, check=False)
    duration = time.monotonic() - start
    if completed.returncode != 0:
        raise RuntimeError(f"NeMo RL exited {completed.returncode}")
    artifact = output_dir.resolve() / "artifact"
    if not artifact.is_dir() or not any(path.is_file() for path in artifact.rglob("*")):
        raise RuntimeError("NeMo RL did not produce a reusable checkpoint artifact")
    result = {
        "protocol_version": BACKEND_PROTOCOL_VERSION,
        "backend": "nemo-rl",
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
    "NemoRLConfigError",
    "build_nemo_rl_command",
    "main",
    "nemo_rl_backend",
    "resolve_nemo_rl_model",
    "resolve_nemo_rl_source_root",
    "run_adapter",
    "verify_nemo_rl_dataset",
]
