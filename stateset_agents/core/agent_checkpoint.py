"""Portable agent artifacts with explicit restoration and staged publication.

These snapshots contain inference components, not optimizer state, conversations,
or externally owned rollout engines. Stop training/generation before saving.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .agent_adapter_checkpoint import _AdapterCheckpoint, capture_adapters
from .agent_backends import StubGenerationConfig, StubModel, StubTokenizer
from .agent_config import AgentConfig
from .checkpoint_io import validate_local_shard_indexes

if TYPE_CHECKING:
    from .agent import Agent, MultiTurnAgent

_DESCRIPTOR = "agent_checkpoint.json"
_KINDS = {"configuration", "stub", "full_model", "adapter"}


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return value


def _write_object(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def _artifact_kind(path: Path) -> str | None:
    adapter = (path / "adapter_config.json").is_file()
    full = (path / "config.json").is_file()
    if adapter and full:
        raise ValueError(
            "Ambiguous checkpoint: both full-model and adapter configurations"
        )
    names: tuple[str, ...]
    if adapter:
        names = ("adapter_model.safetensors", "adapter_model.bin")
    elif full:
        names = (
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
    else:
        # A partial save must not masquerade as a configuration-only snapshot.
        if (
            any(path.glob("*.safetensors"))
            or any(path.glob("*.bin"))
            or any(path.glob("*.index.json"))
        ):
            raise ValueError("Checkpoint weights are missing their model configuration")
        return None
    if not any((path / name).is_file() for name in names):
        raise ValueError("Checkpoint configuration exists but its weights are missing")
    validate_local_shard_indexes(path)
    return "adapter" if adapter else "full_model"


def _verify_metadata(path: Path) -> bool:
    """Inspect published metadata without reading large model weight files.

    Initialization will verify the entire inventory while holding the same lock.
    Legacy directories lack content receipts, so they keep full verification.
    """
    from stateset_agents.training.checkpoint_publication import (
        COMPLETION_FILE,
        LOCK_FILE,
    )

    marker = path / COMPLETION_FILE
    if not marker.exists() and not marker.is_symlink():
        if (path / _DESCRIPTOR).exists():
            raise ValueError("Checkpoint publication is incomplete")
        return False
    if marker.is_symlink() or not marker.is_file():
        raise ValueError("Invalid checkpoint completion marker")
    state = _json_object(marker)
    if (
        set(state) != {"schema_version", "kind", "files"}
        or type(state["schema_version"]) is not int
        or state["schema_version"] != 2
        or state["kind"] != "agent"
        or not isinstance(state["files"], dict)
    ):
        raise ValueError("Invalid agent checkpoint completion marker")
    files = set()
    for item in path.rglob("*"):
        if item.is_symlink() or not (item.is_file() or item.is_dir()):
            raise ValueError(
                "Checkpoint artifacts must be regular files or directories"
            )
        if item.is_file() and item != marker and item.name != LOCK_FILE:
            files.add(item.relative_to(path).as_posix())
    if files != set(state["files"]) or not {"agent_config.json", _DESCRIPTOR}.issubset(
        files
    ):
        raise ValueError("Checkpoint artifacts differ from the completed publication")
    for name in ("agent_config.json", _DESCRIPTOR):
        if (
            hashlib.sha256((path / name).read_bytes()).hexdigest()
            != state["files"][name]
        ):
            raise ValueError(
                "Checkpoint metadata differs from the completed publication"
            )
    return True


def _read_snapshot(
    path: Path, *, verify_weights: bool = True
) -> tuple[dict[str, Any], dict[str, Any], str]:
    from stateset_agents.training.checkpoint_publication import (
        COMPLETION_FILE,
        _inventory,
        validate_checkpoint_publication,
    )

    if path.is_symlink():
        raise ValueError("Agent checkpoint must not be a symlink")
    if not path.is_dir():
        raise FileNotFoundError(f"Agent checkpoint directory does not exist: {path}")
    published = (
        validate_checkpoint_publication(
            path, required=(path / _DESCRIPTOR).exists(), kind="agent"
        )
        if verify_weights
        else _verify_metadata(path)
    )
    # New snapshots are identified by their verified completion marker. Legacy
    # ones need a content inventory to fence changes before deferred initialize.
    identity = hashlib.sha256(
        (path / COMPLETION_FILE).read_bytes()
        if published
        else json.dumps(_inventory(path), sort_keys=True).encode()
    ).hexdigest()
    config = _json_object(path / "agent_config.json")
    if published:
        descriptor = _json_object(path / _DESCRIPTOR)
        version = descriptor.get("schema_version")
        fields = {"schema_version", "kind", "generation_config", "stub"}
        if version == 2:
            fields.add("adapters")
        if (
            set(descriptor) != fields
            or type(version) is not int
            or version not in {1, 2}
            or not isinstance(descriptor["kind"], str)
            or descriptor["kind"] not in _KINDS
            or (version == 2 and descriptor["kind"] != "adapter")
            or (
                descriptor["generation_config"] is not None
                and not isinstance(descriptor["generation_config"], dict)
            )
        ):
            raise ValueError("Invalid agent checkpoint descriptor")
        kind = descriptor["kind"]
        actual: str | None
        if version == 2:
            _AdapterCheckpoint(path, descriptor["adapters"]).validate_files()
            actual = "adapter"
        else:
            actual = _artifact_kind(path)
        if kind in {"full_model", "adapter"} and actual != kind:
            raise ValueError("Agent checkpoint kind does not match its saved artifacts")
        if kind in {"stub", "configuration"} and actual is not None:
            raise ValueError("Unexpected model weights in agent checkpoint")
        if kind == "stub" and not isinstance(descriptor["stub"], dict):
            raise ValueError("Agent checkpoint is missing stub state")
        if kind != "configuration" and descriptor["generation_config"] is None:
            raise ValueError("Agent checkpoint is missing generation settings")
    else:
        descriptor = {
            "schema_version": 1,
            "kind": _artifact_kind(path) or "configuration",
            "generation_config": None,
            "stub": None,
        }
    return config, descriptor, identity


def save_checkpoint(agent: Agent, path: Path, *, save_model: bool) -> None:
    """Publish a complete inference snapshot, retaining a previous one on error."""
    from stateset_agents.training.checkpoint_publication import checkpoint_write

    if save_model:
        agent._ensure_backend_usable()
    config = asdict(agent.config)
    # Never stringify unsupported runtime objects into an unrestorable config.
    json.dumps(config, allow_nan=False)
    descriptor: dict[str, Any] = {
        "schema_version": 1,
        "kind": "configuration",
        "generation_config": None,
        "stub": None,
    }
    model, tokenizer, generation_config = (
        agent.model,
        agent.tokenizer,
        agent.generation_config,
    )
    if save_model:
        if model is None or tokenizer is None or generation_config is None:
            raise ValueError(
                "Initialize all agent components before saving model artifacts"
            )
        if isinstance(model, StubModel):
            if (
                type(model) is not StubModel
                or type(tokenizer) is not StubTokenizer
                or not isinstance(generation_config, StubGenerationConfig)
            ):
                raise ValueError(
                    "Custom stub subclasses need their own checkpoint serializer"
                )
            descriptor["kind"] = "stub"
            descriptor["stub"] = {
                "responses": list(model._responses),
                "index": model._index,
                "training": model.training,
                "model_config": vars(model.config).copy(),
                "tokenizer": vars(tokenizer).copy(),
            }
            descriptor["generation_config"] = asdict(generation_config)
        else:
            descriptor["generation_config"] = generation_config.to_dict()
            adapters = capture_adapters(model, agent.config)
            if adapters is not None:
                # Validate the destination before any model serialization effect.
                _AdapterCheckpoint(path, adapters)
                descriptor.update(schema_version=2, kind="adapter", adapters=adapters)
        json.dumps(descriptor, allow_nan=False)

    with checkpoint_write(path, kind="agent") as staging:
        if not save_model and (path / "agent_config.json").exists():
            _, previous, _ = _read_snapshot(path)
            if previous["kind"] != "configuration":
                raise ValueError(
                    "Use a new path for a configuration-only snapshot; existing weights are retained"
                )
        if save_model and descriptor["kind"] != "stub":
            # The model-save branch above already validated these handles.
            assert model is not None and tokenizer is not None
            model.save_pretrained(staging)
            tokenizer.save_pretrained(staging)
            if descriptor["schema_version"] == 2:
                _AdapterCheckpoint(staging, descriptor["adapters"]).validate_files()
            else:
                descriptor["kind"] = _artifact_kind(staging)
            if descriptor["kind"] is None:
                raise ValueError("Model save did not produce restorable weights")
            if not (staging / "tokenizer_config.json").is_file():
                raise ValueError("Tokenizer save did not produce tokenizer_config.json")
        _write_object(staging / "agent_config.json", config)
        _write_object(staging / _DESCRIPTOR, descriptor)


def _effective_config(
    path: Path, saved: dict[str, Any], descriptor: dict[str, Any]
) -> AgentConfig:
    config = copy.deepcopy(saved)
    kind = descriptor["kind"]
    if kind in {"full_model", "adapter"}:
        config.update(use_stub_model=False, peft_config=None)
        tokenizer_options = dict(config.get("tokenizer_kwargs") or {})
        for option in ("revision", "subfolder"):
            tokenizer_options.pop(option, None)
        # The tokenizer loader below always selects the local saved tokenizer.
        config["tokenizer_kwargs"] = tokenizer_options
        if kind == "full_model":
            config.update(
                model_name=str(path),
                model_revision=None,
                use_peft=False,
                peft_path=None,
            )
            model_options = dict(config.get("model_kwargs") or {})
            for option in ("revision", "subfolder"):
                model_options.pop(option, None)
            model_options["local_files_only"] = True
            config["model_kwargs"] = model_options
        else:
            if descriptor["schema_version"] == 2:
                adapter = descriptor["adapters"]
                base = adapter["base_model"]
                config["model_revision"] = adapter["revision"]
                model_options = dict(config.get("model_kwargs") or {})
                model_options.pop("revision", None)
                config["model_kwargs"] = model_options
            else:
                adapter = _json_object(path / "adapter_config.json")
                base = adapter.get("base_model_name_or_path") or config["model_name"]
            if not isinstance(base, str) or not base.strip():
                raise ValueError("Adapter checkpoint has no base model source")
            config.update(model_name=base, use_peft=True, peft_path=str(path))
            if adapter.get("revision") is not None:
                config["model_revision"] = adapter["revision"]
    elif kind == "stub":
        config.update(
            use_stub_model=True, use_peft=False, peft_config=None, peft_path=None
        )
    return AgentConfig(**config)


@dataclass
class _CheckpointRestore:
    path: Path
    identity: str
    descriptor: dict[str, Any]

    @contextmanager
    def open(self) -> Iterator[None]:
        """Revalidate and lock all artifacts across deferred component loading."""
        from stateset_agents.training.checkpoint_publication import checkpoint_read

        with checkpoint_read(self.path, kind="agent"):
            _, _, identity = _read_snapshot(self.path)
            if identity != self.identity:
                raise ValueError(
                    "Checkpoint changed since load; create a new agent from the checkpoint"
                )
            yield

    def tokenizer(self, config: AgentConfig) -> Any:
        from transformers import AutoTokenizer

        # The checkpoint is local and local_files_only overrides caller options.
        return AutoTokenizer.from_pretrained(  # nosec: B615
            str(self.path),
            trust_remote_code=config.trust_remote_code,
            **{**(config.tokenizer_kwargs or {}), "local_files_only": True},
        )

    def apply(self, agent: Agent) -> None:
        generation = copy.deepcopy(self.descriptor["generation_config"])
        if self.descriptor["kind"] == "stub":
            model, tokenizer = agent.model, agent.tokenizer
            if not isinstance(model, StubModel) or not isinstance(
                tokenizer, StubTokenizer
            ):
                raise ValueError(
                    "Stub checkpoint requires stub model and tokenizer components"
                )
            state = copy.deepcopy(self.descriptor["stub"])
            responses, index = state["responses"], state["index"]
            if (
                not isinstance(responses, list)
                or not responses
                or not all(isinstance(r, str) for r in responses)
            ):
                raise ValueError("Invalid saved stub responses")
            if type(index) is not int or index < 0:
                raise ValueError("Invalid saved stub cursor")
            model._responses = responses
            model._index = index
            model.train(state["training"])
            model.config.__dict__.update(state["model_config"])
            tokenizer.__dict__.update(state["tokenizer"])
            agent.generation_config = StubGenerationConfig(**generation)
        elif generation is not None:
            from transformers import GenerationConfig

            agent.generation_config = GenerationConfig.from_dict(generation)


async def load_checkpoint(path: Path, *, load_model: bool) -> MultiTurnAgent:
    """Construct an agent with deferred restoration bound to these artifact bytes."""
    from stateset_agents.training.checkpoint_publication import checkpoint_read

    from .agent import MultiTurnAgent

    path = path.expanduser().absolute()
    with checkpoint_read(path, kind="agent"):
        saved, descriptor, identity = _read_snapshot(path, verify_weights=False)
        config = _effective_config(path, saved, descriptor)
    restore = _CheckpointRestore(path, identity, descriptor)
    agent = MultiTurnAgent(config)
    agent._checkpoint_restore = restore
    if descriptor["schema_version"] == 2:
        agent._checkpoint_adapters = _AdapterCheckpoint(path, descriptor["adapters"])
    if descriptor["kind"] in {"full_model", "adapter"}:
        agent._tokenizer_loader = restore.tokenizer
    if load_model:
        await agent.initialize()
    return agent
