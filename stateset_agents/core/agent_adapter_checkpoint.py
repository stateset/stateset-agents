"""Named LoRA artifact layout and execution-state restoration.

Attachment is called inside Agent's retained adapter-effect boundary. Nothing
from a partially loaded collection is published as an initialized model.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Any

from .agent_config import AgentConfig
from .agent_initialization import _checked_component
from .checkpoint_io import validate_local_shard_indexes

_FIELDS = {
    "names",
    "primary",
    "active",
    "enabled",
    "merged",
    "wrapper_disabled",
    "base_model",
    "revision",
}


def _names(value: Any, *, empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not empty):
        raise ValueError("Adapter names must be a list")
    for name in value:
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or any(character in name for character in ("/", "\\", ":", "."))
            or any(ord(character) < 32 for character in name)
            or name.endswith(" ")
        ):
            raise ValueError("Adapter names must be safe, nonempty directory names")
    if len({name.casefold() for name in value}) != len(value):
        raise ValueError(
            "Adapter names must be unique across filesystem case conventions"
        )
    return value


def _base_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Adapter checkpoint has no base model source")
    candidate = Path(value).expanduser()
    return str(candidate.resolve()) if candidate.is_dir() else value


def _validate(spec: Any) -> dict[str, Any]:
    if not isinstance(spec, dict) or set(spec) != _FIELDS:
        raise ValueError("Invalid adapter checkpoint metadata")
    names = _names(spec["names"])
    active = _names(spec["active"], empty=True)
    merged = _names(spec["merged"], empty=True)
    if (
        not isinstance(spec["primary"], str)
        or spec["primary"] not in names
        or not set(active + merged).issubset(names)
        or type(spec["enabled"]) is not bool
        or type(spec["wrapper_disabled"]) is not bool
        or (
            spec["revision"] is not None
            and (not isinstance(spec["revision"], str) or not spec["revision"].strip())
        )
    ):
        raise ValueError("Invalid adapter execution state")
    _base_name(spec["base_model"])
    return copy.deepcopy(spec)


def capture_adapters(model: Any, config: AgentConfig) -> dict[str, Any] | None:
    """Capture coherent LoRA state; retain legacy handling for a default tuner."""
    adapters = getattr(model, "peft_config", None)
    if adapters is None:
        return None
    if not isinstance(adapters, dict) or not adapters:
        raise ValueError("Adapter configuration must be a nonempty mapping")
    names = _names(list(adapters))
    if any(getattr(item, "peft_type", None) != "LORA" for item in adapters.values()):
        if names == ["default"]:
            return None
        raise ValueError("Named adapter checkpoints currently require LoRA tuners")
    status = model.get_model_status()
    if set(status.available_adapters) != set(names):
        raise ValueError("Adapter configuration and installed layers disagree")
    # An irregular layer selection/enable/merge state cannot be reproduced by
    # PEFT's model-level controls. Refuse it before saving any artifacts.
    spec = {
        "names": names,
        "primary": model.active_adapter,
        "active": status.active_adapters,
        "enabled": status.enabled,
        "merged": status.merged_adapters,
        "wrapper_disabled": getattr(model, "_adapters_disabled", False),
        "base_model": "",
        "revision": None,
    }
    fallback_revision = (config.model_kwargs or {}).get(
        "revision", config.model_revision
    )
    sources = {
        (
            _base_name(item.base_model_name_or_path or config.model_name),
            getattr(item, "revision", None) or fallback_revision,
        )
        for item in adapters.values()
    }
    if len(sources) != 1:
        raise ValueError(
            "Adapters in a checkpoint must share one base model and revision"
        )
    spec["base_model"], spec["revision"] = sources.pop()
    spec = _validate(spec)
    spec["merged"] = _merge_order(model, spec["merged"])
    return spec


def _merge_order(model: Any, merged: list[str]) -> list[str]:
    # Model status sorts names, losing execution order. Layer receipts preserve
    # it, including when adapters target different subsets of the model.
    predecessors: dict[str, set[str]] = {name: set() for name in merged}
    if not merged:
        return []
    for layer in model.get_layer_status():
        for earlier, later in zip(
            layer.merged_adapters, layer.merged_adapters[1:], strict=False
        ):
            predecessors[later].add(earlier)
    try:
        return list(
            TopologicalSorter(
                {name: sorted(predecessors[name]) for name in sorted(predecessors)}
            ).static_order()
        )
    except CycleError as error:
        raise ValueError("Adapter merge order differs across layers") from error


@dataclass
class _AdapterCheckpoint:
    path: Path
    spec: dict[str, Any]

    def __post_init__(self) -> None:
        self.spec = _validate(self.spec)
        base = Path(self.spec["base_model"]).expanduser()
        if base.is_absolute() or base.is_dir():
            try:
                base.resolve().relative_to(self.path.resolve())
            except ValueError:
                pass
            else:
                raise ValueError(
                    "Adapter checkpoint cannot replace or contain its own base model"
                )

    def directory(self, name: str) -> Path:
        return self.path if name == "default" else self.path / name

    def request(self) -> dict[str, Any]:
        return {
            "kind": "checkpoint",
            "path": str(self.path),
            "adapters": copy.deepcopy(self.spec),
        }

    def validate_files(self) -> None:
        """Check each declared adapter and reject undeclared adapter directories."""
        _validate(self.spec)
        expected = {
            self.directory(name) / "adapter_config.json" for name in self.spec["names"]
        }
        if set(self.path.rglob("adapter_config.json")) != expected:
            raise ValueError(
                "Saved adapter directories differ from the declared collection"
            )
        if (self.path / "config.json").exists():
            raise ValueError(
                "Adapter checkpoint unexpectedly contains a full-model configuration"
            )
        for name in self.spec["names"]:
            directory = self.directory(name)
            if directory.is_symlink():
                raise ValueError("Adapter directory must not be a symlink")
            config_path = directory / "adapter_config.json"
            if config_path.is_symlink() or not config_path.is_file():
                raise ValueError("Adapter configuration must be a regular file")
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(config, dict) or config.get("peft_type") != "LORA":
                raise ValueError("Expected a saved LoRA configuration")
            base = _base_name(
                config.get("base_model_name_or_path") or self.spec["base_model"]
            )
            revision = config.get("revision") or self.spec["revision"]
            if base != self.spec["base_model"] or revision != self.spec["revision"]:
                raise ValueError(
                    "Saved adapters disagree about the recorded base model"
                )
            weights = [
                directory / filename
                for filename in ("adapter_model.safetensors", "adapter_model.bin")
            ]
            if not any(item.is_file() and not item.is_symlink() for item in weights):
                raise ValueError(f"Adapter {name!r} is missing saved weights")
            validate_local_shard_indexes(directory)

    def attach(self, base: Any, peft_model: Any) -> Any:
        """Restore all adapters and their selection inside one owned effect."""
        names = self.spec["names"]
        model = _checked_component(
            "Adapter collection",
            peft_model.from_pretrained(
                base,
                str(self.directory(names[0])),
                adapter_name=names[0],
                is_trainable=False,
                local_files_only=True,
            ),
            owned=True,
        )
        for name in names[1:]:
            model.load_adapter(
                str(self.directory(name)),
                adapter_name=name,
                is_trainable=False,
                local_files_only=True,
            )
        model.set_adapter(self.spec["primary"])
        if model.active_adapters != self.spec["active"]:
            model.base_model.set_adapter(self.spec["active"])
        if self.spec["enabled"]:
            model.base_model.enable_adapter_layers()
        else:
            model.base_model.disable_adapter_layers()
        if hasattr(model, "_adapters_disabled"):
            model._adapters_disabled = self.spec["wrapper_disabled"]
        elif self.spec["wrapper_disabled"]:
            raise ValueError(
                "This PEFT runtime cannot restore the saved wrapper disable state"
            )
        if self.spec["merged"]:
            model.base_model.merge_adapter(
                adapter_names=self.spec["merged"], safe_merge=True
            )
        # This is an inference snapshot. set_adapter() can otherwise re-enable
        # gradients even though all adapters were loaded with is_trainable=False.
        model.requires_grad_(False)
        model.eval()
        status = model.get_model_status()
        if (
            set(model.peft_config) != set(names)
            or status.active_adapters != self.spec["active"]
            or status.enabled != self.spec["enabled"]
            or status.merged_adapters != sorted(self.spec["merged"])
            or _merge_order(model, status.merged_adapters) != self.spec["merged"]
        ):
            raise ValueError(
                "PEFT did not restore the recorded adapter execution state"
            )
        return model
