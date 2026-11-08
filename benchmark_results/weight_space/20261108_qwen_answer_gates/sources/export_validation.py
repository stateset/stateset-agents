"""Identity and behavioral evidence for reloaded session exports."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal

from ...core.errors import ConfigurationError
from .behavior import BehaviorLimits, BehaviorMetrics, ReferenceCheck

if TYPE_CHECKING:
    import torch

    from .session import WeightSpaceSession

__all__ = ["ExportVerification"]

REPORT_NAME = "weight_space_verification.json"
ARTIFACT_NAMES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "weight_space_manifest.json",
)


@dataclass(frozen=True)
class ExportVerification:
    """Validation of exact artifact bytes in one PEFT loading mode.

    Metrics compare reloaded outputs to the live edited session. Reference
    checks additionally compare to the registered fixed model states. Evidence
    covers the declared cases and runtime only, not unseen prompts or kernels.
    """

    verification_id: str
    accepted: bool
    violations: tuple[str, ...]
    mode: Literal["adapter", "merged"]
    artifact_sha256: tuple[tuple[str, str], ...]
    base_state_sha256: str
    edit_ids: tuple[str, ...]
    cases_sha256: str
    metrics: BehaviorMetrics
    limits: BehaviorLimits
    reference_checks: tuple[ReferenceCheck, ...]
    runtime: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _ExportReceipt:
    artifact_sha256: tuple[tuple[str, str], ...]
    edit_ids: tuple[str, ...]


def _artifact_hashes(root: Path) -> tuple[tuple[str, str], ...]:
    hashes = []
    for name in ARTIFACT_NAMES:
        digest = hashlib.sha256()
        with (root / name).open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        hashes.append((name, digest.hexdigest()))
    return tuple(hashes)


def _load_adapter(root: Path, receipt: _ExportReceipt, fresh: Any, mode: str) -> Any:
    """Load verified private copies so concurrent source writes cannot alter reads."""
    from peft import PeftModel

    with TemporaryDirectory(prefix="stateset-weight-export-") as directory:
        snapshot = Path(directory)
        for name in ARTIFACT_NAMES:
            shutil.copyfile(root / name, snapshot / name)
        if _artifact_hashes(snapshot) != receipt.artifact_sha256:
            raise ConfigurationError("export artifacts changed while snapshotting")
        model = PeftModel.from_pretrained(
            fresh,
            str(snapshot),
            is_trainable=False,
            local_files_only=True,
            use_safetensors=True,
        ).eval()
        if mode == "merged":
            model = model.merge_and_unload(safe_merge=True).eval()
        return model


def _runtime(model: Any) -> tuple[tuple[str, str], ...]:
    import torch

    values = {
        "torch": str(torch.__version__),
        "peft": version("peft"),
        "transformers": version("transformers"),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "devices": ",".join(sorted({str(p.device) for p in model.parameters()})),
        "dtypes": ",".join(sorted({str(p.dtype) for p in model.parameters()})),
        "attention_implementation": str(
            getattr(getattr(model, "config", None), "_attn_implementation", None)
        ),
    }
    if hasattr(getattr(model, "config", None), "to_json_string"):
        values["base_config_sha256"] = hashlib.sha256(
            model.config.to_json_string().encode()
        ).hexdigest()
    return tuple(sorted(values.items()))


def _tensor_hash(tensor: torch.Tensor) -> str:
    """Hash exact bytes with bounded CPU transfer windows, including bfloat16."""
    import torch

    if tensor.layout != torch.strided or tensor.device.type == "meta":
        raise ConfigurationError(
            "export verification requires materialized dense tensors"
        )
    digest = hashlib.sha256(f"{tensor.dtype}:{tuple(tensor.shape)}".encode())
    flat = tensor.detach().reshape(-1)
    for start in range(0, flat.numel(), 1_000_000):
        raw = (
            flat[start : start + 1_000_000].cpu().contiguous().view(torch.uint8).numpy()
        )
        digest.update(raw.tobytes())
    return digest.hexdigest()


def _verify_base(session: WeightSpaceSession, fresh: Any) -> str:
    """Reject shared storage and a wrong reconstructed base before PEFT mutates it."""
    if type(fresh) is not type(session.model):
        raise ConfigurationError("fresh base model class differs from the session")
    if {id(m) for m in fresh.modules()} & {id(m) for m in session.model.modules()}:
        raise ConfigurationError("fresh base model shares modules with the session")

    def tensors(model: Any) -> dict[str, torch.Tensor]:
        return {
            **{
                f"parameter:{n}": p
                for n, p in model.named_parameters(remove_duplicate=False)
            },
            **{
                f"buffer:{n}": b for n, b in model.named_buffers(remove_duplicate=False)
            },
        }

    current, base = tensors(session.model), tensors(fresh)
    if current.keys() != base.keys():
        raise ConfigurationError("fresh base tensor names differ from the session")

    def storage(tensor: torch.Tensor) -> tuple[Any, int] | None:
        if tensor.device.type == "meta":
            raise ConfigurationError("fresh base must have materialized tensors")
        return (
            (tensor.device, tensor.untyped_storage().data_ptr())
            if tensor.numel()
            else None
        )

    current_storage = {name: storage(t) for name, t in current.items()}
    base_storage = {name: storage(t) for name, t in base.items()}
    shared = set(current_storage.values()) & set(base_storage.values())
    if shared - {None}:
        raise ConfigurationError("fresh base shares tensor storage with the session")

    def aliases(locations: dict[str, tuple[Any, int] | None]) -> set[tuple[str, ...]]:
        groups: dict[tuple[Any, int], list[str]] = {}
        for name, location in locations.items():
            if location is not None:
                groups.setdefault(location, []).append(name)
        return {tuple(sorted(names)) for names in groups.values() if len(names) > 1}

    if aliases(current_storage) != aliases(base_storage):
        raise ConfigurationError(
            "fresh base tensor storage aliases differ from the session"
        )
    expected = dict(current)
    replaced = set()
    for edit in session._edits:
        name = f"parameter:{edit.module}.weight"
        if name not in replaced:
            expected[name] = edit.previous
            replaced.add(name)
    digest = hashlib.sha256(type(fresh).__qualname__.encode())
    for name in sorted(expected):
        wanted = _tensor_hash(expected[name])
        if wanted != _tensor_hash(base[name]):
            raise ConfigurationError(
                f"fresh base tensor differs from session base: {name}"
            )
        digest.update(name.encode())
        digest.update(wanted.encode())
    return digest.hexdigest()
