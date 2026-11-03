"""Apply an :class:`~.edits.EditSpec` to a safetensors checkpoint and verify the result.

A patched checkpoint is a full copy of the base directory (config, tokenizer,
every shard) with only the shards the spec touches rewritten, plus a
``weight_space_spec.json`` manifest recording the spec, its content hash and
how many weights changed. :func:`verify_checkpoint` re-derives the expected
patch from the spec and checks every tensor of every shard against the base,
so a checkpoint can be audited without trusting whoever produced it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .edits import (
    EditSpec,
    SpecMismatch,
    TensorLayout,
    TokenRowScale,
    apply_spec,
    verify_spec,
)

__all__ = [
    "MANIFEST_NAME",
    "ZERO_CENTERED_NORM_FAMILIES",
    "apply_spec_to_checkpoint",
    "checkpoint_layout",
    "checkpoint_shapes",
    "load_tensors",
    "norm_convention",
    "resolve_checkpoint",
    "verify_checkpoint",
]

MANIFEST_NAME = "weight_space_spec.json"


def _safetensors() -> Any:
    try:
        import safetensors
        import safetensors.torch
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(
            "weight-edit checkpoints need 'safetensors' "
            "(pip install stateset-agents[training])"
        ) from exc
    return safetensors


def resolve_checkpoint(base: str | Path, revision: str | None = None) -> Path:
    """A local checkpoint directory, downloading ``base`` from the Hub if needed."""
    path = Path(base)
    if path.is_dir():
        return path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(
            f"{base!r} is not a directory and huggingface_hub is not installed"
        ) from exc
    return Path(snapshot_download(str(base), revision=revision))  # nosec: B615


def _shards(root: Path) -> list[Path]:
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors files in {root}")
    return shards


def checkpoint_shapes(root: str | Path) -> dict[str, tuple[int, ...]]:
    """Tensor name -> shape for every tensor in the checkpoint (no data loaded)."""
    st = _safetensors()
    shapes: dict[str, tuple[int, ...]] = {}
    for shard in _shards(Path(root)):
        with st.safe_open(str(shard), "pt") as f:
            for name in f.keys():
                shapes[name] = tuple(f.get_slice(name).get_shape())
    return shapes


def checkpoint_layout(root: str | Path) -> TensorLayout:
    """Detect where ``root`` keeps its token matrices and MLP ``down_proj`` tensors."""
    return TensorLayout.detect(checkpoint_shapes(root))


# transformers families whose RMSNorm computes x * (1 + w) ("zero-centred"); every
# other family computes x * w. Learned weights can't tell the two apart (Qwen3.5
# and Qwen3 both ship final-norm weights averaging ~3.5), so read the config.
ZERO_CENTERED_NORM_FAMILIES = frozenset(
    {
        "gemma",
        "gemma2",
        "gemma3",
        "minimax_m3_vl",
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_next",
        "recurrent_gemma",
        "t5gemma",
        "t5gemma2",
        "vaultgemma",
    }
)


def norm_convention(root: str | Path) -> str:
    """``"zero-centered"`` (x * (1 + w)) or ``"plain"`` (x * w) from ``config.json``."""
    config_path = Path(root) / "config.json"
    if not config_path.is_file():
        return "plain"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    types = {
        config.get("model_type"),
        (config.get("text_config") or {}).get("model_type"),
    }
    families = {t.removesuffix("_text") for t in types if isinstance(t, str)}
    return "zero-centered" if families & ZERO_CENTERED_NORM_FAMILIES else "plain"


def load_tensors(root: str | Path, names: set[str]) -> dict[str, Any]:
    """Load only ``names`` from the checkpoint's shards (float32 is the caller's call)."""
    st = _safetensors()
    out: dict[str, Any] = {}
    for shard in _shards(Path(root)):
        with st.safe_open(str(shard), "pt") as f:
            for name in names & set(f.keys()):
                out[name] = f.get_tensor(name)
    missing = names - set(out)
    if missing:
        raise KeyError(f"not in checkpoint: {sorted(missing)}")
    return out


def _touched_keys(spec: EditSpec, layout: TensorLayout) -> set[str]:
    keys: set[str] = set()
    for e in spec.edits:
        if isinstance(e, TokenRowScale):
            keys |= set(layout.token_keys)
        else:
            keys.add(layout.down_fmt.format(layer=e.layer))
    return keys


def apply_spec_to_checkpoint(
    spec: EditSpec,
    base: str | Path,
    out: str | Path,
    *,
    layout: TensorLayout | None = None,
) -> dict[str, Any]:
    """Copy ``base`` to ``out`` with ``spec`` applied; return the manifest.

    ``out`` must not exist. Shards the spec does not touch are copied byte for
    byte; touched shards are loaded, patched and re-saved with their metadata.
    """
    st = _safetensors()
    src, dst = Path(base), Path(out)
    if dst.exists():
        raise FileExistsError(f"{dst} already exists; refusing to overwrite")
    layout = layout or checkpoint_layout(src)
    touched = _touched_keys(spec, layout)

    shutil.copytree(src, dst, symlinks=False)
    changed = 0
    found: set[str] = set()
    for shard in _shards(dst):
        with st.safe_open(str(shard), "pt") as f:
            names = set(f.keys())
            metadata = f.metadata()
        if not names & touched:
            continue
        found |= names & touched
        tensors = st.torch.load_file(str(shard))
        changed += apply_spec(spec, tensors, layout, strict=False)
        st.torch.save_file(tensors, str(shard), metadata=metadata or {"format": "pt"})
    missing = touched - found
    if missing:
        shutil.rmtree(dst)
        raise KeyError(f"spec touches tensors not in the checkpoint: {sorted(missing)}")

    manifest: dict[str, Any] = {
        "spec": spec.to_dict(),
        "spec_hash": spec.content_hash(),
        "changed_weights": changed,
        "base": str(base),
        "layout": {"token_keys": list(layout.token_keys), "down_fmt": layout.down_fmt},
    }
    (dst / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def verify_checkpoint(
    spec: EditSpec,
    base: str | Path,
    edited: str | Path,
    *,
    layout: TensorLayout | None = None,
    tolerance: float = 0.01,
) -> None:
    """Raise :class:`~.edits.SpecMismatch` unless ``edited`` is exactly ``base`` + ``spec``.

    Every tensor in every shard is checked: untouched tensors must be
    bit-identical, untouched rows/columns of touched tensors must be identical,
    and touched slices must carry the spec's scale. Shard boundaries may differ
    between the two checkpoints.
    """
    st = _safetensors()
    src, dst = Path(base), Path(edited)
    layout = layout or checkpoint_layout(src)
    edited_tensors: dict[str, Any] = {}
    for shard in _shards(dst):
        edited_tensors.update(st.torch.load_file(str(shard)))
    seen: set[str] = set()
    for shard in _shards(src):
        base_tensors = st.torch.load_file(str(shard))
        missing = set(base_tensors) - set(edited_tensors)
        if missing:
            raise SpecMismatch(f"{sorted(missing)[0]} missing from edited checkpoint")
        seen |= set(base_tensors)
        verify_spec(
            spec,
            base_tensors,
            {k: edited_tensors[k] for k in base_tensors},
            layout,
            tolerance=tolerance,
        )
    unaccounted = _touched_keys(spec, layout) - seen
    if unaccounted:
        raise SpecMismatch(f"spec names tensors the base lacks: {sorted(unaccounted)}")
    extra = set(edited_tensors) - seen
    if extra:
        raise SpecMismatch(f"edited checkpoint adds tensors: {sorted(extra)}")
