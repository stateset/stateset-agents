"""Surgical weight-editing engine with non-destructive overlay, undo/redo, and verified export."""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .inspector import WeightInspector, _safe_float


@dataclass
class WeightEditRecord:
    """One atomic, reproducible surgical weight-edit operation."""

    edit_id: str
    op: str  # scale, prune, svd_truncate, clamp, noise, region_edit, blend
    target: (
        str  # exact tensor name, or selector like "role:self_attn", "layer:5", "all"
    )
    params: dict[str, Any]
    affected_tensors: list[str]
    changed_weights: int
    frobenius_delta: float
    created_at: float = field(default_factory=time.time)
    enabled: bool = True
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _apply_op_to_array(
    arr: np.ndarray[Any, Any],
    op: str,
    params: dict[str, Any],
    *,
    secondary_arr: np.ndarray[Any, Any] | None = None,
) -> np.ndarray[Any, Any]:
    """Apply a deterministic surgical transformation to a single float32 weight tensor."""
    out: np.ndarray[Any, Any] = np.array(arr, dtype=np.float32, copy=True)

    if op == "scale":
        factor = float(params.get("factor", 1.0))
        out *= factor
        return out

    if op == "prune":
        mode = str(params.get("mode", "percentile"))
        if mode == "threshold":
            threshold = abs(float(params.get("threshold", 0.01)))
            mask = np.abs(out) < threshold
            out[mask] = 0.0
        else:
            sparsity = max(0.0, min(0.99, float(params.get("sparsity", 0.25))))
            if sparsity > 0.0 and out.size > 0:
                cutoff = float(np.quantile(np.abs(out), sparsity))
                out[np.abs(out) <= cutoff] = 0.0
        return out

    if op == "clamp":
        max_abs = abs(float(params.get("max_abs", 0.1)))
        if params.get("mode") == "percentile":
            pct = max(50.0, min(99.99, float(params.get("percentile", 99.0))))
            max_abs = float(np.percentile(np.abs(out), pct))
        np.clip(out, -max_abs, max_abs, out=out)
        return out

    if op == "noise":
        sigma = abs(float(params.get("sigma", 0.005)))
        seed = int(params.get("seed", 42))
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, sigma, size=out.shape).astype(np.float32)
        out += noise
        return out

    if op == "svd_truncate":
        keep_rank = max(1, int(params.get("rank", 8)))
        if out.ndim == 2:
            r, c = out.shape
            max_r = min(r, c)
            if keep_rank < max_r:
                # For very large matrices keep SVD snappy via randomized/direct SVD
                u, s, vt = np.linalg.svd(out, full_matrices=False)
                k = min(keep_rank, len(s))
                out = ((u[:, :k] * s[:k]) @ vt[:k, :]).astype(np.float32)
        return out

    if op == "region_edit":
        # Direct 3D brush / row-col sub-block edit!
        if out.ndim < 2:
            out_2d = out.reshape(1, -1)
        elif out.ndim > 2:
            out_2d = out.reshape(out.shape[0], -1)
        else:
            out_2d = out
        rows, cols = out_2d.shape
        r0 = max(0, min(rows - 1, int(params.get("row_start", 0))))
        r1 = max(r0 + 1, min(rows, int(params.get("row_end", rows))))
        c0 = max(0, min(cols - 1, int(params.get("col_start", 0))))
        c1 = max(c0 + 1, min(cols, int(params.get("col_end", cols))))
        action = str(params.get("action", "scale"))
        block = out_2d[r0:r1, c0:c1]
        if action == "zero":
            block[...] = 0.0
        elif action == "scale":
            block *= float(params.get("value", 0.5))
        elif action == "add":
            block += float(params.get("value", 0.01))
        elif action == "clamp":
            lim = abs(float(params.get("value", 0.05)))
            np.clip(block, -lim, lim, out=block)
        elif action == "invert":
            block *= -1.0
        reshaped: np.ndarray[Any, Any] = out_2d.reshape(out.shape)
        return reshaped

    if op == "blend":
        alpha = max(-1.0, min(2.0, float(params.get("alpha", 0.5))))
        if secondary_arr is not None and secondary_arr.shape == out.shape:
            out = (
                (1.0 - alpha) * out + alpha * secondary_arr.astype(np.float32)
            ).astype(np.float32)
        return out

    raise ValueError(f"Unsupported weight edit operation: {op}")


class WeightEditEngine:
    """Manages non-destructive surgical weight edits, undo/redo stacks, and verified exports."""

    def __init__(
        self, inspector: WeightInspector, export_root: Path | None = None
    ) -> None:
        self.inspector = inspector
        self.export_root = (
            export_root
            or (inspector.workspace_root / "outputs" / "weight_studio_exports")
        ).resolve()
        # Keyed by (model_id, checkpoint_id)
        self._history: dict[tuple[str, str], list[WeightEditRecord]] = {}
        self._redo_stack: dict[tuple[str, str], list[WeightEditRecord]] = {}
        self._overrides: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    def _session_key(self, model_id: str, checkpoint_id: str | None) -> tuple[str, str]:
        return (model_id, checkpoint_id or "default")

    def get_overrides(
        self, model_id: str, checkpoint_id: str | None = None
    ) -> dict[str, np.ndarray]:
        return self._overrides.get(self._session_key(model_id, checkpoint_id), {})

    def get_edited_tensor(
        self, model_id: str, tensor_name: str, checkpoint_id: str | None = None
    ) -> np.ndarray | None:
        return self.get_overrides(model_id, checkpoint_id).get(tensor_name)

    def get_history(
        self, model_id: str, checkpoint_id: str | None = None
    ) -> dict[str, Any]:
        key = self._session_key(model_id, checkpoint_id)
        history = self._history.get(key, [])
        redo = self._redo_stack.get(key, [])
        overrides = self._overrides.get(key, {})
        return {
            "model_id": model_id,
            "checkpoint_id": checkpoint_id or "default",
            "edits": [e.to_dict() for e in history],
            "can_undo": len(history) > 0,
            "can_redo": len(redo) > 0,
            "modified_tensor_count": len(overrides),
            "modified_tensors": sorted(overrides.keys()),
        }

    def _resolve_targets(
        self, model_id: str, checkpoint_id: str | None, target: str
    ) -> list[str]:
        struct = self.inspector.get_model_structure(model_id, checkpoint_id)
        all_tensors: list[dict[str, Any]] = []
        for layer in struct["layers"]:
            all_tensors.extend(layer["tensors"])
        all_tensors.extend(struct["global_tensors"])

        if target == "all":
            return [t["name"] for t in all_tensors]
        if target.startswith("role:"):
            role = target.split(":", 1)[1]
            return [t["name"] for t in all_tensors if t["role"] == role]
        if target.startswith("layer:"):
            raw = target.split(":", 1)[1]
            if "-" in raw:
                lo_s, hi_s = raw.split("-", 1)
                lo, hi = int(lo_s), int(hi_s)
                return [
                    t["name"]
                    for t in all_tensors
                    if t["layer"] is not None and lo <= int(t["layer"]) <= hi
                ]
            idx = int(raw)
            return [t["name"] for t in all_tensors if t["layer"] == idx]
        if target.startswith("suffix:"):
            suf = target.split(":", 1)[1]
            return [t["name"] for t in all_tensors if suf in t["name"]]
        # Exact tensor match
        matched = [t["name"] for t in all_tensors if t["name"] == target]
        if not matched:
            raise KeyError(f"No tensors match target selector: {target}")
        return matched

    def _rebuild_overrides(self, model_id: str, checkpoint_id: str | None) -> None:
        key = self._session_key(model_id, checkpoint_id)
        history = self._history.get(key, [])
        rebuilt: dict[str, np.ndarray] = {}
        for rec in history:
            if not rec.enabled:
                continue
            secondary_model = rec.params.get("secondary_model_id")
            secondary_cp = rec.params.get("secondary_checkpoint_id")
            for t_name in rec.affected_tensors:
                current = (
                    rebuilt[t_name]
                    if t_name in rebuilt
                    else self.inspector.load_tensor_array(
                        model_id, t_name, checkpoint_id
                    )
                )
                sec_arr: np.ndarray | None = None
                if rec.op == "blend" and secondary_model:
                    try:
                        sec_arr = self.inspector.load_tensor_array(
                            str(secondary_model), t_name, secondary_cp
                        )
                    except (KeyError, FileNotFoundError):
                        sec_arr = None
                rebuilt[t_name] = _apply_op_to_array(
                    current, rec.op, rec.params, secondary_arr=sec_arr
                )
        self._overrides[key] = rebuilt

    def apply_edit(
        self,
        model_id: str,
        op: str,
        target: str,
        params: dict[str, Any],
        *,
        checkpoint_id: str | None = None,
        label: str = "",
    ) -> dict[str, Any]:
        """Apply a surgical weight edit and record it in the undoable edit stack."""
        key = self._session_key(model_id, checkpoint_id)
        targets = self._resolve_targets(model_id, checkpoint_id, target)
        # Safety bound for bulk operations on giant adapters so browser stays instant
        if len(targets) > 128 and op == "svd_truncate":
            targets = targets[:128]

        overrides = self._overrides.setdefault(key, {})
        total_changed = 0
        total_delta_sq = 0.0

        secondary_model = params.get("secondary_model_id")
        secondary_cp = params.get("secondary_checkpoint_id")

        for t_name in targets:
            before = (
                overrides[t_name]
                if t_name in overrides
                else self.inspector.load_tensor_array(model_id, t_name, checkpoint_id)
            )
            sec_arr: np.ndarray | None = None
            if op == "blend" and secondary_model:
                try:
                    sec_arr = self.inspector.load_tensor_array(
                        str(secondary_model), t_name, secondary_cp
                    )
                except (KeyError, FileNotFoundError):
                    sec_arr = None
            after = _apply_op_to_array(before, op, params, secondary_arr=sec_arr)
            diff = after - before
            changed = int(np.sum(np.abs(diff) > 1e-7))
            total_changed += changed
            total_delta_sq += float(np.sum(diff * diff))
            overrides[t_name] = after

        auto_label = (
            label
            or f"{op.upper()} · {target.split('.')[-2] if '.' in target else target}"
        )
        record = WeightEditRecord(
            edit_id=uuid.uuid4().hex[:10],
            op=op,
            target=target,
            params=dict(params),
            affected_tensors=targets,
            changed_weights=total_changed,
            frobenius_delta=round(_safe_float(math.sqrt(total_delta_sq)), 6),
            label=auto_label,
        )
        self._history.setdefault(key, []).append(record)
        self._redo_stack[key] = []
        return {
            "edit": record.to_dict(),
            "history": self.get_history(model_id, checkpoint_id),
        }

    def apply_live_step_deltas(
        self,
        model_id: str,
        deltas: dict[str, np.ndarray],
        *,
        checkpoint_id: str | None = None,
    ) -> None:
        """Apply live training step weight updates directly to the active 3D overlay."""
        key = self._session_key(model_id, checkpoint_id)
        overrides = self._overrides.setdefault(key, {})
        for t_name, updated in deltas.items():
            overrides[t_name] = updated

    def undo(self, model_id: str, checkpoint_id: str | None = None) -> dict[str, Any]:
        key = self._session_key(model_id, checkpoint_id)
        history = self._history.get(key, [])
        if history:
            popped = history.pop()
            self._redo_stack.setdefault(key, []).append(popped)
            self._rebuild_overrides(model_id, checkpoint_id)
        return self.get_history(model_id, checkpoint_id)

    def redo(self, model_id: str, checkpoint_id: str | None = None) -> dict[str, Any]:
        key = self._session_key(model_id, checkpoint_id)
        redo = self._redo_stack.get(key, [])
        if redo:
            restored = redo.pop()
            self._history.setdefault(key, []).append(restored)
            self._rebuild_overrides(model_id, checkpoint_id)
        return self.get_history(model_id, checkpoint_id)

    def toggle_edit(
        self, model_id: str, edit_id: str, checkpoint_id: str | None = None
    ) -> dict[str, Any]:
        key = self._session_key(model_id, checkpoint_id)
        for rec in self._history.get(key, []):
            if rec.edit_id == edit_id:
                rec.enabled = not rec.enabled
                break
        self._rebuild_overrides(model_id, checkpoint_id)
        return self.get_history(model_id, checkpoint_id)

    def reset(self, model_id: str, checkpoint_id: str | None = None) -> dict[str, Any]:
        key = self._session_key(model_id, checkpoint_id)
        self._history[key] = []
        self._redo_stack[key] = []
        self._overrides[key] = {}
        return self.get_history(model_id, checkpoint_id)

    def export_verified_checkpoint(
        self,
        model_id: str,
        export_name: str,
        *,
        checkpoint_id: str | None = None,
    ) -> dict[str, Any]:
        """Save all edited + base tensors to a new verified .safetensors checkpoint with SHA-256 manifest."""
        from safetensors.numpy import load_file, save_file

        safe_name = "".join(
            c if c.isalnum() or c in "-_." else "_" for c in export_name
        ).strip("._")
        if not safe_name:
            safe_name = f"edited-{int(time.time())}"

        out_dir = (self.export_root / safe_name).resolve()
        if not str(out_dir).startswith(str(self.export_root)):
            raise ValueError("Export path escaped export root")
        out_dir.mkdir(parents=True, exist_ok=True)

        entry = self.inspector.get_entry(model_id)
        overrides = self.get_overrides(model_id, checkpoint_id)
        history = self._history.get(self._session_key(model_id, checkpoint_id), [])

        tensors_to_save: dict[str, np.ndarray] = {}
        filename = (
            "adapter_model.safetensors"
            if entry.kind == "local_adapter"
            else "model.safetensors"
        )

        if entry.kind != "preset_twin":
            sf_path = self.inspector._resolve_checkpoint_path(entry, checkpoint_id)
            if sf_path is not None and sf_path.is_file():
                base_tensors = load_file(str(sf_path))
                for k, v in base_tensors.items():
                    tensors_to_save[k] = overrides[k] if k in overrides else v
                # Copy companion config JSON if present
                for cfg_name in ("adapter_config.json", "config.json"):
                    src_cfg = sf_path.parent / cfg_name
                    if src_cfg.is_file():
                        (out_dir / cfg_name).write_text(
                            src_cfg.read_text(encoding="utf-8"), encoding="utf-8"
                        )
        else:
            struct = self.inspector.get_model_structure(model_id, checkpoint_id)
            for layer in struct["layers"]:
                for t in layer["tensors"]:
                    name = t["name"]
                    tensors_to_save[name] = (
                        overrides[name]
                        if name in overrides
                        else self.inspector.load_tensor_array(
                            model_id, name, checkpoint_id
                        )
                    )
            for t in struct["global_tensors"]:
                name = t["name"]
                tensors_to_save[name] = (
                    overrides[name]
                    if name in overrides
                    else self.inspector.load_tensor_array(model_id, name, checkpoint_id)
                )

        out_file = out_dir / filename
        save_file(tensors_to_save, str(out_file))

        # Verify exact reload & SHA-256 checksum
        digest = hashlib.sha256(out_file.read_bytes()).hexdigest()
        reloaded = load_file(str(out_file))
        verified = len(reloaded) == len(tensors_to_save) and all(
            np.allclose(reloaded[k], overrides[k], atol=1e-6)
            for k in overrides
            if k in reloaded
        )

        manifest = {
            "schema_version": 1,
            "source_model_id": model_id,
            "source_checkpoint_id": checkpoint_id or "default",
            "export_file": str(out_file.relative_to(self.inspector.workspace_root)),
            "sha256": digest,
            "tensor_count": len(tensors_to_save),
            "modified_tensors": sorted(overrides.keys()),
            "edits": [e.to_dict() for e in history if e.enabled],
            "reload_verified": verified,
            "exported_at": time.time(),
        }
        manifest_path = out_dir / "edit_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        # Refresh inspector so the newly exported checkpoint appears immediately in the studio!
        self.inspector.refresh_catalog()
        return manifest
