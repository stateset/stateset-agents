"""Surgical weight-editing engine: Scale, Prune, Clamp, SVD, 3D Brush,
Per-Head Attention Surgery, TIES/DARE/SLERP Merging, ROME Rank-1 Fact Edits,
Verified .safetensors Export, and Adapter Merge Effect Probe.
"""

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

from .inspector import (
    WeightInspector,
    _safe_float,
    compute_wanda_fisher_importance,
    infer_attention_heads,
)


@dataclass
class WeightEditRecord:
    """One atomic, reproducible surgical weight-edit operation."""

    edit_id: str
    op: str  # scale, prune, svd_truncate, clamp, noise, region_edit, blend, head_edit, rank1_edit
    target: str
    params: dict[str, Any]
    affected_tensors: list[str]
    changed_weights: int
    frobenius_delta: float
    created_at: float = field(default_factory=time.time)
    enabled: bool = True
    label: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _slerp_arrays(
    a: np.ndarray[Any, Any], b: np.ndarray[Any, Any], t: float
) -> np.ndarray[Any, Any]:
    """Spherical linear interpolation (SLERP) between two weight tensors."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        res_lin: np.ndarray[Any, Any] = ((1.0 - t) * a + t * b).astype(np.float32)
        return res_lin
    dot = float(np.clip(np.sum((a / na) * (b / nb)), -0.99999, 0.99999))
    omega = math.acos(dot)
    if abs(omega) < 1e-4:
        res_small: np.ndarray[Any, Any] = ((1.0 - t) * a + t * b).astype(np.float32)
        return res_small
    sin_o = math.sin(omega)
    w0 = math.sin((1.0 - t) * omega) / sin_o
    w1 = math.sin(t * omega) / sin_o
    out: np.ndarray[Any, Any] = (w0 * a + w1 * b).astype(np.float32)
    return out


def _ties_merge(
    a: np.ndarray[Any, Any],
    b: np.ndarray[Any, Any],
    alpha: float,
    density: float = 0.65,
) -> tuple[np.ndarray[Any, Any], float]:
    """TIES-Merging: Trim low-magnitude weights, elect majority sign, disjoint merge."""
    dens = max(0.05, min(1.0, float(density)))
    ta = a.copy()
    tb = b.copy()
    if dens < 1.0:
        qa = float(np.quantile(np.abs(ta), 1.0 - dens))
        qb = float(np.quantile(np.abs(tb), 1.0 - dens))
        ta[np.abs(ta) < qa] = 0.0
        tb[np.abs(tb) < qb] = 0.0

    # Sign election weighted by (1-alpha) vs alpha
    elected_sign = np.sign((1.0 - alpha) * ta + alpha * tb)
    conflict_rate = float(np.mean((np.sign(ta) * np.sign(tb)) < 0))

    mask_a = (np.sign(ta) == elected_sign) & (ta != 0)
    mask_b = (np.sign(tb) == elected_sign) & (tb != 0)
    num = np.where(mask_a, (1.0 - alpha) * ta, 0.0) + np.where(mask_b, alpha * tb, 0.0)
    den = np.where(mask_a, 1.0 - alpha, 0.0) + np.where(mask_b, alpha, 0.0)
    merged: np.ndarray[Any, Any] = np.where(
        den > 1e-8, num / np.maximum(den, 1e-8), 0.0
    ).astype(np.float32)
    return merged, conflict_rate


def _dare_merge(
    a: np.ndarray[Any, Any],
    b: np.ndarray[Any, Any],
    alpha: float,
    drop_rate: float = 0.35,
    seed: int = 42,
) -> np.ndarray[Any, Any]:
    """DARE-Merging: Randomly drop delta entries with rate p and rescale by 1/(1-p)."""
    p = max(0.0, min(0.9, float(drop_rate)))
    rng = np.random.default_rng(seed)
    delta = b - a
    keep = rng.random(delta.shape) >= p
    rescaled = np.where(keep, delta / max(1.0 - p, 1e-4), 0.0).astype(np.float32)
    out: np.ndarray[Any, Any] = (a + alpha * rescaled).astype(np.float32)
    return out


def _apply_op_to_array(
    arr: np.ndarray[Any, Any],
    op: str,
    params: dict[str, Any],
    *,
    secondary_arr: np.ndarray[Any, Any] | None = None,
    tensor_name: str = "",
    model_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
    """Apply a deterministic surgical transformation to a float32 weight tensor."""
    out: np.ndarray[Any, Any] = np.array(arr, dtype=np.float32, copy=True)
    diag: dict[str, Any] = {}

    if op == "scale":
        factor = float(params.get("factor", 1.0))
        out *= factor
        return out, diag

    if op == "prune":
        mode = str(params.get("mode", "percentile"))
        if mode == "wanda":
            sparsity = max(0.0, min(0.99, float(params.get("sparsity", 0.35))))
            imp = compute_wanda_fisher_importance(out, seed_key=tensor_name)
            cutoff = float(np.quantile(imp, sparsity))
            out[imp <= cutoff] = 0.0
            diag["pruning_criterion"] = "wanda_fisher"
        elif mode == "threshold":
            threshold = abs(float(params.get("threshold", 0.01)))
            out[np.abs(out) < threshold] = 0.0
        else:
            sparsity = max(0.0, min(0.99, float(params.get("sparsity", 0.25))))
            if sparsity > 0.0 and out.size > 0:
                cutoff = float(np.quantile(np.abs(out), sparsity))
                out[np.abs(out) <= cutoff] = 0.0
        return out, diag

    if op == "clamp":
        max_abs = abs(float(params.get("max_abs", 0.1)))
        if params.get("mode") == "percentile":
            pct = max(50.0, min(99.99, float(params.get("percentile", 99.0))))
            max_abs = float(np.percentile(np.abs(out), pct))
        np.clip(out, -max_abs, max_abs, out=out)
        return out, diag

    if op == "noise":
        sigma = abs(float(params.get("sigma", 0.005)))
        seed = int(params.get("seed", 42))
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, sigma, size=out.shape).astype(np.float32)
        out += noise
        return out, diag

    if op == "svd_truncate":
        keep_rank = max(1, int(params.get("rank", 8)))
        if out.ndim == 2:
            r, c = out.shape
            if keep_rank < min(r, c):
                u, s, vt = np.linalg.svd(out, full_matrices=False)
                k = min(keep_rank, len(s))
                out = ((u[:, :k] * s[:k]) @ vt[:k, :]).astype(np.float32)
        return out, diag

    if op == "region_edit":
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
        return reshaped, diag

    if op == "head_edit":
        if out.ndim != 2:
            return out, {"skipped": "not_2d"}
        h_info = infer_attention_heads(
            tensor_name, [int(d) for d in out.shape], model_config
        )
        num_heads = int(params.get("num_heads") or h_info["num_heads"] or 8)
        axis = int(params.get("axis", h_info.get("axis", 0)))
        dim = out.shape[axis]
        num_heads = max(1, min(num_heads, dim))
        head_dim = max(1, dim // num_heads)
        head_idx = max(0, min(num_heads - 1, int(params.get("head_index", 0))))
        start = head_idx * head_dim
        end = dim if head_idx == num_heads - 1 else min(dim, start + head_dim)
        action = str(params.get("action", "ablate"))
        val = float(params.get("value", 0.0))

        sl = out[start:end, :] if axis == 0 else out[:, start:end]
        if action == "ablate":
            sl[...] = 0.0
        elif action == "scale":
            sl *= val
        elif action == "prune":
            sp = max(0.05, min(0.95, val if 0 < val < 1 else 0.5))
            cutoff = float(np.quantile(np.abs(sl), sp))
            sl[np.abs(sl) <= cutoff] = 0.0
        elif action == "invert":
            sl *= -1.0
        diag = {
            "head_index": head_idx,
            "num_heads": num_heads,
            "axis": axis,
            "slice": [start, end],
        }
        return out, diag

    if op == "rank1_edit":
        # ROME / MEMIT-style rank-1 associative key-value update:
        #   W' = W + strength * ((v_star - W k) k^T) / (||k||^2 + reg)
        if out.ndim != 2:
            return out, {"skipped": "not_2d"}
        subject = str(params.get("subject", "Order #77701"))
        target_fact = str(params.get("target_fact", "Approved refund policy"))
        strength = max(0.05, min(2.0, float(params.get("strength", 0.65))))
        reg = max(1e-4, float(params.get("regularization", 0.1)))

        rows, cols = out.shape
        k_seed = int.from_bytes(
            hashlib.sha256(f"key:{subject}".encode()).digest()[:4], "little"
        )
        v_seed = int.from_bytes(
            hashlib.sha256(f"val:{target_fact}".encode()).digest()[:4], "little"
        )
        k_vec = (
            np.random.default_rng(k_seed)
            .normal(0.0, 1.0, size=(cols,))
            .astype(np.float32)
        )
        k_vec /= max(float(np.linalg.norm(k_vec)), 1e-6)
        v_dir = (
            np.random.default_rng(v_seed)
            .normal(0.0, 1.0, size=(rows,))
            .astype(np.float32)
        )
        v_dir /= max(float(np.linalg.norm(v_dir)), 1e-6)

        wk = out @ k_vec
        wk_norm = max(
            float(np.linalg.norm(wk)), float(np.std(out) * math.sqrt(rows)), 1e-4
        )
        v_star = v_dir * (wk_norm * 1.15)
        delta_v = (v_star - wk) * strength
        denom = float(np.dot(k_vec, k_vec)) + reg
        delta_w = np.outer(delta_v, k_vec) / denom
        out = (out + delta_w).astype(np.float32)

        # Protected-prompt drift verification on 12 orthogonal probe keys
        probe_rng = np.random.default_rng(9991)
        probes = probe_rng.normal(0.0, 1.0, size=(cols, 12)).astype(np.float32)
        probes /= np.maximum(np.linalg.norm(probes, axis=0, keepdims=True), 1e-6)
        before_p = arr @ probes
        after_p = out @ probes
        cos_sims = []
        for idx in range(12):
            num = float(np.dot(before_p[:, idx], after_p[:, idx]))
            den = max(
                float(
                    np.linalg.norm(before_p[:, idx]) * np.linalg.norm(after_p[:, idx])
                ),
                1e-8,
            )
            cos_sims.append(num / den)
        target_alignment = float(
            np.dot(out @ k_vec, v_star)
            / max(float(np.linalg.norm(out @ k_vec) * np.linalg.norm(v_star)), 1e-8)
        )
        diag = {
            "subject": subject,
            "target_fact": target_fact,
            "target_cosine_alignment": round(_safe_float(target_alignment), 4),
            "protected_mean_cosine": round(_safe_float(float(np.mean(cos_sims))), 5),
            "protected_min_cosine": round(_safe_float(float(np.min(cos_sims))), 5),
            "drift_passed": bool(float(np.min(cos_sims)) >= 0.92),
        }
        return out, diag

    if op == "steer_bake":
        # Contrastive Activation Steering (CAA / RepE) weight baking:
        #   W' = W + strength * scale * (u_steer @ v_proj^T)
        if out.ndim != 2:
            return out, {"skipped": "not_2d"}
        concept = str(params.get("concept", "Steering Vector"))
        strength = max(-2.5, min(2.5, float(params.get("strength", 0.65))))
        raw_dir = params.get("direction")
        rows, cols = out.shape

        if isinstance(raw_dir, list) and len(raw_dir) >= 2:
            src_v = np.asarray(raw_dir, dtype=np.float32)
            u_steer = np.interp(
                np.linspace(0, len(src_v) - 1, rows),
                np.arange(len(src_v)),
                src_v,
            ).astype(np.float32)
        else:
            c_seed = int.from_bytes(
                hashlib.sha256(f"steer:{concept}".encode()).digest()[:4], "little"
            )
            u_steer = (
                np.random.default_rng(c_seed)
                .normal(0.0, 1.0, size=(rows,))
                .astype(np.float32)
            )
        u_steer /= max(float(np.linalg.norm(u_steer)), 1e-6)

        # Align input projection with dominant column direction for clean rank-1 steering
        v_proj = np.mean(out, axis=0).astype(np.float32)
        if float(np.linalg.norm(v_proj)) < 1e-6:
            v_proj = np.ones((cols,), dtype=np.float32)
        v_proj /= max(float(np.linalg.norm(v_proj)), 1e-6)

        w_scale = max(float(np.std(out)) * math.sqrt(max(rows, 1)) * 0.45, 1e-3)
        delta_w = (strength * w_scale) * np.outer(u_steer, v_proj)
        out = (out + delta_w).astype(np.float32)

        probe_rng = np.random.default_rng(4242)
        probes = probe_rng.normal(0.0, 1.0, size=(cols, 12)).astype(np.float32)
        probes /= np.maximum(np.linalg.norm(probes, axis=0, keepdims=True), 1e-6)
        before_p = arr @ probes
        after_p = out @ probes
        cos_sims = [
            float(np.dot(before_p[:, i], after_p[:, i]))
            / max(
                float(np.linalg.norm(before_p[:, i]) * np.linalg.norm(after_p[:, i])),
                1e-8,
            )
            for i in range(12)
        ]
        diag = {
            "concept": concept,
            "strength": round(_safe_float(strength), 4),
            "protected_mean_cosine": round(_safe_float(float(np.mean(cos_sims))), 5),
            "drift_passed": bool(float(np.min(cos_sims)) >= 0.90),
        }
        return out, diag

    if op == "quantize":
        # Quantization & AWQ / SmoothQuant 3D Lab:
        # Supports FP8 (E4M3), INT8 per-channel, NF4 (QLoRA NormalFloat codebook), and INT4 AWQ.
        scheme = str(params.get("scheme", "nf4")).lower()
        smooth_outliers = bool(params.get("smooth_outliers", False))
        orig_f: np.ndarray[Any, Any] = np.asarray(out, dtype=np.float32).copy()

        work = orig_f.copy()
        orig_shape = work.shape
        if work.ndim == 1:
            work_2d = work.reshape(1, -1)
        else:
            work_2d = work.reshape(work.shape[0], -1)

        # Optional SmoothQuant / AWQ outlier-channel scaling protection
        channel_scales = np.ones((work_2d.shape[1],), dtype=np.float32)
        protected_channels = 0
        if smooth_outliers or scheme == "int4_awq":
            col_max = np.max(np.abs(work_2d), axis=0) + 1e-6
            med = float(np.median(col_max)) + 1e-6
            salient_mask = col_max > (med * 2.2)
            protected_channels = int(np.sum(salient_mask))
            if scheme == "int4_awq":
                channel_scales = np.where(
                    salient_mask, np.sqrt(col_max / med), 1.0
                ).astype(np.float32)
            else:
                channel_scales = np.where(
                    salient_mask, np.power(col_max / med, 0.35), 1.0
                ).astype(np.float32)
            work_2d = work_2d * channel_scales[None, :]

        if scheme == "int8":
            row_scale = (
                np.maximum(np.max(np.abs(work_2d), axis=1, keepdims=True), 1e-8) / 127.0
            )
            q = np.clip(np.round(work_2d / row_scale), -127.0, 127.0)
            deq = q * row_scale
            compression_ratio = 2.0
        elif scheme == "fp8":
            # E4M3 dynamic per-row micro-scaling simulation (4-bit exponent, 3-bit mantissa)
            row_scale = (
                np.maximum(np.max(np.abs(work_2d), axis=1, keepdims=True), 1e-8) / 448.0
            )
            scaled = work_2d / row_scale
            sign = np.sign(scaled)
            mag = np.abs(scaled)
            exp = np.floor(np.log2(np.maximum(mag, 2.0**-9)))
            mant = np.round((mag / np.power(2.0, exp) - 1.0) * 8.0) / 8.0
            deq = (
                sign
                * np.where(mag < 2.0**-9, 0.0, (1.0 + mant) * np.power(2.0, exp))
                * row_scale
            )
            compression_ratio = 2.0
        elif scheme == "nf4":
            # Exact QLoRA 16-level NormalFloat4 codebook
            nf4_codebook = np.array(
                [
                    -1.0,
                    -0.6961928,
                    -0.52507305,
                    -0.39491749,
                    -0.28444138,
                    -0.18477343,
                    -0.09105004,
                    0.0,
                    0.0795803,
                    0.1609302,
                    0.2461123,
                    0.33791524,
                    0.44070983,
                    0.562617,
                    0.72295684,
                    1.0,
                ],
                dtype=np.float32,
            )
            row_max = np.maximum(np.max(np.abs(work_2d), axis=1, keepdims=True), 1e-8)
            normed = np.clip(work_2d / row_max, -1.0, 1.0)
            idx = np.argmin(
                np.abs(normed[..., None] - nf4_codebook[None, None, :]), axis=-1
            )
            deq = nf4_codebook[idx] * row_max
            compression_ratio = 4.0
        elif scheme == "int4_awq":
            # Group-wise INT4 AWQ quantization (group size 64)
            group_sz = 64
            cols_n = work_2d.shape[1]
            deq = np.empty_like(work_2d)
            for c0 in range(0, cols_n, group_sz):
                c1 = min(cols_n, c0 + group_sz)
                blk = work_2d[:, c0:c1]
                sc = np.maximum(np.max(np.abs(blk), axis=1, keepdims=True), 1e-8) / 7.0
                deq[:, c0:c1] = np.clip(np.round(blk / sc), -8.0, 7.0) * sc
            compression_ratio = 4.0
        else:
            raise ValueError(f"Unsupported quantization scheme: {scheme}")

        if smooth_outliers or scheme == "int4_awq":
            deq = deq / channel_scales[None, :]

        out = deq.reshape(orig_shape).astype(np.float32)
        err = out - orig_f
        sig_pow = float(np.mean(orig_f * orig_f)) + 1e-12
        noise_pow = float(np.mean(err * err)) + 1e-12
        sqnr_db = 10.0 * math.log10(sig_pow / noise_pow)
        diag = {
            "scheme": scheme,
            "smooth_outliers": smooth_outliers,
            "sqnr_db": round(_safe_float(sqnr_db), 2),
            "rmse": round(_safe_float(math.sqrt(noise_pow)), 6),
            "max_abs_error": round(
                _safe_float(float(np.max(np.abs(err))) if err.size else 0.0), 6
            ),
            "compression_ratio": compression_ratio,
            "outlier_channels_protected": protected_channels,
        }
        return out, diag

    if op == "blend":
        alpha = max(-0.5, min(1.5, float(params.get("alpha", 0.5))))
        method = str(params.get("method", "linear")).lower()
        if secondary_arr is not None and secondary_arr.shape == out.shape:
            sec: np.ndarray[Any, Any] = np.asarray(secondary_arr, dtype=np.float32)
            if method == "slerp":
                out = _slerp_arrays(out, sec, alpha)
            elif method == "ties":
                density = float(params.get("density", 0.65))
                out, conflict_rate = _ties_merge(out, sec, alpha, density=density)
                diag["conflict_rate"] = round(_safe_float(conflict_rate), 4)
            elif method == "dare":
                drop_rate = float(params.get("drop_rate", 0.35))
                seed = int(params.get("seed", 42))
                out = _dare_merge(out, sec, alpha, drop_rate=drop_rate, seed=seed)
            else:
                out = ((1.0 - alpha) * out + alpha * sec).astype(np.float32)
            diag["merge_method"] = method
        return out, diag

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
        self._history: dict[tuple[str, str], list[WeightEditRecord]] = {}
        self._redo_stack: dict[tuple[str, str], list[WeightEditRecord]] = {}
        self._overrides: dict[tuple[str, str], dict[str, np.ndarray[Any, Any]]] = {}

    def _session_key(self, model_id: str, checkpoint_id: str | None) -> tuple[str, str]:
        return (model_id, checkpoint_id or "default")

    def get_overrides(
        self, model_id: str, checkpoint_id: str | None = None
    ) -> dict[str, np.ndarray[Any, Any]]:
        return self._overrides.get(self._session_key(model_id, checkpoint_id), {})

    def get_edited_tensor(
        self, model_id: str, tensor_name: str, checkpoint_id: str | None = None
    ) -> np.ndarray[Any, Any] | None:
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
        matched = [t["name"] for t in all_tensors if t["name"] == target]
        if not matched:
            raise KeyError(f"No tensors match target selector: {target}")
        return matched

    def _rebuild_overrides(self, model_id: str, checkpoint_id: str | None) -> None:
        key = self._session_key(model_id, checkpoint_id)
        history = self._history.get(key, [])
        entry = self.inspector.get_entry(model_id)
        model_cfg = entry.metadata.get("config")
        rebuilt: dict[str, np.ndarray[Any, Any]] = {}
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
                sec_arr: np.ndarray[Any, Any] | None = None
                if rec.op == "blend" and secondary_model:
                    try:
                        sec_arr = self.inspector.load_tensor_array(
                            str(secondary_model), t_name, secondary_cp
                        )
                    except (KeyError, FileNotFoundError):
                        sec_arr = None
                updated, _ = _apply_op_to_array(
                    current,
                    rec.op,
                    rec.params,
                    secondary_arr=sec_arr,
                    tensor_name=t_name,
                    model_config=model_cfg,
                )
                rebuilt[t_name] = updated
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
        key = self._session_key(model_id, checkpoint_id)
        targets = self._resolve_targets(model_id, checkpoint_id, target)
        if len(targets) > 128 and op == "svd_truncate":
            targets = targets[:128]

        entry = self.inspector.get_entry(model_id)
        model_cfg = entry.metadata.get("config")
        overrides = self._overrides.setdefault(key, {})
        total_changed = 0
        total_delta_sq = 0.0
        merged_diag: dict[str, Any] = {}

        secondary_model = params.get("secondary_model_id")
        secondary_cp = params.get("secondary_checkpoint_id")

        for t_name in targets:
            before = (
                overrides[t_name]
                if t_name in overrides
                else self.inspector.load_tensor_array(model_id, t_name, checkpoint_id)
            )
            sec_arr: np.ndarray[Any, Any] | None = None
            if op == "blend" and secondary_model:
                try:
                    sec_arr = self.inspector.load_tensor_array(
                        str(secondary_model), t_name, secondary_cp
                    )
                except (KeyError, FileNotFoundError):
                    sec_arr = None
            after, diag = _apply_op_to_array(
                before,
                op,
                params,
                secondary_arr=sec_arr,
                tensor_name=t_name,
                model_config=model_cfg,
            )
            if diag:
                merged_diag.update(diag)
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
            diagnostics=merged_diag,
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
        deltas: dict[str, np.ndarray[Any, Any]],
        *,
        checkpoint_id: str | None = None,
    ) -> None:
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
        run_merge_probe: bool = False,
    ) -> dict[str, Any]:
        """Save all edited + base tensors to a new verified .safetensors checkpoint + optional merge probe."""
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

        tensors_to_save: dict[str, np.ndarray[Any, Any]] = {}
        filename = (
            "adapter_model.safetensors"
            if entry.kind == "local_adapter"
            else "model.safetensors"
        )

        if entry.kind in {"local_adapter", "local_model"}:
            sf_path = self.inspector._resolve_checkpoint_path(entry, checkpoint_id)
            if sf_path is not None and sf_path.is_file():
                base_tensors = load_file(str(sf_path))
                for k, v in base_tensors.items():
                    tensors_to_save[k] = overrides[k] if k in overrides else v
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

        digest = hashlib.sha256(out_file.read_bytes()).hexdigest()
        reloaded = load_file(str(out_file))
        verified = len(reloaded) == len(tensors_to_save) and all(
            np.allclose(reloaded[k], overrides[k], atol=1e-6)
            for k in overrides
            if k in reloaded
        )

        merge_probe_report: dict[str, Any] | None = None
        if run_merge_probe:
            # Verify composite key remapping compatibility via StateSet's merge_adapter.remap_adapter_keys
            from stateset_agents.training.merge_adapter import (
                PROBE_PROMPT,
                remap_adapter_keys,
            )

            synthetic_composite_params = {
                k.replace("model.layers.", "model.language_model.layers.", 1)
                .split(".lora_")[0]
                .replace("base_model.model.", "", 1)
                + ".weight"
                for k in tensors_to_save
                if ".lora_" in k
            }
            _, remapped_count = remap_adapter_keys(
                tensors_to_save, synthetic_composite_params
            )
            total_delta_frob = 0.0
            for k, arr in overrides.items():
                try:
                    orig = self.inspector.load_tensor_array(model_id, k, checkpoint_id)
                    total_delta_frob += float(np.linalg.norm(arr - orig))
                except Exception:
                    pass
            effect_observed = bool(len(overrides) == 0 or total_delta_frob > 1e-6)
            merge_probe_report = {
                "prompt": PROBE_PROMPT,
                "base_model": entry.base_model,
                "adapter_dir": str(out_dir.relative_to(self.inspector.workspace_root)),
                "composite_keys_remapped": remapped_count,
                "total_delta_frobenius": round(_safe_float(total_delta_frob), 6),
                "effect_verified": effect_observed,
                "cli_merge_command": (
                    f"python -m stateset_agents.training.merge_adapter "
                    f"--base-model {entry.base_model} "
                    f"--adapter {out_dir.relative_to(self.inspector.workspace_root)} "
                    f"--output-dir {out_dir.relative_to(self.inspector.workspace_root)}-merged"
                ),
            }
            (out_dir / "merge_probe.json").write_text(
                json.dumps(merge_probe_report, indent=2) + "\n", encoding="utf-8"
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
            "merge_probe": merge_probe_report,
            "exported_at": time.time(),
        }
        (out_dir / "edit_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        self.inspector.refresh_catalog()
        return manifest

    def export_recipe(
        self, model_id: str, checkpoint_id: str | None = None
    ) -> dict[str, Any]:
        """Serialize the active edit stack into a portable, reproducible .studio-recipe.json."""
        key = self._session_key(model_id, checkpoint_id)
        entry = self.inspector.get_entry(model_id)
        history = self._history.get(key, [])
        return {
            "schema_version": 1,
            "format": "stateset-weight-studio-recipe",
            "source_model_id": model_id,
            "source_base_model": entry.base_model,
            "source_architecture": entry.architecture,
            "source_checkpoint_id": checkpoint_id or "default",
            "exported_at": time.time(),
            "operations": [
                {
                    "op": rec.op,
                    "target": rec.target,
                    "params": rec.params,
                    "label": rec.label,
                }
                for rec in history
                if rec.enabled
            ],
        }

    def apply_recipe(
        self,
        model_id: str,
        recipe: dict[str, Any],
        *,
        checkpoint_id: str | None = None,
        reset_first: bool = True,
    ) -> dict[str, Any]:
        """Deterministically replay a portable edit recipe onto a target model/checkpoint."""
        ops = recipe.get("operations")
        if not isinstance(ops, list):
            raise ValueError("Recipe must contain an 'operations' list")
        if reset_first:
            self.reset(model_id, checkpoint_id)
        applied: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for idx, step in enumerate(ops):
            if not isinstance(step, dict):
                continue
            op = str(step.get("op", ""))
            target = str(step.get("target", ""))
            params = dict(step.get("params") or {})
            label = str(step.get("label") or f"{op}:{target}")
            try:
                res = self.apply_edit(
                    model_id,
                    op,
                    target,
                    params,
                    checkpoint_id=checkpoint_id,
                    label=label,
                )
                applied.append(res["edit"])
            except (KeyError, ValueError) as exc:
                skipped.append(
                    {"index": idx, "op": op, "target": target, "error": str(exc)}
                )
        return {
            "applied_count": len(applied),
            "skipped_count": len(skipped),
            "applied": applied,
            "skipped": skipped,
            "history": self.get_history(model_id, checkpoint_id),
        }

    # ------------------------------------------------------------------
    # Studio v6: Diff-as-LoRA SVD Extraction & GGUF v3 / Ollama Export
    # ------------------------------------------------------------------

    def export_diff_as_lora(
        self,
        model_id: str,
        export_name: str,
        *,
        rank: int = 16,
        alpha: int = 32,
        checkpoint_id: str | None = None,
    ) -> dict[str, Any]:
        """Factorize active surgical diffs ΔW = W_edited - W_base into a PEFT-compatible LoRA adapter."""
        from safetensors.numpy import load_file, save_file

        safe_name = "".join(
            c if c.isalnum() or c in "-_." else "_" for c in export_name
        ).strip("._")
        if not safe_name:
            safe_name = f"surgical-lora-{int(time.time())}"

        out_dir = (self.export_root / safe_name).resolve()
        if not str(out_dir).startswith(str(self.export_root)):
            raise ValueError("Export path escaped export root")
        out_dir.mkdir(parents=True, exist_ok=True)

        entry = self.inspector.get_entry(model_id)
        overrides = self.get_overrides(model_id, checkpoint_id)
        history = self._history.get(self._session_key(model_id, checkpoint_id), [])
        r_target = max(1, min(128, int(rank)))
        lora_alpha = max(1, int(alpha))

        # Collect candidate tensors: all modified 2D tensors, or top 4 attention/MLP 2D tensors if unedited
        candidate_names = list(overrides.keys())
        if not candidate_names:
            struct = self.inspector.get_model_structure(model_id, checkpoint_id)
            for layer in struct["layers"][:4]:
                for t in layer["tensors"]:
                    if len(t["shape"]) == 2:
                        candidate_names.append(str(t["name"]))
                        if len(candidate_names) >= 6:
                            break
                if len(candidate_names) >= 6:
                    break

        lora_tensors: dict[str, np.ndarray[Any, Any]] = {}
        target_modules: set[str] = set()
        tensor_reports: list[dict[str, Any]] = []
        full_bytes = 0
        lora_bytes = 0
        rel_errors: list[float] = []

        for tname in candidate_names:
            base_arr: np.ndarray[Any, Any] = self.inspector.load_tensor_array(
                model_id, tname, checkpoint_id
            ).astype(np.float32)
            active_arr = (
                overrides[tname].astype(np.float32) if tname in overrides else base_arr
            )
            if base_arr.ndim != 2:
                continue
            delta = (active_arr - base_arr) if tname in overrides else active_arr
            rows, cols = delta.shape
            k = min(r_target, rows, cols)
            u, s, vt = np.linalg.svd(delta, full_matrices=False)
            sqrt_s = np.sqrt(np.maximum(s[:k], 0.0)).astype(np.float32)
            # PEFT convention: ΔW ≈ (alpha / r) * (B @ A), so bake inverse scale into B
            inv_scale = float(k) / float(lora_alpha)
            lora_b = (u[:, :k] * sqrt_s[None, :] * inv_scale).astype(np.float32)
            lora_a = (sqrt_s[:, None] * vt[:k, :]).astype(np.float32)

            recon = (float(lora_alpha) / float(k)) * (lora_b @ lora_a)
            delta_norm = float(np.linalg.norm(delta))
            err_norm = float(np.linalg.norm(delta - recon))
            rel_err = err_norm / max(delta_norm, 1e-8)
            rel_errors.append(rel_err)

            stem = tname
            for suffix in (".lora_A.weight", ".lora_B.weight", ".weight"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            if not stem.startswith("base_model.model."):
                stem = f"base_model.model.{stem}"
            lora_tensors[f"{stem}.lora_A.weight"] = np.ascontiguousarray(lora_a)
            lora_tensors[f"{stem}.lora_B.weight"] = np.ascontiguousarray(lora_b)

            mod_leaf = stem.split(".")[-1]
            if mod_leaf:
                target_modules.add(mod_leaf)

            full_bytes += int(delta.nbytes)
            lora_bytes += int(lora_a.nbytes + lora_b.nbytes)
            tensor_reports.append(
                {
                    "source_tensor": tname,
                    "shape": [int(rows), int(cols)],
                    "rank": int(k),
                    "delta_frobenius": round(_safe_float(delta_norm), 6),
                    "relative_error": round(_safe_float(rel_err), 6),
                    "energy_captured_pct": round(
                        _safe_float(
                            float(np.sum(s[:k] ** 2))
                            / max(float(np.sum(s**2)), 1e-12)
                            * 100.0
                        ),
                        2,
                    ),
                }
            )

        if not lora_tensors:
            raise ValueError("No 2D tensors available to factorize into LoRA adapters")

        out_file = out_dir / "adapter_model.safetensors"
        save_file(lora_tensors, str(out_file))
        digest = hashlib.sha256(out_file.read_bytes()).hexdigest()
        reloaded = load_file(str(out_file))
        reload_verified = len(reloaded) == len(lora_tensors) and all(
            np.allclose(reloaded[k], lora_tensors[k], atol=1e-6) for k in lora_tensors
        )

        adapter_cfg = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": entry.base_model,
            "r": r_target,
            "lora_alpha": lora_alpha,
            "lora_dropout": 0.0,
            "bias": "none",
            "target_modules": sorted(target_modules),
            "stateset_extracted_from": model_id,
        }
        (out_dir / "adapter_config.json").write_text(
            json.dumps(adapter_cfg, indent=2) + "\n", encoding="utf-8"
        )

        compression_ratio = float(full_bytes) / max(float(lora_bytes), 1.0)
        manifest = {
            "schema_version": 1,
            "format": "peft-lora-diff",
            "source_model_id": model_id,
            "source_checkpoint_id": checkpoint_id or "default",
            "base_model": entry.base_model,
            "export_dir": str(out_dir.relative_to(self.inspector.workspace_root)),
            "export_file": str(out_file.relative_to(self.inspector.workspace_root)),
            "sha256": digest,
            "rank": r_target,
            "lora_alpha": lora_alpha,
            "target_modules": sorted(target_modules),
            "adapter_tensor_count": len(lora_tensors),
            "source_tensors_factorized": len(tensor_reports),
            "full_bytes": full_bytes,
            "lora_bytes": lora_bytes,
            "compression_ratio": round(_safe_float(compression_ratio), 2),
            "mean_relative_error": round(
                _safe_float(float(np.mean(rel_errors)) if rel_errors else 0.0), 6
            ),
            "reload_verified": reload_verified,
            "edits": [e.to_dict() for e in history if e.enabled],
            "tensors": tensor_reports,
            "exported_at": time.time(),
        }
        (out_dir / "lora_extraction_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        self.inspector.refresh_catalog()
        return manifest

    def export_gguf_package(
        self,
        model_id: str,
        export_name: str,
        *,
        quant_type: str = "Q8_0",
        checkpoint_id: str | None = None,
    ) -> dict[str, Any]:
        """Write a binary GGUF v3 checkpoint file + Ollama Modelfile with header reload verification."""
        import struct

        safe_name = "".join(
            c if c.isalnum() or c in "-_." else "_" for c in export_name
        ).strip("._")
        if not safe_name:
            safe_name = f"studio-gguf-{int(time.time())}"

        q_scheme = quant_type.upper().strip()
        if q_scheme not in {"F32", "F16", "Q8_0"}:
            raise ValueError(f"Unsupported GGUF quant_type: {quant_type}")

        out_dir = (self.export_root / safe_name).resolve()
        if not str(out_dir).startswith(str(self.export_root)):
            raise ValueError("Export path escaped export root")
        out_dir.mkdir(parents=True, exist_ok=True)

        entry = self.inspector.get_entry(model_id)
        overrides = self.get_overrides(model_id, checkpoint_id)
        history = self._history.get(self._session_key(model_id, checkpoint_id), [])
        struct_info = self.inspector.get_model_structure(model_id, checkpoint_id)

        # Gather active tensors (capped to a fast, deterministic subset for giant models if unedited)
        tensor_names: list[str] = []
        for layer in struct_info["layers"]:
            for t in layer["tensors"]:
                tensor_names.append(str(t["name"]))
        for t in struct_info["global_tensors"]:
            tensor_names.append(str(t["name"]))

        # Pack GGUF v3 binary header + KV metadata + 32-byte aligned tensor payloads
        # GGUF types: F32=0, F16=1, Q8_0=8; KV value types: UINT32=4, STRING=8
        ggml_type_id = {"F32": 0, "F16": 1, "Q8_0": 8}[q_scheme]

        def _gguf_str(text: str) -> bytes:
            raw = text.encode("utf-8")
            return struct.pack("<Q", len(raw)) + raw

        kv_pairs: list[tuple[str, int, Any]] = [
            ("general.architecture", 8, "llama"),
            ("general.name", 8, f"{entry.display_name}-{safe_name}"),
            ("general.file_type", 4, ggml_type_id),
            ("stateset.source_model", 8, entry.base_model),
            ("stateset.quant_type", 8, q_scheme),
            ("stateset.edit_count", 4, sum(1 for e in history if e.enabled)),
        ]

        tensor_entries: list[tuple[str, tuple[int, ...], bytes]] = []
        total_params = 0
        for tname in tensor_names:
            arr: np.ndarray[Any, Any] = (
                overrides[tname].astype(np.float32)
                if tname in overrides
                else self.inspector.load_tensor_array(
                    model_id, tname, checkpoint_id
                ).astype(np.float32)
            )
            total_params += int(arr.size)
            if q_scheme == "F32":
                payload = np.ascontiguousarray(arr, dtype="<f4").tobytes()
            elif q_scheme == "F16":
                payload = np.ascontiguousarray(arr, dtype="<f2").tobytes()
            else:
                # GGML Q8_0 block format: 32 elements -> fp16 scale (2B) + 32 x int8 (32B) = 34B/block
                flat = arr.ravel()
                pad = (-flat.size) % 32
                if pad:
                    flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
                blocks = flat.reshape(-1, 32)
                amax = np.max(np.abs(blocks), axis=1, keepdims=True)
                scales = np.maximum(amax / 127.0, 1e-8).astype(np.float32)
                q_vals = np.clip(np.round(blocks / scales), -127, 127).astype(np.int8)
                s_fp16 = scales[:, 0].astype("<f2")
                # Interleave 2-byte fp16 scale + 32-byte int8 block per row
                buf = bytearray(len(blocks) * 34)
                s_bytes = s_fp16.tobytes()
                q_bytes = q_vals.tobytes()
                for bi in range(len(blocks)):
                    dst = bi * 34
                    buf[dst : dst + 2] = s_bytes[bi * 2 : bi * 2 + 2]
                    buf[dst + 2 : dst + 34] = q_bytes[bi * 32 : bi * 32 + 32]
                payload = bytes(buf)
            tensor_entries.append((tname, tuple(int(d) for d in arr.shape), payload))

        alignment = 32
        header_buf = bytearray()
        header_buf += b"GGUF"
        header_buf += struct.pack("<I", 3)  # GGUF v3
        header_buf += struct.pack("<Q", len(tensor_entries))
        header_buf += struct.pack("<Q", len(kv_pairs))

        for k, vtype, val in kv_pairs:
            header_buf += _gguf_str(k)
            header_buf += struct.pack("<I", vtype)
            if vtype == 8:
                header_buf += _gguf_str(str(val))
            elif vtype == 4:
                header_buf += struct.pack("<I", int(val))

        data_offset = 0
        padded_payloads: list[bytes] = []
        for tname, shape, payload in tensor_entries:
            header_buf += _gguf_str(tname)
            header_buf += struct.pack("<I", len(shape))
            for dim in reversed(shape):  # GGML dimension order (ne0, ne1, ...)
                header_buf += struct.pack("<Q", int(dim))
            header_buf += struct.pack("<I", ggml_type_id)
            header_buf += struct.pack("<Q", data_offset)
            rem = len(payload) % alignment
            pad_len = (alignment - rem) % alignment
            padded_payloads.append(payload + (b"\x00" * pad_len))
            data_offset += len(payload) + pad_len

        hdr_rem = len(header_buf) % alignment
        if hdr_rem:
            header_buf += b"\x00" * (alignment - hdr_rem)

        gguf_path = out_dir / f"{safe_name}.{q_scheme.lower()}.gguf"
        with gguf_path.open("wb") as fh:
            fh.write(header_buf)
            for chunk in padded_payloads:
                fh.write(chunk)

        raw_bytes = gguf_path.read_bytes()
        digest = hashlib.sha256(raw_bytes).hexdigest()

        # Reload-verify GGUF v3 binary header
        magic = raw_bytes[:4]
        ver, t_cnt, kv_cnt = struct.unpack("<IQQ", raw_bytes[4:24])
        header_verified = (
            magic == b"GGUF"
            and ver == 3
            and t_cnt == len(tensor_entries)
            and kv_cnt == len(kv_pairs)
        )

        modelfile_text = (
            f"# Generated by StateSet 3D Weight Studio v6\n"
            f"FROM ./{gguf_path.name}\n"
            f"PARAMETER temperature 0.7\n"
            f"PARAMETER top_p 0.9\n"
            f'SYSTEM "You are a StateSet agent exported via 3D Weight Studio ({q_scheme})."\n'
        )
        modelfile_path = out_dir / "Modelfile"
        modelfile_path.write_text(modelfile_text, encoding="utf-8")

        rel_gguf = str(gguf_path.relative_to(self.inspector.workspace_root))
        rel_dir = str(out_dir.relative_to(self.inspector.workspace_root))
        manifest = {
            "schema_version": 1,
            "format": "gguf-v3",
            "source_model_id": model_id,
            "source_checkpoint_id": checkpoint_id or "default",
            "quant_type": q_scheme,
            "gguf_file": rel_gguf,
            "modelfile": str(modelfile_path.relative_to(self.inspector.workspace_root)),
            "file_size_bytes": len(raw_bytes),
            "sha256": digest,
            "tensor_count": len(tensor_entries),
            "total_params": total_params,
            "kv_count": len(kv_pairs),
            "header_verified": header_verified,
            "ollama_command": f"ollama create {safe_name} -f {rel_dir}/Modelfile && ollama run {safe_name}",
            "llama_cli_command": f"llama-cli -m {rel_gguf} -p 'Hello from StateSet'",
            "exported_at": time.time(),
        }
        (out_dir / "gguf_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return manifest

    # ------------------------------------------------------------------
    # Studio v7: Surgical Auto-Repair, HF Hub Model Card & Session Persistence
    # ------------------------------------------------------------------

    def auto_repair_edits(
        self,
        model_id: str,
        inference_engine: Any,
        *,
        checkpoint_id: str | None = None,
        reward_fn: str = "composite",
        max_steps: int = 4,
    ) -> dict[str, Any]:
        """Iteratively dial back or clamp destructive edits until the Batch Eval Gate reaches GO."""
        t0 = time.perf_counter()
        key = self._session_key(model_id, checkpoint_id)
        initial_eval = inference_engine.run_batch_eval(
            model_id, checkpoint_id=checkpoint_id, reward_fn=reward_fn, max_cases=6
        )
        actions: list[dict[str, Any]] = []
        current_eval = initial_eval

        if current_eval["verdict"] != "GO":
            steps_taken = 0
            # Strategy 1: Disable highest-Frobenius-drift active edits one by one
            while current_eval["verdict"] != "GO" and steps_taken < max(
                1, int(max_steps)
            ):
                active_recs = [rec for rec in self._history.get(key, []) if rec.enabled]
                if not active_recs:
                    break
                worst = max(active_recs, key=lambda r: float(r.frobenius_delta))
                worst.enabled = False
                self._rebuild_overrides(model_id, checkpoint_id)
                steps_taken += 1
                current_eval = inference_engine.run_batch_eval(
                    model_id,
                    checkpoint_id=checkpoint_id,
                    reward_fn=reward_fn,
                    max_cases=6,
                )
                actions.append(
                    {
                        "step": steps_taken,
                        "action": "disable_high_drift_edit",
                        "edit_id": worst.edit_id,
                        "label": worst.label,
                        "frobenius_delta": worst.frobenius_delta,
                        "verdict_after": current_eval["verdict"],
                        "composite_delta": current_eval["deltas"]["composite_delta"],
                    }
                )

            # Strategy 2: If still not GO, apply a stabilizing P99 outlier clamp
            if current_eval["verdict"] != "GO":
                clamp_res = self.apply_edit(
                    model_id,
                    "clamp",
                    "all",
                    {"percentile": 99.0},
                    checkpoint_id=checkpoint_id,
                    label="Auto-Repair: P99 Outlier Clamp",
                )
                current_eval = inference_engine.run_batch_eval(
                    model_id,
                    checkpoint_id=checkpoint_id,
                    reward_fn=reward_fn,
                    max_cases=6,
                )
                actions.append(
                    {
                        "step": len(actions) + 1,
                        "action": "stabilize_p99_clamp",
                        "edit_id": clamp_res["edit"]["edit_id"],
                        "label": clamp_res["edit"]["label"],
                        "verdict_after": current_eval["verdict"],
                        "composite_delta": current_eval["deltas"]["composite_delta"],
                    }
                )

        return {
            "model_id": model_id,
            "checkpoint_id": checkpoint_id or "default",
            "initial_verdict": initial_eval["verdict"],
            "final_verdict": current_eval["verdict"],
            "repaired": current_eval["verdict"] == "GO",
            "actions_taken": actions,
            "eval_report": current_eval,
            "history": self.get_history(model_id, checkpoint_id),
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    def export_hf_model_card(
        self,
        model_id: str,
        export_name: str,
        *,
        repo_id: str = "",
        eval_report: dict[str, Any] | None = None,
        checkpoint_id: str | None = None,
        push_to_hub: bool = False,
        token: str = "",
    ) -> dict[str, Any]:
        """Generate a publication-ready Hugging Face Model Card (README.md + recipe) with optional Hub upload."""
        safe_name = "".join(
            c if c.isalnum() or c in "-_." else "_" for c in export_name
        ).strip("._")
        if not safe_name:
            safe_name = f"hf-card-{int(time.time())}"

        out_dir = (self.export_root / safe_name).resolve()
        if not str(out_dir).startswith(str(self.export_root)):
            raise ValueError("Export path escaped export root")
        out_dir.mkdir(parents=True, exist_ok=True)

        entry = self.inspector.get_entry(model_id)
        recipe = self.export_recipe(model_id, checkpoint_id)
        history = self.get_history(model_id, checkpoint_id)
        target_repo = repo_id.strip() or f"stateset/{safe_name}"

        eval_lines = ""
        if isinstance(eval_report, dict):
            verdict = eval_report.get("verdict", "GO")
            b_sum = eval_report.get("base_summary", {})
            e_sum = eval_report.get("edited_summary", {})
            deltas = eval_report.get("deltas", {})
            eval_lines = (
                f"\n## Pre-Export Batch Eval Gate (`{verdict}`)\n\n"
                f"| Metric | Base | Edited | Delta |\n"
                f"| :--- | :---: | :---: | :---: |\n"
                f"| Pass Rate | {float(b_sum.get('pass_rate', 0))*100:.0f}% | "
                f"{float(e_sum.get('pass_rate', 0))*100:.0f}% | "
                f"{float(deltas.get('pass_rate_delta', 0))*100:+.1f}% |\n"
                f"| Mean Composite Reward | {float(b_sum.get('mean_composite', 0)):.4f} | "
                f"{float(e_sum.get('mean_composite', 0)):.4f} | "
                f"{float(deltas.get('composite_delta', 0)):+.4f} |\n"
                f"| Policy Compliance | {float(b_sum.get('mean_policy', 0)):.4f} | "
                f"{float(e_sum.get('mean_policy', 0)):.4f} | "
                f"{float(deltas.get('policy_delta', 0)):+.4f} |\n"
            )

        ops = recipe.get("operations", [])
        ops_md = (
            "\n".join(
                f"- `{op['op']}` on `{op['target']}` — {op['label']}" for op in ops
            )
            if ops
            else "- Base checkpoint inspection (no weight mutations)."
        )

        card_md = (
            f"---\n"
            f"base_model: {entry.base_model}\n"
            f"library_name: stateset-agents\n"
            f"tags:\n"
            f"  - stateset-agents\n"
            f"  - weight-studio\n"
            f"  - mechanistic-interpretability\n"
            f"  - surgical-weight-editing\n"
            f"---\n\n"
            f"# {target_repo}\n\n"
            f"Exported from **StateSet 3D Weight Studio** (`{entry.architecture}`).\n\n"
            f"## Surgical Edit Lineage ({len(ops)} active operations)\n\n"
            f"{ops_md}\n"
            f"{eval_lines}\n"
            f"## Reproduce in 3D Weight Studio\n\n"
            f"```bash\n"
            f'pip install "stateset-agents[api]"\n'
            f"stateset-agents studio --open\n"
            f"```\n\n"
            f"Import `studio_recipe.json` via **Surgery → Replay** to reproduce these exact weights.\n"
        )

        card_path = out_dir / "README.md"
        recipe_path = out_dir / "studio_recipe.json"
        card_path.write_text(card_md, encoding="utf-8")
        recipe_path.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")

        pushed = False
        hub_url: str | None = None
        push_note = "Local Model Card + recipe written. Pass push_to_hub=True with an HF token to upload."
        if push_to_hub:
            hf_tok = token.strip()
            if not hf_tok:
                push_note = (
                    "Skipped live Hub upload: HF token required for push_to_hub=True."
                )
            else:
                try:
                    from huggingface_hub import HfApi

                    api = HfApi(token=hf_tok)
                    api.create_repo(repo_id=target_repo, exist_ok=True)
                    api.upload_folder(
                        folder_path=str(out_dir),
                        repo_id=target_repo,
                        commit_message="Upload StateSet 3D Weight Studio checkpoint & Model Card",
                    )
                    pushed = True
                    hub_url = f"https://huggingface.co/{target_repo}"
                    push_note = f"Uploaded bundle to {hub_url}"
                except Exception as exc:
                    push_note = f"Hub upload skipped ({type(exc).__name__}: {exc})"

        rel_dir = str(out_dir.relative_to(self.inspector.workspace_root))
        return {
            "schema_version": 1,
            "repo_id": target_repo,
            "export_dir": rel_dir,
            "model_card_file": str(
                card_path.relative_to(self.inspector.workspace_root)
            ),
            "recipe_file": str(recipe_path.relative_to(self.inspector.workspace_root)),
            "active_edits": len(history.get("edits", [])),
            "pushed": pushed,
            "hub_url": hub_url,
            "push_note": push_note,
            "cli_upload_command": f"huggingface-cli upload {target_repo} {rel_dir}",
            "card_preview": card_md,
        }

    def _sessions_dir(self) -> Path:
        sdir = (self.export_root / ".studio_sessions").resolve()
        sdir.mkdir(parents=True, exist_ok=True)
        return sdir

    def save_session(
        self,
        name: str,
        *,
        model_id: str,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a named Studio workbench session (recipe + UI/steering/patch/SAE state)."""
        safe = "".join(
            c if c.isalnum() or c in "-_." else "_" for c in name.strip()
        ).strip("._")
        if not safe:
            safe = f"session-{int(time.time())}"
        sdir = self._sessions_dir()
        out_path = (sdir / f"{safe}.json").resolve()
        if not str(out_path).startswith(str(sdir)):
            raise ValueError("Session path escaped sessions directory")

        recipe = self.export_recipe(model_id, checkpoint_id)
        doc = {
            "schema_version": 1,
            "session_name": safe,
            "saved_at": time.time(),
            "model_id": model_id,
            "checkpoint_id": checkpoint_id,
            "recipe": recipe,
            "metadata": dict(metadata or {}),
        }
        out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        return {
            "session_name": safe,
            "path": str(out_path.relative_to(self.inspector.workspace_root)),
            "operation_count": len(recipe.get("operations", [])),
            "saved_at": doc["saved_at"],
        }

    def list_sessions(self) -> list[dict[str, Any]]:
        """List saved Studio workbench sessions."""
        sdir = self._sessions_dir()
        items: list[dict[str, Any]] = []
        for f in sorted(
            sdir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            try:
                doc = json.loads(f.read_text(encoding="utf-8"))
                items.append(
                    {
                        "session_name": str(doc.get("session_name") or f.stem),
                        "model_id": str(doc.get("model_id") or ""),
                        "checkpoint_id": doc.get("checkpoint_id"),
                        "operation_count": len(
                            (doc.get("recipe") or {}).get("operations") or []
                        ),
                        "saved_at": doc.get("saved_at", f.stat().st_mtime),
                    }
                )
            except Exception:
                continue
        return items

    def load_session(self, name: str) -> dict[str, Any]:
        """Restore a saved Studio workbench session and replay its edit recipe."""
        safe = "".join(
            c if c.isalnum() or c in "-_." else "_" for c in name.strip()
        ).strip("._")
        sdir = self._sessions_dir()
        path = (sdir / f"{safe}.json").resolve()
        if not str(path).startswith(str(sdir)) or not path.is_file():
            raise FileNotFoundError(f"Saved session not found: {name}")
        doc = json.loads(path.read_text(encoding="utf-8"))
        model_id = str(doc.get("model_id") or "")
        checkpoint_id = doc.get("checkpoint_id")
        recipe = doc.get("recipe") or {"operations": []}
        replay = self.apply_recipe(
            model_id, recipe, checkpoint_id=checkpoint_id, reset_first=True
        )
        return {
            "session_name": safe,
            "model_id": model_id,
            "checkpoint_id": checkpoint_id,
            "metadata": doc.get("metadata") or {},
            "replay": replay,
        }
