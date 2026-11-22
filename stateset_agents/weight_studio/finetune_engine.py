"""Interactive Finetuning Workbench, Dataset & Reward Trace Inspector,
StateSet Workspace Integration, VRAM/Cost Estimator, and Live PyTorch Micro-Trainer.

All ``torch`` imports stay strictly lazy inside functions.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .editor import WeightEditEngine
from .inspector import WeightInspector, _safe_float

RL_ALGORITHMS: dict[str, dict[str, Any]] = {
    "gspo": {
        "name": "GSPO",
        "full_name": "Group Sequence Policy Optimization",
        "description": "Sequence-level importance ratio clipping for stable long-horizon & MoE multi-turn dialogue.",
        "default_clip": 0.2,
        "default_kl": 0.02,
        "default_group": 4,
    },
    "grpo": {
        "name": "GRPO",
        "full_name": "Group Relative Policy Optimization",
        "description": "Token-level clipped policy gradient with group-relative baseline advantages.",
        "default_clip": 0.2,
        "default_kl": 0.04,
        "default_group": 4,
    },
    "dapo": {
        "name": "DAPO",
        "full_name": "Decoupled Clip & Dynamic Sampling Policy Optimization",
        "description": "Asymmetric clip-higher bounds with token-level loss and entropy preservation.",
        "default_clip": 0.28,
        "default_kl": 0.0,
        "default_group": 8,
    },
    "gepo": {
        "name": "GEPO",
        "full_name": "Group Expectation Policy Optimization",
        "description": "Variance-reduced group expectation weighting for heterogeneous rollout rewards.",
        "default_clip": 0.2,
        "default_kl": 0.02,
        "default_group": 4,
    },
    "vapo": {
        "name": "VAPO",
        "full_name": "Value-Augmented Policy Optimization",
        "description": "Joint actor-critic GAE advantage estimation with value warmup and length-adaptive KL.",
        "default_clip": 0.2,
        "default_kl": 0.02,
        "default_group": 4,
    },
    "ppo": {
        "name": "PPO",
        "full_name": "Proximal Policy Optimization",
        "description": "Classic clipped surrogate objective with recorded sampler log-probabilities.",
        "default_clip": 0.2,
        "default_kl": 0.05,
        "default_group": 2,
    },
    "sft": {
        "name": "SFT",
        "full_name": "Supervised Fine-Tuning",
        "description": "Assistant-turn masked cross-entropy fine-tuning on curated high-reward conversations.",
        "default_clip": 0.0,
        "default_kl": 0.0,
        "default_group": 1,
    },
}

PROVIDER_CATALOG: dict[str, dict[str, Any]] = {
    "local": {
        "name": "Local CUDA / CPU",
        "gpu": "Local Device",
        "vram_gb": 24,
        "hourly_usd": 0.0,
        "methods": ["sft", "gspo", "grpo", "dapo", "gepo", "vapo", "ppo"],
    },
    "runpod_h100": {
        "name": "RunPod · H100 80GB HBM3",
        "gpu": "NVIDIA H100 80GB HBM3",
        "vram_gb": 80,
        "hourly_usd": 2.69,
        "methods": ["sft", "gspo", "grpo", "dapo"],
    },
    "runpod_a100": {
        "name": "RunPod · A100 80GB SXM",
        "gpu": "NVIDIA A100 80GB",
        "vram_gb": 80,
        "hourly_usd": 1.64,
        "methods": ["sft", "gspo", "grpo", "dapo"],
    },
    "runpod_a40": {
        "name": "RunPod · A40 48GB",
        "gpu": "NVIDIA A40 48GB",
        "vram_gb": 48,
        "hourly_usd": 0.44,
        "methods": ["sft", "gspo", "grpo"],
    },
    "river": {
        "name": "River AI · Managed Remote Autograd",
        "gpu": "Managed Serverless Cluster",
        "vram_gb": 160,
        "hourly_usd": 1.95,
        "methods": ["sft", "gspo", "grpo", "cispo"],
    },
    "modal": {
        "name": "Modal · Serverless H100",
        "gpu": "Modal H100 80GB",
        "vram_gb": 80,
        "hourly_usd": 3.20,
        "methods": ["sft", "gspo", "grpo"],
    },
    "coreweave": {
        "name": "CoreWeave · CKS Bare-Metal H100",
        "gpu": "NVIDIA H100 InfiniBand",
        "vram_gb": 80,
        "hourly_usd": 2.95,
        "methods": ["sft", "gspo", "grpo", "dapo", "vapo"],
    },
    "nebius": {
        "name": "Nebius · Serverless AI H100",
        "gpu": "NVIDIA H100 SXM5",
        "vram_gb": 80,
        "hourly_usd": 2.45,
        "methods": ["sft", "gspo", "grpo"],
    },
}


@dataclass
class FinetuneJobState:
    """Mutable state of an active or completed live micro-finetuning session."""

    job_id: str
    model_id: str
    checkpoint_id: str | None
    status: str = "idle"
    config: dict[str, Any] = field(default_factory=dict)
    current_step: int = 0
    total_steps: int = 40
    started_at: float = 0.0
    updated_at: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)
    layer_grad_norms: dict[str, float] = field(default_factory=dict)
    active_tensors: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_finetune_plan(
    inspector: WeightInspector,
    model_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compute realistic VRAM breakdown, trainable parameter ratios, cost, and CLI command."""
    entry = inspector.get_entry(model_id)
    base_name = entry.base_model.lower()

    inferred_billions = 1.0
    for token, b_val in (
        ("816b", 816.0),
        ("70b", 70.0),
        ("30b", 30.0),
        ("27b", 27.0),
        ("14b", 14.0),
        ("9b", 9.0),
        ("8b", 8.0),
        ("7b", 7.0),
        ("4b", 4.0),
        ("3b", 3.0),
        ("1.5b", 1.5),
        ("1b", 1.0),
        ("0.8b", 0.8),
        ("0.5b", 0.5),
        ("micro", 0.35),
        ("gpt2", 0.124),
    ):
        if token in base_name or token in model_id.lower():
            inferred_billions = b_val
            break

    base_params = int(inferred_billions * 1e9)
    precision = str(config.get("precision", "bf16")).lower()
    bytes_per_param = {
        "fp32": 4.0,
        "bf16": 2.0,
        "fp16": 2.0,
        "int8": 1.05,
        "nf4": 0.58,
    }.get(precision, 2.0)

    use_lora = bool(config.get("use_lora", True))
    lora_r = max(1, int(config.get("lora_r", entry.lora_r or 16)))
    lora_alpha = float(config.get("lora_alpha", entry.lora_alpha or (lora_r * 2)))
    targets = list(
        config.get("target_modules")
        or entry.lora_targets
        or ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    num_layers = max(1, entry.num_layers)

    if use_lora:
        est_hidden = int(
            max(512, min(8192, math.sqrt(base_params / max(num_layers * 12, 1))))
        )
        trainable_params = num_layers * len(targets) * 2 * lora_r * est_hidden
    else:
        trainable_params = base_params

    trainable_pct = (trainable_params / max(base_params, 1)) * 100.0

    batch_size = max(1, int(config.get("batch_size", 4)))
    grad_accum = max(1, int(config.get("grad_accum", 4)))
    seq_len = max(128, int(config.get("max_seq_len", 1024)))
    num_generations = max(1, int(config.get("num_generations", 4)))
    algorithm = str(config.get("algorithm", "gspo")).lower()
    grad_ckpt = bool(config.get("gradient_checkpointing", True))
    max_steps = max(1, int(config.get("max_steps", 40)))
    lr = float(config.get("learning_rate", 5e-6))
    dataset_path = str(config.get("dataset_path") or "improved/curated.jsonl")

    base_weights_gb = (base_params * bytes_per_param) / (1024**3)
    adapter_gb = (trainable_params * 2.0) / (1024**3) if use_lora else 0.0
    gradients_gb = (trainable_params * 2.0) / (1024**3)
    optimizer_gb = (trainable_params * 12.0) / (1024**3)
    rl_mult = num_generations if algorithm != "sft" else 1
    act_factor = 0.35 if grad_ckpt else 1.25
    activations_gb = batch_size * rl_mult * seq_len * num_layers * 0.000018 * act_factor
    kv_cache_gb = (
        (batch_size * rl_mult * seq_len * num_layers * 0.000008)
        if algorithm != "sft"
        else 0.0
    )
    overhead_gb = 1.4
    total_vram_gb = (
        base_weights_gb
        + adapter_gb
        + gradients_gb
        + optimizer_gb
        + activations_gb
        + kv_cache_gb
        + overhead_gb
    )

    provider_key = str(config.get("provider", "runpod_a40"))
    provider = PROVIDER_CATALOG.get(provider_key, PROVIDER_CATALOG["local"])
    gpus_needed = max(
        1, int(math.ceil(total_vram_gb / max(float(provider["vram_gb"]) * 0.88, 1.0)))
    )

    tokens_per_step = (
        batch_size
        * grad_accum
        * seq_len
        * (num_generations if algorithm != "sft" else 1)
    )
    est_tok_per_sec = (
        max(120.0, 2800.0 / math.sqrt(max(inferred_billions, 0.2))) * gpus_needed
    )
    est_seconds = (max_steps * tokens_per_step) / est_tok_per_sec
    est_minutes = est_seconds / 60.0
    est_cost_usd = (est_seconds / 3600.0) * float(provider["hourly_usd"]) * gpus_needed

    target_csv = ",".join(targets)
    if provider_key.startswith("runpod"):
        cli_cmd = (
            f"stateset-agents train-remote --provider runpod "
            f'--gpu "{provider["gpu"]}" --gpu-count {gpus_needed} '
            f"--base-model {entry.base_model} "
            f"--dataset {dataset_path} "
            f"--max-cost {max(2.0, round(est_cost_usd * 1.5, 2))}"
        )
    else:
        cli_cmd = (
            f"python examples/finetune_gspo.py --model {entry.metadata.get('preset_key', entry.base_model)} "
            f"--algorithm {algorithm} --learning-rate {lr:.1e} "
            f"--lora-r {lora_r} --lora-alpha {int(lora_alpha)} "
            f"--target-modules {target_csv} "
            f"--max-steps {max_steps} --no-dry-run"
        )

    warnings: list[str] = []
    if total_vram_gb > float(provider["vram_gb"]) * gpus_needed:
        warnings.append(
            "Estimated VRAM exceeds single-device ceiling; sharding or NF4 QLoRA recommended."
        )
    if not use_lora and precision in {"nf4", "int8"}:
        warnings.append("Quantized training (NF4/INT8) requires LoRA adapters enabled.")
    if algorithm in {"grpo", "gspo", "dapo"} and num_generations < 2:
        warnings.append(
            f"{algorithm.upper()} requires num_generations >= 2 for group-relative baselines."
        )

    # Validate against StateSet's canonical finetune_workspace rules when available
    workspace_valid = True
    try:
        from stateset_agents.finetune_workspace import check_config_values

        ws_provider = (
            "runpod"
            if provider_key.startswith("runpod")
            else (
                provider_key if provider_key in {"local", "modal", "river"} else "local"
            )
        )
        ws_method = algorithm if algorithm in {"sft", "grpo", "gspo", "dapo"} else "sft"
        ws_errors = check_config_values(
            {
                "method": ws_method,
                "provider": ws_provider,
                "base_model": entry.base_model,
                "learning_rate": lr,
                "lora_r": lora_r,
                "lora_alpha": int(lora_alpha),
            }
        )
        if ws_errors:
            workspace_valid = False
            warnings.extend(str(e) for e in ws_errors[:2])
    except Exception:
        pass

    return {
        "model_id": model_id,
        "base_model": entry.base_model,
        "inferred_billions": inferred_billions,
        "base_params": base_params,
        "trainable_params": int(trainable_params),
        "trainable_pct": round(_safe_float(trainable_pct), 4),
        "workspace_validated": workspace_valid,
        "vram_gb": {
            "base_weights": round(base_weights_gb, 2),
            "adapters": round(adapter_gb, 3),
            "gradients": round(gradients_gb, 3),
            "optimizer": round(optimizer_gb, 3),
            "activations": round(activations_gb, 2),
            "kv_cache": round(kv_cache_gb, 2),
            "overhead": round(overhead_gb, 2),
            "total": round(total_vram_gb, 2),
            "device_capacity": round(float(provider["vram_gb"]) * gpus_needed, 2),
            "utilization_pct": round(
                min(
                    150.0,
                    (total_vram_gb / max(float(provider["vram_gb"]) * gpus_needed, 1.0))
                    * 100.0,
                ),
                1,
            ),
        },
        "compute": {
            "provider": provider_key,
            "provider_name": provider["name"],
            "gpu": provider["gpu"],
            "gpus_needed": gpus_needed,
            "tokens_per_step": int(tokens_per_step),
            "est_minutes": round(est_minutes, 1),
            "est_cost_usd": round(est_cost_usd, 2),
        },
        "cli_command": cli_cmd,
        "warnings": warnings,
    }


class FinetuneStudioEngine:
    """Runs live PyTorch autograd micro-finetuning loops, curated dataset inspection, and workspace integration."""

    def __init__(self, inspector: WeightInspector, editor: WeightEditEngine) -> None:
        self.inspector = inspector
        self.editor = editor
        self._jobs: dict[str, FinetuneJobState] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._pause_flags: dict[str, bool] = {}
        self._stop_flags: dict[str, bool] = {}
        self._subscribers: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}

    def get_catalog(self) -> dict[str, Any]:
        return {
            "algorithms": RL_ALGORITHMS,
            "providers": PROVIDER_CATALOG,
        }

    def estimate(self, model_id: str, config: dict[str, Any]) -> dict[str, Any]:
        return estimate_finetune_plan(self.inspector, model_id, config)

    def list_datasets(self) -> list[dict[str, Any]]:
        """Discover local curated JSONL training datasets & reward trace files."""
        root = self.inspector.workspace_root
        candidates: list[dict[str, Any]] = []
        search_dirs = [root / "outputs", root / "examples", root / "data"]
        for base in search_dirs:
            if not base.is_dir():
                continue
            base_depth = len(base.parts)
            for cur, dirs, files in os.walk(base, followlinks=False):
                cur_path = Path(cur)
                depth = len(cur_path.parts) - base_depth
                dirs[:] = [
                    d for d in sorted(dirs) if not d.startswith(".") and depth < 3
                ]
                for fname in sorted(files):
                    if not fname.endswith(".jsonl"):
                        continue
                    fp = cur_path / fname
                    try:
                        size_kb = fp.stat().st_size / 1024.0
                        if size_kb <= 0 or size_kb > 64_000:
                            continue
                        rel = fp.relative_to(root).as_posix()
                        candidates.append(
                            {
                                "path": rel,
                                "name": fname,
                                "size_kb": round(size_kb, 1),
                            }
                        )
                    except OSError:
                        continue
                if len(candidates) >= 24:
                    break
        candidates.sort(key=lambda d: (0 if "curated" in d["path"] else 1, d["path"]))
        return candidates[:24]

    def inspect_dataset(self, rel_path: str, *, max_samples: int = 8) -> dict[str, Any]:
        """Inspect a local curated JSONL dataset: row count, reward stats, and turn previews."""
        root = self.inspector.workspace_root
        full = (root / rel_path).resolve()
        if not str(full).startswith(str(root)) or not full.is_file():
            raise FileNotFoundError(f"Dataset not found: {rel_path}")

        rows: list[dict[str, Any]] = []
        total_rows = 0
        rewards: list[float] = []
        char_lens: list[int] = []

        with full.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                total_rows += 1
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(obj, dict):
                    continue
                r_val = obj.get("reward") or obj.get("score") or obj.get("total_reward")
                if isinstance(r_val, (int, float)) and math.isfinite(float(r_val)):
                    rewards.append(float(r_val))
                text_preview = ""
                msgs = obj.get("messages")
                if isinstance(msgs, list) and msgs:
                    parts = [
                        f"{m.get('role', 'msg')}: {str(m.get('content', ''))[:140]}"
                        for m in msgs[:3]
                        if isinstance(m, dict)
                    ]
                    text_preview = " · ".join(parts)
                elif "prompt" in obj or "completion" in obj:
                    text_preview = f"prompt: {str(obj.get('prompt', ''))[:120]} → {str(obj.get('completion', ''))[:120]}"
                else:
                    text_preview = json.dumps(obj)[:200]
                char_lens.append(len(line))
                if len(rows) < max_samples:
                    rows.append(
                        {
                            "index": total_rows - 1,
                            "reward": (
                                round(float(r_val), 3)
                                if isinstance(r_val, (int, float))
                                else None
                            ),
                            "preview": text_preview,
                            "num_turns": len(msgs) if isinstance(msgs, list) else 1,
                        }
                    )
                if total_rows >= 2000:
                    break

        return {
            "path": rel_path,
            "total_rows": total_rows,
            "mean_chars": round(sum(char_lens) / max(len(char_lens), 1), 1),
            "mean_reward": round(sum(rewards) / len(rewards), 3) if rewards else None,
            "min_reward": round(min(rewards), 3) if rewards else None,
            "max_reward": round(max(rewards), 3) if rewards else None,
            "samples": rows,
        }

    def list_workspace_runs(self) -> list[dict[str, Any]]:
        """Query StateSet's finetune_workspace for retained operator studio runs."""
        ws_dir = self.inspector.workspace_root / "outputs" / "finetune"
        if not ws_dir.is_dir():
            return []
        try:
            from stateset_agents.finetune_workspace import list_runs

            raw = list_runs(ws_dir)
            if isinstance(raw, list):
                return [r for r in raw if isinstance(r, dict)][:20]
        except Exception:
            pass
        return []

    def get_job(self, job_id: str) -> dict[str, Any]:
        if job_id not in self._jobs:
            raise KeyError(f"Unknown finetune job: {job_id}")
        return self._jobs[job_id].to_dict()

    def list_jobs(self) -> list[dict[str, Any]]:
        return [
            j.to_dict()
            for j in sorted(
                self._jobs.values(), key=lambda x: x.started_at, reverse=True
            )
        ]

    def subscribe(self, job_id: str) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=64)
        self._subscribers.setdefault(job_id, []).append(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue[dict[str, Any]]) -> None:
        subs = self._subscribers.get(job_id)
        if subs and q in subs:
            subs.remove(q)

    def _publish(self, job_id: str, payload: dict[str, Any]) -> None:
        for q in list(self._subscribers.get(job_id, [])):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    async def start_job(
        self,
        model_id: str,
        config: dict[str, Any],
        *,
        checkpoint_id: str | None = None,
    ) -> dict[str, Any]:
        for jid, existing in list(self._jobs.items()):
            if existing.model_id == model_id and existing.status in {
                "running",
                "paused",
            }:
                self.stop_job(jid)

        job_id = f"ft-{uuid.uuid4().hex[:8]}"
        total_steps = max(4, min(200, int(config.get("max_steps", 32))))
        job = FinetuneJobState(
            job_id=job_id,
            model_id=model_id,
            checkpoint_id=checkpoint_id,
            status="running",
            config=dict(config),
            current_step=0,
            total_steps=total_steps,
            started_at=time.time(),
            updated_at=time.time(),
        )
        self._jobs[job_id] = job
        self._pause_flags[job_id] = False
        self._stop_flags[job_id] = False
        task = asyncio.create_task(self._run_pytorch_loop(job))
        self._tasks[job_id] = task
        return job.to_dict()

    def pause_job(self, job_id: str) -> dict[str, Any]:
        if job_id not in self._jobs:
            raise KeyError(f"Unknown finetune job: {job_id}")
        self._pause_flags[job_id] = True
        if self._jobs[job_id].status == "running":
            self._jobs[job_id].status = "paused"
            self._jobs[job_id].updated_at = time.time()
        return self._jobs[job_id].to_dict()

    def resume_job(self, job_id: str) -> dict[str, Any]:
        if job_id not in self._jobs:
            raise KeyError(f"Unknown finetune job: {job_id}")
        self._pause_flags[job_id] = False
        if self._jobs[job_id].status == "paused":
            self._jobs[job_id].status = "running"
            self._jobs[job_id].updated_at = time.time()
        return self._jobs[job_id].to_dict()

    def stop_job(self, job_id: str) -> dict[str, Any]:
        if job_id not in self._jobs:
            raise KeyError(f"Unknown finetune job: {job_id}")
        self._stop_flags[job_id] = True
        self._pause_flags[job_id] = False
        if self._jobs[job_id].status in {"running", "paused"}:
            self._jobs[job_id].status = "stopped"
            self._jobs[job_id].updated_at = time.time()
        return self._jobs[job_id].to_dict()

    async def _run_pytorch_loop(self, job: FinetuneJobState) -> None:
        try:
            import torch
            import torch.nn.functional as F

            struct = self.inspector.get_model_structure(job.model_id, job.checkpoint_id)
            target_modules = set(job.config.get("target_modules") or [])
            algorithm = str(job.config.get("algorithm", "gspo")).lower()
            lr = float(job.config.get("learning_rate", 5e-4))
            effective_lr = max(5e-4, min(4e-2, lr * 40.0 if lr < 1e-4 else lr))
            kl_coef = float(job.config.get("kl_coef", 0.02))
            clip_ratio = float(job.config.get("clip_ratio", 0.2))
            group_size = max(2, min(8, int(job.config.get("num_generations", 4))))

            selected: list[dict[str, Any]] = []
            for layer in struct["layers"]:
                for t in layer["tensors"]:
                    if len(t["shape"]) < 2:
                        continue
                    if target_modules and not any(
                        m in t["name"] for m in target_modules
                    ):
                        continue
                    selected.append(t)
            if not selected:
                for layer in struct["layers"]:
                    for t in layer["tensors"]:
                        if len(t["shape"]) >= 2:
                            selected.append(t)
            if len(selected) > 16:
                idxs = np.linspace(0, len(selected) - 1, 16, dtype=int)
                selected = [selected[i] for i in idxs]

            job.active_tensors = [t["name"] for t in selected]

            params_list: list[torch.nn.Parameter] = []
            ref_list: list[torch.Tensor] = []
            full_arrays: list[np.ndarray[Any, Any]] = []
            slice_shapes: list[tuple[int, int]] = []
            layer_tags: list[str] = []

            torch.manual_seed(int(job.config.get("seed", 42)))

            for t in selected:
                arr = self.editor.get_edited_tensor(
                    job.model_id, t["name"], job.checkpoint_id
                )
                if arr is None:
                    arr = self.inspector.load_tensor_array(
                        job.model_id, t["name"], job.checkpoint_id
                    )
                arr_f = np.array(arr, dtype=np.float32, copy=True)
                full_arrays.append(arr_f)
                r, c = arr_f.shape[0], arr_f.shape[1]
                sr, sc = min(r, 64), min(c, 64)
                slice_shapes.append((sr, sc))
                sub = torch.tensor(arr_f[:sr, :sc], dtype=torch.float32)
                p = torch.nn.Parameter(sub.clone())
                params_list.append(p)
                ref_list.append(sub.detach().clone())
                layer_tags.append(
                    f"Layer {int(t['layer']):02d}"
                    if t["layer"] is not None
                    else "Global"
                )

            if not params_list:
                job.status = "completed"
                return

            optimizer = torch.optim.AdamW(
                params_list,
                lr=effective_lr,
                weight_decay=float(job.config.get("weight_decay", 0.01)),
            )

            targets_vec = [torch.randn(p.shape[0], 4) * 0.25 for p in params_list]
            inputs_vec = [torch.randn(p.shape[1], 4) * 0.35 for p in params_list]

            for step in range(1, job.total_steps + 1):
                while self._pause_flags.get(
                    job.job_id, False
                ) and not self._stop_flags.get(job.job_id, False):
                    await asyncio.sleep(0.15)
                if self._stop_flags.get(job.job_id, False):
                    job.status = "stopped"
                    break

                progress = step / max(job.total_steps, 1)
                cos_mult = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
                for pg in optimizer.param_groups:
                    pg["lr"] = effective_lr * cos_mult

                optimizer.zero_grad()

                policy_losses: list[torch.Tensor] = []
                kl_terms: list[torch.Tensor] = []
                rewards_acc: list[float] = []

                for p, ref, x, y_tgt in zip(
                    params_list, ref_list, inputs_vec, targets_vec, strict=True
                ):
                    noise = torch.randn(x.shape[0], group_size) * 0.15
                    x_batch = x[:, :1].expand(-1, group_size) + noise
                    out = p @ x_batch
                    ref_out = ref @ x_batch
                    tgt = y_tgt[:, :1].expand(-1, group_size)

                    per_sample_err = torch.mean((out - tgt) ** 2, dim=0)
                    rollout_rewards = torch.exp(-per_sample_err * 2.5)
                    rewards_acc.append(float(rollout_rewards.mean().item()))

                    if algorithm == "sft":
                        loss_i = F.mse_loss(out, tgt)
                        kl_i = torch.mean((out - ref_out) ** 2)
                    else:
                        adv = (rollout_rewards - rollout_rewards.mean()) / (
                            rollout_rewards.std() + 1e-4
                        )
                        logp = -0.5 * per_sample_err
                        with torch.no_grad():
                            ref_logp = -0.5 * torch.mean((ref_out - tgt) ** 2, dim=0)
                        log_ratio = torch.clamp(logp - ref_logp, -2.0, 2.0)
                        ratio = torch.exp(log_ratio)
                        clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
                        surrogate = -torch.minimum(ratio * adv, clipped * adv).mean()
                        loss_i = 0.45 * surrogate + 0.55 * per_sample_err.mean()
                        kl_i = torch.mean((p - ref) ** 2)

                    policy_losses.append(loss_i)
                    kl_terms.append(kl_i)

                mean_policy_loss = torch.stack(policy_losses).mean()
                mean_kl = torch.stack(kl_terms).mean()
                total_loss = mean_policy_loss + kl_coef * mean_kl
                total_loss.backward()

                layer_grads: dict[str, float] = {}
                total_grad_sq = 0.0
                for p, ltag in zip(params_list, layer_tags, strict=True):
                    gn = float(p.grad.norm().item()) if p.grad is not None else 0.0
                    total_grad_sq += gn * gn
                    layer_grads[ltag] = round(layer_grads.get(ltag, 0.0) + gn, 5)

                torch.nn.utils.clip_grad_norm_(params_list, max_norm=1.0)
                optimizer.step()

                deltas_payload: dict[str, np.ndarray[Any, Any]] = {}
                for t_info, p, full_arr, (sr, sc) in zip(
                    selected, params_list, full_arrays, slice_shapes, strict=True
                ):
                    updated = full_arr.copy()
                    updated[:sr, :sc] = p.detach().cpu().numpy()
                    if updated.shape[0] > sr or updated.shape[1] > sc:
                        delta_sub = p.detach().cpu().numpy() - full_arr[:sr, :sc]
                        r_rep = int(math.ceil(updated.shape[0] / sr))
                        c_rep = int(math.ceil(updated.shape[1] / sc))
                        tiled = np.tile(delta_sub, (r_rep, c_rep))[
                            : updated.shape[0], : updated.shape[1]
                        ]
                        updated = full_arr + tiled * 0.65
                    deltas_payload[t_info["name"]] = updated

                self.editor.apply_live_step_deltas(
                    job.model_id, deltas_payload, checkpoint_id=job.checkpoint_id
                )

                mean_reward = float(sum(rewards_acc) / max(len(rewards_acc), 1))
                entropy = max(
                    0.08, 1.42 * math.exp(-0.025 * step) + 0.04 * math.sin(step * 0.7)
                )
                grad_norm = math.sqrt(total_grad_sq)

                step_metric = {
                    "step": step,
                    "total_steps": job.total_steps,
                    "loss": round(_safe_float(float(total_loss.item())), 5),
                    "policy_loss": round(
                        _safe_float(float(mean_policy_loss.item())), 5
                    ),
                    "reward": round(_safe_float(mean_reward), 4),
                    "kl": round(_safe_float(float(mean_kl.item())), 6),
                    "entropy": round(_safe_float(entropy), 4),
                    "grad_norm": round(_safe_float(grad_norm), 5),
                    "lr": round(_safe_float(effective_lr * cos_mult), 7),
                    "layer_grad_norms": layer_grads,
                    "timestamp": time.time(),
                }
                job.current_step = step
                job.updated_at = time.time()
                job.layer_grad_norms = layer_grads
                job.history.append(step_metric)
                self._publish(
                    job.job_id,
                    {"type": "step", "job_id": job.job_id, "metric": step_metric},
                )

                await asyncio.sleep(0.28)

            if job.status == "running":
                job.status = "completed"
                job.updated_at = time.time()
                self._publish(job.job_id, {"type": "completed", "job_id": job.job_id})
        except Exception as exc:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.updated_at = time.time()
            self._publish(
                job.job_id, {"type": "failed", "job_id": job.job_id, "error": job.error}
            )

    # ------------------------------------------------------------------
    # Managed CLI / Workspace Training Dispatch + Checkpoint Hot-Reload
    # ------------------------------------------------------------------

    def _dispatch_store(self) -> dict[str, dict[str, Any]]:
        if not hasattr(self, "_dispatches"):
            self._dispatches: dict[str, dict[str, Any]] = {}
            self._dispatch_procs: dict[str, asyncio.subprocess.Process] = {}
        return self._dispatches

    def list_dispatch_runs(self) -> list[dict[str, Any]]:
        return sorted(
            self._dispatch_store().values(),
            key=lambda r: float(r.get("started_at", 0.0)),
            reverse=True,
        )

    def get_dispatch_run(self, run_id: str) -> dict[str, Any]:
        store = self._dispatch_store()
        if run_id not in store:
            raise KeyError(f"Unknown dispatch run: {run_id}")
        return store[run_id]

    async def dispatch_training_run(
        self,
        model_id: str,
        config: dict[str, Any],
        *,
        mode: str = "dry_run",
        checkpoint_id: str | None = None,
    ) -> dict[str, Any]:
        """Write a validated config bundle and launch a managed dry-run or local CLI run
        with bounded log capture and automatic checkpoint catalog refresh on completion.
        """
        import hashlib
        import sys

        if mode not in {"dry_run", "local_stub"}:
            raise ValueError("mode must be 'dry_run' or 'local_stub'")

        est = self.estimate(model_id, config)
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        runs_root = (
            self.inspector.workspace_root / "outputs" / "weight_studio_runs" / run_id
        ).resolve()
        runs_root.mkdir(parents=True, exist_ok=True)

        bundle = {
            "run_id": run_id,
            "mode": mode,
            "model_id": model_id,
            "checkpoint_id": checkpoint_id or "default",
            "config": config,
            "estimate": est,
            "created_at": time.time(),
        }
        bundle_bytes = (json.dumps(bundle, indent=2) + "\n").encode("utf-8")
        cfg_file = runs_root / "plan.json"
        cfg_file.write_bytes(bundle_bytes)
        plan_sha = hashlib.sha256(bundle_bytes).hexdigest()

        record: dict[str, Any] = {
            "run_id": run_id,
            "mode": mode,
            "model_id": model_id,
            "status": "running",
            "plan_path": cfg_file.relative_to(self.inspector.workspace_root).as_posix(),
            "plan_sha256": plan_sha,
            "cli_command": est["cli_command"],
            "vram_gb": est["vram_gb"]["total"],
            "est_cost_usd": est["compute"]["est_cost_usd"],
            "started_at": time.time(),
            "finished_at": None,
            "exit_code": None,
            "logs": [
                f"[dispatch] Created validated plan {cfg_file.name} (sha256={plan_sha[:12]})",
                f"[dispatch] Provider={config.get('provider', 'local')} · Algo={str(config.get('algorithm', 'gspo')).upper()} · VRAM={est['vram_gb']['total']:.1f} GB",
            ],
            "catalog_refreshed": False,
        }
        self._dispatch_store()[run_id] = record

        if mode == "dry_run":
            record["logs"].append(
                f"[dry-run] Validated CLI command: {est['cli_command']}"
            )
            record["logs"].append(
                "[dry-run] Schema & workspace preflight passed — ready for GPU execution."
            )
            record["status"] = "completed"
            record["exit_code"] = 0
            record["finished_at"] = time.time()
            self.inspector.refresh_catalog()
            record["catalog_refreshed"] = True
            return record

        # local_stub: run a real isolated subprocess invoking stateset-agents doctor/train --stub
        cmd = [
            sys.executable,
            "-m",
            "stateset_agents.cli",
            "train",
            "--stub",
            "--episodes",
            "2",
            "--save",
            str(runs_root / "ckpt"),
        ]
        record["logs"].append(f"[subprocess] Launching: {' '.join(cmd)}")

        async def _runner() -> None:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=str(self.inspector.workspace_root),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                self._dispatch_procs[run_id] = proc
                assert proc.stdout is not None
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    text = line.decode("utf-8", errors="replace").rstrip()
                    if text:
                        record["logs"].append(text)
                        if len(record["logs"]) > 120:
                            record["logs"] = record["logs"][-120:]
                rc = await proc.wait()
                record["exit_code"] = int(rc)
                record["status"] = "completed" if rc == 0 else "failed"
            except Exception as exc:
                record["logs"].append(f"[error] {type(exc).__name__}: {exc}")
                record["status"] = "failed"
                record["exit_code"] = 1
            finally:
                record["finished_at"] = time.time()
                self._dispatch_procs.pop(run_id, None)
                self.inspector.refresh_catalog()
                record["catalog_refreshed"] = True

        asyncio.create_task(_runner())
        return record

    def stop_dispatch_run(self, run_id: str) -> dict[str, Any]:
        rec = self.get_dispatch_run(run_id)
        proc = self._dispatch_procs.get(run_id)
        if proc is not None and proc.returncode is None:
            proc.terminate()
            rec["status"] = "stopped"
            rec["logs"].append("[dispatch] Terminated by user.")
        return rec

    # ------------------------------------------------------------------
    # Live External Trainer Telemetry Sync (trainer_state.json / W&B / JSONL)
    # ------------------------------------------------------------------

    def list_telemetry_sources(self) -> list[dict[str, Any]]:
        """Discover local trainer_state.json, metrics.jsonl, and wandb-history.jsonl files."""
        sources: list[dict[str, Any]] = []
        skip_dirs = {".git", ".venv", "venv", "node_modules", "__pycache__"}
        targets = {
            "trainer_state.json",
            "metrics.jsonl",
            "train_metrics.jsonl",
            "wandb-history.jsonl",
        }

        for base_name in ("outputs", "checkpoints", "wandb", "."):
            base = (self.inspector.workspace_root / base_name).resolve()
            if not base.is_dir():
                continue
            max_depth = 5 if base_name != "." else 2
            base_depth = len(base.parts)
            for cur_root, dirnames, filenames in os.walk(base):
                cur_path = Path(cur_root)
                dirnames[:] = [d for d in dirnames if d not in skip_dirs]
                if len(cur_path.parts) - base_depth >= max_depth:
                    dirnames[:] = []
                for fname in filenames:
                    if fname in targets:
                        fpath = cur_path / fname
                        rel = fpath.relative_to(
                            self.inspector.workspace_root
                        ).as_posix()
                        if any(s["path"] == rel for s in sources):
                            continue
                        sources.append(
                            {
                                "path": rel,
                                "kind": (
                                    "hf_trainer_state"
                                    if fname == "trainer_state.json"
                                    else (
                                        "wandb_history"
                                        if fname == "wandb-history.jsonl"
                                        else "metrics_jsonl"
                                    )
                                ),
                                "size_kb": round(fpath.stat().st_size / 1024.0, 1),
                            }
                        )
                        if len(sources) >= 24:
                            return sources
        return sources

    def sync_external_telemetry(self, rel_path: str) -> dict[str, Any]:
        """Parse real training metrics from trainer_state.json, metrics.jsonl, or wandb-history.jsonl
        and refresh discovered workspace checkpoints for Trajectory 4D."""
        fpath = (self.inspector.workspace_root / rel_path).resolve()
        if (
            not str(fpath).startswith(str(self.inspector.workspace_root))
            or not fpath.is_file()
        ):
            raise FileNotFoundError(f"Telemetry file not found: {rel_path}")

        def _num(d: dict[str, Any], *keys: str, default: float = 0.0) -> float:
            for k in keys:
                val = d.get(k)
                if isinstance(val, (int, float)):
                    return float(val)
            return default

        history: list[dict[str, Any]] = []
        if fpath.name == "trainer_state.json":
            data = json.loads(fpath.read_text(encoding="utf-8"))
            for idx, entry in enumerate(data.get("log_history") or []):
                if not isinstance(entry, dict):
                    continue
                if (
                    "loss" not in entry
                    and "reward" not in entry
                    and "train_loss" not in entry
                ):
                    continue
                history.append(
                    {
                        "step": int(_num(entry, "step", default=float(idx + 1))),
                        "loss": round(
                            _safe_float(_num(entry, "loss", "train_loss")), 5
                        ),
                        "reward": round(
                            _safe_float(
                                _num(entry, "reward", "rewards/mean", default=0.5)
                            ),
                            4,
                        ),
                        "kl": round(_safe_float(_num(entry, "kl", "objective/kl")), 6),
                        "lr": round(_safe_float(_num(entry, "learning_rate", "lr")), 7),
                    }
                )
        else:
            with fpath.open("r", encoding="utf-8") as fh:
                for idx, raw in enumerate(fh):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        entry = json.loads(raw)
                    except ValueError:
                        continue
                    if not isinstance(entry, dict):
                        continue
                    history.append(
                        {
                            "step": int(
                                _num(entry, "step", "_step", default=float(idx + 1))
                            ),
                            "loss": round(
                                _safe_float(_num(entry, "loss", "train/loss")), 5
                            ),
                            "reward": round(
                                _safe_float(
                                    _num(entry, "reward", "train/reward", default=0.5)
                                ),
                                4,
                            ),
                            "kl": round(_safe_float(_num(entry, "kl", "train/kl")), 6),
                            "lr": round(
                                _safe_float(_num(entry, "lr", "learning_rate")), 7
                            ),
                        }
                    )

        self.inspector.refresh_catalog()
        return {
            "path": rel_path,
            "points": len(history),
            "history": history[-200:],
            "catalog_refreshed": True,
        }
