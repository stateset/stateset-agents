"""Declarative Open-Weight RL Blueprint & 1-Command Cloud/Local Launcher.

Unifies the flagship StateSet open-weight RL workflow into one declarative
config (YAML/JSON/Python) supporting:

1. **Step 0 Surgical Warm-Start** (optional Weight Studio Auto-Heal / CAA baking
   before RL begins to eliminate cold-start reward-zero plateaus).
2. **Memory-Efficient Chunked GSPO / GRPO / DAPO** (chunked LM-head projection,
   DAPO dynamic sampling, optional disaggregated async vLLM rollout pool).
3. **Online Reward-Hacking Circuit-Breaker** (`OnlineRewardHackingGuard` +
   `PluggableRewardJudge` calibration).
4. **1-Command Target Launch Plans** (`local`, `runpod`, `modal`, `k8s`) with
   reproducible SHA-256 plan digests.

Zero eager ``torch`` imports.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RLBlueprintConfig:
    """Declarative specification for an end-to-end open-weight RL run."""

    name: str = "stateset-agent-rl-v1"
    model: str = "Qwen/Qwen3-8B"
    algorithm: str = "gspo"
    env: str = "commerce_sandbox"
    reward_fn: str = "customer_support"
    cloud: str = "local"
    gpus: int = 1
    gpu_type: str = "H100-80GB"
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    chunked_loss_chunk_size: int = 256
    use_checkpointing: bool = True
    dynamic_sampling: bool = True
    disaggregated_rollouts: bool = True
    surgical_warm_start: bool = True
    warm_start_model_id: str = "preset:deepseek-r1-1.5b"
    online_hacking_guard: bool = True
    max_steps: int = 100
    group_size: int = 8
    kl_coef: float = 0.02
    clip_low: float = 0.20
    clip_high: float = 0.28
    live_vllm_endpoint: str = ""
    mcp_endpoint: str = ""
    fp8_rollout_tis_clip: float | None = 2.0
    moe_aux_loss_coef: float = 0.0
    moe_expert_lora_targets: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.algorithm not in ("grpo", "gspo", "dapo", "vapo", "gepo"):
            raise ValueError(f"Unsupported RL algorithm: {self.algorithm!r}")
        if self.cloud not in ("local", "runpod", "modal", "k8s"):
            raise ValueError(
                f"Unsupported cloud target {self.cloud!r}; expected local, runpod, modal, or k8s."
            )
        if self.gpus < 1:
            raise ValueError("gpus must be >= 1")
        if self.chunked_loss_chunk_size < 1:
            raise ValueError("chunked_loss_chunk_size must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_file(cls, path: str | Path) -> RLBlueprintConfig:
        p = Path(path).expanduser()
        raw = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(raw) or {}
        else:
            data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"Blueprint file {p} must contain a mapping object.")
        return cls(**data)


def estimate_chunked_vram_savings(
    *,
    batch_size: int = 4,
    seq_len: int = 16384,
    vocab_size: int = 152064,
    chunk_size: int = 256,
    bytes_per_elem: int = 4,
) -> dict[str, Any]:
    """Estimate peak vocabulary-logit memory for full vs. chunked RL loss."""
    full_bytes = batch_size * max(1, seq_len - 1) * vocab_size * bytes_per_elem
    chunk_bytes = (
        min(batch_size * max(1, seq_len - 1), chunk_size) * vocab_size * bytes_per_elem
    )
    saved_pct = round((1.0 - (chunk_bytes / max(1, full_bytes))) * 100.0, 2)
    return {
        "full_logits_gb": round(full_bytes / (1024**3), 3),
        "chunked_logits_gb": round(chunk_bytes / (1024**3), 3),
        "peak_logit_vram_saved_pct": saved_pct,
        "chunk_size": chunk_size,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
    }


def build_cloud_launch_plan(
    config: RLBlueprintConfig,
    *,
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    """Generate a reproducible, auditable launch plan for local, RunPod, Modal, or Kubernetes."""
    ws = (workspace_root or Path(".")).expanduser().resolve()
    payload = config.to_dict()
    plan_sha = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    base_cmd = [
        "stateset-agents",
        "finetune",
        "train",
        "--model",
        config.model,
        "--algorithm",
        config.algorithm,
        "--max-steps",
        str(config.max_steps),
    ]

    dispatch: dict[str, Any]
    if config.cloud == "local":
        dispatch = {
            "provider": "local",
            "command": " ".join(base_cmd),
            "env": {
                "STATESET_CHUNKED_LOSS_SIZE": str(config.chunked_loss_chunk_size),
                "STATESET_DYNAMIC_SAMPLING": "1" if config.dynamic_sampling else "0",
            },
        }
    elif config.cloud == "runpod":
        dispatch = {
            "provider": "runpod",
            "gpu_spec": f"{config.gpus}x {config.gpu_type}",
            "image": "ghcr.io/stateset/stateset-agents:latest-cuda124",
            "command": " ".join(base_cmd),
        }
    elif config.cloud == "modal":
        dispatch = {
            "provider": "modal",
            "gpu_spec": f"{config.gpu_type}:{config.gpus}",
            "entrypoint": f"modal run scripts/modal_rl.py --blueprint {config.name}",
            "command": " ".join(base_cmd),
        }
    else:
        dispatch = {
            "provider": "kubernetes",
            "replicas": max(1, config.gpus),
            "resource_limits": {"nvidia.com/gpu": str(config.gpus)},
            "manifest_template": "deployment/k8s/rl-job.yaml",
            "command": " ".join(base_cmd),
        }

    return {
        "schema_version": 1,
        "plan_sha256_prefix": plan_sha,
        "workspace": str(ws),
        "blueprint": payload,
        "vram_estimate": estimate_chunked_vram_savings(
            chunk_size=config.chunked_loss_chunk_size
        ),
        "stages": [
            {
                "stage": "surgical_warm_start",
                "enabled": config.surgical_warm_start,
                "description": "Mine failing traces & bake Step-0 CAA steering vector before RL.",
            },
            {
                "stage": "disaggregated_chunked_rl",
                "enabled": True,
                "description": (
                    f"{config.algorithm.upper()} with chunk_size={config.chunked_loss_chunk_size}, "
                    f"dynamic_sampling={config.dynamic_sampling}, "
                    f"disaggregated={config.disaggregated_rollouts}."
                ),
            },
            {
                "stage": "online_hacking_guard",
                "enabled": config.online_hacking_guard,
                "description": "Periodic LLM-judge calibration + entropy/length collapse circuit-breaker.",
            },
            {
                "stage": "canary_and_live_lora_swap",
                "enabled": True,
                "description": "Shadow canary gate + zero-downtime vLLM LoRA hot-swap.",
            },
        ],
        "dispatch": dispatch,
    }


def run_surgical_warm_start(
    config: RLBlueprintConfig, *, workspace_root: Path
) -> dict[str, Any]:
    """Execute Step-0 Surgical Warm-Start via Weight Studio Auto-Heal before RL begins."""
    from stateset_agents.weight_studio import StudioRuntime

    rt = StudioRuntime(workspace_root=workspace_root)
    heal = rt.healer.run_healing_loop(
        config.warm_start_model_id,
        reward_fn=config.reward_fn,
        export_format="none",
    )
    return {
        "warm_started": True,
        "model_id": config.warm_start_model_id,
        "target_layer": (heal.get("steering_vector") or {}).get("target_layer"),
        "chosen_strength": (heal.get("steering_vector") or {}).get("chosen_strength"),
        "eval_verdict": (heal.get("eval_gate") or {}).get("verdict"),
    }


__all__ = [
    "RLBlueprintConfig",
    "build_cloud_launch_plan",
    "estimate_chunked_vram_savings",
    "run_surgical_warm_start",
]
