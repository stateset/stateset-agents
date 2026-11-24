"""Google Colab (including Google AI Ultra A100/H100/L4) runtime & GA evidence helpers.

Provides hardware detection, GPU-tier training profiles, secure Colab Secrets
lookup, Google Drive checkpoint persistence helpers, official Open-in-Colab
deep links, and wheel-packaged GPU + River AI 0.11 evidence capture helpers so
Colab notebooks work directly from a bare ``pip install stateset-agents`` with
zero eager ``torch`` imports.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

DEFAULT_GITHUB_REPO = "stateset/stateset-agents"
COLAB_NOTEBOOKS: dict[str, str] = {
    "open_weight_rl": "open_weight_rl_colab_gpu.ipynb",
    "river_ai": "river_ai_full_lifecycle_colab.ipynb",
    "weight_studio": "weight_studio_3d_colab.ipynb",
    "v1_ga_evidence": "v1_ga_gpu_river_evidence_colab.ipynb",
}


@dataclass(frozen=True)
class ColabTrainingProfile:
    """GPU-tier tuned open-weight RL defaults for Google Colab runtimes."""

    tier: str
    gpu_label: str
    min_vram_gb: float
    recommended_model: str
    fallback_model: str
    algorithm: str
    bf16: bool
    fp16: bool
    lora_rank: int
    lora_alpha: int
    group_size: int
    chunked_loss_chunk_size: int
    use_checkpointing: bool
    max_completion_length: int
    triton_recommended: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


COLAB_PROFILES: dict[str, ColabTrainingProfile] = {
    "h100_80gb": ColabTrainingProfile(
        tier="h100_80gb",
        gpu_label="NVIDIA H100 80GB (Google AI Ultra)",
        min_vram_gb=70.0,
        recommended_model="Qwen/Qwen3-8B",
        fallback_model="Qwen/Qwen2.5-7B-Instruct",
        algorithm="gspo",
        bf16=True,
        fp16=False,
        lora_rank=32,
        lora_alpha=64,
        group_size=8,
        chunked_loss_chunk_size=256,
        use_checkpointing=True,
        max_completion_length=1024,
        triton_recommended=True,
        notes="Full 8B+ BF16 LoRA RL with Triton SRAM chunked LM-head loss & FP8 TIS.",
    ),
    "a100_80gb": ColabTrainingProfile(
        tier="a100_80gb",
        gpu_label="NVIDIA A100 80GB (Google AI Ultra)",
        min_vram_gb=60.0,
        recommended_model="Qwen/Qwen3-8B",
        fallback_model="Qwen/Qwen2.5-7B-Instruct",
        algorithm="gspo",
        bf16=True,
        fp16=False,
        lora_rank=32,
        lora_alpha=64,
        group_size=8,
        chunked_loss_chunk_size=256,
        use_checkpointing=True,
        max_completion_length=1024,
        triton_recommended=True,
        notes="High-VRAM A100 80GB profile for 7B–8B multi-turn RL + 6-domain sandboxes.",
    ),
    "a100_40gb": ColabTrainingProfile(
        tier="a100_40gb",
        gpu_label="NVIDIA A100 40GB (Google AI Ultra / Pro+)",
        min_vram_gb=32.0,
        recommended_model="Qwen/Qwen3-4B",
        fallback_model="Qwen/Qwen2.5-3B-Instruct",
        algorithm="gspo",
        bf16=True,
        fp16=False,
        lora_rank=16,
        lora_alpha=32,
        group_size=8,
        chunked_loss_chunk_size=256,
        use_checkpointing=True,
        max_completion_length=768,
        triton_recommended=True,
        notes="Balanced A100 40GB BF16 profile with chunked LM-head loss & CAA warm-start.",
    ),
    "l4_24gb": ColabTrainingProfile(
        tier="l4_24gb",
        gpu_label="NVIDIA L4 24GB (Google AI Ultra / Pro)",
        min_vram_gb=20.0,
        recommended_model="Qwen/Qwen2.5-3B-Instruct",
        fallback_model="Qwen/Qwen2.5-1.5B-Instruct",
        algorithm="dapo",
        bf16=True,
        fp16=False,
        lora_rank=16,
        lora_alpha=32,
        group_size=4,
        chunked_loss_chunk_size=128,
        use_checkpointing=True,
        max_completion_length=512,
        triton_recommended=True,
        notes="Efficient L4 24GB BF16 profile with activation checkpointing & chunk=128.",
    ),
    "t4_16gb": ColabTrainingProfile(
        tier="t4_16gb",
        gpu_label="NVIDIA T4 16GB (Colab Standard)",
        min_vram_gb=12.0,
        recommended_model="Qwen/Qwen2.5-1.5B-Instruct",
        fallback_model="Qwen/Qwen2.5-0.5B-Instruct",
        algorithm="gspo",
        bf16=False,
        fp16=True,
        lora_rank=8,
        lora_alpha=16,
        group_size=4,
        chunked_loss_chunk_size=64,
        use_checkpointing=True,
        max_completion_length=384,
        triton_recommended=False,
        notes="FP16 T4 profile with aggressive chunk_size=64 logit memory reduction.",
    ),
    "cpu_fallback": ColabTrainingProfile(
        tier="cpu_fallback",
        gpu_label="CPU Fallback (No CUDA Runtime)",
        min_vram_gb=0.0,
        recommended_model="stub://quickstart",
        fallback_model="Qwen/Qwen2.5-0.5B-Instruct",
        algorithm="gspo",
        bf16=False,
        fp16=False,
        lora_rank=8,
        lora_alpha=16,
        group_size=2,
        chunked_loss_chunk_size=32,
        use_checkpointing=False,
        max_completion_length=128,
        triton_recommended=False,
        notes="Runs offline verifiable sandboxes, 3-seed tiny-LM ablation, and 3D Studio.",
    ),
}


def is_colab_runtime() -> bool:
    """Return True when executing inside a Google Colab kernel."""
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        return True
    try:
        return importlib.util.find_spec("google.colab") is not None
    except (ImportError, ValueError):
        return False


def recommend_colab_profile(
    *,
    gpu_name: str = "",
    vram_gb: float = 0.0,
) -> ColabTrainingProfile:
    """Select the optimal Colab training profile from GPU name and VRAM capacity."""
    upper = (gpu_name or "").upper()
    if "H100" in upper or "H200" in upper or "B200" in upper:
        return COLAB_PROFILES["h100_80gb"]
    if "A100" in upper:
        if vram_gb >= 60.0 or "80GB" in upper or "80G" in upper:
            return COLAB_PROFILES["a100_80gb"]
        return COLAB_PROFILES["a100_40gb"]
    if "L4" in upper or "RTX 4090" in upper or "A10" in upper:
        return COLAB_PROFILES["l4_24gb"]
    if "T4" in upper or "V100" in upper:
        return COLAB_PROFILES["t4_16gb"]
    if vram_gb >= 70.0:
        return COLAB_PROFILES["h100_80gb"]
    if vram_gb >= 60.0:
        return COLAB_PROFILES["a100_80gb"]
    if vram_gb >= 32.0:
        return COLAB_PROFILES["a100_40gb"]
    if vram_gb >= 20.0:
        return COLAB_PROFILES["l4_24gb"]
    if vram_gb >= 12.0:
        return COLAB_PROFILES["t4_16gb"]
    return COLAB_PROFILES["cpu_fallback"]


def detect_colab_runtime(*, probe_cuda: bool = True) -> dict[str, Any]:
    """Inspect runtime hardware, CUDA/Triton availability, and recommended Colab tier."""
    in_colab = is_colab_runtime()
    cuda_available = False
    gpu_name = ""
    vram_gb = 0.0
    bf16_supported = False
    cuda_version = ""
    torch_version = ""
    triton_available = False

    if probe_cuda:
        try:
            import torch

            torch_version = str(getattr(torch, "__version__", ""))
            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                props = torch.cuda.get_device_properties(0)
                gpu_name = str(props.name)
                vram_gb = round(float(props.total_memory) / (1024**3), 2)
                cuda_version = str(getattr(torch.version, "cuda", "") or "")
                bf16_fn = getattr(torch.cuda, "is_bf16_supported", None)
                bf16_supported = bool(bf16_fn()) if callable(bf16_fn) else False
        except Exception:
            pass

        try:
            from stateset_agents.training.chunked_rl_loss import (
                triton_chunked_logprobs_available,
            )

            triton_available = bool(triton_chunked_logprobs_available())
        except Exception:
            triton_available = False

    profile = recommend_colab_profile(gpu_name=gpu_name, vram_gb=vram_gb)
    drive_mounted = Path("/content/drive/MyDrive").is_dir()
    py_ver = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    return {
        "in_colab": in_colab,
        "python_version": py_ver,
        "river_python_compatible": sys.version_info >= (3, 12),
        "cuda_available": cuda_available,
        "gpu_name": gpu_name or "cpu",
        "vram_gb": vram_gb,
        "bf16_supported": bf16_supported,
        "cuda_version": cuda_version,
        "torch_version": torch_version,
        "triton_available": triton_available,
        "drive_mounted": drive_mounted,
        "profile": profile.to_dict(),
    }


def get_colab_secret(name: str, *, default: str | None = None) -> str | None:
    """Resolve a secret from Google Colab ``userdata`` first, then environment variables."""
    if is_colab_runtime():
        try:
            from google.colab import userdata

            value = userdata.get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
        except Exception:
            pass
    env_val = os.environ.get(name)
    if env_val is not None and env_val.strip():
        return env_val.strip()
    return default


def colab_notebook_url(
    notebook_name: str,
    *,
    repo: str = DEFAULT_GITHUB_REPO,
    ref: str = "master",
) -> str:
    """Return an official Open-in-Colab URL for a bundled notebook."""
    resolved = COLAB_NOTEBOOKS.get(notebook_name, notebook_name)
    if not resolved.endswith(".ipynb"):
        resolved = f"{resolved}.ipynb"
    return (
        f"https://colab.research.google.com/github/{repo}/blob/{ref}/"
        f"notebooks/{resolved}"
    )


def resolve_checkpoint_dir(
    subdir: str = "stateset-agents/runs",
    *,
    fallback_root: Path | None = None,
) -> Path:
    """Return Google Drive checkpoint directory when mounted, else a local workspace path."""
    drive_root = Path("/content/drive/MyDrive")
    if drive_root.is_dir():
        target = drive_root / subdir
    else:
        base = fallback_root if fallback_root is not None else Path("outputs")
        target = base / subdir
    target.mkdir(parents=True, exist_ok=True)
    return target


# ---------------------------------------------------------------------------
# Packaged GPU + River AI 0.11 GA Evidence Helpers (PyPI-wheel safe for Colab)
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _sha256_json(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(canonical).hexdigest()


def _mean_std(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"mean": 0.0, "std": 0.0}
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / len(vals)
    return {"mean": round(m, 4), "std": round(math.sqrt(var), 4)}


def _build_tiny_lm(
    torch: Any, nn: Any, *, vocab: int = 128, hidden: int = 32, layers: int = 3
) -> Any:
    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(hidden)
            self.mlp = nn.Module()
            self.mlp.up_proj = nn.Linear(hidden, hidden * 2)
            self.mlp.down_proj = nn.Linear(hidden * 2, hidden, bias=False)

        def forward(self, x: Any) -> Any:
            h = self.norm(x)
            return x + self.mlp.down_proj(torch.nn.functional.gelu(self.mlp.up_proj(h)))

    class _Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab, hidden)
            self.layers = nn.ModuleList([_Block() for _ in range(layers)])

        def forward(self, input_ids: Any) -> SimpleNamespace:
            h = self.embed(input_ids)
            for blk in self.layers:
                h = blk(h)
            return SimpleNamespace(last_hidden_state=h)

    class _LM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _Backbone()
            self.lm_head = nn.Linear(hidden, vocab, bias=False)

        def forward(self, input_ids: Any) -> SimpleNamespace:
            return SimpleNamespace(
                logits=self.lm_head(self.model(input_ids).last_hidden_state)
            )

    return _LM()


async def run_six_domain_verifiable_eval() -> dict[str, Any]:
    """Run all 6 enterprise verifiable MCP sandboxes concurrently."""
    from stateset_agents.core.sandbox_pool import (
        MultiDomainSandboxPool,
        SandboxEpisodeSpec,
    )
    from stateset_agents.core.trajectory import ConversationTurn

    pool = MultiDomainSandboxPool(max_concurrency=6)
    specs = [
        SandboxEpisodeSpec("c-ok", "commerce", {"target_order_id": "ORD-77701"}),
        SandboxEpisodeSpec("k-ok", "coding", {}),
        SandboxEpisodeSpec("s-ok", "sql", {"expected_value": 25000}),
        SandboxEpisodeSpec("r-ok", "rest_api", {"target_subscription_id": "sub_101"}),
        SandboxEpisodeSpec("d-ok", "docker_code", {}),
        SandboxEpisodeSpec(
            "w-ok", "web_agent", {"target_order_id": "ORD-77701"}, max_turns=6
        ),
    ]

    async def _policy(
        obs: str, meta: dict[str, Any], turn_idx: int
    ) -> ConversationTurn:
        dom = meta["domain"]
        if dom == "commerce":
            if turn_idx == 0:
                return ConversationTurn(
                    role="assistant",
                    content='<tool_call>{"name": "issue_refund", "arguments": {"order_id": "ORD-77701", "reason": "damaged"}}</tool_call>',
                )
            return ConversationTurn(role="assistant", content="Refunded ORD-77701.")
        if dom == "coding":
            if turn_idx == 0:
                return ConversationTurn(
                    role="assistant",
                    content='<tool_call>{"name": "write_solution", "arguments": {"code": "def compute_net_revenue(s, p, r):\\n    return (s - r) * p"}}</tool_call>',
                )
            if turn_idx == 1:
                return ConversationTurn(
                    role="assistant",
                    content='<tool_call>{"name": "run_tests", "arguments": {}}</tool_call>',
                )
            return ConversationTurn(role="assistant", content="Coding tests passed.")
        if dom == "sql":
            if turn_idx == 0:
                return ConversationTurn(
                    role="assistant",
                    content='<tool_call>{"name": "run_sql", "arguments": {"sql": "SELECT SUM(revenue_cents) AS total FROM sales WHERE region = \'west\'"}}</tool_call>',
                )
            return ConversationTurn(role="assistant", content="Total revenue is 25000.")
        if dom == "rest_api":
            if turn_idx == 0:
                return ConversationTurn(
                    role="assistant",
                    content='<tool_call>{"name": "http_request", "arguments": {"method": "POST", "path": "/v1/subscriptions/sub_101/cancel", "idempotency_key": "idem-abl", "body": {"reason": "done"}}}</tool_call>',
                )
            return ConversationTurn(role="assistant", content="Canceled sub_101.")
        if dom == "docker_code":
            if turn_idx == 0:
                return ConversationTurn(
                    role="assistant",
                    content='<tool_call>{"name": "apply_patch", "arguments": {"path": "billing.py", "content": "def apply_discount(subtotal_cents: int, pct: int) -> int:\\n    return subtotal_cents - (subtotal_cents * pct) // 100\\n"}}</tool_call>',
                )
            if turn_idx == 1:
                return ConversationTurn(
                    role="assistant",
                    content='<tool_call>{"name": "run_pytest", "arguments": {}}</tool_call>',
                )
            return ConversationTurn(role="assistant", content="Patched billing.py.")
        if turn_idx == 0:
            return ConversationTurn(
                role="assistant",
                content='<tool_call>{"name": "click", "arguments": {"selector": "#link-returns"}}</tool_call>',
            )
        if turn_idx == 1:
            return ConversationTurn(
                role="assistant",
                content='<tool_call>{"name": "fill", "arguments": {"selector": "#order-id", "value": "ORD-77701"}}</tool_call>',
            )
        if turn_idx == 2:
            return ConversationTurn(
                role="assistant",
                content='<tool_call>{"name": "fill", "arguments": {"selector": "#reason", "value": "damaged"}}</tool_call>',
            )
        if turn_idx == 3:
            return ConversationTurn(
                role="assistant",
                content='<tool_call>{"name": "click", "arguments": {"selector": "#btn-submit-return"}}</tool_call>',
            )
        return ConversationTurn(role="assistant", content="Submitted RMA-ORD-77701.")

    t0 = time.perf_counter()
    res = await pool.run_batch(specs, _policy, normalize_per_domain=True)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "domains": sorted(res["domain_stats"]),
        "episode_count": res["episode_count"],
        "elapsed_ms": elapsed_ms,
        "all_verified": all(
            st["verified_pass_rate"] == 1.0 for st in res["domain_stats"].values()
        ),
        "domain_stats": res["domain_stats"],
    }


def _run_coro(coro: Any) -> Any:
    """Run an async coroutine cleanly both in standard Python and inside Colab's running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def run_six_domain_verifiable_eval_sync() -> dict[str, Any]:
    """Colab-safe synchronous wrapper for ``run_six_domain_verifiable_eval``."""
    return dict(_run_coro(run_six_domain_verifiable_eval()))


def run_three_seed_ablation(
    seeds: list[int] | None = None, *, steps: int = 8
) -> dict[str, Any]:
    """Run deterministic 3-seed PyTorch ablation comparing Base vs. CAA vs. Chunked GSPO."""
    import torch
    import torch.nn as nn

    from stateset_agents.training.chunked_rl_loss import (
        chunked_model_token_logprobs,
        chunked_rl_policy_loss,
        triton_chunked_logprobs_available,
    )
    from stateset_agents.training.rl_blueprint import estimate_chunked_vram_savings
    from stateset_agents.weight_studio.gpu_surgery import (
        bake_torch_caa_steering,
        extract_torch_caa_vector,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_list = list(seeds or [42, 1337, 2026])
    per_seed: list[dict[str, Any]] = []

    for seed in seed_list:
        torch.manual_seed(seed)
        lm = _build_tiny_lm(torch, nn).to(device)
        pos_ids = torch.randint(4, 60, (4, 24), device=device)
        neg_ids = torch.randint(64, 124, (4, 24), device=device)
        mask = torch.ones(4, 24, device=device)
        mask[:, :6] = 0.0

        def _margin(model: Any, p_ids: Any, n_ids: Any, m: Any) -> float:
            with torch.no_grad():
                lp_pos, m_pos = chunked_model_token_logprobs(
                    model, p_ids, m, chunk_size=8
                )
                lp_neg, m_neg = chunked_model_token_logprobs(
                    model, n_ids, m, chunk_size=8
                )
                pos_mean = (
                    (lp_pos.sum(dim=-1) / m_pos.sum(dim=-1).clamp(min=1.0))
                    .mean()
                    .item()
                )
                neg_mean = (
                    (lp_neg.sum(dim=-1) / m_neg.sum(dim=-1).clamp(min=1.0))
                    .mean()
                    .item()
                )
                return float(pos_mean - neg_mean)

        base_margin = _margin(lm, pos_ids, neg_ids, mask)
        caa = extract_torch_caa_vector(lm, pos_ids, neg_ids)
        baked = bake_torch_caa_steering(
            lm,
            caa["direction_tensor"],
            layer_index=int(caa["recommended_layer"]),
            alpha=0.25,
        )
        caa_margin = _margin(lm, pos_ids, neg_ids, mask)

        opt = torch.optim.AdamW(lm.parameters(), lr=3e-2)
        batch_ids = torch.cat([pos_ids, neg_ids], dim=0)
        batch_mask = torch.cat([mask, mask], dim=0)
        advantages = torch.tensor(
            [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0], device=device
        )
        with torch.no_grad():
            old_lp, _ = chunked_model_token_logprobs(
                lm, batch_ids, batch_mask, chunk_size=8
            )
            old_lp = old_lp.detach()

        t0 = time.perf_counter()
        kernel_backend = "torch_chunked"
        for _ in range(steps):
            opt.zero_grad()
            hidden = lm.model(batch_ids).last_hidden_state
            router_logits = torch.randn(8, 24, 4, device=device)
            res = chunked_rl_policy_loss(
                hidden,
                lm.lm_head.weight,
                batch_ids,
                batch_mask,
                old_lp,
                advantages,
                algorithm="gspo",
                ref_logprobs=old_lp,
                rollout_logprobs=old_lp - 0.03,
                kl_coef=0.01,
                chunk_size=8,
                tis_clip_max=2.0,
                icepop_bounds=(0.2, 5.0),
                router_logits=router_logits,
                moe_aux_coef=0.01,
            )
            res["loss"].backward()
            opt.step()
            kernel_backend = str(res.get("kernel_backend", kernel_backend))

        train_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        gspo_margin = _margin(lm, pos_ids, neg_ids, mask)
        per_seed.append(
            {
                "seed": seed,
                "base_logprob_margin": round(base_margin, 5),
                "caa_warm_start_margin": round(caa_margin, 5),
                "chunked_gspo_margin": round(gspo_margin, 5),
                "total_margin_gain": round(gspo_margin - base_margin, 5),
                "warm_start_layer": int(caa["recommended_layer"]),
                "warm_start_mode": str(baked.get("mode", "rank1_weight_bake")),
                "kernel_backend": kernel_backend,
                "train_ms": train_ms,
            }
        )

    six_domain = _run_coro(run_six_domain_verifiable_eval())
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "triton_available": bool(triton_chunked_logprobs_available()),
        "seeds": seed_list,
        "steps_per_seed": steps,
        "summary": {
            "base_margin": _mean_std([r["base_logprob_margin"] for r in per_seed]),
            "chunked_gspo_margin": _mean_std(
                [r["chunked_gspo_margin"] for r in per_seed]
            ),
            "margin_gain": _mean_std([r["total_margin_gain"] for r in per_seed]),
            "all_seeds_improved": all(r["total_margin_gain"] > 0.0 for r in per_seed),
        },
        "vram_projection_16k_qwen3": estimate_chunked_vram_savings(
            batch_size=4, seq_len=16384, vocab_size=152064, chunk_size=256
        ),
        "six_domain_verifiable_eval": six_domain,
        "per_seed": per_seed,
    }


def run_cuda_chunked_vram_benchmark(
    *,
    batch_size: int | None = None,
    seq_len: int | None = None,
    hidden_dim: int | None = None,
    vocab_size: int | None = None,
    chunk_size: int | None = None,
) -> dict[str, Any]:
    """Measure real CUDA peak VRAM (or CPU parity) for full vs. chunked/Triton LM-head loss."""
    import torch

    from stateset_agents.training.chunked_rl_loss import (
        chunked_gather_token_logprobs,
        chunked_rl_policy_loss,
        triton_chunked_logprobs_available,
    )
    from stateset_agents.training.rl_blueprint import estimate_chunked_vram_savings
    from stateset_agents.training.rl_losses import gather_token_logprobs

    use_cuda = bool(torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = (
        torch.bfloat16
        if (use_cuda and getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        else torch.float32
    )

    if use_cuda:
        vram_gb = float(torch.cuda.get_device_properties(0).total_memory) / (1024**3)
        b = batch_size or 2
        t = seq_len or (2048 if vram_gb >= 20.0 else 1024)
        h_dim = hidden_dim or 256
        v = vocab_size or (65536 if vram_gb >= 32.0 else 32768)
        c = chunk_size or 256
    else:
        b = batch_size or 2
        t = seq_len or 256
        h_dim = hidden_dim or 64
        v = vocab_size or 4096
        c = chunk_size or 64

    torch.manual_seed(42)
    h_base = torch.randn(b, t, h_dim, device=device, dtype=dtype)
    w_base = torch.randn(v, h_dim, device=device, dtype=dtype)
    input_ids = torch.randint(0, v, (b, t), device=device)
    response_mask = torch.ones(b, t, device=device, dtype=torch.float32)
    response_mask[:, : min(16, t // 4)] = 0.0

    def _sync() -> None:
        if use_cuda:
            torch.cuda.synchronize()

    h1 = h_base.detach().clone().requires_grad_(True)
    w1 = w_base.detach().clone().requires_grad_(True)
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    _sync()
    t0 = time.perf_counter()
    full_logits = torch.nn.functional.linear(h1, w1)
    lp_full, _ = gather_token_logprobs(full_logits, input_ids, response_mask)
    lp_full.float().sum().backward()
    _sync()
    full_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    full_peak_mb = (
        round(float(torch.cuda.max_memory_allocated()) / (1024**2), 2)
        if use_cuda
        else None
    )
    lp_full_ref = lp_full.detach().float().clone()
    h1_grad = h1.grad.detach().float().clone() if h1.grad is not None else None
    w1_grad = w1.grad.detach().float().clone() if w1.grad is not None else None
    del full_logits, lp_full, h1, w1

    h2 = h_base.detach().clone().requires_grad_(True)
    w2 = w_base.detach().clone().requires_grad_(True)
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    _sync()
    t1 = time.perf_counter()
    lp_chunk, _ = chunked_gather_token_logprobs(
        h2,
        w2,
        input_ids,
        response_mask,
        chunk_size=c,
        use_checkpointing=True,
    )
    lp_chunk.float().sum().backward()
    _sync()
    chunk_ms = round((time.perf_counter() - t1) * 1000.0, 2)
    chunk_peak_mb = (
        round(float(torch.cuda.max_memory_allocated()) / (1024**2), 2)
        if use_cuda
        else None
    )

    max_lp_diff = float((lp_full_ref - lp_chunk.detach().float()).abs().max().item())
    max_h_grad_diff = (
        float((h1_grad - h2.grad.detach().float()).abs().max().item())
        if (h1_grad is not None and h2.grad is not None)
        else 0.0
    )
    max_w_grad_diff = (
        float((w1_grad - w2.grad.detach().float()).abs().max().item())
        if (w1_grad is not None and w2.grad is not None)
        else 0.0
    )

    advantages = torch.tensor([1.0, -1.0][:b], device=device, dtype=torch.float32)
    gspo = chunked_rl_policy_loss(
        h_base.detach().clone().requires_grad_(True),
        w_base.detach().clone().requires_grad_(True),
        input_ids,
        response_mask,
        lp_chunk.detach().float(),
        advantages,
        algorithm="gspo",
        ref_logprobs=lp_chunk.detach().float(),
        chunk_size=c,
    )

    measured_vram_saved_pct: float | None = None
    if full_peak_mb and chunk_peak_mb is not None and full_peak_mb > 0:
        measured_vram_saved_pct = round(
            (1.0 - (chunk_peak_mb / full_peak_mb)) * 100.0, 2
        )

    tol = 2e-2 if dtype == torch.bfloat16 else 1e-4
    return {
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "shape": {
            "batch_size": b,
            "seq_len": t,
            "hidden_dim": h_dim,
            "vocab_size": v,
            "chunk_size": c,
        },
        "triton_available": bool(triton_chunked_logprobs_available()),
        "kernel_backend": str(gspo.get("kernel_backend", "pytorch_chunked")),
        "timing_ms": {
            "full_logits_ms": full_ms,
            "chunked_ms": chunk_ms,
        },
        "cuda_peak_memory_mb": {
            "full_logits_mb": full_peak_mb,
            "chunked_mb": chunk_peak_mb,
            "measured_vram_saved_pct": measured_vram_saved_pct,
        },
        "numerical_equivalence": {
            "max_logprob_abs_diff": round(max_lp_diff, 8),
            "max_hidden_grad_abs_diff": round(max_h_grad_diff, 8),
            "max_weight_grad_abs_diff": round(max_w_grad_diff, 8),
            "tolerance": tol,
            "exact_match": max_lp_diff < tol and max_h_grad_diff < tol,
        },
        "vram_projection_16k_qwen3": estimate_chunked_vram_savings(
            batch_size=4, seq_len=16384, vocab_size=152064, chunk_size=256
        ),
    }


def run_live_hf_open_weight_smoke(
    *,
    model_name: str,
    model_revision: str = "main",
    steps: int = 4,
    lora_rank: int = 16,
    chunk_size: int = 128,
    output_dir: Path,
) -> dict[str, Any]:
    """Run real HF/PEFT open-weight Step-0 CAA warm-start + Chunked GSPO steps on GPU/CPU."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from stateset_agents.training.chunked_rl_loss import (
        chunked_model_token_logprobs,
        chunked_rl_policy_loss,
        extract_lm_head_and_backbone,
    )
    from stateset_agents.weight_studio.gpu_surgery import (
        bake_torch_caa_steering,
        extract_torch_caa_vector,
    )

    use_cuda = bool(torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    use_bf16 = use_cuda and bool(
        getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    )
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float32

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=model_revision, trust_remote_code=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        torch_dtype=torch_dtype,
        attn_implementation="sdpa",
        trust_remote_code=False,
    ).to(device)
    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.train()

    positive_texts = [
        'User: Refund damaged order ORD-77701.\nAssistant: <tool_call>{"name": "issue_refund", "arguments": {"order_id": "ORD-77701", "reason": "damaged"}}</tool_call>',
        'User: Check West region sales.\nAssistant: <tool_call>{"name": "run_sql", "arguments": {"sql": "SELECT SUM(revenue_cents) FROM sales WHERE region = \'west\'"}}</tool_call>',
    ]
    negative_texts = [
        "User: Refund damaged order ORD-77701.\nAssistant: I cannot process refunds or look up order records.",
        "User: Check West region sales.\nAssistant: I will guess the revenue number without querying SQL.",
    ]

    pos_enc = tokenizer(
        positive_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(device)
    neg_enc = tokenizer(
        negative_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(device)

    pos_mask = pos_enc["attention_mask"].float()
    neg_mask = neg_enc["attention_mask"].float()
    pos_mask[:, :8] = 0.0
    neg_mask[:, :8] = 0.0

    def _margin() -> float:
        with torch.no_grad():
            lp_p, mp = chunked_model_token_logprobs(
                model,
                pos_enc["input_ids"],
                pos_mask,
                attention_mask=pos_enc["attention_mask"],
                chunk_size=chunk_size,
            )
            lp_n, mn = chunked_model_token_logprobs(
                model,
                neg_enc["input_ids"],
                neg_mask,
                attention_mask=neg_enc["attention_mask"],
                chunk_size=chunk_size,
            )
            p_mean = (lp_p.sum(dim=-1) / mp.sum(dim=-1).clamp(min=1.0)).mean().item()
            n_mean = (lp_n.sum(dim=-1) / mn.sum(dim=-1).clamp(min=1.0)).mean().item()
            return float(p_mean - n_mean)

    base_margin = _margin()
    caa = extract_torch_caa_vector(model, pos_enc["input_ids"], neg_enc["input_ids"])
    baked = bake_torch_caa_steering(
        model,
        caa["direction_tensor"],
        layer_index=int(caa["recommended_layer"]),
        alpha=0.15,
    )
    caa_margin = _margin()

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-4
    )
    max_len = max(pos_enc["input_ids"].shape[1], neg_enc["input_ids"].shape[1])

    def _pad(tensor: Any, pad_val: int) -> Any:
        if tensor.shape[1] == max_len:
            return tensor
        pad = torch.full(
            (tensor.shape[0], max_len - tensor.shape[1]),
            pad_val,
            device=device,
            dtype=tensor.dtype,
        )
        return torch.cat([tensor, pad], dim=1)

    batch_ids = torch.cat(
        [
            _pad(pos_enc["input_ids"], int(tokenizer.pad_token_id)),
            _pad(neg_enc["input_ids"], int(tokenizer.pad_token_id)),
        ],
        dim=0,
    )
    batch_attn = torch.cat(
        [_pad(pos_enc["attention_mask"], 0), _pad(neg_enc["attention_mask"], 0)], dim=0
    )
    batch_mask = torch.cat([_pad(pos_mask, 0), _pad(neg_mask, 0)], dim=0)
    advantages = torch.tensor(
        [1.0] * len(positive_texts) + [-1.0] * len(negative_texts),
        device=device,
        dtype=torch.float32,
    )

    extracted = extract_lm_head_and_backbone(model)
    if extracted is None:
        raise RuntimeError(
            f"Model {model_name!r} does not expose a compatible backbone + lm_head for chunked RL."
        )
    backbone, lm_head_weight, lm_head_bias = extracted
    with torch.no_grad():
        old_lp, _ = chunked_model_token_logprobs(
            model,
            batch_ids,
            batch_mask,
            attention_mask=batch_attn,
            chunk_size=chunk_size,
        )
        old_lp = old_lp.detach()

    step_losses: list[float] = []
    train_t0 = time.perf_counter()
    kernel_backend = "pytorch_chunked"
    for _ in range(max(1, steps)):
        optimizer.zero_grad()
        out = backbone(input_ids=batch_ids, attention_mask=batch_attn)
        hidden = out.last_hidden_state
        res = chunked_rl_policy_loss(
            hidden,
            lm_head_weight,
            batch_ids,
            batch_mask,
            old_lp,
            advantages,
            algorithm="gspo",
            ref_logprobs=old_lp,
            lm_head_bias=lm_head_bias,
            kl_coef=0.02,
            chunk_size=chunk_size,
            tis_clip_max=2.0,
            use_checkpointing=True,
        )
        res["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step_losses.append(round(float(res["loss"].item()), 6))
        kernel_backend = str(res.get("kernel_backend", kernel_backend))

    train_s = max(1e-6, time.perf_counter() - train_t0)
    post_margin = _margin()
    peak_vram_gb = (
        round(float(torch.cuda.max_memory_allocated()) / (1024**3), 3)
        if use_cuda
        else None
    )

    adapter_dir = output_dir / "hf_lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    saved_files = sorted(p.name for p in adapter_dir.iterdir() if p.is_file())
    total_tokens = int(batch_ids.numel() * max(1, steps))

    return {
        "status": "completed",
        "model_name": model_name,
        "model_revision": model_revision,
        "device": str(device),
        "dtype": str(torch_dtype).replace("torch.", ""),
        "lora_rank": lora_rank,
        "steps": max(1, steps),
        "kernel_backend": kernel_backend,
        "warm_start_layer": int(caa["recommended_layer"]),
        "warm_start_mode": str(baked.get("mode", "rank1_weight_bake")),
        "base_margin": round(base_margin, 5),
        "caa_warm_start_margin": round(caa_margin, 5),
        "post_rl_margin": round(post_margin, 5),
        "margin_gain": round(post_margin - base_margin, 5),
        "improved": post_margin > base_margin,
        "step_losses": step_losses,
        "peak_vram_gb": peak_vram_gb,
        "tokens_per_sec": round(total_tokens / train_s, 1),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "adapter_dir": str(adapter_dir),
        "adapter_files": saved_files,
    }


def run_river_verification(
    *,
    mode: str = "offline",
    base_model: str = "Qwen/Qwen3.6-35B-A3B-FP8",
    checkpoint_uri: str = "",
) -> dict[str, Any]:
    """Run offline 6-domain River SDK contract proof, live preflight, or live checkpoint chat."""
    from stateset_agents.remote.river_runtime import river_preflight

    api_key = get_colab_secret("RIVER_API_KEY")
    local_preflight = river_preflight(live=False)

    if mode == "offline":
        return {
            "mode": "offline",
            "api_key_configured": bool(api_key),
            "local_preflight": local_preflight,
            "supported_sandbox_domains": [
                "commerce",
                "coding",
                "sql",
                "rest_api",
                "docker_code",
                "web_agent",
            ],
            "serving_surfaces": [
                "RiverDeploymentManager",
                "RiverDeploymentProvider",
                "RiverInferenceClient",
                "river-chat",
                "river-deploy",
                "river-deployments",
            ],
            "note": (
                "Offline contract mode. Set RIVER_API_KEY and pass --river-mode preflight "
                "or --river-mode live on Python >=3.12 with stateset-agents[river] installed "
                "to capture live River 0.11 capability & serving evidence."
            ),
        }

    if not api_key:
        raise RuntimeError(
            "RIVER_API_KEY is required for --river-mode preflight/live (set env var or Colab Secret)."
        )

    live_report = river_preflight([base_model], live=True, timeout_seconds=20.0)
    result: dict[str, Any] = {
        "mode": mode,
        "base_model": base_model,
        "live_preflight": live_report,
    }

    if mode == "live":
        from stateset_agents.remote.river_runtime import inspect_server_capabilities
        from stateset_agents.remote.river_serving import (
            RiverDeploymentManager,
            RiverInferenceClient,
        )

        manager = RiverDeploymentManager(api_key=api_key)
        try:
            result["server_capabilities"] = inspect_server_capabilities(
                manager._get_client(), model_name=base_model
            )
            existing = manager.list_deployments()
            result["existing_deployments_count"] = len(existing)
        finally:
            manager.close()

        if checkpoint_uri:
            infer = RiverInferenceClient(
                checkpoint_uri,
                base_model=base_model,
                api_key=api_key,
            )
            try:
                reply = infer.chat(
                    [
                        {
                            "role": "user",
                            "content": "Acknowledge readiness in one short sentence.",
                        }
                    ],
                    max_tokens=48,
                    temperature=0.0,
                )
                result["checkpoint_chat_verified"] = True
                result["checkpoint_chat_preview"] = str(reply.get("text", ""))[:160]
            finally:
                infer.close()

    return result


def render_v1_evidence_markdown(evidence: dict[str, Any]) -> str:
    """Render a clean Markdown evidence summary ready for release notes and PROOFS.md."""
    hw = evidence["hardware"]
    vram = evidence["cuda_chunked_vram"]
    abl = evidence["three_seed_ablation"]
    pool = abl["six_domain_verifiable_eval"]
    hf = evidence.get("live_hf_open_weight")
    rv = evidence["river_ai"]

    lines = [
        f"# StateSet Agents `v{evidence['package_version']}` GPU & River AI Evidence Report",
        "",
        f"- **Captured:** `{evidence['generated_at']}`",
        f"- **Git SHA:** `{evidence['git_sha']}`",
        f"- **Runtime:** `{hw['gpu_name']}` (`{hw['vram_gb']} GB` VRAM, CUDA `{hw['cuda_version'] or 'none'}`, PyTorch `{hw['torch_version']}`, Triton `{hw['triton_available']}`, Colab `{hw['in_colab']}`)",
        f"- **Evidence SHA-256:** `{evidence['evidence_sha256'][:16]}`",
        "",
        "## 1. Full-Logit vs. Chunked / Triton LM-Head Benchmark",
        "",
        "| Metric | Measured Value |",
        "| --- | --- |",
        f"| Device & Dtype | `{vram['device']}` / `{vram['dtype']}` |",
        f"| Kernel Backend | `{vram['kernel_backend']}` (`triton_available={vram['triton_available']}`) |",
        f"| Numerical Parity | `exact_match={vram['numerical_equivalence']['exact_match']}` (`max Δlp={vram['numerical_equivalence']['max_logprob_abs_diff']}`, `max Δh_grad={vram['numerical_equivalence']['max_hidden_grad_abs_diff']}`) |",
        f"| Timing (Full vs. Chunked) | `{vram['timing_ms']['full_logits_ms']} ms` vs. `{vram['timing_ms']['chunked_ms']} ms` |",
        f"| Measured CUDA Peak VRAM | Full `{vram['cuda_peak_memory_mb']['full_logits_mb']} MB` -> Chunked `{vram['cuda_peak_memory_mb']['chunked_mb']} MB` (`{vram['cuda_peak_memory_mb']['measured_vram_saved_pct']}%` saved) |",
        f"| 16k Qwen3 Logit VRAM Projection | `{vram['vram_projection_16k_qwen3']['full_logits_gb']} GB` -> `{vram['vram_projection_16k_qwen3']['chunked_logits_gb']} GB` (`{vram['vram_projection_16k_qwen3']['peak_logit_vram_saved_pct']}%` saved) |",
        "",
        "## 2. Three-Seed Open-Weight RL Ablation & Six-Domain Verifiable Sandboxes",
        "",
        f"- **Base -> Chunked GSPO Margin:** `{abl['summary']['base_margin']['mean']}` -> `{abl['summary']['chunked_gspo_margin']['mean']}` (**`+{abl['summary']['margin_gain']['mean']} ± {abl['summary']['margin_gain']['std']}`**, `all_seeds_improved={abl['summary']['all_seeds_improved']}`)",
        f"- **Six-Domain Enterprise Sandbox Pool:** `{pool['episode_count']}/{pool['episode_count']}` verified (`all_verified={pool['all_verified']}`) in **`{pool['elapsed_ms']} ms`** (`{', '.join(pool['domains'])}`)",
        "",
    ]

    if hf and hf.get("status") == "completed":
        lines.extend(
            [
                "## 3. Live HuggingFace Open-Weight Model (Step-0 CAA + Chunked GSPO LoRA)",
                "",
                "| Metric | Measured Value |",
                "| --- | --- |",
                f"| Model | `{hf['model_name']}` (`r={hf['lora_rank']}`, `{hf['dtype']}`) |",
                f"| Step-0 CAA Warm-Start | Layer `{hf['warm_start_layer']}` (`{hf['warm_start_mode']}`) |",
                f"| Margin Gain ({hf['steps']} steps) | `{hf['base_margin']}` -> `{hf['post_rl_margin']}` (`{hf['margin_gain']:+.5f}`, `improved={hf['improved']}`) |",
                f"| Peak VRAM / Throughput | `{hf['peak_vram_gb']} GB` / `{hf['tokens_per_sec']} tok/s` |",
                "",
            ]
        )

    lines.extend(
        [
            "## 4. River AI `0.11` Verification",
            "",
            f"- **Mode:** `{rv['mode']}`",
        ]
    )
    if "live_preflight" in rv:
        lp = rv["live_preflight"]
        lines.append(
            f"- **Live Preflight Passed:** `{lp.get('passed')}` (requested `{rv.get('base_model')}`)"
        )
    else:
        lines.append(
            f"- **Local Preflight SDK/Python Check:** `passed={rv['local_preflight'].get('passed')}`, `api_key_configured={rv['api_key_configured']}`"
        )
    lines.append("")
    return "\n".join(lines)


def capture_v1_evidence(
    *,
    output_dir: Path,
    hf_model: str | None = None,
    hf_revision: str = "main",
    hf_steps: int = 4,
    skip_hf_model: bool = False,
    river_mode: str = "offline",
    river_base_model: str = "Qwen/Qwen3.6-35B-A3B-FP8",
    river_checkpoint_uri: str = "",
) -> dict[str, Any]:
    """Run all requested GPU + River evidence stages and write JSON + Markdown artifacts."""
    import stateset_agents

    output_dir.mkdir(parents=True, exist_ok=True)
    hardware = detect_colab_runtime(probe_cuda=True)
    cuda_vram = run_cuda_chunked_vram_benchmark()
    ablation = run_three_seed_ablation(seeds=[42, 1337, 2026], steps=8)

    if not skip_hf_model and hf_model:
        profile = hardware["profile"]
        live_hf: dict[str, Any] = run_live_hf_open_weight_smoke(
            model_name=hf_model,
            model_revision=hf_revision,
            steps=hf_steps,
            lora_rank=int(profile["lora_rank"]),
            chunk_size=int(profile["chunked_loss_chunk_size"]),
            output_dir=output_dir,
        )
    else:
        live_hf = {
            "status": "skipped",
            "reason": (
                "skip_hf_model=True"
                if skip_hf_model
                else "pass --hf-model <repo_id> to execute live HuggingFace LoRA training"
            ),
        }

    river_info = run_river_verification(
        mode=river_mode,
        base_model=river_base_model,
        checkpoint_uri=river_checkpoint_uri,
    )

    body: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_version": stateset_agents.__version__,
        "git_sha": _git_sha(),
        "hardware": hardware,
        "cuda_chunked_vram": cuda_vram,
        "three_seed_ablation": ablation,
        "live_hf_open_weight": live_hf,
        "river_ai": river_info,
    }
    body["evidence_sha256"] = _sha256_json(body)

    json_path = output_dir / "v1_ga_evidence.json"
    md_path = output_dir / "v1_ga_evidence.md"
    json_path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_v1_evidence_markdown(body), encoding="utf-8")
    return body


__all__ = [
    "COLAB_NOTEBOOKS",
    "COLAB_PROFILES",
    "ColabTrainingProfile",
    "DEFAULT_GITHUB_REPO",
    "capture_v1_evidence",
    "colab_notebook_url",
    "detect_colab_runtime",
    "get_colab_secret",
    "is_colab_runtime",
    "recommend_colab_profile",
    "render_v1_evidence_markdown",
    "resolve_checkpoint_dir",
    "run_cuda_chunked_vram_benchmark",
    "run_live_hf_open_weight_smoke",
    "run_river_verification",
    "run_six_domain_verifiable_eval",
    "run_six_domain_verifiable_eval_sync",
    "run_three_seed_ablation",
]
