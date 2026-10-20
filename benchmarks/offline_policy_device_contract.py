"""Verify real offline policy updates, complete resume and inference on one device.

Synthetic engineering evidence only. CUDA is required unless --device cpu is
explicitly selected for a local smoke test; failures never fall back to CPU.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from stateset_agents.data.conversation_dataset import (
    ConversationDataset,
    ConversationDatasetConfig,
)
from stateset_agents.evaluation.implementation import package_implementation
from stateset_agents.training.checkpoint_publication import (
    validate_checkpoint_publication,
)
from stateset_agents.training.offline_grpo_trainer import (
    OfflineGRPOConfig,
    OfflineGRPOTrainer,
)
from stateset_agents.training.promotion import artifact_digest
from stateset_agents.training.sampled_workflow import workflow_tokenizer


async def run_contract(output: Path, device: str) -> dict[str, Any]:
    """Exercise the actual packaged trainer and retain measured device evidence."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = workflow_tokenizer()
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    )

    class Features:
        def get_or_compute(self, key: str, text: str) -> torch.Tensor:
            ids = torch.tensor(tokenizer.encode(text, add_special_tokens=False))
            counts = torch.bincount(ids, minlength=len(tokenizer)).float()
            return counts / counts.sum().clamp_min(1)

    dataset = ConversationDataset.from_dict(
        {
            "trajectories": [
                {
                    "trajectory_id": name,
                    "turns": [
                        {"role": "user", "content": "request"},
                        {"role": "assistant", "content": "lookup"},
                        {"role": "tool", "content": name},
                        {"role": "assistant", "content": action, "reward": 1},
                    ],
                }
                for name, action in [("eligible", "refund"), ("ineligible", "deny")]
            ]
        },
        config=ConversationDatasetConfig(normalize_rewards=False),
        embedding_cache=Features(),
    )
    model_config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=32,
        n_embd=32,
        n_layer=1,
        n_head=2,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    # Fix attention numerics for exact resume comparison across both executions.
    model_config._attn_implementation = "eager"
    model = GPT2LMHeadModel(model_config).to(device)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    config = OfflineGRPOConfig(
        model_name="local-synthetic-device-contract",
        state_dim=len(tokenizer),
        action_dim=len(tokenizer),
        value_hidden_size=16,
        value_num_layers=2,
        pretrain_value_epochs=2,
        pretrain_value_batch_size=3,
        batch_size=3,
        gradient_accumulation_steps=2,
        num_epochs=1,
        learning_rate=0.01,
        adam_beta1=0.8,
        adam_beta2=0.95,
        weight_decay=0.1,
        bf16=False,
    )
    trainer = OfflineGRPOTrainer(
        config, model=model, tokenizer=tokenizer, device=device
    )
    first = await trainer.train(dataset=dataset)
    if first["optimizer_updates"] != 1 or not any(
        not torch.equal(before[name], value) for name, value in model.named_parameters()
    ):
        raise RuntimeError("Logged policy training did not update real parameters")
    trainer.save_checkpoint(str(output / "resume"))
    validate_checkpoint_publication(output / "resume", required=True)
    # Reload only the local checkpoint just validated above; no Hub download.
    restored_model = GPT2LMHeadModel.from_pretrained(  # nosec B615
        output / "resume", local_files_only=True, attn_implementation="eager"
    ).to(device)
    restored = OfflineGRPOTrainer(
        config, model=restored_model, tokenizer=tokenizer, device=device
    )
    restored.load(str(output / "resume/training_state.pt"), require_policy=True)
    await trainer.train(num_epochs=2, dataset=dataset)
    metrics = await restored.train(num_epochs=2, dataset=dataset)
    if metrics["optimizer_updates"] != 2 or not all(
        torch.equal(value, restored_model.state_dict()[name])
        for name, value in model.state_dict().items()
    ):
        raise RuntimeError("Restored policy diverged from uninterrupted training")
    restored.save_checkpoint(str(output / "policy"))
    validate_checkpoint_publication(output / "policy", required=True)
    # Reload only the local checkpoint just validated above; no Hub download.
    inference = GPT2LMHeadModel.from_pretrained(  # nosec B615
        output / "policy", local_files_only=True, attn_implementation="eager"
    ).to(device)
    inference.eval()
    inputs = tokenizer("request eligible", return_tensors="pt").to(device)
    with torch.no_grad():
        logits = inference(**inputs).logits
    if not torch.isfinite(logits).all() or logits.device.type != device:
        raise RuntimeError("Reloaded inference did not execute on the requested device")
    if device == "cuda":
        torch.cuda.synchronize()
    report = {
        "schema_version": 1,
        "kind": f"synthetic-offline-{device}-contract",
        "passed": True,
        "representative_quality_demonstrated": False,
        "production_approved": False,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "cuda_version": torch.version.cuda if device == "cuda" else None,
        "peak_vram_bytes": (
            torch.cuda.max_memory_allocated() if device == "cuda" else None
        ),
        "torch_version": str(torch.__version__),
        "config": asdict(config),
        "metrics": metrics,
        "dataset_sha256": restored.dataset_sha256,
        "resumed_weights_match_uninterrupted": True,
        "reload_inference_finite": True,
        "artifact_sha256": artifact_digest(output / "policy"),
        "implementation": package_implementation(),
    }
    (output / "device_contract.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main() -> None:
    """Retain a new device contract run; CUDA availability is mandatory by default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    report = asyncio.run(run_contract(args.output, args.device))
    print(json.dumps({key: report[key] for key in ("kind", "passed", "gpu_name")}))


if __name__ == "__main__":
    main()
