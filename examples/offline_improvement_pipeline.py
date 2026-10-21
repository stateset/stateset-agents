"""Run a synthetic trace → SFT → offline IQL → export → local inference demo.

Requires the training extra; uses a tiny random transformer with no downloads.
This verifies mechanics, not representative agent quality or production serving.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from stateset_agents.data.conversation_dataset import (
    ConversationDataset,
    ConversationDatasetConfig,
)
from stateset_agents.experimental.offline_rl.offline_grpo import (
    OfflineGRPOConfig,
    OfflineGRPOTrainer,
)
from stateset_agents.training.improvement_pipeline import run_improvement_pipeline
from stateset_agents.training.sampled_workflow import workflow_tokenizer

logger = logging.getLogger(__name__)


async def run_demo(output: Path) -> dict[str, Any]:
    """Train real logged assistant tokens and reload both evaluation policies."""
    torch.manual_seed(42)
    tokenizer = workflow_tokenizer()
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=32,
            n_embd=24,
            n_layer=1,
            n_head=2,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
            bos_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    )
    records = [
        {
            "source_id": f"{split}-{index}",
            "timestamp": timestamp,
            "messages": [
                {
                    "role": "user",
                    "content": f"request {split} {'eligible' if index else 'ineligible'}",
                },
                {"role": "assistant", "content": "refund" if index else "deny"},
            ],
            "reward": 1.0,
        }
        for split, timestamp in [
            ("train", "2026-01-01T00:00:00Z"),
            ("validation", "2026-02-01T00:00:00Z"),
            ("test", "2026-03-01T00:00:00Z"),
        ]
        for index in range(2)
    ]

    class TokenCountFeatures:
        """Local deterministic text features, deliberately small for this demo."""

        def get_or_compute(self, key: str, text: str) -> torch.Tensor:
            ids = torch.tensor(tokenizer.encode(text, add_special_tokens=False))
            counts = torch.bincount(ids, minlength=len(tokenizer)).float()
            return counts / counts.sum().clamp_min(1)

    def dataset(rows: list[dict[str, Any]]) -> ConversationDataset:
        return ConversationDataset.from_dict(
            {
                "trajectories": [
                    {
                        "trajectory_id": row["content_sha256"],
                        "turns": [
                            {
                                **message,
                                "reward": (
                                    row["reward"]
                                    if index == len(row["messages"]) - 1
                                    else 0
                                ),
                            }
                            for index, message in enumerate(row["messages"])
                        ],
                    }
                    for row in rows
                ]
            },
            config=ConversationDatasetConfig(normalize_rewards=False),
            embedding_cache=TokenCountFeatures(),
        )

    async def sft(policy: Any, train: list, validation: list) -> Any:
        rows = dataset(train).to_policy_training_format(tokenizer, 32)
        ids = torch.tensor([row["input_ids"] for row in rows])
        labels = torch.tensor([row["labels"] for row in rows])
        optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01)
        policy.train()
        for _ in range(10):
            optimizer.zero_grad(set_to_none=True)
            loss = policy(input_ids=ids, labels=labels).loss
            loss.backward()
            optimizer.step()
        (output / "sft_metrics.json").write_text(
            json.dumps({"optimizer_updates": 10, "final_loss": float(loss.detach())})
            + "\n"
        )
        return policy

    async def rl(policy: Any, train: list, validation: list) -> Any:
        trainer = OfflineGRPOTrainer(
            OfflineGRPOConfig(
                state_dim=len(tokenizer),
                action_dim=len(tokenizer),
                value_hidden_size=16,
                value_num_layers=2,
                pretrain_value_epochs=2,
                pretrain_value_batch_size=2,
                batch_size=2,
                num_epochs=2,
                gradient_accumulation_steps=1,
                learning_rate=0.001,
            ),
            model=policy,
            tokenizer=tokenizer,
            device="cpu",
        )
        metrics = await trainer.train(dataset=dataset(train))
        (output / "rl_metrics.json").write_text(
            json.dumps(
                {**metrics, "dataset_sha256": trainer.dataset_sha256},
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        return policy

    def export(policy: Any, path: Path) -> Path:
        policy.save_pretrained(path)
        tokenizer.save_pretrained(path)
        return path

    def predict(policy: Any, prompt: str) -> str:
        policy.eval()
        with torch.no_grad():
            tokens = tokenizer(prompt, return_tensors="pt")
            response = policy.generate(**tokens, max_new_tokens=1, do_sample=False)
        return tokenizer.decode(response[0, tokens.input_ids.shape[1] :]).strip()

    def evaluate(artifact: Path, test: list) -> dict[str, Any]:
        policy = GPT2LMHeadModel.from_pretrained(artifact, local_files_only=True)
        outcomes = [
            {
                "case_id": row["content_sha256"],
                "response": predict(policy, row["messages"][0]["content"]),
                "expected": row["messages"][-1]["content"],
            }
            for row in test
        ]
        return {
            "synthetic": True,
            "outcomes": outcomes,
            "successes": sum(row["response"] == row["expected"] for row in outcomes),
        }

    def serve(artifact: Path) -> dict[str, Any]:
        reloaded = GPT2LMHeadModel.from_pretrained(artifact, local_files_only=True)
        response = predict(reloaded, "request test eligible")
        return {
            "reloaded": True,
            "served": True,
            "scope": "local inference",
            "response": response,
        }

    return await run_improvement_pipeline(
        records,
        output=output,
        initial_policy=model,
        validation_after="2026-02-01T00:00:00Z",
        test_after="2026-03-01T00:00:00Z",
        supervised_train=sft,
        reinforcement_train=rl,
        evaluate=evaluate,
        export_policy=export,
        reload_and_serve=serve,
    )


def main() -> None:
    """Write a new local workflow receipt and both reusable policy exports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    report = asyncio.run(run_demo(args.output))
    logger.info("Completed synthetic workflow: %s", report["artifact"])


if __name__ == "__main__":
    main()
