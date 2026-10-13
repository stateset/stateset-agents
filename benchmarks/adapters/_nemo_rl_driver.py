"""NeMo-RL GRPO/GSPO driver for the matched GSM8K shootout.

Runs inside a NeMo-RL v0.7.0 source checkout (``uv run --locked python
_nemo_rl_driver.py --config ...``), never in the harness venv. It follows
``examples/run_grpo.py`` from that release and adds only what the matched
protocol needs:

* a data processor that tokenizes the fixed prompt as raw text (no chat
  template), like the TRL adapter;
* a Ray environment that scores completions with the TRL adapter's GSM8K
  scorer (``_matched_gsm8k.score_text``);
* a tee of every ``Logger.log_metrics`` call into a JSONL file the adapter
  parses for steps, rewards and throughput.

Ray workers import this module by its fully-qualified name, so the adapters
directory must be on ``PYTHONPATH`` (the adapter sets it).
"""

from __future__ import annotations

import argparse
import json
import os
import pprint
from pathlib import Path
from typing import Any

import ray
import torch
from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
from nemo_rl.data.processors import register_processor
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES, init_ray
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.environments.utils import register_env
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils import logger as nemo_logger
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from omegaconf import OmegaConf

ENV_FQN = "_nemo_rl_driver.StateSetGSM8KEnvironment"


def raw_prompt_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: Any,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Tokenize the fixed prompt verbatim and carry the gold label."""
    messages = datum_dict["messages"]
    prompt = str(messages[0]["content"])
    gold = str(messages[1]["content"])
    token_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ][0]
    message_log = [{"role": "user", "content": prompt, "token_ids": token_ids}]
    length = int(len(token_ids))
    loss_multiplier = 1.0
    if length >= max_seq_length:
        message_log[0]["token_ids"] = token_ids[: min(4, max_seq_length)]
        loss_multiplier = 0.0
    return {
        "message_log": message_log,
        "length": length,
        "extra_env_info": {"ground_truth": gold},
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": datum_dict["task_name"],
    }


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class StateSetGSM8KEnvironment(EnvironmentInterface):
    """Single-turn environment scoring with the shared GSM8K scorer."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def step(
        self,
        message_log_batch: list[list[dict[str, Any]]],
        metadata: list[dict[str, Any]],
    ) -> EnvironmentReturn:
        from _matched_gsm8k import score_text

        rewards = []
        for conversation, meta in zip(message_log_batch, metadata, strict=True):
            response = "".join(
                str(turn["content"])
                for turn in conversation
                if turn["role"] == "assistant"
            )
            rewards.append(score_text(response, meta["ground_truth"]))
        tensor = torch.tensor(rewards, dtype=torch.float32).cpu()
        return EnvironmentReturn(
            observations=[
                {"role": "environment", "content": f"reward={value}"}
                for value in rewards
            ],
            metadata=metadata,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=tensor,
            terminateds=torch.ones_like(tensor).cpu(),
            answers=None,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float]]:
        rewards = batch["rewards"]
        return batch, {"accuracy": float(rewards.float().mean().item())}

    def shutdown(self) -> None:
        return None


def tee_metrics(path: Path) -> None:
    """Append every logged metric dict to ``path`` as JSON lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    original = nemo_logger.Logger.log_metrics

    def log_metrics(
        self: Any,
        metrics: dict[str, Any],
        step: int,
        prefix: str | None = "",
        *extra_args: Any,
        **extra_kwargs: Any,
    ) -> Any:
        scalars = {}
        for key, value in metrics.items():
            try:
                scalars[key] = float(value)
            except (TypeError, ValueError):
                mean = getattr(value, "mean", None)
                if callable(mean):
                    try:
                        scalars[key] = float(mean())
                    except (TypeError, ValueError):
                        continue
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps({"step": int(step), "prefix": prefix, "metrics": scalars})
                + "\n"
            )
        return original(self, metrics, step, prefix, *extra_args, **extra_kwargs)

    nemo_logger.Logger.log_metrics = log_metrics  # type: ignore[method-assign]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--metrics-jsonl", type=Path, required=True)
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--processor-name", required=True)
    args = parser.parse_args()

    register_omegaconf_resolvers()
    register_processor(args.processor_name, raw_prompt_processor)
    register_env(args.env_name, ENV_FQN)
    ACTOR_ENVIRONMENT_REGISTRY[ENV_FQN] = PY_EXECUTABLES.SYSTEM
    tee_metrics(args.metrics_jsonl)

    config = OmegaConf.to_container(load_config(args.config), resolve=True)
    config = MasterConfig(**config)
    pprint.pprint(config)
    os.makedirs(config.logger["log_dir"], exist_ok=True)
    init_ray()
    tokenizer = get_tokenizer(config.policy["tokenizer"])
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"],
        tokenizer,
        has_refit_draft_weights=False,
        trains_mtp=False,
    )
    dataset, val_dataset, task_to_env, val_task_to_env = setup_response_data(
        tokenizer, config.data, config.env
    )
    (
        policy,
        policy_generation,
        _nemo_gym,
        _cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
        _teachers,
        _aliases,
    ) = setup(config, tokenizer, dataset, val_dataset)
    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        val_task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


if __name__ == "__main__":
    main()
