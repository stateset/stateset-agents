"""Token-exact, single-process online PPO with retained behavior evidence.

Full-vocabulary categorical sampling and learner scoring share temperature.
Each episode ends at EOS or the finite token horizon (no truncation bootstrap).
This correctness-first collector scores complete prefixes without a KV cache;
distributed, nucleus/top-k and quantized execution are not certified here.
"""

from __future__ import annotations

import inspect
import json
import math
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.evaluation.implementation import package_implementation

from .runtime_state import capture_rng, restore_rng, validate_training_config
from .state_fingerprint import state_fingerprint

PROTOCOL = "stateset-ppo-online-v1"


def validate_prompts(prompts: list[str]) -> str:
    """Bind the complete ordered prompt population, preserving duplicates."""
    if (
        not isinstance(prompts, list)
        or not prompts
        or any(not isinstance(p, str) or not p.strip() for p in prompts)
    ):
        raise ValueError("Online PPO requires a nonempty list of nonempty prompts")
    return str(content_hash({"protocol": PROTOCOL, "prompts": prompts}))


def validate_online_config(config: Any) -> None:
    """Reject unsupported sampling/update semantics before model allocation."""
    validate_training_config(config)
    if type(config.seed) is not int or not 0 <= config.seed < 2**32:
        raise ValueError("Online PPO seed must be an integer in [0, 2**32)")
    for name in (
        "num_episodes",
        "num_epochs",
        "num_ppo_epochs",
        "mini_batch_size",
        "num_generations",
        "max_prompt_length",
        "max_completion_length",
    ):
        if type(getattr(config, name)) is not int or getattr(config, name) < 1:
            raise ValueError(f"Online PPO {name} must be a positive integer")
    if config.top_p != 1.0 or type(config.top_k) is not int or config.top_k != 0:
        raise ValueError("Online PPO requires top_p=1 and top_k=0")
    if config.use_4bit or config.use_8bit:
        raise ValueError("Online PPO does not support quantized training")
    if (
        type(config.temperature) not in (int, float)
        or not math.isfinite(config.temperature)
        or config.temperature <= 0
    ):
        raise ValueError("Online PPO temperature must be finite and positive")
    if (
        type(config.gradient_accumulation_steps) is not int
        or config.gradient_accumulation_steps != 1
    ):
        raise ValueError("Online PPO requires gradient_accumulation_steps=1")


def _identity(trainer: Any) -> str:
    """Detect callbacks or concurrent callers changing the behavior learner."""
    return str(
        state_fingerprint(
            {
                "protocol": PROTOCOL,
                "config": trainer.config.to_dict(),
                "objective": asdict(trainer._objective),
                "step": trainer.global_step,
                "policy": trainer.model.state_dict(),
                "critic": trainer.value_head.state_dict(),
                "metrics": trainer.metrics_history,
                "gradients": {
                    prefix + "." + name: p.grad
                    for prefix, model in (
                        ("policy", trainer.model),
                        ("critic", trainer.value_head),
                    )
                    for name, p in model.named_parameters()
                },
                "requires_grad": {
                    prefix + "." + name: p.requires_grad
                    for prefix, model in (
                        ("policy", trainer.model),
                        ("critic", trainer.value_head),
                    )
                    for name, p in model.named_parameters()
                },
                "reference": (
                    None
                    if trainer.ref_model is None
                    else trainer.ref_model.state_dict()
                ),
                "policy_optimizer": trainer.policy_optimizer.state_dict(),
                "value_optimizer": trainer.value_optimizer.state_dict(),
                "policy_scheduler": trainer.policy_scheduler.state_dict(),
                "value_scheduler": trainer.value_scheduler.state_dict(),
                "adaptive_kl": (
                    None
                    if trainer.kl_controller is None
                    else vars(trainer.kl_controller)
                ),
            }
        )
    )


def _write(path: Path, data: dict[str, Any]) -> None:
    # Unique journal directories and exclusive creation retain failed attempts.
    encoded = json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)


def _prompt_ids(trainer: Any, prompts: list[str]) -> list[list[int]]:
    import torch

    result = []
    for prompt in prompts:
        encoded = trainer.tokenizer(
            [prompt],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=trainer.config.max_prompt_length + 1,
        )
        ids = encoded["input_ids"][0][encoded["attention_mask"][0].bool()]
        if ids.dtype != torch.long or ids.numel() == 0 or (ids < 0).any():
            raise ValueError(
                "Online PPO requires nonempty nonnegative prompt token IDs"
            )
        if ids.numel() > trainer.config.max_prompt_length:
            raise ValueError("Online PPO prompts exceed max_prompt_length")
        result.append(ids.tolist())
    return result


async def _collect(
    trainer: Any, prompts: list[str], tokens: list[list[int]]
) -> list[dict[str, Any]]:
    import torch

    config = trainer.config
    eos = getattr(trainer.tokenizer, "eos_token_id", None)
    eos_ids = [] if eos is None else eos if isinstance(eos, list) else [eos]
    if any(type(i) is not int or i < 0 for i in eos_ids):
        raise ValueError("Online PPO requires integer EOS token IDs")
    rows = []
    for offset in range(config.mini_batch_size):
        index = (trainer.global_step * config.mini_batch_size + offset) % len(prompts)
        for _ in range(config.num_generations):
            prefix = list(tokens[index])
            actions, log_probs, values = [], [], []
            termination = "horizon"
            for _ in range(config.max_completion_length):
                with torch.no_grad():
                    ids = torch.tensor(
                        [prefix], dtype=torch.long, device=trainer.device
                    )
                    output = trainer.model(
                        input_ids=ids,
                        attention_mask=torch.ones_like(ids),
                        output_hidden_states=True,
                    )
                    logits = output.logits[0, -1].float() / config.temperature
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError(
                            "Online PPO sampling logits must be finite"
                        )
                    likelihoods = torch.log_softmax(logits, dim=-1)
                    action = int(torch.multinomial(likelihoods.exp(), 1).item())
                    hidden = output.hidden_states[-1][:, -1:].to(
                        next(trainer.value_head.parameters()).dtype
                    )
                    value = float(trainer.value_head(hidden).item())
                    if not math.isfinite(value):
                        raise FloatingPointError(
                            "Online PPO behavior values must be finite"
                        )
                actions.append(action)
                log_probs.append(float(likelihoods[action].item()))
                values.append(value)
                prefix.append(action)
                if action in eos_ids:
                    termination = "eos"
                    break
            completion = trainer.tokenizer.decode(actions, skip_special_tokens=True)
            if not isinstance(completion, str):
                raise ValueError("Online PPO tokenizer.decode must return text")
            reward = trainer.reward_fn(prompts[index], completion)
            if inspect.isawaitable(reward):
                reward = await reward
            if (
                isinstance(reward, bool)
                or not isinstance(reward, (int, float))
                or not math.isfinite(reward)
            ):
                raise ValueError("Online PPO rewards must be finite numbers")
            rows.append(
                {
                    "prompt_index": index,
                    "prompt_ids": list(tokens[index]),
                    "action_ids": actions,
                    "behavior_log_probs": log_probs,
                    "behavior_values": values,
                    "completion": completion,
                    "reward": float(reward),
                    "termination": termination,
                }
            )
    return rows


def _pack(trainer: Any, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    import torch

    width = max(len(r["prompt_ids"]) + len(r["action_ids"]) for r in rows)
    # Padding is masked separately, even when token zero is a real action.
    ids = torch.zeros((len(rows), width), dtype=torch.long, device=trainer.device)
    attention = torch.zeros_like(ids)
    response = torch.zeros_like(ids, dtype=torch.float32)
    old_log_probs = torch.zeros_like(response[:, 1:])
    old_values = torch.zeros_like(old_log_probs)
    for i, row in enumerate(rows):
        start, end = len(row["prompt_ids"]), len(row["prompt_ids"]) + len(
            row["action_ids"]
        )
        ids[i, :end] = torch.tensor(
            row["prompt_ids"] + row["action_ids"], device=trainer.device
        )
        attention[i, :end] = 1
        response[i, start:end] = 1
        old_log_probs[i, start - 1 : end - 1] = torch.tensor(
            row["behavior_log_probs"], device=trainer.device
        )
        old_values[i, start - 1 : end - 1] = torch.tensor(
            row["behavior_values"], device=trainer.device
        )
    rewards = torch.tensor(
        [r["reward"] for r in rows], dtype=torch.float32, device=trainer.device
    )
    if not torch.isfinite(rewards).all():
        raise ValueError("Online PPO rewards must remain finite in the learner dtype")
    return {
        "input_ids": ids,
        "attention_mask": attention,
        "response_mask": response,
        "rewards_tensor": rewards,
        "old_log_probs": old_log_probs,
        "old_values": old_values,
    }


async def train_online(
    trainer: Any, prompts: list[str], num_episodes: int | None
) -> dict[str, Any]:
    """Run bounded online updates and retain raw rollout/commit receipts.

    num_episodes is a total update target, including previously completed steps.
    Checkpoint publication and reentrant training are blocked inside reward
    callbacks. A callback that mutates learner state quarantines the runtime.
    """
    import torch

    from .ppo_trainer import _deterministic_scoring

    if getattr(trainer, "_ppo_online_active", False):
        raise RuntimeError("PPO online operation is already active")
    config = trainer.config
    validate_online_config(config)
    prompt_hash = validate_prompts(prompts)
    target = (
        config.num_episodes * config.num_epochs
        if num_episodes is None
        else num_episodes
    )
    if type(target) is not int or target < 1:
        raise ValueError(
            "Online PPO num_episodes must be a positive total update budget"
        )
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        raise ValueError("Online PPO currently supports one process")
    if getattr(trainer.model.config, "is_encoder_decoder", False):
        raise ValueError("Online PPO requires a causal language model")
    if any(
        getattr(trainer.model, flag, False)
        for flag in ("is_loaded_in_4bit", "is_loaded_in_8bit")
    ):
        raise ValueError("Online PPO does not support quantized training")
    if config.beta > 0:
        reference = trainer.ref_model
        if (
            reference is None
            or reference is trainer.model
            or any(p.requires_grad for p in reference.parameters())
        ):
            raise ValueError(
                "Online PPO requires a separate frozen reference model when beta > 0"
            )
        policy_storage = {p.data_ptr() for p in trainer.model.parameters()}
        if any(p.data_ptr() in policy_storage for p in reference.parameters()):
            raise ValueError("Online PPO reference storage must be independent")
    tokens = _prompt_ids(trainer, prompts)  # Validate all prompts before any update.
    prompt_hash = content_hash(
        {
            "prompts_sha256": prompt_hash,
            "token_ids": tokens,
            "eos_token_id": getattr(trainer.tokenizer, "eos_token_id", None),
        }
    )
    if config.ppo_prompt_sha256 not in (None, prompt_hash):
        raise ValueError("Online PPO prompt population differs from the bound study")
    config.ppo_prompt_sha256 = prompt_hash
    if config.resume_from_checkpoint and not getattr(
        trainer, "_ppo_resume_loaded", False
    ):
        trainer.load_runtime_checkpoint(config.resume_from_checkpoint)
        trainer._ppo_resume_loaded = True
    if target < trainer.global_step:
        raise ValueError("Online PPO total update budget is below the restored step")
    if target == trainer.global_step:
        return {"global_step": trainer.global_step, "updates": 0, "protocol": PROTOCOL}
    root = Path(config.output_dir) / "ppo-rollouts"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    journal = Path(tempfile.mkdtemp(prefix="run-", dir=root))
    _write(
        journal / "manifest.json",
        {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "config": config.to_dict(),
            "resolved_objective": asdict(trainer._objective),
            "prompts": list(prompts),
            "prompts_sha256": prompt_hash,
            "implementation": package_implementation(),
            "torch_version": str(torch.__version__),
            "quality_certified": False,
        },
    )
    initial_step = trainer.global_step
    trainer._ppo_online_active = True
    try:
        while trainer.global_step < target:
            rng = capture_rng(torch)
            identity = _identity(trainer)
            trainer._ppo_update_in_flight = False
            try:
                with _deterministic_scoring(
                    trainer.model, trainer.value_head, trainer.ref_model
                ):
                    rows = await _collect(trainer, prompts, tokens)
                    if _identity(trainer) != identity:
                        trainer._runtime_checkpoint_poisoned = True
                        raise RuntimeError(
                            "PPO learner changed while collecting behavior samples"
                        )
                    batch = _pack(trainer, rows)
                    # Verify causal/padding scoring consistency, without replacing
                    # the likelihoods actually used to sample each action.
                    with torch.no_grad():
                        scored, _ = trainer.compute_log_probs(
                            batch["input_ids"],
                            batch["attention_mask"],
                            config.temperature,
                        )
                        hidden = trainer.get_hidden_states(
                            batch["input_ids"], batch["attention_mask"]
                        )
                        scored_values = trainer.value_head(
                            hidden.to(next(trainer.value_head.parameters()).dtype)
                        ).squeeze(-1)[:, :-1]
                    mask = batch["response_mask"][:, 1:].bool()
                    if not torch.allclose(
                        scored[mask], batch["old_log_probs"][mask], atol=1e-4, rtol=1e-4
                    ):
                        raise ValueError(
                            "PPO sampled and batched behavior likelihoods disagree"
                        )
                    if not torch.allclose(
                        scored_values[mask],
                        batch["old_values"][mask],
                        atol=1e-4,
                        rtol=1e-4,
                    ):
                        raise ValueError(
                            "PPO sampled and batched behavior values disagree"
                        )
                    if _identity(trainer) != identity:
                        trainer._runtime_checkpoint_poisoned = True
                        raise RuntimeError(
                            "PPO learner changed while scoring behavior samples"
                        )
                    step = trainer.global_step
                    record = {
                        "protocol": PROTOCOL,
                        "step": step,
                        "behavior_state_sha256": identity,
                        "prompts_sha256": prompt_hash,
                        "temperature": config.temperature,
                        "rows": rows,
                    }
                    _write(journal / f"rollout-{step}.json", record)
                    metrics = trainer._update_token_batch(
                        **batch, temperature=config.temperature
                    )
                    _write(
                        journal / f"commit-{step}.json",
                        {
                            "protocol": PROTOCOL,
                            "rollout_sha256": content_hash(record),
                            "global_step": trainer.global_step,
                            "metrics": metrics,
                        },
                    )
            except BaseException:
                try:
                    changed = _identity(trainer) != identity
                except Exception:
                    changed = True
                if trainer._ppo_update_in_flight or changed:
                    trainer._runtime_checkpoint_poisoned = True
                else:
                    restore_rng(rng, torch)
                raise
            finally:
                trainer._ppo_update_in_flight = False
    finally:
        trainer._ppo_online_active = False
    return {
        "protocol": PROTOCOL,
        "global_step": trainer.global_step,
        "updates": trainer.global_step - initial_step,
        "rollout_directory": str(journal),
        "metrics": metrics,
    }
