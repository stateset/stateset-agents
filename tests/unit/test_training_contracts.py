"""Training semantics fail before compute and never reward surface phrasing."""

from types import SimpleNamespace

import pytest

from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.rewards.task_outcome import TaskOutcomeReward
from stateset_agents.training.config import TrainingConfig
from stateset_agents.training.rollout_contract import SampledTurn


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_generations": 0},
        {"num_gradient_updates": True},
        {"learning_rate": float("nan")},
        {"temperature": 0},
        {"bf16": True, "fp16": True},
        {"top_p": 0},
        {"rollout_sync": False},
        {"objective": "misspelled"},
        # a zero cadence divided by zero inside the training loop
        {"logging_steps": 0},
        {"save_steps": 0},
        {"eval_steps": 0},
        {"save_steps": 2.5},
        # an unknown scheduler silently trained at a constant rate
        {"lr_scheduler_type": "cosine_with_restarts"},
    ],
)
def test_invalid_semantics_fail_before_training(overrides):
    with pytest.raises(ValueError):
        TrainingConfig(**overrides).validate_for_training()


@pytest.mark.parametrize(
    "metadata",
    [
        {"prompt_token_ids": [1.5], "token_ids": [2]},
        {"prompt_token_ids": [True], "token_ids": [2]},
        {"prompt_token_ids": [1], "token_ids": [-1]},
        {"prompt_token_ids": [1], "token_ids": [2], "sampler_log_probs": []},
        {
            "prompt_token_ids": [1],
            "token_ids": [2],
            "sampler_log_probs": [float("nan")],
        },
        {"prompt_token_ids": [1], "token_ids": [2], "sampler_log_probs": [0.1]},
    ],
)
def test_rollout_contract_rejects_corruption(metadata):
    with pytest.raises(ValueError):
        SampledTurn.from_metadata(metadata)


@pytest.mark.asyncio
async def test_outcome_reward_requires_reference_and_executed_tools():
    reward = TaskOutcomeReward()
    with pytest.raises(ValueError, match="trusted"):
        await reward.compute_reward(
            [ConversationTurn(role="assistant", content="Happy to help!")]
        )
    context = {
        "scenario": {
            "expected_response": "Resolved",
            "expected_tool_calls": [{"tool": "refund", "amount": 10}],
        }
    }
    turns = [ConversationTurn(role="assistant", content="Resolved")]
    with pytest.raises(ValueError, match="executed tool trace"):
        await reward.compute_reward(turns, context)
    context["executed_tool_calls"] = [{"tool": "refund", "amount": 10}]
    assert (await reward.compute_reward(turns, context)).score == 1
    context["policy_violations"] = 1
    assert (await reward.compute_reward(turns, context)).score == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("wrong_amount", [True, 1.0])
async def test_tool_outcomes_preserve_argument_types(wrong_amount):
    reward = TaskOutcomeReward()
    context = {
        "scenario": {
            "expected_response": "Resolved",
            "expected_tool_calls": [{"amount": 1}],
        },
        "executed_tool_calls": [{"amount": wrong_amount}],
    }
    turns = [ConversationTurn(role="assistant", content="Resolved")]
    assert (await reward.compute_reward(turns, context)).score == 0


@pytest.mark.asyncio
async def test_batched_generation_trims_eos_and_restores_mode():
    import torch

    from stateset_agents.training.trainer_runtime import hf_generate_group

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(use_cache=False)
            self.calls = []

        def generate(self, **kwargs):
            assert not self.training and self.config.use_cache
            self.calls.append(kwargs["num_return_sequences"])
            return torch.tensor([[1, 4, 2, 0]] * kwargs["num_return_sequences"])

    class Tokenizer:
        eos_token_id, pad_token_id = 2, 0

        def __call__(self, *args, **kwargs):
            return {
                "input_ids": torch.tensor([[1]]),
                "attention_mask": torch.tensor([[1]]),
            }

        def decode(self, ids, **kwargs):
            return str(ids.tolist())

    model = Model()
    rows = await hf_generate_group(
        model, Tokenizer(), TrainingConfig(generation_batch_size=2), "cpu", "request", 5
    )
    assert model.calls == [2, 2, 1]
    assert len(rows) == 5 and all(row["sequence_length"] == 2 for row in rows)
    assert model.training and model.config.use_cache is False


@pytest.mark.parametrize("scheduler", ["cosine", "linear", "constant"])
def test_native_trainers_accept_the_schedulers_they_implement(scheduler):
    TrainingConfig(lr_scheduler_type=scheduler).validate_for_training()


def test_trl_configs_keep_the_wider_trl_scheduler_set():
    from stateset_agents.training.trl_grpo_config import TRLGRPOConfig

    TRLGRPOConfig(lr_scheduler_type="cosine_with_restarts").validate_for_training()
    with pytest.raises(ValueError, match="lr_scheduler_type"):
        TRLGRPOConfig(lr_scheduler_type="").validate_for_training()
