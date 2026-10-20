"""Resumption reproduces stochastic updates, including pending gradients."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stateset_agents.training.config import TrainingConfig
from stateset_agents.training.runtime_state import (
    TrainingRuntime,
    capture_rng,
    restore_rng,
)


class Trainer(TrainingRuntime):
    def __init__(self):
        from transformers import GPT2Config, GPT2LMHeadModel

        self.config = TrainingConfig(bf16=False, report_to="none")
        self.model = GPT2LMHeadModel(
            GPT2Config(vocab_size=16, n_positions=8, n_embd=8, n_layer=1, n_head=1)
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, 0.9)
        self.global_step = 1
        self._grad_accum_step = 1


def update(trainer):
    trainer.model.train()
    ids = torch.randint(0, 16, (1, 4))
    trainer.model(ids, labels=ids).loss.backward()
    trainer.optimizer.step()
    trainer.scheduler.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    return random.random(), np.random.random(), torch.rand(2)


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_interrupted_update_matches_uninterrupted_update(tmp_path):
    trainer = Trainer()
    ids = torch.tensor([[1, 2, 3, 4]])
    trainer.model(ids, labels=ids).loss.backward()
    trainer.save_runtime_checkpoint(str(tmp_path / "checkpoint"))
    reference_rng = update(trainer)
    expected = {
        key: value.detach().clone() for key, value in trainer.model.state_dict().items()
    }
    resumed = Trainer()
    assert resumed.load_runtime_checkpoint(str(tmp_path / "checkpoint"))
    actual_rng = update(resumed)
    assert actual_rng[:2] == reference_rng[:2]
    torch.testing.assert_close(actual_rng[2], reference_rng[2], rtol=0, atol=0)
    for key, value in resumed.model.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    assert resumed.scheduler.state_dict() == trainer.scheduler.state_dict()


@pytest.mark.parametrize("device_count", [1, 2])
def test_device_mapped_cuda_rng_restores_each_generator(device_count):
    """Device-mapped checkpoint states still restore byte-identical streams."""
    generators = [torch.Generator().manual_seed(17 + i) for i in range(device_count)]

    class DeviceMappedState:
        def __init__(self, value):
            self.value = value

        def cpu(self):
            return self.value

    class CudaBackend:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return len(generators)

        @staticmethod
        def set_rng_state_all(values):
            # Real generators enforce the same CPU ByteTensor requirement as
            # CUDA's API, without requiring a GPU in the regression suite.
            for generator, value in zip(generators, values, strict=True):
                generator.set_state(value)

    state = capture_rng(torch)
    state["cuda"] = [
        DeviceMappedState(generator.get_state()) for generator in generators
    ]
    expected = [torch.rand(4, generator=generator) for generator in generators]
    restore_rng(
        state,
        SimpleNamespace(cuda=CudaBackend(), set_rng_state=torch.set_rng_state),
    )
    for generator, value in zip(generators, expected, strict=True):
        torch.testing.assert_close(
            torch.rand(4, generator=generator), value, rtol=0, atol=0
        )


def test_partial_checkpoint_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="incomplete"):
        Trainer().load_runtime_checkpoint(str(tmp_path))


def test_restored_policy_publication_never_reuses_an_engine_version(tmp_path):
    class Agent:
        def __init__(self):
            self._rollout_backend_version = 4

        @property
        def rollout_backend_version(self):
            return self._rollout_backend_version

        def sync_rollout_backend(self):
            self._rollout_backend_version += 1
            return True

    trainer = Trainer()
    trainer.agent = Agent()
    trainer.save_runtime_checkpoint(str(tmp_path / "checkpoint"))
    trainer.agent._rollout_backend_version = 0
    trainer.load_runtime_checkpoint(str(tmp_path / "checkpoint"))
    assert trainer.agent.rollout_backend_version == 5
    trainer.agent._rollout_backend_version = 10
    trainer.load_runtime_checkpoint(str(tmp_path / "checkpoint"))
    assert trainer.agent.rollout_backend_version == 11


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [None, False, True])
async def test_runtime_cost_metrics_keep_unknown_outcomes_unknown(outcome):
    from stateset_agents.core.trajectory import (
        ConversationTurn,
        MultiTurnTrajectory,
        TrajectoryGroup,
    )

    class Profiled(TrainingRuntime):
        config = TrainingConfig(compute_cost_per_hour=1.0)

        async def generate_trajectories(self):
            state = {} if outcome is None else {"task_success": outcome}
            return [
                TrajectoryGroup(
                    scenario_id="case",
                    trajectories=[
                        MultiTurnTrajectory(
                            turns=[
                                ConversationTurn(
                                    role="assistant",
                                    content="done",
                                    metadata={"token_ids": [1, 2]},
                                )
                            ],
                            final_state=state,
                        )
                    ],
                )
            ]

        async def training_step(self):
            return {"loss": 0.0}

    trainer = Profiled()
    await trainer.generate_trajectories()
    metrics = await trainer.training_step()
    assert (
        metrics["generated_tokens"] == 2 and metrics["estimated_compute_cost_usd"] > 0
    )
    assert ("successful_episodes" in metrics) == (outcome is not None)
    assert ("successful_episodes_per_usd" in metrics) == (outcome is not None)


def test_checkpoint_configuration_mismatch_is_rejected(tmp_path):
    trainer = Trainer()
    trainer.save_runtime_checkpoint(str(tmp_path / "checkpoint"))
    trainer.config.gradient_accumulation_steps += 1
    with pytest.raises(ValueError, match="configuration"):
        trainer.load_runtime_checkpoint(str(tmp_path / "checkpoint"))


def test_real_lora_adapter_round_trip_restores_policy_outputs(tmp_path):
    from peft import LoraConfig, get_peft_model

    trainer = Trainer()
    trainer.model = get_peft_model(
        trainer.model, LoraConfig(r=2, target_modules=["c_attn"], task_type="CAUSAL_LM")
    )
    trainer.optimizer = torch.optim.AdamW(
        [p for p in trainer.model.parameters() if p.requires_grad], lr=0.01
    )
    trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, 1, 0.9)
    update(trainer)
    trainer.model.eval()
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        expected = trainer.model(ids).logits.clone()
    trainer.save_runtime_checkpoint(str(tmp_path / "checkpoint"))
    with torch.no_grad():
        for parameter in trainer.model.parameters():
            if parameter.requires_grad:
                parameter.zero_()
    assert trainer.load_runtime_checkpoint(str(tmp_path / "checkpoint"))
    with torch.no_grad():
        torch.testing.assert_close(trainer.model(ids).logits, expected, rtol=0, atol=0)
