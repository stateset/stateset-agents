"""Rollout publication must not advance the resumed learner's RNG stream."""

import random
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training.multi_turn_trainer import (  # noqa: E402
    MultiTurnGRPOTrainer,
)
from stateset_agents.training.runtime_state import capture_rng  # noqa: E402
from stateset_agents.training.single_turn_trainer import (  # noqa: E402
    SingleTurnGRPOTrainer,
)
from tests.unit.test_multi_turn_checkpoint_accumulation import (  # noqa: E402
    accumulate,
    assert_equal,
)
from tests.unit.test_multi_turn_checkpoint_accumulation import (  # noqa: E402
    trainer as make_native,
)
from tests.unit.test_native_checkpoint_safety import (  # noqa: E402
    snapshot as native_snapshot,
)
from tests.unit.test_runtime_checkpoint_safety import (  # noqa: E402
    TinyRuntime,
    assert_same,
    snapshot,
)


def _random_sample():
    """Use all three saved CPU generators in the next learner input."""
    return [random.random() + float(np.random.random()), float(torch.rand(()))]


def _random_publication(calls):
    def publish():
        # Legitimate backend construction/publication may consume RNG without
        # being part of the learner's next stochastic update.
        random.random()
        np.random.random()
        torch.rand(5)
        calls.append(True)
        return True

    return publish


class StochasticRuntime(TinyRuntime):
    async def train_step(self):
        self.optimizer.zero_grad()
        loss = self.model(torch.tensor([_random_sample()])).square().mean()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1
        self.training_metrics["loss"].append(float(loss.item()))
        return {"loss": float(loss.item())}


async def test_shared_runtime_rng_restored_after_random_rollout_publication(tmp_path):
    random.seed(131)
    np.random.seed(131)
    torch.manual_seed(131)
    original = StochasticRuntime()
    await original.train_step()
    path = tmp_path / "shared"
    original.save_runtime_checkpoint(str(path))
    saved_rng = torch.load(path / "training_state.pt", weights_only=True)["rng"]
    await original.train_step()
    expected = snapshot(original)

    calls = []
    target = StochasticRuntime()
    target.agent = SimpleNamespace(sync_rollout_backend=_random_publication(calls))
    assert target.load_runtime_checkpoint(str(path))
    assert calls == [True]
    assert_same(capture_rng(torch), saved_rng)
    await target.train_step()
    assert_same(snapshot(target), expected)


@pytest.mark.parametrize(
    "trainer_type",
    [SingleTurnGRPOTrainer, MultiTurnGRPOTrainer],
    ids=["single", "multi"],
)
async def test_native_rng_restored_after_random_rollout_publication(
    tmp_path, trainer_type
):
    random.seed(231)
    np.random.seed(231)
    torch.manual_seed(231)
    original = make_native(tmp_path / "original", trainer_type=trainer_type)
    accumulate(original, [1, 2])
    accumulate(original, [2, 3])
    await original.save_checkpoint(checkpoint_name="saved")
    path = tmp_path / "original" / "saved"
    saved_rng = torch.load(path / "training_state.pt", weights_only=True)["rng"]
    calls = []
    original.agent.sync_rollout_backend = _random_publication(calls)
    for _ in range(2):
        accumulate(original, _random_sample())
    expected = native_snapshot(original)

    target = make_native(tmp_path / "target", trainer_type=trainer_type)
    target.agent.sync_rollout_backend = _random_publication(calls)
    assert target.load_checkpoint(path)
    # Once for the uninterrupted optimizer step and once for restored weights.
    assert calls == [True, True]
    assert_equal(capture_rng(torch), saved_rng)
    for _ in range(2):
        accumulate(target, _random_sample())
    assert calls == [True, True, True]
    assert_equal(native_snapshot(target), expected)
