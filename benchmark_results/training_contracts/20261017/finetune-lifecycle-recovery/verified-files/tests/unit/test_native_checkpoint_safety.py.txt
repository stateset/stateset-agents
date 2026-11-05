"""Local native GRPO restores must preflight state and contain failed commits."""

import copy
import random
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training.checkpoint_publication import (  # noqa: E402
    COMPLETION_FILE,
    checkpoint_write,
)
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


@pytest.fixture(
    params=[SingleTurnGRPOTrainer, MultiTurnGRPOTrainer], ids=["single", "multi"]
)
def factory(request, tmp_path):
    return lambda name, **kwargs: make_native(
        tmp_path / name, trainer_type=request.param, **kwargs
    )


@pytest.fixture
async def saved(factory):
    torch.manual_seed(17)
    trainer = factory("original")
    accumulate(trainer, [1, 2])
    accumulate(trainer, [2, 3])
    await trainer.save_checkpoint(checkpoint_name="good")
    return trainer, Path(trainer.config.output_dir) / "good"


def snapshot(trainer):
    return copy.deepcopy(
        {
            "policy": trainer.agent.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "scheduler": trainer.lr_scheduler.state_dict(),
            "scaler": None if trainer.scaler is None else trainer.scaler.state_dict(),
            "gradients": {
                name: value.grad
                for name, value in trainer.agent.model.named_parameters()
            },
            "global_step": trainer.global_step,
            "current_epoch": trainer.current_epoch,
            "next_episode": getattr(trainer, "_next_episode", None),
            "best": trainer.best_eval_metric,
            "patience": trainer.steps_without_improvement,
            "accumulation": trainer._grad_accum_step,
            "normalization": trainer._grpo_accumulation,
            "task": trainer._current_task_id,
            "rng": capture_rng(torch),
            "poisoned": getattr(trainer, "_runtime_checkpoint_poisoned", False),
        }
    )


def republish(source, target, mutate):
    with checkpoint_write(target) as staging:
        for file in source.iterdir():
            if file.name == COMPLETION_FILE:
                continue
            if file.is_dir():
                shutil.copytree(file, staging / file.name)
            else:
                shutil.copy2(file, staging / file.name)
        path = staging / "training_state.pt"
        state = torch.load(path, weights_only=True)
        mutate(state, staging)
        torch.save(state, path)


@pytest.mark.parametrize(
    "failure",
    [
        "slot_nan",
        "slot_shape",
        "negative_lr",
        "bad_scheduler",
        "missing_counter",
        "bool_counter",
        "rng",
        "task",
        "rollout",
        "native_schema",
        "missing_learning",
        "missing_component_identity",
    ],
)
async def test_invalid_published_native_state_does_not_change_learner(
    saved, factory, tmp_path, failure
):
    _, path = saved

    def mutate(state, staging):
        optimizer = state["optimizer_state_dict"]
        if failure == "slot_nan":
            next(iter(optimizer["state"].values()))["momentum_buffer"].fill_(
                float("nan")
            )
        elif failure == "slot_shape":
            next(iter(optimizer["state"].values()))["momentum_buffer"] = torch.ones(19)
        elif failure == "negative_lr":
            optimizer["param_groups"][0]["lr"] = -0.1
        elif failure == "bad_scheduler":
            state["scheduler_state_dict"]["last_epoch"] = "bad"
        elif failure == "missing_counter":
            del state["global_step"]
        elif failure == "bool_counter":
            state["global_step"] = True
        elif failure == "rng":
            state["rng"]["torch"] = torch.ones(2)
        elif failure == "task":
            state["current_task_id"] = "other"
        elif failure == "native_schema":
            state["native_checkpoint_schema"] = True
        elif failure == "missing_learning":
            del state["learning_state"]
        elif failure == "missing_component_identity":
            del state["native_component_identities"]
        else:
            state["rollout_state"] = {"schema_version": 1, "publication_version": -1}

    bad = tmp_path / failure
    republish(path, bad, mutate)
    target = factory("target")
    before = snapshot(target)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        target.load_checkpoint(bad)
    assert_equal(snapshot(target), before)
    assert target.load_checkpoint(path)


@pytest.mark.parametrize(
    "failure", ["model", "optimizer", "scheduler", "tokenizer", "rng"]
)
async def test_late_native_failure_blocks_training_and_save_until_full_reload(
    saved, factory, monkeypatch, failure
):
    from stateset_agents.training import checkpoint_state

    _, path = saved
    target = factory("target")

    def fail(*args, **kwargs):
        raise RuntimeError("injected late native failure")

    with monkeypatch.context() as patch:
        if failure in {"model", "optimizer", "scheduler"}:
            component = {
                "model": target.agent.model,
                "optimizer": target.optimizer,
                "scheduler": target.lr_scheduler,
            }[failure]
            if failure == "scheduler":
                original = checkpoint_state._prepare_components
                original_load = type(component).load_state_dict

                def commit(instance, *args, **kwargs):
                    if instance is component:
                        return fail(*args, **kwargs)
                    return original_load(instance, *args, **kwargs)

                def prepare(*args, **kwargs):
                    prepared = original(*args, **kwargs)
                    # Install the commit failure after schema staging, without
                    # changing the scheduler's recorded state/binding identity.
                    # Patch the class after validation so teardown restores
                    # inherited-method presence, without a bound-method entry
                    # leaking into the scheduler's instance state dictionary.
                    patch.setattr(type(component), "load_state_dict", commit)
                    return prepared

                patch.setattr(checkpoint_state, "_prepare_components", prepare)
                # Runtime identities describe implementation bindings; this
                # local fault injector must not substitute for that preflight.
                original_identity = checkpoint_state._component_identities
                expected_identity = original_identity(
                    checkpoint_state._native_components(target), torch
                )
                patch.setattr(
                    checkpoint_state,
                    "_component_identities",
                    lambda *args: expected_identity,
                )
            else:
                patch.setattr(component, "load_state_dict", fail)
        elif failure == "tokenizer":
            target.agent.tokenizer = SimpleNamespace(from_pretrained=fail)
        else:
            patch.setattr(checkpoint_state, "restore_rng", fail)
        with pytest.raises(RuntimeError, match="injected late native failure"):
            target.load_checkpoint(path)
    assert target._runtime_checkpoint_poisoned
    with pytest.raises(RuntimeError, match="quarantined"):
        await target.train()
    with pytest.raises(RuntimeError, match="quarantined"):
        await target.save_checkpoint(checkpoint_name="blocked")
    with pytest.raises(RuntimeError, match="quarantined"):
        target._apply_optimizer_step(torch)
    assert target.load_checkpoint(path / "missing", trusted=True) is False
    assert target._runtime_checkpoint_poisoned
    if failure == "tokenizer":
        target.agent.tokenizer = None
    assert target.load_checkpoint(path)
    assert not target._runtime_checkpoint_poisoned
    accumulate(target, [1, 4])
    accumulate(target, [2, 1])


@pytest.mark.parametrize(
    "sync", ["false", "error", "true", "none", "absent", "disabled"]
)
async def test_native_restore_publication_is_owned_once(saved, factory, sync):
    _, path = saved
    target = factory("sync-target")
    outcome = {"false": False, "true": True, "none": None}.get(sync, True)
    publish = Mock(
        side_effect=RuntimeError("publication error") if sync == "error" else None,
        return_value=outcome,
    )
    if sync != "absent":
        target.agent.sync_rollout_backend = publish
    if sync == "disabled":
        # Match the recorded opt-out as well: restore cannot change objective
        # controls by pretending an enabled run was asynchronous.
        target.config.rollout_sync = False
        original, _ = saved
        original.config.rollout_sync = False
        await original.save_checkpoint(checkpoint_name="disabled")
        path = Path(original.config.output_dir) / "disabled"
    if sync in {"false", "error"}:
        with pytest.raises(RuntimeError, match="publication"):
            target.load_checkpoint(path)
        assert target._runtime_checkpoint_poisoned
        with pytest.raises(RuntimeError, match="quarantined"):
            await target.train()
        publish.side_effect = None
        publish.return_value = True
        assert target.load_checkpoint(path)
    else:
        assert target.load_checkpoint(path)
    assert not target._runtime_checkpoint_poisoned
    assert publish.call_count == (
        0 if sync in {"absent", "disabled"} else 2 if sync in {"false", "error"} else 1
    )
    assert target.last_rollout_sync is (
        None if sync in {"none", "absent", "disabled"} else True
    )


@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("pending", [False, True])
async def test_actual_native_next_updates_match_uninterrupted(factory, amp, pending):
    torch.manual_seed(24)
    original = factory("continue", amp=amp)
    accumulate(original, [1, 2])
    accumulate(original, [2, 3])
    if pending:
        accumulate(original, [3, 2])
    await original.save_checkpoint(checkpoint_name="saved")
    path = Path(original.config.output_dir) / "saved"
    expected_random = (random.random(), np.random.random(), torch.rand(3))
    target = factory("continued", amp=amp)
    assert target.load_checkpoint(path)
    actual_random = (random.random(), np.random.random(), torch.rand(3))
    assert_equal(actual_random, expected_random)
    for sample in ([2, 1], [1, 4], [4, 2]):
        accumulate(original, sample)
        accumulate(target, sample)
    assert_equal(snapshot(target), snapshot(original))


async def test_cached_native_optimizer_and_gradients_are_owned(factory, monkeypatch):
    from stateset_agents.training import checkpoint_state

    source = factory("cached")
    accumulate(source, [1, 2])
    accumulate(source, [2, 3])
    accumulate(source, [3, 2])
    await source.save_checkpoint(checkpoint_name="saved")
    path = Path(source.config.output_dir) / "saved"
    cached = torch.load(path / "training_state.pt", weights_only=True)
    before = copy.deepcopy(cached)
    original = checkpoint_state.load_checkpoint_file

    def loader(file, **kwargs):
        return (
            cached
            if Path(file).name == "training_state.pt"
            else original(file, **kwargs)
        )

    monkeypatch.setattr(checkpoint_state, "load_checkpoint_file", loader)
    target = factory("cached-target")
    assert target.load_checkpoint(path)
    for parameter in target.agent.model.parameters():
        if parameter.grad is not None:
            parameter.grad.add_(0.1)
    accumulate(target, [2, 4])
    assert_equal(cached, before)


async def test_model_only_import_returns_false_and_keeps_learner_quarantined(
    factory, tmp_path
):
    original = factory("model-original")
    path = tmp_path / "model-only"
    path.mkdir()
    original.agent.model.save_pretrained(path)
    target = factory("model-target")
    assert target.load_checkpoint(path) is False
    assert_equal(target.agent.model.state_dict(), original.agent.model.state_dict())
    assert target._runtime_checkpoint_poisoned
    with pytest.raises(RuntimeError, match="quarantined"):
        await target.train()
    # Deliberate warm start creates fresh optimizer/counters around imported
    # agent weights; it does not certify the previous learner as resumed.
    fresh = factory("fresh")
    fresh.agent.model.load_state_dict(target.agent.model.state_dict())
    accumulate(fresh, [1, 2])


async def test_nonfinite_published_policy_is_quarantined(saved, factory, tmp_path):
    _, path = saved

    def mutate(state, staging):
        file = staging / "pytorch_model.bin"
        weights = torch.load(file, weights_only=True)
        weights["weight"].fill_(float("nan"))
        torch.save(weights, file)

    bad = tmp_path / "nan-policy"
    republish(path, bad, mutate)
    target = factory("nan-target")
    with pytest.raises(ValueError, match="finite"):
        target.load_checkpoint(bad)
    assert target._runtime_checkpoint_poisoned
    with pytest.raises(RuntimeError, match="quarantined"):
        await target.save_checkpoint(checkpoint_name="blocked")


@pytest.mark.parametrize("failure", ["policy", "optimizer"])
async def test_native_invalid_live_state_cannot_publish(saved, failure):
    trainer, _ = saved
    if failure == "policy":
        with torch.no_grad():
            trainer.agent.model.weight.fill_(float("nan"))
    else:
        next(iter(trainer.optimizer.state.values()))["momentum_buffer"].fill_(
            float("nan")
        )
    with pytest.raises(ValueError, match="finite"):
        await trainer.save_checkpoint(checkpoint_name="invalid")
    assert not (Path(trainer.config.output_dir) / "invalid").exists()
