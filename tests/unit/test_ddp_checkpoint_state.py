"""Native checkpoint validation must stage state without mutating the learner."""

import copy
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stateset_agents.core.checkpoint_io import load_checkpoint_file  # noqa: E402
from stateset_agents.training.ddp_checkpoint_state import (  # noqa: E402
    apply_payload,
    capture_payload,
    prepare_payload,
)
from tests.integration.test_learner_preflight import _same  # noqa: E402
from tests.integration.test_replica_consensus import _make, _scheduler  # noqa: E402


class _Wrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module


def _learner(directory, optimizer="sgd"):
    trainer = _make(directory)
    trainer.agent.model.register_buffer(
        "scratch", torch.tensor([3.0]), persistent=False
    )
    trainer.reference_model.register_buffer(
        "scratch", torch.tensor([4.0]), persistent=False
    )
    trainer.agent.model = _Wrapper(trainer.agent.model)
    cls = torch.optim.SGD if optimizer == "sgd" else torch.optim.AdamW
    kwargs = {"momentum": 0.9} if optimizer == "sgd" else {}
    trainer.optimizer = cls(trainer.agent.model.parameters(), lr=0.1, **kwargs)
    trainer.lr_scheduler = _scheduler(trainer.optimizer, "lambda_closure", 0.9)
    for parameter in trainer.agent.model.parameters():
        parameter.grad = torch.ones_like(parameter)
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.lr_scheduler.step()
    trainer.global_step = 1
    trainer._update_global_stats(0.75, 4)
    return trainer


@pytest.mark.parametrize("optimizer", ["sgd", "adamw"])
def test_payload_roundtrip_preserves_slots_buffers_reference_and_rng(
    tmp_path, optimizer
):
    original = _learner(tmp_path / "original", optimizer)
    original.reference_model.requires_grad_(False)
    shared, local, agreement = copy.deepcopy(capture_payload(original))
    expected_draw = (random.random(), np.random.rand(), torch.rand(3))
    restored = _learner(tmp_path / "restored", optimizer)
    restored.reference_model = None
    identities = [id(p) for p in restored.agent.model.parameters()]
    prepared = prepare_payload(restored, shared, local)
    apply_payload(restored, prepared)
    immutable_payload = copy.deepcopy(shared)
    actual_draw = (random.random(), np.random.rand(), torch.rand(3))
    _same(actual_draw, expected_draw, torch)
    assert identities == [id(p) for p in restored.agent.model.parameters()]
    assert all(not p.requires_grad for p in restored.reference_model.parameters())
    actual_shared, actual_local, actual_agreement = capture_payload(restored)
    _same(actual_shared, shared, torch)
    _same(actual_local["buffers"], local["buffers"], torch)
    assert actual_agreement == agreement
    # Prove the slots are usable, not merely serializable.
    for learner in (original, restored):
        for parameter in learner.agent.model.parameters():
            parameter.grad = torch.full_like(parameter, 0.25)
        learner.optimizer.step()
        learner.lr_scheduler.step()
    _same(original.optimizer.state_dict(), restored.optimizer.state_dict(), torch)
    _same(original.agent.model.state_dict(), restored.agent.model.state_dict(), torch)
    _same(shared, immutable_payload, torch)


@pytest.mark.parametrize("cursor", [0, 3, None])
def test_episode_cursor_restores_explicit_or_legacy_position(tmp_path, cursor):
    trainer = _learner(tmp_path)
    shared, local, _ = copy.deepcopy(capture_payload(trainer))
    local["counters"]["current_epoch"] = 2
    if cursor is None:
        del local["counters"]["_next_episode"]
    else:
        local["counters"]["_next_episode"] = cursor
    apply_payload(trainer, prepare_payload(trainer, shared, local))
    assert trainer._resume_start(True, 10) == (3 if cursor is None else cursor)


@pytest.mark.parametrize("cursor", [-1, True, 1.5, "1", 2**63])
def test_invalid_episode_cursor_is_rejected_before_mutation(tmp_path, cursor):
    trainer = _learner(tmp_path)
    shared, local, _ = copy.deepcopy(capture_payload(trainer))
    local["counters"]["_next_episode"] = cursor
    before = copy.deepcopy(capture_payload(trainer))
    with pytest.raises(ValueError, match="episode cursor"):
        prepare_payload(trainer, shared, local)
    _same(capture_payload(trainer), before, torch)


@pytest.mark.parametrize(
    "fault",
    [
        "weight_shape",
        "slot_shape",
        "rng",
        "counter",
        "baseline",
        "bindings",
        "diagnostic_counter",
        "diagnostic_latest",
        "diagnostic_patience",
    ],
)
def test_validation_failure_preserves_live_state(tmp_path, fault):
    trainer = _learner(tmp_path)
    shared, local, _ = copy.deepcopy(capture_payload(trainer))
    if fault == "weight_shape":
        shared["policy"]["lookup.weight"] = torch.zeros(1)
    elif fault == "slot_shape":
        shared["optimizer"]["state"][0]["momentum_buffer"] = torch.zeros(1)
    elif fault == "rng":
        local["rng"]["torch"] = torch.zeros(3, dtype=torch.uint8)
    elif fault == "counter":
        local["counters"]["global_step"] = True
    elif fault == "baseline":
        local["learning_state"]["reward_baseline"]["sum"] = 999
    elif fault.startswith("diagnostic_"):
        from stateset_agents.training.diagnostics import DiagnosticsMonitor

        diagnostic = local["learning_state"]["diagnostics"]
        diagnostic["monitor"] = DiagnosticsMonitor(trainer.config).state_dict()
        if fault == "diagnostic_counter":
            diagnostic["observations"] = True
        elif fault == "diagnostic_latest":
            diagnostic["monitor"]["latest"] = {"loss": float("nan")}
        else:
            diagnostic["monitor"]["patience"] += 1
    else:
        trainer.lr_scheduler = _scheduler(trainer.optimizer, "lambda_closure", 0.5)
    before = copy.deepcopy(capture_payload(trainer))
    with pytest.raises((ValueError, RuntimeError)):
        prepare_payload(trainer, shared, local)
    _same(capture_payload(trainer), before, torch)


def test_safe_loader_supports_mapped_native_payload(tmp_path):
    path = tmp_path / "state.pt"
    torch.save({"tensor": torch.arange(7)}, path)
    loaded = load_checkpoint_file(path, mmap=True)
    torch.testing.assert_close(loaded["tensor"], torch.arange(7))


@pytest.mark.parametrize("fault", ["missing", "dtype", "step"])
def test_adam_slot_validation_is_transactional(tmp_path, fault):
    trainer = _learner(tmp_path, "adamw")
    shared, local, _ = copy.deepcopy(capture_payload(trainer))
    slots = shared["optimizer"]["state"][0]
    if fault == "missing":
        del slots["exp_avg_sq"]
    elif fault == "dtype":
        slots["exp_avg"] = slots["exp_avg"].double()
    else:
        slots["step"] = torch.tensor(-1.0)
    before = copy.deepcopy(capture_payload(trainer))
    with pytest.raises(ValueError, match="slot"):
        prepare_payload(trainer, shared, local)
    _same(capture_payload(trainer), before, torch)


@pytest.mark.parametrize("strategy", ["replay", "ewc", "lwf"])
def test_continual_state_restores_wrapped_ewc_and_private_replay(tmp_path, strategy):
    from stateset_agents.training.continual_learning import (
        ContinualLearningConfig,
        ContinualLearningManager,
    )
    from tests.unit.test_continual_learning import _make_group

    original = _learner(tmp_path / "original")
    manager = ContinualLearningManager(
        ContinualLearningConfig(strategy=strategy), seed=17
    )
    original.continual_manager = manager
    manager.add_trajectory_groups([_make_group(str(i), i) for i in range(8)])
    manager._ewc_fisher = {
        name: torch.ones_like(p) for name, p in original.agent.model.named_parameters()
    }
    manager._ewc_params = {
        name: p.detach().clone() for name, p in original.agent.model.named_parameters()
    }
    if strategy == "lwf":
        manager.reference_model = original.reference_model.requires_grad_(False)
    original._current_task_id = "task-1"
    shared, local, _ = copy.deepcopy(capture_payload(original))
    restored = _learner(tmp_path / "restored")
    restored.continual_manager = ContinualLearningManager(
        copy.deepcopy(manager.config), seed=999
    )
    if strategy == "lwf":
        restored.reference_model = None
    public_manager = restored.continual_manager
    apply_payload(restored, prepare_payload(restored, shared, local))
    assert restored.continual_manager is public_manager
    _same(capture_payload(restored)[:2], (shared, local), torch)
    _same(public_manager.state_dict(), manager.state_dict(), torch)
    assert restored._current_task_id == "task-1"


def test_replay_without_a_teacher_has_no_reference_relationship(tmp_path):
    from stateset_agents.training.continual_learning import (
        ContinualLearningConfig,
        ContinualLearningManager,
    )

    trainer = _learner(tmp_path)
    trainer.config.beta = 0.0
    trainer.config.use_reference_model = False
    trainer.reference_model = None
    trainer.continual_manager = ContinualLearningManager(
        ContinualLearningConfig(strategy="replay"), seed=7
    )
    shared, local, agreement = copy.deepcopy(capture_payload(trainer))
    assert local["learning_state"]["reference_from_continual"] is False
    apply_payload(trainer, prepare_payload(trainer, shared, local))
    assert capture_payload(trainer)[2] == agreement
