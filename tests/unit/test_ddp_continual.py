"""Task boundaries must budget actual updates and retain qualified receipts."""

import copy
from types import SimpleNamespace

import pytest

from stateset_agents.training.config import estimate_grpo_optimizer_steps
from stateset_agents.training.continual_learning import (
    ContinualLearningConfig,
    ContinualLearningManager,
)


@pytest.mark.parametrize(
    "tasks,expected",
    [
        (["A", "B", "A", "B", "A"], 5),
        ([None, "A", None, "B", "A"], 4),
        (["A", "A", "B", "B", "A"], 3),
        ([None] * 5, 3),
        ([], 0),
    ],
)
def test_schedule_accounts_for_each_task_window(tasks, expected):
    assert (
        estimate_grpo_optimizer_steps(
            len(tasks), gradient_accumulation_steps=2, task_ids=tasks
        )
        == expected
    )
    assert (
        estimate_grpo_optimizer_steps(
            len(tasks),
            gradient_accumulation_steps=2,
            num_gradient_updates=3,
            task_ids=tasks,
        )
        == len(tasks) * 3
    )


@pytest.mark.parametrize("tasks", [["A"], ["A", 1]])
def test_invalid_task_schedule_rejects(tasks):
    with pytest.raises(ValueError):
        estimate_grpo_optimizer_steps(2, task_ids=tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize("inner", [1, 3])
@pytest.mark.parametrize("warmup", [0.0, 0.25])
async def test_real_task_windows_use_budget_and_complete_resume_is_idempotent(
    monkeypatch, tmp_path, inner, warmup
):
    from tests.unit.test_grpo_scheduler_budget import (
        assert_equal,
        make_trainer,
        schedule_rate,
    )

    trainer, rates = make_trainer(
        monkeypatch, tmp_path, "multi", inner, "linear", warmup
    )
    trainer.continual_manager = ContinualLearningManager(
        ContinualLearningConfig(strategy="lwf")
    )
    trainer.compute_enhanced_grpo_loss = trainer.compute_grpo_loss
    trainer._get_training_scenarios = lambda: [
        {"task_id": task} for task in ["A", "B", "A", "B", "A"]
    ]
    await trainer.train()
    horizon = 5 * inner
    assert trainer.global_step == horizon
    assert rates == pytest.approx(
        [schedule_rate(step, horizon, warmup, "linear") for step in range(horizon)]
    )
    assert trainer.continual_manager._task_boundary == {
        "task_id": "A",
        "global_step": horizon,
        "buffer_sequence": 0,
    }
    before = copy.deepcopy(trainer.continual_manager.state_dict())
    trainer.config.resume_from_checkpoint = str(tmp_path / f"checkpoint-{horizon}")
    await trainer.train()
    assert trainer.global_step == horizon
    assert len(rates) == horizon
    assert_equal(trainer.continual_manager.state_dict(), before)


@pytest.mark.parametrize(
    "boundary",
    [
        {"task_id": "A", "global_step": True, "buffer_sequence": 0},
        {"task_id": "A", "global_step": 1, "buffer_sequence": 1},
        {"task_id": 1, "global_step": 1, "buffer_sequence": 0},
    ],
)
def test_bad_boundary_receipt_rejects_before_manager_mutation(boundary):
    manager = ContinualLearningManager(
        ContinualLearningConfig(strategy="replay"), seed=7
    )
    before = copy.deepcopy(manager.state_dict())
    saved = copy.deepcopy(before)
    saved["task_boundary"] = boundary
    with pytest.raises(ValueError):
        manager.load_state_dict(saved)
    assert manager.state_dict() == before


def test_snapshot_failure_preserves_previous_reference(monkeypatch):
    torch = pytest.importorskip("torch")
    from stateset_agents.training import continual_learning

    manager = ContinualLearningManager(ContinualLearningConfig(strategy="lwf"))
    model = torch.nn.Linear(2, 1)
    manager._snapshot_reference_model(SimpleNamespace(model=model))
    reference = manager.reference_model

    def failed(_model):
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(continual_learning, "deepcopy", failed)
    with pytest.raises(RuntimeError, match="snapshot failed"):
        manager._snapshot_reference_model(SimpleNamespace(model=model))
    assert manager.reference_model is reference


def test_checkpoint_without_reference_clears_a_stale_teacher(tmp_path):
    pytest.importorskip("torch")
    from stateset_agents.training.ddp_checkpoint_state import (
        apply_payload,
        capture_payload,
        prepare_payload,
    )
    from tests.unit.test_ddp_checkpoint_state import _learner

    original = _learner(tmp_path / "original")
    original.reference_model = None
    saved, local, _ = copy.deepcopy(capture_payload(original))
    restored = _learner(tmp_path / "restored")
    assert restored.reference_model is not None
    apply_payload(restored, prepare_payload(restored, saved, local))
    assert restored.reference_model is None


def test_fast_tokenizer_identity_ignores_per_call_state_but_binds_vocab_and_template():
    from stateset_agents.training.ddp_checkpoint_state import _tokenizer_identity
    from tests.integration.test_ddp_continual import _tokenizer

    tokenizer = _tokenizer()
    original = _tokenizer_identity(tokenizer)
    tokenizer.backend_tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
    tokenizer.backend_tokenizer.enable_truncation(5)
    assert _tokenizer_identity(tokenizer) == original
    tokenizer.chat_template += " "
    assert _tokenizer_identity(tokenizer) != original
    tokenizer.chat_template = tokenizer.chat_template[:-1]
    tokenizer.add_tokens(["new"])
    assert _tokenizer_identity(tokenizer) != original
