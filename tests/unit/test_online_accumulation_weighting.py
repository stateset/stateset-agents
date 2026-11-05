"""Online accumulated updates must match a combined rollout-batch reference."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training.multi_turn_trainer import (  # noqa: E402
    MultiTurnGRPOTrainer,
)
from stateset_agents.training.single_turn_trainer import (  # noqa: E402
    SingleTurnGRPOTrainer,
)
from tests.unit.test_token_group_aggregation import (  # noqa: E402
    TinyPolicy,
    _config,
    _groups,
    _reference,
)


class _Config(SimpleNamespace):
    def to_dict(self):
        return vars(self).copy()


def _trainer(tmp_path, kind, name, entropy, steps):
    config = _config(
        name,
        entropy_coef=entropy,
        gradient_accumulation_steps=steps,
        max_grad_norm=100.0,
        bf16=False,
        fp16=False,
        beta=0.0,
        num_episodes=2,
        max_steps_per_episode=1,
        num_generations=3,
        zero_signal_patience=0,
        output_dir=str(tmp_path),
    )
    environment = SimpleNamespace(
        reset=AsyncMock(return_value={"prompt": "logged"}),
        step=AsyncMock(return_value={"state": {}, "done": True}),
    )
    trainer = kind(
        SimpleNamespace(model=TinyPolicy()), environment, config=_Config(**vars(config))
    )
    trainer.optimizer = torch.optim.SGD(trainer.agent.model.parameters(), lr=0.1)
    trainer._setup_scheduler = lambda _: None
    return trainer


async def _run(trainer, groups):
    if isinstance(trainer, MultiTurnGRPOTrainer):
        for group in groups:
            await trainer.training_step([group])
        trainer._flush_accumulation(torch)
    else:
        trainer.config.num_episodes = len(groups)
        trainer._get_episode_scenario = lambda _: {}
        trainer._generate_trajectory_group = AsyncMock(
            side_effect=[(group, "logged") for group in groups]
        )
        await trainer.train()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [MultiTurnGRPOTrainer, SingleTurnGRPOTrainer])
@pytest.mark.parametrize("name", ["grpo", "dr_grpo", "dapo", "cispo", "rloo"])
@pytest.mark.parametrize("entropy", [0.0, 0.07])
@pytest.mark.parametrize("steps", [2, 4], ids=["complete", "partial"])
async def test_accumulated_updates_match_combined_reference(
    tmp_path, kind, name, entropy, steps
):
    trainer = _trainer(tmp_path, kind, name, entropy, steps)
    reference = TinyPolicy()
    groups = _groups()
    loss, _ = _reference(groups, reference, trainer.config)
    loss.backward()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.add_(parameter.grad, alpha=-0.1)
    await _run(trainer, groups)
    assert trainer.global_step == 1
    for actual, expected in zip(
        trainer.agent.model.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [MultiTurnGRPOTrainer, SingleTurnGRPOTrainer])
async def test_equal_batches_are_a_control(tmp_path, kind):
    trainer = _trainer(tmp_path, kind, "grpo", 0.07, 2)
    groups = [_groups()[0], _groups()[0]]
    reference = TinyPolicy()
    loss, _ = _reference(groups, reference, trainer.config)
    loss.backward()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.add_(parameter.grad, alpha=-0.1)
    await _run(trainer, groups)
    torch.testing.assert_close(
        trainer.agent.model.lookup.weight, reference.lookup.weight, rtol=1e-6, atol=1e-7
    )


async def _pending_batch(trainer, group):
    if isinstance(trainer, MultiTurnGRPOTrainer):
        await trainer.training_step([group])
    else:
        from stateset_agents.training.grpo_accumulation import backward_accumulated_loss
        from stateset_agents.training.loss_computation import compute_enhanced_grpo_loss

        metrics = compute_enhanced_grpo_loss(
            [group], 0.0, trainer.config, trainer.agent
        )
        backward_accumulated_loss(
            trainer, metrics["total_loss"], metrics, torch, scaler=trainer.scaler
        )
        trainer._grad_accum_step += 1


def _enable_amp(trainer, enabled):
    if enabled:
        trainer.scaler = torch.amp.GradScaler("cpu", init_scale=128, growth_interval=2)
        trainer.config.bf16 = True


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [MultiTurnGRPOTrainer, SingleTurnGRPOTrainer])
@pytest.mark.parametrize("steps", [2, 4])
@pytest.mark.parametrize("entropy", [0.0, 0.07])
@pytest.mark.parametrize("amp", [False, True])
async def test_weighted_checkpoint_resume_matches_combined_update(
    tmp_path, kind, steps, entropy, amp
):
    original = _trainer(tmp_path, kind, "grpo", entropy, steps)
    original.agent.model.save_pretrained = lambda path: torch.save(
        original.agent.model.state_dict(), path / "pytorch_model.bin"
    )
    _enable_amp(original, amp)
    groups = _groups()
    await _pending_batch(original, groups[0])
    await original.save_checkpoint(checkpoint_name="pending")
    saved = torch.load(tmp_path / "pending/training_state.pt", weights_only=True)
    accumulation = saved["gradient_accumulation"]
    assert accumulation["schema_version"] == 2
    assert accumulation["normalization"]["policy_count"] == 2
    restored = _trainer(tmp_path / "restored", kind, "grpo", entropy, steps)
    _enable_amp(restored, amp)
    with torch.no_grad():
        restored.agent.model.lookup.weight.add_(1)
    assert restored.load_checkpoint(tmp_path / "pending")
    await _run(original, groups[1:])
    await _run(restored, groups[1:])
    reference = TinyPolicy()
    loss, _ = _reference(groups, reference, original.config)
    loss.backward()
    expected = reference.lookup.weight - 0.1 * reference.lookup.weight.grad
    for trainer in (original, restored):
        torch.testing.assert_close(
            trainer.agent.model.lookup.weight, expected, rtol=1e-6, atol=1e-7
        )
        assert trainer.global_step == 1
        assert trainer._grpo_accumulation is None
    torch.testing.assert_close(
        original.agent.model.lookup.weight,
        restored.agent.model.lookup.weight,
        rtol=0,
        atol=0,
    )
    if amp:
        assert original.scaler.state_dict() == restored.scaler.state_dict()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [MultiTurnGRPOTrainer, SingleTurnGRPOTrainer])
@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "count_zero",
        "count_bool",
        "scale_zero",
        "scale_fraction",
        "entropy_count",
        "entropy_missing",
        "entropy_nan",
        "entropy_shape",
        "measure",
        "measure_type",
        "extra",
    ],
)
async def test_invalid_weighted_checkpoint_preserves_live_state(
    tmp_path, kind, mutation
):
    import copy

    from tests.unit.test_multi_turn_checkpoint_accumulation import (
        assert_equal,
        write_unmanifested_state,
    )

    original = _trainer(tmp_path, kind, "grpo", 0.07, 4)
    original.agent.model.save_pretrained = lambda path: torch.save(
        original.agent.model.state_dict(), path / "pytorch_model.bin"
    )
    await _pending_batch(original, _groups()[0])
    await original.save_checkpoint(checkpoint_name="pending")
    path = tmp_path / "pending/training_state.pt"
    state = torch.load(path, weights_only=True)
    normalized = state["gradient_accumulation"]["normalization"]
    if mutation == "missing":
        del state["gradient_accumulation"]["normalization"]
    elif mutation == "count_zero":
        normalized["policy_count"] = 0
    elif mutation == "count_bool":
        normalized["policy_count"] = True
    elif mutation == "scale_zero":
        normalized["policy_normalizer"] = 0
    elif mutation == "scale_fraction":
        normalized["policy_normalizer"] = 1.5
    elif mutation == "entropy_count":
        normalized["entropy_count"] = 0
    elif mutation == "entropy_missing":
        normalized["entropy_gradients"] = {}
    elif mutation == "entropy_nan":
        normalized["entropy_gradients"]["lookup.weight"].fill_(float("nan"))
    elif mutation == "entropy_shape":
        normalized["entropy_gradients"]["lookup.weight"] = torch.zeros(1)
    elif mutation == "measure":
        normalized["measure"] = "token"
    elif mutation == "measure_type":
        normalized["measure"] = []
    else:
        normalized["extra"] = 1
    write_unmanifested_state(path, state)
    restored = _trainer(tmp_path / "restored", kind, "grpo", 0.07, 4)
    await _pending_batch(restored, _groups()[1])
    weights = copy.deepcopy(restored.agent.model.state_dict())
    gradients = [p.grad.clone() for p in restored.agent.model.parameters()]
    normalization = copy.deepcopy(restored._grpo_accumulation)
    with pytest.raises(ValueError):
        restored.load_checkpoint(path.parent)
    assert_equal(restored.agent.model.state_dict(), weights)
    assert_equal(restored._grpo_accumulation, normalization)
    for parameter, gradient in zip(
        restored.agent.model.parameters(), gradients, strict=True
    ):
        torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
    assert restored._grad_accum_step == 1 and restored.global_step == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [MultiTurnGRPOTrainer, SingleTurnGRPOTrainer])
@pytest.mark.parametrize("amp", [False, True])
async def test_weighted_partial_window_normalizes_before_clipping(tmp_path, kind, amp):
    trainer = _trainer(tmp_path, kind, "grpo", 0.07, 4)
    trainer.config.max_grad_norm = 0.05
    _enable_amp(trainer, amp)
    reference = TinyPolicy()
    groups = _groups()
    loss, _ = _reference(groups, reference, trainer.config)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.05)
    expected = reference.lookup.weight - 0.1 * reference.lookup.weight.grad
    await _run(trainer, groups)
    torch.testing.assert_close(
        trainer.agent.model.lookup.weight, expected, rtol=1e-6, atol=1e-7
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("amp", [False, True])
async def test_auxiliary_overflow_discards_whole_window(tmp_path, amp):
    trainer = _trainer(tmp_path, MultiTurnGRPOTrainer, "grpo", 0.07, 4)
    _enable_amp(trainer, amp)
    before = trainer.agent.model.lookup.weight.detach().clone()
    await _pending_batch(trainer, _groups()[0])
    trainer._grpo_accumulation["entropy_gradients"]["lookup.weight"].fill_(float("inf"))
    if amp:
        assert trainer._flush_accumulation(torch) is False
        assert trainer.scaler.get_scale() == 64
    else:
        with pytest.raises(ValueError, match="finite"):
            trainer._flush_accumulation(torch)
    torch.testing.assert_close(
        trainer.agent.model.lookup.weight, before, rtol=0, atol=0
    )
    assert trainer.global_step == 0 and trainer._grad_accum_step == 0
    assert trainer._grpo_accumulation is None
    assert trainer.agent.model.lookup.weight.grad is None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [MultiTurnGRPOTrainer, SingleTurnGRPOTrainer])
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("immediate_flush", [False, True])
async def test_shared_runtime_checkpoint_restores_both_gradient_banks(
    tmp_path, kind, amp, immediate_flush
):
    original = _trainer(tmp_path, kind, "grpo", 0.07, 4)
    original.agent.model.save_pretrained = lambda path: torch.save(
        original.agent.model.state_dict(), path / "pytorch_model.bin"
    )
    _enable_amp(original, amp)
    groups = _groups()
    await _pending_batch(original, groups[0])
    original.save_runtime_checkpoint(str(tmp_path / "runtime"))
    restored = _trainer(tmp_path / "restored", kind, "grpo", 0.07, 4)
    _enable_amp(restored, amp)
    assert restored.load_runtime_checkpoint(str(tmp_path / "runtime"))
    if immediate_flush:
        for trainer in (original, restored):
            if isinstance(trainer, SingleTurnGRPOTrainer):
                trainer._flush_accumulation(torch, amp, 100.0)
            else:
                trainer._flush_accumulation(torch)
    else:
        await _run(original, groups[1:])
        await _run(restored, groups[1:])
    torch.testing.assert_close(
        original.agent.model.lookup.weight,
        restored.agent.model.lookup.weight,
        rtol=0,
        atol=0,
    )
    assert original.global_step == restored.global_step == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [MultiTurnGRPOTrainer, SingleTurnGRPOTrainer])
@pytest.mark.parametrize(
    "field,value",
    [
        ("entropy_coef", 0.0),
        ("clip_ratio", 0.1),
        ("reward_clip", 1.0),
        ("baseline_type", "group_median"),
    ],
)
async def test_changed_loss_config_rejected_before_policy_load(
    tmp_path, kind, field, value
):
    original = _trainer(tmp_path, kind, "grpo", 0.07, 4)
    original.agent.model.save_pretrained = lambda path: torch.save(
        original.agent.model.state_dict(), path / "pytorch_model.bin"
    )
    await _pending_batch(original, _groups()[0])
    await original.save_checkpoint(checkpoint_name="pending")
    restored = _trainer(tmp_path / "restored", kind, "grpo", 0.07, 4)
    setattr(restored.config, field, value)
    before = restored.agent.model.lookup.weight.detach().clone()
    with pytest.raises(ValueError, match=field):
        restored.load_checkpoint(tmp_path / "pending")
    torch.testing.assert_close(
        restored.agent.model.lookup.weight, before, rtol=0, atol=0
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["grpo", "dapo"])
async def test_regularizer_is_preserved_with_weighted_terms(tmp_path, name):
    trainer = _trainer(tmp_path, MultiTurnGRPOTrainer, name, 0.07, 4)
    trainer.continual_manager = SimpleNamespace(
        reference_model=None,
        get_effective_beta=lambda beta: beta,
        compute_ewc_penalty=lambda agent: 0.2
        * agent.model.lookup.weight.square().mean(),
    )
    reference = TinyPolicy()
    groups = _groups()
    loss, _ = _reference(groups, reference, trainer.config)
    (loss + 0.2 * reference.lookup.weight.square().mean()).backward()
    expected = reference.lookup.weight - 0.1 * reference.lookup.weight.grad
    await _run(trainer, groups)
    torch.testing.assert_close(
        trainer.agent.model.lookup.weight, expected, rtol=1e-6, atol=1e-7
    )


@pytest.mark.asyncio
async def test_late_entropy_backward_failure_discards_all_banks(tmp_path):
    trainer = _trainer(tmp_path, MultiTurnGRPOTrainer, "grpo", 0.07, 4)
    calls = 0

    def fail_second_backward(gradient):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("entropy backward failed")
        return gradient

    handle = trainer.agent.model.lookup.weight.register_hook(fail_second_backward)
    with pytest.raises(RuntimeError, match="entropy backward failed"):
        await trainer.training_step([_groups()[0]])
    handle.remove()
    assert trainer.agent.model.lookup.weight.grad is None
    assert trainer._grpo_accumulation is None
    assert trainer._grad_accum_step == trainer.global_step == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [MultiTurnGRPOTrainer, SingleTurnGRPOTrainer])
async def test_legacy_pending_window_finishes_then_uses_count_weighting(tmp_path, kind):
    trainer = _trainer(tmp_path, kind, "grpo", 0.07, 2)
    groups = _groups()
    reference = TinyPolicy()
    first, _ = _reference([groups[0]], reference, trainer.config)
    second, _ = _reference([groups[1]], reference, trainer.config)
    ((first + second) / 2).backward()
    with torch.no_grad():
        reference.lookup.weight.add_(reference.lookup.weight.grad, alpha=-0.1)
    reference.zero_grad(set_to_none=True)
    # A v1 checkpoint's pending gradients are already divided by the step count.
    first, _ = _reference([groups[0]], trainer.agent.model, trainer.config)
    (first / 2).backward()
    trainer._grad_accum_step = 1
    await _run(trainer, groups[1:])
    torch.testing.assert_close(
        trainer.agent.model.lookup.weight, reference.lookup.weight, rtol=1e-6, atol=1e-7
    )
    combined, _ = _reference(groups, reference, trainer.config)
    combined.backward()
    with torch.no_grad():
        reference.lookup.weight.add_(reference.lookup.weight.grad, alpha=-0.1)
    await _run(trainer, groups)
    torch.testing.assert_close(
        trainer.agent.model.lookup.weight, reference.lookup.weight, rtol=1e-6, atol=1e-7
    )
    assert trainer.global_step == 2


def test_sampled_workflow_rejects_nonpositive_accumulation(tmp_path, monkeypatch):
    from stateset_agents.training import sampled_workflow

    output = tmp_path / "invalid"
    monkeypatch.setattr(
        "sys.argv",
        [
            "sampled_workflow",
            "--output",
            str(output),
            "--gradient-accumulation-steps",
            "0",
        ],
    )
    with pytest.raises(SystemExit) as error:
        sampled_workflow.main()
    assert error.value.code == 2
    assert not output.exists()
