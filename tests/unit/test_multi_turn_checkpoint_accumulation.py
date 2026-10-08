"""Pending gradients and AMP state must survive a real checkpoint round trip."""

import copy
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer
from stateset_agents.training.single_turn_trainer import SingleTurnGRPOTrainer


class TinyModel(torch.nn.Linear):
    def __init__(self):
        super().__init__(2, 1, bias=False)
        self.unused = torch.nn.Parameter(torch.ones(1))

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path / "pytorch_model.bin")


def trainer(path, *, trainer_type=MultiTurnGRPOTrainer, amp=False, steps=2):
    value = trainer_type(
        SimpleNamespace(
            model=TinyModel(), tokenizer=SimpleNamespace(save_pretrained=lambda _: None)
        ),
        object(),
        config=SimpleNamespace(
            output_dir=str(path), gradient_accumulation_steps=steps, max_grad_norm=10.0
        ),
    )
    value.optimizer = torch.optim.SGD(
        value.agent.model.parameters(), lr=0.1, momentum=0.9
    )
    value.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        value.optimizer, lambda step: 0.9**step
    )
    value.scaler = (
        torch.amp.GradScaler("cpu", init_scale=128, growth_interval=2) if amp else None
    )
    return value


@pytest.fixture(
    params=[MultiTurnGRPOTrainer, SingleTurnGRPOTrainer], ids=["multi", "single"]
)
def trainer_factory(request):
    return partial(trainer, trainer_type=request.param)


def accumulate(value, x):
    loss = value.agent.model(torch.tensor([x], dtype=torch.float32)).square().sum() / 2
    if value.scaler is not None:
        value.scaler.scale(loss).backward()
    else:
        loss.backward()
    value._grad_accum_step += 1
    value.current_epoch += 1
    if value._grad_accum_step % value._get_grad_accum_steps() == 0:
        if isinstance(value, SingleTurnGRPOTrainer):
            value._apply_optimizer_step(torch, value.scaler is not None, 10.0)
        else:
            value._apply_optimizer_step(torch)


def write_unmanifested_state(path, state):
    state.pop("checkpoint_publication", None)
    (path.parent / ".stateset-checkpoint.json").unlink(missing_ok=True)
    torch.save(state, path)


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_equal(a, b)
    else:
        assert left == right


@pytest.mark.asyncio
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("pending", [False, True])
async def test_real_checkpoint_resume_matches_uninterrupted_updates(
    tmp_path, trainer_factory, amp, pending
):
    original = trainer_factory(tmp_path, amp=amp)
    accumulate(original, [1, 2])
    accumulate(original, [2, 3])
    if pending:
        accumulate(original, [3, 1])
    await original.save_checkpoint(checkpoint_name="saved")
    checkpoint = tmp_path / "saved"
    raw = torch.load(checkpoint / "training_state.pt", weights_only=True)
    assert raw["gradient_accumulation"]["steps"] == 2
    assert bool(raw["gradient_accumulation"]["gradients"]) is pending
    if pending:
        assert raw["gradient_accumulation"]["gradients"]["unused"] is None
    restored = trainer_factory(tmp_path / "restored", amp=amp)
    # Existing target gradients must not leak into a restored checkpoint.
    restored.agent.model(torch.ones(1, 2)).sum().backward()
    assert restored.load_checkpoint(checkpoint)
    assert_equal(original.agent.model.state_dict(), restored.agent.model.state_dict())
    for first, second in zip(
        original.agent.model.parameters(),
        restored.agent.model.parameters(),
        strict=True,
    ):
        assert_equal(first.grad, second.grad)
    assert_equal(original.optimizer.state_dict(), restored.optimizer.state_dict())
    assert_equal(original.lr_scheduler.state_dict(), restored.lr_scheduler.state_dict())
    if amp:
        assert_equal(original.scaler.state_dict(), restored.scaler.state_dict())
    for x in ([2, 4], [1, 5], [3, 2]):
        accumulate(original, x)
        accumulate(restored, x)
    assert_equal(original.agent.model.state_dict(), restored.agent.model.state_dict())
    assert_equal(original.optimizer.state_dict(), restored.optimizer.state_dict())
    assert_equal(original.lr_scheduler.state_dict(), restored.lr_scheduler.state_dict())
    assert original.global_step == restored.global_step
    if amp:
        assert_equal(original.scaler.state_dict(), restored.scaler.state_dict())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "shape",
        "dtype",
        "nan",
        "infinite",
        "extra",
        "empty",
        "none",
        "counter",
        "schedule",
        "legacy",
        "scaler",
        "completed",
    ],
)
async def test_invalid_pending_state_rejected_before_live_weights_change(
    tmp_path, trainer_factory, mutation
):
    original = trainer_factory(tmp_path)
    accumulate(original, [1, 2])
    await original.save_checkpoint(checkpoint_name="saved")
    path = tmp_path / "saved" / "training_state.pt"
    state = torch.load(path, weights_only=True)
    saved = state["gradient_accumulation"]
    gradients = saved["gradients"]
    if mutation == "missing":
        del gradients["unused"]
    elif mutation == "shape":
        gradients["weight"] = torch.zeros(3)
    elif mutation == "dtype":
        gradients["weight"] = gradients["weight"].double()
    elif mutation in ("nan", "infinite"):
        gradients["weight"].fill_(float("nan") if mutation == "nan" else float("inf"))
    elif mutation == "extra":
        gradients["foreign"] = torch.zeros(1)
    elif mutation == "empty":
        saved["gradients"] = {}
    elif mutation == "none":
        gradients["weight"] = None
    elif mutation == "counter":
        state["grad_accum_step"] = True
    elif mutation == "schedule":
        saved["steps"] = 3
    elif mutation == "legacy":
        del state["gradient_accumulation"]
    elif mutation == "scaler":
        saved["scaler"] = {"scale": float("nan")}
    elif mutation == "completed":
        state["grad_accum_step"] = 2
    write_unmanifested_state(path, state)
    restored = trainer_factory(tmp_path / "restored")
    accumulate(restored, [4, 3])
    before = copy.deepcopy(restored.agent.model.state_dict())
    grads = [
        p.grad.clone() if p.grad is not None else None
        for p in restored.agent.model.parameters()
    ]
    with pytest.raises(ValueError):
        restored.load_checkpoint(path.parent)
    assert_equal(restored.agent.model.state_dict(), before)
    for parameter, gradient in zip(
        restored.agent.model.parameters(), grads, strict=True
    ):
        assert_equal(parameter.grad, gradient)
    assert restored._grad_accum_step == 1 and restored.global_step == 0


@pytest.mark.asyncio
async def test_legacy_completed_window_loads_and_clears_old_gradients(
    tmp_path, trainer_factory
):
    original = trainer_factory(tmp_path)
    accumulate(original, [1, 2])
    accumulate(original, [2, 1])
    await original.save_checkpoint(checkpoint_name="saved")
    path = tmp_path / "saved" / "training_state.pt"
    state = torch.load(path, weights_only=True)
    del state["gradient_accumulation"]
    write_unmanifested_state(path, state)
    restored = trainer_factory(tmp_path / "restored")
    accumulate(restored, [4, 3])
    assert restored.load_checkpoint(path.parent)
    assert all(p.grad is None for p in restored.agent.model.parameters())
    assert_equal(original.agent.model.state_dict(), restored.agent.model.state_dict())


@pytest.mark.asyncio
async def test_invalid_pending_save_does_not_overwrite_checkpoint(
    tmp_path, trainer_factory
):
    value = trainer_factory(tmp_path)
    accumulate(value, [1, 2])
    await value.save_checkpoint(checkpoint_name="saved")
    before = {p: p.read_bytes() for p in (tmp_path / "saved").iterdir()}
    value.agent.model.weight.grad.fill_(float("nan"))
    with pytest.raises(ValueError, match="gradient"):
        await value.save_checkpoint(checkpoint_name="saved")
    assert {p: p.read_bytes() for p in (tmp_path / "saved").iterdir()} == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "missing_weights",
        "wrong_weights",
        "missing_optimizer",
        "empty_optimizer",
        "invalid_optimizer",
        "missing_scheduler",
        "empty_scheduler",
        "absent_optimizer",
    ],
)
async def test_new_checkpoint_cannot_claim_resume_without_complete_components(
    tmp_path, trainer_factory, mutation
):
    original = trainer_factory(tmp_path)
    accumulate(original, [1, 2])
    await original.save_checkpoint(checkpoint_name="saved")
    path = tmp_path / "saved"
    state = torch.load(path / "training_state.pt", weights_only=True)
    restored = trainer_factory(tmp_path / "restored")
    if mutation == "missing_weights":
        (path / "pytorch_model.bin").unlink()
    elif mutation == "wrong_weights":
        torch.save({"unrelated": torch.ones(1)}, path / "pytorch_model.bin")
    elif mutation == "absent_optimizer":
        restored.optimizer = None
    else:
        key = (
            "scheduler_state_dict"
            if "scheduler" in mutation
            else "optimizer_state_dict"
        )
        if mutation.startswith("missing"):
            del state[key]
        elif mutation.startswith("empty"):
            state[key] = {}
        else:
            state[key] = {"state": {}, "param_groups": []}
    write_unmanifested_state(path / "training_state.pt", state)
    with pytest.raises((ValueError, RuntimeError)):
        restored.load_checkpoint(path)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation", ["disabled", "absent", "scale", "growth", "tracker", "overflow"]
)
async def test_amp_scaler_mismatch_is_rejected_before_loading_weights(
    tmp_path, trainer_factory, mutation
):
    original = trainer_factory(tmp_path, amp=True)
    accumulate(original, [1, 2])
    await original.save_checkpoint(checkpoint_name="saved")
    path = tmp_path / "saved" / "training_state.pt"
    state = torch.load(path, weights_only=True)
    scaler = state["gradient_accumulation"]["scaler"]
    if mutation in ("disabled", "absent"):
        state["gradient_accumulation"]["scaler"] = (
            {} if mutation == "disabled" else None
        )
    elif mutation == "scale":
        scaler["scale"] = float("nan")
    elif mutation == "growth":
        scaler["growth_factor"] = 1
    elif mutation == "tracker":
        scaler["_growth_tracker"] = True
    else:
        scaler["scale"] = 10**400
    write_unmanifested_state(path, state)
    restored = trainer_factory(tmp_path / "restored", amp=True)
    before = copy.deepcopy(restored.agent.model.state_dict())
    with pytest.raises(ValueError):
        restored.load_checkpoint(path.parent)
    assert_equal(before, restored.agent.model.state_dict())


@pytest.mark.asyncio
@pytest.mark.parametrize("artifact", ["model", "tokenizer", "training_state"])
async def test_failed_save_propagates_without_success_callback(
    tmp_path, trainer_factory, monkeypatch, artifact
):
    from unittest.mock import AsyncMock

    value = trainer_factory(tmp_path)
    await value.save_checkpoint(checkpoint_name="failed")
    before = {item.name: item.read_bytes() for item in (tmp_path / "failed").iterdir()}
    with torch.no_grad():
        value.agent.model.weight.add_(1)
    callback = SimpleNamespace(on_checkpoint_saved=AsyncMock(), fail_on_error=True)
    value.callbacks = [callback]
    error = OSError("checkpoint storage unavailable")

    def fail(*args, **kwargs):
        raise error

    if artifact == "training_state":
        original_save = torch.save

        def save(payload, destination, *args, **kwargs):
            if destination.name == "training_state.pt":
                raise error
            return original_save(payload, destination, *args, **kwargs)

        monkeypatch.setattr(torch, "save", save)
    else:
        monkeypatch.setattr(getattr(value.agent, artifact), "save_pretrained", fail)
    with pytest.raises(OSError) as caught:
        await value.save_checkpoint(checkpoint_name="failed")
    assert caught.value is error
    callback.on_checkpoint_saved.assert_not_awaited()
    assert {
        item.name: item.read_bytes() for item in (tmp_path / "failed").iterdir()
    } == before
    restored = trainer_factory(tmp_path / "restored")
    assert restored.load_checkpoint(tmp_path / "failed")
    assert not torch.equal(value.agent.model.weight, restored.agent.model.weight)


@pytest.mark.asyncio
async def test_checkpoint_requires_model_but_allows_no_tokenizer(
    tmp_path, trainer_factory
):
    value = trainer_factory(tmp_path)
    value.agent.tokenizer = None
    await value.save_checkpoint(checkpoint_name="saved")
    restored = trainer_factory(tmp_path / "restored")
    restored.agent.tokenizer = None
    assert restored.load_checkpoint(tmp_path / "saved")
    assert_equal(value.agent.model.state_dict(), restored.agent.model.state_dict())
    value.agent.model = None
    with pytest.raises(ValueError, match="without a model"):
        await value.save_checkpoint(checkpoint_name="invalid")
    assert not (tmp_path / "invalid" / "training_state.pt").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("amp", [False, True])
async def test_single_turn_final_flush_checkpoint_cannot_repeat_update(tmp_path, amp):
    from unittest.mock import AsyncMock, MagicMock

    value = trainer(tmp_path, trainer_type=SingleTurnGRPOTrainer, amp=amp)
    # Warm momentum and then leave one batch pending at the end of all episodes.
    accumulate(value, [1, 2])
    accumulate(value, [2, 3])
    accumulate(value, [3, 1])
    await value.save_checkpoint(checkpoint_name="pending")
    value.config.num_episodes = value.current_epoch + 1
    value.config.fp16 = amp
    value.config.resume_from_checkpoint = str(tmp_path / "pending")
    value._setup_scheduler = MagicMock()
    value.environment = SimpleNamespace(reset=AsyncMock())
    await value.train()
    assert value.global_step == 2
    assert value._grad_accum_step == 0
    value.environment.reset.assert_not_awaited()
    await value.save_checkpoint(checkpoint_name="finished")
    restored = trainer(tmp_path, trainer_type=SingleTurnGRPOTrainer, amp=amp)
    restored.config.num_episodes = value.config.num_episodes
    restored.config.fp16 = amp
    restored.config.resume_from_checkpoint = str(tmp_path / "finished")
    restored._setup_scheduler = MagicMock()
    restored.environment = SimpleNamespace(reset=AsyncMock())
    await restored.train()
    restored.environment.reset.assert_not_awaited()
    assert restored._grad_accum_step == 0
    assert restored.global_step == value.global_step
    assert_equal(value.agent.model.state_dict(), restored.agent.model.state_dict())
    assert_equal(value.optimizer.state_dict(), restored.optimizer.state_dict())
    assert_equal(value.lr_scheduler.state_dict(), restored.lr_scheduler.state_dict())
    if amp:
        assert_equal(value.scaler.state_dict(), restored.scaler.state_dict())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation", ["weights", "state", "missing_marker", "missing_state", "extra"]
)
async def test_publication_validation_precedes_live_state_mutation(
    tmp_path, trainer_factory, mutation
):
    original = trainer_factory(tmp_path)
    accumulate(original, [1, 2])
    await original.save_checkpoint(checkpoint_name="saved")
    path = tmp_path / "saved"
    if mutation == "missing_marker":
        (path / ".stateset-checkpoint.json").unlink()
    elif mutation == "missing_state":
        (path / "training_state.pt").unlink()
    elif mutation == "extra":
        (path / "extra.bin").write_bytes(b"stale")
    elif mutation == "weights":
        torch.save(original.agent.model.state_dict(), path / "pytorch_model.bin")
        with (path / "pytorch_model.bin").open("ab") as stream:
            stream.write(b"changed")
    else:
        state = torch.load(path / "training_state.pt", weights_only=True)
        state["global_step"] = 99
        torch.save(state, path / "training_state.pt")
    restored = trainer_factory(tmp_path / "restored")
    accumulate(restored, [4, 3])
    before = copy.deepcopy(restored.agent.model.state_dict())
    grads = [
        p.grad.clone() if p.grad is not None else None
        for p in restored.agent.model.parameters()
    ]
    with pytest.raises(ValueError, match="publication"):
        restored.load_checkpoint(path)
    assert_equal(before, restored.agent.model.state_dict())
    assert restored.global_step == 0 and restored._grad_accum_step == 1
    for parameter, gradient in zip(
        restored.agent.model.parameters(), grads, strict=True
    ):
        assert_equal(parameter.grad, gradient)
