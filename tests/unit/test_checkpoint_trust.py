"""Checkpoint loads must default to ``weights_only=True``.

Every ``torch.load`` in the package is expected to refuse arbitrary pickled
payloads unless the caller explicitly opts in with ``trusted=True``.  These
tests pin that contract for the four checkpoint loaders that carry trainer
state (two sites each in the single-/multi-turn checkpointing helpers, plus
``ValueFunction.load``).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training.multi_turn_checkpointing import (  # noqa: E402
    load_multi_turn_checkpoint,
)
from stateset_agents.training.single_turn_checkpointing import (  # noqa: E402
    load_checkpoint_artifacts,
)

TRAINER_EXCEPTIONS = (
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    KeyError,
    OSError,
)


def _evil_payload() -> dict[str, object]:
    """A checkpoint dict that only a full (unsafe) unpickle can restore."""
    return {"global_step": 7, "callback": pickle.loads, "tensor": torch.zeros(2)}


def _make_trainer() -> SimpleNamespace:
    return SimpleNamespace(
        agent=SimpleNamespace(model=None, tokenizer=None),
        global_step=0,
        current_epoch=0,
        best_eval_metric=0.0,
        steps_without_improvement=0,
        _grad_accum_step=0,
        _current_task_id=None,
        optimizer=None,
        lr_scheduler=None,
        continual_manager=None,
        config=None,
    )


def _single_turn_loader(trainer, path, *, trusted):
    import logging

    return load_checkpoint_artifacts(
        trainer,
        path,
        lambda: torch,
        TRAINER_EXCEPTIONS,
        logging.getLogger("test"),
        trusted=trusted,
    )


def _multi_turn_loader(trainer, path, *, trusted):
    return load_multi_turn_checkpoint(
        trainer,
        path,
        require_torch_fn=lambda: torch,
        trainer_exceptions=TRAINER_EXCEPTIONS,
        trusted=trusted,
    )


CHECKPOINT_LOADERS = [
    pytest.param(_single_turn_loader, id="single_turn_checkpointing"),
    pytest.param(_multi_turn_loader, id="multi_turn_checkpointing"),
]


@pytest.mark.parametrize("loader", CHECKPOINT_LOADERS)
def test_training_state_load_rejects_pickled_payload_by_default(loader, tmp_path):
    torch.save(_evil_payload(), tmp_path / "training_state.pt")
    trainer = _make_trainer()

    with pytest.raises(pickle.UnpicklingError):
        loader(trainer, tmp_path, trusted=False)


@pytest.mark.parametrize("loader", CHECKPOINT_LOADERS)
def test_training_state_load_accepts_pickled_payload_when_trusted(loader, tmp_path):
    torch.save(_evil_payload(), tmp_path / "training_state.pt")
    trainer = _make_trainer()

    assert loader(trainer, tmp_path, trusted=True) is True
    assert trainer.global_step == 7


@pytest.mark.parametrize("loader", CHECKPOINT_LOADERS)
def test_model_weights_load_rejects_pickled_payload_by_default(loader, tmp_path):
    """The ``pytorch_model.bin`` site is hardened too (second site per file)."""
    torch.save(_evil_payload(), tmp_path / "pytorch_model.bin")
    trainer = _make_trainer()
    trainer.agent.model = SimpleNamespace(load_state_dict=lambda *a, **k: None)

    with pytest.raises(pickle.UnpicklingError):
        loader(trainer, tmp_path, trusted=False)


@pytest.mark.parametrize("loader", CHECKPOINT_LOADERS)
def test_model_weights_load_accepts_pickled_payload_when_trusted(loader, tmp_path):
    seen: dict[str, object] = {}
    torch.save(_evil_payload(), tmp_path / "pytorch_model.bin")
    trainer = _make_trainer()
    trainer.agent.model = SimpleNamespace(
        load_state_dict=lambda sd, **k: seen.update(sd)
    )

    loader(trainer, tmp_path, trusted=True)
    assert seen["global_step"] == 7


def _value_function():
    from stateset_agents.core.value_function import ValueFunction

    vf = ValueFunction.__new__(ValueFunction)
    vf.value_head = torch.nn.Linear(2, 1)
    vf.optimizer = torch.optim.Adam(vf.value_head.parameters())
    vf.gamma = 0.0
    vf.gae_lambda = 0.0
    return vf


def test_value_function_load_rejects_pickled_payload_by_default(tmp_path):
    vf = _value_function()
    path = str(tmp_path / "vf.pt")
    vf.save(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["callback"] = pickle.loads
    torch.save(checkpoint, path)

    with pytest.raises(pickle.UnpicklingError):
        _value_function().load(path)


def test_value_function_load_accepts_pickled_payload_when_trusted(tmp_path):
    vf = _value_function()
    vf.gamma = 0.5
    path = str(tmp_path / "vf.pt")
    vf.save(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["callback"] = pickle.loads
    torch.save(checkpoint, path)

    other = _value_function()
    other.load(path, trusted=True)
    assert other.gamma == 0.5


def test_value_function_plain_checkpoint_round_trips_untrusted(tmp_path):
    """The default (weights_only) path still restores an honest checkpoint."""
    vf = _value_function()
    vf.gamma = 0.77
    path = str(tmp_path / "vf.pt")
    vf.save(path)

    other = _value_function()
    other.load(path)
    assert other.gamma == 0.77


def test_no_unhardened_torch_load_remains():
    """Every ``torch.load`` in the package passes ``weights_only``."""
    root = Path(__file__).resolve().parents[2] / "stateset_agents"
    offenders = []
    for py in root.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for idx, chunk in enumerate(text.split("torch.load(")[1:], start=1):
            if "weights_only" not in chunk[:400]:
                offenders.append(f"{py}#{idx}")
    assert offenders == []
