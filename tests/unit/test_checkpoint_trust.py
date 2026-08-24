"""Checkpoint loads must default to ``weights_only=True``.

Every ``torch.load`` in the package is expected to refuse arbitrary pickled
payloads unless the caller explicitly opts in with ``trusted=True``.  These
tests pin that contract for the four checkpoint loaders that carry trainer
state (two sites each in the single-/multi-turn checkpointing helpers, plus
``ValueFunction.load``).
"""

from __future__ import annotations

import ast
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.core.errors import ModelError  # noqa: E402

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

    with pytest.raises(ModelError):
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

    with pytest.raises(ModelError):
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

    with pytest.raises(ModelError):
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


def _torch_load_calls(tree: ast.AST) -> list[ast.Call]:
    """Every ``*.load(...)`` call whose receiver is named ``torch``."""
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "load"
            and isinstance(func.value, ast.Name)
            and func.value.id in {"torch", "_torch", "torch_module"}
        ):
            calls.append(node)
    return calls


def test_only_checkpoint_io_calls_torch_load():
    """All checkpoint loading is funnelled through ``load_checkpoint_file``.

    Centralising it is what makes the ``weights_only`` default enforceable: a
    new ``torch.load`` anywhere else in the package fails this test.
    """
    root = Path(__file__).resolve().parents[2] / "stateset_agents"
    offenders = []
    for py in sorted(root.rglob("*.py")):
        if py.name == "checkpoint_io.py":
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for call in _torch_load_calls(tree):
            offenders.append(f"{py.relative_to(root)}:{call.lineno}")
    assert offenders == [], (
        "torch.load called outside checkpoint_io.py; route it through "
        "load_checkpoint_file(..., trusted=...) instead"
    )


def test_checkpoint_io_pins_weights_only():
    """The single ``torch.load`` site passes a non-``False`` ``weights_only``."""
    src = (
        Path(__file__).resolve().parents[2]
        / "stateset_agents"
        / "training"
        / "checkpoint_io.py"
    )
    calls = _torch_load_calls(ast.parse(src.read_text(encoding="utf-8")))
    assert len(calls) == 1
    keywords = {kw.arg: kw.value for kw in calls[0].keywords}
    assert "weights_only" in keywords
    value = keywords["weights_only"]
    assert not (isinstance(value, ast.Constant) and value.value is False)


# ---------------------------------------------------------------------------
# Round-trip coverage for the trainers whose save() now writes a plain config
# ---------------------------------------------------------------------------


def _bcq():
    from stateset_agents.training.offline_rl_bcq import (
        BatchConstrainedQLearning,
        BCQConfig,
    )

    return BatchConstrainedQLearning(
        state_dim=4, action_dim=2, config=BCQConfig(batch_size=17), device="cpu"
    )


def _bear():
    from stateset_agents.training.offline_rl_bear import BEARConfig, ConversationalBEAR

    return ConversationalBEAR(
        state_dim=4, action_dim=2, config=BEARConfig(batch_size=17), device="cpu"
    )


def _decision_transformer():
    from stateset_agents.training.decision_transformer import (
        DecisionTransformerConfig,
        DecisionTransformerTrainer,
    )

    config = DecisionTransformerConfig(
        state_dim=4,
        action_dim=2,
        n_embd=8,
        n_layer=1,
        n_head=1,
        max_context_length=4,
        batch_size=17,
        use_conversation_embeddings=False,
    )
    return DecisionTransformerTrainer(config, device="cpu")


def _sim_to_real():
    from stateset_agents.training.sim_to_real import SimToRealConfig, SimToRealTransfer

    return SimToRealTransfer(SimToRealConfig(batch_size=17), device="cpu")


def _offline_grpo():
    from stateset_agents.training.offline_grpo_trainer import (
        OfflineGRPOConfig,
        OfflineGRPOTrainer,
    )

    return OfflineGRPOTrainer(OfflineGRPOConfig(batch_size=17), device="cpu")


ROUND_TRIP_TRAINERS = [
    pytest.param(_bcq, id="offline_rl_bcq"),
    pytest.param(_bear, id="offline_rl_bear"),
    pytest.param(_decision_transformer, id="decision_transformer"),
    pytest.param(_sim_to_real, id="sim_to_real"),
    pytest.param(_offline_grpo, id="offline_grpo_trainer"),
]


@pytest.mark.parametrize("factory", ROUND_TRIP_TRAINERS)
def test_save_load_round_trip_rebuilds_plain_config(factory, tmp_path):
    """save() writes the config as a dict; load() rebuilds the dataclass.

    The default ``trusted=False`` path must work end to end — that is the whole
    point of moving the config to plain data.
    """
    saver = factory()
    saver.training_step = 11
    path = str(tmp_path / "ckpt.pt")
    saver.save(path)

    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(raw["config"], dict), "config must be persisted as plain data"

    loader = factory()
    loader.load(path)

    assert loader.training_step == 11
    assert type(loader.config) is type(saver.config)
    assert loader.config.batch_size == 17
