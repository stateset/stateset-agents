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

from tests.unit._paths import rel_posix

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


def _torch_load_calls(source: str) -> list[ast.Call]:
    """Every call in ``source`` that resolves to ``torch.load``.

    Resolves the import graph rather than matching the literal name ``torch``,
    so aliased imports (``import torch as t``) and direct imports
    (``from torch import load as _l``) are caught too.
    """
    tree = ast.parse(source)

    module_aliases: set[str] = set()  # names bound to the torch module
    load_aliases: set[str] = set()  # names bound to torch.load itself
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch" or alias.name.startswith("torch."):
                    module_aliases.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module == "torch":
            for alias in node.names:
                if alias.name == "load":
                    load_aliases.add(alias.asname or alias.name)

    # Handles injected/parameter torch handles, which have no import statement.
    module_aliases.update({"torch", "_torch", "torch_module"})

    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "load"
            and isinstance(func.value, ast.Name)
            and func.value.id in module_aliases
        ):
            calls.append(node)
        elif isinstance(func, ast.Name) and func.id in load_aliases:
            calls.append(node)
    return calls


# The module that is allowed to call torch.load, plus the shim re-exporting it.
_CHECKPOINT_IO = Path(__file__).resolve().parents[2] / (
    "stateset_agents/core/checkpoint_io.py"
)
_CHECKPOINT_IO_SHIM = Path(__file__).resolve().parents[2] / (
    "stateset_agents/training/checkpoint_io.py"
)


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("import torch\ntorch.load('x')", id="plain"),
        pytest.param("import torch as t\nt.load('x')", id="aliased-module"),
        pytest.param("from torch import load\nload('x')", id="from-import"),
        pytest.param("from torch import load as _l\n_l('x')", id="aliased-load"),
        pytest.param("import torch.serialization\ntorch.load('x')", id="submodule"),
    ],
)
def test_collector_catches_every_import_form(source):
    """The guard must not be dodgeable by renaming the import."""
    assert len(_torch_load_calls(source)) == 1


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("import json\njson.load(fh)", id="json-load"),
        pytest.param(
            "from safetensors.torch import load_file\nload_file('x')", id="safetensors"
        ),
        pytest.param("# torch.load('x')\npass", id="comment"),
        pytest.param("s = \"torch.load('x')\"", id="string-literal"),
    ],
)
def test_collector_ignores_non_torch_loads(source):
    """...and must not produce false positives on look-alikes."""
    assert _torch_load_calls(source) == []


def test_only_checkpoint_io_calls_torch_load():
    """All checkpoint loading is funnelled through ``load_checkpoint_file``.

    Centralising it is what makes the ``weights_only`` default enforceable: a
    new ``torch.load`` anywhere else in the package fails this test.
    """
    root = Path(__file__).resolve().parents[2] / "stateset_agents"
    offenders = []
    for py in sorted(root.rglob("*.py")):
        if py == _CHECKPOINT_IO:
            continue
        for call in _torch_load_calls(py.read_text(encoding="utf-8")):
            offenders.append(f"{rel_posix(py, root)}:{call.lineno}")
    assert offenders == [], (
        "torch.load called outside core/checkpoint_io.py; route it through "
        "load_checkpoint_file(..., trusted=...) instead"
    )


def test_checkpoint_io_shim_only_re_exports():
    """The training shim must stay a re-export, not a second load site."""
    assert _torch_load_calls(_CHECKPOINT_IO_SHIM.read_text(encoding="utf-8")) == []


def test_checkpoint_io_pins_weights_only():
    """The single ``torch.load`` site passes a non-``False`` ``weights_only``."""
    calls = _torch_load_calls(_CHECKPOINT_IO.read_text(encoding="utf-8"))
    assert len(calls) == 1
    keywords = {kw.arg: kw.value for kw in calls[0].keywords}
    assert "weights_only" in keywords
    value = keywords["weights_only"]
    assert not (isinstance(value, ast.Constant) and value.value is False)


# ---------------------------------------------------------------------------
# Round-trip coverage for the trainers whose save() now writes a plain config
# ---------------------------------------------------------------------------


SAVED_BATCH_SIZE = 17
LOADER_BATCH_SIZE = 3  # deliberately different, so a no-op load() is visible


def _bcq(batch_size):
    from stateset_agents.training.offline_rl_bcq import (
        BatchConstrainedQLearning,
        BCQConfig,
    )

    return BatchConstrainedQLearning(
        state_dim=4, action_dim=2, config=BCQConfig(batch_size=batch_size), device="cpu"
    )


def _bear(batch_size):
    from stateset_agents.training.offline_rl_bear import BEARConfig, ConversationalBEAR

    return ConversationalBEAR(
        state_dim=4,
        action_dim=2,
        config=BEARConfig(batch_size=batch_size),
        device="cpu",
    )


def _decision_transformer(batch_size):
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
        batch_size=batch_size,
        use_conversation_embeddings=False,
    )
    return DecisionTransformerTrainer(config, device="cpu")


def _sim_to_real(batch_size):
    from stateset_agents.training.sim_to_real import SimToRealConfig, SimToRealTransfer

    return SimToRealTransfer(SimToRealConfig(batch_size=batch_size), device="cpu")


def _offline_grpo(batch_size):
    from stateset_agents.training.offline_grpo_trainer import (
        OfflineGRPOConfig,
        OfflineGRPOTrainer,
    )

    return OfflineGRPOTrainer(OfflineGRPOConfig(batch_size=batch_size), device="cpu")


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

    The saver and the loader are built with *different* ``batch_size`` values so
    the assertion cannot pass unless ``load()`` actually rebuilds the config
    from the checkpoint: deleting a trainer's rebuild block fails this test.
    The default ``trusted=False`` path must work end to end — that is the whole
    point of moving the config to plain data.
    """
    saver = factory(SAVED_BATCH_SIZE)
    saver.training_step = 11
    path = str(tmp_path / "ckpt.pt")
    saver.save(path)

    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(raw["config"], dict), "config must be persisted as plain data"
    assert raw["config"]["batch_size"] == SAVED_BATCH_SIZE

    loader = factory(LOADER_BATCH_SIZE)
    assert loader.config.batch_size == LOADER_BATCH_SIZE  # pre-condition
    loader.load(path)

    assert loader.training_step == 11
    assert type(loader.config) is type(saver.config)
    assert loader.config.batch_size == SAVED_BATCH_SIZE
