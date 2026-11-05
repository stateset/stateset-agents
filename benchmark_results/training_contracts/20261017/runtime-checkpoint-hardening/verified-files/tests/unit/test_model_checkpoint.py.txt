"""Real Hugging Face saves and local shard trust boundaries remain resumable."""

import copy
import json
import pickle
import weakref
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from stateset_agents.core.errors import ModelError
from stateset_agents.training import model_checkpoint as module
from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer
from stateset_agents.training.single_turn_trainer import SingleTurnGRPOTrainer


def tiny_model():
    return GPT2LMHeadModel(
        GPT2Config(
            n_layer=1,
            n_head=1,
            n_embd=8,
            vocab_size=16,
            n_positions=8,
            n_ctx=8,
            bos_token_id=0,
            eos_token_id=1,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
        )
    )


def make_trainer(model, path, trainer_type):
    trainer = trainer_type(
        SimpleNamespace(
            model=model, tokenizer=SimpleNamespace(save_pretrained=lambda _: None)
        ),
        object(),
        config=SimpleNamespace(
            output_dir=str(path), gradient_accumulation_steps=2, max_grad_norm=10
        ),
    )
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    return trainer


def train_batch(trainer, tokens):
    inputs = torch.tensor([tokens])
    loss = trainer.agent.model(input_ids=inputs, labels=inputs).loss / 2
    loss.backward()
    trainer._grad_accum_step += 1
    if trainer._grad_accum_step % 2 == 0:
        if isinstance(trainer, SingleTurnGRPOTrainer):
            trainer._apply_optimizer_step(torch, False, 10.0)
        else:
            trainer._apply_optimizer_step(torch)


@pytest.mark.asyncio
@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize(
    "trainer_type",
    [MultiTurnGRPOTrainer, SingleTurnGRPOTrainer],
    ids=["multi", "single"],
)
async def test_real_transformers_checkpoint_restores_ties_and_continued_updates(
    tmp_path, sharded, trainer_type
):
    model = tiny_model()
    model.save_pretrained = partial(
        model.save_pretrained, max_shard_size="1KB" if sharded else "5GB"
    )
    original = make_trainer(model, tmp_path, trainer_type)
    train_batch(original, [2, 3, 4])
    train_batch(original, [3, 4, 5])
    train_batch(original, [4, 5, 6])
    await original.save_checkpoint(checkpoint_name="saved")
    directory = tmp_path / "saved"
    assert (directory / "model.safetensors.index.json").exists() is sharded
    restored = make_trainer(tiny_model(), tmp_path / "restored", trainer_type)
    assert restored.load_checkpoint(directory)
    assert (
        restored.agent.model.lm_head.weight
        is restored.agent.model.transformer.wte.weight
    )
    for name, value in original.agent.model.state_dict().items():
        assert torch.equal(value, restored.agent.model.state_dict()[name]), name
    train_batch(original, [5, 6, 7])
    train_batch(restored, [5, 6, 7])
    assert original.global_step == restored.global_step == 2
    for name, value in original.agent.model.state_dict().items():
        assert torch.equal(value, restored.agent.model.state_dict()[name]), name


def write_shards(path):
    model = torch.nn.Linear(2, 2)
    weight_map = {}
    for index, (name, value) in enumerate(model.state_dict().items()):
        filename = f"shard-{index}.bin"
        torch.save({name: value}, path / filename)
        weight_map[name] = filename
    (path / "pytorch_model.bin.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    return model, weight_map


def load(model, path, **kwargs):
    return module.load_model_checkpoint(model, path, torch_module=torch, **kwargs)


def test_pickle_shards_load_one_at_a_time_through_trust_wrapper(tmp_path, monkeypatch):
    original, _ = write_shards(tmp_path)
    restored = torch.nn.Linear(2, 2)
    real_read = module._read_weights
    previous = []
    calls = []

    def read(path, torch_module, *, trusted):
        assert all(reference() is None for reference in previous)
        calls.append((path.name, trusted))
        piece = real_read(path, torch_module, trusted=trusted)
        previous[:] = [weakref.ref(value) for value in piece.values()]
        return piece

    monkeypatch.setattr(module, "_read_weights", read)
    assert load(restored, tmp_path)
    assert calls == [("shard-0.bin", False), ("shard-1.bin", False)]
    for name, value in original.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_file",
        "traversal",
        "symlink",
        "missing_weight",
        "unexpected_weight",
        "duplicate_json",
        "ambiguous",
    ],
)
def test_invalid_index_is_rejected_before_model_mutation(tmp_path, mutation):
    _, mapping = write_shards(tmp_path)
    index = tmp_path / "pytorch_model.bin.index.json"
    if mutation == "missing_file":
        (tmp_path / mapping["weight"]).unlink()
    elif mutation == "traversal":
        mapping["weight"] = "../outside.bin"
    elif mutation == "symlink":
        (tmp_path / "alias.bin").symlink_to(tmp_path / mapping["weight"])
        mapping["weight"] = "alias.bin"
    elif mutation == "missing_weight":
        del mapping["weight"]
    elif mutation == "unexpected_weight":
        mapping["foreign"] = mapping["weight"]
    elif mutation == "ambiguous":
        torch.save({}, tmp_path / "pytorch_model.bin")
    index.write_text(json.dumps({"weight_map": mapping}))
    if mutation == "duplicate_json":
        index.write_text(
            '{"weight_map":{"weight":"shard-0.bin","weight":"shard-0.bin","bias":"shard-1.bin"}}'
        )
    model = torch.nn.Linear(2, 2)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        load(model, tmp_path)
    for name, value in before.items():
        assert torch.equal(value, model.state_dict()[name])


@pytest.mark.parametrize("mutation", ["missing", "extra", "wrong_shape"])
def test_shard_contents_must_match_index_and_model(tmp_path, mutation):
    _, mapping = write_shards(tmp_path)
    piece = {"weight": torch.zeros(2, 2)}
    if mutation == "missing":
        piece = {}
    elif mutation == "extra":
        piece["extra"] = torch.zeros(1)
    else:
        piece["weight"] = torch.zeros(3)
    torch.save(piece, tmp_path / mapping["weight"])
    with pytest.raises((ValueError, RuntimeError)):
        load(torch.nn.Linear(2, 2), tmp_path)


def test_sharded_pickle_load_remains_untrusted_by_default(tmp_path):
    torch.save({"weight": pickle.loads}, tmp_path / "shard.bin")
    (tmp_path / "pytorch_model.bin.index.json").write_text(
        json.dumps({"weight_map": {"weight": "shard.bin"}})
    )
    with pytest.raises(ModelError):
        load(torch.nn.Linear(2, 2, bias=False), tmp_path)


@pytest.mark.parametrize("sharded", [False, True])
def test_conflicting_tied_weights_cannot_be_silently_overwritten(tmp_path, sharded):
    class Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.left = torch.nn.Parameter(torch.zeros(2))
            self.right = self.left

    values = {"left": torch.zeros(2), "right": torch.ones(2)}
    if sharded:
        for name, value in values.items():
            torch.save({name: value}, tmp_path / f"{name}.bin")
        (tmp_path / "pytorch_model.bin.index.json").write_text(
            json.dumps({"weight_map": {name: f"{name}.bin" for name in values}})
        )
    else:
        torch.save(values, tmp_path / "pytorch_model.bin")
    with pytest.raises(ValueError, match="Conflicting"):
        load(Tied(), tmp_path)
