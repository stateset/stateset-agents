"""Eval, harvest and SFT encode prompts and generate through one path.

The harvest's sampler had drifted from SFT's eval: it encoded the rendered
chat template with the tokenizer's default special tokens (a second BOS on
Llama/Gemma-style tokenizers), omitted ``generation_compat_kwargs``, loaded
the tokenizer without the preset's ``trust_remote_code``, and never seeded
its sampling.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training import harvest, sft  # noqa: E402


class FakeTokenizer:
    """Adds a BOS id unless ``add_special_tokens=False`` — as Llama does."""

    eos_token_id = 0
    BOS = 1

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw):
        return "<s>user: " + messages[-1]["content"] + "\nassistant:"

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        self.calls.append({"add_special_tokens": add_special_tokens})
        ids = [ord(c) % 50 + 2 for c in text]
        if add_special_tokens:
            ids = [self.BOS, *ids]
        batch = {"input_ids": torch.tensor([ids])}
        return SimpleNamespace(to=lambda device: batch, __getitem__=batch.__getitem__)

    def decode(self, ids, skip_special_tokens=True):
        return "x" * len(ids)


class FakeModel:
    device = "cpu"

    def __init__(self) -> None:
        self.generate_kwargs: list[dict[str, Any]] = []

    def generate(self, input_ids, **kwargs):
        self.generate_kwargs.append(kwargs)
        n = kwargs.get("num_return_sequences", 1)
        return torch.cat(
            [input_ids.repeat(n, 1), torch.zeros(n, 3, dtype=torch.long)], 1
        )


@pytest.fixture
def compat(monkeypatch):
    monkeypatch.setattr(
        sft, "generation_compat_kwargs", lambda model: {"compat_marker": True}
    )


def test_eval_and_harvest_encode_without_extra_special_tokens(compat):
    tok, model = FakeTokenizer(), FakeModel()
    assert sft.generate_completions(model, tok, ["hi"], max_new_tokens=3) == ["xxx"]
    sampled = harvest.sample_completions(
        model, tok, "hi", best_of=2, temperature=0.9, top_p=0.95, max_new_tokens=3
    )
    assert sampled == ["xxx", "xxx"]
    assert [c["add_special_tokens"] for c in tok.calls] == [False, False]
    greedy, sampling = model.generate_kwargs
    assert greedy["do_sample"] is False and greedy["compat_marker"] is True
    assert sampling["do_sample"] is True and sampling["compat_marker"] is True
    assert (sampling["temperature"], sampling["top_p"]) == (0.9, 0.95)
    assert sampling["num_return_sequences"] == 2


def test_tokenizer_honours_the_presets_trust_remote_code(monkeypatch):
    import transformers

    from stateset_agents.core.model_presets import PRESETS

    trusted = next(p.model_id for p in PRESETS.values() if p.trust_remote_code)
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda name, **kwargs: seen.append({"name": name, **kwargs}),
    )
    sft.load_tokenizer_for(trusted)
    sft.load_tokenizer_for("someone/unknown-model")
    assert seen[0]["trust_remote_code"] is True
    assert seen[1]["trust_remote_code"] is False


def test_harvest_loads_the_shared_tokenizer_and_seeds_sampling(tmp_path, monkeypatch):
    loaded, seeds = [], []
    tok, model = FakeTokenizer(), FakeModel()
    monkeypatch.setattr(harvest, "gpu_available", lambda: True)
    monkeypatch.setattr(
        harvest, "load_tokenizer_for", lambda name, **kw: loaded.append(name) or tok
    )
    monkeypatch.setattr(harvest, "load_base_model_for_sft", lambda name: model)
    monkeypatch.setattr(sft, "is_sharded_across_devices", lambda model: True)
    monkeypatch.setattr(model, "eval", lambda: model, raising=False)
    monkeypatch.setattr(harvest, "_seed_sampling", seeds.append)
    summary = harvest.run_harvest_job(
        {
            "base_model": "org/model",
            "harvest_prompts": [{"prompt": "hi", "expect": ["x"]}],
            "output_dir": str(tmp_path),
            "best_of": 2,
            "temperature": 0.9,
            "top_p": 0.95,
            "max_new_tokens": 3,
            "seed": 7,
        }
    )
    assert loaded == ["org/model"] and seeds == [7]
    assert summary["seed"] == 7
    assert json.loads((tmp_path / "harvest_summary.json").read_text())["seed"] == 7
    assert harvest.harvest_seed({}) == harvest.DEFAULT_HARVEST_SEED
    with pytest.raises(ValueError, match="seed"):
        harvest.harvest_seed({"seed": -1})


def test_training_plan_prints_a_tool_call_row(capsys):
    row = {
        "messages": [
            {"role": "user", "content": "where is order 1?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"type": "function", "function": {"name": "lookup"}}],
            },
        ]
    }
    sft.print_training_plan(
        rows=[row],
        base_model="m",
        output_dir=tmp_path_placeholder(),
        num_epochs=1,
        lora_r=8,
        learning_rate=1e-4,
        max_length=64,
    )
    assert "<tool call: lookup>" in capsys.readouterr().out


def tmp_path_placeholder():
    from pathlib import Path

    return Path("out")
