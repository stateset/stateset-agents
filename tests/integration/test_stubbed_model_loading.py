from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training import base_trainer as bt


class DummyTokenizer:
    def __init__(self) -> None:
        self.pad_token = None
        self.eos_token = "</s>"

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


class DummyModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(use_cache=True)
        self.gradient_checkpointing_enabled = False

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing_enabled = True


class DummyAutoModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return DummyModel()


def test_stubbed_model_loading(monkeypatch) -> None:
    monkeypatch.setattr(bt, "_load_transformers", lambda: True)
    monkeypatch.setattr(bt, "_AutoTokenizer", DummyTokenizer)
    monkeypatch.setattr(bt, "_AutoModelForCausalLM", DummyAutoModel)
    monkeypatch.setattr(bt, "_transformers_loaded", True)

    config = bt.BaseTrainerConfig(
        model_name="stub-model",
        gradient_checkpointing=True,
        use_lora=False,
        use_8bit=False,
        use_4bit=False,
        beta=0.0,
        use_reference_model=False,
    )
    manager = bt.BaseModelManager(config)
    model, tokenizer = manager.load_model_and_tokenizer()

    assert isinstance(model, DummyModel)
    assert isinstance(tokenizer, DummyTokenizer)
    assert tokenizer.pad_token == tokenizer.eos_token
    assert model.gradient_checkpointing_enabled is True
    assert model.config.use_cache is False
