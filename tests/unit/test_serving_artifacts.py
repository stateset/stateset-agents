import json
import sys
import types

import pytest

from stateset_agents.training import (
    build_serving_manifest,
    write_serving_manifest,
)
from stateset_agents.training.serving_artifacts import export_merged_model_for_serving


def test_build_serving_manifest_preserves_recommendations():
    manifest = build_serving_manifest(
        "./outputs/qwen3_5_27b",
        "Qwen/Qwen3.5-27B",
        use_lora=True,
        use_vllm=True,
        merged_model_dir="./outputs/qwen3_5_27b/merged",
        recommended={
            "tensor_parallel_size": 8,
            "reasoning_parser": "qwen3",
            "language_model_only": True,
        },
    )

    assert manifest["model"] == "Qwen/Qwen3.5-27B"
    assert manifest["merged_model_dir"] == "./outputs/qwen3_5_27b/merged"
    assert manifest["recommended"]["reasoning_parser"] == "qwen3"
    assert manifest["recommended"]["language_model_only"] is True


def test_build_serving_manifest_empty_recommendations_default_to_dict():
    manifest = build_serving_manifest(
        "./outputs/x",
        "Model",
        use_lora=False,
        use_vllm=False,
    )
    assert manifest["recommended"] == {}
    assert manifest["merged_model_dir"] is None


def test_write_serving_manifest_writes_json(tmp_path):
    path = write_serving_manifest(
        str(tmp_path),
        "Qwen/Qwen3.5-27B",
        use_lora=True,
        use_vllm=True,
        merged_model_dir=str(tmp_path / "merged"),
        recommended={"tensor_parallel_size": 8},
    )

    assert path.name == "serving_manifest.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["model"] == "Qwen/Qwen3.5-27B"
    assert loaded["recommended"]["tensor_parallel_size"] == 8


def test_write_serving_manifest_creates_parent_dirs(tmp_path):
    nested = tmp_path / "a" / "b" / "c"
    path = write_serving_manifest(
        str(nested),
        "Model",
        use_lora=False,
        use_vllm=True,
    )
    assert path.exists()
    assert path.parent == nested


def _install_fake_transformers(monkeypatch, *, architectures, has_image_text=True):
    """Install minimal transformers + peft stubs in sys.modules for merge tests."""
    saved_model = {}

    class _FakeConfig:
        def __init__(self, archs):
            self.architectures = archs

    class _FakeMergedModel:
        def save_pretrained(self, path):
            saved_model["merged_path"] = str(path)

    class _FakePeftWrapper:
        def merge_and_unload(self):
            return _FakeMergedModel()

    class _FakeCausalLM:
        @staticmethod
        def from_pretrained(name, **kwargs):
            saved_model["causal_called_with"] = (name, kwargs)
            return object()

    class _FakeImageTextToText:
        @staticmethod
        def from_pretrained(name, **kwargs):
            saved_model["image_text_called_with"] = (name, kwargs)
            return object()

    class _FakeAutoConfig:
        @staticmethod
        def from_pretrained(name, **kwargs):
            return _FakeConfig(architectures)

    class _FakeTokenizer:
        def save_pretrained(self, path):
            saved_model["tokenizer_saved_to"] = str(path)

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name, **kwargs):
            return _FakeTokenizer()

    class _FakeProcessor:
        def save_pretrained(self, path):
            saved_model["processor_saved_to"] = str(path)

    class _FakeAutoProcessor:
        @staticmethod
        def from_pretrained(name, **kwargs):
            return _FakeProcessor()

    class _FakePeftModel:
        @staticmethod
        def from_pretrained(model, adapter_dir):
            saved_model["adapter_dir"] = adapter_dir
            return _FakePeftWrapper()

    transformers_mod = types.ModuleType("transformers")
    transformers_mod.AutoConfig = _FakeAutoConfig
    transformers_mod.AutoModelForCausalLM = _FakeCausalLM
    transformers_mod.AutoTokenizer = _FakeAutoTokenizer
    transformers_mod.AutoProcessor = _FakeAutoProcessor
    if has_image_text:
        transformers_mod.AutoModelForImageTextToText = _FakeImageTextToText

    peft_mod = types.ModuleType("peft")
    peft_mod.PeftModel = _FakePeftModel

    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    monkeypatch.setitem(sys.modules, "peft", peft_mod)

    return saved_model


def test_export_merged_model_causal_lm(tmp_path, monkeypatch):
    saved = _install_fake_transformers(
        monkeypatch, architectures=["Qwen2ForCausalLM"]
    )

    output = export_merged_model_for_serving(
        base_model_name="Qwen/Qwen3.5-0.8B-Base",
        adapter_dir=str(tmp_path / "adapter"),
        output_dir=str(tmp_path / "merged"),
    )

    assert output == str(tmp_path / "merged")
    assert "causal_called_with" in saved
    assert "image_text_called_with" not in saved
    assert saved["merged_path"] == str(tmp_path / "merged")
    assert saved["adapter_dir"] == str(tmp_path / "adapter")
    assert saved["tokenizer_saved_to"] == str(tmp_path / "merged")


def test_export_merged_model_conditional_generation(tmp_path, monkeypatch):
    saved = _install_fake_transformers(
        monkeypatch,
        architectures=["Qwen3_5_OmniForConditionalGeneration"],
    )

    output = export_merged_model_for_serving(
        base_model_name="Qwen/Qwen3.5-27B",
        adapter_dir=str(tmp_path / "adapter"),
        output_dir=str(tmp_path / "merged"),
    )

    assert output == str(tmp_path / "merged")
    assert "image_text_called_with" in saved
    assert "causal_called_with" not in saved
    assert saved["processor_saved_to"] == str(tmp_path / "merged")


def test_export_merged_model_conditional_without_image_text_raises(
    tmp_path, monkeypatch
):
    _install_fake_transformers(
        monkeypatch,
        architectures=["Qwen3_5_OmniForConditionalGeneration"],
        has_image_text=False,
    )

    with pytest.raises(ImportError, match="AutoModelForImageTextToText"):
        export_merged_model_for_serving(
            base_model_name="Qwen/Qwen3.5-27B",
            adapter_dir=str(tmp_path / "adapter"),
            output_dir=str(tmp_path / "merged"),
        )


def test_export_merged_model_trust_remote_code_propagates(tmp_path, monkeypatch):
    saved = _install_fake_transformers(
        monkeypatch, architectures=["LlamaForCausalLM"]
    )

    export_merged_model_for_serving(
        base_model_name="meta-llama/Llama-3",
        adapter_dir=str(tmp_path / "adapter"),
        output_dir=str(tmp_path / "merged"),
        trust_remote_code=False,
    )
    _, kwargs = saved["causal_called_with"]
    assert kwargs["trust_remote_code"] is False


def test_export_merged_model_missing_dependencies_returns_empty(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "peft" or name.startswith("peft."):
            raise ImportError("peft not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    output = export_merged_model_for_serving(
        base_model_name="Model",
        adapter_dir=str(tmp_path / "adapter"),
        output_dir=str(tmp_path / "merged"),
    )
    assert output == ""
