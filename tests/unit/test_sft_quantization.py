"""QLoRA for SFT: config validation, BitsAndBytesConfig objects, dependency gates."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from stateset_agents.training import quantization as q
from stateset_agents.training import sft
from stateset_agents.training.quantization import (
    QuantizationConfig,
    build_bnb_config,
    prepare_quantized_model,
    quantized_load_kwargs,
    require_bitsandbytes,
)


class RecordingBnbConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


# --- configuration ------------------------------------------------------------


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"mode": "2bit"}, "quantization must be one of"),
        ({"mode": "4bit", "quant_type": "int4"}, "4-bit quant type"),
        ({"mode": "4bit", "compute_dtype": "float64"}, "compute dtype"),
        ({"mode": "4bit", "double_quant": "yes"}, "double_quant"),
    ],
)
def test_invalid_settings_are_rejected(settings, message):
    with pytest.raises(ValueError, match=message):
        QuantizationConfig(**settings)


def test_coerce_and_flags():
    assert QuantizationConfig.coerce(None) == QuantizationConfig()
    assert not QuantizationConfig.coerce(None).enabled
    assert QuantizationConfig.coerce("8bit").mode == "8bit"
    config = QuantizationConfig(mode="4bit")
    assert QuantizationConfig.coerce(config) is config
    assert (
        QuantizationConfig.coerce({"mode": "4bit", "quant_type": "fp4"}).quant_type
        == "fp4"
    )
    with pytest.raises(ValueError, match="Unknown quantization settings"):
        QuantizationConfig.coerce({"mode": "4bit", "bits": 4})
    with pytest.raises(ValueError, match="mode string"):
        QuantizationConfig.coerce(4)
    # GSPOTrainer's precedence: 8-bit wins when both flags are set.
    assert QuantizationConfig.from_flags(use_4bit=True, use_8bit=True).mode == "8bit"
    assert QuantizationConfig.from_flags(use_4bit=True).mode == "4bit"
    assert QuantizationConfig.from_flags().mode == "none"


def test_to_dict_records_only_the_settings_that_apply():
    assert QuantizationConfig().to_dict() == {"mode": "none"}
    assert QuantizationConfig(mode="8bit").to_dict() == {"mode": "8bit"}
    assert QuantizationConfig(mode="4bit").to_dict() == {
        "mode": "4bit",
        "quant_type": "nf4",
        "double_quant": True,
        "compute_dtype": "bfloat16",
    }


# --- BitsAndBytesConfig objects ------------------------------------------------


def test_four_bit_config_is_qlora_nf4_with_double_quant():
    built = build_bnb_config(
        QuantizationConfig(mode="4bit"),
        bnb_config_cls=RecordingBnbConfig,
        dtype_resolver=lambda name: f"dtype:{name}",
    )
    assert built.kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "dtype:bfloat16",
    }


def test_eight_bit_config_sets_only_load_in_8bit():
    built = build_bnb_config(
        QuantizationConfig(mode="8bit"), bnb_config_cls=RecordingBnbConfig
    )
    assert built.kwargs == {"load_in_8bit": True}
    with pytest.raises(ValueError, match="No quantization"):
        build_bnb_config(QuantizationConfig(), bnb_config_cls=RecordingBnbConfig)


def test_real_transformers_config_objects():
    pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    four = build_bnb_config(QuantizationConfig(mode="4bit", compute_dtype="float16"))
    assert four.load_in_4bit and not four.load_in_8bit
    assert four.bnb_4bit_quant_type == "nf4"
    assert four.bnb_4bit_use_double_quant is True
    assert four.bnb_4bit_compute_dtype == torch.float16
    eight = build_bnb_config(QuantizationConfig(mode="8bit"))
    assert eight.load_in_8bit and not eight.load_in_4bit


def test_load_kwargs_add_config_device_map_and_compute_dtype():
    base = {"torch_dtype": "bfloat16", "trust_remote_code": False}
    assert quantized_load_kwargs(QuantizationConfig(), base) == base
    kwargs = quantized_load_kwargs(
        QuantizationConfig(mode="4bit", compute_dtype="float16"),
        base,
        bnb_config_cls=RecordingBnbConfig,
        dtype_resolver=str,
    )
    assert kwargs["device_map"] == "auto" and kwargs["torch_dtype"] == "float16"
    assert kwargs["quantization_config"].kwargs["load_in_4bit"] is True
    assert base == {"torch_dtype": "bfloat16", "trust_remote_code": False}
    sharded = quantized_load_kwargs(
        QuantizationConfig(mode="8bit"),
        {"device_map": {"": 1}},
        bnb_config_cls=RecordingBnbConfig,
    )
    assert sharded["device_map"] == {"": 1}


def test_prepare_quantized_model_uses_peft_kbit_preparation():
    calls = []

    def fake_prepare(model, **kwargs):
        calls.append((model, kwargs))
        return "prepared"

    assert prepare_quantized_model("model", prepare=fake_prepare) == "prepared"
    assert calls == [("model", {"use_gradient_checkpointing": True})]


# --- missing dependencies -------------------------------------------------------


def test_cpu_only_hosts_get_an_actionable_error():
    with pytest.raises(RuntimeError, match="needs a CUDA GPU.*dry runs"):
        require_bitsandbytes(cuda_available=lambda: False)


def test_missing_bitsandbytes_names_the_extra(monkeypatch):
    real_find_spec = q.importlib.util.find_spec
    monkeypatch.setattr(
        q.importlib.util,
        "find_spec",
        lambda name, *a: None if name == "bitsandbytes" else real_find_spec(name, *a),
    )
    with pytest.raises(ImportError, match=r"stateset-agents\[qlora\]"):
        require_bitsandbytes(cuda_available=lambda: True)


def test_broken_bitsandbytes_install_is_reported(monkeypatch):
    monkeypatch.setattr(q.importlib.util, "find_spec", lambda name, *a: object())
    # A None entry makes ``import bitsandbytes`` raise, like a broken install.
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)
    with pytest.raises(ImportError, match="failed to import"):
        require_bitsandbytes(cuda_available=lambda: True)


# --- SFT wiring -----------------------------------------------------------------


def _fake_transformers(monkeypatch):
    calls = {}
    fake = types.ModuleType("transformers")

    class FakeCausal:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls["causal"] = (name, kwargs)
            return "quantized-model"

    fake.AutoModelForCausalLM = FakeCausal
    fake.BitsAndBytesConfig = RecordingBnbConfig
    monkeypatch.setitem(sys.modules, "transformers", fake)
    return calls


def test_sft_loader_passes_the_bnb_config_to_from_pretrained(monkeypatch):
    calls = _fake_transformers(monkeypatch)
    monkeypatch.setattr(sft, "require_bitsandbytes", lambda: None)
    monkeypatch.setattr(q, "_torch_dtype", lambda name: f"torch.{name}")
    assert (
        sft.load_base_model_for_sft("some/model", quantization="4bit")
        == "quantized-model"
    )
    kwargs = calls["causal"][1]
    assert kwargs["quantization_config"].kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "torch.bfloat16",
    }
    assert kwargs["device_map"] == "auto" and kwargs["torch_dtype"] == "bfloat16"


def test_sft_loader_fails_before_loading_when_cuda_is_missing(monkeypatch):
    calls = _fake_transformers(monkeypatch)
    monkeypatch.setattr(q, "_cuda_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        sft.load_base_model_for_sft("some/model", quantization="8bit")
    assert "causal" not in calls
    sft.load_base_model_for_sft("some/model")  # full precision still loads
    assert "quantization_config" not in calls["causal"][1]


def test_placement_and_preparation_helpers(monkeypatch):
    class Quantized:
        is_loaded_in_4bit = True

    class Plain:
        pass

    assert sft.is_placed_at_load(Quantized()) is True
    assert sft.is_placed_at_load(Plain()) is False
    monkeypatch.setattr(
        sft, "prepare_quantized_model", lambda model, **kw: ("prepared", model)
    )
    plain = Plain()
    assert sft.prepare_for_lora(plain, QuantizationConfig()) is plain
    assert sft.prepare_for_lora(plain, QuantizationConfig(mode="4bit")) == (
        "prepared",
        plain,
    )
    assert sft.quantization_hyperparameters(QuantizationConfig()) == {}
    assert sft.quantization_hyperparameters(QuantizationConfig(mode="8bit")) == {
        "quantization": {"mode": "8bit"}
    }


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Where is my order?"},
                    {"role": "assistant", "content": "Please share your order number."},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _job(dataset: Path, tmp_path: Path, **extra):
    return {
        "dataset": str(dataset),
        "base_model": "m",
        "output_dir": str(tmp_path / "out"),
        "num_epochs": 1,
        "lora_r": 8,
        "lora_alpha": 16,
        "learning_rate": 1e-5,
        "max_length": 64,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 1,
        **extra,
    }


def test_cli_flags_reach_the_job(dataset, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        sft,
        "run_sft_job",
        lambda job: captured.update(job)
        or {"returncode": 0, "logs": [], "output_dir": "o"},
    )
    argv = ["--dataset", str(dataset), "--base-model", "m", "--dry-run"]
    assert (
        sft.main(
            [
                *argv,
                "--quantization",
                "4bit",
                "--compute-dtype",
                "float16",
                "--no-double-quant",
            ]
        )
        == 0
    )
    assert captured["quantization"] == {
        "mode": "4bit",
        "quant_type": "nf4",
        "double_quant": False,
        "compute_dtype": "float16",
    }
    with pytest.raises(SystemExit) as excinfo:
        sft.build_parser().parse_args([*argv, "--quantization", "3bit"])
    assert excinfo.value.code == 2


def test_dry_run_plan_shows_qlora_and_bad_settings_fail_early(dataset, tmp_path):
    outcome = sft.run_sft_job(
        _job(dataset, tmp_path, dry_run=True, quantization="4bit")
    )
    assert outcome["returncode"] == 0
    assert any("QLoRA" in line and "mode=4bit" in line for line in outcome["logs"])
    bad = sft.run_sft_job(_job(dataset, tmp_path, dry_run=True, quantization="2bit"))
    assert bad["returncode"] == 1
    assert any("quantization must be one of" in line for line in bad["logs"])


def test_gpu_job_forwards_quantization_to_run_sft(dataset, tmp_path, monkeypatch):
    captured = {}

    def fake_run_sft(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(sft, "gpu_available", lambda: True)
    monkeypatch.setattr(sft, "run_sft", fake_run_sft)
    outcome = sft.run_sft_job(_job(dataset, tmp_path, quantization={"mode": "8bit"}))
    assert outcome["returncode"] == 1
    assert captured["quantization"] == QuantizationConfig(mode="8bit")
