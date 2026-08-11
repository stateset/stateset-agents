"""Tests for ``stateset_agents.training.sft`` — the packaged SFT job.

This logic lives in the installed package, not in ``scripts/``, because
``scripts*`` is excluded from the wheel: a remote worker that ``pip install``s
stateset-agents must still be able to run the job.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stateset_agents.training import sft


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "curated.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": f"q{i}"},
                        {"role": "assistant", "content": f"a{i}"},
                    ]
                }
            )
            for i in range(3)
        )
        + "\n"
    )
    return path


def payload(dataset: Path, **overrides):
    base = {
        "dataset": str(dataset),
        "base_model": "Qwen/Qwen3.5-0.8B",
        "output_dir": str(dataset.parent / "out"),
        "num_epochs": 1,
        "lora_r": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-5,
        "max_length": 1024,
        "per_device_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "dry_run": True,
    }
    base.update(overrides)
    return base


class TestPackagedApi:
    def test_exposes_the_job_functions(self):
        """The wheel must carry everything the remote worker needs."""
        for name in (
            "load_chat_dataset",
            "gpu_available",
            "print_training_plan",
            "run_sft",
            "run_sft_job",
        ):
            assert hasattr(sft, name), name

    def test_script_reexports_them_for_backwards_compatibility(self):
        import importlib.util

        script = Path(__file__).resolve().parents[2] / "scripts" / "sft_from_curated.py"
        spec = importlib.util.spec_from_file_location("_sft_script", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert module.load_chat_dataset is sft.load_chat_dataset
        assert module.run_sft is sft.run_sft


class TestRunSftJob:
    def test_dry_run_succeeds_and_reports_the_plan(self, dataset):
        outcome = sft.run_sft_job(payload(dataset))

        assert outcome["returncode"] == 0
        assert any("Qwen/Qwen3.5-0.8B" in line for line in outcome["logs"])

    def test_empty_dataset_fails(self, dataset, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")

        outcome = sft.run_sft_job(payload(empty))

        assert outcome["returncode"] != 0
        assert outcome["logs"]

    def test_missing_dataset_fails_without_raising(self, tmp_path):
        outcome = sft.run_sft_job(payload(tmp_path / "absent.jsonl"))

        assert outcome["returncode"] != 0

    def test_reports_the_output_directory(self, dataset, tmp_path):
        out = tmp_path / "adapter"

        outcome = sft.run_sft_job(payload(dataset, output_dir=str(out)))

        assert outcome["output_dir"] == str(out)

    def test_ignores_provider_only_fields(self, dataset):
        """A full RemoteJobSpec dict carries resource fields the job must not choke on."""
        outcome = sft.run_sft_job(
            payload(dataset, gpu="A100", timeout_s=60, package_version="0.19.0")
        )

        assert outcome["returncode"] == 0


class TestModuleEntrypoint:
    """The job must be runnable as `python -m stateset_agents.training.sft`.

    That is the only invocation available to a remote worker, which has the
    wheel but no checkout — so it is the one both executors use.
    """

    def test_module_is_runnable(self, dataset, tmp_path):
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "stateset_agents.training.sft",
                "--dataset",
                str(dataset),
                "--base-model",
                "Qwen/Qwen3.5-0.8B",
                "--output-dir",
                str(tmp_path / "out"),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert "Qwen/Qwen3.5-0.8B" in result.stdout

    def test_module_exits_nonzero_on_empty_dataset(self, tmp_path):
        import subprocess
        import sys

        empty = tmp_path / "empty.jsonl"
        empty.write_text("")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "stateset_agents.training.sft",
                "--dataset",
                str(empty),
                "--base-model",
                "Qwen/Qwen3.5-0.8B",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0


class TestLoraTargetModules:
    """peft only infers target_modules for architectures in its built-in map.

    For anything else (Qwen3.5, for one) it raises "Please specify
    `target_modules`". Found on real GPU hardware — the job downloads the
    model and dies at adapter construction, so CPU dry-runs never see it.
    """

    def _model(self, names):
        """A stand-in exposing just the named_modules surface we inspect."""

        class FakeLinear:
            pass

        class FakeModel:
            def named_modules(self):
                return [(n, FakeLinear()) for n in names]

        return FakeModel()

    def test_picks_standard_projection_modules_when_present(self):
        model = self._model(
            [
                "",
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.self_attn.k_proj",
                "model.layers.0.self_attn.v_proj",
                "model.layers.0.self_attn.o_proj",
                "model.layers.0.mlp.gate_proj",
                "model.layers.0.mlp.up_proj",
                "model.layers.0.mlp.down_proj",
                "lm_head",
            ]
        )

        targets = sft.infer_lora_target_modules(model)

        assert set(targets) == {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }

    def test_never_targets_the_output_head(self):
        """Adapting lm_head bloats the adapter and is not what we want."""
        model = self._model(["model.layers.0.self_attn.q_proj", "lm_head"])

        assert "lm_head" not in sft.infer_lora_target_modules(model)

    def test_handles_architectures_using_fused_qkv(self):
        model = self._model(
            [
                "transformer.h.0.attn.c_attn",
                "transformer.h.0.attn.c_proj",
                "transformer.h.0.mlp.c_fc",
            ]
        )

        targets = sft.infer_lora_target_modules(model)

        assert "c_attn" in targets

    def test_returns_empty_when_nothing_recognisable(self):
        """Empty means 'let peft try'; it must not invent a bogus name."""
        model = self._model(["weird.thing", "another.module"])

        assert sft.infer_lora_target_modules(model) == []


class TestLoadBaseModelForSft:
    """Multimodal fallback: AutoModelForCausalLM -> AutoModelForImageTextToText."""

    def _fake_transformers(self, monkeypatch, causal_raises, has_itt=True):
        import sys
        import types

        calls = {}
        fake = types.ModuleType("transformers")

        class FakeCausal:
            @staticmethod
            def from_pretrained(name, **kwargs):
                calls["causal"] = (name, kwargs)
                if causal_raises:
                    raise ValueError(
                        "Unrecognized configuration class for AutoModelForCausalLM"
                    )
                return "causal-model"

        fake.AutoModelForCausalLM = FakeCausal
        if has_itt:

            class FakeITT:
                @staticmethod
                def from_pretrained(name, **kwargs):
                    calls["itt"] = (name, kwargs)
                    return "itt-model"

            fake.AutoModelForImageTextToText = FakeITT
        monkeypatch.setitem(sys.modules, "transformers", fake)
        return calls

    def test_causal_path_used_when_supported(self, monkeypatch):
        calls = self._fake_transformers(monkeypatch, causal_raises=False)
        assert sft.load_base_model_for_sft("some/model") == "causal-model"
        assert "itt" not in calls

    def test_falls_back_to_image_text_to_text(self, monkeypatch):
        calls = self._fake_transformers(monkeypatch, causal_raises=True)
        assert (
            sft.load_base_model_for_sft("meta-models/Muse-Glimmer-30B") == "itt-model"
        )
        assert calls["itt"][0] == "meta-models/Muse-Glimmer-30B"
        assert calls["itt"][1]["trust_remote_code"] is True

    def test_reraises_original_error_without_itt_class(self, monkeypatch):
        self._fake_transformers(monkeypatch, causal_raises=True, has_itt=False)
        with pytest.raises(ValueError, match="Unrecognized configuration"):
            sft.load_base_model_for_sft("meta-models/Muse-Glimmer-30B")


class TestBuildTrainingArguments:
    """transformers-5.x kwarg removals must degrade gracefully, not crash."""

    class _StrictArgs:
        def __init__(self, output_dir, learning_rate=1e-4, bf16=False):
            self.output_dir = output_dir
            self.learning_rate = learning_rate
            self.bf16 = bf16

    def test_passes_supported_kwargs_through(self):
        args = sft.build_training_arguments(
            self._StrictArgs, output_dir="x", learning_rate=2e-5, bf16=True
        )
        assert args.learning_rate == 2e-5 and args.bf16 is True

    def test_drops_removed_kwargs_instead_of_crashing(self, caplog):
        with caplog.at_level("WARNING", logger="sft_from_curated"):
            args = sft.build_training_arguments(
                self._StrictArgs, output_dir="x", warmup_ratio=0.1
            )
        assert args.output_dir == "x"
        assert "warmup_ratio" in caplog.text

    def test_var_keyword_ctor_gets_everything(self):
        class Flexible:
            def __init__(self, **kw):
                self.kw = kw

        args = sft.build_training_arguments(Flexible, anything=1, at_all=2)
        assert args.kw == {"anything": 1, "at_all": 2}


class TestVisionTowerExclusion:
    """Text-only SFT must not adapt vision-tower projections (no gradient
    flows there), even when their leaf names match decoder-MLP candidates."""

    def _model(self, names):
        class FakeLinear:
            pass

        class FakeModel:
            def named_modules(self):
                return [(n, FakeLinear()) for n in names]

        return FakeModel()

    def test_vision_tower_fc_layers_are_skipped(self):
        model = self._model(
            [
                "language_model.layers.0.self_attn.q_proj",
                "language_model.layers.0.mlp.gate_proj",
                "vision_tower.blocks.0.mlp.fc1",
                "vision_tower.blocks.0.mlp.fc2",
                "multi_modal_projector.fc1",
            ]
        )
        targets = sft.infer_lora_target_modules(model)
        assert targets == ["gate_proj", "q_proj"]

    def test_text_stack_fc_layers_still_count(self):
        model = self._model(["model.decoder.layers.0.fc1"])
        assert sft.infer_lora_target_modules(model) == ["fc1"]
