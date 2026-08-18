"""Tests for the on-pod merge module — the hybrid-serving fix."""

from __future__ import annotations

import subprocess
import sys


class TestDryRun:
    def test_prints_the_plan_and_exits_zero_without_a_gpu(self, tmp_path):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "stateset_agents.training.merge_adapter",
                "--base-model",
                "base/model",
                "--adapter",
                str(tmp_path / "adapter"),
                "--output-dir",
                str(tmp_path / "merged"),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert "Merge Plan" in result.stdout
        assert "base/model" in result.stdout


class TestMergeFunction:
    def test_merges_saves_and_returns_the_output_dir(self, tmp_path, monkeypatch):
        """Pin the exact peft/transformers call sequence with fakes: load
        base -> PeftModel.from_pretrained -> merge_and_unload -> save both
        model and tokenizer to the output dir."""
        import stateset_agents.training.merge_adapter as ma

        calls = []

        class FakeMerged:
            def save_pretrained(self, path, safe_serialization=True):
                calls.append(("save_model", path, safe_serialization))

        class FakePeftModel:
            def merge_and_unload(self):
                calls.append(("merge_and_unload",))
                return FakeMerged()

        class FakePeft:
            class PeftModel:
                @staticmethod
                def from_pretrained(model, adapter):
                    calls.append(("peft_load", adapter))
                    return FakePeftModel()

        class FakeTok:
            def save_pretrained(self, path):
                calls.append(("save_tokenizer", path))

        class FakeTransformers:
            class AutoTokenizer:
                @staticmethod
                def from_pretrained(name):
                    calls.append(("tokenizer", name))
                    return FakeTok()

        monkeypatch.setitem(sys.modules, "peft", FakePeft)
        monkeypatch.setitem(sys.modules, "transformers", FakeTransformers)
        monkeypatch.setattr(
            ma, "_load_full_checkpoint", lambda name: calls.append(("base", name))
        )
        monkeypatch.setattr(ma, "_adapter_for_model", lambda m, d: d)
        monkeypatch.setattr(ma, "gpu_available", lambda: False)
        probe = iter(["base completion", "merged completion"])
        monkeypatch.setattr(
            ma,
            "generate_completions",
            lambda m, t, p, max_new_tokens=48: [next(probe)],
        )

        out = ma.merge_adapter("base/m", tmp_path / "adapter", tmp_path / "merged")

        assert out == tmp_path / "merged"
        assert ("merge_and_unload",) in calls
        assert ("save_model", str(tmp_path / "merged"), True) in calls
        assert ("save_tokenizer", str(tmp_path / "merged")) in calls


class TestMergeProbe:
    def _run(self, tmp_path, monkeypatch, completions):
        """Wire fakes so generate_completions returns base then merged."""
        import sys

        import stateset_agents.training.merge_adapter as ma

        outputs = list(completions)

        class FakeMerged:
            def save_pretrained(self, path, safe_serialization=True):
                pass

        class FakePeftModel:
            def merge_and_unload(self):
                return FakeMerged()

        class FakePeft:
            class PeftModel:
                @staticmethod
                def from_pretrained(model, adapter):
                    return FakePeftModel()

        class FakeTok:
            def save_pretrained(self, path):
                pass

        class FakeTransformers:
            class AutoTokenizer:
                @staticmethod
                def from_pretrained(name):
                    return FakeTok()

        monkeypatch.setitem(sys.modules, "peft", FakePeft)
        monkeypatch.setitem(sys.modules, "transformers", FakeTransformers)
        monkeypatch.setattr(ma, "_load_full_checkpoint", lambda name: object())
        monkeypatch.setattr(ma, "_adapter_for_model", lambda m, d: d)
        monkeypatch.setattr(ma, "gpu_available", lambda: False)
        monkeypatch.setattr(
            ma,
            "generate_completions",
            lambda m, t, p, max_new_tokens=48: [outputs.pop(0)],
        )
        return ma.merge_adapter("base/m", tmp_path / "a", tmp_path / "merged")

    def test_differing_probe_passes_and_is_recorded(self, tmp_path, monkeypatch):
        import json

        out = self._run(tmp_path, monkeypatch, ["base says", "tuned says"])
        probe = json.loads((out / "merge_probe.json").read_text())
        assert probe["identical"] is False

    def test_identical_probe_refuses_to_serve_a_lie(self, tmp_path, monkeypatch):
        import json

        import pytest

        with pytest.raises(RuntimeError, match="no observable effect"):
            self._run(tmp_path, monkeypatch, ["same", "same"])
        # The evidence is still on disk for diagnosis.
        probe = json.loads((tmp_path / "merged" / "merge_probe.json").read_text())
        assert probe["identical"] is True


class TestRemapAdapterKeys:
    """Adapters trained through the text extraction silently no-op on the
    composite (measured: probe delta exactly 0.0 with peft's missing-key
    warning). The remap restores the match — measured 372/372 keys with
    real deltas on Qwen3.5-0.8B."""

    def test_remaps_when_the_composite_spelling_exists(self):
        from stateset_agents.training.merge_adapter import remap_adapter_keys

        weights = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": 1,
        }
        params = {"model.language_model.layers.0.self_attn.q_proj.weight"}

        remapped, changed = remap_adapter_keys(weights, params)

        assert changed == 1
        assert (
            "base_model.model.model.language_model.layers.0.self_attn."
            "q_proj.lora_A.weight" in remapped
        )

    def test_text_only_models_pass_through_untouched(self):
        from stateset_agents.training.merge_adapter import remap_adapter_keys

        weights = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": 1,
        }
        params = {"model.layers.0.self_attn.q_proj.weight"}  # no composite

        remapped, changed = remap_adapter_keys(weights, params)

        assert changed == 0
        assert remapped == weights
