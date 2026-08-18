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
            ma, "load_base_model_for_sft", lambda name: calls.append(("base", name))
        )

        out = ma.merge_adapter("base/m", tmp_path / "adapter", tmp_path / "merged")

        assert out == tmp_path / "merged"
        assert ("merge_and_unload",) in calls
        assert ("save_model", str(tmp_path / "merged"), True) in calls
        assert ("save_tokenizer", str(tmp_path / "merged")) in calls
