"""Unit tests for ``scripts/sft_from_curated.py``.

Stub-backed throughout — we test the load/validate/plan paths but not the
real-training path (which requires GPU + transformers).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import sft_from_curated as sft  # noqa: E402


def _write_chat_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


@pytest.fixture
def chat_dataset(tmp_path: Path) -> Path:
    p = tmp_path / "sft_train.jsonl"
    _write_chat_jsonl(
        p,
        [
            {
                "messages": [
                    {"role": "user", "content": "What's a refund?"},
                    {"role": "assistant", "content": "A refund returns your money."},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "How do I cancel?"},
                    {"role": "assistant", "content": "Visit Settings → Subscription."},
                ]
            },
        ],
    )
    return p


class TestLoadChatDataset:
    def test_loads_valid_rows(self, chat_dataset: Path) -> None:
        rows = sft.load_chat_dataset(chat_dataset)
        assert len(rows) == 2
        for r in rows:
            assert "messages" in r
            roles = [m["role"] for m in r["messages"]]
            assert roles == ["user", "assistant"]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            sft.load_chat_dataset(tmp_path / "nope.jsonl")

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "mixed.jsonl"
        p.write_text(
            '{"messages": [{"role": "user", "content": "ok"}]}\n'
            "not json\n"
            '{"no_messages_key": true}\n'
            '{"messages": [{"role": "assistant", "content": "ok"}]}\n'
        )
        rows = sft.load_chat_dataset(p)
        assert len(rows) == 2


class TestGpuDetection:
    def test_returns_bool(self) -> None:
        result = sft.gpu_available()
        assert isinstance(result, bool)


class TestTrainingPlan:
    def test_prints_plan(
        self, chat_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        rows = sft.load_chat_dataset(chat_dataset)
        sft.print_training_plan(
            rows=rows,
            base_model="stub://test",
            output_dir=tmp_path / "out",
            num_epochs=3,
            lora_r=16,
            learning_rate=2e-5,
            max_length=1024,
        )
        captured = capsys.readouterr()
        assert "SFT Training Plan" in captured.out
        assert "Dataset size:     2" in captured.out
        assert "stub://test" in captured.out
        assert "LoRA r:           16" in captured.out


class TestEndToEndScript:
    def test_dry_run_succeeds(self, chat_dataset: Path, tmp_path: Path) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "sft_from_curated.py"),
                "--dataset",
                str(chat_dataset),
                "--base-model",
                "stub://test",
                "--output-dir",
                str(tmp_path / "out"),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "SFT Training Plan" in result.stdout

    def test_empty_dataset_fails(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "sft_from_curated.py"),
                "--dataset",
                str(empty),
                "--base-model",
                "stub://test",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 1
        assert "No usable rows" in result.stderr

    def test_missing_dataset_path(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "sft_from_curated.py"),
                "--dataset",
                "/tmp/does_not_exist.jsonl",
                "--base-model",
                "stub://test",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0
        assert "not found" in (result.stderr + result.stdout).lower()


class TestFullCurationLoop:
    """Verify the prepare → sft pipeline chains correctly via subprocess."""

    def test_prepare_then_sft_dry_run(self, tmp_path: Path) -> None:
        # 1. Synthetic curated.jsonl
        curated = tmp_path / "curated.jsonl"
        curated.write_text(
            '{"prompt": "Q1?", "response": "A1.", "score": 0.85, "source": "s1.jsonl"}\n'
            '{"prompt": "Q2?", "response": "A2.", "score": 0.92, "source": "s2.jsonl"}\n'
        )

        # 2. prepare-sft → chat format
        sft_jsonl = tmp_path / "sft.jsonl"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "prepare_sft_dataset.py"),
                "--input",
                str(curated),
                "--format",
                "chat",
                "--output",
                str(sft_jsonl),
                "--min-score",
                "0.7",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert sft_jsonl.exists()

        # 3. sft-from-curated with --dry-run
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "sft_from_curated.py"),
                "--dataset",
                str(sft_jsonl),
                "--base-model",
                "stub://test",
                "--output-dir",
                str(tmp_path / "out"),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "Dataset size:     2" in result.stdout
