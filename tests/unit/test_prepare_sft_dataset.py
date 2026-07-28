"""Unit tests for ``scripts/prepare_sft_dataset.py``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import prepare_sft_dataset as prep  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


@pytest.fixture
def curated_jsonl(tmp_path: Path) -> Path:
    p = tmp_path / "curated.jsonl"
    _write_jsonl(
        p,
        [
            {
                "prompt": "refund?",
                "response": "yes",
                "score": 0.85,
                "source": "s1.jsonl",
            },
            {
                "prompt": "crash",
                "response": "let me help",
                "score": 0.78,
                "source": "s2.jsonl",
            },
            {
                "prompt": "hours?",
                "response": "9-5",
                "score": 0.65,
                "source": "s1.jsonl",
            },
            {
                "prompt": "refund?",
                "response": "different answer",
                "score": 0.92,
                "source": "s3.jsonl",
            },
        ],
    )
    return p


class TestFormatters:
    def test_hf_trainer(self) -> None:
        out = prep.to_hf_trainer({"prompt": "Q", "response": "A"})
        assert out == {"text": "Q\n\nA"}

    def test_chat(self) -> None:
        out = prep.to_chat({"prompt": "Q", "response": "A"})
        assert out == {
            "messages": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ],
        }

    def test_axolotl(self) -> None:
        out = prep.to_axolotl({"prompt": "Q", "response": "A"})
        assert out == {"instruction": "Q", "input": "", "output": "A"}

    def test_all_three_in_registry(self) -> None:
        assert set(prep.FORMATTERS.keys()) == {"hf-trainer", "chat", "axolotl"}


class TestLoadCurated:
    def test_loads_valid_file(self, curated_jsonl: Path) -> None:
        entries = prep.load_curated(curated_jsonl)
        assert len(entries) == 4

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            prep.load_curated(tmp_path / "nope.jsonl")

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.jsonl"
        p.write_text(
            '{"prompt": "ok", "response": "ok"}\n'
            "not json\n"
            '{"prompt": "ok2", "response": "ok2"}\n'
        )
        entries = prep.load_curated(p)
        assert len(entries) == 2

    def test_skips_missing_required_fields(self, tmp_path: Path) -> None:
        p = tmp_path / "missing.jsonl"
        p.write_text(
            '{"prompt": "ok"}\n'  # no response
            '{"prompt": "ok2", "response": "ok2"}\n'
        )
        entries = prep.load_curated(p)
        assert len(entries) == 1


class TestFilter:
    def test_min_score(self, curated_jsonl: Path) -> None:
        entries = prep.load_curated(curated_jsonl)
        filtered = prep.filter_entries(entries, min_score=0.8)
        assert len(filtered) == 2
        assert all(e["score"] >= 0.8 for e in filtered)

    def test_source_filter(self, curated_jsonl: Path) -> None:
        entries = prep.load_curated(curated_jsonl)
        filtered = prep.filter_entries(entries, sources=["s1.jsonl"])
        assert len(filtered) == 2
        assert all(e["source"] == "s1.jsonl" for e in filtered)

    def test_dedup_by_prompt(self, curated_jsonl: Path) -> None:
        entries = prep.load_curated(curated_jsonl)
        filtered = prep.filter_entries(entries, dedup=True)
        # 4 entries, 3 distinct prompts (refund? appears twice)
        assert len(filtered) == 3
        prompts = [e["prompt"] for e in filtered]
        assert len(set(prompts)) == 3

    def test_combined_filters(self, curated_jsonl: Path) -> None:
        entries = prep.load_curated(curated_jsonl)
        filtered = prep.filter_entries(entries, min_score=0.7, dedup=True)
        # 0.85, 0.78, 0.92 pass min_score=0.7. After dedup (refund? duplicated), 2 entries.
        assert len(filtered) == 2


class TestEndToEndCLI:
    def test_chat_format_output(self, curated_jsonl: Path, tmp_path: Path) -> None:
        import subprocess

        out = tmp_path / "sft.jsonl"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "prepare_sft_dataset.py"),
                "--input",
                str(curated_jsonl),
                "--format",
                "chat",
                "--output",
                str(out),
                "--min-score",
                "0.7",
                "--dedup",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        assert result.returncode == 0
        assert out.exists()
        lines = [
            json.loads(line) for line in out.read_text().splitlines() if line.strip()
        ]
        assert len(lines) == 2
        for line in lines:
            assert "messages" in line
            roles = [m["role"] for m in line["messages"]]
            assert roles == ["user", "assistant"]
