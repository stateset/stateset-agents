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
            json.loads(line)
            for line in out.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(lines) == 2
        for line in lines:
            assert "messages" in line
            roles = [m["role"] for m in line["messages"]]
            assert roles == ["user", "assistant"]


MULTI_TURN = {
    "prompt": "Q2",
    "response": "A2",
    "messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
    ],
    "score": 0.9,
    "source": "s.jsonl",
}


class TestUnscoredEntries:
    """A missing score used to default to 1.0, so unscored rows passed any
    --min-score; stats defaulted it to 0.0 instead."""

    def test_unscored_entry_fails_min_score_and_is_counted(self, caplog) -> None:
        entries = [
            {"prompt": "scored", "response": "r", "score": 0.9},
            {"prompt": "unscored", "response": "r"},
            {"prompt": "junk", "response": "r", "score": "high"},
        ]
        with caplog.at_level("WARNING", logger="prepare_sft_dataset"):
            kept = prep.filter_entries(entries, min_score=0.5)
        assert [e["prompt"] for e in kept] == ["scored"]
        assert "dropped 2 entries with no numeric score" in caplog.text

    def test_unscored_entries_still_pass_without_min_score(self) -> None:
        assert len(prep.filter_entries([{"prompt": "p", "response": "r"}])) == 1

    def test_stats_report_unscored_entries_separately(self, tmp_path: Path) -> None:
        import subprocess

        src, out = tmp_path / "c.jsonl", tmp_path / "o.jsonl"
        _write_jsonl(
            src,
            [
                {"prompt": "a", "response": "r", "score": 0.8},
                {"prompt": "b", "response": "r"},
            ],
        )
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "prepare_sft_dataset.py"),
                *("--input", str(src), "--format", "chat", "--output", str(out)),
                "--stats",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "unscored:  1" in result.stdout
        assert "min score: 0.800" in result.stdout


class TestUtf8:
    def test_utf8_input_loads_under_the_c_locale(self, tmp_path: Path) -> None:
        import os
        import subprocess

        src, out = tmp_path / "c.jsonl", tmp_path / "o.jsonl"
        src.write_text(
            json.dumps(
                {"prompt": "Café order?", "response": "Refund de 12€", "score": 0.9},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        env = {**os.environ, "LC_ALL": "C", "LANG": "C", "PYTHONUTF8": "0"}
        env.pop("PYTHONIOENCODING", None)
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "prepare_sft_dataset.py"),
                *("--input", str(src), "--format", "chat", "--output", str(out)),
            ],
            capture_output=True,
            env=env,
            check=False,
        )
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        row = json.loads(out.read_text(encoding="utf-8"))
        assert row["messages"][0]["content"] == "Café order?"


class TestMultiTurnEntries:
    """Curated entries carry the graded multi-turn context; flattening it to
    (last user, response) trained on a different context than was graded."""

    def test_chat_keeps_the_full_graded_context(self) -> None:
        assert prep.to_chat(MULTI_TURN) == {"messages": MULTI_TURN["messages"]}

    def test_axolotl_puts_earlier_turns_in_input(self) -> None:
        assert prep.to_axolotl(MULTI_TURN) == {
            "instruction": "Q2",
            "input": "Be brief.\n\nQ1\n\nA1",
            "output": "A2",
        }

    def test_hf_trainer_text_has_every_turn(self) -> None:
        assert prep.to_hf_trainer(MULTI_TURN) == {
            "text": "Be brief.\n\nQ1\n\nA1\n\nQ2\n\nA2"
        }

    def test_flatten_restores_the_legacy_pair(self) -> None:
        assert prep.to_chat(prep.flatten(MULTI_TURN))["messages"] == [
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]

    def test_dedup_is_by_context_not_last_prompt(self) -> None:
        other_context = {
            **MULTI_TURN,
            "messages": [
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
            ],
        }
        same_context = {**MULTI_TURN, "response": "A2", "score": 0.5}
        kept = prep.filter_entries(
            [MULTI_TURN, other_context, same_context], dedup=True
        )
        assert kept == [MULTI_TURN, other_context]

    def test_inconsistent_messages_are_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "c.jsonl"
        bad = {**MULTI_TURN, "response": "not the last message"}
        _write_jsonl(p, [MULTI_TURN, bad])
        assert prep.load_curated(p) == [MULTI_TURN]
