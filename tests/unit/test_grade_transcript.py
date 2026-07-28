"""Unit tests for the transcript-grading script."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import grade_transcript as grader  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


class TestLoadTranscript:
    def test_loads_valid_jsonl(self, tmp_path: Path) -> None:
        p = tmp_path / "t.jsonl"
        _write_jsonl(
            p,
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        )
        turns = grader.load_transcript(p)
        assert len(turns) == 2
        assert turns[0]["role"] == "user"

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "t.jsonl"
        p.write_text(
            '{"role": "user", "content": "hi"}\n'
            "not json\n"
            '{"role": "assistant", "content": "ok"}\n'
        )
        turns = grader.load_transcript(p)
        assert len(turns) == 2

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            grader.load_transcript(tmp_path / "nope.jsonl")


class TestLoadContexts:
    def test_none_returns_empty(self) -> None:
        assert grader.load_contexts(None) == []

    def test_loads_jsonl(self, tmp_path: Path) -> None:
        p = tmp_path / "ctx.jsonl"
        _write_jsonl(
            p,
            [
                {"intent": "refund", "must_acknowledge": ["refund"]},
                {"intent": "billing", "must_acknowledge": ["bill"]},
            ],
        )
        ctxs = grader.load_contexts(p)
        assert len(ctxs) == 2
        assert ctxs[0]["intent"] == "refund"


class TestGetReward:
    def test_gsm8k(self) -> None:
        r = grader.get_reward("gsm8k")
        assert r.name == "gsm8k"

    def test_customer_support(self) -> None:
        r = grader.get_reward("customer_support")
        assert r.name == "support_composite"

    def test_tool_calling(self) -> None:
        r = grader.get_reward("tool_calling")
        assert r.name == "tool_call_composite"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown reward"):
            grader.get_reward("not-a-real-reward")


class TestGradeTranscript:
    @pytest.mark.asyncio
    async def test_grades_each_assistant_turn(self) -> None:
        from stateset_agents.data.customer_support_bench import SupportRewardComposite

        turns = [
            {"role": "user", "content": "I want a refund"},
            {
                "role": "assistant",
                "content": "I'd be happy to process your refund for that order.",
            },
            {"role": "user", "content": "It's order 1234"},
            {
                "role": "assistant",
                "content": "Thank you — I'm refunding order 1234 now.",
            },
        ]
        contexts = [
            {"intent": "refund", "must_acknowledge": ["refund", "order"]},
            {"intent": "refund", "must_acknowledge": ["refund", "order"]},
        ]
        rows = await grader.grade_transcript(turns, contexts, SupportRewardComposite())
        assert len(rows) == 2
        assert all(r["score"] > 0 for r in rows)

    @pytest.mark.asyncio
    async def test_renders_markdown_with_summary(self) -> None:
        from stateset_agents.data.customer_support_bench import SupportRewardComposite

        turns = [
            {"role": "user", "content": "refund"},
            {"role": "assistant", "content": "happy to refund your order"},
        ]
        rows = await grader.grade_transcript(
            turns, [{"must_acknowledge": ["refund", "order"]}], SupportRewardComposite()
        )
        md = grader.render_markdown(rows, "customer_support")
        assert "customer_support" in md
        assert "Total assistant turns" in md
        assert "Mean score" in md


class TestWriteCuratedExamples:
    @pytest.mark.asyncio
    async def test_only_high_scoring_turns_kept(self, tmp_path: Path) -> None:
        from stateset_agents.data.customer_support_bench import SupportRewardComposite

        turns = [
            {"role": "user", "content": "I need a refund for my order"},
            {
                "role": "assistant",
                "content": "I'd be happy to refund your order. Could you share the order number please?",
            },
            {"role": "user", "content": "That's terrible"},
            {"role": "assistant", "content": "impossible to help"},
        ]
        contexts = [
            {
                "intent": "refund",
                "must_acknowledge": ["refund", "order"],
                "must_avoid": ["impossible"],
            },
            {
                "intent": "refund",
                "must_acknowledge": ["refund", "order"],
                "must_avoid": ["impossible"],
            },
        ]
        rows = await grader.grade_transcript(turns, contexts, SupportRewardComposite())
        assert len(rows) == 2

        curated_path = tmp_path / "curated.jsonl"
        # The good turn scores ~0.83 (intent match + brand voice + safety pass);
        # the bad turn scores ~0.04 (avoided term penalty + missing acks).
        n = grader.write_curated_examples(
            Path("source.jsonl"),
            turns,
            rows,
            threshold=0.7,
            output_path=curated_path,
        )
        assert (
            n == 1
        ), f"Expected 1 example kept at threshold=0.7. Scores: {[r['score'] for r in rows]}"
        line = curated_path.read_text().strip()
        entry = json.loads(line)
        assert "prompt" in entry
        assert "response" in entry
        assert entry["score"] >= 0.7
        assert entry["source"] == "source.jsonl"

    @pytest.mark.asyncio
    async def test_append_across_calls(self, tmp_path: Path) -> None:
        """Multiple invocations should append, not clobber."""
        from stateset_agents.data.customer_support_bench import SupportRewardComposite

        curated = tmp_path / "curated.jsonl"
        for label in ("s1", "s2"):
            turns = [
                {
                    "role": "user",
                    "content": f"refund please {label}",
                },  # distinct prompts
                {
                    "role": "assistant",
                    "content": "I'd be happy to help with your refund for that order.",
                },
            ]
            contexts = [{"intent": "refund", "must_acknowledge": ["refund", "order"]}]
            rows = await grader.grade_transcript(
                turns, contexts, SupportRewardComposite()
            )
            grader.write_curated_examples(
                Path(f"{label}.jsonl"),
                turns,
                rows,
                threshold=0.5,
                output_path=curated,
            )
        lines = curated.read_text().splitlines()
        assert len(lines) == 2
        sources = {json.loads(line)["source"] for line in lines}
        assert sources == {"s1.jsonl", "s2.jsonl"}

    @pytest.mark.asyncio
    async def test_dedup_across_reruns(self, tmp_path: Path) -> None:
        """Re-grading the same transcript must not duplicate examples."""
        from stateset_agents.data.customer_support_bench import SupportRewardComposite

        curated = tmp_path / "curated.jsonl"
        turns = [
            {"role": "user", "content": "I need a refund"},
            {
                "role": "assistant",
                "content": "I'd be happy to help with your refund for that order.",
            },
        ]
        contexts = [{"intent": "refund", "must_acknowledge": ["refund", "order"]}]
        rows = await grader.grade_transcript(turns, contexts, SupportRewardComposite())

        n1 = grader.write_curated_examples(
            Path("session.jsonl"),
            turns,
            rows,
            threshold=0.5,
            output_path=curated,
        )
        assert n1 == 1

        # Run it again with the exact same data — should add zero new entries.
        n2 = grader.write_curated_examples(
            Path("session.jsonl"),
            turns,
            rows,
            threshold=0.5,
            output_path=curated,
        )
        assert n2 == 0, "Second run should not duplicate"
        assert len(curated.read_text().splitlines()) == 1


class TestSummarizeGradedBatch:
    """Cross-transcript aggregator (scripts/summarize_graded_batch.py)."""

    def test_renders_summary_table(self, tmp_path: Path) -> None:
        import importlib
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "summarize_graded_batch",
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "summarize_graded_batch.py",
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        graded = tmp_path / "graded"
        graded.mkdir()
        (graded / "s1.json").write_text(
            json.dumps(
                [
                    {
                        "assistant_turn_idx": 0,
                        "score": 0.42,
                        "response_preview": "...",
                        "breakdown": {},
                    },
                    {
                        "assistant_turn_idx": 1,
                        "score": 0.85,
                        "response_preview": "...",
                        "breakdown": {},
                    },
                ]
            )
        )
        (graded / "s2.json").write_text(
            json.dumps(
                [
                    {
                        "assistant_turn_idx": 0,
                        "score": 0.10,
                        "response_preview": "...",
                        "breakdown": {},
                    },
                ]
            )
        )
        transcripts = mod.load_graded_jsons(graded)
        assert len(transcripts) == 2
        md = mod.render_summary(transcripts)
        assert "Cross-Session Summary" in md
        assert "Transcripts:** 2" in md
        assert "Total assistant turns:** 3" in md
        # Mean of (0.42 + 0.85 + 0.10) / 3 ≈ 0.457
        assert "0.457" in md

    def test_skips_summary_json_in_dir(self, tmp_path: Path) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "summarize_graded_batch",
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "summarize_graded_batch.py",
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        graded = tmp_path / "graded"
        graded.mkdir()
        (graded / "summary.json").write_text("[]")  # should be skipped
        (graded / "real.json").write_text(json.dumps([{"score": 0.5}]))
        transcripts = mod.load_graded_jsons(graded)
        assert len(transcripts) == 1
        assert transcripts[0][0].name == "real.json"

    def test_empty_dir_renders_no_results(self, tmp_path: Path) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "summarize_graded_batch",
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "summarize_graded_batch.py",
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        graded = tmp_path / "graded"
        graded.mkdir()
        transcripts = mod.load_graded_jsons(graded)
        md = mod.render_summary(transcripts)
        assert "No graded transcripts found" in md


class TestEndToEndScript:
    def test_cli_invocation_produces_markdown(self, tmp_path: Path) -> None:
        import subprocess

        history = tmp_path / "h.jsonl"
        _write_jsonl(
            history,
            [
                {"role": "user", "content": "refund please"},
                {
                    "role": "assistant",
                    "content": "I'll process the refund for your order.",
                },
            ],
        )
        contexts = tmp_path / "c.jsonl"
        _write_jsonl(
            contexts,
            [
                {"intent": "refund", "must_acknowledge": ["refund", "order"]},
            ],
        )
        out = tmp_path / "graded.md"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "grade_transcript.py"),
                "--history",
                str(history),
                "--reward",
                "customer_support",
                "--context-file",
                str(contexts),
                "--output",
                str(out),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert out.exists()
        body = out.read_text()
        assert "Mean score" in body
        assert "Total assistant turns" in body
