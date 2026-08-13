"""Tests for benchmarks/improvement_loop.py — the closed-loop benchmark.

Runs the benchmark in-process (module import, not subprocess) against small
seeded corpora and asserts that:

* the corpus generator is deterministic and honors the planted mix,
* the real ingest -> grade -> curate pipeline produces the metrics dict,
* the precision/recall floors gate the exit code correctly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks import improvement_loop as il  # noqa: E402


class TestGenerateCorpus:
    def test_deterministic_for_same_seed(self):
        a = il.generate_corpus(20, 0.6, seed=42)
        b = il.generate_corpus(20, 0.6, seed=42)
        assert [(c.label, c.flavor, c.messages) for c in a] == [
            (c.label, c.flavor, c.messages) for c in b
        ]

    def test_different_seed_changes_label_order(self):
        a = il.generate_corpus(20, 0.5, seed=1)
        b = il.generate_corpus(20, 0.5, seed=2)
        assert [c.label for c in a] != [c.label for c in b]

    def test_planted_mix_is_ground_truth(self):
        corpus = il.generate_corpus(30, 0.6, seed=42)
        assert sum(1 for c in corpus if c.label == "good") == 18
        assert sum(1 for c in corpus if c.label == "bad") == 12

    def test_bad_flavors_cycle(self):
        corpus = il.generate_corpus(30, 0.0, seed=42)
        flavors = {c.flavor for c in corpus}
        assert flavors == set(il.BAD_FLAVORS)

    def test_openai_jsonl_shape(self, tmp_path):
        corpus = il.generate_corpus(4, 0.5, seed=42)
        path = tmp_path / "logs.jsonl"
        il.write_corpus_jsonl(corpus, path)
        lines = path.read_text().splitlines()
        assert len(lines) == 4
        for line in lines:
            obj = json.loads(line)
            roles = [m["role"] for m in obj["messages"]]
            assert roles == ["user", "assistant"]

    def test_rejects_bad_fraction(self):
        with pytest.raises(ValueError):
            il.generate_corpus(10, 1.5, seed=42)


class TestComputeMetrics:
    def _corpus(self):
        return il.generate_corpus(10, 0.5, seed=42)

    def test_perfect_curation(self):
        corpus = self._corpus()
        good = [c.index for c in corpus if c.label == "good"]
        metrics = il.compute_metrics(corpus, good, {"mean_score": 0.9})
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["yield"] == 0.5

    def test_false_positives_hit_precision_only(self):
        corpus = self._corpus()
        kept = [c.index for c in corpus]  # keeps everything
        metrics = il.compute_metrics(corpus, kept, {})
        assert metrics["recall"] == 1.0
        assert metrics["precision"] == 0.5
        assert metrics["confusion"]["false_positive"] == 5

    def test_empty_curation(self):
        corpus = self._corpus()
        metrics = il.compute_metrics(corpus, [], {})
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["curated_count"] == 0


class TestRealPipeline:
    """End-to-end through the real run_improve (ingest -> grade -> curate)."""

    def test_benchmark_produces_metrics(self, tmp_path):
        metrics = il.run_benchmark(
            conversations=12,
            good_fraction=0.5,
            seed=42,
            reward="customer_support",
            threshold=0.7,
            workdir=tmp_path,
        )
        assert metrics["ground_truth"] == {
            "conversations": 12,
            "planted_good": 6,
            "planted_bad": 6,
        }
        # Every planted-good reply must be curated (recall 1.0); rude and
        # curt bad replies must be dropped, so precision beats the base rate.
        assert metrics["recall"] == 1.0
        assert metrics["precision"] > 0.5
        assert 0.0 < metrics["yield"] < 1.0
        assert metrics["grade_distribution"]["assistant_turns"] == 12
        assert set(metrics["false_positive_flavors"]) <= {"deflection"}
        # The pipeline actually wrote a training-ready curated set.
        curated = Path(metrics["config"]["curated_path"])
        assert curated.exists()
        first = json.loads(curated.read_text().splitlines()[0])
        assert {"prompt", "response", "score", "source"} <= set(first)

    def test_main_passes_with_default_floors(self, tmp_path, capsys):
        rc = il.main(
            [
                "--conversations",
                "12",
                "--workdir",
                str(tmp_path),
                "--output",
                str(tmp_path / "metrics.json"),
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "Improvement-loop benchmark" in out
        saved = json.loads((tmp_path / "metrics.json").read_text())
        assert saved["passed"] is True
        # Ratcheted floors: precision measured 1.0 since the resolution/
        # concreteness component closed the deflection gap (was 0.75/0.818).
        assert saved["floors"] == {"min_precision": 0.95, "min_recall": 0.95}

    def test_floors_gate_exit_code(self, tmp_path, capsys):
        # An all-bad corpus curates nothing: precision and recall both
        # collapse to 0.0, so the floors must trip.
        rc = il.main(
            [
                "--conversations",
                "9",
                "--good-fraction",
                "0.0",
                "--workdir",
                str(tmp_path),
            ]
        )
        assert rc == 1
        captured = capsys.readouterr()
        assert "FAIL" in captured.err

    def test_precision_floor_alone_gates(self, tmp_path, capsys):
        # This test used to plant the deflection blind spot (default corpus
        # measured precision ~0.82, so a 0.99 floor tripped). The resolution
        # component fixed that gap (precision now 1.0), so gate precision
        # alone with an all-bad corpus (nothing curated -> precision 0.0)
        # while disabling the recall floor.
        rc = il.main(
            [
                "--conversations",
                "9",
                "--good-fraction",
                "0.0",
                "--min-recall",
                "0.0",
                "--workdir",
                str(tmp_path),
            ]
        )
        assert rc == 1
        assert "FAIL" in capsys.readouterr().err
