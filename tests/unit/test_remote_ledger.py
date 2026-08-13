"""Tests for the remote-run cost ledger and budget ceiling."""

from __future__ import annotations

import json

import pytest

from stateset_agents.remote.ledger import (
    BudgetExceeded,
    CostEntry,
    check_budget,
    estimate_cost_usd,
    read_entries,
    record_entry,
    summarize,
)


class TestEstimateCost:
    def test_prices_a_run(self):
        # $2/hr for half an hour.
        assert estimate_cost_usd(2.0, 1800) == 1.0

    @pytest.mark.parametrize(
        ("rate", "duration"), [(None, 100.0), (2.0, None), (None, None)]
    )
    def test_unknown_inputs_give_unknown_cost_not_zero(self, rate, duration):
        """A missing price must never render as free — a zero would sail
        through a budget check that an unknown correctly refuses."""
        assert estimate_cost_usd(rate, duration) is None


class TestBudgetCeiling:
    def test_no_ceiling_allows_anything(self):
        assert check_budget(100.0, 3600, None) is None

    def test_allows_a_run_inside_the_ceiling(self):
        # $1/hr for at most 30 minutes = $0.50, under a $2 ceiling.
        assert check_budget(1.0, 1800, 2.0) == pytest.approx(0.5)

    def test_refuses_a_run_that_could_exceed_the_ceiling(self):
        with pytest.raises(BudgetExceeded, match=r"\$4\.00"):
            check_budget(4.0, 3600, 2.0)

    def test_worst_case_counts_every_gpu(self):
        """8 GPUs cost 8x — a per-GPU price under the ceiling can still
        blow it once the pod is multiplied out."""
        check_budget(1.0, 3600, 4.0, gpu_count=2)  # $2, fine
        with pytest.raises(BudgetExceeded):
            check_budget(1.0, 3600, 4.0, gpu_count=8)  # $8, not fine

    def test_unknown_price_with_a_ceiling_is_refused(self):
        """Renting hardware you cannot price, against a budget, is exactly
        the case the ceiling exists to prevent."""
        with pytest.raises(BudgetExceeded, match="did not report a price"):
            check_budget(None, 3600, 5.0)


class TestLedgerIO:
    def _entry(self, **kw):
        base = {
            "provider": "runpod",
            "job_id": "abc123",
            "base_model": "Qwen/Qwen3.5-0.8B",
            "gpu": "NVIDIA RTX A4000",
            "cost_usd": 0.25,
            "duration_s": 900.0,
            "status": "succeeded",
        }
        base.update(kw)
        return CostEntry(**base)

    def test_appends_and_reads_back(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        record_entry(self._entry(), path=path)
        record_entry(self._entry(job_id="def456", cost_usd=0.75), path=path)

        entries = read_entries(path)
        assert [e["job_id"] for e in entries] == ["abc123", "def456"]
        assert entries[0]["recorded_at"]

    def test_missing_ledger_reads_as_empty(self, tmp_path):
        assert read_entries(tmp_path / "nope.jsonl") == []

    def test_corrupt_lines_are_skipped_not_fatal(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        record_entry(self._entry(), path=path)
        with path.open("a", encoding="utf-8") as handle:
            handle.write("{not json\n")
        assert len(read_entries(path)) == 1

    def test_io_failure_never_raises(self, tmp_path):
        """Bookkeeping must not turn an already-paid-for successful run into
        a failure."""
        unwritable = tmp_path / "file.txt"
        unwritable.write_text("x", encoding="utf-8")
        record_entry(self._entry(), path=unwritable / "nested" / "ledger.jsonl")

    def test_summary_totals_and_breakdowns(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        record_entry(self._entry(cost_usd=1.0), path=path)
        record_entry(
            self._entry(job_id="b", cost_usd=2.0, base_model="other/model"), path=path
        )
        record_entry(self._entry(job_id="c", cost_usd=None), path=path)

        summary = summarize(read_entries(path))
        assert summary["runs"] == 3
        assert summary["runs_with_known_cost"] == 2
        assert summary["total_usd"] == 3.0
        assert summary["by_model"]["other/model"] == 2.0
        assert summary["by_gpu"]["NVIDIA RTX A4000"] == 3.0

    def test_entry_serializes_to_json(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        record_entry(self._entry(), path=path)
        row = json.loads(path.read_text(encoding="utf-8").strip())
        assert row["provider"] == "runpod"
        assert row["cost_usd"] == 0.25
