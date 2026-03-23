"""
Experiment tracking and results persistence for the autonomous research loop.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TSV_HEADER = ["experiment_id", "objective", "training_time", "status", "description"]


@dataclass
class ExperimentRecord:
    """A single experiment result."""

    experiment_id: str
    params: dict[str, Any]
    metrics: dict[str, float]
    objective_value: float
    training_time: float
    status: str  # "keep", "discard", "crash"
    description: str = ""
    checkpoint_path: str | None = None
    provenance_path: str | None = None
    proof_path: str | None = None
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentTracker:
    """Tracks experiment results and maintains best-known state.

    Persists results to both a JSON log (full detail) and a TSV summary
    (human-readable, compatible with autoresearch results.tsv format).
    """

    def __init__(
        self,
        output_dir: Path,
        objective_metric: str = "eval_reward",
        direction: str = "maximize",
    ):
        self.output_dir = output_dir
        self.objective_metric = objective_metric
        self.direction = direction
        self.records: list[ExperimentRecord] = []
        self._best_value: float | None = None
        self._best_record: ExperimentRecord | None = None
        self._known_ids: set[str] = set()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._json_path = self.output_dir / "experiments.jsonl"
        self._tsv_path = self.output_dir / "results.tsv"

        self._ensure_tsv_header()

    # ------------------------------------------------------------------
    # Loading from disk
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        output_dir: str | Path,
        objective_metric: str = "eval_reward",
        direction: str = "maximize",
    ) -> ExperimentTracker:
        """Load a completed run's results from disk.

        Use this to analyze results after a run without re-running:

            tracker = ExperimentTracker.load("./auto_research_results")
            tracker.print_summary()
            analysis = tracker.get_analysis()

        Args:
            output_dir: Path to the auto-research output directory.
            objective_metric: Which metric was being optimized.
            direction: "maximize" or "minimize".

        Returns:
            An ExperimentTracker populated with all recorded experiments.
        """
        output_dir = Path(output_dir)
        jsonl_path = output_dir / "experiments.jsonl"

        # Create tracker without writing new files
        tracker = cls(output_dir, objective_metric, direction)

        if not jsonl_path.exists():
            return tracker

        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                record = ExperimentRecord(
                    experiment_id=data.get("experiment_id", "unknown"),
                    params=data.get("params", {}),
                    metrics=data.get("metrics", {}),
                    objective_value=data.get("objective_value", 0.0),
                    training_time=data.get("training_time", 0.0),
                    status=data.get("status", "unknown"),
                    description=data.get("description", ""),
                    checkpoint_path=data.get("checkpoint_path"),
                    provenance_path=data.get("provenance_path"),
                    proof_path=data.get("proof_path"),
                    timestamp=data.get("timestamp", 0.0),
                )
                tracker.load_record(record)

        return tracker

    @classmethod
    def from_legacy_tsv(
        cls,
        tsv_path: str | Path,
        output_dir: str | Path | None = None,
        objective_metric: str = "eval_reward",
        direction: str = "maximize",
    ) -> ExperimentTracker:
        """Import results from a legacy autoresearch results.tsv file.

        The autoresearch project uses a tab-separated format:
            commit  avg_reward  memory_gb  status  description

        This method converts that into an ExperimentTracker so you can
        use print_summary(), get_analysis(), and compare_runs().

        Args:
            tsv_path: Path to the legacy results.tsv file.
            output_dir: Where to store the converted data (defaults to
                a temporary directory next to the TSV file).
            objective_metric: Metric name to use.
            direction: "maximize" or "minimize".

        Returns:
            An ExperimentTracker populated from the TSV.
        """
        tsv_path = Path(tsv_path)
        if not tsv_path.exists():
            raise FileNotFoundError(f"TSV file not found: {tsv_path}")

        if output_dir is None:
            output_dir = tsv_path.parent / "auto_research_imported"

        tracker = cls(Path(output_dir), objective_metric, direction)

        with open(tsv_path, newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            if header is None:
                return tracker

            for row in reader:
                if len(row) < 5:
                    continue

                commit, avg_reward, memory_gb, status, description = (
                    row[0], row[1], row[2], row[3], row[4],
                )

                try:
                    objective = float(avg_reward)
                except (ValueError, TypeError):
                    objective = 0.0

                record = ExperimentRecord(
                    experiment_id=commit,
                    params={},  # Legacy TSV doesn't store params
                    metrics={
                        objective_metric: objective,
                        "memory_gb": float(memory_gb) if memory_gb else 0.0,
                    },
                    objective_value=objective,
                    training_time=0.0,
                    status=status if status in ("keep", "discard", "crash") else "unknown",
                    description=description,
                )
                tracker.load_record(record)

        return tracker

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def best_value(self) -> float | None:
        return self._best_value

    @property
    def best_record(self) -> ExperimentRecord | None:
        return self._best_record

    @property
    def num_experiments(self) -> int:
        return len(self.records)

    @property
    def num_kept(self) -> int:
        return sum(1 for r in self.records if r.status == "keep")

    @property
    def num_discarded(self) -> int:
        return sum(1 for r in self.records if r.status == "discard")

    @property
    def num_crashed(self) -> int:
        return sum(1 for r in self.records if r.status == "crash")

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def is_improvement(self, value: float) -> bool:
        """Check if a value is an improvement over the current best."""
        if self._best_value is None:
            return True
        if self.direction == "maximize":
            return value > self._best_value
        return value < self._best_value

    def record(self, record: ExperimentRecord) -> None:
        """Record an experiment result. Persists to JSONL + TSV.

        Raises ValueError if the experiment_id was already recorded
        (prevents double-persistence on resume).
        """
        if record.experiment_id in self._known_ids:
            raise ValueError(
                f"Experiment {record.experiment_id!r} already recorded. "
                "Did you accidentally re-record a resumed experiment?"
            )

        self.records.append(record)
        self._known_ids.add(record.experiment_id)

        if record.status == "keep" and self.is_improvement(record.objective_value):
            self._best_value = record.objective_value
            self._best_record = record

        # Persist — append to both files
        self._append_jsonl(record)
        self._append_tsv(record)

        logger.info(
            "Experiment %s: %s=%.6f status=%s — %s",
            record.experiment_id,
            self.objective_metric,
            record.objective_value,
            record.status,
            record.description,
        )

    def load_record(self, record: ExperimentRecord) -> None:
        """Load a record from a previous run (resume). Does NOT persist.

        Use this instead of `record()` when replaying history from disk.
        """
        self.records.append(record)
        self._known_ids.add(record.experiment_id)

        if record.status == "keep" and self.is_improvement(record.objective_value):
            self._best_value = record.objective_value
            self._best_record = record

    def get_history_for_proposer(self) -> list[dict[str, Any]]:
        """Return a simplified history suitable for proposer consumption."""
        return [
            {
                "params": r.params,
                "objective": r.objective_value,
                "status": r.status,
            }
            for r in self.records
        ]

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _ensure_tsv_header(self) -> None:
        """Ensure the TSV file exists and has a valid header."""
        if self._tsv_path.exists():
            # Check if it has a valid header
            try:
                with open(self._tsv_path) as f:
                    first_line = f.readline().strip()
                if first_line and "experiment_id" in first_line:
                    return  # Header exists
            except OSError:
                pass

        # Write header
        with open(self._tsv_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(_TSV_HEADER)

    def _append_jsonl(self, record: ExperimentRecord) -> None:
        with open(self._json_path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")
            f.flush()

    def _append_tsv(self, record: ExperimentRecord) -> None:
        with open(self._tsv_path, "a", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([
                record.experiment_id,
                f"{record.objective_value:.6f}",
                f"{record.training_time:.1f}",
                record.status,
                record.description,
            ])
            f.flush()

    # ------------------------------------------------------------------
    # Analysis and summary
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Print a comprehensive summary of the experiment run."""
        print("\n" + "=" * 60)
        print("AUTO-RESEARCH SUMMARY")
        print("=" * 60)
        print(f"Total experiments:  {self.num_experiments}")
        print(f"Kept:               {self.num_kept}")
        print(f"Discarded:          {self.num_discarded}")
        print(f"Crashed:            {self.num_crashed}")

        if not self.records:
            print("=" * 60 + "\n")
            return

        # Best result
        if self._best_record is not None:
            print(f"\nBest {self.objective_metric}: {self._best_value:.6f}")
            print(f"Best experiment:    {self._best_record.experiment_id}")
            print("Best params:")
            for k, v in sorted(self._best_record.params.items()):
                print(f"  {k}: {v}")

        # Convergence trace — accepted best value over time
        successful = [r for r in self.records if r.status != "crash"]
        if len(successful) >= 2:
            print("\nConvergence trace:")
            running_best = None
            for r in successful:
                marker = ""
                if running_best is None:
                    running_best = r.objective_value
                    if r.status == "keep":
                        marker = " *"
                elif r.status == "keep" and (
                    (self.direction == "maximize" and r.objective_value > running_best)
                    or (self.direction == "minimize" and r.objective_value < running_best)
                ):
                    running_best = r.objective_value
                    marker = " *"
                print(
                    f"  {r.experiment_id:>12s}  "
                    f"{r.objective_value:+.6f}  "
                    f"best={running_best:.6f}{marker}"
                )

        # Top 3 experiments
        if len(successful) >= 3:
            if self.direction == "maximize":
                top = sorted(successful, key=lambda r: r.objective_value, reverse=True)
            else:
                top = sorted(successful, key=lambda r: r.objective_value)

            print("\nTop 3 experiments:")
            for r in top[:3]:
                print(f"  {r.experiment_id}: {r.objective_value:.6f} — {r.description}")

            print("\nBottom 3 experiments:")
            for r in top[-3:]:
                print(f"  {r.experiment_id}: {r.objective_value:.6f} — {r.description}")

        # Timing stats
        times = [r.training_time for r in self.records if r.training_time > 0]
        if times:
            avg_time = sum(times) / len(times)
            total_time = sum(times)
            print("\nTiming:")
            print(f"  Avg training time:   {avg_time:.1f}s")
            print(f"  Total training time: {total_time:.1f}s ({total_time/60:.1f}m)")

        # Improvement rate
        if self.num_experiments > 1:
            rate = self.num_kept / self.num_experiments * 100
            print(f"\nImprovement rate: {rate:.0f}% ({self.num_kept}/{self.num_experiments})")

        print(f"\nResults saved to:   {self._tsv_path}")
        print("=" * 60 + "\n")

    def get_analysis(self) -> dict[str, Any]:
        """Return a structured analysis dict for dashboards and Jupyter.

        Includes everything needed for post-hoc analysis without
        re-parsing the JSONL/TSV files:
        - Summary statistics
        - Convergence curve (running best over time)
        - Parameter importance scores
        - Per-experiment records
        """
        successful = [r for r in self.records if r.status != "crash"]
        objectives = [r.objective_value for r in successful]

        analysis: dict[str, Any] = {
            "total_experiments": self.num_experiments,
            "kept": self.num_kept,
            "discarded": self.num_discarded,
            "crashed": self.num_crashed,
            "best_value": self._best_value,
            "best_experiment_id": (
                self._best_record.experiment_id if self._best_record else None
            ),
            "best_params": (
                dict(self._best_record.params) if self._best_record else {}
            ),
        }

        if objectives:
            analysis["mean_objective"] = sum(objectives) / len(objectives)
            analysis["min_objective"] = min(objectives)
            analysis["max_objective"] = max(objectives)

        if self.num_experiments > 1:
            analysis["improvement_rate"] = self.num_kept / self.num_experiments

        times = [r.training_time for r in self.records if r.training_time > 0]
        if times:
            analysis["avg_training_time"] = sum(times) / len(times)
            analysis["total_training_time"] = sum(times)

        # Convergence curve: [(experiment_index, running_best)]
        try:
            from .analysis import (
                compute_convergence_curve,
                compute_diminishing_returns,
                compute_parameter_importance,
            )

            analysis["convergence_curve"] = compute_convergence_curve(
                self.records, self.direction
            )
            analysis["parameter_importance"] = compute_parameter_importance(
                self.records
            )
            dr = compute_diminishing_returns(
                self.records, self.direction
            )
            if dr is not None:
                analysis["diminishing_returns_ratio"] = dr
        except Exception:
            pass

        # Per-experiment records (for Jupyter DataFrames)
        analysis["experiments"] = [
            {
                "id": r.experiment_id,
                "objective": r.objective_value,
                "status": r.status,
                "training_time": r.training_time,
                "description": r.description,
                "params": dict(r.params),
                "metrics": dict(r.metrics),
            }
            for r in self.records
        ]

        return analysis

    def to_dataframe(self) -> Any:
        """Convert experiment records to a pandas DataFrame.

        Each row is one experiment. Param values are flattened into
        individual columns (prefixed with ``param_``), and metric
        values are flattened similarly (prefixed with ``metric_``).

        Requires pandas (``pip install pandas``).

        Returns:
            A ``pandas.DataFrame`` with columns: id, objective, status,
            training_time, description, plus param_* and metric_* columns.

        Example:
            >>> tracker = ExperimentTracker.load("./results")
            >>> df = tracker.to_dataframe()
            >>> df[df.status == "keep"].sort_values("objective", ascending=False)
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "to_dataframe() requires pandas. Install with: pip install pandas"
            ) from exc

        rows: list[dict[str, Any]] = []
        for r in self.records:
            row: dict[str, Any] = {
                "id": r.experiment_id,
                "objective": r.objective_value,
                "status": r.status,
                "training_time": r.training_time,
                "description": r.description,
                "timestamp": r.timestamp,
            }
            # Flatten params into param_ columns
            for k, v in r.params.items():
                row[f"param_{k}"] = v
            # Flatten metrics into metric_ columns
            for k, v in r.metrics.items():
                row[f"metric_{k}"] = v
            rows.append(row)

        df = pd.DataFrame(rows)

        # Add running best column based on accepted improvements only.
        if len(df) > 0:
            running_best = None
            running_best_values: list[float] = []
            for row in rows:
                objective = row["objective"]
                status = row["status"]
                if running_best is None:
                    running_best = objective
                elif status == "keep":
                    if self.direction == "maximize" and objective > running_best:
                        running_best = objective
                    elif self.direction == "minimize" and objective < running_best:
                        running_best = objective
                running_best_values.append(running_best)
            df["running_best"] = running_best_values

        return df
