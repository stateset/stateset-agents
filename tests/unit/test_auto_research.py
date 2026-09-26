"""Tests for the auto_research module."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stateset_agents.training.auto_research.checkpoint_manager import CheckpointManager
from stateset_agents.training.auto_research.config import AutoResearchConfig
from stateset_agents.training.auto_research.experiment_loop import AutoResearchLoop
from stateset_agents.training.auto_research.experiment_tracker import (
    ExperimentRecord,
    ExperimentTracker,
)
from stateset_agents.training.auto_research.proposer import (
    GridProposer,
    PerturbationProposer,
    RandomProposer,
    create_proposer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def search_space():
    from stateset_agents.training.hpo.base import (
        SearchDimension,
        SearchSpace,
        SearchSpaceType,
    )

    return SearchSpace(
        [
            SearchDimension("learning_rate", SearchSpaceType.LOGUNIFORM, 1e-6, 1e-3),
            SearchDimension("num_generations", SearchSpaceType.INT, 2, 16),
            SearchDimension("warmup_ratio", SearchSpaceType.FLOAT, 0.0, 0.5),
            SearchDimension(
                "baseline_type",
                SearchSpaceType.CATEGORICAL,
                choices=["group_mean", "group_median"],
            ),
        ]
    )


# ---------------------------------------------------------------------------
# AutoResearchConfig
# ---------------------------------------------------------------------------


class TestAutoResearchConfig:
    def test_defaults(self):
        config = AutoResearchConfig()
        assert config.time_budget == 300
        assert config.max_experiments == 0
        assert config.objective_metric == "eval_reward"
        assert config.direction == "maximize"
        assert config.proposer == "perturbation"

    def test_validate_ok(self):
        config = AutoResearchConfig()
        assert config.validate() == []

    def test_validate_bad_direction(self):
        config = AutoResearchConfig(direction="sideways")
        warnings = config.validate()
        assert any("direction" in w for w in warnings)

    def test_validate_bad_proposer(self):
        config = AutoResearchConfig(proposer="magic")
        warnings = config.validate()
        assert any("proposer" in w for w in warnings)

    def test_validate_bad_time_budget(self):
        config = AutoResearchConfig(time_budget=-1)
        warnings = config.validate()
        assert any("time_budget" in w for w in warnings)

    def test_output_path(self):
        config = AutoResearchConfig(output_dir="/tmp/test_ar")
        assert config.output_path == Path("/tmp/test_ar")


# ---------------------------------------------------------------------------
# ExperimentTracker
# ---------------------------------------------------------------------------


class TestExperimentTracker:
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_nonfinite_objective_is_rejected(self, value: float):
        with pytest.raises(ValueError, match="objective_value must be finite"):
            ExperimentRecord(
                experiment_id="invalid",
                params={},
                metrics={},
                objective_value=value,
                training_time=0.0,
                status="keep",
            )

    def test_legacy_import_skips_nonfinite_objective(self, tmp_dir: Path):
        tsv_path = tmp_dir / "legacy.tsv"
        tsv_path.write_text(
            "commit\tavg_reward\tmemory_gb\tstatus\tdescription\n"
            "invalid\tnan\t0\tkeep\tbad score\n"
            "valid\t0.4\t0\tkeep\tgood score\n",
            encoding="utf-8",
        )
        tracker = ExperimentTracker.from_legacy_tsv(tsv_path)
        assert tracker.num_experiments == 1
        assert tracker.best_record.experiment_id == "valid"

    def test_record_and_best(self, tmp_dir: Path):
        tracker = ExperimentTracker(tmp_dir, direction="maximize")

        r1 = ExperimentRecord(
            experiment_id="exp_1",
            params={"lr": 1e-5},
            metrics={"eval_reward": 0.5},
            objective_value=0.5,
            training_time=10.0,
            status="keep",
            description="first",
        )
        tracker.record(r1)

        assert tracker.best_value == 0.5
        assert tracker.num_experiments == 1
        assert tracker.num_kept == 1

    def test_improvement_tracking(self, tmp_dir: Path):
        tracker = ExperimentTracker(tmp_dir, direction="maximize")

        tracker.record(
            ExperimentRecord(
                experiment_id="baseline",
                params={},
                metrics={},
                objective_value=0.5,
                training_time=0,
                status="keep",
            )
        )
        assert tracker.is_improvement(0.6) is True
        assert tracker.is_improvement(0.4) is False
        assert tracker.is_improvement(0.5) is False

    def test_minimize_direction(self, tmp_dir: Path):
        tracker = ExperimentTracker(tmp_dir, direction="minimize")

        tracker.record(
            ExperimentRecord(
                experiment_id="baseline",
                params={},
                metrics={},
                objective_value=0.5,
                training_time=0,
                status="keep",
            )
        )
        assert tracker.is_improvement(0.4) is True
        assert tracker.is_improvement(0.6) is False

    def test_minimize_best_survives_discard_crash_and_replay(self, tmp_dir: Path):
        tracker = ExperimentTracker(tmp_dir, direction="minimize")
        for experiment_id, value, status in (
            ("first", 0.5, "keep"),
            ("worse", 0.8, "discard"),
            ("crashed", 0.0, "crash"),
            ("better", 0.2, "keep"),
        ):
            tracker.record(
                ExperimentRecord(
                    experiment_id=experiment_id,
                    params={},
                    metrics={},
                    objective_value=value,
                    training_time=0,
                    status=status,
                )
            )
        assert tracker.best_record.experiment_id == "better"
        assert tracker.best_value == 0.2

        replayed = ExperimentTracker.load(tmp_dir, direction="minimize")
        assert replayed.best_record.experiment_id == "better"
        assert replayed.best_value == 0.2
        assert replayed.num_crashed == 1

    def test_crash_counting(self, tmp_dir: Path):
        tracker = ExperimentTracker(tmp_dir)

        tracker.record(
            ExperimentRecord(
                experiment_id="crash_1",
                params={},
                metrics={},
                objective_value=0.0,
                training_time=0,
                status="crash",
            )
        )
        assert tracker.num_crashed == 1
        assert tracker.num_kept == 0

    def test_crashed_record_cannot_become_best_after_replay(self, tmp_dir: Path):
        tracker = ExperimentTracker(tmp_dir, direction="maximize")
        tracker.record(
            ExperimentRecord(
                experiment_id="baseline",
                params={},
                metrics={},
                objective_value=0.5,
                training_time=0,
                status="keep",
            )
        )
        tracker.record(
            ExperimentRecord(
                experiment_id="crashed",
                params={},
                metrics={},
                objective_value=1.0,
                training_time=0,
                status="crash",
            )
        )
        assert tracker.best_record.experiment_id == "baseline"

        replayed = ExperimentTracker.load(tmp_dir, direction="maximize")
        assert replayed.best_record.experiment_id == "baseline"
        assert replayed.num_crashed == 1

    def test_tsv_output(self, tmp_dir: Path):
        tracker = ExperimentTracker(tmp_dir)

        tracker.record(
            ExperimentRecord(
                experiment_id="exp_1",
                params={"lr": 0.001},
                metrics={},
                objective_value=0.5,
                training_time=10.0,
                status="keep",
                description="test run",
            )
        )

        tsv = (tmp_dir / "results.tsv").read_text(encoding="utf-8")
        lines = tsv.strip().split("\n")
        assert len(lines) == 2  # header + 1 record
        assert "exp_1" in lines[1]
        assert "0.500000" in lines[1]

    def test_jsonl_output(self, tmp_dir: Path):
        tracker = ExperimentTracker(tmp_dir)

        tracker.record(
            ExperimentRecord(
                experiment_id="exp_1",
                params={"lr": 0.001},
                metrics={},
                objective_value=0.5,
                training_time=10.0,
                status="keep",
            )
        )

        jsonl = (tmp_dir / "experiments.jsonl").read_text(encoding="utf-8").strip()
        data = json.loads(jsonl)
        assert data["experiment_id"] == "exp_1"
        assert data["objective_value"] == 0.5

    def test_jsonl_failure_does_not_advance_best(self, tmp_dir: Path):
        tracker = ExperimentTracker(tmp_dir)
        tracker._append_jsonl = MagicMock(side_effect=OSError("disk full"))
        record = ExperimentRecord(
            experiment_id="exp_1",
            params={},
            metrics={},
            objective_value=0.5,
            training_time=0.0,
            status="keep",
        )

        with pytest.raises(OSError, match="disk full"):
            tracker.record(record)
        assert tracker.best_record is None
        assert tracker.num_experiments == 0

    def test_tsv_failure_preserves_jsonl_commit(self, tmp_dir: Path):
        tracker = ExperimentTracker(tmp_dir)
        tracker._append_tsv = MagicMock(side_effect=OSError("TSV unavailable"))
        record = ExperimentRecord(
            experiment_id="exp_1",
            params={},
            metrics={},
            objective_value=0.5,
            training_time=0.0,
            status="keep",
        )

        tracker.record(record)
        assert tracker.best_record is record
        assert '"experiment_id": "exp_1"' in (tmp_dir / "experiments.jsonl").read_text(
            encoding="utf-8"
        )

    def test_jsonl_sync_failure_does_not_advance_best(self, tmp_dir: Path):
        tracker = ExperimentTracker(tmp_dir)
        record = ExperimentRecord(
            experiment_id="exp_1",
            params={},
            metrics={},
            objective_value=0.5,
            training_time=0.0,
            status="keep",
        )

        with patch(
            "stateset_agents.training.auto_research.experiment_tracker.os.fsync",
            side_effect=OSError("sync failed"),
        ):
            with pytest.raises(OSError, match="sync failed"):
                tracker.record(record)
        assert tracker.best_record is None
        assert tracker.num_experiments == 0

    def test_history_for_proposer(self, tmp_dir: Path):
        tracker = ExperimentTracker(tmp_dir)

        tracker.record(
            ExperimentRecord(
                experiment_id="exp_1",
                params={"lr": 0.001},
                metrics={},
                objective_value=0.5,
                training_time=10.0,
                status="keep",
            )
        )
        tracker.record(
            ExperimentRecord(
                experiment_id="exp_2",
                params={"lr": 0.01},
                metrics={},
                objective_value=0.3,
                training_time=10.0,
                status="discard",
            )
        )

        history = tracker.get_history_for_proposer()
        assert len(history) == 2
        assert history[0]["objective"] == 0.5
        assert history[1]["status"] == "discard"


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------


class TestCheckpointManager:
    def test_has_best_initially_false(self, tmp_dir: Path):
        mgr = CheckpointManager(tmp_dir)
        assert mgr.has_best() is False

    def test_save_and_load_metadata(self, tmp_dir: Path):
        mgr = CheckpointManager(tmp_dir)

        agent = MagicMock()
        agent.model = None  # No model to save

        mgr.save_best(agent, "exp_1", {"lr": 0.001})
        assert mgr.has_best() is True

        meta = mgr.load_best_metadata()
        assert meta is not None
        assert meta["experiment_id"] == "exp_1"
        assert meta["params"]["lr"] == 0.001

    def test_save_overwrites_previous_best(self, tmp_dir: Path):
        mgr = CheckpointManager(tmp_dir)

        agent = MagicMock()
        agent.model = None

        mgr.save_best(agent, "exp_1", {"lr": 0.001})
        mgr.save_best(agent, "exp_2", {"lr": 0.01})

        meta = mgr.load_best_metadata()
        assert meta["experiment_id"] == "exp_2"

    def test_save_experiment(self, tmp_dir: Path):
        mgr = CheckpointManager(tmp_dir)

        agent = MagicMock()
        agent.model = None

        path = mgr.save_experiment(agent, "exp_1", {"lr": 0.001})
        assert path.exists()
        assert (path / "metadata.json").exists()

    @pytest.mark.parametrize("experiment_id", ["../escape", "..", "/tmp/escape", "a/b"])
    def test_save_experiment_rejects_unsafe_path(
        self, tmp_dir: Path, experiment_id: str
    ):
        mgr = CheckpointManager(tmp_dir)
        with pytest.raises(ValueError, match="safe checkpoint name"):
            mgr.save_experiment(MagicMock(), experiment_id, {})

    def test_atomic_save_creates_completion_marker(self, tmp_dir: Path):
        mgr = CheckpointManager(tmp_dir)
        agent = MagicMock()
        agent.model = None

        mgr.save_best(agent, "exp_1", {"lr": 0.001})

        # Completion marker must exist for has_best() to return True
        assert (mgr.best_dir / ".complete").exists()
        assert mgr.has_best() is True

    def test_has_best_requires_completion_marker(self, tmp_dir: Path):
        mgr = CheckpointManager(tmp_dir)
        # Create metadata but no completion marker (simulates crash mid-save)
        mgr.best_dir.mkdir(parents=True, exist_ok=True)
        (mgr.best_dir / "metadata.json").write_text('{"experiment_id": "exp_1"}')
        # Should NOT be considered valid
        assert mgr.has_best() is False

    def test_restart_recovers_best_moved_before_install(self, tmp_dir: Path):
        mgr = CheckpointManager(tmp_dir)
        agent = MagicMock()
        agent.model = None
        mgr.save_best(agent, "exp_1", {"lr": 0.001})
        mgr.best_dir.rename(mgr.checkpoints_dir / ".best_old")

        restarted = CheckpointManager(tmp_dir)
        assert restarted.has_best()
        assert restarted.load_best_metadata()["experiment_id"] == "exp_1"

    def test_failed_install_restores_previous_best(
        self, tmp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        mgr = CheckpointManager(tmp_dir)
        agent = MagicMock()
        agent.model = None
        mgr.save_best(agent, "exp_1", {})
        rename = Path.rename

        def fail_install(path: Path, target: Path) -> Path:
            if path.name.startswith(".best_tmp_") and target == mgr.best_dir:
                raise OSError("install failed")
            return rename(path, target)

        monkeypatch.setattr(Path, "rename", fail_install)
        with pytest.raises(OSError, match="install failed"):
            mgr.save_best(agent, "exp_2", {})
        assert mgr.has_best()
        assert mgr.load_best_metadata()["experiment_id"] == "exp_1"

    def test_failed_model_save_preserves_previous_best(self, tmp_dir: Path):
        mgr = CheckpointManager(tmp_dir)
        agent = MagicMock()
        agent.model = None
        mgr.save_best(agent, "baseline", {})

        class UnsavableModel:
            def state_dict(self):
                raise OSError("weights unavailable")

        agent.model = UnsavableModel()
        with pytest.raises(RuntimeError, match="could not save model weights"):
            mgr.save_best(agent, "failed", {})

        assert mgr.has_best()
        assert mgr.load_best_metadata()["experiment_id"] == "baseline"
        assert not list(mgr.checkpoints_dir.glob(".best_tmp_*"))

    def test_unrecorded_pending_best_rolls_back_on_restart(self, tmp_dir: Path):
        mgr = CheckpointManager(tmp_dir)
        agent = MagicMock()
        agent.model = None
        mgr.save_best(agent, "baseline", {})
        mgr.save_best(agent, "exp_1", {}, defer_commit=True)

        restarted = CheckpointManager(tmp_dir)
        assert restarted.reconcile_best("baseline")
        assert restarted.load_best_metadata()["experiment_id"] == "baseline"
        assert len(list(restarted.checkpoints_dir.glob(".uncommitted_*"))) == 1

    def test_recorded_pending_best_commits_on_restart(self, tmp_dir: Path):
        mgr = CheckpointManager(tmp_dir)
        agent = MagicMock()
        agent.model = None
        mgr.save_best(agent, "baseline", {})
        mgr.save_best(agent, "exp_1", {}, defer_commit=True)

        restarted = CheckpointManager(tmp_dir)
        assert restarted.reconcile_best("exp_1")
        assert restarted.load_best_metadata()["experiment_id"] == "exp_1"
        assert not (restarted.best_dir / ".pending").exists()
        assert not (restarted.checkpoints_dir / ".best_old").exists()

    def test_unrecorded_first_checkpoint_is_set_aside(self, tmp_dir: Path):
        mgr = CheckpointManager(tmp_dir)
        agent = MagicMock()
        agent.model = None
        mgr.save_best(agent, "baseline", {}, defer_commit=True)

        restarted = CheckpointManager(tmp_dir)
        assert not restarted.reconcile_best(None)
        assert not restarted.has_best()
        assert len(list(restarted.checkpoints_dir.glob(".uncommitted_*"))) == 1

    def test_restore_returns_false_when_no_checkpoint(self, tmp_dir: Path):
        mgr = CheckpointManager(tmp_dir)
        agent = MagicMock()
        agent.model = None
        assert mgr.restore_best(agent) is False

    def test_restore_rejects_unsafe_local_shard_index(self, tmp_dir: Path):
        mgr = CheckpointManager(tmp_dir)
        agent = MagicMock()
        agent.model = None
        mgr.save_best(agent, "baseline", {})
        model_dir = mgr.best_dir / "model"
        (model_dir / "model.safetensors.index.json").write_text(
            '{"weight_map": {"layer": "../outside"}}', encoding="utf-8"
        )
        agent.model = MagicMock()

        with pytest.raises(ValueError, match="unsafe checkpoint shard path"):
            mgr.restore_best(agent)

    def test_lora_adapter_save_records_base_model_name(self, tmp_dir: Path):
        """When saving a PEFT model, the base model name is recorded."""
        mgr = CheckpointManager(tmp_dir)
        agent = MagicMock()

        # Mock a PeftModel
        mock_model = MagicMock()
        mock_model.save_pretrained = MagicMock()

        # Make isinstance(model, PeftModel) work via class name
        mock_model.__class__.__name__ = "PeftModel"

        # Set up base model name path
        mock_base = MagicMock()
        mock_base.model.name_or_path = "gpt2"
        mock_model.base_model = mock_base

        agent.model = mock_model

        # Need to mock peft import for isinstance check
        with patch(
            "stateset_agents.training.auto_research.checkpoint_manager.Path.write_text"
        ):
            mgr.save_best(agent, "exp_1", {"lr": 0.001})

        # save_pretrained should have been called
        mock_model.save_pretrained.assert_called_once()


def test_resume_reconciles_checkpoint_against_valid_log(tmp_dir: Path) -> None:
    previous = ExperimentTracker(tmp_dir)
    previous.record(
        ExperimentRecord(
            experiment_id="baseline",
            params={},
            metrics={"eval_reward": 0.4},
            objective_value=0.4,
            training_time=0.0,
            status="keep",
        )
    )
    agent = MagicMock()
    agent.model = None
    checkpoints = CheckpointManager(tmp_dir)
    checkpoints.save_best(agent, "baseline", {})
    checkpoints.save_best(agent, "exp_0001", {}, defer_commit=True)
    with (tmp_dir / "experiments.jsonl").open("a", encoding="utf-8") as log:
        log.write(
            json.dumps(
                {
                    "experiment_id": "exp_0001",
                    "objective_value": float("nan"),
                    "status": "keep",
                }
            )
            + "\n"
        )

    loop = AutoResearchLoop.__new__(AutoResearchLoop)
    loop.config = SimpleNamespace(output_path=tmp_dir, objective_metric="eval_reward")
    loop.tracker = ExperimentTracker(tmp_dir)
    loop.checkpoint_mgr = CheckpointManager(tmp_dir)
    loop.agent = agent
    loop._resumed_existing_records = False

    assert loop._try_resume() == 1
    assert loop._resumed_existing_records
    assert loop.tracker.best_record.experiment_id == "baseline"
    assert loop.checkpoint_mgr.load_best_metadata()["experiment_id"] == "baseline"
    assert ExperimentTracker.load(tmp_dir).best_record.experiment_id == "baseline"


@pytest.mark.asyncio
async def test_nonfinite_baseline_is_not_promoted(tmp_dir: Path) -> None:
    loop = AutoResearchLoop.__new__(AutoResearchLoop)
    loop.config = SimpleNamespace(objective_metric="eval_reward")
    loop.baseline_params = {}
    loop._current_experiment_timeout = lambda: 1.0
    loop._evaluate = AsyncMock(return_value={"eval_reward": float("nan")})
    loop._cleanup_gpu = MagicMock()
    loop._record_crash = MagicMock()
    loop.checkpoint_mgr = MagicMock()

    await loop._run_baseline()

    loop._record_crash.assert_called_once_with(
        "baseline", {}, "non-finite baseline objective"
    )
    loop._cleanup_gpu.assert_called_once()
    loop.checkpoint_mgr.save_best.assert_not_called()


@pytest.mark.asyncio
async def test_nonfinite_experiment_is_not_promoted(tmp_dir: Path) -> None:
    loop = AutoResearchLoop.__new__(AutoResearchLoop)
    loop.config = SimpleNamespace(objective_metric="eval_reward")
    loop.baseline_params = {}
    loop.tracker = MagicMock(best_record=None)
    loop.proposer = MagicMock()
    loop.proposer.propose.return_value = ({"lr": 0.01}, "candidate")
    loop._current_experiment_timeout = lambda: 1.0
    loop._execute_experiment = AsyncMock(return_value={"eval_reward": float("inf")})
    loop._cleanup_gpu = MagicMock()
    loop._record_crash = MagicMock()
    loop._report_to_proposer = MagicMock()
    loop.checkpoint_mgr = MagicMock()
    loop.agent = MagicMock()

    await loop._run_experiment("exp_0001")

    loop._record_crash.assert_called_once_with(
        "exp_0001", {"lr": 0.01}, "non-finite experiment objective"
    )
    loop.checkpoint_mgr.save_best.assert_not_called()
    loop.checkpoint_mgr.restore_best.assert_called_once_with(loop.agent)


# ---------------------------------------------------------------------------
# Proposers
# ---------------------------------------------------------------------------


class TestRandomProposer:
    def test_propose_returns_params_and_description(self, search_space):
        proposer = RandomProposer(search_space)
        params, desc = proposer.propose(
            current_best={"learning_rate": 1e-5, "num_generations": 4},
            history=[],
        )
        assert "learning_rate" in params
        assert "num_generations" in params
        assert isinstance(desc, str)
        assert desc.startswith("random:")


class TestPerturbationProposer:
    def test_propose_returns_params_and_description(self, search_space):
        proposer = PerturbationProposer(search_space)
        params, desc = proposer.propose(
            current_best={
                "learning_rate": 1e-5,
                "num_generations": 4,
                "warmup_ratio": 0.1,
                "baseline_type": "group_mean",
            },
            history=[],
        )
        assert isinstance(params, dict)
        assert desc.startswith("perturb:")

    def test_always_makes_at_least_one_change(self, search_space):
        proposer = PerturbationProposer(search_space, num_params_to_change=1)
        current = {
            "learning_rate": 1e-5,
            "num_generations": 4,
            "warmup_ratio": 0.1,
            "baseline_type": "group_mean",
        }
        # Run multiple times — at least one change should happen
        for _ in range(10):
            params, desc = proposer.propose(current, [])
            if params != current:
                return
        # If we get here, something is wrong
        pytest.fail("PerturbationProposer never changed anything in 10 tries")


class TestGridProposer:
    def test_propose_iterates_through_grid(self, search_space):
        proposer = GridProposer(search_space, points_per_dim=2)
        seen: set[str] = set()
        for _ in range(5):
            params, desc = proposer.propose(
                current_best={"learning_rate": 1e-5, "num_generations": 4},
                history=[],
            )
            assert desc.startswith("grid[")
            seen.add(str(params))
        # Should have seen multiple different configs
        assert len(seen) >= 2


class TestPerturbationProposerEdgeCases:
    def test_extra_keys_filtered_out(self, search_space):
        """Keys not in search space should be dropped from proposals."""
        proposer = PerturbationProposer(search_space)
        params, _ = proposer.propose(
            current_best={
                "learning_rate": 1e-5,
                "num_generations": 4,
                "warmup_ratio": 0.1,
                "baseline_type": "group_mean",
                "unknown_param": "should_be_dropped",
            },
            history=[],
        )
        assert "unknown_param" not in params

    def test_non_numeric_value_handled_safely(self, search_space):
        """If a numeric dim has a non-numeric value, it should be re-sampled."""
        proposer = PerturbationProposer(search_space, num_params_to_change=1)
        # Force learning_rate to a string (corrupted state)
        params, _ = proposer.propose(
            current_best={
                "learning_rate": "not_a_number",
                "num_generations": 4,
                "warmup_ratio": 0.1,
                "baseline_type": "group_mean",
            },
            history=[],
        )
        # Should not crash, and learning_rate should be numeric
        assert isinstance(params.get("learning_rate", 0), (int, float))

    def test_missing_search_space_dims_get_defaults(self, search_space):
        """Dims in search space but not in current_best get initialized."""
        proposer = PerturbationProposer(search_space, num_params_to_change=1)
        params, _ = proposer.propose(
            current_best={},  # Empty baseline
            history=[],
        )
        # All search space dims should be present
        for dim in search_space.dimensions:
            assert dim.name in params


class TestParamValidation:
    def test_validate_params_in_bounds(self, search_space):
        from stateset_agents.training.auto_research.search_spaces import (
            validate_params_against_space,
        )

        warnings = validate_params_against_space(
            {"learning_rate": 1e-5, "num_generations": 8},
            search_space,
        )
        assert warnings == []

    def test_validate_params_out_of_bounds(self, search_space):
        from stateset_agents.training.auto_research.search_spaces import (
            validate_params_against_space,
        )

        warnings = validate_params_against_space(
            {"learning_rate": 999.0, "baseline_type": "nonexistent"},
            search_space,
        )
        assert len(warnings) == 2
        assert any("learning_rate" in w for w in warnings)
        assert any("baseline_type" in w for w in warnings)


class TestJsonExtraction:
    def test_plain_json(self):
        from stateset_agents.training.auto_research.llm_proposer import _extract_json

        result = _extract_json('{"a": 1}')
        assert result == {"a": 1}

    def test_markdown_fenced_json(self):
        from stateset_agents.training.auto_research.llm_proposer import _extract_json

        result = _extract_json('```json\n{"a": 1}\n```')
        assert result == {"a": 1}

    def test_json_with_surrounding_text(self):
        from stateset_agents.training.auto_research.llm_proposer import _extract_json

        result = _extract_json('Here is my proposal:\n{"a": 1}\nDone.')
        assert result == {"a": 1}

    def test_nested_json(self):
        from stateset_agents.training.auto_research.llm_proposer import _extract_json

        result = _extract_json('{"params": {"lr": 0.001}, "description": "test"}')
        assert result["params"]["lr"] == 0.001

    def test_invalid_json_raises(self):
        from stateset_agents.training.auto_research.llm_proposer import _extract_json

        with pytest.raises(json.JSONDecodeError):
            _extract_json("not json at all")


class TestCreateProposer:
    def test_random(self, search_space):
        p = create_proposer("random", search_space)
        assert isinstance(p, RandomProposer)

    def test_perturbation(self, search_space):
        p = create_proposer("perturbation", search_space)
        assert isinstance(p, PerturbationProposer)

    def test_grid(self, search_space):
        p = create_proposer("grid", search_space)
        assert isinstance(p, GridProposer)

    def test_unknown_raises(self, search_space):
        with pytest.raises(ValueError, match="Unknown proposer"):
            create_proposer("magic", search_space)


# ---------------------------------------------------------------------------
# Module-level imports
# ---------------------------------------------------------------------------


class TestModuleImports:
    def test_training_level_imports(self):
        from stateset_agents.training import AUTO_RESEARCH_AVAILABLE

        assert AUTO_RESEARCH_AVAILABLE is True

    def test_top_level_imports(self):
        from stateset_agents import AutoResearchConfig

        assert AutoResearchConfig is not None
