"""Tests for the flywheel orchestrator — entirely against a fake executor.

The loop's value is its stopping discipline: plateau, dry harvest, budget,
perfect score. Each stop is pinned here, as is the one subtle contract —
a FAILED training job WITH eval artifacts is the eval gate speaking
(10/12), not an error, and the loop must read the score and keep going.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stateset_agents.flywheel import FlywheelConfig, run_flywheel, run_flywheel_repeats
from stateset_agents.remote.executor import RemoteExecutor
from stateset_agents.remote.job import JobHandle, JobStatus, RemoteJobResult


class ScriptedExecutor(RemoteExecutor):
    """Plays back a scripted sequence of (harvest, train) outcomes.

    Each generation consumes one entry from ``script``:
    ``{"kept": int, "samples": int, "current_eval": int | None,
    "passed": int | None, "total": int, "train_status": JobStatus,
    "cost": float | None}``.
    """

    name = "scripted"
    supports_cost_ceiling = True

    def __init__(self, script, eval_total=12):
        self.script = list(script)
        self.eval_total = eval_total
        self.specs = []  # every spec submitted, in order
        self._results = {}
        self._counter = 0

    def submit(self, spec):
        self._counter += 1
        job_id = str(self._counter)
        self.specs.append(spec)
        is_probe = (
            spec.job_kind == "harvest"
            and Path(spec.dataset).name == "probe_prompts.json"
        )
        step = self.script.pop(0) if is_probe else self.script[0]
        out = Path(spec.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if spec.job_kind == "harvest":
            summary = {
                "kept": step["kept"],
                "samples": step["samples"],
                "dry_run": step.get("summary_preview", False),
                "eval": (
                    {"passed": step["current_eval"], "total": self.eval_total}
                    if step.get("current_eval") is not None
                    else None
                ),
            }
            (out / "harvest_summary.json").write_text(json.dumps(summary))
            rows = [
                {
                    "messages": [
                        {"role": "user", "content": f"q{i}"},
                        {"role": "assistant", "content": f"a{i}"},
                    ]
                }
                for i in range(max(1, step["kept"]))
            ]
            (out / "harvest.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n"
            )
            status = JobStatus.SUCCEEDED
        else:
            step = self.script.pop(0)
            if step.get("passed") is not None:
                # The REAL on-disk shape (write_eval_results): a bare list
                # of rows with the assertion outcome nested under "checks".
                rows = [
                    {
                        "prompt": f"e{i}",
                        "base": "b",
                        "finetuned": "f",
                        "checks": {"passed": i < step["passed"]},
                    }
                    for i in range(step.get("total", self.eval_total))
                ]
                (out / "eval_results.json").write_text(json.dumps(rows))
            status = step.get("train_status", JobStatus.SUCCEEDED)
        self._results[job_id] = RemoteJobResult(
            handle=JobHandle(provider=self.name, job_id=job_id),
            status=status,
            output_dir=None if step.get("missing_output") == spec.job_kind else out,
            cost_usd=step.get("cost", 1.0),
            is_preview=step.get("preview_kind") == spec.job_kind,
        )
        return JobHandle(provider=self.name, job_id=job_id)

    def status(self, handle):
        return self._results[handle.job_id].status

    def logs(self, handle):
        yield from self._results[handle.job_id].logs

    def fetch(self, handle, dest=None):
        return self._results[handle.job_id].output_dir

    def cancel(self, handle):
        pass

    def wait(self, handle, poll_interval_s=0.0):
        return self._results[handle.job_id]


@pytest.fixture
def config(tmp_path):
    return FlywheelConfig(
        base_model="base/model",
        harvest_prompts=[{"prompt": "h1", "expect": ["x"]}],
        eval_prompts=[{"prompt": "e1", "expect": ["x"]}],
        output_root=tmp_path / "fw",
        generations=5,
    )


class TestStops:
    @pytest.mark.parametrize("kind", ["harvest", "sft"])
    def test_live_run_rejects_preview_with_existing_scores(self, config, kind):
        executor = ScriptedExecutor(
            [{"kept": 5, "samples": 10, "passed": 12, "preview_kind": kind}]
        )
        with pytest.raises(RuntimeError, match="returned a preview"):
            run_flywheel(config, executor)
        assert [spec.job_kind for spec in executor.specs] == (
            ["harvest"] if kind == "harvest" else ["harvest", "sft"]
        )
        assert not (config.output_root / "flywheel_report.json").exists()

    def test_dry_harvest_stops_the_loop_with_no_training(self, config):
        executor = ScriptedExecutor(
            [{"kept": 0, "samples": 24, "current_eval": 2, "cost": 0.5}]
        )
        report = run_flywheel(config, executor)
        assert "dry harvest" in report["stop_reason"]
        # Only the harvest was submitted — no training job for silence.
        assert [s.job_kind for s in executor.specs] == ["harvest"]
        assert report["final_adapter"] is None

    @pytest.mark.parametrize("kind", ["harvest", "sft"])
    def test_successful_job_must_return_artifacts(self, config, kind):
        executor = ScriptedExecutor([{"kept": 2, "samples": 2, "missing_output": kind}])
        with pytest.raises(RuntimeError, match="returned no artifacts"):
            run_flywheel(config, executor)

    def test_legacy_harvest_preview_cannot_supply_training_rows(self, config):
        executor = ScriptedExecutor(
            [{"kept": 3, "samples": 8, "summary_preview": True}]
        )
        with pytest.raises(RuntimeError, match="summary records a preview"):
            run_flywheel(config, executor)
        assert [spec.job_kind for spec in executor.specs] == ["harvest"]

    def test_plateau_stops_and_keeps_the_previous_adapter(self, config):
        executor = ScriptedExecutor(
            [
                {"kept": 10, "samples": 80, "current_eval": 2, "passed": 8},
                {"kept": 12, "samples": 80, "current_eval": None, "passed": 8},
            ]
        )
        report = run_flywheel(config, executor)
        assert "plateau" in report["stop_reason"]
        assert report["best_eval_passed"] == 8
        # gen-2 did not beat gen-1, so gen-1's adapter stays final.
        assert Path(report["final_adapter"]).parts[-2:] == ("gen1", "adapter")

    def test_perfect_score_stops_early(self, config):
        executor = ScriptedExecutor(
            [{"kept": 10, "samples": 80, "current_eval": 2, "passed": 12}]
        )
        report = run_flywheel(config, executor)
        assert "perfect score" in report["stop_reason"]
        assert report["best_eval_passed"] == 12

    def test_budget_ceiling_stops_before_renting(self, config):
        config.max_cost_usd = 3.0
        executor = ScriptedExecutor(
            [
                {
                    "kept": 10,
                    "samples": 80,
                    "current_eval": 1,
                    "passed": 5,
                    "cost": 2.0,
                },
                {
                    "kept": 10,
                    "samples": 80,
                    "current_eval": None,
                    "passed": 7,
                    "cost": 2.0,
                },
            ]
        )
        report = run_flywheel(config, executor)
        assert "budget" in report["stop_reason"]
        # One full generation (2 jobs at $2) exceeds $3 -> no second gen.
        assert len(report["generations"]) == 1

    def test_generations_cap_is_respected(self, config):
        config.generations = 2
        executor = ScriptedExecutor(
            [
                {"kept": 5, "samples": 40, "current_eval": 1, "passed": 4},
                {"kept": 5, "samples": 40, "current_eval": None, "passed": 6},
            ]
        )
        report = run_flywheel(config, executor)
        assert report["stop_reason"] == "generations exhausted"
        assert len(report["generations"]) == 2


class TestEvalGateContract:
    def test_failed_training_with_eval_artifacts_is_a_score_not_an_error(self, config):
        """The headroom run's 10/12 FAILED its all-assertions gate while
        being the whole point. The loop reads the score from the fetched
        artifacts and continues."""
        executor = ScriptedExecutor(
            [
                {
                    "kept": 10,
                    "samples": 80,
                    "current_eval": 2,
                    "passed": 10,
                    "train_status": JobStatus.FAILED,
                },
                {
                    "kept": 10,
                    "samples": 80,
                    "current_eval": None,
                    "passed": 10,
                    "train_status": JobStatus.FAILED,
                },
            ]
        )
        report = run_flywheel(config, executor)
        assert report["best_eval_passed"] == 10
        assert report["generations"][0]["eval_passed"] == 10

    def test_failed_training_without_artifacts_raises(self, config):
        executor = ScriptedExecutor(
            [
                {
                    "kept": 10,
                    "samples": 80,
                    "current_eval": 2,
                    "passed": None,
                    "train_status": JobStatus.FAILED,
                }
            ]
        )
        with pytest.raises(RuntimeError, match="no eval artifacts"):
            run_flywheel(config, executor)


class TestSpecWiring:
    @pytest.mark.parametrize("repeats", [None, 1, 3])
    def test_offline_plan_preserves_existing_run(self, config, repeats, monkeypatch):
        from stateset_agents.remote.runpod import RunPodExecutor

        executor = RunPodExecutor()
        monkeypatch.setattr(
            executor, "submit", lambda *_: pytest.fail("submitted a job")
        )
        config.dry_run = True
        config.teacher_base_model = "teacher"
        config.target_harvest_rate = 0.6
        config.output_root.mkdir()
        previous = config.output_root / "flywheel_report.json"
        previous.write_bytes(b"live report")
        plan = (
            run_flywheel(config, executor)
            if repeats is None
            else run_flywheel_repeats(config, executor, repeats)
        )
        jobs = plan["first_generation_jobs"]
        assert len(jobs) == 2 + len(config.probe_temperatures)
        assert jobs[-2]["base_model"] == "teacher"
        assert jobs[-2]["harvest"]["adapter_dir"] is None
        assert jobs[-1]["base_model"] == config.base_model
        root = config.output_root if repeats is None else config.output_root / "run1"
        assert Path(jobs[-1]["output_dir"]) == root / "gen1" / "adapter"
        assert plan["max_jobs"] == (repeats or 1) * config.generations * len(jobs)
        assert list(config.output_root.iterdir()) == [previous]
        assert previous.read_bytes() == b"live report"

    def test_offline_plan_validates_packaged_harvest_prompts(self, config):
        from stateset_agents.remote.runpod import RunPodExecutor

        config.dry_run = True
        config.harvest_prompts = [{"prompt": "Help"}]
        with pytest.raises(ValueError, match="success criterion"):
            run_flywheel(config, RunPodExecutor())
        assert not config.output_root.exists()

    def test_harvest_then_train_and_parent_adapter_chains(self, config):
        config.generations = 2
        executor = ScriptedExecutor(
            [
                {"kept": 5, "samples": 40, "current_eval": 1, "passed": 4},
                {"kept": 5, "samples": 40, "current_eval": None, "passed": 6},
            ]
        )
        run_flywheel(config, executor)
        kinds = [s.job_kind for s in executor.specs]
        assert kinds == ["harvest", "sft", "harvest", "sft"]
        # gen-2's harvest samples FROM gen-1's adapter; gen-2's training
        # records gen-1 as parent.
        gen2_harvest = executor.specs[2]
        assert Path(gen2_harvest.harvest["adapter_dir"]).parts[-2:] == (
            "gen1",
            "adapter",
        )
        gen2_train = executor.specs[3]
        assert Path(gen2_train.parent_adapter).parts[-2:] == ("gen1", "adapter")

    def test_remaining_budget_shrinks_per_job(self, config):
        config.max_cost_usd = 10.0
        executor = ScriptedExecutor(
            [
                {"kept": 5, "samples": 40, "current_eval": 1, "passed": 4, "cost": 3.0},
                {
                    "kept": 5,
                    "samples": 40,
                    "current_eval": None,
                    "passed": 6,
                    "cost": 3.0,
                },
            ]
        )
        run_flywheel(config, executor)
        ceilings = [s.max_cost_usd for s in executor.specs]
        assert ceilings[0] == 10.0  # nothing spent yet
        assert ceilings[1] == 7.0  # harvest cost $3
        assert ceilings[2] == 4.0  # + train $3

    def test_report_is_written_to_disk(self, config):
        executor = ScriptedExecutor(
            [{"kept": 5, "samples": 40, "current_eval": 1, "passed": 12}]
        )
        report = run_flywheel(config, executor)
        on_disk = json.loads((config.output_root / "flywheel_report.json").read_text())
        assert on_disk == report
        assert on_disk["total_cost_usd"] == pytest.approx(2.0)


class TestRepeats:
    """--repeats turns 'reproduced' into a distribution: two live runs
    scored 7/12 and 11/12 — a spread wide enough that any single run
    misstates the mechanism."""

    def _script_for_one_run(self, passed, cost=1.0):
        return [
            {
                "kept": 5,
                "samples": 40,
                "current_eval": 1,
                "passed": passed,
                "cost": cost,
            }
        ]

    def test_aggregates_scores_across_runs(self, config):
        from stateset_agents.flywheel import run_flywheel_repeats

        config.generations = 1
        executor = ScriptedExecutor(
            self._script_for_one_run(7) + self._script_for_one_run(11)
        )
        report = run_flywheel_repeats(config, executor, repeats=2)

        assert report["scores"] == [7, 11]
        assert report["min"] == 7 and report["max"] == 11
        assert report["mean"] == 9.0
        assert report["completed"] == 2

    def test_each_run_gets_its_own_output_root(self, config):
        from stateset_agents.flywheel import run_flywheel_repeats

        config.generations = 1
        executor = ScriptedExecutor(
            self._script_for_one_run(5) + self._script_for_one_run(6)
        )
        run_flywheel_repeats(config, executor, repeats=2)

        assert (config.output_root / "run1" / "flywheel_report.json").exists()
        assert (config.output_root / "run2" / "flywheel_report.json").exists()
        assert (config.output_root / "flywheel_repeats_report.json").exists()

    def test_budget_is_shared_and_exhaustion_skips_later_runs(self, config):
        from stateset_agents.flywheel import run_flywheel_repeats

        config.generations = 1
        config.max_cost_usd = 3.0
        executor = ScriptedExecutor(self._script_for_one_run(5, cost=2.0))
        report = run_flywheel_repeats(config, executor, repeats=3)

        # Run 1 spends 2x$2=4 > $3 total; runs 2 and 3 are skipped, loudly.
        skipped = [r for r in report["runs"] if r.get("skipped")]
        assert len(skipped) == 2
        assert "budget exhausted" in skipped[0]["skipped"]
        assert report["completed"] == 1

    def test_zero_repeats_is_refused(self, config):
        import pytest

        from stateset_agents.flywheel import run_flywheel_repeats

        with pytest.raises(ValueError, match="repeats"):
            run_flywheel_repeats(config, ScriptedExecutor([]), repeats=0)


class TestDistillation:
    """Teacher harvests, student trains: the 35B clears walls the 9B
    cannot, but the 9B is what you want to serve."""

    def test_harvest_uses_the_fixed_teacher_and_train_uses_the_student(self, config):
        config.generations = 2
        config.teacher_base_model = "big/teacher-35b"
        config.teacher_adapter = Path("outputs/teacher_ckpt")
        executor = ScriptedExecutor(
            [
                {"kept": 5, "samples": 40, "current_eval": None, "passed": 4},
                {"kept": 5, "samples": 40, "current_eval": None, "passed": 6},
            ]
        )
        run_flywheel(config, executor)

        harvests = [s for s in executor.specs if s.job_kind == "harvest"]
        trains = [s for s in executor.specs if s.job_kind == "sft"]
        # Every harvest samples the TEACHER with its fixed adapter...
        assert all(h.base_model == "big/teacher-35b" for h in harvests)
        assert all(
            h.harvest["adapter_dir"] == str(Path("outputs/teacher_ckpt"))
            for h in harvests
        )
        # ...the teacher never advances, and never gets eval prompts.
        assert all(h.eval_prompts is None for h in harvests)
        # Every train job trains the STUDENT, chaining student lineage.
        assert all(t.base_model == "base/model" for t in trains)
        assert Path(trains[1].parent_adapter).parts[-2:] == ("gen1", "adapter")

    def test_student_scores_drive_the_stopping_rules(self, config):
        config.teacher_base_model = "big/teacher-35b"
        config.teacher_adapter = Path("t")
        executor = ScriptedExecutor(
            [
                {"kept": 5, "samples": 40, "current_eval": None, "passed": 12},
            ]
        )
        report = run_flywheel(config, executor)
        assert "perfect score" in report["stop_reason"]
        assert report["best_eval_passed"] == 12


class TestRarityController:
    """The thermostat: probe temperatures, harvest in the measured window."""

    def _probe_step(self, kept, samples=24):
        return {"kept": kept, "samples": samples, "current_eval": None}

    def test_all_stages_and_repeats_share_one_budget(self, config):
        config.generations, config.target_harvest_rate = 1, 0.6
        config.max_cost_usd = 10
        script = [self._probe_step(2) for _ in config.probe_temperatures]
        executor = ScriptedExecutor(
            (script + [{"kept": 3, "samples": 8, "passed": 12}]) * 2
        )
        report = run_flywheel_repeats(config, executor, repeats=2)
        assert [s.max_cost_usd for s in executor.specs] == list(range(10, 0, -1))
        assert report["total_cost_usd"] == 10 and report["completed"] == 2
        first = json.loads(
            (config.output_root / "run1/flywheel_report.json").read_text()
        )
        assert first["generations"][0]["cost_usd"] == 5

    @pytest.mark.parametrize("cost", [1, None])
    def test_exhausted_or_unknown_probe_cost_stops_before_next_rental(
        self, config, cost
    ):
        config.max_cost_usd, config.target_harvest_rate = 1, 0.6
        executor = ScriptedExecutor([dict(self._probe_step(2), cost=cost)])
        report = run_flywheel_repeats(config, executor, repeats=2)
        assert len(executor.specs) == 1
        assert "budget" in report["runs"][0]["stop_reason"]
        assert "budget" in report["runs"][1]["skipped"]
        assert report["unpriced_jobs"] == int(cost is None)

    def test_exhausted_harvest_is_reported_before_training(self, config):
        config.max_cost_usd = 1.0
        executor = ScriptedExecutor([{"samples": 8, "kept": 2, "cost": 1.0}])
        report = run_flywheel(config, executor)
        assert [s.job_kind for s in executor.specs] == ["harvest"]
        assert report["generations"][0]["harvest_kept"] == 2
        assert report["stop_reason"] == "budget exhausted"

    def test_probe_submission_error_stops_a_budgeted_campaign(
        self, config, monkeypatch
    ):
        config.max_cost_usd, config.target_harvest_rate = 1, 0.6
        executor = ScriptedExecutor([])
        attempts = []

        def submit(spec):
            attempts.append(spec)
            raise OSError("provider unavailable")

        monkeypatch.setattr(executor, "submit", submit)
        with pytest.raises(RuntimeError, match="provider unavailable"):
            run_flywheel_repeats(config, executor, repeats=2)
        assert len(attempts) == 1 and attempts[0].max_cost_usd == 1

    def test_chooses_the_temperature_nearest_the_target(self, config):
        config.generations = 1
        config.target_harvest_rate = 0.6
        config.probe_temperatures = (0.7, 0.9, 1.1)
        executor = ScriptedExecutor(
            [
                self._probe_step(22),  # t=0.7 -> 92%
                self._probe_step(14),  # t=0.9 -> 58%  <- nearest 60%
                self._probe_step(5),  # t=1.1 -> 21%
                {"kept": 10, "samples": 80, "current_eval": 2, "passed": 12},
            ]
        )
        report = run_flywheel(config, executor)

        assert "perfect score" in report["stop_reason"]
        # The real harvest (the 4th submitted spec) ran at the chosen temp.
        real_harvest = executor.specs[3]
        assert real_harvest.harvest["temperature"] == 0.9
        # Probes were tiny: subset of prompts, small best_of.
        assert executor.specs[0].harvest["best_of"] == config.probe_best_of

    def test_no_target_means_no_probes(self, config):
        config.generations = 1
        executor = ScriptedExecutor(
            [{"kept": 10, "samples": 80, "current_eval": 2, "passed": 12}]
        )
        run_flywheel(config, executor)
        assert len([s for s in executor.specs if s.job_kind == "harvest"]) == 1
