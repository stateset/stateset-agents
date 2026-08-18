"""Tests for the harvest job and its ride on the executor contract."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from stateset_agents.remote.job import RemoteJobSpec
from stateset_agents.training.harvest import build_harvest_rows, run_harvest_job


class TestBuildHarvestRows:
    SPEC = {"prompt": "order #1?", "expect": ["on the way"], "forbid": ["sorry no"]}

    def test_keeps_only_samples_passing_the_checks(self):
        rows = build_harvest_rows(
            self.SPEC,
            ["Your order is On The Way!", "sorry no idea", "no tracking exists"],
        )
        assert len(rows) == 1
        assert rows[0]["messages"][1]["content"] == "Your order is On The Way!"

    def test_keeps_every_passing_sample_not_just_the_first(self):
        """The headroom run's 58-row set came from multiple passes per
        prompt; thinning them would thin the training signal."""
        rows = build_harvest_rows(self.SPEC, ["on the way, A", "on the way, B", "nope"])
        assert len(rows) == 2

    def test_rows_are_ingest_ready_chat_format(self):
        (row,) = build_harvest_rows(self.SPEC, ["on the way"])
        assert [m["role"] for m in row["messages"]] == ["user", "assistant"]
        assert row["messages"][0]["content"] == "order #1?"


class TestRunHarvestJob:
    def _payload(self, tmp_path, **overrides):
        payload = {
            "base_model": "base/model",
            "adapter_dir": None,
            "harvest_prompts": [{"prompt": "p", "expect": ["x"]}],
            "eval_prompts": None,
            "output_dir": str(tmp_path / "out"),
            "best_of": 4,
            "temperature": 0.9,
            "top_p": 0.95,
            "max_new_tokens": 100,
            "dry_run": True,
        }
        payload.update(overrides)
        return payload

    def test_a_prompt_without_checks_is_refused_loudly(self, tmp_path):
        """Without expect/forbid every sample passes and the harvest is
        noise — refused before any GPU spends a cent."""
        payload = self._payload(tmp_path, harvest_prompts=[{"prompt": "unchecked"}])
        with pytest.raises(ValueError, match="no expect/forbid checks"):
            run_harvest_job(payload)

    def test_dry_run_writes_the_summary_and_reports_it(self, tmp_path):
        summary = run_harvest_job(self._payload(tmp_path))
        assert summary["dry_run"] is True
        on_disk = json.loads((tmp_path / "out" / "harvest_summary.json").read_text())
        assert on_disk["kept"] == 0
        assert on_disk["prompts"] == 1

    def test_cli_accepts_a_prompts_file(self, tmp_path):
        """The executors upload the prompts as a file — the module must read
        it. Run as a real subprocess, exactly like the pod would."""
        prompts = tmp_path / "prompts.json"
        prompts.write_text(json.dumps([{"prompt": "p", "expect": ["x"]}]))
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "stateset_agents.training.harvest",
                "--base-model",
                "base/model",
                "--prompts-file",
                str(prompts),
                "--output-dir",
                str(tmp_path / "out"),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "out" / "harvest_summary.json").exists()


class TestSpecHarvestArgs:
    def _spec(self, tmp_path, **overrides):
        prompts = tmp_path / "prompts.json"
        prompts.write_text("[]")
        defaults = {
            "dataset": prompts,
            "base_model": "base/model",
            "output_dir": tmp_path / "out",
            "job_kind": "harvest",
            "harvest": {"adapter_dir": "outputs/gen1", "best_of": 4},
        }
        defaults.update(overrides)
        return RemoteJobSpec(**defaults)

    def test_to_cli_args_targets_the_harvest_module_shape(self, tmp_path):
        args = self._spec(tmp_path).to_cli_args()
        assert "--prompts-file" in args
        assert "--best-of" in args
        assert args[args.index("--best-of") + 1] == "4"
        assert "--adapter" in args
        assert "--num-epochs" not in args  # training-only knob

    def test_adapter_dir_override_wins(self, tmp_path):
        """Executors pass the REMOTE path they shipped the adapter to."""
        args = self._spec(tmp_path).harvest_cli_args(
            adapter_dir="/workspace/current_adapter"
        )
        assert args[args.index("--adapter") + 1] == "/workspace/current_adapter"

    def test_sft_specs_are_unaffected(self, tmp_path):
        dataset = tmp_path / "d.jsonl"
        dataset.write_text("{}\n")
        args = RemoteJobSpec(
            dataset=dataset, base_model="m", output_dir=tmp_path / "o"
        ).to_cli_args()
        assert "--num-epochs" in args
        assert "--best-of" not in args


class TestLocalExecutorHarvest:
    def test_harvest_job_runs_the_harvest_module(self, tmp_path):
        from stateset_agents.remote.job import JobStatus
        from stateset_agents.remote.local import LocalExecutor

        prompts = tmp_path / "prompts.json"
        prompts.write_text(json.dumps([{"prompt": "p", "expect": ["x"]}]))
        spec = RemoteJobSpec(
            dataset=prompts,
            base_model="base/model",
            output_dir=tmp_path / "out",
            job_kind="harvest",
            harvest={"best_of": 2},
            dry_run=True,
        )
        executor = LocalExecutor()
        result = executor.wait(executor.submit(spec))
        assert result.status is JobStatus.SUCCEEDED, "\n".join(result.logs)
        assert (tmp_path / "out" / "harvest_summary.json").exists()


class TestRunPodHarvestCommands:
    def test_remote_commands_run_the_harvest_module_with_remote_paths(self, tmp_path):
        from stateset_agents.remote.runpod import RunPodExecutor

        prompts = tmp_path / "prompts.json"
        prompts.write_text("[]")
        spec = RemoteJobSpec(
            dataset=prompts,
            base_model="base/model",
            output_dir=tmp_path / "out",
            job_kind="harvest",
            harvest={"adapter_dir": str(tmp_path / "gen1"), "best_of": 8},
            package_version="0.31.0",
        )
        executor = RunPodExecutor(api=object(), ssh=object(), public_key="k")
        commands = executor._remote_commands(spec, "/workspace/prompts.json")

        assert commands[0].startswith("pip install")
        run = commands[-1]
        assert "stateset_agents.training.harvest" in run
        assert "/workspace/prompts.json" in run
        assert "/workspace/out" in run
        # The adapter argument points at the pod-side unpack location, never
        # the submitting machine's filesystem.
        assert "--adapter /workspace/current_adapter" in run
        assert str(tmp_path) not in run


class TestJudgeGate:
    """Semantic success criteria on top of (or instead of) substrings —
    the step toward real-data flywheels where success is not a token."""

    SPEC = {
        "prompt": "help me",
        "expect": ["resolved"],
        "judge": "customer_support",
        "min_judge_score": 0.8,
    }

    def _passes(self, monkeypatch, score, spec=None, sample="resolved it for you"):
        import stateset_agents.training.harvest as h

        monkeypatch.setattr(
            h, "judge_completion", lambda judge, prompt, completion: score
        )
        from stateset_agents.training.harvest import sample_passes

        return sample_passes(spec or self.SPEC, sample)

    def test_judge_above_threshold_passes(self, monkeypatch):
        assert self._passes(monkeypatch, 0.9) is True

    def test_judge_below_threshold_rejects(self, monkeypatch):
        assert self._passes(monkeypatch, 0.5) is False

    def test_unavailable_judge_rejects_rather_than_waves_through(self, monkeypatch):
        """A broken judge must not silently become a pass — that harvests
        noise with a green checkmark on it."""
        assert self._passes(monkeypatch, None) is False

    def test_substring_failure_short_circuits_before_the_judge(self, monkeypatch):
        import stateset_agents.training.harvest as h

        def exploding_judge(judge, prompt, completion):
            raise AssertionError("judge must not run when substrings fail")

        monkeypatch.setattr(h, "judge_completion", exploding_judge)
        from stateset_agents.training.harvest import sample_passes

        assert sample_passes(self.SPEC, "no magic word here") is False

    def test_judge_only_specs_are_allowed(self, monkeypatch, tmp_path):
        """No expect/forbid at all — the judge IS the criterion."""
        from stateset_agents.training.harvest import run_harvest_job

        summary = run_harvest_job(
            {
                "base_model": "base/model",
                "adapter_dir": None,
                "harvest_prompts": [{"prompt": "p", "judge": "customer_support"}],
                "eval_prompts": None,
                "output_dir": str(tmp_path / "out"),
                "best_of": 2,
                "temperature": 0.9,
                "top_p": 0.95,
                "max_new_tokens": 50,
                "dry_run": True,
            }
        )
        assert summary["prompts"] == 1
