"""Unit tests for ``stateset-agents train-remote``."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from stateset_agents.cli import app

runner = CliRunner()

# Wide, plain terminal so typer/rich cannot wrap or truncate flag names.
# A narrow CI pty (Windows runners default to one) renders "--provider" as
# "--provi…" and any assertion against help text then fails for reasons that
# have nothing to do with the CLI. Same lesson as _help_flags in
# test_cli_improve.py.
_WIDE_TERMINAL = {"COLUMNS": "200", "TERM": "dumb", "NO_COLOR": "1"}


def invoke_help(*args: str):
    """Invoke ``--help`` under a terminal wide enough to render flags intact."""
    return runner.invoke(app, [*args, "--help"], env=_WIDE_TERMINAL)


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "curated.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            }
        )
        + "\n"
    )
    return path


class TestCommandRegistration:
    def test_command_is_registered(self):
        """Asserted against the parser, not rendered text — help rendering
        depends on terminal width and rich's version, neither of which is
        what this test is about."""
        names = {
            command.name or command.callback.__name__
            for command in app.registered_commands
        }

        assert "train-remote" in names

    def test_command_exposes_the_provider_option(self):
        import inspect

        command = next(c for c in app.registered_commands if c.name == "train-remote")
        params = inspect.signature(command.callback).parameters

        assert "provider" in params

    def test_help_lists_the_available_providers(self):
        result = invoke_help("train-remote")

        assert result.exit_code == 0
        assert "local" in result.output
        assert "modal" in result.output

    def test_remote_job_command_is_registered(self):
        names = {
            command.name or command.callback.__name__
            for command in app.registered_commands
        }
        assert "remote-job" in names

    def test_remote_providers_reports_capability_matrix(self):
        result = runner.invoke(app, ["remote-providers", "--json"])

        assert result.exit_code == 0, result.output
        rows = {row["provider"]: row for row in json.loads(result.output)}
        assert rows["river"]["job_kinds"] == ["harvest", "rl", "sft"]
        assert rows["fireworks"]["durable_handles"] is True
        assert rows["runpod"]["result_kind"] == "local_artifacts"


class TestSuccessfulRun:
    def test_local_dry_run_exits_zero(self, dataset, tmp_path):
        result = runner.invoke(
            app,
            [
                "train-remote",
                "--provider",
                "local",
                "--dataset",
                str(dataset),
                "--base-model",
                "Qwen/Qwen3.5-0.8B",
                "--output-dir",
                str(tmp_path / "out"),
                "--dry-run",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Qwen/Qwen3.5-0.8B" in result.output


class TestFailurePaths:
    def test_unknown_provider_exits_nonzero_and_names_valid_options(
        self, dataset, tmp_path
    ):
        result = runner.invoke(
            app,
            [
                "train-remote",
                "--provider",
                "aws-batch",
                "--dataset",
                str(dataset),
                "--base-model",
                "Qwen/Qwen3.5-0.8B",
            ],
        )

        assert result.exit_code != 0
        assert "local" in result.output

    def test_missing_dataset_exits_nonzero(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "train-remote",
                "--provider",
                "local",
                "--dataset",
                str(tmp_path / "absent.jsonl"),
                "--base-model",
                "Qwen/Qwen3.5-0.8B",
            ],
        )

        assert result.exit_code != 0
        assert "does not exist" in result.output

    def test_failed_job_exits_nonzero(self, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")

        result = runner.invoke(
            app,
            [
                "train-remote",
                "--provider",
                "local",
                "--dataset",
                str(empty),
                "--base-model",
                "Qwen/Qwen3.5-0.8B",
                "--dry-run",
            ],
        )

        assert result.exit_code != 0


def capture_spec(monkeypatch):
    """Spy on LocalExecutor.submit, recording the spec the CLI built."""
    captured = {}

    from stateset_agents.remote.local import LocalExecutor

    original = LocalExecutor.submit

    def spy(self, spec):
        captured["spec"] = spec
        return original(self, spec)

    monkeypatch.setattr(LocalExecutor, "submit", spy)
    return captured


class TestOptionPassthrough:
    """Resource flags must reach the executor, not be silently dropped."""

    def test_resource_and_hyperparameter_flags_reach_the_spec(
        self, dataset, tmp_path, monkeypatch
    ):
        captured = capture_spec(monkeypatch)

        result = runner.invoke(
            app,
            [
                "train-remote",
                "--provider",
                "local",
                "--dataset",
                str(dataset),
                "--base-model",
                "Qwen/Qwen3.5-0.8B",
                "--output-dir",
                str(tmp_path / "out"),
                "--gpu",
                "H100",
                "--gpu-count",
                "2",
                "--timeout",
                "900",
                "--package-version",
                "1.2.3",
                "--container-disk-gb",
                "160",
                "--lora-r",
                "8",
                "--num-epochs",
                "7",
                "--cloud-type",
                "COMMUNITY",
                "--network-volume-id",
                "vol-xyz",
                "--resume",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0, result.output
        spec = captured["spec"]
        assert spec.gpu == "H100"
        assert spec.gpu_count == 2
        assert spec.timeout_s == 900
        assert spec.package_version == "1.2.3"
        assert spec.container_disk_gb == 160
        assert spec.lora_r == 8
        assert spec.num_epochs == 7
        assert spec.cloud_type == "COMMUNITY"
        assert spec.network_volume_id == "vol-xyz"
        assert spec.resume is True

    def test_cloud_type_and_resume_default_off(self, dataset, tmp_path, monkeypatch):
        captured = capture_spec(monkeypatch)

        result = runner.invoke(
            app,
            [
                "train-remote",
                "--provider",
                "local",
                "--dataset",
                str(dataset),
                "--base-model",
                "Qwen/Qwen3.5-0.8B",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0, result.output
        assert captured["spec"].cloud_type == "SECURE"
        assert captured["spec"].resume is False
        assert captured["spec"].network_volume_id is None

    def test_invalid_cloud_type_is_rejected_before_submitting(
        self, dataset, tmp_path, monkeypatch
    ):
        captured = capture_spec(monkeypatch)

        result = runner.invoke(
            app,
            [
                "train-remote",
                "--provider",
                "local",
                "--dataset",
                str(dataset),
                "--base-model",
                "Qwen/Qwen3.5-0.8B",
                "--cloud-type",
                "SPOT",
            ],
        )

        assert result.exit_code == 2
        assert "cloud_type" in result.output
        assert "spec" not in captured

    def test_invalid_hyperparameter_is_rejected_before_submitting(
        self, dataset, tmp_path, monkeypatch
    ):
        captured = capture_spec(monkeypatch)

        result = runner.invoke(
            app,
            [
                "train-remote",
                "--provider",
                "local",
                "--dataset",
                str(dataset),
                "--base-model",
                "Qwen/Qwen3.5-0.8B",
                "--num-epochs",
                "0",
            ],
        )

        assert result.exit_code == 2
        assert "num_epochs" in result.output
        assert "spec" not in captured


class TestEvalPromptsOption:
    """--eval-prompts is a local file, read on this machine — the prompts
    travel inside the spec so pods need no second upload."""

    def _invoke(self, dataset, tmp_path, *extra):
        return runner.invoke(
            app,
            [
                "train-remote",
                "--provider",
                "local",
                "--dataset",
                str(dataset),
                "--base-model",
                "Qwen/Qwen3.5-0.8B",
                "--output-dir",
                str(tmp_path / "out"),
                "--dry-run",
                *extra,
            ],
        )

    def test_prompts_file_is_read_into_the_spec(self, dataset, tmp_path, monkeypatch):
        captured = capture_spec(monkeypatch)
        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("what's the return policy?\n\n  plain prompt  \n")

        result = self._invoke(dataset, tmp_path, "--eval-prompts", str(prompts_file))

        assert result.exit_code == 0, result.output
        assert captured["spec"].eval_prompts == [
            "what's the return policy?",
            "plain prompt",
        ]

    def test_json_object_lines_become_spec_dicts_amid_plain_lines(
        self, dataset, tmp_path, monkeypatch
    ):
        """A line that parses as a JSON object is a prompt spec; every other
        line — including JSON that isn't an object — stays a plain prompt."""
        captured = capture_spec(monkeypatch)
        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text(
            "plain prompt\n"
            '{"prompt": "Say 41.", "expect": ["41"], "forbid": ["sorry"]}\n'
            '["not", "an", "object"]\n'
        )

        result = self._invoke(dataset, tmp_path, "--eval-prompts", str(prompts_file))

        assert result.exit_code == 0, result.output
        assert captured["spec"].eval_prompts == [
            "plain prompt",
            {"prompt": "Say 41.", "expect": ["41"], "forbid": ["sorry"]},
            '["not", "an", "object"]',
        ]

    def test_a_malformed_spec_line_exits_2_with_the_reason(
        self, dataset, tmp_path, monkeypatch
    ):
        captured = capture_spec(monkeypatch)
        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text('{"expect": ["no prompt key"]}\n')

        result = self._invoke(dataset, tmp_path, "--eval-prompts", str(prompts_file))

        assert result.exit_code == 2
        assert "prompt" in result.output
        assert "spec" not in captured

    def test_omitting_the_option_leaves_the_spec_unset(
        self, dataset, tmp_path, monkeypatch
    ):
        captured = capture_spec(monkeypatch)

        result = self._invoke(dataset, tmp_path)

        assert result.exit_code == 0, result.output
        assert captured["spec"].eval_prompts is None
        assert captured["spec"].eval_max_new_tokens == 90

    def test_eval_max_new_tokens_reaches_the_spec(self, dataset, tmp_path, monkeypatch):
        captured = capture_spec(monkeypatch)

        result = self._invoke(dataset, tmp_path, "--eval-max-new-tokens", "300")

        assert result.exit_code == 0, result.output
        assert captured["spec"].eval_max_new_tokens == 300

    def test_missing_prompts_file_exits_2_with_a_clear_message(
        self, dataset, tmp_path, monkeypatch
    ):
        captured = capture_spec(monkeypatch)

        result = self._invoke(
            dataset, tmp_path, "--eval-prompts", str(tmp_path / "absent.txt")
        )

        assert result.exit_code == 2
        assert "does not exist" in result.output
        assert "absent.txt" in result.output
        assert "spec" not in captured


class TestFireworksOptions:
    """`--deploy` rents hardware, so it is opt-in and separately flagged."""

    def _params(self):
        import inspect

        command = next(c for c in app.registered_commands if c.name == "train-remote")
        return inspect.signature(command.callback).parameters

    def test_deploy_is_off_by_default(self):
        assert self._params()["deploy"].default.default is False

    def test_deploy_accelerator_can_be_chosen(self):
        assert "deploy_accelerator" in self._params()

    def test_fireworks_is_offered_as_a_provider(self):
        result = invoke_help("train-remote")

        assert "fireworks" in result.output


class TestRemoteJobCommand:
    def test_status_reconnects_with_provider_handle(self, monkeypatch):
        import stateset_agents.cli_remote as cli_remote
        from stateset_agents.remote.job import JobStatus

        observed = {}

        class StubExecutor:
            name = "fireworks"

            def status(self, handle):
                observed["handle"] = handle
                return JobStatus.RUNNING

            def logs(self, handle):
                yield "50% complete"

        monkeypatch.setattr(cli_remote, "get_executor", lambda provider: StubExecutor())

        result = runner.invoke(app, ["remote-job", "--job-id", "sftj-1"])

        assert result.exit_code == 0, result.output
        assert observed["handle"].job_id == "sftj-1"
        assert "50% complete" in result.output
        assert "running" in result.output

    def test_fetch_uses_requested_destination(self, monkeypatch, tmp_path):
        import stateset_agents.cli_remote as cli_remote
        from stateset_agents.remote.job import JobStatus

        observed = {}

        class StubExecutor:
            name = "fireworks"

            def status(self, handle):
                return JobStatus.SUCCEEDED

            def logs(self, handle):
                return iter(())

            def fetch(self, handle, dest=None):
                observed["dest"] = dest
                return dest

        monkeypatch.setattr(cli_remote, "get_executor", lambda provider: StubExecutor())
        destination = tmp_path / "recovered"

        result = runner.invoke(
            app,
            [
                "remote-job",
                "--job-id",
                "sftj-1",
                "--fetch",
                "--output-dir",
                str(destination),
            ],
        )

        assert result.exit_code == 0, result.output
        assert observed["dest"] == destination
        assert str(destination) in result.output


class TestRunPodOrphansCommand:
    def test_read_only_mode_lists_leases_without_terminating(self, monkeypatch):
        import stateset_agents.cli_remote as cli_remote
        from stateset_agents.remote.runpod import RunPodExecutor

        executor = RunPodExecutor(api=object(), ssh=object(), public_key="key")
        monkeypatch.setattr(
            executor,
            "orphaned_leases",
            lambda: [
                {
                    "pod_id": "pod-1",
                    "job_id": "job-1",
                    "base_model": "Qwen/test",
                    "gpu": "H100",
                }
            ],
        )
        monkeypatch.setattr(
            executor,
            "cleanup_orphans",
            lambda: pytest.fail("read-only mode must not terminate"),
        )
        monkeypatch.setattr(cli_remote, "get_executor", lambda provider: executor)

        result = runner.invoke(app, ["runpod-orphans"])

        assert result.exit_code == 0, result.output
        assert "pod-1" in result.output
        assert "Read-only" in result.output


class TestUndeployCommand:
    """`undeploy` tears down a provider-managed deployment."""

    def test_undeploy_delegates_to_the_executor(self, monkeypatch):
        import stateset_agents.cli_remote as cli_remote

        recorded = {}

        class StubExecutor:
            name = "fireworks"

            def undeploy(self, deployment):
                recorded["deployment"] = deployment

        monkeypatch.setattr(cli_remote, "get_executor", lambda provider: StubExecutor())

        result = runner.invoke(
            app, ["undeploy", "--deployment", "dep-1"], env=_WIDE_TERMINAL
        )

        assert result.exit_code == 0, result.output
        assert recorded["deployment"] == "dep-1"
        assert "dep-1" in result.output

    def test_undeploy_on_a_provider_without_deployments_exits_nonzero(
        self, monkeypatch
    ):
        import stateset_agents.cli_remote as cli_remote
        from stateset_agents.remote.executor import RemoteExecutor

        class BareExecutor(RemoteExecutor):
            name = "bare"

            def submit(self, spec):
                raise NotImplementedError

            def status(self, handle):
                raise NotImplementedError

            def logs(self, handle):
                raise NotImplementedError

            def fetch(self, handle, dest=None):
                raise NotImplementedError

            def cancel(self, handle):
                raise NotImplementedError

        monkeypatch.setattr(cli_remote, "get_executor", lambda provider: BareExecutor())

        result = runner.invoke(
            app, ["undeploy", "--deployment", "dep-1"], env=_WIDE_TERMINAL
        )

        assert result.exit_code == 1
        assert "BareExecutor" in result.output
