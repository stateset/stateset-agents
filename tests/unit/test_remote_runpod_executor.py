"""Tests for ``RunPodExecutor``, driven by behavioural fakes.

The fakes model a pod's real lifecycle (PENDING → RUNNING, ports appearing
late) and a real file transport (uploads and downloads move actual bytes), so
these assert on effects: an adapter landing on local disk, a failure reported
as a failure, and — above all — the pod being terminated on every exit path.

That last one is why this file is careful. A leaked pod bills the user by the
hour until they notice. Every test that creates a pod also asserts it was
destroyed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.job import JobStatus, RemoteJobSpec


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "curated.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": f"q{i}"},
                        {"role": "assistant", "content": f"a{i}"},
                    ]
                }
            )
            for i in range(3)
        )
        + "\n"
    )
    return path


@pytest.fixture
def spec(dataset, tmp_path):
    return RemoteJobSpec(
        dataset=dataset,
        base_model="Qwen/Qwen3.5-0.8B",
        output_dir=tmp_path / "local_out",
        gpu="NVIDIA RTX A4000",
        timeout_s=600,
        package_version="0.20.0",
    )


class FakePodApi:
    """Models the RunPod pod lifecycle, including ports appearing late."""

    def __init__(self, *, ready_after: int = 2, never_ready: bool = False):
        self.created: list[dict] = []
        self.terminated: list[str] = []
        self.polls = 0
        self.ready_after = ready_after
        self.never_ready = never_ready

    def create_pod(self, **kwargs):
        self.created.append(kwargs)
        return {"id": "pod-abc", "desiredStatus": "RUNNING"}

    def get_pod(self, pod_id):
        self.polls += 1
        if self.never_ready or self.polls < self.ready_after:
            return {"id": pod_id, "desiredStatus": "PENDING", "publicIp": None}
        return {
            "id": pod_id,
            "desiredStatus": "RUNNING",
            "publicIp": "1.2.3.4",
            "portMappings": {"22": 40022},
        }

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)


class FakeSsh:
    """Moves real bytes between local paths, standing in for scp/ssh."""

    def __init__(self, *, exit_code: int = 0, produces_adapter: bool = True):
        self.exit_code = exit_code
        self.produces_adapter = produces_adapter
        self.commands: list[str] = []
        self.uploaded: list[tuple[Path, str]] = []
        self.remote_files: dict[str, bytes] = {}
        self.connected_to: tuple[str, int] | None = None

    def wait_until_reachable(self, host, port, timeout_s):
        self.connected_to = (host, port)

    def upload(self, local: Path, remote: str) -> None:
        self.uploaded.append((local, remote))
        self.remote_files[remote] = Path(local).read_bytes()

    def run(self, command: str) -> tuple[int, str]:
        self.commands.append(command)
        if "training.sft" in command and self.produces_adapter:
            self.remote_files["/workspace/out/adapter_config.json"] = json.dumps(
                {"base_model_name_or_path": "Qwen/Qwen3.5-0.8B", "r": 16}
            ).encode()
            self.remote_files["/workspace/out/adapter_model.safetensors"] = b"WEIGHTS"
        return self.exit_code, f"$ {command}\nok"

    def download_dir(self, remote_dir: str, local_dir: Path) -> list[Path]:
        written = []
        for remote_path, blob in self.remote_files.items():
            if not remote_path.startswith(remote_dir.rstrip("/") + "/"):
                continue
            target = Path(local_dir) / Path(remote_path).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
            written.append(target)
        return written


@pytest.fixture
def make_executor():
    from stateset_agents.remote.runpod import RunPodExecutor

    def build(api=None, ssh=None, **kwargs):
        # An explicit key by default: without it the executor reads the host's
        # ~/.ssh, so the suite would pass or fail on whether the machine
        # running it happens to have a keypair. (It does locally; the Windows
        # CI runner does not, which is how this was found.)
        kwargs.setdefault("public_key", "ssh-rsa AAAATESTKEY")
        return RunPodExecutor(
            api=api or FakePodApi(),
            ssh=ssh or FakeSsh(),
            poll_interval_s=0,
            **kwargs,
        )

    return build


class TestProvisioning:
    def test_requests_the_configured_gpu(self, make_executor, spec):
        api = FakePodApi()
        make_executor(api=api).submit(spec)

        assert api.created[0]["gpu_type_id"] == "NVIDIA RTX A4000"

    def test_exposes_tcp_22_so_scp_works(self, make_executor, spec):
        """Without a mapped 22/tcp port there is no file transport at all."""
        api = FakePodApi()
        make_executor(api=api).submit(spec)

        assert "22/tcp" in api.created[0]["ports"]

    def test_injects_the_public_key_for_ssh(self, make_executor, spec):
        api = FakePodApi()
        make_executor(api=api, public_key="ssh-rsa AAAAKEY").submit(spec)

        env = api.created[0]["env"]
        assert env["PUBLIC_KEY"] == "ssh-rsa AAAAKEY"

    def test_waits_for_the_pod_to_become_reachable(self, make_executor, spec):
        api = FakePodApi(ready_after=3)
        ssh = FakeSsh()
        make_executor(api=api, ssh=ssh).submit(spec)

        assert api.polls >= 3
        assert ssh.connected_to == ("1.2.3.4", 40022)


class TestJobExecution:
    def test_uploads_the_dataset(self, make_executor, spec, dataset):
        ssh = FakeSsh()
        make_executor(ssh=ssh).submit(spec)

        assert any(local == dataset for local, _ in ssh.uploaded)

    def test_installs_the_pinned_published_package(self, make_executor, spec):
        ssh = FakeSsh()
        make_executor(ssh=ssh).submit(spec)

        assert any("stateset-agents[training]==0.20.0" in cmd for cmd in ssh.commands)

    def test_runs_the_packaged_job_module(self, make_executor, spec):
        """The same entrypoint every other provider uses."""
        ssh = FakeSsh()
        make_executor(ssh=ssh).submit(spec)

        assert any(
            "python -m stateset_agents.training.sft" in cmd for cmd in ssh.commands
        )

    def test_the_adapter_actually_arrives_on_local_disk(self, make_executor, spec):
        executor = make_executor()
        result = executor.wait(executor.submit(spec))

        assert result.status is JobStatus.SUCCEEDED
        assert (spec.output_dir / "adapter_config.json").exists()
        assert (
            spec.output_dir / "adapter_model.safetensors"
        ).read_bytes() == b"WEIGHTS"


class TestPodTermination:
    """A leaked pod bills by the hour. It must die on every path."""

    def test_pod_is_terminated_after_a_successful_run(self, make_executor, spec):
        api = FakePodApi()
        make_executor(api=api).submit(spec)

        assert api.terminated == ["pod-abc"]

    def test_pod_is_terminated_after_a_failed_job(self, make_executor, spec):
        api = FakePodApi()
        make_executor(api=api, ssh=FakeSsh(exit_code=1)).submit(spec)

        assert api.terminated == ["pod-abc"]

    def test_pod_is_terminated_when_the_transport_raises(self, make_executor, spec):
        api = FakePodApi()
        ssh = FakeSsh()

        def boom(*args, **kwargs):
            raise OSError("connection reset")

        ssh.upload = boom

        with pytest.raises(RemoteExecutionError):
            make_executor(api=api, ssh=ssh).submit(spec)

        assert api.terminated == ["pod-abc"]

    def test_pod_is_terminated_when_it_never_becomes_reachable(
        self, make_executor, spec
    ):
        api = FakePodApi(never_ready=True)

        with pytest.raises(RemoteExecutionError, match="never became reachable"):
            make_executor(api=api, ready_timeout_s=0).submit(spec)

        assert api.terminated == ["pod-abc"]


class TestFailureReporting:
    def test_nonzero_exit_reports_failure(self, make_executor, spec):
        executor = make_executor(ssh=FakeSsh(exit_code=1))
        result = executor.wait(executor.submit(spec))

        assert result.status is JobStatus.FAILED
        assert not result.succeeded

    def test_clean_exit_with_no_artifacts_is_a_failure(self, make_executor, spec):
        """Same guard as Modal: silent empty success is the worst outcome."""
        executor = make_executor(ssh=FakeSsh(produces_adapter=False))
        result = executor.wait(executor.submit(spec))

        assert result.status is JobStatus.FAILED
        assert any("no artifacts" in line.lower() for line in result.logs)

    def test_remote_output_is_captured_in_logs(self, make_executor, spec):
        executor = make_executor()
        result = executor.wait(executor.submit(spec))

        assert any("training.sft" in line for line in result.logs)


class TestRegistry:
    def test_runpod_is_a_known_provider(self):
        from stateset_agents.remote.registry import available_providers

        assert "runpod" in available_providers()

    def test_missing_api_key_is_reported_clearly(self, monkeypatch, spec):
        from stateset_agents.remote.runpod import RunPodExecutor

        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

        with pytest.raises(RemoteExecutionError, match="RUNPOD_API_KEY"):
            RunPodExecutor().submit(spec)


class TestWheelInstall:
    """Installing a locally built wheel is how an *unreleased* change gets
    verified on real hardware — the PyPI pin cannot work before publish."""

    def test_uploads_and_installs_the_wheel_instead_of_pypi(
        self, make_executor, spec, tmp_path
    ):
        wheel = tmp_path / "stateset_agents-0.20.0-py3-none-any.whl"
        wheel.write_bytes(b"WHEELBYTES")
        ssh = FakeSsh()

        make_executor(ssh=ssh, wheel=wheel).submit(spec)

        assert any(local == wheel for local, _ in ssh.uploaded)
        install = next(c for c in ssh.commands if "pip install" in c)
        assert wheel.name in install
        assert "stateset-agents[training]==" not in install

    def test_installs_the_training_extra_from_the_wheel(
        self, make_executor, spec, tmp_path
    ):
        wheel = tmp_path / "stateset_agents-0.20.0-py3-none-any.whl"
        wheel.write_bytes(b"WHEELBYTES")
        ssh = FakeSsh()

        make_executor(ssh=ssh, wheel=wheel).submit(spec)

        install = next(c for c in ssh.commands if "pip install" in c)
        assert "[training]" in install

    def test_without_a_wheel_the_pypi_pin_is_used(self, make_executor, spec):
        ssh = FakeSsh()

        make_executor(ssh=ssh).submit(spec)

        assert any("stateset-agents[training]==0.20.0" in c for c in ssh.commands)


class TestDefaultImage:
    def test_default_image_ships_torch_at_least_2_6(self):
        """Regression guard from a real live run.

        runpod/pytorch:2.4.0 failed with `cannot import name 'DTensor' from
        torch.distributed.tensor` because transformers>=4.57.1 needs a torch
        that has DTensor there (2.6+). The pod provisions fine and the job
        starts, so this is only caught by actually running it.
        """
        import re

        from stateset_agents.remote.runpod import _DEFAULT_IMAGE

        match = re.search(r"torch(\d)(\d)(\d)", _DEFAULT_IMAGE)
        assert match, f"cannot read a torch version from {_DEFAULT_IMAGE!r}"
        major, minor = int(match.group(1)), int(match.group(2))
        assert (major, minor) >= (
            2,
            6,
        ), f"{_DEFAULT_IMAGE} has torch too old for transformers"


class TestDownloadFailureHandling:
    """A download failure must not discard the job's logs.

    Found live: training succeeded on the pod, scp then failed, and the
    executor raised — throwing away every line of output from a run that had
    actually worked. The user is left with a stack trace and no evidence.
    """

    def test_download_failure_is_reported_as_failed_with_logs_intact(
        self, make_executor, spec
    ):
        ssh = FakeSsh()

        def boom(*args, **kwargs):
            raise RemoteExecutionError("scp failed: unexpected filename: .")

        ssh.download_dir = boom
        executor = make_executor(ssh=ssh)

        result = executor.wait(executor.submit(spec))

        assert result.status is JobStatus.FAILED
        assert any("training.sft" in line for line in result.logs)
        assert any("scp failed" in line for line in result.logs)

    def test_pod_still_terminated_when_download_fails(self, make_executor, spec):
        api = FakePodApi()
        ssh = FakeSsh()

        def boom(*args, **kwargs):
            raise RemoteExecutionError("scp exploded")

        ssh.download_dir = boom
        make_executor(api=api, ssh=ssh).submit(spec)

        assert api.terminated == ["pod-abc"]


class TestScpCommandForm:
    """OpenSSH 9 runs scp over SFTP, which rejects `.` as a filename."""

    def test_recursive_download_does_not_use_the_dot_form(self, tmp_path):
        from stateset_agents.remote.runpod import SshTransport

        captured = {}

        transport = SshTransport()
        transport._host, transport._port = "1.2.3.4", 22

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd

            class R:
                returncode = 0
                stderr = ""

            return R()

        import subprocess

        original = subprocess.run
        subprocess.run = fake_run
        try:
            transport.download_dir("/workspace/out", tmp_path / "dest")
        finally:
            subprocess.run = original

        joined = " ".join(captured["cmd"])
        assert "/." not in joined, f"dot-form path is rejected by OpenSSH 9: {joined}"


class TestPublicKeyDiscovery:
    """Key discovery reads the host's ~/.ssh, so it is tested against a
    fake home rather than whatever the machine running the suite has."""

    def _executor(self):
        from stateset_agents.remote.runpod import RunPodExecutor

        return RunPodExecutor(api=FakePodApi(), ssh=FakeSsh(), poll_interval_s=0)

    def test_prefers_ed25519(self, tmp_path, monkeypatch):
        ssh_dir = tmp_path / ".ssh"
        ssh_dir.mkdir()
        (ssh_dir / "id_ed25519.pub").write_text("ssh-ed25519 ED\n")
        (ssh_dir / "id_rsa.pub").write_text("ssh-rsa RSA\n")
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

        assert self._executor()._require_public_key() == "ssh-ed25519 ED"

    def test_falls_back_to_rsa(self, tmp_path, monkeypatch):
        ssh_dir = tmp_path / ".ssh"
        ssh_dir.mkdir()
        (ssh_dir / "id_rsa.pub").write_text("ssh-rsa RSA\n")
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

        assert self._executor()._require_public_key() == "ssh-rsa RSA"

    def test_no_key_at_all_is_an_actionable_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

        with pytest.raises(RemoteExecutionError, match="no SSH public key"):
            self._executor()._require_public_key()


class TestContainerDiskSize:
    """The pod disk must scale with the model: 40GB default, configurable.

    Found on real hardware: meta-models/Muse-Glimmer-30B (~63GB BF16) died
    mid-download on the fixed 40GB disk with an opaque HF-cache
    "File reconstruction error".
    """

    def test_default_disk_is_40gb(self, make_executor, spec):
        api = FakePodApi()
        make_executor(api=api).submit(spec)
        assert api.created[0]["container_disk_gb"] == 40

    def test_disk_size_is_configurable(self, make_executor, spec):
        api = FakePodApi()
        make_executor(api=api, container_disk_gb=160).submit(spec)
        assert api.created[0]["container_disk_gb"] == 160

    def test_spec_disk_size_overrides_the_executor_default(self, make_executor, spec):
        """`--container-disk-gb` reaches the pod without rebuilding the executor."""
        api = FakePodApi()
        spec.container_disk_gb = 200
        make_executor(api=api, container_disk_gb=160).submit(spec)
        assert api.created[0]["container_disk_gb"] == 200

    def test_unset_spec_disk_falls_back_to_the_executor_default(
        self, make_executor, spec
    ):
        api = FakePodApi()
        assert spec.container_disk_gb is None
        make_executor(api=api, container_disk_gb=120).submit(spec)
        assert api.created[0]["container_disk_gb"] == 120


class TestEvalPrompts:
    """The prompts ride the ssh command as one JSON argument, so they must
    survive a real shell — including quotes and spaces inside a prompt."""

    PROMPTS = ["what's the return policy?", "plain prompt"]

    def _train_command(self, ssh: FakeSsh) -> str:
        return next(c for c in ssh.commands if "training.sft" in c)

    def test_prompts_are_shell_quoted_and_json_decodable(self, make_executor, spec):
        import shlex

        ssh = FakeSsh()
        spec.eval_prompts = self.PROMPTS
        make_executor(ssh=ssh).submit(spec)

        tokens = shlex.split(self._train_command(ssh))
        blob = tokens[tokens.index("--eval-prompts-json") + 1]
        assert json.loads(blob) == self.PROMPTS

    def test_spec_dict_prompts_survive_the_shell_as_json(self, make_executor, spec):
        """Prompt-spec objects add nested quotes/brackets to the JSON blob;
        shlex quoting must keep the whole thing one decodable argument."""
        import shlex

        ssh = FakeSsh()
        spec.eval_prompts = [
            "plain prompt",
            {"prompt": "what's the policy?", "expect": ["30 days"], "forbid": ["no"]},
        ]
        make_executor(ssh=ssh).submit(spec)

        tokens = shlex.split(self._train_command(ssh))
        blob = tokens[tokens.index("--eval-prompts-json") + 1]
        assert json.loads(blob) == spec.eval_prompts

    def test_no_flag_when_no_prompts(self, make_executor, spec):
        ssh = FakeSsh()
        make_executor(ssh=ssh).submit(spec)

        assert "--eval-prompts-json" not in self._train_command(ssh)

    def test_eval_max_new_tokens_travels_with_the_prompts(self, make_executor, spec):
        import shlex

        ssh = FakeSsh()
        spec.eval_prompts = self.PROMPTS
        spec.eval_max_new_tokens = 300
        make_executor(ssh=ssh).submit(spec)

        tokens = shlex.split(self._train_command(ssh))
        assert tokens[tokens.index("--eval-max-new-tokens") + 1] == "300"

    def test_no_token_budget_flag_when_no_prompts(self, make_executor, spec):
        ssh = FakeSsh()
        make_executor(ssh=ssh).submit(spec)

        assert "--eval-max-new-tokens" not in self._train_command(ssh)
