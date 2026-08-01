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

        assert any(
            "stateset-agents[training]==0.20.0" in cmd for cmd in ssh.commands
        )

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

        assert any(
            "stateset-agents[training]==0.20.0" in c for c in ssh.commands
        )


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
        assert (major, minor) >= (2, 6), f"{_DEFAULT_IMAGE} has torch too old for transformers"


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
