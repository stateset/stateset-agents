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

    def __init__(
        self,
        *,
        ready_after: int = 2,
        never_ready: bool = False,
        cost_per_hr: float | None = None,
    ):
        self.created: list[dict] = []
        self.terminated: list[str] = []
        self.polls = 0
        self.ready_after = ready_after
        self.never_ready = never_ready
        self.cost_per_hr = cost_per_hr

    def create_pod(self, **kwargs):
        self.created.append(kwargs)
        # First pod keeps the historical id; retries get distinct ids so a
        # test can tell WHICH pod was terminated.
        pod_id = "pod-abc" if len(self.created) == 1 else f"pod-abc{len(self.created)}"
        return {
            "id": pod_id,
            "desiredStatus": "RUNNING",
            "costPerHr": self.cost_per_hr,
        }

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

    def get_network_volume(self, volume_id):
        self.volume_lookups = getattr(self, "volume_lookups", [])
        self.volume_lookups.append(volume_id)
        return {
            "id": volume_id,
            "name": "test-vol",
            "size": 20,
            "dataCenterId": "US-KS-2",
        }


class FakeSsh:
    """Moves real bytes between local paths, standing in for scp/ssh."""

    def __init__(
        self,
        *,
        exit_code: int = 0,
        produces_adapter: bool = True,
        run_failures: int = 0,
    ):
        self.exit_code = exit_code
        self.produces_adapter = produces_adapter
        #: Raise on the first N run() calls — models the pod dying under the
        #: job (connection reset), the failure mode COMMUNITY pods hit.
        self.run_failures_left = run_failures
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
        if self.run_failures_left > 0:
            self.run_failures_left -= 1
            raise OSError("connection reset by peer")
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
def make_executor(tmp_path):
    from stateset_agents.remote.runpod import RunPodExecutor

    def build(api=None, ssh=None, **kwargs):
        # An explicit key by default: without it the executor reads the host's
        # ~/.ssh, so the suite would pass or fail on whether the machine
        # running it happens to have a keypair. (It does locally; the Windows
        # CI runner does not, which is how this was found.)
        kwargs.setdefault("public_key", "ssh-rsa AAAATESTKEY")
        kwargs.setdefault("lease_dir", tmp_path / "runpod-leases")
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


class TestCrashRecoveryLeases:
    def test_pod_has_a_lease_until_termination_is_confirmed(
        self, make_executor, spec, tmp_path
    ):
        api = FakePodApi()
        executor = make_executor(api=api, lease_dir=tmp_path / "leases")
        observed = []
        original_terminate = api.terminate_pod

        def observe_then_terminate(pod_id):
            observed.append(executor._lease_path(pod_id).exists())
            original_terminate(pod_id)

        api.terminate_pod = observe_then_terminate

        executor.submit(spec)

        assert observed == [True]
        assert executor.orphaned_leases() == []

    def test_later_process_can_terminate_a_crash_lease(self, spec, tmp_path):
        from stateset_agents.remote.runpod import RunPodExecutor

        lease_dir = tmp_path / "leases"
        first = RunPodExecutor(
            api=FakePodApi(),
            ssh=FakeSsh(),
            public_key="ssh-rsa AAAATESTKEY",
            lease_dir=lease_dir,
        )
        first._write_lease("pod-orphan", "job-1", spec, 123.0)

        api = FakePodApi()
        restarted = RunPodExecutor(
            api=api,
            ssh=FakeSsh(),
            public_key="ssh-rsa AAAATESTKEY",
            lease_dir=lease_dir,
        )

        assert restarted.cleanup_orphans() == ["pod-orphan"]
        assert api.terminated == ["pod-orphan"]
        assert restarted.orphaned_leases() == []


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

        # A persistent transport failure is retried once on a fresh pod
        # (default max_provision_attempts=2); BOTH pods must die.
        assert api.terminated == ["pod-abc", "pod-abc2"]

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


class TestGpuCount:
    """Multi-GPU pods: the spec's gpu_count must reach create_pod, and the
    REST wrapper must map it onto RunPod's `gpuCount` key."""

    def test_default_is_one_gpu(self, make_executor, spec):
        api = FakePodApi()
        make_executor(api=api).submit(spec)
        assert api.created[0]["gpu_count"] == 1

    def test_gpu_count_reaches_create_pod(self, make_executor, spec):
        api = FakePodApi()
        spec.gpu_count = 2
        make_executor(api=api).submit(spec)
        assert api.created[0]["gpu_count"] == 2

    def test_runpod_whole_pod_price_is_not_multiplied_in_budget(
        self, make_executor, spec
    ):
        api = FakePodApi(cost_per_hr=13.16)
        spec.gpu_count = 4
        spec.timeout_s = 3600
        spec.max_cost_usd = 14.0
        executor = make_executor(api=api)

        handle = executor.submit(spec)

        # $13.16/hr for one hour is below the $14 ceiling. The old code
        # multiplied this whole-Pod price by four and rejected the run.
        assert executor.status(handle) is JobStatus.SUCCEEDED

    def test_real_api_sends_gpucount_in_the_payload(self, monkeypatch):
        from stateset_agents.remote.runpod import RunPodApi

        captured = {}

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"id": "p"}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["json"] = json
            return FakeResponse()

        import requests

        monkeypatch.setattr(requests, "post", fake_post)
        RunPodApi("key").create_pod(
            name="n",
            image="img",
            gpu_type_id="g",
            gpu_count=2,
            ports=["22/tcp"],
            env={},
        )
        assert captured["json"]["gpuCount"] == 2
        assert captured["json"]["gpuTypeIds"] == ["g"]
        assert captured["json"]["computeType"] == "GPU"
        assert captured["json"]["gpuTypePriority"] == "availability"

    def test_real_api_defaults_gpucount_to_one(self, monkeypatch):
        from stateset_agents.remote.runpod import RunPodApi

        captured = {}

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"id": "p"}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["json"] = json
            return FakeResponse()

        import requests

        monkeypatch.setattr(requests, "post", fake_post)
        RunPodApi("key").create_pod(
            name="n", image="img", gpu_type_id="g", ports=["22/tcp"], env={}
        )
        assert captured["json"]["gpuCount"] == 1

    def test_real_api_sends_direct_image_command_fields(self, monkeypatch):
        from stateset_agents.remote.runpod import RunPodApi

        captured = {}

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"id": "p"}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["json"] = json
            return FakeResponse()

        import requests

        monkeypatch.setattr(requests, "post", fake_post)
        RunPodApi("key").create_pod(
            name="n",
            image="img",
            gpu_type_id="g",
            ports=["8000/http"],
            env={},
            docker_entrypoint=["/bin/bash", "-lc"],
            docker_start_cmd=["exec vllm serve m"],
        )
        assert captured["json"]["dockerEntrypoint"] == ["/bin/bash", "-lc"]
        assert captured["json"]["dockerStartCmd"] == ["exec vllm serve m"]

    def test_create_failure_is_sanitized_as_remote_error(self, monkeypatch):
        import requests

        from stateset_agents.remote.runpod import RunPodApi

        response = requests.Response()
        response.status_code = 500

        def fail_send(request):
            raise requests.HTTPError("provider details", response=response)

        monkeypatch.setattr(RunPodApi, "_send", staticmethod(fail_send))

        with pytest.raises(RemoteExecutionError, match="failed.*HTTP 500") as caught:
            RunPodApi("secret-key").create_pod(
                name="n", image="img", gpu_type_id="g", ports=[], env={}
            )

        assert "secret-key" not in str(caught.value)


class TestCloudType:
    """COMMUNITY (~spot) pods are far cheaper; the spec's choice must reach
    the create-pod call — and an invalid value must fail before renting."""

    def test_default_is_secure(self, make_executor, spec):
        api = FakePodApi()
        make_executor(api=api).submit(spec)
        assert api.created[0]["cloud_type"] == "SECURE"

    def test_community_reaches_create_pod(self, make_executor, spec):
        api = FakePodApi()
        spec.cloud_type = "COMMUNITY"
        make_executor(api=api).submit(spec)
        assert api.created[0]["cloud_type"] == "COMMUNITY"

    def test_real_api_sends_cloudtype_in_the_payload(self, monkeypatch):
        """The wrapper must map cloud_type onto RunPod's `cloudType` key."""
        from stateset_agents.remote.runpod import RunPodApi

        captured = {}

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"id": "p"}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["json"] = json
            return FakeResponse()

        import requests

        monkeypatch.setattr(requests, "post", fake_post)
        RunPodApi("key").create_pod(
            name="n",
            image="img",
            gpu_type_id="g",
            ports=["22/tcp"],
            env={},
            cloud_type="COMMUNITY",
        )
        assert captured["json"]["cloudType"] == "COMMUNITY"

    def test_real_api_sends_volume_fields_in_the_payload(self, monkeypatch):
        """Live-verified REST field names: networkVolumeId, volumeMountPath,
        dataCenterIds (a list — volumes are datacenter-scoped)."""
        from stateset_agents.remote.runpod import RunPodApi

        captured = {}

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"id": "p"}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["json"] = json
            return FakeResponse()

        import requests

        monkeypatch.setattr(requests, "post", fake_post)
        RunPodApi("key").create_pod(
            name="n",
            image="img",
            gpu_type_id="g",
            ports=["22/tcp"],
            env={},
            network_volume_id="vol-1",
            volume_mount_path="/workspace",
            data_center_id="US-KS-2",
        )
        assert captured["json"]["networkVolumeId"] == "vol-1"
        assert captured["json"]["volumeMountPath"] == "/workspace"
        assert captured["json"]["dataCenterIds"] == ["US-KS-2"]

    def test_real_api_omits_volume_fields_when_unset(self, monkeypatch):
        from stateset_agents.remote.runpod import RunPodApi

        captured = {}

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"id": "p"}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["json"] = json
            return FakeResponse()

        import requests

        monkeypatch.setattr(requests, "post", fake_post)
        RunPodApi("key").create_pod(
            name="n", image="img", gpu_type_id="g", ports=["22/tcp"], env={}
        )
        for key in ("networkVolumeId", "volumeMountPath", "dataCenterIds"):
            assert key not in captured["json"]

    def test_list_network_volumes_accepts_bare_list_and_envelope(self, monkeypatch):
        from stateset_agents.remote.runpod import RunPodApi

        payloads = iter(
            [
                [{"id": "v1", "dataCenterId": "US-KS-2"}],
                {"networkVolumes": [{"id": "v2", "dataCenterId": "EU-RO-1"}]},
            ]
        )

        class FakeResponse:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                pass

            def json(self):
                return self._payload

        import requests

        monkeypatch.setattr(
            requests, "get", lambda *a, **k: FakeResponse(next(payloads))
        )
        api = RunPodApi("key")
        assert [v["id"] for v in api.list_network_volumes()] == ["v1"]
        assert [v["id"] for v in api.list_network_volumes()] == ["v2"]


class TestRetryOnPodDeath:
    """A pod dying under a running job (the COMMUNITY failure mode, also
    observed live on SECURE) must cost a retry, not the whole run."""

    def test_provisions_a_second_pod_exactly_once_and_terminates_the_first(
        self, make_executor, spec
    ):
        api = FakePodApi()
        ssh = FakeSsh(run_failures=1)
        executor = make_executor(api=api, ssh=ssh)

        result = executor.wait(executor.submit(spec))

        assert len(api.created) == 2
        assert api.terminated == ["pod-abc", "pod-abc2"]
        assert result.status is JobStatus.SUCCEEDED

    def test_retry_is_reported_in_the_logs(self, make_executor, spec):
        executor = make_executor(ssh=FakeSsh(run_failures=1))
        result = executor.wait(executor.submit(spec))

        assert any("restarting training from scratch" in line for line in result.logs)

    def test_persistent_death_gives_up_after_max_attempts(self, make_executor, spec):
        api = FakePodApi()
        ssh = FakeSsh(run_failures=99)

        with pytest.raises(RemoteExecutionError, match="giving up"):
            make_executor(api=api, ssh=ssh, max_provision_attempts=3).submit(spec)

        assert len(api.created) == 3
        assert len(api.terminated) == 3

    def test_ssh_exit_255_is_treated_as_pod_death_and_retried(
        self, make_executor, spec
    ):
        """255 is ssh's own exit code — keepalive-detected death lands there,
        not as an exception."""
        api = FakePodApi()
        ssh = FakeSsh()
        real_run = ssh.run
        calls = {"n": 0}

        def run_255_once(command):
            calls["n"] += 1
            if calls["n"] == 1:
                return 255, "client_loop: send disconnect: Broken pipe"
            return real_run(command)

        ssh.run = run_255_once
        executor = make_executor(api=api, ssh=ssh)

        result = executor.wait(executor.submit(spec))

        assert len(api.created) == 2
        assert result.status is JobStatus.SUCCEEDED

    def test_a_training_failure_is_not_retried(self, make_executor, spec):
        """The job's own non-zero exit is a code/data problem — rerunning it
        on a fresh pod would just bill twice for the same failure."""
        api = FakePodApi()
        executor = make_executor(api=api, ssh=FakeSsh(exit_code=1))

        result = executor.wait(executor.submit(spec))

        assert result.status is JobStatus.FAILED
        assert len(api.created) == 1

    def test_never_reachable_is_not_retried(self, make_executor, spec):
        api = FakePodApi(never_ready=True)

        with pytest.raises(RemoteExecutionError, match="never became reachable"):
            make_executor(api=api, ready_timeout_s=0).submit(spec)

        assert len(api.created) == 1


class TestResumeFlag:
    def test_resume_travels_to_the_remote_command(self, make_executor, spec):
        ssh = FakeSsh()
        spec.resume = True
        make_executor(ssh=ssh).submit(spec)

        train = next(c for c in ssh.commands if "training.sft" in c)
        assert "--resume" in train

    def test_no_resume_flag_by_default(self, make_executor, spec):
        ssh = FakeSsh()
        make_executor(ssh=ssh).submit(spec)

        train = next(c for c in ssh.commands if "training.sft" in c)
        assert "--resume" not in train


class TestNetworkVolume:
    """--network-volume-id mounts durable storage at /workspace, so retries
    resume from the surviving checkpoints instead of restarting."""

    def test_pod_payload_attaches_the_volume_pinned_to_its_datacenter(
        self, make_executor, spec
    ):
        api = FakePodApi()
        spec.network_volume_id = "vol-123"
        make_executor(api=api).submit(spec)

        assert api.volume_lookups == ["vol-123"]
        payload = api.created[0]
        assert payload["network_volume_id"] == "vol-123"
        assert payload["volume_mount_path"] == "/workspace"
        assert payload["data_center_id"] == "US-KS-2"

    def test_pod_payload_omits_volume_fields_when_unset(self, make_executor, spec):
        api = FakePodApi()
        make_executor(api=api).submit(spec)

        payload = api.created[0]
        assert payload["network_volume_id"] is None
        assert payload["volume_mount_path"] is None
        assert payload["data_center_id"] is None

    def test_retry_with_volume_reruns_with_resume(self, make_executor, spec):
        spec.network_volume_id = "vol-123"
        ssh = FakeSsh()
        real_run = ssh.run
        calls = {"n": 0}

        def die_on_first_train(command):
            if "training.sft" in command:
                calls["n"] += 1
                if calls["n"] == 1:
                    ssh.commands.append(command)
                    raise OSError("connection reset by peer")
            return real_run(command)

        ssh.run = die_on_first_train
        executor = make_executor(ssh=ssh)
        result = executor.wait(executor.submit(spec))

        trains = [c for c in ssh.commands if "training.sft" in c]
        assert "--resume" not in trains[0]
        assert "--resume" in trains[-1]
        assert result.status is JobStatus.SUCCEEDED
        assert any("resuming from the newest checkpoint" in ln for ln in result.logs)

    def test_first_attempt_does_not_resume_unless_asked(self, make_executor, spec):
        spec.network_volume_id = "vol-123"
        ssh = FakeSsh()
        make_executor(ssh=ssh).submit(spec)

        train = next(c for c in ssh.commands if "training.sft" in c)
        assert "--resume" not in train

    def test_retry_without_volume_still_restarts_from_scratch(
        self, make_executor, spec
    ):
        ssh = FakeSsh(run_failures=1)
        executor = make_executor(ssh=ssh)
        result = executor.wait(executor.submit(spec))

        trains = [c for c in ssh.commands if "training.sft" in c]
        assert all("--resume" not in c for c in trains)
        assert any("restarting training from scratch" in ln for ln in result.logs)


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


class TestWheelEnvSeam:
    def test_env_var_supplies_the_wheel_for_cli_runs(self, monkeypatch, tmp_path):
        """Discovered live: the flywheel's first spin died with 'No module
        named stateset_agents.training.harvest' because the pod installed
        the PyPI release, which predated the module. STATESET_AGENTS_WHEEL
        lets CLI-constructed executors ship the local build instead."""
        from stateset_agents.remote.runpod import RunPodExecutor

        wheel = tmp_path / "stateset_agents-9.9.9-py3-none-any.whl"
        wheel.write_bytes(b"x")
        monkeypatch.setenv("STATESET_AGENTS_WHEEL", str(wheel))

        executor = RunPodExecutor(api=object(), ssh=object(), public_key="k")

        assert executor.wheel == wheel

    def test_explicit_wheel_argument_wins_over_the_env(self, monkeypatch, tmp_path):
        from stateset_agents.remote.runpod import RunPodExecutor

        monkeypatch.setenv("STATESET_AGENTS_WHEEL", str(tmp_path / "env.whl"))
        explicit = tmp_path / "explicit.whl"

        executor = RunPodExecutor(
            api=object(), ssh=object(), public_key="k", wheel=explicit
        )

        assert executor.wheel == explicit
