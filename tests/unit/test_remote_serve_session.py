"""Unit tests for :mod:`stateset_agents.remote.serve_session`.

Everything is faked at the RunPodApi/SshTransport/http_get seams — no
network, no ssh, no pods. What matters: the pod is provisioned with BOTH
ports, the self-destruct is armed before anything fallible, the endpoint
URL comes from the 8000 port mapping, readiness polls /v1/models with the
Bearer token, startup failures terminate the pod, and success does NOT.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.serve_recovery import recover_serving, serving_lease_dir
from stateset_agents.remote.serve_session import (
    RemoteServeSession,
    find_serve_pod,
    list_serve_pods,
    self_destruct_script,
)


@pytest.fixture(autouse=True)
def isolate_serving_state(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


class FakeApi:
    def __init__(self, pods=None):
        self.api_key = "rp-test-key"
        self.root = "https://rest.runpod.io/v1"
        self.created: list[dict] = []
        self.terminated: list[str] = []
        self.quotes: list[tuple[str, str]] = []
        self.rate = 1.0
        self._pods = pods or []
        self.volumes = {"vol-1": {"id": "vol-1", "dataCenterId": "EUR-IS-1"}}
        self.pod_state = {
            "id": "pod-1",
            "desiredStatus": "RUNNING",
            "publicIp": "1.2.3.4",
            "portMappings": {"22": 2222, "8000": 18000},
        }

    def create_pod(self, **kwargs):
        self.created.append(kwargs)
        return {"id": "pod-1"}

    def get_gpu_rate(self, gpu_type_id, cloud_type):
        self.quotes.append((gpu_type_id, cloud_type))
        return self.rate

    def get_pod(self, pod_id, *, deadline=None):
        return dict(self.pod_state)

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)

    def get_network_volume(self, volume_id):
        return dict(self.volumes[volume_id])

    def list_pods(self):
        return list(self._pods)


class FakeSsh:
    def __init__(self):
        self.commands: list[str] = []
        self.uploads: list[tuple[str, str]] = []
        self.secret_uploads: list[tuple[str, str]] = []
        self.fail_on: str | None = None
        #: Arm a detached step (by label) to report a non-zero exit code.
        self.fail_detached: str | None = None

    def wait_until_reachable(self, host, port, timeout_s):
        self.reachable = (host, port)

    def upload(self, local, remote):
        self.uploads.append((str(local), remote))

    def upload_secret(self, secret, remote):
        self.secret_uploads.append((secret, remote))

    def run(self, command):
        self.commands.append(command)
        if self.fail_on and self.fail_on in command:
            return 1, "boom"
        # Detached work writes its exit code to a marker file which the
        # session then polls; a fake that never answers the poll would spin
        # until the deadline. Report success unless a failure was armed.
        if command.startswith("cat /workspace/."):
            marker = command.split()[1]
            label = marker.rsplit("/", 1)[-1].lstrip(".").rsplit(".rc", 1)[0]
            if self.fail_detached and self.fail_detached in label:
                return 0, "1"
            return 0, "0"
        return 0, "ok"


def make_session(
    api=None,
    ssh=None,
    http_statuses=(200,),
    completions=None,
    log_response=(200, "startup log"),
    **kwargs,
):
    statuses = list(http_statuses)
    calls: list[tuple[str, dict]] = []
    post_calls: list[dict] = []
    #: model name -> greedy completion for the effect probe. Distinct by
    #: default so the probe passes quietly in unrelated tests.
    completions = completions or {}

    def http_get(url, headers):
        calls.append((url, dict(headers)))
        return statuses.pop(0) if len(statuses) > 1 else statuses[0]

    def http_post_json(url, headers, payload):
        post_calls.append(payload)
        model = payload["model"]
        text = completions.get(model, f"completion from {model}")
        return {"choices": [{"message": {"content": text}}]}

    def http_get_text(url, headers):
        calls.append((url, dict(headers)))
        return log_response

    session = RemoteServeSession(
        api or FakeApi(),
        ssh or FakeSsh(),
        public_key="ssh-ed25519 AAA test",
        poll_interval_s=0.0,
        http_get=http_get,
        http_get_text=http_get_text,
        http_post_json=http_post_json,
        **kwargs,
    )
    session._post_calls = post_calls  # test-side telescope
    session._http_calls = calls  # test-side telescope, not API
    return session


class TestStartHappyPath:
    def test_pod_is_created_with_both_ports_and_prefixed_name(self):
        api, ssh = FakeApi(), FakeSsh()
        session = make_session(api, ssh)

        session.start("Qwen/Qwen3.5-0.8B")

        created = api.created[0]
        # 22 needs a real TCP mapping for ssh; the model port is http so
        # RunPod serves it through its proxy, which needs no public IP.
        assert created["ports"] == ["22/tcp", "8000/http"]
        assert created["name"].startswith("stateset-serve-")
        assert created["gpu_type_id"] == RemoteServeSession.DEFAULT_GPU

    def test_endpoint_url_is_the_runpod_http_proxy(self):
        """Five verification attempts hung waiting for a TCP mapping on the
        model port that RunPod was never going to publish: http ports are
        reached through the proxy instead."""
        session = make_session()

        session.start("m")

        assert session.endpoint_url == "https://pod-1-8000.proxy.runpod.net"

    def test_readiness_polls_v1_models_with_the_bearer_token(self):
        session = make_session(token="tok-123")

        session.start("m")

        url, headers = session._http_calls[0]
        assert url == "https://pod-1-8000.proxy.runpod.net/v1/models"
        assert headers["Authorization"] == "Bearer tok-123"

    def test_success_does_not_terminate_the_pod(self):
        api = FakeApi()
        session = make_session(api)

        session.start("m")

        assert api.terminated == []
        assert session.pod_id == "pod-1"
        assert api.quotes == []

    def test_vllm_is_installed_and_launched_with_the_token(self):
        ssh = FakeSsh()
        session = make_session(ssh=ssh, token="tok-abc")

        session.start("Qwen/Qwen3.5-0.8B")

        assert any("pip install --quiet vllm" in c for c in ssh.commands)
        launch = next(c for c in ssh.commands if "vllm serve" in c)
        assert "nohup" in launch
        assert "Qwen/Qwen3.5-0.8B" in launch
        assert "--api-key tok-abc" in launch
        assert "--enable-lora" not in launch

    def test_multi_gpu_ssh_serve_sets_tensor_parallel_size(self):
        ssh = FakeSsh()
        session = make_session(ssh=ssh)

        session.start("model", gpu_count=4)

        launch = next(c for c in ssh.commands if "vllm serve" in c)
        assert "--tensor-parallel-size 4" in launch

    def test_cost_ceiling_also_protects_ssh_managed_serves(self):
        class PricedApi(FakeApi):
            def create_pod(self, **kwargs):
                self.created.append(kwargs)
                return {"id": "pod-1", "costPerHr": 20.0}

        api = PricedApi()
        session = make_session(api)

        with pytest.raises(RemoteExecutionError, match="above.*max-cost"):
            session.start("model", max_hours=0.5, max_cost_usd=5.0)

        assert api.terminated == ["pod-1"]


class TestServeRecovery:
    @pytest.mark.parametrize("crash_after_id", [False, True])
    def test_process_exit_releases_ownership_and_preserves_rental(
        self, tmp_path, crash_after_id
    ):
        provider_path = tmp_path / "provider.json"
        # Exit inside the real start() path without executing Python finally blocks.
        worker = """
import json, os, sys
from pathlib import Path
from stateset_agents.remote.serve_session import RemoteServeSession
class Api:
    def create_pod(self, **kwargs):
        Path(sys.argv[1]).write_text(json.dumps({"name": kwargs["name"], "id": "rented-1"}))
        if sys.argv[2] == "False":
            os._exit(23)
        return {"id": "rented-1", "costPerHr": 1}
    def get_pod(self, pod_id):
        os._exit(23)
RemoteServeSession(Api(), public_key="test", ssh=object()).start("model")
"""
        proc = subprocess.run(
            [sys.executable, "-c", worker, str(provider_path), str(crash_after_id)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 23, proc.stderr
        pod = json.loads(provider_path.read_text())
        root = serving_lease_dir()
        record = root / f"{pod['name']}.json"
        assert json.loads(record.read_text()) == {
            "name": pod["name"],
            "pod_id": "rented-1" if crash_after_id else None,
        }
        api = FakeApi(pods=[{"name": "stateset-serve-unrelated", "id": "other"}, pod])
        assert recover_serving(root, lambda: api)[0].status == "recoverable"
        assert api.terminated == []
        assert (
            recover_serving(root, lambda: api, terminate=True)[0].status
            == "termination accepted"
        )
        assert api.terminated == ["rented-1"]
        assert recover_serving(root, lambda: api, terminate=True) == []
        assert list(root.iterdir()) == []

    @pytest.mark.parametrize("direct", [False, True])
    def test_live_start_is_excluded_and_ready_endpoint_is_retired(self, direct):
        root = serving_lease_dir()

        class ActiveApi(FakeApi):
            def create_pod(self, **kwargs):
                assert (
                    json.loads((root / f"{kwargs['name']}.json").read_text())["pod_id"]
                    is None
                )
                assert (
                    recover_serving(root, lambda: self, terminate=True)[0].status
                    == "busy"
                )
                return super().create_pod(**kwargs)

            def get_pod(self, pod_id, **kwargs):
                assert (
                    json.loads(next(root.glob("*.json")).read_text())["pod_id"]
                    == pod_id
                )
                assert (
                    recover_serving(root, lambda: self, terminate=True)[0].status
                    == "busy"
                )
                return super().get_pod(pod_id, **kwargs)

        api = ActiveApi()
        make_session(api, direct_vllm_image=direct).start("m")
        assert list(root.iterdir()) == []
        assert recover_serving(root, lambda: api, terminate=True) == []
        assert api.terminated == []

    def test_unwritable_record_prevents_allocation(self, tmp_path):
        root = tmp_path / "not-a-directory"
        root.touch()
        api = FakeApi()
        with pytest.raises(
            RemoteExecutionError, match="Cannot lock RunPod allocation record"
        ):
            make_session(api, lease_dir=root).start("m")
        assert api.created == []

    def test_lost_allocation_reply_keeps_name_and_does_not_retry(self):
        class LostReply(FakeApi):
            def create_pod(self, **kwargs):
                super().create_pod(**kwargs)
                self._pods.append({"name": kwargs["name"], "id": "rented-1"})
                raise RemoteExecutionError("response lost")

        api = LostReply()
        with pytest.raises(RemoteExecutionError, match="response lost"):
            make_session(api).start("m")
        assert len(api.created) == 1
        assert (
            recover_serving(serving_lease_dir(), lambda: api, terminate=True)[0].detail
            == "rented-1"
        )
        assert api.terminated == ["rented-1"]

    def test_failure_to_record_returned_id_requests_cleanup(self, monkeypatch):
        from stateset_agents.remote import runpod_recovery

        write = runpod_recovery.write_json

        def fail_id_update(path, row):
            if row["pod_id"] is not None:
                raise OSError("disk full")
            write(path, row)

        monkeypatch.setattr(runpod_recovery, "write_json", fail_id_update)
        api = FakeApi()
        with pytest.raises(
            RemoteExecutionError, match="Cannot record RunPod allocation"
        ):
            make_session(api).start("m")
        assert api.terminated == ["pod-1"]
        assert list(serving_lease_dir().iterdir()) == []


class TestServeBudget:
    @pytest.mark.parametrize("direct", [False, True])
    def test_missing_catalog_quote_refuses_rental(self, direct, monkeypatch):
        api = FakeApi()

        def missing_price(gpu, cloud):
            raise RemoteExecutionError("No usable RunPod catalog price")

        monkeypatch.setattr(api, "get_gpu_rate", missing_price)
        session = make_session(api, direct_vllm_image=direct)
        with pytest.raises(RemoteExecutionError, match="catalog price"):
            session.start("m", max_cost_usd=3)
        assert api.created == []

    @pytest.mark.parametrize(
        "pod", [{}, None, {"id": None}, {"id": ""}, {"id": "bad/id"}]
    )
    def test_ambiguous_allocation_reports_searchable_name_without_retry(self, pod):
        class MissingId(FakeApi):
            def create_pod(self, **kwargs):
                super().create_pod(**kwargs)
                return pod

        api = MissingId()
        session = make_session(api)
        with pytest.raises(RemoteExecutionError, match="no pod ID") as error:
            session.start("m", max_cost_usd=2)
        assert api.created[0]["name"] in str(error.value)
        assert len(api.created) == 1
        assert api.terminated == []

    @pytest.mark.parametrize("direct", [False, True])
    def test_catalog_prices_each_requested_gpu_before_renting(self, direct):
        api = FakeApi()
        api.rate = 3.0
        session = make_session(api, direct_vllm_image=direct)

        with pytest.raises(RemoteExecutionError, match="catalog.*above.*max-cost"):
            session.start(
                "m", gpu="chosen", gpu_count=2, max_hours=0.5, max_cost_usd=2.99
            )

        assert api.quotes == [("chosen", "SECURE")]
        assert api.created == []

    @pytest.mark.parametrize("direct", [False, True])
    def test_rechecks_whole_pod_quote_and_uses_allocation_deadline(
        self, direct, monkeypatch
    ):
        from stateset_agents.remote import serve_session

        class CaptureSsh(FakeSsh):
            def upload(self, local, remote):
                if remote.endswith("self_destruct.sh"):
                    self.watchdog = Path(local).read_text()
                super().upload(local, remote)

        clock = [1000]
        monkeypatch.setattr(serve_session.time, "time", lambda: clock[0])

        class SlowApi(FakeApi):
            def create_pod(self, **kwargs):
                clock[0] += 120
                return super().create_pod(**kwargs)

        api, ssh = SlowApi(), CaptureSsh()
        api.rate = 3.0
        api.pod_state["costPerHr"] = "6.00"
        session = make_session(api, ssh, direct_vllm_image=direct)
        session.start("m", gpu="chosen", gpu_count=2, max_hours=0.5, max_cost_usd=3)

        assert api.quotes == [("chosen", api.created[0]["cloud_type"])]
        assert session.pod_id == "pod-1"
        if direct:
            assert "2800-time.time()" in api.created[0]["docker_start_cmd"][0]
        else:
            assert "2800 - $(date +%s)" in ssh.watchdog

    @pytest.mark.parametrize("direct", [False, True])
    @pytest.mark.parametrize(
        "rate", [None, False, True, "NaN", float("inf"), -1, 0, "bad"]
    )
    def test_unpriceable_allocated_pod_is_terminated(self, direct, rate):
        api = FakeApi()
        api.pod_state["costPerHr"] = rate
        session = make_session(api, direct_vllm_image=direct)
        with pytest.raises(RemoteExecutionError, match="price for this pod"):
            session.start("m", max_cost_usd=5)

        assert api.terminated == ["pod-1"]
        assert session._http_calls == []

    @pytest.mark.parametrize("direct", [False, True])
    @pytest.mark.parametrize("capped", [False, True])
    def test_status_read_failure_always_cleans_up(self, direct, capped):
        class BrokenApi(FakeApi):
            def get_pod(self, pod_id, *, deadline=None):
                raise RuntimeError("provider unavailable")

        api = BrokenApi()
        session = make_session(api, direct_vllm_image=direct)
        with pytest.raises(RuntimeError, match="provider unavailable"):
            session.start("m", max_cost_usd=5 if capped else None)

        assert api.terminated == ["pod-1"]
        assert len(api.created) == 1

    def test_network_retry_deducts_elapsed_cost_through_termination(self, monkeypatch):
        from stateset_agents.remote import serve_session

        clock = [10.0]
        monkeypatch.setattr(serve_session.time, "monotonic", lambda: clock[0])

        class SlowDelete(FakeApi):
            def terminate_pod(self, pod_id):
                clock[0] += 0.01
                super().terminate_pod(pod_id)

        api = SlowDelete()
        api.pod_state.update(publicIp="", costPerHr=1)
        session = make_session(api, network_timeout_s=0)
        # Even a charge too small to display in the ledger consumes budget.
        with pytest.raises(RemoteExecutionError, match="catalog.*max-cost"):
            session.start("m", max_cost_usd=1)
        assert len(api.quotes) == 2
        assert len(api.created) == 1
        assert api.terminated == ["pod-1"]

    def test_affordable_replacement_pod_is_allowed(self):
        class NetworkOnRetry(FakeApi):
            def get_pod(self, pod_id, *, deadline=None):
                return {
                    **super().get_pod(pod_id, deadline=deadline),
                    "publicIp": "1.2.3.4" if len(self.created) == 2 else "",
                    "costPerHr": 1,
                }

        api = NetworkOnRetry()
        session = make_session(api, network_timeout_s=0)
        session.start("m", max_cost_usd=2)
        assert len(api.quotes) == len(api.created) == 2
        assert len(api.terminated) == 1
        assert session.endpoint_url

    def test_unconfirmed_termination_retains_id_and_prevents_new_rental(
        self, monkeypatch
    ):
        api = FakeApi()
        api.pod_state.update(publicIp="", costPerHr=1)
        session = make_session(api, network_timeout_s=0)
        with monkeypatch.context() as patch:

            def failed_delete(pod_id):
                raise OSError("provider unavailable")

            patch.setattr(api, "terminate_pod", failed_delete)
            with pytest.raises(RemoteExecutionError, match="--stop pod-1"):
                session.start("m", max_cost_usd=2)
            assert session.pod_id == "pod-1"
            with pytest.raises(RemoteExecutionError, match="still owns pod pod-1"):
                session.start("m", max_cost_usd=2)

        session.terminate()
        assert session.pod_id is None
        assert len(api.created) == 1
        assert api.terminated == ["pod-1"]

    @pytest.mark.parametrize("direct", [False, True])
    def test_allocation_exhausting_lifetime_is_cleaned_up(self, direct, monkeypatch):
        from stateset_agents.remote import serve_session

        clock = [0.0]
        monkeypatch.setattr(serve_session.time, "monotonic", lambda: clock[0])

        class SlowCreate(FakeApi):
            def create_pod(self, **kwargs):
                clock[0] += 3
                return {**super().create_pod(**kwargs), "costPerHr": 1}

        api = SlowCreate()
        session = make_session(api, direct_vllm_image=direct)
        with pytest.raises(RemoteExecutionError, match="budgeted timeout"):
            session.start("m", max_hours=2 / 3600, max_cost_usd=1)

        assert api.terminated == ["pod-1"]
        assert session._http_calls == []


class TestDirectVllmImage:
    def test_embedded_log_server_is_valid_python(self):
        session = make_session(direct_vllm_image=True, image="image")

        compile(session._direct_log_server_source(), "<log-server>", "exec")

    def test_multi_gpu_prebuilt_image_skips_ssh_and_arms_watchdog(self):
        api, ssh = FakeApi(), FakeSsh()
        session = make_session(
            api,
            ssh,
            direct_vllm_image=True,
            image="vllm/vllm-openai:qwen38-flash-next",
            vllm_args=["--tensor-parallel-size", "4"],
            token="tok-direct",
        )

        session.start("Qwen/Qwen3.8-Flash-Next-FP8", gpu_count=4, max_hours=0.5)

        created = api.created[0]
        assert created["gpu_count"] == 4
        assert created["ports"] == ["8000/http", "8001/http"]
        assert created["docker_entrypoint"] == ["/bin/bash", "-lc"]
        command = created["docker_start_cmd"][0]
        assert "Qwen/Qwen3.8-Flash-Next-FP8" in command
        assert "--tensor-parallel-size 4" in command
        assert "RUNPOD_POD_ID" in command
        assert "RUNPOD_API_KEY" in command
        assert "rp-test-key" not in command
        assert command.count("python3 -c") == 2
        assert "stateset-vllm.log" in command
        assert "Authorization" in command
        assert ssh.commands == []
        assert api.terminated == []

    def test_existing_volume_pins_datacenter_and_hosts_hf_cache(self):
        api = FakeApi()
        session = make_session(api, direct_vllm_image=True, image="image")

        session.start("model", network_volume_id="vol-1")

        created = api.created[0]
        assert created["network_volume_id"] == "vol-1"
        assert created["volume_mount_path"] == "/workspace"
        assert created["data_center_id"] == "EUR-IS-1"
        assert created["env"]["HF_HOME"] == "/workspace/huggingface"
        assert created["env"]["HF_HUB_CACHE"] == "/workspace/huggingface/hub"

    def test_volume_without_datacenter_is_refused_before_renting(self):
        api = FakeApi()
        api.volumes["bad"] = {"id": "bad"}
        session = make_session(api, direct_vllm_image=True, image="image")

        with pytest.raises(RemoteExecutionError, match="dataCenterId"):
            session.start("model", network_volume_id="bad")

        assert api.created == []

    def test_timeout_includes_authenticated_redacted_startup_log(self):
        api = FakeApi()
        session = make_session(
            api,
            direct_vllm_image=True,
            image="image",
            token="secret-token",
            http_statuses=(502,),
            log_response=(200, "failed with secret-token: out of memory"),
        )
        session.ready_timeout_s = 0

        with pytest.raises(RemoteExecutionError) as caught:
            session.start("model")

        message = str(caught.value)
        assert "out of memory" in message
        assert "secret-token" not in message
        assert "[REDACTED]" in message
        log_url, headers = session._http_calls[-1]
        assert log_url == "https://pod-1-8001.proxy.runpod.net/logs"
        assert headers["Authorization"] == "Bearer secret-token"
        assert api.terminated == ["pod-1"]

    def test_timeout_preserves_non_200_diagnostic_body_without_token(self):
        api = FakeApi()
        session = make_session(
            api,
            direct_vllm_image=True,
            image="image",
            token="secret-token",
            http_statuses=(502,),
            log_response=(404, "proxy says secret-token is unavailable"),
        )
        session.ready_timeout_s = 0

        with pytest.raises(RemoteExecutionError) as caught:
            session.start("model")

        message = str(caught.value)
        assert "endpoint returned HTTP 404" in message
        assert "proxy says [REDACTED] is unavailable" in message
        assert "secret-token" not in message

    def test_tensor_parallel_defaults_to_gpu_count(self):
        api = FakeApi()
        session = make_session(api, direct_vllm_image=True, image="image")

        session.start("model", gpu_count=4)

        command = api.created[0]["docker_start_cmd"][0]
        assert "--tensor-parallel-size 4" in command

    def test_direct_image_cost_ceiling_terminates_immediately(self):
        class PricedApi(FakeApi):
            def create_pod(self, **kwargs):
                self.created.append(kwargs)
                return {"id": "pod-1", "costPerHr": 20.0}

        api = PricedApi()
        session = make_session(api, direct_vllm_image=True, image="image")

        with pytest.raises(RemoteExecutionError, match="above.*max-cost"):
            session.start("model", max_hours=0.5, max_cost_usd=5.0)

        assert api.terminated == ["pod-1"]

    def test_direct_image_refuses_adapters_before_renting(self, tmp_path):
        api = FakeApi()
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        session = make_session(api, direct_vllm_image=True, image="image")

        with pytest.raises(RemoteExecutionError, match="base checkpoints only"):
            session.start("model", adapter_dir=adapter)

        assert api.created == []

    def test_flashinfer_annotation_patch_runs_between_install_and_launch(self):
        """flashinfer's `array.array[int]` annotation is a TypeError at import
        on the image's Python 3.11 and kills the vLLM engine before it ever
        listens — observed on the first live-verified endpoint run. The patch
        must run after the install (so the file exists) and before the launch
        (so the engine survives)."""
        ssh = FakeSsh()
        session = make_session(ssh=ssh)

        session.start("Qwen/Qwen3.5-0.8B")

        install = next(
            i for i, c in enumerate(ssh.commands) if "pip install --quiet vllm" in c
        )
        patch = next(i for i, c in enumerate(ssh.commands) if "fd_exchange" in c)
        launch = next(i for i, c in enumerate(ssh.commands) if "vllm serve" in c)
        assert install < patch < launch
        assert "array.array[int]" in ssh.commands[patch]

    def test_connection_refusals_are_retried_until_ready(self):
        statuses = iter([ConnectionError("boot"), ConnectionError("boot"), 200])

        def http_get(url, headers):
            value = next(statuses)
            if isinstance(value, Exception):
                raise value
            return value

        session = RemoteServeSession(
            FakeApi(),
            FakeSsh(),
            public_key="k",
            poll_interval_s=0.0,
            http_get=http_get,
        )

        session.start("m")  # does not raise


class TestAdapter:
    def test_adapter_is_tarred_uploaded_and_lora_flags_added(self, tmp_path):
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_model.safetensors").write_bytes(b"x")
        ssh = FakeSsh()
        session = make_session(ssh=ssh)

        session.start("m", adapter_dir=adapter)

        assert any(r == "/workspace/adapter.tar.gz" for _, r in ssh.uploads)
        assert any("tar xzf /workspace/adapter.tar.gz" in c for c in ssh.commands)
        launch = next(c for c in ssh.commands if "vllm serve" in c)
        assert "--enable-lora" in launch
        assert "--lora-modules adapter=/workspace/adapter" in launch

    def test_multiple_adapters_ride_one_endpoint_under_their_own_names(self, tmp_path):
        """A/B comparison: each adapter is served under its own model name;
        request either (or the base) via the ``model`` field."""
        a, b = tmp_path / "gen1", tmp_path / "gen2"
        for d in (a, b):
            d.mkdir()
            (d / "adapter_model.safetensors").write_bytes(b"x")
        ssh = FakeSsh()
        session = make_session(ssh=ssh)

        session.start("m", adapters={"gen1": a, "gen2": b})

        uploads = [r for _, r in ssh.uploads]
        assert "/workspace/gen1.tar.gz" in uploads
        assert "/workspace/gen2.tar.gz" in uploads
        launch = next(c for c in ssh.commands if "vllm serve" in c)
        assert "gen1=/workspace/gen1" in launch
        assert "gen2=/workspace/gen2" in launch

    def test_adapter_dir_and_adapters_compose(self, tmp_path):
        """The single-adapter sugar keeps working alongside named ones."""
        sugar, named = tmp_path / "s", tmp_path / "n"
        for d in (sugar, named):
            d.mkdir()
        ssh = FakeSsh()
        session = make_session(ssh=ssh)

        session.start("m", adapter_dir=sugar, adapters={"candidate": named})

        launch = next(c for c in ssh.commands if "vllm serve" in c)
        assert "adapter=/workspace/adapter" in launch
        assert "candidate=/workspace/candidate" in launch


class TestSelfDestruct:
    def test_script_sleeps_then_deletes_its_own_pod_reading_the_key_file(self):
        script = self_destruct_script("pod-9", 1.5, "https://rest.runpod.io/v1")

        assert "sleep 5400" in script
        assert "DELETE" in script
        assert "https://rest.runpod.io/v1/pods/pod-9" in script
        # The key is read at fire time, never embedded in the script.
        assert "api_key=$(cat /workspace/.runpod_key)" in script
        assert "rm -f /workspace/.runpod_key" in script
        assert "Authorization: Bearer $api_key" in script
        assert "rp-test-key" not in script

    def test_script_quotes_custom_key_and_provider_url(self):
        script = self_destruct_script(
            "pod-9", 0.01, "https://provider.test/a path", "/tmp/key file"
        )

        assert "cat '/tmp/key file'" in script
        assert "'https://provider.test/a path/pods/pod-9'" in script

    def test_start_uploads_key_and_script_and_arms_before_vllm_install(self):
        ssh = FakeSsh()
        session = make_session(ssh=ssh)

        session.start("m", max_hours=2.0)

        remotes = [r for _, r in ssh.uploads]
        assert ssh.secret_uploads == [("rp-test-key", "/workspace/.runpod_key")]
        assert "/workspace/self_destruct.sh" in remotes
        arm = next(i for i, c in enumerate(ssh.commands) if "self_destruct.sh" in c)
        install = next(i for i, c in enumerate(ssh.commands) if "pip install" in c)
        assert arm < install, "self-destruct must be armed before fallible setup"
        assert "chmod 600 /workspace/.runpod_key" in ssh.commands[arm]
        assert "nohup bash /workspace/self_destruct.sh" in ssh.commands[arm]

    def test_the_arm_backgrounds_only_the_script_not_the_whole_chain(self):
        """In `chmod && nohup script > log & echo`, the `&` backgrounds the
        WHOLE `chmod && nohup` chain, whose subshell runs the hour-long
        script in its foreground holding the ssh session's stdout/stderr —
        so the client blocks on the arm command until the self-destruct
        fires. Observed live: 28 minutes hung on `echo armed`. The subshell
        `(nohup ... &)` scopes the `&` to the script launch alone."""
        ssh = FakeSsh()
        session = make_session(ssh=ssh)

        session.start("m", max_hours=2.0)

        arm = next(c for c in ssh.commands if "self_destruct.sh" in c)
        assert "(nohup bash /workspace/self_destruct.sh" in arm
        assert "< /dev/null &)" in arm

    def test_every_detached_launch_redirects_stdin(self):
        """Without < /dev/null the hour-long nohup'd script inherits the ssh
        session's stdin, sshd keeps the channel open until the self-destruct
        fires, and the client blocks on the arm command for the pod's whole
        lifetime. Observed live: the CLI hung 28 minutes on `echo armed`
        while the pod sat idle. Applies to every backgrounded launch."""
        ssh = FakeSsh()
        session = make_session(ssh=ssh)

        session.start("m", max_hours=2.0)

        backgrounded = [c for c in ssh.commands if "nohup" in c]
        assert backgrounded, "expected nohup'd launches"
        for command in backgrounded:
            assert "< /dev/null" in command, command

    def test_nonpositive_max_hours_is_rejected_before_renting(self):
        api = FakeApi()
        session = make_session(api)

        with pytest.raises(RemoteExecutionError, match="max-hours"):
            session.start("m", max_hours=0)

        assert api.created == []


class TestTransportRetry:
    def test_ssh_255_reconnects_and_retries_the_command_once(self):
        """Observed live: sshd dropped mid-`pip install vllm` (exit 255)."""
        api = FakeApi()
        ssh = FakeSsh()
        flaky = {"tripped": False}
        original_run = ssh.run

        def run(command):
            if "pip install" in command and not flaky["tripped"]:
                flaky["tripped"] = True
                return 255, "Connection closed by remote host"
            return original_run(command)

        ssh.run = run
        reconnects = []
        original_wait = ssh.wait_until_reachable
        ssh.wait_until_reachable = lambda h, p, t: (
            reconnects.append((h, p)),
            original_wait(h, p, t),
        )
        session = make_session(api, ssh)

        session.start("m")  # does not raise

        assert api.terminated == []
        assert sum(1 for c in ssh.commands if "pip install" in c) == 1
        # initial wait + one reconnect
        assert len(reconnects) == 2

    def test_persistent_255_still_fails_and_terminates(self):
        api = FakeApi()
        ssh = FakeSsh()
        ssh.run = lambda command: (255, "gone")
        session = make_session(api, ssh)

        with pytest.raises(RemoteExecutionError, match="255"):
            session.start("m")

        assert api.terminated == ["pod-1"]


class TestStartFailures:
    def test_remote_command_failure_terminates_the_pod(self):
        api, ssh = FakeApi(), FakeSsh()
        ssh.fail_on = "pip install"
        session = make_session(api, ssh)

        with pytest.raises(RemoteExecutionError, match="pip install"):
            session.start("m")

        assert api.terminated == ["pod-1"]

    def test_readiness_timeout_terminates_the_pod_and_includes_the_log(self):
        api = FakeApi()
        ssh = FakeSsh()
        session = make_session(api, ssh, http_statuses=(500,))
        session.ready_timeout_s = 0

        with pytest.raises(RemoteExecutionError, match="did not become ready"):
            session.start("m")

        assert api.terminated == ["pod-1"]
        assert any("tail" in c and "vllm.log" in c for c in ssh.commands)

    def test_401_from_vllm_fails_fast_instead_of_polling_forever(self):
        api = FakeApi()
        session = make_session(api, http_statuses=(401,))

        with pytest.raises(RemoteExecutionError, match="rejected"):
            session.start("m")

        assert api.terminated == ["pod-1"]

    def test_pod_never_publishing_ports_retries_on_a_fresh_pod(self):
        """A pod that reaches RUNNING without networking will never serve —
        observed four times against real RunPod hosts. Waiting out the long
        vLLM-load timeout on it burned 30 minutes of billing, so networking
        now fails fast and a *different* host is tried."""
        api = FakeApi()
        api.pod_state["publicIp"] = ""  # host never publishes an IP
        session = make_session(api)
        session.network_timeout_s = 0

        with pytest.raises(RemoteExecutionError, match="ssh endpoint"):
            session.start("m")

        # Both attempts' pods terminated: a retry must never double the bill.
        assert len(api.terminated) == session.max_provision_attempts
        assert len(api.created) == session.max_provision_attempts

    def test_networking_failure_is_not_bounded_by_the_vllm_load_timeout(self):
        """The two waits are different problems: vLLM legitimately takes many
        minutes, networking either appears in ~2 or never."""
        api = FakeApi()
        api.pod_state["publicIp"] = ""
        session = make_session(api)
        session.network_timeout_s = 0
        session.ready_timeout_s = 10_000  # would hang if it governed this wait

        with pytest.raises(RemoteExecutionError, match="ssh endpoint"):
            session.start("m")

    def test_a_single_attempt_does_not_retry(self):
        api = FakeApi()
        api.pod_state["publicIp"] = ""
        session = make_session(api)
        session.network_timeout_s = 0
        session.max_provision_attempts = 1

        with pytest.raises(RemoteExecutionError, match="ssh endpoint"):
            session.start("m")

        assert len(api.created) == 1
        assert api.terminated == ["pod-1"]

    def test_terminate_is_idempotent(self):
        api = FakeApi()
        session = make_session(api)
        session.start("m")

        session.terminate()
        session.terminate()

        assert api.terminated == ["pod-1"]


class TestListAndFind:
    PODS = [
        {
            "id": "a1",
            "name": "stateset-serve-abc",
            "desiredStatus": "RUNNING",
            "costPerHr": 0.17,
            "createdAt": "2026-08-13T00:00:00Z",
        },
        {"id": "b2", "name": "stateset-sft-xyz", "desiredStatus": "RUNNING"},
        {"id": "c3", "name": "stateset-serve-def", "desiredStatus": "EXITED"},
    ]

    def test_list_shows_only_serve_pods_with_age_and_cost(self):
        rows = list_serve_pods(FakeApi(pods=self.PODS))

        assert [r["id"] for r in rows] == ["a1", "c3"]
        assert rows[0]["cost_per_hr"] == 0.17
        assert rows[0]["age"].endswith("h")
        assert rows[1]["age"] == "?"  # no createdAt

    def test_find_matches_by_id_or_name(self):
        api = FakeApi(pods=self.PODS)

        assert find_serve_pod(api, "a1")["name"] == "stateset-serve-abc"
        assert find_serve_pod(api, "stateset-serve-def")["id"] == "c3"

    def test_find_unknown_lists_running_serve_pods_in_the_error(self):
        with pytest.raises(RemoteExecutionError, match="stateset-serve-abc"):
            find_serve_pod(FakeApi(pods=self.PODS), "nope")

    def test_generated_tokens_are_unique_and_urlsafe(self):
        tokens = {RemoteServeSession(FakeApi(), FakeSsh()).token for _ in range(5)}
        assert len(tokens) == 5
        assert all(len(t) >= 24 for t in tokens)


class TestDetachedSteps:
    """Long installs must survive a dropped ssh link.

    Observed live: the transport died partway through `pip install vllm`
    and took the whole run with it. The install now runs detached and its
    exit code is polled, so a dropped link costs a poll, not the install.
    """

    def test_install_is_launched_detached_not_held_open(self):
        api, ssh = FakeApi(), FakeSsh()
        make_session(api, ssh).start("m")

        launch = next(c for c in ssh.commands if "pip install --quiet vllm" in c)
        assert "nohup" in launch and launch.rstrip().endswith("&")
        assert any(
            c.startswith("cat /workspace/.vllm-install.rc") for c in ssh.commands
        )

    def test_a_failing_install_reports_its_log(self):
        api, ssh = FakeApi(), FakeSsh()
        ssh.fail_detached = "vllm-install"
        session = make_session(api, ssh)

        with pytest.raises(RemoteExecutionError, match="vllm-install failed"):
            session.start("m")

        assert api.terminated == ["pod-1"]

    def test_a_launch_that_cannot_start_is_a_failure(self):
        api, ssh = FakeApi(), FakeSsh()
        ssh.fail_on = "pip install"
        session = make_session(api, ssh)

        with pytest.raises(RemoteExecutionError, match="could not start"):
            session.start("m")


class TestMerge:
    """--merge exists because vLLM loads hybrid-Qwen3.5 LoRA adapters
    without error and silently serves the base weights (byte-identical
    greedy completions — docs/PROOFS.md 2026-08-18). Merging folds the
    deltas in with peft and serves an ordinary full checkpoint."""

    def _started(self, tmp_path, **kwargs):
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        ssh = FakeSsh()
        session = make_session(ssh=ssh)
        session.start("Qwen/Qwen3.5-0.8B", adapters={"adapter": adapter}, **kwargs)
        return ssh

    def test_merge_runs_between_patch_and_launch_and_serves_the_merged_dir(
        self, tmp_path
    ):
        ssh = self._started(tmp_path, merge=True)

        merge = next(i for i, c in enumerate(ssh.commands) if "merge_adapter" in c)
        launch = next(i for i, c in enumerate(ssh.commands) if "vllm serve" in c)
        patch = next(i for i, c in enumerate(ssh.commands) if "fd_exchange" in c)
        assert patch < merge < launch
        merge_cmd = ssh.commands[merge]
        assert "--base-model Qwen/Qwen3.5-0.8B" in merge_cmd
        assert "--adapter /workspace/adapter" in merge_cmd
        assert "--output-dir /workspace/merged" in merge_cmd

    def test_merged_launch_serves_full_weights_not_lora(self, tmp_path):
        ssh = self._started(tmp_path, merge=True)

        launch = next(c for c in ssh.commands if "vllm serve" in c)
        assert "vllm serve /workspace/merged" in launch
        assert "--enable-lora" not in launch
        # Same API name with and without --merge: callers always ask for
        # model "adapter".
        assert "--served-model-name adapter" in launch

    def test_merge_installs_the_training_deps_first(self, tmp_path):
        ssh = self._started(tmp_path, merge=True)

        deps = next(
            i for i, c in enumerate(ssh.commands) if "stateset-agents[training]" in c
        )
        merge = next(i for i, c in enumerate(ssh.commands) if "merge_adapter" in c)
        assert deps < merge

    def test_merge_with_multiple_adapters_is_refused_before_renting(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        for d in (a, b):
            d.mkdir()
        api = FakeApi()
        session = make_session(api)

        with pytest.raises(RemoteExecutionError, match="ONE adapter"):
            session.start("m", adapters={"x": a, "y": b}, merge=True)
        assert api.created == []

    def test_without_merge_nothing_changes(self, tmp_path):
        ssh = self._started(tmp_path, merge=False)

        assert not any("merge_adapter" in c for c in ssh.commands)
        launch = next(c for c in ssh.commands if "vllm serve" in c)
        assert "--enable-lora" in launch

    def test_merge_ships_the_env_wheel_when_set(self, tmp_path, monkeypatch):
        """STATESET_AGENTS_WHEEL reaches the merge deps install — the PyPI
        pin cannot contain an unreleased merge module."""
        wheel = tmp_path / "stateset_agents-9.9.9-py3-none-any.whl"
        wheel.write_bytes(b"x")
        monkeypatch.setenv("STATESET_AGENTS_WHEEL", str(wheel))
        ssh = self._started(tmp_path, merge=True)

        assert any(r.endswith(wheel.name) for _, r in ssh.uploads)
        deps = next(c for c in ssh.commands if "pip install" in c and "whl" in c)
        assert f"/workspace/{wheel.name}[training]" in deps


class TestAdapterEffectProbe:
    """The self-verifying serve: identical greedy completions from adapter
    and base mean the adapter has no effect — the exact silent no-op that
    survived a 'successful' verification once (docs/PROOFS.md 2026-08-18)."""

    def _adapter(self, tmp_path):
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        return adapter

    def test_differing_completions_pass_quietly(self, tmp_path):
        session = make_session(
            completions={"m": "base says hi", "adapter": "TechNest says hi"}
        )
        session.start("m", adapters={"adapter": self._adapter(tmp_path)})
        assert session.effect_warnings == []

    def test_identical_completions_warn_loudly_by_default(self, tmp_path):
        session = make_session(completions={"m": "same", "adapter": "same"})
        session.start("m", adapters={"adapter": self._adapter(tmp_path)})
        (warning,) = session.effect_warnings
        assert "NO EFFECT" in warning
        assert "--merge" in warning

    def test_identical_completions_raise_with_strict(self, tmp_path):
        api = FakeApi()
        session = make_session(api, completions={"m": "same", "adapter": "same"})
        with pytest.raises(RemoteExecutionError, match="NO EFFECT"):
            session.start(
                "m",
                adapters={"adapter": self._adapter(tmp_path)},
                strict_effect=True,
            )
        # A failed strict start terminates the pod like any startup failure.
        assert api.terminated == ["pod-1"]

    def test_probe_is_greedy_and_hits_every_adapter(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        for d in (a, b):
            d.mkdir()
        session = make_session()
        session.start("m", adapters={"x": a, "y": b})
        models = [c["model"] for c in session._post_calls]
        assert models == ["m", "x", "y"]
        assert all(c["temperature"] == 0 for c in session._post_calls)

    def test_probe_transport_failure_never_kills_a_healthy_serve(self, tmp_path):
        def broken_post(url, headers, payload):
            raise ConnectionError("proxy hiccup")

        session = RemoteServeSession(
            FakeApi(),
            FakeSsh(),
            public_key="k",
            poll_interval_s=0.0,
            http_get=lambda url, headers: 200,
            http_post_json=broken_post,
        )
        session.start("m", adapters={"adapter": self._adapter(tmp_path)})
        assert any("probe skipped" in w for w in session.effect_warnings)

    def test_no_probe_without_adapters_or_with_merge(self, tmp_path):
        session = make_session()
        session.start("m")
        assert session._post_calls == []

        merged = make_session()
        merged.start("m", adapters={"adapter": self._adapter(tmp_path)}, merge=True)
        assert merged._post_calls == []
