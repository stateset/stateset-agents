"""Unit tests for :mod:`stateset_agents.remote.serve_session`.

Everything is faked at the RunPodApi/SshTransport/http_get seams — no
network, no ssh, no pods. What matters: the pod is provisioned with BOTH
ports, the self-destruct is armed before anything fallible, the endpoint
URL comes from the 8000 port mapping, readiness polls /v1/models with the
Bearer token, startup failures terminate the pod, and success does NOT.
"""

from __future__ import annotations

import pytest

from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.serve_session import (
    RemoteServeSession,
    find_serve_pod,
    list_serve_pods,
    self_destruct_script,
)


class FakeApi:
    def __init__(self, pods=None):
        self.api_key = "rp-test-key"
        self.root = "https://rest.runpod.io/v1"
        self.created: list[dict] = []
        self.terminated: list[str] = []
        self._pods = pods or []
        self.pod_state = {
            "id": "pod-1",
            "desiredStatus": "RUNNING",
            "publicIp": "1.2.3.4",
            "portMappings": {"22": 2222, "8000": 18000},
        }

    def create_pod(self, **kwargs):
        self.created.append(kwargs)
        return {"id": "pod-1"}

    def get_pod(self, pod_id):
        return dict(self.pod_state)

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)

    def list_pods(self):
        return list(self._pods)


class FakeSsh:
    def __init__(self):
        self.commands: list[str] = []
        self.uploads: list[tuple[str, str]] = []
        self.fail_on: str | None = None
        #: Arm a detached step (by label) to report a non-zero exit code.
        self.fail_detached: str | None = None

    def wait_until_reachable(self, host, port, timeout_s):
        self.reachable = (host, port)

    def upload(self, local, remote):
        self.uploads.append((str(local), remote))

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


def make_session(api=None, ssh=None, http_statuses=(200,), **kwargs):
    statuses = list(http_statuses)
    calls: list[tuple[str, dict]] = []

    def http_get(url, headers):
        calls.append((url, dict(headers)))
        return statuses.pop(0) if len(statuses) > 1 else statuses[0]

    session = RemoteServeSession(
        api or FakeApi(),
        ssh or FakeSsh(),
        public_key="ssh-ed25519 AAA test",
        poll_interval_s=0.0,
        http_get=http_get,
        **kwargs,
    )
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
        assert "$(cat /workspace/.runpod_key)" in script
        assert "rp-test-key" not in script

    def test_start_uploads_key_and_script_and_arms_before_vllm_install(self):
        ssh = FakeSsh()
        session = make_session(ssh=ssh)

        session.start("m", max_hours=2.0)

        remotes = [r for _, r in ssh.uploads]
        assert "/workspace/.runpod_key" in remotes
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
