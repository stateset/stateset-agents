"""The sandbox adapter's local decisions: bootstrap, preflight, error mapping."""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from stateset_agents.fanout.backend import (
    AllocationLostError,
    BackendError,
    CapacityError,
    FatalBackendError,
    TransientBackendError,
)
from stateset_agents.fanout.manifest import FanoutManifest, ManifestError
from stateset_agents.fanout.sandbox_backend import (
    DEFAULT_REQUESTS_PER_MINUTE,
    RequestThrottle,
    SandboxBackend,
    boot_command,
)

ARGS = [
    "--spec",
    "/r/spec.json",
    "--output-dir",
    "/r/output",
    "--stage-dir",
    "/r/stage",
]
CUDA_FREE_INDEX = "https://download.pytorch.org/whl/cpu"


def _manifest(tmp_path: Path, **overrides: Any) -> FanoutManifest:
    data: dict[str, Any] = {
        "schema_version": 1,
        "name": "demo",
        "seeds": [1],
        "workload": {"kind": "command", "argv": ["t", "{seed}", "{output_dir}"]},
        "report": {"metric": "m", "mode": "max"},
    }
    data.update(overrides)
    return FanoutManifest.from_dict(data, base_dir=tmp_path)


class TestBootCommand:
    def test_preinstalled_runs_the_worker_directly(self, tmp_path: Path) -> None:
        manifest = _manifest(
            tmp_path,
            package={"source": "preinstalled"},
            resources={"image": "org:rl"},
        )

        assert boot_command(manifest, root="/r", worker_args=ARGS) == [
            "python3",
            "-m",
            "stateset_agents.fanout.worker",
            *ARGS,
        ]

    def test_pypi_installs_the_pinned_version_with_its_extras(
        self, tmp_path: Path
    ) -> None:
        manifest = _manifest(
            tmp_path,
            package={"source": "pypi", "version": "1.2.3", "extras": ["training"]},
        )

        script = boot_command(manifest, root="/r", worker_args=ARGS)[2]

        assert "stateset-agents[training]==1.2.3" in script
        assert "exec /r/venv/bin/python -m stateset_agents.fanout.worker" in script

    def test_wheel_installs_the_uploaded_file(self, tmp_path: Path) -> None:
        (tmp_path / "pkg-1.0-py3-none-any.whl").write_bytes(b"x")
        manifest = _manifest(
            tmp_path,
            package={"source": "wheel", "wheel": "pkg-1.0-py3-none-any.whl"},
        )

        script = boot_command(manifest, root="/r", worker_args=ARGS)[2]

        assert "/r/package/pkg-1.0-py3-none-any.whl" in script

    def test_no_deps_is_honored(self, tmp_path: Path) -> None:
        manifest = _manifest(
            tmp_path,
            package={"source": "pypi", "version": "1.0", "install_deps": False},
        )

        assert "--no-deps" in boot_command(manifest, root="/r", worker_args=ARGS)[2]

    def test_a_failed_install_stops_before_the_worker(self, tmp_path: Path) -> None:
        """Otherwise an import error would be reported as a workload failure."""
        manifest = _manifest(tmp_path, package={"source": "pypi", "version": "1.0"})

        script = boot_command(manifest, root="/r", worker_args=ARGS)[2]

        assert script.startswith("set -eu;")

    def test_cpu_sandboxes_get_cpu_torch(self, tmp_path: Path) -> None:
        """The default torch wheel pulls gigabytes of CUDA a CPU pod has no room for."""
        manifest = _manifest(
            tmp_path,
            package={"source": "pypi", "version": "1.0", "extras": ["training"]},
        )

        assert CUDA_FREE_INDEX in boot_command(manifest, root="/r", worker_args=ARGS)[2]

    def test_gpu_sandboxes_keep_cuda_torch(self, tmp_path: Path) -> None:
        manifest = _manifest(
            tmp_path,
            package={"source": "pypi", "version": "1.0", "extras": ["training"]},
            resources={"gpu": {"type": "nvidia-l4"}},
        )

        assert (
            CUDA_FREE_INDEX
            not in boot_command(manifest, root="/r", worker_args=ARGS)[2]
        )

    def test_no_training_extra_needs_no_torch_index(self, tmp_path: Path) -> None:
        manifest = _manifest(tmp_path, package={"source": "pypi", "version": "1.0"})

        assert (
            CUDA_FREE_INDEX
            not in boot_command(manifest, root="/r", worker_args=ARGS)[2]
        )

    def test_script_quotes_hostile_arguments(self, tmp_path: Path) -> None:
        manifest = _manifest(tmp_path, package={"source": "pypi", "version": "1.0"})
        hostile = ["--spec", "/r/x y;rm -rf /.json"]

        script = boot_command(manifest, root="/r", worker_args=hostile)[2]

        assert shlex.quote("/r/x y;rm -rf /.json") in script

    def test_generated_script_is_valid_shell(self, tmp_path: Path) -> None:
        manifest = _manifest(
            tmp_path,
            package={"source": "pypi", "version": "1.0", "extras": ["training"]},
        )

        script = boot_command(manifest, root="/r", worker_args=ARGS)[2]

        # Not ``sh -n``: dash ignores -n under -c and would really run the script.
        subprocess.run(["bash", "--posix", "-n", "-c", script], check=True)


class TestControllerCommandRules:
    """``CommandArgSchema`` in the controller: 10000 chars, no control characters."""

    @pytest.mark.parametrize(
        "package",
        [
            {"source": "pypi", "version": "1.0"},
            {"source": "pypi", "version": "1.0", "extras": ["training"]},
            {"source": "pypi", "version": "1.0", "install_deps": False},
        ],
    )
    def test_every_argument_is_acceptable(
        self, tmp_path: Path, package: dict[str, Any]
    ) -> None:
        manifest = _manifest(tmp_path, package=package)

        argv = boot_command(manifest, root="/r", worker_args=ARGS)

        for argument in argv:
            assert not re.search(r"[\0\r\t\n]", argument), repr(argument)
            assert 1 <= len(argument) <= 10000


class TestRequestThrottle:
    def test_spaces_requests_to_the_configured_rate(self) -> None:
        now = [0.0]
        slept: list[float] = []

        def sleep(seconds: float) -> None:
            slept.append(seconds)
            now[0] += seconds

        throttle = RequestThrottle(60, clock=lambda: now[0], sleep=sleep)

        for _ in range(4):
            throttle.acquire()

        assert now[0] == pytest.approx(3.0)
        assert slept == pytest.approx([1.0, 1.0, 1.0])

    def test_time_already_passed_is_not_slept_again(self) -> None:
        now = [0.0]
        slept: list[float] = []
        throttle = RequestThrottle(60, clock=lambda: now[0], sleep=slept.append)

        throttle.acquire()
        now[0] = 10.0
        throttle.acquire()

        assert slept == []

    @pytest.mark.parametrize("rate", [0, -1, 1.5, True])
    def test_rate_must_be_a_positive_integer(self, rate: Any) -> None:
        with pytest.raises(ValueError):
            RequestThrottle(rate)

    def test_default_rate_stays_under_the_smallest_plan_and_ip_limits(self) -> None:
        # Hobby allows 60 requests a minute and the per-IP default is 100.
        assert DEFAULT_REQUESTS_PER_MINUTE < 60


class TestPreflight:
    def _backend(self) -> SandboxBackend:
        return SandboxBackend(
            api_key="k", base_url="http://localhost:1", client=object()
        )

    def test_preinstalled_needs_an_image_that_has_it(self, tmp_path: Path) -> None:
        """The stock sandbox image has no stateset_agents: every run would die."""
        with pytest.raises(ManifestError, match="package.source"):
            self._backend().validate(_manifest(tmp_path))

    @pytest.mark.parametrize(
        "name",
        [
            "AWS_ACCESS_KEY_ID",
            "KUBERNETES_SERVICE_HOST",
            "GCP_THING",
            "EXEC_AGENT_PORT",
            "PATH",
            "path",
            "LD_LIBRARY_PATH",
            "HOME",
            "NODE_OPTIONS",
        ],
    )
    def test_env_the_controller_refuses_is_caught_before_allocating(
        self, tmp_path: Path, name: str
    ) -> None:
        manifest = _manifest(
            tmp_path, resources={"image": "org:rl"}, forward_env=[name]
        )

        with pytest.raises(ManifestError, match=name):
            self._backend().validate(manifest)

    def test_custom_image_may_be_preinstalled(self, tmp_path: Path) -> None:
        self._backend().validate(_manifest(tmp_path, resources={"image": "org:rl"}))

    @pytest.mark.parametrize(
        "workload",
        [
            {"kind": "train", "agent": {"model_name": "stub://x"}},
            {"kind": "workflow_regression"},
        ],
    )
    def test_learning_workloads_need_the_training_extra(
        self, tmp_path: Path, workload: dict[str, Any]
    ) -> None:
        manifest = _manifest(
            tmp_path,
            workload=workload,
            package={"source": "pypi", "version": "1.0"},
        )

        with pytest.raises(ManifestError, match="training"):
            self._backend().validate(manifest)

    def test_learning_workloads_cannot_skip_dependencies(self, tmp_path: Path) -> None:
        manifest = _manifest(
            tmp_path,
            workload={"kind": "workflow_regression"},
            package={
                "source": "pypi",
                "version": "1.0",
                "extras": ["training"],
                "install_deps": False,
            },
        )

        with pytest.raises(ManifestError, match="install_deps"):
            self._backend().validate(manifest)

    def test_a_custom_image_can_carry_the_training_stack(self, tmp_path: Path) -> None:
        self._backend().validate(
            _manifest(
                tmp_path,
                workload={"kind": "workflow_regression"},
                resources={"image": "org:rl"},
            )
        )


class TestCreateResponses:
    """A created sandbox must not be lost to a parsing problem."""

    class _Client:
        def __init__(self, response: Any) -> None:
            self.response = response
            self.keys: list[str] = []

        def _request(
            self, method: str, path: str, body: Any, *, idempotency_key: str
        ) -> Any:
            self.keys.append(idempotency_key)
            return self.response

    def _allocate(self, response: Any) -> Any:
        pytest.importorskip("stateset_sandbox")
        backend = SandboxBackend(
            client=self._Client(response), requests_per_minute=600_000
        )
        manifest = FanoutManifest.from_dict(
            {
                "schema_version": 1,
                "name": "d",
                "seeds": [1],
                "workload": {
                    "kind": "command",
                    "argv": ["t", "{seed}", "{output_dir}"],
                },
                "report": {"metric": "m", "mode": "max"},
            }
        )
        return backend.allocate(
            run_id="r", resources=manifest.resources, idempotency_key="key"
        )

    def test_an_id_is_all_it_takes(self) -> None:
        """A new status or a missing field must not orphan what was created."""
        allocation = self._allocate({"sandbox_id": "sbx-1", "status": "brand-new"})

        assert allocation.id == "sbx-1"
        assert allocation.lifetime_s is None

    @pytest.mark.parametrize(
        "response", [{}, {"sandbox_id": ""}, {"sandbox_id": 7}, []]
    )
    def test_success_without_an_id_means_ask_again(self, response: Any) -> None:
        with pytest.raises(TransientBackendError):
            self._allocate(response)

    def test_reads_the_lifetime_the_platform_granted(self) -> None:
        allocation = self._allocate(
            {
                "sandbox_id": "sbx-1",
                "created_at": "2026-11-08T10:00:00.000Z",
                "expires_at": "2026-11-08T10:45:00.000Z",
            }
        )

        assert allocation.lifetime_s == pytest.approx(2700.0)

    def test_unreadable_times_mean_unknown(self) -> None:
        allocation = self._allocate(
            {"sandbox_id": "sbx-1", "created_at": "soon", "expires_at": 7}
        )

        assert allocation.lifetime_s is None


class TestConstruction:
    def test_missing_key_says_which_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("stateset_sandbox")
        monkeypatch.delenv("STATESET_API_KEY", raising=False)

        with pytest.raises(FatalBackendError, match="STATESET_API_KEY"):
            SandboxBackend()

    def test_missing_sdk_says_how_to_install_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "stateset_sandbox", None)

        with pytest.raises(FatalBackendError, match=r"stateset-agents\[sandbox\]"):
            SandboxBackend(api_key="k")


class TestErrorMapping:
    @pytest.mark.parametrize(
        ("status", "code", "expected"),
        [
            (429, "CONCURRENT_LIMIT_EXCEEDED", CapacityError),
            (429, "RATE_LIMIT_EXCEEDED", CapacityError),
            (429, "VCPU_RATE_LIMIT_EXCEEDED", CapacityError),
            (401, "AUTH_INVALID_API_KEY", FatalBackendError),
            (403, "FORBIDDEN", FatalBackendError),
            (403, "PLAN_UPGRADE_REQUIRED", FatalBackendError),
            (402, "PAYMENT_REQUIRED", FatalBackendError),
            (402, "RESOURCE_CLASS_NOT_BILLABLE", FatalBackendError),
            (400, "FEATURE_DISABLED", FatalBackendError),
            (400, "RESOURCE_CLASS_UNAVAILABLE", FatalBackendError),
            (400, "INVALID_FIELD_VALUE", FatalBackendError),
            (404, "SANDBOX_NOT_FOUND", AllocationLostError),
            (409, "SANDBOX_INVALID_STATE", AllocationLostError),
            (409, "SANDBOX_NOT_RUNNING", AllocationLostError),
            (429, "UNKNOWN_ERROR", CapacityError),
            (429, "PLAN_LIMIT_EXCEEDED", FatalBackendError),
            (500, "INTERNAL_ERROR", TransientBackendError),
            (503, "SERVICE_UNAVAILABLE", TransientBackendError),
            (404, "PROCESS_NOT_FOUND", BackendError),
        ],
    )
    def test_codes_map_to_the_right_response(
        self, status: int, code: str, expected: type[BaseException]
    ) -> None:
        pytest.importorskip("stateset_sandbox")
        from stateset_sandbox.errors import SandboxApiError

        from stateset_agents.fanout.sandbox_backend import translate_error

        error = SandboxApiError.from_response(
            {"error": {"code": code, "message": "m"}}, status
        )

        translated = translate_error(error)

        assert type(translated) is expected

    def test_capacity_error_keeps_retry_after(self) -> None:
        pytest.importorskip("stateset_sandbox")
        from stateset_sandbox.errors import SandboxApiError

        from stateset_agents.fanout.sandbox_backend import translate_error

        error = SandboxApiError.from_response(
            {"error": {"code": "CONCURRENT_LIMIT_EXCEEDED", "message": "m"}},
            429,
            retry_after=7.0,
        )

        translated = translate_error(error)

        assert isinstance(translated, CapacityError)
        assert translated.retry_after == 7.0

    def test_network_errors_are_transient(self) -> None:
        pytest.importorskip("stateset_sandbox")
        from stateset_sandbox.errors import SandboxNetworkError, SandboxTimeoutError

        from stateset_agents.fanout.sandbox_backend import translate_error

        assert isinstance(
            translate_error(SandboxNetworkError("down")), TransientBackendError
        )
        assert isinstance(
            translate_error(SandboxTimeoutError("slow", 1000)), TransientBackendError
        )
