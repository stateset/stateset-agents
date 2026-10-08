"""Resource admission follows measured process limits, never invented defaults."""

import asyncio
from types import SimpleNamespace

import pytest

from stateset_agents.training import advanced_training_orchestrator as orchestrator
from stateset_agents.training import resource_capacity as capacity
from stateset_agents.training.advanced_training_models import (
    ResourceRequirement,
    ResourceType,
)

GIB = 1024**3


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(value))


@pytest.fixture
def v2(tmp_path, monkeypatch):
    monkeypatch.setattr(capacity, "sys", SimpleNamespace(platform="linux"))
    proc, mount = tmp_path / "proc", tmp_path / "cgroup mount"
    mount.mkdir()
    write(proc / "self/cgroup", "0::/team/task\n")
    escaped = str(mount).replace(" ", "\\040")
    write(proc / "self/mountinfo", f"1 0 0:1 / {escaped} rw - cgroup2 cgroup rw\n")
    leaf = mount / "team/task"
    write(leaf / "cpu.max", "200000 100000")
    write(leaf / "memory.max", 2 * GIB)
    write(leaf / "memory.current", GIB // 2)
    return proc, mount, leaf


def test_v2_applies_parent_cpu_quota_and_shared_memory_headroom(v2):
    proc, mount, leaf = v2
    write(mount / "team/cpu.max", "50000 100000")
    write(mount / "team/memory.max", 4 * GIB)
    write(mount / "team/memory.current", 3 * GIB)
    result = capacity.read_cgroup_capacity(proc)
    assert result.cpu == 0.5 and result.memory_bytes == GIB
    assert not result.issues
    write(leaf / "memory.current", 3 * GIB)
    assert capacity.read_cgroup_capacity(proc).memory_bytes == 0


def test_private_namespace_maps_root_membership_to_visible_mount(v2):
    proc, mount, _ = v2
    write(proc / "self/cgroup", "0::/\n")
    escaped = str(mount).replace(" ", "\\040")
    write(
        proc / "self/mountinfo",
        f"1 0 0:1 /docker/container {escaped} rw - cgroup2 cgroup rw\n",
    )
    write(mount / "cpu.max", "75000 100000")
    write(mount / "memory.max", GIB)
    write(mount / "memory.current", GIB // 4)
    result = capacity.read_cgroup_capacity(proc)
    assert result.cpu == 0.75 and result.memory_bytes == 3 * GIB // 4


def test_v1_separate_controllers_honor_ancestor_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(capacity, "sys", SimpleNamespace(platform="linux"))
    proc = tmp_path / "proc"
    write(proc / "self/cgroup", "2:cpu,cpuacct:/team/task\n3:memory:/team/task\n")
    mounts = []
    for index, controller in enumerate(("cpu", "memory"), 1):
        mount = tmp_path / controller
        mounts.append(
            f"{index} 0 0:{index} / {mount} rw - cgroup cgroup rw,{controller}\n"
        )
        for suffix in ("", "team", "team/task"):
            directory = mount / suffix
            if controller == "cpu":
                write(directory / "cpu.cfs_quota_us", -1)
                write(directory / "cpu.cfs_period_us", 100000)
            else:
                write(directory / "memory.limit_in_bytes", 2**63 - 4096)
                write(directory / "memory.usage_in_bytes", 0)
    write(proc / "self/mountinfo", "".join(mounts))
    write(tmp_path / "cpu/team/cpu.cfs_quota_us", 150000)
    write(tmp_path / "memory/team/memory.limit_in_bytes", 2 * GIB)
    write(tmp_path / "memory/team/memory.usage_in_bytes", GIB)
    result = capacity.read_cgroup_capacity(proc)
    assert result.cpu == 1.5 and result.memory_bytes == GIB
    assert not result.issues


@pytest.mark.parametrize("text", ["max", "1 0", "bad 100000", "1 2 extra", "-5 100000"])
def test_bad_cpu_limits_block_cpu_without_erasing_memory(v2, text):
    proc, _, leaf = v2
    write(leaf / "cpu.max", text)
    result = capacity.read_cgroup_capacity(proc)
    assert result.cpu == 0 and result.memory_bytes == 3 * GIB // 2
    assert result.issues


@pytest.mark.parametrize("text", ["", "-1", "bad"])
def test_bad_memory_limits_do_not_turn_into_host_capacity(v2, text):
    proc, _, leaf = v2
    write(leaf / "memory.max", text)
    result = capacity.read_cgroup_capacity(proc)
    assert result.cpu == 2 and result.memory_bytes == 0 and result.issues


def test_unlimited_v2_controllers_do_not_add_a_limit(v2):
    proc, _, leaf = v2
    write(leaf / "cpu.max", "max 100000")
    write(leaf / "memory.max", "max")
    assert capacity.read_cgroup_capacity(proc) == capacity.CgroupCapacity()


def test_unreadable_controller_fails_closed_independently(v2, monkeypatch):
    proc, _, leaf = v2
    path_type = type(leaf)
    original = path_type.read_text

    def read(path, *args, **kwargs):
        if path == leaf / "cpu.max":
            raise PermissionError("controller unreadable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(path_type, "read_text", read)
    result = capacity.read_cgroup_capacity(proc)
    assert result.cpu == 0 and result.memory_bytes == 3 * GIB // 2
    assert "PermissionError" in result.issues[0]


def test_other_platforms_do_not_require_proc_files(tmp_path, monkeypatch):
    monkeypatch.setattr(capacity, "sys", SimpleNamespace(platform="win32"))
    assert capacity.read_cgroup_capacity(tmp_path) == capacity.CgroupCapacity()


@pytest.mark.parametrize("membership", ["0::/missing\n", "0::/../escape\n", "broken"])
def test_unresolvable_membership_fails_closed(v2, membership):
    proc, _, _ = v2
    write(proc / "self/cgroup", membership)
    result = capacity.read_cgroup_capacity(proc)
    assert result.cpu == result.memory_bytes == 0 and result.issues


@pytest.fixture
def detected(monkeypatch):
    monkeypatch.setattr(orchestrator.os, "cpu_count", lambda: 32)
    monkeypatch.setattr(
        orchestrator.os, "sched_getaffinity", lambda pid: set(range(4)), raising=False
    )
    monkeypatch.setattr(orchestrator, "PSUTIL_AVAILABLE", True)
    monkeypatch.setattr(
        orchestrator,
        "psutil",
        SimpleNamespace(
            virtual_memory=lambda: SimpleNamespace(total=64 * GIB, available=12 * GIB)
        ),
    )
    monkeypatch.setattr(
        orchestrator.shutil, "disk_usage", lambda path: SimpleNamespace(free=20 * GIB)
    )
    monkeypatch.setattr(orchestrator, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(
        orchestrator,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2)
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "read_cgroup_capacity",
        lambda: capacity.CgroupCapacity(1.5, 3 * GIB),
    )


def test_detected_capacity_and_admission_respect_process_limits(detected):
    manager = orchestrator.ResourceManager()
    assert manager.available_resources == {
        ResourceType.CPU: 1.5,
        ResourceType.GPU: 2.0,
        ResourceType.MEMORY: 3.0,
        ResourceType.STORAGE: 20.0,
        ResourceType.NETWORK: 0.0,
    }
    assert not asyncio.run(
        manager.allocate_resources(
            "host-sized", [ResourceRequirement(ResourceType.CPU, 2)]
        )
    )
    assert not asyncio.run(
        manager.allocate_resources(
            "too-much-memory", [ResourceRequirement(ResourceType.MEMORY, 4)]
        )
    )


def test_gpu_probe_failure_preserves_other_measurements(detected, monkeypatch):
    def broken():
        raise RuntimeError("driver unavailable")

    monkeypatch.setattr(orchestrator.torch.cuda, "is_available", broken)
    manager = orchestrator.ResourceManager()
    assert manager.available_resources[ResourceType.GPU] == 0
    assert manager.available_resources[ResourceType.CPU] == 1.5
    assert manager.available_resources[ResourceType.MEMORY] == 3
    assert manager.available_resources[ResourceType.STORAGE] == 20
    assert any("CUDA" in issue for issue in manager.detection_issues)


def test_missing_psutil_uses_measured_stdlib_memory(detected, monkeypatch):
    monkeypatch.setattr(orchestrator, "PSUTIL_AVAILABLE", False)
    monkeypatch.setattr(
        orchestrator.os,
        "sysconf",
        lambda name: {"SC_AVPHYS_PAGES": 1024, "SC_PAGE_SIZE": 4096}[name],
        raising=False,
    )
    manager = orchestrator.ResourceManager()
    assert manager.available_resources[ResourceType.MEMORY] == 4 / 1024


def test_storage_uses_selected_filesystem_and_reports_unknown_capacity(
    detected, monkeypatch, tmp_path
):
    def unavailable(path):
        assert path == tmp_path
        raise OSError("not mounted")

    monkeypatch.setattr(orchestrator.shutil, "disk_usage", unavailable)
    manager = orchestrator.ResourceManager(storage_path=tmp_path)
    assert manager.available_resources[ResourceType.STORAGE] == 0
    assert manager.available_resources[ResourceType.CPU] == 1.5
    assert any("Storage" in issue for issue in manager.detection_issues)


def test_explicit_allowances_are_labelled_as_configured(detected):
    manager = orchestrator.ResourceManager(
        capacity_overrides={ResourceType.NETWORK: 50, ResourceType.CPU: 1}
    )
    assert manager.available_resources[ResourceType.NETWORK] == 50
    assert manager.available_resources[ResourceType.CPU] == 1
    assert manager.resource_sources[ResourceType.NETWORK] == "configured"


def test_orchestrator_preserves_injected_capacity_and_exposes_its_source(
    detected, monkeypatch
):
    manager = orchestrator.ResourceManager(
        capacity_overrides={ResourceType.NETWORK: 50}
    )
    monkeypatch.setattr(orchestrator, "get_state_service", SimpleNamespace)
    monkeypatch.setattr(orchestrator, "get_monitoring_service", SimpleNamespace)
    monkeypatch.setattr(
        orchestrator.ResourceManager,
        "_detect_resources",
        lambda self: pytest.fail("replaced configured manager"),
    )
    service = orchestrator.AdvancedTrainingOrchestrator(
        resource_manager=manager,
        start_background_tasks=False,
        enable_experiment_tracking=False,
    )
    assert service.resource_manager is manager
    status = asyncio.run(service.get_system_status())
    assert status["available_resources"]["network"] == 50
    assert status["resource_sources"]["network"] == "configured"
    assert status["resource_detection_issues"] == []


def test_unknown_host_count_can_use_known_affinity(detected, monkeypatch):
    monkeypatch.setattr(orchestrator.os, "cpu_count", lambda: None)
    assert orchestrator.ResourceManager().available_resources[ResourceType.CPU] == 1.5


def test_unavailable_memory_is_not_replaced_by_eight_gib(detected, monkeypatch):
    def unavailable():
        raise OSError("cannot inspect memory")

    monkeypatch.setattr(orchestrator.psutil, "virtual_memory", unavailable)
    manager = orchestrator.ResourceManager()
    assert manager.available_resources[ResourceType.MEMORY] == 0
    assert manager.available_resources[ResourceType.CPU] == 1.5


@pytest.mark.parametrize("value", [-1, True, float("nan"), float("inf"), "4"])
def test_invalid_configured_capacity_fails_before_detection(
    detected, monkeypatch, value
):
    monkeypatch.setattr(
        orchestrator.ResourceManager,
        "_detect_resources",
        lambda self: pytest.fail("invalid override reached detection"),
    )
    with pytest.raises(ValueError):
        orchestrator.ResourceManager(capacity_overrides={ResourceType.CPU: value})
