"""Read visible Linux cgroup CPU limits and memory headroom without dependencies."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class CgroupCapacity:
    """Observed ceilings; None means no visible constraint, zero prevents admission."""

    cpu: float | None = None
    memory_bytes: int | None = None
    issues: tuple[str, ...] = ()


def _unescape(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def _path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Invalid cgroup path")
    return path


def _read_optional(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return None


def _cpu_limit(directory: Path, version: int) -> float | None:
    if version == 2:
        text = _read_optional(directory / "cpu.max")
        if text is None:
            return None  # Controller disabled here, or hierarchy root.
        quota, period = text.split()
    else:
        quota = (directory / "cpu.cfs_quota_us").read_text().strip()
        period = (directory / "cpu.cfs_period_us").read_text().strip()
    interval = int(period)
    if interval <= 0:
        raise ValueError("Invalid CPU quota period")
    if version == 2 and quota == "max":
        return None
    maximum = int(quota)
    if version == 1 and maximum < 0:
        return None
    if maximum < 0:
        raise ValueError("Invalid CPU quota")
    return maximum / interval


def _memory_headroom(directory: Path, version: int) -> int | None:
    limit_name = "memory.max" if version == 2 else "memory.limit_in_bytes"
    usage_name = "memory.current" if version == 2 else "memory.usage_in_bytes"
    text = _read_optional(directory / limit_name)
    if version == 2 and text in (None, "max"):
        return None
    if text is None:
        raise ValueError("Missing memory controller limit")
    limit = int(text)
    used = int((directory / usage_name).read_text().strip())
    if limit < 0 or used < 0:
        raise ValueError("Invalid memory controller counters")
    return max(0, limit - used)


def read_cgroup_capacity(proc_root: Path = Path("/proc")) -> CgroupCapacity:
    """Take the smallest constraint across visible ancestors in v1/v2 mounts.

    Mount roots and process membership determine paths; no fixed Docker or
    Kubernetes layout is assumed. Unreadable or malformed relevant controllers
    yield zero capacity for that resource, never a fabricated host allowance.
    Hidden ancestors and concurrent resource consumption cannot be measured here.
    """
    if sys.platform != "linux":
        return CgroupCapacity()
    try:
        memberships: dict[str, PurePosixPath] = {}
        for line in (proc_root / "self/cgroup").read_text().splitlines():
            _, controller_names, member_text = line.split(":", 2)
            for controller in controller_names.split(","):
                memberships[controller] = _path(member_text)
        mounts = []
        for line in (proc_root / "self/mountinfo").read_text().splitlines():
            before, after = line.split(" - ", 1)
            fields, filesystem = before.split(), after.split()
            if filesystem[0] not in {"cgroup", "cgroup2"}:
                continue
            mounts.append(
                (
                    2 if filesystem[0] == "cgroup2" else 1,
                    _path(_unescape(fields[3])),
                    Path(str(_path(_unescape(fields[4])))),
                    set(filesystem[2].split(",")),
                )
            )
    except (OSError, ValueError, IndexError) as exc:
        return CgroupCapacity(
            0.0, 0, (f"Cannot inspect cgroup membership ({type(exc).__name__})",)
        )

    results: dict[str, float | int | None] = {"cpu": None, "memory": None}
    issues = []
    for controller in results:
        version = 1 if controller in memberships else 2
        member = memberships.get(controller if version == 1 else "")
        if member is None:
            continue
        matched = False
        try:
            for mount_version, root, mount, controllers in mounts:
                if mount_version != version or (
                    version == 1 and controller not in controllers
                ):
                    continue
                if member == PurePosixPath("/"):
                    relative = PurePosixPath(".")  # Private cgroup namespace root.
                else:
                    try:
                        relative = member.relative_to(root)
                    except ValueError:
                        continue
                matched = True
                current = mount / str(relative)
                if not current.is_dir():
                    raise ValueError("Process cgroup is not visible in its mount")
                while True:
                    value = (
                        _cpu_limit(current, version)
                        if controller == "cpu"
                        else _memory_headroom(current, version)
                    )
                    previous = results[controller]
                    if value is not None:
                        results[controller] = (
                            value if previous is None else min(previous, value)
                        )
                    if current == mount:
                        break
                    current = current.parent
            if not matched:
                raise ValueError("No matching cgroup mount is visible")
        except (OSError, ValueError, OverflowError) as exc:
            results[controller] = 0
            issues.append(
                f"Cannot inspect {controller} cgroup capacity ({type(exc).__name__})"
            )
    cpu, memory = results["cpu"], results["memory"]
    return CgroupCapacity(
        float(cpu) if cpu is not None else None,
        int(memory) if memory is not None else None,
        tuple(issues),
    )
