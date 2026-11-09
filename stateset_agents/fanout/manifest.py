"""Declarative description of an RL fan-out: N independent runs from one file.

A fan-out multiplies spend, so this module's job is to refuse the mistakes that
would otherwise be discovered N sandboxes later: a sweep parameter nothing
reads, replicates that never receive their seed, a matrix that explodes, or a
secret written into a file that gets committed.

Validation is local and free. Importing this module needs neither torch nor the
sandbox SDK.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from stateset_agents.core.errors import ConfigurationError

__all__ = [
    "FanoutManifest",
    "GpuRequest",
    "ManifestError",
    "PackageSpec",
    "ReportSpec",
    "Resources",
    "RunSpec",
]

SCHEMA_VERSION = 1

#: A fan-out larger than this is almost certainly a matrix typo. The controller's
#: largest default plan allows 200 concurrent sandboxes.
MAX_RUNS = 1000

#: The controller's JSON body limit is 10 MiB and file content is base64 encoded,
#: so one uploaded file must stay under 7.5 MiB. Round down for the envelope.
MAX_STAGE_FILE_BYTES = 7 * 1024 * 1024

MIN_TIMEOUT_SECONDS = 60
#: How much longer a sandbox lives than its run may take. A sandbox's expiry counts
#: from create and deletes its files, so with equal limits the sandbox would expire
#: first: the run's output gone, and the loss indistinguishable from a preemption.
SANDBOX_LIFETIME_MARGIN_SECONDS = 600
#: Non-durable sandboxes live at most a day, and fan-out does not use durable ones.
MAX_TIMEOUT_SECONDS = 86_400 - SANDBOX_LIFETIME_MARGIN_SECONDS

WORKLOAD_KINDS = ("train", "command", "workflow_regression")
PACKAGE_SOURCES = ("preinstalled", "pypi", "wheel")

#: Placeholders the worker always fills in for a ``command`` workload.
COMMAND_BUILTINS = frozenset({"seed", "output_dir", "stage_dir", "python", "run_id"})
#: A command that omits these would produce fake replicates or nothing to collect.
COMMAND_REQUIRED = ("seed", "output_dir")

_WORKLOAD_KEYS: dict[str, frozenset[str]] = {
    "train": frozenset(
        {
            "kind",
            "agent",
            "environment",
            "reward",
            "training",
            "eval_episodes",
            "save_checkpoint",
        }
    ),
    "command": frozenset({"kind", "argv"}),
    "workflow_regression": frozenset(
        {"kind", "updates", "gradient_accumulation_steps", "device"}
    ),
}
_WORKFLOW_PARAMS = frozenset({"updates", "gradient_accumulation_steps", "device"})
#: ``train()`` arguments that are not ``TrainingConfig`` fields.
_TRAIN_EXTRA_PARAMS = frozenset({"objective", "profile"})
_TRAIN_ENVIRONMENT_KEYS = frozenset({"scenarios", "max_turns"})
#: Ratio levels ``train()``'s multi-turn trainer can feed: its rollouts keep summed
#: log-probs, so token-level objectives are refused at training start. Checked here
#: so a sweep fails for free rather than once per sandbox.
TRAIN_OBJECTIVE_RATIOS = ("sequence", "sequence_token")
#: Reward components a ``train`` workload can name.
TRAIN_REWARD_NAMES = ("conciseness", "engagement", "helpfulness", "safety")

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "name",
        "seeds",
        "matrix",
        "workload",
        "report",
        "resources",
        "package",
        "env",
        "forward_env",
        "stage",
        "collect",
        "concurrency",
        "max_attempts",
    }
)

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SECRET_NAME = re.compile(
    r"TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|CREDENTIAL|PRIVATE_KEY", re.IGNORECASE
)
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ManifestError(ConfigurationError):
    """The manifest is invalid. Raised before any compute is allocated."""

    message_only_str = True


def _only_keys(section: str, data: Mapping[str, Any], allowed: frozenset[str]) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ManifestError(
            f"unknown {section} key(s): {', '.join(unknown)}; "
            f"allowed: {', '.join(sorted(allowed))}"
        )


def _mapping(section: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{section} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ManifestError(f"{section} keys must be strings")
    return dict(value)


def _positive_int(section: str, value: Any, *, maximum: int | None = None) -> int:
    if type(value) is not int or value < 1 or (maximum and value > maximum):
        limit = f" no greater than {maximum}" if maximum else ""
        raise ManifestError(f"{section} must be a positive integer{limit}")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(section: str, value: Any) -> str:
    """Accept only a relative POSIX path that cannot leave its root."""
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{section} paths must be nonempty strings")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ManifestError(f"{section} path must stay inside its root: {value!r}")
    return value


@dataclass(frozen=True)
class GpuRequest:
    """The controller's ``gpu`` create field (see the sandbox repo's docs/GPU.md)."""

    type: str | None = None
    count: int = 1
    memory_gb: int | None = None

    @classmethod
    def from_dict(cls, data: Any) -> GpuRequest:
        values = _mapping("resources.gpu", data)
        _only_keys("resources.gpu", values, frozenset({"type", "count", "memory_gb"}))
        count = _positive_int("resources.gpu.count", values.get("count", 1), maximum=8)
        memory = values.get("memory_gb")
        if memory is not None:
            _positive_int("resources.gpu.memory_gb", memory)
        return cls(type=values.get("type"), count=count, memory_gb=memory)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"count": self.count}
        if self.type is not None:
            data["type"] = self.type
        if self.memory_gb is not None:
            data["memory_gb"] = self.memory_gb
        return data


@dataclass(frozen=True)
class Resources:
    """What one run's sandbox is allowed to use, and spend."""

    cpus: str = "2"
    memory: str = "4Gi"
    timeout_seconds: int = 3600
    resource_class: str | None = None
    region: str | None = None
    image: str | None = None
    ephemeral_storage: str | None = None
    #: Server-enforced ceiling for one sandbox's compute and LLM spend. Enforcement
    #: is periodic, so a run can overshoot by up to one check interval. Network
    #: transfer is billed separately and is not covered.
    max_cost_cents: int | None = None
    gpu: GpuRequest | None = None

    @classmethod
    def from_dict(cls, data: Any) -> Resources:
        values = _mapping("resources", data)
        _only_keys(
            "resources",
            values,
            frozenset(
                {
                    "cpus",
                    "memory",
                    "timeout_seconds",
                    "resource_class",
                    "region",
                    "image",
                    "ephemeral_storage",
                    "max_cost_cents",
                    "gpu",
                }
            ),
        )
        timeout = values.get("timeout_seconds", 3600)
        if (
            type(timeout) is not int
            or not MIN_TIMEOUT_SECONDS <= timeout <= MAX_TIMEOUT_SECONDS
        ):
            raise ManifestError(
                f"resources.timeout_seconds must be an integer in "
                f"[{MIN_TIMEOUT_SECONDS}, {MAX_TIMEOUT_SECONDS}]"
            )
        resource_class = values.get("resource_class")
        if resource_class not in (None, "shared", "dedicated"):
            raise ManifestError("resources.resource_class must be shared or dedicated")
        gpu = GpuRequest.from_dict(values["gpu"]) if values.get("gpu") else None
        if gpu is not None and resource_class == "shared":
            raise ManifestError(
                "resources.resource_class shared cannot be combined with a gpu; "
                "the controller refuses it"
            )
        max_cost = values.get("max_cost_cents")
        if max_cost is not None:
            _positive_int("resources.max_cost_cents", max_cost)
        return cls(
            cpus=str(values.get("cpus", "2")),
            memory=str(values.get("memory", "4Gi")),
            timeout_seconds=timeout,
            resource_class=resource_class,
            region=values.get("region"),
            image=values.get("image"),
            ephemeral_storage=values.get("ephemeral_storage"),
            max_cost_cents=max_cost,
            gpu=gpu,
        )

    @property
    def sandbox_lifetime_seconds(self) -> int:
        """Long enough to finish, collect, and stop after the run hits its limit."""
        return self.timeout_seconds + SANDBOX_LIFETIME_MARGIN_SECONDS

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "cpus": self.cpus,
            "memory": self.memory,
            "timeout_seconds": self.timeout_seconds,
        }
        for name in (
            "resource_class",
            "region",
            "image",
            "ephemeral_storage",
            "max_cost_cents",
        ):
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        if self.gpu is not None:
            data["gpu"] = self.gpu.to_dict()
        return data


@dataclass(frozen=True)
class PackageSpec:
    """How ``stateset_agents`` itself gets into the sandbox."""

    source: str = "preinstalled"
    version: str | None = None
    wheel: str | None = None
    extras: tuple[str, ...] = ()
    #: False installs only the package. ``command`` workloads on an image with
    #: no route to PyPI need nothing else: the worker imports only the stdlib.
    install_deps: bool = True

    @classmethod
    def from_dict(cls, data: Any, *, base_dir: Path) -> PackageSpec:
        values = _mapping("package", data)
        _only_keys(
            "package",
            values,
            frozenset({"source", "version", "wheel", "extras", "install_deps"}),
        )
        source = values.get("source", "preinstalled")
        if source not in PACKAGE_SOURCES:
            raise ManifestError(
                f"package.source must be one of {', '.join(PACKAGE_SOURCES)}"
            )
        version = values.get("version")
        if source == "pypi" and not (isinstance(version, str) and version.strip()):
            raise ManifestError(
                "package.source pypi needs a pinned package.version: a floating "
                "install would not be reproducible across runs"
            )
        wheel = values.get("wheel")
        if source == "wheel":
            if not isinstance(wheel, str) or not wheel.strip():
                raise ManifestError("package.source wheel needs package.wheel")
            wheel_path = base_dir / wheel
            if not wheel_path.is_file():
                raise ManifestError(f"package.wheel does not exist: {wheel_path}")
            if wheel_path.stat().st_size > MAX_STAGE_FILE_BYTES:
                raise ManifestError(
                    f"package.wheel exceeds the {MAX_STAGE_FILE_BYTES} byte "
                    f"upload limit: {wheel_path}"
                )
        extras = values.get("extras", [])
        if not isinstance(extras, list) or any(
            not isinstance(e, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", e)
            for e in extras
        ):
            raise ManifestError("package.extras must be a list of extra names")
        install_deps = values.get("install_deps", True)
        if type(install_deps) is not bool:
            raise ManifestError("package.install_deps must be a boolean")
        return cls(
            source=source,
            version=version,
            wheel=wheel,
            extras=tuple(extras),
            install_deps=install_deps,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "source": self.source,
            "extras": list(self.extras),
            "install_deps": self.install_deps,
        }
        if self.version is not None:
            data["version"] = self.version
        if self.wheel is not None:
            data["wheel"] = self.wheel
        return data


@dataclass(frozen=True)
class ReportSpec:
    """Which metric ranks the arms, and in which direction."""

    metric: str
    mode: str

    @classmethod
    def from_dict(cls, data: Any) -> ReportSpec:
        values = _mapping("report", data)
        _only_keys("report", values, frozenset({"metric", "mode"}))
        metric = values.get("metric")
        if not isinstance(metric, str) or not metric.strip():
            raise ManifestError("report.metric must be a nonempty string")
        mode = values.get("mode")
        if mode not in ("max", "min"):
            raise ManifestError("report.mode must be max or min")
        return cls(metric=metric, mode=mode)

    def to_dict(self) -> dict[str, Any]:
        return {"metric": self.metric, "mode": self.mode}


@dataclass(frozen=True)
class RunSpec:
    """One independent run: an arm of the matrix at one seed."""

    run_id: str
    arm_id: str
    seed: int
    params: dict[str, Any]

    def worker_spec(self, manifest: FanoutManifest) -> dict[str, Any]:
        """The JSON document handed to the in-sandbox worker."""
        return {
            "schema_version": SCHEMA_VERSION,
            "name": manifest.name,
            "run_id": self.run_id,
            "arm_id": self.arm_id,
            "seed": self.seed,
            "params": self.params,
            "workload": manifest.workload,
        }


def _validate_seeds(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ManifestError("seeds must be a nonempty list of integers")
    for seed in value:
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ManifestError(f"seeds must be integers in [0, 2**32): {seed!r}")
    if len(set(value)) != len(value):
        raise ManifestError("seeds must be unique; a repeated seed is not a replicate")
    return tuple(value)


def _validate_matrix(value: Any) -> dict[str, list[Any]]:
    matrix = _mapping("matrix", value)
    for key, values in matrix.items():
        if not isinstance(values, list) or not values:
            raise ManifestError(f"matrix.{key} must be a nonempty list")
        for item in values:
            finite_number = isinstance(item, int) or (
                isinstance(item, float) and math.isfinite(item)
            )
            if not (isinstance(item, str) or finite_number):
                raise ManifestError(
                    f"matrix.{key} values must be finite scalars: {item!r}"
                )
        if len({_canonical_json(item) for item in values}) != len(values):
            raise ManifestError(f"matrix.{key} values must be unique")
    return matrix


def _training_field_names() -> frozenset[str]:
    from dataclasses import fields

    from stateset_agents.training.config import TrainingConfig

    return frozenset(f.name for f in fields(TrainingConfig))


def _validate_train(workload: dict[str, Any], matrix: dict[str, list[Any]]) -> None:
    agent = _mapping("workload.agent", workload.get("agent"))
    if not isinstance(agent.get("model_name"), str) or not agent["model_name"].strip():
        raise ManifestError("workload.agent.model_name must be a nonempty string")
    environment = _mapping("workload.environment", workload.get("environment", {}))
    _only_keys("workload.environment", environment, _TRAIN_ENVIRONMENT_KEYS)
    reward = _mapping("workload.reward", workload.get("reward", {}))
    unknown_rewards = sorted(set(reward) - set(TRAIN_REWARD_NAMES))
    if unknown_rewards:
        raise ManifestError(
            f"unknown reward(s): {', '.join(unknown_rewards)}; "
            f"available: {', '.join(TRAIN_REWARD_NAMES)}"
        )
    training = _mapping("workload.training", workload.get("training", {}))
    allowed = _training_field_names() | _TRAIN_EXTRA_PARAMS
    unknown = sorted((set(matrix) | set(training)) - allowed)
    if unknown:
        raise ManifestError(
            f"train parameter(s) are not TrainingConfig fields: {', '.join(unknown)}"
        )
    if "seed" in matrix or "seed" in training:
        raise ManifestError("set replicates through seeds, not a training parameter")
    objectives = list(matrix.get("objective", []))
    if "objective" in training:
        objectives.append(training["objective"])
    if objectives:
        from stateset_agents.training.objectives import compatible_presets

        usable = compatible_presets(TRAIN_OBJECTIVE_RATIOS)
        for objective in objectives:
            if objective not in usable:
                raise ManifestError(
                    f"train workloads cannot use objective {objective!r}; "
                    f"usable here: {', '.join(usable)}. Reach other objectives "
                    "through a command workload that runs their trainer."
                )


def _validate_command(workload: dict[str, Any], matrix: dict[str, list[Any]]) -> None:
    argv = workload.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(arg, str) or not arg for arg in argv)
    ):
        raise ManifestError("workload.argv must be a nonempty list of nonempty strings")
    shadowing = sorted(set(matrix) & COMMAND_BUILTINS)
    if shadowing:
        raise ManifestError(
            f"matrix key(s) shadow built-in placeholders, so every arm would run "
            f"the same command: {', '.join(shadowing)}"
        )
    used = {name for arg in argv for name in _PLACEHOLDER.findall(arg)}
    for required in COMMAND_REQUIRED:
        if required not in used:
            raise ManifestError(
                f"workload.argv must use {{{required}}}: without it the runs "
                "are not independent replicates with collectable results"
            )
    unknown = sorted(used - COMMAND_BUILTINS - set(matrix))
    if unknown:
        raise ManifestError(
            f"workload.argv uses undefined placeholder(s): {', '.join(unknown)}"
        )
    unused = sorted(set(matrix) - used)
    if unused:
        raise ManifestError(
            f"matrix key(s) never reach the command, so the sweep would run "
            f"identical arms: {', '.join(unused)}"
        )


def _validate_workload(value: Any, matrix: dict[str, list[Any]]) -> dict[str, Any]:
    workload = _mapping("workload", value)
    kind = workload.get("kind")
    if kind not in WORKLOAD_KINDS:
        raise ManifestError(f"workload.kind must be one of {', '.join(WORKLOAD_KINDS)}")
    _only_keys(f"{kind} workload", workload, _WORKLOAD_KEYS[kind])
    if kind == "train":
        _validate_train(workload, matrix)
    elif kind == "command":
        _validate_command(workload, matrix)
    else:
        unknown = sorted(set(matrix) - _WORKFLOW_PARAMS)
        if unknown:
            raise ManifestError(
                f"workflow_regression has no parameter(s): {', '.join(unknown)}; "
                f"allowed: {', '.join(sorted(_WORKFLOW_PARAMS))}"
            )
    return workload


def _validate_env(value: Any) -> dict[str, str]:
    env = _mapping("env", value)
    for name, item in env.items():
        if not _ENV_NAME.match(name):
            raise ManifestError(f"env name is invalid: {name!r}")
        if _SECRET_NAME.search(name):
            raise ManifestError(
                f"env.{name} looks like a secret and manifests get committed; "
                f"list it under forward_env or use an org secret"
            )
        if not isinstance(item, str):
            raise ManifestError(f"env.{name} must be a string")
    return env


def _validate_forward_env(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ManifestError("forward_env must be a list of variable names")
    for name in value:
        if not isinstance(name, str) or not _ENV_NAME.match(name):
            raise ManifestError(f"forward_env name is invalid: {name!r}")
        if name.upper().startswith("STATESET_") and _SECRET_NAME.search(name):
            raise ManifestError(
                f"forward_env must not include {name}: a workload never needs "
                "the credential that creates sandboxes"
            )
    return tuple(value)


def _validate_stage(value: Any, *, base_dir: Path) -> dict[str, str]:
    stage = _mapping("stage", value)
    for destination, source in stage.items():
        _relative_path("stage", destination)
        if not isinstance(source, str) or not source.strip():
            raise ManifestError(f"stage source for {destination!r} must be a path")
        path = base_dir / source
        if not path.is_file():
            raise ManifestError(f"stage source does not exist: {path}")
        if path.stat().st_size > MAX_STAGE_FILE_BYTES:
            raise ManifestError(
                f"stage source exceeds the {MAX_STAGE_FILE_BYTES} byte upload "
                f"limit: {path}; put large datasets behind a URL the run fetches"
            )
    return stage


def _validate_collect(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ManifestError("collect must be a list of output-relative patterns")
    return tuple(_relative_path("collect", pattern) for pattern in value)


@dataclass(frozen=True)
class FanoutManifest:
    """A validated fan-out. Build with :meth:`from_file` or :meth:`from_dict`."""

    name: str
    seeds: tuple[int, ...]
    workload: dict[str, Any]
    report: ReportSpec
    matrix: dict[str, list[Any]] = field(default_factory=dict)
    resources: Resources = field(default_factory=Resources)
    package: PackageSpec = field(default_factory=PackageSpec)
    env: dict[str, str] = field(default_factory=dict)
    forward_env: tuple[str, ...] = ()
    stage: dict[str, str] = field(default_factory=dict)
    collect: tuple[str, ...] = ()
    concurrency: int = 4
    max_attempts: int = 2
    #: Where relative ``stage`` and ``package.wheel`` paths resolve from.
    base_dir: Path = Path(".")

    @classmethod
    def from_file(cls, path: str | Path) -> FanoutManifest:
        target = Path(path)
        try:
            data = yaml.safe_load(target.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ManifestError(f"cannot read manifest {target}: {exc}") from exc
        return cls.from_dict(data, base_dir=target.resolve().parent)

    @classmethod
    def from_dict(
        cls, data: Any, *, base_dir: str | Path | None = None
    ) -> FanoutManifest:
        values = _mapping("manifest", data)
        _only_keys("manifest", values, _TOP_LEVEL_KEYS)
        root = Path(base_dir) if base_dir is not None else Path(".")
        if values.get("schema_version") != SCHEMA_VERSION:
            raise ManifestError(f"schema_version must be {SCHEMA_VERSION}")
        name = values.get("name")
        if not isinstance(name, str) or not _NAME.match(name):
            raise ManifestError(
                "name must be 1-64 letters, digits, dots, underscores or dashes"
            )
        seeds = _validate_seeds(values.get("seeds"))
        matrix = _validate_matrix(values.get("matrix", {}))
        arms = math.prod(len(items) for items in matrix.values())
        if arms * len(seeds) > MAX_RUNS:
            raise ManifestError(
                f"this manifest expands to {arms * len(seeds)} runs, above the "
                f"limit of {MAX_RUNS}; check the matrix for a mistake"
            )
        return cls(
            name=name,
            seeds=seeds,
            matrix=matrix,
            workload=_validate_workload(values.get("workload"), matrix),
            report=ReportSpec.from_dict(values.get("report")),
            resources=Resources.from_dict(values.get("resources", {})),
            package=PackageSpec.from_dict(values.get("package", {}), base_dir=root),
            env=_validate_env(values.get("env", {})),
            forward_env=_validate_forward_env(values.get("forward_env", [])),
            stage=_validate_stage(values.get("stage", {}), base_dir=root),
            collect=_validate_collect(values.get("collect", [])),
            concurrency=_positive_int("concurrency", values.get("concurrency", 4)),
            max_attempts=_positive_int(
                "max_attempts", values.get("max_attempts", 2), maximum=10
            ),
            base_dir=root,
        )

    def expand(self) -> list[RunSpec]:
        """Every run, in a stable order that does not depend on key order."""
        keys = sorted(self.matrix)
        runs: list[RunSpec] = []
        for values in itertools.product(*(self.matrix[key] for key in keys)):
            params = dict(zip(keys, values, strict=True))
            arm_id = (
                "a" + hashlib.sha256(_canonical_json(params).encode()).hexdigest()[:10]
                if params
                else "base"
            )
            runs.extend(
                RunSpec(
                    run_id=f"{arm_id}-seed{seed}",
                    arm_id=arm_id,
                    seed=seed,
                    params=params,
                )
                for seed in self.seeds
            )
        return runs

    def worst_case_cost_usd(self) -> float | None:
        """Every run retried to its limit at its ceiling. None when unbounded.

        Computed from the server-enforced per-sandbox budget rather than a price
        table, so it tracks what the platform measures. That budget covers compute
        and LLM spend; network transfer past the plan's allowance is billed on top.
        """
        if self.resources.max_cost_cents is None:
            return None
        return (
            len(self.expand())
            * self.max_attempts
            * self.resources.max_cost_cents
            / 100.0
        )

    def stage_files(self) -> dict[str, Path]:
        """Sandbox-relative destination to resolved local source."""
        return {dest: self.base_dir / source for dest, source in self.stage.items()}

    def wheel_path(self) -> Path | None:
        return self.base_dir / self.package.wheel if self.package.wheel else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "seeds": list(self.seeds),
            "matrix": self.matrix,
            "workload": self.workload,
            "report": self.report.to_dict(),
            "resources": self.resources.to_dict(),
            "package": self.package.to_dict(),
            "env": self.env,
            "forward_env": list(self.forward_env),
            "stage": self.stage,
            "collect": list(self.collect),
            "concurrency": self.concurrency,
            "max_attempts": self.max_attempts,
        }

    def digest(self) -> str:
        """Identity of what the runs do; a resume must match it exactly.

        Concurrency is excluded because changing it does not change any run.
        Staged and shipped bytes are included: same manifest, different data
        would silently mix two experiments in one report.
        """
        payload = self.to_dict()
        payload.pop("concurrency")
        payload["stage_sha256"] = {
            dest: _sha256_file(path) for dest, path in self.stage_files().items()
        }
        wheel = self.wheel_path()
        if wheel is not None:
            payload["wheel_sha256"] = _sha256_file(wheel)
        return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
