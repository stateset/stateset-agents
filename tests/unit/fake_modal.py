"""A behavioural stand-in for the ``modal`` SDK.

This is not a mock. It *executes* the function it is given and *stores* the
files that function writes, so tests can assert on real effects — an adapter
directory actually arriving on local disk — rather than on recorded call
kwargs. Asserting on recorded kwargs is what let an earlier version of
``ModalExecutor`` report SUCCEEDED without running anything.

It models only the surface ``ModalExecutor`` uses, per Modal's reference docs:
``Image.debian_slim().pip_install()``, ``Volume.from_name(create_if_missing=)``
with batched upload/commit/reload/list/read, named Secrets,
``App.function(...)`` as a decorator, ``app.run()`` as a context manager, and
``Function.remote()``.
"""

from __future__ import annotations

import contextlib
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class FakeImage:
    """Chainable image builder that records what was installed."""

    def __init__(self) -> None:
        self.installed: list[str] = []
        self.local_dirs_added: list[Any] = []

    def debian_slim(self, **kwargs: Any) -> FakeImage:
        return self

    def pip_install(self, *packages: str, **kwargs: Any) -> FakeImage:
        self.installed.extend(packages)
        return self

    # Present so tests can assert they are never called — a working-tree sync
    # is the failure mode this design exists to avoid.
    def add_local_dir(self, *args: Any, **kwargs: Any) -> FakeImage:
        self.local_dirs_added.append(args)
        return self

    def add_local_python_source(self, *args: Any, **kwargs: Any) -> FakeImage:
        self.local_dirs_added.append(args)
        return self


@dataclass
class FakeEntry:
    """One file in a volume, mirroring modal's FileEntry."""

    path: str


class FakeVolume:
    """A volume backed by a real directory on local disk."""

    _instances: dict[str, FakeVolume] = {}
    deleted_names: list[str] = []

    def __init__(self, name: str, root: Path) -> None:
        self.name = name
        self.root = root
        self.storage_root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.reload_count = 0
        self.commit_count = 0

    @classmethod
    def bind(cls, root: Path) -> Callable[..., FakeVolume]:
        """Return a ``from_name`` that materialises volumes under ``root``."""

        def from_name(
            name: str, *, create_if_missing: bool = False, **kwargs: Any
        ) -> FakeVolume:
            if name not in cls._instances:
                if not create_if_missing:
                    raise KeyError(f"volume {name!r} does not exist")
                cls._instances[name] = cls(name, root / name)
            return cls._instances[name]

        return from_name

    @classmethod
    def reset(cls) -> None:
        cls._instances.clear()
        cls.deleted_names.clear()

    def reload(self) -> None:
        self.reload_count += 1

    def commit(self) -> None:
        self.commit_count += 1

    @contextlib.contextmanager
    def batch_upload(self) -> Iterator[FakeVolumeUpload]:
        yield FakeVolumeUpload(self)

    def iterdir(self, path: str, *, recursive: bool = True) -> Iterator[FakeEntry]:
        base = self.root / path.lstrip("/")
        if not base.exists():
            return
        for item in sorted(base.rglob("*") if recursive else base.iterdir()):
            if item.is_file():
                yield FakeEntry(path=str(item.relative_to(self.root)))

    def read_file(self, path: str) -> Iterator[bytes]:
        yield (self.root / path).read_bytes()


@dataclass
class FakeVolumeUpload:
    """Write local files into a fake Volume using Modal's upload surface."""

    volume: FakeVolume

    def put_file(self, local_path: str, remote_path: str) -> None:
        target = self.volume.root / remote_path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, target)


@dataclass
class FakeFunction:
    """A registered function that really runs when ``.remote()`` is called."""

    fn: Callable[..., Any]
    kwargs: dict[str, Any]
    app: FakeApp

    def remote(self, *args: Any, **kwargs: Any) -> Any:
        if not self.app.running:
            raise RuntimeError("function called outside of app.run()")
        self.app.calls.append((args, kwargs))
        # A mounted volume *is* the directory the container writes to. Bind
        # the volume's storage to its mount path so a write at the mount is
        # readable through the volume afterwards, as it is on Modal.
        for mount, volume in (self.kwargs.get("volumes") or {}).items():
            mounted = Path(mount)
            if mounted != volume.root:
                mounted.mkdir(parents=True, exist_ok=True)
                shutil.copytree(volume.root, mounted, dirs_exist_ok=True)
                volume.root = mounted
        return self.fn(*args, **kwargs)

    def spawn(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("spawn is not modelled")


@dataclass
class FakeApp:
    """Records registered functions and models the ephemeral run context."""

    name: str = ""
    running: bool = False
    functions: list[FakeFunction] = field(default_factory=list)
    calls: list[Any] = field(default_factory=list)

    def function(self, **kwargs: Any) -> Callable[[Callable[..., Any]], FakeFunction]:
        def decorator(fn: Callable[..., Any]) -> FakeFunction:
            registered = FakeFunction(fn=fn, kwargs=kwargs, app=self)
            self.functions.append(registered)
            return registered

        return decorator

    @contextlib.contextmanager
    def run(self, *args: Any, **kwargs: Any) -> Iterator[FakeApp]:
        self.running = True
        try:
            yield self
        finally:
            self.running = False


def build(volume_root: Path) -> Any:
    """Construct a fake ``modal`` module whose volumes live under ``volume_root``."""
    import types

    module = types.ModuleType("modal")
    module.Image = FakeImage()

    async def delete(name: str, *, allow_missing: bool = False, **kwargs: Any) -> None:
        volume = FakeVolume._instances.pop(name, None)
        if volume is None:
            if allow_missing:
                return
            raise KeyError(name)
        shutil.rmtree(volume.root, ignore_errors=True)
        if volume.storage_root != volume.root:
            shutil.rmtree(volume.storage_root, ignore_errors=True)
        FakeVolume.deleted_names.append(name)

    module.Volume = types.SimpleNamespace(
        from_name=FakeVolume.bind(volume_root),
        objects=types.SimpleNamespace(delete=delete),
    )

    @dataclass(frozen=True)
    class FakeSecret:
        name: str

    module.Secret = types.SimpleNamespace(from_name=lambda name: FakeSecret(name))

    apps: list[FakeApp] = []

    def make_app(name: str = "", **kwargs: Any) -> FakeApp:
        app = FakeApp(name=name)
        apps.append(app)
        return app

    module.App = make_app
    module.apps = apps
    FakeVolume.reset()
    return module
