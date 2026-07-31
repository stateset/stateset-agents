"""A behavioural stand-in for the ``modal`` SDK.

This is not a mock. It *executes* the function it is given and *stores* the
files that function writes, so tests can assert on real effects — an adapter
directory actually arriving on local disk — rather than on recorded call
kwargs. Asserting on recorded kwargs is what let an earlier version of
``ModalExecutor`` report SUCCEEDED without running anything.

It models only the surface ``ModalExecutor`` uses, per Modal's reference docs:
``Image.debian_slim().pip_install()``, ``Volume.from_name(create_if_missing=)``
with ``reload``/``iterdir``/``read_file``, ``App.function(...)`` as a
decorator, ``app.run()`` as a context manager, and ``Function.remote()``.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Callable, Iterator


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

    def __init__(self, name: str, root: Path) -> None:
        self.name = name
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.reload_count = 0

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

    def reload(self) -> None:
        self.reload_count += 1

    def commit(self) -> None:
        pass

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
            volume.root = Path(mount)
            volume.root.mkdir(parents=True, exist_ok=True)
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
    module.Volume = types.SimpleNamespace(from_name=FakeVolume.bind(volume_root))

    apps: list[FakeApp] = []

    def make_app(name: str = "", **kwargs: Any) -> FakeApp:
        app = FakeApp(name=name)
        apps.append(app)
        return app

    module.App = make_app
    module.apps = apps
    FakeVolume.reset()
    return module
