"""Guard local docker-compose stacks against exposed ports and baked-in secrets."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILES = [
    ROOT / "docker-compose.yml",
    ROOT / "deployment" / "docker" / "docker-compose.yml",
    ROOT / "deployment" / "docker" / "docker-compose.dev.yml",
]
SECRET_VARS = re.compile(r"\$\{(\w*PASSWORD\w*)(:?[-?])?([^}]*)\}")


@pytest.mark.parametrize("path", COMPOSE_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_published_ports_bind_to_loopback(path: Path) -> None:
    compose = yaml.safe_load(path.read_text(encoding="utf-8"))
    for name, service in compose["services"].items():
        for port in service.get("ports", []):
            assert isinstance(port, str), f"{name}: use 'host_ip:host:container'"
            assert port.startswith("127.0.0.1:"), f"{name} publishes {port}"


@pytest.mark.parametrize("path", COMPOSE_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_passwords_are_required_not_defaulted(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for var, operator, _ in SECRET_VARS.findall(text):
        assert operator in ("?", ":?"), f"{var} must not have a default in {path.name}"
    assert not re.search(r"PASSWORD=(?!\$\{)\S", text), "hardcoded password"


def test_redis_requires_password() -> None:
    for path in COMPOSE_FILES[:2]:
        compose = yaml.safe_load(path.read_text(encoding="utf-8"))
        command = compose["services"]["redis"]["command"]
        assert "--requirepass" in command, path.name
