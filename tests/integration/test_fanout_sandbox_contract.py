"""Pin the adapter and its fake controller to the real controller's API contract.

The fake controller is only as useful as it is accurate. These tests compare what
the real SDK actually sends through the fan-out against ``controller/openapi.yaml``
in the sandbox repo, so a route or field that does not exist there fails here,
rather than on the first paid run.

Needs a checkout of the sandbox repo: a sibling directory, or
``STATESET_SANDBOX_REPO``. Skips without one.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytest.importorskip("stateset_sandbox")

from stateset_agents.fanout.manifest import FanoutManifest  # noqa: E402
from stateset_agents.fanout.orchestrator import FanoutRunner  # noqa: E402
from stateset_agents.fanout.sandbox_backend import (  # noqa: E402
    BLOCKED_ENV_NAMES,
    BLOCKED_ENV_PREFIXES,
    SandboxBackend,
)
from tests.integration.fake_sandbox_controller import (  # noqa: E402
    FakeController,
    _Handler,
)

pytestmark = pytest.mark.integration

OPENAPI_RELATIVE = Path("controller") / "openapi.yaml"


def _find_openapi() -> Path | None:
    configured = os.environ.get("STATESET_SANDBOX_REPO")
    if configured:
        candidate = Path(configured) / OPENAPI_RELATIVE
        return candidate if candidate.is_file() else None
    for parent in Path(__file__).resolve().parents:
        candidate = parent.parent / "stateset-sandbox" / OPENAPI_RELATIVE
        if candidate.is_file():
            return candidate
    return None


@pytest.fixture(scope="module")
def openapi() -> dict[str, Any]:
    path = _find_openapi()
    if path is None:
        pytest.skip("no sandbox repo checkout: set STATESET_SANDBOX_REPO")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _dereference(document: dict[str, Any], node: Any) -> Any:
    while isinstance(node, dict) and "$ref" in node:
        current: Any = document
        for part in node["$ref"].lstrip("#/").split("/"):
            current = current[part]
        node = current
    return node


def _route_exists(document: dict[str, Any], method: str, path: str) -> bool:
    """Whether a concrete request path matches a path template in the contract."""
    for template, operations in document["paths"].items():
        pattern = "^" + re.sub(r"\{[^}]+\}", "[^/]+", template) + "$"
        if re.match(pattern, path) and method.lower() in operations:
            return True
    return False


def _request_schema(
    document: dict[str, Any], template: str, method: str
) -> dict[str, Any]:
    operation = document["paths"][template][method]
    schema = operation["requestBody"]["content"]["application/json"]["schema"]
    return _dereference(document, schema)


@pytest.fixture(scope="module")
def recorded(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """A real fan-out through the real SDK, touching every route it can use."""
    tmp_path = tmp_path_factory.mktemp("contract")
    script = (
        "import json, pathlib, sys; "
        "print('hello'); "
        "pathlib.Path(sys.argv[2], 'metrics.json').write_text(json.dumps({'m': 1.0})); "
        "pathlib.Path(sys.argv[2], 'artifact.txt').write_text('x')"
    )
    (tmp_path / "data.txt").write_text("staged")
    manifest = FanoutManifest.from_dict(
        {
            "schema_version": 1,
            "name": "contract",
            "seeds": [1],
            "workload": {
                "kind": "command",
                "argv": ["{python}", "-c", script, "{seed}", "{output_dir}"],
            },
            "report": {"metric": "m", "mode": "max"},
            "resources": {
                "cpus": "2",
                "memory": "4Gi",
                "timeout_seconds": 300,
                "region": "us-central1",
                "image": "org:stateset-rl",
                "ephemeral_storage": "10Gi",
                "max_cost_cents": 5,
                "gpu": {"type": "nvidia-l4", "count": 1, "memory_gb": 24},
            },
            "stage": {"data.txt": "data.txt"},
            "collect": ["*.txt"],
            "env": {"LOG_LEVEL": "debug"},
        },
        base_dir=tmp_path,
    )
    with FakeController(tmp_path / "cluster") as controller:
        summary = FanoutRunner(
            manifest,
            SandboxBackend(
                api_key=controller.api_key,
                base_url=controller.base_url,
                requests_per_minute=600_000,
            ),
            tmp_path / "run",
            poll_interval_s=0.02,
            forwarded_env={},
            ledger_path=tmp_path / "ledger.jsonl",
        ).run()
        assert summary.ok, summary
        yield controller


def test_every_request_the_fan_out_made_is_a_real_route(
    openapi: dict[str, Any], recorded: FakeController
) -> None:
    assert recorded.requests
    unknown = sorted(
        {
            f"{method} {path}"
            for method, path in recorded.requests
            if not _route_exists(openapi, method, path)
        }
    )
    assert unknown == []


def test_every_fake_controller_route_is_a_real_route(openapi: dict[str, Any]) -> None:
    for method, pattern, _ in _Handler.ROUTES:
        concrete = pattern.pattern.strip("^$").replace("([^/]+)", "x")
        assert _route_exists(openapi, method, concrete), f"{method} {concrete}"


def test_create_only_sends_fields_the_controller_accepts(
    openapi: dict[str, Any], recorded: FakeController
) -> None:
    schema = _request_schema(openapi, "/api/v1/sandbox/create", "post")
    properties = schema["properties"]
    body = recorded.create_bodies[0]

    assert set(body) <= set(properties)
    budget = _dereference(openapi, properties["budget"])
    assert set(body["budget"]) <= set(budget["properties"])
    gpu = _dereference(openapi, properties["gpu"])
    assert set(body["gpu"]) <= set(gpu["properties"])


def test_process_start_only_sends_fields_the_controller_accepts(
    openapi: dict[str, Any], recorded: FakeController
) -> None:
    schema = _request_schema(openapi, "/api/v1/sandbox/{id}/processes", "post")
    start = recorded.process_starts[0]
    fields = {key for key in start if key != "sandbox_id"}

    assert fields <= set(schema["properties"])


def test_fake_create_response_uses_the_real_field_names(
    openapi: dict[str, Any], recorded: FakeController
) -> None:
    operation = openapi["paths"]["/api/v1/sandbox/create"]["post"]
    response = _dereference(openapi, operation["responses"]["201"])
    schema = _dereference(openapi, response["content"]["application/json"]["schema"])
    sandbox = next(iter(recorded.sandboxes.values()))

    fake_fields = set(recorded._sandbox_json(sandbox))
    # cpus and memory are echoed by the SDK's model rather than this schema.
    assert fake_fields - {"cpus", "memory"} <= set(schema["properties"])


def test_blocked_env_prefixes_match_the_controller() -> None:
    path = _find_openapi()
    if path is None:
        pytest.skip("no sandbox repo checkout: set STATESET_SANDBOX_REPO")
    source = path.parent / "src" / "services" / "sandbox-env.service.ts"
    match = re.search(r"BLOCKED_ENV_PREFIXES = \[([^\]]+)\]", source.read_text())
    assert match is not None
    controller_prefixes = set(re.findall(r"'([^']+)'", match.group(1)))

    assert controller_prefixes == set(BLOCKED_ENV_PREFIXES)
    exact = re.search(
        r"BLOCKED_ENV_EXACT = new Set\(\[(.*?)\]\)", source.read_text(), re.S
    )
    assert exact is not None
    controller_names = set(re.findall(r"^\s*'([^']+)'", exact.group(1), re.M))

    assert controller_names == set(BLOCKED_ENV_NAMES)


def test_controller_still_refuses_control_characters_in_commands() -> None:
    """The reason the boot script is one line. If this relaxes, the fake should too."""
    path = _find_openapi()
    if path is None:
        pytest.skip("no sandbox repo checkout: set STATESET_SANDBOX_REPO")
    schemas = (path.parent / "src" / "schemas" / "index.ts").read_text()

    assert "CommandArgSchema" in schemas
    assert r"/[\0\r\t\n]/" in schemas
