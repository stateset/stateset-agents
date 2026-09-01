#!/usr/bin/env python3
"""Build and verify StateSet Agents' stable v1 public API contract."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "contracts" / "public_api_v1.json"

STABLE_HTTP_OPERATIONS = (
    "GET /api/v1/health",
    "POST /api/v1/training",
    "GET /api/v1/training/{job_id}",
    "DELETE /api/v1/training/{job_id}",
    "POST /api/v1/conversations",
    "GET /api/v1/conversations/{conversation_id}",
    "DELETE /api/v1/conversations/{conversation_id}",
    "GET /api/v1/rollouts/stats",
    "GET /api/v1/rollouts/workers",
    "POST /api/v1/rollouts/workers/{worker_id}/register",
    "POST /api/v1/rollouts/workers/{worker_id}/heartbeat",
    "POST /api/v1/rollouts/workers/{worker_id}/submit",
    "DELETE /api/v1/rollouts/workers/{worker_id}",
    "POST /v1/chat/completions",
    "POST /v1/messages",
    "GET /v1/models",
    "GET /healthz",
    "GET /ready",
    "GET /live",
)

_HTTP_METHODS = {"get", "post", "put", "patch", "delete"}
_DOCUMENTATION_KEYS = {
    "description",
    "example",
    "examples",
    "externalDocs",
    "summary",
    "tags",
    "title",
    "x-codeSamples",
    "x-code-samples",
}


class ApiCompatibilityError(RuntimeError):
    """Raised when the checked contract cannot be built or verified."""


def _wire_shape(value: Any) -> Any:
    """Remove prose-only OpenAPI fields while retaining wire constraints."""
    if isinstance(value, dict):
        return {
            key: _wire_shape(item)
            for key, item in sorted(value.items())
            if key not in _DOCUMENTATION_KEYS
        }
    if isinstance(value, list):
        return [_wire_shape(item) for item in value]
    return value


def _collect_schema_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, dict):
        ref = value.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
            refs.add(ref.rsplit("/", 1)[-1])
        for item in value.values():
            refs.update(_collect_schema_refs(item))
    elif isinstance(value, list):
        for item in value:
            refs.update(_collect_schema_refs(item))
    return refs


def _stable_default(value: Any) -> Any:
    """Return a deterministic JSON-compatible callable default."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        enum_type = type(value)
        return {
            "enum": f"{enum_type.__module__}.{enum_type.__qualname__}",
            "member": value.name,
        }
    if isinstance(value, tuple):
        return {"tuple": [_stable_default(item) for item in value]}
    return {"repr": repr(value)}


def _call_signature(value: Any) -> dict[str, Any] | None:
    """Describe call shape without interpreter-specific annotation rendering."""
    if inspect.isclass(value) and issubclass(value, Enum):
        return None
    try:
        signature = inspect.signature(value, eval_str=False)
    except (TypeError, ValueError):
        return None

    parameters: list[dict[str, Any]] = []
    for parameter in signature.parameters.values():
        item: dict[str, Any] = {
            "kind": parameter.kind.name,
            "name": parameter.name,
        }
        if parameter.default is not inspect.Parameter.empty:
            item["default"] = _stable_default(parameter.default)
        parameters.append(item)
    return {"parameters": parameters}


def _python_contract() -> dict[str, dict[str, dict[str, Any]]]:
    import stateset_agents
    import stateset_agents.api as api

    lazy_exports = stateset_agents._LAZY_EXPORTS  # type: ignore[attr-defined]
    declared_exports = set(stateset_agents.__all__)
    mapped_exports = set(lazy_exports)
    if declared_exports != mapped_exports:
        missing = sorted(declared_exports - mapped_exports)
        extra = sorted(mapped_exports - declared_exports)
        raise ApiCompatibilityError(
            "stateset_agents.__all__ and _LAZY_EXPORTS differ: "
            f"unmapped={missing}, undeclared={extra}"
        )

    root_exports: dict[str, dict[str, Any]] = {}
    for name, (module_name, attr_name, _hint) in sorted(lazy_exports.items()):
        entry = {"module": module_name, "name": attr_name}
        value = getattr(stateset_agents, name)
        if callable(value):
            signature = _call_signature(value)
            if signature is not None:
                entry["signature"] = signature
        root_exports[name] = entry
    api_exports: dict[str, dict[str, Any]] = {}
    for name in sorted(api.__all__):
        value = getattr(api, name)
        api_exports[name] = {
            "module": str(getattr(value, "__module__", api.__name__)),
            "name": str(getattr(value, "__qualname__", name)),
        }
        if callable(value):
            signature = _call_signature(value)
            if signature is not None:
                api_exports[name]["signature"] = signature
    return {
        "stateset_agents": root_exports,
        "stateset_agents.api": api_exports,
    }


def _http_contract() -> dict[str, Any]:
    os.environ.setdefault("API_ENABLE_TRAINING_LAB", "false")
    os.environ.setdefault("API_JWT_SECRET", "compatibility-contract-only")
    os.environ.setdefault("API_REQUIRE_AUTH", "false")

    from stateset_agents.api.main import create_app

    schema = create_app().openapi()
    operations: dict[str, Any] = {}
    for path, path_item in schema.get("paths", {}).items():
        if not isinstance(path_item, dict):
            continue
        for method, operation in path_item.items():
            if method.lower() not in _HTTP_METHODS or not isinstance(operation, dict):
                continue
            key = f"{method.upper()} {path}"
            if key in STABLE_HTTP_OPERATIONS:
                operations[key] = _wire_shape(operation)

    missing = sorted(set(STABLE_HTTP_OPERATIONS) - set(operations))
    if missing:
        raise ApiCompatibilityError(
            "stable HTTP operations missing from OpenAPI: " + ", ".join(missing)
        )

    all_schemas = schema.get("components", {}).get("schemas", {})
    referenced = _collect_schema_refs(operations)
    selected_schemas: dict[str, Any] = {}
    pending = sorted(referenced)
    while pending:
        name = pending.pop(0)
        if name in selected_schemas:
            continue
        component = all_schemas.get(name)
        if not isinstance(component, dict):
            raise ApiCompatibilityError(f"referenced OpenAPI schema is missing: {name}")
        normalized = _wire_shape(component)
        selected_schemas[name] = normalized
        pending.extend(
            sorted(
                _collect_schema_refs(normalized) - set(selected_schemas) - set(pending)
            )
        )

    return {
        "operations": dict(sorted(operations.items())),
        "schemas": dict(sorted(selected_schemas.items())),
    }


def build_contract() -> dict[str, Any]:
    """Return the deterministic stable API contract for the current tree."""
    return {
        "contract_version": 1,
        "stability": "v1",
        "python": _python_contract(),
        "http": _http_contract(),
    }


def compare_contracts(expected: Any, actual: Any, path: str = "$") -> list[str]:
    """Return deterministic structural differences between two contracts."""
    if type(expected) is not type(actual):
        return [
            f"{path}: type changed from {type(expected).__name__} "
            f"to {type(actual).__name__}"
        ]
    if isinstance(expected, dict):
        differences: list[str] = []
        expected_keys = set(expected)
        actual_keys = set(actual)
        differences.extend(
            f"{path}.{key}: removed" for key in sorted(expected_keys - actual_keys)
        )
        differences.extend(
            f"{path}.{key}: added" for key in sorted(actual_keys - expected_keys)
        )
        for key in sorted(expected_keys & actual_keys):
            differences.extend(
                compare_contracts(expected[key], actual[key], f"{path}.{key}")
            )
        return differences
    if isinstance(expected, list):
        differences = []
        if len(expected) != len(actual):
            differences.append(
                f"{path}: length changed from {len(expected)} to {len(actual)}"
            )
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual, strict=False)
        ):
            differences.extend(
                compare_contracts(expected_item, actual_item, f"{path}[{index}]")
            )
        return differences
    if expected != actual:
        return [f"{path}: changed from {expected!r} to {actual!r}"]
    return []


def load_manifest(path: Path) -> dict[str, Any]:
    """Load a contract manifest and reject invalid top-level payloads."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ApiCompatibilityError(
            f"contract manifest does not exist: {path}"
        ) from exc
    except (json.JSONDecodeError, OSError) as exc:
        raise ApiCompatibilityError(
            f"contract manifest is unreadable: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ApiCompatibilityError(
            f"contract manifest must contain a JSON object: {path}"
        )
    return payload


def write_manifest(path: Path, contract: dict[str, Any]) -> None:
    """Write a canonical contract manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--write",
        action="store_true",
        help="replace the manifest with the current contract (review the diff)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        current = build_contract()
        if args.write:
            write_manifest(args.manifest, current)
            print(f"Wrote stable API contract: {args.manifest}")
            return 0

        expected = load_manifest(args.manifest)
        differences = compare_contracts(expected, current)
        if differences:
            print("Stable v1 API contract changed without a reviewed manifest update.")
            for difference in differences[:100]:
                print(f"- {difference}")
            if len(differences) > 100:
                print(f"- ... and {len(differences) - 100} more differences")
            return 1
        print("Stable v1 API compatibility check passed.")
        return 0
    except ApiCompatibilityError as exc:
        print(f"Stable v1 API compatibility check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
