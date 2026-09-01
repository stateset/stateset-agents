"""Tests for the stable v1 Python and HTTP compatibility contract."""

from __future__ import annotations

import copy
import json

from scripts.check_api_compatibility import (
    DEFAULT_MANIFEST,
    STABLE_HTTP_OPERATIONS,
    build_contract,
    compare_contracts,
    load_manifest,
)


def test_committed_contract_matches_current_public_surface() -> None:
    expected = load_manifest(DEFAULT_MANIFEST)
    actual = build_contract()
    assert compare_contracts(expected, actual) == []


def test_contract_is_deterministic_and_json_serializable() -> None:
    first = build_contract()
    second = build_contract()
    assert first == second
    serialized = json.dumps(first, sort_keys=True)
    assert json.loads(serialized) == first
    assert '"description"' not in serialized
    assert '"x-code-samples"' not in serialized


def test_contract_covers_declared_python_and_http_surfaces() -> None:
    contract = build_contract()
    assert len(contract["python"]["stateset_agents"]) >= 90
    assert set(contract["python"]["stateset_agents.api"]) == {
        "APIConfig",
        "ConversationRequest",
        "ConversationResponse",
        "HealthResponse",
        "TrainingRequest",
        "TrainingResponse",
        "attach_distributed_rollout_control_plane",
        "create_app",
        "get_config",
    }
    assert set(contract["http"]["operations"]) == set(STABLE_HTTP_OPERATIONS)
    assert contract["http"]["schemas"]
    assert contract["python"]["stateset_agents"]["Agent"]["signature"]
    assert "signature" not in contract["python"]["stateset_agents"]["DeviceType"]
    agent_parameters = contract["python"]["stateset_agents"]["Agent"]["signature"][
        "parameters"
    ]
    assert agent_parameters[0] == {
        "default": None,
        "kind": "POSITIONAL_OR_KEYWORD",
        "name": "config",
    }
    assert all("annotation" not in parameter for parameter in agent_parameters)


def test_compare_contracts_reports_removed_and_changed_surface() -> None:
    expected = build_contract()
    actual = copy.deepcopy(expected)
    del actual["python"]["stateset_agents"]["Agent"]
    actual["http"]["operations"]["GET /healthz"]["operationId"] = "changed"

    differences = compare_contracts(expected, actual)
    assert "$.python.stateset_agents.Agent: removed" in differences
    assert any("GET /healthz" in item and "operationId" in item for item in differences)


def test_compare_contracts_reports_unreviewed_additions() -> None:
    expected = {"exports": {"Agent": {"module": "stateset_agents"}}}
    actual = copy.deepcopy(expected)
    actual["exports"]["NewAgent"] = {"module": "stateset_agents"}
    assert compare_contracts(expected, actual) == ["$.exports.NewAgent: added"]
