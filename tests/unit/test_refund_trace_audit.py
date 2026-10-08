"""Fresh sandbox replay must expose fabricated step and outcome evidence."""

import asyncio
import copy
import json
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from stateset_agents.cli import app
from stateset_agents.core.environments.refund_policy_environment import (
    refund_policy_benchmark,
)
from stateset_agents.evaluation.agent_runs import evaluation_report
from stateset_agents.evaluation.implementation import package_implementation
from stateset_agents.evaluation.refund_trace import (
    audit_refund_traces,
    replay_refund_trace,
    select_replayed_validation_checkpoint,
)
from tests.unit.river_fakes import traced_refund_outcome


@pytest.fixture
def evidence():
    cases = refund_policy_benchmark("test", 8, 42)
    outcomes = [
        asyncio.run(traced_refund_outcome(row, successful=i % 2 == 0))
        for i, row in enumerate(cases)
    ]
    report = evaluation_report(
        environment="refund-policy-v2",
        base_model="synthetic-fixture",
        seed=42,
        cases={row["order_id"]: row for row in cases},
        settings={"truncation_reward": -1.0},
        checkpoint={"path": "river://fixture"},
        implementation=package_implementation(),
        outcomes=outcomes,
    )
    return report, cases


def test_audit_replays_both_success_and_failure_without_provider_requests(evidence):
    report, cases = evidence
    before = copy.deepcopy(report)
    result = asyncio.run(audit_refund_traces(report, cases))
    assert result["passed"] and len(result["checked"]) == 8
    assert sum(row["success"] for row in result["checked"]) == 4
    assert result["provider_requests"] == 0
    assert "model_identity" in result["unverified"]
    assert report == before


@pytest.mark.parametrize("mutation", ["missing", "action", "reward", "swapped_case"])
def test_selection_requires_replay_of_every_checkpoint_even_a_losing_one(
    evidence, mutation
):
    report, cases = evidence
    values = [
        {
            "step": step,
            "checkpoint": {"path": f"river://validation-{step}"},
            "metrics": {"reward_mean": 0},
            "case_hashes": copy.deepcopy(report["case_hashes"]),
            "outcomes": copy.deepcopy(report["outcomes"]),
        }
        for step in range(2)
    ]
    options = {
        "steps": 1,
        "cases": {row["order_id"]: row for row in cases},
        "environment": report["environment"],
    }
    selected = asyncio.run(select_replayed_validation_checkpoint(values, **options))
    assert selected["step"] == 0
    # Corrupt the tie-losing checkpoint: it must not escape verification.
    outcome = values[1]["outcomes"][0]
    if mutation == "missing":
        del outcome["environment_trace"]
    elif mutation == "action":
        outcome["environment_trace"]["steps"][0]["action"]["content"] = "not an action"
    elif mutation == "swapped_case":
        outcome["environment_trace"] = copy.deepcopy(
            values[1]["outcomes"][1]["environment_trace"]
        )
    else:
        outcome["reward"] -= 2
        values[1]["metrics"]["reward_mean"] = -0.25
    with pytest.raises(ValueError, match="Validation replay failed at step 1"):
        asyncio.run(select_replayed_validation_checkpoint(values, **options))


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "prompt",
        "observation",
        "step_reward",
        "terminal",
        "extra_action",
        "success",
        "violations",
        "calls",
    ],
)
def test_audit_rejects_fabricated_trace_or_outcome(evidence, mutation):
    report, cases = evidence
    outcome = report["outcomes"][0]
    trace = outcome["environment_trace"]
    if mutation == "missing":
        del outcome["environment_trace"]
    elif mutation == "prompt":
        trace["initial_messages"][0]["content"] = "fabricated"
    elif mutation == "observation":
        trace["steps"][0]["observations"] = []
    elif mutation == "step_reward":
        trace["steps"][0]["reward"] = 10.0
    elif mutation == "terminal":
        trace["terminal"]["reward"] = 99.0
    elif mutation == "extra_action":
        trace["steps"].append(copy.deepcopy(trace["steps"][-1]))
    elif mutation == "success":
        outcome["success"] = not outcome["success"]
    elif mutation == "violations":
        # Keep the generic report valid so the mismatch reaches replay.
        outcome["success"] = False
        outcome["policy_violations"] = 4
    else:
        outcome["tool_calls"] = 0
    result = asyncio.run(audit_refund_traces(report, cases))
    assert not result["passed"]
    assert [issue["case_id"] for issue in result["issues"]] == [outcome["case_id"]]


def test_audit_rejects_changed_cases_and_implementation(evidence):
    report, cases = evidence
    altered = copy.deepcopy(cases)
    altered[0]["paid_cents"] += 1
    with pytest.raises(ValueError, match="cases differ"):
        asyncio.run(audit_refund_traces(report, altered))
    report["implementation"]["files"]["__init__.py"] = "0" * 64
    with pytest.raises(ValueError, match="exact implementation"):
        asyncio.run(audit_refund_traces(report, cases))


def test_external_stop_replays_prefix_without_claiming_to_verify_budget(evidence):
    report, cases = evidence
    by_id = {row["order_id"]: row for row in cases}
    outcome = next(row for row in report["outcomes"] if row["success"])
    trace = copy.deepcopy(outcome["environment_trace"])
    trace["steps"] = trace["steps"][:1]
    trace["terminal"] = {
        "reward": -1.0,
        "status": "timeout",
        "truncated": "generated_tokens",
    }
    result = asyncio.run(
        replay_refund_trace(report["environment"], by_id[outcome["case_id"]], trace)
    )
    assert not result["success"] and result["reward"] == -1
    assert result["tool_calls"] == 1 and result["external_stop_declared"]
    trace["terminal"]["truncated"] = None
    with pytest.raises(ValueError, match="external stop"):
        asyncio.run(
            replay_refund_trace(report["environment"], by_id[outcome["case_id"]], trace)
        )


def test_cli_publishes_bound_audit_and_nonzero_exit_for_replay_failure(
    evidence, tmp_path
):
    report, cases = evidence
    report_path, cases_path, audit_path = (
        tmp_path / name for name in ("report.json", "cases.json", "audit.json")
    )
    report_path.write_text(json.dumps(report))
    cases_path.write_text(json.dumps(cases))
    command = [
        "benchmark",
        "audit-refund-traces",
        "--report",
        str(report_path),
        "--cases",
        str(cases_path),
        "--output",
        str(audit_path),
    ]
    result = CliRunner().invoke(app, command)
    assert result.exit_code == 0, result.output
    assert json.loads(audit_path.read_text())["passed"]
    report["outcomes"][0]["environment_trace"]["steps"][0]["observations"] = []
    report_path.write_text(json.dumps(report))
    result = CliRunner().invoke(app, command)
    assert result.exit_code == 1 and "FAIL" in result.output
    assert not json.loads(audit_path.read_text())["passed"]
    before = report_path.read_bytes()
    result = CliRunner().invoke(app, [*command[:-1], str(report_path)])
    assert result.exit_code == 2 and report_path.read_bytes() == before
    report_path.write_text('{"duplicate": 1, "duplicate": 2}')
    result = CliRunner().invoke(app, command)
    assert result.exit_code == 2 and "Duplicate JSON key" in result.output


def test_replay_module_import_needs_no_river_sdk_or_tensor_libraries():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import stateset_agents.evaluation.refund_trace; assert not any(name in sys.modules for name in ('river_client', 'torch', 'transformers'))",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
