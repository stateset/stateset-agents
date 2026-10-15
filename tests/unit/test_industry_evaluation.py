"""Paired industry evaluation must reject incomplete or misleading evidence."""

from __future__ import annotations

import copy
import json
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from stateset_agents.data.finetuning import group_finetuning_data, load_finetuning_data
from stateset_agents.evaluation.industry import (
    IndustryEvaluationPolicy,
    collect_industry_predictions,
    evaluate_industry_project,
    export_industry_evaluation,
)
from stateset_agents.training.industry import (
    industry_examples,
    init_industry_project,
    prepare_industry_training,
)


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def project(tmp_path):
    # Independent two-turn lookup records; this is evaluator test data only.
    rows = []
    for index in range(8):
        row = industry_examples("retail")[0]
        row.pop("synthetic")
        row["group_id"] = f"case-{index}"
        row["messages"][1]["content"] = f"Look up order {index}."
        row["messages"][2]["tool_calls"][0]["function"]["arguments"] = {
            "record_id": str(index),
            "count": 1,
            "enabled": True,
        }
        rows.append(row)
    source = tmp_path / "rows.jsonl"
    write_rows(source, rows)
    output = tmp_path / "project"
    prepare_industry_training("retail", source, output, validation_fraction=0.5)
    return output


def bundles(project):
    requests = export_industry_evaluation(project)
    validation = load_finetuning_data(project / "validation.jsonl")
    predictions = []
    for case in requests["cases"]:
        target = next(
            row["messages"][len(case["messages"])]
            for row in validation
            if row["messages"][: len(case["messages"])] == case["messages"]
        )
        predictions.append(
            {
                "case_id": case["case_id"],
                "response": copy.deepcopy(target),
                "elapsed_seconds": 0.5,
                "generated_tokens": 10,
                "cost_usd": 0.002,
            }
        )
    baseline = {
        "schema_version": 1,
        "kind": "stateset-industry-predictions",
        "suite_sha256": requests["suite_sha256"],
        "variant": "baseline",
        "model": {
            "base_model": requests["base_model"],
            "revision": "immutable-revision",
        },
        "settings": {"do_sample": False, "max_new_tokens": 100},
        "predictions": predictions,
    }
    candidate = copy.deepcopy(baseline)
    candidate["variant"] = "candidate"
    candidate["model"]["adapter"] = {"id": "test-adapter", "sha256": "a" * 64}
    return baseline, candidate


def test_export_contains_every_prefix_without_current_or_future_targets(project):
    requests = export_industry_evaluation(project)
    assert len(requests["cases"]) == 8
    assert len({case["group_id"] for case in requests["cases"]}) == 4
    assert requests == export_industry_evaluation(project)
    for case in requests["cases"]:
        assert "expected" not in case
        if len(case["messages"]) == 2:
            assert all(m["role"] != "tool" for m in case["messages"])
        else:
            assert len(case["messages"]) == 4
            assert case["messages"][-1]["role"] == "tool"


def test_comparison_retains_quality_usage_and_exact_provenance(project):
    baseline, candidate = bundles(project)
    report = evaluate_industry_project(
        project, baseline, candidate, policy=IndustryEvaluationPolicy(min_groups=4)
    )
    assert report["gate"] == {"passed": True, "reasons": []}
    assert report["candidate"]["group_reference_match_rate"] == 1
    assert report["candidate"]["tool_exact_match_rate"] == 1
    assert report["candidate"]["generated_tokens"] == 80
    assert report["candidate"]["mean_cost_usd"] == 0.002
    assert report["group_reference_match_delta"] == 0
    assert report["identity_verification"] == "caller_declared"
    assert len(report["prediction_sha256"]["candidate"]) == 64
    candidate["model"]["adapter"]["id"] = "mutated"
    assert report["models"]["candidate"]["adapter"]["id"] == "test-adapter"


def test_default_gate_refuses_small_samples_even_when_perfect(project):
    report = evaluate_industry_project(project, *bundles(project))
    assert report["gate"] == {"passed": False, "reasons": ["insufficient_groups"]}


def test_synthetic_examples_can_be_scored_but_never_pass_gate(tmp_path):
    source = init_industry_project("retail", tmp_path / "source")
    project = tmp_path / "project"
    prepare_industry_training("retail", source / "examples.jsonl", project)
    report = evaluate_industry_project(
        project, *bundles(project), policy=IndustryEvaluationPolicy(min_groups=1)
    )
    assert not report["gate"]["passed"]
    assert any("synthetic_data" in reason for reason in report["gate"]["reasons"])


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "duplicate",
        "stale",
        "revision",
        "settings",
        "schema",
        "variant",
        "model",
        "adapter",
        "latency",
        "cost",
        "tokens",
        "error",
        "response",
    ],
)
def test_incomplete_or_incomparable_predictions_fail_closed(project, mutation):
    baseline, candidate = bundles(project)
    first = candidate["predictions"][0]
    if mutation == "missing":
        candidate["predictions"].pop()
    elif mutation == "extra":
        candidate["predictions"].append({**first, "case_id": "unknown"})
    elif mutation == "duplicate":
        candidate["predictions"].append(first)
    elif mutation == "stale":
        candidate["suite_sha256"] = "b" * 64
    elif mutation == "revision":
        candidate["model"]["revision"] = "different-revision"
    elif mutation == "settings":
        candidate["settings"]["max_new_tokens"] = 200
    elif mutation == "schema":
        candidate["schema_version"] = True
    elif mutation == "variant":
        candidate["variant"] = "baseline"
    elif mutation == "model":
        candidate["model"]["base_model"] = "other-model"
    elif mutation == "adapter":
        candidate["model"]["adapter"]["sha256"] = "not-a-hash"
    elif mutation == "latency":
        first["elapsed_seconds"] = float("nan")
    elif mutation == "cost":
        first["cost_usd"] = -1
    elif mutation == "tokens":
        first["generated_tokens"] = True
    elif mutation == "error":
        first["error"] = ""
    else:
        first.pop("response")
    with pytest.raises(ValueError):
        evaluate_industry_project(project, baseline, candidate)


@pytest.mark.parametrize(
    "arguments",
    [{"count": True}, {"count": "1"}, {"enabled": 1}, {"record_id": "wrong"}],
)
def test_tool_argument_types_and_values_are_not_coerced(project, arguments):
    baseline, candidate = bundles(project)
    row = next(r for r in candidate["predictions"] if r["response"].get("tool_calls"))
    row["response"]["tool_calls"][0]["function"]["arguments"].update(arguments)
    report = evaluate_industry_project(
        project, baseline, candidate, policy=IndustryEvaluationPolicy(min_groups=1)
    )
    assert report["regressed_case_ids"] == [row["case_id"]]
    assert report["candidate"]["tool_exact_match_rate"] == 0.75
    assert not report["gate"]["passed"]


def test_transport_ids_argument_json_and_whitespace_do_not_cause_false_regressions(
    project,
):
    baseline, candidate = bundles(project)
    for row in candidate["predictions"]:
        response = row["response"]
        if response.get("tool_calls"):
            call = response["tool_calls"][0]
            call["id"] = "provider-generated-id"
            call["function"]["arguments"] = json.dumps(
                call["function"]["arguments"], sort_keys=True
            )
        else:
            response["content"] = " \n " + response["content"].replace(" ", "  ")
    report = evaluate_industry_project(
        project, baseline, candidate, policy=IndustryEvaluationPolicy(min_groups=1)
    )
    assert report["gate"]["passed"]


@pytest.mark.parametrize(
    "response",
    [
        None,
        "free text",
        {"role": "user", "content": "wrong role"},
        {"role": "assistant", "content": "", "tool_calls": []},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"type": "function", "function": {"name": "invented", "arguments": {}}}
            ],
        },
    ],
)
def test_malformed_model_responses_count_as_failures_instead_of_disappearing(
    project, response
):
    baseline, candidate = bundles(project)
    candidate["predictions"][0]["response"] = response
    report = evaluate_industry_project(
        project, baseline, candidate, policy=IndustryEvaluationPolicy(min_groups=1)
    )
    assert report["candidate"]["cases"] == 8
    assert report["candidate"]["invalid_responses"] == 1
    assert len(report["regressed_case_ids"]) == 1


def test_new_tool_call_on_text_turn_is_a_regression(project):
    baseline, candidate = bundles(project)
    call = next(
        r["response"]["tool_calls"][0]
        for r in candidate["predictions"]
        if r["response"].get("tool_calls")
    )
    row = next(
        r for r in candidate["predictions"] if not r["response"].get("tool_calls")
    )
    row["response"]["tool_calls"] = [copy.deepcopy(call)]
    row["response"]["tool_calls"][0]["id"] = "new-call"
    report = evaluate_industry_project(
        project, baseline, candidate, policy=IndustryEvaluationPolicy(min_groups=1)
    )
    assert row["case_id"] in report["regressed_case_ids"]
    assert not report["gate"]["passed"]


def test_improvements_cannot_hide_regressions_in_aggregate(project):
    baseline, candidate = bundles(project)
    first, second = candidate["predictions"][:2]
    first["response"] = {"role": "assistant", "content": "wrong"}
    baseline["predictions"][1]["response"] = {"role": "assistant", "content": "wrong"}
    report = evaluate_industry_project(
        project,
        baseline,
        candidate,
        policy=IndustryEvaluationPolicy(min_groups=1, min_success_rate=0),
    )
    assert report["regressed_case_ids"] == [first["case_id"]]
    assert report["improved_case_ids"] == [second["case_id"]]
    assert "case_regression_limit_exceeded" in report["gate"]["reasons"]


def test_partial_group_improvements_do_not_count_as_group_success(project):
    baseline, candidate = bundles(project)
    baseline["predictions"][0]["response"] = None
    report = evaluate_industry_project(
        project,
        baseline,
        candidate,
        policy=IndustryEvaluationPolicy(min_groups=1, min_improvement=0.25),
    )
    assert report["group_reference_match_delta"] == 0.25
    assert report["gate"]["passed"]


def test_unknown_cost_is_not_zero_and_optional_limits_are_enforced(project):
    baseline, candidate = bundles(project)
    candidate["predictions"][0].pop("cost_usd")
    candidate["predictions"][0].pop("generated_tokens")
    report = evaluate_industry_project(
        project,
        baseline,
        candidate,
        policy=IndustryEvaluationPolicy(
            min_groups=1, max_mean_cost_usd=1, max_mean_latency_seconds=0.1
        ),
    )
    assert report["candidate"]["mean_cost_usd"] is None
    assert report["candidate"]["generated_tokens"] is None
    assert set(report["gate"]["reasons"]) == {
        "unknown_mean_cost_usd",
        "mean_latency_seconds_limit_exceeded",
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_groups": True},
        {"min_groups": 0},
        {"min_success_rate": float("nan")},
        {"min_improvement": 1.1},
        {"max_regression_rate": -1},
        {"max_mean_cost_usd": False},
        {"max_mean_latency_seconds": float("inf")},
    ],
)
def test_invalid_thresholds_are_rejected(kwargs):
    with pytest.raises(ValueError):
        IndustryEvaluationPolicy(**kwargs)


def test_callback_receives_detached_prefixes_and_records_exceptions(project):
    baseline, _ = bundles(project)
    requests = export_industry_evaluation(project)
    calls = []

    def predictor(messages, tools):
        calls.append(copy.deepcopy(messages))
        messages.clear()
        tools.clear()
        raise RuntimeError("backend unavailable")

    generated = collect_industry_predictions(
        project,
        predictor,
        variant="baseline",
        model=baseline["model"],
        settings=baseline["settings"],
    )
    assert len(calls) == 8
    assert calls == [c["messages"] for c in requests["cases"]]
    assert all(
        r["error"] == "RuntimeError: backend unavailable"
        for r in generated["predictions"]
    )
    assert all(
        r["response"] is None and r["elapsed_seconds"] >= 0
        for r in generated["predictions"]
    )
    assert export_industry_evaluation(project) == requests


def test_callback_usage_survives_and_baseline_error_does_not_drop_case(project):
    baseline, candidate = bundles(project)
    responses = iter(baseline["predictions"])

    def predictor(messages, tools):
        row = next(responses)
        return {"response": row["response"], "generated_tokens": 12, "cost_usd": 0.001}

    generated = collect_industry_predictions(
        project,
        predictor,
        variant="baseline",
        model=baseline["model"],
        settings=baseline["settings"],
    )
    generated["predictions"][0]["error"] = "provider failure"
    report = evaluate_industry_project(
        project, generated, candidate, policy=IndustryEvaluationPolicy(min_groups=1)
    )
    assert report["baseline"]["generated_tokens"] == 96
    assert report["baseline"]["invalid_responses"] == 1
    assert len(report["improved_case_ids"]) == 1


def test_group_count_uses_transitive_prompt_and_source_links():
    def row(prompt, group):
        return {
            "group_id": group,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "answer"},
            ],
        }

    rows = [row("A", "1"), row("B", "1"), row(" b ", "2"), row("C", "2"), row("D", "3")]
    groups = group_finetuning_data(rows + [copy.deepcopy(rows[0])])
    assert sorted(len(group) for group in groups) == [1, 4]
    assert groups == group_finetuning_data(list(reversed(rows)))


def test_changed_holdout_fails_before_callback(project):
    baseline, _ = bundles(project)
    with (project / "validation.jsonl").open("a") as stream:
        stream.write("\n")
    calls = []
    with pytest.raises(ValueError, match="changed after preparation"):
        collect_industry_predictions(
            project,
            lambda *args: calls.append(args),
            variant="baseline",
            model=baseline["model"],
            settings=baseline["settings"],
        )
    assert not calls


def test_predictions_must_match_prepared_model_revision(project):
    path = project / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["model_revision"] = "a" * 40
    path.write_text(json.dumps(manifest))
    baseline, candidate = bundles(project)
    with pytest.raises(ValueError, match="pinned revision"):
        evaluate_industry_project(project, baseline, candidate)
    for bundle in (baseline, candidate):
        bundle["model"]["revision"] = "a" * 40
    report = evaluate_industry_project(
        project, baseline, candidate, policy=IndustryEvaluationPolicy(min_groups=1)
    )
    assert report["gate"]["passed"]


def test_evaluation_imports_no_ml_dependencies(project):
    code = """
import sys
from stateset_agents.evaluation.industry import export_industry_evaluation
assert export_industry_evaluation(sys.argv[1])['cases']
assert not {'torch', 'transformers', 'peft', 'datasets'} & set(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code, str(project)], check=True, timeout=20)


def test_cli_exports_and_persists_both_passed_and_failed_gates(project, tmp_path):
    from stateset_agents.cli import app

    runner = CliRunner()
    output = tmp_path / "requests.json"
    args = ["industry", "eval-export", str(project), str(output)]
    assert runner.invoke(app, args).exit_code == 0
    assert runner.invoke(app, args).exit_code == 2
    base_path, candidate_path = tmp_path / "baseline.json", tmp_path / "candidate.json"
    baseline, candidate = bundles(project)
    base_path.write_text(json.dumps(baseline))
    candidate_path.write_text(json.dumps(candidate))
    report = tmp_path / "report.json"
    args = [
        "industry",
        "evaluate",
        str(project),
        "--baseline",
        str(base_path),
        "--candidate",
        str(candidate_path),
        "--output",
        str(report),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 1, result.output
    assert json.loads(report.read_text())["gate"]["reasons"] == ["insufficient_groups"]
    assert runner.invoke(app, args).exit_code == 2  # Never overwrite evidence.
    args[-1] = str(tmp_path / "passed.json")
    result = runner.invoke(app, args + ["--min-groups", "4"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["gate"]["passed"]
