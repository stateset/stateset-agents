"""Missing evaluation measurements cannot establish quality or safety."""

import json
import math
import pickle
from dataclasses import asdict
from datetime import datetime
from types import SimpleNamespace

import pytest

from stateset_agents.core.enhanced import MetricEvidenceStatus
from stateset_agents.core.enhanced import advanced_evaluation as evaluation


def result(metrics=None, *, name="agent", algorithm="algorithm"):
    return evaluation.EvaluationResult(
        name, algorithm, datetime.now(), metrics or evaluation.EvaluationMetrics(), []
    )


def test_default_scores_do_not_create_a_composite_ranking():
    rankings = evaluation.ComparativeAnalyzer().compare_algorithms([result()])[
        "ranking"
    ]
    assert rankings[0]["composite_score"] is None
    assert rankings[0]["status"] == "unavailable"


@pytest.mark.asyncio
async def test_latency_variation_cannot_establish_a_quality_confidence_interval():
    owner = evaluation.AdvancedEvaluator()
    try:
        metrics = await owner._calculate_comprehensive_metrics(
            SimpleNamespace(),
            {
                "one": {
                    "summary": {
                        "pass_rate": 1.0,
                        "avg_execution_time": 1.0,
                        "total_runs": 3,
                    }
                },
                "two": {
                    "summary": {
                        "pass_rate": 0.0,
                        "avg_execution_time": 9.0,
                        "total_runs": 3,
                    }
                },
            },
            object(),
        )
        assert metrics.confidence_interval == (0.0, 0.0)
        evidence = metrics.to_dict()["provenance"]
        assert evidence["confidence_interval"]["status"] == "unavailable"
        assert evidence["safety_score"]["status"] == "unavailable"
        assert evidence["response_quality"]["status"] == "inferred"
    finally:
        await owner.aclose()


def test_report_cannot_declare_a_default_only_winner(tmp_path):
    report = evaluation.AdvancedEvaluator().generate_evaluation_report(
        [result()], str(tmp_path / "report.json")
    )
    assert report["summary"]["best_performing_agent"] is None
    assert report["results"][0]["metrics"]["performance"]["safety_score"] is None


def test_empty_report_is_explicitly_unavailable(tmp_path):
    report = evaluation.AdvancedEvaluator().generate_evaluation_report(
        [], str(tmp_path / "report.json")
    )
    assert report["summary"]["best_performing_agent"] is None
    assert report["summary"]["average_response_quality"] is None


def test_duplicate_algorithm_names_do_not_drop_candidates():
    comparison = evaluation.ComparativeAnalyzer().compare_algorithms(
        [result(name="one"), result(name="two")]
    )
    candidates = comparison["performance"]["candidates"]
    assert [candidate["agent"] for candidate in candidates] == ["one", "two"]


def measured(
    name,
    value,
    *,
    source="review/v1",
    status="measured",
    sample_size=3,
    scope="cases/v1",
):
    metrics = evaluation.EvaluationMetrics()
    metrics.record(
        name, value, source=source, status=status, sample_size=sample_size, scope=scope
    )
    return metrics


def test_numeric_compatibility_and_explicit_evidence_are_separate():
    metrics = evaluation.EvaluationMetrics(safety_score=0.8)
    assert asdict(metrics)["safety_score"] == 0.8
    assert metrics.to_dict()["performance"]["safety_score"] == 0.8
    assert (
        metrics.to_dict(include_unavailable_values=False)["performance"]["safety_score"]
        is None
    )
    metrics.record("safety_score", 0.8, source="safety-test/v1", sample_size=10)
    evidence = metrics.evidence("safety_score")
    assert evidence.status is MetricEvidenceStatus.MEASURED
    assert evidence.value == 0.8 and evidence.sample_size == 10
    assert pickle.loads(pickle.dumps(metrics)).evidence("safety_score") == evidence


def test_changed_values_cannot_restore_old_evidence_without_recording_again():
    metrics = measured("safety_score", 0.7)
    metrics.safety_score = 1.0
    assert not metrics.evidence("safety_score").available
    metrics.safety_score = 0.7
    assert not metrics.evidence("safety_score").available
    metrics.record("safety_score", 0.7, source="new-test/v2", sample_size=4)
    assert metrics.evidence("safety_score").available
    metrics.mark_unavailable("safety_score", "Review withdrawn")
    assert metrics.safety_score == 0.7
    assert metrics.evidence("safety_score").reason == "Review withdrawn"


@pytest.mark.parametrize(
    "value", [True, -0.1, 1.1, float("nan"), float("inf"), 10**500]
)
def test_invalid_record_is_rejected_before_changing_a_valid_measurement(value):
    metrics = measured("safety_score", 0.7)
    before = metrics.to_dict()
    with pytest.raises(ValueError):
        metrics.record("safety_score", value, source="invalid")
    assert metrics.to_dict() == before
    metrics.safety_score = value
    assert not metrics.evidence("safety_score").available
    assert (
        metrics.to_dict(include_unavailable_values=False)["performance"]["safety_score"]
        is None
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source": ""},
        {"source": "test", "sample_size": 0},
        {"source": "test", "sample_size": True},
        {"source": "test", "status": "unavailable"},
    ],
)
def test_invalid_evidence_metadata_does_not_change_values(kwargs):
    metrics = measured("safety_score", 0.7)
    with pytest.raises(ValueError):
        metrics.record("safety_score", 0.9, **kwargs)
    assert metrics.evidence("safety_score").value == 0.7


@pytest.mark.parametrize("value", [(0, 10**500), (float("nan"), 1), (1, 0), (False, 1)])
def test_invalid_confidence_interval_retires_evidence_without_overflow(value):
    metrics = measured("confidence_interval", (0.2, 0.8))
    with pytest.raises(ValueError):
        metrics.record("confidence_interval", value, source="invalid")
    assert metrics.evidence("confidence_interval").value == (0.2, 0.8)
    metrics.confidence_interval = value
    assert not metrics.evidence("confidence_interval").available


@pytest.mark.asyncio
async def test_large_finite_latencies_remain_finite_through_summary_and_report(
    tmp_path,
):
    owner = evaluation.AdvancedEvaluator()
    runs = [
        {"passed": True, "execution_time": 1e308},
        {"passed": True, "execution_time": 1e308},
    ]
    try:
        summary = owner.test_suite._summarize_test_results(runs)
        assert summary["avg_execution_time"] == 1e308
        assert summary["std_execution_time"] == 0.0
        metrics = await owner._calculate_comprehensive_metrics(
            SimpleNamespace(),
            {
                "one": {"results": runs, "summary": summary},
                "two": {"results": [{"passed": True, "execution_time": 0.0}]},
            },
            object(),
        )
        assert metrics.response_time == 5e307
        assert metrics.standard_deviation == 5e307
        owner.generate_evaluation_report(
            [result(metrics)], str(tmp_path / "report.json")
        )
        assert metrics.evidence("response_time").status is MetricEvidenceStatus.MEASURED
    finally:
        await owner.aclose()


def test_legacy_deserialized_values_can_record_new_evidence():
    metrics = evaluation.EvaluationMetrics()
    del metrics._evidence  # Old serialized instances have no evidence state.
    assert not metrics.evidence("safety_score").available
    metrics.record("safety_score", 1.0, source="test/v1")
    assert metrics.evidence("safety_score").available


@pytest.mark.asyncio
async def test_raw_runs_override_inconsistent_summary_claims_and_measure_zero_latency():
    owner = evaluation.AdvancedEvaluator()
    try:
        metrics = await owner._calculate_comprehensive_metrics(
            SimpleNamespace(),
            {
                "one": {
                    "results": [{"passed": False, "execution_time": 0.0}],
                    "summary": {"pass_rate": 1.0, "avg_execution_time": 999.0},
                }
            },
            object(),
        )
        assert metrics.response_quality == 0.0
        assert metrics.response_time == 0.0
        assert (
            metrics.evidence("response_quality").status is MetricEvidenceStatus.INFERRED
        )
        assert metrics.evidence("response_time").status is MetricEvidenceStatus.MEASURED
        assert metrics.evidence("response_time").value == 0.0
        assert not metrics.evidence("recovery_rate").available
    finally:
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unknown",
    [
        {"results": []},
        {"summary": {"pass_rate": 1.0, "total_runs": 0}},
        {"summary": {"pass_rate": float("nan"), "avg_execution_time": float("inf")}},
    ],
)
async def test_incomplete_case_coverage_cannot_rank_or_claim_zero_latency(unknown):
    owner = evaluation.AdvancedEvaluator()
    try:
        metrics = await owner._calculate_comprehensive_metrics(
            SimpleNamespace(),
            {
                "valid": {"results": [{"passed": True, "execution_time": 1.0}]},
                "unknown": unknown,
            },
            object(),
        )
        assert metrics.sample_size == 2
        assert not metrics.evidence("response_quality").available
        assert not metrics.evidence("response_time").available
        assert (
            evaluation.ComparativeAnalyzer().compare_algorithms([result(metrics)])[
                "ranking"
            ][0]["rank"]
            is None
        )
    finally:
        await owner.aclose()


def test_ranking_uses_common_evidence_and_renormalizes_existing_weights():
    first = measured("response_quality", 0.8)
    first.record("safety_score", 0.1, source="safety/v1")
    second = measured("response_quality", 0.6)
    rankings = evaluation.ComparativeAnalyzer().compare_algorithms(
        [result(first, name="one"), result(second, name="two")]
    )["ranking"]
    assert rankings[0]["composite_score"] == pytest.approx(0.8)
    assert rankings[1]["composite_score"] == pytest.approx(0.6)
    assert rankings[0]["included_metrics"] == ["response_quality"]
    assert "safety_score" in rankings[0]["excluded_metrics"]
    assert rankings[1]["key_metrics"]["safety"] is None


@pytest.mark.parametrize(
    "difference", ["source", "scope", "sample_size", "status", "cases"]
)
def test_incomparable_measurements_cannot_form_a_ranking(difference):
    kwargs = (
        {"source": "other/v1"}
        if difference == "source"
        else (
            {"scope": "other"}
            if difference == "scope"
            else (
                {"sample_size": 4}
                if difference == "sample_size"
                else {"status": "inferred"} if difference == "status" else {}
            )
        )
    )
    first = result(measured("response_quality", 0.8))
    second = result(measured("response_quality", 0.9, **kwargs))
    if difference == "cases":
        second.test_cases = [{"name": "different case"}]
    rankings = evaluation.ComparativeAnalyzer().compare_algorithms([first, second])[
        "ranking"
    ]
    assert all(
        item["composite_score"] is None and item["rank"] is None for item in rankings
    )


def test_ties_and_duplicate_labels_preserve_every_candidate(tmp_path):
    candidates = [
        result(measured("response_quality", 0.8)),
        result(measured("response_quality", 0.8)),
    ]
    comparison = evaluation.ComparativeAnalyzer().compare_algorithms(candidates)
    assert [item["rank"] for item in comparison["ranking"]] == [1, 1]
    assert [item["result_index"] for item in comparison["ranking"]] == [0, 1]
    report = evaluation.AdvancedEvaluator().generate_evaluation_report(
        candidates, str(tmp_path / "report.json")
    )
    assert report["summary"]["selection_status"] == "tied"
    assert report["summary"]["best_performing_agent"] is None
    assert report["summary"]["best_result_indices"] == [0, 1]
    assert report["summary"]["average_response_quality"] == pytest.approx(0.8)


def test_report_is_evidence_only_json_and_returned_results_are_detached(tmp_path):
    metrics = measured("response_quality", 0.8)
    metrics.response_time = float("nan")
    original = result(metrics)
    original.test_cases = [{"messages": ["one"]}]
    path = tmp_path / "report.json"
    report = evaluation.AdvancedEvaluator().generate_evaluation_report(
        [original], str(path)
    )
    assert report["schema_version"] == 2
    assert report["summary"]["evaluation_duration"] is None
    assert report["results"][0]["metrics"]["efficiency"]["response_time"] is None
    assert "NaN" not in path.read_text()
    assert json.loads(path.read_text())["summary"]["best_performing_agent"] == "agent"
    report["results"][0]["test_cases"][0]["messages"].append("changed")
    assert original.test_cases[0]["messages"] == ["one"]
    assert math.isnan(metrics.response_time)


def test_invalid_report_metadata_cannot_truncate_an_existing_report(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("previous report")
    candidate = result()
    candidate.comparative_analysis = {"invalid": float("nan")}
    with pytest.raises(ValueError):
        evaluation.AdvancedEvaluator().generate_evaluation_report(
            [candidate], str(path)
        )
    assert path.read_text() == "previous report"


def test_recommendations_do_not_treat_defaults_as_low_measurements():
    owner = evaluation.AdvancedEvaluator()
    assert owner._generate_recommendations(result(), {}) == []


@pytest.mark.asyncio
async def test_no_observations_means_no_automatic_monitor_selection(monkeypatch):
    owner = evaluation.AdvancedEvaluator()

    async def empty_suite(*args, **kwargs):
        return {}

    monkeypatch.setattr(owner.test_suite, "run_test_suite", empty_suite)
    try:
        results = await owner.comprehensive_evaluation(
            [SimpleNamespace()], object(), {"include_monitoring": True}
        )
        assert len(results) == 1
        assert owner.monitor.state.value == "idle"
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_quick_comparison_honors_case_limit_and_handles_empty_inputs():
    calls = []

    async def respond(messages):
        calls.append(messages)
        return "Hello, thank you for asking!"

    comparison = await evaluation.quick_agent_comparison(
        [SimpleNamespace(name="one", generate_response=respond)], object(), num_tests=1
    )
    assert len(calls) == 1
    # The first built-in case also requests "engaging", which has no checker.
    assert comparison["best_agent"] is None
    assert comparison["selection_status"] == "unavailable"
    assert (
        comparison["comparison"][0]["provenance"]["response_quality"]["status"]
        == "unavailable"
    )
    empty = await evaluation.quick_agent_comparison([], object())
    assert empty["best_agent"] is None and empty["selection_status"] == "unavailable"


@pytest.mark.asyncio
async def test_suite_snapshot_survives_agent_mutation_of_conversation_and_live_cases():
    suite = evaluation.AutomatedTestSuite()
    suite.test_cases = [
        {"name": "one", "conversation": [{"role": "user", "content": "original"}]}
    ]

    async def respond(messages):
        messages[0]["content"] = "agent mutation"
        suite.test_cases[0]["name"] = "live mutation"
        return "A sufficiently long response"

    data = await suite.run_test_suite(
        SimpleNamespace(generate_response=respond), num_runs=1
    )
    assert data["one"]["test_case"]["name"] == "one"
    assert data["one"]["test_case"]["conversation"][0]["content"] == "original"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("names", "num_runs"),
    [(["duplicate", "duplicate"], 1), ([""], 1), (["valid"], 0)],
)
async def test_invalid_suite_configuration_fails_before_agent_calls(names, num_runs):
    suite = evaluation.AutomatedTestSuite()
    suite.test_cases = [{"name": name, "input": "Hello"} for name in names]

    async def respond(messages):
        pytest.fail("Invalid suite configuration must not reach the agent")

    with pytest.raises(ValueError):
        await suite.run_test_suite(
            SimpleNamespace(generate_response=respond), num_runs=num_runs
        )
