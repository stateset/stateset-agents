"""Rubric coverage is separate from a response passing or failing a check."""

import asyncio
import inspect
from types import SimpleNamespace

import pytest

from stateset_agents.core.enhanced import advanced_evaluation as evaluation


async def greeting(messages):
    return "Hello, thank you for asking!"


@pytest.mark.asyncio
async def test_unsupported_criteria_are_unavailable_instead_of_failed():
    suite = evaluation.AutomatedTestSuite()
    suite.test_cases = [
        {
            "name": "privacy",
            "input": "Hello",
            "expected_characteristics": ["privacy_aware"],
        }
    ]
    row = (
        await suite.run_test_suite(
            SimpleNamespace(generate_response=greeting), num_runs=1
        )
    )["privacy"]
    assert row["results"][0]["passed"] is None
    assert row["results"][0]["status"] == "unavailable"
    assert row["summary"]["pass_rate"] is None
    assert row["summary"]["failed_runs"] == 0
    assert row["summary"]["unavailable_runs"] == 1


@pytest.mark.asyncio
async def test_partial_rubric_cannot_establish_quality_but_keeps_latency_and_reasons(
    tmp_path,
):
    owner = evaluation.AdvancedEvaluator()
    owner.test_suite.test_cases = [
        {
            "name": "partial",
            "input": "Hello",
            "expected_characteristics": ["greeting", "privacy_aware"],
        }
    ]
    try:
        results = await owner.comprehensive_evaluation(
            [SimpleNamespace(generate_response=greeting)],
            object(),
            {"num_test_runs": 1, "include_monitoring": True},
        )
        assert not results[0].metrics.evidence("response_quality").available
        assert results[0].metrics.evidence("response_time").available
        assert owner.monitor.state.value == "idle"
        report = owner.generate_evaluation_report(
            results, str(tmp_path / "report.json")
        )
        run = report["results"][0]["test_results"]["partial"]["results"][0]
        assert run["evaluation"]["overall_score"] is None
        assert [item["status"] for item in run["evaluation"]["criteria"]] == [
            "passed",
            "unavailable",
        ]
        assert "privacy_aware" in results[0].metrics.evidence("response_quality").reason
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_verbose_response_is_not_a_factuality_check():
    result = await evaluation.AutomatedTestSuite()._evaluate_response(
        "The moon is made of cheese and orbits Mars.",
        {"expected_characteristics": ["factual"]},
    )
    assert result["status"] == "unavailable"
    assert result["coverage"]["assessed"] == 0


@pytest.mark.asyncio
async def test_implemented_check_failure_is_distinct_from_missing_capability():
    suite = evaluation.AutomatedTestSuite()
    suite.test_cases = [
        {"name": "greeting", "input": "Hello", "expected_characteristics": ["greeting"]}
    ]

    async def respond(messages):
        return "This contains the letters but no greeting."

    row = (
        await suite.run_test_suite(
            SimpleNamespace(generate_response=respond), num_runs=1
        )
    )["greeting"]
    assert row["results"][0]["passed"] is False
    assert row["summary"]["pass_rate"] == 0.0
    assert row["summary"]["failed_runs"] == 1


@pytest.mark.asyncio
async def test_empty_expectations_do_not_establish_a_failed_quality_observation():
    suite = evaluation.AutomatedTestSuite()
    row = await suite._run_single_test(
        SimpleNamespace(generate_response=greeting), {"input": "Hello"}, None
    )
    assert row["passed"] is None
    assert row["evaluation"]["coverage"]["total"] == 0


def test_capabilities_are_detached_and_do_not_advertise_factual_judgments():
    rubric = evaluation.EvaluationRubric()
    capabilities = rubric.capabilities()
    assert set(capabilities["characteristics"]) == {"greeting", "polite"}
    capabilities["characteristics"]["greeting"]["source"] = "mutated"
    assert (
        rubric.capabilities()["characteristics"]["greeting"]["source"]
        == "greeting_word_presence/v2"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, "yes", 1])
async def test_abstentions_and_invalid_checker_outputs_do_not_fail_the_agent(value):
    rubric = evaluation.EvaluationRubric()
    rubric.register_criterion("custom", lambda response: value, source="custom/v1")
    suite = evaluation.AutomatedTestSuite(rubric)
    suite.test_cases = [
        {"name": "custom", "input": "Hello", "expected_characteristics": ["custom"]}
    ]
    row = (
        await suite.run_test_suite(
            SimpleNamespace(generate_response=greeting), num_runs=1
        )
    )["custom"]
    assert row["results"][0]["passed"] is None
    assert row["summary"]["failed_runs"] == 0
    assert row["summary"]["error_runs"] == (0 if value is None else 1)
    assert row["summary"]["unavailable_runs"] == (1 if value is None else 0)


def test_checker_exceptions_are_errors_and_process_control_propagates():
    rubric = evaluation.EvaluationRubric()

    def broken(response):
        raise RuntimeError("Checker failed")

    rubric.register_criterion("custom", broken, source="custom/v1")
    assessment = rubric.evaluate("Hello", {"expected_characteristics": ["custom"]})
    assert assessment["status"] == "error"
    assert assessment["criteria"][0]["reason"] == "RuntimeError: Checker failed"

    class Stop(BaseException):
        pass

    def stop(response):
        raise Stop()

    rubric.register_criterion("custom", stop, source="custom/v2")
    with pytest.raises(Stop):
        rubric.evaluate("Hello", {"expected_characteristics": ["custom"]})


def test_async_checkers_are_rejected_and_hidden_coroutines_are_closed():
    rubric = evaluation.EvaluationRubric()

    async def async_check(response):
        return True

    with pytest.raises(ValueError, match="synchronous"):
        rubric.register_criterion("custom", async_check, source="custom/v1")
    coroutines = []

    def hidden_async(response):
        coroutine = async_check(response)
        coroutines.append(coroutine)
        return coroutine

    rubric.register_criterion("custom", hidden_async, source="custom/v1")
    assessment = rubric.evaluate("Hello", {"expected_characteristics": ["custom"]})
    assert assessment["status"] == "error"
    assert inspect.getcoroutinestate(coroutines[0]) == inspect.CORO_CLOSED


@pytest.mark.asyncio
async def test_generation_failure_is_an_error_without_a_quality_verdict():
    async def broken(messages):
        raise RuntimeError("Generation failed")

    suite = evaluation.AutomatedTestSuite()
    suite.test_cases = [
        {"name": "one", "input": "Hello", "expected_characteristics": ["greeting"]}
    ]
    row = (
        await suite.run_test_suite(
            SimpleNamespace(generate_response=broken), num_runs=1
        )
    )["one"]
    assert row["results"][0]["error_stage"] == "generation"
    assert row["summary"]["pass_rate"] is None
    assert row["summary"]["error_runs"] == 1
    assert row["summary"]["failed_runs"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expectations",
    [
        {"expected_characteristics": "greeting"},
        {"expected_characteristics": ["greeting", "greeting"]},
        {"expected_characteristics": [""]},
        {"expected_answer": ""},
        {"expected_answer": None},
    ],
)
async def test_invalid_rubrics_fail_before_any_agent_call(expectations):
    suite = evaluation.AutomatedTestSuite()
    suite.test_cases = [
        {"name": "valid", "input": "Hello", "expected_answer": "Hello"},
        {"name": "invalid", "input": "Hello", **expectations},
    ]

    async def must_not_call(messages):
        pytest.fail("All cases must be validated before the first agent call")

    with pytest.raises(ValueError):
        await suite.run_test_suite(
            SimpleNamespace(generate_response=must_not_call), num_runs=1
        )


@pytest.mark.asyncio
async def test_concurrent_suites_keep_distinct_captured_rubrics():
    suite = evaluation.AutomatedTestSuite()
    suite.test_cases = [
        {"name": "one", "input": "Hello", "expected_characteristics": ["custom"]}
    ]
    suite.rubric.register_criterion(
        "custom", lambda response: False, source="custom/v1"
    )
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked(messages):
        entered.set()
        await release.wait()
        return "Hello"

    first = asyncio.create_task(
        suite.run_test_suite(SimpleNamespace(generate_response=blocked), num_runs=1)
    )
    try:
        await entered.wait()
        suite.rubric.register_criterion(
            "custom", lambda response: True, source="custom/v2"
        )
        second = await suite.run_test_suite(
            SimpleNamespace(generate_response=greeting), num_runs=1
        )
        release.set()
        original = await first
        old_run = original["one"]["results"][0]
        new_run = second["one"]["results"][0]
        assert old_run["passed"] is False and new_run["passed"] is True
        assert old_run["evaluation"]["criteria"][0]["source"] == "custom/v1"
        assert new_run["evaluation"]["criteria"][0]["source"] == "custom/v2"
    finally:
        release.set()
        await first


@pytest.mark.asyncio
async def test_changed_rubric_methods_cannot_establish_a_shared_quality_winner(
    tmp_path,
):
    owner = evaluation.AdvancedEvaluator()
    owner.test_suite.test_cases = [
        {"name": "one", "input": "Hello", "expected_characteristics": ["custom"]}
    ]
    owner.test_suite.rubric.register_criterion(
        "custom", lambda response: True, source="custom/v1"
    )

    async def change_next_run(messages):
        owner.test_suite.rubric.register_criterion(
            "custom", lambda response: True, source="custom/v2"
        )
        return "Hello"

    try:
        results = await owner.comprehensive_evaluation(
            [
                SimpleNamespace(generate_response=change_next_run),
                SimpleNamespace(generate_response=greeting),
            ],
            object(),
            {"num_test_runs": 1, "include_monitoring": True},
        )
        assert all(
            item.metrics.evidence("response_quality").available for item in results
        )
        assert all(
            "response_quality" not in item["included_metrics"]
            for item in results[0].comparative_analysis["ranking"]
        )
        assert owner.monitor.state.value == "idle"
        report = owner.generate_evaluation_report(
            results, str(tmp_path / "report.json")
        )
        assert report["summary"]["selection_status"] == "unavailable"
        report["results"][0]["test_results"]["one"]["results"][0]["evaluation"][
            "criteria"
        ].clear()
        assert results[0].test_results["one"]["results"][0]["evaluation"]["criteria"]
        results[0].test_results.clear()
        assert owner.results_history[0].test_results
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_complete_registered_rubric_can_select_and_monitor_the_agent():
    owner = evaluation.AdvancedEvaluator()
    owner.test_suite.test_cases = [
        {
            "name": "one",
            "input": "Hello",
            "expected_characteristics": ["greeting", "thanks"],
        }
    ]
    owner.test_suite.rubric.register_criterion(
        "thanks", lambda response: "thank you" in response, source="thanks_phrase/v1"
    )
    try:
        results = await owner.comprehensive_evaluation(
            [SimpleNamespace(generate_response=greeting)],
            object(),
            {"num_test_runs": 1, "include_monitoring": True},
        )
        assert results[0].metrics.response_quality == 1.0
        assert (
            results[0].metrics.evidence("response_quality").status.value == "inferred"
        )
        assert owner.monitor.state.value == "running"
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_unsupported_raw_assessment_cannot_be_overridden_by_a_positive_boolean():
    suite = evaluation.AutomatedTestSuite()
    assessment = await suite._evaluate_response(
        "Hello", {"expected_characteristics": ["privacy_aware"]}
    )
    row = {
        "results": [
            {
                "passed": True,
                "status": "passed",
                "execution_time": 0.1,
                "evaluation": assessment,
            }
        ]
    }
    rate, latency, count, direct = evaluation._case_measurements(row)
    assert rate is None and latency == 0.1 and count == 1 and direct


@pytest.mark.parametrize(
    "unavailable,errors,passed,failed,expected",
    [
        (1, 0, 0, 0, None),
        (0, 1, 0, 0, None),
        (0, 0, 0, 1, 0.0),
        (0, 0, 2, 0, None),
        (False, 0, 1, 0, None),
    ],
)
def test_supplied_summary_counts_take_precedence_over_quality_claims(
    unavailable, errors, passed, failed, expected
):
    row = {
        "summary": {
            "pass_rate": 1.0,
            "avg_execution_time": 0.1,
            "total_runs": 1,
            "passed_runs": passed,
            "failed_runs": failed,
            "unavailable_runs": unavailable,
            "error_runs": errors,
        }
    }
    rate, latency, count, direct = evaluation._case_measurements(row)
    assert rate == expected and latency == 0.1 and count == 1 and not direct


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cleanup_failure", "repeat_cancel"), [(False, False), (False, True), (True, False)]
)
async def test_cancelled_suite_never_replays_after_callback_suppression_or_cleanup_error(
    cleanup_failure,
    repeat_cancel,
):
    suite = evaluation.AutomatedTestSuite()
    suite.test_cases = [{"name": "one", "input": "Hello", "expected_answer": "Hello"}]
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = 0

    async def respond(messages):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cleaning.set()
                await release.wait()
                if cleanup_failure:
                    raise RuntimeError("Cleanup failed") from None
        return "Hello"

    runner = asyncio.create_task(
        suite.run_test_suite(SimpleNamespace(generate_response=respond), num_runs=2)
    )
    try:
        await entered.wait()
        runner.cancel()
        await cleaning.wait()
        assert not runner.done()
        if repeat_cancel:
            runner.cancel()
            await asyncio.sleep(0)
            assert not runner.done()
        release.set()
        with pytest.raises(RuntimeError if cleanup_failure else asyncio.CancelledError):
            await runner
        assert calls == 1
    finally:
        release.set()
        await asyncio.gather(runner, return_exceptions=True)


@pytest.mark.asyncio
async def test_owned_evaluator_retains_suite_cancellation_cleanup_failure():
    owner = evaluation.AdvancedEvaluator()
    owner.test_suite.test_cases = [
        {"name": "one", "input": "Hello", "expected_answer": "Hello"}
    ]
    entered = asyncio.Event()
    calls = 0

    async def respond(messages):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            try:
                await asyncio.Future()
            finally:
                raise RuntimeError("Cleanup failed")
        return "Hello"

    runner = asyncio.create_task(
        owner.comprehensive_evaluation(
            [SimpleNamespace(generate_response=respond)], object(), {"num_test_runs": 2}
        )
    )
    try:
        await entered.wait()
        runner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runner
        assert calls == 1
        assert owner.state.value == "failed"
        assert owner.results_history == []
        with pytest.raises(RuntimeError, match="Cleanup failed"):
            await owner.aclose()
    finally:
        await asyncio.gather(runner, return_exceptions=True)
