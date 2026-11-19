"""Continuous evaluation retains one selected agent and owns its work."""

import asyncio
import inspect
from types import SimpleNamespace

import pytest

from stateset_agents.core.enhanced import advanced_evaluation as evaluation


async def wait_until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=2)


@pytest.mark.asyncio
async def test_repeated_legacy_start_shares_a_single_monitor():
    owner = evaluation.ContinuousMonitor()
    calls = []

    async def respond(messages):
        calls.append("sample")
        await asyncio.Event().wait()

    agent, environment = SimpleNamespace(generate_response=respond), object()
    runners = [
        asyncio.create_task(owner.start_monitoring(agent, environment))
        for _ in range(2)
    ]
    try:
        await wait_until(lambda: calls)
        for _ in range(10):
            await asyncio.sleep(0)
        assert calls == ["sample"]
    finally:
        for runner in runners:
            runner.cancel()
        await asyncio.gather(*runners, return_exceptions=True)
        if hasattr(owner, "aclose"):
            await owner.aclose()


@pytest.mark.asyncio
async def test_managed_start_is_idempotent_stop_is_restartable_and_close_is_terminal():
    owner = evaluation.ContinuousMonitor()
    calls = []

    async def respond(messages):
        calls.append("sample")
        return "Hello"

    agent, environment = SimpleNamespace(generate_response=respond), object()
    try:
        await asyncio.gather(
            owner.start(agent, environment), owner.start(agent, environment)
        )
        await wait_until(lambda: len(owner.performance_history) == 1)
        assert calls == ["sample"]
        first_generation = owner.get_status()["generation"]
        await owner.stop_monitoring()
        assert owner.state.value == "idle"
        assert not owner.get_status()["worker_pending"]
        await owner.start(agent, environment)
        await wait_until(lambda: len(owner.performance_history) == 2)
        assert owner.get_status()["generation"] == first_generation + 1
        await owner.aclose()
        await owner.aclose()
        await owner.stop_monitoring()
        assert owner.state.value == "closed"
        with pytest.raises(RuntimeError, match="closing|closed"):
            await owner.start(agent, environment)
    finally:
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("close_during_replacement", [False, True])
async def test_replacement_waits_for_old_cleanup_and_close_fences_new_start(
    close_during_replacement,
):
    owner = evaluation.ContinuousMonitor()
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = []

    async def old_response(messages):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            calls.append("old drained")

    async def new_response(messages):
        calls.append("new sampled")
        return "hello"

    old = SimpleNamespace(generate_response=old_response)
    new = SimpleNamespace(generate_response=new_response)
    environment = object()
    replacing = closing = None
    try:
        await owner.start(old, environment)
        await asyncio.wait_for(entered.wait(), 2)
        replacing = asyncio.create_task(owner.start(new, environment))
        await asyncio.wait_for(cleaning.wait(), 2)
        assert not replacing.done() and not calls
        if close_during_replacement:
            closing = asyncio.create_task(owner.aclose())
            await wait_until(lambda: owner.state.value == "closing")
        else:
            for _ in range(2):
                replacing.cancel()
                await asyncio.sleep(0)
            assert not replacing.done()
        release.set()
        if close_during_replacement:
            with pytest.raises(RuntimeError, match="closing|closed"):
                await replacing
            await closing
            assert calls == ["old drained"]
        else:
            with pytest.raises(asyncio.CancelledError):
                await replacing
            await wait_until(lambda: len(calls) == 2)
            assert calls == ["old drained", "new sampled"]
            assert owner.get_status()["worker_pending"]
    finally:
        release.set()
        await asyncio.gather(
            *[task for task in (replacing, closing) if task], return_exceptions=True
        )
        await owner.aclose()


@pytest.mark.asyncio
async def test_concurrent_cancelled_close_waiters_retain_async_cleanup():
    owner = evaluation.ContinuousMonitor()
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def response(messages):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()

    first = second = None
    try:
        await owner.start(SimpleNamespace(generate_response=response), object())
        await asyncio.wait_for(entered.wait(), 2)
        first = asyncio.create_task(owner.aclose())
        second = asyncio.create_task(owner.aclose())
        await asyncio.wait_for(cleaning.wait(), 2)
        for _ in range(2):
            first.cancel()
            await asyncio.sleep(0)
        assert not first.done() and not second.done()
        assert owner.get_status()["worker_pending"]
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await second
        assert owner.state.value == "closed"
    finally:
        release.set()
        await asyncio.gather(
            *[task for task in (first, second) if task], return_exceptions=True
        )
        await owner.aclose()


@pytest.mark.asyncio
async def test_legacy_runner_cancellation_stops_and_drains_its_worker():
    owner = evaluation.ContinuousMonitor()
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def response(messages):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()

    runner = asyncio.create_task(
        owner.start_monitoring(SimpleNamespace(generate_response=response), object())
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        runner.cancel()
        await asyncio.wait_for(cleaning.wait(), 2)
        runner.cancel()
        await asyncio.sleep(0)
        assert not runner.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await runner
        assert owner.state.value == "idle"
        assert not owner.get_status()["worker_pending"]
    finally:
        release.set()
        await asyncio.gather(runner, return_exceptions=True)
        await owner.aclose()


@pytest.mark.asyncio
async def test_late_result_after_stop_cannot_publish_history_or_alerts():
    owner = evaluation.ContinuousMonitor()
    entered = asyncio.Event()

    async def response(messages):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return "late result"

    await owner.start(SimpleNamespace(generate_response=response), object())
    await asyncio.wait_for(entered.wait(), 2)
    await owner.aclose()
    assert owner.performance_history == []
    assert owner.alerts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_phase", ["sample", "cleanup"])
async def test_failed_workers_cannot_restart_or_hide_cleanup_failure(failure_phase):
    owner = evaluation.ContinuousMonitor()
    entered = asyncio.Event()
    failure = (
        AssertionError("unexpected sample failure")
        if failure_phase == "sample"
        else RuntimeError("cleanup failed")
    )
    calls = []

    async def response(messages):
        calls.append("sample")
        entered.set()
        if failure_phase == "sample":
            raise failure
        try:
            await asyncio.Event().wait()
        finally:
            raise failure

    agent = SimpleNamespace(generate_response=response)
    await owner.start(agent, object())
    await asyncio.wait_for(entered.wait(), 2)
    if failure_phase == "sample":
        await wait_until(lambda: owner.state.value == "failed")
    for _ in range(2):
        with pytest.raises(type(failure)) as caught:
            await owner.aclose()
        assert caught.value is failure
    assert owner.state.value == "failed"
    assert not owner.get_status()["worker_pending"]
    with pytest.raises(RuntimeError, match="closing|closed"):
        await owner.start(agent, object())
    assert calls == ["sample"]


@pytest.mark.asyncio
async def test_nested_callback_cannot_close_its_monitor():
    owner = evaluation.ContinuousMonitor()
    guarded = []

    async def child():
        with pytest.raises(RuntimeError, match="own monitor"):
            await owner.aclose()
        guarded.append(True)

    async def response(messages):
        await asyncio.create_task(child())
        return "hello"

    try:
        await owner.start(SimpleNamespace(generate_response=response), object())
        await wait_until(lambda: owner.performance_history)
        assert guarded == [True]
        assert owner.state.value == "running"
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_snapshot_history_is_not_mutated_by_alert_handler(monkeypatch):
    owner = evaluation.ContinuousMonitor()
    metrics = {"error": True, "error_message": "failed", "details": {"attempt": 1}}

    async def quick(agent, environment):
        return metrics

    async def alerts(items):
        items[0]["error_message"] = "handler changed it"
        metrics["details"]["attempt"] = 2

    monkeypatch.setattr(owner, "_quick_evaluation", quick)
    monkeypatch.setattr(owner, "_handle_alerts", alerts)
    try:
        await owner.start(SimpleNamespace(generate_response=lambda _: None), object())
        await wait_until(lambda: owner.alerts)
        assert owner.performance_history[0]["metrics"]["details"]["attempt"] == 1
        assert owner.alerts[0]["error_message"] == "failed"
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_evaluator_selects_real_agent_by_result_position(monkeypatch):
    owner = evaluation.AdvancedEvaluator()
    selected = []

    async def suite(agent, environment, num_runs):
        return {
            "sample": {
                "summary": {"pass_rate": agent.quality, "avg_execution_time": 1.0}
            }
        }

    async def start(agent, environment):
        selected.append(agent)

    monkeypatch.setattr(owner.test_suite, "run_test_suite", suite)
    # Support both the old detached runner and the managed-start API under test.
    monkeypatch.setattr(owner.monitor, "start_monitoring", start)
    monkeypatch.setattr(owner.monitor, "start", start, raising=False)
    agents = [
        SimpleNamespace(name="same", quality=0.1),
        SimpleNamespace(name="same", quality=0.9),
    ]
    try:
        results = await owner.comprehensive_evaluation(
            agents,
            object(),
            evaluation.create_evaluation_config(include_monitoring=True),
        )
        await wait_until(lambda: selected)
        assert len(results) == 2
        assert selected == [agents[1]]
    finally:
        if hasattr(owner, "aclose"):
            await owner.aclose()


@pytest.mark.asyncio
async def test_evaluator_drains_monitor_before_foreground_and_honors_monitoring_choice(
    monkeypatch,
):
    owner = evaluation.AdvancedEvaluator()
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    events = []

    async def response(messages):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            events.append("monitor drained")

    async def suite(agent, environment, num_runs):
        assert events == ["monitor drained"]
        events.append("suite")
        return {"case": {"summary": {"pass_rate": 1.0, "avg_execution_time": 0.5}}}

    monkeypatch.setattr(owner.test_suite, "run_test_suite", suite)
    agent, environment = SimpleNamespace(generate_response=response), object()
    evaluating = None
    try:
        await owner.monitor.start(agent, environment)
        await asyncio.wait_for(entered.wait(), 2)
        evaluating = asyncio.create_task(
            owner.comprehensive_evaluation([agent], environment)
        )
        await asyncio.wait_for(cleaning.wait(), 2)
        assert not evaluating.done() and not events
        release.set()
        results = await evaluating
        assert results[0].metrics.response_quality == 1.0
        assert events == ["monitor drained", "suite"]
        assert owner.monitor.state.value == "idle"
        results[0].metrics.response_quality = 0.0
        results[0].test_cases[0]["name"] = "observer mutation"
        assert owner.results_history[0].metrics.response_quality == 1.0
        assert owner.results_history[0].test_cases[0]["name"] != "observer mutation"
    finally:
        release.set()
        if evaluating:
            await asyncio.gather(evaluating, return_exceptions=True)
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_evaluator_close_owns_active_and_queued_work(monkeypatch, cleanup_fails):
    owner = evaluation.AdvancedEvaluator()
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    events = []
    failure = RuntimeError("suite cleanup failed")

    async def suite(agent, environment, num_runs):
        events.append("suite")
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            events.append("suite drained")
            if cleanup_fails:
                raise failure

    original_close = owner.monitor.aclose

    async def close_monitor():
        assert events == ["suite", "suite drained"]
        events.append("monitor closed")
        await original_close()

    monkeypatch.setattr(owner.test_suite, "run_test_suite", suite)
    monkeypatch.setattr(owner.monitor, "aclose", close_monitor)
    agent, environment = SimpleNamespace(), object()
    first = asyncio.create_task(owner.comprehensive_evaluation([agent], environment))
    queued = closing = other = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        queued = asyncio.create_task(
            owner.comprehensive_evaluation([agent], environment)
        )
        await wait_until(lambda: owner.get_status()["active_evaluations"] == 2)
        closing = asyncio.create_task(owner.aclose())
        other = asyncio.create_task(owner.aclose())
        await asyncio.wait_for(cleaning.wait(), 2)
        for _ in range(2):
            closing.cancel()
            first.cancel()
            await asyncio.sleep(0)
        assert not closing.done() and not first.done()
        with pytest.raises(RuntimeError, match="closing|closed"):
            await owner.comprehensive_evaluation([agent], environment)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await closing
        if cleanup_fails:
            with pytest.raises(RuntimeError) as caught:
                await other
            assert caught.value is failure
            assert owner.state.value == "failed"
        else:
            await other
            assert owner.state.value == "closed"
        assert events == ["suite", "suite drained", "monitor closed"]
        assert owner.get_status()["active_evaluations"] == 0
        await asyncio.gather(first, queued, return_exceptions=True)
    finally:
        release.set()
        await asyncio.gather(
            *[task for task in (first, queued, closing, other) if task],
            return_exceptions=True,
        )
        if cleanup_fails:
            with pytest.raises(RuntimeError):
                await owner.aclose()
        else:
            await owner.aclose()


@pytest.mark.asyncio
async def test_cancelled_foreground_request_drains_and_does_not_start_monitor(
    monkeypatch,
):
    owner = evaluation.AdvancedEvaluator()
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def suite(agent, environment, num_runs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()

    monkeypatch.setattr(owner.test_suite, "run_test_suite", suite)
    task = asyncio.create_task(
        owner.comprehensive_evaluation(
            [SimpleNamespace()], object(), {"include_monitoring": True}
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        await asyncio.wait_for(cleaning.wait(), 2)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert owner.monitor.state.value == "idle"
        assert owner.results_history == []
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await owner.aclose()


@pytest.mark.asyncio
async def test_evaluation_captures_agents_and_configuration_before_awaiting(
    monkeypatch,
):
    owner = evaluation.AdvancedEvaluator()
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def suite(agent, environment, num_runs):
        calls.append((agent, num_runs))
        entered.set()
        await release.wait()
        return {}

    monkeypatch.setattr(owner.test_suite, "run_test_suite", suite)
    first, second = SimpleNamespace(), SimpleNamespace()
    agents = [first, second]
    config = {"num_test_runs": 2}
    task = asyncio.create_task(owner.comprehensive_evaluation(agents, object(), config))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        agents.clear()
        config["num_test_runs"] = 99
        config["include_monitoring"] = True
        release.set()
        assert len(await task) == 2
        assert calls == [(first, 2), (second, 2)]
        assert owner.monitor.state.value == "idle"
        assert (
            await owner.comprehensive_evaluation(
                [], object(), {"include_monitoring": True}
            )
            == []
        )
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await owner.aclose()


@pytest.mark.asyncio
async def test_evaluation_callback_cannot_close_its_evaluator(monkeypatch):
    owner = evaluation.AdvancedEvaluator()

    async def child():
        with pytest.raises(RuntimeError, match="own evaluator"):
            await owner.aclose()

    async def suite(agent, environment, num_runs):
        await asyncio.create_task(child())
        return {}

    monkeypatch.setattr(owner.test_suite, "run_test_suite", suite)
    try:
        assert (
            len(await owner.comprehensive_evaluation([SimpleNamespace()], object()))
            == 1
        )
        assert owner.state.value == "ready"
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_rejected_worker_task_retains_failed_start_without_leaking_coroutine(
    monkeypatch,
):
    owner = evaluation.ContinuousMonitor()
    original = asyncio.create_task
    operations = []
    failure = RuntimeError("worker task rejected")

    def create(operation, *args, **kwargs):
        if operation.cr_code.co_name == "_monitor":
            operations.append(operation)
            raise failure
        return original(operation, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(asyncio, "create_task", create)
        with pytest.raises(RuntimeError) as caught:
            await owner.start(
                SimpleNamespace(generate_response=lambda _: None), object()
            )
        assert caught.value is failure
    assert inspect.getcoroutinestate(operations[0]) == inspect.CORO_CLOSED
    assert owner.state.value == "failed"
    assert not owner.get_status()["worker_pending"]
    with pytest.raises(RuntimeError) as caught:
        await owner.aclose()
    assert caught.value is failure


@pytest.mark.asyncio
@pytest.mark.parametrize("interval", [0, -1, float("inf"), float("nan"), True])
async def test_invalid_interval_fails_before_starting_work(interval):
    owner = evaluation.ContinuousMonitor(interval)
    with pytest.raises((TypeError, ValueError), match="interval"):
        await owner.start(SimpleNamespace(generate_response=lambda _: None), object())
    assert owner.state.value == "idle"
    assert not owner.get_status()["worker_pending"]
    await owner.aclose()


@pytest.mark.asyncio
async def test_eager_scheduling_retains_monitor_and_evaluator_ownership(monkeypatch):
    if not hasattr(asyncio, "eager_task_factory"):
        pytest.skip("Eager task factories require Python 3.12")
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    owner = evaluation.AdvancedEvaluator()

    async def suite(agent, environment, num_runs):
        return {"case": {"results": [{"passed": True, "execution_time": 0.1}]}}

    async def respond(messages):
        return "hello"

    monkeypatch.setattr(owner.test_suite, "run_test_suite", suite)
    try:
        loop.set_task_factory(asyncio.eager_task_factory)
        await owner.comprehensive_evaluation(
            [SimpleNamespace(generate_response=respond)],
            object(),
            {"include_monitoring": True},
        )
        await wait_until(lambda: owner.monitor.performance_history)
        assert owner.monitor.get_status()["worker_pending"]
        await owner.aclose()
        assert owner.state.value == "closed"
    finally:
        loop.set_task_factory(previous)
        await owner.aclose()


@pytest.mark.asyncio
async def test_cancelled_old_runner_cannot_stop_a_replacement():
    owner = evaluation.ContinuousMonitor()
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    samples = []

    async def old_response(messages):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()

    async def new_response(messages):
        samples.append("replacement")
        return "hello"

    environment = object()
    runner = asyncio.create_task(
        owner.start_monitoring(
            SimpleNamespace(generate_response=old_response), environment
        )
    )
    replacing = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        replacing = asyncio.create_task(
            owner.start(SimpleNamespace(generate_response=new_response), environment)
        )
        await asyncio.wait_for(cleaning.wait(), 2)
        runner.cancel()
        release.set()
        await replacing
        with pytest.raises(asyncio.CancelledError):
            await runner
        await wait_until(lambda: samples)
        assert owner.state.value == "running"
        assert owner.get_status()["worker_pending"]
    finally:
        release.set()
        await asyncio.gather(
            runner, *([replacing] if replacing else []), return_exceptions=True
        )
        await owner.aclose()


@pytest.mark.asyncio
async def test_unexpected_worker_cancellation_is_not_mistaken_for_successful_stop():
    owner = evaluation.ContinuousMonitor()

    async def response(messages):
        return "hello"

    await owner.start(SimpleNamespace(generate_response=response), object())
    task = owner._run.task
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    with pytest.raises(RuntimeError, match="unexpectedly"):
        await owner.stop_monitoring()
    assert owner.state.value == "failed"
    with pytest.raises(RuntimeError, match="unexpectedly"):
        await owner.aclose()


@pytest.mark.asyncio
async def test_legacy_runner_reports_unexpected_worker_cancellation():
    owner = evaluation.ContinuousMonitor()
    entered = asyncio.Event()

    async def response(messages):
        entered.set()
        return "hello"

    runner = asyncio.create_task(
        owner.start_monitoring(SimpleNamespace(generate_response=response), object())
    )
    await asyncio.wait_for(entered.wait(), 2)
    owner._run.task.cancel()
    with pytest.raises(RuntimeError, match="unexpectedly"):
        await runner
    with pytest.raises(RuntimeError, match="unexpectedly"):
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["monitor", "evaluator"])
async def test_rejected_close_task_keeps_admission_closed_and_permits_cleanup_retry(
    monkeypatch, kind
):
    owner = (
        evaluation.ContinuousMonitor()
        if kind == "monitor"
        else evaluation.AdvancedEvaluator()
    )
    failure = RuntimeError("close task rejected")
    operations = []

    def reject(operation):
        operations.append(operation)
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(asyncio, "create_task", reject)
        with pytest.raises(RuntimeError) as caught:
            await owner.aclose()
        assert caught.value is failure
    assert inspect.getcoroutinestate(operations[0]) == inspect.CORO_CLOSED
    assert owner.state.value == "closing"
    await owner.aclose()
    assert owner.state.value == "closed"


@pytest.mark.asyncio
async def test_cancellation_after_monitor_start_drains_committed_background_work(
    monkeypatch,
):
    owner = evaluation.AdvancedEvaluator()
    started, sampled, cleaning, release = (asyncio.Event() for _ in range(4))
    original_start = owner.monitor.start

    async def start(agent, environment):
        await original_start(agent, environment)
        started.set()
        await asyncio.Event().wait()

    async def suite(agent, environment, num_runs):
        return {"case": {"results": [{"passed": True, "execution_time": 0.1}]}}

    async def response(messages):
        sampled.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()

    monkeypatch.setattr(owner.monitor, "start", start)
    monkeypatch.setattr(owner.test_suite, "run_test_suite", suite)
    task = asyncio.create_task(
        owner.comprehensive_evaluation(
            [SimpleNamespace(generate_response=response)],
            object(),
            {"include_monitoring": True},
        )
    )
    try:
        await asyncio.wait_for(started.wait(), 2)
        await asyncio.wait_for(sampled.wait(), 2)
        task.cancel()
        await asyncio.wait_for(cleaning.wait(), 2)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert owner.monitor.state.value == "idle"
        assert not owner.monitor.get_status()["worker_pending"]
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await owner.aclose()
