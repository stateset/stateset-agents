"""Reasoning completion follows assessed evidence, never a scheduled confidence score."""

import asyncio
import copy
import json
import pickle
from concurrent.futures import ThreadPoolExecutor

import pytest

from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.enhanced import (
    EnhancedMultiTurnAgent,
    ModelReasoningProvider,
    ReasoningAssessment,
    ReasoningEngine,
    ReasoningRequest,
    ReasoningStep,
    create_enhanced_agent,
)

from .test_enhanced_generation import ObservedMemory


def assessed(status="passed", score=1.0):
    return ReasoningAssessment(
        status, score, "refund_window_rule/v1", ("policy.refund_days=30",)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "limit, count, reason",
    [(0, 0, "step_limit"), (2, 2, "step_limit"), (10, 5, "template_complete")],
)
async def test_templates_do_not_invent_confidence_or_evidence(limit, count, reason):
    engine = ReasoningEngine(max_steps=limit, confidence_threshold=0)
    steps = await engine.reason("Explain the refund policy")
    assert len(steps) == count
    assert all(step.confidence == 0 and step.evidence == [] for step in steps)
    assert all(step.to_dict()["confidence_status"] == "unavailable" for step in steps)
    assert all(step.assessment().status == "unavailable" for step in steps)
    status = engine.get_status()
    assert status["mode"] == "template"
    assert status["last_run"]["stop_reason"] == reason
    assert status["last_run"]["state"] == "completed"


@pytest.mark.asyncio
async def test_declared_confidence_and_provider_supplied_assessment_cannot_stop_run():
    calls = []

    async def propose(request: ReasoningRequest):
        calls.append(request.step_index)
        step = ReasoningStep("unsupported conclusion", "answer", 1.0)
        step._bind(assessed())  # Providers do not get to assess their own output.
        return step

    engine = ReasoningEngine(3, 0, step_generator=propose)
    steps = await engine.reason("Explain refunds")
    assert calls == [0, 1, 2]
    assert all(step.assessment().status == "unavailable" for step in steps)
    assert engine.get_status()["last_run"]["stop_reason"] == "step_limit"


@pytest.mark.asyncio
async def test_rejected_proposal_is_revised_against_independent_policy_check():
    supplied = {"policy": {"refund_days": 30}}
    requests, checks = [], []

    async def propose(request):
        requests.append(copy.deepcopy(request))
        days = (
            60
            if not request.previous_steps
            else request.context["policy"]["refund_days"]
        )
        request.context["policy"]["refund_days"] = 999
        if request.previous_steps:
            assert request.previous_steps[0].assessment().status == "failed"
        return ReasoningStep(
            f"Refund window: {days} days", "answer", 1.0, [f"policy says {days}"]
        )

    async def check(request, step):
        expected = f"Refund window: {request.context['policy']['refund_days']} days"
        valid = step.thought == expected
        checks.append(valid)
        # Mutating checker inputs cannot rewrite the proposal being assessed.
        step.thought = "checker mutation"
        request.context["policy"]["refund_days"] = -1
        return assessed("passed" if valid else "failed", 1.0)

    engine = ReasoningEngine(step_generator=propose, assessor=check)
    steps = await engine.reason("Explain refunds", supplied)
    assert checks == [False, True]
    assert [step.thought for step in steps] == [
        "Refund window: 60 days",
        "Refund window: 30 days",
    ]
    assert engine.get_status()["last_run"]["stop_reason"] == "assessment_threshold"
    assert supplied == {"policy": {"refund_days": 30}}
    assert all(request.context["policy"]["refund_days"] == 30 for request in requests)
    snapshot = engine.get_status()
    snapshot["last_run"]["steps"][0]["thought"] = "forged"
    assert engine.get_status()["last_run"]["steps"][0]["thought"] != "forged"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["thought", "action", "confidence", "evidence", "origin"]
)
async def test_assessment_is_bound_to_unchanged_step_content(change):
    async def propose(request):
        return ReasoningStep("Refund window: 30 days", "answer", 0.1)

    async def check(request, step):
        return assessed()

    engine = ReasoningEngine(step_generator=propose, assessor=check)
    step = (await engine.reason("refund"))[0]
    assert step.assessment().status == "passed"
    if change == "evidence":
        step.evidence.append("new reference")
    else:
        setattr(
            step,
            change,
            {"thought": "new", "action": "new", "confidence": 0.2, "origin": "custom"}[
                change
            ],
        )
    assert step.assessment().status == "unavailable"
    assert (
        engine.get_status()["last_run"]["steps"][0]["assessment"]["status"] == "passed"
    )


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True, -0.1, 1.1, "0.9"])
def test_assessment_rejects_invalid_scores(score):
    with pytest.raises(ValueError):
        assessed(score=score)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"status": "success"},
        {"status": "unavailable", "score": 0.9},
        {"status": "passed", "score": 1, "source": "method"},
        {"status": "passed", "score": 1, "source": "", "evidence": ["reference"]},
        {"status": "unavailable", "reason": ""},
    ],
)
def test_assessment_requires_explicit_method_and_evidence(kwargs):
    with pytest.raises(ValueError):
        ReasoningAssessment(**kwargs)


@pytest.mark.parametrize(
    "settings",
    [
        {"max_steps": -1},
        {"max_steps": True},
        {"max_steps": 1.5},
        {"confidence_threshold": float("nan")},
        {"confidence_threshold": 1.1},
        {"step_generator": "not a callback"},
        {"assessor": "not a callback"},
    ],
)
def test_invalid_settings_are_rejected_before_callbacks(settings):
    with pytest.raises((TypeError, ValueError)):
        ReasoningEngine(**settings)


@pytest.mark.asyncio
async def test_factory_accepts_configured_reasoning_without_forwarding_it_to_model_config():
    calls = []

    async def propose(request):
        calls.append(request.query)
        return ReasoningStep("proposal", "check", 0.1)

    engine = ReasoningEngine(1, step_generator=propose)
    agent = create_enhanced_agent(
        model_name="stub://configured",
        memory_enabled=False,
        reasoning_engine=engine,
    )
    await agent.initialize()
    await agent.generate_turn("Explain refunds")
    assert agent.reasoning_engine is engine
    assert calls == ["Explain refunds"]


@pytest.mark.asyncio
async def test_malformed_model_preparation_cannot_reach_answer_generation(monkeypatch):
    async def invalid_json(prompt, context):
        return "invalid JSON"

    agent = create_enhanced_agent(
        model_name="stub://invalid-preparation",
        memory_enabled=False,
        reasoning_engine=ReasoningEngine(
            step_generator=ModelReasoningProvider(invalid_json)
        ),
    )
    await agent.initialize()

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid preparation must not enter answer generation")

    monkeypatch.setattr(agent.model, "generate", forbidden)
    with pytest.raises(ValueError):
        await agent.generate_response("Explain refunds")
    assert agent.get_generation_status()["last_finished"]["state"] == "failed"
    assert agent.turn_count == 0


@pytest.mark.asyncio
async def test_abstention_does_not_become_a_successful_score():
    async def propose(request):
        return ReasoningStep("Need current policy", "request policy", 0.9)

    async def check(request, step):
        return ReasoningAssessment(
            source="policy_check/v1", reason="No policy supplied"
        )

    engine = ReasoningEngine(2, step_generator=propose, assessor=check)
    steps = await engine.reason("refund")
    assert len(steps) == 2
    assert all(step.assessment().score is None for step in steps)
    assert engine.get_status()["last_run"]["stop_reason"] == "step_limit"


@pytest.mark.asyncio
async def test_assessor_failure_retains_generated_proposal_without_claiming_assessment():
    async def propose(request):
        return ReasoningStep("proposal", "check", 0.8)

    async def check(request, step):
        raise OSError("checker unavailable")

    engine = ReasoningEngine(step_generator=propose, assessor=check)
    with pytest.raises(OSError, match="checker unavailable"):
        await engine.reason("refund")
    run = engine.get_status()["last_run"]
    assert run["state"] == "failed" and run["phase"] == "assessing"
    assert run["steps"][0]["thought"] == "proposal"
    assert run["steps"][0]["assessment"]["status"] == "unavailable"
    assert run["steps"][0]["assessment"]["reason"] == "No assessment recorded"


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["provider", "assessor"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_cancelled_reasoning_owns_suppressed_and_repeated_cancellation(
    stage, cleanup_fails
):
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = []

    async def blocked():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleaning.set()
            await release.wait()
            if cleanup_fails:
                raise RuntimeError("cleanup failed") from None

    async def propose(request):
        calls.append("provider")
        if stage == "provider":
            await blocked()
        return ReasoningStep("proposal", "check", 0.9)

    async def check(request, step):
        calls.append("assessor")
        await blocked()
        return assessed()

    engine = ReasoningEngine(step_generator=propose, assessor=check)
    task = asyncio.create_task(engine.reason("refund"))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        await asyncio.wait_for(cleaning.wait(), 2)
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        assert engine.get_status()["active"][0]["state"] == "running"
        with pytest.raises(RuntimeError, match="active run"):
            copy.deepcopy(engine)
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
    run = engine.get_status()["last_run"]
    assert run["state"] == "cancelled"
    assert run["stop_reason"] is None
    assert len(run["steps"]) == (0 if stage == "provider" else 1)
    assert calls == (["provider"] if stage == "provider" else ["provider", "assessor"])
    assert engine.get_status()["active"] == []


@pytest.mark.asyncio
async def test_settings_are_captured_and_callbacks_can_inspect_status_from_thread():
    calls = []
    with ThreadPoolExecutor(max_workers=1) as executor:

        async def should_not_run(request):
            pytest.fail("replacement provider must wait for a new run")

        async def propose(request):
            calls.append(request.step_index)
            status = executor.submit(engine.get_status).result(timeout=2)
            assert len(status["active"]) == 1
            engine.max_steps = 0
            engine.step_generator = should_not_run
            return ReasoningStep("proposal", "check", 1.0)

        engine = ReasoningEngine(2, step_generator=propose)
        await engine.reason("refund")
    assert calls == [0, 1]
    assert engine.get_status()["last_run"]["max_steps"] == 2
    assert await engine.reason("refund") == []


@pytest.mark.asyncio
async def test_recursive_provider_is_rejected_before_another_provider_call():
    calls = []

    async def propose(request):
        calls.append(request.query)
        await engine.reason(request.query)

    engine = ReasoningEngine(step_generator=propose)
    with pytest.raises(RuntimeError, match="reentered"):
        await engine.reason("refund")
    assert calls == ["refund"]
    assert engine.get_status()["last_run"]["state"] == "failed"


@pytest.mark.asyncio
async def test_concurrent_runs_keep_their_own_requests_and_receipts():
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def propose(request):
        calls.append(request.query)
        if len(calls) == 2:
            entered.set()
        await release.wait()
        return ReasoningStep(request.query, "check", 0)

    engine = ReasoningEngine(1, step_generator=propose)
    tasks = [asyncio.create_task(engine.reason(query)) for query in ("one", "two")]
    try:
        await asyncio.wait_for(entered.wait(), 2)
        active = engine.get_status()["active"]
        assert len(active) == 2 and len({run["id"] for run in active}) == 2
    finally:
        release.set()
        results = await asyncio.gather(*tasks)
    assert [steps[0].thought for steps in results] == ["one", "two"]
    assert {steps[0].run_id for steps in results} == {run["id"] for run in active}
    assert {run["query"] for run in active} == {"one", "two"}


@pytest.mark.asyncio
async def test_template_settings_are_captured_before_callbacks_yield(monkeypatch):
    engine = ReasoningEngine(2)
    original = engine._template_step

    async def change_templates(templates, *args):
        engine.reasoning_templates["decision_making"] = "late mutation"
        return await original(templates, *args)

    monkeypatch.setattr(engine, "_template_step", change_templates)
    steps = await engine.reason("refund")
    assert steps[0].thought == "Identify the available options and decision criteria."


@pytest.mark.asyncio
async def test_legacy_step_override_is_reported_as_custom_and_not_implicitly_assessed():
    class CustomEngine(ReasoningEngine):
        async def _generate_reasoning_step(self, *args):
            return ReasoningStep("custom proposal", "check", 1.0)

    engine = CustomEngine(2)
    steps = await engine.reason("refund")
    assert len(steps) == 2
    assert all(step.origin == "custom" for step in steps)
    assert engine.get_status()["mode"] == "custom"


@pytest.mark.asyncio
async def test_legacy_duck_typed_reasoning_engine_can_still_report_agent_status():
    class CustomEngine:
        async def reason(self, query, context):
            return [ReasoningStep("custom proposal", "check", 0.9)]

    agent = EnhancedMultiTurnAgent(
        AgentConfig(model_name="stub://custom"),
        memory_enabled=False,
        reasoning_engine=CustomEngine(),
    )
    await agent.initialize()
    await agent.generate_response("Explain refunds")
    assert agent.get_agent_status()["reasoning_backend"] == {
        "mode": "custom",
        "status": "unavailable",
    }


@pytest.mark.asyncio
async def test_nested_agent_cancellation_cannot_reach_sampling_or_memory(monkeypatch):
    entered, release, cleaning = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def propose(request):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleaning.set()
            await release.wait()
        return ReasoningStep("late proposal", "check", 0.9)

    memory = ObservedMemory()
    engine = ReasoningEngine(step_generator=propose)
    agent = EnhancedMultiTurnAgent(
        AgentConfig(model_name="stub://cancel"),
        memory_system=memory,
        reasoning_engine=engine,
    )
    await agent.initialize()

    def forbidden(*args, **kwargs):
        pytest.fail("Cancelled preparation must not enter sampling")

    monkeypatch.setattr(agent.model, "generate", forbidden)
    task = asyncio.create_task(agent.generate_response("Explain refunds"))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        await asyncio.wait_for(cleaning.wait(), 2)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
    assert memory.writes == []
    assert engine.get_status()["last_run"]["state"] == "cancelled"
    assert agent.get_generation_status()["last_finished"]["state"] == "cancelled"


@pytest.mark.asyncio
async def test_old_imports_and_legacy_pickle_state_remain_readable():
    from stateset_agents.core.enhanced.enhanced_agent import (
        ReasoningEngine as OldEngine,
    )
    from stateset_agents.core.enhanced.enhanced_agent import ReasoningStep as OldStep

    assert OldStep is ReasoningStep and OldEngine is ReasoningEngine
    engine = ReasoningEngine(2)
    await engine.reason("refund")
    restored = pickle.loads(pickle.dumps(engine))
    assert restored.get_status()["last_run"] == engine.get_status()["last_run"]
    legacy = ReasoningEngine.__new__(ReasoningEngine)
    legacy.__setstate__(
        {
            "max_steps": 1,
            "confidence_threshold": 0.7,
            "reasoning_templates": engine.reasoning_templates,
        }
    )
    assert len(await legacy.reason("refund")) == 1
    step = ReasoningStep("legacy", "check", 0.99)
    for name in ("origin", "_assessment", "_assessed_content"):
        step.__dict__.pop(name, None)
    assert pickle.loads(pickle.dumps(step)).assessment().status == "unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        "not JSON",
        "[]",
        '{"thought":"x","action":"check","confidence":NaN,"evidence":[]}',
        '{"thought":"x","thought":"y","action":"check","confidence":0.9,"evidence":[]}',
        '{"thought":"x","action":"check","confidence":0.9,"evidence":[],"assessment":"passed"}',
    ],
)
async def test_model_provider_rejects_malformed_or_ambiguous_output(raw):
    async def generate(prompt, context):
        return raw

    engine = ReasoningEngine(step_generator=ModelReasoningProvider(generate))
    with pytest.raises(ValueError):
        await engine.reason("refund")
    assert engine.get_status()["last_run"]["state"] == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["response", "turn", "batch", "stream"])
async def test_model_preparation_and_scoped_assessment_reach_generation_prompts(
    monkeypatch, surface
):
    planner = MultiTurnAgent(
        AgentConfig(
            model_name="stub://planner",
            stub_responses=[
                json.dumps(
                    {
                        "thought": "Refund window: 30 days",
                        "action": "answer",
                        "confidence": 0.95,
                        "evidence": ["refund policy"],
                    }
                )
            ],
        )
    )
    await planner.initialize()
    checked = []

    async def check(request, step):
        checked.append(request.context["relevant_memories"])
        return (
            assessed()
            if step.thought == "Refund window: 30 days"
            else assessed("failed", 0)
        )

    engine = ReasoningEngine(
        step_generator=ModelReasoningProvider(planner.generate_response), assessor=check
    )
    memory = ObservedMemory()
    agent = EnhancedMultiTurnAgent(
        AgentConfig(model_name="stub://answer", stub_responses=["Thirty days."]),
        memory_system=memory,
        reasoning_engine=engine,
    )
    await agent.initialize()
    await memory.add_entry("Refund policy: 30 days")
    memory.writes.clear()
    prompts = []
    generate = agent.model.generate

    def capture(prompt, context=None):
        prompts.append((prompt, copy.deepcopy(context)))
        return generate(prompt, context)

    monkeypatch.setattr(agent.model, "generate", capture)
    if surface == "response":
        await agent.generate_response("Explain refund policy")
    elif surface == "turn":
        await agent.generate_turn("Explain refund policy")
    elif surface == "batch":
        await agent.generate_turns("Explain refund policy", 2)
    else:
        assert [
            part
            async for part in agent.generate_response_stream("Explain refund policy")
        ]
    assert checked == [["Refund policy: 30 days"]]
    assert planner.turn_count == 1
    assert all("Refund window: 30 days" in prompt for prompt, _ in prompts)
    assert all("refund_window_rule/v1" in prompt for prompt, _ in prompts)
    assert prompts[0][1]["reasoning_evidence"][0]["assessment"]["status"] == "passed"
    assert (
        agent.get_generation_status()["last_finished"]["reasoning_steps"][0]["origin"]
        == "provider"
    )
    assert agent.get_agent_status()["reasoning_backend"]["mode"] == "provider"
