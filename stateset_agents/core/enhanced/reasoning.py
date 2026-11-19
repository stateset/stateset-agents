"""Bounded preparation with explicit providers, assessments, and run ownership."""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import math
import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from threading import RLock
from typing import Any, TypeVar

from stateset_agents.utils.async_calls import run_async_owned

_OWNERS: ContextVar[frozenset[int]] = ContextVar(
    "reasoning_owners", default=frozenset()
)
_T = TypeVar("_T")


def _unit_score(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    if not 0 <= value <= 1 or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return float(value)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


@dataclass(frozen=True)
class ReasoningAssessment:
    """A checker's scoped declaration, not a calibrated probability of correctness.

    ``passed``/``failed`` require a score, named method, and supporting evidence.
    ``unavailable`` is an abstention and cannot satisfy a stopping threshold.
    """

    status: str = "unavailable"
    score: float | None = None
    source: str | None = None
    evidence: tuple[str, ...] = ()
    reason: str | None = "No assessment recorded"

    def __post_init__(self) -> None:
        if self.status not in {"passed", "failed", "unavailable"}:
            raise ValueError("Assessment status must be passed, failed, or unavailable")
        if not isinstance(self.evidence, (tuple, list)) or any(
            not isinstance(item, str) or not item.strip() for item in self.evidence
        ):
            raise ValueError("Assessment evidence must contain nonempty text")
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if self.source is not None:
            _text(self.source, "Assessment source")
        if self.reason is not None:
            _text(self.reason, "Assessment reason")
        if self.status == "unavailable":
            if self.score is not None:
                raise ValueError("Unavailable assessment cannot carry a score")
            _text(self.reason, "Abstention reason")
        else:
            object.__setattr__(
                self, "score", _unit_score(self.score, "Assessment score")
            )
            _text(self.source, "Assessment source")
            if not self.evidence:
                raise ValueError("An assessed result requires supporting evidence")
            if self.reason == "No assessment recorded":
                object.__setattr__(self, "reason", None)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible assessment metadata."""
        return {
            "status": self.status,
            "score": self.score,
            "source": self.source,
            "evidence": list(self.evidence),
            "reason": self.reason,
        }


@dataclass
class ReasoningStep:
    """A concise proposal or planning instruction with separately assessed evidence.

    The legacy ``confidence`` field is a provider declaration; it never authorizes
    completion. Template steps use zero as a compatibility value and report that
    confidence is unavailable. Editing a step invalidates its bound assessment.
    """

    thought: str
    action: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    origin: str = "custom"
    run_id: str | None = None
    step_index: int | None = None
    _assessment: ReasoningAssessment | None = field(
        default=None, init=False, repr=False
    )
    _assessed_content: tuple[Any, ...] | None = field(
        default=None, init=False, repr=False
    )

    def _content(self) -> tuple[Any, ...]:
        return (
            self.thought,
            self.action,
            self.confidence,
            tuple(self.evidence),
            self.origin,
            self.run_id,
            self.step_index,
        )

    def validate(self) -> None:
        """Reject malformed proposals before assessment or prompt preparation."""
        _text(self.thought, "Reasoning thought")
        _text(self.action, "Reasoning action")
        _unit_score(self.confidence, "Declared confidence")
        if not isinstance(self.evidence, list) or any(
            not isinstance(item, str) or not item.strip() for item in self.evidence
        ):
            raise ValueError("Reasoning evidence must be a list of nonempty text")
        if self.origin not in {"template", "provider", "custom"}:
            raise ValueError("Unknown reasoning origin")
        if self.run_id is not None:
            _text(self.run_id, "Reasoning run ID")
        if self.step_index is not None and (
            type(self.step_index) is not int or self.step_index < 0
        ):
            raise ValueError("Reasoning step index must be a nonnegative integer")

    def assessment(self) -> ReasoningAssessment:
        """Read an assessment only while the exact assessed content is unchanged."""
        assessment: ReasoningAssessment | None = getattr(self, "_assessment", None)
        if assessment is None:
            return ReasoningAssessment()
        try:
            self.validate()
            unchanged = self._content() == self._assessed_content
        except (TypeError, ValueError):
            unchanged = False
        if not unchanged:
            return ReasoningAssessment(reason="Step changed after assessment")
        return assessment

    def _bind(self, assessment: ReasoningAssessment) -> None:
        self._assessment = assessment
        self._assessed_content = self._content()

    def to_dict(self) -> dict[str, Any]:
        """Return detached proposal and assessment data for status and prompts."""
        return {
            "thought": self.thought,
            "action": self.action,
            "confidence": self.confidence,
            "confidence_status": (
                "unavailable" if self.origin == "template" else "declared"
            ),
            "evidence": list(self.evidence),
            "origin": self.origin,
            "run_id": self.run_id,
            "step_index": self.step_index,
            "assessment": self.assessment().to_dict(),
        }


@dataclass(frozen=True)
class ReasoningRequest:
    """Detached input for one provider/checker invocation."""

    query: str
    reasoning_type: str
    step_index: int
    context: dict[str, Any]
    previous_steps: tuple[ReasoningStep, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return detached, structured input for a model-backed provider."""
        return {
            "query": self.query,
            "reasoning_type": self.reasoning_type,
            "step_index": self.step_index,
            "context": copy.deepcopy(self.context),
            "previous_steps": [step.to_dict() for step in self.previous_steps],
        }


StepGenerator = Callable[[ReasoningRequest], Awaitable[ReasoningStep]]
StepAssessor = Callable[
    [ReasoningRequest, ReasoningStep], Awaitable[ReasoningAssessment]
]
TextGenerator = Callable[[str, dict[str, Any] | None], Awaitable[str]]


class ModelReasoningProvider:
    """Adapt an explicitly supplied text backend to concise JSON proposals.

    Use a separate agent/backend to avoid recursively invoking the same enhanced
    agent. This makes a model call per step; it does not provide an assessor.
    Provider confidence and citations remain unverified declarations.
    """

    def __init__(self, generate: TextGenerator) -> None:
        if not callable(generate):
            raise TypeError("A callable text generator is required")
        self.generate = generate

    async def __call__(self, request: ReasoningRequest) -> ReasoningStep:
        prompt = (
            "Produce one concise, user-visible proposal for the request below. "
            "Return only a JSON object with thought (brief proposed conclusion), "
            "action (next action), confidence (number 0 to 1), and evidence "
            "(list of supporting references supplied in the request; empty if absent). "
            "Do not include private reasoning or invent supporting references. "
            "A previous assessment may reject a proposal; address its stated reason.\n\n"
            + json.dumps(request.to_dict(), ensure_ascii=False, allow_nan=False)
        )
        response = await self.generate(prompt, copy.deepcopy(request.context))

        def unique_fields(items: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in items:
                if key in result:
                    raise ValueError(f"Duplicate reasoning field: {key}")
                result[key] = value
            return result

        data = json.loads(response, object_pairs_hook=unique_fields)
        if not isinstance(data, dict) or set(data) != {
            "thought",
            "action",
            "confidence",
            "evidence",
        }:
            raise ValueError(
                "Reasoning JSON requires thought, action, confidence, and evidence"
            )
        step = ReasoningStep(**data, origin="provider")
        step.validate()
        return step


@dataclass
class _Run:
    mode: str
    max_steps: int
    threshold: float
    query: str
    assessor_configured: bool
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    state: str = "running"
    phase: str = "preparing"
    stop_reason: str | None = None
    error_type: str | None = None
    steps: list[ReasoningStep] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mode": self.mode,
            "max_steps": self.max_steps,
            "threshold": self.threshold,
            "query": self.query,
            "assessor_configured": self.assessor_configured,
            "state": self.state,
            "phase": self.phase,
            "stop_reason": self.stop_reason,
            "error_type": self.error_type,
            "steps": [step.to_dict() for step in self.steps],
        }


async def _invoke(callback: Callable[..., Awaitable[_T]], *args: Any) -> _T:
    result = callback(*args)
    if not inspect.isawaitable(result):
        raise TypeError("Reasoning providers and assessors must return awaitables")
    return await result


class ReasoningEngine:
    """Bounded preparation whose stopping decision requires a scoped assessment.

    Without a provider this emits planning templates with unavailable confidence.
    A provider can propose actual conclusions; an assessor evaluates them against
    its declared method. A passed assessment at the captured threshold permits
    early stopping. Exhausting the step budget does not establish correctness.
    """

    def __init__(
        self,
        max_steps: int = 5,
        confidence_threshold: float = 0.7,
        *,
        step_generator: StepGenerator | None = None,
        assessor: StepAssessor | None = None,
    ) -> None:
        self.max_steps = max_steps
        self.confidence_threshold = confidence_threshold
        self.step_generator = step_generator
        self.assessor = assessor
        self.reasoning_templates = {
            "analytical": "Identify the relevant facts and relationships.",
            "creative": "Propose alternatives that respect the request's constraints.",
            "problem_solving": "Identify the problem and the evidence needed to diagnose it.",
            "decision_making": "Identify the available options and decision criteria.",
        }
        self._active: dict[str, _Run] = {}
        self._last_run: dict[str, Any] | None = None
        self._lock = RLock()
        self._validate_config()

    def __getstate__(self) -> dict[str, Any]:
        with self._lock:
            if self._active:
                raise RuntimeError(
                    "Cannot copy a reasoning engine during an active run"
                )
            return {key: value for key, value in vars(self).items() if key != "_lock"}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.step_generator = state.get("step_generator")
        self.assessor = state.get("assessor")
        self._active = {}
        self._last_run = state.get("_last_run")
        self._lock = RLock()

    def _validate_config(self) -> None:
        if type(self.max_steps) is not int or self.max_steps < 0:
            raise ValueError("max_steps must be a nonnegative integer")
        _unit_score(self.confidence_threshold, "confidence_threshold")
        for name in ("step_generator", "assessor"):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f"{name} must be callable or None")

    def _mode(self) -> str:
        if self.step_generator is not None:
            return "provider"
        if (
            getattr(self._generate_reasoning_step, "__func__", None)
            is not ReasoningEngine._generate_reasoning_step
        ):
            return "custom"
        return "template"

    def get_status(self) -> dict[str, Any]:
        """Inspect detached active/latest runs and configured provider capabilities."""
        with self._lock:
            return copy.deepcopy(
                {
                    "mode": self._mode(),
                    "assessor_configured": self.assessor is not None,
                    "active": [run.snapshot() for run in self._active.values()],
                    "last_run": self._last_run,
                }
            )

    async def reason(
        self, query: str, context: dict[str, Any] | None = None
    ) -> list[ReasoningStep]:
        """Prepare proposals with captured settings and own every callback through cleanup."""
        self._validate_config()
        if not isinstance(query, str):
            raise TypeError("Reasoning query must be text")
        if context is not None and not isinstance(context, dict):
            raise TypeError("Reasoning context must be a dictionary or None")
        if id(self) in _OWNERS.get():
            raise RuntimeError(
                "Reasoning provider reentered its engine; use a separate backend"
            )
        provider, assessor = self.step_generator, self.assessor
        generate_step: Callable[..., Awaitable[ReasoningStep]] = (
            self._generate_reasoning_step
        )
        run = _Run(
            self._mode(),
            self.max_steps,
            float(self.confidence_threshold),
            query,
            assessor is not None,
        )
        with self._lock:
            self._active[run.id] = run
        owner = _OWNERS.set(_OWNERS.get() | {id(self)})
        try:
            if run.mode == "template":
                generate_step = partial(
                    self._template_step, copy.deepcopy(self.reasoning_templates)
                )
            current_context = copy.deepcopy(context or {})
            reasoning_type = self._classify_reasoning_type(query)
            for index in range(run.max_steps):
                request = ReasoningRequest(
                    query,
                    reasoning_type,
                    index,
                    copy.deepcopy(current_context),
                    tuple(copy.deepcopy(run.steps)),
                )
                run.phase = "generating"
                if provider is not None:
                    step = await run_async_owned(
                        _invoke(provider, copy.deepcopy(request))
                    )
                else:
                    step = await run_async_owned(
                        _invoke(
                            generate_step,
                            query,
                            reasoning_type,
                            index,
                            list(copy.deepcopy(run.steps)),
                            copy.deepcopy(current_context),
                        )
                    )
                if not isinstance(step, ReasoningStep):
                    raise TypeError("Reasoning provider must return ReasoningStep")
                step = copy.deepcopy(step)
                step.origin = "template" if run.mode == "template" else run.mode
                step.run_id, step.step_index = run.id, index
                step.validate()
                step._bind(ReasoningAssessment())
                run.steps.append(step)
                if assessor is not None:
                    run.phase = "assessing"
                    assessment = await run_async_owned(
                        _invoke(
                            assessor,
                            copy.deepcopy(request),
                            copy.deepcopy(step),
                        )
                    )
                    if not isinstance(assessment, ReasoningAssessment):
                        raise TypeError(
                            "Reasoning assessor must return ReasoningAssessment"
                        )
                    assessment = copy.deepcopy(assessment)
                    assessment.__post_init__()
                    step._bind(assessment)
                    if (
                        assessment.status == "passed"
                        and assessment.score is not None
                        and assessment.score >= run.threshold
                    ):
                        run.stop_reason = "assessment_threshold"
                        break
                if run.mode == "template" and index == 4:
                    run.stop_reason = "template_complete"
                    break
                current_context.update(
                    {"step": index + 1, "last_thought": step.thought}
                )
            run.stop_reason = run.stop_reason or "step_limit"
            run.state, run.phase = "completed", "finished"
            return copy.deepcopy(run.steps)
        except BaseException as error:
            run.state = (
                "cancelled" if isinstance(error, asyncio.CancelledError) else "failed"
            )
            run.error_type = type(error).__name__
            raise
        finally:
            _OWNERS.reset(owner)
            with self._lock:
                self._active.pop(run.id, None)
                self._last_run = run.snapshot()

    def _classify_reasoning_type(self, query: str) -> str:
        query_lower = query.lower()
        if any(word in query_lower for word in ["analyze", "compare", "evaluate"]):
            return "analytical"
        if any(word in query_lower for word in ["create", "design", "generate"]):
            return "creative"
        if any(word in query_lower for word in ["solve", "fix", "troubleshoot"]):
            return "problem_solving"
        return "decision_making"

    async def _generate_reasoning_step(
        self,
        query: str,
        reasoning_type: str,
        step_num: int,
        previous_steps: list[ReasoningStep],
        context: dict[str, Any],
    ) -> ReasoningStep:
        """Supply planning instructions without claiming analysis has been performed."""
        return await self._template_step(
            self.reasoning_templates,
            query,
            reasoning_type,
            step_num,
            previous_steps,
            context,
        )

    @staticmethod
    async def _template_step(
        templates: dict[str, str],
        query: str,
        reasoning_type: str,
        step_num: int,
        previous_steps: list[ReasoningStep],
        context: dict[str, Any],
    ) -> ReasoningStep:
        instructions = [
            templates.get(reasoning_type, templates["analytical"]),
            "Identify missing information and distinguish supplied facts from assumptions.",
            "Compare possible answers against the supplied constraints and references.",
            "Check the proposed answer against independently available evidence.",
            "Summarize the supported conclusion and state any unresolved uncertainty.",
        ]
        return ReasoningStep(
            instructions[min(step_num, len(instructions) - 1)],
            f"Plan {reasoning_type} work",
            0.0,
            origin="template",
        )
