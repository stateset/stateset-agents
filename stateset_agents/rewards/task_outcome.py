"""Deterministic outcome checks for reference-labelled agent workflows.

:class:`TaskOutcomeReward` compares the last assistant turn with the
scenario's trusted reference answer(s) and, optionally, the executed tool
trace with ``expected_tool_calls``.

**Reference answers.** ``expected_response`` (a string) and/or
``expected_responses`` (a list of acceptable answers); any one matching is
agreement. At least one non-empty reference is required.

**Match modes** (scenario key ``expected_response_match``):

* ``"normalized"`` (default): equal after :func:`normalize_answer`, which
  ignores case, whitespace, apostrophes and punctuation outside numbers. Words,
  numbers and currency, percent and degree signs must agree token for token,
  so "Resolved." matches "resolved", but "Not resolved", "84" against "840",
  "-5" against "5" and "$5" against "5%" do not.
* ``"exact"``: equal after stripping leading and trailing whitespace (the
  behavior before ``expected_response_match`` existed).
* ``"contains"``: the reference's normalized tokens occur as a contiguous run
  of the reply's ("The answer is 84." contains "84"; "840" does not). A reply
  that lists several candidates ("84 or 85") also contains the reference, so
  use it only where hedging cannot pay.
* ``"regex"``: the stripped reply fully matches the reference, read as a
  Python regular expression (``re.fullmatch``; use ``(?i)`` for case).
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from stateset_agents.core.errors import ConfigurationError
from stateset_agents.core.reward_base import RewardFunction, RewardResult, RewardType
from stateset_agents.core.trajectory import ConversationTurn

#: ``expected_response_match`` values; the first is the default.
MATCH_MODES = ("normalized", "exact", "contains", "regex")

_APOSTROPHES = re.compile(r"['\u2019]")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
#: Words, numbers (with a leading minus and decimal points) and the unit
#: symbols that change an answer: $ € £ ¥ ₹ % °.
_ANSWER_TOKEN = re.compile(
    r"-?\d+(?:\.\d+)*|[^\W_]+|[$\u20ac\u00a3\u00a5\u20b9%\u00b0]"
)
_MISSING_REFERENCE = "Outcome rewards require a trusted expected_response reference"


def normalize_answer(text: str) -> tuple[str, ...]:
    """The tokens of ``text`` compared by the ``normalized`` match mode.

    NFKC and case folding, apostrophes removed ("can't" is "cant"), thousands
    separators removed ("1,299" is "1299"); then words, numbers (keeping a
    leading minus and decimal points) and currency, percent and degree signs
    in order, all other punctuation and whitespace dropped.
    """
    folded = unicodedata.normalize("NFKC", text).replace("\u2212", "-").casefold()
    folded = _THOUSANDS.sub("", _APOSTROPHES.sub("", folded))
    return tuple(_ANSWER_TOKEN.findall(folded))


def _contains(answer: tuple[str, ...], reference: tuple[str, ...]) -> bool:
    size = len(reference)
    return any(
        answer[start : start + size] == reference
        for start in range(len(answer) - size + 1)
    )


def _check_reference(reference: str, mode: str) -> None:
    """Raise ``ConfigurationError`` for a reference ``mode`` cannot match."""
    if mode not in MATCH_MODES:
        raise ConfigurationError(
            f"expected_response_match must be one of {MATCH_MODES}, got {mode!r}"
        )
    if mode == "regex":
        try:
            re.compile(reference)
        except re.error as error:
            raise ConfigurationError(
                f"expected_response {reference!r} is not a valid regular "
                f"expression: {error}"
            ) from error
    elif mode != "exact" and not normalize_answer(reference):
        raise ConfigurationError(
            f"expected_response {reference!r} has no words or numbers to match "
            f"in {mode!r} mode"
        )


def answer_matches(answer: str, reference: str, mode: str = "normalized") -> bool:
    """Whether ``answer`` agrees with ``reference`` under ``mode``
    (:data:`MATCH_MODES`; see the module docs). Raises
    ``ConfigurationError`` for an unknown mode or a reference the mode cannot
    match (an invalid pattern, or no words or numbers)."""
    _check_reference(reference, mode)
    if mode == "exact":
        return answer.strip() == reference.strip()
    if mode == "regex":
        return re.fullmatch(reference, answer.strip()) is not None
    if mode == "contains":
        return _contains(normalize_answer(answer), normalize_answer(reference))
    return normalize_answer(answer) == normalize_answer(reference)


def _references(scenario: Mapping[str, Any]) -> list[str]:
    """The trusted reference answers: ``expected_response`` and
    ``expected_responses``; raises when there is none."""
    references: list[str] = []
    single = scenario.get("expected_response")
    if single is not None:
        if not isinstance(single, str):
            raise ValueError(_MISSING_REFERENCE)
        if single.strip():
            references.append(single)
    many = scenario.get("expected_responses")
    if many is not None:
        if not isinstance(many, (list, tuple)) or not all(
            isinstance(item, str) and item.strip() for item in many
        ):
            raise ConfigurationError(
                "expected_responses must be a list of non-empty strings"
            )
        references.extend(many)
    if not references:
        raise ValueError(_MISSING_REFERENCE)
    return references


class TaskOutcomeReward(RewardFunction):
    """Score a trusted reference and executed tool trace, never helpful phrasing.

    References belong to the evaluator's scenario; callers must keep them out of
    observations. Missing references are errors, not neutral training rewards.
    The reply agrees with the reference under the scenario's
    ``expected_response_match`` mode (default ``"normalized"``; see the
    module docs).
    """

    fail_on_error = True
    requires_environment_context = True

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(weight=weight, reward_type=RewardType.SPARSE)

    async def compute_reward(
        self, turns: list[ConversationTurn], context: dict[str, Any] | None = None
    ) -> RewardResult:
        """Require final reference agreement and optional verified tool outcomes."""
        context = context or {}
        scenario = context.get("scenario", context)
        references = _references(scenario)
        mode = scenario.get("expected_response_match", MATCH_MODES[0])
        for reference in references:
            # A bad mode or reference fails even when there is no reply yet.
            _check_reference(reference, mode)
        assistant = [turn for turn in turns if turn.role == "assistant"]
        final = (assistant[-1].content or "") if assistant else None
        answer_matches_reference = final is not None and any(
            answer_matches(final, reference, mode) for reference in references
        )
        violations = context.get("policy_violations", 0)
        if type(violations) is not int or violations < 0:
            raise ValueError("policy_violations must be a nonnegative integer")
        expected_tools = scenario.get("expected_tool_calls")
        tools_match = True
        if expected_tools is not None:
            # Only environment-executed records count; generated tool text does not.
            executed = context.get("executed_tool_calls")
            if not isinstance(expected_tools, list) or not isinstance(executed, list):
                raise ValueError("Tool outcome rewards require an executed tool trace")
            # JSON identity preserves argument types: True must not equal 1,
            # and a float amount must not equal an integer-cent amount.
            tools_match = json.dumps(
                executed, sort_keys=True, allow_nan=False
            ) == json.dumps(expected_tools, sort_keys=True, allow_nan=False)
        success = answer_matches_reference and tools_match and violations == 0
        return RewardResult(
            score=float(success),
            breakdown={
                "reference_agreement": float(answer_matches_reference),
                "tool_outcome": float(tools_match),
                "task_success": float(success),
            },
            metadata={
                "reward_semantics": "verified_task_outcome_v1",
                "match_mode": mode,
            },
        )
