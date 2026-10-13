"""Reward audits must expose misleading signal and preserve failed calls."""

import asyncio
import copy
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from typer.testing import CliRunner

from stateset_agents.evaluation.reward_audit import (
    RewardAuditPolicy,
    audit_reward,
    load_reward_suite,
)


@pytest.fixture
def suite():
    return {
        "schema_version": 1,
        "cases": [
            {
                "id": "order",
                "messages": [{"role": "user", "content": "What happened to order A?"}],
                "context": {"verified": ["shipped"]},
                "candidates": [
                    {
                        "id": "grounded",
                        "response": {"role": "assistant", "content": "shipped"},
                    },
                    {
                        "id": "invented",
                        "response": {"role": "assistant", "content": "refunded"},
                    },
                ],
                "preferences": [["grounded", "invented"]],
            }
        ],
    }


class GroundedReward:
    async def compute_reward(self, turns, context):
        return SimpleNamespace(score=float(turns[-1].content in context["verified"]))


async def test_audit_exercises_context_and_has_reproducible_suite_identity(suite):
    report = await audit_reward(GroundedReward(), suite)
    assert report["passed"]
    assert report["summary"] == {
        "cases": 1,
        "candidates": 2,
        "informative_groups": 1,
        "informative_fraction": 1.0,
        "failed_preferences": 0,
    }
    assert report["cases"][0]["candidates"]["grounded"]["scores"] == [1.0] * 3
    assert report == await audit_reward(GroundedReward(), copy.deepcopy(suite))
    suite["cases"][0]["context"]["verified"] = ["refunded"]
    changed = await audit_reward(GroundedReward(), suite)
    assert changed["suite_sha256"] != report["suite_sha256"]
    assert not changed["passed"]


@pytest.mark.parametrize("score", [0, 1, -10])
async def test_constant_nonzero_rewards_are_not_learning_signal(suite, score):
    class Constant:
        async def compute_reward(self, turns, context):
            return SimpleNamespace(score=score)

    report = await audit_reward(Constant(), suite)
    assert not report["passed"]
    assert "insufficient_group_signal" in report["failure_reasons"]
    assert report["summary"]["failed_preferences"] == 1


async def test_between_group_variation_does_not_mask_constant_groups(suite):
    second = copy.deepcopy(suite["cases"][0])
    second["id"] = "other"
    second["context"]["constant"] = 20
    suite["cases"][0]["context"]["constant"] = -20
    suite["cases"].append(second)

    class ConstantPerGroup:
        async def compute_reward(self, turns, context):
            return SimpleNamespace(score=context["constant"])

    report = await audit_reward(ConstantPerGroup(), suite)
    assert report["summary"]["informative_groups"] == 0
    assert not report["passed"]


async def test_signal_without_correct_rankings_fails(suite):
    suite["cases"][0]["preferences"] = [["invented", "grounded"]]
    report = await audit_reward(GroundedReward(), suite)
    assert report["summary"]["informative_groups"] == 1
    assert report["failure_reasons"] == ["preference_violations"]


async def test_declared_fraction_includes_flat_and_failed_groups(suite):
    second = copy.deepcopy(suite["cases"][0])
    second["id"] = "flat"
    second["context"]["verified"] = []
    second["preferences"] = []
    suite["cases"].append(second)
    policy = RewardAuditPolicy(min_informative_fraction=0.5)
    assert not (await audit_reward(GroundedReward(), suite))["passed"]
    report = await audit_reward(GroundedReward(), suite, policy=policy)
    assert report["passed"]
    assert report["summary"]["informative_fraction"] == 0.5

    class SometimesBroken(GroundedReward):
        async def compute_reward(self, turns, context):
            if not context["verified"]:
                raise RuntimeError("broken group")
            return await super().compute_reward(turns, context)

    report = await audit_reward(SometimesBroken(), suite, policy=policy)
    assert not report["passed"]
    assert report["summary"]["informative_fraction"] == 0.5
    assert "reward_errors" in report["failure_reasons"]


async def test_tool_responses_preserve_argument_types(suite):
    case = suite["cases"][0]
    for candidate, argument in zip(case["candidates"], [1, True], strict=True):
        candidate["response"] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"function": {"name": "lookup", "arguments": {"count": argument}}}
            ],
        }

    class TypedToolReward:
        async def compute_reward(self, turns, context):
            count = turns[-1].tool_calls[0]["function"]["arguments"]["count"]
            return SimpleNamespace(score=float(type(count) is int))

    assert (await audit_reward(TypedToolReward(), suite))["passed"]


async def test_stateful_scores_fail_repeatability(suite):
    class Counter:
        score = 0

        async def compute_reward(self, turns, context):
            self.score += 1
            return SimpleNamespace(score=self.score)

    report = await audit_reward(Counter(), suite)
    assert "unstable_scores" in report["failure_reasons"]
    assert not report["cases"][0]["stable"]


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), -float("inf"), True, "1", None]
)
async def test_invalid_scores_remain_failed_calls_not_zero_rewards(suite, bad):
    class Broken:
        async def compute_reward(self, turns, context):
            return SimpleNamespace(score=bad)

    report = await audit_reward(Broken(), suite)
    candidate = report["cases"][0]["candidates"]["grounded"]
    assert candidate["scores"] == [None] * 3
    assert len(candidate["errors"]) == 3
    assert "reward_errors" in report["failure_reasons"]
    json.dumps(report, allow_nan=False)


async def test_exceptions_are_retained_and_other_candidates_still_run(suite):
    class Broken(GroundedReward):
        async def compute_reward(self, turns, context):
            if turns[-1].content == "refunded":
                raise RuntimeError("missing lookup result")
            return await super().compute_reward(turns, context)

    report = await audit_reward(Broken(), suite)
    candidates = report["cases"][0]["candidates"]
    assert candidates["grounded"]["scores"] == [1.0] * 3
    assert candidates["invented"]["errors"][0]["type"] == "RuntimeError"
    assert not report["passed"]


async def test_async_timeouts_retained_and_cancellation_propagates(suite):
    class Waiting:
        async def compute_reward(self, turns, context):
            await asyncio.sleep(10)

    report = await audit_reward(
        Waiting(), suite, policy=RewardAuditPolicy(timeout_seconds=0.001)
    )
    assert (
        report["cases"][0]["candidates"]["grounded"]["errors"][0]["type"]
        == "TimeoutError"
    )

    class Cancelled:
        async def compute_reward(self, turns, context):
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await audit_reward(Cancelled(), suite)


async def test_mutating_rewards_get_fresh_context_and_turns(suite):
    original = copy.deepcopy(suite)

    class Mutating(GroundedReward):
        async def compute_reward(self, turns, context):
            result = await super().compute_reward(turns, context)
            context["verified"].clear()
            turns[-1].content = "mutated"
            turns.clear()
            return result

    assert (await audit_reward(Mutating(), suite))["passed"]
    assert suite == original


async def test_tolerance_applies_to_signal_and_rankings(suite):
    class TinyDifference:
        async def compute_reward(self, turns, context):
            return SimpleNamespace(score=1e-10 if turns[-1].content == "shipped" else 0)

    assert not (await audit_reward(TinyDifference(), suite))["passed"]
    assert (
        await audit_reward(
            TinyDifference(), suite, policy=RewardAuditPolicy(score_tolerance=0)
        )
    )["passed"]


async def test_extreme_finite_scores_do_not_overflow_report(suite):
    class Extreme:
        async def compute_reward(self, turns, context):
            return SimpleNamespace(
                score=1e308 if turns[-1].content == "shipped" else -1e308
            )

    report = await audit_reward(Extreme(), suite)
    assert report["passed"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "policy",
    [
        {"repeats": True},
        {"repeats": 1},
        {"score_tolerance": -1},
        {"score_tolerance": float("nan")},
        {"timeout_seconds": 0},
        {"min_informative_fraction": 0},
        {"min_informative_fraction": 1.1},
    ],
)
def test_invalid_policy_rejected(policy):
    with pytest.raises(ValueError):
        RewardAuditPolicy(**policy)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s.update(schema_version=True),
        lambda s: s.update(cases=[]),
        lambda s: s["cases"].append(copy.deepcopy(s["cases"][0])),
        lambda s: s["cases"][0].update(preferences=[]),
        lambda s: s["cases"][0].update(preferences=[["grounded", "unknown"]]),
        lambda s: s["cases"][0].update(preferences=[["grounded", "grounded"]]),
        lambda s: s["cases"][0].update(preferences=[["grounded", "invented"]] * 2),
        lambda s: s["cases"][0].update(context=[]),
        lambda s: s["cases"][0].update(messages=[]),
        lambda s: s["cases"][0]["context"].update(value=float("nan")),
        lambda s: s["cases"][0]["candidates"][1].update(id="grounded"),
        lambda s: s["cases"][0]["candidates"][0]["response"].update(role="system"),
        lambda s: s["cases"][0]["candidates"][0]["response"].update(tool_calls="bad"),
        lambda s: s["cases"][0].update(preferenes=[]),
    ],
)
async def test_invalid_suite_rejected_before_reward_execution(suite, mutation):
    mutation(suite)

    class MustNotRun:
        async def compute_reward(self, turns, context):
            pytest.fail("Invalid suite reached reward code")

    with pytest.raises(ValueError):
        await audit_reward(MustNotRun(), suite)


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version": 1, "schema_version": 1, "cases": []}')
    with pytest.raises(ValueError, match="duplicate JSON field"):
        load_reward_suite(path)


def test_cli_success_failure_and_no_overwrite(suite, tmp_path, monkeypatch):
    from stateset_agents.cli import app

    module = ModuleType("audit_test_factory")
    calls = []

    def factory():
        calls.append(True)
        return GroundedReward()

    module.factory = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    source = tmp_path / "suite.json"
    source.write_text(json.dumps(suite))
    output = tmp_path / "report.json"
    args = [
        "reward-audit",
        str(source),
        "--reward",
        "audit_test_factory:factory",
        "--output",
        str(output),
    ]
    runner = CliRunner()
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text())["passed"]
    before = output.read_bytes()
    assert runner.invoke(app, args).exit_code == 2
    assert output.read_bytes() == before
    assert len(calls) == 1
    suite["cases"][0]["context"]["verified"] = []
    source.write_text(json.dumps(suite))
    args[-1] = str(tmp_path / "failed.json")
    result = runner.invoke(app, args)
    assert result.exit_code == 1, result.output
    assert (
        "insufficient_group_signal"
        in json.loads(Path(args[-1]).read_text())["failure_reasons"]
    )


async def test_bundled_example_uses_real_reward_without_model_download():
    from stateset_agents.data.gsm8k import GSM8KReward

    suite = load_reward_suite(
        Path(__file__).parents[2] / "examples/data/reward_audit_gsm8k.json"
    )
    report = await audit_reward(GSM8KReward(), suite)
    assert report["passed"]
    assert report["summary"]["cases"] == 2


def test_audit_does_not_import_training_stack():
    script = """
import asyncio
import importlib.abc
import sys
class BlockTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "transformers", "datasets"}:
            raise AssertionError("Audit imported training dependency: " + fullname)
sys.meta_path.insert(0, BlockTraining())
from stateset_agents.data.gsm8k import GSM8KReward
from stateset_agents.evaluation.reward_audit import audit_reward, load_reward_suite
suite = load_reward_suite("examples/data/reward_audit_gsm8k.json")
assert asyncio.run(audit_reward(GSM8KReward(), suite))["passed"]
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_cli_factory_failure_is_setup_error(suite, tmp_path, monkeypatch):
    from stateset_agents.cli import app

    module = ModuleType("broken_audit_factory")

    def factory():
        raise RuntimeError("configuration missing")

    module.factory = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    source = tmp_path / "suite.json"
    source.write_text(json.dumps(suite))
    output = tmp_path / "report.json"
    result = CliRunner().invoke(
        app,
        [
            "reward-audit",
            str(source),
            "--reward",
            "broken_audit_factory:factory",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 2
    assert "configuration missing" in result.output
    assert not output.exists()
