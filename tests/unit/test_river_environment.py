"""Stateful outcome and environment-bridge tests without River credentials."""

import json
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from stateset_agents.core.environments.refund_environment import (
    RefundEnvironment,
    refund_benchmark,
)
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.remote.river_environment import river_environment_factory
from tests.unit.river_fakes import traced_refund_trajectory


def action(tool, **args):
    return ConversationTurn(
        role="assistant", content=json.dumps({"tool": tool, "args": args})
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("eligible", [True, False])
async def test_correct_resolution_requires_executed_action(eligible):
    env = RefundEnvironment()
    state = await env.reset(
        {"order_id": "A", "amount_cents": 1234, "eligible": eligible}
    )
    state, reward, done, _ = await env.step(state, action("lookup_order", order_id="A"))
    assert not done and reward == 0
    resolution = (
        action("refund", order_id="A", amount_cents=1234)
        if eligible
        else action("deny", order_id="A")
    )
    state, _, _, _ = await env.step(state, resolution)
    state, reward, done, info = await env.step(state, action("finish"))
    assert done and reward == 1 and info["metrics"]["task_success"] == 1
    assert state.context["refunds"] == ([1234] if eligible else [])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        action("refund", order_id="A", amount_cents=100),
        action("refund", order_id="B", amount_cents=1234),
        ConversationTurn(role="assistant", content="I refunded the correct amount"),
        action("deny", order_id="A"),
    ],
)
async def test_words_wrong_amount_and_policy_violations_cannot_earn_success(bad):
    env = RefundEnvironment()
    state = await env.reset({"order_id": "A", "amount_cents": 1234, "eligible": True})
    state, _, _, _ = await env.step(state, action("lookup_order", order_id="A"))
    state, _, _, _ = await env.step(state, bad)
    state, reward, _, info = await env.step(state, action("finish"))
    assert reward == -1 and info["metrics"]["policy_violations"] == 1


@pytest.mark.asyncio
async def test_duplicate_refund_invalidates_otherwise_correct_episode():
    env = RefundEnvironment()
    state = await env.reset({"order_id": "A", "amount_cents": 1234, "eligible": True})
    for tool in (
        action("lookup_order", order_id="A"),
        action("refund", order_id="A", amount_cents=1234),
        action("refund", order_id="A", amount_cents=1234),
        action("finish"),
    ):
        state, reward, done, info = await env.step(state, tool)
    assert done and reward == -1
    assert state.context["refunds"] == [1234]
    assert info["metrics"]["policy_violations"] == 1


def test_benchmark_is_reproducible_balanced_and_disjoint():
    sets = [refund_benchmark(split) for split in ("train", "validation", "test")]
    ids = [{r["order_id"] for r in rows} for rows in sets]
    assert len(set.union(*ids)) == 384
    assert sets[0] == refund_benchmark("train")
    assert all(sum(r["eligible"] for r in rows) == 64 for rows in sets)


@pytest.mark.asyncio
async def test_bridge_isolates_state_and_returns_only_new_observations(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "river_client", SimpleNamespace(rl=SimpleNamespace(Env=object))
    )
    factory = river_environment_factory(RefundEnvironment)
    left, right = factory(), factory()
    row = {"order_id": "A", "amount_cents": 1234, "eligible": True}
    await left.reset(row)
    await right.reset(row)
    traj = SimpleNamespace(
        messages=[
            {
                "role": "assistant",
                "content": action("lookup_order", order_id="A").content,
            }
        ],
        metrics={},
    )
    observations = await left.on_turn(traj)
    assert len(observations) == 1
    assert left.state.context["looked_up"] and not right.state.context["looked_up"]
    assert len(traj.messages) == 1  # sampled assistant history is never rewritten
    traj.messages[-1]["content"] = action(
        "refund", order_id="A", amount_cents=1234
    ).content
    await left.on_turn(traj)
    traj.messages[-1]["content"] = action("finish").content
    assert await left.on_turn(traj) is None
    assert await left.reward(traj, row) == 1
    assert await left.on_truncated(traj, row, "length") == 0
    assert traj.stateset_episode_id == "A"
    assert traj.metrics["task_success"] == 0
    assert right.state.context["refunds"] == []


@pytest.mark.asyncio
async def test_bridge_propagates_infrastructure_failure(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "river_client", SimpleNamespace(rl=SimpleNamespace(Env=object))
    )

    class Broken(RefundEnvironment):
        async def step(self, *args):
            raise ConnectionError("sandbox unavailable")

    env = river_environment_factory(Broken)()
    await env.reset({"order_id": "A", "amount_cents": 100, "eligible": True})
    with pytest.raises(ConnectionError):
        await env.on_turn(
            SimpleNamespace(
                messages=[{"role": "assistant", "content": "{}"}], metrics={}
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("evaluate_only", [False, True])
@pytest.mark.parametrize("benchmark_name", ["refund-v1", "refund-policy-v2"])
@pytest.mark.parametrize("resume", [False, True])
async def test_native_campaign_selects_validation_checkpoint_before_test(
    tmp_path, monkeypatch, evaluate_only, benchmark_name, resume
):
    from examples.river_refund_rl import campaign

    @dataclass
    class Checkpoint:
        path: str
        step: int = 0
        checkpoint_type: str = "inference"

    tested = []
    from stateset_agents.training.river_refund import BENCHMARKS

    validation_rows = BENCHMARKS[benchmark_name][1]("validation", 1, 42)
    test_rows = BENCHMARKS[benchmark_name][1]("test", 1, 42)

    class Engine:
        def __init__(self, model, **kwargs):
            self.model = model
            from stateset_agents.core.environments.refund_policy_environment import (
                RefundPolicyEnvironment,
            )

            expected = (
                RefundEnvironment
                if benchmark_name == "refund-v1"
                else RefundPolicyEnvironment
            )
            assert isinstance(kwargs["env"]().environment, expected)

        async def rollout(self, rows, **kwargs):
            assert rows == test_rows
            tested.append(self.model.checkpoint.path)
            trajectory = await traced_refund_trajectory(
                rows[0], environment=benchmark_name
            )
            trajectory.generated_tokens, trajectory.elapsed = 20, 1.5
            yield [trajectory]

    class Trainer:
        def __init__(self, **kwargs):
            assert not evaluate_only, "evaluation must never construct a trainer"
            assert kwargs["truncation"] == {"train": "reward"}
            self.evaluator = kwargs["evaluator"]

        async def run(self, rows, steps, after_recovery=None):
            assert rows == BENCHMARKS[benchmark_name][1]("train", 1, 42)
            if resume:
                await after_recovery(1)
            for step, value in ((0, -1), (1, 1), (2, -1)):
                result = SimpleNamespace(
                    step=step,
                    checkpoint=Checkpoint(f"river://round-{step}"),
                    metrics={"reward_mean": value},
                    trajectories=[
                        await traced_refund_trajectory(
                            validation_rows[0],
                            successful=value > 0,
                            environment=benchmark_name,
                        )
                    ],
                )
                await self.evaluator.sink(result)
                if step >= (2 if resume else 1):
                    yield SimpleNamespace(n=step, metrics={"loss": 0.1})

    rl = SimpleNamespace(
        Env=object,
        Budget=lambda **kw: kw,
        Schedule=lambda **kw: kw,
        GroupCompletion=lambda **kw: kw,
        RolloutEngine=Engine,
        CheckpointSampler=lambda session, **kw: SimpleNamespace(**kw),
        AsyncTrainer=Trainer,
        Evaluator=lambda rows, **kw: SimpleNamespace(**kw),
        Adam=lambda **kw: kw,
        GroupCentered=lambda: None,
        Truncation=lambda **kw: kw,
        Checkpointing=lambda *a, **kw: kw,
    )
    monkeypatch.setitem(
        sys.modules, "river_client", SimpleNamespace(rl=rl, Checkpoint=Checkpoint)
    )
    model = SimpleNamespace(
        save_weights=lambda *a, **kw: Checkpoint("river://baseline")
    )
    args = SimpleNamespace(
        output=tmp_path,
        base_model="base",
        seed=42,
        concurrency=8,
        learning_rate=1e-5,
        max_staleness=0,
        steps=2,
        evaluate_only=evaluate_only,
        benchmark=benchmark_name,
        checkpoint=None,
        dry_run=False,
    )
    from stateset_agents.training.river_refund import prepare_run

    splits = {
        "train": BENCHMARKS[benchmark_name][1]("train", 1, 42),
        "validation": validation_rows,
        "test": test_rows,
    }
    prepare_run(args, splits)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    if resume and not evaluate_only:
        (tmp_path / "training_metrics.json").write_text(
            json.dumps(
                [
                    {"step": 1, "metrics": {"loss": 0.25}},
                    {"step": 2, "metrics": {"loss": 0.9}},
                ]
            )
        )
    await campaign(
        model,
        object(),
        SimpleNamespace(tokenizer=object()),
        args,
        splits,
    )
    assert tested == ["river://baseline" if evaluate_only else "river://round-1"]
    report = json.loads((tmp_path / "test_results.json").read_text())
    assert report["run_manifest_hash"] == content_hash(manifest)
    if not evaluate_only:
        metrics = json.loads((tmp_path / "training_metrics.json").read_text())
        assert metrics == [
            {"step": 1, "metrics": {"loss": 0.25 if resume else 0.1}},
            {"step": 2, "metrics": {"loss": 0.1}},
        ]
    assert report["passed"] == report["total"] == 1
    assert report["generated_tokens"] == 20
    assert report["outcomes"][0]["case_id"] == test_rows[0]["order_id"]
    assert report["outcomes"][0]["environment_trace"]["terminal"]["reward"] == 1
    assert report["environment"] == benchmark_name
    if benchmark_name == "refund-policy-v2":
        assert report["families"][test_rows[0]["family"]]["passed"] == 1
