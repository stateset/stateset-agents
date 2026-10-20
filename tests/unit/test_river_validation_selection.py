"""Skipped optimizer batches must not win selection through resampling noise."""

import asyncio
import copy

import pytest

from stateset_agents.core.environments.refund_policy_environment import (
    refund_policy_benchmark,
)
from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.evaluation.refund_trace import (
    select_replayed_validation_checkpoint,
)
from tests.unit.river_fakes import traced_refund_outcome


@pytest.fixture
def validation():
    cases = refund_policy_benchmark("validation", 2, 17)

    async def make():
        values = []
        for step, successes in enumerate((0, 1, 2)):
            values.append(
                {
                    "step": step,
                    "checkpoint": {"path": f"river://fixture-{step}", "step": step},
                    "metrics": {"reward_mean": successes - 1},
                    "case_hashes": {
                        row["order_id"]: content_hash(row) for row in cases
                    },
                    "outcomes": [
                        await traced_refund_outcome(row, successful=index < successes)
                        for index, row in enumerate(cases)
                    ],
                }
            )
        return values

    return asyncio.run(make()), {row["order_id"]: row for row in cases}


def select(validation, flags, **kwargs):
    values, cases = validation
    progress = [
        {"step": step, "metrics": {"train/updated": flag}}
        for step, flag in enumerate(flags, 1)
    ]
    return asyncio.run(
        select_replayed_validation_checkpoint(
            values,
            steps=2,
            cases=cases,
            environment="refund-policy-v2",
            training_progress=progress,
            run_manifest_hash="fixture-manifest",
            **kwargs,
        )
    )


def test_skipped_batch_cannot_promote_a_better_resampling(validation):
    before = copy.deepcopy(validation)
    selected = select(validation, [1, 0])
    assert selected["step"] == 1
    selection = selected["selection_activity"]
    assert selection["eligible_validation_steps"] == [0, 1]
    assert selection["excluded_skipped_batches"] == [2]
    activity = selection["selected_training_activity"]
    assert activity["observed_optimizer_updates"] == 1
    assert activity["completed_batches"] == 1
    assert selection["validation_results_hash"] == content_hash(validation[0])
    assert validation == before


def test_initial_checkpoint_has_no_updates_even_when_run_updated(validation):
    values, _ = validation
    # Put the highest score at baseline and the next highest at the skipped batch.
    for index, source in enumerate(copy.deepcopy(values)[::-1]):
        values[index].update(metrics=source["metrics"], outcomes=source["outcomes"])
    selected = select(validation, [0, 1])
    assert selected["step"] == 0
    activity = selected["selection_activity"]["selected_training_activity"]
    assert activity["status"] == "initial_checkpoint"
    assert activity["observed_optimizer_updates"] == 0
    assert activity["unknown_update_batches"] == 0
    assert activity["completed_batches"] == 0


def test_first_skipped_batch_is_excluded_without_excluding_later_update(validation):
    values, _ = validation
    first, second = copy.deepcopy(values[1]), copy.deepcopy(values[2])
    values[1].update(metrics=second["metrics"], outcomes=second["outcomes"])
    values[2].update(metrics=first["metrics"], outcomes=first["outcomes"])
    selected = select(validation, [0, 1])
    assert selected["step"] == 2
    selection = selected["selection_activity"]
    assert selection["excluded_skipped_batches"] == [1]
    assert selection["selected_training_activity"]["observed_skipped_batches"] == 1
    assert selection["selected_training_activity"]["observed_optimizer_updates"] == 1


def test_every_skipped_validation_is_still_replayed(validation):
    validation[0][2]["outcomes"][0]["environment_trace"]["initial_messages"] = []
    with pytest.raises(ValueError, match="Validation replay failed at step 2"):
        select(validation, [1, 0])


def test_unknown_activity_is_eligible_and_never_invented_as_learning(validation):
    values, cases = validation
    progress = [
        {"step": 1, "metrics": {"train/updated": 1}},
        {"step": 2, "metrics": {"loss": 0.2}},
    ]
    selected = asyncio.run(
        select_replayed_validation_checkpoint(
            values,
            steps=2,
            cases=cases,
            environment="refund-policy-v2",
            training_progress=progress,
        )
    )
    selection = selected["selection_activity"]
    assert selected["step"] == 2
    assert selection["excluded_skipped_batches"] == []
    assert selection["selected_training_activity"]["observed_optimizer_updates"] == 1
    assert selection["selected_training_activity"]["unknown_update_batches"] == 1


@pytest.mark.parametrize(
    "progress", [[], [{"step": 2, "metrics": {"train/updated": 0}}]]
)
def test_incomplete_progress_cannot_authorize_skip_filtering(validation, progress):
    values, cases = validation
    with pytest.raises(ValueError, match="Incomplete RL training progress"):
        asyncio.run(
            select_replayed_validation_checkpoint(
                values,
                steps=2,
                cases=cases,
                environment="refund-policy-v2",
                training_progress=progress,
            )
        )
