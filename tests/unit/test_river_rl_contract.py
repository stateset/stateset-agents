"""Behavioral regressions for scoring, updates, budgets, and recovery."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.job import JobStatus
from stateset_agents.remote.river import RiverExecutor
from stateset_agents.remote.river_rl import (
    RiverRLConfig,
    RLScore,
    exclusive_run,
    normalize_datums,
    retain_group,
    reward_function_scorer,
    sample_record,
    score_text,
)
from stateset_agents.remote.river_rl_runner import _BudgetModel
from tests.unit.test_river_submit_golden import stub_river_renderers  # noqa: F401
from tests.unit.test_river_submit_golden import (
    FakeSample,
    RecordingClient,
    RecordingModel,
    RiverConnectionError,
    scenario_rl,
)


@pytest.mark.parametrize(
    "knobs",
    [
        {"rounds": 0},
        {"best_of": 1},
        {"seed": -1},
        {"temperature": float("nan")},
        {"top_p": 1.1},
        {"loss_fn": "typo"},
        {"normalization": "typo"},
        {"microbatch_size": 0},
        {"max_generated_tokens": -1},
        {"clip_low": 1},
        {"truncation": "silent"},
        {"sead": 5},
    ],
)
def test_invalid_config_is_rejected(knobs):
    with pytest.raises((TypeError, ValueError)):
        RiverRLConfig.from_knobs(knobs)


def test_judge_only_task_uses_judge_for_reward_and_pass(monkeypatch):
    monkeypatch.setattr("stateset_agents.training.sft.judge_completion", lambda *a: 0.1)
    score = score_text(
        {"prompt": "refund", "judge": "customer_support", "min_judge_score": 0.9},
        "unrelated",
    )
    assert score.reward == 0.1
    assert not score.passed


@pytest.mark.parametrize("result", [None, float("nan"), float("inf")])
def test_judge_failure_is_not_a_success(monkeypatch, result):
    monkeypatch.setattr(
        "stateset_agents.training.sft.judge_completion", lambda *a: result
    )
    with pytest.raises(ValueError, match="failed to score"):
        score_text(
            {"prompt": "refund", "judge": "customer_support", "min_judge_score": 0.9},
            "done",
        )


def test_empty_scoring_does_not_mean_success():
    with pytest.raises(ValueError, match="needs"):
        score_text({"prompt": "refund"}, "anything")


@pytest.mark.parametrize("task", [{"expect": [""]}, {"forbid": [" "]}])
def test_empty_string_assertions_are_not_verifiers(task):
    with pytest.raises(ValueError, match="nonempty"):
        score_text({"prompt": "refund", **task}, "anything")


@pytest.mark.parametrize(
    "field,value",
    [
        ("token_data_is_exact", False),
        ("logprobs", [float("nan"), -0.2]),
        ("logprobs", [-0.1]),
        ("logprobs", [0.1, -0.2]),
        ("tokens", [1, -1]),
    ],
)
def test_corrupt_rollout_is_rejected(field, value):
    sample = FakeSample("done")
    setattr(sample, field, value)
    with pytest.raises(ValueError):
        sample_record(sample, [1, 2])


def test_truncation_drops_entire_group_before_group_baseline():
    records = [
        sample_record(FakeSample("done"), [1]),
        sample_record(FakeSample("partial", stop_reason="length"), [1]),
    ]
    assert not retain_group(records, RiverRLConfig())
    with pytest.raises(ValueError, match="Truncated"):
        retain_group(records, RiverRLConfig(truncation="error"))


def test_normalization_has_explicit_length_weighting_and_partition_invariance():
    data = [{"advantages": [0, 2, 0]}, {"advantages": [0, -2, -2, -2, 0]}]
    token, count = normalize_datums(data, "token")
    sequence, _ = normalize_datums(data, "sequence")
    assert count == 4
    assert token[0]["advantages"][1] == 0.5
    assert sum(abs(a) for a in sequence[0]["advantages"]) == 1
    assert sum(abs(a) for a in sequence[1]["advantages"]) == 1
    # Any transport partition has the same coefficients after logical normalization.
    for batch_size in (1, 2, 10):
        flattened = [
            d
            for i in range(0, len(token), batch_size)
            for d in token[i : i + batch_size]
        ]
        assert flattened == token
    assert data[0]["advantages"][1] == 2  # caller's recorded data is preserved


def test_duplicating_logical_batch_preserves_mean_objective_weight():
    data = [{"advantages": [0, 2, 0]}, {"advantages": [0, -1, -1, 0]}]
    for mode in ("token", "sequence"):
        once, _ = normalize_datums(data, mode)
        twice, _ = normalize_datums(data + data, mode)
        assert sum(sum(d["advantages"]) for d in once) == pytest.approx(
            sum(sum(d["advantages"]) for d in twice)
        )


def test_uncertain_sampling_keeps_token_reservation_across_driver_restart(tmp_path):
    class Uncertain:
        calls = 0

        def sample(self, *args, **kwargs):
            self.calls += 1
            raise ConnectionError("request outcome unknown")

    model = Uncertain()
    path = tmp_path / "usage.json"
    path.write_text(
        json.dumps({"fingerprint": "run1", "generated_tokens_or_reserved": 0})
    )
    with pytest.raises(ConnectionError):
        _BudgetModel(model, path, 10, "run1").sample(["prompt"], max_tokens=8)
    assert json.loads(path.read_text())["generated_tokens_or_reserved"] == 8
    with pytest.raises(ValueError, match="budget exhausted"):
        _BudgetModel(model, path, 10, "run1").sample(["prompt"], max_tokens=8)
    assert model.calls == 1


def _budget_model(tmp_path, response):
    class Model:
        calls = 0

        def sample(self, *args, **kwargs):
            self.calls += 1
            return response

    path = tmp_path / "usage.json"
    path.write_text(
        json.dumps({"fingerprint": "run", "generated_tokens_or_reserved": 0})
    )
    return _BudgetModel(Model(), path, 16, "run")


@pytest.mark.parametrize("response", [None, [], [[]], [[FakeSample("ok")]], [None]])
def test_incomplete_sampling_does_not_refund_or_admit_another_request(
    tmp_path, response
):
    model = _budget_model(tmp_path, response)
    with pytest.raises(ValueError, match="Incomplete"):
        model.sample(["prompt"], num_samples=2, max_tokens=8)
    assert json.loads(model.path.read_text())["generated_tokens_or_reserved"] == 16
    with pytest.raises(ValueError, match="budget exhausted"):
        model.sample(["prompt"], max_tokens=1)
    assert model.model.calls == 1


@pytest.mark.parametrize("tokens", [None, "bad", [True], [-1], [1.5]])
def test_invalid_exact_token_data_keeps_reservation(tmp_path, tokens):
    model = _budget_model(tmp_path, [[FakeSample("ok", tokens=tokens)]])
    with pytest.raises(ValueError, match="token data"):
        model.sample(["prompt"], max_tokens=8)
    assert json.loads(model.path.read_text())["generated_tokens_or_reserved"] == 8


@pytest.mark.parametrize("exact", [False, None, "true", 1])
def test_inexact_token_metadata_never_refunds_reserved_tokens(tmp_path, exact):
    model = _budget_model(tmp_path, [[FakeSample("ok", token_data_is_exact=exact)]])
    model.sample(["prompt"], max_tokens=8)
    assert json.loads(model.path.read_text())["generated_tokens_or_reserved"] == 8


def test_complete_exact_response_refunds_only_unused_tokens(tmp_path):
    model = _budget_model(tmp_path, [[FakeSample("ok"), FakeSample("ok")]])
    model.sample(prompt_token_ids=[[1, 2]], num_samples=2, max_tokens=8)
    assert json.loads(model.path.read_text())["generated_tokens_or_reserved"] == 4
    model.sample(["prompt"], num_samples=2, max_tokens=4)
    assert json.loads(model.path.read_text())["generated_tokens_or_reserved"] == 8


def test_provider_token_overrun_is_recorded_and_rejected(tmp_path):
    model = _budget_model(tmp_path, [[FakeSample("ok", tokens=list(range(20)))]])
    with pytest.raises(ValueError, match="exceeded"):
        model.sample(["prompt"], max_tokens=8)
    assert json.loads(model.path.read_text())["generated_tokens_or_reserved"] == 20
    with pytest.raises(ValueError, match="budget exhausted"):
        model.sample(["prompt"], max_tokens=1)
    assert model.model.calls == 1


@pytest.mark.parametrize("field", ["num_samples", "max_tokens"])
@pytest.mark.parametrize("value", [None, True, 0, -1, 1.5, "2"])
def test_invalid_reservations_fail_before_mutation_or_provider_call(
    tmp_path, field, value
):
    model = _budget_model(tmp_path, [[FakeSample("ok")]])
    before = model.path.read_bytes()
    with pytest.raises(ValueError, match="positive integer"):
        model.sample(["prompt"], **{**{"max_tokens": 8}, field: value})
    assert model.path.read_bytes() == before
    assert model.model.calls == 0


def test_usage_disappearing_during_run_does_not_reset_allowance(tmp_path):
    model = _budget_model(tmp_path, [[FakeSample("ok")]])
    model.sample(["prompt"], max_tokens=8)
    model.path.unlink()
    with pytest.raises(ValueError, match="ledger"):
        model.sample(["prompt"], max_tokens=8)
    assert not model.path.exists()
    assert model.model.calls == 1


@pytest.mark.parametrize("stage", ["reservation", "settlement"])
def test_failed_usage_write_never_grants_unrecorded_sampling(
    tmp_path, monkeypatch, stage
):
    import stateset_agents.remote.river_rl_runner as runner

    model = _budget_model(tmp_path, [[FakeSample("ok")]])
    original = runner.atomic_json

    def write(path, usage):
        failing_count = 8 if stage == "reservation" else 2
        if usage["generated_tokens_or_reserved"] == failing_count:
            raise OSError("disk full")
        original(path, usage)

    monkeypatch.setattr(runner, "atomic_json", write)
    with pytest.raises(OSError, match="disk full"):
        model.sample(["prompt"], max_tokens=8)
    assert model.model.calls == (0 if stage == "reservation" else 1)
    assert json.loads(model.path.read_text())["generated_tokens_or_reserved"] == (
        0 if stage == "reservation" else 8
    )


def test_driver_lock_is_exclusive_and_released_after_failure(tmp_path):
    with pytest.raises(RuntimeError):
        with exclusive_run(tmp_path):
            with pytest.raises(ValueError, match="Another driver"):
                with exclusive_run(tmp_path):
                    pytest.fail("second writer acquired an active run")
            raise RuntimeError("driver crashed")
    with exclusive_run(tmp_path):
        pass


def test_stateset_reward_adapter_preserves_turns_context_and_components():
    from stateset_agents.core.reward_base import RewardResult

    class Reward:
        async def compute_reward(self, turns, context):
            assert [t.role for t in turns] == ["user", "assistant"]
            assert context["prompt"] == "refund"
            return RewardResult(0.8, components={"outcome": 0.8})

    score = reward_function_scorer(Reward(), threshold=0.7)(
        {"prompt": "refund"}, "done"
    )
    assert score == RLScore(0.8, True, {"outcome": 0.8})


def _executor(tmp_path, client, **kwargs):
    executor = RiverExecutor(
        client=client, ledger_path=tmp_path / "ledger.jsonl", **kwargs
    )
    executor._sleep = lambda seconds: None
    return executor


class RegressingModel(RecordingModel):
    """Validation: baseline fails, round 1 passes, round 2 regresses."""

    step = 0

    def sample(self, *args, **kwargs):
        if kwargs.get("temperature") == 0:
            return [[FakeSample("done" if self.step == 1 else "nope")]]
        return [[FakeSample("done"), FakeSample("nope")]]

    def optim_step(self, **kwargs):
        self.step += 1
        return super().optim_step(**kwargs)


def test_best_checkpoint_and_its_eval_survive_later_regression(tmp_path):
    spec, _ = scenario_rl(tmp_path)
    client = RecordingClient(model_cls=RegressingModel)
    executor = _executor(tmp_path, client)
    result = executor.wait(executor.submit(spec))
    assert result.status == JobStatus.SUCCEEDED
    report = json.loads((spec.output_dir / "rl_report.json").read_text())
    pointer = json.loads((spec.output_dir / "river_checkpoint.json").read_text())
    assert report["best_round"] == 1
    assert report["rounds"][-1]["passed"] == 0
    assert pointer["checkpoint"].endswith("rl-best-1")
    rows = json.loads((spec.output_dir / "eval_results.json").read_text())
    assert rows[0]["checks"]["passed"] is True


def test_failed_backward_never_submits_optimizer(tmp_path):
    class Broken(RecordingModel):
        def forward_backward(self, *args, **kwargs):
            raise RuntimeError("backward failed")

    spec, _ = scenario_rl(tmp_path)
    client = RecordingClient(model_cls=Broken)
    with pytest.raises(RemoteExecutionError):
        _executor(tmp_path, client).submit(spec)
    assert not any(c["call"] == "optim_step" for c in client.calls)
    state = json.loads((spec.output_dir / "rl_state.json").read_text())
    assert state["completed_round"] == 0
    assert (
        json.loads((spec.output_dir / "rl_report.json").read_text())["status"]
        == "failed"
    )


def test_recovery_restores_committed_training_checkpoint_without_replaying_prior_round(
    tmp_path,
):
    class Flaky(RecordingModel):
        backwards = 0

        def sample(self, *args, **kwargs):
            return (
                [[FakeSample("done"), FakeSample("nope")]]
                if kwargs.get("num_samples") == 2
                else [[FakeSample("done")]]
            )

        def forward_backward(self, *args, **kwargs):
            type(self).backwards += 1
            if type(self).backwards == 2:
                raise RiverConnectionError("lost second round")
            return super().forward_backward(*args, **kwargs)

    spec, _ = scenario_rl(tmp_path)
    client = RecordingClient(model_cls=Flaky)
    executor = _executor(tmp_path, client)
    executor.submit(spec)
    creates = [c for c in client.calls if c["call"] == "create_model"]
    assert len(creates) == 2
    assert "rl-state-1" in creates[1]["checkpoint"]
    assert len([c for c in client.calls if c["call"] == "optim_step"]) == 2
    state = json.loads((spec.output_dir / "rl_state.json").read_text())
    assert state["completed_round"] == 2 and state["steps"] == 2


def test_completed_resume_is_offline_and_rejects_changed_training_plan(tmp_path):
    spec, client = scenario_rl(tmp_path)
    executor = _executor(tmp_path, client)
    executor.submit(spec)
    count = len(client.calls)
    executor.submit(replace(spec, resume=True))
    assert len(client.calls) == count
    with pytest.raises(ValueError, match="differs"):
        executor.submit(replace(spec, resume=True, learning_rate=0.1))
    with pytest.raises(ValueError, match="already contains"):
        executor.submit(spec)


@pytest.mark.parametrize(
    "corruption",
    [
        "missing",
        "malformed",
        "wrong_shape",
        "foreign",
        "negative",
        "fractional",
        "boolean",
    ],
)
@pytest.mark.parametrize("interrupted", [False, True])
def test_resume_rejects_lost_or_invalid_usage_before_provider_calls(
    tmp_path, corruption, interrupted
):
    spec, client = scenario_rl(tmp_path)
    if interrupted:

        class Interrupted(RecordingModel):
            def forward_backward(self, *args, **kwargs):
                raise RuntimeError("interrupted update")

        client = RecordingClient(model_cls=Interrupted)
    executor = _executor(tmp_path, client)
    if interrupted:
        with pytest.raises(RemoteExecutionError, match="interrupted update"):
            executor.submit(spec)
    else:
        executor.submit(spec)
    path = spec.output_dir / "rl_usage.json"
    if corruption == "missing":
        path.unlink()
    elif corruption == "malformed":
        path.write_text("{")
    elif corruption == "wrong_shape":
        path.write_text("[]")
    else:
        usage = json.loads(path.read_text())
        if corruption == "foreign":
            usage["fingerprint"] = "different-run"
        else:
            usage["generated_tokens_or_reserved"] = {
                "negative": -1,
                "fractional": 0.5,
                "boolean": False,
            }[corruption]
        path.write_text(json.dumps(usage))
    calls = list(client.calls)
    with pytest.raises(ValueError, match="usage|budget"):
        executor.submit(replace(spec, resume=True))
    assert client.calls == calls
    if corruption == "missing":
        assert not path.exists()


def test_token_budget_prevents_sampling_before_spend(tmp_path):
    spec, client = scenario_rl(tmp_path)
    spec.harvest["max_generated_tokens"] = 1
    with pytest.raises(RemoteExecutionError, match="budget exhausted"):
        _executor(tmp_path, client).submit(spec)
    assert not any(c["call"] == "sample" for c in client.calls)


def test_dollar_budget_and_train_validation_overlap_fail_before_session(tmp_path):
    spec, client = scenario_rl(tmp_path)
    executor = _executor(tmp_path, client)
    with pytest.raises(ValueError, match="dollar"):
        executor.submit(replace(spec, max_cost_usd=1))
    with pytest.raises(ValueError, match="disjoint"):
        executor.submit(
            replace(spec, eval_prompts=[{"prompt": "fix it", "expect": ["done"]}])
        )
    assert client.calls == []


def test_custom_scorer_drives_training_and_eval(tmp_path):
    spec, client = scenario_rl(tmp_path)
    seen = []

    def scorer(task, response):
        seen.append(task["prompt"])
        return RLScore(float("done" in response), "done" in response)

    _executor(tmp_path, client, rl_scorer=scorer, rl_scorer_id="test-v1").submit(spec)
    assert {"check me", "fix it"} <= set(seen)


def test_microbatches_accumulate_then_update_once(tmp_path):
    spec, client = scenario_rl(tmp_path)
    spec.harvest.update(rounds=1, microbatch_size=1)
    _executor(tmp_path, client).submit(spec)
    calls = [c for c in client.calls if c["call"] in {"forward_backward", "optim_step"}]
    assert [c["call"] for c in calls] == [
        "forward_backward",
        "forward_backward",
        "optim_step",
    ]
    assert [c["zero_out"] for c in calls[:2]] == [True, False]


def test_mixed_policy_data_never_reaches_backward(tmp_path):
    class Mixed(RecordingModel):
        def sample(self, *args, **kwargs):
            groups = super().sample(*args, **kwargs)
            for group in groups:
                for i, sample in enumerate(group):
                    sample.policy_version = SimpleNamespace(id=str(i))
            return groups

    spec, _ = scenario_rl(tmp_path)
    client = RecordingClient(model_cls=Mixed)
    with pytest.raises(RemoteExecutionError, match="mixed behavior"):
        _executor(tmp_path, client).submit(spec)
    assert not any(c["call"] == "forward_backward" for c in client.calls)
