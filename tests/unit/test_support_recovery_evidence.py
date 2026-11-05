"""Retained judge replay and recovery-study runtime safeguards (CPU only)."""

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from stateset_agents.core.agent_config import AgentConfig
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.evaluation.judge_evidence import (
    RecordedSupportJudge,
    replay_judge_evidence,
)
from stateset_agents.evaluation.support_judge import ALL_CHECKS, JudgeItem
from stateset_agents.training.judge_selection import support_judge_evaluator
from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer
from stateset_agents.training.support_recovery import (
    RecoveryRuntimeGuard,
    build_recovery_plan,
    create_recovery_trainer,
    load_recovery_profile,
    validate_recovery_cases,
    validate_recovery_profile,
)
from tests.unit.test_support_reward_recovery import selection as selection_fixture

JUDGE_MODEL = "google/gemma-4-31B-it"
JUDGE_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"


def generate(prompts):
    return [json.dumps({"checks": dict.fromkeys(ALL_CHECKS, True)}) for _ in prompts]


def recorder(path, generate_fn=generate, **kwargs):
    return RecordedSupportJudge(
        generate_fn,
        path,
        judge_model=JUDGE_MODEL,
        judge_revision=JUDGE_REVISION,
        **kwargs,
    )


def item():
    return JudgeItem(
        id="one",
        scenario={"policy_facts": ["Returns are allowed within 30 days."]},
        transcript=(
            ("user", "Can I return it?"),
            ("assistant", "You can return the item within 30 days."),
        ),
    )


def cases(profile):
    return {
        "provenance": {
            "source": "reviewed",
            "reviewer": "synthetic unit-test fixture",
            "test_is_fresh": True,
        },
        **{
            split: [
                {
                    "id": f"{split}-{i}",
                    "user_query": f"Question for synthetic case {split}-{i}.",
                    "policy_facts": [f"Policy for synthetic case {split}-{i}."],
                    "expected_resolution": "State the provided policy.",
                    "must_avoid": [],
                }
                for i in range(minimum)
            ]
            for split, minimum in profile["minimum_cases"].items()
        },
    }


def small_profile():
    profile = load_recovery_profile()
    profile["minimum_cases"] = dict.fromkeys(("train", "validation", "test"), 32)
    profile["training"].update(num_episodes=2, eval_steps=3, save_steps=3, bf16=False)
    profile["selection"].update(every_updates=1, max_evaluations=3)
    return profile


@pytest.mark.parametrize("retry", [False, True])
def test_replay_preserves_transcripts_all_passes_and_parse_retries(tmp_path, retry):
    calls = []

    def backend(prompts):
        calls.append(list(prompts))
        return (
            ["invalid json"] * len(prompts)
            if retry and len(calls) == 1
            else generate(prompts)
        )

    judge = recorder(tmp_path / "recording", backend)
    results = judge.score(
        [item()], context={"split": "validation", "optimizer_step": 0}
    )
    reference = judge.last_evidence
    replay = replay_judge_evidence(
        Path(reference["path"]), expected_manifest_sha256=reference["manifest_sha256"]
    )
    assert replay["items"][0]["transcript"] == [
        list(turn) for turn in item().transcript
    ]
    assert replay["results"][0]["score"] == results[0].score
    assert replay["results"][0]["checks"] == [dict.fromkeys(ALL_CHECKS, True)] * 2
    assert len(calls) == (2 if retry else 1)  # Replay performs no inference.
    assert (Path(reference["path"]) / "request-000.json").is_file()
    if os.name != "nt":  # Windows st_mode carries no POSIX group/other bits.
        assert (Path(reference["path"]).stat().st_mode & 0o077) == 0


def test_transport_failure_keeps_request_and_failure_type_without_secret(tmp_path):
    def fail(_):
        raise RuntimeError("transport token=DO-NOT-RETAIN")

    judge = recorder(tmp_path / "recording", fail, max_batches=1)
    with pytest.raises(RuntimeError, match="transport"):
        judge.score([item()])
    root = judge.output_dir / "batch-000000"
    assert json.loads((root / "failure.json").read_text()) == {
        "error_type": "RuntimeError"
    }
    assert "DO-NOT-RETAIN" not in "".join(p.read_text() for p in root.iterdir())
    assert judge.last_evidence is None
    with pytest.raises(RuntimeError, match="budget exhausted"):
        judge.score([item()])
    manifest = json.loads((root / "manifest.json").read_text())
    with pytest.raises(ValueError, match="integrity"):
        replay_judge_evidence(root, expected_manifest_sha256=content_hash(manifest))


@pytest.mark.parametrize(
    "target", ["response-000.json", "inputs.json", "results.json", "manifest.json"]
)
def test_evidence_tampering_is_rejected(tmp_path, target):
    judge = recorder(tmp_path / "recording")
    judge.score([item()])
    reference = judge.last_evidence
    (Path(reference["path"]) / target).write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        replay_judge_evidence(
            Path(reference["path"]),
            expected_manifest_sha256=reference["manifest_sha256"],
        )


def test_recording_refuses_overwrite_and_symlinks(tmp_path):
    root = tmp_path / "recording"
    recorder(root)
    with pytest.raises(FileExistsError):
        recorder(root)
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        recorder(alias)


@pytest.mark.asyncio
async def test_selector_requires_records_and_rechecks_selected_judge_bytes(tmp_path):
    trainer, selector, _, _, _ = selection_fixture.__wrapped__(tmp_path)
    selector.require_evidence = True
    with pytest.raises(ValueError, match="replayable"):
        await selector.on_training_start(trainer, trainer.config)
    trainer, selector, _, _, _ = selection_fixture.__wrapped__(tmp_path)
    selector.require_evidence = True
    judge = recorder(tmp_path / "judge-evidence")

    class Environment:
        def clone(self):
            return Environment()

        async def run_episode(self, agent_fn, *, scenario, max_turns):
            return SimpleNamespace(
                turns=[
                    ConversationTurn(role="user", content="Policy question."),
                    ConversationTurn(
                        role="assistant", content="The stated policy applies."
                    ),
                ]
            )

    trainer.agent.generate_response = None  # This fixture environment supplies turns.
    selector.evaluate = support_judge_evaluator(judge, Environment())
    await selector.on_training_start(trainer, trainer.config)
    assert selector.selected_checkpoint().name == "judge-selected-step-0"
    receipt = json.loads((tmp_path / "judge-selection.json").read_text())
    recorded = receipt["evaluations"][0]
    assert recorded["items"][0]["parse_failures"] == 0
    assert "checks" in recorded["items"][0]
    assert not Path(recorded["evidence"]["path"]).is_absolute()
    (tmp_path / recorded["evidence"]["path"] / "response-000.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        selector.selected_checkpoint()


def test_default_plan_is_free_and_cannot_authorize_paid_execution():
    profile = load_recovery_profile()
    plan = build_recovery_plan(profile)
    assert plan["training_rollout_episodes"] == 8000
    assert plan["planned_validation_evaluations"] == 11
    assert plan["resolved_objective"]["kl"] == "k3_token"
    assert plan["resolved_objective"]["kl_coef"] == 0.1
    assert plan["ready_for_paid_execution"] is False
    assert plan["estimated_total_cost_usd"] is None
    assert plan["validation_judge_prompt_upper_bound"] is None
    assert len(plan["blockers"]) == 4
    assert plan["validation_decoding"]["do_sample"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_validation_is_greedy_and_restores_both_decoding_objects(tmp_path, fail):
    trainer, _, _, _, _ = selection_fixture.__wrapped__(tmp_path)
    original = AgentConfig(
        model_name="test-policy",
        do_sample=True,
        temperature=0.8,
        repetition_penalty=1.2,
    )
    cached = SimpleNamespace(do_sample=True, temperature=0.8, num_beams=3)
    trainer.agent.config = original
    trainer.agent.generation_config = cached

    async def respond(*args, **kwargs):
        assert trainer.agent.config is not original
        assert trainer.agent.config.do_sample is False
        assert trainer.agent.config.temperature == 0
        assert trainer.agent.config.repetition_penalty == 1
        assert trainer.agent.generation_config is not cached
        assert trainer.agent.generation_config.do_sample is False
        assert trainer.agent.generation_config.num_beams == 1
        if fail:
            raise RuntimeError("generation failed")
        return "The policy applies."

    trainer.agent.generate_response = respond

    class Environment:
        def clone(self):
            return Environment()

        async def run_episode(self, agent_fn, *, scenario, max_turns):
            return SimpleNamespace(
                turns=[ConversationTurn(role="assistant", content=await agent_fn())]
            )

    evaluate = support_judge_evaluator(recorder(tmp_path / "raw"), Environment())
    if fail:
        with pytest.raises(RuntimeError, match="generation failed"):
            await evaluate(trainer, [{"id": "validation"}])
    else:
        assert len(await evaluate(trainer, [{"id": "validation"}])) == 1
    assert trainer.agent.config is original
    assert original.do_sample is True and original.temperature == 0.8
    assert original.repetition_penalty == 1.2
    assert trainer.agent.generation_config is cached
    assert cached.do_sample is True and cached.num_beams == 3
    assert trainer.agent.model.training


def test_reviewed_plan_counts_all_validation_prompts_and_keeps_pricing_gate():
    profile = small_profile()
    data = cases(profile)
    plan = build_recovery_plan(profile, data)
    assert plan["validation_judge_prompt_upper_bound"] == 32 * 3 * 2 * 2
    assert plan["ready_for_paid_execution"] is False
    assert len(plan["blockers"]) == 3
    data["provenance"]["test_is_fresh"] = False
    with pytest.raises(ValueError, match="fresh-test"):
        validate_recovery_cases(data, profile)


@pytest.mark.parametrize(
    "field,value",
    [
        ("use_reference_model", False),
        ("beta", 0),
        ("eval_steps", 1),
        ("gradient_accumulation_steps", 2),
    ],
)
def test_recovery_profile_rejects_missing_safeguards(field, value):
    profile = small_profile()
    profile["training"][field] = value
    with pytest.raises(ValueError):
        validate_recovery_profile(profile)


def test_required_reference_clone_errors_propagate_without_partial_state():
    class Uncopyable(torch.nn.Linear):
        def __deepcopy__(self, memo):
            raise RuntimeError("reference allocation failed")

    trainer = MultiTurnGRPOTrainer(
        SimpleNamespace(model=Uncopyable(1, 1)),
        object(),
        config=SimpleNamespace(use_reference_model=True),
    )
    with pytest.raises(RuntimeError, match="reference allocation failed"):
        trainer._initialize_reference_model()
    assert trainer.reference_model is None


@pytest.mark.asyncio
async def test_native_recovery_batches_real_updates_and_recorded_baseline_selection(
    tmp_path,
):
    from tests.integration.test_replica_consensus import _policy
    from tests.unit.test_token_group_aggregation import _groups

    profile = small_profile()
    model = _policy()()
    agent = SimpleNamespace(
        model=model, config=AgentConfig(**profile["agent"]), tokenizer=None
    )
    data = cases(profile)
    judge = recorder(tmp_path / "judge-evidence", max_batches=3)

    async def response(*_):
        return "The provided store policy applies."

    agent.generate_response = response

    class Environment:
        def clone(self):
            return Environment()

        async def run_episode(self, agent_fn, *, scenario, max_turns):
            text = await agent_fn([], {})
            return SimpleNamespace(
                turns=[
                    ConversationTurn(role="user", content=scenario["user_query"]),
                    ConversationTurn(role="assistant", content=text),
                ]
            )

    trainer = create_recovery_trainer(
        agent,
        Environment(),
        profile=profile,
        cases=data,
        judge=judge,
        output_dir=tmp_path,
    )
    await trainer.initialize()
    initial = copy.deepcopy(model.state_dict())
    anchor = copy.deepcopy(trainer.reference_model.state_dict())
    seen = []

    async def rollouts(scenarios):
        seen.append([case["id"] for case in scenarios])
        groups = []
        for case in scenarios:
            group = copy.deepcopy(_groups()[0])
            group.scenario_id = case["id"]
            group.trajectories = [
                copy.deepcopy(t) for _ in range(4) for t in group.trajectories
            ]
            groups.append(group)
        return groups

    trainer.generate_trajectories = rollouts
    await trainer.train()
    assert trainer.global_step == 2
    assert len(seen) == 2 and all(len(batch) == 4 for batch in seen)
    assert not torch.equal(model.lookup.weight, initial["lookup.weight"])
    for key, value in anchor.items():
        torch.testing.assert_close(
            trainer.reference_model.state_dict()[key], value, rtol=0, atol=0
        )
    assert trainer.judge_selector.selected_checkpoint().name == "judge-selected-step-0"
    receipt = json.loads((tmp_path / "judge-selection.json").read_text())
    assert [row["step"] for row in receipt["evaluations"]] == [0, 1, 2]
    assert all(row.get("evidence") for row in receipt["evaluations"])
    plan = json.loads((tmp_path / "study-plan.json").read_text())
    assert plan["training_schedule"] == seen
    assert not plan["ready_for_paid_execution"]


@pytest.mark.parametrize(
    "failure",
    [
        "missing_reference",
        "trainable_reference",
        "dropout",
        "reward",
        "learning_rate",
        "agent_config",
    ],
)
def test_runtime_guard_checks_actual_components(failure):
    profile = small_profile()
    from stateset_agents.training.config import TrainingConfig

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0))
    reference = copy.deepcopy(model).eval()
    reference.requires_grad_(False)
    trainer = SimpleNamespace(
        config=TrainingConfig(**profile["training"]),
        agent=SimpleNamespace(model=model, config=AgentConfig(**profile["agent"])),
        reference_model=reference,
        reward_fn=SimpleNamespace(name="support_conversation_v3"),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-5),
    )
    guard = RecoveryRuntimeGuard(profile)
    guard.validate(trainer)
    if failure == "missing_reference":
        trainer.reference_model = None
    elif failure == "trainable_reference":
        reference.requires_grad_(True)
    elif failure == "dropout":
        model[1].p = 0.1
    elif failure == "reward":
        trainer.reward_fn.name = "support_conversation_v2"
    elif failure == "learning_rate":
        trainer.optimizer.param_groups[0]["lr"] = 2e-5
    else:
        trainer.agent.config.temperature = 0.5
    with pytest.raises(ValueError):
        guard.validate(trainer)
