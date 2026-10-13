"""Orchestration tests for the on-pod flagship workload (fakes only, no GPU)."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"


def _load(name: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, BENCHMARKS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


workload = _load("flagship_workload")

VERSION = "9.9.9"
RUBRIC = "c" * 40
CONFIG = {
    "num_train_examples": 3,
    "num_eval_examples": 2,
    "max_wall_clock_seconds": 21600,
    "max_cost_usd_per_seed": 25.0,
    "temperature": 0.0,
    "max_turns": 2,
    "use_lora": True,
    "bf16": True,
    "learning_rate": 2e-5,
    "num_generations": 4,
    "scenarios_per_step": 2,
    "max_steps": 3,
    "lora_r": 8,
    "lora_alpha": 16,
    "max_completion_length": 32,
    "rollout_temperature": 1.0,
    "beta": 0.02,
}
JUDGE = {"model": "judge/model", "revision": "d" * 40, "rubric_revision": RUBRIC}


@dataclass
class FakeScenario:
    id: str
    split: str
    user_turns: list[str]
    intent: str = "refund"
    persona: str = "calm"
    product: str = "Widget"
    policy_facts: list[str] = field(default_factory=lambda: ["30-day returns"])
    must_acknowledge: list[str] = field(default_factory=lambda: ["refund"])
    must_avoid: list[str] = field(default_factory=lambda: ["cvv"])
    expected_resolution: str = "Start a return."

    def to_scenario(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_query": self.user_turns[0],
            "intent": self.intent,
            "must_acknowledge": list(self.must_acknowledge),
            "must_avoid": list(self.must_avoid),
        }


def _scenarios(split: str, count: int) -> list[FakeScenario]:
    return [
        FakeScenario(
            id=f"{split}-{index}",
            split=split,
            user_turns=[f"{split} {index} turn {t}" for t in range(1 + index % 3)],
        )
        for index in range(count)
    ]


def _sha(value: dict[str, Any]) -> str:
    return hashlib.sha256(workload.canonical_json(value).encode()).hexdigest()


def _args(tmp_path: Path, **updates: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "seed": 42,
        "mode": "measured",
        "framework_version": VERSION,
        "model": "org/policy-7b",
        "model_revision": "a" * 40,
        "dataset_revision": "b" * 40,
        "dataset_source": None,
        "config_json": workload.canonical_json(CONFIG),
        "config_sha256": _sha(CONFIG),
        "judge_json": workload.canonical_json(JUDGE),
        "judge_sha256": _sha(JUDGE),
        "output": tmp_path / "result.json",
        "artifact_dir": tmp_path / "artifacts",
        "scratch_dir": tmp_path / "scratch",
        "smoke": False,
        "smoke_model": "tiny/model",
    }
    values.update(updates)
    return argparse.Namespace(**values)


class FakeReward:
    async def compute_reward(self, turns: Any, context: Any = None) -> Any:
        text = " ".join(turn.content for turn in turns if turn.role == "assistant")
        return argparse.Namespace(score=1.0 if "adapter" in text else 0.5)


class FakePhases:
    def __init__(self, *, rubric: str = RUBRIC, aborted: Any = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.rubric = rubric
        self.aborted = aborted

    def __call__(self, name: str, payload: Any) -> dict[str, Any]:
        self.calls.append((name, dict(payload)))
        if name == "rollout":
            tag = "adapter" if payload["adapter_path"] else "base"

            def generate(conversations: Any) -> list[str]:
                return [f"{tag} reply #{len(c)}" for c in conversations]

            environment = workload.make_support_environment(
                payload["scenarios"], payload["max_turns"]
            )
            transcripts = asyncio.run(
                workload.rollout_episodes(environment, payload["scenarios"], generate)
            )
            return {
                "transcripts": {
                    k: [list(p) for p in v] for k, v in transcripts.items()
                },
                "backend": "fake",
                "peak_vram_mb": 1000.0 if tag == "base" else 1100.0,
            }
        if name == "train":
            adapter = Path(payload["adapter_dir"])
            adapter.mkdir(parents=True, exist_ok=True)
            (adapter / "adapter_config.json").write_text("{}")
            return {
                "train_log": {"reward_mean": [0.1, 0.2, 0.3]},
                "steps": 3,
                "optimizer_steps": 3,
                "episodes": 3 * 2 * 4,
                "aborted": self.aborted,
                "peak_vram_mb": 52000.0,
            }
        if name == "judge":
            groups = {}
            for group, mean, disagreement in (
                ("baseline", 0.5, 0.02),
                ("final", 0.6, 0.04),
            ):
                items = payload["groups"][group]
                groups[group] = {
                    "results": [{"id": item["id"], "score": mean} for item in items],
                    "summary": {
                        "mean_score": mean,
                        "self_disagreement": disagreement,
                        "parse_failure_rate": 0.01,
                        "n": float(len(items)),
                    },
                }
            return {
                "groups": groups,
                "rubric_revision": self.rubric,
                "backend": "fake",
                "peak_vram_mb": 30000.0,
            }
        raise AssertionError(name)


def _deps(phases: FakePhases, **updates: Any) -> Any:
    kwargs: dict[str, Any] = {
        "run_phase": phases,
        "installed_version": lambda: VERSION,
        "rubric_revision": lambda: RUBRIC,
        "load_dataset": lambda source, revision, split: _scenarios(split, 4),
        "dataset_name": lambda: "stateset/customer-support-multiturn-v1",
        "snapshot": lambda model, revision: f"/snapshots/{revision}",
        "reward_fn": FakeReward,
        "hardware": lambda: {
            "gpu": "NVIDIA H100 80GB HBM3",
            "gpu_count": 1,
            "cuda": "12.9",
            "driver": "570.1",
        },
    }
    kwargs.update(updates)
    return workload.WorkloadDeps(**kwargs)


def test_run_workload_produces_metrics_and_artifacts(tmp_path: Path) -> None:
    phases = FakePhases()
    outcome = workload.run_workload(_args(tmp_path), _deps(phases))

    assert [name for name, _ in phases.calls] == [
        "rollout",
        "train",
        "rollout",
        "judge",
    ]
    metrics = outcome["metrics"]
    assert metrics["baseline_score"] == 0.5
    assert metrics["final_score"] == 0.6
    assert metrics["judge_self_disagreement"] == 0.04
    assert metrics["train_examples"] == 3
    assert metrics["eval_examples"] == 2
    assert metrics["peak_vram_mb"] == 52000.0
    assert outcome["status"] == "completed" and outcome["measured"] is True
    diagnostics = outcome["diagnostics"]
    assert diagnostics["baseline_rule_reward_mean"] == 0.5
    assert diagnostics["final_rule_reward_mean"] == 1.0
    assert diagnostics["final_parse_failure_rate"] == 0.01
    assert diagnostics["train_steps"] == 3 and diagnostics["train_episodes"] == 24
    assert diagnostics["train_unique_scenarios"] == 3
    assert "multi-turn GSPO" in diagnostics["training"]

    # Baseline and final are the same eval scenarios, rollout and decoding;
    # only the adapter differs.
    first, final = phases.calls[0][1], phases.calls[2][1]
    assert first["model_path"] == "/snapshots/" + "a" * 40
    assert first["adapter_path"] is None
    assert final["adapter_path"] == str(tmp_path / "artifacts" / "adapter")
    assert {k: v for k, v in first.items() if k != "adapter_path"} == {
        k: v for k, v in final.items() if k != "adapter_path"
    }
    assert [s["id"] for s in first["scenarios"]] == ["eval-0", "eval-1"]
    assert first["temperature"] == 0.0

    # Training gets the train scenarios (with scripted turns and reward
    # context) and a seeded schedule of scenarios_per_step per step.
    train_payload = phases.calls[1][1]
    assert [s["id"] for s in train_payload["scenarios"]] == [
        "train-0",
        "train-1",
        "train-2",
    ]
    assert train_payload["scenarios"][1]["user_turns"] == [
        "train 1 turn 0",
        "train 1 turn 1",
    ]
    assert train_payload["scenarios"][0]["must_acknowledge"] == ["refund"]
    assert train_payload["schedule"] == workload.training_schedule(3, 2, 3, 42)

    # The judge sees exactly the multi-turn conversations.
    judged = {
        item["id"]: item["transcript"] for item in phases.calls[3][1]["groups"]["final"]
    }
    assert judged["eval-1"] == [
        ["user", "eval 1 turn 0"],
        ["assistant", "adapter reply #2"],
        ["user", "eval 1 turn 1"],
        ["assistant", "adapter reply #4"],
    ]
    judge_payload = phases.calls[3][1]
    assert judge_payload["model"] == "judge/model"
    assert judge_payload["revision"] == "d" * 40
    # The judge loads the pinned local snapshot resolved before any GPU phase.
    assert judge_payload["model_path"] == "/snapshots/" + "d" * 40

    artifacts = tmp_path / "artifacts"
    for relative in (
        "transcripts/baseline.jsonl",
        "transcripts/final.jsonl",
        "judge/baseline.jsonl",
        "judge/final.jsonl",
        "train_log.json",
        "adapter/adapter_config.json",
    ):
        assert (artifacts / relative).is_file(), relative


def test_main_writes_completed_result(tmp_path: Path) -> None:
    args = _args(tmp_path)
    argv = [
        "--seed",
        "42",
        "--mode",
        "preflight",
        "--framework-version",
        VERSION,
        "--model",
        args.model,
        "--model-revision",
        args.model_revision,
        "--dataset-revision",
        args.dataset_revision,
        "--config-json",
        args.config_json,
        "--config-sha256",
        args.config_sha256,
        "--judge-json",
        args.judge_json,
        "--judge-sha256",
        args.judge_sha256,
        "--output",
        str(args.output),
        "--artifact-dir",
        str(args.artifact_dir),
    ]
    assert workload.main(argv, deps=_deps(FakePhases())) == 0
    result = json.loads(args.output.read_text())
    assert result["status"] == "completed" and result["measured"] is True
    assert result["preflight"] is True and result["mode"] == "preflight"
    assert result["config_sha256"] == _sha(CONFIG)
    assert result["hardware"]["gpu"] == "NVIDIA H100 80GB HBM3"
    assert (args.artifact_dir / "workload_result.json").is_file()


def test_smoke_is_never_measured(tmp_path: Path) -> None:
    phases = FakePhases()
    args = _args(tmp_path, smoke=True)
    outcome = workload.run_workload(args, _deps(phases))
    assert outcome["status"] == "smoke" and outcome["measured"] is False
    assert phases.calls[0][1]["model_path"] == "/snapshots/None"
    document = workload.build_result(args, outcome, {}, None)
    assert document["status"] == "smoke" and document["measured"] is False


@pytest.mark.parametrize(
    ("updates", "deps_updates", "message"),
    [
        ({"config_sha256": "0" * 64}, {}, "config sha256 mismatch"),
        ({"judge_sha256": "0" * 64}, {}, "judge sha256 mismatch"),
        ({}, {"installed_version": lambda: "0.0.1"}, "does not match the pinned"),
        ({}, {"rubric_revision": lambda: "e" * 40}, "rubric revision"),
        (
            {},
            {"load_dataset": lambda source, revision, split: _scenarios(split, 1)},
            "are required",
        ),
    ],
)
def test_fail_closed_before_any_phase(
    tmp_path: Path, updates: dict[str, Any], deps_updates: dict[str, Any], message: str
) -> None:
    phases = FakePhases()
    with pytest.raises(workload.WorkloadError, match=message):
        workload.run_workload(_args(tmp_path, **updates), _deps(phases, **deps_updates))
    assert phases.calls == []


def test_phase_judge_loads_snapshot_and_records_findings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stateset_agents.evaluation import support_judge

    loaded: list[tuple[str, Any]] = []

    def fake_vllm(model: str, revision: Any) -> Any:
        loaded.append((model, revision))
        return lambda prompts: [
            json.dumps(dict.fromkeys(support_judge.CRITERIA, 3)) for _ in prompts
        ]

    monkeypatch.setattr(support_judge, "vllm_generate_fn", fake_vllm)
    scenario = {
        "policy_facts": ["Duties are the customer's responsibility."],
        "expected_resolution": "Decline reimbursement of customs duties.",
        "must_avoid": ["reimburse"],
    }
    out = workload.phase_judge(
        {
            "model": "org/judge",
            "revision": "d" * 40,
            "model_path": "/snapshots/" + "d" * 40,
            "backend": "auto",
            "groups": {
                "baseline": [
                    {
                        "id": "s-1",
                        "scenario": scenario,
                        "transcript": [
                            ["user", "Will you cover the duties?"],
                            ["assistant", "Hi [Customer's Name], I'll reimburse you."],
                        ],
                    },
                    {
                        "id": "s-2",
                        "scenario": scenario,
                        "transcript": [
                            ["user", "Will you cover the duties?"],
                            ["assistant", "Sorry, duties are not covered by us."],
                        ],
                    },
                ]
            },
        }
    )
    assert loaded == [("/snapshots/" + "d" * 40, None)]
    group = out["groups"]["baseline"]
    first, second = group["results"]
    assert first["findings"]["must_avoid_hits"][0]["phrase"] == "reimburse"
    assert first["findings"]["placeholders"][0]["text"] == "[Customer's Name]"
    assert second["findings"] == {
        "must_avoid_hits": [],
        "placeholders": [],
        "repeated_turns": [],
    }
    assert group["findings_summary"] == {
        "must_avoid_hits": 1,
        "placeholders": 1,
        "repeated_turns": 0,
    }
    assert out["rubric_revision"] == support_judge.rubric_revision()


def test_judge_phase_rubric_mismatch_fails(tmp_path: Path) -> None:
    with pytest.raises(workload.WorkloadError, match="rubric revision"):
        workload.run_workload(_args(tmp_path), _deps(FakePhases(rubric="f" * 40)))


def test_training_abort_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(workload.WorkloadError, match="aborted"):
        workload.run_workload(_args(tmp_path), _deps(FakePhases(aborted="zero")))


def test_main_records_failure_without_completion(tmp_path: Path) -> None:
    args = _args(tmp_path, config_sha256="0" * 64)
    argv = []
    for key in (
        "seed",
        "framework_version",
        "model",
        "model_revision",
        "dataset_revision",
        "config_json",
        "config_sha256",
        "judge_json",
        "judge_sha256",
        "output",
        "artifact_dir",
    ):
        argv += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    assert workload.main(argv, deps=_deps(FakePhases())) == 1
    result = json.loads(args.output.read_text())
    assert result["status"] == "failed" and result["measured"] is False
    assert "sha256 mismatch" in result["error"]


# --------------------------------------------------------------------------- #
# Multi-turn environment rollout (shared by training and both evaluations)
# --------------------------------------------------------------------------- #


def _record(sid: str, turns: list[str]) -> dict[str, Any]:
    return workload.scenario_record(
        FakeScenario(id=sid, split="eval", user_turns=turns)
    )


def test_rollout_follows_the_scripted_environment_batched_per_turn() -> None:
    scenarios = [
        _record("a", ["hi", "where is it", "x"])["scenario"],
        _record("b", ["refund please"])["scenario"],
    ]
    batches: list[list[list[dict[str, str]]]] = []

    def generate(conversations: Any) -> list[str]:
        batches.append([list(c) for c in conversations])
        # A farewell must not end a scripted episode early.
        return [f"reply{len(batches)} goodbye"] * len(conversations)

    environment = workload.make_support_environment(scenarios, max_turns=2)
    transcripts = asyncio.run(
        workload.rollout_episodes(environment, scenarios, generate)
    )

    assert [len(batch) for batch in batches] == [2, 1]
    system = batches[0][0][0]
    assert system["role"] == "system"
    assert "30-day returns" in system["content"] and "Widget" in system["content"]
    # Grading keys never reach the policy.
    assert "Start a return" not in system["content"]
    assert "cvv" not in system["content"]
    assert batches[1][0][1:] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "reply1 goodbye"},
        {"role": "user", "content": "where is it"},
    ]
    assert transcripts["a"] == [
        ("user", "hi"),
        ("assistant", "reply1 goodbye"),
        ("user", "where is it"),
        ("assistant", "reply2 goodbye"),
    ]  # max_turns caps the three-turn script at two replies
    assert transcripts["b"] == [
        ("user", "refund please"),
        ("assistant", "reply1 goodbye"),
    ]


def test_trainer_episode_and_eval_rollout_see_identical_conversations() -> None:
    scenario = _record("a", ["hi", "and then?"])["scenario"]
    seen_by_trainer: list[list[dict[str, str]]] = []

    async def agent_fn(history: Any, context: Any) -> str:
        seen_by_trainer.append([dict(m) for m in history])
        return f"r{len(seen_by_trainer)}"

    environment = workload.make_support_environment([scenario], max_turns=4)
    trajectory = asyncio.run(environment.run_episode(agent_fn, scenario))
    seen_by_eval: list[list[dict[str, str]]] = []

    def generate(conversations: Any) -> list[str]:
        seen_by_eval.extend([dict(m) for m in c] for c in conversations)
        return [f"r{len(seen_by_eval)}"]

    transcripts = asyncio.run(
        workload.rollout_episodes(environment, [scenario], generate)
    )
    assert seen_by_eval == seen_by_trainer
    assert [(t.role, t.content) for t in trajectory.turns if t.role != "system"] == (
        transcripts["a"]
    )


def test_flagship_agent_keeps_sampled_text_verbatim() -> None:
    from stateset_agents.core.agent_config import AgentConfig

    agent = workload.flagship_agent_cls()(AgentConfig(model_name="stub://flagship"))
    text = "First paragraph.\n\nUser: second paragraph."
    assert agent._clean_response(f"  {text}\n") == text
    # The base agent appends a canned billing sentence here.
    messages = [{"role": "user", "content": "I was charged twice"}]
    assert agent._finalize_response("Sorry about that.", messages) == (
        "Sorry about that."
    )


# --------------------------------------------------------------------------- #
# Reward: composite averaged over every assistant turn
# --------------------------------------------------------------------------- #


class RecordingComposite:
    def __init__(self) -> None:
        self.calls: list[tuple[list[tuple[str, str]], Any]] = []

    async def compute_reward(self, turns: Any, context: Any = None) -> Any:
        self.calls.append(([(t.role, t.content) for t in turns], context))
        reply = turns[-1].content
        return argparse.Namespace(
            score=float(reply.count("!")), breakdown={"marks": float(len(reply))}
        )


def test_conversation_reward_averages_each_assistant_turn() -> None:
    from stateset_agents.core.trajectory import ConversationTurn

    composite = RecordingComposite()
    reward = workload.ConversationSupportReward(composite)
    turns = [
        ConversationTurn(role="system", content="policy"),
        ConversationTurn(role="user", content="u1"),
        ConversationTurn(role="assistant", content="a!"),
        ConversationTurn(role="user", content="u2"),
        ConversationTurn(role="assistant", content="b!!!"),
    ]
    context = {"must_acknowledge": ["refund"]}
    result = asyncio.run(reward.compute_reward(turns, context))

    assert result.score == pytest.approx((1 + 3) / 2)
    assert result.breakdown["turn_0"] == 1.0 and result.breakdown["turn_1"] == 3.0
    assert result.breakdown["assistant_turns"] == 2.0
    assert result.breakdown["marks"] == pytest.approx((2 + 4) / 2)
    # Each reply is scored alone, after the customer turns that preceded it,
    # with the scenario context.
    assert [call[0] for call in composite.calls] == [
        [("user", "u1"), ("assistant", "a!")],
        [("user", "u1"), ("user", "u2"), ("assistant", "b!!!")],
    ]
    assert all(call[1] is context for call in composite.calls)


def test_conversation_reward_with_real_composite_credits_every_turn() -> None:
    from stateset_agents.core.trajectory import ConversationTurn
    from stateset_agents.data.customer_support_bench import SupportRewardComposite

    context = {"must_acknowledge": ["refund"], "must_avoid": [], "intent": "refund"}
    good = (
        "I'm sorry to hear that. I can start your refund today; please share "
        "order #1234 and you will get a confirmation within 2 business days."
    )
    turns = [
        ConversationTurn(role="user", content="I want a refund"),
        ConversationTurn(role="assistant", content=good),
        ConversationTurn(role="user", content="ok"),
        ConversationTurn(role="assistant", content="ok"),
    ]
    composite = SupportRewardComposite()
    wrapped = asyncio.run(
        workload.ConversationSupportReward(composite).compute_reward(turns, context)
    )
    singles = [
        asyncio.run(composite.compute_reward(turns[:2], context)).score,
        asyncio.run(
            composite.compute_reward([turns[0], turns[2], turns[3]], context)
        ).score,
    ]
    assert wrapped.score == pytest.approx(sum(singles) / 2)
    assert singles[0] > singles[1]  # the weak second reply costs the episode


def test_conversation_reward_without_assistant_turns_is_zero() -> None:
    result = asyncio.run(
        workload.ConversationSupportReward(RecordingComposite()).compute_reward(
            [{"role": "user", "content": "hello"}], {}
        )
    )
    assert result.score == 0.0


# --------------------------------------------------------------------------- #
# Training plan and loop (fake trainer)
# --------------------------------------------------------------------------- #


def test_training_schedule_is_seeded_and_covers_each_epoch() -> None:
    first = workload.training_schedule(10, 4, 5, seed=42)
    assert first == workload.training_schedule(10, 4, 5, seed=42)
    assert first != workload.training_schedule(10, 4, 5, seed=43)
    assert len(first) == 5 and all(len(batch) == 4 for batch in first)
    flat = [index for batch in first for index in batch]
    assert sorted(flat[:10]) == list(range(10))  # no repeat within an epoch
    assert sorted(flat[10:20]) == list(range(10))
    with pytest.raises(workload.WorkloadError, match="exceeds"):
        workload.training_schedule(3, 4, 1, seed=0)


def test_nonzero_advantage_fraction() -> None:
    assert workload.nonzero_advantage_fraction([[0.5, 0.5], [1.0, 1.0]]) == 0.0
    assert workload.nonzero_advantage_fraction([[0.0, 1.0], [0.3, 0.3]]) == 0.5
    assert workload.nonzero_advantage_fraction([[0.0, 1.0, 0.5]]) == pytest.approx(
        2 / 3
    )


@dataclass
class FakeTurn:
    role: str
    content: str


@dataclass
class FakeTrajectory:
    total_reward: float
    turns: list[FakeTurn]


@dataclass
class FakeGroup:
    scenario_id: str
    trajectories: list[FakeTrajectory]


class FakeTrainer:
    def __init__(
        self,
        rewards: Any,
        *,
        drop: bool = False,
        path: str = "token",
    ) -> None:
        self.rewards = rewards
        self.drop = drop
        self.path = path
        self.generated: list[tuple[list[str], int]] = []
        self.trained: list[int] = []

    async def generate_trajectories(
        self, scenarios: Any, num_generations: int | None = None
    ) -> list[FakeGroup]:
        assert num_generations is not None
        self.generated.append(([s["id"] for s in scenarios], num_generations))
        step = len(self.generated)
        groups = []
        for scenario in scenarios:
            count = num_generations - (1 if self.drop else 0)
            groups.append(
                FakeGroup(
                    scenario_id=scenario["id"],
                    trajectories=[
                        FakeTrajectory(
                            total_reward=self.rewards(step, index),
                            turns=[
                                FakeTurn("system", "p"),
                                FakeTurn("user", "u"),
                                FakeTurn("assistant", "a"),
                                FakeTurn("user", "u"),
                                FakeTurn("assistant", "b"),
                            ],
                        )
                        for index in range(count)
                    ],
                )
            )
        return groups

    async def training_step(self, groups: Any) -> dict[str, Any]:
        self.trained.append(len(groups))
        return {
            "path": self.path,
            "total_loss": 0.1,
            "kl_penalty": 0.0,
            "ratio_mean": 1.0,
            "clip_fraction": 0.0,
            "learning_rate": 1e-5,
            "optimizer_step": True,
        }


def _loop(trainer: FakeTrainer, steps: int = 3, zero_steps: int = 10) -> Any:
    from stateset_agents.training.callbacks import ZeroSignalGuard

    scenarios = [{"id": f"s{i}"} for i in range(4)]
    schedule = workload.training_schedule(4, 2, steps, seed=7)
    result = asyncio.run(
        workload.run_training_loop(
            trainer, scenarios, schedule, 3, ZeroSignalGuard(max_zero_steps=zero_steps)
        )
    )
    return result, schedule


def test_training_loop_follows_schedule_on_policy() -> None:
    trainer = FakeTrainer(lambda step, index: float(index))
    result, schedule = _loop(trainer)

    assert trainer.generated == [([f"s{i}" for i in batch], 3) for batch in schedule]
    assert trainer.trained == [2, 2, 2]  # one update per sampled batch
    assert result["steps"] == 3 and result["optimizer_steps"] == 3
    assert result["episodes"] == 3 * 2 * 3 and result["aborted"] is None
    log = result["train_log"]
    assert log["reward_mean"] == [1.0, 1.0, 1.0]
    assert log["nonzero_advantage_fraction"][0] == pytest.approx(2 / 3)
    assert log["assistant_turns"] == [12, 12, 12]
    assert log["scenario_ids"][0] == [f"s{i}" for i in schedule[0]]


def test_zero_signal_guard_aborts_constant_reward_runs() -> None:
    trainer = FakeTrainer(lambda step, index: 0.7)
    result, _ = _loop(trainer, steps=5, zero_steps=2)
    assert result["steps"] == 2
    assert "identically zero" in result["aborted"]


def test_signal_resets_the_zero_streak() -> None:
    trainer = FakeTrainer(lambda step, index: float(index * (step % 2)))
    result, _ = _loop(trainer, steps=5, zero_steps=2)
    assert result["aborted"] is None and result["steps"] == 5


def test_incomplete_group_fails_closed() -> None:
    with pytest.raises(workload.WorkloadError, match="2 of 3 episodes"):
        _loop(FakeTrainer(lambda step, index: float(index), drop=True))


def test_sequence_fallback_path_fails_closed() -> None:
    with pytest.raises(workload.WorkloadError, match="token path"):
        _loop(FakeTrainer(lambda step, index: float(index), path="sequence"))


# --------------------------------------------------------------------------- #
# Trainer / agent wiring
# --------------------------------------------------------------------------- #


def test_training_config_is_multi_turn_gspo_with_kl_reference() -> None:
    from stateset_agents.training.config import TrainingConfig

    config = {**CONFIG, "clip_range_left": 3e-4, "clip_range_right": 4e-4}
    kwargs = workload.training_config_kwargs(
        {"model_path": "/snap", "output_dir": "/out", "seed": 7}, config
    )
    assert kwargs["objective"] == "gspo"
    assert kwargs["objective_overrides"] == {"clip_low": 3e-4, "clip_high": 4e-4}
    assert kwargs["beta"] == 0.02 and kwargs["use_reference_model"] is True
    assert kwargs["num_generations"] == 4 and kwargs["learning_rate"] == 2e-5
    assert kwargs["gradient_accumulation_steps"] == 1
    assert kwargs["num_gradient_updates"] == 1 and kwargs["entropy_coef"] == 0.0
    assert kwargs["bf16"] is True and kwargs["seed"] == 7
    assert kwargs["save_steps"] > kwargs["num_episodes"] == 3
    built = TrainingConfig(**kwargs)
    assert built.objective == "gspo" and built.max_completion_length == 32

    no_kl = workload.training_config_kwargs(
        {"model_path": "/snap", "output_dir": "/out", "seed": 7},
        {**CONFIG, "beta": 0.0},
    )
    assert (
        no_kl["use_reference_model"] is False and no_kl["objective_overrides"] is None
    )


def test_agent_config_wires_lora_bf16_and_sampling() -> None:
    from stateset_agents.core.agent_config import AgentConfig

    kwargs = workload.agent_config_kwargs(
        {"model_path": "/snapshots/abc", "bf16": True},
        temperature=1.0,
        max_new_tokens=192,
        lora={**CONFIG, "lora_dropout": 0.05},
    )
    assert kwargs["model_name"] == "/snapshots/abc"
    assert kwargs["torch_dtype"] == "bfloat16" and kwargs["use_chat_template"] is True
    assert kwargs["do_sample"] is True and kwargs["temperature"] == 1.0
    assert kwargs["repetition_penalty"] == 1.0 and kwargs["max_new_tokens"] == 192
    assert kwargs["system_prompt"] is None  # the environment owns the system turn
    assert kwargs["use_peft"] is True
    assert kwargs["peft_config"] == {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": list(workload.DEFAULT_LORA_TARGET_MODULES),
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    AgentConfig(**kwargs)

    greedy = workload.agent_config_kwargs(
        {"model_path": "/m", "bf16": True},
        temperature=0.0,
        max_new_tokens=32,
        adapter_path="/adapter",
    )
    assert greedy["do_sample"] is False and greedy["peft_path"] == "/adapter"
    assert "use_peft" not in greedy
    AgentConfig(**greedy)


def test_disable_top_k_and_gradient_checkpointing_helpers() -> None:
    agent = argparse.Namespace(generation_config=argparse.Namespace(top_k=50))
    workload.disable_top_k(agent)
    assert agent.generation_config.top_k == 0

    calls: list[Any] = []

    class Model:
        def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
            calls.append(kwargs)

        def enable_input_require_grads(self) -> None:
            calls.append("inputs")

    assert workload.enable_gradient_checkpointing(Model()) is True
    assert calls == [
        {"gradient_checkpointing_kwargs": {"use_reentrant": False}},
        "inputs",
    ]
    assert workload.enable_gradient_checkpointing(object()) is False


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"scenarios_per_step": 0}, "scenarios_per_step"),
        ({"rollout_temperature": 0.0}, "rollout_temperature"),
        ({"beta": -0.1}, "beta"),
        ({"lora_target_modules": []}, "lora_target_modules"),
    ],
)
def test_validate_config_rejects_bad_training_knobs(
    updates: dict[str, Any], message: str
) -> None:
    with pytest.raises(workload.WorkloadError, match=message):
        workload.validate_config({**CONFIG, **updates})
    missing = {k: v for k, v in CONFIG.items() if k != "scenarios_per_step"}
    with pytest.raises(workload.WorkloadError, match="missing: scenarios_per_step"):
        workload.validate_config(missing)


def test_pinned_manifest_config_is_valid_for_the_workload() -> None:
    manifest = json.loads((BENCHMARKS / "flagship_manifest.json").read_text())
    workload.validate_config(manifest["config"])
    config = manifest["config"]
    assert config["scenarios_per_step"] <= config["num_train_examples"]
    # Worst-case pod lifetime at the pinned price stays inside the cap.
    assert config["max_wall_clock_seconds"] / 3600 * 3.99 <= (
        config["max_cost_usd_per_seed"]
    )
