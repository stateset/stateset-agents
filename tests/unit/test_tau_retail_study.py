"""Three-arm τ-bench retail study: protocol, schedule, SFT update, comparison."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.training.tau_retail_study import (
    assistant_turns,
    compare,
    evaluation_sampling,
    load_protocol,
    main,
    sft_update,
    training_schedule,
)

PROTOCOL: dict[str, Any] = {
    "measured": False,
    "model": "org/model",
    "model_revision": "0" * 40,
    "seeds": [1, 2, 3],
    "updates": 4,
    "tasks_per_update": 2,
    "generations": 2,
    "learning_rate": 0.001,
    "lora_r": 4,
    "max_prompt_tokens": 16,
    "max_new_tokens": 8,
    "sft_turns_per_update": 2,
    "evaluation_trials": 4,
    "evaluation_temperature": 0.0,
    "analysis_seed": 5,
    "bootstrap_replicates": 200,
    "environment": {"user_llm": "fake"},
}


def write(tmp_path: Path, **changes: Any) -> Path:
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(PROTOCOL | changes))
    return path


def test_protocol_rejects_bad_fields(tmp_path: Path) -> None:
    assert load_protocol(write(tmp_path))["updates"] == 4
    for changes in ({"updates": 0}, {"seeds": [1, 1]}, {"learning_rate": 1}):
        with pytest.raises(ValueError):
            load_protocol(write(tmp_path, **changes))


def test_schedule_is_deterministic_and_shared_by_arms() -> None:
    ids = [str(i) for i in range(5)]
    first = training_schedule(ids, PROTOCOL, seed=3)
    assert first == training_schedule(ids, PROTOCOL, seed=3)
    assert first != training_schedule(ids, PROTOCOL, seed=4)
    assert [len(batch) for batch in first] == [2, 2, 2, 2]
    assert {case["seed"] for case in first[1]} == {4}
    assert {case["id"] for batch in first for case in batch} == set(ids)


def test_measured_protocol_refuses_task_subsets(tmp_path: Path) -> None:
    pytest.importorskip("tau2")
    with pytest.raises(SystemExit):
        main(
            [
                "--protocol",
                str(write(tmp_path, measured=True)),
                "--output",
                str(tmp_path / "out"),
                "--limit-test",
                "2",
            ]
        )


def test_evaluation_sampling_restores_training_settings() -> None:
    agent = SimpleNamespace(config=SimpleNamespace(temperature=1.0, do_sample=True))
    with evaluation_sampling(agent, 0.0):
        assert agent.config.do_sample is False
    assert (agent.config.temperature, agent.config.do_sample) == (1.0, True)


def test_sft_update_raises_likelihood_of_imitated_tokens() -> None:
    torch = pytest.importorskip("torch")
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=16, n_positions=8, n_embd=16, n_layer=1, n_head=2)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    turn = ConversationTurn(
        role="assistant",
        content="x",
        metadata={"prompt_token_ids": [1, 2, 3], "token_ids": [4, 5]},
    )
    losses = [sft_update(model, optimizer, [turn], 16) for _ in range(20)]
    assert losses[-1] < losses[0] * 0.5
    assert not model.training
    trajectory = SimpleNamespace(
        turns=[turn.to_dict(), ConversationTurn(role="user", content="hi")]
    )
    assert [t.content for t in assistant_turns(trajectory)] == ["x"]


def report(seed: int, prompted: int, sft: int, rl: int) -> dict[str, Any]:
    arms = {"prompted": prompted, "sft": sft, "rl": rl}
    return {
        "seed": seed,
        "measured": False,
        "test_tasks": ["a", "b"],
        "arms": {
            arm: {"passed": {"a": passed, "b": passed}} for arm, passed in arms.items()
        },
    }


def test_compare_reports_rl_minus_sft_as_primary() -> None:
    result = compare(PROTOCOL, [report(s, 1, 2, 4) for s in (1, 2, 3)])
    primary = result["comparisons"]["rl-minus-sft"]
    assert result["primary"] == "rl minus sft" and result["measured"] is False
    assert primary["delta_rate"] == pytest.approx(0.5)
    assert primary["positive_seeds"] == 3
    assert result["comparisons"]["sft-minus-prompted"]["delta_rate"] == pytest.approx(
        0.25
    )
    reports = [report(1, 1, 2, 4), report(2, 1, 2, 4) | {"test_tasks": ["a"]}]
    with pytest.raises(ValueError):
        compare(PROTOCOL, reports)
