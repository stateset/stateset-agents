"""Regression contracts for the local sampled workflow experiment."""

from types import SimpleNamespace

import pytest

from stateset_agents.training.sampled_workflow import (
    WorkflowAgent,
    run_seed,
    workflow_success_probabilities,
    workflow_tokenizer,
)

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def test_success_probability_includes_lookup_decision_and_finish():
    tokenizer = workflow_tokenizer()
    probabilities = torch.zeros(len(tokenizer), len(tokenizer))
    for observation, action, probability in (
        ("request", "lookup", 0.8),
        ("eligible", "refund", 0.75),
        ("ineligible", "deny", 0.25),
        ("resolved", "finish", 0.9),
    ):
        row = probabilities[tokenizer.convert_tokens_to_ids(observation)]
        row[tokenizer.convert_tokens_to_ids(action)] = probability
        row[tokenizer.pad_token_id] = 1 - probability

    class TablePolicy(torch.nn.Module):
        device = torch.device("cpu")

        def forward(self, token_ids):
            return SimpleNamespace(logits=probabilities[token_ids].log())

    model = TablePolicy()
    actual = workflow_success_probabilities(WorkflowAgent(model, tokenizer))
    assert actual == pytest.approx(
        {"eligible": 0.54, "ineligible": 0.18, "balanced_mean": 0.36}
    )
    assert model.training


@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.parametrize("balanced", [False, True])
async def test_batch_modes_preserve_rollout_budget_and_seed(
    tmp_path, monkeypatch, balanced
):
    from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

    batches = []
    generate = MultiTurnGRPOTrainer.generate_trajectories

    async def record(self, scenarios):
        groups = await generate(self, scenarios)
        batches.append(
            [
                (case["eligible"], len(group.trajectories))
                for case, group in zip(scenarios, groups, strict=True)
            ]
        )
        return groups

    monkeypatch.setattr(MultiTurnGRPOTrainer, "generate_trajectories", record)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        report = await run_seed(17, tmp_path, updates=2, balanced_batches=balanced)
    finally:
        torch.set_num_threads(threads)
    expected = (
        [[(False, 12), (True, 12)]] * 2 if balanced else [[(False, 24)], [(True, 24)]]
    )
    assert batches == expected
    assert report["optimizer_updates"] == 2
    assert report["training_rollouts"] == 48
    assert 48 <= report["training_action_tokens"] <= 3 * 48
    assert report["resolved_config"]["seed"] == 17
    assert report["reload_verified"]
