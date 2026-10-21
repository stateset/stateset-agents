"""The golden workflow trains real tensors and reloads its actual exported policy."""

import json
from pathlib import Path

import pytest
import torch

from stateset_agents.training.improvement_pipeline import run_improvement_pipeline


async def test_trace_to_policy_updates_exports_evaluates_and_serves_real_model(
    tmp_path,
):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    records = [
        {
            "source_id": split,
            "timestamp": timestamp,
            "messages": [
                {"role": "user", "content": split},
                {"role": "assistant", "content": split},
            ],
            "reward": 1,
        }
        for split, timestamp in [
            ("train", "2026-01-01T00:00:00Z"),
            ("validation", "2026-02-01T00:00:00Z"),
            ("test", "2026-03-01T00:00:00Z"),
        ]
    ]
    seen = []

    async def sft(policy, train, validation):
        seen.extend(row["split"] for row in train + validation)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.zero_grad()
        (model(torch.ones(1, 1)) - 1).square().mean().backward()
        optimizer.step()
        return model

    async def rl(policy, train, validation):
        assert "test" not in seen
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
        optimizer.zero_grad()
        (-policy(torch.ones(1, 1)).mean()).backward()
        optimizer.step()
        return policy

    async def evaluate(artifact, test):
        assert [row["split"] for row in test] == ["test"]
        policy = torch.nn.Linear(1, 1, bias=False)
        policy.load_state_dict(
            torch.load(artifact / "pytorch_model.bin", weights_only=True)
        )
        return {"fixture_reward": float(policy(torch.ones(1, 1)).detach().mean())}

    def export(policy, path):
        path.mkdir()
        torch.save(policy.state_dict(), path / "pytorch_model.bin")
        return path

    def serve(path):
        reloaded = torch.nn.Linear(1, 1, bias=False)
        reloaded.load_state_dict(
            torch.load(path / "pytorch_model.bin", weights_only=True)
        )
        assert torch.equal(reloaded.weight, model.weight)
        return {"reloaded": True, "served": bool(reloaded(torch.ones(1, 1)).item() > 0)}

    result = await run_improvement_pipeline(
        records,
        output=tmp_path / "run",
        initial_policy=model,
        validation_after="2026-02-01T00:00:00Z",
        test_after="2026-03-01T00:00:00Z",
        supervised_train=sft,
        reinforcement_train=rl,
        evaluate=evaluate,
        export_policy=export,
        reload_and_serve=serve,
    )
    assert result["completed"] and not result["production_approved"]
    assert result["evaluation"]["fixture_reward"] > 0
    assert result["baseline_evaluation"]["fixture_reward"] == 0
    assert (tmp_path / "run/pipeline_status.json").is_file()


async def test_failed_training_retains_stage_journal_without_completed_report(tmp_path):
    records = [
        {
            "source_id": name,
            "timestamp": f"2026-0{month}-01T00:00:00Z",
            "messages": [{"role": "assistant", "content": name}],
            "reward": 1,
        }
        for month, name in [(1, "train"), (2, "validation"), (3, "test")]
    ]

    def export(policy, path: Path):
        path.mkdir()
        (path / "fixture.pt").write_bytes(b"control-flow fixture, not quality evidence")
        return path

    def fail_rl(*args):
        raise RuntimeError("training failed")

    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="training failed"):
        await run_improvement_pipeline(
            records,
            output=output,
            initial_policy=object(),
            validation_after="2026-02-01T00:00:00Z",
            test_after="2026-03-01T00:00:00Z",
            supervised_train=lambda policy, train, validation: policy,
            reinforcement_train=fail_rl,
            evaluate=lambda artifact, test: {},
            export_policy=export,
            reload_and_serve=lambda artifact: pytest.fail(
                "must not serve a failed run"
            ),
        )
    status = json.loads((output / "pipeline_status.json").read_text())
    assert status["stages"][-1]["status"] == "failed"
    assert status["stages"][-1]["error_type"] == "RuntimeError"
    assert not (output / "pipeline_report.json").exists()
