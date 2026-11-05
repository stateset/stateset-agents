"""Preset configuration belongs to agents and must not leak into train kwargs."""

import importlib

import pytest

from stateset_agents.training import preset_training


@pytest.mark.parametrize(
    "helper",
    ["train_customer_service_agent", "train_tutoring_agent", "train_task_agent"],
)
async def test_agent_settings_are_consumed_and_revision_preserved(monkeypatch, helper):
    module = importlib.import_module("stateset_agents.training.train")
    calls = []

    async def train(**kwargs):
        calls.append(kwargs)
        assert "agent_config" not in kwargs
        return kwargs["agent"]

    monkeypatch.setattr(module, "train", train)
    settings = {"model_revision": "a" * 40}
    result = await getattr(preset_training, helper)(
        model_name="stub://test", agent_config=settings, num_episodes=2
    )
    assert result.config.model_revision == "a" * 40
    assert settings == {"model_revision": "a" * 40} and calls[0]["num_episodes"] == 2


async def test_tutoring_subject_is_scoped_to_this_run(monkeypatch):
    module = importlib.import_module("stateset_agents.training.train")
    calls = []

    async def train(**kwargs):
        calls.append(kwargs)
        return kwargs["agent"]

    monkeypatch.setattr(module, "train", train)
    await preset_training.train_tutoring_agent(subject="algebra")
    await preset_training.train_tutoring_agent(subject="chemistry")
    assert all(row["subject"] == "algebra" for row in calls[0]["environment"].scenarios)
    assert all(
        row["subject"] == "chemistry" for row in calls[1]["environment"].scenarios
    )
