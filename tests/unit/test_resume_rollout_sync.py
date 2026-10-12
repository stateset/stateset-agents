"""Restoring a checkpoint must push the restored weights to an attached
rollout engine; otherwise it serves the pre-resume weights until the next
optimizer step and reports itself as fresh."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from stateset_agents.training import multi_turn_trainer as multi
from stateset_agents.training import single_turn_trainer as single


@pytest.mark.parametrize("kind", ["single", "multi"])
@pytest.mark.parametrize("loaded", [True, False])
def test_successful_load_syncs_rollout_backend(monkeypatch, tmp_path, kind, loaded):
    module = single if kind == "single" else multi
    loader = (
        "load_checkpoint_artifacts"
        if kind == "single"
        else "load_multi_turn_checkpoint"
    )
    monkeypatch.setattr(module, loader, lambda *a, **k: loaded)
    trainer_type = (
        single.SingleTurnGRPOTrainer if kind == "single" else multi.MultiTurnGRPOTrainer
    )
    sync = MagicMock(return_value=True)
    trainer = trainer_type(
        SimpleNamespace(model=MagicMock(), sync_rollout_backend=sync),
        SimpleNamespace(reset=AsyncMock(), step=AsyncMock()),
        config=SimpleNamespace(output_dir=str(tmp_path)),
    )
    assert trainer.load_checkpoint(tmp_path) is loaded
    assert sync.call_count == (1 if loaded else 0)
