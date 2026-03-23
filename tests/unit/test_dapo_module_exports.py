from stateset_agents.training.dapo_config import DAPOConfig as DirectDAPOConfig
from stateset_agents.training.dapo_trainer import (
    DAPOConfig,
    train_reasoning_with_dapo,
    train_with_dapo,
)


def test_dapo_trainer_reexports_config_and_entrypoints() -> None:
    assert DAPOConfig is DirectDAPOConfig
    assert callable(train_with_dapo)
    assert callable(train_reasoning_with_dapo)
