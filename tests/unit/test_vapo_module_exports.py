from stateset_agents.training.vapo_config import VAPOConfig as DirectVAPOConfig
from stateset_agents.training.vapo_trainer import VAPOConfig, train_with_vapo


def test_vapo_trainer_reexports_config_and_entrypoint() -> None:
    assert VAPOConfig is DirectVAPOConfig
    assert callable(train_with_vapo)
