from stateset_agents.training.gspo_config import GSPOConfig as DirectGSPOConfig
from stateset_agents.training.gspo_generation import (
    GSPOTrajectoryGenerator as DirectGSPOTrajectoryGenerator,
)
from stateset_agents.training.gspo_trainer import (
    GSPOConfig,
    GSPOTrajectoryGenerator,
    train_customer_service_with_gspo,
    train_with_gspo,
)


def test_gspo_trainer_reexports_config_and_entrypoints() -> None:
    assert GSPOConfig is DirectGSPOConfig
    assert GSPOTrajectoryGenerator is DirectGSPOTrajectoryGenerator
    assert callable(train_with_gspo)
    assert callable(train_customer_service_with_gspo)
