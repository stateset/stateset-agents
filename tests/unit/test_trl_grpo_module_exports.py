from stateset_agents.training.trl_grpo_config import (
    TRLGRPOConfig as DirectTRLGRPOConfig,
)
from stateset_agents.training.trl_grpo_trainer import (
    TRLGRPOConfig,
    train_customer_service_with_trl,
    train_with_trl_grpo,
)


def test_trl_grpo_trainer_reexports_config_and_entrypoints() -> None:
    assert TRLGRPOConfig is DirectTRLGRPOConfig
    assert callable(train_with_trl_grpo)
    assert callable(train_customer_service_with_trl)
