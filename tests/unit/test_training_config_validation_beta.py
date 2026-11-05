"""``beta`` without a reference model must not silently disable the KL term."""

from stateset_agents.training.config import TrainingConfig


def test_beta_without_reference_model_warns():
    warnings = TrainingConfig(beta=0.04, use_reference_model=False).validate()
    assert any("use_reference_model" in w for w in warnings)


def test_beta_with_reference_model_does_not_warn():
    warnings = TrainingConfig(beta=0.04, use_reference_model=True).validate()
    assert not any("use_reference_model" in w for w in warnings)
