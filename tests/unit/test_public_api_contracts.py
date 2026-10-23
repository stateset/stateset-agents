import pytest

import stateset_agents as sa
import stateset_agents.api as api

ALWAYS_PRESENT = {
    "GRPOException",
    "TrainingException",
    "ModelException",
    "DataException",
    "NetworkException",
    "ResourceException",
    "ValidationException",
    "ErrorCode",
    "ErrorHandler",
    "RetryConfig",
    "CircuitBreakerConfig",
    "retry_async",
    "circuit_breaker",
    "handle_error",
    "get_error_summary",
    "RewardFunction",
    "CompositeReward",
    "HelpfulnessReward",
    "SafetyReward",
    "CorrectnessReward",
    "ConcisenessReward",
    "EngagementReward",
    "TaskCompletionReward",
    "CustomerServiceReward",
    "TechnicalSupportReward",
    "SalesAssistantReward",
    "DomainSpecificReward",
    "SimilarityAwareReward",
    "create_domain_reward",
    "create_adaptive_reward",
    "Trajectory",
    "MultiTurnTrajectory",
    "ConversationTurn",
    "TypeValidator",
    "ConfigValidator",
    "TypeSafeSerializer",
    "create_typed_config",
    "ensure_type_safety",
    "ensure_async_type_safety",
    "ModelConfig",
    "TrainingConfig",
    "TypedTrainingConfig",
    "TypedConversationTurn",
    "TrajectoryData",
    "RewardMetrics",
    "DeviceType",
    "ModelSize",
    "TrainingStage",
}


def test_public_api_exports_exist() -> None:
    for name in sa.__all__:
        assert hasattr(sa, name), f"Missing public export: {name}"


REMOVED_SIM_TO_REAL_EXPORTS = (
    "SimToRealMetrics",
    "SimToRealEvaluator",
    "ConversationSimulator",
    "ConversationSimulatorConfig",
)


@pytest.mark.parametrize("name", REMOVED_SIM_TO_REAL_EXPORTS)
def test_sim_to_real_root_exports_removed_after_their_deadline(name: str) -> None:
    """Deprecated in v0.64.0 with removal promised for v0.66.0."""
    assert name not in sa.__all__
    with pytest.raises(AttributeError):
        getattr(sa, name)


def test_public_api_exports_are_unique() -> None:
    assert len(sa.__all__) == len(set(sa.__all__))


def test_public_api_required_symbols_not_none() -> None:
    for name in ALWAYS_PRESENT:
        value = getattr(sa, name, None)
        assert value is not None, f"{name} should be available"


def test_package_metadata_is_present() -> None:
    assert isinstance(sa.__version__, str)
    assert sa.__version__


def test_typed_training_config_alias_is_distinct_from_runtime_training_config() -> None:
    assert sa.TypedTrainingConfig is not sa.TrainingConfig


def test_api_version_matches_package_version() -> None:
    assert api.__version__ == sa.__version__
