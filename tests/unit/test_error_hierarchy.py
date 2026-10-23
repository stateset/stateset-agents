"""One exception hierarchy rooted at ``StateSetError``.

Public entry points raise typed errors that keep the builtin they replaced as
a base class (``ConfigurationError`` is a ``ValueError``) and print the same
message, so no ``except ValueError`` handler or error output changes.
"""

from __future__ import annotations

import pickle
import warnings

import pytest

import stateset_agents
from stateset_agents.core import error_handling, errors
from stateset_agents.core.agent_config import AgentConfig, ConfigValidationError
from stateset_agents.core.error_handling import (
    GRPOErrorCode,
    GRPOException,
    ResourceException,
    ValidationException,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    TrainingException = error_handling.TrainingException
    ModelException = error_handling.ModelException
    DataException = error_handling.DataException
    NetworkException = error_handling.NetworkException


def test_root_exports_the_hierarchy_root_and_configuration_error() -> None:
    assert stateset_agents.StateSetError is errors.StateSetError
    assert stateset_agents.ConfigurationError is errors.ConfigurationError
    assert {"StateSetError", "ConfigurationError"} <= set(stateset_agents.__all__)


def test_configuration_error_is_a_value_error_with_a_plain_message() -> None:
    exc = errors.ConfigurationError("learning_rate must be finite", field="lr")
    assert isinstance(exc, errors.StateSetError)
    assert isinstance(exc, ValueError)
    assert str(exc) == "learning_rate must be finite"
    assert exc.code is errors.ErrorCode.CFG_INVALID
    assert exc.context.details == {"field": "lr"}
    restored = pickle.loads(pickle.dumps(exc))
    assert type(restored) is errors.ConfigurationError
    assert str(restored) == str(exc)


def test_other_category_errors_keep_their_structured_str() -> None:
    exc = errors.ModelError("load failed", model_name="gpt2")
    assert str(exc).startswith("[MDL_001] load failed")
    assert not isinstance(exc, ValueError)


@pytest.mark.parametrize(
    ("grpo_type", "category"),
    [
        (TrainingException, errors.TrainingError),
        (ModelException, errors.ModelError),
        (DataException, errors.DataError),
        (NetworkException, errors.NetworkError),
        (ResourceException, errors.ResourceError),
        (ValidationException, errors.ValidationError),
    ],
)
def test_grpo_exceptions_are_category_errors(
    grpo_type: type[GRPOException], category: type[errors.StateSetError]
) -> None:
    exc = grpo_type("boom", details={"k": 1})
    assert isinstance(exc, GRPOException)
    assert isinstance(exc, category)
    assert isinstance(exc, errors.StateSetError)
    # GRPO formatting and the legacy code are unchanged ...
    assert isinstance(exc.error_code, GRPOErrorCode)
    assert str(exc).startswith(f"[{exc.error_code.value}] boom")
    # ... while ``code`` is the canonical enum every StateSetError carries.
    assert isinstance(exc.code, errors.ErrorCode)
    payload = exc.to_dict()
    assert payload["error_code"] == exc.code.value
    assert payload["grpo_error_code"] == exc.error_code.value
    assert payload["context"]["details"] == {"k": 1}


def test_grpo_code_is_the_canonical_enum_of_its_category() -> None:
    """``GRPOException.code`` used to hold the GRPO_E### enum, a different type
    from every other ``StateSetError.code``."""
    assert ValidationException("x").code is errors.ErrorCode.VAL_INVALID_INPUT
    assert ResourceException("x").code is errors.ErrorCode.RES_EXHAUSTED
    assert GRPOException("x").code is errors.ErrorCode.UNKNOWN
    assert ValidationException("x").error_code is GRPOErrorCode.VALIDATION_FAILED


def test_validation_exception_is_not_caught_by_value_error_handlers() -> None:
    """Bridging must not widen which builtin handlers catch GRPO errors."""
    assert not issubclass(ValidationException, ValueError)


def test_agent_config_validation_raises_configuration_error() -> None:
    with pytest.raises(errors.ConfigurationError) as caught:
        AgentConfig(model_name="")
    assert isinstance(caught.value, ConfigValidationError)
    assert isinstance(caught.value, ValueError)
    assert str(caught.value).startswith("Invalid model_name=''")
    assert caught.value.field == "model_name"


def test_training_config_validation_raises_configuration_error() -> None:
    from stateset_agents.training.config import TrainingConfig

    with pytest.raises(errors.ConfigurationError, match="nonfinite_grad_policy"):
        TrainingConfig(nonfinite_grad_policy="sometimes")
    config = TrainingConfig()
    config.top_p = 0
    with pytest.raises(ValueError, match="^top_p must be positive$"):
        config.validate_for_training()


@pytest.mark.asyncio
async def test_train_rejects_bad_arguments_with_configuration_error() -> None:
    from stateset_agents.training.train import train

    with pytest.raises(errors.ConfigurationError, match="training_mode must be"):
        await train(agent=object(), environment=object(), training_mode="bogus")


def test_api_errors_join_the_hierarchy_without_changing_str() -> None:
    from stateset_agents.api.errors import APIError, ErrorCode, NotFoundError

    exc = APIError(ErrorCode.BAD_REQUEST, "bad payload", 400)
    assert isinstance(exc, errors.StateSetError)
    assert str(exc) == "bad payload"
    assert exc.code is ErrorCode.BAD_REQUEST
    assert exc.to_dict()["context"]["details"] == {"status_code": 400}
    assert issubclass(NotFoundError, errors.StateSetError)


@pytest.mark.parametrize(
    ("old", "canonical"),
    [
        ("FrameworkError", "StateSetError"),
        ("AgentError", "StateSetError"),
        ("EnvironmentError", "EnvironmentOperationError"),
        ("TrainingError", "TrainingError"),
        ("ConfigurationError", "ConfigurationError"),
    ],
)
def test_core_types_error_family_is_a_deprecated_alias(
    old: str, canonical: str
) -> None:
    from stateset_agents.core import types as legacy_types

    with pytest.warns(DeprecationWarning, match=r"removed in v0\.71\.0"):
        alias = getattr(legacy_types, old)
    assert alias is vars(errors)[canonical]


def test_no_second_error_root_remains_defined() -> None:
    """``FrameworkError`` and friends are no longer separate classes."""
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2] / "stateset_agents" / "core" / "types.py"
    ).read_text(encoding="utf-8")
    defined = {
        node.name for node in ast.parse(source).body if isinstance(node, ast.ClassDef)
    }
    assert not defined & {
        "FrameworkError",
        "AgentError",
        "EnvironmentError",
        "TrainingError",
        "ConfigurationError",
    }


# --- the errors the framework actually raises are StateSetErrors -----------


def _domain_errors() -> list[tuple[type[BaseException], type[BaseException], str]]:
    from stateset_agents.core.reward_context import RewardUnavailableError
    from stateset_agents.training.async_rollouts import (
        AsyncRolloutError,
        AsyncRolloutTimeout,
    )
    from stateset_agents.training.backends import BackendError, BackendExecutionError
    from stateset_agents.training.diagnostics import TrainingSignalError
    from stateset_agents.training.rollout_contract import RolloutContractError

    return [
        (RewardUnavailableError, RuntimeError, "RWD_UNAVAILABLE"),
        (RolloutContractError, ValueError, "TRN_ROLLOUT_CONTRACT"),
        (TrainingSignalError, RuntimeError, "TRN_NO_SIGNAL"),
        (AsyncRolloutError, RuntimeError, "TRN_ASYNC_ROLLOUT"),
        (AsyncRolloutTimeout, RuntimeError, "TRN_ASYNC_ROLLOUT"),
        (BackendError, RuntimeError, "TRN_BACKEND_FAILED"),
        (BackendExecutionError, RuntimeError, "TRN_BACKEND_FAILED"),
    ]


@pytest.mark.parametrize(
    ("error_type", "builtin", "code"),
    _domain_errors(),
    ids=lambda value: getattr(value, "__name__", value),
)
def test_domain_errors_join_the_hierarchy_and_keep_their_builtin(
    error_type: type[BaseException], builtin: type[BaseException], code: str
) -> None:
    exc = error_type("rollout 7 is stale")
    assert isinstance(exc, errors.StateSetError)
    assert isinstance(exc, builtin)
    # Same output as the builtin they replaced: just the message.
    assert str(exc) == "rollout 7 is stale"
    assert exc.args == ("rollout 7 is stale",)
    assert exc.code is errors.ErrorCode[code]
    restored = pickle.loads(pickle.dumps(exc))
    assert type(restored) is error_type and str(restored) == str(exc)


def test_reward_and_training_errors_share_their_category() -> None:
    from stateset_agents.core.reward_context import RewardUnavailableError
    from stateset_agents.rewards.llm_judge import JudgeUnavailableError
    from stateset_agents.training.diagnostics import TrainingSignalError

    assert issubclass(RewardUnavailableError, errors.RewardError)
    assert issubclass(JudgeUnavailableError, errors.RewardError)
    assert issubclass(TrainingSignalError, errors.TrainingError)
    with pytest.raises(RuntimeError, match="^judge down$"):
        raise RewardUnavailableError("judge down")


def test_an_explicit_code_still_wins_over_the_class_default() -> None:
    exc = errors.TrainingError("oom", code=errors.ErrorCode.TRN_OOM)
    assert exc.code is errors.ErrorCode.TRN_OOM
    assert errors.StateSetError("x").code is errors.ErrorCode.UNKNOWN


def test_api_configuration_error_is_the_canonical_one() -> None:
    from stateset_agents.api import config as api_config

    assert api_config.ConfigurationError is errors.ConfigurationError
    with pytest.raises(errors.StateSetError):
        raise api_config.ConfigurationError("API_JWT_SECRET is required")


def test_resilience_errors_join_the_hierarchy() -> None:
    from stateset_agents.api import resilience

    for name in ("CircuitOpenError", "OperationTimeoutError", "BulkheadFullError"):
        error_type = getattr(resilience, name)
        assert issubclass(error_type, errors.StateSetError)
        assert str(error_type("circuit svc is OPEN")) == "circuit svc is OPEN"
    # The renamed timeout error no longer shadows the builtin.
    assert not issubclass(resilience.OperationTimeoutError, TimeoutError)


# --- duplicates are deprecated aliases of one object ------------------------


@pytest.mark.parametrize(
    ("module_name", "old", "new", "removal"),
    [
        (
            "stateset_agents.core.errors",
            "EnvironmentError",
            "EnvironmentOperationError",
            "0.71.0",
        ),
        ("stateset_agents.core.error_handling", "ErrorCode", "GRPOErrorCode", "0.72.0"),
        (
            "stateset_agents.core.error_handling",
            "ErrorContext",
            "ErrorRecord",
            "0.71.0",
        ),
        (
            "stateset_agents.api.errors",
            "ValidationError",
            "UnprocessableEntityError",
            "0.71.0",
        ),
        (
            "stateset_agents.api.errors",
            "TrainingError",
            "TrainingFailedError",
            "0.71.0",
        ),
        (
            "stateset_agents.api.resilience",
            "TimeoutError",
            "OperationTimeoutError",
            "0.71.0",
        ),
    ],
)
def test_renamed_duplicates_are_dated_aliases(
    module_name: str, old: str, new: str, removal: str
) -> None:
    import importlib

    module = importlib.import_module(module_name)
    assert old not in vars(module), f"{module_name}.{old} is still a real class"
    with pytest.warns(DeprecationWarning, match=rf"removed in v{removal}"):
        alias = getattr(module, old)
    assert alias is getattr(module, new)


@pytest.mark.parametrize(
    ("old", "category"),
    [
        ("TrainingException", errors.TrainingError),
        ("ModelException", errors.ModelError),
        ("DataException", errors.DataError),
        ("NetworkException", errors.NetworkError),
    ],
)
def test_unused_grpo_subclasses_are_deprecated(
    old: str, category: type[errors.StateSetError]
) -> None:
    with pytest.warns(DeprecationWarning, match=r"removed in v0\.72\.0"):
        legacy = getattr(error_handling, old)
    assert legacy.__name__ == old
    assert issubclass(legacy, category)
    restored = pickle.loads(pickle.dumps(legacy("boom")))
    assert type(restored) is legacy


def test_root_error_code_warns_once_and_names_the_canonical_enum() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy = stateset_agents.ErrorCode
    assert legacy is GRPOErrorCode
    messages = [str(w.message) for w in caught if w.category is DeprecationWarning]
    assert len(messages) == 1, messages
    assert "stateset_agents.core.errors.ErrorCode" in messages[0]
    assert "v0.72.0" in messages[0]


def test_framework_raisers_no_longer_use_the_legacy_code_enum() -> None:
    """Nothing in the package imports the deprecated ``ErrorCode`` name."""
    import ast
    from pathlib import Path

    package = Path(stateset_agents.__file__).parent
    offenders = []
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and (node.module or "").endswith("error_handling")
                and any(
                    alias.name in {"ErrorCode", "ErrorContext"} for alias in node.names
                )
            ):
                offenders.append(f"{path.relative_to(package)}:{node.lineno}")
    assert offenders == []


@pytest.mark.parametrize(
    "factory",
    [
        lambda: error_handling.RetryConfig(),
        lambda: error_handling.CircuitBreakerConfig(),
        lambda: error_handling.CircuitBreaker(),
    ],
    ids=["RetryConfig", "CircuitBreakerConfig", "CircuitBreaker"],
)
def test_second_retry_and_breaker_implementation_is_deprecated(factory) -> None:
    with pytest.warns(
        DeprecationWarning, match=r"api\.resilience.*|removed in v0\.72\.0"
    ):
        factory()
