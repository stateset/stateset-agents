"""One exception hierarchy rooted at ``StateSetError``.

Public entry points raise typed errors that keep the builtin they replaced as
a base class (``ConfigurationError`` is a ``ValueError``) and print the same
message, so no ``except ValueError`` handler or error output changes.
"""

from __future__ import annotations

import pickle

import pytest

import stateset_agents
from stateset_agents.core import errors
from stateset_agents.core.agent_config import AgentConfig, ConfigValidationError
from stateset_agents.core.error_handling import (
    DataException,
    GRPOException,
    ModelException,
    NetworkException,
    ResourceException,
    TrainingException,
    ValidationException,
)


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
    # GRPO formatting and fields are unchanged ...
    assert str(exc).startswith(f"[{exc.error_code.value}] boom")
    # ... and the StateSetError interface now works for them too.
    payload = exc.to_dict()
    assert payload["error_code"] == exc.error_code.value
    assert payload["context"]["details"] == {"k": 1}


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
        ("EnvironmentError", "EnvironmentError"),
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
    assert alias is getattr(errors, canonical)


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
