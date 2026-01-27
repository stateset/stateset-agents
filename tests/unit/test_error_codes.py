from stateset_agents.core.error_handling import (
    ErrorCategory,
    ErrorCode,
    ErrorHandler,
    GRPOException,
)


def test_grpo_exception_sets_default_error_code() -> None:
    err = GRPOException("training failed", category=ErrorCategory.TRAINING)
    assert err.error_code == ErrorCode.TRAINING_FAILED


def test_error_context_carries_error_code() -> None:
    err = GRPOException("bad input", category=ErrorCategory.VALIDATION)
    context = err.to_context(component="validator", operation="validate")
    assert context.error_code == ErrorCode.VALIDATION_FAILED


def test_error_handler_uses_system_code_for_generic_exception() -> None:
    handler = ErrorHandler()
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        context = handler.handle_error(exc, component="test", operation="run")

    assert context.error_code == ErrorCode.SYSTEM_FAILED
