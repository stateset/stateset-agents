"""Shared request checks used by every chat route (one implementation)."""

import httpx
import pytest

from stateset_agents.api.messages_utils import (
    MessageLimitError,
    estimate_message_content_length,
    inference_error_text,
    validate_message_limits,
)
from stateset_agents.api.routers import messages as messages_router
from stateset_agents.api.routers import openai as openai_router
from stateset_agents.api.routers import v1 as v1_router


def _make_httpx_response(status_code: int, content: str) -> httpx.Response:
    # httpx.Response requires a request to safely access .json() in some contexts
    req = httpx.Request("GET", "http://testserver")
    return httpx.Response(
        status_code=status_code, content=content.encode("utf-8"), request=req
    )


def test_inference_error_text_variants():
    # {"error": {"message": "..."}}
    resp = _make_httpx_response(400, '{"error":{"message":"bad request"}}')
    exc = httpx.HTTPStatusError("boom", request=resp.request, response=resp)
    assert inference_error_text(exc) == "bad request"

    # {"error": "..."}
    resp2 = _make_httpx_response(400, '{"error":"rate limited"}')
    exc2 = httpx.HTTPStatusError("boom", request=resp2.request, response=resp2)
    assert inference_error_text(exc2) == "rate limited"

    # plain-text body
    resp3 = _make_httpx_response(500, "upstream broke")
    exc3 = httpx.HTTPError("oops")
    exc3.response = resp3  # type: ignore[attr-defined]
    assert inference_error_text(exc3) == "upstream broke"

    # no response attached
    assert inference_error_text(Exception("x")) == "Inference backend request failed"


def test_routers_share_one_implementation() -> None:
    """The three chat routers used to carry byte-identical private copies."""
    for router in (messages_router, openai_router, v1_router):
        assert router.inference_error_text is inference_error_text
        assert not hasattr(router, "_extract_inference_error_text")
        assert not hasattr(router, "_estimate_message_content_length")
    for router in (messages_router, openai_router, v1_router):
        assert router.validate_message_limits is validate_message_limits


def test_estimate_message_content_length():
    assert estimate_message_content_length(None) == 0
    assert estimate_message_content_length("abc") == 3
    # list / dict -> json.dumps path
    assert estimate_message_content_length(["a", "b"]) >= 5
    assert estimate_message_content_length({"k": "v"}) >= 7
    # non-serializable elements fall back to str()
    length = estimate_message_content_length([object()])
    assert isinstance(length, int) and length > 0
    length2 = estimate_message_content_length({"k": object()})
    assert isinstance(length2, int) and length2 > 0
    assert estimate_message_content_length(12345) == 5


class _Msg:
    def __init__(self, content: object) -> None:
        self.content = content


def test_validate_message_limits_count_then_length() -> None:
    validate_message_limits([_Msg("ok")], max_messages=1, max_length=10)

    with pytest.raises(MessageLimitError, match=r"Too many messages \(max 1\)") as e:
        validate_message_limits(
            [_Msg("a"), _Msg("b" * 99)], max_messages=1, max_length=10
        )
    assert e.value.kind == "count" and e.value.index is None

    with pytest.raises(MessageLimitError, match="too long at index 1") as e2:
        validate_message_limits(
            [{"content": "fine"}, {"content": "x" * 11}],
            max_messages=5,
            max_length=10,
        )
    assert e2.value.kind == "length" and e2.value.index == 1
    # Still a ValueError so routes that map ValueError to 422 keep working.
    assert isinstance(e2.value, ValueError)


def test_api_models_is_a_deprecated_duplicate_of_schemas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib
    import sys

    from stateset_agents.api import schemas

    monkeypatch.delitem(sys.modules, "stateset_agents.api.models", raising=False)
    with pytest.warns(DeprecationWarning, match=r"api\.models.*removed in v0\.71\.0"):
        models = importlib.import_module("stateset_agents.api.models")
    assert models.PaginatedResponse is schemas.PaginatedResponse
    assert models.APIResponse is schemas.APIResponse


def test_paginated_response_wire_format_is_unchanged() -> None:
    from datetime import datetime

    from stateset_agents.api.schemas import PaginatedResponse

    page = PaginatedResponse(total=3, timestamp=datetime(2026, 1, 2, 3, 4, 5))
    assert page.model_dump(mode="json") == {
        "request_id": None,
        "timestamp": "2026-01-02T03:04:05Z",
        "total": 3,
        "page": 1,
        "page_size": 20,
        "has_next": False,
        "has_prev": False,
    }
