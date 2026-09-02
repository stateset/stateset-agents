import httpx
import pytest

from stateset_agents.api.routers import messages as messages_router
from stateset_agents.api.routers import openai as openai_router


def _make_httpx_response(status_code: int, content: str) -> httpx.Response:
    # httpx.Response requires a request to safely access .json() in some contexts
    req = httpx.Request("GET", "http://testserver")
    return httpx.Response(status_code=status_code, content=content.encode("utf-8"), request=req)


def test_extract_inference_error_text_messages_variants():
    # {"error": {"message": "..."}}
    resp = _make_httpx_response(400, '{"error":{"message":"bad request"}}')
    exc = httpx.HTTPStatusError("boom", request=resp.request, response=resp)
    assert messages_router._extract_inference_error_text(exc) == "bad request"

    # {"error": "..."}
    resp2 = _make_httpx_response(400, '{"error":"rate limited"}')
    exc2 = httpx.HTTPStatusError("boom", request=resp2.request, response=resp2)
    assert messages_router._extract_inference_error_text(exc2) == "rate limited"

    # plain-text body
    resp3 = _make_httpx_response(500, "upstream broke")
    exc3 = httpx.HTTPError("oops")
    setattr(exc3, "response", resp3)
    assert messages_router._extract_inference_error_text(exc3) == "upstream broke"

    # no response attached
    assert messages_router._extract_inference_error_text(Exception("x")) == "Inference backend request failed"


def test_extract_inference_error_text_openai_variants():
    # {"error": {"message": "..."}}
    resp = _make_httpx_response(400, '{"error":{"message":"invalid model"}}')
    exc = httpx.HTTPStatusError("boom", request=resp.request, response=resp)
    assert openai_router._extract_inference_error_text(exc) == "invalid model"

    # {"error": "..."}
    resp2 = _make_httpx_response(400, '{"error":"blocked content"}')
    exc2 = httpx.HTTPStatusError("boom", request=resp2.request, response=resp2)
    assert openai_router._extract_inference_error_text(exc2) == "blocked content"

    # plain-text body
    resp3 = _make_httpx_response(502, "gateway failed")
    exc3 = httpx.HTTPError("oops")
    setattr(exc3, "response", resp3)
    assert openai_router._extract_inference_error_text(exc3) == "gateway failed"

    # no response attached
    assert openai_router._extract_inference_error_text(Exception("x")) == "Inference backend request failed"


def test_estimate_message_content_length_messages_router():
    # None -> 0
    assert messages_router._estimate_message_content_length(None) == 0
    # str
    assert messages_router._estimate_message_content_length("abc") == 3
    # list -> json.dumps path
    assert messages_router._estimate_message_content_length(["a", "b"]) >= 5
    # list containing non-serializable element -> fallback to str()
    length = messages_router._estimate_message_content_length([object()])
    assert isinstance(length, int) and length > 0
    # dict -> json.dumps path
    assert messages_router._estimate_message_content_length({"k": "v"}) >= 7
    # dict containing non-serializable element -> fallback to str()
    length2 = messages_router._estimate_message_content_length({"k": object()})
    assert isinstance(length2, int) and length2 > 0


def test_estimate_message_content_length_openai_router():
    # Mirror the same coverage on the OpenAI router helpers
    assert openai_router._estimate_message_content_length(None) == 0
    assert openai_router._estimate_message_content_length("xyz") == 3
    assert openai_router._estimate_message_content_length(["a", "b", "c"]) >= 7
    length = openai_router._estimate_message_content_length([object()])
    assert isinstance(length, int) and length > 0
    assert openai_router._estimate_message_content_length({"k": "v"}) >= 7
    length2 = openai_router._estimate_message_content_length({"x": object()})
    assert isinstance(length2, int) and length2 > 0

