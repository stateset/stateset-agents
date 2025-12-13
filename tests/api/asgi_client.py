"""AsyncClient-backed synchronous test client for ASGI apps.

Starlette/FastAPI's ``TestClient`` relies on AnyIO's thread portal, which can
deadlock in some environments. This helper provides a small, synchronous API
(``get``, ``post``, etc.) while executing requests with ``httpx.AsyncClient``
and ``httpx.ASGITransport`` on a private event loop.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import httpx


class SyncASGIClient:
    """Synchronous wrapper around ``httpx.AsyncClient`` for ASGI apps."""

    def __init__(
        self,
        app: Any,
        *,
        base_url: str = "http://testserver",
        follow_redirects: bool = True,
    ) -> None:
        self._app = app
        self._base_url = base_url
        self._follow_redirects = follow_redirects
        self._loop = asyncio.new_event_loop()
        self._client: Optional[httpx.AsyncClient] = None

    def __enter__(self) -> "SyncASGIClient":
        asyncio.set_event_loop(self._loop)
        transport = httpx.ASGITransport(app=self._app)
        self._client = httpx.AsyncClient(
            transport=transport,
            base_url=self._base_url,
            follow_redirects=self._follow_redirects,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        if self._client is not None:
            self._loop.run_until_complete(self._client.aclose())
            self._client = None
        self._loop.close()

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Send a request and return the response."""
        if self._client is None:
            raise RuntimeError("Client not started; use as a context manager.")
        return self._loop.run_until_complete(
            self._client.request(method, url, **kwargs)
        )

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("PUT", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("DELETE", url, **kwargs)

    def options(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("OPTIONS", url, **kwargs)

