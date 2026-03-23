"""
Site customization for test/runtime import stability.

Ensures this repository's root is at the front of sys.path so that
top-level packages like `core`, `rewards`, `utils`, etc. resolve to the
implementations shipped in this repo, even when another project on the
same machine defines similarly named packages earlier on sys.path.

Python automatically imports `sitecustomize` at interpreter startup
if it is importable on sys.path. When running tests from the repo root,
this file will be discovered and applied.
"""

import os
import sys
import inspect
from pathlib import Path

try:
    repo_root = Path(__file__).resolve().parent
    root_str = str(repo_root)
    if root_str in sys.path:
        # Move to front
        sys.path.remove(root_str)
        sys.path.insert(0, root_str)
    else:
        sys.path.insert(0, root_str)

    # Optional: set a flag to help with debugging if needed
    os.environ.setdefault("STATESET_AGENTS_SITE_PATH_PINNED", "1")
except Exception:
    # Do not fail interpreter startup; keep environment as-is
    pass

try:
    # pytest-asyncio 0.21 expects FixtureDef.unittest, which pytest 9 removed.
    # A class attribute restores the old default behaviour for compatible tests.
    from _pytest.fixtures import FixtureDef

    if not hasattr(FixtureDef, "unittest"):
        FixtureDef.unittest = False  # type: ignore[attr-defined]
except Exception:
    pass

try:
    # Starlette/FastAPI TestClient still passes ``app=...`` to httpx.Client in
    # some versions, while httpx>=0.28 removed that parameter.
    import httpx

    if (
        "app" not in inspect.signature(httpx.Client.__init__).parameters
        and not getattr(httpx.Client, "_stateset_agents_app_compat", False)
    ):
        _original_httpx_client_init = httpx.Client.__init__

        def _compat_httpx_client_init(self, *args, **kwargs):
            kwargs.pop("app", None)
            return _original_httpx_client_init(self, *args, **kwargs)

        httpx.Client.__init__ = _compat_httpx_client_init  # type: ignore[assignment]
        httpx.Client._stateset_agents_app_compat = True  # type: ignore[attr-defined]
except Exception:
    pass
