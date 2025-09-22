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

