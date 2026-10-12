"""Which models an API client may make the server load.

Loading a client-named model downloads arbitrary Hugging Face repositories
(outbound traffic, disk and memory exhaustion) or reads local paths. Real
weights therefore load only when an operator opts in with
``API_ALLOW_EXTERNAL_MODELS=true`` (in every environment), the name matches
``API_MODEL_ALLOWLIST`` when one is configured, and the caller has the
``trainer`` or ``admin`` role. Without the opt-in, development and staging
keep the stub backend and production refuses the request.

Operator-chosen models (``serve --checkpoint``, the server's demo agent) do not
go through this policy.
"""

from __future__ import annotations

from collections.abc import Iterable

from fastapi import HTTPException

from .config import get_config

STUB_PREFIX = "stub://"
MODEL_LOADING_ROLES = frozenset({"trainer", "admin"})


def _looks_like_local_path(model_name: str) -> bool:
    return (
        model_name.startswith(("/", "~", ".", "\\"))
        or ".." in model_name.split("/")
        or ":" in model_name.split("/", 1)[0]
    )


def _allow_listed(model_name: str, allowlist: Iterable[str]) -> bool:
    for entry in allowlist:
        if entry.endswith("*"):
            if model_name.startswith(entry[:-1]):
                return True
        elif model_name == entry:
            return True
    return False


def client_model_uses_stub(model_name: str, *, roles: Iterable[str] | None) -> bool:
    """Return whether a client-requested model runs on the stub backend.

    ``roles=None`` means the caller already enforced a model-loading role
    (e.g. the trainer-only training route). Raises ``HTTPException(403)`` when
    the request must be refused.
    """
    name = str(model_name).strip()
    if name.startswith(STUB_PREFIX):
        return True

    config = get_config()
    allowlist = list(config.model_allowlist)
    if not config.allow_external_models:
        if config.is_production():
            raise HTTPException(
                status_code=403,
                detail=(
                    "Loading client-requested models is disabled. Use a stub:// "
                    "model, or ask the operator to set "
                    "API_ALLOW_EXTERNAL_MODELS=true (optionally with "
                    "API_MODEL_ALLOWLIST)."
                ),
            )
        return True

    if allowlist:
        allowed = _allow_listed(name, allowlist)
    else:
        allowed = not _looks_like_local_path(name)
    if not allowed:
        raise HTTPException(
            status_code=403,
            detail=f"Model {name!r} is not permitted by the server's model policy",
        )
    if (
        roles is not None
        and config.security.require_auth
        and not MODEL_LOADING_ROLES.intersection(roles)
    ):
        raise HTTPException(
            status_code=403,
            detail="Loading real models requires the trainer or admin role",
        )
    return False


__all__ = ["MODEL_LOADING_ROLES", "client_model_uses_stub"]
