"""Backward-compatible shim for historical enhanced API entrypoint."""

from __future__ import annotations

from .main import app, create_app, get_config


def main() -> None:
    """Run the current canonical API service."""
    config = get_config()
    import uvicorn

    uvicorn.run(
        "stateset_agents.api.main:app",
        host=config.host,
        port=config.port,
        reload=not config.is_production(),
        log_level=config.observability.log_level.lower(),
    )


__all__ = ["app", "create_app", "main"]


if __name__ == "__main__":
    main()
