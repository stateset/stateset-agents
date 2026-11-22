"""Standalone launcher for the StateSet 3D Weight Studio."""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

from .router import create_studio_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch the StateSet 3D Weight Explorer, Surgical Editor & Finetuning Studio."
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8765, help="Bind port (default: 8765)"
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace directory containing outputs/ and checkpoints/.",
    )
    parser.add_argument(
        "--open", action="store_true", help="Open browser automatically."
    )
    args = parser.parse_args(argv)

    import uvicorn

    url = f"http://{args.host}:{args.port}/studio"
    print(f"\n  StateSet 3D Weight Studio → {url}")
    print(f"  Workspace                → {args.workspace.resolve()}\n")
    if args.open:
        webbrowser.open(url)

    app = create_studio_app(workspace_root=args.workspace)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
