#!/usr/bin/env python3
"""Install and verify the pinned upstream agent-quality suite checkouts.

Each suite in ``agent_quality_harnesses.json`` carries an ``upstream`` block:
the public repository URL, the exact commit the evidence is pinned to, the
install argv lists (run without a shell, inside the checkout) and the
executables the official pipeline expects afterwards. ``install`` clones each
repository into ``--root`` at that commit and runs the install steps;
``verify`` re-checks HEAD, cleanliness and executables and, with
``--manifest``, that every manifest suite revision equals its pin. Both print
the ``STATESET_AQ_UPSTREAM_ROOT`` export the harness config resolves.

Installation downloads public packages but spends no money.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class UpstreamError(RuntimeError):
    """Raised when an upstream checkout cannot be pinned or verified."""


def load_pins(harness_config: Path) -> dict[str, dict[str, Any]]:
    """Return ``{suite: upstream}`` for every suite in the harness config."""
    raw = json.loads(harness_config.read_text(encoding="utf-8"))
    suites = raw.get("suites") if isinstance(raw, Mapping) else None
    if not isinstance(suites, Mapping) or not suites:
        raise UpstreamError(f"{harness_config} defines no suites")
    pins: dict[str, dict[str, Any]] = {}
    for suite, config in suites.items():
        upstream = config.get("upstream") if isinstance(config, Mapping) else None
        if not isinstance(upstream, Mapping):
            raise UpstreamError(f"{suite} has no upstream pin")
        for key in ("url", "revision", "checkout_dir"):
            if not isinstance(upstream.get(key), str) or not upstream[key]:
                raise UpstreamError(f"{suite}.upstream.{key} must be non-empty")
        revision = upstream["revision"]
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise UpstreamError(f"{suite}.upstream.revision must be a 40-hex commit")
        if Path(upstream["checkout_dir"]).name != upstream["checkout_dir"]:
            raise UpstreamError(f"{suite}.upstream.checkout_dir must be a plain name")
        pins[str(suite)] = dict(upstream)
    return pins


def _git(checkout: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=checkout, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise UpstreamError(
            f"git {' '.join(args)} failed in {checkout}: {completed.stderr.strip()}"
        )
    return completed.stdout


def verify_checkout(suite: str, pin: Mapping[str, Any], root: Path) -> Path:
    """Require the checkout at the pinned commit, clean, with its executables."""
    checkout = root / str(pin["checkout_dir"])
    if not checkout.is_dir():
        raise UpstreamError(f"{suite}: checkout missing at {checkout}")
    head = _git(checkout, "rev-parse", "HEAD").strip()
    if head != pin["revision"]:
        raise UpstreamError(
            f"{suite}: checkout HEAD {head} is not the pinned {pin['revision']}"
        )
    if _git(checkout, "status", "--porcelain").strip():
        raise UpstreamError(f"{suite}: checkout {checkout} has local modifications")
    for executable in pin.get("executables", []):
        if not (checkout / executable).is_file():
            raise UpstreamError(f"{suite}: {executable} is not installed")
    return checkout


def install_checkout(suite: str, pin: Mapping[str, Any], root: Path) -> Path:
    """Clone at the pinned commit (idempotently) and run its install steps."""
    root.mkdir(parents=True, exist_ok=True)
    checkout = root / str(pin["checkout_dir"])
    if not checkout.exists():
        completed = subprocess.run(
            ["git", "clone", "--quiet", str(pin["url"]), str(checkout)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise UpstreamError(f"{suite}: clone failed: {completed.stderr.strip()}")
    _git(checkout, "checkout", "--quiet", "--detach", str(pin["revision"]))
    if _git(checkout, "status", "--porcelain").strip():
        raise UpstreamError(f"{suite}: checkout {checkout} has local modifications")
    for step in pin.get("install", []):
        if not isinstance(step, list) or not all(isinstance(p, str) for p in step):
            raise UpstreamError(f"{suite}: install steps must be argv lists")
        step_result = subprocess.run(step, cwd=checkout, check=False)
        if step_result.returncode != 0:
            raise UpstreamError(
                f"{suite}: install step {step} exited {step_result.returncode}"
            )
    return verify_checkout(suite, pin, root)


def verify_manifest(manifest: Path, pins: Mapping[str, Mapping[str, Any]]) -> None:
    """Require every manifest suite revision to equal its upstream pin."""
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    for suite in raw.get("suites", []):
        pin = pins.get(suite.get("name"))
        if pin is None:
            raise UpstreamError(f"manifest suite {suite.get('name')} has no pin")
        if suite.get("revision") != pin["revision"]:
            raise UpstreamError(
                f"manifest {suite['name']} revision {suite.get('revision')} "
                f"!= pinned {pin['revision']}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("install", "verify"))
    parser.add_argument("--harness-config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--suite", action="append", default=[])
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        pins = load_pins(args.harness_config)
        if args.manifest is not None:
            verify_manifest(args.manifest, pins)
        selected = args.suite or list(pins)
        for suite in selected:
            if suite not in pins:
                raise UpstreamError(f"unknown suite {suite}")
            handler = install_checkout if args.action == "install" else verify_checkout
            checkout = handler(suite, pins[suite], root)
            print(f"{suite}: {checkout} @ {pins[suite]['revision']}")
    except UpstreamError as exc:
        print(f"agent-quality upstream rejected: {exc}", file=sys.stderr)
        return 2
    print(f"export STATESET_AQ_UPSTREAM_ROOT={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
