#!/usr/bin/env python3
"""Validate retained, non-billable live-provider canary evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any


class ProviderEvidenceError(ValueError):
    """Raised when retained provider evidence is incomplete or unsafe."""


def _paths(inputs: Sequence[Path]) -> list[Path]:
    paths: list[Path] = []
    for candidate in inputs:
        if candidate.is_dir():
            paths.extend(sorted(candidate.glob("*.json")))
        elif candidate.is_file():
            paths.append(candidate)
        else:
            raise ProviderEvidenceError(f"input does not exist: {candidate}")
    if not paths:
        raise ProviderEvidenceError("no provider evidence found")
    return paths


def load_reports(inputs: Sequence[Path]) -> list[dict[str, Any]]:
    """Load canary reports and enforce their non-billable envelope."""
    reports: list[dict[str, Any]] = []
    for path in _paths(inputs):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ProviderEvidenceError(f"{path}: invalid JSON") from exc
        if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
            raise ProviderEvidenceError(f"{path}: schema_version=1 object required")
        if raw.get("billable_resources_created") != 0:
            raise ProviderEvidenceError(f"{path}: canary created billable resources")
        results = raw.get("results")
        if not isinstance(results, list) or len(results) != 1:
            raise ProviderEvidenceError(
                f"{path}: retained report must contain exactly one provider result"
            )
        result = results[0]
        if not isinstance(result, Mapping):
            raise ProviderEvidenceError(f"{path}: provider result must be an object")
        checked_at = result.get("checked_at")
        if not isinstance(checked_at, str):
            raise ProviderEvidenceError(f"{path}: checked_at must be ISO-8601")
        try:
            parsed = datetime.fromisoformat(checked_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ProviderEvidenceError(f"{path}: checked_at must be ISO-8601") from exc
        if parsed.tzinfo is None:
            raise ProviderEvidenceError(f"{path}: checked_at must include UTC offset")
        checks = result.get("checks")
        if not isinstance(checks, Mapping):
            raise ProviderEvidenceError(f"{path}: checks must be an object")
        if checks.get("billable_resources_created", 0) != 0:
            raise ProviderEvidenceError(
                f"{path}: provider check created billable resources"
            )
        reports.append(dict(raw))
    return reports


def validate_matrix(
    reports: Sequence[Mapping[str, Any]],
    required: Sequence[str] = ("river", "runpod", "fireworks"),
    *,
    allow_skipped: bool = False,
) -> dict[str, Any]:
    """Require one successful cleanup-verified report per provider."""
    by_provider: dict[str, Mapping[str, Any]] = {}
    for report in reports:
        result = report["results"][0]
        provider = str(result.get("provider", "")).strip().lower()
        if not provider:
            raise ProviderEvidenceError("provider must be a non-empty string")
        if provider in by_provider:
            raise ProviderEvidenceError(f"duplicate provider report: {provider}")
        by_provider[provider] = result
    missing = sorted(set(required) - set(by_provider))
    unexpected = sorted(set(by_provider) - set(required))
    if missing or unexpected:
        raise ProviderEvidenceError(
            f"provider matrix mismatch: missing={missing}, unexpected={unexpected}"
        )

    skipped: list[str] = []
    for provider in required:
        result = by_provider[provider]
        status = result.get("status")
        if status == "skipped" and allow_skipped:
            skipped.append(provider)
            continue
        if status != "passed":
            raise ProviderEvidenceError(
                f"{provider}: live canary status is {status!r}, not 'passed'"
            )
        if result.get("cleanup_verified") is not True:
            raise ProviderEvidenceError(f"{provider}: cleanup is not verified")
        if result.get("error") is not None:
            raise ProviderEvidenceError(f"{provider}: passed result contains an error")
    return {
        "schema_version": 1,
        "passed": not skipped,
        "providers": list(required),
        "skipped": skipped,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--required", nargs="+", default=["river", "runpod", "fireworks"]
    )
    parser.add_argument("--allow-skipped", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = validate_matrix(
            load_reports(args.inputs),
            args.required,
            allow_skipped=args.allow_skipped,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    except ProviderEvidenceError as exc:
        print(f"provider evidence rejected: {exc}", file=sys.stderr)
        return 2
    state = "complete" if report["passed"] else "incomplete"
    print(f"validated {len(report['providers'])} provider reports ({state})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
