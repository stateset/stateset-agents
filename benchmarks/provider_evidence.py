#!/usr/bin/env python3
"""Validate retained, non-billable live-provider canary evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class ProviderEvidenceError(ValueError):
    """Raised when retained provider evidence is incomplete or unsafe."""


REQUIRED_PROVIDERS = ("river", "runpod", "fireworks", "coreweave", "nebius")


def _nonnegative_integer(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _validate_provider_checks(provider: str, checks: Mapping[str, Any]) -> None:
    """Require the concrete read-only observations made by each adapter."""
    if checks.get("billable_resources_created") != 0:
        raise ProviderEvidenceError(f"{provider}: billable-resource check is missing")
    if provider == "river":
        if not checks.get("health") or not checks.get("capabilities"):
            raise ProviderEvidenceError(
                "river: health/capability observations are incomplete"
            )
    elif provider == "runpod":
        if (
            not _nonnegative_integer(checks.get("pods_observed"))
            or checks.get("local_cleanup_leases") != 0
            or checks.get("canary_leftovers") != []
            or checks.get("ephemeral_training_leftovers") != []
        ):
            raise ProviderEvidenceError("runpod: inventory/cleanup observations failed")
    elif provider == "fireworks":
        if (
            any(
                not _nonnegative_integer(checks.get(key))
                for key in (
                    "models_observed",
                    "jobs_observed",
                    "deployments_observed",
                )
            )
            or checks.get("canary_leftovers") != []
        ):
            raise ProviderEvidenceError("fireworks: inventory observations failed")
    elif provider == "coreweave":
        if (
            checks.get("provider") != "coreweave"
            or checks.get("authenticated") is not True
            or checks.get("can_create_jobs") is not True
        ):
            raise ProviderEvidenceError("coreweave: authorization observations failed")
    elif provider == "nebius":
        if (
            checks.get("provider") != "nebius"
            or checks.get("authenticated") is not True
            or not isinstance(checks.get("response_type"), str)
            or not checks["response_type"].strip()
        ):
            raise ProviderEvidenceError("nebius: authentication observation failed")


def _paths(inputs: Sequence[Path]) -> list[Path]:
    paths: list[Path] = []
    for candidate in inputs:
        if candidate.is_symlink() or any(
            parent.is_symlink() for parent in candidate.parents
        ):
            raise ProviderEvidenceError(
                f"provider evidence path is symlinked: {candidate}"
            )
        if candidate.is_dir():
            discovered = sorted(candidate.glob("*.json"))
            if any(path.is_symlink() for path in discovered):
                raise ProviderEvidenceError(
                    f"provider evidence directory contains a symlink: {candidate}"
                )
            paths.extend(discovered)
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
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderEvidenceError(f"{path}: invalid JSON") from exc
        if (
            not isinstance(raw, Mapping)
            or not isinstance(raw.get("schema_version"), int)
            or raw["schema_version"] not in {1, 2}
        ):
            raise ProviderEvidenceError(
                f"{path}: schema_version=1 or 2 object required"
            )
        if (
            raw["schema_version"] == 2
            and raw.get("kind") != "stateset-provider-canary-evidence"
        ):
            raise ProviderEvidenceError(f"{path}: schema-v2 kind is invalid")
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
        if (
            result.get("status") not in {"passed", "failed", "skipped"}
            or isinstance(result.get("duration_ms"), bool)
            or not isinstance(result.get("duration_ms"), int)
            or result["duration_ms"] < 0
            or not isinstance(result.get("cleanup_verified"), bool)
            or (
                result.get("error") is not None
                and not isinstance(result.get("error"), str)
            )
        ):
            raise ProviderEvidenceError(f"{path}: provider result envelope is invalid")
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
    required: Sequence[str] = REQUIRED_PROVIDERS,
    *,
    allow_skipped: bool = False,
    max_age_days: int = 30,
    now: datetime | None = None,
    minimum_schema_version: int = 1,
    expected_commit: str | None = None,
    expected_version: str | None = None,
) -> dict[str, Any]:
    """Require one successful cleanup-verified report per provider."""
    if max_age_days < 1:
        raise ProviderEvidenceError("max_age_days must be >= 1")
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        raise ProviderEvidenceError("provider evidence reference time must be aware")
    by_provider: dict[str, Mapping[str, Any]] = {}
    provider_schemas: dict[str, int] = {}
    for report in reports:
        schema_version = report.get("schema_version")
        if (
            not isinstance(schema_version, int)
            or schema_version < minimum_schema_version
        ):
            raise ProviderEvidenceError(
                f"provider evidence requires schema_version>={minimum_schema_version}"
            )
        if schema_version >= 2:
            commit = report.get("harness_commit")
            version = report.get("framework_version")
            if (
                report.get("kind") != "stateset-provider-canary-evidence"
                or not isinstance(commit, str)
                or len(commit) != 40
                or any(character not in "0123456789abcdef" for character in commit)
                or commit == "0" * 40
                or report.get("harness_clean") is not True
                or not isinstance(version, str)
                or not version.strip()
            ):
                raise ProviderEvidenceError(
                    "schema-v2 provider source identity is incomplete"
                )
            if expected_commit is not None and commit != expected_commit:
                raise ProviderEvidenceError("provider harness commit mismatch")
            if expected_version is not None and version != expected_version:
                raise ProviderEvidenceError("provider framework version mismatch")
        result = report["results"][0]
        provider = str(result.get("provider", "")).strip().lower()
        if not provider:
            raise ProviderEvidenceError("provider must be a non-empty string")
        if provider in by_provider:
            raise ProviderEvidenceError(f"duplicate provider report: {provider}")
        by_provider[provider] = result
        provider_schemas[provider] = schema_version
    missing = sorted(set(required) - set(by_provider))
    unexpected = sorted(set(by_provider) - set(required))
    if missing or unexpected:
        raise ProviderEvidenceError(
            f"provider matrix mismatch: missing={missing}, unexpected={unexpected}"
        )

    skipped: list[str] = []
    checked_times: list[datetime] = []
    for provider in required:
        result = by_provider[provider]
        checked_at = datetime.fromisoformat(
            str(result["checked_at"]).replace("Z", "+00:00")
        )
        age = reference - checked_at
        if age < -timedelta(minutes=5):
            raise ProviderEvidenceError(f"{provider}: checked_at is in the future")
        if age > timedelta(days=max_age_days):
            raise ProviderEvidenceError(
                f"{provider}: canary is older than {max_age_days} days"
            )
        checked_times.append(checked_at)
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
        if provider_schemas[provider] >= 2:
            checks = result.get("checks")
            if not isinstance(checks, Mapping):
                raise ProviderEvidenceError(f"{provider}: checks must be an object")
            _validate_provider_checks(provider, checks)
    return {
        "schema_version": 1,
        "passed": not skipped,
        "providers": list(required),
        "skipped": skipped,
        "max_age_days": max_age_days,
        "oldest_checked_at": min(checked_times).isoformat(),
        "newest_checked_at": max(checked_times).isoformat(),
        "minimum_schema_version": minimum_schema_version,
        "harness_commit": expected_commit,
        "framework_version": expected_version,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--required", nargs="+", default=list(REQUIRED_PROVIDERS))
    parser.add_argument("--allow-skipped", action="store_true")
    parser.add_argument("--max-age-days", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = validate_matrix(
            load_reports(args.inputs),
            args.required,
            allow_skipped=args.allow_skipped,
            max_age_days=args.max_age_days,
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
