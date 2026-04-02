# Repo Maintenance Policy

This document records the maintenance rules that keep the repository healthy as
the project grows.

## Canonical imports

Use `stateset_agents.*` imports for all new code.

The top-level shim modules (`api/`, `core/`, `training/`, `rewards/`,
`environments/`) exist only for backwards compatibility and in-repo tests. They
must not receive new features.

## Deprecation plan

The current target is:

1. `0.12.x`: stop adding new in-repo imports that depend on shim modules.
2. `0.13.0`: prune remaining internal shim usage and emit stronger warnings
   where the shim paths are still imported.
3. `0.14.0`: remove shim modules once downstream usage is audited and release
   notes have carried the deprecation window long enough.

## Generated artifacts

Generated or local-only artifacts must not be committed. This includes:

- build outputs such as `dist/`, `build/`, and `*.egg-info/`
- test and coverage outputs such as `htmlcov/`, `.coverage`, and `coverage.xml`
- runtime artifacts such as `outputs/` and `checkpoints/`
- local tool state such as `.codex/` and `.autoresearch/`
- backup files such as `*.backup`, `*.bak`, `*.orig`, and `*.tmp`

CI enforces this with `python scripts/check_repo_hygiene.py`.

## Review bar

Maintenance-oriented changes should prefer:

- fewer parallel code paths
- fewer legacy aliases
- fewer generated files in the repo root
- one canonical example per supported workflow
- explicit tests for exported public helpers and CLI starter paths
