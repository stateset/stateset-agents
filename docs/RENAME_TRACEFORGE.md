# Rename plan: StateSet Agents → TraceForge RL

## Decision

The library is being renamed to **TraceForge RL** (by StateSet). The current
name suggests a general agent runtime; the distinctive job is forging a
better policy from a deployed agent's own traces (`ingest → improve → train
→ serve`). The rename is phased so existing users are never broken without
warning.

The bare name `traceforge` is **not** available: PyPI already hosts an
unrelated [`traceforge`](https://pypi.org/project/traceforge/) agent test
harness that ships its own `traceforge` import and CLI. All public
identifiers therefore use the qualified form:

| Surface | New (TraceForge RL) | Legacy |
|---|---|---|
| Brand | TraceForge RL by StateSet | StateSet Agents |
| PyPI distribution | `traceforge-rl` (from Phase 2) | `stateset-agents` |
| Import package | `traceforge_rl` | `stateset_agents` |
| Console script | `traceforge-rl` | `stateset-agents` |
| Module execution | `python -m traceforge_rl` | `python -m stateset_agents` |

`traceforge-rl` returned HTTP 404 on PyPI when checked during planning; a
404 is a signal, not a reservation. Trademark, GitHub organisation/repo
name, and domain availability must be cleared by an owner before the
Phase 2 publication.

## Phase 1 — non-breaking alias (this change)

- New top-level `traceforge_rl` package that re-exports the same module
  objects as `stateset_agents` (identity is preserved, including for
  `traceforge_rl.training.objectives`), without eager heavy imports.
- New `traceforge-rl` console script and `python -m traceforge_rl`, both
  delegating to `stateset_agents.cli:run`.
- The wheel still publishes as `stateset-agents` and now also ships the
  `traceforge_rl` package. `stateset_agents` remains canonical, emits no
  deprecation warning yet, and every existing import/command keeps working.
- README/QUICKSTART announce the new name and link here. Maintained tests:
  `tests/unit/test_traceforge_rl_alias.py`.

Rollback is trivial: delete `traceforge_rl/` and the new script entry.

## Phase 2 — canonical flip (next minor release)

1. Move the canonical implementation from `stateset_agents/` to
   `traceforge_rl/`, rewriting internal absolute imports.
2. Rename the distribution to `traceforge-rl`; publish a `stateset-agents`
   compatibility release whose only dependency is `traceforge-rl` at the
   same version.
3. Replace `stateset_agents` with a thin shim that re-exports
   `traceforge_rl` and emits `DeprecationWarning` on import; keep the
   `stateset-agents` console script, warning on stderr.
4. Update the long tail: `mypy.ini` `files`, coverage `source`,
   `contracts/public_api_v1.json`, component-maturity refs, CI workflows,
   Sphinx docs, badges/URLs, release scripts, MCP server name, API title,
   examples, benchmark manifests, and the npm/Docker asset names. Rename
   the GitHub repository (GitHub keeps redirects for the old URL).
5. Announce in the changelog and migration guide before tagging.

Out of scope for a silent rename (each needs its own versioned migration):
frozen HTTP paths (`/v1/...`), environment-variable names, checkpoint or
lineage metadata keys, and the `stateset-rl-core` Rust crate name.

## Phase 3 — remove the shim (one release after Phase 2)

After exactly one release with the deprecation warning, delete the
`stateset_agents` shim package and the `stateset-agents` console script.
The `stateset-agents` PyPI project keeps a final release pointing at
`traceforge-rl` so old pins resolve with a clear message.

## Verification gates (every phase)

- Alias identity and laziness tests pass (`test_traceforge_rl_alias.py`).
- `traceforge-rl --help` and `stateset-agents --help` both load.
- A built wheel contains both packages; an isolated install imports both
  names and runs the CLI smoke.
- Full `pytest -q`, `ruff check .`, `python scripts/check_types.py --all`,
  `make api-compatibility`, and the docs-contract tests pass.
