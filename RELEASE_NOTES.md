# StateSet Agents v0.4.0

A developer-experience upgrade that makes the library trivial to demo, unify, and
ship in CI/CD pipelines.

## Highlights
- **Offline-friendly stub backend**
  - `AgentConfig(use_stub_model=True, stub_responses=...)` now powers the entire
    stack, including `ComputationalGRPOEngine`.
  - `stateset-agents train --stub` provides a one-command smoke test.
  - New `examples/backend_switch_demo.py` shows how to flip between stub and HF
    backends programmatically.
- **Canonical import pathway**
  - Every module, example, and test imports from `stateset_agents.core.*`.
  - Legacy `import core.*` remains as a shim but now emits a `DeprecationWarning`
    so downstream projects can migrate safely.
- **Docs, CLI, and tests refreshed**
  - README quick starts, release notes, and CLI help all spotlight the stub flow.
  - Added regression coverage to ensure the stub backend and string prompts keep
    working through training utilities.

## Upgrade Notes
```bash
pip install -U stateset-agents==0.4.0
```

If you have custom extensions that relied on `import core.*`, update them to use
`stateset_agents.core.*`. The legacy namespace will be removed in a future major
release.

## Verification
- `pytest -q`
- `stateset-agents train --stub`
- (Optional) `python examples/backend_switch_demo.py --stub`

## Artifacts
- `dist/stateset_agents-0.4.0-py3-none-any.whl`
- `dist/stateset_agents-0.4.0.tar.gz`

---

## Previous Release: v0.3.4

See prior notes for the packaging/import hotfix that introduced the
`stateset_agents.training` proxy and improved optional-dependency handling.
