# StateSet Agents v0.5.0

A maintainability-focused release that extracts the stub backend, hardens optional
dependency handling, and locks down mission-critical smoke tests.

## Highlights
- **Modular Stub Backend**
  - New `core.agent_backends` module encapsulates tokenizer/model scaffolding so
    `core.agent` stays readable and easier to extend.
  - Persona hinting hooks now live in a focused helper, keeping runtime tweaks controllable.
- **Resilient Optional Dependencies**
  - Performance optimizers and batch utilities now raise actionable guidance when
    Torch or psutil are missing instead of crashing at import time.
  - Monitoring health checks use the active asyncio loop and seamlessly support
    synchronous or asynchronous check functions.
- **Confidence via Tests**
  - Added unit coverage for CLI stub mode, backend factories, and health-check flows.
  - Test fixtures were modernised and now skip cleanly when Torch is unavailable.

## Upgrade Notes
```bash
pip install -U stateset-agents==0.5.0
```

No breaking APIs. If you previously relied on internal stub helpers inside
`core.agent`, import from `core.agent_backends` instead.

## Verification
- `pytest -q`
- `stateset-agents train --stub`
- `python -m build`

## Artifacts
- `dist/stateset_agents-0.5.0-py3-none-any.whl`
- `dist/stateset_agents-0.5.0.tar.gz`

---

## Previous Release: v0.4.0

See prior notes for the canonical import migration and end-to-end stub polish.
