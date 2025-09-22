# StateSet Agents v0.3.4

A small but important packaging/import robustness update that makes it easier to
use the library without optional extras and in multi-repo environments.

## Highlights
- Import resilience for optional deps:
  - Wrapped `stateset_agents.core` proxy imports (`async_pool`, `advanced_monitoring`,
    `enhanced_state_management`) so missing extras (e.g., `aiohttp`, Prometheus, OTel)
    don’t break basic imports.
  - Top-level `stateset_agents` now guards heavy/optional imports (`performance_optimizer`,
    `async_pool`) to keep `import stateset_agents` safe by default.
- Safer module resolution:
  - The `stateset_agents.core` proxy now prefers the sibling top-level `core` package
    shipped with this distribution on `sys.path`, avoiding collisions in monorepos or
    notebooks where another `core` might exist earlier in `sys.path`.
- Namespace proxy for training:
  - Added `stateset_agents.training` proxy so callers can import via the public
    namespace while we keep a single source of truth in the top‑level `training` package.
- No API changes; this is a packaging/import hotfix.

## Changelog Extract (from CHANGELOG.md)

### Fixed
- Made `stateset_agents.core` proxy resilient to missing optional dependencies by wrapping
  `async_pool` import in a safe try/except. This prevents import-time failures when
  `aiohttp` is not installed and allows tests and consumers to import the package without
  optional extras.

### Changed
- Bumped package version to `0.3.4` in `pyproject.toml` and `stateset_agents/__init__.py`
  to keep versions in sync.

### Notes
- No API changes; this release improves import robustness and packaging hygiene only.

## Artifacts
- `dist/stateset_agents-0.3.4-py3-none-any.whl`
- `dist/stateset_agents-0.3.4.tar.gz`

## Install / Upgrade
```bash
pip install -U stateset-agents==0.3.4
```

## Verification
- Basic imports work without optional extras:
```python
import stateset_agents as sa
print(sa.__version__)
```
- Optional components remain available when extras are installed:
```bash
pip install "stateset-agents[api,trl]"
```

---
Thanks to everyone testing and providing feedback on packaging flows.
