# Release v0.3.3 — StateSet Agents

Date: 2025-09-04

## Highlights
- Training modules import cleanly from a PyPI install: switched to absolute imports (`core.*`, `utils.*`, `rewards.*`).
- Distribution now includes implementation packages used by `stateset_agents`:
  - `core`, `rewards`, `utils`, `api`, `environments`, and a lightweight `grpo_agent_framework` proxy.
- CI publish workflow hardened:
  - Idempotent uploads (`twine upload --skip-existing`).
  - Resilient tag creation.
  - Docker and docs jobs run on GitHub release events as well as manual dispatch.

## Fixes
- Resolved import errors when using `training` modules from an installed package.
- Ensured required top-level modules are packaged and importable.

## Install
- `pip install stateset-agents==0.3.3`

## Links
- PyPI: https://pypi.org/project/stateset-agents/0.3.3/
- Docs: https://stateset-agents.readthedocs.io/
- Issues: https://github.com/stateset/stateset-agents/issues

