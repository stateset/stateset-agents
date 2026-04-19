# Repository Guidelines

## Project Structure & Module Organization
- `stateset_agents/`: Canonical, packaged source (this is what ships to PyPI).
  - `stateset_agents/core/`: Core primitives (agents, environments, rewards, trajectories, data processing).
  - `stateset_agents/training/`: Trainers, configs, and training loops.
  - `stateset_agents/rewards/`: Advanced/domain reward functions.
  - `stateset_agents/utils/`: Logging, monitoring, observability, W&B integration.
  - `stateset_agents/api/`: FastAPI gateway (includes `/v1/messages` and OpenAI-compat endpoints).
- `api/`, `core/`, `training/`, `rewards/`, `environments/`: Deprecated shim modules used in-repo only for backwards compatibility/tests. Prefer `stateset_agents.*`.
- `examples/`: Runnable demos and quick starts.
- `tests/` + `test_integration.py`: Unit and integration tests.
- `scripts/`, `deployment/`, `benchmarks/`: Utilities for training, deployment, and performance.

## Build, Test, and Development Commands
- Setup (Python 3.10+):
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -e ".[dev,api]"`
- Run tests: `pytest -q` (use `-k` to filter; async tests use `pytest-asyncio`).
- Format & lint: `ruff check . && black . && isort .`
- Type check: `python scripts/check_types.py --all`
- Repo hygiene: `python scripts/check_repo_hygiene.py`
- Run an example: `python examples/quick_start.py`
- API (optional deps): `python -m stateset_agents.api.main` (requires `fastapi` and `uvicorn`).

## Coding Style & Naming Conventions
- Indentation: 4 spaces; line length 88–100 preferred.
- Type hints required for public APIs; add docstrings (PEP 257).
- Naming: files/functions/vars `snake_case`; classes `PascalCase`; constants `UPPER_CASE`.
- Keep imports sorted (`isort`) and code formatted (`black`). Use `logging` over `print`.

## Testing Guidelines
- Framework: `pytest` (+ `pytest-asyncio` for `async def` tests).
- Location/pattern: place tests in `tests/` and name `test_*.py`.
- Add unit tests with new behavior; prefer fast, deterministic tests. Mark slow/integration tests clearly.
- Run locally: `pytest -q`; select tests with `pytest -k name_fragment`.

## Commit & Pull Request Guidelines
- Commits: clear, imperative subject; concise body. Prefer Conventional Commit prefixes when helpful (e.g., `feat:`, `fix:`, `docs:`).
- PRs: include a summary, rationale, linked issues, test evidence (logs/screenshots), and any breaking-change notes.
- Before opening a PR: run `ruff`, `black`, `isort`, `python scripts/check_types.py --all`, and `pytest` with all tests passing.

## Security & Configuration Tips
- Do not hardcode secrets (e.g., model keys, W&B). Use environment variables (e.g., `WANDB_API_KEY`).
- Large-model/TRL features are optional; gate heavy dependencies and provide fallbacks when possible.
- When adding new agents/rewards, place core abstractions in `core/` and specialized variants in `rewards/`; include minimal docs and tests.
