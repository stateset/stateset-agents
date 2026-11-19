# Development Guide

This guide covers setting up a development environment for StateSet Agents.

For lifecycle changes, consult the [state machine inventory](STATE_MACHINES.md)
for canonical owners, transition boundaries, recovery rules, and known gaps.

For related persistence writes, use `UnitOfWork.transaction()` explicitly.
The [transaction example](../examples/transactional_persistence.py) demonstrates
commit, rollback, and reopening SQLite; ordinary repository calls remain
individually committed outside a transaction.

## Prerequisites

- Python 3.10+
- Git
- pip or uv (recommended)
- Optional: CUDA toolkit for GPU support

## Quick Setup

### 1. Clone the Repository

```bash
git clone https://github.com/stateset/stateset-agents.git
cd stateset-agents
```

### 2. Create Virtual Environment

Using venv:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows
```

Using uv (faster):
```bash
uv venv
source .venv/bin/activate
```

### 3. Install Dependencies

Development install with all extras:
```bash
pip install -e ".[dev,training,api,examples,hpo]"
```

Or using uv:
```bash
uv pip install -e ".[dev,training,api,examples,hpo]"
```

### 4. Install Pre-commit Hooks

```bash
pre-commit install
```

---

## Project Structure

```
stateset-agents/
├── stateset_agents/       # Canonical source packaged for PyPI
│   ├── core/              # Agents, environments, rewards, trajectories
│   ├── training/          # Trainers, configurations, and training loops
│   ├── rewards/           # Specialized reward functions
│   ├── utils/             # Logging, monitoring, and shared utilities
│   └── api/               # FastAPI gateway and service infrastructure
├── tests/                 # Test suite
├── examples/              # Usage examples
├── scripts/               # Development and release utilities
├── docs/                  # Documentation
└── deployment/            # Deployment configurations
```

The top-level `api/`, `core/`, `training/`, `rewards/`, and `environments/`
directories are deprecated compatibility shims for in-repository callers.
Implement changes in `stateset_agents/` and prefer `stateset_agents.*` imports.

---

## Running Tests

### Default Test Suite
```bash
pytest
```

The default selection excludes tests marked `performance`, `slow`, or `gpu`.
Use `-m` to select those categories explicitly when their dependencies and
hardware are available.

### With Coverage
```bash
pytest --cov=stateset_agents --cov-report=html
open htmlcov/index.html
```

### Specific Test Categories
```bash
# Unit tests only
pytest -m unit

# Integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# GPU tests (requires CUDA)
pytest -m gpu
```

### Running a Single Test
```bash
pytest tests/unit/test_error_codes.py::test_grpo_exception_sets_default_error_code -v -n 0
```

---

## Code Quality

### Formatting
```bash
# Format code
black .
isort .

# Check formatting without changes
black --check .
isort --check .
```

### Linting
```bash
# Run ruff linter
ruff check .

# With auto-fix
ruff check --fix .
```

### Type Checking
```bash
python scripts/check_types.py --all
```

### Security Scanning
```bash
bandit -r stateset_agents
```

### All Checks (via pre-commit)
```bash
pre-commit run --all-files
```

---

## Development Workflow

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes
- Write code following existing patterns
- Add tests for new functionality
- Update documentation as needed

### 3. Run Checks
```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Run tests
pytest

# Type check
python scripts/check_types.py --all
```

### 4. Commit
```bash
git add .
git commit -m "feat: description of your changes"
```

Follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance

### 5. Push and Create PR
```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

---

## Working with Optional Dependencies

StateSet Agents uses optional dependencies extensively. When developing:

### Check if Optional Dependency is Available
```python
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
```

### Use Lazy Imports
```python
from stateset_agents.training.trainer_utils import require_torch

def training_function():
    torch = require_torch()  # Raises helpful error if not installed
    # Use torch...
```

### Skip Tests for Missing Dependencies
```python
import pytest

torch = pytest.importorskip("torch")  # Skips test if torch not available
```

---

## Debugging

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Interactive Debugging
```python
import pdb; pdb.set_trace()
# or
breakpoint()  # Python 3.7+
```

### Async Debugging
```python
import asyncio

async def debug_async():
    # Your async code
    pass

asyncio.run(debug_async())
```

---

## Building Documentation

```bash
cd docs
make html
open _build/html/index.html
```

---

## Creating a Release

See [RELEASE_GUIDE.md](./RELEASE_GUIDE.md) for detailed release instructions.

Quick steps:
1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release PR
4. Tag and publish

---

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/stateset/stateset-agents/issues)
- **Discussions**: [GitHub Discussions](https://github.com/stateset/stateset-agents/discussions)
- **Discord**: Link in README

---

## IDE Setup

### VS Code

Recommended extensions:
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- Black Formatter (ms-python.black-formatter)
- Ruff (charliermarsh.ruff)

Settings (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "[python]": {
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        }
    }
}
```

### PyCharm

1. Set Python interpreter to `.venv/bin/python`
2. Enable Black formatter
3. Configure isort for imports
4. Enable mypy plugin

---

*Happy coding!*
