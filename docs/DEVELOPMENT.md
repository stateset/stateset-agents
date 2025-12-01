# Development Guide

This guide covers setting up a development environment for StateSet Agents.

## Prerequisites

- Python 3.8+ (3.10+ recommended)
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
├── core/                   # Core RL abstractions
│   ├── agent.py           # Agent base classes
│   ├── environment.py     # Environment interface
│   ├── reward.py          # Reward functions (facade)
│   ├── reward_base.py     # Base reward classes
│   ├── basic_rewards.py   # Standard reward implementations
│   ├── domain_rewards.py  # Domain-specific rewards
│   ├── reward_factories.py # Reward creation utilities
│   └── trajectory.py      # Trajectory management
├── training/              # Training infrastructure
│   ├── trainer.py         # Trainer exports (facade)
│   ├── trainer_utils.py   # Utility functions
│   ├── single_turn_trainer.py
│   ├── multi_turn_trainer.py
│   ├── loss_computation.py
│   └── config.py          # Training configuration
├── stateset_agents/       # Public package namespace
├── api/                   # REST API services
├── utils/                 # Utilities (W&B, monitoring)
├── tests/                 # Test suite
├── examples/              # Usage examples
├── docs/                  # Documentation
└── deployment/            # Deployment configurations
```

---

## Running Tests

### Full Test Suite
```bash
pytest
```

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
pytest tests/unit/test_reward.py::TestHelpfulnessReward -v
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
mypy stateset_agents
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
mypy stateset_agents
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
from training.trainer_utils import require_torch

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
