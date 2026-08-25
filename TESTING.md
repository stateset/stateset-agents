# Testing Guide

## Test Architecture

Tests live under `tests/` and are organized by type:

```
tests/
├── conftest.py              # Shared fixtures (stub agents, force_cpu, etc.)
├── unit/                    # Fast, isolated unit tests
├── integration/             # Cross-module integration tests
├── api/                     # FastAPI endpoint tests
├── e2e/                     # End-to-end scenario tests
└── performance/             # Load and benchmark tests
```

## When to Use StubBackend vs MagicMock

### Use StubBackend (preferred)

The stub backend (`AgentConfig(use_stub_model=True)`) provides a real model-like
object that tokenizes and generates text deterministically.  Use it when:

- Testing agent behavior (memory window, streaming, conversation history)
- Testing training loops end-to-end
- Testing reward computation on actual text
- Any test that needs `agent.generate_response()` to return real strings

```python
from stateset_agents.core.agent import AgentConfig, MultiTurnAgent

config = AgentConfig(model_name="stub://test", use_stub_model=True)
agent = MultiTurnAgent(config)
await agent.initialize()
response = await agent.generate_response("Hello")
```

### Use MagicMock (sparingly)

Use mocks only when you need to:

- Verify a specific method was called with specific arguments
- Simulate error conditions that the stub can't produce
- Test code that interfaces with external services (W&B, HuggingFace Hub)

Never use mocks to avoid importing real modules — that hides real bugs.

## Key Fixtures

All fixtures are defined in `tests/conftest.py`:

| Fixture                  | Type        | Description                                    |
|--------------------------|-------------|------------------------------------------------|
| `stub_agent_config`      | sync        | `AgentConfig` with stub backend enabled        |
| `initialized_stub_agent` | async       | Fully initialized `MultiTurnAgent` (stub)      |
| `stub_backend`           | sync        | Raw `StubBackend` for direct model access      |
| `force_cpu`              | sync        | Patches `torch.cuda.is_available` to `False`   |

`force_cpu` is opt-in — add it to your test function signature only when your
test explicitly needs to force CPU mode.

## Exception Handling in Tests

Import canonical exception tuples from `stateset_agents.exceptions`:

```python
from stateset_agents.exceptions import (
    IMPORT_EXCEPTIONS,
    GPU_EXCEPTIONS,
    MODEL_IO_EXCEPTIONS,
    INFERENCE_EXCEPTIONS,
)
```

Do **not** define local exception tuples in new code.

## Writing Good Assertions

Prefer value assertions over type checks:

```python
# Bad — passes even if response is empty
assert isinstance(response, str)

# Good — verifies actual content
assert len(response) > 0
assert "expected_keyword" in response

# Bad — only checks existence
assert result is not None

# Good — checks the actual value
assert result.score == 0.75
assert abs(result.score - expected) < 1e-6
```

## Running Tests

```bash
# Full suite
python -m pytest tests/ -q

# Specific module
python -m pytest tests/unit/test_agent.py -v

# With coverage
python -m pytest tests/ --cov=stateset_agents --cov-report=term-missing
```

### Parallel execution (default)

`pytest.ini` sets `-n auto --dist loadfile`, so the suite runs under
[pytest-xdist](https://pytest-xdist.readthedocs.io/) across all available
cores by default (full suite: ~4 minutes on 8 cores). `--dist loadfile`
keeps every test in a file on the same worker, which preserves module-scoped
fixtures and module-level state.

Parallel output interleaves and breakpoints do not work, so disable xdist when
debugging a single test:

```bash
# Run serially — required for pdb/breakpoints and readable live output
python -m pytest tests/unit/test_agent.py -v -n0

# Fully disabling the plugin also means dropping the `-n auto` in addopts,
# otherwise pytest rejects the now-unknown option
python -m pytest tests/unit/test_agent.py -v -p no:xdist -o addopts="-ra"
```

Tests must therefore be parallel-safe: use the `tmp_path` fixture rather than
shared paths under `/tmp`, reset any global registries in a fixture, and bind
servers to an ephemeral port (`port=0`) instead of a hard-coded one.
