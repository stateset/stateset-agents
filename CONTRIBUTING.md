# Contributing to StateSet RL Agent Framework

Thank you for your interest in contributing to the StateSet RL Agent Framework! We welcome contributions from everyone.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Submitting Changes](#submitting-changes)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project follows a [Code of Conduct](CODE_OF_CONDUCT.md) to ensure a welcoming environment for all contributors.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- A GitHub account

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/stateset-agents.git
   cd stateset-agents
   ```
3. Set up the upstream remote:
   ```bash
   git remote add upstream https://github.com/stateset/stateset-agents.git
   ```

## Development Setup

### Quick Setup

```bash
# Install with development dependencies
make dev-setup
# or manually:
# pip install -e ".[dev]"
# pre-commit install
```

### Manual Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

### 1. Choose an Issue

- Check the [Issues](https://github.com/stateset/stateset-agents/issues) page
- Look for issues labeled `good first issue` or `help wanted`
- Comment on the issue to indicate you're working on it

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 3. Make Changes

- Write clear, focused commits
- Follow the coding standards
- Add tests for new functionality
- Update documentation as needed

### 4. Run Quality Checks

```bash
# Run all checks
make ci

# Or run individually:
make lint          # Code formatting and linting
make check-types   # Type checking
make test-unit     # Unit tests
make test-cov      # Tests with coverage
```

### 5. Commit Changes

```bash
# Stage your changes
git add .

# Commit with a clear message
git commit -m "feat: add new feature description

- What was changed
- Why it was changed
- Any breaking changes
"
```

## Submitting Changes

### Pull Request Process

1. Ensure your branch is up to date:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. Push your branch:
   ```bash
   git push origin your-branch-name
   ```

3. Create a Pull Request:
   - Go to your fork on GitHub
   - Click "New Pull Request"
   - Fill out the PR template
   - Link to any relevant issues

### PR Requirements

- ✅ All CI checks pass
- ✅ Code is reviewed and approved
- ✅ Tests are included (if applicable)
- ✅ Documentation is updated (if applicable)
- ✅ Commit messages follow conventional format

## Coding Standards

### Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting (88 character line length)
- **isort**: Import sorting
- **Ruff**: Fast Python linter
- **MyPy**: Static type checking

### Python Version Support

- Python 3.8+ is supported
- Code should work on all supported versions
- Use modern Python features when appropriate

### Type Hints

- Use type hints for all function parameters and return values
- Use `typing` module for complex types
- Document complex type relationships

### Naming Conventions

- **Classes**: `PascalCase`
- **Functions/Methods**: `snake_case`
- **Constants**: `UPPER_CASE`
- **Private members**: `_leading_underscore`

### Docstrings

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """Brief description of what the function does.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When something goes wrong
    """
```

## Testing

### Test Structure

- Unit tests in `tests/` directory
- Test files named `test_*.py`
- Use `pytest` framework
- Aim for 80%+ code coverage

### Writing Tests

```python
import pytest
from stateset_agents.core.agent import Agent

class TestAgent:
    def test_agent_initialization(self):
        """Test that agent initializes correctly."""
        agent = Agent(config)
        assert agent is not None
    
    def test_agent_response_generation(self):
        """Test that agent generates responses."""
        # Test implementation
        pass
```

### Running Tests

```bash
# All tests
make test

# Unit tests only
make test-unit

# With coverage
make test-cov

# Specific test file
pytest tests/test_specific.py

# Tests with markers
pytest -m "slow"  # Run slow tests
pytest -m "gpu"   # Run GPU tests
```

## Documentation

### Documentation Types

- **README.md**: Project overview and quick start
- **API Documentation**: Auto-generated from docstrings
- **Guides**: In-depth tutorials and examples
- **Changelog**: Version history and changes

### Building Documentation

```bash
# Install docs dependencies
pip install sphinx sphinx-rtd-theme

# Build HTML documentation
make docs

# Serve locally
make docs-serve
```

### Documentation Standards

- Write in clear, simple English
- Include code examples where helpful
- Keep screenshots up to date
- Test all code examples

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Steps to reproduce**: Detailed steps
- **Environment**: Python version, OS, dependencies
- **Error messages**: Full traceback if applicable

### Feature Requests

For feature requests, please include:

- **Problem**: What's the problem you're trying to solve?
- **Solution**: Describe your proposed solution
- **Alternatives**: Have you considered alternatives?
- **Use case**: How would this feature be used?

### Security Issues

- **DO NOT** report security issues in public
- Email security concerns to: security@stateset.ai
- Include detailed information about the vulnerability

## Getting Help

- **Documentation**: Check the [docs](https://stateset-agents.readthedocs.io/)
- **Issues**: Search existing [issues](https://github.com/stateset/stateset-agents/issues)
- **Discussions**: Use [GitHub Discussions](https://github.com/stateset/stateset-agents/discussions) for questions
- **Discord**: Join our [Discord community](https://discord.gg/stateset)

## Recognition

Contributors are recognized in:
- GitHub's contributor insights
- CHANGELOG.md for significant contributions
- Release notes
- Our website

Thank you for contributing to the StateSet RL Agent Framework! 🚀
