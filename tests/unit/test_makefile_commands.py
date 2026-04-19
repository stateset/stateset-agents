"""Regression tests for important Makefile entrypoints."""

from __future__ import annotations

from pathlib import Path
import subprocess


def test_make_check_types_uses_stable_wrapper() -> None:
    makefile_path = Path(__file__).resolve().parents[2] / "Makefile"
    contents = makefile_path.read_text()

    assert "check-types: ## Run mypy type checking" in contents
    assert "\tpython scripts/check_types.py --all" in contents
    assert "\tmypy --config-file mypy.ini" not in contents


def test_make_test_cov_targets_packaged_module() -> None:
    makefile_path = Path(__file__).resolve().parents[2] / "Makefile"
    contents = makefile_path.read_text()

    assert "test-cov: ## Run tests with coverage report" in contents
    assert (
        "\tpytest --cov=stateset_agents --cov-report=html --cov-report=term-missing"
        in contents
    )


def test_make_ci_uses_read_only_checks() -> None:
    makefile_path = Path(__file__).resolve().parents[2] / "Makefile"
    contents = makefile_path.read_text()

    assert "ci: ## Simulate CI pipeline locally" in contents
    assert "\t$(MAKE) lint" in contents
    assert "\t$(MAKE) check-types" in contents
    assert "\t$(MAKE) test-unit" in contents
    assert "\t$(MAKE) test-cov" in contents


def test_make_docs_uses_docs_safe_environment() -> None:
    makefile_path = Path(__file__).resolve().parents[2] / "Makefile"
    contents = makefile_path.read_text()

    assert "SPHINX_DOCS_ENV := API_REQUIRE_AUTH=false INFERENCE_BACKEND=stub" in contents
    assert "\t$(SPHINX_DOCS_ENV) sphinx-build docs docs/_build/html" in contents


def test_make_help_renders_available_commands() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["make", "help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Available commands:" in result.stdout
    assert "check-types" in result.stdout
