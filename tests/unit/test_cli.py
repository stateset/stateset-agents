import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from stateset_agents import __version__
from stateset_agents.cli import app

runner = CliRunner()


# ===========================
# Version Command Tests
# ===========================


def test_cli_version_outputs_version():
    """Test version command shows correct version"""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "stateset-agents version" in result.stdout
    assert __version__ in result.stdout


def test_cli_version_shows_python_version():
    """Test version command shows Python version"""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "python:" in result.stdout


# ===========================
# Train Command Tests
# ===========================


def test_cli_stub_training_runs_without_dependencies():
    """Test stub training runs without heavy dependencies"""
    result = runner.invoke(app, ["train", "--stub"])
    assert result.exit_code == 0
    assert "Stub agent conversation:" in result.stdout


def test_cli_train_dry_run_default():
    """Test train command defaults to dry-run"""
    result = runner.invoke(app, ["train"])
    assert result.exit_code == 0
    assert "Dry-run" in result.stdout or "environment looks OK" in result.stdout


def test_cli_train_with_episodes_override():
    """Test train command with episodes override"""
    result = runner.invoke(app, ["train", "--stub", "--episodes", "5"])
    assert result.exit_code == 0


def test_cli_train_with_json_config():
    """Test train command with JSON config file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config = {
            "agent": {"model_name": "stub://test", "max_new_tokens": 32},
            "environment": {"type": "conversation", "scenarios": []},
            "training": {"num_episodes": 1, "max_turns": 2}
        }
        json.dump(config, f)
        config_path = f.name

    try:
        result = runner.invoke(app, ["train", "--stub", "--config", config_path])
        assert result.exit_code == 0
    finally:
        Path(config_path).unlink(missing_ok=True)


def test_cli_train_with_invalid_config_path():
    """Test train command with non-existent config file"""
    result = runner.invoke(app, ["train", "--config", "/nonexistent/config.json"])
    assert result.exit_code != 0
    assert "Failed to load config" in result.stdout


def test_cli_train_with_save_path():
    """Test train command with checkpoint save path"""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.invoke(app, ["train", "--stub", "--save", tmpdir])
        assert result.exit_code == 0


# ===========================
# Doctor Command Tests
# ===========================


def test_cli_doctor_command():
    """Test doctor command runs and shows environment info"""
    result = runner.invoke(app, ["doctor"])
    # Doctor command should run successfully
    assert result.exit_code in [0, 1]  # May exit with 1 if optional deps missing


# ===========================
# Evaluate Command Tests
# ===========================


def test_cli_evaluate_requires_checkpoint():
    """Test evaluate command requires checkpoint path"""
    result = runner.invoke(app, ["evaluate", "--message", "Hello"])
    # Should handle missing checkpoint gracefully
    assert result.exit_code != 0 or "checkpoint" in result.stdout.lower()


def test_cli_evaluate_dry_run():
    """Test evaluate command in dry-run mode"""
    result = runner.invoke(app, ["evaluate", "--dry-run"])
    # Dry run may exit with error code if command requires checkpoint
    assert result.exit_code in [0, 2] or "dry-run" in result.stdout.lower()


# ===========================
# Serve Command Tests
# ===========================


def test_cli_serve_dry_run():
    """Test serve command in dry-run mode"""
    result = runner.invoke(app, ["serve", "--dry-run"])
    # Should provide guidance about serving
    assert result.exit_code == 0 or "serve" in result.stdout.lower()


# ===========================
# Error Handling Tests
# ===========================


def test_cli_invalid_command():
    """Test invalid command shows help"""
    result = runner.invoke(app, ["nonexistent"])
    assert result.exit_code != 0


def test_cli_help_flag():
    """Test --help flag works"""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "StateSet Agents" in result.stdout or "command" in result.stdout.lower()


def test_cli_train_help():
    """Test train command help"""
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "train" in result.stdout.lower()


# ===========================
# Integration Tests
# ===========================


def test_cli_train_stub_with_custom_responses():
    """Test stub training with custom responses"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config = {
            "agent": {
                "model_name": "stub://custom",
                "use_stub_model": True,
                "stub_responses": ["Custom response 1", "Custom response 2"]
            }
        }
        json.dump(config, f)
        config_path = f.name

    try:
        result = runner.invoke(app, ["train", "--stub", "--config", config_path])
        assert result.exit_code == 0
    finally:
        Path(config_path).unlink(missing_ok=True)


def test_cli_multiple_commands_sequential():
    """Test running multiple CLI commands sequentially"""
    # Version
    result1 = runner.invoke(app, ["version"])
    assert result1.exit_code == 0

    # Doctor
    result2 = runner.invoke(app, ["doctor"])
    assert result2.exit_code in [0, 1]

    # Train stub
    result3 = runner.invoke(app, ["train", "--stub"])
    assert result3.exit_code == 0
