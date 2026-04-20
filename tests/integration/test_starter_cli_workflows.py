"""Integration tests for model-specific starter CLI workflows."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("preset", "command_name", "model_name", "task", "starter_profile"),
    [
        (
            "qwen3-5-0-8b",
            "qwen3-5-0-8b",
            "Qwen/Qwen3.5-0.8B-Base",
            "sales",
            "memory",
        ),
        (
            "kimi-k2-6",
            "kimi-k2-6",
            "moonshotai/Kimi-K2.6",
            "technical_support",
            "memory",
        ),
    ],
)
def test_model_specific_starter_configs_round_trip_via_cli(
    tmp_path: Path,
    preset: str,
    command_name: str,
    model_name: str,
    task: str,
    starter_profile: str,
) -> None:
    """Ensure starter configs survive the full CLI create/load workflow."""
    config_path = tmp_path / f"{preset}.json"

    init_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "stateset_agents.cli",
            "init",
            "--preset",
            preset,
            "--task",
            task,
            "--starter-profile",
            starter_profile,
            "--format",
            "json",
            "--path",
            str(config_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert config_path.exists()
    assert "Wrote starter config" in init_result.stdout

    scaffolded_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert scaffolded_config["model_name"] == model_name
    assert scaffolded_config["task"] == task
    assert scaffolded_config["starter_profile"] == starter_profile

    starter_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "stateset_agents.cli",
            command_name,
            "--config",
            str(config_path),
            "--json-output",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    preview = json.loads(starter_result.stdout)
    assert preview["config"]["model_name"] == model_name
    assert preview["config"]["task"] == task
    assert preview["config"]["starter_profile"] == starter_profile
    assert preview["summary"]["starter_profile"] == starter_profile
    assert preview["summary"]["quantization_mode"] == "4bit"
    assert preview["gspo_overrides"]["model_name"] == model_name
    assert preview["gspo_overrides"]["use_4bit"] is True
