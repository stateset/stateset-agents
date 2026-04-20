import json
import tempfile
from pathlib import Path
from unittest.mock import patch

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


def test_cli_version_json_output():
    """Test version command supports JSON output."""
    result = runner.invoke(app, ["version", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["name"] == "stateset-agents"
    assert payload["version"] == __version__


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
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        config = {
            "agent": {"model_name": "stub://test", "max_new_tokens": 32},
            "environment": {"type": "conversation", "scenarios": []},
            "training": {"num_episodes": 1, "max_turns": 2},
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
    assert "not found" in result.stdout.lower()


def test_cli_train_with_save_path():
    """Test train command with checkpoint save path"""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.invoke(app, ["train", "--stub", "--save", tmpdir])
        assert result.exit_code == 0


def test_cli_train_invalid_profile():
    """Test train command rejects unsupported profile values."""
    result = runner.invoke(app, ["train", "--stub", "--profile", "invalid-profile"])
    assert result.exit_code != 0
    assert "unsupported profile" in result.stdout.lower()


def test_cli_train_invalid_config_extension():
    """Test train command rejects unsupported config formats."""
    result = runner.invoke(app, ["train", "--config", "config.txt"])
    assert result.exit_code != 0
    assert "unsupported config format" in result.stdout.lower()


def test_cli_train_invalid_config_type():
    """Test train command rejects invalid config schema."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        config = {"agent": "invalid"}
        json.dump(config, f)
        config_path = f.name

    try:
        result = runner.invoke(app, ["train", "--stub", "--config", config_path])
        assert result.exit_code != 0
        assert "configuration validation failed" in result.stdout.lower()
    finally:
        Path(config_path).unlink(missing_ok=True)


def test_cli_train_rejects_invalid_scenario_shape():
    """Test train rejects malformed scenario items in config."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        config = {
            "environment": {"type": "conversation", "scenarios": [123]},
            "training": {"num_episodes": 1},
            "agent": {"model_name": "stub://test"},
        }
        json.dump(config, f)
        config_path = f.name

    try:
        result = runner.invoke(app, ["train", "--stub", "--config", config_path])
        assert result.exit_code != 0
        assert "environment.scenarios[0]" in result.stdout.lower()
    finally:
        Path(config_path).unlink(missing_ok=True)


def test_cli_train_negative_episodes():
    """Test train command rejects negative episode count."""
    result = runner.invoke(app, ["train", "--stub", "--episodes", "-1"])
    assert result.exit_code != 0
    assert "positive integer" in result.stdout.lower()

def test_cli_qwen3_5_0_8b_dry_run_json():
    """Test the dedicated Qwen 3.5 starter preview payload."""
    result = runner.invoke(app, ["qwen3-5-0-8b", "--json-output"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["config"]["model_name"] == "Qwen/Qwen3.5-0.8B-Base"
    assert payload["config"]["task"] == "customer_service"
    assert payload["config"]["starter_profile"] == "balanced"
    assert payload["gspo_overrides"]["use_lora"] is True
    assert payload["gspo_overrides"]["output_dir"] == "./outputs/qwen3_5_0_8b_gspo"


def test_cli_qwen3_5_0_8b_memory_profile_json():
    """Test the low-memory Qwen starter profile."""
    result = runner.invoke(
        app,
        ["qwen3-5-0-8b", "--starter-profile", "memory", "--json-output"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["config"]["starter_profile"] == "memory"
    assert payload["summary"]["quantization_mode"] == "4bit"
    assert payload["gspo_overrides"]["use_4bit"] is True
    assert payload["gspo_overrides"]["num_generations"] == 2
    assert payload["gspo_overrides"]["num_outer_iterations"] == 15


def test_cli_qwen3_5_0_8b_list_profiles_json():
    """Test the starter can describe all built-in profiles."""
    result = runner.invoke(
        app,
        ["qwen3-5-0-8b", "--task", "sales", "--list-profiles", "--json-output"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["task"] == "sales"
    assert payload["default_profile"] == "balanced"
    assert payload["profiles"]["memory"]["summary"]["quantization_mode"] == "4bit"
    assert payload["profiles"]["quality"]["summary"]["num_generations"] == 6


def test_cli_qwen3_5_0_8b_list_profiles_text():
    """Test the human-readable profile catalog output."""
    result = runner.invoke(app, ["qwen3-5-0-8b", "--list-profiles"])
    assert result.exit_code == 0
    assert "Available Qwen3.5-0.8B starter profiles:" in result.stdout
    assert "- balanced:" in result.stdout
    assert "- memory:" in result.stdout
    assert "- quality:" in result.stdout
    assert "quantization=4bit" in result.stdout


def test_cli_qwen3_5_0_8b_rejects_unknown_task():
    """Test the dedicated Qwen starter rejects unsupported task names."""
    result = runner.invoke(app, ["qwen3-5-0-8b", "--task", "unknown_task"])
    assert result.exit_code != 0
    assert "unsupported task" in result.stdout.lower()


def test_cli_qwen3_5_0_8b_rejects_unknown_starter_profile():
    """Test the dedicated Qwen starter rejects unsupported profile names."""
    result = runner.invoke(app, ["qwen3-5-0-8b", "--starter-profile", "unknown_profile"])
    assert result.exit_code != 0
    assert "unsupported starter profile" in result.stdout.lower()

def test_cli_qwen3_5_0_8b_write_config_json(tmp_path):
    """Test the dedicated Qwen starter can write a reusable config file."""
    cfg_path = tmp_path / "qwen3_5_0_8b.json"
    result = runner.invoke(app, ["qwen3-5-0-8b", "--write-config", str(cfg_path)])
    assert result.exit_code == 0
    assert cfg_path.exists()
    loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert loaded["model_name"] == "Qwen/Qwen3.5-0.8B-Base"
    assert loaded["task"] == "customer_service"


def test_cli_qwen3_5_0_8b_load_config_json(tmp_path):
    """Test the dedicated Qwen starter can load a saved config file."""
    cfg_path = tmp_path / "qwen3_5_custom.json"
    cfg_path.write_text(
        json.dumps(
            {
                "model_name": "Qwen/Qwen3.5-0.8B-Base",
                "task": "sales",
                "output_dir": "./outputs/qwen3_5_loaded",
                "learning_rate": 1e-5,
                "use_lora": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["qwen3-5-0-8b", "--config", str(cfg_path), "--json-output"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["config"]["task"] == "sales"
    assert payload["config"]["output_dir"] == "./outputs/qwen3_5_loaded"
    assert payload["gspo_overrides"]["use_lora"] is False


def test_cli_qwen3_5_0_8b_rejects_config_with_overrides(tmp_path):
    """Test the dedicated Qwen starter rejects config files mixed with override flags."""
    cfg_path = tmp_path / "qwen3_5_conflict.json"
    cfg_path.write_text(json.dumps({"model_name": "Qwen/Qwen3.5-0.8B-Base"}), encoding="utf-8")
    result = runner.invoke(
        app,
        ["qwen3-5-0-8b", "--config", str(cfg_path), "--use-4bit"],
    )
    assert result.exit_code != 0
    assert "cannot be combined" in result.stdout.lower()


def test_cli_qwen3_5_0_8b_rejects_config_with_profile_override(tmp_path):
    """Test the dedicated Qwen starter rejects config files mixed with starter profiles."""
    cfg_path = tmp_path / "qwen3_5_profile_conflict.json"
    cfg_path.write_text(json.dumps({"model_name": "Qwen/Qwen3.5-0.8B-Base"}), encoding="utf-8")
    result = runner.invoke(
        app,
        ["qwen3-5-0-8b", "--config", str(cfg_path), "--starter-profile", "memory"],
    )
    assert result.exit_code != 0
    assert "cannot be combined" in result.stdout.lower()


def test_cli_qwen3_5_0_8b_rejects_list_profiles_with_config(tmp_path):
    """Test profile discovery stays separate from config-file execution mode."""
    cfg_path = tmp_path / "qwen3_5_list_profiles_conflict.json"
    cfg_path.write_text(json.dumps({"model_name": "Qwen/Qwen3.5-0.8B-Base"}), encoding="utf-8")
    result = runner.invoke(
        app,
        ["qwen3-5-0-8b", "--config", str(cfg_path), "--list-profiles"],
    )
    assert result.exit_code != 0
    assert "cannot be combined" in result.stdout.lower()


def test_cli_kimi_k2_6_dry_run_json():
    """Test the dedicated Kimi-K2.6 starter preview payload."""
    result = runner.invoke(app, ["kimi-k2-6", "--json-output"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["config"]["model_name"] == "moonshotai/Kimi-K2.6"
    assert payload["config"]["task"] == "customer_service"
    assert payload["config"]["starter_profile"] == "balanced"
    assert payload["gspo_overrides"]["use_lora"] is True
    assert payload["gspo_overrides"]["output_dir"] == "./outputs/kimi_k2_6_gspo"


def test_cli_kimi_k2_6_memory_profile_json():
    """Test the low-memory Kimi starter profile."""
    result = runner.invoke(
        app,
        ["kimi-k2-6", "--starter-profile", "memory", "--json-output"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["config"]["starter_profile"] == "memory"
    assert payload["summary"]["quantization_mode"] == "4bit"
    assert payload["gspo_overrides"]["use_4bit"] is True
    assert payload["gspo_overrides"]["num_generations"] == 2
    assert payload["gspo_overrides"]["num_outer_iterations"] == 10


def test_cli_kimi_k2_6_list_profiles_json():
    """Test the starter can describe all built-in Kimi profiles."""
    result = runner.invoke(
        app,
        ["kimi-k2-6", "--task", "sales", "--list-profiles", "--json-output"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["task"] == "sales"
    assert payload["default_profile"] == "balanced"
    assert payload["profiles"]["memory"]["summary"]["quantization_mode"] == "4bit"
    assert payload["profiles"]["quality"]["summary"]["num_generations"] == 6


def test_cli_kimi_k2_6_list_profiles_text():
    """Test the human-readable Kimi profile catalog output."""
    result = runner.invoke(app, ["kimi-k2-6", "--list-profiles"])
    assert result.exit_code == 0
    assert "Available Kimi-K2.6 starter profiles:" in result.stdout
    assert "- balanced:" in result.stdout
    assert "- memory:" in result.stdout
    assert "- quality:" in result.stdout
    assert "quantization=4bit" in result.stdout


def test_cli_kimi_k2_6_rejects_unknown_task():
    """Test the dedicated Kimi starter rejects unsupported task names."""
    result = runner.invoke(app, ["kimi-k2-6", "--task", "unknown_task"])
    assert result.exit_code != 0
    assert "unsupported task" in result.stdout.lower()


def test_cli_kimi_k2_6_rejects_unknown_starter_profile():
    """Test the dedicated Kimi starter rejects unsupported profile names."""
    result = runner.invoke(app, ["kimi-k2-6", "--starter-profile", "unknown_profile"])
    assert result.exit_code != 0
    assert "unsupported starter profile" in result.stdout.lower()


def test_cli_kimi_k2_6_write_config_json(tmp_path):
    """Test the dedicated Kimi starter can write a reusable config file."""
    cfg_path = tmp_path / "kimi_k2_6.json"
    result = runner.invoke(app, ["kimi-k2-6", "--write-config", str(cfg_path)])
    assert result.exit_code == 0
    assert cfg_path.exists()
    loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert loaded["model_name"] == "moonshotai/Kimi-K2.6"
    assert loaded["task"] == "customer_service"


def test_cli_kimi_k2_6_load_config_json(tmp_path):
    """Test the dedicated Kimi starter can load a saved config file."""
    cfg_path = tmp_path / "kimi_k2_6_custom.json"
    cfg_path.write_text(
        json.dumps(
            {
                "model_name": "moonshotai/Kimi-K2.6",
                "task": "sales",
                "output_dir": "./outputs/kimi_k2_6_loaded",
                "learning_rate": 1e-5,
                "use_lora": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["kimi-k2-6", "--config", str(cfg_path), "--json-output"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["config"]["task"] == "sales"
    assert payload["config"]["output_dir"] == "./outputs/kimi_k2_6_loaded"
    assert payload["gspo_overrides"]["use_lora"] is False


def test_cli_kimi_k2_6_rejects_config_with_overrides(tmp_path):
    """Test the dedicated Kimi starter rejects config files mixed with override flags."""
    cfg_path = tmp_path / "kimi_k2_6_conflict.json"
    cfg_path.write_text(json.dumps({"model_name": "moonshotai/Kimi-K2.6"}), encoding="utf-8")
    result = runner.invoke(
        app,
        ["kimi-k2-6", "--config", str(cfg_path), "--use-4bit"],
    )
    assert result.exit_code != 0
    assert "cannot be combined" in result.stdout.lower()


def test_cli_kimi_k2_6_rejects_config_with_profile_override(tmp_path):
    """Test the dedicated Kimi starter rejects config files mixed with starter profiles."""
    cfg_path = tmp_path / "kimi_k2_6_profile_conflict.json"
    cfg_path.write_text(json.dumps({"model_name": "moonshotai/Kimi-K2.6"}), encoding="utf-8")
    result = runner.invoke(
        app,
        ["kimi-k2-6", "--config", str(cfg_path), "--starter-profile", "memory"],
    )
    assert result.exit_code != 0
    assert "cannot be combined" in result.stdout.lower()


def test_cli_kimi_k2_6_rejects_list_profiles_with_config(tmp_path):
    """Test Kimi profile discovery stays separate from config-file execution mode."""
    cfg_path = tmp_path / "kimi_k2_6_list_profiles_conflict.json"
    cfg_path.write_text(json.dumps({"model_name": "moonshotai/Kimi-K2.6"}), encoding="utf-8")
    result = runner.invoke(
        app,
        ["kimi-k2-6", "--config", str(cfg_path), "--list-profiles"],
    )
    assert result.exit_code != 0
    assert "cannot be combined" in result.stdout.lower()


def test_cli_validate_config_command_success():
    """Test config validator accepts a valid config."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        config = {
            "agent": {"model_name": "gpt2", "max_new_tokens": 64, "temperature": 0.6},
            "environment": {
                "type": "conversation",
                "max_turns": 3,
                "scenarios": [{"id": "s1", "topic": "help", "context": "x"}],
            },
            "training": {"num_episodes": 3, "max_turns": 2},
            "profile": "balanced",
        }
        json.dump(config, f)
        config_path = f.name

    try:
        result = runner.invoke(app, ["validate-config", "--config", config_path])
        assert result.exit_code == 0
        assert "Configuration validation passed." in result.stdout
    finally:
        Path(config_path).unlink(missing_ok=True)


def test_cli_validate_config_strict_json_reports_errors():
    """Test strict JSON validate-config mode fails on invalid config values."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        config = {
            "agent": {"model_name": "gpt2", "max_new_tokens": "bad"},
            "training": {"num_episodes": -1},
        }
        json.dump(config, f)
        config_path = f.name

    try:
        result = runner.invoke(
            app,
            [
                "validate-config",
                "--config",
                config_path,
                "--strict",
                "--json-output",
            ],
        )
        assert result.exit_code == 2
        payload = json.loads(result.stdout)
        assert payload["valid"] is False
        assert payload["strict"] is True
        assert any("max_new_tokens" in err for err in payload["errors"])
    finally:
        Path(config_path).unlink(missing_ok=True)


def test_cli_validate_config_unknown_key_warning():
    """Test config validator reports unknown top-level keys as warnings."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        config = {"agent": {"model_name": "gpt2"}, "mystery_key": 123}
        json.dump(config, f)
        config_path = f.name

    try:
        result = runner.invoke(app, ["validate-config", "--config", config_path])
        assert result.exit_code == 0
        assert "warning:" in result.stdout.lower()
    finally:
        Path(config_path).unlink(missing_ok=True)


def test_cli_validate_config_rejects_bad_scenario_items():
    """Test validate-config rejects non-dictionary scenario entries."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        config = {
            "environment": {
                "type": "conversation",
                "scenarios": ["not-a-dict"],
            }
        }
        json.dump(config, f)
        config_path = f.name

    try:
        result = runner.invoke(
            app, ["validate-config", "--config", config_path]
        )
        assert result.exit_code != 0
        assert "environment.scenarios[0]" in result.stdout.lower()
        assert "must be an object" in result.stdout.lower()
    finally:
        Path(config_path).unlink(missing_ok=True)


def test_cli_validate_config_rejects_bad_scenario_user_responses():
    """Test validate-config rejects non-string scenario user responses."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        config = {
            "environment": {
                "type": "conversation",
                "scenarios": [
                    {
                        "id": "s1",
                        "user_responses": [42],
                    }
                ],
            }
        }
        json.dump(config, f)
        config_path = f.name

    try:
        result = runner.invoke(
            app, ["validate-config", "--config", config_path]
        )
        assert result.exit_code != 0
        assert "user_responses[0]" in result.stdout.lower()
        assert "must be a string" in result.stdout.lower()
    finally:
        Path(config_path).unlink(missing_ok=True)


def test_cli_validate_config_fail_on_warnings_json():
    """Test warning-only validations can be treated as fatal."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        config = {"agent": {"model_name": "gpt2"}, "mystery_key": 123}
        json.dump(config, f)
        config_path = f.name

    try:
        result = runner.invoke(
            app,
            [
                "validate-config",
                "--config",
                config_path,
                "--fail-on-warnings",
                "--json-output",
            ],
        )
        assert result.exit_code == 2
        payload = json.loads(result.stdout)
        assert payload["failed"] is True
        assert payload["fail_on_warnings"] is True
        assert payload["valid"] is True
        assert payload["warnings"]
    finally:
        Path(config_path).unlink(missing_ok=True)


# ===========================
# Doctor Command Tests
# ===========================


def test_cli_doctor_command():
    """Test doctor command runs and shows environment info"""
    result = runner.invoke(app, ["doctor"])
    # Doctor command should run successfully
    assert result.exit_code in [0, 1]  # May exit with 1 if optional deps missing


def test_cli_doctor_json_output():
    """Test doctor command supports JSON output."""
    result = runner.invoke(app, ["doctor", "--json-output"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["name"] == "stateset-agents"
    assert "required_dependencies" in payload


def test_cli_doctor_json_output_strict_mode():
    """Test strict mode is represented in doctor JSON output."""
    result = runner.invoke(app, ["doctor", "--json-output", "--strict"])
    assert result.exit_code in [0, 2]
    payload = json.loads(result.stdout)
    assert payload["strict"] is True
    assert "required_dependencies" in payload


def test_cli_preflight_json_no_config():
    """Test preflight runs successfully in JSON mode without config."""
    result = runner.invoke(app, ["preflight", "--json-output"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["name"] == "stateset-agents"
    assert payload["config"]["path"] is None
    assert "dependencies" in payload


def test_cli_preflight_strict_config_errors():
    """Test preflight fails when config has validation errors."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        config = {"agent": "invalid"}
        json.dump(config, f)
        config_path = f.name

    try:
        result = runner.invoke(
            app,
            ["preflight", "--config", config_path, "--json-output"],
        )
        assert result.exit_code == 2
        payload = json.loads(result.stdout)
        assert payload["failed"] is True
        assert payload["config"]["valid"] is False
        assert payload["config"]["errors"]
    finally:
        Path(config_path).unlink(missing_ok=True)


def test_cli_preflight_warning_gate():
    """Test preflight can fail on warnings."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        config = {"agent": {"model_name": "gpt2"}, "mystery_key": 123}
        json.dump(config, f)
        config_path = f.name

    try:
        result = runner.invoke(
            app,
            [
                "preflight",
                "--config",
                config_path,
                "--fail-on-warnings",
                "--json-output",
            ],
        )
        assert result.exit_code == 2
        payload = json.loads(result.stdout)
        assert payload["failed"] is True
        assert payload["config"]["warnings"]
    finally:
        Path(config_path).unlink(missing_ok=True)


def test_cli_publish_check_json_no_config():
    """Test publish-check runs successfully in JSON mode without config."""
    result = runner.invoke(app, ["publish-check", "--json-output"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["name"] == "stateset-agents"
    assert payload["config"]["path"] is None
    assert "imports" in payload
    assert payload["publish_ready"] is True


def test_cli_publish_check_strict_required_import_failure():
    """Test publish-check fails when required imports are unavailable."""
    def fake_collect_import_status(modules: list[str]) -> dict[str, bool]:
        modules_set = set(modules)
        if "stateset_agents.core.environment" in modules_set:
            return {name: name != "stateset_agents.training" for name in modules}
        if "stateset_agents.api.main" in modules_set:
            return dict.fromkeys(modules, False)
        return dict.fromkeys(modules, True)

    with patch("stateset_agents.cli._collect_import_status", side_effect=fake_collect_import_status):
        with patch(
            "stateset_agents.cli._collect_dependency_status",
            return_value=({"torch": True, "transformers": True, "datasets": True}, {}),
        ):
            result = runner.invoke(
                app,
                ["publish-check", "--strict", "--json-output"],
            )
            assert result.exit_code == 2
            payload = json.loads(result.stdout)
            assert payload["failed"] is True
            assert payload["publish_ready"] is False
            assert not payload["imports"]["required"]["stateset_agents.training"]


def test_cli_publish_check_warning_gate():
    """Test publish-check can fail on warnings."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        config = {"agent": {"model_name": "gpt2"}, "mystery_key": 123}
        json.dump(config, f)
        config_path = f.name

    try:
        result = runner.invoke(
            app,
            [
                "publish-check",
                "--config",
                config_path,
                "--fail-on-warnings",
                "--json-output",
            ],
        )
        assert result.exit_code == 2
        payload = json.loads(result.stdout)
        assert payload["failed"] is True
        assert payload["config"]["warnings"]
    finally:
        Path(config_path).unlink(missing_ok=True)


def test_cli_publish_check_help_available():
    """Publish-check command should be listed and callable."""
    result = runner.invoke(app, ["publish-check", "--help"])
    assert result.exit_code == 0
    assert "publish-check" in result.stdout.lower()


# ===========================
# Evaluate Command Tests
# ===========================


def test_cli_evaluate_requires_checkpoint():
    """Test evaluate command requires checkpoint path"""
    result = runner.invoke(app, ["evaluate", "--message", "Hello"])
    # Should handle missing checkpoint gracefully
    assert result.exit_code != 0 or "checkpoint" in result.stdout.lower()


def test_cli_evaluate_missing_checkpoint():
    """Test evaluate requires an existing checkpoint."""
    result = runner.invoke(app, ["evaluate", "--checkpoint", "/nonexistent/checkpoint"])
    assert result.exit_code != 0
    assert "not found" in result.stdout.lower()


def test_cli_evaluate_dry_run():
    """Test evaluate command in dry-run mode"""
    result = runner.invoke(app, ["evaluate", "--dry-run"])
    assert result.exit_code == 0
    assert "dry-run" in result.stdout.lower()


# ===========================
# Serve Command Tests
# ===========================


def test_cli_serve_dry_run():
    """Test serve command in dry-run mode"""
    result = runner.invoke(app, ["serve", "--dry-run"])
    assert result.exit_code == 0
    assert "dry-run" in result.stdout.lower()


def test_cli_serve_invalid_port():
    """Test serve rejects non-positive port values."""
    result = runner.invoke(app, ["serve", "--dry-run", "--port", "0"])
    assert result.exit_code != 0
    assert "port must be a positive integer." in result.stdout


def test_cli_evaluate_missing_checkpoint_path():
    """Test evaluate requires an existing checkpoint."""
    result = runner.invoke(app, ["evaluate", "--checkpoint", "/nonexistent/checkpoint"])
    assert result.exit_code != 0
    assert "not found" in result.stdout.lower()


def test_cli_init_requires_overwrite_for_existing_file(tmp_path):
    """Test init refuses to overwrite existing file unless requested."""
    cfg_path = tmp_path / "stateset_agents.yaml"
    cfg_path.write_text("existing", encoding="utf-8")
    result = runner.invoke(app, ["init", "--path", str(cfg_path)])
    assert result.exit_code != 0
    assert "already exists" in result.stdout.lower()

    result = runner.invoke(
        app, ["init", "--path", str(cfg_path), "--overwrite", "--format", "json"]
    )
    assert result.exit_code == 0
    assert "wrote starter config" in result.stdout.lower()


def test_cli_init_json_format():
    """Test init command supports JSON output format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = f"{tmpdir}/stateset_agents.json"
        result = runner.invoke(
            app, ["init-config", "--path", cfg_path, "--format", "json"]
        )
        assert result.exit_code == 0
        assert "Wrote starter config" in result.stdout
        loaded = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        assert loaded["agent"]["model_name"] == "gpt2"
        assert loaded["training"]["num_episodes"] == 5

def test_cli_init_qwen_preset_json(tmp_path):
    """Test init can scaffold the Qwen 3.5 starter preset."""
    cfg_path = tmp_path / "qwen3_5_0_8b.json"
    result = runner.invoke(
        app,
        [
            "init",
            "--preset",
            "qwen3-5-0-8b",
            "--path",
            str(cfg_path),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert loaded["model_name"] == "Qwen/Qwen3.5-0.8B-Base"
    assert loaded["task"] == "customer_service"
    assert loaded["trust_remote_code"] is True
    assert loaded["attn_implementation"] == "sdpa"



def test_cli_init_qwen_preset_custom_task(tmp_path):
    """Test init can scaffold the Qwen starter preset for another task."""
    cfg_path = tmp_path / "qwen3_5_sales.json"
    result = runner.invoke(
        app,
        [
            "init-config",
            "--preset",
            "qwen3-5-0-8b",
            "--task",
            "sales",
            "--path",
            str(cfg_path),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert loaded["task"] == "sales"
    assert "sales assistant" in loaded["system_prompt"].lower()


def test_cli_init_qwen_preset_memory_profile_json(tmp_path):
    """Test init can scaffold the low-memory Qwen starter preset."""
    cfg_path = tmp_path / "qwen3_5_memory.json"
    result = runner.invoke(
        app,
        [
            "init",
            "--preset",
            "qwen3-5-0-8b",
            "--starter-profile",
            "memory",
            "--path",
            str(cfg_path),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert loaded["starter_profile"] == "memory"
    assert loaded["use_4bit"] is True
    assert loaded["max_prompt_length"] == 768


def test_cli_init_kimi_preset_json(tmp_path):
    """Test init can scaffold the Kimi-K2.6 starter preset."""
    cfg_path = tmp_path / "kimi_k2_6.json"
    result = runner.invoke(
        app,
        [
            "init",
            "--preset",
            "kimi-k2-6",
            "--path",
            str(cfg_path),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert loaded["model_name"] == "moonshotai/Kimi-K2.6"
    assert loaded["task"] == "customer_service"
    assert loaded["trust_remote_code"] is True
    assert loaded["use_4bit"] is True
    assert loaded["attn_implementation"] == "sdpa"


def test_cli_init_kimi_preset_custom_task(tmp_path):
    """Test init can scaffold the Kimi starter preset for another task."""
    cfg_path = tmp_path / "kimi_k2_6_sales.json"
    result = runner.invoke(
        app,
        [
            "init-config",
            "--preset",
            "kimi-k2-6",
            "--task",
            "sales",
            "--path",
            str(cfg_path),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert loaded["task"] == "sales"
    assert "sales assistant" in loaded["system_prompt"].lower()


def test_cli_init_kimi_preset_memory_profile_json(tmp_path):
    """Test init can scaffold the low-memory Kimi starter preset."""
    cfg_path = tmp_path / "kimi_k2_6_memory.json"
    result = runner.invoke(
        app,
        [
            "init",
            "--preset",
            "kimi-k2-6",
            "--starter-profile",
            "memory",
            "--path",
            str(cfg_path),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert loaded["starter_profile"] == "memory"
    assert loaded["use_4bit"] is True
    assert loaded["max_prompt_length"] == 2048


def test_cli_init_gemma_preset_json(tmp_path):
    """Test init can scaffold the Gemma 4 31B starter preset."""
    cfg_path = tmp_path / "gemma4_31b.json"
    result = runner.invoke(
        app,
        [
            "init",
            "--preset",
            "gemma-4-31b",
            "--path",
            str(cfg_path),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert loaded["model_name"] == "google/gemma-4-31B-it"
    assert loaded["task"] == "customer_service"
    assert loaded["trust_remote_code"] is True
    assert loaded["use_4bit"] is True
    assert loaded["attn_implementation"] == "sdpa"


def test_cli_init_gemma_preset_memory_profile_json(tmp_path):
    """Test init can scaffold the low-memory Gemma starter preset."""
    cfg_path = tmp_path / "gemma4_memory.json"
    result = runner.invoke(
        app,
        [
            "init",
            "--preset",
            "gemma-4-31b",
            "--starter-profile",
            "memory",
            "--path",
            str(cfg_path),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert loaded["starter_profile"] == "memory"
    assert loaded["use_4bit"] is True
    assert loaded["max_prompt_length"] == 2048


def test_cli_init_invalid_preset():
    """Test init rejects unsupported preset names."""
    result = runner.invoke(app, ["init", "--preset", "unknown-preset"])
    assert result.exit_code != 0
    assert "unsupported preset" in result.stdout.lower()


def test_cli_init_rejects_starter_profile_for_default_preset():
    """Test init rejects starter profiles for the default non-Qwen preset."""
    result = runner.invoke(app, ["init", "--starter-profile", "memory"])
    assert result.exit_code != 0
    assert "only applies" in result.stdout.lower()


def test_cli_init_invalid_format():
    """Test init command rejects unsupported formats."""
    result = runner.invoke(app, ["init", "--format", "toml"])
    assert result.exit_code != 0
    assert "format must be yaml or json" in result.stdout.lower()


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


def test_cli_preflight_help_available():
    """Preflight command should be listed and callable."""
    result = runner.invoke(app, ["preflight", "--help"])
    assert result.exit_code == 0
    assert "preflight" in result.stdout.lower()


def test_cli_advanced_help_available():
    """Advanced command group should be discoverable."""
    result = runner.invoke(app, ["advanced", "--help"])
    assert result.exit_code == 0
    assert "advanced" in result.stdout.lower()


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
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        config = {
            "agent": {
                "model_name": "stub://custom",
                "use_stub_model": True,
                "stub_responses": ["Custom response 1", "Custom response 2"],
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
