import json
from pathlib import Path

import typer
from typer.main import get_group

from stateset_agents import cli as stateset_cli


def test_quickstart_cli_commands_exist() -> None:
    app: typer.Typer = stateset_cli.app
    group = get_group(app)
    # Commands referenced by QUICKSTART.md must remain available
    expected = {"version", "ingest", "improve", "serve", "train-remote"}
    assert expected.issubset(set(group.commands.keys()))


def test_train_remote_default_provider_is_modal() -> None:
    app: typer.Typer = stateset_cli.app
    group = get_group(app)
    cmd = group.commands["train-remote"]
    provider_params = [p for p in cmd.params if getattr(p, "name", "") == "provider"]
    assert provider_params, "provider option not found on train-remote"
    provider = provider_params[0]
    assert (
        provider.default == "modal"
    ), "Default --provider must be 'modal' to match docs and recommended backend"


def test_npm_package_test_script_is_js_only_and_typechecks() -> None:
    pkg_path = Path(__file__).resolve().parents[2] / "npm" / "package.json"
    assert pkg_path.exists(), f"Missing npm package.json at {pkg_path}"
    data = json.loads(pkg_path.read_text(encoding="utf-8"))
    scripts = data.get("scripts") or {}
    test_script = scripts.get("test") or ""
    # Safety: do not accidentally execute *.ts test files in node --test
    assert ".ts" not in test_script, "npm test script must not execute *.ts directly"
    # Contract: JS-only node --test plus a separate typecheck
    assert "node --test" in test_script, "npm test must use node --test"
    assert "typecheck" in test_script, "npm test must include a typecheck phase"
