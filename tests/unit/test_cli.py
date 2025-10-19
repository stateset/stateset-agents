from typer.testing import CliRunner

from stateset_agents import __version__
from stateset_agents.cli import app

runner = CliRunner()


def test_cli_version_outputs_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "stateset-agents version" in result.stdout
    assert __version__ in result.stdout


def test_cli_stub_training_runs_without_dependencies():
    result = runner.invoke(app, ["train", "--stub"])
    assert result.exit_code == 0
    assert "Stub agent conversation:" in result.stdout
