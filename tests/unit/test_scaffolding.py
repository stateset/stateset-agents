"""Unit tests for the starter-project scaffolding."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stateset_agents.scaffolding import (
    SCAFFOLD_TEMPLATES,
    list_templates,
    scaffold_project,
)


class TestTemplateRegistry:
    def test_known_templates_present(self) -> None:
        assert "customer-support" in SCAFFOLD_TEMPLATES
        assert "gsm8k-math" in SCAFFOLD_TEMPLATES
        assert "minimal" in SCAFFOLD_TEMPLATES

    def test_list_templates_sorted(self) -> None:
        names = [t.name for t in list_templates()]
        assert names == sorted(names)

    def test_every_template_has_common_files(self) -> None:
        for t in SCAFFOLD_TEMPLATES.values():
            assert ".gitignore" in t.files
            assert "requirements.txt" in t.files
            assert "README.md" in t.files

    def test_template_render_substitutes_name(self) -> None:
        t = SCAFFOLD_TEMPLATES["minimal"]
        rendered = t.render("my-cool-project")
        assert "my-cool-project" in rendered["README.md"]


class TestCustomerSupportTemplate:
    def test_scaffold_creates_expected_files(self, tmp_path: Path) -> None:
        out = tmp_path / "client_acme"
        created = scaffold_project("customer-support", out)
        expected = {
            "config.yaml",
            "scenarios.jsonl",
            "reward.py",
            "train.py",
            "eval.py",
            "serve.sh",
            "README.md",
            ".gitignore",
            "requirements.txt",
            ".stateset-agents-starter.json",
        }
        names = {p.name for p in created}
        assert expected.issubset(names)

    def test_scenarios_are_valid_jsonl(self, tmp_path: Path) -> None:
        out = tmp_path / "p"
        scaffold_project("customer-support", out)
        scenarios = [
            json.loads(line)
            for line in (out / "scenarios.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert len(scenarios) > 0
        for s in scenarios:
            assert "intent" in s
            assert "user_query" in s
            assert "must_acknowledge" in s

    def test_config_yaml_parses(self, tmp_path: Path) -> None:
        out = tmp_path / "p"
        scaffold_project("customer-support", out)
        import yaml

        cfg = yaml.safe_load((out / "config.yaml").read_text())
        assert "model" in cfg
        assert "training" in cfg
        assert cfg["training"]["algorithm"] == "gspo"
        # Spot-check the GSPO clip range is the tight default, not 0.2.
        assert cfg["training"]["clip_range_left"] < 0.01

    def test_serve_sh_is_executable(self, tmp_path: Path) -> None:
        out = tmp_path / "p"
        scaffold_project("customer-support", out)
        import stat

        mode = (out / "serve.sh").stat().st_mode
        assert mode & stat.S_IXUSR

    def test_marker_file_records_template(self, tmp_path: Path) -> None:
        out = tmp_path / "p"
        scaffold_project("customer-support", out, project_name="client-acme")
        marker = json.loads((out / ".stateset-agents-starter.json").read_text())
        assert marker["template"] == "customer-support"
        assert marker["project_name"] == "client-acme"

    def test_project_name_substitution(self, tmp_path: Path) -> None:
        out = tmp_path / "client_acme"
        scaffold_project("customer-support", out)
        readme = (out / "README.md").read_text()
        assert "client_acme" in readme

    def test_explicit_project_name_overrides_dirname(self, tmp_path: Path) -> None:
        out = tmp_path / "p"
        scaffold_project("customer-support", out, project_name="explicit-name")
        readme = (out / "README.md").read_text()
        assert "explicit-name" in readme


class TestGSM8KMathTemplate:
    def test_scaffold(self, tmp_path: Path) -> None:
        out = tmp_path / "math"
        created = scaffold_project("gsm8k-math", out)
        names = {p.name for p in created}
        assert "config.yaml" in names
        assert "train.py" in names
        assert "README.md" in names

    def test_train_script_imports_gsm8k_helpers(self, tmp_path: Path) -> None:
        out = tmp_path / "math"
        scaffold_project("gsm8k-math", out)
        train = (out / "train.py").read_text()
        assert "load_gsm8k" in train
        assert "GSM8KReward" in train


class TestToolCallingAgentTemplate:
    def test_scaffold(self, tmp_path: Path) -> None:
        out = tmp_path / "tools"
        created = scaffold_project("tool-calling-agent", out)
        names = {p.name for p in created}
        assert {
            "config.yaml",
            "tools.py",
            "scenarios.jsonl",
            "reward.py",
            "train.py",
        }.issubset(names)

    def test_scenarios_have_expected_tool(self, tmp_path: Path) -> None:
        out = tmp_path / "tools"
        scaffold_project("tool-calling-agent", out)
        scenarios = [
            json.loads(line)
            for line in (out / "scenarios.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert len(scenarios) > 0
        for s in scenarios:
            assert "expected_tool" in s
            assert "expected_params" in s
            assert "user_query" in s

    def test_tools_py_exports_sample_tools(self, tmp_path: Path) -> None:
        out = tmp_path / "tools"
        scaffold_project("tool-calling-agent", out)
        body = (out / "tools.py").read_text()
        # tools.py imports the bundled tools and exposes SAMPLE_TOOLS as the union
        # of the bundled set + user-supplied CUSTOM_TOOLS.
        assert "SAMPLE_TOOLS" in body
        assert "from stateset_agents.data.tool_calling_bench import" in body
        assert "CUSTOM_TOOLS" in body

    def test_config_has_three_reward_weights(self, tmp_path: Path) -> None:
        out = tmp_path / "tools"
        scaffold_project("tool-calling-agent", out)
        import yaml

        cfg = yaml.safe_load((out / "config.yaml").read_text())
        r = cfg["reward"]
        # Weights should sum to ~1.0.
        assert (
            abs(
                r["tool_selection_weight"]
                + r["param_correctness_weight"]
                + r["outcome_weight"]
                - 1.0
            )
            < 1e-6
        )


class TestMinimalTemplate:
    def test_scaffold(self, tmp_path: Path) -> None:
        out = tmp_path / "min"
        scaffold_project("minimal", out)
        assert (out / "config.yaml").exists()
        assert (out / "train.py").exists()


class TestClientNameCustomization:
    def test_slugifies_into_output_dir(self, tmp_path: Path) -> None:
        out = tmp_path / "p"
        scaffold_project(
            "customer-support",
            out,
            client_name="Acme Corp",
        )
        config = (out / "config.yaml").read_text()
        # Default "outputs/customer_support_v1" should be replaced.
        assert "outputs/customer_support_v1" not in config
        assert "outputs/acme_corp_v1" in config

    def test_adds_wandb_project(self, tmp_path: Path) -> None:
        out = tmp_path / "p"
        scaffold_project("customer-support", out, client_name="Acme")
        config = (out / "config.yaml").read_text()
        assert "wandb_project: acme" in config

    def test_marker_records_client_name(self, tmp_path: Path) -> None:
        out = tmp_path / "p"
        scaffold_project("customer-support", out, client_name="Stateset Inc")
        marker = json.loads((out / ".stateset-agents-starter.json").read_text())
        assert marker["client_name"] == "Stateset Inc"

    def test_tool_template_also_customized(self, tmp_path: Path) -> None:
        out = tmp_path / "tools"
        scaffold_project("tool-calling-agent", out, client_name="Foo Inc")
        config = (out / "config.yaml").read_text()
        assert "outputs/foo_inc_v1" in config
        assert "outputs/tool_agent_v1" not in config

    def test_no_client_name_leaves_defaults(self, tmp_path: Path) -> None:
        out = tmp_path / "p"
        scaffold_project("customer-support", out)
        config = (out / "config.yaml").read_text()
        assert "outputs/customer_support_v1" in config
        assert "wandb_project" not in config

    def test_special_chars_in_client_name_slugified(self, tmp_path: Path) -> None:
        out = tmp_path / "p"
        scaffold_project("customer-support", out, client_name="A&B! Co.")
        config = (out / "config.yaml").read_text()
        assert "a_b__co" in config or "a_b_co" in config


class TestErrorHandling:
    def test_unknown_template_raises_keyerror(self, tmp_path: Path) -> None:
        with pytest.raises(KeyError, match="Unknown template"):
            scaffold_project("not-a-real-template", tmp_path / "p")

    def test_non_empty_directory_raises_by_default(self, tmp_path: Path) -> None:
        out = tmp_path / "existing"
        out.mkdir()
        (out / "something.txt").write_text("hi")
        with pytest.raises(FileExistsError):
            scaffold_project("minimal", out)

    def test_force_overwrites(self, tmp_path: Path) -> None:
        out = tmp_path / "existing"
        out.mkdir()
        (out / "something.txt").write_text("hi")
        created = scaffold_project("minimal", out, force=True)
        assert len(created) > 0
        # Previous file is preserved (we don't delete what's already there);
        # we only refuse to scaffold *into* a non-empty dir without --force.
        assert (out / "something.txt").exists()

    def test_empty_existing_directory_is_fine(self, tmp_path: Path) -> None:
        out = tmp_path / "empty"
        out.mkdir()
        created = scaffold_project("minimal", out)
        assert len(created) > 0


class TestCLI:
    def test_cli_list_does_not_require_output(self) -> None:
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "stateset_agents.cli", "starter", "list"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "customer-support" in result.stdout
        assert "gsm8k-math" in result.stdout

    def test_cli_missing_output_for_scaffold_exits_2(self) -> None:
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "stateset_agents.cli",
                "starter",
                "customer-support",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 2

    def test_cli_scaffold_succeeds(self, tmp_path: Path) -> None:
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "stateset_agents.cli",
                "starter",
                "minimal",
                str(tmp_path / "p"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert (tmp_path / "p" / "config.yaml").exists()
