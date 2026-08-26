"""Unit tests for the StateSet Agents MCP server (``stateset_agents.mcp_server``).

Skips cleanly when the optional ``mcp`` extra is not installed (CI's [dev]
extra does not include it) — see ``pytest.importorskip`` below. When ``mcp``
*is* installed, exercises each tool's underlying plain function directly
(FastMCP tools are plain functions registered onto the server, so they are
callable/testable without an MCP client), plus improve_run parity with the
CLI and the CLI's missing-dependency behavior.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

mcp = pytest.importorskip("mcp", reason="optional 'mcp' extra not installed")

from stateset_agents.cli import app  # noqa: E402

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[2]

BAD_TURN = "idk"


def _write_transcript(path: Path, good: bool, order_id: str) -> None:
    turn = (
        f"I would be happy to help you with a refund for order {order_id} right away."
        if good
        else BAD_TURN
    )
    rows = [
        {"role": "user", "content": f"I want a refund for order {order_id}"},
        {"role": "assistant", "content": turn},
        {"role": "user", "content": f"Thanks, anything else about order {order_id}?"},
        {"role": "assistant", "content": turn},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _make_transcripts_dir(tmp_path: Path) -> Path:
    d = tmp_path / "transcripts"
    d.mkdir()
    _write_transcript(d / "session1.jsonl", good=True, order_id="1234")
    _write_transcript(d / "session2.jsonl", good=True, order_id="5678")
    _write_transcript(d / "session3.jsonl", good=False, order_id="9999")
    return d


class TestCreateServer:
    def test_create_server_builds_fastmcp_instance(self) -> None:
        from stateset_agents.mcp_server import create_server

        server = create_server()
        assert server.name == "stateset-agents"


class TestListRewards:
    def test_returns_known_rewards(self) -> None:
        from stateset_agents.mcp_server import list_rewards

        result = list_rewards()
        assert result == {
            "rewards": ["gsm8k", "customer_support", "tool_calling", "nsr"]
        }


class TestListModelPresets:
    def test_returns_preset_names_and_fields(self) -> None:
        from examples.model_presets import list_preset_names
        from stateset_agents.mcp_server import list_model_presets

        result = list_model_presets()
        names = {p["name"] for p in result["presets"]}
        assert names == set(list_preset_names())
        for preset in result["presets"]:
            assert "model_id" in preset
            assert "learning_rate" in preset


class TestIngestTranscripts:
    def test_ingest_openai_log(self, tmp_path: Path) -> None:
        from stateset_agents.mcp_server import ingest_transcripts

        openai_log = tmp_path / "logs.jsonl"
        conversation = {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ]
        }
        openai_log.write_text(json.dumps(conversation) + "\n", encoding="utf-8")

        out_dir = tmp_path / "out"
        result = ingest_transcripts(str(openai_log), "openai", str(out_dir))

        assert result["conversation_count"] == 1
        assert result["turn_count"] == 2
        assert (out_dir / "conversation_0.jsonl").exists()

    def test_unsupported_format_returns_error(self, tmp_path: Path) -> None:
        from stateset_agents.mcp_server import ingest_transcripts

        result = ingest_transcripts(str(tmp_path / "x.jsonl"), "bogus", str(tmp_path))
        assert "error" in result

    def test_missing_input_returns_error(self, tmp_path: Path) -> None:
        from stateset_agents.mcp_server import ingest_transcripts

        result = ingest_transcripts(
            str(tmp_path / "does-not-exist.jsonl"), "openai", str(tmp_path)
        )
        assert "error" in result


class TestGradeTranscript:
    async def test_grades_transcript(self, tmp_path: Path) -> None:
        from stateset_agents.mcp_server import grade_transcript

        transcripts_dir = _make_transcripts_dir(tmp_path)
        result = await grade_transcript(
            str(transcripts_dir / "session1.jsonl"), "customer_support"
        )
        assert "error" not in result
        assert result["assistant_turn_count"] == 2
        assert 0.0 <= result["mean_score"] <= 1.0
        assert isinstance(result["breakdown"], dict)

    async def test_unknown_reward_returns_error(self, tmp_path: Path) -> None:
        from stateset_agents.mcp_server import grade_transcript

        transcripts_dir = _make_transcripts_dir(tmp_path)
        result = await grade_transcript(
            str(transcripts_dir / "session1.jsonl"), "not-a-real-reward"
        )
        assert "error" in result

    async def test_missing_history_returns_error(self, tmp_path: Path) -> None:
        from stateset_agents.mcp_server import grade_transcript

        result = await grade_transcript(
            str(tmp_path / "nope.jsonl"), "customer_support"
        )
        assert "error" in result

    async def test_runs_fine_from_inside_a_running_event_loop(
        self, tmp_path: Path
    ) -> None:
        """Regression: the tool must not call ``asyncio.run`` on the caller's
        loop. This test itself runs inside pytest-asyncio's event loop —
        the same situation FastMCP's anyio loop puts the tool in — so a
        naive ``asyncio.run(...)`` inside ``grade_transcript`` would raise
        ``RuntimeError: asyncio.run() cannot be called from a running event
        loop`` instead of returning a structured error result.
        """
        from stateset_agents.mcp_server import grade_transcript

        transcripts_dir = _make_transcripts_dir(tmp_path)
        result = await grade_transcript(
            str(transcripts_dir / "session1.jsonl"), "customer_support"
        )
        assert "error" not in result, result


class TestImproveRun:
    def test_matches_cli_output(self, tmp_path: Path) -> None:
        # Deliberately sync (not `async def`): this test also drives the
        # CLI via `CliRunner.invoke`, which calls `cli_improve.run_improve`
        # -> `asyncio.run` directly (unchanged sync CLI behavior). Awaiting
        # `improve_run` from inside pytest-asyncio's own running loop would
        # put *this* call on that loop's thread too, and the CLI's inner
        # `asyncio.run` would then hit the very
        # "cannot be called from a running event loop" bug this test
        # suite is guarding against — for the CLI path, which is out of
        # scope for that fix (see module docstring). `asyncio.run` here
        # gives `improve_run` (and its internal `asyncio.to_thread` hop) a
        # fresh loop that exits before `runner.invoke` runs.
        import asyncio as _asyncio

        from stateset_agents.mcp_server import improve_run

        transcripts_dir = _make_transcripts_dir(tmp_path)
        mcp_output_dir = tmp_path / "mcp_improved"
        cli_output_dir = tmp_path / "cli_improved"

        mcp_summary = _asyncio.run(
            improve_run(
                transcripts_dir=str(transcripts_dir),
                reward="customer_support",
                output_dir=str(mcp_output_dir),
                threshold=0.7,
                format="transcripts",
            )
        )
        assert "error" not in mcp_summary

        result = runner.invoke(
            app,
            [
                "improve",
                "run",
                "--transcripts",
                str(transcripts_dir),
                "--reward",
                "customer_support",
                "--output",
                str(cli_output_dir),
                "--threshold",
                "0.7",
            ],
        )
        assert result.exit_code == 0, result.output

        cli_summary = json.loads(
            (cli_output_dir / "improve_summary.json").read_text(encoding="utf-8")
        )

        assert mcp_summary["mean_score"] == cli_summary["mean_score"]
        assert mcp_summary["curated_count"] == cli_summary["curated_count"]
        assert mcp_summary["transcript_count"] == cli_summary["transcript_count"]
        assert (mcp_output_dir / "curated.jsonl").exists()

    async def test_missing_reward_returns_error(self, tmp_path: Path) -> None:
        from stateset_agents.mcp_server import improve_run

        transcripts_dir = _make_transcripts_dir(tmp_path)
        result = await improve_run(
            transcripts_dir=str(transcripts_dir),
            reward="",
            output_dir=str(tmp_path / "out"),
        )
        assert "error" in result

    async def test_unknown_reward_returns_error(self, tmp_path: Path) -> None:
        from stateset_agents.mcp_server import improve_run

        transcripts_dir = _make_transcripts_dir(tmp_path)
        result = await improve_run(
            transcripts_dir=str(transcripts_dir),
            reward="bogus",
            output_dir=str(tmp_path / "out"),
        )
        assert "error" in result


class TestImproveStatus:
    async def test_status_after_run(self, tmp_path: Path) -> None:
        from stateset_agents.mcp_server import improve_run, improve_status

        transcripts_dir = _make_transcripts_dir(tmp_path)
        output_dir = tmp_path / "improved"
        await improve_run(
            transcripts_dir=str(transcripts_dir),
            reward="customer_support",
            output_dir=str(output_dir),
        )

        status = improve_status(str(output_dir))
        assert "error" not in status
        assert status["reward"] == "customer_support"

    def test_status_without_prior_run_returns_error(self, tmp_path: Path) -> None:
        from stateset_agents.mcp_server import improve_status

        status = improve_status(str(tmp_path / "never-ran"))
        assert "error" in status


class TestDryRunFinetune:
    def test_unknown_preset_returns_error(self) -> None:
        from stateset_agents.mcp_server import dry_run_finetune

        result = dry_run_finetune("not-a-real-preset")
        assert "error" in result

    def test_runs_dry_run_for_known_preset(self) -> None:
        from examples.model_presets import list_preset_names
        from stateset_agents.mcp_server import dry_run_finetune

        preset_name = list_preset_names()[0]
        result = dry_run_finetune(preset_name)
        assert "error" not in result, result
        assert result["model_preset"] == preset_name
        assert "config" in result


class TestMissingDependency:
    def test_helpful_error_without_mcp_installed(self, tmp_path: Path) -> None:
        """Simulate the 'mcp' extra not being installed via PYTHONPATH shadowing.

        Puts an empty package directory named ``mcp`` earlier on
        ``sys.path`` than the real installation so ``import mcp.server.fastmcp``
        fails the same way it would in an environment without the extra,
        then verifies ``create_server()`` raises a clear, actionable error.
        """
        fake_mcp_root = tmp_path / "fake_site_packages"
        fake_mcp_pkg = fake_mcp_root / "mcp"
        fake_mcp_pkg.mkdir(parents=True)
        (fake_mcp_pkg / "__init__.py").write_text(
            "raise ImportError('no fastmcp here')\n", encoding="utf-8"
        )

        script = (
            "import sys; "
            f"sys.path.insert(0, {str(fake_mcp_root)!r}); "
            f"sys.path.insert(0, {str(REPO_ROOT)!r}); "
            "from stateset_agents.mcp_server import create_server; "
            "create_server()"
        )
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
            check=False,
            env=env,
        )
        assert proc.returncode != 0
        assert "pip install stateset-agents[mcp]" in proc.stderr


class TestCliRegistration:
    def test_mcp_shows_in_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0, result.output
        assert "mcp" in result.output

    def test_mcp_subcommand_help(self) -> None:
        result = runner.invoke(app, ["mcp", "--help"])
        assert result.exit_code == 0, result.output
        assert "transport" in result.output.lower()

    def test_unsupported_transport_errors(self) -> None:
        result = runner.invoke(app, ["mcp", "--transport", "sse"])
        assert result.exit_code == 2
        assert "stdio" in result.output.lower()


@pytest.mark.slow
class TestLiveProtocolSession:
    """End-to-end regression over the real MCP stdio protocol.

    Spawns ``python -m stateset_agents.cli mcp`` as a subprocess and drives
    it with the ``mcp`` SDK's own client session (initialize -> list_tools
    -> call_tool), the same path a real MCP client (Claude Code, Claude
    Desktop, another agent) takes. This is the only test in this module
    that actually exercises FastMCP's request dispatch.
    """

    async def test_initialize_list_and_call_tools_over_stdio(
        self, tmp_path: Path
    ) -> None:
        await self._exercise_live_protocol(tmp_path)

    async def _exercise_live_protocol(self, tmp_path: Path) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        transcripts_dir = _make_transcripts_dir(tmp_path)

        env = dict(os.environ)
        env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "stateset_agents.cli", "mcp"],
            env=env,
            cwd=str(REPO_ROOT),
        )

        # Keep the child process off pytest's capture stream: AnyIO's stderr
        # copier can otherwise block the stdio protocol under capture.
        async with stdio_client(server_params, errlog=sys.__stderr__) as (read, write):
            async with ClientSession(read, write) as session:
                init_result = await asyncio.wait_for(session.initialize(), timeout=30)
                assert init_result.serverInfo.name == "stateset-agents"

                tools_result = await asyncio.wait_for(session.list_tools(), timeout=30)
                tool_names = {tool.name for tool in tools_result.tools}
                assert tool_names == {
                    "list_rewards",
                    "ingest_transcripts",
                    "grade_transcript",
                    "improve_run",
                    "improve_status",
                    "list_model_presets",
                    "dry_run_finetune",
                }
                assert len(tools_result.tools) == 7

                rewards_result = await asyncio.wait_for(
                    session.call_tool("list_rewards", {}), timeout=30
                )
                assert rewards_result.isError is not True
                assert rewards_result.structuredContent is not None
                assert set(rewards_result.structuredContent["rewards"]) == {
                    "gsm8k",
                    "customer_support",
                    "tool_calling",
                    "nsr",
                }

                grade_result = await asyncio.wait_for(
                    session.call_tool(
                        "grade_transcript",
                        {
                            "history_path": str(transcripts_dir / "session1.jsonl"),
                            "reward": "customer_support",
                        },
                    ),
                    timeout=60,
                )
                assert grade_result.isError is not True, grade_result
                assert grade_result.structuredContent is not None
                assert "error" not in grade_result.structuredContent
                assert grade_result.structuredContent["assistant_turn_count"] == 2
