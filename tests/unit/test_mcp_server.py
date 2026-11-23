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

# Windows CI spawns processes and imports the (heavy) reward stack far more
# slowly than Linux, and the live-protocol session pays that cost inside a
# single tool call. A 60s budget was enough until the reward registry grew;
# it then started timing out on the Windows runners only (e.g. run
# https://github.com/stateset/stateset-agents/actions/runs/32886471767).
# Scale the budget instead of skipping, so the protocol stays covered there.
_TIMEOUT_SCALE = 4 if sys.platform == "win32" else 1


def _timeout(seconds: float) -> float:
    """Per-request budget, widened on slow platforms."""
    return seconds * _TIMEOUT_SCALE


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
            timeout=_timeout(60),
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
        assert "workspace" in result.output.lower()

    def test_unsupported_transport_errors(self) -> None:
        result = runner.invoke(app, ["mcp", "--transport", "sse"])
        assert result.exit_code == 2
        assert "stdio" in result.output.lower()


class TestStudioMcpToolsResourcesAndPrompts:
    """Fast unit coverage of all 3D Weight Studio MCP tools, resources, and prompts."""

    def test_studio_lifecycle_tools_and_self_correcting_errors(
        self, tmp_path: Path
    ) -> None:
        from stateset_agents.mcp_server import (
            set_default_workspace,
            studio_apply_edit,
            studio_arena_compare,
            studio_auto_heal,
            studio_auto_repair,
            studio_capabilities_manifest,
            studio_discover_circuit,
            studio_eval_and_export,
            studio_export_artifact,
            studio_export_lora,
            studio_extract_steering,
            studio_inspect_model,
            studio_list_models,
            studio_manage_edits,
            studio_profile_sensitivity,
            studio_residual_stream,
            studio_run_inference,
            studio_sae_analyze,
            studio_save_session,
            studio_sessions,
        )

        prev_ws = set_default_workspace(tmp_path)
        try:
            assert Path(prev_ws) == tmp_path.resolve()

            models = studio_list_models()
            assert "error" not in models
            assert models["count"] >= 1
            model_id = models["models"][0]["model_id"]

            # Self-correcting error when an AI agent passes an unknown model_id
            bad = studio_inspect_model("does-not-exist-model")
            assert "error" in bad
            assert "hint" in bad
            assert isinstance(bad.get("available_models"), list)
            assert model_id in bad["available_models"]

            # Inspection + sensitivity + interpretability suite
            insp = studio_inspect_model(model_id)
            assert "error" not in insp
            assert len(insp["layers"]) >= 1

            sens = studio_profile_sensitivity(model_id, target_compression=0.30)
            assert "error" not in sens
            assert "recommended_recipe" in sens

            inf = studio_run_inference(
                model_id, "Customer asks for a refund on order #9910."
            )
            assert "error" not in inf
            assert "active" in inf

            circ = studio_discover_circuit(model_id, "Process return for order #9910")
            assert "error" not in circ
            assert circ["arc_count"] >= 0

            resid = studio_residual_stream(model_id, "Process return for order #9910")
            assert "error" not in resid
            assert len(resid["stations"]) >= 1

            sae = studio_sae_analyze(
                model_id, "Process return for order #9910", top_k=6
            )
            assert "error" not in sae
            assert len(sae["features"]) >= 1

            vec = studio_extract_steering(
                model_id,
                "Empathetic, policy-grounded refund confirmation with clear timeline.",
                "idk whatever contact support.",
                name="Empathy",
            )
            assert "error" not in vec
            assert "direction" in vec

            # Surgical edit + undo/redo/recipe lifecycle
            applied = studio_apply_edit(
                model_id,
                "scale",
                "layer:0",
                {"factor": 1.05},
                label="Gentle L0 scale",
            )
            assert "error" not in applied
            assert len(applied["history"]["edits"]) == 1

            undone = studio_manage_edits(model_id, action="undo")
            assert "error" not in undone
            assert len(undone["edits"]) == 0
            assert undone["can_redo"] is True

            redone = studio_manage_edits(model_id, action="redo")
            assert "error" not in redone
            assert len(redone["edits"]) == 1

            rec_res = studio_manage_edits(model_id, action="export_recipe")
            assert "error" not in rec_res
            recipe = rec_res["recipe"]
            assert len(recipe["operations"]) == 1

            reset_res = studio_manage_edits(model_id, action="reset")
            assert len(reset_res["edits"]) == 0

            replayed = studio_manage_edits(
                model_id, action="apply_recipe", recipe=recipe
            )
            assert "error" not in replayed
            assert len(replayed["history"]["edits"]) == 1

            # A/B Arena + Auto-Repair + Eval Gate
            arena = studio_arena_compare(
                model_id,
                "Refund order #9910 please",
                candidates=[
                    {
                        "label": "Mild scale",
                        "op": "scale",
                        "target": "layer:0",
                        "params": {"factor": 1.02},
                    }
                ],
                reward_fn="customer_support",
            )
            assert "error" not in arena
            assert len(arena["candidates"]) >= 2

            repaired = studio_auto_repair(model_id, reward_fn="customer_support")
            assert "error" not in repaired
            assert repaired["final_verdict"] in ("GO", "WARN", "BLOCK")

            gate_res = studio_eval_and_export(
                model_id, reward_fn="customer_support", max_cases=3
            )
            assert "error" not in gate_res
            assert gate_res["eval_gate"]["verdict"] in ("GO", "WARN", "BLOCK")

            # Export PEFT LoRA, GGUF, HF Model Card, and offline 3D HTML PR report
            lora = studio_export_lora(model_id, export_name="mcp-lora", rank=4, alpha=8)
            assert "error" not in lora

            gguf = studio_export_artifact(
                model_id, format="gguf", export_name="mcp-gguf", quant_type="Q8_0"
            )
            assert "error" not in gguf
            assert (tmp_path / gguf["gguf_file"]).exists()

            card = studio_export_artifact(
                model_id,
                format="hf_card",
                export_name="mcp-card",
                repo_id="stateset/mcp-demo",
                include_eval=False,
            )
            assert "error" not in card
            assert (tmp_path / card["model_card_file"]).exists()

            html = studio_export_artifact(
                model_id,
                format="html_report",
                export_name="mcp-html",
                include_eval=False,
            )
            assert "error" not in html
            assert Path(html["report_path"]).exists()
            assert html["bytes"] > 1000

            # Workbench sessions save -> list -> load
            saved = studio_save_session(
                "mcp-test-session", model_id, {"owner": "agent"}
            )
            assert "error" not in saved
            listed = studio_sessions(action="list")
            assert listed["count"] >= 1
            loaded = studio_sessions(action="load", session_name="mcp-test-session")
            assert "error" not in loaded

            # Autonomous Transcript-to-Surgery Self-Healing Loop
            healed = studio_auto_heal(
                model_id,
                reward_fn="customer_support",
                export_format="none",
            )
            assert "error" not in healed
            assert healed["eval_gate"]["verdict"] in ("GO", "WARN", "BLOCK")

            # Capability manifest
            caps = studio_capabilities_manifest()
            assert "steer_bake" in caps["edit_ops"]
            assert "auto_heal_from_logs" in caps["recommended_workflows"]
        finally:
            set_default_workspace(None)

    def test_mcp_prompts_return_actionable_playbooks(self) -> None:
        from stateset_agents.mcp_server import (
            prompt_auto_heal_model,
            prompt_compress_and_export_model,
            prompt_interpretability_audit,
            prompt_surgical_behavior_edit,
        )

        p1 = prompt_auto_heal_model("preset:deepseek-r1-1.5b", "logs/bad.jsonl")
        assert "studio_auto_heal" in p1 and "logs/bad.jsonl" in p1

        p2 = prompt_surgical_behavior_edit("preset:deepseek-r1-1.5b", "be empathetic")
        assert "studio_extract_steering" in p2 and "studio_arena_compare" in p2

        p3 = prompt_interpretability_audit("preset:deepseek-r1-1.5b", "Refund #123")
        assert "studio_discover_circuit" in p3 and "studio_sae_analyze" in p3

        p4 = prompt_compress_and_export_model("preset:deepseek-r1-1.5b", 0.4, "gguf")
        assert "studio_profile_sensitivity" in p4 and "studio_auto_repair" in p4


@pytest.mark.slow
class TestLiveProtocolSession:
    """End-to-end regression over the real MCP stdio protocol.

    Spawns ``python -m stateset_agents.cli mcp`` as a subprocess and drives
    it with the ``mcp`` SDK's own client session (initialize -> list_tools
    -> list_resources -> read_resource -> list_prompts -> get_prompt ->
    call_tool), the same path a real MCP client (Claude Code, Claude
    Desktop, Cursor, Antigravity) takes.
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
            args=["-m", "stateset_agents.cli", "mcp", "--workspace", str(tmp_path)],
            env=env,
            cwd=str(REPO_ROOT),
        )

        # Keep the child process off pytest's capture stream: AnyIO's stderr
        # copier can otherwise block the stdio protocol under capture.
        async with stdio_client(server_params, errlog=sys.__stderr__) as (read, write):
            async with ClientSession(read, write) as session:
                init_result = await asyncio.wait_for(
                    session.initialize(), timeout=_timeout(30)
                )
                assert init_result.serverInfo.name == "stateset-agents"

                tools_result = await asyncio.wait_for(
                    session.list_tools(), timeout=_timeout(30)
                )
                tool_by_name = {tool.name: tool for tool in tools_result.tools}
                expected_tools = {
                    "list_rewards",
                    "ingest_transcripts",
                    "grade_transcript",
                    "improve_run",
                    "improve_status",
                    "list_model_presets",
                    "dry_run_finetune",
                    "studio_list_models",
                    "studio_inspect_model",
                    "studio_profile_sensitivity",
                    "studio_run_inference",
                    "studio_discover_circuit",
                    "studio_residual_stream",
                    "studio_sae_analyze",
                    "studio_extract_steering",
                    "studio_apply_edit",
                    "studio_manage_edits",
                    "studio_arena_compare",
                    "studio_auto_repair",
                    "studio_eval_and_export",
                    "studio_export_lora",
                    "studio_export_artifact",
                    "studio_save_session",
                    "studio_sessions",
                    "studio_auto_heal",
                    "studio_redteam_audit",
                    "studio_model_tournament",
                    "studio_canary_gate",
                    "studio_autopilot_release",
                    "studio_live_serving_ab",
                    "studio_ingest_trace",
                    "studio_audit_reward_calibration",
                    "studio_query_traces",
                    "studio_query_rollouts",
                    "studio_prune_traces",
                    "studio_rl_telemetry",
                    "studio_rl_plan",
                    "studio_run_sandbox_pool",
                    "studio_query_rl_trajectories",
                }
                assert set(tool_by_name) == expected_tools
                assert len(tools_result.tools) == len(expected_tools)

                # Verify read-only vs mutating ToolAnnotations for client auto-approval
                assert tool_by_name["studio_list_models"].annotations is not None
                assert (
                    tool_by_name["studio_list_models"].annotations.readOnlyHint is True
                )
                assert tool_by_name["studio_apply_edit"].annotations is not None
                assert (
                    tool_by_name["studio_apply_edit"].annotations.readOnlyHint is False
                )
                assert (
                    tool_by_name["studio_apply_edit"].annotations.destructiveHint
                    is False
                )

                # Verify live MCP Resources
                resources_result = await asyncio.wait_for(
                    session.list_resources(), timeout=_timeout(30)
                )
                resource_uris = {str(r.uri) for r in resources_result.resources}
                assert resource_uris == {
                    "stateset://studio/capabilities",
                    "stateset://studio/models",
                    "stateset://studio/steering-presets",
                    "stateset://studio/scenarios",
                    "stateset://studio/sessions",
                    "stateset://studio/redteam-probes",
                    "stateset://studio/rollouts",
                    "stateset://studio/traces",
                }

                caps_res = await asyncio.wait_for(
                    session.read_resource("stateset://studio/capabilities"),
                    timeout=_timeout(30),
                )
                caps_json = json.loads(caps_res.contents[0].text)
                assert "steer_bake" in caps_json["edit_ops"]

                # Verify guided MCP Workflow Prompts
                prompts_result = await asyncio.wait_for(
                    session.list_prompts(), timeout=_timeout(30)
                )
                prompt_names = {p.name for p in prompts_result.prompts}
                assert prompt_names == {
                    "production_autopilot_release",
                    "auto_heal_model",
                    "redteam_and_immunize",
                    "model_tournament_and_merge",
                    "surgical_behavior_edit",
                    "interpretability_audit",
                    "compress_and_export_model",
                }

                prompt_resp = await asyncio.wait_for(
                    session.get_prompt(
                        "auto_heal_model",
                        {"model_id": "preset:deepseek-r1-1.5b"},
                    ),
                    timeout=_timeout(30),
                )
                assert len(prompt_resp.messages) >= 1

                rewards_result = await asyncio.wait_for(
                    session.call_tool("list_rewards", {}), timeout=_timeout(30)
                )
                assert rewards_result.isError is not True
                assert rewards_result.structuredContent is not None
                assert set(rewards_result.structuredContent["rewards"]) == {
                    "gsm8k",
                    "customer_support",
                    "tool_calling",
                    "nsr",
                }

                # Call Studio tools over real stdio protocol
                models_result = await asyncio.wait_for(
                    session.call_tool("studio_list_models", {}), timeout=_timeout(30)
                )
                assert models_result.isError is not True
                assert models_result.structuredContent is not None
                assert models_result.structuredContent["count"] >= 1

                grade_result = await asyncio.wait_for(
                    session.call_tool(
                        "grade_transcript",
                        {
                            "history_path": str(transcripts_dir / "session1.jsonl"),
                            "reward": "customer_support",
                        },
                    ),
                    timeout=_timeout(60),
                )
                assert grade_result.isError is not True, grade_result
                assert grade_result.structuredContent is not None
                assert "error" not in grade_result.structuredContent
                assert grade_result.structuredContent["assistant_turn_count"] == 2
