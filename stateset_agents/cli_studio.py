"""Weight Studio, Self-Heal, Red-Team, Tournament, Canary, Autopilot & Live Ops CLI subcommands.

Split out of ``stateset_agents/cli.py`` to keep the top-level CLI modular.
Each command attaches directly to ``stateset_agents.cli.app`` at import time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from stateset_agents import cli as _cli
from stateset_agents.cli import app

_echo = _cli._echo


@app.command("studio")
def weight_studio_command(
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Host interface to bind the 3D Weight Studio server.",
    ),
    port: int = typer.Option(
        8765,
        "--port",
        min=1,
        max=65535,
        help="Port for the 3D Weight Studio browser server.",
    ),
    workspace: Path = typer.Option(
        Path("."),
        "--workspace",
        "-w",
        help="Workspace directory containing outputs/ and checkpoints/.",
    ),
    open_browser: bool = typer.Option(
        False,
        "--open",
        help="Open the 3D Weight Studio in your default browser.",
    ),
) -> None:
    """Launch the interactive 3D Weight Studio in your browser."""
    try:
        from stateset_agents.weight_studio.__main__ import main as studio_main

        argv = ["--host", host, "--port", str(port), "--workspace", str(workspace)]
        if open_browser:
            argv.append("--open")
        studio_main(argv)
    except (ImportError, OSError, RuntimeError) as error:
        _echo(f"Could not launch 3D Weight Studio: {error}", err=True)
        raise typer.Exit(code=2) from error


@app.command("heal")
def heal_command(
    model: str = typer.Option(
        "preset:deepseek-r1-1.5b",
        "--model",
        "-m",
        help="Model ID (e.g. local:outputs/my_run or preset:deepseek-r1-1.5b).",
    ),
    transcripts: Path | None = typer.Option(
        None,
        "--transcripts",
        "-t",
        help="Path to failing production transcripts JSONL (uses built-in demo logs if omitted).",
    ),
    reward: str = typer.Option(
        "customer_support",
        "--reward",
        "-r",
        help="StateSet reward scorer: customer_support, tool_calling, gsm8k, or composite.",
    ),
    export_format: str = typer.Option(
        "lora",
        "--export-format",
        "-f",
        help="Verified export artifact: lora, safetensors, gguf, hf_card, or none.",
    ),
    output_name: str = typer.Option(
        "",
        "--output-name",
        "-o",
        help="Export directory name under outputs/weight_studio_exports/.",
    ),
    lora_rank: int = typer.Option(
        16,
        "--lora-rank",
        min=1,
        max=128,
        help="Target LoRA rank r when --export-format lora.",
    ),
    workspace: Path = typer.Option(
        Path("."),
        "--workspace",
        "-w",
        help="Workspace root directory.",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Print structured JSON summary.",
    ),
) -> None:
    """Autonomously diagnose failing transcripts, bake CAA weight surgery, eval-gate & export."""
    try:
        from stateset_agents.weight_studio.surgery_loop import run_autonomous_heal

        report = run_autonomous_heal(
            model,
            workspace_root=workspace,
            transcripts_path=str(transcripts) if transcripts is not None else None,
            reward_fn=reward,
            export_format=export_format,
            output_name=output_name,
            lora_rank=lora_rank,
        )
        if as_json:
            _echo(json.dumps(report, indent=2))
            return

        diag = report["diagnosis"]
        sv = report["steering_vector"]
        gate = report["eval_gate"]
        _echo(f"StateSet Autonomous Surgery Heal — {model}")
        _echo(
            f"  Diagnosis : {diag['failing_count']}/{diag['total_transcripts']} failing "
            f"(dominant={diag['dominant_failure_mode']}, mean={diag['mean_score']:.3f})"
        )
        _echo(
            f"  Surgery   : Baked CAA '{sv['name']}' on L{sv['target_layer']:02d} "
            f"(α={sv['chosen_strength']:+.2f}, sep={sv['peak_separation']})"
        )
        _echo(
            f"  Eval Gate : {gate['verdict']} ({gate['case_count']} cases, "
            f"Δcomp={gate['deltas']['composite_delta']:+.4f})"
        )
        if report["exported"] and report["export_manifest"]:
            _echo(
                f"  Exported  : {report['export_format']} -> {report['summary_path']}"
            )
            if report["html_report_path"]:
                _echo(f"  PR Report : {report['html_report_path']}")
        elif report["export_blocked_reason"]:
            _echo(f"  Blocked   : {report['export_blocked_reason']}", err=True)
            raise typer.Exit(code=1)
    except (OSError, ValueError, KeyError, TypeError) as error:
        _echo(f"Autonomous heal failed: {error}", err=True)
        raise typer.Exit(code=2) from error


@app.command("redteam")
def redteam_command(
    model: str = typer.Option(
        "preset:deepseek-r1-1.5b",
        "--model",
        "-m",
        help="Model ID (e.g. local:outputs/my_run or preset:deepseek-r1-1.5b).",
    ),
    immunize: bool = typer.Option(
        False,
        "--immunize",
        help="Apply one-click CAA Surgical Immunization to harden vulnerable layers.",
    ),
    probes: Path | None = typer.Option(
        None,
        "--probes",
        "-p",
        help="Optional path to custom red-team attack probes JSONL.",
    ),
    export_format: str = typer.Option(
        "none",
        "--export-format",
        "-f",
        help="Optional artifact format when --immunize is set: none, lora, safetensors, gguf, hf_card.",
    ),
    output_name: str = typer.Option(
        "",
        "--output-name",
        "-o",
        help="Export directory name under outputs/weight_studio_exports/.",
    ),
    live_endpoint: str = typer.Option(
        "",
        "--live-endpoint",
        help="Optional OpenAI-compatible vLLM/Ollama endpoint to probe live generations.",
    ),
    live_model: str = typer.Option(
        "",
        "--live-model",
        help="Model name on --live-endpoint when probing live generations.",
    ),
    workspace: Path = typer.Option(
        Path("."),
        "--workspace",
        "-w",
        help="Workspace root directory.",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Print structured JSON summary.",
    ),
) -> None:
    """Run Adversarial Red-Team Safety Audit and optional Surgical Immunization."""
    try:
        from stateset_agents.weight_studio import StudioRuntime

        rt = StudioRuntime(workspace_root=workspace)
        report = rt.redteam.run_redteam_audit(
            model,
            probes_path=str(probes) if probes is not None else None,
            immunize=immunize,
            export_format=export_format,
            output_name=output_name,
            live_endpoint_url=live_endpoint,
            live_model=live_model,
        )
        if as_json:
            _echo(json.dumps(report, indent=2))
            return

        _echo(f"StateSet Red-Team Safety Audit — {model}")
        _echo(
            f"  Verdict   : {report['verdict']} "
            f"(ASR={report['attack_success_rate']*100:.1f}%, "
            f"Pass={report['safety_pass_rate']*100:.1f}%)"
        )
        if report["immunization"]:
            imm = report["immunization"]
            _echo(
                f"  Immunized : L{imm['target_layer']:02d} (α={imm['chosen_strength']:+.2f}), "
                f"ASR {imm['asr_before']*100:.1f}% → {imm['asr_after']*100:.1f}%, "
                f"Benign Gate={imm['benign_eval_verdict']}"
            )
            if report["report_path"]:
                _echo(f"  PR Report : {report['report_path']}")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        _echo(f"Red-team audit failed: {error}", err=True)
        raise typer.Exit(code=2) from error


@app.command("tournament")
def tournament_command(
    models: str = typer.Option(
        "",
        "--models",
        "-m",
        help="Comma-separated model IDs (defaults to top discovered models).",
    ),
    reward: str = typer.Option(
        "customer_support",
        "--reward",
        "-r",
        help="Reward scorer: customer_support, tool_calling, gsm8k, or composite.",
    ),
    include_redteam: bool = typer.Option(
        True,
        "--redteam/--no-redteam",
        help="Include adversarial Red-Team safety pass rate in Pareto fitness.",
    ),
    synthesize_merge: bool = typer.Option(
        True,
        "--synthesize-merge/--no-synthesize-merge",
        help="Synthesize & benchmark TIES/DARE/SLERP surgical merges of Top-2 contenders.",
    ),
    export_format: str = typer.Option(
        "none",
        "--export-format",
        "-f",
        help="Optional champion artifact format: none, lora, safetensors, gguf, hf_card.",
    ),
    output_name: str = typer.Option(
        "",
        "--output-name",
        "-o",
        help="Export directory name under outputs/weight_studio_exports/.",
    ),
    workspace: Path = typer.Option(
        Path("."),
        "--workspace",
        "-w",
        help="Workspace root directory.",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Print structured JSON leaderboard and champion summary.",
    ),
) -> None:
    """Run Multi-Model Pareto Tournament & Evolutionary TIES/DARE/SLERP Merge Lab."""
    try:
        from stateset_agents.weight_studio import StudioRuntime

        rt = StudioRuntime(workspace_root=workspace)
        model_list = [m.strip() for m in models.split(",") if m.strip()] or None
        report = rt.tournament.run_tournament(
            model_ids=model_list,
            reward_fn=reward,
            include_redteam=include_redteam,
            synthesize_merge=synthesize_merge,
            export_format=export_format,
            output_name=output_name,
        )
        if as_json:
            _echo(json.dumps(report, indent=2))
            return

        champ = report["champion"]
        _echo(
            f"StateSet Multi-Model Tournament ({report['contender_count']} contenders)"
        )
        _echo(
            f"  Champion  : {champ['label']} "
            f"(fitness={champ['fitness']:.4f}, reward={champ['reward_score']:.4f}, "
            f"safety={champ['safety_pass_rate']*100:.1f}%)"
        )
        if report["promoted_merge"]:
            pm = report["promoted_merge"]
            _echo(
                f"  Merge Win : {pm['merge_method'].upper()} "
                f"(Δfitness={pm['delta_vs_best_parent']:+.4f})"
            )
        if report["report_path"]:
            _echo(f"  PR Report : {report['report_path']}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        _echo(f"Tournament failed: {error}", err=True)
        raise typer.Exit(code=2) from error


@app.command("canary")
def canary_command(
    model: str = typer.Option(
        "preset:deepseek-r1-1.5b",
        "--model",
        "-m",
        help="Model ID to evaluate on shadow production traffic.",
    ),
    stage: str = typer.Option(
        "canary_10",
        "--stage",
        "-s",
        help="Rollout stage: shadow, canary_10, or canary_50.",
    ),
    traffic: Path | None = typer.Option(
        None,
        "--traffic",
        "-t",
        help="Optional shadow traffic JSONL file.",
    ),
    reward: str = typer.Option(
        "composite",
        "--reward",
        "-r",
        help="Reward scorer: composite, customer_support, tool_calling, or gsm8k.",
    ),
    live_endpoint: str = typer.Option(
        "",
        "--live-endpoint",
        help="Optional live vLLM/Ollama endpoint for real-token canary shadow A/B.",
    ),
    baseline_model: str = typer.Option(
        "",
        "--baseline-model",
        help="Baseline served model name when --live-endpoint is set.",
    ),
    candidate_model: str = typer.Option(
        "",
        "--candidate-model",
        help="Candidate served model/LoRA name when --live-endpoint is set.",
    ),
    workspace: Path = typer.Option(
        Path("."),
        "--workspace",
        "-w",
        help="Workspace root directory.",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Print structured JSON rollout decision and ledger entry.",
    ),
) -> None:
    """Run Shadow-Replay Canary Promotion Gate (PROMOTE / HOLD_CANARY / ROLLBACK)."""
    try:
        from stateset_agents.weight_studio import StudioRuntime

        rt = StudioRuntime(workspace_root=workspace)
        report = rt.canary.run_canary_gate(
            model,
            stage=stage,
            traffic_path=str(traffic) if traffic is not None else None,
            reward_fn=reward,
            live_endpoint_url=live_endpoint,
            live_baseline_model=baseline_model,
            live_candidate_model=candidate_model,
        )
        if as_json:
            _echo(json.dumps(report, indent=2))
            return

        pr = report["paired_record"]
        _echo(f"StateSet Canary Shadow Gate — {model} [{stage}]")
        _echo(
            f"  Decision  : {report['decision']} "
            f"(traffic={report['recommended_traffic_pct']}%, "
            f"W/T/L={pr['wins']}/{pr['ties']}/{pr['losses']}, Δ={pr['mean_delta']:+.4f})"
        )
        _echo(f"  Rationale : {report['rationale']}")
        _echo(f"  Ledger    : {report['ledger_path']}")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        _echo(f"Canary gate failed: {error}", err=True)
        raise typer.Exit(code=2) from error


@app.command("autopilot")
def autopilot_command(
    model: str = typer.Option(
        "preset:deepseek-r1-1.5b",
        "--model",
        "-m",
        help="Model ID to heal, immunize, canary-gate, and export.",
    ),
    transcripts: Path | None = typer.Option(
        None,
        "--transcripts",
        "-t",
        help="Optional failing production transcripts JSONL.",
    ),
    reward: str = typer.Option(
        "customer_support",
        "--reward",
        "-r",
        help="Reward scorer: customer_support, tool_calling, gsm8k, or composite.",
    ),
    export_format: str = typer.Option(
        "lora",
        "--export-format",
        "-f",
        help="Verified export format: lora, safetensors, gguf, hf_card, or none.",
    ),
    output_name: str = typer.Option(
        "",
        "--output-name",
        "-o",
        help="Export bundle name under outputs/weight_studio_exports/.",
    ),
    workspace: Path = typer.Option(
        Path("."),
        "--workspace",
        "-w",
        help="Workspace root directory.",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Print structured JSON autopilot release dossier summary.",
    ),
) -> None:
    """Run Full Production Autopilot: Auto-Heal -> Red-Team Immunize -> Canary Gate -> Export."""
    try:
        from stateset_agents.weight_studio import StudioRuntime

        rt = StudioRuntime(workspace_root=workspace)
        report = rt.canary.run_autopilot_release(
            model,
            transcripts_path=str(transcripts) if transcripts is not None else None,
            reward_fn=reward,
            export_format=export_format,
            output_name=output_name,
        )
        if as_json:
            _echo(json.dumps(report, indent=2))
            return

        _echo(f"StateSet Production Autopilot — {model}")
        _echo(
            f"  Decision  : {report['decision']} "
            f"(rollout={report['rollout_id']}, traffic={report['recommended_traffic_pct']}%)"
        )
        _echo(
            f"  Safety    : {report['redteam']['verdict']} "
            f"(ASR={report['redteam']['attack_success_rate']*100:.1f}%)"
        )
        if report["exported"] and report["dossier_path"]:
            _echo(f"  Dossier   : {report['dossier_path']}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        _echo(f"Autopilot release failed: {error}", err=True)
        raise typer.Exit(code=2) from error


@app.command("traces")
def traces_command(
    ingest: Path | None = typer.Option(
        None,
        "--ingest",
        "-i",
        help="Ingest a JSON or JSONL file (OTLP resourceSpans, OpenAI chat, or prompt/response rows).",
    ),
    audit: bool = typer.Option(
        False,
        "--audit",
        help="Run Reward Calibration & Reward-Hacking audit against human-labeled traces.",
    ),
    auto_heal: bool = typer.Option(
        False,
        "--auto-heal",
        help="Trigger Autonomous Self-Heal directly from collected failing live traces.",
    ),
    model: str = typer.Option(
        "preset:deepseek-r1-1.5b",
        "--model",
        "-m",
        help="Model ID to heal when --auto-heal is set.",
    ),
    reward: str = typer.Option(
        "customer_support",
        "--reward",
        "-r",
        help="Reward scorer: customer_support, tool_calling, gsm8k, or composite.",
    ),
    query: bool = typer.Option(
        False,
        "--query",
        help="Query indexed SQLite trace storage using --failed-only / --source / --search.",
    ),
    failed_only: bool = typer.Option(
        False,
        "--failed-only",
        help="Filter SQLite trace queries to failing traces only.",
    ),
    source: str = typer.Option(
        "",
        "--source",
        help="Filter SQLite trace queries by source (e.g. otlp, webhook, openai_chat).",
    ),
    search: str = typer.Option(
        "",
        "--search",
        help="Substring search over prompt, response, or trace_id in SQLite storage.",
    ),
    prune: int | None = typer.Option(
        None,
        "--prune",
        min=0,
        help="Prune SQLite trace store to keep only the latest N traces.",
    ),
    judge_mode: str = typer.Option(
        "heuristic",
        "--judge-mode",
        help="Ingest scoring mode: heuristic, llm_judge, or hybrid.",
    ),
    judge_endpoint: str = typer.Option(
        "",
        "--judge-endpoint",
        help="OpenAI-compatible LLM-as-a-Judge endpoint URL when --judge-mode is llm_judge/hybrid.",
    ),
    judge_model: str = typer.Option(
        "",
        "--judge-model",
        help="Judge model name when --judge-mode is llm_judge/hybrid.",
    ),
    workspace: Path = typer.Option(
        Path("."),
        "--workspace",
        "-w",
        help="Workspace root directory.",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Print structured JSON output.",
    ),
) -> None:
    """Ingest live OTLP/webhook traces, query/prune SQLite store, audit calibration, or auto-heal."""
    try:
        from stateset_agents.weight_studio import PluggableRewardJudge, StudioRuntime

        rt = StudioRuntime(workspace_root=workspace)
        out: dict[str, Any] = {}

        if ingest is not None:
            raw_text = ingest.read_text(encoding="utf-8").strip()
            if not raw_text:
                raise ValueError(f"Trace file is empty: {ingest}")
            try:
                parsed = json.loads(raw_text)
                payload: dict[str, Any] = (
                    parsed if isinstance(parsed, dict) else {"traces": parsed}
                )
            except json.JSONDecodeError:
                rows = [
                    json.loads(line) for line in raw_text.splitlines() if line.strip()
                ]
                payload = {"traces": rows}
            judge = (
                PluggableRewardJudge(
                    mode=judge_mode,
                    judge_endpoint_url=judge_endpoint,
                    judge_model=judge_model,
                )
                if judge_mode != "heuristic"
                else None
            )
            out["ingest"] = rt.traces.ingest_payload(
                payload, reward_fn=reward, judge=judge
            )

        if prune is not None:
            deleted = rt.traces.prune_traces(keep_latest=prune)
            out["prune"] = {"deleted": deleted, "keep_latest": prune}

        if query or failed_only or source or search:
            matched = rt.traces.query_traces(
                failed_only=failed_only,
                source=source or None,
                search=search or None,
                limit=50,
            )
            out["query"] = {"count": len(matched), "traces": matched}

        if audit:
            out["calibration"] = rt.traces.audit_reward_calibration(reward_fn=reward)

        if auto_heal:
            failing = rt.traces.list_traces(limit=100, failing_only=True)
            jsonl_blob = "\n".join(
                json.dumps({"prompt": r["prompt"], "response": r["response"]})
                for r in failing
            )
            out["auto_heal"] = rt.healer.run_healing_loop(
                model,
                transcripts_jsonl=jsonl_blob or None,
                reward_fn=reward,
                export_format="none",
            )

        out["summary"] = rt.traces.summary()

        if as_json:
            _echo(json.dumps(out, indent=2))
            return

        summary = out["summary"]
        _echo(
            f"StateSet Live Traces — {summary['total_traces']} total, "
            f"{summary['failing_traces']} failing, {summary['human_labeled_traces']} human-labeled "
            f"(mean proxy={summary['mean_proxy_score']:.3f})"
        )
        if "ingest" in out:
            ing = out["ingest"]
            _echo(
                f"  Ingested    : {ing['ingested_count']} trace(s) ({ing['failing_count']} failing)"
            )
        if "calibration" in out:
            cal = out["calibration"]
            _echo(
                f"  Calibration : {cal['verdict']} "
                f"(n={cal.get('sample_count', 0)}, "
                f"ρ={cal.get('spearman_rho', 'n/a')}, "
                f"MAE={cal.get('mae', 'n/a')}, "
                f"hacking={cal.get('hacking_rate', 0.0) * 100:.1f}%)"
            )
        if "auto_heal" in out:
            heal = out["auto_heal"]
            sv = heal.get("steering_vector") or {}
            gate = heal.get("eval_gate") or {}
            _echo(
                f"  Auto-Heal   : {gate.get('verdict', 'GO')} "
                f"(L{int(sv.get('target_layer', 0)):02d}, "
                f"α={float(sv.get('chosen_strength', 0.0)):+.2f})"
            )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        _echo(f"Live traces command failed: {error}", err=True)
        raise typer.Exit(code=2) from error


@app.command("serve-ab")
def serve_ab_command(
    endpoint: str = typer.Option(
        "http://127.0.0.1:8000/v1",
        "--endpoint",
        "-e",
        help="OpenAI-compatible vLLM/Ollama/SGLang base URL.",
    ),
    baseline: str = typer.Option(
        ...,
        "--baseline",
        "-b",
        help="Baseline model name served by the endpoint.",
    ),
    candidate: str = typer.Option(
        ...,
        "--candidate",
        "-c",
        help="Candidate model or hot-loaded LoRA adapter name.",
    ),
    candidate_endpoint: str = typer.Option(
        "",
        "--candidate-endpoint",
        help="Optional separate endpoint URL for the candidate model.",
    ),
    load_lora_name: str = typer.Option(
        "",
        "--load-lora-name",
        help="Optional vLLM LoRA adapter name to hot-load before running A/B.",
    ),
    load_lora_path: str = typer.Option(
        "",
        "--load-lora-path",
        help="Workspace-confined path to Diff-as-LoRA adapter directory to hot-load.",
    ),
    reward: str = typer.Option(
        "customer_support",
        "--reward",
        "-r",
        help="StateSet reward scorer: customer_support, tool_calling, gsm8k, or composite.",
    ),
    api_key: str = typer.Option(
        "",
        "--api-key",
        help="Optional Bearer API key for the inference endpoint.",
    ),
    workspace: Path = typer.Option(
        Path("."),
        "--workspace",
        "-w",
        help="Workspace root directory.",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Print structured JSON shadow A/B report.",
    ),
) -> None:
    """Hot-load a LoRA adapter on live vLLM and run real-token paired shadow A/B comparison."""
    try:
        from stateset_agents.weight_studio import StudioRuntime

        rt = StudioRuntime(workspace_root=workspace)
        lora_result: dict[str, Any] | None = None
        if load_lora_name and load_lora_path:
            lora_result = rt.serving.load_lora_adapter(
                candidate_endpoint or endpoint,
                load_lora_name,
                load_lora_path,
                api_key=api_key,
            )

        report = rt.serving.compare_live_endpoints(
            endpoint,
            baseline,
            candidate,
            reward_fn=reward,
            api_key=api_key,
            candidate_endpoint_url=candidate_endpoint,
        )
        if lora_result is not None:
            report["lora_hot_load"] = lora_result

        if as_json:
            _echo(json.dumps(report, indent=2))
            return

        pr = report.get("paired_record") or {}
        lat = report.get("latency_ms") or {}
        _echo(f"StateSet Live Serving Shadow A/B — {baseline} vs {candidate}")
        _echo(
            f"  Decision  : {report['decision']} "
            f"(Δreward={float(pr.get('mean_delta', 0.0)):+.4f}, "
            f"W/T/L={pr.get('wins', 0)}/{pr.get('ties', 0)}/{pr.get('losses', 0)})"
        )
        _echo(
            f"  Latency   : base p50={lat.get('baseline_p50', 0)}ms / "
            f"cand p50={lat.get('candidate_p50', 0)}ms "
            f"(p95={lat.get('candidate_p95', 0)}ms)"
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        PermissionError,
    ) as error:
        _echo(f"Live serving A/B failed: {error}", err=True)
        raise typer.Exit(code=2) from error


@app.command("rl")
def rl_blueprint_command(
    model: str = typer.Option(
        "Qwen/Qwen3-8B",
        "--model",
        "-m",
        help="Open-weight model ID for the RL blueprint.",
    ),
    algorithm: str = typer.Option(
        "gspo",
        "--algorithm",
        "-a",
        help="RL algorithm: gspo, grpo, dapo, vapo, or gepo.",
    ),
    cloud: str = typer.Option(
        "local",
        "--cloud",
        help="Execution target: local, runpod, modal, or k8s.",
    ),
    gpus: int = typer.Option(
        1,
        "--gpus",
        min=1,
        help="Number of GPUs to allocate.",
    ),
    chunk_size: int = typer.Option(
        256,
        "--chunk-size",
        min=16,
        help="Token chunk size for memory-efficient chunked LM-head RL loss.",
    ),
    blueprint: Path | None = typer.Option(
        None,
        "--blueprint",
        "-b",
        help="Optional YAML/JSON RL blueprint file.",
    ),
    warm_start: bool = typer.Option(
        False,
        "--warm-start",
        help="Execute Step-0 Surgical Warm-Start (CAA steering bake) before RL.",
    ),
    curriculum_from_traces: bool = typer.Option(
        False,
        "--curriculum-from-traces",
        help="Cluster failing SQLite OTLP traces into a balanced RL training curriculum JSONL.",
    ),
    run_shootout: bool = typer.Option(
        False,
        "--run-shootout",
        help="Run the live chunked-loss numerical parity, VRAM & disaggregated RL shootout.",
    ),
    workspace: Path = typer.Option(
        Path("."),
        "--workspace",
        "-w",
        help="Workspace root directory.",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Print structured JSON launch plan and diagnostics.",
    ),
) -> None:
    """Build a 1-command Open-Weight RL launch plan, trace curriculum, or Surgical Warm-Start."""
    try:
        from stateset_agents.training.rl_blueprint import (
            RLBlueprintConfig,
            build_cloud_launch_plan,
            run_surgical_warm_start,
        )
        from stateset_agents.training.rl_flywheel_guard import (
            ProductionTraceCurriculumBuilder,
        )

        cfg = (
            RLBlueprintConfig.from_file(blueprint)
            if blueprint is not None
            else RLBlueprintConfig(
                model=model,
                algorithm=algorithm,
                cloud=cloud,
                gpus=gpus,
                chunked_loss_chunk_size=chunk_size,
                surgical_warm_start=warm_start,
            )
        )
        plan = build_cloud_launch_plan(cfg, workspace_root=workspace)

        if curriculum_from_traces:
            plan["curriculum"] = ProductionTraceCurriculumBuilder(
                workspace
            ).build_curriculum()

        if warm_start:
            plan["warm_start_result"] = run_surgical_warm_start(
                cfg, workspace_root=workspace
            )

        if run_shootout:
            import asyncio

            from stateset_agents.training.disaggregated_rl import (
                DisaggregatedRLConfig,
                DisaggregatedRLRuntime,
            )

            async def _quick_async() -> dict[str, Any]:
                drt = DisaggregatedRLRuntime(
                    DisaggregatedRLConfig(
                        min_batch_size=4,
                        max_batch_size=8,
                        dynamic_sampling=True,
                        num_rollout_workers=2,
                    )
                )

                async def _prod(_w: int, _v: int) -> list[dict[str, Any]]:
                    return [
                        {
                            "reward": 1.0,
                            "token_count": 32,
                            "sampler_log_probs": (-0.2,),
                        },
                        {
                            "reward": 0.1,
                            "token_count": 32,
                            "sampler_log_probs": (-0.6,),
                        },
                    ]

                async def _learn(batch: list[Any], _s: int) -> dict[str, Any]:
                    return {"batch_size": len(batch)}

                res = await drt.run_steps(
                    num_steps=3, producer_fn=_prod, learner_step_fn=_learn
                )
                return dict(res["metrics"])

            plan["shootout"] = {
                "disaggregated_async_rl": asyncio.run(_quick_async()),
                "vram_estimate": plan["vram_estimate"],
            }

        if as_json:
            _echo(json.dumps(plan, indent=2))
            return

        vram = plan["vram_estimate"]
        _echo(
            f"StateSet Open-Weight RL Blueprint — {cfg.model} [{cfg.algorithm.upper()} @ {cfg.cloud}]"
        )
        _echo(f"  Plan SHA  : {plan['plan_sha256_prefix']}")
        _echo(
            f"  VRAM Save : {vram['full_logits_gb']} GB -> {vram['chunked_logits_gb']} GB "
            f"({vram['peak_logit_vram_saved_pct']}% logit VRAM saved, chunk={cfg.chunked_loss_chunk_size})"
        )
        _echo(f"  Dispatch  : {plan['dispatch']['command']}")
        if "curriculum" in plan:
            cur = plan["curriculum"]
            _echo(
                f"  Curriculum: {cur['episode_count']} episodes -> {cur['curriculum_path']}"
            )
        if "warm_start_result" in plan:
            ws_res = plan["warm_start_result"]
            _echo(
                f"  WarmStart : L{ws_res['target_layer']} (α={ws_res['chosen_strength']}) "
                f"Eval={ws_res['eval_verdict']}"
            )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        _echo(f"RL blueprint failed: {error}", err=True)
        raise typer.Exit(code=2) from error
