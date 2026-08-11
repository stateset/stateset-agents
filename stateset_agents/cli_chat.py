"""Interactive chat REPL subcommands for the StateSet Agents CLI.

Split out of stateset_agents/cli.py. Each command attaches to the parent
Typer app exported by cli; helpers _echo, _load_config, etc. are
re-bound locally for readability. Helpers that tests patch on
stateset_agents.cli (_collect_dependency_status, _collect_import_status)
are looked up via late binding through the _cli module reference so the
patches still propagate.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import typer

from stateset_agents import cli as _cli
from stateset_agents.cli import app

_echo = _cli._echo
_load_config = _cli._load_config
_coerce_positive_int = _cli._coerce_positive_int


@app.command("chat")
def chat(
    model: str = typer.Option(
        "stub://chat",
        "--model",
        "-m",
        help="HF model name or stub://<id> for the in-process REPL.",
    ),
    checkpoint: str | None = typer.Option(
        None,
        "--checkpoint",
        "-c",
        help="Path to a LoRA adapter to load on top of --model.",
    ),
    system_prompt: str | None = typer.Option(
        None,
        "--system",
        help="Optional system prompt prepended to every conversation.",
    ),
    max_new_tokens: int = typer.Option(
        256,
        "--max-new-tokens",
        help="Generation length cap per response.",
    ),
    history: str | None = typer.Option(
        None,
        "--history",
        help="Path to a JSONL file to APPEND each turn (one JSON object per line). "
        "Capture interesting conversations to replay or grade later with "
        "`make grade-transcript`.",
    ),
    replay: str | None = typer.Option(
        None,
        "--replay",
        help="Path to a JSONL transcript to replay as initial conversation context. "
        "Useful for resuming a debugging session.",
    ),
    grade: str | None = typer.Option(
        None,
        "--grade",
        help="Score each assistant turn live with the named reward function. "
        "Options: gsm8k, customer_support, tool_calling. "
        "Mismatches between intuition and score surface reward-function bugs.",
    ),
) -> None:
    """Open an interactive REPL against an in-process agent.

    The fastest way to sanity-check a fine-tune locally without spinning up
    the FastAPI gateway. Loads the agent in-process and lets you talk to it
    line by line. Type ``/reset`` to clear conversation history, ``/quit`` or
    Ctrl+D to exit.

    Examples:

        # Stub agent — no GPU, instant
        stateset-agents chat

        # Real HF model
        stateset-agents chat --model Qwen/Qwen3.5-0.8B

        # Fine-tuned LoRA adapter
        stateset-agents chat --model Qwen/Qwen3.5-0.8B --checkpoint outputs/acme_v1

        # With a system prompt
        stateset-agents chat --system "You are a helpful customer support agent."
    """
    import asyncio

    from stateset_agents.core.agent import MultiTurnAgent
    from stateset_agents.core.agent_config import AgentConfig

    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            print(f"Checkpoint path not found: {checkpoint}", file=sys.stderr)
            raise typer.Exit(code=2)

    is_stub = model.startswith("stub://") or model == "gpt2"
    config = AgentConfig(
        model_name=model,
        max_new_tokens=max_new_tokens,
        system_prompt=system_prompt,
        use_stub_model=is_stub,
        peft_path=checkpoint if (checkpoint and not is_stub) else None,
    )

    _echo(f"Initializing agent (model={model}, stub={is_stub})…")
    agent = MultiTurnAgent(config)
    try:
        asyncio.run(agent.initialize())
    except Exception as e:  # noqa: BLE001 — surface the error to the user
        print(f"Failed to initialize agent: {type(e).__name__}: {e}", file=sys.stderr)
        raise typer.Exit(code=2) from e

    # Set up live grader if requested.
    from stateset_agents.core.reward_base import RewardFunction

    live_reward: RewardFunction | None = None
    if grade:
        try:
            if grade == "gsm8k":
                from stateset_agents.data.gsm8k import GSM8KReward

                live_reward = GSM8KReward()
            elif grade == "customer_support":
                from stateset_agents.data.customer_support_bench import (
                    SupportRewardComposite,
                )

                live_reward = SupportRewardComposite()
            elif grade == "tool_calling":
                from stateset_agents.data.tool_calling_bench import ToolCallReward

                live_reward = ToolCallReward()
            else:
                print(
                    f"Unknown --grade reward: {grade!r}. "
                    f"Options: gsm8k, customer_support, tool_calling.",
                    file=sys.stderr,
                )
                raise typer.Exit(code=2)
        except ImportError as e:
            print(f"Failed to load grader: {e}", file=sys.stderr)
            raise typer.Exit(code=2) from e

    # Wire up readline for up-arrow recall + persistent input history.
    # The "input history" (one file per user) is separate from the conversation
    # transcript (`--history` flag). Only sets up if readline is available
    # (Linux/macOS by default; Windows ships pyreadline3 separately).
    _readline_history_path: Path | None = None
    try:
        import readline

        # Use XDG_STATE_HOME if set, else ~/.local/state, else ~ as fallback.
        state_dir = os.environ.get("XDG_STATE_HOME")
        if state_dir:
            base = Path(state_dir) / "stateset-agents"
        else:
            base = Path.home() / ".local" / "state" / "stateset-agents"
        base.mkdir(parents=True, exist_ok=True)
        _readline_history_path = base / "chat_input_history"
        if _readline_history_path.exists():
            try:
                readline.read_history_file(str(_readline_history_path))
            except OSError:
                pass
        readline.set_history_length(1000)
    except ImportError:
        pass  # readline unavailable — REPL still works, just no up-arrow

    _echo("")
    _echo("=" * 60)
    _echo("StateSet Agents — Interactive Chat")
    _echo("Type /reset to clear history, /quit or Ctrl+D to exit.")
    if _readline_history_path:
        _echo(f"Input history: {_readline_history_path} (up-arrow recalls)")
    if history:
        _echo(f"Appending each turn to: {history}")
    if grade:
        _echo(f"Live grading enabled: {grade}")
    _echo("=" * 60)

    messages: list[dict[str, str]] = []
    history_path = Path(history).expanduser() if history else None
    if history_path:
        history_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional replay — preload conversation from a saved transcript.
    if replay:
        replay_path = Path(replay).expanduser()
        if not replay_path.exists():
            print(f"Replay path not found: {replay}", file=sys.stderr)
            raise typer.Exit(code=2)
        for line in replay_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "role" in entry and "content" in entry:
                messages.append({"role": entry["role"], "content": entry["content"]})
        _echo(f"Loaded {len(messages)} turn(s) from {replay}")

    def _append_history(role: str, content: str) -> None:
        if history_path is None:
            return
        with history_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps({"role": role, "content": content}, ensure_ascii=False)
                + "\n"
            )

    async def _send(user_text: str) -> str:
        messages.append({"role": "user", "content": user_text})
        _append_history("user", user_text)
        response = await agent.generate_response(messages)
        messages.append({"role": "assistant", "content": response})
        _append_history("assistant", response)
        return response

    while True:
        try:
            user_input = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input in ("/quit", "/exit"):
            break
        if user_input == "/reset":
            messages = []
            _echo("(conversation reset)")
            continue

        try:
            response = asyncio.run(_send(user_input))
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️  generation failed: {e}", file=sys.stderr)
            continue

        print(f"agent> {response}")

        # Live grading: score this turn against the reward function.
        if live_reward is not None:
            try:
                from stateset_agents.core.trajectory import ConversationTurn

                # Score using the full conversation so far (matches training-time eval).
                turns_for_reward = [
                    ConversationTurn(role=m["role"], content=m["content"])
                    for m in messages
                ]
                reward_result = asyncio.run(
                    live_reward.compute_reward(turns_for_reward, context=None)
                )
                score = float(reward_result.score)
                # Color-code: green for ≥0.5, yellow for 0.1–0.5, red for <0.1.
                if score >= 0.5:
                    marker = "✅"
                elif score >= 0.1:
                    marker = "⚠️ "
                else:
                    marker = "❌"
                print(f"  {marker} reward[{grade}] = {score:.3f}")
            except (
                Exception
            ) as e:  # noqa: BLE001 — grading failure shouldn't kill the REPL
                print(f"  ⚠️  grading failed: {e}", file=sys.stderr)

    # Persist input history for the next chat session.
    if _readline_history_path is not None:
        try:
            import readline

            readline.write_history_file(str(_readline_history_path))
        except (ImportError, OSError):
            pass

    _echo("Bye.")
