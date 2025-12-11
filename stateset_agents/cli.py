import importlib
import sys
import typing as t
from dataclasses import dataclass

import typer

app = typer.Typer(add_completion=False, help="StateSet Agents CLI")


def _echo(s: str) -> None:
    typer.echo(s)


@app.callback()
def main_callback() -> None:
    """StateSet Agents command-line interface."""
    # No-op. Subcommands implement functionality.
    return None


@app.command()
def version() -> None:
    """Show installed version and basic environment info."""
    try:
        from stateset_agents import __version__
    except Exception:
        __version__ = "unknown"

    _echo(f"stateset-agents version: {__version__}")
    _echo(f"python: {sys.version.split()[0]}")


@app.command()
def train(
    config: t.Optional[str] = typer.Option(
        None,
        help="Path to a training config file (YAML/JSON).",
    ),
    episodes: int = typer.Option(
        0,
        help="Override number of episodes.",
    ),
    save: t.Optional[str] = typer.Option(
        None,
        help="Optional checkpoint directory to save the trained agent.",
    ),
    dry_run: bool = typer.Option(
        True,
        help="If true, only validates environment and shows guidance.",
    ),
    stub: bool = typer.Option(
        False,
        "--stub",
        help="Run a lightweight stub demonstration without downloading models.",
    ),
) -> None:
    """Guide or launch training (lightweight)."""
    try:
        from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
        from stateset_agents.core.environment import ConversationEnvironment
        from stateset_agents.core.reward import (
            CompositeReward,
            HelpfulnessReward,
            SafetyReward,
        )
    except Exception as e:
        _echo("Core agent modules unavailable. Install the package with required extras.")
        _echo(f"Details: {e}")
        raise typer.Exit(code=2)

    train_fn = None
    if dry_run and not config and not stub:
        _echo("Dry-run: environment looks OK. To run a full example:")
        _echo("  python examples/quick_start.py")
        _echo("Or use TRL GRPO (if installed):")
        _echo("  python examples/train_with_trl_grpo.py")
        return

    if not stub:
        try:
            from stateset_agents.training.train import train as train_fn  # type: ignore
        except Exception as e:
            _echo("Training components unavailable. Falling back to stub mode.")
            _echo(f"Details: {e}")
            stub = True

    # Load config if provided
    cfg: t.Dict[str, t.Any] = {}
    if config:
        try:
            if config.endswith((".yaml", ".yml")):
                import yaml  # type: ignore

                with open(config, "r") as f:
                    cfg = yaml.safe_load(f) or {}
            else:
                import json

                with open(config, "r") as f:
                    cfg = json.load(f)
        except Exception as e:
            _echo(f"Failed to load config {config}: {e}")
            raise typer.Exit(code=2)

    agent_cfg = cfg.get("agent", {})
    env_cfg = cfg.get("environment", {})
    train_cfg = cfg.get("training", {})

    # Build agent
    ac = AgentConfig(
        model_name=agent_cfg.get(
            "model_name", "stub://demo" if stub else "gpt2"
        ),
        max_new_tokens=agent_cfg.get("max_new_tokens", 64),
        temperature=agent_cfg.get("temperature", 0.7),
        use_stub_model=agent_cfg.get("use_stub_model", stub),
        stub_responses=agent_cfg.get(
            "stub_responses",
            [
                "Stub backend ready. Install training extras for full GRPO",
                "Running in offline stub mode.",
            ],
        )
        if stub
        else None,
    )
    agent = MultiTurnAgent(ac)

    if stub:
        import asyncio

        async def _demo() -> None:
            await agent.initialize()
            history = [
                {
                    "role": "user",
                    "content": "Hi there, can you help me troubleshoot an issue?",
                }
            ]
            reply = await agent.generate_response(history)
            _echo("Stub agent conversation:")
            _echo(f"  user: {history[-1]['content']}")
            _echo(f"  assistant: {reply}")

        asyncio.run(_demo())
        _echo("Stub demonstration complete. Install training extras for full GRPO runs.")
        raise typer.Exit(code=0)

    # Build environment
    if env_cfg.get("type", "conversation") != "conversation":
        _echo("Only conversation environment supported by CLI quick train.")
        raise typer.Exit(code=2)
    scenarios = env_cfg.get("scenarios") or [
        {
            "id": "demo",
            "topic": "general_help",
            "context": "Demo",
            "user_responses": ["Thanks, tell me more.", "Interesting, go on."],
        }
    ]
    environment = ConversationEnvironment(
        scenarios=scenarios, max_turns=train_cfg.get("max_turns", 3)
    )

    # Build reward
    reward_fn = CompositeReward(
        [
            HelpfulnessReward(weight=0.6),
            SafetyReward(weight=0.4),
        ]
    )

    # Episodes override
    num_episodes = episodes or train_cfg.get("num_episodes", 2)

    # Run training
    import asyncio

    if train_fn is None:
        _echo("Training helper unavailable. Install 'stateset-agents[training]' to run GRPO training.")
        raise typer.Exit(code=2)

    async def _run():
        await agent.initialize()
        trained = await train_fn(
            agent=agent,
            environment=environment,
            reward_fn=reward_fn,
            num_episodes=num_episodes,
            profile="balanced",
            save_path=save or None,
        )
        return trained

    try:
        asyncio.run(_run())
    except Exception as e:
        _echo(f"Training failed: {e}")
        raise typer.Exit(code=2)
    _echo("Training complete.")
    raise typer.Exit(code=0)


@app.command()
def evaluate(
    dataset: t.Optional[str] = typer.Option(
        None, help="Evaluation dataset path (optional)."
    ),
    dry_run: bool = typer.Option(True, help="Validate environment only."),
) -> None:
    """Evaluate a trained agent (scaffold)."""
    try:
        import stateset_agents.training as _  # noqa: F401
    except Exception as e:
        _echo("Evaluation components unavailable. Install dev/extras and try again.")
        _echo(f"Details: {e}")
        raise typer.Exit(code=2)

    _echo("Evaluation CLI not fully wired. See examples and docs.")
    raise typer.Exit(code=0)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8001, help="Bind port"),
) -> None:
    """Run the Ultimate GRPO Service (if FastAPI/uvicorn available)."""
    try:
        # Import via package namespace to resolve relative imports in module.
        svc = importlib.import_module("stateset_agents.api.ultimate_grpo_service")
    except Exception as e:
        _echo("API service unavailable. Install 'api' extras (fastapi, uvicorn).")
        _echo(f"Details: {e}")
        raise typer.Exit(code=2)

    # If module exposes 'app' and FastAPI is present, run uvicorn.
    app_obj = getattr(svc, "app", None)
    if app_obj is None:
        _echo("Service module loaded but FastAPI not available.")
        raise typer.Exit(code=2)

    try:
        import uvicorn  # type: ignore
    except Exception:
        _echo("uvicorn not installed. Try: pip install 'stateset-agents[api]'")
        raise typer.Exit(code=2)

    _echo("Starting StateSet Agents service...")
    uvicorn.run(app_obj, host=host, port=port, log_level="info")


@app.command()
def doctor() -> None:
    """Check environment and dependencies for common issues."""
    import importlib
    import platform

    _echo("StateSet Agents - Environment Doctor")
    _echo(f"python: {sys.version.split()[0]} ({platform.platform()})")

    def _check(mod: str) -> None:
        try:
            importlib.import_module(mod)
            _echo(f"✅ {mod} available")
        except Exception as e:
            _echo(f"⚠️  {mod} missing: {e}")

    for mod in ["torch", "transformers", "datasets"]:
        _check(mod)

    # Optional extras
    for mod in ["aiohttp", "fastapi", "uvicorn", "trl", "bitsandbytes"]:
        try:
            importlib.import_module(mod)
            _echo(f"• optional: {mod} installed")
        except Exception:
            pass

    # GPU info
    try:
        import torch  # type: ignore

        cuda = torch.cuda.is_available()
        bf16 = torch.cuda.is_bf16_supported() if cuda else False
        _echo(f"CUDA available: {cuda}; bfloat16: {bf16}")
        if cuda:
            _echo(
                f"GPU count: {torch.cuda.device_count()}; name: {torch.cuda.get_device_name(0)}"
            )
    except Exception:
        pass

    _echo("Done.")


@app.command()


@app.command()
def evaluate(
    checkpoint: str = typer.Option(..., help="Path to a saved checkpoint directory"),
    message: str = typer.Option("Hello", help="Message to evaluate"),
) -> None:
    """Load a saved agent checkpoint and run a single evaluation message."""
    try:
        from stateset_agents.core.agent import load_agent_from_checkpoint
    except Exception as e:
        _echo(f"Failed to import loader: {e}")
        raise typer.Exit(code=2)

    import asyncio

    async def _run():
        agent = await load_agent_from_checkpoint(checkpoint, load_model=True)
        resp = await agent.generate_response([{"role": "user", "content": message}])
        return resp

    try:
        resp = asyncio.run(_run())
        _echo(f"Response: {resp}")
    except Exception as e:
        _echo(f"Evaluation failed: {e}")
        raise typer.Exit(code=2)
def init(
    path: str = typer.Option(
        "./stateset_agents.yaml", help="Path for a starter config"
    ),
) -> None:
    """Scaffold a minimal config to get started."""
    import textwrap

    cfg = (
        textwrap.dedent(
            """
        # StateSet Agents - Starter Config
        agent:
          model_name: gpt2
          max_new_tokens: 64
          temperature: 0.7

        training:
          num_episodes: 5
          max_turns: 3

        environment:
          type: conversation
          scenarios:
            - id: demo
              topic: general_help
              context: "User needs general assistance"
              user_responses:
                - "Thanks! Can you elaborate?"
                - "Interesting, tell me more."
        """
        ).strip()
        + "\n"
    )
    try:
        with open(path, "w") as f:
            f.write(cfg)
        _echo(f"Wrote starter config to {path}")
    except Exception as e:
        _echo(f"Failed to write config: {e}")


def run() -> None:
    app()


if __name__ == "__main__":
    run()
