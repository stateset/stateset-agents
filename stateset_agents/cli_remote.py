"""``stateset-agents train-remote`` — run the fine-tune step on rented compute.

Thin orchestrator over ``stateset_agents.remote``. No training logic lives
here: the command builds a :class:`RemoteJobSpec`, hands it to the executor
registered under ``--provider``, and renders the result.

Picks up where ``improve`` leaves off::

    stateset-agents improve run --transcripts sessions/ -o improved/
    stateset-agents train-remote --provider modal \\
        --dataset improved/curated.jsonl --base-model Qwen/Qwen3.5-0.8B
"""

from __future__ import annotations

from pathlib import Path

import typer

from stateset_agents import cli as _cli
from stateset_agents.cli import app
from stateset_agents.core.errors import StateSetError
from stateset_agents.remote.job import RemoteJobSpec
from stateset_agents.remote.registry import available_providers, get_executor

_echo = _cli._echo


@app.command("chat-remote")
def chat_remote(
    base_model: str = typer.Option(
        ..., "--base-model", help="Hugging Face base model (e.g. Qwen/Qwen3.5-0.8B)."
    ),
    adapter: Path | None = typer.Option(
        None,
        "--adapter",
        help="Local LoRA adapter directory (e.g. outputs/sft_v1) to load on "
        "top of the base model. Uploaded to the pod for the session.",
    ),
    gpu: str = typer.Option(
        "NVIDIA H100 80GB HBM3",
        "--gpu",
        help="RunPod GPU type to rent, in RunPod's own vocabulary.",
    ),
    container_disk_gb: int = typer.Option(
        160,
        "--container-disk-gb",
        help="Container disk in GB for the model download — size it at "
        "roughly 2.5x the checkpoint.",
    ),
    max_turns: int = typer.Option(
        50,
        "--max-turns",
        help="Safety cap on interactive turns; the pod bills while you type.",
    ),
    prompt: list[str] = typer.Option(
        [],
        "--prompt",
        help="Non-interactive mode: send this prompt (repeatable, in order), "
        "print each reply, and exit. Skips the input() loop entirely.",
    ),
) -> None:
    """Chat with a fine-tuned model on a rented RunPod GPU, ephemerally.

    Rents a pod, loads the base model plus your LoRA adapter there, and
    opens a REPL over SSH. The pod is terminated when the session ends —
    no open ports, no idle billing. Type ``exit``/``quit`` or Ctrl+D/Ctrl+C
    to leave.
    """
    from stateset_agents.remote import chat_session

    if adapter is not None and not adapter.exists():
        _echo(f"Adapter directory does not exist: {adapter}", err=True)
        raise typer.Exit(code=2)

    session = chat_session.RemoteChatSession(container_disk_gb=container_disk_gb)
    exit_code = 0
    try:
        _echo(f"Renting a {gpu} pod and loading {base_model}…")
        if adapter is not None:
            _echo(f"With adapter: {adapter}")
        session.start(base_model=base_model, adapter_dir=adapter, gpu=gpu)
        _echo("Model ready. The pod bills until you exit.")

        if prompt:
            for text in prompt:
                _echo(f"you> {text}")
                _echo(f"agent> {session.ask(text)}")
        else:
            turns = 0
            while turns < max_turns:
                try:
                    user_input = input("\nyou> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not user_input:
                    continue
                if user_input in ("exit", "quit"):
                    break
                _echo(f"agent> {session.ask(user_input)}")
                turns += 1
            else:
                _echo(f"Reached --max-turns ({max_turns}); ending the session.")
    except StateSetError as exc:
        _echo(str(exc), err=True)
        exit_code = 1
    finally:
        _echo("Terminating the pod…")
        session.close()

    if exit_code:
        raise typer.Exit(code=exit_code)
    _echo("Session ended; pod terminated.")


@app.command("train-remote")
def train_remote(
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        help="Chat-format JSONL to train on — typically improved/curated.jsonl.",
    ),
    base_model: str = typer.Option(
        ..., "--base-model", help="Hugging Face base model (e.g. Qwen/Qwen3.5-0.8B)."
    ),
    provider: str = typer.Option(
        "local",
        "--provider",
        help=f"Where to run: {', '.join(available_providers())}.",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/sft_v1"),
        "--output-dir",
        help="Where the trained adapter is written.",
    ),
    num_epochs: int = typer.Option(3, "--num-epochs"),
    lora_r: int = typer.Option(16, "--lora-r"),
    lora_alpha: int = typer.Option(32, "--lora-alpha"),
    learning_rate: float = typer.Option(2e-5, "--learning-rate"),
    max_length: int = typer.Option(1024, "--max-length"),
    per_device_batch_size: int = typer.Option(2, "--per-device-batch-size"),
    gradient_accumulation_steps: int = typer.Option(4, "--gradient-accumulation-steps"),
    gpu: str | None = typer.Option(
        None,
        "--gpu",
        help="GPU to request, in the provider's own vocabulary (Modal: "
        '"A10G"; RunPod: "NVIDIA RTX A4000"). Defaults to the provider\'s '
        "own default.",
    ),
    timeout: int = typer.Option(3600, "--timeout", help="Job timeout in seconds."),
    package_version: str | None = typer.Option(
        None,
        "--package-version",
        help="stateset-agents version installed remotely. Defaults to the "
        "running version.",
    ),
    container_disk_gb: int | None = typer.Option(
        None,
        "--container-disk-gb",
        help="RunPod only: GPU-pool container disk in GB for the model "
        "download. Size it at roughly 2.5x the checkpoint (a 30B BF16 "
        "model is ~63GB). Defaults to the executor's own default.",
    ),
    eval_prompts: Path | None = typer.Option(
        None,
        "--eval-prompts",
        help="Local text file of prompts, one per line (blanks skipped). "
        "After training, each prompt is answered by both the base model "
        "and the tuned adapter; the comparison lands in "
        "output_dir/eval_results.json.",
    ),
    eval_max_new_tokens: int = typer.Option(
        90,
        "--eval-max-new-tokens",
        help="Token budget per eval completion. Raise it for reasoning "
        "models whose answers follow a long preamble.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the training plan without training."
    ),
) -> None:
    """Run the SFT job from `improve` on local or rented GPU compute."""
    prompts: list[str] | None = None
    if eval_prompts is not None:
        if not eval_prompts.exists():
            _echo(f"Eval prompts file does not exist: {eval_prompts}", err=True)
            raise typer.Exit(code=2)
        prompts = [
            line.strip()
            for line in eval_prompts.read_text().splitlines()
            if line.strip()
        ]

    try:
        spec = RemoteJobSpec(
            dataset=dataset,
            base_model=base_model,
            output_dir=output_dir,
            num_epochs=num_epochs,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            learning_rate=learning_rate,
            max_length=max_length,
            per_device_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dry_run=dry_run,
            eval_prompts=prompts,
            eval_max_new_tokens=eval_max_new_tokens,
            gpu=gpu,
            timeout_s=timeout,
            package_version=package_version,
            container_disk_gb=container_disk_gb,
        )
    except ValueError as exc:
        _echo(f"Invalid job: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        executor = get_executor(provider)
    except StateSetError as exc:
        _echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    _echo(f"Submitting SFT job to '{provider}' ({spec.gpu})…")
    try:
        result = executor.wait(executor.submit(spec))
    except StateSetError as exc:
        _echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    for line in result.logs:
        _echo(line)

    if not result.succeeded:
        _echo(f"Job {result.status.value}.", err=True)
        raise typer.Exit(code=1)

    _echo(f"Done. Adapter written to {result.output_dir}")
    _echo("Use it with: stateset-agents serve --checkpoint " f"{result.output_dir}")
