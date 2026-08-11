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
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the training plan without training."
    ),
) -> None:
    """Run the SFT job from `improve` on local or rented GPU compute."""
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
            gpu=gpu,
            timeout_s=timeout,
            package_version=package_version,
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
