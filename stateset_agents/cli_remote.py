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

from collections.abc import Callable
from pathlib import Path

import typer

from stateset_agents import cli as _cli
from stateset_agents.cli import app
from stateset_agents.core.errors import StateSetError
from stateset_agents.remote.job import RemoteJobSpec
from stateset_agents.remote.registry import available_providers, get_executor

_echo = _cli._echo


def _save_transcript(
    session: object, path: Path | None, *, now: Callable[[], float]
) -> None:
    """Persist a chat session as one ingest-ready JSONL line, best-effort.

    Runs on the ``finally`` path of ``chat-remote`` so aborted chats persist
    too — every conversation is training data for the ingest -> improve ->
    train-remote flywheel. Nothing is written for an empty conversation, and
    a disk error is reported but never masks the session's own outcome.
    """
    import json
    import time

    transcript = session.transcript  # type: ignore[attr-defined]
    if not transcript["messages"]:
        return
    if path is None:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now()))
        path = Path("chat_transcripts") / f"chat_{stamp}.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(transcript) + "\n")
    except OSError as exc:
        _echo(f"Could not save transcript to {path}: {exc}", err=True)
        return
    _echo(f"Transcript saved to {path}")
    _echo("Feed it back into training with:")
    _echo(
        f"  stateset-agents ingest --format openai --input {path} "
        "--output graded.jsonl"
    )


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
    save_transcript: Path | None = typer.Option(
        None,
        "--save-transcript",
        help="Where to save the conversation transcript (OpenAI chat-format "
        "JSONL, ready for `stateset-agents ingest --format openai`). "
        "Defaults to ./chat_transcripts/chat_<timestamp>.jsonl.",
    ),
    save: bool = typer.Option(
        True,
        "--save/--no-save",
        help="Save the transcript on exit (default: on). Every chat is "
        "training data — the transcript feeds ingest -> improve -> "
        "train-remote.",
    ),
) -> None:
    """Chat with a fine-tuned model on a rented RunPod GPU, ephemerally.

    Rents a pod, loads the base model plus your LoRA adapter there, and
    opens a REPL over SSH. The pod is terminated when the session ends —
    no open ports, no idle billing. Type ``exit``/``quit`` or Ctrl+D/Ctrl+C
    to leave. On exit the conversation is saved as an ingest-ready
    transcript (disable with ``--no-save``).
    """
    import time as _time

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
        # Close first — the pod bills until it dies; the transcript can wait.
        _echo("Terminating the pod…")
        session.close()
        if save:
            _save_transcript(session, save_transcript, now=_time.time)

    if exit_code:
        raise typer.Exit(code=exit_code)
    _echo("Session ended; pod terminated.")


def _parse_adapters(entries: list[str]) -> dict[str, Path]:
    """``[name=]path`` entries -> ``{name: Path}``; bare path -> 'adapter'.

    Duplicate names are refused loudly — vLLM would otherwise register one
    adapter and silently shadow the other.
    """
    adapters: dict[str, Path] = {}
    for entry in entries:
        name, sep, raw = entry.partition("=")
        if not sep:
            name, raw = "adapter", entry
        name = name.strip()
        if not name or "/" in name:
            raise typer.BadParameter(
                f"adapter name {name!r} must be a plain identifier"
            )
        if name in adapters:
            raise typer.BadParameter(f"duplicate adapter name {name!r}")
        adapters[name] = Path(raw.strip())
    return adapters


@app.command("serve-remote")
def serve_remote(
    base_model: str | None = typer.Option(
        None,
        "--base-model",
        help="Hugging Face base model to serve (e.g. Qwen/Qwen3.5-0.8B). "
        "Required unless --stop or --list is given.",
    ),
    adapter: list[str] = typer.Option(
        [],
        "--adapter",
        help="Local LoRA adapter directory to serve on top of the base "
        "model, as '[name=]path' — repeatable, so several fine-tunes can "
        "ride one endpoint for A/B comparison. A bare path serves under "
        "the name 'adapter'.",
    ),
    gpu: str = typer.Option(
        "NVIDIA RTX A4000",
        "--gpu",
        help="RunPod GPU type, in RunPod's own vocabulary. The default's "
        "16 GB VRAM fits ~7B fp16 models; go bigger for bigger models.",
    ),
    container_disk_gb: int = typer.Option(
        60,
        "--container-disk-gb",
        help="Container disk in GB — must fit the vLLM install (~10 GB) plus "
        "roughly 2.5x the model checkpoint.",
    ),
    max_hours: float = typer.Option(
        1.0,
        "--max-hours",
        help="Cost control: a self-destruct armed ON THE POD terminates it "
        "after this many hours, even if this machine goes away. The RunPod "
        "API key is copied to the pod (chmod 600) to make that possible.",
    ),
    stop: str | None = typer.Option(
        None,
        "--stop",
        help="Terminate a running serve pod by name or id, then exit.",
    ),
    list_pods: bool = typer.Option(
        False,
        "--list",
        help="List running serve pods (name, id, status, age, $/hr), then exit.",
    ),
) -> None:
    """Serve a model as a persistent OpenAI-compatible endpoint on RunPod.

    Rents a pod, installs vLLM, loads the base model (plus your adapter),
    and prints the endpoint URL and a generated Bearer token. The pod KEEPS
    RUNNING after this command exits — that is the point — so every run arms
    an on-pod self-destruct at ``--max-hours``, and ``--stop``/``--list``
    exist to manage what is running.
    """
    from stateset_agents.remote import serve_session

    if list_pods:
        try:
            rows = serve_session.list_serve_pods(
                serve_session.RemoteServeSession()._require_api()
            )
        except StateSetError as exc:
            _echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        if not rows:
            _echo("No serve pods running.")
            return
        for row in rows:
            cost = row["cost_per_hr"]
            cost_s = f"${cost}/hr" if cost is not None else "?/hr"
            _echo(
                f"{row['name']}  {row['id']}  {row['status']}  "
                f"age {row['age']}  {cost_s}"
            )
        return

    if stop is not None:
        try:
            api = serve_session.RemoteServeSession()._require_api()
            pod = serve_session.find_serve_pod(api, stop)
            api.terminate_pod(str(pod["id"]))
        except StateSetError as exc:
            _echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        _echo(f"Terminated pod {pod.get('name')} ({pod['id']}). Billing stopped.")
        return

    if base_model is None:
        _echo("--base-model is required (or use --stop / --list).", err=True)
        raise typer.Exit(code=2)
    adapters = _parse_adapters(adapter)
    for directory in adapters.values():
        if not directory.exists():
            _echo(f"Adapter directory does not exist: {directory}", err=True)
            raise typer.Exit(code=2)
    if max_hours <= 0:
        _echo("--max-hours must be positive.", err=True)
        raise typer.Exit(code=2)

    session = serve_session.RemoteServeSession(container_disk_gb=container_disk_gb)
    _echo(f"Renting a {gpu} pod and serving {base_model} with vLLM…")
    for name, directory in adapters.items():
        _echo(f"With adapter: {directory} (served-model name: {name})")
    _echo(f"Self-destruct armed on the pod at {max_hours}h.")
    try:
        session.start(
            base_model=base_model,
            adapters=adapters,
            gpu=gpu,
            max_hours=max_hours,
        )
    except StateSetError as exc:
        _echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    model_name = "adapter" if adapter is not None else base_model
    _echo("")
    _echo(f"Endpoint ready (bills until stopped, max {max_hours}h):")
    _echo(f"  URL:   {session.endpoint_url}/v1")
    _echo(f"  Token: {session.token}")
    _echo(f"  Pod:   {session.pod_name} ({session.pod_id})")
    _echo("")
    _echo("Example:")
    _echo(f"  curl {session.endpoint_url}/v1/chat/completions \\")
    _echo(f'    -H "Authorization: Bearer {session.token}" \\')
    _echo('    -H "Content-Type: application/json" \\')
    _echo(
        f'    -d \'{{"model": "{model_name}", '
        '"messages": [{"role": "user", "content": "Hello"}]}\''
    )
    _echo("")
    _echo("Stop it (billing stops immediately):")
    _echo(f"  stateset-agents serve-remote --stop {session.pod_name}")


def _parse_eval_prompt_line(line: str) -> str | dict:
    """One line of the --eval-prompts file.

    A line that parses as a JSON *object* is a prompt-spec dict
    (``{"prompt", "expect", "forbid", "judge", "min_judge_score"}``);
    anything else — including JSON that isn't an object — is a plain prompt
    string, exactly as before. Spec contents are validated by
    ``RemoteJobSpec``, so a malformed spec exits 2 with the reason.
    """
    import json

    try:
        parsed = json.loads(line)
    except json.JSONDecodeError:
        return line
    return parsed if isinstance(parsed, dict) else line


@app.command("deploy")
def deploy(
    dataset: Path = typer.Option(
        ..., "--dataset", help="Chat-format JSONL to train on."
    ),
    base_model: str = typer.Option(
        ..., "--base-model", help="Hugging Face base model."
    ),
    output_dir: Path = typer.Option(
        Path("outputs/deploy_v1"),
        "--output-dir",
        help="Where the trained adapter is written locally.",
    ),
    gpu: str = typer.Option(
        "NVIDIA H100 80GB HBM3",
        "--gpu",
        help="RunPod GPU used for BOTH the training job and the endpoint.",
    ),
    container_disk_gb: int | None = typer.Option(
        None, "--container-disk-gb", help="~2.5x the checkpoint size."
    ),
    num_epochs: int = typer.Option(3, "--num-epochs"),
    max_cost_usd: float | None = typer.Option(
        None, "--max-cost", help="Ceiling for the TRAINING job."
    ),
    max_hours: float = typer.Option(
        1.0,
        "--max-hours",
        help="Endpoint self-destruct, armed on the serving pod.",
    ),
) -> None:
    """Fine-tune on a rented GPU, then serve the result — one command.

    ``train-remote`` then ``serve-remote``, glued: rent, train, give the
    hardware back, rent again, serve the fresh adapter as an authenticated
    OpenAI-compatible endpoint, print the URL and token. The zero-to-API
    story of docs/GETTING_STARTED_API.md as a single invocation.
    """
    from stateset_agents.remote import serve_session

    spec = RemoteJobSpec(
        dataset=dataset,
        base_model=base_model,
        output_dir=output_dir,
        num_epochs=num_epochs,
        gpu=gpu,
        container_disk_gb=container_disk_gb,
        max_cost_usd=max_cost_usd,
    )
    try:
        executor = get_executor("runpod")
    except StateSetError as exc:
        _echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    _echo(f"[1/2] Training {base_model} on {dataset} ({gpu})…")
    try:
        result = executor.wait(executor.submit(spec))
    except StateSetError as exc:
        _echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    if not result.succeeded or result.output_dir is None:
        for line in result.logs[-20:]:
            _echo(line)
        _echo(f"Training {result.status.value}; not serving.", err=True)
        raise typer.Exit(code=1)
    if result.cost_usd is not None:
        _echo(f"Trained. Cost: ~${result.cost_usd:.2f}")

    _echo(f"[2/2] Serving {base_model} + fresh adapter…")
    session = serve_session.RemoteServeSession(
        container_disk_gb=container_disk_gb or 60
    )
    _echo(f"Self-destruct armed on the pod at {max_hours}h.")
    try:
        session.start(
            base_model=base_model,
            adapters={"adapter": Path(result.output_dir)},
            gpu=gpu,
            max_hours=max_hours,
        )
    except StateSetError as exc:
        _echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    _echo("")
    _echo(f"Endpoint ready (bills until stopped, max {max_hours}h):")
    _echo(f"URL:   {session.endpoint_url}/v1")
    _echo(f"Token: {session.token}")
    _echo(
        "Ask the fine-tune:  curl "
        f"{session.endpoint_url}/v1/chat/completions "
        f'-H "Authorization: Bearer {session.token}" '
        '-H "Content-Type: application/json" '
        '-d \'{"model":"adapter","messages":[{"role":"user",'
        '"content":"hello"}]}\''
    )
    _echo(f"Stop it:  stateset-agents serve-remote --stop {session.pod_name}")


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
        help=(
            f"Where to run: {', '.join(available_providers())}. "
            "'river' is River AI's remote autograd service: it trains without "
            "renting a machine, so the GPU/disk/cloud-type options are "
            "ignored, and the result is a river:// checkpoint pointer rather "
            "than local adapter weights (NOT live-verified — see "
            "docs/RIVER_PROVIDER.md)."
        ),
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
    gpu_count: int = typer.Option(
        1,
        "--gpu-count",
        help="RunPod only: how many GPUs of the requested type to attach. "
        "With more than one, the job shards the model across all of them "
        "(device_map='auto'), letting a checkpoint bigger than one card "
        "train. Billing scales with the count.",
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
    cloud_type: str = typer.Option(
        "SECURE",
        "--cloud-type",
        help="RunPod only: SECURE (default, reserved capacity) or COMMUNITY "
        "(~spot pricing — markedly cheaper, but the pod can be reclaimed "
        "mid-job; the executor then provisions a fresh pod once and "
        "restarts training from scratch).",
    ),
    max_cost_usd: float | None = typer.Option(
        None,
        "--max-cost",
        help=(
            "Refuse to run if the pod could cost more than this many dollars "
            "(its full --timeout at the provider's quoted hourly rate). The "
            "check happens before any work starts."
        ),
    ),
    network_volume_id: str | None = typer.Option(
        None,
        "--network-volume-id",
        help="RunPod only: id of an existing network volume to mount at "
        "/workspace. Checkpoints then survive pod death, and the "
        "pod-died-mid-job retry resumes from the newest checkpoint "
        "instead of restarting from scratch. The pod is pinned to the "
        "volume's datacenter. The volume is yours to manage — it bills "
        "monthly until you delete it.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume from the newest checkpoint-* in --output-dir when one "
        "exists (otherwise trains fresh). Useful for rerunning an "
        "interrupted local job, or a RunPod job whose checkpoints live "
        "on a network volume (--network-volume-id); a fresh RunPod pod "
        "without a volume starts with an empty output dir, so this is a "
        "no-op there.",
    ),
    eval_prompts: Path | None = typer.Option(
        None,
        "--eval-prompts",
        help="Local text file of prompts, one per line (blanks skipped). "
        "A line that parses as a JSON object is a prompt spec — "
        '{"prompt", "expect", "forbid", "judge", "min_judge_score"} — '
        "whose assertions gate the job's exit code; any other line is a "
        "plain prompt. After training, each prompt is answered by both "
        "the base model and the tuned adapter; the comparison lands in "
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
    prompts: list[str | dict] | None = None
    if eval_prompts is not None:
        if not eval_prompts.exists():
            _echo(f"Eval prompts file does not exist: {eval_prompts}", err=True)
            raise typer.Exit(code=2)
        prompts = [
            _parse_eval_prompt_line(line.strip())
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
            resume=resume,
            eval_prompts=prompts,
            eval_max_new_tokens=eval_max_new_tokens,
            gpu=gpu,
            gpu_count=gpu_count,
            timeout_s=timeout,
            package_version=package_version,
            container_disk_gb=container_disk_gb,
            cloud_type=cloud_type,
            network_volume_id=network_volume_id,
            max_cost_usd=max_cost_usd,
        )
    except ValueError as exc:
        _echo(f"Invalid job: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        executor = get_executor(provider)
    except StateSetError as exc:
        _echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    _echo(f"Submitting SFT job to '{provider}' ({spec.gpu or 'provider default'})…")
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

    if result.cost_usd is not None:
        _echo(f"Cost: ~${result.cost_usd:.2f} ({result.duration_s:.0f}s of pod time)")
    if provider.strip().lower() == "river":
        # River keeps the weights; what landed locally is a pointer, so the
        # usual `serve --checkpoint` hint would be a lie.
        _echo(f"Done. River checkpoint pointer written to {result.output_dir}")
        _echo(
            "The trained LoRA lives on River — sample it through the River "
            f"API using the checkpoint in {result.output_dir}/river_checkpoint.json"
        )
        return
    _echo(f"Done. Adapter written to {result.output_dir}")
    _echo("Use it with: stateset-agents serve --checkpoint " f"{result.output_dir}")


@app.command("costs")
def costs(
    ledger: Path | None = typer.Option(
        None,
        "--ledger",
        help="Ledger file to read (default: the shared per-user ledger).",
    ),
    limit: int = typer.Option(10, "--limit", help="How many recent runs to list."),
    json_output: bool = typer.Option(
        False, "--json", "--json-output", help="Emit machine-readable JSON."
    ),
) -> None:
    """Show what remote runs have actually cost.

    Every remote run appends a line to the cost ledger — what it trained, on
    what hardware, for how long, and the dollar amount. This reads it back.
    """
    import json as _json

    from stateset_agents.remote.ledger import read_entries, summarize

    entries = read_entries(ledger)
    summary = summarize(entries)

    if json_output:
        _echo(
            _json.dumps(
                {"summary": summary, "recent": entries[-limit:]},
                indent=2,
                sort_keys=True,
            )
        )
        return

    if not entries:
        _echo("No remote runs recorded yet.")
        return

    _echo(
        f"{summary['runs']} run(s), ${summary['total_usd']:.2f} total "
        f"({summary['runs_with_known_cost']} with a known price)"
    )
    _echo("")
    for entry in entries[-limit:]:
        cost = entry.get("cost_usd")
        cost_text = f"${cost:.2f}" if isinstance(cost, (int, float)) else "  ? "
        duration = entry.get("duration_s")
        mins = f"{duration / 60:.0f}m" if isinstance(duration, (int, float)) else "?"
        _echo(
            f"  {cost_text:>7}  {mins:>4}  {entry.get('status', '?'):<9} "
            f"{entry.get('gpu', '?')}  {entry.get('base_model', '?')}"
        )
    _echo("")
    for model, total in sorted(summary["by_model"].items(), key=lambda kv: -kv[1])[:5]:
        _echo(f"  ${total:>7.2f}  {model}")


@app.command("adapters")
def adapters(
    directory: Path = typer.Option(
        Path("outputs"),
        "--dir",
        "-d",
        help="Directory to scan for trained adapters.",
    ),
    json_output: bool = typer.Option(
        False, "--json", "--json-output", help="Emit machine-readable JSON."
    ),
) -> None:
    """List trained adapters with their provenance and lineage.

    Every training run writes a manifest beside its adapter — base model,
    dataset hash, hyperparameters, eval outcome, and the adapter it descends
    from. This reads them back, so an adapter directory is never anonymous.
    """
    import json as _json

    from stateset_agents.training.lineage import build_lineage, discover_adapters

    found = discover_adapters(directory)
    if json_output:
        _echo(
            _json.dumps(
                {"adapters": found, "lineage": build_lineage(found)},
                indent=2,
                sort_keys=True,
            )
        )
        return

    if not found:
        _echo(f"No adapters found under {directory}/")
        return

    lineage = build_lineage(found)
    _echo(f"{len(found)} adapter(s) under {directory}/")
    _echo("")
    for entry in found:
        manifest = entry.get("manifest")
        _echo(f"  {entry['path']}")
        if not manifest:
            _echo("      (no manifest — trained before provenance was recorded)")
        else:
            _echo(f"      base:    {manifest.get('base_model', '?')}")
            rows = manifest.get("dataset_rows")
            sha = (manifest.get("dataset_sha256") or "")[:12]
            _echo(f"      data:    {rows if rows is not None else '?'} rows  {sha}")
            hyper = manifest.get("hyperparameters") or {}
            if hyper:
                _echo(
                    f"      train:   r={hyper.get('lora_r', '?')} "
                    f"epochs={hyper.get('num_epochs', '?')} "
                    f"lr={hyper.get('learning_rate', '?')}"
                )
            passed, total = manifest.get("eval_passed"), manifest.get("eval_total")
            if total:
                _echo(f"      eval:    {passed}/{total} assertion(s) passed")
            if manifest.get("parent_adapter"):
                _echo(f"      parent:  {manifest['parent_adapter']}")
        children = lineage.get(entry["path"], [])
        for child in children:
            _echo(f"      child:   {child}")
    if lineage:
        _echo("")
        _echo(f"{len(lineage)} lineage link(s) found.")
