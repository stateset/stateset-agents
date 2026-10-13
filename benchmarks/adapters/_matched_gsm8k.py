"""Shared runtime for the matched-GSPO competitor adapters (verl, OpenRLHF, NeMo-RL).

The upstream-TRL adapter (``trl_grpo.py``) defines the measured protocol: the
GSM8K train/eval split, the prompt template, the exact-numeric reward, the
greedy evaluation procedure, and the adapter result schema that
``benchmarks/shootout.py`` validates. Every competitor adapter reuses those
pieces verbatim (``GSM8KTask``, ``canonical_digest`` and ``train_reward_stats``
are imported from ``trl_grpo``) and differs only in how it drives its own
framework's trainer, which always runs as a subprocess in that framework's
own virtual environment.

Flow of one measured run (``run_adapter``):

1. validate the CLI and the manifest config exactly like ``trl_grpo.py``;
2. probe the framework interpreter and fail unless it reports the pinned
   framework version;
3. evaluate the untrained base model with the shared evaluator;
4. write the framework's data/config, launch training, stream its log into
   the artifact directory, and sample device memory while it runs;
5. parse the framework's own metrics, fail closed unless it completed every
   declared optimizer step and reported a training reward;
6. load the produced LoRA adapter on the base model and evaluate it with the
   shared evaluator;
7. write the adapter result only after every step above succeeded.

This module also hosts the framework-side GSM8K reward bridge
(``compute_score`` for verl, ``reward_func`` for OpenRLHF, ``score_text`` for
the NeMo-RL environment). Those functions import only the standard library at
module level so a framework can load this file by path from its own venv;
the scorer itself is ``trl_grpo.GSM8KTask.score_response``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

_ADAPTERS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _ADAPTERS_DIR.parents[1]


def _ensure_import_paths() -> None:
    """Make ``trl_grpo`` and ``stateset_agents`` importable from any venv."""
    for entry in (str(_ADAPTERS_DIR), str(_REPO_ROOT)):
        if entry not in sys.path:
            sys.path.insert(0, entry)


_ensure_import_paths()

from trl_grpo import (  # noqa: E402  (path bootstrap above is deliberate)
    CONFIG_FIELDS,
    GSM8KTask,
    canonical_digest,
    train_reward_stats,
)

MATCHED_OBJECTIVE: dict[str, Any] = {
    "objective": "gspo-sequence",
    "importance_sampling_level": "sequence",
    "epsilon": 3e-4,
    "epsilon_high": 4e-4,
    "loss_type": "grpo",
    "scale_rewards": "group",
}
RAW_PROMPT_CHAT_TEMPLATE = "{{ messages[0].content }}"
MIB = 1024 * 1024


class AdapterFailure(RuntimeError):
    """A measured run that must not produce a completed adapter result."""


# ---------------------------------------------------------------------------
# CLI contract (identical required flags to trl_grpo.py)
# ---------------------------------------------------------------------------


def build_parser(description: str) -> argparse.ArgumentParser:
    """Return the shared adapter CLI (the TRL contract plus optional locators)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--task", choices=["gsm8k"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--adapter-output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--framework-python",
        default=None,
        help="interpreter of the framework's own venv (default: env override "
        "or the current interpreter)",
    )
    parser.add_argument(
        "--framework-dir",
        type=Path,
        default=None,
        help="framework source checkout (NeMo-RL runs through `uv run` there)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="scratch directory for checkpoints and data (default: a fresh "
        "temp dir outside the evidence bundle, removed after success)",
    )
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=6 * 3600)
    return parser


def parse_config(parser: argparse.ArgumentParser, args: argparse.Namespace) -> dict:
    """Apply the exact config validation ``trl_grpo.py`` performs."""
    if len(args.model_revision) != 40 or len(args.dataset_revision) != 40:
        parser.error("model and dataset revisions must be full 40-character commits")
    try:
        config = json.loads(args.config_json)
    except json.JSONDecodeError as exc:
        parser.error(f"invalid --config-json: {exc}")
    if not isinstance(config, dict) or set(config) != CONFIG_FIELDS:
        parser.error("--config-json has an unsupported schema")
    if any(config.get(key) != value for key, value in MATCHED_OBJECTIVE.items()):
        parser.error("--config-json must declare the matched GSPO objective")
    return config


# ---------------------------------------------------------------------------
# Protocol arithmetic shared by every adapter
# ---------------------------------------------------------------------------


def prompts_per_step(config: Mapping[str, Any]) -> int:
    """Distinct prompts sampled per optimizer step.

    Matches the StateSet harness (``generations_per_iteration = batch x
    accumulation``); each prompt then gets ``num_generations`` completions.
    """
    return int(config["per_device_train_batch_size"]) * int(
        config["gradient_accumulation_steps"]
    )


def completions_per_step(config: Mapping[str, Any]) -> int:
    """Completions (rollouts) trained on per optimizer step."""
    return prompts_per_step(config) * int(config["num_generations"])


def samples_processed(config: Mapping[str, Any], steps_completed: int) -> int:
    """Completions processed; the same formula ``trl_grpo.py`` reports."""
    return int(steps_completed) * completions_per_step(config)


def warmup_steps(config: Mapping[str, Any]) -> int:
    """Warmup steps exactly as ``trl_grpo.py`` derives them."""
    return int(int(config["max_steps"]) * float(config["warmup_ratio"]))


def deviation(
    setting: str, matched: Any, framework_value: Any, reason: str
) -> dict[str, Any]:
    """One honest record of a protocol setting the framework cannot match."""
    return {
        "setting": setting,
        "matched_protocol": matched,
        "framework_value": framework_value,
        "reason": reason,
    }


def task_rows(train_examples: Sequence[Any]) -> list[dict[str, Any]]:
    """Training rows carrying the fixed prompt and the gold answer."""
    return [
        {
            "index": index,
            "prompt": GSM8KTask.format_prompt(example),
            "gold_answer": float(example.gold_answer),
        }
        for index, example in enumerate(train_examples)
    ]


def gold_label(value: float) -> str:
    """Lossless text form of a gold answer for string-typed label columns."""
    return repr(float(value))


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write JSON lines deterministically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    return path


# ---------------------------------------------------------------------------
# Framework-side reward bridge (imported by path from framework venvs)
# ---------------------------------------------------------------------------


def score_text(response: str, gold_answer: float | str) -> float:
    """Score one completion with the TRL adapter's exact-numeric reward."""
    _ensure_import_paths()
    example = SimpleNamespace(gold_answer=float(gold_answer))
    return float(GSM8KTask.score_response(example, response)[0])


def compute_score(
    data_source: Any = None,
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: Any = None,
    **_: Any,
) -> float:
    """verl ``reward.custom_reward_function`` entry point."""
    return score_text(solution_str, ground_truth)


_SPECIAL_SUFFIXES = ("<|endoftext|>", "<|im_end|>")


def strip_prompt(query: str, prompt: str) -> str:
    """Recover the completion from OpenRLHF's decoded prompt+response text."""
    response = query[len(prompt) :] if query.startswith(prompt) else query
    changed = True
    while changed:
        changed = False
        stripped = response.rstrip()
        for suffix in _SPECIAL_SUFFIXES:
            if stripped.endswith(suffix):
                response = stripped[: -len(suffix)]
                changed = True
    return response


def reward_func(
    queries: Sequence[str], prompts: Sequence[str], labels: Sequence[Any]
) -> dict[str, Any]:
    """OpenRLHF ``--reward.remote_url <file.py>`` entry point."""
    import torch

    scores = [
        score_text(strip_prompt(query, prompt), label)
        for query, prompt, label in zip(queries, prompts, labels, strict=True)
    ]
    tensor = torch.tensor(scores, dtype=torch.float32)
    return {"rewards": tensor, "scores": tensor, "extra_logs": {}}


# ---------------------------------------------------------------------------
# Subprocess execution with device-memory sampling
# ---------------------------------------------------------------------------


@dataclass
class CommandOutcome:
    """Exit status, elapsed time, and sampled device peak of one subprocess."""

    returncode: int
    elapsed_seconds: float
    peak_device_mb: float | None = None


def query_device_memory_mb(device_index: int = 0) -> float | None:
    """Device-wide used memory from ``nvidia-smi`` (all processes), or None."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                str(device_index),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    try:
        return float(completed.stdout.strip().splitlines()[0])
    except (IndexError, ValueError):
        return None


class DeviceMemorySampler:
    """Poll device memory in a background thread and keep the maximum."""

    def __init__(
        self,
        interval_seconds: float = 2.0,
        query: Callable[[], float | None] = query_device_memory_mb,
    ) -> None:
        self.interval_seconds = interval_seconds
        self.query = query
        self.peak_mb: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            value = self.query()
            if value is not None and (self.peak_mb is None or value > self.peak_mb):
                self.peak_mb = value
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> DeviceMemorySampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=30)


def run_command(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    log_path: Path,
    timeout_seconds: int,
    cwd: Path | None = None,
) -> CommandOutcome:
    """Run one framework command, streaming output into ``log_path``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (
        log_path.open("a", encoding="utf-8") as log,
        DeviceMemorySampler() as sampler,
    ):
        log.write(f"$ {subprocess.list2cmdline(list(command))}\n")
        log.flush()
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(env),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            returncode = -9
    return CommandOutcome(
        returncode=returncode,
        elapsed_seconds=time.monotonic() - started,
        peak_device_mb=sampler.peak_mb,
    )


def probe_framework(command: Sequence[str], env: Mapping[str, str]) -> dict[str, str]:
    """Run a probe that prints one JSON object (``version``/``cuda``)."""
    try:
        completed = subprocess.run(
            list(command),
            env=dict(env),
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AdapterFailure(f"framework probe could not start: {exc}") from exc
    if completed.returncode != 0:
        raise AdapterFailure(
            f"framework probe exited {completed.returncode}: "
            f"{completed.stderr.strip()[-2000:]}"
        )
    for line in reversed(completed.stdout.strip().splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("version"), str):
            return {key: str(value) for key, value in payload.items()}
    raise AdapterFailure("framework probe printed no version JSON")


def probe_code(module: str) -> str:
    """Python source a framework probe runs inside the framework venv."""
    return (
        "import json, torch, " + module + " as m; "
        "print(json.dumps({'version': m.__version__, "
        "'cuda': str(torch.version.cuda)}))"
    )


def normalize_version(raw: str) -> str:
    """Drop a local ``+git...`` suffix that source checkouts append."""
    return raw.split("+", 1)[0].strip()


# ---------------------------------------------------------------------------
# Harness-side model access and evaluation (runs in the harness venv)
# ---------------------------------------------------------------------------


@dataclass
class EvalOutcome:
    """One shared-evaluator score plus the allocator peak it needed."""

    score: float
    peak_vram_mb: float


def cuda_info() -> dict[str, Any]:
    """Hardware identity from the harness venv (the TRL adapter's source)."""
    import torch

    if not torch.cuda.is_available():
        raise AdapterFailure("the measured adapter requires CUDA")
    return {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "cuda": str(torch.version.cuda),
    }


def resolve_model(model: str, revision: str) -> Path:
    """Materialize the pinned model snapshot; frameworks receive this path."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=model, revision=revision))


def load_examples(
    n_train: int, n_eval: int, dataset_revision: str
) -> tuple[list[Any], list[Any]]:
    """The TRL adapter's exact train/eval subsets."""
    train, test = GSM8KTask.load(n_train, n_eval, dataset_revision=dataset_revision)
    return list(train), list(test)


def evaluate(
    model: str,
    revision: str,
    examples: Sequence[Any],
    *,
    max_tokens: int,
    adapter_dir: Path | None = None,
) -> EvalOutcome:
    """Greedy pass@1 with the shared evaluator (base model or base + LoRA)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from stateset_agents.evaluation.framework_protocol import evaluate_causal_lm

    tokenizer = AutoTokenizer.from_pretrained(
        model, revision=revision, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    network: Any = AutoModelForCausalLM.from_pretrained(
        model, revision=revision, torch_dtype=torch.bfloat16
    )
    network.to("cuda")
    if adapter_dir is not None:
        from peft import PeftModel

        network = PeftModel.from_pretrained(network, str(adapter_dir))
        network.to("cuda")
    torch.cuda.reset_peak_memory_stats()
    try:
        score = evaluate_causal_lm(
            network,
            tokenizer,
            examples,
            format_prompt=GSM8KTask.format_prompt,
            score_response=GSM8KTask.score_response,
            max_tokens=max_tokens,
        )["pass_at_1"]
        peak = torch.cuda.max_memory_allocated(0) / MIB
    finally:
        del network
        gc.collect()
        torch.cuda.empty_cache()
    return EvalOutcome(score=float(score), peak_vram_mb=float(peak))


def set_seeds(seed: int) -> None:
    """Seed the harness process (evaluation is greedy; this is hygiene)."""
    from stateset_agents.utils.reproducibility import set_all_seeds

    set_all_seeds(seed)


@dataclass
class Runtime:
    """Side-effecting operations, injectable so tests never touch a GPU."""

    cuda_info: Callable[[], dict[str, Any]] = cuda_info
    probe_framework: Callable[..., dict[str, str]] = probe_framework
    load_examples: Callable[..., tuple[list[Any], list[Any]]] = load_examples
    resolve_model: Callable[[str, str], Path] = resolve_model
    evaluate: Callable[..., EvalOutcome] = evaluate
    run_command: Callable[..., CommandOutcome] = run_command
    set_seeds: Callable[[int], None] = set_seeds


# ---------------------------------------------------------------------------
# Framework adapter contract
# ---------------------------------------------------------------------------


@dataclass
class LaunchContext:
    """Everything a framework needs to build its training launch."""

    args: argparse.Namespace
    config: Mapping[str, Any]
    model_path: Path
    work_dir: Path
    artifact_dir: Path
    train_rows: list[dict[str, Any]]
    env: dict[str, str]


@dataclass
class LaunchPlan:
    """A framework training launch plus its protocol accounting."""

    command: list[str]
    env: dict[str, str]
    settings: dict[str, Any]
    deviations: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    cwd: Path | None = None


@dataclass
class TrainingOutcome:
    """What a framework run produced, parsed from its own outputs."""

    steps_completed: int
    step_rewards: list[float]
    step_reward_stds: list[float]
    adapter_dir: Path
    framework_peak_vram_mb: float | None = None
    framework_metrics: dict[str, float] = field(default_factory=dict)


class FrameworkAdapter(Protocol):
    """What each competitor adapter module implements."""

    name: str
    pinned_version: str

    def probe_command(self, ctx_args: argparse.Namespace) -> list[str]:
        """Command printing ``{"version": ..., "cuda": ...}`` from the venv."""

    def prepare(self, ctx: LaunchContext) -> LaunchPlan:
        """Write framework inputs and return the training launch."""

    def collect(
        self, ctx: LaunchContext, plan: LaunchPlan, runtime: Runtime
    ) -> TrainingOutcome:
        """Parse metrics and locate the trained LoRA adapter (fail closed)."""


def require_completed_steps(steps: int, config: Mapping[str, Any], name: str) -> None:
    """Fail closed when a framework stopped short of the declared steps."""
    expected = int(config["max_steps"])
    if steps != expected:
        raise AdapterFailure(
            f"{name} reported {steps} completed optimizer steps; the protocol "
            f"declares {expected}"
        )


def require_rewards(rewards: Sequence[float], name: str) -> None:
    """Fail closed when the framework metrics carry no training reward."""
    if not rewards:
        raise AdapterFailure(f"{name} metrics contain no per-step training reward")


def find_peft_adapter(root: Path) -> Path:
    """Return the single directory under ``root`` holding a PEFT adapter."""
    if not root.exists():
        raise AdapterFailure(f"framework output does not exist: {root}")
    candidates = sorted(
        {
            config.parent
            for config in root.rglob("adapter_config.json")
            if any(
                (config.parent / name).is_file()
                for name in ("adapter_model.safetensors", "adapter_model.bin")
            )
        }
    )
    if len(candidates) != 1:
        raise AdapterFailure(
            f"expected exactly one PEFT adapter under {root}, found "
            f"{len(candidates)}"
        )
    return candidates[0]


def install_adapter(adapter_dir: Path, artifact_dir: Path) -> None:
    """Copy the trained adapter's files into the hashed artifact directory."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for item in sorted(adapter_dir.iterdir()):
        if item.is_file():
            shutil.copy2(item, artifact_dir / item.name)


def _framework_env(extra: Mapping[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    env.update(extra)
    pythonpath = [str(_ADAPTERS_DIR), str(_REPO_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env


def execute(
    framework: FrameworkAdapter,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    runtime: Runtime,
) -> dict[str, Any]:
    """Run one measured seed and return the completed adapter result."""
    hardware = dict(runtime.cuda_info())
    base_env = _framework_env({})
    probe = runtime.probe_framework(framework.probe_command(args), base_env)
    raw_version = str(probe["version"])
    if normalize_version(raw_version) != framework.pinned_version:
        raise AdapterFailure(
            f"installed {framework.name} version {raw_version!r} does not match "
            f"the pinned {framework.pinned_version}"
        )
    runtime.set_seeds(args.seed)
    train_examples, eval_examples = runtime.load_examples(
        int(config["num_train_examples"]),
        int(config["num_eval_examples"]),
        args.dataset_revision,
    )
    model_path = Path(runtime.resolve_model(args.model, args.model_revision))
    max_tokens = int(config["max_completion_length"])
    baseline = runtime.evaluate(
        args.model, args.model_revision, eval_examples, max_tokens=max_tokens
    )

    created_work_dir = args.work_dir is None
    work_dir = (
        Path(
            tempfile.mkdtemp(
                prefix=f"{framework.name}-seed{args.seed}-",
                dir=os.environ.get("STATESET_FRAMEWORK_WORK_ROOT") or None,
            )
        )
        if created_work_dir
        else Path(args.work_dir)
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    framework_dir = artifact_dir / "framework"
    framework_dir.mkdir(exist_ok=True)
    ctx = LaunchContext(
        args=args,
        config=config,
        model_path=model_path,
        work_dir=work_dir,
        artifact_dir=artifact_dir,
        train_rows=task_rows(train_examples),
        env=base_env,
    )
    plan = framework.prepare(ctx)
    (framework_dir / "launch.json").write_text(
        json.dumps(
            {
                "command": plan.command,
                "settings": plan.settings,
                "protocol_deviations": plan.deviations,
                "protocol_notes": plan.notes,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    outcome = runtime.run_command(
        plan.command,
        env=plan.env,
        log_path=framework_dir / "train.log",
        timeout_seconds=int(args.timeout_seconds),
        cwd=plan.cwd,
    )
    if outcome.returncode != 0:
        raise AdapterFailure(
            f"{framework.name} training exited {outcome.returncode}; see "
            f"{framework_dir / 'train.log'}"
        )
    training = framework.collect(ctx, plan, runtime)
    require_completed_steps(training.steps_completed, config, framework.name)
    require_rewards(training.step_rewards, framework.name)
    install_adapter(training.adapter_dir, artifact_dir)
    final = runtime.evaluate(
        args.model,
        args.model_revision,
        eval_examples,
        max_tokens=max_tokens,
        adapter_dir=artifact_dir,
    )

    log_history: list[dict[str, Any]] = (
        [
            {"reward": reward, "reward_std": std}
            for reward, std in zip(
                training.step_rewards, training.step_reward_stds, strict=True
            )
        ]
        if len(training.step_reward_stds) == len(training.step_rewards)
        else [{"reward": reward} for reward in training.step_rewards]
    )
    if training.framework_peak_vram_mb is not None:
        training_peak, peak_source = (
            training.framework_peak_vram_mb,
            "framework-reported-allocator-peak",
        )
    elif outcome.peak_device_mb is not None:
        training_peak, peak_source = (
            outcome.peak_device_mb,
            "nvml-device-used-sampled",
        )
    else:
        raise AdapterFailure(
            f"{framework.name} reported no memory peak and device sampling failed"
        )
    peak_vram_mb = max(float(training_peak), baseline.peak_vram_mb, final.peak_vram_mb)
    hardware["framework_cuda"] = str(probe.get("cuda", ""))
    result = {
        "status": "completed",
        "measured": True,
        "config_sha256": canonical_digest(dict(config)),
        "framework_version": framework.pinned_version,
        "framework_version_raw": raw_version,
        "artifact_path": str(artifact_dir.resolve()),
        "hardware": hardware,
        "metrics": {
            "samples_processed": samples_processed(config, training.steps_completed),
            "peak_vram_mb": peak_vram_mb,
            "eval_score_baseline": baseline.score,
            "eval_score_final": final.score,
            **train_reward_stats(log_history),
            "train_steps_completed": float(training.steps_completed),
            "framework_training_seconds": outcome.elapsed_seconds,
            "training_peak_vram_mb": float(training_peak),
            **{
                f"framework_{key}": float(value)
                for key, value in training.framework_metrics.items()
            },
        },
        "peak_vram_source": peak_source,
        "protocol": {
            "framework": framework.name,
            "matched_objective": dict(MATCHED_OBJECTIVE),
            "prompts_per_step": prompts_per_step(config),
            "completions_per_step": completions_per_step(config),
            "settings": plan.settings,
        },
        "protocol_deviations": list(plan.deviations),
        "protocol_notes": list(plan.notes),
    }
    if created_work_dir and not args.keep_work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)
    return result


def write_result(path: Path, result: Mapping[str, Any]) -> None:
    """Write the adapter result atomically (only ever for a completed run)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_adapter(
    framework: FrameworkAdapter,
    argv: Sequence[str] | None = None,
    runtime: Runtime | None = None,
    description: str | None = None,
) -> int:
    """CLI entry point shared by the competitor adapters."""
    parser = build_parser(description or f"Matched GSPO adapter for {framework.name}")
    args = parser.parse_args(argv)
    config = parse_config(parser, args)
    try:
        result = execute(framework, args, config, runtime or Runtime())
    except AdapterFailure as exc:
        print(f"{framework.name} adapter failed closed: {exc}", file=sys.stderr)
        return 2
    write_result(args.adapter_output, result)
    return 0
