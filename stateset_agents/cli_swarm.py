"""Render reproducible experiment plans for the StateSet Sandbox swarm coordinator."""

import json
import subprocess
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import typer

app = typer.Typer(help="Plan experiments for the StateSet Sandbox swarm.")


@dataclass(frozen=True)
class Task:
    """One independently runnable experiment and its required worker source."""

    workload: str
    required_file: str
    parameter: int | None = None

    @property
    def key(self) -> str:
        """Return the coordinator task ID and its evidence directory name."""
        if self.parameter is None:
            return self.workload
        return f"{self.workload}-{self.parameter}"

    def script(self) -> str:
        """Render the exact Bash workload for both local and hosted execution."""
        source = files("stateset_agents.swarm").joinpath(f"{self.workload}.sh")
        lines = ["#!/usr/bin/env bash", "set -eu", f"task={self.key}"]
        if self.parameter is not None:
            lines.append(f"parameter={self.parameter}")
        lines.extend(
            [
                'dest="swarm-results/$task"',
                'if [ -e "$dest" ]; then echo "$dest already exists" >&2; exit 1; fi',
                "mkdir -p swarm-results",
                "scratch=$(mktemp -d)",
                """trap 'rm -rf -- "$scratch"' EXIT""",
                'out=$(mktemp -d "swarm-results/.$task.XXXXXXXX")',
                """trap 'rm -rf -- "$scratch" "$out"' EXIT""",
                source.read_text(encoding="utf-8").rstrip(),
                'python -c \'import os, sys; os.rename(*sys.argv[1:])\' "$out" "$dest"',
                'echo "$dest/result.json"',
            ]
        )
        return "\n".join(lines) + "\n"


RL_SEEDS = tuple(
    Task("rl-seed", "stateset_agents/fanout/worker.py", seed) for seed in (42, 43, 44)
)
CHECKPOINT_EDIT = Task(
    "checkpoint-edit", "stateset_agents/experimental/weight_space/checkpoints.py"
)
QWEN_LAYERS = tuple(
    Task("qwen-layer", "examples/fanout/weight_edit/fact-1.json", layer)
    for layer in (6, 14)
)
RECIPES = {
    "starter": (
        "Run one GRPO seed and verify a targeted checkpoint edit.",
        (RL_SEEDS[0], CHECKPOINT_EDIT),
    ),
    "rl-seeds": (
        "Measure synthetic refund-workflow learning across three seeds.",
        RL_SEEDS,
    ),
    "qwen-layers": (
        "Compare two edit sites for the same counterfactual Qwen fact.",
        QWEN_LAYERS,
    ),
}

# Install into a fresh environment on the stock CPU sandbox. Keep model caches,
# training checkpoints and installation files outside the returned git patch.
SETUP = """python3 -m venv /tmp/swarm-venv
source /tmp/swarm-venv/bin/activate
python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-cache-dir '.[training]'"""


@app.command("list")
def list_recipes() -> None:
    """List recipes with the task IDs accepted by the script command."""
    for name, (goal, tasks) in RECIPES.items():
        typer.echo(f"{name}  {goal}\n  {', '.join(task.key for task in tasks)}")


@app.command()
def script(task_id: str) -> None:
    """Print a single task as Bash for a local replay from the worker checkout."""
    for _, tasks in RECIPES.values():
        for task in tasks:
            if task.key == task_id:
                typer.echo(task.script(), nl=False)
                return
    raise typer.BadParameter(f"Unknown task {task_id!r}. See 'swarm list'.")


def _recipe(name: str) -> tuple[str, tuple[Task, ...]]:
    try:
        return RECIPES[name]
    except KeyError:
        raise typer.BadParameter(
            f"Unknown recipe {name!r}. See 'swarm list'."
        ) from None


@app.command()
def report(recipe: str, results: Path = typer.Option(Path("swarm-results"))) -> None:
    """Summarize every expected task from returned evidence and flag missing results."""
    _, tasks = _recipe(recipe)
    rows = {}
    complete = True
    for task in tasks:
        directory = results / task.key
        try:
            result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            if task.workload == "rl-seed":
                if result["run_id"] != task.key or result["seed"] != task.parameter:
                    raise ValueError("RL receipt belongs to another task")
                metrics = result["metrics"]
                rows[task.key] = {**metrics, "status": result["status"]}
                if (
                    result["status"] != "succeeded"
                    or metrics["optimizer_updates"] < 1
                    or metrics["reload_verified"] != 1
                ):
                    complete = False
            elif task.workload == "qwen-layer":
                metrics = json.loads(
                    (directory / "metrics.json").read_text(encoding="utf-8")
                )
                status = result["status"]
                if status not in {
                    "accepted",
                    "rejected",
                    "baseline_unmet",
                    "export_rejected",
                }:
                    raise ValueError(f"Unknown edit decision: {status!r}")
                if metrics["accepted"] != (status == "accepted"):
                    raise ValueError("Edit decision differs between report and metrics")
                module = f"model.layers.{task.parameter}.mlp.down_proj"
                if any(attempt["module"] != module for attempt in result["attempts"]):
                    raise ValueError("Edit receipt belongs to another layer")
                rows[task.key] = {
                    **metrics,
                    "status": status,
                    "export": result["export"],
                }
                errors = [
                    attempt["error"]
                    for attempt in result["attempts"]
                    if attempt["error"]
                ]
                if errors:
                    rows[task.key]["errors"] = errors
                    complete = False
            else:
                rows[task.key] = {
                    k: result[k]
                    for k in ("changed_weights", "before", "after", "reload_verified")
                }
                if result["reload_verified"] is not True:
                    complete = False
        except (OSError, KeyError, ValueError, TypeError) as exc:
            rows[task.key] = {"error": str(exc)}
            complete = False
    typer.echo(json.dumps(rows, indent=2))
    if not complete:
        raise typer.Exit(1)


@app.command()
def plan(
    recipe: str,
    repo: Path = typer.Option(Path("."), help="StateSet Agents git checkout."),
    base: str = typer.Option("v0.73.2", help="Committed worker branch, tag or SHA."),
) -> None:
    """Print a native coordinator plan as JSON. Runs locally without credentials."""
    goal, tasks = _recipe(recipe)

    def git(*args: str) -> str:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True
        )
        if proc.returncode:
            raise typer.BadParameter(
                proc.stderr.strip() or "Could not read worker checkout."
            )
        return proc.stdout.strip()

    root = git("rev-parse", "--show-toplevel")
    revision = git("rev-parse", "--verify", "--end-of-options", f"{base}^{{commit}}")
    for path in sorted({task.required_file for task in tasks}):
        git("cat-file", "-e", f"{revision}:{path}")

    members = []
    for task in tasks:
        destination = f"swarm-results/{task.key}"
        objective = (
            "From the repository root, execute this Bash block in one shell. "
            f"Return the generated evidence under {destination}/ in the patch. "
            "On command failure, report the error.\n\n```bash\nset -eu\n"
            f"{SETUP}\n{task.script()}git add -- {destination}\n```"
        )
        members.append({"id": task.key, "name": task.key, "objective": objective})
    result = {
        "goal": goal,
        "repository": {"path": root, "base": revision},
        "defaults": {
            "where": "sandbox",
            "timeout_seconds": 3600,
            "max_budget_usd": "1",
        },
        "sandbox": {
            "template": "claude-code",
            "cpus": "2",
            "memory": "12Gi" if recipe == "qwen-layers" else "8Gi",
            "ephemeral_storage": "10Gi",
        },
        "tasks": members,
    }
    typer.echo(json.dumps(result, indent=2))
