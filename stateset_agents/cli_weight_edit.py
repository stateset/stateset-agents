"""``stateset-agents weight-edit`` — apply, verify, control and compare direct weight edits.

The ship layer for :mod:`stateset_agents.experimental.weight_space`: an edit
spec (JSON) is the unit of provenance. ``apply`` writes a patched checkpoint
with a manifest, ``verify`` audits a checkpoint against its spec and base,
``control`` derives a structure-matched random control spec, and ``compare``
computes paired statistics (exact McNemar, bootstrap CI) between two
per-problem result files.

Follows the cli_ingest.py pattern: imports the shared ``app`` and attaches a
sub-app at import time. Heavy dependencies (torch, safetensors) load lazily
inside the commands.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import typer

from stateset_agents import cli as _cli
from stateset_agents.cli import app

_echo = _cli._echo

weight_edit_app = typer.Typer(
    add_completion=False,
    help="Apply, verify, control and compare direct weight edits (experimental).",
)
app.add_typer(weight_edit_app, name="weight-edit")


def _load_spec(path: Path) -> Any:
    from stateset_agents.experimental.weight_space.edits import EditSpec

    try:
        return EditSpec.from_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError) as exc:
        _echo(f"Could not read edit spec {path}: {exc}", err=True)
        raise typer.Exit(code=2) from exc


def _load_results(path: Path) -> list[dict[str, Any]]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "id" not in row or "correct" not in row:
            _echo(f"{path}:{n}: each row needs 'id' and 'correct'", err=True)
            raise typer.Exit(code=2)
        rows.append(row)
    return rows


def _resolve(base: str, revision: str | None) -> Path:
    """Local checkpoint dir for ``base``; exit 2 with a message if it can't be had."""
    from stateset_agents.experimental.weight_space.checkpoints import (
        resolve_checkpoint,
    )

    try:
        return resolve_checkpoint(base, revision)
    except (OSError, ImportError, ValueError) as exc:
        _echo(f"Could not resolve checkpoint {base!r}: {exc}", err=True)
        raise typer.Exit(code=2) from exc


@weight_edit_app.command("apply")
def apply_cmd(
    spec: Path = typer.Argument(..., help="Edit spec JSON."),
    base: str = typer.Option(..., "--base", help="Base checkpoint dir or Hub id."),
    out: Path = typer.Option(..., "--out", help="Output dir (must not exist)."),
    revision: str | None = typer.Option(
        None, "--revision", help="Hub revision to pin when --base is a Hub id."
    ),
) -> None:
    """Write a copy of BASE with SPEC applied, plus weight_space_spec.json."""
    from stateset_agents.experimental.weight_space.checkpoints import (
        apply_spec_to_checkpoint,
    )

    edit_spec = _load_spec(spec)
    try:
        manifest = apply_spec_to_checkpoint(edit_spec, _resolve(base, revision), out)
    except (FileExistsError, FileNotFoundError, KeyError, ValueError) as exc:
        _echo(f"apply failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    manifest["base"] = base if revision is None else f"{base}@{revision}"
    _echo(json.dumps(manifest, indent=2, sort_keys=True))


@weight_edit_app.command("verify")
def verify_cmd(
    spec: Path = typer.Argument(..., help="Edit spec JSON."),
    base: str = typer.Option(..., "--base", help="Base checkpoint dir or Hub id."),
    edited: Path = typer.Option(..., "--edited", help="Edited checkpoint dir."),
    revision: str | None = typer.Option(None, "--revision", help="Hub revision."),
    tolerance: float = typer.Option(0.01, help="Allowed error in each slice's scale."),
) -> None:
    """Exit 0 only if EDITED is exactly BASE + SPEC (every tensor checked)."""
    from stateset_agents.experimental.weight_space.checkpoints import (
        verify_checkpoint,
    )
    from stateset_agents.experimental.weight_space.edits import SpecMismatch

    edit_spec = _load_spec(spec)
    try:
        verify_checkpoint(
            edit_spec, _resolve(base, revision), edited, tolerance=tolerance
        )
    except SpecMismatch as exc:
        _echo(f"MISMATCH: {exc}")
        raise typer.Exit(code=1) from exc
    _echo(f"verified: {edited} == base + spec {edit_spec.content_hash()}")


@weight_edit_app.command("control")
def control_cmd(
    spec: Path = typer.Argument(..., help="Edit spec JSON."),
    base: str = typer.Option(..., "--base", help="Base checkpoint dir or Hub id."),
    seed: int = typer.Option(0, "--seed", help="RNG seed for target selection."),
    revision: str | None = typer.Option(None, "--revision", help="Hub revision."),
) -> None:
    """Print a control spec: same kinds, layers and scales on random targets."""
    from stateset_agents.experimental.weight_space.checkpoints import (
        checkpoint_layout,
        checkpoint_shapes,
    )
    from stateset_agents.experimental.weight_space.edits import matched_control

    edit_spec = _load_spec(spec)
    root = _resolve(base, revision)
    shapes = checkpoint_shapes(root)
    layout = checkpoint_layout(root)
    vocab = shapes[layout.token_keys[0]][0]
    neurons = {
        shape[1] for name, shape in shapes.items() if name.endswith("down_proj.weight")
    }
    control = matched_control(
        edit_spec,
        rng=random.Random(seed),
        vocab_size=vocab,
        n_neurons=min(neurons),
    )
    _echo(control.to_json())


@weight_edit_app.command("compare")
def compare_cmd(
    base: Path = typer.Argument(
        ..., help="Base per-problem results (JSONL: id, correct)."
    ),
    other: Path = typer.Argument(..., help="Edited per-problem results (same format)."),
    seed: int = typer.Option(0, "--seed", help="Bootstrap seed."),
) -> None:
    """Paired comparison on shared problem ids: flips, delta, 95% CI, McNemar p."""
    from stateset_agents.experimental.weight_space.protocol import paired_compare

    try:
        stats = paired_compare(_load_results(base), _load_results(other), seed=seed)
    except ValueError as exc:
        _echo(f"compare failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    _echo(json.dumps(stats, indent=2))


@weight_edit_app.command("atlas")
def atlas_cmd(
    base: str = typer.Option(..., "--base", help="Checkpoint dir or Hub id."),
    token_id: list[int] = typer.Option(
        ..., "--token-id", help="Token id(s) to find writers for (repeatable)."
    ),
    top: int = typer.Option(10, "--top", help="How many neurons to list."),
    against: bool = typer.Option(
        False, "--against", help="List neurons that write against the token(s)."
    ),
    norm: str = typer.Option(
        "auto",
        "--norm",
        help="Final RMSNorm convention: auto, zero-centered (x*(1+w)) or plain (x*w).",
    ),
    revision: str | None = typer.Option(None, "--revision", help="Hub revision."),
) -> None:
    """Weight-only atlas: MLP neurons whose output columns write toward TOKEN_ID.

    No forward passes. Each neuron is scored by the cosine between its
    down_proj column and the token's logit direction (final-norm gain times
    unembedding row) and listed with the tokens it promotes most. These are
    candidates to test, not causes.
    """
    import re

    from stateset_agents.experimental.weight_space.atlas import (
        logit_direction,
        neuron_readout,
        top_writers,
    )
    from stateset_agents.experimental.weight_space.checkpoints import (
        checkpoint_layout,
        checkpoint_shapes,
        load_tensors,
        norm_convention,
    )

    if norm not in {"auto", "zero-centered", "plain"}:
        _echo("--norm must be auto, zero-centered or plain", err=True)
        raise typer.Exit(code=2)
    root = _resolve(base, revision)
    layout = checkpoint_layout(root)
    unembed_key = layout.token_keys[0]  # lm_head when stored, else the tied embedding
    embed_key = next(k for k in layout.token_keys if k.endswith("embed_tokens.weight"))
    norm_key = embed_key.replace("embed_tokens.weight", "norm.weight")
    down_re = re.compile(re.escape(layout.down_fmt).replace(r"\{layer\}", r"(\d+)"))
    down_keys = {
        int(m.group(1)): name
        for name in checkpoint_shapes(root)
        if (m := down_re.fullmatch(name))
    }
    tensors = load_tensors(root, {unembed_key, norm_key, *down_keys.values()})
    unembed = tensors[unembed_key].float()
    norm_weight = tensors[norm_key].float()
    zero_centered = (
        norm_convention(root) if norm == "auto" else norm
    ) == "zero-centered"
    direction = logit_direction(
        unembed, token_id, norm_weight, zero_centered=zero_centered
    )
    down = {layer: tensors[key].float() for layer, key in sorted(down_keys.items())}
    writers = top_writers(down, -direction if against else direction, k=top)
    report = {
        "token_ids": token_id,
        "norm": "zero-centered" if zero_centered else "plain",
        "writers": [
            {
                "layer": w.layer,
                "neuron": w.neuron,
                "alignment": round(w.alignment, 6),
                "promotes": neuron_readout(
                    down[w.layer][:, w.neuron],
                    unembed,
                    norm_weight,
                    k=8,
                    zero_centered=zero_centered,
                ),
            }
            for w in writers
        ],
    }
    _echo(json.dumps(report, indent=2))
