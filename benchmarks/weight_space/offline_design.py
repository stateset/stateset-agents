"""Design conditional-neuron specs offline from a pod's saved captures.

    python offline_design.py --captures captures_v5.pt --out designs/ \\
        --lam 100 --gain 4 [--drop-group intermediate_newline ...] [--tag _v5b]

``neuron_job.py --save-captures`` stores the layer's gate/up pre-activations,
labels and groups at every contrast context. Those, plus the base checkpoint's
layer weights (local, no GPU), are all ``design_conditional_neuron`` reads, so
detector settings and training-group choices can be swept for free; only the
chosen spec needs a GPU run. Prints per-group held-out fire rates and AUCs and
writes ``conclude_neuron{tag}_spec.json`` (+ its random-write control).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
THINK_END = 248069
WAIT_IDS = (13784, 13428)
PREFIX = "model.language_model"


def load_layer(layer: int) -> dict:
    from stateset_agents.experimental.weight_space.checkpoints import (
        load_tensors,
        resolve_checkpoint,
    )

    root = resolve_checkpoint(MODEL, REVISION)
    mlp = f"{PREFIX}.layers.{layer}.mlp"
    names = {
        f"{mlp}.gate_proj.weight",
        f"{mlp}.up_proj.weight",
        f"{mlp}.down_proj.weight",
        f"{PREFIX}.embed_tokens.weight",
        f"{PREFIX}.norm.weight",
    }
    t = {k: v.float() for k, v in load_tensors(root, names).items()}
    return {
        "gate_proj": t[f"{mlp}.gate_proj.weight"],
        "up_proj": t[f"{mlp}.up_proj.weight"],
        "down_proj": t[f"{mlp}.down_proj.weight"],
        "embed": t[f"{PREFIX}.embed_tokens.weight"],
        "norm": t[f"{PREFIX}.norm.weight"],
    }


def split_masks(groups: list[str], split: str):
    """(train, held) masks. ``parity``: even / odd index, as on the pod. Each
    trace adds its two positives at consecutive indices, so parity trains on
    the first line break after each answer and holds out the second.
    ``halves``: first / second half of every group (trace-disjoint up to one
    trace per group). ``all``: train on everything; held = everything
    (in-sample)."""
    import torch

    n = len(groups)
    if split == "parity":
        train = torch.arange(n) % 2 == 0
        return train, ~train
    if split == "all":
        everything = torch.ones(n, dtype=torch.bool)
        return everything, everything
    train = torch.zeros(n, dtype=torch.bool)
    for g in dict.fromkeys(groups):
        idx = [i for i, x in enumerate(groups) if x == g]
        train[idx[: len(idx) // 2]] = True
    return train, ~train


def evaluate(design, gate_acts, up_acts, labels, groups, held) -> dict:
    import torch

    from stateset_agents.experimental.weight_space import auc

    act = design.activation(gate_acts, up_acts)
    score = gate_acts.float() @ design.detector
    positive = torch.tensor(labels) > 0
    fire, aucs = {}, {}
    for g in dict.fromkeys(groups):
        m = torch.tensor([x == g for x in groups]) & held
        if not bool(m.any()):
            continue
        fire[g] = round(float((act[m] > 0.5).float().mean()), 4)
        if not bool((m & positive).any()):
            aucs[g] = round(auc(score[held & positive], score[m]), 4)
    return {
        "heldout_detector_auc": round(
            auc(score[held & positive], score[held & ~positive]), 4
        ),
        "heldout_fire_rate_by_group": fire,
        "heldout_detector_auc_by_group": aucs,
    }


def main(argv: list[str] | None = None) -> None:
    import torch

    from stateset_agents.experimental.weight_space import (
        EditSpec,
        NeuronRewrite,
        TokenRowScale,
        design_conditional_neuron,
        logit_direction,
        matched_control,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lam", type=float, default=100.0)
    ap.add_argument("--gain", type=float, default=4.0)
    ap.add_argument("--write-norm", type=float, default=3.0)
    ap.add_argument("--neuron", type=int, default=1998)
    ap.add_argument("--up-unit", type=int, default=44)
    ap.add_argument(
        "--drop-group",
        action="append",
        default=[],
        help="leave this group out of the detector's training contexts "
        "(it is still scored)",
    )
    ap.add_argument(
        "--split",
        choices=("parity", "halves", "all"),
        default="parity",
        help="parity: as on the pod (trains on the first line break after each "
        "answer, holds out the second); halves: trace-disjoint check; all: fit "
        "on everything (reported rates are in-sample, so judge by generation)",
    )
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    cap = torch.load(a.captures, weights_only=True)  # tensors, lists, str, int only
    gate_acts, up_acts = cap["gate_acts"].float(), cap["up_acts"].float()
    labels, groups, layer = list(cap["labels"]), list(cap["groups"]), cap["layer"]
    w = load_layer(layer)
    direction = logit_direction(w["embed"], [THINK_END], w["norm"], zero_centered=True)
    train, held = split_masks(groups, a.split)
    keep = torch.tensor([g not in a.drop_group for g in groups])
    design = design_conditional_neuron(
        layer=layer,
        gate_weight=w["gate_proj"],
        up_weight=w["up_proj"],
        down_weight=w["down_proj"],
        gate_acts=gate_acts,
        up_acts=up_acts,
        labels=torch.tensor(labels),
        write_direction=direction,
        write_norm=a.write_norm,
        gain=a.gain,
        lam=a.lam,
        neuron=a.neuron,
        up_unit=a.up_unit,
        train=train & keep,
    )
    report = {
        "lam": a.lam,
        "gain": a.gain,
        "dropped_groups": a.drop_group,
        "split": a.split,  # "all": the rates below are in-sample
        **evaluate(design, gate_acts, up_acts, labels, groups, held),
    }
    name = f"conclude-neuron-L{layer}{a.tag}"
    spec = EditSpec(name=name, edits=(TokenRowScale(WAIT_IDS, 0.5), design.rewrite))
    ctrl = matched_control(
        EditSpec(name=name, edits=(design.rewrite,)),
        rng=random.Random(0),
        vocab_size=w["embed"].shape[0],
        n_neurons=w["down_proj"].shape[1],
    ).edits[0]
    assert isinstance(ctrl, NeuronRewrite)
    control = EditSpec(
        name=f"control:{name}", edits=(TokenRowScale(WAIT_IDS, 0.5), ctrl)
    )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"conclude_neuron{a.tag}_spec.json").write_text(spec.to_json())
    (out / f"conclude_neuron{a.tag}_control_spec.json").write_text(control.to_json())
    report["spec_hash"] = spec.content_hash()
    (out / f"design{a.tag}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
