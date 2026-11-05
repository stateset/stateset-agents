"""Explicit-threshold variants of a designed neuron, built offline (no GPU).

    python threshold_design.py --captures captures_v5.pt \\
        --start-spec specs/conclude_neuron_v3t_spec.json --out designs/ \\
        --bias -1 0 1 2 --write-norm 2 3 4

The ridge detector behind the conclude neuron has no intercept, so its firing
threshold is whatever the fit's shrinkage left, and refitting it (v4, v5, the
on-policy rounds) moved that threshold along with the detector. This keeps
the start spec's detector fixed and sets the threshold explicitly with
``circuits.shift_threshold``: gate row ``gate - bias * up``, where the up row
reads ~1.0 everywhere, so the pre-activation drops by ``bias``.

``--write-norm`` is the write when firing, as in ``design_conditional_neuron``:
the down column is rescaled so the mean activation over captured positive
contexts that fire (activation > 0.5) times the column norm equals
``write_norm / 3 * `` the start spec's. Bias 0 at write norm 3 is therefore the
start spec, edit for edit. Writes one spec per grid point and
``threshold_grid.json`` with offline fire rates per capture group, which only
describe the base model's contexts; generation decides (``threshold_job.py``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

START_WRITE_NORM = 3.0
FIRE = 0.5


def grid_name(base: str, bias: float, write_norm: float) -> str:
    return f"{base}_b{bias:+.1f}_w{write_norm:.1f}"


def main(argv: list[str] | None = None) -> None:
    import torch
    from offline_design import load_layer

    from stateset_agents.experimental.weight_space import (
        EditSpec,
        NeuronRewrite,
        rewrite_activation,
        shift_threshold,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", required=True)
    ap.add_argument("--start-spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bias", type=float, nargs="+", default=[-1.0, 0.0, 1.0, 2.0])
    ap.add_argument("--write-norm", type=float, nargs="+", default=[2.0, 3.0, 4.0])
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cap = torch.load(a.captures, weights_only=True)  # tensors, lists, str, int only
    start = EditSpec.from_json(Path(a.start_spec).read_text())
    (rw,) = [e for e in start.edits if isinstance(e, NeuronRewrite)]
    w = load_layer(rw.layer)
    gate_acts, up_acts = cap["gate_acts"].float(), cap["up_acts"].float()
    labels = torch.tensor(cap["labels"])
    groups: list[str] = cap["groups"]
    pos = labels > 0

    def activation(r: NeuronRewrite) -> torch.Tensor:
        return rewrite_activation(
            r,
            gate_weight=w["gate_proj"],
            up_weight=w["up_proj"],
            gate_acts=gate_acts,
            up_acts=up_acts,
        )

    def act_on(act: torch.Tensor) -> float:
        firing = act[pos & (act > FIRE)]
        return float(firing.mean())

    on0 = act_on(activation(rw))
    rows = []
    for bias in a.bias:
        shifted = shift_threshold(rw, bias)
        act = activation(shifted)
        on = act_on(act)
        for wn in a.write_norm:
            scale = (wn / START_WRITE_NORM) * (on0 / on)
            final = shift_threshold(rw, bias, write_scale=scale)
            edits = tuple(final if e is rw else e for e in start.edits)
            spec = EditSpec(name=grid_name(start.name, bias, wn), edits=edits)
            path = out / f"{spec.name}_spec.json"
            path.write_text(spec.to_json())
            rows.append(
                {
                    "name": spec.name,
                    "spec": path.name,
                    "spec_hash": spec.content_hash(),
                    "bias": bias,
                    "write_norm": wn,
                    "write_scale": round(scale, 6),
                    "offline_fire_rate_by_group": {
                        g: round(
                            float(
                                (act[torch.tensor([x == g for x in groups])] > FIRE)
                                .float()
                                .mean()
                            ),
                            4,
                        )
                        for g in dict.fromkeys(groups)
                    },
                }
            )
            print(
                json.dumps(
                    {k: rows[-1][k] for k in ("name", "spec_hash", "write_scale")}
                )
            )
    grid = {
        "start_spec": start.name,
        "start_spec_hash": start.content_hash(),
        "start_activation_when_firing": round(on0, 4),
        "grid": rows,
    }
    (out / "threshold_grid.json").write_text(json.dumps(grid, indent=2))


if __name__ == "__main__":
    main()
