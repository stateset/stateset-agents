"""Typed, serializable weight edits and specs: apply to checkpoints, verify, control.

These are the edit kinds that moved Qwen3.5-0.8B on held-out math (see
``benchmarks/weight_space``): scaling rows of the (tied) embedding — "say this
token less/more" — scaling MLP neuron columns, and rewriting a whole MLP neuron
(its gate row, up row and output column) into a hand-designed circuit. An
:class:`EditSpec` is the
unit of provenance: it serializes to canonical JSON, carries a content hash,
patches a checkpoint's tensors in place, verifies a patched checkpoint against
the base, and generates a structure-matched random control.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ...core.errors import StateSetError

if TYPE_CHECKING:
    import torch

__all__ = [
    "EditSpec",
    "NeuronColumnScale",
    "NeuronRewrite",
    "SpecMismatch",
    "TensorLayout",
    "TokenRowScale",
    "apply_spec",
    "matched_control",
    "verify_spec",
]


class SpecMismatch(StateSetError, AssertionError):
    """A patched checkpoint does not match its edit spec."""

    message_only_str = True


@dataclass(frozen=True)
class TokenRowScale:
    """Multiply rows ``ids`` of the token embedding by ``scale``.

    With tied input/output embeddings this scales those tokens' logits by
    ``scale`` everywhere (and how they read as input).
    """

    ids: tuple[int, ...]
    scale: float

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "token_rows", "ids": list(self.ids), "scale": self.scale}


@dataclass(frozen=True)
class NeuronColumnScale:
    """Multiply columns ``neurons`` of layer ``layer``'s MLP ``down_proj`` by ``scale``."""

    layer: int
    neurons: tuple[int, ...]
    scale: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "down_cols",
            "layer": self.layer,
            "neurons": list(self.neurons),
            "scale": self.scale,
        }


@dataclass(frozen=True)
class NeuronRewrite:
    """Replace MLP neuron ``neuron`` of layer ``layer`` with a hand-designed unit.

    ``gate`` and ``up`` are its new ``gate_proj`` / ``up_proj`` rows (what it
    reads from the residual stream; its activation is
    ``act(gate . x) * (up . x)``) and ``down`` its new ``down_proj`` column
    (what it writes back). All three have the model's hidden size.
    """

    layer: int
    neuron: int
    gate: tuple[float, ...]
    up: tuple[float, ...]
    down: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "neuron_rewrite",
            "layer": self.layer,
            "neuron": self.neuron,
            "gate": list(self.gate),
            "up": list(self.up),
            "down": list(self.down),
        }


Edit = TokenRowScale | NeuronColumnScale | NeuronRewrite


def _edit_from_dict(d: Mapping[str, Any]) -> Edit:
    kind = d.get("kind", "token_rows")
    if kind == "token_rows":
        return TokenRowScale(
            ids=tuple(int(i) for i in d["ids"]), scale=float(d["scale"])
        )
    if kind == "down_cols":
        return NeuronColumnScale(
            layer=int(d["layer"]),
            neurons=tuple(int(j) for j in d["neurons"]),
            scale=float(d["scale"]),
        )
    if kind == "neuron_rewrite":
        return NeuronRewrite(
            layer=int(d["layer"]),
            neuron=int(d["neuron"]),
            gate=tuple(float(x) for x in d["gate"]),
            up=tuple(float(x) for x in d["up"]),
            down=tuple(float(x) for x in d["down"]),
        )
    raise ValueError(f"unknown edit kind {kind!r}")


@dataclass(frozen=True)
class EditSpec:
    """A named, ordered stack of edits; identity is the content hash of the edits."""

    name: str
    edits: tuple[Edit, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "edits": [e.to_dict() for e in self.edits]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> EditSpec:
        return cls(
            name=str(d["name"]), edits=tuple(_edit_from_dict(e) for e in d["edits"])
        )

    @classmethod
    def from_json(cls, text: str) -> EditSpec:
        return cls.from_dict(json.loads(text))

    def content_hash(self) -> str:
        canonical = json.dumps([e.to_dict() for e in self.edits], sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


_DOWN_RE = re.compile(r"^(.*\.layers\.)(\d+)(\.mlp\.down_proj\.weight)$")


@dataclass(frozen=True)
class TensorLayout:
    """Where a checkpoint keeps the tensors that edits address.

    ``token_keys`` are every token-indexed matrix a :class:`TokenRowScale`
    scales: the input embedding and, when the checkpoint stores one, the output
    head. Scaling both keeps tied and untied checkpoints consistent (a tied
    model reads only the embedding; a stored copy of the head stays in sync).
    ``down_fmt`` names a layer's MLP ``down_proj`` with a ``{layer}`` field.
    """

    token_keys: tuple[str, ...]
    down_fmt: str

    @property
    def gate_fmt(self) -> str:
        """A layer's MLP ``gate_proj`` name (``{layer}`` field)."""
        return self.down_fmt.replace("down_proj", "gate_proj")

    @property
    def up_fmt(self) -> str:
        """A layer's MLP ``up_proj`` name (``{layer}`` field)."""
        return self.down_fmt.replace("down_proj", "up_proj")

    @classmethod
    def detect(cls, names: Iterable[str]) -> TensorLayout:
        """Infer the layout from a checkpoint's tensor names."""
        names = list(names)
        embeds = sorted(n for n in names if n.endswith(".embed_tokens.weight"))
        if not embeds:
            raise ValueError("no '*.embed_tokens.weight' tensor in the checkpoint")
        heads = sorted(n for n in names if n == "lm_head.weight")
        # The decoder stack is the prefix with the most down_proj layers (this
        # skips auxiliary stacks such as multi-token-prediction heads).
        prefixes = Counter(
            (m.group(1), m.group(3)) for m in map(_DOWN_RE.match, names) if m
        )
        if not prefixes:
            raise ValueError("no '*.layers.N.mlp.down_proj.weight' tensors found")
        (head, tail), _ = prefixes.most_common(1)[0]
        return cls(token_keys=tuple(heads + embeds), down_fmt=f"{head}{{layer}}{tail}")


def _plan(
    spec: EditSpec, layout: TensorLayout
) -> dict[str, dict[tuple[str, int], float]]:
    """key -> {(axis, index): cumulative scale} for every slice the spec touches."""
    plan: dict[str, dict[tuple[str, int], float]] = {}
    for e in spec.edits:
        if isinstance(e, TokenRowScale):
            for key in layout.token_keys:
                slots = plan.setdefault(key, {})
                for i in e.ids:
                    slots[("row", i)] = slots.get(("row", i), 1.0) * e.scale
        elif isinstance(e, NeuronColumnScale):
            slots = plan.setdefault(layout.down_fmt.format(layer=e.layer), {})
            for j in e.neurons:
                slots[("col", j)] = slots.get(("col", j), 1.0) * e.scale
    return plan


def _sets(
    spec: EditSpec, layout: TensorLayout
) -> dict[str, dict[tuple[str, int], tuple[float, ...]]]:
    """key -> {(axis, index): new values} for every slice a NeuronRewrite writes.

    Raises ValueError if a slice is both rewritten and scaled, or rewritten twice.
    """
    scaled = _plan(spec, layout)
    sets: dict[str, dict[tuple[str, int], tuple[float, ...]]] = {}
    for e in spec.edits:
        if not isinstance(e, NeuronRewrite):
            continue
        for key, slot, values in (
            (layout.gate_fmt.format(layer=e.layer), ("row", e.neuron), e.gate),
            (layout.up_fmt.format(layer=e.layer), ("row", e.neuron), e.up),
            (layout.down_fmt.format(layer=e.layer), ("col", e.neuron), e.down),
        ):
            if slot in scaled.get(key, {}) or slot in sets.get(key, {}):
                raise ValueError(
                    f"layer {e.layer} neuron {e.neuron} is rewritten and also "
                    f"edited elsewhere in the spec ({key})"
                )
            sets.setdefault(key, {})[slot] = values
    return sets


def _slice(t: torch.Tensor, axis: str, idx: int) -> torch.Tensor:
    return t[idx] if axis == "row" else t[:, idx]


def apply_spec(
    spec: EditSpec,
    tensors: MutableMapping[str, torch.Tensor],
    layout: TensorLayout,
    *,
    strict: bool = True,
) -> int:
    """Patch ``tensors`` (a checkpoint's state dict) in place; return weights changed.

    With ``strict=False``, tensors the spec names but ``tensors`` lacks are
    skipped (for patching one shard of a sharded checkpoint at a time).
    """
    import torch

    plan = _plan(spec, layout)
    sets = _sets(spec, layout)
    changed = 0
    for key, set_slots in sets.items():
        if key not in tensors:
            if not strict:
                continue
            raise KeyError(f"{key} not in checkpoint")
        t = tensors[key]
        for (axis, idx), values in set_slots.items():
            limit, length = (
                (t.shape[0], t.shape[1]) if axis == "row" else (t.shape[1], t.shape[0])
            )
            if not 0 <= idx < limit:
                raise ValueError(f"{axis} {idx} out of range for {key} (size {limit})")
            if len(values) != length:
                raise ValueError(
                    f"{key}: rewrite has {len(values)} values, the {axis} has {length}"
                )
        for (axis, idx), values in set_slots.items():
            new = torch.tensor(values, dtype=torch.float32).to(t.dtype)
            changed += int((_slice(t, axis, idx) != new).sum())
            if axis == "row":
                t[idx] = new
            else:
                t[:, idx] = new
    for key, slots in plan.items():
        if key not in tensors:
            if not strict:
                continue
            raise KeyError(f"{key} not in checkpoint")
        t = tensors[key]
        for (axis, idx), _ in slots.items():
            limit = t.shape[0] if axis == "row" else t.shape[1]
            if not 0 <= idx < limit:
                raise ValueError(f"{axis} {idx} out of range for {key} (size {limit})")
        rows = sorted(i for (axis, i) in slots if axis == "row")
        cols = sorted(j for (axis, j) in slots if axis == "col")
        before_rows = t[rows].clone() if rows else None
        before_cols = t[:, cols].clone() if cols else None
        for (axis, idx), scale in slots.items():
            if axis == "row":
                t[idx] = (t[idx].float() * scale).to(t.dtype)
            else:
                t[:, idx] = (t[:, idx].float() * scale).to(t.dtype)
        if before_rows is not None:
            changed += int((t[rows] != before_rows).sum())
        if before_cols is not None:
            changed += int((t[:, cols] != before_cols).sum())
    return changed


def verify_spec(
    spec: EditSpec,
    base: Mapping[str, torch.Tensor],
    edited: Mapping[str, torch.Tensor],
    layout: TensorLayout,
    *,
    tolerance: float = 0.01,
) -> None:
    """Raise :class:`SpecMismatch` unless ``edited`` is exactly ``base`` + ``spec``.

    Every tensor the spec does not touch must be bit-identical; within touched
    tensors, untouched rows/columns must be identical and touched slices must
    carry the spec's cumulative scale (median ratio over non-tiny elements).
    """
    import torch

    plan = _plan(spec, layout)
    sets = _sets(spec, layout)
    for key in base:
        if key not in edited:
            raise SpecMismatch(f"{key} missing from edited checkpoint")
        a, b = base[key], edited[key]
        if key not in plan and key not in sets:
            if not torch.equal(a, b):
                raise SpecMismatch(f"{key} changed but the spec does not touch it")
            continue
        for (axis, idx), values in sets.get(key, {}).items():
            want = torch.tensor(values, dtype=torch.float32).to(b.dtype)
            if not torch.equal(_slice(b, axis, idx), want):
                raise SpecMismatch(
                    f"{key} {axis} {idx} does not hold the spec's rewritten values"
                )
        slots: dict[tuple[str, int], Any] = {**plan.get(key, {}), **sets.get(key, {})}
        diff = a != b
        for i in torch.nonzero(diff.any(dim=1)).flatten().tolist():
            if ("row", i) not in slots and not any(ax == "col" for ax, _ in slots):
                raise SpecMismatch(
                    f"{key} row {i} changed but the spec does not name it"
                )
        for j in torch.nonzero(diff.any(dim=0)).flatten().tolist():
            if ("col", j) not in slots and not any(ax == "row" for ax, _ in slots):
                raise SpecMismatch(
                    f"{key} column {j} changed but the spec does not name it"
                )
        for (axis, idx), expected in plan.get(key, {}).items():
            av = (a[idx] if axis == "row" else a[:, idx]).float()
            bv = (b[idx] if axis == "row" else b[:, idx]).float()
            mask = av.abs() > 1e-3
            if not bool(mask.any()):
                continue
            got = float((bv[mask] / av[mask]).median())
            if abs(got - expected) > tolerance:
                raise SpecMismatch(
                    f"{key} {axis} {idx}: scale {got:.4f} != expected {expected:.4f}"
                )


def matched_control(
    spec: EditSpec,
    *,
    rng: random.Random,
    vocab_size: int,
    n_neurons: int,
    token_pool: Sequence[int] | None = None,
) -> EditSpec:
    """Same edit kinds, layers and scales on randomly chosen targets.

    Token targets are drawn from ``token_pool`` (e.g. tokens frequency-matched to
    the originals) or the whole vocabulary; neuron-column targets are drawn from
    the same layer, excluding the original neurons. A :class:`NeuronRewrite`
    keeps its neuron and read vectors and gets a random write direction of the
    same norm.
    """
    originals = {i for e in spec.edits if isinstance(e, TokenRowScale) for i in e.ids}
    pool = list(token_pool) if token_pool is not None else list(range(vocab_size))
    pool = [t for t in pool if t not in originals] or pool
    edits: list[Edit] = []
    for e in spec.edits:
        if isinstance(e, TokenRowScale):
            edits.append(
                TokenRowScale(ids=tuple(rng.choice(pool) for _ in e.ids), scale=e.scale)
            )
        elif isinstance(e, NeuronRewrite):
            # Same neuron, same read (gate/up); a random write direction of the
            # same norm — tests what the neuron writes, not that one was written.
            noise = [rng.gauss(0.0, 1.0) for _ in e.down]
            ratio = math.sqrt(sum(x * x for x in e.down)) / math.sqrt(
                sum(x * x for x in noise)
            )
            edits.append(
                NeuronRewrite(
                    layer=e.layer,
                    neuron=e.neuron,
                    gate=e.gate,
                    up=e.up,
                    down=tuple(x * ratio for x in noise),
                )
            )
        else:
            candidates = [j for j in range(n_neurons) if j not in e.neurons]
            edits.append(
                NeuronColumnScale(
                    layer=e.layer,
                    neurons=tuple(rng.sample(candidates, len(e.neurons))),
                    scale=e.scale,
                )
            )
    return EditSpec(name=f"control:{spec.name}", edits=tuple(edits))
