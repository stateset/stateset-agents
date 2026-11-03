"""Wait-trigger direction: the decision to re-check ("Wait") vs to continue, matched on context.

Pure functions (testable on CPU) + ``find_wait_directions`` that pod_job calls.
Hypothesis from the base traces: 82% of loop cycles contain "Wait" and "Wait" is
3.5x denser inside loops — the loop is the model repeatedly choosing to
re-verify instead of concluding. Contrast MLP outputs at the position that
PREDICTS "Wait" against positions in the same trace with the SAME preceding
token whose next token is not "Wait".
"""

from __future__ import annotations

import random

WAIT_IDS = (13784, 13428)  # "Wait", " Wait" in the Qwen3.5 tokenizer


def wait_contrast_positions(
    gen: list[int],
    rng: random.Random,
    max_pairs: int = 4,
    wait_ids: tuple[int, ...] = WAIT_IDS,
) -> list[tuple[int, int]]:
    """(wait_pos, control_pos) pairs: gen[pos] is the token BEFORE the decision.

    wait_pos: gen[pos + 1] is a Wait token. control_pos: gen[ctrl] == gen[pos]
    (same preceding token) and gen[ctrl + 1] is not a Wait token.
    """
    by_prev: dict[int, list[int]] = {}
    for t in range(len(gen) - 1):
        if gen[t + 1] not in wait_ids:
            by_prev.setdefault(gen[t], []).append(t)
    waits = [t for t in range(len(gen) - 1) if gen[t + 1] in wait_ids]
    rng.shuffle(waits)
    pairs = []
    for t in waits:
        candidates = by_prev.get(gen[t])
        if candidates:
            pairs.append((t, rng.choice(candidates)))
        if len(pairs) >= max_pairs:
            break
    return pairs


def separation(a, b) -> float:
    """d' of the two samples projected on their mean-difference direction."""
    import torch

    diff = a.mean(0) - b.mean(0)
    u = diff / diff.norm().clamp_min(1e-12)
    pa, pb = a @ u, b @ u
    pooled = torch.sqrt((pa.var() + pb.var()) / 2).clamp_min(1e-12)
    return float((pa.mean() - pb.mean()) / pooled)


def find_wait_directions(
    trace: list[dict],
    session,
    n_layers: int,
    budget,
    max_prefixes: int = 192,
    seed: int = 0,
):
    """One forward pass over all layers' down_proj; returns (directions, scores)."""
    rng = random.Random(seed)
    prefixes: dict[str, list[int]] = {}
    wait_keys, ctrl_keys = [], []
    for r in trace:
        for w, c in wait_contrast_positions(r["gen_ids"], rng):
            kw, kc = f"W{len(prefixes)}", f"C{len(prefixes) + 1}"
            prefixes[kw] = r["prompt_ids"] + r["gen_ids"][: w + 1]
            prefixes[kc] = r["prompt_ids"] + r["gen_ids"][: c + 1]
            wait_keys.append(kw)
            ctrl_keys.append(kc)
    pairs = list(zip(wait_keys, ctrl_keys, strict=True))
    rng.shuffle(pairs)
    pairs = pairs[: max_prefixes // 2]
    wait_keys, ctrl_keys = [p[0] for p in pairs], [p[1] for p in pairs]
    keys = sorted(wait_keys + ctrl_keys, key=lambda k: len(prefixes[k]))
    session.encode = lambda key: prefixes[key]
    modules = [f"model.layers.{L}.mlp.down_proj" for L in range(n_layers)]
    budget.spend(
        f"read: all {n_layers} layers' MLP outputs at {len(keys)} wait/control prefixes (one pass)"
    )
    outs = session.capture_many(modules, keys, reason="wait-trigger read")
    row = {k: i for i, k in enumerate(keys)}
    wi, ci = [row[k] for k in wait_keys], [row[k] for k in ctrl_keys]
    directions, scores = {}, {}
    for L, m in enumerate(modules):
        a, b = outs[m][wi], outs[m][ci]
        directions[L] = (a.mean(0) - b.mean(0)).float()
        scores[L] = round(separation(a, b), 3)
    return directions, scores, {"pairs": len(pairs)}
