"""Explicit-loop reference implementations of every objective component.

Deliberately naive: Python lists of floats, no broadcasting, no torch. The
production code in ``stateset_agents.training.objectives`` must agree with
these to 1e-6. Keep this file boring — it is the oracle.
"""

from __future__ import annotations

import math


def _std0(xs: list[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def ref_advantages(
    rewards: list[float], groups: list[int], kind: str, eps: float
) -> list[float]:
    n = len(rewards)
    out = [0.0] * n
    ids = sorted(set(groups))
    for gid in ids:
        idx = [i for i in range(n) if groups[i] == gid]
        rg = [rewards[i] for i in idx]
        if len(rg) <= 1:
            continue  # group of one -> advantage 0
        mean = sum(rg) / len(rg)
        if kind == "leave_one_out":
            for i, r in zip(idx, rg, strict=True):
                out[i] = r - (sum(rg) - r) / (len(rg) - 1)
            continue
        centred = [r - mean for r in rg]
        if kind == "group_mean":
            for i, a in zip(idx, centred, strict=True):
                out[i] = a
        elif kind == "group_norm":
            s = _std0(centred)
            if not math.isfinite(s) or s <= eps:
                continue
            for i, a in zip(idx, centred, strict=True):
                out[i] = a / (s + eps)
        elif kind == "batch_norm":
            for i, a in zip(idx, centred, strict=True):
                out[i] = a
        else:
            raise ValueError(kind)
    if kind == "batch_norm":
        s = _std0(rewards)
        if n <= 1 or not math.isfinite(s) or s <= eps:
            return [0.0] * n
        out = [a / (s + eps) for a in out]
    return out


def _seq_len(mask_row: list[float]) -> float:
    return max(sum(mask_row), 1.0)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def ref_ratio(
    kind: str,
    logp_cur: list[list[float]],
    logp_old,  # list[list[float]] per-token, or list[float] sums
    mask: list[list[float]],
    groups: list[int] | None,
    clamp: float,
) -> list[list[float]]:
    """Return an [N][T] table (sequence-level ratios are repeated over T)."""
    n, t = len(logp_cur), len(logp_cur[0])
    per_token_old = isinstance(logp_old[0], list)
    if kind == "token":
        assert per_token_old
        return [
            [
                math.exp(_clamp(logp_cur[i][j] - logp_old[i][j], -clamp, clamp))
                for j in range(t)
            ]
            for i in range(n)
        ]
    if kind in ("sequence", "sequence_token"):
        out = []
        for i in range(n):
            if per_token_old:
                s = sum(
                    (logp_cur[i][j] - logp_old[i][j]) * mask[i][j] for j in range(t)
                )
            else:
                s = sum(logp_cur[i][j] * mask[i][j] for j in range(t)) - logp_old[i]
            r = math.exp(_clamp(s / _seq_len(mask[i]), -clamp, clamp))
            out.append([r] * t)
        return out
    if kind == "group_expectation":
        assert not per_token_old and groups is not None
        out = [[0.0] * t for _ in range(n)]
        for gid in sorted(set(groups)):
            idx = [i for i in range(n) if groups[i] == gid]
            q = [logp_old[i] for i in idx]
            m2 = max(2 * x for x in q)
            m1 = max(q)
            lse2 = m2 + math.log(sum(math.exp(2 * x - m2) for x in q))
            lse1 = m1 + math.log(sum(math.exp(x - m1) for x in q))
            log_e = lse2 - lse1
            for i in idx:
                s = sum(logp_cur[i][j] * mask[i][j] for j in range(t))
                out[i] = [math.exp(_clamp(s - log_e, -clamp, clamp))] * t
        return out
    raise ValueError(kind)


def ref_surrogate(
    clip: str,
    ratio: list[list[float]],
    adv: list[list[float]],
    logp_cur: list[list[float]],
    clip_low: float,
    clip_high: float,
    delta: float | None,
    is_cap: float,
) -> list[list[float]]:
    n, t = len(ratio), len(ratio[0])
    out = [[0.0] * t for _ in range(n)]
    for i in range(n):
        for j in range(t):
            r, a = ratio[i][j], adv[i][j]
            if clip == "clipped":
                r1 = r if delta is None else min(r, delta)
                c = _clamp(r, 1 - clip_low, 1 + clip_high)
                out[i][j] = -min(r1 * a, c * a)
            elif clip == "cispo":
                out[i][j] = -min(r, is_cap) * a * logp_cur[i][j]
            elif clip == "none":
                out[i][j] = -r * a
            else:
                raise ValueError(clip)
    return out


def ref_k3(cur: float, ref: float) -> float:
    d = ref - cur
    return math.exp(d) - d - 1.0


def ref_aggregate(
    kind: str, loss: list[list[float]], mask: list[list[float]], max_len: int | None
) -> float:
    n, t = len(loss), len(loss[0])
    if kind == "seq_mean":
        rows = []
        for i in range(n):
            rows.append(
                sum(loss[i][j] * mask[i][j] for j in range(t)) / _seq_len(mask[i])
            )
        return sum(rows) / n
    total = sum(loss[i][j] * mask[i][j] for i in range(n) for j in range(t))
    if kind == "token_mean":
        return total / max(sum(mask[i][j] for i in range(n) for j in range(t)), 1.0)
    if kind == "seq_sum_const":
        assert max_len
        return total / (n * max_len)
    raise ValueError(kind)


def ref_policy_loss(
    obj,
    logp_cur: list[list[float]],
    mask: list[list[float]],
    advantages,  # list[float] or list[list[float]]
    logp_old=None,
    logp_ref=None,
    groups: list[int] | None = None,
    kl_ext: list[list[float]] | None = None,
    entropy: list[list[float]] | None = None,
) -> float:
    n, t = len(logp_cur), len(logp_cur[0])
    if logp_old is None:
        logp_old = [row[:] for row in logp_cur]
    ratio = ref_ratio(obj.ratio, logp_cur, logp_old, mask, groups, obj.ratio_clamp)
    if isinstance(advantages[0], list):
        adv = advantages
    else:
        adv = [[a] * t for a in advantages]
    per_token = ref_surrogate(
        obj.clip,
        ratio,
        adv,
        logp_cur,
        obj.clip_low,
        obj.clip_high,
        obj.delta,
        obj.is_cap,
    )
    if obj.kl_coef > 0 and obj.kl == "k3_token":
        for i in range(n):
            for j in range(t):
                k = ref_k3(logp_cur[i][j], logp_ref[i][j])
                if obj.kl_bias_correction:
                    k *= ratio[i][j]
                per_token[i][j] += obj.kl_coef * k
    if obj.kl_coef > 0 and obj.kl == "external":
        for i in range(n):
            for j in range(t):
                per_token[i][j] += obj.kl_coef * kl_ext[i][j]
    loss = ref_aggregate(obj.aggregate, per_token, mask, obj.max_completion_length)
    if obj.kl_coef > 0 and obj.kl == "k3_sequence":
        rows = []
        for i in range(n):
            length = _seq_len(mask[i])
            cur = sum(logp_cur[i][j] * mask[i][j] for j in range(t)) / length
            ref = (
                sum(logp_ref[i][j] * mask[i][j] for j in range(t)) / length
                if isinstance(logp_ref[0], list)
                else logp_ref[i] / length
            )
            rows.append(ref_k3(cur, ref))
        loss += obj.kl_coef * sum(rows) / n
    if obj.entropy_coef > 0 and entropy is not None:
        num = sum(entropy[i][j] * mask[i][j] for i in range(n) for j in range(t))
        den = max(sum(mask[i][j] for i in range(n) for j in range(t)), 1.0)
        loss -= obj.entropy_coef * num / den
    return loss
