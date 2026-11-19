"""Single-round Bellman targets for batches containing integral source operands."""

from __future__ import annotations

import math
from itertools import repeat
from numbers import Integral
from typing import Any

import torch


def _rounded_dyadic(
    n: int, denominator: int, precision: int, minimum: int, maximum: int
) -> float:
    """Round n/denominator to the destination grid. The denominator is a power of two."""
    bits = denominator.bit_length() - 1
    a = abs(n)
    shift = max(0, a.bit_length() - precision, bits + minimum)
    step = 1 << shift
    q, rem = divmod(a, step)
    if 2 * rem > step or (2 * rem == step and q % 2):
        q += 1
    exponent = shift - bits
    value = (
        math.inf
        if q and q.bit_length() + exponent > maximum
        else math.ldexp(q, exponent)
    )
    return -value if n < 0 else value


def exact_backup(
    reward: Any,
    future: Any,
    discount: float,
    dtype: Any,
    device: Any,
    terminal: Any = None,
) -> Any:
    """Round exact CPU dyadics from a tensor or a heterogeneous reward column."""
    flags = None if terminal is None else terminal.reshape(-1).tolist()
    info = torch.finfo(dtype)
    precision = 2 - math.frexp(info.eps)[1]
    minimum = math.frexp(info.tiny)[1] - precision
    maximum = math.frexp(info.max)[1]
    finite_discount = math.isfinite(discount)
    g, gd = discount.as_integer_ratio() if finite_discount else (0, 1)
    result = []
    labels = reward.reshape(-1).tolist()
    values = repeat(0, len(labels)) if future is None else future.reshape(-1).tolist()
    for i, (r, v) in enumerate(zip(labels, values, strict=True)):
        if isinstance(r, Integral):
            r = int(r)
        reward_only = future is None or (flags is not None and flags[i])
        if reward_only:
            v = 0
        if (
            not finite_discount
            or (not isinstance(r, int) and not math.isfinite(r))
            or (not isinstance(v, int) and not math.isfinite(v))
        ):
            result.append(r if future is None else r + discount * v)
            continue
        rn, rd = r.as_integer_ratio()
        vn, vd = v.as_integer_ratio()
        product_den = gd * vd
        den = max(rd, product_den)
        n = rn * (den // rd) + g * vn * (den // product_den)
        if r == 0 and n == 0:
            # IEEE defines the sign of an exact zero from signed zero terms.
            result.append(r if reward_only else r + discount * v)
        else:
            result.append(_rounded_dyadic(n, den, precision, minimum, maximum))
    return torch.tensor(result, device=device, dtype=dtype).reshape(reward.shape)


class IntegerBellman(torch.autograd.Function):
    """Use exact CPU dyadics for integer labels and keep the native Bellman slopes."""

    @staticmethod
    def forward(
        reward: Any, future: Any, discount: float, dtype: Any, terminal: Any
    ) -> Any:
        return exact_backup(reward, future, discount, dtype, reward.device, terminal)

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> None:
        reward, future, ctx.discount, ctx.output_dtype, _ = inputs
        ctx.reward_dtype, ctx.future_dtype = reward.dtype, future.dtype
        ctx.future_device, ctx.output_device = future.device, output.device

    @staticmethod
    def _scaled(value: Any, gamma: float, dtype: Any, device: Any) -> Any:
        work = "cpu" if value.device.type == "mps" else value.device
        return (value.to(work, torch.float64) * gamma).to(device, dtype)

    @staticmethod
    def backward(ctx: Any, gradient: Any) -> tuple[Any, ...]:
        dr = gradient.to(ctx.reward_dtype) if ctx.needs_input_grad[0] else None
        dv = (
            IntegerBellman._scaled(
                gradient, ctx.discount, ctx.future_dtype, ctx.future_device
            )
            if ctx.needs_input_grad[1]
            else None
        )
        return dr, dv, None, None, None

    @staticmethod
    def jvp(
        ctx: Any, dr: Any, dv: Any, _discount: Any, _dtype: Any, _terminal: Any
    ) -> Any:
        if ctx.reward_dtype.is_floating_point:
            return dr.to(ctx.output_dtype)
        return IntegerBellman._scaled(
            dv, ctx.discount, ctx.output_dtype, ctx.output_device
        )

    @staticmethod
    def vmap(
        info: Any,
        in_dims: Any,
        reward: Any,
        future: Any,
        discount: float,
        dtype: Any,
        terminal: Any,
    ) -> tuple[Any, int]:
        def batched(tensor: Any, dim: int | None) -> Any:
            return (
                tensor.expand(info.batch_size, *tensor.shape)
                if dim is None
                else tensor.movedim(dim, 0)
            )

        mask = None if terminal is None else batched(terminal, in_dims[4])
        return (
            IntegerBellman.apply(
                batched(reward, in_dims[0]),
                batched(future, in_dims[1]),
                discount,
                dtype,
                mask,
            ),
            0,
        )
