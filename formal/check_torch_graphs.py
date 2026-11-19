"""Translate small ATen graphs from the installed trainers into Lean equations.

Run after `lake build RLMath`. The input tensors select a fixed shape and dtype.
The resulting Lean inputs range over all real values. Each supported ATen node
has an explicit exact-arithmetic interpretation. Unrecognized nodes fail closed.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterator
from fractions import Fraction
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
from torch._decomp import get_decompositions
from torch.fx import Node
from torch.fx.experimental.proxy_tensor import make_fx
from torch.optim.rmsprop import rmsprop

from stateset_agents.core.enhanced.advanced_rl_algorithms import A2CConfig, A2CTrainer
from stateset_agents.rewards.bayesian_reward_model import _paired_gaussian_nll
from stateset_agents.training import objectives, rl_losses
from stateset_agents.training.causal_likelihood import _small_kl_gap
from stateset_agents.training.continual_learning import ContinualLearningManager
from stateset_agents.training.decision_transformer import _squared_error_sum
from stateset_agents.training.offline_grpo_trainer import _capped_policy_weight
from stateset_agents.training.offline_rl_algorithms import (
    ConservativeQLearning,
    ImplicitQLearning,
)
from stateset_agents.training.offline_rl_targets import _mix
from stateset_agents.training.preference_loss import margin_loss
from stateset_agents.training.sim_to_real import DomainAdaptationModule

FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def numeral(value: int | float) -> str:
    """Encode the actual binary literal supplied to ATen as an exact rational."""
    fraction = Fraction(value)
    return f"(({fraction.numerator} : ℝ) / {fraction.denominator})"


def operation(node: Node, terms: dict[Node, str]) -> str:
    """Interpret one scalar ATen operation. Singleton reshape and mean are identities."""
    op = str(node.target)
    args = node.args

    def term(i: int) -> str:
        arg = args[i]
        if isinstance(arg, Node):
            return terms[arg]
        if type(arg) in (int, float):
            return numeral(arg)
        raise TypeError(f"unsupported operand in {node.format_node()}")

    arities = {
        "aten.scalar_tensor.default": (1,),
        "aten.ones_like.default": (1,),
        "aten.zeros_like.default": (1,),
        "aten.detach.default": (1,),
        "aten.view.default": (2,),
        "aten.unsqueeze.default": (2,),
        "aten.squeeze.dim": (2,),
        "aten.mean.default": (1,),
        "aten.sum.default": (1,),
        "aten.expand.default": (2,),
        "aten.exp.default": (1,),
        "aten.clone.default": (1,),
        "aten.abs.default": (1,),
        "aten.sgn.default": (1,),
        "aten.bitwise_not.default": (1,),
        "aten.bitwise_and.Tensor": (2,),
        "aten.ne.Scalar": (2,),
        "aten.log1p.default": (1,),
        "aten.log.default": (1,),
        "aten.sign.default": (1,),
        "aten.lt.Scalar": (2,),
        "aten.new_zeros.default": (2,),
        "aten.sqrt.default": (1,),
        "aten.addcdiv.default": (3,),
        "aten.addcmul.default": (3,),
        "aten.expm1.default": (1,),
        "aten.neg.default": (1,),
        "aten.clamp.default": (2, 3),
        "aten.clamp_min.default": (2,),
        "aten.clamp_max.default": (2,),
        "aten.clamp.Tensor": (3,),
        "aten.maximum.default": (2,),
        "aten._to_copy.default": (1,),
        "aten.lerp.Scalar": (3,),
        "aten.signbit.default": (1,),
        "aten.ne.Tensor": (2,),
        "aten.minimum.default": (2,),
        "aten.where.self": (3,),
        "aten.pow.Tensor_Scalar": (2,),
        "aten.gt.Scalar": (2,),
        "aten.gt.Tensor": (2,),
        "aten.ge.Tensor": (2,),
        "aten.le.Tensor": (2,),
        "aten.lt.Tensor": (2,),
        "aten.le.Scalar": (2,),
        "aten.ge.Scalar": (2,),
        "aten.eq.Tensor": (2,),
        "aten.eq.Scalar": (2,),
        "aten.div.Scalar": (2,),
        "aten.div.Tensor": (2,),
        "aten.logical_and.default": (2,),
        "aten.masked_fill.Scalar": (3,),
        "aten.add.Tensor": (2,),
        "aten.add.Scalar": (2,),
        "aten.mul.Tensor": (2,),
        "aten.mul.Scalar": (2,),
        "aten.sub.Tensor": (2,),
    }
    if len(args) not in arities.get(op, ()):
        raise ValueError(f"unsupported operator or arguments in {node.format_node()}")
    if op == "aten.scalar_tensor.default":
        # Include rounding to the constant tensor's dtype before real lifting.
        value = torch.tensor(args[0], dtype=node.meta["val"].dtype).item()
        return numeral(value)
    if op == "aten.new_zeros.default":
        if args[1] != [] or node.kwargs != {"pin_memory": False}:
            raise ValueError(f"unsupported zeros construction in {node.format_node()}")
        return "(0 : ℝ)"
    if op == "aten.zeros_like.default":
        if (
            node.kwargs != {"pin_memory": False}
            or node.meta["val"].dtype not in FLOAT_DTYPES
        ):
            raise ValueError(f"unsupported zero tensor in {node.format_node()}")
        return "(0 : ℝ)"
    if op == "aten.ones_like.default":
        return "true" if node.meta["val"].dtype == torch.bool else "(1 : ℝ)"
    if op == "aten._to_copy.default":
        if (
            set(node.kwargs) - {"dtype", "device", "layout"}
            or node.kwargs.get("dtype") not in FLOAT_DTYPES
            or node.kwargs.get("device", torch.device("cpu")) != torch.device("cpu")
            or node.kwargs.get("layout", torch.strided) != torch.strided
            or args[0].meta["val"].dtype not in FLOAT_DTYPES
        ):
            raise ValueError(f"unsupported conversion in {node.format_node()}")
        return term(0)  # Exact-real interpretation of a floating-to-floating cast.
    if op in ("aten.addcmul.default", "aten.addcdiv.default"):
        weight = node.kwargs.get("value", 1)
        if set(node.kwargs) - {"value"} or type(weight) not in (
            int,
            float,
        ):
            raise ValueError(f"unsupported scalar weight in {node.format_node()}")
        sign = "*" if op == "aten.addcmul.default" else "/"
        return f"({term(0)} + {numeral(weight)} * ({term(1)} {sign} {term(2)}))"
    if node.kwargs:
        raise ValueError(f"unsupported keyword arguments in {node.format_node()}")
    if op in (
        "aten.clone.default",
        "aten.detach.default",
        "aten.view.default",
        "aten.unsqueeze.default",
        "aten.squeeze.dim",
        "aten.mean.default",
        "aten.expand.default",
        "aten.sum.default",
    ):
        return term(0)
    if op == "aten.abs.default":
        return f"(abs {term(0)})"
    if op == "aten.log.default":
        return f"(Real.log {term(0)})"
    if op == "aten.bitwise_not.default" and args[0].meta["val"].dtype == torch.bool:
        return f"(¬ {term(0)})"
    if op == "aten.bitwise_and.Tensor" and all(
        a.meta["val"].dtype == torch.bool for a in args
    ):
        return f"({term(0)} ∧ {term(1)})"
    if op in ("aten.sign.default", "aten.sgn.default"):
        return f"(if {term(0)} < 0 then (-1:ℝ) else if {term(0)} = 0 then 0 else 1)"
    if op == "aten.log1p.default" and str(args[0].target) == "aten.exp.default":
        return f"(Real.log (1 + {term(0)}))"
    if op == "aten.sqrt.default":
        return f"(Real.sqrt {term(0)})"
    if op == "aten.exp.default":
        return f"(Real.exp {term(0)})"
    if op == "aten.expm1.default":
        return f"(Real.exp {term(0)} - 1)"
    if op == "aten.neg.default":
        return f"(-{term(0)})"
    if op == "aten.lerp.Scalar":
        return f"({term(0)} + {term(2)} * ({term(1)} - {term(0)}))"
    if op == "aten.signbit.default":
        return f"({term(0)} < 0)"
    if op == "aten.maximum.default":
        return f"(max {term(0)} {term(1)})"
    if op in ("aten.clamp.default", "aten.clamp.Tensor"):
        lo = args[1] if len(args) >= 2 else None
        hi = args[2] if len(args) >= 3 else None
        value = term(0)
        if lo is not None:
            value = f"(max {term(1)} {value})"
        if hi is not None:
            value = f"(min {value} {term(2)})"
        return value
    if op == "aten.minimum.default":
        return f"(min {term(0)} {term(1)})"
    if op in ("aten.clamp_min.default", "aten.clamp_max.default"):
        bound = "max" if op == "aten.clamp_min.default" else "min"
        return f"({bound} {term(0)} {term(1)})"
    if op == "aten.where.self":
        return f"(if {term(0)} then {term(1)} else {term(2)})"
    if op == "aten.masked_fill.Scalar":
        return f"(if {term(1)} then {term(2)} else {term(0)})"
    if op in ("aten.div.Scalar", "aten.div.Tensor") and (
        type(args[1]) not in (int, float) or args[1] == 0
    ):
        den = args[1]
        positive = (
            isinstance(den, Node)
            and str(den.target) == "aten.add.Tensor"
            and len(den.args) == 2
            and den.args[1] == 1
            and not den.kwargs
            and isinstance(den.args[0], Node)
            and str(den.args[0].target) == "aten.exp.default"
        )
        if isinstance(den, Node) and str(den.target) == "aten.masked_fill.Scalar":
            # Lean must discharge the nonzero obligation on every branch.
            checked = (
                f"(⟨{term(1)}, by split_ifs <;> simp_all⟩ : {{v : ℝ // v ≠ 0}}).val"
            )
            return f"({term(0)} / {checked})"
        if not positive:
            raise ValueError("scalar division requires a proven nonzero denominator")
    if op == "aten.pow.Tensor_Scalar":
        power = args[1]
        if type(power) in (int, float) and power >= 0 and int(power) == power:
            return f"({term(0)} ^ {int(power)})"
    infix = {
        "aten.gt.Scalar": ">",
        "aten.gt.Tensor": ">",
        "aten.ge.Tensor": "≥",
        "aten.le.Tensor": "≤",
        "aten.lt.Tensor": "<",
        "aten.lt.Scalar": "<",
        "aten.le.Scalar": "≤",
        "aten.ge.Scalar": "≥",
        "aten.ne.Tensor": "≠",
        "aten.ne.Scalar": "≠",
        "aten.eq.Tensor": "=",
        "aten.eq.Scalar": "=",
        "aten.logical_and.default": "∧",
        "aten.div.Scalar": "/",
        "aten.div.Tensor": "/",
        "aten.add.Tensor": "+",
        "aten.add.Scalar": "+",
        "aten.mul.Tensor": "*",
        "aten.mul.Scalar": "*",
        "aten.sub.Tensor": "-",
    }
    if op in infix:
        return f"({term(0)} {infix[op]} {term(1)})"
    raise ValueError(f"unsupported operation in {node.format_node()}")


def tensor_terms(node: Node, terms: dict[Node, tuple[str, ...]]) -> tuple[str, ...]:
    """Interpret singletons and a two-proposal axis with scalar broadcasting."""
    op = str(node.target)
    count = node.meta["val"].numel()
    if op in ("aten.amax.default", "aten.logsumexp.default"):
        source, *options = node.args
        expected = [[1], True] if op == "aten.amax.default" else [[1]]
        if (
            options != expected
            or node.kwargs
            or count != 1
            or tuple(source.meta["val"].shape) != (1, 2, 1)
        ):
            raise ValueError(f"unsupported reduction in {node.format_node()}")
        a, b = terms[source]
        if op == "aten.amax.default":
            return (f"(max {a} {b})",)
        return (f"(Real.log (Real.exp {a} + Real.exp {b}))",)

    if count == 2:
        if tuple(node.meta["val"].shape) != (1, 2, 1) or op not in (
            "aten.sub.Tensor",
            "aten.mul.Tensor",
            "aten.div.Tensor",
            "aten.exp.default",
        ):
            raise ValueError(f"unsupported vector operation in {node.format_node()}")
    elif count != 1:
        raise ValueError(f"unsupported shape in {node.format_node()}")
    if any(len(terms[arg]) > count for arg in node.all_input_nodes):
        raise ValueError(f"unsupported reduction in {node.format_node()}")
    return tuple(
        operation(
            node,
            {
                arg: terms[arg][0 if len(terms[arg]) == 1 else i]
                for arg in node.all_input_nodes
            },
        )
        for i in range(count)
    )


def equation(
    name: str,
    function: Callable,
    inputs: tuple[torch.Tensor, ...],
    specification: str,
    proof: str,
    hypotheses: str = "",
    decompositions: dict | None = None,
) -> str:
    """Trace the production function and produce a universally quantified example."""
    graph = make_fx(function, decomposition_table=decompositions)(*inputs).graph
    terms: dict[Node, tuple[str, ...]] = {}
    binders = []
    placeholders = iter(inputs)
    for node in graph.nodes:
        if node.op == "placeholder":
            value = next(placeholders)
            if (
                value.numel() != 1 and tuple(value.shape) != (1, 2, 1)
            ) or value.dtype not in (*FLOAT_DTYPES, torch.bool):
                raise ValueError(f"unsupported input to {name}")
            terms[node] = tuple(f"x{len(binders)+i}" for i in range(value.numel()))
            kind = "Bool" if value.dtype == torch.bool else "ℝ"
            binders.extend(f"({label} : {kind})" for label in terms[node])
        elif node.op == "call_function":
            terms[node] = tensor_terms(node, terms)
        elif node.op == "output":
            result = terms[node.args[0]]
            lhs = result[0] if len(result) == 1 else f"({', '.join(result)})"
            return (
                f"theorem {name} {' '.join(binders)} {hypotheses} :\n"
                f"  {lhs} = {specification} := by\n  {proof}\n"
            )
        else:
            raise ValueError(f"unsupported FX node {node.format_node()}")
    raise ValueError(f"missing output in graph for {name}")


def equations() -> Iterator[str]:
    """Independent specifications for forward and generated backward graphs."""

    def real() -> torch.Tensor:
        return torch.tensor(1.0, dtype=torch.float64)

    yield equation(
        "klSeries",
        _small_kl_gap,
        (real(),),
        "RegularizedBatch.smallGap x0",
        "unfold RegularizedBatch.smallGap\n  ring",
    )
    yield equation(
        "klSeriesSlope",
        torch.func.grad(_small_kl_gap),
        (real(),),
        "RegularizedBatch.smallSlope x0",
        "unfold RegularizedBatch.smallSlope\n  ring",
    )

    reward_inputs = (real().reshape(1, 1), real().reshape(1, 1), real().reshape(1))
    for name, fn, spec in (
        ("Value", _paired_gaussian_nll, "BayesianReward.nll x2 x0 x1"),
        ("MeanSlope", torch.func.grad(_paired_gaussian_nll), "(x0-x2)*Real.exp (-x1)"),
        (
            "LogVarianceSlope",
            torch.func.grad(_paired_gaussian_nll, argnums=1),
            "(1-(x0-x2)^2*Real.exp (-x1))/2",
        ),
    ):
        yield equation(
            "gaussianReward" + name,
            fn,
            reward_inputs,
            spec,
            "by_cases tail : |x1| > 1400 <;> rcases lt_trichotomy (x0-x2) 0 with h | h | h <;> "
            "simp [BayesianReward.nll, tail, h, ne_of_lt, ne_of_gt, abs_of_neg, abs_of_pos, *] <;> "
            "ring_nf <;> simp (disch := linarith) only [BayesianReward.squared_precision_factor, "
            "sub_eq_add_neg, mul_two, Real.exp_add, Real.exp_log, if_neg, if_pos] <;> "
            "field_simp (disch := linarith) <;> ring_nf\n  "
            "all_goals first | rw [Real.exp_log (by linarith : 0 < x2-x0)] | "
            "rw [Real.exp_log (by linarith : 0 < x0-x2)]\n  all_goals linarith",
        )

    dt_error = partial(_squared_error_sum, steps=3)
    yield equation("dtSquaredError", dt_error, (real(),), "x0^2/3", "ring")
    yield equation(
        "dtSquaredSlope", torch.func.grad(dt_error), (real(),), "2*x0/3", "ring"
    )

    yield equation(
        "loggedPolicyWeight",
        partial(_capped_policy_weight, temperature=2.0, cap=1.0),
        (real(),),
        "min (Real.exp (x0/2)) 1",
        "convert Offline.saturated_weight (x := x0/2) (cap := 1) (by norm_num) using 1 <;> norm_num",
    )

    log_sigmoid = get_decompositions(
        [torch.ops.aten.log_sigmoid_forward, torch.ops.aten.log_sigmoid_backward]
    )

    def domain_loss(
        source: torch.Tensor, target: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        adapter = SimpleNamespace(
            config=SimpleNamespace(state_dim=1, gradient_reversal_lambda=0.5),
            domain_critic=lambda features: features + bias,
        )
        return DomainAdaptationModule.domain_adversarial_loss(adapter, source, target)

    domain_inputs = (real().reshape(1, 1), real().reshape(1, 1), real())
    sim = "(x0+x2)"
    target = "(x1+x2)"
    yield equation(
        "domainBalancedLoss",
        domain_loss,
        domain_inputs,
        f"DomainAdversarial.loss false {sim} + DomainAdversarial.loss true {target}",
        "unfold DomainAdversarial.loss\n  simp only [Bool.false_eq_true, if_false, if_true]\n"
        "  linarith [Preference.stable_loss (-(x0+x2)), Preference.stable_loss (x1+x2)]",
        decompositions=log_sigmoid,
    )
    sim_residual = f"DomainAdversarial.residual false {sim}"
    real_residual = f"DomainAdversarial.residual true {target}"
    for index, name, expected in (
        (0, "domainSimGradient", f"-(1/2)*{sim_residual}"),
        (1, "domainRealGradient", f"-(1/2)*{real_residual}"),
        (2, "domainCriticGradient", f"{sim_residual} + {real_residual}"),
    ):
        yield equation(
            name,
            torch.func.grad(domain_loss, argnums=index),
            domain_inputs,
            expected,
            "unfold DomainAdversarial.residual\n"
            "  simp only [Bool.false_eq_true, if_false, if_true, div_one, one_mul, mul_one]\n"
            "  simp only [Preference.stable_slope]\n  ring",
            decompositions=log_sigmoid,
        )
    yield equation(
        "dpoValue",
        margin_loss,
        (real(),),
        "Policy.preferenceLoss 1 x0",
        "convert Preference.stable_loss x0 using 1 <;> norm_num",
        decompositions=log_sigmoid,
    )
    yield equation(
        "dpoSlope",
        torch.func.grad(margin_loss),
        (real(),),
        "-1/(1+Real.exp x0)",
        "convert Preference.stable_slope x0 using 1 <;> norm_num",
        decompositions=log_sigmoid,
    )

    small = f"abs x0 < {numeral(0.001)}"
    yield equation(
        "k3Value",
        rl_losses.k3_term,
        (real(),),
        f"if {small} then RegularizedBatch.smallGap x0 else Policy.k3 x0",
        "unfold Policy.k3 RegularizedBatch.smallGap\n  split_ifs <;> ring",
    )
    yield equation(
        "k3Slope",
        torch.func.grad(rl_losses.k3_term),
        (real(),),
        f"if {small} then RegularizedBatch.smallSlope x0 else Real.exp x0 - 1",
        "unfold RegularizedBatch.smallSlope\n  split_ifs <;> ring",
    )
    for name, function, target in (
        ("weightedK3", rl_losses._scaled_k3_tail, "x1 * Policy.k3 x0"),
        (
            "weightedK3Slope",
            torch.func.grad(rl_losses._scaled_k3_tail),
            "x1 * (Real.exp x0 - 1)",
        ),
    ):
        yield equation(
            name,
            function,
            (real(), real()),
            target,
            "simp [Policy.k3, abs_of_pos hw, not_lt.mpr hw.le, hw.ne', Real.exp_add, Real.exp_log hw]\n  ring",
            hypotheses="(hw : 0 < x1)",
        )
    # ATen clamp is min(max(x, lo), hi). Use the bounds to prove equivalence
    # with the independently defined max(lo, min(hi, x)).
    clamp_proof = (
        "unfold Policy.ratio\n  rw [min_max_distrib_right, min_comm x0]\n  norm_num"
    )
    yield equation(
        "safeRatio",
        lambda x: rl_losses.safe_exp_ratio(x),
        (real(),),
        "Policy.ratio 20 x0",
        clamp_proof,
    )
    yield equation(
        "ppoClip",
        lambda r, a: rl_losses.clipped_surrogate(r, a, clip_low=0.125, clip_high=0.25),
        (real(), real()),
        "Policy.surrogate (7/8) (5/4) x0 x1",
        "unfold Policy.surrogate\n  rw [min_max_distrib_right, min_comm x0]\n  norm_num",
    )

    def ppo_clip(r: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return rl_losses.clipped_surrogate(r, a, clip_low=0.125, clip_high=0.25)

    yield equation(
        "ppoBackward",
        torch.func.functionalize(torch.func.grad(ppo_clip)),
        (real(), real()),
        "(if (x0 < 7/8 ∧ x1 < 0) ∨ (5/4 < x0 ∧ 0 < x1) then 0 else -x1)",
        "rw [← Policy.clip_backward_gate (lo := 7/8) (hi := 5/4) x0 x1 (by norm_num)]\n  norm_num [Policy.clipBackward]",
    )

    def cispo(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return objectives.surrogate(objectives.OBJECTIVES["cispo"], x.exp(), a, x)

    yield equation(
        "cispoBackward",
        torch.func.grad(cispo),
        (real(), real()),
        "-min (Real.exp x0) 5 * x1",
        "norm_num",
    )
    yield equation(
        "gspoTokenBackward",
        torch.func.grad(rl_losses.sequence_token_ratio, argnums=1),
        (real(), real()),
        "x0",
        "simp",
    )

    def expectile(v: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return ImplicitQLearning._expectile_loss(None, v, y, 0.75)

    yield equation(
        "expectileValue",
        expectile,
        (real(), real()),
        "Returns.expectile (3/4) (x1-x0)",
        "norm_num [Returns.expectile]\n  ring",
    )
    yield equation(
        "expectileBackward",
        torch.func.grad(expectile),
        (real(), real()),
        "deriv (fun v => Returns.expectile (3/4) (x1-v)) x0",
        "rw [(Offline.expectile_derivative (3/4) x1 x0).deriv]\n"
        "  by_cases h : 0 < x1-x0 <;> simp [h] <;> ring",
    )
    trainer = SimpleNamespace(config=SimpleNamespace(temperature=0.5))

    def cql(q: torch.Tensor, logged: torch.Tensor) -> torch.Tensor:
        gap = ConservativeQLearning._soft_max(
            trainer, q, normalize=False, baseline=logged
        )
        return gap.sum()

    inputs = (torch.zeros(1, 2, 1, dtype=torch.float64), real().reshape(1, 1))
    table = "(![x0, x1] : Fin 2 → ℝ)"
    center = "(max x0 x1)"
    pair_simp = "Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one"
    yield equation(
        "cqlValue",
        cql,
        inputs,
        f"Offline.cqlGap (1/2) {table} x2",
        f"simpa [{pair_simp}, mul_comm] using "
        f"(Offline.cqlGap_centered (t := 1/2) (by norm_num) {table} x2 {center}).symm",
    )
    z = "(Real.exp (x0/(1/2)) + Real.exp (x1/(1/2)))"
    yield equation(
        "cqlProposalBackward",
        torch.func.grad(cql),
        inputs,
        f"(Real.exp (x0/(1/2))/{z}, Real.exp (x1/(1/2))/{z})",
        "apply Prod.ext\n"
        + "\n".join(
            f"  · have h := Offline.softmax_centered (1/2) {table} {center} {i}\n"
            f"    simp only [{pair_simp}] at h\n    norm_num at h ⊢\n    linarith only [h]"
            for i in (0, 1)
        ),
    )
    yield equation(
        "cqlLoggedBackward",
        torch.func.grad(cql, argnums=1),
        inputs,
        "(-1 : ℝ)",
        "norm_num",
    )
    yield equation(
        "bellmanValue",
        lambda r, done, v: rl_losses.bellman_target(r, done, v, 0.5),
        (real().reshape(1), torch.tensor([False]), real().reshape(1, 1)),
        "Returns.backup (1/2) x0 x2 x1",
        "cases x1 <;> simp [Returns.backup]\n  ring",
    )

    gamma = 1.0 - 2.0**-30

    def configured_backup(r, done, v):
        return rl_losses.bellman_target(r, done, v, gamma).sum()

    for dtype, suffix in (
        (torch.float32, ""),
        (torch.float16, "Half"),
        (torch.bfloat16, "BFloat"),
    ):
        td_inputs = (
            real().to(dtype).reshape(1),
            torch.tensor([False]),
            real().to(dtype).reshape(1),
        )
        for label, function, expected in (
            ("Value", configured_backup, f"Returns.backup {numeral(gamma)} x0 x2 x1"),
            (
                "NextSlope",
                torch.func.grad(configured_backup, argnums=2),
                f"(if x1 then 0 else {numeral(gamma)})",
            ),
            ("RewardSlope", torch.func.grad(configured_backup), "(1 : ℝ)"),
        ):
            simplify = (
                "simp only [BellmanRounding.reflected_clip, ne_self_iff_false, false_and, if_false]\n  "
                if label == "Value"
                else ""
            )
            yield equation(
                "configuredBellman" + suffix + label,
                function,
                td_inputs,
                expected,
                simplify + "cases x1 <;> norm_num [Returns.backup] <;> ring",
            )

    for label, function, expected in (
        ("Value", rl_losses.td_regression, "(x0-x1)^2"),
        ("PredictionSlope", torch.func.grad(rl_losses.td_regression), "2*(x0-x1)"),
        ("FrozenTarget", torch.func.grad(rl_losses.td_regression, argnums=1), "(0:ℝ)"),
    ):
        yield equation(
            "tdRegression" + label, function, (real().float(), real()), expected, "ring"
        )

    def no_bootstrap(r, done, v):
        return rl_losses.bellman_target(r, done, v, 0.0).sum()

    inputs = (real().reshape(1), torch.tensor([False]), real().reshape(1, 1))
    yield equation("zeroDiscount", no_bootstrap, inputs, "x0", "norm_num")
    yield equation(
        "zeroDiscountBackward",
        torch.func.grad(no_bootstrap, argnums=2),
        inputs,
        "(0 : ℝ)",
        "norm_num",
    )
    for weight, label in ((0.25, "Quarter"), (0.75, "ThreeQuarters")):
        w = numeral(weight)
        yield equation(
            f"interpolate{label}",
            partial(_mix, weight=weight),
            (real().float(), real().float()),
            f"(1-{w})*x0 + {w}*x1",
            "simp only [min_def, max_def]\n  split_ifs <;> linarith",
        )

    yield equation(
        "ppoValueEnvelope",
        lambda v, old, y: rl_losses.clipped_value_error(v, old, y, 0.25),
        (real(), real(), real()),
        "PPOBatch.valueError x0 x1 x2 (1/4)",
        "unfold PPOBatch.valueError\n  rw [PPOBatch.projected_value]\n"
        "  have h : x1 - 1/4 ≤ x1 + 1/4 := by linarith\n"
        "  rw [min_max_distrib_right, min_eq_left h, min_comm x0]\n  simp only [div_one, mul_one]",
    )

    def critic(
        v: torch.Tensor, old: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return rl_losses.clipped_value_error(v, old, target, 0.25)

    yield equation(
        "vapoCriticSlope",
        torch.func.functionalize(torch.func.grad(critic)),
        (real(), real(), real()),
        "VAPO.criticSlope (x1-1/4) (x1+1/4) x2 x0",
        "unfold VAPO.criticSlope VAPO.project VAPO.maxWeight\n"
        "  have h : x1 - 1/4 ≤ x1 + 1/4 := by linarith\n"
        "  rw [min_max_distrib_right, min_eq_left h, min_comm x0]\n"
        "  simp only [pow_one, mul_one, div_one]\n"
        "  split_ifs <;> first | (exfalso; solve_by_elim [Eq.symm]) | (exfalso; linarith) | ring",
    )

    yield equation(
        "ppoScaledValueEnvelope",
        lambda v, old, y, scale: rl_losses.clipped_value_error(
            v, old, y, 0.25, scale=scale
        ),
        (real(), real(), real(), real()),
        "PPOBatch.valueError x0 x1 x2 (1/4) * x3^2",
        "have h : x1 - 1/4 ≤ x1 + 1/4 := by linarith\n"
        "  rw [min_max_distrib_right, min_eq_left h, min_comm x0]\n"
        "  exact PPOBatch.scaled_value_error x0 x1 x2 (1/4) x3",
    )

    # Trace the scalar CPU kernel used by the configured A2C optimizer. Its
    # incoming second moment is nonnegative for every reachable optimizer state.
    agent = SimpleNamespace(
        tokenizer=[0, 1], model=SimpleNamespace(config=SimpleNamespace(hidden_size=1))
    )
    optimizer = A2CTrainer(agent, A2CConfig(), "cpu").optimizer
    if type(optimizer) is not torch.optim.RMSprop:
        raise ValueError("A2C optimizer changed")
    options = optimizer.defaults

    def optimizer_step(w, grad, square, count):
        param, v, step = w.clone(), square.clone(), count.clone()
        rmsprop([param], [grad], [v], [], [], [step], **options)
        return param, v, step

    step_graph = torch.func.functionalize(optimizer_step)
    alpha, rate, eps = numeral(0.99), numeral(7e-4), numeral(1e-5)
    moment = f"({alpha}*x2 + (1-{alpha})*x1^2)"
    expected = (f"x0 - {rate}*x1/(Real.sqrt {moment}+{eps})", moment, "x3+1")

    def select_output(index):
        def project(w, grad, v, step):
            return step_graph(w, grad, v, step)[index]

        return project

    for i, label in enumerate(("Parameter", "Square", "Counter")):
        yield equation(
            f"a2cRmsprop{label}",
            select_output(i),
            (real(), real(), real(), real()),
            expected[i],
            "ring_nf",
            hypotheses="(_previous_nonnegative : 0 ≤ x2)",
        )

    def ewc(
        weight: torch.Tensor,
        fisher: torch.Tensor,
        anchor: torch.Tensor,
        *,
        strength: float,
    ) -> torch.Tensor:
        model = SimpleNamespace(named_parameters=lambda: [("weight", weight)])
        manager = SimpleNamespace(
            config=SimpleNamespace(uses_ewc=lambda: True, ewc_lambda=strength),
            _ewc_fisher={"weight": fisher},
            _ewc_params={"weight": anchor},
        )
        return ContinualLearningManager.compute_ewc_penalty(
            manager, SimpleNamespace(model=model)
        )

    for suffix, strength, dtype in (
        ("", 0.75, torch.float64),
        ("Small", 0.1875, torch.float64),
        ("Promoted", 0.75, torch.float32),
    ):
        kernel = partial(ewc, strength=strength)
        inputs = tuple(real().to(dtype) for _ in range(3))
        proof = (
            "generalize he : Real.sqrt ((3:ℝ)/8) = r\n  "
            "have hr : r^2 = (3:ℝ)/8 := he ▸ Real.sq_sqrt (by norm_num)\n  "
            "try simp only [max_def, min_def]\n  split_ifs <;> ring_nf <;> "
            "simp only [Real.sq_sqrt hf, hr] <;> ring"
            if dtype == torch.float64
            else "ring"
        )
        for label, fn, expression, prelude in (
            (
                "Value",
                kernel,
                f"GradientBank.ewc {numeral(strength)} x1 x2 x0",
                "unfold GradientBank.ewc\n  ",
            ),
            (
                "Backward",
                torch.func.grad(kernel),
                f"{numeral(strength)} * x1 * (x0 - x2)",
                "",
            ),
        ):
            yield equation(
                f"ewc{suffix}{label}",
                fn,
                inputs,
                expression,
                prelude + proof,
                hypotheses="(hf : 0 ≤ x1)" if dtype == torch.float64 else "",
            )


if __name__ == "__main__":
    checks = list(equations())
    source = (
        "import BellmanRounding\nimport BayesianReward\nimport VAPO\nimport DomainAdversarial\nimport RegularizedBatch\nimport Preference\nimport PPOBatch\nimport GradientBank\nimport OfflineMath\nopen StateSetFormal\n"
        + "\n".join(checks)
    )
    result = subprocess.run(
        ["lake", "env", "lean", "--stdin"],
        cwd=Path(__file__).resolve().parents[1],
        input=source,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise SystemExit(source + "\n" + result.stdout + result.stderr)
    print(
        f"Lean checked {len(checks)} real-arithmetic equations extracted from PyTorch."
    )
