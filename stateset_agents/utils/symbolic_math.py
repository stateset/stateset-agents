"""
Safe expression parsing and evaluation helpers for symbolic RL tasks.
"""

from __future__ import annotations

import ast
import math
import operator
import random
import re
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple

DEFAULT_ALLOWED_FUNCTIONS: Dict[str, Callable[..., float]] = {
    "abs": abs,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
}

DEFAULT_CONSTANTS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def normalize_expression(expression: str) -> str:
    """Normalize expression to Python syntax (e.g., convert ^ to **)."""
    if not expression:
        return ""
    return expression.replace("^", "**")


def sanitize_expression(expression: str) -> str:
    """Strip obvious non-math characters from an expression."""
    if not expression:
        return ""
    expr = expression.strip().strip("`").strip()
    if expr.startswith("$") and expr.endswith("$"):
        expr = expr[1:-1].strip()
    expr = re.sub(r"[^A-Za-z0-9_+\-*/().,^\\s]", "", expr)
    return expr.strip()


def extract_expression(text: str) -> str:
    """Extract a candidate expression from an LLM response."""
    if not text:
        return ""
    cleaned = text.strip()
    code_blocks = re.findall(r"```(?:[a-zA-Z0-9_-]+)?\n(.*?)```", cleaned, re.DOTALL)
    if code_blocks:
        cleaned = code_blocks[0].strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""
    candidate = lines[-1]
    for line in reversed(lines):
        if "=" in line:
            candidate = line.split("=", 1)[1]
            break
    candidate = re.sub(
        r"^(answer|expression|result)\s*[:=]\s*",
        "",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = candidate.strip().strip(";").strip(".")
    return sanitize_expression(candidate)


class _SafeEvaluator(ast.NodeVisitor):
    def __init__(
        self,
        variables: Mapping[str, float],
        functions: Optional[Mapping[str, Callable[..., float]]] = None,
        constants: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.variables = dict(variables)
        self.functions = dict(functions or DEFAULT_ALLOWED_FUNCTIONS)
        self.constants = dict(constants or DEFAULT_CONSTANTS)

    def visit_Expression(self, node: ast.Expression) -> float:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> float:
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        raise ValueError("Unsupported constant type")

    def visit_Num(self, node: ast.Num) -> float:  # pragma: no cover - py<3.8
        return float(node.n)

    def visit_Name(self, node: ast.Name) -> float:
        name = node.id
        if name in self.variables:
            return float(self.variables[name])
        if name in self.constants:
            return float(self.constants[name])
        raise ValueError(f"Unknown symbol: {name}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError("Unsupported unary operator")
        return _UNARY_OPS[op_type](self.visit(node.operand))

    def visit_BinOp(self, node: ast.BinOp) -> float:
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise ValueError("Unsupported binary operator")
        left = self.visit(node.left)
        right = self.visit(node.right)
        return _BIN_OPS[op_type](left, right)

    def visit_Call(self, node: ast.Call) -> float:
        if node.keywords:
            raise ValueError("Keyword arguments are not supported")
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only named functions are supported")
        func_name = node.func.id
        if func_name not in self.functions:
            raise ValueError(f"Unsupported function: {func_name}")
        args = [self.visit(arg) for arg in node.args]
        return float(self.functions[func_name](*args))

    def generic_visit(self, node: ast.AST) -> float:
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def safe_eval_expression(
    expression: str,
    variables: Mapping[str, float],
    allowed_functions: Optional[Mapping[str, Callable[..., float]]] = None,
    constants: Optional[Mapping[str, float]] = None,
) -> float:
    """Safely evaluate a math expression with restricted operations."""
    expr = normalize_expression(expression)
    expr = sanitize_expression(expr)
    if not expr:
        raise ValueError("Empty expression")
    tree = ast.parse(expr, mode="eval")
    evaluator = _SafeEvaluator(
        variables=variables,
        functions=allowed_functions,
        constants=constants,
    )
    value = evaluator.visit(tree)
    if not math.isfinite(value):
        raise ValueError("Non-finite evaluation result")
    return float(value)


def apply_derived_variables(
    values: Mapping[str, float],
    derived_variables: Mapping[str, str],
    allowed_functions: Optional[Mapping[str, Callable[..., float]]] = None,
) -> Dict[str, float]:
    """Compute derived variables in order using existing values."""
    derived = dict(values)
    for name, expr in derived_variables.items():
        derived[name] = safe_eval_expression(expr, derived, allowed_functions)
    return derived


def generate_samples(
    variables: Iterable[str],
    derived_variables: Optional[Mapping[str, str]] = None,
    num_samples: int = 8,
    sample_range: Tuple[float, float] = (-2.0, 2.0),
    seed: Optional[int] = None,
    allowed_functions: Optional[Mapping[str, Callable[..., float]]] = None,
) -> List[Dict[str, float]]:
    """Generate random samples for variables with optional derived relations."""
    rng = random.Random(seed)
    derived_variables = derived_variables or {}
    variables_list = list(variables)
    base_vars = [var for var in variables_list if var not in derived_variables]
    samples: List[Dict[str, float]] = []
    low, high = sample_range
    for _ in range(num_samples):
        values = {var: rng.uniform(low, high) for var in base_vars}
        values = apply_derived_variables(values, derived_variables, allowed_functions)
        for var in variables_list:
            values.setdefault(var, 0.0)
        samples.append(values)
    return samples
