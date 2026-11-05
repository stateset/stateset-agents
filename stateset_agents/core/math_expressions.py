"""Safe parsing, exact evaluation and equivalence of short math answers.

Model output never reaches ``eval``: ``sympy.sympify`` and ``parse_expr``
evaluate Python, so a reply such as ``__import__('os').system(...)`` would run
in the trainer. Answers are instead parsed here by a small recursive-descent
parser into a tuple tree (numbers, single-letter symbols, ``+ - * / ^``,
implicit multiplication and a few whitelisted functions) and only that tree is
evaluated:

* :func:`rational_value` evaluates a tree without symbols or functions
  exactly, as a :class:`fractions.Fraction` (integer powers only, with size
  limits so ``9^9^9`` cannot stall the process);
* :func:`equivalent` decides whether two trees denote the same value or
  function: exactly for numbers, by exact evaluation at pseudo-random
  rational points for rational functions (pure Python, no dependencies; the
  points are seeded from a hash of both expressions, so a verdict is
  deterministic but no fixed, public point set exists to craft an answer
  against), and
  through ``sympy.simplify`` on trees built node by node (never parsed from
  text) when the answer uses functions such as ``sqrt`` and sympy is
  installed. The sympy tree obeys the same size limits as exact evaluation
  (``x^(9^9)`` is refused, not built), and the simplification runs in a
  separate worker process with a hard timeout (:data:`SYMPY_TIMEOUT_SECONDS`;
  the worker is killed and replaced when it runs over), so no answer can stall
  the trainer. A timeout decides nothing (``None``), which a verifier scores
  as not equivalent.

:func:`normalize_answer_text` rewrites the common LaTeX answer forms
(``\\frac{3}{4}``, ``\\sqrt{2}``, ``2 \\cdot 3``, ``x^{2}``) into the plain syntax
the parser reads. The calculator tool of
:mod:`stateset_agents.core.tool_registry` and the math verifier of
:mod:`stateset_agents.rewards.math_verifier` share this module.
"""

from __future__ import annotations

import atexit
import functools
import hashlib
import importlib.util
import json
import logging
import os
import queue
import random
import re
import subprocess
import sys
import threading
from collections.abc import Iterable
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import IO, Any

try:
    from .errors import ValidationError as _InputError
except ImportError:  # pragma: no cover - loaded standalone as the sympy worker

    class _InputError(ValueError):  # type: ignore[no-redef]
        """Stand-in base: the worker process loads this file without the
        package, so it has no ``core.errors`` to import."""


logger = logging.getLogger(__name__)

__all__ = [
    "MAX_EXPRESSION_CHARS",
    "Node",
    "ExpressionError",
    "SYMPY_TIMEOUT_SECONDS",
    "equivalent",
    "evaluate_expression",
    "free_symbols",
    "is_numeric_literal",
    "normalize_answer_text",
    "parse_expression",
    "rational_value",
]

#: A parsed expression: ``("num", Fraction)``, ``("sym", name)``,
#: ``("neg", a)``, ``("add"|"sub"|"mul"|"div"|"pow", a, b)`` or
#: ``("call", function, a)``.
Node = tuple[Any, ...]

#: Longest expression text the parser accepts.
MAX_EXPRESSION_CHARS = 256
#: Deepest nesting the parser accepts.
_MAX_DEPTH = 40
#: Largest integer exponent evaluated exactly.
_MAX_EXPONENT = 256
#: Largest operand (bits of numerator or denominator) evaluated exactly.
_MAX_BITS = 4096
#: Points at which two rational functions are compared.
_PROBE_POINTS = 8
#: Largest decimal exponent of a number literal (``1e300`` is fine).
_MAX_DECIMAL_EXPONENT = 400
#: Longer text is not normalized (it cannot parse as an expression anyway).
_MAX_NORMALIZE_CHARS = 4 * MAX_EXPRESSION_CHARS
#: Seconds sympy may spend deciding one equivalence before its worker process
#: is killed; a timeout decides nothing.
SYMPY_TIMEOUT_SECONDS = 5.0
#: Seconds a fresh sympy worker may take to start (importing sympy is slow).
_WORKER_START_SECONDS = 120.0

FUNCTIONS = frozenset({"sqrt", "sin", "cos", "tan", "log", "ln", "exp", "abs"})
CONSTANTS = frozenset({"pi"})

_TOKEN = re.compile(
    r"\s*(?:(?P<num>\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?)"
    r"|(?P<name>[A-Za-z]+)|(?P<op>\*\*|[-+*/^()]))"
)


class ExpressionError(_InputError, ValueError):
    """Text that is not a supported expression, or exceeds the limits."""

    message_only_str = True


def _number(text: str) -> Fraction:
    try:
        value = Decimal(text)
    except (InvalidOperation, ValueError) as exc:  # pragma: no cover - regex
        raise ExpressionError(f"bad number {text!r}") from exc
    # "1e99999999" would build a 300-million-bit integer: refuse it.
    if not value.is_finite() or abs(value.adjusted()) > _MAX_DECIMAL_EXPONENT:
        raise ExpressionError(f"number {text!r} is out of range")
    return Fraction(value)


def _tokenize(text: str) -> list[tuple[str, Any]]:
    if len(text) > MAX_EXPRESSION_CHARS:
        raise ExpressionError("expression is too long")
    tokens: list[tuple[str, Any]] = []
    position = 0
    stripped = text.rstrip()
    while position < len(stripped):
        match = _TOKEN.match(stripped, position)
        if match is None or match.end() == position:
            raise ExpressionError(f"unexpected text at {stripped[position:]!r}")
        position = match.end()
        if match.group("num") is not None:
            tokens.append(("num", _number(match.group("num"))))
        elif match.group("name") is not None:
            tokens.extend(_name_tokens(match.group("name")))
        else:
            op = match.group("op")
            tokens.append(("op", "^" if op == "**" else op))
    return tokens


def _name_tokens(name: str) -> list[tuple[str, Any]]:
    lowered = name.lower()
    if lowered in FUNCTIONS:
        return [("func", "log" if lowered == "ln" else lowered)]
    if lowered in CONSTANTS or len(name) == 1:
        return [("sym", lowered if lowered in CONSTANTS else name)]
    raise ExpressionError(f"unknown name {name!r}")


class _Parser:
    """Recursive descent over ``expr := term (("+"|"-") term)*`` and below."""

    def __init__(self, tokens: list[tuple[str, Any]]) -> None:
        self.tokens = tokens
        self.index = 0
        self.depth = 0

    def peek(self) -> tuple[str, Any] | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def take(self) -> tuple[str, Any]:
        token = self.peek()
        if token is None:
            raise ExpressionError("unexpected end of expression")
        self.index += 1
        return token

    def expect(self, op: str) -> None:
        if self.take() != ("op", op):
            raise ExpressionError(f"expected {op!r}")

    def parse(self) -> Node:
        node = self.expression()
        leftover = self.peek()
        if leftover is not None:
            raise ExpressionError(f"unexpected {leftover[1]!r}")
        return node

    def at_op(self, ops: str) -> str | None:
        """The next token's operator if it is one of ``ops``, else None."""
        token = self.peek()
        if token is not None and token[0] == "op" and token[1] in ops:
            return str(token[1])
        return None

    def nested(self) -> None:
        self.depth += 1
        if self.depth > _MAX_DEPTH:
            raise ExpressionError("expression is nested too deeply")

    def expression(self) -> Node:
        self.nested()
        node = self.term()
        while (op := self.at_op("+-")) is not None:
            self.take()
            node = ("add" if op == "+" else "sub", node, self.term())
        self.depth -= 1
        return node

    def term(self) -> Node:
        node = self.unary()
        while True:
            token = self.peek()
            op = self.at_op("*/")
            if op is not None:
                self.take()
                node = ("mul" if op == "*" else "div", node, self.unary())
            elif token is not None and (
                token[0] in ("num", "sym", "func") or token == ("op", "(")
            ):
                node = ("mul", node, self.power())  # implicit: 2x, 2(x+1)
            else:
                return node

    def unary(self) -> Node:
        sign = self.at_op("+-")
        if sign is not None:
            self.take()
            self.nested()
            operand = self.unary()
            self.depth -= 1
            return ("neg", operand) if sign == "-" else operand
        return self.power()

    def power(self) -> Node:
        base = self.atom()
        if self.at_op("^") is not None:
            self.take()
            self.nested()
            exponent = self.unary()  # right associative: a^b^c = a^(b^c)
            self.depth -= 1
            return ("pow", base, exponent)
        return base

    def atom(self) -> Node:
        kind, value = self.take()
        if kind == "num":
            return ("num", value)
        if kind == "sym":
            return ("sym", value)
        if kind == "func":
            self.expect("(")
            argument = self.expression()
            self.expect(")")
            return ("call", value, argument)
        if (kind, value) == ("op", "("):
            node = self.expression()
            self.expect(")")
            return node
        raise ExpressionError(f"unexpected {value!r}")


def parse_expression(text: str) -> Node:
    """Parse ``text`` (plain syntax; see :func:`normalize_answer_text`).

    Raises :class:`ExpressionError` for anything outside the grammar: unknown
    multi-letter names, stray characters, unbalanced parentheses, more than
    ``MAX_EXPRESSION_CHARS`` characters or deeper than 40 levels.
    """
    tokens = _tokenize(text)
    if not tokens:
        raise ExpressionError("empty expression")
    return _Parser(tokens).parse()


def free_symbols(node: Node) -> set[str]:
    """Symbol names in ``node`` (``pi`` is a constant, not a symbol)."""
    if node[0] == "sym":
        return set() if node[1] in CONSTANTS else {node[1]}
    if node[0] == "num":
        return set()
    if node[0] == "call":
        return free_symbols(node[2])
    return set().union(*(free_symbols(child) for child in node[1:]))


def _exact_only(node: Node) -> bool:
    """No function calls or constants: exactly evaluable once symbols are bound."""
    if node[0] == "call":
        return False
    if node[0] == "sym":
        return node[1] not in CONSTANTS
    if node[0] == "num":
        return True
    return all(_exact_only(child) for child in node[1:])


def is_numeric_literal(node: Node) -> bool:
    """Whether ``node`` is a number written out: ``84``, ``-0.5`` or ``3/4``.

    An expression that merely evaluates to the number (``12*7``) is not one,
    so a reply cannot earn credit by restating the question's arithmetic.
    """
    if node[0] == "neg":
        return is_numeric_literal(node[1])
    if node[0] == "num":
        return True
    return node[0] == "div" and _integer_literal(node[1]) and _integer_literal(node[2])


def _integer_literal(node: Node) -> bool:
    if node[0] == "neg":
        return _integer_literal(node[1])
    return node[0] == "num" and Fraction(node[1]).denominator == 1


def _bounded(value: Fraction) -> Fraction:
    if (
        value.numerator.bit_length() > _MAX_BITS
        or value.denominator.bit_length() > _MAX_BITS
    ):
        raise ExpressionError("value exceeds the exact-arithmetic limits")
    return value


def _power(base: Fraction, exponent: Fraction) -> Fraction:
    if exponent.denominator != 1 or abs(exponent) > _MAX_EXPONENT:
        raise ExpressionError("only small integer exponents are exact")
    if base == 0 and exponent < 0:
        raise ExpressionError("division by zero")
    estimate = max(base.numerator.bit_length(), base.denominator.bit_length())
    if estimate * abs(int(exponent)) > _MAX_BITS:
        raise ExpressionError("value exceeds the exact-arithmetic limits")
    return base ** int(exponent)


def evaluate_expression(
    node: Node, values: dict[str, Fraction] | None = None
) -> Fraction:
    """Exact value of ``node`` with symbols bound to ``values``.

    Raises :class:`ExpressionError` for an unbound symbol, a function call,
    ``pi``, division by zero, a non-integer exponent or a value beyond the
    size limits.
    """
    kind = node[0]
    if kind == "num":
        return Fraction(node[1])
    if kind == "sym":
        if values is None or node[1] not in values:
            raise ExpressionError(f"{node[1]} has no exact rational value")
        return values[node[1]]
    if kind == "neg":
        return -evaluate_expression(node[1], values)
    if kind == "call":
        raise ExpressionError(f"{node[1]} has no exact rational value")
    left = evaluate_expression(node[1], values)
    right = evaluate_expression(node[2], values)
    if kind == "add":
        return _bounded(left + right)
    if kind == "sub":
        return _bounded(left - right)
    if kind == "mul":
        return _bounded(left * right)
    if kind == "div":
        if right == 0:
            raise ExpressionError("division by zero")
        return _bounded(left / right)
    return _bounded(_power(left, right))


def rational_value(node: Node) -> Fraction | None:
    """Exact value of a tree without symbols or functions, else ``None``."""
    try:
        return evaluate_expression(node)
    except ExpressionError:
        return None


def _probe_seed(first: Node, second: Node) -> int:
    """A seed derived from both expressions (in either order).

    Probe points used to come from fixed, public seeds, so an answer could be
    built to vanish exactly there (``reference + (x - p1)...(x - p8)``).
    Seeding from a hash of the two trees keeps a verdict deterministic while
    any change to the answer moves every point.
    """
    encoded = sorted(
        json.dumps(_encode_node(node), separators=(",", ":"))
        for node in (first, second)
    )
    digest = hashlib.sha256("\x00".join(encoded).encode()).digest()
    return int.from_bytes(digest[:16], "big")


def _probe_values(names: list[str], rng: random.Random) -> dict[str, Fraction]:
    return {
        name: Fraction(rng.randint(-997, 997), rng.randint(1, 97)) for name in names
    }


def _agree_at_points(first: Node, second: Node, names: list[str]) -> bool | None:
    """Rational functions agreeing at ``_PROBE_POINTS`` pseudo-random points.

    Two distinct rational functions of moderate degree agree at a random
    rational point with negligible probability, so agreement at every probe
    is equivalence (Schwartz-Zippel). The points are drawn from a generator
    seeded by both expressions (:func:`_probe_seed`): deterministic, but not a
    fixed public set an answer can be crafted against. Points where either
    side is undefined are skipped; ``None`` when too few points were defined
    to decide.
    """
    rng = random.Random(_probe_seed(first, second))
    checked = 0
    for _ in range(4 * _PROBE_POINTS):
        values = _probe_values(names, rng)
        try:
            a, b = evaluate_expression(first, values), evaluate_expression(
                second, values
            )
        except ExpressionError:
            continue
        if a != b:
            return False
        checked += 1
        if checked == _PROBE_POINTS:
            return True
    return None


def _to_sympy(node: Node, sympy: Any) -> Any:
    """Build a sympy expression node by node: model text is never parsed.

    The exact path's limits apply here too: a power whose exponent is a
    number above ``_MAX_EXPONENT`` (``x^(9^9)``, ``sqrt(x)+9^9^8``), or whose
    exact value would exceed ``_MAX_BITS``, raises :class:`ExpressionError`
    instead of building a multi-million-digit number.
    """
    kind = node[0]
    if kind == "num":
        return sympy.Rational(node[1].numerator, node[1].denominator)
    if kind == "sym":
        return sympy.pi if node[1] == "pi" else sympy.Symbol(node[1])
    if kind == "neg":
        return -_to_sympy(node[1], sympy)
    if kind == "call":
        function = {"abs": sympy.Abs}.get(node[1]) or getattr(sympy, node[1])
        return function(_to_sympy(node[2], sympy))
    if kind == "pow":
        return _sympy_power(node, sympy)
    left, right = _to_sympy(node[1], sympy), _to_sympy(node[2], sympy)
    if kind == "add":
        return left + right
    if kind == "sub":
        return left - right
    if kind == "mul":
        return left * right
    return left / right


def _sympy_power(node: Node, sympy: Any) -> Any:
    """A bounded power: the exact path's exponent and size limits."""
    value = _checked_power(node)
    if value is not None:
        return sympy.Rational(value.numerator, value.denominator)
    return _to_sympy(node[1], sympy) ** _to_sympy(node[2], sympy)


def _checked_power(node: Node) -> Fraction | None:
    """The exact value of a numeric integer power, ``None`` for any other
    allowed power; :class:`ExpressionError` beyond the exact path's limits."""
    exponent = rational_value(node[2])
    if exponent is None and _exact_only(node[2]) and not free_symbols(node[2]):
        # A numeric exponent too large to evaluate exactly (9^9^8).
        raise ExpressionError("exponent exceeds the exact-arithmetic limits")
    if exponent is not None and abs(exponent) > _MAX_EXPONENT:
        raise ExpressionError("exponent exceeds the exact-arithmetic limits")
    base = rational_value(node[1])
    if base is not None and exponent is not None and exponent.denominator == 1:
        return _bounded(_power(base, exponent))
    return None


def _sympy_difference_is_zero(first: Node, second: Node, sympy: Any) -> bool | None:
    """``simplify(first - second) == 0`` in this process (the worker's job)."""
    try:
        difference = sympy.simplify(_to_sympy(first, sympy) - _to_sympy(second, sympy))
    except (
        ExpressionError,
        TypeError,
        ValueError,
        ZeroDivisionError,
        RecursionError,
        OverflowError,
        MemoryError,
    ):
        return None
    # Anything sympy cannot reduce to zero is treated as different.
    return bool(difference == 0)


# --- the sympy worker -----------------------------------------------------------


def _encode_node(node: Node) -> list[Any]:
    """A JSON form of a tree (numbers as exact numerator/denominator)."""
    kind = node[0]
    if kind == "num":
        value = Fraction(node[1])
        return ["num", value.numerator, value.denominator]
    if kind == "sym":
        return ["sym", node[1]]
    if kind == "call":
        return ["call", node[1], _encode_node(node[2])]
    return [kind, *(_encode_node(child) for child in node[1:])]


def _decode_node(data: Any) -> Node:
    kind = data[0]
    if kind == "num":
        return ("num", Fraction(int(data[1]), int(data[2])))
    if kind == "sym":
        return ("sym", str(data[1]))
    if kind == "call":
        return ("call", str(data[1]), _decode_node(data[2]))
    return (str(kind), *(_decode_node(child) for child in data[1:]))


def _serve_sympy(
    requests: Iterable[str] | None = None, out: IO[str] | None = None
) -> None:  # pragma: no cover - runs in the worker process
    """The worker's loop: one JSON request per line, one JSON verdict per line."""
    requests = sys.stdin if requests is None else requests
    out = sys.stdout if out is None else out
    try:
        import sympy
    except ImportError:
        out.write(json.dumps({"ready": False}) + "\n")
        out.flush()
        return
    out.write(json.dumps({"ready": True}) + "\n")
    out.flush()
    for line in requests:
        # An unexpected error (a sympy bug on odd input) ends this worker; the
        # parent reads that as "undecided" and starts a fresh one next time.
        request = json.loads(line)
        verdict = _sympy_difference_is_zero(
            _decode_node(request["a"]), _decode_node(request["b"]), sympy
        )
        out.write(json.dumps({"equal": verdict}) + "\n")
        out.flush()


#: Loads this file as a module in the worker (no package import needed).
_WORKER_BOOT = (
    "import importlib.util, sys\n"
    "spec = importlib.util.spec_from_file_location('_stateset_sympy_worker', "
    "sys.argv[1])\n"
    "module = importlib.util.module_from_spec(spec)\n"
    "sys.modules[spec.name] = module\n"
    "spec.loader.exec_module(module)\n"
    "module._serve_sympy()\n"
)


def _pump(stream: IO[str], lines: queue.Queue[str | None]) -> None:
    for line in stream:
        lines.put(line)
    lines.put(None)


class _SympyWorker:
    """One long-lived ``python`` process that runs ``sympy.simplify``.

    A call that runs over its timeout kills the process (a hard stop, on
    Linux and Windows alike) and the next call starts a fresh one. Calls are
    serialized; the startup time is not counted against a call's timeout.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._owner = os.getpid()
        self._unavailable = False

    def ask(self, first: Node, second: Node, timeout: float) -> bool | None:
        request = json.dumps({"a": _encode_node(first), "b": _encode_node(second)})
        with self._lock:
            if not self._ensure_started():
                return None
            process = self._process
            assert process is not None and process.stdin is not None
            timed_out = False
            try:
                process.stdin.write(request + "\n")
                process.stdin.flush()
                line = self._lines.get(timeout=max(float(timeout), 0.0))
            except queue.Empty:
                line, timed_out = None, True
            except OSError:  # the worker already exited (a broken pipe)
                line = None
            if line is None:
                logger.warning(
                    "the sympy worker %s; it was stopped and the answer scores "
                    "as not equivalent",
                    f"ran over {timeout:.1f}s" if timed_out else "exited",
                )
                self._stop()
                return None
            verdict = json.loads(line).get("equal")
            return verdict if isinstance(verdict, bool) else None

    def _ensure_started(self) -> bool:
        if self._owner != os.getpid():
            # A forked child must not share its parent's pipes.
            self._process, self._lines, self._owner = None, queue.Queue(), os.getpid()
        if self._process is not None and self._process.poll() is None:
            return True
        if self._unavailable:
            return False
        try:
            process = subprocess.Popen(
                [sys.executable, "-c", _WORKER_BOOT, os.path.abspath(__file__)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
            )
        except OSError as error:
            logger.warning("cannot start the sympy worker: %s", error)
            self._unavailable = True
            return False
        lines: queue.Queue[str | None] = queue.Queue()
        assert process.stdout is not None
        threading.Thread(
            target=_pump, args=(process.stdout, lines), daemon=True
        ).start()
        try:
            ready = lines.get(timeout=_WORKER_START_SECONDS)
        except queue.Empty:
            ready = None
        if ready is None or not json.loads(ready).get("ready"):
            process.kill()
            process.wait()
            logger.warning(
                "the sympy worker did not start; symbolic answers "
                "with functions cannot be checked"
            )
            self._unavailable = ready is not None  # sympy missing: stay off
            return False
        self._process, self._lines = process, lines
        return True

    def _stop(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            process.kill()
            process.wait()

    def close(self) -> None:
        with self._lock:
            self._stop()


_WORKER = _SympyWorker()
atexit.register(_WORKER.close)


@functools.lru_cache(maxsize=1)
def _sympy_installed() -> bool:
    return importlib.util.find_spec("sympy") is not None


def _sympy_equivalent(
    first: Node, second: Node, timeout: float | None = None
) -> bool | None:
    if not _sympy_installed():
        return None
    try:
        # Refuse oversized trees here, before any process is involved.
        _check_sympy_bounds(first)
        _check_sympy_bounds(second)
    except ExpressionError:
        return None
    limit = SYMPY_TIMEOUT_SECONDS if timeout is None else float(timeout)
    return _WORKER.ask(first, second, limit)


def _check_sympy_bounds(node: Node) -> None:
    """Raise :class:`ExpressionError` where :func:`_to_sympy` would refuse."""
    if node[0] == "pow":
        _checked_power(node)
    for child in node[1:]:
        if isinstance(child, tuple):
            _check_sympy_bounds(child)


def equivalent(
    first: Node, second: Node, *, timeout: float | None = None
) -> bool | None:
    """Whether two parsed expressions are equal as numbers or functions.

    Exact for numbers; for symbols and the four operations with integer
    powers, exact evaluation at pseudo-random rational points seeded from
    both expressions (same answer with or without sympy); with function
    calls, ``sympy.simplify`` in the worker process, killed after ``timeout``
    seconds (default :data:`SYMPY_TIMEOUT_SECONDS`). ``None`` when it cannot be decided
    (functions without sympy, a timeout, sizes beyond the limits, or answers
    undefined almost everywhere).
    """
    names = sorted(free_symbols(first) | free_symbols(second))
    if _exact_only(first) and _exact_only(second):
        if not names:
            a, b = rational_value(first), rational_value(second)
            if a is not None and b is not None:
                return a == b
        else:
            decided = _agree_at_points(first, second, names)
            if decided is not None:
                return decided
    return _sympy_equivalent(first, second, timeout)


# --- LaTeX answer normalization ----------------------------------------------

#: Sizing and spacing commands: ``\left(``, ``\Big|``, ``\,``, ``\!``, ``\ ``.
_LATEX_DROP = re.compile(
    r"\\(?:left|right|[bB]igg?[lrm]?|displaystyle|textstyle|quad|qquad)"
    r"(?![A-Za-z])|\\[!,;:> ]|~"
)
#: Commands whose argument is plain text: ``\text{pens}`` -> ``pens``.
_TEXT_WRAPPERS = (
    r"\text",
    r"\textbf",
    r"\textit",
    r"\textrm",
    r"\textnormal",
    r"\mathrm",
    r"\mathbf",
    r"\mathit",
    r"\mbox",
)
_LATEX_OPS = {
    r"\cdot": "*",
    r"\times": "*",
    r"\div": "/",
    "\u00d7": "*",
    "\u00b7": "*",
    "\u22c5": "*",
    "\u00f7": "/",
    "\u2212": "-",
    r"\pi": " pi ",
    "\u03c0": " pi ",
    r"\dfrac": r"\frac",
    r"\tfrac": r"\frac",
    r"\ln": " ln ",
    r"\log": " log ",
    r"\sin": " sin ",
    r"\cos": " cos ",
    r"\tan": " tan ",
    r"\exp": " exp ",
}
#: ``\frac12``, ``\frac1{2}``, ``\frac{1}2``: one-digit arguments without braces.
_SHORT_FRACTION = re.compile(r"\\frac\s*(\d|\{[^{}]*\})\s*(\d|\{[^{}]*\})")
#: Degrees are a unit: ``90^\circ``, ``90^{\circ}``, ``90°``.
_DEGREES = re.compile(r"\^\s*\{?\s*\\circ\s*\}?|\\circ|\\degree|\u00b0")
#: A LaTeX thousands separator: ``1\,000``, ``1{,}000``, ``1\ 000``.
_LATEX_THOUSANDS = re.compile(r"(?<=\d)(?:\\,|\{,\}|\\ )(?=\d{3}(?!\d))")
_THOUSANDS = re.compile(r"(?<![\d.])(\d{1,3}(?:,\d{3})+)(?![\d,])")
_ASSIGNMENT = re.compile(r"^\s*[A-Za-z]\s*=\s*(?!=)")


def _group(text: str, start: int) -> tuple[str, int] | None:
    """The brace group opening at ``text[start] == "{"``, and its end."""
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    return None


def _rewrite_command(text: str, command: str, arity: int, template: str) -> str:
    """Replace ``\\command{a}{b}`` by ``template.format(a, b)``, innermost last."""
    while True:
        position = text.find(command + "{")
        if position < 0:
            return text
        cursor = position + len(command)
        groups = []
        for _ in range(arity):
            found = _group(text, cursor)
            if found is None:
                return text
            groups.append(found[0])
            cursor = found[1]
        text = text[:position] + template.format(*groups) + text[cursor:]


def _braced_fraction(match: re.Match[str]) -> str:
    parts = [
        part if part.startswith("{") else "{" + part + "}" for part in match.groups()
    ]
    return r"\frac" + "".join(parts)


def normalize_answer_text(text: str) -> str:
    """Rewrite a LaTeX or plain answer into the syntax :func:`parse_expression` reads.

    Drops ``$``/``\\(``/``\\[`` delimiters, sizing and spacing commands
    (``\\left``, ``\\,``, ``\\!``), ``\\text{}``-style wrappers, degree
    marks, a leading ``x =`` and a trailing period; turns ``\\frac{a}{b}``
    (also ``\\dfrac``, ``\\tfrac`` and ``\\frac12``) into ``((a)/(b))``,
    ``\\sqrt{a}`` into ``sqrt(a)``, ``\\cdot``/``\\times`` into ``*`` and
    remaining braces into parentheses, and reads ``1,299``, ``1{,}299`` and
    ``1\\,299`` as ``1299``. A percent sign is kept (``\\%`` becomes ``%``).
    """
    text = text.strip()
    if len(text) > _MAX_NORMALIZE_CHARS:
        return text
    text = _LATEX_THOUSANDS.sub("", text).replace(r"\%", "%").replace(r"\$", " ")
    for delimiter in ("$$", "$", r"\(", r"\)", r"\[", r"\]"):
        text = text.replace(delimiter, " ")
    for wrapper in _TEXT_WRAPPERS:
        text = _rewrite_command(text, wrapper, 1, " {0} ")
    for latex, plain in _LATEX_OPS.items():
        text = text.replace(latex, plain)
    text = _DEGREES.sub(" ", text)
    text = _LATEX_DROP.sub(" ", text)
    text = _SHORT_FRACTION.sub(_braced_fraction, text)
    text = _rewrite_command(text, r"\frac", 2, "(({0})/({1}))")
    text = _rewrite_command(text, r"\sqrt", 1, "sqrt({0})")
    text = text.replace("{", "(").replace("}", ")")
    text = _THOUSANDS.sub(lambda match: match.group(1).replace(",", ""), text)
    text = _ASSIGNMENT.sub("", text).strip()
    return text[:-1].strip() if text.endswith(".") else text
