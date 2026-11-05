"""Guardrail: exception tuples that catch bug classes only go down.

``except NAME_EXCEPTIONS:`` reads like a narrow handler, but many module
tuples resolve to ``TypeError``/``AttributeError``/``KeyError``/... -- the
exceptions a programming error raises -- so a bug becomes a logged failure
and the blind-except ratchet (``test_blind_except_ratchet.py``) never sees
it. This test resolves every module-level ``*_EXCEPTIONS`` tuple statically
(through imports, concatenation and starred items) and ratchets, per file:

* the tuples whose resolved members include a bug class, and
* the handlers that catch such a tuple and *swallow* it (no bare ``raise``).

Both counts may only shrink: a new bug-class tuple or swallowing handler
fails, and a narrowed one fails until ``BASELINE`` is lowered, so the gain is
locked in. Narrow a tuple to what the guarded code really expects --
``exceptions.RUNTIME_FAILURE_EXCEPTIONS`` (``RuntimeError``, ``ValueError``,
``OSError``) plus the library's own error types -- and let bugs propagate.
"""

from __future__ import annotations

import ast
import builtins
from collections import Counter
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT / "stateset_agents"

#: Exceptions a programming error raises; a handler that swallows them turns
#: bugs into silent fallbacks.
BUG_CLASSES = frozenset(
    {
        "AssertionError",
        "AttributeError",
        "IndexError",
        "KeyError",
        "LookupError",
        "NameError",
        "TypeError",
    }
)

#: path -> (tuples resolving to a bug class, handlers swallowing them).
#: Only shrinks. (80 tuples / 201 swallowing handlers when the rule was
#: written; that pass narrowed api.resilience.HEALTHCHECK_EXCEPTIONS,
#: api.routers.metrics.METRICS_EXCEPTIONS and
#: utils.alerts.NOTIFICATION_EXCEPTIONS.)
BASELINE: dict[str, tuple[int, int]] = {
    "stateset_agents/__init__.py": (1, 2),
    "stateset_agents/api/dependencies.py": (1, 3),
    "stateset_agents/api/distributed_cache.py": (1, 9),
    "stateset_agents/api/grpo/handlers.py": (2, 5),
    "stateset_agents/api/grpo/router_v1.py": (1, 1),
    "stateset_agents/api/grpo/service.py": (2, 3),
    "stateset_agents/api/grpo/service_routes.py": (3, 4),
    "stateset_agents/api/observability.py": (1, 0),
    "stateset_agents/api/resilience.py": (1, 0),
    "stateset_agents/api/routers/agents.py": (1, 2),
    "stateset_agents/api/routers/training.py": (1, 1),
    "stateset_agents/api/services/agent_service.py": (2, 4),
    "stateset_agents/api/services/inference_service.py": (1, 2),
    "stateset_agents/api/shutdown.py": (1, 1),
    "stateset_agents/api/streaming.py": (1, 3),
    "stateset_agents/cli.py": (3, 7),
    "stateset_agents/cli_evaluate.py": (2, 3),
    "stateset_agents/cli_init.py": (1, 1),
    "stateset_agents/cli_meta.py": (0, 1),
    "stateset_agents/cli_research.py": (0, 2),
    "stateset_agents/cli_train.py": (0, 8),
    "stateset_agents/core/__init__.py": (1, 2),
    "stateset_agents/core/advanced_monitoring.py": (1, 6),
    "stateset_agents/core/agent.py": (3, 7),
    "stateset_agents/core/async_pool.py": (1, 4),
    "stateset_agents/core/enhanced/advanced_evaluation.py": (1, 4),
    "stateset_agents/core/enhanced_state_cache.py": (1, 5),
    "stateset_agents/core/enhanced_state_management.py": (0, 7),
    "stateset_agents/core/function_calling.py": (1, 1),
    "stateset_agents/core/gym/adapter.py": (1, 3),
    "stateset_agents/core/multiturn_agent.py": (3, 6),
    "stateset_agents/core/tool_agent.py": (0, 1),
    "stateset_agents/core/type_system.py": (1, 2),
    "stateset_agents/data/conversation_dataset.py": (1, 1),
    "stateset_agents/exceptions.py": (6, 0),
    "stateset_agents/experimental/adaptive_learning_controller.py": (1, 1),
    "stateset_agents/experimental/intelligent_orchestrator.py": (1, 2),
    "stateset_agents/experimental/multimodal_processing.py": (1, 7),
    "stateset_agents/experimental/neural_architecture_search.py": (1, 2),
    "stateset_agents/experimental/sim_to_real/conversation_simulator.py": (1, 1),
    "stateset_agents/rewards/llm_judge.py": (1, 3),
    "stateset_agents/rewards/llm_reward.py": (1, 5),
    "stateset_agents/rewards/multi_objective_components.py": (1, 0),
    "stateset_agents/training/auto_research/checkpoint_manager.py": (1, 5),
    "stateset_agents/training/base_trainer.py": (1, 2),
    "stateset_agents/training/callbacks.py": (1, 1),
    "stateset_agents/training/continual_learning.py": (1, 1),
    "stateset_agents/training/dapo_trainer.py": (1, 1),
    "stateset_agents/training/evaluation.py": (2, 3),
    "stateset_agents/training/gspo_generation.py": (0, 3),
    "stateset_agents/training/gspo_trainer.py": (0, 3),
    "stateset_agents/training/hpo/grpo_hpo_trainer.py": (1, 2),
    "stateset_agents/training/hpo/optuna_backend.py": (1, 0),
    "stateset_agents/training/hpo/ray_tune_backend.py": (1, 1),
    "stateset_agents/training/hpo/wandb_backend.py": (1, 1),
    "stateset_agents/training/loss_computation.py": (2, 4),
    "stateset_agents/training/multi_turn_trainer.py": (1, 2),
    "stateset_agents/training/neural_reward_trainer.py": (1, 0),
    "stateset_agents/training/serving_artifacts.py": (0, 2),
    "stateset_agents/training/single_turn_trainer.py": (1, 2),
    "stateset_agents/training/transformer_reward_model.py": (1, 4),
    "stateset_agents/training/trl_grpo_trainer.py": (0, 4),
    "stateset_agents/training/vllm_backend.py": (1, 4),
    "stateset_agents/utils/logging.py": (1, 0),
    "stateset_agents/utils/monitoring.py": (1, 4),
    "stateset_agents/utils/observability.py": (1, 0),
    "stateset_agents/utils/wandb_integration.py": (1, 5),
    "stateset_agents/utils/wandb_logger.py": (0, 12),
}


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


@cache
def _modules() -> dict[str, tuple[str, ast.Module, dict[str, str]]]:
    """module -> (repo-relative path, tree, local name -> qualified name)."""
    result: dict[str, tuple[str, ast.Module, dict[str, str]]] = {}
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        module = _module_name(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names: dict[str, str] = {}
        for node in ast.walk(tree):  # function-level imports name tuples too
            if isinstance(node, ast.ImportFrom):
                if node.level == 0:
                    base = node.module or ""
                else:
                    package = (
                        module.split(".")
                        if path.name == "__init__.py"
                        else module.split(".")[:-1]
                    )
                    base = ".".join(
                        package[: len(package) - (node.level - 1)]
                        + ([node.module] if node.module else [])
                    )
                for alias in node.names:
                    names[alias.asname or alias.name] = f"{base}.{alias.name}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    names[alias.asname or alias.name.split(".")[0]] = (
                        alias.name if alias.asname else alias.name.split(".")[0]
                    )
        result[module] = (path.relative_to(ROOT).as_posix(), tree, names)
    return result


@cache
def _tuples() -> dict[str, tuple[str, ast.expr]]:
    """``module.NAME_EXCEPTIONS`` -> (module, value expression)."""
    found: dict[str, tuple[str, ast.expr]] = {}
    for module, (_, tree, _) in _modules().items():
        for node in tree.body:
            target = value = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target, value = node.targets[0], node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                target, value = node.target, node.value
            if (
                isinstance(target, ast.Name)
                and target.id.endswith("_EXCEPTIONS")
                and value is not None
            ):
                found[f"{module}.{target.id}"] = (module, value)
    return found


def _qualify(expr: ast.expr, module: str) -> str | None:
    names = _modules()[module][2]
    if isinstance(expr, ast.Name):
        local = f"{module}.{expr.id}"
        if local in _tuples():
            return local
        return names.get(expr.id, f"builtins.{expr.id}")
    if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name):
        root = names.get(expr.value.id)
        return f"{root}.{expr.attr}" if root else None
    return None


@cache
def members(qualified: str) -> frozenset[str]:
    """Builtin exception names a tuple resolves to (third-party ones dropped)."""
    module, value = _tuples()[qualified]
    return _resolve(value, module, frozenset({qualified}))


def _resolve(expr: ast.expr, module: str, seen: frozenset[str]) -> frozenset[str]:
    if isinstance(expr, ast.Tuple | ast.List):
        out: frozenset[str] = frozenset()
        for element in expr.elts:
            out |= _resolve(element, module, seen)
        return out
    if isinstance(expr, ast.Starred):
        return _resolve(expr.value, module, seen)
    if isinstance(expr, ast.BinOp):
        return _resolve(expr.left, module, seen) | _resolve(expr.right, module, seen)
    qualified = _qualify(expr, module)
    if qualified is None or qualified in seen:
        return frozenset()
    if qualified in _tuples():
        return members(qualified) if qualified not in seen else frozenset()
    root, _, name = qualified.rpartition(".")
    if root == "builtins" and isinstance(getattr(builtins, name, None), type):
        return frozenset({name})
    return frozenset()


def bug_tuples() -> dict[str, frozenset[str]]:
    return {
        name: members(name) & BUG_CLASSES
        for name in sorted(_tuples())
        if members(name) & BUG_CLASSES
    }


def _handler_tuples(handler: ast.ExceptHandler, module: str) -> list[str]:
    if handler.type is None:
        return []
    caught = (
        handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    )
    names = []
    for expr in caught:
        if isinstance(expr, ast.BinOp):
            caught.extend([expr.left, expr.right])
            continue
        qualified = _qualify(expr, module)
        if qualified in _tuples():
            names.append(qualified)
    return names


@cache
def measure() -> dict[str, tuple[int, int]]:
    """path -> (bug-class tuples defined, handlers swallowing one)."""
    tuples: Counter[str] = Counter()
    handlers: Counter[str] = Counter()
    bad = bug_tuples()
    for name in bad:
        module = _tuples()[name][0]
        tuples[_modules()[module][0]] += 1
    for module, (path, tree, _) in _modules().items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            if not any(name in bad for name in _handler_tuples(node, module)):
                continue
            reraises = any(
                isinstance(stmt, ast.Raise) and stmt.exc is None for stmt in node.body
            )
            if not reraises:
                handlers[path] += 1
    return {
        path: (tuples[path], handlers[path])
        for path in sorted(set(tuples) | set(handlers))
    }


def _render(counts: dict[str, tuple[int, int]]) -> str:
    return "\n".join(f'    "{path}": {value},' for path, value in counts.items())


def test_bug_class_tuples_and_handlers_only_shrink() -> None:
    counts = measure()
    grew = {
        path: f"{value} > baseline {BASELINE.get(path, (0, 0))}"
        for path, value in counts.items()
        if value[0] > BASELINE.get(path, (0, 0))[0]
        or value[1] > BASELINE.get(path, (0, 0))[1]
    }
    assert not grew, (
        "New exception tuples (or handlers swallowing them) catch bug classes "
        f"({', '.join(sorted(BUG_CLASSES))}); catch what the guarded code "
        "really raises (exceptions.RUNTIME_FAILURE_EXCEPTIONS plus library "
        f"errors) and let bugs propagate: {grew}"
    )


def test_baseline_is_tight() -> None:
    counts = measure()
    shrunk = {
        path: f"{counts.get(path, (0, 0))} < baseline {expected}"
        for path, expected in BASELINE.items()
        if counts.get(path, (0, 0)) != expected
    }
    assert not shrunk, (
        "Bug-class exception tuples were narrowed; lower BASELINE (and drop "
        f"files that reached (0, 0)) so the gain is locked in: {shrunk}\n"
        "Current measurement:\n"
        + _render({k: v for k, v in counts.items() if v != (0, 0)})
    )


def test_resolution_follows_imports_and_concatenation() -> None:
    shared = "stateset_agents.exceptions.SERIALIZATION_EXCEPTIONS"
    assert {"AttributeError", "KeyError", "TypeError"} <= members(shared)
    assert members("stateset_agents.exceptions.RUNTIME_FAILURE_EXCEPTIONS") == {
        "RuntimeError",
        "ValueError",
        "OSError",
    }
    assert "stateset_agents.exceptions.RUNTIME_FAILURE_EXCEPTIONS" not in bug_tuples()
    # An alias of a shared tuple inherits its members.
    assert members("stateset_agents.api.resilience.CIRCUIT_EXCEPTIONS") == members(
        "stateset_agents.exceptions.INFERENCE_EXCEPTIONS"
    )
    assert "stateset_agents.api.resilience.HEALTHCHECK_EXCEPTIONS" not in bug_tuples()
