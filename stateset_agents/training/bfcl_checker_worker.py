"""Score BFCL answers with the official checker, in BFCL's own interpreter.

``bfcl-eval`` pins dependency versions that conflict with this package, so it
is installed into a separate environment and this script is run there. It
uses only the standard library plus ``bfcl_eval`` and speaks JSON lines:

    {"op": "tasks", "category": "simple_python"}
    {"op": "check", "category": "simple_python", "id": "...", "output": "..."}

Model-provider SDKs that ``bfcl_eval`` imports at module load are replaced by
empty stand-ins when absent; the checker and decoder code runs unmodified.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import json
import sys
import types
from typing import Any

CATEGORIES = ("simple_python", "multiple", "parallel", "parallel_multiple")


def _passthrough(*args: Any, **kwargs: Any) -> Any:
    """Act as a decorator when given one callable, else as a factory."""
    if len(args) == 1 and not kwargs and callable(args[0]):
        return args[0]
    return _passthrough


class _StandInMeta(type):
    """Lets a stand-in class be a base class, a decorator or a factory."""

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        return _passthrough(*args, **kwargs)

    def __getattr__(cls, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return _StandInMeta(name, (), {})


class _Missing(types.ModuleType):
    """Stand-in for an absent provider SDK or decorator library."""

    __path__: list[str] = []

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return _StandInMeta(name, (), {})


class _StandInFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve only the top-level packages named in ``names``."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname.split(".")[0] not in self.names:
            return None
        return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

    def create_module(self, spec: Any) -> types.ModuleType:
        return _Missing(spec.name)

    def exec_module(self, module: types.ModuleType) -> None:
        return None


def import_checker() -> list[str]:
    """Import the checker, standing in for each absent package it names.

    Only a package whose absence actually stops the import is replaced, so
    optional imports that libraries guard themselves keep failing normally.
    """
    finder = _StandInFinder()
    sys.meta_path.append(finder)
    for _ in range(100):
        try:
            import bfcl_eval.eval_checker.ast_eval.ast_checker
            import bfcl_eval.utils  # noqa: F401

            return sorted(finder.names)
        except ModuleNotFoundError as error:
            missing = (error.name or "").split(".")[0]
            if not missing or missing == "bfcl_eval" or missing in finder.names:
                raise
            finder.names.add(missing)
            for name in [m for m in sys.modules if m.startswith("bfcl_eval")]:
                del sys.modules[name]
    raise ImportError("Too many absent packages while importing bfcl_eval")


def load() -> dict[str, Any]:
    """Import the official checker, decoder, data and prompt."""
    stood_in = import_checker()
    from importlib import metadata
    from pathlib import Path

    import bfcl_eval
    from bfcl_eval.constants import default_prompts
    from bfcl_eval.constants.enums import Language, ReturnFormat
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    from bfcl_eval.model_handler.utils import default_decode_ast_prompting
    from bfcl_eval.utils import is_function_calling_format_output

    data = Path(bfcl_eval.__file__).parent / "data"

    def rows(path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    tasks: dict[str, dict[str, dict[str, Any]]] = {}
    for category in CATEGORIES:
        name = f"BFCL_v4_{category}.json"
        answers = {
            r["id"]: r["ground_truth"] for r in rows(data / "possible_answer" / name)
        }
        tasks[category] = {
            r["id"]: r | {"ground_truth": answers[r["id"]]} for r in rows(data / name)
        }
    plain = next(
        name
        for name, config in sorted(MODEL_CONFIG_MAPPING.items())
        if not config.underscore_to_dot
    )
    return {
        "version": metadata.version("bfcl-eval"),
        "stood_in": stood_in,
        "tasks": tasks,
        "prompt": default_prompts._DEFAULT_SYSTEM_PROMPT,
        "model_name": plain.replace("/", "_"),
        "check": ast_checker,
        "decode": default_decode_ast_prompting,
        "well_formed": is_function_calling_format_output,
        "python": (Language.PYTHON, ReturnFormat.PYTHON),
    }


def check(state: dict[str, Any], category: str, task_id: str, output: str) -> dict:
    """Decode ``output`` and score it the way the official runner does."""
    task = state["tasks"][category][task_id]
    language, return_format = state["python"]
    try:
        decoded = state["decode"](output, return_format)
    except Exception as error:  # the official runner scores decode errors as wrong
        return {
            "valid": False,
            "error_type": "ast_decoder:decoder_failed",
            "detail": str(error),
        }
    if not state["well_formed"](decoded):
        return {"valid": False, "error_type": "ast_decoder:decoder_wrong_output_format"}
    result = state["check"](
        task["function"],
        decoded,
        task["ground_truth"],
        language,
        category,
        state["model_name"],
    )
    return {"valid": bool(result["valid"]), "error_type": result.get("error_type")}


def main() -> int:
    state = load()
    ready = {"ready": True, "version": state["version"], "stood_in": state["stood_in"]}
    print(json.dumps(ready), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request["op"] == "tasks":
            reply: dict[str, Any] = {
                "prompt": state["prompt"],
                "tasks": [
                    {
                        "id": t["id"],
                        "question": t["question"],
                        "function": t["function"],
                    }
                    for t in state["tasks"][request["category"]].values()
                ],
            }
        else:
            reply = check(state, request["category"], request["id"], request["output"])
        print(json.dumps(reply), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
