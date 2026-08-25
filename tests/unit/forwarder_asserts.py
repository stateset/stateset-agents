"""Structural assertions for the ``examples/finetune_*_gspo.py`` forwarders.

Each of these scripts is a thin shim onto the unified driver
``examples/finetune_gspo.py --model <preset>``. Proving that by spawning a
real interpreter costs ~9 s per script, and ten of them dominated the test
suite's wall time. The forwarding is a *structural* property, so read it out
of the AST instead: the script must import the driver's ``main``, print a
notice naming itself to stderr, and end in
``sys.exit(_driver_main(["--model", <preset>, *sys.argv[1:]]))``.

What the AST cannot prove -- that the driver actually runs end to end from a
fresh interpreter -- is still covered by one real subprocess per test file
(see ``test_example_model_presets.py``), so the spawn count drops from ten to
one rather than to zero.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER_MODULE = "examples.finetune_gspo"


def _driver_main_alias(tree: ast.Module, script_name: str) -> str:
    """Return the local name the driver's ``main`` was imported as."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == DRIVER_MODULE:
            for alias in node.names:
                if alias.name == "main":
                    return alias.asname or alias.name
    raise AssertionError(
        f"{script_name} must forward to the unified driver: expected "
        f"'from {DRIVER_MODULE} import main'"
    )


def _stderr_notice(tree: ast.Module) -> str | None:
    """Return the text of the first ``print(..., file=sys.stderr)`` literal."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "print" or not node.args:
            continue
        goes_to_stderr = any(
            kw.arg == "file"
            and isinstance(kw.value, ast.Attribute)
            and kw.value.attr == "stderr"
            for kw in node.keywords
        )
        first = node.args[0]
        if goes_to_stderr and isinstance(first, ast.Constant):
            if isinstance(first.value, str):
                return first.value
    return None


def _forwarded_argv(tree: ast.Module, alias: str) -> list[str] | None:
    """Return the literal leading argv the script hands the driver.

    Matches ``sys.exit(<alias>([...literals..., *sys.argv[1:]]))`` and returns
    the literal prefix, or ``None`` if the script has no such call.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != alias or len(node.args) != 1:
            continue
        argv = node.args[0]
        if not isinstance(argv, ast.List):
            continue
        literals = [
            element.value
            for element in argv.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        ]
        forwards_rest = any(isinstance(element, ast.Starred) for element in argv.elts)
        if not forwards_rest:
            raise AssertionError(
                "forwarder must pass the caller's own flags through as *sys.argv[1:]"
            )
        return literals
    return None


def assert_forwards_to_driver(
    script_name: str,
    *,
    model: str,
    notice_word: str = "forwarder",
) -> None:
    """Assert ``examples/<script_name>`` is a forwarder for ``--model model``.

    Purely static: no interpreter is spawned. ``notice_word`` is the word the
    stderr notice must contain ("forwarder" or "deprecated").
    """
    from examples.model_presets import get_preset

    path = REPO_ROOT / "examples" / script_name
    assert path.exists(), f"{script_name} does not exist"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    alias = _driver_main_alias(tree, script_name)

    argv = _forwarded_argv(tree, alias)
    assert argv is not None, (
        f"{script_name} imports the driver's main but never calls it; a "
        "forwarder must end in sys.exit(main([...]))"
    )
    assert argv[:2] == [
        "--model",
        model,
    ], f"{script_name} must forward to --model {model!r}, got {argv[:2]!r}"

    # The preset it forwards to has to be one the driver actually knows.
    get_preset(model)  # raises KeyError for an unknown preset

    notice = _stderr_notice(tree)
    assert notice is not None, (
        f"{script_name} must print a notice to stderr telling the reader to "
        "use the unified driver"
    )
    assert (
        notice_word in notice.lower()
    ), f"{script_name}'s stderr notice must contain {notice_word!r}: {notice!r}"
    assert (
        model in notice
    ), f"{script_name}'s stderr notice must name the model {model!r}: {notice!r}"
