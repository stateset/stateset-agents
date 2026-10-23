"""Importing the library never configures logging for the application.

``training/gepo_trainer.py`` and ``experimental/ppo.py`` called
``logging.basicConfig`` at import, so ``import stateset_agents.training.
gepo_trainer`` added a root handler and set the root level to INFO in the
user's process; ``core/rust_accelerator`` logged at INFO on every import.
Each module is imported in a fresh interpreter here (an import that already
happened in this process would prove nothing).
"""

from __future__ import annotations

import json
import os
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_PROBE = r"""
import importlib, json, logging, sys
mode, names = sys.argv[1], sys.argv[2:]
root = logging.getLogger()
records = []
if mode == "info":

    class Capture(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.INFO:
                records.append((record.name, record.getMessage()))

    root.addHandler(Capture(level=logging.DEBUG))
    root.setLevel(logging.DEBUG)
# In "root" mode the root logger starts as an application leaves it before
# configuring logging (no handlers, WARNING), so a basicConfig() is visible.
baseline = (root.level, list(root.handlers))
changed = []
for name in names:
    try:
        importlib.import_module(name)
    except ImportError:
        continue  # an optional dependency; not this test's concern
    if (root.level, list(root.handlers)) != baseline:
        changed.append(name)
        root.setLevel(baseline[0])
        root.handlers[:] = baseline[1]
print(json.dumps({"changed": changed, "info": records}))
"""


def _training_modules() -> list[str]:
    import stateset_agents.training as training

    names = [
        info.name
        for info in pkgutil.walk_packages(
            training.__path__, "stateset_agents.training."
        )
    ]
    return sorted(names)


def _probe(mode: str, modules: list[str]) -> dict[str, list[object]]:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, mode, *modules],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        timeout=600,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.timeout(900)
def test_importing_training_modules_leaves_the_root_logger_alone() -> None:
    modules = [
        *_training_modules(),
        "stateset_agents.experimental.ppo",
        "stateset_agents.core.rust_accelerator",
    ]
    report = _probe("root", modules)
    assert report["changed"] == [], (
        "these imports configured the root logger (logging.basicConfig at "
        f"import): {report['changed']}"
    )


def test_rust_accelerator_import_is_quiet() -> None:
    report = _probe("info", ["stateset_agents.core.rust_accelerator"])
    assert [r for r in report["info"] if "rust" in str(r[0])] == [], report["info"]


def test_gspo_trainer_has_its_docstring() -> None:
    """The docstring sat after a class attribute, so ``__doc__`` was None."""
    pytest.importorskip("torch")
    from stateset_agents.training.gspo_trainer import GSPOTrainer

    assert GSPOTrainer.__doc__ and "Group Sequence Policy Optimization" in (
        GSPOTrainer.__doc__
    )


@pytest.mark.asyncio
async def test_gepo_metrics_history_records_every_key_it_declares() -> None:
    """``kl_divergence``, ``gepo_coefficient`` and ``advantage_std`` were
    declared but never written (GEPO has no KL term: beta must be 0)."""
    pytest.importorskip("transformers")
    from tests.unit.test_dedicated_trainer_nonfinite_grads import _gepo

    trainer = _gepo()
    await trainer.train_step(["q"])
    await trainer.train_step(["q"])
    assert trainer.metrics_history
    assert {
        key: len(values) for key, values in trainer.metrics_history.items()
    } == dict.fromkeys(trainer.metrics_history, 2)
    assert "kl_divergence" not in trainer.metrics_history
