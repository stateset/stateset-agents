"""``stateset_agents.training.train`` is the ``train()`` function whether or not
the ``stateset_agents.training.train`` submodule was imported first (importlib
binds a submodule onto its package under the same name; the package keeps the
public export instead)."""

from __future__ import annotations

import importlib
import inspect
import sys

import stateset_agents.training as training_pkg


def test_train_export_survives_importing_the_submodule():
    submodule = importlib.import_module("stateset_agents.training.train")
    assert sys.modules["stateset_agents.training.train"] is submodule
    from stateset_agents.training import train

    assert train is submodule.train
    assert inspect.iscoroutinefunction(train)
    assert training_pkg.train is submodule.train
    assert training_pkg._train_module is submodule


def test_submodule_stays_importable_by_dotted_path():
    from stateset_agents.training.train import AutoTrainer, train

    assert callable(train) and AutoTrainer is not None
    assert importlib.import_module("stateset_agents.training.train").AutoTrainer is (
        AutoTrainer
    )


def test_other_submodules_still_bind_normally():
    import stateset_agents.training.objectives as objectives

    assert training_pkg.objectives is objectives
    assert "objectives" not in training_pkg._SHADOWED_SUBMODULES
    assert "train" in training_pkg._SHADOWED_SUBMODULES
