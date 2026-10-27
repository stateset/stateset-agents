"""``core.type_system`` stops competing with the canonical API (v0.70.0).

* ``ConfigValidator`` / ``TypedTrainingConfig`` validate with rules that
  differ from ``TrainingConfig``; they are stable, so they warn until v0.72.0.
* The ``Agent``/``Environment``/``RewardFunction`` Protocols shadowed the
  concrete classes; they are ``*Protocol`` now, with dated aliases.
* ``core.advanced_monitoring`` imports ``psutil`` only when it collects.
"""

from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from stateset_agents.core import type_system


def test_config_validator_warns_until_v072() -> None:
    with pytest.warns(DeprecationWarning, match=r"ConfigValidator.*v0\.72\.0"):
        validator = type_system.ConfigValidator()
    assert validator.validate_model_config({"model_name": "gpt2"}) in {True, False}


def test_create_typed_config_does_not_warn_about_its_own_validator() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        config = type_system.create_typed_config(type_system.ModelConfig)
    assert config == {}


def test_typed_training_config_warns_and_is_the_typed_dict() -> None:
    with pytest.warns(DeprecationWarning, match=r"TypedTrainingConfig.*v0\.72\.0"):
        alias = type_system.TypedTrainingConfig
    assert alias is type_system.TrainingConfigDict
    assert "TypedTrainingConfig" not in type_system.__all__


@pytest.mark.parametrize("old", ["Agent", "Environment", "RewardFunction"])
def test_renamed_protocols_keep_a_dated_alias(old: str) -> None:
    new = getattr(type_system, f"{old}Protocol")
    with pytest.warns(DeprecationWarning, match=r"removed in v0\.71\.0"):
        assert getattr(type_system, old) is new
    assert old not in type_system.__all__
    assert f"{old}Protocol" in type_system.__all__


def test_protocols_no_longer_shadow_the_concrete_classes() -> None:
    import stateset_agents

    assert stateset_agents.Agent is not type_system.AgentProtocol
    assert type_system.AgentType.__bound__.__forward_arg__ == "AgentProtocol"


def test_core_advanced_monitoring_imports_psutil_lazily() -> None:
    repo = Path(__file__).resolve().parents[2]
    code = (
        "import sys\n"
        "import stateset_agents.core.advanced_monitoring\n"
        "assert 'psutil' not in sys.modules, 'psutil imported at module level'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
