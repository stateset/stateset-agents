"""Trainers read config fields without restating their defaults.

``getattr(config, "field", default)`` repeats a ``TrainingConfig`` default
inline, and the copies drift: single-turn GRPO trained a config without
``learning_rate`` at 5e-5 against the dataclass's 5e-6, with Adam's beta2 at
0.999 against 0.99. Trainers read ``config.field`` from a typed config, or
``config_value(config, "field")`` from a legacy duck-typed one, whose
default is the dataclass's. This ratchet counts the inline-default reads
left in ``stateset_agents/training``; the count may only go down.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from stateset_agents.training.config import TrainingConfig, config_value

PACKAGE = Path(__file__).resolve().parents[2] / "stateset_agents" / "training"

#: 261 inline-default config reads when the ratchet was introduced (S10 of
#: the training spine); config_value took the 134 that restated the
#: dataclass default verbatim (127), then the trainers' drifting reads
#: (learning rate, Adam beta2, group size, episode and accumulation budgets,
#: seed, W&B tags) settled on the dataclass (108); then the GRPO losses'
#: entropy_coef / generation_batch_size / max_completion_length / prompt and
#: clip reads and the TRL bridge's batch sizing did too (96).
CEILING = 96


def inline_default_reads() -> list[str]:
    """Every ``getattr(<...config...>, "name", default)`` in ``training/``."""
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) == 3
                and isinstance(node.args[1], ast.Constant)
                and "config" in ast.unparse(node.args[0]).lower()
            ):
                relative = path.relative_to(PACKAGE)
                found.append(f"{relative}:{node.lineno} {node.args[1].value}")
    return found


def test_inline_default_reads_never_increase() -> None:
    found = inline_default_reads()
    assert len(found) <= CEILING, (
        f"{len(found)} getattr(config, name, default) reads in training/ "
        f"(ceiling {CEILING}). Read the field directly from a typed config or "
        "use config.config_value(config, name), whose default is the "
        "dataclass's:\n" + "\n".join(found)
    )


def test_ceiling_matches_reality() -> None:
    found = inline_default_reads()
    assert len(found) >= CEILING, (
        f"Only {len(found)} inline-default config reads remain; lower CEILING "
        f"from {CEILING} to {len(found)} to lock in the improvement."
    )


def test_config_value_falls_back_to_the_dataclass_default() -> None:
    defaults = TrainingConfig()
    legacy = SimpleNamespace(learning_rate=1e-3)
    assert config_value(legacy, "learning_rate") == 1e-3
    for name in ("adam_beta2", "num_generations", "num_episodes", "seed"):
        assert config_value(legacy, name) == getattr(defaults, name)
    # a fresh copy of a mutable default each time
    assert config_value(legacy, "wandb_tags") == defaults.wandb_tags
    with pytest.raises(AttributeError, match="not a TrainingConfig field"):
        config_value(legacy, "learnig_rate")
