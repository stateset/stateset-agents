"""Experimental offline RL algorithms for conversational agents.

Moved here from ``stateset_agents.training`` in v0.64.0:

- :mod:`.algorithms` -- Conservative Q-Learning (CQL) and Implicit Q-Learning (IQL)
- :mod:`.bcq` -- Batch-Constrained Q-Learning (BCQ)
- :mod:`.bear` -- Bootstrapping Error Accumulation Reduction (BEAR)
- :mod:`.decision_transformer` -- Decision Transformer
- :mod:`.offline_grpo` -- hybrid offline/online GRPO trainer

These carry no API stability guarantee (see :mod:`stateset_agents.experimental`).
Import from the submodules directly; this package does not import torch.
"""

from typing import Any as _Any

from stateset_agents.training.offline_backend import register_offline_trainer


def _load_offline_grpo() -> tuple[type[_Any], type[_Any]]:
    from .offline_grpo import OfflineGRPOConfig, OfflineGRPOTrainer

    return OfflineGRPOConfig, OfflineGRPOTrainer


# ``train(training_mode="offline"|"hybrid")`` runs this trainer; importing the
# package is the opt-in (it stays torch-free until offline training runs).
register_offline_trainer(_load_offline_grpo)
