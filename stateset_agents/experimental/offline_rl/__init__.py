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
