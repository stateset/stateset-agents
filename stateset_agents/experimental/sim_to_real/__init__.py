"""Experimental sim-to-real transfer for conversational agents.

Moved here in v0.64.0 from ``stateset_agents.training``,
``stateset_agents.evaluation`` and ``stateset_agents.environments``:

- :mod:`.transfer` -- ``SimToRealTransfer``, user behavior models, domain adaptation
- :mod:`.domain_randomization` -- persona/scenario randomization
- :mod:`.conversation_simulator` -- ``ConversationSimulator`` environment
- :mod:`.metrics` -- ``SimToRealMetrics`` and ``SimToRealEvaluator``

These carry no API stability guarantee (see :mod:`stateset_agents.experimental`).
Import from the submodules directly; this package does not import torch.
"""
