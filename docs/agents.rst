Agents
======

The agent abstractions live under `stateset_agents/core/`.

Typical entrypoints:

- `stateset_agents.core.agent.MultiTurnAgent`
- `stateset_agents.core.agent.ToolAgent`

Generation fidelity
-------------------

``MultiTurnAgent`` returns exactly the text the model sampled:

- ``AgentConfig.stop_sequences`` (default ``[]``) lists explicit stop strings;
  text and recorded ``token_ids`` are truncated at the same token boundary.
  ``LEGACY_STOP_SEQUENCES`` restores the old ``"User:"``/``"System:"``/blank
  line stops.
- ``AgentConfig.enable_persona_hints`` (default ``False``) appends canned
  sentences in ``generate_response`` only; rollout turns never get them.
- ``AgentConfig.top_k`` accepts ``0`` or ``None`` to disable top-k filtering.
- System prompt precedence: a system message already in the conversation
  (for example the environment's ``get_initial_prompt`` turn) wins; the
  agent's ``AgentConfig.system_prompt`` is used only when the conversation has
  no system message.

See ``docs/OBJECTIVES.md`` ("Rollout text is exactly what the policy
sampled") for how this ties into the token-path loss.
