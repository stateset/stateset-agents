Environments
============

Environments define episode reset/step logic for training.

Typical entrypoint:

- `stateset_agents.core.environment.ConversationEnvironment`

ConversationEnvironment system prompt and episode end
-----------------------------------------------------

- ``get_initial_prompt`` emits a system turn only when one is explicitly
  provided: the ``system_prompt`` constructor argument, a scenario's
  ``"system_prompt"``, or a scenario's ``"context"`` (``"Context: ..."``);
  ``persona`` is appended to that turn. Otherwise it returns ``""`` so the
  agent's own ``AgentConfig.system_prompt`` applies. When both exist, the
  environment's system turn takes precedence.
- Unscripted episodes end on an assistant farewell (``"goodbye"``, ``"bye"``,
  ``"see you"``, ``"talk to you later"`` as whole words) unless
  ``end_on_farewell=False``. Scenarios with scripted ``user_turns`` never end
  early on a farewell: the script (and ``max_turns``) define their length.
