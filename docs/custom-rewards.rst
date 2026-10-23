Custom Rewards
==============

Custom rewards should typically implement the reward protocol used by the
training loop, or compose existing components in `stateset_agents.rewards`.

A custom reward must keep the `reward contract <REWARD_CONTRACT.md>`_ (finite,
in-range, deterministic scores; the shared context shape; ``fail_on_error`` for
failures that must stop training). ``tests/unit/test_reward_contracts.py``
checks every shipped reward against it.

Verifiable rewards
------------------

``stateset_agents.rewards.MathAnswerVerifier`` (committed final answer equal to
a reference) and ``stateset_agents.rewards.SandboxedCodeVerifier`` (pass rate on
hidden tests, run in a resource-limited subprocess) are provisional verifiers;
``docs/TOOL_USE_RL.md`` covers their semantics, the sandbox's limits (resource
isolation, not a security boundary) and the exploit search that gates them.
