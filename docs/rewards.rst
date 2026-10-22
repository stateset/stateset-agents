Rewards
=======

Reward functions live under `stateset_agents/rewards/`.

Highlights:

- Multi-objective rewards (`stateset_agents.rewards.multi_objective_reward`)
- Rule-based rewards (`stateset_agents.rewards.ruler_reward`)
- Optional LLM-as-judge helpers (`stateset_agents.rewards.llm_judge`)
- Kimi-K2.5 helpers (`stateset_agents.rewards.kimi_k25_reward`)

Every reward follows the `reward contract <REWARD_CONTRACT.md>`_: finite scores
in ``[0, 1]`` unless declared otherwise, one reward-context shape (scenario keys
at the top level plus ``"scenario"``), each episode scored exactly once, the
difference between ``IMMEDIATE`` and ``CUMULATIVE``/``SPARSE`` rewards, and how
failures are flagged instead of silently scored.
