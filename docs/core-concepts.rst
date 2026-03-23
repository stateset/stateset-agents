Core Concepts
=============

StateSet Agents treats conversations as first-class RL episodes.

Core building blocks:

- Agents: `stateset_agents.core.agent.MultiTurnAgent`, `stateset_agents.core.agent.ToolAgent`
- Environments: `stateset_agents.core.environment.ConversationEnvironment`
- Trajectories: `stateset_agents.core.trajectory.MultiTurnTrajectory`
- Rewards: `stateset_agents.rewards` (multi-objective, rule-based, LLM-judge)
- Training: `stateset_agents.training` (GRPO/GSPO family and more)

