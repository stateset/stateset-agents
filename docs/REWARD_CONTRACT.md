# Reward contract

Every reward function and reward component shipped in `stateset_agents`
keeps the rules below, and environments and trainers rely on them. The rules
are enforced by `tests/unit/test_reward_contracts.py`, which discovers every
reward in `stateset_agents.core.reward` and `stateset_agents.rewards` plus every
`create_*_reward` factory, and by `tests/unit/test_reward_gaming.py`, which
replays known reward-hacking strategies.

## Scores

- A score is a finite float. NaN and infinity are never valid results.
- The range is `[0, 1]` unless the reward declares another one in a
  `score_range = (low, high)` class attribute. `CompositeReward` with
  `weighted_sum` spans `[0, sum of weights]` unless its weights sum to 1 or
  `normalize_weights=True`.
- The same turns and context give the same score on a repeated call.
- A reward never mutates the turns or context it is given.

## The reward context

Environments, `MultiTurnGRPOTrainer`, its post-training evaluation, the
distributed trainer and `Environment.get_reward` build the context with
`stateset_agents.core.reward_context.build_reward_context`, so a reward sees one
shape whoever scores the episode:

```python
{
    **scenario,                 # scenario keys at the top level: ground_truth, task, ...
    **environment_state,        # episode state over them (history, completed_actions, ...)
    "scenario": scenario,       # and the scenario itself
}
```

- Environment state wins a key clash, except that an environment value of
  `None` never hides a scenario value.
- Rewards that set `requires_environment_context = True` judge executed
  evidence. They get the environment state and `"scenario"`, but no scenario
  keys lifted to the top level, so reference data in a scenario cannot pose as
  evidence the environment recorded.
- When an environment scores a single step it adds
  `reward_scope="latest_turn"` (see below).
- `evaluate_agent(reward_fn=...)` overrides keep their documented inputs: the
  scenario, or the final environment state for evidence rewards.

## Who scores an episode

Each episode is scored exactly once.

- An environment with a `reward_fn` applies it during the episode and records
  `reward_fn_applied`, `reward_fn_calls`, `reward_breakdown` and, on failure,
  `reward_error` in the trajectory's `metadata`.
- When the trainer's `reward_fn` is the environment's own object (the default
  `train()` wiring) and the environment applied it, the trainer keeps the
  environment's reward and does not call the reward again.
- Otherwise (the environment has no `reward_fn`, or a different one) the
  trainer scores the finished episode once and credits the score to the final
  step, so `sum(turn_rewards) == total_reward` always holds.
- A `ConversationEnvironment` without a `reward_fn` uses a built-in length
  heuristic averaged over the episode's steps (it never grows with episode
  length). The trainer warns when that heuristic is the training signal; pass a
  `reward_fn` for real training.

## Reward types

| Type | When it is applied | Episode reward |
| --- | --- | --- |
| `IMMEDIATE`, `DENSE` | After every assistant turn, before the next user message | Sum of per-step scores |
| `CUMULATIVE`, `SPARSE` | Once, at episode end, on the whole conversation | That one score |
| untyped (no `reward_type`) | Per step in environments that set `scores_steps` (`ConversationEnvironment`), otherwise at episode end | As above |

A per-step call passes the whole conversation so far for context, marked
`context["reward_scope"] == "latest_turn"`. Rewards that aggregate over
assistant turns must then score only the newest one (use
`reward_context.assistant_turns_in_scope(turns, context)`), so each turn counts
once. The built-in rewards, `CompositeReward` and
`MultiObjectiveRewardFunction` honor the marker. Use a `CUMULATIVE` or `SPARSE`
reward when you want one judgement of the whole conversation; with per-step
rewards, longer unscripted episodes collect more reward.

Environments add their own step rewards (for example `TaskEnvironment`'s
progress reward and one-time completion bonus) to the reward function's score.

## Failures

- A reward that sets `fail_on_error = True` may raise; environments,
  composers and trainers propagate the error.
- Any other failure, including a NaN or infinite score, is flagged, never
  silently scored:
  - environments log a warning, record `metadata["reward_error"]` and
    contribute 0.0 for that call; the trainer drops an episode whose own reward
    failed, rather than training on the 0.0;
  - `CompositeReward` and `MultiObjectiveRewardFunction` leave the failed
    component out, spread its weight over the components that scored, and list
    it in `metadata["failed_components"]`;
  - `RulerRewardFunction` rejects judge responses with missing, duplicate or
    unknown trajectory ids or out-of-range scores (retried, then the heuristic
    fallback when enabled);
  - `ModelBasedRewardComponent` without a client warns once that it scores by
    length, or raises with `strict=True`.

## Adding a reward

1. Keep the score finite and in range, and declare `score_range` if the range
   is not `[0, 1]`.
2. Read reference data from the context shape above; honor
   `reward_scope="latest_turn"` if you aggregate over assistant turns.
3. Set `fail_on_error = True` if a missing reference or failed judge must stop
   training rather than be skipped.
4. If the constructor needs arguments, register it in `CONSTRUCTORS` in
   `tests/unit/test_reward_contracts.py` (or in `SKIPPED` with a reason if it
   needs a network judge). Add any exploit you guard against to
   `tests/unit/test_reward_gaming.py`.
