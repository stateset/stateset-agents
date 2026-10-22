# Reward contract

Every reward function and reward component shipped in `stateset_agents`
keeps the rules below, and environments, trainers and evaluators rely on them.
The rules are enforced by:

- `tests/unit/test_reward_contracts.py`, which discovers every reward in
  `stateset_agents.core.reward` and `stateset_agents.rewards` plus every
  `create_*_reward` factory;
- `tests/unit/test_reward_gaming.py`, which replays known reward-hacking
  strategies (including every shipped factory in a real environment);
- `tests/unit/test_episode_scoring_paths.py`, which checks that every scoring
  path gives an episode the same reward;
- `tests/unit/test_reward_fail_closed.py` and
  `tests/unit/test_reward_determinism.py`.

## Scores

- A score is a finite float. NaN and infinity are never valid results.
- The range is `[0, 1]` unless the reward declares another one in a
  `score_range = (low, high)` class attribute. `CompositeReward` with
  `weighted_sum` spans `[0, sum of weights]` unless its weights sum to 1 or
  `normalize_weights=True`.
- The same turns and context give the same score on a repeated call, bit for
  bit: composers combine with `math.fsum` and never renormalize stored weights
  between calls.
- A reward never mutates the turns or context it is given.

## The reward context

Every caller builds the context with
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
- Inside a `CompositeReward` that holds an evidence reward, the evidence
  children get the evidence context and every other child gets the standard
  context above (scenario keys and environment state).
- When an environment scores a single step it adds
  `reward_scope="latest_turn"` (see below).

## Who scores an episode

Each episode is scored exactly once, by one function:
`reward_context.score_episode(trajectory, reward_fn, environment, scenario)`.
`MultiTurnGRPOTrainer` (training and batched rollouts), periodic evaluation
(`evaluate_agent`), post-training evaluation and the distributed trainer all
use it, so the same episode earns the same reward on every path.

- An environment with a `reward_fn` applies it during the episode and records
  `reward_fn_applied`, `reward_fn_calls`, `reward_breakdown`,
  `environment_rewards` and, on failure, `reward_error` in the trajectory's
  `metadata`.
- Without a `reward_fn`, or when `reward_fn` is the environment's own object
  (the default `train()` wiring) and the environment applied it, the
  environment's reward stands; the reward is not called again.
- Otherwise `score_episode` applies `reward_fn` exactly as an environment
  would (per step or at the end, see below) and adds the result to the
  environment's own step rewards. The total therefore does not depend on
  whether the reward is attached to the environment or the trainer; per-step
  rewards it applies are scored against the final environment state.
- `sum(turn_rewards) == total_reward` always holds.
- A `ConversationEnvironment` without any reward uses a built-in length
  heuristic. It is a stand-in reward function: averaged over the episode's
  steps, and dropped when a trainer supplies its own reward. The trainer warns
  when that heuristic is the training signal.

## Reward types

| Type | When it is applied | Episode reward |
| --- | --- | --- |
| `IMMEDIATE`, `DENSE` | After every assistant turn, before the next user message | Mean of the step scores, credited as `score / steps` per step |
| `CUMULATIVE`, `SPARSE` | Once, at episode end, on the whole conversation | That one score, on the last step |
| untyped (no `reward_type`) | Once, at episode end (per step only in a custom environment that sets `scores_steps`) | As above |

A reward function therefore contributes at most its range to an episode,
however long the episode runs: padding or stalling never pays.

A per-step call passes the whole conversation so far for context, marked
`context["reward_scope"] == "latest_turn"`. Rewards that aggregate over
assistant turns must then score only the newest one (use
`reward_context.assistant_turns_in_scope(turns, context)`). The built-in
rewards, `CompositeReward` and `MultiObjectiveRewardFunction` honor the marker.

Environments add their own step rewards to the reward function's score, for
example `TaskEnvironment`'s progress reward and one-time completion bonus.
`TaskEnvironment` completes a required action only when the agent declares it
explicitly (a structured tool call, an `"action"` entry in the turn metadata,
or a JSON object such as `{"action": "load_data"}`); keyword matching is an
opt-in (`match_action_keywords=True`) and is gameable.

## Failures

There are no stand-in scores. A failure either raises or is flagged, and a
flagged episode is never trained on.

- A reward that sets `fail_on_error = True` may raise; environments,
  composers and trainers propagate the error.
- Judges fail closed: `LLMJudge` raises `JudgeUnavailableError` when it gets no
  verdict (never cached), and `LLMJudgeReward`, `LLMJudgeRewardComponent` and
  `ModelBasedRewardComponent` propagate judge failures, replies without a
  score and scores outside `[0, 1]`. `LLMJudgeRewardWithFallback` keeps its
  documented heuristic fallback and records `judge_error` in its metadata.
- `CompositeReward` and `MultiObjectiveRewardFunction` leave a failed or
  non-finite component out, spread its weight over the components that
  scored, and list it in `metadata["failed_components"]`. If no component
  scored (or none is configured) they raise `ValueError`.
- Environments log a warning, record `metadata["reward_error"]` and contribute
  0.0 for that call. `score_episode` raises for such an episode, so the trainer
  drops it (in batched rollouts, only that episode) and `evaluate_agent`
  raises (or, with `fail_on_error=False`, counts it as a failed episode).
- `Environment.get_reward` raises for a failed or non-finite reward and when
  the environment has no `reward_fn`. `ComputationalGRPOEngine` drops
  trajectories whose reward failed and `SingleTurnGRPOTrainer` drops
  generations whose reward failed.
- `RulerRewardFunction` rejects judge responses with missing, duplicate or
  unknown trajectory ids or out-of-range scores (retried, then the heuristic
  fallback when enabled).
- `ModelBasedRewardComponent` without a client warns once that it scores by
  length, or raises with `strict=True`.

## Adding a reward

1. Keep the score finite and in range, and declare `score_range` if the range
   is not `[0, 1]`.
2. Read reference data from the context shape above; honor
   `reward_scope="latest_turn"` if you aggregate over assistant turns.
3. Raise on failure rather than returning a neutral value; set
   `fail_on_error = True` if a failure must stop training rather than drop the
   episode.
4. If the constructor needs arguments, register it in `CONSTRUCTORS` in
   `tests/unit/test_reward_contracts.py` (or in `SKIPPED` with a reason if it
   needs a network judge). Add any exploit you guard against to
   `tests/unit/test_reward_gaming.py`.
