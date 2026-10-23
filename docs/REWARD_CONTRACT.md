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
- `tests/unit/test_reward_fail_closed.py`,
  `tests/unit/test_reward_result_parsing.py` and
  `tests/unit/test_reward_determinism.py`.

## Heuristic factories are shaping signals

The `create_*_reward` factories (`create_helpful_agent_reward`,
`create_customer_service_reward`, `create_domain_reward`, the multi-objective
and Kimi factories, ...) and the starters and presets that default to them
score style: helpfulness phrasing, empathy and action keywords, length,
engagement. They are **shaping signals, not training objectives**. Train
against an outcome, verifier or judge reward wherever one exists
(`TaskOutcomeReward`, `NSRVerifierReward`, `LLMJudgeReward`, a dataset's
ground truth); the starters and presets log this recommendation.

Style alone is gameable, so every style heuristic in a shipped factory is gated
on **task grounding** (`stateset_agents.core.grounding`): a reply earns style
credit only in proportion to how much it engages with the request (terms from
the user's messages and the scenario's `user_query`/`prompt`/`context`/
`user_goal`/`topic`/task description and ground-truth labels) and adds
specific content beyond boilerplate. The boilerplate lexicon covers every
keyword the heuristics reward. A canned template ("I understand, happy to help,
let me check...") or keyword stuffing has no grounding and earns no style
credit, so in every shipped factory, in a real environment, they score
strictly below a task-specific answer (`tests/unit/test_reward_gaming.py`).
Outcome terms (correctness against ground truth, completed actions, safety)
are not gated. Keyword matching in the heuristics is whole-word with
diminishing returns per distinct keyword.

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
  environment's own step rewards (including a subclass's own
  `_calculate_step_reward`). The total therefore does not depend on whether the
  reward is attached to the environment or the trainer: environments record
  the state each step's reward is scored against in `trajectory.state_history`,
  and per-step rewards are replayed on those snapshots with the same turns
  (system prompt included) the environment would have passed.
- Single-turn trainers (`SingleTurnGRPOTrainer`, `GSPOTrainer`,
  `GSPOTokenTrainer`, the TRL reward bridge) build the same context shape
  (`build_reward_context` / `response_reward_context`).
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

## Results

A reward returns a `RewardResult` (or anything with `.score`, or its
`.total_reward` alias), a mapping with `"score"`, or a real scalar (Python,
NumPy or a 0-d tensor). Every caller parses results with
`reward_context.reward_score`; anything else (`None`, a string, a list) raises
`TypeError` on every path. Turns given as `{"role", "content"}` mappings are
normalized to `ConversationTurn` (`reward_context.normalize_turns`).

## Failures

There are no stand-in scores. A failure either raises or is flagged, and a
flagged episode is never trained on.

- Only an outage is absorbed: `RewardUnavailableError` (judge, network or
  verifier unavailable; `JudgeUnavailableError` is one). Any other exception
  from a reward, including `TypeError`/`AttributeError` and non-finite scores,
  is a bug and propagates.

- A reward that sets `fail_on_error = True` may raise; environments,
  composers and trainers propagate the error.
- Judges fail closed: `LLMJudge` raises `JudgeUnavailableError` when it gets no
  verdict (never cached), and `LLMJudgeReward`, `LLMJudgeRewardComponent` and
  `ModelBasedRewardComponent` propagate judge failures, replies without a
  score and scores outside `[0, 1]`. `LLMJudgeRewardWithFallback` keeps its
  documented heuristic fallback and records `judge_error` in its metadata.
- `CompositeReward` and `MultiObjectiveRewardFunction` leave an unavailable
  component out, spread its weight over the components that scored, and list
  it in `metadata["failed_components"]`. If every component was unavailable
  they raise `RewardUnavailableError`; with none configured, `ValueError`.
- Network rewards fail closed by default: `NSRVerifierReward` raises
  `RewardUnavailableError` when the verifier cannot be reached
  (`NSRVerifierConfig.error_score` defaults to `None`) and
  `RulerRewardFunction` raises when the judge fails (`fallback_enabled=False`).
  Setting `error_score` or `fallback_enabled=True` is an explicit opt-in; such
  results carry `metadata["reward_fallback"]`, which composites, environments
  and `score_episode` copy to the trajectory's metadata.
- Environments log a warning, record `metadata["reward_error"]` and contribute
  0.0 for that call. `score_episode` raises for such an episode, so the trainer
  drops it (in batched rollouts, only that episode) and `evaluate_agent`
  raises (or, with `fail_on_error=False`, counts it as a failed episode).
- `Environment.get_reward` raises for a failed or non-finite reward and when
  the environment has no `reward_fn`. `ComputationalGRPOEngine` drops
  trajectories whose reward failed and `SingleTurnGRPOTrainer` drops
  generations whose reward failed.
- `RulerRewardFunction` rejects judge responses with missing, duplicate or
  unknown trajectory ids or out-of-range scores (retried, then
  `RewardUnavailableError`, or the flagged heuristic fallback when enabled).
- `CorrectnessReward` without ground truth scores the neutral 0.5 and flags
  `metadata["no_ground_truth"]`; a correct answer scores 1.0. Evaluation
  counts only scores above 0.5 as successes, so neutral is never success.
- `ModelBasedRewardComponent` without a client warns once that it scores by
  length, or raises with `strict=True`.

## Adding a reward

1. Keep the score finite and in range, and declare `score_range` if the range
   is not `[0, 1]`.
2. Read reference data from the context shape above; honor
   `reward_scope="latest_turn"` if you aggregate over assistant turns.
3. Raise on failure rather than returning a neutral value: raise
   `RewardUnavailableError` for an outage composites may route around, and
   set `fail_on_error = True` if a failure must stop training rather than drop
   the episode.
4. If the reward scores style rather than outcome, set `shaping = True` so the
   factories gate it on grounding, and add its keywords to
   `core.grounding.GENERIC_TERMS` (a test checks the built-in lexicons).
5. If the constructor needs arguments, register it in `CONSTRUCTORS` in
   `tests/unit/test_reward_contracts.py` (or in `SKIPPED` with a reason if it
   needs a network judge). Add any exploit you guard against to
   `tests/unit/test_reward_gaming.py`.
