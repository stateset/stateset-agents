# Reward contract

Every reward function and reward component shipped in `stateset_agents`
keeps the rules below, and environments, trainers and evaluators rely on them.
The rules are enforced by:

- `tests/unit/test_reward_contracts.py`, which discovers every reward in
  `stateset_agents.core.reward` and `stateset_agents.rewards` plus every
  `create_*_reward` factory;
- `tests/unit/test_reward_exploit_search.py`, a CI gate that runs the
  automated exploit search below over every shipped factory;
- `tests/unit/test_reward_gaming.py`, which replays known reward-hacking
  strategies (including every shipped factory in a real environment);
- `tests/unit/test_episode_scoring_paths.py` and
  `tests/unit/test_reward_path_parity.py`, which check that every scoring
  path gives an episode, and every trainer a reply, the same reward;
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
on **task grounding** (`stateset_agents.core.grounding`). A gated term pays
`grounding * (0.75 + 0.25 * style)`: style adjusts a grounded reply's reward
by at most a quarter and earns nothing for an ungrounded one. Outcome terms
(correctness against ground truth, completed actions, safety) are not gated.

`grounding = information * (0.5 + 0.5 * relevance) * quality`, where

- **information** counts the specifics a reply adds that the request did not
  contain: numbers and amounts of two or more characters, affirmed action
  words (`grounding.ACTION_TERMS`: "issued", "blocked", "rejoin",
  "multiply"...; "not issued" does not count), each only in a clause that
  also names something else, and reference labels. It is `n / (n + 6)` for
  `n` distinct specifics, so every specific counts. A short label
  (`ground_truth` answer) earns `CorrectnessReward`'s answer credit instead:
  shared among hedged candidates, none when negated, and numbers other than a
  numeric answer are competing answers rather than information. Any other
  word, real or invented, is not information.
- **relevance** counts terms shared with the request (the user's messages and
  the scenario's `user_query`/`prompt`/`context`/`user_goal`/`topic`/task
  description and labels), saturating at two; a near-verbatim echo (80% of a
  request of four or more terms) halves it.
- **quality** is `(1 - boilerplate) * (1 - filler ** 3)`: *boilerplate* is the
  share of request-independent phrasing (`GENERIC_TERMS`, which covers every
  keyword the heuristics reward; a test pins this) and *filler* adds the
  share of words of no recognized kind.

Declining is grounded only when the request asks for help with something
unsafe (`basic_rewards.harmful_request`: "how do I hack...", "where can I buy
drugs", "... without getting caught"), never when it merely mentions an unsafe
topic ("someone stole my card", "how do I avoid scams"). `SafetyReward` scores
0 for unsafe *assistance* (`gives_unsafe_assistance`: instructions involving
an unsafe topic, an unsafe act by the speaker or reader, getting or making
contraband), not for naming a topic, and does not score by length.
Keyword matching in the heuristics is whole-word.

### What the exploit search verifies

`stateset_agents.evaluation.reward_exploits` generates adversarial replies
from a request and a reference answer, deterministically from a seed (it
seeds with CRC-32, never `hash()`), and `search_reward_exploits` plays each
against the reference in a real `ConversationEnvironment` episode, with a
fresh reward attached to the environment and held by a trainer, both settled
by `score_episode`. The families:

| Family | Adversaries |
| --- | --- |
| Templates and stuffing | two canned service templates; the heuristics' keyword lexicon (a fixed stuffing string and 40 sampled `GENERIC_TERMS`) |
| Echo | the request verbatim, shuffled, and followed by the topic keyword |
| Salad | 15 random words (real English and invented), lower and title case; inserted at random positions into an echo, a template plus echo, and stuffing plus echo |
| Bare lists | six random numbers or action words, alone, after an echo, and after a template |
| Terse and refusals | "Thanks, happy to help!", "Done.", "OK, sure."; "I can't help with that." and a longer refusal |
| Hedges and negations (labeled cases) | the label among four distractors as a comma, newline and sentence list, "Maybe X." sentences and "It could be X or Y"; "It is not X." and "X is not the answer." |
| Content-destroying mutations | the reference with all or half of its specifics deleted, its specifics swapped for salad, its actions negated ("not issued"), or truncated to its first third |
| Padding | the reference with a third as many salad or boilerplate words inserted at random positions |

`tests/unit/test_reward_exploit_search.py` runs every family over every
shipped factory (the core, multi-objective and Kimi `create_*_reward`
factories and the `cli train` demo reward: 15 rewards) and nine cases in
`DEFAULT_CASES` (a refund, a Wi-Fi fault, a product question, a calculus
question, an unscripted starter scenario, a sensitive-but-harmless fraud
report, numeric and text QA with labels, and a two-turn delivery
conversation) for seeds 0 and 1. **Verified guarantee:** on those cases every
adversary scores at least **0.01 below** the reference, on both paths. The
search plays 29 to 36 adversaries per case. Run against the rewards as they
were before this hardening (commit `692b2504`), it found
2,446 adversary episodes at or above that line, in all 15 factories; it now
finds none (eight seeds were checked during development; the smallest gap
was 0.0156, for padding).

Run `search_reward_exploits(make_reward, cases)` with your own
`ExploitCase`s to check a custom reward or your own scenarios; an empty
result is the same guarantee for those cases.

### Known limits (not verified)

These are lexical rewards. The search does not cover, and the rewards do not
defend against:

- **Fabricated specifics.** An invented amount, reference number or action
  claim stated about the request ("I've issued a $73.50 credit, reference
  R-5521") is indistinguishable from a true one. Numbers or verbs written
  into the request's own clause ("...can I get a refund 4093 8812") count as
  stated and can beat the reference.
- **Word order.** Rewards read bags of words (clauses aside): a shuffled
  reference scores within 0.01 of the reference, sometimes slightly above.
- **Paraphrase and coverage.** A correct answer in other words gets no label
  credit; words that are neither specifics nor request terms are not
  information, so deleting them is not penalized (it can raise quality); and
  the reference is not the unique best reply (more specifics score higher).
- **Intent and assistance cues are lexical.** A harmful request phrased
  without the listed cues does not ground a refusal, and unsafe assistance
  phrased without the instruction, act or contraband cues is not penalized.
- **Other requests.** The margin is verified on the nine cases above, not
  proven for every request.

Use outcome, verifier or judge rewards for anything these limits matter to.

## Verifiable rewards

`MathAnswerVerifier` and `SandboxedCodeVerifier` (`stateset_agents.rewards`,
provisional; [TOOL_USE_RL.md](TOOL_USE_RL.md)) keep this contract with
verifier semantics:

- They are **SPARSE**: applied once to the finished conversation, they score
  the newest assistant turn (`turns[-1]` for a single reply), so a tool-use
  episode's intermediate tool calls are never read as answers, and an
  episode that ended without an answer scores 0: when the context records
  `termination` (`ToolUseEnvironment` does), anything but `"final_answer"`
  scores 0 whatever the last turn says
  (`reward_context.unanswered_termination`;
  `tests/unit/test_tool_use_environment.py::test_an_episode_that_ran_out_of_budget_earns_nothing`).
- Reference data is required: the math verifier reads `ground_truth` (bare or
  `{"answer": ...}`, as `CorrectnessReward` does) or `answer=`; the code
  verifier reads `tests` and `entry_point` (from the scenario or the
  constructor). Missing or malformed reference data raises
  `ConfigurationError`, never a neutral score.
- A wrong, hedged, crashing or timed-out candidate is a **score** (0, or the
  code pass rate), not an outage. Only a platform that cannot run the sandbox
  (not POSIX) raises `RewardUnavailableError`.
- Model output is never evaluated as Python in the trainer: the math verifier
  parses answers with its own grammar, and candidate code runs only in the
  sandboxed subprocess (resource isolation, not a security boundary).

The exploit search covers both (cases of kind `"math"` and `"code"`,
`stateset_agents.evaluation.verifier_exploits`): the generic families that do
not preserve the answer plus math families (two answers, alternatives, "Maybe
L.", uncertainty, negation, number dumps, near misses, unevaluated question
arithmetic, answers only inside tool calls, `\text{}` hedges, Python
injection, the reference with its answer removed or swapped, lists, `\pm`,
`\approx` and "or" inside a box, and a box or statement retracted or
contradicted afterwards) and code families
(hard-coded or printed visible outputs, early `sys.exit`/`os._exit`,
`__eq__` tricks, patching `json.dumps`, forged result records, reading a tests
file, infinite loops, fork storms, threads, memory and recursion bombs, and
plausible wrong programs). **Verified guarantee:** on `MATH_CASES` and
`CODE_CASES`, for seeds 0 and 1 and both paths, every adversary scores at least
0.01 below the reference. Padding and other mutations that keep a correct
answer intact are not adversaries for a verifier.

## Scores

- A score is a finite float. NaN and infinity are never valid results.
- The range is `[0, 1]` unless the reward declares another one in a
  `score_range = (low, high)` class attribute. `CompositeReward` with
  `weighted_sum` spans `[0, sum of weights]` unless its weights sum to 1 or
  `normalize_weights=True`.
- A declared range is enforced at runtime: when a reward that declares
  `score_range` returns a score outside it, environments, `score_episode`,
  `score_response`, `score_trajectory` and `CompositeReward` (for its
  children) raise `ValueError` (`reward_context.check_score_range`); the score
  is never clamped. Rewards without a declaration are not checked at runtime
  (`tests/unit/test_reward_contracts.py` checks the shipped ones against
  `[0, 1]`). `MathAnswerVerifier` and `SandboxedCodeVerifier` declare
  `(0.0, 1.0)`.
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
- Every single-response path (`SingleTurnGRPOTrainer`, `GSPOTrainer`,
  `GSPOTokenTrainer`, the TRL reward bridge in plain and conversational
  format, `ComputationalGRPOEngine` and the auto-research DAPO/VAPO adapters)
  scores through `reward_context.score_response`: `[user, assistant]` turns
  (`response_turns`), the standard context (`response_reward_context`) and,
  for IMMEDIATE/DENSE rewards, the latest-turn scope a step gets. A reply
  therefore scores exactly as a one-turn episode does
  (`tests/unit/test_reward_path_parity.py` checks this for every shipped
  factory, the core rewards and a model judge, and pins that these consumers
  never call `compute_reward` themselves).
- `Environment.get_reward` scores through `reward_context.score_trajectory`:
  the whole conversation once, with the trajectory's recorded state and
  scenario (`trajectory_reward_context`), evidence rewards included.
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
example `TaskEnvironment`'s progress reward and one-time completion bonus,
which total `task_reward_scale` (default 1.0: two thirds progress, one third
completion) for a completed task, the same scale as a `[0, 1]` reward
function. `TaskEnvironment` completes a required action only when the agent
declares it explicitly (a structured tool call, an `"action"` entry in the
turn metadata, or a JSON object such as `{"action": "load_data"}`), and
executes at most one per step (the first declared; `max_actions_per_step=None`
for real batches), so declaring every action in one reply is not doing them;
keyword matching is an opt-in (`match_action_keywords=True`) and is gameable.

## What the policy sees

The policy's context (`Environment.observation_context`, passed to
`agent_fn`) is an allowlist: `Environment.policy_state_keys` (identifiers,
the conversation, the problem statement, observations) and, inside the
state's `"scenario"`/`"task"`, `policy_scenario_keys`; subclasses add their
own observation keys, and callers opt extra keys in with
`visible_context_keys`. Reference and grading data (`ground_truth`,
`expected_answer`, `tests`, `required_actions`, `expected_response`...) never
reach the policy, whatever they are called, while the reward context above
keeps all of it (`tests/unit/test_policy_context_allowlist.py`).

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
  from a reward (`TypeError`, `AttributeError`, `KeyError`, `ValueError`), a
  `None` result and a non-finite score is a bug or bad data and propagates
  from environments, `score_episode`, `MultiTurnGRPOTrainer` (sequential and
  batched rollouts, post-training evaluation), `SingleTurnGRPOTrainer` and
  `ComputationalGRPOEngine`.
- A rollout step in which every attempted episode lost its reward to an
  outage raises `RewardUnavailableError` rather than training on nothing.

- A reward that sets `fail_on_error = True` may raise; environments,
  composers and trainers propagate the error.
- Judges fail closed: `LLMJudge` raises `JudgeUnavailableError` when it gets no
  verdict (never cached), and `LLMJudgeReward`, `LLMJudgeRewardComponent` and
  `ModelBasedRewardComponent` propagate judge failures. A verdict is accepted
  only as exactly one JSON object with a numeric `score` inside the judge's
  range (`{"score": 0.8}`); prose, bare numbers ("7/10", "0.8"), fenced or
  embedded JSON, string or boolean scores and out-of-range scores are judge
  failures, never salvaged, rescaled or clamped. A pairwise verdict must name
  `A`, `B` or `tie`. `LLMJudge.evaluate(reference=...)` shows the reference
  to the judge and keys its cache on it. A model judge always sees the
  request (the last user turn, else `context["user_query"]`). `LLMJudgeRewardWithFallback` keeps its
  documented heuristic fallback and records `judge_error` in its metadata.
- `CompositeReward` raises `RewardUnavailableError` when any component is
  unavailable (its default `outage_policy = "raise"`), so the episode is
  dropped instead of being scored by the surviving components alone. With
  `composite.outage_policy = "renormalize"` (an explicit opt-in) it leaves the
  component out, spreads its weight over the components that scored, lists
  it in `metadata["failed_components"]`, counts it in
  `breakdown["unavailable_components"]` and flags the result
  `metadata["reward_fallback"]`; environments and `score_episode` copy both to
  the trajectory (`reward_failed_components`). `MultiObjectiveRewardFunction`
  still leaves an unavailable component out and spreads its weight, listing
  it in `metadata["failed_components"]`. If every component was unavailable
  both raise `RewardUnavailableError`; with none configured, `ValueError`.
- Network rewards fail closed by default: `NSRVerifierReward` raises
  `RewardUnavailableError` when the verifier cannot be reached or replies
  without an `approved`/`denied`/`refused` decision
  (`NSRVerifierConfig.error_score` defaults to `None`) and
  `RulerRewardFunction` raises when the judge fails (`fallback_enabled=False`).
  Setting `error_score` or `fallback_enabled=True` is an explicit opt-in; such
  results carry `metadata["reward_fallback"]`, which composites, environments
  and `score_episode` copy to the trajectory's metadata.
- On an outage, environments log a warning, record
  `metadata["reward_error"]` and contribute 0.0 for that call.
  `score_episode` raises `RewardUnavailableError` for such an episode, so the
  trainer drops it (in batched rollouts, only that episode) and
  `evaluate_agent` raises (or, with `fail_on_error=False`, counts it as a
  failed episode).
- `Environment.get_reward` raises for a failed or non-finite reward and when
  the environment has no `reward_fn`. `ComputationalGRPOEngine` drops
  trajectories and `SingleTurnGRPOTrainer` drops generations whose reward is
  unavailable; anything else propagates.
- `RulerRewardFunction` rejects judge responses with missing, duplicate or
  unknown trajectory ids or out-of-range scores (retried, then
  `RewardUnavailableError`, or the flagged heuristic fallback when enabled).
- `CorrectnessReward` without ground truth scores the neutral 0.5 and flags
  `metadata["no_ground_truth"]`; a committed correct answer scores 1.0.
  Evaluation counts only scores above 0.5 as successes, so neutral is never
  success. A text answer shares its credit with every other candidate the
  whole reply offers (comma, "or"/"and", newline and sentence lists, "Maybe
  X. Maybe Y."); a negated answer ("It is not Paris.", "42 is not the
  answer") scores 0, and "no idea"/"not sure" halves the credit.
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
   `tests/unit/test_reward_gaming.py`, and a new adversary family to
   `stateset_agents/evaluation/reward_exploits.py` so the CI gate searches it
   for every factory.
6. A new `create_*_reward` factory in `core.reward` or `stateset_agents.rewards`
   is picked up by the exploit search gate automatically; it must keep every
   family at least 0.01 below the reference on `DEFAULT_CASES`.
7. A verifier (a reward with reference answers or tests rather than lexical
   scoring) is searched on its own `ExploitCase`s with `kind="math"` or
   `kind="code"` (or a new kind with its own families in
   `stateset_agents.evaluation.verifier_exploits`); register it in
   `VERIFIERS` in `tests/unit/test_reward_exploit_search.py`.
