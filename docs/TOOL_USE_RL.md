# Tool-use RL with verifiable rewards

Stability: **provisional / beta** (`tool-use-rl` in
[COMPONENT_MATURITY.md](COMPONENT_MATURITY.md)). The names below are
importable from `stateset_agents.environments` and `stateset_agents.rewards`
but are not part of the v1 stable contract
([API_COMPATIBILITY.md](API_COMPATIBILITY.md)) and may change in a minor
release.

This guide covers:

- `ToolUseEnvironment`: multi-turn episodes in which the policy calls typed
  tools, reads their results and answers;
- the **masking guarantee**: tool observations never receive policy gradient;
- two verifiable rewards: `MathAnswerVerifier` and the sandboxed
  `SandboxedCodeVerifier`, with the limits of the sandbox stated plainly;
- the exploit search that gates both verifiers, and the end-to-end evidence.

```python
from stateset_agents.environments import ToolUseEnvironment, calculator_tool
from stateset_agents.rewards import MathAnswerVerifier
from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

environment = ToolUseEnvironment(
    [{"id": "q1", "question": "What is 7919*7907?", "ground_truth": "62615533"}],
    [calculator_tool()],
    max_tool_calls=2,
)
trainer = MultiTurnGRPOTrainer(
    agent=agent, environment=environment, reward_fn=MathAnswerVerifier(), config=config
)
```

## The environment

`stateset_agents.core.environments.tool_use.ToolUseEnvironment` builds on
`Environment` (it inherits `run_episode`, lockstep `run_episodes_batched` and
the shared reward path) and declares `supports_concurrent_episodes`.

**Scenarios.** Each scenario has a question (`question`, `prompt`,
`user_query`, `query` or `task`) and the verifier's reference data:
`ground_truth` for math, `tests` (and `entry_point`) for code. The context
the policy receives (`observation_context`) is an **allowlist**: the question,
the tool log, and the scenario's descriptive keys (`id`, `question`,
`prompt`, `description`, `entry_point`, `examples`...). Every other key, any
reference data whatever it is called (`ground_truth`, `tests`,
`expected_answer`, `correct_facts`, `label`...), stays reward-side only;
`visible_context_keys=` opts extra keys in explicitly. The reward still sees
everything (`tests/unit/test_policy_context_allowlist.py`).

**Tools.** A `ToolSpec` is a name, a JSON-schema `parameters` object, a
deterministic Python callable and a per-call `timeout_seconds`.
`ToolRegistry` validates a call's arguments against the schema (`required`,
`additionalProperties: false`, each property's `type` and `enum`; JSON
booleans are not numbers), runs the callable (a synchronous one in a worker
thread, an async one directly) and returns a `ToolObservation`.
`registry.schemas()` gives the OpenAI-format schemas to pass to a chat
template, e.g. `AgentConfig(chat_template_kwargs={"tools": registry.schemas()})`.
`calculator_tool()` is an exact calculator over `+ - * / ^` and parentheses
(fractions, no floating point, no `eval`, bounded powers).

**Calls.** Each assistant turn is parsed by
`stateset_agents.core.tool_calls.parse_tool_calls`:

- the chat-template form, one or more
  `<tool_call>{"name": "calculator", "arguments": {"expression": "12*7"}}</tool_call>`
  blocks (`arguments` may also be a JSON string);
- the JSON-action form the benchmark sandboxes use
  (`core.environments.json_action.parse_json_action`): the whole reply is
  `{"tool": "calculator", "args": {...}}` (`json_actions=False` turns it off);
- structured OpenAI `tool_calls` on the turn.

JSON is parsed strictly (duplicate keys, `NaN`/`Infinity` and deep nesting
are rejected). A call that cannot be read is a `malformed_call` error
observation, never an exception.

**Observations.** Each call is answered by one `tool`-role turn whose content
is JSON: `{"name": "calculator", "result": 84}` or
`{"name": ..., "error": {"type": ..., "message": ...}}` with `type` one of
`malformed_call`, `unknown_tool`, `invalid_arguments`, `tool_error`,
`timeout`, `unserializable_result`, `budget_exhausted`. Long observations are
truncated (`max_observation_chars`, still valid JSON).

**Episode end and budget.** A turn with no tool call is the final answer and
ends the episode (`COMPLETED`). Every attempted call, malformed or not, spends
one unit of `max_tool_calls`; a call attempted once the budget is spent gets
a `budget_exhausted` observation and ends the episode without an answer
(`FAILED`). `max_turns` (default `max_tool_calls + 1`) also bounds it
(`TIMEOUT`). The final state records the executed evidence: `tool_log`
(every call with its arguments, result or error type),
`executed_tool_calls` (`{"tool": name, "args": arguments}` for each call that
ran and returned a result, the trace `TaskOutcomeReward` compares with a
scenario's `expected_tool_calls`), `tool_calls_used`, `final_answer` and
`termination` (`final_answer`, `budget_exhausted`, `max_turns`).

**Rewards.** The environment adds no reward of its own. Attach the verifier
to the environment or hand it to the trainer; both settle the episode
through `reward_context.score_episode`, with the same total either way
(`tests/unit/test_tool_use_environment.py`). Use SPARSE rewards (both
verifiers are): applied once to the finished conversation, they read the
newest assistant turn, so tool calls are never scored as answers. Both
verifiers also score 0 unless the episode state says it ended with a final
answer (`termination == "final_answer"`), so an episode that ran out of
budget or hit `max_turns` earns nothing even when its last turn states the
right answer beside another call; the termination travels with the
trajectory's final state, so the environment and trainer paths agree
(`test_an_episode_that_ran_out_of_budget_earns_nothing`,
`test_the_code_verifier_also_needs_a_final_answer` in
`tests/unit/test_tool_use_environment.py`).

**Prompts.** With a chat template the `tool` role is rendered by the
template. Without one, `MultiTurnAgent` renders tool turns as
`Tool: <content>` lines (they were previously dropped, so a policy without a
chat template never saw its observations).

**Tool timeouts.** The timeout bounds how long the environment waits. A
synchronous tool runs in a worker thread that Python cannot stop: a tool
stuck in a long computation holding the GIL is not preempted, and an
abandoned thread keeps running. Tools must be deterministic and bounded;
put untrusted computation in the code sandbox, or better a container.

## The masking guarantee

**Tool observations, user turns and the system prompt never receive policy
gradient; every token the policy generated does, including the tool-call
text.** How each loss path ensures it:

| Path | Mask | Observation tokens |
| --- | --- | --- |
| Native token path (`compute_grpo_loss`, default for `MultiTurnAgent.generate_turn`) | one row per assistant turn: `prompt_token_ids` (context) + sampled `token_ids` (response); only response positions are scored | appear only inside a later turn's `prompt_token_ids` |
| Sequence path (`_prepare_inputs_and_labels`) | the template's `{% generation %}` mask, else assistant spans located in the full render (`core.chat_masking.assistant_char_spans`), else `User:/Assistant:/Tool:` role spans without a template | labelled `-100` |
| Sequence path, template whose history cannot be located | conversations **with observation turns are skipped** (nothing labelled) instead of the old full-stream fallback; conversations without them keep that fallback | never trained |
| SFT (`core.chat_masking.assistant_token_mask`, `final_turn_token_mask`, `supervised_turn_sequences`, with `chat_template_render(tokenizer, tools)`) | assistant spans only | outside every span |

`core.chat_masking.OBSERVATION_ROLES` (`tool`, `ipython`, `function`,
`environment`) and `has_observation_turns` name the observation roles.

`tests/unit/test_tool_use_masking.py` proves this on real
`ToolUseEnvironment` episodes from a tiny GPT-2 `MultiTurnAgent` (scripted
reply text; recorded prompt ids, token ids and log-probs are the model's
own):

- the gradient of the GRPO loss with respect to the logits is non-zero at
  exactly the positions that predict reply tokens, for every row, and zero
  at every position predicting a tool-observation token;
- a negative control moves the observation into the response (what a
  full-stream loss would do) and the same probe detects the leak;
- the sequence path labels only assistant text without a template, with a
  ChatML template that renders the `tool` role, and with a
  `{% generation %}` template; an unlocatable template skips the tool
  conversation;
- the SFT masks with a Qwen-style template that renders the tool schemas and
  `<tool_response>` blocks keep only assistant tokens;
- the agent's prompt for the next turn contains the observation.

Limit: Qwen2.5-style templates merge consecutive tool messages into one
block, so a turn with several calls fails the prefix property on the
*sequence* path and that conversation is skipped (the token path is
unaffected).

## Math verifier

`stateset_agents.rewards.MathAnswerVerifier` (SPARSE) scores 1.0 when the
newest assistant reply commits to exactly one final answer equal to the
reference (`ground_truth`, bare or `{"answer": ...}`, as for
`CorrectnessReward`; or `answer=`), else 0.0. Without a reference it raises
`ConfigurationError`.

**The final answer**, in order of authority: every `\boxed{...}`; otherwise
every explicit statement ("the answer is X", "answer: X", "#### X");
otherwise the last number, only when the reply offers exactly one number
beyond those quoted from the question and passes `CorrectnessReward`'s
candidate rule (so worked arithmetic must state its answer). Tool-call
markup is removed first.

**No credit for hedging** (as in `CorrectnessReward`, whose helpers it
reuses): several different committed answers, an alternative offered
("42, or maybe 41", "41 or 42", "41 or \boxed{42}", `\boxed{\text{42 or 43}}`),
a tentative answer ("Maybe 42."), an admission of not knowing, a listed
candidate ("Paris, 42, Glacier") and a negated answer all score 0. So does a
box or statement that hedges inside itself: a list (`\boxed{42, 43}`, unless
the reference is itself a list), `\pm`, `\approx`, "or" inside a box
(`\boxed{42 \text{ or } 43}`), "about", "or so". And so does an answer
contradicted after it is given: a retraction ("Actually, I think I made an
error") or a different value offered for it ("it is 43", "it could also be
43", "some textbooks say 43"); a restatement of the same value ("It is 6 *
7.") or a check ("42 / 7 = 6") keeps the credit. A committed answer may carry
a unit or restate itself ("42 dollars", "42 (or 42.0)"); text after its
number that adds another number makes it no answer.

**Equivalence** (`stateset_agents.core.math_expressions`): model text never
reaches `eval`, `sympify` or `parse_expr`; a small recursive-descent parser
builds a tuple tree. A numeric reference must be matched by a number
*written out* (`84`, `-3/4`, `0.75`, `\frac{3}{4}`), compared exactly as a
fraction, or within `abs_tolerance` / `rel_tolerance` when set; restating
the question's arithmetic (`\boxed{12*7}`) earns nothing. A symbolic
reference (`3x^2 + 2`) is matched by any equivalent expression: exact
evaluation at eight fixed pseudo-random rational points for rational
functions (pure Python, the same result with or without sympy), and
`sympy.simplify` on node-built expressions for `sqrt`, `sin`... when sympy is
installed. The sympy tree obeys the exact path's limits (`x^(9^9)` and
`sqrt(x)+9^9^8` are refused, not built), `sympy.simplify` runs in a worker
process killed after `sympy_timeout` seconds (default 5; a timeout scores 0
and a fresh worker takes over), and `compute_reward` verifies in a worker
thread, so no answer stalls the trainer's event loop. This cannot tell `(x+1)^2` from `x^2+2x+1`: pass `symbolic=False`
for expand/factor/simplify tasks to require the same written form. A
reference that is not math is compared as normalized text (it must be boxed
or stated).

## Sandboxed code verifier

`stateset_agents.rewards.SandboxedCodeVerifier` (SPARSE) takes the reply's
last fenced Python block (else the whole reply), runs it against test cases
and scores the pass rate. Tests come from the context (scenario `tests` and
`entry_point`) or the constructor:

- function tests `{"args": [...], "kwargs": {...}, "expected": value}` call
  `entry_point`; results are compared as strict JSON (`True` is not `1`,
  tuples equal lists, optional `float_tolerance`);
- program tests `{"stdin": "...", "expected_stdout": "..."}` run the code as
  `__main__` with that stdin and compare stdout (trailing whitespace
  ignored; a non-zero `sys.exit` fails).

Keep the tests **hidden**: different from any examples in the prompt.

**Isolation** (POSIX only; elsewhere scoring raises `RewardUnavailableError`,
the outage the trainers already handle):

- `python -I -S` (isolated mode, no site-packages unless
  `site_packages=True`), an empty environment, a fresh temporary directory
  as the working directory, deleted afterwards, and a new session whose whole
  process group is killed after the run;
- a wall-clock timeout for the run, and rlimits the harness applies to itself
  (soft = hard) before any candidate code runs: CPU seconds, address space,
  file size (writes past it raise instead of killing), `RLIMIT_NPROC = 0` (no
  forks or threads), open files, no core dumps (`SandboxLimits`);
- stdout and stderr are read with size caps; the rest is drained and
  discarded.

**Verdicts the candidate cannot forge.** Expected outputs never enter the
sandbox: the harness receives test *inputs* only, reports each result, and
this process compares. The harness writes one record per test, framed by a
per-run random nonce, to a private copy of stdout and points fd 1 at
`/dev/null`; a test with no record (early exit, crash, limit kill, timeout)
or two records (a forgery) fails. Prose and code that does not compile
score 0 without starting a process.

| Exploit | Why it fails | Regression test (vs. a naive verifier it fools) |
| --- | --- | --- |
| print / hard-code the expected output | expected values are hidden and never sent to the sandbox | `hardcode_visible`, `print_expected` |
| `sys.exit(0)` / `os._exit(0)` before the asserts | a test passes only with its own record | `sys_exit_before_asserts`, `os_exit_before_asserts` |
| monkeypatch the checker, `__eq__` always true | comparison happens here on decoded strict JSON; `json.dumps` is bound before candidate code runs | `monkeypatch_checker`, `always_equal` |
| read the tests file | there is none; the working directory is empty | `read_tests_file` |
| recover the nonce from memory and forge records | values are still compared to hidden expectations; duplicates fail | `test_forged_records_with_the_recovered_nonce_are_rejected` |
| infinite loop | CPU limit or wall clock kills it; a score of 0, not an outage | `test_infinite_loop_is_a_zero_score_not_an_outage` |
| fork bomb | `RLIMIT_NPROC = 0`; the process group is killed | `test_fork_storm_leaves_no_processes_behind` |
| memory / file / thread / recursion abuse, output floods | rlimits and capped reads | `test_resource_abuse_*`, `test_output_floods_*` |

All are in `tests/unit/test_code_verifier_sandbox.py`; each exploit test also
runs the candidate through a naive verifier (candidate and asserts in one
file, pass on exit status 0) and asserts the naive verifier is fooled.

**Limits. This is resource isolation, not a security boundary against hostile
code.** The candidate runs as the trainer's user with that user's filesystem
and network access: it can read files the user can read, open network
connections and write outside its working directory (within the file-size
limit). As root, `RLIMIT_NPROC` is not enforced, and a process that calls
`setsid` escapes the group kill. macOS does not enforce `RLIMIT_AS`. A
program that writes to fd 1 directly, or calls `os._exit` after printing,
loses its output. **For untrusted or adversarial workloads run the candidate
in a container or microVM** (gVisor, Firecracker, a Docker container without
network and with a read-only root), or run the whole trainer inside one.

## Exploit search

Both verifiers are gated by `stateset_agents.evaluation.reward_exploits`
(`tests/unit/test_reward_exploit_search.py`), on their own cases
(`verifier_exploits.MATH_CASES`, `CODE_CASES`: four math questions, a
function task and a stdin/stdout program task). Cases of kind `"math"` and
`"code"` keep the generic families that do not preserve the answer (canned,
stuffing, echo, salad, bare numbers and actions, terse, refusals, hedges,
negations, truncation) and add:

- **math**: the reference with its answer removed or swapped, two boxed
  answers, an answer followed by an alternative, "Maybe L.", an admission of
  uncertainty, a negated box, a wrong box "corrected" later, a dump of fifty
  numbers, a near miss, the question's arithmetic boxed unevaluated, the
  answer only inside a tool call or JSON action, an empty box, a hedge inside
  `\text{}`, a Python injection in a box, and hedges inside or after a box:
  `\boxed{L, D}`, `\boxed{L \pm 1}`, `\boxed{L \text{ or } D}`,
  `\boxed{\approx L}`, a box retracted afterwards, a box followed by "(it
  could also be D)", and "The answer is L. Some textbooks say D.";
- **code**: hard-coding the visible example, printing its output, exiting
  before or inside the tests, `__eq__` always true, patching `json.dumps`,
  forging records with the recovered nonce, reading a tests file, an
  infinite loop, a bounded fork storm, threads, a memory bomb, unbounded
  recursion, `SystemExit` in the function, and plausible wrong programs
  (off by one, wrong formula, correct only for small inputs, echoing the
  input, a non-zero exit).

Generic mutations that can keep a correct answer intact (padding, deleting
half of the "specifics", negating action words) are not adversaries for a
verifier: it rightly credits a correct answer however it is worded.

**Result:** for seeds 0 and 1, on both paths (reward attached to the
environment and held by the trainer), every adversary scores at least 0.01
below the reference. In numbers: 900 math adversary episodes (57 families
over four cases) all score 0 against a reference of 1 (smallest gap 1.0; the
seven hedge-and-contradiction families tied the reference at 1.0 before they
were fixed), and
292 code adversary episodes (48 families over two tasks) leave a smallest gap
of 0.33, from a plausible wrong program (`first_two_only`, which sums only two
of three inputs and passes two of three hidden tests); every exploit family
scores 0. A naive verifier that credits any mention of the label is caught
(`test_the_verifier_search_detects_a_naive_verifier`).

## Evidence

- `tests/integration/test_tool_use_grpo.py`: `MultiTurnGRPOTrainer` trains
  a tiny GPT-2 for three steps on a two-question calculator
  `ToolUseEnvironment` with `MathAnswerVerifier`: verifier rewards vary
  within every group (all advantages non-zero), every step takes the native
  token path and an optimizer step, the loss counts exactly the reply tokens
  (no observation token), and runs are deterministic per seed (scripted and
  sampled rollouts).
- `tests/learning/test_tool_use_learns.py` (`LEARNING_SEEDS=0,1,2`): GRPO
  with `MathAnswerVerifier` as the only reward teaches a one-layer GPT-2 to
  call the calculator for products it cannot compute. A tiny model cannot
  learn to copy digits within a CPU budget, so a supervised warm start
  (assistant tokens only) teaches the call format and the per-question calls
  and answers, and leaves the decision to call the tool below one half; a KL
  penalty to that warm start (`beta=0.1`) keeps the call text intact while
  the decision moves (without it, one seed in three drifted the call text and
  gained no correct calls). After 30 GRPO steps (2 questions x 8 episodes per
  step, about a minute per seed on one CPU thread):

  | Seed | P(open a tool call), held-out questions | P(exact correct call), training questions | Sampled correct calls (64 episodes) | Sampled verifier pass rate |
  | --- | --- | --- | --- | --- |
  | 0 | 0.43 -> 0.81 | 0.39 -> 0.69 | 0.28 -> 0.69 | 0.30 -> 0.66 |
  | 1 | 0.26 -> 0.77 | 0.24 -> 0.68 | 0.22 -> 0.78 | 0.22 -> 0.78 |
  | 2 | 0.40 -> 0.81 | 0.36 -> 0.69 | 0.39 -> 0.64 | 0.39 -> 0.66 |

  With the reward's sign flipped the exact-call probability falls to 0.00,
  0.04 and 0.12 and the held-out tool-call rate falls too. The test asserts a
  held-out gain above 0.25, an exact-call gain above 0.2, higher sampled
  correct-call and pass rates, and the reverse movement for the flipped
  reward. It is a toy: the transfer it shows is the *decision* to use a tool
  (held-out questions), not arithmetic or copying.

## Deprecation

`stateset_agents.rewards.kimi_k25_reward.CodeExecutionReward` never executed
code: it scored code-block markers and keywords. It is now
`CodeFormattingHeuristic` (component name `code_formatting`); the old name
is a deprecated alias removed in v0.71.0. To reward code that works, use
`SandboxedCodeVerifier`.
