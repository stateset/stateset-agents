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
  `{"tool": "calculator", "args": {...}}` (`json_actions=False` turns it off).
  Only a whole-reply JSON object that names a tool (a `"tool"` key, or
  `"name"` with `"arguments"`) is a call; any other JSON reply, such as
  `{"answer": "84"}`, is a final answer and spends no budget;
- structured OpenAI `tool_calls` on the turn. A structured call keeps its own
  `id`; only calls without one are numbered `call_<n>` (unique within the
  episode).

JSON is parsed strictly (duplicate keys, `NaN`/`Infinity` and deep nesting
are rejected). A call that cannot be read is a `malformed_call` error
observation, never an exception.

**Observations.** Each call is answered by one `tool`-role turn whose content
is JSON: `{"name": "calculator", "result": 84}` or
`{"name": ..., "error": {"type": ..., "message": ...}}` with `type` one of
`malformed_call`, `unknown_tool`, `invalid_arguments`, `tool_error`,
`timeout`, `unserializable_result`, `budget_exhausted`. The turn carries the
call's `tool_call_id` and the tool `name` (in `turn.metadata` and in the
history passed to the policy, next to the assistant turn's `tool_calls`). An
observation's content never exceeds `max_observation_chars` characters (at
least 128), measured after JSON escaping, and stays valid JSON marked
`"truncated": true`; an error keeps its `type`.

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
`termination` comes only from the environment's recorded state; a scenario key
of that name is ignored, so a scenario cannot make an unanswered episode
score.

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

**Tool timeouts.** The timeout bounds the tool's own run time. A synchronous
tool runs in a thread of its own, started for that call, and its timeout
counts from when it starts running there, so concurrent calls never queue into
timeouts even when the process's shared executor is saturated; an async tool
is cancelled at its timeout. Python cannot preempt a thread holding the GIL,
and an abandoned thread keeps running (the interpreter waits for it at exit).
Tools must be deterministic and bounded; put untrusted computation in the code
sandbox (its docker backend: see the code verifier below).

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

The verifier reads its own final answer (a self-contained reader in
`rewards._math_reading`, not `CorrectnessReward`'s helpers), so a full
chain-of-thought is judged by its ending, not its reasoning. Tool-call markup
is removed first, and the text after a `</think>` block is read first.

**The final answer**, in order of authority:

1. the last `\boxed{...}`;
2. otherwise the last explicit statement ("the answer is X", "answer: X",
   "#### X", "the solutions are X"); a statement with no number or expression
   ("the answer is consistent") is ignored;
3. otherwise a closing sentence with one value and nothing numeric or
   questioning after it ("So the bottle holds 2500 mL.");
4. otherwise one lone new number; question numbers are matched by written form
   ("2/6" is not the question's "1/3"), and a reply cut off mid-step gives no
   answer.

**Reasoning before the final answer's sentence costs nothing:** doubt,
checks, other numbers, and an earlier answer followed by a correction cue
(`\boxed{160}. Wait... \boxed{168}` scores 168). An earlier *different* answer
with no correction cue between (`\boxed{41} \boxed{84}`) scores 0. A statement
corrected afterwards and re-concluded with an equation is replaced by the new
conclusion; a box followed by a correction to another value scores 0.

**No credit for hedging.** The hedge and retraction checks apply to the
answer, its own sentence and the text after it (rules of the reader's own, not
`CorrectnessReward`'s): several different committed answers, an alternative
offered ("41 or 42", "41 or \boxed{42}", "43 is also possible", "Another
possible answer is 43", "\boxed{42} (approximately)", "... but I might be
wrong", "Or is it \boxed{42}?"), a tentative answer ("Maybe 42."), an
admission of not knowing, a listed candidate, a negated answer, and a value
offered or retracted after the answer ("On second thought, 41.", "it is 43",
"some textbooks say 43") all score 0. So does a box that hedges inside itself:
a list (`\boxed{42, 43}`), `\pm`, `\approx`, "or" inside a box, "about", "or
so" -- **unless the reference is itself a `\pm`/solution set or list**. A
restatement of the same value ("It is 6 * 7.") or a check ("42 / 7 = 6")
keeps the credit, as do "I hope it is correct", `**168**` and
`\boxed{168\,\text{pens}}`.

**Equivalence** (`stateset_agents.core.math_expressions`): model text never
reaches `eval`, `sympify` or `parse_expr`; a small recursive-descent parser
builds a tuple tree. A numeric reference must be matched by a number
*written out* (`84`, `-3/4`, `0.75`, `\frac{3}{4}`), compared exactly as a
fraction, or within `abs_tolerance` / `rel_tolerance` when set; restating
the question's arithmetic (`\boxed{12*7}`) earns nothing. `50\%` reads as 50 or 0.5, and a reference
written as an expression (`2^{10}`) also accepts that same written form and
`1024` (but not `4^5`); the newly-read LaTeX/markdown forms include `\frac12`,
`\tfrac`/`\dfrac`, `\,`, `\text`/`\textbf`/`\mathbf` wrappers, `**bold**`,
`^\circ`/`°`, `\$`, `×`, `÷`, Unicode minus and `1{,}000`. A `\pm` or
solution-list reference matches the same set in any order and notation
(`\pm 3`, `±3`, `3, -3`, `x = 3 or x = -3`); a tuple or interval is compared
as text. A symbolic reference (`3x^2 + 2`) is matched by any equivalent
expression: exact evaluation at **eight pseudo-random rational points drawn
from a generator seeded by a hash of both expressions** (deterministic per
pair, but not a fixed public point set -- a polynomial built to vanish at the
old fixed points is rejected), and `sympy.simplify` on node-built expressions
for `sqrt`, `sin`... when sympy is
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
  `__main__` with that stdin on a real file on fd 0 (`input()`, `sys.stdin`,
  `open(0).read()`, `os.read(0, n)` work), and `exit()`/`quit()` exist;
  stdout is compared (trailing whitespace ignored; a non-zero `sys.exit`
  fails).

In function mode, code after the entry point that fails at load is reported as
`load_warning` and the function is still tested; an `if __name__ ==
"__main__":` example block does not run, so an "Example usage" block after the
solution never fails it.

Keep the tests **hidden**: different from any examples in the prompt.

**Isolation backends** (`isolation=` on `SandboxedCodeVerifier` and
`run_sandboxed`; default `"auto"`). Both run the same harness with the same
limits, record framing and parent-side comparison, so a candidate scores the
same on either; they differ in what they keep the candidate away from.

- `"docker"`: one fresh container per verification (`docker run --rm`; no
  container is reused, so nothing carries over between candidates) with
  `--network none` and `--ipc none`, a `--read-only` root and one small
  `--tmpfs` at `/sandbox` (`nosuid,nodev,noexec`, 64 MiB plus the stdin files)
  holding the empty working directory, a non-root numeric `--user`
  (`65534:65534`), `--cap-drop ALL`, `--security-opt no-new-privileges`, the
  daemon's default seccomp and AppArmor profiles (never overridden),
  `--pids-limit` (2 plus `max_processes`, at least 32 so `runc init` can start the container; the harness's `RLIMIT_NPROC` is what stops forks), `--memory` = `--memory-swap` (the
  `memory_bytes` limit plus the tmpfs plus 64 MiB, so `RLIMIT_AS` still fails
  an allocation first, as in the rlimit backend) and `--cpus 1`, no host
  mounts (the harness is a command-line argument, the payload arrives on
  stdin), an empty environment (`/usr/bin/env -i`), `--log-driver none` and
  `--pull never`. Configure it with `DockerSandboxConfig` (image, user, cpus,
  tmpfs, labels, `allow_pull`). The harness runs under a tiny PID 1
  supervisor so a `RLIMIT_CPU` kill still reports `cpu_limit` (PID 1 ignores
  `SIGXCPU`) and so a container kills its candidate at its own deadline even
  if the trainer dies. A container that never starts, an unreachable daemon or
  a missing image raise `RewardUnavailableError` (no candidate code ran);
  once the harness runs, every outcome is a score, so a candidate cannot
  forge an outage.
- `"rlimit"`: a local `python -I -S` subprocess (POSIX only): an empty
  environment, a fresh temporary working directory deleted afterwards (the
  stdin files live in a sibling directory, so the cwd stays empty), a new
  session whose whole process group is killed after the run and, on Linux
  5.13+ with Landlock in the kernel's LSM list (`landlock_available()`),
  Landlock filesystem rules the unprivileged harness applies to itself
  before any candidate code runs: it may read the interpreter's standard
  library, `/usr` and the system library directories, its own stdin files
  and its working directory, and nothing else (no `/proc`, home directory,
  credential or dataset files; no writes elsewhere). A Landlock-confined
  process also cannot read the `/proc/<pid>/environ` or memory of processes
  outside its domain, such as the trainer. `metadata["limits_applied"]`
  lists `LANDLOCK` when it was applied. As root this backend needs an
  explicit opt-in (`allow_root_rlimit=True` or `STATESET_SANDBOX_ALLOW_ROOT=1`):
  root ignores `RLIMIT_NPROC`.
- `"auto"` uses docker when the daemon is reachable and the image is present
  (`docker_sandbox_status()`), else rlimit with a one-time warning that names
  what is not isolated. It decides once per process; after choosing docker it
  never falls back (a broken daemon is an outage, not a silent downgrade).
  `resolve_isolation()` tells a training script which backend it will get,
  and `metadata["isolation"]` records the backend that scored each reply.

**Pre-pull the image.** The default is `python:3.12-slim` pinned by digest
(`DEFAULT_DOCKER_IMAGE`); a custom image must be pinned by digest too
(`name@sha256:...` or an image ID) and provide `/usr/bin/env` and the
interpreter at `DockerSandboxConfig.python`. Pull it before training:

```bash
docker pull "$(python -c 'from stateset_agents.rewards.code_verifier import DEFAULT_DOCKER_IMAGE as i; print(i)')"
```

Without it, `isolation="docker"` raises `RewardUnavailableError` naming the
`docker pull` command, and `"auto"` falls back to rlimit with the warning;
nothing is downloaded during training unless `allow_pull=True`. The image's
Python (3.12 for the default) runs the candidate, so pick an image whose
version and packages match what the tasks assume (`site_packages=True`
exposes the image's site-packages, not the trainer's). Requires Docker 20.10+
with a Linux daemon. Every container carries the label
`stateset-agents.sandbox=code-verifier` (plus `DockerSandboxConfig.labels`).

**What each backend isolates**, measured with the harmless probes in
`tests/unit/test_code_verifier_sandbox.py` (Docker 27.4 on Linux 5.15, which
has Landlock ABI 1):

| A candidate tries to | `docker` | `rlimit` + Landlock | `rlimit` without Landlock (macOS, Linux < 5.13) |
| --- | --- | --- | --- |
| read the trainer's environment (`os.environ`) | empty | empty | empty |
| read `/proc/<ppid>/environ` | its parent is the in-container supervisor (empty environment) | denied | **readable** (the trainer's variables, API keys included) |
| list `/proc`, see the trainer's pid | only its own 2 processes | denied | **596 processes, trainer included** |
| read a dataset file holding expected outputs | not visible | denied | **read** |
| write outside its working directory | read-only root | denied | **written** |
| connect to a listener on the host's localhost | refused (own loopback only) | **connected** | **connected** |
| signal (kill) the trainer | no (own pid namespace) | **yes** (same user) | **yes** |
| run as | uid 65534, no capabilities | the trainer's user | the trainer's user |
| see the test *inputs* (not the expected outputs) | yes (they are in its own process) | yes | yes |

The candidate always sees the test inputs: the harness holding them runs in
its process, and frame inspection finds them. Expected outputs never enter
the sandbox, and in a container (no network, nothing persistent) the inputs
cannot leave it. **For untrusted or adversarial workloads use the docker
backend, or a stronger runtime:** a container shares the host kernel, so a
kernel exploit escapes it; gVisor (`runsc`) or a Firecracker microVM around
the trainer, or a dedicated host, closes that.

**Latency.** A container per verification adds container start-up (a bare
`docker run --rm` of the interpreter takes about 0.5 s). Median time per
verification (three hidden tests) on an 8-CPU Linux host with Docker 27.4:
0.60 to 0.75 s for docker against 0.05 s for rlimit at load average 11, and
0.8 to 1.7 s against 0.15 to 0.19 s at load average 40 to 120.
`test_median_verification_latency` prints the medians. A warm container pool
is not used: a reused container would carry one candidate's files and
processes into the next.

**Limits common to both backends:**

- **CPU time decides** (`cpu_seconds`, `RLIMIT_CPU`, `status="cpu_limit"`);
  the wall clock (`timeout_seconds`, default 30 s) is only a backstop for a
  stalled run (`status="wall_timeout"`, plus `metadata["cpu_seconds"]` on
  Linux). In a container the backstop starts when the harness reports that it
  is running, so a slow daemon delays a run without failing it (start-up has
  its own `startup_timeout_seconds`, an outage when exceeded). At most
  `sandbox_concurrency()` sandboxes run at once per process, whichever
  backend (default: usable CPUs; `STATESET_SANDBOX_CONCURRENCY` /
  `set_sandbox_concurrency(n)`); others wait before their clocks start, so 46
  identical correct solutions scored at once all score 1.0. CPU time still
  varies about 2x under full contention, so keep `cpu_seconds` several times
  a correct solution's need. Other rlimits the harness applies to itself
  (soft = hard) before any candidate code runs: address space, file size
  (writes past it raise instead of killing), `RLIMIT_NPROC = 0` (no forks or
  threads), open files, no core dumps (`SandboxLimits`);
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
| read a tests file in the working directory, its parent or /tmp | the working directory is empty; tests never go there | `read_tests_file` |
| recover the nonce from memory and forge records | values are still compared to hidden expectations; duplicates fail | `test_forged_records_with_the_recovered_nonce_are_rejected` |
| infinite loop | CPU limit kills it (`cpu_limit`); a sleeper meets the backstop (`wall_timeout`); both are a score of 0, not an outage | `test_infinite_loop_is_a_zero_score_not_an_outage`, `test_wall_clock_backstop_kills_a_sleeping_candidate` |
| fork bomb | `RLIMIT_NPROC = 0`; the process group is killed | `test_fork_storm_leaves_no_processes_behind` |
| memory / file / thread / recursion abuse, output floods | rlimits and capped reads | `test_resource_abuse_*`, `test_output_floods_*` |
| read expected outputs from a dataset file on disk | docker: the file is not in the container; rlimit: Landlock denies it. **Not defended by rlimit without Landlock** (it runs as the trainer's user) | `test_reading_expected_outputs_from_a_dataset_on_disk` (0 where files are confined, pinned at 1.0 otherwise); the gate excludes it by name only there (`KNOWN_VERIFIER_LIMITS`) |
| read `/proc/<ppid>/environ`, list `/proc`, write outside the working directory, reach a host socket | see the table above | `test_docker_candidates_find_no_way_out`, `test_rlimit_candidates_are_confined_to_their_files`, `test_candidates_cannot_read_the_trainers_environment` |

All are in `tests/unit/test_code_verifier_sandbox.py`; each exploit test also
runs the candidate through a naive verifier (candidate and asserts in one
file, pass on exit status 0) and asserts the naive verifier is fooled.

Every sandbox test runs on both backends (skipped cleanly where one is
unavailable: no POSIX rlimits, no reachable daemon, or the image not pulled),
and the exploit gate below runs the code families on both.

**Other limits.** As root, `RLIMIT_NPROC` is not enforced (hence the opt-in),
and in the rlimit backend a process that calls `setsid` escapes the group
kill. macOS does not enforce `RLIMIT_AS`. A program that writes to fd 1
directly, or calls `os._exit` after printing, loses its output. Landlock ABI 1
also denies renaming or linking a file into another directory inside the
working directory. The rlimit backend never restricts the network or signals
(Landlock ABI 4 and 6 could, on newer kernels; they are not used).

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
  forging records with the recovered nonce, reading a tests file from the
  working directory, its parent or /tmp, an infinite loop, a bounded fork
  storm, threads, a memory bomb, unbounded recursion, `SystemExit` in the
  function, and plausible wrong programs (off by one, wrong formula, correct
  only for small inputs, echoing the input, a non-zero exit); and, as a named,
  pinned known limit **excluded from the gate**, `read_dataset_on_disk` (the
  gate writes each code case's tests to a temp directory standing in for a
  dataset).
- **symbolic** (a provisional third kind for `SymbolicPhysicsRewardFunction`):
  constants (0, 1, pi, `x - x`, `sin**2+cos**2`) that satisfy symmetry, parity
  and zero constraints trivially, and, when the task has a target, near misses
  (scaled, negated, shifted) of the reference relation.

Generic mutations that can keep a correct answer intact (padding, deleting
half of the "specifics", negating action words) are not adversaries for a
verifier: it rightly credits a correct answer however it is worded.

**Result:** for seeds 0 and 1, on both paths (reward attached to the
environment and held by the trainer), every adversary **except the named
`KNOWN_VERIFIER_LIMITS` family** scores at least 0.01 below the reference. In
numbers: 900 math adversary episodes (57 families over four cases) all score 0
against a reference of 1; the symbolic constants and near misses all score 0;
and 300 code adversary episodes (49 families) leave a smallest gap of 0.33,
from a plausible wrong program (`first_two_only`, which sums only two of three
inputs and passes two of three hidden tests); every code family except
`read_dataset_on_disk` scores 0, while `read_dataset_on_disk` scores 1.0 and
is pinned and excluded. A naive verifier that credits any mention of the label
is caught (`test_the_verifier_search_detects_a_naive_verifier`).

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
