# Check a reward before RL training

`reward-audit` runs an actual StateSet reward function against reviewed candidate
responses. Use it when creating a task or changing a reward: missing context,
constant scores, and incentives that favor bad answers can waste a training run.

Run the bundled synthetic math example without loading a model or dataset:

```bash
stateset-agents reward-audit examples/data/reward_audit_gsm8k.json \
  --reward stateset_agents.data.gsm8k:GSM8KReward \
  --output reward-report.json
```

The command exits **0** when all checks pass, **1** when reward checks fail, and
**2** for invalid inputs, factory loading failures, or output errors. A failed
audit still writes its report. The output must be a new file in an existing
directory; existing reports are never overwritten.

## Supply reviewed candidate groups

A suite is a JSON object with `schema_version: 1` and a nonempty `cases` list.
Each case has a unique `id`, shared `messages`, a `context` object passed directly
to `compute_reward`, at least two `candidates`, and a `preferences` list. Each
candidate has a unique local `id` and an assistant `response`.

```json
{
  "schema_version": 1,
  "cases": [{
    "id": "addition",
    "messages": [{"role": "user", "content": "What is 2 + 2?"}],
    "context": {"gold_answer": 4},
    "candidates": [
      {"id": "correct", "response": {"role": "assistant", "content": "4"}},
      {"id": "wrong", "response": {"role": "assistant", "content": "5"}}
    ],
    "preferences": [["correct", "wrong"]]
  }]
}
```

Messages can include `tool_calls`, `tool_results`, and `metadata` using the native
`ConversationTurn` fields. Shared history may include earlier assistant/tool
turns, and must end with a user or tool turn. Each candidate is appended to that
same history. Context and turns are freshly copied for every reward invocation.

A preference `["correct", "wrong"]` requires the first candidate to score higher
than the second, beyond `--score-tolerance` (default `1e-8`). Include known reward
exploits, incorrect tool arguments, unsupported claims, and malformed responses
as rejected candidates. At least one reviewed preference is required per suite;
groups without a strict expected ordering may use `"preferences": []`.

## What the report checks

- **Within-group signal:** candidates for the same prompt must have different
  scores beyond the tolerance. Different average rewards across prompts cannot
  conceal constant scores within each group. By default every group must be
  informative; `--min-informative-fraction` allows a declared fraction in `(0, 1]`.
- **Repeatability:** every candidate is scored three times by default. Changes
  beyond the tolerance fail the deterministic-reward audit. Use `--repeats` to
  increase the checks; the minimum is two.
- **Expected rankings:** every reviewed preference must pass across all repeats.
  A high-variance reward that prefers an exploit still fails.
- **Valid execution:** exceptions, cooperative async timeouts, boolean scores,
  and non-finite scores fail the audit. Failed calls retain null scores and error
  details; they are never silently counted as zero-reward observations.

Reports include the canonical suite SHA-256, reward factory identity, audit
settings, every repeated score, errors, group-level findings, and ranking results.
The factory identity is a label, not an attestation of its code or configuration.
Retain the suite, factory source/configuration, dependency versions, and report
together when comparing revisions.

## Custom rewards and Python API

`--reward my_package.rewards:make_reward` imports and calls a zero-argument
factory. It must return an object implementing the native asynchronous
`compute_reward(turns, context)` interface and returning an object with a numeric
`score`, such as `RewardResult`. A reward class with a zero-argument constructor
also works. The command executes the selected code; judge rewards can make their
usual external calls and incur costs. No training model is loaded by the audit.

```python
from stateset_agents.data.gsm8k import GSM8KReward
from stateset_agents.evaluation.reward_audit import (
    RewardAuditPolicy,
    audit_reward,
    load_reward_suite,
)

suite = load_reward_suite("examples/data/reward_audit_gsm8k.json")
# Inside your existing async application:
report = await audit_reward(
    GSM8KReward(), suite, policy=RewardAuditPolicy(repeats=3)
)
if not report["passed"]:
    raise RuntimeError(report["failure_reasons"])
```

`--timeout-seconds` defaults to 30 per invocation. It uses cooperative async
cancellation; it cannot interrupt synchronous code that blocks the event loop.
Caller cancellation propagates rather than becoming a scored observation.

A passing audit establishes behavior only on the supplied candidates. It does
not prove on-policy reward diversity, useful gradients, absence of unseen reward
exploits, or improvement in agent quality. Keep an independent held-out evaluation
separate from reward development. Stochastic judge rewards need a separate
statistical calibration study; this audit deliberately checks repeatability.
