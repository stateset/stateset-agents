# StateSet NSR Integration — Verifiable Rewards from a Symbolic Verifier

[StateSet NSR](https://api.nsr.stateset.com) is a neuro-symbolic decision
engine: every decision it returns (`approved` / `denied` / `refused`) carries
a machine-checkable proof derived from explicit rules and facts. Unlike an
LLM judge, it is deterministic and cannot be sweet-talked — which makes it a
verifiable reward (RLVR) source, a curation gate, and an independent eval,
all from one API.

This framework integrates NSR at five seams. All of them share one
transport, configured by environment variables (see `.env.example`):

```bash
NSR_API_URL=https://api.nsr.stateset.com   # or a local nsr-server sidecar
STATESET_NSR_API_KEY=...                   # NSR_API_KEY also accepted
NSR_ORG_ID=...                             # sent as X-Org-ID
```

For high-throughput rollouts, prefer a local `nsr-server` sidecar over the
hosted API — group sampling generates many reward calls per step.

## 1. Reward: `NSRVerifierReward`

Scores the policy on agreement with the verified decision. Works with every
trainer that takes a `RewardFunction` (GSPO/GRPO/DAPO/VAPO, multi-turn), and
composes via `CompositeReward` as the hard correctness term next to soft
heuristic terms.

| Situation | Score |
|---|---|
| Model verdict matches NSR `approved`/`denied` | 1.0 |
| Model refuses/escalates when NSR `refused` (unprovable) | 1.0 |
| Disagreement, or asserting a verdict NSR refused | 0.0 |
| No response / no extractable verdict | 0.0 (NSR not called) |
| NSR unreachable | `error_score` (0.5 neutral default) |

```python
from stateset_agents.rewards import NSRVerifierReward
from stateset_agents.rewards.nsr_verifier import (
    decision_request, fact, predicate, rule,
)

reward = NSRVerifierReward()  # config from env

# The scenario supplies the decision question via context["nsr_request"].
# Build it with the request builders — they encode the exact payload shapes
# the API accepts (facts wrap their predicate; rules use if/then lists):
context = {"nsr_request": decision_request(
    "Can order A1 be refunded?",
    action="issue_refund",
    goal=predicate("permit_refund", "A1"),
    rules=[rule("refund_ok", effect="permit",
                when=[predicate("return_received", "?o"),
                      predicate("inspection_passed", "?o")],
                then=[predicate("permit_refund", "?o")])],
    facts=[fact("return_received", "A1", confidence=1.0),
           fact("inspection_passed", "A1", confidence=1.0)],
    hydrate_org_context=False,
    external_ref="episode-42",
)}
```

The model's verdict is extracted from its response — a JSON
`{"decision": "approved"}` field, or approve/deny/refuse keywords
(ambiguous mixes score 0.0 as unparseable).

## 2. Harvest gate (fail-closed curation)

Harvest/eval prompt specs accept an `nsr` key holding a `/v1/decisions`
body. A sampled completion only survives `sample_passes` if the verdict it
asserts agrees with the verified decision. Fail-closed throughout: an
unreachable or erroring verifier REJECTS the sample — a dead endpoint must
never harvest unverified training rows.

```json
{"prompt": "Refund order A1?",
 "nsr": {"query": "Can order A1 be refunded?", "action": "issue_refund"}}
```

## 3. Rollout tools: `nsr_decide` / `nsr_verify_plan`

Expose the verifier to tool-calling policies so they can consult it
mid-episode — paired with the reward, this trains policies that ask for
proof before acting.

```python
from stateset_agents.tools import create_nsr_tools

for tool in create_nsr_tools():
    agent.register_tool(tool)
```

`nsr_verify_plan` mirrors the NSR MCP server's semantics: one decision per
step, `plan_verdict: "approved"` only when every step approves, evaluation
stops at the first blocking step, and an empty plan raises — never a
vacuous approval.

## 4. Outcome loop: `NSROutcomeReporter`

Every decision made during rollouts has a known outcome by episode end.
Posting it back feeds NSR's calibration curve and conjecture mining — the
training flywheel and NSR's learning flywheel become one cycle.

```python
from stateset_agents.rewards import NSROutcomeReporter

reporter = NSROutcomeReporter()
await reporter.record_from_reward(reward_result, outcome="honored")
# or by your correlation key:
await reporter.record(external_ref="episode-42", outcome="reversed")
```

Outcomes: `honored | reversed | overridden | escalated`. Reporting is
best-effort (`False` on transport failure) — it never crashes a training run.

## 5. Independent eval gate

Eval specs with `nsr` gain a per-row `nsr_verified` extra in
`eval_results.json`, and `eval_gate_failures` fails unverified rows.
Unlike the judge (which degrades when unavailable), this gate is
fail-closed — an eval that silently skips its strongest check would
overstate the model.

**Anti-Goodhart doctrine** (see `docs/PROOFS.md`): objective and eval must
not share a code path *or a knowledge base*. Point the reward at one NSR
org/KB and the eval specs at a held-out one; rule gaps the policy learns to
exploit surface as eval failures, and NSR's conjecture mining proposes the
missing rules.

## Testing without a server

Every network path is injectable: `NSRVerifierReward(client=...)`,
`create_nsr_tools(client=...)`, `NSROutcomeReporter(poster=...)` accept
async callables. See `tests/unit/test_nsr_*.py` for fakes, and
`examples/nsr_verified_reward.py` for a runnable demo.
