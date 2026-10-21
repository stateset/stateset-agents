# RunPod reward regression and recovery

The Qwen3.5-9B/Gemma 4 experiment is a useful infrastructure result and a
negative quality result. Completing optimizer steps is not a policy-promotion
criterion. A higher training reward cannot override a worse independent judge.

## Evidence located on 2026-10-16

[PR #109](https://github.com/stateset/stateset-agents/pull/109), inspected at
`5162d211934e369f1988270b979fee0936080936`, documents the following v2
preflight-5 measurements:

| Measurement | Base | Trained |
| --- | --- | --- |
| Gemma 4 mean score | 0.916 | 0.803 |
| Rule reward v2 | 0.636 | 0.677 |
| Retrospective rule reward v3 | 0.625 | 0.281 |

The PR reports 250 completed steps, $22.81 over 20,583 seconds, judge
self-disagreement 0.046 against a 0.05 gate, and direction agreement on
79/200 pairs for v2 versus 161/200 for v3. The user additionally reports
11 repeated-turn findings for the trained model versus zero for the base.
These full-run totals are **reported, not independently recomputed here**:
the full 400 transcripts, raw judge passes, execution manifest and cost record
were not located. `/home/dom/sa-runs/flagship_preflight5/report.json` is an
older failed experiment; its matching directory name does not identify this
v2 run. Do not combine their evidence.

The cost arithmetic is consistent: 20,583 seconds at the historical $3.99/hour
is $22.812825. That checks the calculation, not the provider's bill or the
total cost of all five attempts. Also, the PR's repetition-check pass rate
of 0.945 would imply 11 of 200 conversations failed that check, if all cases
were covered. It does not by itself establish exactly 11 repetition
occurrences; that requires the per-turn findings.

Three paired transcript examples were recovered from PR #109's
`tests/fixtures/support_conversation_reward/preflight5_regressions.json`.
The reward, judge, rubric and regression tests were imported from that pinned
revision. The imported judge test that depended on absent flagship manifests
now checks the installed rubric pin directly. The full flagship workload and
launch manifests were not transplanted over this branch's trainer changes.

The code supports the proposed mechanism: v2 pays for politeness keywords,
order references and per-turn acknowledgements. It checks whole-turn similarity
but misses repeated sentences with changing surrounding text. V3 neutralizes
the keyword bonuses, credits earlier acknowledgement, and adds penalties for
long replies, formulaic openers, boilerplate leads, repeated requests,
ungrounded specifics and sentence reuse. These are heuristics; paraphrases,
negation, valid derived quantities and multilingual replies remain limitations.

The 161/200 number is retrospective agreement on reward-development data,
not evidence that training with v3 improves a new policy. Some automated
detectors are shared with the judge, further limiting independence. Stable
order-reversed judging is not human calibration. Halving learning rate and
doubling KL weight constrain the update but do not establish actual KL or
repair a misaligned objective by themselves.

## Implemented safeguards

- `SupportConversationRewardV3` is available without changing legacy v2
  behavior or relabeling historical results.
- `SupportJudgeReward` scores the full conversation. It can blend a fixed
  fraction of training transcripts with v3 using content-hash sampling, a
  configured weight and an explicit attempted-episode budget. Judge failures,
  incomplete parses and unstable scores raise. There is no 0.5 fallback.
  Hard safety and `must_avoid` vetoes remain zero. The caller must separately
  constrain provider spend; an episode limit is not a dollar limit.
- `JudgeCheckpointSelector` evaluates the baseline, then committed updates
  periodically and at the final partial interval. Only a strict validation
  improvement selects a new checkpoint. Ties and worse candidates retain the
  earlier checkpoint, including the baseline. It verifies complete two-pass
  judge output, full validation coverage, the mean disagreement limit,
  globally distinct case IDs and exact-content separation of all three splits.
  It saves a receipt with per-case scores, input hashes, judge identity and
  selected artifact hash. Save failures do not replace the prior selection.
  This new callback supports fresh single-process studies; distributed and
  resumed execution fail before judging. Its callback state is not resumable.
- `support_judge_evaluator` generates fresh episodes with current policy
  weights and cloned environments. Validation is greedy, with temporary copies
  of both the agent and cached Hugging Face decoding configurations. Case-seeded
  inference restores training settings, local RNGs and model modes even after
  failure. The agent must be used exclusively during validation. A separate
  judge endpoint avoids co-resident
  learner/judge GPU memory pressure. The caller must pin and authenticate that
  endpoint; an identity string alone cannot verify its remote weights.
- `python -m stateset_agents.evaluation.reward_audit` recomputes v2/v3
  scores, direction agreement, explicit tie counts and tied-rank Spearman
  correlation from paired transcripts. An explicit expected pair count stops
  a three-example fixture from masquerading as a 200-case audit. Reports are
  labeled diagnostic only and cannot approve production.

## Wiring into a new training study

```python
from stateset_agents.rewards import SupportConversationRewardV3
from stateset_agents.training.judge_selection import (
    JudgeCheckpointSelector,
    support_judge_evaluator,
)
from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

# judge is SupportJudge with a pinned, bounded generation backend.
# Each partition contains distinct scenario dictionaries with stable IDs.
# These are the proposed 9B recovery settings, not new library defaults.
config.learning_rate = 1e-5
config.beta = 0.1
config.use_reference_model = True
selector = JudgeCheckpointSelector(
    evaluate=support_judge_evaluator(judge, validation_environment),
    train_cases=train_cases,
    validation_cases=validation_cases,
    test_cases=test_cases,
    judge_model="google/gemma-4-31B-it",
    judge_revision="842da3794eaa0b77d5f08bae87a17459d91ff475",
    rubric="25fa6b2a9551e8901a566245965c3b2beb7f044b",
    every_updates=25,
    max_evaluations=12,  # baseline + 10 checkpoints; reserve one final check
)
trainer = MultiTurnGRPOTrainer(
    agent, environment,
    reward_fn=SupportConversationRewardV3(),
    config=config,
    callbacks=[selector],
)
await trainer.initialize()
await trainer.train()
selected = selector.selected_checkpoint()
# Export/reload this artifact for one sealed test evaluation. A selected
# baseline is a valid negative result, not a trained-quality success.
```

Do not use the previously inspected 200 cases as an untouched test set. They
are now reward-development data. Supply disjoint validation cases for periodic
selection and fresh sealed test cases for the final comparison. Exact-content
checks cannot detect semantic duplicates or certify reviewer provenance.

## Recorded recovery study and free planning

The packaged profile is
[`support_recovery_preflight6.json`](../stateset_agents/config/presets/experiments/support_recovery_preflight6.json).
It defines **`stateset-support-recovery-v1`**, a new native protocol; backend and
objective equivalence with PR #109's flagship workload have not been established.
It pins the learner/judge commits, v3, learning rate `1e-5`, reference KL `0.1`,
zero LoRA dropout, four scenarios and eight generations per scenario for each of
250 committed updates. The resolved native objective is included in the plan.
Training samples; validation uses greedy decoding. The baseline and every 25th
update are judged, and an earlier checkpoint survives ties or worse scores.

Generate a plan without allocating a GPU or calling a judge:

```bash
python -m stateset_agents.training.support_recovery --output /tmp/preflight6-plan.json
# Optional: --cases /protected/reviewed-cases.json
```

The output refuses overwrites. It binds the profile, package and supplied case
content; missing cases remain explicit blockers. Cases need a `provenance`
object with `source: "reviewed"`, a nonempty `reviewer`, and
`test_is_fresh: true`, plus `train`, `validation` and `test` lists. Each scenario
contains `id`, `user_query`, nonempty `policy_facts`, `expected_resolution` and
`must_avoid`. The default minimum counts are 500/32/200. IDs and exact scenario
content must be distinct across all splits. The planner and checkpoint selector
also reject repeated support inputs independently of IDs, labels and metadata:
queries and policy facts use Unicode normalization, case folding and collapsed
whitespace, with policy facts compared as an unordered set. Optional `source_id`
values keep related examples in one split. Supply the original conversation or
ticket identity in that field when multiple cases share a source. These checks
do not detect semantic paraphrases or attest to reviewer provenance; both still
require human review.

For a separately reviewed execution environment, construct a
`RecordedSupportJudge(generate, output_dir / "judge-evidence", ...)` with the
profile's judge identity, retry limit and batch budget, then call
`create_recovery_trainer(agent, environment, profile=profile, cases=cases,
judge=judge, output_dir=output_dir)`. This factory returns an uninitialized native
trainer and exposes `trainer.judge_selector`. The caller owns the model,
conversation environment, bounded judge transport and provider lifecycle.
It preserves all four scenario groups, rejects missing generations, checks
actual reference separation/freeze, dropout and optimizer settings before
updates, and requires the planned number of committed updates. Requested
reference initialization errors now propagate instead of silently disabling KL.

Each private judge batch retains full scenarios/transcripts, both passes' prompts,
raw replies, parse retries, checks, forced findings and scores. Failed transport
calls retain requests and the error type, omitting possibly sensitive error
messages; they consume the attempted batch budget. A receipt binds every file
hash and the judge/parser/rubric implementation. The recovery selector requires
these records and rechecks both selected weights and judge evidence before use.
`replay_judge_evidence(path, expected_manifest_sha256=trusted_hash)` reproduces
parsing/scoring offline without inference. Hashes establish integrity relative
to a trusted receipt, not authentication of a remote model's weights.

The planner reports 8,000 training episodes and 11 validation evaluations. With
32 validation cases and one parse retry, its conservative validation bound is
1,408 judge prompts; sealed-test judging is additional. Its approximately
467 GiB checkpoint estimate assumes full policy/reference BF16 copies for every
candidate plus save headroom. This is an unmeasured upper estimate, not a backend
memory/storage certification. The proposed 750 GB disk and $32 ceiling are
unapproved placeholders. Pricing must include setup, training, repeated judging,
storage, collection and termination. `ready_for_paid_execution` remains false;
neither supplying cases nor running the planner authorizes spending.

## Paid-run gate and remaining evidence

No new paid run was launched by this investigation. PR #109's preflight
manifest still declares framework 0.65.0 while this checkout is 0.66.1, and
uses a $32 ceiling with an estimated $31.26 lifetime at its historical price.
It is not a reviewed launch contract for this checkout. Periodic judging adds
work; the old $31 estimate must not be reused without repricing it.

Before preflight-6: retain the original artifacts, run the complete 200-pair
audit, freeze fresh partitions and reward/judge/source identities, verify
tests and the installed wheel, and produce a current RunPod plan including
judging, setup, checkpoint storage, collection and termination time. Pin
learning rate `1e-5` and KL coefficient `0.1` as proposed, and log measured KL,
entropy, length, repetition, penalty rates and independent validation scores.
Only the selected checkpoint is tested once against the matched baseline.

A single positive preflight is a screening result. A separately authorized
three-seed matrix must retain all seeds, failures, costs and null results;
it does not become production approval automatically. PR #125 is still open,
but all its listed GitHub checks, including Windows, were successful when
inspected. That does not establish its fix is in this checkout or launch image.

The broader competitive priorities remain: a representative 8B-class
multi-turn quality study, equal-compute framework comparisons, a certified GPU
configuration with recovery evidence, and independent reproduction. Licensing
and supported-surface decisions remain separate maintainer decisions; this
change does not alter the license or certify additional models/backends.

## Local acceptance

[Buildr job `6570326bc3bc`](../benchmark_results/training_contracts/20261016/reward-recovery/README.md)
passed 2,173 targeted source regressions (seven skips), 906 targeted isolated
installed-wheel regressions, typing for all 416 package modules, and twelve
real two-worker CPU/Gloo contracts against both source and wheel. Lint,
formatting, repository hygiene, release governance, public API compatibility
and workflow lint passed. Package/test Python remained fixed throughout;
the wheel matches the accepted package Python and pinned rubric bytes.
Source and wheel regression counts overlap.

The retained offline audit covers exactly three recovered pairs: v2 agrees
with the recorded judge on one pair, v3 on all three. It also reproduces
aggregate reward-up/judge-down divergence for v2 on these examples. These
selected development examples do not reproduce the full 161/200 result or
establish a training-quality gain. The evidence includes source hashes,
PR import provenance, the three-pair input hash, per-item scores and the
failed verification attempts with their fixes. No paid run was launched.

The subsequent [studio and recorded recovery acceptance](../benchmark_results/training_contracts/20261016/finetune-studio/README.md)
verifies the final recovery factory, retained judge evidence, greedy decoding
restoration and terminal studio. Buildr job `bb7581c10563` passed 2,442 targeted
source regressions (seven skips), 1,080 installed-wheel regressions and typing
for all 421 package modules. Both source and wheel exercised a real terminal
and the two-update tiny CPU recovery contract. These overlapping checks do not
repeat the earlier distributed matrix or establish full-size GPU training,
new policy quality or provider cost evidence. The installed-wheel planner keeps
paid execution blocked. No paid run was launched.
