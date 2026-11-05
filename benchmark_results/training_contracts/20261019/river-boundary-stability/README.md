# Refund policy stability data and validation diagnosis

Offline system improvement following the six-parent SFT replication. No River
job, optimizer update, model evaluation of new holdouts, or deployment ran here.

The exact previous frozen package replayed all 576 prior validation trajectories:
original 171/192, original-repetition 189/192, boundary 189/192. The boundary
failures comprise one date comparison, one refund arithmetic error and one
corrupted order ID. Categories describe the first invalid action, not a proven
cause of the model behavior. No previous test outcomes supply new supervision.

The opt-in `policy_parameter_contrasts_v1` curriculum preserves every original
training case and adds three adjacent-day contrasts around one of 3, 10, 21, 60,
90 days, assigned within source family. It retains monetary, delivery, chargeback
and untrusted-note facts. All eight source families participate. A teacher reads
lookup facts, performs exactly one resolution and finishes with empty arguments;
every transcript is independently replayed. All six legacy curriculum bundles
still reconstruct exactly under the default policy.

Prepared 3,072 teacher transcripts (512 per parent, including 768 original rows)
and 1,824 fresh held-out cases: 624 validation and 1,200 test. Each parent has 32
ordinary plus 72 stress validation cases, and 128 ordinary plus 72 stress test
cases. Stress windows are 5/18/40 for validation and 8/25/75 for test, distinct
from the new training windows. Original training windows 14/30 remain present.
These are changed, explicitly prospective synthetic evaluation distributions.
Every grid cell uses its own source order; private stratum metadata stays outside
model observations. ID and observable-fact overlap is zero against historical
inputs, all new training facts, earlier unused preparations and other fresh cases.

Two offline preparations stopped on identity or fact reuse before any model
evaluation. Their complete training data and partial heldouts remain retained.
The completed preparation reaudits training data and logs prospective input-seed
rejections with a finite 100-candidate limit per parent. These are input checks,
not retries of a model, holdout outcomes, paid jobs or a failed acceptance gate.

Validation: 182 unit/CLI tests passed, full mypy passed for 446 packaged sources,
and Ruff, Black, isort and repository hygiene checks passed. Independent audit
reconstructed/replayed all 3,072 teachers and reconstructed all 1,824 holdouts.
593 existing retained evidence artifacts verified unchanged. The archive retains
current and prior package sources, source inputs, all preparation attempts,
fresh datasets, provenance, audits and test results. Archive members were checked
byte-for-byte. Hashes establish local identity, not provider attestation.

Model quality, actual provider compute and billed cost are unmeasured. The next
live experiment must freeze a bounded matched-control pilot, tokenize and match
exposure with the actual model tokenizer, declare family and stress regression
gates, and seal a single test attempt after validation selection. This preparation
does not establish A+ performance or independently reviewed production readiness.
