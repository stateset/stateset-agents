# River recovery SFT warmup — October 20, 2026

Fixed one-epoch SFT continuation from the retained Qwen/Qwen3.5-9B
arithmetic candidate on 160 unchanged, audited recovery training transcripts.
Ten optimizer updates were acknowledged. Starting weights were loaded
with a fresh optimizer; this is not optimizer-state resume.

| Split | Parent | Warmup | Warmup violation rate | Warmup truncation rate |
| --- | ---: | ---: | ---: | ---: |
| validation | 79/80 | 80/80 | 0.0% | 0.0% |
| test | 147/160 | 160/160 | 0.0% | 0.0% |

Declared pilot acceptance: **True**. Test attempted: **True**.
No automatic production promotion occurred.

| Paired split | Both passed | Warmup only | Parent only | Neither passed |
| --- | ---: | ---: | ---: | ---: |
| test | 147 | 13 | 0 | 0 |
| validation | 79 | 1 | 0 | 0 |

Paired counts describe this fixed checkpoint comparison; passing the
pilot gate does not establish a statistically reliable learning gain.

## Formatting and completion semantics

Each outcome passed exact environment replay before diagnosis. Error
classes distinguish malformed JSON, invalid tool contracts, incorrect
completion outcomes, unsafe retries, reconciliation misuse, and premature
finish. A denial/escalation confirmed applied is a resolved workflow,
not a caller-authorization failure. The original scorer remains unchanged.

| Split / category | Parent cases | Warmup cases |
| --- | ---: | ---: |
| test / format | 0 | 0 |
| test / completion_outcome | 13 | 0 |
| test / tool_contract | 0 | 0 |
| test / unsafe_retry | 0 | 0 |
| validation / format | 0 | 0 |
| validation / completion_outcome | 1 | 0 |
| validation / tool_contract | 0 | 0 |
| validation / unsafe_retry | 0 | 0 |

## Residual training-probe signal

The post-warmup train-only probe succeeded on **67/76**
rollouts. **6/19 groups** had nonzero within-group
reward variance, with four rollouts per group at temperature 1.
Its cases were chosen before provider calls: first case ID per recovery
family and public terminal receipt type. Coverage includes applied
refunds, denials, escalations, unauthorized outcomes, and handoffs where
present in the training set. This expanded probe is not directly paired
with the earlier pilot's smaller ten-case probe.

Probe case error counts: `{"completion_outcome": 0, "format": 9, "premature_finish": 0, "reconciliation": 0, "tool_contract": 0, "unsafe_retry": 0}`.
The probe performed **zero RL optimizer updates**. Later RL requires
fresh on-policy samples with native token/logprob evidence.

## Protocol and provenance

The only training input was the prior canonical train split. No held-out
cases or validation-derived demonstrations were added. All 160 teachers
were replayed, and their exact file bytes matched the audited source.
The final checkpoint was fixed at ten updates, one epoch, batch size 16,
rank 16, learning rate 2e-5, and training seed 2026109401.
Final provider-reported loss: `4.253957582556683e-07`; loss alone is not quality evidence.

The frozen fresh evaluation contains 80 validation and 160 test cases per
arm. Order IDs and fact fingerprints were excluded against all historical
training, prepared, and evaluated inputs before sampling. Both arms use
identical agent prompts/tools and greedy sampling with six model turns,
1,024 output tokens, 256 tokens per turn, and an 8,192-token context limit.

Validation requires every warmup case to succeed without violations or
truncation. Test additionally requires a strict success gain over parent
and no family or error-category regression. Both arms' attempts were
sealed before sampling each split. No test-based checkpoint selection
or retry of sealed stages was allowed.

Limits: ten SFT updates, at most 556 sampled trajectories and 569,344
reserved output tokens, no provider retries, 25 minutes of worker time.
Actual billed cost was unavailable; the token reservation is not a dollar cap.

Independent review replayed **556 model outcomes** and **160 teachers**.
All **670 earlier retained artifacts** matched their hashes.
The previous recovery pilot's 80 test cases and arithmetic pilot's 308
test cases were excluded and remain unattempted.

Local verification: **192 relevant contract tests passed**, including
12 study gate/diagnostic tests. The frozen 455-source package is unchanged
from the preceding validated recovery implementation. The archive includes
runtime sources, controller, exact inputs, attempt seals, training receipts,
full traces, diagnostics, and independently executable review scripts.

Archive byte verification checked **524 files**. Local hashes provide
consistency checks, not independent provider attestation.

This single-parent synthetic pilot does not establish production A+
quality, independent merchant generalization, durable payment idempotency,
optimizer continuity across recovery, or an RL learning improvement.
