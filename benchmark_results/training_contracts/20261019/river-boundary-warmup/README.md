# River boundary SFT warm-start and RL pilot

Separate user-authorized synthetic pilot on Qwen/Qwen3.5-9B, rank 16. Original checkpoint fixed to the first prior study seed. Warm-start SFT uses 255 training-only verified curriculum demonstrations for one epoch (16 batches), then four planned native CISPO RL batches. The 64 fresh validation and 128 fresh test cases are disjoint by identity and policy facts from retained historical cases. Native RL stopped at three consecutive skipped batches, with zero optimizer updates. Its test remains unattempted. The original failed execution and protocol are preserved. An explicit evaluation-completion plan was frozen before test sampling to run the two already planned fixed SFT arms once. This changes stage ordering after the RL stop; it adds no training, output-token allocation, or deadline extension.

| Checkpoint | Successes | Boundary successes |
|---|---:|---:|
| original | 106/128 | 5/16 |
| warmup | 128/128 | 16/16 |
| rl | Not tested | Not tested |

Single-seed observations do not establish reliable gains or A+ certification. Original versus warmup also changes training exposure; it cannot isolate curriculum effects. Data is synthetic and has no independent human review. Check the family, policy-violation and truncation results in `review-summary.json`.

Execution: warmup_evaluated_rl_stopped_without_updates; supervisor exit 1; timeout False. Generated-output admission reservations: 506,880 against 884,736. These are not actual usage or billed dollars. SFT has a separate one-epoch, 16-batch cap with input/assistant token counts in `data-check.json`. No operational retries.

Validation and completed test traces were replayed offline; checkpoint lineage, split hashes, run binding and admission ledgers verified. 370 unit tests and 19 scripted SDK tests passed; 445 sources passed full type checking. 481 older evidence artifacts verified unchanged.

`frozen-run.tar.gz` retains package sources, input data, exact actions, token ledgers, supervisor logs and checkpoint pointers. `source-and-tests.tar.gz` retains the explicit River SFT checkpoint option, regression tests and documentation. The warmup checkpoint pointer is reusable through StateSet; no deployment or test-driven promotion was created. No River credential is retained. Hashes are local provenance, not provider attestation.
