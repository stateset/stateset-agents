# River action-format fix — 2026-10-18

The corrected refund-policy prompt completed a fresh Qwen3.5-9B River run in 398.0 seconds. Two real SDK-reported optimizer updates had finite, nonzero gradient norms. The baseline and both trained checkpoints tied at 7/8 validation successes, so validation selected the untrained baseline. That selected baseline passed 6/8 held-out cases. This is not evidence of an RL learning gain.

The old prompt named tool/args keys but illustrated actions as `lookup_order: {...}`. The replacement provides an executable complete JSON lookup object, JSON-escapes the full order ID, requires exact ID preservation, and supplies an explicit finish object. Policy facts stay private until lookup. Reward rules, policy priority, violation penalties, turn limits, and checkpoint-selection rules are unchanged.

| Measurement | Result |
| --- | --- |
| Fresh seed | 20261019 |
| Cases | 32 train, 8 validation, 8 held-out; synthetic |
| Baseline validation | 7/8 |
| Step-1 validation | 7/8 |
| Step-2 validation | 7/8 |
| Training sampling successes | 28/32 in batch 1; 24/32 in batch 2 |
| Validation-selected checkpoint | Step 0, the untrained baseline |
| Selected checkpoint held-out success | 6/8 (75%; Wilson 95% interval 40.9%–92.9%) |
| Held-out JSON syntax and exact order IDs | 25/25 executed actions valid |
| Held-out policy violation rate | 25% |
| Held-out environment timeouts | 0% |
| Rollout admission | 96 trajectories, 98,304 tokens reserved of 110,000 |
| Local checks | 143 focused tests; 19 real-SDK checks with scripted transport; 439 typed source files; lint, format and hygiene passed |
| Final billed cost | Unknown; token reservations do not enforce a dollar cap |

[Selected baseline pointer](river_checkpoint.json) and [final trained candidate pointer](final-trained-candidate/river_checkpoint.json) are intentionally separate. Both reference River-hosted weights. The final trained candidate was not selected and was not evaluated on the sealed test split. [Review summary](review-summary.json), [held-out report](run/test_results.json), [replay audit](replay-audit.json), and [action-format audit](action-format-audit.json) contain the evidence.

The remaining held-out failures were undelivered-order policy priority and the inclusive return-window boundary. Each used valid JSON with the correct order ID. Correcting an invalid refund later still incurs a violation under the unchanged scoring rules.

No case IDs or order-fact fingerprints overlap with the [previous canary](../river-live-review/README.md). Different seeds and cases prevent treating its 3/8 result and this 6/8 result as a controlled prompt-effect estimate. Within this run, baseline and trained checkpoints used identical validation cases. Eight test cases and one seed are insufficient for production quality or A+ readiness.

The exact prior 439-source implementation was preserved before editing at `outputs/river-integration-review-20261018-live/source-before-prompt-fix/`. This bundle retains the changed environment source and test source; the run manifest fingerprints all 439 package files. Validation selection and all held-out outcomes passed fresh local replay with the source-matched implementation. Historical evidence is preserved unchanged.

The next useful experiment is a training-only expert SFT warmup covering policy priority and inclusive boundaries, followed by a larger preregistered base/SFT/RL comparison on fresh cases. This held-out test remains sealed. No credential is stored in this bundle. SHA-256 hashes are in artifacts.sha256.json.
