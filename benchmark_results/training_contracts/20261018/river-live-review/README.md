# Live River integration review — 2026-10-18

The Qwen3.5-9B refund-policy-v2 canary completed successfully in 471.6 seconds. It made two SDK-reported optimizer updates with finite, nonzero gradient norms. Validation selected the step-1 checkpoint; all validation and test traces passed independent local replay. The run used River SDK 0.11.0 and the exact 439 packaged source files recorded in the run manifest.

| Measurement | Result |
| --- | --- |
| Training data | 32 synthetic cases; 64 sampled training trajectories |
| Baseline validation | 0/8 successful |
| Step-1 validation | 2/8 successful |
| Step-2 validation | 2/8 successful |
| Selected checkpoint | Step 1, chosen using validation only |
| Held-out test | 3/8 successful (37.5%; Wilson 95% interval 13.7–69.4%) |
| Test policy violation rate | 62.5% |
| Test environment timeout rate | 50% |
| Rollout admission | 96 trajectories; 98,304 output tokens reserved of 110,000 |
| Local real-SDK tests with scripted transport | 19 passed in 6.710 seconds |
| Final billed cost | Unknown; rollout reservations are not a dollar cap |

[Checkpoint pointer](river_checkpoint.json), [review summary](review-summary.json), [held-out report](run/test_results.json), [replay audit](replay-audit.json), and [local SDK checks](native-sdk-tests.txt) contain the detailed evidence. The pointer references River-hosted weights and can be supplied as a checkpoint directory to compatible River commands. It contains no local adapter weights.

Authentication, account-scoped model access, tokenizer setup, native asynchronous rollouts, backward/optimizer calls, checkpoint saves, validation selection, and one sealed test evaluation all executed. The client and session context managers exited normally. This run did not test live crash recovery, independent resource cleanup inventory, hosted chat, streaming, or provider billing.

The first sandboxed account check failed before resource creation; the network-enabled retry succeeded. Both reports are retained. An earlier local SDK test attempt stalled and was manually interrupted; the bounded rerun passed all 19 tests. The cause of the first stall is unconfirmed.

The integration result is positive. Model quality remains limited: four test cases timed out with malformed action syntax or altered order IDs. A fifth corrected its action format and completed the policy decision, but its initial malformed action incurred a violation. Prefer explicit complete JSON-action examples or an expert SFT warmup before scaling RL. Use fresh held-out cases for the next study; preserve this test attempt.

This single-seed, two-update synthetic experiment does not establish production quality or A+ readiness. The validation change compares the same validation cases; the test uses different cases and has no matched base-model test score. No API credential is stored in this bundle. SHA-256 hashes are recorded in artifacts.sha256.json.
