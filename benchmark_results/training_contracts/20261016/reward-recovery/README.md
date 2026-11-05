# Support reward recovery and optimizer reporting acceptance

Buildr job `6570326bc3bc` verified frozen source and an isolated installed wheel.
The wheel includes the exact rubric and package Python files from the checkout.

- 2,173 targeted source regressions passed (7 skips).
- 906 targeted installed-wheel regressions passed.
- All 416 package modules passed typing.
- Twelve real two-worker CPU/Gloo contracts passed against source and wheel.
- Sixteen optimizer-reporting scenarios agree across peers and source/wheel.

The new support tests cover the pinned v2/v3 reward examples, whole-conversation
judge output, explicit attempted-episode budgets, hard vetoes, judge parse and
stability failures, independent validation selection, keeping the baseline on
regression or ties, retaining an earlier better checkpoint, byte tampering,
split overlap, RNG/mode isolation, current-policy generation and save failure.
The selector rejects distributed/resumed studies; it does not certify them.

The offline audit reproduces v3's judge-aligned preference on three recovered
paired examples. This is reward-development evidence, not a complete 200-pair
study or proof of a training quality gain. Full preflight-5 judge/cost totals
remain unverified. No paid run was launched. GPU, multi-node and representative
quality gains are not certified. See docs/RUNPOD_REWARD_RECOVERY.md.

Optimizer events cover batch, task-boundary and final updates, normal cadence,
partial-window AMP skips, early stopping and coherent callback/logger failure
recovery. These use tiny CPU/Gloo learners and synthetic rewards. Twelve tests
contain multiple scenarios; source/wheel regression counts overlap.
