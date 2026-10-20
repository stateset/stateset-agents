# Nemotron 3.5 Super VL and Bolt integration evidence

Hosted RL and package/wheel verification completed. These artifacts do not
certify full-size local BF16 GPU training or model-quality improvements.

- `tiny-super-vl-architecture.json`: a real optimizer update through NVIDIA's
  pinned Super VL vision/Mamba/MoE implementation, reduced to 651,592 randomly
  initialized parameters on CPU. No pretrained weights were loaded.
- `reproduce-tiny-super-vl.txt`: the executable Python entrypoint used for that
  contract; run it with Python in the optional Nemotron environment. Random
  initialization means the loss can vary between runs.
- `river-model-access.json`: authenticated account access, without credentials.
  Super VL NVFP4 is available; Bolt is absent from the account capability list.
- `river-attempt-1-failure.json`: the first hosted attempt stopped before sampling
  or optimizer updates because River SDK 0.11 did not recognize Nemotron's image
  placeholder in its structured-input lowering. The compatibility fix preserves
  vocabulary and token IDs and has passed 17 direct SDK/processor regressions.
- `verification-attempt-1.json`: package-wide types and eight pinned processor
  contracts passed; the repository gate rejected one Ruff finding, now fixed.
- `river-sdk-regression.json`: 17 actual SDK/pinned processor regression cases
  passed after the image-placeholder lookup fix, through Buildr v0.15.0.
- `verification-attempt-2.json`: eight pinned processor contracts, types across
  390 source files, repository gates, and 1,270 Nemotron/CLI regressions passed.
  Six legacy recording-fake tests failed because the installed SDK enabled real
  rendering their raw-text fakes did not implement. The corrected fixture
  explicitly isolates these fakes; real SDK contracts remain separate.
- `river-super-vl-smoke/`: successful hosted Super VL NVFP4 run, including
  synthetic datasets, source hashes, training activity and metrics, checkpoint
  references, validation selection, and replay-audited test traces. Start with
  `summary.json` for scope, metrics, and queue provenance.
- `verification.json`: final successful verification, with package hashes and
  wheel SHA-256 `05ee9864a1b1f1428b965bce3770b40728d31bbfdf01187c251d11f4e6771677`.
  It includes 1,270 Nemotron/CLI regressions, 598 River regressions, 19 native
  SDK integration cases, eight pinned NVIDIA processor contracts, and 53
  isolated-wheel cases. Two optional regression cases were skipped. All 390
  packaged source files passed typing, and repository gates passed. The final
  continuation reused earlier passed stages only after verifying unchanged
  package hashes; it reran River regressions after isolating recording fakes.
  This was targeted regression coverage, not a new full-suite run.

The hosted experiment used synthetic refund-policy tasks, seed 42, four groups
of eight training trajectories, and one optimizer batch. The SDK reported one
update, gradient norm 0.06594276428222656, and a step-1 training checkpoint.
Both baseline and step-1 validation mean reward were 0.5; deterministic selection
kept the baseline. Its test result was 3/4 successes, with a wide 95% Wilson
interval of approximately [0.301, 0.954]. This is not evidence of improvement.

The 98,304-token durable generated-token reservation budget admitted 44
trajectories and reserved 45,056 tokens across training, validation, and test.
This is not a spending cap or a representative quality study. Bolt was absent
from the authenticated account capability list; no hosted Bolt run is claimed.
