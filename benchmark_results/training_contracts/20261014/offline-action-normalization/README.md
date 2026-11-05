# Offline action-weighting regression

The previous implementation averaged microbatch means. With two actions in one
microbatch and one in the next, the single final action received half the
optimizer update's weight rather than one third. The measured reproduction
failed all three uneven-partition cases and passed the equal-size control.
`reproduction.json` binds this observation to the preceding package source;
`before.json` retains the actual failing test output.

The fix accumulates action sums at a fixed scale and normalizes by the total
number of actions before gradient clipping and the optimizer update. Assistant
token loss remains averaged within each action, with detached IQL weights.
Checkpoint state preserves pending action counts and scaling. Legacy checkpoints
at completed optimizer updates remain loadable; legacy pending gradients without
counts are rejected before policy weights change. Partial-window metrics now
record the final optimizer update and gradient norm.

Shared queue job `41d9180f92d3` passed **8,263 tests**, with **57 optional skips**,
and type checking for **383 package modules**. Repository checks and fresh-wheel
verification passed. The wheel performed real SFT and offline IQL updates,
export/reload/inference, and exact checkpoint recovery in the CPU device contract.
Package sources stayed unchanged throughout verification and matched the wheel;
`verification.json`, `wheel_proof.json`, `cpu_contract.json` and `after.json` retain
the results. `summary.json` binds them to the exact wheel and regression tests.
The CPU policy artifact was rehashed against its reported digest.

CUDA was not rerun for this change. No paid resources were allocated. These are
synthetic engineering checks, not evidence of representative policy quality or
an RL gain. The prior native CUDA result identifies a different wheel and remains
separate in `../native-offline-device/`.
