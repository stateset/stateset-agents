# Engine rollouts stay on-policy (2026-09-08)

## Problem

`MultiTurnAgent.set_rollout_backend(engine)` (v0.51.0) lets the GRPO trainers
sample from an inference engine, but nothing refreshed the engine's weights
after an optimizer step. Every rollout after the first step came from a stale
policy while the token path treated it as exactly on-policy (`logp_old =
logp_cur.detach()`, ratio 1), so the clipped objective's trust region never
saw the drift. "Refreshing the engine's weights is the caller's
responsibility" was an honest note, not a design.

## Design

Two independent mechanisms, both default-safe:

1. **Weight sync after every optimizer step.** `_apply_optimizer_step` in both
   GRPO trainers ends with `agent.sync_rollout_backend()` when
   `TrainingConfig.rollout_sync` (default `True`). The agent forwards to the
   backend's `sync_weights(model)`, counts successful syncs as
   `rollout_backend_version`, and stamps every turn with that version and
   `rollout_backend_stale`. A backend without `sync_weights`, or one that
   raises, marks the agent stale (warned once, reason kept in
   `rollout_backend_error`) instead of silently sampling from an old policy;
   trainers expose the last outcome as `last_rollout_sync`.
   `VLLMGenerator.sync_weights` merges a PEFT adapter for the read, strips
   `base_model.model.` / `module.` / `_orig_mod.` prefixes, and streams
   `(name, tensor)` pairs into the in-process engine's `load_weights`
   (resolved through the known V0 executor attribute paths; `weight_loader`
   adapts any other layout, and is what tests inject).

2. **Importance correction for a stale engine.**
   `TrainingConfig.old_logprobs_source = "sampler"` makes the token path use
   the `sampler_log_probs` recorded on each turn as `logp_old`, laid out by
   `loss_computation.sampler_old_logprobs` exactly like the forward pass
   (`[rows, width-1]`, response token `k` at column `len(prompt)-1+k`). The
   ratio becomes `exp(logp_cur - logp_sampler)`, the TRL vLLM correction. An
   explicit inner-update snapshot (`num_gradient_updates > 1`) still wins;
   rollouts without sampler log-probs fall back to `"recompute"`. The loss
   dict reports `old_logprobs_source` so runs can prove which policy the
   ratio was taken against.

## Verification

`tests/unit/test_rollout_sync.py`: agent contract (version, staleness,
warn-once, error retention, reattach reset), trainer wiring on both trainers
(sync called once per optimizer step, disable switch, agents without the
hook), `VLLMGenerator.sync_weights` (named parameters streamed, PEFT merged
during the read and unmerged after, prefix stripping, loud failure without an
engine or a recognisable layout, resolution through the known engine path).
`tests/unit/test_grpo_token_path.py`: sampler tensor aligns with the forward
layout, missing/mismatched sampler log-probs handled, `"sampler"` equals
on-policy when the engine is current, ratio equals `exp(shift)` for a stale
engine with clipping engaged, fallback and snapshot precedence.

## Not done

A live vLLM run exercising `load_weights` on a real engine (no vLLM in the
local venv; needs a GPU pod). vLLM V1 engine-core layouts may need
`weight_loader`; the error message says so.
