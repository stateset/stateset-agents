# Per-token GRPO — design

Date: 2026-09-05. Follow-up to the RL objectives spec. Approved in chat.

## Why

The multi-turn and single-turn GRPO trainers never record log-probs on the
trajectories they generate: `MultiTurnAgent._generate_with_model` calls
`generate(..., output_scores=True)` and discards the scores, and both
trainers build `MultiTurnTrajectory` objects from text only. At loss time
`loss_computation.py` re-tokenises the conversation text, runs one forward
pass per trajectory, and, finding no old log-probs, falls back to
`advantage * outputs.loss`: unclipped REINFORCE with a per-sequence NLL.
The eleven objective presets exist, but on this path only the two
sequence-level ones can run, and none of the clipping ever engages.

## Design

1. **Token capture at generation.** `MultiTurnAgent.generate_turn(messages,
   context) -> ConversationTurn` runs the same prompt preparation as
   `generate_response` and returns an assistant turn whose metadata carries
   `prompt_token_ids` (the exact ids fed to `generate`), `token_ids` (the
   sampled response ids, before text cleaning), and `sampler_log_probs`
   (the model's own log-probs of the sampled ids at temperature 1, from
   `compute_transition_scores` on the raw scores). Stub agents return a
   turn without token metadata. `generate_response` is unchanged and shares
   the implementation.
2. **Trainers ask for turns.** `MultiTurnGRPOTrainer.generate_trajectories`
   and `SingleTurnGRPOTrainer._generate_trajectory_group` call
   `generate_turn` when the agent has it; the environment's `run_episode`
   already accepts a `ConversationTurn` from the agent callback and keeps
   its metadata.
3. **Batched token path in `loss_computation.py`.** A group whose every
   trajectory carries token metadata on every assistant turn takes the
   token path: one row per assistant turn (`prompt_token_ids + token_ids`,
   response mask over the response ids), padded into a batch and run
   through the model in chunks of `generation_batch_size` rows (graphs
   retained, log-prob rows concatenated), `rl_losses.gather_token_logprobs`,
   advantages from `objectives.compute_advantages` broadcast to each
   trajectory's rows, and `objectives.policy_loss` with `logp_old=None`
   (both trainers take one optimizer step per rollout batch, so the policy
   is on-policy and the TRL convention `old = current.detach()` is exact;
   `sampler_log_probs` are kept for diagnostics and the async control
   plane). The native objective on this path is the `grpo` preset with
   `clip_ratio` as the symmetric clip; `TrainingConfig.objective` selects
   any preset, including token-level ones. The enhanced path adds
   `kl="k3_token"` from one no-grad reference forward on the same batch,
   and `entropy_coef` uses the differentiable entropy of the same logits.
4. **Fallback preserved.** Any group with a trajectory lacking token
   metadata takes the existing sequence-level path unchanged; its goldens
   still pin it. The loss dict reports `path` (`token` or `sequence`),
   `objective`, and `num_rows`.

## Out of scope

Off-policy correction with `sampler_log_probs` (multiple inner updates),
vLLM generation capture, and `core/multiturn_agent.py`'s backend-based
agent (it returns turns without token metadata and therefore uses the
fallback).
