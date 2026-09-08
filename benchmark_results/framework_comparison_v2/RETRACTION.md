# Retraction: native GSPO rows (2026-09-08)

The three `stateset-agents-gspo` evidence documents in `evidence/` and the
matching row of `report/comparison.md` are withdrawn as learning evidence.

At harness commit `c52bbf175424c8d8c8372777424e1f9bf518544f`,
`stateset_agents.training.gspo_entrypoints.train_with_gspo` derived training
queries from `scenario["context"]`, which GSM8K scenarios (`user_query`,
`gold_answer`, `answer_text`) do not carry, and only from the first
`generations_per_iteration` (4) scenarios. The trainer therefore sampled the
placeholder prompt "Hello" and the reward received no `gold_answer`:
`training_metrics.json` in each run's `artifact/checkpoint-48/` records
`average_reward` 0.0 and `policy_loss` 0.0 at all 48 steps. The identical
before/after pass@1 (0.172, zero seed variance) is the untouched base model;
the throughput and memory figures were measured on a seven-token prompt.

The `stateset-agents` (TRL-backed GRPO) rows are unaffected: their retained
`trainer_state.json` logs non-zero training reward at every logged step
(mean 0.19 / 0.16 / 0.19 for seeds 42 / 1337 / 2026). The direct `trl` rows
share that path's prompts and reward function; the adapter at this harness
revision did not retain TRL's log history, so their signal is not directly
recorded (the `0.53.0` adapter records it).

The documents are kept unmodified so the digest in the report still
verifies; `docs/BENCHMARKS.md` carries the corrected reading. The fix
(task prompts with full scenario context, rotation through every prompt,
and a fail-closed rejection of any run whose training reward was identically
zero) ships in `0.53.0`.
