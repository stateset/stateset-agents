# A+ hardening — design

Date: 2026-08-24. Baseline audit grade: B− (process A−, tests B+, RL correctness B−,
architecture C+, docs B, security B−). Goal: A+ on every dimension without breaking
the public API that `starter_common`, the starters, and the CLI expose.

## 1. RL loss spine — `stateset_agents/training/rl_losses.py`

One pure-torch module, no trainer state, imported by GRPO (`loss_computation.py`),
GSPO (`gspo_trainer.py`), GSPO-token (`gspo_token_trainer.py`), DAPO, GEPO and VAPO.

Functions (all take/return tensors; `torch` obtained via `get_torch()`):

- `gather_token_logprobs(logits, input_ids, response_mask) -> (logprobs, mask)` —
  the shift-by-one gather currently written five times. `mask` is the shifted
  response mask.
- `group_advantages(rewards, group_ids, *, eps=1e-8, unbiased=False)` —
  group-mean baseline, `std(correction=0)`, groups of size 1 get advantage 0,
  non-finite std → 0 (fixes DAPO/GEPO NaN on n=1).
- `k3_kl(logp_cur, logp_ref, mask, *, per_seq=True)` — Schulman k3 estimator
  `exp(r) − r − 1`, `r = logp_ref − logp_cur`, ≥ 0, gradient pulls toward ref.
  Replaces the zero-expectation `(logp_cur − logp_ref)/|y|` in GSPO/GSPO-token.
- `clipped_surrogate(ratio, advantages, *, clip_low, clip_high)` — returns
  `min(ratio·A, clip(ratio)·A)` per element. Used by GSPO-token so out-of-region
  sequences get zero gradient (currently unconditional detached weight).
- `sequence_ratio(logp_cur, logp_old, mask)` — length-normalised sequence-level
  ratio for GSPO.
- `masked_mean(x, mask, *, mode="token"|"seq")` — token-mean vs seq-mean, one
  implementation.

Fixes folded in:
- GRPO `token_level_loss` (`loss_computation.py:333-341`) double normalisation:
  use `masked_mean` over raw per-token NLL, never divide `outputs.loss` again.
- GRPO clip: sequence-mean ratio uses GSPO-scale clip (`3e-4`/`4e-4`) instead of
  the inert ±0.2; documented in `GRPOConfig`.
- `distributed_trainer._compute_grpo_loss` placeholder returning `0.0`: delegate
  to `loss_computation.compute_grpo_loss`. No silent no-op trainers.

Tests (`tests/unit/test_rl_losses.py`, pure CPU torch, skipped if torch absent):
- zero advantages ⇒ zero grad through `clipped_surrogate`
- `k3_kl ≥ 0`; one SGD step on cur decreases k3 toward ref
- ratio outside `[1−clip_low, 1+clip_high]` with sign-matching advantage ⇒
  zero gradient; inside ⇒ non-zero
- `group_advantages` on group of 1 is 0, no NaN; matches manual computation
- `gather_token_logprobs` matches a naive loop
- Each trainer's loss matches the reference formula on a tiny fixture
  (regression pins so refactors can't drift).

## 2. Green gates

- `RemoteExecutor` ABC gains `undeploy(self, deployment_id) -> None` raising
  `NotImplementedError` by default; `FireworksExecutor` overrides. CLI test for
  `undeploy`.
- Fix remaining mypy errors: `nsr_verifier.py:58,202` and `harvest.py:141`
  (`no-any-return` → cast/annotate), `river.py:95` unused ignore.
- Identify and fix the 3 failing unit tests (from the detached run).
- Register an atexit/`pytest_sessionfinish` guard in `tests/conftest.py` that
  closes litellm's async client cleanup before stdout is closed (or sets
  `LITELLM_DISABLE_ASYNC_CLIENT_CLEANUP`), so the suite ends clean.

## 3. Security

- `weights_only=True` + `map_location` on `multi_turn_checkpointing.py:102,134`,
  `single_turn_checkpointing.py:100`, `core/value_function.py:445`. Checkpoints
  that need full pickles must opt in via an explicit `trusted=True` kwarg.
- `SECURITY.md`: supported versions = current minor line; document the checkpoint
  and Redis-pickle trust boundaries; drop unverifiable contact channels.
- Move `sitecustomize.py` logic (pytest-asyncio compat patch) into
  `tests/conftest.py`; delete the root file.

## 4. Docs / DX

- `README.md:1241` → `--no-dry-run`.
- `tests/unit/test_readme_cli_snippets.py`: extract every fenced
  `stateset-agents …` line from README/QUICKSTART, run each subcommand with
  `--help`, assert exit 0 and that every `--flag` in the snippet is in the help.
- `[Unreleased]` CHANGELOG entries for NSR verifier/reward and this work.
- `docs/CLI_REFERENCE.md` wheel example bumped; `docs/ARCHITECTURE.md` layout
  section refreshed to describe `rl_losses` and the `experimental/` rule.
- `pytest.ini`: `addopts = -n auto` (xdist already installed) with
  `-p no:xdist` documented for debugging.

## 5. Architecture debt

- `cli_train.py`: replace the ten ~264-line per-model Typer commands with one
  `_register_model_command(app, preset)` generator driven by
  `core/model_presets.py`. Command names and flags are unchanged (verified by the
  existing CLI tests plus the README snippet test); only the implementation
  collapses.
- Break `core ↔ experimental`: `core/agent.py` and `core/multiturn_agent.py`
  import `long_term_planning` lazily inside the functions that use it;
  `stateset_agents/__init__.py` stops re-exporting experimental symbols
  (they stay importable from `stateset_agents.experimental`). Meta-test:
  no module under `core/` imports `stateset_agents.experimental` at module level.
- Delete `core/enhanced/advanced_rl_algorithms.py` duplicates (PPO/DPO/A2C/GSPO
  config+trainer classes and `GSPOTrainerStub`). Keep a deprecation shim module
  that re-exports `training.ppo_trainer` / `training.gspo_trainer` names with a
  `DeprecationWarning`.
- `TrainingConfig` consolidation: `training/config.py:TrainingConfig` is the
  source of truth; `core/types.py`, `core/type_system.py`, and
  `training/advanced_training_models.py` re-export or alias it (TypedDict
  variant kept only as `TrainingConfigDict` for API payloads).
- Torch import policy meta-test (`tests/unit/test_torch_import_policy.py`):
  every module-level `import torch` outside an allowlist must be inside a
  `try:` and set `TORCH_AVAILABLE`; the ten unguarded sites are fixed. The
  allowlist shrinks over time and is committed.

## Out of scope

Rewriting `api/routers/training_lab.py`; changing any CLI flag names; touching the
Rust core; PyPI trusted-publisher config (needs pypi.org access).

## Order

1 → 2 → 3 → 4 → 5, each a separate commit on `feat/a-plus-hardening` branched from
the current branch, CI gates (ruff, mypy, pytest, coverage ratchet) green after each.
