# Unified RL training spine — design and migration plan

Status: approved by the maintainer 2026-10-22 ("unify behind one RL spine; public trainer
classes become thin presets; legacy duplicates deprecated; reviewed slices with learning
tests as the safety net"). Base: `fix/a-plus-review` @ 19d6e334 (v0.69.0). Paths are
relative to `stateset_agents/` unless prefixed with `tests/`, `scripts/`, `contracts/`,
`docs/`.

## 1. Inventory

**Family A — episode/trajectory loops (the de-facto spine).**
`MultiTurnGRPOTrainer` (training/multi_turn_trainer.py:190; `GRPOTrainer` alias :1895):
rollout via `env.run_episode` / `run_episodes_batched`, scoring via `score_episode`, loss
via the `loss_computation` token path (`objectives.policy_loss`, any preset but
`group_expectation`; sequence fallback without token metadata), frozen old log-probs for
`num_gradient_updates>1`, grouped AdamW + HF scheduler + fp16 GradScaler,
`NativeChunkedBackward` + `safe_optimizer_step` (trainer_utils.py:177-291), non-finite skip
policy, receipts/cadence, `TrainingRuntime` checkpoints/resume. **Stable** (constructor
frozen in contracts/public_api_v1.json). `DistributedTrainer` (training/distributed.py:108)
subclasses it via 12 loop hooks (protocol `stateset-native-loop-v3`). `SingleTurnGRPOTrainer`
(training/single_turn_trainer.py:106) duplicates the inner-update/accumulation/EWC/callback
loop (542-906) and bypasses `score_episode`. Wrappers: `GRPOHPOTrainer`, `RecoveryTrainer`,
`AutoTrainer` / `train` (stable).

**Family B — prompt/response trainers (provisional).** GSPOTrainer (gspo_trainer.py:374),
GSPOTokenTrainer (gspo_token_trainer.py:42), DAPOTrainer (dapo_trainer.py:308),
GEPOTrainer (gepo_trainer.py:195), VAPOTrainer (vapo_trainer.py:386): each has its own
generation, reward call, log-prob gather, optimizer stepping, scheduler and outer loop
(gspo_entrypoints / dapo_entrypoints / gepo_trainer / vapo_entrypoints). DAPO/GEPO/VAPO step
without a finite-gradient check.

**Family C — external / experimental / legacy.** TRL wrapper (TRL owns the loop);
experimental PPO, RLAIF, offline GRPO (IQL/AWR), BCQ/BEAR/CQL/IQL/DT; legacy
`core/enhanced` PPOTrainer + GSPOTrainerStub (removal v0.70.0 already scheduled), DPO/A2C;
`DistributedGRPOTrainer` (warns, cannot train); abstract `BaseTrainer` (no subclasses);
`ComputationalGRPOEngine`.

## 2. Variation points

Algorithmic (strategies): advantage estimator (group / GAE+critic / IQL), the declarative
`PolicyObjective` (ratio, clip, aggregation, KL), GEPO group expectation (not chunkable),
K update passes over frozen old log-probs (`num_iterations` == `num_gradient_updates`),
DAPO dynamic sampling + overlong shaping (batch transforms), VAPO critic (warmup, decoupled
GAE, positive-LM loss, two optimizers), single-turn env-step progression.

Incidental duplication (unify): 4 generation implementations (vLLM setup copied), 6 reward
call variants, full-vocab log-softmax gathers, 5 ad-hoc optimizer steps, 4 scheduler
constructions, per-group weighted accumulation in 3 places, 5 outer loops with their own
logging/save cadence/callbacks. Preserve on purpose: the meaning of `global_step` differs
(commits in A, train_step calls in DAPO/GEPO/VAPO, absent in GSPO).

## 3. Spine architecture (`training/spine/`, provisional)

- `protocols.py`: `RolloutSource` (EpisodeRollouts / PromptGroupRollouts /
  SingleTurnRollouts), `Scorer` (EpisodeScorer=`score_episode`, PromptResponseScorer,
  TurnRewardScorer), `BatchTransform` (DynamicSampling, OverlongShaping),
  `AdvantageEstimator` (GroupAdvantages, GAEAdvantages), `PolicyLoss` (TokenRowsLoss,
  SequenceLoss, GroupExpectationLoss[chunkable=False]) — all via `objectives.policy_loss`;
  optional `Critic`.
- `batch.py`: `RolloutGroup` dataclass for the response-dict schema; Family A keeps
  `TrajectoryGroup` / `SampledTurn`.
- `update.py`: `OptimizerStep` (generalised `safe_optimizer_step`: explicit optimizers,
  schedulers, scaler, params; multi-optimizer; `StepCounter` commits|train_steps|none;
  `nonfinite_grad_policy`) and `UpdateLoop` (K passes, frozen old log-probs, token/row
  shares, chunked vs single-graph, accumulation via `NativeChunkedBackward`).
- `loop.py`: `SpineLoopMixin(TrainingRuntime)` (MultiTurn `train()` + receipts moved
  verbatim, keeping the DistributedTrainer hook names) and `run_dedicated_loop` (shared
  Family B outer loop).
- `presets.py`: one `AlgorithmSpec` per algorithm (GRPO/MultiTurn, SingleTurn, GSPO/-token,
  DAPO, GEPO, VAPO).

Compatibility rules for every wrapper: keep class name + module (checkpoints validate the
trainer name and legacy config filenames), component attribute names, constructor
signatures, public methods as delegating shims; the spine calls the trainer's *bound*
methods (tests spy on them); leave patched module globals in place.

## 4. Slices (each: `make api-compatibility`, unit suite, named learning tests with
`LEARNING_SEEDS=0,1,2`)

- **S0** step goldens (tests only): scripts/capture_step_goldens.py +
  tests/unit/test_trainer_step_goldens.py — one `train_step` per GSPO, GSPO-token, DAPO,
  GEPO, VAPO (post-warmup), MultiTurn, SingleTurn at K=1 and K=2; deterministic weights,
  stubbed fixed rollouts; record loss, metrics, param-delta norm, grad norm, LR, counters;
  CPU fp32, rtol 1e-5.
- **S1** `OptimizerStep`; `trainer_utils.safe_optimizer_step` delegates. Evidence:
  MultiTurn/SingleTurn goldens identical; ddp optimizer-reporting + training-loop tests.
- **S2** GSPO/GSPO-token adopt it (`counter="none"`).
- **S3** DAPO/GEPO/VAPO adopt it (`counter="train_steps"`); fix: non-finite grads skip/raise
  instead of stepping. Don't use `clip_grad_norm_`'s return value; one `optimizer.step()`
  per pass.
- **S4** `UpdateLoop` replaces the DAPO/GEPO/VAPO-policy/GSPO μ-loops (net negative).
- **S5** Scorer adapters (verbatim behaviour; GEPO gains async rewards) — after the reward
  round-3 merge; adapters use only public `core/reward_context` functions; reward-parity
  case per trainer in test_episode_scoring_paths.py.
- **S6** PromptGroupRollouts + gather unification (GEPO → `hf_generate_group`; GSPO/GEPO →
  `rl_losses.gather_token_logprobs`; shared vLLM setup).
- **S7** `SpineLoopMixin` (move multi_turn_trainer.py:1398-1745 verbatim); ddp/native
  checkpoint/resume suites; `__init_subclass__` wrappers fire once.
- **S8** SingleTurn on the spine (update half → MultiTurn step; reward → EpisodeScorer).
- **S9** dedicated outer loops → `run_dedicated_loop`; `train_with_gspo` honours
  `use_gspo_token`; keep `_reject_unsupported_resume`.
- **S10** `AlgorithmSpec` for GSPO/GSPO-token/DAPO/GEPO.
- **S11** VAPO critic last (or keep VAPO as a specialised loop built on S1–S6 parts).

Specialised loops kept and documented: offline GRPO/IQL & offline trainers, experimental
PPO, RLAIF (only its SGD moves to OptimizerStep), TRL wrapper.

## 5. Deprecations

v0.70.0 removal (already scheduled): core/enhanced `PPOTrainer`, `create_ppo_trainer`,
`GSPOTrainerStub`, `create_gspo_trainer`. Warn now, remove v0.71.0: `DistributedGRPOTrainer`
(+ run_/launch_distributed_training), `BaseTrainer` (keep `BaseTrainerConfig`,
`BaseModelManager`), `ComputationalGRPOEngine`. Warn now, remove v0.72.0: `GSPO_Trainer`
alias, `train_with_gspo_token`. Thin presets without warnings: all stable/provisional
trainer classes and `train_with_{gspo,dapo,gepo,vapo}`. Docs: spine column in
docs/OBJECTIVES.md trainer map, spine section in docs/ARCHITECTURE.md,
contracts/component_maturity_v1.json test lists, CHANGELOG entries for S3/S5/S6/S8.

## 6. Non-goals and risks

Non-goals: changing stable signatures, adding resume to dedicated loops, unifying
TRL/offline/experimental loops, moving GSPO onto the token path (already available via
`MultiTurnGRPOTrainer(objective="gspo")`), unifying `global_step` semantics.
Risks: numerical drift (summation order, bf16 gather differences, dropout RNG order) —
mitigated by S0 goldens + 3-seed learning gates; VRAM (keep per-group backward; GEPO not
chunkable); DDP phase names/order are a cross-rank protocol (S7 must not reorder);
checkpoint schema (class/attribute names, counters); ~100 tests monkeypatch trainer
methods/module globals; concurrent reward edits (S5/S8 wait for that merge).
