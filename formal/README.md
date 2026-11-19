# Formal verification

Verification is partial. All eleven registered policy objectives have numerical
comparisons against a scalar implementation. Lean proves specific properties of
the objectives, gradients and updates. TLC exhaustively checks the listed finite
TLA+ configurations. An end-to-end proof for every shipped trainer remains open.
See the [coverage table](#rl-mathematics) and [remaining work](#open-verification-work).

From the repository root, use the pinned Lean toolchain and a JDK with the
TLA+ v1.7.4 tool JAR. Fetching the matching mathlib binaries once avoids
compiling the real-analysis library from source.

```bash
lake exe cache get Mathlib/Analysis/SpecialFunctions/Log/Deriv.lean \
  Mathlib/Analysis/SpecialFunctions/Trigonometric/Basic.lean \
  Mathlib/Topology/MetricSpace/Contracting.lean \
  Mathlib/Probability/Variance.lean \
  Mathlib/Analysis/Calculus/ParametricIntegral.lean \
  Mathlib/Tactic/{FieldSimp,Linarith,Positivity,Ring}.lean
PYTHON=.venv/bin/python bash formal/check.sh /path/to/tla2tools.jar --with-python
```

The default command builds 38 Lean roots and checks 49 finite TLC
configurations. The Python option also traces production tensor functions,
checks the resulting equations with Lean, and runs the refinement and
regression suites. It requires PyTorch. Installing the local
`stateset_rl_core` extension enables the native-extension parity tests.
The provider checks use `river-client==0.11.0` and `tinker==0.39.2` locally, with scripted transport.
The workflow pins Lean 4.15.0 and authenticates the tool JAR by SHA-256.
The Lake manifest pins the mathlib revision and its dependencies.

## RL mathematics

| Implementation | Lean contract | Implementation evidence and scope |
| --- | --- | --- |
| GRPO, Dr. GRPO, BNPO, RLOO, REINFORCE++ baseline | `centered_normalized`, `leave_one_out`, `merge_means`, `constant_length_reduction` | Finite nonempty reward groups, population scaling and three reduction conventions. The scalar loop oracle exercises all eleven entries in `objectives.OBJECTIVES`. |
| PPO, DAPO, GSPO, GSPO-token, CISPO | `surrogate_positive`, `surrogate_negative`, `ratio_bounds`, `geometric_ratio`, `sequence_token_value_gradient`, `cispo_gradient`, `overlong_bounds` | At a fixed state the expected clipped surrogate lower-bounds the exact target advantage under common support. The checks cover both advantage signs, the optional delta cap, geometric sequence ratios before exponent clipping, detached weights and DAPO's interior length penalty. For binary verifier rewards the default filter selects exactly the groups with positive empirical variance. `PolicyReduction.tla` checks token, sequence and fixed-denominator accumulation in either row order. |
| StateSet GEPO | `gepo_denominator`, `anchored_coefficient`, `weighted_mass`, `mean_deficit`, `uniform_mean_le_one`, `row_derivative` | A common log-score shift preserves the coefficient. For identical learner and sampler scores the sampler-weighted mean of the unclipped coefficients is one. Their uniform mean is at most one, with an exact variance term for the difference. The derivative at a fixed denominator includes every retained token. StateSet GEPO sums response log probabilities. Equation 9 in the paper length-normalizes each response first. [GEPO, §3](https://arxiv.org/html/2508.17850v1) |
| Sampled KL regularization | `k3_nonnegative`, `k3_on_policy` | The expectation identity assumes positive normalized probabilities on the same finite support and draws from the current policy. The restricted-support identity gives the exact missing reference mass when the sampling distribution excludes actions. Sequence-geometric scores and stale draws have different sampling measures. The fallback full-vocabulary KL and entropy use the shifted response actions. |
| PPO, VAPO, experimental PPO/A2C, Rust GAE, Decision Transformer returns | `Returns.gae_return`, `terminal_target`, `lambda_zero_td`, `lambda_one_mc`, `adaptiveLambda_bounds` | For any finite list of steps and any bootstrap, the reverse trace plus the state value equals the recursive λ-return. `AdvantageScan.tla` compares the scan to a forward sum of residuals. Python tests exercise episode boundaries, cutoffs and the Rust adapter. [GAE](https://arxiv.org/abs/1506.02438) |
| VAPO joint update | `VAPO.critic_floor`, `selected_subgradient`, `joint_gradient` | The clipped value loss has a fixed error floor. Its implemented slope defines a supporting line for every prediction and target. The shared gradient combines the regularized policy gradient, the token-mean critic gradient and the positive-token NLL gradient. [VAPO coverage](#vapo-actor-and-critic-updates) |
| Experimental A2C | `ActorCritic.shared_gradient`, `logit_loss_gradient`, `rmsprop_bound` | With frozen advantages and targets, the sampled policy, squared value error and entropy derivatives add along every shared parameter. For uncentered RMSprop with nonnegative prior accumulator, positive epsilon and smoothing in [0, 1), the absolute coordinate displacement is at most the learning rate divided by the square root of one minus smoothing. A two-update CPU test independently calculates the gradients through both heads and the shared network, the clipping factor, all RMSprop moments and the new parameters. |
| Offline Bellman policy evaluation | `LoggedBellman.value_certificate`, `critic_certificate`, `approximate_sweeps`, `sharp` | The exact finite logged-policy backup has a unique fixed point whenever the discount times the largest continuing probability is below one. Sup-norm residual and per-state RMS TD bounds give value-error certificates. The residual bound is attained by a continuing self-loop. |
| Episodic Bellman evaluation | `TransientBellman.iterate_envelope`, `block_certificate` | A uniformly contracting block of empirical transitions gives a unique one-step fixed point and a residual-based value bound, including at discount one. |
| Pooled critic loss | `LoggedBellman.pooled_critic_certificate`, `pooled_sharp` | Bounds value error from the pooled squared loss on published targets, within-state label variance, minimum state frequency, mean signed rounding error and target-network lag. CPU examples attain the bound. |
| Exact empirical critic audit | `LoggedBellman.empirical_moments`, `pool_counts`, `tolerance_iff`, `counted_critic_tolerance` | For uniformly weighted logged rows, the count-based decision certifies an absolute value-error tolerance. The audit uses rational arithmetic on captured targets, including losses below the binary64 range. |
| Auxiliary value and Q update | `AuxiliaryCritic.frozen_gradient`, `relu_head`, `relu_drift`, `restored_coordinate`, `published_prediction`, `after_update` | Derives the two-layer ReLU gradient away from activation boundaries and bounds prediction drift even across a boundary. Connects restored AdamW coordinates to the full gradient history. CPU runs compare both heads and the published target before and after checkpoint resume. |
| Integer Bellman inputs | `IntegerBellman.align`, `quotient_increment`, `grid_nearest`, `finer_grid_nearest`, `legal_result`, `clipped_nearest` | Aligns binary fractions exactly before nearest-even rounding. The returned magnitude is representable and at least as close as each competing destination value, including those on a finer grid than the input. Signed and unsigned CPU cases cover cancellation at the 64-bit limits. |
| Bellman target rounding | `BellmanRounding.round_cell`, `compensated_residual`, `residual_sign` | In a cell with two adjacent endpoints, the midpoint correction recovers direct rounding under the stated precision and residual-sign hypotheses. Native CPU tests cover both signs in every normal binade of float16, bfloat16 and float32, the subnormal grid and overflow. |
| CQL | `bellman_optimality_lipschitz`, `softBackup_lipschitz`, `softBackup_bounds`, `cqlGap_centered`, `cqlGap_derivative`, `sampled_critic_gradient` | The finite stochastic optimality backup is γ-Lipschitz in the sup norm. The normalized soft backup stays within proposal extrema. For positive temperature, the unnormalized penalty gradient averages proposal slopes with softmax weights and subtracts the logged action's slope. StateSet's adaptive threshold applies to the sum of the two critics' mean penalties. [CQL](https://arxiv.org/abs/2006.04779) |
| CQL proposal sampling | `CQLSampling.expected_softBackup_bias`, `softBackup_tail`, `expected_slope_bias` | For a frozen critic, pairwise independent proposals with common moments give finite-sample bounds for the expected soft value, individual batches and the softmax-weighted penalty slope. |
| IQL and offline policy extraction | `expectile_derivative`, `expectile_batch_derivative`, `network_expectile_gradient`, `expectile_empirical_optimum`, `capped_weight`, `weighted_likelihood_optimal` | The empirical expectile loss has a derivative at every prediction, including an exact fit. For each nonempty finite sample and interior expectile, a unique scalar minimizer exists between the smallest and largest targets and balances the weighted residuals. At a fixed state, normalized positive empirical weights maximize the unconstrained categorical policy fit. [IQL](https://arxiv.org/abs/2110.06169) |
| Logged-action IQL language policy | `LoggedPolicy.regroup`, `extraction_gradient`, `score_gradient`, `unique_prefix_optimum`, `dataset_optimum` | Each action contributes its capped advantage weight divided by its supervised token count to each observed prefix and next token. At a visited prefix the normalized token masses uniquely minimize finite categorical loss, including when some tokens were never observed. |
| BCQ and BEAR | `bcq_displacement`, `convex_bounds`, `target_interpolation`, `gaussianKL_nonnegative`, `laplacian_nonnegative`, `gaussian_nonnegative`, `dual_loss_derivative` | Convex critic mixing, the BCQ perturbation bound per coordinate, Polyak interpolation, Gaussian latent KL and the BEAR dual derivative. Both exact radial kernels give nonnegative MMD² for any nonempty finite policy sample compared with the one observed action used by the actor. Tests check the kernels under a common offset and their gradient at coincident actions. The first Adam step is checked from zero moments. [BEAR](https://arxiv.org/abs/1906.00949) |
| Decision Transformer | `causal_action_index`, `trajectory_target_counts`, `DecisionReturns.project_actions`, `certified_return`, `conditional_action_fit` | For interleaved return/state/action positions, a state query can attend only to earlier action slots. A two-block PyTorch model checks invariance to masked keys, future inputs and the action being predicted. Epoch action error uses all retained steps. State error counts only steps with a retained successor in the same trajectory. |
| Experimental DPO | `Preference.pair_gradient`, `calibrated_margin`, `population_optimum`, `gibbs_optimal` | At a fixed prompt and positive β, the Gibbs policy maximizes KL-regularized reward among positive policies on a finite response set. It uniquely minimizes population preference loss when positive-weight, reward-calibrated comparisons connect that set. The tests exercise teacher-forced scoring, every parameter gradient in a tiny causal transformer, fixed-reference AdamW updates and low-precision parameter publication. [DPO, §§4–5](https://arxiv.org/abs/2305.18290) |
| Categorical policy gradients and KL | `reward_gradient`, `change_measure`, `corrected_k3`, `fixed_draw_k3_gradient`, `forward_kl_gradient` | A differentiable normalized finite policy has the expected score-function gradient with an action-independent baseline. Exact, unclipped action ratios change measure under common support. Fixed-draw k3 has its own derivative. A three-action tensor test checks both the corrected and uncorrected gradients. [REINFORCE and RLOO](https://arxiv.org/abs/2402.14740) |
| Log-probability scoring and exploration | `log_softmax_shift`, `entropy_bounds`, `novelty_decreases`, `ucb_bonus_antitone` | Adding a common logit offset leaves each log probability unchanged. Entropy of a normalized distribution on n actions lies in [0, log n], including zero-mass actions. Repeated visits lower the curiosity exploration probability. At a fixed total pull count, more action pulls reduce the UCB bonus. Tests cover high-offset logits, unavailable actions, duplicate actions and repeated states. |
| Replay selection and batch averaging | `ReplayMixture.distinguished_draw`, `priority_bias`, `mixed_bias`, `rounding_gap` | The effective replay gradient depends on reservoir capacity, priority, stochastic rounding and the actual denominator of each group. [Selection and batch averaging](#replay-selection-and-batch-averaging) |
| Off-policy GRPO group gradients | `ReplayGradient.peer_bias`, `bias_bound`, `joint_support`, `middle_logit` | Correcting one response leaves an exactly quantified error from the peer distribution. A three-action example gives four distinct expected updates. [Replayed group gradients](#replayed-group-gradients) |
| Continual GRPO replay | `ContinualReplay.uniform_reservoir`, `partial_round`, `fair_quota`, `capped_rounded_total` | Every retained subset of the target size has the same number of draw histories. A partial balanced round gives each active task equal inclusion probability. The expected capped replay count equals the capped requested mean. [Replay sampling](#continual-grpo-replay) |
| Distributed GRPO and continual learning | `GradientBank.finalize_sum`, `average_replicated`, `fisher_online`, `ewc_nonnegative`, `ewc_gradient` | At any finite rank and microbatch counts with positive total population, count weighting followed by all-reduce and final scaling gives the global mean gradient per bank in real arithmetic. The EWC penalty with nonnegative Fisher weights has zero gradient at the anchor. Native two-rank tests compare updates and restore across unequal populations. [Elastic weight consolidation](https://arxiv.org/abs/1612.00796) |
| EWC importance estimates | `FisherInformation.trajectory_gradient`, `token_gap`, `fixed_length_information`, `kl_curvature` | The squared trajectory score can lose all the within-trajectory token variation. At a normalized policy anchor the curvature of KL equals the policy expectation of the squared action score. [Trajectory scores and Fisher information](#trajectory-scores-and-fisher-information) |
| Distributed gradient rounding | `CollectiveRounding.exact_tree`, `tree_error`, `rounded_mean`, `fine_then_coarse`, `finalized_error` | For any rank order and binary reduction tree, signed integer-grid sums are exact when the absolute mass fits the significand and the exponent range suffices. A single rounded rational mean is within half a destination unit. An intermediate nearest-even grid with an even number of subdivisions exceeding the denominator preserves that rounding. The error bound for a merged update weights each bank's error separately. |
| KL controllers | `cosine_bounds`, `cosine_endpoints`, `adaptive_direction` | The default half-cycle reaches its configured minimum. Projected proportional updates move in the direction of the error when the starting coefficient lies inside the bounds. Tests exercise warmup-only and decay-only schedules, and the hybrid controller's fixed baseline. |
| Gaussian MMD and CORAL | `KernelMMD.empirical_nonnegative`, `gaussian_translate`, `gaussian_direction`, `CORAL.covariance_pairs`, `CORAL.gradient` | Gaussian MMD² is nonnegative for arbitrary finite empirical sample sets. The CORAL objective compares unbiased per-domain sample covariances. Pairwise and gradient tests cover translated features and unequal populations. |
| Sim-to-real DANN and distribution metrics | `DomainAdversarial.batch_gradient`, `strict_bayes`, `optimal_risk_js`, `js_zero_iff` | The encoder receives the reversed domain gradient and the discriminator receives the ordinary one. For normalized feature frequencies the optimal balanced discriminator loss equals 2 log 2 minus twice the Jensen–Shannon divergence. [DANN and vocabulary evaluation](#sim-to-real-domain-adaptation) |
| Reward training and RLAIF | `reward_mse_decomposition`, `welford_update`, `calibrated_monotone`, `weighted_loglikelihood_gradient` | Scalar squared error is minimized at the empirical mean. Both reward trainers report MSE over examples, including a short final minibatch. The online variance update matches a centered sum of squares. Frozen positive-scale z-score calibration preserves order. Tests exercise buffer eviction and periodic refresh. The standalone RLAIF loss differentiates fixed reward-weighted log probabilities over the scored tokens. Its configuration caveat is recorded below. |
| RLAIF judge-score normalization | `RewardWeights.repeat_rows`, `centered`, `variance_numerator`, `scaled_weight`, `standardized_mass`, `row_loss_gradient`, `masked_causal_score` | Exact per-completion counts reproduce the flattened token mean, variance and score derivative. For more than one scored token, the weighted sum of squared standardized weights is bounded by the token count minus one. Single-token batches use the raw reward. Tests compare every gradient and SGD parameter to a token-by-token reference. Masked target indices and undefined logits at inactive positions have zero contribution and zero logit gradient. |
| Bayesian reward models | `BayesianReward.negative_log_density`, `paired_batch_derivative`, `unique_best_heads`, `total_epistemic`, `gaussian_kl_formula` | The paired Gaussian loss has the stated mean and log-variance gradients. With positive empirical label variance, the optimal constant heads uniquely match the label mean and log population variance. Predictive variance adds mean conditional noise to variation within and between ensemble members. The diagonal posterior KL is nonnegative. |
| Adam and AdamW | `Optimizer.moment_expansion`, `AdamHistory.history`, `displacement`, `same_decay` | Bias-corrected moments use the gradients observed by each parameter. For arbitrary finite histories, the two normalized moment kernels bound the adaptive displacement from the decayed weight. Equal positive betas give a bound of one learning rate per coordinate. CPU updates are checked against independently expanded histories. |
| Native L2 gradient clipping | `Optimizer.clipped_l2`, `GradientClipping.reduction_error`, `stored_clip`, `head_caps` | For exact coordinates, each independently clipped group satisfies its own cap. The joint norm is bounded by the root sum of squared caps. The floating-point bound includes norm-measurement and coordinate errors. CPU tests exercise million-coordinate reductions and both PPO optimizer steps. |
| River and Tinker RL | `wire_at`, `shifted_target`, `surrogate_scale`, `is_derivative`, `transport_sum`, `prefix_measure_gap` | The causal layout preserves precisely the next retained token at any positive context limit. Global advantage scaling commutes with the documented additive IS, PPO and CISPO terms. An exact conditional ratio changes the action distribution at a sampled prefix; a product of ratios changes the path measure under common support. [River loss definitions](https://docs.river.ai/guides/losses/) · [Tinker importance sampling](https://tinker-docs.thinkingmachines.ai/tinker/losses/importance-sampling/) |

The Bayesian reward model gives each retained draw equal probability. A single draw has zero empirical epistemic variance and retains the variance predicted by its noise head. Training pairs each scalar target with the mean and log variance from the same feature row. `paired_batch_derivative` proves the update direction through both heads for a shared parameter. `broadcast_penalty` proves that the old all-pairs computation charged exactly the target variance at a perfect fit with unit noise. When the residual is exactly zero, lowering the fitted log variance always lowers this unregularized loss. A moment-matched normal interval has distribution-dependent coverage. A native test exhibits 93.75% coverage for its nominal 95% interval in a separated Gaussian mixture. These properties concern the fitted finite mixture. The predictor accumulates binary reward means on the same exact integer grid as the rollout code. The log-variance exponentials and the network that produces the draws use machine arithmetic. Calibration against real reward outcomes requires observations. [Deep Ensembles](https://arxiv.org/abs/1612.01474)

For group reward estimators, `population_numerators` proves that integer sums
and sums of squares on a common grid recover the exact group-centered reward
and population variance for any finite nonempty group. `deviation_gate`
proves the strict standard-deviation cutoff, including equality.
`scaled_population_advantage` proves that rescaling the group residual,
batch variance and epsilon preserves the standardized advantage. The canonical
PyTorch estimators accumulate these statistics over scalar rewards on the CPU
before rounding the final advantages to float32. Rust and the NumPy fallback
also center before rounding. The accelerator retains its historical unnormalized
baseline differences below its normalization threshold.

At a fixed prompt, suppose responses are independent draws from the same
differentiable normalized policy and each reward depends only on its own
response. Then the zero-mean policy score of one response has zero cross
moment with the reward of another. Under these assumptions, `group_score_expectation` proves that the self-including mean
multiplies the expected unclipped reward gradient by (n−1)/n. Leaving out
the scored response recovers the expected gradient for n > 1. The result
is for unstandardized advantages. Python differentiates the actual surrogate
and enumerates every Bernoulli response group of sizes two through five.
[GRPO](https://arxiv.org/abs/2402.03300) · [RLOO](https://arxiv.org/abs/2402.14740)

The native DDP reward-moments collective exchanges 804 bytes per rank for the
integer sum and sum of squares. `moment_packet_bounds` proves their magnitude
bounds for binary64 inputs and int64 counts. CPU Gloo tests compare both
workers with an independent rational oracle, including cancellation and
rewards closer than a float32 spacing. The normalization ends with a binary64
square root and division. Universal correct rounding of the final float32
advantage and accelerator collective execution remain open.

For a finite judge score and finite increasing rubric bounds, `judge_bounds`,
`judge_monotone` and `judge_recover` prove that calibration maps into [0, 1],
preserves order, and recovers the clamped input when the scale is reversed.
The scalar judge computes the affine ratio exactly before rounding it to binary64.
A nonfinite provider score raises on an individual request. The batch API
uses its existing 0.5 fallback at that request's position. A response without a
score returns 0.5 in output units and can be retried. Scripted local
transport exercises the judge, trajectory collection, scoring and SGD together.

For online PPO, the collector saves the sampled token IDs, their behavior log
probabilities and the critic's values. Before each proposed epoch, the trainer
averages `expm1(current − behavior) − (current − behavior)` over the sampled
response tokens. Under common support its expected value at a fixed prefix is
KL(behavior‖current). An epoch runs at equality and stops above the configured
cap. An accepted update can cross the cap. `approx_kl` records the measurement
before the last accepted update. The optional reference penalty averages sampled
k3 values per response and then across responses. After the policy changes,
that penalty evaluates the behavior actions at their original prefixes. A scalar
SGD oracle checks the joint actor and critic update and both scheduler counts
through repeated epochs. The result covers the sampled batch and the tested CPU
optimizer path. [PPO](https://arxiv.org/abs/1707.06347) · [PPO early stopping](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

With EOS or a finite horizon, the token count is itself random. Lean proves
how the summed statistic relates to path KL and how the native token average
depends on which continuations entered the batch. [PPO stopping on sampled trajectories](#ppo-stopping-on-sampled-trajectories)

In DAPO and VAPO, `old_logprobs_source="sampler"` uses the probability of the
selected token after the generator applies temperature and vocabulary filters.
The learner scores the raw categorical policy at that prefix. When temperature
is one, dividing the original raw probability by its top-p conditional
probability gives the total retained mass, so even the first ratio can differ
from one. Lean proves this identity, exact change of measure on the retained
support, and the clipped lower bound there. Tests intercept the actual PyTorch
multinomial weights and check the scalar loss and every selected-token derivative
on two optimizer updates in both trainers. Native agent turns also retain the
raw scores under `model_log_probs`. The default recompute mode freezes raw
scores with dropout disabled. If collection returns no actions, both trainers
skip the optimizers and learning-rate schedules. Regression tests first populate
Adam's moments, then check that an empty batch preserves every parameter,
moment and scheduler field. VAPO counts committed critic updates during value
warmup and restores the count from a partial checkpoint. [DAPO](https://dapo-sia.github.io/static/pdf/dapo_paper.pdf) · [Transformers generation scores](https://huggingface.co/docs/transformers/main/internal/generation_utils)

Hugging Face passes finite softmax weights to `torch.multinomial`. The replay
score normalizes those weights in their original sampling measure. For a
dominant action, Lean's `complement_log` establishes the equivalent expression
`−log1p(other_weight / chosen_weight)`. A 400-digit decimal oracle exercises
the dominant and subnormal actions in float16, float32 and float64 on CPU.
The tests cover the categorical distribution specified by the weights.
Verification of the random-number generator and individual draw frequencies
remains open.

The vLLM checks exercise the adapter with local engine doubles. They verify
the requested log-probability mode, the selected token index, weight
publication and prefix-cache invalidation. Verification of the live GPU
sampler and its distributed cache requires a running engine. [vLLM score modes](https://docs.vllm.ai/en/latest/api/vllm/config/model/)

`PPOBatch.split_return` and `split_advantage` prove that accumulating the
reward and bootstrap contributions separately gives the λ-return and GAE
for any finite trajectory, including internal terminations and a terminal
or nonzero time-limit bootstrap. Rational forward-sum tests cover both
outputs in the tensor scan and the Python and native Rust kernels. The
two outputs round separately in their destination dtypes. The VAPO update
passes the stored critic return through to the value loss. Standalone callers
that supply only a rounded advantage and an old value can lose a small
reward when reconstructing the return. [GAE](https://arxiv.org/abs/1506.02438)

IQL also accepts the endpoint expectiles. At zero, any prediction at or below
the smallest target gives zero loss. At one, any prediction at or above the
largest target gives zero loss. `expectile_endpoint_minima` proves that these
are exactly the minimizers.

`formal/check_torch_graphs.py` traces fixed-shape CPU floating-point kernels through
PyTorch FX. Lean checks 57 extracted equations under real-arithmetic
interpretations of the supported ATen operations. Three Gaussian reward traces check a labeled row and both head gradients, including the log-space tail and a zero residual. Two Decision Transformer traces check a scalar squared-error contribution and its backward with three observed vector targets. The logged-action IQL trace checks the capped exponential weight at temperature two and cap one on both sides of saturation. Two traces check the ATen decompositions of negative log-sigmoid and its backward against the DPO loss and derivative, including a tie. Three traces cover CQL's
two-proposal penalty and its gradients with respect to both proposals and the
logged action. The other traces cover k3, bounded exponentiation, asymmetric
clipping, PPO, CISPO and GSPO-token backward passes, the IQL expectile value
and gradient, the Bellman target, PPO's clipped squared value error with unit or arbitrary residual scaling, and EWC's value and gradient. Nine traces cover a configured binary64 discount and both input slopes for float16, bfloat16 and float32 Bellman outputs. Three further traces check the value, prediction slope and stopped target of the critic squared error. The zero-discount
value and successor gradient have separate traces. Two float32 target-interpolation traces
check both branches at weights ¼ and ¾, including the real-arithmetic meaning of the casts. Each proof
covers the stated real domain at the traced shape and hyperparameters. Three CPU RMSprop traces check the updated parameter, squared-gradient accumulator and step counter with A2C’s default settings and a nonnegative incoming accumulator. [PyTorch RMSprop](https://docs.pytorch.org/docs/2.11/generated/torch.optim.RMSprop.html)
Numerical tests exercise larger batches, masking, low precision and extreme
temperatures, including tied CQL maxima at the smallest positive binary64
temperature. The checker rejects unsupported ATen operations and shapes.

For CQL, `softBackup_derivative` proves that the normalized target's directional derivative weights each proposal slope by its softmax probability. For the two scores −τ and τ, `symmetric_softBackup` gives τ log(cosh 1). Before the correction, the largest finite float32 and float64 temperatures returned 0.306853τ and nonfinite gradients. The correct value rounds to 0.433781τ.

At temperatures above the dtype maximum divided by 1024, the implementation evaluates the forward partition on the CPU with decimal precision set from the smallest output subnormal and computes the analytic derivative on the tensor device. This branch incurs a device transfer for accelerator tensors. For ordinary temperatures, small signed logits with a nearly canceled target use the odd/even identity `expm1_odd_even` on the tensor device. Tests compare forward and reverse derivatives, the finite penalty after subtracting a large baseline, and symmetric backups below the smallest normal value.

| TLC configuration | Finite domain and checked property |
| --- | --- |
| `AdvantageScan.cfg` | Two steps, signed unit rewards/values including the bootstrap, all terminal flags, and γ and λ in {0, ½, 1}. Every completed suffix matches the forward sum. Weak fairness yields completion. |
| `PreferenceBatch.cfg` | Two preference pairs per update, up to two actions per completion, two length caps, a fixed reference and two successive learner heads. All 2,016 reachable states passed. An independent rational table checks the signed mean gradient, including reversed pairs, duplicate pairs, truncation ties, and abort/retry after a partial backward pass. |
| `ActorCriticSegments.cfg` | Two collection calls, each up to two transitions, with terminal or bootstrapped cutoffs, three trace decays and update requests between collections. Across 1,152 reachable states, the model checks segment publication, cancellation or failure before publication, policy version consistency and each target against independently expanded n-step returns. |
| `PPOEpochs.cfg` | Three sampled actions across two unequal response rows, six caps and up to three optimizer steps. The model checks every accepted policy against the frozen behavior scores, stops at the first observed crossing and advances the scheduler once. TLC explores 7,912 states. Lean's `epoch_table` bounds the two nonzero table entries. |
| `StoppedPolicy.cfg` | Three continuation probabilities and one to three sampled responses. Across 225 states TLC checks the probability of each stopped history, the number of scored actions, admission at ten caps and completion. The native test covers all 117 ordered batches with the same action probabilities. |
| `FeatureAlignment.cfg` | Up to two source and three target observations in two bins, any available subset and every insertion order. All 8,800 states agree with the squared difference in bin frequencies for Gaussian MMD² and the centered second moments for covariance. |
| `DomainAdversarial.cfg` | Three domain-labeled rows, two feature bins, three logit values per bin, four reversal strengths and every processing order. The two accumulated gradients agree with an independent calculation grouped by domain and feature value. Missing domains reject the update. TLC visits 16,704 distinct states. |
| `VAPOUpdate.cfg` | Three response rows in two groups, zero to two tokens per row, every positive subset, three loss reductions, optional critic warmup and one or two joint updates. TLC checks the accumulated shared and critic gradients against independent token sums, aborts before commit on a failed group and checks fair completion. It visits 48,000 distinct states. |
| `ReplayStrata.cfg` | Three named tasks of sizes one, two and three, plus two unlabeled entries. Each of nine batch sizes carries all 864 combinations of task and row permutations. TLC checks exact inclusion probabilities, distinct entries and independence of task choices across rounds. Python enumerates the native integer draws for the same pools. |
| `ReplayMixture.cfg` | Every slot history for three groups and capacities one through three, every reward ticket and both rounding outcomes. All 332 states agree with each entry’s conditional priority mass, the marginal selection formula and the normalized gradient formula for either fresh population and either replay size. |
| `FisherInformation.cfg` | Eight recorded examples and every ordered pair of attempts. Across 271 states, TLC checks the preceding-token context, signed score, squared trajectory mean, rejection of partial rows, accepted count and completion. |
| `ReplayGradient.cfg` | Two categorical learner policies, three logit coordinates, four weighting rules, all nine ordered response pairs and every order of accumulation. All 12,312 states satisfy denominator provenance and the independently summed gradient identity. |
| `ContinualReplay.cfg` | Capacities one through three and streams of up to six groups. Each state holds every slot-draw history for its stream length and capacity. TLC checks the exact multiplicity of every possible retained subset in all 21 states. |
| `SamplerReplay.cfg` | Two different prompt lengths, two prefix-dependent categorical heads, two temperatures, optional truncation, empty rows, early EOS and padding. TLC exhausts 464 states, checks each recorded probability and replayed ratio against independent rational tables, and checks that an empty batch commits zero updates while a populated batch commits two. Python checks all 80 completed batch and learner-head combinations. |
| `GroupExpectation.cfg` | The policy-loss input has three responses in two prompt groups, two categorical heads and up to two tokens per response. Its singleton group exercises the loss API. The native trainer admits generated groups of size two or greater. TLC exhausts 4,455 states and checks frozen sampler masses, coefficients, signed clipped gradients and update counts. A PyTorch test compares all 686 batch and learner-head combinations with rational arithmetic. Gradient comparisons cover smooth points, with a separate test at an exactly representable clipping tie. |
| `GroupAdvantages.cfg` | Four signed rewards with three prompt partitions, two common offsets, two deviation cutoffs and every insertion order. At each subset TLC checks both prompt and batch variance against pairwise differences and the centered numerator against a sum of reward differences. All 15,552 states passed. |
| `RegularizedBatch.cfg` | Two groups with three response rows, zero to two tokens per row, three reductions and up to three inner updates. Both group orders reproduce independently summed policy, per-row KL and per-token entropy terms. Empty batches advance neither optimizer nor scheduler. All 16,992 reachable states pass. |
| `PolicyReduction.cfg` | Two rows with lengths 0–2, three ratio levels, signed rewards and three reductions. The streaming numerator and denominator agree with the independent batch expression at every visited subset. |
| `PromptReduction.cfg` | Two prompts have one and two responses, with lengths from zero to two. In both prompt orders, the token-weighted or fixed-denominator policy term plus row-mean KL agrees with independent action and response counts. The native prompt mean is checked separately. |
| `ProviderBatch.cfg` | Two River rows, prompt lengths 1–2, two generated tokens each with independent signed or zero credits, and context limits 1–4. Every order and partition of active rows produces the independent reference update under token, sequence and sum reduction. A fair run completes. An empty batch makes zero optimizer calls and a nonempty batch makes one. |
| `GradientBanks.cfg` | Two ranks with two microbatches each, unequal response and token counts, and empty ranks. TLC checks every represented ordering against independently weighted global policy and entropy gradients. Weak fairness yields finalization. |
| `RoundedGradientBank.cfg` | Two ranks with signed policy and entropy coordinates, independent counts, and optional empty ranks. For every collection order, the final coordinate satisfies the nearest-even distance and parity rule for the exact combined rational. A fair run completes. TLC checks 6,840 states. |
| `RewardTokenWeights.cfg` | One or two completions with one or two scored tokens each, signed rewards and independent signed token slopes. Both row orders give the same sample-variance and policy-gradient numerators as the expanded pairwise token calculation. TLC checks 20,160 states, including the raw single-token branch and unused positions containing a sentinel. |
| `JudgeScores.cfg` | Two requests, three scoring intervals, valid and invalid returned scores, and either completion order. TLC checks 3,600 states for correct per-row calibration and cache reuse. Invalid evaluations bypass the cache and take the batch fallback at their own index. |
| `OptimizerWindow.cfg` | Two windows, 4,972 reachable states, signed gradients along (3,4), two correction factors and two caps. Empty and AMP-overflow windows leave the optimizer and scheduler in place. Every finite update uses the pre-step learning rate and satisfies both clipping bounds. |
| `BayesianReward.cfg` | Collects four conditional reward distributions in every order for two models with two draws each. All 2,401 reachable states check the population variance against a pairwise calculation. Complete mixtures check the sum of the within-model, between-model and conditional noise terms. |
| `AdamHistory.cfg` | All 64 three-step histories of signed unit gradients, zero gradients and missing gradients. A finite working grid, a coarser published grid and an optional checkpoint restart at each step yield 1,072 reachable states. The model checks both local moments and each rounded update against an exact square-root comparison. |
| `GradientClipping.cfg` | Two signed two-coordinate heads, five caps and every order and partition of their coordinates. TLC checks the partial energies, all four output coordinates, separate caps and eventual completion in 2,125 reachable states. |
| `LoggedExtraction.cfg` | Three logged actions, one or two tokens per action, three raw advantage levels, two frozen policy heads and batches of one or two rows. Independent prefix counts check each accumulated gradient and action count, including after an optional checkpoint restore. TLC exhausts 134,784 states. |
| `RecordedReturns.cfg` | Three decision rows, all combinations of termination and truncation, discounts of zero, one half and one, and optional externally observed returns. Across 20,160 reachable states the model checks episode offsets, each complete-suffix label, and rejection before publication when a needed continuing return is unknown. |
| `PooledCritic.cfg` | Two states with unequal or equal frequencies, two labels per state, four termination patterns and two successor maps. Across 24,576 states, the model checks the variance decomposition and the value-error bound against the exact solution of the two Bellman equations. |
| `CriticMoments.cfg` | Across 9,216 states the model collects three rows in every order. It checks the running totals, compares the variance with independently centered observations, and checks accepted tolerances against the exact two-state solution. Cases include missing coverage and unit contraction. |
| `AuxiliaryCritic.cfg` | In 30 reachable states, two heads read one frozen label, take separate scalar AdamW steps and publish the updated target. Both sample orders and a checkpoint between updates agree with independently expanded weights and moments. The finite grid uses signed unit gradients and a target interpolation weight of ½. |
| `IntegerBellman.cfg` | Integer and fractional operands, five discounts, a terminal mask, and a three-bit destination significand. The independent oracle compares every allowed endpoint with the exact target. TLC checks alignment, distance, tie parity and completion in 13,950 states. |
| `BellmanRounding.cfg` | Tests signed two-bit significands, subnormal spacing, a virtual overflow endpoint, 33 discounts and an optional intermediate cast. Across 126,852 reachable states the reference compares the published target with every endpoint by distance and tie parity. |
| `TransientBellman.cfg` | All 5,184 reachable states cover two-state tables with two possible outcomes per state at discount one. TLC checks the finite-horizon residual bound against the exact two-state solution and checks which rows can reach termination. |
| `LoggedBellman.cfg` | Two states, two observations per state, four combinations of termination and cutoff, three discounts, and three synchronous sweeps. All 9,984 reachable states check the target against an unrolled reward path, the exact fixed-point error against its residual certificate, and the set of successor values read. |
| `OfflineBackup.cfg` | Two action proposals, two critics taking values −1, 0, 1, three mixture weights and discounts, both terminal flags. The committed target is attainable and bounds every candidate, including when all Q values are negative. |
| `TargetPolicy.cfg` | Four completed updates with optional checkpoint save and restore, three candidate online parameter triples, and target publication after steps one and three. Every observed target agrees with an independent closed-form parameter history. Inactive backups skip the successor. TLC exhausts 3,945 states. Substituting the live policy at the read produces a counterexample on the first update. |
| `OfflineEpoch.cfg` | One to four transitions, batch capacities one to five, arbitrary shuffle and signed per-row observations that change after each update. Every fair epoch visits all rows, makes the ceiling number of updates and weights the recorded batch means by transition count. |
| `DecisionConditioning.cfg` | Four conversation turns with signed rewards, every assistant-position mask, four discounts, two integer grids and two context capacities. An independent forward polynomial checks the reverse scan and every retained action label. All 186,624 reachable states pass. |
| `TrajectoryEpoch.cfg` | Three trajectories with lengths and context limits one to three, batch capacities one to four and arbitrary order. Both epoch means match independent counts of predictions and observed errors, including zero successor targets. A fair epoch completes. |

### DAPO group accumulation and regularization

DAPO accepts groups with unequal response lengths and incomplete batches after
dynamic filtering. During each inner update, its token-mean surrogate weights
each group by its fraction of the batch's sampled tokens. Sequence means and
fixed-length reductions use the group's fraction of response rows, including
empty responses. Sampled sequence KL also uses the row count. Categorical
entropy always uses the active token count. With token KL or full-vocabulary
forward KL, each penalty follows the surrogate's reduction.

Lean proves that summing the weighted group gradients recovers the gradient
of the full objective in a fixed parameter coordinate. It also proves the
gradient of full forward KL plus negative entropy for a positive normalized
learner and a fixed positive reference. The sparse-support identity accounts
for reference probability on actions the learner assigns zero probability.
Full forward KL is infinite when a learner-supported action has zero
reference probability. The trainer rejects that update.

At a single fixed prefix, the KL and entropy penalties alone have an explicit
minimizer. If the KL coefficient is β and the entropy coefficient is η, with
positive sum, raise each positive reference probability to β/(β+η) and
renormalize. Lean proves that the excess penalty of any positive policy equals
(β+η) times its forward KL to that distribution. The sampled reward surrogate
also contributes to the actual DAPO update.

The numerical tests compare two consecutive CPU AdamW updates with an
independent unpadded calculation in 90 configurations. They cover three
likelihood-ratio definitions, three reductions, entropy on and off, and five
KL settings including bias-corrected token KL. They check every gradient,
parameter and optimizer moment in a categorical policy table.
The trainer snapshots the initial learner or the supplied reference, freezes
the snapshot, and restores it from checkpoints. A failed group discards
pending gradients for that inner update. These checks assume all groups in
one update see the same learner.
Convergence, failures inside the optimizer and accelerator arithmetic remain
open.

The full KL calculation uses a degree-six approximation of exp(d) − 1 − d
when the absolute log-probability difference is below 0.001. Lean bounds its
real-arithmetic value error by |d|⁷/4410 and its derivative error by
7|d|⁶/4320. Two equations traced from PyTorch connect the implemented
polynomial and its backward pass to those bounds. The bounds cover
truncation before floating-point rounding.

### VAPO actor and critic updates

For any ordered clipping interval, Lean proves that the clipped critic loss
equals the larger of the ordinary squared prediction error and the squared
distance from the target to that interval. This characterizes all minimizing
predictions. `selected_subgradient` proves that the slope chosen by the
clamp and maximum gives a global supporting line, including tied errors
and coincident clip bounds. A PyTorch backward trace at clip radius ¼
checks the implementation against that formula. The joint-gradient theorem
sums the actor, critic and positive-token log-likelihood derivatives along a shared parameter.

The CPU regression compares two AdamW updates with an unpadded calculation
across 36 combinations of policy ratio, reduction and KL mode. It checks
individual clipped gradients, updated parameters and optimizer moments for
both the policy network and the value head. Additional tests cover empty
and single-response groups, critic-only warmup, a frozen KL reference,
checkpoint replay and an interrupted optimizer step. A seeded GPT-2 test
checks that changing the entropy coefficient changes the trained policy.
The finite TLA+ model commits both optimizers in one transition. Python
recovers from a partial optimizer commit by requiring a complete checkpoint reload.

### Sim-to-real domain adaptation

DANN gives each nonempty domain equal weight, including when their batch
sizes differ. Lean proves the derivative of both domain means and their
equivalence to a loss on empirical feature frequencies. The discriminator
descends that loss. The encoder descends its task loss minus the reversal
strength times the domain loss. Four PyTorch traces check the public loss and
its source, target and shared critic gradients at a scalar affine critic.
A CPU regression checks two Adam updates of a nonlinear discriminator and
shared encoder against manual backpropagation. It also exercises all 1,728
valid snapshots of the finite TLA+ model. [Ganin et al., DANN](https://jmlr.org/papers/v17/15-239.html)

At a feature with source mass s and target mass r, the optimal real-domain
probability is r/(s+r) when s+r is positive. A finite logit attains the
optimum when both masses are positive. With disjoint supports, the optimum
is approached as the logits diverge. Lean proves that the Jensen–Shannon
divergence is between zero and log 2 and vanishes exactly when the feature
distributions agree. The evaluator computes this quantity from assistant
word frequencies. Its numerical tests compare with a decimal reference,
including disjoint supports, unequal domain sizes and nearly equal bins.
The theorem concerns a freely chosen discriminator at each feature. A trained
neural critic may have less capacity or stop short of the optimum. Transfer
performance on real conversations remains an empirical question.

The evaluator's near-equality JS formula retains the quadratic contribution
when two frequencies nearly coincide. Lean checks the rearrangement and
bounds the KL polynomial truncation error by 2|u|⁷ for |u| ≤ 0.001,
where u is the relative mass difference. Optional per-bin smoothing applies
to normalized histograms. These are real-arithmetic bounds.

For Gaussian MMD², Lean proves that every finite signed quadratic form of
the Gaussian Gram matrix is nonnegative. The proof expands the exponential
in powers of the dot product. Applying it to positive source and negative
target weights covers different sample sizes, feature dimensions and
repeated observations. Lean also proves common-offset invariance and the
kernel derivative along any differentiable feature path. The production BEAR
and transfer losses share the Gaussian implementation. Independent tests
check its value and every coordinate gradient against pairwise decimal
arithmetic. A separate case exercises a finite derivative after the kernel
itself underflows. [Gretton et al., MMD](https://jmlr.org/papers/v13/gretton12a.html)

CORAL uses the squared Frobenius distance between the two sample covariance
matrices divided by 4d² for d features. Lean checks its directional derivative
and the equivalence of centered covariance to a sum over pairwise differences.
Each domain must have at least two observations. The CPU tests check 672
two-bin MMD populations and the 128 of those that admit sample covariance.
Accelerator arithmetic and floating-point accuracy over all input magnitudes
remain open. [Sun and Saenko, Deep CORAL](https://arxiv.org/abs/1607.01719)

### Decision Transformer conditioning

Discounting advances at each conversation turn, including tool and user turns.
The trainer then selects the return at each assistant action. For precomputed
embeddings with conversation turns, both one row per action and one row per turn
are supported. The latter are projected at assistant positions. For embedding-only
trajectories, the provided state, action and return rows must align directly.
Renormalizing rewards refreshes cached returns at their recorded discount.
When a cache has no recorded discount, the dataset configuration supplies it.

Lean proves the reverse scan equals the forward discounted reward sum at each
selected turn. Splitting the conversation preserves the tail contribution with
the discount raised to the number of intervening turns. Integer-grid lower and
upper bounds enclose each exact update. At any finite horizon, agreement of
both endpoints under monotone rounding certifies the rounded forward sum.
Lean also proves that deterministic nearest-value rounding is monotone.
The implementation divides Python integers with `/` to round grid endpoints
to binary64 and refines the grid when the endpoints round differently.
Training then casts the stored return to the model's input dtype and checks
that it is finite. For a discount with denominator d and
n+1 rewards on a common grid, d raised to n suffices to represent the exact
suffix. Python compares binary64 results, including midpoint and subnormal
cases, with independent rational arithmetic.

For a freely chosen action prediction at a fixed conditioning history, squared
error is minimized by the mean observed action vector. Excess mean loss equals
the squared distance to that vector. Action and successor losses each use the
number of observed vectors as their denominator. Tests compare two actual
AdamW updates with independent unpadded evaluations. These results condition
on the logged histories and rewards. Whether a trained policy
can realize a requested future return in an environment remains open.
[Decision Transformer](https://arxiv.org/abs/2106.01345)

Training currently takes the first configured context window of each logged
trajectory. Inference takes the most recent window with absolute assistant
timesteps. Past actions use the same serialized preceding histories as
training. A caller who omits past return values supplies zero historical
conditioning through that API. New checkpoints preserve the conversation
embedder. Restoring text predictions from a legacy checkpoint requires the
original embedder weights separately.

### Logged-action IQL policy extraction

The public offline language-model trainer uses advantage-weighted behavior cloning
with frozen critics. It averages the causal negative log probabilities inside each
assistant action, multiplies by its capped IQL weight, and averages over actions.
Behavior log probabilities are unnecessary for this objective. The implementation
scores only supervised positions and retains double-precision differences in
critic values and policy logits. Float16 and bfloat16 policies accumulate
microbatch gradients on float32 AdamW weights. A checkpoint includes those
working weights, moments, pending gradients and the number of accepted actions.
[IQL policy extraction](https://arxiv.org/abs/2110.06169)

With one equally weighted single-token action beginning with A and two
two-token actions beginning with B, the fitted root probability of B is 1/2.
Each long action contributes half a unit there. The Lean result applies to
independently selectable next-token distributions at visited prefixes with
positive observed mass. A shared neural network can constrain those
distributions, and finite softmax logits give positive mass to unobserved
tokens. A CPU test compares two updates of a small causal transformer against
independent teacher-forced losses and AdamW moments for every parameter. The
TLA+ model checks completed microbatches and checkpoint boundaries.
It assumes a fixed policy within the accumulation window and an atomic
optimizer step. Per-layer floating-point backpropagation and convergence
of the jointly parameterized policy remain open.

### CQL proposal sampling

Fix a state, its critic parameters and temperature $\tau>0$. For one proposal action $A$, write $X=\exp(Q(A)/\tau)$, $a=\mathbb E[X]$ and $v=\operatorname{Var}(X)$. StateSet's sampled backup uses $S_N=\tau\log(\sum_i X_i/N)$. The full-proposal reference is $S=\tau\log a$. For $N\geq 1$ pairwise independent draws with the same mean and finite variance, and an almost-sure floor $X_i\geq m>0$, Lean proves

$$
0\leq S-\mathbb E[S_N]\leq \frac{\tau v}{Nma}.
$$

The Bellman target uses the pointwise minimum of the two frozen critics. Discounting scales its bias by $\gamma$, and terminal targets have zero proposal bias. Each live critic's conservative penalty uses its own scores and adds $\tau\log N-Q(s,a_{\rm logged})$ to the normalized partition. Including the same offset in the full-proposal reference preserves the bias above. A single sampled value can fall on either side of $S$. Without a uniform floor, for $0<\delta<1$ the probability of absolute error at least $-\tau\log(1-\delta)$ is at most $v/[N(a\delta)^2]$.

For a score derivative $D$, let $b=\mathbb E[XD]/a$ be the population softmax-weighted slope and $w=\operatorname{Var}(X(D-b))$. If the proposal pairs $(X_i,D_i)$ are pairwise independent across draws and have the stated common moments, the expected sampled slope differs from $b$ by at most $(w/c+cv)/(2Nma)$ for every $c>0$. Subtracting the fixed logged-action slope preserves that difference. For continuous proposals, `population_derivative` identifies $b$ with the derivative of the population soft value when the score is locally differentiable almost surely and the derivative of its exponentiated mass has an integrable bound throughout that neighborhood.

The CPU tests enumerate every binary proposal bank of sizes 1, 2, 4 and 8. With equally likely masses 1 and 3, and respective slopes 0 and 1, the population slope is 0.75. The expected sampled slope is 0.5 for one draw and about 0.72423 for eight. Repeating a single draw eight times keeps the one-draw bias. A separate test sends a product grid for uniform actions on $[-1,1]$ through the actual clipped twin-Q target and both penalty heads. Each action is within $1/128$ of its cell midpoint. The 1-Lipschitz critics and soft partition therefore give an explicit allowance for approximating the continuous expectation. [Lean proofs](lean/CQLSampling.lean) · [CPU sampling tests](../tests/unit/test_cql_sampling_bias.py)

### Logged Bellman evaluation

Each row contributes its immediate reward and, on a continuing transition, the configured discount times the successor estimate. For normalized nonnegative empirical weights, let $b$ be the largest discount times continuing probability over all states. When $b<1$, the exact tabular update converges from every initial table. A maximum absolute Bellman residual of $e$ certifies value error at most $e/(1-b)$. The same certificate holds if the root mean squared TD error at **each state** is at most $e$. When training against a frozen table that differs from the candidate by at most $d$, the certificate becomes $(e+bd)/(1-b)$. The squared TD loss includes conditional target variance, even at the fixed point. A pooled training loss can conceal errors at rarely visited states. A CPU test attains the lagged bound exactly on a self-loop with discount ½, fitted value 6, target value 8 and true value 2.

A pooled mean squared TD loss can also certify value error when all states are represented. Let $L$ bound the exact weighted mean squared loss on the published labels and let $N$ be a lower bound on their average within-state variance, using the same sampling weights. If each state has empirical frequency at least $p>0$, the mean signed target-rounding error at each state has magnitude at most $\delta$, and the frozen and fitted tables differ by at most $d$, then
$$
\lVert u-V\rVert_\infty \le \frac{\sqrt{(L-N)/p}+\delta+bd}{1-b}.
$$
Setting $N=0$ uses the loss alone. With one rare self-loop in four rows, the native float32 trainer attains value error 5 with $L=1/4$, $p=1/4$, $N=0$, $\delta=1/2$, $d=2$ and $b=1/2$. In a second example the critic equals the exact fixed point. Its targets round in opposite directions, its pooled loss and target variance both equal 1, and the certificate gives zero error. These are guarantees for the empirical transition table. A minibatch that misses a state supplies no positive coverage bound for that state. The reported floating-point loss can round down. For example, a binary64 prediction of $2^{-600}$ against a zero terminal target reports zero squared loss, although its exact loss is $2^{-1200}$. An application using the reported loss must include an upper bound on this numerical error in $L$.

The repository includes an offline audit for a complete finite table. Pass the fitted and frozen values from the same update, and the immediate reward, continuing successor (or `None` for termination), and actual published target for each row. Each occurrence has equal weight. If state $s$ has $c_s$ rows and signed residual sum $S_s$, then $L-N=(\sum_s S_s^2/c_s)/(\sum_s c_s)$. The audit checks the sign before squaring the tolerance inequality. All arithmetic uses exact fractions, including for float inputs. A `True` result bounds the worst error across the declared states of the empirical model. The check requires every declared state to appear. The pooled criterion requires $b<1$.

```python
from fractions import Fraction
from formal.critic_certificate import CriticRow, audit_logged_critic

prediction = 2.0**-600
report = audit_logged_critic([prediction], [0], [CriticRow(0, None, 0, 0)], discount=1)
assert report.loss == Fraction(1, 2**1200)
assert report.certifies(prediction)
assert not report.certifies(0)
```

At discount one, a state can continue with certainty for a step even when all episodes eventually terminate. Let $P$ be the matrix of discounted continuing transition frequencies, $T$ the empirical Bellman operator, and $b_h$ the largest row sum of $P^h$. Whenever $b_h<1$, Lean proves that $T$ has a unique fixed point $V$ and that $\lVert u-V\rVert_\infty \le \lVert T^h u-u\rVert_\infty/(1-b_h)$. At unit one-step survival the audit uses reverse reachability to find the first common horizon and computes both quantities with exact fractions. An explicit `horizon=h` measures the requested block. A closed continuing class prevents certification. In the branching test, state 0 always moves to 1 for reward 1. State 1 earns 2 and either stops or returns to 0 with equal probability. The true values are $(6,5)$. Starting from zero, $b_2=1/2$ and the two-step residual is 3, so the certificate attains the exact worst error of 6.

The public pretraining method can capture the sampled rows and the predictions and targets from the actual auxiliary value loss. With observation enabled, the trainer also evaluates the frozen model on the source states before either auxiliary optimizer moves. It delivers an immutable CPU snapshot after both heads and the target network have completed their updates. The audit concerns the sampled transition table for that single update. To reproduce the two-state CPU check, supply a dataset whose complete batch contains the states represented by the feature vectors `(0.0,)` and `(1.0,)`.

```python
from formal.critic_certificate import audit_critic_snapshot

batches = []
await trainer.pretrain_value_functions(dataset, num_steps=1, on_critic_batch=batches.append)
report = audit_critic_snapshot(batches[0], [(0.0,), (1.0,)])
```

The four-row regression fixture has exact empirical values $2/5$ and $-2/5$. The first fitted table has worst error $0.90$ and the audit certifies a tolerance of $1.85$. The CPU tests compare scalar loss derivatives, separate clipping factors, AdamW moments and new parameters for two successive updates. They exercise linear and two-layer ReLU critics, including a step that changes a hidden unit's activation status. Lean derives the ReLU Jacobian away from zero and bounds finite prediction changes across zero. Restoring the first completed IQL pretraining step reproduces the next sampled batch, targets, optimizer state and weights bit for bit. Earlier critic-only checkpoints stored the inner IQL weights but the loader skipped them. The loader now restores those weights, both inner optimizers and the sampler, and recovers the step count from Adam for older files. The AdamW history proof uses the clipped gradient as input. Native kernel rounding and general neural-network training trajectories remain open.

The inner IQL and CQL updates also have independent CPU checks. For IQL, two successive updates of affine heads exercise positive, negative and zero expectile residuals, twin-Q Bellman labels, joint Q clipping, both Adam optimizers and publication of the frozen value head. For CQL, the tests record the actual actions evaluated by both critics at every state. Scalar softmax weights give the penalty derivatives, which combine with the detached Bellman gradients. Three consecutive updates cover separate critic clipping, fixed and adaptive multipliers, logged and sampled successor actions, and the target update interval. Lean proves both batch derivatives for supplied network sensitivities.

The standalone checkpoint now saves the learner configuration, networks, Adam state, multiplier when used, step count and sampling streams. The outer pretrainer also saves its inner CQL learner. CPU tests restore after a completed update and reproduce two more updates, including the sampled actions, moments and parameters. A legacy standalone weights file starts fresh optimizers. A legacy outer CQL checkpoint retains the auxiliary heads and requires fresh inner pretraining. [CPU critic and replay tests](../tests/unit/test_offline_critic_steps.py)

`approximate_sweeps` bounds the error of an inexact sequence by the initial error times $b^t$ plus the geometrically weighted update errors. `target_error` separates input errors from multiply-add and output-conversion errors. Applying the sweep bound to a neural critic requires a bound on each full-table update. The native tests compare complete empirical sweeps with rational fixed points and exercise the actual CQL, IQL, BCQ, BEAR and auxiliary critic entry points. StateSet previously applied a discount of 0.99 in the outer trainer even with `gamma=0`. A successor value of 2 then produced an erroneous loss of 3.9204 at zero reward.

Standalone and language-model pretraining share the terminal-column reader. Explicit termination takes precedence over a combined episode-end signal at a time cutoff. The input must include the successor observation at the cutoff for a continuing backup. CQL draws successor proposals only for live transitions. It draws conservative-penalty proposals for every sampled source state. Discounted additions use a binary64 multiply-add before conversion to the output dtype. The CPU and CUDA loaders retain binary64 reward labels when narrowing changes their value. MPS native critic tensors use float32. The CPU tests include cancellation and a discount smaller than the float32 range. Real-arithmetic graph checks and these tests leave device-specific rounding and convergence of trained neural critics open. [Gymnasium time limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)

For an immediate reward of $1+2^{-23}$, successor value $2^{-24}$ and discount $1-2^{-53}$, the exact target lies just below a float32 midpoint. The earlier CPU implementation returned $1+2^{-22}$. The Lean theorem `double_round_witness` computes the two distinct grid decisions for all three narrow critic precisions. The corrected implementation returns the nearest float32, $1+2^{-23}$. It compares a candidate endpoint and a reflected probe using the residual of the original multiply-add at their midpoint. A compensated subtraction retains a small reward when the product alone lies on a midpoint. The correction also handles CPU half-precision casts that round through float32. The reverse-mode and forward-mode derivatives continue to use the configured discount.

`BellmanRounding.round_cell` proves the two-endpoint decision for monotone working and intermediate roundings that preserve the indicated grid points, given the sign of the midpoint residual. `compensated_residual` proves exact recovery when the fused residual fits the working significand on its dyadic grid. `residual_sign` extends the sign argument to larger residuals when the tail fits strictly inside the exact grid range. CPU tests check both sides and the exact tie in every float16 and bfloat16 rounding cell, including signed zero and overflow. A separate rational oracle rounds directly to the destination precision and covers mixed dtypes and tiny discounts. Deriving all the grid and exponent hypotheses for a particular PyTorch backend remains open.

With a float64 reward of $-2^{24}$ and an integer successor of $2^{24}+1$, the earlier terminal-mask operation rounded the successor to float32 and produced 0 at discount 1. Converting a reward of $2^{53}+1$ to binary64 before adding $-2^{53}$ also produced 0. Both native targets now return 1. Batches with an integer operand use CPU integer arithmetic to align the reward with the discounted successor and round once to the output format. The public target preserves the previous output dtype. Offline learners keep large integer label columns and publish double-precision targets on devices supporting binary64. The model covers finite inputs. Conversion to and from backend tensors and accelerator execution remain separate validation obligations.

For integral reward sequences, conversation import and decision-level accumulation preserve the original integers. Discounted suffixes round to binary64 after accumulation. Exact interpolated quantiles preserve the effect of small reward differences after a large common offset. A reward column containing both $2^{63}+1$ and $-0.5$ keeps each value in its original precision through sampling. The first row bootstrapped with $-2^{63}$ at discount 1 yields 1. All live successors are evaluated together in the original batch order. Version 3 offline checkpoints bind the values of mixed reward columns as well as integral columns. Resume reconstructs the floating labels visible to version 1 and 2 checkpoints before binding the exact labels supplied for the resumed run.

Even correctly rounded targets can have a stationary value with zero measured TD loss and positive true value error. For a continuing self-loop with discount ½, prediction 1 and reward $½+2^{-24}$, the float32 target rounds back to 1. The exact fixed point is $1+2^{-23}$. `stationary_tie` proves that the resulting value error can reach half a rounding interval divided by $1-\gamma$. Native tests attain that bound in all four critic dtypes.

The Decision Transformer branch of offline pretraining uses ordered episodes and samples its auxiliary TD rows separately. Without supplied return labels, each complete episode is discounted at the configured rate per decision. `DecisionReturns.cutoff_return` gives the omitted continuation as $\gamma^L$ times its return after $L$ recorded decisions. At positive discount, the recorded prefix alone cannot distinguish two futures with different returns. The adapter accepts observed `returns_to_go` for such episodes and keeps the native context-length convention. CPU tests exercise the real sequence trainer, independent episode resets and checkpoint migration from the older float32-only reward fingerprint.

### Sampled offline Bellman targets

BCQ samples actions from its current behavior VAE and applies the **target** perturbation network. BEAR draws from its **target** actor. Both then score each continuing successor's own proposals with the target critics. BCQ interpolates the smaller and larger Q values with the configured weight on the smaller. StateSet BEAR uses the smaller Q value. The maximum is taken separately for each successor. Terminal transitions and zero discount return the immediate reward without drawing proposals. Each target policy and its two target critics update on the existing learner schedule, after completed steps 1, 1 + frequency, and so on. [BCQ, Algorithm 1](https://proceedings.mlr.press/v97/fujimoto19a/fujimoto19a.pdf) · [BEAR reference target](https://github.com/aviralkumar2907/BEAR/blob/master/algos.py#L369-L375)

For arbitrary finite histories, `TargetPolicy.initial_difference` proves that initial target differences shrink by a factor of `(1 − τ)` at each scheduled update when the live snapshots agree. Each coordinate stays inside the historical bounds. If the two critic scores for each paired proposal change by at most ε, `sampled_backup_bound` bounds the Bellman target change by γε, even if a different proposal wins.

For independent draws from a fixed finite distribution, `independent_proposal_bound` bounds the expected shortfall from the top score by the initial score range times the probability of missing that score on all draws. A top score of zero proposal mass gives a trivial bound. For continuous proposals, `sampled_max_miss` proves the threshold probability on an arbitrary probability space. Conditional on fixed policies and critics, if independent proposals each fall below a threshold with probability p, the sampled maximum falls below it with probability pᴷ for K draws. The threshold may sit below the optimum. A guarantee on near-optimal actions therefore depends on how much probability the policy assigns to them.

The CPU checks recover individual targets from the critic loss gradient, compare weighted finite proposal sequences with the sampling bound, and replay two updates across a checkpoint under equal draws, batch, and architecture.

New standalone checkpoints save the policy target, the two critic targets, and Adam state, including the optimized BEAR dual coordinate. Legacy saves initialize the policy target from the saved live weights and start fresh Adam moments. The caller controls the random stream. With a scalar standard-normal score, the actual BCQ VAE sampler and BEAR Gaussian actor also pass six seeded frequency checks at 1, 3 and 7 draws. The interpolation tests cover finite endpoints, opposite signs, ties and subnormals in four dtypes. The exact-real proofs omit intermediate floating-point rounding.

StateSet's BEAR actor uses the single logged action for its MMD constraint. With the default configuration it optimizes Q alone for the first 40,000 updates, then adds the MMD term and enables the dual update. The authors' warm start optimizes the support-matching term first. The tests check StateSet's configured phases. Reproducing the authors' warm start or their learned behavior distribution would change the training objective. [BEAR authors' warm-start objective](https://github.com/aviralkumar2907/BEAR/blob/master/algos.py#L421-L437)

### Distributed rounding

Dividing the smallest positive float32 or float64 coordinate by two on each
rank previously turned two identical gradients into a zero update. The native
trainer now sums float16 inputs in float32 and bfloat16 or float32 inputs in
float64. For float64 inputs it uses per-coordinate powers of two and two
additional extrema collectives per bank to keep the sum in range. It combines
the separately weighted policy and entropy contributions before writing the
parameter gradient. Wider collectives increase communication and temporary
memory. Each numeric collective is bucketed at 25 MiB. The local MPS path uses
float32 as its widest real dtype. [PyTorch MPS dtype support](https://github.com/pytorch/docs/blob/site/main/notes/mps.md#double-precision-types)

Two- and three-rank CPU Gloo tests compare all 810 vectors of signed grid
coordinates from −4 to 4 with exact rational arithmetic in four dtypes. They
also exercise a restored accumulation window, extreme magnitudes, and an SGD
update. Lean also proves that the headroom bound holds for every exact partial
sum and that projecting into bounds enclosing the exact answer cannot increase
its absolute error. The Lean grid proofs assume gradual underflow, adequate exponent range
and the indicated intermediate precision. Local backward rounding, arbitrary
exponent mixtures, and other hardware reduction kernels remain open.

Native GRPO clips after AMP unscaling and population correction. Explicit native
L2 clipping in the PPO, GSPO, GEPO, GRPO, CQL, IQL, BCQ, BEAR and reward-model
updates uses the same calculation. The group selects the coordinates in the
norm. PPO clips the model and value head separately. CQL clips each critic
separately, while IQL, BCQ and BEAR clip the twin critics together. For two
groups with the same cap, the joint real norm can exceed that cap and is at
most `sqrt(2) * cap`. In a PPO regression with cap one, SGD applies the policy
direction `(3/5, −4/5)` and critic direction `(5/13, 12/13)` to float32
gradients of order `1e20`. The default AdamW path records both analytic first
and second moments and takes the corresponding bias-corrected step. Both
learning-rate schedules advance once. Native tests recover the separate CQL
and joint IQL, BCQ and BEAR critic updates from these trainers' parameters.
The GSPO sequence and token trainers advance existing Adam moments when a
finite embedding gradient would overflow a float32 norm.

The coefficient is `min(1, cap / (norm + 1e-6))`. Reductions widen before
squaring and operate on views of at most 1,048,576 real coordinates. Lean checks
the significand and exponent headroom for binary32 squares, the invariance of
exact energy under partition, and a conditional error bound for a reduction
tree of any bounded depth. For the modeled nearest-even binary addition,
the proof derives the relative error bound of `2⁻⁵³` on a common dyadic grid
with sufficient exponent range. Binary64 inputs use power-of-two scaling. A tiny
coefficient can be applied in multiple normal factors before the destination
cast. Tests compare 1,600 twelve-coordinate vectors to a 110-digit decimal
reference, and compare dense, strided float32 gradients to analytic norms under
different partitions.

`stored_clip` bounds the stored norm by the cap plus two terms. Underestimating
the norm by more than epsilon contributes the excess times the coefficient.
Coordinate errors contribute their joint L2 norm. For cap one and gradient
`(1, 2)`, Lean places the real result strictly inside the binary16 rounding
cells centered at `(229/512, 229/256)`. CPU binary16 and bfloat16 store those
coordinates, giving a norm of about 1.00011634. Connecting a particular accelerator kernel to the modeled additions,
identifying its reduction depth and verifying optimizer failure recovery remain
open.

### AdamW moments, precision and resume

A missing gradient skips decay, moment updates and the parameter's step
counter. A provided zero gradient ages both moments and can still move the
parameter. For first- and second-moment weights `pᵢ` and `qᵢ`, Lean proves
`m̂² ≤ (Σ pᵢ²/qᵢ) v̂` when every second-moment weight is positive. This bounds
the adaptive displacement from the decayed weight by
`lr × sqrt(Σ pᵢ²/qᵢ)`. With equal positive betas, the sum is one. With
betas `(0, 1/2)`, epsilon `0.1` and gradients `0, 1`, the second
step exceeds one learning rate even though each gradient is at most one.
With betas `(1/2, 0)` and successive gradients `g, 0`, the second adaptive
displacement is `lr × g / (3 × epsilon)`.
`entire_weight` gives the final real-arithmetic parameter for an arbitrary
sequence of rates and optional gradients. Each earlier adaptive displacement
is multiplied by all later active decay factors. A two-parameter CPU check
compares a stepped scheduler against that expansion after each update.

The native single- and multi-turn GRPO, GSPO, GSPO-token, GEPO, DAPO, VAPO and
PPO trainers keep working weights and AdamW moments in float32 for FP16 and
BF16 learner parameters. Clipped gradients reach the optimizer at working
precision. GRPO widens before AMP unscaling. GSPO and single-turn GRPO use
the configured moment decays. Runtime checkpoints retain hidden progress
between two adjacent published weights. Legacy moments widen on load and the
published parameter supplies the starting weight. An initially frozen layer
allocates its float32 weight storage when it needs a gradient or a saved state.
For each active 16-bit parameter element, the float32 working copy and wider
moment pair add eight bytes of persistent storage over the 16-bit AdamW path.

CPU tests cover all 64 modeled histories, the default PPO optimizer and both
PPO heads across current and legacy checkpoint loads, all listed trainer paths,
and an overflow skip followed by a successful GRPO update. The TLA+ model
uses eight fractional bits for its square root and checks its rounded result
against exact inequalities. Lean proves the real history bound for arbitrary
lengths and the rounding threshold for a sequence of sub-cell moves.
The nearest-even grid proof also checks that the default epsilon of `1e-8`
rounds to zero in binary16 and to a positive value in binary32.
Full IEEE-754 error bounds for AdamW's moments, parameter updates and
accelerator kernels remain open.

### Remote training reductions

StateSet's synchronous River driver normalizes before splitting a logical
update into requests. Its token denominator counts positions with nonzero
advantage. The River 0.11.0 native SDK counts all eligible generated tokens,
including those in a scored trajectory with zero advantage. For a fixed
batch with three contributing and three zero-credit tokens, the native
token-mean gradient is half the driver's. Their sequence modes also have
different units. The driver averages assistant spans, while the native SDK
averages trajectories. For turns of lengths one and three within the same
trajectory, the driver gives the first turn half of the weight and the SDK
gives it a quarter. The regression suite checks both definitions against the
installed SDK. [River batch weighting](https://docs.river.ai/guides/rl-primitives/)

Tinker accepts explicit target IDs, behavior log probabilities and signed
advantages. Its IS objective sums per-token terms. Tests exercise the real
SDK's tensor serialization and StateSet's optimizer-call boundaries with a
local gradient recorder. Coverage of the hosted worker depends on its
implementation of the published loss. [Tinker loss inputs](https://tinker-docs.thinkingmachines.ai/tinker/losses/)

### Experimental DPO

DPO scores the same retained prompt and completion tokens under the learner and
its frozen reference. A common completion prefix cancels. At the first differing
action the two log probabilities share a normalizer, so their difference is the
difference of the selected logits. Later actions retain their own conditional
normalizers. The implementation checks support before cancellation and sums signed
terms in float64. When cancellation could materially affect the scaled margin,
it compensates the sum. If an intermediate compensated sum overflows, it scales
the exact sum of the represented terms before rounding. Large vocabulary buffers
are scored in blocks and recomputed during backward. Float16 and bfloat16 model
parameters use float32 AdamW weights and moments, with each committed update
copied back to the model dtype. The optimizer checkpoint preserves those float32
weights. Restoring an older checkpoint initializes them from the loaded model
weights. To resume training, load the learner, its frozen reference and the optimizer.
A failed pair clears gradients accumulated for the batch.

The proofs condition on a fixed prompt and a finite response set with positive
probabilities. When positive-weight comparisons connect the response set and each
pair's preference log odds equal its reward difference, the normalized Gibbs policy
uniquely minimizes population DPO loss. A CPU counterexample with disconnected
comparisons moves the probability of one component from 2/5 to 8/11 while preserving
every observed comparison and increasing KL to the reference. The trainer scores
the provided tokens through the context cutoff. When one retained response is a
prefix of another, their sequence probabilities describe overlapping events unless
termination is encoded. Applying the normalized-policy result requires a specified
distribution over distinct outcomes. The proof assumes
freely chosen probabilities. Neural parameter sharing, preference noise outside
the assumed model, accelerator rounding and finite-step convergence remain open.

### Experimental A2C

The collector publishes each completed segment after it has obtained its terminal
flag or cutoff value. A pending update waits for collection. A collector exception
or cancellation leaves earlier published segments available for the next update.
The trainer applies the configured `gae_lambda` to both policy advantages and
critic targets. Setting `gae_lambda=1` reproduces the previous trace-decay behavior. At zero the critic sees one-step targets. At one it sees the full
discounted segment with its cutoff bootstrap. For two or more transitions the
actor uses sample-standardized advantages, and a singleton keeps its raw advantage.
The derivation conditions on a fixed rollout and positive categorical probabilities.
The TLA+ model assumes each started operation eventually finishes or aborts. Its
atomic optimizer transition abstracts failures within the optimizer call. The
RMSprop proof uses real arithmetic. Floating-point accumulator overflow with
very large unclipped gradients and accelerator-specific kernels remain open.

### Continual GRPO replay

The reservoir proof starts with an empty buffer and assumes independent uniform
integer draws. After `n ≥ k` arrivals, every size-`k` subset occurs in exactly
`(n−k)!` draw histories. A retained entry has one replacement ticket and the
remaining tickets leave the buffer unchanged. For any specified batch of size
`b ≤ k`, exactly `choose(n−b, k−b)` reservoirs contain it. A uniform batch
from a uniform reservoir therefore covers all historical batches equally.
TLC and Python enumerate the
actual slot transitions for small streams. [Vitter, Algorithm R, §2](https://www.cs.umd.edu/~samir/498/vitter.pdf)

Balanced sampling gives one group per named task in each complete round while
entries remain. A partial round with `r` places and `t` eligible tasks includes
each task with probability `r/t`. Unlabeled groups fill remaining places after
the named pools are exhausted. Reward sampling draws with replacement using
exact integer intervals proportional to `max(reward, 0) + 1e-3`, where the
floor is the stored binary64 constant. Replay rounding uses a separately seeded
generator so a rounding ticket cannot also choose the row. Version 2 manager
checkpoints restore both streams. Loading version 1 deterministically derives
a separate rounding stream from the saved state, which changes subsequent
samples compared with the historical implementation.

Python checks every RNG branch on small balanced pools, both endpoints of
reward intervals with subnormal and near-maximum rewards, and replay rounding
below the resolution of a 53-bit uniform float. The trainer samples from past
groups before storing the current batch. The sampling laws condition on the
contents of the replay buffer.

### Replay selection and batch averaging

On a fixed buffer with positive priorities, reward sampling changes the
expected gradient coordinate by the population covariance of priority and
gradient divided by mean priority. For a uniform reservoir of capacity `K`
from `N` groups, suppose one group has priority `w` and each other group
has priority `u`. That group's marginal probability on one reward draw is
`(K/N) w/[w+(K−1)u]`. Its chance to survive storage and its chance to win the
subsequent draw both enter the calculation.

With one entry per task in a complete balanced round, each task contributes
its mean gradient with equal weight. Uniform entry sampling instead weights
those means by task size. The difference equals minus the covariance of task
size and task mean divided by the mean task size. On six entries split across
tasks of sizes 1, 1 and 4, a native example gives a one-group balanced gradient
of −1/6, a uniform-entry gradient of +1/6, and a most-recent gradient of +1/2.
The within-task counting theorem covers any uniform quota without replacement.

Once the replay warm-up threshold is met, the native `replay_ratio` sets an expected number of replayed groups per fresh
group before the eligible-population cap. For a single-batch update with one fresh
group, ratio 0.5 and equal group sizes, half the batches include one replay.
Replay therefore gets one quarter of the expected update weight. With two
such microbatches accumulated into one optimizer step and independent replay
counts, its expected weight is 7/24. A CPU test checks all four count histories
against the committed SGD parameters. More generally, if fresh mass is
`F`, replay count is `m` or `m+1` with upper probability `t`, and groups have
equal mass, normalizing the expected counts overstates the expected replay
share by `F t(1−t)/[(F+m)(F+m+1)(F+m+t)]`. Here `F` is measured in groups.
For any finite law of total replay mass, the difference equals the expectation
of `F(M−E[M])²/[(F+E[M])²(F+M)]`. For unequal group sizes, each realized
batch uses its actual supervised-token
or trajectory denominator. These gradient formulas assume fixed policy
parameters and advantages computed within each group.

In a three-group example the two lower-priority gradients are +1/2 and the
higher-priority gradient is −1/2. Their priorities are 1, 1 and 5. The uniform
historical mean is +1/6. With capacity two the heavier group wins a reward
draw with probability 5/9, giving expected replay gradient −1/18. Mixing
that draw into one zero-gradient fresh group at ratio 0.5 yields **−1/72**.
Python runs the native reservoir, count rounding, weighted selection and
autograd over every branch. Doubling the heavy group's sequence length checks
the difference between the token and trajectory reductions. Reciprocal
selection weighting recovers the uniform-buffer mean in the mathematical
reference.

### Replayed group gradients

At a fixed prompt let `x` be a scored response, `y` the other responses in its
group, and `f(x,y)` one score-gradient coordinate times its stopped group
advantage. Under full behavior support and independent sampling of the scored
response and peers, an unclipped importance ratio correcting the
response from its behavior law to the current law `p` leaves the peer law
unchanged. The exact excess over fresh sampling is
`Σₓ p(x) Σᵧ [old(y) − fresh(y)] f(x,y)`. If the range over peers has radius at
most `R` for every response, its magnitude is at most `2 R TV(old, fresh)`.
An independent leave-one-out baseline cancels for a zero-mean target score.
With a raw group mean, the fresh and row-corrected replay gradients both
carry the factor `(G−1)/G`. Native autograd checks both estimators across all
27 three-response groups in the example below.
The general joint-law identity specifies the missing mass under limited support
and applies to reward- or task-selected groups when their actual draw law is
known. Group-level weighting is a mathematical reference in this check.
[Related off-policy GRPO analysis](https://arxiv.org/abs/2505.22257)

Each group in the example contains two one-token responses drawn from three
actions with rewards 0, 1 and 2. It uses a uniform learner, behavior probabilities `(1/2, 1/4, 1/4)`, population normalization,
zero normalization epsilon and zero regularization. The expected ascent
coordinate of the middle logit is **0** on fresh groups, **1/16** on unweighted
replay, **1/36** with an unclipped per-response ratio, and **−1/18** with the
native 20% clip. These values follow by exact finite sums in Lean. Python
enumerates every pair through native GRPO autograd and checks all three
coordinates. For any near-gap and far-gap normalized magnitudes `a, b`, the
row-corrected coordinate is `(2a−b)/36` and the clipped coordinate is
`−a/18`. The tests include the default epsilon and both deviation cutoffs.
They also follow real SGD updates for both online trainers when a
fresh group and a replayed group come from different policies, including
a replay group restored from a JSON checkpoint.

With `old_logprobs_source="sampler"` native token GRPO uses the group's recorded
sampler scores across inner updates. Explicit low-level likelihood tensors
retain precedence. Native clipping, token-wise ratios for multi-token responses,
and priority selection all affect which replay objective the learner follows.
The `token_bias` theorem separates the error from historical prefixes from the
error due to later tokens and peers. A two-token native example keeps the second
action distribution fixed and changes how often the first token reaches its
prefix. Across all 16 response pairs, the expected second-token ascent
coordinate moves from 1/16 to 3/32 despite exact conditional sampler scores.
Performance improvement over a multi-step training run remains open.

### PPO stopping on sampled trajectories

`StoppedPolicy.common_support` proves that on any finite tree of categorical
actions with common support, the expected **sum** of per-action k3 values
equals KL(behavior‖current) for complete paths. EOS and the token horizon are
leaves. If the current policy assigns mass to actions absent from behavior,
`stopped_path_kl` gives the exact omitted term. At each visited prefix it is
the current mass on those actions, weighted by the behavior probability of
reaching that prefix. This assumes the current policy supports every behavior
action with positive mass.

PPO compares the cap with a **mean per sampled token**. Suppose the root
policies agree, each response continues with probability `q`, and the second
action has conditional KL `D`. For `B` independent responses of which `K`
continue, the conditional expectation of the reported drift is
`D K/(B+K)`. The expected summed drift per response is `qD`. Replacing the
random token count with its population mean gives `qD/(1+q)`, which
overstates the expected reported drift by exactly the nonnegative remainder
in `token_mean_gap`. With one response and `q=¼`, the three values are
`D/8`, `D/4` and `D/5`, respectively.

All responses end at the root with probability `(1−q)^B`. On this event
the measured drift is zero. `missed_drift` constructs strictly supported
candidate policies whose true path KL exceeds any prescribed positive cap
and whose expected binary terminal reward drops by at least `q/4`. For three
responses at `q=¼`, the probability of seeing only the unchanged root is
27/64. This limits what batch admission establishes about unsampled actions.

Online PPO records the selected action's log probability under the rounded
weights passed to the sampler. A batch rescore checks those same normalized
weights. The learner retains the dominant action's log probability and
gradient through the complementary mass. `SamplerReplay.pivot_score` and
`pivot_gradient` establish that score and its directional derivative for
every finite nonempty vocabulary, including a tie at the maximum. The near-zero k3 branch uses the
degree-six expansion bounded in `RegularizedBatch`. For any realized batch,
`measured_mean` bounds its real-arithmetic mean error by `cutoff⁷/4410` at
the native cutoff of 0.001. The reduced reference
penalty applies token and row weights before evaluating large exponentials.
Forward and backward FX equations check both expressions in real arithmetic.
CPU regressions include a subnormal sampling weight and a raw FP64 exponential
whose weighted contribution still fits. The sampling law here uses the
multinomial operator's weight contract. [PyTorch multinomial](https://docs.pytorch.org/docs/stable/generated/torch.multinomial.html)

### Trajectory scores and Fisher information

For each accepted recorded trajectory, the native trainer evaluates all
response tokens in their saved causal contexts. It computes one negative
mean log likelihood across those tokens, squares its gradient coordinate
by coordinate, and gives the trajectory one vote. If token scores on a
coordinate are +½ and −½, their squared mean is zero. Averaging the two
squares would give ¼. Lean proves that the difference is exactly the
within-trajectory variance for any positive token count.

Lean also proves that the second derivative of KL at the anchor equals the
policy expectation of squared scores for a positive, twice-differentiable,
normalized finite policy along a parameter coordinate. With a Bernoulli
action of probability ¼, this curvature is 3/16. One observed action yields
an estimate of 1/16 or 9/16. For a fixed response length with zero cross-token
score moments, the expected squared trajectory mean equals the sum of token
information entries divided by length squared. The logged-data estimator
makes no assumption that the behavior policy matches the anchor. [Kunstner, Balles and Hennig](https://arxiv.org/abs/1905.12558)

Keeping only the diagonal also changes the geometry. For two logits with
probabilities `p` and `1−p` and respective perturbations `u` and `v`, the
full Fisher quadratic is `p(1−p)(u−v)²`. The diagonal approximation changes
it by `2p(1−p)uv`. A native test shifts both logits by one at an exactly
balanced population. The action probabilities stay fixed and KL is zero,
while the EWC penalty at strength two is ½.

A native multi-turn test obtains diagonal entries of 5/36 and zero from two
unequal trajectories. PyTorch checks all 64 ordered pairs of the TLC examples
and both Bernoulli outcomes in FP64.
Two-rank Gloo tests compare task consolidation with separate cross-entropy
forwards on the saved prefixes and an accepted-count-weighted reduction.
The penalty traces cover FP32 inputs, FP64 inputs, and an FP64 strength
whose scale factors can shrink the displacement. Exact-rational Python
references check nine extreme-scale values and gradients, including a
zero Fisher entry across opposite-sign parameters near the FP64 limit
and a zero displacement with a strength-weight product exceeding FP64.
These results characterize the regularizer. Its effect on later task
performance remains open.

### Open verification work

The theorems certify the listed mathematical properties. End-to-end
trainer refinement remains open for model scoring, stochastic sampling,
optimizer state, distributed reductions, backend implementations and
floating-point error. The remaining obligations include the following.

- The VAPO joint-gradient theorem takes frozen rollout targets and supplied
  derivatives as hypotheses. Proving the complete neural-network AdamW
  trajectory in machine arithmetic and the reliability of its reward verifier
  remains open.
- RLAIF's standalone update scores a possibly rewritten completion using
  reward-weighted log likelihood. The `rl_algorithm` field is a compatibility
  label. An on-policy PPO claim for a rewrite would require behavior
  likelihoods and a critic snapshot. Judge accuracy and how the rewrite
  distribution affects policy return remain open.
- The GSPO compatibility trainers give each prompt equal native weight.
  Their token-mean and fixed-denominator overrides reduce the full update
  batch. An independent gradient test uses unequal group sizes, an empty
  response, batch-normalized advantages and a separate row-mean KL.
  A generator that returns decoded text alone can lose its exact tokenization.
  Sequence sums also cannot reconstruct a full-vocabulary KL or entropy.
- Optional objective combinations, floating-point distributed collectives,
  reward target construction and optimizer/scheduler composition require
  further verification. The gradient-bank theorem assumes a common scale
  across ranks. The exploration results establish probability and bonus
  properties; the controller also has a heuristic exploration gate.
- CQL sampling bounds assume pairwise independent proposals at a fixed state and
  frozen critic. Dependence across optimizer steps and the effect of the adaptive
  multiplier on training remain open. The IQL optimum is for a freely chosen
  scalar at a fixed state. Convergence
  of a shared neural critic and guarantees on its deployed policy return
  require additional assumptions on data coverage and function approximation.
- Provider-submitted objectives depend on the provider's scoring and backward
  implementations. Lean proves the client-side alignment and scaling properties above. A
  remote kernel trace or service implementation is still needed to verify the
  hosted arithmetic and optimizer. Text-only legacy River trajectories rely
  on caller-supplied scores matching the tokenization.
- The categorical gradient, baseline and change-of-measure theorems use
  normalized distributions and the stated support assumptions. Top-p filtering
  can remove actions with positive target mass. `truncated_change_measure` gives the exact omitted target contribution. GEPO's empirical denominator
  remains well-defined on retained samples. Legacy custom GEPO generators
  that omit sampling scores use a frozen raw-model rescore.

## Existing runtime and reward models

| Priority | Artifact | Python implementation | Checked contract |
| --- | --- | --- | --- |
| 1 | `tla/RolloutControl.tla` | `training/distributed_rollouts.py`, `training/async_rollouts.py` | Queue capacity, unique admitted IDs, accepted-counter equality, bounded lag, seen-ID admission, artifact availability, lease fencing at admission, and rejection of bad hashes. |
| 2 | `tla/AsyncRuntime.tla` | `training/async_runtime.py` | Initial publication precedes worker start; each later version is published before visibility; updates are bounded. A pending worker fault cannot produce a successful result, including after the final update. A separate fair, failure-free configuration checks eventual completion. Python regressions cover failed publication and worker failure during learning. |
| 3 | `lean/Objective.lean` | `training/objectives.py`, `training/rl_losses.py` | Log-ratio clipping bounds and idempotence, clipped-surrogate behavior including zero advantage, zero-mask behavior, and exact cross-multiplied group centering. |
| 4 | `tla/AutoResearch.tla`, `tla/CheckpointSwap.tla`, `tla/ResearchCommit.tla` | `training/auto_research/experiment_loop.py`, `experiment_tracker.py`, `checkpoint_manager.py` | Only evaluated improvements become best in either maximize or minimize mode, crashed records cannot become best, interrupted directory swaps recover, and provisional checkpoints reconcile with the experiment log after restart. |
| 5 | `lean/Reward.lean` | `rewards/multi_objective_reward.py`, `core/trajectory.py` | Final score clamp and idempotence, failed/zero-weight component behavior, a weighted-sum bound for normalized, bounded components, and a weighted-average bound including zero total weight. |

The research model ranges over bounded integer scores, including negative
values. The Python record now
rejects NaN and infinite objective values, the loop records such evaluations
as crashes before checkpoint promotion, and replay skips legacy invalid
records. This makes the model's finite-score assumption explicit at the
implementation boundary.

## Rollout counterexamples resolved

The first model revision found `Register; BeginSubmit; Register; FinishSubmit`:
the old generation's submission could complete after replacement. The
coordinator now invokes a synchronous admission check immediately before its
queue append. `FencingSafety` is included in the passing TLC configuration,
and a Python regression test holds a submission behind queue backpressure
while the worker is replaced.

The first model revision also found `Register; Publish; BeginSubmit;
FinishSubmit` with artifact capacity one. The old artifact was pruned and the
hash comparison was skipped. Strict artifact submissions now reject an
unavailable descriptor. `ArtifactSafety` is included in the passing TLC
configuration and has a Python regression test.

`tests/unit/test_formal_rollout_refinement.py` runs every four-operation trace
over registration, publication, heartbeat, submission, and consumption in a
small Python instance. It checks the queue, policy-lag, artifact, and counter
invariants after every step. This found a timeout boundary where the queue
admitted a rollout but the control plane reported a timeout and omitted its
accepted counter. The coordinator now resolves the child operation after a
timeout cancellation before returning either its committed result or a
timeout. The trace suite is a refinement check for these bounded operations,
not a proof that every Python execution refines the TLA+ model.

The model also checks that the accepted counter equals the number of admitted
rollout IDs. A Python regression simulates cancellation immediately after
admission; the control-plane counter now updates in the coordinator's
synchronous admission turn, keeping queue and counter state aligned.

The TLA+ models abstract payloads, network delivery, clock values, and
optimizer state. `CheckpointSwap.tla` checks recovery when the old best
directory was moved but the new one was not installed. `ResearchCommit.tla`
checks reconciliation after a crash between checkpoint installation and
experiment log append. The JSONL append now fsyncs the file and, on first
creation on POSIX, its parent directory before the checkpoint is finalized.
These models do not establish power-loss durability of model-weight files or
every supported filesystem. `Objective.lean` and `Reward.lean`
use exact integers. Their theorems do not establish IEEE floating-point, PyTorch
autograd, or GPU-kernel behavior. `tests/unit/test_formal_numeric_refinement.py`
exhaustively compares small score, weight, failure, clipping, and group domains
against the Python implementations. Larger tensors, GPU kernels, and optimizer
updates remain outside these checks.

`AsyncRuntimeProgress.cfg` checks that a run returns success when no operation
fails and every enabled phase eventually executes. Its `Learn` step assumes a
batch becomes available and the learner finishes; `PublishUpdate` assumes the
publisher finishes. It does not claim progress if a callback hangs, a worker
fails, or the coordinator cannot supply a batch.
The safety configuration also allows a worker fault to arrive during learning,
publication, or cleanup after the last update. The runtime checks that fault
channel at commit boundaries and after stopping producers before returning.

For reward composition, Python now rejects negative or non-finite component
weights at each evaluation, snapshots normalized weights before awaiting
component callbacks, and treats a non-finite component score as a failed
component with zero contribution. The Lean weighted-sum and weighted-average
bounds still assume each successful component score lies in `[0, 1]`; the
Python composer clamps the final score but does not impose that range on every
component. The average
proof permits any nonnegative total weight, including zero.
