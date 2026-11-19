#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tla_jar="${1:-${TLA2TOOLS_JAR:-}}"
with_python="${2:-}"

if [[ -z "$tla_jar" || ! -f "$tla_jar" || $# -gt 2 ||
      ( -n "$with_python" && "$with_python" != "--with-python" ) ]]; then
    echo "Usage: formal/check.sh /path/to/tla2tools.jar [--with-python]" >&2
    exit 2
fi

tla_jar="$(realpath "$tla_jar")"
cd "$repo_root"
command -v lake >/dev/null
command -v java >/dev/null

lake build RLMath

run_tlc() {
    local model="$1"
    local config="$2"
    local output
    output="$(mktemp)"
    if ! java -XX:+UseParallelGC -cp "$tla_jar" tlc2.TLC -deadlock \
        -metadir "$(mktemp -d)" -config "$config" "$model" >"$output" 2>&1; then
        cat "$output" >&2
        return 1
    fi
    if command -v rg >/dev/null 2>&1; then
        rg 'Model checking completed|distinct states found' "$output"
    else
        grep -E 'Model checking completed|distinct states found' "$output"
    fi
}

cd "$repo_root/formal/tla"
run_tlc AdvantageScan.tla AdvantageScan.cfg
run_tlc ActorCriticSegments.tla ActorCriticSegments.cfg
run_tlc PreferenceBatch.tla PreferenceBatch.cfg
run_tlc PPOEpochs.tla PPOEpochs.cfg
run_tlc StoppedPolicy.tla StoppedPolicy.cfg
run_tlc SamplerReplay.tla SamplerReplay.cfg
run_tlc ContinualReplay.tla ContinualReplay.cfg
run_tlc ReplayGradient.tla ReplayGradient.cfg
run_tlc ReplayMixture.tla ReplayMixture.cfg
run_tlc ReplayStrata.tla ReplayStrata.cfg
run_tlc GroupAdvantages.tla GroupAdvantages.cfg
run_tlc GroupExpectation.tla GroupExpectation.cfg
run_tlc PolicyReduction.tla PolicyReduction.cfg
run_tlc RegularizedBatch.tla RegularizedBatch.cfg
run_tlc VAPOUpdate.tla VAPOUpdate.cfg
run_tlc DomainAdversarial.tla DomainAdversarial.cfg
run_tlc FeatureAlignment.tla FeatureAlignment.cfg
run_tlc PromptReduction.tla PromptReduction.cfg
run_tlc ProviderBatch.tla ProviderBatch.cfg
run_tlc GradientBanks.tla GradientBanks.cfg
run_tlc FisherInformation.tla FisherInformation.cfg
run_tlc RoundedGradientBank.tla RoundedGradientBank.cfg
run_tlc RewardTokenWeights.tla RewardTokenWeights.cfg
run_tlc JudgeScores.tla JudgeScores.cfg
run_tlc OptimizerWindow.tla OptimizerWindow.cfg
run_tlc AdamHistory.tla AdamHistory.cfg
run_tlc BayesianReward.tla BayesianReward.cfg
run_tlc GradientClipping.tla GradientClipping.cfg
run_tlc OfflineBackup.tla OfflineBackup.cfg
run_tlc LoggedBellman.tla LoggedBellman.cfg
run_tlc TransientBellman.tla TransientBellman.cfg
run_tlc PooledCritic.tla PooledCritic.cfg
run_tlc CriticMoments.tla CriticMoments.cfg
run_tlc AuxiliaryCritic.tla AuxiliaryCritic.cfg
run_tlc BellmanRounding.tla BellmanRounding.cfg
run_tlc IntegerBellman.tla IntegerBellman.cfg
run_tlc RecordedReturns.tla RecordedReturns.cfg
run_tlc LoggedExtraction.tla LoggedExtraction.cfg
run_tlc TargetPolicy.tla TargetPolicy.cfg
run_tlc OfflineEpoch.tla OfflineEpoch.cfg
run_tlc TrajectoryEpoch.tla TrajectoryEpoch.cfg
run_tlc DecisionConditioning.tla DecisionConditioning.cfg
run_tlc RolloutControl.tla RolloutControl.cfg
run_tlc AsyncRuntime.tla AsyncRuntime.cfg
run_tlc AsyncRuntime.tla AsyncRuntimeProgress.cfg
run_tlc AutoResearch.tla AutoResearch.cfg
run_tlc AutoResearch.tla AutoResearchMin.cfg
run_tlc CheckpointSwap.tla CheckpointSwap.cfg
run_tlc ResearchCommit.tla ResearchCommit.cfg

if [[ "$with_python" == "--with-python" ]]; then
    cd "$repo_root"
    python_bin="${PYTHON:-python3}"
    "$python_bin" -m formal.check_torch_graphs
    "$python_bin" -m pytest -q -o addopts='' \
        tests/integration/test_ddp_bank_rounding.py \
        tests/unit/test_partial_accumulation_mean.py \
        tests/unit/test_judge_score_refinement.py \
        tests/unit/test_formal_rollout_refinement.py \
        tests/unit/test_async_rollouts.py \
        tests/unit/test_distributed_rollouts.py \
        tests/unit/test_policy_artifacts.py \
        tests/unit/test_async_runtime.py \
        tests/unit/test_gspo_update_integrity.py \
        tests/unit/test_provider_math_refinement.py \
        tests/unit/test_offline_extraction_refinement.py \
        tests/unit/test_offline_policy_training.py \
        tests/unit/test_logged_bellman_refinement.py \
        tests/unit/test_auxiliary_critic_update.py \
        tests/unit/test_offline_critic_steps.py \
        tests/unit/test_cql_sampling_bias.py \
        tests/unit/test_offline_target_policies.py \
        tests/unit/test_preference_refinement.py \
        tests/unit/test_actor_critic_refinement.py \
        tests/unit/test_ppo_math_refinement.py \
        tests/integration/test_distributed_advantages.py \
        tests/integration/test_trainer_ratio_invariants.py \
        tests/unit/test_gepo_weight_refinement.py \
        tests/unit/test_gepo_trainer_behavioral.py \
        tests/unit/test_group_reward_refinement.py \
        tests/unit/test_replay_mixture_refinement.py \
        tests/unit/test_replay_gradient_refinement.py \
        tests/unit/test_replay_sampling_refinement.py \
        tests/unit/test_group_sampler_refinement.py \
        tests/unit/test_token_capture.py \
        tests/unit/test_gspo_exact_rollouts.py \
        tests/unit/test_muse_spark.py::test_recipient_parsing_keeps_exact_rollout_ids_and_probabilities \
        tests/unit/test_hf_generator_fallback.py \
        tests/unit/test_vllm_logprob_contract.py \
        tests/unit/test_dapo_regularized_batch.py \
        tests/unit/test_dapo_trainer_behavioral.py \
        tests/unit/test_ppo_online.py \
        tests/unit/test_ppo_safety.py \
        tests/unit/test_ppo_objective.py \
        tests/unit/test_value_function_comprehensive.py \
        tests/unit/test_bayesian_reward_refinement.py \
        tests/unit/test_new_features.py::TestBayesianRewardModel \
        tests/unit/test_adam_history.py \
        tests/unit/test_rl_math_refinement.py \
        tests/unit/test_ewc_estimation.py \
        tests/unit/test_token_group_aggregation.py \
        tests/unit/test_offline_rl_sim_to_real.py::TestSimToRealMetrics \
        tests/unit/test_feature_alignment_refinement.py \
        tests/unit/test_domain_adversarial_refinement.py \
        tests/unit/test_vapo_joint_update.py \
        tests/unit/test_vapo_trainer_behavioral.py \
        tests/unit/test_loss_computation_behavioral.py \
        tests/unit/test_objectives.py \
        tests/unit/test_objective_goldens.py \
        tests/unit/test_decision_returns.py \
        tests/unit/test_decision_transformer_alignment.py \
        tests/unit/test_objectives_rust_parity.py \
        tests/unit/test_vapo_rust_gae_parity.py \
        tests/unit/test_objectives_properties.py \
        tests/unit/test_auto_research.py \
        tests/unit/test_formal_numeric_refinement.py
fi
