//! Generalized Advantage Estimation (GAE) module
//!
//! High-performance GAE computation for policy gradient methods.

/// Compute GAE advantages from rewards and value estimates
///
/// Implements the GAE formula:
/// A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
/// where δ_t = r_t + γV(s_{t+1}) - V(s_t)
///
/// # Arguments
/// * `rewards` - Per-step rewards
/// * `values` - Value estimates (should have length = rewards.len() + 1 for bootstrap)
/// * `gamma` - Discount factor
/// * `gae_lambda` - GAE lambda parameter
pub fn compute_gae_internal(
    rewards: &[f64],
    values: &[f64],
    gamma: f64,
    gae_lambda: f64,
) -> Vec<f64> {
    compute_gae_targets(rewards, values, &[], gamma, gae_lambda).0
}

/// Compute GAE with done flags for episode boundaries.
pub fn compute_gae_with_dones(
    rewards: &[f64],
    values: &[f64],
    dones: &[bool],
    gamma: f64,
    gae_lambda: f64,
) -> Vec<f64> {
    compute_gae_targets(rewards, values, dones, gamma, gae_lambda).0
}

/// Keep the reward and bootstrap parts of the lambda return separate until
/// both the critic target and the actor advantage have been formed.
pub fn compute_gae_targets(
    rewards: &[f64],
    values: &[f64],
    dones: &[bool],
    gamma: f64,
    lambda: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = rewards.len();
    let mut advantages = vec![0.0; n];
    let mut targets = vec![0.0; n];
    let mut reward_part = 0.0;
    let weight = gamma * lambda;
    let mut value_part = values.get(n).copied().unwrap_or(0.0);
    for t in (0..n).rev() {
        let next = values.get(t + 1).copied().unwrap_or(0.0);
        let current = values.get(t).copied().unwrap_or(0.0);
        if gamma == 0.0 || dones.get(t) == Some(&true) {
            reward_part = rewards[t];
            value_part = 0.0;
        } else if lambda == 0.0 {
            reward_part = rewards[t];
            value_part = gamma * next;
        } else {
            reward_part = if weight.abs() < f64::MIN_POSITIVE {
                gamma.mul_add(lambda * reward_part, rewards[t])
            } else {
                weight.mul_add(reward_part, rewards[t])
            };
            if lambda != 1.0 {
                const TINY_SCALE: f64 = 7.458340731200207e-155; // 2^-512
                let unit = if next.abs().max(value_part.abs()) < TINY_SCALE {
                    TINY_SCALE
                } else {
                    1.0
                };
                value_part = unit * ((1.0 - lambda) * (next / unit) + lambda * (value_part / unit));
            }
            value_part *= gamma;
        }
        let target = reward_part + value_part;
        targets[t] = target;
        let (large, small) = if reward_part.abs() >= value_part.abs() {
            (reward_part, value_part)
        } else {
            (value_part, reward_part)
        };
        // FastTwoSum recovers the rounding residual of a finite sum.
        advantages[t] = if target.is_finite() {
            (target - current) + ((large - target) + small)
        } else {
            (large - current) + small
        };
    }
    (advantages, targets)
}

/// Compute returns (advantages + values) for value function training
pub fn compute_returns(advantages: &[f64], values: &[f64]) -> Vec<f64> {
    advantages
        .iter()
        .zip(values.iter())
        .map(|(a, v)| a + v)
        .collect()
}

/// Compute lambda returns directly (alternative to GAE)
pub fn compute_lambda_returns(
    rewards: &[f64],
    values: &[f64],
    gamma: f64,
    lambda: f64,
) -> Vec<f64> {
    compute_gae_targets(rewards, values, &[], gamma, lambda).1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_separate_critic_targets_and_tiny_bootstrap() {
        let tiny = f64::from_bits(1);
        let (_, targets) = compute_gae_targets(&[0.0], &[0.0, tiny], &[], 1.0, 0.5);
        assert_eq!(targets, vec![tiny]);
        let (adv, targets) = compute_gae_targets(
            &[1.0, f64::NAN],
            &[1e30, f64::NAN],
            &[true, false],
            1.0,
            0.5,
        );
        assert_eq!(adv[0], -1e30);
        assert_eq!(targets[0], 1.0);
    }

    #[test]
    fn test_gae_basic() {
        let rewards = vec![1.0, 1.0, 1.0];
        let values = vec![0.5, 0.5, 0.5, 0.0]; // Extra value for bootstrap

        let advantages = compute_gae_internal(&rewards, &values, 0.99, 0.95);

        assert_eq!(advantages.len(), 3);
        // Advantages should be positive since rewards > values
        assert!(advantages[0] > 0.0);
    }

    #[test]
    fn test_gae_with_dones() {
        let rewards = vec![1.0, 1.0, 10.0]; // Terminal reward of 10
        let values = vec![0.5, 0.5, 0.5, 0.0];
        let dones = vec![false, false, true];

        let advantages = compute_gae_with_dones(&rewards, &values, &dones, 0.99, 0.95);

        assert_eq!(advantages.len(), 3);
        // Terminal advantage should be reward - value = 10.0 - 0.5 = 9.5
        assert!((advantages[2] - 9.5).abs() < 1e-6);
        // All advantages should be positive given positive rewards
        assert!(advantages[0] > 0.0);
        assert!(advantages[1] > 0.0);
        assert!(advantages[2] > 0.0);
    }

    #[test]
    fn test_returns_computation() {
        let advantages = vec![1.0, 2.0, 3.0];
        let values = vec![0.5, 0.5, 0.5];

        let returns = compute_returns(&advantages, &values);

        assert_eq!(returns, vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn gae_plus_value_matches_lambda_return_with_bootstrap() {
        let rewards = [1.0, -1.0, 2.0];
        let values = [-0.5, 2.0, 1.0, -2.0];
        for gamma in [0.0, 0.5, 1.0] {
            for lambda in [0.0, 0.5, 1.0] {
                let advantages = compute_gae_internal(&rewards, &values, gamma, lambda);
                assert_eq!(
                    compute_returns(&advantages, &values),
                    compute_lambda_returns(&rewards, &values, gamma, lambda)
                );
                let terminal =
                    compute_gae_with_dones(&rewards, &values, &[false, true, false], gamma, lambda);
                assert_eq!(terminal[1] + values[1], rewards[1]);
                assert_eq!(
                    terminal[..2],
                    compute_gae_internal(&rewards[..2], &values[..2], gamma, lambda)
                );
                assert_eq!(terminal[2], advantages[2]);
            }
        }
    }

    #[test]
    fn test_empty_input() {
        let advantages = compute_gae_internal(&[], &[], 0.99, 0.95);
        assert!(advantages.is_empty());
    }
}
