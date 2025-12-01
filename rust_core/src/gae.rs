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
    let n = rewards.len();
    if n == 0 {
        return vec![];
    }

    let mut advantages = vec![0.0; n];
    let mut gae = 0.0;

    // Compute backwards for numerical stability
    for t in (0..n).rev() {
        let next_value = if t + 1 < values.len() {
            values[t + 1]
        } else {
            0.0 // Terminal state
        };

        let current_value = if t < values.len() {
            values[t]
        } else {
            0.0
        };

        // TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
        let delta = rewards[t] + gamma * next_value - current_value;

        // GAE: A_t = δ_t + (γλ)A_{t+1}
        gae = delta + gamma * gae_lambda * gae;
        advantages[t] = gae;
    }

    advantages
}

/// Compute GAE with done flags for episode boundaries
pub fn compute_gae_with_dones(
    rewards: &[f64],
    values: &[f64],
    dones: &[bool],
    gamma: f64,
    gae_lambda: f64,
) -> Vec<f64> {
    let n = rewards.len();
    if n == 0 {
        return vec![];
    }

    let mut advantages = vec![0.0; n];
    let mut gae = 0.0;

    for t in (0..n).rev() {
        let next_non_terminal = if t < dones.len() && dones[t] {
            0.0
        } else {
            1.0
        };

        let next_value = if t + 1 < values.len() {
            values[t + 1]
        } else {
            0.0
        };

        let current_value = if t < values.len() {
            values[t]
        } else {
            0.0
        };

        let delta = rewards[t] + gamma * next_value * next_non_terminal - current_value;
        gae = delta + gamma * gae_lambda * next_non_terminal * gae;
        advantages[t] = gae;
    }

    advantages
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
    let n = rewards.len();
    if n == 0 {
        return vec![];
    }

    let mut returns = vec![0.0; n];
    let mut next_return = if n < values.len() { values[n] } else { 0.0 };

    for t in (0..n).rev() {
        let next_value = if t + 1 < values.len() {
            values[t + 1]
        } else {
            0.0
        };

        // Lambda return: G_t^λ = (1-λ)(V(s_{t+1}) + r_t) + λ(r_t + γG_{t+1}^λ)
        let one_step_return = rewards[t] + gamma * next_value;
        let multi_step_return = rewards[t] + gamma * next_return;

        returns[t] = (1.0 - lambda) * one_step_return + lambda * multi_step_return;
        next_return = returns[t];
    }

    returns
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // Last advantage should be high due to terminal reward
        assert!(advantages[2] > advantages[0]);
    }

    #[test]
    fn test_returns_computation() {
        let advantages = vec![1.0, 2.0, 3.0];
        let values = vec![0.5, 0.5, 0.5];

        let returns = compute_returns(&advantages, &values);

        assert_eq!(returns, vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn test_empty_input() {
        let advantages = compute_gae_internal(&[], &[], 0.99, 0.95);
        assert!(advantages.is_empty());
    }
}
