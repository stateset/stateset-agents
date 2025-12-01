//! Advantage computation module
//!
//! High-performance advantage calculations for GRPO and related algorithms.

/// Compute advantages for a single group
///
/// Supports multiple baseline types:
/// - "mean": Group mean baseline (default)
/// - "median": Group median baseline
/// - "min": Minimum reward as baseline
pub fn compute_advantages_for_group(
    rewards: &[f64],
    baseline_type: &str,
    normalize: bool,
) -> Vec<f64> {
    if rewards.is_empty() {
        return vec![];
    }

    // Compute baseline
    let baseline = match baseline_type {
        "median" => {
            let mut sorted = rewards.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = sorted.len() / 2;
            if sorted.len() % 2 == 0 {
                (sorted[mid - 1] + sorted[mid]) / 2.0
            } else {
                sorted[mid]
            }
        }
        "min" => rewards.iter().cloned().fold(f64::INFINITY, f64::min),
        _ => rewards.iter().sum::<f64>() / rewards.len() as f64, // mean
    };

    // Compute raw advantages
    let mut advantages: Vec<f64> = rewards.iter().map(|r| r - baseline).collect();

    // Normalize if requested
    if normalize && advantages.len() > 1 {
        let mean = advantages.iter().sum::<f64>() / advantages.len() as f64;
        let variance = advantages.iter().map(|a| (a - mean).powi(2)).sum::<f64>()
            / advantages.len() as f64;
        let std = variance.sqrt();

        if std > 1e-8 {
            for adv in &mut advantages {
                *adv = (*adv - mean) / (std + 1e-8);
            }
        }
    }

    advantages
}

/// Compute advantages with a global baseline
pub fn compute_advantages_global_baseline(
    rewards: &[f64],
    global_mean: f64,
    global_std: f64,
    normalize: bool,
) -> Vec<f64> {
    if rewards.is_empty() {
        return vec![];
    }

    let mut advantages: Vec<f64> = rewards.iter().map(|r| r - global_mean).collect();

    if normalize && global_std > 1e-8 {
        for adv in &mut advantages {
            *adv /= global_std + 1e-8;
        }
    }

    advantages
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_baseline() {
        let rewards = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let advantages = compute_advantages_for_group(&rewards, "mean", false);

        assert_eq!(advantages.len(), 5);
        assert!((advantages[0] - (-2.0)).abs() < 1e-10);
        assert!((advantages[2] - 0.0).abs() < 1e-10);
        assert!((advantages[4] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_median_baseline() {
        let rewards = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let advantages = compute_advantages_for_group(&rewards, "median", false);

        // Median is 3.0
        assert!((advantages[0] - (-2.0)).abs() < 1e-10);
        assert!((advantages[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalized_advantages() {
        let rewards = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let advantages = compute_advantages_for_group(&rewards, "mean", true);

        // After normalization, mean should be ~0 and std ~1
        let mean: f64 = advantages.iter().sum::<f64>() / advantages.len() as f64;
        assert!(mean.abs() < 1e-10);
    }

    #[test]
    fn test_empty_rewards() {
        let rewards: Vec<f64> = vec![];
        let advantages = compute_advantages_for_group(&rewards, "mean", false);
        assert!(advantages.is_empty());
    }
}
