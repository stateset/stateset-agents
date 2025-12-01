//! Reward processing module
//!
//! Efficient reward normalization and statistics computation.

/// Normalize rewards using Welford's online algorithm for running statistics
///
/// Returns (normalized_rewards, new_mean, new_variance, new_count)
pub fn normalize_with_running_stats(
    rewards: &[f64],
    running_mean: f64,
    running_var: f64,
    count: i64,
    epsilon: f64,
) -> (Vec<f64>, f64, f64, i64) {
    if rewards.is_empty() {
        return (vec![], running_mean, running_var, count);
    }

    let mut mean = running_mean;
    let mut m2 = running_var * count as f64; // M2 = variance * n
    let mut n = count;

    // Update running statistics with Welford's algorithm
    for &reward in rewards {
        n += 1;
        let delta = reward - mean;
        mean += delta / n as f64;
        let delta2 = reward - mean;
        m2 += delta * delta2;
    }

    let new_var = if n > 1 { m2 / n as f64 } else { 0.0 };
    let std = (new_var + epsilon).sqrt();

    // Normalize rewards
    let normalized: Vec<f64> = rewards.iter().map(|&r| (r - mean) / std).collect();

    (normalized, mean, new_var, n)
}

/// Batch normalize rewards (mean 0, std 1)
pub fn batch_normalize(rewards: &[f64], epsilon: f64) -> Vec<f64> {
    if rewards.is_empty() {
        return vec![];
    }

    let n = rewards.len() as f64;
    let mean = rewards.iter().sum::<f64>() / n;
    let variance = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std = (variance + epsilon).sqrt();

    rewards.iter().map(|r| (r - mean) / std).collect()
}

/// Compute exponential moving average of rewards
pub fn exponential_moving_average(rewards: &[f64], alpha: f64) -> Vec<f64> {
    if rewards.is_empty() {
        return vec![];
    }

    let mut ema = vec![rewards[0]];

    for i in 1..rewards.len() {
        let prev_ema = ema[i - 1];
        ema.push(alpha * rewards[i] + (1.0 - alpha) * prev_ema);
    }

    ema
}

/// Reward shaping: transform sparse rewards to dense signals
pub fn shape_rewards(
    rewards: &[f64],
    potential_current: &[f64],
    potential_next: &[f64],
    gamma: f64,
) -> Vec<f64> {
    rewards
        .iter()
        .zip(potential_current.iter())
        .zip(potential_next.iter())
        .map(|((&r, &phi), &phi_next)| r + gamma * phi_next - phi)
        .collect()
}

/// Clip rewards to a range
pub fn clip_rewards(rewards: &[f64], min_val: f64, max_val: f64) -> Vec<f64> {
    rewards.iter().map(|&r| r.clamp(min_val, max_val)).collect()
}

/// Apply reward scaling with automatic range detection
pub fn auto_scale_rewards(rewards: &[f64], target_range: (f64, f64)) -> Vec<f64> {
    if rewards.is_empty() {
        return vec![];
    }

    let min_r = rewards.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_r = rewards.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if (max_r - min_r).abs() < 1e-10 {
        // All rewards are the same
        return vec![(target_range.0 + target_range.1) / 2.0; rewards.len()];
    }

    let (target_min, target_max) = target_range;
    let scale = (target_max - target_min) / (max_r - min_r);

    rewards
        .iter()
        .map(|&r| target_min + (r - min_r) * scale)
        .collect()
}

/// Compute reward statistics
pub struct RewardStatistics {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub count: usize,
    pub sum: f64,
}

impl RewardStatistics {
    pub fn compute(rewards: &[f64]) -> Self {
        if rewards.is_empty() {
            return Self {
                mean: 0.0,
                std: 0.0,
                min: 0.0,
                max: 0.0,
                median: 0.0,
                count: 0,
                sum: 0.0,
            };
        }

        let n = rewards.len() as f64;
        let sum: f64 = rewards.iter().sum();
        let mean = sum / n;
        let variance = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        let min = rewards.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = rewards.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let mut sorted = rewards.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        Self {
            mean,
            std,
            min,
            max,
            median,
            count: rewards.len(),
            sum,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_normalize() {
        let rewards = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = batch_normalize(&rewards, 1e-8);

        // Mean should be ~0
        let mean: f64 = normalized.iter().sum::<f64>() / normalized.len() as f64;
        assert!(mean.abs() < 1e-10);

        // Std should be ~1
        let variance: f64 =
            normalized.iter().map(|n| (n - mean).powi(2)).sum::<f64>() / normalized.len() as f64;
        let std = variance.sqrt();
        assert!((std - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_running_stats() {
        let rewards1 = vec![1.0, 2.0, 3.0];
        let (_, mean1, var1, count1) = normalize_with_running_stats(&rewards1, 0.0, 0.0, 0, 1e-8);

        let rewards2 = vec![4.0, 5.0, 6.0];
        let (_, mean2, _, count2) =
            normalize_with_running_stats(&rewards2, mean1, var1, count1, 1e-8);

        // After seeing [1,2,3,4,5,6], mean should be 3.5
        assert!((mean2 - 3.5).abs() < 0.01);
        assert_eq!(count2, 6);
    }

    #[test]
    fn test_ema() {
        let rewards = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ema = exponential_moving_average(&rewards, 0.5);

        assert_eq!(ema.len(), 5);
        assert!((ema[0] - 1.0).abs() < 1e-10);
        // EMA should be smoothed
        assert!(ema[4] < 5.0);
    }

    #[test]
    fn test_auto_scale() {
        let rewards = vec![0.0, 50.0, 100.0];
        let scaled = auto_scale_rewards(&rewards, (-1.0, 1.0));

        assert!((scaled[0] - (-1.0)).abs() < 1e-10);
        assert!((scaled[1] - 0.0).abs() < 1e-10);
        assert!((scaled[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reward_statistics() {
        let rewards = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = RewardStatistics::compute(&rewards);

        assert!((stats.mean - 3.0).abs() < 1e-10);
        assert!((stats.min - 1.0).abs() < 1e-10);
        assert!((stats.max - 5.0).abs() < 1e-10);
        assert!((stats.median - 3.0).abs() < 1e-10);
        assert_eq!(stats.count, 5);
        assert!((stats.sum - 15.0).abs() < 1e-10);
    }
}
