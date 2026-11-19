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

    use num_bigint::BigInt;
    use num_rational::BigRational;
    use num_traits::{Float, Signed, ToPrimitive, Zero};

    if !rewards.iter().all(|r| r.is_finite()) {
        return vec![0.0; rewards.len()];
    }
    // Put binary64 rewards on a common integer grid before any subtraction.
    let parts: Vec<_> = rewards.iter().map(|r| r.integer_decode()).collect();
    let exponent = parts
        .iter()
        .filter(|(m, _, _)| *m != 0)
        .map(|(_, e, _)| *e)
        .min()
        .unwrap_or(0)
        .min(0);
    let ticks: Vec<BigInt> = parts
        .iter()
        .map(|&(m, e, sign)| {
            if m == 0 {
                BigInt::zero()
            } else {
                (BigInt::from(m) * sign) << (e - exponent) as usize
            }
        })
        .collect();
    let count = rewards.len();
    let total: BigInt = ticks.iter().sum();
    let divisor = BigInt::from(count) << (-exponent) as usize;
    let residuals: Vec<_> = ticks.iter().map(|v| v * count - &total).collect();
    let ratio = |n: BigInt, d: &BigInt| {
        BigRational::new_raw(n, d.clone())
            .to_f64()
            .expect("positive denominator")
    };
    if normalize && count > 1 {
        let variance = ticks.iter().map(|v| v * v).sum::<BigInt>() * count - &total * &total;
        let eps = BigRational::from_float(1e-8).expect("finite epsilon");
        let eps_num = eps.numer() * &divisor;
        if &variance * eps.denom().pow(2) > eps_num.pow(2) {
            let magnitude = residuals
                .iter()
                .map(|r| r.abs())
                .max()
                .expect("nonempty group");
            let sd = ratio(variance, &magnitude.pow(2)).sqrt();
            let denominator = sd + ratio(eps_num, &(eps.denom() * &magnitude));
            return residuals
                .into_iter()
                .map(|r| ratio(r, &magnitude) / denominator)
                .collect();
        }
    }

    // The accelerator keeps raw baseline differences below its SD threshold.
    let (baseline, factor) = match baseline_type {
        "min" => (
            residuals.iter().min().expect("nonempty group").clone(),
            1_u8,
        ),
        "median" => {
            let mut ordered = residuals.clone();
            ordered.sort_unstable();
            (&ordered[(count - 1) / 2] + &ordered[count / 2], 2)
        }
        _ => (BigInt::zero(), 1),
    };
    let denominator = divisor * factor;
    residuals
        .into_iter()
        .map(|r| ratio(r * factor - &baseline, &denominator))
        .collect()
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
    fn preserves_mean_through_cancellation_and_opposite_extremes() {
        let r = [-1e30, 1.0, 1e30];
        let a = compute_advantages_for_group(&r, "mean", false);
        assert_eq!(a[1], 2.0 / 3.0);
        for baseline in ["mean", "median", "min"] {
            let normalized = compute_advantages_for_group(&[-f64::MAX, f64::MAX], baseline, true);
            assert_eq!(normalized, [-1.0, 1.0]);
            assert_eq!(
                compute_advantages_for_group(&[f64::MAX, f64::MAX], baseline, false),
                [0.0, 0.0]
            );
        }
    }

    #[test]
    fn test_empty_rewards() {
        let rewards: Vec<f64> = vec![];
        let advantages = compute_advantages_for_group(&rewards, "mean", false);
        assert!(advantages.is_empty());
    }
}
