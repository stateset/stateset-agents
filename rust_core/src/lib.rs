//! StateSet RL Core - High-performance Rust implementations for RL operations
//!
//! This crate provides optimized implementations of performance-critical
//! operations for the StateSet RL framework, exposed to Python via PyO3.
//!
//! # Features
//! - SIMD-accelerated advantage computation
//! - Parallel trajectory processing
//! - Efficient reward normalization
//! - Fast GAE computation

use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray as _};
use ndarray::Array1;
use rayon::prelude::*;
use std::collections::HashMap;

mod advantage;
mod gae;
mod trajectory;
mod rewards;

// Re-export core functions for Rust usage
pub use advantage::*;
pub use gae::*;
pub use trajectory::*;
pub use rewards::{
    normalize_with_running_stats, batch_normalize, exponential_moving_average,
    shape_rewards, auto_scale_rewards, clip_rewards, RewardStatistics,
};

/// Compute group-relative advantages for GRPO training
///
/// This is a high-performance implementation that uses SIMD where available
/// and parallelizes across groups.
///
/// Args:
///     rewards: 2D array of shape (num_groups, group_size) containing rewards
///     baseline_type: "mean", "median", or "min"
///     normalize: Whether to normalize advantages
///
/// Returns:
///     2D array of advantages with same shape as input
#[pyfunction]
fn compute_group_advantages<'py>(
    py: Python<'py>,
    rewards: PyReadonlyArray2<'py, f64>,
    baseline_type: &str,
    normalize: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let rewards = rewards.as_array();
    let (num_groups, group_size) = rewards.dim();

    let mut all_advantages: Vec<f64> = Vec::with_capacity(num_groups * group_size);

    // Process each group in parallel
    let group_advantages: Vec<Vec<f64>> = (0..num_groups)
        .into_par_iter()
        .map(|g| {
            let group_rewards: Vec<f64> = (0..group_size)
                .map(|i| rewards[[g, i]])
                .collect();

            advantage::compute_advantages_for_group(&group_rewards, baseline_type, normalize)
        })
        .collect();

    // Flatten results
    for group in group_advantages {
        all_advantages.extend(group);
    }

    Ok(Array1::from_vec(all_advantages).to_pyarray_bound(py))
}

/// Compute Generalized Advantage Estimation (GAE) for a trajectory
///
/// High-performance GAE computation with configurable gamma and lambda.
///
/// Args:
///     rewards: Array of per-step rewards
///     values: Array of value estimates (one more than rewards for bootstrap)
///     gamma: Discount factor (default 0.99)
///     gae_lambda: GAE lambda parameter (default 0.95)
///
/// Returns:
///     Array of advantage estimates
#[pyfunction]
#[pyo3(signature = (rewards, values, gamma=0.99, gae_lambda=0.95))]
fn compute_gae<'py>(
    py: Python<'py>,
    rewards: PyReadonlyArray1<'py, f64>,
    values: PyReadonlyArray1<'py, f64>,
    gamma: f64,
    gae_lambda: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let rewards = rewards.as_slice()?;
    let values = values.as_slice()?;

    let advantages = gae::compute_gae_internal(rewards, values, gamma, gae_lambda);

    Ok(Array1::from_vec(advantages).to_pyarray_bound(py))
}

/// Batch compute GAE for multiple trajectories in parallel
///
/// Args:
///     all_rewards: List of reward arrays
///     all_values: List of value arrays
///     gamma: Discount factor
///     gae_lambda: GAE lambda parameter
///
/// Returns:
///     List of advantage arrays
#[pyfunction]
#[pyo3(signature = (all_rewards, all_values, gamma=0.99, gae_lambda=0.95))]
fn batch_compute_gae<'py>(
    py: Python<'py>,
    all_rewards: Vec<PyReadonlyArray1<'py, f64>>,
    all_values: Vec<PyReadonlyArray1<'py, f64>>,
    gamma: f64,
    gae_lambda: f64,
) -> PyResult<Vec<Bound<'py, PyArray1<f64>>>> {
    let rewards_vecs: Vec<Vec<f64>> = all_rewards
        .iter()
        .map(|r| r.as_slice().map(|s| s.to_vec()).unwrap_or_default())
        .collect();

    let values_vecs: Vec<Vec<f64>> = all_values
        .iter()
        .map(|v| v.as_slice().map(|s| s.to_vec()).unwrap_or_default())
        .collect();

    // Parallel GAE computation
    let results: Vec<Vec<f64>> = rewards_vecs
        .par_iter()
        .zip(values_vecs.par_iter())
        .map(|(rewards, values)| {
            gae::compute_gae_internal(rewards, values, gamma, gae_lambda)
        })
        .collect();

    Ok(results
        .into_iter()
        .map(|v| Array1::from_vec(v).to_pyarray_bound(py))
        .collect())
}

/// Normalize rewards using running statistics
///
/// Efficient online normalization with Welford's algorithm.
///
/// Args:
///     rewards: Array of rewards to normalize
///     running_mean: Current running mean (will be updated)
///     running_var: Current running variance (will be updated)
///     count: Current count (will be updated)
///     epsilon: Small value for numerical stability
///
/// Returns:
///     Tuple of (normalized_rewards, new_mean, new_var, new_count)
#[pyfunction]
#[pyo3(signature = (rewards, running_mean=0.0, running_var=1.0, count=0, epsilon=1e-8))]
fn normalize_rewards<'py>(
    py: Python<'py>,
    rewards: PyReadonlyArray1<'py, f64>,
    running_mean: f64,
    running_var: f64,
    count: i64,
    epsilon: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64, f64, i64)> {
    let rewards = rewards.as_slice()?;

    let (normalized, new_mean, new_var, new_count) =
        rewards::normalize_with_running_stats(rewards, running_mean, running_var, count, epsilon);

    Ok((Array1::from_vec(normalized).to_pyarray_bound(py), new_mean, new_var, new_count))
}

/// Clip rewards to a specified range
#[pyfunction]
fn clip_rewards_py<'py>(
    py: Python<'py>,
    rewards: PyReadonlyArray1<'py, f64>,
    min_val: f64,
    max_val: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let rewards = rewards.as_slice()?;

    let clipped: Vec<f64> = rewards
        .iter()
        .map(|&r| r.clamp(min_val, max_val))
        .collect();

    Ok(Array1::from_vec(clipped).to_pyarray_bound(py))
}

/// Compute GSPO sequence-level importance ratios
///
/// Implements the length-normalized sequence importance ratio from GSPO:
/// s_i(θ) = (π_θ(y_i|x) / π_θ_old(y_i|x))^(1/|y_i|)
///
/// Args:
///     log_probs_new: Log probabilities under new policy (sum per sequence)
///     log_probs_old: Log probabilities under old policy (sum per sequence)
///     sequence_lengths: Length of each sequence
///
/// Returns:
///     Array of sequence-level importance ratios
#[pyfunction]
fn compute_gspo_importance_ratios<'py>(
    py: Python<'py>,
    log_probs_new: PyReadonlyArray1<'py, f64>,
    log_probs_old: PyReadonlyArray1<'py, f64>,
    sequence_lengths: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let new_probs = log_probs_new.as_slice()?;
    let old_probs = log_probs_old.as_slice()?;
    let lengths = sequence_lengths.as_slice()?;

    let ratios: Vec<f64> = new_probs
        .par_iter()
        .zip(old_probs.par_iter())
        .zip(lengths.par_iter())
        .map(|((&new, &old), &len)| {
            if len <= 0 {
                return 1.0;
            }
            let log_ratio = new - old;
            let normalized_log_ratio = log_ratio / (len as f64);
            normalized_log_ratio.exp()
        })
        .collect();

    Ok(Array1::from_vec(ratios).to_pyarray_bound(py))
}

/// Apply GSPO clipping to importance ratios
///
/// Args:
///     ratios: Importance ratios
///     advantages: Advantage values
///     clip_left: Left clipping bound (default 3e-4)
///     clip_right: Right clipping bound (default 4e-4)
///
/// Returns:
///     Clipped surrogate objectives
#[pyfunction]
#[pyo3(signature = (ratios, advantages, clip_left=0.0003, clip_right=0.0004))]
fn apply_gspo_clipping<'py>(
    py: Python<'py>,
    ratios: PyReadonlyArray1<'py, f64>,
    advantages: PyReadonlyArray1<'py, f64>,
    clip_left: f64,
    clip_right: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let ratios = ratios.as_slice()?;
    let advantages = advantages.as_slice()?;

    let clipped: Vec<f64> = ratios
        .par_iter()
        .zip(advantages.par_iter())
        .map(|(&ratio, &adv)| {
            let unclipped = ratio * adv;
            let clipped_ratio = if adv >= 0.0 {
                ratio.min(1.0 + clip_right)
            } else {
                ratio.max(1.0 - clip_left)
            };
            let clipped_obj = clipped_ratio * adv;
            unclipped.min(clipped_obj)
        })
        .collect();

    Ok(Array1::from_vec(clipped).to_pyarray_bound(py))
}

/// Compute PPO clipped surrogate objective
#[pyfunction]
#[pyo3(signature = (ratios, advantages, clip_epsilon=0.2))]
fn compute_ppo_surrogate<'py>(
    py: Python<'py>,
    ratios: PyReadonlyArray1<'py, f64>,
    advantages: PyReadonlyArray1<'py, f64>,
    clip_epsilon: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let ratios = ratios.as_slice()?;
    let advantages = advantages.as_slice()?;

    let objectives: Vec<f64> = ratios
        .par_iter()
        .zip(advantages.par_iter())
        .map(|(&ratio, &adv)| {
            let unclipped = ratio * adv;
            let clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv;
            unclipped.min(clipped)
        })
        .collect();

    Ok(Array1::from_vec(objectives).to_pyarray_bound(py))
}

/// Compute reward statistics for a batch of trajectories
#[pyfunction]
fn compute_reward_statistics(rewards: Vec<f64>) -> PyResult<HashMap<String, f64>> {
    if rewards.is_empty() {
        return Ok(HashMap::from([
            ("mean".to_string(), 0.0),
            ("std".to_string(), 0.0),
            ("min".to_string(), 0.0),
            ("max".to_string(), 0.0),
            ("median".to_string(), 0.0),
        ]));
    }

    let n = rewards.len() as f64;
    let mean = rewards.iter().sum::<f64>() / n;
    let variance = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();

    let min = rewards.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = rewards.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut sorted = rewards.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    Ok(HashMap::from([
        ("mean".to_string(), mean),
        ("std".to_string(), std),
        ("min".to_string(), min),
        ("max".to_string(), max),
        ("median".to_string(), median),
        ("count".to_string(), n),
    ]))
}

/// Python module definition
#[pymodule]
fn stateset_rl_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_group_advantages, m)?)?;
    m.add_function(wrap_pyfunction!(compute_gae, m)?)?;
    m.add_function(wrap_pyfunction!(batch_compute_gae, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_rewards, m)?)?;
    m.add_function(wrap_pyfunction!(clip_rewards_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_gspo_importance_ratios, m)?)?;
    m.add_function(wrap_pyfunction!(apply_gspo_clipping, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ppo_surrogate, m)?)?;
    m.add_function(wrap_pyfunction!(compute_reward_statistics, m)?)?;

    // Add version info
    m.add("__version__", "0.1.0")?;

    Ok(())
}
