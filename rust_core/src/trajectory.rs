//! Trajectory processing module
//!
//! Efficient trajectory batching and processing utilities.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Lightweight trajectory representation for Rust processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RustTrajectory {
    pub trajectory_id: String,
    pub rewards: Vec<f64>,
    pub log_probs: Vec<f64>,
    pub sequence_length: usize,
    pub total_reward: f64,
    pub metadata: HashMap<String, String>,
}

impl RustTrajectory {
    pub fn new(trajectory_id: String) -> Self {
        Self {
            trajectory_id,
            rewards: Vec::new(),
            log_probs: Vec::new(),
            sequence_length: 0,
            total_reward: 0.0,
            metadata: HashMap::new(),
        }
    }

    pub fn add_step(&mut self, reward: f64, log_prob: f64) {
        self.rewards.push(reward);
        self.log_probs.push(log_prob);
        self.total_reward += reward;
        self.sequence_length += 1;
    }

    pub fn average_reward(&self) -> f64 {
        if self.sequence_length == 0 {
            0.0
        } else {
            self.total_reward / self.sequence_length as f64
        }
    }
}

/// Trajectory group for GRPO processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RustTrajectoryGroup {
    pub scenario_id: String,
    pub trajectories: Vec<RustTrajectory>,
}

impl RustTrajectoryGroup {
    pub fn new(scenario_id: String) -> Self {
        Self {
            scenario_id,
            trajectories: Vec::new(),
        }
    }

    pub fn add_trajectory(&mut self, trajectory: RustTrajectory) {
        self.trajectories.push(trajectory);
    }

    pub fn rewards(&self) -> Vec<f64> {
        self.trajectories.iter().map(|t| t.total_reward).collect()
    }

    pub fn compute_advantages(&self, baseline_type: &str) -> Vec<f64> {
        let rewards = self.rewards();
        crate::advantage::compute_advantages_for_group(&rewards, baseline_type, false)
    }
}

/// Batch multiple trajectory groups for efficient processing
pub fn batch_trajectories(
    groups: &[RustTrajectoryGroup],
) -> (Vec<f64>, Vec<usize>) {
    let mut all_rewards = Vec::new();
    let mut group_indices = Vec::new();

    for (idx, group) in groups.iter().enumerate() {
        for traj in &group.trajectories {
            all_rewards.push(traj.total_reward);
            group_indices.push(idx);
        }
    }

    (all_rewards, group_indices)
}

/// Compute cumulative rewards for a trajectory
pub fn compute_cumulative_rewards(rewards: &[f64]) -> Vec<f64> {
    let mut cumulative = Vec::with_capacity(rewards.len());
    let mut total = 0.0;

    for &reward in rewards {
        total += reward;
        cumulative.push(total);
    }

    cumulative
}

/// Compute discounted cumulative rewards
pub fn compute_discounted_rewards(rewards: &[f64], gamma: f64) -> Vec<f64> {
    let n = rewards.len();
    if n == 0 {
        return vec![];
    }

    let mut discounted = vec![0.0; n];
    let mut running = 0.0;

    for t in (0..n).rev() {
        running = rewards[t] + gamma * running;
        discounted[t] = running;
    }

    discounted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trajectory_creation() {
        let mut traj = RustTrajectory::new("test-1".to_string());
        traj.add_step(1.0, -0.5);
        traj.add_step(2.0, -0.3);

        assert_eq!(traj.sequence_length, 2);
        assert!((traj.total_reward - 3.0).abs() < 1e-10);
        assert!((traj.average_reward() - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_trajectory_group() {
        let mut group = RustTrajectoryGroup::new("scenario-1".to_string());

        let mut t1 = RustTrajectory::new("t1".to_string());
        t1.add_step(1.0, -0.1);
        group.add_trajectory(t1);

        let mut t2 = RustTrajectory::new("t2".to_string());
        t2.add_step(3.0, -0.2);
        group.add_trajectory(t2);

        let rewards = group.rewards();
        assert_eq!(rewards, vec![1.0, 3.0]);

        let advantages = group.compute_advantages("mean");
        assert_eq!(advantages.len(), 2);
        // Mean is 2.0, so advantages should be [-1.0, 1.0]
        assert!((advantages[0] - (-1.0)).abs() < 1e-10);
        assert!((advantages[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cumulative_rewards() {
        let rewards = vec![1.0, 2.0, 3.0];
        let cumulative = compute_cumulative_rewards(&rewards);
        assert_eq!(cumulative, vec![1.0, 3.0, 6.0]);
    }

    #[test]
    fn test_discounted_rewards() {
        let rewards = vec![1.0, 1.0, 1.0];
        let discounted = compute_discounted_rewards(&rewards, 0.9);

        // G_2 = 1.0
        // G_1 = 1.0 + 0.9 * 1.0 = 1.9
        // G_0 = 1.0 + 0.9 * 1.9 = 2.71
        assert!((discounted[2] - 1.0).abs() < 1e-10);
        assert!((discounted[1] - 1.9).abs() < 1e-10);
        assert!((discounted[0] - 2.71).abs() < 1e-10);
    }
}
