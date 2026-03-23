export interface EnvironmentConfig {
  env_type: string;
  max_turns: number;
  scenarios: Record<string, unknown>[];
  reward_weights: Record<string, number>;
  difficulty: 'easy' | 'medium' | 'hard' | 'expert';
}

export interface AgentConfig {
  model_name: string;
  use_stub: boolean;
  temperature: number;
  top_p: number;
  max_new_tokens: number;
  system_prompt: string | null;
  memory_window: number;
}

export interface TrainingConfig {
  num_episodes: number;
  num_generations: number;
  learning_rate: number;
  batch_size: number;
  algorithm: 'grpo' | 'gspo' | 'ppo' | 'dapo' | 'vapo';
  use_kl_penalty: boolean;
  kl_coef: number;
  clip_ratio: number;
  entropy_coef: number;
  gamma: number;
  normalize_advantages: boolean;
}

export interface Experiment {
  id: string;
  name: string;
  description: string;
  status: 'created' | 'running' | 'paused' | 'completed' | 'failed';
  environment: EnvironmentConfig;
  agent: AgentConfig;
  training: TrainingConfig;
  metrics: TrainingMetrics;
  created_at: number;
  updated_at: number;
}

export interface TrainingMetrics {
  total_episodes: number;
  total_reward: number;
  avg_reward: number;
  best_reward: number;
  worst_reward?: number;
  convergence_rate?: number;
  reward_history: number[];
  episode_lengths: number[];
  loss_history: number[];
  lr_history: number[];
  kl_divergence: number[];
  entropy: number[];
  advantages: number[];
  reward_breakdown?: Record<string, number[]>;
}

export interface Episode {
  episode_id: string;
  experiment_id: string;
  episode_num: number;
  turns: EpisodeTurn[];
  total_reward: number;
  turn_rewards: number[];
  status: string;
  duration_ms: number;
  loss?: number;
  kl_divergence?: number;
  entropy?: number;
  advantage?: number;
  scenario?: Record<string, unknown>;
  metadata: Record<string, unknown>;
}

export interface EpisodeTurn {
  turn: number;
  role: string;
  content: string;
  reward: number;
  timestamp: number;
}

export interface EnvironmentPreset {
  id: string;
  name: string;
  description: string;
  icon: string;
  max_turns_default: number;
  reward_components: string[];
}

export interface AlgorithmInfo {
  id: string;
  name: string;
  description: string;
  params: string[];
}

export interface WsMessage {
  type: 'init' | 'episode' | 'status' | 'pong';
  data?: Episode;
  metrics?: TrainingMetrics;
  experiment?: Experiment;
  status?: string;
}
