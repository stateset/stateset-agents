export type RunStatus = 'queued' | 'running' | 'paused' | 'completed' | 'failed';
export type DatasetStatus = 'ready' | 'processing' | 'drift' | 'needs-review';
export type ModelStatus = 'available' | 'training' | 'deployed' | 'warming';
export type BadgeTone = 'neutral' | 'info' | 'active' | 'success' | 'warning' | 'danger';

export interface TrainingMetrics {
  avgReward: number;
  bestReward: number;
  loss: number;
  validationLoss: number;
  progressPercent: number;
  totalEpisodes: number;
  queueWaitMinutes?: number;
  etaMinutes?: number;
  throughputTokPerSec?: number;
}

export interface TrainingEpisode {
  id: string;
  episodeNumber: number;
  reward: number;
  status: string;
  turns: number;
  durationMs: number;
  summary: string;
}

export interface TrainingRun {
  id: string;
  name: string;
  description: string;
  status: RunStatus;
  baseModel: string;
  datasetName: string;
  owner: string;
  preset: string;
  algorithm: string;
  createdAt: number;
  updatedAt: number;
  tags: string[];
  metrics: TrainingMetrics;
  config: {
    batchSize: number;
    learningRate: number;
    numGenerations: number;
    loraRank: number;
    quantization: string;
    maxTurns: number;
    environment: string;
    difficulty: string;
  };
  recentEvents: string[];
  episodes: TrainingEpisode[];
  source: 'mock' | 'live';
}

export interface DatasetSummary {
  id: string;
  name: string;
  domain: string;
  status: DatasetStatus;
  tokens: number;
  examples: number;
  coverage: number;
  quality: number;
  format: string;
  updatedAt: number;
  notes: string;
}

export interface ModelSummary {
  id: string;
  name: string;
  family: string;
  role: string;
  status: ModelStatus;
  baseModel: string;
  evalScore: number;
  latencyMs: number;
  checkpointType: string;
  updatedAt: number;
  summary: string;
  tags: string[];
}

export interface DashboardSnapshot {
  activeRuns: number;
  queueDepth: number;
  avgReward: number;
  avgLoss: number;
  readyDatasets: number;
  deployableModels: number;
}

export interface EnvironmentPreset {
  id: string;
  name: string;
  description: string;
  maxTurnsDefault: number;
  difficulty: string;
}

export interface AlgorithmSummary {
  id: string;
  name: string;
  description: string;
}

export interface TrainingCatalog {
  algorithms: AlgorithmSummary[];
  environments: EnvironmentPreset[];
}
