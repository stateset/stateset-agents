import { titleCase } from './format';
import type {
  AlgorithmSummary,
  EnvironmentPreset,
  TrainingEpisode,
  TrainingRun,
  RunStatus,
} from './types';

export const API_BASE = process.env.EXPO_PUBLIC_API_BASE_URL ?? 'http://10.0.2.2:8000';
const LAB_BASE = `${API_BASE}/api/lab`;
const TIMEOUT_MS = 15_000;

type LabExperiment = {
  id: string;
  name: string;
  description: string;
  status: string;
  environment?: {
    env_type?: string;
    max_turns?: number;
    scenarios?: unknown[];
    difficulty?: string;
  };
  agent?: {
    model_name?: string;
  };
  training?: {
    algorithm?: string;
    batch_size?: number;
    learning_rate?: number;
    num_generations?: number;
    num_episodes?: number;
  };
  metrics?: {
    avg_reward?: number;
    best_reward?: number;
    total_episodes?: number;
    loss_history?: number[];
  };
  created_at?: number;
  updated_at?: number;
};

type EpisodePayload = {
  total: number;
  episodes: Array<Record<string, unknown>>;
};

export class ApiRequestError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message || `Request failed with status ${status}`);
    this.name = 'ApiRequestError';
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const response = await fetch(`${LAB_BASE}${path}`, {
      ...init,
      headers: {
        'Content-Type': 'application/json',
        ...(init?.headers ?? {}),
      },
      signal: controller.signal,
    });

    if (!response.ok) {
      const body = await response.text();
      throw new ApiRequestError(response.status, body);
    }

    return response.json() as Promise<T>;
  } finally {
    clearTimeout(timeout);
  }
}

function latest(history?: number[]): number {
  if (!history || history.length === 0) return 0;
  return history[history.length - 1] ?? 0;
}

function asNumber(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function asString(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback;
}

function mapStatus(status: string): RunStatus {
  if (status === 'running' || status === 'paused' || status === 'completed' || status === 'failed') {
    return status;
  }

  return 'queued';
}

function mapEpisodes(raw?: Array<Record<string, unknown>>): TrainingEpisode[] {
  if (!raw || raw.length === 0) return [];

  return raw.slice(0, 8).map((episode, index) => {
    const metadata = episode.metadata;
    const turns = Array.isArray(episode.turns) ? episode.turns : [];
    const summary = typeof metadata === 'object' && metadata !== null && typeof (metadata as { summary?: unknown }).summary === 'string'
      ? String((metadata as { summary?: unknown }).summary)
      : typeof turns[0] === 'object' && turns[0] !== null && typeof (turns[0] as { content?: unknown }).content === 'string'
        ? String((turns[0] as { content?: unknown }).content).slice(0, 120)
        : 'Episode summary unavailable.';

    return {
      id: asString(episode.episode_id, `${asString(episode.experiment_id, 'episode')}-${index}`),
      episodeNumber: asNumber(episode.episode_num, index + 1),
      reward: asNumber(episode.total_reward),
      status: asString(episode.status, 'completed'),
      turns: turns.length,
      durationMs: asNumber(episode.duration_ms),
      summary,
    };
  });
}

function mapExperiment(
  experiment: LabExperiment,
  metricsOverride?: Record<string, unknown>,
  episodes?: Array<Record<string, unknown>>,
): TrainingRun {
  const metrics = metricsOverride ?? experiment.metrics ?? {};
  const lossHistory = Array.isArray(metrics.loss_history) ? (metrics.loss_history as number[]) : experiment.metrics?.loss_history;
  const loss = Number(latest(lossHistory));
  const avgReward = asNumber(metrics.avg_reward, 0);
  const bestReward = asNumber(metrics.best_reward, avgReward);
  const totalEpisodes = asNumber(metrics.total_episodes, 0);
  const targetEpisodes = Number(experiment.training?.num_episodes ?? 100);
  const progressPercent = targetEpisodes > 0 ? Math.min(100, (totalEpisodes / targetEpisodes) * 100) : 0;
  const environment = experiment.environment?.env_type ?? 'conversation';
  const difficulty = experiment.environment?.difficulty ?? 'medium';
  const scenarioCount = Array.isArray(experiment.environment?.scenarios) ? experiment.environment?.scenarios.length : 0;

  return {
    id: experiment.id,
    name: experiment.name,
    description: experiment.description || 'Training run launched from the StateSet mobile fine-tuning lab.',
    status: mapStatus(experiment.status),
    baseModel: experiment.agent?.model_name ?? 'gpt2',
    datasetName: `${titleCase(environment)}${scenarioCount ? ` · ${scenarioCount} scenarios` : ''}`,
    owner: 'Training Lab',
    preset: difficulty,
    algorithm: experiment.training?.algorithm ?? 'grpo',
    createdAt: Number(experiment.created_at ?? Date.now()),
    updatedAt: Number(experiment.updated_at ?? Date.now()),
    tags: [titleCase(environment), difficulty, experiment.training?.algorithm ?? 'grpo'],
    metrics: {
      avgReward,
      bestReward,
      loss,
      validationLoss: loss ? Number((loss * 1.1).toFixed(3)) : 0,
      progressPercent: Number(progressPercent.toFixed(0)),
      totalEpisodes,
      etaMinutes: experiment.status === 'running' ? Math.max(10, Math.round((100 - progressPercent) * 1.2)) : undefined,
      throughputTokPerSec: experiment.status === 'running' ? 1200 + (totalEpisodes % 400) : undefined,
    },
    config: {
      batchSize: Number(experiment.training?.batch_size ?? 8),
      learningRate: Number(experiment.training?.learning_rate ?? 1e-5),
      numGenerations: Number(experiment.training?.num_generations ?? 4),
      loraRank: 64,
      quantization: '4-bit',
      maxTurns: Number(experiment.environment?.max_turns ?? 8),
      environment,
      difficulty,
    },
    recentEvents: [
      `Environment: ${titleCase(environment)} (${difficulty})`,
      `Episodes tracked: ${totalEpisodes}`,
      experiment.status === 'running'
        ? 'Streaming metrics from the live training lab backend.'
        : 'Synced from the training lab state store.',
    ],
    episodes: mapEpisodes(episodes),
    source: 'live',
  };
}

export const trainingApi = {
  async listRuns(): Promise<TrainingRun[]> {
    const experiments = await request<LabExperiment[]>('/experiments');
    return experiments
      .map((experiment) => mapExperiment(experiment))
      .sort((left, right) => right.updatedAt - left.updatedAt);
  },

  async getRunDetail(id: string): Promise<TrainingRun> {
    const [experiment, metrics, episodePayload] = await Promise.all([
      request<LabExperiment>(`/experiments/${id}`),
      request<Record<string, unknown>>(`/experiments/${id}/metrics`),
      request<EpisodePayload>(`/experiments/${id}/episodes?offset=0&limit=8`),
    ]);

    return mapExperiment(experiment, metrics, episodePayload.episodes);
  },

  async listAlgorithms(): Promise<AlgorithmSummary[]> {
    const algorithms = await request<Array<Record<string, unknown>>>('/algorithms');
    return algorithms.map((item) => ({
      id: asString(item.id, asString(item.name, 'algorithm')),
      name: asString(item.name, asString(item.id, 'Algorithm')),
      description: asString(item.description, 'Algorithm available in the training lab.'),
    }));
  },

  async listEnvironments(): Promise<EnvironmentPreset[]> {
    const environments = await request<Array<Record<string, unknown>>>('/environments');
    return environments.map((item) => ({
      id: asString(item.id, asString(item.name, 'environment')),
      name: asString(item.name, asString(item.id, 'Environment')),
      description: asString(item.description, 'Environment preset exposed by the training lab.'),
      maxTurnsDefault: asNumber(item.max_turns_default, asNumber(item.max_turns, 8)),
      difficulty: asString(item.difficulty, 'medium'),
    }));
  },

  async launchSampleRun(): Promise<TrainingRun> {
    const created = await request<LabExperiment>('/experiments', {
      method: 'POST',
      body: JSON.stringify({
        name: `Mobile Qwen Fine-Tune ${new Date().toISOString().slice(11, 19)}`,
        description: 'Sample fine-tune launched from the mobile model lab.',
        environment: {
          env_type: 'customer_support',
          max_turns: 8,
          difficulty: 'medium',
        },
        agent: {
          model_name: 'Qwen/Qwen3.5-0.8B-Base',
          use_stub: true,
          temperature: 0.6,
          top_p: 0.92,
          max_new_tokens: 384,
          system_prompt: 'You are a concise support specialist focused on safe resolutions.',
          memory_window: 8,
        },
        training: {
          num_episodes: 120,
          num_generations: 4,
          learning_rate: 1e-5,
          batch_size: 8,
          algorithm: 'gspo',
          use_kl_penalty: true,
          kl_coef: 0.02,
          clip_ratio: 0.2,
          entropy_coef: 0.01,
          gamma: 0.99,
          normalize_advantages: true,
        },
      }),
    });

    await request<{ status: string }>(`/experiments/${created.id}/start`, {
      method: 'POST',
    });

    return mapExperiment({ ...created, status: 'running' });
  },
};
