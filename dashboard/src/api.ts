import type {
  Experiment,
  EnvironmentPreset,
  AlgorithmInfo,
  Episode,
} from './types';

const BASE = '/api/lab';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json();
}

export const api = {
  listEnvironments: () => request<EnvironmentPreset[]>('/environments'),
  listAlgorithms: () => request<AlgorithmInfo[]>('/algorithms'),

  createExperiment: (data: {
    name: string;
    description?: string;
    environment?: Partial<Experiment['environment']>;
    agent?: Partial<Experiment['agent']>;
    training?: Partial<Experiment['training']>;
  }) => request<Experiment>('/experiments', { method: 'POST', body: JSON.stringify(data) }),

  listExperiments: () => request<Experiment[]>('/experiments'),
  getExperiment: (id: string) => request<Experiment>(`/experiments/${id}`),
  deleteExperiment: (id: string) =>
    request<{ status: string }>(`/experiments/${id}`, { method: 'DELETE' }),

  startExperiment: (id: string) =>
    request<{ status: string }>(`/experiments/${id}/start`, { method: 'POST' }),
  pauseExperiment: (id: string) =>
    request<{ status: string }>(`/experiments/${id}/pause`, { method: 'POST' }),
  resumeExperiment: (id: string) =>
    request<{ status: string }>(`/experiments/${id}/resume`, { method: 'POST' }),
  stopExperiment: (id: string) =>
    request<{ status: string }>(`/experiments/${id}/stop`, { method: 'POST' }),

  cloneExperiment: (id: string) =>
    request<Experiment>(`/experiments/${id}/clone`, { method: 'POST' }),

  patchExperimentConfig: (id: string, patch: {
    num_episodes?: number;
    learning_rate?: number;
    batch_size?: number;
    temperature?: number;
  }) => request<{ status: string; updated_fields: string[] }>(
    `/experiments/${id}/config`,
    { method: 'PATCH', body: JSON.stringify(patch) },
  ),

  getEpisodes: (id: string, offset = 0, limit = 50) =>
    request<{ total: number; episodes: Episode[] }>(
      `/experiments/${id}/episodes?offset=${offset}&limit=${limit}`
    ),
  getMetrics: (id: string) => request<Record<string, unknown>>(`/experiments/${id}/metrics`),

  exportExperiment: (id: string, format: 'json' | 'csv' = 'json') =>
    request<Record<string, unknown>>(`/experiments/${id}/export?format=${format}`),
};

export function connectWs(experimentId: string): WebSocket {
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return new WebSocket(`${protocol}://${window.location.host}${BASE}/experiments/${experimentId}/ws`);
}
