// Typed fetch wrapper for the FastAPI `/api/lab/*` "Training Lab" router
// (stateset_agents/api/routers/training_lab.py). That router is simulator-
// backed (no real training executes behind it) and is gated behind auth
// plus the `API_ENABLE_TRAINING_LAB` server flag — it is disabled by
// default and NOT deployed anywhere. See ../README.md for status/how to
// run locally. Requests here will 401/404 against a default deployment.
//
// Base URL: defaults to the same-origin `/api/lab` path (works when the
// dashboard is served behind a proxy that forwards to the API). Set
// `VITE_API_BASE_URL` at build time to point at an absolute origin
// instead (e.g. `https://api.example.com/api/lab`) — see ../.env.example.
//
// Auth: the server's auth layer accepts an API key via the `X-API-Key`
// header (HTTP) or `api_key`/`token` query params (WebSocket). Provide one
// via `VITE_API_KEY` at build time, or at runtime with `setApiKey()`
// (persisted to localStorage under `stateset.apiKey`) — runtime values
// take precedence over the build-time default.
import type {
  Experiment,
  EnvironmentPreset,
  AlgorithmInfo,
  Episode,
} from './types';

const BASE = import.meta.env.VITE_API_BASE_URL ?? '/api/lab';
const BUILD_API_KEY = import.meta.env.VITE_API_KEY;
const STORAGE_KEY = 'stateset.apiKey';

export function getApiKey(): string | undefined {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored) return stored;
  } catch {
    // localStorage unavailable (SSR, privacy mode, etc.) — fall through.
  }
  return BUILD_API_KEY || undefined;
}

export function setApiKey(key: string | null): void {
  try {
    if (key) {
      window.localStorage.setItem(STORAGE_KEY, key);
    } else {
      window.localStorage.removeItem(STORAGE_KEY);
    }
  } catch {
    // localStorage unavailable — key can still be supplied via VITE_API_KEY.
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const apiKey = getApiKey();
  const res = await fetch(`${BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(apiKey ? { 'X-API-Key': apiKey } : {}),
    },
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
  const apiKey = getApiKey();
  const query = apiKey ? `?api_key=${encodeURIComponent(apiKey)}` : '';

  let wsBase: string;
  if (/^https?:\/\//i.test(BASE)) {
    // Absolute VITE_API_BASE_URL — derive ws(s):// from its origin.
    const url = new URL(BASE);
    url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
    wsBase = url.toString().replace(/\/$/, '');
  } else {
    // Same-origin relative path (default behavior).
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    wsBase = `${protocol}://${window.location.host}${BASE}`;
  }

  return new WebSocket(`${wsBase}/experiments/${experimentId}/ws${query}`);
}
