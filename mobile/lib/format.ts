import type { BadgeTone } from './types';

export function formatCompactNumber(value: number): string {
  return new Intl.NumberFormat('en-US', {
    notation: 'compact',
    maximumFractionDigits: value >= 100 ? 0 : 1,
  }).format(value);
}

export function formatPercent(value: number, digits = 0): string {
  return `${value.toFixed(digits)}%`;
}

export function formatLearningRate(value: number): string {
  return value < 0.001 ? value.toExponential(1) : value.toFixed(4);
}

export function formatDurationMs(value: number): string {
  if (value >= 1000) return `${(value / 1000).toFixed(1)}s`;
  return `${Math.round(value)}ms`;
}

export function formatRelativeTime(timestamp: number): string {
  const deltaMinutes = Math.max(1, Math.round((Date.now() - timestamp) / 60_000));
  if (deltaMinutes < 60) return `${deltaMinutes}m ago`;
  const hours = Math.round(deltaMinutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

export function titleCase(value: string): string {
  return value
    .replace(/[_-]/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

export function toneForStatus(status: string): BadgeTone {
  switch (status) {
    case 'running':
      return 'active';
    case 'completed':
    case 'available':
    case 'deployed':
    case 'ready':
      return 'success';
    case 'queued':
    case 'warming':
    case 'processing':
      return 'warning';
    case 'failed':
    case 'drift':
    case 'needs-review':
      return 'danger';
    case 'training':
    case 'paused':
      return 'info';
    default:
      return 'neutral';
  }
}
