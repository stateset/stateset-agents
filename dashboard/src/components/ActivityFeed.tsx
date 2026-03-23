import { Play, Pause, CheckCircle2, XCircle, Zap, Clock, TrendingUp } from 'lucide-react';
import type { Experiment } from '../types';
import { Card } from './Card';

interface ActivityFeedProps {
  experiments: Experiment[];
  onSelect: (exp: Experiment) => void;
}

interface Activity {
  id: string;
  type: 'started' | 'paused' | 'completed' | 'failed' | 'created' | 'milestone';
  experiment: Experiment;
  timestamp: number;
  detail?: string;
}

function deriveActivities(experiments: Experiment[]): Activity[] {
  const activities: Activity[] = [];

  for (const exp of experiments) {
    // Created event
    activities.push({
      id: `${exp.id}-created`,
      type: 'created',
      experiment: exp,
      timestamp: exp.created_at,
    });

    // Status-based events
    if (exp.status === 'running') {
      activities.push({
        id: `${exp.id}-running`,
        type: 'started',
        experiment: exp,
        timestamp: exp.updated_at,
      });
    } else if (exp.status === 'paused') {
      activities.push({
        id: `${exp.id}-paused`,
        type: 'paused',
        experiment: exp,
        timestamp: exp.updated_at,
      });
    } else if (exp.status === 'completed') {
      activities.push({
        id: `${exp.id}-completed`,
        type: 'completed',
        experiment: exp,
        timestamp: exp.updated_at,
        detail: `Best reward: ${exp.metrics.best_reward.toFixed(3)}`,
      });
    } else if (exp.status === 'failed') {
      activities.push({
        id: `${exp.id}-failed`,
        type: 'failed',
        experiment: exp,
        timestamp: exp.updated_at,
      });
    }

    // Milestone: reached 50% episodes
    const pct = exp.metrics.total_episodes / exp.training.num_episodes;
    if (pct >= 0.5 && exp.status === 'running') {
      activities.push({
        id: `${exp.id}-milestone-50`,
        type: 'milestone',
        experiment: exp,
        timestamp: exp.updated_at,
        detail: `${Math.round(pct * 100)}% complete — ${exp.metrics.total_episodes}/${exp.training.num_episodes} episodes`,
      });
    }
  }

  return activities
    .sort((a, b) => b.timestamp - a.timestamp)
    .slice(0, 12);
}

const activityMeta: Record<string, { icon: typeof Play; color: string; label: string }> = {
  started: { icon: Play, color: 'var(--green)', label: 'Started training' },
  paused: { icon: Pause, color: 'var(--amber)', label: 'Paused' },
  completed: { icon: CheckCircle2, color: 'var(--accent-light)', label: 'Completed' },
  failed: { icon: XCircle, color: 'var(--red)', label: 'Failed' },
  created: { icon: Zap, color: 'var(--text-muted)', label: 'Created' },
  milestone: { icon: TrendingUp, color: 'var(--cyan)', label: 'Milestone' },
};

function timeAgo(ts: number): string {
  const secs = Math.floor(Date.now() / 1000 - ts);
  if (secs < 60) return 'just now';
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
  return `${Math.floor(secs / 86400)}d ago`;
}

export function ActivityFeed({ experiments, onSelect }: ActivityFeedProps) {
  const activities = deriveActivities(experiments);

  if (activities.length === 0) return null;

  return (
    <Card style={{ padding: 16 }}>
      <div style={{
        fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)',
        marginBottom: 12, display: 'flex', alignItems: 'center', gap: 6,
      }}>
        <Clock size={13} />
        Recent Activity
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {activities.map(act => {
          const meta = activityMeta[act.type] ?? activityMeta.created;
          const Icon = meta.icon;
          return (
            <button
              key={act.id}
              onClick={() => onSelect(act.experiment)}
              style={{
                display: 'flex', alignItems: 'center', gap: 10,
                padding: '6px 8px', borderRadius: 6,
                border: 'none', background: 'transparent',
                color: 'var(--text-primary)', fontSize: 12,
                textAlign: 'left', width: '100%',
                transition: 'background 0.1s',
              }}
              onMouseEnter={e => (e.currentTarget.style.background = 'var(--bg-tertiary)')}
              onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
            >
              <Icon size={13} style={{ color: meta.color, flexShrink: 0 }} />
              <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                <span style={{ fontWeight: 500 }}>{act.experiment.name}</span>
                <span style={{ color: 'var(--text-muted)', marginLeft: 6 }}>
                  {meta.label}
                  {act.detail ? ` — ${act.detail}` : ''}
                </span>
              </span>
              <span style={{ fontSize: 10, color: 'var(--text-muted)', flexShrink: 0 }}>
                {timeAgo(act.timestamp)}
              </span>
            </button>
          );
        })}
      </div>
    </Card>
  );
}
