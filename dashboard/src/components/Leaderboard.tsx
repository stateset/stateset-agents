import { useState, useEffect } from 'react';
import { Trophy, ArrowUpDown, Medal, TrendingUp, TrendingDown } from 'lucide-react';
import { Card } from './Card';
import type { Experiment } from '../types';

interface LeaderboardEntry {
  id: string;
  name: string;
  status: string;
  algorithm: string;
  environment: string;
  difficulty: string;
  total_episodes: number;
  avg_reward: number;
  best_reward: number;
  convergence_rate: number;
  final_loss: number;
  created_at: number;
}

type SortField = 'avg_reward' | 'best_reward' | 'convergence_rate' | 'total_episodes' | 'final_loss';

interface LeaderboardProps {
  onSelectExperiment: (exp: Experiment) => void;
}

export function Leaderboard({ onSelectExperiment }: LeaderboardProps) {
  const [entries, setEntries] = useState<LeaderboardEntry[]>([]);
  const [sortBy, setSortBy] = useState<SortField>('avg_reward');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`/api/lab/leaderboard?sort_by=${sortBy}&limit=50`)
      .then(r => r.json())
      .then(data => { setEntries(data); setLoading(false); })
      .catch(() => setLoading(false));
  }, [sortBy]);

  const sortOptions: { field: SortField; label: string }[] = [
    { field: 'avg_reward', label: 'Avg Reward' },
    { field: 'best_reward', label: 'Best Reward' },
    { field: 'convergence_rate', label: 'Convergence' },
    { field: 'total_episodes', label: 'Episodes' },
    { field: 'final_loss', label: 'Lowest Loss' },
  ];

  const medalColors = ['#fbbf24', '#9ca3af', '#d97706'];

  return (
    <div style={{ padding: 32 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: 24 }}>
        <div>
          <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 4, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Trophy size={20} style={{ color: 'var(--amber)' }} />
            Leaderboard
          </h2>
          <p style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
            Rank experiments by performance metrics.
          </p>
        </div>
      </div>

      {/* Sort pills */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 20 }}>
        {sortOptions.map(opt => (
          <button
            key={opt.field}
            onClick={() => setSortBy(opt.field)}
            style={{
              display: 'flex', alignItems: 'center', gap: 4,
              padding: '6px 14px', borderRadius: 16, fontSize: 12, fontWeight: 500,
              border: sortBy === opt.field ? '1px solid var(--accent)' : '1px solid var(--border)',
              background: sortBy === opt.field ? 'rgba(99,102,241,0.1)' : 'transparent',
              color: sortBy === opt.field ? 'var(--accent-light)' : 'var(--text-secondary)',
            }}
          >
            <ArrowUpDown size={11} />
            {opt.label}
          </button>
        ))}
      </div>

      {loading ? (
        <Card><div style={{ textAlign: 'center', padding: 40, color: 'var(--text-muted)' }}>Loading...</div></Card>
      ) : entries.length === 0 ? (
        <Card>
          <div style={{ textAlign: 'center', padding: 60 }}>
            <Trophy size={40} style={{ color: 'var(--text-muted)', marginBottom: 12 }} />
            <div style={{ fontSize: 14, color: 'var(--text-muted)' }}>No experiments to rank yet.</div>
          </div>
        </Card>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {entries.map((entry, i) => {
            const rank = i + 1;
            const isTop3 = rank <= 3;
            return (
              <Card
                key={entry.id}
                hover
                onClick={() => {
                  // Construct a minimal Experiment object for navigation
                  // Fetch the full experiment to navigate properly
                  fetch(`/api/lab/experiments/${entry.id}`)
                    .then(r => r.json())
                    .then(exp => onSelectExperiment(exp))
                    .catch(() => {})}
                }
                style={{
                  padding: '12px 16px',
                  borderLeft: isTop3 ? `3px solid ${medalColors[i] ?? 'var(--border)'}` : undefined,
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                  {/* Rank */}
                  <div style={{
                    width: 36, height: 36, borderRadius: '50%',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: isTop3 ? 16 : 13, fontWeight: 700, flexShrink: 0,
                    background: isTop3 ? `color-mix(in srgb, ${medalColors[i]} 15%, transparent)` : 'var(--bg-tertiary)',
                    color: isTop3 ? medalColors[i] : 'var(--text-muted)',
                  }}>
                    {isTop3 ? <Medal size={18} /> : rank}
                  </div>

                  {/* Info */}
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 2 }}>{entry.name}</div>
                    <div style={{ display: 'flex', gap: 8, fontSize: 11, color: 'var(--text-muted)' }}>
                      <Tag>{entry.algorithm.toUpperCase()}</Tag>
                      <Tag>{entry.environment}</Tag>
                      <Tag>{entry.difficulty}</Tag>
                      <span>{entry.total_episodes} episodes</span>
                    </div>
                  </div>

                  {/* Metrics */}
                  <div style={{ display: 'flex', gap: 20, alignItems: 'center' }}>
                    <Metric label="Avg Reward" value={entry.avg_reward.toFixed(3)} highlight={sortBy === 'avg_reward'} />
                    <Metric label="Best" value={entry.best_reward.toFixed(3)} color="var(--green)" highlight={sortBy === 'best_reward'} />
                    <Metric label="Loss" value={entry.final_loss.toFixed(4)} highlight={sortBy === 'final_loss'} />
                    <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                      {entry.convergence_rate > 0
                        ? <TrendingUp size={14} style={{ color: 'var(--green)' }} />
                        : <TrendingDown size={14} style={{ color: 'var(--red)' }} />}
                      <div>
                        <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Conv.</div>
                        <div style={{
                          fontSize: 12, fontWeight: 600, fontVariantNumeric: 'tabular-nums',
                          color: entry.convergence_rate > 0 ? 'var(--green)' : 'var(--red)',
                        }}>
                          {entry.convergence_rate > 0 ? '+' : ''}{entry.convergence_rate.toFixed(4)}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}

function Tag({ children }: { children: React.ReactNode }) {
  return (
    <span style={{
      padding: '1px 6px', borderRadius: 4,
      background: 'var(--bg-tertiary)', fontSize: 10,
    }}>
      {children}
    </span>
  );
}

function Metric({ label, value, color, highlight }: {
  label: string; value: string; color?: string; highlight?: boolean;
}) {
  return (
    <div style={{ textAlign: 'right', opacity: highlight ? 1 : 0.8 }}>
      <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>{label}</div>
      <div style={{
        fontSize: highlight ? 14 : 12,
        fontWeight: highlight ? 700 : 600,
        fontVariantNumeric: 'tabular-nums',
        color: color ?? 'var(--text-primary)',
      }}>
        {value}
      </div>
    </div>
  );
}
