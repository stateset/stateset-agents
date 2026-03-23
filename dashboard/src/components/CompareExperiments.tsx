import { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, Brush,
} from 'recharts';
import { Card, StatCard } from './Card';
import { api } from '../api';
import { useToast } from '../hooks/useToast';
import type { Experiment } from '../types';

const COLORS = ['#6366f1', '#22c55e', '#f59e0b', '#ef4444', '#a855f7', '#06b6d4', '#ec4899'];

interface CompareExperimentsProps {
  onBack: () => void;
}

interface CompResult {
  id: string;
  name: string;
  status: string;
  algorithm: string;
  environment: string;
  difficulty: string;
  total_episodes: number;
  avg_reward: number;
  best_reward: number;
  worst_reward: number;
  convergence_rate: number;
  reward_history: number[];
  loss_history: number[];
  final_loss: number;
}

type ChartMetric = 'reward' | 'loss';

export function CompareExperiments({ onBack }: CompareExperimentsProps) {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [results, setResults] = useState<CompResult[]>([]);
  const [chartMetric, setChartMetric] = useState<ChartMetric>('reward');
  const [showConfigDiff, setShowConfigDiff] = useState(false);
  const toast = useToast();

  useEffect(() => {
    api.listExperiments().then(setExperiments).catch(() => {});
  }, []);

  const toggleSelect = (id: string) => {
    setSelectedIds(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id].slice(0, 6)
    );
  };

  const runComparison = async () => {
    if (selectedIds.length < 2) return;
    try {
      const res = await fetch('/api/lab/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(selectedIds),
      });
      const data = await res.json();
      setResults(data.experiments ?? []);
    } catch {
      toast.error('Comparison failed');
    }
  };

  // Build overlay chart data
  const maxLen = Math.max(...results.map(r => r.reward_history.length), 0);
  const rewardOverlay = Array.from({ length: maxLen }, (_, i) => {
    const point: Record<string, unknown> = { episode: i + 1 };
    for (const r of results) {
      if (i < r.reward_history.length) {
        const start = Math.max(0, i - 9);
        const slice = r.reward_history.slice(start, i + 1);
        point[r.name] = +(slice.reduce((a, b) => a + b, 0) / slice.length).toFixed(4);
      }
    }
    return point;
  });

  const lossOverlay = Array.from({ length: maxLen }, (_, i) => {
    const point: Record<string, unknown> = { episode: i + 1 };
    for (const r of results) {
      if (i < r.loss_history.length) point[r.name] = r.loss_history[i];
    }
    return point;
  });

  // Radar data
  const radarData = results.length > 0 ? [
    { metric: 'Avg Reward', ...Object.fromEntries(results.map(r => [r.name, Math.max(0, r.avg_reward * 10)])) },
    { metric: 'Best Reward', ...Object.fromEntries(results.map(r => [r.name, Math.max(0, r.best_reward)])) },
    { metric: 'Convergence', ...Object.fromEntries(results.map(r => [r.name, Math.max(0, r.convergence_rate * 100)])) },
    { metric: 'Low Loss', ...Object.fromEntries(results.map(r => [r.name, Math.max(0, (2 - r.final_loss) * 5)])) },
    { metric: 'Episodes', ...Object.fromEntries(results.map(r => [r.name, r.total_episodes / 10])) },
  ] : [];

  // Config diff
  const selectedExperiments = experiments.filter(e => selectedIds.includes(e.id));
  const configDiffs = getConfigDiffs(selectedExperiments);

  const tooltipStyle = {
    contentStyle: {
      background: '#18181b', border: '1px solid #3f3f46',
      borderRadius: 8, fontSize: 12, color: '#fafafa',
    },
  };

  // Find best performer
  const best = results.length > 0
    ? results.reduce((a, b) => a.avg_reward > b.avg_reward ? a : b)
    : null;

  return (
    <div style={{ padding: 32 }}>
      <button onClick={onBack} style={backBtn}>All Experiments</button>
      <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 4 }}>Compare Experiments</h2>
      <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 20 }}>
        Select 2-6 experiments to compare their training curves and performance.
      </p>

      {/* Selection grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, marginBottom: 16 }}>
        {experiments.map(exp => {
          const isSelected = selectedIds.includes(exp.id);
          const colorIdx = selectedIds.indexOf(exp.id);
          return (
            <button
              key={exp.id}
              onClick={() => toggleSelect(exp.id)}
              style={{
                display: 'flex', alignItems: 'center', gap: 10,
                padding: '10px 14px', borderRadius: 'var(--radius)',
                border: isSelected ? `1px solid ${COLORS[colorIdx % COLORS.length]}` : '1px solid var(--border)',
                background: isSelected ? `color-mix(in srgb, ${COLORS[colorIdx % COLORS.length]} 6%, transparent)` : 'var(--bg-secondary)',
                color: 'var(--text-primary)', textAlign: 'left', fontSize: 13,
              }}
            >
              <div style={{
                width: 18, height: 18, borderRadius: 4,
                border: isSelected ? 'none' : '1px solid var(--border-light)',
                background: isSelected ? COLORS[colorIdx % COLORS.length] : 'transparent',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: '#fff', fontSize: 11, fontWeight: 700, flexShrink: 0,
              }}>
                {isSelected ? '✓' : ''}
              </div>
              <div style={{ minWidth: 0 }}>
                <div style={{ fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{exp.name}</div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                  {exp.training.algorithm.toUpperCase()} · {exp.environment.env_type} · {exp.metrics?.total_episodes ?? 0} eps
                </div>
              </div>
            </button>
          );
        })}
      </div>

      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 24 }}>
        <button
          onClick={runComparison}
          disabled={selectedIds.length < 2}
          style={{
            padding: '8px 20px', borderRadius: 'var(--radius)',
            border: 'none', background: selectedIds.length >= 2 ? 'var(--accent)' : 'var(--bg-tertiary)',
            color: selectedIds.length >= 2 ? '#fff' : 'var(--text-muted)',
            fontSize: 13, fontWeight: 500,
          }}
        >
          Compare ({selectedIds.length} selected)
        </button>

        {selectedIds.length >= 2 && (
          <button
            onClick={() => setShowConfigDiff(!showConfigDiff)}
            style={{
              padding: '8px 16px', borderRadius: 'var(--radius)',
              border: '1px solid var(--border)', background: showConfigDiff ? 'var(--bg-tertiary)' : 'transparent',
              color: 'var(--text-secondary)', fontSize: 12,
            }}
          >
            {showConfigDiff ? 'Hide' : 'Show'} Config Diff
          </button>
        )}
      </div>

      {/* Config diff table */}
      {showConfigDiff && selectedExperiments.length >= 2 && (
        <Card style={{ marginBottom: 16, padding: 0, overflow: 'hidden' }}>
          <div style={{ padding: '10px 14px', background: 'var(--bg-tertiary)', fontSize: 12, fontWeight: 600 }}>
            Configuration Differences
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: 'var(--bg-secondary)' }}>
                <th style={thStyle}>Parameter</th>
                {selectedExperiments.map((exp, i) => (
                  <th key={exp.id} style={thStyle}>
                    <span style={{
                      display: 'inline-block', width: 8, height: 8,
                      borderRadius: '50%', background: COLORS[i % COLORS.length],
                      marginRight: 6, verticalAlign: 'middle',
                    }} />
                    {exp.name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {configDiffs.map(diff => (
                <tr key={diff.param} style={{ borderTop: '1px solid var(--border)' }}>
                  <td style={{ ...tdStyle, color: 'var(--text-secondary)', fontWeight: 500 }}>
                    {diff.param}
                    {diff.isDifferent && (
                      <span style={{
                        marginLeft: 6, fontSize: 9, padding: '1px 5px',
                        borderRadius: 3, background: 'rgba(245,158,11,0.15)',
                        color: 'var(--amber)', fontWeight: 600,
                      }}>
                        DIFF
                      </span>
                    )}
                  </td>
                  {diff.values.map((val, i) => (
                    <td key={i} style={{
                      ...tdStyle,
                      fontVariantNumeric: 'tabular-nums',
                      fontWeight: diff.isDifferent ? 600 : 400,
                      color: diff.isDifferent ? 'var(--text-primary)' : 'var(--text-muted)',
                    }}>
                      {val}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {results.length > 0 && (
        <>
          {/* Winner banner */}
          {best && (
            <div style={{
              padding: '10px 16px', marginBottom: 16, borderRadius: 'var(--radius)',
              background: 'rgba(34,197,94,0.08)', border: '1px solid rgba(34,197,94,0.2)',
              display: 'flex', alignItems: 'center', gap: 8, fontSize: 13,
            }}>
              <span style={{ fontSize: 16 }}>&#9733;</span>
              <span style={{ color: 'var(--text-secondary)' }}>Best performer:</span>
              <span style={{ fontWeight: 600, color: 'var(--green)' }}>{best.name}</span>
              <span style={{ color: 'var(--text-muted)' }}>
                avg reward {best.avg_reward.toFixed(3)} | best {best.best_reward.toFixed(3)}
              </span>
            </div>
          )}

          {/* Summary table */}
          <Card style={{ marginBottom: 16, padding: 0, overflow: 'hidden' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: 'var(--bg-tertiary)' }}>
                  <th style={thStyle}>Experiment</th>
                  <th style={thStyle}>Algorithm</th>
                  <th style={thStyle}>Environment</th>
                  <th style={{ ...thStyle, textAlign: 'right' }}>Episodes</th>
                  <th style={{ ...thStyle, textAlign: 'right' }}>Avg Reward</th>
                  <th style={{ ...thStyle, textAlign: 'right' }}>Best Reward</th>
                  <th style={{ ...thStyle, textAlign: 'right' }}>Final Loss</th>
                  <th style={{ ...thStyle, textAlign: 'right' }}>Convergence</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r, i) => (
                  <tr key={r.id} style={{
                    borderTop: '1px solid var(--border)',
                    background: r.id === best?.id ? 'rgba(34,197,94,0.04)' : undefined,
                  }}>
                    <td style={tdStyle}>
                      <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: COLORS[i % COLORS.length], marginRight: 8 }} />
                      {r.name}
                    </td>
                    <td style={tdStyle}>{r.algorithm.toUpperCase()}</td>
                    <td style={tdStyle}>{r.environment}</td>
                    <td style={{ ...tdStyle, textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{r.total_episodes}</td>
                    <td style={{ ...tdStyle, textAlign: 'right', fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>{r.avg_reward.toFixed(3)}</td>
                    <td style={{ ...tdStyle, textAlign: 'right', color: 'var(--green)', fontVariantNumeric: 'tabular-nums' }}>{r.best_reward.toFixed(3)}</td>
                    <td style={{ ...tdStyle, textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{r.final_loss.toFixed(4)}</td>
                    <td style={{
                      ...tdStyle, textAlign: 'right', fontVariantNumeric: 'tabular-nums',
                      color: r.convergence_rate > 0 ? 'var(--green)' : 'var(--red)',
                    }}>
                      {r.convergence_rate > 0 ? '+' : ''}{r.convergence_rate.toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* Chart metric toggle */}
          <div style={{ display: 'flex', gap: 4, marginBottom: 12 }}>
            <TogglePill active={chartMetric === 'reward'} onClick={() => setChartMetric('reward')}>
              Reward Curves
            </TogglePill>
            <TogglePill active={chartMetric === 'loss'} onClick={() => setChartMetric('loss')}>
              Loss Curves
            </TogglePill>
          </div>

          {/* Charts */}
          <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 16, marginBottom: 16 }}>
            <Card>
              <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12 }}>
                {chartMetric === 'reward' ? 'Reward Curves (10-ep avg)' : 'Loss Curves'}
              </div>
              <ResponsiveContainer width="100%" height={320}>
                <LineChart data={chartMetric === 'reward' ? rewardOverlay : lossOverlay}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                  <XAxis dataKey="episode" stroke="#71717a" fontSize={11} />
                  <YAxis stroke="#71717a" fontSize={11} />
                  <Tooltip {...tooltipStyle} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {results.map((r, i) => (
                    <Line key={r.id} type="monotone" dataKey={r.name} stroke={COLORS[i % COLORS.length]} strokeWidth={2} dot={false} />
                  ))}
                  <Brush dataKey="episode" height={20} stroke="#3f3f46" fill="#18181b" travellerWidth={8} />
                </LineChart>
              </ResponsiveContainer>
            </Card>

            <Card>
              <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12 }}>Performance Radar</div>
              <ResponsiveContainer width="100%" height={320}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="#27272a" />
                  <PolarAngleAxis dataKey="metric" stroke="#71717a" fontSize={11} />
                  <PolarRadiusAxis stroke="#3f3f46" fontSize={10} />
                  {results.map((r, i) => (
                    <Radar key={r.id} name={r.name} dataKey={r.name} stroke={COLORS[i % COLORS.length]} fill={COLORS[i % COLORS.length]} fillOpacity={0.1} />
                  ))}
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                </RadarChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}

// Helper: extract config differences between experiments
function getConfigDiffs(experiments: Experiment[]) {
  if (experiments.length < 2) return [];

  const params: { key: string; extract: (e: Experiment) => string }[] = [
    { key: 'Algorithm', extract: e => e.training.algorithm.toUpperCase() },
    { key: 'Environment', extract: e => e.environment.env_type },
    { key: 'Difficulty', extract: e => e.environment.difficulty },
    { key: 'Max Turns', extract: e => String(e.environment.max_turns) },
    { key: 'Model', extract: e => e.agent.model_name },
    { key: 'Temperature', extract: e => String(e.agent.temperature) },
    { key: 'Stub Backend', extract: e => e.agent.use_stub ? 'Yes' : 'No' },
    { key: 'Episodes', extract: e => String(e.training.num_episodes) },
    { key: 'Learning Rate', extract: e => String(e.training.learning_rate) },
    { key: 'Batch Size', extract: e => String(e.training.batch_size) },
    { key: 'KL Penalty', extract: e => e.training.use_kl_penalty ? `Yes (${e.training.kl_coef})` : 'No' },
    { key: 'Clip Ratio', extract: e => String(e.training.clip_ratio) },
    { key: 'Entropy Coef', extract: e => String(e.training.entropy_coef) },
    { key: 'Gamma', extract: e => String(e.training.gamma) },
  ];

  return params.map(p => {
    const values = experiments.map(e => p.extract(e));
    const isDifferent = new Set(values).size > 1;
    return { param: p.key, values, isDifferent };
  });
}

function TogglePill({ children, active, onClick }: { children: React.ReactNode; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '5px 14px', borderRadius: 16, fontSize: 12, fontWeight: 500,
        border: active ? '1px solid var(--accent)' : '1px solid var(--border)',
        background: active ? 'rgba(99,102,241,0.1)' : 'transparent',
        color: active ? 'var(--accent-light)' : 'var(--text-secondary)',
      }}
    >
      {children}
    </button>
  );
}

const thStyle: React.CSSProperties = {
  padding: '10px 12px', textAlign: 'left', fontSize: 11, fontWeight: 600,
  color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em',
};

const tdStyle: React.CSSProperties = { padding: '10px 12px' };

const backBtn: React.CSSProperties = {
  display: 'inline-flex', alignItems: 'center', gap: 5,
  padding: '4px 10px', borderRadius: 'var(--radius)',
  border: '1px solid var(--border)', background: 'transparent',
  color: 'var(--text-secondary)', fontSize: 12, marginBottom: 12,
};
