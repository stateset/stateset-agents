import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from 'recharts';
import type { TrainingMetrics } from '../types';
import { Card } from './Card';

const COMPONENT_COLORS: Record<string, string> = {
  helpfulness: '#6366f1',
  coherence: '#22c55e',
  safety: '#f59e0b',
  correctness: '#ef4444',
  efficiency: '#a855f7',
  explanation: '#06b6d4',
  resolution: '#ec4899',
  empathy: '#14b8a6',
  accuracy: '#8b5cf6',
  reasoning_quality: '#f97316',
  conciseness: '#64748b',
  retrieval_quality: '#0ea5e9',
  answer_accuracy: '#d946ef',
  citation: '#84cc16',
  refusal_quality: '#fb7185',
};

interface RewardBreakdownProps {
  metrics: TrainingMetrics;
}

export function RewardBreakdown({ metrics }: RewardBreakdownProps) {
  const breakdown = metrics.reward_breakdown;
  if (!breakdown || Object.keys(breakdown).length === 0) return null;

  const components = Object.keys(breakdown);
  const maxLen = Math.max(...components.map(c => breakdown[c].length));

  const data = Array.from({ length: maxLen }, (_, i) => {
    const point: Record<string, unknown> = { episode: i + 1 };
    for (const c of components) {
      // 10-episode moving average
      const arr = breakdown[c];
      if (i < arr.length) {
        const start = Math.max(0, i - 9);
        const slice = arr.slice(start, i + 1);
        point[c] = +(slice.reduce((a, b) => a + b, 0) / slice.length).toFixed(4);
      }
    }
    return point;
  });

  // Current values (last entry)
  const current = components.map(c => {
    const arr = breakdown[c];
    return { name: c, value: arr.length > 0 ? arr[arr.length - 1] : 0 };
  }).sort((a, b) => b.value - a.value);

  const tooltipStyle = {
    contentStyle: {
      background: '#18181b', border: '1px solid #3f3f46',
      borderRadius: 8, fontSize: 12, color: '#fafafa',
    },
  };

  return (
    <Card>
      <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12 }}>Reward Components</div>

      {/* Current values bar */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 16, flexWrap: 'wrap' }}>
        {current.map(({ name, value }) => (
          <div
            key={name}
            style={{
              padding: '4px 10px',
              borderRadius: 6,
              background: `color-mix(in srgb, ${COMPONENT_COLORS[name] ?? '#6366f1'} 12%, transparent)`,
              border: `1px solid color-mix(in srgb, ${COMPONENT_COLORS[name] ?? '#6366f1'} 25%, transparent)`,
              fontSize: 11,
            }}
          >
            <span style={{ color: 'var(--text-muted)' }}>{name}: </span>
            <span style={{ fontWeight: 600, color: COMPONENT_COLORS[name] ?? 'var(--text-primary)', fontVariantNumeric: 'tabular-nums' }}>
              {value.toFixed(4)}
            </span>
          </div>
        ))}
      </div>

      {/* Stacked area chart */}
      <ResponsiveContainer width="100%" height={240}>
        <AreaChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis dataKey="episode" stroke="#71717a" fontSize={11} />
          <YAxis stroke="#71717a" fontSize={11} />
          <Tooltip {...tooltipStyle} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {components.map((c, i) => (
            <Area
              key={c}
              type="monotone"
              dataKey={c}
              stackId="1"
              stroke={COMPONENT_COLORS[c] ?? COMPONENT_COLORS[Object.keys(COMPONENT_COLORS)[i % Object.keys(COMPONENT_COLORS).length]]}
              fill={COMPONENT_COLORS[c] ?? COMPONENT_COLORS[Object.keys(COMPONENT_COLORS)[i % Object.keys(COMPONENT_COLORS).length]]}
              fillOpacity={0.3}
              strokeWidth={1.5}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    </Card>
  );
}
