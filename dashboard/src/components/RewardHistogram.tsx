import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from 'recharts';
import { Card } from './Card';

interface RewardHistogramProps {
  rewards: number[];
  bins?: number;
}

function buildHistogram(data: number[], numBins: number) {
  if (data.length === 0) return [];
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const binWidth = range / numBins;

  const bins = Array.from({ length: numBins }, (_, i) => ({
    label: (min + binWidth * i + binWidth / 2).toFixed(2),
    low: min + binWidth * i,
    high: min + binWidth * (i + 1),
    count: 0,
  }));

  for (const v of data) {
    const idx = Math.min(Math.floor((v - min) / binWidth), numBins - 1);
    bins[idx].count++;
  }

  return bins;
}

const tooltipStyle = {
  contentStyle: {
    background: '#18181b',
    border: '1px solid #3f3f46',
    borderRadius: 8,
    fontSize: 12,
    color: '#fafafa',
  },
};

export function RewardHistogram({ rewards, bins = 20 }: RewardHistogramProps) {
  if (rewards.length < 5) return null;

  const data = buildHistogram(rewards, bins);
  const maxCount = Math.max(...data.map(d => d.count));
  const mean = rewards.reduce((a, b) => a + b, 0) / rewards.length;
  const variance = rewards.reduce((a, b) => a + (b - mean) ** 2, 0) / rewards.length;
  const stddev = Math.sqrt(variance);

  return (
    <Card>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div style={{ fontSize: 13, fontWeight: 600 }}>Reward Distribution</div>
        <div style={{ display: 'flex', gap: 12, fontSize: 11 }}>
          <span style={{ color: 'var(--text-muted)' }}>
            Mean: <span style={{ color: 'var(--green)', fontWeight: 600 }}>{mean.toFixed(3)}</span>
          </span>
          <span style={{ color: 'var(--text-muted)' }}>
            Std: <span style={{ color: 'var(--amber)', fontWeight: 600 }}>{stddev.toFixed(3)}</span>
          </span>
          <span style={{ color: 'var(--text-muted)' }}>
            n={rewards.length}
          </span>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} barCategoryGap="8%">
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
          <XAxis
            dataKey="label"
            stroke="#71717a"
            fontSize={10}
            interval={Math.max(0, Math.floor(bins / 8) - 1)}
          />
          <YAxis stroke="#71717a" fontSize={11} allowDecimals={false} />
          <Tooltip
            {...tooltipStyle}
            formatter={(value: number) => [`${value} episodes`, 'Count']}
            labelFormatter={(label: string) => `Reward: ${label}`}
          />
          <Bar dataKey="count" radius={[3, 3, 0, 0]}>
            {data.map((entry, i) => {
              const ratio = entry.count / Math.max(maxCount, 1);
              const hue = 250 - ratio * 120; // indigo → green
              return <Cell key={i} fill={`hsl(${hue}, 70%, 55%)`} fillOpacity={0.8} />;
            })}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </Card>
  );
}
