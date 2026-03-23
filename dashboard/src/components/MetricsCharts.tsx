import { useState } from 'react';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Legend, Brush, ReferenceLine,
} from 'recharts';
import type { TrainingMetrics } from '../types';
import { Card } from './Card';

const tooltipStyle = {
  contentStyle: {
    background: '#18181b',
    border: '1px solid #3f3f46',
    borderRadius: 8,
    fontSize: 12,
    color: '#fafafa',
  },
};

function ema(data: number[], window: number): number[] {
  if (window <= 1) return data;
  const alpha = 2 / (window + 1);
  const result: number[] = [];
  let prev = data[0] ?? 0;
  for (const val of data) {
    prev = alpha * val + (1 - alpha) * prev;
    result.push(+prev.toFixed(4));
  }
  return result;
}

interface MetricsChartsProps {
  metrics: TrainingMetrics;
}

export function MetricsCharts({ metrics }: MetricsChartsProps) {
  const [smoothing, setSmoothing] = useState(10);
  const [showRaw, setShowRaw] = useState(true);

  const smoothedReward = ema(metrics.reward_history, smoothing);
  const smoothedLoss = ema(metrics.loss_history, smoothing);

  const rewardData = metrics.reward_history.map((r, i) => ({
    episode: i + 1,
    reward: r,
    smoothed: smoothedReward[i],
  }));

  const lossData = metrics.loss_history.map((l, i) => ({
    episode: i + 1,
    loss: l,
    smoothed: smoothedLoss[i],
    kl: metrics.kl_divergence[i] ?? 0,
    entropy: metrics.entropy[i] ?? 0,
  }));

  const lengthData = metrics.episode_lengths.map((l, i) => ({
    episode: i + 1,
    length: l,
  }));

  const lrData = (metrics.lr_history ?? []).map((lr, i) => ({
    episode: i + 1,
    lr,
  }));

  const advData = metrics.advantages.map((a, i) => ({
    episode: i + 1,
    advantage: a,
  }));

  return (
    <div>
      {/* Toolbar */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 16, marginBottom: 12,
        padding: '8px 14px', background: 'var(--bg-secondary)',
        border: '1px solid var(--border)', borderRadius: 'var(--radius)',
        fontSize: 12,
      }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--text-secondary)' }}>
          Smoothing:
          <input
            type="range"
            min={1} max={50} value={smoothing}
            onChange={e => setSmoothing(+e.target.value)}
            style={{ width: 100, accentColor: 'var(--accent)' }}
          />
          <span style={{ fontVariantNumeric: 'tabular-nums', width: 20, color: 'var(--text-muted)' }}>{smoothing}</span>
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: 4, color: 'var(--text-secondary)' }}>
          <input type="checkbox" checked={showRaw} onChange={e => setShowRaw(e.target.checked)} />
          Show raw
        </label>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        {/* Reward Chart */}
        <Card>
          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12 }}>Episode Rewards</div>
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={rewardData}>
              <defs>
                <linearGradient id="rewardGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#6366f1" stopOpacity={0.2} />
                  <stop offset="100%" stopColor="#6366f1" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis dataKey="episode" stroke="#71717a" fontSize={11} />
              <YAxis stroke="#71717a" fontSize={11} />
              <Tooltip {...tooltipStyle} />
              <ReferenceLine y={metrics.avg_reward} stroke="#3f3f46" strokeDasharray="4 4" label={{ value: 'avg', fill: '#71717a', fontSize: 10 }} />
              {showRaw && (
                <Area type="monotone" dataKey="reward" stroke="#6366f1" fill="url(#rewardGrad)" strokeWidth={1} dot={false} opacity={0.5} />
              )}
              <Line type="monotone" dataKey="smoothed" stroke="#22c55e" strokeWidth={2.5} dot={false} name={`EMA(${smoothing})`} />
              <Brush dataKey="episode" height={20} stroke="#3f3f46" fill="#18181b" travellerWidth={8} />
            </AreaChart>
          </ResponsiveContainer>
        </Card>

        {/* Loss Chart */}
        <Card>
          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12 }}>Training Loss & KL</div>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={lossData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis dataKey="episode" stroke="#71717a" fontSize={11} />
              <YAxis stroke="#71717a" fontSize={11} />
              <Tooltip {...tooltipStyle} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {showRaw && (
                <Line type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={1} dot={false} opacity={0.3} />
              )}
              <Line type="monotone" dataKey="smoothed" stroke="#ef4444" strokeWidth={2.5} dot={false} name={`Loss EMA`} />
              <Line type="monotone" dataKey="kl" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="KL div" />
              <Line type="monotone" dataKey="entropy" stroke="#06b6d4" strokeWidth={1.5} dot={false} />
              <Brush dataKey="episode" height={20} stroke="#3f3f46" fill="#18181b" travellerWidth={8} />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        {/* Episode Length */}
        <Card>
          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12 }}>Episode Lengths</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={lengthData.slice(-60)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis dataKey="episode" stroke="#71717a" fontSize={11} />
              <YAxis stroke="#71717a" fontSize={11} />
              <Tooltip {...tooltipStyle} />
              <Bar dataKey="length" radius={[2, 2, 0, 0]}>
                {lengthData.slice(-60).map((_, i) => {
                  const d = lengthData.slice(-60)[i];
                  const maxLen = Math.max(...lengthData.slice(-60).map(x => x.length));
                  const ratio = d ? d.length / Math.max(maxLen, 1) : 0;
                  return (
                    <rect key={i} fill={ratio > 0.8 ? '#a855f7' : ratio > 0.5 ? '#8b5cf6' : '#6d28d9'} />
                  );
                })}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* Learning Rate Schedule */}
        {lrData.length > 0 && (
          <Card>
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12 }}>Learning Rate</div>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={lrData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                <XAxis dataKey="episode" stroke="#71717a" fontSize={11} />
                <YAxis
                  stroke="#71717a" fontSize={11}
                  tickFormatter={(v: number) => v.toExponential(1)}
                />
                <Tooltip {...tooltipStyle} formatter={(v: number) => v.toExponential(3)} />
                <Line type="monotone" dataKey="lr" stroke="#14b8a6" strokeWidth={2} dot={false} name="LR" />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        )}

        {/* Advantages */}
        {lrData.length === 0 && (
          <Card>
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12 }}>Advantages</div>
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={advData}>
                <defs>
                  <linearGradient id="advGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#ec4899" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#ec4899" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                <XAxis dataKey="episode" stroke="#71717a" fontSize={11} />
                <YAxis stroke="#71717a" fontSize={11} />
                <Tooltip {...tooltipStyle} />
                <ReferenceLine y={0} stroke="#3f3f46" />
                <Area type="monotone" dataKey="advantage" stroke="#ec4899" fill="url(#advGrad)" strokeWidth={1.5} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </Card>
        )}
      </div>

      {/* Second row when LR is shown */}
      {lrData.length > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16, marginTop: 16 }}>
          <Card>
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12 }}>Advantages</div>
            <ResponsiveContainer width="100%" height={180}>
              <AreaChart data={advData}>
                <defs>
                  <linearGradient id="advGrad2" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#ec4899" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#ec4899" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                <XAxis dataKey="episode" stroke="#71717a" fontSize={11} />
                <YAxis stroke="#71717a" fontSize={11} />
                <Tooltip {...tooltipStyle} />
                <ReferenceLine y={0} stroke="#3f3f46" />
                <Area type="monotone" dataKey="advantage" stroke="#ec4899" fill="url(#advGrad2)" strokeWidth={1.5} dot={false} />
                <Brush dataKey="episode" height={16} stroke="#3f3f46" fill="#18181b" travellerWidth={8} />
              </AreaChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}
    </div>
  );
}
