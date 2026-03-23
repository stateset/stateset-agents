import { useState, useEffect } from 'react';
import {
  Activity, Wifi, WifiOff, Play, Pause, RotateCcw, List, BarChart3,
  Square, Save, Download,
} from 'lucide-react';
import { Card, StatCard } from './Card';
import { MetricsCharts } from './MetricsCharts';
import { EpisodeViewer } from './EpisodeViewer';
import { EpisodeBrowser } from './EpisodeBrowser';
import { RewardBreakdown } from './RewardBreakdown';
import { RewardHistogram } from './RewardHistogram';
import { TrainingConsole } from './TrainingConsole';
import { useExperimentWs } from '../hooks/useExperimentWs';
import { useToast } from '../hooks/useToast';
import { api } from '../api';
import type { Experiment, TrainingMetrics } from '../types';

interface LiveMonitorProps {
  experiment: Experiment | null;
  onBack: () => void;
}

type Tab = 'overview' | 'episodes' | 'config';

export function LiveMonitor({ experiment: initialExp, onBack }: LiveMonitorProps) {
  const [experiment, setExperiment] = useState(initialExp);
  const [tab, setTab] = useState<Tab>('overview');
  const { metrics: wsMetrics, latestEpisode, connected } = useExperimentWs(experiment?.id ?? null);
  const toast = useToast();

  // Editable config state
  const [editingConfig, setEditingConfig] = useState(false);
  const [editNumEpisodes, setEditNumEpisodes] = useState(0);
  const [editLearningRate, setEditLearningRate] = useState(0);
  const [editBatchSize, setEditBatchSize] = useState(0);
  const [editTemperature, setEditTemperature] = useState(0);

  useEffect(() => {
    if (!experiment?.id) return;
    const interval = setInterval(async () => {
      try {
        const exp = await api.getExperiment(experiment.id);
        setExperiment(exp);
      } catch { /* ignore */ }
    }, 3000);
    return () => clearInterval(interval);
  }, [experiment?.id]);

  // Sync editable config when experiment changes
  useEffect(() => {
    if (!experiment) return;
    setEditNumEpisodes(experiment.training.num_episodes);
    setEditLearningRate(experiment.training.learning_rate);
    setEditBatchSize(experiment.training.batch_size);
    setEditTemperature(experiment.agent.temperature);
  }, [experiment?.id]);

  const metrics: TrainingMetrics = wsMetrics ?? experiment?.metrics ?? {
    total_episodes: 0, total_reward: 0, avg_reward: 0, best_reward: 0,
    reward_history: [], episode_lengths: [], loss_history: [],
    lr_history: [], kl_divergence: [], entropy: [], advantages: [],
  };

  if (!experiment) {
    return (
      <div style={{ padding: 32, textAlign: 'center' }}>
        <Activity size={40} style={{ color: 'var(--text-muted)', marginBottom: 12 }} />
        <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 4 }}>No experiment selected</div>
        <div style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 16 }}>
          Select an experiment from the dashboard or create a new one.
        </div>
        <button onClick={onBack} style={backBtnStyle}>Go to Dashboard</button>
      </div>
    );
  }

  const handleStart = async () => {
    try {
      await api.startExperiment(experiment.id);
      const exp = await api.getExperiment(experiment.id);
      setExperiment(exp);
      toast.success('Training started');
    } catch {
      toast.error('Failed to start training');
    }
  };

  const handlePause = async () => {
    try {
      await api.pauseExperiment(experiment.id);
      const exp = await api.getExperiment(experiment.id);
      setExperiment(exp);
      toast.info('Training paused');
    } catch {
      toast.error('Failed to pause training');
    }
  };

  const handleStop = async () => {
    try {
      await api.stopExperiment(experiment.id);
      const exp = await api.getExperiment(experiment.id);
      setExperiment(exp);
      toast.info('Training stopped');
    } catch {
      toast.error('Failed to stop training');
    }
  };

  const handleSaveConfig = async () => {
    try {
      const patch: Record<string, number> = {};
      if (editNumEpisodes !== experiment.training.num_episodes) patch.num_episodes = editNumEpisodes;
      if (editLearningRate !== experiment.training.learning_rate) patch.learning_rate = editLearningRate;
      if (editBatchSize !== experiment.training.batch_size) patch.batch_size = editBatchSize;
      if (editTemperature !== experiment.agent.temperature) patch.temperature = editTemperature;
      if (Object.keys(patch).length === 0) {
        setEditingConfig(false);
        return;
      }
      await api.patchExperimentConfig(experiment.id, patch);
      const exp = await api.getExperiment(experiment.id);
      setExperiment(exp);
      setEditingConfig(false);
      toast.success('Config updated');
    } catch {
      toast.error('Failed to update config');
    }
  };

  const handleExport = async (format: 'json' | 'csv') => {
    try {
      const data = await api.exportExperiment(experiment.id, format);
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${experiment.name}_export.${format}`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success(`Exported as ${format.toUpperCase()}`);
    } catch {
      toast.error('Export failed');
    }
  };

  const status = experiment.status;
  const convergence = metrics.convergence_rate ?? 0;

  return (
    <div style={{ padding: 32 }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: 16 }}>
        <div>
          <button onClick={onBack} style={{ ...backBtnStyle, marginBottom: 8 }}>
            <RotateCcw size={12} /> All Experiments
          </button>
          <h2 style={{ fontSize: 20, fontWeight: 700 }}>{experiment.name}</h2>
          <div style={{ display: 'flex', gap: 8, marginTop: 6 }}>
            <Tag>{experiment.training.algorithm.toUpperCase()}</Tag>
            <Tag>{experiment.environment.env_type}</Tag>
            <Tag>{experiment.environment.difficulty}</Tag>
            <Tag color={statusColor(status)}>{status}</Tag>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            display: 'flex', alignItems: 'center', gap: 4,
            fontSize: 11, color: connected ? 'var(--green)' : 'var(--text-muted)',
          }}>
            {connected ? <Wifi size={12} /> : <WifiOff size={12} />}
            {connected ? 'Live' : 'Disconnected'}
          </div>

          {/* Export */}
          <button
            onClick={() => handleExport('json')}
            style={{ ...smallBtnStyle, color: 'var(--text-secondary)' }}
            title="Export JSON"
          >
            <Download size={13} />
          </button>

          {(status === 'created' || status === 'paused') && (
            <button onClick={handleStart} style={actionBtnStyle}>
              <Play size={14} /> Start
            </button>
          )}
          {status === 'running' && (
            <>
              <button onClick={handlePause} style={{ ...actionBtnStyle, background: 'var(--amber)' }}>
                <Pause size={14} /> Pause
              </button>
              <button onClick={handleStop} style={{ ...actionBtnStyle, background: 'var(--red)' }}>
                <Square size={12} /> Stop
              </button>
            </>
          )}
        </div>
      </div>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 2, marginBottom: 20, borderBottom: '1px solid var(--border)', paddingBottom: 0 }}>
        {([
          { id: 'overview', label: 'Overview', icon: BarChart3 },
          { id: 'episodes', label: 'Episodes', icon: List },
          { id: 'config', label: 'Config', icon: Activity },
        ] as const).map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '8px 16px', border: 'none',
              borderBottom: tab === t.id ? '2px solid var(--accent)' : '2px solid transparent',
              background: 'transparent',
              color: tab === t.id ? 'var(--text-primary)' : 'var(--text-muted)',
              fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
              marginBottom: -1,
            }}
          >
            <t.icon size={14} />
            {t.label}
          </button>
        ))}
      </div>

      {/* TAB: Overview */}
      {tab === 'overview' && (
        <>
          {/* Stats row */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 12, marginBottom: 20 }}>
            <StatCard label="Episodes" value={metrics.total_episodes} color="var(--accent-light)" />
            <StatCard label="Avg Reward" value={metrics.avg_reward.toFixed(3)} color="var(--text-primary)" />
            <StatCard
              label="Best Reward"
              value={(metrics.best_reward === -Infinity ? 0 : metrics.best_reward).toFixed(3)}
              color="var(--green)"
            />
            <StatCard
              label="Latest Loss"
              value={(metrics.loss_history.length > 0 ? metrics.loss_history[metrics.loss_history.length - 1] : 0).toFixed(4)}
              color="var(--red)"
            />
            <StatCard
              label="Entropy"
              value={(metrics.entropy.length > 0 ? metrics.entropy[metrics.entropy.length - 1] : 0).toFixed(4)}
              color="var(--cyan)"
            />
            <StatCard
              label="Convergence"
              value={(convergence > 0 ? '+' : '') + convergence.toFixed(4)}
              color={convergence > 0 ? 'var(--green)' : 'var(--red)'}
              sub={convergence > 0.01 ? 'Improving' : convergence < -0.01 ? 'Degrading' : 'Stable'}
            />
          </div>

          {/* Charts */}
          {metrics.reward_history.length > 0 ? (
            <>
              <MetricsCharts metrics={metrics} />
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16 }}>
                <RewardHistogram rewards={metrics.reward_history} />
                <RewardBreakdown metrics={metrics} />
              </div>
            </>
          ) : (
            <Card style={{ textAlign: 'center', padding: 60 }}>
              <Activity size={32} style={{ color: 'var(--text-muted)', marginBottom: 8 }} />
              <div style={{ fontSize: 14, color: 'var(--text-muted)' }}>
                {status === 'running' ? 'Waiting for first episode...' : 'Start the experiment to see metrics'}
              </div>
            </Card>
          )}

          {/* Training console */}
          <div style={{ marginTop: 16 }}>
            <TrainingConsole experimentId={experiment.id} />
          </div>

          {/* Latest episode */}
          {latestEpisode && (
            <div style={{ marginTop: 16 }}>
              <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 10 }}>Latest Episode</div>
              <EpisodeViewer episode={latestEpisode} />
            </div>
          )}
        </>
      )}

      {/* TAB: Episodes */}
      {tab === 'episodes' && (
        <EpisodeBrowser experimentId={experiment.id} />
      )}

      {/* TAB: Config */}
      {tab === 'config' && (
        <div>
          <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 12 }}>
            {editingConfig ? (
              <div style={{ display: 'flex', gap: 6 }}>
                <button
                  onClick={() => setEditingConfig(false)}
                  style={smallBtnStyle}
                >
                  Cancel
                </button>
                <button
                  onClick={handleSaveConfig}
                  style={{ ...smallBtnStyle, background: 'var(--green)', color: '#fff', border: 'none' }}
                >
                  <Save size={12} /> Save
                </button>
              </div>
            ) : (
              <button
                onClick={() => setEditingConfig(true)}
                style={smallBtnStyle}
              >
                Edit Config
              </button>
            )}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
            <Card>
              <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--accent-light)', marginBottom: 8 }}>Environment</div>
              <ConfigRow label="Type" value={experiment.environment.env_type} />
              <ConfigRow label="Max Turns" value={String(experiment.environment.max_turns)} />
              <ConfigRow label="Difficulty" value={experiment.environment.difficulty} />
              <div style={{ marginTop: 10, fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 4 }}>
                Reward Weights
              </div>
              {Object.entries(experiment.environment.reward_weights ?? {}).map(([k, v]) => (
                <ConfigRow key={k} label={k} value={String(v)} />
              ))}
            </Card>
            <Card>
              <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--purple)', marginBottom: 8 }}>Agent</div>
              <ConfigRow label="Model" value={experiment.agent.model_name} />
              {editingConfig ? (
                <EditableConfigRow
                  label="Temperature"
                  value={editTemperature}
                  onChange={setEditTemperature}
                  type="number"
                  step={0.05}
                />
              ) : (
                <ConfigRow label="Temperature" value={String(experiment.agent.temperature)} />
              )}
              <ConfigRow label="Top P" value={String(experiment.agent.top_p)} />
              <ConfigRow label="Max Tokens" value={String(experiment.agent.max_new_tokens)} />
              <ConfigRow label="Memory Window" value={String(experiment.agent.memory_window)} />
              <ConfigRow label="Stub Backend" value={experiment.agent.use_stub ? 'Yes' : 'No'} />
            </Card>
            <Card>
              <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--green)', marginBottom: 8 }}>Training</div>
              <ConfigRow label="Algorithm" value={experiment.training.algorithm.toUpperCase()} />
              {editingConfig ? (
                <>
                  <EditableConfigRow label="Episodes" value={editNumEpisodes} onChange={setEditNumEpisodes} type="number" />
                  <EditableConfigRow label="Learning Rate" value={editLearningRate} onChange={setEditLearningRate} type="number" step={1e-6} />
                  <EditableConfigRow label="Batch Size" value={editBatchSize} onChange={setEditBatchSize} type="number" />
                </>
              ) : (
                <>
                  <ConfigRow label="Episodes" value={String(experiment.training.num_episodes)} />
                  <ConfigRow label="Learning Rate" value={String(experiment.training.learning_rate)} />
                  <ConfigRow label="Batch Size" value={String(experiment.training.batch_size)} />
                </>
              )}
              <ConfigRow label="Generations" value={String(experiment.training.num_generations)} />
              <ConfigRow label="KL Penalty" value={experiment.training.use_kl_penalty ? `Yes (${experiment.training.kl_coef})` : 'No'} />
              <ConfigRow label="Clip Ratio" value={String(experiment.training.clip_ratio)} />
              <ConfigRow label="Entropy Coef" value={String(experiment.training.entropy_coef)} />
              <ConfigRow label="Gamma" value={String(experiment.training.gamma)} />
              <ConfigRow label="Norm. Advantages" value={experiment.training.normalize_advantages ? 'Yes' : 'No'} />
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}

function Tag({ children, color }: { children: React.ReactNode; color?: string }) {
  return (
    <span style={{
      padding: '2px 8px',
      borderRadius: 6,
      background: color ? `color-mix(in srgb, ${color} 15%, transparent)` : 'var(--bg-tertiary)',
      color: color ?? 'var(--text-secondary)',
      fontSize: 11,
      fontWeight: 500,
    }}>
      {children}
    </span>
  );
}

function ConfigRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
      <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{label}</span>
      <span style={{ fontSize: 12, fontWeight: 500 }}>{value}</span>
    </div>
  );
}

function EditableConfigRow({ label, value, onChange, type, step }: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  type: string;
  step?: number;
}) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
      <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{label}</span>
      <input
        type={type}
        value={value}
        onChange={e => onChange(+e.target.value)}
        step={step}
        style={{
          width: 90, padding: '2px 6px', borderRadius: 4,
          border: '1px solid var(--accent)', background: 'var(--bg-primary)',
          color: 'var(--text-primary)', fontSize: 12, textAlign: 'right',
          outline: 'none',
        }}
      />
    </div>
  );
}

function statusColor(s: string) {
  return { running: 'var(--green)', paused: 'var(--amber)', completed: 'var(--accent)', failed: 'var(--red)' }[s] ?? 'var(--text-muted)';
}

const backBtnStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 5,
  padding: '4px 10px',
  borderRadius: 'var(--radius)',
  border: '1px solid var(--border)',
  background: 'transparent',
  color: 'var(--text-secondary)',
  fontSize: 12,
};

const actionBtnStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 6,
  padding: '8px 16px',
  borderRadius: 'var(--radius)',
  border: 'none',
  background: 'var(--green)',
  color: '#fff',
  fontSize: 13,
  fontWeight: 500,
};

const smallBtnStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 4,
  padding: '5px 12px',
  borderRadius: 'var(--radius)',
  border: '1px solid var(--border)',
  background: 'transparent',
  color: 'var(--text-secondary)',
  fontSize: 12,
  fontWeight: 500,
};
