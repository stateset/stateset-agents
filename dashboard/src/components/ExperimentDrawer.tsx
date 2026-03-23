import { useState, useEffect } from 'react';
import {
  X, Play, Pause, Square, Copy, Trash2, ExternalLink,
  Clock, TrendingUp, TrendingDown,
} from 'lucide-react';
import { api } from '../api';
import { useToast } from '../hooks/useToast';
import type { Experiment } from '../types';

interface ExperimentDrawerProps {
  experiment: Experiment | null;
  onClose: () => void;
  onNavigate: (exp: Experiment) => void;
  onClone: (exp: Experiment) => void;
  onRefresh: () => void;
}

export function ExperimentDrawer({ experiment, onClose, onNavigate, onClone, onRefresh }: ExperimentDrawerProps) {
  const [exp, setExp] = useState<Experiment | null>(experiment);
  const toast = useToast();

  // Refresh experiment data
  useEffect(() => {
    if (!experiment) { setExp(null); return; }
    setExp(experiment);
    const interval = setInterval(async () => {
      try {
        const fresh = await api.getExperiment(experiment.id);
        setExp(fresh);
      } catch { /* ignore */ }
    }, 3000);
    return () => clearInterval(interval);
  }, [experiment?.id]);

  if (!exp) return null;

  const m = exp.metrics;
  const convergence = m?.convergence_rate ?? 0;
  const progress = exp.training.num_episodes > 0
    ? (m?.total_episodes ?? 0) / exp.training.num_episodes
    : 0;

  const handleAction = async (action: () => Promise<unknown>, successMsg: string) => {
    try {
      await action();
      toast.success(successMsg);
      onRefresh();
      const fresh = await api.getExperiment(exp.id);
      setExp(fresh);
    } catch {
      toast.error('Action failed');
    }
  };

  return (
    <>
      {/* Backdrop */}
      <div
        onClick={onClose}
        style={{
          position: 'fixed', inset: 0, zIndex: 9000,
          background: 'rgba(0,0,0,0.4)', backdropFilter: 'blur(2px)',
        }}
      />

      {/* Drawer */}
      <div style={{
        position: 'fixed', top: 0, right: 0, bottom: 0, width: 420,
        zIndex: 9001, background: 'var(--bg-secondary)',
        borderLeft: '1px solid var(--border)',
        boxShadow: '-8px 0 24px rgba(0,0,0,0.4)',
        display: 'flex', flexDirection: 'column',
        animation: 'drawer-in 0.2s ease-out',
      }}>
        {/* Header */}
        <div style={{
          padding: '16px 20px', borderBottom: '1px solid var(--border)',
          display: 'flex', justifyContent: 'space-between', alignItems: 'start',
        }}>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 4 }}>{exp.name}</div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              <Tag color={statusColor(exp.status)}>{exp.status}</Tag>
              <Tag>{exp.training.algorithm.toUpperCase()}</Tag>
              <Tag>{exp.environment.env_type}</Tag>
              <Tag>{exp.environment.difficulty}</Tag>
            </div>
          </div>
          <button onClick={onClose} style={closeBtnStyle}><X size={16} /></button>
        </div>

        {/* Body */}
        <div style={{ flex: 1, overflow: 'auto', padding: '16px 20px' }}>
          {/* Progress */}
          <div style={{ marginBottom: 20 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>
              <span>Progress</span>
              <span>{m?.total_episodes ?? 0} / {exp.training.num_episodes} episodes ({Math.round(progress * 100)}%)</span>
            </div>
            <div style={{ height: 6, borderRadius: 3, background: 'var(--bg-tertiary)', overflow: 'hidden' }}>
              <div style={{
                width: `${Math.min(100, progress * 100)}%`, height: '100%', borderRadius: 3,
                background: progress >= 1 ? 'var(--green)' : 'var(--accent)',
                transition: 'width 0.3s',
              }} />
            </div>
          </div>

          {/* Key metrics */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginBottom: 20 }}>
            <MiniCard label="Avg Reward" value={(m?.avg_reward ?? 0).toFixed(3)} color="var(--text-primary)" />
            <MiniCard label="Best Reward" value={(m?.best_reward ?? 0).toFixed(3)} color="var(--green)" />
            <MiniCard label="Convergence" value={`${convergence > 0 ? '+' : ''}${convergence.toFixed(4)}`}
              color={convergence > 0 ? 'var(--green)' : 'var(--red)'}
              icon={convergence > 0 ? TrendingUp : TrendingDown}
            />
          </div>

          {/* Sparkline */}
          {(m?.reward_history?.length ?? 0) > 0 && (
            <div style={{ marginBottom: 20 }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>Reward History</div>
              <div style={{ display: 'flex', gap: 1, alignItems: 'end', height: 48, background: 'var(--bg-tertiary)', borderRadius: 6, padding: 4 }}>
                {(m?.reward_history ?? []).slice(-80).map((r, i) => (
                  <div
                    key={i}
                    style={{
                      flex: 1, borderRadius: '1px 1px 0 0',
                      height: `${Math.max(4, (r / Math.max(1, m?.best_reward ?? 1)) * 100)}%`,
                      background: 'var(--accent)', opacity: 0.6,
                    }}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Config summary */}
          <div style={{ marginBottom: 20 }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 8 }}>Configuration</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
              <ConfigItem label="Model" value={exp.agent.model_name} />
              <ConfigItem label="Temperature" value={String(exp.agent.temperature)} />
              <ConfigItem label="Learning Rate" value={String(exp.training.learning_rate)} />
              <ConfigItem label="Batch Size" value={String(exp.training.batch_size)} />
              <ConfigItem label="KL Penalty" value={exp.training.use_kl_penalty ? `Yes (${exp.training.kl_coef})` : 'No'} />
              <ConfigItem label="Stub Backend" value={exp.agent.use_stub ? 'Yes' : 'No'} />
            </div>
          </div>

          {/* Timestamps */}
          <div style={{ fontSize: 11, color: 'var(--text-muted)', display: 'flex', gap: 16 }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <Clock size={10} /> Created {new Date(exp.created_at * 1000).toLocaleString()}
            </span>
          </div>
        </div>

        {/* Footer actions */}
        <div style={{
          padding: '12px 20px', borderTop: '1px solid var(--border)',
          display: 'flex', gap: 8,
        }}>
          {(exp.status === 'created' || exp.status === 'paused') && (
            <ActionBtn onClick={() => handleAction(() => api.startExperiment(exp.id), 'Started')} color="var(--green)">
              <Play size={13} /> Start
            </ActionBtn>
          )}
          {exp.status === 'running' && (
            <>
              <ActionBtn onClick={() => handleAction(() => api.pauseExperiment(exp.id), 'Paused')} color="var(--amber)">
                <Pause size={13} /> Pause
              </ActionBtn>
              <ActionBtn onClick={() => handleAction(() => api.stopExperiment(exp.id), 'Stopped')} color="var(--red)">
                <Square size={11} /> Stop
              </ActionBtn>
            </>
          )}
          <ActionBtn onClick={() => { onClone(exp); onClose(); }}>
            <Copy size={13} /> Clone
          </ActionBtn>
          <div style={{ flex: 1 }} />
          <ActionBtn onClick={() => { onNavigate(exp); onClose(); }} color="var(--accent)">
            <ExternalLink size={13} /> Open
          </ActionBtn>
          <ActionBtn
            onClick={() => handleAction(() => api.deleteExperiment(exp.id), 'Deleted').then(onClose)}
            color="var(--red)"
          >
            <Trash2 size={13} />
          </ActionBtn>
        </div>
      </div>
    </>
  );
}

function Tag({ children, color }: { children: React.ReactNode; color?: string }) {
  return (
    <span style={{
      padding: '2px 8px', borderRadius: 6, fontSize: 10, fontWeight: 500,
      background: color ? `color-mix(in srgb, ${color} 15%, transparent)` : 'var(--bg-tertiary)',
      color: color ?? 'var(--text-secondary)',
    }}>
      {children}
    </span>
  );
}

function MiniCard({ label, value, color, icon: Icon }: {
  label: string; value: string; color: string; icon?: React.ElementType;
}) {
  return (
    <div style={{
      padding: '10px 12px', borderRadius: 8,
      background: 'var(--bg-tertiary)', border: '1px solid var(--border)',
    }}>
      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 16, fontWeight: 700, color, fontVariantNumeric: 'tabular-nums', display: 'flex', alignItems: 'center', gap: 4 }}>
        {Icon && <Icon size={14} />}
        {value}
      </div>
    </div>
  );
}

function ConfigItem({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0' }}>
      <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{label}</span>
      <span style={{ fontSize: 11, fontWeight: 500, color: 'var(--text-primary)' }}>{value}</span>
    </div>
  );
}

function ActionBtn({ children, onClick, color }: {
  children: React.ReactNode; onClick: () => void; color?: string;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 5,
        padding: '6px 12px', borderRadius: 'var(--radius)',
        border: `1px solid ${color ? `color-mix(in srgb, ${color} 30%, transparent)` : 'var(--border)'}`,
        background: color ? `color-mix(in srgb, ${color} 8%, transparent)` : 'transparent',
        color: color ?? 'var(--text-secondary)',
        fontSize: 12, fontWeight: 500,
      }}
    >
      {children}
    </button>
  );
}

function statusColor(s: string) {
  return { running: 'var(--green)', paused: 'var(--amber)', completed: 'var(--accent)', failed: 'var(--red)' }[s] ?? 'var(--text-muted)';
}

const closeBtnStyle: React.CSSProperties = {
  display: 'flex', alignItems: 'center', justifyContent: 'center',
  width: 28, height: 28, borderRadius: 'var(--radius)',
  border: '1px solid var(--border)', background: 'transparent',
  color: 'var(--text-muted)',
};
