import { useState, useEffect, useMemo } from 'react';
import {
  Play, Pause, Trash2, Clock, Zap, BarChart3, Copy, Square,
  Search, Filter, X, Info,
} from 'lucide-react';
import { Card, StatCard } from './Card';
import { SkeletonCard, SkeletonRow } from './Skeleton';
import { ActivityFeed } from './ActivityFeed';
import { ExperimentDrawer } from './ExperimentDrawer';
import { BatchActions } from './BatchActions';
import { Onboarding } from './Onboarding';
import { api } from '../api';
import { useToast } from '../hooks/useToast';
import type { Experiment } from '../types';

interface DashboardProps {
  onSelectExperiment: (exp: Experiment) => void;
  onNavigate: (view: string) => void;
  onClone?: (exp: Experiment) => void;
}

type StatusFilter = 'all' | 'running' | 'paused' | 'completed' | 'created' | 'failed';
type AlgoFilter = 'all' | 'grpo' | 'gspo' | 'ppo' | 'dapo' | 'vapo';

export function Dashboard({ onSelectExperiment, onNavigate, onClone }: DashboardProps) {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(true);
  const toast = useToast();

  // Search & filter state
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');
  const [algoFilter, setAlgoFilter] = useState<AlgoFilter>('all');
  const [showFilters, setShowFilters] = useState(false);

  // Batch selection & drawer
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [drawerExperiment, setDrawerExperiment] = useState<Experiment | null>(null);

  const toggleSelect = (id: string) => {
    setSelectedIds(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]);
  };

  const fetchExperiments = async () => {
    try {
      const exps = await api.listExperiments();
      setExperiments(exps);
    } catch {
      // API not running — show empty state
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchExperiments();
    const interval = setInterval(fetchExperiments, 5000);
    return () => clearInterval(interval);
  }, []);

  // Filtered experiments
  const filtered = useMemo(() => {
    let result = experiments;
    if (search) {
      const q = search.toLowerCase();
      result = result.filter(e =>
        e.name.toLowerCase().includes(q) ||
        e.environment.env_type.toLowerCase().includes(q) ||
        e.training.algorithm.toLowerCase().includes(q)
      );
    }
    if (statusFilter !== 'all') {
      result = result.filter(e => e.status === statusFilter);
    }
    if (algoFilter !== 'all') {
      result = result.filter(e => e.training.algorithm === algoFilter);
    }
    return result;
  }, [experiments, search, statusFilter, algoFilter]);

  const running = experiments.filter(e => e.status === 'running').length;
  const completed = experiments.filter(e => e.status === 'completed').length;
  const bestReward = experiments.reduce(
    (best, e) => Math.max(best, e.metrics?.best_reward ?? 0), 0
  );
  const totalEpisodes = experiments.reduce(
    (sum, e) => sum + (e.metrics?.total_episodes ?? 0), 0
  );

  const hasActiveFilters = statusFilter !== 'all' || algoFilter !== 'all' || search !== '';

  const handleStart = async (id: string) => {
    try {
      await api.startExperiment(id);
      toast.success('Experiment started');
      fetchExperiments();
    } catch {
      toast.error('Failed to start experiment');
    }
  };

  const handlePause = async (id: string) => {
    try {
      await api.pauseExperiment(id);
      toast.info('Experiment paused');
      fetchExperiments();
    } catch {
      toast.error('Failed to pause experiment');
    }
  };

  const handleStop = async (id: string) => {
    try {
      await api.stopExperiment(id);
      toast.info('Experiment stopped');
      fetchExperiments();
    } catch {
      toast.error('Failed to stop experiment');
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await api.deleteExperiment(id);
      toast.success('Experiment deleted');
      fetchExperiments();
    } catch {
      toast.error('Failed to delete experiment');
    }
  };

  const handleClone = async (exp: Experiment) => {
    if (onClone) {
      onClone(exp);
    } else {
      try {
        await api.cloneExperiment(exp.id);
        toast.success('Experiment cloned');
        fetchExperiments();
      } catch {
        toast.error('Failed to clone experiment');
      }
    }
  };

  const clearFilters = () => {
    setSearch('');
    setStatusFilter('all');
    setAlgoFilter('all');
  };

  return (
    <div style={{ padding: 32 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 700 }}>AI Training Lab</h1>
          <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginTop: 2 }}>
            RL gym environment for stateset-agents
          </p>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', padding: '8px 0', display: 'flex', gap: 12 }}>
            <span title="Press N">N new</span>
            <span title="Press D">D dashboard</span>
            <span title="Cmd+K">&#8984;K search</span>
          </div>
          <button
            onClick={() => onNavigate('create')}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 6,
              padding: '8px 16px',
              borderRadius: 'var(--radius)',
              border: 'none',
              background: 'var(--accent)',
              color: '#fff',
              fontSize: 13,
              fontWeight: 500,
            }}
          >
            <Zap size={14} /> New Experiment
          </button>
        </div>
      </div>

      {/* Stat cards */}
      {loading ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 24 }}>
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 24 }}>
          <StatCard label="Total Experiments" value={experiments.length} sub={`${running} running`} color="var(--accent-light)" />
          <StatCard label="Total Episodes" value={totalEpisodes.toLocaleString()} color="var(--text-primary)" />
          <StatCard label="Completed" value={completed} color="var(--green)" />
          <StatCard label="Best Reward" value={bestReward.toFixed(3)} color="var(--amber)" />
        </div>
      )}

      {/* Search & Filter bar */}
      {!loading && experiments.length > 0 && (
        <div style={{
          display: 'flex', gap: 8, marginBottom: 16, alignItems: 'center',
        }}>
          <div style={{
            flex: 1, display: 'flex', alignItems: 'center', gap: 8,
            padding: '7px 12px', borderRadius: 'var(--radius)',
            border: '1px solid var(--border)', background: 'var(--bg-secondary)',
          }}>
            <Search size={14} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search experiments..."
              style={{
                flex: 1, border: 'none', background: 'none', outline: 'none',
                color: 'var(--text-primary)', fontSize: 13,
              }}
            />
            {search && (
              <button onClick={() => setSearch('')} style={{ ...clearBtnStyle }}>
                <X size={12} />
              </button>
            )}
          </div>

          <button
            onClick={() => setShowFilters(!showFilters)}
            style={{
              display: 'flex', alignItems: 'center', gap: 5,
              padding: '7px 12px', borderRadius: 'var(--radius)',
              border: hasActiveFilters ? '1px solid var(--accent)' : '1px solid var(--border)',
              background: hasActiveFilters ? 'rgba(99,102,241,0.08)' : 'var(--bg-secondary)',
              color: hasActiveFilters ? 'var(--accent-light)' : 'var(--text-secondary)',
              fontSize: 12, fontWeight: 500,
            }}
          >
            <Filter size={13} />
            Filters
            {hasActiveFilters && (
              <span style={{
                width: 16, height: 16, borderRadius: '50%',
                background: 'var(--accent)', color: '#fff',
                fontSize: 9, fontWeight: 700,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                {(statusFilter !== 'all' ? 1 : 0) + (algoFilter !== 'all' ? 1 : 0)}
              </span>
            )}
          </button>
        </div>
      )}

      {/* Filter pills */}
      {showFilters && (
        <div style={{
          padding: '12px 14px', marginBottom: 16,
          background: 'var(--bg-secondary)', border: '1px solid var(--border)',
          borderRadius: 'var(--radius)',
          display: 'flex', gap: 20, alignItems: 'center', flexWrap: 'wrap',
        }}>
          <div>
            <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase' }}>
              Status
            </div>
            <div style={{ display: 'flex', gap: 4 }}>
              {(['all', 'running', 'paused', 'completed', 'created', 'failed'] as StatusFilter[]).map(s => (
                <FilterPill key={s} active={statusFilter === s} onClick={() => setStatusFilter(s)}>
                  {s === 'all' ? 'All' : s.charAt(0).toUpperCase() + s.slice(1)}
                </FilterPill>
              ))}
            </div>
          </div>
          <div>
            <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase' }}>
              Algorithm
            </div>
            <div style={{ display: 'flex', gap: 4 }}>
              {(['all', 'grpo', 'gspo', 'ppo', 'dapo', 'vapo'] as AlgoFilter[]).map(a => (
                <FilterPill key={a} active={algoFilter === a} onClick={() => setAlgoFilter(a)}>
                  {a === 'all' ? 'All' : a.toUpperCase()}
                </FilterPill>
              ))}
            </div>
          </div>
          {hasActiveFilters && (
            <button onClick={clearFilters} style={{
              ...clearBtnStyle, fontSize: 11, padding: '4px 10px',
              marginLeft: 'auto',
            }}>
              Clear all
            </button>
          )}
        </div>
      )}

      {/* Results count when filtered */}
      {hasActiveFilters && !loading && (
        <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12 }}>
          Showing {filtered.length} of {experiments.length} experiments
        </div>
      )}

      {/* Experiment list */}
      {loading ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <SkeletonRow />
          <SkeletonRow />
          <SkeletonRow />
        </div>
      ) : experiments.length === 0 ? (
        <Onboarding onNavigate={onNavigate} />
      ) : filtered.length === 0 ? (
        <Card>
          <div style={{ textAlign: 'center', padding: 40 }}>
            <Search size={24} style={{ color: 'var(--text-muted)', marginBottom: 8 }} />
            <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>No experiments match your filters</div>
            <button onClick={clearFilters} style={{
              marginTop: 12, padding: '5px 14px', borderRadius: 'var(--radius)',
              border: '1px solid var(--border)', background: 'transparent',
              color: 'var(--text-secondary)', fontSize: 12,
            }}>
              Clear filters
            </button>
          </div>
        </Card>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {filtered.map(exp => {
            const progress = exp.training.num_episodes > 0
              ? (exp.metrics?.total_episodes ?? 0) / exp.training.num_episodes
              : 0;
            return (
              <Card
                key={exp.id}
                hover
                onClick={() => onSelectExperiment(exp)}
                style={{ padding: 16, cursor: 'pointer' }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  {/* Batch checkbox */}
                  <div onClick={e => e.stopPropagation()}>
                    <input
                      type="checkbox"
                      checked={selectedIds.includes(exp.id)}
                      onChange={() => toggleSelect(exp.id)}
                      style={{ accentColor: 'var(--accent)', cursor: 'pointer', width: 14, height: 14 }}
                    />
                  </div>
                  <StatusDot status={exp.status} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 2 }}>{exp.name}</div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', display: 'flex', gap: 12 }}>
                      <span>{exp.training.algorithm.toUpperCase()}</span>
                      <span>{exp.environment.env_type}</span>
                      <span>{exp.environment.difficulty}</span>
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 3 }}>
                        <Clock size={10} /> {new Date(exp.created_at * 1000).toLocaleDateString()}
                      </span>
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
                    <MiniStat label="Episodes" value={`${exp.metrics?.total_episodes ?? 0}/${exp.training.num_episodes}`} />
                    <MiniStat label="Avg Reward" value={(exp.metrics?.avg_reward ?? 0).toFixed(3)} />
                    <MiniStat label="Best" value={(exp.metrics?.best_reward ?? 0).toFixed(3)} color="var(--green)" />

                    <div style={{ display: 'flex', gap: 4 }} onClick={e => e.stopPropagation()}>
                      <IconButton onClick={() => setDrawerExperiment(exp)} title="Quick view">
                        <Info size={14} />
                      </IconButton>
                      {(exp.status === 'created' || exp.status === 'paused') && (
                        <IconButton onClick={() => handleStart(exp.id)} title="Start">
                          <Play size={14} />
                        </IconButton>
                      )}
                      {exp.status === 'running' && (
                        <>
                          <IconButton onClick={() => handlePause(exp.id)} title="Pause">
                            <Pause size={14} />
                          </IconButton>
                          <IconButton onClick={() => handleStop(exp.id)} title="Stop">
                            <Square size={12} />
                          </IconButton>
                        </>
                      )}
                      <IconButton onClick={() => handleClone(exp)} title="Clone">
                        <Copy size={14} />
                      </IconButton>
                      <IconButton onClick={() => handleDelete(exp.id)} title="Delete" danger>
                        <Trash2 size={14} />
                      </IconButton>
                    </div>
                  </div>
                </div>

                {/* Progress bar + sparkline */}
                <div style={{ marginTop: 10, display: 'flex', gap: 12, alignItems: 'center' }}>
                  {/* Progress bar */}
                  <div style={{
                    width: 100, height: 4, borderRadius: 2,
                    background: 'var(--bg-tertiary)', flexShrink: 0,
                    overflow: 'hidden',
                  }}>
                    <div style={{
                      width: `${Math.min(100, progress * 100)}%`,
                      height: '100%',
                      borderRadius: 2,
                      background: progress >= 1 ? 'var(--green)' : 'var(--accent)',
                      transition: 'width 0.3s',
                    }} />
                  </div>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)', flexShrink: 0, fontVariantNumeric: 'tabular-nums' }}>
                    {Math.round(progress * 100)}%
                  </span>

                  {/* Mini reward sparkline */}
                  {(exp.metrics?.reward_history?.length ?? 0) > 0 && (
                    <div style={{ display: 'flex', gap: 1, alignItems: 'end', height: 20, flex: 1 }}>
                      {(exp.metrics.reward_history ?? []).slice(-60).map((r, i) => (
                        <div
                          key={i}
                          style={{
                            flex: 1,
                            height: `${Math.max(4, (r / Math.max(1, exp.metrics.best_reward)) * 100)}%`,
                            background: 'var(--accent)',
                            opacity: 0.5,
                            borderRadius: '1px 1px 0 0',
                          }}
                        />
                      ))}
                    </div>
                  )}
                </div>
              </Card>
            );
          })}
        </div>
      )}

      {/* Activity Feed */}
      {!loading && experiments.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <ActivityFeed experiments={experiments} onSelect={onSelectExperiment} />
        </div>
      )}

      {/* Experiment quick-view drawer */}
      <ExperimentDrawer
        experiment={drawerExperiment}
        onClose={() => setDrawerExperiment(null)}
        onNavigate={onSelectExperiment}
        onClone={handleClone}
        onRefresh={fetchExperiments}
      />

      {/* Batch actions bar */}
      <BatchActions
        selectedIds={selectedIds}
        onClear={() => setSelectedIds([])}
        onRefresh={fetchExperiments}
      />
    </div>
  );
}

function StatusDot({ status }: { status: string }) {
  const color = {
    created: 'var(--text-muted)',
    running: 'var(--green)',
    paused: 'var(--amber)',
    completed: 'var(--accent)',
    failed: 'var(--red)',
  }[status] ?? 'var(--text-muted)';

  return (
    <div style={{
      width: 8, height: 8,
      borderRadius: '50%',
      background: color,
      boxShadow: status === 'running' ? `0 0 8px ${color}` : undefined,
      animation: status === 'running' ? 'pulse 2s infinite' : undefined,
    }} />
  );
}

function MiniStat({ label, value, color }: { label: string; value: string | number; color?: string }) {
  return (
    <div style={{ textAlign: 'right' }}>
      <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>{label}</div>
      <div style={{ fontSize: 13, fontWeight: 600, color: color ?? 'var(--text-primary)', fontVariantNumeric: 'tabular-nums' }}>{value}</div>
    </div>
  );
}

function FilterPill({ children, active, onClick }: {
  children: React.ReactNode;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '3px 10px', borderRadius: 12, fontSize: 11, fontWeight: 500,
        border: active ? '1px solid var(--accent)' : '1px solid var(--border)',
        background: active ? 'rgba(99,102,241,0.12)' : 'transparent',
        color: active ? 'var(--accent-light)' : 'var(--text-muted)',
        transition: 'all 0.15s',
      }}
    >
      {children}
    </button>
  );
}

function IconButton({ onClick, title, children, danger }: {
  onClick: () => void;
  title: string;
  children: React.ReactNode;
  danger?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: 28, height: 28,
        borderRadius: 'var(--radius)',
        border: '1px solid var(--border)',
        background: 'transparent',
        color: danger ? 'var(--red)' : 'var(--text-secondary)',
        transition: 'all 0.15s',
      }}
    >
      {children}
    </button>
  );
}

const clearBtnStyle: React.CSSProperties = {
  display: 'flex', alignItems: 'center', justifyContent: 'center',
  border: 'none', background: 'transparent',
  color: 'var(--text-muted)', padding: 2, borderRadius: 4,
};
