import { useState, useEffect, useRef } from 'react';
import {
  Bell, Play, Pause, CheckCircle2, XCircle, Zap, TrendingUp, X,
} from 'lucide-react';
import type { Experiment } from '../types';

interface Notification {
  id: string;
  type: 'started' | 'paused' | 'completed' | 'failed' | 'milestone';
  title: string;
  detail: string;
  timestamp: number;
  read: boolean;
  experimentId: string;
}

interface NotificationCenterProps {
  experiments: Experiment[];
  onNavigate: (exp: Experiment) => void;
}

const iconMap: Record<string, { icon: typeof Play; color: string }> = {
  started: { icon: Play, color: 'var(--green)' },
  paused: { icon: Pause, color: 'var(--amber)' },
  completed: { icon: CheckCircle2, color: 'var(--accent-light)' },
  failed: { icon: XCircle, color: 'var(--red)' },
  milestone: { icon: TrendingUp, color: 'var(--cyan)' },
};

function timeAgo(ts: number): string {
  const secs = Math.floor(Date.now() / 1000 - ts);
  if (secs < 60) return 'now';
  if (secs < 3600) return `${Math.floor(secs / 60)}m`;
  if (secs < 86400) return `${Math.floor(secs / 3600)}h`;
  return `${Math.floor(secs / 86400)}d`;
}

export function NotificationCenter({ experiments, onNavigate }: NotificationCenterProps) {
  const [open, setOpen] = useState(false);
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const prevStatusRef = useRef<Map<string, string>>(new Map());
  const panelRef = useRef<HTMLDivElement>(null);

  // Track experiment status changes
  useEffect(() => {
    const prev = prevStatusRef.current;
    const newNotifs: Notification[] = [];

    for (const exp of experiments) {
      const prevStatus = prev.get(exp.id);
      if (prevStatus && prevStatus !== exp.status) {
        // Status changed
        if (exp.status === 'running' && prevStatus !== 'running') {
          newNotifs.push({
            id: `${exp.id}-${Date.now()}`,
            type: 'started',
            title: exp.name,
            detail: 'Training started',
            timestamp: exp.updated_at,
            read: false,
            experimentId: exp.id,
          });
        } else if (exp.status === 'completed') {
          newNotifs.push({
            id: `${exp.id}-${Date.now()}`,
            type: 'completed',
            title: exp.name,
            detail: `Completed — best reward ${exp.metrics.best_reward.toFixed(3)}`,
            timestamp: exp.updated_at,
            read: false,
            experimentId: exp.id,
          });
        } else if (exp.status === 'failed') {
          newNotifs.push({
            id: `${exp.id}-${Date.now()}`,
            type: 'failed',
            title: exp.name,
            detail: 'Training failed',
            timestamp: exp.updated_at,
            read: false,
            experimentId: exp.id,
          });
        } else if (exp.status === 'paused') {
          newNotifs.push({
            id: `${exp.id}-${Date.now()}`,
            type: 'paused',
            title: exp.name,
            detail: 'Training paused',
            timestamp: exp.updated_at,
            read: false,
            experimentId: exp.id,
          });
        }
      }

      // Milestone: >75% completion
      if (exp.status === 'running') {
        const pct = exp.metrics.total_episodes / exp.training.num_episodes;
        if (pct >= 0.75 && prevStatus === 'running') {
          const existing = notifications.find(n => n.id.startsWith(`milestone-75-${exp.id}`));
          if (!existing) {
            newNotifs.push({
              id: `milestone-75-${exp.id}-${Date.now()}`,
              type: 'milestone',
              title: exp.name,
              detail: `75% complete (${exp.metrics.total_episodes}/${exp.training.num_episodes})`,
              timestamp: exp.updated_at,
              read: false,
              experimentId: exp.id,
            });
          }
        }
      }

      prev.set(exp.id, exp.status);
    }

    if (newNotifs.length > 0) {
      setNotifications(p => [...newNotifs, ...p].slice(0, 50));
    }
  }, [experiments]);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  const unreadCount = notifications.filter(n => !n.read).length;

  const markAllRead = () => {
    setNotifications(prev => prev.map(n => ({ ...n, read: true })));
  };

  const handleClick = (notif: Notification) => {
    setNotifications(prev => prev.map(n => n.id === notif.id ? { ...n, read: true } : n));
    const exp = experiments.find(e => e.id === notif.experimentId);
    if (exp) {
      onNavigate(exp);
      setOpen(false);
    }
  };

  const clearAll = () => {
    setNotifications([]);
    setOpen(false);
  };

  return (
    <div ref={panelRef} style={{ position: 'relative' }}>
      {/* Bell button */}
      <button
        onClick={() => setOpen(!open)}
        style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          width: 32, height: 32, borderRadius: 'var(--radius)',
          border: '1px solid var(--border)', background: 'transparent',
          color: 'var(--text-secondary)', position: 'relative',
        }}
      >
        <Bell size={15} />
        {unreadCount > 0 && (
          <span style={{
            position: 'absolute', top: -4, right: -4,
            width: 16, height: 16, borderRadius: '50%',
            background: 'var(--red)', color: '#fff',
            fontSize: 9, fontWeight: 700,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            {unreadCount > 9 ? '9+' : unreadCount}
          </span>
        )}
      </button>

      {/* Dropdown panel */}
      {open && (
        <div style={{
          position: 'absolute', top: 40, right: 0, width: 340,
          background: 'var(--bg-secondary)', border: '1px solid var(--border-light)',
          borderRadius: 'var(--radius-lg)', overflow: 'hidden',
          boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
          zIndex: 9999,
          animation: 'palette-in 0.15s ease-out',
        }}>
          {/* Header */}
          <div style={{
            padding: '10px 14px', borderBottom: '1px solid var(--border)',
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          }}>
            <span style={{ fontSize: 13, fontWeight: 600 }}>
              Notifications
              {unreadCount > 0 && (
                <span style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 400, marginLeft: 6 }}>
                  {unreadCount} new
                </span>
              )}
            </span>
            <div style={{ display: 'flex', gap: 8 }}>
              {unreadCount > 0 && (
                <button onClick={markAllRead} style={linkBtnStyle}>Mark read</button>
              )}
              {notifications.length > 0 && (
                <button onClick={clearAll} style={linkBtnStyle}>Clear</button>
              )}
            </div>
          </div>

          {/* List */}
          <div style={{ maxHeight: 360, overflow: 'auto' }}>
            {notifications.length === 0 ? (
              <div style={{ padding: '32px 16px', textAlign: 'center', color: 'var(--text-muted)', fontSize: 12 }}>
                No notifications yet
              </div>
            ) : (
              notifications.map(notif => {
                const meta = iconMap[notif.type] ?? iconMap.started;
                const Icon = meta.icon;
                return (
                  <button
                    key={notif.id}
                    onClick={() => handleClick(notif)}
                    style={{
                      display: 'flex', alignItems: 'start', gap: 10,
                      width: '100%', padding: '10px 14px', border: 'none',
                      background: notif.read ? 'transparent' : 'rgba(99,102,241,0.04)',
                      color: 'var(--text-primary)', textAlign: 'left',
                      borderBottom: '1px solid var(--border)',
                      transition: 'background 0.1s',
                    }}
                    onMouseEnter={e => (e.currentTarget.style.background = 'var(--bg-tertiary)')}
                    onMouseLeave={e => (e.currentTarget.style.background = notif.read ? 'transparent' : 'rgba(99,102,241,0.04)')}
                  >
                    <Icon size={14} style={{ color: meta.color, flexShrink: 0, marginTop: 2 }} />
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: 12, fontWeight: notif.read ? 400 : 600 }}>{notif.title}</div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{notif.detail}</div>
                    </div>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', flexShrink: 0 }}>
                      {timeAgo(notif.timestamp)}
                    </div>
                    {!notif.read && (
                      <div style={{
                        width: 6, height: 6, borderRadius: '50%',
                        background: 'var(--accent)', flexShrink: 0, marginTop: 4,
                      }} />
                    )}
                  </button>
                );
              })
            )}
          </div>
        </div>
      )}
    </div>
  );
}

const linkBtnStyle: React.CSSProperties = {
  border: 'none', background: 'transparent',
  color: 'var(--accent-light)', fontSize: 11, fontWeight: 500,
};
