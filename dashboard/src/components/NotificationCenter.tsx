import { useState, useEffect, useRef } from 'react';
import {
  Bell, Play, Pause, CheckCircle2, XCircle, TrendingUp,
} from 'lucide-react';
import type { Experiment } from '../types';
import {
  useNotifications,
  markRead,
  markAllRead,
  clearAllNotifications,
} from '../hooks/notifications-store';

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
  const notifications = useNotifications();
  const [open, setOpen] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);

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

  const handleClick = (id: string, experimentId: string) => {
    markRead(id);
    const exp = experiments.find(e => e.id === experimentId);
    if (exp) {
      onNavigate(exp);
      setOpen(false);
    }
  };

  const clearAll = () => {
    clearAllNotifications();
    setOpen(false);
  };

  return (
    <div ref={panelRef} style={{ position: 'relative' }}>
      <button
        onClick={() => setOpen(!open)}
        aria-label={`Notifications${unreadCount > 0 ? ` (${unreadCount} unread)` : ''}`}
        aria-expanded={open}
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

      {open && (
        <div
          role="dialog"
          aria-label="Notifications"
          style={{
            position: 'absolute', top: 40, right: 0, width: 340,
            background: 'var(--bg-secondary)', border: '1px solid var(--border-light)',
            borderRadius: 'var(--radius-lg)', overflow: 'hidden',
            boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
            zIndex: 9999,
            animation: 'palette-in 0.15s ease-out',
          }}
        >
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
                    onClick={() => handleClick(notif.id, notif.experimentId)}
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
