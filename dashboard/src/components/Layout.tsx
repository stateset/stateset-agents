import { ReactNode } from 'react';
import {
  Activity, FlaskConical, GitCompareArrows, LayoutDashboard,
  MessageCircle, Trophy,
} from 'lucide-react';
import { NotificationCenter } from './NotificationCenter';
import type { Experiment } from '../types';

const HOTKEY_MAP: Record<string, string> = {
  dashboard: 'D',
  create: 'N',
  playground: 'P',
  compare: 'C',
  leaderboard: 'L',
};

interface LayoutProps {
  children: ReactNode;
  currentView: string;
  onNavigate: (view: string) => void;
  experiments?: Experiment[];
  onSelectExperiment?: (exp: Experiment) => void;
}

export function Layout({ children, currentView, onNavigate, experiments = [], onSelectExperiment }: LayoutProps) {
  const nav = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'create', label: 'New Experiment', icon: FlaskConical },
    { id: 'monitor', label: 'Live Monitor', icon: Activity },
    { id: 'playground', label: 'Playground', icon: MessageCircle },
    { id: 'compare', label: 'Compare', icon: GitCompareArrows },
    { id: 'leaderboard', label: 'Leaderboard', icon: Trophy },
  ];

  return (
    <div style={{ display: 'flex', minHeight: '100vh' }}>
      {/* Sidebar */}
      <aside style={{
        width: 220,
        background: 'var(--bg-secondary)',
        borderRight: '1px solid var(--border)',
        display: 'flex',
        flexDirection: 'column',
        flexShrink: 0,
      }}>
        <div style={{
          padding: '20px 16px',
          borderBottom: '1px solid var(--border)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{
              width: 32, height: 32,
              borderRadius: 8,
              background: 'linear-gradient(135deg, var(--accent), var(--purple))',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 15, fontWeight: 700, color: '#fff',
            }}>S</div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 14, fontWeight: 600 }}>Training Lab</div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>stateset-agents</div>
            </div>
            {onSelectExperiment && (
              <NotificationCenter
                experiments={experiments}
                onNavigate={(exp) => { onSelectExperiment(exp); onNavigate('monitor'); }}
              />
            )}
          </div>
        </div>

        <nav style={{ padding: '12px 8px', flex: 1 }}>
          <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-muted)', padding: '8px 12px 4px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Train
          </div>
          {nav.slice(0, 3).map(item => (
            <NavButton
              key={item.id}
              item={item}
              active={currentView === item.id}
              hotkey={HOTKEY_MAP[item.id]}
              onClick={() => onNavigate(item.id)}
            />
          ))}

          <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-muted)', padding: '16px 12px 4px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Evaluate
          </div>
          {nav.slice(3).map(item => (
            <NavButton
              key={item.id}
              item={item}
              active={currentView === item.id}
              hotkey={HOTKEY_MAP[item.id]}
              onClick={() => onNavigate(item.id)}
            />
          ))}
        </nav>

        <div style={{
          padding: '12px 16px',
          borderTop: '1px solid var(--border)',
          fontSize: 10,
          color: 'var(--text-muted)',
          display: 'flex', justifyContent: 'space-between',
        }}>
          <span>RL Gym v0.3</span>
          <span style={{ opacity: 0.6 }}>&#8984;K search</span>
        </div>
      </aside>

      {/* Main */}
      <main style={{ flex: 1, overflow: 'auto' }}>
        {children}
      </main>
    </div>
  );
}

function NavButton({ item, active, hotkey, onClick }: {
  item: { id: string; label: string; icon: React.ElementType };
  active: boolean;
  hotkey?: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        width: '100%',
        padding: '7px 12px',
        marginBottom: 1,
        border: 'none',
        borderRadius: 'var(--radius)',
        background: active ? 'var(--bg-tertiary)' : 'transparent',
        color: active ? 'var(--text-primary)' : 'var(--text-secondary)',
        fontSize: 13,
        fontWeight: active ? 500 : 400,
        transition: 'all 0.15s',
        textAlign: 'left',
      }}
    >
      <item.icon size={15} />
      <span style={{ flex: 1 }}>{item.label}</span>
      {hotkey && (
        <span style={{
          fontSize: 9, color: 'var(--text-muted)', fontWeight: 600,
          padding: '1px 5px', borderRadius: 3,
          background: active ? 'var(--bg-hover)' : 'var(--bg-tertiary)',
          opacity: 0.7,
        }}>
          {hotkey}
        </span>
      )}
    </button>
  );
}
