import { useState, useEffect, useRef, useCallback } from 'react';
import {
  LayoutDashboard, FlaskConical, Activity, MessageCircle,
  GitCompareArrows, Trophy, Search, X,
} from 'lucide-react';

const COMMANDS = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard, group: 'Navigate' },
  { id: 'create', label: 'New Experiment', icon: FlaskConical, group: 'Navigate' },
  { id: 'monitor', label: 'Live Monitor', icon: Activity, group: 'Navigate' },
  { id: 'playground', label: 'Playground', icon: MessageCircle, group: 'Navigate' },
  { id: 'compare', label: 'Compare Experiments', icon: GitCompareArrows, group: 'Navigate' },
  { id: 'leaderboard', label: 'Leaderboard', icon: Trophy, group: 'Navigate' },
];

interface CommandPaletteProps {
  onNavigate: (view: string) => void;
}

export function CommandPalette({ onNavigate }: CommandPaletteProps) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const filtered = COMMANDS.filter(c =>
    c.label.toLowerCase().includes(query.toLowerCase())
  );

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setOpen(prev => !prev);
        setQuery('');
        setSelectedIndex(0);
      }
      if (e.key === 'Escape') setOpen(false);
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  useEffect(() => {
    setSelectedIndex(0);
  }, [query]);

  const handleSelect = useCallback((id: string) => {
    onNavigate(id);
    setOpen(false);
    setQuery('');
  }, [onNavigate]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex(i => Math.min(i + 1, filtered.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex(i => Math.max(i - 1, 0));
    } else if (e.key === 'Enter' && filtered[selectedIndex]) {
      handleSelect(filtered[selectedIndex].id);
    }
  };

  if (!open) return null;

  return (
    <div
      style={{
        position: 'fixed', inset: 0, zIndex: 10000,
        background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)',
        display: 'flex', alignItems: 'flex-start', justifyContent: 'center',
        paddingTop: '20vh',
      }}
      onClick={() => setOpen(false)}
    >
      <div
        onClick={e => e.stopPropagation()}
        style={{
          width: 480, background: 'var(--bg-secondary)',
          border: '1px solid var(--border-light)',
          borderRadius: 12, overflow: 'hidden',
          boxShadow: '0 16px 48px rgba(0,0,0,0.5)',
          animation: 'palette-in 0.15s ease-out',
        }}
      >
        {/* Search input */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10,
          padding: '12px 16px', borderBottom: '1px solid var(--border)',
        }}>
          <Search size={16} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
          <input
            ref={inputRef}
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search commands..."
            style={{
              flex: 1, background: 'none', border: 'none', outline: 'none',
              color: 'var(--text-primary)', fontSize: 14,
            }}
          />
          <div style={{
            padding: '2px 6px', borderRadius: 4, background: 'var(--bg-tertiary)',
            fontSize: 10, color: 'var(--text-muted)', fontWeight: 600,
          }}>
            ESC
          </div>
        </div>

        {/* Results */}
        <div style={{ maxHeight: 320, overflow: 'auto', padding: '4px 0' }}>
          {filtered.length === 0 ? (
            <div style={{ padding: '20px 16px', textAlign: 'center', color: 'var(--text-muted)', fontSize: 13 }}>
              No commands found
            </div>
          ) : (
            filtered.map((cmd, i) => (
              <button
                key={cmd.id}
                onClick={() => handleSelect(cmd.id)}
                onMouseEnter={() => setSelectedIndex(i)}
                style={{
                  display: 'flex', alignItems: 'center', gap: 10,
                  width: '100%', padding: '8px 16px', border: 'none',
                  background: i === selectedIndex ? 'var(--bg-tertiary)' : 'transparent',
                  color: i === selectedIndex ? 'var(--text-primary)' : 'var(--text-secondary)',
                  fontSize: 13, textAlign: 'left', cursor: 'pointer',
                }}
              >
                <cmd.icon size={15} style={{ color: 'var(--text-muted)' }} />
                <span style={{ flex: 1 }}>{cmd.label}</span>
                <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>{cmd.group}</span>
              </button>
            ))
          )}
        </div>

        {/* Footer */}
        <div style={{
          padding: '8px 16px', borderTop: '1px solid var(--border)',
          display: 'flex', gap: 12, fontSize: 10, color: 'var(--text-muted)',
        }}>
          <span>↑↓ navigate</span>
          <span>↵ select</span>
          <span>esc close</span>
        </div>
      </div>
    </div>
  );
}
