import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  LayoutDashboard, FlaskConical, Activity, MessageCircle,
  GitCompareArrows, Trophy, Search,
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
  const [trackedQuery, setTrackedQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const previousFocus = useRef<HTMLElement | null>(null);

  const filtered = useMemo(() => COMMANDS.filter(c =>
    c.label.toLowerCase().includes(query.toLowerCase())
  ), [query]);

  // Reset selection when the query changes — done during render via the
  // "adjust state on prop change" pattern (cheaper than an effect, and
  // avoids react-hooks/set-state-in-effect).
  if (query !== trackedQuery) {
    setTrackedQuery(query);
    setSelectedIndex(0);
  }

  const close = useCallback(() => {
    setOpen(false);
    previousFocus.current?.focus?.();
  }, []);

  // Global hotkey: Cmd/Ctrl + K
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setOpen(prev => {
          if (!prev) {
            previousFocus.current = (document.activeElement as HTMLElement) ?? null;
          }
          return !prev;
        });
        setQuery('');
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  // Autofocus the input on open
  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  // Keep the highlighted item in view
  useEffect(() => {
    if (!open || !listRef.current) return;
    const el = listRef.current.querySelector<HTMLElement>(`[data-index="${selectedIndex}"]`);
    el?.scrollIntoView({ block: 'nearest' });
  }, [open, selectedIndex]);

  const handleSelect = useCallback((id: string) => {
    onNavigate(id);
    setQuery('');
    close();
  }, [onNavigate, close]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex(i => Math.min(i + 1, filtered.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex(i => Math.max(i - 1, 0));
    } else if (e.key === 'Home') {
      e.preventDefault();
      setSelectedIndex(0);
    } else if (e.key === 'End') {
      e.preventDefault();
      setSelectedIndex(Math.max(0, filtered.length - 1));
    } else if (e.key === 'Enter' && filtered[selectedIndex]) {
      e.preventDefault();
      handleSelect(filtered[selectedIndex].id);
    } else if (e.key === 'Escape') {
      e.preventDefault();
      close();
    } else if (e.key === 'Tab') {
      // Trap focus inside the dialog.
      const focusable = dialogRef.current?.querySelectorAll<HTMLElement>(
        'button, [href], input, [tabindex]:not([tabindex="-1"])',
      );
      if (!focusable || focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
  };

  if (!open) return null;

  const listboxId = 'command-palette-listbox';
  const activeOptionId = filtered[selectedIndex] ? `cmd-opt-${filtered[selectedIndex].id}` : undefined;

  return (
    <div
      style={{
        position: 'fixed', inset: 0, zIndex: 10000,
        background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)',
        display: 'flex', alignItems: 'flex-start', justifyContent: 'center',
        paddingTop: '20vh',
      }}
      onClick={close}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
        onClick={e => e.stopPropagation()}
        onKeyDown={handleKeyDown}
        style={{
          width: 480, background: 'var(--bg-secondary)',
          border: '1px solid var(--border-light)',
          borderRadius: 12, overflow: 'hidden',
          boxShadow: '0 16px 48px rgba(0,0,0,0.5)',
          animation: 'palette-in 0.15s ease-out',
        }}
      >
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10,
          padding: '12px 16px', borderBottom: '1px solid var(--border)',
        }}>
          <Search size={16} style={{ color: 'var(--text-muted)', flexShrink: 0 }} aria-hidden />
          <input
            ref={inputRef}
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Search commands..."
            aria-label="Search commands"
            aria-controls={listboxId}
            aria-activedescendant={activeOptionId}
            aria-autocomplete="list"
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

        <div
          ref={listRef}
          id={listboxId}
          role="listbox"
          aria-label="Commands"
          style={{ maxHeight: 320, overflow: 'auto', padding: '4px 0' }}
        >
          {filtered.length === 0 ? (
            <div style={{ padding: '20px 16px', textAlign: 'center', color: 'var(--text-muted)', fontSize: 13 }}>
              No commands found
            </div>
          ) : (
            filtered.map((cmd, i) => (
              <button
                key={cmd.id}
                id={`cmd-opt-${cmd.id}`}
                data-index={i}
                role="option"
                aria-selected={i === selectedIndex}
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
                <cmd.icon size={15} style={{ color: 'var(--text-muted)' }} aria-hidden />
                <span style={{ flex: 1 }}>{cmd.label}</span>
                <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>{cmd.group}</span>
              </button>
            ))
          )}
        </div>

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
