import {
  useCallback,
  useMemo,
  useState,
} from 'react';
import type { ReactNode } from 'react';
import { ToastContext } from '../hooks/toast-context';
import type { ToastApi, ToastItem, ToastType } from '../hooks/toast-context';

let nextId = 0;

const typeStyles: Record<ToastType, { bg: string; border: string; color: string }> = {
  success: { bg: 'rgba(34,197,94,0.1)', border: 'rgba(34,197,94,0.3)', color: '#22c55e' },
  error: { bg: 'rgba(239,68,68,0.1)', border: 'rgba(239,68,68,0.3)', color: '#ef4444' },
  info: { bg: 'rgba(99,102,241,0.1)', border: 'rgba(99,102,241,0.3)', color: '#818cf8' },
};

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const toast = useCallback((message: string, type: ToastType = 'info') => {
    const id = ++nextId;
    setToasts(prev => [...prev, { id, message, type }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 3500);
  }, []);

  const api = useMemo<ToastApi>(() => ({
    toast,
    success: (m: string) => toast(m, 'success'),
    error: (m: string) => toast(m, 'error'),
    info: (m: string) => toast(m, 'info'),
  }), [toast]);

  const dismiss = (id: number) => setToasts(prev => prev.filter(t => t.id !== id));

  return (
    <ToastContext.Provider value={api}>
      {children}
      <div
        role="region"
        aria-label="Notifications"
        aria-live="polite"
        style={{
          position: 'fixed', bottom: 20, right: 20, zIndex: 9999,
          display: 'flex', flexDirection: 'column', gap: 8, pointerEvents: 'none',
        }}
      >
        {toasts.map(t => {
          const s = typeStyles[t.type];
          return (
            <div
              key={t.id}
              role="status"
              style={{
                padding: '10px 16px', borderRadius: 8, minWidth: 260, maxWidth: 400,
                background: s.bg, border: `1px solid ${s.border}`,
                backdropFilter: 'blur(12px)',
                display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12,
                fontSize: 13, color: s.color, fontWeight: 500,
                pointerEvents: 'auto',
                animation: 'toast-in 0.25s ease-out',
                boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
              }}
            >
              <span>{t.message}</span>
              <button
                onClick={() => dismiss(t.id)}
                aria-label="Dismiss notification"
                style={{
                  background: 'none', border: 'none', color: 'var(--text-muted)',
                  cursor: 'pointer', fontSize: 16, lineHeight: 1, padding: 0, flexShrink: 0,
                }}
              >
                &times;
              </button>
            </div>
          );
        })}
      </div>
    </ToastContext.Provider>
  );
}
