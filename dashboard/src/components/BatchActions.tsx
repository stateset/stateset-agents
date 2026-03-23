import { Play, Pause, Trash2, Square, X } from 'lucide-react';
import { api } from '../api';
import { useToast } from '../hooks/useToast';

interface BatchActionsProps {
  selectedIds: string[];
  onClear: () => void;
  onRefresh: () => void;
}

export function BatchActions({ selectedIds, onClear, onRefresh }: BatchActionsProps) {
  const toast = useToast();
  const count = selectedIds.length;

  if (count === 0) return null;

  const handleBatch = async (
    action: (id: string) => Promise<unknown>,
    label: string,
  ) => {
    let success = 0;
    let failed = 0;
    for (const id of selectedIds) {
      try {
        await action(id);
        success++;
      } catch {
        failed++;
      }
    }
    if (failed === 0) {
      toast.success(`${label} ${success} experiment${success > 1 ? 's' : ''}`);
    } else {
      toast.error(`${label} ${success}, failed ${failed}`);
    }
    onClear();
    onRefresh();
  };

  return (
    <div style={{
      position: 'fixed', bottom: 24, left: '50%', transform: 'translateX(-50%)',
      zIndex: 8000,
      display: 'flex', alignItems: 'center', gap: 8,
      padding: '10px 16px', borderRadius: 12,
      background: 'var(--bg-secondary)', border: '1px solid var(--border-light)',
      boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
      animation: 'toast-in 0.2s ease-out',
    }}>
      <span style={{ fontSize: 13, fontWeight: 600, marginRight: 4 }}>
        {count} selected
      </span>

      <BatchBtn onClick={() => handleBatch(api.startExperiment, 'Started')} title="Start all">
        <Play size={13} /> Start
      </BatchBtn>
      <BatchBtn onClick={() => handleBatch(api.pauseExperiment, 'Paused')} title="Pause all">
        <Pause size={13} /> Pause
      </BatchBtn>
      <BatchBtn onClick={() => handleBatch(api.stopExperiment, 'Stopped')} title="Stop all">
        <Square size={11} /> Stop
      </BatchBtn>

      <div style={{ width: 1, height: 20, background: 'var(--border)', margin: '0 4px' }} />

      <BatchBtn
        onClick={() => handleBatch(api.deleteExperiment, 'Deleted')}
        title="Delete all"
        danger
      >
        <Trash2 size={13} /> Delete
      </BatchBtn>

      <button
        onClick={onClear}
        style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          width: 24, height: 24, borderRadius: 'var(--radius)',
          border: 'none', background: 'var(--bg-tertiary)',
          color: 'var(--text-muted)', marginLeft: 4,
        }}
        title="Clear selection"
      >
        <X size={12} />
      </button>
    </div>
  );
}

function BatchBtn({ children, onClick, title, danger }: {
  children: React.ReactNode;
  onClick: () => void;
  title: string;
  danger?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 4,
        padding: '5px 10px', borderRadius: 'var(--radius)',
        border: '1px solid var(--border)',
        background: 'transparent',
        color: danger ? 'var(--red)' : 'var(--text-secondary)',
        fontSize: 12, fontWeight: 500,
        transition: 'all 0.15s',
      }}
    >
      {children}
    </button>
  );
}
