import { describe, it, expect } from 'vitest';
import { act, render, renderHook, screen } from '@testing-library/react';
import type { ReactNode } from 'react';
import { ToastProvider } from '../components/ToastProvider';
import { useToast } from '../hooks/useToast';

function wrapper({ children }: { children: ReactNode }) {
  return <ToastProvider>{children}</ToastProvider>;
}

describe('useToast', () => {
  it('exposes success / error / info / toast', () => {
    const { result } = renderHook(() => useToast(), { wrapper });
    expect(typeof result.current.toast).toBe('function');
    expect(typeof result.current.success).toBe('function');
    expect(typeof result.current.error).toBe('function');
    expect(typeof result.current.info).toBe('function');
  });

  it('renders a toast message that the user can read', () => {
    render(
      <ToastProvider>
        <ToastTrigger />
      </ToastProvider>,
    );
    act(() => {
      screen.getByRole('button', { name: /fire/i }).click();
    });
    expect(screen.getByText('hello world')).toBeInTheDocument();
    expect(screen.getByRole('region', { name: /notifications/i })).toBeInTheDocument();
  });
});

function ToastTrigger() {
  const t = useToast();
  return <button onClick={() => t.success('hello world')}>fire</button>;
}
