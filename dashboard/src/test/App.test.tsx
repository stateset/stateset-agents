import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import App from '../App';

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.endsWith('/api/lab/experiments')) {
      return new Response('[]', { status: 200, headers: { 'Content-Type': 'application/json' } });
    }
    return new Response('null', { status: 200, headers: { 'Content-Type': 'application/json' } });
  }));
  // jsdom doesn't ship a WebSocket — stub minimal shape so connectWs() doesn't throw.
  class StubWS {
    readyState = 0;
    onopen: ((e?: unknown) => void) | null = null;
    onclose: (() => void) | null = null;
    onerror: (() => void) | null = null;
    onmessage: ((e: { data: string }) => void) | null = null;
    send() {}
    close() {}
  }
  vi.stubGlobal('WebSocket', StubWS as unknown as typeof WebSocket);
});

describe('App', () => {
  it('renders the sidebar and lands on the dashboard view', async () => {
    render(<App />);
    expect(await screen.findByText('Training Lab')).toBeInTheDocument();
    expect(screen.getByText('AI Training Lab')).toBeInTheDocument();
    // Two "New Experiment" buttons — sidebar nav and dashboard CTA.
    expect(screen.getAllByRole('button', { name: /new experiment/i }).length).toBeGreaterThanOrEqual(2);
  });
});
