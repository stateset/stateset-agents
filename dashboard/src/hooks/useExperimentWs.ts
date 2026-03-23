import { useEffect, useRef, useCallback, useState } from 'react';
import { connectWs } from '../api';
import type { WsMessage, TrainingMetrics, Episode } from '../types';

export function useExperimentWs(experimentId: string | null) {
  const wsRef = useRef<WebSocket | null>(null);
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [latestEpisode, setLatestEpisode] = useState<Episode | null>(null);
  const [connected, setConnected] = useState(false);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setConnected(false);
  }, []);

  useEffect(() => {
    if (!experimentId) return;

    const ws = connectWs(experimentId);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);

    ws.onmessage = (event) => {
      try {
        const msg: WsMessage = JSON.parse(event.data);
        if (msg.metrics) setMetrics(msg.metrics);
        if (msg.type === 'episode' && msg.data) setLatestEpisode(msg.data);
      } catch {
        // ignore parse errors
      }
    };

    // Keepalive ping
    const interval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) ws.send('ping');
    }, 30000);

    return () => {
      clearInterval(interval);
      ws.close();
    };
  }, [experimentId]);

  return { metrics, latestEpisode, connected, disconnect };
}
