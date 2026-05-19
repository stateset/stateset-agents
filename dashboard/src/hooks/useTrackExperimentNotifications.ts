import { useEffect, useRef } from 'react';
import type { Experiment } from '../types';
import { pushNotification } from './notifications-store';

/**
 * Watches `experiments` for status transitions and milestones, pushing
 * notifications into the global store. The store dispatch is external state,
 * so the effect body does not call any `useState` setter directly — which
 * keeps react-hooks/set-state-in-effect happy.
 */
export function useTrackExperimentNotifications(experiments: Experiment[]) {
  const prevStatus = useRef<Map<string, string>>(new Map());
  const milestoneSeen = useRef<Set<string>>(new Set());

  useEffect(() => {
    const prev = prevStatus.current;
    const seen = milestoneSeen.current;

    for (const exp of experiments) {
      const last = prev.get(exp.id);

      if (last && last !== exp.status) {
        const base = {
          id: `${exp.id}-${exp.status}-${exp.updated_at}`,
          title: exp.name,
          timestamp: exp.updated_at,
          read: false,
          experimentId: exp.id,
        };
        if (exp.status === 'running' && last !== 'running') {
          pushNotification({ ...base, type: 'started', detail: 'Training started' });
        } else if (exp.status === 'completed') {
          const best = exp.metrics?.best_reward ?? 0;
          pushNotification({ ...base, type: 'completed', detail: `Completed — best reward ${best.toFixed(3)}` });
        } else if (exp.status === 'failed') {
          pushNotification({ ...base, type: 'failed', detail: 'Training failed' });
        } else if (exp.status === 'paused') {
          pushNotification({ ...base, type: 'paused', detail: 'Training paused' });
        }
      }

      if (exp.status === 'running' && exp.training.num_episodes > 0) {
        const pct = (exp.metrics?.total_episodes ?? 0) / exp.training.num_episodes;
        const milestoneKey = `milestone-75-${exp.id}`;
        if (pct >= 0.75 && !seen.has(milestoneKey)) {
          seen.add(milestoneKey);
          pushNotification({
            id: `${milestoneKey}-${exp.updated_at}`,
            type: 'milestone',
            title: exp.name,
            detail: `75% complete (${exp.metrics?.total_episodes ?? 0}/${exp.training.num_episodes})`,
            timestamp: exp.updated_at,
            read: false,
            experimentId: exp.id,
          });
        }
      }

      prev.set(exp.id, exp.status);
    }
  }, [experiments]);
}
