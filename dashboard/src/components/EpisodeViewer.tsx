import { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import type { Episode } from '../types';
import { Card } from './Card';

interface EpisodeViewerProps {
  episode: Episode;
}

export function EpisodeViewer({ episode }: EpisodeViewerProps) {
  const [expandedTurn, setExpandedTurn] = useState<number | null>(null);
  const [showAll, setShowAll] = useState(false);

  const visibleTurns = showAll ? episode.turns : episode.turns.slice(0, 20);
  const hasMore = episode.turns.length > 20;
  const scenarioTopic = (episode.scenario as Record<string, string>)?.topic ?? null;
  const avgReward = episode.turn_rewards.length > 0
    ? episode.turn_rewards.reduce((a, b) => a + b, 0) / episode.turn_rewards.length
    : 0;

  return (
    <Card>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div>
          <div style={{ fontSize: 14, fontWeight: 600 }}>
            Episode #{episode.episode_num}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', display: 'flex', gap: 8 }}>
            <span>{episode.turns.length} turns</span>
            <span>&middot;</span>
            <span>{episode.duration_ms}ms</span>
            {scenarioTopic && (
              <>
                <span>&middot;</span>
                <span style={{ color: 'var(--accent-light)' }}>{scenarioTopic}</span>
              </>
            )}
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <MetricBadge label="Total" value={episode.total_reward.toFixed(3)} color="var(--green)" />
          <MetricBadge label="Avg/turn" value={avgReward.toFixed(3)} color="var(--accent-light)" />
          {episode.loss != null && (
            <MetricBadge label="Loss" value={episode.loss.toFixed(4)} color="var(--red)" />
          )}
          {episode.entropy != null && (
            <MetricBadge label="Entropy" value={episode.entropy.toFixed(4)} color="var(--cyan)" />
          )}
        </div>
      </div>

      {/* Turn-by-turn conversation */}
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 4,
        maxHeight: showAll ? undefined : 400,
        overflow: showAll ? undefined : 'auto',
        padding: '4px 0',
      }}>
        {visibleTurns.map((turn, i) => {
          const isExpanded = expandedTurn === i;
          const turnReward = episode.turn_rewards[i] ?? turn.reward;
          return (
            <div
              key={i}
              style={{
                padding: '8px 12px',
                borderRadius: 'var(--radius)',
                background: turn.role === 'assistant' ? 'rgba(99, 102, 241, 0.08)' : 'rgba(255,255,255,0.03)',
                borderLeft: `3px solid ${turn.role === 'assistant' ? 'var(--accent)' : 'var(--border-light)'}`,
                cursor: turn.content.length > 200 ? 'pointer' : undefined,
              }}
              onClick={() => {
                if (turn.content.length > 200) {
                  setExpandedTurn(isExpanded ? null : i);
                }
              }}
            >
              <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
                <div style={{
                  fontSize: 10, fontWeight: 600, textTransform: 'uppercase',
                  color: turn.role === 'assistant' ? 'var(--accent-light)' : 'var(--text-muted)',
                  width: 60, flexShrink: 0, paddingTop: 2,
                  display: 'flex', alignItems: 'center', gap: 4,
                }}>
                  {turn.role}
                  <span style={{ fontSize: 9, color: 'var(--text-muted)', fontWeight: 400 }}>
                    #{i + 1}
                  </span>
                </div>
                <div style={{
                  flex: 1, fontSize: 13, lineHeight: 1.6,
                  overflow: 'hidden',
                  ...(turn.content.length > 200 && !isExpanded ? {
                    maxHeight: 60,
                    WebkitMaskImage: 'linear-gradient(to bottom, black 60%, transparent)',
                    maskImage: 'linear-gradient(to bottom, black 60%, transparent)',
                  } : {}),
                }}>
                  {turn.content}
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 2, flexShrink: 0 }}>
                  <span style={{
                    fontSize: 11,
                    color: turnReward > 0.7 ? 'var(--green)' : turnReward < 0.4 ? 'var(--red)' : 'var(--amber)',
                    fontWeight: 600, fontVariantNumeric: 'tabular-nums',
                  }}>
                    {turnReward.toFixed(3)}
                  </span>
                  {turn.content.length > 200 && (
                    isExpanded
                      ? <ChevronUp size={12} style={{ color: 'var(--text-muted)' }} />
                      : <ChevronDown size={12} style={{ color: 'var(--text-muted)' }} />
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Show more button */}
      {hasMore && !showAll && (
        <button
          onClick={() => setShowAll(true)}
          style={{
            display: 'block', width: '100%', marginTop: 8,
            padding: '6px 0', borderRadius: 'var(--radius)',
            border: '1px solid var(--border)', background: 'transparent',
            color: 'var(--text-secondary)', fontSize: 12, textAlign: 'center',
          }}
        >
          Show all {episode.turns.length} turns
        </button>
      )}

      {/* Turn reward sparkline */}
      <div style={{ marginTop: 12 }}>
        <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4 }}>Reward per turn</div>
        <div style={{ display: 'flex', gap: 2, alignItems: 'end', height: 32 }}>
          {episode.turn_rewards.map((r, i) => (
            <div
              key={i}
              style={{
                flex: 1,
                height: `${Math.max(4, r * 100)}%`,
                background: r > 0.7 ? 'var(--green)' : r > 0.4 ? 'var(--amber)' : 'var(--red)',
                borderRadius: '2px 2px 0 0',
                opacity: 0.7,
                transition: 'height 0.3s',
              }}
              title={`Turn ${i + 1}: ${r.toFixed(3)}`}
            />
          ))}
        </div>
      </div>
    </Card>
  );
}

function MetricBadge({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{
      padding: '3px 8px',
      borderRadius: 6,
      background: `color-mix(in srgb, ${color} 12%, transparent)`,
      border: `1px solid color-mix(in srgb, ${color} 25%, transparent)`,
    }}>
      <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>{label} </span>
      <span style={{ fontSize: 12, fontWeight: 600, color, fontVariantNumeric: 'tabular-nums' }}>{value}</span>
    </div>
  );
}
