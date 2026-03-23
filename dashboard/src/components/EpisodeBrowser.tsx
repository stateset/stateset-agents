import { useState, useEffect, useMemo } from 'react';
import { ChevronLeft, ChevronRight, MessageSquare, Search, X, SlidersHorizontal } from 'lucide-react';
import { Card } from './Card';
import { EpisodeViewer } from './EpisodeViewer';
import { api } from '../api';
import type { Episode } from '../types';

interface EpisodeBrowserProps {
  experimentId: string;
}

export function EpisodeBrowser({ experimentId }: EpisodeBrowserProps) {
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<Episode | null>(null);
  const [sort, setSort] = useState<'desc' | 'asc'>('desc');
  const [search, setSearch] = useState('');
  const [minReward, setMinReward] = useState('');
  const [maxReward, setMaxReward] = useState('');
  const [showFilters, setShowFilters] = useState(false);
  const pageSize = 20;

  useEffect(() => {
    api.getEpisodes(experimentId, page * pageSize, pageSize)
      .then(res => {
        setEpisodes(res.episodes);
        setTotal(res.total);
      })
      .catch(() => {});
  }, [experimentId, page, sort]);

  const filtered = useMemo(() => {
    let result = episodes;
    if (search) {
      const q = search.toLowerCase();
      result = result.filter(ep => {
        const topic = (ep.scenario as Record<string, string>)?.topic ?? '';
        return topic.toLowerCase().includes(q) ||
          ep.turns.some(t => t.content.toLowerCase().includes(q));
      });
    }
    if (minReward !== '') {
      const min = parseFloat(minReward);
      if (!isNaN(min)) result = result.filter(ep => ep.total_reward >= min);
    }
    if (maxReward !== '') {
      const max = parseFloat(maxReward);
      if (!isNaN(max)) result = result.filter(ep => ep.total_reward <= max);
    }
    return result;
  }, [episodes, search, minReward, maxReward]);

  const totalPages = Math.ceil(total / pageSize);
  const hasFilters = search !== '' || minReward !== '' || maxReward !== '';

  const clearFilters = () => {
    setSearch('');
    setMinReward('');
    setMaxReward('');
  };

  // Episode stats
  const rewardStats = useMemo(() => {
    if (episodes.length === 0) return null;
    const rewards = episodes.map(e => e.total_reward);
    return {
      min: Math.min(...rewards),
      max: Math.max(...rewards),
      avg: rewards.reduce((a, b) => a + b, 0) / rewards.length,
    };
  }, [episodes]);

  return (
    <div>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ fontSize: 14, fontWeight: 600 }}>
            Episode History
            <span style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 400, marginLeft: 8 }}>
              {total} episodes
            </span>
          </div>
          {rewardStats && (
            <div style={{ display: 'flex', gap: 10, fontSize: 11, color: 'var(--text-muted)' }}>
              <span>avg: <span style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>{rewardStats.avg.toFixed(3)}</span></span>
              <span>range: <span style={{ color: 'var(--green)', fontWeight: 500 }}>{rewardStats.min.toFixed(2)}</span>–<span style={{ color: 'var(--green)', fontWeight: 500 }}>{rewardStats.max.toFixed(2)}</span></span>
            </div>
          )}
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <select
            value={sort}
            onChange={e => setSort(e.target.value as 'asc' | 'desc')}
            style={selectStyle}
          >
            <option value="desc">Newest first</option>
            <option value="asc">Oldest first</option>
          </select>
          <button
            onClick={() => setShowFilters(!showFilters)}
            style={{
              ...selectStyle,
              display: 'flex', alignItems: 'center', gap: 4,
              border: hasFilters ? '1px solid var(--accent)' : '1px solid var(--border)',
              color: hasFilters ? 'var(--accent-light)' : 'var(--text-primary)',
            }}
          >
            <SlidersHorizontal size={12} /> Filter
          </button>
        </div>
      </div>

      {/* Filter bar */}
      {showFilters && (
        <div style={{
          display: 'flex', gap: 12, alignItems: 'center', marginBottom: 16,
          padding: '10px 14px', background: 'var(--bg-secondary)',
          border: '1px solid var(--border)', borderRadius: 'var(--radius)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, flex: 1 }}>
            <Search size={13} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search conversations..."
              style={filterInputStyle}
            />
            {search && (
              <button onClick={() => setSearch('')} style={clearBtnStyle}>
                <X size={10} />
              </button>
            )}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Reward:</span>
            <input
              value={minReward}
              onChange={e => setMinReward(e.target.value)}
              placeholder="min"
              type="number"
              step="0.1"
              style={{ ...filterInputStyle, width: 60, textAlign: 'center' }}
            />
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>–</span>
            <input
              value={maxReward}
              onChange={e => setMaxReward(e.target.value)}
              placeholder="max"
              type="number"
              step="0.1"
              style={{ ...filterInputStyle, width: 60, textAlign: 'center' }}
            />
          </div>
          {hasFilters && (
            <button onClick={clearFilters} style={{ ...clearBtnStyle, fontSize: 11, padding: '2px 8px' }}>
              Clear
            </button>
          )}
        </div>
      )}

      {/* Results count */}
      {hasFilters && (
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
          Showing {filtered.length} of {episodes.length} episodes on this page
        </div>
      )}

      {selected ? (
        <div>
          <button
            onClick={() => setSelected(null)}
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 4,
              padding: '4px 10px', borderRadius: 'var(--radius)', border: '1px solid var(--border)',
              background: 'transparent', color: 'var(--text-secondary)', fontSize: 12, marginBottom: 12,
            }}
          >
            <ChevronLeft size={12} /> Back to list
          </button>
          <EpisodeViewer episode={selected} />
        </div>
      ) : (
        <>
          {/* Table */}
          <div style={{
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-lg)',
            overflow: 'hidden',
          }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: 'var(--bg-secondary)' }}>
                  <Th>#</Th>
                  <Th>Scenario</Th>
                  <Th align="center">Turns</Th>
                  <Th align="right">Reward</Th>
                  <Th align="right">Loss</Th>
                  <Th align="right">Duration</Th>
                  <Th align="center">Status</Th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(ep => {
                  const rewardColor = ep.total_reward > 5 ? 'var(--green)' : ep.total_reward > 2 ? 'var(--amber)' : 'var(--red)';
                  const rewardBar = rewardStats
                    ? ((ep.total_reward - rewardStats.min) / Math.max(rewardStats.max - rewardStats.min, 0.01)) * 100
                    : 0;
                  return (
                    <tr
                      key={ep.episode_id}
                      onClick={() => setSelected(ep)}
                      style={{
                        borderTop: '1px solid var(--border)',
                        cursor: 'pointer',
                        transition: 'background 0.1s',
                      }}
                      onMouseEnter={e => (e.currentTarget.style.background = 'var(--bg-secondary)')}
                      onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
                    >
                      <Td>
                        <span style={{ fontVariantNumeric: 'tabular-nums', fontWeight: 500 }}>{ep.episode_num}</span>
                      </Td>
                      <Td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          <MessageSquare size={12} style={{ color: 'var(--text-muted)' }} />
                          <span>{(ep.scenario as Record<string, string>)?.topic ?? (ep.scenario as Record<string, string>)?.type ?? '-'}</span>
                        </div>
                      </Td>
                      <Td align="center">{ep.turns.length}</Td>
                      <Td align="right">
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6, justifyContent: 'flex-end' }}>
                          <div style={{
                            width: 40, height: 4, borderRadius: 2,
                            background: 'var(--bg-tertiary)', overflow: 'hidden',
                          }}>
                            <div style={{
                              width: `${Math.min(100, rewardBar)}%`,
                              height: '100%', borderRadius: 2,
                              background: rewardColor,
                            }} />
                          </div>
                          <span style={{ color: rewardColor, fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>
                            {ep.total_reward.toFixed(3)}
                          </span>
                        </div>
                      </Td>
                      <Td align="right">
                        <span style={{ fontVariantNumeric: 'tabular-nums' }}>{ep.loss?.toFixed(4) ?? '-'}</span>
                      </Td>
                      <Td align="right">
                        <span style={{ color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>{ep.duration_ms}ms</span>
                      </Td>
                      <Td align="center">
                        <span style={{
                          padding: '2px 6px', borderRadius: 4,
                          background: 'rgba(34, 197, 94, 0.1)', color: 'var(--green)', fontSize: 11,
                        }}>
                          {ep.status}
                        </span>
                      </Td>
                    </tr>
                  );
                })}
                {filtered.length === 0 && (
                  <tr>
                    <td colSpan={7} style={{ padding: 32, textAlign: 'center', color: 'var(--text-muted)' }}>
                      {hasFilters ? 'No episodes match your filters' : 'No episodes yet'}
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 8, marginTop: 16 }}>
              <PagBtn onClick={() => setPage(Math.max(0, page - 1))} disabled={page === 0}>
                <ChevronLeft size={14} />
              </PagBtn>
              <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                Page {page + 1} of {totalPages}
              </span>
              <PagBtn onClick={() => setPage(Math.min(totalPages - 1, page + 1))} disabled={page >= totalPages - 1}>
                <ChevronRight size={14} />
              </PagBtn>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function Th({ children, align }: { children: React.ReactNode; align?: string }) {
  return (
    <th style={{
      padding: '10px 12px',
      textAlign: (align as CanvasTextAlign) ?? 'left',
      fontSize: 11,
      fontWeight: 600,
      color: 'var(--text-muted)',
      textTransform: 'uppercase',
      letterSpacing: '0.05em',
    }}>
      {children}
    </th>
  );
}

function Td({ children, align }: { children: React.ReactNode; align?: string }) {
  return (
    <td style={{ padding: '10px 12px', textAlign: (align as CanvasTextAlign) ?? 'left' }}>
      {children}
    </td>
  );
}

function PagBtn({ onClick, disabled, children }: { onClick: () => void; disabled: boolean; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        width: 28, height: 28, borderRadius: 'var(--radius)',
        border: '1px solid var(--border)', background: 'transparent',
        color: disabled ? 'var(--text-muted)' : 'var(--text-primary)',
        opacity: disabled ? 0.4 : 1,
      }}
    >
      {children}
    </button>
  );
}

const selectStyle: React.CSSProperties = {
  padding: '4px 8px', borderRadius: 'var(--radius)', border: '1px solid var(--border)',
  background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontSize: 12,
};

const filterInputStyle: React.CSSProperties = {
  border: 'none', background: 'none', outline: 'none',
  color: 'var(--text-primary)', fontSize: 12,
};

const clearBtnStyle: React.CSSProperties = {
  display: 'flex', alignItems: 'center', justifyContent: 'center',
  border: 'none', background: 'transparent',
  color: 'var(--text-muted)', padding: 2, borderRadius: 4,
};
