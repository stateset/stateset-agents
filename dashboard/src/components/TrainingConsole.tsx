import { useState, useEffect, useRef, useMemo } from 'react';
import { Terminal, Search, Download, X, Filter } from 'lucide-react';
import { Card } from './Card';

interface TrainingConsoleProps {
  experimentId: string;
}

interface LogEntry {
  ts: number;
  level: string;
  message: string;
  [key: string]: unknown;
}

type LogLevel = 'info' | 'warn' | 'error' | 'debug';

const ALL_LEVELS: LogLevel[] = ['info', 'warn', 'error', 'debug'];

const levelColor: Record<string, string> = {
  info: 'var(--accent-light)',
  warn: 'var(--amber)',
  error: 'var(--red)',
  debug: 'var(--text-muted)',
};

export function TrainingConsole({ experimentId }: TrainingConsoleProps) {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [autoScroll, setAutoScroll] = useState(true);
  const [search, setSearch] = useState('');
  const [enabledLevels, setEnabledLevels] = useState<Set<LogLevel>>(new Set(ALL_LEVELS));
  const [expanded, setExpanded] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const fetchLogs = () => {
      fetch(`/api/lab/experiments/${experimentId}/logs?limit=200`)
        .then(r => r.json())
        .then(setLogs)
        .catch(() => {});
    };
    fetchLogs();
    const interval = setInterval(fetchLogs, 2000);
    return () => clearInterval(interval);
  }, [experimentId]);

  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  const filtered = useMemo(() => {
    let result = logs;
    if (!enabledLevels.has('info') || !enabledLevels.has('warn') || !enabledLevels.has('error') || !enabledLevels.has('debug')) {
      result = result.filter(l => enabledLevels.has(l.level as LogLevel));
    }
    if (search) {
      const q = search.toLowerCase();
      result = result.filter(l => l.message.toLowerCase().includes(q));
    }
    return result;
  }, [logs, search, enabledLevels]);

  const toggleLevel = (level: LogLevel) => {
    setEnabledLevels(prev => {
      const next = new Set(prev);
      if (next.has(level)) {
        next.delete(level);
      } else {
        next.add(level);
      }
      return next;
    });
  };

  const handleDownload = () => {
    const text = filtered.map(l => {
      const date = new Date(l.ts * 1000);
      const time = date.toLocaleTimeString('en-US', { hour12: false });
      return `[${time}] ${l.level.toUpperCase().padEnd(5)} ${l.message}`;
    }).join('\n');
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `training_logs_${experimentId.slice(0, 8)}.log`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const levelCounts = useMemo(() => {
    const counts: Record<string, number> = { info: 0, warn: 0, error: 0, debug: 0 };
    for (const l of logs) {
      if (l.level in counts) counts[l.level]++;
    }
    return counts;
  }, [logs]);

  return (
    <Card style={{ padding: 0, overflow: 'hidden' }}>
      {/* Header */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '10px 14px', borderBottom: '1px solid var(--border)',
        background: 'var(--bg-tertiary)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, fontWeight: 600 }}>
          <Terminal size={14} />
          Training Console
          <span style={{ fontSize: 11, fontWeight: 400, color: 'var(--text-muted)' }}>
            ({filtered.length}/{logs.length})
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <button
            onClick={() => setExpanded(!expanded)}
            style={toolBtnStyle}
            title={expanded ? 'Collapse' : 'Expand'}
          >
            {expanded ? '↕' : '↕'}
          </button>
          <button onClick={handleDownload} style={toolBtnStyle} title="Download logs">
            <Download size={12} />
          </button>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--text-muted)' }}>
            <input type="checkbox" checked={autoScroll} onChange={e => setAutoScroll(e.target.checked)} />
            Auto-scroll
          </label>
        </div>
      </div>

      {/* Filter bar */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8,
        padding: '6px 14px', borderBottom: '1px solid var(--border)',
        background: 'var(--bg-secondary)',
      }}>
        <Search size={12} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
        <input
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Filter logs..."
          style={{
            flex: 1, border: 'none', background: 'none', outline: 'none',
            color: 'var(--text-primary)', fontSize: 11,
            fontFamily: '"JetBrains Mono", "Fira Code", monospace',
          }}
        />
        {search && (
          <button onClick={() => setSearch('')} style={{ ...toolBtnStyle, padding: 2 }}>
            <X size={10} />
          </button>
        )}

        <div style={{ display: 'flex', gap: 3, marginLeft: 4 }}>
          {ALL_LEVELS.map(level => (
            <button
              key={level}
              onClick={() => toggleLevel(level)}
              style={{
                padding: '1px 6px', borderRadius: 4, fontSize: 9, fontWeight: 600,
                textTransform: 'uppercase',
                border: 'none',
                background: enabledLevels.has(level) ? `color-mix(in srgb, ${levelColor[level]} 18%, transparent)` : 'transparent',
                color: enabledLevels.has(level) ? levelColor[level] : 'var(--text-muted)',
                opacity: enabledLevels.has(level) ? 1 : 0.4,
                transition: 'all 0.15s',
              }}
              title={`${level}: ${levelCounts[level]}`}
            >
              {level} {levelCounts[level] > 0 && <span style={{ fontWeight: 400 }}>({levelCounts[level]})</span>}
            </button>
          ))}
        </div>
      </div>

      {/* Log entries */}
      <div
        ref={scrollRef}
        style={{
          height: expanded ? 480 : 240,
          overflow: 'auto',
          padding: '8px 14px',
          fontFamily: '"JetBrains Mono", "Fira Code", monospace',
          fontSize: 11,
          lineHeight: 1.7,
          background: 'var(--bg-primary)',
          transition: 'height 0.2s',
        }}
      >
        {filtered.length === 0 ? (
          <div style={{ color: 'var(--text-muted)', padding: 20, textAlign: 'center' }}>
            {logs.length === 0 ? 'Waiting for training events...' : 'No logs match your filters'}
          </div>
        ) : (
          filtered.map((log, i) => {
            const date = new Date(log.ts * 1000);
            const time = date.toLocaleTimeString('en-US', { hour12: false, fractionalSecondDigits: 0 });
            const highlighted = search && log.message.toLowerCase().includes(search.toLowerCase());
            return (
              <div key={i} style={{ display: 'flex', gap: 10 }}>
                <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>{time}</span>
                <span style={{
                  color: levelColor[log.level] ?? 'var(--text-secondary)',
                  width: 40,
                  flexShrink: 0,
                  textTransform: 'uppercase',
                  fontWeight: 600,
                }}>
                  {log.level}
                </span>
                <span style={{
                  color: 'var(--text-primary)',
                  background: highlighted ? 'rgba(99,102,241,0.15)' : undefined,
                  borderRadius: highlighted ? 2 : undefined,
                  padding: highlighted ? '0 2px' : undefined,
                }}>
                  {log.message}
                </span>
              </div>
            );
          })
        )}
      </div>
    </Card>
  );
}

const toolBtnStyle: React.CSSProperties = {
  display: 'flex', alignItems: 'center', justifyContent: 'center',
  border: 'none', background: 'transparent',
  color: 'var(--text-muted)', padding: 4, borderRadius: 4,
  fontSize: 12,
};
