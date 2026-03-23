import { useState, useRef, useEffect } from 'react';
import { Send, RotateCcw, Bot, User, Sparkles, Settings, Save, FolderOpen } from 'lucide-react';
import { Card } from './Card';
import { useToast } from '../hooks/useToast';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  reward?: number;
  reward_breakdown?: Record<string, number>;
  timestamp: number;
}

interface SavedSession {
  id: string;
  envType: string;
  messages: Message[];
  totalReward: number;
  savedAt: number;
}

const ENV_OPTIONS = [
  { id: 'conversation', label: 'Conversation' },
  { id: 'customer_support', label: 'Customer Support' },
  { id: 'code_assistant', label: 'Code Assistant' },
  { id: 'reasoning', label: 'Reasoning & Math' },
  { id: 'rag_agent', label: 'RAG Agent' },
  { id: 'safety', label: 'Safety & Alignment' },
];

const STORAGE_KEY = 'playground_sessions';

function loadSessions(): SavedSession[] {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]');
  } catch {
    return [];
  }
}

function saveSessions(sessions: SavedSession[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions.slice(0, 20)));
}

export function Playground() {
  const [envType, setEnvType] = useState('customer_support');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [totalReward, setTotalReward] = useState(0);
  const [turnCount, setTurnCount] = useState(0);
  const scrollRef = useRef<HTMLDivElement>(null);
  const toast = useToast();

  // Settings
  const [showSettings, setShowSettings] = useState(false);
  const [temperature, setTemperature] = useState(0.8);
  const [maxTokens, setMaxTokens] = useState(512);
  const [systemPrompt, setSystemPrompt] = useState('');

  // Saved sessions
  const [showSaved, setShowSaved] = useState(false);
  const [savedSessions, setSavedSessions] = useState<SavedSession[]>(loadSessions);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;
    const userMsg = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMsg, timestamp: Date.now() / 1000 }]);
    setLoading(true);

    try {
      const res = await fetch('/api/lab/playground/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          env_type: envType,
          message: userMsg,
          temperature,
          max_tokens: maxTokens,
          system_prompt: systemPrompt || undefined,
        }),
      });
      const data = await res.json();
      setSessionId(data.session_id);
      setTotalReward(data.total_reward);
      setTurnCount(data.turn_count);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.response,
        reward: data.reward,
        reward_breakdown: data.reward_breakdown,
        timestamp: Date.now() / 1000,
      }]);
    } catch (e) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Error: Could not reach the API.',
        timestamp: Date.now() / 1000,
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    if (sessionId) {
      fetch(`/api/lab/playground/${sessionId}`, { method: 'DELETE' }).catch(() => {});
    }
    setSessionId(null);
    setMessages([]);
    setTotalReward(0);
    setTurnCount(0);
  };

  const handleSaveSession = () => {
    if (messages.length === 0) return;
    const session: SavedSession = {
      id: crypto.randomUUID(),
      envType,
      messages,
      totalReward,
      savedAt: Date.now(),
    };
    const updated = [session, ...savedSessions];
    setSavedSessions(updated);
    saveSessions(updated);
    toast.success('Session saved');
  };

  const handleLoadSession = (session: SavedSession) => {
    handleReset();
    setEnvType(session.envType);
    setMessages(session.messages);
    setTotalReward(session.totalReward);
    setTurnCount(session.messages.filter(m => m.role === 'user').length);
    setShowSaved(false);
    toast.info('Session loaded');
  };

  const handleDeleteSession = (id: string) => {
    const updated = savedSessions.filter(s => s.id !== id);
    setSavedSessions(updated);
    saveSessions(updated);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div style={{ padding: 32, maxWidth: 900, margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: 20 }}>
        <div>
          <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 4 }}>Agent Playground</h2>
          <p style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
            Chat with an agent and see real-time reward scoring.
          </p>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {turnCount > 0 && (
            <div style={{
              padding: '4px 10px', borderRadius: 6, fontSize: 11,
              background: 'rgba(99, 102, 241, 0.1)', border: '1px solid rgba(99, 102, 241, 0.2)',
            }}>
              <span style={{ color: 'var(--text-muted)' }}>Turns: </span>
              <span style={{ fontWeight: 600, color: 'var(--accent-light)' }}>{turnCount}</span>
              <span style={{ color: 'var(--text-muted)', margin: '0 6px' }}>|</span>
              <span style={{ color: 'var(--text-muted)' }}>Total Reward: </span>
              <span style={{ fontWeight: 600, color: 'var(--green)' }}>{totalReward.toFixed(3)}</span>
            </div>
          )}
          <button onClick={() => setShowSaved(!showSaved)} style={iconBtnStyle} title="Saved sessions">
            <FolderOpen size={14} />
          </button>
          <button onClick={handleSaveSession} style={iconBtnStyle} title="Save session">
            <Save size={14} />
          </button>
          <button onClick={() => setShowSettings(!showSettings)} style={iconBtnStyle} title="Settings">
            <Settings size={14} />
          </button>
          <button onClick={handleReset} style={iconBtnStyle} title="Reset conversation">
            <RotateCcw size={14} />
          </button>
        </div>
      </div>

      {/* Settings panel */}
      {showSettings && (
        <Card style={{ marginBottom: 16, padding: 16 }}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 12, color: 'var(--text-secondary)' }}>
            Playground Settings
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div>
              <label style={{ fontSize: 11, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>
                Temperature: {temperature}
              </label>
              <input
                type="range" min="0" max="2" step="0.05" value={temperature}
                onChange={e => setTemperature(+e.target.value)}
                style={{ width: '100%', accentColor: 'var(--accent)' }}
              />
            </div>
            <div>
              <label style={{ fontSize: 11, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>
                Max Tokens: {maxTokens}
              </label>
              <input
                type="range" min="64" max="2048" step="64" value={maxTokens}
                onChange={e => setMaxTokens(+e.target.value)}
                style={{ width: '100%', accentColor: 'var(--accent)' }}
              />
            </div>
          </div>
          <div style={{ marginTop: 12 }}>
            <label style={{ fontSize: 11, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>
              System Prompt (optional)
            </label>
            <textarea
              value={systemPrompt}
              onChange={e => setSystemPrompt(e.target.value)}
              placeholder="You are a helpful assistant..."
              rows={3}
              style={{
                width: '100%', padding: '8px 10px', borderRadius: 6,
                border: '1px solid var(--border)', background: 'var(--bg-primary)',
                color: 'var(--text-primary)', fontSize: 12, resize: 'vertical',
                outline: 'none', lineHeight: 1.5,
              }}
            />
          </div>
        </Card>
      )}

      {/* Saved sessions panel */}
      {showSaved && (
        <Card style={{ marginBottom: 16, padding: 16, maxHeight: 200, overflow: 'auto' }}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 10, color: 'var(--text-secondary)' }}>
            Saved Sessions
          </div>
          {savedSessions.length === 0 ? (
            <div style={{ fontSize: 12, color: 'var(--text-muted)', textAlign: 'center', padding: 16 }}>
              No saved sessions yet
            </div>
          ) : (
            savedSessions.map(s => (
              <div key={s.id} style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                padding: '6px 8px', borderRadius: 6, marginBottom: 4,
                background: 'var(--bg-tertiary)', fontSize: 12,
              }}>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <span style={{ color: 'var(--text-muted)' }}>{s.envType}</span>
                  <span>{s.messages.length} msgs</span>
                  <span style={{ color: 'var(--green)', fontWeight: 500 }}>R: {s.totalReward.toFixed(3)}</span>
                  <span style={{ color: 'var(--text-muted)', fontSize: 10 }}>
                    {new Date(s.savedAt).toLocaleString()}
                  </span>
                </div>
                <div style={{ display: 'flex', gap: 4 }}>
                  <button
                    onClick={() => handleLoadSession(s)}
                    style={{ ...iconBtnStyle, width: 24, height: 24 }}
                  >
                    <FolderOpen size={11} />
                  </button>
                  <button
                    onClick={() => handleDeleteSession(s.id)}
                    style={{ ...iconBtnStyle, width: 24, height: 24, color: 'var(--red)' }}
                  >
                    &times;
                  </button>
                </div>
              </div>
            ))
          )}
        </Card>
      )}

      {/* Environment selector */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 16, flexWrap: 'wrap' }}>
        {ENV_OPTIONS.map(opt => (
          <button
            key={opt.id}
            onClick={() => { setEnvType(opt.id); handleReset(); }}
            style={{
              padding: '5px 12px', borderRadius: 16, fontSize: 12, fontWeight: 500,
              border: envType === opt.id ? '1px solid var(--accent)' : '1px solid var(--border)',
              background: envType === opt.id ? 'rgba(99,102,241,0.1)' : 'transparent',
              color: envType === opt.id ? 'var(--accent-light)' : 'var(--text-secondary)',
              transition: 'all 0.15s',
            }}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {/* Chat area */}
      <Card style={{ padding: 0, overflow: 'hidden', marginBottom: 0 }}>
        <div
          ref={scrollRef}
          style={{
            height: 480,
            overflow: 'auto',
            padding: 20,
            display: 'flex',
            flexDirection: 'column',
            gap: 16,
          }}
        >
          {messages.length === 0 && (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
              <Sparkles size={32} style={{ color: 'var(--text-muted)' }} />
              <div style={{ fontSize: 14, color: 'var(--text-muted)' }}>Start a conversation</div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)', maxWidth: 320, textAlign: 'center' }}>
                Type a message to interact with the agent. Each response is scored
                by the reward model in real-time.
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div
              key={i}
              style={{
                display: 'flex',
                gap: 12,
                flexDirection: msg.role === 'user' ? 'row-reverse' : 'row',
              }}
            >
              {/* Avatar */}
              <div style={{
                width: 32, height: 32, borderRadius: '50%', flexShrink: 0,
                background: msg.role === 'user' ? 'var(--accent)' : 'var(--bg-tertiary)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                {msg.role === 'user' ? <User size={16} color="#fff" /> : <Bot size={16} color="var(--text-secondary)" />}
              </div>

              {/* Bubble */}
              <div style={{
                maxWidth: '75%',
                padding: '10px 14px',
                borderRadius: 12,
                background: msg.role === 'user' ? 'var(--accent)' : 'var(--bg-tertiary)',
                color: msg.role === 'user' ? '#fff' : 'var(--text-primary)',
                fontSize: 13,
                lineHeight: 1.6,
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
              }}>
                {msg.content}

                {/* Reward badge for assistant messages */}
                {msg.role === 'assistant' && msg.reward != null && (
                  <div style={{
                    marginTop: 10,
                    paddingTop: 8,
                    borderTop: '1px solid rgba(255,255,255,0.08)',
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: 4,
                  }}>
                    <span style={{
                      padding: '2px 6px', borderRadius: 4, fontSize: 10, fontWeight: 600,
                      background: msg.reward > 2 ? 'rgba(34,197,94,0.15)' : 'rgba(245,158,11,0.15)',
                      color: msg.reward > 2 ? 'var(--green)' : 'var(--amber)',
                    }}>
                      R: {msg.reward.toFixed(3)}
                    </span>
                    {msg.reward_breakdown && Object.entries(msg.reward_breakdown).map(([k, v]) => (
                      <span key={k} style={{
                        padding: '2px 6px', borderRadius: 4, fontSize: 10,
                        background: 'rgba(255,255,255,0.06)',
                        color: 'var(--text-muted)',
                      }}>
                        {k}: {v.toFixed(3)}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div style={{ display: 'flex', gap: 12 }}>
              <div style={{
                width: 32, height: 32, borderRadius: '50%', flexShrink: 0,
                background: 'var(--bg-tertiary)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <Bot size={16} color="var(--text-secondary)" />
              </div>
              <div style={{
                padding: '12px 16px', borderRadius: 12, background: 'var(--bg-tertiary)',
                fontSize: 13, color: 'var(--text-muted)',
              }}>
                <span className="typing-dots">Thinking</span>
              </div>
            </div>
          )}
        </div>

        {/* Input */}
        <div style={{
          display: 'flex', gap: 8, padding: '12px 16px',
          borderTop: '1px solid var(--border)', background: 'var(--bg-secondary)',
        }}>
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a message..."
            rows={1}
            style={{
              flex: 1, padding: '8px 12px', borderRadius: 'var(--radius)',
              border: '1px solid var(--border)', background: 'var(--bg-primary)',
              color: 'var(--text-primary)', fontSize: 13, resize: 'none',
              outline: 'none', lineHeight: 1.5,
              minHeight: 38, maxHeight: 120,
            }}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || loading}
            style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              width: 38, height: 38, borderRadius: 'var(--radius)',
              border: 'none', background: input.trim() ? 'var(--accent)' : 'var(--bg-tertiary)',
              color: input.trim() ? '#fff' : 'var(--text-muted)',
              transition: 'all 0.15s', flexShrink: 0,
            }}
          >
            <Send size={16} />
          </button>
        </div>
      </Card>
    </div>
  );
}

const iconBtnStyle: React.CSSProperties = {
  display: 'flex', alignItems: 'center', justifyContent: 'center',
  width: 32, height: 32, borderRadius: 'var(--radius)',
  border: '1px solid var(--border)', background: 'transparent',
  color: 'var(--text-secondary)',
};
