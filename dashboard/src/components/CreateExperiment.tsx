import { useState, useEffect } from 'react';
import {
  MessageSquare, Headphones, Code, Brain, Search, Shield,
  ChevronRight, Zap,
} from 'lucide-react';
import { Card } from './Card';
import { api } from '../api';
import type { EnvironmentPreset, AlgorithmInfo, Experiment } from '../types';

const iconMap: Record<string, typeof MessageSquare> = {
  MessageSquare, Headphones, Code, Brain, Search, Shield,
};

interface CreateExperimentProps {
  onCreated: (exp: Experiment) => void;
  cloneSource?: Experiment | null;
}

export function CreateExperiment({ onCreated, cloneSource }: CreateExperimentProps) {
  const [envs, setEnvs] = useState<EnvironmentPreset[]>([]);
  const [algos, setAlgos] = useState<AlgorithmInfo[]>([]);
  const [step, setStep] = useState(0);

  // Form state
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [envType, setEnvType] = useState('conversation');
  const [difficulty, setDifficulty] = useState<'easy' | 'medium' | 'hard' | 'expert'>('medium');
  const [maxTurns, setMaxTurns] = useState(10);
  const [algorithm, setAlgorithm] = useState('grpo');
  const [numEpisodes, setNumEpisodes] = useState(100);
  const [learningRate, setLearningRate] = useState(1e-5);
  const [batchSize, setBatchSize] = useState(8);
  const [temperature, setTemperature] = useState(0.8);
  const [modelName, setModelName] = useState('gpt2');
  const [useStub, setUseStub] = useState(true);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    api.listEnvironments().then(setEnvs).catch(() => {});
    api.listAlgorithms().then(setAlgos).catch(() => {});
  }, []);

  // Pre-fill from clone source
  useEffect(() => {
    if (!cloneSource) return;
    setName(`${cloneSource.name} (copy)`);
    setDescription(cloneSource.description ?? '');
    setEnvType(cloneSource.environment.env_type);
    setDifficulty(cloneSource.environment.difficulty);
    setMaxTurns(cloneSource.environment.max_turns);
    setAlgorithm(cloneSource.training.algorithm);
    setNumEpisodes(cloneSource.training.num_episodes);
    setLearningRate(cloneSource.training.learning_rate);
    setBatchSize(cloneSource.training.batch_size);
    setTemperature(cloneSource.agent.temperature);
    setModelName(cloneSource.agent.model_name);
    setUseStub(cloneSource.agent.use_stub);
    setStep(3); // Jump to review
  }, [cloneSource]);

  const selectedEnv = envs.find(e => e.id === envType);

  const handleSubmit = async () => {
    if (!name.trim()) return;
    setSubmitting(true);
    try {
      const exp = await api.createExperiment({
        name,
        description,
        environment: {
          env_type: envType,
          max_turns: maxTurns,
          difficulty,
          reward_weights: Object.fromEntries(
            (selectedEnv?.reward_components ?? []).map(c => [c, 1 / (selectedEnv?.reward_components.length ?? 1)])
          ),
        },
        agent: {
          model_name: modelName,
          use_stub: useStub,
          temperature,
        },
        training: {
          num_episodes: numEpisodes,
          learning_rate: learningRate,
          batch_size: batchSize,
          algorithm: algorithm as Experiment['training']['algorithm'],
        },
      });
      onCreated(exp);
    } catch (e) {
      console.error(e);
    } finally {
      setSubmitting(false);
    }
  };

  const steps = ['Environment', 'Agent', 'Training', 'Review'];

  return (
    <div style={{ padding: 32, maxWidth: 900, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4 }}>
        {cloneSource ? 'Clone Experiment' : 'New Experiment'}
      </h2>
      <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 28 }}>
        {cloneSource
          ? `Cloning from "${cloneSource.name}". Adjust parameters and launch.`
          : 'Configure your RL training environment, agent, and training parameters.'}
      </p>

      {/* Step indicator */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 32 }}>
        {steps.map((s, i) => (
          <button
            key={s}
            onClick={() => setStep(i)}
            style={{
              flex: 1,
              padding: '10px 0',
              border: 'none',
              borderRadius: 'var(--radius)',
              background: i === step ? 'var(--accent)' : i < step ? 'var(--accent-dim)' : 'var(--bg-tertiary)',
              color: i <= step ? '#fff' : 'var(--text-muted)',
              fontSize: 12,
              fontWeight: 600,
              transition: 'all 0.2s',
            }}
          >
            {i + 1}. {s}
          </button>
        ))}
      </div>

      {/* Step 0: Environment */}
      {step === 0 && (
        <div>
          <div style={{ marginBottom: 20 }}>
            <label style={labelStyle}>Experiment Name</label>
            <input
              value={name}
              onChange={e => setName(e.target.value)}
              placeholder="My GRPO Experiment"
              style={inputStyle}
            />
          </div>
          <div style={{ marginBottom: 20 }}>
            <label style={labelStyle}>Description</label>
            <textarea
              value={description}
              onChange={e => setDescription(e.target.value)}
              placeholder="Optional description..."
              rows={2}
              style={{ ...inputStyle, resize: 'vertical' }}
            />
          </div>

          <label style={labelStyle}>Environment Type</label>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, marginBottom: 20 }}>
            {envs.map(env => {
              const Icon = iconMap[env.icon] ?? MessageSquare;
              const selected = envType === env.id;
              return (
                <Card
                  key={env.id}
                  hover
                  onClick={() => {
                    setEnvType(env.id);
                    setMaxTurns(env.max_turns_default);
                  }}
                  style={{
                    padding: 14,
                    border: selected ? '1px solid var(--accent)' : undefined,
                    background: selected ? 'rgba(99, 102, 241, 0.06)' : undefined,
                  }}
                >
                  <Icon size={18} style={{ color: selected ? 'var(--accent-light)' : 'var(--text-muted)', marginBottom: 6 }} />
                  <div style={{ fontSize: 13, fontWeight: 600 }}>{env.name}</div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>{env.description}</div>
                </Card>
              );
            })}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div>
              <label style={labelStyle}>Difficulty</label>
              <select value={difficulty} onChange={e => setDifficulty(e.target.value as typeof difficulty)} style={inputStyle}>
                <option value="easy">Easy</option>
                <option value="medium">Medium</option>
                <option value="hard">Hard</option>
                <option value="expert">Expert</option>
              </select>
            </div>
            <div>
              <label style={labelStyle}>Max Turns</label>
              <input type="number" value={maxTurns} onChange={e => setMaxTurns(+e.target.value)} min={1} max={100} style={inputStyle} />
            </div>
          </div>

          {selectedEnv && (
            <div style={{ marginTop: 16 }}>
              <label style={labelStyle}>Reward Components</label>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                {selectedEnv.reward_components.map(c => (
                  <span key={c} style={{
                    padding: '3px 10px',
                    borderRadius: 12,
                    background: 'var(--bg-tertiary)',
                    fontSize: 11,
                    color: 'var(--text-secondary)',
                  }}>
                    {c}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Step 1: Agent */}
      {step === 1 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div>
            <label style={labelStyle}>Model</label>
            <input value={modelName} onChange={e => setModelName(e.target.value)} style={inputStyle} placeholder="gpt2" />
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <input type="checkbox" checked={useStub} onChange={e => setUseStub(e.target.checked)} id="stub" />
            <label htmlFor="stub" style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
              Use stub backend (no GPU required)
            </label>
          </div>
          <div>
            <label style={labelStyle}>Temperature: {temperature}</label>
            <input
              type="range" min="0" max="2" step="0.05"
              value={temperature}
              onChange={e => setTemperature(+e.target.value)}
              style={{ width: '100%', accentColor: 'var(--accent)' }}
            />
          </div>
        </div>
      )}

      {/* Step 2: Training */}
      {step === 2 && (
        <div>
          <label style={labelStyle}>Algorithm</label>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 8, marginBottom: 20 }}>
            {algos.map(a => (
              <button
                key={a.id}
                onClick={() => setAlgorithm(a.id)}
                style={{
                  padding: '10px 8px',
                  border: algorithm === a.id ? '1px solid var(--accent)' : '1px solid var(--border)',
                  borderRadius: 'var(--radius)',
                  background: algorithm === a.id ? 'rgba(99,102,241,0.08)' : 'var(--bg-secondary)',
                  color: algorithm === a.id ? 'var(--accent-light)' : 'var(--text-secondary)',
                  fontSize: 12,
                  fontWeight: 600,
                  textAlign: 'center',
                }}
              >
                <div>{a.name}</div>
                <div style={{ fontSize: 10, fontWeight: 400, color: 'var(--text-muted)', marginTop: 2 }}>
                  {a.description.split(' ').slice(0, 3).join(' ')}
                </div>
              </button>
            ))}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
            <div>
              <label style={labelStyle}>Episodes</label>
              <input type="number" value={numEpisodes} onChange={e => setNumEpisodes(+e.target.value)} min={1} style={inputStyle} />
            </div>
            <div>
              <label style={labelStyle}>Learning Rate</label>
              <input type="number" value={learningRate} onChange={e => setLearningRate(+e.target.value)} step={1e-6} style={inputStyle} />
            </div>
            <div>
              <label style={labelStyle}>Batch Size</label>
              <input type="number" value={batchSize} onChange={e => setBatchSize(+e.target.value)} min={1} style={inputStyle} />
            </div>
          </div>
        </div>
      )}

      {/* Step 3: Review */}
      {step === 3 && (
        <Card style={{ background: 'var(--bg-tertiary)' }}>
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 16 }}>
            {cloneSource ? 'Cloned Experiment Summary' : 'Experiment Summary'}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <ReviewItem label="Name" value={name || '(untitled)'} />
            <ReviewItem label="Environment" value={envType} />
            <ReviewItem label="Difficulty" value={difficulty} />
            <ReviewItem label="Max Turns" value={String(maxTurns)} />
            <ReviewItem label="Algorithm" value={algorithm.toUpperCase()} />
            <ReviewItem label="Model" value={modelName} />
            <ReviewItem label="Episodes" value={String(numEpisodes)} />
            <ReviewItem label="Learning Rate" value={String(learningRate)} />
            <ReviewItem label="Batch Size" value={String(batchSize)} />
            <ReviewItem label="Temperature" value={String(temperature)} />
            <ReviewItem label="Stub Backend" value={useStub ? 'Yes' : 'No'} />
          </div>
        </Card>
      )}

      {/* Navigation */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 28 }}>
        <button
          onClick={() => setStep(Math.max(0, step - 1))}
          disabled={step === 0}
          style={{
            ...btnStyle,
            opacity: step === 0 ? 0.3 : 1,
          }}
        >
          Back
        </button>
        {step < 3 ? (
          <button onClick={() => setStep(step + 1)} style={{ ...btnStyle, ...btnPrimary }}>
            Next <ChevronRight size={14} />
          </button>
        ) : (
          <button
            onClick={handleSubmit}
            disabled={submitting || !name.trim()}
            style={{
              ...btnStyle,
              ...btnPrimary,
              background: 'var(--green)',
              opacity: submitting || !name.trim() ? 0.5 : 1,
            }}
          >
            <Zap size={14} />
            {submitting ? 'Creating...' : 'Create & Launch'}
          </button>
        )}
      </div>
    </div>
  );
}

function ReviewItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{label}</div>
      <div style={{ fontSize: 13, fontWeight: 500 }}>{value}</div>
    </div>
  );
}

const labelStyle: React.CSSProperties = {
  display: 'block',
  fontSize: 12,
  fontWeight: 500,
  color: 'var(--text-secondary)',
  marginBottom: 6,
};

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '8px 12px',
  borderRadius: 'var(--radius)',
  border: '1px solid var(--border)',
  background: 'var(--bg-primary)',
  color: 'var(--text-primary)',
  fontSize: 13,
  outline: 'none',
};

const btnStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 6,
  padding: '8px 20px',
  borderRadius: 'var(--radius)',
  border: '1px solid var(--border)',
  background: 'var(--bg-secondary)',
  color: 'var(--text-primary)',
  fontSize: 13,
  fontWeight: 500,
};

const btnPrimary: React.CSSProperties = {
  background: 'var(--accent)',
  border: 'none',
  color: '#fff',
};
