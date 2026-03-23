import { Zap, FlaskConical, Activity, MessageCircle, GitCompareArrows, Trophy } from 'lucide-react';

interface OnboardingProps {
  onNavigate: (view: string) => void;
}

const STEPS = [
  {
    icon: FlaskConical,
    title: 'Create an Experiment',
    description: 'Pick an environment, configure your agent, and set training parameters.',
    action: 'create',
    color: 'var(--accent)',
  },
  {
    icon: Activity,
    title: 'Monitor Training',
    description: 'Watch real-time charts, episode transcripts, and reward curves.',
    action: 'monitor',
    color: 'var(--green)',
  },
  {
    icon: MessageCircle,
    title: 'Test in Playground',
    description: 'Chat with your agent and see per-turn reward scoring live.',
    action: 'playground',
    color: 'var(--cyan)',
  },
  {
    icon: GitCompareArrows,
    title: 'Compare Results',
    description: 'Overlay training curves, compare hyperparameters, find the best config.',
    action: 'compare',
    color: 'var(--amber)',
  },
  {
    icon: Trophy,
    title: 'Leaderboard',
    description: 'Rank experiments by reward, convergence, or loss.',
    action: 'leaderboard',
    color: 'var(--purple)',
  },
];

export function Onboarding({ onNavigate }: OnboardingProps) {
  return (
    <div style={{ maxWidth: 700, margin: '0 auto', padding: '48px 32px' }}>
      <div style={{ textAlign: 'center', marginBottom: 40 }}>
        <div style={{
          width: 56, height: 56, borderRadius: 16, margin: '0 auto 16px',
          background: 'linear-gradient(135deg, var(--accent), var(--purple))',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <Zap size={28} color="#fff" />
        </div>
        <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 8 }}>
          Welcome to the Training Lab
        </h1>
        <p style={{ fontSize: 14, color: 'var(--text-secondary)', maxWidth: 420, margin: '0 auto' }}>
          Train, evaluate, and compare RL agents across different environments.
          Get started by creating your first experiment.
        </p>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {STEPS.map((step, i) => {
          const Icon = step.icon;
          return (
            <button
              key={step.action}
              onClick={() => onNavigate(step.action)}
              style={{
                display: 'flex', alignItems: 'center', gap: 16,
                padding: '16px 20px', borderRadius: 'var(--radius-lg)',
                border: '1px solid var(--border)', background: 'var(--bg-secondary)',
                color: 'var(--text-primary)', textAlign: 'left', width: '100%',
                transition: 'all 0.15s',
              }}
              onMouseEnter={e => {
                (e.currentTarget as HTMLElement).style.borderColor = step.color;
                (e.currentTarget as HTMLElement).style.transform = 'translateX(4px)';
              }}
              onMouseLeave={e => {
                (e.currentTarget as HTMLElement).style.borderColor = 'var(--border)';
                (e.currentTarget as HTMLElement).style.transform = 'translateX(0)';
              }}
            >
              <div style={{
                width: 40, height: 40, borderRadius: 10, flexShrink: 0,
                background: `color-mix(in srgb, ${step.color} 12%, transparent)`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <Icon size={20} style={{ color: step.color }} />
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 2 }}>
                  <span style={{ color: 'var(--text-muted)', fontSize: 12, marginRight: 8 }}>{i + 1}.</span>
                  {step.title}
                </div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{step.description}</div>
              </div>
              <div style={{ fontSize: 18, color: 'var(--text-muted)' }}>&rsaquo;</div>
            </button>
          );
        })}
      </div>

      <div style={{
        textAlign: 'center', marginTop: 32, fontSize: 12, color: 'var(--text-muted)',
      }}>
        Press <kbd style={kbdStyle}>&#8984;K</kbd> anytime to search commands
        &nbsp;&middot;&nbsp;
        <kbd style={kbdStyle}>N</kbd> new experiment
        &nbsp;&middot;&nbsp;
        <kbd style={kbdStyle}>D</kbd> dashboard
      </div>
    </div>
  );
}

const kbdStyle: React.CSSProperties = {
  padding: '2px 6px', borderRadius: 4,
  background: 'var(--bg-tertiary)', border: '1px solid var(--border)',
  fontSize: 10, fontWeight: 600, fontFamily: 'monospace',
};
