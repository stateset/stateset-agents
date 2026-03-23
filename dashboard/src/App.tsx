import { useState, useEffect, useMemo } from 'react';
import './App.css';
import { Layout } from './components/Layout';
import { Dashboard } from './components/Dashboard';
import { CreateExperiment } from './components/CreateExperiment';
import { LiveMonitor } from './components/LiveMonitor';
import { CompareExperiments } from './components/CompareExperiments';
import { Playground } from './components/Playground';
import { Leaderboard } from './components/Leaderboard';
import { CommandPalette } from './components/CommandPalette';
import { ToastProvider } from './hooks/useToast';
import { useHotkeys } from './hooks/useHotkeys';
import { api } from './api';
import type { Experiment } from './types';

function AppContent() {
  const [view, setView] = useState('dashboard');
  const [selectedExperiment, setSelectedExperiment] = useState<Experiment | null>(null);
  const [cloneSource, setCloneSource] = useState<Experiment | null>(null);
  const [experiments, setExperiments] = useState<Experiment[]>([]);

  // Fetch experiments for NotificationCenter (global awareness)
  useEffect(() => {
    const fetch = async () => {
      try { setExperiments(await api.listExperiments()); } catch { /* api not running */ }
    };
    fetch();
    const interval = setInterval(fetch, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleSelectExperiment = (exp: Experiment) => {
    setSelectedExperiment(exp);
    setView('monitor');
  };

  const handleExperimentCreated = (exp: Experiment) => {
    setSelectedExperiment(exp);
    setCloneSource(null);
    setView('monitor');
  };

  const handleClone = (exp: Experiment) => {
    setCloneSource(exp);
    setView('create');
  };

  // Global keyboard shortcuts
  const hotkeys = useMemo(() => ({
    'n': () => setView('create'),
    'd': () => setView('dashboard'),
    'p': () => setView('playground'),
    'l': () => setView('leaderboard'),
    'c': () => setView('compare'),
  }), []);

  useHotkeys(hotkeys);

  return (
    <>
      <Layout currentView={view} onNavigate={setView} experiments={experiments} onSelectExperiment={handleSelectExperiment}>
        {view === 'dashboard' && (
          <Dashboard
            onSelectExperiment={handleSelectExperiment}
            onNavigate={setView}
            onClone={handleClone}
          />
        )}
        {view === 'create' && (
          <CreateExperiment
            onCreated={handleExperimentCreated}
            cloneSource={cloneSource}
          />
        )}
        {view === 'monitor' && (
          <LiveMonitor
            experiment={selectedExperiment}
            onBack={() => setView('dashboard')}
          />
        )}
        {view === 'compare' && (
          <CompareExperiments onBack={() => setView('dashboard')} />
        )}
        {view === 'playground' && (
          <Playground />
        )}
        {view === 'leaderboard' && (
          <Leaderboard onSelectExperiment={handleSelectExperiment} />
        )}
      </Layout>
      <CommandPalette onNavigate={setView} />
    </>
  );
}

function App() {
  return (
    <ToastProvider>
      <AppContent />
    </ToastProvider>
  );
}

export default App;
