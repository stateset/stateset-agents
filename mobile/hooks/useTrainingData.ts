import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { trainingApi } from '@/lib/api';
import {
  createMockLaunchRun,
  mockAlgorithms,
  mockDatasets,
  mockEnvironments,
  mockModels,
  mockRuns,
} from '@/lib/mockData';
import type {
  DashboardSnapshot,
  TrainingCatalog,
  TrainingRun,
} from '@/lib/types';

type RunFeed = {
  runs: TrainingRun[];
  source: 'live' | 'mock';
};

type RunDetailFeed = {
  run: TrainingRun;
  source: 'live' | 'mock';
};

const RUNS_QUERY_KEY = ['training', 'runs'] as const;
const CATALOG_QUERY_KEY = ['training', 'catalog'] as const;

async function loadRuns(): Promise<RunFeed> {
  try {
    const runs = await trainingApi.listRuns();
    if (runs.length === 0) {
      return { runs: mockRuns, source: 'mock' };
    }
    return { runs, source: 'live' };
  } catch {
    return { runs: mockRuns, source: 'mock' };
  }
}

async function loadCatalog(): Promise<TrainingCatalog> {
  try {
    const [algorithms, environments] = await Promise.all([
      trainingApi.listAlgorithms(),
      trainingApi.listEnvironments(),
    ]);

    return {
      algorithms: algorithms.length > 0 ? algorithms : mockAlgorithms,
      environments: environments.length > 0 ? environments : mockEnvironments,
    };
  } catch {
    return {
      algorithms: mockAlgorithms,
      environments: mockEnvironments,
    };
  }
}

function average(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function buildSnapshot(runs: TrainingRun[]): DashboardSnapshot {
  const rewardRuns = runs.filter((run) => run.metrics.totalEpisodes > 0);
  const avgReward = average(rewardRuns.map((run) => run.metrics.avgReward));
  const avgLoss = average(rewardRuns.map((run) => run.metrics.loss));

  return {
    activeRuns: runs.filter((run) => run.status === 'running').length,
    queueDepth: runs.filter((run) => run.status === 'queued').length,
    avgReward,
    avgLoss,
    readyDatasets: mockDatasets.filter((dataset) => dataset.status === 'ready').length,
    deployableModels: mockModels.filter((model) => model.status === 'deployed' || model.status === 'available').length,
  };
}

function mergeRuns(nextRun: TrainingRun, currentRuns: TrainingRun[]): TrainingRun[] {
  const seen = new Set<string>();
  return [nextRun, ...currentRuns].filter((run) => {
    if (seen.has(run.id)) return false;
    seen.add(run.id);
    return true;
  });
}

export function useTrainingData() {
  const queryClient = useQueryClient();
  const runsQuery = useQuery({
    queryKey: RUNS_QUERY_KEY,
    queryFn: loadRuns,
  });
  const catalogQuery = useQuery({
    queryKey: CATALOG_QUERY_KEY,
    queryFn: loadCatalog,
  });

  const launchMutation = useMutation({
    mutationFn: async () => {
      try {
        return await trainingApi.launchSampleRun();
      } catch {
        return createMockLaunchRun();
      }
    },
    onSuccess: (run) => {
      queryClient.setQueryData<RunFeed | undefined>(RUNS_QUERY_KEY, (current) => ({
        source: current?.source === 'live' && run.source === 'live' ? 'live' : 'mock',
        runs: mergeRuns(run, current?.runs ?? mockRuns),
      }));

      queryClient.setQueryData<RunDetailFeed>(['training', 'run', run.id], {
        run,
        source: run.source,
      });
    },
  });

  const runs = runsQuery.data?.runs ?? mockRuns;
  const source = runsQuery.data?.source ?? 'mock';

  return {
    runs,
    datasets: mockDatasets,
    models: mockModels,
    algorithms: catalogQuery.data?.algorithms ?? mockAlgorithms,
    environments: catalogQuery.data?.environments ?? mockEnvironments,
    snapshot: buildSnapshot(runs),
    source,
    isLoading: runsQuery.isLoading,
    isRefreshing: runsQuery.isFetching || catalogQuery.isFetching,
    isLaunching: launchMutation.isPending,
    refetch: async () => {
      await Promise.all([runsQuery.refetch(), catalogQuery.refetch()]);
    },
    launchSampleRun: launchMutation.mutateAsync,
  };
}

export function useTrainingRun(id: string) {
  const queryClient = useQueryClient();

  return useQuery({
    queryKey: ['training', 'run', id],
    enabled: id.length > 0,
    queryFn: async (): Promise<RunDetailFeed> => {
      try {
        const run = await trainingApi.getRunDetail(id);
        return { run, source: 'live' };
      } catch {
        const cachedRuns = queryClient.getQueryData<RunFeed>(RUNS_QUERY_KEY)?.runs ?? mockRuns;
        const run = cachedRuns.find((item) => item.id === id) ?? mockRuns.find((item) => item.id === id);

        if (!run) {
          throw new Error('Training run not found');
        }

        return {
          run,
          source: run.source,
        };
      }
    },
  });
}
