import React, { useDeferredValue, useState } from 'react';
import {
  Alert,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { useRouter } from 'expo-router';

import { RunCard } from '@/components/training/RunCard';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { EmptyState } from '@/components/ui/EmptyState';
import { LoadingScreen } from '@/components/ui/LoadingScreen';
import { Screen } from '@/components/ui/Screen';
import { useTrainingData } from '@/hooks/useTrainingData';
import type { RunStatus } from '@/lib/types';
import { useTheme } from '@/theme';

const FILTERS: Array<'all' | RunStatus> = ['all', 'running', 'queued', 'paused', 'completed', 'failed'];

export default function RunsScreen() {
  const theme = useTheme();
  const router = useRouter();
  const {
    runs,
    source,
    isLoading,
    isRefreshing,
    isLaunching,
    refetch,
    launchSampleRun,
  } = useTrainingData();
  const [query, setQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<'all' | RunStatus>('all');
  const deferredQuery = useDeferredValue(query);

  if (isLoading) {
    return <LoadingScreen title="Loading runs" detail="Pulling the latest training run feed." />;
  }

  const normalizedQuery = deferredQuery.trim().toLowerCase();
  const filteredRuns = runs.filter((run) => {
    const matchesStatus = statusFilter === 'all' || run.status === statusFilter;
    const matchesQuery = normalizedQuery.length === 0
      || run.name.toLowerCase().includes(normalizedQuery)
      || run.baseModel.toLowerCase().includes(normalizedQuery)
      || run.datasetName.toLowerCase().includes(normalizedQuery)
      || run.tags.some((tag) => tag.toLowerCase().includes(normalizedQuery));
    return matchesStatus && matchesQuery;
  });

  const handleLaunch = async () => {
    const launched = await launchSampleRun();
    if (launched.source === 'mock') {
      Alert.alert('Preview launch created', 'The API is unreachable, so the app created a local sample run instead.');
    }
    router.push(`/run/${launched.id}`);
  };

  return (
    <Screen refreshing={isRefreshing} onRefresh={() => void refetch()}>
      <View
        style={[
          styles.headerCard,
          {
            backgroundColor: theme.colors.card,
            borderColor: theme.colors.border,
            borderRadius: theme.radius.xl,
          },
        ]}
      >
        <View style={styles.headerTop}>
          <View style={styles.headerCopy}>
            <Badge text={source === 'live' ? 'Live stream' : 'Preview mode'} tone={source === 'live' ? 'success' : 'info'} />
            <Text
              style={[
                styles.title,
                {
                  color: theme.colors.text,
                  fontFamily: theme.fontFamily.display,
                  fontSize: theme.typography.display,
                },
              ]}
            >
              Training runs
            </Text>
          </View>
          <Button label={isLaunching ? 'Launching…' : 'Launch'} size="sm" onPress={() => void handleLaunch()} disabled={isLaunching} />
        </View>
        <Text style={[styles.copy, { color: theme.colors.textSecondary }]}>
          Filter active and historical fine-tunes by status, model family, or dataset.
        </Text>
        <TextInput
          value={query}
          onChangeText={setQuery}
          placeholder="Search by model, run, dataset, or tag"
          placeholderTextColor={theme.colors.textMuted}
          style={[
            styles.input,
            {
              color: theme.colors.text,
              backgroundColor: theme.colors.background,
              borderColor: theme.colors.border,
              borderRadius: theme.radius.pill,
            },
          ]}
        />
      </View>

      <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.filterRow}>
        {FILTERS.map((filter) => (
          <Button
            key={filter}
            label={filter}
            size="sm"
            variant={statusFilter === filter ? 'primary' : 'ghost'}
            onPress={() => setStatusFilter(filter)}
          />
        ))}
      </ScrollView>

      {filteredRuns.length > 0 ? (
        <View style={styles.stack}>
          {filteredRuns.map((run) => (
            <RunCard key={run.id} run={run} onPress={() => router.push(`/run/${run.id}`)} />
          ))}
        </View>
      ) : (
        <EmptyState
          icon="search-outline"
          title="No runs match this filter"
          detail="Broaden the query or switch status filters to reveal more fine-tuning runs."
        />
      )}
    </Screen>
  );
}

const styles = StyleSheet.create({
  headerCard: {
    borderWidth: 1,
    padding: 18,
    gap: 12,
  },
  headerTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 12,
  },
  headerCopy: {
    flex: 1,
    gap: 10,
  },
  title: {
    fontWeight: '700',
  },
  copy: {
    fontSize: 14,
    lineHeight: 20,
  },
  input: {
    borderWidth: 1,
    minHeight: 46,
    paddingHorizontal: 16,
    fontSize: 15,
  },
  filterRow: {
    gap: 10,
  },
  stack: {
    gap: 12,
  },
});
