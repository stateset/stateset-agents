import React from 'react';
import { Alert, StyleSheet, Text, View } from 'react-native';
import { useRouter } from 'expo-router';

import { MetricStrip } from '@/components/training/MetricStrip';
import { ModelCard } from '@/components/training/ModelCard';
import { RunCard } from '@/components/training/RunCard';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { EmptyState } from '@/components/ui/EmptyState';
import { LoadingScreen } from '@/components/ui/LoadingScreen';
import { Screen } from '@/components/ui/Screen';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { formatPercent } from '@/lib/format';
import { useTrainingData } from '@/hooks/useTrainingData';
import { useTheme } from '@/theme';

export default function DashboardScreen() {
  const theme = useTheme();
  const router = useRouter();
  const {
    runs,
    datasets,
    models,
    snapshot,
    source,
    isLoading,
    isRefreshing,
    isLaunching,
    refetch,
    launchSampleRun,
  } = useTrainingData();

  if (isLoading) {
    return <LoadingScreen />;
  }

  const featuredRuns = runs.slice(0, 3);
  const featuredModels = models.slice(0, 2);
  const topDataset = datasets[0];

  const handleLaunch = async () => {
    const launched = await launchSampleRun();
    if (launched.source === 'mock') {
      Alert.alert('Preview launch created', 'The training lab API is unavailable, so the app created a local preview run.');
    }
    router.push(`/run/${launched.id}`);
  };

  return (
    <Screen refreshing={isRefreshing} onRefresh={() => void refetch()}>
      <View
        style={[
          styles.hero,
          {
            backgroundColor: theme.colors.hero,
            borderRadius: theme.radius.xl,
          },
        ]}
      >
        <View style={[styles.heroOrbLarge, { backgroundColor: theme.colors.heroSoft }]} />
        <View style={[styles.heroOrbSmall, { backgroundColor: theme.colors.accent }]} />
        <Badge text={source === 'live' ? 'Live API' : 'Mock preview'} tone={source === 'live' ? 'success' : 'info'} />
        <Text
          style={[
            styles.heroTitle,
            {
              color: theme.isDark ? theme.colors.background : '#FFF7F2',
              fontFamily: theme.fontFamily.display,
              fontSize: theme.typography.hero,
            },
          ]}
        >
          Fine-tune models from the lab floor.
        </Text>
        <Text
          style={[
            styles.heroBody,
            {
              color: theme.isDark ? theme.colors.textSecondary : '#F7D8CB',
              fontFamily: theme.fontFamily.body,
              fontSize: theme.typography.body,
            },
          ]}
        >
          Launch RL fine-tunes, watch reward movement, and keep datasets and checkpoints in the same StateSet mobile shell used by the console apps.
        </Text>
        <View style={styles.heroActions}>
          <Button label={isLaunching ? 'Launching…' : 'Launch sample run'} onPress={() => void handleLaunch()} disabled={isLaunching} />
          <Button label="Open runs" variant="secondary" onPress={() => router.push('/(tabs)/runs')} />
        </View>
      </View>

      <MetricStrip
        items={[
          { label: 'Active runs', value: String(snapshot.activeRuns), helper: 'Currently training' },
          { label: 'Queue depth', value: String(snapshot.queueDepth), helper: 'Waiting for slots' },
          { label: 'Avg reward', value: formatPercent(snapshot.avgReward * 100), helper: 'Across completed episodes' },
          { label: 'Deployable', value: String(snapshot.deployableModels), helper: 'Models ready for promotion' },
        ]}
      />

      <SectionHeader
        title="Recent runs"
        subtitle="The most recent fine-tuning activity in the lab."
        actionLabel="All runs"
        onActionPress={() => router.push('/(tabs)/runs')}
      />
      {featuredRuns.length > 0 ? (
        <View style={styles.stack}>
          {featuredRuns.map((run) => (
            <RunCard key={run.id} run={run} onPress={() => router.push(`/run/${run.id}`)} />
          ))}
        </View>
      ) : (
        <EmptyState
          icon="rocket-outline"
          title="No runs yet"
          detail="Launch a sample fine-tune to seed the dashboard and verify the API wiring."
        />
      )}

      {topDataset ? (
        <>
          <SectionHeader
            title="Dataset pulse"
            subtitle="Highest readiness pack for the next adapter wave."
            actionLabel="Datasets"
            onActionPress={() => router.push('/(tabs)/datasets')}
          />
          <View
            style={[
              styles.datasetSpotlight,
              {
                backgroundColor: theme.colors.cardMuted,
                borderColor: theme.colors.border,
                borderRadius: theme.radius.xl,
              },
            ]}
          >
            <Text style={[styles.kicker, { color: theme.colors.primary }]}>Ready now</Text>
            <Text
              style={[
                styles.datasetTitle,
                {
                  color: theme.colors.text,
                  fontFamily: theme.fontFamily.display,
                  fontSize: theme.typography.title,
                },
              ]}
            >
              {topDataset.name}
            </Text>
            <Text style={[styles.datasetNotes, { color: theme.colors.textSecondary }]}>{topDataset.notes}</Text>
            <View style={styles.datasetStats}>
              <Text style={[styles.datasetStat, { color: theme.colors.text }]}>{topDataset.coverage}% coverage</Text>
              <Text style={[styles.datasetStat, { color: theme.colors.text }]}>{topDataset.quality}% quality</Text>
              <Text style={[styles.datasetStat, { color: theme.colors.text }]}>{topDataset.examples.toLocaleString()} examples</Text>
            </View>
          </View>
        </>
      ) : null}

      <SectionHeader
        title="Model shelf"
        subtitle="Current bases, in-flight adapters, and deployable checkpoints."
        actionLabel="Models"
        onActionPress={() => router.push('/(tabs)/models')}
      />
      <View style={styles.stack}>
        {featuredModels.map((model) => (
          <ModelCard key={model.id} model={model} />
        ))}
      </View>
    </Screen>
  );
}

const styles = StyleSheet.create({
  hero: {
    overflow: 'hidden',
    padding: 22,
    gap: 14,
  },
  heroOrbLarge: {
    position: 'absolute',
    width: 180,
    height: 180,
    borderRadius: 999,
    opacity: 0.18,
    top: -60,
    right: -40,
  },
  heroOrbSmall: {
    position: 'absolute',
    width: 96,
    height: 96,
    borderRadius: 999,
    opacity: 0.16,
    bottom: -18,
    left: -10,
  },
  heroTitle: {
    fontWeight: '700',
    maxWidth: '85%',
  },
  heroBody: {
    lineHeight: 22,
    maxWidth: '92%',
  },
  heroActions: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  stack: {
    gap: 12,
  },
  datasetSpotlight: {
    borderWidth: 1,
    padding: 18,
    gap: 10,
  },
  kicker: {
    fontSize: 12,
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  datasetTitle: {
    fontWeight: '700',
  },
  datasetNotes: {
    fontSize: 14,
    lineHeight: 20,
  },
  datasetStats: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  datasetStat: {
    fontSize: 13,
    fontWeight: '700',
  },
});
