import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { Stack, useLocalSearchParams, useRouter } from 'expo-router';

import { MetricStrip } from '@/components/training/MetricStrip';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { EmptyState } from '@/components/ui/EmptyState';
import { LoadingScreen } from '@/components/ui/LoadingScreen';
import { Screen } from '@/components/ui/Screen';
import { SectionHeader } from '@/components/ui/SectionHeader';
import {
  formatCompactNumber,
  formatDurationMs,
  formatLearningRate,
  formatPercent,
  formatRelativeTime,
  toneForStatus,
} from '@/lib/format';
import { useTrainingRun } from '@/hooks/useTrainingData';
import { useTheme } from '@/theme';

export default function RunDetailScreen() {
  const router = useRouter();
  const theme = useTheme();
  const params = useLocalSearchParams<{ id?: string | string[] }>();
  const runId = Array.isArray(params.id) ? params.id[0] ?? '' : params.id ?? '';
  const runQuery = useTrainingRun(runId);

  if (runQuery.isLoading) {
    return <LoadingScreen title="Loading run detail" detail="Pulling metrics, events, and episode slices for this run." />;
  }

  const run = runQuery.data?.run;
  const source = runQuery.data?.source ?? 'mock';

  if (!run) {
    return (
      <Screen>
        <EmptyState
          icon="warning-outline"
          title="Run not found"
          detail="The selected training run could not be loaded from the API or local preview state."
        />
      </Screen>
    );
  }

  const configItems = [
    ['Batch size', String(run.config.batchSize)],
    ['Learning rate', formatLearningRate(run.config.learningRate)],
    ['Generations', String(run.config.numGenerations)],
    ['LoRA rank', String(run.config.loraRank)],
    ['Quantization', run.config.quantization],
    ['Max turns', String(run.config.maxTurns)],
    ['Environment', run.config.environment],
    ['Difficulty', run.config.difficulty],
  ];

  return (
    <>
      <Stack.Screen options={{ headerShown: false }} />
      <Screen>
        <View style={styles.topBar}>
          <Button label="Close" variant="ghost" size="sm" onPress={() => router.back()} />
          <Text style={[styles.topBarMeta, { color: theme.colors.textMuted }]}>{formatRelativeTime(run.updatedAt)}</Text>
        </View>

        <View
          style={[
            styles.hero,
            {
              backgroundColor: theme.colors.card,
              borderColor: theme.colors.border,
              borderRadius: theme.radius.xl,
            },
          ]}
        >
          <View style={styles.heroBadges}>
            <Badge text={run.status} tone={toneForStatus(run.status)} />
            <Badge text={source === 'live' ? 'live detail' : 'preview detail'} tone={source === 'live' ? 'success' : 'info'} />
          </View>
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
            {run.name}
          </Text>
          <Text style={[styles.description, { color: theme.colors.textSecondary }]}>{run.description}</Text>
          <View style={styles.heroMeta}>
            <Text style={[styles.heroMetaText, { color: theme.colors.text }]}>Base: {run.baseModel}</Text>
            <Text style={[styles.heroMetaText, { color: theme.colors.text }]}>Dataset: {run.datasetName}</Text>
            <Text style={[styles.heroMetaText, { color: theme.colors.text }]}>Algorithm: {run.algorithm}</Text>
          </View>
        </View>

        <MetricStrip
          items={[
            { label: 'Avg reward', value: formatPercent(run.metrics.avgReward * 100), helper: 'Current mean reward' },
            { label: 'Best reward', value: formatPercent(run.metrics.bestReward * 100), helper: 'Best observed episode' },
            { label: 'Loss', value: run.metrics.loss.toFixed(2), helper: 'Latest training loss' },
            { label: 'Episodes', value: formatCompactNumber(run.metrics.totalEpisodes), helper: formatPercent(run.metrics.progressPercent) + ' progress' },
          ]}
        />

        <SectionHeader title="Run config" subtitle="The current optimizer and environment envelope for this fine-tune." />
        <Card tone="muted">
          <View style={styles.configGrid}>
            {configItems.map(([label, value]) => (
              <View key={label} style={styles.configItem}>
                <Text style={[styles.configLabel, { color: theme.colors.textMuted }]}>{label}</Text>
                <Text style={[styles.configValue, { color: theme.colors.text }]}>{value}</Text>
              </View>
            ))}
          </View>
        </Card>

        <SectionHeader title="Recent events" subtitle="Latest notes and milestones recorded for this run." />
        <Card>
          <View style={styles.eventList}>
            {run.recentEvents.map((event) => (
              <View key={event} style={styles.eventRow}>
                <View style={[styles.eventDot, { backgroundColor: theme.colors.primary }]} />
                <Text style={[styles.eventText, { color: theme.colors.textSecondary }]}>{event}</Text>
              </View>
            ))}
          </View>
        </Card>

        <SectionHeader title="Episodes" subtitle="Recent episode slices and reward outcomes." />
        {run.episodes.length > 0 ? (
          <View style={styles.episodeStack}>
            {run.episodes.map((episode) => (
              <Card key={episode.id} tone="muted">
                <View style={styles.episodeHeader}>
                  <Text style={[styles.episodeTitle, { color: theme.colors.text }]}>Episode {episode.episodeNumber}</Text>
                  <Badge text={episode.status} tone={toneForStatus(episode.status)} />
                </View>
                <Text style={[styles.episodeSummary, { color: theme.colors.textSecondary }]}>{episode.summary}</Text>
                <View style={styles.episodeMeta}>
                  <Text style={[styles.episodeMetaText, { color: theme.colors.text }]}>Reward {formatPercent(episode.reward * 100)}</Text>
                  <Text style={[styles.episodeMetaText, { color: theme.colors.text }]}>Turns {episode.turns}</Text>
                  <Text style={[styles.episodeMetaText, { color: theme.colors.text }]}>Duration {formatDurationMs(episode.durationMs)}</Text>
                </View>
              </Card>
            ))}
          </View>
        ) : (
          <EmptyState
            icon="list-outline"
            title="No episodes yet"
            detail="This run has not emitted episode slices yet, or the backend does not expose them."
          />
        )}
      </Screen>
    </>
  );
}

const styles = StyleSheet.create({
  topBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  topBarMeta: {
    fontSize: 12,
    fontWeight: '600',
  },
  hero: {
    borderWidth: 1,
    padding: 18,
    gap: 12,
  },
  heroBadges: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  title: {
    fontWeight: '700',
  },
  description: {
    fontSize: 14,
    lineHeight: 21,
  },
  heroMeta: {
    gap: 6,
  },
  heroMetaText: {
    fontSize: 13,
    fontWeight: '600',
  },
  configGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 14,
  },
  configItem: {
    width: '47%',
    gap: 4,
  },
  configLabel: {
    fontSize: 11,
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  configValue: {
    fontSize: 14,
    fontWeight: '600',
  },
  eventList: {
    gap: 12,
  },
  eventRow: {
    flexDirection: 'row',
    gap: 10,
    alignItems: 'flex-start',
  },
  eventDot: {
    width: 8,
    height: 8,
    borderRadius: 999,
    marginTop: 6,
  },
  eventText: {
    flex: 1,
    fontSize: 14,
    lineHeight: 20,
  },
  episodeStack: {
    gap: 12,
  },
  episodeHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 12,
    alignItems: 'center',
  },
  episodeTitle: {
    fontSize: 18,
    fontWeight: '700',
  },
  episodeSummary: {
    marginTop: 10,
    fontSize: 14,
    lineHeight: 20,
  },
  episodeMeta: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginTop: 14,
  },
  episodeMetaText: {
    fontSize: 13,
    fontWeight: '700',
  },
});
