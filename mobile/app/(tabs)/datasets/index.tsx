import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { DatasetCard } from '@/components/training/DatasetCard';
import { MetricStrip } from '@/components/training/MetricStrip';
import { Badge } from '@/components/ui/Badge';
import { LoadingScreen } from '@/components/ui/LoadingScreen';
import { Screen } from '@/components/ui/Screen';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { useTrainingData } from '@/hooks/useTrainingData';
import { formatCompactNumber } from '@/lib/format';
import { useTheme } from '@/theme';

export default function DatasetsScreen() {
  const theme = useTheme();
  const { datasets, source, isLoading, isRefreshing, refetch } = useTrainingData();

  if (isLoading) {
    return <LoadingScreen title="Loading datasets" detail="Preparing the dataset registry for mobile review." />;
  }

  const readyCount = datasets.filter((dataset) => dataset.status === 'ready').length;
  const reviewCount = datasets.filter((dataset) => dataset.status === 'needs-review').length;
  const tokenTotal = datasets.reduce((total, dataset) => total + dataset.tokens, 0);

  return (
    <Screen refreshing={isRefreshing} onRefresh={() => void refetch()}>
      <View
        style={[
          styles.hero,
          {
            backgroundColor: theme.colors.cardMuted,
            borderColor: theme.colors.border,
            borderRadius: theme.radius.xl,
          },
        ]}
      >
        <Badge text={source === 'live' ? 'Lab linked' : 'Preview catalog'} tone={source === 'live' ? 'success' : 'info'} />
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
          Dataset packs
        </Text>
        <Text style={[styles.copy, { color: theme.colors.textSecondary }]}>
          Review readiness, token volume, and coverage before launching the next adapter pass.
        </Text>
      </View>

      <MetricStrip
        items={[
          { label: 'Ready', value: String(readyCount), helper: 'Datasets launchable now' },
          { label: 'Review', value: String(reviewCount), helper: 'Require curation or relabeling' },
          { label: 'Tokens', value: formatCompactNumber(tokenTotal), helper: 'Across the visible catalog' },
          { label: 'Formats', value: String(new Set(datasets.map((dataset) => dataset.format)).size), helper: 'Distinct dataset storage types' },
        ]}
      />

      <SectionHeader
        title="All datasets"
        subtitle="Coverage, quality, and packaging state for each training pack."
      />
      <View style={styles.stack}>
        {datasets.map((dataset) => (
          <DatasetCard key={dataset.id} dataset={dataset} />
        ))}
      </View>
    </Screen>
  );
}

const styles = StyleSheet.create({
  hero: {
    borderWidth: 1,
    padding: 18,
    gap: 12,
  },
  title: {
    fontWeight: '700',
  },
  copy: {
    fontSize: 14,
    lineHeight: 20,
  },
  stack: {
    gap: 12,
  },
});
