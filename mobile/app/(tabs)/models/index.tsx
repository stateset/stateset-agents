import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { MetricStrip } from '@/components/training/MetricStrip';
import { ModelCard } from '@/components/training/ModelCard';
import { Badge } from '@/components/ui/Badge';
import { LoadingScreen } from '@/components/ui/LoadingScreen';
import { Screen } from '@/components/ui/Screen';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { useTrainingData } from '@/hooks/useTrainingData';
import { useTheme } from '@/theme';

export default function ModelsScreen() {
  const theme = useTheme();
  const { models, source, isLoading, isRefreshing, refetch } = useTrainingData();

  if (isLoading) {
    return <LoadingScreen title="Loading models" detail="Syncing the model registry and checkpoint shelf." />;
  }

  const deployedCount = models.filter((model) => model.status === 'deployed').length;
  const trainingCount = models.filter((model) => model.status === 'training').length;
  const averageEval = models.length > 0
    ? Math.round(models.reduce((total, model) => total + model.evalScore, 0) / models.length)
    : 0;

  return (
    <Screen refreshing={isRefreshing} onRefresh={() => void refetch()}>
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
        <Badge text={source === 'live' ? 'Registry online' : 'Preview registry'} tone={source === 'live' ? 'success' : 'info'} />
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
          Checkpoints and bases
        </Text>
        <Text style={[styles.copy, { color: theme.colors.textSecondary }]}>
          Track which models are still training, which adapters are warming, and which checkpoints are safe to promote.
        </Text>
      </View>

      <MetricStrip
        items={[
          { label: 'Deployed', value: String(deployedCount), helper: 'Serving or ready for traffic' },
          { label: 'Training', value: String(trainingCount), helper: 'Still accumulating reward' },
          { label: 'Avg eval', value: `${averageEval}%`, helper: 'Across the current shelf' },
          { label: 'Families', value: String(new Set(models.map((model) => model.family)).size), helper: 'Distinct base families' },
        ]}
      />

      <SectionHeader
        title="Model shelf"
        subtitle="Active bases, in-flight adapters, and merged checkpoints."
      />
      <View style={styles.stack}>
        {models.map((model) => (
          <ModelCard key={model.id} model={model} />
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
