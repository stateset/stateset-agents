import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { API_BASE } from '@/lib/api';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { LoadingScreen } from '@/components/ui/LoadingScreen';
import { Screen } from '@/components/ui/Screen';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { useTrainingData } from '@/hooks/useTrainingData';
import type { ThemeMode } from '@/theme';
import { useTheme } from '@/theme';

const MODES: ThemeMode[] = ['system', 'light', 'dark'];

export default function MoreScreen() {
  const theme = useTheme();
  const {
    algorithms,
    environments,
    source,
    isLoading,
    isRefreshing,
    refetch,
  } = useTrainingData();

  if (isLoading) {
    return <LoadingScreen title="Loading lab settings" detail="Pulling algorithms, environments, and endpoint status." />;
  }

  return (
    <Screen refreshing={isRefreshing} onRefresh={() => void refetch()}>
      <Card>
        <View style={styles.topRow}>
          <View style={styles.topCopy}>
            <Badge text={source === 'live' ? 'Endpoint healthy' : 'Preview only'} tone={source === 'live' ? 'success' : 'warning'} />
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
              Lab settings
            </Text>
          </View>
          <Button label="Refresh" size="sm" variant="secondary" onPress={() => void refetch()} />
        </View>
        <Text style={[styles.copy, { color: theme.colors.textSecondary }]}>
          Mobile endpoint status, theme mode, RL algorithms, and environment presets exposed by the training lab.
        </Text>
        <View
          style={[
            styles.endpoint,
            {
              backgroundColor: theme.colors.background,
              borderColor: theme.colors.border,
              borderRadius: theme.radius.lg,
            },
          ]}
        >
          <Text style={[styles.endpointLabel, { color: theme.colors.textMuted }]}>API base</Text>
          <Text style={[styles.endpointValue, { color: theme.colors.text }]}>{API_BASE}</Text>
        </View>
      </Card>

      <SectionHeader title="Appearance" subtitle="Switch between system, light, and dark themes." />
      <View style={styles.modeRow}>
        {MODES.map((mode) => (
          <Button
            key={mode}
            label={mode}
            size="sm"
            variant={theme.mode === mode ? 'primary' : 'ghost'}
            onPress={() => theme.setMode(mode)}
          />
        ))}
      </View>

      <SectionHeader title="Algorithms" subtitle="Optimization strategies surfaced by `/api/lab/algorithms`." />
      <View style={styles.stack}>
        {algorithms.map((algorithm) => (
          <Card key={algorithm.id} tone="muted">
            <Text style={[styles.cardTitle, { color: theme.colors.text }]}>{algorithm.name}</Text>
            <Text style={[styles.cardBody, { color: theme.colors.textSecondary }]}>{algorithm.description}</Text>
          </Card>
        ))}
      </View>

      <SectionHeader title="Environment presets" subtitle="Default scenario envelopes available for training runs." />
      <View style={styles.stack}>
        {environments.map((environment) => (
          <Card key={environment.id}>
            <View style={styles.presetHeader}>
              <Text style={[styles.cardTitle, { color: theme.colors.text }]}>{environment.name}</Text>
              <Badge text={environment.difficulty} tone="warning" />
            </View>
            <Text style={[styles.cardBody, { color: theme.colors.textSecondary }]}>{environment.description}</Text>
            <Text style={[styles.presetMeta, { color: theme.colors.textMuted }]}>Default max turns: {environment.maxTurnsDefault}</Text>
          </Card>
        ))}
      </View>
    </Screen>
  );
}

const styles = StyleSheet.create({
  topRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 12,
  },
  topCopy: {
    flex: 1,
    gap: 10,
  },
  title: {
    fontWeight: '700',
  },
  copy: {
    fontSize: 14,
    lineHeight: 20,
    marginTop: 12,
  },
  endpoint: {
    borderWidth: 1,
    marginTop: 16,
    padding: 14,
    gap: 6,
  },
  endpointLabel: {
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 0.5,
    textTransform: 'uppercase',
  },
  endpointValue: {
    fontSize: 13,
    fontWeight: '600',
  },
  modeRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  stack: {
    gap: 12,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '700',
  },
  cardBody: {
    marginTop: 8,
    fontSize: 14,
    lineHeight: 20,
  },
  presetHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 12,
    alignItems: 'center',
  },
  presetMeta: {
    marginTop: 12,
    fontSize: 12,
    fontWeight: '600',
  },
});
