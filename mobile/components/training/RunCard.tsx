import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { Badge } from '@/components/ui/Badge';
import { Card } from '@/components/ui/Card';
import {
  formatCompactNumber,
  formatPercent,
  formatRelativeTime,
  toneForStatus,
} from '@/lib/format';
import type { TrainingRun } from '@/lib/types';
import { useTheme } from '@/theme';

interface RunCardProps {
  run: TrainingRun;
  onPress?: () => void;
}

export function RunCard({ run, onPress }: RunCardProps) {
  const theme = useTheme();

  return (
    <Card onPress={onPress}>
      <View style={styles.header}>
        <View style={styles.headerCopy}>
          <View style={styles.badges}>
            <Badge text={run.status} tone={toneForStatus(run.status)} />
            <Badge text={run.source} tone={run.source === 'live' ? 'success' : 'info'} />
          </View>
          <Text
            style={[
              styles.title,
              {
                color: theme.colors.text,
                fontFamily: theme.fontFamily.display,
                fontSize: theme.typography.subtitle,
              },
            ]}
          >
            {run.name}
          </Text>
        </View>
        <Text
          style={[
            styles.updated,
            {
              color: theme.colors.textMuted,
              fontFamily: theme.fontFamily.body,
              fontSize: theme.typography.caption,
            },
          ]}
        >
          {formatRelativeTime(run.updatedAt)}
        </Text>
      </View>

      <Text
        numberOfLines={2}
        style={[
          styles.description,
          {
            color: theme.colors.textSecondary,
            fontFamily: theme.fontFamily.body,
            fontSize: theme.typography.body,
          },
        ]}
      >
        {run.description}
      </Text>

      <View style={styles.metaRow}>
        <View style={styles.metaBlock}>
          <Text style={[styles.metaLabel, { color: theme.colors.textMuted }]}>Base</Text>
          <Text numberOfLines={1} style={[styles.metaValue, { color: theme.colors.text }]}>{run.baseModel}</Text>
        </View>
        <View style={styles.metaBlock}>
          <Text style={[styles.metaLabel, { color: theme.colors.textMuted }]}>Dataset</Text>
          <Text numberOfLines={1} style={[styles.metaValue, { color: theme.colors.text }]}>{run.datasetName}</Text>
        </View>
      </View>

      <View style={styles.stats}>
        <View style={styles.statBlock}>
          <Text style={[styles.metaLabel, { color: theme.colors.textMuted }]}>Reward</Text>
          <Text style={[styles.statValue, { color: theme.colors.text }]}>{formatPercent(run.metrics.avgReward * 100)}</Text>
        </View>
        <View style={styles.statBlock}>
          <Text style={[styles.metaLabel, { color: theme.colors.textMuted }]}>Progress</Text>
          <Text style={[styles.statValue, { color: theme.colors.text }]}>{formatPercent(run.metrics.progressPercent)}</Text>
        </View>
        <View style={styles.statBlock}>
          <Text style={[styles.metaLabel, { color: theme.colors.textMuted }]}>Episodes</Text>
          <Text style={[styles.statValue, { color: theme.colors.text }]}>{formatCompactNumber(run.metrics.totalEpisodes)}</Text>
        </View>
      </View>

      <View style={styles.tags}>
        {run.tags.map((tag) => (
          <Badge key={tag} text={tag} tone="neutral" />
        ))}
      </View>
    </Card>
  );
}

const styles = StyleSheet.create({
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 12,
  },
  headerCopy: {
    flex: 1,
    gap: 10,
  },
  badges: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  title: {
    fontWeight: '700',
  },
  updated: {
    paddingTop: 4,
  },
  description: {
    lineHeight: 21,
    marginTop: 14,
  },
  metaRow: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 16,
  },
  metaBlock: {
    flex: 1,
    gap: 4,
  },
  metaLabel: {
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 0.5,
    textTransform: 'uppercase',
  },
  metaValue: {
    fontSize: 13,
    fontWeight: '600',
  },
  stats: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 16,
  },
  statBlock: {
    flex: 1,
    gap: 4,
  },
  statValue: {
    fontSize: 18,
    fontWeight: '700',
  },
  tags: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginTop: 16,
  },
});
