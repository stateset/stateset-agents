import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { Badge } from '@/components/ui/Badge';
import { Card } from '@/components/ui/Card';
import {
  formatCompactNumber,
  formatRelativeTime,
  toneForStatus,
} from '@/lib/format';
import type { DatasetSummary } from '@/lib/types';
import { useTheme } from '@/theme';

interface DatasetCardProps {
  dataset: DatasetSummary;
}

export function DatasetCard({ dataset }: DatasetCardProps) {
  const theme = useTheme();

  return (
    <Card tone="muted">
      <View style={styles.header}>
        <View style={styles.headerCopy}>
          <Badge text={dataset.status} tone={toneForStatus(dataset.status)} />
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
            {dataset.name}
          </Text>
        </View>
        <Text style={[styles.updated, { color: theme.colors.textMuted }]}>{formatRelativeTime(dataset.updatedAt)}</Text>
      </View>

      <Text style={[styles.domain, { color: theme.colors.primary }]}>{dataset.domain}</Text>
      <Text style={[styles.notes, { color: theme.colors.textSecondary }]}>{dataset.notes}</Text>

      <View style={styles.stats}>
        <View style={styles.statBlock}>
          <Text style={[styles.statLabel, { color: theme.colors.textMuted }]}>Tokens</Text>
          <Text style={[styles.statValue, { color: theme.colors.text }]}>{formatCompactNumber(dataset.tokens)}</Text>
        </View>
        <View style={styles.statBlock}>
          <Text style={[styles.statLabel, { color: theme.colors.textMuted }]}>Examples</Text>
          <Text style={[styles.statValue, { color: theme.colors.text }]}>{formatCompactNumber(dataset.examples)}</Text>
        </View>
        <View style={styles.statBlock}>
          <Text style={[styles.statLabel, { color: theme.colors.textMuted }]}>Format</Text>
          <Text style={[styles.statValue, { color: theme.colors.text }]}>{dataset.format}</Text>
        </View>
      </View>

      <View style={styles.bars}>
        <MetricBar label="Coverage" value={dataset.coverage} />
        <MetricBar label="Quality" value={dataset.quality} />
      </View>
    </Card>
  );
}

function MetricBar({ label, value }: { label: string; value: number }) {
  const theme = useTheme();

  return (
    <View style={styles.barBlock}>
      <View style={styles.barHeader}>
        <Text style={[styles.statLabel, { color: theme.colors.textMuted }]}>{label}</Text>
        <Text style={[styles.statLabel, { color: theme.colors.text }]}>{value}%</Text>
      </View>
      <View style={[styles.track, { backgroundColor: theme.colors.overlay, borderRadius: theme.radius.pill }]}>
        <View
          style={[
            styles.fill,
            {
              backgroundColor: theme.colors.primary,
              borderRadius: theme.radius.pill,
              width: `${Math.max(6, value)}%`,
            },
          ]}
        />
      </View>
    </View>
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
  updated: {
    paddingTop: 4,
    fontSize: 12,
    fontWeight: '600',
  },
  title: {
    fontWeight: '700',
  },
  domain: {
    marginTop: 12,
    fontSize: 12,
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  notes: {
    marginTop: 8,
    fontSize: 14,
    lineHeight: 20,
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
  statLabel: {
    fontSize: 11,
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  statValue: {
    fontSize: 16,
    fontWeight: '700',
  },
  bars: {
    marginTop: 18,
    gap: 10,
  },
  barBlock: {
    gap: 6,
  },
  barHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  track: {
    height: 10,
    overflow: 'hidden',
  },
  fill: {
    height: '100%',
  },
});
