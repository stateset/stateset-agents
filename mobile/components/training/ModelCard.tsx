import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { Badge } from '@/components/ui/Badge';
import { Card } from '@/components/ui/Card';
import { formatRelativeTime, toneForStatus } from '@/lib/format';
import type { ModelSummary } from '@/lib/types';
import { useTheme } from '@/theme';

interface ModelCardProps {
  model: ModelSummary;
}

export function ModelCard({ model }: ModelCardProps) {
  const theme = useTheme();

  return (
    <Card>
      <View style={styles.header}>
        <View style={styles.headerCopy}>
          <View style={styles.badges}>
            <Badge text={model.status} tone={toneForStatus(model.status)} />
            <Badge text={model.checkpointType} tone="neutral" />
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
            {model.name}
          </Text>
        </View>
        <Text style={[styles.updated, { color: theme.colors.textMuted }]}>{formatRelativeTime(model.updatedAt)}</Text>
      </View>

      <Text style={[styles.role, { color: theme.colors.primary }]}>{model.family} · {model.role}</Text>
      <Text style={[styles.summary, { color: theme.colors.textSecondary }]}>{model.summary}</Text>

      <View style={styles.metaRow}>
        <View style={styles.metaBlock}>
          <Text style={[styles.metaLabel, { color: theme.colors.textMuted }]}>Base model</Text>
          <Text style={[styles.metaValue, { color: theme.colors.text }]}>{model.baseModel}</Text>
        </View>
      </View>

      <View style={styles.stats}>
        <View style={styles.statBlock}>
          <Text style={[styles.metaLabel, { color: theme.colors.textMuted }]}>Eval</Text>
          <Text style={[styles.statValue, { color: theme.colors.text }]}>{model.evalScore}%</Text>
        </View>
        <View style={styles.statBlock}>
          <Text style={[styles.metaLabel, { color: theme.colors.textMuted }]}>Latency</Text>
          <Text style={[styles.statValue, { color: theme.colors.text }]}>{model.latencyMs || '—'}{model.latencyMs ? ' ms' : ''}</Text>
        </View>
      </View>

      <View style={styles.tags}>
        {model.tags.map((tag) => (
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
    fontSize: 12,
    fontWeight: '600',
  },
  role: {
    marginTop: 12,
    fontSize: 12,
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  summary: {
    marginTop: 8,
    lineHeight: 20,
    fontSize: 14,
  },
  metaRow: {
    marginTop: 16,
  },
  metaBlock: {
    gap: 4,
  },
  metaLabel: {
    fontSize: 11,
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
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
