import React from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';

import { useTheme } from '@/theme';

export interface MetricItem {
  label: string;
  value: string;
  helper?: string;
}

interface MetricStripProps {
  items: MetricItem[];
}

export function MetricStrip({ items }: MetricStripProps) {
  const theme = useTheme();

  return (
    <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.row}>
      {items.map((item) => (
        <View
          key={item.label}
          style={[
            styles.card,
            {
              backgroundColor: theme.colors.card,
              borderColor: theme.colors.border,
              borderRadius: theme.radius.lg,
            },
          ]}
        >
          <Text
            style={[
              styles.label,
              {
                color: theme.colors.textMuted,
                fontFamily: theme.fontFamily.body,
                fontSize: theme.typography.micro,
              },
            ]}
          >
            {item.label}
          </Text>
          <Text
            style={[
              styles.value,
              {
                color: theme.colors.text,
                fontFamily: theme.fontFamily.display,
                fontSize: theme.typography.title,
              },
            ]}
          >
            {item.value}
          </Text>
          {item.helper ? (
            <Text
              style={[
                styles.helper,
                {
                  color: theme.colors.textSecondary,
                  fontFamily: theme.fontFamily.body,
                  fontSize: theme.typography.caption,
                },
              ]}
            >
              {item.helper}
            </Text>
          ) : null}
        </View>
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  row: {
    gap: 12,
  },
  card: {
    width: 148,
    borderWidth: 1,
    padding: 16,
    gap: 8,
  },
  label: {
    textTransform: 'uppercase',
    letterSpacing: 0.7,
    fontWeight: '700',
  },
  value: {
    fontWeight: '700',
  },
  helper: {
    lineHeight: 18,
  },
});
