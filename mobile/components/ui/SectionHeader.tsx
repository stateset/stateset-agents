import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';

import { useTheme } from '@/theme';

interface SectionHeaderProps {
  title: string;
  subtitle?: string;
  actionLabel?: string;
  onActionPress?: () => void;
}

export function SectionHeader({
  title,
  subtitle,
  actionLabel,
  onActionPress,
}: SectionHeaderProps) {
  const theme = useTheme();

  return (
    <View style={styles.row}>
      <View style={styles.copy}>
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
          {title}
        </Text>
        {subtitle ? (
          <Text
            style={[
              styles.subtitle,
              {
                color: theme.colors.textSecondary,
                fontFamily: theme.fontFamily.body,
                fontSize: theme.typography.caption,
              },
            ]}
          >
            {subtitle}
          </Text>
        ) : null}
      </View>
      {actionLabel && onActionPress ? (
        <Pressable onPress={onActionPress}>
          <Text
            style={[
              styles.action,
              {
                color: theme.colors.primary,
                fontFamily: theme.fontFamily.body,
                fontSize: theme.typography.caption,
              },
            ]}
          >
            {actionLabel}
          </Text>
        </Pressable>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-end',
    gap: 12,
  },
  copy: {
    flex: 1,
    gap: 4,
  },
  title: {
    fontWeight: '700',
  },
  subtitle: {
    lineHeight: 18,
  },
  action: {
    fontWeight: '700',
    letterSpacing: 0.2,
  },
});
