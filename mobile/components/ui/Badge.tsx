import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { useTheme } from '@/theme';
import type { BadgeTone } from '@/lib/types';

interface BadgeProps {
  text: string;
  tone?: BadgeTone;
}

export function Badge({ text, tone = 'neutral' }: BadgeProps) {
  const theme = useTheme();

  const textColor = {
    neutral: theme.colors.textSecondary,
    info: theme.colors.info,
    active: theme.colors.primary,
    success: theme.colors.success,
    warning: theme.colors.warning,
    danger: theme.colors.danger,
  }[tone];

  return (
    <View
      style={[
        styles.badge,
        {
          backgroundColor: theme.colors.cardMuted,
          borderColor: theme.colors.border,
          borderRadius: theme.radius.pill,
        },
      ]}
    >
      <Text
        style={[
          styles.text,
          {
            color: textColor,
            fontFamily: theme.fontFamily.body,
            fontSize: theme.typography.micro,
          },
        ]}
      >
        {text}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    borderWidth: 1,
    paddingHorizontal: 10,
    paddingVertical: 5,
    alignSelf: 'flex-start',
  },
  text: {
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 0.6,
  },
});
