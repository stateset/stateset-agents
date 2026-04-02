import React from 'react';
import { Ionicons } from '@expo/vector-icons';
import { StyleSheet, Text, View } from 'react-native';

import { useTheme } from '@/theme';

interface EmptyStateProps {
  icon: keyof typeof Ionicons.glyphMap;
  title: string;
  detail: string;
}

export function EmptyState({ icon, title, detail }: EmptyStateProps) {
  const theme = useTheme();

  return (
    <View
      style={[
        styles.container,
        {
          backgroundColor: theme.colors.card,
          borderColor: theme.colors.border,
          borderRadius: theme.radius.xl,
        },
      ]}
    >
      <View
        style={[
          styles.iconWrap,
          {
            backgroundColor: theme.colors.primarySoft,
            borderRadius: theme.radius.pill,
          },
        ]}
      >
        <Ionicons name={icon} size={22} color={theme.colors.primary} />
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
        {title}
      </Text>
      <Text
        style={[
          styles.detail,
          {
            color: theme.colors.textSecondary,
            fontFamily: theme.fontFamily.body,
            fontSize: theme.typography.body,
          },
        ]}
      >
        {detail}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    borderWidth: 1,
    padding: 22,
    alignItems: 'center',
    gap: 10,
  },
  iconWrap: {
    width: 48,
    height: 48,
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    fontWeight: '700',
  },
  detail: {
    textAlign: 'center',
    lineHeight: 21,
  },
});
