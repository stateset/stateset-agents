import React from 'react';
import { Ionicons } from '@expo/vector-icons';
import { StyleSheet, Text, View } from 'react-native';

import { useTheme } from '@/theme';

/**
 * Renders a small, user-visible notice when a screen is showing bundled
 * mock data instead of live data from the training lab API (see
 * `useTrainingData().isMockData` in mobile/hooks/useTrainingData.ts). The
 * fallback already logs a console warning once per session; this banner
 * makes the same fact visible in the UI itself.
 */
export function DemoDataBanner() {
  const theme = useTheme();

  return (
    <View
      style={[
        styles.container,
        {
          backgroundColor: theme.colors.cardMuted,
          borderColor: theme.colors.warning,
          borderRadius: theme.radius.lg,
        },
      ]}
    >
      <Ionicons name="information-circle-outline" size={16} color={theme.colors.warning} />
      <Text
        style={[
          styles.text,
          {
            color: theme.colors.textSecondary,
            fontFamily: theme.fontFamily.body,
            fontSize: theme.typography.micro,
          },
        ]}
      >
        Showing demo data — the training lab API is unreachable or empty.
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    borderWidth: 1,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  text: {
    flexShrink: 1,
  },
});
