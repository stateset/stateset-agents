import React from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { useTheme } from '@/theme';

interface LoadingScreenProps {
  title?: string;
  detail?: string;
}

export function LoadingScreen({
  title = 'Preparing the lab',
  detail = 'Loading fine-tuning runs, datasets, and model state.',
}: LoadingScreenProps) {
  const theme = useTheme();

  return (
    <SafeAreaView style={[styles.safe, { backgroundColor: theme.colors.background }]}>
      <View
        style={[
          styles.card,
          {
            backgroundColor: theme.colors.card,
            borderColor: theme.colors.border,
            borderRadius: theme.radius.xl,
          },
        ]}
      >
        <ActivityIndicator size="large" color={theme.colors.primary} />
        <Text
          style={[
            styles.title,
            {
              color: theme.colors.text,
              fontFamily: theme.fontFamily.display,
              fontSize: theme.typography.title,
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
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    justifyContent: 'center',
    padding: 20,
  },
  card: {
    borderWidth: 1,
    padding: 24,
    gap: 12,
  },
  title: {
    fontWeight: '700',
  },
  detail: {
    lineHeight: 22,
  },
});
