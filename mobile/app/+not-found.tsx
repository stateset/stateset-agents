import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { useRouter } from 'expo-router';

import { Button } from '@/components/ui/Button';
import { EmptyState } from '@/components/ui/EmptyState';
import { Screen } from '@/components/ui/Screen';
import { useTheme } from '@/theme';

export default function NotFoundScreen() {
  const theme = useTheme();
  const router = useRouter();

  return (
    <Screen scroll={false}>
      <View style={styles.wrap}>
        <EmptyState
          icon="alert-circle-outline"
          title="Route not found"
          detail="This screen is outside the fine-tuning lab route tree."
        />
        <Button label="Back to dashboard" onPress={() => router.replace('/(tabs)/dashboard')} />
        <Text
          style={[
            styles.footnote,
            {
              color: theme.colors.textMuted,
              fontFamily: theme.fontFamily.body,
              fontSize: theme.typography.caption,
            },
          ]}
        >
          Check the Expo Router path if you intended to add another modal or tab.
        </Text>
      </View>
    </Screen>
  );
}

const styles = StyleSheet.create({
  wrap: {
    flex: 1,
    justifyContent: 'center',
    gap: 16,
  },
  footnote: {
    textAlign: 'center',
    lineHeight: 18,
  },
});
