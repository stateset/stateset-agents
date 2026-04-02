import React from 'react';
import {
  Pressable,
  StyleProp,
  StyleSheet,
  View,
  ViewStyle,
} from 'react-native';

import { useTheme } from '@/theme';

type CardTone = 'default' | 'muted';

interface CardProps {
  children: React.ReactNode;
  tone?: CardTone;
  onPress?: () => void;
  style?: StyleProp<ViewStyle>;
}

export function Card({ children, tone = 'default', onPress, style }: CardProps) {
  const theme = useTheme();
  const backgroundColor = tone === 'muted' ? theme.colors.cardMuted : theme.colors.card;

  const content = (
    <View
      style={[
        styles.content,
        {
          backgroundColor,
          borderColor: theme.colors.border,
          borderRadius: theme.radius.lg,
        },
        style,
      ]}
    >
      {children}
    </View>
  );

  if (!onPress) {
    return content;
  }

  return (
    <Pressable onPress={onPress} style={({ pressed }) => [pressed && styles.pressed]}>
      {content}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  content: {
    borderWidth: 1,
    padding: 18,
  },
  pressed: {
    opacity: 0.9,
  },
});
