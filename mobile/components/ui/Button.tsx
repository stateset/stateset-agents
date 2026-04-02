import React from 'react';
import {
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';

import { useTheme } from '@/theme';

type ButtonVariant = 'primary' | 'secondary' | 'ghost';
type ButtonSize = 'md' | 'sm';

interface ButtonProps {
  label: string;
  onPress?: () => void;
  variant?: ButtonVariant;
  size?: ButtonSize;
  disabled?: boolean;
  fullWidth?: boolean;
}

export function Button({
  label,
  onPress,
  variant = 'primary',
  size = 'md',
  disabled = false,
  fullWidth = false,
}: ButtonProps) {
  const theme = useTheme();
  const isPrimary = variant === 'primary';
  const isSecondary = variant === 'secondary';

  const backgroundColor = isPrimary
    ? theme.colors.primary
    : isSecondary
      ? theme.colors.primarySoft
      : 'transparent';

  const textColor = isPrimary
    ? theme.isDark
      ? '#1A1614'
      : '#FFFDFC'
    : isSecondary
      ? theme.colors.primary
      : theme.colors.text;

  return (
    <Pressable
      disabled={disabled}
      onPress={onPress}
      style={({ pressed }) => [
        styles.base,
        size === 'sm' ? styles.small : styles.medium,
        {
          backgroundColor,
          borderRadius: theme.radius.pill,
          borderColor: variant === 'ghost' ? theme.colors.border : backgroundColor,
          alignSelf: fullWidth ? 'stretch' : 'flex-start',
          opacity: disabled ? 0.45 : pressed ? 0.9 : 1,
        },
      ]}
    >
      <View>
        <Text
          style={[
            styles.label,
            {
              color: textColor,
              fontFamily: theme.fontFamily.body,
              fontSize: size === 'sm' ? theme.typography.caption : theme.typography.body,
            },
          ]}
        >
          {label}
        </Text>
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  base: {
    borderWidth: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  medium: {
    minHeight: 44,
    paddingHorizontal: 16,
    paddingVertical: 10,
  },
  small: {
    minHeight: 34,
    paddingHorizontal: 12,
    paddingVertical: 7,
  },
  label: {
    fontWeight: '700',
    letterSpacing: 0.2,
  },
});
