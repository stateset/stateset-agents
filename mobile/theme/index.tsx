import React, { createContext, useContext, useMemo, useState } from 'react';
import { useColorScheme } from 'react-native';

import { darkColors, lightColors, type Colors } from './colors';
import { radius, spacing } from './spacing';
import { fontFamily, fontWeight, typography } from './typography';

export type ThemeMode = 'system' | 'light' | 'dark';

export interface Theme {
  colors: Colors;
  spacing: typeof spacing;
  radius: typeof radius;
  typography: typeof typography;
  fontWeight: typeof fontWeight;
  fontFamily: typeof fontFamily;
  isDark: boolean;
  mode: ThemeMode;
  setMode: (mode: ThemeMode) => void;
}

const ThemeContext = createContext<Theme | null>(null);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const systemScheme = useColorScheme();
  const [mode, setMode] = useState<ThemeMode>('system');
  const isDark = mode === 'dark' || (mode === 'system' && systemScheme === 'dark');

  const value = useMemo<Theme>(
    () => ({
      colors: isDark ? darkColors : lightColors,
      spacing,
      radius,
      typography,
      fontWeight,
      fontFamily,
      isDark,
      mode,
      setMode,
    }),
    [isDark, mode],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): Theme {
  const theme = useContext(ThemeContext);
  if (!theme) throw new Error('useTheme must be used within ThemeProvider');
  return theme;
}
