import { Platform } from 'react-native';

export const typography = {
  hero: 40,
  display: 32,
  title: 24,
  subtitle: 18,
  body: 15,
  bodySmall: 13,
  caption: 12,
  micro: 11,
} as const;

export const fontWeight = {
  regular: '400',
  medium: '500',
  semibold: '600',
  bold: '700',
  heavy: '800',
} as const;

export const fontFamily = {
  display: Platform.select({ ios: 'Georgia', android: 'serif', default: 'Georgia' }) ?? 'Georgia',
  body: Platform.select({ ios: 'Avenir Next', android: 'sans-serif', default: 'System' }) ?? 'System',
  mono: Platform.select({ ios: 'Menlo', android: 'monospace', default: 'monospace' }) ?? 'monospace',
} as const;
