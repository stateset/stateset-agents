import { useEffect, useCallback } from 'react';

interface HotkeyMap {
  [key: string]: () => void;
}

/**
 * Global hotkey handler. Keys use format: "ctrl+n", "shift+d", "1", etc.
 * Ignores hotkeys when user is typing in an input/textarea/select.
 */
export function useHotkeys(hotkeys: HotkeyMap) {
  const handler = useCallback(
    (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      const tag = target.tagName.toLowerCase();
      if (tag === 'input' || tag === 'textarea' || tag === 'select' || target.isContentEditable) {
        return;
      }

      for (const [combo, action] of Object.entries(hotkeys)) {
        const parts = combo.toLowerCase().split('+');
        const key = parts[parts.length - 1];
        const needCtrl = parts.includes('ctrl') || parts.includes('meta');
        const needShift = parts.includes('shift');
        const needAlt = parts.includes('alt');

        if (
          e.key.toLowerCase() === key &&
          (needCtrl ? (e.ctrlKey || e.metaKey) : true) &&
          (needShift ? e.shiftKey : true) &&
          (needAlt ? e.altKey : true) &&
          (!needCtrl ? !(e.ctrlKey || e.metaKey) : true) &&
          (!needShift ? !e.shiftKey : true) &&
          (!needAlt ? !e.altKey : true)
        ) {
          e.preventDefault();
          action();
          return;
        }
      }
    },
    [hotkeys],
  );

  useEffect(() => {
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [handler]);
}
