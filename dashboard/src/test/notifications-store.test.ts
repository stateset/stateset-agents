import { describe, it, expect, beforeEach } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import {
  clearAllNotifications,
  markAllRead,
  markRead,
  pushNotification,
  useNotifications,
} from '../hooks/notifications-store';

const sampleNotification = (id: string, read = false) => ({
  id,
  type: 'started' as const,
  title: 'exp',
  detail: 'msg',
  timestamp: Date.now() / 1000,
  read,
  experimentId: 'e-1',
});

describe('notifications-store', () => {
  beforeEach(() => clearAllNotifications());

  it('pushes new notifications to the front', () => {
    const { result } = renderHook(() => useNotifications());
    expect(result.current).toEqual([]);

    act(() => pushNotification(sampleNotification('a')));
    act(() => pushNotification(sampleNotification('b')));

    expect(result.current.map(n => n.id)).toEqual(['b', 'a']);
  });

  it('caps the log at 50 entries', () => {
    const { result } = renderHook(() => useNotifications());
    act(() => {
      for (let i = 0; i < 60; i++) pushNotification(sampleNotification(`n-${i}`));
    });
    expect(result.current).toHaveLength(50);
    expect(result.current[0].id).toBe('n-59');
  });

  it('marks notifications read', () => {
    const { result } = renderHook(() => useNotifications());
    act(() => pushNotification(sampleNotification('a')));
    act(() => markRead('a'));
    expect(result.current[0].read).toBe(true);
  });

  it('marks all read', () => {
    const { result } = renderHook(() => useNotifications());
    act(() => {
      pushNotification(sampleNotification('a'));
      pushNotification(sampleNotification('b'));
    });
    act(() => markAllRead());
    expect(result.current.every(n => n.read)).toBe(true);
  });
});
