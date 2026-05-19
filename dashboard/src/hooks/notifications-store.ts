import { useSyncExternalStore } from 'react';

export interface Notification {
  id: string;
  type: 'started' | 'paused' | 'completed' | 'failed' | 'milestone';
  title: string;
  detail: string;
  timestamp: number;
  read: boolean;
  experimentId: string;
}

type Listener = () => void;

let notifications: Notification[] = [];
const listeners = new Set<Listener>();

function emit() {
  for (const l of listeners) l();
}

export function pushNotification(n: Notification) {
  notifications = [n, ...notifications].slice(0, 50);
  emit();
}

export function markRead(id: string) {
  notifications = notifications.map(n => (n.id === id ? { ...n, read: true } : n));
  emit();
}

export function markAllRead() {
  notifications = notifications.map(n => ({ ...n, read: true }));
  emit();
}

export function clearAllNotifications() {
  notifications = [];
  emit();
}

function subscribe(cb: Listener) {
  listeners.add(cb);
  return () => {
    listeners.delete(cb);
  };
}

function getSnapshot() {
  return notifications;
}

export function useNotifications(): Notification[] {
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}
