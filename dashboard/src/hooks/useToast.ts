import { useContext } from 'react';
import { ToastContext } from './toast-context';
import type { ToastApi } from './toast-context';

export function useToast(): ToastApi {
  return useContext(ToastContext);
}

export type { ToastApi, ToastType, ToastItem } from './toast-context';
