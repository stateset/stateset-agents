import { CSSProperties } from 'react';

interface SkeletonProps {
  width?: string | number;
  height?: string | number;
  borderRadius?: string | number;
  style?: CSSProperties;
}

function SkeletonBase({ width = '100%', height = 16, borderRadius = 6, style }: SkeletonProps) {
  return (
    <div
      className="skeleton-shimmer"
      style={{
        width, height, borderRadius,
        background: 'var(--bg-tertiary)',
        ...style,
      }}
    />
  );
}

export function SkeletonLine({ width, height = 14 }: { width?: string | number; height?: number }) {
  return <SkeletonBase width={width} height={height} borderRadius={4} />;
}

export function SkeletonCard() {
  return (
    <div style={{
      background: 'var(--bg-secondary)', border: '1px solid var(--border)',
      borderRadius: 12, padding: 20,
    }}>
      <SkeletonLine width={80} height={10} />
      <div style={{ marginTop: 10 }}>
        <SkeletonBase height={28} width={120} borderRadius={4} />
      </div>
      <div style={{ marginTop: 8 }}>
        <SkeletonLine width={60} height={10} />
      </div>
    </div>
  );
}

export function SkeletonRow() {
  return (
    <div style={{
      background: 'var(--bg-secondary)', border: '1px solid var(--border)',
      borderRadius: 12, padding: 16,
      display: 'flex', alignItems: 'center', gap: 16,
    }}>
      <SkeletonBase width={8} height={8} borderRadius="50%" />
      <div style={{ flex: 1 }}>
        <SkeletonLine width="40%" height={14} />
        <div style={{ marginTop: 6 }}>
          <SkeletonLine width="25%" height={10} />
        </div>
      </div>
      <SkeletonBase width={50} height={24} borderRadius={4} />
      <SkeletonBase width={50} height={24} borderRadius={4} />
      <SkeletonBase width={50} height={24} borderRadius={4} />
    </div>
  );
}

export function SkeletonChart() {
  return (
    <div style={{
      background: 'var(--bg-secondary)', border: '1px solid var(--border)',
      borderRadius: 12, padding: 20,
    }}>
      <SkeletonLine width={120} height={14} />
      <div style={{ marginTop: 16 }}>
        <SkeletonBase height={240} borderRadius={8} />
      </div>
    </div>
  );
}

export function SkeletonTable({ rows = 5 }: { rows?: number }) {
  return (
    <div style={{
      border: '1px solid var(--border)', borderRadius: 12, overflow: 'hidden',
    }}>
      <div style={{ padding: '10px 12px', background: 'var(--bg-secondary)', display: 'flex', gap: 16 }}>
        {[40, 120, 50, 60, 60, 60].map((w, i) => (
          <SkeletonLine key={i} width={w} height={10} />
        ))}
      </div>
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} style={{ padding: '12px', borderTop: '1px solid var(--border)', display: 'flex', gap: 16, alignItems: 'center' }}>
          {[30, 140, 40, 50, 50, 50].map((w, j) => (
            <SkeletonLine key={j} width={w} height={12} />
          ))}
        </div>
      ))}
    </div>
  );
}
