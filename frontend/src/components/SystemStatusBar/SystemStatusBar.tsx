import { useSystemMonitor, type SystemStats } from '../../hooks/useSystemMonitor';

export function SystemStatusBar() {
  const stats = useSystemMonitor();

  if (!stats) {
    return (
      <div style={barStyle}>
        <span style={{ color: '#555', fontSize: 11 }}>Connecting to system monitor...</span>
      </div>
    );
  }

  return (
    <div style={barStyle}>
      <Metric label="CPU" value={stats.cpu} />
      <Sep />
      <Metric label="RAM" value={stats.ram} />
      {stats.gpu_mem_used != null && stats.gpu_mem_total != null && (
        <>
          <Sep />
          <VramMetric used={stats.gpu_mem_used} total={stats.gpu_mem_total} />
        </>
      )}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  const pct = Math.min(100, Math.max(0, value));
  const color = pct < 60 ? '#22c55e' : pct < 85 ? '#eab308' : '#ef4444';
  const filled = Math.round(pct / 100 * 8);

  return (
    <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
      <span style={{ color: '#888', fontSize: 10, fontWeight: 600, minWidth: 24 }}>{label}</span>
      <span style={{ fontFamily: 'monospace', fontSize: 10, letterSpacing: -0.5 }}>
        <span style={{ color }}>{'\u2588'.repeat(filled)}</span>
        <span style={{ color: '#1e1e2e' }}>{'\u2588'.repeat(8 - filled)}</span>
      </span>
      <span style={{ color: '#999', fontSize: 10, fontFamily: 'monospace', minWidth: 28, textAlign: 'right' }}>
        {Math.round(pct)}%
      </span>
    </span>
  );
}

function VramMetric({ used, total }: { used: number; total: number }) {
  const pct = total > 0 ? (used / total) * 100 : 0;
  const color = pct < 60 ? '#22c55e' : pct < 85 ? '#eab308' : '#ef4444';
  const filled = Math.round(pct / 100 * 8);

  return (
    <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
      <span style={{ color: '#888', fontSize: 10, fontWeight: 600 }}>VRAM</span>
      <span style={{ fontFamily: 'monospace', fontSize: 10, letterSpacing: -0.5 }}>
        <span style={{ color }}>{'\u2588'.repeat(filled)}</span>
        <span style={{ color: '#1e1e2e' }}>{'\u2588'.repeat(8 - filled)}</span>
      </span>
      <span style={{ color: '#999', fontSize: 10, fontFamily: 'monospace' }}>
        {used} / {total} GB
      </span>
    </span>
  );
}

function Sep() {
  return <span style={{ color: '#2a2a3e', margin: '0 6px', fontSize: 10 }}>|</span>;
}

const barStyle: React.CSSProperties = {
  background: '#0d0d16',
  borderBottom: '1px solid #1a1a2e',
  padding: '3px 16px',
  display: 'flex',
  alignItems: 'center',
  gap: 4,
  height: 24,
  flexShrink: 0,
};
