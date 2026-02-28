import { useEffect, useRef, useState, useCallback } from 'react';

export interface SystemStats {
  cpu: number;
  ram: number;
  gpu_util: number | null;
  gpu_mem_used: number | null;
  gpu_mem_total: number | null;
}

export function useSystemMonitor(): SystemStats | null {
  const [stats, setStats] = useState<SystemStats | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/system/monitor`);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'system_status') {
        setStats({
          cpu: data.cpu,
          ram: data.ram,
          gpu_util: data.gpu_util,
          gpu_mem_used: data.gpu_mem_used,
          gpu_mem_total: data.gpu_mem_total,
        });
      }
    };

    ws.onclose = () => {
      reconnectTimer.current = setTimeout(connect, 3000);
    };

    wsRef.current = ws;
  }, []);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return stats;
}
