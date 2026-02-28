import { useEffect, useRef, useCallback } from 'react';
import { useExecutionStore } from '../store/executionStore';
import type { TrainingProgress } from '../types/graph';

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const sessionId = useExecutionStore((s) => s.sessionId);
  const addProgress = useExecutionStore((s) => s.addProgress);
  const setRunning = useExecutionStore((s) => s.setRunning);
  const setPaused = useExecutionStore((s) => s.setPaused);
  const setExecutionId = useExecutionStore((s) => s.setExecutionId);
  const setResults = useExecutionStore((s) => s.setResults);
  const setErrors = useExecutionStore((s) => s.setErrors);
  const clearProgress = useExecutionStore((s) => s.clearProgress);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/training/${sessionId}`);

    ws.onopen = () => {
      console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'training_progress') {
        addProgress(data as TrainingProgress);
      } else if (data.type === 'execution_complete') {
        if (data.results) setResults(data.results);
        setRunning(false);
        setExecutionId(null);
      } else if (data.type === 'execution_error') {
        if (data.error) setErrors([data.error]);
        setRunning(false);
        setExecutionId(null);
      } else if (data.type === 'training_paused') {
        setPaused(true);
      } else if (data.type === 'training_resumed') {
        setPaused(false);
      } else if (data.type === 'training_stopped') {
        clearProgress();
        setRunning(false);
        setExecutionId(null);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected, reconnecting...');
      reconnectTimer.current = setTimeout(connect, 2000);
    };

    wsRef.current = ws;
  }, [sessionId, addProgress, setRunning, setPaused, setExecutionId, setResults, setErrors, clearProgress]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return wsRef;
}
