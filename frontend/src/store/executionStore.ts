import { create } from 'zustand';
import type { TrainingProgress } from '../types/graph';

export interface AblationRun {
  id: string;
  name: string;
  history: { epoch: number; train_loss: number; val_loss: number | null }[];
  finalTrainLoss: number | null;
  finalValLoss: number | null;
}

interface ExecutionState {
  isRunning: boolean;
  isPaused: boolean;
  sessionId: string;
  executionId: string | null;
  progress: TrainingProgress[];
  pipelineStatus: string | null;
  results: Record<string, unknown>;
  errors: string[];
  ablationRuns: AblationRun[];

  setRunning: (running: boolean) => void;
  setPaused: (paused: boolean) => void;
  setSessionId: (id: string) => void;
  setExecutionId: (id: string | null) => void;
  addProgress: (p: TrainingProgress, executionId?: string) => void;
  setPipelineStatus: (s: string | null) => void;
  setResults: (r: Record<string, unknown>) => void;
  setErrors: (e: string[]) => void;
  clearProgress: (executionId?: string) => void;
  addAblationRun: (run: AblationRun) => void;
  clearAblationRuns: () => void;
}

export const useExecutionStore = create<ExecutionState>((set) => ({
  isRunning: false,
  isPaused: false,
  sessionId: `session_${Date.now()}`,
  executionId: null,
  progress: [],
  pipelineStatus: null,
  results: {},
  errors: [],
  ablationRuns: [],

  setRunning: (running) => set({ isRunning: running, ...(running ? {} : { isPaused: false, pipelineStatus: null }) }),
  setPaused: (paused) => set({ isPaused: paused }),
  setSessionId: (id) => set({ sessionId: id }),
  setExecutionId: (id) => set({ executionId: id }),
  addProgress: (p, executionId) => set((s) => {
    if (!s.isRunning) return s;
    if (executionId && s.executionId && executionId !== s.executionId) return s;
    return { progress: [...s.progress, p], pipelineStatus: null };
  }),
  setPipelineStatus: (s) => set({ pipelineStatus: s }),
  setResults: (r) => set({ results: r }),
  setErrors: (e) => set({ errors: e }),
  clearProgress: (executionId) => set((s) => {
    if (executionId && s.executionId && executionId !== s.executionId) return s;
    return { progress: [], errors: [], isPaused: false };
  }),
  addAblationRun: (run) => set((s) => ({ ablationRuns: [...s.ablationRuns, run] })),
  clearAblationRuns: () => set({ ablationRuns: [] }),
}));
