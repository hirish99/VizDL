import { create } from 'zustand';
import type { PipelineConfig } from '../types/graph';

interface ConfigState extends PipelineConfig {
  // CSV metadata (populated after upload)
  availableColumns: string[];
  testAvailableColumns: string[];
  numRows: number | null;

  // Actions
  setField: <K extends keyof PipelineConfig>(key: K, value: PipelineConfig[K]) => void;
  setAvailableColumns: (cols: string[]) => void;
  setTestAvailableColumns: (cols: string[]) => void;
  setNumRows: (n: number | null) => void;
  getConfig: () => PipelineConfig;
  loadConfig: (config: Partial<PipelineConfig>) => void;
  reset: () => void;
}

const DEFAULT_CONFIG: PipelineConfig = {
  file_id: '',
  input_columns: '',
  target_columns: '',
  val_ratio: 0.2,
  batch_size: 32,
  shuffle: true,
  loss_fn: 'MSELoss',
  optimizer: 'Adam',
  lr: 0.01,
  epochs: 50,
  export_name: '',
  test_file_id: null,
  test_input_columns: null,
  test_target_columns: null,
  resume_from: null,
};

export const useConfigStore = create<ConfigState>((set, get) => ({
  ...DEFAULT_CONFIG,
  availableColumns: [],
  testAvailableColumns: [],
  numRows: null,

  setField: (key, value) => set({ [key]: value }),

  setAvailableColumns: (cols) => set({ availableColumns: cols }),
  setTestAvailableColumns: (cols) => set({ testAvailableColumns: cols }),
  setNumRows: (n) => set({ numRows: n }),

  getConfig: () => {
    const s = get();
    return {
      file_id: s.file_id,
      input_columns: s.input_columns,
      target_columns: s.target_columns,
      val_ratio: s.val_ratio,
      batch_size: s.batch_size,
      shuffle: s.shuffle,
      loss_fn: s.loss_fn,
      optimizer: s.optimizer,
      lr: s.lr,
      epochs: s.epochs,
      export_name: s.export_name,
      test_file_id: s.test_file_id,
      test_input_columns: s.test_input_columns,
      test_target_columns: s.test_target_columns,
      resume_from: s.resume_from,
    };
  },

  loadConfig: (config) => set(() => {
    // Strip runtime-only fields — file_ids are session-specific (user must upload/select data)
    const { file_id: _fid, test_file_id: _tfid, resume_from: _resume, ...rest } = config;
    return { ...rest };
  }),

  reset: () => set({ ...DEFAULT_CONFIG, availableColumns: [], testAvailableColumns: [], numRows: null }),
}));
