import axios from 'axios';
import type {
  NodeDefinition, GraphSchema, PipelineConfig, ExecuteResponse,
  UploadResponse, SavedGraphSummary,
} from './types';

const api = axios.create({ baseURL: '/api' });

export async function fetchNodes(): Promise<Record<string, NodeDefinition>> {
  const { data } = await api.get('/nodes');
  return data;
}

export async function executeGraph(
  graph: GraphSchema,
  config: PipelineConfig,
  sessionId?: string,
): Promise<ExecuteResponse> {
  const { data } = await api.post('/execute', {
    graph,
    config,
    session_id: sessionId,
  });
  return data;
}

export async function getResults(executionId: string) {
  const { data } = await api.get(`/results/${executionId}`);
  return data;
}

export async function uploadCSV(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append('file', file);
  const { data } = await api.post('/upload/csv', form);
  return data;
}

export interface ServerFile {
  file_id: string;
  filename: string;
  size_mb: number;
}

export async function listServerFiles(): Promise<ServerFile[]> {
  const { data } = await api.get('/upload/server-files');
  return data;
}

export async function useServerFile(fileId: string): Promise<UploadResponse> {
  const { data } = await api.post(`/upload/server-files/${fileId}`);
  return data;
}

export async function saveGraph(
  graph: GraphSchema,
  config: PipelineConfig,
  id: string,
  name: string,
  description = '',
) {
  const { data } = await api.post('/graphs', {
    id, name, description, graph, config,
  });
  return data;
}

export async function listGraphs(): Promise<Record<string, SavedGraphSummary>> {
  const { data } = await api.get('/graphs');
  return data;
}

export async function loadGraph(graphId: string): Promise<{
  id: string; name: string; description: string; graph: GraphSchema; config?: PipelineConfig;
}> {
  const { data } = await api.get(`/graphs/${graphId}`);
  return data;
}

export async function deleteGraph(graphId: string) {
  await api.delete(`/graphs/${graphId}`);
}

export async function pauseTraining(executionId: string) {
  await api.post(`/execute/${executionId}/pause`);
}

export async function resumeTraining(executionId: string) {
  await api.post(`/execute/${executionId}/resume`);
}

export async function stopTraining(executionId: string) {
  await api.post(`/execute/${executionId}/stop`);
}

export interface AutoBatchSizeResult {
  max_batch_size: number;
  steps_tried: number;
  search_log: { batch_size: number; status: 'ok' | 'oom' }[];
}

export interface SavedModel {
  path: string;
  name: string;
  architecture: string;
  parameter_count: number | null;
  final_train_loss: number | null;
  final_val_loss: number | null;
  total_epochs: number;
  timestamp: string;
  graph: any | null;
  config: Record<string, any> | null;
}

export async function listModels(): Promise<SavedModel[]> {
  const { data } = await api.get('/models');
  return data;
}

export async function findMaxBatchSize(
  graph: GraphSchema,
  optimizer: string,
  fileId: string,
  inputColumns: string,
  numTrainSamples?: number | null,
): Promise<AutoBatchSizeResult> {
  const { data } = await api.post('/auto-batch-size', {
    graph,
    optimizer,
    file_id: fileId,
    input_columns: inputColumns,
    num_train_samples: numTrainSamples ?? undefined,
  });
  return data;
}
