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

export interface VramEstimate {
  param_count: number;
  effective_batch_size: number;
  params_mb: number;
  gradients_mb: number;
  optimizer_mb: number;
  activations_mb: number;
  batch_data_mb: number;
  total_mb: number;
  available_mb: number | null;
  fits: boolean | null;
}

export async function estimateVram(
  graph: GraphSchema,
  inputDim: number,
  batchSize: number,
  optimizer: string,
  numTrainSamples?: number | null,
): Promise<VramEstimate> {
  const { data } = await api.post('/estimate-vram', {
    graph,
    input_dim: inputDim,
    batch_size: batchSize,
    optimizer,
    num_train_samples: numTrainSamples ?? undefined,
  });
  return data;
}
