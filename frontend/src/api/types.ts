export type { NodeDefinition, InputSpec, OutputSpec } from '../types/nodes';
export type {
  GraphNode, GraphEdge, GraphSchema, PipelineConfig,
  ExecuteResponse, TrainingProgress,
} from '../types/graph';

export interface UploadResponse {
  file_id: string;
  filename: string;
  columns: string[];
  rows: number;
}

export interface SavedGraphSummary {
  id: string;
  name: string;
  description: string;
}
