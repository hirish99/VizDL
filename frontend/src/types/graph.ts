export interface GraphNode {
  id: string;
  node_type: string;
  params: Record<string, unknown>;
  disabled: boolean;
  position: { x: number; y: number };
}

export interface GraphEdge {
  id: string;
  source_node: string;
  source_output: number;
  target_node: string;
  target_input: string;
  order: number;
}

export interface GraphSchema {
  nodes: GraphNode[];
  edges: GraphEdge[];
  name: string;
  description: string;
}

export interface PipelineConfig {
  file_id: string;
  input_columns: string;
  target_columns: string;
  val_ratio: number;
  batch_size: number;
  shuffle: boolean;
  loss_fn: string;
  optimizer: string;
  lr: number;
  epochs: number;
  export_name: string;
  test_file_id: string | null;
  test_input_columns: string | null;
  test_target_columns: string | null;
}

export interface ExecuteResponse {
  execution_id: string;
  status: string;
  results: Record<string, unknown>;
  errors: string[];
}

export interface TrainingProgress {
  type: string;
  epoch: number;
  total_epochs: number;
  train_loss: number;
  val_loss: number | null;
  samples_trained?: number;
  total_samples?: number;
  throughput?: number;
}
