export interface InputSpec {
  dtype: string;
  default: unknown;
  required: boolean;
  min_val: number | null;
  max_val: number | null;
  choices: unknown[] | null;
  is_handle: boolean;
}

export interface OutputSpec {
  dtype: string;
  name: string;
}

export interface NodeDefinition {
  node_type: string;
  display_name: string;
  category: string;
  description: string;
  inputs: Record<string, InputSpec>;
  outputs: OutputSpec[];
}
