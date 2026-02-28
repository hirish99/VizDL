import { create } from 'zustand';
import {
  type Node,
  type Edge,
  type Connection,
  type NodeChange,
  type EdgeChange,
  applyNodeChanges,
  applyEdgeChanges,
  addEdge as rfAddEdge,
} from '@xyflow/react';
import type { NodeDefinition } from '../types/nodes';

export interface NodeData extends Record<string, unknown> {
  nodeType: string;
  definition: NodeDefinition;
  params: Record<string, unknown>;
  disabled: boolean;
}

interface GraphState {
  nodes: Node<NodeData>[];
  edges: Edge[];
  selectedNodeId: string | null;
  nodeDefinitions: Record<string, NodeDefinition>;

  setNodeDefinitions: (defs: Record<string, NodeDefinition>) => void;
  onNodesChange: (changes: NodeChange[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (connection: Connection) => void;
  addNode: (nodeType: string, position: { x: number; y: number }) => void;
  setSelectedNode: (id: string | null) => void;
  updateNodeParam: (nodeId: string, key: string, value: unknown) => void;
  toggleNodeDisabled: (nodeId: string) => void;
  clearGraph: () => void;

  // Serialization
  toGraphSchema: () => { nodes: any[]; edges: any[]; name: string; description: string };
  loadFromSchema: (schema: any) => void;
}

let nodeCounter = 0;

export const useGraphStore = create<GraphState>((set, get) => ({
  nodes: [],
  edges: [],
  selectedNodeId: null,
  nodeDefinitions: {},

  setNodeDefinitions: (defs) => set({ nodeDefinitions: defs }),

  onNodesChange: (changes) =>
    set((state) => ({ nodes: applyNodeChanges(changes, state.nodes) as Node<NodeData>[] })),

  onEdgesChange: (changes) =>
    set((state) => ({ edges: applyEdgeChanges(changes, state.edges) })),

  onConnect: (connection) =>
    set((state) => ({ edges: rfAddEdge(connection, state.edges) })),

  addNode: (nodeType, position) => {
    const def = get().nodeDefinitions[nodeType];
    if (!def) return;

    // Build default params from input specs
    const params: Record<string, unknown> = {};
    for (const [key, spec] of Object.entries(def.inputs)) {
      if (!spec.is_handle && spec.default !== null && spec.default !== undefined) {
        params[key] = spec.default;
      }
    }

    const id = `node_${++nodeCounter}_${Date.now()}`;
    const newNode: Node<NodeData> = {
      id,
      type: nodeType,
      position,
      data: {
        nodeType,
        definition: def,
        params,
        disabled: false,
      },
    };

    set((state) => ({ nodes: [...state.nodes, newNode] }));
  },

  setSelectedNode: (id) => set({ selectedNodeId: id }),

  updateNodeParam: (nodeId, key, value) =>
    set((state) => ({
      nodes: state.nodes.map((n) =>
        n.id === nodeId
          ? { ...n, data: { ...n.data, params: { ...n.data.params, [key]: value } } }
          : n,
      ),
    })),

  toggleNodeDisabled: (nodeId) =>
    set((state) => ({
      nodes: state.nodes.map((n) =>
        n.id === nodeId
          ? { ...n, data: { ...n.data, disabled: !n.data.disabled } }
          : n,
      ),
    })),

  clearGraph: () => set({ nodes: [], edges: [], selectedNodeId: null }),

  toGraphSchema: () => {
    const { nodes, edges } = get();
    return {
      nodes: nodes.map((n) => ({
        id: n.id,
        node_type: n.data.nodeType,
        params: n.data.params,
        disabled: n.data.disabled,
        position: n.position,
      })),
      edges: edges.map((e, i) => ({
        id: e.id,
        source_node: e.source,
        source_output: parseInt(e.sourceHandle?.replace('output_', '') ?? '0'),
        target_node: e.target,
        target_input: e.targetHandle?.replace('input_', '') ?? '',
        order: i,
      })),
      name: '',
      description: '',
    };
  },

  loadFromSchema: (schema) => {
    const defs = get().nodeDefinitions;
    const nodes: Node<NodeData>[] = schema.nodes.map((n: any) => {
      const def = defs[n.node_type];
      return {
        id: n.id,
        type: n.node_type,
        position: n.position || { x: 0, y: 0 },
        data: {
          nodeType: n.node_type,
          definition: def,
          params: n.params || {},
          disabled: n.disabled || false,
        },
      };
    });
    const edges: Edge[] = schema.edges.map((e: any) => ({
      id: e.id,
      source: e.source_node,
      sourceHandle: `output_${e.source_output}`,
      target: e.target_node,
      targetHandle: `input_${e.target_input}`,
    }));
    set({ nodes, edges, selectedNodeId: null });
  },
}));
