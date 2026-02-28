import { useCallback, useRef, useMemo } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  type ReactFlowInstance,
  type Node,
  type Edge,
  type Connection,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useGraphStore, type NodeData } from '../../store/graphStore';
import { buildNodeTypes } from '../NodeTypes';

// Type compatibility for edge validation
const isTypeCompatible = (srcDtype: string, tgtDtype: string): boolean => {
  if (srcDtype === tgtDtype) return true;
  if (srcDtype === 'ANY' || tgtDtype === 'ANY') return true;
  if (srcDtype === 'LAYER_SPEC' && tgtDtype === 'LAYER_SPECS') return true;
  return false;
};

export function Canvas() {
  const nodes = useGraphStore((s) => s.nodes);
  const edges = useGraphStore((s) => s.edges);
  const onNodesChange = useGraphStore((s) => s.onNodesChange);
  const onEdgesChange = useGraphStore((s) => s.onEdgesChange);
  const onConnect = useGraphStore((s) => s.onConnect);
  const addNode = useGraphStore((s) => s.addNode);
  const definitions = useGraphStore((s) => s.nodeDefinitions);

  const rfInstance = useRef<ReactFlowInstance<Node<NodeData>, Edge> | null>(null);

  const nodeTypes = useMemo(
    () => buildNodeTypes(Object.keys(definitions)),
    [definitions],
  );

  const isValidConnection = useCallback((connection: Connection | Edge) => {
    const { source, target, sourceHandle, targetHandle } = connection;

    if (source === target) return false;
    if (!source || !target || !sourceHandle || !targetHandle) return false;

    const targetInputName = (targetHandle as string).replace('input_', '');
    const targetNode = nodes.find((n) => n.id === target);
    const sourceNode = nodes.find((n) => n.id === source);
    if (!targetNode || !sourceNode) return false;

    const targetDef = definitions[targetNode.data.nodeType];
    const sourceDef = definitions[sourceNode.data.nodeType];
    if (!targetDef || !sourceDef) return false;

    const targetInput = targetDef.inputs[targetInputName];
    if (!targetInput) return false;

    // Only allow one connection per input (layer chain is linear)
    const alreadyConnected = edges.some(
      (e) => e.target === target && e.targetHandle === targetHandle,
    );
    if (alreadyConnected) return false;

    // Only LAYER_SPECS connections on canvas
    const outputIndex = parseInt((sourceHandle as string).replace('output_', ''));
    const sourceOutput = sourceDef.outputs[outputIndex];
    if (!sourceOutput) return false;

    if (!isTypeCompatible(sourceOutput.dtype, targetInput.dtype)) return false;

    return true;
  }, [nodes, edges, definitions]);

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      const nodeType = event.dataTransfer.getData('application/visdl-node');
      if (!nodeType || !rfInstance.current) return;

      const position = rfInstance.current.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });
      addNode(nodeType, position);
    },
    [addNode],
  );

  return (
    <div style={{ flex: 1, height: '100%' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        isValidConnection={isValidConnection}
        onInit={(instance) => { rfInstance.current = instance; }}
        onDragOver={onDragOver}
        onDrop={onDrop}
        nodeTypes={nodeTypes}
        fitView
        deleteKeyCode={['Backspace', 'Delete']}
        colorMode="dark"
        defaultEdgeOptions={{
          type: 'smoothstep',
          style: { stroke: '#4a4a6a', strokeWidth: 2 },
        }}
      >
        <Background color="#2a2a3e" gap={20} />
        <Controls />
        <MiniMap
          style={{ background: '#12121a' }}
          nodeColor="#4a4a6a"
          maskColor="rgba(0,0,0,0.7)"
        />
      </ReactFlow>
    </div>
  );
}
