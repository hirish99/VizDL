import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { NodeData } from '../../store/graphStore';
import { useGraphStore } from '../../store/graphStore';

function BaseNodeComponent({ id, data, selected }: NodeProps & { data: NodeData }) {
  const toggleDisabled = useGraphStore((s) => s.toggleNodeDisabled);
  const setSelected = useGraphStore((s) => s.setSelectedNode);
  const { definition, disabled, params } = data;

  const handleInputs = Object.entries(definition.inputs).filter(([, s]) => s.is_handle);
  const outputs = definition.outputs;

  const categoryColors: Record<string, string> = {
    Layers: '#6366f1',
    Data: '#22c55e',
    Loss: '#ef4444',
    Optimizer: '#f59e0b',
    Model: '#8b5cf6',
    Training: '#3b82f6',
    Metrics: '#06b6d4',
    Structural: '#ec4899',
  };
  const color = categoryColors[definition.category] || '#6b7280';

  return (
    <div
      onClick={() => setSelected(id)}
      style={{
        background: disabled ? '#1a1a2e' : '#1e1e2e',
        border: `2px solid ${selected ? '#60a5fa' : disabled ? '#4a4a5a' : color}`,
        borderRadius: 8,
        minWidth: 180,
        opacity: disabled ? 0.5 : 1,
        fontFamily: 'system-ui, sans-serif',
      }}
    >
      {/* Header */}
      <div
        style={{
          background: color,
          padding: '6px 12px',
          borderRadius: '6px 6px 0 0',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <span style={{ color: '#fff', fontSize: 12, fontWeight: 600 }}>
          {definition.display_name}
        </span>
        <button
          onClick={(e) => { e.stopPropagation(); toggleDisabled(id); }}
          style={{
            background: 'none',
            border: 'none',
            color: '#fff',
            cursor: 'pointer',
            fontSize: 10,
            padding: '2px 4px',
            borderRadius: 3,
            opacity: 0.7,
          }}
          title={disabled ? 'Enable' : 'Disable (ablate)'}
        >
          {disabled ? 'OFF' : 'ON'}
        </button>
      </div>

      {/* Handles */}
      <div style={{ padding: '8px 12px', position: 'relative', minHeight: 30 }}>
        {/* Input handles */}
        {handleInputs.map(([name, spec]) => (
          <div key={name} style={{ position: 'relative', marginBottom: 4 }}>
            <Handle
              type="target"
              position={Position.Left}
              id={`input_${name}`}
              style={{
                background: '#60a5fa',
                width: 10,
                height: 10,
                top: '50%',
              }}
            />
            <span style={{ color: '#a0a0b0', fontSize: 10, marginLeft: 8 }}>
              {name} <span style={{ color: '#666', fontSize: 9 }}>({spec.dtype})</span>
            </span>
          </div>
        ))}

        {/* Property previews */}
        {Object.entries(definition.inputs)
          .filter(([, s]) => !s.is_handle)
          .slice(0, 3)
          .map(([name]) => (
            <div key={name} style={{ color: '#808090', fontSize: 10, marginBottom: 2 }}>
              {name}: {String(params[name] ?? definition.inputs[name].default ?? '—')}
            </div>
          ))}

        {/* Output handles */}
        {outputs.map((output, i) => (
          <div key={output.name} style={{ position: 'relative', textAlign: 'right', marginBottom: 4 }}>
            <span style={{ color: '#a0a0b0', fontSize: 10, marginRight: 8 }}>
              {output.name} <span style={{ color: '#666', fontSize: 9 }}>({output.dtype})</span>
            </span>
            <Handle
              type="source"
              position={Position.Right}
              id={`output_${i}`}
              style={{
                background: '#f472b6',
                width: 10,
                height: 10,
                top: '50%',
              }}
            />
          </div>
        ))}
      </div>
    </div>
  );
}

export const BaseNode = memo(BaseNodeComponent);
