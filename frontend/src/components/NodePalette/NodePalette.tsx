import { useCallback, useEffect, useMemo, useState } from 'react';
import type { NodeDefinition } from '../../types/nodes';
import { useGraphStore } from '../../store/graphStore';
import { useConfigStore } from '../../store/configStore';
import { listModels, type SavedModel } from '../../api/client';

interface Props {
  definitions: Record<string, NodeDefinition>;
}

let clickOffset = 0;

function PaletteItem({ type, def, onDragStart, addNode }: {
  type: string;
  def: NodeDefinition;
  onDragStart: (e: React.DragEvent, t: string) => void;
  addNode: (type: string, pos: { x: number; y: number }) => void;
}) {
  return (
    <div
      draggable
      onDragStart={(e) => onDragStart(e, type)}
      onClick={() => {
        addNode(type, { x: 300 + clickOffset, y: 200 + clickOffset });
        clickOffset = (clickOffset + 40) % 200;
      }}
      style={{
        padding: '6px 12px 6px 16px',
        color: '#c0c0d0',
        fontSize: 12,
        cursor: 'grab',
        borderLeft: '3px solid transparent',
        transition: 'all 0.15s',
      }}
      onMouseEnter={(e) => {
        (e.target as HTMLElement).style.background = '#1e1e2e';
        (e.target as HTMLElement).style.borderLeftColor = '#6366f1';
      }}
      onMouseLeave={(e) => {
        (e.target as HTMLElement).style.background = 'transparent';
        (e.target as HTMLElement).style.borderLeftColor = 'transparent';
      }}
      title={def.description}
    >
      {def.display_name}
    </div>
  );
}

function formatModelLabel(m: SavedModel): string {
  // Extract custom name: if name != architecture_timestamp, user gave a custom name
  const hasCustomName = !m.name.startsWith(m.architecture);
  // Parse timestamp like "20260303_204043" → "Mar 3 20:40"
  let timeStr = '';
  const ts = m.timestamp;
  if (ts && ts.length >= 15) {
    const mon = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    const month = mon[parseInt(ts.slice(4, 6), 10) - 1] || '';
    const day = parseInt(ts.slice(6, 8), 10);
    const hour = ts.slice(9, 11);
    const min = ts.slice(11, 13);
    timeStr = `${month} ${day} ${hour}:${min}`;
  }

  const name = hasCustomName
    ? m.name.replace(/_\d{8}_\d{6}$/, '')  // strip timestamp suffix
    : m.architecture;
  const short = name.length > 20 ? name.slice(0, 20) + '..' : name;
  const loss = m.final_val_loss != null
    ? m.final_val_loss.toExponential(1)
    : m.final_train_loss != null
      ? m.final_train_loss.toExponential(1)
      : '';

  const parts = [short];
  if (m.total_epochs) parts.push(`${m.total_epochs}ep`);
  if (loss) parts.push(loss);
  if (timeStr) parts.push(timeStr);
  return parts.join(' · ');
}

function ResumeSection() {
  const setField = useConfigStore((s) => s.setField);
  const loadConfig = useConfigStore((s) => s.loadConfig);
  const resumeFrom = useConfigStore((s) => s.resume_from);
  const loadFromSchema = useGraphStore((s) => s.loadFromSchema);

  const [models, setModels] = useState<SavedModel[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchModels = useCallback(async () => {
    setLoading(true);
    try {
      setModels(await listModels());
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch on mount
  useEffect(() => { fetchModels(); }, [fetchModels]);

  const selected = models.find((m) => m.path === resumeFrom);

  const handleSelect = (path: string) => {
    if (!path) {
      setField('resume_from', null);
      return;
    }
    const model = models.find((m) => m.path === path);
    if (!model) return;

    setField('resume_from', path);

    // Auto-load graph
    if (model.graph) {
      loadFromSchema({ ...model.graph, name: '', description: '' });
    }
    // Auto-load config (preserving file_id)
    if (model.config) {
      loadConfig(model.config);
    }
  };

  return (
    <div style={{ borderTop: '1px solid #2a2a3e', padding: '8px 12px' }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 4, marginBottom: 6,
      }}>
        <div style={{
          color: '#22c55e', fontSize: 10, fontWeight: 600,
          textTransform: 'uppercase', letterSpacing: 1, flex: 1,
        }}>
          Continue Training
        </div>
        <button
          onClick={fetchModels}
          disabled={loading}
          style={{
            background: 'transparent', border: '1px solid #2a2a3e',
            borderRadius: 3, color: '#808090', fontSize: 9,
            padding: '1px 6px', cursor: loading ? 'wait' : 'pointer',
          }}
        >
          {loading ? '...' : 'Refresh'}
        </button>
      </div>
      <select
        value={resumeFrom || ''}
        onChange={(e) => handleSelect(e.target.value)}
        style={{
          width: '100%', background: '#1e1e2e', border: '1px solid #2a2a3e',
          borderRadius: 4, color: '#c0c0d0', padding: '4px 6px',
          fontSize: 10, outline: 'none', marginBottom: 4,
        }}
      >
        <option value="">None (fresh)</option>
        {models.map((m) => {
          const label = formatModelLabel(m);
          return (
            <option key={m.path} value={m.path}>{label}</option>
          );
        })}
      </select>

      {selected && (
        <div style={{ fontSize: 9, color: '#808090', lineHeight: 1.5 }}>
          <div style={{ color: '#a0a0b0' }}>{selected.architecture}</div>
          {selected.final_val_loss != null && (
            <div>Val: {selected.final_val_loss.toExponential(2)}</div>
          )}
          {selected.parameter_count != null && (
            <div>{selected.parameter_count.toLocaleString()} params</div>
          )}
          {!selected.graph && (
            <div style={{ color: '#f59e0b', marginTop: 2 }}>No graph saved (old export)</div>
          )}
        </div>
      )}
    </div>
  );
}

export function NodePalette({ definitions }: Props) {
  const addNode = useGraphStore((s) => s.addNode);
  const { layerNodes, structuralNodes, modelNodes } = useMemo(() => {
    const layers: { type: string; def: NodeDefinition }[] = [];
    const structural: { type: string; def: NodeDefinition }[] = [];
    const models: { type: string; def: NodeDefinition }[] = [];
    for (const [type, def] of Object.entries(definitions)) {
      if (def.category === 'Layers') {
        layers.push({ type, def });
      } else if (def.category === 'Structural') {
        structural.push({ type, def });
      } else if (def.category === 'Model') {
        models.push({ type, def });
      }
    }
    return { layerNodes: layers, structuralNodes: structural, modelNodes: models };
  }, [definitions]);

  const onDragStart = (event: React.DragEvent, nodeType: string) => {
    event.dataTransfer.setData('application/visdl-node', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div style={{
      width: 180,
      background: '#12121a',
      borderRight: '1px solid #2a2a3e',
      overflow: 'auto',
      fontFamily: 'system-ui, sans-serif',
      display: 'flex',
      flexDirection: 'column',
    }}>
      <div style={{ flex: 1, overflow: 'auto', padding: '12px 0' }}>
        <div style={{
          padding: '0 12px 8px',
          color: '#6366f1',
          fontSize: 10,
          fontWeight: 600,
          textTransform: 'uppercase',
          letterSpacing: 1,
        }}>
          Layers
        </div>
        {layerNodes.map(({ type, def }) => (
          <PaletteItem key={type} type={type} def={def} onDragStart={onDragStart} addNode={addNode} />
        ))}

        {structuralNodes.length > 0 && (
          <>
            <div style={{
              padding: '12px 12px 8px',
              color: '#6366f1',
              fontSize: 10,
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: 1,
            }}>
              Structural
            </div>
            {structuralNodes.map(({ type, def }) => (
              <PaletteItem key={type} type={type} def={def} onDragStart={onDragStart} addNode={addNode} />
            ))}
          </>
        )}

        {modelNodes.length > 0 && (
          <>
            <div style={{
              padding: '12px 12px 8px',
              color: '#6366f1',
              fontSize: 10,
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: 1,
            }}>
              Model
            </div>
            {modelNodes.map(({ type, def }) => (
              <PaletteItem key={type} type={type} def={def} onDragStart={onDragStart} addNode={addNode} />
            ))}
          </>
        )}
      </div>

      <ResumeSection />
    </div>
  );
}
