import { useCallback, useEffect, useRef, useState } from 'react';
import { useConfigStore } from '../../store/configStore';
import { useGraphStore, type NodeData } from '../../store/graphStore';
import { uploadCSV, findMaxBatchSize, listServerFiles, useServerFile, type ServerFile } from '../../api/client';
import type { Node } from '@xyflow/react';

export function ConfigPanel() {
  const selectedId = useGraphStore((s) => s.selectedNodeId);
  const nodes = useGraphStore((s) => s.nodes);
  const updateParam = useGraphStore((s) => s.updateNodeParam);
  const toggleDisabled = useGraphStore((s) => s.toggleNodeDisabled);

  const selectedNode: Node<NodeData> | undefined = nodes.find((n) => n.id === selectedId);

  return (
    <div style={panelStyle}>
      <DataSection />
      <TrainingSection />
      <TestDataSection />
      <ExportSection />
      {selectedNode && (
        <SelectedNodeSection
          node={selectedNode}
          updateParam={updateParam}
          toggleDisabled={toggleDisabled}
        />
      )}
    </div>
  );
}

// --- Data Section ---
function DataSection() {
  const config = useConfigStore();
  const toSchema = useGraphStore((s) => s.toGraphSchema);
  const optimizer = useConfigStore((s) => s.optimizer);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const [filename, setFilename] = useState('');
  const [serverFiles, setServerFiles] = useState<ServerFile[]>([]);
  const [autoSearching, setAutoSearching] = useState(false);
  const [autoError, setAutoError] = useState('');

  useEffect(() => {
    listServerFiles().then(setServerFiles).catch(() => {});
  }, []);

  const selectedInputSpecs = config.input_columns.split(',').filter(Boolean);
  const selectedTargetSpecs = config.target_columns.split(',').filter(Boolean);

  // Expand glob patterns so toggle buttons reflect what the backend will match
  const expandSpecs = (specs: string[], available: string[]) => {
    const result = new Set<string>();
    for (const spec of specs) {
      if (spec.includes('*') || spec.includes('?') || spec.includes('[')) {
        const re = new RegExp('^' + spec.replace(/[.+^${}()|\\]/g, '\\$&').replace(/\*/g, '.*').replace(/\?/g, '.') + '$');
        for (const col of available) {
          if (re.test(col)) result.add(col);
        }
      } else {
        result.add(spec);
      }
    }
    return result;
  };
  const selectedInputs = expandSpecs(selectedInputSpecs, config.availableColumns);
  const selectedTargets = expandSpecs(selectedTargetSpecs, config.availableColumns);

  const toggleColumn = (col: string, field: 'input_columns' | 'target_columns', currentSpecs: string[]) => {
    // When toggling, replace any glob patterns with the expanded exact column list
    const available = config.availableColumns;
    const expanded = expandSpecs(currentSpecs, available);
    let next: string[];
    if (expanded.has(col)) {
      expanded.delete(col);
      next = available.filter((c) => expanded.has(c));
    } else {
      expanded.add(col);
      next = available.filter((c) => expanded.has(c));
    }
    config.setField(field, next.join(','));
  };

  return (
    <div style={sectionStyle}>
      <div style={sectionHeaderStyle}>Data</div>
      <div style={sectionBodyStyle}>
        <label style={labelStyle}>Data File</label>
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.pt"
          onChange={async (e: React.ChangeEvent<HTMLInputElement>) => {
            const file = e.target.files?.[0];
            if (!file) return;
            setUploading(true);
            setUploadError('');
            try {
              const res = await uploadCSV(file);
              config.setField('file_id', res.file_id);
              if (!config.input_columns) config.setField('input_columns', '');
              if (!config.target_columns) config.setField('target_columns', '');
              config.setAvailableColumns(res.columns);
              config.setNumRows(res.rows);
              setFilename(res.filename);
            } catch (err: any) {
              setUploadError(err?.response?.data?.detail || err?.message || 'Upload failed');
            } finally {
              setUploading(false);
            }
          }}
          style={{ display: 'none' }}
        />
        <select
          value=""
          onChange={(e) => {
            if (e.target.value === '__upload__') {
              fileInputRef.current?.click();
            }
          }}
          style={{ ...inputStyle, fontSize: 10, marginBottom: 4 }}
        >
          <option value="">Local file...</option>
          <option value="__upload__">Upload from computer</option>
        </select>
        <select
          value={config.file_id || ''}
          onChange={async (e) => {
            const fid = e.target.value;
            if (!fid) return;
            setUploading(true);
            setUploadError('');
            try {
              const res = await useServerFile(fid);
              config.setField('file_id', res.file_id);
              if (!config.input_columns) config.setField('input_columns', '');
              if (!config.target_columns) config.setField('target_columns', '');
              config.setAvailableColumns(res.columns);
              config.setNumRows(res.rows);
              setFilename(res.filename);
            } catch (err: any) {
              setUploadError(err?.response?.data?.detail || err?.message || 'Failed');
            } finally {
              setUploading(false);
            }
          }}
          style={{ ...inputStyle, fontSize: 10, marginBottom: 4 }}
        >
          <option value="">Server file...</option>
          {serverFiles.map((f) => (
            <option key={f.file_id} value={f.file_id}>
              {f.filename} ({(f.size_mb / 1024).toFixed(1)} GB)
            </option>
          ))}
        </select>
        {uploading && <div style={{ color: '#22c55e', fontSize: 10 }}>Loading...</div>}
        {uploadError && <div style={{ color: '#ef4444', fontSize: 10, marginBottom: 4 }}>{uploadError}</div>}
        {filename && <div style={{ color: '#808090', fontSize: 10, marginBottom: 4 }}>{filename}</div>}

        {config.availableColumns.length > 0 && (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
              <label style={{ ...labelStyle, marginBottom: 0, marginTop: 0, flex: 1 }}>Input Columns</label>
              <button onClick={() => config.setField('input_columns', config.availableColumns.join(','))} style={bulkBtnStyle}>All</button>
              <button onClick={() => config.setField('input_columns', '')} style={bulkBtnStyle}>None</button>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, marginBottom: 6 }}>
              {config.availableColumns.map((col) => (
                <button
                  key={`in_${col}`}
                  onClick={() => toggleColumn(col, 'input_columns', selectedInputSpecs)}
                  style={{
                    background: selectedInputs.has(col) ? '#22c55e' : '#1a1a2e',
                    color: selectedInputs.has(col) ? '#fff' : '#808090',
                    border: `1px solid ${selectedInputs.has(col) ? '#22c55e' : '#2a2a3e'}`,
                    borderRadius: 3, padding: '2px 6px', fontSize: 9, cursor: 'pointer',
                  }}
                >
                  {col}
                </button>
              ))}
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
              <label style={{ ...labelStyle, marginBottom: 0, marginTop: 0, flex: 1 }}>Target Columns</label>
              <button onClick={() => config.setField('target_columns', config.availableColumns.join(','))} style={bulkBtnStyle}>All</button>
              <button onClick={() => config.setField('target_columns', '')} style={bulkBtnStyle}>None</button>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, marginBottom: 6 }}>
              {config.availableColumns.map((col) => (
                <button
                  key={`tgt_${col}`}
                  onClick={() => toggleColumn(col, 'target_columns', selectedTargetSpecs)}
                  style={{
                    background: selectedTargets.has(col) ? '#ef4444' : '#1a1a2e',
                    color: selectedTargets.has(col) ? '#fff' : '#808090',
                    border: `1px solid ${selectedTargets.has(col) ? '#ef4444' : '#2a2a3e'}`,
                    borderRadius: 3, padding: '2px 6px', fontSize: 9, cursor: 'pointer',
                  }}
                >
                  {col}
                </button>
              ))}
            </div>
          </>
        )}

        <div style={{ display: 'flex', gap: 8, marginBottom: 4 }}>
          <div style={{ flex: 1 }}>
            <label style={labelStyle}>Val Split</label>
            <NumericInput
              value={config.val_ratio}
              onChange={(v) => config.setField('val_ratio', v)}
              fallback={0.2}
              min={0.01} max={0.99} step={0.05}
              style={inputStyle}
            />
          </div>
          <div style={{ flex: 1 }}>
            <label style={labelStyle}>Batch Size</label>
            <NumericInput
              value={config.batch_size}
              onChange={(v) => config.setField('batch_size', v)}
              fallback={32}
              min={1} step={1} integer
              style={inputStyle}
            />
            <button
              onClick={async () => {
                setAutoSearching(true);
                setAutoError('');
                try {
                  const schema = toSchema();
                  if (schema.nodes.length === 0) { setAutoError('No nodes'); return; }
                  if (!config.file_id) { setAutoError('No data file loaded'); return; }
                  if (!config.input_columns) { setAutoError('No input columns selected'); return; }
                  const trainSamples = config.numRows != null
                    ? Math.floor(config.numRows * (1 - config.val_ratio))
                    : undefined;
                  const result = await findMaxBatchSize(schema, optimizer, config.file_id, config.input_columns, trainSamples);
                  config.setField('batch_size', result.max_batch_size);
                } catch (err: any) {
                  setAutoError(err?.response?.data?.detail || err.message || 'Search failed');
                } finally {
                  setAutoSearching(false);
                }
              }}
              disabled={autoSearching}
              style={{
                width: '100%',
                padding: '2px 6px',
                background: '#1a1a2e',
                border: '1px solid #f59e0b',
                borderRadius: 3,
                color: '#f59e0b',
                fontSize: 9,
                cursor: autoSearching ? 'wait' : 'pointer',
                marginTop: 2,
              }}
            >
              {autoSearching ? 'Searching GPU...' : 'Auto (GPU)'}
            </button>
            {autoError && (
              <div style={{ color: '#ef4444', fontSize: 9, marginTop: 2 }}>{autoError}</div>
            )}
          </div>
        </div>

        <label style={{ ...labelStyle, display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={config.shuffle}
            onChange={(e) => config.setField('shuffle', e.target.checked)}
          />
          Shuffle
        </label>

      </div>
    </div>
  );
}

// --- Training Section ---
function TrainingSection() {
  const config = useConfigStore();

  return (
    <div style={sectionStyle}>
      <div style={sectionHeaderStyle}>Training</div>
      <div style={sectionBodyStyle}>
        <label style={labelStyle}>Loss Function</label>
        <select
          value={config.loss_fn}
          onChange={(e) => config.setField('loss_fn', e.target.value)}
          style={inputStyle}
        >
          <option value="MSELoss">MSE Loss</option>
          <option value="RelativeMSELoss">Relative MSE Loss</option>
          <option value="CrossEntropyLoss">Cross Entropy</option>
          <option value="L1Loss">L1 Loss (MAE)</option>
        </select>

        <label style={labelStyle}>Optimizer</label>
        <select
          value={config.optimizer}
          onChange={(e) => config.setField('optimizer', e.target.value)}
          style={inputStyle}
        >
          <option value="Adam">Adam</option>
          <option value="SGD">SGD</option>
          <option value="AdamW">AdamW</option>
        </select>

        <label style={labelStyle}>LR Scheduler</label>
        <select
          value={config.scheduler}
          onChange={(e) => config.setField('scheduler', e.target.value)}
          style={inputStyle}
        >
          <option value="None">None</option>
          <option value="CosineAnnealing">Cosine Annealing</option>
          <option value="ReduceOnPlateau">Reduce on Plateau</option>
        </select>

        <div style={{ display: 'flex', gap: 8, marginBottom: 4 }}>
          <div style={{ flex: 1 }}>
            <label style={labelStyle}>Learning Rate</label>
            <NumericInput
              value={config.lr}
              onChange={(v) => config.setField('lr', v)}
              fallback={0.001}
              min={1e-8} max={10} step={0.001}
              style={inputStyle}
            />
          </div>
          <div style={{ flex: 1 }}>
            <label style={labelStyle}>Epochs</label>
            <NumericInput
              value={config.epochs}
              onChange={(v) => config.setField('epochs', v)}
              fallback={10}
              min={1} max={10000} step={1} integer
              style={inputStyle}
            />
          </div>
        </div>
      </div>
    </div>
  );
}


// --- Test Data Section (collapsible) ---
function TestDataSection() {
  const config = useConfigStore();
  const [open, setOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const [filename, setFilename] = useState('');

  const handleUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadError('');
    try {
      const res = await uploadCSV(file);
      config.setField('test_file_id', res.file_id);
      if (!config.test_input_columns) config.setField('test_input_columns', '');
      if (!config.test_target_columns) config.setField('test_target_columns', '');
      config.setTestAvailableColumns(res.columns);
      setFilename(res.filename);
    } catch (err: any) {
      const detail = err?.response?.data?.detail || err?.message || 'Upload failed';
      setUploadError(detail);
    } finally {
      setUploading(false);
    }
  }, [config]);

  const selectedInputSpecs = (config.test_input_columns || '').split(',').filter(Boolean);
  const selectedTargetSpecs = (config.test_target_columns || '').split(',').filter(Boolean);

  const expandSpecs = (specs: string[], available: string[]) => {
    const result = new Set<string>();
    for (const spec of specs) {
      if (spec.includes('*') || spec.includes('?') || spec.includes('[')) {
        const re = new RegExp('^' + spec.replace(/[.+^${}()|\\]/g, '\\$&').replace(/\*/g, '.*').replace(/\?/g, '.') + '$');
        for (const col of available) {
          if (re.test(col)) result.add(col);
        }
      } else {
        result.add(spec);
      }
    }
    return result;
  };
  const selectedInputs = expandSpecs(selectedInputSpecs, config.testAvailableColumns);
  const selectedTargets = expandSpecs(selectedTargetSpecs, config.testAvailableColumns);

  const toggleColumn = (col: string, field: 'test_input_columns' | 'test_target_columns', currentSpecs: string[]) => {
    const available = config.testAvailableColumns;
    const expanded = expandSpecs(currentSpecs, available);
    let next: string[];
    if (expanded.has(col)) {
      expanded.delete(col);
      next = available.filter((c) => expanded.has(c));
    } else {
      expanded.add(col);
      next = available.filter((c) => expanded.has(c));
    }
    config.setField(field, next.join(','));
  };

  return (
    <div style={sectionStyle}>
      <div
        style={{ ...sectionHeaderStyle, cursor: 'pointer', userSelect: 'none' }}
        onClick={() => setOpen(!open)}
      >
        Test Data {open ? '\u25B4' : '\u25BE'}
        <span style={{ fontSize: 9, color: '#555', marginLeft: 4 }}>(optional)</span>
      </div>
      {open && (
        <div style={sectionBodyStyle}>
          <label style={labelStyle}>Test Data</label>
          <input
            type="file"
            accept=".csv,.pt"
            onChange={handleUpload}
            style={{ fontSize: 10, color: '#ccc', width: '100%', marginBottom: 4 }}
          />
          {uploading && <div style={{ color: '#22c55e', fontSize: 10 }}>Uploading...</div>}
          {uploadError && <div style={{ color: '#ef4444', fontSize: 10, marginBottom: 4 }}>{uploadError}</div>}
          {filename && <div style={{ color: '#808090', fontSize: 10, marginBottom: 4 }}>{filename}</div>}

          {config.testAvailableColumns.length > 0 && (
            <>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
                <label style={{ ...labelStyle, marginBottom: 0, marginTop: 0, flex: 1 }}>Input Columns</label>
                <button onClick={() => config.setField('test_input_columns', config.testAvailableColumns.join(','))} style={bulkBtnStyle}>All</button>
                <button onClick={() => config.setField('test_input_columns', '')} style={bulkBtnStyle}>None</button>
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, marginBottom: 6 }}>
                {config.testAvailableColumns.map((col) => (
                  <button
                    key={`tin_${col}`}
                    onClick={() => toggleColumn(col, 'test_input_columns', selectedInputSpecs)}
                    style={{
                      background: selectedInputs.has(col) ? '#22c55e' : '#1a1a2e',
                      color: selectedInputs.has(col) ? '#fff' : '#808090',
                      border: `1px solid ${selectedInputs.has(col) ? '#22c55e' : '#2a2a3e'}`,
                      borderRadius: 3, padding: '2px 6px', fontSize: 9, cursor: 'pointer',
                    }}
                  >
                    {col}
                  </button>
                ))}
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
                <label style={{ ...labelStyle, marginBottom: 0, marginTop: 0, flex: 1 }}>Target Columns</label>
                <button onClick={() => config.setField('test_target_columns', config.testAvailableColumns.join(','))} style={bulkBtnStyle}>All</button>
                <button onClick={() => config.setField('test_target_columns', '')} style={bulkBtnStyle}>None</button>
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, marginBottom: 6 }}>
                {config.testAvailableColumns.map((col) => (
                  <button
                    key={`ttgt_${col}`}
                    onClick={() => toggleColumn(col, 'test_target_columns', selectedTargetSpecs)}
                    style={{
                      background: selectedTargets.has(col) ? '#ef4444' : '#1a1a2e',
                      color: selectedTargets.has(col) ? '#fff' : '#808090',
                      border: `1px solid ${selectedTargets.has(col) ? '#ef4444' : '#2a2a3e'}`,
                      borderRadius: 3, padding: '2px 6px', fontSize: 9, cursor: 'pointer',
                    }}
                  >
                    {col}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

// --- Export Section ---
function ExportSection() {
  const config = useConfigStore();

  return (
    <div style={sectionStyle}>
      <div style={sectionHeaderStyle}>Export</div>
      <div style={sectionBodyStyle}>
        <label style={labelStyle}>Model Name</label>
        <input
          type="text"
          value={config.export_name}
          onChange={(e) => config.setField('export_name', e.target.value)}
          placeholder="auto-generated if empty"
          style={inputStyle}
        />
      </div>
    </div>
  );
}

// --- Selected Node Section ---
function SelectedNodeSection({
  node,
  updateParam,
  toggleDisabled,
}: {
  node: Node<NodeData>;
  updateParam: (nodeId: string, key: string, value: unknown) => void;
  toggleDisabled: (nodeId: string) => void;
}) {
  const { definition, params, disabled } = node.data;
  const properties = Object.entries(definition.inputs).filter(([, s]) => !s.is_handle);

  return (
    <div style={{ ...sectionStyle, borderTop: '2px solid #6366f1' }}>
      <div style={{ ...sectionHeaderStyle, color: '#6366f1' }}>
        {definition.display_name}
        <span style={{ fontSize: 9, color: '#555', marginLeft: 6 }}>{node.id}</span>
      </div>
      <div style={sectionBodyStyle}>
        <label style={{ ...labelStyle, display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={disabled}
            onChange={() => toggleDisabled(node.id)}
          />
          Disabled (ablation)
        </label>

        {properties.map(([key, spec]) => {
          const value = params[key] ?? spec.default ?? '';
          return (
            <div key={key} style={{ marginBottom: 4 }}>
              <label style={labelStyle}>
                {key}
                {spec.required && <span style={{ color: '#ef4444' }}> *</span>}
              </label>
              {spec.choices ? (
                <select
                  value={String(value)}
                  onChange={(e) => updateParam(node.id, key, e.target.value)}
                  style={inputStyle}
                >
                  {spec.choices.map((c: unknown) => (
                    <option key={String(c)} value={String(c)}>{String(c)}</option>
                  ))}
                </select>
              ) : spec.dtype === 'BOOL' ? (
                <input
                  type="checkbox"
                  checked={Boolean(value)}
                  onChange={(e) => updateParam(node.id, key, e.target.checked)}
                />
              ) : spec.dtype === 'INT' || spec.dtype === 'FLOAT' ? (
                <input
                  type="number"
                  value={value === null || value === undefined ? '' : Number(value)}
                  min={spec.min_val ?? undefined}
                  max={spec.max_val ?? undefined}
                  step={spec.dtype === 'FLOAT' ? 0.001 : 1}
                  onChange={(e) => {
                    if (e.target.value === '') {
                      updateParam(node.id, key, null);
                    } else {
                      const v = spec.dtype === 'INT'
                        ? parseInt(e.target.value)
                        : parseFloat(e.target.value);
                      updateParam(node.id, key, isNaN(v) ? null : v);
                    }
                  }}
                  style={inputStyle}
                />
              ) : (
                <input
                  type="text"
                  value={String(value)}
                  onChange={(e) => updateParam(node.id, key, e.target.value)}
                  style={inputStyle}
                />
              )}
            </div>
          );
        })}

        {properties.length === 0 && (
          <div style={{ color: '#555', fontSize: 11, textAlign: 'center', padding: 8 }}>
            No configurable properties
          </div>
        )}
      </div>
    </div>
  );
}

// --- Numeric Input (allows free typing, syncs on valid values) ---
function NumericInput({
  value,
  onChange,
  fallback,
  min,
  max,
  step,
  integer,
  style,
}: {
  value: number;
  onChange: (v: number) => void;
  fallback: number;
  min?: number;
  max?: number;
  step?: number;
  integer?: boolean;
  style?: React.CSSProperties;
}) {
  const [local, setLocal] = useState(String(value));
  const prevValue = useRef(value);

  // Sync from store when it changes externally (not from our own edits)
  useEffect(() => {
    if (value !== prevValue.current) {
      setLocal(String(value));
      prevValue.current = value;
    }
  }, [value]);

  return (
    <input
      type="text"
      inputMode="decimal"
      value={local}
      step={step}
      onChange={(e) => setLocal(e.target.value)}
      onBlur={() => {
        const v = integer ? parseInt(local) : parseFloat(local);
        if (isNaN(v) || (min !== undefined && v < min) || (max !== undefined && v > max)) {
          setLocal(String(fallback));
          onChange(fallback);
        } else {
          prevValue.current = v;
          setLocal(String(v));
          onChange(v);
        }
      }}
      style={style}
    />
  );
}

// --- Styles ---
const panelStyle: React.CSSProperties = {
  width: 280,
  background: '#12121a',
  borderLeft: '1px solid #2a2a3e',
  overflow: 'auto',
  fontFamily: 'system-ui, sans-serif',
};

const sectionStyle: React.CSSProperties = {
  borderBottom: '1px solid #2a2a3e',
};

const sectionHeaderStyle: React.CSSProperties = {
  padding: '8px 12px',
  color: '#a0a0b0',
  fontSize: 11,
  fontWeight: 600,
  textTransform: 'uppercase',
  letterSpacing: 0.5,
  background: '#0e0e16',
};

const sectionBodyStyle: React.CSSProperties = {
  padding: '8px 12px',
};

const labelStyle: React.CSSProperties = {
  color: '#808090',
  fontSize: 10,
  display: 'block',
  marginBottom: 2,
  marginTop: 4,
};

const bulkBtnStyle: React.CSSProperties = {
  background: 'transparent',
  border: '1px solid #2a2a3e',
  borderRadius: 3,
  color: '#808090',
  fontSize: 9,
  padding: '1px 6px',
  cursor: 'pointer',
};

const inputStyle: React.CSSProperties = {
  width: '100%',
  background: '#1e1e2e',
  border: '1px solid #2a2a3e',
  borderRadius: 4,
  color: '#c0c0d0',
  padding: '4px 8px',
  fontSize: 11,
  outline: 'none',
  marginBottom: 4,
};
