import { useState } from 'react';
import { useGraphStore } from '../../store/graphStore';
import { useConfigStore } from '../../store/configStore';
import { useExecutionStore } from '../../store/executionStore';
import { executeGraph, saveGraph, listGraphs, loadGraph, pauseTraining, resumeTraining, stopTraining } from '../../api/client';

export function Toolbar() {
  const toSchema = useGraphStore((s) => s.toGraphSchema);
  const loadFromSchema = useGraphStore((s) => s.loadFromSchema);
  const clearGraph = useGraphStore((s) => s.clearGraph);
  const getConfig = useConfigStore((s) => s.getConfig);
  const loadConfig = useConfigStore((s) => s.loadConfig);
  const resetConfig = useConfigStore((s) => s.reset);
  const sessionId = useExecutionStore((s) => s.sessionId);
  const isRunning = useExecutionStore((s) => s.isRunning);
  const isPaused = useExecutionStore((s) => s.isPaused);
  const executionId = useExecutionStore((s) => s.executionId);
  const setRunning = useExecutionStore((s) => s.setRunning);
  const setExecutionId = useExecutionStore((s) => s.setExecutionId);
  const setResults = useExecutionStore((s) => s.setResults);
  const setErrors = useExecutionStore((s) => s.setErrors);
  const clearProgress = useExecutionStore((s) => s.clearProgress);

  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [loadDialogOpen, setLoadDialogOpen] = useState(false);
  const [graphName, setGraphName] = useState('');
  const [graphList, setGraphList] = useState<{ id: string; name: string }[]>([]);

  const handleExecute = async () => {
    if (isRunning) return;
    const config = getConfig();
    if (!config.file_id) {
      setErrors(['No CSV file uploaded. Upload a dataset in the Config panel.']);
      return;
    }
    if (!config.input_columns) {
      setErrors(['No input columns selected.']);
      return;
    }
    if (!config.target_columns) {
      setErrors(['No target columns selected.']);
      return;
    }
    setRunning(true);
    clearProgress();
    setErrors([]);
    const schema = toSchema();
    try {
      const res = await executeGraph(schema, config, sessionId);
      setExecutionId(res.execution_id);
      // Training runs in background; WebSocket handles completion/errors.
    } catch (err: any) {
      setErrors([err.message || 'Execution failed']);
      setRunning(false);
    }
  };

  const handlePause = async () => {
    if (executionId) await pauseTraining(executionId);
  };

  const handleResume = async () => {
    if (executionId) await resumeTraining(executionId);
  };

  const handleStop = async () => {
    if (executionId) await stopTraining(executionId);
  };

  const handleSave = async () => {
    const schema = toSchema();
    schema.name = graphName;
    await saveGraph(schema, getConfig(), '', graphName);
    setSaveDialogOpen(false);
    setGraphName('');
  };

  const handleOpenLoad = async () => {
    const graphs = await listGraphs();
    setGraphList(Object.values(graphs));
    setLoadDialogOpen(true);
  };

  const handleLoad = async (id: string) => {
    const data = await loadGraph(id);
    loadFromSchema(data.graph);
    if (data.config) {
      loadConfig(data.config);
    }
    setLoadDialogOpen(false);
  };

  const handleClear = () => {
    clearGraph();
    resetConfig();
  };

  return (
    <div style={{
      background: '#0a0a12',
      borderBottom: '1px solid #2a2a3e',
      padding: '6px 16px',
      display: 'flex',
      alignItems: 'center',
      gap: 8,
      fontFamily: 'system-ui, sans-serif',
    }}>
      <span style={{ color: '#6366f1', fontWeight: 700, fontSize: 14, marginRight: 16 }}>
        VisDL
      </span>

      <button onClick={handleExecute} disabled={isRunning} style={{
        ...btnStyle,
        background: isRunning ? '#2a2a3e' : '#22c55e',
      }}>
        {isRunning ? (isPaused ? 'Paused' : 'Running...') : 'Execute'}
      </button>

      {isRunning && !isPaused && (
        <button onClick={handlePause} style={{ ...btnStyle, background: '#f59e0b' }}>
          Pause
        </button>
      )}
      {isRunning && isPaused && (
        <button onClick={handleResume} style={{ ...btnStyle, background: '#22c55e' }}>
          Resume
        </button>
      )}
      {isRunning && (
        <button onClick={handleStop} style={{ ...btnStyle, background: '#ef4444' }}>
          Stop
        </button>
      )}

      <button onClick={() => setSaveDialogOpen(true)} style={btnStyle}>Save</button>
      <button onClick={handleOpenLoad} style={btnStyle}>Load</button>
      <button onClick={handleClear} style={{ ...btnStyle, background: '#2a2a3e' }}>Clear</button>

      {/* Save dialog */}
      {saveDialogOpen && (
        <div style={dialogStyle}>
          <input
            type="text"
            value={graphName}
            onChange={(e) => setGraphName(e.target.value)}
            placeholder="Graph name..."
            style={inputStyle}
            autoFocus
          />
          <button onClick={handleSave} style={btnStyle}>Save</button>
          <button onClick={() => setSaveDialogOpen(false)} style={{ ...btnStyle, background: '#2a2a3e' }}>
            Cancel
          </button>
        </div>
      )}

      {/* Load dialog */}
      {loadDialogOpen && (
        <div style={dialogStyle}>
          {graphList.length === 0 ? (
            <span style={{ color: '#555', fontSize: 11 }}>No saved graphs</span>
          ) : (
            graphList.map((g) => (
              <button
                key={g.id}
                onClick={() => handleLoad(g.id)}
                style={{ ...btnStyle, background: '#1e1e2e' }}
              >
                {g.name}
              </button>
            ))
          )}
          <button onClick={() => setLoadDialogOpen(false)} style={{ ...btnStyle, background: '#2a2a3e' }}>
            Cancel
          </button>
        </div>
      )}
    </div>
  );
}

const btnStyle: React.CSSProperties = {
  background: '#3b82f6',
  color: '#fff',
  border: 'none',
  borderRadius: 4,
  padding: '5px 12px',
  fontSize: 11,
  cursor: 'pointer',
  fontWeight: 500,
};

const dialogStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 6,
  marginLeft: 8,
  padding: '4px 8px',
  background: '#1a1a2a',
  borderRadius: 6,
  border: '1px solid #2a2a3e',
};

const inputStyle: React.CSSProperties = {
  background: '#1e1e2e',
  border: '1px solid #2a2a3e',
  borderRadius: 4,
  color: '#c0c0d0',
  padding: '4px 8px',
  fontSize: 11,
  outline: 'none',
};
