import { Canvas } from './components/Canvas/Canvas';
import { NodePalette } from './components/NodePalette/NodePalette';
import { ConfigPanel } from './components/ConfigPanel/ConfigPanel';
import { TrainingDashboard } from './components/TrainingDashboard/TrainingDashboard';
import { Toolbar } from './components/Toolbar/Toolbar';
import { SystemStatusBar } from './components/SystemStatusBar/SystemStatusBar';
import { useNodeRegistry } from './hooks/useNodeRegistry';
import { useWebSocket } from './hooks/useWebSocket';

export default function App() {
  const definitions = useNodeRegistry();
  useWebSocket();

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      background: '#0a0a12',
      color: '#e0e0f0',
    }}>
      <Toolbar />
      <SystemStatusBar />
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <NodePalette definitions={definitions} />
        <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
          <Canvas />
          <TrainingDashboard />
        </div>
        <ConfigPanel />
      </div>
    </div>
  );
}
