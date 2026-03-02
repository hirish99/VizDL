# VizDL

Visual Deep Learning Research Tool — a ComfyUI-style web app for constructing, training, and ablating neural network architectures through a node graph UI.

<img width="1789" height="960" alt="image" src="https://github.com/user-attachments/assets/d57d5597-b8d2-47a0-8aae-ba2a7ee0b183" />

<img width="1195" height="837" alt="Screenshot from 2026-03-01 20-06-55" src="https://github.com/user-attachments/assets/78b3c023-1819-4933-83f4-60dfa60cf2b6" />

## Prerequisites

- Python 3.10+
- Node.js 18+
- NVIDIA GPU with CUDA (optional, but recommended for training)

## Setup

### 1. Clone the repo

```bash
git clone <repo-url>
cd VizDL
```

### 2. Create and activate a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install backend dependencies

```bash
pip install -r backend/requirements.txt
```

> **Note on PyTorch:** The `requirements.txt` lists `torch>=2.1.0`, which installs CPU-only by default. For GPU support, install PyTorch with CUDA separately first — see https://pytorch.org/get-started/locally/ for the correct command for your CUDA version. Example:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu128
> ```

### 4. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

## Running

You need **two terminals** (or use `&` / tmux / screen).

### Terminal 1 — Backend

```bash
cd VizDL
.venv/bin/python backend/run.py
```

The API server starts at **http://localhost:8000**.

### Terminal 2 — Frontend

```bash
cd VizDL/frontend
npm run dev
```

The UI starts at **http://localhost:5173**. Open this URL in your browser. The Vite dev server proxies `/api` and `/ws` requests to the backend automatically.

### Remote / SSH Access

If the server is on a remote machine, forward both ports:

```bash
ssh -L 5173:localhost:5173 -L 8000:localhost:8000 user@host
```

Then open **http://localhost:5173** on your local machine.

## How It Works

VizDL splits the workflow between the **canvas** (left) and the **config panel** (right):

### Canvas (Architecture)

The canvas is where you visually design your neural network architecture by dragging nodes from the palette and connecting them with edges. Only **layer nodes** and **structural nodes** belong on the canvas:

- **Layer nodes**: Linear, ReLU, Sigmoid, Tanh, GELU, Dropout, BatchNorm1d, LayerNorm
- **Structural nodes**: Split, Concat, DotProduct, Add (for branching/merging)

Connect nodes by dragging from an output port to an input port. Each connection carries an `ARCH` data type — a lazy computation graph reference. The pipeline automatically compiles your canvas architecture into a trainable `nn.Module` when you execute.

### Config Panel (Data, Training, Export)

The right-side config panel controls everything outside the architecture:

| Section | Fields | Description |
|---------|--------|-------------|
| **Data** | CSV File, Input Columns, Target Columns, Val Split, Batch Size, Shuffle | Upload a CSV, select which columns are features vs targets, configure data splitting |
| **Training** | Loss Function, Optimizer, Learning Rate, Epochs | Select loss (MSE/CrossEntropy/L1), optimizer (Adam/SGD/AdamW), set hyperparameters |
| **Test Data** | Test CSV, Test Input/Target Columns | (Optional, collapsible) Upload a separate test set for post-training evaluation |
| **Export** | Model Name | Name for the exported model weights file |
| **Node Properties** | (varies per node) | Click a node on the canvas to edit its parameters (e.g., `out_features` for Linear, `p` for Dropout). Each node also has a "Disabled (ablation)" checkbox. |

After uploading a CSV, the available columns appear as clickable toggle buttons — green for input features, red for targets. Use the "All" / "None" buttons for bulk selection.

### Toolbar

The toolbar above the canvas has:

- **Execute**: Run the pipeline (build model from canvas architecture + train with config panel settings)
- **Pause / Resume / Stop**: Control a running training session
- **Save**: Save the current canvas graph + config for reuse
- **Load**: Load a previously saved graph (dropdown with available graphs)

### End-to-End Workflow

1. **Design architecture** on the canvas (drag nodes from palette, connect them)
2. **Upload CSV** in the config panel Data section
3. **Select columns**: click column buttons to mark features (green) and targets (red)
4. **Set training params**: loss function, optimizer, learning rate, epochs
5. **Click Execute** in the toolbar
6. **Monitor training**: live loss curves appear in the training dashboard; system stats (CPU/RAM/VRAM) show in the status bar
7. **Pause/Resume/Stop** as needed — checkpoints are saved automatically
8. **Results**: trained weights exported to `data/weights/`, metrics displayed in the dashboard

## Architecture

```
Browser (React Flow)  <->  FastAPI Backend (Python)
   |                          |
   +- Node Palette           +- Node Registry (auto-discovery)
   +- Canvas (drag/connect)  +- Graph Validator (DAG, types)
   +- Config Panel           +- Pipeline Executor
   +- Training Dashboard     +- Training Loop (PyTorch, GPU)
   +- System Status Bar      +- WebSocket (real-time progress)
```

### Execution Pipeline

When you click Execute, the backend runs a fixed pipeline:

1. **Topo-sort canvas nodes**, execute each → produces an `ArchRef` computation graph
2. **CSVLoader** → loads and parses the uploaded CSV using config panel settings
3. **DataSplitter** → splits into train/val loaders
4. **GraphModel** → traces the ArchRef DAG backward, infers shapes, compiles into `GraphModule(nn.Module)`
5. **Optimizer** → creates the selected optimizer with the configured learning rate
6. **Loss function** → creates the selected loss
7. **TrainingLoop** → trains with live progress via WebSocket
8. **MetricsCollector** → collects final metrics
9. **Evaluator** → (optional) evaluates on test data
10. **ModelExport** → saves trained weights

Steps 2-10 are handled automatically by the pipeline — you only configure them via the config panel.

### DAG Model Architecture

VizDL uses a **DAG-based architecture system** for maximum flexibility. Instead of building models as a linear chain (`nn.Sequential`), layer nodes produce `ArchRef` objects — a lazy computation graph that can represent arbitrary DAG topologies:

1. **Layer nodes** (Linear, ReLU, etc.) produce `ArchRef` outputs connected via the `ARCH` data type
2. **Structural nodes** (Split, Concat, DotProduct, Add) enable branching and merging
3. **GraphModel** (called automatically by the pipeline) traces the `ArchRef` DAG backward from the terminal node and compiles it into a `GraphModule(nn.Module)` with a DAG-shaped forward pass

This enables skip connections, parallel branches, split/concat, and operator networks (like DeepONet) — all configured visually on the canvas.

## Tech Stack

- **Backend**: Python, FastAPI, PyTorch, Pydantic
- **Frontend**: React, TypeScript, React Flow, Zustand, Recharts, Vite
- **Communication**: REST for CRUD, WebSocket for real-time training telemetry and system monitoring

## Node Types (23)

| Category | Nodes | Description |
|----------|-------|-------------|
| Layers | Linear, ReLU, Sigmoid, Tanh, GELU, Dropout, BatchNorm1d, LayerNorm | Neural network layer building blocks. Place on canvas, connect via ARCH ports. |
| Structural | Split, Concat, DotProduct, Add | DAG topology operations for branching and merging. Split has multiple output ports; Concat, DotProduct, Add have two input ports. |
| Data | CSV Loader, Data Splitter | Used internally by the pipeline. Configured via config panel, not placed on canvas. |
| Model | Graph Model, Model Export | Used internally by the pipeline. Graph Model compiles the canvas ArchRef DAG into nn.Module. |
| Loss | MSE Loss, Cross Entropy Loss, L1 Loss | Used internally by the pipeline. Selected via config panel dropdown. |
| Optimizer | SGD, Adam, AdamW | Used internally by the pipeline. Selected via config panel dropdown. |
| Training | Training Loop | Used internally by the pipeline. Epochs and learning rate set in config panel. |
| Metrics | Metrics Collector, Evaluator | Used internally by the pipeline. Results shown in training dashboard. |

**Canvas nodes** (what you drag and connect): Layers + Structural.
**Pipeline nodes** (configured via config panel, run automatically): Data, Model, Loss, Optimizer, Training, Metrics.

## CSV Data Format

VizDL loads training data from CSV files. Upload a CSV via the config panel, then select input and target columns.

### Column Selection

After uploading, columns appear as clickable toggle buttons:
- Click a column to mark it as **input** (green) or **target** (red)
- Use **All** / **None** buttons for bulk selection
- The underlying column config is a comma-separated string (e.g., `x1,x2,x3`)

### Glob Pattern Support

The backend supports `fnmatch`-style glob patterns in column names. This is useful for saved graph configs or API calls with high-dimensional datasets:

| Pattern | Matches |
|---------|---------|
| `feature_*` | All columns matching `feature_0`, `feature_1`, `feature_2`, ... |
| `x?` | `x1`, `x2`, ... but not `x10` |
| `col_[0-9]*` | Columns starting with `col_` followed by digits |

### CSV Requirements

- All values must be numeric (float)
- First row must be column headers
- No index column (or exclude it from input/target selection)

### Example: Simple Regression

```csv
x1,x2,target
0.1,0.2,0.5
0.3,0.4,1.1
```

Config panel: select `x1`, `x2` as inputs (green); `target` as target (red).

### Example: High-Dimensional

```csv
feature_0,feature_1,...,feature_63,coord_x,coord_y,target
0.12,0.34,...,0.56,1.0,2.0,0.789
```

Saved config uses glob: `input_columns: feature_*,coord_x,coord_y` / `target_columns: target`

## Saved Graphs

Three pre-built graphs are included in `data/graphs/`:

| Graph | Architecture | Use Case |
|-------|-------------|----------|
| **Regression (3-layer)** | Linear(32) → ReLU → Linear(16) → ReLU → Linear(1) | Tabular regression |
| **Classification (wide)** | Linear(64) → ReLU → Dropout(0.3) → Linear(32) → ReLU → Linear(1) → Sigmoid | Binary classification |
| **DeepONet** | Split(64,2) → branch net (3×Linear+ReLU) + trunk net (3×Linear+ReLU) → DotProduct | Operator learning (PDE solving) |

Load a saved graph from the **Load** dropdown in the toolbar. Saved graphs include both the canvas architecture and the config panel settings.

## Ablation

Every node has a **Disabled (ablation)** checkbox in the config panel's node properties section. Click a node on the canvas, then check the box to disable it. Disabled nodes are bypassed during execution:

- **Activations** (ReLU, GELU, Sigmoid, Tanh) + **Dropout**: vanish from the graph entirely (upstream ArchRef passes through)
- **Linear, BatchNorm1d, LayerNorm**: replaced with `nn.Identity` (preserves tensor shape)
- **Structural nodes**: pass through first input

This enables systematic ablation studies — disable individual layers, re-execute, and compare model performance.

## Training Control

During training:
- **Pause**: saves a checkpoint and suspends training (button changes to "Resume")
- **Resume**: restores from checkpoint and continues training
- **Stop**: saves a checkpoint and ends early with partial results
- Checkpoints saved to `data/weights/checkpoints/`
- API endpoints: `POST /api/execute/{id}/pause`, `/resume`, `/stop`

## Adding New Nodes

Drop a decorated class in `backend/app/nodes/`:

```python
from .base import BaseNode, DataType, InputSpec, OutputSpec
from .registry import NodeRegistry

@NodeRegistry.register("MyNode")
class MyNode(BaseNode):
    CATEGORY = "MyCategory"
    DISPLAY_NAME = "My Node"

    @classmethod
    def INPUT_TYPES(cls):
        return {"x": InputSpec(dtype=DataType.TENSOR, required=True)}

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.TENSOR, name="output")]

    def execute(self, **kwargs):
        return (kwargs["x"] * 2,)
```

Auto-discovered on startup. No registration boilerplate elsewhere.

### Adding Layer Nodes (ARCH type)

Layer nodes use the `ARCH` data type to participate in the DAG architecture system:

```python
@NodeRegistry.register("MyLayer")
class MyLayerNode(BaseNode):
    CATEGORY = "Layers"
    DISPLAY_NAME = "My Layer"

    @classmethod
    def INPUT_TYPES(cls):
        return {"input": InputSpec(dtype=DataType.ARCH, required=True)}

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.ARCH, name="output")]

    def execute(self, **kwargs):
        from ..engine.graph_module import ArchNode, ArchRef
        upstream = kwargs["input"]
        node = ArchNode(self._node_id, "MyTorchModule", {"param": value}, [upstream])
        return (ArchRef(node),)

    def on_disable(self, **kwargs):
        # Return upstream directly (vanish) or return Identity ArchNode
        return (kwargs["input"],)
```

## Key Design Decisions

- **Canvas vs config panel**: The canvas controls only the neural network architecture (layers + structural nodes). Data loading, training, loss, optimizer, and export are configured in the config panel and run automatically by the pipeline. This keeps the canvas focused on what matters — the model topology.
- **DAG architecture via ArchRef**: Layer nodes emit lazy `ArchRef` graph nodes. The pipeline's GraphModel step traces the DAG and compiles into a `GraphModule(nn.Module)` with automatic shape inference. Supports arbitrary topologies including skip connections, parallel branches, and operator networks.
- **GPU-aware**: Model placed on CUDA if available; training loop moves batches to model's device.
- **Ablation as first-class**: Every node has a disable toggle. Disabled layers become `nn.Identity` or vanish from the graph entirely.
- **Glob column patterns**: CSV Loader supports `fnmatch` patterns (`feature_*`) for selecting columns in high-dimensional datasets without listing them individually.
- **Pause/resume/stop training**: Thread-safe training controller with checkpoint save/restore.
- **Large dataset support**: Streaming file upload (no memory limit), chunked CSV reading for files >100MB, zero-copy train/val splitting via `torch.utils.data.Subset`.
- **Live system monitoring**: CPU, RAM, and VRAM usage streamed over WebSocket and displayed in the status bar.

## Testing

```bash
# Run all tests (excluding slow/stress tests)
cd VizDL
.venv/bin/python -m pytest backend/tests/ -v

# Run slow/stress tests (large datasets, CLI integration)
.venv/bin/python -m pytest backend/tests/ -v -m slow
```

Tests covering: graph module (DAG tracing, shape inference, forward pass), structural nodes (Split, Concat, DotProduct, Add), layer ablation, model assembly, pipeline execution, graph validation, data node glob patterns, training control (pause/resume/stop, checkpoints), and large dataset handling.

## Project Structure

```
backend/
  run.py                          # Entry point (starts uvicorn)
  requirements.txt                # Python dependencies
  pytest.ini                      # Test configuration
  tests/                          # Test suite (190+ tests)
  app/
    main.py                       # FastAPI app, CORS, lifespan, WS endpoints
    config.py                     # Pydantic settings (upload dir, max size, etc.)
    api/routes.py                 # REST endpoints (execute, pause/resume/stop, graphs, upload)
    api/websocket.py              # WS connection manager (training telemetry)
    api/system_monitor.py         # WS system stats (CPU/RAM/GPU)
    engine/
      executor.py                 # Topo sort + execute nodes in graph
      graph_module.py             # ArchNode, ArchRef, trace_graph, infer_shapes_graph, GraphModule
      pipeline.py                 # Fixed pipeline: canvas graph → data → model → train → export
      validator.py                # DAG/type/input validation
      graph.py                    # Graph data structure (nodes + edges)
      training_control.py         # Thread-safe pause/resume/stop controller
      checkpoint.py               # Model checkpoint save/load
      session.py                  # Active execution session tracking
    nodes/                        # All node implementations (auto-discovered)
      base.py                     # BaseNode, DataType (incl. ARCH), InputSpec, OutputSpec
      registry.py                 # NodeRegistry (decorator-based auto-registration)
      layers.py                   # Linear, ReLU, Sigmoid, Tanh, GELU, Dropout, BatchNorm1d, LayerNorm
      structural.py               # Split, Concat, DotProduct, Add
      data.py                     # CSVLoader (with glob patterns), DataSplitter
      model_assembly.py           # GraphModel, ModelExport
    models/schemas.py             # Pydantic schemas (GraphSchema, PipelineConfig, ExecuteRequest, etc.)
frontend/
  package.json                    # Node dependencies
  vite.config.ts                  # Dev server + proxy to backend
  src/
    App.tsx                       # Main layout (Canvas + ConfigPanel + StatusBar + Toolbar)
    api/client.ts                 # REST/WS client (execute, upload, save/load graphs)
    api/types.ts                  # TypeScript API types
    types/graph.ts                # Graph and PipelineConfig types
    store/
      graphStore.ts               # Canvas state (nodes, edges, selection) via Zustand
      executionStore.ts           # Execution state (running, paused, results) via Zustand
      configStore.ts              # Config panel state (data, training, export settings) via Zustand
    components/
      Canvas/Canvas.tsx           # React Flow canvas (drag-and-drop node graph)
      NodePalette/NodePalette.tsx # Sidebar with draggable node types
      NodeTypes/BaseNode.tsx      # Base visual node component (ports, labels, colors)
      ConfigPanel/ConfigPanel.tsx # Right panel (Data, Training, Test, Export, Node Properties)
      Toolbar/Toolbar.tsx         # Top bar (Execute, Pause/Resume/Stop, Save, Load)
      StatusBar/                  # System status (CPU/RAM/VRAM)
      TrainingDashboard/          # Live loss curves (Recharts)
    hooks/
      useWebSocket.ts             # Training progress WebSocket
      useNodeRegistry.ts          # Fetch node definitions from backend
      useSystemMonitor.ts         # System stats WebSocket
data/
  graphs/                         # Saved graph architectures (regression, classification, DeepONet)
  uploads/                        # Uploaded CSV files
  weights/                        # Trained model weights + checkpoints
```
