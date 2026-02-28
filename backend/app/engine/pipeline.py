"""Pipeline executor: orchestrates the fixed training pipeline using layer graph + config."""
import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch

from ..config import settings
from ..nodes.registry import NodeRegistry
from .executor import topological_sort
from .graph import Graph
from .training_control import TrainingController


ProgressCallback = Callable[[dict[str, Any]], None]


def _save_training_log(
    model: torch.nn.Module,
    training_result: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    """Auto-save training log (loss curves + metadata) to data/training_logs/.

    Always called after training completes or is stopped, so the data
    is never lost even if the pipeline fails later.
    """
    from ..nodes.export import _describe_architecture

    history = training_result.get("history", {})
    arch = _describe_architecture(model)
    export_name = config.get("export_name", "").strip()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    stopped = training_result.get("stopped_early", False)
    status = "stopped" if stopped else "completed"

    label = export_name or arch
    filename = f"{label}_{timestamp}_{status}.json"

    log = {
        "model": {
            "architecture": arch,
            "structure": str(model),
            "parameter_count": sum(p.numel() for p in model.parameters()),
        },
        "config": {
            "loss_fn": config.get("loss_fn"),
            "optimizer": config.get("optimizer"),
            "lr": config.get("lr"),
            "epochs": config.get("epochs"),
            "batch_size": config.get("batch_size"),
            "val_ratio": config.get("val_ratio"),
        },
        "training": {
            "epochs": history.get("epoch", []),
            "train_loss": history.get("train_loss", []),
            "val_loss": history.get("val_loss", []),
            "final_train_loss": training_result.get("final_train_loss"),
            "final_val_loss": training_result.get("final_val_loss"),
            "epochs_completed": len(history.get("epoch", [])),
            "epochs_requested": config.get("epochs"),
            "status": status,
        },
        "timestamp": timestamp,
    }

    path = settings.training_logs_dir / filename
    path.write_text(json.dumps(log, indent=2, default=str))
    return path


def execute_pipeline(
    layer_graph: Graph,
    config: dict[str, Any],
    progress_callback: ProgressCallback | None = None,
    training_controller: TrainingController | None = None,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Execute the full training pipeline.

    1. Topo-sort layer nodes, execute chain → ArchRef graph
    2. CSVLoader → dataset
    3. DataSplitter → train_loader, val_loader
    4. GraphModel → model (compiles ArchRef DAG into nn.Module)
    5. Optimizer → optimizer
    6. Loss → loss_fn
    7. TrainingLoop → training_result
    8. MetricsCollector → metrics
    9. ModelExport → file_path
    10. (Optional) Evaluator on test data
    """
    results: dict[str, Any] = {}

    # --- 1. Execute layer graph to get layer_specs ---
    order = topological_sort(layer_graph)
    layer_results: dict[str, tuple] = {}

    for node_id in order:
        node_inst = layer_graph.nodes[node_id]
        node_cls = NodeRegistry.get(node_inst.node_type)
        node = node_cls()
        node._node_id = node_id

        kwargs: dict[str, Any] = {}
        incoming = layer_graph.get_incoming_edges(node_id)

        input_groups: dict[str, list] = {}
        for edge in incoming:
            src_results = layer_results.get(edge.source_node)
            if src_results is None:
                continue
            value = src_results[edge.source_output]
            if edge.target_input not in input_groups:
                input_groups[edge.target_input] = []
            input_groups[edge.target_input].append((edge.order, value))

        for input_name, values in input_groups.items():
            values.sort(key=lambda x: x[0])
            if len(values) == 1:
                kwargs[input_name] = values[0][1]
            else:
                kwargs[input_name] = [v for _, v in values]

        for k, v in node_inst.params.items():
            if k not in kwargs:
                kwargs[k] = v

        if node_inst.disabled:
            outputs = node.on_disable(**kwargs)
        else:
            outputs = node.execute(**kwargs)

        layer_results[node_id] = outputs

        if progress_callback:
            progress_callback({
                "type": "node_complete",
                "node_id": node_id,
                "node_type": node_inst.node_type,
            })

    # Find the final layer node (the one with no outgoing edges in the layer graph)
    all_sources = {e.source_node for e in layer_graph.edges}
    all_targets = {e.target_node for e in layer_graph.edges}
    terminal_nodes = [nid for nid in order if nid not in all_sources or nid not in all_targets]
    # The last node in topo order that has no outgoing edges
    terminal = None
    for nid in reversed(order):
        if nid not in all_sources:
            terminal = nid
            break
    if terminal is None:
        terminal = order[-1]

    arch_ref = layer_results[terminal][0]

    # --- 2. CSVLoader ---
    csv_node = NodeRegistry.get("CSVLoader")()
    dataset = csv_node.execute(
        file_id=config["file_id"],
        input_columns=config["input_columns"],
        target_columns=config["target_columns"],
    )[0]

    if progress_callback:
        progress_callback({"type": "node_complete", "node_id": "_csv", "node_type": "CSVLoader"})

    # --- 3. DataSplitter ---
    splitter_node = NodeRegistry.get("DataSplitter")()
    train_loader, val_loader = splitter_node.execute(
        dataset=dataset,
        val_ratio=config.get("val_ratio", 0.2),
        batch_size=config.get("batch_size", 32),
        shuffle=config.get("shuffle", True),
    )

    if progress_callback:
        progress_callback({"type": "node_complete", "node_id": "_split", "node_type": "DataSplitter"})

    # --- 4. GraphModel ---
    graph_model_node = NodeRegistry.get("GraphModel")()
    graph_model_node._node_id = "_model"
    model = graph_model_node.execute(
        architecture=arch_ref,
        dataset=dataset,
    )[0]

    if progress_callback:
        progress_callback({"type": "node_complete", "node_id": "_model", "node_type": "GraphModel"})

    # --- 5. Loss ---
    loss_type = config.get("loss_fn", "MSELoss")
    loss_node = NodeRegistry.get(loss_type)()
    loss_fn = loss_node.execute()[0]

    # --- 6. Optimizer ---
    optim_type = config.get("optimizer", "Adam")
    optim_node = NodeRegistry.get(optim_type)()
    optimizer = optim_node.execute(
        model=model,
        lr=config.get("lr", 0.001),
    )[0]

    if progress_callback:
        progress_callback({"type": "node_complete", "node_id": "_optim", "node_type": optim_type})

    # --- 7. TrainingLoop ---
    train_node = NodeRegistry.get("TrainingLoop")()
    training_result = train_node.execute(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.get("epochs", 10),
        progress_callback=progress_callback,
        training_controller=training_controller,
        checkpoint_path=checkpoint_path,
    )[0]

    results["training"] = {
        "history": training_result["history"],
        "final_train_loss": training_result.get("final_train_loss"),
        "final_val_loss": training_result.get("final_val_loss"),
        "stopped_early": training_result.get("stopped_early", False),
    }

    # --- 7b. Auto-save training log (always, even if later steps fail) ---
    try:
        log_path = _save_training_log(training_result["model"], training_result, config)
        results["training_log_path"] = str(log_path)
    except Exception:
        pass  # best-effort — don't break pipeline on log write failure

    # --- 8. MetricsCollector ---
    metrics_node = NodeRegistry.get("MetricsCollector")()
    metrics = metrics_node.execute(training_result=training_result)[0]
    results["metrics"] = metrics

    # --- 9. (Optional) Evaluator on test data ---
    test_metrics = None
    test_file_id = config.get("test_file_id")
    if test_file_id:
        test_csv_node = NodeRegistry.get("CSVLoader")()
        test_dataset = test_csv_node.execute(
            file_id=test_file_id,
            input_columns=config.get("test_input_columns", config["input_columns"]),
            target_columns=config.get("test_target_columns", config["target_columns"]),
        )[0]

        eval_node = NodeRegistry.get("Evaluator")()
        test_metrics = eval_node.execute(
            training_result=training_result,
            test_dataset=test_dataset,
        )[0]
        results["test_metrics"] = test_metrics

    # --- 10. ModelExport ---
    export_name = config.get("export_name", "")
    export_node = NodeRegistry.get("ModelExport")()
    file_path = export_node.execute(
        training_result=training_result,
        test_metrics=test_metrics,
        name=export_name,
    )[0]
    results["export_path"] = file_path

    # Cleanup: free GPU memory
    del model, optimizer, loss_fn, train_loader, val_loader, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results
