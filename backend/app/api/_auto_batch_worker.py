"""Subprocess worker for auto-batch-size search.

Runs in a separate process so ALL GPU memory is freed on exit —
prevents OOM leaks from corrupted autograd graphs during binary search.

Usage: echo '{"graph": ..., "input_dim": 512, ...}' | python -m backend.app.api._auto_batch_worker
"""
import gc
import json
import sys

import torch


def _build_model(graph_dict: dict, input_dim: int):
    """Build a GraphModule from a graph schema dict."""
    from backend.app.nodes.registry import NodeRegistry
    from backend.app.engine.executor import topological_sort
    from backend.app.engine.graph import Edge, Graph, NodeInstance
    from backend.app.engine.graph_module import trace_graph, infer_shapes_graph, GraphModule

    # Trigger node auto-discovery with correct package path
    NodeRegistry.discover("backend.app.nodes")

    nodes = {
        n["id"]: NodeInstance(
            id=n["id"], node_type=n["node_type"],
            params=n.get("params", {}), disabled=n.get("disabled", False),
            position=n.get("position"),
        )
        for n in graph_dict["nodes"]
    }
    edges = [
        Edge(
            id=e["id"], source_node=e["source_node"], source_output=e["source_output"],
            target_node=e["target_node"], target_input=e["target_input"], order=e.get("order", 0),
        )
        for e in graph_dict["edges"]
    ]
    graph = Graph(nodes=nodes, edges=edges)

    order = topological_sort(graph)
    layer_results: dict[str, tuple] = {}
    for node_id in order:
        node_inst = graph.nodes[node_id]
        node_cls = NodeRegistry.get(node_inst.node_type)
        node = node_cls()
        node._node_id = node_id
        kwargs: dict = {}
        for edge in graph.get_incoming_edges(node_id):
            src = layer_results.get(edge.source_node)
            if src is not None:
                kwargs[edge.target_input] = src[edge.source_output]
        for k, v in node_inst.params.items():
            if k not in kwargs:
                kwargs[k] = v
        if node_inst.disabled:
            layer_results[node_id] = node.on_disable(**kwargs)
        else:
            layer_results[node_id] = node.execute(**kwargs)

    all_sources = {e.source_node for e in graph.edges}
    terminal = None
    for nid in reversed(order):
        if nid not in all_sources:
            terminal = nid
            break
    if terminal is None:
        terminal = order[-1]
    arch_ref = layer_results[terminal][0]

    all_nodes = trace_graph([arch_ref])
    infer_shapes_graph(all_nodes, input_dim)
    return GraphModule(all_nodes, [arch_ref])


def run_search(config: dict) -> dict:
    graph_dict = config["graph"]
    input_dim = config["input_dim"]
    optimizer_name = config["optimizer"]
    max_cap = config.get("max_cap")

    model = _build_model(graph_dict, input_dim)
    model = model.to("cuda")
    model.train()

    device = torch.device("cuda")
    log: list[dict] = []
    opt_cls = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW,
               "SGD": torch.optim.SGD}

    output_dim = config.get("output_dim", 1)
    loss_fn = torch.nn.MSELoss()

    def _try(bs: int) -> bool:
        x = y = out = loss = opt = None
        try:
            opt = opt_cls.get(optimizer_name, torch.optim.Adam)(model.parameters(), lr=1e-3)
            x = torch.randn(bs, input_dim, device=device)
            y = torch.randn(bs, output_dim, device=device)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            log.append({"batch_size": bs, "status": "ok"})
            return True
        except RuntimeError as e:
            if "out of memory" not in str(e):
                raise
            log.append({"batch_size": bs, "status": "oom"})
            return False
        finally:
            del x, y, out, loss, opt
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

    # Exponential growth: 2, 4, 8, ...
    last_ok = 0
    bs = 2
    while True:
        if max_cap and bs > max_cap:
            bs = max_cap
        if not _try(bs):
            break
        last_ok = bs
        if max_cap and bs >= max_cap:
            return {"max_bs": bs, "log": log}
        bs *= 2

    # Binary search between last_ok and first failure
    lo, hi = last_ok, bs
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _try(mid):
            lo = mid
        else:
            hi = mid

    # 20% safety margin — accounts for CUDA context overhead in the main
    # process, DataLoader pinned memory, and other training-time allocations
    # that this synthetic test doesn't replicate.
    safe = int(lo * 0.80)
    return {"max_bs": max(1, safe), "log": log}


if __name__ == "__main__":
    config = json.loads(sys.stdin.read())
    try:
        result = run_search(config)
        print(json.dumps(result), flush=True)
    except Exception as e:
        print(json.dumps({"error": str(e)}), flush=True)
    finally:
        # Explicit GPU cleanup before exit to prevent slow CUDA teardown
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        # Force-exit to skip slow Python/CUDA cleanup
        import os
        os._exit(0)
