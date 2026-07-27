import argparse
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from tqdm import tqdm

INPUT_ROOT = ""
OUTPUT_ROOT = ""
FORCE_REBUILD = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default=INPUT_ROOT)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force-rebuild", action="store_true", default=False)
    return parser.parse_args()

def _format_tqdm_log(parts):
    return "|".join(str(part) for part in parts)

def _stable_seed_offset(name):
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(name))) % 1000003

def graph_distance(G):
    if G.number_of_edges() == 0:
        n = G.number_of_nodes()
        return np.zeros((n, n))
    return np.asarray(nx.floyd_warshall_numpy(G, weight=None), dtype=float)

def _total_magnitude(distances):
    distances = np.asarray(distances, dtype=float)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("D must be square.")
    n = distances.shape[0]
    if n <= 1:
        return float(n)
    scaled = np.power(np.exp(-distances), 1.0)
    denom = np.sum(scaled, axis=0)
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    return float(np.sum(1.0 / denom))

def _magnitudes_per_subgraph(G, subgraphs=None):
    if subgraphs is None:
        subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    distances_list = []
    mags = []
    for subgraph in subgraphs:
        distances = graph_distance(subgraph)
        distances_list.append(distances)
        mags.append(_total_magnitude(distances))
    total_magnitude = float(np.sum(mags)) if mags else 0.0
    return total_magnitude, subgraphs, distances_list, mags

def _edge_removal_scores(edges, original_total, subgraphs, distances_list, mags):
    subgraph_index = [{node: idx for idx, node in enumerate(subgraph.nodes())} for subgraph in subgraphs]
    scores = []
    for node_a, node_b in edges:
        mags_steps = []
        for i, subgraph in enumerate(subgraphs):
            index = subgraph_index[i]
            if node_a not in index or node_b not in index:
                mags_steps.append(mags[i])
                continue
            neighbors_a = set(subgraph.neighbors(node_a)) - {node_b}
            neighbors_b = set(subgraph.neighbors(node_b)) - {node_a}
            merged_neighbors = neighbors_b.difference(neighbors_a)
            dmat = distances_list[i]
            distances_sub = dmat.copy()
            idx_a = index[node_a]
            idx_b = index[node_b]
            for neighbor in merged_neighbors:
                idx_n = index[neighbor]
                distances_sub[idx_a, idx_n] = min(dmat[idx_a, idx_n], dmat[idx_b, idx_n])
            distances_sub = np.delete(distances_sub, idx_b, axis=0)
            distances_sub = np.delete(distances_sub, idx_b, axis=1)
            mags_steps.append(_total_magnitude(distances_sub))
        scores.append(abs(original_total - sum(mags_steps)))
    return scores

def assignment_to_clusters(assignment_matrix):
    rows, cols = np.where(assignment_matrix != 0)
    assignment = [[] for _ in range(assignment_matrix.shape[1])]
    for row, col in zip(rows, cols):
        assignment[col].append(row)
    assignment = np.array(assignment)
    return assignment.squeeze()

def mag_edge_pool(g, n_steps=None):
    original_total, subgraphs, distances_list, mags = _magnitudes_per_subgraph(g)
    this_graph = g.copy()
    n = g.number_of_nodes()
    if n_steps is None:
        n_steps = n - 1
    edges = list(this_graph.edges())
    scores = _edge_removal_scores(edges, original_total, subgraphs, distances_list, mags)
    scores = np.asarray(scores, dtype=float).reshape(-1).tolist()
    n_steps = min(n_steps, len(scores))
    assignment_matrix = np.eye(n)
    order_this = np.argsort(scores).flatten().astype(int).tolist()
    scores = [scores[i] for i in order_this]
    edges = [edges[i] for i in order_this]
    for step_idx in range(0, n_steps):
        min_score_indices = np.flatnonzero(np.isclose(scores, scores[0], rtol=1e-8, atol=1e-12))
        chosen_index = random.choice(min_score_indices)
        node_a, node_b = edges[chosen_index]
        this_nodes = list(this_graph.nodes())
        indx_a = this_nodes.index(node_a)
        indx_b = this_nodes.index(node_b)
        if 0 <= indx_b < assignment_matrix.shape[0]:
            assignment_matrix[indx_a, :] += assignment_matrix[indx_b, :]
            assignment_matrix = np.delete(assignment_matrix, indx_b, axis=0)
        else:
            raise IndexError(f"Index indx_b={indx_b} is out of bounds for array S with shape {assignment_matrix.shape}")
        scores.pop(chosen_index)
        edges.pop(chosen_index)
        this_graph = nx.contracted_nodes(this_graph, node_a, node_b, self_loops=False)
        if this_graph.number_of_edges() == 0:
            break
        new_edges = []
        new_scores = []
        for idx, edge in enumerate(edges):
            if (node_b in edge) | (node_a in edge):
                continue
            new_edges.append(edge)
            new_scores.append(scores[idx])
        edges = new_edges
        scores = new_scores
        if (len(edges) == 0) & (step_idx < n_steps - 1):
            break
    row_sums = assignment_matrix.sum(axis=1, keepdims=True)
    assignment_matrix = assignment_matrix / row_sums
    cluster = assignment_to_clusters(assignment_matrix)
    return this_graph, assignment_matrix, cluster

def _split_for_graph(graph_path):
    graph_path = Path(graph_path).resolve()
    for part in reversed(graph_path.parts[:-1]):
        if part.startswith("Test"):
            return part
        if part in {"Train", "Validation", "Split"}:
            return "Train"
    return graph_path.parent.name

def cluster_path_for_graph(graph_path):
    graph_path = Path(graph_path)
    return Path(OUTPUT_ROOT) / _split_for_graph(graph_path) / graph_path.name

def _normalize_cluster(cluster):
    if cluster.numel() == 0:
        return cluster.long()
    _, inverse = torch.unique(cluster.long(), sorted=True, return_inverse=True)
    return inverse

def _cluster_np_to_labels(cluster_np, expected_size):
    cluster = np.asarray(cluster_np, dtype=np.int64)
    if cluster.ndim == 0:
        cluster = np.full((expected_size,), int(cluster), dtype=np.int64)
    cluster = cluster.reshape(-1)
    if cluster.size != expected_size:
        raise RuntimeError(f"Cluster size mismatch: expected={expected_size} actual={cluster.size}")
    unique_values = {value: idx for idx, value in enumerate(sorted(set(cluster.tolist())))}
    return np.asarray([unique_values[value] for value in cluster.tolist()], dtype=np.int64)

def _coarsen_graph_from_cluster(graph, cluster):
    cluster = _normalize_cluster(cluster).cpu().numpy().astype(np.int64)
    next_graph = nx.Graph()
    for cluster_id in sorted(set(cluster.tolist())):
        next_graph.add_node(int(cluster_id))
    for node_a, node_b in graph.edges():
        cluster_a = int(cluster[int(node_a)])
        cluster_b = int(cluster[int(node_b)])
        if cluster_a != cluster_b:
            next_graph.add_edge(cluster_a, cluster_b)
    return nx.convert_node_labels_to_integers(next_graph, ordering="sorted")

def mag_edge_pool_disjoint(g, n_steps=None):
    original_total, subgraphs, distances_list, mags = _magnitudes_per_subgraph(g)
    edges = list(g.edges())
    n = g.number_of_nodes()
    if n_steps is None:
        n_steps = n - 1
    if not edges or n_steps <= 0:
        cluster = torch.arange(n, dtype=torch.long)
        return g.copy(), None, cluster
    scores = _edge_removal_scores(edges, original_total, subgraphs, distances_list, mags)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    node_to_idx = {node: idx for idx, node in enumerate(g.nodes())}
    cluster = torch.arange(n, dtype=torch.long)
    used_nodes = set()
    selected = 0
    for edge_idx in np.argsort(scores).flatten().astype(int).tolist():
        node_a, node_b = edges[edge_idx]
        if node_a in used_nodes or node_b in used_nodes:
            continue
        idx_a = node_to_idx[node_a]
        idx_b = node_to_idx[node_b]
        cluster[idx_b] = cluster[idx_a]
        used_nodes.add(node_a)
        used_nodes.add(node_b)
        selected += 1
        if selected >= n_steps:
            break
    cluster = _normalize_cluster(cluster)
    return _coarsen_graph_from_cluster(g, cluster), None, cluster

def _list_to_mask(values, expected_size, threshold):
    tensor = torch.as_tensor(values)
    if tensor.numel() != expected_size:
        raise RuntimeError(f"Interface mask size mismatch: expected={expected_size} actual={tensor.numel()}")
    tensor = tensor.reshape(-1)
    if tensor.dtype == torch.bool:
        return tensor.bool()
    return tensor.float() > float(threshold)

def _resolve_interface_mask(graph, metadata):
    n = int(graph.x.size(0))
    if hasattr(graph, "bsa_values"):
        return _list_to_mask(graph.bsa_values, n, 0.0)
    if isinstance(metadata, dict) and "bsa_values" in metadata:
        return _list_to_mask(metadata["bsa_values"], n, 0.0)
    raise RuntimeError("Missing interface source: bsa_values")

def _as_float_tensor(value):
    return value.float() if torch.is_tensor(value) else torch.as_tensor(value, dtype=torch.float32)

def _as_long_tensor(value):
    return value.long() if torch.is_tensor(value) else torch.as_tensor(value, dtype=torch.long)

def _resolve_graph_payload(payload):
    if isinstance(payload, dict):
        if "graph" in payload:
            graph = payload["graph"]
        elif "data" in payload:
            graph = payload["data"]
        elif "tokens" in payload:
            tokens = payload["tokens"]
            x = _as_float_tensor(tokens["x"])
            pos_value = tokens.get("pos")
            pos = _as_float_tensor(pos_value) if pos_value is not None else torch.zeros((x.size(0), 3), dtype=torch.float32)
            edge_index_value = tokens.get("edge_index")
            edge_index = _as_long_tensor(edge_index_value) if edge_index_value is not None else torch.empty((2, 0), dtype=torch.long)
            graph = Data(x=x, pos=pos, edge_index=edge_index)
        elif "x" in payload:
            x = _as_float_tensor(payload["x"])
            pos_value = payload.get("pos")
            pos = _as_float_tensor(pos_value) if pos_value is not None else torch.zeros((x.size(0), 3), dtype=torch.float32)
            edge_index_value = payload.get("edge_index")
            edge_index = _as_long_tensor(edge_index_value) if edge_index_value is not None else torch.empty((2, 0), dtype=torch.long)
            graph = Data(x=x, pos=pos, edge_index=edge_index)
        else:
            graph = payload
    else:
        graph = payload
    if not hasattr(graph, "x"):
        raise KeyError("x")
    graph.x = _as_float_tensor(graph.x)
    if not hasattr(graph, "pos"):
        graph.pos = torch.zeros((graph.x.size(0), 3), dtype=torch.float32)
    else:
        graph.pos = _as_float_tensor(graph.pos)
    if not hasattr(graph, "edge_index"):
        graph.edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        graph.edge_index = _as_long_tensor(graph.edge_index)
    return graph

def _load_graph_package_from_path(graph_path):
    payload = torch.load(graph_path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    return _resolve_graph_payload(payload), metadata

def _to_nx_graph(graph):
    nx_graph = to_networkx(graph, to_undirected=True, remove_self_loops=True)
    return nx.convert_node_labels_to_integers(nx_graph, ordering="sorted")

def compute_cluster(graph, interface_mask):
    current_graph = _to_nx_graph(graph)
    n = current_graph.number_of_nodes()
    interface_mask = interface_mask.bool().reshape(-1)
    if interface_mask.numel() != n:
        raise RuntimeError(f"Interface mask size mismatch: expected={n} actual={interface_mask.numel()}")
    if n == 0:
        return torch.empty(0, dtype=torch.long), n
    if n < max(2, 16) or current_graph.number_of_edges() == 0:
        return torch.arange(n, dtype=torch.long), n
    mask = interface_mask.tolist()
    poolable_nodes = [idx for idx in range(n) if not mask[idx]]
    interface_nodes = [idx for idx in range(n) if mask[idx]]
    if len(poolable_nodes) < 2:
        return torch.arange(n, dtype=torch.long), n
    poolable_subgraph = current_graph.subgraph(poolable_nodes).copy()
    if poolable_subgraph.number_of_edges() == 0:
        return torch.arange(n, dtype=torch.long), n
    n_steps = int(round((1.0 - 0.7) * n))
    n_steps = min(n_steps, max(0, len(poolable_nodes) - 1))
    if n_steps <= 0:
        return torch.arange(n, dtype=torch.long), n
    local_nodes = sorted(poolable_nodes)
    local_mapping = {node: idx for idx, node in enumerate(local_nodes)}
    local_graph = nx.relabel_nodes(poolable_subgraph, local_mapping, copy=True)
    if n_steps <= max(1, len(local_nodes) // 2):
        _, _, cluster_np = mag_edge_pool_disjoint(local_graph, n_steps=n_steps)
    else:
        _, _, cluster_np = mag_edge_pool(local_graph, n_steps=n_steps)
    local_cluster = _cluster_np_to_labels(cluster_np, len(local_nodes))
    full_cluster = torch.full((n,), -1, dtype=torch.long)
    for cluster_id, node in enumerate(interface_nodes):
        full_cluster[node] = cluster_id
    offset = len(interface_nodes)
    for local_idx, node in enumerate(local_nodes):
        full_cluster[node] = int(local_cluster[local_idx]) + offset
    if (full_cluster < 0).any():
        raise RuntimeError("Failed to assign all nodes to clusters.")
    return _normalize_cluster(full_cluster), n

def _row_from_payload(graph_path, payload, status, cluster_path):
    return {
        "graph_path": str(graph_path),
        "graph_file": graph_path.name,
        "pdb_id": graph_path.stem,
        "status": status,
        "num_nodes": int(payload.get("nodes_per_level", [0])[0] if payload.get("nodes_per_level") else 0),
        "num_levels": int(payload.get("levels", 0)),
        "interface_count": int(payload.get("interface_count", 0)),
        "nodes_per_level": ",".join(str(x) for x in payload.get("nodes_per_level", [])),
        "cluster_path": str(cluster_path),
        "error": "",
    }

def _worker_compute_cluster(graph_path_str):
    graph_path = Path(graph_path_str)
    cluster_path = cluster_path_for_graph(graph_path)
    if cluster_path.exists() and not FORCE_REBUILD:
        payload = torch.load(cluster_path, map_location="cpu", weights_only=False)
        return _row_from_payload(graph_path, payload, "cached", cluster_path)
    try:
        seed_offset = _stable_seed_offset(graph_path.name)
        random.seed(42 + seed_offset)
        np.random.seed((42 + seed_offset) % (2**32 - 1))
        torch.manual_seed(42 + seed_offset)
        graph, metadata = _load_graph_package_from_path(graph_path)
        interface_mask = _resolve_interface_mask(graph, metadata)
        cluster, num_nodes = compute_cluster(graph, interface_mask)
        payload = {
            "graph_file": graph_path.name,
            "graph_path": str(graph_path),
            "pdb_id": graph_path.stem,
            "levels": 1,
            "ratio": 0.7,
            "min_nodes": 16,
            "metric": "shortest_path_distance",
            "mag_method": "spread",
            "pooling_method": "SpreadEdgePoolApprox",
            "nodes_per_level": [num_nodes],
            "preserve_interface": True,
            "interface_source": "bsa_values",
            "interface_threshold": 0.0,
            "interface_count": int(interface_mask.sum().item()),
            "cluster": cluster.long(),
        }
        cluster_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, cluster_path)
        row = _row_from_payload(graph_path, payload, "ok", cluster_path)
        row["num_levels"] = 1
        return row
    except Exception as exc:
        return {
            "graph_path": str(graph_path),
            "graph_file": graph_path.name,
            "pdb_id": graph_path.stem,
            "status": "fail",
            "num_nodes": 0,
            "num_levels": 0,
            "interface_count": 0,
            "nodes_per_level": "",
            "cluster_path": str(cluster_path),
            "error": f"{type(exc).__name__}: {exc}",
        }

def _collect_graph_paths(input_dirs, input_files, pattern, recursive):
    paths = []
    for input_dir in tqdm(input_dirs, desc="CollectDirs", disable=not input_dirs):
        directory = Path(input_dir)
        if not directory.exists():
            raise FileNotFoundError(directory)
        iterator = directory.rglob(pattern) if recursive else directory.glob(pattern)
        paths.extend(path for path in iterator if path.is_file())
    for input_file in tqdm(input_files, desc="CollectFiles", disable=not input_files):
        path = Path(input_file)
        if not path.exists():
            raise FileNotFoundError(path)
        paths.append(path)
    return sorted(dict.fromkeys(path.resolve() for path in paths))

def build_clusters(graph_paths):
    if not graph_paths:
        return []
    rows = []
    missing_paths = []
    scan = tqdm(graph_paths, desc="ClusterScan")
    for path in scan:
        cluster_path = cluster_path_for_graph(path)
        if cluster_path.exists() and not FORCE_REBUILD:
            row = _worker_compute_cluster(str(path))
            rows.append(row)
        else:
            missing_paths.append(path)
        scan.set_postfix(cached=sum(item["status"] == "cached" for item in rows), missing=len(missing_paths))
    workers = max(1, min(8, len(missing_paths))) if missing_paths else 1
    if missing_paths and workers == 1:
        iterator = tqdm(missing_paths, desc="ClusterBuild")
        for path in iterator:
            row = _worker_compute_cluster(str(path))
            rows.append(row)
            iterator.set_postfix(
                ok=sum(item["status"] == "ok" for item in rows),
                fail=sum(item["status"] == "fail" for item in rows),
                cached=sum(item["status"] == "cached" for item in rows),
            )
    elif missing_paths:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_worker_compute_cluster, str(path)) for path in missing_paths]
            iterator = tqdm(as_completed(futures), total=len(futures), desc="ClusterBuild")
            for future in iterator:
                row = future.result()
                rows.append(row)
                iterator.set_postfix(
                    ok=sum(item["status"] == "ok" for item in rows),
                    fail=sum(item["status"] == "fail" for item in rows),
                    cached=sum(item["status"] == "cached" for item in rows),
                )
    return sorted(rows, key=lambda item: item["graph_file"])

def main():
    args = parse_args()
    global INPUT_ROOT, OUTPUT_ROOT, FORCE_REBUILD
    INPUT_ROOT = args.input_root
    OUTPUT_ROOT = args.output_root
    FORCE_REBUILD = bool(args.force_rebuild)
    graph_paths = _collect_graph_paths([INPUT_ROOT], [], "*.pt", recursive=True)
    if not graph_paths:
        raise RuntimeError("No graph .pt files found.")
    tqdm.write(f"Graphs={len(graph_paths)} | Ratio=0.7 | Output={OUTPUT_ROOT}")
    rows = build_clusters(graph_paths)
    ok = sum(item["status"] == "ok" for item in rows)
    cached = sum(item["status"] == "cached" for item in rows)
    fail = sum(item["status"] == "fail" for item in rows)
    tqdm.write(_format_tqdm_log([f"TotalOK={ok}", f"TotalCached={cached}", f"TotalFail={fail}"]))
    if fail > 0:
        for row in rows:
            if row["status"] == "fail":
                tqdm.write(_format_tqdm_log([row["graph_file"], row["error"]]))
        raise RuntimeError("Cluster build failed.")

if __name__ == "__main__":
    main()
