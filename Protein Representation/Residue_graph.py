import argparse
import csv
import glob
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from Residue_esmc import _compute_esmc_residue_map, _normalize_esmc_device
from Residue_Interface import build_bsa_interface_values, normalize_chain_ids, parse_atom_record

DATASET_ROOT = ""
OUTPUT_ROOT = ""
DATASET_NAMES = ("")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_NAMES))
    parser.add_argument("--esmc-device", default="cuda")
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()

def build_graph_payload(graph: Data, metadata: Dict[str, object]) -> Dict[str, object]:
    return {"graph": graph, "metadata": metadata}

def build_graph_metadata(
    pdb_id: str,
    num_nodes: int,
    num_edges: int,
    node_dim: int,
    edge_dim: int,
    cutoff: float,
    esmc_dim: int,
    bsa_values: Optional[Sequence[object]] = None,
    interface_mask: Optional[Sequence[object]] = None,
) -> Dict[str, object]:
    feature_dims = {"esmc": int(esmc_dim)}
    feature_slices = {"esmc": (0, int(esmc_dim)) if int(esmc_dim) > 0 else None}
    metadata_bsa_values: List[float] = []
    if bsa_values is not None:
        for value in tqdm(bsa_values, desc="metadata_bsa_values", leave=False):
            metadata_bsa_values.append(float(value))
    metadata_interface_mask: List[bool] = []
    if interface_mask is not None:
        for value in tqdm(interface_mask, desc="metadata_interface_mask", leave=False):
            metadata_interface_mask.append(bool(value))
    if metadata_bsa_values and len(metadata_bsa_values) != int(num_nodes):
        raise ValueError(f"metadata_bsa_values_len={len(metadata_bsa_values)} num_nodes={int(num_nodes)}")
    if metadata_interface_mask and len(metadata_interface_mask) != int(num_nodes):
        raise ValueError(f"metadata_interface_mask_len={len(metadata_interface_mask)} num_nodes={int(num_nodes)}")
    interface_count = sum(1 for value in tqdm(metadata_interface_mask, desc="metadata_interface_count", leave=False) if bool(value))
    return {
        "pdb_id": pdb_id,
        "num_nodes": int(num_nodes),
        "num_edges": int(num_edges),
        "node_dim": int(node_dim),
        "edge_dim": int(edge_dim),
        "feature_dims": feature_dims,
        "feature_slices": feature_slices,
        "cutoff": float(cutoff),
        "edge_filter": "bsa_filtered_radius",
        "interface_source": "bsa_values",
        "interface_threshold": 0.0,
        "interface_count": int(interface_count),
        "bsa_values": metadata_bsa_values,
        "interface_mask": metadata_interface_mask,
    }

def normalize_pdb_id(value: object) -> str:
    return os.path.splitext(os.path.basename(str(value).strip()))[0].upper()

def parse_chain_group(group: str) -> List[str]:
    chains: List[str] = []
    seen = set()
    for ch in tqdm(str(group).strip(), desc="parse_chain_group", leave=False):
        if not ch.isalnum() or ch in seen:
            continue
        chains.append(ch)
        seen.add(ch)
    return chains

def parse_binding_chain_pair(raw: str) -> Tuple[List[str], List[str]]:
    for token in tqdm(str(raw).split(","), desc="parse_binding_pair", leave=False):
        token = token.strip()
        if ":" not in token:
            continue
        left, right = token.split(":", 1)
        return parse_chain_group(left), parse_chain_group(right)
    raise ValueError(f"Invalid BindingChainPair: {raw}")

def load_csv_inputs(csv_path: str) -> Tuple[List[str], Dict[str, Tuple[List[str], List[str]]]]:
    binding_pair_map: Dict[str, Tuple[List[str], List[str]]] = {}
    seen = set()
    pdb_ids: List[str] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        for row in tqdm(csv.DictReader(handle), desc=f"csv_inputs:{os.path.basename(csv_path)}", leave=False):
            pdb_id = normalize_pdb_id(row.get("PDB", ""))
            raw_pair = row.get("BindingChainPair", "")
            if pdb_id and pdb_id not in seen:
                pdb_ids.append(pdb_id)
                seen.add(pdb_id)
            if pdb_id and str(raw_pair).strip():
                binding_pair_map[pdb_id] = parse_binding_chain_pair(str(raw_pair))
    return pdb_ids, binding_pair_map

def collect_pdb_paths(input_dir: str) -> Dict[str, str]:
    paths: List[str] = []
    for pattern in tqdm(("*.pdb", "*.pqr"), desc=f"collect_pdb:{os.path.basename(input_dir)}", leave=False):
        paths.extend(sorted(glob.glob(os.path.join(input_dir, pattern))))
    out: Dict[str, str] = {}
    for path in tqdm(paths, desc="index_pdb_paths", leave=False):
        if os.path.isfile(path):
            out[normalize_pdb_id(path)] = path
    return out

def build_dataset_configs(dataset_root: str, output_root: str, dataset_names: Sequence[str]) -> List[Dict[str, str]]:
    configs: List[Dict[str, str]] = []
    for dataset_name in tqdm(dataset_names, desc="dataset_configs", leave=False):
        dataset_dir = os.path.join(dataset_root, dataset_name)
        configs.append({
            "dataset_name": dataset_name,
            "pdb_dir": os.path.join(dataset_dir, "PDBFIXER"),
            "fasta_dir": os.path.join(dataset_dir, "FASTA"),
            "csv_path": os.path.join(dataset_dir, f"{dataset_name}.csv"),
            "output_dir": os.path.join(output_root, dataset_name),
        })
    return configs

def prepare_pdb_batch(input_pdb_dir: str, input_csv_path: str) -> Tuple[List[str], Dict[str, Tuple[List[str], List[str]]]]:
    csv_pdb_ids, binding_pair_map = load_csv_inputs(input_csv_path)
    pdb_paths = collect_pdb_paths(input_pdb_dir)
    selected_pdb_paths: List[str] = []
    selected_binding_pair_map: Dict[str, Tuple[List[str], List[str]]] = {}
    for pdb_id in tqdm(csv_pdb_ids, desc=f"eligible_pdb_ids:{os.path.basename(input_csv_path)}", leave=False):
        if pdb_id in binding_pair_map and pdb_id in pdb_paths:
            selected_pdb_paths.append(pdb_paths[pdb_id])
            selected_binding_pair_map[pdb_id] = binding_pair_map[pdb_id]
    return selected_pdb_paths, selected_binding_pair_map

def parse_pdb_residues(pdb_path: str) -> List[Dict[str, object]]:
    residues: Dict[Tuple[str, int, str, str], Dict[str, object]] = {}
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in tqdm(handle, desc=f"parse_residues:{os.path.basename(pdb_path)}", leave=False):
            record = parse_atom_record(line)
            if record is None:
                continue
            atom_name = str(record["atom_name"])
            resname = str(record["resname"])
            chain_id = str(record["chain_id"])
            resseq = int(record["resseq"])
            icode = str(record["icode"])
            key = (chain_id, resseq, icode, resname)
            if key not in residues:
                residues[key] = {
                    "resname": resname,
                    "chain_id": chain_id,
                    "resseq": resseq,
                    "icode": icode,
                    "ca": None,
                }
            item = residues[key]
            coord = np.asarray([float(record["x"]), float(record["y"]), float(record["z"])], dtype=np.float32)
            if atom_name == "CA":
                item["ca"] = coord
    parsed: List[Dict[str, object]] = []
    for item in tqdm(residues.values(), desc="finalize_residues", leave=False):
        if item["ca"] is None:
            continue
        pos = np.asarray(item["ca"], dtype=np.float32)
        parsed.append({
            "resname": item["resname"],
            "chain_id": item["chain_id"],
            "resseq": item["resseq"],
            "icode": item["icode"],
            "pos": pos,
        })
    return parsed

def build_radius_edges(
    pos: torch.Tensor,
    cutoff: Optional[float] = None,
    block_size: Optional[int] = None,
    segment: Optional[torch.Tensor] = None,
    bsa_positive_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    cutoff = 10.0 if cutoff is None else float(cutoff)
    block_size = 512 if block_size is None else max(1, int(block_size))
    n = pos.size(0)
    if n == 0:
        return torch.empty((2, 0), dtype=torch.long)
    if segment is not None:
        segment = segment.to(device=pos.device, dtype=torch.long)
    if bsa_positive_mask is not None:
        bsa_positive_mask = bsa_positive_mask.to(device=pos.device).bool()
    pos = pos.to(torch.float32)
    row_chunks = []
    col_chunks = []
    for start in tqdm(range(0, n, block_size), desc="radius_edges", leave=False):
        end = min(start + block_size, n)
        dist = torch.cdist(pos[start:end], pos)
        local_row, col = torch.nonzero(dist <= cutoff, as_tuple=True)
        row = local_row + start
        keep = row != col
        if keep.any():
            row_chunks.append(row[keep].to(torch.long))
            col_chunks.append(col[keep].to(torch.long))
    if not row_chunks:
        return torch.empty((2, 0), dtype=torch.long)
    edge_index = torch.stack([torch.cat(row_chunks), torch.cat(col_chunks)], dim=0)
    if segment is not None and bsa_positive_mask is not None:
        row, col = edge_index
        same_group = segment[row] == segment[col]
        allowed = same_group | ((~same_group) & bsa_positive_mask[row] & bsa_positive_mask[col])
        edge_index = edge_index[:, allowed]
    return edge_index.to(torch.long)

def build_cpu_graph_payload(
    pdb_path: str,
    cutoff: float,
    block_size: int,
    binding_pair: Tuple[Sequence[str], Sequence[str]],
) -> Dict[str, object]:
    residues = parse_pdb_residues(pdb_path)
    pdb_id = normalize_pdb_id(pdb_path)
    n = len(residues)
    right_chain_ids = normalize_chain_ids(binding_pair[1])
    right_chain_set = set(right_chain_ids)
    bsa_values = build_bsa_interface_values(pdb_path, residues, binding_pair)
    coords = np.empty((n, 3), dtype=np.float32)
    segment_list: List[int] = []
    residue_stubs: List[Dict[str, object]] = []
    for idx, residue in enumerate(tqdm(residues, desc=f"node_arrays:{pdb_id}", leave=False)):
        chain = str(residue["chain_id"])
        resseq = int(residue["resseq"])
        icode = str(residue["icode"])
        rn = str(residue["resname"]).upper()
        coords[idx] = np.asarray(residue["pos"], dtype=np.float32)
        segment_list.append(1 if chain in right_chain_set else 0)
        residue_stubs.append({
            "chain_id": chain,
            "resname": rn,
            "resseq": resseq,
            "icode": icode,
        })
    pos = torch.from_numpy(coords) if n > 0 else torch.zeros((0, 3), dtype=torch.float32)
    segment = torch.as_tensor(segment_list, dtype=torch.long)
    bsa_tensor = torch.as_tensor(bsa_values, dtype=torch.float32)
    if bsa_tensor.numel() != n:
        raise ValueError(f"bsa_values_len={bsa_tensor.numel()} num_nodes={n}")
    bsa_positive_mask = bsa_tensor > 0.0
    edge_index = build_radius_edges(pos, cutoff=cutoff, block_size=block_size, segment=segment, bsa_positive_mask=bsa_positive_mask)
    graph = Data(x=torch.zeros((n, 0), dtype=torch.float32), pos=pos, edge_index=edge_index)
    graph.bsa_values = bsa_tensor
    graph.interface_mask = bsa_positive_mask
    return {
        "pdb_id": pdb_id,
        "pdb_path": pdb_path,
        "graph": graph,
        "cutoff": float(cutoff),
        "residues": residue_stubs,
        "bsa_values": bsa_values,
        "interface_mask": bsa_positive_mask,
    }

def add_esmc_to_payload(
    payload: Dict[str, object],
    fasta_dir: str,
    esmc_device: str,
) -> Dict[str, object]:
    graph = payload["graph"]
    pdb_path = str(payload["pdb_path"])
    residues = list(payload.get("residues", []))
    pdb_id = str(payload["pdb_id"])
    esmc_dim = 0
    esmc_feats: Optional[np.ndarray] = None
    if residues:
        esmc_map, esmc_dim = _compute_esmc_residue_map(
            pdb_path=pdb_path,
            residues=residues,
            device=esmc_device,
            fasta_dir=fasta_dir,
        )
        if esmc_dim > 0:
            esmc_feats = np.zeros((len(residues), esmc_dim), dtype=np.float32)
        for idx, residue in enumerate(tqdm(residues, desc=f"attach_esmc:{pdb_id}", leave=False)):
            key = (str(residue["chain_id"]), int(residue["resseq"]), str(residue["icode"]))
            key_no_icode = (str(residue["chain_id"]), int(residue["resseq"]), "")
            esmc_feat = esmc_map.get(key, esmc_map.get(key_no_icode))
            if esmc_feats is not None and esmc_feat is not None:
                esmc_feats[idx] = np.asarray(esmc_feat, dtype=np.float32)
    if esmc_feats is not None and esmc_feats.size > 0:
        graph.x = torch.from_numpy(esmc_feats)
    payload["esmc_dim"] = int(esmc_dim)
    return payload

def build_cpu_payload_worker(
    pdb_path: str,
    cutoff: float,
    block_size: int,
    binding_pair_map: Dict[str, Tuple[List[str], List[str]]],
) -> Dict[str, object]:
    pdb_id = normalize_pdb_id(pdb_path)
    if pdb_id not in binding_pair_map:
        raise KeyError(f"missing binding pair: {pdb_id}")
    return build_cpu_graph_payload(
        pdb_path=pdb_path,
        cutoff=cutoff,
        block_size=block_size,
        binding_pair=binding_pair_map[pdb_id],
    )

def build_cpu_payloads(
    pdb_files: Sequence[str],
    binding_pair_map: Dict[str, Tuple[List[str], List[str]]],
    cutoff: float,
    block_size: int,
    workers: int,
) -> Tuple[List[Dict[str, object]], int]:
    payloads: List[Dict[str, object]] = []
    skipped = 0
    if workers <= 1:
        for pdb_path in tqdm(pdb_files, desc="Build residue CPU graphs"):
            pdb_id = normalize_pdb_id(pdb_path)
            try:
                payloads.append(build_cpu_payload_worker(pdb_path, cutoff, block_size, binding_pair_map))
            except Exception as exc:
                skipped += 1
                tqdm.write(f"[SKIP] {pdb_id}: {exc}")
        return payloads, skipped
    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        futures = {}
        for pdb_path in tqdm(pdb_files, desc="Submit residue CPU graphs", leave=False):
            futures[executor.submit(build_cpu_payload_worker, pdb_path, cutoff, block_size, binding_pair_map)] = pdb_path
        for future in tqdm(as_completed(futures), total=len(futures), desc="Build residue CPU graphs"):
            pdb_path = futures[future]
            pdb_id = normalize_pdb_id(pdb_path)
            try:
                payloads.append(future.result())
            except Exception as exc:
                skipped += 1
                tqdm.write(f"[SKIP] {pdb_id}: {exc}")
    return payloads, skipped

def save_final_payloads(
    payloads: Sequence[Dict[str, object]],
    output_dir: str,
    fasta_dir: str,
    esmc_device: str,
    dataset_name: str,
) -> Tuple[List[Dict[str, object]], int]:
    os.makedirs(output_dir, exist_ok=True)
    saved_results: List[Dict[str, object]] = []
    skipped = 0
    for payload in tqdm(payloads, desc="Attach ESMC and save"):
        pdb_id = str(payload["pdb_id"])
        try:
            payload = add_esmc_to_payload(payload, fasta_dir=fasta_dir, esmc_device=esmc_device)
            graph = payload["graph"]
            graph.split = dataset_name
            graph.dataset_name = dataset_name
            esmc_dim = int(payload.get("esmc_dim", 0))
            metadata = build_graph_metadata(
                pdb_id=pdb_id,
                num_nodes=int(graph.x.size(0)),
                num_edges=int(graph.edge_index.size(1)),
                node_dim=int(graph.x.size(1)) if graph.x.ndim == 2 else 0,
                edge_dim=0,
                cutoff=float(payload.get("cutoff", 10.0)),
                esmc_dim=esmc_dim,
                bsa_values=payload.get("bsa_values"),
                interface_mask=payload.get("interface_mask"),
            )
            out_path = os.path.join(output_dir, f"{pdb_id}.pt")
            torch.save(build_graph_payload(graph, metadata), out_path)
            saved_results.append({
                "pdb_id": pdb_id,
                "num_nodes": int(metadata["num_nodes"]),
                "num_edges": int(metadata["num_edges"]),
                "esmc_dim": esmc_dim,
                "out_path": out_path,
            })
        except Exception as exc:
            skipped += 1
            tqdm.write(f"[SKIP] {pdb_id}: {exc}")
    return saved_results, skipped

def process_dataset(config: Dict[str, str], esmc_device: str, workers: int) -> Dict[str, object]:
    dataset_name = str(config["dataset_name"])
    pdb_dir = str(config["pdb_dir"])
    fasta_dir = str(config["fasta_dir"])
    csv_path = str(config["csv_path"])
    output_dir = str(config["output_dir"])
    graph_cutoff = 10.0
    radius_block_size = 512
    if not os.path.isdir(pdb_dir):
        raise SystemExit(f"missing_pdb_dir: {pdb_dir}")
    if not os.path.isdir(fasta_dir):
        raise SystemExit(f"missing_fasta_dir: {fasta_dir}")
    if not os.path.isfile(csv_path):
        raise SystemExit(f"missing_csv_path: {csv_path}")
    os.makedirs(output_dir, exist_ok=True)
    pdb_files, binding_pair_map = prepare_pdb_batch(pdb_dir, csv_path)
    if not pdb_files:
        print(f"{dataset_name}: no eligible PDB files found")
        return {"dataset_name": dataset_name, "saved": 0, "skipped": 0, "output_dir": output_dir}
    cpu_payloads, cpu_skipped = build_cpu_payloads(
        pdb_files=pdb_files,
        binding_pair_map=binding_pair_map,
        cutoff=graph_cutoff,
        block_size=radius_block_size,
        workers=workers,
    )
    saved_results, esmc_skipped = save_final_payloads(
        payloads=cpu_payloads,
        output_dir=output_dir,
        fasta_dir=fasta_dir,
        esmc_device=esmc_device,
        dataset_name=dataset_name,
    )
    saved_ids = [str(result["pdb_id"]) for result in tqdm(saved_results, desc="saved_ids", leave=False)]
    expected_ids = [normalize_pdb_id(pdb_path) for pdb_path in tqdm(pdb_files, desc="expected_ids", leave=False)]
    missing_ids = sorted(set(expected_ids) - set(saved_ids))
    if missing_ids:
        tqdm.write(f"[WARN] missing_saved_graphs={len(missing_ids)} first={missing_ids[:20]}")
    saved = len(saved_results)
    skipped = int(cpu_skipped + esmc_skipped)
    esmc_dim = int(saved_results[0].get("esmc_dim", 0)) if saved_results else 0
    print(f"{dataset_name}: saved={saved}, skipped={skipped}, feature_dim={esmc_dim}")
    return {"dataset_name": dataset_name, "saved": saved, "skipped": skipped, "feature_dim": esmc_dim, "output_dir": output_dir}

def main() -> None:
    args = parse_args()
    dataset_root = os.path.abspath(str(args.dataset_root))
    output_root = os.path.abspath(str(args.output_root))
    esmc_device = _normalize_esmc_device(str(args.esmc_device))
    workers = max(1, int(args.workers))
    dataset_names = [str(name) for name in args.datasets]
    invalid = [name for name in dataset_names if name not in DATASET_NAMES]
    if invalid:
        raise SystemExit(f"unknown_datasets: {invalid}")
    configs = build_dataset_configs(dataset_root, output_root, dataset_names)
    results = []
    for config in tqdm(configs, desc="Process residue datasets"):
        results.append(process_dataset(config, esmc_device=esmc_device, workers=workers))
    total_saved = sum(int(result["saved"]) for result in tqdm(results, desc="total_saved", leave=False))
    total_skipped = sum(int(result["skipped"]) for result in tqdm(results, desc="total_skipped", leave=False))
    print(f"total_saved: {total_saved}")
    print(f"total_skipped: {total_skipped}")
    print(f"output_root: {output_root}")

if __name__ == "__main__":
    main()
