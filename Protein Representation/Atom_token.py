import csv
import os
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from tqdm import tqdm
from Atom_APBS import parse_dx, run_apbs, sample_dx_at_coords, scale_apbs_numpy, stack_entry_coords, write_apbs_pqr
from Atom_CircularVariance import get_circular_variance_dim, circular_variance_features
from Atom_crosscontacts import get_crosscontacts_dim, cross_contact_features
from Atom_onehot import AA20, ELEMENTS, SCALAR_DIM, build_base_features, build_entries_from_pqr_text
from Atom_SASA import (
    build_delta_sasa_from_entries,
    build_sasa_maps_from_entries,
    coord_round_tuple,
    entry_atom_identity,
    lookup_sasa_value,
    scale_delta_sasa_numpy,
    scale_sasa_numpy,
)
from Atom_type import build_atom_type_features

PROJECT_ROOT = ""
DATASET_ROOT = f"{PROJECT_ROOT}/Datasets"
TOKEN_ROOT = f"{PROJECT_ROOT}/Representation/Atom_Token"
RUNTIME_ROOT = f"{TOKEN_ROOT}/_runtime"
APBS_WORK_ROOT = f"{RUNTIME_ROOT}/apbs"
FREESASA_WORK_ROOT = f"{RUNTIME_ROOT}/freesasa"
OPENBABEL_TMP_ROOT = f"{RUNTIME_ROOT}/openbabel"
APBS_EXECUTABLE = "apbs"

def make_dataset_config(name: str, output_name: Optional[str] = None) -> Dict[str, str]:
    resolved_output_name = output_name if output_name is not None else name
    return {
        "NAME": name,
        "INPUT_PQR_DIR": f"{DATASET_ROOT}/{name}/Interface",
        "FULL_PQR_DIR": f"{DATASET_ROOT}/{name}/PQR",
        "INPUT_CSV_PATH": f"{DATASET_ROOT}/{name}/{name}.csv",
        "OUTPUT_DIR": f"{TOKEN_ROOT}/{resolved_output_name}",
    }

DATASET_CONFIGS = [
    make_dataset_config("Test1"),
    make_dataset_config("Test2"),
    make_dataset_config("Test3"),
    make_dataset_config("Test4"),
    make_dataset_config("Test5"),
    make_dataset_config("Train"),
]

def resolve_feature_layout() -> Tuple[int, Dict[str, int], Dict[str, Optional[Tuple[int, int]]], int]:
    raw_dims = {
        "element": len(ELEMENTS),
        "residue": len(AA20),
        "scalar": SCALAR_DIM,
        "cv": get_circular_variance_dim(),
        "atom_type": 8,
        "cross_contacts": get_crosscontacts_dim(),
    }
    offset = 0
    feature_dims: Dict[str, int] = {}
    feature_slices: Dict[str, Optional[Tuple[int, int]]] = {}
    for name in ("element", "residue", "scalar", "cv", "atom_type", "cross_contacts"):
        dim = raw_dims[name]
        feature_dims[name] = int(dim)
        if dim > 0:
            feature_slices[name] = (int(offset), int(offset + dim))
            offset += dim
        else:
            feature_slices[name] = None
    selected_base_dim = int(feature_dims["element"] + feature_dims["residue"] + feature_dims["scalar"])
    return int(offset), feature_dims, feature_slices, selected_base_dim

def build_sasa_feature(
    pdb_id: str,
    ordered_entries: Sequence[Dict[str, object]],
    freesasa_tmp_root: str,
    sasa_maps: Tuple[Dict[object, float], Dict[object, List[float]]],
) -> Dict[str, object]:
    exact_map, identity_map = sasa_maps
    interface_sasa_raw = np.empty((len(ordered_entries), 1), dtype=np.float32)
    missing_atoms = 0
    with tqdm(total=len(ordered_entries), desc=f"sasa:{pdb_id}", leave=False) as pbar:
        pbar.set_postfix_str("lookup")
        for idx, entry in enumerate(ordered_entries):
            sasa_value = lookup_sasa_value(entry, exact_map, identity_map)
            if sasa_value is None:
                missing_atoms += 1
            else:
                interface_sasa_raw[idx, 0] = float(sasa_value)
            pbar.update(1)
    if missing_atoms > 0:
        raise RuntimeError(f"{pdb_id}: missing sasa mapping: {missing_atoms}")
    interface_sasa_x = scale_sasa_numpy(interface_sasa_raw)
    return {
        "x": torch.from_numpy(interface_sasa_x).to(torch.float32),
    }

def build_delta_sasa_feature(
    pdb_id: str,
    full_entries: Sequence[Dict[str, object]],
    binding_pair: Tuple[List[str], List[str]],
    ordered_entries: Sequence[Dict[str, object]],
    freesasa_tmp_root: str,
    bound_sasa_maps: Optional[Tuple[Dict[object, float], Dict[object, List[float]]]] = None,
) -> Dict[str, object]:
    chain_to_side: Dict[str, int] = {}
    for chain_id in binding_pair[0]:
        chain_to_side[str(chain_id)] = 0
    for chain_id in binding_pair[1]:
        chain_to_side.setdefault(str(chain_id), 1)
    side_context_entries = [entry for entry in full_entries if str(entry["chain_id"]) in chain_to_side]
    def side_of_entry(entry: Dict[str, object]) -> int:
        return chain_to_side[str(entry["chain_id"])]
    reuse_bound_maps = bound_sasa_maps if (bound_sasa_maps is not None and len(side_context_entries) == len(full_entries)) else None
    _, _, dsasa_raw = build_delta_sasa_from_entries(
        side_context_entries,
        side_of_entry,
        f"{pdb_id}_delta_sasa",
        freesasa_tmp_root,
        bound_maps=reuse_bound_maps,
    )
    dsasa_exact: Dict[Tuple[str, str, str, str, float, float, float], float] = {}
    dsasa_identity: Dict[Tuple[str, str, str, str], List[float]] = {}
    for idx, entry in enumerate(side_context_entries):
        value = float(dsasa_raw[idx, 0])
        identity_key = entry_atom_identity(entry)
        dsasa_exact.setdefault(identity_key + coord_round_tuple(entry["coord"]), value)
        dsasa_identity.setdefault(identity_key, []).append(value)
    interface_dsasa_raw = np.zeros((len(ordered_entries), 1), dtype=np.float32)
    missing_atoms = 0
    with tqdm(total=len(ordered_entries), desc=f"dsasa:{pdb_id}", leave=False) as pbar:
        pbar.set_postfix_str("lookup")
        for idx, entry in enumerate(ordered_entries):
            dsasa_value = lookup_sasa_value(entry, dsasa_exact, dsasa_identity)
            if dsasa_value is None:
                missing_atoms += 1
            else:
                interface_dsasa_raw[idx, 0] = float(dsasa_value)
            pbar.update(1)
    if missing_atoms > 0:
        warnings.warn(
            f"{pdb_id}: missing dsasa mapping for {missing_atoms}/{len(ordered_entries)} "
            f"interface atoms; set to 0.",
            RuntimeWarning,
            stacklevel=2,
        )
    interface_dsasa_x = scale_delta_sasa_numpy(interface_dsasa_raw)
    return {
        "x": torch.from_numpy(interface_dsasa_x).to(torch.float32),
    }

def build_apbs_feature(
    pdb_id: str,
    full_entries: Sequence[Dict[str, object]],
    ordered_entries: Sequence[Dict[str, object]],
    apbs_exe: str,
    apbs_work_root: str,
) -> Dict[str, object]:
    with tqdm(total=5, desc=f"apbs:{pdb_id}", leave=False) as pbar:
        pbar.set_postfix_str("full_coords")
        full_coords = stack_entry_coords(full_entries, f"ordered_full:{pdb_id}")
        pbar.update(1)
        pbar.set_postfix_str("interface_coords")
        interface_coords = stack_entry_coords(ordered_entries, f"ordered_interface:{pdb_id}")
        pbar.update(1)
        os.makedirs(apbs_work_root, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"apbs_{pdb_id}_", dir=apbs_work_root) as work_dir:
            apbs_pqr_path = os.path.join(work_dir, f"{pdb_id}.pqr")
            write_apbs_pqr(full_entries, apbs_pqr_path)
            pbar.set_postfix_str("run_apbs")
            dx_path = run_apbs(apbs_pqr_path, full_coords, apbs_exe, work_dir, pdb_id)
            pbar.update(1)
            pbar.set_postfix_str("parse_dx")
            dx = parse_dx(dx_path)
            pbar.update(1)
            pbar.set_postfix_str("sample")
            phi_raw = sample_dx_at_coords(dx, interface_coords)
            pbar.update(1)
    phi_raw_2d = phi_raw[:, None]
    phi_scaled_2d = scale_apbs_numpy(phi_raw_2d)
    return {
        "x": torch.from_numpy(phi_scaled_2d).to(torch.float32),
    }

def normalize_pdb_id(value: object) -> str:
    return Path(str(value).strip()).stem.upper()

def parse_chain_group(group: str) -> List[str]:
    chains = [ch for ch in str(group).strip() if ch.isalnum()]
    return list(dict.fromkeys(chains))

def parse_binding_chain_pair(raw: str) -> Tuple[List[str], List[str]]:
    first_pair = next((pair_chunk.strip() for pair_chunk in str(raw).split(",") if ":" in pair_chunk), "")
    if not first_pair:
        raise ValueError(f"Missing binding chain pair delimiter: {raw}")
    left, right = first_pair.split(":", 1)
    return parse_chain_group(left), parse_chain_group(right)

def load_csv_pdb_entries(csv_path: str) -> Tuple[List[str], Dict[str, Tuple[List[str], List[str]]]]:
    pdb_ids: List[str] = []
    seen = set()
    binding_pair_map: Dict[str, Tuple[List[str], List[str]]] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        for row in tqdm(csv.DictReader(handle), desc=f"csv_entries:{os.path.basename(csv_path)}", leave=False):
            pdb_id = normalize_pdb_id(row.get("PDB", ""))
            if not pdb_id:
                continue
            if pdb_id not in seen:
                pdb_ids.append(pdb_id)
                seen.add(pdb_id)
            raw_pair = row.get("BindingChainPair", "")
            if str(raw_pair).strip():
                binding_pair_map[pdb_id] = parse_binding_chain_pair(raw_pair)
    return pdb_ids, binding_pair_map

def mapped_pqr_path(pdb_id: str, pqr_dir: str) -> str:
    return os.path.join(pqr_dir, f"{normalize_pdb_id(pdb_id)}.pqr")

def select_binding_group_entries(
    entries: Sequence[Dict[str, object]],
    binding_pair: Tuple[List[str], List[str]],
) -> Tuple[List[str], List[str], List[Dict[str, object]], List[Dict[str, object]]]:
    requested_left = set(binding_pair[0])
    requested_right = set(binding_pair[1])
    available_left = set()
    available_right = set()
    side_left_entries = []
    side_right_entries = []
    for entry in tqdm(entries, desc="select_binding_groups", leave=False):
        chain_id = str(entry["chain_id"])
        if chain_id in requested_left:
            available_left.add(chain_id)
            side_left_entries.append(entry)
        if chain_id in requested_right:
            available_right.add(chain_id)
            side_right_entries.append(entry)
    used_left = [chain_id for chain_id in binding_pair[0] if chain_id in available_left]
    used_right = [chain_id for chain_id in binding_pair[1] if chain_id in available_right]
    if not used_left or not used_right:
        raise ValueError(f"Missing binding groups in PQR: {binding_pair[0]} vs {binding_pair[1]}")
    if not side_left_entries or not side_right_entries:
        raise ValueError(f"Missing binding groups in PQR: {binding_pair[0]} vs {binding_pair[1]}")
    return used_left, used_right, side_left_entries, side_right_entries

def build_merged_features(
    side_a_entries: Sequence[Dict[str, object]],
    side_b_entries: Sequence[Dict[str, object]],
    context_entries: Optional[Sequence[Dict[str, object]]] = None,
    atom_type_tmp_root: str = OPENBABEL_TMP_ROOT,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    num_a = len(side_a_entries)
    num_b = len(side_b_entries)
    ordered_entries = list(side_a_entries) + list(side_b_entries)
    joint_context = context_entries if context_entries is not None else ordered_entries
    feat_dim, feature_dims, _, _ = resolve_feature_layout()
    if num_a == 0 or num_b == 0:
        return (
            torch.zeros((0, feat_dim), dtype=torch.float32),
            torch.zeros((0, 3), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.long),
            torch.zeros((0,), dtype=torch.long),
            0,
            0,
        )
    base_x, pos, elem = build_base_features(ordered_entries)
    feature_parts: List[torch.Tensor] = []
    base_offset = 0
    if feature_dims["element"] > 0:
        feature_parts.append(base_x[:, base_offset:base_offset + len(ELEMENTS)])
    base_offset += len(ELEMENTS)
    if feature_dims["residue"] > 0:
        feature_parts.append(base_x[:, base_offset:base_offset + len(AA20)])
    base_offset += len(AA20)
    if feature_dims["scalar"] > 0:
        feature_parts.append(base_x[:, base_offset:base_offset + SCALAR_DIM])
    cv = circular_variance_features(joint_context, ordered_entries)
    feature_parts.append(torch.from_numpy(cv).to(dtype=torch.float32))
    atom_type = build_atom_type_features(joint_context, ordered_entries, "merged_atom_type", atom_type_tmp_root)
    feature_parts.append(torch.from_numpy(atom_type).to(dtype=torch.float32))
    cross = cross_contact_features(ordered_entries, num_a, num_b)
    feature_parts.append(torch.from_numpy(cross).to(dtype=torch.float32))
    x = torch.cat(feature_parts, dim=1)
    segment = torch.as_tensor(([0] * num_a) + ([1] * num_b), dtype=torch.long)
    return x, pos, elem, segment, int(num_a), int(num_b)

def resolve_token_feature_layout(
    use_sasa_feature: bool = True,
    use_dsasa_feature: bool = True,
    use_apbs_feature: bool = True,
) -> Tuple[int, Dict[str, int], Dict[str, Optional[Tuple[int, int]]], int]:
    feature_dim, feature_dims, feature_slices, selected_base_dim = resolve_feature_layout()
    feature_dims = dict(feature_dims)
    feature_slices = dict(feature_slices)
    offset = int(feature_dim)
    if use_sasa_feature:
        feature_dims["sasa"] = 1
        feature_slices["sasa"] = (offset, offset + 1)
        offset += 1
    else:
        feature_dims["sasa"] = 0
        feature_slices["sasa"] = None
    if use_dsasa_feature:
        feature_dims["dsasa"] = 1
        feature_slices["dsasa"] = (offset, offset + 1)
        offset += 1
    else:
        feature_dims["dsasa"] = 0
        feature_slices["dsasa"] = None
    if use_apbs_feature:
        feature_dims["apbs"] = 1
        feature_slices["apbs"] = (offset, offset + 1)
        offset += 1
    else:
        feature_dims["apbs"] = 0
        feature_slices["apbs"] = None
    return offset, feature_dims, feature_slices, int(selected_base_dim)

def build_feature_metadata(
    feature_dim: int,
    feature_dims: Dict[str, int],
    feature_slices: Dict[str, Optional[Tuple[int, int]]],
    base_dim: int,
) -> Dict[str, object]:
    return {
        "feature_dim": int(feature_dim),
        "base_dim": int(base_dim),
        "feature_dims": {name: int(dim) for name, dim in feature_dims.items()},
        "feature_slices": dict(feature_slices),
    }

def assign_feature_layout(
    tokens: Dict[str, object],
    metadata: Optional[Dict[str, object]],
    x: torch.Tensor,
    feature_dims: Dict[str, int],
    feature_slices: Dict[str, Optional[Tuple[int, int]]],
    base_dim: int,
) -> None:
    feature_metadata = build_feature_metadata(int(x.size(1)), feature_dims, feature_slices, base_dim)
    tokens["x"] = x
    tokens.update(feature_metadata)
    if isinstance(metadata, dict):
        metadata.update(feature_metadata)

def build_token_payload(
    pdb_id: str,
    full_entries: Sequence[Dict[str, object]],
    selected_groups: Tuple[List[str], List[str], List[Dict[str, object]], List[Dict[str, object]]],
) -> Tuple[Dict[str, object], Dict[str, object]]:
    _, _, side_left_entries, side_right_entries = selected_groups
    if len(side_left_entries) == 0 or len(side_right_entries) == 0:
        raise RuntimeError(f"{pdb_id}: empty binding group atoms")
    context_entries = list(full_entries)
    atom_x, atom_pos, atom_elem, atom_segment, _, _ = build_merged_features(
        side_left_entries,
        side_right_entries,
        context_entries,
    )
    if atom_x.size(0) == 0:
        raise RuntimeError(f"{pdb_id}: empty chain features")
    atom_to_token = torch.arange(atom_x.size(0), dtype=torch.long)
    token_x = atom_x
    token_pos = atom_pos
    token_elem = atom_elem
    token_segment = atom_segment
    token_atom_count = torch.ones(atom_x.size(0), dtype=torch.long)
    raw_feature_dim, raw_feature_dims, raw_feature_slices, selected_base_dim = resolve_token_feature_layout(
        use_sasa_feature=False,
        use_dsasa_feature=False,
        use_apbs_feature=False,
    )
    feature_metadata = build_feature_metadata(raw_feature_dim, raw_feature_dims, raw_feature_slices, selected_base_dim)
    tokens = {
        "x": token_x,
        "pos": token_pos,
        "elem": token_elem,
        "segment": token_segment,
        "token_atom_count": token_atom_count,
        "atom_to_token": atom_to_token,
        "pdb_id": pdb_id,
        "num_tokens": int(token_x.size(0)),
        **feature_metadata,
    }
    metadata = {
        "pdb_id": pdb_id,
        "num_tokens": int(token_x.size(0)),
        **feature_metadata,
    }
    return tokens, metadata

def process_complex(task: Tuple[str, str, Tuple[List[str], List[str]], str, str, Optional[str], str, str]) -> Dict[str, object]:
    pdb_id, interface_pqr_path, binding_pair, full_pqr_path, output_dir, apbs_exe, freesasa_tmp_root, apbs_work_root = task
    interface_entries = build_entries_from_pqr_text(interface_pqr_path)
    full_entries = build_entries_from_pqr_text(full_pqr_path)
    selected_groups = select_binding_group_entries(interface_entries, binding_pair)
    _, _, side_left_entries, side_right_entries = selected_groups
    out, metadata = build_token_payload(
        pdb_id,
        full_entries=full_entries,
        selected_groups=selected_groups,
    )
    ordered_entries = list(side_left_entries) + list(side_right_entries)
    bound_exact_map, bound_identity_map, _ = build_sasa_maps_from_entries(full_entries, f"{pdb_id}_full_sasa", freesasa_tmp_root)
    bound_sasa_maps = (bound_exact_map, bound_identity_map)
    sasa_feature = build_sasa_feature(
        pdb_id,
        ordered_entries,
        freesasa_tmp_root,
        sasa_maps=bound_sasa_maps,
    )
    dsasa_feature = build_delta_sasa_feature(
        pdb_id,
        full_entries,
        binding_pair,
        ordered_entries,
        freesasa_tmp_root,
        bound_sasa_maps=bound_sasa_maps,
    )
    if not apbs_exe:
        raise RuntimeError(f"{pdb_id}: missing apbs executable")
    apbs_feature = build_apbs_feature(
        pdb_id,
        full_entries,
        ordered_entries,
        apbs_exe,
        apbs_work_root,
    )
    x_parts = [
        out["x"],
        sasa_feature["x"].to(dtype=out["x"].dtype),
        dsasa_feature["x"].to(dtype=out["x"].dtype),
        apbs_feature["x"].to(dtype=out["x"].dtype),
    ]
    x = torch.cat(x_parts, dim=1)
    _, feature_dims, feature_slices, selected_base_dim = resolve_token_feature_layout()
    assign_feature_layout(out, metadata, x, feature_dims, feature_slices, selected_base_dim)
    output_path = os.path.join(output_dir, f"{pdb_id}.pt")
    torch.save({"tokens": out, "metadata": metadata}, output_path)
    return {
        "pdb_id": pdb_id,
        "output_path": output_path,
    }

def prepare_tasks(input_pqr_dir: str, input_csv_path: str, full_pqr_dir: str) -> List[Tuple[str, str, Tuple[List[str], List[str]], str]]:
    csv_pdb_ids, binding_pair_map = load_csv_pdb_entries(input_csv_path)
    tasks: List[Tuple[str, str, Tuple[List[str], List[str]], str]] = []
    for pdb_id in tqdm(csv_pdb_ids, desc=f"Prepare tasks:{Path(input_pqr_dir).name}", leave=False):
        interface_pqr_path = mapped_pqr_path(pdb_id, input_pqr_dir)
        full_pqr_path = mapped_pqr_path(pdb_id, full_pqr_dir)
        binding_pair = binding_pair_map.get(pdb_id)
        if binding_pair is None:
            raise RuntimeError(f"{pdb_id}: missing BindingChainPair in {input_csv_path}")
        if not os.path.isfile(interface_pqr_path):
            raise FileNotFoundError(f"{pdb_id}: missing interface PQR: {interface_pqr_path}")
        if not os.path.isfile(full_pqr_path):
            raise FileNotFoundError(f"{pdb_id}: missing full PQR: {full_pqr_path}")
        tasks.append((pdb_id, interface_pqr_path, binding_pair, full_pqr_path))
    return tasks

def process_task_batch(
    prepared_tasks: Sequence[Tuple[str, str, Tuple[List[str], List[str]], str]],
    output_dir: str,
    desc: str,
    apbs_exe: Optional[str],
    freesasa_tmp_root: str,
    apbs_work_root: str,
) -> List[Dict[str, object]]:
    saved_results: List[Dict[str, object]] = []
    os.makedirs(output_dir, exist_ok=True)
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = []
        for pdb_id, interface_pqr_path, chain_pair, full_pqr_path in tqdm(prepared_tasks, desc=f"Submit {desc}", unit="complex", leave=False):
            futures.append(executor.submit(process_complex, (pdb_id, interface_pqr_path, chain_pair, full_pqr_path, output_dir, apbs_exe, freesasa_tmp_root, apbs_work_root)))
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="complex"):
            result = future.result()
            saved_results.append(result)
    return saved_results

def main() -> None:
    os.makedirs(TOKEN_ROOT, exist_ok=True)
    for runtime_dir in tqdm((RUNTIME_ROOT, FREESASA_WORK_ROOT, OPENBABEL_TMP_ROOT, APBS_WORK_ROOT), desc="Ensure atom token runtime dirs", leave=False):
        os.makedirs(runtime_dir, exist_ok=True)
    for dataset_config in tqdm(DATASET_CONFIGS, desc="Ensure atom token output dirs", leave=False):
        os.makedirs(dataset_config["OUTPUT_DIR"], exist_ok=True)
    if not os.path.isfile(APBS_EXECUTABLE) or not os.access(APBS_EXECUTABLE, os.X_OK):
        raise FileNotFoundError(f"APBS executable not configured or not executable: {APBS_EXECUTABLE}")
    apbs_exe = APBS_EXECUTABLE
    dataset_configs = DATASET_CONFIGS
    processed_count = 0
    for dataset_idx, dataset_config in enumerate(tqdm(dataset_configs, desc="Build atom tokens"), start=1):
        dataset_name = str(dataset_config.get("NAME", f"dataset_{dataset_idx}"))
        dataset_tasks = prepare_tasks(dataset_config["INPUT_PQR_DIR"], dataset_config["INPUT_CSV_PATH"], dataset_config["FULL_PQR_DIR"])
        if not dataset_tasks:
            raise RuntimeError(f"{dataset_name}: no eligible PQR files found")
        dataset_saved_results = process_task_batch(
            dataset_tasks,
            dataset_config["OUTPUT_DIR"],
            f"Building {dataset_name} atom tokens",
            apbs_exe,
            FREESASA_WORK_ROOT,
            APBS_WORK_ROOT,
        )
        processed_count += len(dataset_saved_results)
    if processed_count == 0:
        raise RuntimeError("No eligible PQR files found")
    feature_dim, feature_dims, _, _ = resolve_token_feature_layout()
    saved_features = [f"{name}({int(dim)})" for name, dim in feature_dims.items() if int(dim) > 0]
    print(f"saved_features: {', '.join(saved_features)}")
    print(f"feature_dim: {int(feature_dim)}")

if __name__ == "__main__":
    main()
