import argparse
import csv
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import freesasa
from tqdm import tqdm

DATASET_ROOT = ""
DATASET_NAMES = ("")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_NAMES))
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()

def build_dataset_configs(dataset_root: str, dataset_names: Sequence[str]) -> List[Dict[str, str]]:
    configs: List[Dict[str, str]] = []
    for dataset_name in dataset_names:
        dataset_dir = os.path.join(dataset_root, dataset_name)
        configs.append({
            "dataset_name": dataset_name,
            "input_pqr_dir": os.path.join(dataset_dir, "PQR"),
            "input_csv_path": os.path.join(dataset_dir, f"{dataset_name}.csv"),
            "output_dir": os.path.join(dataset_dir, "Interface"),
        })
    return configs

def normalize_pdb_id(value: object) -> str:
    return Path(str(value).strip()).stem.upper()

def parse_chain_group(group: str) -> List[str]:
    chains = [ch for ch in str(group).strip() if ch.isalnum()]
    return list(dict.fromkeys(chains))

def parse_binding_chain_pair(raw: str) -> Tuple[List[str], List[str]]:
    for token in str(raw).split(","):
        token = token.strip()
        if ":" not in token:
            continue
        left, right = token.split(":", 1)
        left_ids = parse_chain_group(left)
        right_ids = parse_chain_group(right)
        if left_ids and right_ids:
            return left_ids, right_ids
    raise ValueError(f"Invalid BindingChainPair: {raw}")

def filter_pqr_lines(pqr_path: str, chain_ids: Sequence[str]) -> List[str]:
    keep = set(str(ch).strip() or "_" for ch in chain_ids)
    out: List[str] = []
    with open(pqr_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            chain = (line[21].strip() if len(line) > 21 else "") or "_"
            if chain in keep:
                out.append(line)
    return out

def write_lines(path: str, lines: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line)

def line_key(line: str) -> str:
    return line.rstrip("\n")

def accepted_and_index(lines: Sequence[str], structure: freesasa.Structure) -> List[str]:
    accepted: List[str] = []
    j = 0
    n = structure.nAtoms()

    def key_from_struct(idx: int) -> Tuple[str, str, str, str, float, float, float]:
        coord = structure.coord(idx)
        return (
            structure.atomName(idx).strip(),
            structure.residueName(idx).strip(),
            structure.chainLabel(idx).strip(),
            str(structure.residueNumber(idx)).strip(),
            round(coord[0], 3),
            round(coord[1], 3),
            round(coord[2], 3),
        )

    for line in lines:
        if j >= n:
            break
        key = (
            line[12:16].strip(),
            line[17:20].strip(),
            line[21].strip(),
            (line[22:26] + line[26]).strip(),
            round(float(line[30:38]), 3),
            round(float(line[38:46]), 3),
            round(float(line[46:54]), 3),
        )
        if key == key_from_struct(j):
            accepted.append(line)
            j += 1
    return accepted

def worker(task: Tuple[str, str, str, str]) -> Dict[str, object]:
    pdb_id, binding_pair_raw, pqr_path, out_dir = task
    freesasa.setVerbosity(freesasa.nowarnings)
    atom_pqr_out_path = os.path.join(out_dir, f"{pdb_id}.pqr")
    try:
        group1, group2 = parse_binding_chain_pair(binding_pair_raw)
    except Exception:
        return {"PDB": pdb_id, "Status": "bad_binding_chain_pair"}
    if not os.path.exists(pqr_path):
        return {"PDB": pdb_id, "Status": "missing_pqr"}
    lines1 = filter_pqr_lines(pqr_path, group1)
    lines2 = filter_pqr_lines(pqr_path, group2)
    if not lines1 or not lines2:
        return {"PDB": pdb_id, "Status": "empty_selected_atoms"}
    lines_complex = lines1 + lines2
    with tempfile.TemporaryDirectory() as td:
        g1_path = os.path.join(td, "group1.pqr")
        g2_path = os.path.join(td, "group2.pqr")
        cx_path = os.path.join(td, "complex.pqr")
        write_lines(g1_path, lines1)
        write_lines(g2_path, lines2)
        write_lines(cx_path, lines_complex)
        s1 = freesasa.Structure(g1_path, options={"hetatm": False, "skip-unknown": False})
        s2 = freesasa.Structure(g2_path, options={"hetatm": False, "skip-unknown": False})
        sc = freesasa.Structure(cx_path, options={"hetatm": False, "skip-unknown": False})
        accepted1 = accepted_and_index(lines1, s1)
        accepted2 = accepted_and_index(lines2, s2)
        accepted_complex = accepted_and_index(lines_complex, sc)
        if len(accepted1) != s1.nAtoms() or len(accepted2) != s2.nAtoms() or len(accepted_complex) != sc.nAtoms():
            return {"PDB": pdb_id, "Status": "matching_failed"}
        r1 = freesasa.calc(s1)
        r2 = freesasa.calc(s2)
        rc = freesasa.calc(sc)
    complex_index = {line_key(line): idx for idx, line in enumerate(accepted_complex)}
    atom_lines: List[str] = []
    delta_sum = 0.0
    contributing_group1 = 0
    contributing_group2 = 0
    for i, line in enumerate(accepted1):
        cidx = complex_index[line_key(line)]
        delta = float(r1.atomArea(i) - rc.atomArea(cidx))
        if delta > 1e-6:
            atom_lines.append(line if line.endswith("\n") else f"{line}\n")
            delta_sum += delta
            contributing_group1 += 1
    for i, line in enumerate(accepted2):
        cidx = complex_index[line_key(line)]
        delta = float(r2.atomArea(i) - rc.atomArea(cidx))
        if delta > 1e-6:
            atom_lines.append(line if line.endswith("\n") else f"{line}\n")
            delta_sum += delta
            contributing_group2 += 1
    bsa = delta_sum / 2.0
    if not (600.0 <= bsa <= 2000.0):
        if os.path.exists(atom_pqr_out_path):
            os.remove(atom_pqr_out_path)
        return {
            "PDB": pdb_id,
            "Status": "bsa_out_of_range",
            "BSA": bsa,
            "ContributingAtomsTotal": contributing_group1 + contributing_group2,
        }
    write_lines(atom_pqr_out_path, atom_lines)
    return {
        "PDB": pdb_id,
        "Status": "ok",
        "BSA": bsa,
        "ContributingAtomsTotal": contributing_group1 + contributing_group2,
    }

def main() -> None:
    args = parse_args()
    dataset_configs = build_dataset_configs(args.dataset_root, args.datasets)
    overall_results: List[Dict[str, object]] = []
    for config in tqdm(dataset_configs, desc="dataset_configs", unit="dataset"):
        dataset_name = str(config["dataset_name"])
        pqr_dir = os.path.abspath(str(config["input_pqr_dir"]))
        csv_path = os.path.abspath(str(config["input_csv_path"]))
        out_dir = os.path.abspath(str(config["output_dir"]))
        if not os.path.isdir(pqr_dir):
            raise SystemExit(f"missing_pqr_dir: {pqr_dir}")
        if not os.path.isfile(csv_path):
            raise SystemExit(f"missing_csv_path: {csv_path}")
        not_saved_txt_path = os.path.join(out_dir, "not_saved_pdb_list.txt")
        os.makedirs(out_dir, exist_ok=True)
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        tasks: List[Tuple[str, str, str, str]] = []
        for row in tqdm(rows, desc=f"build_tasks:{dataset_name}"):
            pdb_id = normalize_pdb_id(row.get("PDB", ""))
            binding_pair = str(row.get("BindingChainPair", "")).strip()
            if not pdb_id or not binding_pair:
                continue
            pqr_path = os.path.join(pqr_dir, f"{pdb_id}.pqr")
            tasks.append((pdb_id, binding_pair, pqr_path, out_dir))
        results: List[Dict[str, object]] = []
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            futures = [executor.submit(worker, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"freesasa_interface:{dataset_name}"):
                results.append(future.result())
        ok_results = [row for row in results if row.get("Status") == "ok"]
        fail_results = [row for row in results if row.get("Status") != "ok"]
        with open(not_saved_txt_path, "w", encoding="utf-8") as handle:
            for row in tqdm(sorted(fail_results, key=lambda x: str(x.get("PDB", ""))), desc=f"write_not_saved_txt:{dataset_name}"):
                handle.write(f"{row.get('PDB', '')}\t{row.get('Status', '')}\n")
        overall_results.extend(results)
        print(f"dataset: {dataset_name}")
        print(f"input_pqr_dir: {pqr_dir}")
        print(f"input_csv_path: {csv_path}")
        print(f"out_dir: {out_dir}")
        print(f"not_saved_txt_path: {not_saved_txt_path}")
        print(f"tasks: {len(tasks)}")
        print(f"success: {len(ok_results)}")
        print(f"failed: {len(fail_results)}")
        if ok_results:
            bsa_values = [float(row["BSA"]) for row in ok_results]
            atom_values = [int(row["ContributingAtomsTotal"]) for row in ok_results]
            print(f"bsa_mean: {sum(bsa_values) / len(bsa_values)}")
            print(f"bsa_min: {min(bsa_values)}")
            print(f"bsa_max: {max(bsa_values)}")
            print(f"contributing_atoms_mean: {sum(atom_values) / len(atom_values)}")
    print(f"datasets: {', '.join(str(config['dataset_name']) for config in dataset_configs)}")
    print(f"total_results: {len(overall_results)}")

if __name__ == "__main__":
    main()
