from __future__ import annotations
import argparse
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from openmm.app import PDBFile
from pdbfixer import PDBFixer
from tqdm import tqdm

DATASET_ROOT = ""
DATASET_NAMES = ("")

def normalize_pdb_id(value: object) -> str:
    return Path(str(value).strip()).stem.upper()

def is_hydrogen_record(line: str) -> bool:
    if not line.startswith(("ATOM", "HETATM")):
        return False
    element = line[76:78].strip().upper()
    if element:
        return element == "H"
    atom_name = line[12:16].strip().upper().lstrip("0123456789")
    return atom_name.startswith("H")

def strip_hydrogens_inplace(pdb_path: Path) -> None:
    lines = pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    with pdb_path.open("w", encoding="utf-8") as handle:
        for line in lines:
            if is_hydrogen_record(line):
                continue
            handle.write(line)
            handle.write("\n")

def build_dataset_configs(dataset_root: str, dataset_names: Sequence[str]) -> List[Dict[str, str]]:
    configs: List[Dict[str, str]] = []
    for dataset_name in dataset_names:
        dataset_dir = os.path.join(dataset_root, dataset_name)
        configs.append({
            "dataset_name": dataset_name,
            "input_pdb_dir": os.path.join(dataset_dir, "PDB"),
            "output_pdbfixer_dir": os.path.join(dataset_dir, "PDBFIXER"),
        })
    return configs

def collect_tasks(dataset_configs: Sequence[Dict[str, str]], overwrite: bool) -> List[Tuple[str, Path, Path, bool]]:
    tasks: List[Tuple[str, Path, Path, bool]] = []
    for config in tqdm(dataset_configs, desc="collect_datasets"):
        dataset_name = config["dataset_name"]
        input_dir = Path(config["input_pdb_dir"])
        output_dir = Path(config["output_pdbfixer_dir"])
        if not input_dir.is_dir():
            tqdm.write(f"[SKIP] {dataset_name}: missing input dir {input_dir}")
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        for input_path in tqdm(sorted(input_dir.glob("*.pdb")), desc=f"collect:{dataset_name}", leave=False):
            output_path = output_dir / input_path.name
            if output_path.exists() and not overwrite:
                continue
            tasks.append((dataset_name, input_path, output_path, overwrite))
    return tasks

def fix_one(task: Tuple[str, Path, Path, bool]) -> Dict[str, str]:
    dataset_name, input_path, output_path, overwrite = task
    pdb_id = normalize_pdb_id(input_path)
    try:
        if output_path.exists() and not overwrite:
            return {
                "dataset": dataset_name,
                "pdb_id": pdb_id,
                "status": "skipped_exists",
                "input_path": str(input_path),
                "output_path": str(output_path),
                "reason": "",
            }
        fixer = PDBFixer(filename=str(input_path))
        fixer.missingResidues = {}
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
        strip_hydrogens_inplace(output_path)
        return {
            "dataset": dataset_name,
            "pdb_id": pdb_id,
            "status": "success",
            "input_path": str(input_path),
            "output_path": str(output_path),
            "reason": "",
        }
    except Exception as exc:
        if output_path.exists():
            output_path.unlink()
        return {
            "dataset": dataset_name,
            "pdb_id": pdb_id,
            "status": "failed",
            "input_path": str(input_path),
            "output_path": str(output_path),
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback_tail": " | ".join(traceback.format_exc().strip().splitlines()[-6:]),
        }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_NAMES))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    dataset_configs = build_dataset_configs(args.dataset_root, args.datasets)
    tasks = collect_tasks(dataset_configs, bool(args.overwrite))
    if not tasks:
        print("No PDB files to fix")
        return
    rows: List[Dict[str, str]] = []
    workers = max(1, int(args.workers))
    if workers <= 1:
        for task in tqdm(tasks, desc="PDBFixer", unit="pdb"):
            rows.append(fix_one(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(fix_one, task): task for task in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="PDBFixer", unit="pdb"):
                rows.append(future.result())
    rows.sort(key=lambda row: (row.get("dataset", ""), row.get("pdb_id", "")))
    success = sum(1 for row in rows if row.get("status") == "success")
    failed = [row for row in rows if row.get("status") == "failed"]
    for row in tqdm(failed, desc="failed_rows", leave=False):
        print(f"[FAIL] {row['dataset']} {row['pdb_id']}: {row.get('reason', '')}")
    print(f"total: {len(rows)}")
    print(f"success: {success}")
    print(f"failed: {len(failed)}")

if __name__ == "__main__":
    main()
