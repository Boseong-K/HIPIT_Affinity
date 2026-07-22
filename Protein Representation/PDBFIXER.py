from __future__ import annotations
import argparse
import csv
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple
from openmm.app import PDBFile
from pdbfixer import PDBFixer
from tqdm import tqdm

DATASET_ROOT = Path("")
DATASET_NAMES = ("Train", "Test1", "Test2", "Test3", "Test4", "Test5")

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

def collect_tasks(dataset_root: Path, dataset_names: Tuple[str, ...], overwrite: bool) -> List[Tuple[str, Path, Path, bool]]:
    tasks: List[Tuple[str, Path, Path, bool]] = []
    for dataset_name in tqdm(dataset_names, desc="collect_datasets"):
        input_dir = dataset_root / dataset_name / "PDB"
        output_dir = dataset_root / dataset_name / "PDBFIXER"
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
        fixer.findMissingResidues()
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

def write_summary(rows: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "pdb_id", "status", "input_path", "output_path", "reason", "traceback_tail"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in tqdm(rows, desc="write_summary", leave=False):
            writer.writerow({key: row.get(key, "") for key in fieldnames})

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--summary-path", default=str(DATASET_ROOT / "pdbfixer_only_summary.csv"))
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    tasks = collect_tasks(dataset_root, DATASET_NAMES, bool(args.overwrite))
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
    write_summary(rows, Path(args.summary_path))
    success = sum(1 for row in rows if row.get("status") == "success")
    failed = [row for row in rows if row.get("status") == "failed"]
    for row in tqdm(failed, desc="failed_rows", leave=False):
        print(f"[FAIL] {row['dataset']} {row['pdb_id']}: {row.get('reason', '')}")
    print(f"total: {len(rows)}")
    print(f"success: {success}")
    print(f"failed: {len(failed)}")
    print(f"summary_path: {Path(args.summary_path)}")

if __name__ == "__main__":
    main()
