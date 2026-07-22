from __future__ import annotations
import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
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
            "input_pdb_dir": os.path.join(dataset_dir, "PDBFIXER"),
            "output_pqr_dir": os.path.join(dataset_dir, "PQR"),
        })
    return configs

def normalize_pdb_id(value: object) -> str:
    return Path(str(value).strip()).stem.upper()

def collect_pdb_paths(input_dir: Path) -> Dict[str, Path]:
    return {
        normalize_pdb_id(path.stem): path
        for path in sorted(input_dir.glob("*.pdb"))
        if path.is_file()
    }

def run_pdb2pqr(input_pdb: Path, output_pqr: Path) -> Tuple[bool, str]:
    output_pqr.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "pdb2pqr",
            "--ff",
            "PARSE",
            "--keep-chain",
            "--drop-water",
            str(input_pdb),
            str(output_pqr),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    log_text = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
    if result.returncode != 0:
        if output_pqr.exists():
            output_pqr.unlink()
        return False, log_text or "PDB2PQR failed"
    if not output_pqr.exists() or output_pqr.stat().st_size == 0:
        if output_pqr.exists():
            output_pqr.unlink()
        return False, log_text or "PDB2PQR reported success but no PQR file was written"
    return True, log_text

def process_one(task: Tuple[str, str, Path, Path]) -> Dict[str, str]:
    _, pdb_id, input_pdb_path, output_pqr_dir = task
    output_pqr_path = output_pqr_dir / f"{pdb_id}.pqr"

    success, message = run_pdb2pqr(input_pdb_path, output_pqr_path)
    if success:
        return {
            "pdb_id": pdb_id,
            "status": "success",
            "reason": "",
        }
    return {
        "pdb_id": pdb_id,
        "status": "failed",
        "reason": message,
    }

def main() -> None:
    args = parse_args()
    dataset_configs = build_dataset_configs(args.dataset_root, args.datasets)
    tasks: List[Tuple[str, str, Path, Path]] = []
    for config in tqdm(dataset_configs, desc="collect_datasets", unit="dataset"):
        input_pdb_dir = Path(config["input_pdb_dir"])
        output_pqr_dir = Path(config["output_pqr_dir"])
        if not input_pdb_dir.is_dir():
            raise SystemExit(f"Input directory does not exist: {input_pdb_dir}")
        output_pqr_dir.mkdir(parents=True, exist_ok=True)
        pdb_paths = collect_pdb_paths(input_pdb_dir)
        tasks.extend(
            (config["dataset_name"], pdb_id, pdb_path, output_pqr_dir)
            for pdb_id, pdb_path in pdb_paths.items()
        )
    if not tasks:
        print("No PDB files found")
        return

    results: List[Dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = [executor.submit(process_one, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating PQR", unit="pdb", dynamic_ncols=True):
            results.append(future.result())

    results.sort(key=lambda row: row["pdb_id"])
    success = sum(1 for row in results if row["status"] == "success")
    failed = [row for row in results if row["status"] != "success"]

    for row in failed:
        print(f"[FAIL] {row['pdb_id']}: {row['status']} | {row['reason']}")

    print(f"total: {len(results)}")
    print(f"success: {success}")
    print(f"failed: {len(failed)}")
    print(f"datasets: {', '.join(str(config['dataset_name']) for config in dataset_configs)}")

if __name__ == "__main__":
    main()
