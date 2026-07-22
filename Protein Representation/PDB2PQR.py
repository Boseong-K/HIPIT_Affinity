from __future__ import annotations
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

DATASET_CONFIGS = (
    {
        "name": "<DATASET>",
        "input_pdb_dir": Path(""),
        "output_pqr_dir": Path(""),
    },
)

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
    log_path = output_pqr.with_suffix(".log")
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
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(log_text)
        if log_text and not log_text.endswith("\n"):
            handle.write("\n")
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
    tasks: List[Tuple[str, str, Path, Path]] = []
    for config in tqdm(DATASET_CONFIGS, desc="collect_datasets", unit="dataset"):
        input_pdb_dir = Path(config["input_pdb_dir"])
        output_pqr_dir = Path(config["output_pqr_dir"])
        if not input_pdb_dir.is_dir():
            raise SystemExit(f"Input directory does not exist: {input_pdb_dir}")
        output_pqr_dir.mkdir(parents=True, exist_ok=True)
        pdb_paths = collect_pdb_paths(input_pdb_dir)
        tasks.extend(
            (str(config["name"]), pdb_id, pdb_path, output_pqr_dir)
            for pdb_id, pdb_path in pdb_paths.items()
        )
    if not tasks:
        print("No PDB files found")
        return

    results: List[Dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=16) as executor:
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
    print(f"datasets: {', '.join(str(config['name']) for config in DATASET_CONFIGS)}")

if __name__ == "__main__":
    main()
