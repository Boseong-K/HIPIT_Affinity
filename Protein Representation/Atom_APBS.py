import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from tqdm import tqdm

def compute_grid_lengths(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coord_min = coords.min(axis=0).astype(np.float64)
    coord_max = coords.max(axis=0).astype(np.float64)
    fine_lengths = np.maximum(coord_max - coord_min + (2.0 * 20.0), 48.0)
    coarse_lengths = np.maximum(fine_lengths * 1.7, fine_lengths + (2.0 * 20.0))
    center = (coord_min + coord_max) / 2.0
    return center, fine_lengths, coarse_lengths

def stack_entry_coords(entries: Sequence[Dict[str, object]], desc: str) -> np.ndarray:
    coords = np.empty((len(entries), 3), dtype=np.float32)
    for idx, entry in enumerate(tqdm(entries, desc=desc, leave=False)):
        coords[idx] = entry["coord"]
    return coords

def format_apbs_pqr_line(serial: int, entry: Dict[str, object]) -> str:
    residue_key = entry["residue_key"]
    atom_name = (str(entry["atom_name"]).strip() or "X")[:4]
    resname = (str(entry["resname"]).strip().upper() or "UNK")[:3]
    chain_id = (str(entry["chain_id"]).strip() or "_")[:1]
    resseq = int(str(residue_key[1]).strip())
    coord = entry["coord"]
    charge = float(entry["charge"])
    radius = float(entry["radius"])
    serial_value = ((int(serial) - 1) % 99999) + 1
    return (
        f"ATOM {serial_value:d} {atom_name} {resname} {chain_id} {resseq:d} "
        f"{float(coord[0]):.3f} {float(coord[1]):.3f} {float(coord[2]):.3f} "
        f"{charge:.4f} {radius:.4f}\n"
    )

def write_apbs_pqr(entries: Sequence[Dict[str, object]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        for serial, entry in tqdm(enumerate(entries, start=1), total=len(entries), desc=f"write_apbs_pqr:{Path(output_path).stem}", leave=False):
            handle.write(format_apbs_pqr_line(serial, entry))
        handle.write("TER\nEND\n")

def write_apbs_input(pqr_path: str, input_path: str, output_stem: str, coords: np.ndarray) -> None:
    center, fine_lengths, coarse_lengths = compute_grid_lengths(coords)
    lines = [
        "read",
        f"    mol pqr {pqr_path}",
        "end",
        "elec name pot",
        "    mg-auto",
        "    dime 129 129 129",
        f"    cglen {coarse_lengths[0]:.6f} {coarse_lengths[1]:.6f} {coarse_lengths[2]:.6f}",
        f"    fglen {fine_lengths[0]:.6f} {fine_lengths[1]:.6f} {fine_lengths[2]:.6f}",
        f"    cgcent {center[0]:.6f} {center[1]:.6f} {center[2]:.6f}",
        f"    fgcent {center[0]:.6f} {center[1]:.6f} {center[2]:.6f}",
        "    mol 1",
        "    lpbe",
        "    bcfl sdh",
        "    pdie 2.000000",
        "    sdie 78.540000",
        "    ion charge 1 conc 0.150000 radius 2.000000",
        "    ion charge -1 conc 0.150000 radius 1.800000",
        "    srfm smol",
        "    chgm spl2",
        "    sdens 10.000000",
        "    srad 1.400000",
        "    swin 0.300000",
        "    temp 298.150000",
        "    calcenergy no",
        "    calcforce no",
        f"    write pot dx {output_stem}",
        "end",
        "quit",
    ]
    with open(input_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

def locate_dx_file(work_dir: str, output_stem: str) -> str:
    expected = os.path.join(work_dir, f"{output_stem}.dx")
    if os.path.isfile(expected):
        return expected
    raise FileNotFoundError(f"APBS output dx not found: {expected}")

def run_apbs(pqr_path: str, coords: np.ndarray, apbs_exe: str, work_dir: str, output_stem: str) -> str:
    input_path = os.path.join(work_dir, f"{output_stem}.in")
    if coords.size == 0:
        raise RuntimeError("No atoms parsed from PQR")
    write_apbs_input(pqr_path, input_path, output_stem, coords)
    completed = subprocess.run(
        [apbs_exe, input_path],
        cwd=work_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "APBS failed").strip())
    return locate_dx_file(work_dir, output_stem)

def parse_counts_line(line: str) -> Tuple[int, int, int]:
    match = re.search(r"counts\s+(\d+)\s+(\d+)\s+(\d+)", line)
    if match is None:
        raise ValueError("DX counts line not found")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))

def parse_dx(dx_path: str) -> Dict[str, object]:
    counts: Optional[Tuple[int, int, int]] = None
    origin: Optional[np.ndarray] = None
    deltas: List[np.ndarray] = []
    value_chunks: List[np.ndarray] = []
    in_data = False
    expected_values = 0
    with open(dx_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in tqdm(handle, desc=f"parse_dx:{Path(dx_path).name}", leave=False):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("object 1 class gridpositions counts"):
                counts = parse_counts_line(stripped)
                expected_values = int(counts[0] * counts[1] * counts[2])
                continue
            if stripped.startswith("origin"):
                parts = stripped.split()
                origin = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
                continue
            if stripped.startswith("delta"):
                parts = stripped.split()
                deltas.append(np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64))
                continue
            if "data follows" in stripped:
                in_data = True
                continue
            if in_data:
                if stripped.startswith("attribute") or stripped.startswith("object") or stripped.startswith("component"):
                    continue
                chunk = np.fromstring(stripped, sep=" ", dtype=np.float64)
                if chunk.size > 0:
                    value_chunks.append(chunk)
    if counts is None or origin is None or len(deltas) != 3:
        raise ValueError(f"Incomplete DX header: {dx_path}")
    values = np.concatenate(value_chunks, axis=0) if value_chunks else np.empty((0,), dtype=np.float64)
    if len(values) < expected_values:
        raise ValueError(f"Incomplete DX values: {dx_path}")
    dx = values[:expected_values]
    steps = np.asarray([np.linalg.norm(delta) for delta in deltas], dtype=np.float64)
    if np.any(steps <= 0.0):
        raise ValueError(f"Invalid DX spacing: {dx_path}")
    return {
        "counts": counts,
        "origin": origin,
        "steps": steps,
        "values": dx,
    }

def sample_dx_at_coords(dx: Dict[str, object], coords: np.ndarray) -> np.ndarray:
    nx, ny, nz = dx["counts"]
    origin = dx["origin"]
    steps = dx["steps"]
    values = dx["values"]
    frac = (coords.astype(np.float64) - origin[None, :]) / steps[None, :]
    max_index = np.asarray([max(nx - 1, 0), max(ny - 1, 0), max(nz - 1, 0)], dtype=np.float64)
    frac = np.clip(frac, 0.0, max_index[None, :])
    i0 = np.floor(frac).astype(np.int64)
    i1 = np.minimum(i0 + 1, max_index.astype(np.int64)[None, :])
    w = frac - i0.astype(np.float64)
    wx = w[:, 0]
    wy = w[:, 1]
    wz = w[:, 2]

    def gather(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray) -> np.ndarray:
        flat = ix + (nx * (iy + (ny * iz)))
        return values[flat]

    c000 = gather(i0[:, 0], i0[:, 1], i0[:, 2])
    c100 = gather(i1[:, 0], i0[:, 1], i0[:, 2])
    c010 = gather(i0[:, 0], i1[:, 1], i0[:, 2])
    c110 = gather(i1[:, 0], i1[:, 1], i0[:, 2])
    c001 = gather(i0[:, 0], i0[:, 1], i1[:, 2])
    c101 = gather(i1[:, 0], i0[:, 1], i1[:, 2])
    c011 = gather(i0[:, 0], i1[:, 1], i1[:, 2])
    c111 = gather(i1[:, 0], i1[:, 1], i1[:, 2])
    out = (
        c000 * (1.0 - wx) * (1.0 - wy) * (1.0 - wz)
        + c100 * wx * (1.0 - wy) * (1.0 - wz)
        + c010 * (1.0 - wx) * wy * (1.0 - wz)
        + c110 * wx * wy * (1.0 - wz)
        + c001 * (1.0 - wx) * (1.0 - wy) * wz
        + c101 * wx * (1.0 - wy) * wz
        + c011 * (1.0 - wx) * wy * wz
        + c111 * wx * wy * wz
    )
    return out.astype(np.float32)

def scale_apbs_numpy(values: np.ndarray) -> np.ndarray:
    return np.arcsinh(values / 15.0).astype(np.float32)
