import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Tuple
import freesasa
import numpy as np
from tqdm import tqdm

FREESASA_CLASSIFIER_NAME = "protor"

def entry_atom_identity(entry: Dict[str, object]) -> Tuple[str, str, str, str]:
    chain_id = str(entry["chain_id"]).strip() or "_"
    residue_key = entry["residue_key"]
    residue_number = f"{str(residue_key[1]).strip()}{str(residue_key[2]).strip()}"
    return (
        str(entry["atom_name"]).strip(),
        str(entry["resname"]).strip().upper(),
        chain_id,
        residue_number,
    )

def coord_round_tuple(coord: object) -> Tuple[float, float, float]:
    return (
        round(float(coord[0]), 3),
        round(float(coord[1]), 3),
        round(float(coord[2]), 3),
    )

def structure_atom_identity(structure: freesasa.Structure, atom_index: int) -> Tuple[str, str, str, str]:
    chain_id = str(structure.chainLabel(atom_index)).strip() or "_"
    return (
        str(structure.atomName(atom_index)).strip(),
        str(structure.residueName(atom_index)).strip().upper(),
        chain_id,
        str(structure.residueNumber(atom_index)).strip(),
    )

def infer_pdb_element(atom_name: str, element: object = "") -> str:
    raw = str(element).strip().upper()
    if raw:
        return raw[:2]
    atom = re.sub(r"[^A-Za-z]", "", str(atom_name).strip()).upper()
    if not atom:
        return "C"
    if atom.startswith("SE"):
        return "SE"
    return atom[:1]

def format_pdb_atom_line(serial: int, entry: Dict[str, object]) -> str:
    residue_key = entry["residue_key"]
    atom_name = str(entry["atom_name"]).strip()
    resname = str(entry["resname"]).strip().upper()[:3]
    chain_id = (str(entry["chain_id"]).strip() or "_")[:1]
    resseq = int(str(residue_key[1]).strip())
    icode = (str(residue_key[2]).strip() or " ")[:1]
    coord = entry["coord"]
    atom_field = atom_name[:4]
    if len(atom_field) < 4:
        atom_field = atom_field.rjust(4)
    element = infer_pdb_element(atom_name, entry.get("element") or entry.get("element_symbol") or "")
    serial_value = ((int(serial) - 1) % 99999) + 1
    return (
        f"ATOM  {serial_value:5d} {atom_field} {resname:>3} {chain_id}"
        f"{resseq:4d}{icode}   {float(coord[0]):8.3f}{float(coord[1]):8.3f}{float(coord[2]):8.3f}"
        f"  1.00  0.00          {element:>2}\n"
    )

def write_pqr_entries_as_pdb(entries: Sequence[Dict[str, object]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        for serial, entry in tqdm(enumerate(entries, start=1), total=len(entries), desc=f"write_pdb:{Path(output_path).stem}", leave=False):
            handle.write(format_pdb_atom_line(serial, entry))
        handle.write("TER\nEND\n")

def get_freesasa_classifier() -> freesasa.Classifier:
    return freesasa.Classifier.getStandardClassifier(FREESASA_CLASSIFIER_NAME)

def build_sasa_maps_from_entries(entries: Sequence[Dict[str, object]], structure_name: str, tmp_root: str = None) -> Tuple[Dict[Tuple[str, str, str, str, float, float, float], float], Dict[Tuple[str, str, str, str], List[float]], np.ndarray]:
    freesasa.setVerbosity(freesasa.nowarnings)
    if not entries:
        return {}, {}, np.zeros((0, 1), dtype=np.float32)
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", str(structure_name))
    classifier = get_freesasa_classifier()
    if tmp_root:
        os.makedirs(tmp_root, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"freesasa_{safe_name}_", dir=tmp_root if tmp_root else None) as tmp_dir:
        pdb_path = os.path.join(tmp_dir, f"{safe_name}.pdb")
        write_pqr_entries_as_pdb(entries, pdb_path)
        structure = freesasa.Structure(pdb_path, classifier=classifier, options={"hetatm": False, "skip-unknown": False, "hydrogen": True})
        if structure.nAtoms() != len(entries):
            raise RuntimeError(f"FreeSASA atom count mismatch: pqr_entries={len(entries)} freesasa_atoms={structure.nAtoms()} structure={structure_name}")
        result = freesasa.calc(structure)
        exact_map: Dict[Tuple[str, str, str, str, float, float, float], float] = {}
        identity_map: Dict[Tuple[str, str, str, str], List[float]] = {}
        full_raw = np.zeros((structure.nAtoms(), 1), dtype=np.float32)
        for atom_index in tqdm(range(structure.nAtoms()), desc=f"sasa:{safe_name}", leave=False):
            area = float(result.atomArea(atom_index))
            identity_key = structure_atom_identity(structure, atom_index)
            exact_key = identity_key + coord_round_tuple(structure.coord(atom_index))
            if exact_key not in exact_map:
                exact_map[exact_key] = area
            identity_map.setdefault(identity_key, []).append(area)
        return exact_map, identity_map, full_raw

def lookup_sasa_value(
    entry: Dict[str, object],
    exact_map: Dict[Tuple[str, str, str, str, float, float, float], float],
    identity_map: Dict[Tuple[str, str, str, str], List[float]],
) -> Optional[float]:
    identity_key = entry_atom_identity(entry)
    exact_value = exact_map.get(identity_key + coord_round_tuple(entry["coord"]))
    if exact_value is not None:
        return float(exact_value)
    candidates = identity_map.get(identity_key)
    if candidates is None or len(candidates) != 1:
        return None
    return float(candidates[0])

def scale_sasa_numpy(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(values, 0.0)).astype(np.float32)

def scale_delta_sasa_numpy(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(values, 0.0)).astype(np.float32)

def build_delta_sasa_from_entries(
    entries: Sequence[Dict[str, object]],
    side_of_entry: Callable[[Dict[str, object]], Hashable],
    structure_name: str,
    tmp_root: str = None,
    max_lookup_failure_ratio: float = 0.5,
    bound_maps: Optional[Tuple[Dict[Tuple[str, str, str, str, float, float, float], float], Dict[Tuple[str, str, str, str], List[float]]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(entries)
    if n == 0:
        empty = np.zeros((0, 1), dtype=np.float32)
        return empty.copy(), empty.copy(), empty.copy()

    if bound_maps is not None:
        bound_exact, bound_identity = bound_maps
    else:
        bound_exact, bound_identity, _ = build_sasa_maps_from_entries(entries, structure_name, tmp_root)

    side_to_entries: Dict[Hashable, List[Dict[str, object]]] = {}
    side_order: List[Hashable] = []
    entry_sides: List[Hashable] = []
    for entry in entries:
        side = side_of_entry(entry)
        entry_sides.append(side)
        bucket = side_to_entries.get(side)
        if bucket is None:
            side_to_entries[side] = [entry]
            side_order.append(side)
        else:
            bucket.append(entry)

    bound_raw = np.zeros((n, 1), dtype=np.float32)
    unbound_raw = np.zeros((n, 1), dtype=np.float32)
    dsasa_raw = np.zeros((n, 1), dtype=np.float32)

    bound_values: List[Optional[float]] = []
    bound_failures = 0
    for index, entry in enumerate(entries):
        bound = lookup_sasa_value(entry, bound_exact, bound_identity)
        bound_values.append(bound)
        if bound is None:
            bound_failures += 1
        else:
            bound_raw[index, 0] = bound

    if len(side_order) < 2:
        warnings.warn(
            f"build_delta_sasa_from_entries: only {len(side_order)} binding side(s) for "
            f"structure={structure_name}; no partner present, ΔSASA=0 for all atoms.",
            RuntimeWarning,
            stacklevel=2,
        )
        unbound_raw[:] = bound_raw
        return bound_raw, unbound_raw, dsasa_raw

    if len(side_order) > 2:
        warnings.warn(
            f"build_delta_sasa_from_entries: expected 2 binding sides but found "
            f"{len(side_order)} for structure={structure_name}; computing per-side unbound "
            f"SASA (ΔSASA reflects burial against all other sides combined).",
            RuntimeWarning,
            stacklevel=2,
        )

    side_exact: Dict[Hashable, Dict[Tuple[str, str, str, str, float, float, float], float]] = {}
    side_identity: Dict[Hashable, Dict[Tuple[str, str, str, str], List[float]]] = {}
    for side in side_order:
        side_token = re.sub(r"[^A-Za-z0-9._-]", "_", str(side))
        s_exact, s_identity, _ = build_sasa_maps_from_entries(
            side_to_entries[side], f"{structure_name}_side_{side_token}", tmp_root
        )
        side_exact[side] = s_exact
        side_identity[side] = s_identity

    unbound_failures = 0
    for index, entry in enumerate(entries):
        side = entry_sides[index]
        unbound = lookup_sasa_value(entry, side_exact[side], side_identity[side])
        if unbound is None:
            unbound_failures += 1
            continue
        unbound_raw[index, 0] = unbound
        bound = bound_values[index]
        if bound is None:
            continue
        dsasa_raw[index, 0] = max(0.0, unbound - bound)

    total_failures = bound_failures + unbound_failures
    if total_failures:
        ratio = total_failures / float(2 * n)
        message = (
            f"build_delta_sasa_from_entries: {total_failures} SASA lookup failure(s) "
            f"(bound={bound_failures}, unbound={unbound_failures}) over {2 * n} lookups "
            f"({ratio:.1%}) for structure={structure_name}; affected atoms set to 0."
        )
        if ratio > max_lookup_failure_ratio:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    return bound_raw, unbound_raw, dsasa_raw
