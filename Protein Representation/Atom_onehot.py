import re
from pathlib import Path
from typing import Dict, Final, List, Sequence, Tuple
import numpy as np
import torch
from tqdm import tqdm

ELEMENTS: Final[Tuple[str, ...]] = ("C", "H", "O", "N", "S")
WATER_RESIDUE_NAMES: Final[set[str]] = {"HOH", "WAT", "H2O", "SOL", "TIP", "TIP3", "TP3"}
AA20: Final[Tuple[str, ...]] = (
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
)

ELEMENT_TO_IDX: Final[Dict[str, int]] = {element: idx for idx, element in enumerate(ELEMENTS)}
AA_TO_IDX: Final[Dict[str, int]] = {aa: idx for idx, aa in enumerate(AA20)}

SCALAR_DIM: Final[int] = 2

def canonical_resname(resname: str) -> str:
    return str(resname).strip().upper()

def is_water_residue_name(resname: str) -> bool:
    return str(resname).strip().upper() in WATER_RESIDUE_NAMES

def normalize_element(raw: str) -> str:
    element = str(raw).strip().upper()
    return element if element in ELEMENT_TO_IDX else ""

def normalize_element_symbol(raw: str) -> str:
    symbol = re.sub(r"[^A-Za-z]", "", str(raw))
    if not symbol:
        return ""
    return symbol[:1].upper() + symbol[1:].lower()

def infer_element_symbol_from_pdb_line(line: str, atom_name_value: str = "") -> str:
    element = normalize_element_symbol(line[76:78] if len(line) >= 78 else "")
    if element:
        return element
    atom_name = re.sub(r"[^A-Za-z]", "", atom_name_value if atom_name_value else line[12:16])
    if not atom_name:
        return ""
    if line.startswith("HETATM"):
        element = normalize_element_symbol(atom_name[:2])
        if element:
            return element
    element = normalize_element_symbol(atom_name[:1])
    if element:
        return element
    return ""

def is_float_atom_value(atom_value: str) -> bool:
    try:
        float(atom_value)
        return True
    except ValueError:
        return False

def split_atom_resseq_icode(atom_value: str) -> Tuple[str, str]:
    match = re.match(r"^(-?\d+)([A-Za-z]?)$", str(atom_value).strip())
    if match:
        return match.group(1), match.group(2)
    return str(atom_value).strip(), ""

def looks_like_atom_resseq_value(atom_value: str) -> bool:
    return re.match(r"^-?\d+[A-Za-z]?$", str(atom_value).strip()) is not None

def parse_pqr_chain_and_residue(line: str, fields: Sequence[str]) -> Tuple[str, str, str]:
    if len(fields) < 6:
        return "", "", ""
    atom_field_4 = fields[4].strip()
    atom_field_5 = fields[5].strip()
    if (
        len(fields) >= 7
        and len(atom_field_4) == 1
        and atom_field_4.isalnum()
        and looks_like_atom_resseq_value(atom_field_5)
        and is_float_atom_value(fields[6])
    ):
        resseq, icode = split_atom_resseq_icode(atom_field_5)
        return atom_field_4, resseq, icode
    combined_match = re.match(r"^([A-Za-z])(-?\d+[A-Za-z]?)$", atom_field_4)
    if combined_match and is_float_atom_value(atom_field_5):
        resseq, icode = split_atom_resseq_icode(combined_match.group(2))
        return combined_match.group(1), resseq, icode
    if looks_like_atom_resseq_value(atom_field_4) and is_float_atom_value(atom_field_5):
        resseq, icode = split_atom_resseq_icode(atom_field_4)
        return "", resseq, icode
    fixed_chain = line[21].strip() if len(line) > 21 else ""
    fixed_resseq = line[22:26].strip() if len(line) > 26 else ""
    if fixed_resseq and (fixed_chain or not looks_like_atom_resseq_value(atom_field_4)):
        return fixed_chain, fixed_resseq, line[26].strip() if len(line) > 26 else ""
    return "", "", ""

def parse_pqr_xyzcr(line: str, fields: Sequence[str]) -> Tuple[float, float, float, float, float]:
    if len(fields) < 5:
        raise ValueError("Failed to parse xyzcr from PQR line")
    atom_field_4 = fields[4].strip()
    atom_field_5 = fields[5].strip() if len(fields) >= 6 else ""
    combined_match = re.match(r"^([A-Za-z])(-?\d+[A-Za-z]?)$", atom_field_4)
    if (
        len(fields) >= 7
        and len(atom_field_4) == 1
        and atom_field_4.isalnum()
        and looks_like_atom_resseq_value(atom_field_5)
    ):
        tail_text = " ".join(fields[6:])
    elif combined_match and len(fields) >= 6 and is_float_atom_value(atom_field_5):
        tail_text = " ".join(fields[5:])
    elif looks_like_atom_resseq_value(fields[4]):
        tail_text = " ".join(fields[5:])
    else:
        tail_text = " ".join(fields[4:])
    tail_atom_values = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", tail_text)
    if len(tail_atom_values) < 5:
        tail_atom_values = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line[30:])
    if len(tail_atom_values) < 5:
        raise ValueError("Failed to parse xyzcr from PQR line")
    x, y, z, charge, radius = map(float, tail_atom_values[:5])
    return x, y, z, charge, radius

def build_entries_from_pqr_text(pqr_path: str) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    with open(pqr_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in tqdm(handle, desc=f"parse_pqr:{Path(pqr_path).name}", leave=False):
            if not line.startswith("ATOM"):
                continue
            fields = line.split()
            if len(fields) < 6:
                continue
            atom_name = fields[2].strip() if len(fields) > 2 else line[12:16].strip()
            resname = fields[3].strip().upper() if len(fields) > 3 else line[17:20].strip().upper()
            if is_water_residue_name(resname):
                continue
            chain_id, resseq, icode = parse_pqr_chain_and_residue(line, fields)
            try:
                x, y, z, charge, radius = parse_pqr_xyzcr(line, fields)
                coord = np.array([x, y, z], dtype=np.float32)
            except ValueError:
                continue
            element_symbol = infer_element_symbol_from_pdb_line(line, atom_name)
            entries.append(
                {
                    "coord": coord,
                    "atom_name": atom_name,
                    "element": normalize_element(element_symbol),
                    "element_symbol": element_symbol,
                    "resname": resname,
                    "chain_id": chain_id,
                    "residue_key": (chain_id, resseq, icode),
                    "charge": charge,
                    "radius": radius,
                }
            )
    return entries

def build_base_features(
    target_entries: Sequence[Dict[str, object]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base_dim = len(ELEMENTS) + len(AA20) + SCALAR_DIM
    if len(target_entries) == 0:
        return (
            torch.zeros((0, base_dim), dtype=torch.float32),
            torch.zeros((0, 3), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.long),
        )
    coords_list = []
    elem_list = []
    residue_list = []
    scalar_rows = []
    for entry in target_entries:
        coords_list.append(entry["coord"])
        elem_list.append(ELEMENT_TO_IDX.get(entry["element"], -1))
        residue_list.append(AA_TO_IDX.get(canonical_resname(entry["resname"]), -1))
        scalar_rows.append([entry["charge"], entry["radius"]])
    pos = torch.as_tensor(np.stack(coords_list, axis=0), dtype=torch.float32)
    elem = torch.as_tensor(elem_list, dtype=torch.long)
    residue = torch.as_tensor(residue_list, dtype=torch.long)
    scalar = torch.as_tensor(scalar_rows, dtype=torch.float32)
    base = torch.zeros((len(target_entries), base_dim), dtype=torch.float32)
    valid_elem = elem >= 0
    if valid_elem.any():
        base[valid_elem, elem[valid_elem]] = 1.0
    valid_residue = residue >= 0
    if valid_residue.any():
        base[valid_residue, len(ELEMENTS) + residue[valid_residue]] = 1.0
    scalar_start = len(ELEMENTS) + len(AA20)
    base[:, scalar_start:scalar_start + scalar.size(1)] = scalar
    return (
        base,
        pos,
        elem,
    )
