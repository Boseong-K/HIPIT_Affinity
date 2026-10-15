import os
import re
import tempfile
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple
from tqdm import tqdm

def normalize_chain_ids(chain_ids: Optional[Sequence[str]]) -> List[str]:
    if chain_ids is None:
        return []
    out: List[str] = []
    seen = set()
    for raw in tqdm(chain_ids, desc="normalize_chains", leave=False):
        chain = str(raw).strip()
        if not chain or chain in seen:
            continue
        out.append(chain)
        seen.add(chain)
    return out

def safe_float(text: str, default: float = 0.0) -> float:
    try:
        return float(text)
    except Exception:
        return default

def infer_element(atom_name: str, resname: str = "") -> str:
    atom = str(atom_name).strip().upper()
    residue = str(resname).strip().upper()
    if not atom:
        return "C"
    if residue == "MSE" and atom.startswith("SE"):
        return "SE"
    for prefix in tqdm(("SE", "ZN", "FE", "MG", "CL", "BR"), desc="infer_element_prefix", leave=False):
        if atom.startswith(prefix):
            return prefix
    if atom.startswith("CA"):
        return "C"
    if atom[0].isdigit() and len(atom) > 1:
        atom = atom[1:]
    return atom[0]

def parse_atom_record(line: str) -> Optional[Dict[str, object]]:
    if not line.startswith("ATOM"):
        return None
    atom_name = line[12:16].strip()
    altloc = line[16].strip()
    resname = line[17:20].strip().upper()
    chain_id = line[21].strip() or "_"
    resseq_text = line[22:26].strip()
    icode = line[26].strip()
    if altloc not in {"", "A"}:
        return None
    try:
        resseq = int(resseq_text)
    except ValueError:
        return None
    return {
        "atom_name": atom_name,
        "resname": resname,
        "chain_id": chain_id,
        "resseq": resseq,
        "icode": icode,
        "x": safe_float(line[30:38]),
        "y": safe_float(line[38:46]),
        "z": safe_float(line[46:54]),
    }

def format_pdb_atom_line(record: Dict[str, object], serial: int) -> str:
    atom_name = str(record["atom_name"]).strip()
    atom_field = atom_name.rjust(4) if len(atom_name) < 4 else atom_name[:4]
    resname = str(record["resname"]).strip().upper()[:3].rjust(3)
    chain_id = (str(record["chain_id"]).strip() or "_")[0]
    icode = (str(record["icode"]).strip() or " ")[:1]
    element = infer_element(atom_name, str(record["resname"])).strip().upper()[:2].rjust(2)
    return (
        f"ATOM  {int(serial):5d} {atom_field} {resname} {chain_id}"
        f"{int(record['resseq']):4d}{icode}   {float(record['x']):8.3f}{float(record['y']):8.3f}{float(record['z']):8.3f}"
        f"  1.00  0.00          {element}\n"
    )

def write_atom_pdb(src_path: str, dst_path: str, chain_ids: Optional[Sequence[str]] = None) -> int:
    chain_id_set = set(normalize_chain_ids(chain_ids)) if chain_ids is not None else None
    atom_count = 0
    with open(src_path, "r", encoding="utf-8", errors="ignore") as fin, open(dst_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"write_atom_pdb:{os.path.basename(dst_path)}", leave=False):
            record = parse_atom_record(line)
            if record is None:
                continue
            chain_id = str(record["chain_id"])
            if chain_id_set is not None and chain_id not in chain_id_set:
                continue
            atom_count += 1
            fout.write(format_pdb_atom_line(record, atom_count))
        if atom_count > 0:
            fout.write("TER\n")
        fout.write("END\n")
    return atom_count

@contextmanager
def temporary_pdb_from_structure(structure_path: str):
    abs_path = os.path.abspath(structure_path)
    tmp_pdb_path = None
    try:
        if abs_path.lower().endswith(".pdb"):
            yield abs_path
            return
        tmp_pdb = tempfile.NamedTemporaryFile(prefix="residue_interface_", suffix=".pdb", delete=False)
        tmp_pdb_path = tmp_pdb.name
        tmp_pdb.close()
        write_atom_pdb(abs_path, tmp_pdb_path, None)
        yield tmp_pdb_path
    finally:
        if tmp_pdb_path is not None and os.path.exists(tmp_pdb_path):
            os.remove(tmp_pdb_path)

def split_residue_number(raw: object) -> Tuple[int, str]:
    text = str(raw).strip()
    match = re.match(r"^(-?\d+)([A-Za-z]?)$", text)
    if match:
        return int(match.group(1)), match.group(2)
    values = re.findall(r"-?\d+", text)
    if values:
        return int(values[0]), ""
    return 0, ""

def residue_sasa_map(pdb_path: str) -> Dict[Tuple[str, int, str], float]:
    import freesasa

    freesasa.setVerbosity(freesasa.nowarnings)
    structure = freesasa.Structure(pdb_path, options={"hetatm": False, "skip-unknown": False, "hydrogen": True})
    result = freesasa.calc(structure)
    areas: Dict[Tuple[str, int, str], float] = {}
    residue_areas = result.residueAreas()
    for chain_id, chain_areas in tqdm(residue_areas.items(), desc=f"sasa_chains:{os.path.basename(pdb_path)}", leave=False):
        chain = str(chain_id).strip() or "_"
        for residue_number, area in tqdm(chain_areas.items(), desc=f"sasa_residues:{chain}", leave=False):
            resseq, icode = split_residue_number(residue_number)
            if resseq == 0:
                continue
            areas[(chain, resseq, icode)] = float(getattr(area, "total", 0.0))
            if icode:
                areas.setdefault((chain, resseq, ""), float(getattr(area, "total", 0.0)))
    return areas

def run_freesasa_bsa_map(
    pdb_path: str,
    chain_ids: Sequence[str],
    binding_pair: Optional[Tuple[Sequence[str], Sequence[str]]] = None,
) -> Dict[Tuple[str, int, str], float]:
    if binding_pair is not None:
        left = binding_pair[0]
        right = binding_pair[1]
        left_set = set(left)
        groups = [left, right]
        complex_chains = left + [chain for chain in tqdm(right, desc="complex_right_chains", leave=False) if chain not in left_set]
    else:
        all_chain_ids = normalize_chain_ids(chain_ids)
        groups = [[chain] for chain in tqdm(all_chain_ids, desc="bsa_chain_groups", leave=False)]
        complex_chains = all_chain_ids
    bsa_map: Dict[Tuple[str, int, str], float] = {}
    with tempfile.TemporaryDirectory(prefix="residue_bsa_") as tmp_dir:
        complex_pdb = os.path.join(tmp_dir, "complex.pdb")
        if write_atom_pdb(pdb_path, complex_pdb, complex_chains) == 0:
            return bsa_map
        complex_map = residue_sasa_map(complex_pdb)
        for group_idx, group in enumerate(tqdm(groups, desc="bsa_groups", leave=False)):
            if not group:
                continue
            monomer_pdb = os.path.join(tmp_dir, f"monomer_{group_idx}.pdb")
            if write_atom_pdb(pdb_path, monomer_pdb, group) == 0:
                continue
            monomer_map = residue_sasa_map(monomer_pdb)
            for key, monomer_area in tqdm(monomer_map.items(), desc=f"bsa_map:{group_idx}", leave=False):
                bsa_map[key] = max(0.0, float(monomer_area) - float(complex_map.get(key, 0.0)))
    return bsa_map

def build_bsa_interface_values(
    pdb_path: str,
    residues: Sequence[Dict[str, object]],
    binding_pair: Tuple[Sequence[str], Sequence[str]],
) -> List[float]:
    left_chain_ids = normalize_chain_ids(binding_pair[0])
    right_chain_ids = normalize_chain_ids(binding_pair[1])
    with temporary_pdb_from_structure(pdb_path) as structure_pdb_path:
        bsa_map_abs = run_freesasa_bsa_map(structure_pdb_path, [], binding_pair=(left_chain_ids, right_chain_ids))
    bsa_values: List[float] = []
    for residue in tqdm(residues, desc="bsa_interface_values", leave=False):
        chain = str(residue["chain_id"])
        resseq = int(residue["resseq"])
        icode = str(residue["icode"])
        residue_key = (chain, resseq, icode)
        residue_key_no_icode = (chain, resseq, "")
        bsa_abs = float(bsa_map_abs.get(residue_key, bsa_map_abs.get(residue_key_no_icode, 0.0)))
        bsa_values.append(bsa_abs)
    return bsa_values
