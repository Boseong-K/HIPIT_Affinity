import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Sequence
import numpy as np
from openbabel import openbabel as ob
from tqdm import tqdm
from Atom_SASA import coord_round_tuple, entry_atom_identity, write_pqr_entries_as_pdb

def get_temp_root(tmp_root: str) -> str:
    tmp_root = str(tmp_root).strip()
    if not tmp_root:
        raise ValueError("tmp_root is required")
    os.makedirs(tmp_root, exist_ok=True)
    return tmp_root

def prepare_openbabel_mol_from_entries(entries: Sequence[Dict[str, object]], molecule_name: str, tmp_root: str) -> ob.OBMol:
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", str(molecule_name))
    ob.obErrorLog.SetOutputLevel(0)
    with tempfile.TemporaryDirectory(prefix=f"openbabel_atom_type_{safe_name}_", dir=get_temp_root(tmp_root)) as tmp_dir:
        pdb_path = os.path.join(tmp_dir, f"{safe_name}.pdb")
        write_pqr_entries_as_pdb(entries, pdb_path)
        conversion = ob.OBConversion()
        if not conversion.SetInFormat("pdb"):
            raise RuntimeError("OpenBabel failed to set PDB input format")
        mol = ob.OBMol()
        if not conversion.ReadFile(mol, pdb_path):
            raise RuntimeError(f"OpenBabel failed to read PDB: {pdb_path}")
        if mol.NumAtoms() != len(entries):
            raise RuntimeError(f"OpenBabel atom count mismatch: entries={len(entries)} openbabel_atoms={mol.NumAtoms()} molecule={molecule_name}")
        mol.ConnectTheDots()
        mol.PerceiveBondOrders()
        mol.FindRingAtomsAndBonds()
        mol.SetTitle(Path(str(molecule_name)).stem)
        return mol

def atom_type_vector(atom: ob.OBAtom) -> np.ndarray:
    is_hbond_donor = bool(atom.IsHbondDonor())
    is_hbond_acceptor = bool(atom.IsHbondAcceptor())
    is_donor_hydrogen = bool(atom.IsHbondDonorH())
    is_polar_hydrogen = bool(atom.IsPolarHydrogen())
    is_nonpolar_hydrogen = bool(atom.IsNonPolarHydrogen())
    atomic_num = int(atom.GetAtomicNum())
    is_polar_atom = bool(is_hbond_donor or is_hbond_acceptor or is_donor_hydrogen or is_polar_hydrogen)
    is_apolar_atom = bool((not is_polar_atom) and (is_nonpolar_hydrogen or atomic_num == 1 or atomic_num == 6))
    return np.asarray(
        [
            float(is_hbond_donor),
            float(is_hbond_acceptor),
            float(is_donor_hydrogen),
            float(is_polar_atom),
            float(is_apolar_atom),
            float(atom.IsAromatic()),
            float(atom.IsInRing()),
            float(atom.IsHeteroatom()),
        ],
        dtype=np.float32,
    )

def build_atom_type_feature_maps(entries: Sequence[Dict[str, object]], molecule_name: str, tmp_root: str):
    exact_map = {}
    identity_map = {}
    if len(entries) == 0:
        return exact_map, identity_map
    mol = prepare_openbabel_mol_from_entries(entries, molecule_name, tmp_root)
    atoms = list(ob.OBMolAtomIter(mol))
    for entry, atom in tqdm(zip(entries, atoms), total=len(entries), desc=f"atom_type:{molecule_name}", leave=False):
        vector = atom_type_vector(atom)
        identity_key = entry_atom_identity(entry)
        exact_map.setdefault(identity_key + coord_round_tuple(entry["coord"]), vector)
        identity_map.setdefault(identity_key, []).append(vector)
    return exact_map, identity_map

def lookup_atom_type_feature(entry: Dict[str, object], exact_map, identity_map) -> np.ndarray:
    identity_key = entry_atom_identity(entry)
    exact_value = exact_map.get(identity_key + coord_round_tuple(entry["coord"]))
    if exact_value is not None:
        return exact_value
    candidates = identity_map.get(identity_key)
    if candidates is None or len(candidates) != 1:
        raise RuntimeError(f"missing atom type mapping: {identity_key}")
    return candidates[0]

def build_atom_type_features(
    context_entries: Sequence[Dict[str, object]],
    target_entries: Sequence[Dict[str, object]] = None,
    molecule_name: str = "atom_type",
    tmp_root: str = "",
) -> np.ndarray:
    resolved_target_entries = context_entries if target_entries is None else target_entries
    out = np.zeros((len(resolved_target_entries), 8), dtype=np.float32)
    if len(resolved_target_entries) == 0:
        return out
    exact_map, identity_map = build_atom_type_feature_maps(context_entries, molecule_name, tmp_root)
    for idx, entry in enumerate(tqdm(resolved_target_entries, desc=f"map_atom_type:{molecule_name}", leave=False)):
        out[idx, :] = lookup_atom_type_feature(entry, exact_map, identity_map)
    return out
