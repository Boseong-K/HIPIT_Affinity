import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import re
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from tqdm import tqdm

AA1_ALPHABET = set("ACDEFGHIKLMNPQRSTVWYX")
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
def _normalize_sequence_chars(sequence: str) -> str:
    out: List[str] = []
    for ch in tqdm(str(sequence).upper(), desc="normalize_sequence", leave=False):
        if not ch.isalpha():
            continue
        out.append(ch if ch in AA1_ALPHABET else "X")
    return "".join(out)

def _parse_fasta_records(fasta_path: str) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    header: Optional[str] = None
    seq_parts: List[str] = []
    with open(fasta_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in tqdm(f, desc=f"parse_fasta:{os.path.basename(fasta_path)}", leave=False):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    sequence = _normalize_sequence_chars("".join(seq_parts))
                    if sequence:
                        records.append((header, sequence))
                header = line[1:].strip()
                seq_parts = []
                continue
            seq_parts.append(line)
    if header is not None:
        sequence = _normalize_sequence_chars("".join(seq_parts))
        if sequence:
            records.append((header, sequence))
    return records

def _extract_chain_id_from_fasta_header(header: str) -> Optional[str]:
    text = str(header).strip()
    for part in tqdm(re.split(r"\|", text), desc="parse_fasta_header", leave=False):
        m = re.match(r"\s*Chain\s+([A-Za-z0-9_])\b", part, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    m = re.search(r"\bChain\s+([A-Za-z0-9_])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return None

@lru_cache(maxsize=None)
def _fasta_stem_index(fasta_dir: str) -> Dict[str, str]:
    index: Dict[str, str] = {}
    if not os.path.isdir(fasta_dir):
        return index
    for name in tqdm(os.listdir(fasta_dir), desc=f"fasta_index:{os.path.basename(fasta_dir)}", leave=False):
        path = os.path.join(fasta_dir, name)
        if not os.path.isfile(path):
            continue
        stem, ext = os.path.splitext(name)
        if ext.lower() != ".fasta":
            continue
        index.setdefault(stem.casefold(), path)
    return index

def _fasta_path_for_pdb(pdb_path: str, fasta_dir: str) -> str:
    pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
    exact_path = os.path.join(fasta_dir, f"{pdb_id}.fasta")
    if os.path.exists(exact_path):
        return exact_path
    for alt_stem in tqdm((pdb_id.upper(), pdb_id.lower()), desc="fasta_alt_stems", leave=False):
        alt_path = os.path.join(fasta_dir, f"{alt_stem}.fasta")
        if os.path.exists(alt_path):
            return alt_path
    indexed = _fasta_stem_index(fasta_dir).get(pdb_id.casefold())
    if indexed is not None:
        return indexed
    return exact_path

def _build_observed_chain_map(
    residues: Sequence[Dict[str, object]],
) -> Dict[str, Dict[str, object]]:
    chains: Dict[str, Dict[str, object]] = {}
    for r in tqdm(residues, desc="observed_chain_map", leave=False):
        chain = str(r["chain_id"])
        resseq = int(r["resseq"])
        icode = str(r["icode"])
        aa1 = AA3_TO_1.get(str(r["resname"]).upper(), "X")
        if chain not in chains:
            chains[chain] = {"sequence_parts": [], "keys": []}
        chains[chain]["sequence_parts"].append(aa1)
        chains[chain]["keys"].append((chain, resseq, icode))
    out: Dict[str, Dict[str, object]] = {}
    for chain, info in tqdm(chains.items(), desc="observed_chain_finalize", leave=False):
        out[chain] = {
            "sequence": _normalize_sequence_chars("".join(info["sequence_parts"])),
            "keys": list(info["keys"]),
        }
    return out

def _load_fasta_chain_sequences(
    pdb_path: str,
    fasta_dir: str,
    observed_chains: Sequence[str],
) -> Dict[str, str]:
    fasta_path = _fasta_path_for_pdb(pdb_path, fasta_dir)
    if not os.path.exists(fasta_path):
        raise FileNotFoundError
    records = _parse_fasta_records(fasta_path)
    if not records:
        raise ValueError
    chain_sequences: Dict[str, str] = {}
    for header, sequence in tqdm(records, desc="fasta_chain_sequences", leave=False):
        chain_id = _extract_chain_id_from_fasta_header(header)
        if chain_id is None:
            continue
        chain_sequences[str(chain_id)] = sequence
    observed_unique: List[str] = []
    for chain in tqdm(observed_chains, desc="observed_unique_chains", leave=False):
        observed_unique.append(str(chain))
    missing: List[str] = []
    for chain in tqdm(observed_unique, desc="missing_fasta_chains", leave=False):
        if chain not in chain_sequences:
            missing.append(chain)
    if missing:
        if len(observed_unique) == 1 and len(records) == 1:
            chain_sequences[observed_unique[0]] = records[0][1]
        else:
            raise KeyError
    out: Dict[str, str] = {}
    for chain in tqdm(observed_unique, desc="selected_fasta_chains", leave=False):
        out[chain] = chain_sequences[chain]
    return out

def _normalize_esmc_device(device: str) -> str:
    resolved_device = str(device).strip().lower()
    if not resolved_device.startswith("cuda"):
        raise ValueError
    if not torch.cuda.is_available():
        raise RuntimeError
    return resolved_device

def _load_esmc_client(device: str):
    resolved_device = _normalize_esmc_device(device)
    cuda_device = torch.device(resolved_device)
    if cuda_device.index is not None:
        torch.cuda.set_device(cuda_device.index)
    client = ESMC.from_pretrained("esmc_600m").to(resolved_device)
    if hasattr(client, "eval"):
        client.eval()
    return client

def _release_esmc_client(client: object) -> None:
    if hasattr(client, "cpu"):
        client.cpu()
    del client
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def _normalize_esmc_embeddings(sequence: str, embeddings: object) -> torch.Tensor:
    emb = embeddings
    if not torch.is_tensor(emb):
        emb = torch.as_tensor(emb)
    emb = emb.detach().to(torch.float32).cpu()
    if emb.ndim == 3:
        if emb.size(0) != 1:
            raise ValueError
        emb = emb[0]
    if emb.ndim != 2:
        raise ValueError
    seq_len = len(sequence)
    if emb.size(0) == seq_len + 2:
        emb = emb[1:-1]
    elif emb.size(0) == seq_len + 1:
        emb = emb[1:]
    elif emb.size(0) != seq_len:
        raise ValueError
    return emb.contiguous()

def _embed_sequence_with_esmc(
    sequence: str,
    client: object,
) -> torch.Tensor:
    if not sequence:
        return torch.zeros((0, 0), dtype=torch.float32)
    protein = ESMProtein(sequence=sequence)
    with torch.inference_mode():
        protein_tensor = client.encode(protein)
        output = client.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True),
        )
    return _normalize_esmc_embeddings(sequence=sequence, embeddings=output.embeddings)

def _score_residue_pair(obs_aa: str, fasta_aa: str) -> int:
    if obs_aa == fasta_aa:
        return 3
    if obs_aa == "X" or fasta_aa == "X":
        return 1
    return -3

def _align_observed_to_fasta_sequence(
    observed: str,
    fasta: str,
) -> List[Optional[int]]:
    m = len(observed)
    n = len(fasta)
    if m == 0:
        return []
    if n == 0:
        return [None] * m
    gap_skip_fasta = -1
    gap_skip_observed = -6
    neg_inf = -10**9
    score = np.full((m + 1, n + 1), neg_inf, dtype=np.int32)
    trace = np.zeros((m + 1, n + 1), dtype=np.int8)
    score[0, :] = 0
    trace[0, 1:] = 1
    for i in tqdm(range(1, m + 1), desc="align_init", leave=False):
        score[i, 0] = score[i - 1, 0] + gap_skip_observed
        trace[i, 0] = 2
    for i in tqdm(range(1, m + 1), desc="align_dp", leave=False):
        obs_aa = observed[i - 1]
        for j in tqdm(range(1, n + 1), desc="align_dp_cols", leave=False, disable=True):
            fasta_aa = fasta[j - 1]
            diag = score[i - 1, j - 1] + _score_residue_pair(obs_aa, fasta_aa)
            left = score[i, j - 1] + gap_skip_fasta
            up = score[i - 1, j] + gap_skip_observed
            best = diag
            op = 0
            if left > best:
                best = left
                op = 1
            if up > best:
                best = up
                op = 2
            score[i, j] = best
            trace[i, j] = op
    end_j = int(np.argmax(score[m, :]))
    mapping: List[Optional[int]] = [None] * m
    i = m
    j = end_j
    with tqdm(total=m + max(end_j, 0), desc="align_traceback", leave=False) as pbar:
        while i > 0:
            if j == 0:
                mapping[i - 1] = None
                i -= 1
                pbar.update(1)
                continue
            op = int(trace[i, j])
            if op == 0:
                mapping[i - 1] = j - 1
                i -= 1
                j -= 1
                pbar.update(2)
            elif op == 1:
                j -= 1
                pbar.update(1)
            else:
                mapping[i - 1] = None
                i -= 1
                pbar.update(1)
    return mapping

def _validate_esmc_alignment(
    chain_id: str,
    observed: str,
    fasta: str,
    mapping: Sequence[Optional[int]],
) -> None:
    if len(mapping) != len(observed):
        raise ValueError(f"chain {chain_id}: mapping_len={len(mapping)} observed_len={len(observed)}")
    mapped = 0
    mismatched = 0
    for obs_idx, fasta_idx in enumerate(tqdm(mapping, desc=f"validate_alignment:{chain_id}", leave=False)):
        if fasta_idx is None:
            continue
        if fasta_idx < 0 or fasta_idx >= len(fasta):
            raise ValueError
        mapped += 1
        obs_aa = observed[obs_idx]
        fasta_aa = fasta[fasta_idx]
        if obs_aa != "X" and fasta_aa != "X" and obs_aa != fasta_aa:
            mismatched += 1
    unmapped = len(observed) - mapped
    if mapped == 0:
        raise ValueError(f"chain {chain_id}: mapped=0 observed_len={len(observed)} fasta_len={len(fasta)}")
    if unmapped > max(5, len(observed) // 10):
        raise ValueError(f"chain {chain_id}: unmapped={unmapped} observed_len={len(observed)} fasta_len={len(fasta)}")
    if mismatched > max(3, len(observed) // 20):
        raise ValueError(f"chain {chain_id}: mismatched={mismatched} observed_len={len(observed)} fasta_len={len(fasta)}")

def _drop_mismatched_alignment_positions(
    observed: str,
    fasta: str,
    mapping: Sequence[Optional[int]],
) -> List[Optional[int]]:
    cleaned: List[Optional[int]] = list(mapping)
    for obs_idx, fasta_idx in enumerate(tqdm(cleaned, desc="drop_mismatches", leave=False)):
        if fasta_idx is None:
            continue
        if fasta_idx < 0 or fasta_idx >= len(fasta):
            cleaned[obs_idx] = None
            continue
        obs_aa = observed[obs_idx]
        fasta_aa = fasta[fasta_idx]
        if obs_aa != "X" and fasta_aa != "X" and obs_aa != fasta_aa:
            cleaned[obs_idx] = None
    return cleaned

def _map_embeddings_from_fasta_alignment(
    chain_id: str,
    observed_info: Dict[str, object],
    fasta_sequence: str,
    embeddings: torch.Tensor,
) -> Dict[Tuple[str, int, str], np.ndarray]:
    observed_sequence = str(observed_info["sequence"])
    keys = list(observed_info["keys"])
    emb = embeddings.detach().to(torch.float32).cpu()
    if emb.ndim != 2 or emb.size(0) != len(fasta_sequence):
        raise ValueError
    observed = observed_sequence
    fasta = fasta_sequence
    start = fasta.find(observed) if observed else -1
    if observed and start >= 0 and fasta.find(observed, start + 1) < 0:
        mapping = list(range(start, start + len(observed)))
    else:
        mapping = _align_observed_to_fasta_sequence(observed, fasta)
        mapping = _drop_mismatched_alignment_positions(observed, fasta, mapping)
    _validate_esmc_alignment(chain_id, observed, fasta, mapping)
    residue_map: Dict[Tuple[str, int, str], np.ndarray] = {}
    emb_np = emb.numpy().astype(np.float32, copy=False)
    for idx, key in enumerate(tqdm(keys, desc=f"map_esmc:{chain_id}", leave=False)):
        fasta_idx = mapping[idx]
        if fasta_idx is None:
            continue
        residue_map[key] = emb_np[fasta_idx]
    return residue_map

def _compute_esmc_residue_map(
    pdb_path: str,
    residues: Sequence[Dict[str, object]],
    device: str,
    fasta_dir: str,
) -> Tuple[Dict[Tuple[str, int, str], np.ndarray], int]:
    observed_chains = _build_observed_chain_map(residues)
    fasta_chain_sequences = _load_fasta_chain_sequences(
        pdb_path=pdb_path,
        fasta_dir=fasta_dir,
        observed_chains=list(observed_chains.keys()),
    )
    residue_map: Dict[Tuple[str, int, str], np.ndarray] = {}
    esmc_dim = 0
    client = _load_esmc_client(device=device)
    try:
        for chain, info in tqdm(observed_chains.items(), desc="embed_chains", leave=False):
            fasta_sequence = str(fasta_chain_sequences[chain])
            emb = _embed_sequence_with_esmc(
                sequence=fasta_sequence,
                client=client,
            )
            if emb.size(0) != len(fasta_sequence):
                raise ValueError
            esmc_dim = int(emb.size(1))
            chain_residue_map = _map_embeddings_from_fasta_alignment(
                chain_id=chain,
                observed_info=info,
                fasta_sequence=fasta_sequence,
                embeddings=emb,
            )
            residue_map.update(chain_residue_map)
    finally:
        _release_esmc_client(client)
    return residue_map, int(esmc_dim)
