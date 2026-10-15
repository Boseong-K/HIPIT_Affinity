from typing import Dict, Sequence
import numpy as np
from scipy.spatial import cKDTree

def get_crosscontacts_dim() -> int:
    return 1

def _stack_coords(entries: Sequence[Dict[str, object]]) -> np.ndarray:
    if not entries:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack([np.asarray(entry["coord"], dtype=np.float64) for entry in entries], axis=0)

def cross_contact_features(
    target_entries: Sequence[Dict[str, object]],
    num_a: int,
    num_b: int,
) -> np.ndarray:
    n = len(target_entries)
    out = np.zeros((n, 1), dtype=np.float32)
    if n == 0 or num_a == 0 or num_b == 0:
        return out
    coords = _stack_coords(target_entries)
    a = coords[:num_a]
    b = coords[num_a:num_a + num_b]
    tree_a = cKDTree(a)
    tree_b = cKDTree(b)
    out[:num_a, 0] = tree_b.query_ball_point(a, r=6.0, return_length=True)
    out[num_a:num_a + num_b, 0] = tree_a.query_ball_point(b, r=6.0, return_length=True)
    return np.log1p(out).astype(np.float32)
